/*
⚡ TinyCoder AI

Copyright (c) 2026 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
 * CUDA GPU offload engine for TinyCoder.
 *
 * Two execution strategies (RTX 2080 Ti / Turing reference target):
 *
 *   1. Single-token generation (seqLen == 1) is DRAM-bandwidth-bound.
 *      We stream the COMPACT quantized weights (Q2_K 84 B/block, Q3_K
 *      110 B/block, Q4_K 144 B/block) straight from VRAM with a warp-per-row
 *      dequant-GEMV kernel — the same memory traffic llama.cpp's CUDA GEMV
 *      uses.  The FP16 twin (2 B/elem) would read ~5x the bytes and cap
 *      generation near ~270 tok/s, below llama.cpp's 295 tok/s on this part.
 *
 *   2. Batched prefill (seqLen > 1) is compute-bound.  Turing has no
 *      quantized tensor cores, so we materialize FP16 twins of all weights
 *      once at upload and run cublasGemmEx fp16 GEMMs (fp32 accumulate).
 *
 * Partial layer offload: upload() takes numGpuLayers (default = numLayers)
 * so only the first that many layers live on the GPU; the remainder can be
 * computed on the CPU by the caller (forward() copies the final hidden state
 * back instead of running the LM head when fewer than numLayers layers are
 * offloaded).
 *
 * Layer math (Qwen2 dense block), replicated vs tinycoder's CPU reference:
 *   hidden -> rmsNorm -> Q/K/V (+biases) -> RoPE(Q) + store KV(rotated K)
 *   -> warp flash attention -> attnO -> residual -> rmsNorm -> gate*up
 *   (SwiGLU silu fused) -> down -> residual; final rmsNorm -> LM head.
 */

#ifdef USE_CUDA

#include "GPUCompute.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace tinycoder::gpu {

    namespace {


        // GGML type codes (GGUFLoader.hpp).
        constexpr uint32_t kTypeQ2K = 10;
        constexpr uint32_t kTypeQ3K = 11;
        constexpr uint32_t kTypeQ4K = 12;
        constexpr uint32_t kTypeQ6K = 14;

        constexpr uint32_t kQ2K_BYTES = 84;
        constexpr uint32_t kQ3K_BYTES = 110;
        constexpr uint32_t kQ4K_BYTES = 144;
        constexpr uint32_t kQ6K_BYTES = 210;

        constexpr float LOG2E = 1.4426950408889634f;

        cudaStream_t g_stream = nullptr;
        cublasHandle_t g_cublas = nullptr;
        std::once_flag g_cudaInit;
        // TINYCODER_GPU_VERBOSE=1: per-stage stream syncs + print (debug aid).
        const bool g_verbose = [] {
            const char *e = std::getenv("TINYCODER_GPU_VERBOSE");
            return e != nullptr && e[0] != '\0' && e[0] != '0';
        }();

        bool ensureCuda() {
            std::call_once(g_cudaInit, []() {
                if (cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking) !=
                    cudaSuccess)
                    return;
                if (cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS) return;
                cublasSetStream(g_cublas, g_stream);
            });
            return g_stream && g_cublas;
        }

        // ------------------------------------------------------------------
        // Elementwise
        // ------------------------------------------------------------------

        __global__ void kAddResidual(float *dst, const float *src, uint32_t n) {
            uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) dst[i] += src[i];
        }

        // Broadcast a per-row bias across every row of a [rows][biasLen] tensor
        // (Qwen2 Q/K/V biases).  One block per row; the same bias[c] is added to
        // every row, so the bias is never indexed past its biasLen elements.
        __global__ void kAddBias(float *dst, const float *bias, uint32_t rows,
                                 uint32_t biasLen) {
            uint32_t s = blockIdx.x;
            if (s >= rows) return;
            float *row = dst + static_cast<size_t>(s) * biasLen;
            for (uint32_t c = threadIdx.x; c < biasLen; c += blockDim.x) {
                row[c] += bias[c];
            }
        }

        __global__ void kSiluMul(float *a, const float *b, uint32_t n) {
            uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                float x = a[i];
                a[i] = (x / (1.0f + __expf(-x))) * b[i];
            }
        }

        // Elementwise f32 -> f16: x[rows*n] fp32 -> out[rows*n] fp16.
        __global__ void kF32ToF16(const float *__restrict__ x,
                                  __half2 *__restrict__ out, uint32_t pairs) {
            uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= pairs) return;
            // each thread converts 2 consecutive floats into one __half2
            const uint32_t e0 = 2u * i;
            float a = x[e0];
            float b = x[e0 + 1];
            out[i] = __floats2half2_rn(a, b);
        }

        // One block (256 threads) per row of the [rows][n] matrix.
        // OUT-OF-PLACE: y = rmsNorm(x, w).  `x` is the residual stream and MUST
        // NOT be overwritten (the CPU reference norms into a separate buffer and
        // keeps the residual stream intact for the residual adds).
        __global__ void kRMSNormRow(const float *x, float *y, const float *w,
                                    uint32_t n, uint32_t rows, float eps) {
            uint32_t row = blockIdx.x;
            if (row >= rows) return;
            const float *rp = x + static_cast<size_t>(row) * n;
            float *op = y + static_cast<size_t>(row) * n;
            extern __shared__ float ssum[];
            float acc = 0.0f;
            for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
                float v = rp[i];
                acc = fmaf(v, v, acc);
            }
            ssum[threadIdx.x] = acc;
            __syncthreads();
            for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
                __syncthreads();
            }
            float rms = rsqrtf(ssum[0] / static_cast<float>(n) + eps);
            for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
                op[i] = rp[i] * rms * w[i];
            }
        }

        // ------------------------------------------------------------------
        // RoPE on Q, and K/V store with fused K rotation (matches
        // ModelPrimitives.cpp applyRoPE + storeKVWithRoPE element order).
        // ------------------------------------------------------------------

        // q[k] into q[k] (in place).  One block per token, 128 threads.
        __global__ void kRoPEQ(float *q, const float *cosT, const float *sinT,
                               uint32_t seqLen, uint32_t qHeads, uint32_t headDim,
                               uint32_t pos) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            uint32_t p = pos + s;
            uint32_t pairs = headDim / 2;
            const float *c = cosT + static_cast<size_t>(p) * pairs;
            const float *sn = sinT + static_cast<size_t>(p) * pairs;
            // every thread walks its own column slice across all heads
            for (uint32_t j = threadIdx.x; j < pairs; j += blockDim.x) {
                float cc = c[j], ss = sn[j];
                for (uint32_t h = 0; h < qHeads; ++h) {
                    float *head = q + (static_cast<size_t>(s) * qHeads + h) * headDim;
                    float x0 = head[2 * j], x1 = head[2 * j + 1];
                    head[2 * j] = fmaf(x0, cc, -x1 * ss);
                    head[2 * j + 1] = fmaf(x0, ss, x1 * cc);
                }
            }
        }

        // kSrc/vSrc [seqLen][kHeads*headDim] -> kDst/vDst cache at cachePos.
        // One block (256 threads) per token.
        // K rotation pairing MUST match the CPU reference (ModelPrimitives.cpp
        // storeKVWithRoPE): the rotation is applied to the (2j,2j+1) elements of
        // each head with the angle index j == cosT[pairs*pos + j].  A previous
        // version rotated the flat (t, t+pairs) elements with angle index
        // t%headDim, which for headDim=128 pairs the WRONG columns
        // ((d, d+64) instead of (2d, 2d+1)) and ranks all cached K differently
        // than the CPU's -- corrupting every q.k attention score (garbage /
        // EOG-dominated logits) once the KV cache has more than a couple tokens.
        __global__ void kStoreKVRope(const float *kSrc, const float *vSrc, float *kDst,
                                     float *vDst, uint32_t seqLen, uint32_t kHeads,
                                     uint32_t headDim, uint32_t cachePos, uint32_t pos,
                                     const float *cosT, const float *sinT) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            uint32_t p = pos + s;
            uint32_t pairs = headDim / 2;
            uint32_t kvSize = kHeads * headDim;
            const float *ks = kSrc + static_cast<size_t>(s) * kvSize;
            const float *vs = vSrc + static_cast<size_t>(s) * kvSize;
            float *kd = kDst + static_cast<size_t>(cachePos + s) * kvSize;
            float *vd = vDst + static_cast<size_t>(cachePos + s) * kvSize;
            const float *c = cosT + static_cast<size_t>(p) * pairs;
            const float *sn = sinT + static_cast<size_t>(p) * pairs;
            uint32_t t = threadIdx.x;
            // V copied verbatim for ALL columns.
            for (uint32_t e = t; e < kvSize; e += blockDim.x) {
                vd[e] = vs[e];
            }
            // K rotated per-head on the (2j,2j+1) pairs, angle index j.
            const uint32_t numPairs = kHeads * pairs;
            for (uint32_t e = t; e < numPairs; e += blockDim.x) {
                uint32_t h = e / pairs;
                uint32_t j = e - h * pairs;
                uint32_t base = h * headDim + 2u * j;
                float k0 = ks[base], k1 = ks[base + 1u];
                kd[base] = fmaf(k0, c[j], -k1 * sn[j]);
                kd[base + 1u] = fmaf(k0, sn[j], k1 * c[j]);
            }
        }

        // ------------------------------------------------------------------
        // Flash attention: one warp per (token, q-head).  Online softmax,
        // fp32 accumulate, causal mask via csEnd = cachePos + s.
        // ------------------------------------------------------------------

        __device__ __forceinline__ float kWarpReduceSum(float v) {
            for (uint32_t off = 16; off; off >>= 1) {
                v += __shfl_down_sync(0xffffffffu, v, off);
            }
            return __shfl_sync(0xffffffffu, v, 0);
        }

        // Flash attention: one warp per (token, q-head), compile-time headDim.
        // HD/32 register accumulators per lane (lane owns elements lane+i*32),
        // so nothing spills to local memory (vs. the old float o[128] indexed by
        // a runtime loop bound, which NVCC keeps in local memory).  Grid.y must
        // cover nHeads warps; idle warps exit immediately.
        template<uint32_t HD>
        __global__ void kWarpAttention(const float *__restrict__ q,
                                       const float *__restrict__ kCache,
                                       const float *__restrict__ vCache,
                                       float *__restrict__ out, uint32_t seqLen,
                                       uint32_t nHeads, uint32_t nKVHeads,
                                       uint32_t cachePos, float invSqrtHeadDim) {
            static_assert(HD == 128 || HD == 64 || HD == 32, "HD must be 32/64/128");
            constexpr uint32_t NV = HD / 32;
            uint32_t s = blockIdx.x;
            uint32_t warpId = blockIdx.y * blockDim.y + threadIdx.y;
            uint32_t lane = threadIdx.x;
            if (s >= seqLen || warpId >= nHeads) return;
            uint32_t qHead = warpId;
            uint32_t kvHead = qHead / (nHeads / nKVHeads);
            uint32_t csEnd = cachePos + s;

            const float *qPtr = q + (static_cast<size_t>(s) * nHeads + qHead) * HD;
            float acc[NV];
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
            float m = -1e30f, l = 0.0f;

            for (uint32_t cs = 0; cs <= csEnd; ++cs) {
                const float *kPtr =
                        kCache + (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
                const float *vPtr =
                        vCache + (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
                // score = dot(q, k) * invSqrt, warp-reduced
                float sc = 0.0f;
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) {
                    sc = fmaf(qPtr[lane + i * 32], kPtr[lane + i * 32], sc);
                }
                sc = kWarpReduceSum(sc) * invSqrtHeadDim;

                if (sc > m) {
                    float mNew = sc;
                    float alpha = exp2f((m - mNew) * LOG2E);
                    if (alpha != 1.0f) {
#pragma unroll
                        for (uint32_t i = 0; i < NV; ++i) acc[i] *= alpha;
                    }
                    l *= alpha;
                    m = mNew;
                }
                float pp = exp2f((sc - m) * LOG2E);
                l += pp;
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) {
                    acc[i] = fmaf(pp, vPtr[lane + i * 32], acc[i]);
                }
            }
            float invL = 1.0f / l;
            float *outPtr = out + (static_cast<size_t>(s) * nHeads + qHead) * HD;
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) {
                outPtr[lane + i * 32] = acc[i] * invL;
            }
        }

        // ------------------------------------------------------------------
        // Quantized GEMV: y[rows] = x[cols] @ W^T, warp-per-row.
        //
        // x is staged in shared memory once per block by all 256 threads.
        // Each warp owns one output row.  Each lane of the warp owns elements
        // [lane*8, lane*8+8) of each 256-element block and precomputes the
        // exact (scale, min, byte-base, shift) indices matching
        // GGMLDequantize::dequantize{Q2_K,Q3_K,Q4_K}Block so the dequant loop
        // is branch-free and bit-identical.
        // ------------------------------------------------------------------

        // cols must be a multiple of 32 (hiddenSize = 1536, intermediateSize =
        // 8960, headDim*heads = 1536 are; the launcher pads cols up to the next
        // 32 when they are not).  No shared memory is used: x is read directly
        // from global (coalesced, and tiny enough to stay L1/L2-resident).
        template<int TYPE>
        __global__ void kQGemv(const uint8_t *__restrict__ w,
                               const float *__restrict__ x, float *__restrict__ out,
                               uint32_t rows, uint32_t cols, uint32_t rowBytes,
                               uint32_t blocksPerRow) {
            // x is read DIRECTLY from global (coalesced, and tiny enough to
            // stay L1/L2-resident); no shared-memory staging/barrier needed.
            uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= rows) return;
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;

            float acc = 0.0f;
            const uint32_t lane = threadIdx.x;
            const uint32_t e0 = lane * 8u;

            if (TYPE == kTypeQ2K) {
                // Coalesced byte scheme (bit-exact with dequantizeQ2_KBlock):
                // read x DIRECTLY from global (skip the shared-memory staging,
                // which added latency and occupancy pressure).  Reads of x are
                // coalesced (lanes read consecutive floats within each 32-group)
                // and x is tiny (cols <= 1536 floats), so it stays L1/L2-hot
                // across all rows.
                const uint32_t sub = lane / 16u;
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 80);
                    __half dmin = *reinterpret_cast<const __half *>(rp + 82);
                    float df = __half2float(d), dmf = __half2float(dmin);
                    const uint8_t *sc = rp;
                    const uint8_t *q = rp + 16;
#pragma unroll
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                        for (uint32_t jj = 0; jj < 4; ++jj) {
                            const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                            float dl = df * static_cast<float>(scv & 0xF);
                            float ml = dmf * static_cast<float>(scv >> 4);
                            float qv = static_cast<float>(
                                    static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                            acc = fmaf(fmaf(dl, qv, -ml),
                                       x[b * 256u + half * 128u + jj * 32u + lane],
                                       acc);
                        }
                    }
                    rp += kQ2K_BYTES;
                }
            } else if (TYPE == kTypeQ3K) {
                // Same coalesced-byte scheme as Q2_K.  hm (32 bytes, block+0)
                // holds the high-bit mask: bit 1<<jj for the same (sub,l)
                // position => hm[lane] with the 1<<jj mask.  qv = (qb>>2jj&3) -
                // (hm[lane]&(1<<jj) ? 0 : 4).  Scale is the repacked int8
                // sc16[half*8+jj*2+sub].  No dmin (Q3_K has none).
                const uint32_t sub = lane / 16u;
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 108);
                    float df = __half2float(d);
                    const uint8_t *hm = rp;
                    const uint8_t *q = rp + 32;
                    const uint8_t *scales = rp + 96;
                    uint32_t aux[4];
                    std::memcpy(&aux[0], scales, 12);
                    uint32_t tmp = aux[2];
                    aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) |
                             (((tmp >> 4) & 0x03030303u) << 4);
                    aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) |
                             (((tmp >> 6) & 0x03030303u) << 4);
                    aux[0] = (aux[0] & 0x0f0f0f0fu) |
                             (((tmp >> 0) & 0x03030303u) << 4);
                    aux[1] = (aux[1] & 0x0f0f0f0fu) |
                             (((tmp >> 2) & 0x03030303u) << 4);
                    int8_t sc16[16];
                    std::memcpy(sc16, aux, 16);
                    const uint8_t hmb = hm[lane];
#pragma unroll
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                        for (uint32_t jj = 0; jj < 4; ++jj) {
                            float dl = df * static_cast<float>(
                                                    sc16[half * 8u + jj * 2u + sub] - 32);
                            // the hmask bit advances ACROSS halves too:
                            // half0 uses bits 0..3, half1 bits 4..7.
                            const uint32_t maskBit = 1u << (half * 4u + jj);
                            float qv = static_cast<float>(static_cast<int8_t>(
                                    ((qb >> (jj * 2u)) & 3) -
                                    ((hmb & maskBit) ? 0 : 4)));
                            acc = fmaf(dl * qv,
                                       x[b * 256u + half * 128u + jj * 32u + lane],
                                       acc);
                        }
                    }
                    rp += kQ3K_BYTES;
                }
            } else if (TYPE == kTypeQ4K) {
                // Coalesced byte scheme (bit-exact with dequantizeQ4_KBlock):
                // each 64-weight group g reads 32 q bytes.  Byte
                // qs[g*32 + lane] supplies out[g*64 + lane] from its LOW
                // nibble (sub-scale is+0) and out[g*64 + 32 + lane] from its
                // HIGH nibble (sub-scale is+1).  32 lanes x 2 nibbles = 64
                // weights per group, 4 groups = 256 per block.
                auto getScaleMin = [](int j, const uint8_t *q, uint8_t *d_out,
                                      uint8_t *m_out) {
                    if (j < 4) {
                        *d_out = q[j] & 63;
                        *m_out = q[j + 4] & 63;
                    } else {
                        *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                        *m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
                    }
                };
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    __half dmin = *reinterpret_cast<const __half *>(rp + 2);
                    float df = __half2float(d), dmf = __half2float(dmin);
                    const uint8_t *scales = rp + 4;
                    const uint8_t *qs = rp + 16;
#pragma unroll
                    for (uint32_t g = 0; g < 4; ++g) {
                        const uint8_t qb = qs[g * 32u + lane];
                        uint8_t sc0, mm0, sc1, mm1;
                        getScaleMin(static_cast<int>(g * 2 + 0), scales, &sc0,
                                    &mm0);
                        getScaleMin(static_cast<int>(g * 2 + 1), scales, &sc1,
                                    &mm1);
                        float dl0 = df * static_cast<float>(sc0);
                        float ml0 = dmf * static_cast<float>(mm0);
                        float dl1 = df * static_cast<float>(sc1);
                        float ml1 = dmf * static_cast<float>(mm1);
                        float qv0 = static_cast<float>(qb & 0xF);
                        float qv1 = static_cast<float>(qb >> 4);
                        acc = fmaf(fmaf(dl0, qv0, -ml0), x[b * 256u + g * 64u + lane],
                                   acc);
                        acc = fmaf(fmaf(dl1, qv1, -ml1),
                                   x[b * 256u + g * 64u + 32u + lane], acc);
                    }
                    rp += kQ4K_BYTES;
                }
            } else if (TYPE == kTypeQ6K) {
                // Q6_K block layout (210 B): ql(128) + qh(64) + sc(16) + d(2).
                // Each 128-weight half: qh[l] holds 2-bit fields for weights at
                // l, l+32, l+64, l+96; q1/q3 use ql[l] (low/high nibble), q2/q4
                // use ql[32+l]; 8 int8 scales, one per 16-weight group:
                //   sc[half*8 + sub*2 + (l/16)], value = d*scale*(q-32).
                const uint32_t half = e0 / 128u;
                const uint32_t sub = (e0 % 128u) / 32u;
                const uint32_t l = e0 % 32u;
                const uint32_t g16 = l / 16u;
                const uint32_t qlBase = half * 64u + ((sub & 1u) ? 32u : 0u);
                const uint32_t qhBase = half * 32u;
                const uint32_t shift = sub * 2u;            // 0,2,4,6
                const uint32_t hiNib = (sub & 2u) ? 4u : 0u;// q3/q4 use high nibble
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    const uint8_t *rb = rp;
                    const uint8_t *ql = rb;
                    const uint8_t *qh = rb + 128;
                    const int8_t *sc = reinterpret_cast<const int8_t *>(rb + 192);
                    __half d = *reinterpret_cast<const __half *>(rb + 208);
                    float dl = __half2float(d) *
                               static_cast<float>(sc[half * 8u + sub * 2u + g16]);
                    const float *xs = x + b * 256u + e0;
#pragma unroll
                    for (uint32_t k = 0; k < 8; ++k) {
                        uint32_t qb = (hiNib) ? (ql[qlBase + l + k] >> 4)
                                              : (ql[qlBase + l + k] & 0xF);
                        qb |= ((qh[qhBase + l + k] >> shift) & 3u) << 4u;
                        float qv = static_cast<float>(static_cast<int>(qb) - 32);
                        acc = fmaf(dl * qv, xs[k], acc);
                    }
                    rp += kQ6K_BYTES;
                }
            } else {
                acc = 0.0f;
            }
// Warp reduction: each lane holds the partial dot product over its
// disjoint 8-column-per-block subset; the full row result is the
// sum across all 32 lanes.  (5 shuffles + 5 adds per row.)
#pragma unroll
            for (uint32_t off = 16; off > 0; off >>= 1) {
                acc += __shfl_xor_sync(0xffffffffu, acc, off);
            }
            if (lane == 0) out[row] = acc;
        }

        void launchQGemv(int type, const void *wq, const float *x, float *out,
                         uint32_t rows, uint32_t cols, uint32_t rowBytes,
                         uint32_t blocksPerRow) {
            // x is read directly from global inside the kernel (coalesced, and
            // tiny enough to stay L1/L2-resident), so no shared memory is used.
            // cols must be a multiple of 32.  For this model hiddenSize=1536,
            // intermediateSize=8960, qLen=1536 and kvLen=256 are all multiples;
            // the Model layer pads any non-multiple up when building the desc.
            // 8 warps per block (256 threads) => 8 rows per block.
            dim3 block(32, 8);
            uint32_t nBlocks = (rows + 7) / 8;
            if (type == kTypeQ2K) {
                kQGemv<kTypeQ2K><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeQ3K) {
                kQGemv<kTypeQ3K><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeQ4K) {
                kQGemv<kTypeQ4K><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeQ6K) {
                kQGemv<kTypeQ6K><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else {
                cudaMemsetAsync(out, 0, sizeof(float) * rows, g_stream);
            }
        }

        // Token embedding dequant (quantized Q2_K rows, one warp per token).
        __global__ void kEmbedDequant(const uint8_t *__restrict__ embed,
                                      const int32_t *__restrict__ tokens,
                                      float *__restrict__ hidden, uint32_t seqLen,
                                      uint32_t hiddenSize, uint32_t blocksPerRow,
                                      uint32_t rowBytes, uint32_t vocabSize) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            int32_t t = tokens[s];
            if (t < 0 || static_cast<uint32_t>(t) >= vocabSize) return;
            const uint8_t *row = embed + static_cast<uint64_t>(t) * rowBytes;
            float *h = hidden + static_cast<size_t>(s) * hiddenSize;
            const uint32_t lane = threadIdx.x;
            const uint32_t e0 = lane * 8u;
            const uint32_t half = e0 / 128u, jj = (e0 % 128u) / 32u,
                           sub = (e0 % 32u) / 16u, l16 = e0 % 16u;
            const uint32_t is = half * 8u + jj * 2u + sub;
            // q indexing matches dequantizeQ2_KBlock: same 32 q bytes per half,
            // reused for all four shifts.
            const uint32_t qbase = half * 32u + sub * 16u + l16;
            const uint32_t shift = jj * 2u;
            const uint32_t typeBytes = rowBytes / blocksPerRow;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                __half d = *reinterpret_cast<const __half *>(row + 80);
                __half dmin = *reinterpret_cast<const __half *>(row + 82);
                float df = __half2float(d), dmf = __half2float(dmin);
                float dl = df * static_cast<float>(row[is] & 0xF);
                float ml = dmf * static_cast<float>(row[is] >> 4);
#pragma unroll
                for (uint32_t k = 0; k < 8; ++k) {
                    float qv = static_cast<float>(static_cast<int8_t>(
                            ((row[16 + qbase + k]) >> shift) & 3));
                    h[b * 256u + e0 + k] = fmaf(dl, qv, -ml);
                }
                row += typeBytes;
            }
        }

    }// namespace

    // ----------------------------------------------------------------------
    // Public API
    // ----------------------------------------------------------------------

    // GPU is enabled by default in any build compiled with USE_CUDA
    // (scripts/build.sh and npm run build:native produce such builds).
    // $TINYCODER_GPU=0 disables the offload engine (CPU-only), which is the
    // opt-out for systems without a usable CUDA device.
    bool gpuEnabled() {
        const char *e = std::getenv("TINYCODER_GPU");
        if (e == nullptr || e[0] == '\0') return true;
        return e[0] != '0';
    }

    bool GPUModel::ensureScratch(uint32_t seqLen, std::string &errMsg) {
        if (scratchAlloc_ && scratch_.seqCap >= seqLen) return true;
        destroyScratch();
        // Allocate for the requested capacity (grow-on-demand; never shrink).
        uint32_t cap = std::max(seqLen, 1u);
        uint32_t H = geom_.hiddenSize, I = geom_.intermediateSize;
        uint32_t qLen = geom_.numAttentionHeads * geom_.headDim;
        uint32_t kvLen = geom_.numKVHeads * geom_.headDim;
        uint64_t v = static_cast<uint64_t>(cap) * geom_.vocabSize * sizeof(float);
        auto allocT = [&](auto *&p, uint64_t n, const char *what) -> bool {
            p = nullptr;
            cudaError_t e = cudaMalloc(&p, n);
            if (e != cudaSuccess) {
                p = nullptr;
                errMsg = std::string("cudaMalloc(scratch ") + what + "): " +
                         cudaGetErrorString(e);
                return false;
            }
            return true;
        };
        if (!allocT(scratch_.tokens, static_cast<uint64_t>(cap) * sizeof(int32_t),
                    "tokens") ||
            !allocT(scratch_.hidden, static_cast<uint64_t>(cap) * H * sizeof(float),
                    "hidden") ||
            !allocT(scratch_.norm, static_cast<uint64_t>(cap) * H * sizeof(float),
                    "norm") ||
            !allocT(scratch_.hiddenF16,
                    static_cast<uint64_t>(cap) * H * sizeof(uint16_t), "hiddenF16") ||
            !allocT(scratch_.q, static_cast<uint64_t>(cap) * qLen * sizeof(float),
                    "q") ||
            !allocT(scratch_.k, static_cast<uint64_t>(cap) * kvLen * sizeof(float),
                    "k") ||
            !allocT(scratch_.v, static_cast<uint64_t>(cap) * kvLen * sizeof(float),
                    "v") ||
            !allocT(scratch_.attnOut,
                    static_cast<uint64_t>(cap) * qLen * sizeof(float), "attnOut") ||
            !allocT(scratch_.attnOutF16,
                    static_cast<uint64_t>(cap) * qLen * sizeof(uint16_t),
                    "attnOutF16") ||
            !allocT(scratch_.attnProj,
                    static_cast<uint64_t>(cap) * H * sizeof(float), "attnProj") ||
            !allocT(scratch_.gate, static_cast<uint64_t>(cap) * I * sizeof(float),
                    "gate") ||
            !allocT(scratch_.gateF16,
                    static_cast<uint64_t>(cap) * I * sizeof(uint16_t), "gateF16") ||
            !allocT(scratch_.up, static_cast<uint64_t>(cap) * I * sizeof(float),
                    "up") ||
            !allocT(scratch_.ffnOut,
                    static_cast<uint64_t>(cap) * H * sizeof(float), "ffnOut") ||
            !allocT(scratch_.logits, v, "logits")) {
            destroyScratch();
            return false;
        }
        scratch_.seqCap = cap;
        scratchAlloc_ = true;
        return true;
    }

    void GPUModel::destroyScratch() {
        if (!scratchAlloc_) return;
        cudaFree(scratch_.tokens);
        cudaFree(scratch_.hidden);
        cudaFree(scratch_.norm);
        cudaFree(scratch_.hiddenF16);
        scratch_.hiddenF16 = nullptr;
        cudaFree(scratch_.q);
        cudaFree(scratch_.k);
        cudaFree(scratch_.v);
        cudaFree(scratch_.attnOut);
        cudaFree(scratch_.attnOutF16);
        cudaFree(scratch_.attnProj);
        cudaFree(scratch_.gate);
        cudaFree(scratch_.gateF16);
        cudaFree(scratch_.up);
        cudaFree(scratch_.ffnOut);
        cudaFree(scratch_.logits);
        // Value-initialize (zero all pointers) instead of std::memset: the
        // struct has non-trivial members, and std::memset on it trips
        // -Wclass-memaccess.
        scratch_ = DeviceScratch();
        scratchAlloc_ = false;
    }

    bool GPUModel::upload(const std::vector<DeviceLayer> &layers,
                          const ModelGeometry &geom, const void *embedQ,
                          uint32_t embedType, uint32_t embedRowBytes,
                          const void *lmHeadQ, uint32_t lmHeadType,
                          uint32_t lmHeadRowBytes, std::string &errMsg) {
        if (allocated_) destroy();
        if (!ensureCuda()) {
            errMsg = "CUDA runtime initialization failed";
            return false;
        }
        // numGpuLayers defaults to numLayers; caller may pass a smaller value in
        // geom (via the numGpuLayers field below) for partial offload.
        uint32_t nGpu = geom.numGpuLayers;
        if (nGpu == 0) nGpu = geom.numLayers;
        if (nGpu > geom.numLayers || nGpu > layers.size()) nGpu = geom.numLayers;

        geom_ = geom;
        geom_.numGpuLayers = nGpu;

        auto alloc = [&](void **p, size_t bytes, const char *what) -> bool {
            cudaError_t e = cudaMalloc(p, bytes);
            if (e != cudaSuccess) {
                errMsg = std::string("cudaMalloc(") + what + "): " +
                         cudaGetErrorString(e);
                return false;
            }
            return true;
        };
        auto uploadBytes = [&](void *dst, const void *src, size_t bytes,
                               const char *what) -> bool {
            cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
            if (e != cudaSuccess) {
                errMsg = std::string("cudaMemcpy(upload ") + what + "): " +
                         cudaGetErrorString(e);
                return false;
            }
            return true;
        };

        // ---- Device LayerWeights (first nGpu layers only) ----
        layers_ = new DeviceLayer[nGpu];
        for (uint32_t L = 0; L < nGpu; ++L) {
            const DeviceLayer &hl = layers[L];
            DeviceLayer &dl = layers_[L];
            auto uploadMat = [&](const DeviceMatrix &src, DeviceMatrix &dst,
                                 const char *name) -> bool {
                if (src.q == nullptr) return true;
                dst = src;
                if (!alloc(&dst.q, static_cast<uint64_t>(src.rows) * src.rowBytes, name))
                    return false;
                if (!uploadBytes(dst.q, src.q,
                                 static_cast<uint64_t>(src.rows) * src.rowBytes, name))
                    return false;
                // FP16 twin for prefill GEMMs
                if (src.f16 != nullptr) {
                    if (!alloc(&dst.f16, static_cast<uint64_t>(src.rows) * src.cols * sizeof(__half),
                               name))
                        return false;
                    if (!uploadBytes(dst.f16, src.f16,
                                     static_cast<uint64_t>(src.rows) * src.cols *
                                             sizeof(__half),
                                     name))
                        return false;
                }
                return true;
            };
            auto uploadF32 = [&](float *&dst, const float *src, uint32_t n,
                                 const char *name) -> bool {
                if (src == nullptr || n == 0) return true;
                if (!alloc(reinterpret_cast<void **>(&dst), sizeof(float) * n, name))
                    return false;
                if (!uploadBytes(dst, src, sizeof(float) * n, name)) return false;
                return true;
            };
            if (!uploadMat(hl.attnQ, dl.attnQ, "attnQ") ||
                !uploadMat(hl.attnK, dl.attnK, "attnK") ||
                !uploadMat(hl.attnV, dl.attnV, "attnV") ||
                !uploadMat(hl.attnO, dl.attnO, "attnO") ||
                !uploadMat(hl.ffnGate, dl.ffnGate, "ffnGate") ||
                !uploadMat(hl.ffnUp, dl.ffnUp, "ffnUp") ||
                !uploadMat(hl.ffnDown, dl.ffnDown, "ffnDown")) {
                return false;
            }
            if (L == 0) {
                if (const char *te = std::getenv("TINYCODER_GPU_VERBOSE")) {
                    if (te[0] != '\0' && te[0] != '0') {
                        std::fprintf(stderr,
                                     "[gpu] types L0: attnQ=%u attnK=%u attnV=%u "
                                     "attnO=%u gate=%u up=%u down=%u\n",
                                     hl.attnQ.type, hl.attnK.type, hl.attnV.type,
                                     hl.attnO.type, hl.ffnGate.type, hl.ffnUp.type,
                                     hl.ffnDown.type);
                    }
                }
            }
            uint32_t qLen = geom_.numAttentionHeads * geom_.headDim;
            uint32_t kvLen = geom_.numKVHeads * geom_.headDim;
            // For Qwen2, biases are [qLen]/[kvLen]; for Gemma/Qwen35MoE they are
            // absent (qwen2Bias == 0 routes them to null).
            uint32_t qb = geom_.qwen2Bias ? qLen : 0;
            uint32_t kb = geom_.qwen2Bias ? kvLen : 0;
            if (!uploadF32(dl.attnQBias, hl.attnQBias, qb, "attnQBias") ||
                !uploadF32(dl.attnKBias, hl.attnKBias, kb, "attnKBias") ||
                !uploadF32(dl.attnVBias, hl.attnVBias, kb, "attnVBias") ||
                !uploadF32(dl.rmsNormAttn, hl.rmsNormAttn, geom_.hiddenSize, "rmsAttn") ||
                !uploadF32(dl.rmsNormFFN, hl.rmsNormFFN, geom_.hiddenSize, "rmsFFN")) {
                return false;
            }
        }

        // ---- Final norm ----
        if (geom.finalNorm != nullptr) {
            if (!alloc(reinterpret_cast<void **>(&finalNorm_),
                       geom.hiddenSize * sizeof(float), "finalNorm"))
                return false;
            if (!uploadBytes(finalNorm_, geom.finalNorm,
                             geom.hiddenSize * sizeof(float), "finalNorm"))
                return false;
        }

        // ---- Embedding (quantized rows) ----
        if (embedQ != nullptr) {
            uint64_t embedBytes = static_cast<uint64_t>(geom.vocabSize) * embedRowBytes;
            if (!alloc(reinterpret_cast<void **>(&embedQ_), embedBytes, "embedQ"))
                return false;
            if (!uploadBytes(embedQ_, embedQ, embedBytes, "embedQ")) return false;
            embedType_ = embedType;
            embedRowBytes_ = embedRowBytes;
        }

        // ---- Separate LM head (quantized rows; nullable for tied heads) ----
        if (lmHeadQ != nullptr) {
            uint64_t lmBytes = static_cast<uint64_t>(geom.vocabSize) * lmHeadRowBytes;
            if (!alloc(reinterpret_cast<void **>(&lmHeadQ_), lmBytes, "lmHeadQ"))
                return false;
            if (!uploadBytes(lmHeadQ_, lmHeadQ, lmBytes, "lmHeadQ")) return false;
            lmHeadType_ = lmHeadType;
            lmHeadRowBytes_ = lmHeadRowBytes;
        }

        // ---- RoPE tables [maxSeqLen][headDim/2] ----
        {
            uint32_t pairs = geom_.headDim / 2;
            std::vector<float> cosT(static_cast<size_t>(geom_.maxSeqLen) * pairs);
            std::vector<float> sinT(static_cast<size_t>(geom_.maxSeqLen) * pairs);
            std::vector<float> freq(pairs);
            for (uint32_t d2 = 0; d2 < pairs; ++d2) {
                freq[d2] = 1.0f / std::pow(geom_.ropeTheta,
                                           static_cast<float>(2 * d2) / geom_.headDim);
            }
            for (uint32_t p = 0; p < geom_.maxSeqLen; ++p) {
                for (uint32_t d2 = 0; d2 < pairs; ++d2) {
                    float ang = static_cast<float>(p) * freq[d2];
                    cosT[static_cast<size_t>(p) * pairs + d2] = std::cos(ang);
                    sinT[static_cast<size_t>(p) * pairs + d2] = std::sin(ang);
                }
            }
            if (!alloc(reinterpret_cast<void **>(&ropeCos_),
                       cosT.size() * sizeof(float), "ropeCos") ||
                !alloc(reinterpret_cast<void **>(&ropeSin_),
                       sinT.size() * sizeof(float), "ropeSin"))
                return false;
            if (!uploadBytes(ropeCos_, cosT.data(), cosT.size() * sizeof(float),
                             "ropeCos") ||
                !uploadBytes(ropeSin_, sinT.data(), sinT.size() * sizeof(float),
                             "ropeSin"))
                return false;
        }

        // ---- KV cache (device, only nGpu layers) ----
        uint64_t kvBytes = static_cast<uint64_t>(geom_.maxSeqLen) *
                           geom_.numKVHeads * geom_.headDim * sizeof(float);
        if (!alloc(reinterpret_cast<void **>(&kvK_),
                   static_cast<size_t>(nGpu) * kvBytes, "kvK") ||
            !alloc(reinterpret_cast<void **>(&kvV_),
                   static_cast<size_t>(nGpu) * kvBytes, "kvV"))
            return false;
        cudaMemsetAsync(kvK_, 0, static_cast<size_t>(nGpu) * kvBytes, g_stream);
        cudaMemsetAsync(kvV_, 0, static_cast<size_t>(nGpu) * kvBytes, g_stream);

        kvPos_ = 0;
        allocated_ = true;
        cudaStreamSynchronize(g_stream);
        return true;
    }

    bool GPUModel::forward(const std::vector<int32_t> &tokens, bool computeAllLogits,
                           float *logitsOut, std::string &errMsg) {
        if (!allocated_) {
            errMsg = "GPU model not uploaded";
            return false;
        }
        if (tokens.empty()) return true;
        auto t0 = std::chrono::steady_clock::now();
#define TIMING_TAG(what)                                               \
    do {                                                               \
        if (g_verbose) {                                               \
            double ms = std::chrono::duration<double, std::milli>(     \
                                std::chrono::steady_clock::now() - t0) \
                                .count();                              \
            std::fprintf(stderr, "[gpu fwd %6.2f ms] %s\n", ms, what); \
        }                                                              \
    } while (0)

        uint32_t seqLen = static_cast<uint32_t>(tokens.size());
        uint32_t nGpu = geom_.numGpuLayers;
        if (nGpu == 0) nGpu = geom_.numLayers;
        uint32_t H = geom_.hiddenSize, I = geom_.intermediateSize;
        uint32_t nHeads = geom_.numAttentionHeads, nKV = geom_.numKVHeads;
        uint32_t hd = geom_.headDim;
        uint32_t qLen = nHeads * hd, kvLen = nKV * hd;
        bool fullOffload = (nGpu == geom_.numLayers);

        if (!ensureScratch(seqLen, errMsg)) return false;
        DeviceScratch &s = scratch_;
        cudaMemcpyAsync(s.tokens, tokens.data(), seqLen * sizeof(int32_t),
                        cudaMemcpyHostToDevice, g_stream);
        cudaStreamSynchronize(g_stream);

        uint32_t pos = static_cast<uint32_t>(kvPos_);

        // ---- Embedding ----
        {
            uint32_t blocksPerRow = (H + 255) / 256;
            kEmbedDequant<<<seqLen, 32, 0, g_stream>>>(
                    reinterpret_cast<const uint8_t *>(embedQ_), s.tokens, s.hidden,
                    seqLen, H, blocksPerRow, embedRowBytes_, geom_.vocabSize);
        }
        TIMING_TAG("embed queued");

        // ---- Layer loop (only the offloaded prefix) ----
        cudaEvent_t evL0 = nullptr, evL1 = nullptr;
        std::vector<cudaEvent_t> evQKV, evAttn, evFFN;
        // per-matrix decode breakdown: attnO, gate(+up, up to silu),
        // down.  Only meaningful for seqLen == 1 (decode).
        std::vector<cudaEvent_t> evMtxAttnO, evMtxGateUp, evMtxDown;
        const bool traceStages = g_verbose && seqLen == 1;
        if (g_verbose) {
            cudaEventCreate(&evL0);
            cudaEventCreate(&evL1);
            cudaEventRecord(evL0, g_stream);
        }
        if (traceStages) {
            evQKV.resize(nGpu);
            evAttn.resize(nGpu);
            evFFN.resize(nGpu);
            evMtxAttnO.resize(nGpu);
            evMtxGateUp.resize(nGpu);
            evMtxDown.resize(nGpu);
            for (uint32_t i = 0; i < nGpu; ++i) {
                cudaEventCreate(&evQKV[i]);
                cudaEventCreate(&evAttn[i]);
                cudaEventCreate(&evFFN[i]);
                cudaEventCreate(&evMtxAttnO[i]);
                cudaEventCreate(&evMtxGateUp[i]);
                cudaEventCreate(&evMtxDown[i]);
            }
        }
        for (uint32_t L = 0; L < nGpu; ++L) {
            if (L == 0) TIMING_TAG("layer loop start");
            const DeviceLayer &w = layers_[L];
            float *kvK = kvK_ + static_cast<size_t>(L) * geom_.maxSeqLen * kvLen;
            float *kvV = kvV_ + static_cast<size_t>(L) * geom_.maxSeqLen * kvLen;


            // Attention RMSNorm -> s.norm (OUT-OF-PLACE: s.hidden is the
            // residual stream and must survive until the residual add below).
            // Grid = seqLen: the kernel uses ONE BLOCK PER ROW (row ==
            // blockIdx.x), so `(seqLen + 255) / 256` would only normalize the
            // first row for seqLen <= 256.
            kRMSNormRow<<<seqLen, 256, 256 * sizeof(float), g_stream>>>(
                    s.hidden, s.norm, w.rmsNormAttn, H, seqLen, 1e-6f);

            // Q/K/V projections read the NORMED hidden (s.norm), not the
            // residual stream (s.hidden).
            if (seqLen == 1) {
                launchQGemv(w.attnQ.type, w.attnQ.q, s.norm, s.q, w.attnQ.rows,
                            w.attnQ.cols, w.attnQ.rowBytes, w.attnQ.blocksPerRow);
                launchQGemv(w.attnK.type, w.attnK.q, s.norm, s.k, w.attnK.rows,
                            w.attnK.cols, w.attnK.rowBytes, w.attnK.blocksPerRow);
                launchQGemv(w.attnV.type, w.attnV.q, s.norm, s.v, w.attnV.rows,
                            w.attnV.cols, w.attnV.rowBytes, w.attnV.blocksPerRow);
            } else {
                // cuBLAS fp16 GEMM (batch): cublasGemmEx tensor-op requires BOTH
                // A and B in fp16 (fp32 x fp16 returns CUBLAS_STATUS_NOT_SUPPORTED),
                // so keep an fp16 twin of the RMSNorm'd hidden here.
                kF32ToF16<<<(seqLen * H + 511) / 512, 256, 0, g_stream>>>(
                        s.norm, reinterpret_cast<__half2 *>(s.hiddenF16),
                        (seqLen * H) / 2);
                // out[seqLen][rows] = x[seqLen][cols] @ W[rows][cols]^T
                // CUBLAS is column-major: put W in the A slot (op=T gives
                // W^T) and x in the B slot (op=N), with C sized [rows][seqLen]
                // in column-major order == the row-major [seqLen][rows] output
                // buffer.  (The old transposed parameterization -- m=seqLen,
                // n=rows, ldc=seqLen -- wrote element (i,j) at flat[i+j*seqLen]
                // instead of flat[i*rows+j], scrambling every matrix output
                // whenever seqLen > 1.  Decode (seqLen==1) was unaffected,
                // which is why single-token generation looked sane while the
                // 40-token prefill produced NaN.)
                auto gemmXWt = [&](float *out, const void *x, const void *wF16,
                                   uint32_t rows, uint32_t cols) -> bool {
                    // computeType CUDA_R_32F requires alpha/beta as float*.
                    // (Passing __half* made cuBLAS read the half 1.0/0.0 bits
                    // plus the adjacent half as one float -- effectively a
                    // tiny alpha that underflowed the prefill GEMM results.)
                    float alpha = 1.0f, beta = 0.0f;
                    cublasStatus_t st = cublasGemmEx(
                            g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            static_cast<int>(rows), static_cast<int>(seqLen),
                            static_cast<int>(cols), &alpha, wF16, CUDA_R_16F,
                            static_cast<int>(cols), x, CUDA_R_16F,
                            static_cast<int>(cols), &beta, out, CUDA_R_32F,
                            static_cast<int>(rows), CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    return st == CUBLAS_STATUS_SUCCESS;
                };
                if (!gemmXWt(s.q, s.hiddenF16, w.attnQ.f16, w.attnQ.rows, w.attnQ.cols) ||
                    !gemmXWt(s.k, s.hiddenF16, w.attnK.f16, w.attnK.rows, w.attnK.cols) ||
                    !gemmXWt(s.v, s.hiddenF16, w.attnV.f16, w.attnV.rows, w.attnV.cols)) {
                    errMsg = "cublasGemmEx Q/K/V failed";
                    return false;
                }
                if (g_verbose) {
                    cudaError_t e = cudaStreamSynchronize(g_stream);
                    std::fprintf(stderr, "[gpu loop] L=%u QKV: %s\n", L,
                                 e == cudaSuccess ? "ok"
                                                  : cudaGetErrorString(e));
                }
            }

            // Biases (Qwen2): bias is ONE row (qLen/kvLen floats) broadcast
            // across every token — kAddBias, never kAddResidual (which would
            // index the bias array up to seqLen*biasLen and overread it).
            if (w.attnQBias) {
                kAddBias<<<seqLen, 256, 0, g_stream>>>(s.q, w.attnQBias, seqLen,
                                                       qLen);
            }
            if (w.attnKBias) {
                kAddBias<<<seqLen, 256, 0, g_stream>>>(s.k, w.attnKBias, seqLen,
                                                       kvLen);
            }
            if (w.attnVBias) {
                kAddBias<<<seqLen, 256, 0, g_stream>>>(s.v, w.attnVBias, seqLen,
                                                       kvLen);
            }
            if (traceStages) cudaEventRecord(evQKV[L], g_stream);

            // RoPE(Q) + store K/V (K rotation fused)
            kRoPEQ<<<seqLen, 128, 0, g_stream>>>(s.q, ropeCos_, ropeSin_, seqLen,
                                                 nHeads, hd, pos);
            kStoreKVRope<<<seqLen, 256, 0, g_stream>>>(
                    s.k, s.v, kvK, kvV, seqLen, nKV, hd, pos, pos, ropeCos_, ropeSin_);

            // Flash attention (warp per token-head).  Compile-time headDim
            // so the accumulators live in registers; grid.y covers exactly
            // nHeads warps (no redundant warps).
            float invSqrt = 1.0f / std::sqrt(static_cast<float>(hd));
            dim3 blk(32, 4);
            if (hd == 128) {
                kWarpAttention<128><<<dim3(seqLen, (nHeads + 3) / 4), blk, 0,
                                      g_stream>>>(s.q, kvK, kvV, s.attnOut, seqLen,
                                                  nHeads, nKV, pos, invSqrt);
            } else if (hd == 64) {
                kWarpAttention<64><<<dim3(seqLen, (nHeads + 3) / 4), blk, 0,
                                     g_stream>>>(s.q, kvK, kvV, s.attnOut, seqLen,
                                                 nHeads, nKV, pos, invSqrt);
            } else {
                kWarpAttention<32><<<dim3(seqLen, (nHeads + 3) / 4), blk, 0,
                                     g_stream>>>(s.q, kvK, kvV, s.attnOut, seqLen,
                                                 nHeads, nKV, pos, invSqrt);
            }
            if (g_verbose) {
                cudaError_t e = cudaStreamSynchronize(g_stream);
                std::fprintf(stderr, "[gpu loop] L=%u kv+attn: %s\n", L,
                             e == cudaSuccess ? "ok"
                                              : cudaGetErrorString(e));
            }
            if (traceStages) cudaEventRecord(evAttn[L], g_stream);

            // attnO projection + residual
            if (traceStages) cudaEventRecord(evMtxAttnO[L], g_stream);
            if (seqLen == 1) {
                launchQGemv(w.attnO.type, w.attnO.q, s.attnOut, s.attnProj,
                            w.attnO.rows, w.attnO.cols, w.attnO.rowBytes,
                            w.attnO.blocksPerRow);
            } else {
                // attnOut is the fp32 attention result; make an fp16 twin for
                // the fp16 tensor-core GEMM.
                kF32ToF16<<<(seqLen * qLen + 511) / 512, 256, 0, g_stream>>>(
                        s.attnOut, reinterpret_cast<__half2 *>(s.attnOutF16),
                        (seqLen * qLen) / 2);
                // CUDA_R_32F compute type -> alpha/beta must be float*.
                float alpha = 1.0f, beta = 0.0f;
                // attnProj[seqLen][rows] = attnOut[seqLen][cols] @ W[rows][cols]^T
                // (see the Q/K/V gemmXWt comment: CUBLAS column-major requires
                // W in the A slot, x in B, ldc=rows).
                cublasStatus_t st = cublasGemmEx(
                        g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        static_cast<int>(w.attnO.rows), static_cast<int>(seqLen),
                        static_cast<int>(w.attnO.cols), &alpha, w.attnO.f16, CUDA_R_16F,
                        static_cast<int>(w.attnO.cols), s.attnOutF16, CUDA_R_16F,
                        static_cast<int>(w.attnO.cols), &beta, s.attnProj, CUDA_R_32F,
                        static_cast<int>(w.attnO.rows), CUDA_R_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st != CUBLAS_STATUS_SUCCESS) {
                    errMsg = "cublasGemmEx attnO failed";
                    return false;
                }
            }
            kAddResidual<<<(seqLen * H + 255) / 256, 256, 0, g_stream>>>(
                    s.hidden, s.attnProj, seqLen * H);

            // FFN RMSNorm -> s.norm (OUT-OF-PLACE, one block per row).
            kRMSNormRow<<<seqLen, 256, 256 * sizeof(float), g_stream>>>(
                    s.hidden, s.norm, w.rmsNormFFN, H, seqLen, 1e-6f);

            // (TEMP DEBUG diagnostics moved to after the FFN residual below.)

            // gate / up + SwiGLU: inputs are the FFN RMSNorm'd vector (s.norm).
            if (traceStages) cudaEventRecord(evMtxGateUp[L], g_stream);
            if (seqLen == 1) {
                launchQGemv(w.ffnGate.type, w.ffnGate.q, s.norm, s.gate,
                            w.ffnGate.rows, w.ffnGate.cols, w.ffnGate.rowBytes,
                            w.ffnGate.blocksPerRow);
                launchQGemv(w.ffnUp.type, w.ffnUp.q, s.norm, s.up, w.ffnUp.rows,
                            w.ffnUp.cols, w.ffnUp.rowBytes, w.ffnUp.blocksPerRow);
            } else {
                // hiddenF16 was converted before Q/K/V, but s.norm has since
                // been refreshed by the FFN RMSNorm, so rebuild the fp16 twin
                // from s.norm for the FFN GEMMs.
                kF32ToF16<<<(seqLen * H + 511) / 512, 256, 0, g_stream>>>(
                        s.norm, reinterpret_cast<__half2 *>(s.hiddenF16),
                        (seqLen * H) / 2);
                // CUDA_R_32F compute type -> alpha/beta must be float*.
                float alpha = 1.0f, beta = 0.0f;
                // gate/up[seqLen][rows] = norm[seqLen][cols] @ W[rows][cols]^T
                // (CUBLAS column-major: W in A, x in B, ldc=rows).
                cublasStatus_t st = cublasGemmEx(
                        g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        static_cast<int>(w.ffnGate.rows), static_cast<int>(seqLen),
                        static_cast<int>(w.ffnGate.cols), &alpha, w.ffnGate.f16,
                        CUDA_R_16F, static_cast<int>(w.ffnGate.cols), s.hiddenF16,
                        CUDA_R_16F, static_cast<int>(w.ffnGate.cols), &beta,
                        s.gate, CUDA_R_32F, static_cast<int>(w.ffnGate.rows),
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cublasStatus_t st2 = cublasGemmEx(
                        g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        static_cast<int>(w.ffnUp.rows), static_cast<int>(seqLen),
                        static_cast<int>(w.ffnUp.cols), &alpha, w.ffnUp.f16,
                        CUDA_R_16F, static_cast<int>(w.ffnUp.cols), s.hiddenF16,
                        CUDA_R_16F, static_cast<int>(w.ffnUp.cols), &beta,
                        s.up, CUDA_R_32F, static_cast<int>(w.ffnUp.rows),
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st != CUBLAS_STATUS_SUCCESS || st2 != CUBLAS_STATUS_SUCCESS) {
                    errMsg = "cublasGemmEx gate/up failed";
                    return false;
                }
            }
            kSiluMul<<<(seqLen * I + 255) / 256, 256, 0, g_stream>>>(s.gate, s.up,
                                                                     seqLen * I);

            // down projection + residual
            if (traceStages) cudaEventRecord(evMtxDown[L], g_stream);
            if (seqLen == 1) {
                launchQGemv(w.ffnDown.type, w.ffnDown.q, s.gate, s.ffnOut,
                            w.ffnDown.rows, w.ffnDown.cols, w.ffnDown.rowBytes,
                            w.ffnDown.blocksPerRow);
            } else {
                // gate holds silu(gate)*up in fp32; fp16 twin for the GEMM.
                kF32ToF16<<<(seqLen * I + 511) / 512, 256, 0, g_stream>>>(
                        s.gate, reinterpret_cast<__half2 *>(s.gateF16),
                        (seqLen * I) / 2);
                // CUDA_R_32F compute type -> alpha/beta must be float*.
                float alpha = 1.0f, beta = 0.0f;
                // ffnOut[seqLen][rows] = gate[seqLen][cols] @ W[rows][cols]^T
                // (CUBLAS column-major: W in A, x in B, ldc=rows).
                cublasStatus_t st = cublasGemmEx(
                        g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        static_cast<int>(w.ffnDown.rows), static_cast<int>(seqLen),
                        static_cast<int>(w.ffnDown.cols), &alpha, w.ffnDown.f16,
                        CUDA_R_16F, static_cast<int>(w.ffnDown.cols), s.gateF16,
                        CUDA_R_16F, static_cast<int>(w.ffnDown.cols), &beta,
                        s.ffnOut, CUDA_R_32F, static_cast<int>(w.ffnDown.rows),
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st != CUBLAS_STATUS_SUCCESS) {
                    errMsg = "cublasGemmEx ffnDown failed";
                    return false;
                }
            }
            kAddResidual<<<(seqLen * H + 255) / 256, 256, 0, g_stream>>>(
                    s.hidden, s.ffnOut, seqLen * H);

            if (g_verbose) {
                cudaError_t e = cudaStreamSynchronize(g_stream);
                std::fprintf(stderr, "[gpu loop] L=%u ffn: %s\n", L,
                             e == cudaSuccess ? "ok"
                                              : cudaGetErrorString(e));
            }
            if (traceStages) cudaEventRecord(evFFN[L], g_stream);
        }
        cudaError_t llErr = cudaStreamSynchronize(g_stream);
        if (llErr != cudaSuccess) {
            if (evL0) cudaEventDestroy(evL0);
            if (evL1) cudaEventDestroy(evL1);
            errMsg = std::string("GPU layer loop sync: ") + cudaGetErrorString(llErr);
            return false;
        }
        if (g_verbose) {
            cudaEventRecord(evL1, g_stream);
            cudaEventSynchronize(evL1);
            float gpuMs = 0.0f;
            cudaEventElapsedTime(&gpuMs, evL0, evL1);
            std::fprintf(stderr, "[gpu fwd   gpu-ms] layer loop GPU elapsed = %.2f ms (%u layers)\n",
                         gpuMs, nGpu);
            cudaEventDestroy(evL0);
            cudaEventDestroy(evL1);
            evL0 = evL1 = nullptr;
        }
        if (traceStages && g_verbose) {
            double accQKV = 0.0, accAttn = 0.0, accFFN = 0.0;
            for (uint32_t i = 0; i < nGpu; ++i) {
                float a = 0.0f, b = 0.0f, c = 0.0f;
                if (i == 0) {
                    cudaEventElapsedTime(&a, evL0, evQKV[0]);
                } else {
                    cudaEventElapsedTime(&a, evFFN[i - 1], evQKV[i]);
                }
                cudaEventElapsedTime(&b, evQKV[i], evAttn[i]);
                cudaEventElapsedTime(&c, evAttn[i], evFFN[i]);
                accQKV += a;
                accAttn += b;
                accFFN += c;
            }
            std::fprintf(stderr,
                         "[gpu fwd   gpu-ms] decode stage sums: QKV/rms=%.2f ms, kv/attn/rope=%.2f ms, attnO/ffn=%.2f ms\n",
                         accQKV, accAttn, accFFN);
            // per-matrix sums inside the FFN-heavy stage (event stream order per
            // layer: evMtxAttnO, evMtxGateUp, evMtxDown, evFFN).  attnO span =
            // attnO GEMV + attnO residual + FFN RMSNorm; gate+up span = gate
            // GEMV + up GEMV + kSiluMul; down span = down GEMV + residual.
            double accAO = 0.0, accGU = 0.0, accDn = 0.0;
            for (uint32_t i = 0; i < nGpu; ++i) {
                float aa = 0.0f, gu = 0.0f, dn = 0.0f;
                cudaEventElapsedTime(&aa, evMtxAttnO[i], evMtxGateUp[i]);
                cudaEventElapsedTime(&gu, evMtxGateUp[i], evMtxDown[i]);
                cudaEventElapsedTime(&dn, evMtxDown[i], evFFN[i]);
                accAO += aa;
                accGU += gu;
                accDn += dn;
            }
            std::fprintf(stderr,
                         "[gpu fwd   gpu-ms] decode matrix sums: attnO=%.2f ms, gate+up=%.2f ms, down=%.2f ms\n",
                         accAO, accGU, accDn);
            for (uint32_t i = 0; i < nGpu; ++i) {
                cudaEventDestroy(evQKV[i]);
                cudaEventDestroy(evAttn[i]);
                cudaEventDestroy(evFFN[i]);
                cudaEventDestroy(evMtxAttnO[i]);
                cudaEventDestroy(evMtxGateUp[i]);
                cudaEventDestroy(evMtxDown[i]);
            }
        }

        kvPos_ += seqLen;

        if (!fullOffload) {
            // Partial offload: forward() stops after the offloaded layer prefix
            // and leaves the hidden state on the device.  The caller picks it up
            // via copyHiddenOut() and continues the remaining layers on the CPU
            // against the host KV cache.  (kvPos_ has already advanced.)
            cudaStreamSynchronize(g_stream);
            return true;
        }

        // ---- Full offload: final RMSNorm + LM head ----
        TIMING_TAG("layer loop done");
        // Final RMSNorm -> place the result INTO s.hidden (the residual stream
        // is no longer needed once the last layer's residual add has run).
        // One block per row.
        if (finalNorm_) {
            kRMSNormRow<<<seqLen, 256, 256 * sizeof(float), g_stream>>>(
                    s.hidden, s.hidden, finalNorm_, H, seqLen, 1e-6f);
        }

        // LM head: quantized GEMV. Tied head -> token-embedding matrix
        // (embedQ_). Separate output.weight -> lmHeadQ_ (when uploaded).
        TIMING_TAG("lmhead start");
        const void *lmQ = lmHeadQ_ != nullptr ? lmHeadQ_ : embedQ_;
        uint32_t lmType = lmHeadQ_ != nullptr ? lmHeadType_ : embedType_;
        uint32_t lmRowBytes = lmHeadQ_ != nullptr ? lmHeadRowBytes_ : embedRowBytes_;
        uint32_t blocksPerRow = (H + 255) / 256;
        if (computeAllLogits) {
            for (uint32_t si = 0; si < seqLen; ++si) {
                launchQGemv(lmType, lmQ, s.hidden + si * H,
                            s.logits + si * geom_.vocabSize, geom_.vocabSize, H,
                            lmRowBytes, blocksPerRow);
            }
            cudaMemcpyAsync(logitsOut, s.logits,
                            seqLen * geom_.vocabSize * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream);
        } else {
            uint32_t si = seqLen - 1;
            launchQGemv(lmType, lmQ, s.hidden + si * H, s.logits,
                        geom_.vocabSize, H, lmRowBytes, blocksPerRow);
            cudaMemcpyAsync(logitsOut, s.logits, geom_.vocabSize * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream);
        }
        cudaError_t lerr = cudaGetLastError();
        if (lerr != cudaSuccess) {
            errMsg = std::string("GPU lmhead launch: ") + cudaGetErrorString(lerr);
            return false;
        }

        cudaError_t fin = cudaStreamSynchronize(g_stream);
        if (fin != cudaSuccess) {
            errMsg = std::string("GPU forward final sync: ") +
                     cudaGetErrorString(fin);
            return false;
        }
        TIMING_TAG("forward complete");
        return true;
    }

    bool GPUModel::copyHiddenOut(float *hiddenOut, uint32_t seqLen,
                                 std::string &errMsg) {
        if (!allocated_ || !scratchAlloc_) {
            errMsg = "GPU model not uploaded / no hidden state";
            return false;
        }
        if (seqLen > scratch_.seqCap) {
            errMsg = "copyHiddenOut: seqLen exceeds scratch capacity";
            return false;
        }
        cudaError_t e = cudaMemcpy(hiddenOut, scratch_.hidden,
                                   static_cast<size_t>(seqLen) * geom_.hiddenSize *
                                           sizeof(float),
                                   cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            errMsg = std::string("copyHiddenOut: ") + cudaGetErrorString(e);
            return false;
        }
        return true;
    }

    void GPUModel::clearKVCache() {
        if (!allocated_) return;
        uint32_t nGpu = geom_.numGpuLayers;
        uint64_t kvBytes = static_cast<uint64_t>(geom_.maxSeqLen) *
                           geom_.numKVHeads * geom_.headDim * sizeof(float);
        cudaMemsetAsync(kvK_, 0, static_cast<size_t>(nGpu) * kvBytes, g_stream);
        cudaMemsetAsync(kvV_, 0, static_cast<size_t>(nGpu) * kvBytes, g_stream);
        kvPos_ = 0;
        cudaStreamSynchronize(g_stream);
    }

    void GPUModel::destroy() {
        if (!allocated_) return;
        if (layers_) {
            uint32_t nGpu = geom_.numGpuLayers;
            for (uint32_t L = 0; L < nGpu; ++L) {
                DeviceLayer &l = layers_[L];
                auto freeMat = [](DeviceMatrix &m) {
                    cudaFree(m.q);
                    cudaFree(m.f16);
                    m.q = m.f16 = nullptr;
                };
                auto freeF32 = [](float *&p) {
                    cudaFree(p);
                    p = nullptr;
                };
                freeMat(l.attnQ);
                freeMat(l.attnK);
                freeMat(l.attnV);
                freeMat(l.attnO);
                freeMat(l.ffnGate);
                freeMat(l.ffnUp);
                freeMat(l.ffnDown);
                freeF32(l.attnQBias);
                freeF32(l.attnKBias);
                freeF32(l.attnVBias);
                freeF32(l.rmsNormAttn);
                freeF32(l.rmsNormFFN);
            }
            delete[] layers_;
            layers_ = nullptr;
        }
        cudaFree(kvK_);
        cudaFree(kvV_);
        cudaFree(finalNorm_);
        cudaFree(embedQ_);
        cudaFree(lmHeadQ_);
        cudaFree(ropeCos_);
        cudaFree(ropeSin_);
        kvK_ = kvV_ = nullptr;
        finalNorm_ = embedQ_ = nullptr;
        lmHeadQ_ = nullptr;
        ropeCos_ = ropeSin_ = nullptr;
        destroyScratch();
        allocated_ = false;
    }

    GPUModel::~GPUModel() { destroy(); }

}// namespace tinycoder::gpu

#endif// USE_CUDA
