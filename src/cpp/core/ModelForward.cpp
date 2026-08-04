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

#include "GGMLDequantize.hpp"
#include "LMHead.hpp"
#include "Model.hpp"
#include "ModelInternal.hpp"
#include "SIMDMatMulVec.hpp"
#include "ThreadPool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

#ifdef USE_CUDA
#include "GPUCompute.hpp"
#include "LMHeadCUDA.hpp"
#endif

namespace tinycoder {

    using namespace detail;

    np::Array<float> Model::forward(const std::vector<int32_t> &tokens,
                                    bool computeAllLogits) {
        if (tokens.empty()) {
            return np::Array<float>{};
        }

#ifdef USE_CUDA
        // GPU offload fast path: the GPU engine owns its own KV cache (kvPos_)
        // and is authoritative for the whole session once the first pass
        // succeeds.  A session is eligible when:
        //   * it starts fresh from a clean cache (kvCache_.pos == 0), OR
        //   * the GPU engine already owns the session (gpuSessionActive_ latch;
        //     decode steps have kvCache_.pos > 0).
        // clearKVCache() resets both caches and clears the latch, so a new
        // session re-enters on pos == 0.
        // The dispatch is deliberately conservative: only qwen2 (dense, no
        // MoE/SSM). Both tied and separate LM heads are supported by the GPU
        // engine (separate output.weight is uploaded as a dedicated Q6_K
        // tensor).  Other architectures keep the CPU path.
        if (gpu::gpuEnabled() && config_.architecture == ARCH_QWEN2 &&
            !tokens.empty() && (kvCache_.pos == 0 || gpuSessionActive_)) {
            std::string err;
            // Same construction idiom as the CPU logits path below (rank-2
            // [seqLen, vocabSize]): np::Array<float>(np::Shape{...}) directly.
            // A default-constructed rank-0 Array resized with a rank-1 shape
            // hits a UB trap in np's NDArrayShaped machinery (compiled into an
            // infinite loop by GCC here).
            np::Array<float> logits = np::Array<float>(np::Shape{computeAllLogits
                                                                         ? tokens.size()
                                                                         : size_t(1),
                                                                 config_.vocabSize});
            if (gpuForward(tokens, computeAllLogits, logits.data(), &err)) {
                kvCache_.pos += tokens.size();
                gpuSessionActive_ = true;// latch: session now owned by GPU
                return logits;
            }
            if (!err.empty()) {
                std::fprintf(stderr, "[gpu] forward fallback to CPU: %s\n", err.c_str());
            } else {
                std::fprintf(stderr, "[gpu] forward fallback to CPU (no detail)\n");
            }
            // Fall through to the CPU path below.  The GPU engine's KV cache is
            // NOT consumed on the fallback (kvPos_ reverts to its pre-pass
            // value) and the session latch is cleared, so the CPU-side KV
            // cache (kvCache_) remains authoritative for the remaining steps.
            gpuSessionActive_ = false;
        }
#endif

        uint32_t seqLen = static_cast<uint32_t>(tokens.size());
        uint32_t hiddenSize = config_.hiddenSize;
        uint32_t nHeads = config_.numAttentionHeads;
        uint32_t nKVHeads = config_.numKVHeads;
        uint32_t headDim = config_.headDim;
        uint32_t nLayers = config_.numLayers;
        uint32_t maxSeqLen = config_.maxSeqLen;
        uint32_t vocabSize = config_.vocabSize;

        // Fast path for single-token generation (seqLen == 1): run the per-token
        // work directly on the calling thread, bypassing ThreadPool dispatch
        // (re-entrancy guard, atomic loads, and std::function call overhead).
        // For seqLen > 1, dispatch through the thread pool as usual.
        auto runParallel = [&](auto &&func) {
            if (seqLen == 1) {
                func(0);
            } else {
                ThreadPool::instance().parallelFor(0, seqLen,
                                                   std::forward<decltype(func)>(func));
            }
        };

        // Reusable per-thread scratch buffers. These are hoisted onto the Model
        // (one pool per thread) so that steady-state token-by-token generation
        // performs zero heap allocations for intermediate state. Buffers grow on
        // demand and are retained across forward() calls.
        ScratchPool &scratch = scratchPool();

        // Allocate hidden state: [seqLen, hiddenSize]
        scratch.hidden.resize(static_cast<size_t>(seqLen) * hiddenSize);
        float *hiddenData = scratch.hidden.data();

        // Token embeddings: [seqLen, hiddenSize]
        // Dequantize from quantized format on-the-fly (parallel over tokens)
        runParallel([&](uint32_t i) {
            int32_t tokenId = tokens[i];
            if (tokenId >= 0 &&
                tokenId < static_cast<int32_t>(quantizedEmbeddings_.vocabSize)) {
                auto embRow = quantizedEmbeddings_.getRow(tokenId);
                float *hRow = hiddenData + i * hiddenSize;
                std::memcpy(hRow, embRow.data(), hiddenSize * sizeof(float));
            }
        });

        // Pre-allocate per-layer buffers (reused across layers)
        // attnNorm: [seqLen, hiddenSize]
        scratch.attnNorm.resize(static_cast<size_t>(seqLen) * hiddenSize);
        float *attnNormData = scratch.attnNorm.data();

        // Q, K, V projections
        scratch.q.resize(static_cast<size_t>(seqLen) * nHeads * headDim);
        scratch.k.resize(static_cast<size_t>(seqLen) * nKVHeads * headDim);
        scratch.v.resize(static_cast<size_t>(seqLen) * nKVHeads * headDim);
        float *qData = scratch.q.data();
        float *kData = scratch.k.data();
        float *vData = scratch.v.data();

        // Attention output and projection
        scratch.attnOut.resize(static_cast<size_t>(seqLen) * nHeads * headDim);
        float *attnOutData = scratch.attnOut.data();
        scratch.attnProj.resize(static_cast<size_t>(seqLen) * hiddenSize);
        float *attnProjData = scratch.attnProj.data();

        // FFN buffers
        uint32_t intermediateSize = config_.intermediateSize;
        scratch.ffnNorm.resize(static_cast<size_t>(seqLen) * hiddenSize);
        float *ffnNormData = scratch.ffnNorm.data();
        scratch.gate.resize(static_cast<size_t>(seqLen) * intermediateSize);
        scratch.up.resize(static_cast<size_t>(seqLen) * intermediateSize);
        float *gateData = scratch.gate.data();
        float *upData = scratch.up.data();
        scratch.ffnOut.resize(static_cast<size_t>(seqLen) * hiddenSize);

        for (uint32_t layer = 0; layer < nLayers; ++layer) {
            auto &w = layers_[layer];

            // True when the single-token Q3_K attnO kernel fused the attention
            // residual into hidden (hidden += attnO@attnOut) in its store
            // epilogue, in which case the separate addSIMD residual pass below
            // must be skipped.
            bool attnResidualFused = false;

            // Check if this is an SSM layer for Qwen35MoE architecture
            bool isSSMLayer = (config_.architecture == ARCH_QWEN35MOE &&
                               config_.fullAttentionInterval > 0 &&
                               (layer % config_.fullAttentionInterval) != 0 &&
                               !w.ssmOut.empty());

            if (isSSMLayer) {
                // ---- SSM (Mamba-style) block replaces attention ----

                // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                const float *rmsNormAttnData = w.rmsNormAttn.data();
                runParallel([&](uint32_t s) {
                    rmsNormSIMD(hiddenData + s * hiddenSize, attnNormData + s * hiddenSize,
                                rmsNormAttnData, hiddenSize);
                });

                // SSM computation for each token
                uint32_t ssmInnerSize = config_.ssmInnerSize;
                uint32_t ssmStateSize = config_.ssmStateSize;
                uint32_t ssmConvKernel = config_.ssmConvKernel;

                // Pre-allocate SSM input buffer (reused across tokens)
                scratch.ssmIn.resize(ssmInnerSize);
                float *ssmInBuf = scratch.ssmIn.data();

                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *hRowPtr = attnNormData + s * hiddenSize;

                    // Step 1: Input projection (hiddenSize → ssmInnerSize)
                    // Write directly to pre-allocated buffer, avoiding heap allocation + memcpy
                    w.ssmOut.matMulVec(hRowPtr, ssmInBuf);

                    // Step 2: Conv1d with past buffer
                    scratch.ssmConvOut.resize(ssmInnerSize);
                    float *convOut = scratch.ssmConvOut.data();
                    if (ssmConvKernel > 1 && !w.ssmConv1d.empty()) {
                        scratch.ssmConvInput.resize(ssmConvKernel * ssmInnerSize);
                        float *convInput = scratch.ssmConvInput.data();
                        auto &convBuf = kvCache_.ssmConvBuf[layer];
                        uint32_t bufLen = ssmConvKernel - 1;
                        std::memcpy(convInput, convBuf.data(), bufLen * ssmInnerSize * sizeof(float));
                        std::memcpy(convInput + bufLen * ssmInnerSize,
                                    ssmInBuf, ssmInnerSize * sizeof(float));

                        // Update conv buffer with current input (shift)
                        std::memmove(convBuf.data(), convBuf.data() + ssmInnerSize,
                                     (bufLen - 1) * ssmInnerSize * sizeof(float));
                        std::memcpy(convBuf.data() + (bufLen - 1) * ssmInnerSize,
                                    ssmInBuf, ssmInnerSize * sizeof(float));

                        for (uint32_t c = 0; c < ssmInnerSize; ++c) {
                            const float *wRow = reinterpret_cast<const float *>(w.ssmConv1d.data.data()) + static_cast<size_t>(c) * ssmConvKernel;
                            convOut[c] = dotProductFMA(wRow, convInput + c * ssmConvKernel, ssmConvKernel);
                        }
                    } else {
                        std::memcpy(convOut, ssmInBuf, ssmInnerSize * sizeof(float));
                    }

                    // Step 3: SiLU activation on conv output
                    siluSIMD(convOut, ssmInnerSize);

                    // Step 4: SSM state update
                    auto &ssmState = kvCache_.ssmState[layer];
                    for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                        float dt = std::log(1.0f + std::exp(w.ssmDtBias.data()[i]));
                        for (uint32_t j = 0; j < ssmStateSize; ++j) {
                            float aVal = w.ssmA.data()[i * ssmStateSize + j];
                            float aBar = std::exp(aVal * dt);
                            uint32_t idx = i * ssmStateSize + j;
                            ssmState[idx] = aBar * ssmState[idx] + convOut[i];
                        }
                    }

                    // Step 5: Output from SSM state
                    scratch.ssmOut.resize(ssmInnerSize);
                    float *ssmOutBuf = scratch.ssmOut.data();
                    for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                        double hVal = 0.0;
                        for (uint32_t j = 0; j < ssmStateSize; ++j) {
                            hVal += ssmState[i * ssmStateSize + j];
                        }
                        float gateVal = w.ssmAlpha.data()[i] * static_cast<float>(hVal) + w.ssmBeta.data()[i];
                        float gateAct = gateVal / (1.0f + std::exp(-gateVal));
                        ssmOutBuf[i] = convOut[i] * gateAct;
                    }

                    // Step 6: Output projection back to hiddenSize using attnO
                    // Write directly to pre-allocated attnProjData buffer
                    deqMatMulVecF16(w.attnO_deq_f16.data(), ssmOutBuf,
                                    w.attnO.rows, w.attnO.cols,
                                    attnProjData + s * hiddenSize);
                }

                // SSM residual (standard residual, no post-norm)
                for (uint32_t s = 0; s < seqLen; ++s) {
                    addSIMD(hiddenData + s * hiddenSize, attnProjData + s * hiddenSize, hiddenSize);
                }
            } else {
                // ---- Attention block ----

                // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                const float *rmsNormAttnData = w.rmsNormAttn.data();
                runParallel([&](uint32_t s) {
                    rmsNormSIMD(hiddenData + s * hiddenSize, attnNormData + s * hiddenSize,
                                rmsNormAttnData, hiddenSize);
                });

                // Project to Q, K, V. For prefill (seqLen>1) use batched GEMM:
                // parallelize over output rows and reuse each weight row across all
                // tokens, so the weight matrix is read once instead of seqLen times.
                // For single-token generation keep the per-token fused GEMV path.
                if (seqLen > 1) {
                    // Prefill QKV: use the register-tiled Q8_K batch GEMM for the
                    // pre-packed Q2_K Q/K projections (reuses each weight row across
                    // all tokens with the int8 _mm256_maddubs_epi16 kernel). This was
                    // the remaining prefill bottleneck (BENCHMARK_REPORT §7.2). attnV
                    // is usually Q4_K and stays on the generic row-parallel path.
                    if (w.attnQ.type == GGML_TYPE_Q2_K && !w.attnQ.prepackedData.empty() &&
                        w.attnQ.cols % 256 == 0) {
                        matMulVecBatchQ2_K_PrePacked_Q8_Batch_SIMD(
                                w.attnQ.prepackedData.data(), attnNormData, seqLen,
                                w.attnQ.rows, w.attnQ.cols, qData);
                    } else {
                        matMulVecBatchQuantized(w.attnQ, attnNormData, seqLen, qData);
                    }
                    // Route K through the register-tiled Q8_K batch GEMM (P1). The
                    // "known prepacked bug" was an unverified assumption: attnK's
                    // prepackedData is populated and its dims (rows=256 divisible by 8,
                    // cols=1536 divisible by 256) are handled correctly by the
                    // register-tiled kernel. The generic path re-reads each weight row
                    // once per token with the slower per-block scalar-scale handling;
                    // the register-tiled path reuses each weight row across all tokens
                    // with the int8 _mm256_maddubs_epi16 kernel, exactly like Q.
                    if (w.attnK.type == GGML_TYPE_Q2_K && !w.attnK.prepackedData.empty() &&
                        w.attnK.cols % 256 == 0) {
                        matMulVecBatchQ2_K_PrePacked_Q8_Batch_SIMD(
                                w.attnK.prepackedData.data(), attnNormData, seqLen,
                                w.attnK.rows, w.attnK.cols, kData);
                    } else {
                        matMulVecBatchQuantized(w.attnK, attnNormData, seqLen, kData);
                    }
                    // Route V through the register-tiled Q4_K batch GEMM (the
                    // generic row-parallel path re-reads each Q4_K weight row once
                    // per token with a scalar double-precision per-block loop).
                    // The vector kernel quantizes x to Q8_K once, tiles over 8
                    // rows and hoists the Q4_K scale/min unpacking out of the
                    // token loop (weight-stationary), with _mm256_maddubs_epi16
                    // sub-block dots.
                    {
                        ScopedProfile sp("matMulVecBatchQ4K");
                        if (w.attnV.type == GGML_TYPE_Q4_K && w.attnV.cols % 256 == 0 &&
                            matMulVecBatchQ4K_SIMD(w.attnV.data.data(), attnNormData,
                                                   seqLen, w.attnV.rows,
                                                   w.attnV.cols, vData)) {
                            // handled by the Q4_K batch kernel
                        } else {
                            matMulVecBatchQuantized(w.attnV, attnNormData, seqLen, vData);
                        }
                    }
                    if (!w.attnQBias.empty()) {
                        for (uint32_t s = 0; s < seqLen; ++s) {
                            addSIMD(qData + s * nHeads * headDim, w.attnQBias.data(),
                                    nHeads * headDim);
                        }
                    }
                    if (!w.attnKBias.empty()) {
                        for (uint32_t s = 0; s < seqLen; ++s) {
                            addSIMD(kData + s * nKVHeads * headDim, w.attnKBias.data(),
                                    nKVHeads * headDim);
                        }
                    }
                    if (!w.attnVBias.empty()) {
                        for (uint32_t s = 0; s < seqLen; ++s) {
                            addSIMD(vData + s * nKVHeads * headDim, w.attnVBias.data(),
                                    nKVHeads * headDim);
                        }
                    }
                } else {
                    // Single-token generation (seqLen == 1). Route Q/K/V through
                    // the register-tiled compact kernels: Q and K are compact
                    // Q2_K (84 B/block) and V is Q4_K (144 B/block) in this
                    // model. The fused compact Q+K kernel reads each Q/K block
                    // once and reuses the single Q8_K quantization of x across
                    // all rows of both matrices (a 3.3x weight-traffic cut vs
                    // the pre-packed Q2_K copy used by the prefill batch kernels,
                    // which matters because generation is DRAM-bandwidth-bound).
                    // The old fused scalar path (matMulVecFusedQKV) was
                    // compute-bound: ~344k indirect GGMLDequantize::dotProductFused
                    // calls per token (~5 GB/s effective, 5x below DRAM
                    // saturation) and was the largest unoptimized generation
                    // stage. (Called with seqLen==1, so runParallel invokes
                    // func(0) inline; the kernels' internal row tiling parallels
                    // across the pool.)
                    bool genFastQK = (w.attnQ.type == GGML_TYPE_Q2_K &&
                                      w.attnK.type == GGML_TYPE_Q2_K &&
                                      w.attnQ.cols == w.attnK.cols &&
                                      w.attnQ.cols % 256 == 0);
                    bool genFastV = (w.attnV.type == GGML_TYPE_Q4_K &&
                                     w.attnV.cols % 256 == 0);
                    runParallel([&](uint32_t s) {
                        const float *hRowPtr = attnNormData + s * hiddenSize;
                        float *qRowPtr = qData + s * nHeads * headDim;
                        float *kRowPtr = kData + s * nKVHeads * headDim;
                        float *vRowPtr = vData + s * nKVHeads * headDim;

                        if (genFastQK && genFastV) {
                            ScopedProfile sp("gen_matMulVecFusedQKV");
                            matMulVecFusedQKQ2_K_Compact_Q8_SIMD(
                                    w.attnQ.data.data(), w.attnK.data.data(),
                                    hRowPtr, w.attnQ.rows, w.attnK.rows,
                                    w.attnQ.cols, qRowPtr, kRowPtr);
                            matMulVecBatchQ4K_SIMD(w.attnV.data.data(), hRowPtr, 1,
                                                   w.attnV.rows, w.attnV.cols,
                                                   vRowPtr);
                        } else {
                            // Fallback for other quant types / hosts without the
                            // vector kernels: fused scalar QKV.
                            matMulVecFusedQKV(w.attnQ, w.attnK, w.attnV, hRowPtr,
                                              qRowPtr, kRowPtr, vRowPtr);
                        }

                        // Qwen2: separate Q/K/V with biases. Gemma4/Qwen35MoE do
                        // not carry Q/K/V biases (linear attention projections).
                        if (config_.architecture == ARCH_QWEN2) {
                            if (!w.attnQBias.empty()) {
                                addSIMD(qRowPtr, w.attnQBias.data(), nHeads * headDim);
                            }
                            if (!w.attnKBias.empty()) {
                                addSIMD(kRowPtr, w.attnKBias.data(), nKVHeads * headDim);
                            }
                            if (!w.attnVBias.empty()) {
                                addSIMD(vRowPtr, w.attnVBias.data(), nKVHeads * headDim);
                            }
                        }
                    });
                }

                // Apply Q/K norms before RoPE (Gemma4 and Qwen35MoE)
                if (config_.architecture == ARCH_GEMMA4) {
                    if (!w.attnQNorm.empty() && !w.attnKNorm.empty()) {
                        applyQKNorms(qData, kData, seqLen, nHeads, nKVHeads,
                                     w.attnQNorm.data(), w.attnKNorm.data());
                    }
                } else if (config_.architecture == ARCH_QWEN35MOE) {
                    if (!w.attnQNormMoe.empty() && !w.attnKNormMoe.empty()) {
                        applyQKNorms(qData, kData, seqLen, nHeads, nKVHeads,
                                     w.attnQNormMoe.data(), w.attnKNormMoe.data());
                    }
                }

                // Apply RoPE (P3: K rotation fused into the store below, so
                // skip the separate K-rotation pass here).
                applyRoPE(qData, kData, seqLen, seqLen, nHeads, nKVHeads,
                          static_cast<uint32_t>(kvCache_.pos), /*rotateK=*/false);

                // Store K, V in cache (K-RoPE fused into the K write, P3).
                uint32_t cachePos = static_cast<uint32_t>(kvCache_.pos);
                float *kCacheLayer =
                        kvCache_.k.data() + layer * maxSeqLen * nKVHeads * headDim;
                float *vCacheLayer =
                        kvCache_.v.data() + layer * maxSeqLen * nKVHeads * headDim;

                storeKVWithRoPE(kData, vData, kCacheLayer, vCacheLayer, seqLen,
                                cachePos, nKVHeads);

                // Attention with cached K, V
                uint32_t totalCacheLen = cachePos + seqLen;
                attentionFused(qData, kCacheLayer, vCacheLayer, attnOutData, seqLen,
                               cachePos, totalCacheLen, layer);

                // Output projection. For prefill use batched GEMM so the weight
                // matrix is read once for all tokens. For generation (seqLen==1)
                // the model stores attnO as Q3_K (110 B/block = 0.43 B/elem),
                // so the compact Q3_K batch GEMM is used in preference to the
                // Q8_K copy (1.14 B/elem) and FP16 (2 B/elem): generation is
                // DRAM-bandwidth-bound, so reading the raw Q3_K blocks directly
                // cuts weight traffic ~2.65x vs Q8_K (measurement: attnO+ffnDown
                // ~35% of per-token time).
                {
                    ScopedProfile sp("matMulVecBatchQ3K");
                    // Stage fusion (attnO + residual): for single-token
                    // generation the compact Q3_K batch kernel accumulates the
                    // attention output projection DIRECTLY into hidden
                    // (out == residual == hiddenData, so the store epilogue
                    // performs hidden[i] += attnO@attnOut[i]). This eliminates
                    // the attnProj buffer write/read round-trip (~6 KB) and the
                    // separate addSIMD pass over the hidden vector per layer.
                    // Prefill (seqLen > 1) and all fallbacks keep the plain
                    // attnProjData path (residual applied by the addSIMD below).
                    attnResidualFused = false;
                    if (seqLen == 1 && w.attnO.type == GGML_TYPE_Q3_K &&
                        w.attnO.cols % 256 == 0 &&
                        matMulVecBatchQ3K_SIMD(w.attnO.data.data(), attnOutData,
                                               seqLen, w.attnO.rows,
                                               w.attnO.cols, hiddenData,
                                               /*residual=*/hiddenData)) {
                        attnResidualFused = true;
                        // handled by the compact Q3_K batch kernel (fused)
                    } else if (w.attnO.type == GGML_TYPE_Q3_K &&
                               w.attnO.cols % 256 == 0 &&
                               matMulVecBatchQ3K_SIMD(w.attnO.data.data(), attnOutData,
                                                      seqLen, w.attnO.rows,
                                                      w.attnO.cols, attnProjData)) {
                        // handled by the compact Q3_K batch kernel (prefill)
                    } else if (!w.attnO_q8k.empty()) {
                        // P4 + gen: Q8_K batch GEMM (vector kernel supports seqLen==1).
                        matMulVecBatchQ8K(w.attnO_q8k.data(), attnOutData, seqLen,
                                          w.attnO.rows, w.attnO.cols, attnProjData);
                    } else if (seqLen > 1) {
                        deqMatMulVecF16_Batch(w.attnO_deq_f16.data(), attnOutData, seqLen,
                                              w.attnO.rows, w.attnO.cols, attnProjData);
                    } else {
                        runParallel([&](uint32_t s) {
                            const float *attnRowPtr = attnOutData + s * nHeads * headDim;
                            deqMatMulVecF16(w.attnO_deq_f16.data(), attnRowPtr,
                                            w.attnO.rows, w.attnO.cols,
                                            attnProjData + s * hiddenSize);
                        });
                    }
                }

                // Attention residual + post-attention processing
                if (config_.architecture == ARCH_GEMMA4 && !w.postAttnNorm.empty()) {
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        float *hPtr = hiddenData + s * hiddenSize;
                        const float *aPtr = attnProjData + s * hiddenSize;
                        addSIMD(hPtr, aPtr, hiddenSize);
                        rmsNormSIMD(hPtr, hPtr, w.postAttnNorm.data(), hiddenSize);
                        if (!w.layerOutputScale.empty()) {
                            scaleSIMD(hPtr, w.layerOutputScale.data()[0], hiddenSize);
                        }
                    }
                } else if (!attnResidualFused) {
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        addSIMD(hiddenData + s * hiddenSize, attnProjData + s * hiddenSize, hiddenSize);
                    }
                }
                // (else: attnResidualFused == true — the Q3_K batch kernel already
                // applied hidden += attnO@attnOut in its store epilogue.)
            }

            // ---- FFN block ----

            // RMSNorm: ffnNorm = rmsNorm(hidden, w.rmsNormFFN)
            const float *rmsNormFFNData = w.rmsNormFFN.data();
            runParallel([&](uint32_t s) {
                rmsNormSIMD(hiddenData + s * hiddenSize, ffnNormData + s * hiddenSize,
                            rmsNormFFNData, hiddenSize);
            });

            // MoE path (expertCount > 0)
            if (config_.architecture == ARCH_GEMMA4 && config_.expertCount > 0) {
                computeGemma4MoE(ffnNormData, scratch.ffnOut.data(), seqLen, hiddenSize, intermediateSize, w);
                for (uint32_t s = 0; s < seqLen; ++s) {
                    float *hPtr = hiddenData + s * hiddenSize;
                    const float *fPtr = scratch.ffnOut.data() + s * hiddenSize;
                    addSIMD(hPtr, fPtr, hiddenSize);
                    if (!w.layerOutputScale.empty()) {
                        scaleSIMD(hPtr, w.layerOutputScale.data()[0], hiddenSize);
                    }
                }
            } else if (config_.architecture == ARCH_QWEN35MOE && config_.expertCount > 0) {
                computeQwen35MoE(ffnNormData, scratch.ffnOut.data(), seqLen, hiddenSize, intermediateSize, w);
                for (uint32_t s = 0; s < seqLen; ++s) {
                    addSIMD(hiddenData + s * hiddenSize, scratch.ffnOut.data() + s * hiddenSize, hiddenSize);
                }
            } else {
                // Dense FFN path
                if (seqLen > 1) {
                    // Prefill: batched GEMM for gate+up and down projections.
                    // Parallelize over output rows, reusing each weight row across
                    // all tokens (weight matrix read once instead of seqLen times).
                    // P5: for SwiGLU architectures the silu(gate)*up activation is
                    // fused into the gate+up kernel's epilogue, so no separate
                    // activation pass is needed here. Gemma4 (GeGLU) must still
                    // apply gelu(gate)*up after the matmul.
                    matMulVecFusedGateUp_Batch(
                            w.ffnGate, w.ffnUp, ffnNormData, seqLen,
                            gateData, upData,
                            config_.architecture == ARCH_GEMMA4 ? false : true);
                    if (config_.architecture == ARCH_GEMMA4) {
                        // GeGLU: gelu(gate) * up
                        for (uint32_t s = 0; s < seqLen; ++s) {
                            float *gatePtr = gateData + s * intermediateSize;
                            float *upPtr = upData + s * intermediateSize;
                            geluInPlace(gatePtr, intermediateSize);
                            for (uint32_t i = 0; i < intermediateSize; ++i) {
                                gatePtr[i] *= upPtr[i];
                            }
                        }
                    }

                    // Down projection fused with residual.
                    // Prefer the compact Q3_K batch GEMM (raw GGUF blocks, 0.43
                    // B/elem) when available — ffnDown is stored as Q3_K in the
                    // model file — then the Q8_K batch GEMM (1.14 B/elem). Both
                    // reuse each weight row across all tokens and use the int8
                    // _mm256_maddubs_epi16 kernel, cutting weight-matrix memory
                    // traffic vs the FP16 path.
                    // Prefer the exact Q3_K source blocks (raw GGUF bytes, 0.43
                    // B/elem) via the compact batch GEMM — this is the
                    // quality-correct default (llama.cpp streams the same Q3_K
                    // bytes; no lossy re-quant). Falls back to the Q8_K batch
                    // GEMM (1.14 B/elem) which is built losslessly from the
                    // same Q3_K source at load time.
                    if (w.ffnDown.type == GGML_TYPE_Q3_K &&
                        w.ffnDown.cols % 256 == 0 &&
                        matMulVecBatchQ3K_SIMD(w.ffnDown.data.data(), gateData,
                                               seqLen, w.ffnDown.rows,
                                               w.ffnDown.cols,
                                               scratch.ffnOut.data())) {
                        // handled by the compact Q3_K batch kernel
                    } else if (!w.ffnDown_q8k.empty()) {
                        // P4: Q8_K batch GEMM (int8 maddubs kernel, ~4x compute vs
                        // the FP16 path) for the largest prefill matmul.
                        matMulVecBatchQ8K(w.ffnDown_q8k.data(), gateData, seqLen,
                                          w.ffnDown.rows, w.ffnDown.cols,
                                          scratch.ffnOut.data());
                    } else if (w.ffnDown.type == GGML_TYPE_Q2_K &&
                               !w.ffnDown.prepackedData.empty() &&
                               w.ffnDown.cols % 256 == 0) {
                        matMulVecBatchQ2_K_PrePacked_Q8_Batch_SIMD(
                                w.ffnDown.prepackedData.data(), gateData, seqLen,
                                w.ffnDown.rows, w.ffnDown.cols,
                                scratch.ffnOut.data());
                    } else {
                        deqMatMulVecF16_Batch(w.ffnDown_deq_f16.data(), gateData, seqLen,
                                              w.ffnDown.rows, w.ffnDown.cols,
                                              scratch.ffnOut.data());
                    }
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        addSIMD(hiddenData + s * hiddenSize,
                                scratch.ffnOut.data() + s * hiddenSize, hiddenSize);
                    }
                } else if (config_.architecture == ARCH_GEMMA4) {
                    // Gemma4: GeGLU activation (gelu(gate) * up)
                    runParallel([&](uint32_t s) {
                        const float *ffnRowPtr = ffnNormData + s * hiddenSize;
                        float *gatePtr = gateData + s * intermediateSize;
                        float *upPtr = upData + s * intermediateSize;

                        // Fused gate+up: single pass over x for both projections.
                        // Reads x once instead of twice, halving the dominant
                        // memory traffic for the FFN gate+up matmuls.
                        w.ffnGate.matMulVecFusedGateUp(w.ffnUp, ffnRowPtr, gatePtr, upPtr);

                        // GeGLU: gelu(gate) * up
                        geluInPlace(gatePtr, intermediateSize);
                        for (uint32_t i = 0; i < intermediateSize; ++i) {
                            gatePtr[i] *= upPtr[i];
                        }
                    });
                } else {
                    // Qwen2, Qwen35MoE: SwiGLU activation (silu(gate) * up)
                    // For single-token generation, use the fused gate+up+down
                    // kernel that combines gate=Q2_K, up=Q2_K, down=Q3_K into
                    // a single kernel, eliminating the intermediate gate+up
                    // round-trip and the per-row x-vector reloads in ffnDown.
                    if (seqLen == 1 &&
                        w.ffnGate.type == GGML_TYPE_Q2_K &&
                        w.ffnUp.type == GGML_TYPE_Q2_K &&
                        w.ffnDown.type == GGML_TYPE_Q3_K &&
                        w.ffnGate.cols == hiddenSize &&
                        w.ffnUp.cols == hiddenSize &&
                        w.ffnDown.cols == intermediateSize &&
                        hiddenSize % 256 == 0 &&
                        intermediateSize % 256 == 0) {
                        // Fused gate+up+down kernel: eliminate intermediate
                        // gatePtr/upPtr buffers and the separate ffnDown dispatch.
                        // The kernel computes out = residual + down@(silu(gate@x)
                        // *up@x) and folds the residual into its store epilogue,
                        // so pass hidden as BOTH out and residual (in-place,
                        // exactly like the attnO+residual fusion in
                        // matMulVecBatchQ3K_SIMD) — hidden receives the FFN
                        // output, removing the downBuf round-trip and the
                        // separate addSIMD pass.
                        // The Q3_K source blocks for ffnDown are streamed directly
                        // (84 B/block gate+up, 110 B/block down), matching
                        // llama.cpp's exact-weight semantics at full quality.
                        ScopedProfile sp("fused_gateUp_ffnDown");
                        runParallel([&](uint32_t s) {
                            const float *ffnRowPtr = ffnNormData + s * hiddenSize;
                            float *hPtr = hiddenData + s * hiddenSize;
                            matMulVecFusedGateUpDownQ2K_Q3K_SIMD(
                                    w.ffnGate.data.data(),
                                    w.ffnUp.data.data(),
                                    w.ffnDown.data.data(),
                                    ffnRowPtr,
                                    intermediateSize,// gRows
                                    hiddenSize,      // hRows
                                    hiddenSize,      // cols (input to gate+up)
                                    hPtr,            // out (hiddenSize)
                                    hPtr             // residual (hidden += down)
                            );
                        });
                    } else {
                        // Fallback: separate gate+up then ffnDown path
                        ScopedProfile spGenGateUp("gen_gateUpSwiGLU");
                        runParallel([&](uint32_t s) {
                            const float *ffnRowPtr = ffnNormData + s * hiddenSize;
                            float *gatePtr = gateData + s * intermediateSize;
                            float *upPtr = upData + s * intermediateSize;

                            // Fused gate+up: single pass over x for both projections.
                            // Reads x once instead of twice, halving the dominant
                            // memory traffic for the FFN gate+up matmuls. The SwiGLU
                            // epilogue (silu(gate) * up) is fused into the kernel's
                            // store loop, so no separate activation pass is needed —
                            // gatePtr receives the activated output directly (upPtr
                            // stays the raw up projection, consumed by ffnDown below).
                            w.ffnGate.matMulVecFusedGateUp(w.ffnUp, ffnRowPtr, gatePtr, upPtr,
                                                           /*applySwish=*/true);
                        });

                        // Down projection fused with residual (single-token path)
                        if (seqLen == 1) {
                            runParallel([&](uint32_t s) {
                                const float *ffnActPtr = gateData + s * intermediateSize;
                                float *downBuf = scratch.ffnOut.data() + s * hiddenSize;
                                float *hPtr = hiddenData + s * hiddenSize;
                                // Generation is DRAM-bandwidth-bound: favor the compact
                                // Q3_K kernel (0.43 B/elem, raw GGUF blocks) over the
                                // Q8_K copy (1.14 B/elem) and FP16 (2 B/elem).
                                {
                                    ScopedProfile sp("gen_matMulVecQ3K_ffnDown");
                                    if (w.ffnDown.type == GGML_TYPE_Q3_K &&
                                        w.ffnDown.cols % 256 == 0 &&
                                        matMulVecBatchQ3K_SIMD(w.ffnDown.data.data(), ffnActPtr,
                                                               1, w.ffnDown.rows,
                                                               w.ffnDown.cols, downBuf)) {
                                        // handled by the compact Q3_K batch kernel
                                    } else if (!w.ffnDown_q8k.empty()) {
                                        matMulVecBatchQ8K(w.ffnDown_q8k.data(), ffnActPtr, 1,
                                                          w.ffnDown.rows, w.ffnDown.cols,
                                                          downBuf);
                                    } else {
                                        deqMatMulVecF16(w.ffnDown_deq_f16.data(), ffnActPtr,
                                                        w.ffnDown.rows, w.ffnDown.cols,
                                                        downBuf);
                                    }
                                }
                                addSIMD(hPtr, downBuf, hiddenSize);
                            });
                        }
                    }
                }

                // FFN residual + post-FFN processing
                if (config_.architecture == ARCH_GEMMA4 && !w.postFFWNorm.empty()) {
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        float *hPtr = hiddenData + s * hiddenSize;
                        rmsNormSIMD(hPtr, hPtr, w.postFFWNorm.data(), hiddenSize);
                        if (!w.layerOutputScale.empty()) {
                            scaleSIMD(hPtr, w.layerOutputScale.data()[0], hiddenSize);
                        }
                    }
                }
            }
        }

        // Final RMSNorm (parallel over tokens)
        const float *finalNormData = finalNorm_.data();
        runParallel([&](uint32_t s) {
            rmsNormSIMD(hiddenData + s * hiddenSize, hiddenData + s * hiddenSize,
                        finalNormData, hiddenSize);
        });

        // LM head (logits)
        np::Array<float> logits = np::Array<float>(np::Shape{seqLen, vocabSize});
        float *logitsData = logits.data();

        // Prefill optimization: when computeAllLogits is false, only the last
        // token's logits are needed (generate() samples from the last token).
        // The LM head reads the full vocabSize x hiddenSize embedding matrix per
        // token, so skipping it for all but the last token avoids re-reading that
        // large matrix seqLen times.
        uint32_t logitStart = computeAllLogits ? 0 : (seqLen > 0 ? seqLen - 1 : 0);
        uint32_t logitCount = computeAllLogits ? seqLen : (seqLen > 0 ? 1 : 0);

        ScopedProfile spGenLmHead(seqLen == 1 ? "gen_lmHead" : "prefill_lmHead");
        if (lmHeadTied_) {
#ifdef USE_CUDA
            // CUDA path (unchanged)
            static bool embedUploaded = false;
            static const float *d_embeddings = nullptr;
            static uint32_t cachedVocabSize = 0;
            static uint32_t cachedHiddenSize = 0;

            if (!embedUploaded || cachedVocabSize != quantizedEmbeddings_.vocabSize ||
                cachedHiddenSize != quantizedEmbeddings_.hiddenSize) {
                uint64_t embedElements =
                        static_cast<uint64_t>(quantizedEmbeddings_.vocabSize) *
                        quantizedEmbeddings_.hiddenSize;
                auto embedDeq = GGMLDequantize::dequantize(
                        quantizedEmbeddings_.type, quantizedEmbeddings_.data.data(),
                        embedElements);
                if (!embedDeq.empty()) {
                    try {
                        d_embeddings = cuda::uploadEmbeddings(embedDeq.data(), embedElements);
                        cachedVocabSize = quantizedEmbeddings_.vocabSize;
                        cachedHiddenSize = quantizedEmbeddings_.hiddenSize;
                        embedUploaded = true;
                    } catch (const std::exception &e) {
                        std::fprintf(stderr, "CUDA upload failed: %s\n", e.what());
                        embedUploaded = false;
                    }
                }
            }

            if (embedUploaded) {
                for (uint32_t s = logitStart; s < logitStart + logitCount; ++s) {
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    try {
                        cuda::computeLMHead(d_embeddings, cachedHiddenSize, cachedVocabSize,
                                            hPtr, logitRow);
                    } catch (const std::exception &e) {
                        std::fprintf(stderr, "CUDA LM head failed: %s\n", e.what());
                        if (!dequantizedEmbeddings_.empty()) {
                            LMHead::computeCPU(hPtr, dequantizedEmbeddings_.data.data(),
                                               dequantizedEmbeddings_.vocabSize,
                                               dequantizedEmbeddings_.hiddenSize, logitRow);
                        } else {
                            LMHead::computeCPUQuantized(hPtr, quantizedEmbeddings_.data.data(),
                                                        quantizedEmbeddings_.type,
                                                        quantizedEmbeddings_.vocabSize,
                                                        quantizedEmbeddings_.hiddenSize, logitRow);
                        }
                    }
                }
            } else {
                // Consolidated single parallelFor over the requested (token, vocab) pairs
                ThreadPool::instance().parallelFor(0, logitCount * vocabSize, [&](uint32_t flatIdx) {
                    uint32_t s = logitStart + flatIdx / vocabSize;
                    uint32_t i = flatIdx % vocabSize;
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    float dot = 0.0f;
                    if (!dequantizedEmbeddings_.empty()) {
                        const float *embRow = dequantizedEmbeddings_.data.data() + static_cast<uint64_t>(i) * hiddenSize;
                        dot = dotProductFMA(hPtr, embRow, hiddenSize);
                    } else {
                        uint32_t blockSize = ggmlBlockSize(quantizedEmbeddings_.type);
                        uint32_t typeSize = ggmlTypeSize(quantizedEmbeddings_.type);
                        uint32_t numBlocks = (hiddenSize + blockSize - 1) / blockSize;
                        for (uint32_t b = 0; b < numBlocks; ++b) {
                            uint64_t blockOffset = static_cast<uint64_t>(i) * numBlocks + b;
                            const uint8_t *blockData = quantizedEmbeddings_.data.data() + blockOffset * typeSize;
                            float blockOut[256];
                            GGMLDequantize::dequantizeBlock(quantizedEmbeddings_.type, blockData, blockOut, blockSize);
                            uint32_t start = b * blockSize;
                            uint32_t n = std::min(blockSize, hiddenSize - start);
                            dot += dotProductFMA(hPtr + start, blockOut, n);
                        }
                    }
                    logitRow[i] = dot;
                });
            }
#else
            // CPU path with pre-dequantized embeddings: consolidated single parallelFor.
            // P1 (BENCHMARK_REPORT §4.2): prefer the native Q2_K embeddings when they
            // are available (tied-embedding case). The LM head is the one matrix large
            // enough to be memory-bound in single-token generation; reading the native
            // Q2_K matrix (~77 MB/token) instead of a Q8_K copy (~233 MB/token) cuts the
            // dominant DRAM traffic ~3×. Each 256-element Q2_K block is 84 bytes and the
            // fused dot product dequantizes on the fly.
            if (quantizedEmbeddings_.type == GGML_TYPE_Q2_K &&
                !quantizedEmbeddings_.data.empty()) {
                constexpr uint32_t kBlockSize = 256;
                uint32_t typeSize = ggmlTypeSize(GGML_TYPE_Q2_K);
                uint32_t blocksPerRow = (hiddenSize + kBlockSize - 1) / kBlockSize;
                uint64_t rowStride = static_cast<uint64_t>(blocksPerRow) * typeSize;
                // Hoist the SIMD dispatch out of the per-vocab-row inner loop (P3).
                auto dotQ2K = dotProductQ2_K_get();
                for (uint32_t s = logitStart; s < logitStart + logitCount; ++s) {
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    ThreadPool::instance().parallelFor(0, vocabSize, [&](uint32_t i) {
                        const uint8_t *embRow =
                                quantizedEmbeddings_.data.data() + static_cast<uint64_t>(i) * rowStride;
                        // Software prefetch the next row's weight data to hide DRAM
                        // latency on the large embedding matrix (vocabSize × hiddenSize).
                        if (i + 1 < vocabSize) {
                            prefetchRow(embRow + rowStride);
                        }
                        double dot = 0.0;
                        for (uint32_t b = 0; b < blocksPerRow; ++b) {
                            dot += dotQ2K(embRow + static_cast<uint64_t>(b) * typeSize,
                                          hPtr + static_cast<uint64_t>(b) * kBlockSize);
                        }
                        logitRow[i] = static_cast<float>(dot);
                    });
                }
            } else if (!dequantizedEmbeddings_.dataQ8K.empty()) {
                uint32_t blocksPerRow = (hiddenSize + 255) / 256;
                const Q8KBlock *q8kData = dequantizedEmbeddings_.dataQ8K.data();
                // Hoist the SIMD dispatch out of the per-vocab-row inner loop
                // (BENCHMARK_REPORT §4.6 / P3): resolve the implementation pointer
                // once per token instead of doing an atomic acquire load on every
                // block of every vocab row.
                auto dotQ8K = dotProductQ8K_Q8K_get();
                // Quantize each token's hidden vector to Q8_K once (reused across all
                // vocab rows), then parallelize over vocab. This avoids re-quantizing
                // the hidden vector for every (token, vocab) pair.
                for (uint32_t s = logitStart; s < logitStart + logitCount; ++s) {
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    // P7: reusable grow-only scratch so steady-state prefill
                    // performs no heap allocation here. SHARED buffer: written by
                    // the calling thread, read by workers in the parallelFor below.
                    static std::vector<Q8KBlock> q8Hidden;
                    q8Hidden.resize(blocksPerRow);
                    GGMLDequantize::quantizeQ8K(hPtr, hiddenSize, q8Hidden.data());
                    ThreadPool::instance().parallelFor(0, vocabSize, [&](uint32_t i) {
                        const Q8KBlock *embRow = q8kData + static_cast<uint64_t>(i) * blocksPerRow;
                        // Software prefetch the next row's weight data to hide DRAM
                        // latency on the large embedding matrix (vocabSize × hiddenSize).
                        if (i + 1 < vocabSize) {
                            prefetchRow(embRow + blocksPerRow);
                        }
                        double dot = 0.0;
                        for (uint32_t b = 0; b < blocksPerRow; ++b) {
                            dot += static_cast<double>(dotQ8K(&q8Hidden[b], &embRow[b]));
                        }
                        logitRow[i] = static_cast<float>(dot);
                    });
                }
            } else if (!dequantizedEmbeddings_.dataF16.empty()) {
                ThreadPool::instance().parallelFor(0, logitCount * vocabSize, [&](uint32_t flatIdx) {
                    uint32_t s = logitStart + flatIdx / vocabSize;
                    uint32_t i = flatIdx % vocabSize;
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    const uint16_t *embRow = dequantizedEmbeddings_.dataF16.data() + static_cast<uint64_t>(i) * hiddenSize;
                    logitRow[i] = dotProductFMA_F16(hPtr, embRow, hiddenSize);
                });
            } else if (!dequantizedEmbeddings_.empty()) {
                ThreadPool::instance().parallelFor(0, logitCount * vocabSize, [&](uint32_t flatIdx) {
                    uint32_t s = logitStart + flatIdx / vocabSize;
                    uint32_t i = flatIdx % vocabSize;
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    const float *embRow = dequantizedEmbeddings_.data.data() + static_cast<uint64_t>(i) * hiddenSize;
                    logitRow[i] = dotProductFMA(hPtr, embRow, hiddenSize);
                });
            } else {
                ThreadPool::instance().parallelFor(0, logitCount * vocabSize, [&](uint32_t flatIdx) {
                    uint32_t s = logitStart + flatIdx / vocabSize;
                    uint32_t i = flatIdx % vocabSize;
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    float dot = 0.0f;
                    uint32_t blockSize = ggmlBlockSize(quantizedEmbeddings_.type);
                    uint32_t typeSize = ggmlTypeSize(quantizedEmbeddings_.type);
                    uint32_t numBlocks = (hiddenSize + blockSize - 1) / blockSize;
                    for (uint32_t b = 0; b < numBlocks; ++b) {
                        uint64_t blockOffset = static_cast<uint64_t>(i) * numBlocks + b;
                        const uint8_t *blockData = quantizedEmbeddings_.data.data() + blockOffset * typeSize;
                        float blockOut[256];
                        GGMLDequantize::dequantizeBlock(quantizedEmbeddings_.type, blockData, blockOut, blockSize);
                        uint32_t start = b * blockSize;
                        uint32_t n = std::min(blockSize, hiddenSize - start);
                        dot += dotProductFMA(hPtr + start, blockOut, n);
                    }
                    logitRow[i] = dot;
                });
            }
#endif
        } else {
            // Separate LM head. For the single-token generation path (seqLen==1,
            // one logit row) with active top-K pruning, route the LM head through
            // the exact two-pass prune-then-compute path. This reads only the
            // candidate vocab rows' weight data from the large separate LM head
            // (~182 MB for Q6_K on Qwen2.5-Coder) instead of the whole matrix every
            // token. All other logit rows are set to the pruned sentinel; the
            // subsequent softmax maps them to zero probability, and the candidate
            // dots are bit-identical to the unpruned reference path.
            bool usePruning = pendingPruneActive_ && !lmHeadPruneUseless_ &&
                              seqLen == 1 && logitCount == 1 && logitStart == 0 &&
                              !lmHeadBounds_.empty() &&
                              lmHeadBoundsBlocksPerRow_ == (hiddenSize + 255) / 256;
            if (usePruning) {
                uint32_t cand = computeLogitsTopK(
                        lmHead_.data.data(), lmHead_.type, lmHeadBounds_,
                        lmHeadBoundsBlocksPerRow_, lmHeadBoundsSubgroupsPerBlock_,
                        vocabSize, hiddenSize, hiddenData, pendingPruneTopK_,
                        pendingForceInclude_, logitsData);
                // Self-disable P0 when it provides no pruning: if the candidate
                // set encompasses (almost) the whole vocab, the per-subgroup
                // [min,max] interval is too loose to shrink it (fundamental for
                // K-quants; see BENCHMARK_REPORT §7.5), so the 5-pass machinery
                // only adds overhead. Subsequent tokens then use the normal
                // parallel-for LM head path.
                if (cand * 2 > vocabSize) {
                    if (!lmHeadPruneUseless_) {
                        fprintf(stderr,
                                "[TinyCoder] P0 LM-head top-K pruning disabled: "
                                "candidate set covers %u/%u vocab rows (bounds too "
                                "loose for this quant). Falling back to full LM head.\n",
                                cand, vocabSize);
                    }
                    lmHeadPruneUseless_ = true;
                }
            } else if (lmHead_.type == GGML_TYPE_Q6_K && hiddenSize % 256 == 0) {
                // Q6_K raw LM head kernel (0.82 B/elem vs 1.14 B/elem for the
                // Q8_K copy below) — ~28% less per-token LM-head weight traffic,
                // matching llama.cpp's working set. The hidden vector is
                // quantized to Q8_K once per token inside the kernel (reused
                // across all vocab rows); the -32 offset of the 6-bit weights
                // folds through the Q8KBlock bsums (per-lane vector, Q3K pattern).
                for (uint32_t s = logitStart; s < logitStart + logitCount; ++s) {
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    matMulVecBatchQ6K_SIMD(lmHead_.data.data(), hPtr, 1,
                                           vocabSize, hiddenSize, logitRow);
                }
            } else if (!lmHeadQ8K_.empty()) {
                // Plan §3: pre-quantized Q8_K copy of the separate LM head. Quantize
                // each token's hidden vector to Q8_K once (reused across all vocab
                // rows), then parallelize over vocab rows with the int8
                // _mm256_maddubs_epi16 kernels — cuts per-token LM-head memory
                // traffic ~2x vs the on-the-fly dequantized fallback below.
                uint32_t blocksPerRow = (hiddenSize + 255) / 256;
                const Q8KBlock *q8kData = lmHeadQ8K_.data();
                auto dotQ8K = dotProductQ8K_Q8K_get();
                for (uint32_t s = logitStart; s < logitStart + logitCount; ++s) {
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    // P7: reusable grow-only scratch (written by the calling
                    // thread, read by workers) — no per-token heap allocation.
                    static std::vector<Q8KBlock> q8Hidden;
                    q8Hidden.resize(blocksPerRow);
                    GGMLDequantize::quantizeQ8K(hPtr, hiddenSize, q8Hidden.data());
                    ThreadPool::instance().parallelFor(0, vocabSize, [&](uint32_t i) {
                        const Q8KBlock *embRow = q8kData + static_cast<uint64_t>(i) * blocksPerRow;
                        if (i + 1 < vocabSize) {
                            prefetchRow(embRow + blocksPerRow);
                        }
                        double dot = 0.0;
                        for (uint32_t b = 0; b < blocksPerRow; ++b) {
                            dot += static_cast<double>(dotQ8K(&q8Hidden[b], &embRow[b]));
                        }
                        logitRow[i] = static_cast<float>(dot);
                    });
                }
            } else {
                // Consolidated single parallelFor over the requested (token, vocab) pairs
                ThreadPool::instance().parallelFor(0, logitCount * vocabSize, [&](uint32_t flatIdx) {
                    uint32_t s = logitStart + flatIdx / vocabSize;
                    uint32_t i = flatIdx % vocabSize;
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    float dot = 0.0f;
                    uint32_t blockSize = ggmlBlockSize(lmHead_.type);
                    uint32_t typeSize = ggmlTypeSize(lmHead_.type);
                    uint32_t numBlocks = (hiddenSize + blockSize - 1) / blockSize;
                    for (uint32_t b = 0; b < numBlocks; ++b) {
                        uint64_t blockOffset = static_cast<uint64_t>(i) * numBlocks + b;
                        const uint8_t *blockData = lmHead_.data.data() + blockOffset * typeSize;
                        float blockOut[256];
                        GGMLDequantize::dequantizeBlock(lmHead_.type, blockData, blockOut, blockSize);
                        uint32_t start = b * blockSize;
                        uint32_t n = std::min(blockSize, hiddenSize - start);
                        dot += dotProductFMA(hPtr + start, blockOut, n);
                    }
                    logitRow[i] = dot;
                });
            }
        }

        // Apply final logit softcapping (Gemma4 architecture)
        if (config_.architecture == ARCH_GEMMA4 && config_.finalLogitSoftcapping > 0.0f) {
            float cap = config_.finalLogitSoftcapping;
            float invCap = 1.0f / cap;
            for (uint32_t s = 0; s < seqLen; ++s) {
                float *logitRow = logitsData + s * vocabSize;
                for (uint32_t i = 0; i < vocabSize; ++i) {
                    logitRow[i] = std::tanh(logitRow[i] * invCap) * cap;
                }
            }
        }

        // Update KV cache position
        kvCache_.pos += seqLen;

        // Consume any pending one-shot P0 pruning request now that the core
        // forward has run, so it never leaks into a subsequent forward call.
        pendingPruneActive_ = false;
        pendingPruneTopK_ = 0;
        pendingForceInclude_ = nullptr;

        return logits;
    }

    uint32_t Model::computeLogitsTopK(const uint8_t *data, uint32_t type,
                                      const std::vector<float> &bounds,
                                      uint32_t blocksPerRow, uint32_t subgroupsPerBlock,
                                      uint32_t vocabSize, uint32_t hiddenSize,
                                      const float *hidden, int32_t pruneTopK,
                                      const std::vector<int32_t> *forceInclude,
                                      float *logitsOut) {
        if (pruneTopK <= 0 || vocabSize == 0 || blocksPerRow == 0) {
            return 0;
        }
        uint32_t k = static_cast<uint32_t>(pruneTopK);
        if (k > vocabSize) {
            k = vocabSize;
        }

        uint32_t bSize = ggmlBlockSize(type);
        uint32_t tSize = ggmlTypeSize(type);
        if (subgroupsPerBlock == 0) {
            subgroupsPerBlock = 1;
        }
        constexpr uint32_t SCG = 16;
        uint32_t subgroupStride = blocksPerRow * subgroupsPerBlock;// per-row subgroups

        // Per-subgroup sum of positive and negative hidden elements. A subgroup is
        // a 16-element contiguous run. Must match the exact column boundaries used
        // in the reference (unpruned) dot product so the bounds tightly enclose the
        // true logit.
        topKScratchSumPos_.resize(subgroupStride);
        topKScratchSumNeg_.resize(subgroupStride);
        float *sgSumPos = topKScratchSumPos_.data();
        float *sgSumNeg = topKScratchSumNeg_.data();
        for (uint32_t sg = 0; sg < subgroupStride; ++sg) {
            uint32_t start = sg * SCG;
            uint32_t n = std::min(SCG, hiddenSize - start);
            float sp = 0.0f, sn = 0.0f;
            for (uint32_t j = 0; j < n; ++j) {
                float xj = hidden[start + j];
                if (xj >= 0.0f) {
                    sp += xj;
                } else {
                    sn += xj;
                }
            }
            sgSumPos[sg] = sp;
            sgSumNeg[sg] = sn;
        }

        // h = number of force-include (repeat-penalty history) tokens that must
        // always have exact logits regardless of the top-K threshold. Build a
        // compact mark list to avoid an O(vocabSize * h) force-include scan later.
        uint32_t h = 0;
        std::vector<uint32_t> forcedIds;
        if (forceInclude) {
            for (int32_t t: *forceInclude) {
                if (t >= 0 && static_cast<uint32_t>(t) < vocabSize) {
                    ++h;
                    forcedIds.push_back(static_cast<uint32_t>(t));
                }
            }
        }
        uint32_t targetCount = std::min<uint32_t>(k + h, vocabSize);

        // Pass 1: compute per-row lower/upper logit bounds from per-subgroup
        // (min,max). For each subgroup sg: hi += mx*sgSumPos + mn*sgSumNeg;
        // lo += mn*sgSumPos + mx*sgSumNeg. Store [row*2+0]=lower and
        // [row*2+1]=upper contiguously so pass 3 does not re-read bounds metadata
        // nor recompute. Parallelized over rows (read-only bounds, disjoint writes).
        topKScratchBounds_.resize(static_cast<size_t>(vocabSize) * 2);
        float *boundsRow2 = topKScratchBounds_.data();
        const float *browBase = bounds.data();
        ThreadPool::instance().parallelFor(0, vocabSize, [&](uint32_t i) {
            const float *rb = browBase + static_cast<size_t>(i) * subgroupStride * 2;
            float lo = 0.0f, hi = 0.0f;
            for (uint32_t sg = 0; sg < subgroupStride; ++sg) {
                float mn = rb[sg * 2 + 0];
                float mx = rb[sg * 2 + 1];
                float sp = sgSumPos[sg];
                float sn = sgSumNeg[sg];
                hi += mx * sp + mn * sn;
                lo += mn * sp + mx * sn;
            }
            boundsRow2[i * 2 + 0] = lo;
            boundsRow2[i * 2 + 1] = hi;
        });
        const float *lowerB = topKScratchBounds_.data();// stride-2 (lower)

        // Pass 2: find the targetCount-th largest lower bound (theta).
        topKScratchSort_.resize(vocabSize);
        float *sortBuf = topKScratchSort_.data();
        for (uint32_t i = 0; i < vocabSize; ++i) {
            sortBuf[i] = lowerB[i * 2 + 0];
        }
        std::nth_element(sortBuf, sortBuf + (targetCount - 1), sortBuf + vocabSize,
                         std::greater<float>());
        float theta = sortBuf[targetCount - 1];

        // Pass 3: collect candidate rows (those whose upper bound >= theta), plus
        // all force-include rows, into topKCandidates_.
        topKCandidates_.resize(vocabSize);
        uint32_t candCount = 0;
        for (uint32_t i = 0; i < vocabSize; ++i) {
            float hi = boundsRow2[i * 2 + 1];
            if (hi >= theta) {
                topKCandidates_[candCount++] = i;
            }
        }
        // Append force-include rows if not already present.
        for (uint32_t fid: forcedIds) {
            bool already = false;
            for (uint32_t c = 0; c < candCount; ++c) {
                if (topKCandidates_[c] == fid) {
                    already = true;
                    break;
                }
            }
            if (!already) {
                topKCandidates_[candCount++] = fid;
            }
        }

        // Pass 4: compute exact logits only for candidates (bit-identical to the
        // reference path: dequantizeBlock + dotProductFMA per block). All other
        // rows are set to the pruned sentinel (-> zero softmax probability).
        // Parallelized over candidates via the thread pool; each task writes a
        // distinct logitsOut[i].
        std::fill(logitsOut, logitsOut + vocabSize, kPrunedLogitSentinel);
        ThreadPool::instance().parallelFor(0, candCount, [&](uint32_t c) {
            uint32_t i = topKCandidates_[c];
            float dot = 0.0f;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                uint64_t blockOffset = static_cast<uint64_t>(i) * blocksPerRow + b;
                const uint8_t *blockData = data + blockOffset * tSize;
                float blockOut[256];
                GGMLDequantize::dequantizeBlock(type, blockData, blockOut, bSize);
                uint32_t start = b * bSize;
                uint32_t n = std::min(bSize, hiddenSize - start);
                dot += dotProductFMA(hidden + start, blockOut, n);
            }
            logitsOut[i] = dot;
        });

        return candCount;
    }

    np::Array<float> Model::forward(const std::vector<int32_t> &tokens,
                                    bool computeAllLogits, int32_t pruneTopK,
                                    const std::vector<int32_t> *forceInclude) {
        if (pruneTopK <= 0) {
            return forward(tokens, computeAllLogits);
        }
        // Arm the one-shot pruning request for the core single-token forward.
        pendingPruneTopK_ = pruneTopK;
        pendingForceInclude_ = forceInclude;
        pendingPruneActive_ = true;
        np::Array<float> result = forward(tokens, computeAllLogits);
        // The core forward consumes (clears) the pending flags before returning.
        return result;
    }

}// namespace tinycoder