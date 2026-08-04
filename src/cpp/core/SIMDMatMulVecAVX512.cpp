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

// ============================================================================
// AVX-512 implementations of the SIMD mat-vec kernels.
//
// This translation unit is compiled with -mavx512f -mfma (see CMakeLists.txt)
// so the AVX-512/FMA intrinsics below are always available here. The kernels
// are declared in SIMDMatMulVecInternal.hpp and referenced by the runtime
// dispatch in SIMDMatMulVec.cpp.
// ============================================================================

#include "SIMDMatMulVecInternal.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__AVX512F__) && defined(__FMA__)
#include <immintrin.h>

namespace tinycoder::simd {

    void accumulateFMA_AVX512(float *local, const float *blockOut, float alpha,
                              uint32_t n) {
        __m512 alphaVec = _mm512_set1_ps(alpha);
        uint32_t j = 0;
        for (; j + 16 <= n; j += 16) {
            __m512 blockVec = _mm512_loadu_ps(blockOut + j);
            __m512 localVec = _mm512_loadu_ps(local + j);
            __m512 result = _mm512_fmadd_ps(alphaVec, blockVec, localVec);
            _mm512_storeu_ps(local + j, result);
        }
        for (; j < n; ++j) {
            local[j] += alpha * blockOut[j];
        }
    }

    float dotProductFMA_AVX512(const float *hidden, const float *blockOut,
                               uint32_t n) {
        __m512 sumVec = _mm512_setzero_ps();
        uint32_t j = 0;
        for (; j + 16 <= n; j += 16) {
            __m512 hVec = _mm512_loadu_ps(hidden + j);
            __m512 bVec = _mm512_loadu_ps(blockOut + j);
            sumVec = _mm512_fmadd_ps(hVec, bVec, sumVec);
        }
        float result = _mm512_reduce_add_ps(sumVec);
        for (; j < n; ++j) {
            result += hidden[j] * blockOut[j];
        }
        return result;
    }

    void rmsNorm_AVX512(const float *x, float *out, const float *weight,
                        uint32_t n, float eps) {
        __m512 sumSqVec = _mm512_setzero_ps();
        uint32_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 xVec = _mm512_loadu_ps(x + i);
            sumSqVec = _mm512_fmadd_ps(xVec, xVec, sumSqVec);
        }
        double sumSq = static_cast<double>(_mm512_reduce_add_ps(sumSqVec));
        for (; i < n; ++i) {
            sumSq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        }
        float invRms = 1.0f / (std::sqrt(static_cast<float>(sumSq / static_cast<double>(n))) + eps);

        __m512 invRmsVec = _mm512_set1_ps(invRms);
        i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 xVec = _mm512_loadu_ps(x + i);
            __m512 wVec = _mm512_loadu_ps(weight + i);
            _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_mul_ps(xVec, invRmsVec), wVec));
        }
        for (; i < n; ++i) {
            out[i] = x[i] * invRms * weight[i];
        }
    }

    void softmax_AVX512(float *x, uint32_t n) {
        // Use scalar for simplicity and accuracy
        float maxVal = -std::numeric_limits<float>::infinity();
        for (uint32_t i = 0; i < n; ++i) {
            if (x[i] > maxVal) maxVal = x[i];
        }
        __m512 maxVec = _mm512_set1_ps(maxVal);
        __m512 sumExpVec = _mm512_setzero_ps();
        uint32_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 xVec = _mm512_loadu_ps(x + i);
            __m512 subVec = _mm512_sub_ps(xVec, maxVec);
            // AVX-512 has no native exp, use scalar
            float vals[16];
            _mm512_storeu_ps(vals, subVec);
            for (int k = 0; k < 16; ++k) {
                vals[k] = std::exp(vals[k]);
            }
            __m512 expVec = _mm512_loadu_ps(vals);
            _mm512_storeu_ps(x + i, expVec);
            sumExpVec = _mm512_add_ps(sumExpVec, expVec);
        }
        double sumExp = static_cast<double>(_mm512_reduce_add_ps(sumExpVec));
        for (; i < n; ++i) {
            float ev = std::exp(x[i] - maxVal);
            x[i] = ev;
            sumExp += static_cast<double>(ev);
        }
        float invSum = static_cast<float>(1.0 / sumExp);
        __m512 invSumVec = _mm512_set1_ps(invSum);
        i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 xVec = _mm512_loadu_ps(x + i);
            _mm512_storeu_ps(x + i, _mm512_mul_ps(xVec, invSumVec));
        }
        for (; i < n; ++i) {
            x[i] *= invSum;
        }
    }

    void silu_AVX512(float *x, uint32_t n) {
        uint32_t i = 0;
        for (; i + 16 <= n; i += 16) {
            float vals[16];
            _mm512_storeu_ps(vals, _mm512_loadu_ps(x + i));
            for (int k = 0; k < 16; ++k) {
                vals[k] = vals[k] / (1.0f + std::exp(-vals[k]));
            }
            _mm512_storeu_ps(x + i, _mm512_loadu_ps(vals));
        }
        for (; i < n; ++i) {
            x[i] = x[i] / (1.0f + std::exp(-x[i]));
        }
    }

    void swiGLU_AVX512(float *x, const float *y, uint32_t n) {
        uint32_t i = 0;
        for (; i + 16 <= n; i += 16) {
            float xVals[16], yVals[16];
            _mm512_storeu_ps(xVals, _mm512_loadu_ps(x + i));
            _mm512_storeu_ps(yVals, _mm512_loadu_ps(y + i));
            for (int k = 0; k < 16; ++k) {
                float siluVal = xVals[k] / (1.0f + std::exp(-xVals[k]));
                xVals[k] = siluVal * yVals[k];
            }
            _mm512_storeu_ps(x + i, _mm512_loadu_ps(xVals));
        }
        for (; i < n; ++i) {
            float siluVal = x[i] / (1.0f + std::exp(-x[i]));
            x[i] = siluVal * y[i];
        }
    }

    void add_AVX512(float *x, const float *y, uint32_t n) {
        uint32_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 xVec = _mm512_loadu_ps(x + i);
            __m512 yVec = _mm512_loadu_ps(y + i);
            _mm512_storeu_ps(x + i, _mm512_add_ps(xVec, yVec));
        }
        for (; i < n; ++i) {
            x[i] += y[i];
        }
    }

    void scale_AVX512(float *x, float alpha, uint32_t n) {
        __m512 alphaVec = _mm512_set1_ps(alpha);
        uint32_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 xVec = _mm512_loadu_ps(x + i);
            _mm512_storeu_ps(x + i, _mm512_mul_ps(xVec, alphaVec));
        }
        for (; i < n; ++i) {
            x[i] *= alpha;
        }
    }

    // ---- AVX-512 F16 dot product ----
    // Computes dot(hidden, W) where W is stored as FP16 (uint16_t).
    // Uses F16C _mm512_cvtph_ps to convert 16 FP16 values to FP32 in one
    // instruction (vs 8 per instruction in the AVX2 kernel), doubling the
    // throughput of the memory-bound FFN-down / attnO mat-vecs.
    float dotProductFMA_F16_AVX512(const float *hidden, const uint16_t *W_f16,
                                   uint32_t n) {
        __m512 sumVec = _mm512_setzero_ps();
        uint32_t j = 0;
        for (; j + 16 <= n; j += 16) {
            // Load 16 FP16 values and convert to FP32 using F16C
            __m256i f16 = _mm256_loadu_si256((const __m256i *) (W_f16 + j));
            __m512 wVec = _mm512_cvtph_ps(f16);
            __m512 hVec = _mm512_loadu_ps(hidden + j);
            sumVec = _mm512_fmadd_ps(hVec, wVec, sumVec);
        }
        float result = _mm512_reduce_add_ps(sumVec);
        for (; j < n; ++j) {
            // Scalar tail: convert FP16 to float manually
            uint16_t h = W_f16[j];
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f32;
            if (exp == 0) {
                if (mant == 0) {
                    f32 = sign << 31;
                } else {
                    int nz = 0;
                    while ((mant & 0x200) == 0 && nz < 10) {
                        mant <<= 1;
                        nz++;
                    }
                    mant &= 0x3FF;
                    uint32_t mant_low = mant - 512;
                    exp = 112 - nz;
                    f32 = (sign << 31) | (exp << 23) | (mant_low << 14);
                }
            } else if (exp == 31) {
                f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
            } else {
                exp = exp + (127 - 15);
                f32 = (sign << 31) | (exp << 23) | (mant << 13);
            }
            float wVal;
            std::memcpy(&wVal, &f32, sizeof(float));
            result += hidden[j] * wVal;
        }
        return result;
    }

    // ---- AVX-512 Q8_K x Q8_K dot product ----
    // Computes dot(x, w) where both x and w are Q8_K blocks (256 int8 values +
    // a block scale). Uses _mm512_maddubs_epi16 (64 int8xint8->int16
    // multiply-adds per instruction) instead of 32 per instruction in the AVX2
    // kernel. This is the hot kernel of the LM head Q8_K path.
    float dotProductQ8K_Q8K_AVX512(const tinycoder::Q8KBlock *x,
                                   const tinycoder::Q8KBlock *w) {
        __m512i sumi = _mm512_setzero_si512();
        for (int i = 0; i < 256; i += 64) {
            // _mm512_maddubs_epi16 treats its FIRST operand as unsigned bytes
            // and the second as signed bytes. Both qs arrays are signed int8,
            // so raw signed values cannot go into the unsigned slot (a negative
            // byte would be reinterpreted as value+256). Substitute (x, w) with
            // (|x|, w*sign(x)): |x| is safe in the unsigned slot and
            // |x| * (w*sign(x)) == x*w exactly (same trick as llama.cpp).
            // Products are bounded by 127*127, so the int16 intermediate in
            // maddubs cannot saturate.
            __m512i xv = _mm512_loadu_si512((const __m512i *) &x->qs[i]);
            __m512i wv = _mm512_loadu_si512((const __m512i *) &w->qs[i]);
            __m512i xabs = _mm512_sign_epi8(xv, xv);
            __m512i wsig = _mm512_sign_epi8(wv, xv);
            __m512i p = _mm512_maddubs_epi16(xabs, wsig);
            sumi = _mm512_add_epi32(sumi, _mm512_madd_epi16(p, _mm512_set1_epi16(1)));
        }
        int sum = _mm512_reduce_add_epi32(sumi);
        return x->d * w->d * static_cast<float>(sum);
    }

    // ---- AVX-512 Q8_K dot product for pre-packed Q2_K ----
    // x is quantized to int8 (Q8_K) once per matmul. The dot product then uses
    // _mm512_maddubs_epi16 (64 int8xint8->int16 multiply-adds per instruction,
    // vs 32 in the AVX2 kernel). This is the hot kernel of the FFN gate+up
    // fused mat-vec.
    //
    // Pre-packed block layout (276 bytes):
    //   Offset 0-15:   scales[16]   (4-bit scale low, 4-bit min high)
    //   Offset 16-17:  d            (fp16)
    //   Offset 18-19:  dmin         (fp16)
    //   Offset 20-275: qs_expanded[256] (each byte is 0-3, in element order)
    //
    // For each 16-element group g with scale dl and min ml:
    //   sum_xq = d * sum(qs[i] * q2[i])   via maddubs
    //   sum_x  = d * bsums[g]             via madd (mins * bsums)
    //   dot   += dl * sum_xq - ml * sum_x
    float dotProductQ2_K_PrePacked_Q8_AVX512(const uint8_t *prepackedBlock,
                                             const tinycoder::Q8KBlock *q8) {
        auto halfToFloat = [](uint16_t h) -> float {
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f32;
            if (exp == 0) {
                if (mant == 0) {
                    f32 = sign << 31;
                } else {
                    int n = 0;
                    while ((mant & 0x200) == 0 && n < 10) {
                        mant <<= 1;
                        n++;
                    }
                    mant &= 0x3FF;
                    uint32_t mant_low = mant - 512;
                    exp = 112 - n;
                    f32 = (sign << 31) | (exp << 23) | (mant_low << 14);
                }
            } else if (exp == 31) {
                f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
            } else {
                exp = exp + (127 - 15);
                f32 = (sign << 31) | (exp << 23) | (mant << 13);
            }
            float result;
            std::memcpy(&result, &f32, sizeof(float));
            return result;
        };

        const float d = q8->d;
        const float dq = halfToFloat(*(const uint16_t *) (prepackedBlock + 16));
        const float dmin = halfToFloat(*(const uint16_t *) (prepackedBlock + 18));
        const uint8_t *scales = prepackedBlock;
        const uint8_t *qs_expanded = prepackedBlock + 20;

        __m512 acc = _mm512_setzero_ps();

        // Fold the min term: acc += -dmin * d * sum_g mins[g] * bsums[g].
        // mins are 4-bit (0-15), bsums are int16. Only 16 groups, so a scalar
        // loop is negligible compared to the 256-element main loop.
        {
            int minsSum = 0;
            for (int g = 0; g < 16; ++g) {
                uint8_t m = (scales[g] >> 4) & 0xF;
                minsSum += static_cast<int>(m) * q8->bsums[g];
            }
            acc = _mm512_fmadd_ps(_mm512_set1_ps(-dmin * d),
                                  _mm512_set1_ps(static_cast<float>(minsSum)), acc);
        }

        for (int n = 0; n < 256; n += 128) {
            const int chunk = n / 128;// 0 or 1
            __m512i sumi = _mm512_setzero_si512();

            // 2 sub-chunks of 64 elements each = 128 elements.
            // Each 64-element sub-chunk spans 4 groups of 16.
            //   chunk 0, sub 0 -> groups 0-3   (scales 0-3)
            //   chunk 0, sub 1 -> groups 4-7   (scales 4-7)
            //   chunk 1, sub 0 -> groups 8-11  (scales 8-11)
            //   chunk 1, sub 1 -> groups 12-15 (scales 12-15)
            for (int sub = 0; sub < 2; ++sub) {
                __m512i q2 = _mm512_loadu_si512((const __m512i *) (qs_expanded + sub * 64));
                __m512i q8v = _mm512_loadu_si512((const __m512i *) &q8->qs[n + sub * 64]);
                // q2 values are 0-3 (unsigned), q8 are int8 (signed).
                // maddubs treats the first operand as unsigned, second as signed.
                __m512i p = _mm512_maddubs_epi16(q2, q8v);

                // Build the scale vector for the 4 groups in this sub-chunk.
                // Each group's 16-bit scale is duplicated across 8 int16 lanes
                // (16 bytes). _mm512_set_epi16 lists args high-to-low, so the
                // lowest 8 int16 (bytes 0-15) = group 0 = s0.
                const int scaleBase = chunk * 8 + sub * 4;
                const int s0 = scales[scaleBase + 0] & 0xF;
                const int s1 = scales[scaleBase + 1] & 0xF;
                const int s2 = scales[scaleBase + 2] & 0xF;
                const int s3 = scales[scaleBase + 3] & 0xF;
                __m512i scales_v = _mm512_set_epi16(
                        s3, s3, s3, s3, s3, s3, s3, s3, // bytes 48-63 (group 3)
                        s2, s2, s2, s2, s2, s2, s2, s2, // bytes 32-47 (group 2)
                        s1, s1, s1, s1, s1, s1, s1, s1, // bytes 16-31 (group 1)
                        s0, s0, s0, s0, s0, s0, s0, s0);// bytes 0-15 (group 0)

                sumi = _mm512_add_epi32(sumi, _mm512_madd_epi16(scales_v, p));
            }

            // acc += dq * d * sumi
            acc = _mm512_fmadd_ps(_mm512_set1_ps(dq * d), _mm512_cvtepi32_ps(sumi), acc);

            qs_expanded += 128;
        }

        return _mm512_reduce_add_ps(acc);
    }

}// namespace tinycoder::simd
#endif// __AVX512F__ && __FMA__