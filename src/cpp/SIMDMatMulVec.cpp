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

#include "SIMDMatMulVec.hpp"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

// Include np library's CPU feature detection
#include <np/internal/CpuDispatch.hpp>

// ============================================================================
// Scalar (no SIMD) implementations — always available
// ============================================================================
namespace {

    void accumulateFMA_Scalar(float *local, const float *blockOut, float alpha,
                              uint32_t n) {
        for (uint32_t j = 0; j < n; ++j) {
            local[j] += alpha * blockOut[j];
        }
    }

    float dotProductFMA_Scalar(const float *hidden, const float *blockOut,
                               uint32_t n) {
        double sum = 0.0;
        for (uint32_t j = 0; j < n; ++j) {
            sum += static_cast<double>(hidden[j]) * blockOut[j];
        }
        return static_cast<float>(sum);
    }

    // ---- Scalar F16 dot product ----
    // Computes dot(hidden, W) where W is stored as FP16 (uint16_t).
    // Converts FP16 to float on-the-fly.
    float dotProductFMA_F16_Scalar(const float *hidden, const uint16_t *W_f16,
                                   uint32_t n) {
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
        double sum = 0.0;
        for (uint32_t j = 0; j < n; ++j) {
            sum += static_cast<double>(hidden[j]) * halfToFloat(W_f16[j]);
        }
        return static_cast<float>(sum);
    }

    void rmsNorm_Scalar(const float *x, float *out, const float *weight,
                        uint32_t n, float eps) {
        double sumSq = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            sumSq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        }
        float invRms = 1.0f / (std::sqrt(static_cast<float>(sumSq / static_cast<double>(n))) + eps);
        for (uint32_t i = 0; i < n; ++i) {
            out[i] = x[i] * invRms * weight[i];
        }
    }

    void softmax_Scalar(float *x, uint32_t n) {
        float maxVal = -std::numeric_limits<float>::infinity();
        for (uint32_t i = 0; i < n; ++i) {
            if (x[i] > maxVal) maxVal = x[i];
        }
        double sumExp = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            sumExp += static_cast<double>(std::exp(static_cast<double>(x[i] - maxVal)));
        }
        float invSum = static_cast<float>(1.0 / sumExp);
        for (uint32_t i = 0; i < n; ++i) {
            x[i] = static_cast<float>(std::exp(static_cast<double>(x[i] - maxVal))) * invSum;
        }
    }

    void silu_Scalar(float *x, uint32_t n) {
        for (uint32_t i = 0; i < n; ++i) {
            x[i] = x[i] / (1.0f + std::exp(-x[i]));
        }
    }

    void swiGLU_Scalar(float *x, const float *y, uint32_t n) {
        for (uint32_t i = 0; i < n; ++i) {
            float siluVal = x[i] / (1.0f + std::exp(-x[i]));
            x[i] = siluVal * y[i];
        }
    }

    void add_Scalar(float *x, const float *y, uint32_t n) {
        for (uint32_t i = 0; i < n; ++i) {
            x[i] += y[i];
        }
    }

    void scale_Scalar(float *x, float alpha, uint32_t n) {
        for (uint32_t i = 0; i < n; ++i) {
            x[i] *= alpha;
        }
    }

    // ---- Scalar Q2_K fused dot product ----
    float dotProductQ2_K_Scalar(const uint8_t *blockData, const float *x) {
        // Q2_K block layout (84 bytes):
        //   Offset 0-15:   scales[16] (16 bytes, 4-bit quantized scales and mins)
        //   Offset 16-79:  qs[64]   (64 bytes, 2-bit quantized data)
        //   Offset 80-81:  d        (2 bytes, fp16 - super-block scale)
        //   Offset 82-83:  dmin     (2 bytes, fp16 - super-block min)
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

        float d = halfToFloat(*(const uint16_t *) (blockData + 80));
        float dmin = halfToFloat(*(const uint16_t *) (blockData + 82));
        const uint8_t *scales = blockData;
        const uint8_t *q = blockData + 16;

        double dot = 0.0;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                double sum_xq = 0.0, sum_x = 0.0;
                for (int l = 0; l < 16; ++l) {
                    float xv = x[n + j * 32 + l];
                    int8_t quant = (int8_t) ((q[l] >> shift) & 3);
                    sum_xq += static_cast<double>(xv) * quant;
                    sum_x += static_cast<double>(xv);
                }
                dot += static_cast<double>(dl) * sum_xq - static_cast<double>(ml) * sum_x;

                sc = scales[is++];
                dl = d * (sc & 0xF);
                ml = dmin * (sc >> 4);
                sum_xq = 0.0;
                sum_x = 0.0;
                for (int l = 0; l < 16; ++l) {
                    float xv = x[n + j * 32 + 16 + l];
                    int8_t quant = (int8_t) ((q[l + 16] >> shift) & 3);
                    sum_xq += static_cast<double>(xv) * quant;
                    sum_x += static_cast<double>(xv);
                }
                dot += static_cast<double>(dl) * sum_xq - static_cast<double>(ml) * sum_x;

                shift += 2;
            }
            q += 32;
        }
        return static_cast<float>(dot);
    }

    // ---- Scalar pre-packed Q2_K fused dot product ----
    // Pre-packed blocks have qs_expanded[256] (bytes 0-3) in element order,
    // eliminating the 2-bit extraction overhead.
    float dotProductQ2_K_PrePacked_Scalar(const uint8_t *prepackedBlock, const float *x) {
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

        float d = halfToFloat(*(const uint16_t *) (prepackedBlock + 16));
        float dmin = halfToFloat(*(const uint16_t *) (prepackedBlock + 18));
        const uint8_t *scales = prepackedBlock;
        const uint8_t *qs_expanded = prepackedBlock + 20;

        double dot = 0.0;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                double sum_xq = 0.0, sum_x = 0.0;
                int base = n + j * 32;
                for (int l = 0; l < 16; ++l) {
                    sum_xq += static_cast<double>(x[base + l]) * qs_expanded[l];
                    sum_x += static_cast<double>(x[base + l]);
                }
                dot += static_cast<double>(dl) * sum_xq - static_cast<double>(ml) * sum_x;

                sc = scales[is++];
                dl = d * (sc & 0xF);
                ml = dmin * (sc >> 4);
                sum_xq = 0.0;
                sum_x = 0.0;
                for (int l = 0; l < 16; ++l) {
                    sum_xq += static_cast<double>(x[base + 16 + l]) * qs_expanded[16 + l];
                    sum_x += static_cast<double>(x[base + 16 + l]);
                }
                dot += static_cast<double>(dl) * sum_xq - static_cast<double>(ml) * sum_x;
            }
            qs_expanded += 128;
        }
        return static_cast<float>(dot);
    }

}// anonymous namespace

// ============================================================================
// AVX2 implementations (compiled with -mavx2 -mfma)
// ============================================================================
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

namespace {

    void accumulateFMA_AVX2(float *local, const float *blockOut, float alpha,
                            uint32_t n) {
        __m256 alphaVec = _mm256_set1_ps(alpha);
        uint32_t j = 0;
        for (; j + 8 <= n; j += 8) {
            __m256 blockVec = _mm256_loadu_ps(blockOut + j);
            __m256 localVec = _mm256_loadu_ps(local + j);
            __m256 result = _mm256_fmadd_ps(alphaVec, blockVec, localVec);
            _mm256_storeu_ps(local + j, result);
        }
        for (; j < n; ++j) {
            local[j] += alpha * blockOut[j];
        }
    }

    float dotProductFMA_AVX2(const float *hidden, const float *blockOut,
                             uint32_t n) {
        __m256 sumVec = _mm256_setzero_ps();
        uint32_t j = 0;
        for (; j + 8 <= n; j += 8) {
            __m256 hVec = _mm256_loadu_ps(hidden + j);
            __m256 bVec = _mm256_loadu_ps(blockOut + j);
            sumVec = _mm256_fmadd_ps(hVec, bVec, sumVec);
        }
        __m128 hi = _mm256_extractf128_ps(sumVec, 1);
        __m128 lo = _mm256_castps256_ps128(sumVec);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float result = _mm_cvtss_f32(sum128);
        for (; j < n; ++j) {
            result += hidden[j] * blockOut[j];
        }
        return result;
    }

    // ---- AVX2 F16 dot product ----
    // Computes dot(hidden, W) where W is stored as FP16 (uint16_t).
    // Uses F16C _mm256_cvtph_ps to convert 8 FP16 values to FP32 in one instruction.
    // -mavx2 implies -mf16c on GCC/Clang, so this intrinsic is always available
    // when AVX2 is enabled.
    float dotProductFMA_F16_AVX2(const float *hidden, const uint16_t *W_f16,
                                 uint32_t n) {
        __m256 sumVec = _mm256_setzero_ps();
        uint32_t j = 0;
        for (; j + 8 <= n; j += 8) {
            // Load 8 FP16 values and convert to FP32 using F16C
            __m128i f16_lo = _mm_loadu_si128((const __m128i *) (W_f16 + j));
            __m256 wVec = _mm256_cvtph_ps(f16_lo);
            __m256 hVec = _mm256_loadu_ps(hidden + j);
            sumVec = _mm256_fmadd_ps(hVec, wVec, sumVec);
        }
        __m128 hi = _mm256_extractf128_ps(sumVec, 1);
        __m128 lo = _mm256_castps256_ps128(sumVec);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float result = _mm_cvtss_f32(sum128);
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

    void rmsNorm_AVX2(const float *x, float *out, const float *weight,
                      uint32_t n, float eps) {
        // Compute sum of squares using double-precision accumulation
        __m256 sumSqVec = _mm256_setzero_ps();
        uint32_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 xVec = _mm256_loadu_ps(x + i);
            sumSqVec = _mm256_fmadd_ps(xVec, xVec, sumSqVec);
        }
        // Horizontal add
        __m128 hi = _mm256_extractf128_ps(sumSqVec, 1);
        __m128 lo = _mm256_castps256_ps128(sumSqVec);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        double sumSq = static_cast<double>(_mm_cvtss_f32(sum128));
        for (; i < n; ++i) {
            sumSq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        }
        float invRms = 1.0f / (std::sqrt(static_cast<float>(sumSq / static_cast<double>(n))) + eps);

        __m256 invRmsVec = _mm256_set1_ps(invRms);
        i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 xVec = _mm256_loadu_ps(x + i);
            __m256 wVec = _mm256_loadu_ps(weight + i);
            __m256 outVec = _mm256_mul_ps(_mm256_mul_ps(xVec, invRmsVec), wVec);
            _mm256_storeu_ps(out + i, outVec);
        }
        for (; i < n; ++i) {
            out[i] = x[i] * invRms * weight[i];
        }
    }

    void softmax_AVX2(float *x, uint32_t n) {
        // Find max using AVX2
        __m256 maxVec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        uint32_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 xVec = _mm256_loadu_ps(x + i);
            maxVec = _mm256_max_ps(maxVec, xVec);
        }
        // Horizontal max of 8-wide vector
        __m128 lo = _mm256_castps256_ps128(maxVec);
        __m128 hi = _mm256_extractf128_ps(maxVec, 1);
        __m128 max128 = _mm_max_ps(lo, hi);
        max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, 0x55));
        max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, 0xAA));
        max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, 0xFF));
        float maxVal = _mm_cvtss_f32(max128);
        // Scalar tail
        for (; i < n; ++i) {
            if (x[i] > maxVal) maxVal = x[i];
        }

        // Compute exp(x - maxVal) and sum, store exp in-place
        __m256 maxBroadcast = _mm256_set1_ps(maxVal);
        __m256 sumVec = _mm256_setzero_ps();
        i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 xVec = _mm256_loadu_ps(x + i);
            __m256 sub = _mm256_sub_ps(xVec, maxBroadcast);
            // Store to temporary buffer for scalar exp
            float buf[8];
            _mm256_storeu_ps(buf, sub);
            float expBuf[8];
            for (int k = 0; k < 8; ++k) expBuf[k] = std::exp(buf[k]);
            __m256 expVec = _mm256_loadu_ps(expBuf);
            _mm256_storeu_ps(x + i, expVec);
            sumVec = _mm256_add_ps(sumVec, expVec);
        }
        // Horizontal sum of 8-wide vector
        __m128 sumLo = _mm256_castps256_ps128(sumVec);
        __m128 sumHi = _mm256_extractf128_ps(sumVec, 1);
        __m128 sum128 = _mm_add_ps(sumLo, sumHi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        double sumExp = static_cast<double>(_mm_cvtss_f32(sum128));
        // Scalar tail
        for (; i < n; ++i) {
            float ev = std::exp(x[i] - maxVal);
            x[i] = ev;
            sumExp += static_cast<double>(ev);
        }

        // Normalize
        float invSum = static_cast<float>(1.0 / sumExp);
        __m256 invBroadcast = _mm256_set1_ps(invSum);
        i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 val = _mm256_loadu_ps(x + i);
            _mm256_storeu_ps(x + i, _mm256_mul_ps(val, invBroadcast));
        }
        for (; i < n; ++i) {
            x[i] *= invSum;
        }
    }

    void silu_AVX2(float *x, uint32_t n) {
        uint32_t i = 0;
        for (; i + 8 <= n; i += 8) {
            float vals[8];
            _mm256_storeu_ps(vals, _mm256_loadu_ps(x + i));
            for (int k = 0; k < 8; ++k) {
                vals[k] = vals[k] / (1.0f + std::exp(-vals[k]));
            }
            _mm256_storeu_ps(x + i, _mm256_loadu_ps(vals));
        }
        for (; i < n; ++i) {
            x[i] = x[i] / (1.0f + std::exp(-x[i]));
        }
    }

    void swiGLU_AVX2(float *x, const float *y, uint32_t n) {
        uint32_t i = 0;
        for (; i + 8 <= n; i += 8) {
            float xVals[8], yVals[8];
            _mm256_storeu_ps(xVals, _mm256_loadu_ps(x + i));
            _mm256_storeu_ps(yVals, _mm256_loadu_ps(y + i));
            for (int k = 0; k < 8; ++k) {
                float siluVal = xVals[k] / (1.0f + std::exp(-xVals[k]));
                xVals[k] = siluVal * yVals[k];
            }
            _mm256_storeu_ps(x + i, _mm256_loadu_ps(xVals));
        }
        for (; i < n; ++i) {
            float siluVal = x[i] / (1.0f + std::exp(-x[i]));
            x[i] = siluVal * y[i];
        }
    }

    void add_AVX2(float *x, const float *y, uint32_t n) {
        uint32_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 xVec = _mm256_loadu_ps(x + i);
            __m256 yVec = _mm256_loadu_ps(y + i);
            _mm256_storeu_ps(x + i, _mm256_add_ps(xVec, yVec));
        }
        for (; i < n; ++i) {
            x[i] += y[i];
        }
    }

    void scale_AVX2(float *x, float alpha, uint32_t n) {
        __m256 alphaVec = _mm256_set1_ps(alpha);
        uint32_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 xVec = _mm256_loadu_ps(x + i);
            _mm256_storeu_ps(x + i, _mm256_mul_ps(xVec, alphaVec));
        }
        for (; i < n; ++i) {
            x[i] *= alpha;
        }
    }

    // ---- AVX2 Q2_K fused dot product ----
    // Computes dot(x, dequantize(blockData)) for one Q2_K block (256 weights, 84 bytes).
    //
    // Q2_K block layout:
    //   Offset 0-15:   scales[16] (16 bytes, 4-bit quantized scales and mins)
    //   Offset 16-79:  qs[64]   (64 bytes, 2-bit quantized data)
    //   Offset 80-81:  d        (2 bytes, fp16 - super-block scale)
    //   Offset 82-83:  dmin     (2 bytes, fp16 - super-block min)
    //
    // Strategy:
    //   For each group of 16 weights sharing the same (scale, min):
    //     1. Load 16 bytes of q data, extract 2-bit values via 32-bit shift+mask
    //     2. Convert extracted values to float
    //     3. Compute val = dl * quant - ml (exact float computation)
    //     4. Accumulate acc += x * val (exact float computation via FMA)
    //
    // This processes 8 elements per iteration using full float precision.
    static float dotProductQ2_K_AVX2(const uint8_t *blockData, const float *x) {
        // Helper: convert IEEE 754 half-precision (16-bit) to float
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

        uint16_t d_half, dmin_half;
        std::memcpy(&d_half, blockData + 80, sizeof(uint16_t));
        std::memcpy(&dmin_half, blockData + 82, sizeof(uint16_t));
        float d = halfToFloat(d_half);
        float dmin = halfToFloat(dmin_half);

        const uint8_t *scales = blockData;
        const uint8_t *q = blockData + 16;

        __m256 acc = _mm256_setzero_ps();
        int is = 0;

        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                // ---- Group 0: q[0..15] ----
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);

                // Load 16 bytes of q data
                __m128i q16 = _mm_loadu_si128((const __m128i *) (q));

                // Lower 8 bytes -> 8 x 32-bit ints
                __m256i q_lo = _mm256_cvtepu8_epi32(q16);
                // Variable shift right by `shift` bits (0,2,4,6)
                q_lo = _mm256_srlv_epi32(q_lo, _mm256_set1_epi32(shift));
                // Mask to 2 bits
                q_lo = _mm256_and_si256(q_lo, _mm256_set1_epi32(3));
                __m256 q_lo_f = _mm256_cvtepi32_ps(q_lo);

                // Upper 8 bytes -> 8 x 32-bit ints
                __m128i q16_hi = _mm_srli_si128(q16, 8);
                __m256i q_hi = _mm256_cvtepu8_epi32(q16_hi);
                q_hi = _mm256_srlv_epi32(q_hi, _mm256_set1_epi32(shift));
                q_hi = _mm256_and_si256(q_hi, _mm256_set1_epi32(3));
                __m256 q_hi_f = _mm256_cvtepi32_ps(q_hi);

                // Load x values for this group
                int base = n + j * 32;
                __m256 x_lo = _mm256_loadu_ps(x + base);
                __m256 x_hi = _mm256_loadu_ps(x + base + 8);

                __m256 dl_vec = _mm256_set1_ps(dl);
                __m256 ml_vec = _mm256_set1_ps(ml);

                // val = dl * quant - ml
                __m256 val_lo = _mm256_fmsub_ps(dl_vec, q_lo_f, ml_vec);
                __m256 val_hi = _mm256_fmsub_ps(dl_vec, q_hi_f, ml_vec);

                // acc += x * val
                acc = _mm256_fmadd_ps(x_lo, val_lo, acc);
                acc = _mm256_fmadd_ps(x_hi, val_hi, acc);

                // ---- Group 1: q[16..31] ----
                sc = scales[is++];
                dl = d * (sc & 0xF);
                ml = dmin * (sc >> 4);

                __m128i q16_2 = _mm_loadu_si128((const __m128i *) (q + 16));

                __m256i q2_lo = _mm256_cvtepu8_epi32(q16_2);
                q2_lo = _mm256_srlv_epi32(q2_lo, _mm256_set1_epi32(shift));
                q2_lo = _mm256_and_si256(q2_lo, _mm256_set1_epi32(3));
                __m256 q2_lo_f = _mm256_cvtepi32_ps(q2_lo);

                __m128i q16_2_hi = _mm_srli_si128(q16_2, 8);
                __m256i q2_hi = _mm256_cvtepu8_epi32(q16_2_hi);
                q2_hi = _mm256_srlv_epi32(q2_hi, _mm256_set1_epi32(shift));
                q2_hi = _mm256_and_si256(q2_hi, _mm256_set1_epi32(3));
                __m256 q2_hi_f = _mm256_cvtepi32_ps(q2_hi);

                __m256 x2_lo = _mm256_loadu_ps(x + base + 16);
                __m256 x2_hi = _mm256_loadu_ps(x + base + 24);

                dl_vec = _mm256_set1_ps(dl);
                ml_vec = _mm256_set1_ps(ml);

                __m256 val2_lo = _mm256_fmsub_ps(dl_vec, q2_lo_f, ml_vec);
                __m256 val2_hi = _mm256_fmsub_ps(dl_vec, q2_hi_f, ml_vec);

                acc = _mm256_fmadd_ps(x2_lo, val2_lo, acc);
                acc = _mm256_fmadd_ps(x2_hi, val2_hi, acc);

                shift += 2;
            }
            q += 32;
        }

        // Horizontal sum of 8-wide accumulator
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        return _mm_cvtss_f32(sum128);
    }

    // ---- AVX2 pre-packed Q2_K fused dot product (float precision) ----
    // Pre-packed blocks have qs_expanded[256] (bytes 0-3) in element order,
    // eliminating the 2-bit extraction overhead entirely.
    //
    // Pre-packed block format (276 bytes):
    //   Offset 0-15:   scales[16]   (copied from original)
    //   Offset 16-17:  d            (fp16, copied from original)
    //   Offset 18-19:  dmin         (fp16, copied from original)
    //   Offset 20-275: qs_expanded[256] (each byte is 0-3, in element order)
    //
    // Compared to dotProductQ2_K_AVX2, this kernel:
    //   - Eliminates the shift/mask/expand/pack sequence (10 instructions per group)
    //   - Loads pre-expanded byte values directly with _mm_loadu_si128
    //   - Saves ~80 instructions per block
    //   - Uses full float precision (no maddubs int8 rounding)
    static float dotProductQ2_K_AVX2_PrePacked(const uint8_t *prepackedBlock, const float *x) {
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

        uint16_t d_half, dmin_half;
        std::memcpy(&d_half, prepackedBlock + 16, sizeof(uint16_t));
        std::memcpy(&dmin_half, prepackedBlock + 18, sizeof(uint16_t));
        float d = halfToFloat(d_half);
        float dmin = halfToFloat(dmin_half);

        const uint8_t *scales = prepackedBlock;
        const uint8_t *qs_expanded = prepackedBlock + 20;

        __m256 acc = _mm256_setzero_ps();
        int is = 0;

        for (int n = 0; n < 256; n += 128) {
            for (int j = 0; j < 4; ++j) {
                // ---- Group 0: qs_expanded[0..15] ----
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);

                // Load 16 pre-expanded byte values (0-3) directly — no bit extraction needed!
                __m128i q16 = _mm_loadu_si128((const __m128i *) (qs_expanded));

                // Expand to 32-bit ints, convert to float
                __m256i q_lo = _mm256_cvtepu8_epi32(q16);
                __m256i q_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(q16, 8));
                __m256 q_lo_f = _mm256_cvtepi32_ps(q_lo);
                __m256 q_hi_f = _mm256_cvtepi32_ps(q_hi);

                // Load 16 x values for this group
                int base = n + j * 32;
                __m256 x_lo = _mm256_loadu_ps(x + base);
                __m256 x_hi = _mm256_loadu_ps(x + base + 8);

                __m256 dl_vec = _mm256_set1_ps(dl);
                __m256 ml_vec = _mm256_set1_ps(ml);

                // val = dl * quant - ml (exact float computation)
                __m256 val_lo = _mm256_fmsub_ps(dl_vec, q_lo_f, ml_vec);
                __m256 val_hi = _mm256_fmsub_ps(dl_vec, q_hi_f, ml_vec);

                // acc += x * val
                acc = _mm256_fmadd_ps(x_lo, val_lo, acc);
                acc = _mm256_fmadd_ps(x_hi, val_hi, acc);

                // ---- Group 1: qs_expanded[16..31] ----
                sc = scales[is++];
                dl = d * (sc & 0xF);
                ml = dmin * (sc >> 4);

                // Load next 16 pre-expanded byte values directly
                __m128i q16_2 = _mm_loadu_si128((const __m128i *) (qs_expanded + 16));

                __m256i q2_lo = _mm256_cvtepu8_epi32(q16_2);
                __m256i q2_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(q16_2, 8));
                __m256 q2_lo_f = _mm256_cvtepi32_ps(q2_lo);
                __m256 q2_hi_f = _mm256_cvtepi32_ps(q2_hi);

                __m256 x2_lo = _mm256_loadu_ps(x + base + 16);
                __m256 x2_hi = _mm256_loadu_ps(x + base + 24);

                dl_vec = _mm256_set1_ps(dl);
                ml_vec = _mm256_set1_ps(ml);

                __m256 val2_lo = _mm256_fmsub_ps(dl_vec, q2_lo_f, ml_vec);
                __m256 val2_hi = _mm256_fmsub_ps(dl_vec, q2_hi_f, ml_vec);

                acc = _mm256_fmadd_ps(x2_lo, val2_lo, acc);
                acc = _mm256_fmadd_ps(x2_hi, val2_hi, acc);

                qs_expanded += 32;
            }
        }

        // Horizontal sum of 8-wide accumulator
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        return _mm_cvtss_f32(sum128);
    }

}// anonymous namespace
#endif// __AVX2__ && __FMA__

// ============================================================================
// AVX-512 implementations
// ============================================================================
#if defined(__AVX512F__) && defined(__FMA__)
#include <immintrin.h>

namespace {

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

}// anonymous namespace
#endif// __AVX512F__ && __FMA__

// ============================================================================
// Runtime dispatch — selects the best implementation once and caches it
// ============================================================================

namespace tinycoder {

    // ---- accumulateFMA ----
    void accumulateFMA(float *local, const float *blockOut, float alpha,
                       uint32_t n) {
        static std::atomic<void (*)(float *, const float *, float, uint32_t)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX512F__) && defined(__FMA__)
                case SimdLevel::AVX512:
                    impl = accumulateFMA_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = accumulateFMA_AVX2;
                    break;
#endif
                default:
                    impl = accumulateFMA_Scalar;
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        impl(local, blockOut, alpha, n);
    }

    // ---- dotProductFMA ----
    float dotProductFMA(const float *hidden, const float *blockOut, uint32_t n) {
        static std::atomic<float (*)(const float *, const float *, uint32_t)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX512F__) && defined(__FMA__)
                case SimdLevel::AVX512:
                    impl = dotProductFMA_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = dotProductFMA_AVX2;
                    break;
#endif
                default:
                    impl = dotProductFMA_Scalar;
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        return impl(hidden, blockOut, n);
    }

    // ---- dotProductFMA_F16 ----
    // Computes dot(hidden, W) where W is stored as FP16 (uint16_t).
    // Uses F16C _mm256_cvtph_ps for on-the-fly FP16->FP32 conversion.
    float dotProductFMA_F16(const float *hidden, const uint16_t *W_f16, uint32_t n) {
        static std::atomic<float (*)(const float *, const uint16_t *, uint32_t)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = dotProductFMA_F16_AVX2;
                    break;
#endif
                default:
                    impl = dotProductFMA_F16_Scalar;
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        return impl(hidden, W_f16, n);
    }

    // ---- rmsNormSIMD ----
    void rmsNormSIMD(const float *x, float *out, const float *weight,
                     uint32_t n, float eps) {
        static std::atomic<void (*)(const float *, float *, const float *, uint32_t, float)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX512F__) && defined(__FMA__)
                case SimdLevel::AVX512:
                    impl = rmsNorm_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = rmsNorm_AVX2;
                    break;
#endif
                default:
                    impl = rmsNorm_Scalar;
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        impl(x, out, weight, n, eps);
    }

    // ---- softmaxSIMD ----
    void softmaxSIMD(float *x, uint32_t n) {
        static std::atomic<void (*)(float *, uint32_t)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX512F__) && defined(__FMA__)
                case SimdLevel::AVX512:
                    impl = softmax_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = softmax_AVX2;
                    break;
#endif
                default:
                    impl = softmax_Scalar;
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        impl(x, n);
    }

    // ---- siluSIMD ----
    void siluSIMD(float *x, uint32_t n) {
        static std::atomic<void (*)(float *, uint32_t)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX512F__) && defined(__FMA__)
                case SimdLevel::AVX512:
                    impl = silu_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = silu_AVX2;
                    break;
#endif
                default:
                    impl = silu_Scalar;
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        impl(x, n);
    }

    // ---- swiGLUSIMD ----
    void swiGLUSIMD(float *x, const float *y, uint32_t n) {
        static std::atomic<void (*)(float *, const float *, uint32_t)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX512F__) && defined(__FMA__)
                case SimdLevel::AVX512:
                    impl = swiGLU_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = swiGLU_AVX2;
                    break;
#endif
                default:
                    impl = swiGLU_Scalar;
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        impl(x, y, n);
    }

    // ---- addSIMD ----
    void addSIMD(float *x, const float *y, uint32_t n) {
        static std::atomic<void (*)(float *, const float *, uint32_t)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX512F__) && defined(__FMA__)
                case SimdLevel::AVX512:
                    impl = add_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = add_AVX2;
                    break;
#endif
                default:
                    impl = add_Scalar;
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        impl(x, y, n);
    }

    // ---- scaleSIMD ----
    void scaleSIMD(float *x, float alpha, uint32_t n) {
        static std::atomic<void (*)(float *, float, uint32_t)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX512F__) && defined(__FMA__)
                case SimdLevel::AVX512:
                    impl = scale_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = scale_AVX2;
                    break;
#endif
                default:
                    impl = scale_Scalar;
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        impl(x, alpha, n);
    }

    // ---- dotProductQ2_K_SIMD ----
    float dotProductQ2_K_SIMD(const uint8_t *blockData, const float *x) {
        static std::atomic<float (*)(const uint8_t *, const float *)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = dotProductQ2_K_AVX2;
                    break;
#endif
                default:
                    impl = dotProductQ2_K_Scalar;
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        return impl(blockData, x);
    }

    // ---- dotProductQ2_K_PrePacked_SIMD ----
    float dotProductQ2_K_PrePacked_SIMD(const uint8_t *prepackedBlock, const float *x) {
        static std::atomic<float (*)(const uint8_t *, const float *)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = dotProductQ2_K_AVX2_PrePacked;
                    break;
#endif
                default:
                    impl = dotProductQ2_K_PrePacked_Scalar;
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        return impl(prepackedBlock, x);
    }

}// namespace tinycoder
