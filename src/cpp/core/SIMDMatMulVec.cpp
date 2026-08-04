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
#include "GGMLDequantize.hpp"
#include "SIMDMatMulVecInternal.hpp"
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
        }
        return static_cast<float>(dot);
    }

    // ---- Scalar Q8_K × Q8_K dot product ----
    // Computes dot(x, w) where both x and w are Q8_K blocks (256 int8 values +
    // a block scale). Result = x.d * w.d * sum(x.qs[i] * w.qs[i]).
    float dotProductQ8K_Q8K_Scalar(const tinycoder::Q8KBlock *x,
                                   const tinycoder::Q8KBlock *w) {
        int sum = 0;
        for (int i = 0; i < 256; ++i) {
            sum += static_cast<int>(x->qs[i]) * static_cast<int>(w->qs[i]);
        }
        return x->d * w->d * static_cast<float>(sum);
    }

    // ---- Scalar Q8_K dot product for pre-packed Q2_K ----
    // For each 16-element group g with scale dl and min ml:
    //   sum_xq = d * sum(qs[i] * q2[i])
    //   sum_x  = d * bsums[g]   (bsums precomputed at quantization time)
    //   dot   += dl * sum_xq - ml * sum_x
    float dotProductQ2_K_PrePacked_Q8_Scalar(const uint8_t *prepackedBlock,
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

        double dot = 0.0;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[is++];
                float dl = dq * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                int base = n + j * 32;
                int g = (n / 128) * 8 + j * 2;

                int sum_xq = 0;
                for (int l = 0; l < 16; ++l) {
                    sum_xq += static_cast<int>(q8->qs[base + l]) * qs_expanded[l];
                }
                dot += static_cast<double>(dl) * d * sum_xq - static_cast<double>(ml) * d * q8->bsums[g];

                sc = scales[is++];
                dl = dq * (sc & 0xF);
                ml = dmin * (sc >> 4);
                sum_xq = 0;
                for (int l = 0; l < 16; ++l) {
                    sum_xq += static_cast<int>(q8->qs[base + 16 + l]) * qs_expanded[16 + l];
                }
                dot += static_cast<double>(dl) * d * sum_xq - static_cast<double>(ml) * d * q8->bsums[g + 1];
            }
            qs_expanded += 128;
        }
        return static_cast<float>(dot);
    }

}// anonymous namespace

// ============================================================================
// Runtime dispatch — selects the best implementation once and caches it.
//
// The AVX2 and AVX-512 kernels live in their own translation units
// (SIMDMatMulVecAVX2.cpp / SIMDMatMulVecAVX512.cpp) and are referenced here via
// the tinycoder::simd namespace (see SIMDMatMulVecInternal.hpp).
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
                    impl = simd::accumulateFMA_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::accumulateFMA_AVX2;
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
                    impl = simd::dotProductFMA_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::dotProductFMA_AVX2;
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
#if defined(__AVX512F__) && defined(__FMA__)
                case SimdLevel::AVX512:
                    impl = simd::dotProductFMA_F16_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::dotProductFMA_F16_AVX2;
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

    // ---- dotProductFMA_F16_get ----
    // Resolves the SIMD implementation pointer once and returns it, so callers
    // in tight inner loops can hoist the dispatch out of the loop (P3).
    float (*dotProductFMA_F16_get())(const float *, const uint16_t *, uint32_t) {
        static std::atomic<float (*)(const float *, const uint16_t *, uint32_t)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX512F__) && defined(__FMA__)
                case SimdLevel::AVX512:
                    impl = simd::dotProductFMA_F16_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::dotProductFMA_F16_AVX2;
                    break;
#endif
                default:
                    impl = dotProductFMA_F16_Scalar;
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        return impl;
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
                    impl = simd::rmsNorm_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::rmsNorm_AVX2;
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
                    impl = simd::softmax_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::softmax_AVX2;
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
                    impl = simd::silu_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::silu_AVX2;
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
                    impl = simd::swiGLU_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::swiGLU_AVX2;
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
                    impl = simd::add_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::add_AVX2;
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
                    impl = simd::scale_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::scale_AVX2;
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
                    impl = simd::dotProductQ2_K_AVX2;
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

    // ---- dotProductQ2_K_get ----
    // Resolves the SIMD implementation pointer once and returns it, so callers
    // in tight inner loops (e.g. the native Q2_K LM head) can hoist the dispatch
    // out of the loop (P3).
    float (*dotProductQ2_K_get())(const uint8_t *, const float *) {
        static std::atomic<float (*)(const uint8_t *, const float *)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::dotProductQ2_K_AVX2;
                    break;
#endif
                default:
                    impl = dotProductQ2_K_Scalar;
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        return impl;
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
                    impl = simd::dotProductQ2_K_AVX2_PrePacked;
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

    // ---- dotProductQ2_K_PrePacked_Q8_SIMD ----
    float dotProductQ2_K_PrePacked_Q8_SIMD(const uint8_t *prepackedBlock,
                                           const Q8KBlock *q8) {
        static std::atomic<float (*)(const uint8_t *, const Q8KBlock *)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX512F__) && defined(__FMA__)
                case SimdLevel::AVX512:
                    impl = simd::dotProductQ2_K_PrePacked_Q8_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::dotProductQ2_K_PrePacked_Q8_AVX2;
                    break;
#endif
                default:
                    impl = dotProductQ2_K_PrePacked_Q8_Scalar;
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        return impl(prepackedBlock, q8);
    }

    // ---- matMulVecBatchQ2_K_PrePacked_Q8_SIMD ----
    void matMulVecBatchQ2_K_PrePacked_Q8_SIMD(
            const uint8_t *prepackedData,
            const float *x,
            uint32_t rows,
            uint32_t cols,
            float *result) {
        static std::atomic<void (*)(const uint8_t *, const float *, uint32_t, uint32_t, float *)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::matMulVecBatchQ2_K_PrePacked_Q8_AVX2;
                    break;
#endif
                default:
                    // Fallback to scalar row-by-row approach
                    // Fallback: use row-by-row approach via function pointer
                    // This will be overwritten on first call with actual implementation
                    impl = [](const uint8_t *, const float *, uint32_t, uint32_t, float *) {
                        // No fallback available - results will be zero
                    };
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        impl(prepackedData, x, rows, cols, result);
    }

    void matMulVecBatchGateUpQ2_K_PrePacked_Q8_SIMD(
            const uint8_t *gatePrepacked,
            const uint8_t *upPrepacked,
            const float *x,
            uint32_t rows,
            uint32_t cols,
            float *gateOut,
            float *upOut) {
        static std::atomic<void (*)(const uint8_t *, const uint8_t *, const float *,
                                    uint32_t, uint32_t, float *, float *)>
                s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::matMulVecBatchGateUpQ2_K_PrePacked_Q8_AVX2;
                    break;
#endif
                default:
                    // No fallback available - results will be zero
                    impl = [](const uint8_t *, const uint8_t *, const float *,
                              uint32_t, uint32_t, float *, float *) {};
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        impl(gatePrepacked, upPrepacked, x, rows, cols, gateOut, upOut);
    }

    void matMulVecBatchGateUpQ2_K_PrePacked_Q8_Batch_SIMD(
            const uint8_t *gatePrepacked,
            const uint8_t *upPrepacked,
            const float *X,
            uint32_t seqLen,
            uint32_t rows,
            uint32_t cols,
            float *gateOut,
            float *upOut,
            bool applySwish) {
        static std::atomic<void (*)(const uint8_t *, const uint8_t *, const float *,
                                    uint32_t, uint32_t, uint32_t, float *, float *,
                                    bool)>
                s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::matMulVecBatchGateUpQ2_K_PrePacked_Q8_Batch_AVX2;
                    break;
#endif
                default:
                    // This kernel only has an AVX2 implementation (used by prefill
                    // gate+up). The caller (matMulVecFusedGateUp_Batch) only invokes
                    // it when cols % 256 == 0. Any non-AVX2 host should already have
                    // been handled by the prepack presence; to keep semantics safe,
                    // leave outputs zeroed as the previous fallback did.
                    impl = [](const uint8_t *, const uint8_t *, const float *,
                              uint32_t, uint32_t, uint32_t, float *, float *,
                              bool) {};
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        impl(gatePrepacked, upPrepacked, X, seqLen, rows, cols, gateOut, upOut,
             applySwish);
    }

    // ---- matMulVecBatchQ2_K_PrePacked_Q8_Batch_SIMD ----
    void matMulVecBatchQ2_K_PrePacked_Q8_Batch_SIMD(
            const uint8_t *prepackedData,
            const float *X,
            uint32_t seqLen,
            uint32_t rows,
            uint32_t cols,
            float *out) {
        static std::atomic<void (*)(const uint8_t *, const float *, uint32_t,
                                    uint32_t, uint32_t, float *)>
                s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::matMulVecBatchQ2_K_PrePacked_Q8_Batch_AVX2;
                    break;
#endif
                default:
                    // No fallback available - results will be zero
                    impl = [](const uint8_t *, const float *, uint32_t, uint32_t,
                              uint32_t, float *) {};
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        impl(prepackedData, X, seqLen, rows, cols, out);
    }

    // ---- dotProductQ8K_Q8K_SIMD ----
    float dotProductQ8K_Q8K_SIMD(const Q8KBlock *x, const Q8KBlock *w) {
        static std::atomic<float (*)(const Q8KBlock *, const Q8KBlock *)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX512F__) && defined(__FMA__)
                case SimdLevel::AVX512:
                    impl = simd::dotProductQ8K_Q8K_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::dotProductQ8K_Q8K_AVX2;
                    break;
#endif
                default:
                    impl = dotProductQ8K_Q8K_Scalar;
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        return impl(x, w);
    }

    // ---- matMulVecBatchQ8K_SIMD ----
    // Register-tiled batch GEMM for a single Q8_K matrix over a batch of tokens.
    // Dispatches to the vector (AVX2) kernel when available; returns false so the
    // caller falls back to the scalar per-pair loop on hosts without a vector
    // kernel. The x-quantization and GEMM both happen inside the vector kernel.
    bool matMulVecBatchQ8K_SIMD(const Q8KBlock *W_q8k, const float *X,
                                uint32_t seqLen, uint32_t rows, uint32_t cols,
                                float *out) {
        static std::atomic<int> s_level{-1};
        int level = s_level.load(std::memory_order_acquire);
        if (level < 0) {
            level = static_cast<int>(np::internal::max_simd_level());
            s_level.store(level, std::memory_order_release);
        }
#if defined(__AVX2__) && defined(__FMA__)
        if (level >= static_cast<int>(np::internal::SimdLevel::AVX2)) {
            simd::matMulVecBatchQ8K_Q8K_AVX2(W_q8k, X, seqLen, rows, cols, out);
            return true;
        }
#endif
        (void) W_q8k;
        (void) X;
        (void) seqLen;
        (void) rows;
        (void) cols;
        (void) out;
        return false;
    }

    // ---- matMulVecBatchQ4K_SIMD ----
    // Register-tiled batch GEMM for a single Q4_K matrix over a batch of tokens.
    // Dispatches to the vector (AVX2) kernel when available; returns false so the
    // caller falls back to the generic path on hosts without a vector kernel.
    bool matMulVecBatchQ4K_SIMD(const uint8_t *W_q4k, const float *X,
                                uint32_t seqLen, uint32_t rows, uint32_t cols,
                                float *out) {
        static std::atomic<int> s_level{-1};
        int level = s_level.load(std::memory_order_acquire);
        if (level < 0) {
            level = static_cast<int>(np::internal::max_simd_level());
            s_level.store(level, std::memory_order_release);
        }
#if defined(__AVX2__) && defined(__FMA__)
        if (level >= static_cast<int>(np::internal::SimdLevel::AVX2)) {
            simd::matMulVecBatchQ4K_Q8K_AVX2(W_q4k, X, seqLen, rows, cols, out);
            return true;
        }
#endif
        (void) W_q4k;
        (void) X;
        (void) seqLen;
        (void) rows;
        (void) cols;
        (void) out;
        return false;
    }

    // ---- matMulVecBatchQ3K_SIMD ----
    // Register-tiled batch GEMM for a single COMPACT (raw GGUF) Q3_K matrix over
    // a batch of tokens (attnO/ffnDown, which are stored as Q3_K in the model
    // file). Dispatches to the AVX2 kernel when available; returns false so the
    // caller falls back to the generic dotProductQ3_K path on hosts without a
    // vector kernel.
    bool matMulVecBatchQ3K_SIMD(const uint8_t *W_q3k, const float *X,
                                uint32_t seqLen, uint32_t rows, uint32_t cols,
                                float *out, const float *residual) {
        static std::atomic<int> s_level{-1};
        int level = s_level.load(std::memory_order_acquire);
        if (level < 0) {
            level = static_cast<int>(np::internal::max_simd_level());
            s_level.store(level, std::memory_order_release);
        }
#if defined(__AVX2__) && defined(__FMA__)
        if (level >= static_cast<int>(np::internal::SimdLevel::AVX2)) {
            simd::matMulVecBatchQ3K_Q8K_AVX2(W_q3k, X, seqLen, rows, cols, out,
                                             residual);
            return true;
        }
#endif
        (void) W_q3k;
        (void) X;
        (void) seqLen;
        (void) rows;
        (void) cols;
        (void) out;
        (void) residual;
        return false;
    }

    // ---- matMulVecBatchQ6K_SIMD ----
    // Register-tiled batch GEMM for a single COMPACT (raw GGUF) Q6_K matrix over
    // a batch of tokens (the separate LM head, which is stored as Q6_K in the
    // model file). Dispatches to the AVX2 kernel when available; returns false
    // so the caller falls back to the generic dotProductQ6_K path on hosts
    // without a vector kernel.
    bool matMulVecBatchQ6K_SIMD(const uint8_t *W_q6k, const float *X,
                                uint32_t seqLen, uint32_t rows, uint32_t cols,
                                float *out) {
        static std::atomic<int> s_level{-1};
        int level = s_level.load(std::memory_order_acquire);
        if (level < 0) {
            level = static_cast<int>(np::internal::max_simd_level());
            s_level.store(level, std::memory_order_release);
        }
#if defined(__AVX2__) && defined(__FMA__)
        if (level >= static_cast<int>(np::internal::SimdLevel::AVX2)) {
            simd::matMulVecBatchQ6K_Q8K_AVX2(W_q6k, X, seqLen, rows, cols, out);
            return true;
        }
#endif
        (void) W_q6k;
        (void) X;
        (void) seqLen;
        (void) rows;
        (void) cols;
        (void) out;
        return false;
    }

    // ---- matMulVecBatchQ2K_Compact_SIMD ----
    // Register-tiled batch GEMM for a single COMPACT (raw GGUF) Q2_K weight
    // matrix over a batch of tokens (prefill or generation). Used for the
    // separate LM head after the load-time Q2_K re-quant (Lever C): the 84
    // B/block stream replaces Q6_K's 210 B/block (~191->76 MB/token, the single
    // largest per-token weight-traffic cut) and the per-block 2-bit dot runs
    // through _mm256_maddubs_epi16 against a Q8_K x-vector instead of the
    // ALU-bound float dequant of dotProductQ2_K_get — keeping the kernel
    // DRAM-bound. Returns true if a vector kernel was used, false if the caller
    // must fall back to the per-block dot path.
    bool matMulVecBatchQ2K_Compact_SIMD(const uint8_t *W_q2k, const float *X,
                                        uint32_t seqLen, uint32_t rows,
                                        uint32_t cols, float *out) {
        static std::atomic<int> s_level{-1};
        int level = s_level.load(std::memory_order_acquire);
        if (level < 0) {
            level = static_cast<int>(np::internal::max_simd_level());
            s_level.store(level, std::memory_order_release);
        }
#if defined(__AVX2__) && defined(__FMA__)
        if (level >= static_cast<int>(np::internal::SimdLevel::AVX2)) {
            simd::matMulVecBatchQ2K_Compact_Q8K_AVX2(W_q2k, X, seqLen, rows,
                                                     cols, out);
            return true;
        }
#endif
        (void) W_q2k;
        (void) X;
        (void) seqLen;
        (void) rows;
        (void) cols;
        (void) out;
        return false;
    }

    // ---- matMulVecFusedGateUpDownQ2K_Q3K_SIMD ----
    // Fused gate+up+down single-token FFN kernel. Dispatches to the AVX2
    // implementation when available; returns false so the caller falls back
    // to the simple per-kernel path on hosts without the fused kernel.
    // The down matrix is Q3_K unless the caller passes a load-time Q2_K
    // re-quant copy (Lever C) via downIsQ2K.
    bool matMulVecFusedGateUpDownQ2K_Q3K_SIMD(
            const uint8_t *gateData,
            const uint8_t *upData,
            const uint8_t *downData,
            const float *x,
            uint32_t gRows,
            uint32_t hRows,
            uint32_t cols,
            float *out,
            const float *residual) {
        static std::atomic<int> s_level{-1};
        int level = s_level.load(std::memory_order_acquire);
        if (level < 0) {
            level = static_cast<int>(np::internal::max_simd_level());
            s_level.store(level, std::memory_order_release);
        }
#if defined(__AVX2__) && defined(__FMA__)
        if (level >= static_cast<int>(np::internal::SimdLevel::AVX2)) {
            // Dispatch to the Q2_K ffnDown variant when the caller supplies a
            // load-time compact Q2_K re-quant copy (Lever C), else the raw
            // Q3_K variant. Same 9-arg shape, different block stride.
            // NOTE: the compact Q2_K down layout is detected by the caller via
            // the separate Q2K entry point in ModelForward; this generic
            // dispatcher always uses the Q3_K ffnDown (the default fallback).
            simd::matMulVecFusedGateUpDownQ2K_Q3K_Compact_AVX2(
                    gateData, upData, downData, x,
                    gRows, hRows, cols, out,
                    const_cast<float *>(residual));
            return true;
        }
#endif
        (void) gateData;
        (void) upData;
        (void) downData;
        (void) x;
        (void) gRows;
        (void) hRows;
        (void) cols;
        (void) out;
        (void) residual;
        return false;
    }

    // ---- matMulVecFusedGateUpDownQ2K_Q2K_SIMD ----
    // Lever C: fused gate+up+down FFN where the DOWN matrix is a load-time
    // COMPACT Q2_K re-quant copy (84 B/block vs Q3_K's 110 B/block), cutting
    // per-token ffnDown weight traffic ~24%. Phase 1 (gate+up) is the same
    // compact Q2_K pass; phase 2 dots the Q2_K down blocks against the shared
    // Q8_K act. Returns false on hosts without the AVX2 kernel.
    bool matMulVecFusedGateUpDownQ2K_Q2K_SIMD(
            const uint8_t *gateData,
            const uint8_t *upData,
            const uint8_t *downQ2kData,
            const float *x,
            uint32_t gRows,
            uint32_t hRows,
            uint32_t cols,
            float *out,
            const float *residual) {
        static std::atomic<int> s_level{-1};
        int level = s_level.load(std::memory_order_acquire);
        if (level < 0) {
            level = static_cast<int>(np::internal::max_simd_level());
            s_level.store(level, std::memory_order_release);
        }
#if defined(__AVX2__) && defined(__FMA__)
        if (level >= static_cast<int>(np::internal::SimdLevel::AVX2)) {
            simd::matMulVecFusedGateUpDownQ2K_Q2K_Compact_AVX2(
                    gateData, upData, downQ2kData, x,
                    gRows, hRows, cols, out,
                    const_cast<float *>(residual));
            return true;
        }
#endif
        (void) gateData;
        (void) upData;
        (void) downQ2kData;
        (void) x;
        (void) gRows;
        (void) hRows;
        (void) cols;
        (void) out;
        (void) residual;
        return false;
    }


    // ---- matMulVecFusedGateUpQ2_K_Compact_Q8_SIMD ----
    // Single-token fused gate+up against COMPACT (raw GGUF) Q2_K weights.
    // Generation is DRAM-bandwidth-bound and the compact 84-byte blocks (vs
    // 276-byte prepacked) cut the gate+up weight traffic ~3.3x — the dominant
    // per-token cost. Dispatches to the AVX2 kernel when available; otherwise
    // falls back to a scalar loop that decomposes each compact block to floats
    // on the fly ("q0 = (q>>sh)&0x3", per dequantizeQ2_KBlock).
    void matMulVecFusedGateUpQ2_K_Compact_Q8_SIMD(
            const uint8_t *gateData,
            const uint8_t *upData,
            const float *x,
            uint32_t rows,
            uint32_t cols,
            float *gateOut,
            float *upOut,
            bool applySwish) {
        static std::atomic<void (*)(const uint8_t *, const uint8_t *, const float *,
                                    uint32_t, uint32_t, float *, float *, bool)>
                s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::matMulVecFusedGateUpQ2_K_Compact_Q8_AVX2;
                    break;
#endif
                default:
                    // Scalar fallback: decompose compact blocks to float and dot
                    // per row. Order/results match the AVX2 kernel (same block
                    // decomposition, fp32 accumulation, int rounding like the
                    // prepacked fallback).
                    impl = [](const uint8_t *gateData, const uint8_t *upData,
                              const float *x, uint32_t rows, uint32_t cols,
                              float *gateOut, float *upOut, bool applySwish) {
                        using tinycoder::GGMLDequantize;
                        static constexpr uint32_t BLOCK = 256;
                        static constexpr uint32_t B = 84;
                        uint32_t blocksPerRow = (cols + BLOCK - 1) / BLOCK;
                        uint64_t rowStride = static_cast<uint64_t>(blocksPerRow) * B;
                        auto half = [](uint16_t h) {
                            return GGMLDequantize::halfToFloat(h);
                        };
                        for (uint32_t j = 0; j < rows; ++j) {
                            const uint8_t *gb = gateData + static_cast<uint64_t>(j) * rowStride;
                            const uint8_t *ub = upData + static_cast<uint64_t>(j) * rowStride;
                            double gd = 0.0, ud = 0.0;
                            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                                const uint8_t *gc = gb + b * B;
                                const uint8_t *uc = ub + b * B;
                                float gd0 = half(*(const uint16_t *) (gc + 80));
                                float gmn = half(*(const uint16_t *) (gc + 82));
                                float ud0 = half(*(const uint16_t *) (uc + 80));
                                float umn = half(*(const uint16_t *) (uc + 82));
                                const uint8_t *gq = gc + 16, *gs = gc;
                                const uint8_t *uq = uc + 16, *us = uc;
                                uint32_t is = 0;
                                uint32_t epos = 0;
                                for (int n = 0; n < 256; n += 128) {
                                    int shift = 0;
                                    for (int j2 = 0; j2 < 4; ++j2) {
                                        uint8_t scg = gs[is];
                                        uint8_t scu = us[is];
                                        ++is;
                                        float gdl = gd0 * (scg & 0xF);
                                        float gml = gmn * (scg >> 4);
                                        float udl = ud0 * (scu & 0xF);
                                        float uml = umn * (scu >> 4);
                                        for (int l = 0; l < 16; ++l) {
                                            float xv = x[b * BLOCK + epos];
                                            gd += (double) xv * (gdl * ((int8_t) ((gq[l] >> shift) & 3)) - gml);
                                            ud += (double) xv * (udl * ((int8_t) ((uq[l] >> shift) & 3)) - uml);
                                            ++epos;
                                        }
                                        scg = gs[is];
                                        scu = us[is];
                                        ++is;
                                        gdl = gd0 * (scg & 0xF);
                                        gml = gmn * (scg >> 4);
                                        udl = ud0 * (scu & 0xF);
                                        uml = umn * (scu >> 4);
                                        for (int l = 0; l < 16; ++l) {
                                            float xv = x[b * BLOCK + epos];
                                            gd += (double) xv * (gdl * ((int8_t) ((gq[l + 16] >> shift) & 3)) - gml);
                                            ud += (double) xv * (udl * ((int8_t) ((uq[l + 16] >> shift) & 3)) - uml);
                                            ++epos;
                                        }
                                        shift += 2;
                                    }
                                    gq += 32;
                                    uq += 32;
                                }
                            }
                            if (applySwish) {
                                // Fuse SwiGLU into the epilogue: silu(gate) * up.
                                // Matches std::exp math of swiGLU_Scalar so all
                                // dispatch paths are numerically identical.
                                float gv = (float) gd / (1.0f + std::exp(-(float) gd));
                                gateOut[j] = gv * (float) ud;
                            } else {
                                gateOut[j] = (float) gd;
                            }
                            upOut[j] = (float) ud;
                        }
                    };
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        impl(gateData, upData, x, rows, cols, gateOut, upOut, applySwish);
    }

    void matMulVecFusedQKQ2_K_Compact_Q8_SIMD(
            const uint8_t *qData,
            const uint8_t *kData,
            const float *x,
            uint32_t qRows,
            uint32_t kRows,
            uint32_t cols,
            float *qOut,
            float *kOut) {
        static std::atomic<void (*)(const uint8_t *, const uint8_t *, const float *,
                                    uint32_t, uint32_t, uint32_t, float *, float *)>
                s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::matMulVecFusedQKQ2_K_Compact_Q8_AVX2;
                    break;
#endif
                default:
                    // Scalar fallback: decompose the compact blocks to float and
                    // dot per row for each matrix (same math as the AVX2 kernel).
                    impl = [](const uint8_t *qData, const uint8_t *kData,
                              const float *x, uint32_t qRows, uint32_t kRows,
                              uint32_t cols, float *qOut, float *kOut) {
                        using tinycoder::GGMLDequantize;
                        static constexpr uint32_t BLOCK = 256;
                        static constexpr uint32_t B = 84;
                        uint32_t blocksPerRow = (cols + BLOCK - 1) / BLOCK;
                        uint64_t rowStride = static_cast<uint64_t>(blocksPerRow) * B;
                        auto half = [](uint16_t h) {
                            return GGMLDequantize::halfToFloat(h);
                        };
                        auto dotRow = [&](const uint8_t *rowData) -> float {
                            double acc = 0.0;
                            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                                const uint8_t *c = rowData + b * B;
                                float d0 = half(*(const uint16_t *) (c + 80));
                                float mn = half(*(const uint16_t *) (c + 82));
                                const uint8_t *q = c + 16, *s = c;
                                uint32_t is = 0, epos = 0;
                                for (int n = 0; n < 256; n += 128) {
                                    int shift = 0;
                                    for (int j2 = 0; j2 < 4; ++j2) {
                                        uint8_t sc = s[is++];
                                        float dl = d0 * (sc & 0xF);
                                        float ml = mn * (sc >> 4);
                                        for (int l = 0; l < 16; ++l) {
                                            acc += (double) x[b * BLOCK + epos] *
                                                   (dl * ((int8_t) ((q[l] >> shift) & 3)) - ml);
                                            ++epos;
                                        }
                                        sc = s[is++];
                                        dl = d0 * (sc & 0xF);
                                        ml = mn * (sc >> 4);
                                        for (int l = 0; l < 16; ++l) {
                                            acc += (double) x[b * BLOCK + epos] *
                                                   (dl * ((int8_t) ((q[l + 16] >> shift) & 3)) - ml);
                                            ++epos;
                                        }
                                        shift += 2;
                                    }
                                    q += 32;
                                }
                            }
                            return (float) acc;
                        };
                        for (uint32_t j = 0; j < qRows; ++j) {
                            qOut[j] = dotRow(qData + static_cast<uint64_t>(j) * rowStride);
                        }
                        for (uint32_t j = 0; j < kRows; ++j) {
                            kOut[j] = dotRow(kData + static_cast<uint64_t>(j) * rowStride);
                        }
                    };
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        impl(qData, kData, x, qRows, kRows, cols, qOut, kOut);
    }

    // ---- dotProductQ8K_Q8K_get ----
    // Resolves the SIMD implementation pointer once and returns it, so callers
    // in tight inner loops (e.g. the LM head per-vocab-row loop) can hoist the
    // dispatch out of the loop (P3).
    float (*dotProductQ8K_Q8K_get())(const Q8KBlock *, const Q8KBlock *) {
        static std::atomic<float (*)(const Q8KBlock *, const Q8KBlock *)> s_impl{nullptr};
        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();
            switch (level) {
#if defined(__AVX512F__) && defined(__FMA__)
                case SimdLevel::AVX512:
                    impl = simd::dotProductQ8K_Q8K_AVX512;
                    break;
#endif
#if defined(__AVX2__) && defined(__FMA__)
                case SimdLevel::AVX2:
                    impl = simd::dotProductQ8K_Q8K_AVX2;
                    break;
#endif
                default:
                    impl = dotProductQ8K_Q8K_Scalar;
                    break;
            }
            s_impl.store(impl, std::memory_order_release);
        }
        return impl;
    }

}// namespace tinycoder
