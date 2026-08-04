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
// AVX2 implementations of the SIMD mat-vec kernels.
//
// This translation unit is compiled with -mavx2 -mfma -mf16c (see CMakeLists.txt)
// so the AVX2/FMA/F16C intrinsics below are always available here. The kernels
// are declared in SIMDMatMulVecInternal.hpp and referenced by the runtime
// dispatch in SIMDMatMulVec.cpp.
// ============================================================================

#include "GGMLDequantize.hpp"
#include "SIMDMatMulVecInternal.hpp"
#include "ThreadPool.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#include <vector>

namespace tinycoder::simd {

// Storing SIMD vector types (__m256 carries an `aligned(32)` attribute) as a
// std::vector template argument triggers GCC's benign `-Wignored-attributes`
// diagnostic: the compiler cannot statically verify the alignment attribute
// propagates through the template. The AlignedAllocator below provides correct
// 32-byte-aligned storage at runtime, so this warning is spurious here. GCC and
// Clang both accept this pragma.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

    /// @brief Minimal aligned allocator for `std::vector<__m256>`.
    ///
    /// `std::vector<__m256>` with the default allocator triggers GCC's
    /// `-Wignored-attributes` warning because `::operator new` does not honor
    /// `__m256`'s 32-byte alignment. This allocator backs the buffer with
    /// `posix_memalign(32)`, so every `_mm256_*` load/store in the batch GEMM
    /// kernels hits a 32-byte-aligned base and the warning is eliminated.
    template<typename T>
    class AlignedAllocator {
    public:
        using value_type = T;

        AlignedAllocator() = default;
        template<typename U>
        AlignedAllocator(const AlignedAllocator<U> &) noexcept {}

        T *allocate(std::size_t n) {
            if (n == 0) {
                return nullptr;
            }
            void *ptr = nullptr;
            if (posix_memalign(&ptr, alignof(T),
                               n * sizeof(T)) != 0 ||
                ptr == nullptr) {
                throw std::bad_alloc();
            }
            return static_cast<T *>(ptr);
        }

        void deallocate(T *p, std::size_t) noexcept {
            std::free(p);
        }

        template<typename U>
        struct rebind {
            using other = AlignedAllocator<U>;
        };

        template<typename U, typename... Args>
        void construct(U *p, Args &&...args) {
            ::new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
        }

        template<typename U>
        void destroy(U *p) noexcept {
            p->~U();
        }
    };

    template<typename T, typename U>
    inline bool operator==(const AlignedAllocator<T> &, const AlignedAllocator<U> &) noexcept {
        return true;
    }

    template<typename T, typename U>
    inline bool operator!=(const AlignedAllocator<T> &, const AlignedAllocator<U> &) noexcept {
        return false;
    }

    /// @brief A `std::vector<__m256>` with 32-byte-aligned storage.
    using Vector256 = std::vector<__m256, AlignedAllocator<__m256>>;

#pragma GCC diagnostic pop

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
        // Four independent FP32 accumulator chains break the serial FMA
        // dependency chain on the compute-bound single-token FP16 projections
        // (BENCHMARK_REPORT §9.2 item 1). The old single accumulator forced each
        // _mm256_fmadd_ps to wait on the previous one, serializing behind the
        // ~4-cycle FMA latency and the F16C _mm256_cvtph_ps conversion latency.
        // Four independent chains let the FMAs and vcvtph2ps conversions overlap
        // (instruction-level parallelism), the same technique ggml_fp16_mul_mat
        // uses. The 32-FP16-per-iteration body issues 4 loads + 4 converts +
        // 4 FMAs with no cross-iteration dependency.
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        uint32_t j = 0;
        for (; j + 32 <= n; j += 32) {
            acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(hidden + j),
                    _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (W_f16 + j))),
                    acc0);
            acc1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(hidden + j + 8),
                    _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (W_f16 + j + 8))),
                    acc1);
            acc2 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(hidden + j + 16),
                    _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (W_f16 + j + 16))),
                    acc2);
            acc3 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(hidden + j + 24),
                    _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (W_f16 + j + 24))),
                    acc3);
        }
        for (; j + 8 <= n; j += 8) {
            // Remainder: keep accumulating into acc0 (single chain).
            acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(hidden + j),
                    _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (W_f16 + j))),
                    acc0);
        }
        __m256 sumVec = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                      _mm256_add_ps(acc2, acc3));
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

        // Compute exp(x - maxVal) and sum, store exp in-place.
        // §5 Optimization: Vectorized exp polynomial approximation using AVX2.
        // Uses a Taylor series approximation: exp(x) ≈ sum_{k=0}^{7} x^k/k!
        // evaluated in SIMD for 8-wide vectors in the typical [-60, 0] softmax range.
        auto expVectorized_AVX2 = [](__m256 v) -> __m256 {
            // Minimax polynomial (7th order) on [-128, 127]:
            // exp(x) ≈ x*(a1 + x*(a2 + x*(a3 + x*(a4 + x*(a5 + x*(a6 + x*a7)))))) + 1
            const __m256 a1 = _mm256_set1_ps(1.0f);
            const __m256 a2 = _mm256_set1_ps(1.0f);
            const __m256 a3 = _mm256_set1_ps(0.5f);
            const __m256 a4 = _mm256_set1_ps(0.16666666666666666f);
            const __m256 a5 = _mm256_set1_ps(0.041666666666666664f);
            const __m256 a6 = _mm256_set1_ps(0.008333333333333333f);
            const __m256 a7 = _mm256_set1_ps(0.001388888888888889f);

            __m256 p = _mm256_set1_ps(1.0f);
            p = _mm256_fmadd_ps(v, a7, p);
            p = _mm256_fmadd_ps(v, a6, p);
            p = _mm256_fmadd_ps(v, a5, p);
            p = _mm256_fmadd_ps(v, a4, p);
            p = _mm256_fmadd_ps(v, a3, p);
            p = _mm256_fmadd_ps(v, a2, p);
            p = _mm256_fmadd_ps(v, a1, p);
            p = _mm256_fmadd_ps(v, _mm256_set1_ps(1.0f), p);
            return p;
        };

        __m256 maxBroadcast = _mm256_set1_ps(maxVal);
        __m256 sumVec = _mm256_setzero_ps();
        i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 xVec = _mm256_loadu_ps(x + i);
            __m256 sub = _mm256_sub_ps(xVec, maxBroadcast);
            __m256 expVec = expVectorized_AVX2(sub);
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
    float dotProductQ2_K_AVX2(const uint8_t *blockData, const float *x) {
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
    float dotProductQ2_K_AVX2_PrePacked(const uint8_t *prepackedBlock, const float *x) {
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

    // ---- AVX2 Q8_K dot product for pre-packed Q2_K ----
    // x is quantized to int8 (Q8_K) once per matmul. The dot product then uses
    // _mm256_maddubs_epi16, which computes 32 int8×int8->int16 multiply-adds per
    // instruction (vs 8 float FMAs per instruction in the float kernel).
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
    float dotProductQ2_K_PrePacked_Q8_AVX2(const uint8_t *prepackedBlock,
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

        __m256 acc = _mm256_setzero_ps();

        // Fold the min term: acc += -dmin * d * sum_g mins[g] * bsums[g].
        // mins are 4-bit (0-15), bsums are int16. madd_epi16 -> int32.
        {
            __m128i mins_and_scales = _mm_loadu_si128((const __m128i *) scales);
            // Extract high nibble of each byte: (scales >> 4) & 0xF
            __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), _mm_set1_epi8(0xF));
            // Sign-extend to 16-bit
            __m256i mins = _mm256_cvtepi8_epi16(mins8);
            __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i *) q8->bsums));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(-dmin * d), _mm256_cvtepi32_ps(prod), acc);
        }

        // Scale shuffle table (q3k pattern): for each 16-element group, duplicate
        // its 16-bit scale across 8 int16 lanes. scales_shuf holds 16-bit scales,
        // so each scale occupies 2 consecutive bytes (low, high).
        static const uint8_t k_shuffle[128] = {
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
        };

        // Optimization: pre-load all 4 shuffle vectors into registers before the
        // inner loop to avoid redundant _mm256_loadu_si256 in the inner loop.
        __m256i k_shuf[4] = {
                _mm256_loadu_si256((const __m256i *) (k_shuffle + 0)),
                _mm256_loadu_si256((const __m256i *) (k_shuffle + 32)),
                _mm256_loadu_si256((const __m256i *) (k_shuffle + 64)),
                _mm256_loadu_si256((const __m256i *) (k_shuffle + 96)),
        };

        // Load all 16 scale bytes once. scales_shuf[0] holds scales 0-7 (16-bit),
        // scales_shuf[1] holds scales 8-15.
        __m128i sc128 = _mm_loadu_si128((const __m128i *) scales);
        __m128i scales8 = _mm_and_si128(sc128, _mm_set1_epi8(0xF));
        __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
        __m128i slo = _mm256_extracti128_si256(scales16, 0);
        __m128i shi = _mm256_extracti128_si256(scales16, 1);
        __m256i scales_shuf[2] = {
                _mm256_set_m128i(slo, slo),
                _mm256_set_m128i(shi, shi),
        };

        for (int n = 0; n < 256; n += 128) {
            const int chunk = n / 128;// 0 or 1
            __m256i sumi = _mm256_setzero_si256();

            // 4 sub-chunks of 32 elements each = 128 elements.
            // sub 0 -> groups 0,1 (scales 0,1) -> k_shuffle + 0
            // sub 1 -> groups 2,3 (scales 2,3) -> k_shuffle + 32
            // sub 2 -> groups 4,5 (scales 4,5) -> k_shuffle + 64
            // sub 3 -> groups 6,7 (scales 6,7) -> k_shuffle + 96
            for (int sub = 0; sub < 4; ++sub) {
                __m256i q2 = _mm256_loadu_si256((const __m256i *) (qs_expanded + sub * 32));
                __m256i q8v = _mm256_loadu_si256((const __m256i *) &q8->qs[n + sub * 32]);
                // q2 values are 0-3 (unsigned), q8 are int8 (signed).
                // maddubs treats the first operand as unsigned, second as signed.
                __m256i p = _mm256_maddubs_epi16(q2, q8v);
                __m256i shuf = k_shuf[sub];
                p = _mm256_madd_epi16(_mm256_shuffle_epi8(scales_shuf[chunk], shuf), p);
                sumi = _mm256_add_epi32(sumi, p);
            }

            // acc += dq * d * sumi
            acc = _mm256_fmadd_ps(_mm256_set1_ps(dq * d), _mm256_cvtepi32_ps(sumi), acc);

            qs_expanded += 128;
        }

        // Horizontal sum of 8-wide accumulator
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        return _mm_cvtss_f32(sum128);
    }

    // ---- AVX2 register-tiled batch GEMM for Q2_K (8 rows at a time) ----
    //
    // This kernel processes BATCH_SIZE=8 rows in parallel using AVX2 registers.
    // For each tile of 8 rows, we accumulate into ymm registers across all
    // block columns before storing. This eliminates redundant reads of the
    // Q8_K x-vector data and keeps intermediate results in registers.
    //
    // The key optimization over the row-by-row approach:
    //   - Q8_K data is loaded once per block column and reused across 8 rows
    //   - Accumulators stay in ymm registers across all block columns
    //   - Reduced memory bandwidth for x-vector (Q8_K) by ~8x
    //
    void matMulVecBatchQ2_K_PrePacked_Q8_AVX2(
            const uint8_t *prepackedData,
            const float *x,
            uint32_t rows,
            uint32_t cols,
            float *result) {
        static constexpr uint32_t BATCH_SIZE = 8;
        static constexpr uint32_t BLOCK_SIZE = 256;
        static constexpr uint32_t PREPACKED_BLOCK_BYTES = 276;

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

        // Quantize x to Q8_K once
        // Copy x to a mutable buffer to handle partial blocks (zero-padding)
        uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::vector<float> xCopy(cols + blocksPerRow * BLOCK_SIZE - cols);
        std::memcpy(xCopy.data(), x, cols * sizeof(float));
        // Zero-pad remaining elements
        std::memset(xCopy.data() + cols, 0, (xCopy.size() - cols) * sizeof(float));

        std::vector<Q8KBlock> q8(blocksPerRow);
        {
            // Inline quantizeQ8K
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const float *xb = xCopy.data() + b * BLOCK_SIZE;
                Q8KBlock &blk = q8[b];
                float max = 0.0f;
                float amax = 0.0f;
                for (uint32_t j = 0; j < BLOCK_SIZE; ++j) {
                    float ax = std::fabs(xb[j]);
                    if (ax > amax) {
                        amax = ax;
                        max = xb[j];
                    }
                }
                if (amax == 0.0f) {
                    blk.d = 0.0f;
                    std::memset(blk.qs, 0, BLOCK_SIZE);
                    std::memset(blk.bsums, 0, sizeof(blk.bsums));
                    continue;
                }
                const float iscale = -127.0f / max;
                for (uint32_t j = 0; j < BLOCK_SIZE; ++j) {
                    int v = static_cast<int>(std::lrintf(iscale * xb[j]));
                    blk.qs[j] = static_cast<int8_t>(std::min(127, v));
                }
                for (uint32_t j = 0; j < BLOCK_SIZE / 16; ++j) {
                    int sum = 0;
                    for (uint32_t ii = 0; ii < 16; ++ii) {
                        sum += blk.qs[j * 16 + ii];
                    }
                    blk.bsums[j] = static_cast<int16_t>(sum);
                }
                blk.d = 1.0f / iscale;
            }
        }

        // Process rows in tiles of BATCH_SIZE, parallelized over tiles.
        // Each tile of 8 rows is independent; the Q8_K x-vector data (q8) is
        // shared read-only across threads.
        uint32_t numTiles = (rows + BATCH_SIZE - 1) / BATCH_SIZE;
        ThreadPool::instance().parallelForSlab(0, numTiles, [&](uint32_t tile) {
            uint32_t rowStart = tile * BATCH_SIZE;
            uint32_t batchSize = std::min(BATCH_SIZE, rows - rowStart);

            // Initialize accumulators for this batch
            // acc[8][8] = accumulators for 8 rows x 8 float lanes
            __m256 acc[8] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                             _mm256_setzero_ps(), _mm256_setzero_ps(),
                             _mm256_setzero_ps(), _mm256_setzero_ps(),
                             _mm256_setzero_ps(), _mm256_setzero_ps()};

            // Process each block column
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const Q8KBlock &q8Block = q8[b];
                const float d = q8Block.d;

                // Load Q8_K block into registers once per batch
                // We'll process it for all 8 rows
                __m256i q8_ymm[8];// 8 x 32-byte blocks = 256 elements
                for (int i = 0; i < 8; ++i) {
                    q8_ymm[i] = _mm256_loadu_si256((const __m256i *) (q8Block.qs + i * 32));
                }

                // Process each row in the batch
                for (uint32_t r = 0; r < batchSize; ++r) {
                    const uint8_t *rowData = prepackedData +
                                             static_cast<uint64_t>(rowStart + r) *
                                                     static_cast<uint64_t>(blocksPerRow) *
                                                     PREPACKED_BLOCK_BYTES;
                    const uint8_t *blockData = rowData + b * PREPACKED_BLOCK_BYTES;

                    const float dq = halfToFloat(*(const uint16_t *) (blockData + 16));
                    const float dmin = halfToFloat(*(const uint16_t *) (blockData + 18));
                    const uint8_t *scales = blockData;
                    const uint8_t *qs_expanded = blockData + 20;

                    // Compute min term: -dmin * d * sum(mins * bsums)
                    // NOTE: The min term does NOT include dq (unlike the main term which uses dq * d)
                    __m128i mins_and_scales = _mm_loadu_si128((const __m128i *) scales);
                    __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), _mm_set1_epi8(0xF));
                    __m256i mins = _mm256_cvtepi8_epi16(mins8);
                    __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i *) q8Block.bsums));
                    __m256 minTerm = _mm256_fmadd_ps(_mm256_set1_ps(-dmin * d), _mm256_cvtepi32_ps(prod), _mm256_setzero_ps());

                    // Scale shuffle table (same as in dotProductQ2_K_PrePacked_Q8_AVX2)
                    // §3 Optimization: hoist shuffle table load out of inner loop — register copy
                    static const uint8_t k_shuffle[128] = {
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                            2,
                            3,
                            2,
                            3,
                            2,
                            3,
                            2,
                            3,
                            2,
                            3,
                            2,
                            3,
                            2,
                            3,
                            2,
                            3,
                            4,
                            5,
                            4,
                            5,
                            4,
                            5,
                            4,
                            5,
                            4,
                            5,
                            4,
                            5,
                            4,
                            5,
                            4,
                            5,
                            6,
                            7,
                            6,
                            7,
                            6,
                            7,
                            6,
                            7,
                            6,
                            7,
                            6,
                            7,
                            6,
                            7,
                            6,
                            7,
                            8,
                            9,
                            8,
                            9,
                            8,
                            9,
                            8,
                            9,
                            8,
                            9,
                            8,
                            9,
                            8,
                            9,
                            8,
                            9,
                            10,
                            11,
                            10,
                            11,
                            10,
                            11,
                            10,
                            11,
                            10,
                            11,
                            10,
                            11,
                            10,
                            11,
                            10,
                            11,
                            12,
                            13,
                            12,
                            13,
                            12,
                            13,
                            12,
                            13,
                            12,
                            13,
                            12,
                            13,
                            12,
                            13,
                            12,
                            13,
                            14,
                            15,
                            14,
                            15,
                            14,
                            15,
                            14,
                            15,
                            14,
                            15,
                            14,
                            15,
                            14,
                            15,
                            14,
                            15,
                    };

                    // Load scale bytes — plus hoist shuffle tables into registers
                    __m128i sc128 = _mm_loadu_si128((const __m128i *) scales);
                    __m128i scales8 = _mm_and_si128(sc128, _mm_set1_epi8(0xF));
                    __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
                    __m128i slo = _mm256_extracti128_si256(scales16, 0);
                    __m128i shi = _mm256_extracti128_si256(scales16, 1);
                    __m256i scales_shuf[2] = {
                            _mm256_set_m128i(slo, slo),
                            _mm256_set_m128i(shi, shi),
                    };

                    // Hoist 4 shuffle vectors into registers once per block row setup
                    __m256i k_shuf[4] = {
                            _mm256_loadu_si256((const __m256i *) (k_shuffle + 0)),
                            _mm256_loadu_si256((const __m256i *) (k_shuffle + 32)),
                            _mm256_loadu_si256((const __m256i *) (k_shuffle + 64)),
                            _mm256_loadu_si256((const __m256i *) (k_shuffle + 96)),
                    };

                    // Load scale bytes (re-do, scales needed below)
                    sc128 = _mm_loadu_si128((const __m128i *) scales);
                    scales8 = _mm_and_si128(sc128, _mm_set1_epi8(0xF));
                    scales16 = _mm256_cvtepi8_epi16(scales8);
                    slo = _mm256_extracti128_si256(scales16, 0);
                    shi = _mm256_extracti128_si256(scales16, 1);
                    scales_shuf[0] = _mm256_set_m128i(slo, slo);
                    scales_shuf[1] = _mm256_set_m128i(shi, shi);

                    // Process 2 chunks of 128 elements each (256 total).
                    // chunk 0 -> elements 0-127, scales_shuf[0] (scales 0-7)
                    // chunk 1 -> elements 128-255, scales_shuf[1] (scales 8-15)
                    for (int chunk = 0; chunk < 2; ++chunk) {
                        __m256i sumi = _mm256_setzero_si256();

                        // 4 sub-chunks of 32 elements each = 128 elements.
                        for (int sub = 0; sub < 4; ++sub) {
                            int idx = chunk * 4 + sub;
                            __m256i q2 = _mm256_loadu_si256((const __m256i *) (qs_expanded + idx * 32));
                            __m256i q8v = q8_ymm[idx];// Already loaded Q8_K for this sub-chunk
                            __m256i p = _mm256_maddubs_epi16(q2, q8v);
                            __m256i shuf = k_shuf[sub];
                            p = _mm256_madd_epi16(_mm256_shuffle_epi8(scales_shuf[chunk], shuf), p);
                            sumi = _mm256_add_epi32(sumi, p);
                        }

                        // acc[r] += dq * d * sumi
                        acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(dq * d), _mm256_cvtepi32_ps(sumi), acc[r]);
                    }

                    // acc[r] -= dmin * d * sum(mins * bsums)
                    acc[r] = _mm256_add_ps(acc[r], minTerm);
                }
            }

            // Store results for this batch
            for (uint32_t r = 0; r < batchSize; ++r) {
                __m128 hi = _mm256_extractf128_ps(acc[r], 1);
                __m128 lo = _mm256_castps256_ps128(acc[r]);
                __m128 sum128 = _mm_add_ps(lo, hi);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                result[rowStart + r] = _mm_cvtss_f32(sum128);
            }
        });
    }

    // ---- AVX2 register-tiled batch GEMM for fused Q2_K gate+up (8 rows) ----
    //
    // Computes gate = x * W_gate^T and up = x * W_up^T in a single register-tiled
    // pass. For each tile of BATCH_SIZE=8 rows, the Q8_K x-vector data is loaded
    // once and reused across all 8 rows AND both matrices, halving the dominant
    // x-vector memory traffic vs. computing gate and up separately.
    void matMulVecBatchGateUpQ2_K_PrePacked_Q8_AVX2(
            const uint8_t *gatePrepacked,
            const uint8_t *upPrepacked,
            const float *x,
            uint32_t rows,
            uint32_t cols,
            float *gateOut,
            float *upOut) {
        static constexpr uint32_t BATCH_SIZE = 8;
        static constexpr uint32_t BLOCK_SIZE = 256;
        static constexpr uint32_t PREPACKED_BLOCK_BYTES = 276;

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

        // Quantize x to Q8_K once (reused across all rows AND both matrices)
        uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::vector<float> xCopy(cols + blocksPerRow * BLOCK_SIZE - cols);
        std::memcpy(xCopy.data(), x, cols * sizeof(float));
        std::memset(xCopy.data() + cols, 0, (xCopy.size() - cols) * sizeof(float));

        std::vector<Q8KBlock> q8(blocksPerRow);
        {
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const float *xb = xCopy.data() + b * BLOCK_SIZE;
                Q8KBlock &blk = q8[b];
                float max = 0.0f;
                float amax = 0.0f;
                for (uint32_t j = 0; j < BLOCK_SIZE; ++j) {
                    float ax = std::fabs(xb[j]);
                    if (ax > amax) {
                        amax = ax;
                        max = xb[j];
                    }
                }
                if (amax == 0.0f) {
                    blk.d = 0.0f;
                    std::memset(blk.qs, 0, BLOCK_SIZE);
                    std::memset(blk.bsums, 0, sizeof(blk.bsums));
                    continue;
                }
                const float iscale = -127.0f / max;
                for (uint32_t j = 0; j < BLOCK_SIZE; ++j) {
                    int v = static_cast<int>(std::lrintf(iscale * xb[j]));
                    blk.qs[j] = static_cast<int8_t>(std::min(127, v));
                }
                for (uint32_t j = 0; j < BLOCK_SIZE / 16; ++j) {
                    int sum = 0;
                    for (uint32_t ii = 0; ii < 16; ++ii) {
                        sum += blk.qs[j * 16 + ii];
                    }
                    blk.bsums[j] = static_cast<int16_t>(sum);
                }
                blk.d = 1.0f / iscale;
            }
        }

        // Scale shuffle table (q3k pattern): for each 16-element group, duplicate
        // its 16-bit scale across 8 int16 lanes.
        static const uint8_t k_shuffle[128] = {
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
        };

        // Process rows in tiles of BATCH_SIZE, parallelized over tiles.
        // Each tile of 8 rows is independent; the Q8_K x-vector data (q8) is
        // shared read-only across threads.
        uint32_t numTiles = (rows + BATCH_SIZE - 1) / BATCH_SIZE;
        ThreadPool::instance().parallelForSlab(0, numTiles, [&](uint32_t tile) {
            uint32_t rowStart = tile * BATCH_SIZE;
            uint32_t batchSize = std::min(BATCH_SIZE, rows - rowStart);

            __m256 gateAcc[8] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                                 _mm256_setzero_ps(), _mm256_setzero_ps(),
                                 _mm256_setzero_ps(), _mm256_setzero_ps(),
                                 _mm256_setzero_ps(), _mm256_setzero_ps()};
            __m256 upAcc[8] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                               _mm256_setzero_ps(), _mm256_setzero_ps(),
                               _mm256_setzero_ps(), _mm256_setzero_ps(),
                               _mm256_setzero_ps(), _mm256_setzero_ps()};

            // Process each block column
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const Q8KBlock &q8Block = q8[b];
                const float d = q8Block.d;

                // Load Q8_K block into registers once per batch
                __m256i q8_ymm[8];// 8 x 32-byte blocks = 256 elements
                for (int i = 0; i < 8; ++i) {
                    q8_ymm[i] = _mm256_loadu_si256((const __m256i *) (q8Block.qs + i * 32));
                }

                // Compute the block contribution for one matrix row block.
                // Returns the accumulated __m256 (main term + min term).
                auto blockDot = [&](const uint8_t *blockData) -> __m256 {
                    const float dq = halfToFloat(*(const uint16_t *) (blockData + 16));
                    const float dmin = halfToFloat(*(const uint16_t *) (blockData + 18));
                    const uint8_t *scales = blockData;
                    const uint8_t *qs_expanded = blockData + 20;

                    // Min term: -dmin * d * sum(mins * bsums)
                    __m128i mins_and_scales = _mm_loadu_si128((const __m128i *) scales);
                    __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), _mm_set1_epi8(0xF));
                    __m256i mins = _mm256_cvtepi8_epi16(mins8);
                    __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i *) q8Block.bsums));
                    __m256 minTerm = _mm256_fmadd_ps(_mm256_set1_ps(-dmin * d), _mm256_cvtepi32_ps(prod), _mm256_setzero_ps());

                    // Load scale bytes
                    __m128i sc128 = _mm_loadu_si128((const __m128i *) scales);
                    __m128i scales8 = _mm_and_si128(sc128, _mm_set1_epi8(0xF));
                    __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
                    __m128i slo = _mm256_extracti128_si256(scales16, 0);
                    __m128i shi = _mm256_extracti128_si256(scales16, 1);
                    __m256i scales_shuf[2] = {
                            _mm256_set_m128i(slo, slo),
                            _mm256_set_m128i(shi, shi),
                    };

                    // §3 Optimization: Hoist k_shuffle tables into registers before inner loop
                    __m256i k_shuf[4] = {
                            _mm256_loadu_si256((const __m256i *) (k_shuffle + 0)),
                            _mm256_loadu_si256((const __m256i *) (k_shuffle + 32)),
                            _mm256_loadu_si256((const __m256i *) (k_shuffle + 64)),
                            _mm256_loadu_si256((const __m256i *) (k_shuffle + 96)),
                    };

                    __m256 acc = _mm256_setzero_ps();
                    // 2 chunks of 128 elements each (256 total)
                    for (int chunk = 0; chunk < 2; ++chunk) {
                        __m256i sumi = _mm256_setzero_si256();
                        for (int sub = 0; sub < 4; ++sub) {
                            int idx = chunk * 4 + sub;
                            __m256i q2 = _mm256_loadu_si256((const __m256i *) (qs_expanded + idx * 32));
                            __m256i q8v = q8_ymm[idx];
                            __m256i p = _mm256_maddubs_epi16(q2, q8v);
                            __m256i shuf = k_shuf[sub];
                            p = _mm256_madd_epi16(_mm256_shuffle_epi8(scales_shuf[chunk], shuf), p);
                            sumi = _mm256_add_epi32(sumi, p);
                        }
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(dq * d), _mm256_cvtepi32_ps(sumi), acc);
                    }
                    acc = _mm256_add_ps(acc, minTerm);
                    return acc;
                };

                // Process each row in the batch
                for (uint32_t r = 0; r < batchSize; ++r) {
                    const uint8_t *gateRow = gatePrepacked +
                                             static_cast<uint64_t>(rowStart + r) *
                                                     static_cast<uint64_t>(blocksPerRow) *
                                                     PREPACKED_BLOCK_BYTES;
                    const uint8_t *upRow = upPrepacked +
                                           static_cast<uint64_t>(rowStart + r) *
                                                   static_cast<uint64_t>(blocksPerRow) *
                                                   PREPACKED_BLOCK_BYTES;
                    const uint8_t *gateBlock = gateRow + b * PREPACKED_BLOCK_BYTES;
                    const uint8_t *upBlock = upRow + b * PREPACKED_BLOCK_BYTES;

                    gateAcc[r] = _mm256_add_ps(gateAcc[r], blockDot(gateBlock));
                    upAcc[r] = _mm256_add_ps(upAcc[r], blockDot(upBlock));
                }
            }

            // Store results for this batch
            for (uint32_t r = 0; r < batchSize; ++r) {
                auto hsum = [](__m256 v) -> float {
                    __m128 hi = _mm256_extractf128_ps(v, 1);
                    __m128 lo = _mm256_castps256_ps128(v);
                    __m128 sum128 = _mm_add_ps(lo, hi);
                    sum128 = _mm_hadd_ps(sum128, sum128);
                    sum128 = _mm_hadd_ps(sum128, sum128);
                    return _mm_cvtss_f32(sum128);
                };
                gateOut[rowStart + r] = hsum(gateAcc[r]);
                upOut[rowStart + r] = hsum(upAcc[r]);
            }
        });
    }

    // ---- AVX2 register-tiled batch GEMM for fused Q2_K gate+up over a batch ----
    //
    // Prefill kernel. Computes gateOut[s*rows + j] and upOut[s*rows + j] for all
    // tokens s and output rows j. The dominant prefill cost is re-reading the
    // weight matrix once per token. This kernel fixes that: for each tile of 8
    // rows, the weight blocks are loaded once and reused across all seqLen tokens,
    // cutting weight-matrix memory traffic by ~seqLen. Q8_K x-vector data is
    // loaded once per (token, block) and reused across the 8 rows in the tile.
    void matMulVecBatchGateUpQ2_K_PrePacked_Q8_Batch_AVX2(
            const uint8_t *gatePrepacked,
            const uint8_t *upPrepacked,
            const float *X,
            uint32_t seqLen,
            uint32_t rows,
            uint32_t cols,
            float *gateOut,
            float *upOut,
            bool applySwish) {
        static constexpr uint32_t BATCH_SIZE = 8;
        static constexpr uint32_t BLOCK_SIZE = 256;
        static constexpr uint32_t PREPACKED_BLOCK_BYTES = 276;

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

        uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64_t rowStrideBytes = static_cast<uint64_t>(blocksPerRow) * PREPACKED_BLOCK_BYTES;

        // Quantize each token's x to Q8_K once (reused across all rows AND both
        // matrices).
        // P7: reusable grow-only scratch instead of a fresh heap allocation on
        // every prefill call. This buffer is written by the CALLING thread
        // before the parallelFor and only read by workers afterwards, so it
        // must be SHARED (a thread_local would give workers an empty copy).
        // The kernels are invoked serially from the forward() path, so a plain
        // static is safe.
        static std::vector<Q8KBlock> q8All;
        q8All.resize(static_cast<size_t>(seqLen) * blocksPerRow);
        for (uint32_t s = 0; s < seqLen; ++s) {
            GGMLDequantize::quantizeQ8K(X + static_cast<size_t>(s) * cols, cols,
                                        q8All.data() + static_cast<size_t>(s) * blocksPerRow);
        }

        // Scale shuffle table (q3k pattern): for each 16-element group, duplicate
        // its 16-bit scale across 8 int16 lanes.
        static const uint8_t k_shuffle[128] = {
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
        };

        // Process rows in tiles of BATCH_SIZE, parallelized over tiles.
        uint32_t numTiles = (rows + BATCH_SIZE - 1) / BATCH_SIZE;
        ThreadPool::instance().parallelForSlab(0, numTiles, [&](uint32_t tile) {
            uint32_t rowStart = tile * BATCH_SIZE;
            uint32_t batchSize = std::min(BATCH_SIZE, rows - rowStart);

            // Accumulators kept as __m256 in memory: [batchSize][seqLen].
            // Vector256 uses a 32-byte-aligned allocator so the _mm256_* stores
            // below stay aligned and no -Wignored-attributes warning is emitted.
            // P7: thread-local reusable accumulators (grows on demand, zeroed per
            // tile) — no per-tile heap allocation.
            static thread_local Vector256 gateAcc;
            static thread_local Vector256 upAcc;
            gateAcc.assign(static_cast<size_t>(batchSize) * seqLen,
                           _mm256_setzero_ps());
            upAcc.assign(static_cast<size_t>(batchSize) * seqLen,
                         _mm256_setzero_ps());

            // Per-block weight setup: the fp16 d/dmin, the int16 mins (for the
            // min term), the two shuffled scale vectors, and the 256 expanded q2
            // values. This is computed ONCE per (block column, row) and reused
            // across all tokens by running the token loop inside the row loop
            // (weight-stationary registers), instead of re-loading the q2 values
            // and recomputing the port-5-heavy scale setup for every token.
            struct QKSetup {
                float dq;
                float ndmin;
                __m256i mins;
                __m256i scales_shuf[2];
                const uint8_t *qs;
            };
            auto makeSetup = [&](const uint8_t *blockData) -> QKSetup {
                QKSetup st;
                st.dq = halfToFloat(*(const uint16_t *) (blockData + 16));
                st.ndmin = -halfToFloat(*(const uint16_t *) (blockData + 18));
                const uint8_t *scales = blockData;
                // Min term mins (high nibbles).
                __m128i ms = _mm_loadu_si128((const __m128i *) scales);
                __m128i mins8 = _mm_and_si128(_mm_srli_epi16(ms, 4), _mm_set1_epi8(0xF));
                st.mins = _mm256_cvtepi8_epi16(mins8);
                // Scales (low nibbles) -> shuffled scale vectors.
                __m128i scales8 = _mm_and_si128(ms, _mm_set1_epi8(0xF));
                __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
                __m128i slo = _mm256_extracti128_si256(scales16, 0);
                __m128i shi = _mm256_extracti128_si256(scales16, 1);
                st.scales_shuf[0] = _mm256_set_m128i(slo, slo);
                st.scales_shuf[1] = _mm256_set_m128i(shi, shi);
                st.qs = blockData + 20;
                return st;
            };

            // Compute the block contribution for one matrix row block.
            // q8_ymm holds the Q8_K block's 256 int8 values in 8 ymm registers.
            auto blockDot = [&](const QKSetup &st, const __m256i *q8_ymm,
                                const int16_t *bsums, float d) -> __m256 {
                // Min term: -dmin * d * sum(mins * bsums)
                __m256i prod = _mm256_madd_epi16(st.mins, _mm256_loadu_si256((const __m256i *) bsums));
                __m256 minTerm = _mm256_fmadd_ps(_mm256_set1_ps(st.ndmin * d), _mm256_cvtepi32_ps(prod), _mm256_setzero_ps());

                // §3 Optimization: Hoist k_shuffle tables into registers before inner loop
                __m256i k_shuf[4] = {
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 0)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 32)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 64)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 96)),
                };

                __m256 acc = _mm256_setzero_ps();
                // 2 chunks of 128 elements each (256 total)
                for (int chunk = 0; chunk < 2; ++chunk) {
                    __m256i sumi = _mm256_setzero_si256();
                    for (int sub = 0; sub < 4; ++sub) {
                        int idx = chunk * 4 + sub;
                        __m256i q2 = _mm256_loadu_si256((const __m256i *) (st.qs + idx * 32));
                        __m256i q8v = q8_ymm[idx];
                        __m256i p = _mm256_maddubs_epi16(q2, q8v);
                        __m256i shuf = k_shuf[sub];
                        p = _mm256_madd_epi16(_mm256_shuffle_epi8(st.scales_shuf[chunk], shuf), p);
                        sumi = _mm256_add_epi32(sumi, p);
                    }
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(st.dq * d), _mm256_cvtepi32_ps(sumi), acc);
                }
                return _mm256_add_ps(acc, minTerm);
            };

            // For each block column, process each row with the weight data held in
            // registers across the whole token batch (weight-stationary).
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                for (uint32_t r = 0; r < batchSize; ++r) {
                    const uint8_t *gateBlock =
                            gatePrepacked + static_cast<uint64_t>(rowStart + r) * rowStrideBytes +
                            static_cast<uint64_t>(b) * PREPACKED_BLOCK_BYTES;
                    const uint8_t *upBlock =
                            upPrepacked + static_cast<uint64_t>(rowStart + r) * rowStrideBytes +
                            static_cast<uint64_t>(b) * PREPACKED_BLOCK_BYTES;
                    QKSetup gateSt = makeSetup(gateBlock);
                    QKSetup upSt = makeSetup(upBlock);

                    for (uint32_t s = 0; s < seqLen; ++s) {
                        const Q8KBlock &q8 = q8All[static_cast<size_t>(s) * blocksPerRow + b];
                        const float d = q8.d;
                        __m256i q8_ymm[8];
                        for (int i = 0; i < 8; ++i) {
                            q8_ymm[i] = _mm256_loadu_si256((const __m256i *) (q8.qs + i * 32));
                        }
                        size_t accIdx = static_cast<size_t>(r) * seqLen + s;
                        gateAcc[accIdx] = _mm256_add_ps(gateAcc[accIdx],
                                                        blockDot(gateSt, q8_ymm, q8.bsums, d));
                        upAcc[accIdx] = _mm256_add_ps(upAcc[accIdx],
                                                      blockDot(upSt, q8_ymm, q8.bsums, d));
                    }
                }
            }

            // Store results (horizontal sum each accumulator once).
            for (uint32_t r = 0; r < batchSize; ++r) {
                for (uint32_t s = 0; s < seqLen; ++s) {
                    __m256 g = gateAcc[static_cast<size_t>(r) * seqLen + s];
                    __m256 u = upAcc[static_cast<size_t>(r) * seqLen + s];
                    __m128 ghi = _mm256_extractf128_ps(g, 1);
                    __m128 glo = _mm256_castps256_ps128(g);
                    __m128 gs = _mm_hadd_ps(_mm_add_ps(glo, ghi), _mm_add_ps(glo, ghi));
                    gs = _mm_hadd_ps(gs, gs);
                    __m128 uhi = _mm256_extractf128_ps(u, 1);
                    __m128 ulo = _mm256_castps256_ps128(u);
                    __m128 us = _mm_hadd_ps(_mm_add_ps(ulo, uhi), _mm_add_ps(ulo, uhi));
                    us = _mm_hadd_ps(us, us);
                    float gVal = _mm_cvtss_f32(gs);
                    float uVal = _mm_cvtss_f32(us);
                    if (applySwish) {
                        // P5: fuse the SwiGLU activation into the kernel's epilogue
                        // (silu(gate) * up) so the separate swiGLUSIMD activation
                        // pass over the (seqLen x rows) activation tensors is
                        // removed from the prefill hot loop.
                        gVal = gVal / (1.0f + std::exp(-gVal)) * uVal;
                    }
                    gateOut[static_cast<size_t>(s) * rows + rowStart + r] = gVal;
                    upOut[static_cast<size_t>(s) * rows + rowStart + r] = uVal;
                }
            }
        });
    }

    // ---- AVX2 register-tiled fused gate+up matvec for COMPACT Q2_K weights ----
    //
    // Single-token (generation) kernel. Generation is DRAM-bandwidth-bound and
    // the fused gate+up is the dominant per-token cost; the compact 84-byte
    // Q2_K blocks (vs 276-byte prepacked) cut the weight traffic ~3.3x. The
    // 2-bit quants are unpacked to bytes on the fly (repeated srli_epi16 + and
    // 0x03), producing exactly the element indices that prepackQ2_K produces,
    // so this kernel's math is bit-identical to the prepacked batch kernel.
    void matMulVecFusedGateUpQ2_K_Compact_Q8_AVX2(
            const uint8_t *gateData,
            const uint8_t *upData,
            const float *x,
            uint32_t rows,
            uint32_t cols,
            float *gateOut,
            float *upOut,
            bool applySwish) {
        static constexpr uint32_t BATCH_SIZE = 8;
        static constexpr uint32_t BLOCK_SIZE = 256;
        static constexpr uint32_t COMPACT_BLOCK_BYTES = 84;

        // Branch-free fp16->fp32 via the shared LUT (this fused Q+K kernel
        // calls it per (row, block) for both matrices — per generated token).
        auto halfToFloat = [](uint16_t h) -> float {
            return tinycoder::GGMLDequantize::halfToFloatBranchFree(h);
        };

        uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64_t rowStrideBytes = static_cast<uint64_t>(blocksPerRow) * COMPACT_BLOCK_BYTES;

        // Quantize each token's x to Q8_K once (reused across all rows AND both
        // matrices). SHARED buffer — written by the calling thread before
        // parallelFor, read by workers.
        static std::vector<Q8KBlock> q8All;
        q8All.resize(blocksPerRow);
        GGMLDequantize::quantizeQ8K(x, cols, q8All.data());

        // Scale shuffle table (q3k pattern): for each 16-element group,
        // duplicate its 16-bit scale across 8 int16 lanes. Identical to the
        // prepacked kernels (element order matches the prepacked expansion).
        static const uint8_t k_shuffle[128] = {
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
        };

        const __m256i mask3 = _mm256_set1_epi8(0x03);

        uint32_t numTiles = (rows + BATCH_SIZE - 1) / BATCH_SIZE;
        ThreadPool::instance().parallelForSlab(0, numTiles, [&](uint32_t tile) {
            uint32_t rowStart = tile * BATCH_SIZE;
            uint32_t batchSize = std::min(BATCH_SIZE, rows - rowStart);

            // Accumulators kept as __m256 in memory: [batchSize] (one per row).
            static thread_local Vector256 gateAcc;
            static thread_local Vector256 upAcc;
            gateAcc.assign(batchSize, _mm256_setzero_ps());
            upAcc.assign(batchSize, _mm256_setzero_ps());

            // Per-block weight setup for a COMPACT block: the fp16 d/dmin, the
            // int16 mins, the two shuffled scale vectors, and the 8 unpacked q2
            // planes (256 expanded values). Computed ONCE per (block column,
            // row); with seqLen==1 there is no token loop to hoist out of.
            struct QKSetup {
                float dq;
                float ndmin;
                __m256i mins;
                __m256i scales_shuf[2];
                __m256i q2[8];
            };
            auto makeSetup = [&](const uint8_t *blockData) -> QKSetup {
                QKSetup st;
                st.dq = halfToFloat(*(const uint16_t *) (blockData + 80));
                st.ndmin = -halfToFloat(*(const uint16_t *) (blockData + 82));
                const uint8_t *scales = blockData;
                // Min term mins (high nibbles).
                __m128i ms = _mm_loadu_si128((const __m128i *) scales);
                __m128i mins8 = _mm_and_si128(_mm_srli_epi16(ms, 4), _mm_set1_epi8(0xF));
                st.mins = _mm256_cvtepi8_epi16(mins8);
                // Scales (low nibbles) -> shuffled scale vectors.
                __m128i scales8 = _mm_and_si128(ms, _mm_set1_epi8(0xF));
                __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
                __m128i slo = _mm256_extracti128_si256(scales16, 0);
                __m128i shi = _mm256_extracti128_si256(scales16, 1);
                st.scales_shuf[0] = _mm256_set_m128i(slo, slo);
                st.scales_shuf[1] = _mm256_set_m128i(shi, shi);
                // Unpack the 2-bit planes. Each of the two 32-byte qs quarters
                // contains 4 planes (2 bits per byte): plane p = (byte >> 2p) & 3.
                // The element order produced (quarter 0 shift 0,2,4,6 then
                // quarter 1 shift 0,2,4,6) is byte-identical to prepackQ2_K's
                // expansion, so the scale/min shuffle math stays valid.
                for (int h = 0; h < 2; ++h) {
                    const uint8_t *qsrc = blockData + 16 + h * 32;
                    __m256i b = _mm256_loadu_si256((const __m256i *) qsrc);
                    st.q2[h * 4 + 0] = _mm256_and_si256(b, mask3);
                    __m256i s2 = _mm256_srli_epi16(b, 2);
                    st.q2[h * 4 + 1] = _mm256_and_si256(s2, mask3);
                    __m256i s4 = _mm256_srli_epi16(s2, 2);
                    st.q2[h * 4 + 2] = _mm256_and_si256(s4, mask3);
                    __m256i s6 = _mm256_srli_epi16(s4, 2);
                    st.q2[h * 4 + 3] = _mm256_and_si256(s6, mask3);
                }
                return st;
            };

            // Compute the block contribution for one matrix row block.
            auto blockDot = [&](const QKSetup &st, const __m256i *q8_ymm,
                                const int16_t *bsums, float d) -> __m256 {
                // Min term: -dmin * d * sum(mins * bsums)
                __m256i prod = _mm256_madd_epi16(
                        st.mins, _mm256_loadu_si256((const __m256i *) bsums));
                __m256 minTerm = _mm256_fmadd_ps(
                        _mm256_set1_ps(st.ndmin * d), _mm256_cvtepi32_ps(prod),
                        _mm256_setzero_ps());

                // §3 Optimization: Hoist k_shuffle tables into registers.
                __m256i k_shuf[4] = {
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 0)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 32)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 64)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 96)),
                };

                __m256 acc = _mm256_setzero_ps();
                // 2 chunks of 128 elements each (256 total)
                for (int chunk = 0; chunk < 2; ++chunk) {
                    __m256i sumi = _mm256_setzero_si256();
                    for (int sub = 0; sub < 4; ++sub) {
                        int idx = chunk * 4 + sub;
                        __m256i p = _mm256_maddubs_epi16(st.q2[idx], q8_ymm[idx]);
                        __m256i shuf = k_shuf[sub];
                        p = _mm256_madd_epi16(
                                _mm256_shuffle_epi8(st.scales_shuf[chunk], shuf), p);
                        sumi = _mm256_add_epi32(sumi, p);
                    }
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(st.dq * d),
                                          _mm256_cvtepi32_ps(sumi), acc);
                }
                return _mm256_add_ps(acc, minTerm);
            };

            // For each block column, load the Q8_K x data once and reuse it
            // across all rows in the tile (and both matrices).
            // §4 Optimization: Prefetch weights one block ahead to hide DRAM latency.
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                // Prefetch next block's gate+up weights ~128 blocks ahead.
                {
                    uint32_t prefetchB = std::min(b + 128, blocksPerRow - 1);
                    const uint8_t *prefetchGate = gateData +
                                                  static_cast<uint64_t>(rowStart) * rowStrideBytes +
                                                  static_cast<uint64_t>(prefetchB) * COMPACT_BLOCK_BYTES;
                    const uint8_t *prefetchUp = upData +
                                                static_cast<uint64_t>(rowStart) * rowStrideBytes +
                                                static_cast<uint64_t>(prefetchB) * COMPACT_BLOCK_BYTES;
                    _mm_prefetch((const char *) prefetchGate, _MM_HINT_T0);
                    _mm_prefetch((const char *) prefetchUp, _MM_HINT_T0);
                }

                const Q8KBlock &q8 = q8All[b];
                const float d = q8.d;
                __m256i q8_ymm[8];
                for (int i = 0; i < 8; ++i) {
                    q8_ymm[i] = _mm256_loadu_si256(
                            (const __m256i *) (q8.qs + i * 32));
                }
                for (uint32_t r = 0; r < batchSize; ++r) {
                    const uint8_t *gateBlock =
                            gateData + static_cast<uint64_t>(rowStart + r) * rowStrideBytes +
                            static_cast<uint64_t>(b) * COMPACT_BLOCK_BYTES;
                    const uint8_t *upBlock =
                            upData + static_cast<uint64_t>(rowStart + r) * rowStrideBytes +
                            static_cast<uint64_t>(b) * COMPACT_BLOCK_BYTES;
                    QKSetup gateSt = makeSetup(gateBlock);
                    QKSetup upSt = makeSetup(upBlock);

                    gateAcc[r] = _mm256_add_ps(gateAcc[r],
                                               blockDot(gateSt, q8_ymm, q8.bsums, d));
                    upAcc[r] = _mm256_add_ps(upAcc[r],
                                             blockDot(upSt, q8_ymm, q8.bsums, d));
                }
            }

            // Store results (horizontal sum each accumulator once).
            for (uint32_t r = 0; r < batchSize; ++r) {
                __m256 g = gateAcc[r];
                __m256 u = upAcc[r];
                __m128 ghi = _mm256_extractf128_ps(g, 1);
                __m128 glo = _mm256_castps256_ps128(g);
                __m128 gs = _mm_hadd_ps(_mm_add_ps(glo, ghi), _mm_add_ps(glo, ghi));
                gs = _mm_hadd_ps(gs, gs);
                __m128 uhi = _mm256_extractf128_ps(u, 1);
                __m128 ulo = _mm256_castps256_ps128(u);
                __m128 us = _mm_hadd_ps(_mm_add_ps(ulo, uhi), _mm_add_ps(ulo, uhi));
                us = _mm_hadd_ps(us, us);
                float gVal = _mm_cvtss_f32(gs);
                float uVal = _mm_cvtss_f32(us);
                if (applySwish) {
                    // Fuse SwiGLU (silu(gate) * up) into the epilogue so the
                    // separate swiGLUSIMD pass over the (rows) gate/up arrays is
                    // removed from the per-token generation hot loop. gateOut
                    // receives the activated output; upOut stays the raw up
                    // projection (the ffnDown matmul below consumes the combined
                    // activation directly).
                    gVal = gVal / (1.0f + std::exp(-gVal)) * uVal;
                }
                gateOut[rowStart + r] = gVal;
                upOut[rowStart + r] = uVal;
            }
        });
    }

    // ---- AVX2 fused Q+K matvec for COMPACT (raw GGUF) Q2_K weights, single token ----
    //
    // Generation kernel. Computes qOut = x * Q^T and kOut = x * K^T in a single
    // pass, where both Q and K are compact Q2_K (84 B/block) with the same
    // column count (so the row stride is identical). The triangle is DRAM-
    // bandwidth-bound; compact Q2_K (84 B/block) cuts the Q/K weight traffic
    // ~3.3x vs the pre-packed layout (276 B/block) that the int8 batch kernels
    // consume. The combined row space (qRows + kRows) is tiled in 8-row batches
    // exactly like the fused gate+up kernel; the Q8_K x-vector is quantized once
    // and reused across all rows of both matrices.
    void matMulVecFusedQKQ2_K_Compact_Q8_AVX2(
            const uint8_t *qData,
            const uint8_t *kData,
            const float *x,
            uint32_t qRows,
            uint32_t kRows,
            uint32_t cols,
            float *qOut,
            float *kOut) {
        static constexpr uint32_t BATCH_SIZE = 8;
        static constexpr uint32_t BLOCK_SIZE = 256;
        static constexpr uint32_t COMPACT_BLOCK_BYTES = 84;

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

        uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64_t rowStrideBytes = static_cast<uint64_t>(blocksPerRow) * COMPACT_BLOCK_BYTES;

        // Quantize x to Q8_K once (reused across all rows of both matrices).
        static std::vector<Q8KBlock> q8All;
        q8All.resize(blocksPerRow);
        GGMLDequantize::quantizeQ8K(x, cols, q8All.data());

        // Scale shuffle table (identical to the compact fused gate+up kernel).
        static const uint8_t k_shuffle[128] = {
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
        };

        const __m256i mask3 = _mm256_set1_epi8(0x03);

        uint32_t totalRows = qRows + kRows;
        uint32_t numTiles = (totalRows + BATCH_SIZE - 1) / BATCH_SIZE;
        ThreadPool::instance().parallelForSlab(0, numTiles, [&](uint32_t tile) {
            uint32_t rowStart = tile * BATCH_SIZE;
            uint32_t batchSize = std::min(BATCH_SIZE, totalRows - rowStart);

            static thread_local Vector256 acc;
            acc.assign(batchSize, _mm256_setzero_ps());

            struct QKSetup {
                float dq;
                float ndmin;
                __m256i mins;
                __m256i scales_shuf[2];
                __m256i q2[8];
            };
            auto makeSetup = [&](const uint8_t *blockData) -> QKSetup {
                QKSetup st;
                st.dq = halfToFloat(*(const uint16_t *) (blockData + 80));
                st.ndmin = -halfToFloat(*(const uint16_t *) (blockData + 82));
                const uint8_t *scales = blockData;
                __m128i ms = _mm_loadu_si128((const __m128i *) scales);
                __m128i mins8 = _mm_and_si128(_mm_srli_epi16(ms, 4), _mm_set1_epi8(0xF));
                st.mins = _mm256_cvtepi8_epi16(mins8);
                __m128i scales8 = _mm_and_si128(ms, _mm_set1_epi8(0xF));
                __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
                __m128i slo = _mm256_extracti128_si256(scales16, 0);
                __m128i shi = _mm256_extracti128_si256(scales16, 1);
                st.scales_shuf[0] = _mm256_set_m128i(slo, slo);
                st.scales_shuf[1] = _mm256_set_m128i(shi, shi);
                for (int h = 0; h < 2; ++h) {
                    const uint8_t *qsrc = blockData + 16 + h * 32;
                    __m256i b = _mm256_loadu_si256((const __m256i *) qsrc);
                    st.q2[h * 4 + 0] = _mm256_and_si256(b, mask3);
                    __m256i s2 = _mm256_srli_epi16(b, 2);
                    st.q2[h * 4 + 1] = _mm256_and_si256(s2, mask3);
                    __m256i s4 = _mm256_srli_epi16(s2, 2);
                    st.q2[h * 4 + 2] = _mm256_and_si256(s4, mask3);
                    __m256i s6 = _mm256_srli_epi16(s4, 2);
                    st.q2[h * 4 + 3] = _mm256_and_si256(s6, mask3);
                }
                return st;
            };

            auto blockDot = [&](const QKSetup &st, const __m256i *q8_ymm,
                                const int16_t *bsums, float d) -> __m256 {
                // Min term: -dmin * d * sum(mins * bsums). Per-lane vector term:
                // each lane holds its own bsum contribution (NEVER scalar-broadcast
                // offsets — the store-time horizontal sum would count a broadcast
                // 8x, the Q3K maxDiff 10.06 bug).
                __m256i prod = _mm256_madd_epi16(
                        st.mins, _mm256_loadu_si256((const __m256i *) bsums));
                __m256 minTerm = _mm256_fmadd_ps(
                        _mm256_set1_ps(st.ndmin * d), _mm256_cvtepi32_ps(prod),
                        _mm256_setzero_ps());

                // §3 Optimization: Hoist k_shuffle tables into registers.
                __m256i k_shuf[4] = {
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 0)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 32)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 64)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 96)),
                };

                __m256 accB = _mm256_setzero_ps();
                for (int chunk = 0; chunk < 2; ++chunk) {
                    __m256i sumi = _mm256_setzero_si256();
                    for (int sub = 0; sub < 4; ++sub) {
                        int idx = chunk * 4 + sub;
                        __m256i p = _mm256_maddubs_epi16(st.q2[idx], q8_ymm[idx]);
                        __m256i shuf = k_shuf[sub];
                        p = _mm256_madd_epi16(
                                _mm256_shuffle_epi8(st.scales_shuf[chunk], shuf), p);
                        sumi = _mm256_add_epi32(sumi, p);
                    }
                    accB = _mm256_fmadd_ps(_mm256_set1_ps(st.dq * d),
                                           _mm256_cvtepi32_ps(sumi), accB);
                }
                return _mm256_add_ps(accB, minTerm);
            };

            // Decode each combined-row-space index: [0, qRows) -> Q matrix,
            // [qRows, totalRows) -> K matrix. Both share the identical compact
            // row stride, so the block pointers are computed uniformly.
            auto rowBase = [&](uint32_t globalRow) -> const uint8_t * {
                if (globalRow < qRows) {
                    return qData + static_cast<uint64_t>(globalRow) * rowStrideBytes;
                }
                return kData + static_cast<uint64_t>(globalRow - qRows) * rowStrideBytes;
            };

            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const Q8KBlock &q8 = q8All[b];
                const float d = q8.d;
                __m256i q8_ymm[8];
                for (int i = 0; i < 8; ++i) {
                    q8_ymm[i] = _mm256_loadu_si256(
                            (const __m256i *) (q8.qs + i * 32));
                }
                for (uint32_t r = 0; r < batchSize; ++r) {
                    const uint8_t *blockData = rowBase(rowStart + r) +
                                               static_cast<uint64_t>(b) * COMPACT_BLOCK_BYTES;
                    QKSetup st = makeSetup(blockData);
                    acc[r] = _mm256_add_ps(acc[r], blockDot(st, q8_ymm, q8.bsums, d));
                }
            }

            for (uint32_t r = 0; r < batchSize; ++r) {
                uint32_t globalRow = rowStart + r;
                __m256 g = acc[r];
                __m128 ghi = _mm256_extractf128_ps(g, 1);
                __m128 glo = _mm256_castps256_ps128(g);
                __m128 gs = _mm_hadd_ps(_mm_add_ps(glo, ghi), _mm_add_ps(glo, ghi));
                gs = _mm_hadd_ps(gs, gs);
                float val = _mm_cvtss_f32(gs);
                if (globalRow < qRows) {
                    qOut[globalRow] = val;
                } else {
                    kOut[globalRow - qRows] = val;
                }
            }
        });
    }

    // ---- AVX2 register-tiled batch GEMM for a single Q2_K matrix over a batch ----
    //
    // Prefill kernel. Computes out[s*rows + j] for all tokens s and output rows j.
    // The dominant prefill cost is re-reading the weight matrix once per token.
    // This kernel fixes that: for each tile of 8 rows, the weight blocks are
    // loaded once and reused across all seqLen tokens, cutting weight-matrix
    // memory traffic by ~seqLen. Q8_K x-vector data is loaded once per
    // (token, block) and reused across the 8 rows in the tile. This is the
    // single-matrix analogue of matMulVecBatchGateUpQ2_K_PrePacked_Q8_Batch_AVX2,
    // used for the ffnDown projection.
    void matMulVecBatchQ2_K_PrePacked_Q8_Batch_AVX2(
            const uint8_t *prepackedData,
            const float *X,
            uint32_t seqLen,
            uint32_t rows,
            uint32_t cols,
            float *out) {
        static constexpr uint32_t BATCH_SIZE = 8;
        static constexpr uint32_t BLOCK_SIZE = 256;
        static constexpr uint32_t PREPACKED_BLOCK_BYTES = 276;

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

        uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64_t rowStrideBytes = static_cast<uint64_t>(blocksPerRow) * PREPACKED_BLOCK_BYTES;

        // Quantize each token's x to Q8_K once (reused across all rows).
        // P7: reusable grow-only scratch instead of a fresh heap allocation on
        // every prefill call. SHARED buffer — written by the calling thread
        // before the parallelFor, read by workers (see the gate/up kernel note).
        static std::vector<Q8KBlock> q8All;
        q8All.resize(static_cast<size_t>(seqLen) * blocksPerRow);
        for (uint32_t s = 0; s < seqLen; ++s) {
            GGMLDequantize::quantizeQ8K(X + static_cast<size_t>(s) * cols, cols,
                                        q8All.data() + static_cast<size_t>(s) * blocksPerRow);
        }

        // Scale shuffle table (q3k pattern): for each 16-element group, duplicate
        // its 16-bit scale across 8 int16 lanes.
        static const uint8_t k_shuffle[128] = {
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
        };

        // Process rows in tiles of BATCH_SIZE, parallelized over tiles.
        uint32_t numTiles = (rows + BATCH_SIZE - 1) / BATCH_SIZE;
        ThreadPool::instance().parallelForSlab(0, numTiles, [&](uint32_t tile) {
            uint32_t rowStart = tile * BATCH_SIZE;
            uint32_t batchSize = std::min(BATCH_SIZE, rows - rowStart);

            // Accumulators kept as __m256 in memory: [batchSize][seqLen].
            // P7: thread-local reusable accumulator (grows on demand, zeroed per
            // tile) — no per-tile heap allocation.
            static thread_local Vector256 acc;
            acc.assign(static_cast<size_t>(batchSize) * seqLen,
                       _mm256_setzero_ps());

            // Per-block weight setup (fp16 d/dmin, int16 mins, shuffled scale
            // vectors, and the 256 expanded q2 values), computed once per
            // (block column, row) and reused across all tokens by running the
            // token loop inside the row loop (weight-stationary registers).
            struct QKSetup {
                float dq;
                float ndmin;
                __m256i mins;
                __m256i scales_shuf[2];
                const uint8_t *qs;
            };
            auto makeSetup = [&](const uint8_t *blockData) -> QKSetup {
                QKSetup st;
                st.dq = halfToFloat(*(const uint16_t *) (blockData + 16));
                st.ndmin = -halfToFloat(*(const uint16_t *) (blockData + 18));
                const uint8_t *scales = blockData;
                __m128i ms = _mm_loadu_si128((const __m128i *) scales);
                __m128i mins8 = _mm_and_si128(_mm_srli_epi16(ms, 4), _mm_set1_epi8(0xF));
                st.mins = _mm256_cvtepi8_epi16(mins8);
                __m128i scales8 = _mm_and_si128(ms, _mm_set1_epi8(0xF));
                __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
                __m128i slo = _mm256_extracti128_si256(scales16, 0);
                __m128i shi = _mm256_extracti128_si256(scales16, 1);
                st.scales_shuf[0] = _mm256_set_m128i(slo, slo);
                st.scales_shuf[1] = _mm256_set_m128i(shi, shi);
                st.qs = blockData + 20;
                return st;
            };

            // Compute the block contribution for one matrix row block.
            auto blockDot = [&](const QKSetup &st, const __m256i *q8_ymm,
                                const int16_t *bsums, float d) -> __m256 {
                // Min term: -dmin * d * sum(mins * bsums)
                __m256i prod = _mm256_madd_epi16(st.mins, _mm256_loadu_si256((const __m256i *) bsums));
                __m256 minTerm = _mm256_fmadd_ps(_mm256_set1_ps(st.ndmin * d), _mm256_cvtepi32_ps(prod), _mm256_setzero_ps());

                // §3 Optimization: Hoist k_shuffle tables into registers before inner loop
                __m256i k_shuf[4] = {
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 0)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 32)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 64)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 96)),
                };

                __m256 accv = _mm256_setzero_ps();
                // 2 chunks of 128 elements each (256 total)
                for (int chunk = 0; chunk < 2; ++chunk) {
                    __m256i sumi = _mm256_setzero_si256();
                    for (int sub = 0; sub < 4; ++sub) {
                        int idx = chunk * 4 + sub;
                        __m256i q2 = _mm256_loadu_si256((const __m256i *) (st.qs + idx * 32));
                        __m256i q8v = q8_ymm[idx];
                        __m256i p = _mm256_maddubs_epi16(q2, q8v);
                        __m256i shuf = k_shuf[sub];
                        p = _mm256_madd_epi16(_mm256_shuffle_epi8(st.scales_shuf[chunk], shuf), p);
                        sumi = _mm256_add_epi32(sumi, p);
                    }
                    accv = _mm256_fmadd_ps(_mm256_set1_ps(st.dq * d), _mm256_cvtepi32_ps(sumi), accv);
                }
                return _mm256_add_ps(accv, minTerm);
            };

            // For each block column, process each row with the weight data held in
            // registers across the whole token batch (weight-stationary).
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                for (uint32_t r = 0; r < batchSize; ++r) {
                    const uint8_t *block = prepackedData +
                                           static_cast<uint64_t>(rowStart + r) * rowStrideBytes +
                                           static_cast<uint64_t>(b) * PREPACKED_BLOCK_BYTES;
                    QKSetup st = makeSetup(block);

                    for (uint32_t s = 0; s < seqLen; ++s) {
                        const Q8KBlock &q8 = q8All[static_cast<size_t>(s) * blocksPerRow + b];
                        const float d = q8.d;
                        __m256i q8_ymm[8];
                        for (int i = 0; i < 8; ++i) {
                            q8_ymm[i] = _mm256_loadu_si256((const __m256i *) (q8.qs + i * 32));
                        }
                        size_t accIdx = static_cast<size_t>(r) * seqLen + s;
                        acc[accIdx] = _mm256_add_ps(acc[accIdx],
                                                    blockDot(st, q8_ymm, q8.bsums, d));
                    }
                }
            }

            // Store results (horizontal sum each accumulator once).
            for (uint32_t r = 0; r < batchSize; ++r) {
                for (uint32_t s = 0; s < seqLen; ++s) {
                    __m256 v = acc[static_cast<size_t>(r) * seqLen + s];
                    __m128 hi = _mm256_extractf128_ps(v, 1);
                    __m128 lo = _mm256_castps256_ps128(v);
                    __m128 sum128 = _mm_hadd_ps(_mm_add_ps(lo, hi), _mm_add_ps(lo, hi));
                    sum128 = _mm_hadd_ps(sum128, sum128);
                    out[static_cast<size_t>(s) * rows + rowStart + r] = _mm_cvtss_f32(sum128);
                }
            }
        });
    }

    // ---- AVX2 register-tiled batch GEMM for a single Q8_K matrix over a batch ----
    //
    // Prefill kernel. Computes out[s*rows + j] = sum_i X[s*cols + i] * W[j*cols + i]
    // for all tokens s and output rows j, using Q8_K weights (attnO, ffnDown).
    // For each tile of 8 rows the weight blocks are loaded once and reused across
    // all seqLen tokens. The Q8_K x-vector is loaded and sign-processed once per
    // (token, block) and reused across the 8 rows in the tile, and the 8 partial
    // sums are accumulated in __m256 registers and horizontal-summed once per
    // output. This replaces the previous per-(row, token, block) scalar
    // function-pointer dispatch (dotProductQ8K_Q8K_SIMD) plus scalar
    // accumulator round-trips that dominated the Q8_K prefill GEMMs.
    void matMulVecBatchQ8K_Q8K_AVX2(const tinycoder::Q8KBlock *W_q8k,
                                    const float *X, uint32_t seqLen,
                                    uint32_t rows, uint32_t cols,
                                    float *out) {
        static constexpr uint32_t BATCH_SIZE = 8;
        static constexpr uint32_t BLOCK_SIZE = 256;
        uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Quantize each token's x to Q8_K once (reused across all rows). SHARED
        // buffer — written by the calling thread before parallelFor, read by
        // workers (same grow-only-reusable scheme as the Q2_K batch kernels).
        static std::vector<tinycoder::Q8KBlock> q8All;
        q8All.resize(static_cast<size_t>(seqLen) * blocksPerRow);
        for (uint32_t s = 0; s < seqLen; ++s) {
            GGMLDequantize::quantizeQ8K(X + static_cast<size_t>(s) * cols, cols,
                                        q8All.data() + static_cast<size_t>(s) * blocksPerRow);
        }

        const __m256i one16 = _mm256_set1_epi16(1);

        // Process rows in tiles of BATCH_SIZE, parallelized over tiles.
        uint32_t numTiles = (rows + BATCH_SIZE - 1) / BATCH_SIZE;
        ThreadPool::instance().parallelForSlab(0, numTiles, [&](uint32_t tile) {
            uint32_t rowStart = tile * BATCH_SIZE;
            uint32_t batchSize = std::min(BATCH_SIZE, rows - rowStart);

            // Accumulators kept as __m256 in memory: [batchSize][seqLen].
            // P7: thread-local reusable accumulator (grows on demand, zeroed per
            // tile) — no per-tile heap allocation.
            static thread_local Vector256 acc;
            acc.assign(static_cast<size_t>(batchSize) * seqLen,
                       _mm256_setzero_ps());

            // For each block column, load the batchSize rows' weight blocks once
            // and reuse them across all seqLen tokens.
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const tinycoder::Q8KBlock *wblocks[BATCH_SIZE];
                for (uint32_t r = 0; r < batchSize; ++r) {
                    wblocks[r] = W_q8k +
                                 static_cast<size_t>(rowStart + r) * blocksPerRow + b;
                }

                for (uint32_t s = 0; s < seqLen; ++s) {
                    const tinycoder::Q8KBlock &q8 =
                            q8All[static_cast<size_t>(s) * blocksPerRow + b];
                    // Load + sign-process the x block once, reuse across all rows.
                    __m256i xv[BATCH_SIZE];
                    __m256i xabs[BATCH_SIZE];
                    for (int i = 0; i < 8; ++i) {
                        xv[i] = _mm256_loadu_si256(
                                (const __m256i *) (q8.qs + i * 32));
                        // |x| depends only on the token block, NOT on the weight
                        // row — hoist it out of the row loop (was recomputed for
                        // every one of the 8 tile rows below).
                        xabs[i] = _mm256_sign_epi8(xv[i], xv[i]);
                    }
                    const float xd = q8.d;

                    for (uint32_t r = 0; r < batchSize; ++r) {
                        const tinycoder::Q8KBlock &w = *wblocks[r];
                        // Raw int8 dot accumulated in int32 lanes: because both
                        // operands are signed, substitute (|x|, w*sign(x)) so the
                        // unsigned operand slot of maddubs is safe (llama.cpp trick).
                        __m256i sumi = _mm256_setzero_si256();
                        for (int i = 0; i < 8; ++i) {
                            __m256i wv = _mm256_loadu_si256(
                                    (const __m256i *) (w.qs + i * 32));
                            __m256i wsig = _mm256_sign_epi8(wv, xv[i]);
                            __m256i p = _mm256_maddubs_epi16(xabs[i], wsig);
                            sumi = _mm256_add_epi32(
                                    sumi, _mm256_madd_epi16(p, one16));
                        }
                        // acc += x.d * w.d * sum(int dot)  (float accumulate)
                        __m256 scale = _mm256_set1_ps(xd * w.d);
                        size_t accIdx = static_cast<size_t>(r) * seqLen + s;
                        acc[accIdx] = _mm256_fmadd_ps(
                                scale, _mm256_cvtepi32_ps(sumi), acc[accIdx]);
                    }
                }
            }

            // Store results (horizontal sum each accumulator once).
            for (uint32_t r = 0; r < batchSize; ++r) {
                for (uint32_t s = 0; s < seqLen; ++s) {
                    __m256 v = acc[static_cast<size_t>(r) * seqLen + s];
                    __m128 hi = _mm256_extractf128_ps(v, 1);
                    __m128 lo = _mm256_castps256_ps128(v);
                    __m128 sum128 = _mm_hadd_ps(_mm_add_ps(lo, hi),
                                                _mm_add_ps(lo, hi));
                    sum128 = _mm_hadd_ps(sum128, sum128);
                    out[static_cast<size_t>(s) * rows + rowStart + r] =
                            _mm_cvtss_f32(sum128);
                }
            }
        });
    }

    // ---- AVX2 register-tiled batch GEMM for a single COMPACT Q3_K matrix over a batch ----
    //
    // attnO and ffnDown are stored as Q3_K in the GGUF (110 bytes per 256-element
    // block = 0.43 B/elem). Generation is DRAM-bandwidth-bound, so reading these
    // compact blocks directly (instead of rebuilding and streaming Q8_K copies at
    // 292 B/block = 1.14 B/elem) cuts the weight traffic ~2.65x, matching
    // llama.cpp's working set for these two large matmuls.
    //
    // Q3_K block layout (dequantizeQ3_KBlock):
    //   hm[32]    @0     sign mask (bit s selects sub-block s, never advanced)
    //   q[64]     @32    2-bit planes, two 32-byte halves (one per 128-chunk)
    //   scales[12] @96   16 packed 6-bit scales (see aux transform below)
    //   d_all     @108   fp16 super-block scale
    // A sub-block s (0..7) covers elements [32s, 32s+32): q2 = (q[32*(s>>2)+l] >> 2*(s&3)) & 3,
    // hm bit s folded as w_eff = q2 - 4*hm_bit, scale = d_all*(sc[g]-32) per 16-group g=2s,2s+1.
    // The kernel rewrites w_eff as w' = w_eff + 4 (0..7, safe in maddubs' unsigned slot A)
    // and compensates with a per-block offset: dot -= 4*sum_g (sc[g]-32)*bsum[g].
    void matMulVecBatchQ3K_Q8K_AVX2(const uint8_t *W_q3k, const float *X,
                                    uint32_t seqLen, uint32_t rows,
                                    uint32_t cols, float *out,
                                    const float *residual) {
        static constexpr uint32_t BATCH_SIZE = 8;
        static constexpr uint32_t BLOCK_SIZE = 256;
        static constexpr uint32_t Q3K_BLOCK_BYTES = 110;
        uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64_t rowStride = static_cast<uint64_t>(blocksPerRow) * Q3K_BLOCK_BYTES;

        // Branch-free fp16->fp32 via the shared LUT (this Q3_K batch kernel
        // calls it once per (row, block) to unpack BOTH the d_all fp16 and
        // the 8 sub-block d/sc scales — the LUT removes the per-call branches).
        auto halfToFloat = [](uint16_t h) -> float {
            return tinycoder::GGMLDequantize::halfToFloatBranchFree(h);
        };

        // Quantize each token's x to Q8_K once (reused across all rows). SHARED
        // buffer — written by the calling thread before parallelFor, read by workers.
        static std::vector<tinycoder::Q8KBlock> q8All;
        q8All.resize(static_cast<size_t>(seqLen) * blocksPerRow);
        for (uint32_t s = 0; s < seqLen; ++s) {
            GGMLDequantize::quantizeQ8K(X + static_cast<size_t>(s) * cols, cols,
                                        q8All.data() + static_cast<size_t>(s) * blocksPerRow);
        }

        // Scale shuffle table (q3k pattern): DIFFERENT from the Q2_K compact
        // kernel. Q3_K has 16 scales (one per 16-element half), so each 32-element
        // sub-block uses TWO scales: sc[2s] for its first 16 lanes (elements 0-15)
        // and sc[2s+1] for its last 16 lanes (elements 16-31) — matching the
        // scalar's sc[is++] per 16-element half. The 4 rows below operate on the
        // 32-byte scale vector {sc[0..7], sc[0..7]} (chunk 0) or {sc[8..15],
        // sc[8..15]} (chunk 1); each row broadcasts bytes (4s, 4s+1) = sc[2s] to
        // lanes 0-7 and bytes (4s+2, 4s+3) = sc[2s+1] to lanes 8-15.
        static const uint8_t k_shuffle[128] = {
                // sub 0: sc[0] -> lanes 0-7, sc[1] -> lanes 8-15
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                // sub 1: sc[2] -> lanes 0-7, sc[3] -> lanes 8-15
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                // sub 2: sc[4] -> lanes 0-7, sc[5] -> lanes 8-15
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                // sub 3: sc[6] -> lanes 0-7, sc[7] -> lanes 8-15
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
        };
        static const uint32_t kmask1 = 0x03030303;
        static const uint32_t kmask2 = 0x0f0f0f0f;
        const __m256i mask3 = _mm256_set1_epi8(0x03);
        const __m256i mask1 = _mm256_set1_epi8(0x01);
        const __m256i thirtytwo = _mm256_set1_epi16(32);

        // Process rows in tiles of BATCH_SIZE, parallelized over tiles.
        uint32_t numTiles = (rows + BATCH_SIZE - 1) / BATCH_SIZE;
        ThreadPool::instance().parallelForSlab(0, numTiles, [&](uint32_t tile) {
            uint32_t rowStart = tile * BATCH_SIZE;
            uint32_t batchSize = std::min(BATCH_SIZE, rows - rowStart);

            // Accumulators kept as __m256 in memory: [batchSize][seqLen].
            static thread_local Vector256 acc;
            acc.assign(static_cast<size_t>(batchSize) * seqLen,
                       _mm256_setzero_ps());

            // Per-block weight setup for a COMPACT Q3_K block: the fp16 super
            // scale, the 16 signed int16 scales (sc-32), the two shuffled scale
            // vectors, and the 8 w' byte planes (q2 - 4*hm + 4). Computed ONCE
            // per (block, row), then reused across all seqLen tokens.
            struct Q3KSetup {
                float dq;
                __m256i s16;// 16 signed int16 (sc[g]-32)
                __m256i scales_shuf[2];
                __m256i w[8];// 8 sub-blocks of 32 w' bytes (0..7)
            };
            auto makeSetup = [&](const uint8_t *blockData) -> Q3KSetup {
                Q3KSetup st;
                st.dq = halfToFloat(*(const uint16_t *) (blockData + 108));
                const uint8_t *hm = blockData;
                const uint8_t *q = blockData + 32;
                const uint8_t *scales = blockData + 96;

                // Unpack the 16 6-bit scales into 16 signed int8 (0..63), then
                // subtract 32 to match dl = d_all * (sc[is] - 32).
                uint32_t aux[4];
                std::memcpy(aux, scales, 12);
                uint32_t tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                const int8_t *sc = (const int8_t *) aux;

                __m256i s16 = _mm256_cvtepi8_epi16(
                        _mm_loadu_si128((const __m128i *) sc));
                st.s16 = _mm256_sub_epi16(s16, thirtytwo);
                __m128i slo = _mm256_extracti128_si256(st.s16, 0);
                __m128i shi = _mm256_extracti128_si256(st.s16, 1);
                st.scales_shuf[0] = _mm256_set_m128i(slo, slo);
                st.scales_shuf[1] = _mm256_set_m128i(shi, shi);

                // Load each unique source ONCE (3 vector loads) and derive all
                // eight w' planes in registers. Sub-blocks 0-3 come from
                // q[0..31] (bits [2*(s&3), 2*(s&3)+1]), sub-blocks 4-7 from
                // q[32..63]. hm is never advanced per chunk: bit s of each of the
                // 32 hm bytes selects this sub-block's sign, so the single hb
                // load is reused for all 8 planes with different shift amounts.
                // This removes 13 redundant L1 loads per (block, row) setup — the
                // Q3_K kernel recomputes the setup for every token on every block
                // (~1.5M setups/token), so load-port pressure is first-order.
                //
                // The scalar reference is quant = q2 - (hm_bit ? 0 : 4) =
                // q2 - 4 + 4*hm_bit, so w' = quant + 4 = q2 + 4*hm_bit (bytes
                // 0..7, unsigned-slot safe for _mm256_maddubs_epi16); the -4
                // folds through the bsum compensation in blockDot
                // (-4*Σ(sc-32)*bsum).
                __m256i qb0 = _mm256_loadu_si256((const __m256i *) q);
                __m256i qb1 = _mm256_loadu_si256((const __m256i *) (q + 32));
                __m256i hb = _mm256_loadu_si256((const __m256i *) hm);
                // Split into two fixed-source loops so the compiler never needs a
                // register-select (vblendvb) for qb0/qb1 and every vpsrlw uses an
                // immediate shift count.
                for (int s = 0; s < 4; ++s) {
                    __m256i q2 = _mm256_and_si256(
                            _mm256_srli_epi16(qb0, 2 * s), mask3);
                    __m256i hbit = _mm256_and_si256(
                            _mm256_srli_epi16(hb, s), mask1);
                    st.w[s] = _mm256_add_epi8(q2, _mm256_slli_epi16(hbit, 2));
                }
                for (int s = 4; s < 8; ++s) {
                    __m256i q2 = _mm256_and_si256(
                            _mm256_srli_epi16(qb1, 2 * (s - 4)), mask3);
                    __m256i hbit = _mm256_and_si256(
                            _mm256_srli_epi16(hb, s), mask1);
                    st.w[s] = _mm256_add_epi8(q2, _mm256_slli_epi16(hbit, 2));
                }
                return st;
            };

            // Compute the block contribution for one matrix row block:
            //   d_all * d * sum_s sub-block( (sc-32) * w' * qs_x ) - 4 * d_all * d * sum_g (sc[g]-32)*bsum[g]
            auto blockDot = [&](const Q3KSetup &st, const __m256i *xv,
                                const int16_t *bsums, float d) -> __m256 {
                // Offset term (folds the -4 of w_eff = w' - 4 through the bsums):
                //   -4 * d_all * d * sum_g (sc[g]-32)*bsum[g]
                //
                // IMPORTANT: keep this as a VECTOR of per-lane products. The
                // store code horizontally sums the 8 accumulator lanes, so a
                // scalar broadcast (_mm256_set1_ps) of the TOTAL would count it
                // 8x. Scaling each lane of bprod and summing at store counts it
                // exactly once (same trick as the Q2_K compact kernel's minTerm).
                __m256i bprod = _mm256_madd_epi16(
                        st.s16, _mm256_loadu_si256((const __m256i *) bsums));
                __m256 acc = _mm256_mul_ps(
                        _mm256_set1_ps(-4.0f * st.dq * d),
                        _mm256_cvtepi32_ps(bprod));
                // §3 Optimization: Hoist k_shuffle tables into registers.
                __m256i k_shuf[4] = {
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 0)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 32)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 64)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 96)),
                };

                for (int chunk = 0; chunk < 2; ++chunk) {
                    __m256i sumi = _mm256_setzero_si256();
                    for (int sub = 0; sub < 4; ++sub) {
                        int idx = chunk * 4 + sub;
                        // w' in the unsigned slot, x (signed int8) in the signed slot.
                        __m256i p = _mm256_maddubs_epi16(st.w[idx], xv[idx]);
                        __m256i shuf = k_shuf[sub];
                        p = _mm256_madd_epi16(
                                _mm256_shuffle_epi8(
                                        st.scales_shuf[chunk], shuf),
                                p);
                        sumi = _mm256_add_epi32(sumi, p);
                    }
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(st.dq * d),
                                          _mm256_cvtepi32_ps(sumi), acc);
                }
                return acc;
            };

            // §4 Optimization: Prefetch weights one block ahead to hide DRAM latency.
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                // Prefetch next block's Q3_K weights ~128 blocks ahead.
                {
                    uint32_t prefetchB = std::min(b + 128, blocksPerRow - 1);
                    const uint8_t *prefetchBlock = W_q3k +
                                                   static_cast<uint64_t>(rowStart) * rowStride +
                                                   static_cast<uint64_t>(prefetchB) * Q3K_BLOCK_BYTES;
                    _mm_prefetch((const char *) prefetchBlock, _MM_HINT_T0);
                }

                for (uint32_t r = 0; r < batchSize; ++r) {
                    const uint8_t *block =
                            W_q3k + static_cast<uint64_t>(rowStart + r) * rowStride +
                            static_cast<uint64_t>(b) * Q3K_BLOCK_BYTES;
                    Q3KSetup st = makeSetup(block);

                    for (uint32_t s = 0; s < seqLen; ++s) {
                        const tinycoder::Q8KBlock &q8 =
                                q8All[static_cast<size_t>(s) * blocksPerRow + b];
                        __m256i xv[8];
                        for (int i = 0; i < 8; ++i) {
                            xv[i] = _mm256_loadu_si256(
                                    (const __m256i *) (q8.qs + i * 32));
                        }
                        size_t accIdx = static_cast<size_t>(r) * seqLen + s;
                        acc[accIdx] = _mm256_add_ps(
                                acc[accIdx],
                                blockDot(st, xv, q8.bsums, q8.d));
                    }
                }
            }

            // Store results (horizontal sum each accumulator once).
            // Residual-fused store: when `residual` is non-null, fold it into
            // the epilogue (out[i] = dot + residual[i]). The attnO single-token
            // path passes hidden as BOTH out and residual, so this performs the
            // attention residual add in-place inside the kernel — eliminating
            // the attnProj buffer round-trip and the separate addSIMD pass over
            // the hidden vector (stage fusion: attnO + residual chain). The
            // branch is hoisted OUT of the store loop (residual is loop-
            // invariant) so the fused path has no per-iteration branch.
            if (residual != nullptr) {
                for (uint32_t r = 0; r < batchSize; ++r) {
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        __m256 v = acc[static_cast<size_t>(r) * seqLen + s];
                        __m128 hi = _mm256_extractf128_ps(v, 1);
                        __m128 lo = _mm256_castps256_ps128(v);
                        __m128 sum128 = _mm_hadd_ps(_mm_add_ps(lo, hi),
                                                    _mm_add_ps(lo, hi));
                        sum128 = _mm_hadd_ps(sum128, sum128);
                        size_t outIdx =
                                static_cast<size_t>(s) * rows + rowStart + r;
                        out[outIdx] = _mm_cvtss_f32(sum128) + residual[outIdx];
                    }
                }
            } else {
                for (uint32_t r = 0; r < batchSize; ++r) {
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        __m256 v = acc[static_cast<size_t>(r) * seqLen + s];
                        __m128 hi = _mm256_extractf128_ps(v, 1);
                        __m128 lo = _mm256_castps256_ps128(v);
                        __m128 sum128 = _mm_hadd_ps(_mm_add_ps(lo, hi),
                                                    _mm_add_ps(lo, hi));
                        sum128 = _mm_hadd_ps(sum128, sum128);
                        out[static_cast<size_t>(s) * rows + rowStart + r] =
                                _mm_cvtss_f32(sum128);
                    }
                }
            }
        });
    }

    // ---- AVX2 register-tiled batch GEMM for a single Q4_K matrix over a batch ----
    //
    // Prefill kernel. Computes out[s*rows + j] = sum_i X[s*cols + i] * W[j*cols + i]
    // for all tokens s and output rows j using Q4_K weights (attnV). The generic
    // path (dotProductQ4_K) runs a scalar double-precision loop per (row, token,
    // block); this kernel instead quantizes each token's x to Q8_K once, tiles
    // over 8 rows, and hoists the per-block Q4_K scale/min unpacking and nibble
    // expansion out of the token loop (weight-stationary), so that setup is done
    // once per (block, row) instead of once per (block, row, token). The inner
    // 32-element sub-block dots run through _mm256_maddubs_epi16 (Q4_K nibbles are
    // unsigned-safe in the unsigned operand slot; the Q8_K x values are signed).
    //
    // Per 256-block, Q4_K holds 8 sub-blocks of 32 elements. Sub-block i uses
    // qs[32*(i>>1)]..qs[32*(i>>1)+31], with the LOW nibbles for even i and the
    // HIGH nibbles for odd i, scaled by d*sc[i] minus dmin*m[i] (dotProductQ4_K
    // layout). Each sub-block contributes:
    //   xd * ( d*sc[i]*sum32(qs_x*nibble) - dmin*m[i]*sum32(qs_x) )
    // where sum32(qs_x) comes from the Q8KBlock bsums.
    void matMulVecBatchQ4K_Q8K_AVX2(const uint8_t *W_q4k, const float *X,
                                    uint32_t seqLen, uint32_t rows,
                                    uint32_t cols, float *out) {
        static constexpr uint32_t BATCH_SIZE = 8;
        static constexpr uint32_t BLOCK_SIZE = 256;
        static constexpr uint32_t Q4K_BLOCK_BYTES = 144;
        uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64_t rowStride = static_cast<uint64_t>(blocksPerRow) * Q4K_BLOCK_BYTES;

        // Branch-free fp16->fp32 via the shared LUT (this Q4_K batch kernel
        // calls it per (row, block) for d/dmin — 6 calls per Q4_K block).
        auto halfToFloat = [](uint16_t h) -> float {
            return tinycoder::GGMLDequantize::halfToFloatBranchFree(h);
        };

        // Quantize each token's x to Q8_K once (reused across all rows). SHARED
        // buffer, written by the calling thread before parallelFor, read by workers.
        static std::vector<tinycoder::Q8KBlock> q8All;
        q8All.resize(static_cast<size_t>(seqLen) * blocksPerRow);
        for (uint32_t s = 0; s < seqLen; ++s) {
            GGMLDequantize::quantizeQ8K(X + static_cast<size_t>(s) * cols, cols,
                                        q8All.data() + static_cast<size_t>(s) * blocksPerRow);
        }

        const __m256i one16 = _mm256_set1_epi16(1);
        const __m256i lowmask = _mm256_set1_epi8(0x0F);

        // Process rows in tiles of BATCH_SIZE, parallelized over tiles.
        uint32_t numTiles = (rows + BATCH_SIZE - 1) / BATCH_SIZE;
        ThreadPool::instance().parallelForSlab(0, numTiles, [&](uint32_t tile) {
            uint32_t rowStart = tile * BATCH_SIZE;
            uint32_t batchSize = std::min(BATCH_SIZE, rows - rowStart);

            // Per-output scalar accumulators [batchSize][seqLen].
            static thread_local std::vector<float> acc;
            acc.assign(static_cast<size_t>(batchSize) * seqLen, 0.0f);

            // Per-block weight setup, computed once per (block, row) and reused
            // across all tokens (weight-stationary): the 8 sub-block scales/mins
            // (float) and the 8 expanded nibble vectors.
            struct Q4KSetup {
                float ds[8];
                float mn[8];
                __m256i nv[8];
            };
            auto makeSetup = [&](const uint8_t *blockData) -> Q4KSetup {
                Q4KSetup st;
                const float d = halfToFloat(*(const uint16_t *) (blockData + 0));
                const float dmin = halfToFloat(*(const uint16_t *) (blockData + 2));
                const uint8_t *scales = blockData + 4;
                const uint8_t *qs = blockData + 16;
                for (int j = 0; j < 8; ++j) {
                    uint8_t sc, m;
                    if (j < 4) {
                        sc = scales[j] & 63;
                        m = scales[j + 4] & 63;
                    } else {
                        sc = (uint8_t) ((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
                        m = (uint8_t) ((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
                    }
                    st.ds[j] = d * static_cast<float>(sc);
                    st.mn[j] = dmin * static_cast<float>(m);
                }
                for (int i = 0; i < 8; ++i) {
                    __m256i b = _mm256_loadu_si256(
                            (const __m256i *) (qs + 32 * (i >> 1)));
                    if ((i & 1) == 0) {
                        st.nv[i] = _mm256_and_si256(b, lowmask);
                    } else {
                        st.nv[i] = _mm256_and_si256(_mm256_srli_epi16(b, 4), lowmask);
                    }
                }
                return st;
            };

            // §4 Optimization: Prefetch weights one block ahead.
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                {
                    uint32_t prefetchB = std::min(b + 128, blocksPerRow - 1);
                    const uint8_t *prefetchBlock = W_q4k +
                                                   static_cast<uint64_t>(rowStart) * rowStride +
                                                   static_cast<uint64_t>(prefetchB) * Q4K_BLOCK_BYTES;
                    _mm_prefetch((const char *) prefetchBlock, _MM_HINT_T0);
                }
                for (uint32_t r = 0; r < batchSize; ++r) {
                    const uint8_t *block =
                            W_q4k + static_cast<uint64_t>(rowStart + r) * rowStride +
                            static_cast<uint64_t>(b) * Q4K_BLOCK_BYTES;
                    Q4KSetup st = makeSetup(block);

                    for (uint32_t s = 0; s < seqLen; ++s) {
                        const tinycoder::Q8KBlock &q8 =
                                q8All[static_cast<size_t>(s) * blocksPerRow + b];
                        const float xd = q8.d;
                        const int16_t *bsums = q8.bsums;

                        __m256i xraw[8];
                        for (int i = 0; i < 8; ++i) {
                            xraw[i] = _mm256_loadu_si256(
                                    (const __m256i *) (q8.qs + i * 32));
                        }

                        // Main term: each sub-block i has its own scale ds[i], so
                        // reduce its 32-product sum to a scalar and scale it
                        // individually (the lanes of _mm256_madd_epi16 split a
                        // single sub-block into 4-element groups, not sub-blocks).
                        float mainS = 0.0f;
                        for (int i = 0; i < 8; ++i) {
                            __m256i p = _mm256_maddubs_epi16(st.nv[i], xraw[i]);
                            __m256i pi = _mm256_madd_epi16(p, one16);
                            __m128i lo = _mm256_castsi256_si128(pi);
                            __m128i hi = _mm256_extracti128_si256(pi, 1);
                            __m128i s = _mm_add_epi32(lo, hi);
                            s = _mm_hadd_epi32(s, s);
                            s = _mm_hadd_epi32(s, s);
                            int s32q = _mm_cvtsi128_si32(s);
                            mainS += st.ds[i] * static_cast<float>(s32q);
                        }

                        // Min term: sum_i mn[i] * sum32(qs_x).
                        float minS = 0.0f;
                        for (int i = 0; i < 8; ++i) {
                            minS += st.mn[i] *
                                    static_cast<float>(bsums[2 * i] + bsums[2 * i + 1]);
                        }

                        acc[static_cast<size_t>(r) * seqLen + s] +=
                                xd * (mainS - minS);
                    }
                }
            }

            // Store results.
            for (uint32_t r = 0; r < batchSize; ++r) {
                for (uint32_t s = 0; s < seqLen; ++s) {
                    out[static_cast<size_t>(s) * rows + rowStart + r] =
                            acc[static_cast<size_t>(r) * seqLen + s];
                }
            }
        });
    }

    // ---- AVX2 register-tiled batch GEMM for a single Q6_K matrix over a batch ----
    //
    // Generation-path kernel for the separate LM head (stored as Q6_K in the
    // model file). Computes out[s*rows + j] = sum_i X[s*cols + i] * W[j*cols + i]
    // for all tokens s and output rows j. Like the Q3_K/Q4_K kernels, each
    // token's x is quantized to Q8_K once (reused across all rows), rows are
    // tiled over 8 at a time, and the per-block Q6_K unpacking (scales, weight
    // planes) is hoisted out of the token loop (weight-stationary).
    //
    // Q6_K block layout (210 bytes per 256 elements = 0.82 B/elem vs 1.14 B/elem
    // for the Q8_K copy that would otherwise be streamed — ~28% less per-token
    // LM-head weight traffic):
    //   ql[128] @0    low 4 bits, 2 elems/byte
    //   qh[64]  @128  high 2 bits, 4 elems/byte
    //   sc[16]  @192  one SIGNED int8 scale per 16-element group
    //   d       @208  fp16 block scale
    // Sub-block s (8 sub-blocks of 32 elements) reads ql + 32*(s&1) + 64*(s>>2),
    // taking the LOW nibbles for s&2==0 (subs 0,1,4,5 — the scalar's q1/q2) or
    // the HIGH nibbles for s&2==2 (subs 2,3,6,7 — q3/q4), and qh + 32*(s>>2)
    // shifted by 2*(s&3): w' = nibble | ((qh >> 2*(s&3)) & 3) << 4 is the raw
    // 6-bit value (0..63), value = d*sc[2s or 2s+1]*(w' - 32). w' is
    // unsigned-safe in _mm256_maddubs_epi16's unsigned operand slot; the -32
    // folds through the per-lane bsum compensation (-32*sum_g sc[g]*bsum[g]),
    // the same vector-accumulator trick as the Q3_K kernel (never a scalar
    // broadcast — that counted the offset 8x and caused the Q3K bug).
    void matMulVecBatchQ6K_Q8K_AVX2(const uint8_t *W_q6k, const float *X,
                                    uint32_t seqLen, uint32_t rows,
                                    uint32_t cols, float *out) {
        static constexpr uint32_t BATCH_SIZE = 8;
        static constexpr uint32_t BLOCK_SIZE = 256;
        static constexpr uint32_t Q6K_BLOCK_BYTES = 210;
        uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64_t rowStride = static_cast<uint64_t>(blocksPerRow) * Q6K_BLOCK_BYTES;

        // Branch-free fp16->fp32 via the shared LUT (branch-per-call
        // elimination: this kernel calls it 2x per (row, block) — ~97k vocab
        // rows x 6 blocks x 2 per generated token).
        auto halfToFloat = [](uint16_t h) -> float {
            return tinycoder::GGMLDequantize::halfToFloatBranchFree(h);
        };

        // Quantize each token's x to Q8_K once (reused across all rows). SHARED
        // buffer — written by the calling thread before parallelFor, read by workers.
        static std::vector<tinycoder::Q8KBlock> q8All;
        q8All.resize(static_cast<size_t>(seqLen) * blocksPerRow);
        for (uint32_t s = 0; s < seqLen; ++s) {
            GGMLDequantize::quantizeQ8K(X + static_cast<size_t>(s) * cols, cols,
                                        q8All.data() + static_cast<size_t>(s) * blocksPerRow);
        }

        // Scale shuffle table (q3k pattern): Q6_K also has 16 scales (one per
        // 16-element group), so each 32-element sub-block uses TWO scales:
        // sc[2s] for its first 16 lanes and sc[2s+1] for its last 16 lanes.
        // The 4 rows below operate on the 32-byte scale vector {sc[0..7],
        // sc[0..7]} (chunk 0) or {sc[8..15], sc[8..15]} (chunk 1); each row
        // broadcasts bytes (4s, 4s+1) = sc[2s] to lanes 0-7 and bytes (4s+2,
        // 4s+3) = sc[2s+1] to lanes 8-15.
        static const uint8_t k_shuffle[128] = {
                // sub 0: sc[0] -> lanes 0-7, sc[1] -> lanes 8-15
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                // sub 1: sc[2] -> lanes 0-7, sc[3] -> lanes 8-15
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                // sub 2: sc[4] -> lanes 0-7, sc[5] -> lanes 8-15
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                // sub 3: sc[6] -> lanes 0-7, sc[7] -> lanes 8-15
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
        };
        const __m256i mask3 = _mm256_set1_epi8(0x03);
        const __m256i lowmask = _mm256_set1_epi8(0x0F);

        // Process rows in tiles of BATCH_SIZE, parallelized over tiles.
        uint32_t numTiles = (rows + BATCH_SIZE - 1) / BATCH_SIZE;
        ThreadPool::instance().parallelForSlab(0, numTiles, [&](uint32_t tile) {
            uint32_t rowStart = tile * BATCH_SIZE;
            uint32_t batchSize = std::min(BATCH_SIZE, rows - rowStart);

            // Accumulators kept as __m256 in memory: [batchSize][seqLen].
            static thread_local Vector256 acc;
            acc.assign(static_cast<size_t>(batchSize) * seqLen,
                       _mm256_setzero_ps());

            // Per-block weight setup for a COMPACT Q6_K block: the fp16 block
            // scale d, the 16 SIGNED int16 scales (raw sc[g], no -32 shift —
            // unlike Q3_K), the two shuffled scale vectors, and the 8 w' byte
            // planes (raw 6-bit 0..63). Computed ONCE per (block, row), then
            // reused across all seqLen tokens.
            struct Q6KSetup {
                float d;               // fp16 block scale
                __m256i s16;           // 16 signed int16 scales sc[0..15]
                __m256i scales_shuf[2];// chunk c = {sc[8c..8c+7] x2}
                __m256i w[8];          // 8 sub-blocks of 32 w' bytes (0..63)
            };
            auto makeSetup = [&](const uint8_t *blockData) -> Q6KSetup {
                Q6KSetup st;
                st.d = halfToFloat(*(const uint16_t *) (blockData + 208));
                const uint8_t *ql = blockData;
                const uint8_t *qh = blockData + 128;
                const int8_t *sc = reinterpret_cast<const int8_t *>(blockData + 192);

                // Sign-extend the 16 raw int8 scales to int16 for both the bsum
                // compensation (madd_epi16) and the shuffle broadcast.
                __m256i s16 = _mm256_cvtepi8_epi16(
                        _mm_loadu_si128((const __m128i *) sc));
                st.s16 = s16;
                __m128i slo = _mm256_extracti128_si256(s16, 0);
                __m128i shi = _mm256_extracti128_si256(s16, 1);
                st.scales_shuf[0] = _mm256_set_m128i(slo, slo);
                st.scales_shuf[1] = _mm256_set_m128i(shi, shi);

                for (int s = 0; s < 8; ++s) {
                    const uint8_t *qsrc = ql + 32 * (s & 1) + 64 * (s >> 2);
                    __m256i qb = _mm256_loadu_si256((const __m256i *) qsrc);
                    __m256i qlq;
                    if ((s & 2) == 0) {
                        // subs 0,1,4,5: the scalar's q1/q2 low-nibble pairs
                        qlq = _mm256_and_si256(qb, lowmask);
                    } else {
                        // subs 2,3,6,7: the scalar's q3/q4 high-nibble pairs
                        qlq = _mm256_and_si256(_mm256_srli_epi16(qb, 4), lowmask);
                    }
                    const uint8_t *qhsrc = qh + 32 * (s >> 2);
                    __m256i qhb = _mm256_loadu_si256((const __m256i *) qhsrc);
                    __m256i qhi = _mm256_and_si256(
                            _mm256_srli_epi16(qhb, 2 * (s & 3)), mask3);
                    st.w[s] = _mm256_or_si256(qlq, _mm256_slli_epi16(qhi, 4));
                }
                return st;
            };

            // Compute the block contribution for one matrix row block:
            //   d * d8 * sum_s sub-block( sc * w' * qs_x ) - 32 * d * d8 * sum_g sc[g]*bsum[g]
            // L7: the k_shuffle table is hoisted to the TILE level (identical
            // for every block/row of the matrix) instead of being reloaded per
            // (row, block) — ~97k vocab rows × 6 blocks × 4 loads per token.
            __m256i k_shuf[4] = {
                    _mm256_loadu_si256((const __m256i *) (k_shuffle + 0)),
                    _mm256_loadu_si256((const __m256i *) (k_shuffle + 32)),
                    _mm256_loadu_si256((const __m256i *) (k_shuffle + 64)),
                    _mm256_loadu_si256((const __m256i *) (k_shuffle + 96)),
            };
            auto blockDot = [&](const Q6KSetup &st, const __m256i *xv,
                                const int16_t *bsums, float d8) -> __m256 {
                // Offset term (folds the -32 of w_eff = w' - 32 through the
                // bsums): -32 * d * d8 * sum_g sc[g]*bsum[g].
                //
                // IMPORTANT: keep this as a VECTOR of per-lane products. The
                // store code horizontally sums the 8 accumulator lanes, so a
                // scalar broadcast (_mm256_set1_ps) of the TOTAL would count it
                // 8x (the Q3K bug). Scaling each lane of bprod and summing at
                // store counts it exactly once.
                __m256i bprod = _mm256_madd_epi16(
                        st.s16, _mm256_loadu_si256((const __m256i *) bsums));
                __m256 acc = _mm256_mul_ps(
                        _mm256_set1_ps(-32.0f * st.d * d8),
                        _mm256_cvtepi32_ps(bprod));

                for (int chunk = 0; chunk < 2; ++chunk) {
                    __m256i sumi = _mm256_setzero_si256();
                    for (int sub = 0; sub < 4; ++sub) {
                        int idx = chunk * 4 + sub;
                        // w' in the unsigned slot, x (signed int8) in the signed slot.
                        __m256i p = _mm256_maddubs_epi16(st.w[idx], xv[idx]);
                        __m256i shuf = k_shuf[sub];
                        p = _mm256_madd_epi16(
                                _mm256_shuffle_epi8(
                                        st.scales_shuf[chunk], shuf),
                                p);
                        sumi = _mm256_add_epi32(sumi, p);
                    }
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(st.d * d8),
                                          _mm256_cvtepi32_ps(sumi), acc);
                }
                return acc;
            };

            // §5 Optimization: ROW-OUTER / block-inner sequential streaming.
            // The old b-outer/r-inner order opened 6 interleaved strided weight
            // streams per 8-row tile, defeating the HW prefetcher on DRAM-bound
            // generation (same lesson the fused Q3_K phase-2 rewrite taught:
            // b-outer -> r-outer recovered 24.5->21.6 ms/t there). Streaming one
            // row's ~1.2 KB sequentially per tile maximizes DRAM page hits, and
            // the per-tile __m256 acc[] keeps the horizontal sums until the store.
            for (uint32_t r = 0; r < batchSize; ++r) {
                // Sequential row base: start block 0 of this row.
                const uint8_t *rowBase = W_q6k +
                                         static_cast<uint64_t>(rowStart + r) * rowStride;
                if (r + 1 < batchSize) {
                    // Prefetch the NEXT tile's first blocks (this tile's row
                    // streams are already sequential, so the prefetcher handles
                    // the in-row stream; prefetch across the tile boundary).
                    _mm_prefetch((const char *) (rowBase + rowStride), _MM_HINT_T0);
                }
                Q6KSetup st;
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    const uint8_t *block = rowBase +
                                           static_cast<uint64_t>(b) * Q6K_BLOCK_BYTES;
                    // Prefetch 2 blocks ahead within this row (the sequential
                    // stream is prefetch-friendly, 210 B granularity).
                    if (b + 2 < blocksPerRow) {
                        _mm_prefetch(
                                (const char *) (block + 2 * Q6K_BLOCK_BYTES), _MM_HINT_T0);
                    }
                    st = makeSetup(block);

                    for (uint32_t s = 0; s < seqLen; ++s) {
                        const tinycoder::Q8KBlock &q8 =
                                q8All[static_cast<size_t>(s) * blocksPerRow + b];
                        __m256i xv[8];
                        for (int i = 0; i < 8; ++i) {
                            xv[i] = _mm256_loadu_si256(
                                    (const __m256i *) (q8.qs + i * 32));
                        }
                        size_t accIdx = static_cast<size_t>(r) * seqLen + s;
                        acc[accIdx] = _mm256_add_ps(
                                acc[accIdx],
                                blockDot(st, xv, q8.bsums, q8.d));
                    }
                }
            }

            // Store results (horizontal sum each accumulator once).
            for (uint32_t r = 0; r < batchSize; ++r) {
                for (uint32_t s = 0; s < seqLen; ++s) {
                    __m256 v = acc[static_cast<size_t>(r) * seqLen + s];
                    __m128 hi = _mm256_extractf128_ps(v, 1);
                    __m128 lo = _mm256_castps256_ps128(v);
                    __m128 sum128 = _mm_hadd_ps(_mm_add_ps(lo, hi),
                                                _mm_add_ps(lo, hi));
                    sum128 = _mm_hadd_ps(sum128, sum128);
                    out[static_cast<size_t>(s) * rows + rowStart + r] =
                            _mm_cvtss_f32(sum128);
                }
            }
        });
    }

    // ---- AVX2 register-tiled batch GEMM for COMPACT (raw GGUF) Q2_K weights ----
    // Used for the separate LM head after the load-time Q2_K re-quant (Lever C:
    // 84 B/block vs Q6_K's 210 B/block, ~191->76 MB/token of LM-head weight
    // traffic). The per-block setup (fp16 d/dmin, 4-bit scale/min nibbles, 8
    // unpacked 2-bit planes) is hoisted out of the token loop
    // (weight-stationary) and the inner 32-element sub-block dots run through
    // _mm256_maddubs_epi16 against the Q8_K x-vector (the min term folds through
    // the Q8KBlock bsums) — the same machinery as
    // matMulVecFusedGateUpQ2_K_Compact_Q8_AVX2's blockDot, but streaming a
    // single LM-head matrix over a batch of tokens. Accumulators stay in __m256
    // registers; out is written [seqLen, rows] row-major to match the other
    // LM-head batch kernels.
    void matMulVecBatchQ2K_Compact_Q8K_AVX2(const uint8_t *W_q2k,
                                            const float *X,
                                            uint32_t seqLen, uint32_t rows,
                                            uint32_t cols, float *out) {
        static constexpr uint32_t BATCH_SIZE = 8;
        static constexpr uint32_t BLOCK_SIZE = 256;
        static constexpr uint32_t COMPACT_BLOCK_BYTES = 84;
        uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64_t rowStrideBytes = static_cast<uint64_t>(blocksPerRow) * COMPACT_BLOCK_BYTES;

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

        // Quantize each token's x to Q8_K once (reused across all rows). SHARED
        // buffer — written by the calling thread before parallelFor, read by workers.
        static std::vector<tinycoder::Q8KBlock> q8All;
        q8All.resize(static_cast<size_t>(seqLen) * blocksPerRow);
        for (uint32_t s = 0; s < seqLen; ++s) {
            GGMLDequantize::quantizeQ8K(X + static_cast<size_t>(s) * cols, cols,
                                        q8All.data() + static_cast<size_t>(s) * blocksPerRow);
        }

        // Scale shuffle table (q2k compact pattern): for each 16-element group,
        // duplicate its (scale, next-scale) 16-bit pair across 8 int16 lanes.
        // Identical to the fused gate+up compact kernel (element order matches
        // the compact block's 4-bit scale nibble expansion).
        static const uint8_t k_shuffle[128] = {
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                2,
                3,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                4,
                5,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                6,
                7,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                8,
                9,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                10,
                11,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                12,
                13,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
                14,
                15,
        };
        const __m256i mask3 = _mm256_set1_epi8(0x03);

        // Process rows in tiles of BATCH_SIZE, parallelized over tiles.
        uint32_t numTiles = (rows + BATCH_SIZE - 1) / BATCH_SIZE;
        ThreadPool::instance().parallelForSlab(0, numTiles, [&](uint32_t tile) {
            uint32_t rowStart = tile * BATCH_SIZE;
            uint32_t batchSize = std::min(BATCH_SIZE, rows - rowStart);

            // Accumulators kept as __m256 in memory: [batchSize][seqLen].
            static thread_local Vector256 acc;
            acc.assign(static_cast<size_t>(batchSize) * seqLen,
                       _mm256_setzero_ps());

            // Per-block weight setup for a COMPACT Q2_K block: the fp16 d/dmin,
            // the min nibbles (int16), the two shuffled scale vectors, and the
            // 8 unpacked q2 planes (256 expanded values). Computed ONCE per
            // (block column, row), then reused across all seqLen tokens.
            struct QKSetup {
                float dq;
                float ndmin;
                __m256i mins;
                __m256i scales_shuf[2];
                __m256i q2[8];
            };
            auto makeSetup = [&](const uint8_t *blockData) -> QKSetup {
                QKSetup st;
                st.dq = halfToFloat(*(const uint16_t *) (blockData + 80));
                st.ndmin = -halfToFloat(*(const uint16_t *) (blockData + 82));
                const uint8_t *scales = blockData;
                // Min term mins (high nibbles).
                __m128i ms = _mm_loadu_si128((const __m128i *) scales);
                __m128i mins8 = _mm_and_si128(_mm_srli_epi16(ms, 4), _mm_set1_epi8(0xF));
                st.mins = _mm256_cvtepi8_epi16(mins8);
                // Scales (low nibbles) -> shuffled scale vectors.
                __m128i scales8 = _mm_and_si128(ms, _mm_set1_epi8(0xF));
                __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
                __m128i slo = _mm256_extracti128_si256(scales16, 0);
                __m128i shi = _mm256_extracti128_si256(scales16, 1);
                st.scales_shuf[0] = _mm256_set_m128i(slo, slo);
                st.scales_shuf[1] = _mm256_set_m128i(shi, shi);
                // Unpack the 2-bit planes (byte-identical to prepackQ2_K's
                // expansion, so the scale/min shuffle math stays valid).
                for (int h = 0; h < 2; ++h) {
                    const uint8_t *qsrc = blockData + 16 + h * 32;
                    __m256i b = _mm256_loadu_si256((const __m256i *) qsrc);
                    st.q2[h * 4 + 0] = _mm256_and_si256(b, mask3);
                    __m256i s2 = _mm256_srli_epi16(b, 2);
                    st.q2[h * 4 + 1] = _mm256_and_si256(s2, mask3);
                    __m256i s4 = _mm256_srli_epi16(s2, 2);
                    st.q2[h * 4 + 2] = _mm256_and_si256(s4, mask3);
                    __m256i s6 = _mm256_srli_epi16(s4, 2);
                    st.q2[h * 4 + 3] = _mm256_and_si256(s6, mask3);
                }
                return st;
            };

            // Compute the block contribution for one matrix row block:
            //   dq * d * sum_g sc[g]*(q2 . q8)_g - dmin * d * sum_g mn[g]*bsum[g]
            auto blockDot = [&](const QKSetup &st, const __m256i *q8_ymm,
                                const int16_t *bsums, float d) -> __m256 {
                // Min term: -dmin * d * sum(mins * bsums). Kept as a per-lane
                // vector so the horizontal sum at store counts it exactly once.
                __m256i prod = _mm256_madd_epi16(
                        st.mins, _mm256_loadu_si256((const __m256i *) bsums));
                __m256 minTerm = _mm256_fmadd_ps(
                        _mm256_set1_ps(st.ndmin * d), _mm256_cvtepi32_ps(prod),
                        _mm256_setzero_ps());

                // §3 Optimization: Hoist k_shuffle tables into registers.
                __m256i k_shuf[4] = {
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 0)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 32)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 64)),
                        _mm256_loadu_si256((const __m256i *) (k_shuffle + 96)),
                };

                __m256 acc = _mm256_setzero_ps();
                // 2 chunks of 128 elements each (256 total)
                for (int chunk = 0; chunk < 2; ++chunk) {
                    __m256i sumi = _mm256_setzero_si256();
                    for (int sub = 0; sub < 4; ++sub) {
                        int idx = chunk * 4 + sub;
                        // q2 (0..3 unsigned) in the unsigned slot, x (int8) in
                        // the signed slot.
                        __m256i p = _mm256_maddubs_epi16(st.q2[idx], q8_ymm[idx]);
                        __m256i shuf = k_shuf[sub];
                        p = _mm256_madd_epi16(
                                _mm256_shuffle_epi8(st.scales_shuf[chunk], shuf), p);
                        sumi = _mm256_add_epi32(sumi, p);
                    }
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(st.dq * d),
                                          _mm256_cvtepi32_ps(sumi), acc);
                }
                return _mm256_add_ps(acc, minTerm);
            };

            // §4 Optimization: Prefetch weights one block ahead to hide DRAM latency.
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                {
                    uint32_t prefetchB = std::min(b + 128, blocksPerRow - 1);
                    const uint8_t *prefetchBlock = W_q2k +
                                                   static_cast<uint64_t>(rowStart) * rowStrideBytes +
                                                   static_cast<uint64_t>(prefetchB) * COMPACT_BLOCK_BYTES;
                    _mm_prefetch((const char *) prefetchBlock, _MM_HINT_T0);
                }
                for (uint32_t r = 0; r < batchSize; ++r) {
                    const uint8_t *block =
                            W_q2k + static_cast<uint64_t>(rowStart + r) * rowStrideBytes +
                            static_cast<uint64_t>(b) * COMPACT_BLOCK_BYTES;
                    QKSetup st = makeSetup(block);

                    // Q8_K x data loaded once per (block, token), reused across
                    // the batchSize rows in this tile.
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        const tinycoder::Q8KBlock &q8 =
                                q8All[static_cast<size_t>(s) * blocksPerRow + b];
                        __m256i xv[8];
                        for (int i = 0; i < 8; ++i) {
                            xv[i] = _mm256_loadu_si256(
                                    (const __m256i *) (q8.qs + i * 32));
                        }
                        size_t accIdx = static_cast<size_t>(r) * seqLen + s;
                        acc[accIdx] = _mm256_add_ps(
                                acc[accIdx],
                                blockDot(st, xv, q8.bsums, q8.d));
                    }
                }
            }

            // Store results (horizontal sum each accumulator once).
            for (uint32_t r = 0; r < batchSize; ++r) {
                for (uint32_t s = 0; s < seqLen; ++s) {
                    __m256 v = acc[static_cast<size_t>(r) * seqLen + s];
                    __m128 hi = _mm256_extractf128_ps(v, 1);
                    __m128 lo = _mm256_castps256_ps128(v);
                    __m128 sum128 = _mm_hadd_ps(_mm_add_ps(lo, hi),
                                                _mm_add_ps(lo, hi));
                    sum128 = _mm_hadd_ps(sum128, sum128);
                    out[static_cast<size_t>(s) * rows + rowStart + r] =
                            _mm_cvtss_f32(sum128);
                }
            }
        });
    }

    // ---- AVX2 Q8_K × Q8_K dot product ----
    // Computes dot(x, w) where both x and w are Q8_K blocks (256 int8 values +
    // a block scale). Uses _mm256_maddubs_epi16 (32 int8×int8->int16
    // multiply-adds per instruction) instead of float FMAs (8 per instruction).
    // Result = x.d * w.d * sum(x.qs[i] * w.qs[i]).
    float dotProductQ8K_Q8K_AVX2(const tinycoder::Q8KBlock *x,
                                 const tinycoder::Q8KBlock *w) {
        __m256i sumi = _mm256_setzero_si256();
        for (int i = 0; i < 256; i += 32) {
            // _mm256_maddubs_epi16 treats its FIRST operand as unsigned bytes
            // and the second as signed bytes. Both qs arrays are signed int8,
            // so the raw values cannot be fed into the unsigned slot: a negative
            // byte there would be reinterpreted as value+256, corrupting the
            // product (e.g. x=-5, w=3 would compute 251*3 instead of -15).
            //
            // Fix (same trick as llama.cpp): substitute (x, w) with
            // (|x|, w*sign(x)). |x| is non-negative so it is safe in the
            // unsigned slot, and |x| * (w*sign(x)) == x*w exactly. Per-byte
            // products are bounded by 127*127, so the int16 intermediate in
            // maddubs cannot saturate.
            __m256i xv = _mm256_loadu_si256((const __m256i *) &x->qs[i]);
            __m256i wv = _mm256_loadu_si256((const __m256i *) &w->qs[i]);
            __m256i xabs = _mm256_sign_epi8(xv, xv);
            __m256i wsig = _mm256_sign_epi8(wv, xv);
            __m256i p = _mm256_maddubs_epi16(xabs, wsig);
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
        }
        __m128i lo = _mm256_castsi256_si128(sumi);
        __m128i hi = _mm256_extracti128_si256(sumi, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        int sum = _mm_cvtsi128_si32(sum128);
        return x->d * w->d * static_cast<float>(sum);
    }

    // ---- AVX2 fused gate+up+down FFN kernel for single-token generation ----
    //
    // Combines gate=Q2_K, up=Q2_K, down=Q3_K into a single kernel that
    // computes: out = residual + down @ (silu(gate)*up)
    // where gate = x * W_gate^T, up = x * W_up^T (both Q2_K),
    // and down = act * W_down^T (act=silu(gate)*up, W_down is Q3_K).
    //
    // Algorithm (two phases; phase 1 and phase 2 are separated by the pool
    // barrier so each is a single parallelForSlab with thread-local state):
    //   Phase 1: For each row j [0, gRows), compute gate_j and up_j using the
    //            Q2_K compact decomposition (Q8_K-quantize x once for all
    //            blocksPerRow, tile over 8 rows, dot via _mm256_maddubs_epi16),
    //            then apply silu(gate_j) * up_j = act_j.
    //   Phase 2: Run Q3_K ffnDown on 'act' (Q8_K-quantize the act vector once
    //            per block, shared across all hRows), folding 'residual' into
    //            the store epilogue.
    //
    // NOTE: the phase-1 inner loop must NOT read x directly. The call site
    // enables this kernel only when cols % 256 == 0 (a multiple of the Q2_K
    // block size); when that holds, phase 1 reproduces the verified compact
    // gate+up kernel (matMulVecFusedGateUpQ2_K_Compact_Q8_AVX2) exactly.
    template<bool DownIsQ2K>
    void matMulVecFusedGateUpDownQ2K_Compact_AVX2_impl(
            const uint8_t *gateData,
            const uint8_t *upData,
            const uint8_t *downData,
            const float *x,
            uint32_t gRows,
            uint32_t hRows,
            uint32_t cols,
            float *out,
            float *residual) {
        static constexpr uint32_t BATCH_SIZE = 16;
        static constexpr uint32_t BLOCK_SIZE = 256;
        static constexpr uint32_t Q2K_COMPACT_BLOCK_BYTES = 84;
        static constexpr uint32_t Q3K_BLOCK_BYTES = 110;
        // ffnDown row stride (bytes): Q3_K source 110 B/block vs load-time
        // Q2_K re-quant 84 B/block (Lever C).
        static constexpr uint32_t DOWN_BLOCK_BYTES =
                DownIsQ2K ? Q2K_COMPACT_BLOCK_BYTES : Q3K_BLOCK_BYTES;

        // Fixed-point table for branch-free fp16->fp32 (the fused kernel calls
        // this per (row, block) for gate+up+down — the table is branch-free and
        // L1-resident, removing the unpredictable branches + shifts from the
        // hot loop). One 64 KiB table covers all 65536 fp16 bit patterns; the
        // output is bit-exact vs the scalar halfToFloat above.
        //
        // NOTE (measurement): this is an experiment to establish whether the
        // dominant fused FFN stage is ALU-bound (per-block makeSetup) or
        // DRAM-bound. If gen time drops, per-block setup is the wall and we
        // should pursue the deeper pre-flight re-layout (prepacked 8-row tiles).
        // If it stays flat, we are DRAM-bound and the table only adds footprint.
        static const float *s_h2fTable = []() -> const float * {
            static std::vector<float> table(65536);
            for (uint32_t i = 0; i < 65536; ++i) {
                // Replicate the scalar conversion exactly.
                uint16_t h = static_cast<uint16_t>(i);
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
                std::memcpy(&table[i], &f32, sizeof(float));
            }
            return table.data();
        }();
        auto halfToFloat = [](uint16_t h) -> float {
            return s_h2fTable[h];
        };

        // Dimension mapping:
        //   gRows = intermediateSize (gate/up output dimension)
        //   hRows = hiddenSize (down output dimension)
        //   cols = hiddenSize (input dimension to gate+up)
        //   down has shape [hRows x gRows] (rows x cols_in_down)
        uint32_t gBlocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint32_t dBlocksPerRow = (gRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64_t gRowStride = static_cast<uint64_t>(gBlocksPerRow) * Q2K_COMPACT_BLOCK_BYTES;
        uint64_t hRowStride = static_cast<uint64_t>(dBlocksPerRow) * DOWN_BLOCK_BYTES;

        // Q8_K-quantized form of the shared act vector (phase-2 input). The
        // `between` step (main thread) fills it after phase 1; phase-2 workers
        // read it. Declared at kernel scope so both lambdas see it.
        struct ActQ8Block {
            float d;
            int8_t qs[256];
            int16_t bsums[16];
        };
        static std::vector<ActQ8Block> q8actBuf;
        if (q8actBuf.size() < static_cast<size_t>(dBlocksPerRow)) {
            q8actBuf.resize(dBlocksPerRow);
        }

        // SHARED activation buffer (like the reference kernels' q8All): written
        // by the phase-1 workers (each writes only its own slab's disjoint rows
        // — no cross-thread write race), then read by the main thread's phase-2
        // quantize loop. The pool barrier between the two parallelForSlab calls
        // orders the writes before phase-2 reads. A thread_local buffer would be
        // wrong here: each worker's slab writes are invisible to other threads.
        static std::vector<float> actBuf;
        if (actBuf.size() < static_cast<size_t>(gRows)) {
            actBuf.resize(gRows);
        }
        float *act = actBuf.data();

#if defined(TINYCODER_USE_COOP_QUANTIZE)
        // Cooperative act→Q8_K quantize machinery — llama.cpp `from_float`
        // LAST-ARRIVER pattern (plans/generation_optimizations.md lever O1
        // re-attempt at 4 threads, 2026-08-28).
        //
        // Each 256-row Q8_K block of `act` is produced by 16 phase-1 tiles
        // (BATCH_SIZE == 16 rows each). Instead of the serial `between`
        // callback (main thread quantizes ALL blocks after barrier-1 while
        // every worker spins on phase2Ready_), every phase-1 tile increments
        // its block's arrival counter as soon as it stores; the 16th (LAST)
        // arriver runs that block's quantize inline. The quantize ALU thus
        // overlaps the rest of phase-1's DRAM streaming on other threads
        // instead of forming a serial bubble between the phases — llama.cpp's
        // from_float does exactly this per block with a `++cnt == nb` check.
        //
        // Ordering: the acq_rel fetch_add chain orders the other 15 tiles'
        // act[] writes before the last arriver's quantize reads; the pool's
        // barrier-1 + phase1Arrived_ orders every quantize write before any
        // phase-2 reader (phase2Ready_ release → worker acquire). Phase 2 is
        // still gated on the full barrier because every phase-2 down row
        // consumes all dBlocksPerRow act blocks.
        // Fixed-size heap array (std::atomic is non-movable, so a std::vector
        // cannot reallocate it — unique_ptr arrays carry no move/copy cost).
        static std::unique_ptr<std::atomic<uint32_t>[]> actArrivals;
        static uint32_t actArrivalsSize = 0;
        if (actArrivalsSize < dBlocksPerRow) {
            actArrivals.reset(new std::atomic<uint32_t>[dBlocksPerRow]);
            actArrivalsSize = dBlocksPerRow;
        }
        for (uint32_t ab = 0; ab < dBlocksPerRow; ++ab) {
            actArrivals[ab].store(0, std::memory_order_relaxed);
        }
        // Phase-1 tile count per act block: BLOCK_SIZE/BATCH_SIZE (16), except
        // a partial tail block when gRows % BLOCK_SIZE != 0.
        static std::vector<uint32_t> actArrivalExpected;
        actArrivalExpected.resize(dBlocksPerRow);
        for (uint32_t ab = 0; ab < dBlocksPerRow; ++ab) {
            const uint32_t base = ab * BLOCK_SIZE;
            const uint32_t rows = std::min(BLOCK_SIZE, gRows - base);
            actArrivalExpected[ab] = (rows + BATCH_SIZE - 1) / BATCH_SIZE;
        }

        // Per-block act → Q8_K quantize, bit-identical to the scalar reference
        // math (this was the `between` loop body; now the LAST ARRIVER calls
        // it inline from phase 1 — the serial `between` fallback keeps its own
        // byte-identical copy under #else below).
        auto quantizeActBlock = [&](uint32_t b) {
            // Vectorized act → Q8_K quantize (AVX2), bit-identical to the
            // scalar reference math:
            //   amax = max |x|;  maxVal = first x with |x|==amax
            //   iscale = -127/maxVal;  q = clamp(lrintf(iscale*x))
            // The earlier corruption was a pack bug: vpacks operates in 128-bit
            // lanes, so packs_epi32(vi,vi) + packs_epi16(p,p) stored only lanes
            // 0..3 (4 bytes) per 8-element group. Fixed by narrowing through the
            // two 128-bit halves with _mm_packs_epi32(lo,hi) -> 8 int16 ->
            // _mm_packs_epi16 -> low 64 bits = 8 contiguous int8. Rounding:
            // vcvtps2dq honors the MXCSR rounding mode (RN by default) exactly
            // like lrintf, so q values match bit-for-bit.
            const __m256 vabsMask = _mm256_castsi256_ps(
                    _mm256_set1_epi32(static_cast<int>(0x7FFFFFFF)));
            const uint32_t base = b * BLOCK_SIZE;
            const uint32_t n = std::min(BLOCK_SIZE, gRows - base);
            const float *src = act + base;
            int8_t *qs = q8actBuf[b].qs;

            // ---- Pass 1: |x| max (vectorized max-of-8) ----
            float amaxVal = 0.0f, maxVal = 0.0f;
            uint32_t i = 0;
            for (; i + 8 <= n; i += 8) {
                const __m256 v = _mm256_loadu_ps(src + i);
                const __m256 av = _mm256_and_ps(v, vabsMask);// |x|
                __m128 lo = _mm256_castps256_ps128(av);
                __m128 hi = _mm256_extractf128_ps(av, 1);
                __m128 m = _mm_max_ps(lo, hi);
                m = _mm_max_ps(m, _mm_shuffle_ps(m, m, 0xB1));// [1,0,3,2]
                m = _mm_max_ps(m, _mm_shuffle_ps(m, m, 0x4E));// [2,3,0,1]
                const float mv = _mm_cvtss_f32(m);
                if (mv > amaxVal) {
                    amaxVal = mv;
                }
            }
            for (; i < n; ++i) {
                const float ax = std::fabs(src[i]);
                if (ax > amaxVal) {
                    amaxVal = ax;
                }
            }
            // Sign resolution (first element achieving max |x|).
            for (uint32_t k = 0; k < n; ++k) {
                if (std::fabs(src[k]) == amaxVal) {
                    maxVal = src[k];
                    break;
                }
            }
            if (amaxVal == 0.0f) {
                q8actBuf[b].d = 0.0f;
                std::memset(qs, 0, BLOCK_SIZE);
                std::memset(q8actBuf[b].bsums, 0, sizeof(q8actBuf[b].bsums));
                return;
            }
            const float iscale = -127.0f / maxVal;
            q8actBuf[b].d = 1.0f / iscale;

            // ---- Pass 2: q = clamp(lrintf(iscale*x)) via vcvtps2dq ----
            const __m256 viscale = _mm256_set1_ps(iscale);
            const __m256 v127 = _mm256_set1_ps(127.0f);
            const __m256 vm128 = _mm256_set1_ps(-128.0f);
            i = 0;
            for (; i + 8 <= n; i += 8) {
                __m256 v = _mm256_loadu_ps(src + i);
                v = _mm256_mul_ps(v, viscale);
                v = _mm256_min_ps(_mm256_max_ps(v, vm128), v127);
                const __m256i vi = _mm256_cvtps_epi32(v);
                // Narrow 8xint32 -> 8xint8 correctly across the 128-bit lane
                // boundary.
                const __m128i lo32 = _mm256_castsi256_si128(vi);
                const __m128i hi32 = _mm256_extracti128_si256(vi, 1);
                const __m128i p16 = _mm_packs_epi32(lo32, hi32);// 8xint16
                const __m128i p8 = _mm_packs_epi16(p16, p16);   // low64 = 8xint8
                _mm_storel_epi64(reinterpret_cast<__m128i *>(qs + i), p8);
            }
            for (; i < n; ++i) {
                const int v = static_cast<int>(std::lrintf(iscale * src[i]));
                qs[i] = static_cast<int8_t>(std::min(127, std::max(-128, v)));
            }
            if (n < BLOCK_SIZE) {
                std::memset(qs + n, 0, BLOCK_SIZE - n);
            }
            // Vectorized per-16-group byte sums (AVX2). Each group is 16 int8;
            // two adjacent groups (32 bytes) fit one ymm:
            //   _mm256_maddubs_epi16(1, v) -> 16 int16 lanes,
            //     lane k = v[2k] + v[2k+1]
            //   _mm256_madd_epi16(p, 1)   -> 8 int32 lanes,
            //     lane k = bytes v[4k..4k+3]
            // Lanes 0-3 = bytes 0..15 (group j), lanes 4-7 = bytes 16..31
            // (group j+1). Each group sums to <= 2032, well inside int16, so
            // the int32 accumulation is exact and identical to the scalar
            // reference (parity checked).
            int j = 0;
            for (; j + 2 <= 16; j += 2) {
                const __m256i v = _mm256_loadu_si256(
                        (const __m256i *) (qs + j * 16));
                const __m256i p = _mm256_maddubs_epi16(
                        _mm256_set1_epi8(1), v);
                const __m256i s32 = _mm256_madd_epi16(
                        p, _mm256_set1_epi16(1));
                const __m128i loS = _mm256_castsi256_si128(s32);
                const __m128i hiS = _mm256_extracti128_si256(s32, 1);
                __m128i bj = _mm_hadd_epi32(loS, loS);
                bj = _mm_hadd_epi32(bj, bj);
                __m128i bj1 = _mm_hadd_epi32(hiS, hiS);
                bj1 = _mm_hadd_epi32(bj1, bj1);
                q8actBuf[b].bsums[j] =
                        static_cast<int16_t>(_mm_cvtsi128_si32(bj));
                q8actBuf[b].bsums[j + 1] =
                        static_cast<int16_t>(_mm_cvtsi128_si32(bj1));
            }
            for (; j < 16; ++j) {
                int s = 0;
                for (int ii = 0; ii < 16; ++ii) {
                    s += qs[j * 16 + ii];
                }
                q8actBuf[b].bsums[j] = static_cast<int16_t>(s);
            }
        };
#endif

        // ---- Phase 1: act[j] = silu(gate_j) * up_j for j in [0, gRows) ----
        //
        // Reuse the tiled maddubs machinery of the verified compact gate+up
        // kernel: quantize x to Q8_K once (reused across all rows AND both
        // matrices), unpack each compact Q2_K block's 2-bit planes on the fly,
        // and keep the per-tile gate/up accumulators in __m256 registers.
        //
        // SEQUENTIAL-STREAMING LAYOUT (llama.cpp-style, lever A): each thread
        // owns a contiguous slab of ROWS; within a row it iterates BLOCKS
        // sequentially (r-outer / b-inner). This gives each thread ONE
        // contiguous 84-byte-block stream per matrix (gate / up) instead of the
        // 16 interleaved strided streams the old b-outer/r-inner tiling opened
        // (which exceeded the HW prefetcher's ~8-stream capacity and caused DRAM
        // row-buffer misses). Row-block accumulators stay in float registers
        // (one __m256 per active row), so the r-outer loop costs no reloads.
        {
            static std::vector<Q8KBlock> q8All;
            q8All.resize(gBlocksPerRow);
            GGMLDequantize::quantizeQ8K(x, cols, q8All.data());
            // Scale shuffle table (q2k/q3k pattern): for each 16-element group,
            // duplicate its 16-bit scale across 8 int16 lanes — identical to the
            // prepacked kernels / the compact gate+up kernel (element order of
            // the on-the-fly unpack matches prepackQ2_K's expansion).
            static const uint8_t k_shuffle[128] = {
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    4,
                    5,
                    4,
                    5,
                    4,
                    5,
                    4,
                    5,
                    4,
                    5,
                    4,
                    5,
                    4,
                    5,
                    4,
                    5,
                    6,
                    7,
                    6,
                    7,
                    6,
                    7,
                    6,
                    7,
                    6,
                    7,
                    6,
                    7,
                    6,
                    7,
                    6,
                    7,
                    8,
                    9,
                    8,
                    9,
                    8,
                    9,
                    8,
                    9,
                    8,
                    9,
                    8,
                    9,
                    8,
                    9,
                    8,
                    9,
                    10,
                    11,
                    10,
                    11,
                    10,
                    11,
                    10,
                    11,
                    10,
                    11,
                    10,
                    11,
                    10,
                    11,
                    10,
                    11,
                    12,
                    13,
                    12,
                    13,
                    12,
                    13,
                    12,
                    13,
                    12,
                    13,
                    12,
                    13,
                    12,
                    13,
                    12,
                    13,
                    14,
                    15,
                    14,
                    15,
                    14,
                    15,
                    14,
                    15,
                    14,
                    15,
                    14,
                    15,
                    14,
                    15,
                    14,
                    15,
            };
            const __m256i mask3 = _mm256_set1_epi8(0x03);

            uint32_t numTiles = (gRows + BATCH_SIZE - 1) / BATCH_SIZE;
            uint32_t phase2NumTiles = (hRows + BATCH_SIZE - 1) / BATCH_SIZE;
#if defined(TINYCODER_USE_FFN_STEAL)
            // Lever 1 (plans/generation_optimizations.md): dynamic chunked
            // work-stealing dispatch (llama.cpp
            // ggml_compute_forward_mul_mat_one_chunk style). Each thread pulls
            // 4-tile chunks via fetch_add(4) instead of running a static slab,
            // so DRAM stays saturated at the tail - the mechanism behind
            // llama.cpp's 4-thread edge. Phase 2 steals 4 tiles at a time.
            ThreadPool::instance().parallelForSteal2(
                    0, numTiles, /*chunk1=*/TINYCODER_STEAL_CHUNK,
#else
            ThreadPool::instance().parallelForSlab2(
                    0, numTiles,
#endif
                    [&](uint32_t tile) {
                        (void) numTiles;
                        uint32_t rowStart = tile * BATCH_SIZE;
                        uint32_t batchSize = std::min(BATCH_SIZE, gRows - rowStart);

                        // Accumulators kept as __m256 in memory: [batchSize] (one per row).
                        static thread_local Vector256 gateAcc;
                        static thread_local Vector256 upAcc;
                        gateAcc.assign(batchSize, _mm256_setzero_ps());
                        upAcc.assign(batchSize, _mm256_setzero_ps());

                        // Per-block weight setup for a COMPACT Q2_K block: the fp16
                        // d/dmin, the int16 mins (for the min term), the two shuffled
                        // scale vectors, and the 8 unpacked q2 planes. Computed ONCE per
                        // (block column, row); with seqLen==1 there is no token loop to
                        // hoist out of.
                        struct QKSetup {
                            float dq;
                            float ndmin;
                            __m256i mins;
                            __m256i scales_shuf[2];
                            __m256i q2[8];
                        };
                        auto makeSetup = [&](const uint8_t *blockData) -> QKSetup {
                            QKSetup st;
                            st.dq = halfToFloat(*(const uint16_t *) (blockData + 80));
                            st.ndmin = -halfToFloat(*(const uint16_t *) (blockData + 82));
                            const uint8_t *scales = blockData;
                            // Min term mins (high nibbles).
                            __m128i ms = _mm_loadu_si128((const __m128i *) scales);
                            __m128i mins8 = _mm_and_si128(_mm_srli_epi16(ms, 4), _mm_set1_epi8(0xF));
                            st.mins = _mm256_cvtepi8_epi16(mins8);
                            // Scales (low nibbles) -> shuffled scale vectors.
                            __m128i scales8 = _mm_and_si128(ms, _mm_set1_epi8(0xF));
                            __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
                            __m128i slo = _mm256_extracti128_si256(scales16, 0);
                            __m128i shi = _mm256_extracti128_si256(scales16, 1);
                            st.scales_shuf[0] = _mm256_set_m128i(slo, slo);
                            st.scales_shuf[1] = _mm256_set_m128i(shi, shi);
                            // Unpack the 2-bit planes. Each of the two 32-byte qs
                            // quarters contains 4 planes (2 bits per byte): plane p =
                            // (byte >> 2p) & 3. The element order produced (quarter 0
                            // shift 0,2,4,6 then quarter 1 shift 0,2,4,6) is
                            // byte-identical to prepackQ2_K's expansion, so the
                            // scale/min shuffle math stays valid.
                            for (int h = 0; h < 2; ++h) {
                                const uint8_t *qsrc = blockData + 16 + h * 32;
                                __m256i b = _mm256_loadu_si256((const __m256i *) qsrc);
                                st.q2[h * 4 + 0] = _mm256_and_si256(b, mask3);
                                __m256i s2 = _mm256_srli_epi16(b, 2);
                                st.q2[h * 4 + 1] = _mm256_and_si256(s2, mask3);
                                __m256i s4 = _mm256_srli_epi16(s2, 2);
                                st.q2[h * 4 + 2] = _mm256_and_si256(s4, mask3);
                                __m256i s6 = _mm256_srli_epi16(s4, 2);
                                st.q2[h * 4 + 3] = _mm256_and_si256(s6, mask3);
                            }
                            return st;
                        };

                        // Hoist the shuffle table out of the per-row loop (the §3
                        // pattern): identical for every (block, row) of both matrices.
                        __m256i k_shuf[4] = {
                                _mm256_loadu_si256((const __m256i *) (k_shuffle + 0)),
                                _mm256_loadu_si256((const __m256i *) (k_shuffle + 32)),
                                _mm256_loadu_si256((const __m256i *) (k_shuffle + 64)),
                                _mm256_loadu_si256((const __m256i *) (k_shuffle + 96)),
                        };

                        // Compute the block contribution for one matrix row block.
                        auto blockDot = [&](const QKSetup &st, const __m256i *q8_ymm,
                                            const int16_t *bsums, float d) -> __m256 {
                            // Min term: -dmin * d * sum(mins * bsums). Kept as a VECTOR
                            // of per-lane products so the store-time horizontal sum
                            // counts it exactly once (scalar broadcast would count 8x).
                            __m256i prod = _mm256_madd_epi16(
                                    st.mins, _mm256_loadu_si256((const __m256i *) bsums));
                            __m256 minTerm = _mm256_fmadd_ps(
                                    _mm256_set1_ps(st.ndmin * d), _mm256_cvtepi32_ps(prod),
                                    _mm256_setzero_ps());

                            // Uses the k_shuf[4] table hoisted at the tile level (the
                            // §3 pattern) — rebuilt here per (block,row) would defeat
                            // the point of the hoist.

                            __m256 acc = _mm256_setzero_ps();
                            // 2 chunks of 128 elements each (256 total)
                            for (int chunk = 0; chunk < 2; ++chunk) {
                                __m256i sumi = _mm256_setzero_si256();
                                for (int sub = 0; sub < 4; ++sub) {
                                    int idx = chunk * 4 + sub;
                                    __m256i p = _mm256_maddubs_epi16(st.q2[idx], q8_ymm[idx]);
                                    __m256i shuf = k_shuf[sub];
                                    p = _mm256_madd_epi16(
                                            _mm256_shuffle_epi8(st.scales_shuf[chunk], shuf), p);
                                    sumi = _mm256_add_epi32(sumi, p);
                                }
                                acc = _mm256_fmadd_ps(_mm256_set1_ps(st.dq * d),
                                                      _mm256_cvtepi32_ps(sumi), acc);
                            }
                            return _mm256_add_ps(acc, minTerm);
                        };

                        // SEQUENTIAL-STREAMING LOOP (lever A): iterate ROWS outer,
                        // BLOCKS inner. Each thread walks its rows' weight blocks
                        // CONTIGUOUSLY — gate blocks for row r are [r*stride, (r+1)*
                        // stride) — giving the HW prefetcher a single sequential stream
                        // per matrix instead of the 8 interleaved strided streams the
                        // old b-outer tiling opened. The Q8_K x-vector is reloaded per
                        // row (it's L1-resident, ~6 KB), which is far cheaper than the
                        // DRAM row-buffer misses saved.
                        //
                        // MEASURED 2026-08-24: b-outer/row-inner (LM-head style, one
                        // x-load per block) measured 1064 ms vs this row-outer at
                        // 1009 ms (+5.5%, net loss) — the 2 matrices x batchSize rows
                        // of b-outer open 16 interleaved strided streams that defeat
                        // both the HW prefetcher and L3 (shared with phase-2's down
                        // matrix), so the sequential row-outer streaming is kept.
                        for (uint32_t r = 0; r < batchSize; ++r) {
                            const uint8_t *gateRow = gateData +
                                                     static_cast<uint64_t>(rowStart + r) * gRowStride;
                            const uint8_t *upRow = upData +
                                                   static_cast<uint64_t>(rowStart + r) * gRowStride;

                            // Per-row accumulators: one __m256 per matrix, 8 partial
                            // lanes (32 elements each), register-resident across the
                            // WHOLE block stream; hsum ONCE at the store below (llama
                            for (uint32_t b = 0; b < gBlocksPerRow; ++b) {
                                // Prefetch one block ahead (sequential stream).
                                // T1 (L2-leaning) keeps the gate/up stream out of the
                                // way of phase-2's down stream within the shared 8 MiB
                                // L3, which the per-token effective rate (~18 GB/s)
                                // shows is the contention point.
                                {
                                    uint32_t prefetchB = std::min(b + 64, gBlocksPerRow - 1);
                                    _mm_prefetch((const char *) (gateRow +
                                                                 static_cast<uint64_t>(prefetchB) * Q2K_COMPACT_BLOCK_BYTES),
                                                 _MM_HINT_T1);
                                    _mm_prefetch((const char *) (upRow +
                                                                 static_cast<uint64_t>(prefetchB) * Q2K_COMPACT_BLOCK_BYTES),
                                                 _MM_HINT_T1);
                                }

                                const Q8KBlock &q8 = q8All[b];
                                const float d = q8.d;
                                __m256i q8_ymm[8];
                                for (int i = 0; i < 8; ++i) {
                                    q8_ymm[i] = _mm256_loadu_si256(
                                            (const __m256i *) (q8.qs + i * 32));
                                }
                                const uint8_t *gateBlock = gateRow +
                                                           static_cast<uint64_t>(b) * Q2K_COMPACT_BLOCK_BYTES;
                                const uint8_t *upBlock = upRow +
                                                         static_cast<uint64_t>(b) * Q2K_COMPACT_BLOCK_BYTES;
                                QKSetup gateSt = makeSetup(gateBlock);
                                QKSetup upSt = makeSetup(upBlock);

                                gateAcc[r] = _mm256_add_ps(gateAcc[r],
                                                           blockDot(gateSt, q8_ymm, q8.bsums, d));
                                upAcc[r] = _mm256_add_ps(upAcc[r],
                                                         blockDot(upSt, q8_ymm, q8.bsums, d));
                            }
                        }

                        // Store results with fused SwiGLU: act[row] = silu(gate)*up.
                        for (uint32_t r = 0; r < batchSize; ++r) {
                            __m256 g = gateAcc[r];
                            __m256 u = upAcc[r];
                            __m128 ghi = _mm256_extractf128_ps(g, 1);
                            __m128 glo = _mm256_castps256_ps128(g);
                            __m128 gs = _mm_hadd_ps(_mm_add_ps(glo, ghi), _mm_add_ps(glo, ghi));
                            gs = _mm_hadd_ps(gs, gs);
                            __m128 uhi = _mm256_extractf128_ps(u, 1);
                            __m128 ulo = _mm256_castps256_ps128(u);
                            __m128 us = _mm_hadd_ps(_mm_add_ps(ulo, uhi), _mm_add_ps(ulo, uhi));
                            us = _mm_hadd_ps(us, us);
                            float gVal = _mm_cvtss_f32(gs);
                            float uVal = _mm_cvtss_f32(us);
                            act[rowStart + r] = gVal / (1.0f + std::exp(-gVal)) * uVal;
                        }
#if defined(TINYCODER_USE_COOP_QUANTIZE)
                        // Cooperative quantize — llama.cpp `from_float` LAST-ARRIVER
                        // pattern (lever O1 re-attempt, 2026-08-28). This tile stored
                        // act rows [rowStart, rowStart+batchSize), all within act block
                        // coopB = rowStart/256 (BLOCK_SIZE divides BATCH_SIZE exactly).
                        // Increment the block's arrival counter (acq_rel: the RMW chain
                        // orders the other 15 tiles' act[] writes before this read);
                        // the expected-th arriver quantizes the whole block inline,
                        // overlapping the quantize ALU with other threads' phase-1 DRAM
                        // streaming instead of a serial `between` bubble after barrier-1.
                        const uint32_t coopB = rowStart / BLOCK_SIZE;
                        const uint32_t prev = actArrivals[coopB].fetch_add(
                                1, std::memory_order_acq_rel);
                        if (prev + 1 == actArrivalExpected[coopB]) {
                            quantizeActBlock(coopB);
                        }
#endif
                    },
            // ---- Phase 2: Q3_K ffnDown on 'act', fold residual into store ----
            //
            // Quantize the act vector to Q8_K ONCE per block (shared across ALL
            // hRows — the down matrix rows all consume the same act input). Then
            // tile over hRows in 8-row batches, unpack each compact Q3_K block's
            // 2-bit planes + folded sign mask (w' = q2 + 4*hm), and run the proven
            // _mm256_maddubs_epi16 sub-block dots with the (sc-32) shuffle and the
            // -4*(sc-32).bsums offset term folded through the Q8KBlock bsums.
            // 'residual' (the transformer hidden state) is added in the store
            // epilogue — exactly like the attnO+residual fusion.
            //
            // SINGLE-LAUNCH MERGE (lever B): the quantize below runs as the
            // `between` callback (main thread) of ONE parallelForSlab2 whose
            // phase-1 work is the gate+up pass above (the phase-1 tiles wrote
            // act[]), and the ffnDown tiles run as its phase-2 slab. This removes
            // one task publication + one full cv barrier from every layer (28 per
            // token) vs the previous two separate parallelForSlab launches.
            // act → Q8_K quantize of the shared act vector (one Q8_K block per
            // dBlocksPerRow act-column, consumed by every phase-2 down row).
#if defined(TINYCODER_USE_COOP_QUANTIZE)
                    // O1 cooperative path (default): each block was quantized inline by
                    // the LAST ARRIVER during phase 1 (see the phase-1 lambda above), so
                    // barrier-1 + the arrival-counter release chains already published
                    // the whole q8actBuf[]. Nothing to do — release phase 2 directly.
                    nullptr,
#else
                    // Serial fallback (pre-O1 baseline, byte-for-byte): the main thread
                    // quantizes ALL blocks after barrier-1 while every worker spins on
                    // phase2Ready_ (each block's quantize is shared across all down rows;
                    // the pool barrier-1 guarantees all act[] writes are visible).
                    [&]() {
                        // Vectorized act → Q8_K quantize (AVX2), bit-identical to
                        // the scalar reference math:
                        //   amax = max |x|;  maxVal = first x with |x|==amax
                        //   iscale = -127/maxVal;  q = clamp(lrintf(iscale*x))
                        // The earlier corruption was a pack bug: vpacks operates in
                        // 128-bit lanes, so packs_epi32(vi,vi) + packs_epi16(p,p)
                        // stored only lanes 0..3 (4 bytes) per 8-element group.
                        // Fixed by narrowing through the two 128-bit halves with
                        // _mm_packs_epi32(lo,hi) -> 8 int16 -> _mm_packs_epi16
                        // -> low 64 bits = 8 contiguous int8. Rounding: vcvtps2dq
                        // honors the MXCSR rounding mode (RN by default) exactly
                        // like lrintf, so q values match bit-for-bit.
                        const __m256 vabsMask = _mm256_castsi256_ps(
                                _mm256_set1_epi32(static_cast<int>(0x7FFFFFFF)));
                        for (uint32_t b = 0; b < dBlocksPerRow; ++b) {
                            uint32_t base = b * BLOCK_SIZE;
                            uint32_t n = std::min(BLOCK_SIZE, gRows - base);
                            const float *src = act + base;
                            int8_t *qs = q8actBuf[b].qs;

                            // ---- Pass 1: |x| max (vectorized max-of-8) ----
                            float amaxVal = 0.0f, maxVal = 0.0f;
                            uint32_t i = 0;
                            for (; i + 8 <= n; i += 8) {
                                const __m256 v = _mm256_loadu_ps(src + i);
                                const __m256 av = _mm256_and_ps(v, vabsMask);// |x|
                                __m128 lo = _mm256_castps256_ps128(av);
                                __m128 hi = _mm256_extractf128_ps(av, 1);
                                __m128 m = _mm_max_ps(lo, hi);
                                m = _mm_max_ps(m, _mm_shuffle_ps(m, m, 0xB1));// [1,0,3,2]
                                m = _mm_max_ps(m, _mm_shuffle_ps(m, m, 0x4E));// [2,3,0,1]
                                const float mv = _mm_cvtss_f32(m);
                                if (mv > amaxVal) {
                                    amaxVal = mv;
                                }
                            }
                            for (; i < n; ++i) {
                                const float ax = std::fabs(src[i]);
                                if (ax > amaxVal) {
                                    amaxVal = ax;
                                }
                            }
                            // Sign resolution (first element achieving max |x|).
                            for (uint32_t k = 0; k < n; ++k) {
                                if (std::fabs(src[k]) == amaxVal) {
                                    maxVal = src[k];
                                    break;
                                }
                            }
                            if (amaxVal == 0.0f) {
                                q8actBuf[b].d = 0.0f;
                                std::memset(qs, 0, BLOCK_SIZE);
                                std::memset(q8actBuf[b].bsums, 0, sizeof(q8actBuf[b].bsums));
                                continue;
                            }
                            const float iscale = -127.0f / maxVal;
                            q8actBuf[b].d = 1.0f / iscale;

                            // ---- Pass 2: q = clamp(lrintf(iscale*x)) via vcvtps2dq ----
                            const __m256 viscale = _mm256_set1_ps(iscale);
                            const __m256 v127 = _mm256_set1_ps(127.0f);
                            const __m256 vm128 = _mm256_set1_ps(-128.0f);
                            i = 0;
                            for (; i + 8 <= n; i += 8) {
                                __m256 v = _mm256_loadu_ps(src + i);
                                v = _mm256_mul_ps(v, viscale);
                                v = _mm256_min_ps(_mm256_max_ps(v, vm128), v127);
                                const __m256i vi = _mm256_cvtps_epi32(v);
                                // Narrow 8xint32 -> 8xint8 correctly across the
                                // 128-bit lane boundary.
                                const __m128i lo32 = _mm256_castsi256_si128(vi);
                                const __m128i hi32 = _mm256_extracti128_si256(vi, 1);
                                const __m128i p16 = _mm_packs_epi32(lo32, hi32);// 8xint16
                                const __m128i p8 = _mm_packs_epi16(p16, p16);   // low64 = 8xint8
                                _mm_storel_epi64(reinterpret_cast<__m128i *>(qs + i), p8);
                            }
                            for (; i < n; ++i) {
                                const int v = static_cast<int>(std::lrintf(iscale * src[i]));
                                qs[i] = static_cast<int8_t>(std::min(127, std::max(-128, v)));
                            }
                            if (n < BLOCK_SIZE) {
                                std::memset(qs + n, 0, BLOCK_SIZE - n);
                            }
                            // Vectorized per-16-group byte sums (AVX2). Each group
                            // is 16 int8; two adjacent groups (32 bytes) fit one ymm.
                            //   _mm256_maddubs_epi16(1, v) -> 16 int16 lanes,
                            //     lane k = v[2k] + v[2k+1]
                            //   _mm256_madd_epi16(p, 1)   -> 8 int32 lanes,
                            //     lane k = bytes v[4k..4k+3]
                            // Lanes 0-3 = bytes 0..15 (group j), lanes 4-7 = bytes
                            // 16..31 (group j+1). Each group sums to <= 2032, well
                            // inside int16, so the int32 accumulation is exact and
                            // identical to the scalar reference (parity checked).
                            int j = 0;
                            for (; j + 2 <= 16; j += 2) {
                                const __m256i v = _mm256_loadu_si256(
                                        (const __m256i *) (qs + j * 16));
                                const __m256i p = _mm256_maddubs_epi16(
                                        _mm256_set1_epi8(1), v);
                                const __m256i s32 = _mm256_madd_epi16(
                                        p, _mm256_set1_epi16(1));
                                const __m128i loS = _mm256_castsi256_si128(s32);
                                const __m128i hiS = _mm256_extracti128_si256(s32, 1);
                                __m128i bj = _mm_hadd_epi32(loS, loS);
                                bj = _mm_hadd_epi32(bj, bj);
                                __m128i bj1 = _mm_hadd_epi32(hiS, hiS);
                                bj1 = _mm_hadd_epi32(bj1, bj1);
                                q8actBuf[b].bsums[j] =
                                        static_cast<int16_t>(_mm_cvtsi128_si32(bj));
                                q8actBuf[b].bsums[j + 1] =
                                        static_cast<int16_t>(_mm_cvtsi128_si32(bj1));
                            }
                            for (; j < 16; ++j) {
                                int s = 0;
                                for (int ii = 0; ii < 16; ++ii) {
                                    s += qs[j * 16 + ii];
                                }
                                q8actBuf[b].bsums[j] = static_cast<int16_t>(s);
                            }
                        }
                    },
#endif
#if defined(TINYCODER_USE_FFN_STEAL)
                    0, phase2NumTiles, /*chunk2=*/TINYCODER_STEAL_CHUNK,
#else
                    0, phase2NumTiles,
#endif
                    [&](uint32_t tile) {
                        uint32_t rowStart = tile * BATCH_SIZE;
                        uint32_t batchSize = std::min(BATCH_SIZE, hRows - rowStart);

                        static thread_local std::vector<float> accum;
                        accum.assign(batchSize, 0.0f);
                        // Q2_K down-path accumulation in __m256 registers (hsum at
                        // store). Only the DownIsQ2K instantiation uses it.
                        [[maybe_unused]] static thread_local Vector256 accumV;
                        if constexpr (DownIsQ2K) {
                            accumV.assign(batchSize, _mm256_setzero_ps());
                        }

                        // Scale shuffle table (q3k pattern): for each 16-element group,
                        // duplicate its 16-bit (sc-32) across 8 int16 lanes. This is the
                        // SAME table the verified batch Q3_K kernel uses.
                        static const uint8_t k_shuffle3k[128] = {
                                0,
                                1,
                                0,
                                1,
                                0,
                                1,
                                0,
                                1,
                                0,
                                1,
                                0,
                                1,
                                0,
                                1,
                                0,
                                1,
                                2,
                                3,
                                2,
                                3,
                                2,
                                3,
                                2,
                                3,
                                2,
                                3,
                                2,
                                3,
                                2,
                                3,
                                2,
                                3,
                                4,
                                5,
                                4,
                                5,
                                4,
                                5,
                                4,
                                5,
                                4,
                                5,
                                4,
                                5,
                                4,
                                5,
                                4,
                                5,
                                6,
                                7,
                                6,
                                7,
                                6,
                                7,
                                6,
                                7,
                                6,
                                7,
                                6,
                                7,
                                6,
                                7,
                                6,
                                7,
                                8,
                                9,
                                8,
                                9,
                                8,
                                9,
                                8,
                                9,
                                8,
                                9,
                                8,
                                9,
                                8,
                                9,
                                8,
                                9,
                                10,
                                11,
                                10,
                                11,
                                10,
                                11,
                                10,
                                11,
                                10,
                                11,
                                10,
                                11,
                                10,
                                11,
                                10,
                                11,
                                12,
                                13,
                                12,
                                13,
                                12,
                                13,
                                12,
                                13,
                                12,
                                13,
                                12,
                                13,
                                12,
                                13,
                                12,
                                13,
                                14,
                                15,
                                14,
                                15,
                                14,
                                15,
                                14,
                                15,
                                14,
                                15,
                                14,
                                15,
                                14,
                                15,
                                14,
                                15,
                        };
                        const __m256i mask3 = _mm256_set1_epi8(0x03);
                        const __m256i mask1 = _mm256_set1_epi8(0x01);
                        const __m256i thirtytwo = _mm256_set1_epi16(32);
                        const uint32_t kmask1 = 0x03030303;
                        const uint32_t kmask2 = 0x0f0f0f0f;

                        // Hoist the Q3K shuffle table out of the per-row loop (the §3
                        // pattern): identical for every (block, row) of the down matrix.
                        __m256i k_shuf[4] = {
                                _mm256_loadu_si256((const __m256i *) (k_shuffle3k + 0)),
                                _mm256_loadu_si256((const __m256i *) (k_shuffle3k + 32)),
                                _mm256_loadu_si256((const __m256i *) (k_shuffle3k + 64)),
                                _mm256_loadu_si256((const __m256i *) (k_shuffle3k + 96)),
                        };

                        // Q2_K down-path (Lever C): inlined block dot of one COMPACT
                        // Q2_K weight block (84 B, the load-time re-quant layout)
                        // against the shared act Q8_K x-block. Returns a __m256 of
                        // per-lane partials (NOT a horizontal sum) so phase-2 keeps
                        // the row accumulator in registers and hsums ONCE at store.
                        // Inlined (not a lambda) so the k_shuf[4]/q8 registers stay
                        // live and the accumV accumulator never spills.
                        [[maybe_unused]] auto q2kRowDot =
                                [&](const uint8_t *dB, const __m256i *q8,
                                    const int16_t *bsums, float xd) -> __m256 {
                            const float dq = halfToFloat(*(const uint16_t *) (dB + 80));
                            const float ndmin = -halfToFloat(*(const uint16_t *) (dB + 82));
                            const __m128i ms = _mm_loadu_si128((const __m128i *) dB);
                            __m256i mins = _mm256_cvtepi8_epi16(
                                    _mm_and_si128(_mm_srli_epi16(ms, 4),
                                                  _mm_set1_epi8(0xF)));
                            __m256i scales16 = _mm256_cvtepi8_epi16(
                                    _mm_and_si128(ms, _mm_set1_epi8(0xF)));
                            __m128i slo = _mm256_extracti128_si256(scales16, 0);
                            __m128i shi = _mm256_extracti128_si256(scales16, 1);
                            const __m256i scales_shuf[2] = {
                                    _mm256_set_m128i(slo, slo),
                                    _mm256_set_m128i(shi, shi)};
                            __m256i prod = _mm256_madd_epi16(
                                    mins, _mm256_loadu_si256((const __m256i *) bsums));
                            __m256 minTerm = _mm256_fmadd_ps(
                                    _mm256_set1_ps(ndmin * xd),
                                    _mm256_cvtepi32_ps(prod),
                                    _mm256_setzero_ps());
                            __m256 acc = _mm256_setzero_ps();
                            // 2 qs quarters (128 elems each), 4 planes per quarter.
                            for (int h = 0; h < 2; ++h) {
                                const __m256i bq =
                                        _mm256_loadu_si256((const __m256i *) (dB + 16 + h * 32));
                                __m256i q2[4];
                                q2[0] = _mm256_and_si256(bq, mask3);
                                __m256i s2 = _mm256_srli_epi16(bq, 2);
                                q2[1] = _mm256_and_si256(s2, mask3);
                                __m256i s4 = _mm256_srli_epi16(s2, 2);
                                q2[2] = _mm256_and_si256(s4, mask3);
                                __m256i s6 = _mm256_srli_epi16(s4, 2);
                                q2[3] = _mm256_and_si256(s6, mask3);
                                __m256i sumi = _mm256_setzero_si256();
                                for (int sub = 0; sub < 4; ++sub) {
                                    const int idx = h * 4 + sub;
                                    __m256i p = _mm256_maddubs_epi16(q2[sub], q8[idx]);
                                    p = _mm256_madd_epi16(
                                            _mm256_shuffle_epi8(scales_shuf[h], k_shuf[sub]),
                                            p);
                                    sumi = _mm256_add_epi32(sumi, p);
                                }
                                acc = _mm256_fmadd_ps(
                                        _mm256_set1_ps(dq * xd),
                                        _mm256_cvtepi32_ps(sumi), acc);
                            }
                            return _mm256_add_ps(acc, minTerm);
                        };

                        if constexpr (DownIsQ2K) {
                            // Q2_K phase-2: ROW-OUTER / BLOCK-INNER sequential streaming
                            // (the phase-1 proving pattern). The old b-outer/r-inner
                            // layout opened hRowStride-strided streams per tile row —
                            // 16 interleaved streams per thread capped the LSU and
                            // delivered only ~2.4 GB/s. Row-outer walks each row's
                            // blocks CONTIGUOUSLY (one sequential stream per row) and
                            // hits the ~12.5 GB/s the LM-head kernel sustains.
                            for (uint32_t r = 0; r < batchSize; ++r) {
                                const uint32_t jr = rowStart + r;
                                const uint8_t *dRow =
                                        downData + static_cast<uint64_t>(jr) * hRowStride;
                                for (uint32_t b = 0; b < dBlocksPerRow; ++b) {
                                    // Prefetch one block ahead (sequential row stream).
                                    const uint32_t prefetchB =
                                            std::min(b + 64, dBlocksPerRow - 1);
                                    _mm_prefetch(
                                            (const char *) (dRow + static_cast<uint64_t>(prefetchB) * Q2K_COMPACT_BLOCK_BYTES),
                                            _MM_HINT_T0);

                                    const uint8_t *dB = dRow +
                                                        static_cast<uint64_t>(b) * Q2K_COMPACT_BLOCK_BYTES;
                                    // Lever C: compact Q2_K ffnDown (84 B/block) — dot
                                    // against the shared Q8_K act block, INLINED (no
                                    // helper call). Unpack the 2-bit planes on the fly
                                    // and accumulate the __m256 partial directly into
                                    // accumV (hsum once at store). The Q8_K act x-vectors
                                    // are reloaded per (row, block) — L1-resident at ~6 KB,
                                    // far cheaper than the strided-stream stalls avoided.
                                    __m256i q8[8];
                                    for (int i = 0; i < 8; ++i) {
                                        q8[i] = _mm256_loadu_si256(
                                                (const __m256i *) (q8actBuf[b].qs + i * 32));
                                    }
                                    const float dq =
                                            halfToFloat(*(const uint16_t *) (dB + 80));
                                    const float ndmin =
                                            -halfToFloat(*(const uint16_t *) (dB + 82));
                                    const __m128i ms =
                                            _mm_loadu_si128((const __m128i *) dB);
                                    const __m256i mins = _mm256_cvtepi8_epi16(
                                            _mm_and_si128(_mm_srli_epi16(ms, 4),
                                                          _mm_set1_epi8(0xF)));
                                    __m256i scales16 = _mm256_cvtepi8_epi16(
                                            _mm_and_si128(ms, _mm_set1_epi8(0xF)));
                                    const __m128i slo =
                                            _mm256_extracti128_si256(scales16, 0);
                                    const __m128i shi =
                                            _mm256_extracti128_si256(scales16, 1);
                                    const __m256i ssc[2] = {
                                            _mm256_set_m128i(slo, slo),
                                            _mm256_set_m128i(shi, shi)};
                                    const __m256i bsumsV = _mm256_loadu_si256(
                                            (const __m256i *) q8actBuf[b].bsums);
                                    const __m256 gminT = _mm256_fmadd_ps(
                                            _mm256_set1_ps(ndmin * q8actBuf[b].d),
                                            _mm256_cvtepi32_ps(
                                                    _mm256_madd_epi16(mins, bsumsV)),
                                            _mm256_setzero_ps());
                                    __m256 pacc = _mm256_setzero_ps();
                                    for (int h = 0; h < 2; ++h) {
                                        const __m256i bq = _mm256_loadu_si256(
                                                (const __m256i *) (dB + 16 + h * 32));
                                        __m256i q2[4];
                                        q2[0] = _mm256_and_si256(bq, mask3);
                                        __m256i s2 = _mm256_srli_epi16(bq, 2);
                                        q2[1] = _mm256_and_si256(s2, mask3);
                                        __m256i s4 = _mm256_srli_epi16(s2, 2);
                                        q2[2] = _mm256_and_si256(s4, mask3);
                                        __m256i s6 = _mm256_srli_epi16(s4, 2);
                                        q2[3] = _mm256_and_si256(s6, mask3);
                                        __m256i sumi = _mm256_setzero_si256();
                                        for (int sub = 0; sub < 4; ++sub) {
                                            const int idx = h * 4 + sub;
                                            __m256i p = _mm256_maddubs_epi16(q2[sub], q8[idx]);
                                            p = _mm256_madd_epi16(
                                                    _mm256_shuffle_epi8(ssc[h], k_shuf[sub]),
                                                    p);
                                            sumi = _mm256_add_epi32(sumi, p);
                                        }
                                        pacc = _mm256_fmadd_ps(
                                                _mm256_set1_ps(dq * q8actBuf[b].d),
                                                _mm256_cvtepi32_ps(sumi), pacc);
                                    }
                                    accumV[r] = _mm256_add_ps(
                                            _mm256_add_ps(pacc, gminT), accumV[r]);
                                }
                            }
                        } else {
                            // Q3_K phase-2: ROW-OUTER / BLOCK-INNER sequential
                            // streaming, matching the Q2_K phase-2 pattern (which
                            // measures ~12.5 GB/s vs the old b-outer layout's ~5.3
                            // GB/s — the b-outer/r-inner loop opened 16 interleaved
                            // hRowStride-strided streams per thread that defeated
                            // both the HW prefetcher and the shared 8 MiB L3).
                            // Each thread's per-row walk is ONE contiguous 9.9
                            // MB/layer stream; block accumulators stay in __m256
                            // registers and the horizontal sum runs ONCE per row in
                            // the store epilogue (matching the Q2K path's
                            // hsum-once-at-store pattern).
                            for (uint32_t r = 0; r < batchSize; ++r) {
                                const uint32_t jr = rowStart + r;
                                const uint8_t *dRow =
                                        downData + static_cast<uint64_t>(jr) * hRowStride;
                                __m256 accRow = _mm256_setzero_ps();
                                for (uint32_t b = 0; b < dBlocksPerRow; ++b) {
                                    // Prefetch one block ahead (sequential row stream).
                                    const uint32_t prefetchB =
                                            std::min(b + 64, dBlocksPerRow - 1);
                                    _mm_prefetch(
                                            (const char *) (dRow + static_cast<uint64_t>(prefetchB) * Q3K_BLOCK_BYTES),
                                            _MM_HINT_T0);

                                    const uint8_t *dB =
                                            dRow + static_cast<uint64_t>(b) * Q3K_BLOCK_BYTES;
                                    float dq3k = halfToFloat(*(const uint16_t *) (dB + 108));
                                    const uint8_t *hm = dB;
                                    const uint8_t *q3k = dB + 32;

                                    // Unpack the 16 6-bit scales into 16 signed int8
                                    // (0..63), then subtract 32 to match dl = d_all*(sc-32).
                                    uint32_t aux[4];
                                    std::memcpy(aux, dB + 96, 12);
                                    uint32_t tmp32 = aux[2];
                                    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp32 >> 4) & kmask1) << 4);
                                    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp32 >> 6) & kmask1) << 4);
                                    aux[0] = (aux[0] & kmask2) | (((tmp32 >> 0) & kmask1) << 4);
                                    aux[1] = (aux[1] & kmask2) | (((tmp32 >> 2) & kmask1) << 4);
                                    const int8_t *sc = reinterpret_cast<const int8_t *>(aux);

                                    __m256i s16v = _mm256_cvtepi8_epi16(
                                            _mm_loadu_si128((const __m128i *) sc));
                                    s16v = _mm256_sub_epi16(s16v, thirtytwo);
                                    __m128i slo = _mm256_extracti128_si256(s16v, 0);
                                    __m128i shi = _mm256_extracti128_si256(s16v, 1);
                                    __m256i scales_shuf[2] = {
                                            _mm256_set_m128i(slo, slo),
                                            _mm256_set_m128i(shi, shi)};

                                    // Q3_K weight planes (once per row/block): split into
                                    // two fixed-source loops so the compiler never needs a
                                    // register-select for qb0/qb1 and every vpsrlw uses an
                                    // immediate shift.
                                    __m256i qb0 = _mm256_loadu_si256((const __m256i *) q3k);
                                    __m256i qb1 = _mm256_loadu_si256((const __m256i *) (q3k + 32));
                                    __m256i hb = _mm256_loadu_si256((const __m256i *) hm);
                                    __m256i w[8];
                                    for (int sp = 0; sp < 4; ++sp) {
                                        __m256i q2sp = _mm256_and_si256(
                                                _mm256_srli_epi16(qb0, 2 * sp), mask3);
                                        __m256i hbit = _mm256_and_si256(
                                                _mm256_srli_epi16(hb, sp), mask1);
                                        w[sp] = _mm256_add_epi8(q2sp, _mm256_slli_epi16(hbit, 2));
                                    }
                                    for (int sp = 4; sp < 8; ++sp) {
                                        __m256i q2sp = _mm256_and_si256(
                                                _mm256_srli_epi16(qb1, 2 * (sp - 4)), mask3);
                                        __m256i hbit = _mm256_and_si256(
                                                _mm256_srli_epi16(hb, sp), mask1);
                                        w[sp] = _mm256_add_epi8(q2sp, _mm256_slli_epi16(hbit, 2));
                                    }

                                    // Shared Q8_K act block for this block column
                                    // (reloaded per row — L1-resident, ~6 KB).
                                    __m256i q8[8];
                                    for (int i = 0; i < 8; ++i) {
                                        q8[i] = _mm256_loadu_si256(
                                                (const __m256i *) (q8actBuf[b].qs + i * 32));
                                    }
                                    __m256i bsumsV = _mm256_loadu_si256(
                                            (const __m256i *) q8actBuf[b].bsums);

                                    // Offset term (folds the -4 of w_eff = w' - 4 through
                                    // the bsums): -4*d_all*d*sum_g (sc[g]-32)*bsum[g]. Kept
                                    // as a VECTOR of per-lane products so the store-time
                                    // horizontal sum counts it exactly once (the reference
                                    // Q3K bug: a scalar broadcast counts it 8x).
                                    __m256i bprod = _mm256_madd_epi16(s16v, bsumsV);
                                    __m256 acc = _mm256_mul_ps(
                                            _mm256_set1_ps(-4.0f * dq3k * q8actBuf[b].d),
                                            _mm256_cvtepi32_ps(bprod));

                                    for (int chunk = 0; chunk < 2; ++chunk) {
                                        __m256i sumi = _mm256_setzero_si256();
                                        for (int sub = 0; sub < 4; ++sub) {
                                            int idx = chunk * 4 + sub;
                                            __m256i p = _mm256_maddubs_epi16(w[idx], q8[idx]);
                                            __m256i shuf = k_shuf[sub];
                                            p = _mm256_madd_epi16(
                                                    _mm256_shuffle_epi8(scales_shuf[chunk], shuf),
                                                    p);
                                            sumi = _mm256_add_epi32(sumi, p);
                                        }
                                        acc = _mm256_fmadd_ps(
                                                _mm256_set1_ps(dq3k * q8actBuf[b].d),
                                                _mm256_cvtepi32_ps(sumi), acc);
                                    }
                                    accRow = _mm256_add_ps(accRow, acc);
                                }
                                // Horizontal sum ONCE per row.
                                const __m128 hi = _mm256_extractf128_ps(accRow, 1);
                                const __m128 lo = _mm256_castps256_ps128(accRow);
                                __m128 s = _mm_hadd_ps(_mm_add_ps(lo, hi),
                                                       _mm_add_ps(lo, hi));
                                s = _mm_hadd_ps(s, s);
                                accum[r] += static_cast<float>(_mm_cvtss_f32(s));
                            }
                        }

                        // Store results with residual fold (hidden += down). The branch
                        // is loop-invariant; the fused path has no per-iteration branch.
                        if constexpr (DownIsQ2K) {
                            // Q2_K path accumulated __m256 partials in accumV; hsum
                            // ONCE per row here.
                            if (residual != nullptr) {
                                for (uint32_t r = 0; r < batchSize; ++r) {
                                    const __m256 v = accumV[r];
                                    const __m128 hi = _mm256_extractf128_ps(v, 1);
                                    const __m128 lo = _mm256_castps256_ps128(v);
                                    __m128 s = _mm_hadd_ps(_mm_add_ps(lo, hi),
                                                           _mm_add_ps(lo, hi));
                                    s = _mm_hadd_ps(s, s);
                                    out[rowStart + r] =
                                            _mm_cvtss_f32(s) + residual[rowStart + r];
                                }
                            } else {
                                for (uint32_t r = 0; r < batchSize; ++r) {
                                    const __m256 v = accumV[r];
                                    const __m128 hi = _mm256_extractf128_ps(v, 1);
                                    const __m128 lo = _mm256_castps256_ps128(v);
                                    __m128 s = _mm_hadd_ps(_mm_add_ps(lo, hi),
                                                           _mm_add_ps(lo, hi));
                                    s = _mm_hadd_ps(s, s);
                                    out[rowStart + r] = _mm_cvtss_f32(s);
                                }
                            }
                        } else if (residual != nullptr) {
                            for (uint32_t r = 0; r < batchSize; ++r) {
                                out[rowStart + r] = accum[r] + residual[rowStart + r];
                            }
                        } else {
                            for (uint32_t r = 0; r < batchSize; ++r) {
                                out[rowStart + r] = accum[r];
                            }
                        }
                    });
        }
    }

    // Non-template entry points. The Q3_K ffnDown (raw GGUF) variant is the
    // default; when the caller supplies a load-time Q2_K re-quant copy of the
    // down matrix (Lever C), the DownIsQ2K=true instantiation streams 84-byte
    // blocks with the Q2_K dot (24% less weight traffic than Q3_K's 110 B).
    void matMulVecFusedGateUpDownQ2K_Q3K_Compact_AVX2(
            const uint8_t *gateData,
            const uint8_t *upData,
            const uint8_t *downData,
            const float *x,
            uint32_t gRows,
            uint32_t hRows,
            uint32_t cols,
            float *out,
            float *residual) {
        matMulVecFusedGateUpDownQ2K_Compact_AVX2_impl<false>(
                gateData, upData, downData, x, gRows, hRows, cols, out, residual);
    }

    void matMulVecFusedGateUpDownQ2K_Q2K_Compact_AVX2(
            const uint8_t *gateData,
            const uint8_t *upData,
            const uint8_t *downData,
            const float *x,
            uint32_t gRows,
            uint32_t hRows,
            uint32_t cols,
            float *out,
            float *residual) {
        matMulVecFusedGateUpDownQ2K_Compact_AVX2_impl<true>(
                gateData, upData, downData, x, gRows, hRows, cols, out, residual);
    }

}// namespace tinycoder::simd
#endif// __AVX2__ && __FMA__
