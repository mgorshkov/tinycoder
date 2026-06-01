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
#include <cstdint>
#include <cstring>

// Include np library's CPU feature detection
#include <np/internal/CpuDispatch.hpp>

// ---------------------------------------------------------------------------
// Scalar (no SIMD) implementation — always available
// ---------------------------------------------------------------------------
namespace {

    void accumulateFMA_Scalar(float *local, const float *blockOut, float alpha,
                              uint32_t n) {
        for (uint32_t j = 0; j < n; ++j) {
            local[j] += alpha * blockOut[j];
        }
    }

}// anonymous namespace

// ---------------------------------------------------------------------------
// AVX2 implementation (compiled with -mavx2 -mfma)
// ---------------------------------------------------------------------------
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

namespace {

    void accumulateFMA_AVX2(float *local, const float *blockOut, float alpha,
                            uint32_t n) {
        __m256 alphaVec = _mm256_set1_ps(alpha);

        uint32_t j = 0;

        // Process 8 floats at a time with AVX2 FMA
        for (; j + 8 <= n; j += 8) {
            __m256 blockVec = _mm256_loadu_ps(blockOut + j);
            __m256 localVec = _mm256_loadu_ps(local + j);
            // local = local + alpha * block
            __m256 result = _mm256_fmadd_ps(alphaVec, blockVec, localVec);
            _mm256_storeu_ps(local + j, result);
        }

        // Handle remaining elements (0-7) with scalar fallback
        for (; j < n; ++j) {
            local[j] += alpha * blockOut[j];
        }
    }

}// anonymous namespace
#endif// __AVX2__ && __FMA__

// ---------------------------------------------------------------------------
// AVX-512 implementation (compiled with -mavx512f -mfma)
// ---------------------------------------------------------------------------
#if defined(__AVX512F__) && defined(__FMA__)
#include <immintrin.h>

namespace {

    void accumulateFMA_AVX512(float *local, const float *blockOut, float alpha,
                              uint32_t n) {
        __m512 alphaVec = _mm512_set1_ps(alpha);

        uint32_t j = 0;

        // Process 16 floats at a time with AVX-512 FMA
        for (; j + 16 <= n; j += 16) {
            __m512 blockVec = _mm512_loadu_ps(blockOut + j);
            __m512 localVec = _mm512_loadu_ps(local + j);
            // local = local + alpha * block
            __m512 result = _mm512_fmadd_ps(alphaVec, blockVec, localVec);
            _mm512_storeu_ps(local + j, result);
        }

        // Handle remaining elements (0-15) with scalar fallback
        for (; j < n; ++j) {
            local[j] += alpha * blockOut[j];
        }
    }

}// anonymous namespace
#endif// __AVX512F__ && __FMA__

// ---------------------------------------------------------------------------
// AMX (Intel Advanced Matrix Extensions) implementation
// AMX operates on tiles (2D registers). For a 1D axpy, we use a tile-based
// approach: load 16x16 tiles and accumulate. This is beneficial when n is
// large (>= 16) and we have many consecutive calls (amortized tile config).
//
// AMX tile config is set once and reused across calls.
// ---------------------------------------------------------------------------
#if defined(__AMX_TILE__) && defined(__AMX_INT8__)
// Note: AMX is primarily for INT8. For FP32 axpy, AVX-512 is generally faster.
// We include AMX here for completeness but it's mainly useful when combined
// with INT8 quantization paths in the future.
#include <immintrin.h>

namespace {

    // AMX tile configuration for 16x16 float tiles
    // AMX doesn't natively support FP32 tiles, so this is a placeholder
    // that falls back to AVX-512 or scalar. Real AMX FP32 support requires
    // future ISA extensions (AMX-FP16 or similar).
    void accumulateFMA_AMX(float *local, const float *blockOut, float alpha,
                           uint32_t n) {
        // AMX tile operations work on INT8/FP16 tiles, not FP32.
        // For FP32 axpy, AVX-512 is the optimal path.
        // This function is a placeholder that delegates to AVX-512 when available,
        // or falls back to scalar.
#if defined(__AVX512F__) && defined(__FMA__)
        accumulateFMA_AVX512(local, blockOut, alpha, n);
#else
        accumulateFMA_Scalar(local, blockOut, alpha, n);
#endif
    }

}// anonymous namespace
#endif// __AMX_TILE__

// ---------------------------------------------------------------------------
// dotProductFMA — scalar implementation
// ---------------------------------------------------------------------------
namespace {

    float dotProductFMA_Scalar(const float *hidden, const float *blockOut,
                               uint32_t n) {
        double sum = 0.0;
        for (uint32_t j = 0; j < n; ++j) {
            sum += static_cast<double>(hidden[j]) * blockOut[j];
        }
        return static_cast<float>(sum);
    }

}// anonymous namespace

// ---------------------------------------------------------------------------
// dotProductFMA — AVX2 implementation
// ---------------------------------------------------------------------------
#if defined(__AVX2__) && defined(__FMA__)
namespace {

    float dotProductFMA_AVX2(const float *hidden, const float *blockOut,
                             uint32_t n) {
        __m256 sumVec = _mm256_setzero_ps();

        uint32_t j = 0;
        for (; j + 8 <= n; j += 8) {
            __m256 hVec = _mm256_loadu_ps(hidden + j);
            __m256 bVec = _mm256_loadu_ps(blockOut + j);
            // sumVec += hidden[j] * blockOut[j]
            sumVec = _mm256_fmadd_ps(hVec, bVec, sumVec);
        }

        // Horizontal reduction of 8 floats to 1
        __m128 hi = _mm256_extractf128_ps(sumVec, 1);
        __m128 lo = _mm256_castps256_ps128(sumVec);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float result = _mm_cvtss_f32(sum128);

        // Handle remaining elements (0-7)
        for (; j < n; ++j) {
            result += hidden[j] * blockOut[j];
        }
        return result;
    }

}// anonymous namespace
#endif// __AVX2__ && __FMA__

// ---------------------------------------------------------------------------
// dotProductFMA — AVX-512 implementation
// ---------------------------------------------------------------------------
#if defined(__AVX512F__) && defined(__FMA__)
namespace {

    float dotProductFMA_AVX512(const float *hidden, const float *blockOut,
                               uint32_t n) {
        __m512 sumVec = _mm512_setzero_ps();

        uint32_t j = 0;
        for (; j + 16 <= n; j += 16) {
            __m512 hVec = _mm512_loadu_ps(hidden + j);
            __m512 bVec = _mm512_loadu_ps(blockOut + j);
            sumVec = _mm512_fmadd_ps(hVec, bVec, sumVec);
        }

        // Horizontal reduction of 16 floats to 1
        float result = _mm512_reduce_add_ps(sumVec);

        // Handle remaining elements (0-15)
        for (; j < n; ++j) {
            result += hidden[j] * blockOut[j];
        }
        return result;
    }

}// anonymous namespace
#endif// __AVX512F__ && __FMA__

// ---------------------------------------------------------------------------
// Runtime dispatch — selects the best implementation once and caches it
// ---------------------------------------------------------------------------

namespace tinycoder {

    void accumulateFMA(float *local, const float *blockOut, float alpha,
                       uint32_t n) {
        // Cached function pointer for the best available implementation.
        // Initialized on first call with a relaxed atomic store (benign race:
        // if two threads race, both compute the same result and store the same
        // pointer).
        static std::atomic<void (*)(float *, const float *, float, uint32_t)> s_impl{
                nullptr};

        auto impl = s_impl.load(std::memory_order_acquire);
        if (!impl) {
            // Determine the best SIMD level at runtime
            using np::internal::SimdLevel;
            SimdLevel level = np::internal::max_simd_level();

            // Select implementation based on runtime CPU capabilities
            // (capped by compile-time ENABLE_* macros inside max_simd_level())
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

    float dotProductFMA(const float *hidden, const float *blockOut, uint32_t n) {
        // Cached function pointer for the best available implementation.
        static std::atomic<float (*)(const float *, const float *, uint32_t)> s_impl{
                nullptr};

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

}// namespace tinycoder
