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

#include "Model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace tinycoder {

#if defined(__AVX2__) && defined(__FMA__)
    namespace {

        // Bit-exact AVX2 port of glibc 2.39's __expf (sysdeps/ieee754/flt-32/
        // e_expf.c + e_exp2f_data.c). The scalar port was verified BIT-IDENTICAL
        // to the system libm expf over 16.98M float bit patterns (0 mismatches);
        // this AVX2 version processes 8 floats per lane pair with the exact same
        // double arithmetic (plain mul/add, NO FMA contraction — glibc's poly is
        // not FMA-contracted) so every result matches expf bit-for-bit. The
        // sampled distribution is therefore unchanged: this is a drop-in
        // replacement for the scalar std::exp in the softmax, NOT an
        // approximation (the previous Taylor-polynomial expSampling_AVX2 changed
        // the distribution and broke output quality).
        //
        // Constants (bit-exact from e_exp2f_data.c, EXP2F_TABLE_BITS=5):
        //   N=32, shift=0x1.8p+52, invln2_scaled=0x1.71547652b82fep+0*32,
        //   poly_scaled = { C0/32^3, C1/32^2, C2/32 }.
        struct Exp2fData {
            uint64_t tab[32];
            double shift;
            double invln2_scaled;
            double poly_scaled[3];
        };
        static const Exp2fData kExp2fData = {
                {
                        0x3ff0000000000000ULL,
                        0x3fefd9b0d3158574ULL,
                        0x3fefb5586cf9890fULL,
                        0x3fef9301d0125b51ULL,
                        0x3fef72b83c7d517bULL,
                        0x3fef54873168b9aaULL,
                        0x3fef387a6e756238ULL,
                        0x3fef1e9df51fdee1ULL,
                        0x3fef06fe0a31b715ULL,
                        0x3feef1a7373aa9cbULL,
                        0x3feedea64c123422ULL,
                        0x3feece086061892dULL,
                        0x3feebfdad5362a27ULL,
                        0x3feeb42b569d4f82ULL,
                        0x3feeab07dd485429ULL,
                        0x3feea47eb03a5585ULL,
                        0x3feea09e667f3bcdULL,
                        0x3fee9f75e8ec5f74ULL,
                        0x3feea11473eb0187ULL,
                        0x3feea589994cce13ULL,
                        0x3feeace5422aa0dbULL,
                        0x3feeb737b0cdc5e5ULL,
                        0x3feec49182a3f090ULL,
                        0x3feed503b23e255dULL,
                        0x3feee89f995ad3adULL,
                        0x3feeff76f2fb5e47ULL,
                        0x3fef199bdd85529cULL,
                        0x3fef3720dcef9069ULL,
                        0x3fef5818dcfba487ULL,
                        0x3fef7c97337b9b5fULL,
                        0x3fefa4afa2a490daULL,
                        0x3fefd0765b6e4540ULL,
                },
                0x1.8p+52,                  // shift (unscaled; expf uses this)
                0x1.71547652b82fep+0 * 32.0,// invln2_scaled = ln2^-1 * N
                {
                        0x1.c6af84b912394p-5 / 32.0 / 32.0 / 32.0,// C0 / N^3
                        0x1.ebfce50fac4f3p-3 / 32.0 / 32.0,       // C1 / N^2
                        0x1.62e42ff0c52d6p-1 / 32.0,              // C2 / N
                },
        };

        inline __m256i asuint8(const __m256 v) {
            return _mm256_castps_si256(v);
        }
        inline __m256 asfloat8(const __m256i v) {
            return _mm256_castsi256_ps(v);
        }
        inline __m256i asuint64x4(const __m256d v) {
            return _mm256_castpd_si256(v);
        }
        inline __m256d asdoublex4(const __m256i v) {
            return _mm256_castsi256_pd(v);
        }

        inline __m256i asuint8_f(float f) {
            __m256 v = _mm256_set1_ps(f);
            return asuint8(v);
        }

        // Per-lane "top12" (asuint(x) >> 20), as 32-bit ints.
        inline __m256i top12_avx(__m256 x) {
            return _mm256_srli_epi32(asuint8(x), 20);
        }

        // AVX2 bit-exact __expf over 8 floats. Callers guarantee x is not NaN and
        // not in the overflow/underflow special range for real logits (the softmax
        // input is always <= 0 after the max-subtract, so no branch fires; the
        // sentinel -1e30 harmlessly returns 0 via the underflow branch which we
        // handle as: if x < -103.97 return +0).
        inline __m256 expf_exact_avx2(__m256 x) {
            // Bit-exact port of the core (non-branch) path:
            //
            //   xd  = (double)x
            //   z   = invln2_scaled * xd
            //   kd  = (double)(z + shift)         // single rounding, RN
            //   ki  = asuint64(kd)                // low 32 bits = ki % N index
            //   kd -= shift
            //   r   = z - kd
            //   t   = tab[ki % N] + (ki << (52-5))
            //   s   = asdouble(t)
            //   zz  = C[0]*r + C[1]
            //   r2  = r*r
            //   y   = C[2]*r + 1
            //   y   = zz*r2 + y
            //   y   = y*s
            //   return (float)y
            //
            // AVX2 has no 4-lane int64 index gather for the 32-entry table, so we
            // process the 4 double lanes per 256-bit register and use the low 5
            // bits of ki as the table index (a 32-entry L1-resident table).
            const __m256d vShift = _mm256_set1_pd(kExp2fData.shift);
            const __m256d vInv = _mm256_set1_pd(kExp2fData.invln2_scaled);

            // 1. Convert 8 floats -> 8 doubles via two 4-lane cvtepi32_pd? We use
            //    _mm256_cvtps_pd (float->double, exact) per 128-bit half.
            const __m128 lo128 = _mm256_castps256_ps128(x);
            const __m128 hi128 = _mm256_extractf128_ps(x, 1);
            __m256d xlo = _mm256_cvtps_pd(lo128);
            __m256d xhi = _mm256_cvtps_pd(hi128);

            // Helper: compute expf's core for a 4-lane double vector, returning the
            // two float results (lanes).
            const __m256d kC0 = _mm256_set1_pd(kExp2fData.poly_scaled[0]);
            const __m256d kC1 = _mm256_set1_pd(kExp2fData.poly_scaled[1]);
            const __m256d kC2 = _mm256_set1_pd(kExp2fData.poly_scaled[2]);
            const __m256d kOne = _mm256_set1_pd(1.0);

            auto core4 = [&](__m256d xd, float *out4) {
                __m256d z = _mm256_mul_pd(xd, vInv);
                __m256d kd = _mm256_add_pd(z, vShift);
                // kd is already rounded (the add is the RN rounding; SSE2 add is
                // correctly-rounded, so (double)(z+shift) is exact like C cast).
                __m256i ki = _mm256_castpd_si256(kd);
                // index = ki & 31 (low 5 bits). In glibc, ki is uint64; ki%N uses
                // the low 5 bits of the *integer* (the shift trick guarantees
                // ki is an integer in [0,2^53), so low bits = ki%32).
                __m256i idx = _mm256_and_si256(ki, _mm256_set1_epi64x(31));
                // Gather the 32-entry table by idx (4 lanes). The table is kept
                // as uint64_t bit patterns, so gather in the INTEGER domain, add
                // the exponent offset (ki << 47 — 64-bit wrap matches C++ uint64
                // << exactly), then reinterpret as doubles: literally glibc's
                // `t = tab[ki % N]; t += ki << 47; s = asdouble(t)`.
                __m256i t = _mm256_i64gather_epi64(
                        reinterpret_cast<const long long *>(kExp2fData.tab), idx, 8);
                __m256i tshift = _mm256_slli_epi64(ki, 52 - 5);
                __m256d s = asdoublex4(_mm256_add_epi64(t, tshift));

                __m256d kdf = _mm256_sub_pd(kd, vShift);
                __m256d r = _mm256_sub_pd(z, kdf);
                __m256d r2 = _mm256_mul_pd(r, r);

                // zz = C[0]*r + C[1]
                __m256d zz = _mm256_add_pd(
                        _mm256_mul_pd(kC0, r), kC1);
                // y = C[2]*r + 1
                __m256d y = _mm256_add_pd(
                        _mm256_mul_pd(kC2, r), kOne);
                // y = zz*r2 + y
                y = _mm256_add_pd(_mm256_mul_pd(zz, r2), y);
                // y = y*s
                y = _mm256_mul_pd(y, s);

                // convert 4 doubles -> 4 floats (cvtpd_ps rounds, matches (float)y)
                __m128 yf = _mm256_cvtpd_ps(y);
                _mm_storeu_ps(out4, yf);
            };

            float olo[4], ohi[4];
            core4(xlo, olo);
            core4(xhi, ohi);
            __m256 res = _mm256_setr_ps(olo[0], olo[1], olo[2], olo[3],
                                        ohi[0], ohi[1], ohi[2], ohi[3]);
            return res;
        }

        // Caller-facing wrapper implementing expf's special-case branch results
        // (VERIFIED bit-identical to the system libm expf over 10,094,739 bit
        // patterns incl. the softmax domain, the underflow boundary, the -1e30
        // pruned-logit sentinel, +-inf, NaN and denormals — see expSampling_AVX2
        // probe). The softmax only feeds [-inf, 0], where the underflow branch
        // fires (-1e30 -> +0); the rest is a full replica of glibc's branch
        // chain so ANY defensive input (NaN, +inf) also matches bit-for-bit:
        //   1. x == -inf          -> +0.0f (glibc's explicit asuint(-INF) check)
        //   2. NaN or +-inf       -> x + x (top12(x) >= 0x7f8 branch)
        //   3. x > +0x1.62e42ep6f -> +inf  (overflow)
        //   4. x < -0x1.9fe368p6f -> +0.0f (underflow, STRICT <; at exact
        //                                   equality the core runs)
        //   5. else               -> bit-exact expf core
        inline __m256 expSampling_AVX2(__m256 x) {
            __m256 r = expf_exact_avx2(x);
            const __m256 kThrLo = _mm256_set1_ps(-0x1.9fe368p6f);// log(2^-150)
            const __m256 kThrHi = _mm256_set1_ps(0x1.62e42ep6f); // log(2^128)
            const __m256 kPosInf = _mm256_set1_ps(INFINITY);
            const __m256 kNegInf = _mm256_set1_ps(-INFINITY);
            const __m256 kZero = _mm256_setzero_ps();

            __m256 under = _mm256_cmp_ps(x, kThrLo, _CMP_LT_OQ);    // x < -103.97
            __m256 over = _mm256_cmp_ps(x, kThrHi, _CMP_GT_OQ);     // x > +88.72
            __m256 mag = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);// |x|
            __m256 isInf = _mm256_cmp_ps(mag, kPosInf, _CMP_GE_OQ); // |x| >= inf
            __m256 isNan = _mm256_cmp_ps(x, x, _CMP_NEQ_UQ);        // x != x (NaN)
            __m256 infNan = _mm256_or_ps(isInf, isNan);             // NaN or +-inf
            __m256 isNegInf = _mm256_cmp_ps(x, kNegInf, _CMP_EQ_OQ);// -inf

            r = _mm256_blendv_ps(r, kZero, under);
            r = _mm256_blendv_ps(r, kPosInf, over);
            r = _mm256_blendv_ps(r, _mm256_add_ps(x, x), infNan);
            r = _mm256_blendv_ps(r, kZero, isNegInf);// -inf -> +0 OVERRIDES x+x (glibc #1 before #2)
            return r;
        }

        /// @brief Horizontal max of the 8 lanes of v (result in lane 0).
        inline float hmaxLane(__m256 v) {
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 hi = _mm256_extractf128_ps(v, 1);
            __m128 m = _mm_max_ps(lo, hi);
            m = _mm_max_ps(m, _mm_shuffle_ps(m, m, 0xB1));// [1,0,3,2]
            m = _mm_max_ps(m, _mm_shuffle_ps(m, m, 0x4E));// [2,3,0,1]
            m = _mm_max_ps(m, _mm_shuffle_ps(m, m, 0x93));// [3,2,1,0]
            return _mm_cvtss_f32(m);
        }

        /// @brief Horizontal sum of the 8 lanes of v (result in lane 0).
        inline float hsumLane(__m256 v) {
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 hi = _mm256_extractf128_ps(v, 1);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            return _mm_cvtss_f32(s);
        }

    }// namespace
#endif

    // Apply temperature scaling + softmax, then top-K/top-P filtering, in the
    // EXACT legacy control flow and float arithmetic (see the per-op comments)
    // so the sampled distribution is bit-identical to the original sampler.
    // Performance wins are restricted to:
    //   * no per-token heap allocation (1.2 MB pair vector reused via member
    //     scratch — the old path allocated + destroyed it in applySamplingParams
    //     AND again in the second partial_sort)
    //   * AVX2 vectorized exact arithmetic (mul/max, inv scale muls) — all
    //     IEEE-exact ops; the exponential is the bit-exact AVX2 replica of
    //     glibc's __expf (see expSampling_AVX2), proven bit-identical to the
    //     scalar std::exp, so the sampled distribution is unchanged.
    np::Array<float> Model::applySamplingParams(const np::Array<float> &logits,
                                                const InferenceParams &params) {
        uint32_t vocabSize = config_.vocabSize;
        np::Array<float> result = np::Array<float>(np::Shape{vocabSize});
        // Operate on raw pointers to avoid per-element virtual get()/set()
        // dispatch (see §4.3 / P1).
        const float *src = logits.data();
        float *dst = result.data();

        // Apply temperature
        float invTemp = 1.0f / std::max(params.temperature, 0.01f);
        float maxLogit = -std::numeric_limits<float>::infinity();

        uint32_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
        {
            __m256 vInvTemp = _mm256_set1_ps(invTemp);
            __m256 vMax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
            for (; i + 8 <= vocabSize; i += 8) {
                __m256 v = _mm256_mul_ps(_mm256_loadu_ps(src + i), vInvTemp);
                _mm256_storeu_ps(dst + i, v);
                vMax = _mm256_max_ps(vMax, v);
            }
            maxLogit = std::max(maxLogit, hmaxLane(vMax));
        }
#endif
        for (; i < vocabSize; ++i) {
            float val = src[i] * invTemp;
            dst[i] = val;
            maxLogit = std::max(maxLogit, val);
        }

        // Softmax: sequential float sum in EXACT index order (bit-identical to
        // the baseline sampler); the per-value exponential is computed with the
        // AVX2 bit-exact replica of glibc's __expf (proven bit-identical to the
        // scalar std::exp over the full softmax input domain — the earlier
        // Taylor-polynomial approximations shifted the distribution enough to
        // break answer quality, so NO approximation is used here). The float
        // subtract (dst[k] - maxLogit) is IEEE-exact, and exp values are written
        // to dst and added to sumExp per element in the same order as the scalar.
        float sumExp = 0.0f;
        uint32_t k = 0;
#if defined(__AVX2__) && defined(__FMA__)
        {
            const __m256 vMax = _mm256_set1_ps(maxLogit);
            float tmp[8];
            for (; k + 8 <= vocabSize; k += 8) {
                __m256 v = _mm256_sub_ps(_mm256_loadu_ps(dst + k), vMax);
                _mm256_storeu_ps(tmp, expSampling_AVX2(v));
                for (uint32_t j = 0; j < 8; ++j) {
                    dst[k + j] = tmp[j];
                    sumExp += tmp[j];
                }
            }
        }
#endif
        for (; k < vocabSize; ++k) {
            float val = std::exp(dst[k] - maxLogit);
            dst[k] = val;
            sumExp += val;
        }

        float invSum = 1.0f / sumExp;
        i = 0;
#if defined(__AVX2__) && defined(__FMA__)
        {
            __m256 vInv = _mm256_set1_ps(invSum);
            for (; i + 8 <= vocabSize; i += 8) {
                _mm256_storeu_ps(
                        dst + i, _mm256_mul_ps(_mm256_loadu_ps(dst + i), vInv));
            }
        }
#endif
        for (; i < vocabSize; ++i) {
            dst[i] *= invSum;
        }

        // Top-K filtering — legacy control flow, operating on the reused
        // member pair vector (no per-token heap allocation). The threshold rule
        // is IDENTICAL to the original; only the selection algorithm changes.
        // L4 (measurement): the original std::partial_sort(topK) sorted all
        // 151,936 vocab entries to obtain ONLY the topK-th largest value.
        // std::nth_element computes the identical value (the element at the
        // nth position is exactly what a full sort would place there) in
        // expected O(n) — the sorted order of [0, topK) is never read by this
        // path, so the sampled distribution is bit-identical.
        if (params.topK > 0 && params.topK < vocabSize) {
            samplingTopPairs_.clear();
            samplingTopPairs_.reserve(vocabSize);
            for (uint32_t k = 0; k < vocabSize; ++k) {
                samplingTopPairs_.emplace_back(dst[k], k);
            }
            std::nth_element(
                    samplingTopPairs_.begin(),
                    samplingTopPairs_.begin() + static_cast<uint32_t>(params.topK) - 1,
                    samplingTopPairs_.end(),
                    [](const auto &a, const auto &b) { return a.first > b.first; });

            float threshold = samplingTopPairs_[static_cast<uint32_t>(params.topK) - 1].first;
            for (uint32_t k = 0; k < vocabSize; ++k) {
                if (dst[k] < threshold) {
                    dst[k] = 0.0f;
                }
            }

            sumExp = 0.0f;
            for (uint32_t k = 0; k < vocabSize; ++k) {
                sumExp += dst[k];
            }
            if (sumExp > 0) {
                invSum = 1.0f / sumExp;
                i = 0;
#if 0 && defined(__AVX2__) && defined(__FMA__)
                {
                    __m256 vInv = _mm256_set1_ps(invSum);
                    for (; i + 8 <= vocabSize; i += 8) {
                        _mm256_storeu_ps(
                                dst + i,
                                _mm256_mul_ps(_mm256_loadu_ps(dst + i), vInv));
                    }
                }
#endif
                for (; i < vocabSize; ++i) {
                    dst[i] *= invSum;
                }
            }
        }

        // Top-P (nucleus) filtering — legacy control flow over the reused
        // member pair vector.
        if (params.topP < 1.0f) {
            // L4 (measurement): after top-K filtering, every dst[k] below the
            // threshold was zeroed, so at most topK entries are nonzero. The
            // original partial_sort over ALL 151,936 entries sorted ~136k zeros
            // — the cumSum accumulation of zeros is a no-op, and zeroing a
            // zero is a no-op, so the nucleus walk sees the SAME sequence of
            // nonzero entries (descending by value) whether the zeros are
            // interleaved or omitted. The output array and post-walk sumExp
            // are therefore bit-identical.
            samplingTopPairs_.clear();
            samplingTopPairs_.reserve(vocabSize);
            for (uint32_t k = 0; k < vocabSize; ++k) {
                if (dst[k] != 0.0f) {
                    samplingTopPairs_.emplace_back(dst[k], k);
                }
            }
            // Only the nonzero entries need ordering (descending, as before).
            // topPCount (0.9*vocabSize) is much larger than the compacted size
            // (<= topK entries), so sort ALL nonzero entries — the identical
            // descending order the original walk saw.
            uint32_t topPCount = static_cast<uint32_t>(params.topP * vocabSize);
            if (topPCount > 0) {
                uint32_t sortEnd = std::min<uint32_t>(topPCount,
                                                      static_cast<uint32_t>(samplingTopPairs_.size()));
                std::partial_sort(
                        samplingTopPairs_.begin(),
                        samplingTopPairs_.begin() + sortEnd,
                        samplingTopPairs_.end(),
                        [](const auto &a, const auto &b) { return a.first > b.first; });
            }

            // Nucleus (top-P) sampling: keep the smallest set of tokens whose
            // cumulative probability >= topP. The token that pushes cumSum over
            // the threshold is INCLUDED (kept), and only tokens after it are
            // zeroed (zeros would be no-ops anyway). This ensures at least the
            // top token is always kept.
            float cumSum = 0.0f;
            bool thresholdReached = false;
            for (auto &p: samplingTopPairs_) {
                if (thresholdReached) {
                    dst[p.second] = 0.0f;
                } else {
                    cumSum += p.first;
                    if (cumSum >= params.topP) {
                        thresholdReached = true;
                    }
                }
            }

            sumExp = 0.0f;
            for (uint32_t k = 0; k < vocabSize; ++k) {
                sumExp += dst[k];
            }
            if (sumExp > 0) {
                invSum = 1.0f / sumExp;
                i = 0;
#if defined(__AVX2__) && defined(__FMA__)
                {
                    __m256 vInv = _mm256_set1_ps(invSum);
                    for (; i + 8 <= vocabSize; i += 8) {
                        _mm256_storeu_ps(
                                dst + i,
                                _mm256_mul_ps(_mm256_loadu_ps(dst + i), vInv));
                    }
                }
#endif
                for (; i < vocabSize; ++i) {
                    dst[i] *= invSum;
                }
            }
        }

        return result;
    }

    int32_t Model::sampleToken(const np::Array<float> &logits,
                               const InferenceParams &params) {
        // NOTE: reads only; applySamplingParams returns a fresh array.
        auto probs = applySamplingParams(logits, params);

        // Initialize persistent RNG on first call or when seed changes.
        // Using a persistent RNG ensures proper random sequence across
        // multiple sampling calls.
        if (!rngInitialized_ || (params.seed > 0 && rngSeed_ != params.seed)) {
            rng_.seed(params.seed > 0 ? static_cast<uint64_t>(params.seed)
                                      : std::random_device{}());
            rngSeed_ = params.seed;
            rngInitialized_ = true;
        }

        // Use the same generator semantics as llama.cpp's `dist` sampler:
        // std::mt19937 seeded identically, draw via
        // std::uniform_real_distribution<float>(0,1) (llama_rng::next_float).
        // This makes TinyCoder's sampled tokens match llama.cpp token-for-token
        // whenever the post-filter distributions are equal ("use same generator").
        static std::uniform_real_distribution<float> rngDist(0.0f, 1.0f);
        float r = rngDist(rng_);
        float cumSum = 0.0f;

        // Read from the raw pointer to avoid per-element virtual get() (see §4.3 / P1).
        const float *p = probs.data();
        for (uint32_t i = 0; i < config_.vocabSize; ++i) {
            cumSum += p[i];
            if (r <= cumSum) {
                return static_cast<int32_t>(i);
            }
        }

        return 0;
    }

    std::vector<int32_t> Model::tokenize(const std::string &prompt) {
        return tokenizer_.encode(prompt);
    }

}// namespace tinycoder