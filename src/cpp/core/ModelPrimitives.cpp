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

#include "ModelInternal.hpp"
#include "ThreadPool.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace tinycoder {

    // Internal helpers shared with the other Model translation units (profiler
    // and FP16/Q8_K batch mat-mul helpers) live in tinycoder::detail. Importing
    // them here lets the original unqualified call sites keep working verbatim.
    using namespace detail;

    void Model::rmsNormInPlace(const float *x, float *out, const float *weight,
                               uint32_t n, float eps) const {
        // Delegate to SIMD-accelerated rmsNormSIMD
        (void) eps;
        rmsNormSIMD(x, out, weight, n);
    }

    void Model::ensureRoPETables(uint32_t neededPos) {
        uint32_t headDim = config_.headDim;
        float theta = config_.ropeTheta;
        uint32_t pairs = headDim / 2;

        // Rebuild if the model's geometry changed (new model load).
        if (ropeHeadDim_ != headDim || ropeTheta_ != theta) {
            ropeHeadDim_ = headDim;
            ropeTheta_ = theta;
            ropePairs_ = pairs;
            ropeTablePos_ = 0;
            ropeCosTable_.clear();
            ropeSinTable_.clear();
        }

        if (neededPos <= ropeTablePos_) {
            return;
        }
        uint32_t oldRows = ropeTablePos_;
        ropeCosTable_.resize(static_cast<size_t>(neededPos) * pairs);
        ropeSinTable_.resize(static_cast<size_t>(neededPos) * pairs);

        // freq[d2] = theta^(-2*d2/headDim) = 1/pow(theta, 2*d2/headDim)
        float freq[128];// max headDim/2 is 64
        for (uint32_t d2 = 0; d2 < pairs; ++d2) {
            freq[d2] = 1.0f / std::pow(theta,
                                       static_cast<float>(2 * d2) /
                                               static_cast<float>(headDim));
        }
        for (uint32_t p = oldRows; p < neededPos; ++p) {
            for (uint32_t d2 = 0; d2 < pairs; ++d2) {
                float angle = static_cast<float>(p) * freq[d2];
                ropeCosTable_[static_cast<size_t>(p) * pairs + d2] =
                        std::cos(angle);
                ropeSinTable_[static_cast<size_t>(p) * pairs + d2] =
                        std::sin(angle);
            }
        }
        ropeTablePos_ = neededPos;
    }

    void Model::applyRoPE(float *q, float *k, uint32_t qSeqLen, uint32_t kSeqLen,
                          uint32_t qHeads, uint32_t kHeads, uint32_t pos,
                          bool rotateK) {
        uint32_t headDim = config_.headDim;
        uint32_t pairs = headDim / 2;
        uint32_t neededPos = pos + std::max(qSeqLen, kSeqLen);
        ensureRoPETables(neededPos);

        // Apply to Q using the precomputed cos/sin tables. The rotation
        // formula is bit-identical to the original (same std::cos/std::sin of the
        // same angle, same freq definition); only the redundant per-(pos,dim,head)
        // trig recomputation is removed.
        for (uint32_t s = 0; s < qSeqLen; ++s) {
            uint32_t p = pos + s;
            const float *cosRow = ropeCosTable_.data() +
                                  static_cast<size_t>(p) * pairs;
            const float *sinRow = ropeSinTable_.data() +
                                  static_cast<size_t>(p) * pairs;
            for (uint32_t h = 0; h < qHeads; ++h) {
                float *head = q + (static_cast<size_t>(s) * qHeads + h) * headDim;
                for (uint32_t j = 0; j < pairs; ++j) {
                    float c = cosRow[j];
                    float sn = sinRow[j];
                    float x0 = head[2 * j];
                    float x1 = head[2 * j + 1];
                    head[2 * j] = x0 * c - x1 * sn;
                    head[2 * j + 1] = x0 * sn + x1 * c;
                }
            }
        }

        // Apply to K (skipped when rotateK == false; the caller fuses the K
        // rotation into the KV-cache store via storeKVWithRoPE, P3).
        if (!rotateK) {
            return;
        }
        for (uint32_t s = 0; s < kSeqLen; ++s) {
            uint32_t p = pos + s;
            const float *cosRow = ropeCosTable_.data() +
                                  static_cast<size_t>(p) * pairs;
            const float *sinRow = ropeSinTable_.data() +
                                  static_cast<size_t>(p) * pairs;
            for (uint32_t h = 0; h < kHeads; ++h) {
                float *head = k + (static_cast<size_t>(s) * kHeads + h) * headDim;
                for (uint32_t j = 0; j < pairs; ++j) {
                    float c = cosRow[j];
                    float sn = sinRow[j];
                    float x0 = head[2 * j];
                    float x1 = head[2 * j + 1];
                    head[2 * j] = x0 * c - x1 * sn;
                    head[2 * j + 1] = x0 * sn + x1 * c;
                }
            }
        }
    }

    void Model::storeKVWithRoPE(const float *kSrc, const float *vSrc, float *kDst,
                                float *vDst, uint32_t seqLen, uint32_t cachePos,
                                uint32_t kHeads) {
        uint32_t headDim = config_.headDim;
        uint32_t pairs = headDim / 2;
        uint32_t kvSize = kHeads * headDim;
        for (uint32_t s = 0; s < seqLen; ++s) {
            uint32_t p = cachePos + s;
            const float *cosRow = ropeCosTable_.data() +
                                  static_cast<size_t>(p) * pairs;
            const float *sinRow = ropeSinTable_.data() +
                                  static_cast<size_t>(p) * pairs;
            const float *ks = kSrc + static_cast<size_t>(s) * kvSize;
            const float *vs = vSrc + static_cast<size_t>(s) * kvSize;
            float *kd = kDst + static_cast<size_t>(cachePos + s) * kvSize;
            float *vd = vDst + static_cast<size_t>(cachePos + s) * kvSize;
            for (uint32_t h = 0; h < kHeads; ++h) {
                const float *kh = ks + static_cast<size_t>(h) * headDim;
                float *kdh = kd + static_cast<size_t>(h) * headDim;
                for (uint32_t j = 0; j < pairs; ++j) {
                    float c = cosRow[j];
                    float sn = sinRow[j];
                    float x0 = kh[2 * j];
                    float x1 = kh[2 * j + 1];
                    kdh[2 * j] = x0 * c - x1 * sn;
                    kdh[2 * j + 1] = x0 * sn + x1 * c;
                }
            }
            std::memcpy(vd, vs, kvSize * sizeof(float));
        }
    }

    void Model::attentionFused(const float *q, const float *kCache,
                               const float *vCache, float *output, uint32_t seqLen,
                               uint32_t cachePos, uint32_t /*cacheLen*/,
                               uint32_t /*layerIdx*/) {
        ScopedProfile sp("attentionFused");
        // FlashAttention-style fused kernel (P0 in plans/prefill_optimization_plan.md).
        //
        // Replaces the previous three-pass double-precision attention (score to
        // softmax to weighted sum over the full K/V cache, with materialized
        // localScores[] / probs[] arrays and scalar std::exp) with a single fused
        // pass using the online (rescaled) softmax. Key wins:
        //   1. One pass over K/V per query: the score matrix / probs array are
        //      never materialized.
        //   2. Online softmax maintains running max `m` and sum `l`, rescaling the
        //      accumulated output only when a new max is found.
        //   3. FP32 accumulation everywhere (was FP64), with the dominant o*v
        //      accumulation vectorized via the dispatched accumulateFMA() kernel.
        //   4. exp2f() for the softmax probabilities (log2e-scaled), avoiding the
        //      scalar std::exp call per cache slot.
        //
        // The causal mask is handled by bounding the cache loop to csEnd =
        // cachePos + s (this query token's own cache position), so no explicit
        // -inf masking is needed.
        uint32_t nHeads = config_.numAttentionHeads;
        uint32_t nKVHeads = config_.numKVHeads;
        uint32_t headDim = config_.headDim;
        uint32_t nGroups = nHeads / nKVHeads;

        float invSqrtHeadDim = 1.0f / std::sqrt(static_cast<float>(headDim));
        constexpr float LOG2E = 1.4426950408889634f;
        // Fixed stack max for headDim (headDim <= 128 for the supported models).
        constexpr uint32_t MAX_HEAD_DIM = 128;
        // Cache positions per block processed before the next rescale. Blocking
        // amortises the (rare) rescale of the o accumulator across several
        // positions and improves cache reuse of K/V.
        constexpr uint32_t BLOCK = 32;

        ThreadPool::instance().parallelFor2D(seqLen, nKVHeads,
                                             [&](uint32_t s, uint32_t g) {
                                                 // Causal mask: this query token can attend to cache positions
                                                 // up to (cachePos + s): its own position in the cache.
                                                 uint32_t csEnd = cachePos + s;// inclusive end position

                                                 for (uint32_t h = 0; h < nGroups; ++h) {
                                                     uint32_t qHead = g * nGroups + h;
                                                     const float *qPtr =
                                                             q + (s * nHeads * headDim + qHead * headDim);

                                                     // FlashAttention online-softmax accumulators.
                                                     float o[MAX_HEAD_DIM];
                                                     for (uint32_t d = 0; d < headDim; ++d) {
                                                         o[d] = 0.0f;
                                                     }
                                                     float m = -std::numeric_limits<float>::infinity();
                                                     float l = 0.0f;

                                                     uint32_t cs = 0;
                                                     // Blocked main loop.
                                                     for (; cs + BLOCK <= csEnd + 1; cs += BLOCK) {
                                                         float scores[BLOCK];
                                                         for (uint32_t b = 0; b < BLOCK; ++b) {
                                                             const float *kPtr = kCache +
                                                                                 ((cs + b) * nKVHeads * headDim +
                                                                                  g * headDim);
                                                             scores[b] =
                                                                     dotProductFMA(qPtr, kPtr, headDim) *
                                                                     invSqrtHeadDim;
                                                         }
                                                         // Block-local max, then rescale accumulated output/sum.
                                                         float blockMax = scores[0];
                                                         for (uint32_t b = 1; b < BLOCK; ++b) {
                                                             blockMax = std::max(blockMax, scores[b]);
                                                         }
                                                         float mNew = std::max(m, blockMax);
                                                         float alpha = std::exp2f((m - mNew) * LOG2E);
                                                         if (alpha != 1.0f) {
                                                             // o = o * alpha, implemented as o += (alpha - 1) * o
                                                             accumulateFMA(o, o, alpha - 1.0f, headDim);
                                                         }
                                                         l *= alpha;
                                                         for (uint32_t b = 0; b < BLOCK; ++b) {
                                                             const float *vPtr = vCache +
                                                                                 ((cs + b) * nKVHeads * headDim +
                                                                                  g * headDim);
                                                             float p = std::exp2f((scores[b] - mNew) * LOG2E);
                                                             l += p;
                                                             // o += p * v (vectorized via dispatched accumulateFMA)
                                                             accumulateFMA(o, vPtr, p, headDim);
                                                         }
                                                         m = mNew;
                                                     }
                                                     // Scalar tail (partial block / final positions).
                                                     for (; cs <= csEnd; ++cs) {
                                                         const float *kPtr = kCache +
                                                                             (cs * nKVHeads * headDim +
                                                                              g * headDim);
                                                         float score = dotProductFMA(qPtr, kPtr, headDim) *
                                                                       invSqrtHeadDim;
                                                         if (score > m) {
                                                             float mNew = score;
                                                             float alpha = std::exp2f((m - mNew) * LOG2E);
                                                             if (alpha != 1.0f) {
                                                                 accumulateFMA(o, o, alpha - 1.0f, headDim);
                                                             }
                                                             l *= alpha;
                                                             m = mNew;
                                                         }
                                                         const float *vPtr = vCache +
                                                                             (cs * nKVHeads * headDim +
                                                                              g * headDim);
                                                         float p = std::exp2f((score - m) * LOG2E);
                                                         l += p;
                                                         accumulateFMA(o, vPtr, p, headDim);
                                                     }

                                                     float invL = 1.0f / l;
                                                     float *outPtr =
                                                             output + (s * nHeads * headDim + qHead * headDim);
                                                     for (uint32_t d = 0; d < headDim; ++d) {
                                                         outPtr[d] = o[d] * invL;
                                                     }
                                                 }
                                             });
    }

    void Model::siluInPlace(float *x, uint32_t n) {
        for (uint32_t i = 0; i < n; ++i) {
            x[i] = x[i] / (1.0f + std::exp(-x[i]));
        }
    }

    void Model::swiGLUInPlace(float *x, const float *y, uint32_t n) {
        // x = silu(x) * y, stored in-place in x
        for (uint32_t i = 0; i < n; ++i) {
            float siluVal = x[i] / (1.0f + std::exp(-x[i]));
            x[i] = siluVal * y[i];
        }
    }

    void Model::geluInPlace(float *x, uint32_t n) const {
        // GeLU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        constexpr float sqrt2OverPi = 0.7978845608028654f;// sqrt(2.0 / M_PI)
        for (uint32_t i = 0; i < n; ++i) {
            float x3 = x[i] * x[i] * x[i];
            x[i] = 0.5f * x[i] * (1.0f + std::tanh(sqrt2OverPi * (x[i] + 0.044715f * x3)));
        }
    }

    void Model::softcapInPlace(float *x, uint32_t n, float cap) {
        // tanh(x / cap) * cap
        if (cap > 0.0f) {
            float invCap = 1.0f / cap;
            for (uint32_t i = 0; i < n; ++i) {
                x[i] = std::tanh(x[i] * invCap) * cap;
            }
        }
    }

    void Model::applyQKNorms(float *q, float *k, uint32_t seqLen,
                             uint32_t qHeads, uint32_t kHeads,
                             const float *qNorm, const float *kNorm) {
        // Per-head RMSNorm applied to Q and K before RoPE.
        // q layout: [seqLen, qHeads, headDim]
        // k layout: [seqLen, kHeads, headDim]
        // qNorm/kNorm: [headDim] (shared across all heads)
        uint32_t headDim = config_.headDim;
        for (uint32_t s = 0; s < seqLen; ++s) {
            for (uint32_t h = 0; h < qHeads; ++h) {
                float *qHead = q + (s * qHeads + h) * headDim;
                rmsNormInPlace(qHead, qHead, qNorm, headDim);
            }
            for (uint32_t h = 0; h < kHeads; ++h) {
                float *kHead = k + (s * kHeads + h) * headDim;
                rmsNormInPlace(kHead, kHead, kNorm, headDim);
            }
        }
    }

}// namespace tinycoder