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

#pragma once

#include "Model.hpp"
#include "SIMDMatMulVec.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// Internal helpers shared between the Model translation units (the Model.cpp
// refactor split it into ModelLoad / ModelPrimitives / ModelMoE / ModelDebug /
// ModelForward / ModelForwardDebug / ModelSampling / ModelGeneration).
//
// Two groups live here:
//   1. The temporary prefill profiler (P2 investigation), gated by the
//      TINYCODER_PROFILE env var; prints accumulated wall time per named stage
//      at process exit. Diagnostic aid only.
//   2. The FP16 / Q8_K batch mat-mul helpers that were previously in an
//      anonymous namespace inside Model.cpp. They are shared by the main
//      forward pass, the debug forwards and the token-by-token forward, so they
//      now live in the tinycoder::detail namespace with external linkage.
//
// All declarations here are model-internal; do not use them outside Model*.cpp.
// ============================================================================

namespace tinycoder::detail {

    // ---------------------------------------------------------------------
    // Temporary prefill profiler (P2 investigation). Gated by TINYCODER_PROFILE
    // env var; prints accumulated wall time per named stage at process exit.
    // This is a diagnostic aid only and is removed after the bottleneck is
    // identified.
    // ---------------------------------------------------------------------
    struct StageProfile {
        std::chrono::steady_clock::duration total{};
        uint64_t calls = 0;
    };
    inline std::mutex g_profileMutex;
    inline std::unordered_map<std::string, StageProfile> g_profile;
    inline bool g_profileEnabled = [] {
        const char *e = std::getenv("TINYCODER_PROFILE");
        return e != nullptr && e[0] != '\0';
    }();

    class ScopedProfile {
    public:
        explicit ScopedProfile(const char *tag) : tag_(tag) {
            if (g_profileEnabled) start_ = std::chrono::steady_clock::now();
        }
        ~ScopedProfile() {
            if (!g_profileEnabled) return;
            auto now = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> lk(g_profileMutex);
            auto &p = g_profile[tag_];
            p.total += now - start_;
            ++p.calls;
        }

    private:
        const char *tag_;
        std::chrono::steady_clock::time_point start_;
    };
    struct ProfileDumper {
        ~ProfileDumper() {
            if (!g_profileEnabled) return;
            std::lock_guard<std::mutex> lk(g_profileMutex);
            std::cout << "\n=== FORWARD PROFILE (ms) ===\n";
            for (auto &kv: g_profile) {
                double ms = std::chrono::duration_cast<std::chrono::microseconds>(kv.second.total).count() / 1000.0;
                std::cout << "  " << kv.first << ": " << ms << " ms  (" << kv.second.calls << " calls)\n";
            }
            std::cout << "=============================\n";
        }
    };
    inline ProfileDumper g_profileDumper;

    // ---------------------------------------------------------------------
    // FP16 / Q8_K batched mat-mul helpers (moved verbatim from Model.cpp's
    // anonymous namespace; only the `static` qualifier was removed so they can
    // be shared across translation units).
    // ---------------------------------------------------------------------

    /// @brief Print min/max/mean and the first 8 elements of a vector (debug aid).
    void dumpVecStats(const float *v, uint32_t n, const std::string &label);

    /// @brief Matrix-vector multiply with FP16-stored weights (return-value version).
    ///        Computes y_j = sum_i x[i] * W_f16[j][i] for j in [0, rows).
    np::Array<float> deqMatMulVecF16(const uint16_t *W_f16, const float *x,
                                     uint32_t rows, uint32_t cols);

    /// @brief Matrix-vector multiply with FP16-stored weights (out-parameter version).
    void deqMatMulVecF16(const uint16_t *W_f16, const float *x,
                         uint32_t rows, uint32_t cols, float *out);

    /// @brief Fused QKV matrix-vector multiply.
    void matMulVecFusedQKV(const QuantizedMatrix &qMat, const QuantizedMatrix &kMat,
                           const QuantizedMatrix &vMat, const float *x,
                           float *qOut, float *kOut, float *vOut);

    /// @brief Batched matrix-matrix multiply with FP16-stored weights (prefill path).
    void deqMatMulVecF16_Batch(const uint16_t *W_f16, const float *X,
                               uint32_t seqLen, uint32_t rows, uint32_t cols,
                               float *out);

    /// @brief Register-tiled Q8_K batch GEMM (P4 prefill path).
    void matMulVecBatchQ8K(const Q8KBlock *W_q8k, const float *X,
                           uint32_t seqLen, uint32_t rows, uint32_t cols,
                           float *out);

    /// @brief Batched quantized matrix-matrix multiply (prefill path).
    void matMulVecBatchQuantized(const QuantizedMatrix &W, const float *X,
                                 uint32_t seqLen, float *out);

    /// @brief Batched fused gate+up matrix-matrix multiply (optionally fused SwiGLU).
    void matMulVecFusedGateUp_Batch(const QuantizedMatrix &gate,
                                    const QuantizedMatrix &up, const float *X,
                                    uint32_t seqLen, float *gateOut, float *upOut,
                                    bool applySwish);

}// namespace tinycoder::detail