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

/**
 * TinyCoder GPU vs CPU comparison tests.
 *
 * These tests run the SAME forward pass through both the CPU engine and the
 * GPU offload engine and compare the resulting logits. The goal is to catch
 * GPU-only regressions that produce "garbage" output while the CPU path is
 * correct (e.g. in-place RMSNorm destroying the residual stream, or an
 * RMSNorm launch grid that only normalizes the first token).
 *
 * The comparison is intentionally lossy-tolerant:
 *  - Q2_K quantization is inherently lossy vs the CPU reference math
 *    (fp16 GEMM prefill vs fp32), so exact logit equality is NOT expected.
 *  - The top-1/top-5 token *identity* and the logit magnitude/scale must
 *    agree - a broken GPU engine produces a nearly flat or random
 *    distribution whose argmax diverges.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "Model.hpp"
#include "ModelConfig.hpp"
#include "SharedTestEnv.hpp"
#include "Tokenizer.hpp"

#ifdef USE_CUDA
#include "GPUCompute.hpp"
#endif

// ---------------------------------------------------------------------------
// Fixture + helpers
// ---------------------------------------------------------------------------

class GPUCpuCompareTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_NE(SharedTestEnv::model, nullptr) << "Model not loaded -- skipping test";
        ASSERT_TRUE(SharedTestEnv::modelLoaded);
    }
};

/// @brief The chat-formatted Paris prompt (the failing generation in the bug
/// report: 40 prompt tokens).
static std::string parisChatPrompt() {
    return SharedTestEnv::model->formatChat(
            {{"system", "You are TinyCoder, an AI coding assistant. Be concise."},
             {"user", "What is the capital of France?"}},
            true);
}

/// @brief Normalize a forward() result to the flat LAST-token logit row.
///
/// Shape normalization: the CPU engine allocates the full [seqLen, vocabSize]
/// array but (with computeAllLogits=false) only computes the LAST token's logit
/// row - the other rows are uninitialized padding. The GPU engine returns a
/// compact [1, vocabSize] array. Normalize both to the flat last-token row so
/// the comparisons operate on identical shapes.
static np::Array<float> lastLogitRow(np::Array<float> logits) {
    const uint32_t vocab = SharedTestEnv::config.vocabSize;
    if (logits.size() <= vocab) {
        // Single-token pass: shapes already agree ([1, vocabSize]).
        return logits;
    }

    // Multi-token pass: strip everything except the last token's row.
    np::Array<float> lastRow = np::Array<float>(np::Shape{vocab});
    const float *src = logits.data() + (logits.size() - vocab);
    std::memcpy(lastRow.data(), src, vocab * sizeof(float));
    return lastRow;
}

/// @brief Run one forward pass with the GPU engine explicitly enabled or
/// disabled via $TINYCODER_GPU, returning ONLY the last token's logits.
/// Clears BOTH KV caches first (fresh-session semantics).
static np::Array<float> forwardLastLogits(const std::vector<int32_t> &tokens,
                                          bool wantGpu) {
    if (wantGpu) {
        setenv("TINYCODER_GPU", "1", 1);
    } else {
        setenv("TINYCODER_GPU", "0", 1);
    }
    SharedTestEnv::model->clearKVCache();
    return lastLogitRow(SharedTestEnv::model->forward(tokens,
                                                      /*computeAllLogits=*/false));
}

/// @brief Top-N (logit, id) pairs in descending logit order.
static std::vector<std::pair<float, int32_t>> topN(const np::Array<float> &logits,
                                                   size_t n) {
    std::vector<std::pair<float, int32_t>> v;
    v.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        v.emplace_back(logits.get(static_cast<uint32_t>(i)),
                       static_cast<int32_t>(i));
    }
    std::partial_sort(v.begin(), v.begin() + std::min(n, v.size()), v.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });
    v.resize(std::min(n, v.size()));
    return v;
}

static void printTop(tinycoder::Tokenizer &tok, const std::string &label,
                     const np::Array<float> &logits, size_t n) {
    std::cout << "  " << label << ":";
    auto top = topN(logits, n);
    for (const auto &kv: top) {
        std::string text = tok.decodeToken(kv.second);
        std::cout << " [" << kv.second << " " << text << " logit=" << kv.first
                  << "]";
    }
    std::cout << std::endl;
}

// ---------------------------------------------------------------------------
// GPU vs CPU logits comparison
// ---------------------------------------------------------------------------

// The GPU engine is compiled in only when USE_CUDA is defined. Without CUDA
// this test trivially passes (both "GPU" and "CPU" runs are the CPU engine).
#ifdef USE_CUDA

TEST_F(GPUCpuCompareTest, ParisPromptLogitsAgree) {
    tinycoder::Model *model = SharedTestEnv::model;
    tinycoder::Tokenizer &tokenizer = model->tokenizer();

    // Skip when no usable CUDA device is present (the GPU forward falls back
    // to CPU internally, so the test would compare CPU vs CPU).
    if (!tinycoder::gpu::gpuEnabled()) {
        GTEST_SKIP() << "GPU disabled ($TINYCODER_GPU=0) or no CUDA device";
    }

    std::string prompt = parisChatPrompt();
    auto tokens = tokenizer.encode(prompt);
    ASSERT_GT(tokens.size(), 1u) << "Prompt must tokenize to >= 2 tokens";

    std::cout << "GPUCpuCompareTest: " << prompt.size() << " chars -> "
              << tokens.size() << " prompt tokens" << std::endl;

    // CPU logits (the known-good reference path).
    np::Array<float> cpuLogits = forwardLastLogits(tokens, /*wantGpu=*/false);
    ASSERT_FALSE(cpuLogits.empty());

    // GPU logits (the offload engine).
    np::Array<float> gpuLogits = forwardLastLogits(tokens, /*wantGpu=*/true);
    ASSERT_FALSE(gpuLogits.empty());

    const uint32_t vocab = static_cast<uint32_t>(cpuLogits.size());
    ASSERT_EQ(gpuLogits.size(), cpuLogits.size());

    printTop(tokenizer, "CPU top-5", cpuLogits, 5);
    printTop(tokenizer, "GPU top-5", gpuLogits, 5);

    auto cpuTop = topN(cpuLogits, 5);
    auto gpuTop = topN(gpuLogits, 5);
    ASSERT_FALSE(cpuTop.empty());
    ASSERT_FALSE(gpuTop.empty());

    // The top-1 token must agree.  A GPU engine that silently computes garbage
    // (broken RMSNorm / residual stream) yields a nearly flat distribution and
    // a different argmax.
    EXPECT_EQ(cpuTop[0].second, gpuTop[0].second)
            << "GPU top-1 token differs from CPU top-1: cpu=" << cpuTop[0].second
            << " gpu=" << gpuTop[0].second;

    // Allow >= 60% of the CPU top-5 to appear in the GPU top-5 (Q2_K noise).
    size_t overlap = 0;
    for (const auto &c: cpuTop) {
        for (const auto &g: gpuTop) {
            if (c.second == g.second) {
                ++overlap;
                break;
            }
        }
    }
    EXPECT_GE(overlap, 3u) << "GPU top-5 shares too few tokens with CPU top-5";

    // Logits must have comparable magnitude/scale.  Garbage distributions are
    // typically flat (near-zero range) or wildly inflated.
    float cpuMax = -std::numeric_limits<float>::max();
    float gpuMax = -std::numeric_limits<float>::max();
    float cpuMin = std::numeric_limits<float>::max();
    float gpuMin = std::numeric_limits<float>::max();
    for (uint32_t i = 0; i < vocab; ++i) {
        cpuMax = std::max(cpuMax, cpuLogits.get(i));
        cpuMin = std::min(cpuMin, cpuLogits.get(i));
        gpuMax = std::max(gpuMax, gpuLogits.get(i));
        gpuMin = std::min(gpuMin, gpuLogits.get(i));
    }
    const float cpuRange = cpuMax - cpuMin;
    const float gpuRange = gpuMax - gpuMin;
    std::cout << "  CPU logit range: [" << cpuMin << ", " << cpuMax
              << "] (width " << cpuRange << ")" << std::endl;
    std::cout << "  GPU logit range: [" << gpuMin << ", " << gpuMax
              << "] (width " << gpuRange << ")" << std::endl;
    // The GPU prefill uses fp16 GEMMs (lossy vs CPU fp32) but the global scale
    // must match within a generous factor.  A garbage engine has a range off
    // by >> 2x.
    EXPECT_GT(cpuRange, 1.0f) << "CPU logits unexpectedly flat -- test is broken";
    EXPECT_GT(gpuRange, cpuRange / 4.0f)
            << "GPU logit range collapsed vs CPU -- residual/RMSNorm bug";
    EXPECT_LT(gpuRange, cpuRange * 8.0f)
            << "GPU logit range inflated vs CPU -- garbage distribution";
}

// Single-token decode parity: replay an IDENTICAL token stream one token at a
// time through both engines (KV cache growing on each side) - the exact path
// from Model::generate's loop - and compare each step's top-1 token.
//
// Both engines consume the same fixed stream, so this is an apples-to-apples
// comparison: differences can only come from the GPU engine's own math, never
// from divergent context.
TEST_F(GPUCpuCompareTest, SequentialDecodeArgmaxAgrees) {
    tinycoder::Model *model = SharedTestEnv::model;
    tinycoder::Tokenizer &tokenizer = model->tokenizer();

    if (!tinycoder::gpu::gpuEnabled()) {
        GTEST_SKIP() << "GPU disabled ($TINYCODER_GPU=0) or no CUDA device";
    }

    // A short deterministic token stream (start-of-assistant + a few tokens),
    // exercising the multi-position KV-cache decode path.
    std::string prompt = parisChatPrompt();
    auto tokens = tokenizer.encode(prompt);
    ASSERT_GT(tokens.size(), 1u);

    std::vector<int32_t> stream(tokens.begin(), tokens.begin() + 8);
    std::cout << "GPUCpuCompareTest sequential: " << stream.size()
              << " tokens fed one-by-one" << std::endl;

    // CPU reference session: clear once, then let the KV cache grow.
    setenv("TINYCODER_GPU", "0", 1);
    model->clearKVCache();
    // Per-step CPU top-5 reference (the corruption signal: a broken GPU engine
    // ranks EOG/chat-special tokens first, which never appear in the CPU's top-5).
    std::vector<std::vector<std::pair<float, int32_t>>> cpuTop5;
    cpuTop5.reserve(stream.size());
    for (size_t i = 0; i < stream.size(); ++i) {
        auto logits = lastLogitRow(model->forward({stream[i]},
                                                  /*computeAllLogits=*/false));
        ASSERT_FALSE(logits.empty());
        cpuTop5.push_back(topN(logits, 5));
        ASSERT_FALSE(cpuTop5[i].empty());
    }

    // GPU session: same stream, own KV cache.
    setenv("TINYCODER_GPU", "1", 1);
    model->clearKVCache();
    for (size_t i = 0; i < stream.size(); ++i) {
        auto logits = lastLogitRow(model->forward({stream[i]},
                                                  /*computeAllLogits=*/false));
        ASSERT_FALSE(logits.empty());
        auto gpuTop = topN(logits, 5);
        ASSERT_FALSE(gpuTop.empty());

        // The two engines compute the SAME math with different precision on the
        // decode kernels (GPU streams quantized weights via warp-per-row GEMV vs
        // CPU's fp32 reference).  At position 0 / minimal-context steps the
        // per-vocabulary logits sit on sharp quantization noise, so an adjacent
        // token can flip CPU vs GPU top-1 within the same near-tie cluster.
        //
        // Assertion intent: catch GROSS corruption (the pre-fix symptom) where
        // the GPU ranks EOG/chat-special tokens (151644/151645/151643) first,
        // which never rank in the CPU's top-5.  Tolerate near-tie reorderings by
        // requiring the GPU top-1 token to be a plausible CPU-top-5 candidate.
        bool plausible = false;
        for (const auto &kv: cpuTop5[i]) {
            if (kv.second == gpuTop[0].second) {
                plausible = true;
                break;
            }
        }
        EXPECT_TRUE(plausible)
                << "Step " << i << " (token " << stream[i]
                << "): GPU top-1=" << gpuTop[0].second
                << " not among CPU top-5 (GPU distribution diverged - corruption?)";
    }
}

#else// !USE_CUDA

TEST_F(GPUCpuCompareTest, ParityTestSkippedWithoutCuda) {
    GTEST_SKIP() << "Built without USE_CUDA -- no GPU engine to compare";
}

#endif// USE_CUDA
