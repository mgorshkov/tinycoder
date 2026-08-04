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
 * TinyCoder Model Unit Test (Google Test)
 *
 * Tests model loading, tokenization, forward pass, KV cache management,
 * and sample question answering.
 *
 * GPU acceleration is used automatically when the library is compiled
 * with ENABLE_CUDA=ON and a CUDA-capable GPU is available.
 *
 * Usage:
 *   TINYCODER_MODEL_PATH=<path_to_model.gguf> ./tinycoder_test
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

#include "GGUFLoader.hpp"
#include "Model.hpp"
#include "ModelConfig.hpp"
#include "SharedTestEnv.hpp"
#include "Tokenizer.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace fs = std::filesystem;
using namespace tinycoder;

// ---------------------------------------------------------------------------
// Test fixture: uses the model loaded once globally by SharedTestEnv
// ---------------------------------------------------------------------------

class ModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_NE(SharedTestEnv::model, nullptr) << "Model not loaded — skipping test";
        ASSERT_TRUE(SharedTestEnv::modelLoaded);
    }
};

// ---------------------------------------------------------------------------
// Test: GGUF Metadata Loading
// ---------------------------------------------------------------------------

TEST(GGUFMetadataTest, LoadMetadata) {
    ASSERT_FALSE(SharedTestEnv::modelPath.empty());
    ASSERT_TRUE(fs::exists(SharedTestEnv::modelPath));

    GGUFLoader loader;
    ASSERT_TRUE(loader.loadMetadata(SharedTestEnv::modelPath));

    const auto &cfg = loader.config();
    EXPECT_GT(cfg.numLayers, 0);
    EXPECT_GT(cfg.hiddenSize, 0);
    EXPECT_GT(cfg.numAttentionHeads, 0);
    EXPECT_GT(cfg.numKVHeads, 0);
    EXPECT_GT(cfg.vocabSize, 0);
    EXPECT_EQ(cfg.headDim, cfg.hiddenSize / cfg.numAttentionHeads);
}

// ---------------------------------------------------------------------------
// Test: Memory Estimation
// ---------------------------------------------------------------------------

TEST(GGUFMetadataTest, MemoryEstimation) {
    ASSERT_FALSE(SharedTestEnv::modelPath.empty());

    uint64_t estimatedBytes = Model::estimateMemory(SharedTestEnv::modelPath);
    EXPECT_GT(estimatedBytes, 0);
}

// ---------------------------------------------------------------------------
// Test: Model Loading
// ---------------------------------------------------------------------------

TEST_F(ModelTest, ModelLoaded) {
    ASSERT_NE(SharedTestEnv::model, nullptr);
    ASSERT_TRUE(SharedTestEnv::modelLoaded);

    const auto &config = SharedTestEnv::config;
    EXPECT_GT(config.numLayers, 0);
    EXPECT_GT(config.hiddenSize, 0);
    EXPECT_GT(config.numAttentionHeads, 0);
    EXPECT_GT(config.numKVHeads, 0);
    EXPECT_GT(config.vocabSize, 0);
    EXPECT_GT(config.intermediateSize, 0);
    EXPECT_GT(config.headDim, 0);
    EXPECT_GT(config.ropeTheta, 0.0f);
}

// ---------------------------------------------------------------------------
// Test: Tokenizer Encoding/Decoding
// ---------------------------------------------------------------------------

TEST_F(ModelTest, TokenizerEncodeDecode) {
    auto &tokenizer = SharedTestEnv::model->tokenizer();

    // Test encoding
    std::string testStr = "Hello";
    auto tokens = tokenizer.encode(testStr);
    EXPECT_GT(tokens.size(), 0);

    // Test round-trip
    std::string decoded;
    for (auto t: tokens) {
        decoded += tokenizer.decodeToken(t);
    }
    EXPECT_FALSE(decoded.empty());

    // Test special tokens
    int32_t bos = tokenizer.bosTokenId();
    int32_t eos = tokenizer.eosTokenId();
    EXPECT_GE(bos, 0);
    EXPECT_GE(eos, 0);
}

// ---------------------------------------------------------------------------
// Test: Single Token Forward Pass
// ---------------------------------------------------------------------------

TEST_F(ModelTest, SingleTokenForward) {
    auto &tokenizer = SharedTestEnv::model->tokenizer();
    auto tokens = tokenizer.encode("Hello");
    ASSERT_GT(tokens.size(), 0);

    SharedTestEnv::model->clearKVCache();
    auto logits = SharedTestEnv::model->forward({tokens[0]});

    const auto &config = SharedTestEnv::config;
    EXPECT_EQ(logits.size(), config.vocabSize);

    // Check logits are finite
    bool hasNan = false, hasInf = false;
    for (uint32_t i = 0; i < logits.size(); ++i) {
        if (std::isnan(logits.get(i))) hasNan = true;
        if (std::isinf(logits.get(i))) hasInf = true;
    }
    EXPECT_FALSE(hasNan) << "Logits contain NaN";
    EXPECT_FALSE(hasInf) << "Logits contain Inf";
}

// ---------------------------------------------------------------------------
// Test: Multi-Token Forward Pass
// ---------------------------------------------------------------------------

TEST_F(ModelTest, MultiTokenForward) {
    auto &tokenizer = SharedTestEnv::model->tokenizer();
    auto tokens = tokenizer.encode("Hello world");
    ASSERT_GT(tokens.size(), 1);

    SharedTestEnv::model->clearKVCache();
    auto logits = SharedTestEnv::model->forward(tokens);

    const auto &config = SharedTestEnv::config;
    EXPECT_EQ(logits.size(), tokens.size() * config.vocabSize);

    // Check logits are finite
    bool hasNan = false, hasInf = false;
    for (uint32_t i = 0; i < logits.size(); ++i) {
        if (std::isnan(logits.get(i))) hasNan = true;
        if (std::isinf(logits.get(i))) hasInf = true;
    }
    EXPECT_FALSE(hasNan) << "Logits contain NaN";
    EXPECT_FALSE(hasInf) << "Logits contain Inf";
}

// ---------------------------------------------------------------------------
// Test: KV Cache Management
// ---------------------------------------------------------------------------

TEST_F(ModelTest, KVCacheClear) {
    auto &tokenizer = SharedTestEnv::model->tokenizer();
    auto tokens = tokenizer.encode("Hello");
    ASSERT_GT(tokens.size(), 0);

    // First forward pass
    SharedTestEnv::model->clearKVCache();
    auto logits1 = SharedTestEnv::model->forward({tokens[0]});

    // Second forward pass (should use KV cache)
    auto logits2 = SharedTestEnv::model->forward({tokens[0]});

    // Clear and re-run
    SharedTestEnv::model->clearKVCache();
    auto logits3 = SharedTestEnv::model->forward({tokens[0]});

    // logits1 and logits3 should match (both start fresh)
    const auto &config = SharedTestEnv::config;
    float maxDiff = 0.0f;
    for (uint32_t i = 0; i < config.vocabSize; ++i) {
        float diff = std::abs(logits1.get(i) - logits3.get(i));
        maxDiff = std::max(maxDiff, diff);
    }
    EXPECT_LT(maxDiff, 1e-4f) << "KV cache clear should produce identical results";
}

// ---------------------------------------------------------------------------
// Test: Token Generation (basic)
// ---------------------------------------------------------------------------

TEST_F(ModelTest, GenerateTokens) {
    InferenceParams params;
    params.maxTokens = 10;
    params.temperature = 0.0f;// greedy decoding for reproducibility
    params.seed = 42;

    std::string generatedText;
    int tokenCount = 0;

    SharedTestEnv::model->generate("Hello", params,
                                   [&](int32_t /*token*/, const std::string &text) -> bool {
                                       generatedText += text;
                                       tokenCount++;
                                       return true;
                                   });

    EXPECT_GT(tokenCount, 0);
    EXPECT_FALSE(generatedText.empty());
}

// ---------------------------------------------------------------------------
// Test: Sample question answering (parameterized)
// ---------------------------------------------------------------------------

struct QuestionAnswer {
    std::string question;
    std::string expectedKeyword; // keyword expected in the answer
    std::string forbiddenKeyword;// keyword that must NOT appear (language guard)
    int minTokens;               // minimum expected tokens
};

class SampleQuestionTest
    : public ModelTest,
      public ::testing::WithParamInterface<QuestionAnswer> {
public:
    static bool quickMode;
};
bool SampleQuestionTest::quickMode = false;

TEST_P(SampleQuestionTest, AnswersQuestion) {
    const auto &qa = GetParam();

    // Use formatChat() to build the prompt in an architecture-aware way
    std::string prompt = SharedTestEnv::model->formatChat(
            {{"system", "You are TinyCoder, an AI coding assistant. Be concise."},
             {"user", qa.question}},
            true);

    InferenceParams params;
    params.maxTokens = 64;
    params.temperature = 0.7f;
    params.topP = 0.9f;
    params.topK = 40;
    params.repeatPenalty = 1.1f;
    params.seed = 42;

    std::string generatedText;
    int tokenCount = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    SharedTestEnv::model->generate(prompt, params,
                                   [&](int32_t /*token*/, const std::string &text) -> bool {
                                       generatedText += text;
                                       tokenCount++;
                                       return true;
                                   });
    auto t1 = std::chrono::high_resolution_clock::now();
    auto genMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    float tokPerSec = tokenCount > 0 ? (tokenCount / (genMs / 1000.0f)) : 0.0f;

    std::cout << "  Q: " << qa.question << std::endl;
    std::cout << "  Generated " << tokenCount << " tokens in " << genMs << " ms ("
              << tokPerSec << " tok/s)" << std::endl;
    std::cout << "  Output: \"" << generatedText << "\"" << std::endl;
    std::cout << "  (Prefill and generation timing printed by Model::generate above)"
              << std::endl;

    // Check minimum token count
    EXPECT_GE(tokenCount, qa.minTokens);

    // Check keyword presence (case-insensitive)
    std::string lowerOutput = generatedText;
    std::string lowerKeyword = qa.expectedKeyword;
    std::transform(lowerOutput.begin(), lowerOutput.end(), lowerOutput.begin(),
                   ::tolower);
    std::transform(lowerKeyword.begin(), lowerKeyword.end(), lowerKeyword.begin(),
                   ::tolower);
    EXPECT_NE(lowerOutput.find(lowerKeyword), std::string::npos)
            << "Expected keyword '" << qa.expectedKeyword << "' not found in output";

    // Language guard: a forbidden keyword (e.g. model answering the C++ question
    // in Python) must NOT appear. Catches the lossy Q2_K re-quant quality
    // regression where Q1 starts emitting "def add(a, b):" instead of C++.
    if (!qa.forbiddenKeyword.empty()) {
        std::string lowerForbidden = qa.forbiddenKeyword;
        std::transform(lowerForbidden.begin(), lowerForbidden.end(),
                       lowerForbidden.begin(), ::tolower);
        EXPECT_EQ(lowerOutput.find(lowerForbidden), std::string::npos)
                << "Output must NOT contain forbidden keyword '"
                << qa.forbiddenKeyword << "'";
    }

    // Clear KV cache between questions
    SharedTestEnv::model->clearKVCache();
}

// Define test questions
// Note: These are end-to-end generation tests with sampling. The model is
// heavily quantized (IQ3_XXS, 3.06 bpw), so output quality is limited.
// Keywords are chosen to match the model's actual sampled output.
// Language-guard keywords: Q1 and Q3 ask for C++ code, so a bare-Python answer
// ("def ...:") is a regression signal (lossy Q2_K re-quant). Q2 (factual) and
// Q4 (Python) have no forbidden keyword.
static const QuestionAnswer fullQuestions[] = {
        {"Write a C++ function to add two numbers.", "return", "def ", 10},
        {"What is the capital of France?", "Paris", "", 5},
        {"Explain what a pointer is in C++.", "address", "def ", 10},
        {"Write a for loop in Python that prints numbers 1 to 5.", "for", "", 10},
};

INSTANTIATE_TEST_SUITE_P(FullQuestions, SampleQuestionTest,
                         ::testing::ValuesIn(fullQuestions));

// ===========================================================================
// Diagnostic test: Process Paris prompt tokens one by one and track how
// results change as the KV cache grows. This tests whether the bug is in
// the FIRST token (position 0) or only appears with a multi-position cache.
// ===========================================================================
TEST_F(ModelTest, TracePrefillTokenByToken) {
    std::string question = "What is the capital of France?";
    std::string prompt = SharedTestEnv::model->formatChat(
            {{"system", "You are TinyCoder, an AI coding assistant. Be concise."},
             {"user", question}},
            true);

    auto &tokenizer = SharedTestEnv::model->tokenizer();
    const auto &config = SharedTestEnv::config;
    uint32_t vocabSize = config.vocabSize;

    auto tokens = tokenizer.encode(prompt);
    std::cout << "Prompt: " << tokens.size() << " tokens" << std::endl;

    // First, compute the "fresh" result for each token alone at position 0
    // This is the same code path tested by CompareLogitsWithReference
    std::cout << "\n=== Fresh single-token results ===" << std::endl;
    std::cout << "(each token processed alone at position 0, should all be correct)" << std::endl;
    for (size_t t = 0; t < std::min(tokens.size(), size_t(5)); ++t) {
        SharedTestEnv::model->clearKVCache();
        auto logits = SharedTestEnv::model->forward({tokens[t]});

        std::vector<std::pair<float, int32_t>> top5;
        for (uint32_t i = 0; i < vocabSize; ++i) {
            top5.emplace_back(logits.get(i), static_cast<int32_t>(i));
        }
        std::partial_sort(top5.begin(), top5.begin() + 5, top5.end(),
                          [](const auto &a, const auto &b) { return a.first > b.first; });

        std::string text = tokenizer.decodeToken(tokens[t]);
        std::cout << "Token " << t << ": id=" << tokens[t] << " text=\"";
        for (char c: text) {
            if (c >= 32 && c < 127) std::cout << c;
            else
                std::cout << "\\x" << std::hex << (unsigned) (unsigned char) c << std::dec;
        }
        std::cout << "\"" << std::endl;
        std::cout << "  top-1: id=" << top5[0].second
                  << " logit=" << top5[0].first
                  << " text=\"" << tokenizer.decodeToken(top5[0].second) << "\"" << std::endl;
    }

    // Now process tokens sequentially and check how results change
    std::cout << "\n=== Sequential processing ===" << std::endl;
    std::cout << "(building KV cache incrementally - tests multi-position cache code path)" << std::endl;
    SharedTestEnv::model->clearKVCache();

    for (size_t t = 0; t < std::min(tokens.size(), size_t(5)); ++t) {
        auto logits = SharedTestEnv::model->forward({tokens[t]});

        std::vector<std::pair<float, int32_t>> top5;
        for (uint32_t i = 0; i < vocabSize; ++i) {
            top5.emplace_back(logits.get(i), static_cast<int32_t>(i));
        }
        std::partial_sort(top5.begin(), top5.begin() + 5, top5.end(),
                          [](const auto &a, const auto &b) { return a.first > b.first; });

        std::string text = tokenizer.decodeToken(tokens[t]);
        std::cout << "Token " << t << ": id=" << tokens[t] << " text=\"";
        for (char c: text) {
            if (c >= 32 && c < 127) std::cout << c;
            else
                std::cout << "\\x" << std::hex << (unsigned) (unsigned char) c << std::dec;
        }
        std::cout << "\"" << std::endl;
        std::cout << "  top-1: id=" << top5[0].second
                  << " logit=" << top5[0].first
                  << " text=\"" << tokenizer.decodeToken(top5[0].second) << "\"" << std::endl;
    }
}

// ===========================================================================
// Diagnostic test: Compare batch prefill vs sequential prefill for the
// full Paris prompt (56 tokens). If they produce the same (wrong) result,
// the bug is not in batch-specific code paths.
// ===========================================================================
TEST_F(ModelTest, CompareBatchVsSequentialPrefill) {
    std::string question = "What is the capital of France?";
    std::string prompt = SharedTestEnv::model->formatChat(
            {{"system", "You are TinyCoder, an AI coding assistant. Be concise."},
             {"user", question}},
            true);

    auto &tokenizer = SharedTestEnv::model->tokenizer();
    const auto &config = SharedTestEnv::config;
    uint32_t vocabSize = config.vocabSize;

    // Tokenize prompt
    auto tokens = tokenizer.encode(prompt);
    ASSERT_GE(tokens.size(), 2) << "Prompt must have at least 2 tokens";
    std::cout << "Prompt: " << tokens.size() << " tokens" << std::endl;

    // ---- Method 1: Batch prefill (forward with all tokens) ----
    SharedTestEnv::model->clearKVCache();
    auto batchLogitsAll = SharedTestEnv::model->forward(tokens);
    ASSERT_FALSE(batchLogitsAll.empty());

    // Extract last token logits
    uint32_t lastIdx = static_cast<uint32_t>(tokens.size()) - 1;
    np::Array<float> batchLastLogits(np::Shape{vocabSize});
    for (uint32_t i = 0; i < vocabSize; ++i) {
        batchLastLogits.set(i, batchLogitsAll.get(lastIdx * vocabSize + i));
    }

    // Batch top-5
    float batchMax = -std::numeric_limits<float>::max();
    float batchMin = std::numeric_limits<float>::max();
    std::vector<std::pair<float, int32_t>> batchTop5;
    for (uint32_t i = 0; i < vocabSize; ++i) {
        float v = batchLastLogits.get(i);
        batchMax = std::max(batchMax, v);
        batchMin = std::min(batchMin, v);
        batchTop5.emplace_back(v, static_cast<int32_t>(i));
    }
    std::partial_sort(batchTop5.begin(), batchTop5.begin() + 5, batchTop5.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });
    std::cout << "\nBatch prefill logits: min=" << batchMin << " max=" << batchMax << std::endl;
    std::cout << "Batch top-5:" << std::endl;
    for (int r = 0; r < 5; ++r) {
        std::string dt = tokenizer.decodeToken(batchTop5[r].second);
        std::cout << "  [" << r << "] id=" << batchTop5[r].second
                  << " logit=" << batchTop5[r].first
                  << " text=\"" << dt << "\"" << std::endl;
    }

    // ---- Method 2: Sequential (token-by-token, building KV cache) ----
    SharedTestEnv::model->clearKVCache();

    // Fill cache with all but the last token
    for (size_t t = 0; t + 1 < tokens.size(); ++t) {
        SharedTestEnv::model->forward({tokens[t]});
    }

    // Now process the last token (KV cache has positions 0..n-2 filled)
    auto seqLogitsAll = SharedTestEnv::model->forward({tokens.back()});
    ASSERT_FALSE(seqLogitsAll.empty());

    // Extract logits (seqLen=1, so position 0)
    np::Array<float> seqLastLogits(np::Shape{vocabSize});
    for (uint32_t i = 0; i < vocabSize; ++i) {
        seqLastLogits.set(i, seqLogitsAll.get(i));
    }

    // Sequential top-5
    float seqMax = -std::numeric_limits<float>::max();
    float seqMin = std::numeric_limits<float>::max();
    std::vector<std::pair<float, int32_t>> seqTop5;
    for (uint32_t i = 0; i < vocabSize; ++i) {
        float v = seqLastLogits.get(i);
        seqMax = std::max(seqMax, v);
        seqMin = std::min(seqMin, v);
        seqTop5.emplace_back(v, static_cast<int32_t>(i));
    }
    std::partial_sort(seqTop5.begin(), seqTop5.begin() + 5, seqTop5.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });
    std::cout << "\nSequential prefill logits: min=" << seqMin << " max=" << seqMax << std::endl;
    std::cout << "Sequential top-5:" << std::endl;
    for (int r = 0; r < 5; ++r) {
        std::string dt = tokenizer.decodeToken(seqTop5[r].second);
        std::cout << "  [" << r << "] id=" << seqTop5[r].second
                  << " logit=" << seqTop5[r].first
                  << " text=\"" << dt << "\"" << std::endl;
    }

    // ---- Compare batch vs sequential ----
    float maxDiff = 0.0f;
    int diffCount = 0;
    float avgDiff = 0.0f;
    for (uint32_t i = 0; i < vocabSize; ++i) {
        float diff = std::abs(batchLastLogits.get(i) - seqLastLogits.get(i));
        maxDiff = std::max(maxDiff, diff);
        avgDiff += diff;
        if (diff > 1e-4f) diffCount++;
    }
    avgDiff /= static_cast<float>(vocabSize);

    std::cout << "\n=== Batch vs Sequential Comparison ===" << std::endl;
    std::cout << "  maxDiff=" << maxDiff << " avgDiff=" << avgDiff
              << " mismatches (>1e-4): " << diffCount << "/" << vocabSize << std::endl;

    // Top-1 token match check
    bool top1Match = (batchTop5[0].second == seqTop5[0].second);
    std::cout << "  Top-1 token match: " << (top1Match ? "YES" : "NO") << std::endl;
    if (!top1Match) {
        std::cout << "  Batch top-1: id=" << batchTop5[0].second
                  << " (" << tokenizer.decodeToken(batchTop5[0].second) << ")"
                  << " logit=" << batchTop5[0].first << std::endl;
        std::cout << "  Sequential top-1: id=" << seqTop5[0].second
                  << " (" << tokenizer.decodeToken(seqTop5[0].second) << ")"
                  << " logit=" << seqTop5[0].first << std::endl;
    }

    if (maxDiff < 1e-3f) {
        std::cout << "\nBatch and Sequential produce IDENTICAL results." << std::endl;
        // Reference value check is model-specific; only check for Qwen2 0.5B
        if (config.architecture == "qwen2" && config.hiddenSize == 1024 && config.numLayers == 24) {
            std::cout << "Comparing against reference (prefill):" << std::endl;
            std::cout << "  Reference top-1: id=785 (\"The\") logit=22.8516" << std::endl;
            std::cout << "  Our top-1: id=" << batchTop5[0].second
                      << " logit=" << batchTop5[0].first << std::endl;
            EXPECT_NEAR(batchTop5[0].first, 22.8516f, 5.0f)
                    << "Top-1 logit far from reference (22.8516).";
        } else {
            std::cout << "\n(Reference value check skipped for model: "
                      << config.modelName << " [" << config.architecture << "])" << std::endl;
        }
    } else {
        std::cout << "\n*** BATCH AND SEQUENTIAL DIFFER! ***" << std::endl;
        std::cout << "maxDiff=" << maxDiff << std::endl;
    }

    // Numeric gate: batch and sequential run identical deterministic kernels, so
    // a large diff signals a routing/math bug (e.g. a scalar-broadcast bsum
    // compensation counted 8x). top-1 must match exactly (Q3_K precision keeps
    // the coherence guard meaningful).
    EXPECT_LT(maxDiff, 5.0f)
            << "Batch prefill differs significantly from sequential! maxDiff=" << maxDiff;
    EXPECT_TRUE(top1Match)
            << "Batch prefill produces different top-1 token than sequential!";
}

// SharedTestEnv static members are defined in SharedTestEnv.cpp

// ---------------------------------------------------------------------------
// Custom main: reads TINYCODER_MODEL_PATH from environment, registers the
// shared test environment, and runs all tests.
// ---------------------------------------------------------------------------

// Main function moved to ModelTestMain.cpp
