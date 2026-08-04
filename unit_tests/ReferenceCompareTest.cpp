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
 * TinyCoder Reference Comparison Tests
 *
 * These tests dump intermediate values (dequantized weights, matMulVec outputs,
 * hidden states, logits) for step-by-step comparison with reference
 * output. They are designed to help identify the root cause of the garbage
 * output bug by comparing TinyCoder's computation against a known-good
 * reference implementation at each step.
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "GGMLDequantize.hpp"
#include "GGUFLoader.hpp"
#include "IQ3XXS.hpp"
#include "Model.hpp"
#include "ModelConfig.hpp"
#include "SharedTestEnv.hpp"
#include "Tokenizer.hpp"

namespace fs = std::filesystem;
using namespace tinycoder;

// ---------------------------------------------------------------------------
// Test fixture: uses the model loaded once globally by SharedTestEnv
// ---------------------------------------------------------------------------

class ReferenceCompareTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_NE(SharedTestEnv::model, nullptr) << "Model not loaded -- skipping test";
        ASSERT_TRUE(SharedTestEnv::modelLoaded);
    }

    /// @brief Print a hex dump of raw block data for comparison.
    static void dumpBlockHex(const uint8_t *blockData, uint32_t typeSize,
                             const std::string &label) {
        std::cout << "  [" << label << "] raw hex (" << typeSize << " bytes): ";
        for (uint32_t i = 0; i < typeSize && i < 98; ++i) {
            std::cout << std::hex << std::setfill('0') << std::setw(2)
                      << static_cast<int>(blockData[i]) << " ";
        }
        std::cout << std::dec << std::endl;
    }

    /// @brief Print dequantized values for a block (first N values).
    static void printDequantizedBlock(const float *values, uint32_t n,
                                      const std::string &label) {
        std::cout << "  [" << label << "] dequantized (" << n << " values): ";
        for (uint32_t i = 0; i < n && i < 16; ++i) {
            std::cout << std::fixed << std::setprecision(6) << values[i] << " ";
        }
        if (n > 16)
            std::cout << "...";
        std::cout << std::endl;

        float minV = std::numeric_limits<float>::max();
        float maxV = -std::numeric_limits<float>::max();
        float sumV = 0.0f;
        for (uint32_t i = 0; i < n; ++i) {
            minV = std::min(minV, values[i]);
            maxV = std::max(maxV, values[i]);
            sumV += values[i];
        }
        std::cout << "  [" << label << "] stats: min=" << minV << " max=" << maxV
                  << " mean=" << (sumV / n) << std::endl;
    }

    /// @brief Print matMulVec output (first N values).
    static void printMatMulVecOutput(const float *result, uint32_t n,
                                     const std::string &label) {
        std::cout << "  [" << label << "] output (" << n << " values): ";
        for (uint32_t i = 0; i < n && i < 16; ++i) {
            std::cout << std::fixed << std::setprecision(6) << result[i] << " ";
        }
        if (n > 16)
            std::cout << "...";
        std::cout << std::endl;

        float minV = std::numeric_limits<float>::max();
        float maxV = -std::numeric_limits<float>::max();
        float sumV = 0.0f;
        int nanV = 0, infV = 0;
        for (uint32_t i = 0; i < n; ++i) {
            if (std::isnan(result[i]))
                nanV++;
            if (std::isinf(result[i]))
                infV++;
            minV = std::min(minV, result[i]);
            maxV = std::max(maxV, result[i]);
            sumV += result[i];
        }
        std::cout << "  [" << label << "] stats: min=" << minV << " max=" << maxV
                  << " mean=" << (sumV / n) << " nan=" << nanV << " inf=" << infV
                  << std::endl;
    }

    /// @brief Check stats of a float array and assert against reference values.
    static void checkStats(const std::string &label, const float *vals, uint32_t n,
                           float refMin, float refMax, float refMean, float tol = 1e-4f) {
        float minV = std::numeric_limits<float>::max();
        float maxV = -std::numeric_limits<float>::max();
        double sumV = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            minV = std::min(minV, vals[i]);
            maxV = std::max(maxV, vals[i]);
            sumV += static_cast<double>(vals[i]);
        }
        float meanV = static_cast<float>(sumV / n);
        EXPECT_NEAR(minV, refMin, tol) << label << " min mismatch";
        EXPECT_NEAR(maxV, refMax, tol) << label << " max mismatch";
        EXPECT_NEAR(meanV, refMean, tol) << label << " mean mismatch";
    }

    /// @brief Compute compression ratio for a quantized matrix.
    static float compressionRatioFor(const QuantizedMatrix &m) {
        uint64_t numElements = static_cast<uint64_t>(m.rows) * m.cols;
        uint64_t f32Bytes = numElements * sizeof(float);
        uint64_t compressedBytes = m.data.size();
        return compressedBytes > 0 ? static_cast<float>(f32Bytes) / compressedBytes : 0;
    }
};

// ---------------------------------------------------------------------------
// Reference values keyed by GGML quantization type
// ---------------------------------------------------------------------------

/// @brief All reference stats for a given quantization type.
struct QuantRefValues {
    // DumpLayer1FFNBlockData
    float gateBlock0_min, gateBlock0_max, gateBlock0_mean;
    float upBlock0_min, upBlock0_max, upBlock0_mean;
    float downBlock0_min, downBlock0_max, downBlock0_mean;
    float downBlockLast_min, downBlockLast_max, downBlockLast_mean;
    // MatMulVecUnitVector
    float gateUnit_min, gateUnit_max, gateUnit_mean;
    float upUnit_min, upUnit_max, upUnit_mean;
    float downUnit_min, downUnit_max, downUnit_mean;
    // MatMulVecAlternatingInput
    float gateAlt_min, gateAlt_max, gateAlt_mean;
    float upAlt_min, upAlt_max, upAlt_mean;
    float downAlt_min, downAlt_max, downAlt_mean;
};

/// @brief Map from GGML quantization type to reference values.
/// Populated for known types; unknown types will print a warning and skip assertions.
static const std::map<uint32_t, QuantRefValues> kQuantRefValues = {
        // -----------------------------------------------------------------------
        // IQ3_XXS (type 18) — original reference model
        // -----------------------------------------------------------------------
        {GGML_TYPE_IQ3_XXS,
         {
                 /* gateBlock0 */ -0.069945f,
                 0.080307f,
                 0.002016f,
                 /* upBlock0 */ -0.050664f,
                 0.050664f,
                 -0.000932f,
                 /* downBlock0 */ -0.051890f,
                 0.049634f,
                 -0.000408f,
                 /* downBlockLast */ -0.066674f,
                 0.053769f,
                 -0.001292f,
                 /* gateUnit */ -4.020172f,
                 3.549840f,
                 -0.090253f,
                 /* upUnit */ -3.935336f,
                 3.305185f,
                 -0.002194f,
                 /* downUnit */ -5.704785f,
                 5.692990f,
                 0.030858f,
                 /* gateAlt */ -3.990353f,
                 3.666003f,
                 -0.030212f,
                 /* upAlt */ -3.820618f,
                 3.210736f,
                 0.006702f,
                 /* downAlt */ -5.967931f,
                 5.870908f,
                 0.042684f,
         }},
        // -----------------------------------------------------------------------
        // Q2_K (type 10) — current test model
        // -----------------------------------------------------------------------
        {GGML_TYPE_Q2_K,
         {
                 /* gateBlock0 */ -0.048523f,
                 0.049374f,
                 0.001327f,
                 /* upBlock0 */ -0.061684f,
                 0.064083f,
                 -0.001074f,
                 /* downBlock0 */ -0.030228f,
                 0.017003f,
                 -0.000153f,
                 /* downBlockLast */ -0.056849f,
                 0.043584f,
                 -0.001451f,
                 /* gateUnit */ -3.185523f,
                 2.027240f,
                 -0.186748f,
                 /* upUnit */ -8.662849f,
                 8.035492f,
                 0.019161f,
                 /* downUnit */ -2.509111f,
                 2.804322f,
                 0.184117f,
                 /* gateAlt */ -2.555386f,
                 2.772243f,
                 0.061738f,
                 /* upAlt */ -9.303680f,
                 7.193542f,
                 0.022913f,
                 /* downAlt */ -3.176412f,
                 4.576875f,
                 0.006571f,
         }},
};

// ===========================================================================
// Test 1: Dump raw block data from layer 1 FFN weights
// ===========================================================================

TEST_F(ReferenceCompareTest, DumpLayer1FFNBlockData) {
    const auto &layers = SharedTestEnv::model->debugGetLayers();
    ASSERT_GE(layers.size(), 2) << "Model must have at least 2 layers";

    // Skip for MoE architectures that don't use standard FFN weights
    const auto &config = SharedTestEnv::config;
    if (config.architecture == "gemma4" || config.architecture == "qwen35moe") {
        std::cout << "\n=== Skipping FFN block dump for MoE architecture: "
                  << config.architecture << " ===" << std::endl;
        return;
    }

    const auto &gate = layers[1].ffnGate;
    const auto &up = layers[1].ffnUp;
    const auto &down = layers[1].ffnDown;

    uint32_t blockSize = ggmlBlockSize(gate.type);
    uint32_t typeSize = ggmlTypeSize(gate.type);

    std::cout << "\n=== Layer 1 FFN Weight Matrix Info ===" << std::endl;
    std::cout << "  ffnGate: rows=" << gate.rows << " cols=" << gate.cols
              << " type=" << gate.type << " blockSize=" << blockSize
              << " typeSize=" << typeSize << std::endl;
    std::cout << "  ffnUp:   rows=" << up.rows << " cols=" << up.cols
              << " type=" << up.type << std::endl;
    std::cout << "  ffnDown: rows=" << down.rows << " cols=" << down.cols
              << " type=" << down.type << std::endl;

    // Look up reference values for this quantization type
    auto refIt = kQuantRefValues.find(gate.type);
    bool hasRef = (refIt != kQuantRefValues.end());
    if (!hasRef) {
        std::cout << "\n  [WARNING] No reference values for quantization type "
                  << gate.type << " — printing stats only, skipping assertions."
                  << std::endl;
    }
    const QuantRefValues &ref = hasRef ? refIt->second : QuantRefValues{};

    uint32_t blocksPerRowDown = (down.cols + blockSize - 1) / blockSize;

    // Dump first block of each weight matrix
    std::cout << "\n=== Layer 1 FFN Gate: First Block (row 0, block 0) ===" << std::endl;
    const uint8_t *gateBlock0 = gate.data.data() + 0 * typeSize;
    dumpBlockHex(gateBlock0, typeSize, "gate.block0");

    float gateDeq0[256];
    GGMLDequantize::dequantizeBlock(gate.type, gateBlock0, gateDeq0, blockSize);
    printDequantizedBlock(gateDeq0, blockSize, "gate.block0");
    if (hasRef)
        checkStats("gate.block0", gateDeq0, blockSize, ref.gateBlock0_min, ref.gateBlock0_max, ref.gateBlock0_mean, 1e-4f);

    // Dump first block of up projection
    std::cout << "\n=== Layer 1 FFN Up: First Block (row 0, block 0) ===" << std::endl;
    const uint8_t *upBlock0 = up.data.data() + 0 * typeSize;
    dumpBlockHex(upBlock0, typeSize, "up.block0");

    float upDeq0[256];
    GGMLDequantize::dequantizeBlock(up.type, upBlock0, upDeq0, blockSize);
    printDequantizedBlock(upDeq0, blockSize, "up.block0");
    if (hasRef)
        checkStats("up.block0", upDeq0, blockSize, ref.upBlock0_min, ref.upBlock0_max, ref.upBlock0_mean, 1e-4f);

    // Dump first block of down projection
    std::cout << "\n=== Layer 1 FFN Down: First Block (row 0, block 0) ===" << std::endl;
    const uint8_t *downBlock0 = down.data.data() + 0 * typeSize;
    dumpBlockHex(downBlock0, typeSize, "down.block0");

    float downDeq0[256];
    GGMLDequantize::dequantizeBlock(down.type, downBlock0, downDeq0, blockSize);
    printDequantizedBlock(downDeq0, blockSize, "down.block0");
    if (hasRef)
        checkStats("down.block0", downDeq0, blockSize, ref.downBlock0_min, ref.downBlock0_max, ref.downBlock0_mean, 1e-4f);

    // Dump last block of down projection (to check edge cases)
    uint32_t lastBlockDown = blocksPerRowDown - 1;
    std::cout << "\n=== Layer 1 FFN Down: Last Block (row 0, block " << lastBlockDown
              << ") ===" << std::endl;
    const uint8_t *downBlockLast = down.data.data() + lastBlockDown * typeSize;
    dumpBlockHex(downBlockLast, typeSize, "down.blockLast");

    float downDeqLast[256];
    GGMLDequantize::dequantizeBlock(down.type, downBlockLast, downDeqLast, blockSize);
    uint32_t lastBlockElements = down.cols - lastBlockDown * blockSize;
    printDequantizedBlock(downDeqLast, lastBlockElements, "down.blockLast");
    if (hasRef)
        checkStats("down.blockLast", downDeqLast, lastBlockElements, ref.downBlockLast_min, ref.downBlockLast_max, ref.downBlockLast_mean, 1e-4f);
}

// ===========================================================================
// Test 2: matMulVec with unit vector input
// ===========================================================================

TEST_F(ReferenceCompareTest, MatMulVecUnitVector) {
    const auto &layers = SharedTestEnv::model->debugGetLayers();
    ASSERT_GE(layers.size(), 2);

    // Skip for MoE architectures that don't use standard FFN weights
    const auto &config = SharedTestEnv::config;
    if (config.architecture == "gemma4" || config.architecture == "qwen35moe") {
        std::cout << "\n=== Skipping matMulVec unit vector test for MoE architecture: "
                  << config.architecture << " ===" << std::endl;
        return;
    }

    const auto &gate = layers[1].ffnGate;
    const auto &up = layers[1].ffnUp;
    const auto &down = layers[1].ffnDown;

    // Look up reference values for this quantization type
    auto refIt = kQuantRefValues.find(gate.type);
    bool hasRef = (refIt != kQuantRefValues.end());
    if (!hasRef) {
        std::cout << "\n  [WARNING] No reference values for quantization type "
                  << gate.type << " — printing stats only, skipping assertions."
                  << std::endl;
    }
    const QuantRefValues &ref = hasRef ? refIt->second : QuantRefValues{};

    // Create a unit vector: x[i] = 1.0 for all i
    std::vector<float> unitVecGate(gate.cols, 1.0f);
    std::vector<float> unitVecUp(up.cols, 1.0f);
    std::vector<float> unitVecDown(down.cols, 1.0f);

    std::cout << "\n=== Layer 1 FFN Gate: matMulVec with unit vector ===" << std::endl;
    auto gateResult = gate.matMulVec(unitVecGate.data());
    printMatMulVecOutput(gateResult.data(), gate.rows, "gate.unit");
    if (hasRef)
        checkStats("gate.unit", gateResult.data(), gate.rows, ref.gateUnit_min, ref.gateUnit_max, ref.gateUnit_mean, 1e-2f);

    std::cout << "\n=== Layer 1 FFN Up: matMulVec with unit vector ===" << std::endl;
    auto upResult = up.matMulVec(unitVecUp.data());
    printMatMulVecOutput(upResult.data(), up.rows, "up.unit");
    if (hasRef)
        checkStats("up.unit", upResult.data(), up.rows, ref.upUnit_min, ref.upUnit_max, ref.upUnit_mean, 1e-2f);

    std::cout << "\n=== Layer 1 FFN Down: matMulVec with unit vector ===" << std::endl;
    auto downResult = down.matMulVec(unitVecDown.data());
    printMatMulVecOutput(downResult.data(), down.rows, "down.unit");
    if (hasRef)
        checkStats("down.unit", downResult.data(), down.rows, ref.downUnit_min, ref.downUnit_max, ref.downUnit_mean, 1e-2f);
}

// ===========================================================================
// Test 3: matMulVec with alternating +/-1 input
// ===========================================================================

TEST_F(ReferenceCompareTest, MatMulVecAlternatingInput) {
    const auto &layers = SharedTestEnv::model->debugGetLayers();
    ASSERT_GE(layers.size(), 2);

    // Skip for MoE architectures that don't use standard FFN weights
    const auto &config = SharedTestEnv::config;
    if (config.architecture == "gemma4" || config.architecture == "qwen35moe") {
        std::cout << "\n=== Skipping matMulVec alternating input test for MoE architecture: "
                  << config.architecture << " ===" << std::endl;
        return;
    }

    const auto &gate = layers[1].ffnGate;
    const auto &up = layers[1].ffnUp;
    const auto &down = layers[1].ffnDown;

    // Look up reference values for this quantization type
    auto refIt = kQuantRefValues.find(gate.type);
    bool hasRef = (refIt != kQuantRefValues.end());
    if (!hasRef) {
        std::cout << "\n  [WARNING] No reference values for quantization type "
                  << gate.type << " — printing stats only, skipping assertions."
                  << std::endl;
    }
    const QuantRefValues &ref = hasRef ? refIt->second : QuantRefValues{};

    // Create alternating +/-1 vector: x[i] = (i % 2 == 0) ? 1.0 : -1.0
    std::vector<float> altVecGate(gate.cols);
    std::vector<float> altVecUp(up.cols);
    std::vector<float> altVecDown(down.cols);
    for (uint32_t i = 0; i < gate.cols; ++i)
        altVecGate[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    for (uint32_t i = 0; i < up.cols; ++i)
        altVecUp[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    for (uint32_t i = 0; i < down.cols; ++i)
        altVecDown[i] = (i % 2 == 0) ? 1.0f : -1.0f;

    std::cout << "\n=== Layer 1 FFN Gate: matMulVec with alternating +/-1 ===" << std::endl;
    auto gateResult = gate.matMulVec(altVecGate.data());
    printMatMulVecOutput(gateResult.data(), gate.rows, "gate.alt");
    if (hasRef)
        checkStats("gate.alt", gateResult.data(), gate.rows, ref.gateAlt_min, ref.gateAlt_max, ref.gateAlt_mean, 1e-2f);

    std::cout << "\n=== Layer 1 FFN Up: matMulVec with alternating +/-1 ===" << std::endl;
    auto upResult = up.matMulVec(altVecUp.data());
    printMatMulVecOutput(upResult.data(), up.rows, "up.alt");
    if (hasRef)
        checkStats("up.alt", upResult.data(), up.rows, ref.upAlt_min, ref.upAlt_max, ref.upAlt_mean, 1e-2f);

    std::cout << "\n=== Layer 1 FFN Down: matMulVec with alternating +/-1 ===" << std::endl;
    auto downResult = down.matMulVec(altVecDown.data());
    printMatMulVecOutput(downResult.data(), down.rows, "down.alt");
    if (hasRef)
        checkStats("down.alt", downResult.data(), down.rows, ref.downAlt_min, ref.downAlt_max, ref.downAlt_mean, 1e-2f);
}

// ===========================================================================
// Test 4: Dump embedding stats for common tokens
// ===========================================================================

TEST_F(ReferenceCompareTest, EmbeddingStats) {
    std::cout << "\n=== Embedding Stats for Common Tokens ===" << std::endl;

    // Test a few common token IDs
    std::vector<int32_t> testTokens = {0, 1, 100, 1000, 10000};
    for (auto tokenId: testTokens) {
        if (tokenId >= static_cast<int32_t>(SharedTestEnv::model->debugGetEmbeddings().vocabSize))
            continue;

        auto emb = SharedTestEnv::model->debugGetEmbedding(tokenId);
        if (emb.empty())
            continue;

        float minV = *std::min_element(emb.begin(), emb.end());
        float maxV = *std::max_element(emb.begin(), emb.end());
        float sumV = 0.0f;
        int nanV = 0;
        for (auto v: emb) {
            if (std::isnan(v))
                nanV++;
            sumV += v;
        }
        float meanV = sumV / static_cast<float>(emb.size());
        std::string tokenText = SharedTestEnv::model->tokenizer().decodeToken(tokenId);
        std::string display;
        for (char c: tokenText) {
            if (c >= 32 && c < 127)
                display += c;
            else
                display += "\\x" + std::to_string(static_cast<unsigned char>(c));
        }
        std::cout << "  Token " << tokenId << " (\"" << display << "\"): min=" << minV
                  << " max=" << maxV << " mean=" << meanV
                  << " nan=" << nanV << " first5=";
        for (int i = 0; i < 5 && i < static_cast<int>(emb.size()); ++i)
            std::cout << std::fixed << std::setprecision(6) << emb[i] << " ";
        std::cout << std::endl;

        // Sanity assertions: embeddings must be finite and have reasonable range
        EXPECT_FALSE(std::isnan(minV)) << "Token " << tokenId << " embedding has NaN";
        EXPECT_FALSE(std::isinf(maxV)) << "Token " << tokenId << " embedding has Inf";
        EXPECT_GT(maxV, minV) << "Token " << tokenId << " embedding has no range";
        EXPECT_GT(maxV, 0.0f) << "Token " << tokenId << " embedding max should be positive";
        EXPECT_LT(minV, 0.0f) << "Token " << tokenId << " embedding min should be negative";
    }
}

// ===========================================================================
// Test 5: Dump all weight matrix types and sizes
// ===========================================================================

TEST_F(ReferenceCompareTest, WeightMatrixInfo) {
    const auto &layers = SharedTestEnv::model->debugGetLayers();
    ASSERT_GE(layers.size(), 1);

    std::cout << "\n=== Weight Matrix Info (Layer 0) ===" << std::endl;

    auto printMatrixInfo = [](const std::string &name, const QuantizedMatrix &m) {
        uint32_t blockSize = ggmlBlockSize(m.type);
        uint32_t typeSize = ggmlTypeSize(m.type);
        uint64_t numElements = static_cast<uint64_t>(m.rows) * m.cols;
        uint64_t f32Bytes = numElements * sizeof(float);
        uint64_t compressedBytes = m.data.size();
        float compressionRatio =
                compressedBytes > 0 ? static_cast<float>(f32Bytes) / compressedBytes : 0;

        std::cout << "  " << name << ": " << m.rows << "x" << m.cols << " type="
                  << m.type << " blockSize=" << blockSize << " typeSize=" << typeSize
                  << " compressed=" << (compressedBytes / 1024) << " KB"
                  << " f32=" << (f32Bytes / 1024) << " KB"
                  << " ratio=" << std::fixed << std::setprecision(2) << compressionRatio
                  << "x" << std::endl;
    };

    printMatrixInfo("attnQ", layers[0].attnQ);
    printMatrixInfo("attnK", layers[0].attnK);
    printMatrixInfo("attnV", layers[0].attnV);
    printMatrixInfo("attnO", layers[0].attnO);
    printMatrixInfo("ffnGate", layers[0].ffnGate);
    printMatrixInfo("ffnUp", layers[0].ffnUp);
    printMatrixInfo("ffnDown", layers[0].ffnDown);

    // Also print embedding info
    const auto &emb = SharedTestEnv::model->debugGetEmbeddings();
    uint32_t embBlockSize = ggmlBlockSize(emb.type);
    uint32_t embTypeSize = ggmlTypeSize(emb.type);
    uint64_t embElements = static_cast<uint64_t>(emb.vocabSize) * emb.hiddenSize;
    uint64_t embF32Bytes = embElements * sizeof(float);
    uint64_t embCompressedBytes = emb.data.size();
    float embRatio = embCompressedBytes > 0
                             ? static_cast<float>(embF32Bytes) / embCompressedBytes
                             : 0;
    std::cout << "  embeddings: " << emb.vocabSize << "x" << emb.hiddenSize
              << " type=" << emb.type << " blockSize=" << embBlockSize
              << " typeSize=" << embTypeSize
              << " compressed=" << (embCompressedBytes / (1024 * 1024)) << " MB"
              << " f32=" << (embF32Bytes / (1024 * 1024)) << " MB"
              << " ratio=" << std::fixed << std::setprecision(2) << embRatio
              << "x" << std::endl;

    // Sanity assertions: compression ratio must be > 1.0 for quantized types
    EXPECT_GT(compressionRatioFor(layers[0].attnQ), 1.0f) << "attnQ compression ratio should be > 1.0";
    EXPECT_GT(compressionRatioFor(layers[0].attnK), 1.0f) << "attnK compression ratio should be > 1.0";
    EXPECT_GT(compressionRatioFor(layers[0].attnV), 1.0f) << "attnV compression ratio should be > 1.0";
    EXPECT_GT(compressionRatioFor(layers[0].attnO), 1.0f) << "attnO compression ratio should be > 1.0";
    EXPECT_GT(compressionRatioFor(layers[0].ffnGate), 1.0f) << "ffnGate compression ratio should be > 1.0";
    EXPECT_GT(compressionRatioFor(layers[0].ffnUp), 1.0f) << "ffnUp compression ratio should be > 1.0";
    EXPECT_GT(compressionRatioFor(layers[0].ffnDown), 1.0f) << "ffnDown compression ratio should be > 1.0";
    EXPECT_GT(embRatio, 1.0f) << "Embedding compression ratio should be > 1.0";
}

// ===========================================================================
// Test 6: Dump embedding block data (generic, works with any quantization type)
// ===========================================================================

TEST_F(ReferenceCompareTest, DumpEmbeddingBlock) {
    const auto &emb = SharedTestEnv::model->debugGetEmbeddings();

    uint32_t blockSize = ggmlBlockSize(emb.type);
    uint32_t typeSize = ggmlTypeSize(emb.type);
    uint32_t numBlocks = (emb.hiddenSize + blockSize - 1) / blockSize;

    std::cout << "\n=== Embedding Block Dump ===" << std::endl;
    std::cout << "  Embeddings: " << emb.vocabSize << "x" << emb.hiddenSize
              << " type=" << emb.type << " blockSize=" << blockSize
              << " typeSize=" << typeSize << " numBlocks=" << numBlocks << std::endl;

    // Dump first block of token 0
    uint32_t tokenId = 0;
    uint64_t blockOffset = static_cast<uint64_t>(tokenId) * numBlocks + 0;
    const uint8_t *blockData = emb.data.data() + blockOffset * typeSize;

    std::cout << "\n  Token " << tokenId << " block 0 raw hex (" << typeSize << " bytes):" << std::endl;
    std::cout << "    ";
    for (uint32_t i = 0; i < typeSize && i < 64; ++i)
        std::cout << std::hex << std::setfill('0') << std::setw(2)
                  << static_cast<int>(blockData[i]) << " ";
    std::cout << std::dec << std::endl;

    // Dequantize using TinyCoder's implementation
    float deqTinyCoder[256];
    GGMLDequantize::dequantizeBlock(emb.type, blockData, deqTinyCoder, blockSize);

    float minTC = deqTinyCoder[0], maxTC = deqTinyCoder[0], sumTC = 0;
    int posTC = 0, negTC = 0, zeroTC = 0;
    for (uint32_t i = 0; i < blockSize; ++i) {
        if (deqTinyCoder[i] < minTC) minTC = deqTinyCoder[i];
        if (deqTinyCoder[i] > maxTC) maxTC = deqTinyCoder[i];
        sumTC += deqTinyCoder[i];
        if (deqTinyCoder[i] > 0) posTC++;
        else if (deqTinyCoder[i] < 0)
            negTC++;
        else
            zeroTC++;
    }
    std::cout << "\n  TinyCoder dequantized block 0:" << std::endl;
    std::cout << "    stats: min=" << minTC << " max=" << maxTC
              << " mean=" << (sumTC / blockSize) << std::endl;
    std::cout << "    positive=" << posTC << " negative=" << negTC
              << " zero=" << zeroTC << std::endl;
    std::cout << "    first 16 values: ";
    for (uint32_t i = 0; i < 16; ++i)
        std::cout << std::fixed << std::setprecision(6) << deqTinyCoder[i] << " ";
    std::cout << std::endl;

    // Now dequantize all blocks of token 0 and get full embedding stats
    std::vector<float> fullEmbedding(emb.hiddenSize);
    for (uint32_t b = 0; b < numBlocks; ++b) {
        blockOffset = static_cast<uint64_t>(tokenId) * numBlocks + b;
        blockData = emb.data.data() + blockOffset * typeSize;
        float blockOut[256];
        GGMLDequantize::dequantizeBlock(emb.type, blockData, blockOut, blockSize);
        uint32_t start = b * blockSize;
        uint32_t end = std::min(start + blockSize, emb.hiddenSize);
        for (uint32_t i = start; i < end; ++i)
            fullEmbedding[i] = blockOut[i - start];
    }

    float fmin = fullEmbedding[0], fmax = fullEmbedding[0], fsum = 0;
    int fpos = 0, fneg = 0, fzero = 0;
    for (uint32_t i = 0; i < emb.hiddenSize; ++i) {
        if (fullEmbedding[i] < fmin) fmin = fullEmbedding[i];
        if (fullEmbedding[i] > fmax) fmax = fullEmbedding[i];
        fsum += fullEmbedding[i];
        if (fullEmbedding[i] > 0) fpos++;
        else if (fullEmbedding[i] < 0)
            fneg++;
        else
            fzero++;
    }
    std::cout << "\n  Full embedding token " << tokenId << ":" << std::endl;
    std::cout << "    stats: min=" << fmin << " max=" << fmax
              << " mean=" << (fsum / emb.hiddenSize) << std::endl;
    std::cout << "    positive=" << fpos << " negative=" << fneg
              << " zero=" << fzero << std::endl;
    std::cout << "    first 8 values: ";
    for (uint32_t i = 0; i < 8; ++i)
        std::cout << std::fixed << std::setprecision(6) << fullEmbedding[i] << " ";
    std::cout << std::endl;

    // Sanity assertions
    EXPECT_FALSE(std::isnan(fmin)) << "Embedding has NaN";
    EXPECT_FALSE(std::isinf(fmax)) << "Embedding has Inf";
    EXPECT_GT(fmax, fmin) << "Embedding has no range";
    EXPECT_GT(fmax, 0.0f) << "Embedding max should be positive";
    EXPECT_LT(fmin, 0.0f) << "Embedding min should be negative";
}

// ---------------------------------------------------------------------------
// Q3_K batch kernel vs scalar reference (real weights)
// ---------------------------------------------------------------------------
// Decisive check for the Q3_K routing introduced for attnO/ffnDown: the AVX2
// register-tiled batch kernel must reproduce GGMLDequantize::matMulVecFused
// (scalar Q3_K reference) within the Q8_K x-quantization tolerance. If this
// passes, any margin in CompareBatchVsSequentialPrefill is attributable to
// Q3_K's coarser quantization (0.43 B/elem vs 1.14 for Q8_K) amplifying the
// intrinsic flash-attention batch-vs-sequential activation differences —
// NOT to a defect in the vector kernel.
TEST_F(ReferenceCompareTest, Q3K_BatchKernelVsScalar) {
    const auto &layers = SharedTestEnv::model->debugGetLayers();
    ASSERT_FALSE(layers.empty());

    struct Case {
        const QuantizedMatrix *m;
        std::string name;
    };
    std::vector<Case> cases;
    const QuantizedMatrix &attnO = layers[0].attnO;
    if (attnO.type == GGML_TYPE_Q3_K && attnO.cols % 256 == 0) {
        cases.push_back({&attnO, "attnO"});
    }
    const QuantizedMatrix &ffnDown = layers[0].ffnDown;
    if (ffnDown.type == GGML_TYPE_Q3_K && ffnDown.cols % 256 == 0) {
        cases.push_back({&ffnDown, "ffnDown"});
    }
    ASSERT_FALSE(cases.empty()) << "model has no Q3_K attnO/ffnDown to test";

    // ---- Phase 0: pinpoint the defect on ONE real block ----
    // Dissect the kernel formula on block 0 of row 0:
    //   scalarFloat : dotProductQ3_K against exact float x (LLaMA reference math)
    //   scalarQ8    : dotProductQ3_K against dequantized Q8_K x (isolates the
    //                 Q8_K x-quantization error from the formula error)
    //   mimic       : the kernel's exact formula (q2+4*hm, (sc-32) scales,
    //                 bsum compensation) evaluated in plain C++ on the same Q8_
    //                 x. If mimic matches scalarQ8 but the vector kernel does not,
    //                 the bug is in the AVX2 implementation; if mimic also differs,
    //                 the block layout/formula itself disagrees with the scalar.
    {
        const auto &m0 = *cases[0].m;
        const uint32_t cols0 = m0.cols;
        std::mt19937 rngB(777);
        std::normal_distribution<float> ndB(0.0f, 1.0f);
        std::vector<float> xb(cols0);
        for (uint32_t i = 0; i < cols0; ++i) {
            xb[i] = ndB(rngB);
        }

        const uint32_t blockBytes = 110;
        const uint32_t bpr = (cols0 + 255) / 256;
        const uint8_t *blk = m0.data.data() + static_cast<uint64_t>(0) * bpr * blockBytes + 0 * blockBytes;

        float dAll = GGMLDequantize::halfToFloat(*(const uint16_t *) (blk + 108));
        const uint8_t *hm = blk + 0;
        const uint8_t *q = blk + 32;
        const uint8_t *scRaw = blk + 96;

        // Unpack the 16 6-bit scales (identical transform to both references).
        uint32_t aux[4];
        std::memcpy(aux, scRaw, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
        aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
        aux[0] = (aux[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);
        aux[1] = (aux[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);
        const int8_t *sc = reinterpret_cast<const int8_t *>(aux);

        // 1) scalar vs exact float x
        float scalarFloat = GGMLDequantize::dotProductQ3_K(blk, xb.data());

        // 2) Q8_K quantize x and re-feed as dequantized floats
        Q8KBlock q8b;
        GGMLDequantize::quantizeQ8K(xb.data(), 256, &q8b);
        std::vector<float> xq(256);
        for (int i = 0; i < 256; ++i) {
            xq[i] = q8b.qs[i] * q8b.d;
        }
        float scalarQ8 = GGMLDequantize::dotProductQ3_K(blk, xq.data());

        // 3) mimic the kernel formula exactly (chunk/sub decomposition)
        double mainTerm = 0.0;
        int g = 0;
        for (int cn = 0; cn < 2; ++cn) {
            int shift = 0;
            for (int sub = 0; sub < 4; ++sub) {
                // Chunk cn reads the q quarter at offset cn*32 (NOT 8*(cn*4+sub)):
                // the kernel's qsrc = q + 32*(s>>2) equals q + 32*cn for the
                // 4 sub-blocks of each chunk.
                const uint8_t *qsrc = q + 32 * cn;
                for (int l = 0; l < 16; ++l) {
                    int q2 = (qsrc[l] >> shift) & 3;
                    int hb = (hm[l] >> (cn * 4 + sub)) & 1;
                    int wp = q2 + 4 * hb;// w' = q2 + 4*hm
                    int wg = g;
                    mainTerm += static_cast<double>(sc[wg] - 32) * wp * q8b.qs[cn * 128 + sub * 32 + l];
                }
                for (int l = 0; l < 16; ++l) {
                    int q2 = (qsrc[l + 16] >> shift) & 3;
                    int hb = (hm[l + 16] >> (cn * 4 + sub)) & 1;
                    int wp = q2 + 4 * hb;
                    int wg = g + 1;
                    mainTerm += static_cast<double>(sc[wg] - 32) * wp * q8b.qs[cn * 128 + sub * 32 + 16 + l];
                }
                g += 2;
                shift += 2;
            }
        }
        double bsumTerm = 0.0;
        for (int gg = 0; gg < 16; ++gg) {
            bsumTerm += static_cast<double>(sc[gg] - 32) * q8b.bsums[gg];
        }
        bsumTerm *= -4.0 * dAll * q8b.d;
        double mimic = dAll * q8b.d * mainTerm + bsumTerm;

        std::cout << "  [dissect " << cases[0].name << " block0] dAll=" << dAll
                  << " scalarFloat=" << scalarFloat << " scalarQ8=" << scalarQ8
                  << " mimic=" << mimic
                  << " |mimic-scalarQ8|=" << std::fabs(mimic - scalarQ8)
                  << " |scalarQ8-scalarFloat|=" << std::fabs(scalarQ8 - scalarFloat)
                  << std::endl;
        // If the mimic (kernel formula) is far from scalarQ8, the block layout
        // decoding itself is wrong (hm/scales/q mapping).
        // If mimic==scalarQ8, the formula is exact and any kernel error is in the
        // AVX2 code paths (shuffle lanes, maddubs operand order, etc.).
    }

    // ---- Phase 1: isolate per-block math on a single-block-per-row matrix ----
    // Build a fake matrix whose each row holds ONLY block 0 (256 cols) of the
    // real weights, then compare the AVX2 kernel row-by-row against the scalar
    // dotProductQ3_K. If rows diverge here, the per-block AVX2 math itself is
    // wrong on some blocks/rows; if all rows match, the defect is in the
    // multi-block indexing of the full kernel.
    {
        const auto &m0 = *cases[0].m;
        const uint32_t rows0 = m0.rows;
        const uint32_t bpr = m0.cols / 256;
        std::mt19937 rng1(222);
        std::normal_distribution<float> nd1(0.0f, 1.0f);
        std::vector<float> x1(256);
        for (uint32_t i = 0; i < 256; ++i) {
            x1[i] = nd1(rng1);
        }

        std::vector<uint8_t> oneBlk(static_cast<size_t>(rows0) * 110);
        for (uint32_t r = 0; r < rows0; ++r) {
            std::memcpy(oneBlk.data() + static_cast<size_t>(r) * 110,
                        m0.data.data() + static_cast<size_t>(r) * bpr * 110, 110);
        }
        std::vector<float> kOut1(rows0), sOut1(rows0);
        bool used1 = matMulVecBatchQ3K_SIMD(oneBlk.data(), x1.data(), 1, rows0, 256,
                                            kOut1.data());
        ASSERT_TRUE(used1);
        for (uint32_t r = 0; r < rows0; ++r) {
            sOut1[r] = GGMLDequantize::dotProductQ3_K(
                    oneBlk.data() + static_cast<size_t>(r) * 110, x1.data());
        }
        double ph1Max = 0.0;
        uint32_t ph1BadRow = 0;
        for (uint32_t r = 0; r < rows0; ++r) {
            double a = std::fabs(static_cast<double>(kOut1[r]) -
                                 static_cast<double>(sOut1[r]));
            if (a > ph1Max) {
                ph1Max = a;
                ph1BadRow = r;
            }
        }
        std::cout << "  [phase1 single-block] rows=" << rows0
                  << " maxAbs=" << ph1Max << " at row " << ph1BadRow
                  << "  first8 kernel=";
        for (uint32_t r = 0; r < std::min(8u, rows0); ++r) {
            std::cout << kOut1[r] << ",";
        }
        std::cout << " scalar=";
        for (uint32_t r = 0; r < std::min(8u, rows0); ++r) {
            std::cout << sOut1[r] << ",";
        }
        std::cout << std::endl;

        // Phase 1b: dissect row 0 of the single-block matrix with the SAME x1.
        // Recompute the proven-exact mimic and print the split pieces so we can
        // see whether the SIMD's main term, bsum term, or both diverge.
        {
            const uint8_t *blk0 = oneBlk.data();// row 0, block 0
            Q8KBlock q8b;
            GGMLDequantize::quantizeQ8K(x1.data(), 256, &q8b);
            float dAll0 = GGMLDequantize::halfToFloat(*(const uint16_t *) (blk0 + 108));
            const uint8_t *hm0 = blk0;
            const uint8_t *q0 = blk0 + 32;
            uint32_t aux0[4];
            std::memcpy(aux0, blk0 + 96, 12);
            uint32_t tmp0 = aux0[2];
            aux0[2] = ((aux0[0] >> 4) & 0x0f0f0f0fu) | (((tmp0 >> 4) & 0x03030303u) << 4);
            aux0[3] = ((aux0[1] >> 4) & 0x0f0f0f0fu) | (((tmp0 >> 6) & 0x03030303u) << 4);
            aux0[0] = (aux0[0] & 0x0f0f0f0fu) | (((tmp0 >> 0) & 0x03030303u) << 4);
            aux0[1] = (aux0[1] & 0x0f0f0f0fu) | (((tmp0 >> 2) & 0x03030303u) << 4);
            const int8_t *sc0 = reinterpret_cast<const int8_t *>(aux0);
            double mainTerm0 = 0.0;
            int g0 = 0;
            for (int cn = 0; cn < 2; ++cn) {
                int shift0 = 0;
                for (int sub = 0; sub < 4; ++sub) {
                    const uint8_t *qsrc0 = q0 + 32 * cn;
                    for (int l = 0; l < 16; ++l) {
                        int q2 = (qsrc0[l] >> shift0) & 3;
                        int hb = (hm0[l] >> (cn * 4 + sub)) & 1;
                        mainTerm0 += static_cast<double>(sc0[g0] - 32) *
                                     (q2 + 4 * hb) *
                                     q8b.qs[cn * 128 + sub * 32 + l];
                    }
                    for (int l = 0; l < 16; ++l) {
                        int q2 = (qsrc0[l + 16] >> shift0) & 3;
                        int hb = (hm0[l + 16] >> (cn * 4 + sub)) & 1;
                        mainTerm0 += static_cast<double>(sc0[g0 + 1] - 32) *
                                     (q2 + 4 * hb) *
                                     q8b.qs[cn * 128 + sub * 32 + 16 + l];
                    }
                    g0 += 2;
                    shift0 += 2;
                }
            }
            double bsumTerm0 = 0.0;
            for (int gg = 0; gg < 16; ++gg) {
                bsumTerm0 += static_cast<double>(sc0[gg] - 32) * q8b.bsums[gg];
            }
            bsumTerm0 *= -4.0 * dAll0 * q8b.d;
            double mimic0 = dAll0 * q8b.d * mainTerm0 + bsumTerm0;

            std::cout << "  [phase1b row0 block0] dAll=" << dAll0 << " d=" << q8b.d
                      << " kernel=" << kOut1[0] << " scalar=" << sOut1[0]
                      << " mimic=" << mimic0 << " main=" << dAll0 * q8b.d * mainTerm0
                      << " bsum=" << bsumTerm0 << std::endl;
            std::cout << "    scales(sc-32)=";
            for (int gg = 0; gg < 16; ++gg) {
                std::cout << (int) (sc0[gg] - 32) << ",";
            }
            std::cout << " bsums=";
            for (int gg = 0; gg < 16; ++gg) {
                std::cout << q8b.bsums[gg] << ",";
            }
            std::cout << std::endl;
        }
    }

    for (const auto &c: cases) {
        const uint32_t rows = c.m->rows;
        const uint32_t cols = c.m->cols;

        // Deterministic pseudo-activation input (unit-normal).
        std::vector<float> x(cols);
        std::mt19937 rng(12345);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (uint32_t i = 0; i < cols; ++i) {
            x[i] = nd(rng);
        }

        std::vector<float> kernelOut(rows);
        bool used = matMulVecBatchQ3K_SIMD(
                c.m->data.data(), x.data(), 1, rows, cols, kernelOut.data());
        ASSERT_TRUE(used) << "Q3_K vector kernel not dispatched on this host";

        // The kernel computes against a Q8_K-quantized x (per 256-block). The
        // scalar reference must see the SAME Q8_K-dequantized x; comparing
        // against the exact float x would inject the Q8_K quantization error
        // (~1% per element) into the tolerance gate.
        const uint32_t blockSize = 256;
        const uint32_t bpr = cols / blockSize;
        std::vector<float> xq(cols);
        std::vector<Q8KBlock> q8ref(bpr);
        for (uint32_t b = 0; b < bpr; ++b) {
            GGMLDequantize::quantizeQ8K(x.data() + b * blockSize, blockSize,
                                        &q8ref[b]);
            for (uint32_t i = 0; i < blockSize; ++i) {
                xq[b * blockSize + i] = q8ref[b].qs[i] * q8ref[b].d;
            }
        }
        std::vector<float> scalarOut(rows);
        GGMLDequantize::matMulVecFused(c.m->type, c.m->data.data(), xq.data(),
                                       rows, cols, scalarOut.data());

        double maxAbs = 0.0, maxRel = 0.0;
        float refMax = 0.0f;
        for (uint32_t j = 0; j < rows; ++j) {
            refMax = std::max(refMax, std::fabs(scalarOut[j]));
        }
        for (uint32_t j = 0; j < rows; ++j) {
            double a = std::fabs(static_cast<double>(kernelOut[j]) -
                                 static_cast<double>(scalarOut[j]));
            maxAbs = std::max(maxAbs, a);
            maxRel = std::max(maxRel,
                              a / std::max(1e-6, static_cast<double>(std::fabs(
                                                         scalarOut[j]))));
        }

        std::cout << "  Q3K " << c.name << " rows=" << rows << " cols=" << cols
                  << " refMax=" << refMax << " maxAbs=" << maxAbs
                  << " maxRel=" << maxRel << std::endl;

        // Q8_K x-quantization + fp32 accumulation error is ~0.4% per element;
        // allow 1% relative and a small absolute floor. A real kernel defect
        // (wrong scale pairing, bit mis-extraction, missing bsum term) blows
        // far past this.
        const float absTol = 0.01f * std::max(1.0f, refMax);
        EXPECT_LT(maxAbs, static_cast<double>(absTol))
                << "Q3_K batch kernel deviates from scalar reference (" << c.name << ")";
    }
}

TEST_F(ReferenceCompareTest, Q6K_BatchKernelVsScalar) {
    // Q6_K is how the separate LM head is stored in the model file. Exercise
    // the AVX2 batch kernel against the scalar dotProductQ6_K (through
    // matMulVecFused), feeding the reference the SAME Q8_K-dequantized x.
    const QuantizedMatrix &lmHead = SharedTestEnv::model->debugGetLMHead();
    if (lmHead.type != GGML_TYPE_Q6_K || lmHead.cols % 256 != 0) {
        GTEST_SKIP() << "model has no Q6_K LM head to test";
    }
    const uint32_t blockBytes = 210;

    // ---- Phase 0: dissect ONE real block ----
    // scalarFloat : dotProductQ6_K against exact float x
    // scalarQ8    : dotProductQ6_K against dequantized Q8_K x
    // mimic       : the kernel's exact formula (raw 6-bit w', raw signed sc,
    //               -32 bsum compensation) on the same Q8_K x
    {
        std::mt19937 rngB(777);
        std::normal_distribution<float> ndB(0.0f, 1.0f);
        std::vector<float> xb(256);
        for (int i = 0; i < 256; ++i) {
            xb[i] = ndB(rngB);
        }
        const uint32_t bpr = lmHead.cols / 256;
        const uint8_t *blk = lmHead.data.data() + static_cast<uint64_t>(0) * bpr * blockBytes;

        float dAll = GGMLDequantize::halfToFloat(*(const uint16_t *) (blk + 208));
        const int8_t *sc = reinterpret_cast<const int8_t *>(blk + 192);

        float scalarFloat = GGMLDequantize::dotProductQ6_K(blk, xb.data());

        Q8KBlock q8b;
        GGMLDequantize::quantizeQ8K(xb.data(), 256, &q8b);
        std::vector<float> xq(256);
        for (int i = 0; i < 256; ++i) {
            xq[i] = q8b.qs[i] * q8b.d;
        }
        float scalarQ8 = GGMLDequantize::dotProductQ6_K(blk, xq.data());

        // Mimic the kernel formula: w' = low_nibble | (high2 << 4), value =
        // d*sc*(w'-32); the -32 folds through the bsum term.
        double mainTerm = 0.0;
        for (int n = 0; n < 256; n += 128) {
            const uint8_t *ql = blk + (n / 2);
            const uint8_t *qh = blk + 128 + (n / 4);
            const int8_t *scb = sc + (n / 128) * 8;
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int w1 = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
                int w2 = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
                int w3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
                int w4 = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
                mainTerm += static_cast<double>(scb[is + 0]) * w1 * q8b.qs[n + l];
                mainTerm += static_cast<double>(scb[is + 2]) * w2 * q8b.qs[n + l + 32];
                mainTerm += static_cast<double>(scb[is + 4]) * w3 * q8b.qs[n + l + 64];
                mainTerm += static_cast<double>(scb[is + 6]) * w4 * q8b.qs[n + l + 96];
            }
        }
        double bsumTerm = 0.0;
        for (int g = 0; g < 16; ++g) {
            bsumTerm += static_cast<double>(sc[g]) * q8b.bsums[g];
        }
        bsumTerm *= -32.0 * dAll * q8b.d;
        double mimic = dAll * q8b.d * mainTerm + bsumTerm;

        std::cout << "  [Q6K dissect block0] d=" << dAll
                  << " scalarFloat=" << scalarFloat << " scalarQ8=" << scalarQ8
                  << " mimic=" << mimic
                  << " |mimic-scalarQ8|=" << std::fabs(mimic - scalarQ8)
                  << " |scalarQ8-scalarFloat|=" << std::fabs(scalarQ8 - scalarFloat)
                  << std::endl;
    }

    // ---- Phase 1: single-block-per-row matrix, kernel vs scalar ----
    {
        const uint32_t rows0 = lmHead.rows;
        std::mt19937 rng1(222);
        std::normal_distribution<float> nd1(0.0f, 1.0f);
        std::vector<float> x1(256);
        for (uint32_t i = 0; i < 256; ++i) {
            x1[i] = nd1(rng1);
        }
        const uint32_t bpr = lmHead.cols / 256;
        std::vector<uint8_t> oneBlk(static_cast<size_t>(rows0) * blockBytes);
        for (uint32_t r = 0; r < rows0; ++r) {
            std::memcpy(oneBlk.data() + static_cast<size_t>(r) * blockBytes,
                        lmHead.data.data() + static_cast<size_t>(r) * bpr * blockBytes,
                        blockBytes);
        }
        // Limit Phase 1 to the first 4096 rows for runtime sanity while still
        // spanning many distinct blocks.
        uint32_t testRows = std::min(rows0, 4096u);
        std::vector<float> kOut1(testRows), sOut1(testRows);
        bool used1 = matMulVecBatchQ6K_SIMD(oneBlk.data(), x1.data(), 1, testRows, 256,
                                            kOut1.data());
        ASSERT_TRUE(used1) << "Q6_K vector kernel not dispatched on this host";
        for (uint32_t r = 0; r < testRows; ++r) {
            sOut1[r] = GGMLDequantize::dotProductQ6_K(
                    oneBlk.data() + static_cast<size_t>(r) * blockBytes, x1.data());
        }
        double ph1Max = 0.0;
        for (uint32_t r = 0; r < testRows; ++r) {
            ph1Max = std::max(ph1Max,
                              std::fabs(static_cast<double>(kOut1[r]) -
                                        static_cast<double>(sOut1[r])));
        }
        std::cout << "  [Q6K phase1 single-block] rows=" << testRows
                  << " maxAbs=" << ph1Max << "  first4 kernel=";
        for (uint32_t r = 0; r < std::min(4u, testRows); ++r) {
            std::cout << kOut1[r] << ",";
        }
        std::cout << " scalar=";
        for (uint32_t r = 0; r < std::min(4u, testRows); ++r) {
            std::cout << sOut1[r] << ",";
        }
        std::cout << std::endl;
    }

    // ---- Full matrix, kernel vs scalar reference on the same Q8_K x ----
    const uint32_t rows = lmHead.rows;
    const uint32_t cols = lmHead.cols;

    std::vector<float> x(cols);
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (uint32_t i = 0; i < cols; ++i) {
        x[i] = nd(rng);
    }

    std::vector<float> kernelOut(rows);
    auto t0 = std::chrono::steady_clock::now();
    bool used = matMulVecBatchQ6K_SIMD(lmHead.data.data(), x.data(), 1, rows, cols,
                                       kernelOut.data());
    auto t1 = std::chrono::steady_clock::now();
    ASSERT_TRUE(used) << "Q6_K vector kernel not dispatched on this host";
    std::cout << "  [Q6K timing] seqLen=1 rows=" << rows << " cols=" << cols
              << " kernel="
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms" << std::endl;

    // Feed the scalar reference the Q8_K-dequantized x (same as the kernel).
    const uint32_t bpr = cols / 256;
    std::vector<float> xq(cols);
    std::vector<Q8KBlock> q8ref(bpr);
    for (uint32_t b = 0; b < bpr; ++b) {
        GGMLDequantize::quantizeQ8K(x.data() + b * 256, 256, &q8ref[b]);
        for (uint32_t i = 0; i < 256; ++i) {
            xq[b * 256 + i] = q8ref[b].qs[i] * q8ref[b].d;
        }
    }
    std::vector<float> scalarOut(rows);
    GGMLDequantize::matMulVecFused(lmHead.type, lmHead.data.data(), xq.data(),
                                   rows, cols, scalarOut.data());

    double maxAbs = 0.0, maxRel = 0.0;
    float refMax = 0.0f;
    for (uint32_t j = 0; j < rows; ++j) {
        refMax = std::max(refMax, std::fabs(scalarOut[j]));
    }
    for (uint32_t j = 0; j < rows; ++j) {
        double a = std::fabs(static_cast<double>(kernelOut[j]) -
                             static_cast<double>(scalarOut[j]));
        maxAbs = std::max(maxAbs, a);
        maxRel = std::max(maxRel,
                          a / std::max(1e-6, static_cast<double>(std::fabs(
                                                     scalarOut[j]))));
    }
    std::cout << "  Q6K lmHead rows=" << rows << " cols=" << cols
              << " refMax=" << refMax << " maxAbs=" << maxAbs
              << " maxRel=" << maxRel << std::endl;

    // Q8_K x-quantization + fp32 accumulation error is ~0.4% per element;
    // allow 1% relative and a small absolute floor (same gate as the Q3_K test).
    const float absTol = 0.01f * std::max(1.0f, refMax);
    EXPECT_LT(maxAbs, static_cast<double>(absTol))
            << "Q6_K batch kernel deviates from scalar reference (lmHead)";
}

/// @brief Compare the fused compact Q2_K Q+K generation kernel against the
/// scalar dotProductQ2_K reference for the real attnQ / attnK matrices.
///
/// The kernel is used for single-token generation (Task 13); the Q (1536 rows)
/// and K (256 rows) projections share compact Q2_K 84-byte blocks, so the fused
/// kernel reads each once and reuses the single Q8_K quantization of x across
/// both matrices. Feeding the scalar the same Q8_K-dequantized x isolates the
/// kernel's block math from the x-quantization error.
TEST_F(ReferenceCompareTest, Q2K_FusedQK_KernelVsScalar) {
    const auto &layers = SharedTestEnv::model->debugGetLayers();
    ASSERT_FALSE(layers.empty());

    const QuantizedMatrix &attnQ = layers[0].attnQ;
    const QuantizedMatrix &attnK = layers[0].attnK;
    if (attnQ.type != GGML_TYPE_Q2_K || attnK.type != GGML_TYPE_Q2_K ||
        attnQ.cols != attnK.cols || attnQ.cols % 256 != 0) {
        GTEST_SKIP() << "layer 0 is not compact Q2_K Q/K with cols%256==0";
    }

    const uint32_t rowsQ = attnQ.rows;
    const uint32_t rowsK = attnK.rows;
    const uint32_t cols = attnQ.cols;

    std::vector<float> x(cols);
    std::mt19937 rng(4242);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (uint32_t i = 0; i < cols; ++i) {
        x[i] = nd(rng);
    }

    std::vector<float> qKern(rowsQ), kKern(rowsK);
    matMulVecFusedQKQ2_K_Compact_Q8_SIMD(
            attnQ.data.data(), attnK.data.data(), x.data(),
            rowsQ, rowsK, cols, qKern.data(), kKern.data());

    // Scalar reference on the SAME Q8_K-dequantized x (same as the kernel).
    const uint32_t bpr = cols / 256;
    std::vector<float> xq(cols);
    std::vector<Q8KBlock> q8ref(bpr);
    for (uint32_t b = 0; b < bpr; ++b) {
        GGMLDequantize::quantizeQ8K(x.data() + b * 256, 256, &q8ref[b]);
        for (uint32_t i = 0; i < 256; ++i) {
            xq[b * 256 + i] = q8ref[b].qs[i] * q8ref[b].d;
        }
    }
    std::vector<float> qScalar(rowsQ), kScalar(rowsK);
    GGMLDequantize::matMulVecFused(GGML_TYPE_Q2_K, attnQ.data.data(), xq.data(),
                                   rowsQ, cols, qScalar.data());
    GGMLDequantize::matMulVecFused(GGML_TYPE_Q2_K, attnK.data.data(), xq.data(),
                                   rowsK, cols, kScalar.data());

    auto report = [&](const std::string &label, const std::vector<float> &kern,
                      const std::vector<float> &ref, uint32_t n) {
        double maxAbs = 0.0, refMax = 0.0;
        for (uint32_t j = 0; j < n; ++j) {
            refMax = std::max(refMax, std::fabs(static_cast<double>(ref[j])));
        }
        for (uint32_t j = 0; j < n; ++j) {
            maxAbs = std::max(maxAbs,
                              std::fabs(static_cast<double>(kern[j]) -
                                        static_cast<double>(ref[j])));
        }
        std::cout << "  Q2K fusedQK " << label << " rows=" << n << " cols=" << cols
                  << " refMax=" << refMax << " maxAbs=" << maxAbs << std::endl;
        return maxAbs;
    };

    double qAbs = report("Q", qKern, qScalar, rowsQ);
    double kAbs = report("K", kKern, kScalar, rowsK);

    double qRefMax = 0.0, kRefMax = 0.0;
    for (uint32_t j = 0; j < rowsQ; ++j) qRefMax = std::max(qRefMax, std::fabs((double) qScalar[j]));
    for (uint32_t j = 0; j < rowsK; ++j) kRefMax = std::max(kRefMax, std::fabs((double) kScalar[j]));
    EXPECT_LT(qAbs, 0.01 * std::max(1.0, qRefMax)) << "Q2K fused Q+Q kernel (Q rows) deviates";
    EXPECT_LT(kAbs, 0.01 * std::max(1.0, kRefMax)) << "Q2K fused Q+K kernel (K rows) deviates";
}

// Note: main() is in ModelTest.cpp - this file is compiled together with it.
