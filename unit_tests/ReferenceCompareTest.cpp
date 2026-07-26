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

// Note: main() is in ModelTest.cpp - this file is compiled together with it.
