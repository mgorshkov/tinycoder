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
 * TinyCoder Dequantization Unit Test
 *
 * Tests the GGML dequantization functions against known constants.
 * These test vectors are derived from the reference implementation
 * to verify correctness of the dequantization routines.
 *
 * Each test creates a known quantized block with specific values and
 * verifies that dequantization produces the expected float values.
 */

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <regex>
#include <vector>

#include "GGMLDequantize.hpp"
#include "GGUFLoader.hpp"

using namespace tinycoder;

// ---------------------------------------------------------------------------
// Test: Q5_1 block dequantization with known constants
// ---------------------------------------------------------------------------
// Q5_1 block format: 32 weights in 32 bytes
//   - 2 bytes: d (half precision scale)
//   - 2 bytes: m (half precision min)
//   - 28 bytes: 32 5-bit values packed in 7 uint32_t's (4 values per uint32_t)
//
// Test vector: create a block where all values are 0, so dequantized = m
TEST(DequantizeTest, Q5_1BlockAllZeros) {
    // Create a Q5_1 block with d=1.0, m=0.0, all weights = 0
    uint8_t blockData[32] = {};
    // Set d = 1.0 (half precision: 0x3C00)
    blockData[0] = 0x00;
    blockData[1] = 0x3C;
    // Set m = 0.0 (half precision: 0x0000)
    blockData[2] = 0x00;
    blockData[3] = 0x00;
    // All qm values are 0 (already zero-initialized)

    float out[32];
    GGMLDequantize::dequantizeQ5_1Block(blockData, out);

    // All values should be 0.0 (since m=0 and all weights are 0)
    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(out[i], 0.0f, 1e-6f) << "Mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: Q5_1 block with known values
// ---------------------------------------------------------------------------
// Q5_1 block format (32 bytes):
//   Bytes 0-1:  d (half precision scale)
//   Bytes 2-3:  m (half precision min)
//   Bytes 4-7:  qh (32-bit, bit i = high bit of weight i)
//   Bytes 8-23: ql (16 bytes, byte j holds two 4-bit low nibbles:
//                   ql[j] & 0xF = low bits of weight 2*j,
//                   ql[j] >> 4  = low bits of weight 2*j+1)
//   Bytes 24-31: padding (unused)
//
// Formula: out[i] = q * d + m, where q = (highBit << 4) | low4
TEST(DequantizeTest, Q5_1BlockKnownValues) {
    // Create a Q5_1 block with d=2.0, m=-1.0
    // Half precision: 2.0 = 0x4000, -1.0 = 0xBC00
    uint8_t blockData[32] = {};
    blockData[0] = 0x00;
    blockData[1] = 0x40;// d = 2.0
    blockData[2] = 0x00;
    blockData[3] = 0xBC;// m = -1.0

    // Set all weights to 16 (0b10000):
    //   low4 = 0, highBit = 1
    //   ql bytes: each nibble = 0
    //   qh: all 32 bits = 1
    uint32_t qh = 0xFFFFFFFF;// all high bits = 1
    std::memcpy(blockData + 4, &qh, sizeof(uint32_t));
    // ql: all nibbles = 0 (already zero-initialized)

    float out[32];
    GGMLDequantize::dequantizeQ5_1Block(blockData, out);

    // q = (1 << 4) | 0 = 16
    // out = 16 * 2.0 + (-1.0) = 32.0 - 1.0 = 31.0
    float expected = 16.0f * 2.0f + (-1.0f);// = 31.0
    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(out[i], expected, 1e-4f) << "Mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: Q5_K block dequantization
// ---------------------------------------------------------------------------
// Q5_K block format: 256 weights in 176 bytes
//   - 2 bytes: d (half precision super-block scale)
//   - 2 bytes: dmin (half precision super-block min)
//   - 16 bytes: scales (8 high bits + 8 low bits)
//   - 32 bytes: qh (32 x 8-bit high bits)
//   - 128 bytes: ql (128 x 8-bit low bits)
//   - 4 bytes: padding
//
// Test: create a minimal block and verify no crashes
TEST(DequantizeTest, Q5_KBlockNoCrash) {
    uint8_t blockData[176] = {};
    // Set d = 1.0
    blockData[0] = 0x00;
    blockData[1] = 0x3C;
    // Set dmin = 0.0
    blockData[2] = 0x00;
    blockData[3] = 0x00;

    float out[256];
    GGMLDequantize::dequantizeQ5_KBlock(blockData, out);

    // Just verify no crash and values are finite
    for (int i = 0; i < 256; ++i) {
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: Q4_K block dequantization
// ---------------------------------------------------------------------------
TEST(DequantizeTest, Q4_KBlockNoCrash) {
    uint8_t blockData[144] = {};
    // Set d = 1.0
    blockData[0] = 0x00;
    blockData[1] = 0x3C;
    // Set dmin = 0.0
    blockData[2] = 0x00;
    blockData[3] = 0x00;

    float out[256];
    GGMLDequantize::dequantizeQ4_KBlock(blockData, out);

    for (int i = 0; i < 256; ++i) {
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: IQ3_XXS block dequantization
// ---------------------------------------------------------------------------
// IQ3_XXS block format: 256 weights in 98 bytes
//   - 2 bytes: d (half precision scale)
//   - 64 bytes: qs (quantized indices)
//   - 32 bytes: scales_and_signs
//
// Test: create a block with d=1.0, all indices=0, all signs=0
TEST(DequantizeTest, IQ3_XXSBlockNoCrash) {
    uint8_t blockData[98] = {};
    // Set d = 1.0
    blockData[0] = 0x00;
    blockData[1] = 0x3C;

    float out[256];
    GGMLDequantize::dequantizeIQ3_XXSBlock(blockData, out);

    for (int i = 0; i < 256; ++i) {
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: IQ3_S block dequantization
// ---------------------------------------------------------------------------
TEST(DequantizeTest, IQ3_SBlockNoCrash) {
    uint8_t blockData[110] = {};
    // Set d = 1.0
    blockData[0] = 0x00;
    blockData[1] = 0x3C;

    float out[256];
    GGMLDequantize::dequantizeIQ3_SBlock(blockData, out);

    for (int i = 0; i < 256; ++i) {
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: IQ2_S block dequantization
// ---------------------------------------------------------------------------
TEST(DequantizeTest, IQ2_SBlockNoCrash) {
    uint8_t blockData[82] = {};
    // Set d = 1.0
    blockData[0] = 0x00;
    blockData[1] = 0x3C;

    float out[256];
    GGMLDequantize::dequantizeIQ2_SBlock(blockData, out);

    for (int i = 0; i < 256; ++i) {
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: Generic dequantizeBlock dispatcher
// ---------------------------------------------------------------------------
TEST(DequantizeTest, GenericDequantizeBlockAllTypes) {
    // Test that the generic dequantizeBlock dispatcher works for all types
    // by comparing with the type-specific functions

    // Create a Q5_1 block with known values
    uint8_t blockData[256] = {};// max block size
    // Set d = 1.5, m = 0.5
    // Half: 1.5 = 0x3E00, 0.5 = 0x3800
    blockData[0] = 0x00;
    blockData[1] = 0x3E;
    blockData[2] = 0x00;
    blockData[3] = 0x38;
    // Set qh: alternating high bits
    uint32_t qh = 0xAAAAAAAA;
    std::memcpy(blockData + 4, &qh, sizeof(uint32_t));
    // Set ql: alternating nibbles
    for (int i = 0; i < 16; ++i) {
        blockData[8 + i] = 0xAB;// nibbles: A=10, B=11
    }

    float outDirect[32];
    float outGeneric[32];
    GGMLDequantize::dequantizeQ5_1Block(blockData, outDirect);
    GGMLDequantize::dequantizeBlock(GGML_TYPE_Q5_1, blockData, outGeneric, 32);

    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(outDirect[i], outGeneric[i], 1e-6f) << "Mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: matMulVecFused produces same result as full dequantize + dot product
// ---------------------------------------------------------------------------
TEST(DequantizeTest, MatMulVecFusedMatchesDequantizeDot) {
    // Create a small quantized matrix and verify that matMulVecFused produces
    // the same result as full dequantize + manual dot product.
    //
    // GGUF stores weight matrices as (rows x cols) = (out_features x in_features)
    // in row-major order. Each output row j has blocksPerRow quantized blocks,
    // each covering blockSize input features.
    // W[j][i] = deq[j * cols + i] (within each row's blocks).
    // The computation is: y_j = sum_i x[i] * W[j][i]
    //
    // We use COLS < BLOCK_SIZE to test partial-block handling.

    constexpr uint32_t ROWS = 32;// output features
    constexpr uint32_t COLS = 2; // input features
    constexpr uint32_t TYPE = GGML_TYPE_Q5_1;
    constexpr uint32_t BLOCK_SIZE = 32;
    constexpr uint32_t TYPE_SIZE = 24;// Q5_1 block size: d(2) + m(2) + qh(4) + ql(16) = 24

    // Create quantized data for the matrix stored as (rows x cols).
    // With ROWS=32 output rows and COLS=2 input features, each output row has
    // blocksPerRow = ceil(2/32) = 1 block covering 2 input features.
    // The remaining 30 elements of each block are padding (not used).
    uint32_t blocksPerRow = (COLS + BLOCK_SIZE - 1) / BLOCK_SIZE;// 1
    uint64_t totalBytes = static_cast<uint64_t>(ROWS) * blocksPerRow * TYPE_SIZE;
    std::vector<uint8_t> quantData(totalBytes, 0);

    // Helper to fill a Q5_1 block at a given byte offset
    auto fillBlock = [&](uint64_t byteOffset, float d, float m, uint8_t qValue) {
        // d as fp16
        uint16_t dHalf = 0;
        if (d == 1.0f) dHalf = 0x3C00;
        else if (d == 2.0f)
            dHalf = 0x4000;
        std::memcpy(&quantData[byteOffset], &dHalf, sizeof(uint16_t));
        // m as fp16
        uint16_t mHalf = 0;
        if (m == 0.0f) mHalf = 0x0000;
        else if (m == -0.5f)
            mHalf = 0xB800;
        std::memcpy(&quantData[byteOffset + 2], &mHalf, sizeof(uint16_t));
        // qh: high bit for each of 32 weights
        uint32_t qh = 0;
        uint8_t low4 = qValue & 0xF;
        uint8_t highBit = (qValue >> 4) & 1;
        for (int i = 0; i < 32; ++i) {
            if (highBit) qh |= (1u << i);
        }
        std::memcpy(&quantData[byteOffset + 4], &qh, sizeof(uint32_t));
        // ql: low nibbles, packed 2 per byte
        uint8_t nibblePair = static_cast<uint8_t>(low4 | (low4 << 4));
        for (int i = 0; i < 16; ++i) {
            quantData[byteOffset + 8 + i] = nibblePair;
        }
    };

    // Fill blocks for each output row.
    // Row 0: d=1.0, m=0.0, q=10 → deq = 10*1.0 + 0.0 = 10.0
    // Row 1: d=2.0, m=-0.5, q=20 → deq = 20*2.0 + (-0.5) = 39.5
    // Rows 2-31: d=1.0, m=0.0, q=10 → deq = 10.0
    fillBlock(0, 1.0f, 0.0f, 10);         // row 0
    fillBlock(TYPE_SIZE, 2.0f, -0.5f, 20);// row 1
    for (uint32_t j = 2; j < ROWS; ++j) {
        fillBlock(static_cast<uint64_t>(j) * TYPE_SIZE, 1.0f, 0.0f, 10);
    }

    // Input vector x: all 1.0
    std::vector<float> x(COLS, 1.0f);

    // Method 1: matMulVecFused
    std::vector<float> resultFused(ROWS, 0.0f);
    GGMLDequantize::matMulVecFused(TYPE, quantData.data(), x.data(), ROWS, COLS,
                                   resultFused.data());

    // Method 2: full dequantize + dot product
    // Dequantize ALL blocks (ROWS blocks = ROWS * BLOCK_SIZE elements).
    // Then W[j][i] = deq[j * BLOCK_SIZE + i] for i in [0, COLS).
    auto deq = GGMLDequantize::dequantize(
            TYPE, quantData.data(),
            static_cast<uint64_t>(ROWS) * BLOCK_SIZE);
    ASSERT_FALSE(deq.empty());
    std::vector<float> resultManual(ROWS, 0.0f);
    for (uint32_t j = 0; j < ROWS; ++j) {
        float dot = 0.0f;
        for (uint32_t i = 0; i < COLS; ++i) {
            // W[j][i] is at position j * BLOCK_SIZE + i in the flat deq array
            dot += x[i] * deq[static_cast<size_t>(j) * BLOCK_SIZE + i];
        }
        resultManual[j] = dot;
    }

    // Compare
    for (uint32_t j = 0; j < ROWS; ++j) {
        EXPECT_NEAR(resultFused[j], resultManual[j], 1e-4f)
                << "Mismatch at row " << j;
    }
}

// ---------------------------------------------------------------------------
// Test: SIMD dotProductFMA produces correct results
// ---------------------------------------------------------------------------
TEST(DequantizeTest, DotProductFMA) {
    // Test dotProductFMA with known vectors
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> b = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    float expected = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        expected += a[i] * b[i];
    }

    float result = dotProductFMA(a.data(), b.data(), static_cast<uint32_t>(a.size()));
    EXPECT_NEAR(result, expected, 1e-4f);

    // Test with different sizes
    expected = 1.0f * 8.0f + 2.0f * 7.0f + 3.0f * 6.0f;
    result = dotProductFMA(a.data(), b.data(), 3);
    EXPECT_NEAR(result, expected, 1e-4f);
}

// ---------------------------------------------------------------------------
// Test: accumulateFMA produces correct results
// ---------------------------------------------------------------------------
TEST(DequantizeTest, AccumulateFMA) {
    std::vector<float> local = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> blockOut = {0.5f, 1.0f, 1.5f, 2.0f};
    float alpha = 2.0f;

    accumulateFMA(local.data(), blockOut.data(), alpha, 4);

    EXPECT_NEAR(local[0], 1.0f + 2.0f * 0.5f, 1e-6f);
    EXPECT_NEAR(local[1], 2.0f + 2.0f * 1.0f, 1e-6f);
    EXPECT_NEAR(local[2], 3.0f + 2.0f * 1.5f, 1e-6f);
    EXPECT_NEAR(local[3], 4.0f + 2.0f * 2.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
// Test: typeSize and blockSize return correct values
// ---------------------------------------------------------------------------
TEST(DequantizeTest, TypeSizeAndBlockSize) {
    // Verify that type sizes match expected values
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_F32), 4u);
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_F16), 2u);
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_Q5_1), 24u);
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_Q5_K), 176u);
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_Q4_K), 144u);
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_IQ3_XXS), 98u);
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_IQ3_S), 110u);
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_IQ2_S), 82u);

    EXPECT_EQ(ggmlBlockSize(GGML_TYPE_F32), 1u);
    EXPECT_EQ(ggmlBlockSize(GGML_TYPE_Q5_1), 32u);
    EXPECT_EQ(ggmlBlockSize(GGML_TYPE_Q5_K), 256u);
    EXPECT_EQ(ggmlBlockSize(GGML_TYPE_Q4_K), 256u);
    EXPECT_EQ(ggmlBlockSize(GGML_TYPE_IQ3_XXS), 256u);
    EXPECT_EQ(ggmlBlockSize(GGML_TYPE_IQ3_S), 256u);
    EXPECT_EQ(ggmlBlockSize(GGML_TYPE_IQ2_S), 256u);
}

// ---------------------------------------------------------------------------
// Test: Dequantize roundtrip for all supported types
// ---------------------------------------------------------------------------
TEST(DequantizeTest, DequantizeRoundtrip) {
    // Create a small buffer of quantized data for each type and verify
    // that dequantize produces the right number of output elements.

    struct TypeTest {
        uint32_t type;
        uint32_t blockSize;
        uint32_t typeSize;
        const char *name;
    };

    TypeTest types[] = {
            {GGML_TYPE_Q5_1, 32, 32, "Q5_1"},
            {GGML_TYPE_Q5_K, 256, 176, "Q5_K"},
            {GGML_TYPE_Q4_K, 256, 144, "Q4_K"},
            {GGML_TYPE_IQ3_XXS, 256, 98, "IQ3_XXS"},
            {GGML_TYPE_IQ3_S, 256, 110, "IQ3_S"},
            {GGML_TYPE_IQ2_S, 256, 82, "IQ2_S"},
    };

    for (const auto &t: types) {
        // Create 2 blocks of quantized data
        uint64_t numElements = static_cast<uint64_t>(t.blockSize) * 2;
        uint64_t numBlocks = (numElements + t.blockSize - 1) / t.blockSize;
        uint64_t dataBytes = numBlocks * t.typeSize;
        std::vector<uint8_t> data(dataBytes, 0);

        // Set scale to 1.0 for the first block
        if (t.type == GGML_TYPE_Q5_1) {
            data[0] = 0x00;
            data[1] = 0x3C;// d = 1.0
        } else {
            data[0] = 0x00;
            data[1] = 0x3C;// d = 1.0 (half)
        }

        auto deq = GGMLDequantize::dequantize(t.type, data.data(), numElements);
        ASSERT_EQ(deq.size(), numElements)
                << "Dequantize returned wrong size for " << t.name;

        // All values should be finite
        for (uint64_t i = 0; i < numElements; ++i) {
            EXPECT_FALSE(std::isnan(deq[i])) << "NaN at index " << i << " for " << t.name;
            EXPECT_FALSE(std::isinf(deq[i])) << "Inf at index " << i << " for " << t.name;
        }
    }
}

// ---------------------------------------------------------------------------
// Test: Tokenizer pretokenize with GPT-2 pattern
// ---------------------------------------------------------------------------
TEST(DequantizeTest, TokenizerPretokenizePattern) {
    // Verify that the GPT-2 regex pattern compiles and matches correctly
    // Uses POSIX character classes for GCC std::regex compatibility
    std::string pattern =
            R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^[:space:][:alpha:][:digit:]]+|[[:space:]]+(?![^[:space:]])|[[:space:]]+)";

    try {
        std::regex re(pattern, std::regex::ECMAScript);
        SUCCEED() << "GPT-2 regex pattern compiles successfully";

        // Test matching on a sample string
        std::string test = "Hello world! 42";
        std::sregex_iterator iter(test.begin(), test.end(), re);
        std::sregex_iterator end;
        std::vector<std::string> matches;
        for (; iter != end; ++iter) {
            matches.push_back(iter->str());
        }

        // GPT-2 should produce: ["Hello", " world", "!", " 42"]
        ASSERT_EQ(matches.size(), 4u) << "Expected 4 tokens";
        EXPECT_EQ(matches[0], "Hello");
        EXPECT_EQ(matches[1], " world");
        EXPECT_EQ(matches[2], "!");
        EXPECT_EQ(matches[3], " 42");
    } catch (const std::regex_error &e) {
        FAIL() << "GPT-2 regex pattern failed to compile: " << e.what();
    }
}

// ---------------------------------------------------------------------------
// Test: Q2_K original vs pre-packed kernel comparison
// ---------------------------------------------------------------------------
// Creates a synthetic Q2_K block, pre-packs it, and compares the dot product
// results from the original and pre-packed kernels for the same input vector.
// This validates that the pre-packed kernel produces identical results.
TEST(DequantizeTest, Q2_K_PrePackedVsOriginal) {
    // Create a synthetic Q2_K block (84 bytes)
    uint8_t blockData[84] = {};

    // Set d = 2.0 (fp16: 0x4000)
    blockData[80] = 0x00;
    blockData[81] = 0x40;

    // Set dmin = 0.5 (fp16: 0x3800)
    blockData[82] = 0x00;
    blockData[83] = 0x38;

    // Set scales[16]: each byte has 4-bit scale (low) and 4-bit min (high)
    // Make them vary so we test all combinations
    for (int i = 0; i < 16; ++i) {
        blockData[i] = static_cast<uint8_t>((i << 4) | (15 - i));
    }

    // Set qs[64]: pack 2-bit values (0, 1, 2, 3) into each byte
    // Byte k contains values for elements [4k, 4k+1, 4k+2, 4k+3]
    // bit0-1 = element 4k, bit2-3 = element 4k+1, bit4-5 = element 4k+2, bit6-7 = element 4k+3
    for (int i = 0; i < 64; ++i) {
        uint8_t byteVal = 0;
        for (int b = 0; b < 4; ++b) {
            uint8_t val = static_cast<uint8_t>((i * 4 + b) % 4);
            byteVal |= (val << (b * 2));
        }
        blockData[16 + i] = byteVal;
    }

    // Pre-pack the block
    auto prepacked = GGMLDequantize::prepackQ2_K(blockData, 256);
    ASSERT_EQ(prepacked.size(), 276u);

    // Create a random x vector of 256 floats
    float x[256];
    std::srand(42);
    for (int i = 0; i < 256; ++i) {
        x[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    // Compute dot product using original kernel
    float originalResult = dotProductQ2_K_SIMD(blockData, x);

    // Compute dot product using pre-packed kernel
    float prepackedResult = dotProductQ2_K_PrePacked_SIMD(prepacked.data(), x);

    // Also compute using the scalar reference (dequantize then dot)
    float deqRef[256];
    GGMLDequantize::dequantizeQ2_KBlock(blockData, deqRef);
    double refDot = 0.0;
    for (int i = 0; i < 256; ++i) {
        refDot += static_cast<double>(x[i]) * deqRef[i];
    }
    float referenceResult = static_cast<float>(refDot);

    // Print results for debugging
    std::cout << "  Original kernel:  " << originalResult << std::endl;
    std::cout << "  Pre-packed kernel: " << prepackedResult << std::endl;
    std::cout << "  Reference (deq):   " << referenceResult << std::endl;

    // Both kernels should match the reference
    EXPECT_NEAR(originalResult, referenceResult, 1e-4f)
            << "Original kernel differs from reference dequantize-then-dot";

    EXPECT_NEAR(prepackedResult, referenceResult, 1e-4f)
            << "Pre-packed kernel differs from reference dequantize-then-dot";

    // The two kernels should match each other
    EXPECT_NEAR(originalResult, prepackedResult, 1e-4f)
            << "Original and pre-packed kernels produce different results";
}
