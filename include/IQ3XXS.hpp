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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <np/Array.hpp>

#include "ThreadPool.hpp"

namespace tinycoder {

    /// @brief IQ3_XXS quantization constants and dequantization routines.
    ///
    /// IQ3_XXS is a 3.0625-bit quantization scheme:
    /// - 32 weights per block (blockSize = 32)
    /// - 3 bits per weight stored in a 96-bit packed representation
    /// - 6-bit shared scale per block (quantized to 6 bits)
    /// - Total: (32 * 3 + 6) / 32 = 3.0625 bits per weight
    ///
    /// The 32 weights are packed into 12 bytes (96 bits = 32 * 3 bits).
    /// Each weight is stored as a 3-bit unsigned value [0..7].
    /// Dequantization: w_deq = (w_3bit - 4) * scale / 4.0
    /// where scale = 2^(scale_6bit - 32) / max_abs
    struct IQ3XXS {
        static constexpr uint32_t BLOCK_SIZE = 32;
        static constexpr uint32_t WEIGHT_BITS = 3;
        static constexpr uint32_t SCALE_BITS = 6;
        static constexpr uint32_t PACKED_BYTES = 12;// 32 * 3 / 8 = 12 bytes

        /// @brief A single IQ3_XXS quantized block (32 weights).
        struct Block {
            uint8_t qdata[PACKED_BYTES];// Packed 3-bit weights (96 bits)
            uint8_t scale;              // 6-bit scale (upper 6 bits used)
        };

        static_assert(sizeof(Block) == 13, "IQ3_XXS block must be 13 bytes");

        /// @brief A quantized weight matrix stored as IQ3_XXS blocks.
        ///
        /// Keeps weights in quantized format to save memory (~3.1 bits/weight).
        /// Dequantization happens on-the-fly during matrix-vector multiplication.
        struct QuantizedMatrix {
            std::vector<Block> blocks;
            uint32_t rows = 0;
            uint32_t cols = 0;

            /// @brief Number of blocks per row (columns are split into blocks of
            /// BLOCK_SIZE).
            uint32_t blocksPerRow() const {
                return (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            }

            /// @brief Total number of blocks.
            uint32_t totalBlocks() const { return rows * blocksPerRow(); }

            /// @brief Get pointer to blocks for a given row.
            const Block *rowBlocks(uint32_t row) const {
                return blocks.data() + row * blocksPerRow();
            }

            /// @brief Check if matrix is empty.
            bool empty() const { return blocks.empty(); }

            /// @brief Get memory usage in bytes (quantized).
            size_t memoryBytes() const { return blocks.size() * sizeof(Block); }

            /// @brief Get equivalent F32 memory usage in bytes for comparison.
            size_t f32MemoryBytes() const {
                return static_cast<size_t>(rows) * cols * sizeof(float);
            }
        };

        /// @brief Compute y = x * W^T where W is a quantized matrix.
        /// @param x Input vector of size cols
        /// @param w Quantized weight matrix (rows x cols)
        /// @param y Output vector of size rows
        static void matVecMul(const float *x, const QuantizedMatrix &w, float *y) {
            uint32_t rows = w.rows;
            uint32_t cols = w.cols;
            uint32_t bpr = w.blocksPerRow();

            ThreadPool::instance().parallelFor(0, rows, [&](uint32_t r) {
                const Block *rowBlks = w.rowBlocks(r);
                float sum = 0.0f;

                for (uint32_t bc = 0; bc < bpr; ++bc) {
                    const Block &blk = rowBlks[bc];
                    float scale = decodeScale(blk.scale);
                    uint32_t colStart = bc * BLOCK_SIZE;
                    uint32_t colEnd = std::min(colStart + BLOCK_SIZE, cols);

                    for (uint32_t c = colStart; c < colEnd; ++c) {
                        uint32_t i = c - colStart;
                        // Unpack 3-bit weight
                        uint32_t bitPos = i * 3;
                        uint32_t byteIdx = bitPos / 8;
                        uint32_t bitOffset = bitPos % 8;

                        uint32_t val;
                        if (bitOffset <= 5) {
                            val = (blk.qdata[byteIdx] >> bitOffset) & 0x7;
                            if (byteIdx + 1 < PACKED_BYTES && (bitOffset + 3 > 8)) {
                                val |= (blk.qdata[byteIdx + 1] << (8 - bitOffset)) & 0x7;
                            }
                        } else {
                            val = (blk.qdata[byteIdx] >> bitOffset) & 0x7;
                            val |= (blk.qdata[byteIdx + 1] << (8 - bitOffset)) & 0x7;
                        }
                        val &= 0x7;

                        float wVal =
                                (static_cast<float>(static_cast<int32_t>(val) - 4)) * scale;
                        sum += x[c] * wVal;
                    }
                }
                y[r] = sum;
            });
        }

        /// @brief Dequantize a single block into np::Array<float>.
        /// @param block The quantized block
        /// @return Dequantized float array of size BLOCK_SIZE
        static np::Array<float> dequantizeBlock(const Block &block) {
            std::array<float, BLOCK_SIZE> result{};
            float scale = decodeScale(block.scale);

            for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
                uint32_t bitPos = i * 3;
                uint32_t byteIdx = bitPos / 8;
                uint32_t bitOffset = bitPos % 8;

                uint32_t val;
                if (bitOffset <= 5) {
                    val = (block.qdata[byteIdx] >> bitOffset) & 0x7;
                    if (byteIdx + 1 < PACKED_BYTES && (bitOffset + 3 > 8)) {
                        val |= (block.qdata[byteIdx + 1] << (8 - bitOffset)) & 0x7;
                    }
                } else {
                    val = (block.qdata[byteIdx] >> bitOffset) & 0x7;
                    val |= (block.qdata[byteIdx + 1] << (8 - bitOffset)) & 0x7;
                }
                val &= 0x7;

                result[i] = (static_cast<float>(static_cast<int32_t>(val) - 4)) * scale;
            }

            return np::Array<float>(result);
        }

        /// @brief Dequantize multiple blocks into a contiguous float array.
        static np::Array<float> dequantize(const Block *blocks, uint32_t numBlocks) {
            std::vector<float> result(numBlocks * BLOCK_SIZE);

            ThreadPool::instance().parallelFor(0, numBlocks, [&](uint32_t b) {
                auto deq = dequantizeBlock(blocks[b]);
                std::memcpy(&result[b * BLOCK_SIZE], deq.data(),
                            BLOCK_SIZE * sizeof(float));
            });

            return np::Array<float>(result, np::Shape{numBlocks * BLOCK_SIZE});
        }

        /// @brief Dequantize a weight matrix (2D: rows x cols) into np::Array<float>.
        static np::Array<float> dequantizeMatrix(const Block *blocks, uint32_t rows,
                                                 uint32_t cols) {
            uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

            std::vector<float> result(rows * cols);

            ThreadPool::instance().parallelFor(0, rows, [&](uint32_t r) {
                for (uint32_t bc = 0; bc < blocksPerRow; ++bc) {
                    uint32_t blockIdx = r * blocksPerRow + bc;
                    auto deq = dequantizeBlock(blocks[blockIdx]);

                    uint32_t colStart = bc * BLOCK_SIZE;
                    uint32_t colEnd = std::min(colStart + BLOCK_SIZE, cols);
                    for (uint32_t c = colStart; c < colEnd; ++c) {
                        result[r * cols + c] = static_cast<float>(deq.get(c - colStart));
                    }
                }
            });

            return np::Array<float>(result, np::Shape{rows, cols});
        }

        /// @brief Decode a 6-bit scale value to float.
        /// Public so that on-the-fly dequantization in Model.cpp can use it.
        static float decodeScale(uint8_t encodedScale) {
            uint8_t scaleVal = (encodedScale >> 2) & 0x3F;
            int32_t exponent = static_cast<int32_t>(scaleVal) - 32;
            return std::ldexp(1.0f, exponent) / 4.0f;
        }
    };

}// namespace tinycoder
