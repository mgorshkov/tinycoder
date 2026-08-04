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

#include "GGMLDequantize.hpp"
#include "GGUFLoader.hpp"
#include "Model.hpp"
#include "SIMDMatMulVec.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace tinycoder {

    /// @brief Dequantize a single block of quantized data, dispatching by type.
    ///        Block size is always 256 for K-quant and IQ types.
    /// @param ggmlType  The quantization type.
    /// @param blockData Pointer to the block's quantized data.
    /// @param out       Output buffer (must hold at least blockSize floats).
    /// @param blockSize Number of elements in the block (e.g., 256).
    static void dequantizeBlock(uint32_t ggmlType, const uint8_t *blockData,
                                float *out, uint32_t blockSize) {
        // Delegate to GGMLDequantize::dequantizeBlock which has proper
        // block-level dequantizers for all supported types (Q5_1, Q5_K, Q4_K,
        // IQ3_XXS, IQ3_S, IQ2_S).
        GGMLDequantize::dequantizeBlock(ggmlType, blockData, out, blockSize);
    }

    std::vector<float> Model::QuantizedEmbedding::getRow(uint32_t tokenId) const {
        uint32_t blockSize = ggmlBlockSize(type);
        uint32_t typeSize = ggmlTypeSize(type);
        if (blockSize == 0 || typeSize == 0) {
            // Fallback: use generic dequantize
            return GGMLDequantize::dequantize(
                    type, data.data() + static_cast<uint64_t>(tokenId) * hiddenSize,
                    hiddenSize);
        }

        std::vector<float> result(hiddenSize);
        uint32_t numBlocks = (hiddenSize + blockSize - 1) / blockSize;

        for (uint32_t b = 0; b < numBlocks; ++b) {
            uint64_t blockOffset = static_cast<uint64_t>(tokenId) * numBlocks + b;
            const uint8_t *blockData = data.data() + blockOffset * typeSize;
            float blockOut[256];// max block size is 256
            dequantizeBlock(type, blockData, blockOut, blockSize);
            uint32_t start = b * blockSize;
            uint32_t end = std::min(start + blockSize, hiddenSize);
            for (uint32_t i = start; i < end; ++i) {
                result[i] = blockOut[i - start];
            }
        }
        return result;
    }

    float Model::QuantizedEmbedding::dotRow(const float *vec,
                                            uint32_t tokenId) const {
        // Compute dot product of vec with a single quantized embedding row.
        // Dequantize one block at a time and accumulate the dot product directly,
        // avoiding the intermediate float vector allocation.
        uint32_t blockSize = ggmlBlockSize(type);
        uint32_t typeSize = ggmlTypeSize(type);
        if (blockSize == 0 || typeSize == 0) {
            // Fallback: use generic dequantize for the full row
            auto deq = GGMLDequantize::dequantize(
                    type, data.data() + static_cast<uint64_t>(tokenId) * hiddenSize,
                    hiddenSize);
            float result = 0.0f;
            for (uint32_t i = 0; i < hiddenSize; ++i) {
                result += vec[i] * deq[i];
            }
            return result;
        }

        uint32_t numBlocks = (hiddenSize + blockSize - 1) / blockSize;
        float result = 0.0f;

        for (uint32_t b = 0; b < numBlocks; ++b) {
            uint64_t blockOffset = static_cast<uint64_t>(tokenId) * numBlocks + b;
            const uint8_t *blockData = data.data() + blockOffset * typeSize;
            float blockOut[256];// max block size is 256
            dequantizeBlock(type, blockData, blockOut, blockSize);
            uint32_t start = b * blockSize;
            uint32_t n = std::min(blockSize, hiddenSize - start);
            // SIMD-accelerated dot product: result += sum(vec[start..start+n) *
            // blockOut[0..n))
            result += dotProductFMA(vec + start, blockOut, n);
        }
        return result;
    }

}// namespace tinycoder
