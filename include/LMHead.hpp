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

#include <cstdint>
#include <cstring>
#include <vector>

#include "GGMLDequantize.hpp"
#include "GGUFLoader.hpp"
#include "SIMDMatMulVec.hpp"

namespace tinycoder {

    /// @brief Optimized LM head computation: logits = hidden × embeddings^T
    ///
    /// For the tied-embedding case (weight tying), the LM head is a matrix-vector
    /// multiply of shape [1, hiddenSize] × [vocabSize, hiddenSize]^T = [1,
    /// vocabSize].
    ///
    /// This is equivalent to computing dot(hidden, embedding[i]) for each vocab
    /// entry i.
    ///
    /// Two optimized paths are provided:
    ///   1. CUDA: Uses cublasSgemv with the embedding matrix persistently on GPU
    ///   2. CPU:  Uses OpenMP parallelization over the vocabulary loop
    struct LMHead {
        /// @brief Compute LM head logits on CPU with OpenMP parallelization.
        ///
        /// Fast path: uses pre-dequantized (float32) embedding matrix.
        /// This is the optimal path when the model pre-dequantizes embeddings during load.
        ///
        /// @param hidden      Input hidden state vector (size hiddenSize)
        /// @param embedData   Pre-dequantized embedding matrix (vocabSize × hiddenSize, row-major)
        /// @param vocabSize   Number of vocabulary entries
        /// @param hiddenSize  Hidden dimension size
        /// @param logits      Output logits vector (size vocabSize, pre-allocated)
        static void computeCPU(const float *hidden, const float *embedData,
                               uint32_t vocabSize, uint32_t hiddenSize, float *logits) {
// Parallel over vocabulary, SIMD dot product inside
#pragma omp parallel for schedule(static)
            for (int32_t i = 0; i < static_cast<int32_t>(vocabSize); ++i) {
                const float *embRow = embedData + static_cast<uint64_t>(i) * hiddenSize;
                // SIMD-accelerated dot product: dot = sum(hidden[0..hiddenSize) * embRow[0..hiddenSize))
                float dot = dotProductFMA(hidden, embRow, hiddenSize);
                logits[i] = dot;
            }
        }

        /// @brief Compute LM head logits on CPU with on-the-fly dequantization.
        ///
        /// For each vocabulary entry i, computes dot(hidden, embedding[i]) where
        /// embedding[i] is dequantized on-the-fly from the quantized embedding
        /// matrix. Used as fallback when pre-dequantized embeddings are unavailable.
        ///
        /// @param hidden      Input hidden state vector (size hiddenSize)
        /// @param embedData   Raw quantized embedding matrix data
        /// @param embedType   GGML quantization type of embeddings
        /// @param vocabSize   Number of vocabulary entries
        /// @param hiddenSize  Hidden dimension size
        /// @param logits      Output logits vector (size vocabSize, pre-allocated)
        static void computeCPUQuantized(const float *hidden, const uint8_t *embedData,
                                        uint32_t embedType, uint32_t vocabSize,
                                        uint32_t hiddenSize, float *logits) {
            uint32_t blockSize = ggmlBlockSize(embedType);
            uint32_t typeSize = ggmlTypeSize(embedType);

            if (blockSize == 0 || typeSize == 0) {
                // Fallback for unknown types
                for (uint32_t i = 0; i < vocabSize; ++i) {
                    auto deq = GGMLDequantize::dequantize(
                            embedType, embedData + static_cast<uint64_t>(i) * hiddenSize,
                            hiddenSize);
                    float dot = 0.0f;
                    for (uint32_t j = 0; j < hiddenSize; ++j) {
                        dot += hidden[j] * deq[j];
                    }
                    logits[i] = dot;
                }
                return;
            }

            uint32_t numBlocks = (hiddenSize + blockSize - 1) / blockSize;

// Parallel over vocabulary with SIMD dot product inside
#pragma omp parallel for schedule(static)
            for (int32_t i = 0; i < static_cast<int32_t>(vocabSize); ++i) {
                float dot = 0.0f;
                for (uint32_t b = 0; b < numBlocks; ++b) {
                    uint64_t blockOffset = static_cast<uint64_t>(i) * numBlocks + b;
                    const uint8_t *blockData = embedData + blockOffset * typeSize;
                    float blockOut[256];// max block size is 256
                    GGMLDequantize::dequantizeBlock(embedType, blockData, blockOut,
                                                    blockSize);
                    uint32_t start = b * blockSize;
                    uint32_t n = std::min(blockSize, hiddenSize - start);
                    // SIMD-accelerated dot product: dot += sum(hidden[start..start+n) *
                    // blockOut[0..n))
                    dot += dotProductFMA(hidden + start, blockOut, n);
                }
                logits[i] = dot;
            }
        }

        /// @brief Compute LM head logits on CPU with OpenMP parallelization (separate
        /// LM head).
        ///
        /// For a separate (non-tied) LM head stored as a QuantizedMatrix.
        ///
        /// @param hidden      Input hidden state vector (size hiddenSize)
        /// @param lmHeadData  Raw quantized LM head matrix data
        /// @param lmHeadType  GGML quantization type of LM head
        /// @param vocabSize   Number of vocabulary entries
        /// @param hiddenSize  Hidden dimension size
        /// @param logits      Output logits vector (size vocabSize, pre-allocated)
        static void computeCPUSeparate(const float *hidden, const uint8_t *lmHeadData,
                                       uint32_t lmHeadType, uint32_t vocabSize,
                                       uint32_t hiddenSize, float *logits) {
            // The separate LM head is a quantized matrix of shape (vocabSize x
            // hiddenSize). We need y = x * W^T where W is (vocabSize x hiddenSize).
            // This is the same as the tied case but with a different data pointer.
            computeCPUQuantized(hidden, lmHeadData, lmHeadType, vocabSize, hiddenSize, logits);
        }
    };

}// namespace tinycoder
