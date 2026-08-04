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
#include "ThreadPool.hpp"

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
            ThreadPool::instance().parallelFor(0, vocabSize, [&](uint32_t i) {
                const float *embRow = embedData + static_cast<uint64_t>(i) * hiddenSize;
                // SIMD-accelerated dot product: dot = sum(hidden[0..hiddenSize) * embRow[0..hiddenSize))
                float dot = dotProductFMA(hidden, embRow, hiddenSize);
                logits[i] = dot;
            });
        }

        /// @brief Compute LM head logits on CPU using FP16-stored embeddings.
        ///
        /// Same as computeCPU() but reads the embedding matrix from FP16 storage,
        /// halving memory bandwidth. The source embeddings are Q2_K (~2-bit), so
        /// FP16 is lossless. Uses F16C _mm256_cvtph_ps for on-the-fly conversion.
        ///
        /// @param hidden      Input hidden state vector (size hiddenSize)
        /// @param embedData   Pre-dequantized embedding matrix in FP16 (vocabSize × hiddenSize, row-major)
        /// @param vocabSize   Number of vocabulary entries
        /// @param hiddenSize  Hidden dimension size
        /// @param logits      Output logits vector (size vocabSize, pre-allocated)
        static void computeCPUF16(const float *hidden, const uint16_t *embedData,
                                  uint32_t vocabSize, uint32_t hiddenSize, float *logits) {
            ThreadPool::instance().parallelFor(0, vocabSize, [&](uint32_t i) {
                const uint16_t *embRow = embedData + static_cast<uint64_t>(i) * hiddenSize;
                logits[i] = dotProductFMA_F16(hidden, embRow, hiddenSize);
            });
        }

        /// @brief Compute LM head logits on CPU using Q8_K-stored embeddings.
        ///
        /// Same as computeCPU() but reads the embedding matrix from Q8_K storage
        /// (256 int8 values + a block scale per block), cutting memory bandwidth
        /// ~2× vs FP16 and ~4× vs F32. The dot product uses _mm256_maddubs_epi16
        /// int8 kernels. Since the source embeddings are Q2_K (~2-bit), Q8_K is
        /// lossless.
        ///
        /// @param hidden      Input hidden state vector (size hiddenSize)
        /// @param embedData   Pre-quantized embedding matrix in Q8_K (vocabSize × blocksPerRow, row-major)
        /// @param vocabSize   Number of vocabulary entries
        /// @param hiddenSize  Hidden dimension size
        /// @param logits      Output logits vector (size vocabSize, pre-allocated)
        static void computeCPUQ8K(const float *hidden, const Q8KBlock *embedData,
                                  uint32_t vocabSize, uint32_t hiddenSize, float *logits) {
            uint32_t blocksPerRow = (hiddenSize + 255) / 256;

            // Quantize the hidden vector to Q8_K once per token (reused across all
            // vocab rows). This is the key win: the original FP16/F32 paths read the
            // full hidden vector from RAM for every vocab row.
            // P7: reusable grow-only scratch (shared buffer, written by the calling
            // thread before parallelFor, only read by workers) so steady-state
            // generation performs no heap allocation here.
            static std::vector<Q8KBlock> q8Hidden;
            q8Hidden.resize(blocksPerRow);
            GGMLDequantize::quantizeQ8K(hidden, hiddenSize, q8Hidden.data());

            ThreadPool::instance().parallelFor(0, vocabSize, [&](uint32_t i) {
                const Q8KBlock *embRow = embedData + static_cast<uint64_t>(i) * blocksPerRow;
                // Software prefetch the next row's weight data to hide DRAM latency
                // on the large embedding matrix (vocabSize × hiddenSize).
                if (i + 1 < vocabSize) {
                    prefetchRow(embRow + blocksPerRow);
                }
                double dot = 0.0;
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    dot += static_cast<double>(dotProductQ8K_Q8K_SIMD(&q8Hidden[b], &embRow[b]));
                }
                logits[i] = static_cast<float>(dot);
            });
        }
        /// @brief Compute LM head logits using pre‑quantized Q8_K embedding matrix for a single token.
        /// This thin wrapper forwards to computeCPUQ8K, which performs the actual computation.
        static void matMulVecQ8K_Single(const float *hidden, const Q8KBlock *embedData,
                                        uint32_t vocabSize, uint32_t hiddenSize, float *logits) {
            computeCPUQ8K(hidden, embedData, vocabSize, hiddenSize, logits);
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
            uint64_t rowStrideBytes = static_cast<uint64_t>(numBlocks) * typeSize;

            // Initialize logits to zero (will accumulate block-by-block)
            std::memset(logits, 0, static_cast<size_t>(vocabSize) * sizeof(float));

            // Pre-resolve the dot product function to hoist the type switch
            // outside the inner loop (~912K calls per token otherwise).
            auto dotFunc = GGMLDequantize::getDotProductFunc(embedType);

            // Cache-blocked iteration: process one block column across ALL vocab
            // entries, then move to the next block. This keeps the hidden segment
            // (blockSize floats, ~1KB for K-quant) in L1 cache.
            //
            // In the original row-major iteration, the full hidden vector (6KB for
            // hiddenSize=1536) is read from RAM vocabSize=151936 times per token,
            // totaling ~930MB of hidden vector reads.
            //
            // In this cache-blocked version, the hidden vector is read only
            // numBlocks=6 times per token (~36KB total), with the hot segment
            // staying in L1 cache throughout the inner vocab loop.
            for (uint32_t b = 0; b < numBlocks; ++b) {
                uint32_t start = b * blockSize;
                uint32_t n = std::min(blockSize, hiddenSize - start);
                const float *xBlock = hidden + start;

                if (dotFunc && n == blockSize) {
                    // Fast path: pre-resolved fused dot product for full blocks
                    ThreadPool::instance().parallelFor(0, vocabSize, [&](uint32_t i) {
                        const uint8_t *blockData = embedData + static_cast<uint64_t>(i) * rowStrideBytes + static_cast<uint64_t>(b) * typeSize;
                        logits[i] += dotFunc(blockData, xBlock);
                    });
                } else {
                    // Fallback: general fused dot product (handles partial blocks
                    // or types without a dedicated fast path)
                    ThreadPool::instance().parallelFor(0, vocabSize, [&](uint32_t i) {
                        const uint8_t *blockData = embedData + static_cast<uint64_t>(i) * rowStrideBytes + static_cast<uint64_t>(b) * typeSize;
                        logits[i] += GGMLDequantize::dotProductFused(embedType, blockData, xBlock, n);
                    });
                }
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
