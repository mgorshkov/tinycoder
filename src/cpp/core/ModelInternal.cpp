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

#include "ModelInternal.hpp"

#include "GGMLDequantize.hpp"
#include "ThreadPool.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>

namespace tinycoder::detail {

    // ---------------------------------------------------------------------
    // FP16 / Q8_K batched mat-mul helpers. Moved verbatim from Model.cpp's
    // anonymous namespace; only the `static` qualifier was removed so they
    // can be shared across the Model translation units.
    // ---------------------------------------------------------------------

    void dumpVecStats(const float *v, uint32_t n, const std::string &label) {
        float minV = v[0], maxV = v[0], sumV = 0;
        for (uint32_t i = 0; i < n; ++i) {
            minV = std::min(minV, v[i]);
            maxV = std::max(maxV, v[i]);
            sumV += v[i];
        }
        std::cout << "  " << label << ": first8=";
        for (uint32_t i = 0; i < 8 && i < n; ++i)
            std::cout << std::fixed << std::setprecision(6) << v[i] << " ";
        std::cout << "min=" << minV << " max=" << maxV
                  << " mean=" << (sumV / static_cast<float>(n)) << std::endl;
    }

    // Forward declaration of out-parameter version needed by return-value version
    void deqMatMulVecF16(const uint16_t *W_f16, const float *x,
                         uint32_t rows, uint32_t cols, float *out);

    /// @brief Matrix-vector multiply with FP16-stored weights (return-value version).
    ///
    /// Computes y_j = sum_i x[i] * W_f16[j][i] for j in [0, rows).
    /// W_f16 is stored as FP16 (uint16_t) to halve memory bandwidth.
    /// Uses F16C _mm256_cvtph_ps for on-the-fly FP16->FP32 conversion.
    np::Array<float> deqMatMulVecF16(const uint16_t *W_f16, const float *x,
                                     uint32_t rows, uint32_t cols) {
        np::Array<float> result(np::Shape{rows});
        deqMatMulVecF16(W_f16, x, rows, cols, result.data());
        return result;
    }

    /// @brief Matrix-vector multiply with FP16-stored weights (out-parameter version).
    ///
    /// Computes y_j = sum_i x[i] * W_f16[j][i] for j in [0, rows).
    /// W_f16 is stored as FP16 (uint16_t) to halve memory bandwidth.
    /// Uses F16C _mm256_cvtph_ps for on-the-fly FP16->FP32 conversion.
    ///
    /// This function is ALWAYS called from within a ThreadPool::parallelFor
    /// lambda, so it uses a plain sequential loop (no nested parallelFor).
    void deqMatMulVecF16(const uint16_t *W_f16, const float *x,
                         uint32_t rows, uint32_t cols, float *out) {
        ScopedProfile sp("gen_deqMatMulVecF16");
        // Cache-blocked iteration over block columns. For each block column
        // we iterate over ALL rows, keeping the hot x segment (kBlockCols
        // floats) resident in L1 cache. This avoids re-reading the full x
        // vector (6KB for hiddenSize=1536) from RAM for every output row.
        //
        // In the original row-major iteration, x is read rows times per
        // call (e.g. 8960 times for ffnDown), totaling ~54MB of x reads.
        // In this cache-blocked version, x is read only numBlocks times per
        // call, with the hot segment staying in L1 throughout the inner
        // row loop.
        constexpr uint32_t kBlockCols = 256;// 256 floats = 1KB of x in L1

        // Initialize outputs to zero (accumulate block-by-block)
        std::memset(out, 0, static_cast<size_t>(rows) * sizeof(float));

        // Hoist the SIMD dispatch out of the per-row/per-block inner loop
        // (BENCHMARK_REPORT §4.6 / P3): resolve the implementation pointer
        // once per matmul call instead of doing an atomic acquire load on
        // every block.
        auto dotF16 = dotProductFMA_F16_get();

        // Parallelize over output rows. Each thread owns a contiguous range
        // of rows and iterates over all block columns for those rows, so the
        // x segment stays in L1 per-thread. For single-token generation
        // (seqLen==1) the outer token-level parallelFor is bypassed, making
        // this the only source of parallelism (uses all threads). For
        // seqLen>1 the ThreadPool re-entrancy guard runs this sequentially.
        ThreadPool::instance().parallelFor(0, rows, [&](uint32_t j) {
            const uint16_t *Wrow = W_f16 + static_cast<size_t>(j) * cols;
            // Software prefetch the next row's weight data to hide DRAM latency
            // on the large FP16 weight matrix (rows × cols). Rows are handed out
            // in contiguous ranges, so the next row is the natural next access.
            if (j + 1 < rows) {
                prefetchRow(Wrow + cols);
            }
            float acc = 0.0f;
            for (uint32_t b = 0; b < cols; b += kBlockCols) {
                uint32_t n = std::min(kBlockCols, cols - b);
                acc += dotF16(x + b, Wrow + b, n);
            }
            out[j] = acc;
        });
    }

    /// @brief Fused QKV matrix-vector multiplication.
    ///
    /// Computes Q = x * attnQ^T, K = x * attnK^T, V = x * attnV^T in a single
    /// pass over the input vector x. This reduces x-vector reads from 3× to 1×
    /// per token per layer, improving cache utilization.
    ///
    /// All three matrices must use the same quantized type (typically Q2_K).
    /// For F32 matrices, falls back to three separate calls.
    /// @brief Fused Q/K/V projection with the combined row space parallelized
    /// over the thread pool (plan §5).
    ///
    /// Dispatch note: a plain `parallelFor(0, 3, ...)` over the three projections
    /// is a NO-OP here — ThreadPool runs `end - start <= numThreads_` serially
    /// (3 <= 8 on this box). Instead we tile the combined Q+K+V row space
    /// (qRows + kRows + vRows = 1536 + 256 + 256 = 2048 for Qwen2.5-1.5B), which
    /// is well above the parallelism threshold. Each output row is independent
    /// (single writer), x is read-only and stays L1-resident (~hiddenSize floats),
    /// and the per-row block summation order is unchanged, so results are
    /// bit-identical to the serial block-outer loop.
    void matMulVecFusedQKV(
            const QuantizedMatrix &qMat, const QuantizedMatrix &kMat, const QuantizedMatrix &vMat,
            const float *x,
            float *qOut, float *kOut, float *vOut) {
        ScopedProfile sp("gen_matMulVecFusedQKV");
        uint32_t qRows = qMat.rows;
        uint32_t kRows = kMat.rows;
        uint32_t vRows = vMat.rows;
        uint32_t cols = qMat.cols;// All three share the same input dimension

        uint32_t blockSizeQ = ggmlBlockSize(qMat.type);
        uint32_t typeSizeQ = ggmlTypeSize(qMat.type);
        uint32_t blockSizeK = ggmlBlockSize(kMat.type);
        uint32_t typeSizeK = ggmlTypeSize(kMat.type);
        uint32_t blockSizeV = ggmlBlockSize(vMat.type);
        uint32_t typeSizeV = ggmlTypeSize(vMat.type);

        bool fusedQ = blockSizeQ != 0 && typeSizeQ != 0;
        bool fusedK = blockSizeK != 0 && typeSizeK != 0;
        bool fusedV = blockSizeV != 0 && typeSizeV != 0;

        if (!fusedQ || !fusedK || !fusedV || cols == 0) {
            // F32 or unknown quantisation types: fall back to the per-matrix
            // matMulVec paths (each is internally parallelized when profitable).
            qMat.matMulVec(x, qOut);
            kMat.matMulVec(x, kOut);
            vMat.matMulVec(x, vOut);
            return;
        }

        uint32_t blocksPerRowQ = (cols + blockSizeQ - 1) / blockSizeQ;
        uint64_t rowStrideQ = static_cast<uint64_t>(blocksPerRowQ) * typeSizeQ;
        uint32_t blocksPerRowK = (cols + blockSizeK - 1) / blockSizeK;
        uint64_t rowStrideK = static_cast<uint64_t>(blocksPerRowK) * typeSizeK;
        uint32_t blocksPerRowV = (cols + blockSizeV - 1) / blockSizeV;
        uint64_t rowStrideV = static_cast<uint64_t>(blocksPerRowV) * typeSizeV;

        // Combined row space: Q rows, then K rows, then V rows.
        uint32_t totalRows = qRows + kRows + vRows;
        if (totalRows <= ThreadPool::instance().numThreads()) {
            // Small combined matrix: keep the serial block-outer loop (avoids the
            // spin-barrier + atomic dispatch overhead on tiny ranges).
            for (uint32_t b = 0; b < blocksPerRowQ; ++b) {
                uint32_t start = b * blockSizeQ;
                uint32_t n = std::min(blockSizeQ, cols - start);
                const float *xBlock = x + start;
                for (uint32_t j = 0; j < qRows; ++j) {
                    const uint8_t *blockData = qMat.data.data() +
                                               static_cast<uint64_t>(j) * rowStrideQ + static_cast<uint64_t>(b) * typeSizeQ;
                    qOut[j] += GGMLDequantize::dotProductFused(qMat.type, blockData, xBlock, n);
                }
                for (uint32_t j = 0; j < kRows; ++j) {
                    const uint8_t *blockData = kMat.data.data() +
                                               static_cast<uint64_t>(j) * rowStrideK + static_cast<uint64_t>(b) * typeSizeK;
                    kOut[j] += GGMLDequantize::dotProductFused(kMat.type, blockData, xBlock, n);
                }
                for (uint32_t j = 0; j < vRows; ++j) {
                    const uint8_t *blockData = vMat.data.data() +
                                               static_cast<uint64_t>(j) * rowStrideV + static_cast<uint64_t>(b) * typeSizeV;
                    vOut[j] += GGMLDequantize::dotProductFused(vMat.type, blockData, xBlock, n);
                }
            }
            return;
        }

        // Tile the combined row space over the pool. Each task owns a contiguous
        // run of output rows of a single projection; the next row index decoded
        // from the tile offset. Per-row accumulation (blocks in ascending order)
        // is identical to the serial path.
        constexpr uint32_t kTileRows = 16;
        uint32_t numTiles = (totalRows + kTileRows - 1) / kTileRows;

        ThreadPool::instance().parallelFor(0, numTiles, [&](uint32_t tile) {
            uint32_t rowStart = tile * kTileRows;
            uint32_t rowEnd = std::min(rowStart + kTileRows, totalRows);
            for (uint32_t j = rowStart; j < rowEnd; ++j) {
                uint32_t ggmlType;
                const uint8_t *data;
                uint32_t blocksPerRow;
                uint64_t rowStride;
                uint32_t typeSize;
                uint32_t r;
                float *out;
                if (j < qRows) {
                    ggmlType = qMat.type;
                    data = qMat.data.data();
                    blocksPerRow = blocksPerRowQ;
                    rowStride = rowStrideQ;
                    typeSize = typeSizeQ;
                    r = j;
                    out = qOut;
                } else if (j < qRows + kRows) {
                    ggmlType = kMat.type;
                    data = kMat.data.data();
                    blocksPerRow = blocksPerRowK;
                    rowStride = rowStrideK;
                    typeSize = typeSizeK;
                    r = j - qRows;
                    out = kOut;
                } else {
                    ggmlType = vMat.type;
                    data = vMat.data.data();
                    blocksPerRow = blocksPerRowV;
                    rowStride = rowStrideV;
                    typeSize = typeSizeV;
                    r = j - qRows - kRows;
                    out = vOut;
                }

                uint32_t col = 0;
                float acc = 0.0f;
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    uint32_t n = std::min(ggmlBlockSize(ggmlType), cols - col);
                    const float *xBlock = x + col;
                    const uint8_t *blockData = data +
                                               static_cast<uint64_t>(r) * rowStride + static_cast<uint64_t>(b) * typeSize;
                    acc += GGMLDequantize::dotProductFused(ggmlType, blockData, xBlock, n);
                    col += n;
                }
                out[r] = acc;
            }
        });
    }

    /// @brief Batched matrix-matrix multiply with FP16-stored weights.
    ///
    /// Computes out[s*rows + j] = sum_i X[s*cols + i] * W_f16[j*cols + i]
    /// for s in [0, seqLen), j in [0, rows). X is [seqLen, cols], W_f16 is
    /// [rows, cols], out is [seqLen, rows] (row-major).
    ///
    /// This is the key prefill optimization: instead of a per-token GEMV that
    /// re-reads the full weight matrix from DRAM for every token, we parallelize
    /// over output rows and reuse each weight row across all tokens. The weight
    /// matrix is read once instead of seqLen times.
    void deqMatMulVecF16_Batch(const uint16_t *W_f16, const float *X,
                               uint32_t seqLen, uint32_t rows,
                               uint32_t cols, float *out) {
        if (seqLen == 0 || rows == 0 || cols == 0) {
            return;
        }
        ScopedProfile sp("deqMatMulVecF16_Batch");
        // Cache-blocked (P2): the previous version parallelised over every
        // output row and, for each row, looped over all seqLen tokens reading
        // the entire X matrix. With rows=1536 and seqLen=34 the X matrix is
        // re-read once per row (~1536x) from DRAM — a huge bandwidth waste for
        // the large FP16 attnO / ffnDown matmuls. Here we process a tile of
        // TILE_ROWS output rows at a time and iterate column-blocks so the X
        // column-slice for all tokens stays hot in L1/L2 and is reused across
        // the TILE_ROWS rows in the tile. This reduces X DRAM traffic by
        // ~TILE_ROWSx (standard GEMM blocking).
        constexpr uint32_t kBlockCols = 256;
        constexpr uint32_t kTileRows = 8;
        uint32_t numRowTiles = (rows + kTileRows - 1) / kTileRows;

        ThreadPool::instance().parallelFor(0, numRowTiles, [&](uint32_t tile) {
            uint32_t rowStart = tile * kTileRows;
            uint32_t tRows = std::min(kTileRows, rows - rowStart);

            // Per-tile accumulators [tRows][seqLen]. P7: thread-local reusable
            // buffer — grows on demand and is zeroed per tile, so steady-state
            // prefill performs no heap allocation here.
            static thread_local std::vector<float> acc;
            acc.assign(static_cast<size_t>(tRows) * seqLen, 0.0f);

            for (uint32_t b = 0; b < cols; b += kBlockCols) {
                uint32_t n = std::min(kBlockCols, cols - b);
                for (uint32_t r = 0; r < tRows; ++r) {
                    const uint16_t *Wrow =
                            W_f16 + static_cast<size_t>(rowStart + r) * cols + b;
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        const float *xrow = X + static_cast<size_t>(s) * cols + b;
                        acc[static_cast<size_t>(r) * seqLen + s] +=
                                dotProductFMA_F16(xrow, Wrow, n);
                    }
                }
            }

            for (uint32_t r = 0; r < tRows; ++r) {
                for (uint32_t s = 0; s < seqLen; ++s) {
                    out[static_cast<size_t>(s) * rows + rowStart + r] =
                            acc[static_cast<size_t>(r) * seqLen + s];
                }
            }
        });
    }

    /// @brief Register-tiled Q8_K batch GEMM (P4).
    ///
    /// Computes out[s*rows + j] = sum_i X[s*cols + i] * W[j*cols + i] where W
    /// is a pre-packed Q8_K matrix (row-major Q8KBlock arrays). X is quantized
    /// to Q8_K once, then each (row, token, block) dot uses the int8
    /// _mm256_maddubs_epi16 kernel (32 MACs/instr vs 8 for the FP16 path),
    /// giving ~4x compute throughput on the largest prefill matmuls (attnO and
    /// ffnDown). Rows are processed in tiles so the Q8_K x data is reused
    /// across the tile rows.
    void matMulVecBatchQ8K(const Q8KBlock *W_q8k, const float *X,
                           uint32_t seqLen, uint32_t rows,
                           uint32_t cols, float *out) {
        if (seqLen == 0 || rows == 0 || cols == 0) {
            return;
        }
        ScopedProfile sp("matMulVecBatchQ8K");
        // Prefer the register-tiled vector batch GEMM: it keeps the 8 partial
        // sums in __m256 accumulators (horizontal-summed once per output),
        // loads and sign-processes the Q8_K x-vector once per (token, block)
        // and reuses it across the 8 tile rows, and avoids the scalar
        // function-pointer dispatch per (row, token, block). The vector
        // kernel performs its own x-quantization. Falls back to the scalar
        // per-pair loop below on hosts without a vector kernel.
        if (matMulVecBatchQ8K_SIMD(W_q8k, X, seqLen, rows, cols, out)) {
            return;
        }
        constexpr uint32_t BLOCK_SIZE = 256;
        uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Quantize each token's x to Q8_K once (reused across all rows).
        // P7: reusable grow-only scratch instead of a fresh heap allocation
        // on every prefill call. This buffer is written by the CALLING thread
        // before parallelFor and only read by workers afterwards, so it must
        // be SHARED (a thread_local would give workers an empty copy). The
        // kernels are invoked serially from the forward() path, so a plain
        // static is safe.
        static std::vector<Q8KBlock> q8All;
        q8All.resize(static_cast<size_t>(seqLen) * blocksPerRow);
        for (uint32_t s = 0; s < seqLen; ++s) {
            GGMLDequantize::quantizeQ8K(X + static_cast<size_t>(s) * cols, cols,
                                        q8All.data() + static_cast<size_t>(s) * blocksPerRow);
        }

        // Tile rows so the Q8_K x data is reused across the tile rows.
        constexpr uint32_t kTileRows = 8;
        uint32_t numRowTiles = (rows + kTileRows - 1) / kTileRows;

        ThreadPool::instance().parallelFor(0, numRowTiles, [&](uint32_t tile) {
            uint32_t rowStart = tile * kTileRows;
            uint32_t tRows = std::min(kTileRows, rows - rowStart);
            // P7: thread-local reusable accumulator (grows on demand, zeroed
            // per tile) — no per-tile heap allocation.
            static thread_local std::vector<float> acc;
            acc.assign(static_cast<size_t>(tRows) * seqLen, 0.0f);

            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                for (uint32_t r = 0; r < tRows; ++r) {
                    const Q8KBlock &wrow = W_q8k[static_cast<size_t>(rowStart + r) * blocksPerRow + b];
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        const Q8KBlock &q8 = q8All[static_cast<size_t>(s) * blocksPerRow + b];
                        acc[static_cast<size_t>(r) * seqLen + s] +=
                                dotProductQ8K_Q8K_SIMD(&q8, &wrow);
                    }
                }
            }
            for (uint32_t r = 0; r < tRows; ++r) {
                for (uint32_t s = 0; s < seqLen; ++s) {
                    out[static_cast<size_t>(s) * rows + rowStart + r] =
                            acc[static_cast<size_t>(r) * seqLen + s];
                }
            }
        });
    }

    /// @brief Batched quantized matrix-matrix multiply.
    ///
    /// Computes out[s*rows + j] = sum_i X[s*cols + i] * W[j*cols + i] for
    /// s in [0, seqLen), j in [0, rows). X is [seqLen, cols], W is [rows, cols]
    /// (quantized), out is [seqLen, rows] (row-major).
    ///
    /// Parallelizes over output rows, reusing each weight row across all tokens
    /// (the prefill GEMM optimization). Uses the pre-packed Q2_K + Q8_K kernel
    /// when available, otherwise the general fused quantized dot product.
    void matMulVecBatchQuantized(const QuantizedMatrix &W, const float *X,
                                 uint32_t seqLen, float *out) {
        uint32_t rows = W.rows;
        uint32_t cols = W.cols;
        if (seqLen == 0 || rows == 0 || cols == 0) {
            return;
        }
        ScopedProfile sp("matMulVecBatchQuantized");

        // Q2_K pre-packed fast path: quantize each token's x to Q8_K once,
        // then use the int8 SIMD kernel for every (row, token) pair.
        if (W.type == GGML_TYPE_Q2_K && !W.prepackedData.empty()) {
            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t PREPACKED_BLOCK_BYTES = 276;
            uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            uint64_t rowStrideBytes =
                    static_cast<uint64_t>(blocksPerRow) * PREPACKED_BLOCK_BYTES;

            // P7: reusable grow-only scratch (see matMulVecBatchQ8K note —
            // SHARED buffer, written by the calling thread before the
            // parallelFor, read by workers).
            static std::vector<Q8KBlock> q8All;
            q8All.resize(static_cast<size_t>(seqLen) * blocksPerRow);
            for (uint32_t s = 0; s < seqLen; ++s) {
                GGMLDequantize::quantizeQ8K(X + static_cast<size_t>(s) * cols, cols,
                                            q8All.data() + static_cast<size_t>(s) * blocksPerRow);
            }

            ThreadPool::instance().parallelFor(0, rows, [&](uint32_t j) {
                const uint8_t *rowData = W.prepackedData.data() + static_cast<size_t>(j) * rowStrideBytes;
                if (j + 1 < rows) {
                    prefetchRow(rowData + rowStrideBytes);
                }
                for (uint32_t s = 0; s < seqLen; ++s) {
                    const Q8KBlock *q8 = q8All.data() + static_cast<size_t>(s) * blocksPerRow;
                    const float *xrow = X + static_cast<size_t>(s) * cols;
                    double dot = 0.0;
                    for (uint32_t b = 0; b < blocksPerRow; ++b) {
                        const uint8_t *blockData = rowData + static_cast<size_t>(b) * PREPACKED_BLOCK_BYTES;
                        uint32_t start = b * BLOCK_SIZE;
                        uint32_t n = std::min(BLOCK_SIZE, cols - start);
                        if (n < BLOCK_SIZE) {
                            float blockOut[256];
                            GGMLDequantize::dequantizeQ2_K_PrePackedBlock(blockData, blockOut, n);
                            for (uint32_t i = 0; i < n; ++i) {
                                dot += static_cast<double>(xrow[start + i]) * blockOut[i];
                            }
                        } else {
                            dot += static_cast<double>(dotProductQ2_K_PrePacked_Q8_SIMD(blockData, &q8[b]));
                        }
                    }
                    out[static_cast<size_t>(s) * rows + j] = static_cast<float>(dot);
                }
            });
            return;
        }

        uint32_t blockSize = ggmlBlockSize(W.type);
        uint32_t typeSize = ggmlTypeSize(W.type);

        if (blockSize == 0 || typeSize == 0) {
            // F32 (or unknown) fallback: direct float dot product.
            const float *Wf = reinterpret_cast<const float *>(W.data.data());
            ThreadPool::instance().parallelFor(0, rows, [&](uint32_t j) {
                const float *wrow = Wf + static_cast<size_t>(j) * cols;
                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *xrow = X + static_cast<size_t>(s) * cols;
                    double dot = 0.0;
                    for (uint32_t i = 0; i < cols; ++i) {
                        dot += static_cast<double>(xrow[i]) * wrow[i];
                    }
                    out[static_cast<size_t>(s) * rows + j] = static_cast<float>(dot);
                }
            });
            return;
        }

        uint32_t blocksPerRow = (cols + blockSize - 1) / blockSize;
        uint64_t rowStrideBytes = static_cast<uint64_t>(blocksPerRow) * typeSize;

        ThreadPool::instance().parallelFor(0, rows, [&](uint32_t j) {
            const uint8_t *rowData = W.data.data() + static_cast<size_t>(j) * rowStrideBytes;
            if (j + 1 < rows) {
                prefetchRow(rowData + rowStrideBytes);
            }
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *xrow = X + static_cast<size_t>(s) * cols;
                double dot = 0.0;
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    const uint8_t *blockData = rowData + static_cast<size_t>(b) * typeSize;
                    uint32_t start = b * blockSize;
                    uint32_t n = std::min(blockSize, cols - start);
                    dot += static_cast<double>(GGMLDequantize::dotProductFused(W.type, blockData, xrow + start, n));
                }
                out[static_cast<size_t>(s) * rows + j] = static_cast<float>(dot);
            }
        });
    }

    /// @brief Batched fused gate+up matrix-matrix multiply.
    ///
    /// Computes gateOut[s*rows + j] = sum_i X[s*cols + i] * gate[j*cols + i] and
    /// upOut[s*rows + j] = sum_i X[s*cols + i] * up[j*cols + i] for s in
    /// [0, seqLen), j in [0, rows). Both matrices share the same dimensions and
    /// quantized type. Quantizes each token's x to Q8_K once and reuses it for
    /// both gate and up, halving the x-quantization cost.
    ///
    /// When applySwish is true, the SwiGLU activation
    /// gateOut[s*rows+j] = silu(gateOut[s*rows+j]) * upOut[s*rows+j] is fused
    /// into the epilogue (P5) for every dispatch route below, so the caller
    /// must NOT apply a separate activation pass afterwards.
    void matMulVecFusedGateUp_Batch(const QuantizedMatrix &gate,
                                    const QuantizedMatrix &up,
                                    const float *X, uint32_t seqLen,
                                    float *gateOut, float *upOut,
                                    bool applySwish) {
        uint32_t rows = gate.rows;
        uint32_t cols = gate.cols;
        if (seqLen == 0 || rows == 0 || cols == 0) {
            return;
        }
        ScopedProfile sp("matMulVecFusedGateUp_Batch");
        // Fused SwiGLU epilogue applied on the non-AVX2 fallback routes so
        // all dispatch paths are semantically identical to the fused kernel.
        auto applySwishIf = [&]() {
            if (!applySwish) {
                return;
            }
            for (uint32_t s = 0; s < seqLen; ++s) {
                swiGLUSIMD(gateOut + static_cast<size_t>(s) * rows,
                           upOut + static_cast<size_t>(s) * rows, rows);
            }
        };
        if (gate.type != up.type || gate.rows != up.rows || gate.cols != up.cols) {
            matMulVecBatchQuantized(gate, X, seqLen, gateOut);
            matMulVecBatchQuantized(up, X, seqLen, upOut);
            applySwishIf();
            return;
        }

        if (gate.type == GGML_TYPE_Q2_K && !gate.prepackedData.empty() &&
            !up.prepackedData.empty()) {
            // Register-tiled batch GEMM: for each tile of 8 rows, the weight
            // blocks are loaded once and reused across all seqLen tokens,
            // eliminating the seqLen-fold re-read of the weight matrix that
            // dominates prefill.
            if (cols % 256 == 0) {
                matMulVecBatchGateUpQ2_K_PrePacked_Q8_Batch_SIMD(
                        gate.prepackedData.data(), up.prepackedData.data(),
                        X, seqLen, rows, cols, gateOut, upOut, applySwish);
                return;
            }

            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t PREPACKED_BLOCK_BYTES = 276;
            uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            uint64_t rowStrideBytes =
                    static_cast<uint64_t>(blocksPerRow) * PREPACKED_BLOCK_BYTES;

            // P7: reusable grow-only scratch (SHARED buffer, written by the
            // calling thread before the parallelFor, read by workers).
            static std::vector<Q8KBlock> q8All;
            q8All.resize(static_cast<size_t>(seqLen) * blocksPerRow);
            for (uint32_t s = 0; s < seqLen; ++s) {
                GGMLDequantize::quantizeQ8K(X + static_cast<size_t>(s) * cols, cols,
                                            q8All.data() + static_cast<size_t>(s) * blocksPerRow);
            }

            ThreadPool::instance().parallelFor(0, rows, [&](uint32_t j) {
                const uint8_t *gateRow = gate.prepackedData.data() + static_cast<size_t>(j) * rowStrideBytes;
                const uint8_t *upRow = up.prepackedData.data() + static_cast<size_t>(j) * rowStrideBytes;
                if (j + 1 < rows) {
                    prefetchRow(gateRow + rowStrideBytes);
                    prefetchRow(upRow + rowStrideBytes);
                }
                for (uint32_t s = 0; s < seqLen; ++s) {
                    const Q8KBlock *q8 = q8All.data() + static_cast<size_t>(s) * blocksPerRow;
                    const float *xrow = X + static_cast<size_t>(s) * cols;
                    double gateDot = 0.0;
                    double upDot = 0.0;
                    for (uint32_t b = 0; b < blocksPerRow; ++b) {
                        const uint8_t *gateBlock = gateRow + static_cast<size_t>(b) * PREPACKED_BLOCK_BYTES;
                        const uint8_t *upBlock = upRow + static_cast<size_t>(b) * PREPACKED_BLOCK_BYTES;
                        uint32_t start = b * BLOCK_SIZE;
                        uint32_t n = std::min(BLOCK_SIZE, cols - start);
                        if (n < BLOCK_SIZE) {
                            float blockOut[256];
                            GGMLDequantize::dequantizeQ2_K_PrePackedBlock(gateBlock, blockOut, n);
                            for (uint32_t i = 0; i < n; ++i) {
                                gateDot += static_cast<double>(xrow[start + i]) * blockOut[i];
                            }
                            GGMLDequantize::dequantizeQ2_K_PrePackedBlock(upBlock, blockOut, n);
                            for (uint32_t i = 0; i < n; ++i) {
                                upDot += static_cast<double>(xrow[start + i]) * blockOut[i];
                            }
                        } else {
                            gateDot += static_cast<double>(dotProductQ2_K_PrePacked_Q8_SIMD(gateBlock, &q8[b]));
                            upDot += static_cast<double>(dotProductQ2_K_PrePacked_Q8_SIMD(upBlock, &q8[b]));
                        }
                    }
                    gateOut[static_cast<size_t>(s) * rows + j] = static_cast<float>(gateDot);
                    upOut[static_cast<size_t>(s) * rows + j] = static_cast<float>(upDot);
                }
            });
            return;
        }

        // General fallback: two separate batched matmuls.
        matMulVecBatchQuantized(gate, X, seqLen, gateOut);
        matMulVecBatchQuantized(up, X, seqLen, upOut);
        applySwishIf();
    }

}// namespace tinycoder::detail