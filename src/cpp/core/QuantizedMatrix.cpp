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
#include "Model.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

#include <np/Array.hpp>

#ifdef USE_CUDA
#include <np/internal/cuda/Dot1d2d.hpp>
#endif

namespace tinycoder {

    np::Array<float> QuantizedMatrix::matMulVec(const float *x) const {
        // Allocate result and delegate to the out-parameter version
        np::Array<float> result(np::Shape{rows});
        matMulVec(x, result.data());
        return result;
    }

    void QuantizedMatrix::matMulVec(const float *x, float *out) const {
        // Compute y = x * W where W is this quantized matrix.
        // W has dimensions (rows x cols) = (out_features x in_features), stored
        // row-major in GGUF format. x is a row vector of size cols (in_features).
        // y is a row vector of size rows (out_features).
        //
        // GGUF stores weight matrices as (out_features x in_features) row-major:
        //   W[j][i] = data[j * cols + i]
        // where j indexes output features and i indexes input features.
        //
        // The computation is: y_j = sum_i x[i] * W[j][i]  for j in [0, rows), i in
        // [0, cols)
        //
        // OPTIMIZATION: Instead of dequantizing the entire matrix to float and then
        // doing the dot product, we use a block-level fused dequantize-dot approach:
        // for each output row, we dequantize one block at a time and compute the
        // dot product directly. This avoids:
        //   1. Allocating a large float buffer for the full dequantized matrix
        //      (e.g., 13.76M elements = ~55 MB for ffnGate)
        //   2. The memory bandwidth bottleneck of writing/reading all those floats
        //   3. The overhead of copying into np::Array for the dot product
        //
        // For F32 matrices, we still use the CUDA/CPU path.

        // For F32 type, use the CUDA/CPU path (no dequantization needed)
        if (type == GGML_TYPE_F32) {
            const float *W_f32 = reinterpret_cast<const float *>(data.data());
            // For F32, compute directly into out (avoiding the CUDA path's allocation)
            for (uint32_t j = 0; j < rows; ++j) {
                double dot = 0.0;
                for (uint32_t i = 0; i < cols; ++i) {
                    dot += static_cast<double>(x[i]) * W_f32[static_cast<size_t>(j) * cols + i];
                }
                out[j] = static_cast<float>(dot);
            }
            return;
        }

        // For quantized types, use block-level fused dequantize-dot
        // matMulVecFused computes y_j = sum_i x[i] * W[j][i]
        // where x has size cols, result has size rows

        // OPTIMIZATION: Use pre-packed kernel for Q2_K matrices that have been
        // pre-packed at load time. The pre-packed format eliminates the 2-bit
        // extraction overhead in the SIMD kernel.
        if (type == GGML_TYPE_Q2_K && !prepackedData.empty()) {
            // quantize x to Q8_K (int8) once per matmul,
            // then use _mm256_maddubs_epi16 (32 int8×int8->int16 multiply-adds
            // per instruction) instead of float FMAs (8 per instruction).
            GGMLDequantize::matMulVecFusedQ2_K_PrePacked_Q8(prepackedData.data(), x, rows, cols, out);
            return;
        }

        // Use cache-blocked version for large matrices (rows > 256) to keep
        // the x vector in L1 cache. For small matrices, the non-blocked version
        // is fine and avoids the memset overhead.
        if (rows > 256) {
            GGMLDequantize::matMulVecFusedCacheBlocked(type, data.data(), x, rows, cols, out);
        } else {
            GGMLDequantize::matMulVecFused(type, data.data(), x, rows, cols, out);
        }
    }

    np::Array<float> QuantizedMatrix::matMulVecRows(const float *x, uint32_t rowStart, uint32_t numRows) const {
        // Allocate result and delegate to the out-parameter version
        np::Array<float> result(np::Shape{numRows});
        matMulVecRows(x, rowStart, numRows, result.data());
        return result;
    }

    void QuantizedMatrix::matMulVecFusedGateUp(const QuantizedMatrix &other,
                                               const float *x, float *gateOut,
                                               float *upOut,
                                               bool applySwish) const {
        // Compute gate = x * this^T and up = x * other^T in a single pass over x.
        // Both matrices must share the same dimensions and quantized type.
        if (rows != other.rows || cols != other.cols || type != other.type) {
            // Fallback: two separate matmuls
            matMulVec(x, gateOut);
            other.matMulVec(x, upOut);
            return;
        }

        // Fast path: fused Q2_K gate+up kernel over the COMPACT (raw) blocks.
        // This fn is used for single-token generation, which is
        // DRAM-bandwidth-bound; the compact 84-byte Q2_K blocks (vs 276-byte
        // prepacked) cut the gate+up weight traffic ~3.3x — the dominant
        // per-token cost — matching llama.cpp's working set. The kernel unpacks
        // the 2-bit quants on the fly (verified index-identical to the prepacked
        // expansion) and computes both dot products per row, quantizing x to
        // Q8_K once.
        if (type == GGML_TYPE_Q2_K && !data.empty() && !other.data.empty()) {
            GGMLDequantize::matMulVecFusedGateUpQ2_K_Compact_Q8(
                    data.data(), other.data.data(), x, rows, cols,
                    gateOut, upOut, applySwish);
            return;
        }

        // Fallback fast path: pre-packed Q2_K gate+up fused kernel (same maths,
        // 3.3x larger weight read — used when the compact data is unavailable).
        if (type == GGML_TYPE_Q2_K && !prepackedData.empty() && !other.prepackedData.empty()) {
            GGMLDequantize::matMulVecFusedGateUpQ2_K_PrePacked_Q8(
                    prepackedData.data(), other.prepackedData.data(), x, rows, cols,
                    gateOut, upOut);
            return;
        }

        // General fallback: two separate matmuls
        matMulVec(x, gateOut);
        other.matMulVec(x, upOut);
    }

    void QuantizedMatrix::matMulVecRows(const float *x, uint32_t rowStart, uint32_t numRows, float *out) const {
        // Compute y = x * W for a contiguous range of rows [rowStart, rowStart + numRows).
        // W has dimensions (rows x cols) = (out_features x in_features), stored row-major.
        // This is used for expert sub-matrices in MoE architectures where multiple
        // experts are stored in a single QuantizedMatrix.
        //
        // The data layout is: expert 0 rows, expert 1 rows, ..., expert N-1 rows.
        // Each expert has expertFF rows (for gate/up) or hiddenSize rows (for down).
        //
        // We compute y_j = sum_i x[i] * W[rowStart + j][i] for j in [0, numRows)

        if (rowStart + numRows > rows) {
            std::cerr << "[TinyCoder] matMulVecRows: rowStart=" << rowStart
                      << " numRows=" << numRows << " exceeds rows=" << rows << std::endl;
            std::memset(out, 0, static_cast<size_t>(numRows) * sizeof(float));
            return;
        }

        if (type == GGML_TYPE_F32) {
            const float *W_f32 = reinterpret_cast<const float *>(data.data());
            const float *W_start = W_f32 + static_cast<size_t>(rowStart) * cols;
            for (uint32_t j = 0; j < numRows; ++j) {
                double dot = 0.0;
                for (uint32_t i = 0; i < cols; ++i) {
                    dot += static_cast<double>(x[i]) * W_start[static_cast<size_t>(j) * cols + i];
                }
                out[j] = static_cast<float>(dot);
            }
        } else {
            // For quantized types, compute row by row using the fused quantized dot product.
            // This eliminates the float blockBuf[256] temporary and the extra memory pass.
            uint32_t blockSize = ggmlBlockSize(type);
            uint32_t typeSize = ggmlTypeSize(type);
            uint64_t blocksPerRow = (static_cast<uint64_t>(cols) + blockSize - 1) / blockSize;
            uint64_t bytesPerRow = blocksPerRow * typeSize;
            const uint8_t *rowData = data.data() + static_cast<size_t>(rowStart) * bytesPerRow;

            for (uint32_t j = 0; j < numRows; ++j) {
                const uint8_t *rowPtr = rowData + j * bytesPerRow;
                double dot = 0.0;
                uint32_t remaining = cols;
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    uint32_t blockN = std::min(blockSize, remaining);
                    // Fused quantized dot product: dequantize and dot in one pass
                    dot += static_cast<double>(GGMLDequantize::dotProductFused(type, rowPtr + b * typeSize, x + b * blockSize, blockN));
                    remaining -= blockN;
                }
                out[j] = static_cast<float>(dot);
            }
        }
    }

}// namespace tinycoder
