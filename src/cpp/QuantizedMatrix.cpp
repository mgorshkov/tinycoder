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

    /// @brief Compute y = x * W where W is (rows x cols) = (out_features x
    /// in_features), stored row-major in GGUF format.
    ///        x has size cols (in_features), result has size rows (out_features).
    ///
    ///        GGUF stores weight matrices as (out_features x in_features)
    ///        in row-major order, so W[j][i] = data[j * cols + i] where j indexes
    ///        output features and i indexes input features.
    ///        The computation is: y_j = sum_i x[i] * W[j][i] = sum_i x[i] * data[j * cols + i]
    static np::Array<float> matMulVecCUDA(const float *x, const float *W,
                                          uint32_t rows, uint32_t cols) {
#ifdef USE_CUDA
        // Only use CUDA for matrices large enough to justify GPU overhead
        constexpr uint64_t CUDA_MIN_ELEMENTS = 256 * 256;// ~256KB

        uint64_t numElements = static_cast<uint64_t>(rows) * cols;
        if (numElements >= CUDA_MIN_ELEMENTS) {
            try {
                np::Array<float> result(np::Shape{rows});
                float *resultData = result.data();

                // Use np library's CUDA-accelerated dot1d2d
                // dot1d2d computes y_j = sum_i x_i * W[j][i] (y = x * W^T)
                // where W is (rows x cols). Our matrix is stored as
                // (rows x cols) = (out_features x in_features) in GGUF row-major.
                np::internal::cuda::dot1d2d(x, W, rows, cols, resultData);

                return result;
            } catch (const std::exception &e) {
                // CUDA failed (no GPU, no driver, OOM, etc.) — fall back to CPU
                // Log the error for debugging
                std::fprintf(stderr, "CUDA dot1d2d failed, falling back to CPU: %s\n",
                             e.what());
            }
        }
#endif
        // Fallback to CPU: compute y = x * W
        // W is stored as (rows x cols) row-major: W[j][i] = data[j * cols + i]
        // y_j = sum_i x[i] * W[j][i] for j in [0, rows), i in [0, cols)
        np::Array<float> result(np::Shape{rows});
        float *resultData = result.data();
        for (uint32_t j = 0; j < rows; ++j) {
            double dot = 0.0;
            for (uint32_t i = 0; i < cols; ++i) {
                dot += static_cast<double>(x[i]) * W[static_cast<size_t>(j) * cols + i];
            }
            resultData[j] = static_cast<float>(dot);
        }
        return result;
    }

    np::Array<float> QuantizedMatrix::matMulVec(const float *x) const {
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

        // DIAGNOSTIC: Print parameters for Q/K/V calls for all tokens
        static int callCount = 0;
        ++callCount;
        // Print for ALL calls to see if any specific call produces wrong results
        // Q/K/V for 28 tokens = 84 calls. Print all.
        if (callCount <= 90) {
            std::cout << "[DIAG_MATMUL] call=" << callCount
                      << " type=" << type << " rows=" << rows << " cols=" << cols
                      << " data.size=" << data.size()
                      << " data.ptr=" << (void *) data.data()
                      << " x[0]=" << x[0] << " x[1]=" << x[1] << std::endl;
        }

        // For F32 type, use the CUDA/CPU path (no dequantization needed)
        if (type == GGML_TYPE_F32) {
            const float *W_f32 = reinterpret_cast<const float *>(data.data());
            return matMulVecCUDA(x, W_f32, rows, cols);
        }

        // For quantized types, use block-level fused dequantize-dot
        // matMulVecFused computes y_j = sum_i x[i] * W[j][i]
        // where x has size cols, result has size rows
        np::Array<float> result(np::Shape{rows});
        float *resultData = result.data();

        GGMLDequantize::matMulVecFused(type, data.data(), x, rows, cols, resultData);

        // DIAGNOSTIC: Also compute using a separate buffer and compare
        if (callCount <= 90) {
            std::vector<float> diagResult(rows);
            GGMLDequantize::matMulVecFused(type, data.data(), x, rows, cols, diagResult.data());
            float rms1 = 0.0f, rms2 = 0.0f;
            for (uint32_t i = 0; i < rows; ++i) {
                rms1 += resultData[i] * resultData[i];
                rms2 += diagResult[i] * diagResult[i];
            }
            rms1 = std::sqrt(rms1 / rows);
            rms2 = std::sqrt(rms2 / rows);
            bool match = true;
            for (uint32_t i = 0; i < std::min(rows, 8u); ++i) {
                if (std::abs(resultData[i] - diagResult[i]) > 1e-5f) match = false;
            }
            std::cout << "[DIAG_MATMUL] call=" << callCount
                      << " result rms=" << rms1 << " diag rms=" << rms2
                      << " match=" << (match ? "YES" : "NO")
                      << " result[0]=" << resultData[0]
                      << " diag[0]=" << diagResult[0] << std::endl;
            if (!match) {
                std::cout << "[DIAG_MATMUL] call=" << callCount << " MISMATCH! result first 8: ";
                for (uint32_t i = 0; i < 8; ++i) std::cout << resultData[i] << " ";
                std::cout << std::endl;
                std::cout << "[DIAG_MATMUL] call=" << callCount << " MISMATCH! diag first 8: ";
                for (uint32_t i = 0; i < 8; ++i) std::cout << diagResult[i] << " ";
                std::cout << std::endl;
            }
        }

        return result;
    }

}// namespace tinycoder
