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

/*
 * CUDA-accelerated LM head computation using cublasSgemv.
 *
 * The LM head is: logits = hidden × embeddings^T
 * where hidden is [1, hiddenSize] and embeddings is [vocabSize, hiddenSize].
 *
 * This is a single cublasSgemv call:
 *   logits[j] = sum_i hidden[i] * embeddings[j * hiddenSize + i]
 *
 * With cuBLAS, we compute y = alpha * A * x + beta * y where:
 *   A = embeddings (vocabSize × hiddenSize, row-major → CUBLAS_OP_T)
 *   x = hidden (hiddenSize × 1)
 *   y = logits (vocabSize × 1)
 *   alpha = 1.0, beta = 0.0
 *
 * The embedding matrix is kept persistently on GPU to avoid
 * re-uploading it for every token generation step.
 */

#ifdef USE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <vector>

namespace tinycoder {
    namespace cuda {

        // Singleton cuBLAS handle
        static cublasHandle_t g_cublasHandle = nullptr;
        static std::once_flag g_cublasInitFlag;

        static void initCublas() {
            cublasStatus_t stat = cublasCreate(&g_cublasHandle);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Failed to create cuBLAS handle");
            }
        }

        static cublasHandle_t getCublasHandle() {
            std::call_once(g_cublasInitFlag, initCublas);
            return g_cublasHandle;
        }

        /// @brief Persistent GPU storage for the dequantized embedding matrix.
        ///        Allocated once and reused across all forward passes.
        static float *g_embedDevPtr = nullptr;
        static uint64_t g_embedElements = 0;

        /// @brief Upload the dequantized embedding matrix to GPU (if not already
        /// there).
        /// @param embedData Host-side dequantized embedding matrix data
        /// @param numElements Number of float elements in the matrix
        /// @return Device pointer to the embedding matrix on GPU
        float *uploadEmbeddings(const float *embedData, uint64_t numElements) {
            if (g_embedDevPtr != nullptr && g_embedElements == numElements) {
                // Already uploaded with the same size — reuse
                return g_embedDevPtr;
            }

            // Free previous allocation if size changed
            if (g_embedDevPtr != nullptr) {
                cudaFree(g_embedDevPtr);
                g_embedDevPtr = nullptr;
            }

            cudaError_t err = cudaMalloc(&g_embedDevPtr, numElements * sizeof(float));
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate GPU memory for embeddings: " +
                                         std::string(cudaGetErrorString(err)));
            }

            err = cudaMemcpy(g_embedDevPtr, embedData, numElements * sizeof(float),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                cudaFree(g_embedDevPtr);
                g_embedDevPtr = nullptr;
                throw std::runtime_error("Failed to copy embeddings to GPU: " +
                                         std::string(cudaGetErrorString(err)));
            }

            g_embedElements = numElements;
            return g_embedDevPtr;
        }

        /// @brief Compute LM head logits using cublasSgemv.
        ///
        /// Computes: logits = hidden × embeddings^T
        /// where embeddings is (vocabSize × hiddenSize) stored row-major.
        ///
        /// cuBLAS expects column-major matrices. For a row-major matrix A of shape
        /// (vocabSize × hiddenSize), the column-major equivalent is A^T of shape
        /// (hiddenSize × vocabSize). So we compute:
        ///   logits^T = embeddings^T × hidden^T
        /// which is: y = alpha * op(A) * x + beta * y
        /// with op(A) = CUBLAS_OP_T (transpose), A = embeddings as column-major
        /// (which is row-major embeddings^T), x = hidden, y = logits.
        ///
        /// In cuBLAS terms for row-major A (vocabSize × hiddenSize):
        ///   cublasSgemv(handle, CUBLAS_OP_T,
        ///               hiddenSize, vocabSize,  // A is (hiddenSize × vocabSize) in
        ///               col-major &alpha, d_embeddings, hiddenSize,  // leading
        ///               dimension = hiddenSize d_hidden, 1,               // incx
        ///               &beta,
        ///               d_logits, 1)               // incy
        ///
        /// @param d_embeddings  Device pointer to embedding matrix (vocabSize ×
        /// hiddenSize, row-major)
        /// @param hiddenSize    Hidden dimension size
        /// @param vocabSize     Vocabulary size
        /// @param h_hidden      Host-side hidden state vector (size hiddenSize)
        /// @param h_logits      Host-side output logits vector (size vocabSize,
        /// pre-allocated)
        void computeLMHead(const float *d_embeddings, uint32_t hiddenSize,
                           uint32_t vocabSize, const float *h_hidden, float *h_logits) {
            cublasHandle_t handle = getCublasHandle();

            // Allocate GPU memory for hidden and logits
            float *d_hidden, *d_logits;
            cudaError_t err;

            err = cudaMalloc(&d_hidden, hiddenSize * sizeof(float));
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate GPU memory for hidden: " +
                                         std::string(cudaGetErrorString(err)));
            }

            err = cudaMalloc(&d_logits, vocabSize * sizeof(float));
            if (err != cudaSuccess) {
                cudaFree(d_hidden);
                throw std::runtime_error("Failed to allocate GPU memory for logits: " +
                                         std::string(cudaGetErrorString(err)));
            }

            // Copy hidden state to GPU
            err = cudaMemcpy(d_hidden, h_hidden, hiddenSize * sizeof(float),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                cudaFree(d_hidden);
                cudaFree(d_logits);
                throw std::runtime_error("Failed to copy hidden to GPU: " +
                                         std::string(cudaGetErrorString(err)));
            }

            // Compute logits = embeddings × hidden (matrix-vector multiply)
            // embeddings is (vocabSize × hiddenSize) stored row-major.
            // cuBLAS expects column-major, so we treat it as (hiddenSize × vocabSize)
            // column-major and use CUBLAS_OP_T to transpose.
            //
            // cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
            // computes y = alpha * op(A) * x + beta * y
            //
            // With CUBLAS_OP_T: op(A) = A^T, so A is (n × m) column-major
            // We want: logits[j] = sum_i hidden[i] * embeddings[j * hiddenSize + i]
            //
            // Set A = embeddings as column-major (hiddenSize × vocabSize)
            // op(A) = A^T = (vocabSize × hiddenSize) in row-major terms
            // x = hidden (hiddenSize × 1)
            // y = logits (vocabSize × 1)
            //
            // cublasSgemv(handle, CUBLAS_OP_T,
            //             hiddenSize, vocabSize,  // A is (hiddenSize × vocabSize)
            //             col-major &alpha, d_embeddings, hiddenSize,  // lda =
            //             hiddenSize d_hidden, 1,               // incx = 1 &beta,
            //             d_logits, 1)               // incy = 1

            float alpha = 1.0f;
            float beta = 0.0f;

            cublasStatus_t stat = cublasSgemv(
                    handle, CUBLAS_OP_T, static_cast<int>(hiddenSize),
                    static_cast<int>(vocabSize), &alpha, d_embeddings,
                    static_cast<int>(hiddenSize), d_hidden, 1, &beta, d_logits, 1);

            if (stat != CUBLAS_STATUS_SUCCESS) {
                cudaFree(d_hidden);
                cudaFree(d_logits);
                throw std::runtime_error("cublasSgemv failed for LM head");
            }

            // Copy result back to host
            err = cudaMemcpy(h_logits, d_logits, vocabSize * sizeof(float),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                cudaFree(d_hidden);
                cudaFree(d_logits);
                throw std::runtime_error("Failed to copy logits from GPU: " +
                                         std::string(cudaGetErrorString(err)));
            }

            cudaFree(d_hidden);
            cudaFree(d_logits);
        }

        /// @brief Free persistent GPU resources.
        void cleanupLMHead() {
            if (g_embedDevPtr != nullptr) {
                cudaFree(g_embedDevPtr);
                g_embedDevPtr = nullptr;
            }
            g_embedElements = 0;

            if (g_cublasHandle != nullptr) {
                cublasDestroy(g_cublasHandle);
                g_cublasHandle = nullptr;
            }
        }

    }// namespace cuda
}// namespace tinycoder

#endif// USE_CUDA
