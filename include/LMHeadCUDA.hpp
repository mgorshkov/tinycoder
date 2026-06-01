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

#ifdef USE_CUDA

namespace tinycoder {
    namespace cuda {

        /// @brief Upload the dequantized embedding matrix to GPU (persistent storage).
        /// @param embedData Host-side dequantized embedding matrix data
        /// @param numElements Number of float elements in the matrix
        /// @return Device pointer to the embedding matrix on GPU
        float *uploadEmbeddings(const float *embedData, uint64_t numElements);

        /// @brief Compute LM head logits using cublasSgemv.
        /// @param d_embeddings  Device pointer to embedding matrix (vocabSize ×
        /// hiddenSize, row-major)
        /// @param hiddenSize    Hidden dimension size
        /// @param vocabSize     Vocabulary size
        /// @param h_hidden      Host-side hidden state vector (size hiddenSize)
        /// @param h_logits      Host-side output logits vector (size vocabSize,
        /// pre-allocated)
        void computeLMHead(const float *d_embeddings, uint32_t hiddenSize,
                           uint32_t vocabSize, const float *h_hidden, float *h_logits);

        /// @brief Free persistent GPU resources.
        void cleanupLMHead();

    }// namespace cuda
}// namespace tinycoder

#endif// USE_CUDA
