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
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <np/Array.hpp>

#include "GGMLDequantize.hpp"
#include "GGUFLoader.hpp"
#include "ModelConfig.hpp"
#include "Tokenizer.hpp"

namespace tinycoder {

    /// @brief Stores a quantized weight matrix and provides quantized matrix-vector
    /// multiplication. Weights are kept in their native GGML quantization format
    /// and dequantized on-the-fly during computation, saving significant memory
    /// compared to storing all weights as F32.
    struct QuantizedMatrix {
        std::vector<uint8_t> data;// Raw quantized data
        uint32_t rows = 0;        // Number of rows (output dim)
        uint32_t cols = 0;        // Number of columns (input dim)
        uint32_t type = 0;        // GGML_TYPE_* enum

        bool empty() const { return data.empty(); }

        /// @brief Compute y = x * W^T where W is this quantized matrix.
        /// Dequantizes the matrix and uses np::Array::dot() for multiplication.
        /// @param x Input vector of size rows
        /// @return Output vector of size cols as np::Array<float>
        np::Array<float> matMulVec(const float *x) const;
    };

    /// @brief Qwen2.5-Coder transformer model.
    ///
    /// Weights are stored in their native quantized format and dequantized
    /// on-the-fly during matrix-vector multiplication. This keeps memory
    /// usage close to the compressed file size (~637 MB) rather than the
    /// F32 dequantized size (~5.2 GB).
    ///
    /// Architecture: Qwen2.5-Coder (decoder-only transformer)
    /// - RoPE (Rotary Position Embedding)
    /// - GQA (Grouped Query Attention)
    /// - SwiGLU activation in FFN
    /// - Pre-norm with RMSNorm
    /// - Bias-free linear layers
    class Model {
    public:
        // ---- Public type aliases for debug access ----

        /// @brief Token embeddings stored quantized. Dequantized on-the-fly.
        struct QuantizedEmbedding {
            std::vector<uint8_t> data;// Raw quantized data
            uint32_t vocabSize;
            uint32_t hiddenSize;
            uint32_t type;// GGML_TYPE_* enum

            bool empty() const { return data.empty(); }

            /// @brief Dequantize a single token embedding row.
            std::vector<float> getRow(uint32_t tokenId) const;

            /// @brief Compute dot product of a vector with a single embedding row
            /// (dequantized on-the-fly).
            float dotRow(const float *vec, uint32_t tokenId) const;
        };

        /// @brief Pre-dequantized embedding matrix for fast LM head computation.
        /// Stores the full float32 embedding matrix (vocabSize × hiddenSize) which
        /// eliminates per-token dequantization overhead.
        struct DequantizedEmbedding {
            std::vector<float> data;// Flattened [vocabSize * hiddenSize] in row-major order
            uint32_t vocabSize = 0;
            uint32_t hiddenSize = 0;

            bool empty() const { return data.empty(); }
        };

        /// @brief Per-layer weights (stored in native quantized format).
        struct LayerWeights {
            QuantizedMatrix attnQ;// hiddenSize x (nHeads * headDim)
            QuantizedMatrix attnK;// hiddenSize x (nKVHeads * headDim)
            QuantizedMatrix attnV;// hiddenSize x (nKVHeads * headDim)
            QuantizedMatrix attnO;// (nHeads * headDim) x hiddenSize (quantized)

            np::Array<float> attnO_deq;// (nHeads * headDim) x hiddenSize (F32)

            // FFN weights (SwiGLU)
            QuantizedMatrix ffnGate;// hiddenSize x intermediateSize
            QuantizedMatrix ffnUp;  // hiddenSize x intermediateSize
            QuantizedMatrix ffnDown;// intermediateSize x hiddenSize

            // Dequantized F32 copy of ffnDown for exact float dot product matching
            // the reference. ffnDown is Q3_K (type 11) in the Q2_K model, same as
            // attnO. The block-level fused dequantize-dot in matMulVecFused produces
            // slightly different results than full dequantize + float dot product.
            np::Array<float> ffnDown_deq;// intermediateSize x hiddenSize (F32)

            // RMSNorm (F32 — tiny, just hiddenSize elements)
            np::Array<float> rmsNormAttn;// hiddenSize
            np::Array<float> rmsNormFFN; // hiddenSize

            // Q, K, V biases (F32)
            np::Array<float> attnQBias;// nHeads * headDim
            np::Array<float> attnKBias;// nKVHeads * headDim
            np::Array<float> attnVBias;// nKVHeads * headDim
        };

        /// @brief Progress callback type for model loading.
        /// @param progress Progress from 0.0 to 1.0
        /// @param stage Description of the current loading stage
        using ProgressCallback = std::function<void(float progress, const std::string &stage)>;

        Model();
        ~Model();

        /// @brief Load model from a GGUF file.
        /// @param modelPath Path to the GGUF model file
        /// @param outError Optional pointer to a string that will receive a detailed
        /// error message on failure
        /// @param progressCb Optional progress callback
        /// @return true if loading succeeded
        bool load(const std::string &modelPath, std::string *outError = nullptr,
                  ProgressCallback progressCb = nullptr);

        /// @brief Check if model is loaded and ready.
        bool isLoaded() const { return loaded_; }

        /// @brief Get the model configuration.
        const ModelConfig &config() const { return config_; }

        /// @brief Get the tokenizer.
        Tokenizer &tokenizer() { return tokenizer_; }
        const Tokenizer &tokenizer() const { return tokenizer_; }

        /// @brief Run a forward pass through the transformer.
        /// @param tokens Input token IDs (batch_size x seq_len)
        /// @return Logits tensor (batch_size x seq_len x vocab_size)
        np::Array<float> forward(const std::vector<int32_t> &tokens);

        /// @brief Generate text token by token.
        /// @param prompt Input prompt text
        /// @param params Generation parameters
        /// @param callback Called for each generated token with (token_id,
        /// token_text)
        /// @return Generated token IDs
        std::vector<int32_t>
        generate(const std::string &prompt, const InferenceParams &params,
                 std::function<bool(int32_t, const std::string &)> callback);

        /// @brief Get the KV cache size (number of cached tokens).
        size_t kvCacheSize() const { return kvCache_.pos; }

        /// @brief Clear the KV cache.
        void clearKVCache();

        /// @brief Debug: get dequantized embedding for a token.
        std::vector<float> debugGetEmbedding(int32_t tokenId) const {
            if (tokenId >= 0 &&
                tokenId < static_cast<int32_t>(quantizedEmbeddings_.vocabSize)) {
                return quantizedEmbeddings_.getRow(tokenId);
            }
            return {};
        }

        /// @brief Debug: get const reference to layer weights.
        const std::vector<LayerWeights> &debugGetLayers() const { return layers_; }

        /// @brief Debug: get const reference to quantized embeddings.
        const QuantizedEmbedding &debugGetEmbeddings() const {
            return quantizedEmbeddings_;
        }

        /// @brief Debug: run full forward pass with per-layer intermediate state
        /// dumps. Prints token embedding, after each layer's attention+residual,
        /// after each layer's FFN+residual, after final norm, and final logits.
        /// @param tokenId Input token ID
        /// @return Logits vector (size vocabSize)
        std::vector<float> debugForwardWithDumps(int32_t tokenId);

        /// @brief Debug: run forward pass and return both the final hidden state
        /// (after all layers + final RMSNorm) and the logits.
        /// @param tokenId Input token ID
        /// @return Pair of (hidden_state, logits), each a vector of floats
        std::pair<std::vector<float>, std::vector<float>>
        debugForwardWithHidden(int32_t tokenId);

        /// @brief Debug: run multi-token forward pass and return both the final
        /// hidden state (after all layers + final RMSNorm) and the logits for
        /// each token position.
        /// @param tokens Input token IDs
        /// @return Pair of (hidden_states, logits), each a vector of floats.
        ///         hidden_states has size seqLen * hiddenSize
        ///         logits has size seqLen * vocabSize
        std::pair<std::vector<float>, std::vector<float>>
        forwardWithHidden(const std::vector<int32_t> &tokens);

        /// @brief Debug: run multi-token forward pass and return only the final
        /// hidden state (after all layers + final RMSNorm), WITHOUT computing the
        /// expensive LM head. This is useful for comparing hidden states with
        /// @param tokens Input token IDs
        /// @return Hidden state vector of size seqLen * hiddenSize
        std::vector<float> forwardHiddenOnly(const std::vector<int32_t> &tokens);

        /// @brief Debug: run multi-token forward pass and return per-layer hidden
        /// states (after each layer's FFN residual, before final RMSNorm).
        /// @param tokens Input token IDs
        /// @return Vector of per-layer hidden states, each of size seqLen * hiddenSize.
        ///         Index 0 = after layer 0 FFN residual, ..., index nLayers-1 = after last layer FFN residual.
        std::vector<std::vector<float>>
        forwardWithPerLayerStates(const std::vector<int32_t> &tokens);

        /// @brief Debug: run multi-token forward pass token-by-token. Processes tokens one at a time with KV
        /// cache, instead of batched prefill with causal masking. This isolates
        /// whether batched prefill is the source of divergence from the reference.
        /// @param tokens Input token IDs
        /// @return Pair of (per-layer hidden states for last token, logits for last token).
        ///         per-layer hidden states: vector of nLayers vectors, each of size hiddenSize.
        ///         logits: vector of size vocabSize.
        std::pair<std::vector<std::vector<float>>, std::vector<float>>
        forwardTokenByToken(const std::vector<int32_t> &tokens);

        /// @brief Estimate memory required to load the model from a GGUF file.
        /// @param modelPath Path to the GGUF file
        /// @return Estimated memory in bytes, or 0 if file cannot be read
        static uint64_t estimateMemory(const std::string &modelPath);

        /// @brief Sample next token from logits.
        int32_t sampleToken(const np::Array<float> &logits,
                            const InferenceParams &params);

        /// @brief Apply temperature and top-p (nucleus) filtering.
        np::Array<float> applySamplingParams(const np::Array<float> &logits,
                                             const InferenceParams &params);

        /// @brief Build prefill tokens from prompt.
        std::vector<int32_t> tokenize(const std::string &prompt);

    private:
        // ---- Model weights (stored in native quantized format) ----

        QuantizedEmbedding quantizedEmbeddings_;
        std::vector<LayerWeights> layers_;

        // Final norm (F32)
        np::Array<float> finalNorm_;// hiddenSize

        // LM head (quantized, or tied with embeddings)
        QuantizedMatrix lmHead_;
        bool lmHeadTied_ = true;

        // Pre-dequantized embedding matrix for fast LM head (populated during load)
        DequantizedEmbedding dequantizedEmbeddings_;

        // ---- Runtime state ----
        bool loaded_ = false;
        ModelConfig config_;
        Tokenizer tokenizer_;

        // Persistent 32-bit RNG for sampling (mt19937)
        std::mt19937 rng_;
        uint32_t rngSeed_ = 0;
        bool rngInitialized_ = false;

        // KV cache: [numLayers][2][maxSeqLen x numKVHeads x headDim]
        struct KVCache {
            np::Array<float> k;// numLayers x maxSeqLen x numKVHeads x headDim
            np::Array<float> v;// numLayers x maxSeqLen x numKVHeads x headDim
            size_t pos = 0;    // Current write position
        };
        KVCache kvCache_;

        // ---- Forward pass components ----

        /// @brief Apply RMSNorm (optimized with direct pointer access).
        void rmsNormInPlace(const float *x, float *out, const float *weight,
                            uint32_t n, float eps = 1e-6f);

        /// @brief Apply RoPE (Rotary Position Embedding) with precomputed
        /// frequencies.
        void applyRoPE(float *q, float *k, uint32_t qSeqLen, uint32_t kSeqLen,
                       uint32_t qHeads, uint32_t kHeads, uint32_t pos);

        /// @brief Compute attention scores and output (optimized, fused).
        /// @param cachePos The starting position in the cache for the current batch
        ///        (used for causal masking: query token s attends to positions
        ///        0..cachePos+s).
        void attentionFused(const float *q, const float *kCache, const float *vCache,
                            float *output, uint32_t seqLen, uint32_t cachePos,
                            uint32_t cacheLen, uint32_t layerIdx);

        /// @brief SwiGLU activation: silu(x) * y (in-place on x).
        void swiGLUInPlace(float *x, const float *y, uint32_t n);

        /// @brief SiLU (Sigmoid Linear Unit) activation in-place.
        void siluInPlace(float *x, uint32_t n);

        /// @brief Load weights from GGUF (stores in native quantized format).
        bool loadWeights(GGUFLoader &loader);

        /// @brief Initialize KV cache.
        bool initKVCache();

        /// @brief Build pre-dequantized embedding matrix for fast LM head.
        /// Called after loadWeights() to avoid per-token dequantization overhead.
        bool buildDequantizedEmbeddings();

        // Make these methods public for unit tests
        friend class ParisQuestionTest;
        friend class DebugForwardTest;
    };

}// namespace tinycoder
