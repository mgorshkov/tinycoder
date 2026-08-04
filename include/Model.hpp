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

#include "AlignedVector.hpp"
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
        AlignedVector<uint8_t> data;// Raw quantized data (64-byte aligned)
        uint32_t rows = 0;          // Number of rows (output dim)
        uint32_t cols = 0;          // Number of columns (input dim)
        uint32_t type = 0;          // GGML_TYPE_* enum

        // Pre-packed Q2_K data (expanded 2-bit values to bytes 0-3).
        // Only populated for ffnGate and ffnUp when type == GGML_TYPE_Q2_K.
        // Eliminates bit extraction overhead in the SIMD kernel.
        AlignedVector<uint8_t> prepackedData;

        bool empty() const { return data.empty(); }

        /// @brief Compute y = x * W^T where W is this quantized matrix.
        /// Dequantizes the matrix and uses np::Array::dot() for multiplication.
        /// @param x Input vector of size rows
        /// @return Output vector of size cols as np::Array<float>
        np::Array<float> matMulVec(const float *x) const;

        /// @brief Compute y = x * W^T where W is this quantized matrix.
        /// Out-parameter version that writes directly to a pre-allocated buffer,
        /// avoiding the heap allocation and memcpy of the return-value version.
        /// @param x Input vector of size cols
        /// @param out Output buffer of size rows (must be pre-allocated by caller)
        void matMulVec(const float *x, float *out) const;

        /// @brief Compute y = x * W^T for a contiguous range of rows.
        /// Used for expert sub-matrices in MoE architectures where multiple
        /// experts are stored in a single QuantizedMatrix.
        /// @param x Input vector of size cols
        /// @param rowStart Starting row index
        /// @param numRows Number of rows to compute
        /// @return Output vector of size numRows
        np::Array<float> matMulVecRows(const float *x, uint32_t rowStart, uint32_t numRows) const;

        /// @brief Compute y = x * W^T for a contiguous range of rows.
        /// Out-parameter version that writes directly to a pre-allocated buffer.
        /// @param x Input vector of size cols
        /// @param rowStart Starting row index
        /// @param numRows Number of rows to compute
        /// @param out Output buffer of size numRows (must be pre-allocated by caller)
        void matMulVecRows(const float *x, uint32_t rowStart, uint32_t numRows, float *out) const;

        /// @brief Fused gate+up matrix-vector multiply.
        ///
        /// Computes gate = x * this^T and up = x * other^T in a single pass over x.
        /// Both matrices must share the same dimensions (rows x cols) and the same
        /// quantized type. This reads x once instead of twice, halving the dominant
        /// memory traffic for the FFN gate+up projections.
        ///
        /// @param other The up matrix (same dims/type as this gate matrix)
        /// @param x     Input vector of size cols
        /// @param gateOut Output buffer for gate (size rows)
        /// @param upOut   Output buffer for up (size rows)
        /// @param applySwish When true, the SwiGLU activation
        ///                   gateOut[j] = silu(gateOut[j]) * upOut[j] is fused
        ///                   into the kernel epilogue (removes the separate
        ///                   swiGLUSIMD pass for single-token generation).
        void matMulVecFusedGateUp(const QuantizedMatrix &other, const float *x,
                                  float *gateOut, float *upOut,
                                  bool applySwish = false) const;
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
            std::vector<float> data;// Flattened [vocabSize * hiddenSize] in row-major order (F32)
            // FP16 copy of the same matrix. Halves memory bandwidth for the LM head
            // mat-vec (the largest single memory read per token). Since the source
            // embeddings are Q2_K (~2-bit), FP16 is lossless (see BENCHMARK_REPORT §7.2).
            std::vector<uint16_t> dataF16;// Flattened [vocabSize * hiddenSize] in row-major order (FP16)
            // Q8_K copy of the same matrix. Cuts memory bandwidth ~2× vs FP16 and
            // ~4× vs F32 for the LM head mat-vec. Each 256-element block stores 256
            // int8 values + a block scale (257 bytes vs 512 bytes for FP16). The
            // dot product uses _mm256_maddubs_epi16 int8 kernels.
            std::vector<Q8KBlock> dataQ8K;// Flattened [vocabSize * blocksPerRow] in row-major order (Q8_K)
            uint32_t vocabSize = 0;
            uint32_t hiddenSize = 0;

            bool empty() const { return data.empty(); }
        };

        /// @brief GPU adapter runtime state (opaque; defined in ModelGPU.cpp).
        struct ModelGPUState;

        /// @brief Run a forward pass on the GPU (see GPUCompute.hpp).
        /// @param logitsOut Host buffer: vocabSize floats when computeAllLogits is
        ///        false, else seqLen*vocabSize.
        bool gpuForward(const std::vector<int32_t> &tokens, bool computeAllLogits,
                        float *logitsOut, std::string *errMsg = nullptr);

        /// @brief Upload model weights to the GPU (default enabled in CUDA
        /// builds; $TINYCODER_GPU=0 opts out).
        /// @return true if the GPU path is now active (weights uploaded).
        bool gpuUploadIfEnabled(std::string *errMsg = nullptr);

        /// @brief Clear the GPU KV cache (called by clearKVCache()).
        void gpuClearKV();

        /// @brief Tear down GPU resources (called by ~Model).
        void gpuShutdown();

        /// @brief Per-layer weights (stored in native quantized format).
        struct LayerWeights {
            // ---- Common attention weights ----
            QuantizedMatrix attnQ;// hiddenSize x (nHeads * headDim)
            QuantizedMatrix attnK;// hiddenSize x (nKVHeads * headDim)
            QuantizedMatrix attnV;// hiddenSize x (nKVHeads * headDim)
            QuantizedMatrix attnO;// (nHeads * headDim) x hiddenSize (quantized)

            // Dequantized FP16 copy of attnO (halves memory bandwidth vs F32)
            // 64-byte aligned for cache-friendly SIMD access.
            AlignedVector<uint16_t> attnO_deq_f16;// (nHeads * headDim) x hiddenSize (FP16)
            // Q8_K prepacked copy of attnO (row-major [rows x blocksPerRow]
            // Q8KBlock arrays). Used by the prefill Q8_K batch GEMM (P4) for the
            // large attnO matmul, replacing the FP16 path with an int8
            // _mm256_maddubs_epi16 kernel (32 MACs/instr vs 8 for FP16).
            std::vector<Q8KBlock> attnO_q8k;// rows x blocksPerRow

            // ---- Common FFN weights (SwiGLU) ----
            QuantizedMatrix ffnGate;// hiddenSize x intermediateSize
            QuantizedMatrix ffnUp;  // hiddenSize x intermediateSize
            QuantizedMatrix ffnDown;// intermediateSize x hiddenSize

            // Dequantized FP16 copy of ffnDown (halves memory bandwidth vs F32)
            // 64-byte aligned for cache-friendly SIMD access.
            AlignedVector<uint16_t> ffnDown_deq_f16;// intermediateSize x hiddenSize (FP16)
            // Q8_K prepacked copy of ffnDown (row-major [rows x blocksPerRow]
            // Q8KBlock arrays). Used by the prefill Q8_K batch GEMM (P4) for the
            // largest matmul (ffnDown), replacing the FP16 path.
            std::vector<Q8KBlock> ffnDown_q8k;// rows x blocksPerRow

            // ---- Common RMSNorm (F32) ----
            np::Array<float> rmsNormAttn;// hiddenSize
            np::Array<float> rmsNormFFN; // hiddenSize

            // ---- Qwen2-specific: Q, K, V biases (F32) ----
            np::Array<float> attnQBias;// nHeads * headDim
            np::Array<float> attnKBias;// nKVHeads * headDim
            np::Array<float> attnVBias;// nKVHeads * headDim

            // ---- Gemma4-specific: Q/K norms, post norms, layer scale ----
            np::Array<float> attnQNorm;       // headDim (QK RMSNorm before RoPE)
            np::Array<float> attnKNorm;       // headDim (QK RMSNorm before RoPE)
            np::Array<float> postAttnNorm;    // hiddenSize (post-attention norm)
            np::Array<float> postFFWNorm;     // hiddenSize (post-FFN norm)
            np::Array<float> layerOutputScale;// 1 (per-layer scaling factor)

            // ---- Gemma4 MoE-specific: expert weights ----
            QuantizedMatrix ffnGateInp;   // hiddenSize x expertCount (router)
            QuantizedMatrix ffnGateUpExps;// expertCount x (expertFF * 2) x hiddenSize (fused gate+up)
            QuantizedMatrix ffnDownExps;  // expertCount x hiddenSize x expertFF
            np::Array<float> preFFWNorm2; // hiddenSize (second pre-FFN norm)
            np::Array<float> postFFWNorm1;// hiddenSize (first post-FFN norm)
            np::Array<float> postFFWNorm2;// hiddenSize (second post-FFN norm)

            // ---- Qwen35MoE-specific: attention gate, QKV fused ----
            QuantizedMatrix attnQKV;      // hiddenSize x (nHeads + 2*nKVHeads) * headDim (fused QKV)
            QuantizedMatrix attnGate;     // hiddenSize x (nHeads * headDim) (attention gate)
            np::Array<float> attnQNormMoe;// headDim (Q norm for Qwen35MoE)
            np::Array<float> attnKNormMoe;// headDim (K norm for Qwen35MoE)

            // ---- Qwen35MoE: SSM (Mamba-style) ----
            QuantizedMatrix ssmConv1d; // ssmInnerSize x ssmConvKernel
            QuantizedMatrix ssmOut;    // ssmInnerSize x hiddenSize
            np::Array<float> ssmA;     // ssmInnerSize x ssmStateSize (log)
            np::Array<float> ssmDtBias;// ssmInnerSize
            np::Array<float> ssmAlpha; // ssmInnerSize
            np::Array<float> ssmBeta;  // ssmInnerSize
            np::Array<float> ssmNorm;  // ssmInnerSize

            // ---- Qwen35MoE: MoE FFN ----
            QuantizedMatrix ffnGateInpMoe;  // hiddenSize x expertCount (router)
            QuantizedMatrix ffnGateExps;    // expertCount x expertFF x hiddenSize
            QuantizedMatrix ffnUpExps;      // expertCount x expertFF x hiddenSize
            QuantizedMatrix ffnDownExpsMoe; // expertCount x hiddenSize x expertFF
            QuantizedMatrix ffnGateShexp;   // hiddenSize x sharedExpertFF (shared expert gate)
            QuantizedMatrix ffnUpShexp;     // hiddenSize x sharedExpertFF (shared expert up)
            QuantizedMatrix ffnDownShexp;   // sharedExpertFF x hiddenSize (shared expert down)
            QuantizedMatrix ffnGateInpShexp;// hiddenSize x 1 (shared expert router)

            // ---- Qwen35MoE: MTP (Multi-Token Prediction) ----
            QuantizedMatrix nextnEhProj;         // hiddenSize x hiddenSize (embedding head projection)
            np::Array<float> nextnEnorm;         // hiddenSize (embedding head norm)
            np::Array<float> nextnHnorm;         // hiddenSize (hidden norm)
            np::Array<float> nextnSharedHeadNorm;// hiddenSize (shared head norm)
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
        /// @param computeAllLogits If true (default), compute the LM head for every
        ///        token. If false, only compute the LM head for the last token (the
        ///        other logit rows are left uninitialized). This is a prefill
        ///        optimization: the LM head reads the full vocabSize x hiddenSize
        ///        embedding matrix per token, so skipping it for all but the last
        ///        token avoids re-reading that large matrix seqLen times.
        /// @return Logits tensor (batch_size x seq_len x vocab_size)
        np::Array<float> forward(const std::vector<int32_t> &tokens,
                                 bool computeAllLogits = true);

        /// @brief Forward pass with P0 exact two-pass top-K pruning of the LM head.
        /// Computes logits exactly only for the top-K (plus forceInclude) candidate
        /// vocab rows; all other logit rows are set to the pruned sentinel. Only
        /// valid for a single token (seqLen == 1). Used by the single-token
        /// generation loop.
        /// @param tokens        Input token IDs (must be length 1).
        /// @param computeAllLogits Ignored in this overload (pruning implies a
        ///                          single last-token LM head computation).
        /// @param pruneTopK     Target top-K count; 0 or negative disables pruning.
        /// @param forceInclude  Token IDs that must always be candidates (repeat
        ///                      penalty history). May be null.
        /// @return Logits tensor (1 x 1 x vocabSize) with pruned rows set to
        ///         the pruned sentinel.
        np::Array<float> forward(const std::vector<int32_t> &tokens,
                                 bool computeAllLogits, int32_t pruneTopK,
                                 const std::vector<int32_t> *forceInclude = nullptr);

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

        /// @brief Debug: get the quantized separate LM head matrix (empty if tied).
        const QuantizedMatrix &debugGetLMHead() const { return lmHead_; }

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

        /// @brief Format a chat prompt using the model's chat template.
        /// @param messages Vector of {role, content} pairs (e.g., {"user", "Hello"})
        /// @param addGenerationPrompt Whether to append the assistant prefix
        /// @return Formatted prompt string
        std::string formatChat(const std::vector<std::pair<std::string, std::string>> &messages,
                               bool addGenerationPrompt = true) const;

        /// @brief Compute MoE FFN for Gemma4 architecture.
        /// Routes tokens to top-k experts, computes expert FFN, and combines outputs.
        /// @param ffnNorm Input to the MoE FFN (seqLen x hiddenSize)
        /// @param ffnOut Output buffer (seqLen x hiddenSize)
        /// @param seqLen Number of tokens
        /// @param hiddenSize Hidden dimension
        /// @param intermediateSize FFN intermediate dimension (per-expert)
        /// @param w Layer weights containing MoE tensors
        void computeGemma4MoE(const float *ffnNorm, float *ffnOut,
                              uint32_t seqLen, uint32_t hiddenSize,
                              uint32_t intermediateSize,
                              const LayerWeights &w) const;

        /// @brief Compute MoE FFN for Qwen35MoE architecture.
        /// Routes tokens to top-k experts, computes expert FFN + shared expert, and combines outputs.
        /// @param ffnNorm Input to the MoE FFN (seqLen x hiddenSize)
        /// @param ffnOut Output buffer (seqLen x hiddenSize)
        /// @param seqLen Number of tokens
        /// @param hiddenSize Hidden dimension
        /// @param intermediateSize FFN intermediate dimension (per-expert)
        /// @param w Layer weights containing MoE tensors
        void computeQwen35MoE(const float *ffnNorm, float *ffnOut,
                              uint32_t seqLen, uint32_t hiddenSize,
                              uint32_t intermediateSize,
                              const LayerWeights &w) const;

    private:
        // ---- Model weights (stored in native quantized format) ----

        QuantizedEmbedding quantizedEmbeddings_;
        std::vector<LayerWeights> layers_;

        // Final norm (F32)
        np::Array<float> finalNorm_;// hiddenSize

        // LM head (quantized, or tied with embeddings)
        QuantizedMatrix lmHead_;
        bool lmHeadTied_ = true;

        // Plan §3: pre-quantized Q8_K copy of the separate (non-tied) LM head.
        // The LM head mat-vec is the largest single memory read per token
        // (vocabSize × hiddenSize); Q8_K stores 256 int8 values + a block scale
        // per 256-element block, and the dot products run through the
        // _mm256_maddubs_epi16 int8 kernels with the hidden vector quantized
        // once per token (reused across all vocab rows). Built at load time in
        // buildDequantizedEmbeddings() when the LM head is separate.
        // Layout: [row][blocksPerRow] with blocksPerRow = (hiddenSize+255)/256.
        std::vector<Q8KBlock> lmHeadQ8K_;

        // Pre-dequantized embedding matrix for fast LM head (populated during load)
        DequantizedEmbedding dequantizedEmbeddings_;

        // P0 (BENCHMARK_REPORT §5/P0) LM-head top-k pruning metadata. For a
        // separate (non-tied) LM head, per-16-element-subgroup dequantized
        // (min,max) bounds built at load time. K-quants store a per-16-element
        // scale, so per-16 bounds are tight (candidate set approximates top-K),
        // enabling the pruning pass to compute exact top-K candidates without
        // reading the full large LM head every token.
        // Layout: [row][blocksPerRow][SCG (=16)][min,max].
        std::vector<float> lmHeadBounds_;// [row][block][sub][min,max]
        uint32_t lmHeadBoundsBlocksPerRow_ = 0;
        uint32_t lmHeadBoundsSubgroupsPerBlock_ = 0;

        // P0 pruning scratch buffers (reused across forward() calls to avoid
        // per-token heap allocation for the 151,936-element vocab arrays).
        mutable std::vector<float> topKScratchBounds_;// per-vocab-row [lower,upper] (stride 2, size 2*vocabSize)
        mutable std::vector<float> topKScratchSort_;  // copy for nth_element (size vocabSize)
        mutable std::vector<uint32_t> topKCandidates_;// candidate vocab indices
        mutable std::vector<float> topKScratchSumPos_;// per-subgroup positive-sum of hidden
        mutable std::vector<float> topKScratchSumNeg_;// per-subgroup negative-sum of hidden
        mutable std::vector<float> topKScratchDotBuf_;// per candidate exact-dot scratch (size cols)

        // Sampling scratch (reused across tokens to avoid per-token heap
        // allocations for the 151,936-element vocab arrays; see
        // ModelSampling.cpp). The top-K/top-P pipelines in applySamplingParams
        // use this reused pair vector instead of allocating a fresh 1.2 MB
        // std::vector<std::pair<float,uint32_t>> on every sampling call.
        mutable std::vector<std::pair<float, uint32_t>> samplingTopPairs_;

        // One-shot P0 pruning request consumed by the next single-token forward().
        // Set by the 4-arg forward(...,pruneTopK) overload before delegating to the
        // core forward(); cleared afterward.
        int32_t pendingPruneTopK_ = 0;
        const std::vector<int32_t> *pendingForceInclude_ = nullptr;
        bool pendingPruneActive_ = false;

        // Once P0 pruning is observed to be ineffective for this model/LM-head
        // (candidate set stays ~ the full vocab because the per-subgroup interval
        // [min,max] bounds are too loose to shrink it for K-quants), pruning is
        // disabled permanently so the 5-pass machinery stops adding per-token
        // overhead. See BENCHMARK_REPORT §7.5.
        mutable bool lmHeadPruneUseless_ = false;

        /// @brief Sentinel for pruned (non-candidate) logit rows.
        static constexpr float kPrunedLogitSentinel = -1e30f;

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

            // SSM state for Qwen35MoE architecture (per-layer)
            // convBuf: [numLayers][ssmConvKernel - 1] (past conv1d inputs)
            std::vector<std::vector<float>> ssmConvBuf;
            // ssmState: [numLayers][ssmInnerSize * ssmStateSize] (SSM hidden state)
            std::vector<std::vector<float>> ssmState;
        };
        KVCache kvCache_;

        // GPU offload runtime state (opaque).  Only alive when USE_CUDA and the
        // GPU is enabled (default ON; TINYCODER_GPU=0 opts out); the adapter is
        // created lazily on first use.
        ModelGPUState *gpuState_ = nullptr;

        // Latch: this forward session is currently owned by the GPU engine
        // (GPU KV cache at kvPos_ is authoritative).  Set on the first
        // successful GPU forward of a session (kvCache_.pos == 0), cleared
        // by clearKVCache() and whenever a GPU forward fails (so the session
        // continues on the CPU from the CPU-side KV cache without divergence).
        bool gpuSessionActive_ = false;

        // ---- RoPE tables (P3: plans/prefill_optimization_plan.md) ----
        // Precomputed cos/sin(p * theta^{-2*d/headDim}) for each position p and
        // frequency pair d2 = d/2 in [0, headDim/2). Layout:
        //   ropeCosTable_[p * ropePairs_ + d2]  = cos(p * freq[d2])
        //   ropeSinTable_[p * ropePairs_ + d2]  = sin(p * freq[d2])
        // with freq[d2] = 1/pow(theta, 2*d2/headDim). Grown on demand so each
        // (position, pair) is computed once instead of recomputing std::cos/sin
        // for every (position, dim, head) in every forward pass (the previous
        // applyRoPE recomputed trig ~2M times per 40-token prefill). Rebuilt
        // automatically if headDim or ropeTheta changes (new model load).
        mutable std::vector<float> ropeCosTable_;
        mutable std::vector<float> ropeSinTable_;
        uint32_t ropeHeadDim_ = 0;
        float ropeTheta_ = 0.0f;
        uint32_t ropePairs_ = 0;   // headDim / 2
        uint32_t ropeTablePos_ = 0;// number of positions currently filled

        /// @brief Grow the RoPE cos/sin tables to cover positions [0, neededPos).
        void ensureRoPETables(uint32_t neededPos);

        // ---- Forward pass components ----

        /// @brief Reusable per-thread scratch buffers for the forward pass.
        ///
        /// Every forward() call previously allocated ~11 intermediate buffers
        /// (hidden, attnNorm, q, k, v, attnOut, attnProj, ffnNorm, gate, up,
        /// ffnOut) plus several SSM temporaries on the heap. Hoisting them here
        /// eliminates those per-token heap allocations. Buffers grow on demand
        /// and are retained across calls, so steady-state generation performs
        /// zero allocations for intermediate state.
        struct ScratchPool {
            // [seqLen, hiddenSize]
            std::vector<float> hidden;
            std::vector<float> attnNorm;
            std::vector<float> attnProj;
            std::vector<float> ffnNorm;
            std::vector<float> ffnOut;

            // [seqLen, nHeads, headDim]
            std::vector<float> q;
            std::vector<float> attnOut;

            // [seqLen, nKVHeads, headDim]
            std::vector<float> k;
            std::vector<float> v;

            // [seqLen, intermediateSize]
            std::vector<float> gate;
            std::vector<float> up;

            // SSM temporaries (Qwen35MoE)
            std::vector<float> ssmIn;
            std::vector<float> ssmConvOut;
            std::vector<float> ssmConvInput;
            std::vector<float> ssmOut;
        };

        /// @brief Get the per-thread scratch pool for the calling thread.
        /// @return Reference to the calling thread's ScratchPool.
        ScratchPool &scratchPool() const;

        /// @brief Apply RMSNorm (optimized with direct pointer access).
        void rmsNormInPlace(const float *x, float *out, const float *weight,
                            uint32_t n, float eps = 1e-6f) const;

        /// @brief Apply RoPE (Rotary Position Embedding) with precomputed
        /// frequencies.
        /// @param rotateK When false, the K rotation is skipped entirely; the
        ///        caller fuses it into the KV-cache store via storeKVWithRoPE
        ///        (P3, eliminating the separate K-rotation pass and the K
        ///        memcpy for the prefill store).
        void applyRoPE(float *q, float *k, uint32_t qSeqLen, uint32_t kSeqLen,
                       uint32_t qHeads, uint32_t kHeads, uint32_t pos,
                       bool rotateK = true);

        /// @brief Store K and V into the KV cache, fusing the RoPE rotation
        /// into the K write (P3). Reads the *unrotated* K from kSrc, applies the
        /// position-based rotation for cache positions [cachePos, cachePos+seqLen),
        /// and writes the rotated result directly to kDst; V is copied verbatim.
        /// Produces bit-identical cache values to the old applyRoPE(k)+memcpy.
        /// Requires the RoPE tables to already cover cachePos+seqLen (the Q
        /// rotation in applyRoPE guarantees this).
        void storeKVWithRoPE(const float *kSrc, const float *vSrc, float *kDst,
                             float *vDst, uint32_t seqLen, uint32_t cachePos,
                             uint32_t kHeads);

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

        /// @brief GeLU (Gaussian Error Linear Unit) activation in-place.
        /// Used by Gemma4 architecture.
        void geluInPlace(float *x, uint32_t n) const;

        /// @brief Apply final logit softcapping: tanh(x / cap) * cap.
        /// Used by Gemma4 architecture.
        void softcapInPlace(float *x, uint32_t n, float cap);

        /// @brief Compute top-K-pruned logits for a single token's LM head
        /// (BENCHMARK_REPORT §5/P0), generic over the LM-head quantization type.
        ///
        /// Exact two-pass pruning: pass 1 computes provable per-row lower/upper
        /// logit bounds from the precomputed per-block (min,max) bounds (in
        /// `bounds`) and the input per-block positive/negative sums, so the
        /// candidate set is a superset of the true top-K. Pass 2 computes exact
        /// block-wise dot products (dequantize + FMA, matching the reference path)
        /// only for the candidates. Non-candidate rows are set to a large negative
        /// sentinel, preserving the exact top-K ordering and the sampled token.
        ///
        /// @param data        Raw quantized LM-head matrix (vocabSize x hiddenSize).
        /// @param type        GGML quantization type of `data`.
        /// @param bounds      Per-block dequantized (min,max): [row][block][min,max].
        /// @param blocksPerRow Number of 256-element blocks per vocab row.
        /// @param vocabSize   Number of vocab rows.
        /// @param hiddenSize  Hidden dimension size.
        /// @param hidden      Input hidden state vector (size hiddenSize).
        /// @param pruneTopK   Target top-K count (topK from InferenceParams).
        /// @param forceInclude Token IDs that must always be candidates (repeat
        ///                     penalty history). May be null.
        /// @param logitsOut   Output buffer (size vocabSize, pre-allocated).
        /// @return Number of candidates computed (for diagnostics).
        uint32_t computeLogitsTopK(const uint8_t *data, uint32_t type,
                                   const std::vector<float> &bounds,
                                   uint32_t blocksPerRow, uint32_t subgroupsPerBlock,
                                   uint32_t vocabSize, uint32_t hiddenSize,
                                   const float *hidden, int32_t pruneTopK,
                                   const std::vector<int32_t> *forceInclude,
                                   float *logitsOut);

        /// @brief Apply Q/K norms (per-head RMSNorm before RoPE).
        /// Used by Gemma4 and Qwen35MoE architectures.
        void applyQKNorms(float *q, float *k, uint32_t seqLen,
                          uint32_t qHeads, uint32_t kHeads,
                          const float *qNorm, const float *kNorm);

        /// @brief Load weights from GGUF (stores in native quantized format).
        bool loadWeights(GGUFLoader &loader);

        /// @brief Initialize KV cache.
        bool initKVCache();

        /// @brief Build pre-dequantized embedding matrix for fast LM head.
        /// Called after loadWeights() to avoid per-token dequantization overhead.
        bool buildDequantizedEmbeddings();

        // GPU offload adapter pokes the private weights/KV cache.
        friend class ModelGPUAdapter;

        // Make these methods public for unit tests
        friend class ParisQuestionTest;
        friend class DebugForwardTest;
    };

}// namespace tinycoder
