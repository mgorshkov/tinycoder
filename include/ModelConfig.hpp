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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tinycoder {

    /// @brief Configuration for a Qwen2.5-Coder model.
    ///
    /// Weights are stored in their native quantized format and dequantized
    /// on-the-fly during matrix-vector multiplication. This keeps memory
    /// usage close to the compressed file size (~637 MB) rather than the
    /// F32 dequantized size (~5.2 GB).
    /// @brief Supported model architecture identifier (from GGUF
    /// `general.architecture`).
    static constexpr const char *ARCH_QWEN2 = "qwen2";

    /// @brief Supported model names (from GGUF `general.name`).
    static constexpr const char *MODEL_QWEN2_5_CODER_0_5B = "Qwen2.5 Coder 0.5B";
    static constexpr const char *MODEL_QWEN2_5_CODER_1_5B = "Qwen2.5 Coder 1.5B";
    static constexpr const char *MODEL_QWEN2_5_CODER_1_5B_INSTRUCT =
            "Qwen2.5 Coder 1.5B Instruct";
    static constexpr const char *MODEL_QWEN2_5_CODER_1_5B_INSTRUCT_GGUF =
            "Qwen2.5 Coder 1.5B Instruct GGUF";
    static constexpr const char *MODEL_QWEN2_5_CODER_7B_INSTRUCT =
            "Qwen2.5 Coder 7B Instruct";

    /// @brief Check if a model name is in the supported list.
    inline bool isSupportedModel(const std::string &name) {
        return name == MODEL_QWEN2_5_CODER_0_5B ||
               name == MODEL_QWEN2_5_CODER_1_5B ||
               name == MODEL_QWEN2_5_CODER_1_5B_INSTRUCT ||
               name == MODEL_QWEN2_5_CODER_1_5B_INSTRUCT_GGUF ||
               name == MODEL_QWEN2_5_CODER_7B_INSTRUCT;
    }

    struct ModelConfig {
        std::string modelPath;    ///< Path to the GGUF file
        std::string tokenizerPath;///< Path to the tokenizer (optional, can be
                                  ///< embedded in GGUF)
        std::string architecture; ///< Model architecture (e.g., "qwen2")
        std::string modelName;    ///< Model name (e.g., "Qwen2.5 Coder 1.5B")

        // Architecture parameters (from model metadata)
        uint32_t vocabSize = 151936;     // Qwen2.5-Coder tokenizer vocab size
        uint32_t hiddenSize = 1536;      // 0.5B: 1024, 1.5B: 1536, 7B: 3584
        uint32_t intermediateSize = 8960;// 0.5B: 4096, 1.5B: 8960, 7B: 18944
        uint32_t numLayers = 28;         // 0.5B: 24, 1.5B: 28, 7B: 28
        uint32_t numAttentionHeads = 12; // 0.5B: 16, 1.5B: 12, 7B: 28
        uint32_t numKVHeads = 2;         // 0.5B: 4, 1.5B: 2, 7B: 4 (GQA)
        uint32_t maxSeqLen =
                2048;// Max context length (reduced from 8192 to save memory)
        float ropeTheta = 1000000.0f;
        uint32_t headDim = 128;// hiddenSize / numAttentionHeads

        // Runtime
        uint32_t nThreads = 4;// OpenMP threads

        /// @brief Estimate total memory needed to load this model.
        /// Weights are stored in native quantized format (~637 MB total).
        /// @return Estimated memory in bytes
        uint64_t estimatedMemoryBytes() const {
            // Memory estimate for Qwen2.5-Coder-1.5B with quantized weights.
            //
            // Weights are stored in their native GGML quantization format
            // (IQ3_XXS, IQ3_S, IQ2_S, Q5_K, Q5_1, Q4_K) and dequantized
            // on-the-fly during matrix-vector multiplication.
            //
            // Steady state after loading:
            //   Quantized weights (~637 MB) + KV cache + norms + activations
            //   ≈ 637 + 112 + norms = ~750 MB
            //
            // The file size is ~637 MB, and quantized weights in memory
            // are approximately the same size.

            // Use file size as a proxy for quantized weight memory.
            // The actual memory for quantized weights is very close to
            // the on-disk size since we store them in their native format.
            // We estimate ~700 MB for weights (slightly above file size
            // to account for in-memory overhead).

            // Quantized weights: use the file size as an approximation.
            // For the 1.5B model, the GGUF file is ~637 MB.
            // We add a small overhead for the std::vector<uint8_t> containers.
            uint64_t weightBytes =
                    700ULL * 1024 * 1024;// ~700 MB for quantized weights

            // Token embeddings (Q5_K quantized, 170 bytes per 256 weights)
            uint64_t numEmbedElements = static_cast<uint64_t>(vocabSize) * hiddenSize;
            uint64_t numEmbedBlocks = (numEmbedElements + 255) / 256;
            uint64_t embedBytes = numEmbedBlocks * 170;

            // KV cache (F32)
            uint64_t kvCacheBytes = static_cast<uint64_t>(numLayers) * maxSeqLen *
                                    numKVHeads * headDim * sizeof(float) * 2;

            // Norms (F32)
            uint64_t normBytes =
                    static_cast<uint64_t>(numLayers + 1) * hiddenSize * sizeof(float);

            // Activations (F32, temporary buffers during forward pass)
            uint64_t activationBytes =
                    static_cast<uint64_t>(hiddenSize) * sizeof(float) * 20;

            return weightBytes + embedBytes + kvCacheBytes + normBytes +
                   activationBytes;
        }
    };

    /// @brief Inference parameters for text generation.
    struct InferenceParams {
        int32_t maxTokens = 2048;  // Max tokens to generate
        float temperature = 0.7f;  // Sampling temperature
        float topP = 0.9f;         // Nucleus sampling threshold
        float topK = 40.0f;        // Top-K sampling
        float repeatPenalty = 1.1f;// Repetition penalty
        int32_t repeatLastN = 64;  // Number of tokens to consider for penalty
        float presencePenalty = 0.0f;
        float frequencyPenalty = 0.0f;
        uint32_t seed = 0;// 0 = random
    };

}// namespace tinycoder
