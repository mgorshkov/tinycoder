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

    /// @brief Supported model architecture identifiers (from GGUF
    /// `general.architecture`).
    static constexpr const char *ARCH_QWEN2 = "qwen2";
    static constexpr const char *ARCH_GEMMA4 = "gemma4";
    static constexpr const char *ARCH_QWEN35MOE = "qwen35moe";

    /// @brief Check if an architecture string is supported.
    inline bool isSupportedArchitecture(const std::string &arch) {
        return arch == ARCH_QWEN2 ||
               arch == ARCH_GEMMA4 ||
               arch == ARCH_QWEN35MOE;
    }

    /// @brief Check if a model name is in the supported list.
    inline bool isSupportedModel(const std::string &name) {
        // Qwen2 models
        if (name == "Qwen2.5 Coder 0.5B" ||
            name == "Qwen2.5 Coder 1.5B" ||
            name == "Qwen2.5 Coder 1.5B Instruct" ||
            name == "Qwen2.5 Coder 1.5B Instruct GGUF" ||
            name == "Qwen2.5 Coder 7B Instruct")
            return true;
        // Gemma4 models
        if (name == "Gemma4 Coding Merged Fp16" ||
            name == "Gemma-4 26B-A4B IT (smart Q4_0, QAT-lossless)")
            return true;
        // Qwen35MoE models
        if (name == "Safetensors" ||
            name == "Qwen3.6 35B A3B Claude 4.7 Opus Reasoning Distilled" ||
            name == "Qwen3.6-35B-A3B")
            return true;
        return false;
    }

    struct ModelConfig {
        std::string modelPath;    ///< Path to the GGUF file
        std::string tokenizerPath;///< Path to the tokenizer (optional, can be
                                  ///< embedded in GGUF)
        std::string architecture; ///< Model architecture (e.g., "qwen2", "gemma4", "qwen35moe")
        std::string modelName;    ///< Model name (e.g., "Qwen2.5 Coder 1.5B")
        std::string chatTemplate; ///< Jinja chat template from GGUF metadata (tokenizer.chat_template)

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

        // Gemma4-specific parameters
        float finalLogitSoftcapping = 0.0f;  // 0 = disabled
        uint32_t expertCount = 0;            // 0 = dense (non-MoE)
        uint32_t expertUsedCount = 0;        // 0 = dense (non-MoE)
        uint32_t expertFeedForwardLength = 0;// per-expert FFN size
        uint32_t ropeDimensionCount = 0;     // 0 = use headDim

        // Qwen35MoE-specific parameters
        uint32_t fullAttentionInterval = 0;// 0 = all layers use attention
        uint32_t ssmInnerSize = 0;
        uint32_t ssmStateSize = 0;
        uint32_t ssmConvKernel = 0;
        uint32_t ssmGroupCount = 0;
        uint32_t ssmTimeStepRank = 0;
        uint32_t expertSharedFeedForwardLength = 0;
        uint32_t nextnPredictLayers = 0;

        // Runtime
        uint32_t nThreads = 4;// OpenMP threads

        /// @brief Estimate total memory needed to load this model.
        /// @return Estimated memory in bytes
        uint64_t estimatedMemoryBytes() const {
            // Quantized weights: use a rough estimate based on hidden size and layers
            // For a typical transformer: ~2 * hiddenSize * intermediateSize * numLayers * bytes_per_weight
            // This is a rough estimate; actual size depends on quantization type
            uint64_t weightBytes = 700ULL * 1024 * 1024;// ~700 MB default

            // Token embeddings
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
