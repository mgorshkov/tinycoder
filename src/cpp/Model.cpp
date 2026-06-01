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

#include "Model.hpp"
#include "GGMLDequantize.hpp"
#include "LMHead.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sys/sysinfo.h>

#ifdef USE_CUDA
#include "LMHeadCUDA.hpp"
#endif

namespace tinycoder {

    Model::Model() = default;
    Model::~Model() = default;

    // ---- Model loading ----

    bool Model::load(const std::string &modelPath, std::string *outError,
                     ProgressCallback progressCb) {
        std::cout << "[TinyCoder] Loading model from: " << modelPath << std::endl;

        auto reportProgress = [&](float p, const std::string &stage) {
            std::cout << "[TinyCoder] " << stage << " (" << static_cast<int>(p * 100) << "%)" << std::endl;
            if (progressCb) {
                progressCb(p, stage);
            }
        };

        // Helper to set error message
        auto setError = [outError](const std::string &msg) {
            std::cerr << "[TinyCoder] " << msg << std::endl;
            if (outError) {
                *outError = msg;
            }
        };

        reportProgress(0.0f, "Checking model file...");

        // Check file existence
        std::ifstream testFile(modelPath, std::ios::binary);
        if (!testFile.is_open()) {
            setError("File not found or cannot be opened: " + modelPath);
            return false;
        }
        testFile.close();

        // Estimate memory requirements before loading
        uint64_t fileSize = 0;
        {
            std::ifstream sizeFile(modelPath, std::ios::binary | std::ios::ate);
            fileSize = static_cast<uint64_t>(sizeFile.tellg());
        }

        reportProgress(0.05f, "Reading GGUF metadata...");

        // Load GGUF metadata first to get config
        GGUFLoader metaLoader;
        if (!metaLoader.loadMetadata(modelPath)) {
            setError("Failed to read GGUF metadata");
            return false;
        }
        config_ = metaLoader.config();

        // Cap maxSeqLen to a reasonable value to avoid excessive memory usage.
        // The model may support up to 32768 tokens, but that requires ~1.8 GB for KV
        // cache alone. We default to 2048 which gives ~112 MB KV cache.
        if (config_.maxSeqLen > 2048) {
            std::cout << "[TinyCoder] Reducing maxSeqLen from " << config_.maxSeqLen
                      << " to 2048 to fit available memory" << std::endl;
            config_.maxSeqLen = 2048;
        }

        // Estimate total memory needed
        uint64_t estimatedMem = config_.estimatedMemoryBytes();
        uint64_t fileMem = fileSize;

        // Get available system memory
        struct sysinfo si;
        uint64_t availableMem = 0;
        if (sysinfo(&si) == 0) {
            availableMem = static_cast<uint64_t>(si.freeram) * si.mem_unit;
        }

        std::cout << "[TinyCoder] Memory estimate: " << (estimatedMem / (1024 * 1024))
                  << " MB (weights: ~" << (fileMem / (1024 * 1024)) << " MB on disk)"
                  << ", available: " << (availableMem / (1024 * 1024)) << " MB"
                  << std::endl;

        // Bypass memory check for debugging
        if (availableMem > 0 && estimatedMem > availableMem) {
            std::cout << "[TinyCoder] WARNING: Insufficient memory: need ~"
                      << (estimatedMem / (1024 * 1024)) << " MB but only "
                      << (availableMem / (1024 * 1024))
                      << " MB available. Attempting to load anyway (may swap)..."
                      << std::endl;
        }

        reportProgress(0.1f, "Loading GGUF file...");

        // Load full GGUF file
        GGUFLoader loader;
        if (!loader.load(modelPath)) {
            setError("Failed to parse GGUF file (invalid or corrupted format)");
            return false;
        }

        config_ = loader.config();

        // Re-apply maxSeqLen cap (config_ was overwritten by loader.config())
        if (config_.maxSeqLen > 2048) {
            config_.maxSeqLen = 2048;
        }

        // Validate model architecture
        if (config_.architecture != ARCH_QWEN2) {
            setError("Unsupported model architecture: \"" + config_.architecture +
                     "\". Only \"" + ARCH_QWEN2 + "\" is supported.");
            return false;
        }

        // Validate model name is in the supported list
        if (!config_.modelName.empty() && !isSupportedModel(config_.modelName)) {
            setError("Unsupported model: \"" + config_.modelName +
                     "\". Supported models: " + MODEL_QWEN2_5_CODER_0_5B + ", " +
                     MODEL_QWEN2_5_CODER_1_5B + ", " +
                     MODEL_QWEN2_5_CODER_1_5B_INSTRUCT + ", " +
                     MODEL_QWEN2_5_CODER_1_5B_INSTRUCT_GGUF + ", " +
                     MODEL_QWEN2_5_CODER_7B_INSTRUCT);
            return false;
        }

        std::cout << "[TinyCoder] Model: \"" << config_.modelName << "\" ("
                  << config_.architecture << ")" << std::endl;
        std::cout << "[TinyCoder] Model config: " << config_.numLayers << " layers, "
                  << config_.hiddenSize << " hidden, " << config_.numAttentionHeads
                  << " heads, " << config_.numKVHeads << " KV heads" << std::endl;

        // Validate config
        if (config_.numLayers == 0) {
            setError("Model has zero layers \u2014 GGUF metadata may be incomplete or "
                     "unsupported architecture");
            return false;
        }
        if (config_.hiddenSize == 0 || config_.vocabSize == 0) {
            setError("Model has invalid dimensions (hiddenSize=" +
                     std::to_string(config_.hiddenSize) +
                     ", vocabSize=" + std::to_string(config_.vocabSize) + ")");
            return false;
        }

        reportProgress(0.2f, "Loading tokenizer...");

        // Load tokenizer from GGUF
        if (!tokenizer_.loadFromGGUF(modelPath)) {
            std::cerr << "[TinyCoder] Failed to load tokenizer, using embedded data"
                      << std::endl;
        }

        reportProgress(0.25f, "Loading model weights...");

        // Load weights (stored in native quantized format)
        if (!loadWeights(loader)) {
            setError("Failed to load model weights");
            return false;
        }

        reportProgress(0.7f, "Building pre-dequantized embeddings...");

        // Build pre-dequantized embedding matrix for fast LM head computation
        // This eliminates the expensive dequantization from every forward pass
        std::cout << "[TinyCoder] Building pre-dequantized embedding matrix..."
                  << std::endl;
        if (!buildDequantizedEmbeddings()) {
            setError("Failed to build pre-dequantized embeddings");
            return false;
        }

        reportProgress(0.9f, "Initializing KV cache...");

        // Initialize KV cache
        if (!initKVCache()) {
            setError("Failed to initialize KV cache (out of memory?)");
            return false;
        }

        loaded_ = true;
        reportProgress(1.0f, "Model loaded successfully");
        std::cout << "[TinyCoder] Model loaded successfully" << std::endl;
        return true;
    }

    uint64_t Model::estimateMemory(const std::string &modelPath) {
        GGUFLoader loader;
        if (!loader.loadMetadata(modelPath)) {
            return 0;
        }
        return loader.config().estimatedMemoryBytes();
    }

    bool Model::loadWeights(GGUFLoader &loader) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Helper to load a quantized weight tensor into a QuantizedMatrix.
        // Copies the raw quantized data without dequantizing.
        auto loadQuantized = [&](const std::string &name) -> QuantizedMatrix {
            auto info = loader.getTensorInfo(name);
            if (!info) {
                std::cerr << "[TinyCoder] Tensor info not found: " << name << std::endl;
                return QuantizedMatrix{};
            }

            const uint8_t *data = loader.getTensor(name);
            if (!data) {
                std::cerr << "[TinyCoder] Tensor data not found: " << name << std::endl;
                return QuantizedMatrix{};
            }

            // GGUF stores shapes in reverse (numpy) order.
            // For a 2D tensor, shape[0] = cols, shape[1] = rows.
            uint32_t rows =
                    info->shape.size() >= 2 ? static_cast<uint32_t>(info->shape[1]) : 1;
            uint32_t cols =
                    info->shape.size() >= 1 ? static_cast<uint32_t>(info->shape[0]) : 1;
            if (info->shape.size() == 1) {
                rows = 1;
                cols = static_cast<uint32_t>(info->shape[0]);
            }

            uint32_t blockSize = ggmlBlockSize(info->type);
            uint32_t typeSize = ggmlTypeSize(info->type);
            uint64_t dataBytes = static_cast<uint64_t>(rows) * cols * sizeof(float);// fallback

            if (blockSize > 1 && typeSize > 0) {
                // Blocks are per-row: each row has (cols + blockSize - 1) / blockSize blocks.
                // Total blocks = rows * blocksPerRow.
                uint64_t blocksPerRow = (static_cast<uint64_t>(cols) + blockSize - 1) / blockSize;
                uint64_t totalBlocks = static_cast<uint64_t>(rows) * blocksPerRow;
                dataBytes = totalBlocks * typeSize;
            } else if (typeSize > 0) {
                dataBytes = static_cast<uint64_t>(rows) * cols * typeSize;
            }

            std::cout << "[TinyCoder] Loading " << name << ": type=" << info->type
                      << " shape=[";
            for (size_t si = 0; si < info->shape.size(); ++si) {
                if (si > 0)
                    std::cout << ",";
                std::cout << info->shape[si];
            }
            std::cout << "] rows=" << rows << " cols=" << cols
                      << " dataBytes=" << (dataBytes / (1024 * 1024)) << " MB"
                      << std::endl;

            QuantizedMatrix qm;
            qm.rows = rows;
            qm.cols = cols;
            qm.type = info->type;
            qm.data.assign(data, data + dataBytes);
            return qm;
        };

        // Helper to load a 1D F32 tensor (norms, etc.)
        auto loadF32_1D = [&](const std::string &name) -> np::Array<float> {
            auto info = loader.getTensorInfo(name);
            if (!info)
                return np::Array<float>{};

            const uint8_t *data = loader.getTensor(name);
            if (!data)
                return np::Array<float>{};

            uint32_t n = static_cast<uint32_t>(info->shape[0]);
            std::vector<float> vec(n);
            std::memcpy(vec.data(), data, n * sizeof(float));
            return np::Array<float>(vec, np::Shape{n});
        };

        // Load token embeddings (keep quantized)
        {
            auto info = loader.getTensorInfo("token_embd.weight");
            if (!info) {
                std::cerr << "[TinyCoder] Missing tensor: token_embd.weight" << std::endl;
                return false;
            }

            const uint8_t *data = loader.getTensor("token_embd.weight");
            if (!data) {
                std::cerr << "[TinyCoder] Failed to get tensor data: token_embd.weight"
                          << std::endl;
                return false;
            }

            // GGUF stores shapes in reverse (numpy) order.
            // For a 2D tensor, shape[0] = cols (hiddenSize), shape[1] = rows (vocabSize).
            uint32_t rows =
                    info->shape.size() >= 2 ? static_cast<uint32_t>(info->shape[1]) : 1;
            uint32_t cols =
                    info->shape.size() >= 1 ? static_cast<uint32_t>(info->shape[0]) : 1;
            if (info->shape.size() == 1) {
                rows = 1;
                cols = static_cast<uint32_t>(info->shape[0]);
            }

            // Calculate compressed size based on quantization type
            uint32_t blockSize = ggmlBlockSize(info->type);
            uint32_t typeSize = ggmlTypeSize(info->type);
            uint64_t numElements = static_cast<uint64_t>(rows) * cols;
            uint64_t numBlocks = (numElements + blockSize - 1) / blockSize;
            uint64_t compressedBytes = numBlocks * typeSize;

            quantizedEmbeddings_.data.assign(data, data + compressedBytes);
            quantizedEmbeddings_.vocabSize = rows;
            quantizedEmbeddings_.hiddenSize = cols;
            quantizedEmbeddings_.type = info->type;

            std::cout << "[TinyCoder] Token embeddings: " << rows << " x " << cols
                      << " (type " << info->type << ", "
                      << (compressedBytes / (1024 * 1024)) << " MB quantized)"
                      << std::endl;
        }

        // Check if LM head is tied (output.weight exists and has same pointer as
        // token_embd.weight)
        auto lmHeadInfo = loader.getTensorInfo("output.weight");
        if (lmHeadInfo) {
            auto lmHeadData = loader.getTensor("output.weight");
            auto embData = loader.getTensor("token_embd.weight");
            if (lmHeadData && embData && lmHeadData == embData) {
                // Same data pointer = weight tying
                lmHeadTied_ = true;
                std::cout << "[TinyCoder] LM head: tied with token embeddings"
                          << std::endl;
            } else if (lmHeadData) {
                // Separate LM head - load as quantized matrix
                lmHead_ = loadQuantized("output.weight");
                if (!lmHead_.empty()) {
                    lmHeadTied_ = false;
                    std::cout << "[TinyCoder] LM head: separate, "
                              << (lmHead_.data.size() / (1024 * 1024)) << " MB quantized"
                              << std::endl;
                } else {
                    lmHeadTied_ = true;
                    std::cout << "[TinyCoder] LM head: tied (failed to load separate)"
                              << std::endl;
                }
            }
        } else {
            lmHeadTied_ = true;
            std::cout << "[TinyCoder] LM head: tied (no output.weight tensor)"
                      << std::endl;
        }

        // Load final norm (F32)
        finalNorm_ = loadF32_1D("output_norm.weight");
        if (finalNorm_.empty()) {
            finalNorm_ = loadF32_1D("token_embd_norm.weight");
        }

        // Load per-layer weights
        layers_.resize(config_.numLayers);
        for (uint32_t i = 0; i < config_.numLayers; ++i) {
            std::string prefix = "blk." + std::to_string(i) + ".";

            // Quantized attention weights
            layers_[i].attnQ = loadQuantized(prefix + "attn_q.weight");
            layers_[i].attnK = loadQuantized(prefix + "attn_k.weight");
            layers_[i].attnV = loadQuantized(prefix + "attn_v.weight");
            layers_[i].attnO = loadQuantized(prefix + "attn_output.weight");

            // Dequantize attnO to F32 for exact float dot product matching the
            // reference (llama_ref_dequant). The quantized matMulVec uses block-level
            // fused dequantize-dot which can produce slightly different results
            // than full dequantize + float dot product.
            {
                auto &qm = layers_[i].attnO;
                uint64_t numElements = static_cast<uint64_t>(qm.rows) * qm.cols;
                auto deq = GGMLDequantize::dequantize(qm.type, qm.data.data(), numElements);
                layers_[i].attnO_deq = np::Array<float>(deq, np::Shape{qm.rows, qm.cols});
            }

            // Quantized FFN weights
            layers_[i].ffnGate = loadQuantized(prefix + "ffn_gate.weight");
            layers_[i].ffnUp = loadQuantized(prefix + "ffn_up.weight");
            layers_[i].ffnDown = loadQuantized(prefix + "ffn_down.weight");

            // Dequantize ffnDown to F32 for exact float dot product matching the
            // reference. ffnDown is Q3_K (type 11) in the Q2_K model, same as attnO.
            {
                auto &qm = layers_[i].ffnDown;
                uint64_t numElements = static_cast<uint64_t>(qm.rows) * qm.cols;
                auto deq = GGMLDequantize::dequantize(qm.type, qm.data.data(), numElements);
                layers_[i].ffnDown_deq = np::Array<float>(deq, np::Shape{qm.rows, qm.cols});
            }

            // F32 norms (tiny, just hiddenSize elements)
            layers_[i].rmsNormAttn = loadF32_1D(prefix + "attn_norm.weight");
            layers_[i].rmsNormFFN = loadF32_1D(prefix + "ffn_norm.weight");

            // F32 Q, K, V biases (Qwen2.5-Coder has these, unlike LLaMA)
            layers_[i].attnQBias = loadF32_1D(prefix + "attn_q.bias");
            layers_[i].attnKBias = loadF32_1D(prefix + "attn_k.bias");
            layers_[i].attnVBias = loadF32_1D(prefix + "attn_v.bias");

            // Validate all tensors loaded
            if (layers_[i].attnQ.empty() || layers_[i].attnK.empty() ||
                layers_[i].attnV.empty() || layers_[i].attnO.empty()) {
                std::cerr << "[TinyCoder] Missing attention weights for layer " << i
                          << std::endl;
                return false;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "[TinyCoder] Weights loaded in " << ms << " ms" << std::endl;

        return true;
    }

    bool Model::initKVCache() {
        uint32_t maxSeq = config_.maxSeqLen;
        uint32_t nLayers = config_.numLayers;
        uint32_t nKVHeads = config_.numKVHeads;
        uint32_t headDim = config_.headDim;

        uint64_t cacheMemBytes = static_cast<uint64_t>(nLayers) * maxSeq * nKVHeads *
                                 headDim * sizeof(float) * 2;

        std::cout << "[TinyCoder] KV cache: " << nLayers << " layers x " << maxSeq
                  << " seq x " << nKVHeads << " heads x " << headDim
                  << " dim = " << (cacheMemBytes / (1024 * 1024)) << " MB"
                  << std::endl;

        kvCache_.k = np::Array<float>(np::Shape{nLayers, maxSeq, nKVHeads, headDim});
        kvCache_.v = np::Array<float>(np::Shape{nLayers, maxSeq, nKVHeads, headDim});
        kvCache_.pos = 0;

        return true;
    }

    bool Model::buildDequantizedEmbeddings() {
        if (quantizedEmbeddings_.empty()) {
            std::cerr << "[TinyCoder] No quantized embeddings to build" << std::endl;
            return false;
        }

        uint32_t vocabSize = quantizedEmbeddings_.vocabSize;
        uint32_t hiddenSize = quantizedEmbeddings_.hiddenSize;

        // Compute total memory needed
        uint64_t embedElements = static_cast<uint64_t>(vocabSize) * hiddenSize;
        uint64_t embedMemBytes = embedElements * sizeof(float);

        std::cout << "[TinyCoder] Pre-dequantizing embeddings: "
                  << vocabSize << " x " << hiddenSize
                  << " = " << (embedElements / (1024 * 1024)) << "M elements ("
                  << (embedMemBytes / (1024 * 1024)) << " MB)" << std::endl;

        auto t0 = std::chrono::high_resolution_clock::now();

        // Full dequantization of the embedding matrix
        auto deqData = GGMLDequantize::dequantize(
                quantizedEmbeddings_.type,
                quantizedEmbeddings_.data.data(),
                embedElements);

        if (deqData.empty()) {
            std::cerr << "[TinyCoder] Failed to dequantize embeddings" << std::endl;
            return false;
        }

        // Move data into persistent buffer
        dequantizedEmbeddings_.data = std::move(deqData);
        dequantizedEmbeddings_.vocabSize = vocabSize;
        dequantizedEmbeddings_.hiddenSize = hiddenSize;

        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        std::cout << "[TinyCoder] Pre-dequantized embeddings in " << ms << " ms"
                  << std::endl;

        return true;
    }

    void Model::clearKVCache() {
        // Clear KV cache position and reset RNG state
        kvCache_.pos = 0;
        // Reset RNG state so generation is reproducible from a clean cache
        rngInitialized_ = false;

        // Zero out KV cache tensors to prevent residual data from previous generations
        std::fill(kvCache_.k.data(), kvCache_.k.data() + kvCache_.k.size(), 0.0f);
        std::fill(kvCache_.v.data(), kvCache_.v.data() + kvCache_.v.size(), 0.0f);
    }

    void Model::rmsNormInPlace(const float *x, float *out, const float *weight,
                               uint32_t n, float eps) {
        // Compute RMSNorm for a single vector of length n.
        // x: input vector
        // out: output vector (can alias x)
        // weight: scale vector of length n
        //
        // Use double precision for sumSq accumulation to match the
        // reference implementation and avoid precision loss with large n.

        double sumSq = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            sumSq += (double) x[i] * (double) x[i];
        }
        float rms = std::sqrt((float) (sumSq / (double) n) + eps);
        float invRms = 1.0f / rms;

        for (uint32_t i = 0; i < n; ++i) {
            out[i] = x[i] * invRms * weight[i];
        }
    }

    void Model::applyRoPE(float *q, float *k, uint32_t qSeqLen, uint32_t kSeqLen,
                          uint32_t qHeads, uint32_t kHeads, uint32_t pos) {
        uint32_t headDim = config_.headDim;
        float theta = config_.ropeTheta;

        // Precompute frequencies for all positions in this call
        // freq[d] = pow(theta, -2*d/headDim) for d = 0, 2, 4, ...
        // This matches the standard RoPE formula: theta_d = base^(-2d/D)
        // freq[d/2] = theta^(-d/headDim) for d = 0, 2, 4, ...
        // Standard RoPE formula: theta_j = base^(-2j/D) where j = d/2, D = headDim
        // So theta_j = base^(-(d)/D) = 1/pow(theta, d/D)
        float freq[128];// max headDim is 128
        for (uint32_t d = 0; d < headDim; d += 2) {
            freq[d / 2] = 1.0f / std::pow(theta, static_cast<float>(d) /
                                                         static_cast<float>(headDim));
        }

        // Apply to Q
        for (uint32_t s = 0; s < qSeqLen; ++s) {
            uint32_t p = pos + s;
            for (uint32_t h = 0; h < qHeads; ++h) {
                uint32_t headOffset = s * qHeads * headDim + h * headDim;
                for (uint32_t d = 0; d < headDim; d += 2) {
                    float f = freq[d / 2];
                    float cosVal = std::cos(p * f);
                    float sinVal = std::sin(p * f);

                    float x0 = q[headOffset + d];
                    float x1 = q[headOffset + d + 1];

                    q[headOffset + d] = x0 * cosVal - x1 * sinVal;
                    q[headOffset + d + 1] = x0 * sinVal + x1 * cosVal;
                }
            }
        }

        // Apply to K
        for (uint32_t s = 0; s < kSeqLen; ++s) {
            uint32_t p = pos + s;
            for (uint32_t h = 0; h < kHeads; ++h) {
                uint32_t headOffset = s * kHeads * headDim + h * headDim;
                for (uint32_t d = 0; d < headDim; d += 2) {
                    float f = freq[d / 2];
                    float cosVal = std::cos(p * f);
                    float sinVal = std::sin(p * f);

                    float x0 = k[headOffset + d];
                    float x1 = k[headOffset + d + 1];

                    k[headOffset + d] = x0 * cosVal - x1 * sinVal;
                    k[headOffset + d + 1] = x0 * sinVal + x1 * cosVal;
                }
            }
        }
    }

    void Model::attentionFused(const float *q, const float *kCache,
                               const float *vCache, float *output, uint32_t seqLen,
                               uint32_t cachePos, uint32_t cacheLen,
                               uint32_t /*layerIdx*/) {
        uint32_t nHeads = config_.numAttentionHeads;
        uint32_t nKVHeads = config_.numKVHeads;
        uint32_t headDim = config_.headDim;
        uint32_t nGroups = nHeads / nKVHeads;

        float invSqrtHeadDim = 1.0f / std::sqrt(static_cast<float>(headDim));

        // Each thread needs its own local scores buffer to avoid data races.
        // We use a thread-local stack buffer (no per-call heap alloc).
        // Head dim is 128, nHeads=12, nKVHeads=2, nGroups=6.
        // Parallelize over (s, g) pairs: collapse(2) on seqLen × nKVHeads avoids
        // intervening declarations that would prevent collapse(3).
        constexpr uint32_t STACK_SCORE_LIMIT = 4096;

#pragma omp parallel for collapse(2) schedule(static)
        for (uint32_t s = 0; s < seqLen; ++s) {
            for (uint32_t g = 0; g < nKVHeads; ++g) {
                // Causal mask: this query token can only attend to cache positions
                // up to (cachePos + s), which is its own position in the cache.
                // Positions beyond that are future tokens that haven't been generated yet.
                uint32_t csEnd = cachePos + s;// inclusive end position

                for (uint32_t h = 0; h < nGroups; ++h) {
                    // Each thread gets its own local scores buffer on the stack.
                    // For cacheLen up to 2048, this is ~8KB — well within stack limits.
                    float localScores[STACK_SCORE_LIMIT];

                    uint32_t qHead = g * nGroups + h;
                    const float *qPtr = q + (s * nHeads * headDim + qHead * headDim);

                    // Compute scores for this query head against allowed cached keys
                    // Use double-precision accumulation, then store as float (matching reference)
                    float maxScore = -1e30f;
                    for (uint32_t cs = 0; cs <= csEnd; ++cs) {
                        const float *kPtr = kCache + (cs * nKVHeads * headDim + g * headDim);
                        double score = 0.0;
                        for (uint32_t d = 0; d < headDim; ++d) {
                            score += static_cast<double>(qPtr[d]) * static_cast<double>(kPtr[d]);
                        }
                        localScores[cs] = static_cast<float>(static_cast<double>(score) * static_cast<double>(invSqrtHeadDim));
                        if (localScores[cs] > maxScore)
                            maxScore = localScores[cs];
                    }
                    // Mask out future positions (set to -infinity before softmax)
                    for (uint32_t cs = csEnd + 1; cs < cacheLen; ++cs) {
                        localScores[cs] = -std::numeric_limits<float>::infinity();
                    }

                    // Softmax: compute sumExp in double from float scores (matching reference)
                    double sumExp = 0.0;
                    for (uint32_t cs = 0; cs < cacheLen; ++cs) {
                        sumExp += static_cast<double>(std::exp(static_cast<double>(localScores[cs] - maxScore)));
                    }
                    float invSumExp = static_cast<float>(1.0 / sumExp);

                    // Weighted sum of values: recompute softmax probabilities from float scores
                    // (matching reference behavior exactly)
                    float *outPtr = output + (s * nHeads * headDim + qHead * headDim);
                    for (uint32_t d = 0; d < headDim; ++d) {
                        double val = 0.0;
                        for (uint32_t cs = 0; cs < cacheLen; ++cs) {
                            // Match reference exactly:
                            //   float sv = std::exp((double)(scores[p] - maxScore)) * invSumExp;
                            //   This is: double * float -> double, then truncated to float
                            //   sum += (double)sv * (double)v_cache[...];
                            double expVal = std::exp(static_cast<double>(localScores[cs] - maxScore));
                            float sv = static_cast<float>(expVal * static_cast<double>(invSumExp));
                            val += static_cast<double>(sv) *
                                   static_cast<double>(vCache[cs * nKVHeads * headDim + g * headDim + d]);
                        }
                        outPtr[d] = std::isinf(val) ? 0.0f : (std::isnan(val) ? 0.0f : static_cast<float>(val));
                    }
                }
            }
        }
    }

    void Model::siluInPlace(float *x, uint32_t n) {
        for (uint32_t i = 0; i < n; ++i) {
            x[i] = x[i] / (1.0f + std::exp(-x[i]));
        }
    }

    void Model::swiGLUInPlace(float *x, const float *y, uint32_t n) {
        // x = silu(x) * y, stored in-place in x
        for (uint32_t i = 0; i < n; ++i) {
            float siluVal = x[i] / (1.0f + std::exp(-x[i]));
            x[i] = siluVal * y[i];
        }
    }

    // -----------------------------------------------------------------------
    // Debug: dump vector stats (first 8, min, max, mean)
    // -----------------------------------------------------------------------
    namespace {
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

        /// @brief Compute y = x * W using dequantized F32 weights (exact float dot product).
        /// This matches the reference implementation in llama_ref_dequant.cpp's mat_mul_vec().
        /// Used for weights that are Q3_K quantized in the Q2_K model (attnO, ffnDown) -
        /// the block-level fused dequantize-dot in matMulVecFused produces slightly different
        /// results than full dequantize + float dot product, and this difference compounds
        /// through RMSNorm to produce large divergences in subsequent layers.
        /// @param W Dequantized F32 weight matrix, stored row-major: W[j][i] = data[j * cols + i]
        /// @param x Input vector of size cols
        /// @param rows Number of rows (output dimension)
        /// @param cols Number of columns (input dimension)
        /// @return Output vector of size rows as np::Array<float>
        np::Array<float> deqMatMulVec(const float *W, const float *x,
                                      uint32_t rows, uint32_t cols) {
            np::Array<float> result(np::Shape{rows});
            float *resultData = result.data();
            // Parallelize over output rows: each row W[j] dot x is independent.
            // rows = hiddenSize (1536) for attnO, or intermediateSize (8960) for ffnDown.
#pragma omp parallel for schedule(static)
            for (uint32_t j = 0; j < rows; ++j) {
                double dot = 0.0;
                const float *Wrow = W + static_cast<size_t>(j) * cols;
                for (uint32_t i = 0; i < cols; ++i) {
                    dot += static_cast<double>(x[i]) * Wrow[i];
                }
                resultData[j] = static_cast<float>(dot);
            }
            return result;
        }
    }// namespace

    std::vector<float> Model::debugForwardWithDumps(int32_t tokenId) {
        uint32_t hiddenSize = config_.hiddenSize;
        uint32_t nHeads = config_.numAttentionHeads;
        uint32_t nKVHeads = config_.numKVHeads;
        uint32_t headDim = config_.headDim;
        uint32_t nLayers = config_.numLayers;
        uint32_t maxSeqLen = config_.maxSeqLen;
        uint32_t vocabSize = config_.vocabSize;
        uint32_t intermediateSize = config_.intermediateSize;

        std::cout << "\n=== Debug Forward Pass (Per-Layer Dumps) ===" << std::endl;
        std::cout << "  Token ID: " << tokenId << std::endl;
        std::cout << "  Config: " << nLayers << " layers, " << hiddenSize
                  << " hidden, " << nHeads << " heads, " << nKVHeads << " KV heads, "
                  << intermediateSize << " intermediate, " << vocabSize << " vocab"
                  << std::endl;

        // ---- Step 1: Token Embedding ----
        std::vector<float> hidden(hiddenSize);
        if (tokenId >= 0 &&
            tokenId < static_cast<int32_t>(quantizedEmbeddings_.vocabSize)) {
            auto embRow = quantizedEmbeddings_.getRow(tokenId);
            for (uint32_t j = 0; j < hiddenSize; ++j) {
                hidden[j] = embRow[j];
            }
        }
        dumpVecStats(hidden.data(), hiddenSize, "Embedding");

        // Pre-allocate per-layer buffers
        std::vector<float> attnNorm(hiddenSize);
        std::vector<float> q(nHeads * headDim);
        std::vector<float> k(nKVHeads * headDim);
        std::vector<float> v(nKVHeads * headDim);
        std::vector<float> attnOut(nHeads * headDim);
        std::vector<float> attnProj(hiddenSize);
        std::vector<float> ffnNorm(hiddenSize);
        std::vector<float> gate(intermediateSize);
        std::vector<float> up(intermediateSize);
        std::vector<float> ffnOut(hiddenSize);

        // Process through all transformer layers
        for (uint32_t layer = 0; layer < nLayers; ++layer) {
            auto &w = layers_[layer];
            std::cout << "\n--- Layer " << layer << " ---" << std::endl;

            // ---- Attention block ----

            // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
            rmsNormInPlace(hidden.data(), attnNorm.data(), w.rmsNormAttn.data(), hiddenSize);
            dumpVecStats(attnNorm.data(), hiddenSize, "  After attn_norm");

            // Q projection
            np::Array<float> qRow = w.attnQ.matMulVec(attnNorm.data());
            std::memcpy(q.data(), qRow.data(), nHeads * headDim * sizeof(float));
            if (!w.attnQBias.empty()) {
                const float *qBias = w.attnQBias.data();
                for (uint32_t i = 0; i < nHeads * headDim; ++i)
                    q[i] += qBias[i];
            }
            dumpVecStats(q.data(), nHeads * headDim, "  Q (after proj)");

            // K projection
            np::Array<float> kRow = w.attnK.matMulVec(attnNorm.data());
            std::memcpy(k.data(), kRow.data(), nKVHeads * headDim * sizeof(float));
            if (!w.attnKBias.empty()) {
                const float *kBias = w.attnKBias.data();
                for (uint32_t i = 0; i < nKVHeads * headDim; ++i)
                    k[i] += kBias[i];
            }

            // V projection
            np::Array<float> vRow = w.attnV.matMulVec(attnNorm.data());
            std::memcpy(v.data(), vRow.data(), nKVHeads * headDim * sizeof(float));
            if (!w.attnVBias.empty()) {
                const float *vBias = w.attnVBias.data();
                for (uint32_t i = 0; i < nKVHeads * headDim; ++i)
                    v[i] += vBias[i];
            }

            // Apply RoPE
            applyRoPE(q.data(), k.data(), 1, 1, nHeads, nKVHeads,
                      static_cast<uint32_t>(kvCache_.pos));
            dumpVecStats(q.data(), nHeads * headDim, "  Q (after RoPE)");

            // Store K, V in cache
            uint32_t cachePos = static_cast<uint32_t>(kvCache_.pos);
            float *kCacheLayer =
                    kvCache_.k.data() + layer * maxSeqLen * nKVHeads * headDim;
            float *vCacheLayer =
                    kvCache_.v.data() + layer * maxSeqLen * nKVHeads * headDim;
            uint32_t kvSize = nKVHeads * headDim;
            for (uint32_t i = 0; i < kvSize; ++i) {
                kCacheLayer[cachePos * nKVHeads * headDim + i] = k[i];
                vCacheLayer[cachePos * nKVHeads * headDim + i] = v[i];
            }

            // Attention
            uint32_t totalCacheLen = cachePos + 1;
            attentionFused(q.data(), kCacheLayer, vCacheLayer, attnOut.data(),
                           1, cachePos, totalCacheLen, layer);
            dumpVecStats(attnOut.data(), nHeads * headDim, "  After attention");

            // Output projection (using dequantized F32 weights for exact float dot product)
            np::Array<float> projRow = deqMatMulVec(w.attnO_deq.data(), attnOut.data(),
                                                    w.attnO.rows, w.attnO.cols);
            std::memcpy(attnProj.data(), projRow.data(), hiddenSize * sizeof(float));
            dumpVecStats(attnProj.data(), hiddenSize, "  After attnO proj");

            // Residual: hidden += attnProj
            for (uint32_t i = 0; i < hiddenSize; ++i) {
                hidden[i] += attnProj[i];
            }
            dumpVecStats(hidden.data(), hiddenSize, "  After attention + residual");

            // ---- FFN block ----

            // RMSNorm: ffnNorm = rmsNorm(hidden, w.rmsNormFFN)
            rmsNormInPlace(hidden.data(), ffnNorm.data(), w.rmsNormFFN.data(), hiddenSize);
            dumpVecStats(ffnNorm.data(), hiddenSize, "  After ffn_norm");

            // Gate projection
            np::Array<float> gateRow = w.ffnGate.matMulVec(ffnNorm.data());
            std::memcpy(gate.data(), gateRow.data(), intermediateSize * sizeof(float));
            dumpVecStats(gate.data(), intermediateSize, "  Gate (before SwiGLU)");

            // Up projection
            np::Array<float> upRow = w.ffnUp.matMulVec(ffnNorm.data());
            std::memcpy(up.data(), upRow.data(), intermediateSize * sizeof(float));
            dumpVecStats(up.data(), intermediateSize, "  Up (before SwiGLU)");

            // SwiGLU: gate = silu(gate) * up
            swiGLUInPlace(gate.data(), up.data(), intermediateSize);
            dumpVecStats(gate.data(), intermediateSize, "  After SwiGLU");

            // Down projection (using dequantized F32 weights for exact float dot product)
            np::Array<float> downRow = deqMatMulVec(w.ffnDown_deq.data(), gate.data(),
                                                    w.ffnDown.rows, w.ffnDown.cols);
            std::memcpy(ffnOut.data(), downRow.data(), hiddenSize * sizeof(float));
            dumpVecStats(ffnOut.data(), hiddenSize, "  After ffnDown proj");

            // Residual: hidden += ffnOut
            for (uint32_t i = 0; i < hiddenSize; ++i) {
                hidden[i] += ffnOut[i];
            }
            dumpVecStats(hidden.data(), hiddenSize, "  After FFN + residual");
        }

        // Advance KV cache position after all layers processed
        kvCache_.pos++;

        // ---- Final RMSNorm ----
        rmsNormInPlace(hidden.data(), hidden.data(), finalNorm_.data(), hiddenSize);
        dumpVecStats(hidden.data(), hiddenSize, "After final norm");

        // ---- LM head (logits) ----
        std::vector<float> logits(vocabSize);
        if (lmHeadTied_) {
            // Use pre-dequantized embeddings for much faster computation
            if (!dequantizedEmbeddings_.empty()) {
                LMHead::computeCPU(hidden.data(),
                                   dequantizedEmbeddings_.data.data(),
                                   dequantizedEmbeddings_.vocabSize,
                                   dequantizedEmbeddings_.hiddenSize, logits.data());
            } else {
                // Fallback to quantized path
                LMHead::computeCPUQuantized(hidden.data(),
                                            quantizedEmbeddings_.data.data(),
                                            quantizedEmbeddings_.type,
                                            quantizedEmbeddings_.vocabSize,
                                            quantizedEmbeddings_.hiddenSize, logits.data());
            }
        } else {
            LMHead::computeCPUSeparate(hidden.data(), lmHead_.data.data(), lmHead_.type,
                                       lmHead_.rows, lmHead_.cols, logits.data());
        }
        dumpVecStats(logits.data(), vocabSize, "Final logits");

        // Print top-10 logits
        std::vector<std::pair<float, int32_t>> top10;
        for (uint32_t i = 0; i < vocabSize; ++i)
            top10.emplace_back(logits[i], static_cast<int32_t>(i));
        std::partial_sort(top10.begin(), top10.begin() + 10, top10.end(),
                          [](const auto &a, const auto &b) { return a.first > b.first; });
        std::cout << "\n  Top-10 logits:" << std::endl;
        for (int r = 0; r < 10; ++r) {
            std::string dt = tokenizer_.decodeToken(top10[r].second);
            std::cout << "    [" << r << "] id=" << top10[r].second
                      << " logit=" << std::fixed << std::setprecision(4) << top10[r].first
                      << " text=\"";
            for (char c: dt) {
                if (c >= 32 && c < 127) std::cout << c;
                else
                    std::cout << "\\x" << std::hex << (static_cast<unsigned>(c) & 0xFF) << std::dec;
            }
            std::cout << "\"" << std::endl;
        }

        return logits;
    }

    std::pair<std::vector<float>, std::vector<float>>
    Model::debugForwardWithHidden(int32_t tokenId) {
        uint32_t hiddenSize = config_.hiddenSize;
        uint32_t nHeads = config_.numAttentionHeads;
        uint32_t nKVHeads = config_.numKVHeads;
        uint32_t headDim = config_.headDim;
        uint32_t nLayers = config_.numLayers;
        uint32_t maxSeqLen = config_.maxSeqLen;
        uint32_t vocabSize = config_.vocabSize;
        uint32_t intermediateSize = config_.intermediateSize;

        // ---- Step 1: Token Embedding ----
        std::vector<float> hidden(hiddenSize);
        if (tokenId >= 0 &&
            tokenId < static_cast<int32_t>(quantizedEmbeddings_.vocabSize)) {
            auto embRow = quantizedEmbeddings_.getRow(tokenId);
            for (uint32_t j = 0; j < hiddenSize; ++j) {
                hidden[j] = embRow[j];
            }
        }

        // Pre-allocate per-layer buffers
        std::vector<float> attnNorm(hiddenSize);
        std::vector<float> q(nHeads * headDim);
        std::vector<float> k(nKVHeads * headDim);
        std::vector<float> v(nKVHeads * headDim);
        std::vector<float> attnOut(nHeads * headDim);
        std::vector<float> attnProj(hiddenSize);
        std::vector<float> ffnNorm(hiddenSize);
        std::vector<float> gate(intermediateSize);
        std::vector<float> up(intermediateSize);
        std::vector<float> ffnOut(hiddenSize);

        // Process through all transformer layers
        for (uint32_t layer = 0; layer < nLayers; ++layer) {
            auto &w = layers_[layer];

            // ---- Attention block ----

            // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
            rmsNormInPlace(hidden.data(), attnNorm.data(), w.rmsNormAttn.data(), hiddenSize);

            // Q projection
            np::Array<float> qRow = w.attnQ.matMulVec(attnNorm.data());
            std::memcpy(q.data(), qRow.data(), nHeads * headDim * sizeof(float));
            if (!w.attnQBias.empty()) {
                const float *qBias = w.attnQBias.data();
                for (uint32_t i = 0; i < nHeads * headDim; ++i)
                    q[i] += qBias[i];
            }

            // K projection
            np::Array<float> kRow = w.attnK.matMulVec(attnNorm.data());
            std::memcpy(k.data(), kRow.data(), nKVHeads * headDim * sizeof(float));
            if (!w.attnKBias.empty()) {
                const float *kBias = w.attnKBias.data();
                for (uint32_t i = 0; i < nKVHeads * headDim; ++i)
                    k[i] += kBias[i];
            }

            // V projection
            np::Array<float> vRow = w.attnV.matMulVec(attnNorm.data());
            std::memcpy(v.data(), vRow.data(), nKVHeads * headDim * sizeof(float));
            if (!w.attnVBias.empty()) {
                const float *vBias = w.attnVBias.data();
                for (uint32_t i = 0; i < nKVHeads * headDim; ++i)
                    v[i] += vBias[i];
            }

            // Apply RoPE
            applyRoPE(q.data(), k.data(), 1, 1, nHeads, nKVHeads,
                      static_cast<uint32_t>(kvCache_.pos));

            // Store K, V in cache
            uint32_t cachePos = static_cast<uint32_t>(kvCache_.pos);
            float *kCacheLayer =
                    kvCache_.k.data() + layer * maxSeqLen * nKVHeads * headDim;
            float *vCacheLayer =
                    kvCache_.v.data() + layer * maxSeqLen * nKVHeads * headDim;
            uint32_t kvSize = nKVHeads * headDim;
            for (uint32_t i = 0; i < kvSize; ++i) {
                kCacheLayer[cachePos * nKVHeads * headDim + i] = k[i];
                vCacheLayer[cachePos * nKVHeads * headDim + i] = v[i];
            }

            // Attention
            uint32_t totalCacheLen = cachePos + 1;
            attentionFused(q.data(), kCacheLayer, vCacheLayer, attnOut.data(),
                           1, cachePos, totalCacheLen, layer);

            // Output projection (using dequantized F32 weights for exact float dot product)
            np::Array<float> projRow = deqMatMulVec(w.attnO_deq.data(), attnOut.data(),
                                                    w.attnO.rows, w.attnO.cols);
            std::memcpy(attnProj.data(), projRow.data(), hiddenSize * sizeof(float));

            // Residual: hidden += attnProj
            for (uint32_t i = 0; i < hiddenSize; ++i) {
                hidden[i] += attnProj[i];
            }

            // ---- FFN block ----

            // RMSNorm: ffnNorm = rmsNorm(hidden, w.rmsNormFFN)
            rmsNormInPlace(hidden.data(), ffnNorm.data(), w.rmsNormFFN.data(), hiddenSize);

            // Gate projection
            np::Array<float> gateRow = w.ffnGate.matMulVec(ffnNorm.data());
            std::memcpy(gate.data(), gateRow.data(), intermediateSize * sizeof(float));

            // Up projection
            np::Array<float> upRow = w.ffnUp.matMulVec(ffnNorm.data());
            std::memcpy(up.data(), upRow.data(), intermediateSize * sizeof(float));

            // SwiGLU: gate = silu(gate) * up
            swiGLUInPlace(gate.data(), up.data(), intermediateSize);

            // Down projection (using dequantized F32 weights for exact float dot product)
            np::Array<float> downRow = deqMatMulVec(w.ffnDown_deq.data(), gate.data(),
                                                    w.ffnDown.rows, w.ffnDown.cols);
            std::memcpy(ffnOut.data(), downRow.data(), hiddenSize * sizeof(float));

            // Residual: hidden += ffnOut
            for (uint32_t i = 0; i < hiddenSize; ++i) {
                hidden[i] += ffnOut[i];
            }
        }

        // Advance KV cache position after all layers processed
        kvCache_.pos++;

        // ---- Final RMSNorm ----
        rmsNormInPlace(hidden.data(), hidden.data(), finalNorm_.data(), hiddenSize);

        // Save the raw hidden state (after final RMSNorm, before L2 normalization).
        // The raw hidden state is used for LM head computation.
        // L2 normalization is only needed when comparing with the /v1/embeddings
        // endpoint, and is done in the test code, not here.
        std::vector<float> finalHidden(hidden.begin(), hidden.end());

        // ---- LM head (logits) ----
        std::vector<float> logits(vocabSize);
        if (lmHeadTied_) {
            // Use pre-dequantized embeddings for much faster computation
            if (!dequantizedEmbeddings_.empty()) {
                LMHead::computeCPU(hidden.data(),
                                   dequantizedEmbeddings_.data.data(),
                                   dequantizedEmbeddings_.vocabSize,
                                   dequantizedEmbeddings_.hiddenSize, logits.data());
            } else {
                // Fallback to quantized path
                LMHead::computeCPUQuantized(hidden.data(),
                                            quantizedEmbeddings_.data.data(),
                                            quantizedEmbeddings_.type,
                                            quantizedEmbeddings_.vocabSize,
                                            quantizedEmbeddings_.hiddenSize, logits.data());
            }
        } else {
            LMHead::computeCPUSeparate(hidden.data(), lmHead_.data.data(), lmHead_.type,
                                       lmHead_.rows, lmHead_.cols, logits.data());
        }

        return {std::move(finalHidden), std::move(logits)};
    }

    np::Array<float> Model::forward(const std::vector<int32_t> &tokens) {
        if (tokens.empty()) {
            return np::Array<float>{};
        }

        uint32_t seqLen = static_cast<uint32_t>(tokens.size());
        uint32_t hiddenSize = config_.hiddenSize;
        uint32_t nHeads = config_.numAttentionHeads;
        uint32_t nKVHeads = config_.numKVHeads;
        uint32_t headDim = config_.headDim;
        uint32_t nLayers = config_.numLayers;
        uint32_t maxSeqLen = config_.maxSeqLen;
        uint32_t vocabSize = config_.vocabSize;

        // Allocate hidden state: [seqLen, hiddenSize]
        np::Array<float> hidden = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *hiddenData = hidden.data();

        // Token embeddings: [seqLen, hiddenSize]
        // Dequantize from quantized format on-the-fly (parallel over tokens)
#pragma omp parallel for schedule(static)
        for (uint32_t i = 0; i < seqLen; ++i) {
            int32_t tokenId = tokens[i];
            if (tokenId >= 0 &&
                tokenId < static_cast<int32_t>(quantizedEmbeddings_.vocabSize)) {
                auto embRow = quantizedEmbeddings_.getRow(tokenId);
                float *hRow = hiddenData + i * hiddenSize;
                for (uint32_t j = 0; j < hiddenSize; ++j) {
                    hRow[j] = embRow[j];
                }
            }
        }

        // Pre-allocate per-layer buffers (reused across layers)
        // attnNorm: [seqLen, hiddenSize]
        np::Array<float> attnNorm = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *attnNormData = attnNorm.data();

        // Q, K, V projections
        np::Array<float> q = np::Array<float>(np::Shape{seqLen, nHeads, headDim});
        np::Array<float> k = np::Array<float>(np::Shape{seqLen, nKVHeads, headDim});
        np::Array<float> v = np::Array<float>(np::Shape{seqLen, nKVHeads, headDim});
        float *qData = q.data();
        float *kData = k.data();
        float *vData = v.data();

        // Attention output and projection
        np::Array<float> attnOut =
                np::Array<float>(np::Shape{seqLen, nHeads, headDim});
        float *attnOutData = attnOut.data();
        np::Array<float> attnProj = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *attnProjData = attnProj.data();

        // FFN buffers
        uint32_t intermediateSize = config_.intermediateSize;
        np::Array<float> ffnNorm = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *ffnNormData = ffnNorm.data();
        np::Array<float> gate = np::Array<float>(np::Shape{seqLen, intermediateSize});
        np::Array<float> up = np::Array<float>(np::Shape{seqLen, intermediateSize});
        float *gateData = gate.data();
        float *upData = up.data();
        np::Array<float> ffnOut = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *ffnOutData = ffnOut.data();

        // Debug: print initial hidden state stats for the last token
        if (seqLen > 1) {
            float hMin = hiddenData[(seqLen - 1) * hiddenSize], hMax = hiddenData[(seqLen - 1) * hiddenSize];
            float hSumSq = 0.0f;
            for (uint32_t i = 0; i < hiddenSize; ++i) {
                float v = hiddenData[(seqLen - 1) * hiddenSize + i];
                hMin = std::min(hMin, v);
                hMax = std::max(hMax, v);
                hSumSq += v * v;
            }
            float hNorm = std::sqrt(hSumSq / hiddenSize);
            std::cout << "[DEBUG] Layer init: last token hidden min=" << hMin << " max=" << hMax << " rms=" << hNorm << std::endl;
        }

        // Process through all transformer layers
        for (uint32_t layer = 0; layer < nLayers; ++layer) {
            auto &w = layers_[layer];

            // ---- Attention block ----

            // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
            // Each token's RMSNorm is independent.
            const float *rmsNormAttnData = w.rmsNormAttn.data();
#pragma omp parallel for schedule(static)
            for (uint32_t s = 0; s < seqLen; ++s) {
                rmsNormInPlace(hiddenData + s * hiddenSize, attnNormData + s * hiddenSize,
                               rmsNormAttnData, hiddenSize);
            }

            // Debug: print attnNorm stats for layer 0
            if (layer == 0 && seqLen > 1) {
                float nMin = attnNormData[(seqLen - 1) * hiddenSize], nMax = attnNormData[(seqLen - 1) * hiddenSize];
                float nSumSq = 0.0f;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = attnNormData[(seqLen - 1) * hiddenSize + i];
                    nMin = std::min(nMin, v);
                    nMax = std::max(nMax, v);
                    nSumSq += v * v;
                }
                float nNorm = std::sqrt(nSumSq / hiddenSize);
                std::cout << "[DEBUG] Layer 0 attnNorm: last token min=" << nMin << " max=" << nMax << " rms=" << nNorm << std::endl;
                // Also print first 8 values for comparison with test
                std::cout << "[DEBUG] Layer 0 attnNorm first 8: ";
                for (uint32_t i = 0; i < 8; ++i)
                    std::cout << std::fixed << std::setprecision(6) << attnNormData[(seqLen - 1) * hiddenSize + i] << " ";
                std::cout << std::endl;
                // Print hidden state first 8 for the last token
                std::cout << "[DEBUG] Layer 0 hidden (last token) first 8: ";
                for (uint32_t i = 0; i < 8; ++i)
                    std::cout << std::fixed << std::setprecision(6) << hiddenData[(seqLen - 1) * hiddenSize + i] << " ";
                std::cout << std::endl;
            }

            // Project to Q, K, V using quantized matrix-vector multiplication.
            // Each token's Q/K/V is computed independently — parallelize over tokens.
#pragma omp parallel for schedule(static)
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *hRowPtr = attnNormData + s * hiddenSize;

                // Q projection
                np::Array<float> qRow = w.attnQ.matMulVec(hRowPtr);
                float *qRowPtr = qData + s * nHeads * headDim;
                std::memcpy(qRowPtr, qRow.data(), nHeads * headDim * sizeof(float));
                if (!w.attnQBias.empty()) {
                    const float *qBias = w.attnQBias.data();
                    for (uint32_t i = 0; i < nHeads * headDim; ++i)
                        qRowPtr[i] += qBias[i];
                }

                // K projection
                np::Array<float> kRow = w.attnK.matMulVec(hRowPtr);
                float *kRowPtr = kData + s * nKVHeads * headDim;
                std::memcpy(kRowPtr, kRow.data(), nKVHeads * headDim * sizeof(float));
                if (!w.attnKBias.empty()) {
                    const float *kBias = w.attnKBias.data();
                    for (uint32_t i = 0; i < nKVHeads * headDim; ++i)
                        kRowPtr[i] += kBias[i];
                }

                // V projection
                np::Array<float> vRow = w.attnV.matMulVec(hRowPtr);
                float *vRowPtr = vData + s * nKVHeads * headDim;
                std::memcpy(vRowPtr, vRow.data(), nKVHeads * headDim * sizeof(float));
                if (!w.attnVBias.empty()) {
                    const float *vBias = w.attnVBias.data();
                    for (uint32_t i = 0; i < nKVHeads * headDim; ++i)
                        vRowPtr[i] += vBias[i];
                }
            }

            // Debug: print Q, K, V stats for layer 0
            if (layer == 0 && seqLen > 1) {
                auto printVecStats = [&](const float *data, uint32_t n, const std::string &label) {
                    float mn = data[0], mx = data[0];
                    double ssq = 0.0;
                    for (uint32_t i = 0; i < n; ++i) {
                        float v = data[i];
                        mn = std::min(mn, v);
                        mx = std::max(mx, v);
                        ssq += (double) v * v;
                    }
                    float r = std::sqrt((float) (ssq / n));
                    std::cout << "[DEBUG] Layer 0 " << label << ": last token min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                };
                printVecStats(qData + (seqLen - 1) * nHeads * headDim, nHeads * headDim, "Q (before RoPE)");
                printVecStats(kData + (seqLen - 1) * nKVHeads * headDim, nKVHeads * headDim, "K (before RoPE)");
                printVecStats(vData + (seqLen - 1) * nKVHeads * headDim, nKVHeads * headDim, "V");
            }

            // Apply RoPE
            applyRoPE(qData, kData, seqLen, seqLen, nHeads, nKVHeads,
                      static_cast<uint32_t>(kvCache_.pos));

            // Debug: print Q, K stats after RoPE for layer 0
            if (layer == 0 && seqLen > 1) {
                auto printVecStats = [&](const float *data, uint32_t n, const std::string &label) {
                    float mn = data[0], mx = data[0];
                    double ssq = 0.0;
                    for (uint32_t i = 0; i < n; ++i) {
                        float v = data[i];
                        mn = std::min(mn, v);
                        mx = std::max(mx, v);
                        ssq += (double) v * v;
                    }
                    float r = std::sqrt((float) (ssq / n));
                    std::cout << "[DEBUG] Layer 0 " << label << ": last token min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                };
                printVecStats(qData + (seqLen - 1) * nHeads * headDim, nHeads * headDim, "Q (after RoPE)");
                printVecStats(kData + (seqLen - 1) * nKVHeads * headDim, nKVHeads * headDim, "K (after RoPE)");
            }

            // Store K, V in cache (direct pointer access, no intermediate copies)
            uint32_t cachePos = static_cast<uint32_t>(kvCache_.pos);
            float *kCacheLayer =
                    kvCache_.k.data() + layer * maxSeqLen * nKVHeads * headDim;
            float *vCacheLayer =
                    kvCache_.v.data() + layer * maxSeqLen * nKVHeads * headDim;

            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *kSrc = kData + s * nKVHeads * headDim;
                const float *vSrc = vData + s * nKVHeads * headDim;
                float *kDst = kCacheLayer + (cachePos + s) * nKVHeads * headDim;
                float *vDst = vCacheLayer + (cachePos + s) * nKVHeads * headDim;
                uint32_t kvSize = nKVHeads * headDim;
                for (uint32_t i = 0; i < kvSize; ++i) {
                    kDst[i] = kSrc[i];
                    vDst[i] = vSrc[i];
                }
            }

            // Attention with cached K, V (read directly from cache, no intermediate
            // copy)
            uint32_t totalCacheLen = cachePos + seqLen;
            attentionFused(qData, kCacheLayer, vCacheLayer, attnOutData, seqLen,
                           cachePos, totalCacheLen, layer);

            // Debug: print attention output stats for layer 0
            if (layer == 0 && seqLen > 1) {
                float aMin = attnOutData[(seqLen - 1) * nHeads * headDim], aMax = attnOutData[(seqLen - 1) * nHeads * headDim];
                float aSumSq = 0.0f;
                for (uint32_t i = 0; i < nHeads * headDim; ++i) {
                    float v = attnOutData[(seqLen - 1) * nHeads * headDim + i];
                    aMin = std::min(aMin, v);
                    aMax = std::max(aMax, v);
                    aSumSq += v * v;
                }
                float aNorm = std::sqrt(aSumSq / (nHeads * headDim));
                std::cout << "[DEBUG] Layer 0 attnOut: last token min=" << aMin << " max=" << aMax << " rms=" << aNorm << std::endl;
            }

            // Output projection using quantized weights
            // attnOut: [seqLen, nHeads * headDim]
            // Output projection (using dequantized F32 weights for exact float dot product)
            // Each token's output projection is independent.
#pragma omp parallel for schedule(static)
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *attnRowPtr = attnOutData + s * nHeads * headDim;
                np::Array<float> projRow = deqMatMulVec(w.attnO_deq.data(), attnRowPtr,
                                                        w.attnO.rows, w.attnO.cols);
                float *projPtr = attnProjData + s * hiddenSize;
                std::memcpy(projPtr, projRow.data(), hiddenSize * sizeof(float));
            }

            // Residual connection: hidden += attnProj
            for (uint32_t s = 0; s < seqLen; ++s) {
                float *hPtr = hiddenData + s * hiddenSize;
                const float *aPtr = attnProjData + s * hiddenSize;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hPtr[i] += aPtr[i];
                }
            }

            // Debug: print attention projection stats for the last token
            if (seqLen > 1) {
                float aMin = attnProjData[(seqLen - 1) * hiddenSize], aMax = attnProjData[(seqLen - 1) * hiddenSize];
                float aSumSq = 0.0f;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = attnProjData[(seqLen - 1) * hiddenSize + i];
                    aMin = std::min(aMin, v);
                    aMax = std::max(aMax, v);
                    aSumSq += v * v;
                }
                float aNorm = std::sqrt(aSumSq / hiddenSize);
                std::cout << "[DEBUG] Layer " << layer << " attnProj: last token min=" << aMin << " max=" << aMax << " rms=" << aNorm << std::endl;
            }

            // ---- FFN block ----

            // RMSNorm: ffnNorm = rmsNorm(hidden, w.rmsNormFFN)
            // Each token's FFN norm is independent.
            const float *rmsNormFFNData = w.rmsNormFFN.data();
#pragma omp parallel for schedule(static)
            for (uint32_t s = 0; s < seqLen; ++s) {
                rmsNormInPlace(hiddenData + s * hiddenSize, ffnNormData + s * hiddenSize,
                               rmsNormFFNData, hiddenSize);
            }

            // Debug: print ffnNorm stats for layer 0
            if (layer == 0 && seqLen > 1) {
                float nMin = ffnNormData[(seqLen - 1) * hiddenSize], nMax = ffnNormData[(seqLen - 1) * hiddenSize];
                float nSumSq = 0.0f;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = ffnNormData[(seqLen - 1) * hiddenSize + i];
                    nMin = std::min(nMin, v);
                    nMax = std::max(nMax, v);
                    nSumSq += v * v;
                }
                float nNorm = std::sqrt(nSumSq / hiddenSize);
                std::cout << "[DEBUG] Layer 0 ffnNorm: last token min=" << nMin << " max=" << nMax << " rms=" << nNorm << std::endl;
            }

            // SwiGLU FFN using quantized weights
            // Fuse gate+up projections: each token's FFN is independent.
#pragma omp parallel for schedule(static)
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *ffnRowPtr = ffnNormData + s * hiddenSize;

                // Gate projection
                np::Array<float> gateRow = w.ffnGate.matMulVec(ffnRowPtr);
                float *gatePtr = gateData + s * intermediateSize;
                std::memcpy(gatePtr, gateRow.data(), intermediateSize * sizeof(float));

                // Up projection
                np::Array<float> upRow = w.ffnUp.matMulVec(ffnRowPtr);
                float *upPtr = upData + s * intermediateSize;
                std::memcpy(upPtr, upRow.data(), intermediateSize * sizeof(float));

                // SwiGLU activation in-place (fuse into projection loop)
                for (uint32_t i = 0; i < intermediateSize; ++i) {
                    gatePtr[i] = gatePtr[i] / (1.0f + std::exp(-gatePtr[i])) * upPtr[i];
                }
            }

            // Debug: print gate and up stats for layer 0
            if (layer == 0 && seqLen > 1) {
                auto printVecStats = [&](const float *data, uint32_t n, const std::string &label) {
                    float mn = data[0], mx = data[0];
                    double ssq = 0.0;
                    for (uint32_t i = 0; i < n; ++i) {
                        float v = data[i];
                        mn = std::min(mn, v);
                        mx = std::max(mx, v);
                        ssq += (double) v * v;
                    }
                    float r = std::sqrt((float) (ssq / n));
                    std::cout << "[DEBUG] Layer 0 " << label << ": last token min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                };
                printVecStats(gateData + (seqLen - 1) * intermediateSize, intermediateSize, "ffnGate+swiglu (merged)");
            }

            // Down projection (using dequantized F32 weights for exact float dot product)
            // Fuse with residual: hidden += ffnDown directly, avoiding extra copy
            // Each token's down projection is independent.
#pragma omp parallel for schedule(static)
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *ffnActPtr = gateData + s * intermediateSize;
                np::Array<float> downRow = deqMatMulVec(w.ffnDown_deq.data(), ffnActPtr,
                                                        w.ffnDown.rows, w.ffnDown.cols);
                float *hPtr = hiddenData + s * hiddenSize;
                const float *downData = downRow.data();
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hPtr[i] += downData[i];
                }
            }

            // Debug: print FFN output stats for the last token
            if (seqLen > 1) {
                float fMin = ffnOutData[(seqLen - 1) * hiddenSize], fMax = ffnOutData[(seqLen - 1) * hiddenSize];
                float fSumSq = 0.0f;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = ffnOutData[(seqLen - 1) * hiddenSize + i];
                    fMin = std::min(fMin, v);
                    fMax = std::max(fMax, v);
                    fSumSq += v * v;
                }
                float fNorm = std::sqrt(fSumSq / hiddenSize);
                std::cout << "[DEBUG] Layer " << layer << " ffnOut: last token min=" << fMin << " max=" << fMax << " rms=" << fNorm << std::endl;
            }

            // Residual connection: hidden += ffnOut
            for (uint32_t s = 0; s < seqLen; ++s) {
                float *hPtr = hiddenData + s * hiddenSize;
                const float *fPtr = ffnOutData + s * hiddenSize;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hPtr[i] += fPtr[i];
                }
            }

            // Debug: print hidden state stats for the last token after each layer
            if (seqLen > 1) {
                float hMin = hiddenData[(seqLen - 1) * hiddenSize], hMax = hiddenData[(seqLen - 1) * hiddenSize];
                float hSumSq = 0.0f;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = hiddenData[(seqLen - 1) * hiddenSize + i];
                    hMin = std::min(hMin, v);
                    hMax = std::max(hMax, v);
                    hSumSq += v * v;
                }
                float hNorm = std::sqrt(hSumSq / hiddenSize);
                std::cout << "[DEBUG] Layer " << layer << " after FFN: last token min=" << hMin << " max=" << hMax << " rms=" << hNorm << std::endl;
            }
        }

        // Final RMSNorm (parallel over tokens)
        const float *finalNormData = finalNorm_.data();
#pragma omp parallel for schedule(static)
        for (uint32_t s = 0; s < seqLen; ++s) {
            rmsNormInPlace(hiddenData + s * hiddenSize, hiddenData + s * hiddenSize,
                           finalNormData, hiddenSize);
        }

        // LM head (logits)
        np::Array<float> logits = np::Array<float>(np::Shape{seqLen, vocabSize});
        float *logitsData = logits.data();

        if (lmHeadTied_) {
            // Use token embeddings as LM head (weight tying)
            // Compute logits[s][i] = dot(hidden[s], embedding[i]) for all i
            // This is equivalent to: logits = hidden * embeddings^T
            // where embeddings is (vocabSize x hiddenSize) quantized matrix.
            //
            // For large vocabularies (151k+), this is the most expensive operation.
            // We use optimized paths:
            //   CUDA: cublasSgemv with persistent GPU embedding matrix
            //   CPU:  OpenMP parallelization over the vocabulary loop
#ifdef USE_CUDA
            // CUDA path: dequantize the full embedding matrix once, upload to GPU,
            // and use cublasSgemv for each token position.
            //
            // The embedding matrix is kept persistently on GPU so subsequent
            // forward passes don't need to re-upload it.
            static bool embedUploaded = false;
            static const float *d_embeddings = nullptr;
            static uint32_t cachedVocabSize = 0;
            static uint32_t cachedHiddenSize = 0;

            if (!embedUploaded || cachedVocabSize != quantizedEmbeddings_.vocabSize ||
                cachedHiddenSize != quantizedEmbeddings_.hiddenSize) {
                // Dequantize the full embedding matrix on CPU
                uint64_t embedElements =
                        static_cast<uint64_t>(quantizedEmbeddings_.vocabSize) *
                        quantizedEmbeddings_.hiddenSize;
                auto embedDeq = GGMLDequantize::dequantize(
                        quantizedEmbeddings_.type, quantizedEmbeddings_.data.data(),
                        embedElements);

                if (!embedDeq.empty()) {
                    try {
                        d_embeddings = cuda::uploadEmbeddings(embedDeq.data(), embedElements);
                        cachedVocabSize = quantizedEmbeddings_.vocabSize;
                        cachedHiddenSize = quantizedEmbeddings_.hiddenSize;
                        embedUploaded = true;
                    } catch (const std::exception &e) {
                        std::fprintf(stderr,
                                     "CUDA upload failed for LM head embeddings: %s\n",
                                     e.what());
                        embedUploaded = false;
                    }
                }
            }

            if (embedUploaded) {
                // Use cublasSgemv for each token position
                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    try {
                        cuda::computeLMHead(d_embeddings, cachedHiddenSize, cachedVocabSize,
                                            hPtr, logitRow);
                    } catch (const std::exception &e) {
                        std::fprintf(
                                stderr,
                                "CUDA cublasSgemv failed for LM head, falling back to CPU: %s\n",
                                e.what());
                        // Fallback to CPU with quantized computation
                        if (!dequantizedEmbeddings_.empty()) {
                            LMHead::computeCPU(hPtr,
                                               dequantizedEmbeddings_.data.data(),
                                               dequantizedEmbeddings_.vocabSize,
                                               dequantizedEmbeddings_.hiddenSize, logitRow);
                        } else {
                            LMHead::computeCPUQuantized(hPtr,
                                                        quantizedEmbeddings_.data.data(),
                                                        quantizedEmbeddings_.type,
                                                        quantizedEmbeddings_.vocabSize,
                                                        quantizedEmbeddings_.hiddenSize, logitRow);
                        }
                    }
                }
            } else {
                // Fallback to CPU with OpenMP parallelization
                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    if (!dequantizedEmbeddings_.empty()) {
                        LMHead::computeCPU(hPtr,
                                           dequantizedEmbeddings_.data.data(),
                                           dequantizedEmbeddings_.vocabSize,
                                           dequantizedEmbeddings_.hiddenSize, logitRow);
                    } else {
                        LMHead::computeCPUQuantized(hPtr,
                                                    quantizedEmbeddings_.data.data(),
                                                    quantizedEmbeddings_.type,
                                                    quantizedEmbeddings_.vocabSize,
                                                    quantizedEmbeddings_.hiddenSize, logitRow);
                    }
                }
            }
#else
            // CPU path with OpenMP parallelization over the vocabulary loop.
            // Uses pre-dequantized embeddings for speed.
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *hPtr = hiddenData + s * hiddenSize;
                float *logitRow = logitsData + s * vocabSize;
                if (!dequantizedEmbeddings_.empty()) {
                    LMHead::computeCPU(hPtr,
                                       dequantizedEmbeddings_.data.data(),
                                       dequantizedEmbeddings_.vocabSize,
                                       dequantizedEmbeddings_.hiddenSize, logitRow);
                } else {
                    LMHead::computeCPUQuantized(hPtr,
                                                quantizedEmbeddings_.data.data(),
                                                quantizedEmbeddings_.type,
                                                quantizedEmbeddings_.vocabSize,
                                                quantizedEmbeddings_.hiddenSize, logitRow);
                }
            }
#endif
        } else {
            // Separate LM head (quantized) with OpenMP parallelization
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *hiddenPtr = hiddenData + s * hiddenSize;
                float *logitRow = logitsData + s * vocabSize;
                LMHead::computeCPUSeparate(hiddenPtr, lmHead_.data.data(), lmHead_.type,
                                           lmHead_.rows, lmHead_.cols, logitRow);
            }
        }

        // Update KV cache position
        kvCache_.pos += seqLen;

        return logits;
    }

    std::pair<std::vector<float>, std::vector<float>>
    Model::forwardWithHidden(const std::vector<int32_t> &tokens) {
        if (tokens.empty()) {
            return {};
        }

        uint32_t seqLen = static_cast<uint32_t>(tokens.size());
        uint32_t hiddenSize = config_.hiddenSize;
        uint32_t nHeads = config_.numAttentionHeads;
        uint32_t nKVHeads = config_.numKVHeads;
        uint32_t headDim = config_.headDim;
        uint32_t nLayers = config_.numLayers;
        uint32_t maxSeqLen = config_.maxSeqLen;
        uint32_t vocabSize = config_.vocabSize;

        // Allocate hidden state: [seqLen, hiddenSize]
        np::Array<float> hidden = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *hiddenData = hidden.data();

        // Token embeddings
        for (uint32_t i = 0; i < seqLen; ++i) {
            int32_t tokenId = tokens[i];
            if (tokenId >= 0 &&
                tokenId < static_cast<int32_t>(quantizedEmbeddings_.vocabSize)) {
                auto embRow = quantizedEmbeddings_.getRow(tokenId);
                float *hRow = hiddenData + i * hiddenSize;
                for (uint32_t j = 0; j < hiddenSize; ++j) {
                    hRow[j] = embRow[j];
                }
            }
        }

        // Pre-allocate per-layer buffers
        np::Array<float> attnNorm = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *attnNormData = attnNorm.data();

        np::Array<float> q = np::Array<float>(np::Shape{seqLen, nHeads, headDim});
        np::Array<float> k = np::Array<float>(np::Shape{seqLen, nKVHeads, headDim});
        np::Array<float> v = np::Array<float>(np::Shape{seqLen, nKVHeads, headDim});
        float *qData = q.data();
        float *kData = k.data();
        float *vData = v.data();

        np::Array<float> attnOut = np::Array<float>(np::Shape{seqLen, nHeads, headDim});
        float *attnOutData = attnOut.data();
        np::Array<float> attnProj = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *attnProjData = attnProj.data();

        uint32_t intermediateSize = config_.intermediateSize;
        np::Array<float> ffnNorm = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *ffnNormData = ffnNorm.data();
        np::Array<float> gate = np::Array<float>(np::Shape{seqLen, intermediateSize});
        np::Array<float> up = np::Array<float>(np::Shape{seqLen, intermediateSize});
        float *gateData = gate.data();
        float *upData = up.data();
        np::Array<float> ffnOut = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *ffnOutData = ffnOut.data();

        // Process through all transformer layers
        for (uint32_t layer = 0; layer < nLayers; ++layer) {
            auto &w = layers_[layer];

            // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
            const float *rmsNormAttnData = w.rmsNormAttn.data();
            for (uint32_t s = 0; s < seqLen; ++s) {
                rmsNormInPlace(hiddenData + s * hiddenSize, attnNormData + s * hiddenSize,
                               rmsNormAttnData, hiddenSize);
            }

            // Q, K, V projections
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *hRowPtr = attnNormData + s * hiddenSize;

                // Q
                np::Array<float> qRow = w.attnQ.matMulVec(hRowPtr);
                float *qRowPtr = qData + s * nHeads * headDim;
                std::memcpy(qRowPtr, qRow.data(), nHeads * headDim * sizeof(float));
                if (!w.attnQBias.empty()) {
                    const float *qBias = w.attnQBias.data();
                    for (uint32_t i = 0; i < nHeads * headDim; ++i)
                        qRowPtr[i] += qBias[i];
                }

                // K
                np::Array<float> kRow = w.attnK.matMulVec(hRowPtr);
                float *kRowPtr = kData + s * nKVHeads * headDim;
                std::memcpy(kRowPtr, kRow.data(), nKVHeads * headDim * sizeof(float));
                if (!w.attnKBias.empty()) {
                    const float *kBias = w.attnKBias.data();
                    for (uint32_t i = 0; i < nKVHeads * headDim; ++i)
                        kRowPtr[i] += kBias[i];
                }

                // V
                np::Array<float> vRow = w.attnV.matMulVec(hRowPtr);
                float *vRowPtr = vData + s * nKVHeads * headDim;
                std::memcpy(vRowPtr, vRow.data(), nKVHeads * headDim * sizeof(float));
                if (!w.attnVBias.empty()) {
                    const float *vBias = w.attnVBias.data();
                    for (uint32_t i = 0; i < nKVHeads * headDim; ++i)
                        vRowPtr[i] += vBias[i];
                }
            }

            // Apply RoPE
            applyRoPE(qData, kData, seqLen, seqLen, nHeads, nKVHeads,
                      static_cast<uint32_t>(kvCache_.pos));

            // Store K, V in cache
            uint32_t cachePos = static_cast<uint32_t>(kvCache_.pos);
            float *kCacheLayer =
                    kvCache_.k.data() + layer * maxSeqLen * nKVHeads * headDim;
            float *vCacheLayer =
                    kvCache_.v.data() + layer * maxSeqLen * nKVHeads * headDim;

            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *kSrc = kData + s * nKVHeads * headDim;
                const float *vSrc = vData + s * nKVHeads * headDim;
                float *kDst = kCacheLayer + (cachePos + s) * nKVHeads * headDim;
                float *vDst = vCacheLayer + (cachePos + s) * nKVHeads * headDim;
                uint32_t kvSize = nKVHeads * headDim;
                for (uint32_t i = 0; i < kvSize; ++i) {
                    kDst[i] = kSrc[i];
                    vDst[i] = vSrc[i];
                }
            }

            // Attention
            uint32_t totalCacheLen = cachePos + seqLen;
            attentionFused(qData, kCacheLayer, vCacheLayer, attnOutData, seqLen,
                           cachePos, totalCacheLen, layer);

            // Output projection (using dequantized F32 weights for exact float dot product)
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *attnRowPtr = attnOutData + s * nHeads * headDim;
                np::Array<float> projRow = deqMatMulVec(w.attnO_deq.data(), attnRowPtr,
                                                        w.attnO.rows, w.attnO.cols);
                float *projPtr = attnProjData + s * hiddenSize;
                std::memcpy(projPtr, projRow.data(), hiddenSize * sizeof(float));
            }

            // Residual: hidden += attnProj
            for (uint32_t s = 0; s < seqLen; ++s) {
                float *hPtr = hiddenData + s * hiddenSize;
                const float *aPtr = attnProjData + s * hiddenSize;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hPtr[i] += aPtr[i];
                }
            }

            // ---- FFN block ----

            // RMSNorm: ffnNorm = rmsNorm(hidden, w.rmsNormFFN)
            const float *rmsNormFFNData = w.rmsNormFFN.data();
            for (uint32_t s = 0; s < seqLen; ++s) {
                rmsNormInPlace(hiddenData + s * hiddenSize, ffnNormData + s * hiddenSize,
                               rmsNormFFNData, hiddenSize);
            }

            // SwiGLU FFN
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *ffnRowPtr = ffnNormData + s * hiddenSize;

                // Gate projection
                np::Array<float> gateRow = w.ffnGate.matMulVec(ffnRowPtr);
                float *gatePtr = gateData + s * intermediateSize;
                std::memcpy(gatePtr, gateRow.data(), intermediateSize * sizeof(float));

                // Up projection
                np::Array<float> upRow = w.ffnUp.matMulVec(ffnRowPtr);
                float *upPtr = upData + s * intermediateSize;
                std::memcpy(upPtr, upRow.data(), intermediateSize * sizeof(float));
            }

            // SwiGLU activation
            swiGLUInPlace(gateData, upData, seqLen * intermediateSize);

            // Down projection (using dequantized F32 weights for exact float dot product)
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *ffnActPtr = gateData + s * intermediateSize;
                np::Array<float> downRow = deqMatMulVec(w.ffnDown_deq.data(), ffnActPtr,
                                                        w.ffnDown.rows, w.ffnDown.cols);
                float *outPtr = ffnOutData + s * hiddenSize;
                std::memcpy(outPtr, downRow.data(), hiddenSize * sizeof(float));
            }

            // Residual: hidden += ffnOut
            for (uint32_t s = 0; s < seqLen; ++s) {
                float *hPtr = hiddenData + s * hiddenSize;
                const float *fPtr = ffnOutData + s * hiddenSize;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hPtr[i] += fPtr[i];
                }
            }
        }

        // Final RMSNorm
        const float *finalNormData = finalNorm_.data();
        for (uint32_t s = 0; s < seqLen; ++s) {
            rmsNormInPlace(hiddenData + s * hiddenSize, hiddenData + s * hiddenSize,
                           finalNormData, hiddenSize);
        }

        // Save hidden state (after final RMSNorm, before LM head)
        std::vector<float> hiddenState(hiddenData, hiddenData + seqLen * hiddenSize);

        // LM head (logits)
        np::Array<float> logits = np::Array<float>(np::Shape{seqLen, vocabSize});
        float *logitsData = logits.data();

        if (lmHeadTied_) {
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *hPtr = hiddenData + s * hiddenSize;
                float *logitRow = logitsData + s * vocabSize;
                if (!dequantizedEmbeddings_.empty()) {
                    LMHead::computeCPU(hPtr,
                                       dequantizedEmbeddings_.data.data(),
                                       dequantizedEmbeddings_.vocabSize,
                                       dequantizedEmbeddings_.hiddenSize, logitRow);
                } else {
                    LMHead::computeCPUQuantized(hPtr,
                                                quantizedEmbeddings_.data.data(),
                                                quantizedEmbeddings_.type,
                                                quantizedEmbeddings_.vocabSize,
                                                quantizedEmbeddings_.hiddenSize, logitRow);
                }
            }
        } else {
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *hiddenPtr = hiddenData + s * hiddenSize;
                float *logitRow = logitsData + s * vocabSize;
                LMHead::computeCPUSeparate(hiddenPtr, lmHead_.data.data(), lmHead_.type,
                                           lmHead_.rows, lmHead_.cols, logitRow);
            }
        }

        // Update KV cache position
        kvCache_.pos += seqLen;

        // Return as flat vectors
        std::vector<float> logitsVec(logitsData, logitsData + seqLen * vocabSize);
        return {std::move(hiddenState), std::move(logitsVec)};
    }

    std::vector<float>
    Model::forwardHiddenOnly(const std::vector<int32_t> &tokens) {
        if (tokens.empty()) {
            return {};
        }

        uint32_t seqLen = static_cast<uint32_t>(tokens.size());
        uint32_t hiddenSize = config_.hiddenSize;
        uint32_t nHeads = config_.numAttentionHeads;
        uint32_t nKVHeads = config_.numKVHeads;
        uint32_t headDim = config_.headDim;
        uint32_t nLayers = config_.numLayers;
        uint32_t maxSeqLen = config_.maxSeqLen;

        // Allocate hidden state: [seqLen, hiddenSize]
        np::Array<float> hidden = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *hiddenData = hidden.data();

        // Token embeddings
        for (uint32_t i = 0; i < seqLen; ++i) {
            int32_t tokenId = tokens[i];
            if (tokenId >= 0 &&
                tokenId < static_cast<int32_t>(quantizedEmbeddings_.vocabSize)) {
                auto embRow = quantizedEmbeddings_.getRow(tokenId);
                float *hRow = hiddenData + i * hiddenSize;
                for (uint32_t j = 0; j < hiddenSize; ++j) {
                    hRow[j] = embRow[j];
                }
            }
        }

        // Pre-allocate per-layer buffers
        np::Array<float> attnNorm = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *attnNormData = attnNorm.data();

        np::Array<float> q = np::Array<float>(np::Shape{seqLen, nHeads, headDim});
        np::Array<float> k = np::Array<float>(np::Shape{seqLen, nKVHeads, headDim});
        np::Array<float> v = np::Array<float>(np::Shape{seqLen, nKVHeads, headDim});
        float *qData = q.data();
        float *kData = k.data();
        float *vData = v.data();

        np::Array<float> attnOut = np::Array<float>(np::Shape{seqLen, nHeads, headDim});
        float *attnOutData = attnOut.data();
        np::Array<float> attnProj = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *attnProjData = attnProj.data();

        uint32_t intermediateSize = config_.intermediateSize;
        np::Array<float> ffnNorm = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *ffnNormData = ffnNorm.data();
        np::Array<float> gate = np::Array<float>(np::Shape{seqLen, intermediateSize});
        np::Array<float> up = np::Array<float>(np::Shape{seqLen, intermediateSize});
        float *gateData = gate.data();
        float *upData = up.data();
        np::Array<float> ffnOut = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *ffnOutData = ffnOut.data();

        // Process through all transformer layers
        for (uint32_t layer = 0; layer < nLayers; ++layer) {
            auto &w = layers_[layer];

            // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
            const float *rmsNormAttnData = w.rmsNormAttn.data();
            for (uint32_t s = 0; s < seqLen; ++s) {
                rmsNormInPlace(hiddenData + s * hiddenSize, attnNormData + s * hiddenSize,
                               rmsNormAttnData, hiddenSize);
            }

            // Debug: print attnNorm stats for Layer 0
            if (layer == 0 && seqLen > 1) {
                float nMin = attnNormData[(seqLen - 1) * hiddenSize], nMax = attnNormData[(seqLen - 1) * hiddenSize];
                double nSumSq = 0.0;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = attnNormData[(seqLen - 1) * hiddenSize + i];
                    nMin = std::min(nMin, v);
                    nMax = std::max(nMax, v);
                    nSumSq += (double) v * v;
                }
                float nNorm = std::sqrt((float) (nSumSq / hiddenSize));
                std::cout << "[DEBUG] Layer 0 attnNorm: last token min=" << nMin << " max=" << nMax << " rms=" << nNorm << std::endl;
                // Print first 8 values for comparison with test
                std::cout << "[DEBUG] Layer 0 attnNorm first 8: ";
                for (uint32_t i = 0; i < 8; ++i)
                    std::cout << std::fixed << std::setprecision(6) << attnNormData[(seqLen - 1) * hiddenSize + i] << " ";
                std::cout << std::endl;
                // Also print embedding first 8 for the last token
                std::cout << "[DEBUG] Layer 0 embedding first 8: ";
                for (uint32_t i = 0; i < 8; ++i)
                    std::cout << std::fixed << std::setprecision(6) << hiddenData[(seqLen - 1) * hiddenSize + i] << " ";
                std::cout << std::endl;
            }

            // Q, K, V projections
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *hRowPtr = attnNormData + s * hiddenSize;

                // Q
                np::Array<float> qRow = w.attnQ.matMulVec(hRowPtr);
                float *qRowPtr = qData + s * nHeads * headDim;
                std::memcpy(qRowPtr, qRow.data(), nHeads * headDim * sizeof(float));
                if (!w.attnQBias.empty()) {
                    const float *qBias = w.attnQBias.data();
                    for (uint32_t i = 0; i < nHeads * headDim; ++i)
                        qRowPtr[i] += qBias[i];
                }

                // K
                np::Array<float> kRow = w.attnK.matMulVec(hRowPtr);
                float *kRowPtr = kData + s * nKVHeads * headDim;
                std::memcpy(kRowPtr, kRow.data(), nKVHeads * headDim * sizeof(float));
                if (!w.attnKBias.empty()) {
                    const float *kBias = w.attnKBias.data();
                    for (uint32_t i = 0; i < nKVHeads * headDim; ++i)
                        kRowPtr[i] += kBias[i];
                }

                // V
                np::Array<float> vRow = w.attnV.matMulVec(hRowPtr);
                float *vRowPtr = vData + s * nKVHeads * headDim;
                std::memcpy(vRowPtr, vRow.data(), nKVHeads * headDim * sizeof(float));
                if (!w.attnVBias.empty()) {
                    const float *vBias = w.attnVBias.data();
                    for (uint32_t i = 0; i < nKVHeads * headDim; ++i)
                        vRowPtr[i] += vBias[i];
                }

                // DIAGNOSTIC: For layer 0 last token, compare raw matMulVec result (before bias)
                // with direct matMulVecFused call to verify correctness
                if (layer == 0 && s == seqLen - 1) {
                    // qRow contains the raw matMulVec result (before bias)
                    const float *qRaw = qRow.data();
                    float qRawRms = 0.0f;
                    for (uint32_t i = 0; i < w.attnQ.rows; ++i) qRawRms += qRaw[i] * qRaw[i];
                    qRawRms = std::sqrt(qRawRms / w.attnQ.rows);
                    std::cout << "[DIAG] Q raw (before bias) rms=" << qRawRms << std::endl;
                    std::cout << "[DIAG] Q raw first 8: ";
                    for (uint32_t i = 0; i < 8; ++i) std::cout << std::fixed << std::setprecision(6) << qRaw[i] << " ";
                    std::cout << std::endl;
                    // qRowPtr has bias added
                    std::cout << "[DIAG] Q with bias first 8: ";
                    for (uint32_t i = 0; i < 8; ++i) std::cout << std::fixed << std::setprecision(6) << qRowPtr[i] << " ";
                    std::cout << std::endl;
                    // Compare with reference: compute using dequantize+dot
                    auto deqQ = GGMLDequantize::dequantize(w.attnQ.type, w.attnQ.data.data(),
                                                           static_cast<uint64_t>(w.attnQ.rows) * w.attnQ.cols);
                    std::vector<float> qRef(w.attnQ.rows, 0.0f);
                    for (uint32_t j = 0; j < w.attnQ.rows; ++j) {
                        double dot = 0.0;
                        for (uint32_t i = 0; i < w.attnQ.cols; ++i)
                            dot += (double) hRowPtr[i] * deqQ[static_cast<size_t>(j) * w.attnQ.cols + i];
                        qRef[j] = (float) dot;
                    }
                    float qRefRms = 0.0f;
                    for (uint32_t i = 0; i < w.attnQ.rows; ++i) qRefRms += qRef[i] * qRef[i];
                    qRefRms = std::sqrt(qRefRms / w.attnQ.rows);
                    bool qMatch = true;
                    for (uint32_t i = 0; i < w.attnQ.rows; ++i) {
                        if (std::abs(qRaw[i] - qRef[i]) > 1e-4f) {
                            qMatch = false;
                            break;
                        }
                    }
                    std::cout << "[DIAG] Q ref (deq+dot) rms=" << qRefRms << " match=" << (qMatch ? "YES" : "NO") << std::endl;
                    if (!qMatch) {
                        std::cout << "[DIAG] Q ref first 8: ";
                        for (uint32_t i = 0; i < 8; ++i) std::cout << std::fixed << std::setprecision(6) << qRef[i] << " ";
                        std::cout << std::endl;
                    }

                    // Same for K
                    const float *kRaw = kRow.data();
                    float kRawRms = 0.0f;
                    for (uint32_t i = 0; i < w.attnK.rows; ++i) kRawRms += kRaw[i] * kRaw[i];
                    kRawRms = std::sqrt(kRawRms / w.attnK.rows);
                    std::cout << "[DIAG] K raw (before bias) rms=" << kRawRms << std::endl;
                    std::cout << "[DIAG] K raw first 8: ";
                    for (uint32_t i = 0; i < 8; ++i) std::cout << std::fixed << std::setprecision(6) << kRaw[i] << " ";
                    std::cout << std::endl;
                    auto deqK = GGMLDequantize::dequantize(w.attnK.type, w.attnK.data.data(),
                                                           static_cast<uint64_t>(w.attnK.rows) * w.attnK.cols);
                    std::vector<float> kRef(w.attnK.rows, 0.0f);
                    for (uint32_t j = 0; j < w.attnK.rows; ++j) {
                        double dot = 0.0;
                        for (uint32_t i = 0; i < w.attnK.cols; ++i)
                            dot += (double) hRowPtr[i] * deqK[static_cast<size_t>(j) * w.attnK.cols + i];
                        kRef[j] = (float) dot;
                    }
                    float kRefRms = 0.0f;
                    for (uint32_t i = 0; i < w.attnK.rows; ++i) kRefRms += kRef[i] * kRef[i];
                    kRefRms = std::sqrt(kRefRms / w.attnK.rows);
                    bool kMatch = true;
                    for (uint32_t i = 0; i < w.attnK.rows; ++i) {
                        if (std::abs(kRaw[i] - kRef[i]) > 1e-4f) {
                            kMatch = false;
                            break;
                        }
                    }
                    std::cout << "[DIAG] K ref (deq+dot) rms=" << kRefRms << " match=" << (kMatch ? "YES" : "NO") << std::endl;
                    if (!kMatch) {
                        std::cout << "[DIAG] K ref first 8: ";
                        for (uint32_t i = 0; i < 8; ++i) std::cout << std::fixed << std::setprecision(6) << kRef[i] << " ";
                        std::cout << std::endl;
                    }

                    // Same for V
                    const float *vRaw = vRow.data();
                    float vRawRms = 0.0f;
                    for (uint32_t i = 0; i < w.attnV.rows; ++i) vRawRms += vRaw[i] * vRaw[i];
                    vRawRms = std::sqrt(vRawRms / w.attnV.rows);
                    std::cout << "[DIAG] V raw (before bias) rms=" << vRawRms << std::endl;
                    auto deqV = GGMLDequantize::dequantize(w.attnV.type, w.attnV.data.data(),
                                                           static_cast<uint64_t>(w.attnV.rows) * w.attnV.cols);
                    std::vector<float> vRef(w.attnV.rows, 0.0f);
                    for (uint32_t j = 0; j < w.attnV.rows; ++j) {
                        double dot = 0.0;
                        for (uint32_t i = 0; i < w.attnV.cols; ++i)
                            dot += (double) hRowPtr[i] * deqV[static_cast<size_t>(j) * w.attnV.cols + i];
                        vRef[j] = (float) dot;
                    }
                    float vRefRms = 0.0f;
                    for (uint32_t i = 0; i < w.attnV.rows; ++i) vRefRms += vRef[i] * vRef[i];
                    vRefRms = std::sqrt(vRefRms / w.attnV.rows);
                    bool vMatch = true;
                    for (uint32_t i = 0; i < w.attnV.rows; ++i) {
                        if (std::abs(vRaw[i] - vRef[i]) > 1e-4f) {
                            vMatch = false;
                            break;
                        }
                    }
                    std::cout << "[DIAG] V ref (deq+dot) rms=" << vRefRms << " match=" << (vMatch ? "YES" : "NO") << std::endl;
                }
            }

            // Debug: print Q, K, V stats and first-8 values for Layer 0
            if (layer == 0 && seqLen > 1) {
                auto printVecStats = [&](const float *data, uint32_t n, const std::string &label) {
                    float mn = data[0], mx = data[0];
                    double ssq = 0.0;
                    for (uint32_t i = 0; i < n; ++i) {
                        float v = data[i];
                        mn = std::min(mn, v);
                        mx = std::max(mx, v);
                        ssq += (double) v * v;
                    }
                    float r = std::sqrt((float) (ssq / n));
                    std::cout << "[DEBUG] Layer 0 " << label << ": last token min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                };
                // Print attnNorm input used for Q/K/V (should match test)
                std::cout << "[DEBUG] Layer 0 attnNorm (input to Q/K/V) first 8: ";
                for (uint32_t i = 0; i < 8; ++i)
                    std::cout << std::fixed << std::setprecision(6) << attnNormData[(seqLen - 1) * hiddenSize + i] << " ";
                std::cout << std::endl;
                // Q stats + first 8 values (before bias)
                printVecStats(qData + (seqLen - 1) * nHeads * headDim, nHeads * headDim, "Q (before RoPE)");
                std::cout << "[DEBUG] Layer 0 Q (before bias) first 8: ";
                for (uint32_t i = 0; i < 8; ++i)
                    std::cout << std::fixed << std::setprecision(6) << qData[(seqLen - 1) * nHeads * headDim + i] << " ";
                std::cout << std::endl;
                // K stats + first 8 values (before bias)
                printVecStats(kData + (seqLen - 1) * nKVHeads * headDim, nKVHeads * headDim, "K (before RoPE)");
                std::cout << "[DEBUG] Layer 0 K (before bias) first 8: ";
                for (uint32_t i = 0; i < 8; ++i)
                    std::cout << std::fixed << std::setprecision(6) << kData[(seqLen - 1) * nKVHeads * headDim + i] << " ";
                std::cout << std::endl;
                // V stats + first 8 values
                printVecStats(vData + (seqLen - 1) * nKVHeads * headDim, nKVHeads * headDim, "V");
                std::cout << "[DEBUG] Layer 0 V first 8: ";
                for (uint32_t i = 0; i < 8; ++i)
                    std::cout << std::fixed << std::setprecision(6) << vData[(seqLen - 1) * nKVHeads * headDim + i] << " ";
                std::cout << std::endl;
            }

            // Apply RoPE
            applyRoPE(qData, kData, seqLen, seqLen, nHeads, nKVHeads,
                      static_cast<uint32_t>(kvCache_.pos));

            // Debug: print Q, K stats after RoPE for Layer 0
            if (layer == 0 && seqLen > 1) {
                auto printVecStats = [&](const float *data, uint32_t n, const std::string &label) {
                    float mn = data[0], mx = data[0];
                    double ssq = 0.0;
                    for (uint32_t i = 0; i < n; ++i) {
                        float v = data[i];
                        mn = std::min(mn, v);
                        mx = std::max(mx, v);
                        ssq += (double) v * v;
                    }
                    float r = std::sqrt((float) (ssq / n));
                    std::cout << "[DEBUG] Layer 0 " << label << ": last token min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                };
                printVecStats(qData + (seqLen - 1) * nHeads * headDim, nHeads * headDim, "Q (after RoPE)");
                printVecStats(kData + (seqLen - 1) * nKVHeads * headDim, nKVHeads * headDim, "K (after RoPE)");
            }

            // Store K, V in cache
            uint32_t cachePos = static_cast<uint32_t>(kvCache_.pos);
            float *kCacheLayer =
                    kvCache_.k.data() + layer * maxSeqLen * nKVHeads * headDim;
            float *vCacheLayer =
                    kvCache_.v.data() + layer * maxSeqLen * nKVHeads * headDim;

            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *kSrc = kData + s * nKVHeads * headDim;
                const float *vSrc = vData + s * nKVHeads * headDim;
                float *kDst = kCacheLayer + (cachePos + s) * nKVHeads * headDim;
                float *vDst = vCacheLayer + (cachePos + s) * nKVHeads * headDim;
                uint32_t kvSize = nKVHeads * headDim;
                for (uint32_t i = 0; i < kvSize; ++i) {
                    kDst[i] = kSrc[i];
                    vDst[i] = vSrc[i];
                }
            }

            // Attention
            uint32_t totalCacheLen = cachePos + seqLen;
            attentionFused(qData, kCacheLayer, vCacheLayer, attnOutData, seqLen,
                           cachePos, totalCacheLen, layer);

            // Debug: print attention output stats for Layer 0
            if (layer == 0 && seqLen > 1) {
                float aMin = attnOutData[(seqLen - 1) * nHeads * headDim], aMax = attnOutData[(seqLen - 1) * nHeads * headDim];
                double aSumSq = 0.0;
                for (uint32_t i = 0; i < nHeads * headDim; ++i) {
                    float v = attnOutData[(seqLen - 1) * nHeads * headDim + i];
                    aMin = std::min(aMin, v);
                    aMax = std::max(aMax, v);
                    aSumSq += (double) v * v;
                }
                float aNorm = std::sqrt((float) (aSumSq / (nHeads * headDim)));
                std::cout << "[DEBUG] Layer 0 attnOut: last token min=" << aMin << " max=" << aMax << " rms=" << aNorm << std::endl;
            }

            // Output projection (using dequantized F32 weights for exact float dot product)
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *attnRowPtr = attnOutData + s * nHeads * headDim;
                np::Array<float> projRow = deqMatMulVec(w.attnO_deq.data(), attnRowPtr,
                                                        w.attnO.rows, w.attnO.cols);
                float *projPtr = attnProjData + s * hiddenSize;
                std::memcpy(projPtr, projRow.data(), hiddenSize * sizeof(float));
            }

            // Debug: print attention projection stats for Layer 0
            if (layer == 0 && seqLen > 1) {
                float aMin = attnProjData[(seqLen - 1) * hiddenSize], aMax = attnProjData[(seqLen - 1) * hiddenSize];
                double aSumSq = 0.0;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = attnProjData[(seqLen - 1) * hiddenSize + i];
                    aMin = std::min(aMin, v);
                    aMax = std::max(aMax, v);
                    aSumSq += (double) v * v;
                }
                float aNorm = std::sqrt((float) (aSumSq / hiddenSize));
                std::cout << "[DEBUG] Layer 0 attnProj: last token min=" << aMin << " max=" << aMax << " rms=" << aNorm << std::endl;
            }

            // Residual: hidden += attnProj
            for (uint32_t s = 0; s < seqLen; ++s) {
                float *hPtr = hiddenData + s * hiddenSize;
                const float *aPtr = attnProjData + s * hiddenSize;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hPtr[i] += aPtr[i];
                }
            }

            // ---- FFN block ----

            // RMSNorm: ffnNorm = rmsNorm(hidden, w.rmsNormFFN)
            const float *rmsNormFFNData = w.rmsNormFFN.data();
            for (uint32_t s = 0; s < seqLen; ++s) {
                rmsNormInPlace(hiddenData + s * hiddenSize, ffnNormData + s * hiddenSize,
                               rmsNormFFNData, hiddenSize);
            }

            // Debug: print ffnNorm stats for Layer 0
            if (layer == 0 && seqLen > 1) {
                float nMin = ffnNormData[(seqLen - 1) * hiddenSize], nMax = ffnNormData[(seqLen - 1) * hiddenSize];
                double nSumSq = 0.0;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = ffnNormData[(seqLen - 1) * hiddenSize + i];
                    nMin = std::min(nMin, v);
                    nMax = std::max(nMax, v);
                    nSumSq += (double) v * v;
                }
                float nNorm = std::sqrt((float) (nSumSq / hiddenSize));
                std::cout << "[DEBUG] Layer 0 ffnNorm: last token min=" << nMin << " max=" << nMax << " rms=" << nNorm << std::endl;
            }

            // SwiGLU FFN
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *ffnRowPtr = ffnNormData + s * hiddenSize;

                // Gate projection
                np::Array<float> gateRow = w.ffnGate.matMulVec(ffnRowPtr);
                float *gatePtr = gateData + s * intermediateSize;
                std::memcpy(gatePtr, gateRow.data(), intermediateSize * sizeof(float));

                // Up projection
                np::Array<float> upRow = w.ffnUp.matMulVec(ffnRowPtr);
                float *upPtr = upData + s * intermediateSize;
                std::memcpy(upPtr, upRow.data(), intermediateSize * sizeof(float));
            }

            // Debug: print gate and up stats for Layer 0
            if (layer == 0 && seqLen > 1) {
                auto printVecStats = [&](const float *data, uint32_t n, const std::string &label) {
                    float mn = data[0], mx = data[0];
                    double ssq = 0.0;
                    for (uint32_t i = 0; i < n; ++i) {
                        float v = data[i];
                        mn = std::min(mn, v);
                        mx = std::max(mx, v);
                        ssq += (double) v * v;
                    }
                    float r = std::sqrt((float) (ssq / n));
                    std::cout << "[DEBUG] Layer 0 " << label << ": last token min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                };
                printVecStats(gateData + (seqLen - 1) * intermediateSize, intermediateSize, "ffnGate (before SwiGLU)");
                printVecStats(upData + (seqLen - 1) * intermediateSize, intermediateSize, "ffnUp");
            }

            // SwiGLU activation
            swiGLUInPlace(gateData, upData, seqLen * intermediateSize);

            // Debug: print SwiGLU output stats for Layer 0
            if (layer == 0 && seqLen > 1) {
                float gMin = gateData[(seqLen - 1) * intermediateSize], gMax = gateData[(seqLen - 1) * intermediateSize];
                double gSumSq = 0.0;
                for (uint32_t i = 0; i < intermediateSize; ++i) {
                    float v = gateData[(seqLen - 1) * intermediateSize + i];
                    gMin = std::min(gMin, v);
                    gMax = std::max(gMax, v);
                    gSumSq += (double) v * v;
                }
                float gNorm = std::sqrt((float) (gSumSq / intermediateSize));
                std::cout << "[DEBUG] Layer 0 swigluOut: last token min=" << gMin << " max=" << gMax << " rms=" << gNorm << std::endl;
            }

            // Down projection (using dequantized F32 weights for exact float dot product)
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *ffnActPtr = gateData + s * intermediateSize;
                np::Array<float> downRow = deqMatMulVec(w.ffnDown_deq.data(), ffnActPtr,
                                                        w.ffnDown.rows, w.ffnDown.cols);
                float *outPtr = ffnOutData + s * hiddenSize;
                std::memcpy(outPtr, downRow.data(), hiddenSize * sizeof(float));
            }

            // Debug: print FFN down projection stats for Layer 0
            if (layer == 0 && seqLen > 1) {
                float fMin = ffnOutData[(seqLen - 1) * hiddenSize], fMax = ffnOutData[(seqLen - 1) * hiddenSize];
                double fSumSq = 0.0;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = ffnOutData[(seqLen - 1) * hiddenSize + i];
                    fMin = std::min(fMin, v);
                    fMax = std::max(fMax, v);
                    fSumSq += (double) v * v;
                }
                float fNorm = std::sqrt((float) (fSumSq / hiddenSize));
                std::cout << "[DEBUG] Layer 0 ffnDown: last token min=" << fMin << " max=" << fMax << " rms=" << fNorm << std::endl;
            }

            // Residual: hidden += ffnOut
            for (uint32_t s = 0; s < seqLen; ++s) {
                float *hPtr = hiddenData + s * hiddenSize;
                const float *fPtr = ffnOutData + s * hiddenSize;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hPtr[i] += fPtr[i];
                }
            }

            // Debug: print hidden state stats for the last token after each layer
            if (seqLen > 1) {
                float hMin = hiddenData[(seqLen - 1) * hiddenSize], hMax = hiddenData[(seqLen - 1) * hiddenSize];
                float hSumSq = 0.0f;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = hiddenData[(seqLen - 1) * hiddenSize + i];
                    hMin = std::min(hMin, v);
                    hMax = std::max(hMax, v);
                    hSumSq += v * v;
                }
                float hNorm = std::sqrt(hSumSq / hiddenSize);
                std::cout << "[DEBUG] Layer " << layer << " after FFN: last token min=" << hMin << " max=" << hMax << " rms=" << hNorm << std::endl;
            }
        }

        // Final RMSNorm
        const float *finalNormData = finalNorm_.data();
        for (uint32_t s = 0; s < seqLen; ++s) {
            rmsNormInPlace(hiddenData + s * hiddenSize, hiddenData + s * hiddenSize,
                           finalNormData, hiddenSize);
        }

        // Update KV cache position
        kvCache_.pos += seqLen;

        // Return hidden state as flat vector (no LM head computation)
        return std::vector<float>(hiddenData, hiddenData + seqLen * hiddenSize);
    }

    std::vector<std::vector<float>>
    Model::forwardWithPerLayerStates(const std::vector<int32_t> &tokens) {
        if (tokens.empty()) {
            return {};
        }

        uint32_t seqLen = static_cast<uint32_t>(tokens.size());
        uint32_t hiddenSize = config_.hiddenSize;
        uint32_t nHeads = config_.numAttentionHeads;
        uint32_t nKVHeads = config_.numKVHeads;
        uint32_t headDim = config_.headDim;
        uint32_t nLayers = config_.numLayers;
        uint32_t maxSeqLen = config_.maxSeqLen;

        // Allocate hidden state: [seqLen, hiddenSize]
        np::Array<float> hidden = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *hiddenData = hidden.data();

        // Token embeddings
        for (uint32_t i = 0; i < seqLen; ++i) {
            int32_t tokenId = tokens[i];
            if (tokenId >= 0 &&
                tokenId < static_cast<int32_t>(quantizedEmbeddings_.vocabSize)) {
                auto embRow = quantizedEmbeddings_.getRow(tokenId);
                float *hRow = hiddenData + i * hiddenSize;
                for (uint32_t j = 0; j < hiddenSize; ++j) {
                    hRow[j] = embRow[j];
                }
            }
        }

        // Pre-allocate per-layer buffers
        np::Array<float> attnNorm = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *attnNormData = attnNorm.data();

        np::Array<float> q = np::Array<float>(np::Shape{seqLen, nHeads, headDim});
        np::Array<float> k = np::Array<float>(np::Shape{seqLen, nKVHeads, headDim});
        np::Array<float> v = np::Array<float>(np::Shape{seqLen, nKVHeads, headDim});
        float *qData = q.data();
        float *kData = k.data();
        float *vData = v.data();

        np::Array<float> attnOut = np::Array<float>(np::Shape{seqLen, nHeads, headDim});
        float *attnOutData = attnOut.data();
        np::Array<float> attnProj = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *attnProjData = attnProj.data();

        uint32_t intermediateSize = config_.intermediateSize;
        np::Array<float> ffnNorm = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *ffnNormData = ffnNorm.data();
        np::Array<float> gate = np::Array<float>(np::Shape{seqLen, intermediateSize});
        np::Array<float> up = np::Array<float>(np::Shape{seqLen, intermediateSize});
        float *gateData = gate.data();
        float *upData = up.data();
        np::Array<float> ffnOut = np::Array<float>(np::Shape{seqLen, hiddenSize});
        float *ffnOutData = ffnOut.data();

        // Per-layer hidden state capture
        std::vector<std::vector<float>> perLayerStates;
        perLayerStates.reserve(nLayers);

        // Process through all transformer layers
        for (uint32_t layer = 0; layer < nLayers; ++layer) {
            auto &w = layers_[layer];

            // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
            const float *rmsNormAttnData = w.rmsNormAttn.data();
            for (uint32_t s = 0; s < seqLen; ++s) {
                rmsNormInPlace(hiddenData + s * hiddenSize, attnNormData + s * hiddenSize,
                               rmsNormAttnData, hiddenSize);
            }

            // Debug: print attnNorm stats for Layer 0
            if (layer == 0 && seqLen > 1) {
                float nMin = attnNormData[(seqLen - 1) * hiddenSize], nMax = attnNormData[(seqLen - 1) * hiddenSize];
                double nSumSq = 0.0;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = attnNormData[(seqLen - 1) * hiddenSize + i];
                    nMin = std::min(nMin, v);
                    nMax = std::max(nMax, v);
                    nSumSq += (double) v * v;
                }
                float nNorm = std::sqrt((float) (nSumSq / hiddenSize));
                std::cout << "[DEBUG-PL] Layer 0 attnNorm: last token min=" << nMin << " max=" << nMax << " rms=" << nNorm << std::endl;
            }

            // Q, K, V projections
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *hRowPtr = attnNormData + s * hiddenSize;

                // Q
                np::Array<float> qRow = w.attnQ.matMulVec(hRowPtr);
                float *qRowPtr = qData + s * nHeads * headDim;
                std::memcpy(qRowPtr, qRow.data(), nHeads * headDim * sizeof(float));
                if (!w.attnQBias.empty()) {
                    const float *qBias = w.attnQBias.data();
                    for (uint32_t i = 0; i < nHeads * headDim; ++i)
                        qRowPtr[i] += qBias[i];
                }

                // K
                np::Array<float> kRow = w.attnK.matMulVec(hRowPtr);
                float *kRowPtr = kData + s * nKVHeads * headDim;
                std::memcpy(kRowPtr, kRow.data(), nKVHeads * headDim * sizeof(float));
                if (!w.attnKBias.empty()) {
                    const float *kBias = w.attnKBias.data();
                    for (uint32_t i = 0; i < nKVHeads * headDim; ++i)
                        kRowPtr[i] += kBias[i];
                }

                // V
                np::Array<float> vRow = w.attnV.matMulVec(hRowPtr);
                float *vRowPtr = vData + s * nKVHeads * headDim;
                std::memcpy(vRowPtr, vRow.data(), nKVHeads * headDim * sizeof(float));
                if (!w.attnVBias.empty()) {
                    const float *vBias = w.attnVBias.data();
                    for (uint32_t i = 0; i < nKVHeads * headDim; ++i)
                        vRowPtr[i] += vBias[i];
                }
            }

            // Debug: print Q, K, V stats for Layer 0
            if (layer == 0 && seqLen > 1) {
                auto printVecStats = [&](const float *data, uint32_t n, const std::string &label) {
                    float mn = data[0], mx = data[0];
                    double ssq = 0.0;
                    for (uint32_t i = 0; i < n; ++i) {
                        float v = data[i];
                        mn = std::min(mn, v);
                        mx = std::max(mx, v);
                        ssq += (double) v * v;
                    }
                    float r = std::sqrt((float) (ssq / n));
                    std::cout << "[DEBUG-PL] Layer 0 " << label << ": last token min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                };
                printVecStats(qData + (seqLen - 1) * nHeads * headDim, nHeads * headDim, "Q (before RoPE)");
                printVecStats(kData + (seqLen - 1) * nKVHeads * headDim, nKVHeads * headDim, "K (before RoPE)");
                printVecStats(vData + (seqLen - 1) * nKVHeads * headDim, nKVHeads * headDim, "V");
            }

            // Apply RoPE
            applyRoPE(qData, kData, seqLen, seqLen, nHeads, nKVHeads,
                      static_cast<uint32_t>(kvCache_.pos));

            // Debug: print Q, K stats after RoPE for Layer 0
            if (layer == 0 && seqLen > 1) {
                auto printVecStats = [&](const float *data, uint32_t n, const std::string &label) {
                    float mn = data[0], mx = data[0];
                    double ssq = 0.0;
                    for (uint32_t i = 0; i < n; ++i) {
                        float v = data[i];
                        mn = std::min(mn, v);
                        mx = std::max(mx, v);
                        ssq += (double) v * v;
                    }
                    float r = std::sqrt((float) (ssq / n));
                    std::cout << "[DEBUG-PL] Layer 0 " << label << ": last token min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                };
                printVecStats(qData + (seqLen - 1) * nHeads * headDim, nHeads * headDim, "Q (after RoPE)");
                printVecStats(kData + (seqLen - 1) * nKVHeads * headDim, nKVHeads * headDim, "K (after RoPE)");
            }

            // Store K, V in cache
            uint32_t cachePos = static_cast<uint32_t>(kvCache_.pos);
            float *kCacheLayer =
                    kvCache_.k.data() + layer * maxSeqLen * nKVHeads * headDim;
            float *vCacheLayer =
                    kvCache_.v.data() + layer * maxSeqLen * nKVHeads * headDim;

            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *kSrc = kData + s * nKVHeads * headDim;
                const float *vSrc = vData + s * nKVHeads * headDim;
                float *kDst = kCacheLayer + (cachePos + s) * nKVHeads * headDim;
                float *vDst = vCacheLayer + (cachePos + s) * nKVHeads * headDim;
                uint32_t kvSize = nKVHeads * headDim;
                for (uint32_t i = 0; i < kvSize; ++i) {
                    kDst[i] = kSrc[i];
                    vDst[i] = vSrc[i];
                }
            }

            // Attention
            uint32_t totalCacheLen = cachePos + seqLen;
            attentionFused(qData, kCacheLayer, vCacheLayer, attnOutData, seqLen,
                           cachePos, totalCacheLen, layer);

            // Debug: print attention output stats for Layer 0
            if (layer == 0 && seqLen > 1) {
                float aMin = attnOutData[(seqLen - 1) * nHeads * headDim], aMax = attnOutData[(seqLen - 1) * nHeads * headDim];
                double aSumSq = 0.0;
                for (uint32_t i = 0; i < nHeads * headDim; ++i) {
                    float v = attnOutData[(seqLen - 1) * nHeads * headDim + i];
                    aMin = std::min(aMin, v);
                    aMax = std::max(aMax, v);
                    aSumSq += (double) v * v;
                }
                float aNorm = std::sqrt((float) (aSumSq / (nHeads * headDim)));
                std::cout << "[DEBUG-PL] Layer 0 attnOut: last token min=" << aMin << " max=" << aMax << " rms=" << aNorm << std::endl;
            }

            // Output projection (using dequantized F32 weights for exact float dot product)
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *attnRowPtr = attnOutData + s * nHeads * headDim;
                np::Array<float> projRow = deqMatMulVec(w.attnO_deq.data(), attnRowPtr,
                                                        w.attnO.rows, w.attnO.cols);
                float *projPtr = attnProjData + s * hiddenSize;
                std::memcpy(projPtr, projRow.data(), hiddenSize * sizeof(float));
            }

            // Debug: print attnProj stats for the last token
            if (seqLen > 1) {
                float aMin = attnProjData[(seqLen - 1) * hiddenSize], aMax = attnProjData[(seqLen - 1) * hiddenSize];
                double aSumSq = 0.0;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = attnProjData[(seqLen - 1) * hiddenSize + i];
                    aMin = std::min(aMin, v);
                    aMax = std::max(aMax, v);
                    aSumSq += (double) v * v;
                }
                float aNorm = std::sqrt((float) (aSumSq / hiddenSize));
                std::cout << "[DEBUG-PL] Layer " << layer << " attnProj: last token min=" << aMin << " max=" << aMax << " rms=" << aNorm << std::endl;
            }

            // Debug: print hidden state BEFORE attention residual for Layer 0
            if (layer == 0 && seqLen > 1) {
                float hMin = hiddenData[(seqLen - 1) * hiddenSize], hMax = hiddenData[(seqLen - 1) * hiddenSize];
                double hSumSq = 0.0;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = hiddenData[(seqLen - 1) * hiddenSize + i];
                    hMin = std::min(hMin, v);
                    hMax = std::max(hMax, v);
                    hSumSq += (double) v * v;
                }
                float hNorm = std::sqrt((float) (hSumSq / hiddenSize));
                std::cout << "[DEBUG-PL] Layer 0 hidden BEFORE attn residual: last token min=" << hMin << " max=" << hMax << " rms=" << hNorm << std::endl;
                std::cout << "[DEBUG-PL] Layer 0 hidden BEFORE attn residual first 8: ";
                for (uint32_t i = 0; i < 8; ++i)
                    std::cout << std::fixed << std::setprecision(6) << hiddenData[(seqLen - 1) * hiddenSize + i] << " ";
                std::cout << std::endl;
            }

            // Residual: hidden += attnProj
            for (uint32_t s = 0; s < seqLen; ++s) {
                float *hPtr = hiddenData + s * hiddenSize;
                const float *aPtr = attnProjData + s * hiddenSize;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hPtr[i] += aPtr[i];
                }
            }

            // Debug: print hidden state AFTER attention residual for Layer 0
            if (layer == 0 && seqLen > 1) {
                float hMin = hiddenData[(seqLen - 1) * hiddenSize], hMax = hiddenData[(seqLen - 1) * hiddenSize];
                double hSumSq = 0.0;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = hiddenData[(seqLen - 1) * hiddenSize + i];
                    hMin = std::min(hMin, v);
                    hMax = std::max(hMax, v);
                    hSumSq += (double) v * v;
                }
                float hNorm = std::sqrt((float) (hSumSq / hiddenSize));
                std::cout << "[DEBUG-PL] Layer 0 hidden AFTER attn residual: last token min=" << hMin << " max=" << hMax << " rms=" << hNorm << std::endl;
                std::cout << "[DEBUG-PL] Layer 0 hidden AFTER attn residual first 8: ";
                for (uint32_t i = 0; i < 8; ++i)
                    std::cout << std::fixed << std::setprecision(6) << hiddenData[(seqLen - 1) * hiddenSize + i] << " ";
                std::cout << std::endl;
            }

            // ---- FFN block ----

            // RMSNorm: ffnNorm = rmsNorm(hidden, w.rmsNormFFN)
            const float *rmsNormFFNData = w.rmsNormFFN.data();
            for (uint32_t s = 0; s < seqLen; ++s) {
                rmsNormInPlace(hiddenData + s * hiddenSize, ffnNormData + s * hiddenSize,
                               rmsNormFFNData, hiddenSize);
            }

            // Debug: print ffnNorm stats for Layer 0
            if (layer == 0 && seqLen > 1) {
                float nMin = ffnNormData[(seqLen - 1) * hiddenSize], nMax = ffnNormData[(seqLen - 1) * hiddenSize];
                double nSumSq = 0.0;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = ffnNormData[(seqLen - 1) * hiddenSize + i];
                    nMin = std::min(nMin, v);
                    nMax = std::max(nMax, v);
                    nSumSq += (double) v * v;
                }
                float nNorm = std::sqrt((float) (nSumSq / hiddenSize));
                std::cout << "[DEBUG-PL] Layer 0 ffnNorm: last token min=" << nMin << " max=" << nMax << " rms=" << nNorm << std::endl;
                std::cout << "[DEBUG-PL] Layer 0 ffnNorm first 8: ";
                for (uint32_t i = 0; i < 8; ++i)
                    std::cout << std::fixed << std::setprecision(6) << ffnNormData[(seqLen - 1) * hiddenSize + i] << " ";
                std::cout << std::endl;
            }

            // SwiGLU FFN
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *ffnRowPtr = ffnNormData + s * hiddenSize;

                // Gate projection
                np::Array<float> gateRow = w.ffnGate.matMulVec(ffnRowPtr);
                float *gatePtr = gateData + s * intermediateSize;
                std::memcpy(gatePtr, gateRow.data(), intermediateSize * sizeof(float));

                // Up projection
                np::Array<float> upRow = w.ffnUp.matMulVec(ffnRowPtr);
                float *upPtr = upData + s * intermediateSize;
                std::memcpy(upPtr, upRow.data(), intermediateSize * sizeof(float));
            }

            // Debug: print gate and up stats for Layer 0
            if (layer == 0 && seqLen > 1) {
                auto printVecStats = [&](const float *data, uint32_t n, const std::string &label) {
                    float mn = data[0], mx = data[0];
                    double ssq = 0.0;
                    for (uint32_t i = 0; i < n; ++i) {
                        float v = data[i];
                        mn = std::min(mn, v);
                        mx = std::max(mx, v);
                        ssq += (double) v * v;
                    }
                    float r = std::sqrt((float) (ssq / n));
                    std::cout << "[DEBUG-PL] Layer 0 " << label << ": last token min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                };
                printVecStats(gateData + (seqLen - 1) * intermediateSize, intermediateSize, "ffnGate (before SwiGLU)");
                printVecStats(upData + (seqLen - 1) * intermediateSize, intermediateSize, "ffnUp");
            }

            // SwiGLU activation
            swiGLUInPlace(gateData, upData, seqLen * intermediateSize);

            // Debug: print SwiGLU output stats for Layer 0
            if (layer == 0 && seqLen > 1) {
                float gMin = gateData[(seqLen - 1) * intermediateSize], gMax = gateData[(seqLen - 1) * intermediateSize];
                double gSumSq = 0.0;
                for (uint32_t i = 0; i < intermediateSize; ++i) {
                    float v = gateData[(seqLen - 1) * intermediateSize + i];
                    gMin = std::min(gMin, v);
                    gMax = std::max(gMax, v);
                    gSumSq += (double) v * v;
                }
                float gNorm = std::sqrt((float) (gSumSq / intermediateSize));
                std::cout << "[DEBUG-PL] Layer 0 swigluOut: last token min=" << gMin << " max=" << gMax << " rms=" << gNorm << std::endl;
            }

            // Down projection (using dequantized F32 weights for exact float dot product)
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *ffnActPtr = gateData + s * intermediateSize;
                np::Array<float> downRow = deqMatMulVec(w.ffnDown_deq.data(), ffnActPtr,
                                                        w.ffnDown.rows, w.ffnDown.cols);
                float *outPtr = ffnOutData + s * hiddenSize;
                std::memcpy(outPtr, downRow.data(), hiddenSize * sizeof(float));
            }

            // Debug: print FFN down projection stats for Layer 0
            if (layer == 0 && seqLen > 1) {
                float fMin = ffnOutData[(seqLen - 1) * hiddenSize], fMax = ffnOutData[(seqLen - 1) * hiddenSize];
                double fSumSq = 0.0;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = ffnOutData[(seqLen - 1) * hiddenSize + i];
                    fMin = std::min(fMin, v);
                    fMax = std::max(fMax, v);
                    fSumSq += (double) v * v;
                }
                float fNorm = std::sqrt((float) (fSumSq / hiddenSize));
                std::cout << "[DEBUG-PL] Layer 0 ffnDown: last token min=" << fMin << " max=" << fMax << " rms=" << fNorm << std::endl;
            }

            // Residual: hidden += ffnOut
            for (uint32_t s = 0; s < seqLen; ++s) {
                float *hPtr = hiddenData + s * hiddenSize;
                const float *fPtr = ffnOutData + s * hiddenSize;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hPtr[i] += fPtr[i];
                }
            }

            // Debug: print hidden state stats for the last token after each layer
            if (seqLen > 1) {
                float hMin = hiddenData[(seqLen - 1) * hiddenSize], hMax = hiddenData[(seqLen - 1) * hiddenSize];
                double hSumSq = 0.0;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = hiddenData[(seqLen - 1) * hiddenSize + i];
                    hMin = std::min(hMin, v);
                    hMax = std::max(hMax, v);
                    hSumSq += (double) v * v;
                }
                float hNorm = std::sqrt((float) (hSumSq / hiddenSize));
                std::cout << "[DEBUG-PL] Layer " << layer << " after FFN: last token min=" << hMin << " max=" << hMax << " rms=" << hNorm << std::endl;
            }

            // Capture hidden state after this layer's FFN residual
            perLayerStates.emplace_back(hiddenData, hiddenData + seqLen * hiddenSize);
        }

        // Update KV cache position
        kvCache_.pos += seqLen;

        return perLayerStates;
    }

    std::pair<std::vector<std::vector<float>>, std::vector<float>>
    Model::forwardTokenByToken(const std::vector<int32_t> &tokens) {
        // Process tokens one-by-one, exactly matching the reference
        // (token-by-token with KV cache accumulation). This isolates whether
        // batched prefill with causal masking is the source of divergence.
        //
        // Reference: unit_tests/llama_ref_dequant.cpp forward_multi()
        //
        // Returns:
        //   first: per-layer hidden states for the LAST token only
        //          (after each layer's FFN residual, before final RMSNorm).
        //          vector of nLayers vectors, each of size hiddenSize.
        //   second: logits for the LAST token (size vocabSize).

        if (tokens.empty()) {
            return {};
        }

        uint32_t seqLen = static_cast<uint32_t>(tokens.size());
        uint32_t hiddenSize = config_.hiddenSize;
        uint32_t nHeads = config_.numAttentionHeads;
        uint32_t nKVHeads = config_.numKVHeads;
        uint32_t headDim = config_.headDim;
        uint32_t nLayers = config_.numLayers;
        uint32_t maxSeqLen = config_.maxSeqLen;
        uint32_t vocabSize = config_.vocabSize;
        uint32_t intermediateSize = config_.intermediateSize;

        // Clear KV cache before starting
        clearKVCache();

        // Per-layer hidden states for the last token
        std::vector<std::vector<float>> perLayerStates;
        perLayerStates.reserve(nLayers);

        // Single-token buffers (reused for each position)
        std::vector<float> hidden(hiddenSize, 0.0f);
        std::vector<float> attnNorm(hiddenSize);
        std::vector<float> q(nHeads * headDim);
        std::vector<float> k(nKVHeads * headDim);
        std::vector<float> v(nKVHeads * headDim);
        std::vector<float> attnOut(nHeads * headDim);
        std::vector<float> attnProj(hiddenSize);
        std::vector<float> ffnNorm(hiddenSize);
        std::vector<float> gate(intermediateSize);
        std::vector<float> up(intermediateSize);
        std::vector<float> ffnOut(hiddenSize);

        for (uint32_t pos = 0; pos < seqLen; ++pos) {
            int32_t tokenId = tokens[pos];

            // ---- Token embedding ----
            if (tokenId >= 0 &&
                tokenId < static_cast<int32_t>(quantizedEmbeddings_.vocabSize)) {
                auto embRow = quantizedEmbeddings_.getRow(tokenId);
                std::memcpy(hidden.data(), embRow.data(), hiddenSize * sizeof(float));
            } else {
                std::fill(hidden.begin(), hidden.end(), 0.0f);
            }

            // Debug: print embedding stats for the last token
            if (pos == seqLen - 1) {
                float mn = hidden[0], mx = hidden[0];
                double ssq = 0.0;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = hidden[i];
                    mn = std::min(mn, v);
                    mx = std::max(mx, v);
                    ssq += (double) v * v;
                }
                float r = std::sqrt((float) (ssq / hiddenSize));
                std::cout << "[TBT] Embedding (last token): min=" << mn << " max=" << mx << " rms=" << r << std::endl;
            }

            for (uint32_t layer = 0; layer < nLayers; ++layer) {
                auto &w = layers_[layer];

                // ---- Attention block ----

                // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                rmsNormInPlace(hidden.data(), attnNorm.data(), w.rmsNormAttn.data(), hiddenSize);

                // Debug: print attnNorm stats for layer 0, last token
                if (layer == 0 && pos == seqLen - 1) {
                    float mn = attnNorm[0], mx = attnNorm[0];
                    double ssq = 0.0;
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        float v = attnNorm[i];
                        mn = std::min(mn, v);
                        mx = std::max(mx, v);
                        ssq += (double) v * v;
                    }
                    float r = std::sqrt((float) (ssq / hiddenSize));
                    std::cout << "[TBT] Layer 0 attnNorm: min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                }

                // Q projection (save raw matMulVec output before bias for diagnostic)
                np::Array<float> qRowRaw;
                {
                    qRowRaw = w.attnQ.matMulVec(attnNorm.data());
                    std::memcpy(q.data(), qRowRaw.data(), (nHeads * headDim) * sizeof(float));
                    if (!w.attnQBias.empty()) {
                        const float *biasData = w.attnQBias.data();
                        for (uint32_t i = 0; i < nHeads * headDim; ++i) {
                            q[i] += biasData[i];
                        }
                    }
                }

                // K projection (save raw matMulVec output before bias for diagnostic)
                np::Array<float> kRowRaw;
                {
                    kRowRaw = w.attnK.matMulVec(attnNorm.data());
                    std::memcpy(k.data(), kRowRaw.data(), (nKVHeads * headDim) * sizeof(float));
                    if (!w.attnKBias.empty()) {
                        const float *biasData = w.attnKBias.data();
                        for (uint32_t i = 0; i < nKVHeads * headDim; ++i) {
                            k[i] += biasData[i];
                        }
                    }
                }

                // V projection (save raw matMulVec output before bias for diagnostic)
                np::Array<float> vRowRaw;
                {
                    vRowRaw = w.attnV.matMulVec(attnNorm.data());
                    std::memcpy(v.data(), vRowRaw.data(), (nKVHeads * headDim) * sizeof(float));
                    if (!w.attnVBias.empty()) {
                        const float *biasData = w.attnVBias.data();
                        for (uint32_t i = 0; i < nKVHeads * headDim; ++i) {
                            v[i] += biasData[i];
                        }
                    }
                }

                // Debug: print Q, K, V stats for layer 0, last token
                if (layer == 0 && pos == seqLen - 1) {
                    auto printStats = [&](const float *data, uint32_t n, const std::string &label) {
                        float mn = data[0], mx = data[0];
                        double ssq = 0.0;
                        for (uint32_t i = 0; i < n; ++i) {
                            float v = data[i];
                            mn = std::min(mn, v);
                            mx = std::max(mx, v);
                            ssq += (double) v * v;
                        }
                        float r = std::sqrt((float) (ssq / n));
                        std::cout << "[TBT] Layer 0 " << label << ": min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                    };
                    printStats(q.data(), nHeads * headDim, "Q (before RoPE, with bias)");
                    printStats(k.data(), nKVHeads * headDim, "K (before RoPE, with bias)");
                    printStats(v.data(), nKVHeads * headDim, "V (with bias)");

                    // DIAGNOSTIC: Compare quantized matMulVec vs dequantized matMulVec for Q, K, V
                    // Compare BEFORE bias addition to isolate matMulVec correctness
                    {
                        std::vector<float> qDeq = GGMLDequantize::dequantize(
                                w.attnQ.type, w.attnQ.data.data(), (uint64_t) w.attnQ.rows * w.attnQ.cols);
                        np::Array<float> qRef = deqMatMulVec(qDeq.data(), attnNorm.data(),
                                                             w.attnQ.rows, w.attnQ.cols);
                        double qDiffSq = 0.0;
                        for (uint32_t i = 0; i < nHeads * headDim; ++i) {
                            double d = qRowRaw.data()[i] - qRef.data()[i];
                            qDiffSq += d * d;
                        }
                        float qRmse = std::sqrt((float) (qDiffSq / (nHeads * headDim)));
                        std::cout << "[TBT] Layer 0 Q quant vs dequant (no bias) RMSE=" << qRmse;
                        if (qRmse > 0.01f) std::cout << " *** LARGE DIFF ***";
                        std::cout << std::endl;
                        if (qRmse > 0.01f) {
                            std::cout << "[TBT] Layer 0 Q quant first 8 (no bias): ";
                            for (uint32_t i = 0; i < 8; ++i) std::cout << qRowRaw.data()[i] << " ";
                            std::cout << std::endl;
                            std::cout << "[TBT] Layer 0 Q dequant first 8: ";
                            for (uint32_t i = 0; i < 8; ++i) std::cout << qRef.data()[i] << " ";
                            std::cout << std::endl;
                        }
                    }
                    {
                        std::vector<float> kDeq = GGMLDequantize::dequantize(
                                w.attnK.type, w.attnK.data.data(), (uint64_t) w.attnK.rows * w.attnK.cols);
                        np::Array<float> kRef = deqMatMulVec(kDeq.data(), attnNorm.data(),
                                                             w.attnK.rows, w.attnK.cols);
                        double kDiffSq = 0.0;
                        for (uint32_t i = 0; i < nKVHeads * headDim; ++i) {
                            double d = kRowRaw.data()[i] - kRef.data()[i];
                            kDiffSq += d * d;
                        }
                        float kRmse = std::sqrt((float) (kDiffSq / (nKVHeads * headDim)));
                        std::cout << "[TBT] Layer 0 K quant vs dequant (no bias) RMSE=" << kRmse;
                        if (kRmse > 0.01f) std::cout << " *** LARGE DIFF ***";
                        std::cout << std::endl;
                        if (kRmse > 0.01f) {
                            std::cout << "[TBT] Layer 0 K quant first 8 (no bias): ";
                            for (uint32_t i = 0; i < 8; ++i) std::cout << kRowRaw.data()[i] << " ";
                            std::cout << std::endl;
                            std::cout << "[TBT] Layer 0 K dequant first 8: ";
                            for (uint32_t i = 0; i < 8; ++i) std::cout << kRef.data()[i] << " ";
                            std::cout << std::endl;
                        }
                    }
                    {
                        std::vector<float> vDeq = GGMLDequantize::dequantize(
                                w.attnV.type, w.attnV.data.data(), (uint64_t) w.attnV.rows * w.attnV.cols);
                        np::Array<float> vRef = deqMatMulVec(vDeq.data(), attnNorm.data(),
                                                             w.attnV.rows, w.attnV.cols);
                        double vDiffSq = 0.0;
                        for (uint32_t i = 0; i < nKVHeads * headDim; ++i) {
                            double d = vRowRaw.data()[i] - vRef.data()[i];
                            vDiffSq += d * d;
                        }
                        float vRmse = std::sqrt((float) (vDiffSq / (nKVHeads * headDim)));
                        std::cout << "[TBT] Layer 0 V quant vs dequant (no bias) RMSE=" << vRmse;
                        if (vRmse > 0.01f) std::cout << " *** LARGE DIFF ***";
                        std::cout << std::endl;
                        if (vRmse > 0.01f) {
                            std::cout << "[TBT] Layer 0 V quant first 8 (no bias): ";
                            for (uint32_t i = 0; i < 8; ++i) std::cout << vRowRaw.data()[i] << " ";
                            std::cout << std::endl;
                            std::cout << "[TBT] Layer 0 V dequant first 8: ";
                            for (uint32_t i = 0; i < 8; ++i) std::cout << vRef.data()[i] << " ";
                            std::cout << std::endl;
                        }
                    }
                }

                // Apply RoPE at position pos (single token, so qSeqLen=kSeqLen=1)
                applyRoPE(q.data(), k.data(), 1, 1, nHeads, nKVHeads, pos);

                // Debug: print Q, K stats after RoPE for layer 0, last token
                if (layer == 0 && pos == seqLen - 1) {
                    auto printStats = [&](const float *data, uint32_t n, const std::string &label) {
                        float mn = data[0], mx = data[0];
                        double ssq = 0.0;
                        for (uint32_t i = 0; i < n; ++i) {
                            float v = data[i];
                            mn = std::min(mn, v);
                            mx = std::max(mx, v);
                            ssq += (double) v * v;
                        }
                        float r = std::sqrt((float) (ssq / n));
                        std::cout << "[TBT] Layer 0 " << label << ": min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                    };
                    printStats(q.data(), nHeads * headDim, "Q (after RoPE)");
                    printStats(k.data(), nKVHeads * headDim, "K (after RoPE)");
                }

                // Store K, V in cache
                uint32_t cachePos = static_cast<uint32_t>(kvCache_.pos);
                float *kCacheLayer =
                        kvCache_.k.data() + layer * maxSeqLen * nKVHeads * headDim;
                float *vCacheLayer =
                        kvCache_.v.data() + layer * maxSeqLen * nKVHeads * headDim;

                uint32_t kvSize = nKVHeads * headDim;
                float *kDst = kCacheLayer + cachePos * kvSize;
                float *vDst = vCacheLayer + cachePos * kvSize;
                for (uint32_t i = 0; i < kvSize; ++i) {
                    kDst[i] = k[i];
                    vDst[i] = v[i];
                }

                // Attention: single query against all cached positions
                uint32_t totalCacheLen = cachePos + 1;
                attentionFused(q.data(), kCacheLayer, vCacheLayer, attnOut.data(),
                               1, cachePos, totalCacheLen, layer);

                // Debug: print attention output stats for layer 0, last token
                if (layer == 0 && pos == seqLen - 1) {
                    float mn = attnOut[0], mx = attnOut[0];
                    double ssq = 0.0;
                    for (uint32_t i = 0; i < nHeads * headDim; ++i) {
                        float v = attnOut[i];
                        mn = std::min(mn, v);
                        mx = std::max(mx, v);
                        ssq += (double) v * v;
                    }
                    float r = std::sqrt((float) (ssq / (nHeads * headDim)));
                    std::cout << "[TBT] Layer 0 attnOut: min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                }

                // Output projection (using dequantized F32 weights for exact float dot product)
                {
                    np::Array<float> projRow = deqMatMulVec(w.attnO_deq.data(), attnOut.data(),
                                                            w.attnO.rows, w.attnO.cols);
                    std::memcpy(attnProj.data(), projRow.data(), hiddenSize * sizeof(float));
                }

                // Debug: print attention projection stats for layer 0, last token
                if (layer == 0 && pos == seqLen - 1) {
                    float mn = attnProj[0], mx = attnProj[0];
                    double ssq = 0.0;
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        float v = attnProj[i];
                        mn = std::min(mn, v);
                        mx = std::max(mx, v);
                        ssq += (double) v * v;
                    }
                    float r = std::sqrt((float) (ssq / hiddenSize));
                    std::cout << "[TBT] Layer 0 attnProj: min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                }

                // Residual: hidden += attnProj
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hidden[i] += attnProj[i];
                }

                // Debug: print hidden state AFTER attention residual for layer 0, last token
                if (layer == 0 && pos == seqLen - 1) {
                    float mn = hidden[0], mx = hidden[0];
                    double ssq = 0.0;
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        float v = hidden[i];
                        mn = std::min(mn, v);
                        mx = std::max(mx, v);
                        ssq += (double) v * v;
                    }
                    float r = std::sqrt((float) (ssq / hiddenSize));
                    std::cout << "[TBT] Layer 0 hidden AFTER attn residual: min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                    std::cout << "[TBT] Layer 0 hidden AFTER attn residual first 8: ";
                    for (uint32_t i = 0; i < 8; ++i)
                        std::cout << std::fixed << std::setprecision(6) << hidden[i] << " ";
                    std::cout << std::endl;
                }

                // ---- FFN block ----

                // RMSNorm: ffnNorm = rmsNorm(hidden, w.rmsNormFFN)
                rmsNormInPlace(hidden.data(), ffnNorm.data(), w.rmsNormFFN.data(), hiddenSize);

                // Debug: print ffnNorm stats for layer 0, last token
                if (layer == 0 && pos == seqLen - 1) {
                    float mn = ffnNorm[0], mx = ffnNorm[0];
                    double ssq = 0.0;
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        float v = ffnNorm[i];
                        mn = std::min(mn, v);
                        mx = std::max(mx, v);
                        ssq += (double) v * v;
                    }
                    float r = std::sqrt((float) (ssq / hiddenSize));
                    std::cout << "[TBT] Layer 0 ffnNorm: min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                    std::cout << "[TBT] Layer 0 ffnNorm first 8: ";
                    for (uint32_t i = 0; i < 8; ++i)
                        std::cout << std::fixed << std::setprecision(6) << ffnNorm[i] << " ";
                    std::cout << std::endl;
                }

                // Gate projection
                {
                    np::Array<float> gateRow = w.ffnGate.matMulVec(ffnNorm.data());
                    std::memcpy(gate.data(), gateRow.data(), intermediateSize * sizeof(float));
                }

                // Up projection
                {
                    np::Array<float> upRow = w.ffnUp.matMulVec(ffnNorm.data());
                    std::memcpy(up.data(), upRow.data(), intermediateSize * sizeof(float));
                }

                // Debug: print gate and up stats for layer 0, last token
                if (layer == 0 && pos == seqLen - 1) {
                    auto printStats = [&](const float *data, uint32_t n, const std::string &label) {
                        float mn = data[0], mx = data[0];
                        double ssq = 0.0;
                        for (uint32_t i = 0; i < n; ++i) {
                            float v = data[i];
                            mn = std::min(mn, v);
                            mx = std::max(mx, v);
                            ssq += (double) v * v;
                        }
                        float r = std::sqrt((float) (ssq / n));
                        std::cout << "[TBT] Layer 0 " << label << ": min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                    };
                    printStats(gate.data(), intermediateSize, "ffnGate (before SwiGLU)");
                    printStats(up.data(), intermediateSize, "ffnUp");
                }

                // SwiGLU activation
                swiGLUInPlace(gate.data(), up.data(), intermediateSize);

                // Debug: print SwiGLU output stats for layer 0, last token
                if (layer == 0 && pos == seqLen - 1) {
                    float mn = gate[0], mx = gate[0];
                    double ssq = 0.0;
                    for (uint32_t i = 0; i < intermediateSize; ++i) {
                        float v = gate[i];
                        mn = std::min(mn, v);
                        mx = std::max(mx, v);
                        ssq += (double) v * v;
                    }
                    float r = std::sqrt((float) (ssq / intermediateSize));
                    std::cout << "[TBT] Layer 0 swigluOut: min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                }

                // Down projection (using dequantized F32 weights for exact float dot product)
                {
                    np::Array<float> downRow = deqMatMulVec(w.ffnDown_deq.data(), gate.data(),
                                                            w.ffnDown.rows, w.ffnDown.cols);
                    std::memcpy(ffnOut.data(), downRow.data(), hiddenSize * sizeof(float));
                }

                // Debug: print FFN down projection stats for layer 0, last token
                if (layer == 0 && pos == seqLen - 1) {
                    float mn = ffnOut[0], mx = ffnOut[0];
                    double ssq = 0.0;
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        float v = ffnOut[i];
                        mn = std::min(mn, v);
                        mx = std::max(mx, v);
                        ssq += (double) v * v;
                    }
                    float r = std::sqrt((float) (ssq / hiddenSize));
                    std::cout << "[TBT] Layer 0 ffnDown: min=" << mn << " max=" << mx << " rms=" << r << std::endl;
                }

                // Residual: hidden += ffnOut
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hidden[i] += ffnOut[i];
                }

                // Store per-layer hidden state for the last token
                if (pos == seqLen - 1) {
                    perLayerStates.emplace_back(hidden.begin(), hidden.end());
                }
            }

            // Final RMSNorm
            rmsNormInPlace(hidden.data(), hidden.data(), finalNorm_.data(), hiddenSize);

            // Update KV cache position
            kvCache_.pos += 1;

            if (pos == seqLen - 1 || pos < 5) {
                float mn = hidden[0], mx = hidden[0];
                double ssq = 0.0;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    float v = hidden[i];
                    mn = std::min(mn, v);
                    mx = std::max(mx, v);
                    ssq += (double) v * v;
                }
                float r = std::sqrt((float) (ssq / hiddenSize));
                std::cout << "[TBT] Token " << pos << " final hidden state: min=" << mn << " max=" << mx << " rms=" << r << std::endl;
            }
        }

        // ---- LM Head for last token ----
        std::vector<float> logits(vocabSize);
        if (lmHeadTied_) {
            // Use pre-dequantized embeddings for much faster computation
            if (!dequantizedEmbeddings_.empty()) {
                LMHead::computeCPU(hidden.data(),
                                   dequantizedEmbeddings_.data.data(),
                                   dequantizedEmbeddings_.vocabSize, hiddenSize,
                                   logits.data());
            } else {
                // Fallback to quantized path
                LMHead::computeCPUQuantized(hidden.data(),
                                            quantizedEmbeddings_.data.data(),
                                            quantizedEmbeddings_.type,
                                            quantizedEmbeddings_.vocabSize, hiddenSize,
                                            logits.data());
            }
        } else {
            LMHead::computeCPUSeparate(hidden.data(), lmHead_.data.data(), lmHead_.type,
                                       lmHead_.rows, lmHead_.cols, logits.data());
        }

        // Print per-layer hidden state stats for the last token
        std::cout << "\n[TBT] Per-Layer Hidden States (last token, after FFN residual) ===" << std::endl;
        for (uint32_t layer = 0; layer < nLayers; ++layer) {
            float mn = perLayerStates[layer][0], mx = perLayerStates[layer][0];
            double ssq = 0.0;
            for (uint32_t i = 0; i < hiddenSize; ++i) {
                float v = perLayerStates[layer][i];
                mn = std::min(mn, v);
                mx = std::max(mx, v);
                ssq += (double) v * v;
            }
            float r = std::sqrt((float) (ssq / hiddenSize));
            std::cout << "[TBT] Layer " << layer << ": min=" << mn << " max=" << mx << " rms=" << r << std::endl;
        }

        // Print top-10 logits
        std::vector<std::pair<float, int32_t>> top10;
        for (uint32_t t = 0; t < vocabSize; ++t)
            top10.emplace_back(logits[t], (int32_t) t);
        std::partial_sort(top10.begin(), top10.begin() + 10, top10.end(),
                          [](const auto &a, const auto &b) { return a.first > b.first; });

        std::cout << "\n[TBT] Top-10 Logits (last token):" << std::endl;
        for (int r = 0; r < 10 && r < (int) top10.size(); ++r) {
            std::cout << "  [" << r << "] id=" << top10[r].second
                      << " logit=" << std::fixed << std::setprecision(4) << top10[r].first << std::endl;
        }

        return {std::move(perLayerStates), std::move(logits)};
    }

    np::Array<float> Model::applySamplingParams(const np::Array<float> &logits,
                                                const InferenceParams &params) {
        uint32_t vocabSize = config_.vocabSize;
        np::Array<float> result = np::Array<float>(np::Shape{vocabSize});

        // Apply temperature
        float invTemp = 1.0f / std::max(params.temperature, 0.01f);
        float maxLogit = -std::numeric_limits<float>::infinity();

        for (uint32_t i = 0; i < vocabSize; ++i) {
            float val = logits.get(i) * invTemp;
            result.set(i, val);
            maxLogit = std::max(maxLogit, val);
        }

        // Softmax
        float sumExp = 0.0f;
        for (uint32_t i = 0; i < vocabSize; ++i) {
            float val = std::exp(result.get(i) - maxLogit);
            result.set(i, val);
            sumExp += val;
        }

        float invSum = 1.0f / sumExp;
        for (uint32_t i = 0; i < vocabSize; ++i) {
            result.set(i, result.get(i) * invSum);
        }

        // Top-K filtering
        if (params.topK > 0 && params.topK < vocabSize) {
            std::vector<std::pair<float, uint32_t>> probs;
            probs.reserve(vocabSize);
            for (uint32_t i = 0; i < vocabSize; ++i) {
                probs.emplace_back(result.get(i), i);
            }
            std::partial_sort(
                    probs.begin(), probs.begin() + static_cast<uint32_t>(params.topK),
                    probs.end(),
                    [](const auto &a, const auto &b) { return a.first > b.first; });

            float threshold = probs[static_cast<uint32_t>(params.topK) - 1].first;
            for (uint32_t i = 0; i < vocabSize; ++i) {
                if (result.get(i) < threshold) {
                    result.set(i, 0.0f);
                }
            }

            sumExp = 0.0f;
            for (uint32_t i = 0; i < vocabSize; ++i) {
                sumExp += result.get(i);
            }
            if (sumExp > 0) {
                invSum = 1.0f / sumExp;
                for (uint32_t i = 0; i < vocabSize; ++i) {
                    result.set(i, result.get(i) * invSum);
                }
            }
        }

        // Top-P (nucleus) filtering
        if (params.topP < 1.0f) {
            std::vector<std::pair<float, uint32_t>> probs;
            probs.reserve(vocabSize);
            for (uint32_t i = 0; i < vocabSize; ++i) {
                probs.emplace_back(result.get(i), i);
            }
            std::sort(probs.begin(), probs.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });

            // Nucleus (top-P) sampling: keep the smallest set of tokens whose
            // cumulative probability >= topP. The token that pushes cumSum over
            // the threshold is INCLUDED (kept), and only tokens after it are zeroed.
            // This ensures at least the top token is always kept.
            float cumSum = 0.0f;
            bool thresholdReached = false;
            for (auto &p: probs) {
                if (thresholdReached) {
                    result.set(p.second, 0.0f);
                } else {
                    cumSum += p.first;
                    if (cumSum >= params.topP) {
                        thresholdReached = true;
                    }
                }
            }

            sumExp = 0.0f;
            for (uint32_t i = 0; i < vocabSize; ++i) {
                sumExp += result.get(i);
            }
            if (sumExp > 0) {
                invSum = 1.0f / sumExp;
                for (uint32_t i = 0; i < vocabSize; ++i) {
                    result.set(i, result.get(i) * invSum);
                }
            }
        }

        return result;
    }

    int32_t Model::sampleToken(const np::Array<float> &logits,
                               const InferenceParams &params) {
        auto probs = applySamplingParams(logits, params);

        // Initialize persistent RNG on first call or when seed changes.
        // Using a persistent RNG ensures proper random sequence across
        // multiple sampling calls.
        if (!rngInitialized_ || (params.seed > 0 && rngSeed_ != params.seed)) {
            rng_.seed(params.seed > 0 ? static_cast<uint64_t>(params.seed)
                                      : std::random_device{}());
            rngSeed_ = params.seed;
            rngInitialized_ = true;
        }

        // Use RNG method: rng() / max() for [0,1) range
        float r = static_cast<float>(rng_()) / static_cast<float>(std::mt19937::max());
        float cumSum = 0.0f;

        for (uint32_t i = 0; i < config_.vocabSize; ++i) {
            cumSum += probs.get(i);
            if (r <= cumSum) {
                return static_cast<int32_t>(i);
            }
        }

        return 0;
    }

    std::vector<int32_t> Model::tokenize(const std::string &prompt) {
        return tokenizer_.encode(prompt);
    }

    std::vector<int32_t>
    Model::generate(const std::string &prompt, const InferenceParams &params,
                    std::function<bool(int32_t, const std::string &)> callback) {
        if (!loaded_) {
            std::cerr << "[TinyCoder] Model not loaded" << std::endl;
            return {};
        }

        // Clear KV cache to ensure a clean generation state
        clearKVCache();

        auto t0 = std::chrono::high_resolution_clock::now();

        // Tokenize prompt
        auto tokens = tokenize(prompt);
        if (tokens.empty()) {
            std::cerr << "[TinyCoder] Empty prompt" << std::endl;
            return {};
        }

        std::cout << "[TinyCoder] Prompt: " << tokens.size() << " tokens"
                  << std::endl;

        // Debug: print first 10 prompt tokens and their decoded text
        std::cout << "[TinyCoder] Prompt tokens: ";
        for (size_t i = 0; i < std::min<size_t>(tokens.size(), 10); ++i) {
            std::cout << tokens[i] << " ";
        }
        if (tokens.size() > 10)
            std::cout << "...";
        std::cout << std::endl;
        std::cout << "[TinyCoder] Prompt decoded (first 100 chars): \""
                  << tokenizer_.decode(
                             std::vector<int32_t>(tokens.begin(),
                                                  tokens.begin() + std::min<size_t>(tokens.size(), 20)))
                  << "\"" << std::endl;

        // Prefill (process all prompt tokens at once)
        auto logits = forward(tokens);

        // Get the last token's logits
        uint32_t lastIdx = static_cast<uint32_t>(tokens.size()) - 1;
        np::Array<float> lastLogits = np::Array<float>(np::Shape{config_.vocabSize});
        for (uint32_t i = 0; i < config_.vocabSize; ++i) {
            lastLogits.set(i, logits.get(lastIdx * config_.vocabSize + i));
        }

        // Debug: print logit stats for the last prompt token
        {
            float minL = std::numeric_limits<float>::max();
            float maxL = -std::numeric_limits<float>::max();
            int nanL = 0, infL = 0;
            for (uint32_t i = 0; i < config_.vocabSize; ++i) {
                float v = lastLogits.get(i);
                if (std::isnan(v))
                    nanL++;
                if (std::isinf(v))
                    infL++;
                minL = std::min(minL, v);
                maxL = std::max(maxL, v);
            }
            std::cout << "[TinyCoder] Prefill logit stats: min=" << minL
                      << " max=" << maxL << " nan=" << nanL << " inf=" << infL
                      << std::endl;
            // Print top-5 token IDs and their decoded text
            std::vector<std::pair<float, int32_t>> top5;
            for (uint32_t i = 0; i < config_.vocabSize; ++i) {
                top5.emplace_back(lastLogits.get(i), static_cast<int32_t>(i));
            }
            std::partial_sort(top5.begin(), top5.begin() + 5, top5.end(),
                              [](const auto &a, const auto &b) { return a.first > b.first; });
            std::cout << "[TinyCoder] Top-5 prefill tokens:" << std::endl;
            for (int r = 0; r < 5 && r < static_cast<int>(top5.size()); ++r) {
                std::string dt = tokenizer_.decodeToken(top5[r].second);
                // Sanitize output for display
                std::string display;
                for (char c: dt) {
                    if (c >= 32 && c < 127)
                        display += c;
                    else
                        display += "\\x" + std::to_string(static_cast<unsigned char>(c));
                }
                std::cout << "  [" << r << "] id=" << top5[r].second << " logit="
                          << top5[r].first << " text=\"" << display << "\"" << std::endl;
            }
        }

        // Sample first generated token
        int32_t nextToken = sampleToken(lastLogits, params);
        std::string tokenText = tokenizer_.decodeToken(nextToken);

        std::cout << "[TinyCoder] First sampled token: id=" << nextToken
                  << " text=\"" << tokenText << "\"" << std::endl;

        std::vector<int32_t> generated;
        generated.push_back(nextToken);

        if (!callback(nextToken, tokenText)) {
            return generated;
        }

        // Generate remaining tokens
        for (int32_t i = 1; i < params.maxTokens; ++i) {
            // Forward pass for single token
            auto newLogits = forward({nextToken});

            // Get logits
            for (uint32_t j = 0; j < config_.vocabSize; ++j) {
                lastLogits.set(j, newLogits.get(j));
            }

            // Apply repetition penalty
            if (params.repeatPenalty != 1.0f) {
                int32_t startPenalty = std::max(
                        0, static_cast<int32_t>(generated.size()) - params.repeatLastN);
                for (int32_t p = startPenalty; p < static_cast<int32_t>(generated.size());
                     ++p) {
                    int32_t prevToken = generated[p];
                    float logit = lastLogits.get(prevToken);
                    if (logit < 0) {
                        lastLogits.set(prevToken, logit * params.repeatPenalty);
                    } else {
                        lastLogits.set(prevToken, logit / params.repeatPenalty);
                    }
                }
            }

            nextToken = sampleToken(lastLogits, params);

            // Check for end of generation (multiple EOG tokens for Qwen2.5-Coder)
            if (tokenizer_.isEogToken(nextToken)) {
                std::cout << "[TinyCoder] EOG token reached, stopping" << std::endl;
                break;
            }

            tokenText = tokenizer_.decodeToken(nextToken);
            generated.push_back(nextToken);

            if (!callback(nextToken, tokenText)) {
                break;
            }

            // Debug: print every 10th token
            if ((i % 10) == 0) {
                std::string display;
                for (char c: tokenText) {
                    if (c >= 32 && c < 127)
                        display += c;
                    else
                        display += "\\x" + std::to_string(static_cast<unsigned char>(c));
                }
                std::cout << "[TinyCoder] Token " << i << ": id=" << nextToken
                          << " text=\"" << display << "\"" << std::endl;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        float tokensPerSec = generated.size() / (ms / 1000.0f);

        std::cout << "[TinyCoder] Generated " << generated.size() << " tokens in "
                  << ms << " ms (" << tokensPerSec << " tok/s)" << std::endl;

        return generated;
    }

}// namespace tinycoder
