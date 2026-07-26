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
#include "ChatTemplateRenderer.hpp"
#include "GGMLDequantize.hpp"
#include "LMHead.hpp"
#include "SIMDMatMulVec.hpp"
#include "ThreadPool.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
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
        if (!isSupportedArchitecture(config_.architecture)) {
            setError("Unsupported model architecture: \"" + config_.architecture +
                     "\". Supported architectures: " + ARCH_QWEN2 + ", " +
                     ARCH_GEMMA4 + ", " + ARCH_QWEN35MOE);
            return false;
        }

        // Validate model name is in the supported list (optional check)
        if (!config_.modelName.empty() && !isSupportedModel(config_.modelName)) {
            std::cout << "[TinyCoder] Warning: model name \"" << config_.modelName
                      << "\" not in known list, but architecture \""
                      << config_.architecture << "\" is supported. Proceeding..."
                      << std::endl;
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

        // Configure tokenizer for the model architecture FIRST (sets defaults)
        tokenizer_.configureForArchitecture(config_.architecture);

        // Load tokenizer from GGUF (overrides BOS/EOS/PAD from metadata)
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
            // For a 3D tensor [D0, D1, D2] in numpy → GGUF shape [D2, D1, D0]:
            //   shape[0] = D2 (cols), shape[1] = D1, shape[2] = D0
            //   total rows = D0 * D1 = shape[1] * shape[2]
            uint32_t rows =
                    info->shape.size() >= 2 ? static_cast<uint32_t>(info->shape[1]) : 1;
            uint32_t cols =
                    info->shape.size() >= 1 ? static_cast<uint32_t>(info->shape[0]) : 1;
            if (info->shape.size() == 1) {
                rows = 1;
                cols = static_cast<uint32_t>(info->shape[0]);
            }
            // For 3D tensors (expert weights), multiply rows by the third dimension
            if (info->shape.size() >= 3) {
                rows *= static_cast<uint32_t>(info->shape[2]);
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

        // Load per-layer weights (architecture-aware)
        layers_.resize(config_.numLayers);
        for (uint32_t i = 0; i < config_.numLayers; ++i) {
            std::string prefix = "blk." + std::to_string(i) + ".";

            if (config_.architecture == ARCH_QWEN2) {
                // ---- Qwen2 architecture ----
                // Quantized attention weights
                layers_[i].attnQ = loadQuantized(prefix + "attn_q.weight");
                layers_[i].attnK = loadQuantized(prefix + "attn_k.weight");
                layers_[i].attnV = loadQuantized(prefix + "attn_v.weight");
                layers_[i].attnO = loadQuantized(prefix + "attn_output.weight");

                // Dequantize attnO to FP16 for exact float dot product (halves memory bandwidth vs F32)
                {
                    auto &qm = layers_[i].attnO;
                    uint64_t numElements = static_cast<uint64_t>(qm.rows) * qm.cols;
                    layers_[i].attnO_deq_f16 = GGMLDequantize::dequantizeToF16(qm.type, qm.data.data(), numElements);
                }

                // Quantized FFN weights (SwiGLU)
                layers_[i].ffnGate = loadQuantized(prefix + "ffn_gate.weight");
                layers_[i].ffnUp = loadQuantized(prefix + "ffn_up.weight");
                layers_[i].ffnDown = loadQuantized(prefix + "ffn_down.weight");

                // Dequantize ffnDown to FP16
                {
                    auto &qm = layers_[i].ffnDown;
                    uint64_t numElements = static_cast<uint64_t>(qm.rows) * qm.cols;
                    layers_[i].ffnDown_deq_f16 = GGMLDequantize::dequantizeToF16(qm.type, qm.data.data(), numElements);
                }

                // Pre-pack ffnGate and ffnUp for gather-free SIMD access (Q2_K only)
                // This eliminates the 2-bit extraction overhead in the SIMD kernel.
                if (layers_[i].ffnGate.type == GGML_TYPE_Q2_K) {
                    uint64_t numElements = static_cast<uint64_t>(layers_[i].ffnGate.rows) * layers_[i].ffnGate.cols;
                    layers_[i].ffnGate.prepackedData = GGMLDequantize::prepackQ2_K(layers_[i].ffnGate.data.data(), numElements);
                }
                if (layers_[i].ffnUp.type == GGML_TYPE_Q2_K) {
                    uint64_t numElements = static_cast<uint64_t>(layers_[i].ffnUp.rows) * layers_[i].ffnUp.cols;
                    layers_[i].ffnUp.prepackedData = GGMLDequantize::prepackQ2_K(layers_[i].ffnUp.data.data(), numElements);
                }

                // F32 norms
                layers_[i].rmsNormAttn = loadF32_1D(prefix + "attn_norm.weight");
                layers_[i].rmsNormFFN = loadF32_1D(prefix + "ffn_norm.weight");

                // F32 Q, K, V biases (Qwen2.5-Coder has these)
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
            } else if (config_.architecture == ARCH_GEMMA4) {
                // ---- Gemma4 architecture (dense and MoE) ----
                // Quantized attention weights (no biases)
                layers_[i].attnQ = loadQuantized(prefix + "attn_q.weight");
                layers_[i].attnK = loadQuantized(prefix + "attn_k.weight");
                layers_[i].attnV = loadQuantized(prefix + "attn_v.weight");
                layers_[i].attnO = loadQuantized(prefix + "attn_output.weight");

                // Dequantize attnO to FP16 for exact float dot product (halves memory bandwidth vs F32)
                {
                    auto &qm = layers_[i].attnO;
                    uint64_t numElements = static_cast<uint64_t>(qm.rows) * qm.cols;
                    layers_[i].attnO_deq_f16 = GGMLDequantize::dequantizeToF16(qm.type, qm.data.data(), numElements);
                }

                // Quantized FFN weights (GeGLU: gate+up, down)
                layers_[i].ffnGate = loadQuantized(prefix + "ffn_gate.weight");
                layers_[i].ffnUp = loadQuantized(prefix + "ffn_up.weight");
                layers_[i].ffnDown = loadQuantized(prefix + "ffn_down.weight");

                // Dequantize ffnDown to FP16
                {
                    auto &qm = layers_[i].ffnDown;
                    uint64_t numElements = static_cast<uint64_t>(qm.rows) * qm.cols;
                    layers_[i].ffnDown_deq_f16 = GGMLDequantize::dequantizeToF16(qm.type, qm.data.data(), numElements);
                }

                // Pre-pack ffnGate and ffnUp for gather-free SIMD access (Q2_K only)
                if (layers_[i].ffnGate.type == GGML_TYPE_Q2_K) {
                    uint64_t numElements = static_cast<uint64_t>(layers_[i].ffnGate.rows) * layers_[i].ffnGate.cols;
                    layers_[i].ffnGate.prepackedData = GGMLDequantize::prepackQ2_K(layers_[i].ffnGate.data.data(), numElements);
                }
                if (layers_[i].ffnUp.type == GGML_TYPE_Q2_K) {
                    uint64_t numElements = static_cast<uint64_t>(layers_[i].ffnUp.rows) * layers_[i].ffnUp.cols;
                    layers_[i].ffnUp.prepackedData = GGMLDequantize::prepackQ2_K(layers_[i].ffnUp.data.data(), numElements);
                }

                // F32 norms
                layers_[i].rmsNormAttn = loadF32_1D(prefix + "attn_norm.weight");
                layers_[i].rmsNormFFN = loadF32_1D(prefix + "ffn_norm.weight");

                // Gemma4-specific: Q/K norms (RMSNorm before RoPE)
                layers_[i].attnQNorm = loadF32_1D(prefix + "attn_q_norm.weight");
                layers_[i].attnKNorm = loadF32_1D(prefix + "attn_k_norm.weight");

                // Gemma4-specific: post-attention and post-FFW norms
                layers_[i].postAttnNorm = loadF32_1D(prefix + "post_attention_norm.weight");
                layers_[i].postFFWNorm = loadF32_1D(prefix + "post_ffw_norm.weight");

                // Gemma4-specific: layer output scale
                layers_[i].layerOutputScale = loadF32_1D(prefix + "layer_output_scale.weight");

                // Gemma4 MoE-specific weights
                if (config_.expertCount > 0) {
                    // Expert router
                    layers_[i].ffnGateInp = loadQuantized(prefix + "ffn_gate_inp.weight");
                    // Expert scales (loaded as F32 1D)
                    {
                        auto info = loader.getTensorInfo(prefix + "ffn_gate_inp.scale");
                        if (info) {
                            const uint8_t *data = loader.getTensor(prefix + "ffn_gate_inp.scale");
                            if (data) {
                                uint32_t n = static_cast<uint32_t>(info->shape[0]);
                                std::vector<float> vec(n);
                                std::memcpy(vec.data(), data, n * sizeof(float));
                                // Store in layerOutputScale (reuse field)
                                // ffn_gate_inp.scale is a per-expert scale factor
                            }
                        }
                    }
                    // Fused gate+up expert weights: [expertCount, expertFF * 2, hiddenSize]
                    layers_[i].ffnGateUpExps = loadQuantized(prefix + "ffn_gate_up_exps.weight");
                    // Down expert weights: [expertCount, hiddenSize, expertFF]
                    layers_[i].ffnDownExps = loadQuantized(prefix + "ffn_down_exps.weight");

                    // Additional norms for MoE
                    layers_[i].preFFWNorm2 = loadF32_1D(prefix + "pre_ffw_norm_2.weight");
                    layers_[i].postFFWNorm1 = loadF32_1D(prefix + "post_ffw_norm_1.weight");
                    layers_[i].postFFWNorm2 = loadF32_1D(prefix + "post_ffw_norm_2.weight");
                }

                // Validate all tensors loaded
                if (layers_[i].attnQ.empty() || layers_[i].attnK.empty() ||
                    layers_[i].attnV.empty() || layers_[i].attnO.empty()) {
                    std::cerr << "[TinyCoder] Missing attention weights for layer " << i
                              << std::endl;
                    return false;
                }
            } else if (config_.architecture == ARCH_QWEN35MOE) {
                // ---- Qwen35MoE architecture (MoE + SSM + MTP) ----
                // Quantized attention weights (separate Q, K, V)
                layers_[i].attnQ = loadQuantized(prefix + "attn_q.weight");
                layers_[i].attnK = loadQuantized(prefix + "attn_k.weight");
                layers_[i].attnV = loadQuantized(prefix + "attn_v.weight");
                layers_[i].attnO = loadQuantized(prefix + "attn_output.weight");

                // Dequantize attnO to FP16 for exact float dot product (halves memory bandwidth vs F32)
                {
                    auto &qm = layers_[i].attnO;
                    uint64_t numElements = static_cast<uint64_t>(qm.rows) * qm.cols;
                    layers_[i].attnO_deq_f16 = GGMLDequantize::dequantizeToF16(qm.type, qm.data.data(), numElements);
                }

                // Fused QKV projection (optional, may not be present in all layers)
                layers_[i].attnQKV = loadQuantized(prefix + "attn_qkv.weight");

                // Attention gate (element-wise gating for attention output)
                layers_[i].attnGate = loadQuantized(prefix + "attn_gate.weight");

                // Q/K norms (RMSNorm before RoPE)
                layers_[i].attnQNormMoe = loadF32_1D(prefix + "attn_q_norm.weight");
                layers_[i].attnKNormMoe = loadF32_1D(prefix + "attn_k_norm.weight");

                // F32 norms
                layers_[i].rmsNormAttn = loadF32_1D(prefix + "attn_norm.weight");
                layers_[i].postAttnNorm = loadF32_1D(prefix + "post_attention_norm.weight");

                // ---- SSM (Mamba-style) weights ----
                layers_[i].ssmConv1d = loadQuantized(prefix + "ssm_conv1d.weight");
                layers_[i].ssmOut = loadQuantized(prefix + "ssm_out.weight");
                layers_[i].ssmA = loadF32_1D(prefix + "ssm_a");
                layers_[i].ssmDtBias = loadF32_1D(prefix + "ssm_dt.bias");
                layers_[i].ssmAlpha = loadF32_1D(prefix + "ssm_alpha.weight");
                layers_[i].ssmBeta = loadF32_1D(prefix + "ssm_beta.weight");
                layers_[i].ssmNorm = loadF32_1D(prefix + "ssm_norm.weight");

                // ---- MoE FFN weights ----
                // Expert router
                layers_[i].ffnGateInpMoe = loadQuantized(prefix + "ffn_gate_inp.weight");
                // Expert weights (gate, up, down)
                layers_[i].ffnGateExps = loadQuantized(prefix + "ffn_gate_exps.weight");
                layers_[i].ffnUpExps = loadQuantized(prefix + "ffn_up_exps.weight");
                layers_[i].ffnDownExpsMoe = loadQuantized(prefix + "ffn_down_exps.weight");

                // Shared expert weights
                layers_[i].ffnGateInpShexp = loadQuantized(prefix + "ffn_gate_inp_shexp.weight");
                layers_[i].ffnGateShexp = loadQuantized(prefix + "ffn_gate_shexp.weight");
                layers_[i].ffnUpShexp = loadQuantized(prefix + "ffn_up_shexp.weight");
                layers_[i].ffnDownShexp = loadQuantized(prefix + "ffn_down_shexp.weight");

                // ---- MTP (Multi-Token Prediction) weights ----
                layers_[i].nextnEhProj = loadQuantized(prefix + "nextn.eh_proj.weight");
                layers_[i].nextnEnorm = loadF32_1D(prefix + "nextn.enorm.weight");
                layers_[i].nextnHnorm = loadF32_1D(prefix + "nextn.hnorm.weight");
                layers_[i].nextnSharedHeadNorm = loadF32_1D(prefix + "nextn.shared_head_norm.weight");

                // Validate core tensors loaded
                if (layers_[i].attnQ.empty() || layers_[i].attnK.empty() ||
                    layers_[i].attnV.empty() || layers_[i].attnO.empty()) {
                    std::cerr << "[TinyCoder] Missing attention weights for layer " << i
                              << std::endl;
                    return false;
                }
            } else {
                std::cerr << "[TinyCoder] Unknown architecture: " << config_.architecture
                          << " for layer " << i << std::endl;
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

        // Initialize SSM state for Qwen35MoE architecture
        if (config_.architecture == ARCH_QWEN35MOE && config_.ssmInnerSize > 0) {
            uint32_t ssmConvKernel = config_.ssmConvKernel;
            uint32_t ssmStateSize = config_.ssmStateSize;
            uint32_t ssmInnerSize = config_.ssmInnerSize;

            kvCache_.ssmConvBuf.resize(nLayers);
            kvCache_.ssmState.resize(nLayers);
            for (uint32_t i = 0; i < nLayers; ++i) {
                // Conv buffer: store (ssmConvKernel - 1) past inputs, each of size ssmInnerSize
                if (ssmConvKernel > 1) {
                    kvCache_.ssmConvBuf[i].resize((ssmConvKernel - 1) * ssmInnerSize, 0.0f);
                }
                // SSM state: ssmInnerSize x ssmStateSize
                kvCache_.ssmState[i].resize(ssmInnerSize * ssmStateSize, 0.0f);
            }

            uint64_t ssmMemBytes = static_cast<uint64_t>(nLayers) * ssmInnerSize *
                                   (ssmConvKernel + ssmStateSize) * sizeof(float);
            std::cout << "[TinyCoder] SSM state: " << nLayers << " layers x "
                      << ssmInnerSize << " inner x " << ssmStateSize << " state + "
                      << ssmConvKernel << " conv = " << (ssmMemBytes / (1024 * 1024)) << " MB"
                      << std::endl;
        }

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

        // Clear SSM state for Qwen35MoE architecture
        for (auto &buf: kvCache_.ssmConvBuf) {
            std::fill(buf.begin(), buf.end(), 0.0f);
        }
        for (auto &state: kvCache_.ssmState) {
            std::fill(state.begin(), state.end(), 0.0f);
        }
    }

    void Model::rmsNormInPlace(const float *x, float *out, const float *weight,
                               uint32_t n, float eps) const {
        // Delegate to SIMD-accelerated rmsNormSIMD
        rmsNormSIMD(x, out, weight, n);
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
        // Parallelize over (s, g) pairs using the thread pool.
        // Flatten the 2D iteration space into 1D.
        constexpr uint32_t STACK_SCORE_LIMIT = 4096;

        ThreadPool::instance().parallelFor2D(seqLen, nKVHeads,
                                             [&](uint32_t s, uint32_t g) {
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
                                                     // Use SIMD-accelerated dot product
                                                     float maxScore = -1e30f;
                                                     for (uint32_t cs = 0; cs <= csEnd; ++cs) {
                                                         const float *kPtr = kCache + (cs * nKVHeads * headDim + g * headDim);
                                                         float score = dotProductFMA(qPtr, kPtr, headDim) * invSqrtHeadDim;
                                                         localScores[cs] = score;
                                                         if (score > maxScore)
                                                             maxScore = score;
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
                                             });
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

    void Model::geluInPlace(float *x, uint32_t n) const {
        // GeLU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        constexpr float sqrt2OverPi = 0.7978845608028654f;// sqrt(2.0 / M_PI)
        for (uint32_t i = 0; i < n; ++i) {
            float x3 = x[i] * x[i] * x[i];
            x[i] = 0.5f * x[i] * (1.0f + std::tanh(sqrt2OverPi * (x[i] + 0.044715f * x3)));
        }
    }

    void Model::softcapInPlace(float *x, uint32_t n, float cap) {
        // tanh(x / cap) * cap
        if (cap > 0.0f) {
            float invCap = 1.0f / cap;
            for (uint32_t i = 0; i < n; ++i) {
                x[i] = std::tanh(x[i] * invCap) * cap;
            }
        }
    }

    void Model::applyQKNorms(float *q, float *k, uint32_t seqLen,
                             uint32_t qHeads, uint32_t kHeads,
                             const float *qNorm, const float *kNorm) {
        // Per-head RMSNorm applied to Q and K before RoPE.
        // q layout: [seqLen, qHeads, headDim]
        // k layout: [seqLen, kHeads, headDim]
        // qNorm/kNorm: [headDim] (shared across all heads)
        uint32_t headDim = config_.headDim;
        for (uint32_t s = 0; s < seqLen; ++s) {
            for (uint32_t h = 0; h < qHeads; ++h) {
                float *qHead = q + (s * qHeads + h) * headDim;
                rmsNormInPlace(qHead, qHead, qNorm, headDim);
            }
            for (uint32_t h = 0; h < kHeads; ++h) {
                float *kHead = k + (s * kHeads + h) * headDim;
                rmsNormInPlace(kHead, kHead, kNorm, headDim);
            }
        }
    }

    void Model::computeGemma4MoE(const float *ffnNorm, float *ffnOut,
                                 uint32_t seqLen, uint32_t hiddenSize,
                                 uint32_t intermediateSize,
                                 const LayerWeights &w) const {
        // Gemma4 MoE FFN with top-k expert routing.
        //
        // Architecture:
        //   1. Second pre-FFN norm (preFFWNorm2) applied to input
        //   2. Router (ffnGateInp) computes expert scores
        //   3. Top-k experts selected (k = expertUsedCount)
        //   4. For each selected expert:
        //      a. Gate+Up projection from ffnGateUpExps (fused gate+up per expert)
        //      b. GeGLU activation: gelu(gate) * up
        //      c. Down projection from ffnDownExps
        //   5. Weighted sum of expert outputs by routing probabilities
        //   6. Post-FFN norm 1 (postFFWNorm1)
        //   7. Post-FFN norm 2 (postFFWNorm2)
        //
        // Tensor layouts (after loadQuantized handles 3D→2D flattening):
        //   ffnGateUpExps: [expertCount * expertFF * 2, hiddenSize]
        //     - Expert e: rows [e * expertFF * 2, (e+1) * expertFF * 2)
        //       - Gate: rows [e * expertFF * 2, e * expertFF * 2 + expertFF)
        //       - Up:   rows [e * expertFF * 2 + expertFF, (e+1) * expertFF * 2)
        //   ffnDownExps: [expertCount * hiddenSize, expertFF]
        //     - Expert e: rows [e * hiddenSize, (e+1) * hiddenSize)
        //   ffnGateInp: [hiddenSize, expertCount] (router)

        uint32_t expertCount = config_.expertCount;
        uint32_t expertUsedCount = config_.expertUsedCount;
        uint32_t expertFF = config_.expertFeedForwardLength;

        if (expertCount == 0 || expertUsedCount == 0) {
            std::cerr << "[TinyCoder] computeGemma4MoE: MoE not configured" << std::endl;
            return;
        }

        // Temporary buffers per token
        std::vector<float> routerScores(expertCount);
        std::vector<float> gateBuf(expertFF);
        std::vector<float> upBuf(expertFF);
        std::vector<float> expertOut(hiddenSize);

        for (uint32_t s = 0; s < seqLen; ++s) {
            const float *inputPtr = ffnNorm + s * hiddenSize;

            // Step 1: Apply second pre-FFN norm (preFFWNorm2)
            std::vector<float> normedInput(hiddenSize);
            if (!w.preFFWNorm2.empty()) {
                rmsNormInPlace(inputPtr, normedInput.data(), w.preFFWNorm2.data(), hiddenSize);
            } else {
                std::memcpy(normedInput.data(), inputPtr, hiddenSize * sizeof(float));
            }

            // Step 2: Router - compute expert scores
            np::Array<float> routerOut = w.ffnGateInp.matMulVec(normedInput.data());
            std::memcpy(routerScores.data(), routerOut.data(), expertCount * sizeof(float));

            // Step 3: Select top-k experts
            // Build list of (score, expert_idx) pairs
            std::vector<std::pair<float, uint32_t>> scoredExperts(expertCount);
            for (uint32_t e = 0; e < expertCount; ++e) {
                scoredExperts[e] = {routerScores[e], e};
            }
            // Partial sort to get top-k
            std::partial_sort(scoredExperts.begin(), scoredExperts.begin() + expertUsedCount,
                              scoredExperts.end(),
                              [](const auto &a, const auto &b) { return a.first > b.first; });

            // Step 4-5: Compute expert outputs and combine
            std::fill(expertOut.begin(), expertOut.end(), 0.0f);

            for (uint32_t r = 0; r < expertUsedCount; ++r) {
                uint32_t expertIdx = scoredExperts[r].second;
                float routingWeight = scoredExperts[r].first;

                // Step 4a: Gate+Up projection for this expert
                // Gate: rows [expertIdx * expertFF * 2, expertIdx * expertFF * 2 + expertFF)
                np::Array<float> gateRow = w.ffnGateUpExps.matMulVecRows(
                        normedInput.data(),
                        expertIdx * expertFF * 2,
                        expertFF);
                std::memcpy(gateBuf.data(), gateRow.data(), expertFF * sizeof(float));

                // Up: rows [expertIdx * expertFF * 2 + expertFF, (expertIdx+1) * expertFF * 2)
                np::Array<float> upRow = w.ffnGateUpExps.matMulVecRows(
                        normedInput.data(),
                        expertIdx * expertFF * 2 + expertFF,
                        expertFF);
                std::memcpy(upBuf.data(), upRow.data(), expertFF * sizeof(float));

                // Step 4b: GeGLU activation: gelu(gate) * up
                geluInPlace(gateBuf.data(), expertFF);
                for (uint32_t i = 0; i < expertFF; ++i) {
                    gateBuf[i] *= upBuf[i];
                }

                // Step 4c: Down projection for this expert
                // rows [expertIdx * hiddenSize, (expertIdx+1) * hiddenSize)
                np::Array<float> downRow = w.ffnDownExps.matMulVecRows(
                        gateBuf.data(),
                        expertIdx * hiddenSize,
                        hiddenSize);

                // Step 5: Weighted sum (routing weight * expert output)
                float weight = routingWeight;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    expertOut[i] += weight * downRow.data()[i];
                }
            }

            // Step 6-7: Post-FFN norms
            float *outPtr = ffnOut + s * hiddenSize;
            if (!w.postFFWNorm1.empty()) {
                rmsNormInPlace(expertOut.data(), outPtr, w.postFFWNorm1.data(), hiddenSize);
            } else {
                std::memcpy(outPtr, expertOut.data(), hiddenSize * sizeof(float));
            }
            if (!w.postFFWNorm2.empty()) {
                rmsNormInPlace(outPtr, outPtr, w.postFFWNorm2.data(), hiddenSize);
            }
        }
    }

    void Model::computeQwen35MoE(const float *ffnNorm, float *ffnOut,
                                 uint32_t seqLen, uint32_t hiddenSize,
                                 uint32_t intermediateSize,
                                 const LayerWeights &w) const {
        // Qwen35MoE MoE FFN with top-k expert routing + shared expert.
        //
        // Architecture:
        //   1. Router (ffnGateInpMoe) computes expert scores
        //   2. Top-k experts selected (k = expertUsedCount)
        //   3. For each selected expert:
        //      a. Gate projection from ffnGateExps
        //      b. Up projection from ffnUpExps
        //      c. SwiGLU activation: silu(gate) * up
        //      d. Down projection from ffnDownExpsMoe
        //   4. Shared expert (ffnGateShexp, ffnUpShexp, ffnDownShexp) with router (ffnGateInpShexp)
        //   5. Weighted sum of expert outputs + shared expert output
        //
        // Tensor layouts (after loadQuantized handles 3D→2D flattening):
        //   ffnGateExps: [expertCount * expertFF, hiddenSize]
        //     - Expert e: rows [e * expertFF, (e+1) * expertFF)
        //   ffnUpExps: [expertCount * expertFF, hiddenSize]
        //     - Expert e: rows [e * expertFF, (e+1) * expertFF)
        //   ffnDownExpsMoe: [expertCount * hiddenSize, expertFF]
        //     - Expert e: rows [e * hiddenSize, (e+1) * hiddenSize)
        //   ffnGateInpMoe: [hiddenSize, expertCount] (router)

        uint32_t expertCount = config_.expertCount;
        uint32_t expertUsedCount = config_.expertUsedCount;
        uint32_t expertFF = config_.expertFeedForwardLength;
        uint32_t sharedExpertFF = config_.expertSharedFeedForwardLength;

        if (expertCount == 0 || expertUsedCount == 0) {
            std::cerr << "[TinyCoder] computeQwen35MoE: MoE not configured" << std::endl;
            return;
        }

        // Temporary buffers per token
        std::vector<float> routerScores(expertCount);
        std::vector<float> gateBuf(expertFF);
        std::vector<float> upBuf(expertFF);
        std::vector<float> expertOut(hiddenSize);

        for (uint32_t s = 0; s < seqLen; ++s) {
            const float *inputPtr = ffnNorm + s * hiddenSize;

            // Step 1: Router - compute expert scores
            np::Array<float> routerOut = w.ffnGateInpMoe.matMulVec(inputPtr);
            std::memcpy(routerScores.data(), routerOut.data(), expertCount * sizeof(float));

            // Step 2: Select top-k experts
            std::vector<std::pair<float, uint32_t>> scoredExperts(expertCount);
            for (uint32_t e = 0; e < expertCount; ++e) {
                scoredExperts[e] = {routerScores[e], e};
            }
            std::partial_sort(scoredExperts.begin(), scoredExperts.begin() + expertUsedCount,
                              scoredExperts.end(),
                              [](const auto &a, const auto &b) { return a.first > b.first; });

            // Step 3-5: Compute expert outputs and combine
            std::fill(expertOut.begin(), expertOut.end(), 0.0f);

            for (uint32_t r = 0; r < expertUsedCount; ++r) {
                uint32_t expertIdx = scoredExperts[r].second;
                float routingWeight = scoredExperts[r].first;

                // Gate projection for this expert
                {
                    np::Array<float> gateRow = w.ffnGateExps.matMulVecRows(
                            inputPtr, expertIdx * expertFF, expertFF);
                    std::memcpy(gateBuf.data(), gateRow.data(), expertFF * sizeof(float));
                }

                // Up projection for this expert
                {
                    np::Array<float> upRow = w.ffnUpExps.matMulVecRows(
                            inputPtr, expertIdx * expertFF, expertFF);
                    std::memcpy(upBuf.data(), upRow.data(), expertFF * sizeof(float));
                }

                // SwiGLU activation: silu(gate) * up
                for (uint32_t i = 0; i < expertFF; ++i) {
                    gateBuf[i] = (gateBuf[i] / (1.0f + std::exp(-gateBuf[i]))) * upBuf[i];
                }

                // Down projection for this expert
                {
                    np::Array<float> downRow = w.ffnDownExpsMoe.matMulVecRows(
                            gateBuf.data(), expertIdx * hiddenSize, hiddenSize);
                    // Weighted sum into expertOut
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        expertOut[i] += routingWeight * downRow.data()[i];
                    }
                }
            }

            // Shared expert
            if (!w.ffnGateShexp.empty() && !w.ffnUpShexp.empty() && !w.ffnDownShexp.empty()) {
                // Shared expert router (sigmoid gate)
                float sharedGateWeight = 1.0f;
                if (!w.ffnGateInpShexp.empty()) {
                    np::Array<float> sharedRouterOut = w.ffnGateInpShexp.matMulVec(inputPtr);
                    sharedGateWeight = sharedRouterOut.data()[0];
                    sharedGateWeight = 1.0f / (1.0f + std::exp(-sharedGateWeight));// sigmoid
                }

                // Shared expert gate projection
                np::Array<float> sharedGate = w.ffnGateShexp.matMulVec(inputPtr);
                // Shared expert up projection
                np::Array<float> sharedUp = w.ffnUpShexp.matMulVec(inputPtr);

                // SwiGLU: silu(gate) * up
                for (uint32_t i = 0; i < sharedExpertFF; ++i) {
                    sharedGate.data()[i] = (sharedGate.data()[i] / (1.0f + std::exp(-sharedGate.data()[i]))) * sharedUp.data()[i];
                }

                // Shared expert down projection
                np::Array<float> sharedDown = w.ffnDownShexp.matMulVec(sharedGate.data());

                // Add weighted shared expert output
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    expertOut[i] += sharedGateWeight * sharedDown.data()[i];
                }
            }

            // Write output
            float *outPtr = ffnOut + s * hiddenSize;
            std::memcpy(outPtr, expertOut.data(), hiddenSize * sizeof(float));
        }
    }

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

        // Forward declaration of out-parameter version needed by return-value version
        void deqMatMulVec(const float *W, const float *x,
                          uint32_t rows, uint32_t cols, float *out);

        np::Array<float> deqMatMulVec(const float *W, const float *x,
                                      uint32_t rows, uint32_t cols) {
            // Allocate result and delegate to the out-parameter version
            np::Array<float> result(np::Shape{rows});
            deqMatMulVec(W, x, rows, cols, result.data());
            return result;
        }

        void deqMatMulVec(const float *W, const float *x,
                          uint32_t rows, uint32_t cols, float *out) {
            // This function is ALWAYS called from within a ThreadPool::parallelFor
            // lambda (attention output projection, FFN down projection).
            // Using ThreadPool::parallelFor here would trigger the re-entrancy
            // guard and execute sequentially anyway, but with significant overhead
            // from waking/synchronizing idle workers.
            // Instead, use a plain sequential loop.
            for (uint32_t j = 0; j < rows; ++j) {
                const float *Wrow = W + static_cast<size_t>(j) * cols;
                out[j] = dotProductFMA(x, Wrow, cols);
            }
        }

        // Forward declaration of out-parameter version needed by return-value version
        static void deqMatMulVecF16(const uint16_t *W_f16, const float *x,
                                    uint32_t rows, uint32_t cols, float *out);

        /// @brief Matrix-vector multiply with FP16-stored weights (return-value version).
        ///
        /// Computes y_j = sum_i x[i] * W_f16[j][i] for j in [0, rows).
        /// W_f16 is stored as FP16 (uint16_t) to halve memory bandwidth.
        /// Uses F16C _mm256_cvtph_ps for on-the-fly FP16->FP32 conversion.
        static np::Array<float> deqMatMulVecF16(const uint16_t *W_f16, const float *x,
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
        static void deqMatMulVecF16(const uint16_t *W_f16, const float *x,
                                    uint32_t rows, uint32_t cols, float *out) {
            for (uint32_t j = 0; j < rows; ++j) {
                const uint16_t *Wrow = W_f16 + static_cast<size_t>(j) * cols;
                out[j] = dotProductFMA_F16(x, Wrow, cols);
            }
        }

        /// @brief Fused QKV matrix-vector multiplication.
        ///
        /// Computes Q = x * attnQ^T, K = x * attnK^T, V = x * attnV^T in a single
        /// pass over the input vector x. This reduces x-vector reads from 3× to 1×
        /// per token per layer, improving cache utilization.
        ///
        /// All three matrices must use the same quantized type (typically Q2_K).
        /// For F32 matrices, falls back to three separate calls.
        static void matMulVecFusedQKV(
                const QuantizedMatrix &qMat, const QuantizedMatrix &kMat, const QuantizedMatrix &vMat,
                const float *x,
                float *qOut, float *kOut, float *vOut) {

            uint32_t qRows = qMat.rows;
            uint32_t kRows = kMat.rows;
            uint32_t vRows = vMat.rows;
            uint32_t cols = qMat.cols;// All three share the same input dimension

            // Fall back to separate calls if any matrix is F32 or types differ.
            // The fused optimization requires all matrices to share the same quantized
            // type (and thus the same block size and row stride). In practice, Qwen2
            // models use Q2_K for Q/K but Q4_K for V, so the types differ.
            if (qMat.type != kMat.type || qMat.type != vMat.type ||
                qMat.type == GGML_TYPE_F32) {
                qMat.matMulVec(x, qOut);
                kMat.matMulVec(x, kOut);
                vMat.matMulVec(x, vOut);
                return;
            }

            uint32_t blockSize = ggmlBlockSize(qMat.type);
            uint32_t typeSize = ggmlTypeSize(qMat.type);

            if (blockSize == 0 || typeSize == 0) {
                // Fallback for unknown types
                qMat.matMulVec(x, qOut);
                kMat.matMulVec(x, kOut);
                vMat.matMulVec(x, vOut);
                return;
            }

            uint32_t blocksPerRow = (cols + blockSize - 1) / blockSize;
            uint64_t rowStride = static_cast<uint64_t>(blocksPerRow) * typeSize;

            // Initialize outputs to zero
            std::memset(qOut, 0, static_cast<size_t>(qRows) * sizeof(float));
            std::memset(kOut, 0, static_cast<size_t>(kRows) * sizeof(float));
            std::memset(vOut, 0, static_cast<size_t>(vRows) * sizeof(float));

            // Cache-blocked loop: process one block column across all three matrices.
            // This reads the x vector once instead of three times.
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                uint32_t start = b * blockSize;
                uint32_t n = std::min(blockSize, cols - start);
                const float *xBlock = x + start;

                // Process this block across all Q rows
                for (uint32_t j = 0; j < qRows; ++j) {
                    const uint8_t *blockData = qMat.data.data() + static_cast<uint64_t>(j) * rowStride + static_cast<uint64_t>(b) * typeSize;
                    qOut[j] += GGMLDequantize::dotProductFused(qMat.type, blockData, xBlock, n);
                }

                // Process this block across all K rows
                for (uint32_t j = 0; j < kRows; ++j) {
                    const uint8_t *blockData = kMat.data.data() + static_cast<uint64_t>(j) * rowStride + static_cast<uint64_t>(b) * typeSize;
                    kOut[j] += GGMLDequantize::dotProductFused(kMat.type, blockData, xBlock, n);
                }

                // Process this block across all V rows
                for (uint32_t j = 0; j < vRows; ++j) {
                    const uint8_t *blockData = vMat.data.data() + static_cast<uint64_t>(j) * rowStride + static_cast<uint64_t>(b) * typeSize;
                    vOut[j] += GGMLDequantize::dotProductFused(vMat.type, blockData, xBlock, n);
                }
            }
        }
    }// namespace

    // -----------------------------------------------------------------------

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

            // Check if this is an SSM layer for Qwen35MoE architecture
            bool isSSMLayer = (config_.architecture == ARCH_QWEN35MOE &&
                               config_.fullAttentionInterval > 0 &&
                               (layer % config_.fullAttentionInterval) != 0 &&
                               !w.ssmOut.empty());

            if (isSSMLayer) {
                // ---- SSM (Mamba-style) block replaces attention ----

                // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                rmsNormInPlace(hidden.data(), attnNorm.data(), w.rmsNormAttn.data(), hiddenSize);
                dumpVecStats(attnNorm.data(), hiddenSize, "  After attn_norm (SSM)");

                // SSM computation for single token
                uint32_t ssmInnerSize = config_.ssmInnerSize;
                uint32_t ssmStateSize = config_.ssmStateSize;
                uint32_t ssmConvKernel = config_.ssmConvKernel;

                // Step 1: Input projection (hiddenSize → ssmInnerSize)
                np::Array<float> ssmIn = w.ssmOut.matMulVec(attnNorm.data());
                float *ssmInData = ssmIn.data();

                // Step 2: Conv1d with past buffer
                std::vector<float> convOut(ssmInnerSize);
                if (ssmConvKernel > 1 && !w.ssmConv1d.empty()) {
                    std::vector<float> convInput(ssmConvKernel * ssmInnerSize);
                    auto &convBuf = kvCache_.ssmConvBuf[layer];
                    uint32_t bufLen = ssmConvKernel - 1;
                    for (uint32_t i = 0; i < bufLen * ssmInnerSize; ++i) {
                        convInput[i] = convBuf[i];
                    }
                    std::memcpy(convInput.data() + bufLen * ssmInnerSize,
                                ssmInData, ssmInnerSize * sizeof(float));

                    for (uint32_t i = 0; i < (bufLen - 1) * ssmInnerSize; ++i) {
                        convBuf[i] = convBuf[i + ssmInnerSize];
                    }
                    std::memcpy(convBuf.data() + (bufLen - 1) * ssmInnerSize,
                                ssmInData, ssmInnerSize * sizeof(float));

                    for (uint32_t c = 0; c < ssmInnerSize; ++c) {
                        double dot = 0.0;
                        const float *wRow = reinterpret_cast<const float *>(w.ssmConv1d.data.data()) + static_cast<size_t>(c) * ssmConvKernel;
                        for (uint32_t k = 0; k < ssmConvKernel; ++k) {
                            dot += static_cast<double>(wRow[k]) * convInput[c * ssmConvKernel + k];
                        }
                        convOut[c] = static_cast<float>(dot);
                    }
                } else {
                    std::memcpy(convOut.data(), ssmInData, ssmInnerSize * sizeof(float));
                }

                // Step 3: SiLU activation on conv output
                for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                    convOut[i] = convOut[i] / (1.0f + std::exp(-convOut[i]));
                }

                // Step 4: SSM state update
                auto &ssmState = kvCache_.ssmState[layer];
                for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                    float dt = std::log(1.0f + std::exp(w.ssmDtBias.data()[i]));
                    for (uint32_t j = 0; j < ssmStateSize; ++j) {
                        float aVal = w.ssmA.data()[i * ssmStateSize + j];
                        float aBar = std::exp(aVal * dt);
                        uint32_t idx = i * ssmStateSize + j;
                        ssmState[idx] = aBar * ssmState[idx] + convOut[i];
                    }
                }

                // Step 5: Output from SSM state
                std::vector<float> ssmOutBuf(ssmInnerSize);
                for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                    double hVal = 0.0;
                    for (uint32_t j = 0; j < ssmStateSize; ++j) {
                        hVal += ssmState[i * ssmStateSize + j];
                    }
                    float gate = w.ssmAlpha.data()[i] * static_cast<float>(hVal) + w.ssmBeta.data()[i];
                    float gateAct = gate / (1.0f + std::exp(-gate));
                    ssmOutBuf[i] = convOut[i] * gateAct;
                }

                // Step 6: Output projection back to hiddenSize using attnO
                np::Array<float> projRow = deqMatMulVecF16(w.attnO_deq_f16.data(), ssmOutBuf.data(),
                                                           w.attnO.rows, w.attnO.cols);
                std::memcpy(attnProj.data(), projRow.data(), hiddenSize * sizeof(float));
                dumpVecStats(attnProj.data(), hiddenSize, "  After SSM proj");

                // SSM residual (standard residual, no post-norm)
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hidden[i] += attnProj[i];
                }
                dumpVecStats(hidden.data(), hiddenSize, "  After SSM + residual");
            } else {
                // ---- Attention block ----

                // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                rmsNormInPlace(hidden.data(), attnNorm.data(), w.rmsNormAttn.data(), hiddenSize);
                dumpVecStats(attnNorm.data(), hiddenSize, "  After attn_norm");

                // Q projection (architecture-aware bias)
                np::Array<float> qRow = w.attnQ.matMulVec(attnNorm.data());
                std::memcpy(q.data(), qRow.data(), nHeads * headDim * sizeof(float));
                if (config_.architecture == ARCH_QWEN2 && !w.attnQBias.empty()) {
                    const float *qBias = w.attnQBias.data();
                    for (uint32_t i = 0; i < nHeads * headDim; ++i)
                        q[i] += qBias[i];
                }
                dumpVecStats(q.data(), nHeads * headDim, "  Q (after proj)");

                // K projection (architecture-aware bias)
                np::Array<float> kRow = w.attnK.matMulVec(attnNorm.data());
                std::memcpy(k.data(), kRow.data(), nKVHeads * headDim * sizeof(float));
                if (config_.architecture == ARCH_QWEN2 && !w.attnKBias.empty()) {
                    const float *kBias = w.attnKBias.data();
                    for (uint32_t i = 0; i < nKVHeads * headDim; ++i)
                        k[i] += kBias[i];
                }

                // V projection (architecture-aware bias)
                np::Array<float> vRow = w.attnV.matMulVec(attnNorm.data());
                std::memcpy(v.data(), vRow.data(), nKVHeads * headDim * sizeof(float));
                if (config_.architecture == ARCH_QWEN2 && !w.attnVBias.empty()) {
                    const float *vBias = w.attnVBias.data();
                    for (uint32_t i = 0; i < nKVHeads * headDim; ++i)
                        v[i] += vBias[i];
                }

                // Apply Q/K norms before RoPE (Gemma4 and Qwen35MoE)
                if (config_.architecture == ARCH_GEMMA4 && !w.attnQNorm.empty() && !w.attnKNorm.empty()) {
                    applyQKNorms(q.data(), k.data(), 1, nHeads, nKVHeads,
                                 w.attnQNorm.data(), w.attnKNorm.data());
                } else if (config_.architecture == ARCH_QWEN35MOE && !w.attnQNormMoe.empty() && !w.attnKNormMoe.empty()) {
                    applyQKNorms(q.data(), k.data(), 1, nHeads, nKVHeads,
                                 w.attnQNormMoe.data(), w.attnKNormMoe.data());
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
                np::Array<float> projRow = deqMatMulVecF16(w.attnO_deq_f16.data(), attnOut.data(),
                                                           w.attnO.rows, w.attnO.cols);
                std::memcpy(attnProj.data(), projRow.data(), hiddenSize * sizeof(float));
                dumpVecStats(attnProj.data(), hiddenSize, "  After attnO proj");

                // Attention residual + post-attention processing (architecture-aware)
                if (config_.architecture == ARCH_GEMMA4 && !w.postAttnNorm.empty()) {
                    // Gemma4: post-attention norm + layer scale
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        hidden[i] += attnProj[i];
                    }
                    rmsNormInPlace(hidden.data(), hidden.data(), w.postAttnNorm.data(), hiddenSize);
                    if (!w.layerOutputScale.empty()) {
                        float scale = w.layerOutputScale.data()[0];
                        for (uint32_t i = 0; i < hiddenSize; ++i) {
                            hidden[i] *= scale;
                        }
                    }
                } else {
                    // Qwen2, Qwen35MoE: standard residual (no post-norm)
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        hidden[i] += attnProj[i];
                    }
                }
                dumpVecStats(hidden.data(), hiddenSize, "  After attention + residual");
            }

            // ---- FFN block ----

            // RMSNorm: ffnNorm = rmsNorm(hidden, w.rmsNormFFN)
            rmsNormInPlace(hidden.data(), ffnNorm.data(), w.rmsNormFFN.data(), hiddenSize);
            dumpVecStats(ffnNorm.data(), hiddenSize, "  After ffn_norm");

            // Gemma4 MoE path (expertCount > 0)
            if (config_.architecture == ARCH_GEMMA4 && config_.expertCount > 0) {
                computeGemma4MoE(ffnNorm.data(), ffnOut.data(), 1, hiddenSize, intermediateSize, w);
                dumpVecStats(ffnOut.data(), hiddenSize, "  After MoE FFN");
                // Residual + layer scale
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hidden[i] += ffnOut[i];
                }
                if (!w.layerOutputScale.empty()) {
                    float scale = w.layerOutputScale.data()[0];
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        hidden[i] *= scale;
                    }
                }
            } else if (config_.architecture == ARCH_QWEN35MOE && config_.expertCount > 0) {
                // Qwen35MoE MoE path
                computeQwen35MoE(ffnNorm.data(), ffnOut.data(), 1, hiddenSize, intermediateSize, w);
                dumpVecStats(ffnOut.data(), hiddenSize, "  After Qwen35MoE MoE FFN");
                // Standard residual (no post-norm)
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hidden[i] += ffnOut[i];
                }
            } else {
                // Dense FFN path

                // Gate projection
                np::Array<float> gateRow = w.ffnGate.matMulVec(ffnNorm.data());
                std::memcpy(gate.data(), gateRow.data(), intermediateSize * sizeof(float));
                dumpVecStats(gate.data(), intermediateSize, "  Gate (before activation)");

                // Up projection
                np::Array<float> upRow = w.ffnUp.matMulVec(ffnNorm.data());
                std::memcpy(up.data(), upRow.data(), intermediateSize * sizeof(float));
                dumpVecStats(up.data(), intermediateSize, "  Up (before activation)");

                // FFN activation (architecture-aware)
                if (config_.architecture == ARCH_GEMMA4) {
                    // Gemma4: GeGLU activation (gelu(gate) * up)
                    geluInPlace(gate.data(), intermediateSize);
                    for (uint32_t i = 0; i < intermediateSize; ++i) {
                        gate[i] *= up[i];
                    }
                    dumpVecStats(gate.data(), intermediateSize, "  After GeGLU");
                } else {
                    // Qwen2, Qwen35MoE: SwiGLU activation (silu(gate) * up)
                    swiGLUInPlace(gate.data(), up.data(), intermediateSize);
                    dumpVecStats(gate.data(), intermediateSize, "  After SwiGLU");
                }

                // Down projection (using dequantized F32 weights for exact float dot product)
                np::Array<float> downRow = deqMatMulVecF16(w.ffnDown_deq_f16.data(), gate.data(),
                                                           w.ffnDown.rows, w.ffnDown.cols);
                std::memcpy(ffnOut.data(), downRow.data(), hiddenSize * sizeof(float));
                dumpVecStats(ffnOut.data(), hiddenSize, "  After ffnDown proj");

                // FFN residual + post-FFN processing (architecture-aware)
                if (config_.architecture == ARCH_GEMMA4 && !w.postFFWNorm.empty()) {
                    // Gemma4: post-FFN norm + layer scale
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        hidden[i] += ffnOut[i];
                    }
                    rmsNormInPlace(hidden.data(), hidden.data(), w.postFFWNorm.data(), hiddenSize);
                    if (!w.layerOutputScale.empty()) {
                        float scale = w.layerOutputScale.data()[0];
                        for (uint32_t i = 0; i < hiddenSize; ++i) {
                            hidden[i] *= scale;
                        }
                    }
                } else {
                    // Qwen2, Qwen35MoE: standard residual (no post-norm)
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        hidden[i] += ffnOut[i];
                    }
                }
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

        // Apply final logit softcapping (Gemma4 architecture)
        if (config_.architecture == ARCH_GEMMA4 && config_.finalLogitSoftcapping > 0.0f) {
            softcapInPlace(logits.data(), vocabSize, config_.finalLogitSoftcapping);
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

            // Check if this is an SSM layer for Qwen35MoE architecture
            bool isSSMLayer = (config_.architecture == ARCH_QWEN35MOE &&
                               config_.fullAttentionInterval > 0 &&
                               (layer % config_.fullAttentionInterval) != 0 &&
                               !w.ssmOut.empty());

            if (isSSMLayer) {
                // ---- SSM (Mamba-style) block replaces attention ----

                // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                rmsNormInPlace(hidden.data(), attnNorm.data(), w.rmsNormAttn.data(), hiddenSize);

                // SSM computation for single token
                uint32_t ssmInnerSize = config_.ssmInnerSize;
                uint32_t ssmStateSize = config_.ssmStateSize;
                uint32_t ssmConvKernel = config_.ssmConvKernel;

                // Step 1: Input projection (hiddenSize → ssmInnerSize)
                np::Array<float> ssmIn = w.ssmOut.matMulVec(attnNorm.data());
                float *ssmInData = ssmIn.data();

                // Step 2: Conv1d with past buffer
                std::vector<float> convOut(ssmInnerSize);
                if (ssmConvKernel > 1 && !w.ssmConv1d.empty()) {
                    std::vector<float> convInput(ssmConvKernel * ssmInnerSize);
                    auto &convBuf = kvCache_.ssmConvBuf[layer];
                    uint32_t bufLen = ssmConvKernel - 1;
                    for (uint32_t i = 0; i < bufLen * ssmInnerSize; ++i) {
                        convInput[i] = convBuf[i];
                    }
                    std::memcpy(convInput.data() + bufLen * ssmInnerSize,
                                ssmInData, ssmInnerSize * sizeof(float));

                    for (uint32_t i = 0; i < (bufLen - 1) * ssmInnerSize; ++i) {
                        convBuf[i] = convBuf[i + ssmInnerSize];
                    }
                    std::memcpy(convBuf.data() + (bufLen - 1) * ssmInnerSize,
                                ssmInData, ssmInnerSize * sizeof(float));

                    for (uint32_t c = 0; c < ssmInnerSize; ++c) {
                        double dot = 0.0;
                        const float *wRow = reinterpret_cast<const float *>(w.ssmConv1d.data.data()) + static_cast<size_t>(c) * ssmConvKernel;
                        for (uint32_t k = 0; k < ssmConvKernel; ++k) {
                            dot += static_cast<double>(wRow[k]) * convInput[c * ssmConvKernel + k];
                        }
                        convOut[c] = static_cast<float>(dot);
                    }
                } else {
                    std::memcpy(convOut.data(), ssmInData, ssmInnerSize * sizeof(float));
                }

                // Step 3: SiLU activation on conv output
                for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                    convOut[i] = convOut[i] / (1.0f + std::exp(-convOut[i]));
                }

                // Step 4: SSM state update
                auto &ssmState = kvCache_.ssmState[layer];
                for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                    float dt = std::log(1.0f + std::exp(w.ssmDtBias.data()[i]));
                    for (uint32_t j = 0; j < ssmStateSize; ++j) {
                        float aVal = w.ssmA.data()[i * ssmStateSize + j];
                        float aBar = std::exp(aVal * dt);
                        uint32_t idx = i * ssmStateSize + j;
                        ssmState[idx] = aBar * ssmState[idx] + convOut[i];
                    }
                }

                // Step 5: Output from SSM state
                std::vector<float> ssmOutBuf(ssmInnerSize);
                for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                    double hVal = 0.0;
                    for (uint32_t j = 0; j < ssmStateSize; ++j) {
                        hVal += ssmState[i * ssmStateSize + j];
                    }
                    float gate = w.ssmAlpha.data()[i] * static_cast<float>(hVal) + w.ssmBeta.data()[i];
                    float gateAct = gate / (1.0f + std::exp(-gate));
                    ssmOutBuf[i] = convOut[i] * gateAct;
                }

                // Step 6: Output projection back to hiddenSize using attnO
                np::Array<float> projRow = deqMatMulVecF16(w.attnO_deq_f16.data(), ssmOutBuf.data(),
                                                           w.attnO.rows, w.attnO.cols);
                std::memcpy(attnProj.data(), projRow.data(), hiddenSize * sizeof(float));

                // SSM residual (standard residual, no post-norm)
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hidden[i] += attnProj[i];
                }
            } else {
                // ---- Attention block ----

                // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                rmsNormInPlace(hidden.data(), attnNorm.data(), w.rmsNormAttn.data(), hiddenSize);

                // Q projection (architecture-aware bias)
                np::Array<float> qRow = w.attnQ.matMulVec(attnNorm.data());
                std::memcpy(q.data(), qRow.data(), nHeads * headDim * sizeof(float));
                if (config_.architecture == ARCH_QWEN2 && !w.attnQBias.empty()) {
                    const float *qBias = w.attnQBias.data();
                    for (uint32_t i = 0; i < nHeads * headDim; ++i)
                        q[i] += qBias[i];
                }

                // K projection (architecture-aware bias)
                np::Array<float> kRow = w.attnK.matMulVec(attnNorm.data());
                std::memcpy(k.data(), kRow.data(), nKVHeads * headDim * sizeof(float));
                if (config_.architecture == ARCH_QWEN2 && !w.attnKBias.empty()) {
                    const float *kBias = w.attnKBias.data();
                    for (uint32_t i = 0; i < nKVHeads * headDim; ++i)
                        k[i] += kBias[i];
                }

                // V projection (architecture-aware bias)
                np::Array<float> vRow = w.attnV.matMulVec(attnNorm.data());
                std::memcpy(v.data(), vRow.data(), nKVHeads * headDim * sizeof(float));
                if (config_.architecture == ARCH_QWEN2 && !w.attnVBias.empty()) {
                    const float *vBias = w.attnVBias.data();
                    for (uint32_t i = 0; i < nKVHeads * headDim; ++i)
                        v[i] += vBias[i];
                }

                // Apply Q/K norms before RoPE (Gemma4 and Qwen35MoE)
                if (config_.architecture == ARCH_GEMMA4 && !w.attnQNorm.empty() && !w.attnKNorm.empty()) {
                    applyQKNorms(q.data(), k.data(), 1, nHeads, nKVHeads,
                                 w.attnQNorm.data(), w.attnKNorm.data());
                } else if (config_.architecture == ARCH_QWEN35MOE && !w.attnQNormMoe.empty() && !w.attnKNormMoe.empty()) {
                    applyQKNorms(q.data(), k.data(), 1, nHeads, nKVHeads,
                                 w.attnQNormMoe.data(), w.attnKNormMoe.data());
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
                np::Array<float> projRow = deqMatMulVecF16(w.attnO_deq_f16.data(), attnOut.data(),
                                                           w.attnO.rows, w.attnO.cols);
                std::memcpy(attnProj.data(), projRow.data(), hiddenSize * sizeof(float));

                // Attention residual + post-attention processing (architecture-aware)
                if (config_.architecture == ARCH_GEMMA4 && !w.postAttnNorm.empty()) {
                    // Gemma4: post-attention norm + layer scale
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        hidden[i] += attnProj[i];
                    }
                    rmsNormInPlace(hidden.data(), hidden.data(), w.postAttnNorm.data(), hiddenSize);
                    if (!w.layerOutputScale.empty()) {
                        float scale = w.layerOutputScale.data()[0];
                        for (uint32_t i = 0; i < hiddenSize; ++i) {
                            hidden[i] *= scale;
                        }
                    }
                } else {
                    // Qwen2, Qwen35MoE: standard residual (no post-norm)
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        hidden[i] += attnProj[i];
                    }
                }
            }

            // ---- FFN block ----

            // RMSNorm: ffnNorm = rmsNorm(hidden, w.rmsNormFFN)
            rmsNormInPlace(hidden.data(), ffnNorm.data(), w.rmsNormFFN.data(), hiddenSize);

            // Gemma4 MoE path (expertCount > 0)
            if (config_.architecture == ARCH_GEMMA4 && config_.expertCount > 0) {
                computeGemma4MoE(ffnNorm.data(), ffnOut.data(), 1, hiddenSize, intermediateSize, w);
                // Residual + layer scale
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hidden[i] += ffnOut[i];
                }
                if (!w.layerOutputScale.empty()) {
                    float scale = w.layerOutputScale.data()[0];
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        hidden[i] *= scale;
                    }
                }
            } else if (config_.architecture == ARCH_QWEN35MOE && config_.expertCount > 0) {
                // Qwen35MoE MoE path
                computeQwen35MoE(ffnNorm.data(), ffnOut.data(), 1, hiddenSize, intermediateSize, w);
                // Standard residual (no post-norm)
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    hidden[i] += ffnOut[i];
                }
            } else {
                // Dense FFN path

                // Gate projection
                np::Array<float> gateRow = w.ffnGate.matMulVec(ffnNorm.data());
                std::memcpy(gate.data(), gateRow.data(), intermediateSize * sizeof(float));

                // Up projection
                np::Array<float> upRow = w.ffnUp.matMulVec(ffnNorm.data());
                std::memcpy(up.data(), upRow.data(), intermediateSize * sizeof(float));

                // FFN activation (architecture-aware)
                if (config_.architecture == ARCH_GEMMA4) {
                    // Gemma4: GeGLU activation (gelu(gate) * up)
                    geluInPlace(gate.data(), intermediateSize);
                    for (uint32_t i = 0; i < intermediateSize; ++i) {
                        gate[i] *= up[i];
                    }
                } else {
                    // Qwen2, Qwen35MoE: SwiGLU activation (silu(gate) * up)
                    swiGLUInPlace(gate.data(), up.data(), intermediateSize);
                }

                // Down projection (using dequantized F32 weights for exact float dot product)
                np::Array<float> downRow = deqMatMulVecF16(w.ffnDown_deq_f16.data(), gate.data(),
                                                           w.ffnDown.rows, w.ffnDown.cols);
                std::memcpy(ffnOut.data(), downRow.data(), hiddenSize * sizeof(float));

                // FFN residual + post-FFN processing (architecture-aware)
                if (config_.architecture == ARCH_GEMMA4 && !w.postFFWNorm.empty()) {
                    // Gemma4: post-FFN norm + layer scale
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        hidden[i] += ffnOut[i];
                    }
                    rmsNormInPlace(hidden.data(), hidden.data(), w.postFFWNorm.data(), hiddenSize);
                    if (!w.layerOutputScale.empty()) {
                        float scale = w.layerOutputScale.data()[0];
                        for (uint32_t i = 0; i < hiddenSize; ++i) {
                            hidden[i] *= scale;
                        }
                    }
                } else {
                    // Qwen2, Qwen35MoE: standard residual (no post-norm)
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        hidden[i] += ffnOut[i];
                    }
                }
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

        // Apply final logit softcapping (Gemma4 architecture)
        if (config_.architecture == ARCH_GEMMA4 && config_.finalLogitSoftcapping > 0.0f) {
            softcapInPlace(logits.data(), vocabSize, config_.finalLogitSoftcapping);
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
        ThreadPool::instance().parallelFor(0, seqLen, [&](uint32_t i) {
            int32_t tokenId = tokens[i];
            if (tokenId >= 0 &&
                tokenId < static_cast<int32_t>(quantizedEmbeddings_.vocabSize)) {
                auto embRow = quantizedEmbeddings_.getRow(tokenId);
                float *hRow = hiddenData + i * hiddenSize;
                std::memcpy(hRow, embRow.data(), hiddenSize * sizeof(float));
            }
        });

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

        // Process through all transformer layers
        for (uint32_t layer = 0; layer < nLayers; ++layer) {
            auto &w = layers_[layer];

            // Check if this is an SSM layer for Qwen35MoE architecture
            bool isSSMLayer = (config_.architecture == ARCH_QWEN35MOE &&
                               config_.fullAttentionInterval > 0 &&
                               (layer % config_.fullAttentionInterval) != 0 &&
                               !w.ssmOut.empty());

            if (isSSMLayer) {
                // ---- SSM (Mamba-style) block replaces attention ----

                // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                const float *rmsNormAttnData = w.rmsNormAttn.data();
                ThreadPool::instance().parallelFor(0, seqLen, [&](uint32_t s) {
                    rmsNormSIMD(hiddenData + s * hiddenSize, attnNormData + s * hiddenSize,
                                rmsNormAttnData, hiddenSize);
                });

                // SSM computation for each token
                uint32_t ssmInnerSize = config_.ssmInnerSize;
                uint32_t ssmStateSize = config_.ssmStateSize;
                uint32_t ssmConvKernel = config_.ssmConvKernel;

                // Pre-allocate SSM input buffer (reused across tokens)
                std::vector<float> ssmInBuf(ssmInnerSize);

                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *hRowPtr = attnNormData + s * hiddenSize;

                    // Step 1: Input projection (hiddenSize → ssmInnerSize)
                    // Write directly to pre-allocated buffer, avoiding heap allocation + memcpy
                    w.ssmOut.matMulVec(hRowPtr, ssmInBuf.data());

                    // Step 2: Conv1d with past buffer
                    std::vector<float> convOut(ssmInnerSize);
                    if (ssmConvKernel > 1 && !w.ssmConv1d.empty()) {
                        std::vector<float> convInput(ssmConvKernel * ssmInnerSize);
                        auto &convBuf = kvCache_.ssmConvBuf[layer];
                        uint32_t bufLen = ssmConvKernel - 1;
                        std::memcpy(convInput.data(), convBuf.data(), bufLen * ssmInnerSize * sizeof(float));
                        std::memcpy(convInput.data() + bufLen * ssmInnerSize,
                                    ssmInBuf.data(), ssmInnerSize * sizeof(float));

                        // Update conv buffer with current input (shift)
                        std::memmove(convBuf.data(), convBuf.data() + ssmInnerSize,
                                     (bufLen - 1) * ssmInnerSize * sizeof(float));
                        std::memcpy(convBuf.data() + (bufLen - 1) * ssmInnerSize,
                                    ssmInBuf.data(), ssmInnerSize * sizeof(float));

                        for (uint32_t c = 0; c < ssmInnerSize; ++c) {
                            const float *wRow = reinterpret_cast<const float *>(w.ssmConv1d.data.data()) + static_cast<size_t>(c) * ssmConvKernel;
                            convOut[c] = dotProductFMA(wRow, convInput.data() + c * ssmConvKernel, ssmConvKernel);
                        }
                    } else {
                        std::memcpy(convOut.data(), ssmInBuf.data(), ssmInnerSize * sizeof(float));
                    }

                    // Step 3: SiLU activation on conv output
                    siluSIMD(convOut.data(), ssmInnerSize);

                    // Step 4: SSM state update
                    auto &ssmState = kvCache_.ssmState[layer];
                    for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                        float dt = std::log(1.0f + std::exp(w.ssmDtBias.data()[i]));
                        for (uint32_t j = 0; j < ssmStateSize; ++j) {
                            float aVal = w.ssmA.data()[i * ssmStateSize + j];
                            float aBar = std::exp(aVal * dt);
                            uint32_t idx = i * ssmStateSize + j;
                            ssmState[idx] = aBar * ssmState[idx] + convOut[i];
                        }
                    }

                    // Step 5: Output from SSM state
                    std::vector<float> ssmOutBuf(ssmInnerSize);
                    for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                        double hVal = 0.0;
                        for (uint32_t j = 0; j < ssmStateSize; ++j) {
                            hVal += ssmState[i * ssmStateSize + j];
                        }
                        float gateVal = w.ssmAlpha.data()[i] * static_cast<float>(hVal) + w.ssmBeta.data()[i];
                        float gateAct = gateVal / (1.0f + std::exp(-gateVal));
                        ssmOutBuf[i] = convOut[i] * gateAct;
                    }

                    // Step 6: Output projection back to hiddenSize using attnO
                    // Write directly to pre-allocated attnProjData buffer
                    deqMatMulVecF16(w.attnO_deq_f16.data(), ssmOutBuf.data(),
                                    w.attnO.rows, w.attnO.cols,
                                    attnProjData + s * hiddenSize);
                }

                // SSM residual (standard residual, no post-norm)
                for (uint32_t s = 0; s < seqLen; ++s) {
                    addSIMD(hiddenData + s * hiddenSize, attnProjData + s * hiddenSize, hiddenSize);
                }
            } else {
                // ---- Attention block ----

                // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                const float *rmsNormAttnData = w.rmsNormAttn.data();
                ThreadPool::instance().parallelFor(0, seqLen, [&](uint32_t s) {
                    rmsNormSIMD(hiddenData + s * hiddenSize, attnNormData + s * hiddenSize,
                                rmsNormAttnData, hiddenSize);
                });

                // Project to Q, K, V using fused quantized matrix-vector multiplication.
                // Fused QKV processes all three projections in a single pass over the
                // input vector x, reducing x-vector reads from 3× to 1× per token.
                if (config_.architecture == ARCH_QWEN2) {
                    // Qwen2: separate Q/K/V with biases
                    ThreadPool::instance().parallelFor(0, seqLen, [&](uint32_t s) {
                        const float *hRowPtr = attnNormData + s * hiddenSize;
                        float *qRowPtr = qData + s * nHeads * headDim;
                        float *kRowPtr = kData + s * nKVHeads * headDim;
                        float *vRowPtr = vData + s * nKVHeads * headDim;

                        // Fused QKV: single pass over x for all three projections
                        matMulVecFusedQKV(w.attnQ, w.attnK, w.attnV, hRowPtr,
                                          qRowPtr, kRowPtr, vRowPtr);

                        if (!w.attnQBias.empty()) {
                            addSIMD(qRowPtr, w.attnQBias.data(), nHeads * headDim);
                        }
                        if (!w.attnKBias.empty()) {
                            addSIMD(kRowPtr, w.attnKBias.data(), nKVHeads * headDim);
                        }
                        if (!w.attnVBias.empty()) {
                            addSIMD(vRowPtr, w.attnVBias.data(), nKVHeads * headDim);
                        }
                    });
                } else {
                    // Gemma4, Qwen35MoE: no Q/K/V biases
                    ThreadPool::instance().parallelFor(0, seqLen, [&](uint32_t s) {
                        const float *hRowPtr = attnNormData + s * hiddenSize;
                        float *qRowPtr = qData + s * nHeads * headDim;
                        float *kRowPtr = kData + s * nKVHeads * headDim;
                        float *vRowPtr = vData + s * nKVHeads * headDim;

                        // Fused QKV: single pass over x for all three projections
                        matMulVecFusedQKV(w.attnQ, w.attnK, w.attnV, hRowPtr,
                                          qRowPtr, kRowPtr, vRowPtr);
                    });
                }

                // Apply Q/K norms before RoPE (Gemma4 and Qwen35MoE)
                if (config_.architecture == ARCH_GEMMA4) {
                    if (!w.attnQNorm.empty() && !w.attnKNorm.empty()) {
                        applyQKNorms(qData, kData, seqLen, nHeads, nKVHeads,
                                     w.attnQNorm.data(), w.attnKNorm.data());
                    }
                } else if (config_.architecture == ARCH_QWEN35MOE) {
                    if (!w.attnQNormMoe.empty() && !w.attnKNormMoe.empty()) {
                        applyQKNorms(qData, kData, seqLen, nHeads, nKVHeads,
                                     w.attnQNormMoe.data(), w.attnKNormMoe.data());
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
                    std::memcpy(kDst, kSrc, kvSize * sizeof(float));
                    std::memcpy(vDst, vSrc, kvSize * sizeof(float));
                }

                // Attention with cached K, V
                uint32_t totalCacheLen = cachePos + seqLen;
                attentionFused(qData, kCacheLayer, vCacheLayer, attnOutData, seqLen,
                               cachePos, totalCacheLen, layer);

                // Output projection using dequantized F32 weights
                // Write directly to pre-allocated attnProjData buffer
                ThreadPool::instance().parallelFor(0, seqLen, [&](uint32_t s) {
                    const float *attnRowPtr = attnOutData + s * nHeads * headDim;
                    deqMatMulVecF16(w.attnO_deq_f16.data(), attnRowPtr,
                                    w.attnO.rows, w.attnO.cols,
                                    attnProjData + s * hiddenSize);
                });

                // Attention residual + post-attention processing
                if (config_.architecture == ARCH_GEMMA4 && !w.postAttnNorm.empty()) {
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        float *hPtr = hiddenData + s * hiddenSize;
                        const float *aPtr = attnProjData + s * hiddenSize;
                        addSIMD(hPtr, aPtr, hiddenSize);
                        rmsNormSIMD(hPtr, hPtr, w.postAttnNorm.data(), hiddenSize);
                        if (!w.layerOutputScale.empty()) {
                            scaleSIMD(hPtr, w.layerOutputScale.data()[0], hiddenSize);
                        }
                    }
                } else {
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        addSIMD(hiddenData + s * hiddenSize, attnProjData + s * hiddenSize, hiddenSize);
                    }
                }
            }

            // ---- FFN block ----

            // RMSNorm: ffnNorm = rmsNorm(hidden, w.rmsNormFFN)
            const float *rmsNormFFNData = w.rmsNormFFN.data();
            ThreadPool::instance().parallelFor(0, seqLen, [&](uint32_t s) {
                rmsNormSIMD(hiddenData + s * hiddenSize, ffnNormData + s * hiddenSize,
                            rmsNormFFNData, hiddenSize);
            });

            // MoE path (expertCount > 0)
            if (config_.architecture == ARCH_GEMMA4 && config_.expertCount > 0) {
                computeGemma4MoE(ffnNormData, ffnOut.data(), seqLen, hiddenSize, intermediateSize, w);
                for (uint32_t s = 0; s < seqLen; ++s) {
                    float *hPtr = hiddenData + s * hiddenSize;
                    const float *fPtr = ffnOut.data() + s * hiddenSize;
                    addSIMD(hPtr, fPtr, hiddenSize);
                    if (!w.layerOutputScale.empty()) {
                        scaleSIMD(hPtr, w.layerOutputScale.data()[0], hiddenSize);
                    }
                }
            } else if (config_.architecture == ARCH_QWEN35MOE && config_.expertCount > 0) {
                computeQwen35MoE(ffnNormData, ffnOut.data(), seqLen, hiddenSize, intermediateSize, w);
                for (uint32_t s = 0; s < seqLen; ++s) {
                    addSIMD(hiddenData + s * hiddenSize, ffnOut.data() + s * hiddenSize, hiddenSize);
                }
            } else {
                // Dense FFN path
                if (config_.architecture == ARCH_GEMMA4) {
                    // Gemma4: GeGLU activation (gelu(gate) * up)
                    ThreadPool::instance().parallelFor(0, seqLen, [&](uint32_t s) {
                        const float *ffnRowPtr = ffnNormData + s * hiddenSize;
                        float *gatePtr = gateData + s * intermediateSize;
                        float *upPtr = upData + s * intermediateSize;

                        // Write directly to pre-allocated gate/up buffers
                        w.ffnGate.matMulVec(ffnRowPtr, gatePtr);
                        w.ffnUp.matMulVec(ffnRowPtr, upPtr);

                        // GeGLU: gelu(gate) * up
                        geluInPlace(gatePtr, intermediateSize);
                        for (uint32_t i = 0; i < intermediateSize; ++i) {
                            gatePtr[i] *= upPtr[i];
                        }
                    });
                } else {
                    // Qwen2, Qwen35MoE: SwiGLU activation (silu(gate) * up)
                    ThreadPool::instance().parallelFor(0, seqLen, [&](uint32_t s) {
                        const float *ffnRowPtr = ffnNormData + s * hiddenSize;
                        float *gatePtr = gateData + s * intermediateSize;
                        float *upPtr = upData + s * intermediateSize;

                        // Write directly to pre-allocated gate/up buffers
                        w.ffnGate.matMulVec(ffnRowPtr, gatePtr);
                        w.ffnUp.matMulVec(ffnRowPtr, upPtr);

                        // SwiGLU: silu(gate) * up
                        swiGLUSIMD(gatePtr, upPtr, intermediateSize);
                    });
                }

                // Down projection fused with residual
                // Use ffnOut as temp buffer for the down projection result,
                // then add to hidden (residual connection)
                ThreadPool::instance().parallelFor(0, seqLen, [&](uint32_t s) {
                    const float *ffnActPtr = gateData + s * intermediateSize;
                    float *downBuf = ffnOut.data() + s * hiddenSize;
                    float *hPtr = hiddenData + s * hiddenSize;
                    deqMatMulVecF16(w.ffnDown_deq_f16.data(), ffnActPtr,
                                    w.ffnDown.rows, w.ffnDown.cols,
                                    downBuf);
                    addSIMD(hPtr, downBuf, hiddenSize);
                });

                // FFN residual + post-FFN processing
                if (config_.architecture == ARCH_GEMMA4 && !w.postFFWNorm.empty()) {
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        float *hPtr = hiddenData + s * hiddenSize;
                        rmsNormSIMD(hPtr, hPtr, w.postFFWNorm.data(), hiddenSize);
                        if (!w.layerOutputScale.empty()) {
                            scaleSIMD(hPtr, w.layerOutputScale.data()[0], hiddenSize);
                        }
                    }
                }
            }
        }

        // Final RMSNorm (parallel over tokens)
        const float *finalNormData = finalNorm_.data();
        ThreadPool::instance().parallelFor(0, seqLen, [&](uint32_t s) {
            rmsNormSIMD(hiddenData + s * hiddenSize, hiddenData + s * hiddenSize,
                        finalNormData, hiddenSize);
        });

        // LM head (logits)
        np::Array<float> logits = np::Array<float>(np::Shape{seqLen, vocabSize});
        float *logitsData = logits.data();

        if (lmHeadTied_) {
#ifdef USE_CUDA
            // CUDA path (unchanged)
            static bool embedUploaded = false;
            static const float *d_embeddings = nullptr;
            static uint32_t cachedVocabSize = 0;
            static uint32_t cachedHiddenSize = 0;

            if (!embedUploaded || cachedVocabSize != quantizedEmbeddings_.vocabSize ||
                cachedHiddenSize != quantizedEmbeddings_.hiddenSize) {
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
                        std::fprintf(stderr, "CUDA upload failed: %s\n", e.what());
                        embedUploaded = false;
                    }
                }
            }

            if (embedUploaded) {
                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    try {
                        cuda::computeLMHead(d_embeddings, cachedHiddenSize, cachedVocabSize,
                                            hPtr, logitRow);
                    } catch (const std::exception &e) {
                        std::fprintf(stderr, "CUDA LM head failed: %s\n", e.what());
                        if (!dequantizedEmbeddings_.empty()) {
                            LMHead::computeCPU(hPtr, dequantizedEmbeddings_.data.data(),
                                               dequantizedEmbeddings_.vocabSize,
                                               dequantizedEmbeddings_.hiddenSize, logitRow);
                        } else {
                            LMHead::computeCPUQuantized(hPtr, quantizedEmbeddings_.data.data(),
                                                        quantizedEmbeddings_.type,
                                                        quantizedEmbeddings_.vocabSize,
                                                        quantizedEmbeddings_.hiddenSize, logitRow);
                        }
                    }
                }
            } else {
                // Consolidated single parallelFor over all (token, vocab) pairs
                ThreadPool::instance().parallelFor(0, seqLen * vocabSize, [&](uint32_t flatIdx) {
                    uint32_t s = flatIdx / vocabSize;
                    uint32_t i = flatIdx % vocabSize;
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    float dot = 0.0f;
                    if (!dequantizedEmbeddings_.empty()) {
                        const float *embRow = dequantizedEmbeddings_.data.data() + static_cast<uint64_t>(i) * hiddenSize;
                        dot = dotProductFMA(hPtr, embRow, hiddenSize);
                    } else {
                        uint32_t blockSize = ggmlBlockSize(quantizedEmbeddings_.type);
                        uint32_t typeSize = ggmlTypeSize(quantizedEmbeddings_.type);
                        uint32_t numBlocks = (hiddenSize + blockSize - 1) / blockSize;
                        for (uint32_t b = 0; b < numBlocks; ++b) {
                            uint64_t blockOffset = static_cast<uint64_t>(i) * numBlocks + b;
                            const uint8_t *blockData = quantizedEmbeddings_.data.data() + blockOffset * typeSize;
                            float blockOut[256];
                            GGMLDequantize::dequantizeBlock(quantizedEmbeddings_.type, blockData, blockOut, blockSize);
                            uint32_t start = b * blockSize;
                            uint32_t n = std::min(blockSize, hiddenSize - start);
                            dot += dotProductFMA(hPtr + start, blockOut, n);
                        }
                    }
                    logitRow[i] = dot;
                });
            }
#else
            // CPU path with pre-dequantized embeddings: consolidated single parallelFor
            if (!dequantizedEmbeddings_.empty()) {
                ThreadPool::instance().parallelFor(0, seqLen * vocabSize, [&](uint32_t flatIdx) {
                    uint32_t s = flatIdx / vocabSize;
                    uint32_t i = flatIdx % vocabSize;
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    const float *embRow = dequantizedEmbeddings_.data.data() + static_cast<uint64_t>(i) * hiddenSize;
                    logitRow[i] = dotProductFMA(hPtr, embRow, hiddenSize);
                });
            } else {
                ThreadPool::instance().parallelFor(0, seqLen * vocabSize, [&](uint32_t flatIdx) {
                    uint32_t s = flatIdx / vocabSize;
                    uint32_t i = flatIdx % vocabSize;
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    float dot = 0.0f;
                    uint32_t blockSize = ggmlBlockSize(quantizedEmbeddings_.type);
                    uint32_t typeSize = ggmlTypeSize(quantizedEmbeddings_.type);
                    uint32_t numBlocks = (hiddenSize + blockSize - 1) / blockSize;
                    for (uint32_t b = 0; b < numBlocks; ++b) {
                        uint64_t blockOffset = static_cast<uint64_t>(i) * numBlocks + b;
                        const uint8_t *blockData = quantizedEmbeddings_.data.data() + blockOffset * typeSize;
                        float blockOut[256];
                        GGMLDequantize::dequantizeBlock(quantizedEmbeddings_.type, blockData, blockOut, blockSize);
                        uint32_t start = b * blockSize;
                        uint32_t n = std::min(blockSize, hiddenSize - start);
                        dot += dotProductFMA(hPtr + start, blockOut, n);
                    }
                    logitRow[i] = dot;
                });
            }
#endif
        } else {
            // Separate LM head: consolidated single parallelFor over all (token, vocab) pairs
            ThreadPool::instance().parallelFor(0, seqLen * vocabSize, [&](uint32_t flatIdx) {
                uint32_t s = flatIdx / vocabSize;
                uint32_t i = flatIdx % vocabSize;
                const float *hPtr = hiddenData + s * hiddenSize;
                float *logitRow = logitsData + s * vocabSize;
                float dot = 0.0f;
                uint32_t blockSize = ggmlBlockSize(lmHead_.type);
                uint32_t typeSize = ggmlTypeSize(lmHead_.type);
                uint32_t numBlocks = (hiddenSize + blockSize - 1) / blockSize;
                for (uint32_t b = 0; b < numBlocks; ++b) {
                    uint64_t blockOffset = static_cast<uint64_t>(i) * numBlocks + b;
                    const uint8_t *blockData = lmHead_.data.data() + blockOffset * typeSize;
                    float blockOut[256];
                    GGMLDequantize::dequantizeBlock(lmHead_.type, blockData, blockOut, blockSize);
                    uint32_t start = b * blockSize;
                    uint32_t n = std::min(blockSize, hiddenSize - start);
                    dot += dotProductFMA(hPtr + start, blockOut, n);
                }
                logitRow[i] = dot;
            });
        }

        // Apply final logit softcapping (Gemma4 architecture)
        if (config_.architecture == ARCH_GEMMA4 && config_.finalLogitSoftcapping > 0.0f) {
            float cap = config_.finalLogitSoftcapping;
            float invCap = 1.0f / cap;
            for (uint32_t s = 0; s < seqLen; ++s) {
                float *logitRow = logitsData + s * vocabSize;
                for (uint32_t i = 0; i < vocabSize; ++i) {
                    logitRow[i] = std::tanh(logitRow[i] * invCap) * cap;
                }
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
                std::memcpy(hRow, embRow.data(), hiddenSize * sizeof(float));
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

            // Check if this is an SSM layer for Qwen35MoE architecture
            bool isSSMLayer = (config_.architecture == ARCH_QWEN35MOE &&
                               config_.fullAttentionInterval > 0 &&
                               (layer % config_.fullAttentionInterval) != 0 &&
                               !w.ssmOut.empty());

            if (isSSMLayer) {
                // ---- SSM (Mamba-style) block replaces attention ----

                // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                const float *rmsNormAttnData = w.rmsNormAttn.data();
                for (uint32_t s = 0; s < seqLen; ++s) {
                    rmsNormSIMD(hiddenData + s * hiddenSize, attnNormData + s * hiddenSize,
                                rmsNormAttnData, hiddenSize);
                }

                // SSM computation for each token
                uint32_t ssmInnerSize = config_.ssmInnerSize;
                uint32_t ssmStateSize = config_.ssmStateSize;
                uint32_t ssmConvKernel = config_.ssmConvKernel;

                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *hRowPtr = attnNormData + s * hiddenSize;

                    // Step 1: Input projection (hiddenSize → ssmInnerSize)
                    np::Array<float> ssmIn = w.ssmOut.matMulVec(hRowPtr);
                    float *ssmInData = ssmIn.data();

                    // Step 2: Conv1d with past buffer
                    std::vector<float> convOut(ssmInnerSize);
                    if (ssmConvKernel > 1 && !w.ssmConv1d.empty()) {
                        std::vector<float> convInput(ssmConvKernel * ssmInnerSize);
                        auto &convBuf = kvCache_.ssmConvBuf[layer];
                        uint32_t bufLen = ssmConvKernel - 1;
                        std::memcpy(convInput.data(), convBuf.data(), bufLen * ssmInnerSize * sizeof(float));
                        std::memcpy(convInput.data() + bufLen * ssmInnerSize,
                                    ssmInData, ssmInnerSize * sizeof(float));

                        std::memmove(convBuf.data(), convBuf.data() + ssmInnerSize,
                                     (bufLen - 1) * ssmInnerSize * sizeof(float));
                        std::memcpy(convBuf.data() + (bufLen - 1) * ssmInnerSize,
                                    ssmInData, ssmInnerSize * sizeof(float));

                        for (uint32_t c = 0; c < ssmInnerSize; ++c) {
                            const float *wRow = reinterpret_cast<const float *>(w.ssmConv1d.data.data()) + static_cast<size_t>(c) * ssmConvKernel;
                            convOut[c] = dotProductFMA(wRow, convInput.data() + c * ssmConvKernel, ssmConvKernel);
                        }
                    } else {
                        std::memcpy(convOut.data(), ssmInData, ssmInnerSize * sizeof(float));
                    }

                    // Step 3: SiLU activation on conv output
                    siluSIMD(convOut.data(), ssmInnerSize);

                    // Step 4: SSM state update
                    auto &ssmState = kvCache_.ssmState[layer];
                    for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                        float dt = std::log(1.0f + std::exp(w.ssmDtBias.data()[i]));
                        for (uint32_t j = 0; j < ssmStateSize; ++j) {
                            float aVal = w.ssmA.data()[i * ssmStateSize + j];
                            float aBar = std::exp(aVal * dt);
                            uint32_t idx = i * ssmStateSize + j;
                            ssmState[idx] = aBar * ssmState[idx] + convOut[i];
                        }
                    }

                    // Step 5: Output from SSM state
                    std::vector<float> ssmOut(ssmInnerSize);
                    for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                        double hVal = 0.0;
                        for (uint32_t j = 0; j < ssmStateSize; ++j) {
                            hVal += ssmState[i * ssmStateSize + j];
                        }
                        float gate = w.ssmAlpha.data()[i] * static_cast<float>(hVal) + w.ssmBeta.data()[i];
                        float gateAct = gate / (1.0f + std::exp(-gate));
                        ssmOut[i] = convOut[i] * gateAct;
                    }

                    // Step 6: Output projection back to hiddenSize using attnO
                    np::Array<float> projRow = deqMatMulVecF16(w.attnO_deq_f16.data(), ssmOut.data(),
                                                               w.attnO.rows, w.attnO.cols);
                    float *projPtr = attnProjData + s * hiddenSize;
                    std::memcpy(projPtr, projRow.data(), hiddenSize * sizeof(float));
                }

                // SSM residual (standard residual, no post-norm)
                for (uint32_t s = 0; s < seqLen; ++s) {
                    addSIMD(hiddenData + s * hiddenSize, attnProjData + s * hiddenSize, hiddenSize);
                }
            } else {
                // ---- Attention block ----

                // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                const float *rmsNormAttnData = w.rmsNormAttn.data();
                for (uint32_t s = 0; s < seqLen; ++s) {
                    rmsNormSIMD(hiddenData + s * hiddenSize, attnNormData + s * hiddenSize,
                                rmsNormAttnData, hiddenSize);
                }

                // Q, K, V projections (architecture-aware bias)
                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *hRowPtr = attnNormData + s * hiddenSize;

                    // Q
                    np::Array<float> qRow = w.attnQ.matMulVec(hRowPtr);
                    float *qRowPtr = qData + s * nHeads * headDim;
                    std::memcpy(qRowPtr, qRow.data(), nHeads * headDim * sizeof(float));
                    if (config_.architecture == ARCH_QWEN2 && !w.attnQBias.empty()) {
                        addSIMD(qRowPtr, w.attnQBias.data(), nHeads * headDim);
                    }

                    // K
                    np::Array<float> kRow = w.attnK.matMulVec(hRowPtr);
                    float *kRowPtr = kData + s * nKVHeads * headDim;
                    std::memcpy(kRowPtr, kRow.data(), nKVHeads * headDim * sizeof(float));
                    if (config_.architecture == ARCH_QWEN2 && !w.attnKBias.empty()) {
                        addSIMD(kRowPtr, w.attnKBias.data(), nKVHeads * headDim);
                    }

                    // V
                    np::Array<float> vRow = w.attnV.matMulVec(hRowPtr);
                    float *vRowPtr = vData + s * nKVHeads * headDim;
                    std::memcpy(vRowPtr, vRow.data(), nKVHeads * headDim * sizeof(float));
                    if (config_.architecture == ARCH_QWEN2 && !w.attnVBias.empty()) {
                        addSIMD(vRowPtr, w.attnVBias.data(), nKVHeads * headDim);
                    }
                }

                // Apply Q/K norms before RoPE (Gemma4 and Qwen35MoE)
                if (config_.architecture == ARCH_GEMMA4 && !w.attnQNorm.empty() && !w.attnKNorm.empty()) {
                    applyQKNorms(qData, kData, seqLen, nHeads, nKVHeads,
                                 w.attnQNorm.data(), w.attnKNorm.data());
                } else if (config_.architecture == ARCH_QWEN35MOE && !w.attnQNormMoe.empty() && !w.attnKNormMoe.empty()) {
                    applyQKNorms(qData, kData, seqLen, nHeads, nKVHeads,
                                 w.attnQNormMoe.data(), w.attnKNormMoe.data());
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
                    std::memcpy(kDst, kSrc, kvSize * sizeof(float));
                    std::memcpy(vDst, vSrc, kvSize * sizeof(float));
                }

                // Attention
                uint32_t totalCacheLen = cachePos + seqLen;
                attentionFused(qData, kCacheLayer, vCacheLayer, attnOutData, seqLen,
                               cachePos, totalCacheLen, layer);

                // Output projection (using dequantized F32 weights for exact float dot product)
                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *attnRowPtr = attnOutData + s * nHeads * headDim;
                    np::Array<float> projRow = deqMatMulVecF16(w.attnO_deq_f16.data(), attnRowPtr,
                                                               w.attnO.rows, w.attnO.cols);
                    float *projPtr = attnProjData + s * hiddenSize;
                    std::memcpy(projPtr, projRow.data(), hiddenSize * sizeof(float));
                }

                // Attention residual + post-attention processing (architecture-aware)
                if (config_.architecture == ARCH_GEMMA4 && !w.postAttnNorm.empty()) {
                    // Gemma4: post-attention norm + layer scale
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        float *hPtr = hiddenData + s * hiddenSize;
                        const float *aPtr = attnProjData + s * hiddenSize;
                        addSIMD(hPtr, aPtr, hiddenSize);
                        rmsNormSIMD(hPtr, hPtr, w.postAttnNorm.data(), hiddenSize);
                        if (!w.layerOutputScale.empty()) {
                            scaleSIMD(hPtr, w.layerOutputScale.data()[0], hiddenSize);
                        }
                    }
                } else {
                    // Qwen2, Qwen35MoE: standard residual (no post-norm)
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        addSIMD(hiddenData + s * hiddenSize, attnProjData + s * hiddenSize, hiddenSize);
                    }
                }
            }

            // ---- FFN block ----

            // RMSNorm: ffnNorm = rmsNorm(hidden, w.rmsNormFFN)
            const float *rmsNormFFNData = w.rmsNormFFN.data();
            for (uint32_t s = 0; s < seqLen; ++s) {
                rmsNormSIMD(hiddenData + s * hiddenSize, ffnNormData + s * hiddenSize,
                            rmsNormFFNData, hiddenSize);
            }

            // MoE path (expertCount > 0)
            if (config_.architecture == ARCH_GEMMA4 && config_.expertCount > 0) {
                computeGemma4MoE(ffnNormData, ffnOutData, seqLen, hiddenSize, intermediateSize, w);
                // Residual + layer scale
                for (uint32_t s = 0; s < seqLen; ++s) {
                    float *hPtr = hiddenData + s * hiddenSize;
                    const float *fPtr = ffnOutData + s * hiddenSize;
                    addSIMD(hPtr, fPtr, hiddenSize);
                    if (!w.layerOutputScale.empty()) {
                        scaleSIMD(hPtr, w.layerOutputScale.data()[0], hiddenSize);
                    }
                }
            } else if (config_.architecture == ARCH_QWEN35MOE && config_.expertCount > 0) {
                // Qwen35MoE MoE path
                computeQwen35MoE(ffnNormData, ffnOutData, seqLen, hiddenSize, intermediateSize, w);
                // Standard residual (no post-norm)
                for (uint32_t s = 0; s < seqLen; ++s) {
                    addSIMD(hiddenData + s * hiddenSize, ffnOutData + s * hiddenSize, hiddenSize);
                }
            } else {
                // Dense FFN path

                // FFN gate+up projections (architecture-aware activation)
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

                // FFN activation (architecture-aware)
                if (config_.architecture == ARCH_GEMMA4) {
                    // Gemma4: GeGLU activation (gelu(gate) * up)
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        float *gatePtr = gateData + s * intermediateSize;
                        const float *upPtr = upData + s * intermediateSize;
                        geluInPlace(gatePtr, intermediateSize);
                        for (uint32_t i = 0; i < intermediateSize; ++i) {
                            gatePtr[i] *= upPtr[i];
                        }
                    }
                } else {
                    // Qwen2, Qwen35MoE: SwiGLU activation (silu(gate) * up)
                    swiGLUSIMD(gateData, upData, seqLen * intermediateSize);
                }

                // Down projection (using dequantized F32 weights for exact float dot product)
                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *ffnActPtr = gateData + s * intermediateSize;
                    np::Array<float> downRow = deqMatMulVecF16(w.ffnDown_deq_f16.data(), ffnActPtr,
                                                               w.ffnDown.rows, w.ffnDown.cols);
                    float *outPtr = ffnOutData + s * hiddenSize;
                    std::memcpy(outPtr, downRow.data(), hiddenSize * sizeof(float));
                }

                // FFN residual + post-FFN processing (architecture-aware)
                if (config_.architecture == ARCH_GEMMA4 && !w.postFFWNorm.empty()) {
                    // Gemma4: post-FFN norm + layer scale
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        float *hPtr = hiddenData + s * hiddenSize;
                        const float *fPtr = ffnOutData + s * hiddenSize;
                        addSIMD(hPtr, fPtr, hiddenSize);
                        rmsNormSIMD(hPtr, hPtr, w.postFFWNorm.data(), hiddenSize);
                        if (!w.layerOutputScale.empty()) {
                            scaleSIMD(hPtr, w.layerOutputScale.data()[0], hiddenSize);
                        }
                    }
                } else {
                    // Qwen2, Qwen35MoE: standard residual (no post-norm)
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        addSIMD(hiddenData + s * hiddenSize, ffnOutData + s * hiddenSize, hiddenSize);
                    }
                }
            }
        }

        // Final RMSNorm
        const float *finalNormData = finalNorm_.data();
        for (uint32_t s = 0; s < seqLen; ++s) {
            rmsNormSIMD(hiddenData + s * hiddenSize, hiddenData + s * hiddenSize,
                        finalNormData, hiddenSize);
        }

        // Save hidden state (after final RMSNorm, before LM head)
        std::vector<float> hiddenState(hiddenData, hiddenData + seqLen * hiddenSize);

        // LM head (logits)
        np::Array<float> logits = np::Array<float>(np::Shape{seqLen, vocabSize});
        float *logitsData = logits.data();

        if (lmHeadTied_) {
            if (!dequantizedEmbeddings_.empty()) {
                // Consolidated single parallelFor over all (token, vocab) pairs
                ThreadPool::instance().parallelFor(0, seqLen * vocabSize, [&](uint32_t flatIdx) {
                    uint32_t s = flatIdx / vocabSize;
                    uint32_t i = flatIdx % vocabSize;
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    const float *embRow = dequantizedEmbeddings_.data.data() + static_cast<uint64_t>(i) * hiddenSize;
                    logitRow[i] = dotProductFMA(hPtr, embRow, hiddenSize);
                });
            } else {
                ThreadPool::instance().parallelFor(0, seqLen * vocabSize, [&](uint32_t flatIdx) {
                    uint32_t s = flatIdx / vocabSize;
                    uint32_t i = flatIdx % vocabSize;
                    const float *hPtr = hiddenData + s * hiddenSize;
                    float *logitRow = logitsData + s * vocabSize;
                    float dot = 0.0f;
                    uint32_t blockSize = ggmlBlockSize(quantizedEmbeddings_.type);
                    uint32_t typeSize = ggmlTypeSize(quantizedEmbeddings_.type);
                    uint32_t numBlocks = (hiddenSize + blockSize - 1) / blockSize;
                    for (uint32_t b = 0; b < numBlocks; ++b) {
                        uint64_t blockOffset = static_cast<uint64_t>(i) * numBlocks + b;
                        const uint8_t *blockData = quantizedEmbeddings_.data.data() + blockOffset * typeSize;
                        float blockOut[256];
                        GGMLDequantize::dequantizeBlock(quantizedEmbeddings_.type, blockData, blockOut, blockSize);
                        uint32_t start = b * blockSize;
                        uint32_t n = std::min(blockSize, hiddenSize - start);
                        dot += dotProductFMA(hPtr + start, blockOut, n);
                    }
                    logitRow[i] = dot;
                });
            }
        } else {
            // Separate LM head: consolidated single parallelFor
            ThreadPool::instance().parallelFor(0, seqLen * vocabSize, [&](uint32_t flatIdx) {
                uint32_t s = flatIdx / vocabSize;
                uint32_t i = flatIdx % vocabSize;
                const float *hPtr = hiddenData + s * hiddenSize;
                float *logitRow = logitsData + s * vocabSize;
                float dot = 0.0f;
                uint32_t blockSize = ggmlBlockSize(lmHead_.type);
                uint32_t typeSize = ggmlTypeSize(lmHead_.type);
                uint32_t numBlocks = (hiddenSize + blockSize - 1) / blockSize;
                for (uint32_t b = 0; b < numBlocks; ++b) {
                    uint64_t blockOffset = static_cast<uint64_t>(i) * numBlocks + b;
                    const uint8_t *blockData = lmHead_.data.data() + blockOffset * typeSize;
                    float blockOut[256];
                    GGMLDequantize::dequantizeBlock(lmHead_.type, blockData, blockOut, blockSize);
                    uint32_t start = b * blockSize;
                    uint32_t n = std::min(blockSize, hiddenSize - start);
                    dot += dotProductFMA(hPtr + start, blockOut, n);
                }
                logitRow[i] = dot;
            });
        }

        // Apply final logit softcapping (Gemma4 architecture)
        if (config_.architecture == ARCH_GEMMA4 && config_.finalLogitSoftcapping > 0.0f) {
            float cap = config_.finalLogitSoftcapping;
            float invCap = 1.0f / cap;
            for (uint32_t s = 0; s < seqLen; ++s) {
                float *logitRow = logitsData + s * vocabSize;
                for (uint32_t i = 0; i < vocabSize; ++i) {
                    logitRow[i] = std::tanh(logitRow[i] * invCap) * cap;
                }
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
                std::memcpy(hRow, embRow.data(), hiddenSize * sizeof(float));
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

            // Check if this is an SSM layer for Qwen35MoE architecture
            bool isSSMLayer = (config_.architecture == ARCH_QWEN35MOE &&
                               config_.fullAttentionInterval > 0 &&
                               (layer % config_.fullAttentionInterval) != 0 &&
                               !w.ssmOut.empty());

            if (isSSMLayer) {
                // ---- SSM (Mamba-style) block replaces attention ----

                // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                const float *rmsNormAttnData = w.rmsNormAttn.data();
                for (uint32_t s = 0; s < seqLen; ++s) {
                    rmsNormSIMD(hiddenData + s * hiddenSize, attnNormData + s * hiddenSize,
                                rmsNormAttnData, hiddenSize);
                }

                // SSM computation for each token
                uint32_t ssmInnerSize = config_.ssmInnerSize;
                uint32_t ssmStateSize = config_.ssmStateSize;
                uint32_t ssmConvKernel = config_.ssmConvKernel;

                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *hRowPtr = attnNormData + s * hiddenSize;

                    // Step 1: Input projection (hiddenSize → ssmInnerSize)
                    np::Array<float> ssmIn = w.ssmOut.matMulVec(hRowPtr);
                    float *ssmInData = ssmIn.data();

                    // Step 2: Conv1d with past buffer
                    std::vector<float> convOut(ssmInnerSize);
                    if (ssmConvKernel > 1 && !w.ssmConv1d.empty()) {
                        std::vector<float> convInput(ssmConvKernel * ssmInnerSize);
                        auto &convBuf = kvCache_.ssmConvBuf[layer];
                        uint32_t bufLen = ssmConvKernel - 1;
                        std::memcpy(convInput.data(), convBuf.data(), bufLen * ssmInnerSize * sizeof(float));
                        std::memcpy(convInput.data() + bufLen * ssmInnerSize,
                                    ssmInData, ssmInnerSize * sizeof(float));

                        for (uint32_t i = 0; i < (bufLen - 1) * ssmInnerSize; ++i) {
                            convBuf[i] = convBuf[i + ssmInnerSize];
                        }
                        std::memcpy(convBuf.data() + (bufLen - 1) * ssmInnerSize,
                                    ssmInData, ssmInnerSize * sizeof(float));

                        for (uint32_t c = 0; c < ssmInnerSize; ++c) {
                            double dot = 0.0;
                            const float *wRow = reinterpret_cast<const float *>(w.ssmConv1d.data.data()) + static_cast<size_t>(c) * ssmConvKernel;
                            for (uint32_t k = 0; k < ssmConvKernel; ++k) {
                                dot += static_cast<double>(wRow[k]) * convInput[c * ssmConvKernel + k];
                            }
                            convOut[c] = static_cast<float>(dot);
                        }
                    } else {
                        std::memcpy(convOut.data(), ssmInData, ssmInnerSize * sizeof(float));
                    }

                    // Step 3: SiLU activation on conv output
                    siluSIMD(convOut.data(), ssmInnerSize);

                    // Step 4: SSM state update
                    auto &ssmState = kvCache_.ssmState[layer];
                    for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                        float dt = std::log(1.0f + std::exp(w.ssmDtBias.data()[i]));
                        for (uint32_t j = 0; j < ssmStateSize; ++j) {
                            float aVal = w.ssmA.data()[i * ssmStateSize + j];
                            float aBar = std::exp(aVal * dt);
                            uint32_t idx = i * ssmStateSize + j;
                            ssmState[idx] = aBar * ssmState[idx] + convOut[i];
                        }
                    }

                    // Step 5: Output from SSM state
                    std::vector<float> ssmOut(ssmInnerSize);
                    for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                        double hVal = 0.0;
                        for (uint32_t j = 0; j < ssmStateSize; ++j) {
                            hVal += ssmState[i * ssmStateSize + j];
                        }
                        float gate = w.ssmAlpha.data()[i] * static_cast<float>(hVal) + w.ssmBeta.data()[i];
                        float gateAct = gate / (1.0f + std::exp(-gate));
                        ssmOut[i] = convOut[i] * gateAct;
                    }

                    // Step 6: Output projection back to hiddenSize using attnO
                    np::Array<float> projRow = deqMatMulVecF16(w.attnO_deq_f16.data(), ssmOut.data(),
                                                               w.attnO.rows, w.attnO.cols);
                    float *projPtr = attnProjData + s * hiddenSize;
                    std::memcpy(projPtr, projRow.data(), hiddenSize * sizeof(float));
                }

                // SSM residual (standard residual, no post-norm)
                for (uint32_t s = 0; s < seqLen; ++s) {
                    addSIMD(hiddenData + s * hiddenSize, attnProjData + s * hiddenSize, hiddenSize);
                }
            } else {
                // ---- Attention block ----

                // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                const float *rmsNormAttnData = w.rmsNormAttn.data();
                for (uint32_t s = 0; s < seqLen; ++s) {
                    rmsNormSIMD(hiddenData + s * hiddenSize, attnNormData + s * hiddenSize,
                                rmsNormAttnData, hiddenSize);
                }


                // Q, K, V projections
                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *hRowPtr = attnNormData + s * hiddenSize;

                    // Q
                    np::Array<float> qRow = w.attnQ.matMulVec(hRowPtr);
                    float *qRowPtr = qData + s * nHeads * headDim;
                    std::memcpy(qRowPtr, qRow.data(), nHeads * headDim * sizeof(float));
                    if (config_.architecture == ARCH_QWEN2 && !w.attnQBias.empty()) {
                        addSIMD(qRowPtr, w.attnQBias.data(), nHeads * headDim);
                    }

                    // K
                    np::Array<float> kRow = w.attnK.matMulVec(hRowPtr);
                    float *kRowPtr = kData + s * nKVHeads * headDim;
                    std::memcpy(kRowPtr, kRow.data(), nKVHeads * headDim * sizeof(float));
                    if (config_.architecture == ARCH_QWEN2 && !w.attnKBias.empty()) {
                        addSIMD(kRowPtr, w.attnKBias.data(), nKVHeads * headDim);
                    }

                    // V
                    np::Array<float> vRow = w.attnV.matMulVec(hRowPtr);
                    float *vRowPtr = vData + s * nKVHeads * headDim;
                    std::memcpy(vRowPtr, vRow.data(), nKVHeads * headDim * sizeof(float));
                    if (config_.architecture == ARCH_QWEN2 && !w.attnVBias.empty()) {
                        addSIMD(vRowPtr, w.attnVBias.data(), nKVHeads * headDim);
                    }
                }


                // Apply Q/K norms before RoPE (Gemma4 and Qwen35MoE)
                if (config_.architecture == ARCH_GEMMA4 && !w.attnQNorm.empty() && !w.attnKNorm.empty()) {
                    applyQKNorms(qData, kData, seqLen, nHeads, nKVHeads,
                                 w.attnQNorm.data(), w.attnKNorm.data());
                } else if (config_.architecture == ARCH_QWEN35MOE && !w.attnQNormMoe.empty() && !w.attnKNormMoe.empty()) {
                    applyQKNorms(qData, kData, seqLen, nHeads, nKVHeads,
                                 w.attnQNormMoe.data(), w.attnKNormMoe.data());
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
                    std::memcpy(kDst, kSrc, kvSize * sizeof(float));
                    std::memcpy(vDst, vSrc, kvSize * sizeof(float));
                }

                // Attention
                uint32_t totalCacheLen = cachePos + seqLen;
                attentionFused(qData, kCacheLayer, vCacheLayer, attnOutData, seqLen,
                               cachePos, totalCacheLen, layer);


                // Output projection (using dequantized F32 weights for exact float dot product)
                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *attnRowPtr = attnOutData + s * nHeads * headDim;
                    np::Array<float> projRow = deqMatMulVecF16(w.attnO_deq_f16.data(), attnRowPtr,
                                                               w.attnO.rows, w.attnO.cols);
                    float *projPtr = attnProjData + s * hiddenSize;
                    std::memcpy(projPtr, projRow.data(), hiddenSize * sizeof(float));
                }


                // Attention residual + post-attention processing (architecture-aware)
                if (config_.architecture == ARCH_GEMMA4 && !w.postAttnNorm.empty()) {
                    // Gemma4: post-attention norm + layer scale
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        float *hPtr = hiddenData + s * hiddenSize;
                        const float *aPtr = attnProjData + s * hiddenSize;
                        addSIMD(hPtr, aPtr, hiddenSize);
                        rmsNormSIMD(hPtr, hPtr, w.postAttnNorm.data(), hiddenSize);
                        if (!w.layerOutputScale.empty()) {
                            scaleSIMD(hPtr, w.layerOutputScale.data()[0], hiddenSize);
                        }
                    }
                } else {
                    // Qwen2, Qwen35MoE: standard residual (no post-norm)
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        addSIMD(hiddenData + s * hiddenSize, attnProjData + s * hiddenSize, hiddenSize);
                    }
                }
            }

            // ---- FFN block ----

            // RMSNorm: ffnNorm = rmsNorm(hidden, w.rmsNormFFN)
            const float *rmsNormFFNData = w.rmsNormFFN.data();
            for (uint32_t s = 0; s < seqLen; ++s) {
                rmsNormSIMD(hiddenData + s * hiddenSize, ffnNormData + s * hiddenSize,
                            rmsNormFFNData, hiddenSize);
            }


            // MoE path (expertCount > 0)
            if (config_.architecture == ARCH_GEMMA4 && config_.expertCount > 0) {
                computeGemma4MoE(ffnNormData, ffnOutData, seqLen, hiddenSize, intermediateSize, w);
                // Residual + layer scale
                for (uint32_t s = 0; s < seqLen; ++s) {
                    float *hPtr = hiddenData + s * hiddenSize;
                    const float *fPtr = ffnOutData + s * hiddenSize;
                    addSIMD(hPtr, fPtr, hiddenSize);
                    if (!w.layerOutputScale.empty()) {
                        scaleSIMD(hPtr, w.layerOutputScale.data()[0], hiddenSize);
                    }
                }
            } else if (config_.architecture == ARCH_QWEN35MOE && config_.expertCount > 0) {
                // Qwen35MoE MoE path
                computeQwen35MoE(ffnNormData, ffnOutData, seqLen, hiddenSize, intermediateSize, w);
                // Standard residual (no post-norm)
                for (uint32_t s = 0; s < seqLen; ++s) {
                    addSIMD(hiddenData + s * hiddenSize, ffnOutData + s * hiddenSize, hiddenSize);
                }
            } else {
                // Dense FFN path

                // FFN gate+up projections (architecture-aware activation)
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


                // FFN activation (architecture-aware)
                if (config_.architecture == ARCH_GEMMA4) {
                    // Gemma4: GeGLU activation (gelu(gate) * up)
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        float *gatePtr = gateData + s * intermediateSize;
                        const float *upPtr = upData + s * intermediateSize;
                        geluInPlace(gatePtr, intermediateSize);
                        for (uint32_t i = 0; i < intermediateSize; ++i) {
                            gatePtr[i] *= upPtr[i];
                        }
                    }
                } else {
                    // Qwen2, Qwen35MoE: SwiGLU activation (silu(gate) * up)
                    swiGLUSIMD(gateData, upData, seqLen * intermediateSize);
                }


                // Down projection (using dequantized F32 weights for exact float dot product)
                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *ffnActPtr = gateData + s * intermediateSize;
                    np::Array<float> downRow = deqMatMulVecF16(w.ffnDown_deq_f16.data(), ffnActPtr,
                                                               w.ffnDown.rows, w.ffnDown.cols);
                    float *outPtr = ffnOutData + s * hiddenSize;
                    std::memcpy(outPtr, downRow.data(), hiddenSize * sizeof(float));
                }


                // FFN residual + post-FFN processing (architecture-aware)
                if (config_.architecture == ARCH_GEMMA4 && !w.postFFWNorm.empty()) {
                    // Gemma4: post-FFN norm + layer scale
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        float *hPtr = hiddenData + s * hiddenSize;
                        const float *fPtr = ffnOutData + s * hiddenSize;
                        addSIMD(hPtr, fPtr, hiddenSize);
                        rmsNormSIMD(hPtr, hPtr, w.postFFWNorm.data(), hiddenSize);
                        if (!w.layerOutputScale.empty()) {
                            scaleSIMD(hPtr, w.layerOutputScale.data()[0], hiddenSize);
                        }
                    }
                } else {
                    // Qwen2, Qwen35MoE: standard residual (no post-norm)
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        float *hPtr = hiddenData + s * hiddenSize;
                        const float *fPtr = ffnOutData + s * hiddenSize;
                        for (uint32_t i = 0; i < hiddenSize; ++i) {
                            hPtr[i] += fPtr[i];
                        }
                    }
                }
            }
        }

        // Final RMSNorm
        const float *finalNormData = finalNorm_.data();
        for (uint32_t s = 0; s < seqLen; ++s) {
            rmsNormSIMD(hiddenData + s * hiddenSize, hiddenData + s * hiddenSize,
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
                std::memcpy(hRow, embRow.data(), hiddenSize * sizeof(float));
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

            // Check if this is an SSM layer for Qwen35MoE architecture
            bool isSSMLayer = (config_.architecture == ARCH_QWEN35MOE &&
                               config_.fullAttentionInterval > 0 &&
                               (layer % config_.fullAttentionInterval) != 0 &&
                               !w.ssmOut.empty());

            if (isSSMLayer) {
                // ---- SSM (Mamba-style) block replaces attention ----

                // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                const float *rmsNormAttnData = w.rmsNormAttn.data();
                for (uint32_t s = 0; s < seqLen; ++s) {
                    rmsNormSIMD(hiddenData + s * hiddenSize, attnNormData + s * hiddenSize,
                                rmsNormAttnData, hiddenSize);
                }

                // SSM computation for each token
                uint32_t ssmInnerSize = config_.ssmInnerSize;
                uint32_t ssmStateSize = config_.ssmStateSize;
                uint32_t ssmConvKernel = config_.ssmConvKernel;

                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *hRowPtr = attnNormData + s * hiddenSize;

                    // Step 1: Input projection (hiddenSize → ssmInnerSize)
                    np::Array<float> ssmIn = w.ssmOut.matMulVec(hRowPtr);
                    float *ssmInData = ssmIn.data();

                    // Step 2: Conv1d with past buffer
                    std::vector<float> convOut(ssmInnerSize);
                    if (ssmConvKernel > 1 && !w.ssmConv1d.empty()) {
                        std::vector<float> convInput(ssmConvKernel * ssmInnerSize);
                        auto &convBuf = kvCache_.ssmConvBuf[layer];
                        uint32_t bufLen = ssmConvKernel - 1;
                        std::memcpy(convInput.data(), convBuf.data(), bufLen * ssmInnerSize * sizeof(float));
                        std::memcpy(convInput.data() + bufLen * ssmInnerSize,
                                    ssmInData, ssmInnerSize * sizeof(float));

                        for (uint32_t i = 0; i < (bufLen - 1) * ssmInnerSize; ++i) {
                            convBuf[i] = convBuf[i + ssmInnerSize];
                        }
                        std::memcpy(convBuf.data() + (bufLen - 1) * ssmInnerSize,
                                    ssmInData, ssmInnerSize * sizeof(float));

                        for (uint32_t c = 0; c < ssmInnerSize; ++c) {
                            double dot = 0.0;
                            const float *wRow = reinterpret_cast<const float *>(w.ssmConv1d.data.data()) + static_cast<size_t>(c) * ssmConvKernel;
                            for (uint32_t k = 0; k < ssmConvKernel; ++k) {
                                dot += static_cast<double>(wRow[k]) * convInput[c * ssmConvKernel + k];
                            }
                            convOut[c] = static_cast<float>(dot);
                        }
                    } else {
                        std::memcpy(convOut.data(), ssmInData, ssmInnerSize * sizeof(float));
                    }

                    // Step 3: SiLU activation on conv output
                    siluSIMD(convOut.data(), ssmInnerSize);

                    // Step 4: SSM state update
                    auto &ssmState = kvCache_.ssmState[layer];
                    for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                        float dt = std::log(1.0f + std::exp(w.ssmDtBias.data()[i]));
                        for (uint32_t j = 0; j < ssmStateSize; ++j) {
                            float aVal = w.ssmA.data()[i * ssmStateSize + j];
                            float aBar = std::exp(aVal * dt);
                            uint32_t idx = i * ssmStateSize + j;
                            ssmState[idx] = aBar * ssmState[idx] + convOut[i];
                        }
                    }

                    // Step 5: Output from SSM state
                    std::vector<float> ssmOut(ssmInnerSize);
                    for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                        double hVal = 0.0;
                        for (uint32_t j = 0; j < ssmStateSize; ++j) {
                            hVal += ssmState[i * ssmStateSize + j];
                        }
                        float gate = w.ssmAlpha.data()[i] * static_cast<float>(hVal) + w.ssmBeta.data()[i];
                        float gateAct = gate / (1.0f + std::exp(-gate));
                        ssmOut[i] = convOut[i] * gateAct;
                    }

                    // Step 6: Output projection back to hiddenSize using attnO
                    np::Array<float> projRow = deqMatMulVecF16(w.attnO_deq_f16.data(), ssmOut.data(),
                                                               w.attnO.rows, w.attnO.cols);
                    float *projPtr = attnProjData + s * hiddenSize;
                    std::memcpy(projPtr, projRow.data(), hiddenSize * sizeof(float));
                }

                // SSM residual (standard residual, no post-norm)
                for (uint32_t s = 0; s < seqLen; ++s) {
                    addSIMD(hiddenData + s * hiddenSize, attnProjData + s * hiddenSize, hiddenSize);
                }
            } else {
                // ---- Attention block ----

                // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                const float *rmsNormAttnData = w.rmsNormAttn.data();
                for (uint32_t s = 0; s < seqLen; ++s) {
                    rmsNormSIMD(hiddenData + s * hiddenSize, attnNormData + s * hiddenSize,
                                rmsNormAttnData, hiddenSize);
                }


                // Q, K, V projections
                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *hRowPtr = attnNormData + s * hiddenSize;

                    // Q
                    np::Array<float> qRow = w.attnQ.matMulVec(hRowPtr);
                    float *qRowPtr = qData + s * nHeads * headDim;
                    std::memcpy(qRowPtr, qRow.data(), nHeads * headDim * sizeof(float));
                    if (config_.architecture == ARCH_QWEN2 && !w.attnQBias.empty()) {
                        addSIMD(qRowPtr, w.attnQBias.data(), nHeads * headDim);
                    }

                    // K
                    np::Array<float> kRow = w.attnK.matMulVec(hRowPtr);
                    float *kRowPtr = kData + s * nKVHeads * headDim;
                    std::memcpy(kRowPtr, kRow.data(), nKVHeads * headDim * sizeof(float));
                    if (config_.architecture == ARCH_QWEN2 && !w.attnKBias.empty()) {
                        addSIMD(kRowPtr, w.attnKBias.data(), nKVHeads * headDim);
                    }

                    // V
                    np::Array<float> vRow = w.attnV.matMulVec(hRowPtr);
                    float *vRowPtr = vData + s * nKVHeads * headDim;
                    std::memcpy(vRowPtr, vRow.data(), nKVHeads * headDim * sizeof(float));
                    if (config_.architecture == ARCH_QWEN2 && !w.attnVBias.empty()) {
                        addSIMD(vRowPtr, w.attnVBias.data(), nKVHeads * headDim);
                    }
                }


                // Apply Q/K norms before RoPE (Gemma4 and Qwen35MoE)
                if (config_.architecture == ARCH_GEMMA4 && !w.attnQNorm.empty() && !w.attnKNorm.empty()) {
                    applyQKNorms(qData, kData, seqLen, nHeads, nKVHeads,
                                 w.attnQNorm.data(), w.attnKNorm.data());
                } else if (config_.architecture == ARCH_QWEN35MOE && !w.attnQNormMoe.empty() && !w.attnKNormMoe.empty()) {
                    applyQKNorms(qData, kData, seqLen, nHeads, nKVHeads,
                                 w.attnQNormMoe.data(), w.attnKNormMoe.data());
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
                    std::memcpy(kDst, kSrc, kvSize * sizeof(float));
                    std::memcpy(vDst, vSrc, kvSize * sizeof(float));
                }

                // Attention
                uint32_t totalCacheLen = cachePos + seqLen;
                attentionFused(qData, kCacheLayer, vCacheLayer, attnOutData, seqLen,
                               cachePos, totalCacheLen, layer);


                // Output projection (using dequantized F32 weights for exact float dot product)
                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *attnRowPtr = attnOutData + s * nHeads * headDim;
                    np::Array<float> projRow = deqMatMulVecF16(w.attnO_deq_f16.data(), attnRowPtr,
                                                               w.attnO.rows, w.attnO.cols);
                    float *projPtr = attnProjData + s * hiddenSize;
                    std::memcpy(projPtr, projRow.data(), hiddenSize * sizeof(float));
                }


                // Attention residual + post-attention processing (architecture-aware)
                if (config_.architecture == ARCH_GEMMA4 && !w.postAttnNorm.empty()) {
                    // Gemma4: post-attention norm + layer scale
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        float *hPtr = hiddenData + s * hiddenSize;
                        const float *aPtr = attnProjData + s * hiddenSize;
                        addSIMD(hPtr, aPtr, hiddenSize);
                        rmsNormSIMD(hPtr, hPtr, w.postAttnNorm.data(), hiddenSize);
                        if (!w.layerOutputScale.empty()) {
                            scaleSIMD(hPtr, w.layerOutputScale.data()[0], hiddenSize);
                        }
                    }
                } else {
                    // Qwen2, Qwen35MoE: standard residual (no post-norm)
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        addSIMD(hiddenData + s * hiddenSize, attnProjData + s * hiddenSize, hiddenSize);
                    }
                }
            }

            // ---- FFN block ----

            // RMSNorm: ffnNorm = rmsNorm(hidden, w.rmsNormFFN)
            const float *rmsNormFFNData = w.rmsNormFFN.data();
            for (uint32_t s = 0; s < seqLen; ++s) {
                rmsNormSIMD(hiddenData + s * hiddenSize, ffnNormData + s * hiddenSize,
                            rmsNormFFNData, hiddenSize);
            }


            // MoE path (expertCount > 0)
            if (config_.architecture == ARCH_GEMMA4 && config_.expertCount > 0) {
                computeGemma4MoE(ffnNormData, ffnOutData, seqLen, hiddenSize, intermediateSize, w);
                // Residual + layer scale
                for (uint32_t s = 0; s < seqLen; ++s) {
                    float *hPtr = hiddenData + s * hiddenSize;
                    const float *fPtr = ffnOutData + s * hiddenSize;
                    addSIMD(hPtr, fPtr, hiddenSize);
                    if (!w.layerOutputScale.empty()) {
                        scaleSIMD(hPtr, w.layerOutputScale.data()[0], hiddenSize);
                    }
                }
            } else if (config_.architecture == ARCH_QWEN35MOE && config_.expertCount > 0) {
                // Qwen35MoE MoE path
                computeQwen35MoE(ffnNormData, ffnOutData, seqLen, hiddenSize, intermediateSize, w);
                // Standard residual (no post-norm)
                for (uint32_t s = 0; s < seqLen; ++s) {
                    addSIMD(hiddenData + s * hiddenSize, ffnOutData + s * hiddenSize, hiddenSize);
                }
            } else {
                // Dense FFN path

                // FFN gate+up projections (architecture-aware activation)
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


                // FFN activation (architecture-aware)
                if (config_.architecture == ARCH_GEMMA4) {
                    // Gemma4: GeGLU activation (gelu(gate) * up)
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        float *gatePtr = gateData + s * intermediateSize;
                        const float *upPtr = upData + s * intermediateSize;
                        geluInPlace(gatePtr, intermediateSize);
                        for (uint32_t i = 0; i < intermediateSize; ++i) {
                            gatePtr[i] *= upPtr[i];
                        }
                    }
                } else {
                    // Qwen2, Qwen35MoE: SwiGLU activation (silu(gate) * up)
                    swiGLUSIMD(gateData, upData, seqLen * intermediateSize);
                }


                // Down projection (using dequantized F32 weights for exact float dot product)
                for (uint32_t s = 0; s < seqLen; ++s) {
                    const float *ffnActPtr = gateData + s * intermediateSize;
                    np::Array<float> downRow = deqMatMulVecF16(w.ffnDown_deq_f16.data(), ffnActPtr,
                                                               w.ffnDown.rows, w.ffnDown.cols);
                    float *outPtr = ffnOutData + s * hiddenSize;
                    std::memcpy(outPtr, downRow.data(), hiddenSize * sizeof(float));
                }


                // FFN residual + post-FFN processing (architecture-aware)
                if (config_.architecture == ARCH_GEMMA4 && !w.postFFWNorm.empty()) {
                    // Gemma4: post-FFN norm + layer scale
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        float *hPtr = hiddenData + s * hiddenSize;
                        const float *fPtr = ffnOutData + s * hiddenSize;
                        addSIMD(hPtr, fPtr, hiddenSize);
                        rmsNormSIMD(hPtr, hPtr, w.postFFWNorm.data(), hiddenSize);
                        if (!w.layerOutputScale.empty()) {
                            scaleSIMD(hPtr, w.layerOutputScale.data()[0], hiddenSize);
                        }
                    }
                } else {
                    // Qwen2, Qwen35MoE: standard residual (no post-norm)
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        float *hPtr = hiddenData + s * hiddenSize;
                        const float *fPtr = ffnOutData + s * hiddenSize;
                        for (uint32_t i = 0; i < hiddenSize; ++i) {
                            hPtr[i] += fPtr[i];
                        }
                    }
                }
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


            for (uint32_t layer = 0; layer < nLayers; ++layer) {
                auto &w = layers_[layer];

                bool isSSMLayer = (config_.architecture == ARCH_QWEN35MOE &&
                                   config_.fullAttentionInterval > 0 &&
                                   (layer % config_.fullAttentionInterval) != 0 &&
                                   !w.ssmOut.empty());

                if (isSSMLayer) {
                    // ---- SSM block (replaces attention) ----

                    // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                    rmsNormSIMD(hidden.data(), attnNorm.data(), w.rmsNormAttn.data(), hiddenSize);

                    // SSM input projection: hiddenSize -> ssmInnerSize via ssmOut
                    uint32_t ssmInnerSize = config_.ssmInnerSize;
                    uint32_t ssmStateSize = config_.ssmStateSize;
                    uint32_t ssmConvKernel = config_.ssmConvKernel;
                    std::vector<float> ssmIn(ssmInnerSize);
                    w.ssmOut.matMulVec(attnNorm.data(), ssmIn.data());

                    // Conv1d with past buffer
                    std::vector<float> convOut(ssmInnerSize);
                    if (ssmConvKernel > 1 && !w.ssmConv1d.empty()) {
                        std::vector<float> convInput(ssmConvKernel * ssmInnerSize);
                        auto &convBuf = kvCache_.ssmConvBuf[layer];
                        uint32_t bufLen = ssmConvKernel - 1;
                        std::memcpy(convInput.data(), convBuf.data(), bufLen * ssmInnerSize * sizeof(float));
                        std::memcpy(convInput.data() + bufLen * ssmInnerSize,
                                    ssmIn.data(), ssmInnerSize * sizeof(float));

                        for (uint32_t i = 0; i < (bufLen - 1) * ssmInnerSize; ++i) {
                            convBuf[i] = convBuf[i + ssmInnerSize];
                        }
                        std::memcpy(convBuf.data() + (bufLen - 1) * ssmInnerSize,
                                    ssmIn.data(), ssmInnerSize * sizeof(float));

                        for (uint32_t c = 0; c < ssmInnerSize; ++c) {
                            double dot = 0.0;
                            const float *wRow = reinterpret_cast<const float *>(w.ssmConv1d.data.data()) + static_cast<size_t>(c) * ssmConvKernel;
                            for (uint32_t k = 0; k < ssmConvKernel; ++k) {
                                dot += static_cast<double>(wRow[k]) * convInput[c * ssmConvKernel + k];
                            }
                            convOut[c] = static_cast<float>(dot);
                        }
                    } else {
                        std::memcpy(convOut.data(), ssmIn.data(), ssmInnerSize * sizeof(float));
                    }

                    // SiLU activation on conv output
                    for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                        convOut[i] = convOut[i] / (1.0f + std::exp(-convOut[i]));
                    }

                    // SSM state update
                    auto &ssmState = kvCache_.ssmState[layer];
                    for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                        float dt = std::log(1.0f + std::exp(w.ssmDtBias.data()[i]));
                        for (uint32_t j = 0; j < ssmStateSize; ++j) {
                            float aVal = w.ssmA.data()[i * ssmStateSize + j];
                            float aBar = std::exp(aVal * dt);
                            uint32_t idx = i * ssmStateSize + j;
                            ssmState[idx] = aBar * ssmState[idx] + convOut[i];
                        }
                    }

                    // Output from SSM state
                    std::vector<float> ssmOut(ssmInnerSize);
                    for (uint32_t i = 0; i < ssmInnerSize; ++i) {
                        double hVal = 0.0;
                        for (uint32_t j = 0; j < ssmStateSize; ++j) {
                            hVal += ssmState[i * ssmStateSize + j];
                        }
                        float gate = w.ssmAlpha.data()[i] * static_cast<float>(hVal) + w.ssmBeta.data()[i];
                        float gateAct = gate / (1.0f + std::exp(-gate));
                        ssmOut[i] = convOut[i] * gateAct;
                    }

                    // Output projection back to hiddenSize using attnO
                    deqMatMulVecF16(w.attnO_deq_f16.data(), ssmOut.data(),
                                    w.attnO.rows, w.attnO.cols,
                                    attnProj.data());

                    // SSM residual (standard residual, no post-norm)
                    addSIMD(hidden.data(), attnProj.data(), hiddenSize);
                } else {
                    // ---- Attention block ----

                    // RMSNorm: attnNorm = rmsNorm(hidden, w.rmsNormAttn)
                    rmsNormSIMD(hidden.data(), attnNorm.data(), w.rmsNormAttn.data(), hiddenSize);

                    // Fused QKV projection: single pass over x for all three projections
                    matMulVecFusedQKV(w.attnQ, w.attnK, w.attnV, attnNorm.data(),
                                      q.data(), k.data(), v.data());

                    if (config_.architecture == ARCH_QWEN2 && !w.attnQBias.empty()) {
                        addSIMD(q.data(), w.attnQBias.data(), nHeads * headDim);
                    }
                    if (config_.architecture == ARCH_QWEN2 && !w.attnKBias.empty()) {
                        addSIMD(k.data(), w.attnKBias.data(), nKVHeads * headDim);
                    }
                    if (config_.architecture == ARCH_QWEN2 && !w.attnVBias.empty()) {
                        addSIMD(v.data(), w.attnVBias.data(), nKVHeads * headDim);
                    }


                    // Apply Q/K norms before RoPE (Gemma4 and Qwen35MoE)
                    if (config_.architecture == ARCH_GEMMA4 && !w.attnQNorm.empty() && !w.attnKNorm.empty()) {
                        applyQKNorms(q.data(), k.data(), 1, nHeads, nKVHeads,
                                     w.attnQNorm.data(), w.attnKNorm.data());
                    } else if (config_.architecture == ARCH_QWEN35MOE && !w.attnQNormMoe.empty() && !w.attnKNormMoe.empty()) {
                        applyQKNorms(q.data(), k.data(), 1, nHeads, nKVHeads,
                                     w.attnQNormMoe.data(), w.attnKNormMoe.data());
                    }

                    // Apply RoPE at position pos (single token, so qSeqLen=kSeqLen=1)
                    applyRoPE(q.data(), k.data(), 1, 1, nHeads, nKVHeads, pos);


                    // Store K, V in cache
                    uint32_t cachePos = static_cast<uint32_t>(kvCache_.pos);
                    float *kCacheLayer =
                            kvCache_.k.data() + layer * maxSeqLen * nKVHeads * headDim;
                    float *vCacheLayer =
                            kvCache_.v.data() + layer * maxSeqLen * nKVHeads * headDim;

                    uint32_t kvSize = nKVHeads * headDim;
                    float *kDst = kCacheLayer + cachePos * kvSize;
                    float *vDst = vCacheLayer + cachePos * kvSize;
                    std::memcpy(kDst, k.data(), kvSize * sizeof(float));
                    std::memcpy(vDst, v.data(), kvSize * sizeof(float));

                    // Attention: single query against all cached positions
                    uint32_t totalCacheLen = cachePos + 1;
                    attentionFused(q.data(), kCacheLayer, vCacheLayer, attnOut.data(),
                                   1, cachePos, totalCacheLen, layer);


                    // Output projection (using dequantized F32 weights for exact float dot product)
                    deqMatMulVecF16(w.attnO_deq_f16.data(), attnOut.data(),
                                    w.attnO.rows, w.attnO.cols,
                                    attnProj.data());


                    // Attention residual + post-attention processing (architecture-aware)
                    if (config_.architecture == ARCH_GEMMA4 && !w.postAttnNorm.empty()) {
                        // Gemma4: post-attention norm + layer scale
                        addSIMD(hidden.data(), attnProj.data(), hiddenSize);
                        rmsNormSIMD(hidden.data(), hidden.data(), w.postAttnNorm.data(), hiddenSize);
                        if (!w.layerOutputScale.empty()) {
                            scaleSIMD(hidden.data(), w.layerOutputScale.data()[0], hiddenSize);
                        }
                    } else {
                        // Qwen2, Qwen35MoE: standard residual (no post-norm)
                        addSIMD(hidden.data(), attnProj.data(), hiddenSize);
                    }

                }// end of else (attention block for non-SSM layers)

                // ---- FFN block ----

                // RMSNorm: ffnNorm = rmsNorm(hidden, w.rmsNormFFN)
                rmsNormSIMD(hidden.data(), ffnNorm.data(), w.rmsNormFFN.data(), hiddenSize);


                // MoE path (expertCount > 0)
                if (config_.architecture == ARCH_GEMMA4 && config_.expertCount > 0) {
                    computeGemma4MoE(ffnNorm.data(), ffnOut.data(), 1, hiddenSize, intermediateSize, w);
                    // Residual + layer scale
                    addSIMD(hidden.data(), ffnOut.data(), hiddenSize);
                    if (!w.layerOutputScale.empty()) {
                        scaleSIMD(hidden.data(), w.layerOutputScale.data()[0], hiddenSize);
                    }
                } else if (config_.architecture == ARCH_QWEN35MOE && config_.expertCount > 0) {
                    // Qwen35MoE MoE path
                    computeQwen35MoE(ffnNorm.data(), ffnOut.data(), 1, hiddenSize, intermediateSize, w);
                    // Standard residual (no post-norm)
                    addSIMD(hidden.data(), ffnOut.data(), hiddenSize);
                } else {
                    // Dense FFN path
                    // Gate and up projections using out-parameter calls
                    w.ffnGate.matMulVec(ffnNorm.data(), gate.data());
                    w.ffnUp.matMulVec(ffnNorm.data(), up.data());


                    // FFN activation (architecture-aware)
                    if (config_.architecture == ARCH_GEMMA4) {
                        // Gemma4: GeGLU activation (gelu(gate) * up)
                        geluInPlace(gate.data(), intermediateSize);
                        for (uint32_t i = 0; i < intermediateSize; ++i) {
                            gate[i] *= up[i];
                        }
                    } else {
                        // Qwen2, Qwen35MoE: SwiGLU activation (silu(gate) * up)
                        swiGLUSIMD(gate.data(), up.data(), intermediateSize);
                    }


                    // Down projection (using dequantized F32 weights for exact float dot product)
                    deqMatMulVecF16(w.ffnDown_deq_f16.data(), gate.data(),
                                    w.ffnDown.rows, w.ffnDown.cols,
                                    ffnOut.data());


                    // FFN residual + post-FFN processing (architecture-aware)
                    if (config_.architecture == ARCH_GEMMA4 && !w.postFFWNorm.empty()) {
                        // Gemma4: post-FFN norm + layer scale
                        addSIMD(hidden.data(), ffnOut.data(), hiddenSize);
                        rmsNormSIMD(hidden.data(), hidden.data(), w.postFFWNorm.data(), hiddenSize);
                        if (!w.layerOutputScale.empty()) {
                            scaleSIMD(hidden.data(), w.layerOutputScale.data()[0], hiddenSize);
                        }
                    } else {
                        // Qwen2, Qwen35MoE: standard residual (no post-norm)
                        addSIMD(hidden.data(), ffnOut.data(), hiddenSize);
                    }
                }

                // Store per-layer hidden state for the last token
                if (pos == seqLen - 1) {
                    perLayerStates.emplace_back(hidden.begin(), hidden.end());
                }
            }

            // Final RMSNorm
            rmsNormSIMD(hidden.data(), hidden.data(), finalNorm_.data(), hiddenSize);

            // Update KV cache position
            kvCache_.pos += 1;
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

        // Apply final logit softcapping (Gemma4 architecture)
        if (config_.architecture == ARCH_GEMMA4 && config_.finalLogitSoftcapping > 0.0f) {
            float cap = config_.finalLogitSoftcapping;
            float invCap = 1.0f / cap;
            for (uint32_t i = 0; i < vocabSize; ++i) {
                logits[i] = std::tanh(logits[i] * invCap) * cap;
            }
        }

        // Print per-layer hidden state stats for the last token
        for (uint32_t layer = 0; layer < nLayers; ++layer) {
            float mn = perLayerStates[layer][0], mx = perLayerStates[layer][0];
            double ssq = 0.0;
            for (uint32_t i = 0; i < hiddenSize; ++i) {
                float v = perLayerStates[layer][i];
                mn = std::min(mn, v);
                mx = std::max(mx, v);
                ssq += (double) v * v;
            }
        }

        // Print top-10 logits
        std::vector<std::pair<float, int32_t>> top10;
        for (uint32_t t = 0; t < vocabSize; ++t)
            top10.emplace_back(logits[t], (int32_t) t);
        std::partial_sort(top10.begin(), top10.begin() + 10, top10.end(),
                          [](const auto &a, const auto &b) { return a.first > b.first; });

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

    std::string Model::formatChat(
            const std::vector<std::pair<std::string, std::string>> &messages,
            bool addGenerationPrompt) const {
        // If no chat template is available, fall back to architecture-specific default
        if (config_.chatTemplate.empty()) {
            if (config_.architecture == ARCH_GEMMA4) {
                // Gemma4 default: <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
                std::string result;
                for (const auto &msg: messages) {
                    result += "<start_of_turn>" + msg.first + "\n" + msg.second + "<end_of_turn>\n";
                }
                if (addGenerationPrompt) {
                    result += "<start_of_turn>model\n";
                }
                return result;
            } else {
                // Qwen2 / Qwen35MoE default: <|im_start|>role\n...<|im_end|>\n
                std::string result;
                for (const auto &msg: messages) {
                    result += "<|im_start|>" + msg.first + "\n" + msg.second + "<|im_end|>\n";
                }
                if (addGenerationPrompt) {
                    result += "<|im_start|>assistant\n";
                }
                return result;
            }
        }

        // Use the dedicated Jinja template renderer
        std::string rendered = ChatTemplateRenderer::render(config_.chatTemplate, messages, addGenerationPrompt);
        // Print rendered prompt with escaped newlines for clarity
        {
            std::string escaped;
            for (char c: rendered) {
                if (c == '\n') escaped += "\\n";
                else if (c == '\r')
                    escaped += "\\r";
                else if (c == '\t')
                    escaped += "\\t";
                else
                    escaped += c;
            }
            std::cout << "[TinyCoder] Rendered prompt (" << rendered.size() << " chars): \"" << escaped << "\"" << std::endl;
        }
        return rendered;
    }

    std::vector<int32_t>
    Model::generate(const std::string &prompt, const InferenceParams &params,
                    std::function<bool(int32_t, const std::string &)> callback) {
        if (!loaded_) {
            return {};
        }

        // Clear KV cache to ensure a clean generation state
        clearKVCache();

        auto t0 = std::chrono::high_resolution_clock::now();

        // Tokenize prompt
        auto tokens = tokenize(prompt);
        if (tokens.empty()) {
            return {};
        }


        if (tokens.size() > 10)
            std::cout << "...";
        std::cout << std::endl;

        // Prefill (process all prompt tokens at once)
        auto logits = forward(tokens);

        // Get the last token's logits
        uint32_t lastIdx = static_cast<uint32_t>(tokens.size()) - 1;
        np::Array<float> lastLogits = np::Array<float>(np::Shape{config_.vocabSize});
        for (uint32_t i = 0; i < config_.vocabSize; ++i) {
            lastLogits.set(i, logits.get(lastIdx * config_.vocabSize + i));
        }


        // Sample first generated token
        int32_t nextToken = sampleToken(lastLogits, params);
        std::string tokenText = tokenizer_.decodeToken(nextToken);


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
                break;
            }

            tokenText = tokenizer_.decodeToken(nextToken);
            generated.push_back(nextToken);

            if (!callback(nextToken, tokenText)) {
                break;
            }
        }

        return generated;
    }

}// namespace tinycoder
