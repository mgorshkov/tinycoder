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

#include "MemHints.hpp"
#include "Model.hpp"

#include "GGMLDequantize.hpp"
#include "ModelInternal.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/sysinfo.h>
#include <vector>

namespace tinycoder {

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

        // Helper to convert a packed FP16 weight matrix to a row-major Q8_K block
        // array (rows x blocksPerRow Q8KBlock), for the prefill Q8_K batch GEMM.
        auto buildQ8K = [](const uint16_t *W_f16, uint32_t rows, uint32_t cols) {
            constexpr uint32_t BLOCK_SIZE = 256;
            uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::vector<Q8KBlock> out(static_cast<size_t>(rows) * blocksPerRow);
            std::vector<float> rowF32(cols);
            // Manual FP16 -> FP32 (Model.cpp is not compiled with F16C).
            for (uint32_t r = 0; r < rows; ++r) {
                const uint16_t *src = W_f16 + static_cast<size_t>(r) * cols;
                for (uint32_t i = 0; i < cols; ++i) {
                    uint16_t h = src[i];
                    uint32_t sign = (h >> 15) & 1;
                    uint32_t exp = (h >> 10) & 0x1F;
                    uint32_t mant = h & 0x3FF;
                    uint32_t f32;
                    if (exp == 0) {
                        if (mant == 0) {
                            f32 = sign << 31;
                        } else {
                            int n = 0;
                            while ((mant & 0x200) == 0 && n < 10) {
                                mant <<= 1;
                                ++n;
                            }
                            mant &= 0x3FF;
                            exp = 112 - n;
                            f32 = (sign << 31) | (exp << 23) | ((mant - 512) << 14);
                        }
                    } else if (exp == 31) {
                        f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
                    } else {
                        exp = exp + (127 - 15);
                        f32 = (sign << 31) | (exp << 23) | (mant << 13);
                    }
                    float v;
                    std::memcpy(&v, &f32, sizeof(float));
                    rowF32[i] = v;
                }
                GGMLDequantize::quantizeQ8K(rowF32.data(), cols,
                                            out.data() + static_cast<size_t>(r) * blocksPerRow);
            }
            return out;
        };

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
                // Pre-pack attnO as Q8_K for the prefill Q8_K batch GEMM (P4).
                layers_[i].attnO_q8k =
                        buildQ8K(layers_[i].attnO_deq_f16.data(),
                                 layers_[i].attnO.rows, layers_[i].attnO.cols);

                // Pre-pack attnQ and attnK for the register-tiled Q8_K batch GEMM
                // prefill path (Q2_K only). This lets the prefill Q/K projections
                // reuse each weight row across all prompt tokens and use the int8
                // _mm256_maddubs_epi16 kernel, instead of the generic row-parallel
                // quantized mat-mul path (see BENCHMARK_REPORT §7.2 "remaining
                // prefill bottleneck"). attnV is usually Q4_K and stays on the
                // generic path.
                if (layers_[i].attnQ.type == GGML_TYPE_Q2_K) {
                    uint64_t numElements = static_cast<uint64_t>(layers_[i].attnQ.rows) * layers_[i].attnQ.cols;
                    layers_[i].attnQ.prepackedData = GGMLDequantize::prepackQ2_K(layers_[i].attnQ.data.data(), numElements);
                }
                if (layers_[i].attnK.type == GGML_TYPE_Q2_K) {
                    uint64_t numElements = static_cast<uint64_t>(layers_[i].attnK.rows) * layers_[i].attnK.cols;
                    layers_[i].attnK.prepackedData = GGMLDequantize::prepackQ2_K(layers_[i].attnK.data.data(), numElements);
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
                // Pre-pack ffnDown as Q8_K for the prefill Q8_K batch GEMM (P4).
                layers_[i].ffnDown_q8k =
                        buildQ8K(layers_[i].ffnDown_deq_f16.data(),
                                 layers_[i].ffnDown.rows, layers_[i].ffnDown.cols);

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
                // Pre-pack ffnDown for the register-tiled Q8_K batch GEMM prefill
                // path (Q2_K only). This lets the prefill ffnDown projection reuse
                // each weight row across all tokens and use the int8
                // _mm256_maddubs_epi16 kernel.
                if (layers_[i].ffnDown.type == GGML_TYPE_Q2_K) {
                    uint64_t numElements = static_cast<uint64_t>(layers_[i].ffnDown.rows) * layers_[i].ffnDown.cols;
                    layers_[i].ffnDown.prepackedData = GGMLDequantize::prepackQ2_K(layers_[i].ffnDown.data.data(), numElements);
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
                // Pre-pack ffnDown for the register-tiled Q8_K batch GEMM prefill
                // path (Q2_K only).
                if (layers_[i].ffnDown.type == GGML_TYPE_Q2_K) {
                    uint64_t numElements = static_cast<uint64_t>(layers_[i].ffnDown.rows) * layers_[i].ffnDown.cols;
                    layers_[i].ffnDown.prepackedData = GGMLDequantize::prepackQ2_K(layers_[i].ffnDown.data.data(), numElements);
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
        // Hint 2 MB huge pages for the ~890 MB F32 embedding table (streamed
        // every token by the LM head; collapses the dTLB working set — see
        // plans/generation_optimizations.md §6.10).
        tinycoder::adviseHugePages(dequantizedEmbeddings_.data.data(),
                                   dequantizedEmbeddings_.data.size() * sizeof(float));

        // Build an FP16 copy of the embedding matrix. The LM head mat-vec reads the
        // entire matrix every token (vocabSize x hiddenSize = ~933 MB for Qwen2.5-Coder).
        // Storing it as FP16 halves that memory traffic. Since the source embeddings
        // are Q2_K (~2-bit), FP16 is lossless (see BENCHMARK_REPORT §7.2).
        auto deqF16 = GGMLDequantize::dequantizeToF16(
                quantizedEmbeddings_.type,
                quantizedEmbeddings_.data.data(),
                embedElements);
        if (!deqF16.empty()) {
            dequantizedEmbeddings_.dataF16 = std::move(deqF16);
            tinycoder::adviseHugePages(dequantizedEmbeddings_.dataF16.data(),
                                       dequantizedEmbeddings_.dataF16.size() * sizeof(uint16_t));
        }

        // Build a Q8_K copy of the embedding matrix. This is the fastest LM head
        // path: it cuts memory bandwidth ~2× vs FP16 and ~4× vs F32 (the LM head
        // mat-vec is the largest single memory read per token). Each 256-element
        // block stores 256 int8 values + a block scale, and the dot product uses
        // _mm256_maddubs_epi16 int8 kernels. Since the source embeddings are Q2_K
        // (~2-bit), Q8_K is lossless.
        {
            uint32_t blocksPerRow = (hiddenSize + 255) / 256;
            std::vector<Q8KBlock> q8k(static_cast<size_t>(vocabSize) * blocksPerRow);
            // Quantize each row independently (Q8_K blocks are per-row).
            // NOTE: deqData was moved into dequantizedEmbeddings_.data above, so
            // read from the persistent buffer (deqData is now empty/moved-from).
            const float *deqBase = dequantizedEmbeddings_.data.data();
            for (uint32_t r = 0; r < vocabSize; ++r) {
                const float *row = deqBase + static_cast<uint64_t>(r) * hiddenSize;
                GGMLDequantize::quantizeQ8K(row, hiddenSize,
                                            q8k.data() + static_cast<uint64_t>(r) * blocksPerRow);
            }
            dequantizedEmbeddings_.dataQ8K = std::move(q8k);
            tinycoder::adviseHugePages(dequantizedEmbeddings_.dataQ8K.data(),
                                       dequantizedEmbeddings_.dataQ8K.size() * sizeof(Q8KBlock));
        }

        // P0 (BENCHMARK_REPORT §5/P0): build per-16-element-subgroup dequantized
        // (min,max) bounds for a separate (non-tied) LM head. K-quants store a
        // per-16-element scale, so per-16 bounds are tight (candidate set
        // approximates top-K). The pruning pass reads only this small table
        // instead of the full ~182 MB Q6_K matrix to determine candidates, then
        // computes exact top-K logits only for those candidates.
        if (!lmHeadTied_ && !lmHead_.empty()) {
            uint32_t bSize = ggmlBlockSize(lmHead_.type);
            uint32_t tSize = ggmlTypeSize(lmHead_.type);
            if (bSize == 256) {
                uint32_t bpr = (lmHead_.cols + 255) / 256;
                constexpr uint32_t SCG = 16;
                uint32_t sgpb = bSize / SCG;// subgroups per block
                lmHeadBoundsBlocksPerRow_ = bpr;
                lmHeadBoundsSubgroupsPerBlock_ = sgpb;
                uint32_t totalSubgroups = bpr * sgpb;
                lmHeadBounds_.assign(static_cast<size_t>(lmHead_.rows) *
                                             totalSubgroups * 2,
                                     0.0f);
                const uint8_t *base = lmHead_.data.data();
                for (uint32_t r = 0; r < lmHead_.rows; ++r) {
                    float *boundsRow = lmHeadBounds_.data() +
                                       static_cast<size_t>(r) * totalSubgroups * 2;
                    for (uint32_t b = 0; b < bpr; ++b) {
                        const uint8_t *block = base +
                                               (static_cast<uint64_t>(r) * bpr + b) * tSize;
                        float blockOut[256];
                        GGMLDequantize::dequantizeBlock(lmHead_.type, block, blockOut, bSize);
                        // Per-16-subgroup min/max over the actual valid columns of
                        // the last (partial) block.
                        uint32_t blkCols = std::min<uint32_t>(bSize, lmHead_.cols - b * bSize);
                        for (uint32_t sg = 0; sg < sgpb; ++sg) {
                            uint32_t start = sg * SCG;
                            uint32_t n = std::min(SCG, blkCols - start);
                            float mn = blockOut[start], mx = blockOut[start];
                            for (uint32_t i = 1; i < n; ++i) {
                                mn = std::min(mn, blockOut[start + i]);
                                mx = std::max(mx, blockOut[start + i]);
                            }
                            boundsRow[(b * sgpb + sg) * 2 + 0] = mn;
                            boundsRow[(b * sgpb + sg) * 2 + 1] = mx;
                        }
                    }
                }
                std::cout << "[TinyCoder] Built LM head bounds: "
                          << lmHead_.rows << " rows x " << bpr << " blocks x "
                          << sgpb << " subgroups ("
                          << (lmHeadBounds_.size() * sizeof(float) / (1024 * 1024))
                          << " MB) for top-K pruning" << std::endl;
            } else {
                std::cerr << "[TinyCoder] P0 pruning requires 256-block quant type (got "
                          << bSize << "); pruning disabled" << std::endl;
            }
        }

        // Plan §3: pre-quantize the separate (non-tied) LM head to Q8_K. The LM
        // head mat-vec is the largest single memory read per token (vocabSize x
        // hiddenSize); storing it as Q8_K (256 int8 + block scale per block) cuts
        // that traffic ~2x vs on-the-fly dequantization, and the dot products
        // run through the int8 _mm256_maddubs_epi16 kernels with the hidden
        // vector quantized once per token. Since the source LM head uses K-quant
        // types, Q8_K is lossless. Mirrors the tied-embedding Q8_K build above.
        if (!lmHeadTied_ && !lmHead_.empty()) {
            uint32_t srcBlockSize = ggmlBlockSize(lmHead_.type);
            uint32_t srcTypeSize = ggmlTypeSize(lmHead_.type);
            uint32_t srcBlocksPerRow = (lmHead_.cols + srcBlockSize - 1) / srcBlockSize;
            uint32_t bpr = (lmHead_.cols + 255) / 256;
            std::vector<Q8KBlock> q8k(static_cast<size_t>(lmHead_.rows) * bpr);
            std::vector<float> rowF32(lmHead_.cols);
            const uint8_t *base = lmHead_.data.data();
            float blockBuf[256];
            for (uint32_t r = 0; r < lmHead_.rows; ++r) {
                float *row = rowF32.data();
                for (uint32_t b = 0; b < srcBlocksPerRow; ++b) {
                    const uint8_t *block =
                            base + (static_cast<uint64_t>(r) * srcBlocksPerRow + b) * srcTypeSize;
                    uint32_t start = b * srcBlockSize;
                    uint32_t n = std::min(srcBlockSize, lmHead_.cols - start);
                    GGMLDequantize::dequantizeBlock(lmHead_.type, block, blockBuf, n);
                    std::memcpy(row + start, blockBuf, n * sizeof(float));
                }
                GGMLDequantize::quantizeQ8K(rowF32.data(), lmHead_.cols,
                                            q8k.data() + static_cast<size_t>(r) * bpr);
            }
            lmHeadQ8K_ = std::move(q8k);
            std::cout << "[TinyCoder] Pre-quantized separate LM head to Q8_K: "
                      << lmHead_.rows << " rows x " << bpr << " blocks ("
                      << (lmHeadQ8K_.size() * sizeof(Q8KBlock) / (1024 * 1024))
                      << " MB)" << std::endl;
        }


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

#ifdef USE_CUDA
        // If the GPU engine is active, reset its (persistent) KV cache too so a
        // new generation session starts from a clean state on both sides.
        gpuClearKV();
#endif

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

}// namespace tinycoder