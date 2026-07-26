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

#include "GGUFLoader.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace tinycoder {

    // GGUF magic bytes
    static constexpr uint32_t GGUF_MAGIC = 0x46554747;// "GGUF" little-endian
    static constexpr uint32_t GGUF_VERSION = 3;

    // GGUF metadata value types
    enum GGUFValueType : uint32_t {
        GGUF_TYPE_UINT8 = 0,
        GGUF_TYPE_INT8 = 1,
        GGUF_TYPE_UINT16 = 2,
        GGUF_TYPE_INT16 = 3,
        GGUF_TYPE_UINT32 = 4,
        GGUF_TYPE_INT32 = 5,
        GGUF_TYPE_FLOAT32 = 6,
        GGUF_TYPE_BOOL = 7,
        GGUF_TYPE_STRING = 8,
        GGUF_TYPE_ARRAY = 9,
        GGUF_TYPE_UINT64 = 10,
        GGUF_TYPE_INT64 = 11,
        GGUF_TYPE_FLOAT64 = 12,
    };

    uint32_t ggmlTypeSize(uint32_t type) {
        switch (type) {
            case GGML_TYPE_F32:
                return 4;
            case GGML_TYPE_F16:
                return 2;
            case GGML_TYPE_I8:
                return 1;
            case GGML_TYPE_I16:
                return 2;
            case GGML_TYPE_I32:
                return 4;
            // Quantized types: bytes per block
            // Reference: quants.h (block_q4_0, block_q4_1, etc.)
            case GGML_TYPE_Q4_0:
                return 18;// 32 weights: d(fp16,2) + qs(16) = 18 bytes
            case GGML_TYPE_Q4_1:
                return 20;// 32 weights: d(fp16,2) + m(fp16,2) + qs(16) = 20 bytes
            case GGML_TYPE_Q5_0:
                return 22;// 32 weights: d(fp16,2) + qh(4) + ql(16) = 22 bytes
            case GGML_TYPE_Q5_1:
                return 24;// 32 weights: d(fp16,2) + m(fp16,2) + qh(4) + ql(16) = 24 bytes
            case GGML_TYPE_Q8_0:
                return 34;// 32 weights: d(fp16,2) + qs(32) = 34 bytes
            case GGML_TYPE_Q2_K:
                return 84;// 256 weights in 84 bytes (block_q2_K)
            case GGML_TYPE_Q3_K:
                return 110;// 256 weights in 110 bytes (block_q3_K: 2+64+32+12)
            case GGML_TYPE_Q4_K:
                return 144;// 256 weights in 144 bytes (block_q4_K)
            case GGML_TYPE_Q5_K:
                return 176;// 256 weights in 176 bytes (block_q5_K)
            case GGML_TYPE_Q6_K:
                return 210;// 256 weights in 210 bytes (block_q6_K)
            case GGML_TYPE_IQ2_XXS:
                return 36;// 256 weights in 36 bytes
            case GGML_TYPE_IQ2_XS:
                return 74;// 256 weights in 74 bytes (d:2 + qs:64 + scales:8)
            case GGML_TYPE_IQ3_XXS:
                return 98;// 256 weights in 98 bytes (3.0625 bpw)
            case GGML_TYPE_IQ1_S:
                return 34;// 256 weights in 34 bytes
            case GGML_TYPE_IQ4_NL:
                return 72;// 256 weights in 72 bytes
            case GGML_TYPE_IQ3_S:
                return 110;// 256 weights in 110 bytes (block_iq3_s)
            case GGML_TYPE_IQ2_S:
                return 82;// 256 weights in 82 bytes (block_iq2_s)
            case GGML_TYPE_IQ4_XS:
                return 72;// 256 weights in 72 bytes
            default:
                return 0;
        }
    }

    uint32_t ggmlBlockSize(uint32_t type) {
        switch (type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q8_1:
                return 32;
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_Q8_K:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ4_XS:
                return 256;
            case GGML_TYPE_IQ3_XXS:
                return 256;
            default:
                return 1;
        }
    }

    bool GGUFLoader::loadMetadata(const std::string &path) {
        file_.open(path, std::ios::binary);
        if (!file_.is_open()) {
            std::cerr << "[TinyCoder] Failed to open model file: " << path << std::endl;
            return false;
        }

        if (!readHeader()) {
            file_.close();
            return false;
        }
        if (!readMetadata()) {
            file_.close();
            return false;
        }

        file_.close();
        return true;
    }

    bool GGUFLoader::load(const std::string &path) {
        filePath_ = path;
        file_.open(path, std::ios::binary);
        if (!file_.is_open()) {
            std::cerr << "[TinyCoder] Failed to open model file: " << path << std::endl;
            return false;
        }

        if (!readHeader())
            return false;
        if (!readMetadata())
            return false;
        if (!readTensorInfos())
            return false;
        if (!readTensorData())
            return false;

        file_.close();
        std::cout << "[TinyCoder] Model loaded: " << config_.numLayers
                  << " layers, " << config_.hiddenSize << " hidden, "
                  << config_.numAttentionHeads << " heads." << std::endl;
        return true;
    }

    bool GGUFLoader::readHeader() {
        file_.read(reinterpret_cast<char *>(&header_), sizeof(GGUFHeader));

        if (header_.magic != GGUF_MAGIC) {
            std::cerr << "[TinyCoder] Invalid GGUF magic: 0x" << std::hex
                      << header_.magic << std::endl;
            return false;
        }

        if (header_.version != GGUF_VERSION) {
            std::cerr << "[TinyCoder] Unsupported GGUF version: " << header_.version
                      << std::endl;
            return false;
        }

        return true;
    }

    /// @brief Get the byte size of a single GGUF value type.
    static uint32_t ggufValueTypeSize(uint32_t type) {
        switch (type) {
            case GGUF_TYPE_UINT8:
                return 1;
            case GGUF_TYPE_INT8:
                return 1;
            case GGUF_TYPE_UINT16:
                return 2;
            case GGUF_TYPE_INT16:
                return 2;
            case GGUF_TYPE_UINT32:
                return 4;
            case GGUF_TYPE_INT32:
                return 4;
            case GGUF_TYPE_FLOAT32:
                return 4;
            case GGUF_TYPE_BOOL:
                return 1;
            case GGUF_TYPE_UINT64:
                return 8;
            case GGUF_TYPE_INT64:
                return 8;
            case GGUF_TYPE_FLOAT64:
                return 8;
            default:
                return 0;
        }
    }

    bool GGUFLoader::readMetadata() {
        for (uint64_t i = 0; i < header_.metadataKVCount; ++i) {
            // Read key
            uint64_t keyLen;
            file_.read(reinterpret_cast<char *>(&keyLen), sizeof(uint64_t));
            std::string key(keyLen, '\0');
            file_.read(key.data(), keyLen);

            // Read value type
            uint32_t valueType;
            file_.read(reinterpret_cast<char *>(&valueType), sizeof(uint32_t));

            // Read value based on type
            switch (valueType) {
                case GGUF_TYPE_UINT8: {
                    uint8_t val;
                    file_.read(reinterpret_cast<char *>(&val), sizeof(uint8_t));
                    metadata_[key] = std::to_string(val);
                    break;
                }
                case GGUF_TYPE_INT8: {
                    int8_t val;
                    file_.read(reinterpret_cast<char *>(&val), sizeof(int8_t));
                    metadata_[key] = std::to_string(val);
                    break;
                }
                case GGUF_TYPE_UINT16: {
                    uint16_t val;
                    file_.read(reinterpret_cast<char *>(&val), sizeof(uint16_t));
                    metadata_[key] = std::to_string(val);
                    break;
                }
                case GGUF_TYPE_INT16: {
                    int16_t val;
                    file_.read(reinterpret_cast<char *>(&val), sizeof(int16_t));
                    metadata_[key] = std::to_string(val);
                    break;
                }
                case GGUF_TYPE_UINT32: {
                    uint32_t val;
                    file_.read(reinterpret_cast<char *>(&val), sizeof(uint32_t));
                    metadata_[key] = std::to_string(val);
                    break;
                }
                case GGUF_TYPE_INT32: {
                    int32_t val;
                    file_.read(reinterpret_cast<char *>(&val), sizeof(int32_t));
                    metadata_[key] = std::to_string(val);
                    break;
                }
                case GGUF_TYPE_FLOAT32: {
                    float val;
                    file_.read(reinterpret_cast<char *>(&val), sizeof(float));
                    metadata_[key] = std::to_string(val);
                    break;
                }
                case GGUF_TYPE_BOOL: {
                    uint8_t val;
                    file_.read(reinterpret_cast<char *>(&val), sizeof(uint8_t));
                    metadata_[key] = val ? "true" : "false";
                    break;
                }
                case GGUF_TYPE_STRING: {
                    uint64_t strLen;
                    file_.read(reinterpret_cast<char *>(&strLen), sizeof(uint64_t));
                    std::string val(strLen, '\0');
                    file_.read(val.data(), strLen);
                    metadata_[key] = val;
                    break;
                }
                case GGUF_TYPE_UINT64: {
                    uint64_t val;
                    file_.read(reinterpret_cast<char *>(&val), sizeof(uint64_t));
                    metadata_[key] = std::to_string(val);
                    break;
                }
                case GGUF_TYPE_INT64: {
                    int64_t val;
                    file_.read(reinterpret_cast<char *>(&val), sizeof(int64_t));
                    metadata_[key] = std::to_string(val);
                    break;
                }
                case GGUF_TYPE_FLOAT64: {
                    double val;
                    file_.read(reinterpret_cast<char *>(&val), sizeof(double));
                    metadata_[key] = std::to_string(val);
                    break;
                }
                case GGUF_TYPE_ARRAY: {
                    uint32_t arrayType;
                    file_.read(reinterpret_cast<char *>(&arrayType), sizeof(uint32_t));
                    uint64_t arrayLen;
                    file_.read(reinterpret_cast<char *>(&arrayLen), sizeof(uint64_t));
                    // Store first element for numeric arrays (used by parseArchKey)
                    // Some metadata (e.g. head_count_kv) is stored as per-layer arrays
                    if (arrayLen > 0) {
                        switch (arrayType) {
                            case GGUF_TYPE_UINT8: {
                                uint8_t val;
                                file_.read(reinterpret_cast<char *>(&val), sizeof(uint8_t));
                                metadata_[key] = std::to_string(val);
                                // Skip remaining elements
                                for (uint64_t j = 1; j < arrayLen; ++j) {
                                    file_.seekg(1, std::ios::cur);
                                }
                                break;
                            }
                            case GGUF_TYPE_INT8: {
                                int8_t val;
                                file_.read(reinterpret_cast<char *>(&val), sizeof(int8_t));
                                metadata_[key] = std::to_string(val);
                                for (uint64_t j = 1; j < arrayLen; ++j) {
                                    file_.seekg(1, std::ios::cur);
                                }
                                break;
                            }
                            case GGUF_TYPE_UINT16: {
                                uint16_t val;
                                file_.read(reinterpret_cast<char *>(&val), sizeof(uint16_t));
                                metadata_[key] = std::to_string(val);
                                for (uint64_t j = 1; j < arrayLen; ++j) {
                                    file_.seekg(2, std::ios::cur);
                                }
                                break;
                            }
                            case GGUF_TYPE_INT16: {
                                int16_t val;
                                file_.read(reinterpret_cast<char *>(&val), sizeof(int16_t));
                                metadata_[key] = std::to_string(val);
                                for (uint64_t j = 1; j < arrayLen; ++j) {
                                    file_.seekg(2, std::ios::cur);
                                }
                                break;
                            }
                            case GGUF_TYPE_UINT32: {
                                uint32_t val;
                                file_.read(reinterpret_cast<char *>(&val), sizeof(uint32_t));
                                metadata_[key] = std::to_string(val);
                                for (uint64_t j = 1; j < arrayLen; ++j) {
                                    file_.seekg(4, std::ios::cur);
                                }
                                break;
                            }
                            case GGUF_TYPE_INT32: {
                                int32_t val;
                                file_.read(reinterpret_cast<char *>(&val), sizeof(int32_t));
                                metadata_[key] = std::to_string(val);
                                for (uint64_t j = 1; j < arrayLen; ++j) {
                                    file_.seekg(4, std::ios::cur);
                                }
                                break;
                            }
                            case GGUF_TYPE_FLOAT32: {
                                float val;
                                file_.read(reinterpret_cast<char *>(&val), sizeof(float));
                                metadata_[key] = std::to_string(val);
                                for (uint64_t j = 1; j < arrayLen; ++j) {
                                    file_.seekg(4, std::ios::cur);
                                }
                                break;
                            }
                            case GGUF_TYPE_BOOL: {
                                uint8_t val;
                                file_.read(reinterpret_cast<char *>(&val), sizeof(uint8_t));
                                metadata_[key] = val ? "true" : "false";
                                for (uint64_t j = 1; j < arrayLen; ++j) {
                                    file_.seekg(1, std::ios::cur);
                                }
                                break;
                            }
                            case GGUF_TYPE_UINT64: {
                                uint64_t val;
                                file_.read(reinterpret_cast<char *>(&val), sizeof(uint64_t));
                                metadata_[key] = std::to_string(val);
                                for (uint64_t j = 1; j < arrayLen; ++j) {
                                    file_.seekg(8, std::ios::cur);
                                }
                                break;
                            }
                            case GGUF_TYPE_INT64: {
                                int64_t val;
                                file_.read(reinterpret_cast<char *>(&val), sizeof(int64_t));
                                metadata_[key] = std::to_string(val);
                                for (uint64_t j = 1; j < arrayLen; ++j) {
                                    file_.seekg(8, std::ios::cur);
                                }
                                break;
                            }
                            case GGUF_TYPE_FLOAT64: {
                                double val;
                                file_.read(reinterpret_cast<char *>(&val), sizeof(double));
                                metadata_[key] = std::to_string(val);
                                for (uint64_t j = 1; j < arrayLen; ++j) {
                                    file_.seekg(8, std::ios::cur);
                                }
                                break;
                            }
                            case GGUF_TYPE_STRING: {
                                // String arrays (e.g. tokenizer.ggml.tokens) - skip all elements
                                for (uint64_t j = 0; j < arrayLen; ++j) {
                                    uint64_t elemLen;
                                    file_.read(reinterpret_cast<char *>(&elemLen), sizeof(uint64_t));
                                    file_.seekg(elemLen, std::ios::cur);
                                }
                                break;
                            }
                            default: {
                                // Unknown element type - skip all elements
                                uint32_t elemSize = ggufValueTypeSize(arrayType);
                                if (elemSize > 0) {
                                    file_.seekg(elemSize * arrayLen, std::ios::cur);
                                }
                                break;
                            }
                        }
                    }
                    break;
                }
                default:
                    std::cerr << "[TinyCoder] Unknown metadata type: " << valueType
                              << std::endl;
                    return false;
            }

            // Parse model configuration from metadata (architecture-agnostic)
            if (key == "general.architecture") {
                config_.architecture = metadata_[key];
            } else if (key == "general.name") {
                config_.modelName = metadata_[key];
            } else if (key == "tokenizer.ggml.vocab_size") {
                config_.vocabSize = std::stoul(metadata_[key]);
            } else if (key == "tokenizer.chat_template") {
                config_.chatTemplate = metadata_[key];
            }

            // Architecture-specific keys - detect prefix from key
            // We check for known architecture prefixes
            auto parseArchKey = [&](const std::string &prefix) {
                if (key == prefix + ".block_count") {
                    config_.numLayers = std::stoul(metadata_[key]);
                } else if (key == prefix + ".context_length") {
                    config_.maxSeqLen = std::stoul(metadata_[key]);
                } else if (key == prefix + ".embedding_length") {
                    config_.hiddenSize = std::stoul(metadata_[key]);
                } else if (key == prefix + ".feed_forward_length") {
                    config_.intermediateSize = std::stoul(metadata_[key]);
                } else if (key == prefix + ".attention.head_count") {
                    config_.numAttentionHeads = std::stoul(metadata_[key]);
                } else if (key == prefix + ".attention.head_count_kv") {
                    // KV heads may be an array (per-layer) - use first element
                    // For now, just use the first value
                    config_.numKVHeads = std::stoul(metadata_[key]);
                } else if (key == prefix + ".rope.freq_base") {
                    config_.ropeTheta = std::stof(metadata_[key]);
                } else if (key == prefix + ".vocab_size") {
                    config_.vocabSize = std::stoul(metadata_[key]);
                } else if (key == prefix + ".final_logit_softcapping") {
                    config_.finalLogitSoftcapping = std::stof(metadata_[key]);
                } else if (key == prefix + ".expert_count") {
                    config_.expertCount = std::stoul(metadata_[key]);
                } else if (key == prefix + ".expert_used_count") {
                    config_.expertUsedCount = std::stoul(metadata_[key]);
                } else if (key == prefix + ".expert_feed_forward_length") {
                    config_.expertFeedForwardLength = std::stoul(metadata_[key]);
                } else if (key == prefix + ".expert_shared_feed_forward_length") {
                    config_.expertSharedFeedForwardLength = std::stoul(metadata_[key]);
                } else if (key == prefix + ".rope.dimension_count") {
                    config_.ropeDimensionCount = std::stoul(metadata_[key]);
                } else if (key == prefix + ".full_attention_interval") {
                    config_.fullAttentionInterval = std::stoul(metadata_[key]);
                } else if (key == prefix + ".ssm.inner_size") {
                    config_.ssmInnerSize = std::stoul(metadata_[key]);
                } else if (key == prefix + ".ssm.state_size") {
                    config_.ssmStateSize = std::stoul(metadata_[key]);
                } else if (key == prefix + ".ssm.conv_kernel") {
                    config_.ssmConvKernel = std::stoul(metadata_[key]);
                } else if (key == prefix + ".ssm.group_count") {
                    config_.ssmGroupCount = std::stoul(metadata_[key]);
                } else if (key == prefix + ".ssm.time_step_rank") {
                    config_.ssmTimeStepRank = std::stoul(metadata_[key]);
                } else if (key == prefix + ".nextn_predict_layers") {
                    config_.nextnPredictLayers = std::stoul(metadata_[key]);
                }
            };

            // Try all known architecture prefixes
            parseArchKey("qwen2");
            parseArchKey("gemma4");
            parseArchKey("qwen35moe");
        }

        config_.headDim = config_.hiddenSize / config_.numAttentionHeads;

        return true;
    }

    bool GGUFLoader::readTensorInfos() {
        tensorInfos_.reserve(header_.tensorCount);

        for (uint64_t i = 0; i < header_.tensorCount; ++i) {
            GGUFTensorInfo info;

            // Read name
            uint64_t nameLen;
            file_.read(reinterpret_cast<char *>(&nameLen), sizeof(uint64_t));
            info.name.resize(nameLen);
            file_.read(info.name.data(), nameLen);

            // Read number of dimensions
            uint32_t nDims;
            file_.read(reinterpret_cast<char *>(&nDims), sizeof(uint32_t));

            // Read dimensions (reversed: numpy shape order)
            // GGUF stores dimensions as uint64_t, but we store them as uint32_t
            info.shape.resize(nDims);
            for (uint32_t d = 0; d < nDims; ++d) {
                uint64_t dim;
                file_.read(reinterpret_cast<char *>(&dim), sizeof(uint64_t));
                info.shape[d] = static_cast<uint32_t>(dim);
            }

            // Read tensor type
            file_.read(reinterpret_cast<char *>(&info.type), sizeof(uint32_t));

            // Read offset (relative to tensor data start)
            file_.read(reinterpret_cast<char *>(&info.offset), sizeof(uint64_t));

            tensorInfos_.push_back(info);
            tensorNameIndex_[info.name] = tensorInfos_.size() - 1;
        }

        // Align to 32 bytes (GGUF alignment requirement)
        uint64_t pos = file_.tellg();
        uint64_t aligned = (pos + 31) & ~31;
        tensorDataOffset_ = aligned;
        file_.seekg(aligned);

        return true;
    }

    bool GGUFLoader::readTensorData() {
        // Calculate the actual total tensor data size from the file.
        // GGUF v3 stores tensors sequentially with 32-byte alignment padding
        // between them. The tensor offsets in the file are absolute positions
        // within the tensor data section, so we must read the section as-is
        // (including padding gaps) for the offsets to be valid.
        uint64_t totalSize = 0;
        for (const auto &info: tensorInfos_) {
            uint64_t tensorSize = ggmlTypeSize(info.type);
            uint32_t blockSize = ggmlBlockSize(info.type);

            // Calculate number of elements
            uint64_t numElements = 1;
            for (auto dim: info.shape) {
                numElements *= dim;
            }

            // For quantized types, calculate block count
            if (blockSize > 1) {
                uint64_t numBlocks = (numElements + blockSize - 1) / blockSize;
                tensorSize = numBlocks * tensorSize;
            } else {
                tensorSize = numElements * tensorSize;
            }

            // The end of this tensor (offset + size), which may include
            // padding before the next tensor
            uint64_t tensorEnd = info.offset + tensorSize;
            if (tensorEnd > totalSize) {
                totalSize = tensorEnd;
            }
        }

        if (totalSize == 0) {
            std::cerr << "[TinyCoder] No tensor data to read" << std::endl;
            return false;
        }

        tensorData_.resize(totalSize);
        file_.read(reinterpret_cast<char *>(tensorData_.data()), totalSize);

        if (!file_) {
            std::cerr << "[TinyCoder] Failed to read tensor data: expected "
                      << totalSize << " bytes but read " << file_.gcount() << " bytes"
                      << std::endl;
            return false;
        }

        return true;
    }

    const uint8_t *GGUFLoader::getTensor(const std::string &name) const {
        auto it = tensorNameIndex_.find(name);
        if (it == tensorNameIndex_.end()) {
            return nullptr;
        }

        const auto &info = tensorInfos_[it->second];
        return tensorData_.data() + info.offset;
    }

    const GGUFLoader::TensorInfo *
    GGUFLoader::getTensorInfo(const std::string &name) const {
        auto it = tensorNameIndex_.find(name);
        if (it == tensorNameIndex_.end()) {
            return nullptr;
        }

        const auto &info = tensorInfos_[it->second];
        static thread_local TensorInfo result;
        result.offset = info.offset;
        result.size = 0;// Will be calculated
        result.shape = info.shape;
        result.type = info.type;
        return &result;
    }

}// namespace tinycoder
