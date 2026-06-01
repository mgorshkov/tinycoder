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
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ModelConfig.hpp"

// GGML tensor types (shared between GGUFLoader and Model)
enum GGMLType : uint32_t {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S = 19,
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_IQ3_S = 21,
    GGML_TYPE_IQ2_S = 22,
    GGML_TYPE_IQ4_XS = 23,
    GGML_TYPE_I8 = 24,
    GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26,
    GGML_TYPE_COUNT = 27,
};

namespace tinycoder {

    /// @brief Get the byte size of one block for a GGML quantization type.
    /// For non-quantized types (F32, etc.), returns 1.
    /// For quantized types (Q4_K, Q5_K, etc.), returns the block size (e.g., 256).
    uint32_t ggmlBlockSize(uint32_t type);

    /// @brief Get the byte size of one block's data for a GGML quantization type.
    /// For non-quantized types (F32, etc.), returns the element size (e.g., 4).
    /// For quantized types (Q4_K, Q5_K, etc.), returns the block byte size (e.g.,
    /// 144).
    uint32_t ggmlTypeSize(uint32_t type);

    /// @brief GGUF file format reader for loading quantized models.
    ///
    /// Supports GGUF v3 format with IQ3_XXS quantization blocks.
    class GGUFLoader {
    public:
        GGUFLoader() = default;
        ~GGUFLoader() = default;

        /// @brief Load a GGUF model file and extract tensor data.
        /// @param path Path to the .gguf file
        /// @return true if loading succeeded
        bool load(const std::string &path);

        /// @brief Load only metadata from a GGUF file (no tensor data).
        /// Useful for estimating memory requirements before a full load.
        /// @param path Path to the .gguf file
        /// @return true if metadata was read successfully
        bool loadMetadata(const std::string &path);

        /// @brief Get the model configuration parsed from GGUF metadata.
        const ModelConfig &config() const { return config_; }

        /// @brief Get raw tensor data by name.
        /// @param name Tensor name (e.g., "blk.0.attn_q.weight")
        /// @return Pointer to tensor data, or nullptr if not found
        const uint8_t *getTensor(const std::string &name) const;

        /// @brief Get tensor info (offset, size, shape).
        struct TensorInfo {
            uint64_t offset;
            uint64_t size;
            std::vector<uint32_t> shape;
            uint32_t type;// GGML tensor type enum
        };

        const TensorInfo *getTensorInfo(const std::string &name) const;

    private:
        // GGUF header structures
        struct GGUFHeader {
            uint32_t magic;          // "GGUF" magic bytes
            uint32_t version;        // Format version (3)
            uint64_t tensorCount;    // Number of tensors
            uint64_t metadataKVCount;// Number of metadata key-value pairs
        };

        struct GGUFTensorInfo {
            std::string name;
            uint32_t type;              // GGML_TYPE_* enum
            uint64_t offset;            // Offset from start of tensor data
            std::vector<uint32_t> shape;// Dimensions (numpy-style, reversed)
        };

        bool readHeader();
        bool readMetadata();
        bool readTensorInfos();
        bool readTensorData();

        std::ifstream file_;
        std::string filePath_;
        ModelConfig config_;
        GGUFHeader header_{};

        // Metadata storage
        std::unordered_map<std::string, std::string> metadata_;

        // Tensor storage
        std::vector<GGUFTensorInfo> tensorInfos_;
        std::unordered_map<std::string, size_t> tensorNameIndex_;
        std::vector<uint8_t> tensorData_;// All tensor data concatenated

        uint64_t tensorDataOffset_ = 0;
    };

}// namespace tinycoder
