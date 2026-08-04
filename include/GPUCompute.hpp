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
#include <string>
#include <vector>

#ifdef USE_CUDA

namespace tinycoder::gpu {

    /// @brief One quantized weight matrix uploaded to the device.
    ///
    /// `q` holds the raw GGUF quantized bytes (Q2_K/Q3_K/Q4_K/...) so single-token
    /// generation streams the compact block layout straight from VRAM (the same
    /// memory traffic llama.cpp's CUDA GEMV uses).  `f16` holds the FP16
    /// dequantized twin used by the cuBLAS tensor-core batch GEMM prefill path
    /// (Turing has no quantized tensor cores, so prefill must dequantize).
    struct DeviceMatrix {
        void *q = nullptr;        // raw quantized bytes (row-major block layout)
        void *f16 = nullptr;      // FP16 dequantized [rows][cols] (row-major)
        uint32_t rows = 0;        // output rows
        uint32_t cols = 0;        // input columns
        uint32_t type = 0;        // GGML_TYPE_* enum
        uint32_t blocksPerRow = 0;// number of 256-elem blocks per row
        uint32_t rowBytes = 0;    // q row stride in bytes

        // Host-side FP16 twin (owned by the adapter, freed after upload).
        uint16_t *f16Host = nullptr;
        bool empty_ = false;
        bool hasF16() const { return f16 != nullptr; }
        bool hasQ() const { return q != nullptr; }
    };

    /// @brief Qwen2 dense LayerWeight block on the device (host pointers BEFORE
    /// upload: upload() copies them to device and keeps the descriptors).
    struct DeviceLayer {
        DeviceMatrix attnQ, attnK, attnV, attnO;
        DeviceMatrix ffnGate, ffnUp, ffnDown;
        float *attnQBias = nullptr, *attnKBias = nullptr, *attnVBias = nullptr;
        float *rmsNormAttn = nullptr, *rmsNormFFN = nullptr;
    };

    /// @brief Geometry + final norm shared by all kernels.
    struct ModelGeometry {
        uint32_t hiddenSize = 1536;
        uint32_t intermediateSize = 8960;
        uint32_t numLayers = 28;
        uint32_t numGpuLayers = 0;// 0 = offload all layers; >0 = offload only the
                                  // first N layers (llama.cpp -ngl style partial
                                  // offload, CPU handles the rest)
        uint32_t numAttentionHeads = 12;
        uint32_t numKVHeads = 2;
        uint32_t headDim = 128;
        uint32_t maxSeqLen = 2048;
        uint32_t vocabSize = 151936;
        float ropeTheta = 1000000.0f;
        uint32_t qwen2Bias = 1;          // Qwen2 has Q/K/V biases; Gemma4/Qwen35MoE don't
        const float *finalNorm = nullptr;// hiddenSize floats (host)
        uint32_t lmHeadTied = 1;         // 1: LM head == token embeddings (embedQ). 0: the
                                         // adapter uploads the separate output.weight matrix
                                         // as embedQ (rows == vocabSize).
    };

    /// @brief GPU model runtime: owns device memory, hosts the persistent KV cache
    /// and drives a full forward pass (quantized GEMV generation or cuBLAS fp16
    /// GEMM prefill).
    class GPUModel {
    public:
        /// @brief Allocate device buffers and upload weights.
        /// @param layers Host layer weight descriptors (first geom.numGpuLayers of
        ///        them are uploaded; 0 in geom = all)
        /// @param geom Geometry + final norm
        /// @param embedQ Raw quantized token-embedding matrix (vocab x hidden)
        /// @param embedType GGML type of the embedding matrix
        /// @param embedRowBytes Byte stride of one embedding row
        /// @param lmHeadQ Raw quantized separate LM-head matrix (rows==vocabSize,
        ///        cols==hiddenSize); NULL when the LM head is tied to the embeddings
        /// @param lmHeadType GGML type of the separate LM head
        /// @param lmHeadRowBytes Byte stride of one separate-LM-head row
        /// @param errMsg Error message on failure
        bool upload(const std::vector<DeviceLayer> &layers, const ModelGeometry &geom,
                    const void *embedQ, uint32_t embedType, uint32_t embedRowBytes,
                    const void *lmHeadQ, uint32_t lmHeadType, uint32_t lmHeadRowBytes,
                    std::string &errMsg);

        /// @brief Copy the current GPU-side hidden state (after the offloaded layer
        /// prefix) back to the host.  Used by the partial-offload path: the caller
        /// feeds this vector into the CPU Model::forward for the remaining layers.
        /// @param hiddenOut Host buffer (seqLen * hiddenSize floats)
        /// @param seqLen Number of tokens whose hidden state to copy
        bool copyHiddenOut(float *hiddenOut, uint32_t seqLen, std::string &errMsg);

        /// @brief Run one forward pass entirely on GPU.
        /// @param tokens Input token IDs
        /// @param computeAllLogits Compute logits for ALL tokens (prefill debug
        ///        path); otherwise only the last token's logits.
        /// @param logitsOut Host buffer for the result.  When computeAllLogits is
        ///        false: vocabSize floats.  Otherwise: seqLen*vocabSize floats.
        bool forward(const std::vector<int32_t> &tokens, bool computeAllLogits,
                     float *logitsOut, std::string &errMsg);

        /// @brief Zero the GPU KV cache and reset the cache position.
        void clearKVCache();

        /// @brief Free all device resources.
        ~GPUModel();

        GPUModel() = default;
        GPUModel(const GPUModel &) = delete;
        GPUModel &operator=(const GPUModel &) = delete;

    private:
        // Grow-on-demand device scratch, sized for the current seqLen.
        // NOTE: fp16 scratch buffers are stored as uint16_t* (bit-compatible with
        // CUDA __half) so this header stays valid in host-only translation units;
        // GPUCompute.cu casts them to __half*/__half2* at the use sites.
        struct DeviceScratch {
            int32_t *tokens = nullptr;
            float *hidden = nullptr;      // [seqLen][hidden]
            uint16_t *hiddenF16 = nullptr;// [seqLen][hidden] fp16 twin for prefill GEMMs
            // [seqLen][hidden] RMSNorm output.  The norm results must NOT be
            // written back into `hidden`: `hidden` is the residual stream the
            // attention/FFN blocks add into (the CPU path norms into separate
            // attnNorm/ffnNorm buffers). Overwriting it with the normed vector
            // destroys the residual connections and produces garbage.
            float *norm = nullptr;
            float *q = nullptr;            // [seqLen][qHeads*headDim]
            float *k = nullptr;            // [seqLen][kHeads*headDim]
            float *v = nullptr;            // [seqLen][kHeads*headDim]
            float *attnOut = nullptr;      // [seqLen][qHeads*headDim]
            uint16_t *attnOutF16 = nullptr;// [seqLen][qHeads*headDim] fp16 twin
            float *attnProj = nullptr;     // [seqLen][hidden]
            float *gate = nullptr;         // [seqLen][intermediate]
            uint16_t *gateF16 = nullptr;   // [seqLen][intermediate] fp16 twin
            float *up = nullptr;           // [seqLen][intermediate]
            float *ffnOut = nullptr;       // [seqLen][hidden]
            float *logits = nullptr;       // [seqLen*vocab] (max of the two)
            uint32_t seqCap = 0;
        };

        bool allocated_ = false;
        DeviceLayer *layers_ = nullptr;// device-backed descriptors
        float *kvK_ = nullptr;         // [nGpuLayers][maxSeqLen*kHeads*headDim]
        float *kvV_ = nullptr;
        float *finalNorm_ = nullptr;// device [hidden]
        float *embedQ_ = nullptr;   // raw quantized token-embedding bytes
        float *lmHeadQ_ = nullptr;  // raw quantized separate LM head (nullable)
        float *ropeCos_ = nullptr;  // [maxSeqLen][headDim/2]
        float *ropeSin_ = nullptr;
        ModelGeometry geom_{};
        DeviceScratch scratch_{};
        uint32_t embedType_ = 0;
        uint32_t embedRowBytes_ = 0;
        uint32_t lmHeadType_ = 0;
        uint32_t lmHeadRowBytes_ = 0;
        size_t kvPos_ = 0;
        bool scratchAlloc_ = false;

        bool ensureScratch(uint32_t seqLen, std::string &errMsg);
        void destroyScratch();
        void destroy();
    };

    /// @brief Global GPU enable flag (default ON in CUDA builds; $TINYCODER_GPU=0
    /// disables the offload engine, forcing the CPU path).
    bool gpuEnabled();

}// namespace tinycoder::gpu

#endif// USE_CUDA