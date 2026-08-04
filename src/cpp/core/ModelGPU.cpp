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

#ifdef USE_CUDA

#include "GGUFLoader.hpp"
#include "GPUCompute.hpp"
#include "ModelInternal.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace tinycoder {

    using namespace gpu;

    namespace {

        uint32_t typeBlockBytes(uint32_t type) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return 84;
                case GGML_TYPE_Q3_K:
                    return 110;
                case GGML_TYPE_Q4_K:
                    return 144;
                case GGML_TYPE_Q6_K:
                    return 210;
                case GGML_TYPE_Q8_K:
                    return 292;
                default:
                    return 0;
            }
        }

        // fp32 -> fp16 (IEEE half, host side)
        uint16_t f32ToF16(float f) {
            uint32_t u;
            std::memcpy(&u, &f, 4);
            uint32_t sign = (u >> 31) & 1;
            int32_t e = static_cast<int32_t>((u >> 23) & 0xFF) - 127;
            uint32_t m = u & 0x7FFFFF;
            uint16_t h;
            if (e > 15) {
                h = static_cast<uint16_t>((sign << 15) | (0x1F << 10));
                if (m) h |= 0x200;
            } else if (e > -14) {
                h = static_cast<uint16_t>(
                        (sign << 15) | ((static_cast<uint32_t>(e + 15) & 0x1F) << 10) |
                        (m >> 13));
            } else if (e > -24) {
                m = (m | 0x800000) >> (14 - e);
                h = static_cast<uint16_t>((sign << 15) | (m >> 13));
            } else {
                h = static_cast<uint16_t>(sign << 15);
            }
            return h;
        }

        // Host wrapper: dequantize a [rows x cols] quantized matrix to FP16,
        // row-major, using the shared GGML block dequantizers. Writes DIRECTLY
        // into the caller-provided destination buffer (no 3 GB temporary
        // std::vector + memcpy; that doubled peak host RSS and could swap on
        // low-RAM hosts).
        //
        // Rows are independent in the row-major block layout, so the work is
        // split across hardware threads.  For the 1.5B model this turns the
        // one-time upload from a ~60 s single-threaded pass into a few seconds.
        uint64_t dequantizeToF16Into(void *dst, const void *qData, uint32_t type,
                                     uint32_t rows, uint32_t cols) {
            if (dst == nullptr || qData == nullptr || rows == 0 || cols == 0)
                return 0;
            const uint8_t *base = static_cast<const uint8_t *>(qData);
            uint16_t *out = static_cast<uint16_t *>(dst);
            const uint32_t blockSize = ggmlBlockSize(type);
            const uint32_t typeSize = ggmlTypeSize(type);
            if (blockSize == 0 || typeSize == 0) return 0;

            const uint32_t blocksPerRow = (cols + blockSize - 1) / blockSize;

            uint32_t nThreads = std::max(1u, std::thread::hardware_concurrency());
            if (nThreads > rows) nThreads = rows;
            std::atomic<uint32_t> nextRow{0};

            auto worker = [&]() {
                float blockBuf[256];
                for (;;) {
                    uint32_t r = nextRow.fetch_add(1);
                    if (r >= rows) break;
                    const uint8_t *rowBase =
                            base + static_cast<uint64_t>(r) * blocksPerRow * typeSize;
                    uint16_t *dstRow = out + static_cast<uint64_t>(r) * cols;
                    for (uint32_t b = 0; b < blocksPerRow; ++b) {
                        uint32_t start = b * blockSize;
                        uint32_t n = std::min(blockSize, cols - start);
                        GGMLDequantize::dequantizeBlock(
                                type, rowBase + static_cast<uint64_t>(b) * typeSize,
                                blockBuf, n);
                        for (uint32_t i = 0; i < n; ++i) {
                            dstRow[start + i] = f32ToF16(blockBuf[i]);
                        }
                    }
                }
            };

            std::vector<std::thread> pool;
            pool.reserve(nThreads);
            for (uint32_t t = 0; t < nThreads; ++t) {
                pool.emplace_back(worker);
            }
            for (auto &th: pool) th.join();
            return static_cast<uint64_t>(rows) * cols;
        }

        DeviceMatrix toDeviceMatrix(const QuantizedMatrix &m) {
            DeviceMatrix dm;
            if (m.empty()) return dm;
            dm.rows = m.rows;
            dm.cols = m.cols;
            dm.type = m.type;
            dm.blocksPerRow = (m.cols + 255) / 256;
            dm.rowBytes = dm.blocksPerRow * typeBlockBytes(m.type);
            if (dm.rowBytes == 0) {
                dm.empty_ = true;
                return dm;
            }
            // Host pointers captured here; upload() copies them to the device.
            dm.q = const_cast<uint8_t *>(m.data.data());
            // FP16 twin for the cuBLAS prefill path (only if cols is a multiple
            // of 32; otherwise the GEMM stays on CPU).
            if ((dm.cols % 32) == 0 && m.rows > 0 && m.cols > 0) {
                dm.f16Host = new uint16_t[static_cast<uint64_t>(m.rows) * m.cols];
                if (dequantizeToF16Into(dm.f16Host, m.data.data(), m.type, m.rows,
                                        m.cols) == 0) {
                    delete[] dm.f16Host;
                    dm.f16Host = nullptr;
                } else {
                    dm.f16 = dm.f16Host;// upload() reads src.f16 as host data
                }
            }
            return dm;
        }

    }// namespace

    /// @brief GPU offload adapter: mirrors the Model weights to the device.
    /// Owns the FP16 twins and the tinycoder::gpu::GPUModel runtime.
    class ModelGPUAdapter {
    public:
        ModelGPUAdapter() = default;
        ~ModelGPUAdapter() { destroy(); }

        // Worker that assumes mtx_ is ALREADY held (no locking inside).
        bool ensureUploadedLocked(Model &m, std::string &err) {
            if (uploaded_) return true;
            if (!gpuEnabled()) {
                err = "GPU disabled (opt out with TINYCODER_GPU=0)";
                return false;
            }
            const ModelConfig &cfg = m.config_;
            const auto &layers = m.layers_;
            const auto &emb = m.quantizedEmbeddings_;

            // ---- Build device layer descriptors for the offloaded prefix ----
            // Default: offload ALL layers (full model on GPU). $TINYCODER_NGL
            // overrides with a partial offload count (llama.cpp -ngl style).
            uint32_t nGpu = cfg.numLayers;
            if (const char *e = std::getenv("TINYCODER_NGL")) {
                int ngl = std::atoi(e);
                if (ngl > 0 && static_cast<uint32_t>(ngl) < nGpu) nGpu = static_cast<uint32_t>(ngl);
            }
            std::vector<DeviceLayer> deviceLayers;
            deviceLayers.reserve(nGpu);
            for (uint32_t L = 0; L < nGpu; ++L) {
                const auto &w = layers[L];
                DeviceLayer dl;
                dl.attnQ = toDeviceMatrix(w.attnQ);
                dl.attnK = toDeviceMatrix(w.attnK);
                dl.attnV = toDeviceMatrix(w.attnV);
                dl.attnO = toDeviceMatrix(w.attnO);
                dl.ffnGate = toDeviceMatrix(w.ffnGate);
                dl.ffnUp = toDeviceMatrix(w.ffnUp);
                dl.ffnDown = toDeviceMatrix(w.ffnDown);
                if (cfg.architecture == ARCH_QWEN2) {
                    dl.attnQBias = const_cast<float *>(
                            w.attnQBias.empty() ? nullptr : w.attnQBias.data());
                    dl.attnKBias = const_cast<float *>(
                            w.attnKBias.empty() ? nullptr : w.attnKBias.data());
                    dl.attnVBias = const_cast<float *>(
                            w.attnVBias.empty() ? nullptr : w.attnVBias.data());
                }
                dl.rmsNormAttn = const_cast<float *>(
                        w.rmsNormAttn.empty() ? nullptr : w.rmsNormAttn.data());
                dl.rmsNormFFN = const_cast<float *>(
                        w.rmsNormFFN.empty() ? nullptr : w.rmsNormFFN.data());
                deviceLayers.push_back(dl);
            }

            // ---- Geometry + embedding ----
            gpu::ModelGeometry geom;
            geom.hiddenSize = cfg.hiddenSize;
            geom.intermediateSize = cfg.intermediateSize;
            geom.numLayers = cfg.numLayers;
            geom.numGpuLayers = nGpu;
            geom.numAttentionHeads = cfg.numAttentionHeads;
            geom.numKVHeads = cfg.numKVHeads;
            geom.headDim = cfg.headDim;
            geom.maxSeqLen = cfg.maxSeqLen;
            geom.vocabSize = cfg.vocabSize;
            geom.ropeTheta = cfg.ropeTheta;
            geom.qwen2Bias = (cfg.architecture == ARCH_QWEN2) ? 1u : 0u;
            geom.finalNorm = m.finalNorm_.empty() ? nullptr : m.finalNorm_.data();

            // ---- Token embedding (always the embedding matrix) ----
            uint32_t embBlockBytes = typeBlockBytes(emb.type);
            uint32_t embBlocksPerRow = (emb.hiddenSize + 255) / 256;
            uint32_t embRowBytes = embBlockBytes * embBlocksPerRow;

            // ---- Separate LM head (output.weight), if any ----
            // When the LM head is tied, geom.lmHeadTied == 1 and the engine's
            // logits GEMV uses the uploaded token-embedding matrix. When it is
            // separate, the adapter uploads the quantized output.weight matrix
            // (rows == vocabSize, cols == hiddenSize) into a distinct buffer.
            const void *lmHeadQ = nullptr;
            uint32_t lmHeadType = emb.type;
            uint32_t lmHeadRowBytes = embRowBytes;
            if (!m.lmHeadTied_ && !m.lmHead_.empty()) {
                lmHeadQ = m.lmHead_.data.data();
                lmHeadType = m.lmHead_.type;
                lmHeadRowBytes = typeBlockBytes(m.lmHead_.type) *
                                 ((m.lmHead_.cols + 255) / 256);
            }
            geom.lmHeadTied = m.lmHeadTied_ ? 1u : 0u;

            if (!gpu_.upload(deviceLayers, geom, emb.data.data(), emb.type,
                             embRowBytes, lmHeadQ, lmHeadType, lmHeadRowBytes,
                             err)) {
                destroyF16Hosts(deviceLayers);
                return false;
            }
            // Keep FP16 twins alive until destructor (upload() already copied
            // them to device, but the DeviceLayer descriptors in gpu_ hold device
            // pointers now; the host twins are only needed for the transfer, and
            // they can be freed after upload because upload() copied the bytes).
            f16Hosts_.swap(deviceLayers);
            uploaded_ = true;
            return true;
        }

        bool forward(Model &m, const std::vector<int32_t> &tokens,
                     bool computeAllLogits, float *logitsOut, std::string &err) {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!uploaded_) {
                if (!ensureUploadedLocked(m, err)) return false;
            }
            return gpu_.forward(tokens, computeAllLogits, logitsOut, err);
        }

        void clearKVCache() {
            std::lock_guard<std::mutex> lk(mtx_);
            if (uploaded_) gpu_.clearKVCache();
        }

        // Public entry for explicit pre-upload (Model::gpuUploadIfEnabled).
        bool uploadIfEnabled(Model &m, std::string &err) {
            std::lock_guard<std::mutex> lk(mtx_);
            return ensureUploadedLocked(m, err);
        }

        void destroy() {
            std::lock_guard<std::mutex> lk(mtx_);
            if (uploaded_) {
                gpu_.~GPUModel();
                new (&gpu_) GPUModel();
                uploaded_ = false;
            }
            destroyF16Hosts(f16Hosts_);
            f16Hosts_.clear();
        }

    private:
        void destroyF16Hosts(std::vector<DeviceLayer> &v) {
            for (auto &dl: v) {
                auto del = [](DeviceMatrix &dm) {
                    delete[] dm.f16Host;
                    dm.f16Host = nullptr;
                };
                del(dl.attnQ);
                del(dl.attnK);
                del(dl.attnV);
                del(dl.attnO);
                del(dl.ffnGate);
                del(dl.ffnUp);
                del(dl.ffnDown);
            }
        }

        std::mutex mtx_;
        bool uploaded_ = false;
        gpu::GPUModel gpu_;
        std::vector<DeviceLayer> f16Hosts_;
    };

    // ---- Model::GPU integration hooks (implemented here) ----

    // Opaque ModelGPUState definition (declared in Model.hpp).
    struct Model::ModelGPUState {
        ModelGPUAdapter *adapter = nullptr;
        ~ModelGPUState() {
            delete adapter;
            adapter = nullptr;
        }
    };

    // (ModelGPUState's own destructor is defined in class above.)

    bool Model::gpuForward(const std::vector<int32_t> &tokens, bool computeAllLogits,
                           float *logitsOut, std::string *errMsg) {
        if (!gpuState_) gpuState_ = new ModelGPUState();
        if (!gpuState_->adapter) gpuState_->adapter = new ModelGPUAdapter();
        std::string err;
        bool ok = gpuState_->adapter->forward(*this, tokens, computeAllLogits,
                                              logitsOut, err);
        if (errMsg && !err.empty()) *errMsg = std::move(err);
        return ok;
    }

    bool Model::gpuUploadIfEnabled(std::string *errMsg) {
        if (!gpu::gpuEnabled()) return false;
        if (!gpuState_) gpuState_ = new ModelGPUState();
        if (!gpuState_->adapter) gpuState_->adapter = new ModelGPUAdapter();
        std::string err;
        bool ok = gpuState_->adapter->uploadIfEnabled(*this, err);
        if (errMsg && !err.empty()) *errMsg = std::move(err);
        return ok;
    }

    void Model::gpuClearKV() {
        if (gpuState_ && gpuState_->adapter) {
            gpuState_->adapter->clearKVCache();
        }
    }

    void Model::gpuShutdown() {
        if (gpuState_) {
            delete gpuState_;
            gpuState_ = nullptr;
        }
    }

}// namespace tinycoder

#endif// USE_CUDA