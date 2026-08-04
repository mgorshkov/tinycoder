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

#include "GGMLDequantize.hpp"
#include "LMHead.hpp"
#include "Model.hpp"
#include "ModelInternal.hpp"
#include "SIMDMatMulVec.hpp"
#include "ThreadPool.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

namespace tinycoder {

    using namespace detail;

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

                // Apply RoPE (P3: K rotation fused into the store below).
                applyRoPE(qData, kData, seqLen, seqLen, nHeads, nKVHeads,
                          static_cast<uint32_t>(kvCache_.pos), /*rotateK=*/false);

                // Store K, V in cache (K-RoPE fused into the K write, P3).
                uint32_t cachePos = static_cast<uint32_t>(kvCache_.pos);
                float *kCacheLayer =
                        kvCache_.k.data() + layer * maxSeqLen * nKVHeads * headDim;
                float *vCacheLayer =
                        kvCache_.v.data() + layer * maxSeqLen * nKVHeads * headDim;

                storeKVWithRoPE(kData, vData, kCacheLayer, vCacheLayer, seqLen,
                                cachePos, nKVHeads);

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

                // Apply RoPE (P3: K rotation fused into the store below).
                applyRoPE(qData, kData, seqLen, seqLen, nHeads, nKVHeads,
                          static_cast<uint32_t>(kvCache_.pos), /*rotateK=*/false);


                // Store K, V in cache (K-RoPE fused into the K write, P3).
                uint32_t cachePos = static_cast<uint32_t>(kvCache_.pos);
                float *kCacheLayer =
                        kvCache_.k.data() + layer * maxSeqLen * nKVHeads * headDim;
                float *vCacheLayer =
                        kvCache_.v.data() + layer * maxSeqLen * nKVHeads * headDim;

                storeKVWithRoPE(kData, vData, kCacheLayer, vCacheLayer, seqLen,
                                cachePos, nKVHeads);

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

                // Apply RoPE (P3: K rotation fused into the store below).
                applyRoPE(qData, kData, seqLen, seqLen, nHeads, nKVHeads,
                          static_cast<uint32_t>(kvCache_.pos), /*rotateK=*/false);


                // Store K, V in cache (K-RoPE fused into the K write, P3).
                uint32_t cachePos = static_cast<uint32_t>(kvCache_.pos);
                float *kCacheLayer =
                        kvCache_.k.data() + layer * maxSeqLen * nKVHeads * headDim;
                float *vCacheLayer =
                        kvCache_.v.data() + layer * maxSeqLen * nKVHeads * headDim;

                storeKVWithRoPE(kData, vData, kCacheLayer, vCacheLayer, seqLen,
                                cachePos, nKVHeads);

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
            // Plan §3: prefer the compact Q6_K raw kernel when the separate LM
            // head is stored as Q6_K (0.82 B/elem vs 1.14 B/elem for the Q8_K
            // copy below — ~28% less per-token LM-head memory traffic). Falls
            // back to the pre-quantized Q8_K copy.
            if (lmHead_.type == GGML_TYPE_Q6_K && lmHead_.cols % 256 == 0) {
                matMulVecBatchQ6K_SIMD(lmHead_.data.data(), hidden.data(), 1,
                                       lmHead_.rows, lmHead_.cols, logits.data());
            } else if (!lmHeadQ8K_.empty()) {
                LMHead::matMulVecQ8K_Single(hidden.data(), lmHeadQ8K_.data(),
                                            lmHead_.rows, lmHead_.cols, logits.data());
            } else {
                LMHead::computeCPUSeparate(hidden.data(), lmHead_.data.data(), lmHead_.type,
                                           lmHead_.rows, lmHead_.cols, logits.data());
            }
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
}// namespace tinycoder
