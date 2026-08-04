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

#include "LMHead.hpp"
#include "Model.hpp"
#include "ModelInternal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace tinycoder {

    using namespace detail;

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
            // Plan §3: prefer the compact Q6_K raw kernel when the separate LM
            // head is stored as Q6_K (0.82 B/elem vs 1.14 B/elem for the Q8_K
            // copy below — ~28% less per-token LM-head memory traffic, matching
            // llama.cpp's working set). Falls back to the pre-quantized Q8_K
            // copy (int8 _mm256_maddubs_epi16 kernels with the hidden vector
            // quantized once per token, ~2x vs on-the-fly dequantization).
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
            // Use pre-dequantized embeddings for much faster computation.
            // Prefer the FP16 copy (halves memory bandwidth for the LM head mat-vec).
            if (!dequantizedEmbeddings_.dataF16.empty()) {
                LMHead::computeCPUF16(hidden.data(),
                                      dequantizedEmbeddings_.dataF16.data(),
                                      dequantizedEmbeddings_.vocabSize,
                                      dequantizedEmbeddings_.hiddenSize, logits.data());
            } else if (!dequantizedEmbeddings_.empty()) {
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
            softcapInPlace(logits.data(), vocabSize, config_.finalLogitSoftcapping);
        }

        return {std::move(finalHidden), std::move(logits)};
    }

}// namespace tinycoder