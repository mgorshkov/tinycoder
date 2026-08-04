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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

namespace tinycoder {

    void Model::computeGemma4MoE(const float *ffnNorm, float *ffnOut,
                                 uint32_t seqLen, uint32_t hiddenSize,
                                 uint32_t intermediateSize,
                                 const LayerWeights &w) const {
        (void) intermediateSize;
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
        (void) intermediateSize;
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

}// namespace tinycoder