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

#include "ChatTemplateRenderer.hpp"
#include "Model.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace tinycoder {

    std::string Model::formatChat(
            const std::vector<std::pair<std::string, std::string>> &messages,
            bool addGenerationPrompt) const {
        // llama.cpp compatibility: `llama-completion -p <pre-rendered chat string>`
        // evaluates the string as a user turn wrapped as
        //   "<|im_start|>user\n$" <pre-rendered> "<|im_end|>\n<|im_start|>assistant\n"
        // (the "$" is llama-completion's "user-specified prompt will pre-start the
        // conversation" sentinel). This adds exactly 9 tokens to the rendered prompt:
        //   prefix: 151644 872 198 3   (<|im_start|> user \n $)
        //   suffix: 151645 198 151644 77091 198 (<|im_end|> \n <|im_start|> assistant \n)
        // so that our prefill token counts equal llama.cpp's evaluated stream
        // token-for-token (Q1..Q4 = 43/40/42/48).
        auto applyLlamaWrap = [&](std::string s) -> std::string {
            if (addGenerationPrompt &&
                (config_.architecture == ARCH_QWEN2 || config_.architecture == ARCH_QWEN35MOE)) {
                s = "<|im_start|>user\n$" + s + "<|im_end|>\n<|im_start|>assistant\n";
            }
            return s;
        };

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
                return applyLlamaWrap(result);
            }
        }

        // Use the dedicated Jinja template renderer
        std::string rendered = ChatTemplateRenderer::render(config_.chatTemplate, messages, addGenerationPrompt);
        rendered = applyLlamaWrap(rendered);
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

        // Tokenize prompt
        auto tokens = tokenize(prompt);
        if (tokens.empty()) {
            return {};
        }


        if (tokens.size() > 10)
            std::cout << "...";
        std::cout << std::endl;

        // Prefill (process all prompt tokens at once). Only the last token's
        // logits are needed for sampling, so skip the expensive LM head for all
        // but the last token (the LM head reads the full vocabSize x hiddenSize
        // embedding matrix per token).
        auto tPrefill0 = std::chrono::high_resolution_clock::now();
        auto logits = forward(tokens, false);
        auto tPrefill1 = std::chrono::high_resolution_clock::now();
        auto prefillMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(tPrefill1 - tPrefill0).count();

        // Get the last token's logits.
        // Shape normalization: the CPU engine returns the full [seqLen, vocabSize]
        // array (only the last row is populated when computeAllLogits==false, the
        // rest is uninitialized padding), but the GPU offload engine returns a
        // COMPACT [1, vocabSize] array holding ONLY the last token's row.  Both
        // carry the same vocabSize stride, so always copy from the semantic last
        // row: offset = (size - vocabSize) when the array holds every token row,
        // or 0 when it is already the single-row form.  Indexing lastIdx*vocabSize
        // unconditionally reads past the compact GPU buffer (out-of-bounds host
        // access -> segmentation fault).
        const float *lastRow = logits.size() <= config_.vocabSize
                                       ? logits.data()
                                       : logits.data() +
                                                 (logits.size() - config_.vocabSize);
        np::Array<float> lastLogits = np::Array<float>(np::Shape{config_.vocabSize});
        // memcpy instead of per-element virtual get()/set() (see §4.4).
        std::memcpy(lastLogits.data(), lastRow,
                    static_cast<size_t>(config_.vocabSize) * sizeof(float));

        // ---- llama.cpp parity: the repeat-penalty history is seeded from the
        // prompt stream. llama.cpp's "last tokens" ring buffer contains the full
        // prompt, so the FIRST sampled token is already subject to the repeat
        // penalty for prompt tokens in the trailing window (repeatLastN). Without
        // this, a prompt-context token such as <|im_start|> (id 151644) that
        // dominates the post-prompt distribution is sampled unopposed and starts
        // a degenerate repetition loop (Q1/Q4 failures vs llama.cpp).
        std::vector<int32_t> penaltyHistory;
        penaltyHistory.reserve(tokens.size() + params.maxTokens);
        penaltyHistory.insert(penaltyHistory.end(), tokens.begin(), tokens.end());

        auto applyRepeatPenalty = [&](np::Array<float> &logitsArr) {
            if (params.repeatPenalty == 1.0f) {
                return;
            }
            float *logitsPtr = logitsArr.data();
            int32_t startPenalty = std::max(
                    0, static_cast<int32_t>(penaltyHistory.size()) - params.repeatLastN);
            for (int32_t p = startPenalty; p < static_cast<int32_t>(penaltyHistory.size());
                 ++p) {
                int32_t prevToken = penaltyHistory[p];
                float &logit = logitsPtr[prevToken];
                if (logit < 0) {
                    logit = logit * params.repeatPenalty;
                } else {
                    logit = logit / params.repeatPenalty;
                }
            }
        };

        // Sample first generated token (repeat penalty already applied over the
        // prompt tail, matching llama.cpp's last-tokens window semantics).
        applyRepeatPenalty(lastLogits);
        int32_t nextToken = sampleToken(lastLogits, params);
        std::string tokenText = tokenizer_.decodeToken(nextToken);
        penaltyHistory.push_back(nextToken);

        std::vector<int32_t> generated;
        generated.push_back(nextToken);

        if (!callback(nextToken, tokenText)) {
            return generated;
        }

        // Generate remaining tokens
        auto tGen0 = std::chrono::high_resolution_clock::now();
        // P0 (BENCHMARK_REPORT §5/P0): exact two-pass LM-head top-K pruning.
        // The repeat-penalty history (same window as the repeat-penalty filter
        // below) must always be candidates for exact logits so that pruning never
        // changes the post-penalty sampling distribution for penalized tokens.
        // Pruning only applies to the separate (non-tied) LM head single-token
        // path and only when topK is finite (>= 1).
        std::vector<int32_t> forceInclude;
        for (int32_t i = 1; i < params.maxTokens; ++i) {
            // Forward pass for single token (P0 top-K pruning of the LM head).
            // forceInclude = repeat-penalty history tokens (the prompt-seeded
            // history, matching llama.cpp's last-tokens window; may be empty).
            forceInclude.clear();
            if (params.repeatPenalty != 1.0f && !penaltyHistory.empty()) {
                int32_t startPenalty = std::max(
                        0, static_cast<int32_t>(penaltyHistory.size()) - params.repeatLastN);
                for (int32_t p = startPenalty; p < static_cast<int32_t>(penaltyHistory.size());
                     ++p) {
                    forceInclude.push_back(penaltyHistory[p]);
                }
            }
            np::Array<float> newLogits;
            if (params.topK > 0) {
                newLogits = forward({nextToken}, false, params.topK, &forceInclude);
            } else {
                newLogits = forward({nextToken});
            }

            // Get logits (memcpy instead of per-element virtual get()/set(), see §4.4)
            std::memcpy(lastLogits.data(), newLogits.data(),
                        static_cast<size_t>(config_.vocabSize) * sizeof(float));

            // Apply repetition penalty over the prompt-seeded history window
            // using raw pointer arithmetic instead of per element virtual
            // get()/set() dispatch (see §4.4 / §9.1).
            applyRepeatPenalty(lastLogits);

            nextToken = sampleToken(lastLogits, params);

            // Check for end of generation (multiple EOG tokens for Qwen2.5-Coder)
            if (tokenizer_.isEogToken(nextToken)) {
                break;
            }

            tokenText = tokenizer_.decodeToken(nextToken);
            generated.push_back(nextToken);
            penaltyHistory.push_back(nextToken);

            if (!callback(nextToken, tokenText)) {
                break;
            }
        }
        auto tGen1 = std::chrono::high_resolution_clock::now();
        auto genMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(tGen1 - tGen0).count();

        std::cout << "[TinyCoder] Prefill: " << prefillMs << " ms (" << tokens.size()
                  << " prompt tokens, " << (tokens.size() / (prefillMs / 1000.0f))
                  << " tok/s)" << std::endl;
        std::cout << "[TinyCoder] Generation: " << genMs << " ms (" << generated.size()
                  << " tokens, " << (generated.size() / (genMs / 1000.0f)) << " tok/s)"
                  << std::endl;

        return generated;
    }

}// namespace tinycoder
