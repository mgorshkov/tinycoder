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
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tinycoder {

    /// @brief Qwen2.5-Coder tokenizer (tiktoken-compatible BPE encoding).
    ///
    /// Supports the Qwen2.5-Coder tokenizer which uses:
    /// - Byte-level BPE (tiktoken style, no explicit merge strings)
    /// - Token scores for greedy merging (highest score = merge first)
    /// - Special tokens: <|im_start|>, <|im_end|>, <|fim_prefix|>,
    ///   <|fim_middle|>, <|fim_suffix|>
    /// - Regex-based pre-tokenization (GPT-2 pattern)
    class Tokenizer {
    public:
        Tokenizer() = default;
        ~Tokenizer() = default;

        /// @brief Load tokenizer from a GGUF file (embedded tokenizer data).
        /// @param ggufPath Path to the GGUF file containing tokenizer data
        /// @return true if loading succeeded
        bool loadFromGGUF(const std::string &ggufPath);

        /// @brief Load tokenizer from separate files.
        /// @param vocabPath Path to vocab file (tiktoken .tiktoken or BPE .model)
        /// @return true if loading succeeded
        bool load(const std::string &vocabPath);

        /// @brief Encode text to token IDs.
        /// @param text Input text
        /// @return Vector of token IDs
        std::vector<int32_t> encode(const std::string &text) const;

        /// @brief Decode token IDs back to text.
        /// @param tokens Token IDs
        /// @return Decoded text
        std::string decode(const std::vector<int32_t> &tokens) const;

        /// @brief Decode a single token ID.
        /// @param token Token ID
        /// @return Decoded text for this token
        std::string decodeToken(int32_t token) const;

        /// @brief Get the vocabulary size.
        size_t vocabSize() const { return vocab_.size(); }

        /// @brief Check if a token ID is a special token.
        bool isSpecialToken(int32_t token) const;

        /// @brief Check if a token ID is an end-of-generation token.
        /// For generation, only <|endoftext|> (151643) stops generation.
        /// <|im_end|> (151645) is part of normal chat output and should not stop.
        bool isEogToken(int32_t token) const {
            return token == 151643;
        }

        // Special token IDs for Qwen2.5-Coder
        int32_t bosTokenId() const { return 151643; }// <|endoftext|>
        int32_t eosTokenId() const { return 151643; }// <|endoftext|>
        int32_t padTokenId() const { return 151643; }
        int32_t imStartId() const { return 151644; }  // <|im_start|>
        int32_t imEndId() const { return 151645; }    // <|im_end|>
        int32_t fimPrefixId() const { return 151659; }// <|fim_prefix|>
        int32_t fimMiddleId() const { return 151660; }// <|fim_middle|>
        int32_t fimSuffixId() const { return 151661; }// <|fim_suffix|>

    private:
        // Vocabulary: token ID -> token string
        std::unordered_map<int32_t, std::string> vocab_;
        // Reverse vocabulary: token string -> token ID
        std::unordered_map<std::string, int32_t> reverseVocab_;
        // Token scores (log probabilities) for tiktoken-style greedy merging
        std::unordered_map<int32_t, float> scores_;

        // Special tokens set
        std::unordered_set<int32_t> specialTokens_;

        // Regex pattern for pre-tokenization (GPT-2 pattern)
        std::string pretokenizeRegex_;

        // Cache for BPE encodings
        mutable std::unordered_map<std::string, std::vector<int32_t>> bpeCache_;

        // Byte-to-token-ID mapping for tiktoken-style byte encoding.
        // In tiktoken, each byte (0-255) is mapped to a Unicode character via
        // bytes_to_unicode(), and the vocabulary stores those Unicode characters
        // as UTF-8 strings. This array maps raw byte values to their
        // corresponding token IDs for the initial BPE symbol sequence.
        // Initialized during loadFromGGUF / load.
        int32_t byteToTokenId_[256] = {};

        // Token-ID-to-byte mapping for decoding.
        // Maps token IDs that represent single bytes back to their raw byte value.
        // Initialized during loadFromGGUF / load.
        std::unordered_map<int32_t, uint8_t> tokenIdToByte_;

        /// @brief Get the string representation of a token, falling back to
        ///        byte-level decoding for tokens 0-255.
        std::string getTokenString(int32_t id) const;

        std::vector<int32_t> bpeEncode(const std::string &token) const;
        std::vector<std::string> pretokenize(const std::string &text) const;

        /// @brief Build the byte-to-token-ID mapping from the vocabulary.
        ///        In tiktoken, bytes are mapped to Unicode characters via
        ///        bytes_to_unicode(), and the vocabulary stores those characters.
        ///        This function finds the token ID for each byte by looking up
        ///        the corresponding Unicode character in the reverse vocabulary.
        void buildByteToTokenId();
    };

}// namespace tinycoder
