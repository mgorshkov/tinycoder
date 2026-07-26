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

    /// @brief Multi-architecture tokenizer supporting tiktoken-compatible BPE
    /// and SentencePiece-style tokenizers.
    ///
    /// Supports:
    /// - Qwen2/Qwen35MoE: tiktoken-style BPE with token scores for greedy merging
    /// - Gemma4: SentencePiece-style BPE with explicit merge strings
    ///
    /// Special tokens and BOS/EOS IDs are loaded from GGUF metadata and are
    /// architecture-aware.
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

        /// @brief Configure tokenizer for a specific model architecture.
        /// Sets up special tokens, pre-tokenization regex, and BOS/EOS IDs
        /// based on the architecture.
        /// @param architecture Model architecture string (e.g., "qwen2", "gemma4", "qwen35moe")
        void configureForArchitecture(const std::string &architecture);

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
        /// Uses the EOS token ID loaded from GGUF metadata.
        bool isEogToken(int32_t token) const {
            return token == eosTokenId_;
        }

        // Special token IDs (loaded from GGUF metadata or configured per architecture)
        int32_t bosTokenId() const { return bosTokenId_; }
        int32_t eosTokenId() const { return eosTokenId_; }
        int32_t padTokenId() const { return padTokenId_; }

        // Qwen2-specific special token IDs (for chat template rendering)
        int32_t imStartId() const { return imStartId_; }
        int32_t imEndId() const { return imEndId_; }

    private:
        // Vocabulary: token ID -> token string
        std::unordered_map<int32_t, std::string> vocab_;
        // Reverse vocabulary: token string -> token ID
        std::unordered_map<std::string, int32_t> reverseVocab_;
        // Token scores (log probabilities) for tiktoken-style greedy merging
        std::unordered_map<int32_t, float> scores_;

        // Special tokens set
        std::unordered_set<int32_t> specialTokens_;

        // Special token text -> ID mappings for encode()
        // These are populated by configureForArchitecture()
        std::vector<std::pair<std::string, int32_t>> specialTokenTexts_;

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

        // Special token IDs (loaded from GGUF metadata)
        int32_t bosTokenId_ = 151643;// Default: Qwen2 <|endoftext|>
        int32_t eosTokenId_ = 151643;// Default: Qwen2 <|endoftext|>
        int32_t padTokenId_ = 151643;// Default: Qwen2 <|endoftext|>
        int32_t imStartId_ = 151644; // Default: Qwen2 <|im_start|>
        int32_t imEndId_ = 151645;   // Default: Qwen2 <|im_end|>

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
