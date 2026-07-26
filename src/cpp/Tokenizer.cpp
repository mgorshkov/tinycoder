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

#include "Tokenizer.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>

namespace tinycoder {

    // Qwen2 pre-tokenization regex pattern
    // Qwen2 uses a modified GPT-2 pattern where underscores are kept
    // attached to the following word (e.g., "_start" stays as one token).
    // Uses POSIX character classes [[:alpha:]] and [[:digit:]] instead of
    // Unicode \p{L} and \p{N} because GCC's std::regex doesn't support
    // Unicode character classes.
    // The key difference from GPT-2: underscore (_) is treated as part of
    // a word (included in the alpha class via [[:alpha:]_]).
    static const std::string QWEN2_PATTERN =
            R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]_]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]_]+|\s+(?!\S)|\s+)";

    // Gemma4 pre-tokenization regex pattern (SentencePiece-style)
    // Gemma4 uses a SentencePiece tokenizer with the standard unigram pattern.
    // The pattern is similar to GPT-2 but without the underscore-in-word behavior.
    static const std::string GEMMA4_PATTERN =
            R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";

    void Tokenizer::configureForArchitecture(const std::string &architecture) {
        if (architecture == "gemma4") {
            // Gemma4 uses SentencePiece tokenizer
            // BOS=2 (<bos>), EOS=1 (<eos>), PAD=0 (<pad>)
            bosTokenId_ = 2;
            eosTokenId_ = 1;
            padTokenId_ = 0;
            imStartId_ = -1;// Not used for Gemma4
            imEndId_ = -1;  // Not used for Gemma4
            pretokenizeRegex_ = GEMMA4_PATTERN;

            // Gemma4 special tokens for encode()
            specialTokenTexts_ = {
                    {"<bos>", 2},
                    {"<eos>", 1},
                    {"<pad>", 0},
                    {"<unk>", 3},
            };
        } else if (architecture == "qwen35moe") {
            // Qwen35MoE uses tiktoken-style BPE (same as Qwen2)
            // BOS/EOS=27 (<|endoftext|> equivalent)
            bosTokenId_ = 27;
            eosTokenId_ = 27;
            padTokenId_ = 27;
            imStartId_ = 151644;// <|im_start|>
            imEndId_ = 151645;  // <|im_end|>
            pretokenizeRegex_ = QWEN2_PATTERN;

            // Qwen35MoE special tokens for encode()
            specialTokenTexts_ = {
                    {"<|endoftext|>", 27},
                    {"<|im_start|>", 151644},
                    {"<|im_end|>", 151645},
            };
        } else {
            // Qwen2 default (tiktoken-style BPE)
            bosTokenId_ = 151643;
            eosTokenId_ = 151643;
            padTokenId_ = 151643;
            imStartId_ = 151644;
            imEndId_ = 151645;
            pretokenizeRegex_ = QWEN2_PATTERN;

            // Qwen2 special tokens for encode()
            specialTokenTexts_ = {
                    {"<|endoftext|>", 151643},
                    {"<|im_start|>", 151644},
                    {"<|im_end|>", 151645},
                    {"<|fim_prefix|>", 151659},
                    {"<|fim_middle|>", 151660},
                    {"<|fim_suffix|>", 151661},
                    {"<|repo_name|>", 151662},
                    {"<|file_sep|>", 151663},
                    {"<|im_extra_id_0|>", 151664},
            };
        }

        // Update special tokens set
        specialTokens_.clear();
        specialTokens_.insert(bosTokenId_);
        specialTokens_.insert(eosTokenId_);
        specialTokens_.insert(padTokenId_);
        if (imStartId_ >= 0) specialTokens_.insert(imStartId_);
        if (imEndId_ >= 0) specialTokens_.insert(imEndId_);
    }

    bool Tokenizer::loadFromGGUF(const std::string &ggufPath) {
        // Load tokenizer data embedded in the GGUF file.
        // Supports both tiktoken-style (Qwen2, Qwen35MoE) and
        // SentencePiece-style (Gemma4) tokenizers.
        //
        // Tiktoken-style stores:
        // - tokenizer.model: "gpt2"
        // - tokenizer.tokens: array of token strings
        // - tokenizer.scores: array of token scores (log probabilities)
        // - tokenizer.merges: NOT present in tiktoken-style tokenizers
        //
        // SentencePiece-style stores:
        // - tokenizer.model: "gemma4" or "sentencepiece"
        // - tokenizer.tokens: array of token strings
        // - tokenizer.scores: array of token scores
        // - tokenizer.merges: array of merge strings
        //
        // Tiktoken does NOT use explicit merge strings. Instead, it uses
        // token scores for greedy BPE merging: adjacent tokens whose
        // concatenation exists in the vocabulary are merged in order of
        // highest score.

        std::ifstream file(ggufPath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[TinyCoder] Failed to open GGUF for tokenizer: " << ggufPath
                      << std::endl;
            return false;
        }

        // Read GGUF header to find metadata
        uint32_t magic;
        file.read(reinterpret_cast<char *>(&magic), sizeof(uint32_t));

        uint32_t version;
        file.read(reinterpret_cast<char *>(&version), sizeof(uint32_t));

        uint64_t tensorCount;
        file.read(reinterpret_cast<char *>(&tensorCount), sizeof(uint64_t));

        uint64_t metadataKVCount;
        file.read(reinterpret_cast<char *>(&metadataKVCount), sizeof(uint64_t));

        // Scan metadata for tokenizer data
        bool foundTokens = false;
        bool foundScores = false;
        bool foundMerges = false;
        std::vector<std::string> tokenStrings;
        std::vector<float> tokenScores;
        std::vector<std::string> mergeStrings;

        for (uint64_t i = 0; i < metadataKVCount; ++i) {
            // Read key
            uint64_t keyLen;
            file.read(reinterpret_cast<char *>(&keyLen), sizeof(uint64_t));
            std::string key(keyLen, '\0');
            file.read(key.data(), keyLen);

            // Read value type
            uint32_t valueType;
            file.read(reinterpret_cast<char *>(&valueType), sizeof(uint32_t));

            if (key == "tokenizer.ggml.tokens") {
                // Array of strings
                uint32_t arrayType;
                file.read(reinterpret_cast<char *>(&arrayType), sizeof(uint32_t));
                uint64_t arrayLen;
                file.read(reinterpret_cast<char *>(&arrayLen), sizeof(uint64_t));

                tokenStrings.reserve(arrayLen);
                for (uint64_t j = 0; j < arrayLen; ++j) {
                    uint64_t strLen;
                    file.read(reinterpret_cast<char *>(&strLen), sizeof(uint64_t));
                    std::string token(strLen, '\0');
                    file.read(token.data(), strLen);
                    tokenStrings.push_back(token);
                }
                foundTokens = true;
            } else if (key == "tokenizer.ggml.scores") {
                // Array of floats (log probabilities for tiktoken-style merging)
                uint32_t arrayType;
                file.read(reinterpret_cast<char *>(&arrayType), sizeof(uint32_t));
                uint64_t arrayLen;
                file.read(reinterpret_cast<char *>(&arrayLen), sizeof(uint64_t));

                tokenScores.reserve(arrayLen);
                for (uint64_t j = 0; j < arrayLen; ++j) {
                    float score;
                    file.read(reinterpret_cast<char *>(&score), sizeof(float));
                    tokenScores.push_back(score);
                }
                foundScores = true;
            } else if (key == "tokenizer.ggml.merges") {
                // SentencePiece-style tokenizers use explicit merge strings.
                // Tiktoken-style tokenizers don't use explicit merges.
                uint32_t arrayType;
                file.read(reinterpret_cast<char *>(&arrayType), sizeof(uint32_t));
                uint64_t arrayLen;
                file.read(reinterpret_cast<char *>(&arrayLen), sizeof(uint64_t));

                mergeStrings.reserve(arrayLen);
                for (uint64_t j = 0; j < arrayLen; ++j) {
                    uint64_t strLen;
                    file.read(reinterpret_cast<char *>(&strLen), sizeof(uint64_t));
                    std::string merge(strLen, '\0');
                    file.read(merge.data(), strLen);
                    mergeStrings.push_back(merge);
                }
                foundMerges = true;
            } else if (key == "tokenizer.ggml.model") {
                // String value
                uint64_t strLen;
                file.read(reinterpret_cast<char *>(&strLen), sizeof(uint64_t));
                std::string modelType(strLen, '\0');
                file.read(modelType.data(), strLen);
                // Expected: "gpt2" for BPE tokenizer, "gemma4" or "sentencepiece" for SentencePiece
            } else if (key == "tokenizer.ggml.bos_token_id") {
                // Read BOS token ID (INT32)
                int32_t bosId;
                file.read(reinterpret_cast<char *>(&bosId), sizeof(int32_t));
                bosTokenId_ = bosId;
            } else if (key == "tokenizer.ggml.eos_token_id") {
                // Read EOS token ID (INT32)
                int32_t eosId;
                file.read(reinterpret_cast<char *>(&eosId), sizeof(int32_t));
                eosTokenId_ = eosId;
            } else if (key == "tokenizer.ggml.padding_token_id") {
                // Read PAD token ID (INT32)
                int32_t padId;
                file.read(reinterpret_cast<char *>(&padId), sizeof(int32_t));
                padTokenId_ = padId;
            } else {
                // Skip unknown metadata values
                switch (valueType) {
                    case 0:
                    case 1:
                        file.seekg(1, std::ios::cur);
                        break;// UINT8, INT8
                    case 2:
                    case 3:
                        file.seekg(2, std::ios::cur);
                        break;// UINT16, INT16
                    case 4:
                    case 5:
                        file.seekg(4, std::ios::cur);
                        break;// UINT32, INT32, FLOAT32
                    case 6:
                        file.seekg(4, std::ios::cur);
                        break;// FLOAT32
                    case 7:
                        file.seekg(1, std::ios::cur);
                        break;// BOOL
                    case 8: { // STRING
                        uint64_t strLen;
                        file.read(reinterpret_cast<char *>(&strLen), sizeof(uint64_t));
                        file.seekg(strLen, std::ios::cur);
                        break;
                    }
                    case 9: {// ARRAY
                        uint32_t arrType;
                        file.read(reinterpret_cast<char *>(&arrType), sizeof(uint32_t));
                        uint64_t arrLen;
                        file.read(reinterpret_cast<char *>(&arrLen), sizeof(uint64_t));
                        // Skip elements based on type
                        // GGUF value types: 0=UINT8, 1=INT8, 2=UINT16, 3=INT16,
                        // 4=UINT32, 5=INT32, 6=FLOAT32, 7=BOOL, 8=STRING,
                        // 9=ARRAY, 10=UINT64, 11=INT64, 12=FLOAT64
                        if (arrType == 8) {
                            // String array: each element has a length prefix
                            for (uint64_t j = 0; j < arrLen; ++j) {
                                uint64_t elemLen;
                                file.read(reinterpret_cast<char *>(&elemLen), sizeof(uint64_t));
                                file.seekg(elemLen, std::ios::cur);
                            }
                        } else if (arrType == 0 || arrType == 1 || arrType == 7) {
                            // UINT8, INT8, BOOL: 1 byte each
                            file.seekg(static_cast<std::streamoff>(arrLen), std::ios::cur);
                        } else if (arrType == 2 || arrType == 3) {
                            // UINT16, INT16: 2 bytes each
                            file.seekg(static_cast<std::streamoff>(arrLen * 2), std::ios::cur);
                        } else if (arrType == 4 || arrType == 5 || arrType == 6) {
                            // UINT32, INT32, FLOAT32: 4 bytes each
                            file.seekg(static_cast<std::streamoff>(arrLen * 4), std::ios::cur);
                        } else if (arrType == 10 || arrType == 11 || arrType == 12) {
                            // UINT64, INT64, FLOAT64: 8 bytes each
                            file.seekg(static_cast<std::streamoff>(arrLen * 8), std::ios::cur);
                        } else {
                            // Unknown array type, skip by reading element size
                            // (shouldn't happen with standard GGUF files)
                            for (uint64_t j = 0; j < arrLen; ++j) {
                                uint64_t elemLen;
                                file.read(reinterpret_cast<char *>(&elemLen), sizeof(uint64_t));
                                file.seekg(elemLen, std::ios::cur);
                            }
                        }
                        break;
                    }
                    case 10:
                    case 11:
                        file.seekg(8, std::ios::cur);
                        break;// UINT64, INT64
                    case 12:
                        file.seekg(8, std::ios::cur);
                        break;// FLOAT64
                    default:
                        break;
                }
            }
        }

        file.close();

        if (!foundTokens || tokenStrings.empty()) {
            std::cerr << "[TinyCoder] No tokenizer data found in GGUF file"
                      << std::endl;
            return false;
        }

        // Build vocabulary and scores
        for (size_t i = 0; i < tokenStrings.size(); ++i) {
            int32_t id = static_cast<int32_t>(i);
            vocab_[id] = tokenStrings[i];
            reverseVocab_[tokenStrings[i]] = id;

            if (foundScores && i < tokenScores.size()) {
                scores_[id] = tokenScores[i];
            } else {
                // Default score: byte tokens (0-255) get a neutral score,
                // merged tokens get decreasing scores based on position
                if (id < 256) {
                    scores_[id] = 0.0f;
                } else {
                    // Higher token IDs = later merges = lower priority
                    scores_[id] = -static_cast<float>(id);
                }
            }
        }

        // Build byte-to-token-ID mapping for tiktoken-style byte encoding
        buildByteToTokenId();

        // Set up special tokens
        specialTokens_.insert(bosTokenId_);
        specialTokens_.insert(eosTokenId_);
        specialTokens_.insert(padTokenId_);

        std::cout << "[TinyCoder] Tokenizer loaded: " << vocab_.size() << " tokens, "
                  << scores_.size() << " scores, "
                  << (foundMerges ? mergeStrings.size() : 0) << " merges"
                  << std::endl;

        return true;
    }

    bool Tokenizer::load(const std::string &vocabPath) {
        // Load from a tiktoken .tiktoken file or BPE .model file
        std::ifstream file(vocabPath);
        if (!file.is_open()) {
            std::cerr << "[TinyCoder] Failed to open vocab file: " << vocabPath
                      << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty())
                continue;

            // Format: "token_string id" or "token_string id score"
            auto spacePos = line.find(' ');
            if (spacePos == std::string::npos)
                continue;

            std::string token = line.substr(0, spacePos);
            std::string rest = line.substr(spacePos + 1);

            auto spacePos2 = rest.find(' ');
            int32_t id;
            float score = 0.0f;
            if (spacePos2 != std::string::npos) {
                id = std::stoi(rest.substr(0, spacePos2));
                score = std::stof(rest.substr(spacePos2 + 1));
            } else {
                id = std::stoi(rest);
            }

            vocab_[id] = token;
            reverseVocab_[token] = id;
            scores_[id] = score;
        }

        file.close();

        // Build byte-to-token-ID mapping for tiktoken-style byte encoding
        buildByteToTokenId();

        // Set up special tokens
        specialTokens_.insert(bosTokenId_);
        specialTokens_.insert(eosTokenId_);
        specialTokens_.insert(padTokenId_);
        if (imStartId_ >= 0) specialTokens_.insert(imStartId_);
        if (imEndId_ >= 0) specialTokens_.insert(imEndId_);

        pretokenizeRegex_ = QWEN2_PATTERN;

        std::cout << "[TinyCoder] Tokenizer loaded: " << vocab_.size() << " tokens"
                  << std::endl;

        return true;
    }

    std::string Tokenizer::getTokenString(int32_t id) const {
        // Always return the vocabulary string for BPE merging.
        // The vocabulary stores bytes_to_unicode characters (e.g., U+0120 'Ġ'
        // for byte 0x20 ' '), and BPE merging needs to concatenate these
        // Unicode strings to check if the result exists in the vocabulary.
        auto it = vocab_.find(id);
        if (it != vocab_.end()) {
            return it->second;
        }
        return "";
    }

    void Tokenizer::buildByteToTokenId() {
        // Build the bytes_to_unicode() mapping from tiktoken.
        // This maps each byte (0-255) to a Unicode code point.
        // The mapping is:
        //   bytes 33-126 ('!' to '~') → same code points
        //   bytes 161-172 ('¡' to '¬') → same code points
        //   bytes 174-255 ('®' to 'ÿ') → same code points
        //   bytes 0-32 → code points 256-288
        //   bytes 127-160 → code points 289-322
        //   byte 173 → code point 323

        // First, determine which bytes map to themselves
        bool isGood[256] = {};
        for (int b = 33; b <= 126; b++) isGood[b] = true;
        for (int b = 161; b <= 172; b++) isGood[b] = true;
        for (int b = 174; b <= 255; b++) isGood[b] = true;

        // Build byte-to-unicode code point mapping
        int unicodeCP[256];
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (isGood[b]) {
                unicodeCP[b] = b;
            } else {
                unicodeCP[b] = 256 + n;
                n++;
            }
        }

        // For each byte, convert the Unicode code point to UTF-8 and look it up
        // in the reverse vocabulary to find the corresponding token ID.
        for (int b = 0; b < 256; b++) {
            int cp = unicodeCP[b];
            std::string utf8;
            if (cp < 128) {
                utf8 += static_cast<char>(cp);
            } else if (cp < 2048) {
                utf8 += static_cast<char>(0xC0 | (cp >> 6));
                utf8 += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                utf8 += static_cast<char>(0xE0 | (cp >> 12));
                utf8 += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                utf8 += static_cast<char>(0x80 | (cp & 0x3F));
            }

            auto it = reverseVocab_.find(utf8);
            if (it != reverseVocab_.end()) {
                byteToTokenId_[b] = it->second;
                tokenIdToByte_[it->second] = static_cast<uint8_t>(b);
            } else {
                // Fallback: use raw byte value as token ID
                byteToTokenId_[b] = b;
                tokenIdToByte_[b] = static_cast<uint8_t>(b);
            }
        }
    }

    std::vector<std::string> Tokenizer::pretokenize(const std::string &text) const {
        // GPT-2 style pre-tokenization using the regex pattern.
        // The GPT-2 BPE tokenizer expects words to have a leading space
        // (except the first word in the text). For example, "Hello world"
        // should be pre-tokenized as ["Hello", " world"].
        //
        // The regex pattern matches:
        //   's|'t|'re|'ve|'m|'ll|'d          - contractions
        //   ?\p{L}+                           - words with optional leading space
        //   ?\p{N}+                           - numbers with optional leading space
        //   ?[^\s\p{L}\p{N}]+                 - punctuation with optional leading space
        //   \s+(?!\S)                         - trailing whitespace
        //   \s+                               - whitespace sequences
        //
        // We use std::regex to apply this pattern. This is slower than a manual
        // implementation but guarantees compatibility with the GPT-2 tokenizer.
        std::vector<std::string> tokens;

        if (text.empty()) {
            return tokens;
        }

        try {
            std::regex pattern(pretokenizeRegex_, std::regex::ECMAScript);
            std::sregex_iterator iter(text.begin(), text.end(), pattern);
            std::sregex_iterator end;

            for (; iter != end; ++iter) {
                tokens.push_back(iter->str());
            }
        } catch (const std::regex_error &e) {
            // Fallback to simple split if regex fails
            std::cerr << "[TinyCoder] Regex error in pretokenize: " << e.what()
                      << ", falling back to simple split" << std::endl;
            std::string current;
            for (size_t i = 0; i < text.size(); ++i) {
                char c = text[i];
                if (std::isalnum(static_cast<unsigned char>(c)) || c == '\'') {
                    current += c;
                } else {
                    if (!current.empty()) {
                        tokens.push_back(current);
                        current.clear();
                    }
                    if (!std::isspace(static_cast<unsigned char>(c))) {
                        tokens.push_back(std::string(1, c));
                    }
                }
            }
            if (!current.empty()) {
                tokens.push_back(current);
            }
        }

        return tokens;
    }

    std::vector<int32_t> Tokenizer::bpeEncode(const std::string &token) const {
        // Tiktoken-style BPE encoding using token scores.
        //
        // Unlike traditional BPE which uses explicit merge strings
        // (e.g., "a b" -> "ab"), tiktoken determines merges by checking
        // if the concatenation of two adjacent tokens exists in the
        // vocabulary. If it does, the pair can be merged. The algorithm
        // greedily merges the pair with the highest score at each step.
        //
        // This approach works because tiktoken stores all possible merged
        // tokens directly in the vocabulary, each with a score (log
        // probability). The scores guide the merging order: higher score
        // = more likely to be merged first.

        // Check cache first
        auto cacheIt = bpeCache_.find(token);
        if (cacheIt != bpeCache_.end()) {
            return cacheIt->second;
        }

        // Convert token to byte-level representation using the
        // bytes_to_unicode() mapping from tiktoken.
        // In tiktoken, each byte is mapped to a Unicode character via
        // bytes_to_unicode(), and the vocabulary stores those characters.
        // We use the pre-built byteToTokenId_ array to find the correct
        // token ID for each byte.
        std::vector<int32_t> symbols;
        for (size_t i = 0; i < token.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(token[i]);
            symbols.push_back(byteToTokenId_[c]);
        }

        if (symbols.empty())
            return symbols;

        // Greedy merging using token scores.
        // At each step, find the adjacent pair whose concatenation exists
        // in the vocabulary with the highest score, and merge it.
        while (symbols.size() > 1) {
            float bestScore = -std::numeric_limits<float>::infinity();
            size_t bestPos = 0;
            bool found = false;

            for (size_t i = 0; i < symbols.size() - 1; ++i) {
                std::string merged =
                        getTokenString(symbols[i]) + getTokenString(symbols[i + 1]);
                auto revIt = reverseVocab_.find(merged);
                if (revIt != reverseVocab_.end()) {
                    int32_t mergedId = revIt->second;
                    auto scoreIt = scores_.find(mergedId);
                    float score = (scoreIt != scores_.end())
                                          ? scoreIt->second
                                          : -std::numeric_limits<float>::infinity();
                    if (score > bestScore) {
                        bestScore = score;
                        bestPos = i;
                        found = true;
                    }
                }
            }

            if (!found) {
                break;// No more merges possible
            }

            // Merge the best pair
            std::string mergedStr =
                    getTokenString(symbols[bestPos]) + getTokenString(symbols[bestPos + 1]);
            int32_t mergedId = reverseVocab_.at(mergedStr);

            std::vector<int32_t> newSymbols;
            for (size_t i = 0; i < symbols.size(); ++i) {
                if (i == bestPos) {
                    newSymbols.push_back(mergedId);
                    i++;// Skip the second element of the pair
                } else {
                    newSymbols.push_back(symbols[i]);
                }
            }
            symbols = newSymbols;
        }

        // Cache the result
        bpeCache_[token] = symbols;

        return symbols;
    }

    std::vector<int32_t> Tokenizer::encode(const std::string &text) const {
        // ---- Scan for special tokens before BPE encoding ----
        // Special tokens like <|im_start|> (151644), <|im_end|> (151645),
        // <|endoftext|> (151643) must be emitted as single token IDs rather
        // than being BPE-encoded into sub-token pieces.
        //
        // The pretokenizer regex splits these into many pieces (<, |, im, etc.),
        // which would produce a completely different prompt structure than
        // what the model was trained on.
        //
        // We split the input text on special token boundaries, emit the
        // special token IDs directly, and BPE-encode only the regular text
        // segments in between.

        std::vector<int32_t> result;
        size_t pos = 0;

        while (pos < text.size()) {
            // Look for the nearest special token
            size_t bestMatch = std::string::npos;
            int32_t bestId = 0;
            size_t bestLen = 0;

            for (const auto &st: specialTokenTexts_) {
                size_t found = text.find(st.first, pos);
                if (found == pos) {
                    // Special token starts at current position
                    bestMatch = found;
                    bestId = st.second;
                    bestLen = st.first.size();
                    break;
                }
                if (found != std::string::npos && (bestMatch == std::string::npos || found < bestMatch)) {
                    bestMatch = found;
                    bestId = st.second;
                    bestLen = st.first.size();
                }
            }

            if (bestMatch == pos) {
                // Emit special token
                result.push_back(bestId);
                pos += bestLen;
            } else if (bestMatch != std::string::npos) {
                // Emit text up to the special token
                std::string segment = text.substr(pos, bestMatch - pos);
                auto pretokens = pretokenize(segment);
                for (const auto &pretoken: pretokens) {
                    auto revIt = reverseVocab_.find(pretoken);
                    if (revIt != reverseVocab_.end()) {
                        result.push_back(revIt->second);
                    } else {
                        auto bpeTokens = bpeEncode(pretoken);
                        result.insert(result.end(), bpeTokens.begin(), bpeTokens.end());
                    }
                }
                pos = bestMatch;
            } else {
                // No more special tokens - encode remaining text
                std::string segment = text.substr(pos);
                auto pretokens = pretokenize(segment);
                for (const auto &pretoken: pretokens) {
                    auto revIt = reverseVocab_.find(pretoken);
                    if (revIt != reverseVocab_.end()) {
                        result.push_back(revIt->second);
                    } else {
                        auto bpeTokens = bpeEncode(pretoken);
                        result.insert(result.end(), bpeTokens.begin(), bpeTokens.end());
                    }
                }
                break;
            }
        }

        return result;
    }

    std::string Tokenizer::decode(const std::vector<int32_t> &tokens) const {
        std::string result;
        for (int32_t token: tokens) {
            result += decodeToken(token);
        }
        return result;
    }

    std::string Tokenizer::decodeToken(int32_t token) const {
        // Check if this token ID maps to a single byte via bytes_to_unicode()
        auto byteIt = tokenIdToByte_.find(token);
        if (byteIt != tokenIdToByte_.end()) {
            return std::string(1, static_cast<char>(byteIt->second));
        }

        // Look up in vocabulary
        auto it = vocab_.find(token);
        if (it != vocab_.end()) {
            // Convert bytes_to_unicode characters back to raw bytes.
            // The vocabulary stores Unicode characters (e.g., U+0120 'Ġ' for
            // byte 0x20 ' '), but we need to return the raw bytes.
            const std::string &vocabStr = it->second;
            std::string result;
            for (size_t i = 0; i < vocabStr.size();) {
                unsigned char c = static_cast<unsigned char>(vocabStr[i]);
                // Decode UTF-8 character
                uint32_t cp;
                size_t len;
                if (c < 0x80) {
                    cp = c;
                    len = 1;
                } else if ((c & 0xE0) == 0xC0) {
                    cp = c & 0x1F;
                    len = 2;
                } else if ((c & 0xF0) == 0xE0) {
                    cp = c & 0x0F;
                    len = 3;
                } else {
                    cp = c & 0x07;
                    len = 4;
                }
                for (size_t j = 1; j < len && i + j < vocabStr.size(); ++j) {
                    cp = (cp << 6) | (static_cast<unsigned char>(vocabStr[i + j]) & 0x3F);
                }
                // Check if this code point is in the bytes_to_unicode range
                // (256-323). If so, convert back to the corresponding byte.
                if (cp >= 256 && cp <= 323) {
                    // Build the inverse bytes_to_unicode mapping
                    bool isGood[256] = {};
                    for (int b = 33; b <= 126; b++) isGood[b] = true;
                    for (int b = 161; b <= 172; b++) isGood[b] = true;
                    for (int b = 174; b <= 255; b++) isGood[b] = true;
                    int n = 0;
                    for (int b = 0; b < 256; b++) {
                        if (!isGood[b]) {
                            if (256 + n == static_cast<int>(cp)) {
                                result += static_cast<char>(b);
                                break;
                            }
                            n++;
                        }
                    }
                } else if (cp < 256) {
                    result += static_cast<char>(cp);
                } else {
                    // Not a bytes_to_unicode character, keep as-is
                    result.append(vocabStr.data() + i, len);
                }
                i += len;
            }
            return result;
        }

        return "";
    }

    bool Tokenizer::isSpecialToken(int32_t token) const {
        return specialTokens_.find(token) != specialTokens_.end();
    }

}// namespace tinycoder
