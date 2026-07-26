#pragma once

#include <string>
#include <utility>
#include <vector>

namespace tinycoder {

    /// @brief Simple Jinja template renderer for LLM chat templates.
    ///
    /// Supports the subset of Jinja syntax commonly found in GGUF
    /// `tokenizer.chat_template` metadata for Qwen2, Gemma4, and similar
    /// architectures.
    ///
    /// Supported features:
    ///   - {{ expr }} and {{- expr -}} (with whitespace control)
    ///   - {% tag %} and {%- tag -%} (with whitespace control)
    ///   - {% for X in messages %} ... {% endfor %}
    ///   - {% if condition %} ... {% elif %} ... {% else %} ... {% endif %}
    ///   - {% set var = value %} (ignored)
    ///   - String literals: 'text' and "text"
    ///   - String concatenation with +
    ///   - Escaped sequences: \n, \t, \", \'
    ///   - Dot notation: message.role, message.content, loop.index0
    ///   - Bracket notation: message['role'], messages[0]['role']
    ///   - Filters: expr | tojson (basic support)
    ///   - Tests: expr is defined
    ///   - Variables: add_generation_prompt, bos_token, eos_token,
    ///                loop.first, loop.last, loop.index0, tools
    class ChatTemplateRenderer {
    public:
        /// @brief Render a Jinja chat template with the given messages.
        /// @param tmpl The Jinja template string (from tokenizer.chat_template).
        /// @param messages Vector of (role, content) pairs.
        /// @param addGenerationPrompt If true, append the assistant turn start.
        /// @return The rendered prompt string.
        static std::string render(
                const std::string &tmpl,
                const std::vector<std::pair<std::string, std::string>> &messages,
                bool addGenerationPrompt);

    private:
        // ---- Helper types ----

        /// @brief Evaluation context passed through recursive calls.
        struct EvalCtx {
            const std::vector<std::pair<std::string, std::string>> &messages;
            bool addGenerationPrompt;
            size_t msgIdx;// current message index in loop
            bool isFirst; // true if msgIdx == 0
            bool isLast;  // true if msgIdx == messages.size() - 1
        };

        // ---- Expression evaluation ----

        /// @brief Evaluate a simple expression (no + concatenation).
        /// Returns the string value. For boolean contexts, returns "true"/"false".
        static std::string evalExpr(const std::string &expr, const EvalCtx &ctx);

        /// @brief Evaluate a concatenation expression (parts joined by +).
        static std::string evalConcat(const std::string &expr, const EvalCtx &ctx);

        /// @brief Evaluate a condition (for {% if %}).
        static bool evalCondition(const std::string &cond, const EvalCtx &ctx);

        // ---- Template parsing ----

        /// @brief Find matching end tag (endfor/endif), handling nesting.
        /// @return Position right after the closing "%}".
        static size_t findMatchingEnd(const std::string &tmpl, size_t startPos,
                                      const std::string &openPrefix,
                                      const std::string &closeTag);

        /// @brief Find matching endif and optional else/elif.
        /// @return Pair of (endif_pos, else_pos).
        static std::pair<size_t, size_t> findMatchingEndif(
                const std::string &tmpl, size_t startPos);

        /// @brief Process a template block recursively.
        static void processBlock(const std::string &tmpl, size_t &pos,
                                 std::string &out, const EvalCtx &ctx);

        // ---- String utilities ----

        /// @brief Trim whitespace and newlines from both ends.
        static void trim(std::string &s);

        /// @brief Unescape common escape sequences.
        static std::string unescape(const std::string &s);

        /// @brief Check if a string is a quoted literal.
        static bool isStringLiteral(const std::string &s);

        /// @brief Extract content from a string literal (strip quotes, unescape).
        static std::string extractStringLiteral(const std::string &s);

        /// @brief Strip trailing whitespace/newlines.
        static void stripTrailingWhitespace(std::string &s);

        /// @brief Strip leading whitespace/newlines.
        static void stripLeadingWhitespace(std::string &s);

        /// @brief Strip whitespace control hyphens from a Jinja tag/expr.
        /// @return (cleaned, stripLeft, stripRight)
        static std::tuple<std::string, bool, bool> stripWhitespaceControl(
                const std::string &tag);
    };

}// namespace tinycoder