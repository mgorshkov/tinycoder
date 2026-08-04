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

#include "SIMDMatMulVec.hpp"

#include <cstdint>

// ============================================================================
// Internal SIMD kernel declarations.
//
// The AVX2 and AVX-512 implementations are compiled in their own translation
// units (SIMDMatMulVecAVX2.cpp / SIMDMatMulVecAVX512.cpp) so that per-file
// compiler flags (-mavx2 -mfma -mf16c / -mavx512f -mfma) can be applied without
// forcing those ISAs onto the rest of the codebase. Each kernel is declared
// here with external linkage (in the tinycoder::simd namespace) so the runtime
// dispatch in SIMDMatMulVec.cpp can reference them across translation units.
//
// The declarations are guarded by the same preprocessor macros used in the
// implementation files, so they are only visible when the corresponding ISA is
// enabled at compile time.
// ============================================================================

namespace tinycoder::simd {

#if defined(__AVX2__) && defined(__FMA__)

    void accumulateFMA_AVX2(float *local, const float *blockOut, float alpha,
                            uint32_t n);
    float dotProductFMA_AVX2(const float *hidden, const float *blockOut,
                             uint32_t n);
    float dotProductFMA_F16_AVX2(const float *hidden, const uint16_t *W_f16,
                                 uint32_t n);
    void rmsNorm_AVX2(const float *x, float *out, const float *weight,
                      uint32_t n, float eps);
    void softmax_AVX2(float *x, uint32_t n);
    void silu_AVX2(float *x, uint32_t n);
    void swiGLU_AVX2(float *x, const float *y, uint32_t n);
    void add_AVX2(float *x, const float *y, uint32_t n);
    void scale_AVX2(float *x, float alpha, uint32_t n);
    float dotProductQ2_K_AVX2(const uint8_t *blockData, const float *x);
    float dotProductQ2_K_AVX2_PrePacked(const uint8_t *prepackedBlock,
                                        const float *x);
    float dotProductQ2_K_PrePacked_Q8_AVX2(const uint8_t *prepackedBlock,
                                           const Q8KBlock *q8);
    float dotProductQ8K_Q8K_AVX2(const Q8KBlock *x, const Q8KBlock *w);

    // Register-tiled batch GEMM for Q2_K (8 rows at a time)
    // Processes BATCH_SIZE=8 rows in parallel using AVX2 registers.
    // For each tile of 8 rows, accumulates into ymm registers across
    // all block columns before storing.
    void matMulVecBatchQ2_K_PrePacked_Q8_AVX2(
            const uint8_t *prepackedData,
            const float *x,
            uint32_t rows,
            uint32_t cols,
            float *result);

    // Register-tiled batch GEMM for fused Q2_K gate+up (8 rows at a time)
    // Computes gate = x * W_gate^T and up = x * W_up^T in a single pass,
    // reusing the Q8_K x-vector data across all 8 rows AND both matrices.
    void matMulVecBatchGateUpQ2_K_PrePacked_Q8_AVX2(
            const uint8_t *gatePrepacked,
            const uint8_t *upPrepacked,
            const float *x,
            uint32_t rows,
            uint32_t cols,
            float *gateOut,
            float *upOut);

    // Register-tiled batch GEMM for fused Q2_K gate+up over a batch of tokens
    // (prefill). For each tile of 8 rows, the weight blocks are loaded once and
    // reused across all seqLen tokens, eliminating the seqLen-fold re-read of the
    // weight matrix that dominates prefill. Q8_K x-vector data is reused across
    // the 8 rows in the tile.
    //
    // If applySwish is true, the epilogue computes
    //   gateOut[s*rows+j] = silu(gateOut[s*rows+j]) * upOut[s*rows+j]
    // (i.e. fuses the SwiGLU activation into the kernel's store) instead of
    // storing the raw gate result (P5).
    void matMulVecBatchGateUpQ2_K_PrePacked_Q8_Batch_AVX2(
            const uint8_t *gatePrepacked,
            const uint8_t *upPrepacked,
            const float *X,
            uint32_t seqLen,
            uint32_t rows,
            uint32_t cols,
            float *gateOut,
            float *upOut,
            bool applySwish);

    // Register-tiled fused gate+up matvec for COMPACT (raw GGUF) Q2_K weights,
    // single token (generation). Single-token fused gate+up is the dominant
    // per-token cost and is DRAM-bandwidth-bound; the compact 84-byte Q2_K
    // blocks (vs 276-byte prepacked) cut the weight traffic ~3.3x, matching
    // llama.cpp's working set. The 2-bit quants are unpacked to bytes on the
    // fly inside the kernel, producing exactly the element indices that the
    // prepacked layout would have (so the math is identical).
    void matMulVecFusedGateUpQ2_K_Compact_Q8_AVX2(
            const uint8_t *gateData,
            const uint8_t *upData,
            const float *x,
            uint32_t rows,
            uint32_t cols,
            float *gateOut,
            float *upOut,
            bool applySwish);

    // Fused Q+K matvec for COMPACT (raw GGUF) Q2_K weights, single token
    // (generation). Computes qOut = x * Q^T and kOut = x * K^T in a single pass.
    // Q (attnQ, 1536 rows) and K (attnK, 256 rows) are both compact Q2_K with
    // the same column count, so the combined row space (qRows + kRows) tiles in
    // 8-row batches with a single Q8_K quantization of x shared across both
    // matrices. Compact 84-byte blocks (vs 276-byte prepacked) cut generation
    // Q/K weight traffic ~3.3x — the single-token QKV stage is memory-bound.
    void matMulVecFusedQKQ2_K_Compact_Q8_AVX2(
            const uint8_t *qData,
            const uint8_t *kData,
            const float *x,
            uint32_t qRows,
            uint32_t kRows,
            uint32_t cols,
            float *qOut,
            float *kOut);

    // Register-tiled batch GEMM for a single Q2_K matrix over a batch of tokens
    // (prefill). For each tile of 8 rows, the weight blocks are loaded once and
    // reused across all seqLen tokens, eliminating the seqLen-fold re-read of the
    // weight matrix that dominates prefill. Q8_K x-vector data is reused across
    // the 8 rows in the tile. This is the single-matrix analogue of the fused
    // gate+up batch kernel, used for the ffnDown projection.
    void matMulVecBatchQ2_K_PrePacked_Q8_Batch_AVX2(
            const uint8_t *prepackedData,
            const float *X,
            uint32_t seqLen,
            uint32_t rows,
            uint32_t cols,
            float *out);

    // Register-tiled batch GEMM for a single Q8_K weight matrix over a batch of
    // tokens (prefill). Q8_K analogue of the Q2_K batch kernels, used for the
    // large attnO and ffnDown prefill matmuls (P4). For each tile of 8 rows the
    // weight blocks are loaded once and reused across all seqLen tokens, the
    // Q8_K x-vector is loaded and sign-processed once per (token, block) and
    // reused across the 8 rows, and the 8 partial sums are kept in __m256
    // accumulators (accumulated in registers, horizontal-summed once per
    // output) instead of a scalar function-pointer dispatch per
    // (row, token, block).
    void matMulVecBatchQ8K_Q8K_AVX2(const Q8KBlock *W_q8k, const float *X,
                                    uint32_t seqLen, uint32_t rows,
                                    uint32_t cols, float *out);

    // Register-tiled batch GEMM for a single Q4_K weight matrix over a batch of
    // tokens (prefill). Used for the attnV projection (Q4_K has no prepacked
    // batch kernel otherwise). For each tile of 8 rows the per-block Q4_K scale/
    // min unpacking and nibble expansion are hoisted out of the token loop
    // (weight-stationary); the inner 32-element sub-block dots use
    // _mm256_maddubs_epi16 against the Q8_K x-vector.
    void matMulVecBatchQ4K_Q8K_AVX2(const uint8_t *W_q4k, const float *X,
                                    uint32_t seqLen, uint32_t rows,
                                    uint32_t cols, float *out);

    // Register-tiled batch GEMM for a single COMPACT (raw GGUF) Q3_K weight
    // matrix over a batch of tokens. Used for attnO and ffnDown, which are
    // stored as Q3_K in the model file (110 bytes/block = 0.43 B/elem vs
    // 1.14 B/elem for the Q8_K copies that would otherwise be streamed).
    // Generation is DRAM-bandwidth-bound, so reading the compact blocks
    // directly cuts the weight traffic ~2.65x, matching llama.cpp's working
    // set. The 2-bit planes unpack identically to the compact Q2_K kernel;
    // the Q3_K sign mask (hm) is folded into the planes as w_eff = q2 + 4*hm
    // (value = d_all*(sc-32)*(w_eff - 4)), and the per-16-group signed scales
    // (sc-32) are broadcast by the same q3k shuffle table as the prepacked
    // Q2_K kernels. The offset term -4*(sc-32)*bsum folds through the Q8KBlock
    // bsums.
    void matMulVecBatchQ3K_Q8K_AVX2(const uint8_t *W_q3k, const float *X,
                                    uint32_t seqLen, uint32_t rows,
                                    uint32_t cols, float *out,
                                    const float *residual = nullptr);

    // Register-tiled batch GEMM for a single COMPACT (raw GGUF) Q6_K weight
    // matrix over a batch of tokens. Used for the separate LM head (stored as
    // Q6_K in the model file; 210 bytes/block = 0.82 B/elem vs 1.14 B/elem for
    // the Q8_K copy that would otherwise be streamed — ~28% less LM-head weight
    // traffic per token). The 4-bit low/high nibble planes and the 2 high bits
    // are unpacked once per (block, row) (weight-stationary), the 16 signed
    // per-16-group scales are both sign-extended into the bsum compensation
    // (-32*sum_g sc[g]*bsum[g], accumulated per-lane like Q3_K) and shuffled
    // into the main term, and the inner 32-element sub-block dots run through
    // _mm256_maddubs_epi16 against the Q8_K x-vector (w' 0..63 is unsigned-safe
    // in the unsigned operand slot).
    void matMulVecBatchQ6K_Q8K_AVX2(const uint8_t *W_q6k, const float *X,
                                    uint32_t seqLen, uint32_t rows,
                                    uint32_t cols, float *out);

    // Register-tiled batch GEMM for a single COMPACT (raw GGUF) Q2_K weight
    // matrix over a batch of tokens. Used for the separate LM head after the
    // load-time Q2_K re-quant (Lever C: 84 B/block vs Q6_K's 210 B/block —
    // ~191->76 MB/token of LM-head weight traffic, the largest single per-token
    // saving). The 4-bit scale/min nibbles and the 8 unpacked 2-bit planes are
    // hoisted out of the token loop (weight-stationary), the min term folds
    // through the Q8KBlock bsums, and the inner 32-element sub-block dots run
    // through _mm256_maddubs_epi16 against the Q8_K x-vector — the same
    // machinery as the fused gate+up compact kernel's blockDot.
    void matMulVecBatchQ2K_Compact_Q8K_AVX2(const uint8_t *W_q2k,
                                            const float *X,
                                            uint32_t seqLen, uint32_t rows,
                                            uint32_t cols, float *out);

    // Fused gate+up+down FFN kernel for single-token generation.
    // Combines gate=Q2_K, up=Q2_K, and down=Q3_K into a single kernel that
    // computes: out = residual + down @ silu(gate @ x) * up @ x
    // This eliminates the intermediate gate+up→ffnDown dispatch and rounds-
    // trip, and hoists the Q8_K x-vector loads so they're done once per
    // block instead of per (block, row) in the Q3_K code path.
    void matMulVecFusedGateUpDownQ2K_Q3K_Compact_AVX2(
            const uint8_t *gateData,
            const uint8_t *upData,
            const uint8_t *downData,
            const float *x,
            uint32_t gRows,
            uint32_t hRows,
            uint32_t cols,
            float *out,
            float *residual);

    // Fused gate+up+down FFN kernel with a load-time COMPACT Q2_K ffnDown
    // (Lever C): down streams 84 B/block instead of Q3_K's 110 B/block (~24%
    // less weight traffic on the second-largest per-token matmul). Phase 1
    // (gate+up, compact Q2_K) is byte-identical to the Q3K variant; phase 2
    // runs the Q2_K dot against the shared Q8_K act blocks.
    void matMulVecFusedGateUpDownQ2K_Q2K_Compact_AVX2(
            const uint8_t *gateData,
            const uint8_t *upData,
            const uint8_t *downData,
            const float *x,
            uint32_t gRows,
            uint32_t hRows,
            uint32_t cols,
            float *out,
            float *residual);

#endif// __AVX2__ && __FMA__

#if defined(__AVX512F__) && defined(__FMA__)

    void accumulateFMA_AVX512(float *local, const float *blockOut, float alpha,
                              uint32_t n);
    float dotProductFMA_AVX512(const float *hidden, const float *blockOut,
                               uint32_t n);
    float dotProductFMA_F16_AVX512(const float *hidden, const uint16_t *W_f16,
                                   uint32_t n);
    void rmsNorm_AVX512(const float *x, float *out, const float *weight,
                        uint32_t n, float eps);
    void softmax_AVX512(float *x, uint32_t n);
    void silu_AVX512(float *x, uint32_t n);
    void swiGLU_AVX512(float *x, const float *y, uint32_t n);
    void add_AVX512(float *x, const float *y, uint32_t n);
    void scale_AVX512(float *x, float alpha, uint32_t n);
    float dotProductQ2_K_PrePacked_Q8_AVX512(const uint8_t *prepackedBlock,
                                             const Q8KBlock *q8);
    float dotProductQ8K_Q8K_AVX512(const Q8KBlock *x, const Q8KBlock *w);

#endif// __AVX512F__ && __FMA__

}// namespace tinycoder::simd