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

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace tinycoder {

    /// @brief SIMD-accelerated accumulation: local[j] += alpha * blockOut[j] for j
    /// in [0, n).
    ///
    /// Uses runtime dispatch to select the best available SIMD implementation
    /// (scalar, AVX2, AVX-512). The dispatch is performed once at first call
    /// and cached thereafter.
    ///
    /// @param local     Output accumulator array (size >= n)
    /// @param blockOut  Input block of dequantized values (size >= n)
    /// @param alpha     Scalar multiplier
    /// @param n         Number of elements to process
    void accumulateFMA(float *local, const float *blockOut, float alpha,
                       uint32_t n);

    /// @brief SIMD-accelerated dot product: result = sum(hidden[j] * blockOut[j])
    /// for j in [0, n).
    ///
    /// Uses runtime dispatch to select the best available SIMD implementation.
    ///
    /// @param hidden  Input vector (size >= n)
    /// @param blockOut  Input block of dequantized values (size >= n)
    /// @param n         Number of elements to process
    /// @return Dot product of hidden and blockOut
    float dotProductFMA(const float *hidden, const float *blockOut, uint32_t n);

    /// @brief SIMD-accelerated RMSNorm: out[i] = x[i] * invRms * weight[i]
    /// where invRms = 1.0 / sqrt(mean(x^2) + eps).
    ///
    /// Uses runtime dispatch to select the best available SIMD implementation.
    ///
    /// @param x      Input vector (size >= n)
    /// @param out    Output vector (size >= n, can alias x)
    /// @param weight Scale vector (size >= n)
    /// @param n      Number of elements
    /// @param eps    Epsilon for numerical stability (default 1e-6f)
    void rmsNormSIMD(const float *x, float *out, const float *weight,
                     uint32_t n, float eps = 1e-6f);

    /// @brief SIMD-accelerated softmax in-place: x[i] = exp(x[i] - max) / sum(exp(x - max))
    /// @param x   Input/output array
    /// @param n   Number of elements
    void softmaxSIMD(float *x, uint32_t n);

    /// @brief SIMD-accelerated SiLU (Sigmoid Linear Unit): x[i] = x[i] / (1 + exp(-x[i]))
    /// @param x   Input/output array
    /// @param n   Number of elements
    void siluSIMD(float *x, uint32_t n);

    /// @brief SIMD-accelerated SwiGLU: x[i] = silu(x[i]) * y[i]
    /// @param x   Input/output array (silu applied in-place, then multiplied by y)
    /// @param y   Gate array
    /// @param n   Number of elements
    void swiGLUSIMD(float *x, const float *y, uint32_t n);

    /// @brief SIMD-accelerated element-wise add: x[i] += y[i]
    /// @param x   Output accumulator
    /// @param y   Input to add
    /// @param n   Number of elements
    void addSIMD(float *x, const float *y, uint32_t n);

    /// @brief SIMD-accelerated element-wise multiply by scalar: x[i] *= alpha
    /// @param x     Input/output array
    /// @param alpha Scalar multiplier
    /// @param n     Number of elements
    void scaleSIMD(float *x, float alpha, uint32_t n);

    /// @brief SIMD-accelerated fused dot product for a Q2_K block (256 weights, 84 bytes).
    ///
    /// Computes dot(x, dequantize(blockData)) directly during dequantization,
    /// eliminating the float blockOut[256] temporary and the extra memory pass.
    /// Uses runtime dispatch to select the best available SIMD implementation
    /// (scalar, AVX2). The dispatch is performed once at first call and cached
    /// thereafter.
    ///
    /// @param blockData  Pointer to the Q2_K block data (84 bytes)
    /// @param x          Input vector (size >= 256)
    /// @return Dot product of x and the dequantized block
    float dotProductQ2_K_SIMD(const uint8_t *blockData, const float *x);

    /// @brief Resolve and return the cached SIMD implementation pointer for
    /// dotProductQ2_K_SIMD. Callers in tight inner loops (e.g. the native Q2_K
    /// LM head per-vocab-row loop) can resolve once per token and invoke the
    /// pointer directly, avoiding the per-call atomic load (see BENCHMARK_REPORT
    /// §4.6 / P3).
    float (*dotProductQ2_K_get())(const uint8_t *, const float *);

    /// @brief SIMD-accelerated dot product with FP16-stored weights.
    ///
    /// Computes dot(hidden, W) where W is stored as FP16 (uint16_t).
    /// Uses F16C _mm256_cvtph_ps for on-the-fly FP16->FP32 conversion
    /// when AVX2 is available, falling back to scalar FP16 conversion.
    ///
    /// This halves memory bandwidth compared to F32 storage (55 MB -> 27.5 MB
    /// for ffnDown) while adding minimal conversion overhead.
    ///
    /// @param hidden  Input vector (size >= n)
    /// @param W_f16   Weight matrix row stored as FP16 (size >= n)
    /// @param n       Number of elements to process
    /// @return Dot product of hidden and W_f16
    float dotProductFMA_F16(const float *hidden, const uint16_t *W_f16, uint32_t n);

    /// @brief Resolve and return the cached SIMD implementation pointer for
    /// dotProductFMA_F16. The pointer is resolved once on first call and cached.
    /// Callers in tight inner loops (e.g. per-block in matmuls) can resolve once
    /// per matmul call and invoke the pointer directly, avoiding the per-call
    /// atomic load with acquire semantics (see BENCHMARK_REPORT §4.6 / P3).
    float (*dotProductFMA_F16_get())(const float *, const uint16_t *, uint32_t);

    /// @brief SIMD-accelerated dot product for a pre-packed Q2_K block (276 bytes).
    ///
    /// Pre-packed blocks have the 2-bit values expanded to full bytes (0-3) in
    /// element order, eliminating the shift/mask/pack extraction overhead.
    ///
    /// Pre-packed block format (276 bytes):
    ///   Offset 0-15:   scales[16]   (copied from original)
    ///   Offset 16-17:  d            (fp16, copied from original)
    ///   Offset 18-19:  dmin         (fp16, copied from original)
    ///   Offset 20-275: qs_expanded[256] (each byte is 0-3, in element order)
    ///
    /// @param prepackedBlock  Pointer to the pre-packed Q2_K block (276 bytes)
    /// @param x               Input vector (size >= 256)
    /// @return Dot product of x and the dequantized block
    float dotProductQ2_K_PrePacked_SIMD(const uint8_t *prepackedBlock, const float *x);

    /// @brief A Q8_K block: the input vector x quantized to int8
    struct Q8KBlock {
        float d;          // block scale
        int8_t qs[256];   // quantized values
        int16_t bsums[16];// sum of each 16-element group
    };

    /// @brief SIMD-accelerated dot product for a pre-packed Q2_K block against a
    /// Q8_K-quantized x vector.
    ///
    /// @param prepackedBlock  Pointer to the pre-packed Q2_K block (276 bytes)
    /// @param q8              Pointer to the Q8_K-quantized x block
    /// @return Dot product of x and the dequantized block
    float dotProductQ2_K_PrePacked_Q8_SIMD(const uint8_t *prepackedBlock,
                                           const Q8KBlock *q8);

    /// @brief SIMD-accelerated register-tiled batch GEMM for Q2_K matrices.
    ///
    /// Processes BATCH_SIZE=8 rows in parallel using AVX2 registers.
    /// For each tile of 8 rows, accumulates into ymm registers across
    /// all block columns before storing. This eliminates redundant reads
    /// of the Q8_K x-vector data and keeps intermediate results in registers.
    ///
    /// @param prepackedData Pre-packed Q2_K data (from prepackQ2_K)
    /// @param x             Input vector (size cols)
    /// @param rows          Number of output rows
    /// @param cols          Number of input columns
    /// @param result        Output buffer (size rows)
    void matMulVecBatchQ2_K_PrePacked_Q8_SIMD(
            const uint8_t *prepackedData,
            const float *x,
            uint32_t rows,
            uint32_t cols,
            float *result);

    /// @brief Register-tiled batch GEMM for fused Q2_K gate+up (8 rows at a time).
    ///
    /// Computes gate = x * W_gate^T and up = x * W_up^T in a single pass. For
    /// each tile of 8 rows, the Q8_K x-vector data is loaded once and reused
    /// across all 8 rows AND both matrices, halving the dominant x-vector memory
    /// traffic vs. computing gate and up separately.
    ///
    /// @param gatePrepacked Pre-packed Q2_K data for the gate matrix
    /// @param upPrepacked   Pre-packed Q2_K data for the up matrix
    /// @param x             Input vector (size cols)
    /// @param rows          Number of output rows
    /// @param cols          Number of input columns
    /// @param gateOut       Output buffer for gate (size rows)
    /// @param upOut         Output buffer for up (size rows)
    void matMulVecBatchGateUpQ2_K_PrePacked_Q8_SIMD(
            const uint8_t *gatePrepacked,
            const uint8_t *upPrepacked,
            const float *x,
            uint32_t rows,
            uint32_t cols,
            float *gateOut,
            float *upOut);

    /// @brief Register-tiled fused gate+up matvec for COMPACT (raw GGUF) Q2_K
    /// weights, single token (generation).
    ///
    /// Identical maths to matMulVecBatchGateUpQ2_K_PrePacked_Q8_SIMD but reads
    /// the raw 84-byte Q2_K blocks (scales[16], qs[64] packed 2-bit, fp16
    /// d/dmin) and unpacks the 2-bit quants to bytes on the fly inside the
    /// kernel. Generation is DRAM-bandwidth-bound, and the compact layout is
    /// 84 B/block vs 276 B/block for the prepacked copy — a ~3.3× reduction in
    /// the weight traffic that dominates per-token time. The unpack produces
    /// indices identical to the prepacked layout (verified against prepackQ2_K),
    /// so the math is exactly the same.
    ///
    /// @param gateData Raw (compact) Q2_K data for the gate matrix
    /// @param upData   Raw (compact) Q2_K data for the up matrix
    /// @param x        Input vector (size cols)
    /// @param rows     Number of output rows
    /// @param cols     Number of input columns
    /// @param gateOut  Output buffer for gate (size rows)
    /// @param upOut    Output buffer for up (size rows)
    /// @param applySwish When true, fuse the SwiGLU activation
    ///                   gateOut[j] = silu(gateOut[j]) * upOut[j] into the
    ///                   epilogue, removing the separate swiGLUSIMD pass over
    ///                   the gate/up activation arrays (mirrors the prefill
    ///                   batch kernel's applySwish).
    void matMulVecFusedGateUpQ2_K_Compact_Q8_SIMD(
            const uint8_t *gateData,
            const uint8_t *upData,
            const float *x,
            uint32_t rows,
            uint32_t cols,
            float *gateOut,
            float *upOut,
            bool applySwish);

    /// @brief Fused Q+K matvec for COMPACT (raw GGUF) Q2_K weights, single token
    /// (generation).
    ///
    /// Computes qOut = x * Q^T and kOut = x * K^T in a single pass over x.
    /// attnQ (1536 rows, 12 heads x 128) and attnK (256 rows, 2 KV heads x 128)
    /// are both compact Q2_K in this model with an identical column count, so
    /// the combined row space tiles in
    /// 8-row batches like the fused gate+up kernel, with one Q8_K quantization
    /// of x shared across both matrices. Generation is DRAM-bandwidth-bound, and
    /// the compact 84-byte Q2_K blocks (vs 276-byte prepacked) cut the Q/K
    /// weight traffic ~3.3x.
    ///
    /// @param qData  Raw (compact) Q2_K data for the Q matrix
    /// @param kData  Raw (compact) Q2_K data for the K matrix
    /// @param x      Input vector (size cols)
    /// @param qRows  Number of output rows of the Q matrix
    /// @param kRows  Number of output rows of the K matrix
    /// @param cols   Shared input dimension of both matrices
    /// @param qOut   Output buffer for Q (size qRows)
    /// @param kOut   Output buffer for K (size kRows)
    void matMulVecFusedQKQ2_K_Compact_Q8_SIMD(
            const uint8_t *qData,
            const uint8_t *kData,
            const float *x,
            uint32_t qRows,
            uint32_t kRows,
            uint32_t cols,
            float *qOut,
            float *kOut);

    /// @brief Register-tiled batch GEMM for fused Q2_K gate+up over a batch of
    /// tokens (prefill).
    ///
    /// Computes gateOut[s*rows + j] = sum_i X[s*cols + i] * gate[j*cols + i] and
    /// upOut[s*rows + j] = sum_i X[s*cols + i] * up[j*cols + i] for s in
    /// [0, seqLen), j in [0, rows). For each tile of 8 rows, the weight blocks
    /// are loaded once and reused across all seqLen tokens, eliminating the
    /// seqLen-fold re-read of the weight matrix that dominates prefill.
    ///
    /// @param gatePrepacked Pre-packed Q2_K data for the gate matrix
    /// @param upPrepacked   Pre-packed Q2_K data for the up matrix
    /// @param X             Input matrix (seqLen x cols, row-major)
    /// @param seqLen        Number of tokens in the batch
    /// @param rows          Number of output rows
    /// @param cols          Number of input columns
    /// @param gateOut       Output buffer for gate (seqLen x rows)
    /// @param upOut         Output buffer for up (seqLen x rows)
    /// @param applySwish    When true, fuse SwiGLU into the epilogue:
    ///                      gateOut[s*rows+j] *= silu(gateOut[s*rows+j])
    ///                      (P5: removes the separate activation pass over the
    ///                      seqLen x rows activation tensors).
    void matMulVecBatchGateUpQ2_K_PrePacked_Q8_Batch_SIMD(
            const uint8_t *gatePrepacked,
            const uint8_t *upPrepacked,
            const float *X,
            uint32_t seqLen,
            uint32_t rows,
            uint32_t cols,
            float *gateOut,
            float *upOut,
            bool applySwish);

    /// @brief Register-tiled batch GEMM for a single Q2_K matrix over a batch of
    /// tokens (prefill).
    ///
    /// Computes out[s*rows + j] = sum_i X[s*cols + i] * W[j*cols + i] for s in
    /// [0, seqLen), j in [0, rows). For each tile of 8 rows, the weight blocks
    /// are loaded once and reused across all seqLen tokens, eliminating the
    /// seqLen-fold re-read of the weight matrix that dominates prefill. Q8_K
    /// x-vector data is reused across the 8 rows in the tile. This is the
    /// single-matrix analogue of the fused gate+up batch kernel, used for the
    /// ffnDown projection.
    ///
    /// @param prepackedData Pre-packed Q2_K data (from prepackQ2_K)
    /// @param X             Input matrix (seqLen x cols, row-major)
    /// @param seqLen        Number of tokens in the batch
    /// @param rows          Number of output rows
    /// @param cols          Number of input columns
    /// @param out           Output buffer (seqLen x rows)
    void matMulVecBatchQ2_K_PrePacked_Q8_Batch_SIMD(
            const uint8_t *prepackedData,
            const float *X,
            uint32_t seqLen,
            uint32_t rows,
            uint32_t cols,
            float *out);

    /// @brief SIMD-accelerated dot product of two Q8_K-quantized blocks.
    ///
    /// Computes dot(x, w) where both x and w are stored as Q8_K blocks (256
    /// int8 values + a block scale). Uses _mm256_maddubs_epi16 (32 int8×int8->
    /// int16 multiply-adds per instruction) instead of float FMAs (8 per
    /// instruction). This is used by the Q8_K LM head path, which stores the
    /// embedding matrix as Q8_K (~1 byte/element) to cut memory bandwidth ~2×
    /// vs FP16 and ~4× vs F32.
    ///
    /// @param x  Pointer to the Q8_K-quantized input block
    /// @param w  Pointer to the Q8_K-quantized weight block
    /// @return Dot product of the two dequantized blocks
    float dotProductQ8K_Q8K_SIMD(const Q8KBlock *x, const Q8KBlock *w);

    /// @brief Register-tiled batch GEMM for a single Q8_K weight matrix over a
    /// batch of tokens (prefill), used for the large attnO and ffnDown matmuls.
    ///
    /// Computes out[s*rows + j] = sum_i X[s*cols + i] * W_q8k[j*cols + i] for
    /// all tokens s and output rows j. On a host with a vector (AVX2) kernel the
    /// implementation keeps the 8 partial sums in __m256 accumulators and reuses
    /// the Q8_K x-vector across the 8 tile rows, replacing the previous per
    /// (row, token, block) scalar function-pointer dispatch. Returns true if a
    /// vector kernel was used, false if the caller must fall back to the scalar
    /// per-pair loop (non-AVX2 host).
    ///
    /// @param W_q8k  Q8_K weights, row-major [rows * blocksPerRow] Q8KBlock array
    /// @param X      Input activations [seqLen, cols] (row-major)
    /// @param seqLen Number of tokens in the batch
    /// @param rows   Number of output rows
    /// @param cols   Number of input columns
    /// @param out    Output [seqLen, rows] (row-major)
    /// @return true if the vector batch kernel was dispatched
    bool matMulVecBatchQ8K_SIMD(const Q8KBlock *W_q8k, const float *X,
                                uint32_t seqLen, uint32_t rows, uint32_t cols,
                                float *out);

    /// @brief Register-tiled batch GEMM for a single Q4_K weight matrix over a
    /// batch of tokens (prefill), used for the attnV projection.
    ///
    /// Computes out[s*rows + j] = sum_i X[s*cols + i] * W_q4k[j*cols + i] for
    /// all tokens s and output rows j, where W_q4k is raw Q4_K data (144 bytes
    /// per 256-element block). On a host with a vector (AVX2) kernel the
    /// per-block scale/min unpacking and nibble expansion are hoisted out of the
    /// token loop (weight-stationary) and the inner sub-block dots run through
    /// _mm256_maddubs_epi16 against a Q8_K x-vector. Returns true if a vector
    /// kernel was used, false if the caller must fall back to the generic path.
    ///
    /// @param W_q4k  Q4_K weights, row-major [rows * blocksPerRow] raw blocks
    /// @param X      Input activations [seqLen, cols] (row-major)
    /// @param seqLen Number of tokens in the batch
    /// @param rows   Number of output rows
    /// @param cols   Number of input columns
    /// @param out    Output [seqLen, rows] (row-major)
    /// @return true if the vector batch kernel was dispatched
    bool matMulVecBatchQ4K_SIMD(const uint8_t *W_q4k, const float *X,
                                uint32_t seqLen, uint32_t rows, uint32_t cols,
                                float *out);

    /// @brief Register-tiled batch GEMM for a single COMPACT (raw GGUF) Q3_K
    /// weight matrix over a batch of tokens, used for the attnO and ffnDown
    /// projections.
    ///
    /// Computes out[s*rows + j] = sum_i X[s*cols + i] * W_q3k[j*cols + i] for
    /// all tokens s and output rows j, where W_q3k is raw Q3_K data (110 bytes
    /// per 256-element block). Q3_K is how attnO/ffnDown are stored in the model
    /// file (0.43 B/elem); generation is DRAM-bandwidth-bound, so streaming the
    /// compact blocks directly (instead of the Q8_K copies at 1.14 B/elem that
    /// would otherwise be rebuilt and read) cuts the weight traffic ~2.65x,
    /// matching llama.cpp's working set. On a host with a vector (AVX2) kernel
    /// the per-block setup (fp16 super scale, per-16-group signed scales,
    /// expanded 2-bit planes with the sign mask folded in) is hoisted out of the
    /// token loop (weight-stationary) and the inner 32-element sub-block dots
    /// run through _mm256_maddubs_epi16 against a Q8_K x-vector. Returns true if
    /// a vector kernel was used, false if the caller must fall back to the
    /// generic path.
    ///
    /// @param W_q3k  Q3_K weights, row-major [rows * blocksPerRow] raw blocks
    /// @param X      Input activations [seqLen, cols] (row-major)
    /// @param seqLen Number of tokens in the batch
    /// @param rows   Number of output rows
    /// @param cols   Number of input columns
    /// @param out    Output [seqLen, rows] (row-major)
    /// @param residual Optional residual vector folded into the store epilogue
    ///                 (out[i] = dot + residual[i]). The attnO single-token
    ///                 path passes hidden as both out and residual to fuse the
    ///                 attention residual add into the kernel and remove the
    ///                 separate addSIMD pass. Nullptr (default) for the plain
    ///                 matmul.
    /// @return true if the vector batch kernel was dispatched
    bool matMulVecBatchQ3K_SIMD(const uint8_t *W_q3k, const float *X,
                                uint32_t seqLen, uint32_t rows, uint32_t cols,
                                float *out, const float *residual = nullptr);

    /// @brief Register-tiled batch GEMM for a single COMPACT (raw GGUF) Q6_K
    /// weight matrix over a batch of tokens, used for the separate LM head.
    ///
    /// Computes out[s*rows + j] = sum_i X[s*cols + i] * W_q6k[j*cols + i] for
    /// all tokens s and output rows j, where W_q6k is raw Q6_K data (210 bytes
    /// per 256-element block). Q6_K is how the separate LM head is stored in the
    /// model file (0.82 B/elem); generation is DRAM-bandwidth-bound, so
    /// streaming the compact blocks directly (instead of the Q8_K copies at
    /// 1.14 B/elem that would otherwise be rebuilt and read) cuts the LM-head
    /// weight traffic ~28%, matching llama.cpp's working set. On a host with a
    /// vector (AVX2) kernel the per-block setup (fp16 block scale, per-16-group
    /// signed int8 scales, expanded 6-bit planes) is hoisted out of the token
    /// loop (weight-stationary) and the inner 32-element sub-block dots run
    /// through _mm256_maddubs_epi16 against a Q8_K x-vector. The -32 offset of
    /// w_eff = w' - 32 folds through the Q8KBlock bsums as a per-lane vector
    /// (-32*sum_g sc[g]*bsum[g]). Returns true if a vector kernel was used,
    /// false if the caller must fall back to the generic path.
    ///
    /// @param W_q6k  Q6_K weights, row-major [rows * blocksPerRow] raw blocks
    /// @param X      Input activations [seqLen, cols] (row-major)
    /// @param seqLen Number of tokens in the batch
    /// @param rows   Number of output rows
    /// @param cols   Number of input columns
    /// @param out    Output [seqLen, rows] (row-major)
    /// @return true if the vector batch kernel was dispatched
    bool matMulVecBatchQ6K_SIMD(const uint8_t *W_q6k, const float *X,
                                uint32_t seqLen, uint32_t rows, uint32_t cols,
                                float *out);

    /// @brief Register-tiled batch GEMM for a single COMPACT (raw GGUF) Q2_K
    /// weight matrix over a batch of tokens, used for the separate LM head when
    /// a load-time Q2_K re-quant copy exists (Lever C, Plan §7).
    ///
    /// Computes out[s*rows + j] = sum_i X[s*cols + i] * W_q2k[j*cols + i] for
    /// all tokens s and output rows j, where W_q2k is the load-time re-quantized
    /// Q2_K layout (84 bytes per 256-element block). The re-quant cuts LM-head
    /// weight traffic ~2.8x vs the Q6_K raw blocks (0.82 B/elem -> 0.33 B/elem,
    /// ~191->76 MB/token) and, unlike a per-element float dequant dot, the
    /// 2-bit planes are dotted through _mm256_maddubs_epi16 against a Q8_K
    /// x-vector (per-block setup hoisted out of the token loop, min term folded
    /// through bsums) — keeping the kernel ALU-cheap and DRAM-bound. Returns
    /// true if a vector kernel was used, false if the caller must fall back to
    /// the per-block dot path.
    ///
    /// @param W_q2k  COMPACT Q2_K weights, row-major [rows * blocksPerRow] blocks
    /// @param X      Input activations [seqLen, cols] (row-major)
    /// @param seqLen Number of tokens in the batch
    /// @param rows   Number of output rows (vocabSize)
    /// @param cols   Number of input columns (hiddenSize)
    /// @param out    Output [seqLen, rows] (row-major)
    /// @return true if the vector batch kernel was dispatched
    bool matMulVecBatchQ2K_Compact_SIMD(const uint8_t *W_q2k, const float *X,
                                        uint32_t seqLen, uint32_t rows,
                                        uint32_t cols, float *out);

    /// @brief Fused gate+up+down FFN kernel for single-token generation.
    ///
    /// Combines gate=Q2_K, up=Q2_K, and down=Q3_K into a single kernel that
    /// computes: out = residual + down @ (silu(gate @ x) * up @ x)
    /// in a single pass. This eliminates the intermediate gate+up→ffnDown
    /// dispatch and round-trip, and hoists the Q3_K makeSetup so it runs once
    /// per (block, row) instead of inside the per-token loop.
    ///
    /// @param gateData   Q2_K compact gate weights (raw GGUF)
    /// @param upData     Q2_K compact up weights (raw GGUF)
    /// @param downData   Q3_K compact down weights (raw GGUF)
    /// @param x          Input activation vector
    /// @param gRows      Gate/up output dimension (intermediateSize)
    /// @param hRows      Down output dimension (hiddenSize)
    /// @param cols       Input vector dimension (hiddenSize)
    /// @param out        Output buffer (hRows)
    /// @param residual   Optional residual added to output (can be nullptr or in=out)
    /// @return true if the vector kernel was dispatched
    bool matMulVecFusedGateUpDownQ2K_Q3K_SIMD(
            const uint8_t *gateData,
            const uint8_t *upData,
            const uint8_t *downData,
            const float *x,
            uint32_t gRows,
            uint32_t hRows,
            uint32_t cols,
            float *out,
            const float *residual);

    /// @brief Fused gate+up+down where the DOWN matrix is a load-time COMPACT
    /// Q2_K re-quant copy (Lever C). Identical semantics to
    /// matMulVecFusedGateUpDownQ2K_Q3K_SIMD but the down weights stream at
    /// 84 B/block instead of Q3_K's 110 B/block, cutting per-token ffnDown
    /// weight traffic ~24%.
    ///
    /// @param gateData    Q2_K gate weights [gRows x cols] (compact, 84 B/block)
    /// @param upData      Q2_K up weights [gRows x cols] (compact, 84 B/block)
    /// @param downQ2kData Q2_K down weights [hRows x gRows] (compact, 84 B/block)
    /// @param x           Input vector (size cols = hiddenSize)
    /// @param gRows       intermediateSize (down rows)
    /// @param hRows       hiddenSize (down cols, out size)
    /// @param cols        hiddenSize (gate/up input dim)
    /// @param out         Output buffer (hRows)
    /// @param residual    Optional residual added to output
    /// @return true if the AVX2 kernel was dispatched
    bool matMulVecFusedGateUpDownQ2K_Q2K_SIMD(
            const uint8_t *gateData,
            const uint8_t *upData,
            const uint8_t *downData,
            const float *x,
            uint32_t gRows,
            uint32_t hRows,
            uint32_t cols,
            float *out,
            const float *residual);

    /// @brief Fused gate+up+down where the DOWN matrix is compact Q2_K AND the
    /// gate/up are the load-time PREPACKED layouts (276 B/block with 2-bit
    /// planes pre-expanded). Phase 1's per-(row,block) makeSetup becomes 8
    /// plain plane loads instead of 26 shift/and unpack ops, eliminating
    /// ~107k ALU-heavy block-unpacks per token on the dominant gate+up stage.
    ///
    /// @param gatePrepacked Q2_K gate weights, prepacked 276 B/block
    /// @param upPrepacked   Q2_K up weights, prepacked 276 B/block
    /// @param downQ2kData   Q2_K down weights [hRows x gRows] (compact, 84 B/block)
    /// @param x             Input vector (size cols = hiddenSize)
    /// @param gRows         intermediateSize (down rows)
    /// @param hRows         hiddenSize (down cols, out size)
    /// @param cols          hiddenSize (gate/up input dim)
    /// @param out           Output buffer (hRows)
    /// @param residual      Optional residual added to output

    /// @brief Resolve and return the cached SIMD implementation pointer for
    /// dotProductQ8K_Q8K_SIMD. The pointer is resolved once on first call and
    /// cached. Callers in tight inner loops (e.g. per-block in the LM head) can
    /// resolve once per token and invoke the pointer directly, avoiding the
    /// per-call atomic load with acquire semantics (see BENCHMARK_REPORT §4.6 / P3).
    float (*dotProductQ8K_Q8K_get())(const Q8KBlock *, const Q8KBlock *);

    /// @brief Issue a non-temporal prefetch hint for a cache line.
    ///
    /// Wraps the x86 _mm_prefetch intrinsic with a portable no-op fallback so it
    /// can be used in row loops of large matrix-vector products (e.g. ffnGate,
    /// ffnUp, ffnDown) to hide DRAM latency. The hint is advisory and never
    /// changes program semantics.
    ///
    /// @param p  Pointer to the start of the cache line to prefetch
    inline void prefetchRow(const void *p) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        _mm_prefetch(static_cast<const char *>(p), _MM_HINT_T0);
#else
        (void) p;
#endif
    }

}// namespace tinycoder
