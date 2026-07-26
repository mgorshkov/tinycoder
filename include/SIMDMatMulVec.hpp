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

}// namespace tinycoder
