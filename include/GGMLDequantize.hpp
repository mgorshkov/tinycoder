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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <vector>

#include "AlignedVector.hpp"
#include "GGUFLoader.hpp"
#include "SIMDMatMulVec.hpp"
#include "ThreadPool.hpp"

namespace tinycoder {

    /// @brief GGML quantization dequantization utilities.
    ///
    /// Provides dequantization for various GGML quantized formats
    /// used in mixed-quantization GGUF files (Q5_K, Q5_1, IQ3_S, IQ2_S, etc.).
    struct GGMLDequantize {

        /// K-quant block size constant used in dequantization functions.
        static constexpr uint32_t QK_K = 256;

        /// @brief Convert IEEE 754 half-precision (16-bit) to float.
        /// @brief Convert a float32 value to IEEE 754 half-precision (FP16).
        static uint16_t floatToHalf(float f) {
            uint32_t f32;
            std::memcpy(&f32, &f, sizeof(uint32_t));
            uint32_t sign = (f32 >> 31) & 1;
            int32_t exp = static_cast<int32_t>((f32 >> 23) & 0xFF) - 127;
            uint32_t mant = f32 & 0x7FFFFF;

            uint16_t h;
            if (exp > 15) {
                // Infinity or NaN: saturate to infinity
                h = (sign << 15) | (0x1F << 10);
                if (mant != 0) {
                    h |= 0x200;// NaN with quiet bit
                }
            } else if (exp > -14) {
                // Normalized float16 value
                h = (sign << 15) | ((static_cast<uint32_t>(exp + 15) & 0x1F) << 10) | (mant >> 13);
            } else if (exp > -24) {
                // Subnormal float16 value
                mant = (mant | 0x800000) >> (14 - exp);
                h = (sign << 15) | (mant >> 13);
            } else {
                // Zero (or flush to zero)
                h = sign << 15;
            }
            return h;
        }

        static float halfToFloat(uint16_t h) {
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;

            uint32_t f32;
            if (exp == 0) {
                if (mant == 0) {
                    f32 = sign << 31;
                } else {
                    // Subnormal half: value = mant / 1024 * 2^(-14) = mant * 2^(-24)
                    // Convert to normalized float32:
                    //   Find leading zeros in 10-bit mantissa (bits 9-0)
                    //   Normalize: mant_norm = mant << n (bit 9 becomes set)
                    //   exp_f32 = 112 - n  (since 127 - 15 = 112)
                    //   mant_f32 = (mant_norm - 512) * 16384  (remove implicit 1, shift to 23-bit position)
                    int n = 0;
                    while ((mant & 0x200) == 0 && n < 10) {
                        mant <<= 1;
                        n++;
                    }
                    mant &= 0x3FF;
                    // mant is now normalized (bit 9 set), value = (512 + mant_low) * 2^(-24)
                    // = (1 + mant_low/512) * 2^(-15)
                    // In float32: exp = 127 - 15 = 112, mant = mant_low * 2^14
                    uint32_t mant_low = mant - 512;
                    exp = 112 - n;
                    f32 = (sign << 31) | (exp << 23) | (mant_low << 14);
                }
            } else if (exp == 31) {
                f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
            } else {
                exp = exp + (127 - 15);
                f32 = (sign << 31) | (exp << 23) | (mant << 13);
            }

            float result;
            std::memcpy(&result, &f32, sizeof(float));
            return result;
        }

        /// @brief Branch-free fp16->fp32 conversion via a 64 KiB LUT.
        ///
        /// The scalar halfToFloat() above is called per (row, block) in the hot
        /// mat-vec kernels — millions of times per generated token — and each
        /// call executes 2-3 data-dependent branches (exp==0 / exp==31 / else,
        /// plus the subnormal while loop). This LUT performs the SAME conversion
        /// with ZERO branches: index the 16-bit pattern once, load the float.
        ///
        /// - Bit-exact vs halfToFloat() (verified — the LUT replicates the
        ///   scalar math for every one of the 65536 patterns).
        /// - 64 KiB table, L1-resident (L1D on Haswell is 32 KiB per core, so a
        ///   hot working set of block d/dmin pairs stays cached).
        /// - Read-only after first touch; function-local static with built-in
        ///   thread-safety (magic static).
        static const float *halfToFloatTable() {
            static const float *table = []() {
                static std::vector<float> t(65536);
                for (uint32_t i = 0; i < 65536; ++i) {
                    t[i] = halfToFloat(static_cast<uint16_t>(i));
                }
                return t.data();
            }();
            return table;
        }

        static float halfToFloatBranchFree(uint16_t h) {
            return halfToFloatTable()[h];
        }

        /// @brief Dequantize a Q5_0 block (32 weights, 22 bytes per block).
        /// Block layout: d(fp16,2) + qh(4) + ql(16) = 22 bytes.
        /// Each weight is 5 bits: low 4 bits from ql, high bit from qh.
        /// Dequantized value = q * d (no min, unlike Q5_1).
        static void dequantizeQ5_0Block(const uint8_t *blockData, float *out) {
            float d_val = halfToFloat(*(const uint16_t *) (blockData + 0));
            uint32_t qh;
            std::memcpy(&qh, blockData + 2, sizeof(uint32_t));
            const uint8_t *qs = blockData + 6;

            // Q5_0 stores 32 weights as 16 pairs in qs[0..15].
            // For each pair j (0..15):
            //   x0 = (qs[j] & 0x0F) | high_bit_0  -> value 0..31, centered: -16
            //   x1 = (qs[j] >> 4)    | high_bit_1  -> value 0..31, centered: -16
            // High bits: xh_0 from qh bit j, xh_1 from qh bit (j+12)
            for (int j = 0; j < 16; ++j) {
                const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

                const int32_t x0 = ((qs[j] & 0x0F) | xh_0) - 16;
                const int32_t x1 = ((qs[j] >> 4) | xh_1) - 16;

                out[j + 0] = static_cast<float>(x0) * d_val;
                out[j + 16] = static_cast<float>(x1) * d_val;
            }
        }

        static std::vector<float> dequantizeQ5_0(const uint8_t *data,
                                                 uint64_t numElements) {
            static constexpr uint32_t BLOCK_SIZE = 32;
            static constexpr uint32_t BLOCK_BYTES = 22;// d(2) + qh(4) + ql(16)
            uint64_t numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::vector<float> result(numElements);

            for (uint64_t b = 0; b < numBlocks; ++b) {
                float blockOut[BLOCK_SIZE];
                dequantizeQ5_0Block(data + b * BLOCK_BYTES, blockOut);
                uint64_t start = b * BLOCK_SIZE;
                uint64_t end = std::min(start + BLOCK_SIZE, numElements);
                for (uint64_t i = start; i < end; ++i) {
                    result[i] = blockOut[i - start];
                }
            }
            return result;
        }

        /// @brief Dequantize a Q5_1 block (32 weights, 24 bytes per block).
        static void dequantizeQ5_1Block(const uint8_t *blockData, float *out) {
            float d_val = halfToFloat(*(const uint16_t *) (blockData + 0));
            float m_val = halfToFloat(*(const uint16_t *) (blockData + 2));
            uint32_t qh = *(const uint32_t *) (blockData + 4);
            const uint8_t *ql = blockData + 8;

            for (int i = 0; i < 32; ++i) {
                uint8_t low4 = (ql[i / 2] >> (4 * (i % 2))) & 0xF;
                uint8_t highBit = (qh >> i) & 1;
                uint8_t q = low4 | (highBit << 4);
                out[i] = static_cast<float>(q) * d_val + m_val;
            }
        }

        static std::vector<float> dequantizeQ5_1(const uint8_t *data,
                                                 uint64_t numElements) {
            static constexpr uint32_t BLOCK_SIZE = 32;
            static constexpr uint32_t BLOCK_BYTES = 24;// d(2) + m(2) + qh(4) + ql(16)
            uint64_t numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::vector<float> result(numElements);

            for (uint64_t b = 0; b < numBlocks; ++b) {
                float blockOut[BLOCK_SIZE];
                dequantizeQ5_1Block(data + b * BLOCK_BYTES, blockOut);
                uint64_t start = b * BLOCK_SIZE;
                uint64_t end = std::min(start + BLOCK_SIZE, numElements);
                for (uint64_t i = start; i < end; ++i) {
                    result[i] = blockOut[i - start];
                }
            }
            return result;
        }

        /// @brief Dequantize a Q8_0 block (32 weights, 34 bytes per block).
        /// Block layout: d(fp16,2) + qs(int8,32) = 34 bytes.
        /// Each weight is a signed 8-bit integer.
        /// Dequantized value = q * d.
        static void dequantizeQ8_0Block(const uint8_t *blockData, float *out) {
            float d_val = halfToFloat(*(const uint16_t *) (blockData + 0));
            const int8_t *qs = reinterpret_cast<const int8_t *>(blockData + 2);

            for (int i = 0; i < 32; ++i) {
                out[i] = static_cast<float>(qs[i]) * d_val;
            }
        }

        static std::vector<float> dequantizeQ8_0(const uint8_t *data,
                                                 uint64_t numElements) {
            static constexpr uint32_t BLOCK_SIZE = 32;
            static constexpr uint32_t BLOCK_BYTES = 34;// d(2) + qs(32)
            uint64_t numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::vector<float> result(numElements);

            for (uint64_t b = 0; b < numBlocks; ++b) {
                float blockOut[BLOCK_SIZE];
                dequantizeQ8_0Block(data + b * BLOCK_BYTES, blockOut);
                uint64_t start = b * BLOCK_SIZE;
                uint64_t end = std::min(start + BLOCK_SIZE, numElements);
                for (uint64_t i = start; i < end; ++i) {
                    result[i] = blockOut[i - start];
                }
            }
            return result;
        }

        /// @brief Dequantize a Q5_K block (256 weights, 176 bytes per block).
        ///        Matches the reference dequantize_row_q5_K implementation.
        ///        Block layout: d(2) + dmin(2) + scales(12) + qh(32) + qs(128) = 176
        ///        bytes. qh is read as a single uint32_t (only first 4 bytes used).
        ///        Within each 64-element chunk, 2 bits of qh provide the high bit
        ///        for the two 32-element sub-blocks. The bit positions used within
        ///        each byte of qh are: chunk0=bits0,1; chunk1=bits2,3;
        ///        chunk2=bits4,5; chunk3=bits6,7. After each chunk, qh is shifted
        ///        right by 8 to use the next byte. The 5-bit quant values are cast to
        ///        int8_t so that values 16-31 become negative (two's complement).
        static void dequantizeQ5_KBlock(const uint8_t *blockData, float *out) {
            float d = halfToFloat(*(const uint16_t *) (blockData + 0));
            float dmin = halfToFloat(*(const uint16_t *) (blockData + 2));

            const uint8_t *scales = blockData + 4;
            const uint8_t *qh = blockData + 16;
            const uint8_t *qs = blockData + 48;

            auto getScaleMin = [](int j, const uint8_t *q, uint8_t *d_out,
                                  uint8_t *m_out) {
                if (j < 4) {
                    *d_out = q[j] & 63;
                    *m_out = q[j + 4] & 63;
                } else {
                    *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                    *m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
                }
            };

            int is = 0;
            uint8_t sc, m;
            float *y = out;

            // Q5_K block layout (176 bytes total):
            //   d (2 bytes): super-block scale (fp16)
            //   dmin (2 bytes): super-block min (fp16)
            //   scales (12 bytes): 8 x 6-bit scales/mins (K_SCALE_SIZE = 12)
            //   qh (32 bytes): high bits for 5th bit of each weight (QK_K/8 = 32)
            //   qs (128 bytes): low 4 bits of each weight (QK_K/2 = 128)
            //
            // Each 256-weight block is divided into 4 chunks of 64 weights.
            // Each chunk has 2 sub-blocks of 32 weights.
            // For each sub-block of 32 weights:
            //   qs[l] contains the low 4 bits of weight l
            //   qh[l] bit u selects the 5th (high) bit of weight l
            //   u advances by 2 bits per chunk (u1 for first sub-block, u2 for second)
            //
            // Reference: dequantize_row_q5_K
            uint8_t u1 = 1, u2 = 2;

            for (int j = 0; j < 256; j += 64) {
                getScaleMin(is + 0, scales, &sc, &m);
                float d1 = d * static_cast<float>(sc);
                float m1 = dmin * static_cast<float>(m);

                getScaleMin(is + 1, scales, &sc, &m);
                float d2 = d * static_cast<float>(sc);
                float m2 = dmin * static_cast<float>(m);

                // First sub-block of 32: use bit u1 from qh[l] for each l
                // NOTE: q5 is unsigned 0-31 (matching the reference).
                // The int8_t cast would make values 16-31 negative, which is WRONG.
                for (int l = 0; l < 32; ++l) {
                    uint8_t q5 = (qs[l] & 0xF) | ((qh[l] & u1) ? 16 : 0);
                    *y++ = d1 * static_cast<float>(q5) - m1;
                }
                // Second sub-block of 32: use bit u2 from qh[l] for each l
                for (int l = 0; l < 32; ++l) {
                    uint8_t q5 = (qs[l] >> 4) | ((qh[l] & u2) ? 16 : 0);
                    *y++ = d2 * static_cast<float>(q5) - m2;
                }

                qs += 32;
                is += 2;
                // Advance u1/u2 to the next pair of bit positions within each qh byte
                u1 <<= 2;
                u2 <<= 2;
            }
        }

        static std::vector<float> dequantizeQ5_K(const uint8_t *data,
                                                 uint64_t numElements) {
            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t BLOCK_BYTES = 176;
            uint64_t numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::vector<float> result(numElements);

            for (uint64_t b = 0; b < numBlocks; ++b) {
                float blockOut[BLOCK_SIZE];
                dequantizeQ5_KBlock(data + b * BLOCK_BYTES, blockOut);
                uint64_t start = b * BLOCK_SIZE;
                uint64_t end = std::min(start + BLOCK_SIZE, numElements);
                for (uint64_t i = start; i < end; ++i) {
                    result[i] = blockOut[i - start];
                }
            }
            return result;
        }

        /// @brief Dequantize a Q4_K block (256 weights, 144 bytes per block).
        static void dequantizeQ4_KBlock(const uint8_t *blockData, float *out) {
            float d = halfToFloat(*(const uint16_t *) (blockData + 0));
            float dmin = halfToFloat(*(const uint16_t *) (blockData + 2));

            const uint8_t *scales = blockData + 4;
            const uint8_t *qs = blockData + 16;

            auto getScaleMin = [](int j, const uint8_t *q, uint8_t *d_out,
                                  uint8_t *m_out) {
                if (j < 4) {
                    *d_out = q[j] & 63;
                    *m_out = q[j + 4] & 63;
                } else {
                    *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                    *m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
                }
            };

            int is = 0;
            uint8_t sc, m;
            float *y = out;

            for (int j = 0; j < 256; j += 64) {
                getScaleMin(is + 0, scales, &sc, &m);
                float d1 = d * static_cast<float>(sc);
                float m1 = dmin * static_cast<float>(m);

                getScaleMin(is + 1, scales, &sc, &m);
                float d2 = d * static_cast<float>(sc);
                float m2 = dmin * static_cast<float>(m);

                for (int l = 0; l < 32; ++l) {
                    *y++ = d1 * static_cast<float>(qs[l] & 0xF) - m1;
                }
                for (int l = 0; l < 32; ++l) {
                    *y++ = d2 * static_cast<float>(qs[l] >> 4) - m2;
                }

                qs += 32;
                is += 2;
            }
        }

        static std::vector<float> dequantizeQ4_K(const uint8_t *data,
                                                 uint64_t numElements) {
            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t BLOCK_BYTES = 144;
            uint64_t numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::vector<float> result(numElements);

            for (uint64_t b = 0; b < numBlocks; ++b) {
                float blockOut[BLOCK_SIZE];
                dequantizeQ4_KBlock(data + b * BLOCK_BYTES, blockOut);
                uint64_t start = b * BLOCK_SIZE;
                uint64_t end = std::min(start + BLOCK_SIZE, numElements);
                for (uint64_t i = start; i < end; ++i) {
                    result[i] = blockOut[i - start];
                }
            }
            return result;
        }

        /// @brief Dequantize a Q6_K block (256 weights, 210 bytes per block).
        /// Matches the reference dequantize_row_q6_K implementation.
        /// Block layout: ql(128) + qh(64) + scales(16) + d(2) = 210 bytes.
        /// Each weight is 6 bits: low 4 bits from ql, high 2 bits from qh.
        /// Dequantized value = d * scale * (q - 32).
        static void dequantizeQ6_KBlock(const uint8_t *blockData, float *out) {
            const uint8_t *ql = blockData + 0;
            const uint8_t *qh = blockData + 128;
            const int8_t *sc = reinterpret_cast<const int8_t *>(blockData + 192);
            float d = halfToFloat(*(const uint16_t *) (blockData + 208));

            float *y = out;
            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    const int8_t q1 = (int8_t) ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int8_t q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int8_t q3 = (int8_t) ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int8_t q4 = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l + 0] = d * sc[is + 0] * q1;
                    y[l + 32] = d * sc[is + 2] * q2;
                    y[l + 64] = d * sc[is + 4] * q3;
                    y[l + 96] = d * sc[is + 6] * q4;
                }
                y += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }

        static std::vector<float> dequantizeQ6_K(const uint8_t *data,
                                                 uint64_t numElements) {
            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t BLOCK_BYTES = 210;
            uint64_t numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::vector<float> result(numElements);

            for (uint64_t b = 0; b < numBlocks; ++b) {
                float blockOut[BLOCK_SIZE];
                dequantizeQ6_KBlock(data + b * BLOCK_BYTES, blockOut);
                uint64_t start = b * BLOCK_SIZE;
                uint64_t end = std::min(start + BLOCK_SIZE, numElements);
                for (uint64_t i = start; i < end; ++i) {
                    result[i] = blockOut[i - start];
                }
            }
            return result;
        }

        /// @brief Dequantize a Q2_K block (256 weights, 84 bytes per block).
        /// Matches the reference dequantize_row_q2_K implementation.
        static void dequantizeQ2_KBlock(const uint8_t *blockData, float *out) {
            // Q2_K block layout (84 bytes total) - from the block_q2_K struct:
            //   Offset 0-15:   scales[16] (16 bytes, 4-bit quantized scales and mins)
            //   Offset 16-79:  qs[64]   (64 bytes, 2-bit quantized data)
            //   Offset 80-81:  d        (2 bytes, fp16 - super-block scale)
            //   Offset 82-83:  dmin     (2 bytes, fp16 - super-block min)
            //
            // Reference: block_q2_K struct
            //   typedef struct {
            //       uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
            //       uint8_t qs[QK_K/4];      // quants
            //       fp16 d;                  // super-block scale for quantized scales
            //       fp16 dmin;               // super-block scale for quantized mins
            //   } block_q2_K;
            float d = halfToFloat(*(const uint16_t *) (blockData + 80));
            float dmin = halfToFloat(*(const uint16_t *) (blockData + 82));
            const uint8_t *scales = blockData + 0;
            const uint8_t *q = blockData + 16;

            float *y = out;
            int is = 0;
            for (int n = 0; n < 256; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    uint8_t sc = scales[is++];
                    float dl = d * (sc & 0xF);
                    float ml = dmin * (sc >> 4);
                    for (int l = 0; l < 16; ++l) {
                        *y++ = dl * ((int8_t) ((q[l] >> shift) & 3)) - ml;
                    }
                    sc = scales[is++];
                    dl = d * (sc & 0xF);
                    ml = dmin * (sc >> 4);
                    for (int l = 0; l < 16; ++l) {
                        *y++ = dl * ((int8_t) ((q[l + 16] >> shift) & 3)) - ml;
                    }
                    shift += 2;
                }
                q += 32;
            }
        }

        static std::vector<float> dequantizeQ2_K(const uint8_t *data,
                                                 uint64_t numElements) {
            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t BLOCK_BYTES = 84;
            uint64_t numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::vector<float> result(numElements);
            for (uint64_t b = 0; b < numBlocks; ++b) {
                float blockOut[BLOCK_SIZE];
                dequantizeQ2_KBlock(data + b * BLOCK_BYTES, blockOut);
                uint64_t start = b * BLOCK_SIZE;
                uint64_t end = std::min(start + BLOCK_SIZE, numElements);
                for (uint64_t i = start; i < end; ++i) {
                    result[i] = blockOut[i - start];
                }
            }
            return result;
        }

        /// @brief Dequantize a Q3_K block (256 weights, 112 bytes per block).
        /// Matches the reference dequantize_row_q3_K implementation.
        static void dequantizeQ3_KBlock(const uint8_t *blockData, float *out) {
            static const uint32_t kmask1 = 0x03030303;
            static const uint32_t kmask2 = 0x0f0f0f0f;

            float d_all = halfToFloat(*(const uint16_t *) (blockData + 108));
            const uint8_t *hm = blockData + 0;
            const uint8_t *q = blockData + 32;
            const uint8_t *scales = blockData + 96;

            uint32_t aux[4];
            std::memcpy(aux, scales, 12);
            uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            const int8_t *sc = (const int8_t *) aux;
            float *y = out;
            int is = 0;
            uint8_t m = 1;
            for (int n = 0; n < 256; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    float dl = d_all * (sc[is++] - 32);
                    for (int l = 0; l < 16; ++l) {
                        *y++ = dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                    }
                    dl = d_all * (sc[is++] - 32);
                    for (int l = 0; l < 16; ++l) {
                        *y++ = dl * ((int8_t) ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                    }
                    shift += 2;
                    m <<= 1;
                }
                q += 32;
            }
        }

        static std::vector<float> dequantizeQ3_K(const uint8_t *data,
                                                 uint64_t numElements) {
            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t BLOCK_BYTES = 110;
            uint64_t numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::vector<float> result(numElements);
            for (uint64_t b = 0; b < numBlocks; ++b) {
                float blockOut[BLOCK_SIZE];
                dequantizeQ3_KBlock(data + b * BLOCK_BYTES, blockOut);
                uint64_t start = b * BLOCK_SIZE;
                uint64_t end = std::min(start + BLOCK_SIZE, numElements);
                for (uint64_t i = start; i < end; ++i) {
                    result[i] = blockOut[i - start];
                }
            }
            return result;
        }

        // Bit masks for sign extraction in IQ2/3 dequantization (8 positions in a
        // byte)
        static constexpr uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

        /// @brief Sign lookup table for IQ2/3 dequantization.
        static constexpr uint8_t ksigns_iq2xs[128] = {
                0,
                129,
                130,
                3,
                132,
                5,
                6,
                135,
                136,
                9,
                10,
                139,
                12,
                141,
                142,
                15,
                144,
                17,
                18,
                147,
                20,
                149,
                150,
                23,
                24,
                153,
                154,
                27,
                156,
                29,
                30,
                159,
                160,
                33,
                34,
                163,
                36,
                165,
                166,
                39,
                40,
                169,
                170,
                43,
                172,
                45,
                46,
                175,
                48,
                177,
                178,
                51,
                180,
                53,
                54,
                183,
                184,
                57,
                58,
                187,
                60,
                189,
                190,
                63,
                192,
                65,
                66,
                195,
                68,
                197,
                198,
                71,
                72,
                201,
                202,
                75,
                204,
                77,
                78,
                207,
                80,
                209,
                210,
                83,
                212,
                85,
                86,
                215,
                216,
                89,
                90,
                219,
                92,
                221,
                222,
                95,
                96,
                225,
                226,
                99,
                228,
                101,
                102,
                231,
                232,
                105,
                106,
                235,
                108,
                237,
                238,
                111,
                240,
                113,
                114,
                243,
                116,
                245,
                246,
                119,
                120,
                249,
                250,
                123,
                252,
                125,
                126,
                255,
        };

        // IQ2_S grid lookup table: 1024 entries, each storing 8 x 4-bit values packed
        // into a uint64_t (each byte is one value 0-15).
        static const uint64_t IQ2S_GRID[1024];

        // IQ3_XXS grid lookup table: 256 entries, each storing 4 x 4-bit values
        // packed into a uint32_t (each byte is one value 0-15).
        static const uint32_t IQ3XXS_GRID[256];

        // IQ3_S grid lookup table: 512 entries, each storing 4 x 4-bit values packed
        // into a uint32_t (each byte is one value 0-15).
        static const uint32_t IQ3S_GRID[512];

        // IQ2_XS grid lookup table: 512 entries, each storing a 16-bit grid index.
        // (ksigns_iq2xs and kmask_iq2xs are already defined inline above and reused)
        static const uint16_t iq2xs_grid[512];

        /// @brief Dequantize IQ2_S tensor data to float32.
        /// IQ2_S: 2.5625 bpw, 256 weights per block, 82 bytes per block.
        /// Block layout: d(2) + qs[64] + qh[8] + scales[8] = 82 bytes.
        static std::vector<float> dequantizeIQ2_S(const uint8_t *data,
                                                  uint64_t numElements) {
            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t BLOCK_BYTES = 82;
            uint64_t numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::vector<float> result(numElements);

            for (uint64_t b = 0; b < numBlocks; ++b) {
                const uint8_t *blockData = data + b * BLOCK_BYTES;
                float d_val = halfToFloat(*(const uint16_t *) (blockData + 0));
                const uint8_t *qs = blockData + 2;
                const uint8_t *qh = blockData + 66;
                const uint8_t *scales = blockData + 74;
                const uint8_t *signs = qs + QK_K / 8;

                uint64_t base = b * BLOCK_SIZE;
                float db[2];
                for (uint32_t ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                    db[0] = d_val * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
                    db[1] = d_val * (0.5f + (scales[ib32] >> 4)) * 0.25f;
                    for (int l = 0; l < 4; ++l) {
                        float dl = db[l / 2];
                        uint16_t gridIdx = qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
                        const uint8_t *grid =
                                reinterpret_cast<const uint8_t *>(&IQ2S_GRID[gridIdx]);
                        for (int j = 0; j < 8; ++j) {
                            float w = dl * static_cast<float>(grid[j]) *
                                      (signs[l] & kmask_iq2xs[j] ? -1.0f : 1.0f);
                            uint64_t idx = base + ib32 * 32 + l * 8 + j;
                            if (idx < numElements) {
                                result[idx] = w;
                            }
                        }
                    }
                    qs += 4;
                    signs += 4;
                }
            }
            return result;
        }

        /// @brief Dequantize IQ2_XS tensor data to float32.
        /// IQ2_XS: 2.3125 bpw, 256 weights per block, 74 bytes per block.
        /// Block layout: d(2) + qs[64] + scales[8] = 74 bytes.
        static std::vector<float> dequantizeIQ2_XS(const uint8_t *data,
                                                   uint64_t numElements) {
            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t BLOCK_BYTES = 74;
            uint64_t numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::vector<float> result(numElements);

            for (uint64_t b = 0; b < numBlocks; ++b) {
                const uint8_t *blockData = data + b * BLOCK_BYTES;
                float d_val = halfToFloat(*(const uint16_t *) (blockData + 0));
                const uint16_t *qs16 = reinterpret_cast<const uint16_t *>(blockData + 2);
                const uint8_t *scales = blockData + 66;

                uint64_t base = b * BLOCK_SIZE;
                for (uint32_t ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                    float db[2];
                    db[0] = d_val * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
                    db[1] = d_val * (0.5f + (scales[ib32] >> 4)) * 0.25f;
                    for (uint32_t l = 0; l < 4; ++l) {
                        uint16_t qval = qs16[ib32 * 4 + l];
                        uint16_t gridIdx = qval & 0x1FF;
                        uint8_t signIdx = static_cast<uint8_t>(qval >> 9);
                        uint16_t gridPacked = iq2xs_grid[gridIdx];
                        const uint8_t signs = ksigns_iq2xs[signIdx];
                        for (uint32_t j = 0; j < 8; ++j) {
                            // Extract 2-bit value from packed uint16_t
                            // Map: 0->-2, 1->-1, 2->1, 3->2
                            static const int8_t iq2xs_vals[4] = {-2, -1, 1, 2};
                            int8_t val = iq2xs_vals[(gridPacked >> (2 * j)) & 3];
                            float w = db[l / 2] * static_cast<float>(val) *
                                      (signs & kmask_iq2xs[j] ? -1.0f : 1.0f);
                            uint64_t idx = base + ib32 * 32 + l * 8 + j;
                            if (idx < numElements) {
                                result[idx] = w;
                            }
                        }
                    }
                }
            }
            return result;
        }

        /// @brief Dequantize IQ3_S tensor data to float32.
        /// IQ3_S: 3.4375 bpw, 256 weights per block, 110 bytes per block.
        /// Block layout: d(2) + qs[64] + qh[8] + signs[32] + scales[4] = 110 bytes.
        static std::vector<float> dequantizeIQ3_S(const uint8_t *data,
                                                  uint64_t numElements) {
            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t BLOCK_BYTES = 110;
            uint64_t numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::vector<float> result(numElements);

            for (uint64_t b = 0; b < numBlocks; ++b) {
                const uint8_t *blockData = data + b * BLOCK_BYTES;
                float d_val = halfToFloat(*(const uint16_t *) (blockData + 0));
                const uint8_t *qs = blockData + 2;
                const uint8_t *qh = blockData + 66;
                const uint8_t *signs = blockData + 74;
                const uint8_t *scales = blockData + 106;

                uint64_t base = b * BLOCK_SIZE;
                for (uint32_t ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
                    float db1 = d_val * (1.0f + 2.0f * (scales[ib32 / 2] & 0xf));
                    float db2 = d_val * (1.0f + 2.0f * (scales[ib32 / 2] >> 4));
                    for (int l = 0; l < 4; ++l) {
                        uint16_t gridIdx1 = qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256);
                        uint16_t gridIdx2 = qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256);
                        const uint8_t *grid1 =
                                reinterpret_cast<const uint8_t *>(&IQ3S_GRID[gridIdx1]);
                        const uint8_t *grid2 =
                                reinterpret_cast<const uint8_t *>(&IQ3S_GRID[gridIdx2]);
                        for (int j = 0; j < 4; ++j) {
                            float w1 = db1 * static_cast<float>(grid1[j]) *
                                       (signs[l] & kmask_iq2xs[j + 0] ? -1.0f : 1.0f);
                            float w2 = db1 * static_cast<float>(grid2[j]) *
                                       (signs[l] & kmask_iq2xs[j + 4] ? -1.0f : 1.0f);
                            uint64_t idx1 = base + ib32 * 32 + l * 8 + j;
                            uint64_t idx2 = base + ib32 * 32 + l * 8 + j + 4;
                            if (idx1 < numElements)
                                result[idx1] = w1;
                            if (idx2 < numElements)
                                result[idx2] = w2;
                        }
                    }
                    qs += 8;
                    signs += 4;
                    for (int l = 0; l < 4; ++l) {
                        uint16_t gridIdx1 = qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256);
                        uint16_t gridIdx2 = qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256);
                        const uint8_t *grid1 =
                                reinterpret_cast<const uint8_t *>(&IQ3S_GRID[gridIdx1]);
                        const uint8_t *grid2 =
                                reinterpret_cast<const uint8_t *>(&IQ3S_GRID[gridIdx2]);
                        for (int j = 0; j < 4; ++j) {
                            float w1 = db2 * static_cast<float>(grid1[j]) *
                                       (signs[l] & kmask_iq2xs[j + 0] ? -1.0f : 1.0f);
                            float w2 = db2 * static_cast<float>(grid2[j]) *
                                       (signs[l] & kmask_iq2xs[j + 4] ? -1.0f : 1.0f);
                            uint64_t idx1 = base + (ib32 + 1) * 32 + l * 8 + j;
                            uint64_t idx2 = base + (ib32 + 1) * 32 + l * 8 + j + 4;
                            if (idx1 < numElements)
                                result[idx1] = w1;
                            if (idx2 < numElements)
                                result[idx2] = w2;
                        }
                    }
                    qh += 2;
                    qs += 8;
                    signs += 4;
                }
            }
            return result;
        }

        /// @brief Dequantize IQ3_XXS tensor data to float32.
        /// IQ3_XXS: 3.0625 bpw, 256 weights per block, 98 bytes per block.
        /// Block layout: d(2) + qs[96] = 98 bytes.
        static std::vector<float> dequantizeIQ3_XXS(const uint8_t *data,
                                                    uint64_t numElements) {
            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t BLOCK_BYTES = 98;
            uint64_t numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::vector<float> result(numElements);

            for (uint64_t b = 0; b < numBlocks; ++b) {
                const uint8_t *blockData = data + b * BLOCK_BYTES;
                float d_val = halfToFloat(*(const uint16_t *) (blockData + 0));
                const uint8_t *qs = blockData + 2;
                const uint8_t *scales_and_signs = qs + QK_K / 4;

                uint64_t base = b * BLOCK_SIZE;
                uint32_t aux32;
                for (uint32_t ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                    std::memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
                    float db = d_val * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
                    for (int l = 0; l < 4; ++l) {
                        uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                        const uint8_t *grid1 =
                                reinterpret_cast<const uint8_t *>(&IQ3XXS_GRID[qs[2 * l + 0]]);
                        const uint8_t *grid2 =
                                reinterpret_cast<const uint8_t *>(&IQ3XXS_GRID[qs[2 * l + 1]]);
                        for (int j = 0; j < 4; ++j) {
                            float w1 = db * static_cast<float>(grid1[j]) *
                                       (signs & kmask_iq2xs[j + 0] ? -1.0f : 1.0f);
                            float w2 = db * static_cast<float>(grid2[j]) *
                                       (signs & kmask_iq2xs[j + 4] ? -1.0f : 1.0f);
                            uint64_t idx1 = base + ib32 * 32 + l * 8 + j;
                            uint64_t idx2 = base + ib32 * 32 + l * 8 + j + 4;
                            if (idx1 < numElements)
                                result[idx1] = w1;
                            if (idx2 < numElements)
                                result[idx2] = w2;
                        }
                    }
                    qs += 8;
                }
            }
            return result;
        }

        /// @brief Dequantize a single IQ3_XXS block (256 weights, 98 bytes).
        static void dequantizeIQ3_XXSBlock(const uint8_t *blockData, float *out) {
            float d_val = halfToFloat(*(const uint16_t *) (blockData + 0));
            const uint8_t *qs = blockData + 2;
            const uint8_t *scales_and_signs = qs + QK_K / 4;

            uint32_t aux32;
            for (uint32_t ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                std::memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
                float db = d_val * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
                for (int l = 0; l < 4; ++l) {
                    uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                    const uint8_t *grid1 =
                            reinterpret_cast<const uint8_t *>(&IQ3XXS_GRID[qs[2 * l + 0]]);
                    const uint8_t *grid2 =
                            reinterpret_cast<const uint8_t *>(&IQ3XXS_GRID[qs[2 * l + 1]]);
                    for (int j = 0; j < 4; ++j) {
                        out[ib32 * 32 + l * 8 + j + 0] =
                                db * static_cast<float>(grid1[j]) *
                                (signs & kmask_iq2xs[j + 0] ? -1.0f : 1.0f);
                        out[ib32 * 32 + l * 8 + j + 4] =
                                db * static_cast<float>(grid2[j]) *
                                (signs & kmask_iq2xs[j + 4] ? -1.0f : 1.0f);
                    }
                }
                qs += 8;
            }
        }

        /// @brief Dequantize a single IQ3_S block (256 weights, 110 bytes).
        static void dequantizeIQ3_SBlock(const uint8_t *blockData, float *out) {
            float d_val = halfToFloat(*(const uint16_t *) (blockData + 0));
            const uint8_t *qs = blockData + 2;
            const uint8_t *qh = blockData + 66;
            const uint8_t *signs = blockData + 74;
            const uint8_t *scales = blockData + 106;

            for (uint32_t ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
                float db1 = d_val * (1.0f + 2.0f * (scales[ib32 / 2] & 0xf));
                float db2 = d_val * (1.0f + 2.0f * (scales[ib32 / 2] >> 4));
                for (int l = 0; l < 4; ++l) {
                    uint16_t gridIdx1 = qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256);
                    uint16_t gridIdx2 = qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256);
                    const uint8_t *grid1 =
                            reinterpret_cast<const uint8_t *>(&IQ3S_GRID[gridIdx1]);
                    const uint8_t *grid2 =
                            reinterpret_cast<const uint8_t *>(&IQ3S_GRID[gridIdx2]);
                    for (int j = 0; j < 4; ++j) {
                        out[ib32 * 32 + l * 8 + j + 0] =
                                db1 * static_cast<float>(grid1[j]) *
                                (signs[l] & kmask_iq2xs[j + 0] ? -1.0f : 1.0f);
                        out[ib32 * 32 + l * 8 + j + 4] =
                                db1 * static_cast<float>(grid2[j]) *
                                (signs[l] & kmask_iq2xs[j + 4] ? -1.0f : 1.0f);
                    }
                }
                qs += 8;
                signs += 4;
                for (int l = 0; l < 4; ++l) {
                    uint16_t gridIdx1 = qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256);
                    uint16_t gridIdx2 = qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256);
                    const uint8_t *grid1 =
                            reinterpret_cast<const uint8_t *>(&IQ3S_GRID[gridIdx1]);
                    const uint8_t *grid2 =
                            reinterpret_cast<const uint8_t *>(&IQ3S_GRID[gridIdx2]);
                    for (int j = 0; j < 4; ++j) {
                        out[(ib32 + 1) * 32 + l * 8 + j + 0] =
                                db2 * static_cast<float>(grid1[j]) *
                                (signs[l] & kmask_iq2xs[j + 0] ? -1.0f : 1.0f);
                        out[(ib32 + 1) * 32 + l * 8 + j + 4] =
                                db2 * static_cast<float>(grid2[j]) *
                                (signs[l] & kmask_iq2xs[j + 4] ? -1.0f : 1.0f);
                    }
                }
                qh += 2;
                qs += 8;
                signs += 4;
            }
        }

        /// @brief Dequantize a single IQ2_S block (256 weights, 82 bytes).
        static void dequantizeIQ2_SBlock(const uint8_t *blockData, float *out) {
            float d_val = halfToFloat(*(const uint16_t *) (blockData + 0));
            const uint8_t *qs = blockData + 2;
            const uint8_t *qh = blockData + 66;
            const uint8_t *scales = blockData + 74;
            const uint8_t *signs = qs + QK_K / 8;

            float db[2];
            for (uint32_t ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                db[0] = d_val * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
                db[1] = d_val * (0.5f + (scales[ib32] >> 4)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    float dl = db[l / 2];
                    uint16_t gridIdx = qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
                    const uint8_t *grid =
                            reinterpret_cast<const uint8_t *>(&IQ2S_GRID[gridIdx]);
                    for (int j = 0; j < 8; ++j) {
                        out[ib32 * 32 + l * 8 + j] =
                                dl * static_cast<float>(grid[j]) *
                                (signs[l] & kmask_iq2xs[j] ? -1.0f : 1.0f);
                    }
                }
                qs += 4;
                signs += 4;
            }
        }

        /// @brief Dequantize a single IQ2_XS block (256 weights, 74 bytes).
        /// Block layout: d(fp16,2) + qs[64] + scales[8] = 74 bytes.
        /// IQ2_XS uses 16-bit qs entries (32 entries) with 9-bit grid index + 8-bit sign.
        static void dequantizeIQ2_XSBlock(const uint8_t *blockData, float *out) {
            float d_val = halfToFloat(*(const uint16_t *) (blockData + 0));
            const uint16_t *qs16 = reinterpret_cast<const uint16_t *>(blockData + 2);
            const uint8_t *scales = blockData + 66;

            float db[2];
            for (uint32_t ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                db[0] = d_val * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
                db[1] = d_val * (0.5f + (scales[ib32] >> 4)) * 0.25f;
                for (uint32_t l = 0; l < 4; ++l) {
                    uint16_t qval = qs16[ib32 * 4 + l];
                    // Grid index: lower 9 bits
                    uint16_t gridIdx = qval & 0x1FF;
                    // Sign index: upper 7 bits (shifted right by 9)
                    uint8_t signIdx = static_cast<uint8_t>(qval >> 9);
                    uint16_t gridPacked = iq2xs_grid[gridIdx];
                    const uint8_t signs = ksigns_iq2xs[signIdx];
                    for (uint32_t j = 0; j < 8; ++j) {
                        // Extract 2-bit value from packed uint16_t
                        // Map: 0->-2, 1->-1, 2->1, 3->2
                        static const int8_t iq2xs_vals[4] = {-2, -1, 1, 2};
                        int8_t val = iq2xs_vals[(gridPacked >> (2 * j)) & 3];
                        out[ib32 * 32 + l * 8 + j] =
                                db[l / 2] * static_cast<float>(val) *
                                (signs & kmask_iq2xs[j] ? -1.0f : 1.0f);
                    }
                }
            }
        }

        /// @brief Dequantize a single block of quantized data into a provided float
        /// buffer.
        ///        Dispatches to the correct block-level dequantizer based on type.
        /// @param ggmlType  The quantization type.
        /// @param blockData Pointer to the block's quantized data.
        /// @param out       Output buffer (must hold at least blockSize floats).
        /// @param blockSize Number of elements in the block (e.g., 256 for K-quant,
        /// 32 for Q5_1).
        static void dequantizeBlock(uint32_t ggmlType, const uint8_t *blockData,
                                    float *out, uint32_t blockSize) {
            switch (ggmlType) {
                case GGML_TYPE_F32:
                    std::memcpy(out, blockData, blockSize * sizeof(float));
                    break;
                case GGML_TYPE_Q5_0:
                    dequantizeQ5_0Block(blockData, out);
                    break;
                case GGML_TYPE_Q5_1:
                    dequantizeQ5_1Block(blockData, out);
                    break;
                case GGML_TYPE_Q8_0:
                    dequantizeQ8_0Block(blockData, out);
                    break;
                case GGML_TYPE_Q5_K:
                    dequantizeQ5_KBlock(blockData, out);
                    break;
                case GGML_TYPE_Q4_K:
                    dequantizeQ4_KBlock(blockData, out);
                    break;
                case GGML_TYPE_Q6_K:
                    dequantizeQ6_KBlock(blockData, out);
                    break;
                case GGML_TYPE_Q2_K:
                    dequantizeQ2_KBlock(blockData, out);
                    break;
                case GGML_TYPE_Q3_K:
                    dequantizeQ3_KBlock(blockData, out);
                    break;
                case GGML_TYPE_IQ3_XXS:
                    dequantizeIQ3_XXSBlock(blockData, out);
                    break;
                case GGML_TYPE_IQ3_S:
                    dequantizeIQ3_SBlock(blockData, out);
                    break;
                case GGML_TYPE_IQ2_S:
                    dequantizeIQ2_SBlock(blockData, out);
                    break;
                case GGML_TYPE_IQ2_XS:
                    dequantizeIQ2_XSBlock(blockData, out);
                    break;
                default: {
                    auto deq = dequantize(ggmlType, blockData, blockSize);
                    uint32_t n = std::min(blockSize, static_cast<uint32_t>(deq.size()));
                    for (uint32_t i = 0; i < n; ++i) {
                        out[i] = deq[i];
                    }
                } break;
            }
        }

        // -----------------------------------------------------------------------
        // Fused quantized dot product functions
        //
        // These compute dot(x, dequantize(block)) directly during dequantization,
        // eliminating the float blockOut[256] temporary array and the extra memory
        // pass. For each group of weights sharing the same (scale, min), we
        // accumulate:
        //   sum_x_quant += x[i] * quant
        //   sum_x += x[i]
        // then compute: dot += scale * sum_x_quant - min * sum_x
        //
        // This is mathematically equivalent to:
        //   val = scale * quant - min
        //   dot += x[i] * val
        // but avoids writing 256 floats to the stack and reading them back.
        // -----------------------------------------------------------------------

        /// @brief Fused dot product for Q2_K block (256 weights, 84 bytes).
        ///
        /// Delegates to the SIMD-dispatched implementation in SIMDMatMulVec.cpp
        /// which uses AVX2 when available (runtime dispatch, cached after first call).
        /// The scalar fallback is mathematically identical to the original inline
        /// implementation.
        static float dotProductQ2_K(const uint8_t *blockData, const float *x) {
            return dotProductQ2_K_SIMD(blockData, x);
        }

        /// @brief Fused dot product for Q3_K block (256 weights, 112 bytes).
        static float dotProductQ3_K(const uint8_t *blockData, const float *x) {
            static const uint32_t kmask1 = 0x03030303;
            static const uint32_t kmask2 = 0x0f0f0f0f;

            float d_all = halfToFloat(*(const uint16_t *) (blockData + 108));
            const uint8_t *hm = blockData + 0;
            const uint8_t *q = blockData + 32;
            const uint8_t *scales = blockData + 96;

            uint32_t aux[4];
            std::memcpy(aux, scales, 12);
            uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            const int8_t *sc = (const int8_t *) aux;
            double dot = 0.0;
            int is = 0;
            uint8_t m = 1;
            for (int n = 0; n < 256; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    float dl = d_all * (sc[is++] - 32);
                    double sum_xq = 0.0, sum_x = 0.0;
                    for (int l = 0; l < 16; ++l) {
                        float xv = x[n + j * 32 + l];
                        int8_t quant = (int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
                        sum_xq += static_cast<double>(xv) * quant;
                        sum_x += static_cast<double>(xv);
                    }
                    dot += static_cast<double>(dl) * sum_xq;

                    dl = d_all * (sc[is++] - 32);
                    sum_xq = 0.0;
                    for (int l = 0; l < 16; ++l) {
                        float xv = x[n + j * 32 + 16 + l];
                        int8_t quant = (int8_t) ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
                        sum_xq += static_cast<double>(xv) * quant;
                    }
                    dot += static_cast<double>(dl) * sum_xq;

                    shift += 2;
                    m <<= 1;
                }
                q += 32;
            }
            return static_cast<float>(dot);
        }

        /// @brief Fused dot product for Q4_K block (256 weights, 144 bytes).
        static float dotProductQ4_K(const uint8_t *blockData, const float *x) {
            float d = halfToFloat(*(const uint16_t *) (blockData + 0));
            float dmin = halfToFloat(*(const uint16_t *) (blockData + 2));

            const uint8_t *scales = blockData + 4;
            const uint8_t *qs = blockData + 16;

            auto getScaleMin = [](int j, const uint8_t *q, uint8_t *d_out,
                                  uint8_t *m_out) {
                if (j < 4) {
                    *d_out = q[j] & 63;
                    *m_out = q[j + 4] & 63;
                } else {
                    *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                    *m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
                }
            };

            double dot = 0.0;
            int is = 0;
            uint8_t sc, m;
            int outIdx = 0;

            for (int j = 0; j < 256; j += 64) {
                getScaleMin(is + 0, scales, &sc, &m);
                float d1 = d * static_cast<float>(sc);
                float m1 = dmin * static_cast<float>(m);

                getScaleMin(is + 1, scales, &sc, &m);
                float d2 = d * static_cast<float>(sc);
                float m2 = dmin * static_cast<float>(m);

                // First sub-block of 32: low 4 bits
                double sum_xq = 0.0, sum_x = 0.0;
                for (int l = 0; l < 32; ++l) {
                    float xv = x[outIdx + l];
                    uint8_t quant = qs[l] & 0xF;
                    sum_xq += static_cast<double>(xv) * quant;
                    sum_x += static_cast<double>(xv);
                }
                dot += static_cast<double>(d1) * sum_xq - static_cast<double>(m1) * sum_x;

                // Second sub-block of 32: high 4 bits
                sum_xq = 0.0;
                sum_x = 0.0;
                for (int l = 0; l < 32; ++l) {
                    float xv = x[outIdx + 32 + l];
                    uint8_t quant = qs[l] >> 4;
                    sum_xq += static_cast<double>(xv) * quant;
                    sum_x += static_cast<double>(xv);
                }
                dot += static_cast<double>(d2) * sum_xq - static_cast<double>(m2) * sum_x;

                qs += 32;
                is += 2;
                outIdx += 64;
            }
            return static_cast<float>(dot);
        }

        /// @brief Fused dot product for Q5_K block (256 weights, 176 bytes).
        static float dotProductQ5_K(const uint8_t *blockData, const float *x) {
            float d = halfToFloat(*(const uint16_t *) (blockData + 0));
            float dmin = halfToFloat(*(const uint16_t *) (blockData + 2));

            const uint8_t *scales = blockData + 4;
            const uint8_t *qh = blockData + 16;
            const uint8_t *qs = blockData + 48;

            auto getScaleMin = [](int j, const uint8_t *q, uint8_t *d_out,
                                  uint8_t *m_out) {
                if (j < 4) {
                    *d_out = q[j] & 63;
                    *m_out = q[j + 4] & 63;
                } else {
                    *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                    *m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
                }
            };

            double dot = 0.0;
            int is = 0;
            uint8_t sc, m;
            uint8_t u1 = 1, u2 = 2;
            int outIdx = 0;

            for (int j = 0; j < 256; j += 64) {
                getScaleMin(is + 0, scales, &sc, &m);
                float d1 = d * static_cast<float>(sc);
                float m1 = dmin * static_cast<float>(m);

                getScaleMin(is + 1, scales, &sc, &m);
                float d2 = d * static_cast<float>(sc);
                float m2 = dmin * static_cast<float>(m);

                // First sub-block of 32: low 4 bits + high bit from qh (u1)
                double sum_xq = 0.0, sum_x = 0.0;
                for (int l = 0; l < 32; ++l) {
                    float xv = x[outIdx + l];
                    uint8_t q5 = (qs[l] & 0xF) | ((qh[l] & u1) ? 16 : 0);
                    sum_xq += static_cast<double>(xv) * q5;
                    sum_x += static_cast<double>(xv);
                }
                dot += static_cast<double>(d1) * sum_xq - static_cast<double>(m1) * sum_x;

                // Second sub-block of 32: high 4 bits + high bit from qh (u2)
                sum_xq = 0.0;
                sum_x = 0.0;
                for (int l = 0; l < 32; ++l) {
                    float xv = x[outIdx + 32 + l];
                    uint8_t q5 = (qs[l] >> 4) | ((qh[l] & u2) ? 16 : 0);
                    sum_xq += static_cast<double>(xv) * q5;
                    sum_x += static_cast<double>(xv);
                }
                dot += static_cast<double>(d2) * sum_xq - static_cast<double>(m2) * sum_x;

                qs += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
                outIdx += 64;
            }
            return static_cast<float>(dot);
        }

        /// @brief Fused dot product for Q6_K block (256 weights, 210 bytes).
        static float dotProductQ6_K(const uint8_t *blockData, const float *x) {
            const uint8_t *ql = blockData + 0;
            const uint8_t *qh = blockData + 128;
            const int8_t *sc = reinterpret_cast<const int8_t *>(blockData + 192);
            float d = halfToFloat(*(const uint16_t *) (blockData + 208));

            double dot = 0.0;
            int outIdx = 0;
            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    const int8_t q1 = (int8_t) ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int8_t q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int8_t q3 = (int8_t) ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int8_t q4 = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                    dot += static_cast<double>(d) * sc[is + 0] * q1 * x[outIdx + l + 0];
                    dot += static_cast<double>(d) * sc[is + 2] * q2 * x[outIdx + l + 32];
                    dot += static_cast<double>(d) * sc[is + 4] * q3 * x[outIdx + l + 64];
                    dot += static_cast<double>(d) * sc[is + 6] * q4 * x[outIdx + l + 96];
                }
                outIdx += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
            return static_cast<float>(dot);
        }

        /// @brief Fused dot product for Q8_0 block (32 weights, 34 bytes).
        static float dotProductQ8_0(const uint8_t *blockData, const float *x) {
            float d_val = halfToFloat(*(const uint16_t *) (blockData + 0));
            const int8_t *q = reinterpret_cast<const int8_t *>(blockData + 2);

            double dot = 0.0;
            for (int i = 0; i < 32; ++i) {
                dot += static_cast<double>(x[i]) * q[i];
            }
            return static_cast<float>(dot) * d_val;
        }

        /// @brief Fused dot product for Q5_0 block (32 weights, 22 bytes).
        static float dotProductQ5_0(const uint8_t *blockData, const float *x) {
            float d_val = halfToFloat(*(const uint16_t *) (blockData + 0));
            uint32_t qh;
            std::memcpy(&qh, blockData + 2, sizeof(uint32_t));
            const uint8_t *qs = blockData + 6;

            double dot = 0.0;
            for (int j = 0; j < 16; ++j) {
                const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
                const int32_t x0 = ((qs[j] & 0x0F) | xh_0) - 16;
                const int32_t x1 = ((qs[j] >> 4) | xh_1) - 16;
                dot += static_cast<double>(x[j + 0]) * x0;
                dot += static_cast<double>(x[j + 16]) * x1;
            }
            return static_cast<float>(dot) * d_val;
        }

        /// @brief Fused dot product for Q5_1 block (32 weights, 24 bytes).
        static float dotProductQ5_1(const uint8_t *blockData, const float *x) {
            float d_val = halfToFloat(*(const uint16_t *) (blockData + 0));
            float m_val = halfToFloat(*(const uint16_t *) (blockData + 2));
            uint32_t qh = *(const uint32_t *) (blockData + 4);
            const uint8_t *ql = blockData + 8;

            double sum_xq = 0.0, sum_x = 0.0;
            for (int i = 0; i < 32; ++i) {
                uint8_t low4 = (ql[i / 2] >> (4 * (i % 2))) & 0xF;
                uint8_t highBit = (qh >> i) & 1;
                uint8_t q = low4 | (highBit << 4);
                float xv = x[i];
                sum_xq += static_cast<double>(xv) * q;
                sum_x += static_cast<double>(xv);
            }
            return static_cast<float>(static_cast<double>(d_val) * sum_xq + static_cast<double>(m_val) * sum_x);
        }

        /// @brief Dispatch to type-specific fused dot product.
        /// @return dot(x, dequantize(blockData)) for one block.
        /// @note The type-specific functions always process the full block
        ///       (e.g., 256 elements for K-quant types, 32 for Q8_0/Q5_0/Q5_1).
        ///       If blockSize < fullBlockSize (partial last block), we fall back
        ///       to dequantize+dot to avoid reading garbage from x beyond the
        ///       valid elements. In practice, this only happens in unit tests
        ///       with artificially small matrices; real model matrices always
        ///       have full blocks.
        static float dotProductFused(uint32_t ggmlType, const uint8_t *blockData,
                                     const float *x, uint32_t blockSize) {
            uint32_t fullBlockSize = ggmlBlockSize(ggmlType);
            if (blockSize < fullBlockSize) {
                // Partial block: fall back to dequantize then dot product
                float blockOut[256];
                dequantizeBlock(ggmlType, blockData, blockOut, blockSize);
                return dotProductFMA(x, blockOut, blockSize);
            }
            switch (ggmlType) {
                case GGML_TYPE_Q2_K:
                    return dotProductQ2_K(blockData, x);
                case GGML_TYPE_Q3_K:
                    return dotProductQ3_K(blockData, x);
                case GGML_TYPE_Q4_K:
                    return dotProductQ4_K(blockData, x);
                case GGML_TYPE_Q5_K:
                    return dotProductQ5_K(blockData, x);
                case GGML_TYPE_Q6_K:
                    return dotProductQ6_K(blockData, x);
                case GGML_TYPE_Q8_0:
                    return dotProductQ8_0(blockData, x);
                case GGML_TYPE_Q5_0:
                    return dotProductQ5_0(blockData, x);
                case GGML_TYPE_Q5_1:
                    return dotProductQ5_1(blockData, x);
                default: {
                    // Fallback: dequantize then dot product
                    float blockOut[256];
                    dequantizeBlock(ggmlType, blockData, blockOut, blockSize);
                    return dotProductFMA(x, blockOut, blockSize);
                }
            }
        }

        /// @brief Type-erased function for fused dot product of one full block.
        /// The function processes exactly blockSize elements (e.g., 256 for K-quant).
        /// Returns nullptr for types without a dedicated fused dot product
        /// (will fall back to dequantize+dot).
        static std::function<float(const uint8_t *, const float *)> getDotProductFunc(uint32_t ggmlType) {
            switch (ggmlType) {
                case GGML_TYPE_Q2_K:
                    return dotProductQ2_K;
                case GGML_TYPE_Q3_K:
                    return dotProductQ3_K;
                case GGML_TYPE_Q4_K:
                    return dotProductQ4_K;
                case GGML_TYPE_Q5_K:
                    return dotProductQ5_K;
                case GGML_TYPE_Q6_K:
                    return dotProductQ6_K;
                case GGML_TYPE_Q8_0:
                    return dotProductQ8_0;
                case GGML_TYPE_Q5_0:
                    return dotProductQ5_0;
                case GGML_TYPE_Q5_1:
                    return dotProductQ5_1;
                default:
                    return nullptr;
            }
        }

        /// @brief Compute y = x * W where W is a quantized matrix stored in
        /// GGUF format as (rows x cols) = (out_features x in_features) row-major.
        ///
        /// Uses fused quantized dot product to eliminate the float blockOut[256]
        /// temporary and the extra memory pass.
        ///
        /// The GGUF file stores weight matrices in row-major order, so
        /// W[j][i] = data[j * cols + i] where j indexes output features (rows)
        /// and i indexes input features (cols).
        ///
        /// The computation is: y_j = sum_i x[i] * W[j][i] for j in [0, rows), i in [0, cols)
        ///
        /// The quantized data is stored block-by-block per output row:
        ///   data[j * blocksPerRow * typeSize + b * typeSize] = block b of output row j
        /// where blocksPerRow = ceil(cols / blockSize).
        ///
        /// For each output row j, we dequantize its blocks and compute the dot
        /// product with the input vector x:
        ///   result[j] = sum_b sum_k x[start_b + k] * blockOut[k]
        ///   where start_b = b * blockSize
        static void matMulVecFused(uint32_t ggmlType, const uint8_t *data,
                                   const float *x, uint32_t rows, uint32_t cols,
                                   float *result) {
            uint32_t blockSize = ggmlBlockSize(ggmlType);
            uint32_t typeSize = ggmlTypeSize(ggmlType);

            if (blockSize == 0 || typeSize == 0) {
                // Fallback: full dequantize then matrix multiply
                // W is (rows x cols) row-major: W[j][i] = deq[j * cols + i]
                auto deq = dequantize(ggmlType, data, static_cast<uint64_t>(rows) * cols);
                if (deq.empty()) {
                    std::memset(result, 0, static_cast<size_t>(rows) * sizeof(float));
                    return;
                }
                for (uint32_t j = 0; j < rows; ++j) {
                    double dot = 0.0;
                    for (uint32_t i = 0; i < cols; ++i) {
                        dot += static_cast<double>(x[i]) * deq[static_cast<size_t>(j) * cols + i];
                    }
                    result[j] = static_cast<float>(dot);
                }
                return;
            }

            // Weight matrix is stored as (rows x cols) row-major in quantized block form.
            // Each output row j has `cols` elements stored in blocksPerRow quantized blocks.
            //   data[j * blocksPerRow * typeSize + b * typeSize] = block b of output row j
            // where blocksPerRow = ceil(cols / blockSize).
            //
            // We compute: y_j = sum_i x[i] * W[j][i] for j in [0, rows), i in [0, cols)
            //
            // For each output row j, dequantize its blocks and compute the dot product.
            uint32_t blocksPerRow = (cols + blockSize - 1) / blockSize;
            uint64_t rowStrideBytes = static_cast<uint64_t>(blocksPerRow) * typeSize;

            // Sequential loop over output rows.
            // This function is always called from within a parallelFor lambda
            // (e.g. the outer loop over tokens in Model::forward), so using
            // ThreadPool::parallelFor here would cause massive overhead from
            // repeatedly waking/synchronizing workers for each mat-vec call.
            // With 6+ mat-vec calls per layer × 28 layers = 168+ calls per
            // forward pass, the overhead of waking 7 workers each time is
            // enormous. Instead, we run sequentially — the outer token-level
            // parallelism is sufficient.
            //
            // OPTIMIZATION: Use fused quantized dot product to eliminate the
            // float blockOut[256] temporary and the extra memory pass.
            for (uint32_t j = 0; j < rows; ++j) {
                const uint8_t *rowData = data + static_cast<uint64_t>(j) * rowStrideBytes;
                double dot = 0.0;

                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    const uint8_t *blockData = rowData + static_cast<uint64_t>(b) * typeSize;
                    uint32_t start = b * blockSize;
                    uint32_t n = std::min(blockSize, cols - start);

                    // Fused quantized dot product: dequantize and dot in one pass
                    dot += static_cast<double>(dotProductFused(ggmlType, blockData, x + start, n));
                }
                result[j] = static_cast<float>(dot);
            }
        }

        /// @brief Cache-blocked version of matMulVecFused.
        ///
        /// Instead of processing all blocks for one output row at a time (which
        /// reads the entire x vector for each row), this processes one block
        /// across all output rows. This keeps the x segment (blockSize floats)
        /// in L1 cache and improves temporal locality.
        ///
        /// For large matrices (e.g., ffnGate: 8960×1536), the x vector is 6KB
        /// and is read 8960 times in the non-blocked version (53MB of x reads).
        /// The cache-blocked version reads x once (6KB stays in L1).
        ///
        /// Mathematically equivalent to matMulVecFused (sum over blocks is
        /// commutative), so results are bit-identical.
        static void matMulVecFusedCacheBlocked(uint32_t ggmlType, const uint8_t *data,
                                               const float *x, uint32_t rows,
                                               uint32_t cols, float *result) {
            uint32_t blockSize = ggmlBlockSize(ggmlType);
            uint32_t typeSize = ggmlTypeSize(ggmlType);

            if (blockSize == 0 || typeSize == 0) {
                // Fallback to non-blocked version
                matMulVecFused(ggmlType, data, x, rows, cols, result);
                return;
            }

            uint32_t blocksPerRow = (cols + blockSize - 1) / blockSize;
            uint64_t rowStrideBytes = static_cast<uint64_t>(blocksPerRow) * typeSize;

            // Initialize result to zero
            std::memset(result, 0, static_cast<size_t>(rows) * sizeof(float));

            // Parallelize over output rows. Each thread owns a contiguous range of
            // rows and iterates over all block columns for those rows, keeping the
            // x segment in L1 per-thread. For single-token generation (seqLen==1)
            // the outer token-level parallelFor is bypassed, making this the only
            // source of parallelism (uses all threads). For seqLen>1 the ThreadPool
            // re-entrancy guard runs this sequentially.
            ThreadPool::instance().parallelFor(0, rows, [&](uint32_t j) {
                const uint8_t *rowData = data + static_cast<uint64_t>(j) * rowStrideBytes;
                double dot = 0.0;
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    const uint8_t *blockData = rowData + static_cast<uint64_t>(b) * typeSize;
                    uint32_t start = b * blockSize;
                    uint32_t n = std::min(blockSize, cols - start);
                    // Fused quantized dot product for this block
                    dot += static_cast<double>(dotProductFused(ggmlType, blockData, x + start, n));
                }
                result[j] = static_cast<float>(dot);
            });
        }

        /// @brief Dequantize any supported GGML type to float32.
        /// @return Float vector of dequantized values, or empty if type is
        /// unsupported.
        static std::vector<float> dequantize(uint32_t ggmlType, const uint8_t *data,
                                             uint64_t numElements) {
            switch (ggmlType) {
                case GGML_TYPE_F32: {
                    std::vector<float> result(numElements);
                    std::memcpy(result.data(), data, numElements * sizeof(float));
                    return result;
                }
                case GGML_TYPE_Q5_0:
                    return dequantizeQ5_0(data, numElements);
                case GGML_TYPE_Q5_1:
                    return dequantizeQ5_1(data, numElements);
                case GGML_TYPE_Q8_0:
                    return dequantizeQ8_0(data, numElements);
                case GGML_TYPE_Q5_K:
                    return dequantizeQ5_K(data, numElements);
                case GGML_TYPE_Q4_K:
                    return dequantizeQ4_K(data, numElements);
                case GGML_TYPE_Q6_K:
                    return dequantizeQ6_K(data, numElements);
                case GGML_TYPE_Q2_K:
                    return dequantizeQ2_K(data, numElements);
                case GGML_TYPE_Q3_K:
                    return dequantizeQ3_K(data, numElements);
                case GGML_TYPE_IQ2_S:
                    return dequantizeIQ2_S(data, numElements);
                case GGML_TYPE_IQ2_XS:
                    return dequantizeIQ2_XS(data, numElements);
                case GGML_TYPE_IQ3_S:
                    return dequantizeIQ3_S(data, numElements);
                case GGML_TYPE_IQ3_XXS:
                    return dequantizeIQ3_XXS(data, numElements);
                default:
                    return {};// Unsupported
            }
        }

        /// @brief Dequantize any supported GGML type directly to FP16 (uint16_t).
        ///
        /// Unlike dequantize() which allocates a full F32 vector, this function
        /// dequantizes one block at a time into a small stack buffer, then converts
        /// each value to FP16 inline. This avoids the large intermediate F32
        /// allocation, halving memory bandwidth for dequantized weight storage.
        ///
        /// @return Vector of FP16 values, or empty if type is unsupported.
        static std::vector<uint16_t> dequantizeToF16(uint32_t ggmlType,
                                                     const uint8_t *data,
                                                     uint64_t numElements) {
            // Determine block size for this type
            uint32_t blockSize = ggmlBlockSize(ggmlType);
            uint32_t typeSize = ggmlTypeSize(ggmlType);
            if (blockSize == 0 || typeSize == 0) {
                return {};// Unsupported type
            }

            uint64_t numBlocks = (numElements + blockSize - 1) / blockSize;
            std::vector<uint16_t> result(numElements);

            // Stack buffer for one block of floats (max 256 for K-quant)
            float blockBuf[256];

            for (uint64_t b = 0; b < numBlocks; ++b) {
                uint64_t start = b * blockSize;
                uint64_t end = std::min(start + blockSize, numElements);
                uint32_t n = static_cast<uint32_t>(end - start);

                // Dequantize one block to float buffer
                dequantizeBlock(ggmlType, data + b * typeSize, blockBuf, n);

                // Convert each float to FP16
                for (uint32_t i = 0; i < n; ++i) {
                    result[start + i] = floatToHalf(blockBuf[i]);
                }
            }

            return result;
        }

        /// @brief Pre-pack a Q2_K matrix for gather-free SIMD access.
        ///
        /// The original Q2_K block stores 256 2-bit values packed into 64 bytes (4 values
        /// per byte). The AVX2 kernel spends ~10 instructions per group extracting these
        /// 2-bit values (shift, mask, expand, pack). With 8 groups per block, that's 80
        /// instructions per block just for bit extraction.
        ///
        /// Pre-packing expands the 2-bit values to full bytes (0-3) in element order,
        /// so the SIMD kernel can load them with a single _mm_loadu_si128 per group.
        ///
        /// Pre-packed block format (276 bytes):
        ///   Offset 0-15:   scales[16]   (copied from original)
        ///   Offset 16-17:  d            (fp16, copied from original)
        ///   Offset 18-19:  dmin         (fp16, copied from original)
        ///   Offset 20-275: qs_expanded[256] (each byte is 0-3, in element order)
        ///
        /// Memory overhead: 276/84 ≈ 3.29× per block. For ffnGate (8960×1536) and
        /// ffnUp (8960×1536) in Qwen2.5-Coder-7B, this adds ~28.3 MB total.
        ///
        /// @param data        Raw Q2_K quantized data
        /// @param numElements Total number of elements in the matrix
        /// @return Pre-packed data vector, or empty if numElements is 0
        static std::vector<uint8_t> prepackQ2_K(const uint8_t *data,
                                                uint64_t numElements) {
            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t BLOCK_BYTES = 84;
            static constexpr uint32_t PREPACKED_BLOCK_BYTES = 276;

            if (numElements == 0) {
                return {};
            }

            uint64_t numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::vector<uint8_t> result(numBlocks * PREPACKED_BLOCK_BYTES);

            for (uint64_t b = 0; b < numBlocks; ++b) {
                const uint8_t *src = data + b * BLOCK_BYTES;
                uint8_t *dst = result.data() + b * PREPACKED_BLOCK_BYTES;

                // Copy scales[16]
                std::memcpy(dst, src, 16);

                // Copy d (fp16) at offset 16
                std::memcpy(dst + 16, src + 80, 2);

                // Copy dmin (fp16) at offset 18
                std::memcpy(dst + 18, src + 82, 2);

                // Expand qs[64] to qs_expanded[256] in element order
                // Element order matches the dequantizeQ2_KBlock iteration:
                //   For n=0,128:
                //     For j=0..3 (shift = j*2):
                //       Group 0: qs[0..15] bits [shift:shift+1] -> elements [n + j*32 + 0..15]
                //       Group 1: qs[16..31] bits [shift:shift+1] -> elements [n + j*32 + 16..31]
                //     qs += 32
                const uint8_t *qs = src + 16;
                uint8_t *qs_expanded = dst + 20;

                for (int n = 0; n < 256; n += 128) {
                    for (int j = 0; j < 4; ++j) {
                        int shift = j * 2;
                        int base = n + j * 32;

                        // Group 0: qs[0..15] bits [shift:shift+1]
                        for (int l = 0; l < 16; ++l) {
                            qs_expanded[base + l] = (qs[l] >> shift) & 3;
                        }
                        // Group 1: qs[16..31] bits [shift:shift+1]
                        for (int l = 0; l < 16; ++l) {
                            qs_expanded[base + 16 + l] = (qs[l + 16] >> shift) & 3;
                        }
                    }
                    qs += 32;
                }
            }

            return result;
        }

        /// @brief Dequantize one pre-packed Q2_K block to float (for partial blocks).
        static void dequantizeQ2_K_PrePackedBlock(const uint8_t *prepackedBlock,
                                                  float *out, uint32_t n) {
            float d = halfToFloat(*(const uint16_t *) (prepackedBlock + 16));
            float dmin = halfToFloat(*(const uint16_t *) (prepackedBlock + 18));
            const uint8_t *scales = prepackedBlock;
            const uint8_t *qs_expanded = prepackedBlock + 20;

            for (uint32_t i = 0; i < n; ++i) {
                uint8_t sc = scales[i / 16];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                int q = qs_expanded[i];
                out[i] = dl * static_cast<float>(q) - ml;
            }
        }

        /// @brief Matrix-vector multiply using pre-packed Q2_K data.
        ///
        /// Uses the pre-packed dot product kernel which eliminates the 2-bit extraction
        /// overhead by loading pre-expanded byte values directly.
        ///
        /// @param prepackedData Pre-packed Q2_K data (from prepackQ2_K)
        /// @param x             Input vector (size cols)
        /// @param rows          Number of output rows
        /// @param cols          Number of input columns
        /// @param result        Output buffer (size rows)
        static void matMulVecFusedQ2_K_PrePacked(const uint8_t *prepackedData,
                                                 const float *x, uint32_t rows,
                                                 uint32_t cols, float *result) {
            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t PREPACKED_BLOCK_BYTES = 276;

            uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            uint64_t rowStrideBytes = static_cast<uint64_t>(blocksPerRow) * PREPACKED_BLOCK_BYTES;

            for (uint32_t j = 0; j < rows; ++j) {
                const uint8_t *rowData = prepackedData + static_cast<uint64_t>(j) * rowStrideBytes;
                double dot = 0.0;

                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    const uint8_t *blockData = rowData + static_cast<uint64_t>(b) * PREPACKED_BLOCK_BYTES;
                    uint32_t start = b * BLOCK_SIZE;
                    uint32_t n = std::min(BLOCK_SIZE, cols - start);

                    if (n < BLOCK_SIZE) {
                        // Partial block: dequantize then dot product
                        float blockOut[256];
                        dequantizeQ2_K_PrePackedBlock(blockData, blockOut, n);
                        for (uint32_t i = 0; i < n; ++i) {
                            dot += static_cast<double>(x[start + i]) * blockOut[i];
                        }
                    } else {
                        // Full block: use pre-packed SIMD kernel
                        dot += static_cast<double>(dotProductQ2_K_PrePacked_SIMD(blockData, x + start));
                    }
                }
                result[j] = static_cast<float>(dot);
            }
        }

        /// @brief Quantize a float vector x into Q8_K blocks
        /// @param x        Input vector (size >= numElements)
        /// @param numElements Number of elements to quantize (must be multiple of 256)
        /// @param out      Output Q8_K blocks (size numElements/256)
        static void quantizeQ8K(const float *x, uint32_t numElements,
                                Q8KBlock *out) {
            static constexpr uint32_t BLOCK_SIZE = 256;
            uint32_t numBlocks = numElements / BLOCK_SIZE;

            for (uint32_t b = 0; b < numBlocks; ++b) {
                const float *xb = x + static_cast<uint64_t>(b) * BLOCK_SIZE;
                Q8KBlock &blk = out[b];

                float max = 0.0f;
                float amax = 0.0f;
                for (uint32_t j = 0; j < BLOCK_SIZE; ++j) {
                    float ax = std::fabs(xb[j]);
                    if (ax > amax) {
                        amax = ax;
                        max = xb[j];
                    }
                }
                if (amax == 0.0f) {
                    blk.d = 0.0f;
                    std::memset(blk.qs, 0, BLOCK_SIZE);
                    std::memset(blk.bsums, 0, sizeof(blk.bsums));
                    continue;
                }
                const float iscale = -127.0f / max;
                for (uint32_t j = 0; j < BLOCK_SIZE; ++j) {
                    int v = static_cast<int>(std::lrintf(iscale * xb[j]));
                    blk.qs[j] = static_cast<int8_t>(std::min(127, v));
                }
                for (uint32_t j = 0; j < BLOCK_SIZE / 16; ++j) {
                    int sum = 0;
                    for (uint32_t ii = 0; ii < 16; ++ii) {
                        sum += blk.qs[j * 16 + ii];
                    }
                    blk.bsums[j] = static_cast<int16_t>(sum);
                }
                blk.d = 1.0f / iscale;
            }
        }


        /// @brief Re-quantize a full row of floats to COMPACT Q2_K blocks (84
        /// bytes per 256-element block), matching the dequantizeQ2_KBlock layout:
        /// scales[16] (4-bit scale low nibble / min high nibble), qs[64] (2-bit
        /// planes), fp16 d at +80, fp16 dmin at +82.
        ///
        /// This is a load-time byte-reduction primitive (plan §7 "re-quant to a
        /// lower-bit K-quant at load time"): matrices streamed whole every token
        /// (separate LM head Q6_K→Q2_K: 210→84 B/block; ffnDown Q3_K→Q2_K:
        /// 110→84 B/block) shrink their per-token DRAM traffic ~2.5×/1.3×.
        ///
        /// Quantization scheme per 16-element group:
        ///   - mn>=0 groups: dl = mx/3, min term ml = 0
        ///   - mn<0  groups: dl = (mx-mn)/3, min term ml = -mn
        ///   - dequant out = dl*q - ml with q in {0,1,2,3} → covers [mn, mx]
        /// Block super-scales d = max(dl)/15 and dmin = max(ml)/15 (fp16), with
        /// 4-bit per-group indices rounded into 0..15.
        ///
        /// @param x            Row of floats (numElements elements)
        /// @param numElements  Must be a multiple of 256
        /// @param outQ2K       Output compact Q2_K bytes
        ///                     (numElements/256 blocks × 84 bytes)
        static void quantizeQ2KRow(const float *x, uint32_t numElements,
                                   uint8_t *outQ2K) {
            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t BLOCK_BYTES = 84;
            const uint32_t numBlocks = numElements / BLOCK_SIZE;

            for (uint32_t b = 0; b < numBlocks; ++b) {
                const float *xb = x + static_cast<uint64_t>(b) * BLOCK_SIZE;
                uint8_t *blk = outQ2K + static_cast<uint64_t>(b) * BLOCK_BYTES;

                float groupDl[16] = {0.0f};
                float groupMl[16] = {0.0f};
                uint8_t qvals[256] = {0};
                float maxDl = 0.0f, maxMl = 0.0f;

                for (uint32_t g = 0; g < 16; ++g) {
                    const float *xr = xb + g * 16;
                    float mn = xr[0], mx = xr[0];
                    for (uint32_t l = 1; l < 16; ++l) {
                        mn = std::min(mn, xr[l]);
                        mx = std::max(mx, xr[l]);
                    }
                    float dl, ml;
                    if (mn >= 0.0f) {
                        // Fully non-negative group: no min term needed.
                        dl = mx / 3.0f;
                        ml = 0.0f;
                    } else {
                        // Group has negatives: min term -ml maps q=0 -> mn.
                        dl = (mx - mn) / 3.0f;
                        ml = -mn;
                    }
                    groupDl[g] = dl;
                    groupMl[g] = ml;
                    maxDl = std::max(maxDl, dl);
                    maxMl = std::max(maxMl, ml);

                    // 2-bit quants: dequant(q) = dl*q - ml, so the exact inverse
                    // is q = (x + ml)/dl (ml is 0 for non-negative groups).
                    // WITHOUT the +ml the min-shifted codebook maps x=0 to ~mn
                    // and the round-trip error is ~1.4x the signal (found via
                    // the QuantizeQ2KRowRoundtrip test).
                    for (uint32_t l = 0; l < 16; ++l) {
                        int q;
                        if (dl > 1e-30f) {
                            q = static_cast<int>(std::lrintf((xr[l] + ml) / dl));
                        } else {
                            q = 0;
                        }
                        q = std::min(3, std::max(0, q));
                        qvals[g * 16 + l] = static_cast<uint8_t>(q);
                    }
                }

                // Block super-scales: d = max(dl)/15, dmin = max(ml)/15 (fp16).
                const uint16_t hD = (maxDl > 0.0f) ? floatToHalf(maxDl / 15.0f) : 0;
                const uint16_t hDmin = (maxMl > 0.0f) ? floatToHalf(maxMl / 15.0f) : 0;
                const float dF = halfToFloat(hD);
                const float dminF = halfToFloat(hDmin);

                for (uint32_t g = 0; g < 16; ++g) {
                    int lowIdx = (dF > 1e-30f)
                                         ? static_cast<int>(std::lrintf(groupDl[g] / dF))
                                         : 0;
                    int highIdx = (dminF > 1e-30f)
                                          ? static_cast<int>(std::lrintf(groupMl[g] / dminF))
                                          : 0;
                    lowIdx = std::min(15, std::max(0, lowIdx));
                    highIdx = std::min(15, std::max(0, highIdx));
                    blk[g] = static_cast<uint8_t>(lowIdx | (highIdx << 4));
                }

                // Pack 2-bit planes: qs[byteBase + l] |= q << shift. The
                // (byteBase, shift) mapping above is EXACTLY the read order of
                // dequantizeQ2_KBlock / prepackQ2_K / the compact Q2_K SIMD
                // kernels (quarter 0 shift 0,2,4,6 then quarter 1).
                std::memset(blk + 16, 0, BLOCK_SIZE / 4);
                for (uint32_t e = 0; e < BLOCK_SIZE; ++e) {
                    const uint32_t g = e >> 4;
                    const uint32_t l = e & 15;
                    const int shift = static_cast<int>(((g >> 1) & 3) * 2);
                    const int byteBase = static_cast<int>((g & 1) * 16 +
                                                          ((g & 8) ? 32 : 0));
                    blk[16 + byteBase + l] |= static_cast<uint8_t>(qvals[e] << shift);
                }
                std::memcpy(blk + 80, &hD, 2);
                std::memcpy(blk + 82, &hDmin, 2);
            }
        }

        /// @brief Matrix-vector multiply using pre-packed Q2_K data
        /// The x vector is quantized to Q8_K (int8) once per matmul, then each
        /// row's dot product uses _mm256_maddubs_epi16 (32 int8×int8->int16
        /// multiply-adds per instruction) instead of float FMAs (8 per
        /// instruction).
        ///
        /// @param prepackedData Pre-packed Q2_K data (from prepackQ2_K)
        /// @param x             Input vector (size cols)
        /// @param rows          Number of output rows
        /// @param cols          Number of input columns
        /// @param result        Output buffer (size rows)
        static void matMulVecFusedQ2_K_PrePacked_Q8(const uint8_t *prepackedData,
                                                    const float *x, uint32_t rows,
                                                    uint32_t cols, float *result) {
            // Row-parallel prepacked Q8 path: quantize x to Q8_K once, then
            // parallelize over output rows. This is preferred over the register-tiled
            // batch GEMM kernel for single-token generation because the batch GEMM
            // allocates xCopy/q8 vectors on the heap on every call (slow) and its
            // 8-row register tiling gives no benefit when the matrix is cache-resident.
            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t PREPACKED_BLOCK_BYTES = 276;
            // Max Q8_K blocks we can hold on the stack (covers cols up to 16384).
            // Q8KBlock is 292 B, so 64 blocks = ~18 KB of stack.
            static constexpr uint32_t MAX_STACK_BLOCKS = 64;

            uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            uint64_t rowStrideBytes = static_cast<uint64_t>(blocksPerRow) * PREPACKED_BLOCK_BYTES;

            // Quantize x to Q8_K once per matmul (reused across all rows).
            // Use a stack buffer to avoid per-call heap allocation.
            Q8KBlock stackQ8[MAX_STACK_BLOCKS];
            Q8KBlock *q8 = stackQ8;
            std::vector<Q8KBlock> heapQ8;
            if (blocksPerRow > MAX_STACK_BLOCKS) {
                heapQ8.resize(blocksPerRow);
                q8 = heapQ8.data();
            }
            quantizeQ8K(x, cols, q8);

            // Parallelize over output rows.
            ThreadPool::instance().parallelFor(0, rows, [&](uint32_t j) {
                const uint8_t *rowData = prepackedData + static_cast<uint64_t>(j) * rowStrideBytes;
                double dot = 0.0;

                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    const uint8_t *blockData = rowData + static_cast<uint64_t>(b) * PREPACKED_BLOCK_BYTES;
                    uint32_t start = b * BLOCK_SIZE;
                    uint32_t n = std::min(BLOCK_SIZE, cols - start);

                    if (n < BLOCK_SIZE) {
                        // Partial block: dequantize then dot product
                        float blockOut[256];
                        dequantizeQ2_K_PrePackedBlock(blockData, blockOut, n);
                        for (uint32_t i = 0; i < n; ++i) {
                            dot += static_cast<double>(x[start + i]) * blockOut[i];
                        }
                    } else {
                        // Full block: use Q8_K SIMD kernel
                        dot += static_cast<double>(dotProductQ2_K_PrePacked_Q8_SIMD(blockData, &q8[b]));
                    }
                }
                result[j] = static_cast<float>(dot);
            });
        }

        /// @brief Fused gate+up matrix-vector multiply for pre-packed Q2_K data.
        ///
        /// Computes gate = x * W_gate^T and up = x * W_up^T in a single pass over
        /// the input vector x. Both matrices share the same dimensions (rows x cols)
        /// and read the same x, so quantizing x to Q8_K once (instead of once per
        /// matrix) halves the dominant x-vector memory traffic. Each output row's
        /// dot product uses _mm256_maddubs_epi16 (32 int8×int8->int16 multiply-adds
        /// per instruction) instead of float FMAs (8 per instruction).
        ///
        /// @param gatePrepacked Pre-packed Q2_K data for the gate matrix
        /// @param upPrepacked   Pre-packed Q2_K data for the up matrix
        /// @param x             Input vector (size cols)
        /// @param rows          Number of output rows
        /// @param cols          Number of input columns
        /// @param gateOut       Output buffer for gate (size rows)
        /// @param upOut         Output buffer for up (size rows)
        static void matMulVecFusedGateUpQ2_K_PrePacked_Q8(
                const uint8_t *gatePrepacked, const uint8_t *upPrepacked,
                const float *x, uint32_t rows, uint32_t cols,
                float *gateOut, float *upOut) {
            // Use register-tiled batch GEMM for large matrices (BATCH_SIZE=8 rows).
            // This keeps Q8_K data in registers across all rows in the batch AND
            // across both gate/up matrices, eliminating redundant memory reads.
            static constexpr uint32_t MIN_ROWS_FOR_BATCH = 64;

            if (rows >= MIN_ROWS_FOR_BATCH && cols % 256 == 0) {
                tinycoder::matMulVecBatchGateUpQ2_K_PrePacked_Q8_SIMD(
                        gatePrepacked, upPrepacked, x, rows, cols, gateOut, upOut);
                return;
            }

            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t PREPACKED_BLOCK_BYTES = 276;

            uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            uint64_t rowStrideBytes = static_cast<uint64_t>(blocksPerRow) * PREPACKED_BLOCK_BYTES;

            // Quantize x to Q8_K once per fused matmul (reused across all rows of
            // BOTH matrices). This is the key win: the original code quantized x
            // separately for ffnGate and ffnUp, reading x twice.
            std::vector<Q8KBlock> q8(blocksPerRow);
            quantizeQ8K(x, cols, q8.data());

            ThreadPool::instance().parallelFor(0, rows, [&](uint32_t j) {
                const uint8_t *gateRow = gatePrepacked + static_cast<uint64_t>(j) * rowStrideBytes;
                const uint8_t *upRow = upPrepacked + static_cast<uint64_t>(j) * rowStrideBytes;
                double gateDot = 0.0;
                double upDot = 0.0;

                // Software prefetch the next row's weight data to hide DRAM latency
                // on the large gate/up matrices (8960×1536 each).
                if (j + 1 < rows) {
                    prefetchRow(gateRow + rowStrideBytes);
                    prefetchRow(upRow + rowStrideBytes);
                }

                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    const uint8_t *gateBlock = gateRow + static_cast<uint64_t>(b) * PREPACKED_BLOCK_BYTES;
                    const uint8_t *upBlock = upRow + static_cast<uint64_t>(b) * PREPACKED_BLOCK_BYTES;
                    uint32_t start = b * BLOCK_SIZE;
                    uint32_t n = std::min(BLOCK_SIZE, cols - start);

                    if (n < BLOCK_SIZE) {
                        // Partial block: dequantize then dot product
                        float blockOut[256];
                        dequantizeQ2_K_PrePackedBlock(gateBlock, blockOut, n);
                        for (uint32_t i = 0; i < n; ++i) {
                            gateDot += static_cast<double>(x[start + i]) * blockOut[i];
                        }
                        dequantizeQ2_K_PrePackedBlock(upBlock, blockOut, n);
                        for (uint32_t i = 0; i < n; ++i) {
                            upDot += static_cast<double>(x[start + i]) * blockOut[i];
                        }
                    } else {
                        // Full block: use Q8_K SIMD kernel for both matrices
                        gateDot += static_cast<double>(dotProductQ2_K_PrePacked_Q8_SIMD(gateBlock, &q8[b]));
                        upDot += static_cast<double>(dotProductQ2_K_PrePacked_Q8_SIMD(upBlock, &q8[b]));
                    }
                }
                gateOut[j] = static_cast<float>(gateDot);
                upOut[j] = static_cast<float>(upDot);
            });
        }

        /// @brief Fused gate+up matrix-vector multiply for COMPACT (raw) Q2_K
        /// data — single-token (generation) fast path.
        ///
        /// Identical maths to matMulVecFusedGateUpQ2_K_PrePacked_Q8 but reads the
        /// raw 84-byte Q2_K blocks (scales[16], qs[64] packed 2-bit, fp16 d/dmin)
        /// and unpacks the 2-bit quants to bytes on the fly inside the kernel.
        /// Generation is DRAM-bandwidth-bound and gate+up is the dominant
        /// per-token cost; the compact layout is 84 B/block vs 276 B/block for the
        /// prepacked copy — a ~3.3x reduction in weight traffic (matches llama.cpp).
        ///
        /// @param gateData Raw (compact) Q2_K data for the gate matrix
        /// @param upData   Raw (compact) Q2_K data for the up matrix
        /// @param x        Input vector (size cols)
        /// @param rows     Number of output rows
        /// @param cols     Number of input columns
        /// @param gateOut  Output buffer for gate (size rows)
        /// @param upOut    Output buffer for up (size rows)
        static void matMulVecFusedGateUpQ2_K_Compact_Q8(
                const uint8_t *gateData, const uint8_t *upData,
                const float *x, uint32_t rows, uint32_t cols,
                float *gateOut, float *upOut, bool applySwish) {
            // Use the register-tiled batch GEMM for large matrices (BATCH_SIZE=8
            // rows). This keeps Q8_K data in registers across all rows in the
            // batch AND across both gate/up matrices, eliminating redundant
            // memory reads.
            static constexpr uint32_t MIN_ROWS_FOR_BATCH = 64;

            if (rows >= MIN_ROWS_FOR_BATCH && cols % 256 == 0) {
                tinycoder::matMulVecFusedGateUpQ2_K_Compact_Q8_SIMD(
                        gateData, upData, x, rows, cols, gateOut, upOut,
                        applySwish);
                return;
            }

            static constexpr uint32_t BLOCK_SIZE = 256;
            static constexpr uint32_t COMPACT_BLOCK_BYTES = 84;

            uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            uint64_t rowStrideBytes =
                    static_cast<uint64_t>(blocksPerRow) * COMPACT_BLOCK_BYTES;

            // Quantize x to Q8_K once per fused matmul (reused across all rows of
            // BOTH matrices). This is the key win: the original code quantized x
            // separately for ffnGate and ffnUp, reading x twice.
            std::vector<Q8KBlock> q8(blocksPerRow);
            quantizeQ8K(x, cols, q8.data());

            ThreadPool::instance().parallelFor(0, rows, [&](uint32_t j) {
                const uint8_t *gateRow = gateData + static_cast<uint64_t>(j) * rowStrideBytes;
                const uint8_t *upRow = upData + static_cast<uint64_t>(j) * rowStrideBytes;
                double gateDot = 0.0;
                double upDot = 0.0;

                // Software prefetch the next row's weight data to hide DRAM
                // latency on the large gate/up matrices (8960x1536 each).
                if (j + 1 < rows) {
                    prefetchRow(gateRow + rowStrideBytes);
                    prefetchRow(upRow + rowStrideBytes);
                }

                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    const uint8_t *gateBlock = gateRow + static_cast<uint64_t>(b) * COMPACT_BLOCK_BYTES;
                    const uint8_t *upBlock = upRow + static_cast<uint64_t>(b) * COMPACT_BLOCK_BYTES;
                    uint32_t start = b * BLOCK_SIZE;
                    uint32_t n = std::min(BLOCK_SIZE, cols - start);

                    if (n < BLOCK_SIZE) {
                        // Partial block: dequantize then dot product
                        float blockOut[256];
                        dequantizeQ2_KBlock(gateBlock, blockOut);
                        for (uint32_t i = 0; i < n; ++i) {
                            gateDot += static_cast<double>(x[start + i]) * blockOut[i];
                        }
                        dequantizeQ2_KBlock(upBlock, blockOut);
                        for (uint32_t i = 0; i < n; ++i) {
                            upDot += static_cast<double>(x[start + i]) * blockOut[i];
                        }
                    } else {
                        // Full block: use Q8_K SIMD kernel for both matrices.
                        // Decompose each compact block once (to the same float
                        // values the prepacked expansion produces), then dot it
                        // against the same Q8_K x data.
                        for (uint32_t mat = 0; mat < 2; ++mat) {
                            const uint8_t *blk = mat == 0 ? gateBlock : upBlock;
                            float d = halfToFloat(*(const uint16_t *) (blk + 80));
                            float dmin = halfToFloat(*(const uint16_t *) (blk + 82));
                            const uint8_t *sc = blk;
                            const uint8_t *q = blk + 16;
                            double &dotRef = mat == 0 ? gateDot : upDot;
                            int is = 0;
                            for (int nn = 0; nn < 256; nn += 128) {
                                int shift = 0;
                                for (int jj = 0; jj < 4; ++jj) {
                                    uint8_t scv = sc[is++];
                                    float dl = d * (scv & 0xF);
                                    float ml = dmin * (scv >> 4);
                                    for (int l = 0; l < 16; ++l) {
                                        float w = dl * ((int8_t) ((q[l] >> shift) & 3)) - ml;
                                        dotRef += static_cast<double>(x[start + nn + jj * 32 + l]) * w;
                                    }
                                    scv = sc[is++];
                                    dl = d * (scv & 0xF);
                                    ml = dmin * (scv >> 4);
                                    for (int l = 0; l < 16; ++l) {
                                        float w = dl * ((int8_t) ((q[l + 16] >> shift) & 3)) - ml;
                                        dotRef += static_cast<double>(x[start + nn + jj * 32 + 16 + l]) * w;
                                    }
                                    shift += 2;
                                }
                                q += 32;
                            }
                        }
                    }
                }
                if (applySwish) {
                    // Fuse SwiGLU into the epilogue (silu(gate) * up) so all
                    // dispatch routes are semantically identical to the fused
                    // AVX2 kernel.
                    float gv = static_cast<float>(gateDot);
                    gv = gv / (1.0f + std::exp(-gv));
                    gateOut[j] = gv * static_cast<float>(upDot);
                } else {
                    gateOut[j] = static_cast<float>(gateDot);
                }
                upOut[j] = static_cast<float>(upDot);
            });
        }
    };

}// namespace tinycoder
