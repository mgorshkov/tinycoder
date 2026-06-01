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
#include <iostream>
#include <vector>

#include "GGUFLoader.hpp"
#include "SIMDMatMulVec.hpp"

namespace tinycoder {

    /// @brief GGML quantization dequantization utilities.
    ///
    /// Provides dequantization for various GGML quantized formats
    /// used in mixed-quantization GGUF files (Q5_K, Q5_1, IQ3_S, IQ2_S, etc.).
    struct GGMLDequantize {

        /// K-quant block size constant used in dequantization functions.
        static constexpr uint32_t QK_K = 256;

        /// @brief Convert IEEE 754 half-precision (16-bit) to float.
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
                        const uint8_t *grid = reinterpret_cast<const uint8_t *>(&iq2xs_grid[gridIdx]);
                        const uint8_t signs = ksigns_iq2xs[signIdx];
                        for (uint32_t j = 0; j < 8; ++j) {
                            float w = db[l / 2] * static_cast<float>(grid[j]) *
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
                    const uint8_t *grid = reinterpret_cast<const uint8_t *>(&iq2xs_grid[gridIdx]);
                    const uint8_t signs = ksigns_iq2xs[signIdx];
                    for (uint32_t j = 0; j < 8; ++j) {
                        out[ib32 * 32 + l * 8 + j] =
                                db[l / 2] * static_cast<float>(grid[j]) *
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

        /// @brief Compute y = x * W where W is a quantized matrix stored in
        /// GGUF format as (rows x cols) = (out_features x in_features) row-major.
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

            // Parallelize over output rows when not already inside an outer parallel region.
            // With default OpenMP nesting (disabled), this uses 8 threads when called from
            // generation (seqLen=1) and serializes to 1 thread when called inside the
            // prefill per-token parallel loop - optimal for both modes.
#pragma omp parallel for schedule(static)
            for (uint32_t j = 0; j < rows; ++j) {
                const uint8_t *rowData = data + static_cast<uint64_t>(j) * rowStrideBytes;
                double dot = 0.0;

                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    const uint8_t *blockData = rowData + static_cast<uint64_t>(b) * typeSize;
                    float blockOut[256];
                    dequantizeBlock(ggmlType, blockData, blockOut, blockSize);

                    uint32_t start = b * blockSize;
                    uint32_t n = std::min(blockSize, cols - start);

                    // Dot product: dot += x[start..start+n) * blockOut[0..n)
                    for (uint32_t k = 0; k < n; ++k) {
                        dot += static_cast<double>(x[start + k]) * blockOut[k];
                    }
                }
                result[j] = static_cast<float>(dot);
            }
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
    };

}// namespace tinycoder
