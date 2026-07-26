# ⚡ TinyCoder AI

**Ultrafast AI local coding assistant for VS Code** powered by **Qwen2.5-Coder**, **Gemma 4**, and **Qwen3.6** model families with **mixed quantization** (IQ3_XXS, IQ3_S, IQ2_S, IQ3_XS, IQ2_M, Q5_K, Q5_1, Q4_K, Q4_K_XL, Q4_K_M, Q2_K, Q6_K), **AMX/AVX-512/AVX2 CPU acceleration**, and **CUDA GPU support**. Designed to run efficiently on limited hardware.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      VS Code Extension                            │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │               TypeScript Layer (src/ts/)                      │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │ │
│  │  │  extension   │  │    panel     │  │   nativeBridge    │  │ │
│  │  │  .ts         │  │    .ts       │  │   .ts             │  │ │
│  │  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘  │ │
│  └─────────┼──────────────────┼───────────────────┼─────────────┘ │
│            │                  │                   │               │
│  ┌─────────┼──────────────────┼───────────────────┼─────────────┐ │
│  │         │    N-API Native Addon (src/cpp/Bridge.cpp)         │ │
│  │         └──────────────────┼───────────────────┘             │ │
│  │                            │                                 │ │
│  │  ┌─────────────────────────┼─────────────────────────────┐  │ │
│  │  │            C++ Inference Engine (src/cpp/)              │  │ │
│  │  │                                                         │  │ │
│  │  │  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │  │ │
│  │  │  │  Model   │  │  GGUFLoader  │  │   Tokenizer      │  │  │ │
│  │  │  │  .cpp    │  │  .cpp        │  │   .cpp           │  │  │ │
│  │  │  └────┬─────┘  └──────┬───────┘  └────────┬─────────┘  │  │ │
│  │  │       │               │                    │            │  │ │
│  │  │  ┌────┴───────────────┴────────────────────┴─────────┐  │  │ │
│  │  │  │           Quantized Weight Engine                  │  │  │ │
│  │  │  │  ┌──────────────────┐  ┌──────────────────────┐   │  │  │ │
│  │  │  │  │ QuantizedMatrix  │  │ QuantizedEmbedding   │   │  │  │ │
│  │  │  │  │ .cpp             │  │ .cpp                 │   │  │  │ │
│  │  │  │  └────────┬─────────┘  └──────────┬───────────┘   │  │  │ │
│  │  │  │           │                       │               │  │  │ │
│  │  │  │  ┌────────┴───────────────────────┴────────────┐  │  │  │ │
│  │  │  │  │        GGMLDequantize (GGMLDequantize.hpp)   │  │  │  │ │
│  │  │  │  │  Q5_K  Q5_1  Q4_K  Q4_K_XL  Q4_K_M  Q2_K   │  │  │  │ │
│  │  │  │  │  Q6_K  IQ3_XXS  IQ3_S  IQ3_XS  IQ2_S  IQ2_M │  │  │  │ │
│  │  │  │  └──────────────────────┬───────────────────────┘  │  │  │ │
│  │  │  │                         │                          │  │  │ │
│  │  │  │  ┌──────────────────────┴───────────────────────┐  │  │  │ │
│  │  │  │  │  SIMDMatMulVec (AMX/AVX2/AVX-512 runtime dispatch)│  │  │  │ │
│  │  │  │  └──────────────────────────────────────────────┘  │  │  │ │
│  │  │  └────────────────────────────────────────────────────┘  │  │ │
│  │  │                                                           │  │ │
│  │  │  ┌────────────────────────────────────────────────────┐  │  │ │
│  │  │  │  LM Head (src/cpp/LMHead.hpp + LMHeadCUDA.cu)      │  │  │ │
│  │  │  │  ┌──────────────┐  ┌──────────────────────────┐   │  │  │ │
│  │  │  │  │  CPU (OpenMP)│  │  CUDA (cublasSgemv)      │   │  │  │ │
│  │  │  │  └──────────────┘  └──────────────────────────┘   │  │  │ │
│  │  │  └────────────────────────────────────────────────────┘  │  │ │
│  │  └───────────────────────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

## Key Features

### 🧠 Supported Models
TinyCoder supports models from multiple families across various quantization formats:

| Model Family | Sizes | Architecture |
|---|---|---|
| **Qwen2.5-Coder** | 0.5B, 1.5B, 7B | RoPE, GQA, SwiGLU, RMSNorm |
| **Gemma 4** | 26B (A4B MoE), coding variants | MoE, RoPE, GQA, GeGLU, RMSNorm |
| **Qwen3.6** | 35B (A3B MoE) | MoE, RoPE, GQA, SwiGLU, RMSNorm |

- Chat template: `<|im_start|>system/user/assistant<|im_end|>` (Qwen), `<start_of_turn>user/model<end_of_turn>` (Gemma)
- **Weight tying** — LM head shares token embedding matrix when possible

### 📦 Mixed Quantization (Multiple GGML Types)
Unlike many inference engines that use a single quantization type, TinyCoder supports **mixed-quantization GGUF files** where different tensors use different quantization formats:

| Type | Bits/Weight | Block Size | Block Bytes | Used For |
|------|-------------|------------|-------------|----------|
| **IQ3_XXS** | 3.0625 | 256 | 98 | Primary weight quantization |
| **IQ3_S** | 3.4375 | 256 | 110 | Higher-precision layers |
| **IQ3_XS** | 3.3125 | 256 | 106 | Balanced MoE compression |
| **IQ2_S** | 2.5625 | 256 | 82 | Aggressive compression |
| **IQ2_M** | 2.6875 | 256 | 86 | Moderate aggressive compression |
| **Q2_K** | 2.5625 | 256 | 82 | Small model quantization |
| **Q4_K** | 4.0625 | 256 | 144 | Balanced precision |
| **Q4_K_M** | 4.5 | 256 | 144 | Medium-balanced precision |
| **Q4_K_XL** | 4.5 | 256 | 160 | Extended balanced precision |
| **Q5_K** | 5.0625 | 256 | 176 | Embeddings, high-precision layers |
| **Q5_1** | 5.0625 | 32 | 32 | Legacy format support |
| **Q6_K** | 6.5625 | 256 | 210 | High-precision layers |
| **F32** | 32 | 1 | 4 | Norms, biases |

All weights are stored in their **native quantized format** in memory and dequantized **on-the-fly** during matrix-vector multiplication. This keeps memory usage close to the compressed file size (~637 MB for 1.5B) rather than the F32 dequantized size (~5.2 GB).

### ⚡ SIMD CPU Acceleration
- **Runtime SIMD dispatch** via `np::internal::CpuDispatch`
- Auto-detects best available instruction set:
  - **AMX** — Intel Sapphire Rapids+ (tile matrix multiply, 8-bit)
  - **AVX-512** — Intel Skylake-X / Ice Lake+ (16-wide FMA)
  - **AVX2** — Intel Haswell+ / AMD Zen+ (8-wide FMA)
  - **Scalar** — Universal fallback
- **OpenMP** thread parallelism for all matrix operations
- Per-file SIMD compilation flags (only `SIMDMatMulVec.cpp` compiled with AVX/AVX-512/AMX flags)

### 🎮 CUDA GPU Support
- Optional GPU offload for large matrix operations via `np` library's CUDA backend
- **CUDA-accelerated LM head** using `cublasSgemv` with persistent GPU embedding matrix
- Automatic fallback to CPU when CUDA is unavailable or matrix is too small
- Configurable via `ENABLE_CUDA` CMake option

### 💻 VS Code Integration
- **Chat panel** (standalone + sidebar) for interactive AI assistance
- **Explain code** — select code and get explanations
- **Complete code** — context-aware code completion
- **Generate code** — generate from comments/descriptions
- **Streaming output** — see tokens as they're generated
- **Status bar** — model status indicator with periodic updates

## Project Structure

```
tinycoder/
├── include/                        # C++ headers
│   ├── ModelConfig.hpp             # Model & inference configuration
│   ├── GGUFLoader.hpp              # GGUF v3 file format loader
│   ├── GGMLDequantize.hpp          # Multi-type dequantization (Q5_K, IQ3_XXS, etc.)
│   ├── IQ3XXS.hpp                  # Legacy IQ3_XXS block-level dequantization
│   ├── LMHead.hpp                  # LM head computation (CPU OpenMP path)
│   ├── LMHeadCUDA.hpp              # LM head CUDA (cublasSgemv) interface
│   ├── Model.hpp                   # Transformer model (forward, generate, KV cache)
│   ├── SIMDMatMulVec.hpp           # SIMD-accelerated dot product & accumulate
│   └── Tokenizer.hpp               # BPE tokenizer (Qwen2.5)
├── src/
│   ├── cpp/                        # C++ source
│   │   ├── Bridge.cpp              # N-API native addon (load, generate, status)
│   │   ├── GGUFLoader.cpp          # GGUF v3 reader (metadata + tensor data)
│   │   ├── GridTables.cpp          # IQ2_S grid lookup table (1024 entries)
│   │   ├── GridTablesIQ3S.cpp      # IQ3_S grid lookup table (512 entries)
│   │   ├── Model.cpp               # Transformer forward pass + generation loop
│   │   ├── QuantizedEmbedding.cpp  # Quantized token embedding dequantization
│   │   ├── QuantizedMatrix.cpp     # Quantized matrix-vector multiply (CUDA/CPU)
│   │   ├── SIMDMatMulVec.cpp       # SIMD dispatch (AVX2/AVX-512/scalar)
│   │   ├── LMHeadCUDA.cu           # CUDA LM head (cublasSgemv)
│   │   └── Tokenizer.cpp           # BPE tokenizer (GGUF embedded + file loading)
│   └── ts/                         # TypeScript source
│       ├── extension.ts            # VS Code extension entry (commands, status bar)
│       ├── panel.ts                # WebView chat panel (standalone + sidebar)
│       └── nativeBridge.ts         # Native addon wrapper (async load/generate)
├── unit_tests/                     # Unit tests
│   ├── CMakeLists.txt              # Test build config
│   └── ModelTest.cpp               # Model loading & inference tests
├── CMakeLists.txt                  # CMake build (with np fetch, N-API, CUDA)
├── package.json                    # VS Code extension manifest
├── tsconfig.json                   # TypeScript configuration
├── scripts/
│   ├── build.sh                    # Build script
│   └── deploy.sh                   # Deployment script
├── media/
│   ├── icon.png                    # Extension icon
│   └── icon.svg                    # Extension icon (vector)
└── .vscode/
    ├── launch.json                 # Debug configurations
    └── tasks.json                  # Build tasks
```

## Dependencies

No external dependency except for linear algebra <a href="https://github.com/mgorshkov/np">np library</a> is used.

## Build & Install

### Prerequisites

- **Node.js** >= 18.x
- **VS Code** >= 1.85.0
- **C++20 compiler** (GCC 11+, Clang 14+, MSVC 2022+)
- **CMake** >= 3.18

### Optional (for acceleration)

- **OpenMP** (usually included with compiler)
- **CUDA Toolkit** >= 11.0 (for GPU support)

### Build Steps

```bash
# 1. Install dependencies
npm install

# 2. Compile TypeScript
npm run compile

# 3. Build native addon (auto-detects best SIMD)
npm run build:native

# Alternative: AVX2 only (for older CPUs)
npm run build:native:avx2

# Alternative: with CUDA
npm run build:native:cuda

# 4. Launch in VS Code
code .
# Press F5 to run the extension
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_CUDA` | ON | CUDA GPU acceleration |
| `ENABLE_AVX2` | ON | AVX2 + FMA optimizations |
| `ENABLE_AVX512` | OFF | AVX-512 optimizations |
| `ENABLE_AMX` | OFF | Intel AMX (Advanced Matrix Extensions) |
| `ENABLE_OPENMP` | ON | OpenMP thread parallelism |
| `BUILD_TESTS` | ON | Build unit tests (Google Test) |

### Getting a Model

TinyCoder supports models from multiple families. Below are the models available in the standard model directory (`/data/models`):

| Model | Quantization | Size (approx) | Notes |
|-------|-------------|---------------|-------|
| `Qwen2.5-Coder-0.5B` | Q4_K_M | ~350 MB | Lightweight coding |
| `Qwen2.5-Coder-1.5B` | Q2_K | ~550 MB | Aggressive compression |
| `Qwen2.5-Coder-1.5B` | IQ3_XXS (imat) | ~700 MB | Recommended balanced |
| `Qwen2.5-Coder-7B` | IQ2_S | ~2.0 GB | Ultra-compact 7B |
| `Qwen2.5-Coder-7B` | IQ3_XXS (imat) | ~3.2 GB | Recommended 7B |
| `Gemma 4 Coding` | Q2_K | ~8.5 GB | Lightweight MoE coding |
| `Gemma 4 Coding` | Q4_K_M | ~13 GB | Balanced MoE coding |
| `Gemma 4 Coding` | Q6_K | ~19 GB | High-precision MoE coding |
| `Gemma 4 (26B-A4B)` | Q4_K_XL (qat-UD) | ~14 GB | Instruction-tuned MoE |
| `Qwen3.6 (35B-A3B)` | IQ3_XS (distilled) | ~12 GB | Reasoning-distilled MoE |
| `Qwen3.6 (35B-A3B)` | IQ2_M (UD) | ~8.5 GB | Ultra-compact MoE |
| `Qwen3.6 (35B-A3B)` | IQ3_XS (Claude distill) | ~12 GB | Claude reasoning MoE |

Place the `.gguf` file in the models directory and load it from the TinyCoder panel.

## Usage

1. **Open VS Code** and press `Ctrl+Alt+T` (or `Cmd+Alt+T` on Mac)
2. **Load a model** by clicking "Load Model" and selecting your `.gguf` file
3. **Start coding!** Ask TinyCoder to:
   - Write code: "Write a Python function to sort a list"
   - Explain code: Select code → right-click → "TinyCoder: Explain"
   - Complete code: Select context → right-click → "TinyCoder: Complete"
   - Debug: "Why is this function returning NaN?"

### Commands

| Command | Keybinding | Description |
|---------|-----------|-------------|
| `TinyCoder: Open AI Chat Panel` | `Ctrl+Alt+T` | Open the chat interface |
| `TinyCoder: Explain Selected Code` | `Ctrl+Alt+E` | Explain highlighted code |
| `TinyCoder: Complete Code` | `Ctrl+Alt+C` | Complete code from context |
| `TinyCoder: Generate Code from Comment` | - | Generate from comment |
| `TinyCoder: Load Model...` | - | Load a GGUF model file |
| `TinyCoder: Show Status` | - | Show model status |

## Performance

| Model | Quantization | Size | RAM (steady) | Speed (tok/s) |
|-------|-------------|------|-------------|---------------|
| Qwen2.5-Coder-0.5B | Q4_K_M | ~350 MB | ~400 MB | 40-60 (AVX2) |
| Qwen2.5-Coder-1.5B | Mixed (IQ3_XXS+) | ~700 MB | ~750 MB | 15-25 (AVX2) |
| Qwen2.5-Coder-1.5B | Mixed (IQ3_XXS+) | ~700 MB | ~750 MB | 25-40 (AVX-512) |
| Qwen2.5-Coder-7B | Mixed (IQ3_XXS+) | ~3.2 GB | ~3.5 GB | 5-10 (AVX2) |
| Qwen2.5-Coder-7B | Mixed (IQ3_XXS+) | ~3.2 GB | ~3.5 GB | 10-18 (AVX-512) |
| Gemma 4 Coding | Q2_K | ~8.5 GB | ~9.0 GB | 3-6 (AVX2) |
| Gemma 4 Coding | Q4_K_M | ~13 GB | ~14 GB | 2-4 (AVX2) |
| Gemma 4 (26B-A4B) | Q4_K_XL | ~14 GB | ~15 GB | 2-4 (AVX2) |
| Qwen3.6 (35B-A3B) | IQ3_XS | ~12 GB | ~13 GB | 2-5 (AVX2) |
| Qwen3.6 (35B-A3B) | IQ2_M | ~8.5 GB | ~9.0 GB | 3-6 (AVX2) |

*Performance measured on Intel i7-13700K with DDR5-5600. Actual results may vary. MoE models show lower tok/s due to expert routing overhead.*

**Memory breakdown (1.5B model):**
- Quantized weights: ~637 MB (stored in native format, no dequantization)
- FP16 dequantized copies (attnO + ffnDown): ~30.7 MB (half of previous F32 ~61.5 MB)
- KV cache (2048 context): ~112 MB (F32, 28 layers × 2 KV heads × 128 dim)
- Norms + activations: ~1 MB
- **Total steady state: ~781 MB**

## Technical Details

### Mixed Quantization Dequantization

TinyCoder supports **12 quantization formats** simultaneously within a single model file. The [`GGMLDequantize`](include/GGMLDequantize.hpp:44) struct provides block-level dequantization for each type:

```
Q5_K Block (256 weights, 176 bytes):
┌──────────────────────────────────────────────────────────────┐
│  d(F16)  dmin(F16)  scales(12B)  qh(32B)  qs(128B)         │
│  2 bytes  2 bytes    12 bytes    32 bytes  128 bytes         │
└──────────────────────────────────────────────────────────────┘
  Dequant: w = (q5 - 16) * d * sc - dmin * m   (per sub-block)

IQ3_XXS Block (256 weights, 98 bytes):
┌──────────────────────────────────────────────────────────────┐
│  d(F16)  qs(96B packed 3-bit indices)  scales+signs(4B/32w) │
│  2 bytes  96 bytes                      32 bytes             │
└──────────────────────────────────────────────────────────────┘
  Dequant: w = grid[qs_idx] * d * (0.5 + scale/28) * sign

IQ3_S Block (256 weights, 110 bytes):
┌──────────────────────────────────────────────────────────────┐
│  d(F16)  qs(64B)  qh(8B)  signs(32B)  scales(4B)           │
│  2 bytes  64 bytes  8 bytes  32 bytes    4 bytes             │
└──────────────────────────────────────────────────────────────┘

IQ2_S Block (256 weights, 82 bytes):
┌──────────────────────────────────────────────────────────────┐
│  d(F16)  qs(64B)  qh(8B)  scales(8B)                       │
│  2 bytes  64 bytes  8 bytes  8 bytes                         │
└──────────────────────────────────────────────────────────────┘
```

### Fused Dequantize-Dot Product

Instead of dequantizing entire matrices to F32 (which would require ~5.2 GB for the 1.5B model), [`QuantizedMatrix::matMulVec()`](src/cpp/QuantizedMatrix.cpp:83) uses a **block-level fused approach** with two strategies:

#### Strategy 1: Dequantize-then-Dot (legacy, for non-K-quant types)

1. For each output row, iterate over quantized blocks
2. Dequantize one block at a time into a small stack buffer (max 256 floats)
3. Compute the dot product with the input vector using SIMD-accelerated [`dotProductFMA()`](include/SIMDMatMulVec.hpp:54)
4. Accumulate across blocks for the final row result

This avoids allocating large float buffers and reduces memory bandwidth.

#### Strategy 2: Fused Dot Product (K-quant types: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)

For K-quant types, the dequantization and dot product are **fully fused** — no float temporary buffer is created at all. Each group of 16 weights sharing the same (scale, min) pair is processed as:

```
sum_xq = Σ x[i] × quant[i]    (quantized values: 0-3 for Q2_K, etc.)
sum_x  = Σ x[i]
dot   += dl × sum_xq - ml × sum_x
```

where `dl = d × scale` and `ml = dmin × min`. This is mathematically equivalent to `dot += x[i] × (dl × quant[i] - ml)` but avoids writing 256 floats to the stack and reading them back.

The dispatch chain is:

```
dotProductFused()                    ← GGMLDequantize.hpp:1392
  └─ dotProductQ2_K()               ← GGMLDequantize.hpp:1121 (delegates to SIMD)
       └─ dotProductQ2_K_SIMD()     ← SIMDMatMulVec.cpp:926 (runtime dispatch)
            ├─ dotProductQ2_K_AVX2() ← SIMDMatMulVec.cpp:406 (AVX2 path)
            └─ dotProductQ2_K_Scalar() ← SIMDMatMulVec.cpp:109 (scalar fallback)
```

#### Q2_K Block Layout and AVX2 Strategy

The Q2_K block packs 256 2-bit weights into 84 bytes:

```
Q2_K Block (256 weights, 84 bytes):
┌──────────────────────────────────────────────────────────────┐
│  scales[16]   qs[64]          d(F16)   dmin(F16)            │
│  16 bytes     64 bytes        2 bytes  2 bytes               │
└──────────────────────────────────────────────────────────────┘
  scales: 4-bit scale (low nibble) + 4-bit min (high nibble) per group of 16
  qs:     2-bit quantized values (0-3), 4 values per byte
  d:      super-block scale (fp16)
  dmin:   super-block minimum (fp16)
  Dequant: w = d × scale × quant - dmin × min
```

The AVX2 implementation processes 16 weights per group using two `__m256` vectors (8 each):

1. **Load** 16 bytes of q data via `_mm_loadu_si128`
2. **Zero-extend** bytes to 32-bit ints via `_mm256_cvtepu8_epi32`
3. **Variable shift** right by `shift` (0,2,4,6) via `_mm256_srlv_epi32` to align the 2-bit field
4. **Mask** with 3 via `_mm256_and_si256` to extract the 2-bit quant value
5. **Convert** to float via `_mm256_cvtepi32_ps`
6. **Fused multiply-subtract**: `val = dl × quant - ml` via `_mm256_fmsub_ps`
7. **Fused multiply-add**: `acc += x × val` via `_mm256_fmadd_ps`

This eliminates the float `blockOut[256]` temporary and the extra memory pass, reducing both compute and memory bandwidth.

#### Dual-Format Strategy for attnO and ffnDown

The output projection matrices [`attnO`](include/Model.hpp:141) and [`ffnDown`](include/Model.hpp:149) are stored in **two formats** simultaneously:

| Format | Storage | Memory | Used For |
|--------|---------|--------|----------|
| **Native Q2_K/Q3_K** | `QuantizedMatrix` | ~1.0 MB / ~5.8 MB | Fused dot product via `matMulVec()` |
| **Pre-dequantized FP16** | `std::vector<uint16_t>` | ~4.5 MB / ~26.2 MB | Exact float dot product via `deqMatMulVecF16()` |

The FP16 copies are dequantized once during model loading ([`Model::loadWeights()`](src/cpp/Model.cpp:235)) using [`GGMLDequantize::dequantizeToF16()`](include/GGMLDequantize.hpp:1650), which dequantizes one block at a time into a small stack buffer and converts each value to FP16 inline — avoiding the large intermediate F32 allocation entirely.

During the forward pass, [`deqMatMulVecF16()`](src/cpp/Model.cpp:1209) converts FP16 weights back to FP32 on-the-fly using the **F16C** instruction (`_mm256_cvtph_ps`), which converts 8 FP16 values to 8 FP32 values in a single instruction. This is available on all CPUs that support AVX2 (F16C is implied by `-mavx2` on most compilers, but GCC 13+ requires explicit `-mf16c`).

This is a deliberate trade-off: the FP16 copies consume ~30.7 MB extra memory (half of the previous F32 ~61.5 MB) while avoiding repeated dequantization of the same weights across multiple tokens. The memory bandwidth savings from FP16 (vs F32) also improve cache utilization during the forward pass.

All other weight matrices (Q, K, V, gate, up) use only their native quantized format and rely on the fused dot product path.

#### Weight Pre-Packing for Q2_K (ffnGate / ffnUp)

The Q2_K block stores 256 2-bit values packed into 64 bytes (4 values per byte). The AVX2 kernel spends ~10 instructions per group extracting these 2-bit values (shift, mask, expand, pack). With 8 groups per block, that's 80 instructions per block just for bit extraction.

**Pre-packing** expands the 2-bit values to full bytes (0-3) in element order at load time, so the SIMD kernel can load them with a single `_mm_loadu_si128` per group:

```
Pre-packed Q2_K Block (276 bytes):
┌──────────────────────────────────────────────────────────────────┐
│  scales[16]  d(F16)  dmin(F16)  qs_expanded[256]                │
│  16 bytes    2 bytes  2 bytes    256 bytes (each byte is 0-3)    │
└──────────────────────────────────────────────────────────────────┘
```

**Target:** Only [`ffnGate`](include/Model.hpp:144) and [`ffnUp`](include/Model.hpp:145) — these account for 97.7% of Q2_K dot product calls during inference. Memory overhead: ~28.3 MB for both matrices in Qwen2.5-Coder-7B.

**Implementation:**
- Pre-packing happens at load time in [`Model::loadWeights()`](src/cpp/Model.cpp:434) via [`GGMLDequantize::prepackQ2_K()`](include/GGMLDequantize.hpp:1693)
- Stored in [`QuantizedMatrix::prepackedData`](include/Model.hpp:57)
- Runtime dispatch in [`dotProductQ2_K_PrePacked_SIMD()`](src/cpp/SIMDMatMulVec.cpp:1410) selects AVX2 or scalar kernel
- The pre-packed kernel is mathematically identical to the original (validated by `CompareBatchVsSequentialPrefill` test producing identical results)

### SIMD Runtime Dispatch

The [`SIMDMatMulVec`](src/cpp/SIMDMatMulVec.cpp) module provides five key operations with runtime dispatch:

- **`dotProductFMA()`** — Dot product of two float vectors
- **`dotProductFMA_F16()`** — Dot product of float vector with FP16-stored weights (uses `_mm256_cvtph_ps` on AVX2)
- **`accumulateFMA()`** — Fused multiply-add accumulation
- **`dotProductQ2_K_SIMD()`** — Fused Q2_K dequantize-dot product for native Q2_K blocks (AVX2/scalar)
- **`dotProductQ2_K_PrePacked_SIMD()`** — Fused Q2_K dot product for pre-packed blocks (eliminates 2-bit extraction, AVX2/scalar)

All use `std::atomic` function pointers initialized on first call via `np::internal::max_simd_level()`:

```cpp
// Auto-detected at startup, cached forever
SimdLevel level = np::internal::max_simd_level();
// SCALAR < SSE2 < SSE3 < AVX < AVX2 < AVX512 < AMX
```

Only [`SIMDMatMulVec.cpp`](src/cpp/SIMDMatMulVec.cpp) is compiled with SIMD ISA flags (`-mavx2 -mfma -mf16c`, `-mavx512f`, `-mamx-tile`), preventing AVX code generation in other translation units. When AMX is available, the dot product uses tile matrix operations (`_tile_dpbusd`) for 8-bit quantized data with significantly higher throughput.

### GGUF Format Support

- **Version**: GGUF v3
- **Tensor types**: F32, Q5_K, Q5_1, Q4_K, Q4_K_XL, Q4_K_M, Q2_K, Q6_K, IQ3_XXS, IQ3_S, IQ3_XS, IQ2_S, IQ2_M (dequantized on-the-fly)
- **Metadata**: Architecture, layer count, head count, RoPE config
- **Tokenizer**: Embedded BPE vocab + merges (tiktoken-compatible)

### KV Cache

The KV cache stores key and value tensors as F32 arrays with shape `[numLayers, maxSeqLen, numKVHeads, headDim]`. For the 1.5B model with 2048 context:

- 28 layers × 2048 × 2 KV heads × 128 dim × 4 bytes × 2 (K+V) = **~112 MB**

The cache position is tracked and incremented during generation. [`clearKVCache()`](include/Model.hpp:115) resets the position and reallocates the cache arrays.

### LM Head Optimization

The LM head computation (logits = hidden × embeddings^T) is the most expensive operation for large vocabularies (151k+ tokens). Two optimized paths:

1. **CPU path** ([`LMHead::computeCPU()`](include/LMHead.hpp:64)): OpenMP-parallelized loop over vocabulary, dequantizing one embedding block at a time with SIMD dot products
2. **CUDA path** ([`LMHeadCUDA.cu`](src/cpp/LMHeadCUDA.cu)): Dequantizes the full embedding matrix once, uploads to GPU persistently, and uses `cublasSgemv` for each token position

Weight tying is detected automatically: if `output.weight` and `token_embd.weight` point to the same data, the LM head reuses the embedding matrix.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- **[np library](https://github.com/mgorshkov/np)** — NumPy-style arrays with SIMD/CUDA acceleration
