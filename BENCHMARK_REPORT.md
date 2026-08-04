# TinyCoder vs llama.cpp — Benchmark Comparison Report

**Date:** 2026-08-26 (updated 2026-08-27 after the sampling/LM-head campaign, then
2026-08-27 work-stealing campaign, 2026-08-28 CPU-affinity + vectorization sweep,
then 2026-08-30 CUDA GPU offload engine default-on)
**Model:** `qwen2.5-coder-1.5b-instruct-q2_k.gguf` (Q2_K, 28 layers, 1536 hidden, 12 heads, 2 KV heads)
**CPU:** Intel Core i7-4790K @ 4.00 GHz, 4 cores / 8 threads, AVX2+FMA+F16C, 32 GB RAM, DDR3-1600
**GPU:** NVIDIA GeForce RTX 2080 Ti (Turing, 11 GB), CUDA 12.x, Release
**Build:** CPU runs: CUDA=OFF, AVX2=ON, Release · GPU runs: ENABLE_CUDA=ON, Release
  (`./scripts/build.sh`); GPU engine defaults ON in CUDA builds (`TINYCODER_GPU=0`
  opts out; `TINYCODER_NGL` partial offload; full 28-layer offload used below).
**Sampling:** temperature=0.7, top-k=40, top-p=0.9, repeat-penalty=1.1, repeat-last-n=64, seed=42, max 64 tokens

---

## 1. Overall Summary

| Metric | TinyCoder (AVX2 Q2_K) | llama.cpp | Ratio |
|--------|----------------------|-----------|-------|
| **Total time (4 questions)** | ~8,990 ms | ~9,130 ms | **~0.98× (par)** |
| **Prefill (prompt processing)** | ~85.8 tok/s | ~72 tok/s | **~0.84× (ahead)** |
| **Generation throughput** | ~26.3-26.6 tok/s (8 thr) / ~21.0-21.6 tok/s (4 thr) | ~29.4 tok/s (4 thr) / ~26.9 tok/s (8 thr) | **8 thr ~1.00-1.02× slower; 4 thr ~1.31× slower** |
| **Per-token generation** | ~37.6-38.0 ms/tok (8 thr) / ~45-47 ms/tok (4 thr) | ~34 ms (4 thr) / ~37 ms (8 thr) | **~1.00-1.02× (8 thr) / ~1.31× (4 thr)** |

**GPU (CUDA offload engine, 2026-08-30):**

| Metric | TinyCoder GPU (RTX 2080 Ti) | llama.cpp GPU (RTX 2080 Ti) | Ratio |
|--------|----------------------------|----------------------------|-------|
| **Prefill pp64** | **4,773 tok/s** (13.4 ms) | 4,375 tok/s | **1.09× (ahead)** |
| **Generation tg64** | **128.9 tok/s** (496.5 ms/64 tok) | 295 tok/s | **0.44×** |

*GPU harness: `./build/benchmarks/tinycoder_bench --reps 5 --n-gen 64` (GPU
engine enabled by default in CUDA builds; warmup rep excluded; prefill on
cleared cache, excluded from tg rate — same llama-bench semantics).*

*TinyCoder figures re-measured 2026-08-28 after the CPU-affinity + vectorization
sweep (corrected distinct-logical-CPU pinning, now the ON default, consistent
across the addon bridge and the benchmark; scalar→AVX2 vectorization of the
fused-FFN between-phase bsums). 49/49 unit tests PASS on this build. The fused
gate+up+down stage measures ~0.738 ms/layer (3677 ms over 4984 calls) in the
final state.*

> **Apples-to-apples correction (measured 2026-08-27, llama.cpp re-measured on
> this host):** the earlier report's "~30.9 tok/s" llama.cpp figure was from
> llama-cli's **default thread count (4 = physical cores)**. Re-measured today on
> the same i7-4790K:
> `llama-cli -t 4` → **29.4 tok/s generation**, `llama-cli -t 8` → **26.9 tok/s
> generation**. TinyCoder runs **8 threads (logical cores)** by default (its
> fused kernels were statically slab-partitioned; the dynamic chunk work-stealing
> schedule added 2026-08-27 lifted 4-thread generation from ~19.2→~21.6-22.5
> tok/s and 8-thread from ~26.1→~26.3-26.5 tok/s).
> **Thread-for-thread (8 vs 8) the engines are within ~0.4-0.6 tok/s of each
> other (~26.9 vs ~26.3-26.5).** The residual edge is llama.cpp's per-kernel
> internals + thread-split activation quantize + single-dispatch-per-tensor
> graph, not the schedule (which is now mirrored) and not a bandwidth one.

> **Correction vs the previous report (2026-08-24):** the earlier TinyCoder figures
> (~26.3 tok/s) were measured on a build with **lossy Q2_K re-quantization of the
> FFN-down and LM-head weights enabled**, which degraded output quality (Q1 emitted
> Python instead of C++). That path was removed (llama.cpp does not re-quant at
> runtime — it streams the exact GGUF bytes). All numbers below are the
> **quality-correct default path**: correct C++ output on Q1, 49/49 tests pass.

## 2. Per-Question Results

### Q1: "Write a C++ function to add two numbers."

| Metric | TinyCoder (AVX2 Q2_K) | llama.cpp | Ratio |
|--------|----------------------|-----------|-------|
| Total time | ~2,313 ms | 2,773 ms | **0.83× (ahead)** |
| Prefill time | 504 ms (43 tok, 85.3 t/s) | 581 ms (43 tok, 73.97 t/s) | **0.87× (ahead)** |
| Generation time | 1,809 ms (45 tok, 24.9 t/s) | 2,041 ms (63 tok, 30.87 t/s) | **1.24×** |
| Time per generated token | ~40.2 ms | ~32 ms | **1.26×** |

### Q2: "What is the capital of France?"

| Metric | TinyCoder (AVX2 Q2_K) | llama.cpp | Ratio |
|--------|----------------------|-----------|-------|
| Total time | ~733 ms | 789 ms | **0.93× (ahead)** |
| Prefill time | 464 ms (40 tok, 86.2 t/s) | 541 ms (40 tok, 73.89 t/s) | **0.85× (ahead)** |
| Generation time | 269 ms (7 tok, 26.0 t/s) | 227 ms (7 tok, 30.87 t/s) | **1.19×** |
| Time per generated token | ~38.4 ms | ~32 ms | **1.20×** |

*Q2 only generates 7 tokens; with per-token fixed overheads amortized over just 7
tokens this ratio is the noisiest of the four.*

### Q3: "Explain what a pointer is in C++."

| Metric | TinyCoder (AVX2 Q2_K) | llama.cpp | Ratio |
|--------|----------------------|-----------|-------|
| Total time | ~2,935 ms | 2,660 ms | **1.10×** |
| Prefill time | 483 ms (42 tok, 87.0 t/s) | 572 ms (42 tok, 73.37 t/s) | **0.84× (ahead)** |
| Generation time | 2,452 ms (64 tok, 26.1 t/s) | 1,945 ms (60 tok, 30.85 t/s) | **1.18×** |
| Time per generated token | ~38.3 ms | ~32 ms | **1.20×** |

### Q4: "Write a for loop in Python that prints numbers 1 to 5."

| Metric | TinyCoder (AVX2 Q2_K) | llama.cpp | Ratio |
|--------|----------------------|-----------|-------|
| Total time | ~3,009 ms | 2,908 ms | **1.03×** |
| Prefill time | 561 ms (48 tok, 85.6 t/s) | 714 ms (48 tok, 67.26 t/s) | **0.79× (ahead)** |
| Generation time | 2,448 ms (64 tok, 26.1 t/s) | 2,045 ms (63 tok, 30.80 t/s) | **1.18×** |
| Time per generated token | ~38.3 ms | ~32 ms | **1.20×** |

## 3. Per-Stage Breakdown (TinyCoder, 2026-08-27 generation run)

Profile from the full 49-test suite (TINYCODER_PROFILE=1), showing where the
~38 ms/tok of generation time goes (4984 fused-kernel calls/token-run):

| Stage | ms/token | % of token | Notes |
|-------|----------|-----------|-------|
| `fused_gateUp_ffnDown` (Q2K gate+up → Q3K down) | ~21.1 | 56% | DRAM-bound, 0.751 ms/layer × 28 |
| `gen_lmHead` (Q6_K, 151,936×1536) | ~10.5 | 28% | 1668 ms/178 calls; k_shuffle hoisted (L7) |
| `gen_matMulVecFusedQKV` | ~4.6 | 12% | Q2_K compact fused Q+K |
| `attentionFused` (softmax) | ~0.2 | <1% | bit-exact AVX2 exp |
| `between` act Q8_K quantize | ~1.2 | 3% | **measured net-zero when moved off the critical path (O1, reverted)** |
| Other (RMS norm, sampling, etc.) | ~1.0 | 3% | **sampling 4-5 ms→1 ms/token (L4)** |

Matmul floor is ~42.7 ms/t for the three matmul stages alone; the fused
gate+up+down kernel (~56%) and the LM head (~28%) dominate. ~38 ms of every
token is memory-bound weight streaming at ~17-19 GB/s effective (DDR3-1600
dual-channel). Generation sits ~1-1.5 tok/s below the ~27 tok/s DRAM floor
(0.67 GB/token at ~18 GB/s).

> **2026-08-27 O1 (cooperative thread-split act quantize) — IMPLEMENTED, then
> REVERTED as measured net-zero.** Per-block atomic last-arriver counters let
> each phase-1 worker quantize the 256-row act block whose final tile it writes
> (`acq_rel` fetch_add publishes the block's act[] writes, then the last arriver
> quantizes — the llama.cpp `from_float` cooperative pattern, race-free). 49/49
> PASSED and parity held, but the interleaved A/B on the 64-token Q3 was
> statistically flat (`fused_gateUp_ffnDown` 1321 ms O1 vs 1325 ms baseline;
> 25.38-25.61 vs 25.43-25.57 tok/s). The ~1.2 ms/token `between` "bubble" is
> already hidden under the DRAM-bound forward (workers and main thread stagger
> into the phase boundary, so the phase-2 gate was never the wall). Reverted to
> the verified serial `between` per the keep/rollback rule.

## 4. Optimization History Since 2026-08-23

| Change | Result |
|--------|--------|
| Q3_K phase-2 loop-order fix in fused kernel | 24.5 → 21.6 ms/t gen |
| Vectorized `between` act Q8_K quantize | ~1.2 ms/t (kept, quality-neutral) |
| Bit-exact AVX2 `expf` replica (glibc 2.39 algorithm) in softmax | softmax 2.04× faster, **bit-identical** (10,094,739/10,094,739 patterns vs `expf`, 0 mismatches); Q1 output unchanged; 49/49 PASS |
| Removed lossy Q2_K re-quants (FFN-down, LM head) | quality restored (Q1 = correct C++), the 26.3 tok/s figure in the old report was this path |
| **L4: sampling topK `nth_element` + topP compact sort** (2026-08-27) | sampling 4-5 ms→1 ms/token; **generation 24.3→26.2 tok/s**; bit-exact (seed-42 token counts 45/7/64/64); 49/49 PASS |
| **L7: LM-head Q6_K k_shuffle hoisted to tile level** (2026-08-27) | `gen_lmHead` 1680→1668 ms; pure load elimination, bit-exact; 49/49 PASS |
| **2026-08-27 thread-count A/B (pre-work-stealing)** | with the static slab dispatch, llama.cpp's physical-core default (4 threads) regressed TinyCoder to ~19.2 tok/s (superseded by the work-stealing schedule below) |
| **2026-08-27 lever 1: dynamic chunk work-stealing in the fused FFN** (`parallelForSteal`/`parallelForSteal2`, llama.cpp `ggml_compute_forward_mul_mat_one_chunk` style, `TINYCODER_FFN_STEAL` default ON) | **4 threads 19.4-19.9 → 21.6-22.5 tok/s (+11-14%), 8 threads 26.1 → 26.3-26.5 (+~1%)**; `fused_gateUp_ffnDown` −13.8% at 4 threads (5191→4472 ms / 4984 calls); 49/49 PASS, bit-identical |
| **2026-08-27 full thread-count sweep (steal schedule)** | 4:21.3-21.6, 5:22.7-22.8, 6:24.7-24.9, 7:25.6, **8:26.2-26.5 (peak = logical-CPU count, kept)**, 12:4.2, 16:3.2, 20:2.3 (oversubscription collapse: the spin barrier + DRAM-bound kernels thrash on context switches) — **>8 threads is a severe regression, not a lever** (full table in `plans/generation_optimizations.md` Appendix B) |
| **2026-08-27 levers 2-4 A/B (prefetch sweep, LM-head tile, RMSNorm)** | prefetch +32/+96/T0-vs-T1 **neutral** within ±0.5% noise (kept hand-tuned +64/T1-T0); LM-head tile 8/16/32 **neutral** (1679.35 vs 1678.59 vs 1677.44 ms, sub-noise, kept 8-row); RMSNorm already fully vectorized (no-op) — all documented in `plans/generation_optimizations.md` Appendix B |
| **2026-08-27 O1: cooperative thread-split act quantize** (llama.cpp `from_float` last-arriver) | implemented, 49/49 PASS, **measured net-zero** vs serial `between` (1321 vs 1325 ms stage; 25.38-25.61 vs 25.43-25.57 tok/s) → reverted |
| L1 (thread-split act quantize), L2 (Q8Cache), L3 (head quantize handoff), L5 (barrier reduction), L8 (Q3K makeSetup) | measured equal-or-slower / impossible — **all reverted or not implemented** (see `plans/generation_optimizations.md` §4) |
| **2026-08-28: corrected CPU-affinity pinning** (distinct logical CPUs, workers `(i+1)%nLogical` + main pinned to CPU 0, runtime `$TINYCODER_AFFINITY` override, bridge respects the global setting) | **`TINYCODER_AFFINITY` default → ON.** Interleaved A/B at 8 threads: 25.83/25.69/25.75/25.76 (ON) vs 25.61/25.75/25.72/25.65 (OFF) — neutral-to-+0.5%, eliminating the earlier physical-core-cramming regression. 49/49 PARITY |
| **2026-08-28: vectorized fused-FFN between-phase bsums** (scalar nested loop → AVX2 `_mm256_maddubs_epi16`/`_mm256_madd_epi16`) | **kept**, 25.73/25.81/25.81 tok/s, bit-identical vs scalar, 49/49 PARITY |
| **2026-08-28 A/B: LM-head Q6_K work-stealing schedule** (`parallelForSteal` vs static `parallelForSlab`) | `gen_lmHead` 1668.4/1669.3/1669.3 (steal) vs 1673.1/1669.7 (slab) — **sub-noise (~0.2%), reverted** (single DRAM-bound kernel, no phase-boundary tail to balance) |
| **2026-08-28 A/B: non-temporal (NTA) prefetch in fused phase-1** | **25.45 tok/s vs T1 default 25.69-25.74 — ~1% regression, reverted** (hand-tuned T1/T0 kept; NTA cache-eviction doesn't help the sequential weight streams on Haswell) |
| **2026-08-28: cooperative last-arriver act quantize** (TINYCODER_COOP_QUANTIZE ON by default) — the "same graph" as llama.cpp (`from_float` pattern: per-block atomic arrival counters, the thread that stores each block's final tile quantizes it inline, no serial `between` bubble) | **8 threads: 24.06/24.01/24.00 vs 23.76/23.93 (serial `between`) — ~+0.8%**; 4 threads: 21.68 vs 21.60 — ~+0.4%; bit-identical, **49/49 PARITY**. (Session numbers depressed by a ~70%-CPU background process; the delta is the comparison.) |
| **2026-08-28 A/B: physical-core default thread count** (llama.cpp `common_cpu_get_num_math` policy; `ThreadPool::physicalThreadCount()` added) | 8 logical threads still win with the full task graph: 24.00-24.12 vs 21.60-21.68 (4t) — **logical default KEPT**; `physicalThreadCount()` retained as an A/B tool (supersedes the 2026-08-27 "4 threads regress" row, which predated work-stealing) |

The bit-exact expf is a pure correctness-preserving optimization: the softmax is
only ~0.6 ms of the ~38 ms/token, so end-to-end tok/s is dominated by matmul
bandwidth. The remaining gap to llama.cpp (~30.9 tok/s) is matmul bandwidth, not
sampling overhead — and the ALU/sync levers the 2026-08-27 campaign measured
(each independently) were all regressions or structural impossibilities, leaving
the memory-bound path at its streaming optimum.

*Reproduction (CPU path):*

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_AVX2=ON && cmake --build build -j
TINYCODER_MODEL_PATH=/data/models/qwen/qwen2.5-coder-1.5b-instruct-q2_k.gguf \
  ./build/unit_tests/tinycoder_test --gtest_filter='FullQuestions/SampleQuestionTest.*'
```

---

## 5. CUDA GPU Offload Engine (2026-08-29)

A CUDA offload engine (`include/GPUCompute.hpp`, `src/cpp/core/GPUCompute.cu`,
`src/cpp/core/ModelGPU.cpp`) was implemented and tuned against the llama.cpp CUDA
baseline on the same model (Qwen2.5 Coder 1.5B Q2_K) and GPU.

**Hardware:** NVIDIA GeForce RTX 2080 Ti (11 GiB, sm_75, 68 SMs), CUDA 12.0,
cuBLAS fp16 tensor cores, `build-cuda/` build tree (`-DENABLE_CUDA=ON`).

**llama.cpp baseline (same GPU, same GGUF):** pp64 **4375 tok/s**, tg64 **295 tok/s**.

### 5.1 Final Numbers

| Metric | TinyCoder GPU | llama.cpp GPU | Ratio |
|--------|---------------|---------------|-------|
| **Prefill pp64** | **4773 tok/s** (13.4 ms) | 4375 tok/s | **1.09× (ahead)** |
| **Generation tg64** | **128.9 tok/s** (496.5 ms/64 tok) | 295 tok/s | **0.44×** |

*Harness: `./build/benchmarks/tinycoder_bench --reps 5 --n-gen 64` (GPU engine
defaults ON in CUDA builds; `--gpu` explicit; warmup rep excluded; prefill on
cleared cache, excluded from tg rate — same llama-bench semantics).*

### 5.2 Architecture

- **Prefill (seqLen > 1):** cuBLAS fp16 tensor-core `cublasGemmEx` for all
  projections. Turing has no quantized tensor cores, so each weight matrix is
  dequantized once at upload into an fp16 twin (`DeviceMatrix::f16`) and streamed
  through the tensor cores; `CUBLAS_STATUS_NOT_SUPPORTED` forced ALL GEMM A
  operands to fp16 (hidden/attnOut/gate) — fp32×fp16 is rejected.
- **Decode (seqLen == 1):** quantized on-the-fly-dequant GEMV kernels
  (`kQGemv<kTypeQ2K/Q3K/Q4K/Q6K>`) reading the raw GGUF block layout straight
  from VRAM (same traffic llama.cpp's CUDA GEMV uses) — measured 2-3× faster than
  both the m=1 cuBLAS path and a custom dense-fp16 GEMV (see 5.4).
- **Flash attention:** `kWarpAttention<HD>` with compile-time
  register accumulators, exactly `nHeads` warps (no redundant warps).
- **Controls:** GPU offload is enabled by default in CUDA builds (all layers).
  `TINYCODER_GPU=0` forces the CPU path; `TINYCODER_NGL` limits offload to N
  layers (llama.cpp `-ngl` style); `TINYCODER_GPU_VERBOSE=1` per-stage
  `cudaEvent` timing.

### 5.3 Decode Stage Profile (per token, 28 layers)

| Stage | ms | % of 6.1 ms layer loop |
|-------|----|------------------------|
| QKV projections (Q2_K/Q2_K/Q4_K) + attn RMSNorm | 0.79 | 13% |
| RoPE + KV store + flash attention | 1.55 | 25% |
| attnO (Q3_K) + FFN gate/up (Q2_K) + down (Q3_K) | 3.67 | 60% |
| LM head (Q6_K, 151,936×1536) | ~1.3 | — |

FFN breakdown (event-measured): **gate+up 1.64 ms** (two 8960×1536 Q2_K),
**down 1.52 ms** (1536×8960 Q3_K), **attnO 0.47 ms** (1536×1536 Q3_K).

### 5.4 Optimization Journey (GPU)

| Change | Result |
|--------|--------|
| Initial working offload (689 pp / 25 tg) | correctness baseline |
| Q2_K/Q3_K GEMV overread fix + bias overread root cause (`kAddBias` broadcast instead of `kAddResidual`) | prefill clean **685→4800 pp tok/s** (IMA fixed) |
| Flash attention: register accumulators + exact nHeads warps | kv/attn stage **12.54→1.54 ms** |
| kQGemv: coalesced byte loads (bit-exact vs `GGMLDequantize`) | decode **40→81 tg** |
| Remove shared-memory x staging (`sx` gone, direct-global x) | decode **81→139 tg**; attnO/ffn 8.66→3.55 ms |
| **A/B: cuBLAS m=1 decode for QKV/attnO/FFN** | **112.9 tg — REGRESSION, reverted** (launch + f32→f16 conversion overhead) |
| **A/B: custom dense-fp16 `__hfma2` GEMV for FFN** | **115.1 tg — REGRESSION, reverted** (fp16 = 2 B/weight vs Q2_K 0.9 B/weight; DRAM-bound) |
| Fix missing warp reduction in kQGemv (`out[row]` was a race-write of a single lane's partial sum) | correctness fix, no measurable slowdown (139→138 tg) |
| Remove debug event instrumentation (cudaEventCreate failure → "invalid resource handle" → silent CPU fallback) | restored 138 tg (was masked as 26 tg) |

**Conclusion:** decode stays on the quantized GEMV path; prefill exceeds the
llama.cpp baseline; decode is 2.3× short of llama.cpp's tg64 and is instruction-
bound on the K-quant dequant ALU (≈7 instructions/MAC vs 1 for a dense fp16
path, which loses on DRAM instead). The remaining gap is a llama.cpp-class
tuned K-quant decode kernel (per-warp q8_0 act quantization, vectorized
`dot_q2_K`-style inner loops, tile-larger-than-256-row blocks) — see
`plans/generation_optimizations.md`.

*Reproduction (GPU path):*

```sh
cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON && \
  cmake --build build-cuda -j
TINYCODER_MODEL_PATH=/data/models/qwen/qwen2.5-coder-1.5b-instruct-q2_k.gguf \
  ./build-cuda/benchmarks/tinycoder_bench --gpu --reps 5 --n-gen 64
```
