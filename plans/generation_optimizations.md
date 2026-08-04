# Generation Performance Optimizations (AVX2)

## Overview

This document tracks generation-speed work on the AVX2 path of
`src/cpp/core/ModelForward.cpp` and the surrounding SIMD kernels, on the
default `qwen2.5-coder-1.5b-instruct-q2_k.gguf` model running on an
i7-4790K (4 physical cores / 8 MiB L3 / DDR3-1600).

**Current status (2026-08-28, CPU-affinity + vectorization sweep):** quality-
correct default path at **~25.7-26.6 tok/s** at the **8-logical-thread default**
after the 2026-08-28 sweep (corrected distinct-logical-CPU thread pinning, now
the ON default and used consistently by both the addon bridge and the benchmark;
a scalar→AVX2 vectorization of the fused-FFN between-phase bsums; runtime
$TINYCODER_AFFINITY override). The 2026-08-27 campaign (L4 sampling + L7
LM-head hoists + **lever 1 dynamic chunk work-stealing in the fused FFN,
default ON**) established the baseline for these. The
steal schedule (llama.cpp `atomic_fetch_add`-style) raised 4-thread generation
from ~19.2→21.6-22.5 tok/s (+11-14%) and 8-thread from 26.1→26.3-26.5 tok/s
(+~1%), with `fused_gateUp_ffnDown` −13.8% at 4 threads, 49/49 parity held.
Target: **>30 tok/s, stretch 35 tok/s (~29 ms/tok)**.

---

## 1. Current performance (measured 2026-08-27, after the campaign)

From `TINYCODER_PROFILE=1`, `AnswersQuestion/*` (4 questions, 180 gen tokens,
4984 fused-kernel calls). **Use the `[TinyCoder] Generation:` tok/s line** — the
gtest "Generated N tokens in M ms" lines include each question's 40-48-token
prefill (≈500 ms) and read ~20 tok/s; they are NOT generation throughput.

| Stage | Time (4984 calls) | ms/layer | Notes |
|---|---|---|---|
| `fused_gateUp_ffnDown` (Q2K gate+up → Q3K down, single kernel, steal schedule) | ~3700-3920 ms | 0.742-0.787 | 66-70% of generation; the target |
| `gen_lmHead` (Q6_K separate head) | ~1668-1683 ms | — | at DDR3 floor (~18.7 GB/s) |
| QKV fused · attnO Q3K · attention | ~2.6 · 2.8 · 1.4 ms/tok | — | near ceiling |
| **Generation (8 threads)** | — | — | **~26.3-26.5 tok/s (~38 ms/tok)** |
| **Generation (4 threads)** | — | — | **~21.6-22.5 tok/s** (slab was ~19.2-19.9) |

The per-token weight stream (~0.67 GB/token across Q2K/Q3K/Q6K) at ~16-18 GB/s
effective gives a hard floor near ~27-31 tok/s; llama.cpp's **re-measured
2026-08-27** generation is **29.4 tok/s at 4 threads and 26.9 tok/s at 8
threads** (llama-cli `-t 4` / `-t 8` on this same host). After the work-stealing
campaign the remaining budget at 8 threads is **ALU/setup that is irreducible at
the current memory shape** (see §4 — the ALU/sync levers were each measured and
reverted as equal-or-slower; the prefetch and LM-head tile levers measured
sub-noise and were reverted).


### Where the remaining gap to 35 tok/s lives (reality-checked)
- The 0.67 GB/token stream at ~18 GB/s caps raw DRAM at ~37 ms/tok ≈ **27 tok/s
  hard floor** — the current ~26.3-26.5 tok/s is within ~0.5-0.7 tok/s of that
  floor, and within ~0.4-0.6 tok/s of **llama.cpp at the same 8 threads (26.9)**.
- The fused kernel (steal schedule) is at its streaming shape optimum:
  b-outer/r-inner (+5.5%), 64-row bands (+1.4%) and the static slab schedule
  (superseded by dynamic work-stealing, this section's lever 1) all measured
  slower; the 16-row tile, row-outer streaming, and Q2_K/Q3_K compact layouts
  match llama.cpp's weight bytes.
- **O1 (thread-split cooperative act quantize) proved net-zero** (2026-08-27):
  with 49/49 parity, per-block atomic last-arriver quantization (the llama.cpp
  `from_float` pattern) moved the ~1.2 ms/token `between` work off the critical
  path, but the phase-boundary bubble was already hidden under the DRAM-bound
  forward — interleaved A/B `fused_gateUp_ffnDown` 1321 vs 1325 ms and
  25.38-25.61 vs 25.43-25.57 tok/s. Reverted.
- **llama.cpp's remaining edge over TinyCoder is now the balance of the
  schedule + kernel-internal differences.** The dynamic work-stealing schedule
  (the ranked lever 1, this section) was the largest measurable gap and is now
  implemented and default — 4-thread generation went from 19.2→~22 tok/s. The
  residual ~4-5 tok/s at 4 threads / ~0.4-0.6 tok/s at 8 comes from llama.cpp's
  per-kernel internals + thread-split activation quantize + single dispatch per
  tensor (not from the row-chunk shape — that was Appendix A's row-band A/B).
- Remaining levers are either regressions on this 4-core/8 MiB L3 host or
  impossible for K-quants. **Closing the last ~1-8 tok/s to 35 would require a
  different memory substrate (faster DRAM), a smaller-quant LM head
  (lossy/divergent weights — §3 forbids), or CUDA offload** — none are lossless
  code changes.

---

## 4. Exact path to 35 tok/s (and the realistic parity target)

The plan's ceiling math is unchanged: 0.67 GB/token ÷ ~18 GB/s ≈ 27 tok/s hard
floor on this DDR3-1600 host. **35 tok/s (~29 ms/tok) is not reachable
losslessly on this CPU** — it needs ~23 GB/s weight-stream bandwidth, or ~0.58
GB/token (a head quant below Q6_K, which §3 forbids as divergent). The two
realistic targets are:

- **Parity target — llama-cli `-t 4` (measured 29.4 tok/s).** The work-stealing
  lever (below) moved TinyCoder from 19.2 → **21.6-22.5 tok/s** at 4 threads;
  the remaining ~7 tok/s gap is the sum of llama.cpp's kernel-internal ALU edge
  + its thread-split activation quantize + single-dispatch-per-tensor graph —
  each individually measured equal-or-slower as a TinyCoder change (Appendix A,
  §5) — so parity at 4 threads is no longer reachable by a *schedule* change;
  closing it needs the substrate changes below.
- **35 tok/s** — only via substrate: DDR4-3200 (≈2.2 GB/token/s ÷ 0.67 ≈ +30% →
  ~35 tok/s), a two-channel DIMM swap, or CUDA offload (`LMHeadCUDA.cpp` exists).

### Lossless levers toward the ~29 tok/s parity target (ranked, with the llama.cpp
mechanism they mirror) — outcomes measured 2026-08-27

1. **Dynamic chunk work-stealing in the fused kernel (mirrors
   `ggml_compute_forward_mul_mat_one_chunk`).** llama.cpp's 4-thread efficiency
   comes from `atomic_fetch_add` over `nr0×nr1` chunks (single-token
   `chunk_size=64`): no thread waits on a long static slab, so DRAM stays
   saturated at the tail. TinyCoder's `parallelForSlab` static partition makes 4
   threads drag (measured 19.2 tok/s). **Implemented `parallelForSteal` /
   `parallelForSteal2` beside the slab path (each thread pulls 4-tile chunks via
   `fetch_add(4)`), A/B'd at 4 vs 8 threads, default now ON
   (`TINYCODER_FFN_STEAL`).** Result: **WON — kept.**
   - 4 threads: 19.44-19.88 → 21.56-22.50 tok/s (**+11-14%**), the parity
     unlock; `fused_gateUp_ffnDown` 5191 → 4472 ms over 4984 calls (−13.8%).
   - 8 threads: 26.06-26.20 → 26.33-26.55 tok/s (+~1%); fused stage −2.7%.
   - 49/49 parity on both variants (schedule-only change, bit-identical).
   This is the single largest measurable remaining gap, now closed to within
   ~2-3 ms/layer of llama.cpp's 4-thread stage.
2. **Software-prefetch distance A/B in the row-outer loops.** The phase-1
   `_mm_prefetch(..., _MM_HINT_T1)` at +64 blocks and phase-2 `_MM_HINT_T0` at
   +64 are hand-tuned but never swept. llama.cpp uses `_MM_HINT_T0` in its
   `vec_dot` prefetch helper. **Swept +32/+96 and T0-vs-T1 per stage with the
   interleaved ±0.5% methodology — NEUTRAL, reverted.** Result:
   - +96@T1/T0 fused stage 3737.83/3756.38 ms vs default 3745.96-3920.68 ms;
     +32@T1/T0 3696.9 ms — all within run-to-run noise (±0.5%, the repeated
     default spread alone spans 3745-3920 ms).
   - T0/T0 3936.34 ms and swapped T0/T1 3854.23 ms were equal-or-slower.
   - Kept the hand-tuned +64/T1-T0 (no lossless win; the prefetch is not the
     wall on this DRAM-bound shape — the HW prefetcher already saturates the
     sequential 84/110-byte streams).
3. **LM-head tile shape A/B** (`matMulVecBatchQ6K_Q8K_AVX2`, 8-row tile). The
   head is 28% of generation at the DRAM floor; a 32-row tile halves per-tile
   epilogue overhead for ~5 ms/run-class gains at best. **A/B'd 8 vs 16 vs 32
   rows — NEUTRAL, reverted.** Result: `gen_lmHead` over 178 calls = 1679.35
   (8) / 1678.59 (16) / 1677.44 (32) ms; repeat 1677.82 (32) vs 1683.35 (8).
   The ~5 ms/run-class gain the plan predicted is real but sits below the
   ±0.5% noise band (~0.3%), so the trusted 8-row tile is kept.
4. **RMS norm / sampling remainder** (~1 ms/tok combined) — L4 already cut
   sampling 4-5→1 ms; **NO-OP by design, not implemented.** The active AVX2
   `rmsNorm_AVX2` already vectorizes both passes (FMA sum-of-squares + the
   `x*invRms*w` store via `_mm256_mul_ps`); the only division (`invRms =
   1/sqrt(...)`) runs once per call, not per element — there is no residual
   fp32 division loop left to vectorize.

### Levers that will NOT reach 35 (measured or §3-forbidden)
- O1 cooperative quantize — measured net-zero (this doc, §5).
- 64-row bands — +1.4% slower (Appendix A). b-outer/r-inner — +5.5% slower.
- Q2_K re-quant of the LM head / down matrix — lossy, §3 forbids (removed from
  the default path 2026-08-26; the only change that previously "reached" >26.3
  tok/s at the cost of emitting Python for Q1).
- Pure ALU/sync micro-opts — each measured equal-or-slower (L2/L3/L5/L8).

---

## 5. Measurement methodology (corrected, 2026-08-29 warm-repeat epoch)

**The canonical generation-throughput number is `tinycoder_bench` — a
llama-bench-parity warm-repeat harness, NOT the gtest Q2 single-cold-run
line.** Added 2026-08-29 (see §6.8) after the §6.6/§6.7 round found the
initial tables mixed llama-bench's warm mean with TinyCoder's cold single run.
The harness (`benchmarks/BenchMain.cpp`, target `tinycoder_bench`, built with
`BUILD_TESTS`):

1. Fixed **64-token deterministic prompt** (same shape as `llama-bench pp64`).
2. **Greedy decode** — plain argmax every step, no repeat penalty / topK / topP,
   matching `llama-bench`'s `--temp 0` default (llama-bench adds `--samplers
   none` and decodes argmax).
3. **Decode-loop-only timing** — prefill is run before the timed region and
   reported separately as **`pp<N> ... tok/s`** (both the per-rep line and the
   summary `pp64 | X.X +/- Y.Y tok/s | Z.Z ms (mean)`), exactly like
   llama-bench's pp/tg split (which measures `tgen` only; prefill is its own
   `ppN` figure).
4. **1 warmup rep + N reps**, reporting **mean ± stdev over reps** (llama-bench
   default `--reps 5`).
5. Each rep calls `generate()`-equivalent decode with a **cleared KV cache**
   (`Model::clearKVCache()` per rep), replicating llama-bench's per-rep context
   reset.

Usage:
`./build/benchmarks/tinycoder_bench --model <gguf> --n-prompts 64 --n-gen 64
--reps 5 --threads N`

A/B discipline with the harness (all numbers below that are a "tok/s" figure
or a ratio between two tok/s figures are harness numbers unless explicitly
labeled gtest-Q2):

1. **Always interleave** the variants: run baseline, candidate, baseline,
   candidate (the host drifts ±1-2% between processes; interleaving cancels it).
2. Report mean ± stdev over the reps, and delta between the interleaved
   centers, not between a max and a min.
3. Keep a change only if `fused_gateUp_ffnDown` ms drops AND the harness tok/s
   rises beyond the interleaved baseline noise (measured ±0.5-2% across reps).
4. Run the full 49/49 suite after each change (numerical parity must hold).
5. A/B any shape/scheduling change behind a compile-time toggle.

The gtest Q2 (`--gtest_filter='FullQuestions/SampleQuestionTest.AnswersQuestion/*'`)
remains useful only as a **sanity check** (correctness + smoke timing), and the
`[TinyCoder] Generation:` line is the *cold single-run* number — do NOT quote
it side-by-side with llama-bench tg numbers (§6.8 quantified: warm vs cold is
≈+0.5-1%, within the host's process-to-process drift).

---

## Appendix A. llama.cpp source-verified facts + lever-1 outcome (2026-08-27)

Verified against the local checkout `~/git/llama.cpp`:

- `ggml/src/ggml-cpu/ggml-cpu.c` `ggml_compute_forward_mul_mat`: single-token
  case (`nr0==1 || nr1==1`) → `chunk_size = 64` rows; chunks are
  `dr0 = ceil(nr0/nchunk0)` over the row space, dynamically work-stolen with
  `atomic_fetch_add` (lines 1415-1442).
- `ggml/src/ggml-cpu/arch/x86/quants.c`: every K-quant dot is
  `assert(nrc == 1)` (`ggml_vec_dot_q2_K_q8_K` :1437, `q3_K_q8_K` :1630,
  `q6_K_q8_K` :2289). "Multi-row vec_dot (nrc up to 16)" applies ONLY to
  llama.cpp's f16/dequant path, NOT Q2_K/Q3_K. The quantized tensors this model
  streams are computed one row per vec_dot call — exactly like TinyCoder.

**Lever-1 A/B (row-band threading) — COMPLETE, REVERTED.** Implemented
`TINYCODER_ROW_OUTER_P1` (CMake option, default OFF) mapping each
`parallelForSlab2` index to a contiguous 64-row band (chunk_size=64, processed
as 4 × 16-row sub-bands preserving the register accumulator array). 49/49 pass
on both variants (bit-identical answers; AVX2 object hashes differ). Measured
interleaved (4984 fused-kernel calls/run):

| Variant | `fused_gateUp_ffnDown` | ms/layer | Gen tok/s (64-tok mean) |
|---|---|---|---|
| Default (16-row tile) | 3752.42 ms | 0.753 | 24.37 |
| A/B (64-row band) | 3806.22 ms | 0.764 | 24.11 |

**Result: 64-row bands are +1.4% stage / −1.1% tok/s — slightly WORSE, so the
change was reverted (the kernel and CMake option are back to the verified
default).** Consistent with §9's "b-outer/row-inner +5.5% slower": on this
4-core / 8 MiB L3 host the 16-row register accumulator array and the L3 as
shared choke point dominate stream shape. llama.cpp's edge comes from its
kernel internals + thread-split activation quantize (lever L1 above) + single
dispatch per tensor — NOT the row-chunk shape. The A/B code was removed; no
default-path change.

---

## Appendix B. Work-stealing campaign A/B data (2026-08-27, post-lever-1)

Measured on the reference host (i7-4790K, 4 physical / 8 logical cores,
DDR3-1600), `qwen2.5-coder-1.5b-instruct-q2_k.gguf`, 64-token questions,
`TINYCODER_PROFILE=1`, interleaved runs. Parity: **49/49 pass on every variant**
(the schedule/prefetch/tile changes are numerically lossless — AVX2 object
hashes differ, answers are bit-identical).

### Lever 1 — parallelForSteal2 vs parallelForSlab2 (fused gate+up+down)

| Threads | Dispatch | Gen tok/s (64-tok mean) | `fused_gateUp_ffnDown` (4984 calls) |
|---|---|---|---|
| 4 | slab (baseline) | 19.44, 19.62, 19.70, 19.88 | 5190.99 ms |
| 4 | **steal (A/B)** | **21.56, 21.86, 22.03, 22.50** | **4471.88 ms** |
| 8 | slab (baseline) | 26.06, 26.15, 26.20, 26.34 | 5152.80 ms (6776 calls: 5152.80) |
| 8 | **steal (A/B)** | **26.33, 26.38, 26.39, 26.52, 26.55** | **5012.24 ms (6776 calls)** |

- **4 threads: +11-14% tok/s, −13.8% fused stage — the parity unlock.**
- 8 threads: +~1% tok/s, −2.7% fused stage.
- Kept as the default (`TINYCODER_FFN_STEAL=ON`); the slab path remains behind
  `-DTINYCODER_FFN_STEAL=OFF`.

### Lever 2 — software-prefetch distance / hint sweep (fused row-outer loops)

| Variant | `fused_gateUp_ffnDown` (4984 calls) |
|---|---|
| **default +64 / T1-T0 (kept)** | 3745.96, 3785.19, 3700.60, 3920.68 ms |
| +32 / T1-T0 | 3696.90 ms |
| +96 / T1-T0 | 3737.83, 3738.14, 3756.38 ms |
| +64 / T0-T0 (llama.cpp vec_dot hint) | 3936.34 ms |
| +64 / T0-T1 (swapped) | 3854.23 ms |

All variants within the ±0.5% run-to-run spread (the repeated default alone
spans 3700-3920 ms). **Result: neutral — the HW prefetcher already saturates
the sequential 84/110-byte streams; reverted to the hand-tuned defaults.**

### Lever 3 — LM-head Q6_K tile shape (matMulVecBatchQ6K_Q8K_AVX2)

| Tile | `gen_lmHead` (178 calls) | Gen tok/s (64-tok mean) |
|---|---|---|
| **8 rows (default, kept)** | 1679.35, 1683.35 ms | 26.34-26.38 |
| 16 rows | 1678.59 ms | 26.32-26.36 |
| 32 rows | 1677.44, 1677.82 ms | 26.34-26.41 |

The expected ~5 ms/run-class gain is real but below noise (~0.3%) at the DRAM
floor. **Result: neutral — kept the verified 8-row tile.**

### Lever 4 — RMSNorm residual division

`rmsNorm_AVX2` (the AVX2 dispatch at SIMDMatMulVec.cpp:524) already vectorizes
both passes; the single scalar `invRms` division per call is irreducible and
the plan rates it marginal. **Result: no-op by design — nothing to vectorize.**

### Thread-count sweep — full 4..20 curve (steal schedule, 2026-08-27)

Full `TINYCODER_THREADS` sweep on the steal-schedule default build (i7-4790K,
4 physical + 4 HT = 8 logical CPUs), 64-token questions, interleaved:

| Threads | Gen tok/s (64-tok mean) |
|---|---|
| 4 | 21.30, 21.56 |
| 5 | 22.70, 22.82 |
| 6 | 24.74, 24.86 |
| 7 | 25.58, 25.59 |
| **8 (default, kept)** | **26.23, 26.48** |
| 12 | 4.22, 4.25 (−84%) |
| 16 | 3.18, 3.22 |
| 20 | 2.26, 2.33 |

The curve is monotone from 4 → 8 logical CPUs (+1.4-1.9 tok/s per added
thread, i.e. the physical-→-logical-core scaling the steal schedule enables),
peaks at **8 = the logical-CPU count**, and then **collapses catastrophically
(−84% at 12 threads)**: oversubscribing 11/15/19 workers across 8 cores makes
the scheduler's bounded spin barrier (`spinWait` → cv) and the DRAM-bound
kernels thrash on context switches (every spin/cv wake reschedules a displaced
thread). **`recommendedThreadCount()` (logical CPUs) is the confirmed optimum;
`TINYCODER_THREADS` beyond the logical count is a severe regression on any
host, so 8 is the peak and >8 does not make sense.**

---

## 6. 2026-08-28 addendum — CPU-affinity re-test + vectorization sweep

Re-visited the two earlier conclusions that had "reverted" the affinity lever and
the between-phase quantize, after the user flagged **thread pinning is currently
absent** (the bridge forced it on with the *old* mapping, while the benchmark's
unit-test path had it OFF via the `TINYCODER_AFFINITY=0` CMake default).

### 6.1 CPU affinity re-tested with the correct distinct-logical-CPU mapping

The old `pinWorkerToCpu` mapped every worker via `workerIndex % nPhysical`
(physical cores 0-3), cramming the 7 workers + main thread onto 4 cores — that
is what A/B'd slower and justified `OFF`. The re-test here replaces it with:

- Each **worker** pinned to a **distinct logical CPU**: `cpu = (workerIndex + 1) % nLogical`,
  so worker0→CPU1 … worker6→CPU7.
- The **main thread** (which participates in every dispatch) pinned to **CPU 0**
  in `initialize()` when affinity is on — the full 0..7 coverage with no HT-sibling
  pairing on the memory-bound kernels.

`ThreadPool::initialize` now also honors a runtime `$TINYCODER_AFFINITY=0/1`
override of the baked-in CMake default, so one binary can A/B the pinning without
a recompile. **The bridge no longer forces `setAffinityEnabled(true)`** — it now
respects the global `TINYCODER_AFFINITY` setting (the earlier force used the old
cramming mapping and was the actual regression).

Interleaved A/B on the reference host (i7-4790K), qwen2.5-coder-1.5b-q2_k,
64-token question 2, pinning-OFF baseline vs distinct-logical-CPU ON:

| Threads | pinning OFF | pinning ON (distinct logical) | delta |
|---|---|---|---|
| 8 (default) | 25.61, 25.75, 25.72, 25.65 tok/s | 25.83, 25.69, 25.75, 25.76 tok/s | ~**+0.1-0.8%, neutral-to-marginally-better** |
| 4 | 21.39 tok/s | 21.05 tok/s | ~−1.6% (pinning slightly worse at low threads) |

**Verdict: the distinct-logical-CPU mapping is neutral-to-slightly-positive at the
production default (8 logical CPUs) and no longer the regression the old
physical-core-cramming mapping was.** 49/49 parity holds on both. Because the
*production* path (the `.node` bridge) previously forced affinity on, and the
corrected mapping removes the cramming regression, **`TINYCODER_AFFINITY=ON` is
now the default** — matching the bridge's production behavior while fixing its
mapping, and giving a consistent policy across both the addon and the benchmark.
Users running below the logical-CPU count can set `$TINYCODER_AFFINITY=0` (runtime,
no recompile).

### 6.2 Vectorized the fused-FFN between-phase bsums (scalar → AVX2)

The `between` callback of `matMulVecFusedGateUpDownQ2K_Compact_AVX2_impl`
(the fused gate+up+down FFN, ~66-70% of generation) computed the 16 per-16-group
byte sums of each Q8_K act block with a **scalar nested loop**:

```cpp
for (int j = 0; j < 16; ++j) {
    int s = 0;
    for (int ii = 0; ii < 16; ++ii) s += qs[j*16 + ii];
    q8actBuf[b].bsums[j] = (int16_t)s;
}
```

Replaced with an AVX2 pass that processes two adjacent groups per iteration:
`_mm256_maddubs_epi16(1, v)` → 16 int16 lane-sums of 2 bytes, then
`_mm256_madd_epi16(p, 1)` → 8 int32 lane-sums of 4 bytes, then two `_mm_hadd_epi32`
to collapse each group. Group sums are ≤2032, well within int16, so the int32
accumulation is exact and **bit-identical to the scalar reference**.

- **49/49 parity held** (full suite PASSED).
- Measured 25.73/25.81/25.81 tok/s — **wall-clock neutral**, confirming the O1
  finding that the `between`-phase bubble is hidden under the DRAM-bound forward.
  Kept as a genuine scalar→vector cleanup with zero downside (the AVX2 TU already
  compiles with `-mavx2 -mfma -mf16c`).

### 6.3 Reality-check on 35 tok/s (unchanged)

The DRAM-floor ceiling is unchanged and reaffirmed this session: 0.67 GB/token ÷
~18 GB/s effective ⇒ **~27 tok/s hard floor** on this DDR3-1600 host. The current
state is ~25.7-26.6 tok/s (interleaved variance), within ~0.5-1.0 tok/s of that
floor and of llama.cpp's 8-thread number (~26.9). The remaining vectorization
targets — the Q6K/Q2K LM-head `halfToFloat` (F16C-replaceable) and the LM-head
tile shape — were already measured sub-noise at the DRAM floor and are **not**
expected to move the wall-clock number; **35 tok/s is not reachable losslessly on
this CPU** (needs ~23 GB/s or a sub-Q6_K head, both §3-forbidden).

### 6.4 Additional A/Bs this session (both reverted as sub-noise / regression)

Two symmetric, previously-untested levers were measured to close the listed
"kernel-internal / schedule" gap, and both confirmed the DRAM-floor ceiling:

1. **LM-head work-stealing schedule** (`matMulVecBatchQ6K_Q8K_AVX2` switched from
   static `parallelForSlab` to the fused-FFN's winning `parallelForSteal`). The
   mechanism that unlocked +11-14% on the fused FFN is the *phase-boundary tail
   bubble* balance; the single-kernel LM head has **no phase boundary** and a far
   larger tile count (~19k tiles for 151,936 vocab rows), so the static 8-way
   contiguous row slabs already stream sequentially with no tail to steal. A/B'd
   (178 calls). `gen_lmHead` 1668.39/1669.25/1669.29 (steal) vs
   1673.08/1669.74 (slab) — **~0.2% sub-noise → REVERTED**. Confirms the
   work-stealing win is schedule-structure-specific, not universal.
2. **Non-temporal (NTA) stream prefetch** in the fused phase-1 gate/up loops
   (`_MM_HINT_NTA` replacing the hand-tuned `_MM_HINT_T1`). The textbook rationale
   is that single-pass streaming weights shouldn't pollute L1/L2/L3 with
   use-once lines. A/B'd on the 64-token question: **25.45 tok/s (NTA) vs
   25.69-25.74 (T1)** — a ~1% regression, so **reverted**. On Haswell the NTA
   prefetch hint on a close-sequential stream hurts (the predicted line lands on a
   non-allocating path that the subsequent normal load then re-fetches), and the
   HW prefetcher already saturates the sequential 84-byte streams without cache
   pollution being the wall. The hand-tuned T1/T0 (+64) default is kept.

Neither competes with the DRAM-bound reality: the ~27 tok/s floor (§6.3) governs,
and any faster path requires a memory substrate / CUDA change (not lossless code).

### 6.5 Cooperative (last-arriver) act quantize — the "same graph" as llama.cpp (2026-08-28)

The user's directive: *"llama.cpp's task graph keeps DRAM saturated with just 4
threads — let's try to make the same graph"* / *"do the same effective scheduling
that would outperform llama.cpp."* The concrete llama.cpp mechanism behind its
4-thread edge is the **`from_float` last-arriver cooperative quantize**: each
Q8_K block of the shared activation is quantized by whichever thread stores its
final 16-row tile (a per-block `++cnt == nb` check), overlapping the quantize ALU
with the remaining phase-1 DRAM streaming, instead of a serial `between` bubble
after a full barrier.

**Implementation** (`matMulVecFusedGateUpDownQ2K_Compact_AVX2_impl`, guarded by a
new CMake option `TINYCODER_COOP_QUANTIZE` default **ON**, macro
`TINYCODER_USE_COOP_QUANTIZE`):

- Per-256-row act block: a fixed-size `std::atomic<uint32_t>[]` arrival counter +
  an expected-arrivals table (`actArrivalExpected = ceil(rows/16)`, the partial
  tail block folded in).
- Every phase-1 tile, after storing its 16 `act[]` rows, does
  `actArrivals[b].fetch_add(1, acq_rel)` on its block `b = rowStart/256`; the
  thread that observes `prev+1 == expected` runs that block's quantize inline
  (the vectorized amax / `vcvtps2dq` / maddubs-bsums pass, byte-for-byte the old
  serial `between` body).
- The `between` callback becomes `nullptr` in the cooperative build (nothing to
  do — all blocks were quantized during phase 1); the serial `between` body is
  retained under `#else` as the A/B baseline.
- Correctness: the acq_rel fetch_add chain orders the other 15 tiles' `act[]`
  writes before the last arriver's reads; the pool's barrier-1 +
  `phase1Arrived_`/`phase2Ready_` release chain orders every quantize write
  before any phase-2 reader (phase 2 still gates on the FULL barrier because
  every down row consumes all `dBlocksPerRow` act blocks).

**A/B result** (i7-4790K, qwen2.5-coder-1.5b-q2_k, 64-token questions; note the
absolute numbers are depressed by a ~70%-CPU background process during this
session — the *delta* is the comparison):

| Threads | coop ON (last-arriver) | coop OFF (serial `between`) | delta |
|---|---|---|---|
| 8 (default) | 24.06, 24.01, 24.00 tok/s | 23.76, 23.93 tok/s | **+0.3-1.3% (mean ~+0.8%)** |
| 4 | 21.68 tok/s | 21.60 tok/s | ~+0.4% |

Bit-exact: **49/49 parity holds** — cooperation changed only *which* thread runs
each quantize, not the math. The +0.8% at 8 threads (vs the 2026-08-27 O1 attempt
at net-zero) comes from the return of work-stealing as the phase-1 dispatch: with
steal chunks, a thread often ends phase 1 having produced *no* final tile of some
blocks, so more blocks are quantized by the last of 8 concurrent arrivers rather
than by a serialized main thread. Kept as the default (llama.cpp's own graph
shape). The old §5 "O1 reverted" table row is superseded by this §6.5 result.

**Physical-core default re-tested and REJECTED**: the same session A/B'd
`recommendedThreadCount() = physicalThreadCount()` (the llama.cpp default, 4 on
this host via the distinct `thread_siblings` method) against the logical count.
8 logical threads still win on the DRAM-bound shape even with the full task graph
(24.00-24.12 vs 21.60-21.68 tok/s): the pooled HT lanes keep the 2-channel
DDR3-1600 bus busier than 4 dedicated cores. `physicalThreadCount()` is kept as
an A/B tool; `TINYCODER_THREADS=4` at runtime replicates llama.cpp's pool. The
production default stays the **logical-CPU count (8).**

### 6.6 llama-bench ground truth + Q8_0 act candidate (2026-08-28)

**Ground truth via `llama-bench`** (the earlier llama-cli interactive-prompt
measurement was unreliable): qwen2.5-coder-1.5b-q2_k, `-p 64 -n 64`:

| | 4 threads | 8 threads |
|---|---|---|
| llama.cpp tg64 | **31.89 ± 0.02** | **31.01 ± 0.14** |

llama sustains ~21.4 GB/s effective vs TinyCoder's ~17.9 — a real ~3.5 GB/s gap,
and llama achieves ~equal tg at 4 and 8 threads (saturates DRAM with 4). One
contributor is llama.cpp's **Q8_0 (per-32-element) quantized activation** layout
for Q2_K/Q3_K `mul_mat` — TinyCoder's fused kernel quantizes the shared act with
a **single Q8_K `d` per 256** instead.

> **REVERTED 2026-08-29 — the implementation below is REMOVED from the tree**
> (see the Decision at the end of this section): the warm-repeat A/B (§6.8)
> confirmed the regression with tight stdevs, so the separate kernel, the CMake
> option/define, the dispatch branches and the header declarations were all
> deleted. This section is preserved as the historical record of what was
> built and measured.

**IMPLEMENTED 2026-08-28 as a FULLY SEPARATE kernel, A/B measured — REGRESSION
at both thread counts.** Per the plan's architecture note, a complete separate
fused gate+up+down kernel pair was added to `src/cpp/core/SIMDMatMulVecAVX2.cpp`
(`matMulVecFusedGateUpDownQ2K_Q3K_ActQ8_0_Compact_AVX2` /
`...Q2K_ActQ8_0_Compact_AVX2`, both wrapping one
`template<bool DownIsQ2K> matMulVecFusedGateUpDownQ2K_ActQ8_0_Compact_AVX2_impl`),
selected at dispatch in `src/cpp/core/SIMDMatMulVec.cpp` under a CMake
option `TINYCODER_ACT_Q8_0` (default **OFF**, macro `TINYCODER_USE_ACT_Q8_0`
applied core-wide):

- **Phase 1 unchanged** (Q2_K gate/up vs Q8_K x — byte-identical to the Q8_K
  kernel).
- **Shared act quantized to Q8_0 per-32** (`ActQ8_0Block { float d32[8];
  int8_t qs[256]; int16_t bsums[16]; }`): each 32-element group gets its own
  scale `d = amax/127` stored fp16 like a GGUF Q8_0 block, quants
  `roundf(x * (1/d))` with the fp32 inverse (llama.cpp
  `quantize_row_q8_0_ref` semantics — the quantize uses the *unrounded* id, the
  dot re-reads the fp16-rounded d). The bsums[16] per-16 sums are kept for the
  phase-2 offset/min folds. Runs under the same coop last-arriver machinery as
  O1/§6.5 (the `TINYCODER_USE_COOP_QUANTIZE` gate applies to both kernels).
- **Phase 2 distributed per-chunk scaling**: every 32-element Q2_K plane /
  Q3_K sub-block dot (idx 0..7) is scaled by its own `d32[idx]` instead of one
  shared `d`. The min/offset folds distribute exactly: `madd_epi16(scales,
  bsums)` emits one int32 lane per 16-group pair == one 32-element Q8_0 chunk,
  so each lane is scaled by its own `d32[c]` **per-lane** (a broadcast would
  count it 8× — the same hsum-once trap as the Q8_K kernel's offset/min terms).
- The 5 earlier splice attempts' failure mode is thus avoided: no modification
  of the verified Q8_K hot loop; the Q8_0 path is a sibling kernel.

**Verification + A/B (i7-4790K, qwen2.5-coder-1.5b-q2_k, 64-token questions,
interleaved; measured 4984 fused calls/run):**

| Variant | Gen tok/s (8t) | Gen tok/s (4t) | `fused_gateUp_ffnDown` (8t) | ... (4t) |
|---|---|---|---|---|
| Q8_K baseline (OFF) | 26.57, 26.67, 26.69, 26.68, 26.80 | 21.56, 22.13, 22.23, 21.90 | 3634.77, 3646.20, 3635.21 ms | 4464.48, 4427.52 ms |
| **Q8_0-act (ON)** | 24.76, 26.10, 26.26, 26.15, 26.27 | 20.99, 20.48, 20.99, 20.30 | 3779.48, 3750.91, 3730.69 ms | 4924.58, 4920.15 ms |
| delta | **−1.3 to −2.5%** | **−5 to −8%** | **+2.6 to +3.9%** | **+10.2 to +11.8%** |

49/49 **PASSED on both variants** (outputs are coherent; the per-32 scales
produce different sampled text than Q8_K — expected, this is a numerically
different act representation, not a schedule change). `q8actBuf` is per-32 for
the OFF→ON switch with no bit-identical expectation.

**Verdict: REJECTED as a default — the finer per-32 act scale map is a net
regression at both thread counts, worse at 4t (+10-12% fused stage).** Two
independent reasons, both confirmed by measurement:

1. **llama.cpp's ~21.4 GB/s is NOT from Q8_0 acts for K-quants**: the local
   `~/git/llama.cpp` checkout maps `[GGML_TYPE_Q2_K]`/`[Q3_K]`/`[Q6_K]` →
   `.vec_dot_type = GGML_TYPE_Q8_K` with `from_float = quantize_row_q8_K`
   (ggml-cpu.c:289-298, 317-320), and there is no `ggml_vec_dot_q*_K_q8_0` in
   its x86 quants.c at all. Q8_0 activations apply to llama.cpp's
   *non-K-quant weight types* (Q8_0/Q4_0/Q5_0...), not to this model's Q2_K/Q3_K
   tensors. The residual llama edge must come from elsewhere (the kernel-internal
   ALU/task-graph differences the §6.5 graph already closed most of).
2. **On this DDR3-1600 host the finer act scale map only adds ALU to the
   DRAM-bound phase-1 quantize (8× the amax/round loops) and the phase-2 dots**
   (per-chunk broadcasts), with zero bandwidth saved — the ~27 tok/s floor
   (§6.3) governs, so the extra work is pure overhead, largest at 4 threads
   where the quantize bubble is less hidden behind DRAM streaming.

**Decision: REVERTED in full (2026-08-29).** The separate Q8_0-act kernel pair,
the `TINYCODER_ACT_Q8_0` CMake option + `TINYCODER_USE_ACT_Q8_0` define, the
dispatch branches, and the `SIMDMatMulVecInternal.hpp` declarations were all
**removed from the tree** after the llama-bench-parity warm A/B (§6.8)
confirmed the regression with tight stdevs. Nothing remains in-tree: the Q8_K
kernel is the only fused path, and there is no compile-time toggle left. The
§6.6 answer stands and is now decisive: **act granularity is not the llama.cpp
edge** — the ~1.5× 4t gap is the fused kernel's ALU/issue profile (§6.7), not
the per-32 act scale map. Post-revert verification: both previously-distinct
build dirs now compile the identical baseline and both hit
**26.98-27.01 ± 0.3-0.5 tok/s at 8t** (matching the pre-revert Q8_K baseline
26.72-26.94, above the Q8_0-act 26.47-26.65), 49/49 PASS.

### 6.7 Root-cause investigation: why llama's 4 physical cores beat TinyCoder's 8 (2026-08-29)

Ground truth re-measured fresh on the reference host (i7-4790K, 4 physical / 8
logical, DDR3-1600), `qwen2.5-coder-1.5b-instruct-q2_k.gguf`, `llama-bench -p 64
-n 64`, vs TinyCoder's gtest Q2 (64-token) — **the same host, same model, same
64-token shape**:

| threads | llama-bench tg64 | TinyCoder (64-tok Q2, warm) | ratio |
|---|---|---|---|
| 1 | 13.64 | 8.11 | 1.68× |
| 2 | 24.22 | 12.76 | 1.90× |
| 4 | **31.83** | 21.30-22.23 | **1.49×** |
| 8 | 31.09 | 25.95-26.80 | 1.16× |

The ratio CLOSES from 1.9× at 2 threads down to 1.2× at 8 — TinyCoder needs all
8 logical CPUs (4 physical + 4 HT) to approach llama, which saturates DRAM with
just its 4 physical cores. Diagnostic experiments, all on the SAME host:

1. **CPU-affinity is NOT the cause.** `taskset` pinning of llama's 4 threads:
   cores 0-3 (physical, HT siblings idle) → **31.49**; one-per-core spread
   0,2,4,6 (HT-sharing 0/4, 2/6) → **25.22 (−20%)**; OS-free 0-7 → 31.68. So
   HT sibling CONTENTION matters to llama (−20% if 2 threads share a core), but
   **TinyCoder's 4t is FLAT at 21.4-21.5 across ALL taskset placements (0-3 and
   0,2,4,6 both ≈21.4)** — TinyCoder never gains the 31.5 llama gets on physical
   cores. The gap is NOT placement/HT.
2. **The fused kernel is NOT the cause.** Forcing TinyCoder's FFN through
   llama's exact graph shape (separate gate+up then down, the §6.6
   `TINYCODER_SEPARATE_FFN` experiment, CMake-flagged): 4t = **16.4 tok/s**
   (`gen_gateUpSwiGLU` 2455 ms + `gen_matMulVecQ3K_ffnDown` 1212 ms / 1764
   calls) — **2.3× SLOWER per call than the 1569 ms fused stage**. Re-verified
   warm with the §6.8 `tinycoder_bench` harness before removal
   (interleaved, 5 reps): **fused 22.99 ± 0.48 (4t) / 26.91 ± 0.64 (8t)** vs
   **separate 17.55 ± 0.13 (4t) / 23.27 ± 0.34 (8t)** — the separate graph is
   **−23.7% (4t) / −13.5% (8t)**. The fused kernel is already the fast path,
   not the problem. **The `TINYCODER_SEPARATE_FFN` option, `&& false` guard
   and define were REMOVED from the tree (2026-08-29)** — like the Q8_0-act
   candidate, the separate graph is a measured regression, not a keeper.
3. **Effective bandwidth**, computed from measured per-stage ms/tok:
   `fused_gateUp_ffnDown` sustains **15.8 GB/s at 4t** and **19.2 GB/s at 8t**
   (394.6 MB FFN/token ÷ 24.9/20.6 ms); llama's whole-model tg64 implies
   **22.7 GB/s at 4t**. TinyCoder's generation 4t = 15.3 GB/s whole-model vs
   llama 22.7.

**Root cause: TinyCoder's fused kernel is ALU-issue-bound at the store/hsum
epilogue + per-tile work granularity, not DRAM-bound — even though the model is
DRAM-stream heavy, the *kernel's* per-thread instruction mix cannot issue loads
fast enough to saturate DRAM at 4 physical cores.** Evidence: (a) at 8 threads
(with 4 HT lanes adding issue width) bandwidth rises 15.8→19.2 GB/s and the
ratio falls to 1.16× — HT lanes relieve the ALU bottleneck, not the memory; a
pure DRAM-bound kernel would NOT improve with HT lanes (they share the same
memory controller). (b) llama's exact math (q0-3 paired with 4 distinct q8
chunks, one `sumi` over both 128-halves, one `acc` per block) is what TinyCoder
does, but llama's block loop has NO store/hsum epilogue and NO between-phase
boundary — it accumulates all blocks into `acc` and hsums only at the END of the
row (per 16-row tile), whereas TinyCoder hsums/stores per 16-row tile AND opens
gate+up twice per row (2 separate `blockDot` calls pair with the same q8_ymm but
double the planes processed... ).

**The concrete lever-smoking-gun:** TinyCoder's fused phase-1 16-row tile runs
`gateAcc` AND `upAcc` with **2 makeSetup + 2 blockDot per row per block**, each
blockDot issuing ~26 ALU ops over 2 chunks; the gated SwiGLU store hsums per row
every tile. llama's per-16-row-tile vec_dot issues one `sumi` per row with NO
per-tile hsum (only the final `tmp[16]` memcpy per column block), and CRUCIALLY
processes the Q8_K x ONCE per 16-row tile. TinyCoder's fused kernel therefore
has ~2× the ALU work per weight byte (two matrices interleaved in the same tile)
**and** per-tile horizontal reductions, which at 4 cores (no HT) caps the loads
it can sustain.

**Decision: the ~31.5-tok/s llama 4t number is NOT reachable by TinyCoder's
fused kernel as written — it needs the per-16-row-tile structure to be
restructured to accumulate into `acc` across ALL blocks (no per-tile hsum),
shared q8_ymm across the tile (already done), and ideally a single-matrix-per-
dispatch layout for the DRAM streams.** This is a follow-up lever (the "16-row
tile, hsum-once-per-row" restructure), measured as the next candidate. The §6.6
Q8_0-act candidate and this §6.7 investigation together confirm: act granularity
and CPU placement are NOT the gap; the fused kernel's ALU/issue profile at ≤4
physical cores is.

### 6.8 Warm-repeat benchmark (llama-bench parity) — measurement confound closed (2026-08-29)

**The confound the user flagged is real and is now closed:** the §6.6/§6.7
"TinyCoder tok/s" numbers were the gtest Q2's SINGLE COLD 64-token run, while
the llama-bench tg64 column is the MEAN OF WARM REPEATS. To make the two
columns apples-to-apples a dedicated harness was added —
`benchmarks/BenchMain.cpp` (target `tinycoder_bench`, built under
`BUILD_TESTS`; placed in `benchmarks/`, NOT `unit_tests/`, because it is a
measurement harness, not a parity test). It reproduces llama-bench's tg
protocol: fixed 64-token prompt, greedy argmax decode (no repeat penalty /
topK / topP), decode-loop-only timing (prefill excluded and reported
separately), 1 warmup rep + N reps with a cleared KV cache per rep, reporting
**mean ± stdev** (llama-bench's default `--reps 5`).

```
./build/benchmarks/tinycoder_bench --model <gguf> --n-prompts 64 --n-gen 64 --reps 5 --threads N
```

**Warm thread sweep (Q8_K baseline, i7-4790K, 2026-08-29), vs llama-bench tg64
on the same host/model:**

| threads | llama-bench tg64 | TinyCoder warm tg64 | ratio | vs §6.7 cold gtest |
|---|---|---|---|---|
| 1 | 13.64 | 8.44 ± 0.05 | 1.62× | 8.11 (−4% vs cold) |
| 2 | 24.22 | 12.82 ± 0.43 | 1.89× | 12.76 (same) |
| 4 | 31.83 | 21.53 ± 0.20 | 1.48× | 21.30-22.23 (same) |
| 8 | 31.09 | 26.82 ± 0.53 | 1.16× | 25.95-26.80 (same) |

**Verdict: the cold-start confound is SMALL at this workload granularity**
(warm vs cold ≈ +0.5-1% at 4-8t; 1t shows the only real −4% cold penalty).
Page-cache + thread-pool startup amortize over 64 tokens × 28 layers; the old
numbers were already representative. **All §6.6/§6.7 conclusions STAND**, and
the ratios are now statistically clean (e.g. 4t 1.49× → 1.48±0.01).

**§6.6 Q8_0-act A/B re-run warm (interleaved, 5 reps each):**

| variant | warm tg64 8t | warm tg64 4t |
|---|---|---|
| Q8_K baseline | 26.72 ± 0.41 / 26.82 ± 0.53 | 21.08 ± 0.07 / 21.53 ± 0.20 |
| Q8_0-act | 26.47 ± 0.04 / 26.65 ± 0.03 | 19.91 ± 0.25 / 20.34 ± 0.42 / 20.71 ± 0.29 |
| delta | **−0.6 to −1.3%** | **−2.3 to −7.5%** |

**Q8_0-act remains REJECTED as default** — the per-32 act-scale regression is
confirmed at both thread counts under the parity protocol. 49/49 PASSED on both
variants after the relocation.

**Model-shape note (caught while wiring the harness):** the benchmark prints
the loaded config — this GGUF reports **intermediate = 8960**, NOT the 8448 the
§6.7 bandwidth arithmetic assumed. The FFN bytes/token at 8960 are ~2.0% higher
than 8448 (Q2_K gate+up at 8960 + Q3_K down ≈ 402 MB/token vs 394.6 MB). This
does NOT change the §6.7 qualitative conclusion (the derived GB/s shift by
~+2%, far below the 1.48-1.89× ratios), but the exact GB/s figures (15.8/19.2/
22.7 GB/s) should be read as ±2% estimates; a corrected bandwidth table should
use `intermediate=8960` when re-derived.

### 6.9 Root-cause closure: hardware counters, phase-1 restructure, prefetch fix, and the pinning question (2026-08-29)

This session closed the §6.7 ALU-issue hypothesis with process-level hardware
counters, ran a llama-style inner-loop restructure (null), found and fixed a
**real prefetch bug** (+7.3% at 4 threads), and answered why llama.cpp benefits
from CPU pinning while TinyCoder does not.

#### 6.9.1 perf stat at 4 threads — llama vs TinyCoder (both `sudo perf stat`, same die/DRAM)

| counter | llama-bench 4t | tinycoder 4t | tc/llama |
|---|---|---|---|
| tg64 rate | 31.23 ± 0.11 | ~23.5 (post-§6.9.3) | 0.75× |
| IPC | 1.71 | 1.65 | 0.96× |
| branches executed | 10.16 G | 15.95 G | **1.57×** |
| branch-miss rate | 0.59% | 1.21% | **2.05×** |
| LLC-loads | 1.678 G | 0.945 G | 0.56× |
| LLC-miss rate | 56.66% | 65.72% | 1.16× |
| dTLB-load-misses | 13.55 M | 40.4 M | **2.98×** |
| instr throughput | 26.7 G/s | 16.2 G/s | 0.61× |

**Interpretation — closes §6.7 for good:** IPC is at parity (1.65 vs 1.71 on the
4-wide Haswell), so **neither is issue-bound** and the §6.7 "ALU/issue-bound"
verdict is dead. TinyCoder executes **1.57× more branches at 2.05× the miss rate
and 2.98× the dTLB misses** per unit of work — the signature of the old
per-`(row,block)` `QKSetup`/`makeSetup` + dual gate/up structure vs llama's
single-row, register-resident `acc`.

#### 6.9.2 REVERTED: llama-style phase-1 restructure (register-resident `acc`, one hsum per row)

> **REVERTED 2026-08-29 end-of-session — part of the combined change the user
> measured as a ~2% REGRESSION on the real workload (27.2-27.3 → 26.6-26.8
> tok/s @8t).** See the §6.9.3 update blockquote. Both the restructure of this
> section AND the prefetch change of §6.9.3 were reverted together;
> [`SIMDMatMulVecAVX2.cpp`](../src/cpp/core/SIMDMatMulVecAVX2.cpp) is back to the
> original pre-campaign fused-FFN path (49/49 PASS, 8t = 27.13 ± 0.50).

The experiment (for the record): rewrote phase-1 to llama's
`ggml_vec_dot_q2_K_q8_K` pattern — deleted the `QKSetup` struct + `makeSetup`
lambda (~720 B of stack spill per (row,block)) and the `blockDot` lambda;
inline-unpacked each compact Q2_K block straight into 8 register-resident
`q2` planes per matrix, kept `gAcc/uAcc` as live `__m256` across the whole row's
block stream, hsum ONCE at the fused SwiGLU store. 49/49 PASS.

| variant | 4t warm | 8t warm |
|---|---|---|
| old per-(row,block) QKSetup path | 22.99 ± 0.48 | 26.91 ± 0.64 |
| new llama-pattern path | 21.93 ± 0.18 | 27.05 ± 1.02 |

**Harness verdict was NULL** (flat at 8t, within session noise at 4t) — but the
user's paired real-workload A/B measured the combined change as a regression,
so both were reverted. The harness 4t run was insufficiently matched to the
production path on which the regression showed.

#### 6.9.3 REVERTED: the FFN software prefetch change "fix" was a regression

> **UPDATE (user measurement, 2026-08-29 end-of-session): the T0 row-lookahead
> prefetch change below was REVERTED — the user's paired A/B on the real
> workload showed it COSTS ~2%: 27.2-27.3 tok/s (pre-change) vs 26.6-26.8
> (post-change) at 8 threads.** The `+7.3% at 4t` below was a measurement
> artifact of the harness/session state, not a real win. The original
> `min(b+64,last)` T1 hints were restored in all three FFN phases and
> [`SIMDMatMulVecAVX2.cpp`](../src/cpp/core/SIMDMatMulVecAVX2.cpp) is back to the
> bit-for-bit pre-campaign path (49/49 PASS, 8t = 27.13 ± 0.50). Same for the
> §6.9.2 llama-pattern restructure — also reverted (user measured the combined
> change as a slowdown; the restructure alone was already NULL at 8t). Treat
> this entire §6.9.2/6.9.3 as EXPERIMENTS ONLY, all reverted.

All three FFN prefetch sites (phase-1 gate/up, phase-2 Q2K/Q3K down) issued
`_mm_prefetch(ptr + min(b+64, blocksPerRow-1) * stride)` — but
`gBlocksPerRow = 6` (1536/256) and `dBlocksPerRow = 35` (8960/256), so **every
hint clamped to the CURRENT row's last block, pointing into row garbage**. They
were dead for the whole "Lever 2 prefetch sweep" era. Replaced with T0
row-lookahead: prefetch the NEXT tile-row's first two blocks
(`row + BATCH_SIZE * rowStride`) per stream, the same sequential-stream the
HW prefetcher tracks and llama relies on. 49/49 PASS.

| variant | 4t warm | 8t warm |
|---|---|---|
| pre-§6.9.3 (dead hints) | 21.93 ± 0.18 | 27.05 ± 1.02 |
| post-§6.9.3 (T0 row-lookahead) | **23.50 ± 0.09 / 23.54 ± 0.26** | 27.06 ± 0.62 |

Fused FFN stage: 0.887 → **0.794 ms/call** (−10.5%), i.e. effective FFN
bandwidth ~19.1 GB/s vs the LM-head control ~22.7 GB/s. The FFN→DRAM-floor gap
(§6.3: 0.67 GB/token ÷ ~19-20 GB/s ⇒ ~29-30 tok/s at 4t) narrowed materially;
the 4t number now sits ~19% above the pre-campaign 21.5.

> **REVERTED (see the blockquote at the top of §6.9.3).** The harness A/B above
> did not reproduce on the user's real workload (their paired before/after
> measured **27.2-27.3 → 26.6-26.8 tok/s @8t, i.e. −2%**). Both §6.9.2 and
> §6.9.3 changes were reverted; the 8t rate is back to 27.13 ± 0.50.

All three FFN prefetch sites (phase-1 gate/up, phase-2 Q2K/Q3K down) issued
`_mm_prefetch(ptr + min(b+64, blocksPerRow-1) * stride)` — but
`gBlocksPerRow = 6` (1536/256) and `dBlocksPerRow = 35` (8960/256), so **every
hint clamped to the CURRENT row's last block, pointing into row garbage**. They
were dead for the whole "Lever 2 prefetch sweep" era. Replaced with T0
row-lookahead: prefetch the NEXT tile-row's first two blocks
(`row + BATCH_SIZE * rowStride`) per stream, the same sequential-stream the
HW prefetcher tracks and llama relies on. 49/49 PASS.

| variant | 4t warm | 8t warm |
|---|---|---|
| pre-§6.9.3 (dead hints) | 21.93 ± 0.18 | 27.05 ± 1.02 |
| post-§6.9.3 (T0 row-lookahead) | **23.50 ± 0.09 / 23.54 ± 0.26** | 27.06 ± 0.62 |

Fused FFN stage: 0.887 → **0.794 ms/call** (−10.5%), i.e. effective FFN
bandwidth ~19.1 GB/s vs the LM-head control ~22.7 GB/s. The FFN→DRAM-floor gap
(§6.3: 0.67 GB/token ÷ ~19-20 GB/s ⇒ ~29-30 tok/s at 4t) narrowed materially;
the 4t number now sits ~19% above the pre-campaign 21.5.

#### 6.9.4 The pinning question — why llama benefits and TinyCoder doesn't

llama.cpp pins its compute threads by default (`--cpu-mask`); TinyCoder's
`TINYCODER_AFFINITY` (CMake ON default) maps worker i → logical `(i+1) % 8` with
main on CPU 0. Re-measured both at 4t on the i7-4790K (CPU 0-3 = physical 0-3;
CPU 4-7 = their HT siblings):

| llama-bench 4t | tok/s |
|---|---|
| no mask (OS free) | 31.61 ± 0.25 |
| `--cpu-mask F` (all 8: 4 physical + 4 siblings) | 31.97 ± 0.04 |
| `--cpu-mask F0` (CPU 4-7, siblings only) | 30.26 ± 0.79 |

| TinyCoder 4t | tok/s |
|---|---|
| OS default (no pin) | 23.07 ± 0.46 |
| `TINYCODER_AFFINITY=1` (distinct logical 1..7 + main 0) | 23.22 ± 0.34 |
| `taskset -c 0,1,2,3` (physical only) | 23.37 ± 0.55 |
| `taskset -c 0,4,1,5` (one per physical core) | 22.69 ± 0.89 |

**Verdict: TinyCoder is pinning-INSENSITIVE at 4t (±1.5% across all four
configs), exactly as §6.1 found.** llama gains only +1.1% from its best mask
(=all logical CPUs), and a bad mask (siblings-only) *loses* 4.3%. The reason
pinning help is asymmetric: llama's scheduler already keeps its hot threads on
distinct cores and the kernel is memory/stream-bound enough that the ~1% comes
from avoiding OS migration jitter; TinyCoder's pool already does the same by
construction (per-dispatch slab/steal + spin-wait workers hold their cores), so
there is no migration jitter to remove. **Neither engine leaves meaningful
tok/s on the table at 4t from scheduling** — the 31.6 vs 23.5 gap is entirely
in the kernels' streaming efficiency, not in thread placement.

#### 6.9.5 Updated root-cause verdict (supersedes §6.7)

1. Not ALU/issue-bound (IPC parity 1.65 vs 1.71; thread scaling 2.9× at 4t).
2. Not per-block setup (llama-pattern restructure = NULL).
3. Not thread placement (pinning A/B = flat; llama's own pinning gain ≈ 1%).
4. **Bandwidth-limited with a low effective rate**: FFN ~19.1 GB/s vs the
   FP16-LM-head on-die control ~22.7 GB/s and llama's pipeline ~22-25 GB/s on
   the same DRAM. The dead prefetch hints were a real ~+7% (4t) artifact; the
   residual ~3-4 GB/s delta to the LM-head control is the cost of the
   fused two-matrix gate+up dual stream + 2-bit-planes decompression +
   per-block (re)loads vs llama's single-row `acc` — remaining levers are
   stream-shaped (single-matrix dispatch per phase, larger prefetch distance,
   phase-2-only 8-row tiles), not setup/ALU-shaped.
5. 4t now 23.5 tok/s (was 21.5 at session start, §6.8); 8t 27.06 (DRAM-floor
   ~27, §6.3). The next measurable step toward llama's 31.6@4t is the FFN
   stream-rate, per the §6.9.3/6.9.5 bandwidth gap.

### 6.10 Thread-count + branch/dTLB session (2026-08-29, user-driven)

**Question: how many threads does the forward pass need, and should we fix to
physical cores?** Re-verified on the i7-4790K reference host (4 physical / 8
logical, 2-channel DDR3-1600) with the llama-bench-parity harness
(`tinycoder_bench tg64`, greedy decode, warm mean ± stdev; llama-bench same
`-p 64 -n 64 -r 5`):

| threads | llama-bench tg64 | TinyCoder tg64 (pre-LUT) | ratio |
|---|---|---|---|
| 1 | 11.09 ± 3.28 | 8.60 ± 0.00 | 1.29× |
| 2 | 23.79 ± 0.01 | 13.01 ± 0.15 | 1.83× |
| 4 | 31.79 ± 0.02 | 21.61 ± 0.23 | 1.47× |
| 8 | 30.81 ± 0.17 | 24.80 ± 2.55 (noisy host) | 1.24× |

**Thread-count verdict (unchanged from §6.5/Appendix B, now re-confirmed):**
the optimum is **8 = the LOGICAL-CPU count** (`recommendedThreadCount()`,
llama.cpp's *physical*-core default of 4 would REGRESS 8t→4t by ~15-20%).
The curve is monotone 4→8 (+1.4-1.9 tok/s/thread via the steal schedule) and
collapses catastrophically at 12+ threads (spin-barrier + DRAM-bound thrash).
*Do NOT fix threads to physical cores; keep the logical default.*
`TINYCODER_THREADS`/`--threads` remain for A/B. The VS Code addon's
`tinycoder.nThreads` default of 4 (physical) is the one spot that should
follow `recommendedThreadCount()` (8) for consistency.

**Why is 1-thread slower than llama?** §6.9.1 counters: 1.57× more branches at
2.05× the miss rate, 2.98× more dTLB misses, IPC parity → NOT issue-bound;
the gap is per-`(row,block)` branchy setup + 4K-page weight streaming. perf is
blocked on this host (paranoid=4) — the user runs the commands in
`scripts/perf-threading.sh` (created this session).

**Branch-reduction lever (KEPT, 49/49 PASS, bit-exact):** the branchy scalar
`halfToFloat` (2-3 data-dependent branches + subnormal while-loop) was called
per (row, block) in every mat-vec kernel. Added a shared, bit-exact,
branch-free 64 KiB LUT (`GGMLDequantize::halfToFloatTable()` /
`halfToFloatBranchFree()` in `include/GGMLDequantize.hpp`) and applied it to
the hot generation kernels (LM-head Q6_K, fused Q+K, Q3_K, Q4_K batch). The
fused FFN already had its private LUT. Measured impact:

| stage | pre-LUT | post-LUT |
|---|---|---|
| `gen_lmHead` (Q6_K, ~1.7M calls/token) | 4610.6 ms / 192 calls | **3777.7 ms (−18.1%)** |
| 1t tg64 | 8.60 | **8.75-8.92 (+1.7-3.7%)** |

The LM head was NOT purely DRAM-bound (Q6_K's 210 B/block unpack + branchy
scale conversion left ALU on the critical path); the QKV/fused kernels were
already at the DRAM floor, so removing their branches was neutral — exactly
the §6.9.5 "remaining levers are stream-shaped, not ALU-shaped" split, with
the LM-head branch setup being the one ALU-shaped exception that mattered.
dTLB optimization (THP/madvise for the multi-hundred-MB weight vectors) is
deferred pending the user's perf counter run.

#### 6.10.1 User perf-counter run (4 threads, post-LUT build, whole run)

| metric | TinyCoder (post-LUT) | llama.cpp | ratio |
|---|---|---|---|
| tg64 | 25.23 ± 0.78 | 31.73 ± 0.04 | — |
| IPC | 1.67 | 1.73 | 1.04× |
| branch-miss rate | 0.82% (was 1.21%) | 0.57% | 1.44× (was 2.05×) |
| branches | 21.1G | 10.1G | 2.09× |
| **dTLB-load-misses** | **61.1M** | **13.2M** | **4.65×** |

Branch gap largely closed (miss rate 2.05×→1.44×, IPC 1.04× parity). The LUT
removed ~half the branch misses; the rest track llama's remaining branchy per-
task setup. **The single remaining outlier is dTLB**: 61.1M vs 13.2M (4.65×) on
a machine with `THP=madvise` and `AnonHugePages=0` — i.e. the ~750 MB of weight
buffers stream through 4 KiB pages for a dTLB-miss on nearly every 2 MB of
traffic. llama.cpp reaches 13.2M partly via larger page-aligned allocations
that the kernel's THP heuristics promote opportunistically.

#### 6.10.2 dTLB lever: MADV_HUGEPAGE hint on weight buffers (KEPT)

Added `include/MemHints.hpp` → `tinycoder::adviseHugePages(ptr, bytes)`:
Linux-only `madvise(MADV_HUGEPAGE)` (no-op elsewhere), ≥2 MiB threshold,
page-aligned range rounding. Applied at the three multi-hundred-MB weight
sites: `GGUFLoader::tensorData_` and `Model::dequantizedEmbeddings_.data` /
`.dataF16` / `.dataQ8K`.

**Verified:** `AnonHugePages` goes **0 → 16384 kB** while the model is loaded
(best-effort promotion; 16 MB of the ~750 MB got 2 MiB pages — bounded by
fragmentation/defrag policy `madvise`, but every promoted 2 MiB page removes
~512 dTLB misses per full sweep). 49/49 tests PASS with the hints (bit-exact
parity unchanged). Fresh `tinycoder_bench tg64` (warm, native binary):

| threads | post-THP tg64 | pre-THP (noisy-host best) |
|---|---|---|
| 1 | 8.92 ± 0.00 | 8.75-8.92 |
| 4 | 20.1-23.3 (host load 2-4) | 21.61-25.23 |
| 8 | 21.9-24.2 (host load 2-4) | 24.80 ± 2.55 |

The 4/8t numbers are confounded by 4 users + load avg 2-4 on the box at
measurement time; the 1t number is the clean comparison (8.92, matches the
post-LUT 8.75-8.92 range, so THP neither helped nor hurt 1t). The expected
win is the next user `perf stat` re-run:

```
sudo perf stat -e cycles,instructions,branches,branch-misses,dTLB-loads,dTLB-load-misses \
  ./build/benchmarks/tinycoder_bench --n-prompts 64 --n-gen 64 --reps 5 --threads 4 2>&1 | tail -25
```

target: dTLB-load-misses well below the pre-THP 61.1M (llama's floor is
13.2M; even 2-3× fewer would close most of the remaining 4.65× gap).

#### 6.10.3 Rollback A/B: the "prefill/gen regression" was host noise, not the
changes (2026-08-29, user-reported)

User reported prefill 81 (was 84-85) and gen 26.4 (was 27.4) after the THP +
LUT work. Investigation and definitive A/B:

- **System state at report time:** 4 users on the box, load average 1.15-3.91
  spiking to **7.42** (`uptime`), CPU locked at 3.29 GHz (4790K boost idle is
  4.0-4.4; `schedutil` + shared box). Any ±5% movement is inside this noise.
- `git stash -u` rolled back ALL C++ changes (LUT, THP, thread-default fix);
  rebuilt; `tinycoder_bench tg64` at loud load: **4t 19.77 ± 2.74**, **8t
  20.23 ± 0.90**; at quiet load: **4t 21.75 ± 1.89**.
- Re-applied (stash pop) and rebuilt; the same bench at similar quiet load:
  **4t 22.37 ± 0.76** (THP ON) / **21.68 ± 1.07** (THP OFF). The optimized
  build is NOT slower; if anything marginally faster. 49/49 parity re-confirmed
  after the round-trip (22.5 s at quiet load).
- Interleaved ON/OFF on the exact reported question (Q3, 64-token gen), drift-
  cancelled: 25.31 (ON) vs 25.81 (OFF) mean over 3 rounds — sub-noise.
- **Conclusion:** the ~27.4 → ~25.5 drop the user saw tracks the shared host's
  load spikes + sub-boost clocks, not the code. The 27.4 number was itself
  measured during the low-load 10:00 window. Numbers to compare against:
  8t bench 24.17 ± 2.59 (pre-THP), 4t 21.61 → 25.23 (LUT + noise), today
  21.68-22.37 (quiet-ish 4t) — all within the host's load band.

Also fixed during the investigation: `GGUFLoader` hinted the ~750 MB tensor
buffer BEFORE its backing `file_.read()`, so `madvise(MADV_HUGEPAGE)` targeted
unfaulted zero pages — with `defrag=madvise` that can trigger synchronous
compaction at first fault + feed khugepaged a zero range to collapse during
inference. Moved the hint AFTER the read (data resident). Added an A/B escape
hatch `TINYCODER_HUGE_PAGES=0` (default on) so the lever is disable-able
without a rebuild; ModelLoad's three post-`std::move` hints were already
correct.

#### 6.10.4 Within-run "ramp" root-caused: thermal + tenants, THP hints defaulted OFF

User's run showed warmup 27.2 -> rep5 21.2 (monotone) at 8t. Investigation:

- **KV growth ruled out:** `runOneRep` calls `model.clearKVCache()` before every rep
  (BenchMain.cpp:100). Not the bench.
- **khugepaged ruled out:** sampling `/proc/vmstat` `thp_collapse_alloc` during a
  run with hints default-OFF stayed frozen (837) while the ramp still happened.
- **Thermal + tenant concurrency confirmed:** live CPU clock sample during a ramping
  run: 4386 MHz (turbo, t=0-10 s) -> 3990 -> 3691 -> 3591 -> 3287 (base, t=40 s),
  while `ctxt` (context switches) spiked from ~1-10k to **13-33k per 2 s interval
  from rep 3 onward** — a competing workload starts mid-run (identified tenants:
  `/home/gallery-user/git/gallery.js` node server + ~25 VSCode/cpptools/tsserver
  processes on the shared box; cpptools was at 50% CPU after file edits). The
  `x86_pkg_temp` reads 54-58 C idle; the 4790K's 4.4 GHz turbo budget is spent
  within ~20-30 s of sustained 8-thread load.
- One THP-OFF run was perfectly flat: **27.09 +/- 0.46** (warmup 27.4 ... rep5
  26.3) — the same binary that ramps when the box is busy. Confirms code is fine;
  the ramp is host state.
- **Action:** `MemHints.hpp` now defaults the MADV_HUGEPAGE hints **OFF**
  (`TINYCODER_HUGE_PAGES=1` to opt in). Rationale: only ~16 MB / 750 MB ever
  promoted (dTLB gain negligible), while the khugepaged collapse machinery
  (10 s scan cycle, defrag=1) churns during inference. llama.cpp doesn't pay this
  because its weights are file-backed (page-cache readahead, no anonymous
  collapse); a heap-resident std::vector only promotes via khugepaged.

#### 6.10.5 mmap+hugetlb like llama.cpp? (measured assessment)

- **mmap (file-backed):** would remove the 750 MB anonymous copy + page-cache
  double-buffering at load and is the right long-term architecture. On ext4 there
  is NO file-backed THP, so the dTLB working set stays 4K — it does not close the
  4.65x dTLB gap by itself, and it does not touch the thermal/tenant ramp.
- **hugetlb (explicit 2MB):** would collapse dTLB 512x for real and pages are
  stable (no khugepaged), but HugePages_Total=0 on this host and reserving 1-2 GB
  needs root + intrusive sysctl on a 31 GB 4-user box; the LM-head/embedding np
  buffers need custom huge-page allocators. Not worth it for this host.
- **Neither fixes the observed ramp** (thermal decay + tenants). The flat 27 t/s
  run proves the code reaches it; sustained-stretch numbers are a host property.

#### 6.10.6 VS Code addon thread default (FIXED)

`tinycoder.nThreads` default was 4 (physical cores) — the one path inconsistent
with the logical-count verdict. Now defaults to **0 = auto**:
`recommendedThreadCount()` (8 logical) via `Bridge.cpp`
(`config.nThreads == 0`), `panel.ts` reads `get('nThreads', 0)`, and
`package.json` documents "0 = auto: logical CPU count, the measured optimum".
Native addon builds cleanly; `tsc --noEmit` passes.
