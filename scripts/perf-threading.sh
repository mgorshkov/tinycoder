#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Thread-count + perf counter measurement harness (2026-08-29).
#
# Background (from plans/generation_optimizations.md §6.9.1, session perf):
#   llama   4t: IPC 1.71, branches 10.16G, branch-miss 0.59%, dTLB-miss 13.55M
#   TinyCoder 4t: IPC 1.65, branches 15.95G, branch-miss 1.21%, dTLB-miss 40.4M
# => TinyCoder is NOT issue-bound (IPC parity); it executes 1.57x more
#    branches at 2.05x the miss rate and 2.98x the dTLB misses per unit of
#    work. This script re-measures those counters on THIS host so we can
#    verify the branch/dTLB reduction after code changes.
#
# NOTE: `perf stat` on this host needs CAP_PERFMON (perf_event_paranoid=4).
#   Either run the perf block under sudo, or lower the limit once:
#     echo 2 | sudo tee /proc/sys/kernel/perf_event_paranoid
#   (2 denies kernel profiling only; the user-space counters below work.)
# -----------------------------------------------------------------------------
set -u

MODEL="${TINYCODER_MODEL_PATH:-/data/models/qwen/qwen2.5-coder-1.5b-instruct-q2_k.gguf}"
BENCH=./build/benchmarks/tinycoder_bench
# Prefer the CUDA-enabled build (2026-08-29: rebuilt with -DGGML_CUDA=ON;
# measured tg64 @ -ngl 99 = 290.8 t/s vs 27.35 CPU-only on the RTX 2080 Ti).
# Set LLAMA to the CPU-only build path to re-run the plain-CPU comparison.
LLAMA="${LLAMA_GPU:-$HOME/git/llama.cpp/build-cuda/bin/llama-bench}"
[ -x "$LLAMA" ] || LLAMA="$HOME/git/llama.cpp/build/bin/llama-bench"
COUNTERS="cycles,instructions,branches,branch-misses,dTLB-loads,dTLB-load-misses,LLC-loads,LLC-load-misses,LLC-load-misses,context-switches,cpu-migrations"

t_stats() { # $1 = label, rest = command...
    local label="$1"; shift
    echo
    echo "==================================================================="
    echo "  $label"
    echo "==================================================================="
    "$@"
}

demand_build() {
    [ -x "$BENCH" ] || { echo "building $BENCH..."; cmake --build build --target tinycoder_bench -j >/dev/null; }
}

demand_build

# 1. ---- Warm thread sweep (no sudo needed) --------------------------------
# llama-bench tg64 protocol (pp64/tg64, greedy, warm mean of --reps 5).
for T in 1 2 4 8; do
    t_stats "TinyCoder tg64 @${T} threads (x2 for noise)" \
        env TINYCODER_MODEL_PATH="$MODEL" "$BENCH" --n-prompts 64 --n-gen 64 --reps 5 --threads "$T" 2>&1 | grep '^tg64'
    # llama-bench pads columns: `|            tg64 |` — match just 'tg64'.
    t_stats "llama-bench tg64 @${T} threads" \
        "$LLAMA" -m "$MODEL" -p 64 -n 64 -t "$T" -r 5 2>&1 | grep 'tg64'
done

# 1b. ---- GPU offload (llama.cpp CUDA backend) -----------------------------
# The block below AUTO-DETECTS a usable CUDA backend (llama-bench -ngl 1
# prints a `CUDA` backend row) and only runs the GPU measurement when llama
# contacts an NVIDIA device. Otherwise it prints the exact rebuild command
# instead of silently producing CPU-only numbers that look like GPU results.
#
# The Q2_K 1.5B model is ~750 MB -> fully offloadable: -ngl 99 puts all 28
# layers on the 11 GB card. Sweep {0, 99} to show the CPU-only vs full-GPU
# delta; both use the same tg64 protocol (pp64/tg64, warm mean of --reps 5)
# for apples-to-apples comparison with the section-1 CPU sweep.
if "$LLAMA" -m "$MODEL" -p 1 -n 1 -ngl 1 2>&1 | grep -qiE "CUDA[0-9]|NVIDIA|backend.*cuda"; then
    echo "[perf] CUDA backend detected in llama.cpp build."
    NGLS=(0 99)
    for NGL in "${NGLS[@]}"; do
        t_stats "llama-bench tg64 @GPU offload -ngl ${NGL} (t=4)" \
            "$LLAMA" -m "$MODEL" -p 64 -n 64 -t 4 -ngl "$NGL" -r 5 2>&1 | grep 'tg64'
    done
else
    echo
    echo "==================================================================="
    echo "  CUDA backend NOT detected in this llama.cpp build."
    echo "  GPU hardware present: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "  Rebuild llama.cpp with CUDA to measure GPU performance:"
    echo "    cmake -B ~/git/llama.cpp/build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release"
    echo "    cmake --build ~/git/llama.cpp/build-cuda --target llama-bench -j"
    echo "  then set LLAMA=~/git/llama.cpp/build-cuda/bin/llama-bench at the"
    echo "  top of this script and re-run."
    echo "==================================================================="
fi

# 1c. ---- GPU utilization proof (nvidia-smi during llama-bench -ngl 99) ----
# Proof that the GPU is genuinely engaged: snapshot clocks/mem/util in the
# background while a full-offload decode runs, then show the samples.
if command -v nvidia-smi >/dev/null 2>&1 && "$LLAMA" -m "$MODEL" -p 1 -n 1 -ngl 1 2>&1 | grep -qiE "CUDA[0-9]|NVIDIA|backend.*cuda"; then
    t_stats "GPU utilization while llama-bench -ngl 99 runs (nvidia-smi sample)" \
        bash -c 'nvidia-smi -l 1 > /tmp/nvsmi.log & NV=$!; "$LLAMA" -m "$MODEL" -p 64 -n 64 -t 4 -ngl 99 -r 3 >/dev/null 2>&1; kill $NV 2>/dev/null; awk "/Utilization|Memory-Usage|Clocks/{print}" /tmp/nvsmi.log | head -8'
fi

# 2. ---- perf stat whole-run counters (needs sudo / paranoid<=2) -----------
# Whole-run counters include the one-time 750MB model load (both engines pay
# it); the decode-phase share dominates at --reps 5 (2.5-3s of ~4s) so the
# BRANCHES / dTLB ratios are still representative, exactly as in §6.9.1.
if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    PRFX="sudo"
else
    PRFX=""
fi
for T in 4; do
    t_stats "perf stat TinyCoder @${T}t" \
        ${PRFX} perf stat -e "$COUNTERS" env TINYCODER_MODEL_PATH="$MODEL" "$BENCH" \
            --n-prompts 64 --n-gen 64 --reps 5 --threads "$T" 2>&1 | tail -25
    t_stats "perf stat llama-bench @${T}t" \
        ${PRFX} perf stat -e "$COUNTERS" "$LLAMA" -m "$MODEL" -p 64 -n 64 -t "$T" -r 5 2>&1 | tail -25
done

# 3. ---- perf record: decode-loop hotspots (symbol/line attribution) -------
# For the 1-thread gap: profile a whole run, then inspect which kernel
# functions own branches + dTLB misses.
t_stats "perf record (1t, decode-dominated) -> /tmp/tc.perf.data" \
    ${PRFX} perf record -F 999 --call-graph dwarf -o /tmp/tc.perf.data \
        env TINYCODER_MODEL_PATH="$MODEL" "$BENCH" --n-prompts 64 --n-gen 64 --reps 5 --threads 1 >/dev/null 2>&1
[ -f /tmp/tc.perf.data ] && ${PRFX} perf report -i /tmp/tc.perf.data --stdio | head -45

echo
echo "done."