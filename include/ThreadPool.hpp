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

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace tinycoder {

    /// @brief Portable pause hint for spin-wait loops.
    ///
    /// On x86 this emits the PAUSE instruction (reduces power draw and avoids
    /// memory-order pipeline stalls while spinning). On other architectures it
    /// is a no-op. Used by the persistent-worker scheduler's spin barrier.
    inline void spinPause() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        _mm_pause();
#else
        // Portable fallback: a compiler barrier that prevents the loop from
        // being optimized into a tight unbounded spin.
        std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
    }

    /// @brief Reusable thread pool that replaces OpenMP parallel regions.
    ///
    /// Threads are created once during initialization and reused across all
    /// parallel for loops, eliminating the thread creation/destruction overhead
    /// of OpenMP's fork-join model.
    ///
    /// # Persistent-worker / graph-style scheduler
    ///
    /// This pool uses a persistent-worker scheduler: workers stay resident across the whole
    /// layer loop and synchronize through a fast lock-free spin barrier instead
    /// of a mutex + condition_variable on every dispatch.
    ///
    /// The dispatch protocol is:
    ///   1. The main thread publishes the task (function, index range, worker
    ///      count) using plain stores, then sets `hasWork_` (release) and bumps
    ///      the generation counter `taskId_` (acq_rel).
    ///   2. Workers spin on `hasWork_`/`taskId_` to detect new work, process
    ///      their share via an atomic index fetch, then spin on `taskId_` to
    ///      wait for the next task (the "ack" that the previous task is done).
    ///   3. To avoid burning CPU while idle (e.g. between tokens), workers fall
    ///      back to blocking on a condition_variable after a bounded spin. The
    ///      main thread wakes any blocked workers via `cv_.notify_all()`.
    ///
    /// The generation-counter completion scheme safely supports consecutive
    /// parallelFor calls: each parallelFor gets a unique taskId, and workers
    /// wait for taskId to change before looping back. This eliminates the race
    /// where hasWork_ toggles between "task done" and "new task ready" signals.
    class ThreadPool {
    public:
        static ThreadPool &instance();

        void initialize(size_t numThreads);
        size_t numThreads() const { return numThreads_; }
        bool isInitialized() const { return initialized_; }

        /// @brief Recommended initial thread count.
        ///
        /// Defaults to the number of LOGICAL CPUs (8 on the i7-4790K reference
        /// host) — measured 2026-08-28 to be a clear WIN over llama.cpp's
        /// physical-core default (4) even after the task-graph levers (dynamic
        /// chunk work-stealing + last-arriver cooperative act quantize):
        /// 24.06/24.01 tok/s at 8t vs 21.68/21.60 at 4t on the 64-token
        /// questions. The finer 8-way row split plus the pooled HT lanes keeps
        /// DRAM busier than 4 dedicated cores on this 2-channel DDR3-1600
        /// memory-bound shape. physicalThreadCount() (llama.cpp's policy)
        /// remains available for A/B; `TINYCODER_THREADS` overrides at runtime.
        static size_t recommendedThreadCount();

        /// @brief Number of physical cores (HT siblings folded), detected like
        /// llama.cpp's common_cpu_get_num_physical_cores() (distinct
        /// /sys/.../topology/thread_siblings masks on Linux; logical/2
        /// elsewhere).
        static size_t physicalThreadCount();

        /// @brief Enable/disable CPU affinity pinning for worker threads.
        ///
        /// When enabled, each worker thread is pinned to a distinct physical core
        /// (or logical CPU) at creation time. This avoids cache thrashing between
        /// HT siblings on memory-bound kernels (e.g. the LM head mat-vec and FFN
        /// matmuls). Must be called before initialize().
        ///
        /// @param enable true to pin workers to distinct CPUs
        void setAffinityEnabled(bool enable) { affinityEnabled_ = enable; }
        bool affinityEnabled() const { return affinityEnabled_; }

        void parallelFor(uint32_t start, uint32_t end,
                         const std::function<void(uint32_t)> &func);

        void parallelFor2D(uint32_t dim1, uint32_t dim2,
                           const std::function<void(uint32_t, uint32_t)> &func);

        /// @brief Parallel-for with static (contiguous-slab) work partitioning.
        ///
        /// Divides [start, end) into one contiguous slab per participating thread
        /// (workers + main) and runs each slab on its assigned thread, instead of
        /// the fine-grained per-item `fetch_add` dispatch used by parallelFor.
        /// This removes the shared-atomic contention on large-tile dispatches
        /// (e.g. the ~192-tile prefill batch kernels). Intended for workloads
        /// where every item is roughly equal cost, so the static partition is
        /// naturally load-balanced. P6 in plans/prefill_optimization_plan.md.
        void parallelForSlab(uint32_t start, uint32_t end,
                             const std::function<void(uint32_t)> &func);

        /// @brief Single-launch two-phase parallel-for with static slabs and a
        /// barrier-free (spin-flag) phase transition.
        ///
        /// Runs phase1 over [start1, end1) on all threads (one contiguous slab
        /// per thread), then a single `between` callback on the MAIN thread
        /// (e.g. to serialize the Q8_K quantization of a shared activation
        /// vector), then phase2 over [start2, end2) on all threads.
        ///
        /// The phase transition uses a plain acquire/release flag: workers that
        /// finish phase1 spin on `phase2Ready_` until the main thread publishes
        /// `between()`'s result and releases the flag. This merges what would
        /// otherwise be TWO parallelForSlab launches into ONE — eliminating one
        /// task publication, one generation bump, and one full spin/condition-
        /// variable barrier per invocation. For the fused FFN kernel (invoked
        /// once per layer per token) that removes ~28 extra task dispatches and
        /// cv wake-ups per generated token.
        ///
        /// Precondition: the caller guarantees phase2 reads only data written by
        /// phase1 + between (the flag orders those writes). Workers participate
        /// in both phases; the slab partition for each phase is computed
        /// independently from [start1,end1) / [start2,end2).
        void parallelForSlab2(
                uint32_t start1, uint32_t end1,
                const std::function<void(uint32_t)> &phase1,
                const std::function<void()> &between,
                uint32_t start2, uint32_t end2,
                const std::function<void(uint32_t)> &phase2);

        /// @brief Parallel-for with dynamic chunked work-stealing (llama.cpp
        /// `ggml_compute_forward_mul_mat_one_chunk` style dispatch).
        ///
        /// Workers (and the main thread) pull [chunkSize]-sized index chunks from
        /// a single shared atomic via `fetch_add(chunkSize)` instead of running a
        /// statically partitioned contiguous slab. This keeps DRAM saturated at
        /// the tail of memory-bound kernels: no thread can idle on a long static
        /// slab while others finish their share, so the last sub-chunk of work is
        /// stolen by whoever is free (the phase-boundary bubble llama.cpp's
        /// `atomic_fetch_add` scheduling removes).
        ///
        /// Intended for single-token generation kernels (the fused gate+up+down
        /// FFN, ranked lever 1 in plans/generation_optimizations.md) where tile
        /// counts are small (hundreds) and per-tile cost is weight-stream-bound
        /// and roughly uniform. Use a chunk of a few tiles (e.g. 4) to amortize
        /// the shared-atomic contention while keeping the tail balanced.
        void parallelForSteal(uint32_t start, uint32_t end, uint32_t chunkSize,
                              const std::function<void(uint32_t)> &func);

        /// @brief Single-launch two-phase parallel-for with chunked work-stealing
        /// (the steal counterpart of parallelForSlab2).
        ///
        /// Same two-phase protocol as parallelForSlab2 (phase1 → `between` on the
        /// main thread → phase2, one task publication, spin-flag phase
        /// transition), but each phase pulls [chunk1/chunk2]-sized index chunks
        /// from the shared atomic instead of running static slabs. For the fused
        /// FFN kernel this replaces the static 4- or 8-way row partition of
        /// parallelForSlab2 — llama.cpp's single-token `nr0x nr1` loop steals
        /// 64-row work chunks dynamically, which is where its 4-thread edge over
        /// TinyCoder's static slabs comes from (measured 19.2 vs 29.4 tok/s at 4
        /// threads, plans/generation_optimizations.md lever 1).
        void parallelForSteal2(
                uint32_t start1, uint32_t end1, uint32_t chunk1,
                const std::function<void(uint32_t)> &phase1,
                const std::function<void()> &between,
                uint32_t start2, uint32_t end2, uint32_t chunk2,
                const std::function<void(uint32_t)> &phase2);

    private:
        ThreadPool() = default;
        ~ThreadPool();
        ThreadPool(const ThreadPool &) = delete;
        ThreadPool &operator=(const ThreadPool &) = delete;
        void workerMain(size_t workerIndex);
        void pinWorkerToCpu(size_t workerIndex);

        /// @brief Partition [start,end) into one contiguous slab per thread
        /// (workers + main). Workers get the first numWorkers_ slabs, the main
        /// thread the last. Shared by parallelForSlab and parallelForSlab2.
        void partitionSlabs(uint32_t start, uint32_t end);

        /// @brief Spin for a bounded number of iterations, then return false so
        /// the caller can fall back to blocking. Returns true if the predicate
        /// became true during the spin.
        template<typename Pred>
        bool spinWait(Pred pred) {
            constexpr int kSpinIterations = 4096;
            for (int i = 0; i < kSpinIterations; ++i) {
                if (pred()) {
                    return true;
                }
                spinPause();
            }
            return pred();
        }

        size_t numThreads_ = 1;
        bool initialized_ = false;
        bool affinityEnabled_ = false;
        std::vector<std::thread> workers_;

        std::mutex mutex_;
        std::condition_variable cv_;

        // Task data
        std::function<void(uint32_t)> task_;
        std::atomic<uint32_t> nextIndex_{0};
        uint32_t endIndex_ = 0;

        // Static-slab task data (parallelForSlab, P6): each worker owns a
        // contiguous [slabStart_, slabEnd_) range; the main thread owns
        // [mainSlabStart_, mainSlabEnd_). When slabTask_ is true, workers ignore
        // nextIndex_ and iterate their own slab, eliminating shared-atomic
        // contention on large-tile dispatches.
        bool slabTask_ = false;
        std::vector<uint32_t> slabStart_;
        std::vector<uint32_t> slabEnd_;
        uint32_t mainSlabStart_ = 0;
        uint32_t mainSlabEnd_ = 0;

        // Two-phase (parallelForSlab2) task data. The two phases share ONE task
        // publication: workers run phase1 over their slab, then (if a phase2
        // range was supplied) spin on phase2Ready_ until the MAIN thread (which
        // ran `between` on its own slab's completion) releases it, then run
        // phase2 over their (independently computed) phase-2 slab.
        uint32_t phase2Start_ = 0;
        uint32_t phase2End_ = 0;
        std::function<void(uint32_t)> phase2Func_;
        std::atomic<bool> phase2Ready_{false};
        // Phase-1 arrival counter (parallelForSlab2): each worker increments it
        // after finishing its phase-1 slab; the main thread's barrier-1 wait
        // uses it (pendingWorkers_ is only decremented after phase 2, so it
        // cannot serve as the phase-1 barrier).
        std::atomic<uint32_t> phase1Arrived_{0};
        // Set by parallelForSlab2 to mark the published task as two-phase;
        // workers consult it to decide whether to run a phase-2 slab after the
        // phase-2 flag. Reset to false when the task completes (and by the
        // plain parallelFor/parallelForSlab publishers before publishing).
        bool twoPhaseTask_ = false;
        // Phase-1 slab bounds for a two-phase task. Published BEFORE the task
        // (workers read these for phase 1 only). Phase-2 bounds use the same
        // slabStart_/slabEnd_ vectors, which the main thread repartitions after
        // `between` — the two phases must not share bounds (a worker may still
        // be computing phase 1 while the main thread repartitions for phase 2).
        std::vector<uint32_t> slabStart1_;
        std::vector<uint32_t> slabEnd1_;
        uint32_t mainSlabStart1_ = 0;
        uint32_t mainSlabEnd1_ = 0;

        // Chunked work-stealing task data (parallelForSteal / parallelForSteal2):
        // when stealTask_ is true, workers (and the main thread) pull
        // [stealChunkSize_]-sized index chunks from nextIndex_ via fetch_add
        // (llama.cpp ggml_compute_forward_mul_mat_one_chunk style) instead of
        // running a static slab or a 1-item fetch_add. Phase-1 chunk size is
        // phaseChunk1_, phase-2 chunk size phaseChunk2_ (parallelForSteal2).
        bool stealTask_ = false;
        bool twoPhaseStealTask_ = false;
        uint32_t stealChunk_ = 1;
        uint32_t phaseChunk1_ = 1;
        uint32_t phaseChunk2_ = 1;

        // Generation counter: incremented by the main thread BEFORE
        // setting up each new task. Workers snapshot the current value
        // at wakeup and wait for it to change before looping back,
        // providing a clear "old task done" signal.
        std::atomic<uint64_t> taskId_{0};

        // Number of workers still processing the current task
        std::atomic<uint32_t> pendingWorkers_{0};
        // Worker-thread count (numThreads_ - 1), stored for notify logic
        uint32_t numWorkers_ = 0;

        // Re-entrancy guard
        std::atomic<uint32_t> reentrantDepth_{0};

        // Wake / stop signals
        std::atomic<bool> hasWork_{false};
        std::atomic<bool> stop_{false};

        // Number of workers currently blocked on the condition variable
        // (either waiting for work or waiting for the ack). The main thread
        // checks this to decide whether a cv_.notify_all() is required.
        std::atomic<uint32_t> blockedWorkers_{0};
    };

}// namespace tinycoder
