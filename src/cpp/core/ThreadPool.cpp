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

#include "ThreadPool.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace tinycoder {

    ThreadPool &ThreadPool::instance() {
        static ThreadPool pool;
        return pool;
    }

    size_t ThreadPool::recommendedThreadCount() {
        const char *env = std::getenv("TINYCODER_THREADS");
        if (env != nullptr && env[0] != '\0') {
            long v = std::strtol(env, nullptr, 10);
            if (v > 0 && v <= 1024) {
                return static_cast<size_t>(v);
            }
        }

        // LOGICAL-CPU count (sysconf(_SC_NPROCESSORS_ONLN) on Linux,
        // std::thread::hardware_concurrency elsewhere) — the measured optimum.
        // Measured 2026-08-28 AFTER the task-graph levers (dynamic chunk
        // work-stealing + last-arriver cooperative act quantize): 8 threads
        // holds a clear edge over llama.cpp's 4-physical default on the
        // reference host (i7-4790K, 4 physical / 8 logical): 24.06/24.01 (8t)
        // vs 21.68/21.60 (4t) tok/s with coop-ON, i.e. 8-way parallelism keeps
        // DRAM saturated even at the physical-core pool's ~equal efficiency —
        // the fine 8-way row split plus the pooled HT lanes keep the memory
        // bus busier than 4 dedicated cores on this 2-channel DDR3-1600.
        // physicalThreadCount() remains available for A/B'ing llama.cpp's
        // default; set TINYCODER_THREADS to override at runtime.
#if defined(__linux__)
        long nCpus = sysconf(_SC_NPROCESSORS_ONLN);
        if (nCpus > 0) {
            return static_cast<size_t>(nCpus);
        }
#endif
        unsigned int hw = std::thread::hardware_concurrency();
        if (hw > 0) {
            return static_cast<size_t>(hw);
        }
        return 1;
    }

    size_t ThreadPool::physicalThreadCount() {
#if defined(__linux__)
        // llama.cpp's common_cpu_get_num_physical_cores() (Linux path): read
        // /sys/devices/system/cpu/cpuN/topology/thread_siblings for each online
        // CPU and count DISTINCT sibling masks. Two logical CPUs sharing an HT
        // lane report identical masks, so the distinct-mask count equals the
        // number of physical cores (4 on the i7-4790K reference: masks 1,2,4,8
        // — source-verified in ~/git/llama.cpp/common/common.cpp).
        {
            unsigned long long masks[64];
            uint32_t n = 0;
            long nCpus = sysconf(_SC_NPROCESSORS_ONLN);
            for (long cpu = 0; cpu < nCpus && n < 64; ++cpu) {
                char path[256];
                std::snprintf(path, sizeof(path),
                              "/sys/devices/system/cpu/cpu%ld/topology/thread_siblings",
                              cpu);
                std::ifstream f(path);
                if (!f.is_open()) {
                    continue;
                }
                unsigned long long mask = 0;
                f >> std::hex >> mask;
                if (mask == 0) {
                    continue;
                }
                bool dup = false;
                for (uint32_t i = 0; i < n; ++i) {
                    if (masks[i] == mask) {
                        dup = true;
                        break;
                    }
                }
                if (!dup) {
                    masks[n++] = mask;
                }
            }
            if (n > 0) {
                return static_cast<size_t>(n);
            }
        }
#endif
        // Fallback (mirrors llama.cpp's Windows cpu_count_math_cpus()): half
        // the logical count — "hyperthreading isn't useful for linear algebra".
        unsigned int hw = std::thread::hardware_concurrency();
        if (hw > 1) {
            return static_cast<size_t>(hw / 2);
        }
        return 1;
    }

    void ThreadPool::initialize(size_t numThreads) {
        if (initialized_) {
            return;
        }

        if (numThreads < 1) {
            numThreads = 1;
        }

        numThreads_ = numThreads;
        numWorkers_ = static_cast<uint32_t>(numThreads - 1);

        // Default from the CMake option (-DTINYCODER_AFFINITY=0/1): OFF (0,
        // default) lets the OS schedule workers across all logical CPUs, which
        // is faster at low thread counts on memory-bound generation (matches
        // llama.cpp's 4-thread behaviour). The TINYCODER_AFFINITY environment
        // variable ("1"/"0") overrides the baked-in default at runtime so one
        // binary can A/B the pinning without a recompile. A later
        // setAffinityEnabled() call (e.g. a caller forcing a specific policy)
        // still overrides both.
        bool affinityDefault = false;
#if defined(TINYCODER_AFFINITY)
        affinityDefault = (TINYCODER_AFFINITY != 0);
#endif
        const char *envAf = std::getenv("TINYCODER_AFFINITY");
        if (envAf != nullptr && envAf[0] != '\0') {
            affinityDefault = (envAf[0] == '1' || envAf[0] == 't' || envAf[0] == 'T');
        }
        affinityEnabled_ = affinityDefault;

        workers_.reserve(numWorkers_);
        for (size_t i = 0; i < numWorkers_; ++i) {
            workers_.emplace_back(&ThreadPool::workerMain, this, i);
        }

        // Pin the main thread (which participates in every parallel dispatch)
        // to CPU 0 so it never shares an HT lane with a worker. pinWorkerToCpu
        // is used here with a sentinel index -1 -> workerIndex+1 wraps to CPU0.
#if defined(__linux__) && defined(TINYCODER_AFFINITY)
        if (affinityEnabled_) {
            long nCpus = sysconf(_SC_NPROCESSORS_ONLN);
            if (nCpus > 0) {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(0, &set);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
            }
        }
#endif

        initialized_ = true;
    }

    ThreadPool::~ThreadPool() {
        if (initialized_) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_ = true;
                hasWork_ = true;// Wake up workers so they see stop_
            }
            cv_.notify_all();

            for (auto &worker: workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }
    }

    void ThreadPool::pinWorkerToCpu(size_t workerIndex) {
#if defined(__linux__)
        if (!affinityEnabled_) {
            return;
        }

        long nCpus = sysconf(_SC_NPROCESSORS_ONLN);
        if (nCpus <= 0) {
            return;
        }
        size_t nLogical = static_cast<size_t>(nCpus);

        // Pin each worker to a DISTINCT logical CPU so no two workers contend
        // for the same HT lane. The main thread participates in every parallel
        // dispatch too, so CPU 0 is reserved for it (see initialize(), which
        // pins the main thread to CPU 0); workers occupy the remaining CPUs
        // 1..numThreads_-1. On the reference host (i7-4790K, 4 physical / 8
        // logical) this gives the full 0..7 coverage:
        //   main -> CPU0, worker0 -> CPU1, ..., worker6 -> CPU7,
        // keeping every lane busy and NOT pairing a worker with its HT sibling
        // on a memory-bound kernel (which would thrash the shared L3). Earlier
        // pinning mapped workers modulo the physical-core count (0-3), which
        // crammed the 7 workers + main onto 4 cores — that is what AB/measured
        // SLOWER than the OS default and why OFF is the default. This distinct
        // logical-CPU distribution is the correct pin for the 8-thread model.
        int cpu = static_cast<int>((workerIndex + 1) % nLogical);

        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
#endif
    }

    void ThreadPool::workerMain(size_t workerIndex) {
        pinWorkerToCpu(workerIndex);

        // The generation of the last task this worker processed. Starts at 0,
        // matching the initial taskId_ value, so the first published task
        // (taskId_ == 1) is immediately recognized as new work.
        uint64_t myGen = 0;

        while (true) {
            // ---- Wait for new work ----
            // The generation counter is the single synchronization point: a
            // worker waits for taskId_ to advance past myGen, which signals both
            // "the previous task is acknowledged" and "new work is available".
            // Spin briefly on the lock-free signal, then fall back to blocking
            // on the condition variable to avoid burning CPU while idle (e.g.
            // between tokens). The predicate is re-checked inside cv_.wait, so
            // no wakeup is ever lost.
            bool haveWork = spinWait([&]() {
                return stop_.load(std::memory_order_acquire) ||
                       taskId_.load(std::memory_order_acquire) != myGen;
            });
            if (!haveWork) {
                std::unique_lock<std::mutex> lock(mutex_);
                blockedWorkers_.fetch_add(1, std::memory_order_acq_rel);
                cv_.wait(lock, [&]() {
                    return stop_.load(std::memory_order_acquire) ||
                           taskId_.load(std::memory_order_acquire) != myGen;
                });
                blockedWorkers_.fetch_sub(1, std::memory_order_acq_rel);
            }

            // Check stop_ on BOTH paths. spinWait may have returned true because
            // stop_ was set (not because new work arrived), in which case we must
            // exit without touching the (possibly stale) task data.
            if (stop_.load(std::memory_order_acquire)) {
                return;
            }

            // Snapshot the generation for this task. The main thread publishes
            // the task data (task_, nextIndex_, endIndex_) with release
            // semantics before incrementing taskId_ (acq_rel), so observing
            // taskId_ != myGen guarantees the task data is visible.
            myGen = taskId_.load(std::memory_order_acquire);

            // ---- Do the work ----
            if (slabTask_) {
                // Static contiguous slab (P6 / P7 two-phase): iterate our own
                // [start,end), no shared-atomic contention. For a two-phase
                // task the phase-1 bounds are published separately (slabStart1_)
                // so the main thread's phase-2 repartition of slabStart_ cannot
                // race with a worker still running phase 1.
                uint32_t s = twoPhaseTask_ ? slabStart1_[workerIndex] : slabStart_[workerIndex];
                uint32_t e = twoPhaseTask_ ? slabEnd1_[workerIndex] : slabEnd_[workerIndex];
                for (uint32_t idx = s; idx < e; ++idx) {
                    task_(idx);
                }

                // ---- Two-phase task (P7 fused FFN): run phase 2 ----
                // If the published task carries a phase-2 range, wait for the
                // main thread to publish `between`'s result (phase2Ready_,
                // release), then run our phase-2 slab. This keeps phase 1 + 2
                // inside ONE task publication, eliminating the second
                // cv_/mutex publication-barrier per layer.
                if (twoPhaseTask_) {
                    // Signal phase-1 completion (release). The main thread's
                    // barrier-1 wait spins on phase1Arrived_ == numWorkers_
                    // (pendingWorkers_ is only decremented after phase 2).
                    phase1Arrived_.fetch_add(1, std::memory_order_release);

                    // Spin on the phase-2 release flag (workers that finished
                    // phase 1 ahead of the main thread wait here). This is an
                    // INTRA-KERNEL barrier (bounded, balanced phases), so a
                    // pure spin — never a cv wait — avoids the lost-wakeup
                    // deadlock where nothing notifies workers blocked on
                    // phase2Ready_ (the main thread releases the flag without
                    // holding the mutex). This is the "context-switch-free"
                    // barrier llama.cpp's task graph relies on.
                    while (!phase2Ready_.load(std::memory_order_acquire)) {
                        spinPause();
                    }
                    // Run our phase-2 slab (recomputed by main thread before
                    // releasing the flag; workers read the new bounds).
                    uint32_t s2 = slabStart_[workerIndex];
                    uint32_t e2 = slabEnd_[workerIndex];
                    for (uint32_t idx = s2; idx < e2; ++idx) {
                        phase2Func_(idx);
                    }
                }
            } else if (stealTask_) {
                // Dynamic chunked work-stealing (parallelForSteal /
                // parallelForSteal2): pull [chunk]-sized index chunks from the
                // shared atomic via fetch_add — llama.cpp's
                // ggml_compute_forward_mul_mat_one_chunk dispatch. This keeps
                // DRAM saturated at the tail on memory-bound generation kernels
                // (no thread idles on a long static slab).
                uint32_t chunk = stealChunk_;
                while (true) {
                    uint32_t idx = nextIndex_.fetch_add(chunk, std::memory_order_relaxed);
                    if (idx >= endIndex_) {
                        break;
                    }
                    uint32_t end = std::min(idx + chunk, endIndex_);
                    for (uint32_t t = idx; t < end; ++t) {
                        task_(t);
                    }
                }

                // ---- Two-phase steal task (fused FFN, steal schedule): ----
                if (twoPhaseStealTask_) {
                    phase1Arrived_.fetch_add(1, std::memory_order_release);
                    while (!phase2Ready_.load(std::memory_order_acquire)) {
                        spinPause();
                    }
                    uint32_t chunk2 = phaseChunk2_;
                    while (true) {
                        uint32_t idx = nextIndex_.fetch_add(chunk2, std::memory_order_relaxed);
                        if (idx >= phase2End_) {
                            break;
                        }
                        uint32_t end = std::min(idx + chunk2, phase2End_);
                        for (uint32_t t = idx; t < end; ++t) {
                            phase2Func_(t);
                        }
                    }
                }
            } else {
                while (true) {
                    uint32_t idx = nextIndex_.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= endIndex_) {
                        break;
                    }
                    task_(idx);
                }
            }

            // ---- Signal completion ----
            uint32_t prev = pendingWorkers_.fetch_sub(1, std::memory_order_acq_rel);
            if (prev == 1) {
                // Was the last worker — notify the main thread (which may be
                // blocked waiting for pendingWorkers_ == 0). The mutex must be
                // held so the notify cannot be lost between the main thread's
                // predicate check and its sleep (classic lost-wakeup race).
                std::lock_guard<std::mutex> lock(mutex_);
                cv_.notify_all();
            }

            // Loop back to wait for the next task (taskId_ != myGen). Since
            // myGen now equals the current taskId_, the worker will not
            // re-process this task; it waits for the generation to advance.
        }
    }

    void ThreadPool::parallelFor(uint32_t start, uint32_t end,
                                 const std::function<void(uint32_t)> &func) {
        if (!initialized_ || numThreads_ <= 1 || start >= end) {
            for (uint32_t i = start; i < end; ++i) {
                func(i);
            }
            return;
        }

        if (end - start <= numThreads_) {
            for (uint32_t i = start; i < end; ++i) {
                func(i);
            }
            return;
        }

        // Re-entrancy guard
        if (reentrantDepth_.load(std::memory_order_acquire) > 0) {
            for (uint32_t i = start; i < end; ++i) {
                func(i);
            }
            return;
        }

        reentrantDepth_.store(1, std::memory_order_release);

        // ---- Publish the new task ----
        // Write the task data FIRST (plain stores), then publish it with
        // release semantics. Workers that observe taskId_ != myGen (acquire)
        // are guaranteed to see the fully-written task data. Incrementing
        // taskId_ releases the previous task's workers from their wait.
        task_ = func;
        slabTask_ = false;
        twoPhaseTask_ = false;// plain (index-atomic) task: no phase 2
        stealTask_ = false;   // plain (index-atomic) task: no chunked steal
        twoPhaseStealTask_ = false;
        nextIndex_.store(start, std::memory_order_relaxed);
        endIndex_ = end;
        pendingWorkers_.store(numWorkers_, std::memory_order_relaxed);

        // Publish the generation and wake blocked workers while holding the
        // mutex. This is REQUIRED to avoid a lost wakeup: a worker may have
        // evaluated its wait predicate (false) and be about to sleep when we
        // notify. If the notify happened without the mutex, it could land
        // between the worker's predicate check and its sleep, leaving the
        // worker blocked forever even though its predicate is now true.
        // Holding the mutex makes the predicate-check-and-sleep atomic with
        // respect to the notify. In the common fast path (workers spinning),
        // the mutex is uncontended, so this is cheap.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            taskId_.fetch_add(1, std::memory_order_acq_rel);
            hasWork_.store(true, std::memory_order_release);
            cv_.notify_all();
        }

        // ---- Main thread participates ----
        while (true) {
            uint32_t idx = nextIndex_.fetch_add(1, std::memory_order_relaxed);
            if (idx >= end) {
                break;
            }
            func(idx);
        }

        // ---- Wait for all workers to finish ----
        // Spin briefly, then block on the condition variable (the last worker
        // notifies via cv_.notify_all() while holding the mutex).
        bool done = spinWait([&]() {
            return pendingWorkers_.load(std::memory_order_acquire) == 0;
        });
        if (!done) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&]() {
                return pendingWorkers_.load(std::memory_order_acquire) == 0;
            });
        }

        // Clear hasWork_ HERE, after ALL workers from this task have finished.
        // This is only used by the destructor's wake-up logic; the worker wait
        // predicate relies solely on taskId_, so clearing hasWork_ here is safe.
        hasWork_.store(false, std::memory_order_release);

        reentrantDepth_.store(0, std::memory_order_release);
    }

    void ThreadPool::parallelForSlab(uint32_t start, uint32_t end,
                                     const std::function<void(uint32_t)> &func) {
        if (!initialized_ || numThreads_ <= 1 || start >= end) {
            for (uint32_t i = start; i < end; ++i) {
                func(i);
            }
            return;
        }

        if (end - start <= numThreads_) {
            for (uint32_t i = start; i < end; ++i) {
                func(i);
            }
            return;
        }

        // Re-entrancy guard (matches parallelFor)
        if (reentrantDepth_.load(std::memory_order_acquire) > 0) {
            for (uint32_t i = start; i < end; ++i) {
                func(i);
            }
            return;
        }

        reentrantDepth_.store(1, std::memory_order_release);

        // Partition [start,end) into numThreads contiguous slabs (workers get
        // the first numWorkers_ slabs, the main thread the last). Balanced by
        // giving the first `rem` threads one extra item.
        uint32_t total = end - start;
        uint32_t nT = static_cast<uint32_t>(numThreads_);
        uint32_t base = total / nT;
        uint32_t rem = total % nT;

        slabStart_.resize(numWorkers_);
        slabEnd_.resize(numWorkers_);
        uint32_t cur = start;
        for (uint32_t t = 0; t < nT; ++t) {
            uint32_t size = base + (t < rem ? 1u : 0u);
            uint32_t s = cur;
            uint32_t e = cur + size;
            cur = e;
            if (t < numWorkers_) {
                slabStart_[t] = s;
                slabEnd_[t] = e;
            } else {
                mainSlabStart_ = s;
                mainSlabEnd_ = e;
            }
        }

        // ---- Publish the new task ----
        task_ = func;
        slabTask_ = true;
        twoPhaseTask_ = false;// plain slab task: no phase 2
        stealTask_ = false;
        twoPhaseStealTask_ = false;
        nextIndex_.store(start, std::memory_order_relaxed);
        endIndex_ = end;
        pendingWorkers_.store(numWorkers_, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            taskId_.fetch_add(1, std::memory_order_acq_rel);
            hasWork_.store(true, std::memory_order_release);
            cv_.notify_all();
        }

        // ---- Main thread runs its own slab ----
        for (uint32_t idx = mainSlabStart_; idx < mainSlabEnd_; ++idx) {
            func(idx);
        }

        // ---- Wait for all workers to finish ----
        bool done = spinWait([&]() {
            return pendingWorkers_.load(std::memory_order_acquire) == 0;
        });
        if (!done) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&]() {
                return pendingWorkers_.load(std::memory_order_acquire) == 0;
            });
        }

        hasWork_.store(false, std::memory_order_release);

        // Clear the slab flag so subsequent parallelFor calls don't misinterpret
        // it as a slab task (workerMain reads slabTask_ on every task).
        slabTask_ = false;

        reentrantDepth_.store(0, std::memory_order_release);
    }

    // Partition [start,end) into nT contiguous slabs; worker t owns
    // [slabStart_[t], slabEnd_[t]), the main thread [mainSlabStart_, mainSlabEnd_).
    void ThreadPool::partitionSlabs(uint32_t start, uint32_t end) {
        uint32_t total = end - start;
        uint32_t nT = static_cast<uint32_t>(numThreads_);
        uint32_t base = total / nT;
        uint32_t rem = total % nT;

        slabStart_.resize(numWorkers_);
        slabEnd_.resize(numWorkers_);
        uint32_t cur = start;
        for (uint32_t t = 0; t < nT; ++t) {
            uint32_t size = base + (t < rem ? 1u : 0u);
            uint32_t s = cur;
            uint32_t e = cur + size;
            cur = e;
            if (t < numWorkers_) {
                slabStart_[t] = s;
                slabEnd_[t] = e;
            } else {
                mainSlabStart_ = s;
                mainSlabEnd_ = e;
            }
        }
    }

    // Single-launch two-phase parallel-for with static slabs.
    //
    // Publishes ONE task covering phase1 + (between) + phase2. The main thread:
    //   - publishes the task, then runs its own phase-1 slab
    //   - waits for all workers to finish phase 1 (spin + cv fallback)
    //   - runs `between`, then releases phase2Ready_ (release), then runs its
    //     own phase-2 slab
    //   - waits for all workers to finish phase 2
    // Workers (see workerMain): run phase-1 slab, then spin on phase2Ready_,
    // then run phase-2 slab (if a phase-2 range was given), then decrement
    // pendingWorkers_ — so the second cv barrier is genuinely eliminated and
    // the task cost is one publication + one barrier instead of two of each.
    void ThreadPool::parallelForSlab2(
            uint32_t start1, uint32_t end1,
            const std::function<void(uint32_t)> &phase1,
            const std::function<void()> &between,
            uint32_t start2, uint32_t end2,
            const std::function<void(uint32_t)> &phase2) {
        bool hasPhase2 = (phase2 && start2 < end2);
        if (!initialized_ || numThreads_ <= 1 || start1 >= end1) {
            for (uint32_t i = start1; i < end1; ++i) {
                phase1(i);
            }
            if (between) {
                between();
            }
            if (hasPhase2) {
                for (uint32_t i = start2; i < end2; ++i) {
                    phase2(i);
                }
            }
            return;
        }

        // Re-entrancy guard (matches parallelFor/parallelForSlab): a nested
        // pool call from inside this task runs serially.
        if (reentrantDepth_.load(std::memory_order_acquire) > 0) {
            for (uint32_t i = start1; i < end1; ++i) {
                phase1(i);
            }
            if (between) {
                between();
            }
            if (hasPhase2) {
                for (uint32_t i = start2; i < end2; ++i) {
                    phase2(i);
                }
            }
            return;
        }
        reentrantDepth_.store(1, std::memory_order_release);

        // Partition phase 1 into the DEDICATED phase-1 arrays (workers get the
        // first numWorkers_ slabs, the main thread the last). Phase-2 bounds
        // will later reuse the shared slabStart_/slabEnd_ (partitioned by the
        // main thread after `between`, which cannot race phase-1 reads).
        {
            uint32_t total = end1 - start1;
            uint32_t nT = static_cast<uint32_t>(numThreads_);
            uint32_t base = total / nT;
            uint32_t rem = total % nT;
            slabStart1_.resize(numWorkers_);
            slabEnd1_.resize(numWorkers_);
            uint32_t cur = start1;
            for (uint32_t t = 0; t < nT; ++t) {
                uint32_t size = base + (t < rem ? 1u : 0u);
                uint32_t s = cur;
                uint32_t e = cur + size;
                cur = e;
                if (t < numWorkers_) {
                    slabStart1_[t] = s;
                    slabEnd1_[t] = e;
                } else {
                    mainSlabStart1_ = s;
                    mainSlabEnd1_ = e;
                }
            }
        }

        // ---- Publish the single task ----
        task_ = phase1;
        slabTask_ = true;
        twoPhaseTask_ = hasPhase2;
        stealTask_ = false;
        twoPhaseStealTask_ = false;
        phase2Start_ = start2;
        phase2End_ = end2;
        phase2Func_ = hasPhase2 ? phase2 : std::function<void(uint32_t)>();
        phase2Ready_.store(false, std::memory_order_relaxed);
        phase1Arrived_.store(0, std::memory_order_relaxed);
        pendingWorkers_.store(numWorkers_, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            taskId_.fetch_add(1, std::memory_order_acq_rel);
            hasWork_.store(true, std::memory_order_release);
            cv_.notify_all();
        }

        // ---- Main thread runs its own phase-1 slab ----
        for (uint32_t idx = mainSlabStart1_; idx < mainSlabEnd1_; ++idx) {
            phase1(idx);
        }

        // ---- Barrier 1: all workers finished phase 1 ----
        // Workers signal by incrementing phase1Arrived_ (their pendingWorkers_
        // decrement happens only after phase 2). This is an INTRA-KERNEL
        // barrier: the phase-1 slabs are bounded and roughly balanced, so the
        // main thread merely spins until all workers arrive. A cv fallback
        // would deadlock here — nothing notifies the cv when phase1Arrived_ is
        // incremented. Spinning is the "context-switch-free" barrier llama.cpp
        // relies on.
        while (phase1Arrived_.load(std::memory_order_acquire) < numWorkers_) {
            spinPause();
        }

        // ---- Between: serialized main-thread work, then release the flag ----
        if (between) {
            between();
        }
        // Partition phase 2 BEFORE releasing the flag: workers read
        // slabStart_/slabEnd_ after their acquire load of phase2Ready_, so the
        // partition must happen-before the release (otherwise the worker reads
        // race the main thread's writes / resize).
        if (hasPhase2) {
            partitionSlabs(start2, end2);
        }
        // Release the phase-2 flag (release orders between()'s and
        // partitionSlabs()'s writes before the workers' acquire loads).
        phase2Ready_.store(true, std::memory_order_release);

        // ---- Main thread runs its own phase-2 slab ----
        if (hasPhase2) {
            for (uint32_t idx = mainSlabStart_; idx < mainSlabEnd_; ++idx) {
                phase2(idx);
            }
        }

        // ---- Barrier 2: all workers finished phase 2 ----
        // Each worker decrements pendingWorkers_ exactly once (after its phase-2
        // slab); spin until all arrive. Bounded, balanced work ⇒ safe to spin.
        while (pendingWorkers_.load(std::memory_order_acquire) != 0) {
            spinPause();
        }

        hasWork_.store(false, std::memory_order_release);
        slabTask_ = false;
        // Clear the two-phase task data so a subsequent plain parallelForSlab /
        // parallelFor does not observe a stale phase2Ready_ / phase2Func_.
        twoPhaseTask_ = false;
        phase2Ready_.store(false, std::memory_order_relaxed);
        phase2Func_ = std::function<void(uint32_t)>();
        phase1Arrived_.store(0, std::memory_order_relaxed);

        reentrantDepth_.store(0, std::memory_order_release);
    }

    // Dynamic chunked work-stealing parallel-for (llama.cpp
    // ggml_compute_forward_mul_mat_one_chunk style). Each participant (workers +
    // main thread) pulls [chunkSize]-sized index ranges from the shared atomic
    // nextIndex_ via fetch_add(chunkSize), so the tail of the index space is
    // stolen by whichever thread finishes its share first — no static slab can
    // leave a thread idle while others still work. This keeps DRAM saturated
    // on memory-bound generation kernels at BOTH thread counts (4 and 8).
    //
    // chunkSize is in ITEMS (e.g. 4 tiles); the main thread and each worker
    // run the same fetch_add loop, and the final partial chunk (idx+chunk might
    // overshoot end) is clamped with min().
    void ThreadPool::parallelForSteal(uint32_t start, uint32_t end,
                                      uint32_t chunkSize,
                                      const std::function<void(uint32_t)> &func) {
        if (chunkSize == 0) {
            chunkSize = 1;
        }
        if (!initialized_ || numThreads_ <= 1 || start >= end) {
            for (uint32_t i = start; i < end; ++i) {
                func(i);
            }
            return;
        }

        if (end - start <= numThreads_) {
            for (uint32_t i = start; i < end; ++i) {
                func(i);
            }
            return;
        }

        // Re-entrancy guard (matches parallelFor/parallelForSlab)
        if (reentrantDepth_.load(std::memory_order_acquire) > 0) {
            for (uint32_t i = start; i < end; ++i) {
                func(i);
            }
            return;
        }
        reentrantDepth_.store(1, std::memory_order_release);

        // ---- Publish the new task ----
        task_ = func;
        slabTask_ = false;
        twoPhaseTask_ = false;
        stealTask_ = true;
        twoPhaseStealTask_ = false;
        stealChunk_ = chunkSize;
        nextIndex_.store(start, std::memory_order_relaxed);
        endIndex_ = end;
        pendingWorkers_.store(numWorkers_, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            taskId_.fetch_add(1, std::memory_order_acq_rel);
            hasWork_.store(true, std::memory_order_release);
            cv_.notify_all();
        }

        // ---- Main thread participates (same fetch_add loop as workers) ----
        while (true) {
            uint32_t idx = nextIndex_.fetch_add(chunkSize, std::memory_order_relaxed);
            if (idx >= end) {
                break;
            }
            uint32_t last = std::min(idx + chunkSize, end);
            for (uint32_t t = idx; t < last; ++t) {
                func(t);
            }
        }

        // ---- Wait for all workers to finish ----
        bool done = spinWait([&]() {
            return pendingWorkers_.load(std::memory_order_acquire) == 0;
        });
        if (!done) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&]() {
                return pendingWorkers_.load(std::memory_order_acquire) == 0;
            });
        }

        hasWork_.store(false, std::memory_order_release);

        // Clear the steal flag so subsequent plain parallelFor / slab calls
        // don't misinterpret the published task.
        stealTask_ = false;
        twoPhaseStealTask_ = false;

        reentrantDepth_.store(0, std::memory_order_release);
    }

    // Single-launch two-phase parallel-for with chunked work-stealing.
    //
    // Mirrors parallelForSlab2's two-phase protocol (ONE task publication for
    // phase1 + `between` + phase2, spin-flag phase transition) but each phase
    // pulls [chunk1/chunk2]-sized chunks from the shared atomic instead of
    // running static slabs. This is the steal variant the fused gate+up+down
    // kernel uses as lever 1: llama.cpp's single-token mul_mat runs
    // chunk_size=64 dynamic work-stealing over the nr0×nr1 row space, which is
    // where its 4-thread efficiency (29.4 vs TinyCoder's 19.2 tok/s) comes
    // from.
    //
    // The protocol: workers run phase-1 chunks (task_), signal arrival via
    // phase1Arrived_, spin on phase2Ready_, then run phase-2 chunks
    // (phase2Func_) pulling from the shared atomic with phaseChunk2_ granularity
    // against phase2End_. The main thread runs its own phase-1 chunks, waits
    // for all workers via phase1Arrived_, runs `between`, publishes phase-2
    // bounds (nextIndex_ = start2, endIndex_ = end2), releases phase2Ready_,
    // then runs its own phase-2 chunks directly (same fetch_add loop).
    void ThreadPool::parallelForSteal2(
            uint32_t start1, uint32_t end1, uint32_t chunk1,
            const std::function<void(uint32_t)> &phase1,
            const std::function<void()> &between,
            uint32_t start2, uint32_t end2, uint32_t chunk2,
            const std::function<void(uint32_t)> &phase2) {
        bool hasPhase2 = (phase2 && start2 < end2);
        if (chunk1 == 0) {
            chunk1 = 1;
        }
        if (chunk2 == 0) {
            chunk2 = 1;
        }
        if (!initialized_ || numThreads_ <= 1 || start1 >= end1) {
            for (uint32_t i = start1; i < end1; ++i) {
                phase1(i);
            }
            if (between) {
                between();
            }
            if (hasPhase2) {
                for (uint32_t i = start2; i < end2; ++i) {
                    phase2(i);
                }
            }
            return;
        }

        // Re-entrancy guard
        if (reentrantDepth_.load(std::memory_order_acquire) > 0) {
            for (uint32_t i = start1; i < end1; ++i) {
                phase1(i);
            }
            if (between) {
                between();
            }
            if (hasPhase2) {
                for (uint32_t i = start2; i < end2; ++i) {
                    phase2(i);
                }
            }
            return;
        }
        reentrantDepth_.store(1, std::memory_order_release);

        // ---- Publish the single steal task ----
        task_ = phase1;
        slabTask_ = false;
        twoPhaseTask_ = false;
        stealTask_ = true;
        twoPhaseStealTask_ = hasPhase2;
        stealChunk_ = chunk1;
        phaseChunk1_ = chunk1;
        phaseChunk2_ = chunk2;
        phase2Start_ = start2;
        phase2End_ = end2;
        // Phase 1 pulls from nextIndex_/endIndex_; the main thread repoints
        // them to phase 2 (start2/end2) before releasing phase2Ready_.
        nextIndex_.store(start1, std::memory_order_relaxed);
        endIndex_ = end1;
        phase2Func_ = hasPhase2 ? phase2 : std::function<void(uint32_t)>();
        phase2Ready_.store(false, std::memory_order_relaxed);
        phase1Arrived_.store(0, std::memory_order_relaxed);
        pendingWorkers_.store(numWorkers_, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            taskId_.fetch_add(1, std::memory_order_acq_rel);
            hasWork_.store(true, std::memory_order_release);
            cv_.notify_all();
        }

        // ---- Main thread runs its own phase-1 chunks (fetch_add loop) ----
        while (true) {
            uint32_t idx = nextIndex_.fetch_add(chunk1, std::memory_order_relaxed);
            if (idx >= end1) {
                break;
            }
            uint32_t last = std::min(idx + chunk1, end1);
            for (uint32_t t = idx; t < last; ++t) {
                phase1(t);
            }
        }

        // ---- Barrier 1: all workers finished phase 1 ----
        while (phase1Arrived_.load(std::memory_order_acquire) < numWorkers_) {
            spinPause();
        }

        // ---- Between: serialized main-thread work ----
        if (between) {
            between();
        }
        // Publish the phase-2 index range BEFORE releasing the flag: workers
        // read nextIndex_/endIndex_ (as phase-2 bounds) after their acquire
        // load of phase2Ready_, so the stores must happen-before the release.
        if (hasPhase2) {
            nextIndex_.store(start2, std::memory_order_relaxed);
            endIndex_ = end2;
        }
        phase2Ready_.store(true, std::memory_order_release);

        // ---- Main thread runs its own phase-2 chunks ----
        if (hasPhase2) {
            while (true) {
                uint32_t idx = nextIndex_.fetch_add(chunk2, std::memory_order_relaxed);
                if (idx >= end2) {
                    break;
                }
                uint32_t last = std::min(idx + chunk2, end2);
                for (uint32_t t = idx; t < last; ++t) {
                    phase2(t);
                }
            }
        }

        // ---- Barrier 2: all workers finished phase 2 ----
        while (pendingWorkers_.load(std::memory_order_acquire) != 0) {
            spinPause();
        }

        hasWork_.store(false, std::memory_order_release);
        slabTask_ = false;
        stealTask_ = false;
        twoPhaseStealTask_ = false;
        twoPhaseTask_ = false;
        phase2Ready_.store(false, std::memory_order_relaxed);
        phase2Func_ = std::function<void(uint32_t)>();
        phase1Arrived_.store(0, std::memory_order_relaxed);

        reentrantDepth_.store(0, std::memory_order_release);
    }

    void ThreadPool::parallelFor2D(uint32_t dim1, uint32_t dim2,
                                   const std::function<void(uint32_t, uint32_t)> &func) {
        uint64_t total = static_cast<uint64_t>(dim1) * dim2;
        if (total == 0) {
            return;
        }

        if (reentrantDepth_.load(std::memory_order_acquire) > 0) {
            for (uint32_t flatIdx = 0; flatIdx < static_cast<uint32_t>(total); ++flatIdx) {
                uint32_t i = flatIdx / dim2;
                uint32_t j = flatIdx % dim2;
                func(i, j);
            }
            return;
        }

        parallelFor(0, static_cast<uint32_t>(total),
                    [dim1, dim2, &func](uint32_t flatIdx) {
                        uint32_t i = flatIdx / dim2;
                        uint32_t j = flatIdx % dim2;
                        func(i, j);
                    });
    }

}// namespace tinycoder
