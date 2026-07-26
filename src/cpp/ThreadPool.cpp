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

namespace tinycoder {

    ThreadPool &ThreadPool::instance() {
        static ThreadPool pool;
        return pool;
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

        workers_.reserve(numWorkers_);
        for (size_t i = 0; i < numWorkers_; ++i) {
            workers_.emplace_back(&ThreadPool::workerMain, this);
        }

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

    void ThreadPool::workerMain() {
        while (true) {
            uint64_t myGen;

            // ---- Phase 1: Wait for new work ----
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return hasWork_.load() || stop_.load(); });

                if (stop_) {
                    return;
                }
            }

            // ---- Phase 2: Snapshot the current generation ----
            // Released from hasWork_ wait. Snapshot taskId_ so we can detect
            // when the main thread marks this task as complete (by incrementing
            // taskId_ in the next parallelFor call).
            myGen = taskId_.load(std::memory_order_acquire);

            // ---- Phase 3: Do the work ----
            while (true) {
                uint32_t idx = nextIndex_.fetch_add(1, std::memory_order_relaxed);
                if (idx >= endIndex_) {
                    break;
                }
                task_(idx);
            }

            // ---- Phase 4: Signal completion ----
            {
                std::lock_guard<std::mutex> lock(mutex_);
                uint32_t prev = pendingWorkers_.fetch_sub(1, std::memory_order_acq_rel);
                if (prev == 1) {
                    // Was the last worker — notify the main thread
                    // (waiting for pendingWorkers_ == 0).
                    cv_.notify_all();
                }
            }

            // ---- Phase 5: Wait for task completion acknowledgment ----
            // Wait here until the MAIN THREAD (in the next parallelFor call)
            // increments taskId_. This ensures two things:
            //   1. We don't loop back and re-acquire work from the same task
            //      (which would cause pendingWorkers_ double-decrement).
            //   2. We don't race with the main thread recycling hasWork_
            //      because we only check hasWork_ again AFTER taskId_ changes.
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this, myGen]() {
                    return taskId_.load(std::memory_order_acquire) != myGen || stop_.load();
                });
            }
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

        // ---- Phase 1: Increment generation to release previous task's workers ----
        // This signals to ALL workers from the previous parallelFor call that
        // the old task is complete and it is safe to loop back.
        taskId_.fetch_add(1, std::memory_order_acq_rel);

        // ---- Phase 2: Set up the new task under the mutex ----
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Clear hasWork_ first — workers still in completion wait from the
            // previous task need the taskId_ change (already done above) to
            // exit that wait. Once they exit, they loop back to wait for
            // hasWork_, which we immediately set to true below.
            // The mutex prevents workers from checking hasWork_ between the
            // false and true assignments.
            hasWork_ = false;
            // Now set up the new task
            task_ = func;
            nextIndex_.store(start, std::memory_order_relaxed);
            endIndex_ = end;
            pendingWorkers_.store(numWorkers_, std::memory_order_relaxed);
            // Signal new work available
            hasWork_ = true;
        }
        cv_.notify_all();

        // ---- Phase 3: Main thread participates ----
        while (true) {
            uint32_t idx = nextIndex_.fetch_add(1, std::memory_order_relaxed);
            if (idx >= end) {
                break;
            }
            func(idx);
        }

        // ---- Phase 4: Wait for all workers to finish ----
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() {
                return pendingWorkers_.load(std::memory_order_acquire) == 0;
            });

            // Clear hasWork_ HERE, after ALL workers from this task have
            // finished. Otherwise, if hasWork_ remains true, a worker
            // whose completion wait (Phase 5) exits due to the NEXT
            // parallelFor's taskId_ increment will loop back, see
            // hasWork_ = true (from THIS task), and start working with
            // stale task data (nextIndex_, endIndex_, task_ from this
            // task, NOT from the next parallelFor which hasn't set up
            // its task yet).
            hasWork_ = false;
        }

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
