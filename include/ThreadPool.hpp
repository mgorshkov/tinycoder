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

namespace tinycoder {

    /// @brief Reusable thread pool that replaces OpenMP parallel regions.
    ///
    /// Threads are created once during initialization and reused across all
    /// parallel for loops, eliminating the thread creation/destruction overhead
    /// of OpenMP's fork-join model.
    ///
    /// Uses a generation-counter completion scheme to safely support
    /// consecutive parallelFor calls: each parallelFor gets a unique
    /// taskId, and workers wait for taskId to change before looping back.
    /// This eliminates the race where hasWork_ toggles between "task done"
    /// and "new task ready" signals.
    class ThreadPool {
    public:
        static ThreadPool &instance();

        void initialize(size_t numThreads);
        size_t numThreads() const { return numThreads_; }
        bool isInitialized() const { return initialized_; }

        void parallelFor(uint32_t start, uint32_t end,
                         const std::function<void(uint32_t)> &func);

        void parallelFor2D(uint32_t dim1, uint32_t dim2,
                           const std::function<void(uint32_t, uint32_t)> &func);

    private:
        ThreadPool() = default;
        ~ThreadPool();
        ThreadPool(const ThreadPool &) = delete;
        ThreadPool &operator=(const ThreadPool &) = delete;
        void workerMain();

        size_t numThreads_ = 1;
        bool initialized_ = false;
        std::vector<std::thread> workers_;

        std::mutex mutex_;
        std::condition_variable cv_;

        // Task data
        std::function<void(uint32_t)> task_;
        std::atomic<uint32_t> nextIndex_{0};
        uint32_t endIndex_ = 0;

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
    };

}// namespace tinycoder
