// ThreadPoolReentrancyTest.cpp
// Unit tests for ThreadPool nested parallelFor behavior

#include "ThreadPool.hpp"
#include <atomic>
#include <gtest/gtest.h>

// Test that a nested parallelFor completes correctly and does not deadlock.
TEST(ThreadPool, NestedParallelFor) {
    // Initialize the global thread pool with a few threads.
    tinycoder::ThreadPool::instance().initialize(4);

    std::atomic<int> counter{0};
    const uint32_t outer = 10;
    const uint32_t inner = 5;

    tinycoder::ThreadPool::instance().parallelFor(0, outer, [&](uint32_t i) {
        // Inside each outer iteration, run an inner parallel loop.
        tinycoder::ThreadPool::instance().parallelFor(0, inner, [&](uint32_t j) {
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    });

    EXPECT_EQ(counter.load(std::memory_order_relaxed), outer * inner);
}

// Test the edge case where the outer loop has a single iteration.
TEST(ThreadPool, NestedParallelForSingleOuter) {
    tinycoder::ThreadPool::instance().initialize(4);

    std::atomic<int> counter{0};
    const uint32_t inner = 3;

    tinycoder::ThreadPool::instance().parallelFor(0, 1, [&](uint32_t i) {
        tinycoder::ThreadPool::instance().parallelFor(0, inner, [&](uint32_t j) {
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    });

    EXPECT_EQ(counter.load(std::memory_order_relaxed), inner);
}
