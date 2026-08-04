// ThreadPoolReentrancyTest.cpp
// Unit tests for ThreadPool nested parallelFor behavior

#include "ThreadPool.hpp"
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

// Two-phase single-launch pipeline (parallelForSlab2): phase 1 writes into a
// shared array (one distinct element per invocation index), the main-thread
// `between` sums them serially, and phase 2 must observe the FULL phase-1
// output + the between result. Verifies the flag-release ordering and that the
// barrier-free phase transition does not lose or double-count work.
TEST(ThreadPool, TwoPhaseSlabPipeline) {
    tinycoder::ThreadPool::instance().initialize(4);

    constexpr uint32_t N1 = 1000;
    constexpr uint32_t N2 = 200;
    std::vector<int> phase1Out(N1, 0);
    std::atomic<int> phase1Sum{0};
    std::atomic<int> finalSum{0};
    std::atomic<int> betweenCalls{0};

    tinycoder::ThreadPool::instance().parallelForSlab2(
            0, N1,
            [&](uint32_t i) {
                phase1Out[i] = static_cast<int>(i) + 1;
            },
            [&]() {
                // Serialized between-step: every phase-1 element must be set.
                for (uint32_t i = 0; i < N1; ++i) {
                    phase1Sum.fetch_add(phase1Out[i], std::memory_order_relaxed);
                }
                betweenCalls.fetch_add(1, std::memory_order_relaxed);
            },
            0, N2,
            [&](uint32_t) {
                // Phase 2 must observe between()'s release: each phase-2 item
                // adds the whole phase-1 sum.
                int s = phase1Sum.load(std::memory_order_acquire);
                finalSum.fetch_add(s, std::memory_order_relaxed);
            });

    const int phase1Total = static_cast<int>(N1) * (N1 + 1) / 2;
    EXPECT_EQ(phase1Sum.load(std::memory_order_relaxed), phase1Total);
    EXPECT_EQ(betweenCalls.load(std::memory_order_relaxed), 1);
    EXPECT_EQ(finalSum.load(std::memory_order_relaxed), phase1Total * static_cast<int>(N2));
}

// Repeated invocation stress (no idle wait between calls): exercises the
// phase2Ready_ flag reset between consecutive single-launch pipelines.
TEST(ThreadPool, TwoPhaseSlabStress) {
    tinycoder::ThreadPool::instance().initialize(4);

    constexpr uint32_t N1 = 500;
    constexpr uint32_t N2 = 120;
    constexpr uint32_t iterations = 200;
    std::atomic<uint64_t> total{0};

    for (uint32_t it = 0; it < iterations; ++it) {
        tinycoder::ThreadPool::instance().parallelForSlab2(
                0, N1,
                [](uint32_t i) {
                    (void) i;
                },
                [&]() {
                    // between runs once per iteration; phase-2 count feeds total.
                },
                0, N2,
                [&](uint32_t j) {
                    total.fetch_add(j + 1, std::memory_order_relaxed);
                });
    }

    const uint64_t perIter = static_cast<uint64_t>(N2) * (N2 + 1) / 2;
    EXPECT_EQ(total.load(std::memory_order_relaxed), perIter * iterations);
}

// Test that a nested parallelFor completes correctly and does not deadlock.
TEST(ThreadPool, NestedParallelFor) {
    // Initialize the global thread pool with a few threads.
    tinycoder::ThreadPool::instance().initialize(4);

    std::atomic<int> counter{0};
    const uint32_t outer = 10;
    const uint32_t inner = 5;

    tinycoder::ThreadPool::instance().parallelFor(0, outer, [&](uint32_t) {
        // Inside each outer iteration, run an inner parallel loop.
        tinycoder::ThreadPool::instance().parallelFor(0, inner, [&](uint32_t) {
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

    tinycoder::ThreadPool::instance().parallelFor(0, 1, [&](uint32_t) {
        tinycoder::ThreadPool::instance().parallelFor(0, inner, [&](uint32_t) {
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    });

    EXPECT_EQ(counter.load(std::memory_order_relaxed), inner);
}

// Stress test for the persistent-worker scheduler: many consecutive parallelFor
// calls with no intervening work, simulating the per-matmul barrier pattern in
// the transformer forward pass (~168 syncs per token). This exercises the
// lock-free spin barrier path (workers stay resident across calls) and verifies
// that the generation-counter ack scheme never loses or double-counts work.
TEST(ThreadPool, ConsecutiveParallelForStress) {
    tinycoder::ThreadPool::instance().initialize(4);

    std::atomic<uint64_t> total{0};
    const uint32_t iterations = 2000;
    const uint32_t range = 1000;

    for (uint32_t it = 0; it < iterations; ++it) {
        tinycoder::ThreadPool::instance().parallelFor(0, range, [&](uint32_t i) {
            total.fetch_add(i, std::memory_order_relaxed);
        });
    }

    // Each iteration sums 0..(range-1) = range*(range-1)/2.
    const uint64_t perIter = static_cast<uint64_t>(range) * (range - 1) / 2;
    EXPECT_EQ(total.load(std::memory_order_relaxed), perIter * iterations);
}

// Verify the hybrid spin/block fallback: after a burst of parallelFor calls,
// workers fall back to blocking on the condition variable when idle. A
// subsequent parallelFor must still wake them and complete correctly. This
// exercises the blockedWorkers_ notify path.
TEST(ThreadPool, BlockFallbackAfterIdle) {
    tinycoder::ThreadPool::instance().initialize(4);

    std::atomic<int> counter{0};
    const uint32_t range = 500;

    // Burst of work to warm the spin barrier.
    for (int it = 0; it < 100; ++it) {
        tinycoder::ThreadPool::instance().parallelFor(0, range, [&](uint32_t) {
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    }

    // Sleep to let workers exhaust their spin budget and block on the CV.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // A new parallelFor must wake the blocked workers and complete.
    tinycoder::ThreadPool::instance().parallelFor(0, range, [&](uint32_t) {
        counter.fetch_add(1, std::memory_order_relaxed);
    });

    EXPECT_EQ(counter.load(std::memory_order_relaxed), 101 * static_cast<int>(range));
}

// Regression test for the batch-prefill deadlock. The transformer forward pass
// runs an outer parallelFor over tokens, and inside each token does heavy
// sequential work (nested matmuls that run sequentially due to the reentrancy
// guard). This pattern previously triggered a runaway in the persistent-worker
// scheduler where workers re-processed the last task with pendingWorkers_
// underflowing, hanging the batch prefill test (seqLen=31). The fix uses a
// single generation-counter predicate (taskId_ != myGen) for both the work-wait
// and the ack-wait.
TEST(ThreadPool, OuterParallelForWithHeavyWork) {
    tinycoder::ThreadPool::instance().initialize(4);

    std::atomic<uint64_t> total{0};
    const uint32_t outer = 31;
    const uint32_t inner = 1000;
    const uint32_t reps = 50;

    for (uint32_t rep = 0; rep < reps; ++rep) {
        tinycoder::ThreadPool::instance().parallelFor(0, outer, [&](uint32_t s) {
            uint64_t acc = 0;
            for (uint32_t i = 0; i < inner; ++i) {
                acc += i + s;
            }
            total.fetch_add(acc, std::memory_order_relaxed);
        });
    }

    // Per (rep, s): sum_{i=0}^{inner-1} (i + s) = inner*(inner-1)/2 + inner*s.
    // Summed over s in [0, outer) and reps.
    const uint64_t perS = static_cast<uint64_t>(inner) * (inner - 1) / 2;
    const uint64_t perRep = outer * perS + inner * (static_cast<uint64_t>(outer) * (outer - 1) / 2);
    EXPECT_EQ(total.load(std::memory_order_relaxed), perRep * reps);
}
