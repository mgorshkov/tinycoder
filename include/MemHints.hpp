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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace tinycoder {

    /// @brief Transparent-Huge-Page (THP) hints for large inference buffers.
    ///
    /// Background (measured 2026-08-29, plans/generation_optimizations.md
    /// §6.10): on the reference host the forward pass streams multi-hundred-MB
    /// weight buffers (tensorData_ ~750 MB, pre-dequantized embeddings ~890 MB,
    /// LM-head Q8_K ~253 MB) from DRAM every token. With THP=madvise and
    /// AnonHugePages=0, those buffers are backed by 4 KB pages, so a sequential
    /// sweep touches ~179K dTLB entries per buffer pass — perf showed TinyCoder
    /// at 4.65x llama.cpp's dTLB misses (61.1 M vs 13.2 M) with llama mmapping
    /// huge pages. Hinting the kernel to back big buffers with 2 MB pages
    /// collapses that dTLB working set ~512x.
    ///
    /// - `madviseHintHuge(page-aligned ptr, bytes)` issues
    ///   `madvise(MADV_HUGEPAGE)` — a pure hint: the kernel promotes pages
    ///   opportunistically and NEVER faults on the hint (no success check
    ///   needed; failure is a no-op).
    /// - `adviseHugePages(ptr, bytes)` rounds the range down to page
    ///   boundaries and only acts on allocations >= 2 MiB (a single huge page),
    ///   skipping tiny std::vector bananas like norms/embeddings.
    ///
    /// Available on Linux only; on other platforms it is a no-op. Safe to call
    /// after std::vector/np::Array resize/move (the realloc'd buffer remains
    /// heap-resident and madvise works on any mapped private anonymous range).
    ///
    /// Opt-in: `TINYCODER_HUGE_PAGES=1` enables the hints. **Default OFF**
    /// (2026-08-29, measured): with THP=madvise on the reference host the
    /// hint hands khugepaged a ~2.4 GB range to collapse DURING inference
    /// (10 s scan cycle, `defrag=1`). Each collapse copies 512 x 4K pages into
    /// a 2 MB page + TLB-shootdowns all cores; only ~16 MB / 750 MB actually
    /// promoted, so the dTLB benefit was negligible while the churn cost was
    /// real: tg64 8t with hints ON was 24.59 +/- 2.67 (ramping 27.2 -> 21.2
    /// across 5 reps) vs **27.09 +/- 0.46 (flat)** with hints OFF. llama.cpp
    /// gets here for free because it mmaps weights (file-backed PMD pages via
    /// page-cache readahead); a heap-resident std::vector promotes only via
    /// khugepaged, which is the regression. Keep the gate for hosts where
    /// promotion is cheap (defrag=never, uncongested) or until a proper
    /// mmap-backed loader replaces the heap buffers.
    inline void adviseHugePages(void *ptr, size_t bytes) {
#if defined(__linux__)
        static const int enabled = [] {
            const char *e = std::getenv("TINYCODER_HUGE_PAGES");
            return e != nullptr && e[0] == '1';
        }();
        if (!enabled) {
            return;
        }
        if (ptr == nullptr || bytes < (2u * 1024 * 1024)) {
            return;
        }
        constexpr size_t kPage = 4096;
        uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t end = start + bytes;
        uintptr_t a = (start + kPage - 1) & ~(kPage - 1);
        uintptr_t b = end & ~(kPage - 1);
        if (a >= b) {
            return;
        }
        ::madvise(reinterpret_cast<void *>(a), b - a, MADV_HUGEPAGE);
#else
        (void) ptr;
        (void) bytes;
#endif
    }

}// namespace tinycoder