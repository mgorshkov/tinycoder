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
#include <utility>
#include <vector>

namespace tinycoder {

    /// @brief A std::vector-like contiguous buffer whose data is aligned to a
    /// cache-line boundary (default 64 bytes).
    ///
    /// Weight matrices are read by the SIMD mat-vec kernels with `_mm256_load_ps`
    /// / `_mm256_loadu_si128` etc. 64-byte alignment guarantees each row block
    /// starts on a cache-line boundary, which reduces L1/L2 cache-miss penalties
    /// and lets the compiler use aligned (non-`loadu`) loads where possible.
    ///
    /// The API intentionally mirrors the subset of std::vector used by the model
    /// loader and kernels, so call sites change only by type.
    template<typename T, size_t ALIGN = 64>
    class AlignedVector {
    public:
        static_assert(ALIGN >= sizeof(void *), "alignment must be at least a pointer");
        static_assert((ALIGN & (ALIGN - 1)) == 0, "alignment must be a power of two");

        AlignedVector() = default;
        ~AlignedVector() { deallocate(); }

        AlignedVector(const AlignedVector &other) { assign(other.data_, other.n_); }
        AlignedVector(AlignedVector &&other) noexcept { moveFrom(other); }

        AlignedVector &operator=(const AlignedVector &other) {
            if (this != &other) {
                assign(other.data_, other.n_);
            }
            return *this;
        }
        AlignedVector &operator=(AlignedVector &&other) noexcept {
            if (this != &other) {
                deallocate();
                moveFrom(other);
            }
            return *this;
        }

        /// @brief Copy-assign from a std::vector (e.g. helper functions that return
        /// std::vector accumulate into the aligned buffer). Copies the contents and
        /// reallocates the aligned backing store.
        template<typename U, typename Alloc>
        AlignedVector &operator=(const std::vector<U, Alloc> &src) {
            assign(src.data(), src.data() + src.size());
            return *this;
        }

        /// @brief Resize the buffer and leave contents uninitialized.
        void resize(size_t n) {
            if (n == n_) {
                return;
            }
            allocate(n);
        }

        /// @brief Fill the buffer from a contiguous range [first, last).
        void assign(const T *first, const T *last) {
            const size_t count = static_cast<size_t>(last - first);
            allocate(count);
            if (count > 0) {
                std::memcpy(data_, first, count * sizeof(T));
            }
        }

        /// @brief Fill from a std::vector source (convenience for existing callers).
        template<typename U, typename Alloc>
        void assign(const std::vector<U, Alloc> &src) {
            assign(src.data(), src.data() + src.size());
        }

        T *data() noexcept { return data_; }
        const T *data() const noexcept { return data_; }

        size_t size() const noexcept { return n_; }
        bool empty() const noexcept { return n_ == 0; }

        T &operator[](size_t i) noexcept { return data_[i]; }
        const T &operator[](size_t i) const noexcept { return data_[i]; }

        void clear() {
            deallocate();
            n_ = 0;
        }

    private:
        void allocate(size_t n) {
            deallocate();
            n_ = n;
            if (n_ == 0) {
                data_ = nullptr;
                return;
            }
            size_t bytes = n_ * sizeof(T);
            if (bytes == 0) {
                data_ = nullptr;
                return;
            }
            void *ptr = nullptr;
            // posix_memalign requires the size to be a multiple of the alignment.
            size_t allocSize = ((bytes + ALIGN - 1) / ALIGN) * ALIGN;
            if (allocSize < bytes) {
                // overflow guard; fall back to null (caller treats as OOM)
                data_ = nullptr;
                n_ = 0;
                return;
            }
            if (posix_memalign(&ptr, ALIGN, allocSize) != 0 || ptr == nullptr) {
                data_ = nullptr;
                n_ = 0;
                return;
            }
            data_ = static_cast<T *>(ptr);
        }

        void deallocate() {
            if (data_ != nullptr) {
                std::free(data_);
                data_ = nullptr;
            }
        }

        void moveFrom(AlignedVector &other) noexcept {
            data_ = other.data_;
            n_ = other.n_;
            other.data_ = nullptr;
            other.n_ = 0;
        }

        T *data_ = nullptr;
        size_t n_ = 0;
    };

}// namespace tinycoder