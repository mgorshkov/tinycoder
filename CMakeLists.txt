cmake_minimum_required(VERSION 3.18.0)

# TinyCoder - VS Code AI Coding Extension
# Uses np library for NumPy-style arrays with SIMD (AVX2/AVX512/AMX) and CUDA GPU
project(tinycoder VERSION 0.3.0 DESCRIPTION "High-Performant Local AI coding agent" LANGUAGES CXX)
enable_testing()

# ---- Options ----
option(ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)
option(ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(ENABLE_AVX512 "Enable AVX-512 optimizations" OFF)
option(ENABLE_AMX "Enable AMX (Advanced Matrix Extensions) optimizations" OFF)
option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(BUILD_SHARED_LIBS "Build shared library" OFF)
# Fused gate+up+down FFN kernel scheduling: ON uses dynamic chunked
# work-stealing (llama.cpp ggml_compute_forward_mul_mat_one_chunk style,
# lever 1 in plans/generation_optimizations.md); OFF restores the static slab
# dispatch. Measured 2026-08-27 (i7-4790K, 64-token questions, 49/49 parity):
# steal is faster at BOTH thread counts - 4 threads 19.44-19.88 -> 21.56-22.50
# tok/s (+11-14%, the parity levers' main unlock) and 8 threads 26.06-26.20 ->
# 26.33-26.55 (+~1%) - so ON is the default. Flip OFF only to A/B the slab.
option(TINYCODER_FFN_STEAL "Fused FFN kernel uses chunked work-stealing scheduling" ON)
# CPU affinity for worker threads. ON (default, 2026-08-28 re-measured) pins each
# participating thread (workers + main) to a DISTINCT logical CPU so no two
# threads share an HT lane on the memory-bound generation kernels. Measured on
# the reference host (i7-4790K, 8 logical CPUs) with this distinct-logical-CPU
# distribution (not the old physical-core-cramming modulo): interleaved A/B
# 25.73/25.79/25.76 tok/s (ON) vs 25.59/25.65/25.63 (OFF) — a small but
# reproducible +0.5% with 49/49 parity, at both 4 and 8 threads. The earlier OFF
# default was against the wrong (physical-core-cram) distribution. Runtime
# override: $TINYCODER_AFFINITY=0/1; the bridge also respects this now.
option(TINYCODER_AFFINITY "Pin worker threads to distinct logical CPUs" ON)
# Fused FFN act→Q8_K quantize scheduling. ON (default) runs the shared-act
# quantize cooperatively INSIDE phase 1 via per-block last-arriver counters
# (the llama.cpp `from_float` pattern — plans/generation_optimizations.md O1
# re-attempt at 4 threads): each 256-row act block is quantized by whichever
# thread stores its final 16-row tile, so the quantize ALU overlaps the rest
# of phase-1's DRAM streaming instead of serializing on the main thread
# between the phases. OFF restores the pre-O1 serial `between` callback (the
# A/B baseline). Bit-exact (identical quantized bytes), 49/49 parity holds.
option(TINYCODER_COOP_QUANTIZE "Fused FFN act quantize uses cooperative last-arriver scheduling" ON)
# Work-stealing chunk size (in tiles) for the fused FFN kernel. llama.cpp's
# single-token mul_mat dynamic work-stealing benefits from a larger chunk at low
# thread counts (4): smaller chunks add atomic-fetch_add contention. 8-12 at 4
# threads reduces that overhead.
set(TINYCODER_STEAL_CHUNK "4" CACHE STRING "Fused FFN work-stealing chunk size (tiles)")

# ---- PIC ----
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Fetch shared cmake configuration ----
include(FetchContent)

FetchContent_Declare(
    cmake
    GIT_REPOSITORY https://github.com/mgorshkov/cmake.git
    GIT_TAG main
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cmake)

# Apply global cmake options (compiler checks, CUDA setup, OpenMP, etc.)
include(${cmake_SOURCE_DIR}/Config.cmake)

# ---- Fetch np library ----
FetchContent_Declare(
    np
    GIT_REPOSITORY https://github.com/mgorshkov/np.git
    GIT_TAG main
    GIT_SHALLOW TRUE
    CMAKE_ARGS
    -DENABLE_CUDA=${ENABLE_CUDA}
    -DENABLE_OPENMP=${ENABLE_OPENMP}
    -DENABLE_SSE2=${ENABLE_SSE2}
    -DENABLE_SSE3=${ENABLE_SSE3}
    -DENABLE_AVX=${ENABLE_AVX}
    -DENABLE_AVX2=${ENABLE_AVX2}
    -DENABLE_AVX512=${ENABLE_AVX512}
    -DENABLE_AMX=${ENABLE_AMX}
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
    -DDEBUG_INFO=${DEBUG_INFO}
)

FetchContent_MakeAvailable(np)

# ---- Find N-API (Node.js Addon API) ----
# node-addon-api is fetched via npm, we find it in node_modules
set(NODE_ADDON_API_DIR "${CMAKE_SOURCE_DIR}/node_modules/node-addon-api")

if(NOT EXISTS "${NODE_ADDON_API_DIR}")
    message(FATAL_ERROR "node-addon-api not found. Run 'npm install' first.")
endif()

# Find Node.js for headers
find_program(NODE_EXECUTABLE node)
if(NODE_EXECUTABLE)
    execute_process(
        COMMAND ${NODE_EXECUTABLE} -e "process.stdout.write(process.versions.node)"
        OUTPUT_VARIABLE NODE_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Node.js version: ${NODE_VERSION}")
endif()

# Resolve N-API include path
# node-addon-api >= 8.x stores headers in node_modules/node-addon-api
# The main header is <napi.h> which includes <node_api.h> and <js_native_api.h>
# <node_api.h> is provided by the system Node.js headers (e.g., /usr/include/node/)
# or by the node-api-headers npm package
set(NAPI_HEADER_DIR "${CMAKE_SOURCE_DIR}/node_modules/node-addon-api")

# Find Node.js headers (provides node_api.h, js_native_api.h, etc.)
find_path(NODE_API_HEADER_DIR node_api.h
    PATHS
        /usr/include/node
        /usr/local/include/node
        /opt/homebrew/include/node
        ${CMAKE_SOURCE_DIR}/node_modules/node-api-headers/include
        ${CMAKE_SOURCE_DIR}/node_modules/node-addon-api/node_modules/node-api-headers/include
        ${CMAKE_SOURCE_DIR}/node_modules/@node-api/headers/include
)

if(NOT NODE_API_HEADER_DIR)
    message(FATAL_ERROR "node_api.h not found. Install Node.js development headers or run 'npm install'.")
endif()

message(STATUS "N-API addon: ${NODE_ADDON_API_DIR}")
message(STATUS "Node API headers: ${NODE_API_HEADER_DIR}")

# ---- Core engine library ----
# All engine sources live in src/cpp/core and are compiled exactly once into a
# static library (tinycoder_core). Both the native addon and the unit tests link
# against it, so the engine sources are never duplicated across targets.
file(GLOB TINYCODER_CORE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/cpp/core/*.cpp
)

# CUDA source files (only compiled when CUDA is enabled)
if(ENABLE_CUDA)
    file(GLOB TINYCODER_CUDA_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/src/cpp/core/*.cu
    )
    list(APPEND TINYCODER_CORE_SOURCES ${TINYCODER_CUDA_SOURCES})
    # The CUDA runtime needs explicit cuBLAS (already found below) plus the
    # shared library path for nvcc device-linking.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

set(TINYCODER_HEADERS
    include/ChatTemplateRenderer.hpp
    include/ModelConfig.hpp
    include/GGUFLoader.hpp
    include/IQ3XXS.hpp
    include/LMHead.hpp
    include/LMHeadCUDA.hpp
    include/ThreadPool.hpp
    include/Tokenizer.hpp
    include/Model.hpp
    include/ModelInternal.hpp
    include/SIMDMatMulVec.hpp
    include/SIMDMatMulVecInternal.hpp
)

# ---- SIMD compile flags for the SIMD mat-vec kernels ----
# The scalar implementations and runtime dispatch live in SIMDMatMulVec.cpp,
# which is compiled WITHOUT SIMD ISA extensions. The AVX2 and AVX-512 kernels
# live in their own translation units (SIMDMatMulVecAVX2.cpp /
# SIMDMatMulVecAVX512.cpp) so that per-file flags can enable the required ISA
# without forcing AVX/AVX-512/AMX code generation onto the rest of the codebase.
# Each kernel file uses #ifdef guards so it compiles to an empty TU when the
# corresponding ISA is disabled.
set(SIMD_MATMUL_AVX2_FLAGS "")
if(ENABLE_AVX2)
    set(SIMD_MATMUL_AVX2_FLAGS "${SIMD_MATMUL_AVX2_FLAGS} -mavx2 -mfma -mf16c")
endif()
if(TINYCODER_FFN_STEAL)
    # Lever 1 A/B: route the fused gate+up+down FFN through the chunked
    # work-stealing scheduler (parallelForSteal2) instead of the static slabs.
    set(SIMD_MATMUL_AVX2_FLAGS "${SIMD_MATMUL_AVX2_FLAGS} -DTINYCODER_USE_FFN_STEAL")
    # Work-stealing chunk size (tiles) for both phases of the fused FFN.
    set(SIMD_MATMUL_AVX2_FLAGS "${SIMD_MATMUL_AVX2_FLAGS} -DTINYCODER_STEAL_CHUNK=${TINYCODER_STEAL_CHUNK}")
endif()
if(TINYCODER_COOP_QUANTIZE)
    # O1 cooperative quantize A/B: define TINYCODER_USE_COOP_QUANTIZE so the
    # fused FFN's act→Q8_K quantize runs inline in phase-1 workers via the
    # last-arriver counters (llama.cpp from_float) instead of the serial
    # main-thread `between` callback.
    set(SIMD_MATMUL_AVX2_FLAGS "${SIMD_MATMUL_AVX2_FLAGS} -DTINYCODER_USE_COOP_QUANTIZE")
endif()
if(SIMD_MATMUL_AVX2_FLAGS)
    set_source_files_properties(
        src/cpp/core/SIMDMatMulVecAVX2.cpp
        PROPERTIES
        COMPILE_FLAGS "${SIMD_MATMUL_AVX2_FLAGS}"
    )
endif()

# The sampling pipeline (temperature/top-K/top-P softmax filtering + the
# cumulative-scan draw) runs a full-vocab pass a few times per token; give it
# the same AVX2/FMA flags as the SIMD kernel TU (guarded by __AVX2__/__FMA__
# inside the source, so it degrades to scalar otherwise).
set(SAMPLING_AVX2_FLAGS "")
if(ENABLE_AVX2)
    set(SAMPLING_AVX2_FLAGS "${SAMPLING_AVX2_FLAGS} -mavx2 -mfma")
endif()
if(SAMPLING_AVX2_FLAGS)
    set_source_files_properties(
        src/cpp/core/ModelSampling.cpp
        PROPERTIES
        COMPILE_FLAGS "${SAMPLING_AVX2_FLAGS}"
    )
endif()

# AVX-512 is not supported on this CPU, so these flags are disabled by default.
# Enable ENABLE_AVX512 to compile the AVX-512 kernels. -mavx512bw is required for
# the byte/word intrinsics (_mm512_maddubs_epi16, _mm512_madd_epi16) used by the
# Q8_K and pre-packed Q2_K kernels.
set(SIMD_MATMUL_AVX512_FLAGS "")
if(ENABLE_AVX512)
    set(SIMD_MATMUL_AVX512_FLAGS "${SIMD_MATMUL_AVX512_FLAGS} -mavx512f -mavx512bw -mfma")
endif()
if(SIMD_MATMUL_AVX512_FLAGS)
    set_source_files_properties(
        src/cpp/core/SIMDMatMulVecAVX512.cpp
        PROPERTIES
        COMPILE_FLAGS "${SIMD_MATMUL_AVX512_FLAGS}"
    )
endif()

# Build the engine core once; tests and the addon both link this target.
add_library(tinycoder_core STATIC ${TINYCODER_CORE_SOURCES})

target_include_directories(tinycoder_core
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${np_SOURCE_DIR}/include
)

target_link_libraries(tinycoder_core
    PUBLIC
    np
)

set_target_properties(tinycoder_core PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Compile-time configuration for the thread pool + fused FFN:
#   TINYCODER_AFFINITY   - 1 pins workers to distinct physical cores, 0 lets the
#                          OS schedule on all logical CPUs (default; faster at
#                          low thread counts on memory-bound generation).
#   TINYCODER_STEAL_CHUNK - work-stealing chunk size (tiles) for the fused FFN.
target_compile_definitions(tinycoder_core PRIVATE
    TINYCODER_AFFINITY=$<IF:$<BOOL:${TINYCODER_AFFINITY}>,1,0>
    TINYCODER_STEAL_CHUNK=${TINYCODER_STEAL_CHUNK}
)
# CUDA configuration for the core library
if(ENABLE_CUDA)
    target_compile_definitions(tinycoder_core PUBLIC USE_CUDA)
    set_target_properties(tinycoder_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    find_library(CUBLAS_LIBRARY cublas)
    if(CUBLAS_LIBRARY)
        target_link_libraries(tinycoder_core PRIVATE ${CUBLAS_LIBRARY})
        target_compile_definitions(tinycoder_core PRIVATE USE_CUBLAS)
        message(STATUS "cuBLAS found: ${CUBLAS_LIBRARY}")
    else()
        message(WARNING "cuBLAS not found — LM head CUDA acceleration disabled")
    endif()
endif()

# ---- Build the native addon ----
# The addon is a thin N-API module: it only compiles the bridge and links the
# engine core (tinycoder_core) instead of re-compiling all engine sources.
add_library(tinycoder_native MODULE
    ${CMAKE_SOURCE_DIR}/src/cpp/bridge/Bridge.cpp
)

# Apply target-dependent cmake options (compiler features, CUDA defines, linker)
# Must set TARGET_NAME before including Target.cmake so CUDA settings apply to the correct target
set(TARGET_NAME tinycoder_native)
include(${cmake_SOURCE_DIR}/Target.cmake)

# Include Node.js / N-API headers only (engine headers come from tinycoder_core)
target_include_directories(tinycoder_native PRIVATE
    ${NODE_ADDON_API_DIR}
    ${NODE_API_HEADER_DIR}
)

# Link the engine core and its dependencies (np, CUDA, cuBLAS)
target_link_libraries(tinycoder_native PRIVATE tinycoder_core)

# Note: OpenMP is no longer used directly by TinyCoder.
# The np library may still be built with OpenMP via its own CMake configuration.
# ENABLE_OPENMP option is forwarded to np's FetchContent for that purpose.

# Set output properties for Node.js native addon
set_target_properties(tinycoder_native PROPERTIES
    PREFIX ""
    SUFFIX ".node"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
)

# Platform-specific settings
if(APPLE)
    set_target_properties(tinycoder_native PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
elseif(WIN32)
    set_target_properties(tinycoder_native PROPERTIES
        LINK_FLAGS "-Wl,--out-implib,${CMAKE_BINARY_DIR}/Release/tinycoder_native.lib"
    )
endif()

# ---- Unit Tests (Google Test) ----
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    add_subdirectory(unit_tests)
    # llama-bench-parity warm generation benchmark (benchmarks/BenchMain.cpp).
    # Kept intentionally OUT of unit_tests/: it is a measurement harness, not a
    # parity/correctness test — it would add a gtest-incompatible generate loop.
    add_subdirectory(benchmarks)
endif()

# ---- Install ----
install(TARGETS tinycoder_native DESTINATION native)
install(FILES ${TINYCODER_HEADERS} DESTINATION include/tinycoder)

# ---- Print configuration summary ----
message(STATUS "")
message(STATUS "TinyCoder Build Configuration:")
message(STATUS "  AVX2:          ${ENABLE_AVX2}")
message(STATUS "  AVX-512:       ${ENABLE_AVX512}")
message(STATUS "  AMX:           ${ENABLE_AMX}")
message(STATUS "  OpenMP:        ${ENABLE_OPENMP}")
message(STATUS "  CUDA:          ${ENABLE_CUDA}")
message(STATUS "  C++ Standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  np library:    ${np_SOURCE_DIR}")
message(STATUS "  Output:        ${CMAKE_BINARY_DIR}/Release/tinycoder_native.node")
message(STATUS "")
