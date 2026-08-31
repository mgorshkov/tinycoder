cmake_minimum_required(VERSION 3.18.0)

# TinyCoder - VS Code AI Coding Extension
# Uses np library for NumPy-style arrays with SIMD (AVX2/AVX512/AMX) and CUDA GPU
# The C++/CUDA inference engine lives in a separate repository
# (https://github.com/mgorshkov/tinycoder-inference) and is consumed here via
# FetchContent as the tinycoder_core static library. This tree keeps only the
# N-API bridge (src/cpp/bridge) and the TypeScript extension sources.
project(tinycoder VERSION 0.4.0 DESCRIPTION "High-Performant Local AI coding agent" LANGUAGES CXX)
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

# ---- Fetch the inference engine ----
# The C++/CUDA inference core now lives in its own repository
# (https://github.com/mgorshkov/tinycoder-inference). It compiles the engine sources
# (src/cpp/core) exactly once into the tinycoder_core static library, which the
# native addon links against. All engine options (SIMD flags, fused FFN
# scheduling, thread affinity, CUDA, ...) and the engine's unit tests +
# benchmarks live there; BUILD_TESTS is forced OFF here because the tests are
# run standalone inside the inference repository.
FetchContent_Declare(
    tinycoder_inference
    GIT_REPOSITORY https://github.com/mgorshkov/tinycoder-inference.git
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
    -DTINYCODER_FFN_STEAL=${TINYCODER_FFN_STEAL}
    -DTINYCODER_AFFINITY=${TINYCODER_AFFINITY}
    -DTINYCODER_COOP_QUANTIZE=${TINYCODER_COOP_QUANTIZE}
    -DTINYCODER_STEAL_CHUNK=${TINYCODER_STEAL_CHUNK}
    -DBUILD_TESTS=OFF
)

# Force the engine's tests/benchmarks OFF when embedded. FetchContent_MakeAvailable
# adds the subproject with add_subdirectory(), which ignores the CMAKE_ARGS above;
# without this cache variable the engine's option(BUILD_TESTS ... ON) default would
# pull googletest and the engine test/bench targets into the addon build.
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinycoder_inference)

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

# ---- Build the native addon ----
# The addon is a thin N-API module: it only compiles the bridge and links the
# engine core (tinycoder_core) from the fetched tinycoder-inference repository.
# The engine sources are NOT compiled in this tree anymore.
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

# Link the engine core and its dependencies (np, CUDA, cuBLAS). The np library
# is fetched and built by the inference engine's own CMakeLists.txt; the addon
# receives it transitively through tinycoder_core's PUBLIC link.
target_link_libraries(tinycoder_native PRIVATE tinycoder_core)

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

# ---- Unit tests & benchmarks ----
# The unit tests (unit_tests/, Google Test) and the llama-bench-parity benchmark
# harness (benchmarks/) moved to the tinycoder-inference repository together
# with the engine. Build and run them there (BUILD_TESTS=ON), not here.

# ---- Install ----
install(TARGETS tinycoder_native DESTINATION native)

# ---- Print configuration summary ----
message(STATUS "")
message(STATUS "TinyCoder Build Configuration:")
message(STATUS "  AVX2:          ${ENABLE_AVX2}")
message(STATUS "  AVX-512:       ${ENABLE_AVX512}")
message(STATUS "  AMX:           ${ENABLE_AMX}")
message(STATUS "  OpenMP:        ${ENABLE_OPENMP}")
message(STATUS "  CUDA:          ${ENABLE_CUDA}")
message(STATUS "  C++ Standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  Engine core:   ${tinycoder_inference_SOURCE_DIR}")
message(STATUS "  Output:        ${CMAKE_BINARY_DIR}/Release/tinycoder_native.node")
message(STATUS "")
