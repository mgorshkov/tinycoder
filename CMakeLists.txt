cmake_minimum_required(VERSION 3.18.0)

# TinyCoder - VS Code AI Coding Extension with Qwen2.5-Coder
# Uses np library for NumPy-style arrays with SIMD (AVX2/AVX512/AMX) and CUDA GPU
project(tinycoder VERSION 0.1.0 DESCRIPTION "VS Code AI Coding Assistant with Qwen2.5-Coder" LANGUAGES CXX)

# ---- Options ----
option(ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)
option(ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(ENABLE_AVX512 "Enable AVX-512 optimizations" OFF)
option(ENABLE_AMX "Enable AMX (Advanced Matrix Extensions) optimizations" OFF)
option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(BUILD_SHARED_LIBS "Build shared library" OFF)

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

# ---- Source files ----
set(TINYCODER_SOURCES
    src/cpp/GGUFLoader.cpp
    src/cpp/GridTables.cpp
    src/cpp/GridTablesIQ3S.cpp
    src/cpp/Model.cpp
    src/cpp/QuantizedEmbedding.cpp
    src/cpp/QuantizedMatrix.cpp
    src/cpp/SIMDMatMulVec.cpp
    src/cpp/Tokenizer.cpp
    src/cpp/Bridge.cpp
)

# CUDA source files (only compiled when CUDA is enabled)
if(ENABLE_CUDA)
    set(TINYCODER_CUDA_SOURCES
        src/cpp/LMHeadCUDA.cu
    )
    list(APPEND TINYCODER_SOURCES ${TINYCODER_CUDA_SOURCES})
endif()

set(TINYCODER_HEADERS
    include/ModelConfig.hpp
    include/GGUFLoader.hpp
    include/IQ3XXS.hpp
    include/LMHead.hpp
    include/LMHeadCUDA.hpp
    include/Tokenizer.hpp
    include/Model.hpp
)

# ---- Build the native addon ----
add_library(tinycoder_native MODULE ${TINYCODER_SOURCES})

# Apply target-dependent cmake options (compiler features, CUDA defines, linker)
# Must set TARGET_NAME before including Target.cmake so CUDA settings apply to the correct target
set(TARGET_NAME tinycoder_native)
include(${cmake_SOURCE_DIR}/Target.cmake)

# Include directories
target_include_directories(tinycoder_native PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${np_SOURCE_DIR}/include
    ${NODE_ADDON_API_DIR}
    ${NODE_API_HEADER_DIR}
)

# Link np library
target_link_libraries(tinycoder_native PRIVATE np)

# Link cuBLAS when CUDA is enabled
if(ENABLE_CUDA)
    find_library(CUBLAS_LIBRARY cublas)
    if(CUBLAS_LIBRARY)
        target_link_libraries(tinycoder_native PRIVATE ${CUBLAS_LIBRARY})
        target_compile_definitions(tinycoder_native PRIVATE USE_CUBLAS)
        message(STATUS "cuBLAS found: ${CUBLAS_LIBRARY}")
    else()
        message(WARNING "cuBLAS not found — LM head CUDA acceleration disabled")
    endif()
endif()

# Link OpenMP if enabled (np library was built with OpenMP)
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(tinycoder_native PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---- SIMD compile flags for SIMDMatMulVec.cpp ----
# The file uses #ifdef guards to select the correct code path at compile time,
# and runtime CPUID detection to dispatch to the best available implementation.
# We apply per-file flags so only SIMDMatMulVec.cpp is compiled with SIMD ISA
# extensions, avoiding AVX/AVX-512/AMX code generation in other translation units.
set(SIMD_MATMUL_FLAGS "")
if(ENABLE_AVX2)
    set(SIMD_MATMUL_FLAGS "${SIMD_MATMUL_FLAGS} -mavx2 -mfma")
endif()
# AVX-512 and AMX are not supported on this CPU, so these flags are disabled
# if(ENABLE_AVX512)
#     set(SIMD_MATMUL_FLAGS "${SIMD_MATMUL_FLAGS} -mavx512f -mfma")
# endif()
# if(ENABLE_AMX)
#     set(SIMD_MATMUL_FLAGS "${SIMD_MATMUL_FLAGS} -mamx-tile")
# endif()
if(SIMD_MATMUL_FLAGS)
    set_source_files_properties(
        src/cpp/SIMDMatMulVec.cpp
        PROPERTIES
        COMPILE_FLAGS "${SIMD_MATMUL_FLAGS}"
    )
endif()

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

# ---- Common static library (TinyCoder core sources) ----
# Build all shared sources into a static library so test executables
# only need to link against it, avoiding source list duplication and
# ensuring consistent CUDA configuration across all test targets.

set(TINYCODER_COMMON_SOURCES
    src/cpp/Model.cpp
    src/cpp/Tokenizer.cpp
    src/cpp/QuantizedMatrix.cpp
    src/cpp/QuantizedEmbedding.cpp
    src/cpp/GGUFLoader.cpp
    src/cpp/GridTables.cpp
    src/cpp/GridTablesIQ3S.cpp
    src/cpp/SIMDMatMulVec.cpp
)

# CUDA source files (only added when CUDA is enabled)
if(ENABLE_CUDA)
    list(APPEND TINYCODER_COMMON_SOURCES
        src/cpp/LMHeadCUDA.cu
    )
endif()

add_library(tinycoder_common STATIC ${TINYCODER_COMMON_SOURCES})

target_include_directories(tinycoder_common
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(tinycoder_common
    PUBLIC
    np
)

# Set C++ standard for the common library
set_target_properties(tinycoder_common PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# CUDA configuration for the common library
if(ENABLE_CUDA)
    target_compile_definitions(tinycoder_common PUBLIC USE_CUDA)
    set_target_properties(tinycoder_common PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    find_library(CUBLAS_LIBRARY cublas)
    if(CUBLAS_LIBRARY)
        target_link_libraries(tinycoder_common PRIVATE ${CUBLAS_LIBRARY})
        target_compile_definitions(tinycoder_common PRIVATE USE_CUBLAS)
        message(STATUS "cuBLAS found for tests: ${CUBLAS_LIBRARY}")
    else()
        message(WARNING "cuBLAS not found — LM head CUDA acceleration disabled in tests")
    endif()
endif()

# ---- Unit Tests (Google Test) ----
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    add_subdirectory(unit_tests)
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
