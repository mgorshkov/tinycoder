cmake_minimum_required(VERSION 3.18.0)

set(PROJECT_NAME tinycoder_unit_tests)

project(${PROJECT_NAME} VERSION 0.3.0 DESCRIPTION "TinyCoder Unit Tests")

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# tinycoder_core static library is built in the root CMakeLists.txt
# and is available as a target in this subdirectory scope.

# ---- Google Test executable ----
add_executable(tinycoder_test
    Main.cpp
    ModelTest.cpp
    DequantizeTest.cpp
    ReferenceCompareTest.cpp
    ThreadPoolReentrancyTest.cpp
    GPUCpuCompareTest.cpp
    SharedTestEnv.cpp
)

target_link_libraries(tinycoder_test
    PRIVATE
    gtest
    tinycoder_core
)

target_include_directories(tinycoder_test
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Testing
)

# Set C++ standard for the test executable
set_target_properties(tinycoder_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Register the test with CTest
add_test(NAME tinycoder_test COMMAND tinycoder_test)
