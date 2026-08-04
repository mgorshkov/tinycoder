#!/bin/bash
#
# build.sh — Build the TinyCoder VS Code extension and package it as a .vsix.
#
# This script:
#   1. Installs npm dependencies (if needed).
#   2. Compiles TypeScript sources.
#   3. Builds the native C++ addon via CMake.
#   4. Packages the extension into a .vsix file using vsce.
#
# CUDA (GPU offload) is the DEFAULT build path — the engine is developed and
# benchmarked against the llama.cpp GPU baseline with -DENABLE_CUDA=ON.  The
# native addon always lands in `build/Release/` (the path the extension runtime
# and the .vscodeignore whitelist expect), so the GPU and CPU builds share the
# same `build/` tree.  On a machine without nvcc/CUDA, pass --no-cuda (or
# --cpu) to request a CPU-only build instead of failing the configure.
#
# Prerequisites:
#   - Node.js >= 18 and npm
#   - C++20 compiler (gcc >= 13, clang >= 14, or MSVC 2019+)
#   - CMake >= 3.18
#   - vsce (npm install -g @vscode/vsce)
#   - CUDA Toolkit >= 11.0 + nvcc (default GPU build only)
#
# Usage:
#   ./scripts/build.sh [--cuda | --no-cuda | --cpu | --help]
#
#   (default)  GPU build: -DENABLE_CUDA=ON
#   --cuda     Explicitly request the GPU build (same as default)
#   --no-cuda  CPU-only build: -DENABLE_CUDA=OFF
#   --cpu      Alias for --no-cuda
#   --help     Show this help
#
# Examples:
#   ./scripts/build.sh                          # GPU offload build (default)
#   ./scripts/build.sh --no-cuda                # CPU-only build
#   ./scripts/build.sh 2>&1 | tee build.log
#

set -euo pipefail

# ---- CLI parsing ----
CUDA_FLAGS="-DENABLE_CUDA=ON"      # GPU build is the default
BUILD_DIR="build"                  # single build tree (extension expects build/Release)
for arg in "$@"; do
    case "${arg}" in
        --cuda)
            CUDA_FLAGS="-DENABLE_CUDA=ON"
            ;;
        --no-cuda|--cpu)
            CUDA_FLAGS="-DENABLE_CUDA=OFF"
            ;;
        --help|-h)
            # Print the header comment block (lines 2..36) with '#' stripped.
            sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument '${arg}'" >&2
            echo "Usage: ./scripts/build.sh [--cuda | --no-cuda | --cpu | --help]" >&2
            exit 1
            ;;
    esac
done

ROOT_DIR="$(readlink -f "$(dirname "$BASH_SOURCE")/..")"
cd "${ROOT_DIR}"

echo "=== TinyCoder Build ==="
echo "Root: ${ROOT_DIR}"
if [[ "${CUDA_FLAGS}" == "-DENABLE_CUDA=ON" ]]; then
    echo "Mode: GPU offload build (-DENABLE_CUDA=ON)"
else
    echo "Mode: CPU-only build (-DENABLE_CUDA=OFF)"
fi

# ---- 1. Install npm dependencies ----
echo ""
echo "--- Step 1/4: Installing npm dependencies ---"
if [ ! -d "node_modules" ]; then
    npm install
else
    echo "node_modules exists, skipping npm install (remove node_modules to force reinstall)."
fi

# ---- 2. Compile TypeScript ----
echo ""
echo "--- Step 2/4: Compiling TypeScript ---"
npx tsc -p ./ --outDir out

# ---- 3. Build native addon ----
echo ""
echo "--- Step 3/4: Building native C++ addon ---"
# Remove the previous .vsix plugin
rm -f *.vsix

# Clean the build tree to avoid stale CMake cache issues (a stale cache with a
# different generator than the npm lifecycle scripts' `cmake -B build` aborts
# the configure — see the "generator mismatch" CMake error).
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release ${CUDA_FLAGS}
cmake --build build --config Release

# Verify the native addon was produced
NATIVE_ADDON="build/Release/tinycoder_native.node"
if [ ! -f "${NATIVE_ADDON}" ]; then
    echo "ERROR: Native addon not found at ${NATIVE_ADDON}"
    exit 1
fi
echo "Native addon: ${NATIVE_ADDON}"

# ---- 4. Package .vsix ----
echo ""
echo "--- Step 4/4: Packaging VS Code extension ---"
VSIX_OUTPUT="${ROOT_DIR}/tinycoder-$(node -e "console.log(require('./package.json').version)").vsix"
NODE_OPTIONS=--no-deprecation npx vsce package --out "${VSIX_OUTPUT}"

echo ""
echo "=== Build complete ==="
echo "VSIX: ${VSIX_OUTPUT}"
echo "Native addon: ${NATIVE_ADDON}"
