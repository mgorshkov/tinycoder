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
# Prerequisites:
#   - Node.js >= 18 and npm
#   - C++20 compiler (gcc >= 13, clang >= 14, or MSVC 2019+)
#   - CMake >= 3.18
#   - vsce (npm install -g @vscode/vsce)
#
# Usage:
#   ./scripts/build.sh
#
# Examples:
#   ./scripts/build.sh
#   ./scripts/build.sh 2>&1 | tee build.log
#

set -euo pipefail

ROOT_DIR="$(readlink -f "$(dirname "$BASH_SOURCE")/..")"
cd "${ROOT_DIR}"

echo "=== TinyCoder Build ==="
echo "Root: ${ROOT_DIR}"

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

# Clean the build directory to avoid stale CMake cache issues
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release
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
