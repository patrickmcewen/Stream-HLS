#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

# Script options.
while getopts 'j:p:' opt
do
  case $opt in
    j) JOBS="${OPTARG}";;
    p) PYBIND="${OPTARG}";;
  esac
done

# If ninja is available, use it.
CMAKE_GENERATOR="Unix Makefiles"
if which ninja &>/dev/null; then
  CMAKE_GENERATOR="Ninja"
fi

# The absolute path to the directory of this script.
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Make sure polygeist submodule is up-to-date.
git submodule sync
git submodule update --init --recursive

echo ""
echo ">>> Unified MLIR, Clang, and ScaleHLS build..."
echo ""

# Got to the build directory.
cd "${ROOT_DIR}"
mkdir -p build_poly
cd build_poly

# Configure CMake.
if [ ! -f "CMakeCache.txt" ]; then
  cmake -G "${CMAKE_GENERATOR}" \
    ../Polygeist/llvm-project/llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DMLIR_ENABLE_BINDINGS_PYTHON="${PYBIND:=OFF}" \
    -DSCALEHLS_ENABLE_BINDINGS_PYTHON="${PYBIND:=OFF}" \
    -DLLVM_PARALLEL_LINK_JOBS="${JOBS:=}" \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
fi

# Run building.
if [ "${CMAKE_GENERATOR}" == "Ninja" ]; then
  ninja
else 
  make -j "$(nproc)"
fi

echo ""
echo ">>> Polygeist build..."
echo ""

# Got to the build directory.
cd "${ROOT_DIR}/Polygeist"
mkdir -p build
cd build

# Configure CMake.
if [ ! -f "CMakeCache.txt" ]; then
  cmake -G "${CMAKE_GENERATOR}" \
    .. \
    -DMLIR_DIR="${ROOT_DIR}/build/lib/cmake/mlir" \
    -DCLANG_DIR="${ROOT_DIR}/build/lib/cmake/clang" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
fi

# Run building.
if [ "${CMAKE_GENERATOR}" == "Ninja" ]; then
  ninja
  ninja check-cgeist
else
  make -j "$(nproc)"
  make -j "$(nproc)" check-cgeist
fi