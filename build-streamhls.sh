#!/usr/bin/env bash

# If ninja is available, use it.
CMAKE_GENERATOR="Unix Makefiles"
if which ninja &>/dev/null; then
  CMAKE_GENERATOR="Ninja"
fi

echo "Building LLVM..."

# The absolute path to the directory of this script.
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Got to the build directory.
cd "${ROOT_DIR}"
mkdir -p extern/llvm-project/build
cd extern/llvm-project/build

# Configure CMake.
if [ ! -f "CMakeCache.txt" ]; then
  cmake -G "${CMAKE_GENERATOR}" \
    ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON
cmake --build . --target check-mlir
fi

# Run building.
if [ "${CMAKE_GENERATOR}" == "Ninja" ]; then
  ninja
else 
  make -j "$(nproc)"
fi
echo "LLVM built successfully."

source setup-env.sh

echo "Building Stream-HLS..."
cd "${ROOT_DIR}"
mkdir -p build
cd build

LLVM_PRJ_PATH="${ROOT_DIR}/extern/llvm-project/build"

# Check if the LLVM and MLIR paths are set.
if [ -z "${LLVM_PRJ_PATH}" ]; then
  echo "Error: LLVM project path is not set."
  echo "Usage: ./build-streamhls.sh <LLVM_PROJECT_PATH>"
  echo "Example: ./build-streamhls.sh /path/to/llvm-project"
  exit 1
fi
# check if path exists
if [ ! -d "${LLVM_PRJ_PATH}" ]; then
  echo "Error: LLVM project path does not exist."
  echo "Make sure the path is correct."
  exit 1
fi

echo ""
echo "Building Stream-HLS..."
echo ""

cmake -G "${CMAKE_GENERATOR}" .. \
  -DMLIR_DIR=${LLVM_PRJ_PATH}/build/lib/cmake/mlir \
  -DLLVM_DIR=${LLVM_PRJ_PATH}/build/lib/cmake/llvm \

# Run building.
if [ "${CMAKE_GENERATOR}" == "Ninja" ]; then
  ninja
else
  make
fi

echo "Stream-HLS built successfully."

cd "${ROOT_DIR}"