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