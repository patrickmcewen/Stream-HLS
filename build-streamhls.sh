#!/bin/bash

source ../codesign/miniconda3/etc/profile.d/conda.sh
conda activate streamhls

# If ninja is available, use it.
CMAKE_GENERATOR="Unix Makefiles"
if which ninja &>/dev/null; then
  CMAKE_GENERATOR="Ninja"
fi

mkdir -p build
cd build

LLVM_PRJ_PATH=/scratch/patrick/Stream-HLS/extern/llvm-project

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

# Export PATH and source Xilinx tools
# Note: These exports will only work if script is sourced, not executed
export PATH=$PATH:/scratch/patrick/Stream-HLS/build/bin
export PATH=$PATH:/scratch/patrick/Stream-HLS/ampl.linux-intel64
# Set LD_LIBRARY_PATH for Gurobi solver
export LD_LIBRARY_PATH=/scratch/patrick/Stream-HLS/ampl.linux-intel64:$LD_LIBRARY_PATH

# Source Xilinx HLS settings if available
if [ -f /afs/ece.cmu.edu/support/xilinx/xilinx.release/Vivado-2022.1/Vitis_HLS/2022.1/settings64.sh ]; then
    source /afs/ece.cmu.edu/support/xilinx/xilinx.release/Vivado-2022.1/Vitis_HLS/2022.1/settings64.sh
fi

# Return to original directory (important when sourcing)
cd /scratch/patrick/Stream-HLS