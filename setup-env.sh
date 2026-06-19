#!/bin/bash
# Setup environment for Stream-HLS
# Source this file: source setup-env.sh

# Activate conda environment
if conda env list | grep -q "\\bstreamhls\\b"; then
  conda activate streamhls
else
  echo "The environment 'streamhls' does not exist."
  echo "Creating environment 'streamhls'..."
  conda create -n streamhls python=3.11 -y
  conda activate streamhls
fi

# Always ensure requirements are installed
# Quick check: if torch and torch_mlir are both importable, skip pip install
if python -c "import torch; import torch_mlir" 2>/dev/null; then
  echo "Requirements already installed, skipping pip install..."
else
  echo "Installing/updating requirements..."
  pip install -r requirements.txt
fi

export ROOT_DIR="$(pwd)"
# Add Stream-HLS tools to PATH
export PATH=$PATH:$ROOT_DIR/build/bin
export PATH=$PATH:$ROOT_DIR/ampl.linux-intel64
# Set LD_LIBRARY_PATH for Gurobi solver
export LD_LIBRARY_PATH=$ROOT_DIR/ampl.linux-intel64:$LD_LIBRARY_PATH

# Source Xilinx HLS settings if available
if [ -f /afs/ece.cmu.edu/support/xilinx/xilinx.release/Vivado-2023.2/Vitis_HLS/2023.2/settings64.sh ]; then
    source /afs/ece.cmu.edu/support/xilinx/xilinx.release/Vivado-2023.2/Vitis_HLS/2023.2/settings64.sh
fi

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module load vitis/2022.1
fi

echo "Stream-HLS environment setup complete!"
echo "  - Conda environment: streamhls"
echo "  - AMPL: $(which ampl 2>/dev/null || echo 'not found')"
echo "  - streamhls-opt: $(which streamhls-opt 2>/dev/null || echo 'not found')"
echo "  - streamhls-translate: $(which streamhls-translate 2>/dev/null || echo 'not found')"