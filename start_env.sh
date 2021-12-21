#!/bin/bash

module reset

# Start venv
# ----------
module load cuda/10.0
module load cudnn/7.6.2
module load python3 # -> python3/3.7.0

source venv/bin/activate

# Test venv
# ---------
echo "test gpu is available"
python3 tests/tf_gpu.py

# Clean up venv
# -------------
#deactivate
#rm -r venv


