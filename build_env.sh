#!/bin/bash

module reset

# Setup venv
# ----------
#module load intel/19.1.1
#module unload impi
module load python3/3.7
module load cuda/10.0
module load cudnn/7.6.2
module load nccl/2.4.7

python3 -m virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Test venv
# ---------
echo "test gpu is available"
python3 tests/tf_gpu.py

# Clean up venv
# -------------
deactivate
#rm -r venv

