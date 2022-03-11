#!/bin/bash

# Fail on any error.
set -e

# Display commands being run.
set -x

module load cuda/10.0
module load cudnn/7.6.2

virtualenv --python=python3.6 venv
source venv/bin/activate 

# Install dependencies.
pip install -r requirements.txt

