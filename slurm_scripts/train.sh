#!/bin/bash

#SBATCH -J tf_train         # Job name
#SBATCH -o tf_train.o%j     # Name of stdout output file
#SBATCH -e tf_train.e%j     # Name of stderr error file
#SBATCH -p rtx               # Queue (partition) name
#SBATCH -N 1                 # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH --mail-user=jvantassel@tacc.utexas.edu
#SBATCH -A BCS20003          # Project/Allocation name (req'd if you have more than 1)

# fail on error
set -e

# start in slurm_scripts
cd ..
source start_venv.sh

cd ..

# assume data is already downloaded and hardcode WaterDropSample
data="Sand"
DATA_PATH="${WORK}/gns_tensorflow/${data}/dataset"
MODEL_PATH="${WORK}/gns_tensorflow/${data}/models"

python3 -m learning_to_simulate.train \
--data_path=${DATA_PATH} \
--model_path=${MODEL_PATH} \
--num_steps="1000000"

