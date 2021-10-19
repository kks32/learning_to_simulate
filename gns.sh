cd /work/05873/kks32/frontera
TMP_DIR="/work/05873/kks32/frontera/"
source "${TMP_DIR}/learning_to_simulate/bin/activate"
module load cuda/10.0
module load cudnn/7.6.2
DATASET_NAME="WaterDropSample"
DATA_PATH="${TMP_DIR}/datasets/${DATASET_NAME}"
MODEL_PATH="${TMP_DIR}/models/${DATASET_NAME}"
ROLLOUT_PATH="${TMP_DIR}/rollouts/${DATASET_NAME}"
python -m learning_to_simulate.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --num_steps=120000

