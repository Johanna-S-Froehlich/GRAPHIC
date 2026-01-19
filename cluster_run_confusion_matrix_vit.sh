#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100

# Prepare the environment
source graphic/bin/activate

# Constants/Arguments
model=efficient-vit
alpha=${1}
start_index=${2}
end_index=${3}
depth=${4}
checkpoints_dir=vit_checkpoints
output_dir=vit_conf_matrices

# Run confusion matrix
python conf_matrices_vit.py --checkpoint_dir $checkpoints_dir --output_dir $output_dir --model $model --start_index $start_index --end_index $end_index --alpha "$alpha" --num_classes 100 --depth "$depth" --weight_decay

