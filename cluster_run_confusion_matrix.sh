#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100

# Prepare the environment
source graphic/bin/activate

# Constants/Arguments
checkpoints_dir=resnet_checkpoints
output_dir=resnet_conf_matrices
model=resnet50

# Run confusion matrix
python cifar-100-confusion-matrices.py --checkpoint_dir $checkpoints_dir --output_dir $output_dir --model $model --small_conv1 --only layer1 layer2 layer3 layer4 --start_index 1 --end_index 72 --epoch_step 5 --alpha 0.0 --iterations 1 --weight_decay

