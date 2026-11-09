#!/bin/bash -l
#SBATCH --time=08:30:00
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100


# Prepare the environment 
source graphic/bin/activate

# Constants/Arguments
outdir=resnet_checkpoints
runname=$1
lr=$2
epochs=$3
batchsize=$4
model=$5
optimizer=$6
scheduler=$7
autoaugment=$8
weight_decay=${9}

# Check if autoaugment is true
if [ "$autoaugment" = true ]; then
    autoaugment="--autoaugment"
else
    autoaugment=""
fi

mkdir -p $outdir

# Run training
python train.py --lr $lr --epochs $epochs --batch_size $batchsize --outdir $outdir --runname $runname --gradclip 0.1 --small_conv1 --model $model --optimizer $optimizer --scheduler $scheduler $autoaugment --weight_decay $weight_decay
