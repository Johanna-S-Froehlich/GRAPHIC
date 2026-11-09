#!/bin/bash -l

mkdir logs

timestamp=$(date +%Y%m%d%H%M%S)

lr=0.003
epochs=71
batchsize=1024
model=resnet50
optimizer=adam
scheduler=plateau
weight_decay=0.4e-04
autoaugment=true
prefix="supervised_plateau"

runname="resnet_training_${timestamp}"

sbatch --job-name=$runname --output=logs/$runname.log --mail-user='<YOUR_EMAIL_ADDRESS>' --mail-type=ALL cluster_run_training.sh $runname $lr $epochs $batchsize $model $optimizer $scheduler $autoaugment $weight_decay
