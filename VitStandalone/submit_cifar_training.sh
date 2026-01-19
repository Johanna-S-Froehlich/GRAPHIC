#!/bin/bash -l

mkdir -p logs

timestamp=$(date +%Y%m%d%H%M%S)

runname="vit_training_${timestamp}"

sbatch --job-name=$runname --output=logs/$runname.log --mail-user='<YOUR_EMAIL_ADDRESS>' --mail-type=ALL varsize_1gpu_best_cifar.sh
