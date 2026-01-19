#!/bin/bash -l

timestamp=$(date +%Y%m%d%H%M%S)
mkdir logs

sbatch --job-name="create_confusion_matrices_resnet_${timestamp}" --output=logs/create_confusion_matrices_${timestamp}.log --mail-user='<YOUR_EMAIL_ADDRESS>' --mail-type=ALL cluster_run_confusion_matrix.sh
