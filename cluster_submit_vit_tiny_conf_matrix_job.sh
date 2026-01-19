#!/bin/bash -l

timestamp=$(date +%Y%m%d%H%M%S)
mkdir logs

depth=12
email_address='<YOUR_MAIL_ADDRESS>'

alpha=0.0
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_1.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit_tiny.sh $alpha 0 10 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_2.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit_tiny.sh $alpha 10 20 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_3.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit_tiny.sh $alpha 20 30 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_4.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit_tiny.sh $alpha 30 40 $depth

alpha=1.0
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_1.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit_tiny.sh $alpha 0 10 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_2.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit_tiny.sh $alpha 10 20 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_3.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit_tiny.sh $alpha 20 30 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_4.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit_tiny.sh $alpha 30 40 $depth