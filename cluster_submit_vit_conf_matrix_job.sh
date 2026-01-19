#!/bin/bash -l

timestamp=$(date +%Y%m%d%H%M%S)
mkdir logs

depth=12
email_address='<YOUR_EMAIL_ADDRESS>'

alpha=0.0
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_1.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 0 10 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_2.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 10 20 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_3.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 20 30 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_4.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 30 40 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_5.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 40 50 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_6.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 50 60 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_7.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 60 70 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_8.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 70 80 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_9.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 80 90 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_A.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 90 100 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_B.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 100 110 $depth

alpha=1.0
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_1.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 0 10 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_2.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 10 20 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_3.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 20 30 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_4.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 30 40 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_5.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 40 50 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_6.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 50 60 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_7.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 60 70 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_8.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 70 80 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_9.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 80 90 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_A.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 90 100 $depth
sbatch --job-name="conf_mat_vit_${alpha}_${timestamp}" --output=logs/conf_mat_vit_${alpha}_${timestamp}_B.log --mail-user=$email_address --mail-type=ALL cluster_run_confusion_matrix_vit.sh $alpha 100 110 $depth
