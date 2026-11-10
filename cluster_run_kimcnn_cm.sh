#!/bin/bash -l
#SBATCH --time=03:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100

# Build per-layer confusion matrices for trained KimCNN checkpoints.
#
# Positional arguments:
#   $1  dataset              r52 | wos46985
#   $2  embedding_type       glove | bert
#   $3  data_dir             e.g. r52 / wos46985
#   $4  embedding_path       glove .txt or bert cache .pt
#   $5  checkpoint_dir       directory of trained KimCNN .pt files
#   $6  output_dir           where the CMs are written
#   $7  alpha                CE mix; 0 = pred-labels only, 1 = true only
#   $8  start_index          (optional, default 0)
#   $9  end_index            (optional, default 100)
#  $10  epoch_step           (optional, default 1)
#  $11  iterations           (optional, space-separated LC repetition
#                             indices, default "0"; e.g. "0 1 2 3 4")

module load python

source graphic/bin/activate

# Same proxy export as cluster_run_kimcnn.sh; CM extraction reuses the BERT
# cache built during training, but the HF model resolution still calls out for
# the tokenizer config.
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export PYTHONUNBUFFERED=1

dataset=$1
embedding_type=$2
data_dir=$3
embedding_path=$4
checkpoint_dir=$5
output_dir=$6
alpha=$7
start_index=${8:-0}
end_index=${9:-100}
epoch_step=${10:-1}
iterations=${11:-0}

mkdir -p "$output_dir"

# Dataset-specific LC hyperparameters of the paper:
#   R52      -> lr=0.01,  wd=1e-3, 100 cosine epochs
#   WoS46985 -> lr=0.003, wd=1e-3, 100 cosine epochs
case "$dataset" in
    r52)        lc_lr=0.01  ;;
    wos46985)   lc_lr=0.003 ;;
    *)          lc_lr=0.01  ;;
esac
lc_wd=1e-3
lc_epochs=100

for iteration in $iterations; do
if [ "$embedding_type" = "glove" ]; then
    python confusion_matrices_cnn.py \
        --dataset $dataset \
        --data-dir $data_dir \
        --embedding-type glove \
        --glove-path $embedding_path \
        --embed-dim 300 \
        --max-len 512 \
        --checkpoint-dir $checkpoint_dir \
        --output-dir $output_dir \
        --alpha $alpha \
        --iteration $iteration \
        --start-index $start_index \
        --end-index $end_index \
        --epoch-step $epoch_step \
        --lc-lr $lc_lr --weight-decay $lc_wd --lc-epochs $lc_epochs \
        --save-confusions
elif [ "$embedding_type" = "bert" ]; then
    python confusion_matrices_cnn.py \
        --dataset $dataset \
        --data-dir $data_dir \
        --embedding-type bert \
        --bert-cache $embedding_path \
        --max-len 512 \
        --checkpoint-dir $checkpoint_dir \
        --output-dir $output_dir \
        --alpha $alpha \
        --iteration $iteration \
        --start-index $start_index \
        --end-index $end_index \
        --epoch-step $epoch_step \
        --lc-lr $lc_lr --weight-decay $lc_wd --lc-epochs $lc_epochs \
        --save-confusions
else
    echo "Unknown embedding_type: $embedding_type" >&2
    exit 1
fi
done
