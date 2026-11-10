#!/bin/bash -l
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
# Train KimCNN-static on R52 or WOS-46985 with GloVe or BERT embeddings.
#
# Positional arguments:
#   $1  runname              e.g. kimcnn_r52_glove_20260515
#   $2  dataset              r52 | wos46985
#   $3  embedding_type       glove | bert
#   $4  data_dir             e.g. r52 / wos46985
#   $5  embedding_path       glove_embeddings/glove.840B.300d.txt
#                            OR <dataset>_bert_cache/bert_uncased_512.pt
#   $6  output_dir           checkpoint output directory
#   $7  epochs               (optional, default 200)
#   $8  batch_size           (optional, default 64)
#   $9  lr                   (optional, default 1e-3)
#   $10 weight_decay         (optional, default 0)
#
# Configure the modules for your cluster below.

module load python

source graphic/bin/activate

# Our compute nodes reach the internet via a proxy. Needed only for BERT mode
# (downloads bert-base-uncased on first cache build). Adjust or remove for your
# cluster.
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
# Silence the per-batch "tokenizers parallelism after fork" warnings the
# HuggingFace tokenizer emits inside DataLoader workers. Cosmetic only --
# does not change behavior.
export TOKENIZERS_PARALLELISM=false

runname=$1
dataset=$2
embedding_type=$3
data_dir=$4
embedding_path=$5
output_dir=$6
epochs=${7:-200}
batch_size=${8:-64}
lr=${9:-1e-3}
weight_decay=${10:-0}

mkdir -p "$output_dir"

# A checkpoint every 3 epochs on both datasets, as in the paper.
checkpoint_every=3

if [ "$embedding_type" = "glove" ]; then
    python CNNStandalone/train_kimcnn.py \
        --dataset $dataset \
        --data-dir $data_dir \
        --embedding-type glove \
        --glove-path $embedding_path \
        --embed-dim 300 \
        --max-len 512 \
        --epochs $epochs \
        --batch-size $batch_size \
        --lr $lr \
        --weight-decay $weight_decay \
        --scheduler cosine \
        --checkpoint-every $checkpoint_every \
        --output-dir "$output_dir"
elif [ "$embedding_type" = "bert" ]; then
    python CNNStandalone/train_kimcnn.py \
        --dataset $dataset \
        --data-dir $data_dir \
        --embedding-type bert \
        --bert-cache $embedding_path \
        --max-len 512 \
        --epochs $epochs \
        --batch-size $batch_size \
        --lr $lr \
        --weight-decay $weight_decay \
        --scheduler cosine \
        --checkpoint-every $checkpoint_every \
        --output-dir "$output_dir"
else
    echo "Unknown embedding_type: $embedding_type (expected glove or bert)" >&2
    exit 1
fi
