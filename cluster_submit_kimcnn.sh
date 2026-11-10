#!/bin/bash -l

# Submit one KimCNN training job per (dataset, embedding-type) combo.
# Adjust the email address below and the partition/GPU settings to your
# cluster before submitting.

mkdir -p logs

timestamp=$(date +%Y%m%d%H%M%S)
email_address='<YOUR_EMAIL_ADDRESS>'

# Each entry: "dataset|embedding_type|data_dir|embedding_path|output_dir|epochs|lr|weight_decay"
#
# The hyperparameters of the paper. Epochs are dataset-specific based on
# observed convergence under cosine LR; no early stopping, the last checkpoint
# is the trained model. R52 BERT uses lr=3e-4 + wd=1e-4 to dampen the
# overfitting seen at lr=1e-3 once the training loss bottoms out.
# GloVe vectors are read from $WORK/glove_embeddings, or from glove_embeddings/
# in the repository if $WORK is not set (glove.840B.300d.txt is 5.6 GB); adjust
# the path if yours are elsewhere. BERT caches build into the repo-local
# *_bert_cache/ dirs on first run (about 7 GB for R52, 37 GB for WOS-46985).
glove_path="${WORK:-$PWD}/glove_embeddings/glove.840B.300d.txt"
configs=(
    "r52|glove|r52|${glove_path}|kimcnn_r52_glove|50|1e-3|0"
    "r52|bert|r52|r52_bert_cache/bert_uncased_512.pt|kimcnn_r52_bert|50|3e-4|1e-4"
    "wos46985|glove|wos46985|${glove_path}|kimcnn_wos46985_glove|100|1e-3|0"
    "wos46985|bert|wos46985|wos46985_bert_cache/bert_uncased_512.pt|kimcnn_wos46985_bert|100|1e-3|0"
)

for cfg in "${configs[@]}"; do
    IFS='|' read -r dataset emb data_dir emb_path out_dir epochs lr wd <<< "$cfg"
    runname="kimcnn_${dataset}_${emb}_${timestamp}"
    sbatch \
        --job-name="$runname" \
        --output="logs/${runname}.log" \
        --mail-user="$email_address" \
        --mail-type=ALL \
        --partition=a100 \
        --gres=gpu:a100:1 \
        cluster_run_kimcnn.sh \
        "$runname" "$dataset" "$emb" "$data_dir" "$emb_path" "$out_dir" \
        "$epochs" 64 "$lr" "$wd"
done
