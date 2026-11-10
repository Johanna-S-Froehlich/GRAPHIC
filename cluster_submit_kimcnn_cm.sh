#!/bin/bash -l

# Submit per-checkpoint confusion-matrix extraction for trained KimCNN runs.
# Iterates over (dataset, embedding-type) x alpha. Adjust the email address
# below and the partition/GPU settings to your cluster before submitting.

mkdir -p logs

timestamp=$(date +%Y%m%d%H%M%S)
email_address='<YOUR_EMAIL_ADDRESS>'

# Each entry: "dataset|embedding_type|data_dir|embedding_path|checkpoint_dir|cm_output_dir"
# Same GloVe path as in cluster_submit_kimcnn.sh.
glove_path="${WORK:-$PWD}/glove_embeddings/glove.840B.300d.txt"
configs=(
    "r52|glove|r52|${glove_path}|kimcnn_r52_glove|cm_kimcnn_r52_glove"
    "r52|bert|r52|r52_bert_cache/bert_uncased_512.pt|kimcnn_r52_bert|cm_kimcnn_r52_bert"
    "wos46985|glove|wos46985|${glove_path}|kimcnn_wos46985_glove|cm_kimcnn_wos46985_glove"
    "wos46985|bert|wos46985|wos46985_bert_cache/bert_uncased_512.pt|kimcnn_wos46985_bert|cm_kimcnn_wos46985_bert"
)

alphas=(0.0 1.0)

# LC repetitions per (config, alpha) job, run sequentially inside the job.
# 5 iterations x ~3h30 worst case (WoS-BERT) fits the 20h walltime below.
iterations="0 1 2 3 4"

for cfg in "${configs[@]}"; do
    IFS='|' read -r dataset emb data_dir emb_path ckpt_dir cm_dir <<< "$cfg"
    for alpha in "${alphas[@]}"; do
        runname="cm_kimcnn_${dataset}_${emb}_alpha${alpha}_${timestamp}"
        sbatch \
            --job-name="$runname" \
            --output="logs/${runname}.log" \
            --mail-user="$email_address" \
            --mail-type=ALL \
            --partition=a100 \
            --gres=gpu:a100:1 \
            --time=20:00:00 \
            cluster_run_kimcnn_cm.sh \
            "$dataset" "$emb" "$data_dir" "$emb_path" "$ckpt_dir" "$cm_dir" "$alpha" \
            0 100 1 "$iterations"
    done
done
