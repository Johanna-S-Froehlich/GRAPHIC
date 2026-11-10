#!/bin/bash -l

# Submit the full Text-GCN pipeline for each dataset and alpha.
# Adjust the email address below and the partition/GPU settings to your
# cluster before submitting.

mkdir -p logs

timestamp=$(date +%Y%m%d%H%M%S)
email_address='<YOUR_EMAIL_ADDRESS>'

# One GCN training run per dataset. The CM stage trains a fresh LC head per
# alpha against the same activations; nothing about the GCN itself depends on
# alpha.
datasets=("R52" "WOS46985")
alphas="0.0 1.0"

# Five LC repetitions for the mean +/- std bands; iteration 0 is the
# canonical run (graphs, confusion dumps). Propagates to
# cluster_run_gcn.sh via sbatch's default --export=ALL.
export LC_ITERATIONS="0 1 2 3 4"

for dataset in "${datasets[@]}"; do
    runname="gcn_${dataset}_${timestamp}"
    # --time overrides the run script's 4h header: the CM stage runs
    # 2 alphas x 5 LC iterations (10 driver invocations). --export=ALL
    # re-asserts env propagation so the exported LC_ITERATIONS reaches the
    # job even if the site config changes sbatch's default export behaviour.
    sbatch \
        --job-name="$runname" \
        --output="logs/${runname}.log" \
        --mail-user="$email_address" \
        --mail-type=ALL \
        --partition=a100 \
        --gres=gpu:a100:1 \
        --time=20:00:00 \
        --export=ALL \
        cluster_run_gcn.sh \
        "$runname" "$dataset" "cm_gcn_${dataset}_${timestamp}" "$alphas"
done
