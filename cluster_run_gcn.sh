#!/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
# WoS46985's adjacency (nnz ~ 4.2e7) overflows the GPU SparseTensorDenseMatMul
# kernel's int32 index when cols * nnz > 2^31. train.py splits the dense operand
# along columns so each piece fits.

# End-to-end Text-GCN pipeline for one dataset: prepare data, train the GCN
# (TF2 venv), then build per-epoch confusion matrices for each alpha on the
# saved activations (PyTorch venv). One GCN training run feeds all alphas --
# the LC head trained inside confusion_matrices_gcn.py is what alpha tunes,
# not the GCN itself.
#
# Activations land in a runname-scoped directory so earlier runs are not
# overwritten.
#
# Positional arguments:
#   $1  runname          e.g. gcn_R52_20260520223909
#   $2  dataset          R52 | WOS46985
#   $3  output_dir       directory to save CMs to (alpha subdirs are created)
#   $4  alphas           space-separated alpha values, e.g. "0.0 1.0"
#   $5  epoch_step       (optional, default 5)
#   $6  start_epoch      (optional, default 0)
#   $7  end_epoch        (optional, default 151: includes the final epoch_<EPOCHS> snapshot)
#   $8  hidden1          (optional, dataset default if empty: R52->200, WoS->536)
#   $9  learning_rate    (optional, dataset default if empty: R52->0.02, WoS->0.08)
#   $10 weight_decay     (optional, train.py default if empty: 0; Yao uses 5e-4)
#
# Env vars:
#   SKIP_TRAIN=1   skip step 2 and reuse the activations of an earlier run.
#   SKIP_CMS=1     skip step 3 (CM generation).
#   NUM_LAYERS=N   set GCN depth (train.py default = 2 = Yao).
#   EPOCHS=N       total GCN training epochs (train.py default = 150).
#   LC_ITERATIONS  space-separated LC repetition indices (default "0").

runname=$1
dataset=$2
output_dir=$3
# CM outputs include up to 134 per-class text dumps per (checkpoint, alpha)
# under --save-confusions; for WoS46985 that fills $HOME quickly. Resolve
# the output_dir against $WORK by default, mirroring GCN_ACTIVATIONS_BASE.
# Skip the prefix if the caller already passed an absolute path.
if [[ "$output_dir" != /* ]]; then
    if [[ -z "${GCN_CM_BASE:-}" ]]; then
        if [[ -n "${WORK:-}" ]]; then
            GCN_CM_BASE="$WORK/gcn_cm"
        else
            GCN_CM_BASE="$HOME/gcn_cm"
        fi
    fi
    mkdir -p "$GCN_CM_BASE"
    output_dir="${GCN_CM_BASE}/${output_dir}"
fi
mkdir -p "$output_dir"
alphas=$4
epoch_step=${5:-5}
start_epoch=${6:-0}
end_epoch=${7:-151}
hidden1=${8:-}
learning_rate=${9:-}
weight_decay=${10:-}

# Per-dataset defaults (the hyperparameters of the paper): R52 hidden 200 /
# lr 0.02 (Yao's reference), WoS hidden 536 (= 4x num_classes) / lr 0.08.
# Explicit $8/$9 arguments override them.
case "$dataset" in
    R52)       hidden1=${hidden1:-200} ; learning_rate=${learning_rate:-0.02} ;;
    WOS46985)  hidden1=${hidden1:-536} ; learning_rate=${learning_rate:-0.08} ;;
    *)         hidden1=${hidden1:-200} ; learning_rate=${learning_rate:-0.02} ;;
esac

# Activations are bulky (~1-20 GB per run depending on hidden1).
# Resolution order: explicit override -> $WORK -> $HOME.
if [[ -z "${GCN_ACTIVATIONS_BASE:-}" ]]; then
    if [[ -n "${WORK:-}" ]]; then
        GCN_ACTIVATIONS_BASE="$WORK/gcn_activations"
    else
        GCN_ACTIVATIONS_BASE="$HOME/gcn_activations"
    fi
fi
mkdir -p "$GCN_ACTIVATIONS_BASE"
activations_dir="${GCN_ACTIVATIONS_BASE}/${runname}"

# Yao's Text GCN with our patch (see GCNStandalone/README.md). Unpatched, Yao's
# code would silently build the graph with a validation split and train on the CPU.
if ! grep -qs cosine_lr GCNStandalone/text_gcn_tf2/train.py; then
    echo "GCNStandalone/text_gcn_tf2 is missing or not patched, see GCNStandalone/README.md" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Prepare dataset in Yao's expected format (lightweight; PyTorch venv).
# ---------------------------------------------------------------------------
module load python
source graphic/bin/activate

python GCNStandalone/prepare_for_gcn.py --dataset $dataset

deactivate
module unload python

# ---------------------------------------------------------------------------
# 2. Train the Text-GCN under the TensorFlow 2 venv.
# Skipped when SKIP_TRAIN=1, in which case we expect $activations_dir to
# already contain a previous training run's outputs (useful for re-running
# only the CM stage on existing activations, e.g. after re-tuning the LC).
# ---------------------------------------------------------------------------
if [[ "${SKIP_TRAIN:-}" == "1" ]]; then
    if [[ ! -f "$activations_dir/${dataset}_shuffle.txt" ]]; then
        echo "SKIP_TRAIN=1 but $activations_dir has no shuffle file; aborting." >&2
        exit 1
    fi
    echo "SKIP_TRAIN=1 -- reusing existing activations at $activations_dir"
else

module load python
source GCNStandalone/text_gcn/bin/activate

# tensorflow[and-cuda] installs the CUDA libs to site-packages/nvidia/*/lib
# but doesn't prep LD_LIBRARY_PATH, so TF's runtime dlopen can't find them
# on nodes without a matching system CUDA. Glob the bundled lib dirs and
# prepend them. Also unbuffer python so log progress streams in real time
# (compute-node SLURM log is a file, not a TTY -> full buffering otherwise).
nv_libs=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
export LD_LIBRARY_PATH="$(ls -d $nv_libs/*/lib 2>/dev/null | paste -sd: -)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

# Our compute nodes reach the internet via a proxy. Needed for nltk's download
# of the stopwords corpus in remove_words.py; recent nltk versions only download
# through a proxy when NLTK_ALLOW_PROXIED_URLOPEN is set. Adjust or remove for
# your cluster.
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export NLTK_ALLOW_PROXIED_URLOPEN=1
# build_graph.py builds the vocabulary from a set, whose order follows Python's
# string hashing. A fixed hash seed makes the graph reproducible between runs.
export PYTHONHASHSEED=0

cd GCNStandalone/text_gcn_tf2
# The jobs of both datasets may start at once and share the nltk data folder.
( flock -x 200; python -c "import nltk; nltk.download('stopwords')" ) 200>"$HOME/.gcn_prep_nltk.lock"
# Serialize data prep across concurrent jobs on the same dataset. The
# pipeline writes shared `data/ind.${dataset}.*` pickle files; without a
# lock, two jobs starting at the same instant clobber each other (we hit
# UnpicklingError and ZeroDivisionError on WoS46985 production runs).
# First job through does the prep; subsequent jobs block on the lock,
# then skip when they see ind.<dataset>.x already exists.
# Lockfile on $HOME (shared filesystem; /tmp would be node-local).
lock="$HOME/.gcn_prep_${dataset}.lock"
(
    flock -x 200
    if [[ ! -f "data/ind.${dataset}.x" ]]; then
        python remove_words.py "$dataset" || exit 1
        python build_graph.py  "$dataset" || exit 1
    fi
) 200>"$lock" || { echo "data preparation for $dataset failed" >&2; exit 1; }
# train.py auto-sizes --sparse_matmul_split from the real adjacency nnz
# and hidden1 -- the GPU SparseTensorDenseMatMul kernel rejects
# cols * nnz >= 2^31, so split must grow with hidden1 on WoS46985.
lr_flag=""
if [[ -n "$learning_rate" ]]; then lr_flag="--learning_rate $learning_rate"; fi
wd_flag=""
if [[ -n "$weight_decay" ]]; then wd_flag="--weight_decay $weight_decay"; fi
nl_flag=""
if [[ -n "${NUM_LAYERS:-}" ]]; then nl_flag="--num_layers $NUM_LAYERS"; fi
ep_flag=""
if [[ -n "${EPOCHS:-}" ]]; then ep_flag="--epochs $EPOCHS"; fi
python train.py $dataset --activations_dir "$activations_dir" \
    --hidden1 "$hidden1" $lr_flag $wd_flag $nl_flag $ep_flag

cd ../..
deactivate
module unload python

fi  # end of SKIP_TRAIN guard

# ---------------------------------------------------------------------------
# 3. Build CMs (one per alpha) on the saved activations under PyTorch venv.
# Skipped when SKIP_CMS=1.
# ---------------------------------------------------------------------------
if [[ "${SKIP_CMS:-}" == "1" ]]; then
    echo "SKIP_CMS=1 -- skipping CM generation; activations remain at $activations_dir"
    exit 0
fi

module load python
source graphic/bin/activate

# Dataset-specific LC hyperparameters of the paper:
#   R52       -> lr=0.3,  wd=0, 100 cosine epochs
#   WOS46985  -> lr=0.01, wd=0, 100 cosine epochs
case "$dataset" in
    R52)        lc_lr=0.3  ;;
    WOS46985)   lc_lr=0.01 ;;
    *)          lc_lr=0.3  ;;
esac
lc_wd=0
lc_epochs=100

# LC_ITERATIONS: space-separated repetition indices (default just the
# canonical iteration 0). E.g. LC_ITERATIONS="0 1 2 3 4" for the 5-run
# mean/std bands; budget the walltime accordingly (the CM stage scales
# linearly with the iteration count).
for alpha in $alphas; do
    for iteration in ${LC_ITERATIONS:-0}; do
        python confusion_matrices_gcn.py \
            --dataset $dataset \
            --activations-dir "$activations_dir" \
            --corpus-path GCNStandalone/text_gcn_tf2/data/corpus/$dataset.txt \
            --meta-path GCNStandalone/text_gcn_tf2/data/$dataset.txt \
            --output-dir $output_dir \
            --alpha $alpha \
            --iteration $iteration \
            --epoch-step $epoch_step \
            --start-epoch $start_epoch \
            --end-epoch $end_epoch \
            --lc-lr $lc_lr --weight-decay $lc_wd --lc-epochs $lc_epochs \
            --save-confusions
    done
done
