#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=24:00:00

source train_vit/bin/activate

python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 53533 --use_env main.py --model deit_small_patch16_224 --batch-size 64 --data-path ../tiny-imagenet-200 --data-set TINYIMNET --output_dir ../vit_checkpoints_tiny --lr 1e-3 --localvit --localvit-act 'hs' --init-size 32 --epoch-step 5 --patch-step 2 --eval-on-final-size --epochs 340 --depth 12

