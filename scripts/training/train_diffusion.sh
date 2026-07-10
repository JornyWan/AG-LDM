#!/bin/bash
# Stage 2: train the channel-conditioned latent diffusion model with the WM/GM volume loss.
# Run from the repository root: bash scripts/training/train_diffusion.sh
set -e

# --- Paths (edit these) ---
DATASET_CSV="/path/to/dataset.csv"
AEKL_CKPT="/path/to/finetuned_autoencoder.pth"      # Stage-1 output
SEG_CKPT="/path/to/WarpSeg/Tissue_model.pth"        # WarpSeg tissue-segmentation checkpoint
OUTPUT_DIR="./output/diffusion"
CACHE_DIR="./cache/diffusion"

# --- Hyperparameters (paper settings) ---
LR=2.5e-5
N_EPOCHS=20
BATCH_SIZE=8
VOLUME_LOSS_WEIGHT=1e-5
VOL_LOSS_FREQ=1
VOL_LOSS_START_STEP=10

python scripts/training/train_diffusion.py \
  --dataset_csv "${DATASET_CSV}" \
  --aekl_ckpt   "${AEKL_CKPT}" \
  --seg_ckpt    "${SEG_CKPT}" \
  --output_dir  "${OUTPUT_DIR}" \
  --cache_dir   "${CACHE_DIR}" \
  --n_epochs "${N_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --use_volume_loss \
  --volume_loss_weight "${VOLUME_LOSS_WEIGHT}" \
  --vol_loss_freq "${VOL_LOSS_FREQ}" \
  --vol_loss_start_step "${VOL_LOSS_START_STEP}"
