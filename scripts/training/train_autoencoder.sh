#!/bin/bash
# Stage 1: fine-tune the KL autoencoder with anatomical-consistency (WarpSeg) supervision.
# Run from the repository root: bash scripts/training/train_autoencoder.sh
set -e

# --- Paths (edit these) ---
DATASET_CSV="/path/to/dataset.csv"
AEKL_CKPT="/path/to/pretrained_autoencoder.pth"     # base AE to fine-tune (see BrLP)
SEG_CKPT="/path/to/WarpSeg/Tissue_model.pth"        # WarpSeg tissue-segmentation checkpoint
OUTPUT_DIR="./output/autoencoder"
CACHE_DIR="./cache/ae"

python scripts/training/train_autoencoder.py \
  --dataset_csv "${DATASET_CSV}" \
  --aekl_ckpt   "${AEKL_CKPT}" \
  --seg_ckpt    "${SEG_CKPT}" \
  --output_dir  "${OUTPUT_DIR}" \
  --cache_dir   "${CACHE_DIR}" \
  --n_epochs 5 \
  --batch_size 4 \
  --lr 5e-6 \
  --anatomical_loss_weight 1.0
