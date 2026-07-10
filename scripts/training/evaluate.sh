#!/bin/bash
# Evaluation: sample follow-up MRIs and compute MSE, tissue Dice, ROI volume errors, WASABI.
# Run from the repository root: bash scripts/training/evaluate.sh
# Requires FreeSurfer (mri_synthseg) on PATH for the ROI metrics, and POT for WASABI.
set -e

# --- Paths (edit these) ---
DATASET_CSV="/path/to/dataset.csv"
AEKL_CKPT="/path/to/finetuned_autoencoder.pth"
DIFF_CKPT="/path/to/diffusion.pth"
SEG_CKPT="/path/to/WarpSeg/Tissue_model.pth"
OUTPUT_DIR="./output/eval"
SYNTHSEG_CODE_MAP="scripts/synthseg_code_map.json"

python scripts/training/evaluate.py \
  --dataset_csv       "${DATASET_CSV}" \
  --aekl_ckpt         "${AEKL_CKPT}" \
  --diff_ckpt         "${DIFF_CKPT}" \
  --segment_ckpt      "${SEG_CKPT}" \
  --synthseg_code_map "${SYNTHSEG_CODE_MAP}" \
  --output_dir        "${OUTPUT_DIR}" \
  --batch_size 1 \
  --use_mask \
  --roi_list all \
  --wasabi_k 1000
