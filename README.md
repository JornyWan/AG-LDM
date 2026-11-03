# AG-LDM: Anatomically Guided Latent Diffusion for Brain MRI Progression Modeling

Training and evaluation code for **AG-LDM**, a segmentation-guided latent diffusion model
for longitudinal brain-MRI progression. The framework and preprocessing build on
[BrLP](https://github.com/LemuelPuglisi/BrLP); anatomical supervision comes from the frozen
**WarpSeg** tissue segmenter.

## ⭐️ We will have an updated version coming soon~

## Structure

```
AG-LDM/
├── src/brlp/                     # framework, adapted from BrLP
├── scripts/
│   ├── synthseg_code_map.json
│   └── training/
│       ├── train_autoencoder.py  train_diffusion.py  evaluate.py
│       ├── *.sh                  # example launch wrappers
│       ├── segment_utils/        # WarpSeg wrapper only (model files not shipped)
│       └── wasabi_utils/         # WASABI metric
├── pyproject.toml  requirements.txt  LICENSE
```

Two stages: (1) fine-tune a MONAI `AutoencoderKL` with a spectral-norm patch discriminator
and a WarpSeg anatomical-consistency loss; (2) train a channel-conditioned
`DiffusionModelUNet` that predicts the follow-up latent from the starting latent plus five
covariate channels, with an optional WM/GM soft-Dice volume loss. `evaluate.py` reports
MSE, tissue Dice, SynthSeg ROI volume errors, and the WASABI metric.

## Installation

```bash
conda create -n agldm python=3.10 -y && conda activate agldm
pip install -e .          # or: pip install -r requirements.txt
```

Needs `torch>=2`, `monai==1.3.*`, `monai-generative==0.2.3`, `einops`, `nibabel`, `POT`,
`wandb`. Run all scripts from the repository root.

## Data

Preprocessing follows [BrLP](https://github.com/LemuelPuglisi/BrLP) (skull-strip, MNI
registration, 1.5 mm resample to `122x146x122`). Each script takes a `--dataset_csv` of
paired baseline/follow-up records (image + latent paths, covariates, `split` column). The
Stage-2 volume loss also reads `<followup_dir>/repreprocess/resampled.nii.gz`.

## Segmentation teacher (WarpSeg)

This repo ships only the wrapper (`segment_utils/segmentor_module.py`). Obtain the model
definition (`mga_net.py`, `mga_net_updated.py`) and checkpoint from
[WarpSeg](https://github.com/BahramJafrasteh/WarpSeg), place the model files in
`scripts/training/segment_utils/`, and pass the checkpoint via `--seg_ckpt` /
`--segment_ckpt`.

## Usage

Run from the repository root (edit the paths in each `.sh`, or call the scripts directly).

```bash
# Stage 1 — autoencoder
python scripts/training/train_autoencoder.py --dataset_csv DATA.csv \
  --aekl_ckpt PRETRAINED_AE.pth --seg_ckpt WarpSeg/Tissue_model.pth \
  --output_dir ./output/ae --n_epochs 5 --batch_size 4 --lr 5e-6 --anatomical_loss_weight 1.0

# Stage 2 — diffusion
python scripts/training/train_diffusion.py --dataset_csv DATA.csv \
  --aekl_ckpt FINETUNED_AE.pth --seg_ckpt WarpSeg/Tissue_model.pth \
  --output_dir ./output/diff --n_epochs 20 --batch_size 8 --lr 2.5e-5 \
  --use_volume_loss --volume_loss_weight 1e-5 --vol_loss_freq 1 --vol_loss_start_step 10

# Evaluation
python scripts/training/evaluate.py --dataset_csv DATA.csv \
  --aekl_ckpt FINETUNED_AE.pth --diff_ckpt DIFF.pth --segment_ckpt WarpSeg/Tissue_model.pth \
  --synthseg_code_map scripts/synthseg_code_map.json --output_dir ./output/eval \
  --batch_size 1 --use_mask --roi_list all --wasabi_k 1000
```

Evaluation ROI metrics call FreeSurfer `mri_synthseg` (must be on `PATH`); WASABI needs
`POT`.

## Pretrained weights

The AG-LDM checkpoints used for the reported results are on Hugging Face:
[**Jorny/AG-LDM**](https://huggingface.co/Jorny/AG-LDM).

| File | Description | Params |
|---|---|---|
| `checkpoints/ag-ldm_stage1_autoencoder.pth` | Stage-1 fine-tuned KL autoencoder | 13.77 M |
| `checkpoints/ag-ldm_stage2_diffusion.pth` | Stage-2 channel-conditioned UNet | 475.44 M |

```bash
hf download Jorny/AG-LDM --local-dir ./AG-LDM-weights
```

```python
from huggingface_hub import hf_hub_download
from brlp import networks

ae   = networks.init_autoencoder(hf_hub_download('Jorny/AG-LDM', 'checkpoints/ag-ldm_stage1_autoencoder.pth'))
diff = networks.init_latent_diffusion_channel_cond(hf_hub_download('Jorny/AG-LDM', 'checkpoints/ag-ldm_stage2_diffusion.pth'))
```

Pass them to the scripts above as `--aekl_ckpt` and `--diff_ckpt`. The WarpSeg teacher is
needed only for training, not for inference with these weights.

## Acknowledgments

- **BrLP** — framework and preprocessing. https://github.com/LemuelPuglisi/BrLP
- **MONAI** / **MONAI Generative Models** — networks, schedulers, transforms.
- **WarpSeg** — tissue-segmentation teacher. https://github.com/BahramJafrasteh/WarpSeg
- **WASABI** — evaluation metric. Data: **ADNI**, **OASIS-3**.

## Citation

```bibtex
@article{wan2025agldm,
  title   = {Anatomically Guided Latent Diffusion for Brain MRI Progression Modeling},
  author  = {Wan, Cheng and Jafrasteh, Bahram and Adeli, Ehsan and Zhang, Miaomiao and Zhao, Qingyu},
  journal = {arXiv preprint},
  year    = {2025}
}
```

## License

[MIT](LICENSE), consistent with the upstream BrLP project.
