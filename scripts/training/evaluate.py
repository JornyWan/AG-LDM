#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluation: sample follow-up MRIs with a trained AG-LDM model and compute image MSE, in-house tissue Dice, SynthSeg-based ROI volume errors, and the WASABI distributional metric over a test set."""

import os
import argparse
import sys
import subprocess
import shutil
from typing import Optional, Union, Iterable, Set, Dict, List, Tuple

import json
import torch
import torch.nn.functional as F
import pandas as pd
import nibabel as nib
import numpy as np

from torch.utils.data import DataLoader
from monai.data.meta_tensor import MetaTensor
from monai import transforms
from monai.data.image_reader import NumpyReader
from monai.utils import set_determinism
from tqdm import tqdm

# ====== Project paths ======
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# ====== WASABI Utils ======
wasabi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wasabi_utils')
if wasabi_path not in sys.path:
    sys.path.append(wasabi_path)
try:
    from metrics import compute_metrics
    HAS_WASABI = True
except ImportError as e:
    print(f"[WARN] WASABI import failed: {e}. 'pot' library or metrics.py may be missing.")
    HAS_WASABI = False

from src.brlp import const, utils, networks
from src.brlp import get_dataset_from_pd
from src.brlp import sample_using_channel_cond
from segment_utils.segmentor_module import DiffusionSegmentationModule_2

# ---- torch.load compatibility for legacy weights ----
_orig_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = patched_torch_load

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
set_determinism(0)

# ---------------- Utils ----------------
def save_nifti_array(arr, path, affine=const.MNI152_1P5MM_AFFINE):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().float().numpy()
    nib.save(nib.Nifti1Image(arr, affine), path)

def normalize_like(x: np.ndarray, mode: str) -> np.ndarray:
    x = x.astype(np.float32)
    if mode == 'none':
        return x
    if mode == 'minmax':
        lo, hi = float(x.min()), float(x.max())
    else:  # 'pclip'
        lo, hi = np.percentile(x, 5), np.percentile(x, 99.5)
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return x

def apply_mask_if_needed(arr_np: np.ndarray, mask_path: Optional[str]) -> np.ndarray:
    if isinstance(mask_path, str) and os.path.isfile(mask_path):
        try:
            mask = nib.load(mask_path).get_fdata() > 0
            arr_np = arr_np * mask.astype(arr_np.dtype)
        except Exception as e:
            print(f"[WARN] apply_mask failed: {mask_path} ({e})")
    return arr_np

def list_from_file(p: str) -> Set[str]:
    ids: Set[str] = set()
    with open(p, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ids.add(s.split(',')[0])
    return ids

def write_ids_to_file(ids: Iterable[str], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for sid in ids:
            f.write(str(sid) + '\n')
    print(f"[INFO] Saved {len(list(ids))} eval ids to {out_path}")

def compute_seg_dice_argmax(seg_p, seg_r, label_map):
    if isinstance(seg_p, torch.Tensor): seg_p = seg_p.cpu().numpy()
    if isinstance(seg_r, torch.Tensor): seg_r = seg_r.cpu().numpy()
    out = {}
    for lid, name in label_map.items():
        ip = (seg_p == lid)
        ir = (seg_r == lid)
        inter = (ip & ir).sum()
        denom = ip.sum() + ir.sum()
        if denom > 0:
            dice_val = 2 * float(inter) / float(denom)
        else:
            dice_val = 1.0 if inter == 0 else 0.0
        out[f"dice_own_argmax_lbl{lid}_{name}"] = dice_val
    return out

def compute_seg_dice_thresholding(raw_pred_seg_logits: torch.Tensor, raw_real_seg_logits: torch.Tensor,
                                  label_map: dict, threshold=0.5):
    dice_results = {}
    probs_pred = torch.softmax(raw_pred_seg_logits, dim=1)
    probs_real = torch.softmax(raw_real_seg_logits, dim=1)
    for channel_idx, name in label_map.items():
        bin_p = (probs_pred[0, channel_idx] > threshold).cpu().numpy()
        bin_r = (probs_real[0, channel_idx] > threshold).cpu().numpy()
        intersection = np.sum(bin_p & bin_r)
        denominator = np.sum(bin_p) + np.sum(bin_r)
        if denominator > 0:
            dice_val = 2.0 * float(intersection) / float(denominator)
        else:
            dice_val = 1.0 if intersection == 0 else 0.0
        dice_key = f'dice_own_thresh_lbl{channel_idx}_{name}'
        dice_results[dice_key] = dice_val
    return dice_results

def compute_volume_loss(pred_img, real_img, seg_model):
    with torch.no_grad():
        pred_img_dev = pred_img.to(next(seg_model.parameters()).device)
        real_img_dev = real_img.to(next(seg_model.parameters()).device)
        seg_r = seg_model(real_img_dev, return_logits=True)
        seg_p = seg_model(pred_img_dev, return_logits=True)
    probs_p = torch.softmax(seg_p, dim=1)
    probs_r = torch.softmax(seg_r, dim=1)
    vols_p = torch.stack([probs_p[:,1].sum(), probs_p[:,2].sum(), probs_p[:,4].sum()])
    vols_r = torch.stack([probs_r[:,1].sum(), probs_r[:,2].sum(), probs_r[:,4].sum()])
    vols_p_norm = vols_p / (vols_p.sum() + 1e-6)
    vols_r_norm = vols_r / (vols_r.sum() + 1e-6)
    return torch.nn.functional.mse_loss(vols_p_norm, vols_r_norm).item()

def run_synthseg(img_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(img_path).replace('.nii.gz','').replace('.nii','')
    seg_p = os.path.join(out_dir, f'{base}_seg.mgz')
    csv_p = os.path.join(out_dir, f'{base}_volumes.csv')
    qc_d  = os.path.join(out_dir, 'qc')
    cmd   = f"mri_synthseg --i {img_path} --o {seg_p} --fast --vol {csv_p} --qc {qc_d} --threads 8"
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"SynthSeg failed for {img_path}. Error: {e.stderr.decode()}")
        raise
    return seg_p, csv_p

def parse_seg(path):
    return nib.load(path).get_fdata().astype(np.int32)

# ---------------- ROI helpers (SynthSeg-based) ----------------
def load_code_map(json_path: str) -> Dict[int, str]:
    with open(json_path, 'r') as f:
        mp = json.load(f)
    return {int(k): v for k, v in mp.items()}

def build_region_code_sets(code_map: Dict[int, str], roi_list: Union[List[str], str]) -> Dict[str, List[int]]:
    """
    Merge left/right sides into a single ROI; e.g. left_hippocampus/right_hippocampus -> 'hippocampus'.
    Finer-grained labels in code_map are also allowed and unioned together.
    If roi_list is 'all', automatically extracts all valid regions from code_map (merging L/R).
    """
    if isinstance(roi_list, str) and roi_list == 'all':
        region2codes = {}
        for code, name in code_map.items():
            # Skip background/unknown
            if 'background' in name.lower() or 'unknown' in name.lower():
                continue
            
            nm = name.lower()
            nm = nm.replace('left-', '').replace('right-', '')
            nm = nm.replace('left_', '').replace('right_', '')
            key = nm
            
            if key not in region2codes:
                region2codes[key] = []
            region2codes[key].append(code)
        return region2codes

    region2codes: Dict[str, List[int]] = {r: [] for r in roi_list}
    for code, name in code_map.items():
        # Normalize naming
        nm = name.lower()
        nm = nm.replace('left_', '').replace('right_', '')
        # Simple synonym handling
        if 'lateral_ventricle' in nm:
            key = 'lateral_ventricle'
        elif 'hippocampus' in nm:
            key = 'hippocampus'
        elif 'amygdala' in nm:
            key = 'amygdala'
        elif 'thalamus' in nm:
            key = 'thalamus'
        elif nm == 'csf' or 'csf' in nm:
            key = 'csf'
        else:
            key = None
        if key in region2codes:
            region2codes[key].append(code)
    return region2codes

def region_mask(seg: np.ndarray, codes: List[int]) -> np.ndarray:
    m = np.zeros_like(seg, dtype=bool)
    for c in codes:
        m |= (seg == c)
    return m

def measure_regions(seg: np.ndarray, region2codes: Dict[str, List[int]]) -> Tuple[Dict[str, int], int]:
    head = (seg > 0).sum()
    out: Dict[str, int] = {}
    for r, codes in region2codes.items():
        if len(codes) == 0:
            out[r] = 0
        else:
            out[r] = int(region_mask(seg, codes).sum())
    return out, int(head)

def compute_roi_metrics(pred_seg: np.ndarray, real_seg: np.ndarray,
                        region2codes: Dict[str, List[int]]) -> Dict[str, float]:
    """
    For each ROI, return:
      - normalized volume (pred/real)
      - absolute percentage error abs_pct (%)
      - signed percentage error signed_pct (%)
      - mask Dice
    """
    (pred_counts, pred_head) = measure_regions(pred_seg, region2codes)
    (real_counts, real_head) = measure_regions(real_seg, region2codes)

    # Guard against division by zero when head=0
    pred_head = max(pred_head, 1)
    real_head = max(real_head, 1)

    out: Dict[str, float] = {}
    for roi, _ in region2codes.items():
        pv = pred_counts[roi] / pred_head
        rv = real_counts[roi] / real_head
        # Error (percentage)
        abs_pct   = abs(pv - rv) * 100.0
        signed_pct= (pv - rv) * 100.0
        # Dice (merged into a single ROI mask)
        pm = region_mask(pred_seg, region2codes[roi])
        rm = region_mask(real_seg, region2codes[roi])
        inter = (pm & rm).sum()
        dice = 2.0 * inter / (pm.sum() + rm.sum() + 1e-6)

        out[f'roi_{roi}_vol_pred_norm']  = float(pv)
        out[f'roi_{roi}_vol_real_norm']  = float(rv)
        out[f'roi_{roi}_vol_abs_pct']    = float(abs_pct)
        out[f'roi_{roi}_vol_signed_pct'] = float(signed_pct)
        out[f'roi_{roi}_dice']           = float(dice)
    return out

# ---------------- Seg Wrapper ----------------
class SegWrapper(DiffusionSegmentationModule_2):
    ROI_NAMES = ['background', 'gray_matter', 'white_matter',
                 'ventricles', 'csf', 'deep_gray']

    def forward(self, img, *, return_logits=True):
        logits = super().forward(img)  # (B,6,D,H,W)
        logits = F.interpolate(logits, size=(122,146,122),
                               mode='trilinear', align_corners=False)
        return logits if return_logits else torch.softmax(logits, dim=1)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_csv', required=True)
    ap.add_argument('--cache_dir',   required=True)
    ap.add_argument('--output_dir',  required=True)
    ap.add_argument('--aekl_ckpt',   required=True)
    ap.add_argument('--diff_ckpt',   required=True)
    ap.add_argument('--segment_ckpt',required=True)
    ap.add_argument('--batch_size',  type=int, default=1)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--inference_steps', type=int, default=50)
    ap.add_argument('--use_mask',     action='store_true')
    ap.add_argument('--mask_column',  default='starting_segm_path')
    ap.add_argument('--results_csv',  default='test_results.csv')
    ap.add_argument('--summary_csv',  default='test_summary.csv')
    ap.add_argument('--latent_id', default=None, type=str, help="autoencoder/latent id string")
    ap.add_argument('--fixed_eval_ids', default=None,
                    help="TXT/CSV whose first column is subject_id; fixed evaluation subset")
    ap.add_argument('--save_eval_ids', default=None,
                    help="write the subject_id list used in this run to a file")
    ap.add_argument('--norm', default='pclip', choices=['none','minmax','pclip'],
                    help="intensity normalization applied jointly to pred/real")
    ap.add_argument('--cs_fair_cond', action='store_true',
                    help="condition only on [sex, followup_age]")
    # === ROI volume evaluation ===
    ap.add_argument('--synthseg_code_map', required=True,
                    help="JSON mapping from SynthSeg label to name")
    ap.add_argument('--roi_list', default='all',
                    help="comma-separated ROI names (L/R merged), or 'all' for every ROI")
    ap.add_argument('--wasabi_k', type=int, default=1000, help="WASABI bootstrap iterations")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, 'tmp')
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)

    latent_id = args.latent_id
    key_start = f'starting_latent_path_{latent_id}'
    key_follow= f'followup_latent_path_{latent_id}'
    keys_to_load = [key_start, key_follow]

    # ---------- Load and subsample ----------
    df = pd.read_csv(args.dataset_csv)
    test_df = df[df['split']=='test'].reset_index(drop=True)

    if args.fixed_eval_ids:
        keep_ids = list_from_file(args.fixed_eval_ids)
        test_df = test_df[test_df['subject_id'].astype(str).isin(keep_ids)].reset_index(drop=True)
        print(f"[INFO] Using fixed_eval_ids: {len(test_df)} samples")
    else:
        test_df = test_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampled 10% of test with random_state=42: {len(test_df)} samples")
        if args.save_eval_ids:
            write_ids_to_file(test_df['subject_id'].astype(str).tolist(), args.save_eval_ids)

    if len(test_df) == 0:
        raise RuntimeError("No test rows!")

    # ---------- Dataset / Loader ----------
    npz_reader = NumpyReader(npz_keys=['data'])
    trans = transforms.Compose([
        transforms.LoadImageD(keys=keys_to_load, reader=npz_reader),
        transforms.EnsureChannelFirstD(keys=keys_to_load, channel_dim=0),
        transforms.DivisiblePadD(keys=keys_to_load, k=4),
    ])
    ds = get_dataset_from_pd(test_df, trans, cache_dir=None)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # ---------- Models ----------
    autoenc = networks.init_autoencoder(args.aekl_ckpt).to(DEVICE).eval()
    diff_model_params = {'latent_channels': 3, 'num_covariates': 5}
    diff = networks.init_latent_diffusion_channel_cond(
        ckpt_path=None, **diff_model_params
    ).to(DEVICE)
    ckpt = torch.load(args.diff_ckpt, map_location=DEVICE)
    diff.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    diff.eval()
    scale_factor = ckpt.get('scale_factor', 1.0)
    print(f"[INFO] Using scale_factor from checkpoint: {scale_factor}")

    seg = SegWrapper(input_size=(224,256,192),
                     nrois=6, tissue_channels=6,
                     segmentation_checkpoint=args.segment_ckpt
                     ).to(DEVICE).eval()

    real_tf = transforms.Compose([
        transforms.LoadImageD(keys=['image'], image_only=True),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image']),
    ])

    synth_labels = [2,3,24,41,42]
    our_seg_labels_map = {1:'gm', 2:'wm', 4:'csf'}

    # === ROI volume setup ===
    code_map = load_code_map(args.synthseg_code_map)
    if args.roi_list.strip().lower() == 'all':
        roi_arg = 'all'
    else:
        roi_arg = [r.strip().lower() for r in args.roi_list.split(',') if r.strip()]

    region2codes = build_region_code_sets(code_map, roi_arg)
    # Ensure roi_list contains every generated key for later saving and evaluation
    roi_list = sorted(list(region2codes.keys()))

    rows = []

    for batch in tqdm(dl, desc='Testing'):
        sid = str(batch['subject_id'][0])
        real_path = batch['followup_image_path'][0]
        mask_path = batch.get(args.mask_column, [None])[0] if args.use_mask else None

        sex  = batch['sex'].to(DEVICE)
        fage = batch['followup_age'].to(DEVICE)
        if args.cs_fair_cond:
            s_age = None; s_diag = None; f_diag = None
        else:
            s_age  = batch['starting_age'].to(DEVICE)
            s_diag = batch['starting_diagnosis'].to(DEVICE)
            f_diag = batch['followup_diagnosis'].to(DEVICE)

        # Sampling
        z_start = batch[key_start].to(DEVICE)
        with torch.no_grad():
            pred = sample_using_channel_cond(
                autoencoder=autoenc, diffusion=diff, starting_z=z_start,
                starting_age=(s_age if not args.cs_fair_cond else fage*0),
                followup_age=fage,
                sex=sex,
                starting_diagnosis=(s_diag if not args.cs_fair_cond else fage*0),
                followup_diagnosis=(f_diag if not args.cs_fair_cond else fage*0),
                device=DEVICE, scale_factor=scale_factor,
                num_inference_steps=args.inference_steps
            )

        # Assemble pred
        if   pred.ndim == 5 and pred.shape[:2] == (1,1): pred_np = pred[0,0].cpu().numpy()
        elif pred.ndim == 4 and pred.shape[0] == 1:      pred_np = pred[0].cpu().numpy()
        elif pred.ndim == 3:                              pred_np = pred.cpu().numpy()
        else:
            print(f"[WARN] unexpected pred shape {tuple(pred.shape)}, squeezing...")
            pred_np = pred.squeeze().cpu().numpy()

        # Joint normalization and masking
        real_arr  = real_tf({'image':real_path})['image'].squeeze(0).float().cpu().numpy()
        real_np   = normalize_like(real_arr, args.norm)
        real_np   = apply_mask_if_needed(real_np, mask_path) if args.use_mask else real_np

        pred_np   = normalize_like(pred_np, args.norm)
        pred_np   = apply_mask_if_needed(pred_np, mask_path) if args.use_mask else pred_np

        # Evaluation size
        pad_crop = transforms.ResizeWithPadOrCrop(spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum')
        pred_arr = pad_crop(torch.from_numpy(pred_np).unsqueeze(0)).squeeze(0)
        real_arr = pad_crop(torch.from_numpy(real_np).unsqueeze(0)).squeeze(0)

        # Save NIfTI
        fn_pred = os.path.join(args.output_dir, f'{sid}_pred.nii.gz')
        fn_real = os.path.join(args.output_dir, f'{sid}_real.nii.gz')
        save_nifti_array(pred_arr, fn_pred)
        save_nifti_array(real_arr, fn_real)

        # Metrics (image / in-house seg)
        mse_val = F.mse_loss(pred_arr.to(DEVICE), real_arr.to(DEVICE)).item()
        pred_img_for_seg = pred_arr.unsqueeze(0).unsqueeze(0).to(DEVICE)
        real_img_for_seg = real_arr.unsqueeze(0).unsqueeze(0).to(DEVICE)
        vol_loss = compute_volume_loss(pred_img_for_seg, real_img_for_seg, seg)

        with torch.no_grad():
            raw_seg_output_pred = seg(pred_img_for_seg, return_logits=True)
            raw_seg_output_real = seg(real_img_for_seg, return_logits=True)

        seg_pred_argmax = torch.softmax(raw_seg_output_pred, dim=1).argmax(1)[0]
        seg_real_argmax = torch.softmax(raw_seg_output_real, dim=1).argmax(1)[0]
        own_dice_argmax_results = compute_seg_dice_argmax(seg_pred_argmax, seg_real_argmax, our_seg_labels_map)
        own_dice_thresh_results = compute_seg_dice_thresholding(raw_seg_output_pred, raw_seg_output_real,
                                                                our_seg_labels_map, threshold=0.5)

        # SynthSeg (produce segmentations for ROI volumes)
        p_tmp = os.path.join(tmp_dir, f'{sid}_pred_for_synthseg.nii.gz'); save_nifti_array(pred_arr, p_tmp)
        r_tmp = os.path.join(tmp_dir, f'{sid}_real_for_synthseg.nii.gz'); save_nifti_array(real_arr, r_tmp)

        dice_syn_results = {}
        roi_metrics = {}
        try:
            sp_path, _ = run_synthseg(p_tmp, os.path.join(tmp_dir, f'{sid}_pred_synthseg_out'))
            rp_path, _ = run_synthseg(r_tmp, os.path.join(tmp_dir, f'{sid}_real_synthseg_out'))

            # 1) Classic SynthSeg Dice (for the selected labels)
            synth_label_map_for_func = {lbl_id: str(lbl_id) for lbl_id in [2,3,24,41,42]}
            def _parse_seg(path): return nib.load(path).get_fdata().astype(np.int32)
            temp_dice_syn = compute_seg_dice_argmax(_parse_seg(sp_path), _parse_seg(rp_path), synth_label_map_for_func)
            dice_syn_results = {f'dice_synth_lbl{k.split("_")[-1]}': v for k, v in temp_dice_syn.items()}

            # 2) ROI volume metrics
            pred_seg_arr = parse_seg(sp_path)
            real_seg_arr = parse_seg(rp_path)
            roi_metrics = compute_roi_metrics(pred_seg_arr, real_seg_arr, region2codes)

        except Exception as e:
            print(f"[WARN] SynthSeg failed for {sid}: {e}")
            for lid in [2,3,24,41,42]:
                dice_syn_results[f'dice_synth_lbl{lid}'] = np.nan
            # Set ROI metrics to NaN
            for roi in roi_list:
                for k in ['vol_pred_norm','vol_real_norm','vol_abs_pct','vol_signed_pct','dice']:
                    roi_metrics[f'roi_{roi}_{k}'] = np.nan

        # Record
        row = {
            'subject_id': sid,
            'start_age': float(batch['starting_age'][0]) if 'starting_age' in batch else np.nan,
            'follow_age': float(batch['followup_age'][0]),
            'mse': mse_val,
            'vol_loss_ourseg': vol_loss,
            'pred_min': float(pred_arr.min()), 'pred_max': float(pred_arr.max()),
            'real_min': float(real_arr.min()), 'real_max': float(real_arr.max()),
        }
        row.update(own_dice_argmax_results)
        row.update(own_dice_thresh_results)
        row.update(dice_syn_results)
        row.update(roi_metrics)
        rows.append(row)

    # ---------- Save per-sample details ----------
    df_res = pd.DataFrame(rows)
    df_res.to_csv(os.path.join(args.output_dir, args.results_csv), index=False)

    # ---------- Overall summary ----------
    num_cols = [c for c in df_res.columns if pd.api.types.is_numeric_dtype(df_res[c])]
    if num_cols:
        means = df_res[num_cols].mean(skipna=True).add_suffix('_mean').to_dict()
        stds  = df_res[num_cols].std(skipna=True).add_suffix('_std').to_dict()
        pd.DataFrame([dict(means, **stds)]).to_csv(os.path.join(args.output_dir, args.summary_csv), index=False)
    else:
        print("[WARN] No numeric data to summarize.")

    # ---------- Per-ROI overall summary ----------
    # Compute: MAE(%) / signed_mean(%) / Pearson r (pred_norm vs real_norm) / Dice_mean
    roi_rows = []
    for roi in roi_list:
        pcol = f'roi_{roi}_vol_pred_norm'
        rcol = f'roi_{roi}_vol_real_norm'
        ecol = f'roi_{roi}_vol_abs_pct'
        scol = f'roi_{roi}_vol_signed_pct'
        dcol = f'roi_{roi}_dice'

        sub = df_res[[pcol, rcol, ecol, scol, dcol]].dropna()
        if len(sub) == 0:
            roi_rows.append({'roi': roi, 'n': 0})
            continue

        # Pearson r
        try:
            pr = float(np.corrcoef(sub[pcol], sub[rcol])[0,1])
        except Exception:
            pr = np.nan

        roi_rows.append({
            'roi': roi,
            'n': int(len(sub)),
            'MAE_%': float(sub[ecol].mean()),
            'SignedMean_%': float(sub[scol].mean()),
            'PearsonR': pr,
            'Dice_mean': float(sub[dcol].mean()),
            'Dice_std': float(sub[dcol].std())
        })

    pd.DataFrame(roi_rows).to_csv(os.path.join(args.output_dir, 'roi_volume_summary.csv'), index=False)
    print(f"[OK] ROI volume summary saved to roi_volume_summary.csv")

    # ---------- WASABI metric (save data + attempt computation) ----------
    if len(df_res) > 0:
        print("[INFO] Preparing data for WASABI metric...")
        try:
            # 1. Prepare data and save a clean copy for later offline computation
            valid_rois = [r for r in roi_list]
            roi_cols_pred = [f'roi_{r}_vol_pred_norm' for r in valid_rois]
            roi_cols_real = [f'roi_{r}_vol_real_norm' for r in valid_rois]
            
            # Select columns: subject_id + all required ROI columns
            cols_to_keep = ['subject_id'] + roi_cols_pred + roi_cols_real
            # Keep only columns that exist
            cols_to_keep = [c for c in cols_to_keep if c in df_res.columns]
            
            df_wasabi_subset = df_res[cols_to_keep].copy()
            
            # Save the CSV for WASABI computation (includes NaNs; downstream decides how to drop)
            wasabi_data_path = os.path.join(args.output_dir, 'wasabi_data_raw.csv')
            df_wasabi_subset.to_csv(wasabi_data_path, index=False)
            print(f"[OK] Saved raw WASABI input data to {wasabi_data_path}")
            
            # Drop NaN rows for computation
            df_clean = df_wasabi_subset.dropna()
            
            # Save the cleaned WASABI CSV for offline computation
            wasabi_clean_path = os.path.join(args.output_dir, 'wasabi_data_clean.csv')
            df_clean.to_csv(wasabi_clean_path, index=False)
            print(f"[OK] Saved clean WASABI input data ({len(df_clean)} samples) to {wasabi_clean_path}")

            # 2. If the WASABI library is available, attempt in-process computation
            if HAS_WASABI:
                if len(df_clean) < 5:
                    print(f"[WARN] Not enough clean samples ({len(df_clean)}) for WASABI calculation.")
                else:
                    print("[INFO] Running in-process WASABI calculation...")
                    data_pred = df_clean[roi_cols_pred].values.astype(float)
                    data_real = df_clean[roi_cols_real].values.astype(float)
                    
                    wasabi_scores = compute_metrics(data_real, data_pred, K=args.wasabi_k, criterion='wasabi')
                    
                    w_vals = list(wasabi_scores.values())
                    w_mean = np.mean(w_vals)
                    w_std  = np.std(w_vals)
                    
                    print(f"WASABI Score: {w_mean:.6f} +/- {w_std:.6f}")
                    
                    with open(os.path.join(args.output_dir, 'wasabi_score.txt'), 'w') as f:
                        f.write(f"WASABI_mean: {w_mean}\n")
                        f.write(f"WASABI_std: {w_std}\n")
                        f.write(f"K: {args.wasabi_k}\n")
                        f.write(f"N_samples: {len(df_clean)}\n")
                        f.write(f"ROIs: {','.join(valid_rois)}\n")
            else:
                print("[INFO] WASABI utils not found or import failed. Skipping calculation. The cleaned WASABI CSV ('wasabi_data_clean.csv') was saved for offline computation.")
                
        except Exception as e:
            print(f"[ERROR] Failed during WASABI preparation/calculation: {e}")
            import traceback
            traceback.print_exc()

    print(f"[OK] All done. Results saved to {args.output_dir}")
    try:
        shutil.rmtree(tmp_dir)
        print(f"[OK] Temporary directory {tmp_dir} removed.")
    except OSError as e:
        print(f"[WARN] Removing temp dir {tmp_dir} failed: {e}")

if __name__ == '__main__':
    main()
