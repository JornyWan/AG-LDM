"""Stage 2: train the channel-conditioned latent diffusion model. The follow-up latent is predicted from the starting latent plus five spatially-broadcast covariate channels (channel concatenation; no cross-attention). An optional differentiable WM/GM soft-Dice volume loss uses a frozen tissue-segmentation teacher for anatomical guidance."""

import os
import argparse
import sys
import subprocess
import tempfile

import torch
import torch.nn.functional as F
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt

_orig_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)

torch.load = patched_torch_load

from torch.serialization import safe_globals
from monai.data.meta_tensor import MetaTensor

from monai import transforms
from monai.utils import set_determinism
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from generative.inferers import DiffusionInferer
from tqdm import tqdm
import nibabel as nib
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.brlp import const
from src.brlp import utils
from src.brlp import networks
from src.brlp import (
    get_dataset_from_pd,
    sample_using_channel_cond,
)

# Pretrained tissue-segmentation module, used as a frozen teacher.
from segment_utils.segmentor_module import DiffusionSegmentationModule_2
from matplotlib.colors import ListedColormap

set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

USE_SOFTMAX = True   # Global switch: True uses softmax, False uses sigmoid.

# Constants (kept consistent with the real-data script).
ROI_NAMES  = ['background', 'gray_matter', 'white_matter', 'ventricles', 'csf', 'deep_gray']
ROI_COLORS = ['black',     'green',       'red',          'yellow',    'blue', 'purple']
CMAP       = ListedColormap(ROI_COLORS[:len(ROI_NAMES)])
INPUT_SHAPE_1P5MM = (122, 146, 122)               # Final generated volumes should all be this size.
WM_INDEX = 2                                      # white_matter channel index
GM_INDEX = 1

def logits_to_prob(logits):
    """Convert logits to a [0,1] probability map according to the global switch."""
    return (torch.softmax(logits, dim=1) if USE_SOFTMAX
            else torch.sigmoid(logits))

class SegWrapper(DiffusionSegmentationModule_2):
    """forward(img, return_logits=True, resize_back=True), matching the data-processing script interface."""
    def forward(self, img, *, return_logits=True, resize_back=True):
        logits = super().forward(img)                    # Min-max normalized internally.
        if resize_back:                                  # Resample back to the 1.5 mm grid.
            logits = F.interpolate(logits,
                                    size=INPUT_SHAPE_1P5MM,
                                    mode='trilinear',
                                    align_corners=False)
        if return_logits:
            return logits
        # 0-1 for visualization
        mn, mx = logits.amin((2,3,4), True), logits.amax((2,3,4), True)
        return (logits - mn) / (mx - mn + 1e-6)

def save_nifti(tensor, path, affine=None):
    """Save tensor as NIfTI file"""
    if affine is None:
        affine = getattr(const, 'MNI152_1P5MM_AFFINE', np.eye(4) * 1.5) # Default 1.5mm isotropic
        affine[3, 3] = 1.0 # Ensure last element is 1

    # Ensure tensor is CPU NumPy array (float32)
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().float().numpy()
    else:
        array = np.asarray(tensor, dtype=np.float32)

    # Ensure array is 3D
    if array.ndim == 4 and array.shape[0] == 1: # Remove channel dim if present
        array = array.squeeze(0)
    elif array.ndim != 3:
        raise ValueError(f"Expected 3D array for NIfTI, got shape {array.shape}")

    nifti_saver = NiftiSaver(output_dir=os.path.dirname(path), output_postfix="", output_ext=".nii.gz", dtype=np.float32)
    # MONAI saver expects metadata, provide minimal required
    meta_data = {'affine': torch.from_numpy(affine), 'original_affine': torch.from_numpy(affine)}
    # Wrap array in MetaTensor
    meta_tensor = MetaTensor(x=torch.from_numpy(array), meta=meta_data)
    nifti_saver.save(meta_tensor, meta_data) # Pass meta_data again for saver compatibility

    # Rename file to exact path specified
    default_name = nifti_saver.get_filename(meta_tensor) # saver might append '_trans' etc.
    default_path = os.path.join(os.path.dirname(path), default_name)
    if default_path != path and os.path.exists(default_path):
         os.rename(default_path, path)

    return path



# Visualization for intermediate segmentation.
def plot_overlay(img_vol: np.ndarray,
                 mask_vol: np.ndarray,
                 save_png: str):
    mids = (img_vol.shape[0]//2, img_vol.shape[1]//2, img_vol.shape[2]//2)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a,slc,axid,ttl in zip(ax, mids, [0,1,2], ['Sagittal','Coronal','Axial']):
        im,msk = (img_vol[slc,:,:], mask_vol[slc,:,:])   if axid==0 else \
                 (img_vol[:,slc,:], mask_vol[:,slc,:])   if axid==1 else \
                 (img_vol[:,:,slc], mask_vol[:,:,slc])
        a.imshow(im.T,  cmap='gray', origin='lower')
        a.imshow(msk.T, cmap=CMAP,  origin='lower',
                 alpha=0.6, vmin=0, vmax=len(ROI_NAMES)-1)
        a.set_title(f'{ttl} {slc}'); a.axis('off')
    handles = [plt.Line2D([0],[0],marker='s',color=ROI_COLORS[i],
               markersize=9,label=ROI_NAMES[i])
               for i in range(len(ROI_NAMES))]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.02, .5))
    plt.tight_layout(); plt.savefig(save_png, dpi=140, bbox_inches='tight'); plt.close()

def images_to_wandb(
    epoch,
    mode,
    autoencoder,
    diffusion,
    scale_factor,
    dataset,
    global_step, # Use unified global step
    latent_id
):
    """
    Visualize the generation using W&B with paired comparison.
    Uses channel-based conditioning.
    """
    resample_fn = transforms.Spacing(pixdim=1.5)
    num_samples_to_show = min(3, len(dataset)) # Show fewer if dataset is small
    if num_samples_to_show == 0:
        print(f"[WARN] images_to_wandb: Dataset for mode '{mode}' is empty. Skipping visualization.")
        return
    random_indices = np.random.choice(range(len(dataset)), num_samples_to_show, replace=False)

    diffusion.eval() # Ensure model is in eval mode for sampling
    autoencoder.eval()

    for tag_i, i in enumerate(random_indices):
        # Fetch all required data from the dataset sample
        sample = dataset[i]
        starting_z_unscaled = sample[f'starting_latent_path_{latent_id}'] # Keep unscaled for condition prep
        starting_age_val = sample['starting_age']
        followup_age_val = sample['followup_age']
        sex_val = sample['sex']
        starting_diagnosis_val = sample['starting_diagnosis']
        followup_diagnosis_val = sample['followup_diagnosis']

        # Prepare inputs for sampling function (needs batch dim)
        starting_z_batch = (starting_z_unscaled * scale_factor).unsqueeze(0).to(DEVICE) # Scale here
        starting_age_batch = torch.tensor([starting_age_val], dtype=torch.float32).to(DEVICE)
        followup_age_batch = torch.tensor([followup_age_val], dtype=torch.float32).to(DEVICE)
        sex_batch = torch.tensor([sex_val], dtype=torch.float32).to(DEVICE)
        starting_diagnosis_batch = torch.tensor([starting_diagnosis_val], dtype=torch.float32).to(DEVICE)
        followup_diagnosis_batch = torch.tensor([followup_diagnosis_val], dtype=torch.float32).to(DEVICE)

        # Load actual images for comparison
        try:
            starting_img_path = sample['starting_image_path']
            followup_img_path = sample['followup_image_path']
            # Use MONAI LoadImage for robust loading
            loader = transforms.LoadImage(image_only=True)
            starting_image_orig = loader(starting_img_path).unsqueeze(0) # Add batch dim
            followup_image_orig = loader(followup_img_path).unsqueeze(0) # Add batch dim

            # Resample and remove batch dim for display function
            starting_image_disp = resample_fn(starting_image_orig).squeeze(0)
            followup_image_disp = resample_fn(followup_image_orig).squeeze(0)

        except Exception as e:
            print(f"[ERROR] Error loading images for index {i}: {e}")
            continue

        # Generate image using the new channel-based conditioning sampling function
        predicted_image = sample_using_channel_cond(
            autoencoder=autoencoder,
            diffusion=diffusion,
            starting_z=starting_z_batch, # Pass scaled starting_z
            starting_age=starting_age_batch,
            followup_age=followup_age_batch,
            sex=sex_batch,
            starting_diagnosis=starting_diagnosis_batch,
            followup_diagnosis=followup_diagnosis_batch,
            device=DEVICE,
            scale_factor=scale_factor,
            num_inference_steps=50,
            verbose=False # Use a reasonable number for visualization
        )

        # Display the results
        tag = f'{mode}/comparison_{tag_i}'

        # Pass metadata to display function
        utils.wandb_display_cond_generation(
            step=global_step,
            tag=tag,
            starting_image=starting_image_disp,
            followup_image=followup_image_disp,
            predicted_image=predicted_image, # Remove batch dim for display
            start_diag=starting_diagnosis_val,
            follow_diag=followup_diagnosis_val,
            sex_value=sex_val,
            start_age=starting_age_val,
            follow_age=followup_age_val
        )
    diffusion.train() # Set back to train mode if it was training


def generate_sample_for_volume_loss(
    diffusion,
    autoencoder,
    starting_z,             # Shape: (N, C, D, H, W), scaled
    starting_age,           # Shape: (N,)
    followup_age,           # Shape: (N,)
    sex,                    # Shape: (N,)
    starting_diagnosis,     # Shape: (N,)
    followup_diagnosis,     # Shape: (N,)
    scale_factor,
    num_inference_steps=10  # More steps reduce noise, but too many slow down training.
):
    """Use DDIM for better few-step generation quality while avoiding extra spatial transforms."""
    device = starting_z.device
    n = starting_z.shape[0]
    latent_shape = starting_z.shape[2:] # (D, H, W)

    # Ensure covariates are on the correct device
    starting_age = starting_age.to(device)
    followup_age = followup_age.to(device)
    sex = sex.to(device)
    starting_diagnosis = starting_diagnosis.to(device)
    followup_diagnosis = followup_diagnosis.to(device)

    # Create spatial channels for all covariates
    start_age_channel = starting_age.view(n, 1, 1, 1, 1).expand(n, 1, *latent_shape)
    followup_age_channel = followup_age.view(n, 1, 1, 1, 1).expand(n, 1, *latent_shape)
    sex_channel = sex.view(n, 1, 1, 1, 1).expand(n, 1, *latent_shape)
    start_diag_channel = starting_diagnosis.view(n, 1, 1, 1, 1).expand(n, 1, *latent_shape)
    follow_diag_channel = followup_diagnosis.view(n, 1, 1, 1, 1).expand(n, 1, *latent_shape)

    # Concatenate starting_z and all covariate channels as the condition part
    # Shape: (N, C + 5, D, H, W)
    condition_channels = torch.cat([
        starting_z, # Already scaled
        start_age_channel,
        followup_age_channel,
        sex_channel,
        start_diag_channel,
        follow_diag_channel
    ], dim=1)

    # Initialize noise
    latents = torch.randn_like(starting_z).to(device) # Noise matches target latent shape

    # Use a DDIM scheduler for better few-step generation quality.
    sample_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.0205,
        clip_sample=False
    )
    sample_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # Denoising loop - *WITH GRADIENTS ENABLED*
    for t in sample_scheduler.timesteps:
        timestep = torch.tensor([t] * n, device=device).long()

        # Concatenate current noisy latent with condition channels
        model_input = torch.cat([latents, condition_channels], dim=1)

        # Predict noise
        noise_pred = diffusion(
            x=model_input.float(),
            timesteps=timestep,
            context=None
        )

        # DDIM step (supports gradient propagation).
        latents, _ = sample_scheduler.step(noise_pred, t, latents)

    # Decode directly, without any extra spatial-transform trick.
    generated_image = autoencoder.decode_stage_2_outputs(latents / scale_factor)

    return generated_image


def to_vae_latent_trick_grad(z: torch.Tensor, unpadded_z_shape: tuple = (3, 15, 18, 15)) -> torch.Tensor:
    """Gradient-friendly version of to_vae_latent_trick that stays on the same device (no CPU transfer)."""
    from monai.data.meta_tensor import MetaTensor

    # Create the padder.
    padder = transforms.DivisiblePad(k=4)

    # Create a zero tensor on the same device.
    zeros = torch.zeros(unpadded_z_shape, device=z.device, dtype=z.dtype)
    meta_zeros = MetaTensor(zeros)

    # Apply padding, then add the input.
    padded_zeros = padder(meta_zeros)
    z_with_padding = padded_zeros + z

    # Apply the inverse transform.
    z_unpadded = padder.inverse(z_with_padding)

    # Ensure a plain tensor is returned instead of a MetaTensor.
    if isinstance(z_unpadded, MetaTensor):
        z_unpadded = z_unpadded.as_tensor()
    
    return z_unpadded



def to_mni_space_1p5mm_trick_grad_v2(x: torch.Tensor, mni1p5_dim: tuple = (122, 146, 122)) -> torch.Tensor:
    """Use native PyTorch operations to avoid CPU transfer."""
    if x.shape[-3:] == mni1p5_dim:
        return x

    # Ensure the correct dimensions.
    if x.dim() == 3:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 4:
        x = x.unsqueeze(0)

    # Resample with interpolate.
    x_resized = F.interpolate(
        x.float(),  # Ensure float32.
        size=mni1p5_dim,
        mode='trilinear',
        align_corners=False
    )

    # Restore the dimensions.
    while x_resized.dim() > 4:
        x_resized = x_resized.squeeze(0)
    
    return x_resized
    
def compute_volume_loss_differentiable(gen_img, tgt_img, seg_module, smooth: float = 1e-6):
    """Compute the mean Soft-Dice loss over WM and GM."""
    with torch.no_grad():
        logits_tgt = seg_module(tgt_img, return_logits=True)
    logits_gen = seg_module(gen_img, return_logits=True)

    # --- Step 1: compute probability maps from logits (softmax). ---
    # (B, num_classes, D, H, W)
    probs_gen = F.softmax(logits_gen, dim=1)
    with torch.no_grad():
        probs_tgt = F.softmax(logits_tgt, dim=1)

    # --- Step 2: compute the Dice loss for WM and GM separately. ---
    total_loss = 0
    # Iterate over the tissue indices of interest.
    for tissue_index in [GM_INDEX, WM_INDEX]:
        # Extract the probability map for this channel (B, D, H, W).
        prob_gen_channel = probs_gen[:, tissue_index]
        prob_tgt_channel = probs_tgt[:, tissue_index]

        # Compute the Dice score for this channel.
        intersection = (prob_gen_channel * prob_tgt_channel).sum(dim=(1, 2, 3))
        union = prob_gen_channel.sum(dim=(1, 2, 3)) + prob_tgt_channel.sum(dim=(1, 2, 3))

        dice_score = (2.0 * intersection + smooth) / (union + smooth)

        # Accumulate the loss (1 - Dice).
        total_loss += (1 - dice_score)

    # --- Step 3: average the loss. ---
    # .mean() handles batch > 1, then divide by the number of tissues.
    final_loss = total_loss.mean() / 2.0

    return final_loss, logits_gen, logits_tgt


def plot_image_and_mask_pair(gen_img: np.ndarray,
                             gen_mask: np.ndarray,
                             tgt_img: np.ndarray,
                             tgt_mask: np.ndarray,
                             save_png: str):
    """Row 1: generated image | generated mask. Row 2: target image | target mask. Each column shows the same slice."""
    mids = (gen_img.shape[0]//2,
            gen_img.shape[1]//2,
            gen_img.shape[2]//2)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    # Compare a single slice: the middle sagittal slice.
    slc = mids[0]

    # Row 0: Gen
    axes[0,0].imshow(gen_img[slc,:,:].T, cmap='gray', origin='lower')
    axes[0,0].set_title('Gen Image'); axes[0,0].axis('off')
    axes[0,1].imshow(gen_mask[slc,:,:].T, cmap=CMAP, origin='lower',
                     vmin=0, vmax=len(ROI_NAMES)-1)
    axes[0,1].set_title('Gen Mask'); axes[0,1].axis('off')

    # Row 1: Tgt
    axes[1,0].imshow(tgt_img[slc,:,:].T, cmap='gray', origin='lower')
    axes[1,0].set_title('Tgt Image'); axes[1,0].axis('off')
    axes[1,1].imshow(tgt_mask[slc,:,:].T, cmap=CMAP, origin='lower',
                     vmin=0, vmax=len(ROI_NAMES)-1)
    axes[1,1].set_title('Tgt Mask'); axes[1,1].axis('off')

    # Legend (optional).
    handles = [plt.Line2D([0],[0],marker='s',color=ROI_COLORS[i],
                          markersize=8,label=ROI_NAMES[i])
               for i in range(len(ROI_NAMES))]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, .9))

    plt.tight_layout()
    plt.savefig(save_png, dpi=140, bbox_inches='tight')
    plt.close()

# ==============================================================
#  overlay_pair: left = generated, right = target
# ==============================================================
def plot_overlay_pair(gen_img, gen_mask,
                      tgt_img, tgt_mask,
                      save_png: str):

    mids = (gen_img.shape[0]//2,
            gen_img.shape[1]//2,
            gen_img.shape[2]//2)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))   # 2 rows, 3 columns.

    # ------- helper -------
    def _draw(a, im, msk, title):
        a.imshow(im.T,  cmap='gray', origin='lower')
        a.imshow(msk.T, cmap=CMAP,  origin='lower',
                 alpha=0.6, vmin=0, vmax=len(ROI_NAMES)-1)
        a.set_title(title); a.axis('off')

    # ---------- row 0 : Generated ----------
    for col,(slc,axid,ttl) in enumerate(zip(mids,[0,1,2],
                                     ['Sagittal','Coronal','Axial'])):
        im,msk = (gen_img[slc,:,:], gen_mask[slc,:,:])   if axid==0 else \
                 (gen_img[:,slc,:], gen_mask[:,slc,:])   if axid==1 else \
                 (gen_img[:,:,slc], gen_mask[:,:,slc])
        _draw(ax[0,col], im, msk, f'Gen {ttl} {slc}')

    # ---------- row 1 : Target ----------
    for col,(slc,axid,ttl) in enumerate(zip(mids,[0,1,2],
                                     ['Sagittal','Coronal','Axial'])):
        im,msk = (tgt_img[slc,:,:], tgt_mask[slc,:,:])   if axid==0 else \
                 (tgt_img[:,slc,:], tgt_mask[:,slc,:])   if axid==1 else \
                 (tgt_img[:,:,slc], tgt_mask[:,:,slc])
        _draw(ax[1,col], im, msk, f'Tgt {ttl} {slc}')

    # ------- legend -------
    handles = [plt.Line2D([0],[0],marker='s',color=ROI_COLORS[i],
                          markersize=8,label=ROI_NAMES[i])
               for i in range(len(ROI_NAMES))]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.02, .5))

    plt.tight_layout()
    plt.savefig(save_png, dpi=140, bbox_inches='tight')
    plt.close()


def align_images_for_loss(generated_image, target_image_path, device):
    """1) Load the target NIfTI from disk. 2) Convert to (1,1,D_orig,H_orig,W_orig). 3) Resample with F.interpolate to the same (D,H,W) as generated_image. 4) Return (1,1,D,H,W)."""
    # 1) Load into numpy.
    target_np = nib.load(target_image_path).get_fdata()  # shape (D0,H0,W0)
    # 2) Convert to torch and add batch + channel dims.
    tgt = torch.from_numpy(target_np).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D0,H0,W0)
    # 3) Target spatial size.
    D, H, W = generated_image.shape[2:]  # (D,H,W) of the generated image.
    # 4) Trilinear resampling.
    tgt_resampled = F.interpolate(
        tgt,
        size=(D, H, W),
        mode='trilinear',
        align_corners=False,
    )
    return tgt_resampled  # (1,1,D,H,W)
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',  required=True, type=str)
    parser.add_argument('--cache_dir',  required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--aekl_ckpt',  required=True, type=str)
    # Make diff_ckpt optional for starting from scratch
    parser.add_argument('--diff_ckpt',   default=None, type=str, help="Path to diffusion model checkpoint to resume training, if any.")
    parser.add_argument('--num_workers', default=8,     type=int)
    parser.add_argument('--n_epochs',    default=60,    type=int) # Example default
    parser.add_argument('--batch_size',  default=8,     type=int) # Example default
    parser.add_argument('--lr',          default=2.5e-5,  type=float)
    parser.add_argument('--wandb_project', default='brain-latent-diffusion', type=str)
    parser.add_argument('--wandb_entity', default=None, type=str)
    parser.add_argument('--wandb_name', default=None, type=str)
    parser.add_argument('--temp_dir', default='/tmp/agldm_vol_loss', type=str, help='Temporary file directory')
    parser.add_argument('--volume_loss_weight', default=1.0, type=float, help='Volume loss weight') # Example default
    parser.add_argument('--vol_loss_freq', default=10, type=int, help='Calculate volume loss every N steps')
    parser.add_argument('--use_volume_loss', action='store_true', help='Whether to use volume loss')
    parser.add_argument('--vol_loss_start_step', default=20, type=int, help='Global step to start calculating volume loss')
    parser.add_argument('--seg_ckpt', default='segment_utils/Tissue_model.pth', type=str, help="Path to the WarpSeg tissue-segmentation checkpoint.")
    parser.add_argument('--latent_id', default=None, type=str, help="ID for your AE latent keys (e.g. 629608)")
    args = parser.parse_args()

    # Display parameter settings
    print("\n========== Parameter Configuration ==========")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=============================================\n")

    # Create temporary directory
    if args.use_volume_loss:
        print(f"Creating temporary directory: {args.temp_dir}")
        os.makedirs(args.temp_dir, exist_ok=True)
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    latent_id = args.latent_id
    key_start = f'starting_latent_path_{latent_id}'
    key_follow= f'followup_latent_path_{latent_id}'

    # Initialize wandb
    run_name = args.wandb_name if args.wandb_name else f"paired-diffusion-channel-cond-{wandb.util.generate_id()}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
    )

    # Setup data loading (no Lambda for context needed now)
    img_reader = NumpyReader(npz_keys=['data'])
    # Ensure follow_image_path is loaded if needed for volume loss GT
    keys_to_load = [key_start, key_follow]
    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=keys_to_load, reader=img_reader, image_only=True, AllowMissingKeys=True), # Allow missing followup_image_path if not used for GT
        transforms.EnsureChannelFirstD(keys=[key_start, key_follow], channel_dim=0),
        transforms.DivisiblePadD(keys=[key_start, key_follow], k=4, mode='constant'),
        # No Lambda(func=concat_covariates) needed here
    ])

    # Load dataset
    dataset_df = pd.read_csv(args.dataset_csv)
    train_df = dataset_df[dataset_df.split == 'train']
    valid_df = dataset_df[dataset_df.split == 'valid']
    # Pass cache_dir=None if caching is handled internally or not desired
    trainset = get_dataset_from_pd(train_df, transforms_fn, cache_dir=None) # Use provided cache dir
    validset = get_dataset_from_pd(valid_df, transforms_fn, cache_dir=None) # Use provided cache dir

    print(f"DEBUG: trainset length = {len(trainset)}")
    print(f"DEBUG: validset length = {len(validset)}")
    if len(trainset) == 0:
        raise ValueError("Training dataset is empty. Check CSV path and split column.")

    # Create data loaders
    train_loader = DataLoader(
        dataset=trainset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        persistent_workers=args.num_workers > 0, # Only if using workers
        pin_memory=torch.cuda.is_available()
    )
    valid_loader = DataLoader(
        dataset=validset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        persistent_workers=args.num_workers > 0, # Only if using workers
        pin_memory=torch.cuda.is_available()
    )

    # Initialize models
    autoencoder = networks.init_autoencoder(args.aekl_ckpt).to(DEVICE)

    # Initialize the channel-conditioned diffusion model. Its input channels must
    # match the concatenation below: noisy latent (C) + starting latent (C) + 5
    # covariate channels. For example, latent C=16 gives 16+16+5=37 input channels.
    latent_channels = 3 # Or get dynamically from loaded autoencoder if needed/possible
    num_covariates = 5  # start_age, followup_age, sex, start_diag, follow_diag

    print("[INFO] Initializing channel-conditioned diffusion model...")
    diffusion = networks.init_latent_diffusion_channel_cond(
        ckpt_path=args.diff_ckpt,
        latent_channels=latent_channels,
        num_covariates=num_covariates
    ).to(DEVICE)
    print("[INFO] Diffusion model initialized.")


    seg_module = SegWrapper(
        input_size=(224, 256, 192),
        nrois=len(ROI_NAMES),
        tissue_channels=len(ROI_NAMES),
        segmentation_checkpoint=args.seg_ckpt
    ).to(DEVICE).eval()

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.0205
    )

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)
    # Use torch.amp GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Determine scaling factor (only needs to be done once)
    print("[INFO] Determining scaling factor...")
    with torch.no_grad():
        # more robust scaling factor
        print("[INFO] Determining scaling factor from a subset of the training data...")
        scale_factor = torch.tensor(1.0) # default sf
        if len(trainset) > 0:
            # Build a temporary loader to efficiently load a subset of samples.
            # shuffle=True gives a representative random subset.
            subset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

            # Collect samples until reaching the target count or exhausting the dataset.
            num_samples_to_check = min(1000, len(trainset))
            latents_to_check = []
            samples_collected = 0
            
            print(f"Calculating std from approx. {num_samples_to_check} samples...")
            with torch.no_grad():
                for batch in tqdm(subset_loader, total=len(subset_loader)):
                    # Ensure the key name matches the one used in the data-loading code.
                    # This key may need to be built dynamically per autoencoder ID.
                    latent_key = next((k for k in batch if key_follow in k), None)
                    if latent_key is None:
                        raise KeyError("Could not find a valid latent path key in the batch.")

                    latents_to_check.append(batch[latent_key].cpu())
                    samples_collected += len(batch[latent_key])
                    if samples_collected >= num_samples_to_check:
                        break
            
            if latents_to_check:
                all_latents = torch.cat(latents_to_check, dim=0)
                # Compute the standard deviation over the whole subset.
                std_dev = torch.std(all_latents)
                if std_dev > 1e-6:
                    scale_factor = 1.0 / std_dev
                print(f"Std dev of subset: {std_dev.item()}")
        else:
            raise ValueError("Training dataset is empty, cannot determine scale factor.")

    print(f"Robust scaling factor set to {scale_factor.item()}")
    wandb.log({"scale_factor": scale_factor.item()}, step=0) # Log initial scale factor

    # Use unified global step counter for wandb logging
    global_step = 0

    loaders = {'train': train_loader, 'valid': valid_loader}
    datasets = {'train': trainset, 'valid': validset}

    # --- Training Loop ---
    for epoch in range(args.n_epochs):
        print(f"\n--- Starting Epoch {epoch}/{args.n_epochs - 1} ---")
        for mode in loaders.keys():
            print(f"DEBUG: Starting epoch {epoch} in mode {mode} with {len(loaders[mode].dataset)} samples")
            loader = loaders[mode]

            if mode == 'train':
                diffusion.train() # Set diffusion model to train mode
                seg_module.eval() # Keep seg module in eval mode
            else:
                diffusion.eval()  # Set diffusion model to eval mode for validation
                seg_module.eval() # Keep seg module in eval mode

            epoch_loss_sum = 0.0
            epoch_noise_loss_sum = 0.0
            epoch_volume_loss_sum = 0.0
            volume_loss_steps = 0 # Count steps where volume loss was calculated

            progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} [{mode}]")

            for step, batch in progress_bar:
                # Use torch.amp.autocast
                with torch.amp.autocast(device_type=DEVICE.split(':')[0], enabled=torch.cuda.is_available()):
                    if mode == 'train':
                        optimizer.zero_grad(set_to_none=True)

                    # --- Get Data ---
                    starting_z_unscaled = batch[f'starting_latent_path_{latent_id}'].to(DEVICE)
                    followup_z_unscaled = batch[f'followup_latent_path_{latent_id}'].to(DEVICE)
                    # Fetch covariates directly
                    starting_age = batch['starting_age'].to(DEVICE)
                    followup_age = batch['followup_age'].to(DEVICE)
                    sex = batch['sex'].to(DEVICE)
                    starting_diagnosis = batch['starting_diagnosis'].to(DEVICE)
                    followup_diagnosis = batch['followup_diagnosis'].to(DEVICE)

                    n = starting_z_unscaled.shape[0]
                    latent_shape = starting_z_unscaled.shape[-3:] # D, H, W

                    # Scale latents
                    starting_z = starting_z_unscaled * scale_factor
                    followup_z = followup_z_unscaled * scale_factor # This is the target noisy prediction should match

                    # --- Prepare Input Channels ---
                    # Create spatial channels for covariates
                    start_age_channel = starting_age.view(n, 1, 1, 1, 1).expand(n, 1, *latent_shape)
                    followup_age_channel = followup_age.view(n, 1, 1, 1, 1).expand(n, 1, *latent_shape)
                    sex_channel = sex.view(n, 1, 1, 1, 1).expand(n, 1, *latent_shape)
                    start_diag_channel = starting_diagnosis.view(n, 1, 1, 1, 1).expand(n, 1, *latent_shape)
                    follow_diag_channel = followup_diagnosis.view(n, 1, 1, 1, 1).expand(n, 1, *latent_shape)

                    # Condition channels: starting_z + 5 covariates
                    # Shape: (N, C + 5, D, H, W)
                    condition_channels = torch.cat([
                        starting_z,
                        start_age_channel,
                        followup_age_channel,
                        sex_channel,
                        start_diag_channel,
                        follow_diag_channel
                    ], dim=1)

                    # --- Standard Diffusion Step ---
                    noise = torch.randn_like(followup_z).to(DEVICE) # Noise matches target latent shape
                    timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=DEVICE).long()
                    noisy_latents = scheduler.add_noise(original_samples=followup_z, noise=noise, timesteps=timesteps)

                    # Concatenate noisy latent and condition channels for model input
                    # Input shape: (N, C_latent + C_condition, D, H, W) -> (N, C + C + 5, D, H, W)
                    model_input = torch.cat([noisy_latents, condition_channels], dim=1)

                    # --- Forward Pass ---
                    # Ensure gradients are enabled only during training
                    with torch.set_grad_enabled(mode == 'train'):
                        # Predict noise (context=None)
                        noise_pred = diffusion(
                            x=model_input.float(),
                            timesteps=timesteps,
                        )
                        # Calculate noise prediction loss
                        noise_loss = F.mse_loss(noise.float(), noise_pred.float())
                        loss = noise_loss
                        current_vol_loss = torch.tensor(0.0, device=DEVICE) # Keep track for logging

                        # --- Volume Loss Calculation (only during training) ---
                        if args.use_volume_loss and mode == 'train' and \
                           (step + 1) % args.vol_loss_freq == 0 and global_step >= args.vol_loss_start_step:
                            try:
                                # Use only the first sample in the batch for efficiency
                                first_idx = 0
                                # Generate sample *with gradients enabled*
                                # Pass covariates individually to the sampling function
                                generated_image = generate_sample_for_volume_loss(
                                    diffusion=diffusion,
                                    autoencoder=autoencoder,
                                    starting_z=starting_z[first_idx:first_idx+1], # Pass scaled starting_z
                                    starting_age=starting_age[first_idx:first_idx+1],
                                    followup_age=followup_age[first_idx:first_idx+1],
                                    sex=sex[first_idx:first_idx+1],
                                    starting_diagnosis=starting_diagnosis[first_idx:first_idx+1],
                                    followup_diagnosis=followup_diagnosis[first_idx:first_idx+1],
                                    scale_factor=scale_factor,
                                    num_inference_steps=10,
                                )

                                with torch.no_grad():
                                    # 1. Get the original ground-truth image path.
                                    original_target_path = batch['followup_image_path'][first_idx]

                                    # 2. Build a path to the preprocessed image from the original path.
                                    target_dir = os.path.dirname(original_target_path)
                                    preprocessed_target_path = os.path.join(target_dir, 'repreprocess', 'resampled.nii.gz')

                                    # 3. Load and align the image using the preprocessed path.
                                    target_image = align_images_for_loss(
                                        generated_image,
                                        preprocessed_target_path,
                                        device=DEVICE
                                    )

                                vol_loss, full_logits_gen, full_logits_tgt = compute_volume_loss_differentiable(generated_image, target_image, seg_module)
                                vol_loss = vol_loss.to(DEVICE)
                                current_vol_loss = vol_loss # Store for logging
                                seg_module.train() if mode == 'train' else seg_module.eval() # Restore mode

                                # --- Save debug overlay images at selected epochs. ---
                                if mode == 'train' and epoch in [2,4,6,10,15,19]:

                                    with torch.no_grad():
                                        hard_gen = torch.softmax(full_logits_gen, dim=1).argmax(1).squeeze(0).cpu().numpy()
                                        hard_tgt = torch.softmax(full_logits_tgt, dim=1).argmax(1).squeeze(0).cpu().numpy()
                                    # Grayscale of the generated image.
                                    gen_np = generated_image.detach().cpu().squeeze().numpy()            # (D,H,W)
                                    tgt_np = target_image.detach().cpu().squeeze().numpy()

                                    # Create the directory and save.
                                    debug_save_dir = os.path.join(args.output_dir, 'debug_segmentation', f'epoch_{epoch}')
                                    os.makedirs(debug_save_dir, exist_ok=True)
                                    png_path = os.path.join(debug_save_dir, f'overlay_pairs_gs{global_step}.png')
                                    plot_image_and_mask_pair(gen_np, hard_gen,
                                                    tgt_np, hard_tgt,
                                                    png_path)
                                    print(f"[INFO] Saved overlay pairs -> {png_path}")

                                # Add volume loss to total loss
                                loss = noise_loss + args.volume_loss_weight * vol_loss
                                volume_loss_steps += 1
                                epoch_volume_loss_sum += vol_loss.item()

                            except Exception as e:
                                print(f"[ERROR] Failed during volume loss calculation: {e}")
                                import traceback
                                traceback.print_exc()
                                loss = noise_loss # Revert to noise loss only if volume loss fails
                        else:
                            loss = noise_loss

                # --- Backward Pass and Optimization (only during training) ---
                if mode == 'train':
                    # Use scaler for backward pass
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                # --- Logging ---
                # Log metrics using the unified global_step
                log_data = {f'{mode}/batch_mse': noise_loss.item()}
                if args.use_volume_loss and mode == 'train' and current_vol_loss.item() > 0:
                     log_data[f'{mode}/batch_volume_loss'] = current_vol_loss.item()

                wandb.log(log_data, step=global_step)

                epoch_loss_sum += loss.item()
                epoch_noise_loss_sum += noise_loss.item()

                progress_bar.set_postfix({
                    "loss": epoch_loss_sum / (step + 1),
                    "noise_loss": epoch_noise_loss_sum / (step + 1),
                    "vol_loss (avg)": epoch_volume_loss_sum / max(volume_loss_steps, 1) if volume_loss_steps > 0 else 0
                })

                # Increment global step after processing each batch
                global_step += 1

            # --- End of Epoch Logging ---
            avg_epoch_loss = epoch_loss_sum / len(loader)
            avg_epoch_noise_loss = epoch_noise_loss_sum / len(loader)
            avg_epoch_volume_loss = epoch_volume_loss_sum / max(volume_loss_steps, 1) if volume_loss_steps > 0 else 0

            epoch_log_data = {
                f'{mode}/epoch_loss': avg_epoch_loss,
                f'{mode}/epoch_noise_loss': avg_epoch_noise_loss,
            }
            if volume_loss_steps > 0:
                 epoch_log_data[f'{mode}/epoch_volume_loss'] = avg_epoch_volume_loss

            wandb.log(epoch_log_data, step=global_step) # Use global_step for epoch summary

            print(f"Epoch {epoch} [{mode}] Avg Loss: {avg_epoch_loss:.4f}, Avg Noise Loss: {avg_epoch_noise_loss:.4f}" +
                  (f", Avg Vol Loss: {avg_epoch_volume_loss:.4f}" if volume_loss_steps > 0 else ""))

            # --- Visualize Results ---
            # Call visualization function using the unified global_step
            images_to_wandb(
                epoch=epoch,
                mode=mode,
                autoencoder=autoencoder,
                diffusion=diffusion,
                scale_factor=scale_factor,
                dataset=datasets[mode],
                global_step=global_step, # Pass unified global_step
                latent_id=latent_id
            )

        if epoch == args.n_epochs - 1:
            # Save the last checkpoint ---
            savepath = os.path.join(args.output_dir, f'diffusion-channelcond-ep-{epoch}.pth')
            save_content = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scale_factor': scale_factor,
                'args': args # Save args for reproducibility
            }
            torch.save(save_content, savepath)
            print(f"Epoch {epoch} checkpoint saved to {savepath}")

    # --- End Training ---
    wandb.finish()
    print("Training finished.")