"""Stage 1: fine-tune the KL autoencoder (AutoencoderKL) with a spectral-norm patch discriminator (adversarial + feature-matching + KL + perceptual + L1 losses) plus an anatomical-consistency loss from a frozen tissue-segmentation teacher."""
import os
import argparse
import warnings

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm import tqdm
from monai import transforms
from monai.utils import set_determinism
from monai.losses import DiceCELoss

from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from generative.losses import PerceptualLoss, PatchAdversarialLoss

# --- Project imports (brlp and segment_utils must be on the Python path) ---
from brlp import const
from brlp import utils
from brlp import (
    KLDivergenceLoss, GradientAccumulation,
    init_autoencoder, init_patch_discriminator_spectral,
    get_dataset_from_pd
)
from segment_utils.segmentor_module import DiffusionSegmentationModule_2

# --- Patch torch.load to default weights_only=False for compatibility with older caches ---
_orig_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = patched_torch_load

set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Segmentation model wrapper (kept consistent with the diffusion training setup) ---
class SegWrapper(DiffusionSegmentationModule_2):
    def forward(self, img, *, return_logits=True):
        logits = super().forward(img)
        logits = torch.nn.functional.interpolate(logits,
                                    size=(122, 146, 122), # ensure a consistent output size
                                    mode='trilinear',
                                    align_corners=False)
        return logits if return_logits else torch.softmax(logits, dim=1)

def log_reconstructions_to_wandb(original_img, recon_img, global_step):
    """Extract central sagittal, coronal, and axial slices from the 3D volume, place original and reconstruction side by side, and log them to wandb."""
    original_img = original_img.detach().cpu().numpy().squeeze()
    recon_img = recon_img.detach().cpu().numpy().squeeze()

    mid_d, mid_h, mid_w = (original_img.shape[0] // 2, original_img.shape[1] // 2, original_img.shape[2] // 2)

    orig_sag = original_img[mid_d, :, :]; recon_sag = recon_img[mid_d, :, :]
    orig_cor = original_img[:, mid_h, :]; recon_cor = recon_img[:, mid_h, :]
    orig_axi = original_img[:, :, mid_w]; recon_axi = recon_img[:, :, mid_w]

    comp_sag = np.hstack((orig_sag, recon_sag))
    comp_cor = np.hstack((orig_cor, recon_cor))
    comp_axi = np.hstack((orig_axi, recon_axi))

    wandb.log({
        "Reconstruction/Sagittal": wandb.Image(comp_sag, caption=f"Sagittal | Step: {global_step}"),
        "Reconstruction/Coronal": wandb.Image(comp_cor, caption=f"Coronal | Step: {global_step}"),
        "Reconstruction/Axial": wandb.Image(comp_axi, caption=f"Axial | Step: {global_step}")
    }, step=global_step)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Fine-tune Autoencoder with anatomical guidance.")
    # --- Core arguments ---
    parser.add_argument('--dataset_csv',    required=True, type=str)
    parser.add_argument('--cache_dir',      required=True, type=str)
    parser.add_argument('--output_dir',     required=True, type=str)
    parser.add_argument('--aekl_ckpt',      default=None,  type=str, help="Path to pre-trained autoencoder for fine-tuning.")
    parser.add_argument('--disc_ckpt',      default=None,  type=str, help="Path to pre-trained discriminator checkpoint.")
    parser.add_argument('--seg_ckpt',       required=True, type=str, help="Path to the pre-trained 'expert' segmentation model.")

    # --- Training hyperparameters ---
    parser.add_argument('--n_epochs',       default=10,    type=int)
    parser.add_argument('--max_batch_size', default=2,     type=int)
    parser.add_argument('--batch_size',     default=16,    type=int)
    parser.add_argument('--lr',             default=5e-6,  type=float, help="Learning rate for fine-tuning.")
    parser.add_argument('--num_workers',    default=8,     type=int)

    # --- Loss-weight hyperparameters ---
    parser.add_argument('--adv_weight', type=float, default=0.1, help="Weight for the adversarial loss.")
    parser.add_argument('--perceptual_weight', type=float, default=0.08, help="Weight for the perceptual loss.")
    parser.add_argument('--kl_weight', type=float, default=1e-6, help="Weight for the KL regularization loss.")
    parser.add_argument('--fm_loss_weight', type=float, default=0, help="Weight for the feature matching loss.")
    parser.add_argument('--anatomical_loss_weight', type=float, default=1.0, help="Weight for the anatomical consistency loss.")
    
    # --- wandb arguments ---
    parser.add_argument('--wandb_project', default='brlp_autoencoder', type=str)
    parser.add_argument('--wandb_entity', default=None, type=str)
    parser.add_argument('--wandb_name', default=None, type=str)
    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name if args.wandb_name else f"aekl-finetune-{wandb.util.generate_id()}",
        config=vars(args)
    )
    os.makedirs(args.output_dir, exist_ok=True)
    
    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    train_df = dataset_df[dataset_df.split == 'train']
    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(dataset=trainset,
                              num_workers=args.num_workers,
                              batch_size=args.max_batch_size,
                              shuffle=True,
                              persistent_workers=args.num_workers > 0,
                              pin_memory=True)

    # --- Initialize all models ---
    autoencoder   = init_autoencoder(args.aekl_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator_spectral(args.disc_ckpt).to(DEVICE)
    # Load the frozen "expert" segmentation teacher model
    seg_module = SegWrapper(
        input_size=(224, 256, 192), nrois=6, tissue_channels=6,
        segmentation_checkpoint=args.seg_ckpt
    ).to(DEVICE).eval()
    for param in seg_module.parameters():
        param.requires_grad = False

    # --- Initialize all loss functions ---
    l1_loss_fn = L1Loss()
    kl_loss_fn = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")
    # Anatomical consistency loss
    seg_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).to(DEVICE)

    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    
    # Use the recommended torch.amp.GradScaler API
    scaler_g = torch.amp.GradScaler('cuda')
    scaler_d = torch.amp.GradScaler('cuda')
    
    gradacc_g = GradientAccumulation(actual_batch_size=args.max_batch_size, expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader), optimizer=optimizer_g, grad_scaler=scaler_g)
    gradacc_d = GradientAccumulation(actual_batch_size=args.max_batch_size, expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader), optimizer=optimizer_d, grad_scaler=scaler_d)

    global_step = 0
    for epoch in range(args.n_epochs):
        autoencoder.train()
        discriminator.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:
            images = batch["image"].to(DEVICE)

            # --- Train the generator (Autoencoder) ---
            optimizer_g.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                reconstruction, z_mu, z_sigma = autoencoder(images)

                # 1. Reconstruction, KL, and perceptual losses (standard terms)
                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                kld_loss = args.kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = args.perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())

                # 2. Adversarial and feature-matching losses
                outputs_fake = discriminator(reconstruction.contiguous().float())
                gen_loss = args.adv_weight * adv_loss_fn(outputs_fake[-1], target_is_real=True, for_discriminator=False)
                
                loss_fm = 0.0
                with torch.no_grad():
                    outputs_real = discriminator(images.contiguous().float())
                for feat_f, feat_r in zip(outputs_fake[:-1], outputs_real[:-1]):
                    loss_fm += l1_loss_fn(feat_f, feat_r)
                fm_loss = args.fm_loss_weight * (loss_fm / len(outputs_fake[:-1]))
                
                # 3. Anatomical consistency loss
                seg_logits_pred = seg_module(reconstruction.contiguous().float())
                with torch.no_grad():
                    seg_logits_true = seg_module(images.contiguous().float())
                
                target_labels = torch.argmax(seg_logits_true, dim=1, keepdim=True)
                anatomical_loss = args.anatomical_loss_weight * seg_loss_fn(seg_logits_pred, target_labels)

                # 4. Combine all generator losses
                loss_g = rec_loss + kld_loss + per_loss + gen_loss + fm_loss + anatomical_loss
                
            gradacc_g.step(loss_g, step)

            # --- Train the discriminator ---
            optimizer_d.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                loss_d = args.adv_weight * discriminator_loss

            gradacc_d.step(loss_d, step)

            # --- wandb logging ---
            if global_step % 10 == 0:
                wandb.log({
                    'epoch': epoch,
                    'Generator/total_loss': loss_g.item(),
                    'Generator/reconstruction_loss': rec_loss.item(),
                    'Generator/perceptual_loss': per_loss.item(),
                    'Generator/adversarial_loss': gen_loss.item(),
                    'Generator/kl_regularization': kld_loss.item(),
                    'Generator/feature_matching_loss': fm_loss.item(),
                    'Generator/anatomical_loss': anatomical_loss.item(),
                    'Discriminator/total_loss': loss_d.item()
                }, step=global_step)

            if global_step % 100 == 0:
                log_reconstructions_to_wandb(images[0], reconstruction[0], global_step)

            global_step += 1

        # Save models at the end of each epoch
        torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f'discriminator-ep-{epoch}.pth'))
        torch.save(autoencoder.state_dict(),   os.path.join(args.output_dir, f'autoencoder-ep-{epoch}.pth'))
    
    wandb.finish()
    print("Fine-tuning finished.")