# ---------------------------------------------------------------------------
# Thin wrapper that adapts the WarpSeg tissue segmenter (MGA-Net) as a frozen
# anatomical-guidance teacher for AG-LDM.
#
# Only this wrapper is distributed here. The WarpSeg model definition
# (mga_net.py, mga_net_updated.py) and its pretrained checkpoint are NOT
# included; obtain them from the WarpSeg repository and place the two model
# files in this directory (scripts/training/segment_utils/):
#     https://github.com/BahramJafrasteh/WarpSeg
# ---------------------------------------------------------------------------

"""
diffusion_segmentation_module.py

This module wraps a pretrained PyTorch segmentation model (MGA_NET) to add an
auxiliary segmentation loss branch during diffusion-model training.

Main functionality:
    1. Normalize the input image by scaling intensities to [0, 1] using the
       global minimum and the 99.9th percentile.
    2. Resize the input image from its original size (e.g. 122x146x122) to the
       size required by the pretrained segmentation model (224x256x192) via
       trilinear interpolation.
    3. Feed the transformed image into the pretrained segmentation model
       (MGA_NET) to compute the probability output, then apply a simple
       normalization to that output.

Note: all operations are implemented with torch functions so that gradients
propagate correctly.
"""
import sys
import os

sys.path.append('../../')
sys.path.append(os.path.dirname(__file__)) # Add current directory to path

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from mga_net import MGA_NET
    from mga_net_updated import MultiStageNet
except ImportError as exc:
    raise ImportError(
        "WarpSeg model definition not found. This repository ships only the "
        "AG-LDM wrapper, not the WarpSeg model itself. Download mga_net.py and "
        "mga_net_updated.py from https://github.com/BahramJafrasteh/WarpSeg and "
        "place them in scripts/training/segment_utils/ (and pass the WarpSeg "
        "checkpoint via --seg_ckpt / --segment_ckpt)."
    ) from exc

class DiffusionSegmentationModule_2(nn.Module):
    def __init__(self, input_size=(224, 256, 192), nrois=6, tissue_channels=6, segmentation_checkpoint=None):
        """
        Args:
            input_size (tuple): Input size (D, H, W) required by the segmentation
                model, defaults to (224, 256, 192).
            nrois (int): Number of output segmentation channels (number of ROIs).
            tissue_channels (int): Number of tissue segmentation channels required
                by the new model.
            segmentation_checkpoint (str): If provided, path to pretrained weights
                to load.
        """
        super(DiffusionSegmentationModule_2, self).__init__()
        self.input_size = input_size

        # Use the new MultiStageNet
        self.seg_model = MultiStageNet(channels=nrois, tissue_channels=tissue_channels)

        if segmentation_checkpoint is not None:
            checkpoint = torch.load(segmentation_checkpoint, map_location='cuda:0')
            state_dict = checkpoint['model']

            # Handle possible key-name prefix issues.
            # First try loading directly.
            try:
                self.seg_model.load_state_dict(state_dict, strict=True)
            except:
                # On failure, try stripping the prefix.
                new_state_dict = {}
                for k, v in state_dict.items():
                    # Handle possible prefixes: seg_model., model.
                    if k.startswith('seg_model.'):
                        new_key = k.replace('seg_model.', '')
                    elif k.startswith('model.'):
                        new_key = k.replace('model.', '')
                    else:
                        new_key = k
                    new_state_dict[new_key] = v

                # Use the filtered state_dict, loading only matching weights.
                model_state = self.seg_model.state_dict()
                filtered_state = {
                    k: v for k, v in new_state_dict.items()
                    if k in model_state and model_state[k].shape == v.shape
                }
                model_state.update(filtered_state)
                self.seg_model.load_state_dict(model_state)

        # Freeze parameters
        for param in self.seg_model.parameters():
            param.requires_grad = False

    def normalize_image(self, img):
        """Keep the original normalization logic."""
        B = img.size(0)
        normalized = []
        for i in range(B):
            img_i = img[i, 0].float()
            q_low = torch.min(img_i)
            q_high = torch.quantile(img_i, 0.999)
            eps = 1e-6
            img_i = torch.clamp(img_i, q_low, q_high)
            norm_i = (img_i - q_low) / (q_high - q_low + eps)
            normalized.append(norm_i.unsqueeze(0))
        normalized = torch.stack(normalized, dim=0)
        return normalized

    def resize_to_target(self, img):
        """Keep the original resize logic."""
        return F.interpolate(img, size=self.input_size, mode='trilinear', align_corners=False)

    def forward(self, img):
        """
        The forward pass is unchanged, but now returns the output of the new model.
        """
        # normalize
        norm_img = self.normalize_image(img)
        # adjust size
        resized_img = self.resize_to_target(norm_img)

        # Get segmentation output
        seg_output = self.seg_model(resized_img)

        return seg_output



class DiffusionSegmentationModule(nn.Module):
    def __init__(self, input_size=(224, 256, 192), nrois=6, segmentation_checkpoint=None):
        """
        Args:
            input_size (tuple): Input size (D, H, W) required by the segmentation
                model, defaults to (224, 256, 192).
            nrois (int): Number of output segmentation channels (number of ROIs).
            segmentation_checkpoint (str): If provided, path to pretrained weights
                to load.
        """
        super(DiffusionSegmentationModule, self).__init__()
        self.input_size = input_size  # Target size (D, H, W)
        self.seg_model = MGA_NET(channels=nrois)
        if segmentation_checkpoint is not None:
            checkpoint = torch.load(segmentation_checkpoint, map_location='cuda:0')
            state_dict = checkpoint['model']
            # If keys carry the 'seg_model.' prefix, strip it.
            new_state_dict = {k.replace("seg_model.", ""): v for k, v in state_dict.items()}
            self.seg_model.load_state_dict(new_state_dict, strict=True)
        # Freeze the segmentation model parameters so they are not updated later.
        for param in self.seg_model.parameters():
            param.requires_grad = False

    def normalize_image(self, img):
        """
        Normalize the input image, scaling its intensity values to [0, 1].

        Approach:
            1. Compute the global minimum and 99.9th percentile (a robust upper
               bound) per sample.
            2. Clamp the image, then apply a linear normalization.

        Args:
            img (torch.Tensor): Input image with shape (B, 1, D, H, W).

        Returns:
            torch.Tensor: Normalized image with the same shape as the input.
        """
        B = img.size(0)
        normalized = []
        for i in range(B):
            # Assume the image is (1, D, H, W)
            img_i = img[i, 0].float()
            q_low = torch.min(img_i)
            q_high = torch.quantile(img_i, 0.999)
            eps = 1e-6
            # Clamp to keep values within [q_low, q_high]
            img_i = torch.clamp(img_i, q_low, q_high)
            norm_i = (img_i - q_low) / (q_high - q_low + eps)
            normalized.append(norm_i.unsqueeze(0))
        normalized = torch.stack(normalized, dim=0)  # (B, 1, D, H, W)
        return normalized

    def resize_to_target(self, img):
        """
        Resize the image to the target size using trilinear interpolation.

        Args:
            img (torch.Tensor): Input image with shape (B, 1, D, H, W).

        Returns:
            torch.Tensor: Resized image with shape (B, 1, target_D, target_H, target_W).
        """
        return F.interpolate(img, size=self.input_size, mode='trilinear', align_corners=False)

    def forward(self, img):
        """
        Forward pass:
            1. Normalize the input image.
            2. Resize the normalized image to the size required by the pretrained
               segmentation model.
            3. Feed the processed image into the segmentation model to obtain the
               segmentation probability output.
            4. Apply a simple normalization to the segmentation output (normalized
               to [0, 1] over the spatial extent).

        Args:
            img (torch.Tensor): Input image with shape (B, 1, d, h, w), where
                d, h, w are the diffusion model output dimensions.

        Returns:
            torch.Tensor: Segmentation probability output whose shape is determined
                by MGA_NET.
        """
        # normalize
        norm_img = self.normalize_image(img)
        # adjust size
        resized_img = self.resize_to_target(norm_img)

        seg_output = self.seg_model(resized_img)

        min_vals = torch.amin(seg_output, dim=[2, 3, 4], keepdim=True)
        max_vals = torch.amax(seg_output, dim=[2, 3, 4], keepdim=True)
        seg_output = (seg_output - min_vals) / (max_vals - min_vals + 1e-6)
        return seg_output
