import os
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from generative.networks.nets import (
    AutoencoderKL, 
    PatchDiscriminator,
    DiffusionModelUNet, 
    ControlNet
)


def load_if(checkpoints_path: Optional[str], network: nn.Module) -> nn.Module:
    """
    Load pretrained weights if available.

    Args:
        checkpoints_path (Optional[str]): path of the checkpoints
        network (nn.Module): the neural network to initialize 

    Returns:
        nn.Module: the initialized neural network
    """
    if checkpoints_path is not None:
        assert os.path.exists(checkpoints_path), 'Invalid path'
        device = next(network.parameters()).device # Use the same device as the model
        checkpoint = torch.load(checkpoints_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # New format with additional training info
                network.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                # Old format with direct state dict
                network.load_state_dict(checkpoint)
        else:
            # Fallback: assume it's a direct state dict
            network.load_state_dict(checkpoint)

    return network


def init_autoencoder(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the KL autoencoder (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the KL autoencoder
    """
    autoencoder = AutoencoderKL(spatial_dims=3, 
                                in_channels=1, 
                                out_channels=1, 
                                latent_channels=3,
                                num_channels=(64, 128, 128, 128),
                                num_res_blocks=2, 
                                norm_num_groups=32,
                                norm_eps=1e-06,
                                attention_levels=(False, False, False, False), 
                                with_decoder_nonlocal_attn=False, 
                                with_encoder_nonlocal_attn=False)
    return load_if(checkpoints_path, autoencoder)


def init_patch_discriminator(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the patch discriminator (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the patch discriminator
    """
    patch_discriminator = PatchDiscriminator(spatial_dims=3, 
                                             num_layers_d=3, 
                                             num_channels=32, 
                                             in_channels=1, 
                                             out_channels=1)
    return load_if(checkpoints_path, patch_discriminator)



def apply_spectral_norm_to_conv_layers(module: nn.Module):
    """
    Recursively traverse all submodules of a module and apply spectral
    normalization to every Conv3d layer.
    """
    for name, child_module in module.named_children():
        # If the child is a Conv3d layer, replace it with its spectral-normalized version
        if isinstance(child_module, nn.Conv3d):
            setattr(module, name, spectral_norm(child_module))
        # If the child still has submodules, recurse into it
        elif list(child_module.children()):
            apply_spectral_norm_to_conv_layers(child_module)
    return module


def init_patch_discriminator_spectral(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: a patch discriminator with spectral normalization.
    """
    patch_discriminator = PatchDiscriminator(
        spatial_dims=3, 
        num_layers_d=3, 
        num_channels=32, 
        in_channels=1, 
        out_channels=1
    )

    print("Applying spectral normalization to the discriminator...")
    patch_discriminator = apply_spectral_norm_to_conv_layers(patch_discriminator)

    return load_if(checkpoints_path, patch_discriminator)


def init_latent_diffusion(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the UNet from the diffusion model (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the UNet
    """
    latent_diffusion = DiffusionModelUNet(spatial_dims=3, 
                                          in_channels=3, 
                                          out_channels=3, 
                                          num_res_blocks=2, 
                                          num_channels=(256, 512, 768), 
                                          attention_levels=(False, True, True), 
                                          norm_num_groups=32, 
                                          norm_eps=1e-6, 
                                          resblock_updown=True, 
                                          num_head_channels=(0, 512, 768), 
                                          transformer_num_layers=1,
                                          with_conditioning=True,
                                          cross_attention_dim=8,
                                          num_class_embeds=None, 
                                          upcast_attention=True, 
                                          use_flash_attention=False)
    return load_if(checkpoints_path, latent_diffusion)

def init_latent_diffusion_for_paired(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Initialize the diffusion model for paired inputs.
    This model handles paired data by concatenating the source and target
    latent representations directly along the input channels.
    
    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.
        
    Returns:
        nn.Module: the UNet for paired inputs
    """
    # Define the diffusion UNet with 7 input channels (3 target latent + 3 source latent + 1 age)
    latent_diffusion = DiffusionModelUNet(
        spatial_dims=3, 
        in_channels=7,                   # 3 (target) + 3 (source) + 1 (age)
        out_channels=3,                  # output channels unchanged
        num_res_blocks=2, 
        num_channels=(256, 512, 768),    # same architecture as the original model
        attention_levels=(False, True, True), 
        norm_num_groups=32, 
        norm_eps=1e-6, 
        resblock_updown=True, 
        num_head_channels=(0, 512, 768), 
        transformer_num_layers=1,
        with_conditioning=True,
        cross_attention_dim=3,           # context dimension (age, sex, diagnosis)
        num_class_embeds=None, 
        upcast_attention=True, 
        use_flash_attention=False
    )
    
    return load_if(checkpoints_path, latent_diffusion)
    

def init_latent_diffusion_for_paired_cross_4(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Initialize the diffusion model for paired inputs.
    This model handles paired data by concatenating the source and target
    latent representations directly along the input channels.
    
    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.
        
    Returns:
        nn.Module: the UNet for paired inputs
    """
    # Define the diffusion UNet with 7 input channels (3 target latent + 3 source latent + 1 age)
    latent_diffusion = DiffusionModelUNet(
        spatial_dims=3, 
        in_channels=7,                   # 3 (target) + 3 (source) + 1 (age)
        out_channels=3,                  # output channels unchanged
        num_res_blocks=2, 
        num_channels=(256, 512, 768),    # same architecture as the original model
        attention_levels=(False, True, True), 
        norm_num_groups=32, 
        norm_eps=1e-6, 
        resblock_updown=True, 
        num_head_channels=(0, 512, 768), 
        transformer_num_layers=1,
        with_conditioning=True,
        cross_attention_dim=4,           # context dimension (age, sex, diagnosis)
        num_class_embeds=None, 
        upcast_attention=True, 
        use_flash_attention=False
    )
    
    return load_if(checkpoints_path, latent_diffusion)




# --- NEW Diffusion Model for Channel Conditioning ---
def init_latent_diffusion_channel_cond(
    ckpt_path: Optional[str] = None,
    latent_channels: int = 3,
    num_covariates: int = 5 # start_age, followup_age, sex, start_diag, follow_diag
) -> nn.Module:
    """
    Initializes the Diffusion Model UNet designed for paired input where conditioning
    (starting latent + covariates) is provided via input channels.

    Args:
        ckpt_path (Optional[str], optional): Path to the checkpoint to load weights from. Defaults to None.
        latent_channels (int, optional): Number of channels in the autoencoder's latent space. Defaults to 3.
        num_covariates (int, optional): Number of covariate channels being concatenated (e.g., ages, sex, diagnoses). Defaults to 5.

    Returns:
        nn.Module: The initialized UNet for channel-based conditioning.
    """
    # Calculate total input channels:
    # noisy target latent (C) + starting latent (C) + covariates (num_covariates)
    in_channels = latent_channels + latent_channels + num_covariates
    out_channels = latent_channels # Output should match the target latent channels

    print(f"Initializing DiffusionModelUNet for channel conditioning:")
    print(f"  Latent channels (C): {latent_channels}")
    print(f"  Number of covariate channels: {num_covariates}")
    print(f"  Total input channels (noisy C + starting C + covariates): {in_channels}")
    print(f"  Output channels: {out_channels}")

    # Define the UNet. Architecture details (num_channels, attention_levels, etc.)
    # are kept the same as the previous models for consistency, adjust if needed.
    latent_diffusion = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=in_channels,         # Calculated total input channels
        out_channels=out_channels,       # Output matches latent channels
        num_res_blocks=2,
        num_channels=(256, 512, 768),    # Keep consistent with previous models
        attention_levels=(False, True, True), # Keep consistent
        norm_num_groups=32,
        norm_eps=1e-6,
        resblock_updown=True,
        num_head_channels=(0, 512, 768), # Keep consistent
        transformer_num_layers=1,
        num_class_embeds=None,
        upcast_attention=True,
        use_flash_attention=False
    )

    # Print model summary
    total_params = sum(p.numel() for p in latent_diffusion.parameters() if p.requires_grad)
    print(f"Initialized Diffusion UNet (Channel Cond.) - Total Params: {total_params:,}")

    # Load weights if checkpoint path is provided
    return load_if(ckpt_path, latent_diffusion)






def init_controlnet(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the ControlNet (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the ControlNet
    """
    controlnet = ControlNet(spatial_dims=3, 
                            in_channels=3,
                            num_res_blocks=2, 
                            num_channels=(256, 512, 768), 
                            attention_levels=(False, True, True), 
                            norm_num_groups=32, 
                            norm_eps=1e-6, 
                            resblock_updown=True, 
                            num_head_channels=(0, 512, 768), 
                            transformer_num_layers=1, 
                            with_conditioning=True,
                            cross_attention_dim=8, 
                            num_class_embeds=None, 
                            upcast_attention=True, 
                            use_flash_attention=False, 
                            conditioning_embedding_in_channels=4,  
                            conditioning_embedding_num_channels=(256,))
    return load_if(checkpoints_path, controlnet)


def init_latent_diffusion_with_guidance(
    ckpt_path: Optional[str] = None,
    latent_channels: int = 4,  # number of latent channels in the autoencoder
    num_covariates: int = 5,
    context_dim: int = 512  # dimension of the cross-attention conditioning context
) -> nn.Module:
    """
    Initialize a Diffusion U-Net that supports hybrid conditioning.
    It combines channel concatenation (for the base conditioning) with
    cross-attention (for anatomical guidance).

    Args:
        ckpt_path (Optional[str]): checkpoint path.
        latent_channels (int): number of latent channels of the autoencoder.
        num_covariates (int): number of covariates concatenated along the input channels.
        context_dim (int): dimension of the cross-attention context vector.
                           This must match the encoded segmentation feature dimension.

    Returns:
        nn.Module: The initialized UNet for hybrid conditioning.
    """
    # Compute the number of input channels for the channel-wise conditioning:
    # noisy_latents (latent_channels) + starting_latent (latent_channels) + covariates (num_covariates)
    in_channels = latent_channels + latent_channels + num_covariates
    out_channels = latent_channels

    print(f"Initializing DiffusionModelUNet for HYBRID conditioning (Channel Concat + Cross-Attention):")
    print(f"  - Channel-wise in_channels (noisy {latent_channels} + starting {latent_channels} + covariates {num_covariates}): {in_channels}")
    print(f"  - Cross-Attention context_dim (for anatomical guidance): {context_dim}")
    print(f"  - Output channels: {out_channels}")

    # Define the U-Net; the key is to enable and configure cross_attention_dim
    latent_diffusion = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=in_channels,         # channel-wise conditioning input
        out_channels=out_channels,       # number of output channels
        num_res_blocks=2,
        num_channels=(256, 512, 768),    # keep the same architecture as the other models
        attention_levels=(False, True, True), 
        norm_num_groups=32,
        norm_eps=1e-6,
        resblock_updown=True,
        num_head_channels=(0, 512, 768), 
        transformer_num_layers=1,
        
        # --- enable and configure cross-attention ---
        with_conditioning=True,          # must be True to enable time embeddings and cross-attention
        cross_attention_dim=context_dim, # set to the guidance conditioning dimension
        # ------------------------------------

        num_class_embeds=None,
        upcast_attention=True,
        use_flash_attention=False
    )

    total_params = sum(p.numel() for p in latent_diffusion.parameters() if p.requires_grad)
    print(f"Initialized Diffusion UNet (Hybrid Cond.) - Total Params: {total_params:,}")

    return load_if(ckpt_path, latent_diffusion)


def init_latent_diffusion_with_guidance_3(
    ckpt_path: Optional[str] = None,
    latent_channels: int = 3,  # number of latent channels in the autoencoder
    num_covariates: int = 5,
    context_dim: int = 512  # dimension of the cross-attention conditioning context
) -> nn.Module:
    """
    Initialize a Diffusion U-Net that supports hybrid conditioning.
    It combines channel concatenation (for the base conditioning) with
    cross-attention (for anatomical guidance).

    Args:
        ckpt_path (Optional[str]): checkpoint path.
        latent_channels (int): number of latent channels of the autoencoder.
        num_covariates (int): number of covariates concatenated along the input channels.
        context_dim (int): dimension of the cross-attention context vector.
                           This must match the encoded segmentation feature dimension.

    Returns:
        nn.Module: The initialized UNet for hybrid conditioning.
    """
    # Compute the number of input channels for the channel-wise conditioning:
    # noisy_latents (latent_channels) + starting_latent (latent_channels) + covariates (num_covariates)
    in_channels = latent_channels + latent_channels + num_covariates
    out_channels = latent_channels

    print(f"Initializing DiffusionModelUNet for HYBRID conditioning (Channel Concat + Cross-Attention):")
    print(f"  - Channel-wise in_channels (noisy {latent_channels} + starting {latent_channels} + covariates {num_covariates}): {in_channels}")
    print(f"  - Cross-Attention context_dim (for anatomical guidance): {context_dim}")
    print(f"  - Output channels: {out_channels}")

    # Define the U-Net; the key is to enable and configure cross_attention_dim
    latent_diffusion = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=in_channels,         # channel-wise conditioning input
        out_channels=out_channels,       # number of output channels
        num_res_blocks=2,
        num_channels=(256, 512, 768),    # keep the same architecture as the other models
        attention_levels=(False, True, True), 
        norm_num_groups=32,
        norm_eps=1e-6,
        resblock_updown=True,
        num_head_channels=(0, 512, 768), 
        transformer_num_layers=1,
        
        # --- enable and configure cross-attention ---
        with_conditioning=True,          # must be True to enable time embeddings and cross-attention
        cross_attention_dim=context_dim, # set to the guidance conditioning dimension
        # ------------------------------------

        num_class_embeds=None,
        upcast_attention=True,
        use_flash_attention=False
    )

    total_params = sum(p.numel() for p in latent_diffusion.parameters() if p.requires_grad)
    print(f"Initialized Diffusion UNet (Hybrid Cond.) - Total Params: {total_params:,}")

    return load_if(ckpt_path, latent_diffusion)