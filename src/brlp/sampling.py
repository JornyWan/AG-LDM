import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from generative.networks.schedulers import DDIMScheduler
from tqdm import tqdm
import numpy as np
from . import utils
from . import const


@torch.no_grad()
def sample_using_imgcond_diffusion(
    autoencoder: nn.Module, 
    diffusion: nn.Module, 
    starting_z: torch.Tensor,
    starting_a: torch.Tensor, 
    context: torch.Tensor, 
    device: str,
    scale_factor: int = 1,
    average_over_n: int = 1,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015, 
    beta_end: float = 0.0205, 
    verbose: bool = True
) -> torch.Tensor:
    """
    Sampling brain MRIs using the paired diffusion model with channel concatenation.

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet with 7 input channels (3+3+1)
        starting_z (torch.Tensor): the latent from the MRI of the starting visit 
        starting_a (torch.Tensor): the starting age
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        average_over_n (int, optional): LAS parameter m. Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.

    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    scheduler = DDIMScheduler(
        num_train_timesteps=num_training_steps,
        schedule=schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False
    )

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    
    # Log shape of initial latent if verbose
    if verbose:
        utils.log_tensor_shape(starting_z, "Initial latent (starting_z)")
    
    # Prepare starting latent
    starting_z = starting_z.unsqueeze(0).to(device) if starting_z.dim() == 4 else starting_z.to(device)
    
    # Prepare age as spatial condition
    if isinstance(starting_a, (float, int)):
        starting_a = torch.tensor([starting_a]).to(device)
    elif isinstance(starting_a, torch.Tensor):
        starting_a = starting_a.to(device)
        
    # Create the age channel with same spatial dimensions as starting_z
    concatenating_age = starting_a.view(1, 1, 1, 1, 1).expand(1, 1, *starting_z.shape[-3:]).to(device)
    
    # The context vector contains demographic information
    if context.dim() == 1:
        context = context.unsqueeze(0)
    context = context.unsqueeze(0).to(device) if context.dim() == 2 else context.to(device)
    
    # If performing LAS, repeat inputs for parallel diffusion processes
    if average_over_n > 1:
        context = context.repeat(average_over_n, 1, 1)
        starting_z = starting_z.repeat(average_over_n, 1, 1, 1, 1)
        concatenating_age = concatenating_age.repeat(average_over_n, 1, 1, 1, 1)
    
    # Initialize with random noise for the target latent part
    z = torch.randn(average_over_n, 3, *starting_z.shape[-3:]).to(device)
    
    # Progress through denoising steps
    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):
                # Prepare timestep
                timestep = torch.tensor([t]).repeat(average_over_n).to(device)
                
                # Create the concatenated input
                # First prepare the condition part (starting_z + age)
                condition_input = torch.cat([starting_z, concatenating_age], dim=1)
                
                # Then concatenate with the noisy target latent
                model_input = torch.cat([z, condition_input], dim=1)
                
                # Predict noise using the diffusion model
                noise_pred = diffusion(
                    x=model_input.float(), 
                    timesteps=timestep, 
                    context=context.float()
                )
                
                # Perform denoising step
                z, _ = scheduler.step(noise_pred, t, z)
    
    # Apply LAS if multiple samples generated
    if verbose and average_over_n > 1:
        utils.log_tensor_shape(z, "Final denoised latent (before averaging)")
    
    z = (z / scale_factor).sum(axis=0) / average_over_n if average_over_n > 1 else z / scale_factor
    
    if verbose and average_over_n > 1:
        utils.log_tensor_shape(z, "Final denoised latent (after averaging)")
    
    # Decode the latent
    z = utils.to_vae_latent_trick(z.squeeze(0).cpu())
    x = autoencoder.decode_stage_2_outputs(z.unsqueeze(0).to(device))
    x = utils.to_mni_space_1p5mm_trick(x.squeeze(0).cpu()).squeeze(0)
    return x



@torch.no_grad()
def sample_using_channel_cond(
    autoencoder: nn.Module,
    diffusion: nn.Module,      # UNet accepting channel conditions (context=None)
    starting_z: torch.Tensor,  # Shape (C, D, H, W) or (N, C, D, H, W); must already be scaled
    starting_age: torch.Tensor, # Shape (,) or (N,)
    # Covariate arguments (replace the context vector):
    followup_age: torch.Tensor, # Shape (,) or (N,)
    sex: torch.Tensor,          # Shape (,) or (N,)
    starting_diagnosis: torch.Tensor, # Shape (,) or (N,)
    followup_diagnosis: torch.Tensor, # Shape (,) or (N,)
    # --- End covariate args ---
    device: str,
    scale_factor: int = 1, # VAE scale factor (applied before decoding)
    average_over_n: int = 1,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015,
    beta_end: float = 0.0205,
    verbose: bool = True
) -> torch.Tensor:
    """
    Sampling brain MRIs using the paired diffusion model with channel concatenation
    for starting latent and ALL covariates. Structure similar to original sample_using_imgcond_diffusion.

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet configured for channel conditioning (context=None)
        starting_z (torch.Tensor): the latent from the MRI of the starting visit (ALREADY SCALED)
        starting_age (torch.Tensor): the starting age
        followup_age (torch.Tensor): the target followup age
        sex (torch.Tensor): the sex covariate
        starting_diagnosis (torch.Tensor): the starting diagnosis covariate
        followup_diagnosis (torch.Tensor): the followup diagnosis covariate
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): VAE scale factor (used for scaling back). Defaults to 1.
        average_over_n (int, optional): LAS parameter m. Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.

    Returns:
        torch.Tensor: the inferred follow-up MRI, shape (D, H, W) on CPU.
    """
    diffusion.eval() # Ensure diffusion model is in eval mode
    autoencoder.eval() # Ensure autoencoder is in eval mode

    scheduler = DDIMScheduler(
        num_train_timesteps=num_training_steps,
        schedule=schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False
    )
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # --- Prepare Inputs (Similar to original, but handles more covariates) ---
    if verbose:
        utils.log_tensor_shape(starting_z, "Initial latent (starting_z - scaled)")

    # Ensure starting_z has batch dimension
    if starting_z.dim() == 4: # (C, D, H, W) -> (1, C, D, H, W)
        starting_z = starting_z.unsqueeze(0)
    starting_z = starting_z.to(device) # Already scaled by caller
    n = starting_z.shape[0]
    latent_channels = starting_z.shape[1]
    latent_shape = starting_z.shape[2:] # (D, H, W)

    # Prepare all covariates (age + others)
    covariates = {
        "starting_age": starting_age,
        "followup_age": followup_age,
        "sex": sex,
        "starting_diagnosis": starting_diagnosis,
        "followup_diagnosis": followup_diagnosis,
    }
    processed_covariates = {}
    covariate_channels_list = [] # To store spatial channels

    for name, cov in covariates.items():
        # Ensure tensor, correct device, and add batch dim if needed
        if isinstance(cov, (float, int, np.number)): # Handle single numbers
             cov = torch.tensor([cov] * n, device=device)
        elif isinstance(cov, torch.Tensor):
             if cov.dim() == 0: # Scalar tensor
                 cov = cov.unsqueeze(0).repeat(n)
             elif cov.dim() == 1 and cov.shape[0] != n:
                 raise ValueError(f"Covariate '{name}' batch size {cov.shape[0]} doesn't match starting_z batch size {n}")
             elif cov.dim() > 1:
                 raise ValueError(f"Covariate '{name}' has too many dimensions: {cov.dim()}")
             cov = cov.to(device) # Ensure device
        else:
             raise TypeError(f"Unsupported type for covariate '{name}': {type(cov)}")

        processed_covariates[name] = cov.float() # Ensure float

        # Create spatial channel: (N,) -> (N, 1, 1, 1, 1) -> (N, 1, D, H, W)
        cov_channel = processed_covariates[name].view(n, 1, 1, 1, 1).expand(n, 1, *latent_shape)
        covariate_channels_list.append(cov_channel)
        if verbose:
             utils.log_tensor_shape(cov_channel, f"Spatial channel for {name}")

    # --- Concatenate Condition Channels ---
    # Condition = starting_z + all covariate channels
    condition_channels_tensor = torch.cat([starting_z] + covariate_channels_list, dim=1)
    if verbose:
         utils.log_tensor_shape(condition_channels_tensor, "Concatenated condition channels")

    # Repeat for Latent Averaging Sampling (LAS) if needed
    if average_over_n > 1:
        condition_channels_tensor = condition_channels_tensor.repeat_interleave(average_over_n, dim=0)
        n_effective = n * average_over_n
        print(f"LAS enabled: Repeating conditions {average_over_n} times. Effective batch size: {n_effective}")
    else:
        n_effective = n

    # Initialize with random noise for the target latent part
    # Noise matches the *output* latent shape (N_eff, C_out, D, H, W)
    z = torch.randn(n_effective, latent_channels, *latent_shape).to(device)

    # --- Denoising Loop ---
    progress_bar = tqdm(scheduler.timesteps, desc="Sampling", disable=not verbose)
    for t in progress_bar:
        # Prepare timestep for the effective batch size
        timestep = torch.tensor([t] * n_effective, device=device).long()

        # Use autocast like original
        with autocast(enabled=torch.cuda.is_available()): # Check if cuda is available
            # Concatenate current noisy latent 'z' with the condition channels
            # Input shape: (N_eff, C_out + C_condition, D, H, W)
            model_input = torch.cat([z, condition_channels_tensor], dim=1)

            # Predict noise using the diffusion model (no cross-attention context)
            noise_pred = diffusion(
                x=model_input.float(),
                timesteps=timestep,
                context=None  # channel conditioning uses no cross-attention context
            )

            # Perform denoising step
            z, _ = scheduler.step(noise_pred, t, z)
    # --- End Denoising Loop ---

    # --- Post-processing and Decoding (Keep original logic) ---
    if verbose and average_over_n > 1:
        utils.log_tensor_shape(z, "Final denoised latent (before averaging)")

    # Apply LAS averaging if needed
    if average_over_n > 1:
        # Reshape z: (N * avg_n, C, D, H, W) -> (N, avg_n, C, D, H, W)
        z = z.view(n, average_over_n, latent_channels, *latent_shape)
        # Average over the avg_n dimension
        z = z.mean(dim=1) # Shape: (N, C, D, H, W)

    # Scale back before decoding (original logic)
    z = z / scale_factor

    if verbose:
        utils.log_tensor_shape(z, "Final denoised latent (after averaging/scaling)")

    # Decode the latent (original logic, assuming N=1 was the primary use case)
    if z.shape[0] > 1:
        print(f"[WARN] Sampling function produced batch size {z.shape[0]} > 1, but original return logic assumes N=1. Decoding only the first sample.")
    z_single = z[0] # Take the first sample if batch > 1

    # Apply original VAE and MNI tricks
    z_processed = utils.to_vae_latent_trick(z_single.cpu()) # Original trick expects CPU tensor?
    x_decoded = autoencoder.decode_stage_2_outputs(z_processed.unsqueeze(0).to(device)) # Add batch dim for decoder
    x_final = utils.to_mni_space_1p5mm_trick(x_decoded.squeeze(0).cpu()).squeeze(0) # Original trick expects CPU, squeezes batch/channel

    if verbose:
        utils.log_tensor_shape(x_final, "Final decoded image (output)")

    return x_final # Returns (D, H, W) tensor on CPU




@torch.no_grad()
def sample_using_diffusion(
    autoencoder: nn.Module, 
    diffusion: nn.Module, 
    context: torch.Tensor,
    device: str, 
    scale_factor: int = 1,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015, 
    beta_end: float = 0.0205, 
    verbose: bool = True
) -> torch.Tensor: 
    """
    Sampling random brain MRIs that follow the covariates in `context`.

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet 
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # the subject-specific variables and the progression-related 
    # covariates are concatenated into a vector outside this function. 
    context = context.unsqueeze(0).to(device).to(device)

    # drawing a random z_T ~ N(0,I)
    z = torch.randn(const.LATENT_SHAPE_DM).unsqueeze(0).to(device)
    
    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):

                timestep = torch.tensor([t]).to(device)
                
                # predict the noise
                noise_pred = diffusion(
                    x=z.float(), 
                    timesteps=timestep, 
                    context=context.float(), 
                )

                # the scheduler applies the formula to get the 
                # denoised step z_{t-1} from z_t and the predicted noise
                z, _ = scheduler.step(noise_pred, t, z)
    
    # decode the latent
    z = z / scale_factor
    z = utils.to_vae_latent_trick( z.squeeze(0).cpu() )
    x = autoencoder.decode_stage_2_outputs( z.unsqueeze(0).to(device) )
    x = utils.to_mni_space_1p5mm_trick( x.squeeze(0).cpu() ).squeeze(0)
    return x


@torch.no_grad()
def sample_using_diffusion_las(
    autoencoder: nn.Module, 
    diffusion: nn.Module, 
    context: torch.Tensor,
    device: str, 
    scale_factor: int = 1,
    average_over_n: int = 1,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015, 
    beta_end: float = 0.0205, 
    verbose: bool = True
) -> torch.Tensor: 
    """
    Sampling random brain MRIs that follow the covariates in `context`,
    with Latent Average Stabilization (LAS).
    """
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # Add batch and sequence dimensions to the context vector.
    context = context.unsqueeze(0).unsqueeze(0).to(device)

    # if performing LAS, replicate context for parallel processing
    if average_over_n > 1:
        context = context.repeat(average_over_n, 1, 1)

    if verbose:
        print(f"Context shape: {context.shape}")

    # drawing random z_T ~ N(0,I) for each sample
    z = torch.randn(average_over_n, *const.LATENT_SHAPE_DM).to(device)
    
    if verbose:
        print(f"Initial noise z shape: {z.shape}")
    
    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):
                # repeat timestep for batch processing
                timestep = torch.tensor([t]).repeat(average_over_n).to(device)
                
                # predict the noise
                noise_pred = diffusion(
                    x=z.float(), 
                    timesteps=timestep, 
                    context=context.float()
                )

                if t == scheduler.timesteps[0] and verbose:
                    print(f"First noise_pred shape: {noise_pred.shape}")

                # the scheduler applies the formula to get the 
                # denoised step z_{t-1} from z_t and the predicted noise
                z, _ = scheduler.step(noise_pred, t, z)
    
    # LAS average
    if verbose:
        print(f"Final z shape before averaging: {z.shape}")
    
    z = (z / scale_factor).sum(axis=0) / average_over_n
    
    if verbose:
        print(f"Final z shape after averaging: {z.shape}")
    
    # decode the latent
    z = utils.to_vae_latent_trick(z.cpu())
    x = autoencoder.decode_stage_2_outputs(z.unsqueeze(0).to(device))
    x = utils.to_mni_space_1p5mm_trick(x.squeeze(0).cpu()).squeeze(0)
    return x


@torch.no_grad()
def sample_using_controlnet_and_z(
    autoencoder: nn.Module, 
    diffusion: nn.Module,
    controlnet: nn.Module,
    starting_z: torch.Tensor,
    starting_a: int, 
    context: torch.Tensor, 
    device: str,
    scale_factor: int = 1,
    average_over_n: int = 1,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015, 
    beta_end: float = 0.0205, 
    verbose: bool = True
) -> torch.Tensor:
    """
    The inference process described in the paper.

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet 
        controlnet (nn.Module): the ControlNet
        starting_z (torch.Tensor): the latent from the MRI of the starting visit 
        starting_a (int): the starting age
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        average_over_n (int, optional): LAS parameter m. Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.

    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    
    # Log shape of initial latent if verbose
    if verbose:
        utils.log_tensor_shape(starting_z, "Initial latent (starting_z)")
    
    # preparing controlnet spatial condition.
    # Check if starting_z already has batch dimension
    if starting_z.dim() == 4:  # (C, D, H, W) - add batch dimension
        starting_z = starting_z.unsqueeze(0).to(device)
    else:  # Already has batch dimension - just move to device
        starting_z = starting_z.to(device)
    
    concatenating_age      = torch.tensor([ starting_a ]).view(1, 1, 1, 1, 1).expand(1, 1, *starting_z.shape[-3:]).to(device)
    controlnet_condition   = torch.cat([ starting_z, concatenating_age ], dim=1).to(device)
    
    # Log shape of ControlNet condition if verbose
    if verbose:
        utils.log_tensor_shape(controlnet_condition, "ControlNet input condition")

    # the subject-specific variables and the progression-related 
    # covariates are concatenated into a vector outside this function. 
    # Ensure context has the right shape for cross-attention: (batch_size, seq_len, features)
    if context.dim() == 1:  # (8,) -> (1, 1, 8)
        context = context.unsqueeze(0).unsqueeze(0).to(device)
    elif context.dim() == 2:  # (1, 8) -> (1, 1, 8) or (batch_size, 8) -> (batch_size, 1, 8)
        context = context.unsqueeze(1).to(device)
    elif context.dim() == 3:  # Already correct format
        context = context.to(device)
    else:  # 4D or higher, might be wrong format
        print(f"[WARNING] Unexpected context shape: {context.shape}")
        context = context.to(device)

    # if performing LAS, we repeat the inputs for the diffusion process
    # m times (as specified in the paper) and perform the reverse diffusion
    # process in parallel to avoid overheads.
    if average_over_n > 1:
        context               = context.repeat(average_over_n, 1, 1)  # (average_over_n, 1, 8)
        controlnet_condition  = controlnet_condition.repeat(average_over_n, 1, 1, 1, 1) 
    
    # this is z_T - the starting noise.
    z = torch.randn(average_over_n, *starting_z.shape[1:]).to(device)

    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps

    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):

                # convert the timestep to a tensor.
                timestep = torch.tensor([t]).repeat(average_over_n).to(device)

                # get the intermediate features from the ControlNet
                # by feeding the starting latent, the covariates and the timestep
                down_h, mid_h = controlnet(
                    x=z.float(), 
                    timesteps=timestep, 
                    context=context,
                    controlnet_cond=controlnet_condition.float()
                )
                
                # the diffusion takes the intermediate features and predicts
                # the noise. This is why we conceptualize the two networks as
                # as a unified network.
                noise_pred = diffusion(
                    x=z.float(), 
                    timesteps=timestep, 
                    context=context.float(), 
                    down_block_additional_residuals=down_h,
                    mid_block_additional_residual=mid_h
                )

                # the scheduler applies the formula to get the 
                # denoised step z_{t-1} from z_t and the predicted noise
                z, _ = scheduler.step(noise_pred, t, z)

    # Log shape of final latent before decoding if verbose
    if verbose:
        utils.log_tensor_shape(z, "Final denoised latent (before averaging)")
    z = (z / scale_factor).sum(axis=0) / average_over_n
    if verbose:
        utils.log_tensor_shape(z, "Final denoised latent (after averaging)")
    
    # Here we conclude Latent Average Stabilization by averaging 
    # m different latents from m different samplings.
    z = utils.to_vae_latent_trick(z.squeeze(0).cpu())

    # decode the latent using the Decoder block from the KL autoencoder.
    x = autoencoder.decode_stage_2_outputs( z.unsqueeze(0).to(device) )
    x = utils.to_mni_space_1p5mm_trick( x.squeeze(0).cpu() ).squeeze(0)
    return x