from typing import Union, Tuple

import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to
from monai import transforms
from monai.data.meta_tensor import MetaTensor
import wandb
from . import const

class AverageLoss:
    """
    Utility class to track losses
    and metrics during training.
    """

    def __init__(self):
        self.losses_accumulator = {}
    
    def put(self, loss_key:str, loss_value:Union[int,float]) -> None:
        """
        Store value

        Args:
            loss_key (str): Metric name
            loss_value (int | float): Metric value to store
        """
        if loss_key not in self.losses_accumulator:
            self.losses_accumulator[loss_key] = []
        self.losses_accumulator[loss_key].append(loss_value)
    
    def pop_avg(self, loss_key:str) -> float:
        """
        Average the stored values of a given metric

        Args:
            loss_key (str): Metric name

        Returns:
            float: average of the stored values
        """
        if loss_key not in self.losses_accumulator:
            return None
        losses = self.losses_accumulator[loss_key]
        self.losses_accumulator[loss_key] = []
        return sum(losses) / len(losses)
    
    def to_wandb(self, step: int):
        """
        Logs the average value of all the metrics stored 
        into Weights & Biases.

        Args:
            step (int): Global step for wandb logging
        """
        metrics = {}
        for metric_key in self.losses_accumulator.keys():
            value = self.pop_avg(metric_key)
            # Ensure value is JSON serializable (convert tensors to Python scalars)
            if torch.is_tensor(value):
                value = value.item() if value.numel() == 1 else value.detach().cpu().numpy().tolist()
            metrics[metric_key] = value
        wandb.log(metrics, step=step)

def log_tensor_shape(tensor: Union[torch.Tensor, Tuple[torch.Tensor, ...], list], name: str = ""):
    """Log the shape of a tensor, tuple of tensors, or list of tensors."""
    if isinstance(tensor, (tuple, list)):
        print(f"{name} ({type(tensor).__name__}) shapes:")
        for i, t in enumerate(tensor):
            print(f"  [{i}]: {t.shape}")
    else:
        print(f"{name} shape: {tensor.shape}")

            
def to_vae_latent_trick(z: torch.Tensor, unpadded_z_shape: tuple = (3, 15, 18, 15)) -> torch.Tensor:
    """
    The latent for the VAE is not divisible by 4 (required to
    go through the UNet), therefore we apply padding before using 
    it with the UNet. This function removes the padding.

    Args:
        z (torch.Tensor): Padded latent
        unpadded_z_shape (tuple, optional): unpadded latent dimensions. Defaults to (3, 15, 18, 15).

    Returns:
        torch.Tensor: Latent without padding
    """
    padder = transforms.DivisiblePad(k=4)
    z = padder(MetaTensor(torch.zeros(unpadded_z_shape))) + z
    z = padder.inverse(z)
    return z


def to_mni_space_1p5mm_trick(x: torch.Tensor, mni1p5_dim: tuple = (122, 146, 122)) -> torch.Tensor:
    """
    The volume is resized to be divisible by 8 (required by 
    the autoencoder). This function restores the initial dimensions
    (i.e., the MNI152 space dimensions at 1.5 mm^3). 

    Args:
        x (torch.Tensor): Resized volume
        mni1p5_dim (tuple, optional): MNI152 space dims at 1.5 mm^3. Defaults to (122, 146, 122).

    Returns:
        torch.Tensor: input resized to original shape
    """
    resizer = transforms.ResizeWithPadOrCrop(spatial_size=mni1p5_dim, mode='minimum')
    return resizer(x)


def wandb_display_reconstruction(step, image, recon):
    """
    Display reconstruction in W&B during AE training.
    """
    # Convert tensors to numpy arrays for matplotlib
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    if torch.is_tensor(recon):
        recon = recon.detach().cpu().numpy()
    
    # Handle MetaTensor type
    if hasattr(image, 'array'):
        image = image.array
    if hasattr(recon, 'array'):
        recon = recon.array
        
    plt.style.use('dark_background')
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(7, 5))
    for _ax in ax.flatten(): _ax.set_axis_off()

    if len(image.shape) == 4: image = image.squeeze(0) 
    if len(recon.shape) == 4: recon = recon.squeeze(0)

    ax[0, 0].set_title('original image', color='cyan')
    ax[0, 0].imshow(image[image.shape[0] // 2, :, :], cmap='gray')
    ax[0, 1].imshow(image[:, image.shape[1] // 2, :], cmap='gray')
    ax[0, 2].imshow(image[:, :, image.shape[2] // 2], cmap='gray')

    ax[1, 0].set_title('reconstructed image', color='magenta')
    ax[1, 0].imshow(recon[recon.shape[0] // 2, :, :], cmap='gray')
    ax[1, 1].imshow(recon[:, recon.shape[1] // 2, :], cmap='gray')
    ax[1, 2].imshow(recon[:, :, recon.shape[2] // 2], cmap='gray')

    plt.tight_layout()
    
    # Log to wandb
    wandb.log({"Reconstruction": wandb.Image(fig)}, step=step)
    plt.close(fig)

    
def wandb_display_generation(step, tag, image):
    """
    Display generation result in W&B during Diffusion Model training.
    """
    # Convert tensor to numpy array for matplotlib
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    
    # Handle MetaTensor type
    if hasattr(image, 'array'):
        image = image.array
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(ncols=3, figsize=(7, 3))
    for _ax in ax.flatten(): _ax.set_axis_off()

    ax[0].imshow(image[image.shape[0] // 2, :, :], cmap='gray')
    ax[1].imshow(image[:, image.shape[1] // 2, :], cmap='gray')
    ax[2].imshow(image[:, :, image.shape[2] // 2], cmap='gray')

    plt.tight_layout()
    
    # Log to wandb
    wandb.log({tag: wandb.Image(fig)}, step=step)
    plt.close(fig)

def wandb_display_cond_generation(
    step, 
    tag, 
    starting_image, 
    followup_image, 
    predicted_image,
    start_diag=None,
    follow_diag=None,
    sex_value=None, 
    start_age=None,
    follow_age=None
):
    """
    Display conditional generation result in W&B during training.
    Diagnosis, age, and sex are embedded in the subplot titles using
    the custom label mappings defined below.
    """

    # === 1. Map start_diag, follow_diag, and sex_value to display labels ===
    # Custom mappings are defined below.
    def diag_to_str(x):
        # x is expected to be 0, 0.5, or 1.
        if x == 0:
            return "NC"
        elif x == 0.5:
            return "MCI"
        elif x == 1:
            return "AD"
        else:
            return f"diag?({x})"  # unknown value

    def sex_to_str(s):
        # s is expected to be 0 or 1.
        if s == 0:
            return "male"
        elif s == 1:
            return "female"
        else:
            return f"sex?({s})"  # unknown value
    
    start_diag_str = diag_to_str(start_diag) if start_diag is not None else "N/A"
    follow_diag_str = diag_to_str(follow_diag) if follow_diag is not None else "N/A"
    sex_str = sex_to_str(sex_value) if sex_value is not None else "N/A"
    
    # === 2. Prepare images for plotting ===
    # Convert tensors to numpy arrays for matplotlib
    if torch.is_tensor(starting_image):
        starting_image = starting_image.detach().cpu().numpy()
    if torch.is_tensor(followup_image):
        followup_image = followup_image.detach().cpu().numpy()
    if torch.is_tensor(predicted_image):
        predicted_image = predicted_image.detach().cpu().numpy()
    
    # Handle MetaTensor
    if hasattr(starting_image, 'array'):
        starting_image = starting_image.array
    if hasattr(followup_image, 'array'):
        followup_image = followup_image.array
    if hasattr(predicted_image, 'array'):
        predicted_image = predicted_image.array
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(7, 7))
    for _ax in ax.flatten():
        _ax.set_axis_off()

    # === 3. Build title strings and plot ===
    start_title_str = f"starting image\n diag={start_diag_str}, age={start_age}, sex={sex_str}"
    follow_title_str = f"follow-up image\n diag={follow_diag_str}, age={follow_age}, sex={sex_str}"
    
    ax[0, 0].set_title(start_title_str, color='cyan', loc='left')
    ax[0, 0].imshow(starting_image[starting_image.shape[0] // 2, :, :], cmap='gray')
    ax[0, 1].imshow(starting_image[:, starting_image.shape[1] // 2, :], cmap='gray')
    ax[0, 2].imshow(starting_image[:, :, starting_image.shape[2] // 2], cmap='gray')

    ax[1, 0].set_title(follow_title_str, color='magenta', loc='left')
    ax[1, 0].imshow(followup_image[followup_image.shape[0] // 2, :, :], cmap='gray')
    ax[1, 1].imshow(followup_image[:, followup_image.shape[1] // 2, :], cmap='gray')
    ax[1, 2].imshow(followup_image[:, :, followup_image.shape[2] // 2], cmap='gray')

    ax[2, 0].set_title('predicted follow-up', color='yellow', loc='left')
    ax[2, 0].imshow(predicted_image[predicted_image.shape[0] // 2, :, :], cmap='gray')
    ax[2, 1].imshow(predicted_image[:, predicted_image.shape[1] // 2, :], cmap='gray')
    ax[2, 2].imshow(predicted_image[:, :, predicted_image.shape[2] // 2], cmap='gray')
    
    plt.tight_layout()
    
    # === 4. Log to wandb ===
    wandb.log({tag: wandb.Image(fig)}, step=step)
    plt.close(fig)


# Compatibility wrappers for the original TensorBoard-based API.
def tb_display_reconstruction(writer, step, image, recon):
    """
    Display reconstruction in TensorBoard during AE training.
    Compatibility wrapper for wandb version.
    """
    wandb_display_reconstruction(step, image, recon)

    
def tb_display_generation(writer, step, tag, image):
    """
    Display generation result in TensorBoard during Diffusion Model training.
    Compatibility wrapper for wandb version.
    """
    wandb_display_generation(step, tag, image)


def tb_display_cond_generation(writer, step, tag, starting_image, followup_image, predicted_image):
    """
    Display conditional generation result in TensorBoard during ControlNet training.
    Compatibility wrapper for wandb version.
    """
    wandb_display_cond_generation(step, tag, starting_image, followup_image, predicted_image)


def percnorm_nifti(mri, lperc=1, uperc=99):
    '''
    Apply percnorm to NiFTI1Image class
    '''
    norm_arr = percnorm(mri.get_fdata(), lperc, uperc)
    return nib.Nifti1Image(norm_arr, mri.affine, mri.header)


def percnorm(arr, lperc=1, uperc=99):
    '''
    Remove outlier intensities from a brain component,
    similar to Tukey's fences method.
    '''
    upperbound = np.percentile(arr, uperc)
    lowerbound = np.percentile(arr, lperc)
    arr[arr > upperbound] = upperbound
    arr[arr < lowerbound] = lowerbound
    return arr


def apply_mask(mri, segm):
    """
    Performs brain extraction.
    """
    segm = resample_from_to(segm, mri, order=0)
    mask = segm.get_fdata() > 0
    mri_arr = mri.get_fdata()
    mri_arr[ mask == 0 ] = 0
    return nib.Nifti1Image(mri_arr, mri.affine, mri.header)
