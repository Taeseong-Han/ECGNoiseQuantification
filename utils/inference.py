import torch

import numpy as np

from pathlib import Path
from typing import Union, List
from dataclasses import dataclass
from models.vanilla_diffusion import DiffusionModel
from models.latent_diffusion import LatentDiffusionModel
from preprocessing.superlet_transform import compute_superlet_scalogram
from utils.scalogram_utils import scalogram_to_model_input, model_output_to_image


@dataclass
class Output:
    psnr: np.ndarray  # shape: (leads, segments)
    original_image: Union[np.ndarray, None]  # shape: (leads, segments, H, W)
    cleaned_image: Union[np.ndarray, None]  # shape: (leads, segments, H, W)


def compute_psnr_matrix(
        orig: np.ndarray,
        clean: np.ndarray,
        rem: int,
        sampling_freq: int,
        n_partitions: int,
        segment_seconds: int = 10,

) -> np.ndarray:
    """
    Compute PSNR matrix for all leads and segments, dividing each segment into
    n_partitions along the time dimension. For the last segment, only partitions
    containing valid data (based on remainder) are included.

    Args:
        orig, clean: Arrays of shape (leads, segments, H, W).
        rem: Remainder timepoints of the last segment before padding.
        sampling_freq: Sampling frequency in Hz.
        segment_seconds: Segment duration in seconds (default: 10).
        n_partitions: Number of equal-width partitions per segment (default: 1).

    Returns:
        psnr: Array of shape (leads, total_valid_partitions), where total_valid_partitions
              is the number of partitions containing valid data across all segments.
    """
    leads, segments, H, W = orig.shape
    seg_len = sampling_freq * segment_seconds
    part_width = W // n_partitions  # Width of each partition

    psnr_list = []  # Collect PSNR values for valid partitions

    for seg in range(segments):
        # Determine valid data range for the segment
        if seg == segments - 1 and rem:
            valid_ratio = rem / seg_len
            width_cut = int(W * valid_ratio)
            valid_start = W - width_cut
        else:
            valid_start = 0

        # Process each partition
        for p in range(n_partitions):
            start = valid_start + p * part_width
            end = start + part_width if p < n_partitions - 1 else W
            end = min(end, W)  # Ensure end does not exceed W

            # Skip partitions with no valid data
            if start >= W or end <= valid_start:
                continue

            # Compute PSNR for the partition
            section_orig = orig[:, seg, :, start:end]
            section_clean = clean[:, seg, :, start:end]
            diff = section_orig - section_clean
            mse = np.mean(diff ** 2, axis=(1, 2))  # Shape: (leads,)
            psnr = 10 * np.log10((255 ** 2) / mse)
            psnr[mse == 0] = np.inf
            psnr_list.append(psnr)

    # Convert to array with shape (leads, total_valid_partitions)
    return np.stack(psnr_list, axis=1)


def segment_ecg(ecg: np.ndarray, sampling_freq: int, segment_seconds: int = 10) -> (np.ndarray, int):
    if ecg.ndim == 1:
        ecg = ecg.reshape(1, -1)
    elif ecg.ndim == 2 and ecg.shape[0] > ecg.shape[1]:
        ecg = ecg.T
    elif ecg.ndim == 3:
        raise ValueError("ECG input must be a 1D or 2D array. 3D input not supported.")

    seg_len = sampling_freq * segment_seconds
    total_len = ecg.shape[-1]
    if total_len < seg_len:
        raise ValueError(f"ECG must be at least {segment_seconds}s long.")

    full = total_len // seg_len
    rem = total_len % seg_len
    segs = [ecg[:, i * seg_len:(i + 1) * seg_len] for i in range(full)]
    if rem:
        tail = ecg[:, -seg_len:]
        segs.append(tail)

    return np.stack(segs, axis=0), rem


def adjust_batch_size(batch_size, leads):
    return round(batch_size / leads) * leads


def load_diffusion_model(
        checkpoint_path: Union[str, Path],
        noise_scheduler_type: str,
        use_ldm: bool,
        device: torch.device
) -> torch.nn.Module:
    weights = torch.load(checkpoint_path, map_location=device)
    model = LatentDiffusionModel(noise_scheduler_type=noise_scheduler_type) if use_ldm \
        else DiffusionModel(noise_scheduler_type=noise_scheduler_type)
    model.load_state_dict(weights)
    return model.to(device)


def ecg_noise_quantification(
    ecg: np.ndarray,
    sampling_freq: int,
    checkpoint_path: Union[str, Path],
    n_partitions: int = 1,
    batch_size: int = 64,
    diffusion_timestep: Union[int, torch.Tensor] = 30,
    noise_scheduler_type: str = 'ddim',
    discretize: bool = True,
    use_ldm: bool = True,
    step_interval: int = 10,
    seed: int = 123,
    return_images: bool = False,
) -> Output:
    """
    Quantify noise in an ECG signal using a diffusion-based anomaly detection framework.

    Args:
        ecg: Raw ECG signal array, shape (channels, samples) or (samples,).
        sampling_freq: Sampling frequency in Hz.
        checkpoint_path: Path to model weights.
        n_partitions: Number of partitions per segment for PSNR calculation.
        batch_size: Maximum number of scalograms per batch.
        diffusion_timestep: Number of reverse diffusion steps or tensor of timesteps.
        noise_scheduler_type: Scheduler name for diffusion ('ddpm' or 'ddim').
        discretize: Whether to discretize model output into image form.
        use_ldm: If True, use LatentDiffusionModel; otherwise use vanilla DiffusionModel.
        step_interval: Step interval for generating denoised samples.
        seed: Random seed for reproducibility.
        return_images: If True, returns original and cleaned images.

    Returns:
        Output dataclass with fields:
            - psnr: PSNR matrix, shape (leads, total_valid_partitions)
            - original_image: Optional array of original scalogram images
            - cleaned_image: Optional array of cleaned scalogram images
    """
    # Reproducibility and device setup
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_diffusion_model(checkpoint_path, noise_scheduler_type, use_ldm, device)
    if isinstance(diffusion_timestep, int):
        diffusion_timestep = torch.tensor([diffusion_timestep], dtype=torch.int32, device=device)

    # Segment ECG into fixed-length windows
    ecg_segs, rem = segment_ecg(ecg, sampling_freq)
    segments, leads, _ = ecg_segs.shape
    flattened_ecg = ecg_segs.reshape(-1, ecg_segs.shape[-1])

    # Adjust batch size to be multiple of leads
    total_samples = flattened_ecg.shape[0]
    batch_size = min(((batch_size + leads - 1) // leads) * leads, total_samples)

    psnr_batches = []
    orig_images, clean_images = [], []

    # Process in batches
    for idx in range(0, total_samples, batch_size):
        batch = flattened_ecg[idx: idx + batch_size]

        # Compute input scalograms and model tensor
        scalograms = compute_superlet_scalogram(batch, sampling_freq)
        model_input = torch.from_numpy(
            scalogram_to_model_input(scalograms)
        ).unsqueeze(1).float().to(device)

        # Generate denoised output
        denoised = model.generate_denoised_sample(
            model_input, diffusion_timestep, step_interval
        )

        # Convert tensors back to numpy images
        orig_img = model_output_to_image(model_input.squeeze(1).cpu().numpy(), discretize)
        clean_img = model_output_to_image(denoised.squeeze(1).cpu().numpy(), discretize)

        # Reshape to (leads, segments_in_batch, H, W)
        batch_segments = orig_img.shape[0] // leads
        H, W = orig_img.shape[1], orig_img.shape[2]
        orig = orig_img.reshape(batch_segments, leads, H, W).transpose(1, 0, 2, 3)
        clean = clean_img.reshape(batch_segments, leads, H, W).transpose(1, 0, 2, 3)

        # Compute PSNR for this batch
        is_last_batch = (idx + batch_size) >= total_samples
        rem_for_last = rem if is_last_batch else 0
        psnr_matrix = compute_psnr_matrix(orig, clean, rem_for_last, sampling_freq, n_partitions)
        psnr_batches.append(psnr_matrix)

        # Optionally collect images
        if return_images:
            orig_images.append(orig)
            clean_images.append(clean)

    # Concatenate results
    psnr = np.concatenate(psnr_batches, axis=1)
    orig_out = np.concatenate(orig_images, axis=1) if return_images else None
    clean_out = np.concatenate(clean_images, axis=1) if return_images else None

    return Output(psnr=psnr, original_image=orig_out, cleaned_image=clean_out)

