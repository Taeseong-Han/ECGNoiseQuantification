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
    original_image: np.ndarray  # shape: (leads, segments, H, W)
    cleaned_image: np.ndarray  # shape: (leads, segments, H, W)
    psnr: np.ndarray  # shape: (leads, segments)


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
) -> Output:
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Segment and prepare input
    ecg_segs, rem = segment_ecg(ecg, sampling_freq)
    segments, leads, _ = ecg_segs.shape
    batch_ecg = ecg_segs.reshape(-1, ecg_segs.shape[-1])

    scalograms = compute_superlet_scalogram(batch_ecg, sampling_freq)
    model_in = scalogram_to_model_input(scalograms)
    tensor_in = torch.FloatTensor(model_in).unsqueeze(1).to(device)

    model = load_diffusion_model(checkpoint_path, noise_scheduler_type, use_ldm, device)
    if isinstance(diffusion_timestep, int):
        diffusion_timestep = torch.IntTensor([diffusion_timestep]).to(device)

    # Batched denoising to avoid OOM
    cleaned_batches: List[torch.Tensor] = []
    for start in range(0, tensor_in.size(0), batch_size):
        end = start + batch_size
        batch = tensor_in[start:end]
        cleaned_batch = model.generate_denoised_sample(batch, diffusion_timestep, step_interval)
        cleaned_batches.append(cleaned_batch.cpu())
    cleaned = torch.cat(cleaned_batches, dim=0)

    orig_img = model_output_to_image(tensor_in.squeeze(1).cpu().numpy(), discretize)
    clean_img = model_output_to_image(cleaned.squeeze(1).cpu().numpy(), discretize)

    # reshape to (leads, segments, H, W)
    H, W = orig_img.shape[1], orig_img.shape[2]
    orig = orig_img.reshape(segments, leads, H, W).transpose(1, 0, 2, 3)
    clean = clean_img.reshape(segments, leads, H, W).transpose(1, 0, 2, 3)

    # Compute PSNR matrix
    psnr_matrix = compute_psnr_matrix(orig, clean, rem, sampling_freq, n_partitions)

    return Output(original_image=orig, cleaned_image=clean, psnr=psnr_matrix)
