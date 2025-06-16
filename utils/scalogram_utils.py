import torch

import numpy as np


def scalogram_to_model_input(
    scalogram: np.ndarray,
    ref_min: float = -8,
    ref_max: float = 0,
    discretize: bool = True
) -> np.ndarray:
    """
    Convert a scalogram to model input format in [-1, 1].

    If `discretize` is True, the scalogram is first quantized to 8-bit (0–255)
    and then rescaled back to [-1, 1], simulating precision loss.
    If False, it's directly normalized to [-1, 1] with full precision.
    """
    scalogram = np.clip(scalogram, ref_min, ref_max)
    if discretize:
        scalogram = (np.round((scalogram - ref_min) / (ref_max - ref_min) * 255)
                     .astype(np.uint8)
                     .astype(np.float32))
        scalogram = scalogram / 127.5 - 1
    else:
        scalogram = ((scalogram - ref_min) / (ref_max - ref_min)) * 2 - 1

    return scalogram


def scalogram_to_image(
    scalogram: np.ndarray,
    ref_min: float = -8,
    ref_max: float = 0,
    discretize: bool = True
) -> np.ndarray:
    """
    Convert scalogram to image format in [0, 255] as float32.

    If `discretize` is True, it is rounded to 8-bit integers and then cast back to float32.
    Otherwise, values are mapped linearly to float values in [0, 255].
    """
    scalogram = np.clip(scalogram, ref_min, ref_max)
    if discretize:
        scalogram = (np.round((scalogram - ref_min) / (ref_max - ref_min) * 255)
                     .astype(np.uint8)
                     .astype(np.float32))
    else:
        scalogram = ((scalogram - ref_min) / (ref_max - ref_min)) * 255

    return scalogram


def model_output_to_image(
    sample: torch.Tensor | np.ndarray,
    discretize: bool = True
) -> torch.Tensor | np.ndarray:
    """
    Convert model output from range [-1, 1] to image format in [0, 255].

    If `discretize` is True, result is rounded to integers before converting to float32.
    """
    if discretize:
        if torch.is_tensor(sample):
            return ((sample + 1) * 127.5).add(0.5).clamp(0, 255).to(torch.uint8).to(torch.float32)
        else:
            return (((sample + 1) * 127.5) + 0.5).clip(0, 255).astype(np.uint8).astype(np.float32)
    else:
        if torch.is_tensor(sample):
            return ((sample + 1) * 127.5).clamp(0, 255)
        else:
            return ((sample + 1) * 127.5).clip(0, 255)
