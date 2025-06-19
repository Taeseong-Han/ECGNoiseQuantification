import numpy as np

from dataclasses import dataclass
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class NoiseQuantificationMetrics:
    psnr: np.ndarray
    ssim: np.ndarray
    mse: np.ndarray
    mae: np.ndarray


def compute_noise_metrics(
        orig_imgs: np.ndarray,
        clean_imgs: np.ndarray
) -> NoiseQuantificationMetrics:
    """
    Compute PSNR, SSIM, MSE, and MAE between corresponding original
    and cleaned images.

    Args:
        orig_imgs: Array of shape (N, H, W)
        clean_imgs: Array of shape (N, H, W)
    Returns:
        NoiseQuantificationMetrics with arrays of length N
    """
    N = orig_imgs.shape[0]

    psnr_vals = np.empty(N, dtype=float)
    ssim_vals = np.empty(N, dtype=float)
    mse_vals = np.empty(N, dtype=float)
    mae_vals = np.empty(N, dtype=float)

    for i in range(N):
        o, c = orig_imgs[i], clean_imgs[i]
        psnr_vals[i] = peak_signal_noise_ratio(o, c, data_range=255.)
        ssim_vals[i] = structural_similarity(o, c, data_range=255.)
        diff = o - c
        mse_vals[i] = np.mean(diff ** 2)
        mae_vals[i] = np.mean(np.abs(diff))

    return NoiseQuantificationMetrics(
        psnr=psnr_vals,
        ssim=ssim_vals,
        mse=mse_vals,
        mae=mae_vals,
    )


def compute_w1_distance(u_values, v_values):
    scaler = StandardScaler()
    scaler.fit(np.array(u_values + v_values).reshape(-1, 1))
    u_values = scaler.transform(np.array(u_values).reshape(-1, 1)).flatten()
    v_values = scaler.transform(np.array(v_values).reshape(-1, 1)).flatten()

    return wasserstein_distance(u_values, v_values)
