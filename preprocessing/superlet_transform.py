import torch

import numpy as np
import jax.numpy as jnp
import torch.nn.functional as F

from preprocessing.superlet_core import adaptive_superlet_transform


def compute_superlet_scalogram(
        signal: np.ndarray,
        sampling_freq: int,
        resize_shape: int = 256,
        freqs_len: int = 32,
) -> np.ndarray:
    def _compute_superlet(sig):
        freqs = jnp.linspace(0.5, 40, freqs_len)
        superlet = adaptive_superlet_transform(
            sig,
            base_cycle=1,
            freqs=freqs,
            sampling_freq=sampling_freq,
            min_order=1,
            max_order=16,
        )
        superlet = np.log10(np.abs(superlet) ** 2)
        superlet_tensor = torch.tensor(superlet, dtype=torch.float32).unsqueeze(0)  # (1, freq, time)
        resized = F.interpolate(
            superlet_tensor, size=resize_shape, mode='linear', align_corners=False
        ).squeeze(0).flip(0)
        return resized.numpy()

    if signal.ndim == 1:
        # 1-lead ECG
        return _compute_superlet(signal)

    elif signal.ndim == 2:
        # multi-lead ECG
        resized_list = [_compute_superlet(signal[i]) for i in range(signal.shape[0])]
        return np.stack(resized_list)

    else:
        raise ValueError(f"Unexpected signal shape: {signal.shape}")
