import ast
import torch

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader
from data.ptbxl_dataset import PTBXLDataset
from utils.inference import load_diffusion_model
from utils.scalogram_utils import model_output_to_image
from evaluation.eval_metrics import NoiseQuantificationMetrics, compute_noise_metrics


def noise_quantification(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        diffusion_timestep: Union[int, torch.Tensor],
        step_interval: int,
        discretize: bool
) -> NoiseQuantificationMetrics:
    """
    Generate denoised images and compute noise metrics for a batch.
    """
    denoised = model.generate_denoised_sample(
        inputs, diffusion_timestep, step_interval
    )
    orig_imgs = model_output_to_image(inputs.squeeze(1).cpu().numpy(), discretize)
    clean_imgs = model_output_to_image(denoised.squeeze(1).cpu().numpy(), discretize)

    return compute_noise_metrics(orig_imgs, clean_imgs)


def noise_quantification_ptbxl(
        checkpoint_path: Union[str, Path],
        ref_min: float,
        ref_max: float,
        superlet_dir: Union[str, Path],
        batch_size: int,
        diffusion_timestep: Union[int, torch.Tensor],
        noise_scheduler_type: str,
        use_ldm: bool,
        step_interval: int,
        include_all_folds: bool,
        discretize: bool = True,
        seed: int = 123,
):
    """
    Run noise quantification over the PTB-XL dataset and save results.
    """
    # Setup
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = PTBXLDataset(
        ref_min=ref_min,
        ref_max=ref_max,
        superlet_dir=superlet_dir,
        discretize=discretize,
        train=False,
        include_all_folds=include_all_folds,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model once
    model = load_diffusion_model(checkpoint_path, noise_scheduler_type, use_ldm, device)
    if isinstance(diffusion_timestep, int):
        diffusion_timestep = torch.tensor([diffusion_timestep], dtype=torch.int32, device=device)

    # Prepare accumulator
    metrics_accum = {
        'psnr': [], 'ssim': [], 'mse': [], 'mae': []
    }

    # Inference loop
    for batch in tqdm(loader, desc='Quantifying Noise'):
        batch = batch.to(device)
        metrics = noise_quantification(
            model, batch, diffusion_timestep, step_interval, discretize
        )
        metrics_accum['psnr'].extend(metrics.psnr.tolist())
        metrics_accum['ssim'].extend(metrics.ssim.tolist())
        metrics_accum['mse'].extend(metrics.mse.tolist())
        metrics_accum['mae'].extend(metrics.mae.tolist())

    # Load reference labels
    label_path = Path(__file__).resolve().parents[1] / 'data' / 'database' / 'ptbxl_label_dropna.csv'
    labels = pd.read_csv(label_path)
    base_cols = ['filename_hr', 'scp_codes', 'baseline_drift', 'static_noise',
                 'burst_noise', 'electrodes_problems', 'clean', 'strat_fold']

    # Define output directory and file naming
    out_dir = Path('results')
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = 'ldm' if use_ldm else 'dm'
    scheduler = f"{noise_scheduler_type}-{step_interval}" if noise_scheduler_type != 'ddpm' else noise_scheduler_type
    timestep = int(diffusion_timestep.item())
    prefix = 'all_' if include_all_folds else ''

    # Metrics to export
    metrics_list = ['psnr', 'ssim', 'mse', 'mae']

    for metric in metrics_list:
        rows = []
        idx_ptr = 0

        for idx, row in labels.iterrows():
            fold = int(row['strat_fold'])
            clean_leads = ast.literal_eval(row['clean'])
            noise_leads = [i for i in range(12) if i not in clean_leads]
            slice_len = 12 if (include_all_folds or fold == 10) else len(noise_leads)

            metric_vals = metrics_accum[metric][idx_ptr: idx_ptr + slice_len]
            idx_ptr += slice_len

            lead_vals = [np.nan] * 12
            if include_all_folds or fold == 10:
                lead_vals = metric_vals[:12]
            else:
                for pos, lead in enumerate(noise_leads):
                    lead_vals[lead] = metric_vals[pos]

            row_dict = row[base_cols].to_dict()
            row_dict.update({f'lead_{i}': lead_vals[i] for i in range(12)})
            rows.append(row_dict)

        df = pd.DataFrame(rows, columns=base_cols + [f'lead_{i}' for i in range(12)])
        filename = f"{prefix}{tag}_{scheduler}_t{timestep}_{metric}.csv"
        df.to_csv(out_dir / filename, index=False)
