import ast
import torch
import numpy as np
import pandas as pd
from typing import Union, List
from pathlib import Path

from tqdm import tqdm
from torch.utils.data import DataLoader
from models.vanilla_diffusion import DiffusionModel
from models.latent_diffusion import LatentDiffusionModel
from data.ptbxl_dataset import PTBXLDataset
from model.model_utils.forward_reverse import forward_reverse_process
from utils.inference import load_diffusion_model


def measure_noise_ptb(
        checkpoint_path: Union[str, Path],
        ref_min: float,
        ref_max: float,
        superlet_dir: Union[str, Path],
        batch_size: int,
        diffusion_timestep: Union[int, torch.Tensor],
        noise_scheduler_type: str,
        use_ldm: bool,
        step_interval: int,
        include_all_folds: bool ,
        discretize: bool = True,
        seed: int = 123,
):
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
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = load_diffusion_model(checkpoint_path, noise_scheduler_type, use_ldm, device)
    if isinstance(diffusion_timestep, int):
        diffusion_timestep = torch.IntTensor([diffusion_timestep]).to(device)

    psnr, ssim, mse, mae = [], [], [], []
    for image in tqdm(dataloader):
        output = forward_reverse_process(
            model,
            image,
            device,
            timestep,
            quantization,
            step=step,
        )
        psnr.extend(output.psnr)
        ssim.extend(output.ssim)
        mse.extend(output.mse)
        mae.extend(output.mae)

    columns = ['filename_hr', 'scp_codes', 'baseline_drift', 'static_noise',
               'burst_noise', 'electrodes_problems', 'clean', 'strat_fold']
    columns.extend([i for i in range(12)])

    psnr_df = pd.DataFrame(columns=columns)
    ssim_df = pd.DataFrame(columns=columns)
    mse_df = pd.DataFrame(columns=columns)
    mae_df = pd.DataFrame(columns=columns)

    Y = pd.read_csv(PTBXL_DIR / 'ptb_label_dropna.csv')
    start_idx, end_idx = 0, 0
    for idx, file in tqdm(enumerate(Y.filename_hr)):
        clean_idx = ast.literal_eval(Y.clean[idx])
        noise_idx = sorted(set(range(12)) - set(clean_idx))

        if int(Y.strat_fold[idx]) == 10 or (all_data is True):
            end_idx = start_idx + 12
            psnr_ = psnr[start_idx:end_idx]
            ssim_ = ssim[start_idx:end_idx]
            mse_ = mse[start_idx:end_idx]
            mae_ = mae[start_idx:end_idx]
            start_idx = end_idx

        else:
            psnr_, ssim_, mse_, mae_ = [np.nan] * 12, [np.nan] * 12, [np.nan] * 12, [np.nan] * 12
            if len(noise_idx) > 0:
                for i in noise_idx:
                    psnr_[i] = psnr[start_idx]
                    ssim_[i] = ssim[start_idx]
                    mse_[i] = mse[start_idx]
                    mae_[i] = mae[start_idx]
                    start_idx += 1
                    end_idx += 1

        psnr_df.loc[idx] = Y.loc[idx].tolist() + psnr_
        ssim_df.loc[idx] = Y.loc[idx].tolist() + ssim_
        mse_df.loc[idx] = Y.loc[idx].tolist() + mse_
        mae_df.loc[idx] = Y.loc[idx].tolist() + mae_

    if noise_scheduler_type == 'ddpm':
        save_name = f'{model_directory}_{noise_scheduler_type}_ema{model_id}_ts{int(timestep)}'
    else:
        save_name = f'{model_directory}_{noise_scheduler_type}{step}_ema{model_id}_ts{int(timestep)}'

    if all_data is True:
        save_name = f'all_{save_name}'

    psnr_df.to_csv(PTBXL_METRIC_DIR / f"{save_name}_psnr.csv", index=False)
    ssim_df.to_csv(PTBXL_METRIC_DIR / f"{save_name}_ssim.csv", index=False)
    mse_df.to_csv(PTBXL_METRIC_DIR / f"{save_name}_mse.csv", index=False)
    mae_df.to_csv(PTBXL_METRIC_DIR / f"{save_name}_mae.csv", index=False)
