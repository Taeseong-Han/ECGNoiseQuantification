import torch

import torch.nn as nn
import torch.nn.functional as F

from models.unet import build_dm_unet
from diffusers import DDPMScheduler, DDIMScheduler


class DiffusionModel(nn.Module):
    def __init__(
            self,
            noise_scheduler_type='ddpm',  # ddim
            max_timestep: int = 1000,
    ):
        super().__init__()
        self.max_timestep = max_timestep

        # denoising unet
        self.unet = build_dm_unet()

        # noise scheduler
        self.noise_scheduler_type = noise_scheduler_type
        if noise_scheduler_type == 'ddpm':
            self.noise_scheduler = DDPMScheduler()
        elif noise_scheduler_type == 'ddim':
            self.noise_scheduler = DDIMScheduler()
        else:
            raise Exception("noise scheduler is wrong")

    def forward(self, x):
        self.train()

        # 1. add noise
        noise = torch.randn(x.shape, device=x.device)
        timesteps = torch.randint(0, self.max_timestep, (x.shape[0],), device=x.device, dtype=torch.int64)
        noisy_images = self.noise_scheduler.add_noise(x, noise, timesteps)

        # 2. pred_noise
        noise_pred = self.unet(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def generate_denoised_sample(self, x, timestep: int, step_interval: int = 10):
        self.eval()
        device = x.device

        # 1. set timestep
        if self.noise_scheduler_type == "ddim":
            self.noise_scheduler.set_timesteps(1000 // step_interval)
            timestep_idx = int(timestep // step_interval)
        else:
            timestep_idx = int(timestep)

        # 2. diffusion process
        noise = torch.randn(x.shape, device=device)
        timesteps = self.noise_scheduler.timesteps[-(timestep_idx + 1)].repeat(x.shape[0])

        input_image = self.noise_scheduler.add_noise(x, noise, timesteps).to(device)

        # 3. reverse process
        for t in self.noise_scheduler.timesteps[-(timestep_idx + 1):]:
            with torch.no_grad():
                noisy_residual = self.unet(input_image, torch.tensor([t]).to(device))
            previous_noisy_sample = self.noise_scheduler.step(noisy_residual, t, input_image).prev_sample
            input_image = previous_noisy_sample

        x = input_image

        return x
