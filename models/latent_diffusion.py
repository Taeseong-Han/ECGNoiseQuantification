import torch

import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from .unet import build_ldm_unet
from .autoencoder import build_ae
from diffusers import DDPMScheduler, DDIMScheduler


class LatentDiffusionModel(nn.Module):
    def __init__(
            self,
            ae_weight_path=None,
            noise_scheduler_type='ddpm',  # ddim
            max_timestep=1000,
    ):
        super().__init__()
        self.max_timestep = max_timestep

        # autoencoder
        self.ae = build_ae()

        # load pretrained autoencoder wights
        if ae_weight_path is not None:
            ae_weight_path = Path(ae_weight_path)
            if not ae_weight_path.exists():
                raise FileNotFoundError(f"[LDM] AE weight file not found: {ae_weight_path}")
            try:
                self.ae.load_state_dict(torch.load(ae_weight_path, map_location='cpu'))
                print(f"[LDM] Loaded AE weights from {ae_weight_path}")
            except Exception as e:
                raise RuntimeError(f"[LDM] Failed to load AE weights: {e}")

        # freeze autoencoder
        for param in self.ae.parameters():
            param.requires_grad = False

        # denoising unet
        self.unet = build_ldm_unet()

        # noise scheduler
        self.noise_scheduler_type = noise_scheduler_type
        if noise_scheduler_type == 'ddpm':
            self.noise_scheduler = DDPMScheduler()
        elif noise_scheduler_type == 'ddim':
            self.noise_scheduler = DDIMScheduler()
        else:
            raise Exception("noise scheduler is wrong")

    def encode(self, x):
        h = self.ae.encode(x)

        return h * self.ae.scaling_factor

    def decode(self, z):
        decoded = self.ae.decode(z / self.ae.scaling_factor)

        return decoded

    def forward(self, x):
        self.train()

        # 1. ae encode
        x = self.encode(x)

        # 2. add noise
        noise = torch.randn(x.shape, device=x.device)
        timesteps = torch.randint(0, self.max_timestep, (x.shape[0],), device=x.device, dtype=torch.int64)
        noisy_images = self.noise_scheduler.add_noise(x, noise, timesteps)

        # 3. pred_noise
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

        # 2. ae encode
        x = self.encode(x.to(device))

        # 3.diffusion process
        noise = torch.randn(x.shape, device=device)
        timesteps = self.noise_scheduler.timesteps[-(timestep_idx + 1)].repeat(x.shape[0])
        input_image = self.noise_scheduler.add_noise(x.to(device), noise, timesteps).to(device)

        # 4. reverse process
        for t in self.noise_scheduler.timesteps[-(timestep_idx + 1):]:
            with torch.no_grad():
                noisy_residual = self.unet(input_image, torch.tensor([t]).to(device))
            previous_noisy_sample = self.noise_scheduler.step(noisy_residual, t, input_image).prev_sample

            input_image = previous_noisy_sample

        # 5. ae decode
        x = self.decode(input_image)

        return x
