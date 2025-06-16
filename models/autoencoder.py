import torch

import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.autoencoders.vae import Encoder, Decoder


class Autoencoder(nn.Module):
    def __init__(
            self,
            model_channels,
            down_block_types,
            up_block_types,
            channel_mult=(1, 2, 4),
            layers_per_block=2,
            latent_channels=4,
            in_channels=1,
            out_channels=1,
            act_fn="silu",
            norm_num_groups=32,
            scaling_factor=0.18215,
            use_quant_conv=True,
            use_post_quant_conv=True,
            mid_block_add_attention=True,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor

        # input
        block_out_channels = tuple(i * model_channels for i in channel_mult)

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=False,
            mid_block_add_attention=mid_block_add_attention,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        if self.quant_conv is not None:
            enc = self.quant_conv(enc)

        return enc

    def encode(self, x):
        h = self._encode(x)

        return h

    def _decode(self, z):
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        dec = self.decoder(z)

        return dec

    def decode(self, z):
        decoded = self._decode(z)

        return decoded

    def forward(self, x):
        sample = x.clone()

        x = self.encode(x)
        x = self.decode(x)

        loss = F.mse_loss(sample, x)

        return loss


def build_ae():
    return Autoencoder(
        model_channels=64,
        down_block_types=(
            "DownEncoderBlock2D",
            "AttnDownEncoderBlock2D",
            "AttnDownEncoderBlock2D"
        ),
        up_block_types=(
            "AttnUpDecoderBlock2D",
            "AttnUpDecoderBlock2D",
            "UpDecoderBlock2D"
        ),
        channel_mult=(1, 2, 4),
        layers_per_block=2,
        latent_channels=2,
    )
