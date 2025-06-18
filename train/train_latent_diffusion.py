import torch
import argparse

from pathlib import Path
from train.train_utils import train_loop
from torch.utils.data import DataLoader
from accelerate import notebook_launcher
from data.ptbxl_dataset import PTBXLDataset
from models.latent_diffusion import LatentDiffusionModel
from diffusers.optimization import get_cosine_schedule_with_warmup


def get_parser():
    parser = argparse.ArgumentParser(description="Train Latent Diffusion on PTB-XL Superlet Dataset")

    parser.add_argument('--log_dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / "output/logs")
    parser.add_argument('--superlet_dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/database/ptbxl_superlet32")
    parser.add_argument('--save_path', type=Path,
                        default=Path(__file__).resolve().parents[1] / "output/ldm_model")
    parser.add_argument('--ae_checkpoint_path', type=Path,
                        default=Path(__file__).resolve().parents[1] / "output/ae_model/ema_100.pth",
                        help='Path to the pretrained autoencoder weights')

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--ema_rate', type=float, default=0.999,
                        help="Exponential moving average rate for model weights.")
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr_warmup_steps', type=int, default=0)
    parser.add_argument('--num_train_timesteps', type=int, default=1000)
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--num_processes', type=int, default=1,
                        help='Number of processes to launch for distributed training (used by notebook_launcher).')

    parser.add_argument('--ref_min', type=float, default=0.0,
                        help="Minimum reference value for scalogram normalization.")
    parser.add_argument('--ref_max', type=float, default=-8.0,
                        help="Maximum reference value for scalogram normalization.")
    parser.add_argument('--discretization', action='store_false')
    parser.add_argument('--noise_scheduler_type', type=str, default='ddpm', choices=['ddpm', 'ddim'])
    parser.add_argument('--seed', type=int, default=123)

    return parser


def main(args):
    torch.manual_seed(args.seed)

    dataset = PTBXLDataset(
        ref_min=args.ref_min,
        ref_max=args.ref_max,
        superlet_dir=args.superlet_dir,
        discretize=args.discretization,
        train=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = LatentDiffusionModel(
        ae_weight_path=args.ae_checkpoint_path,
        noise_scheduler_type=args.noise_scheduler_type,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if args.lr_warmup_steps > 0:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=len(dataloader) * args.lr_warmup_steps,
            num_training_steps=len(dataloader) * args.num_epochs,
        )
    else:
        lr_scheduler = None

    # Prepare training arguments tuple
    training_args = (args, model, optimizer, dataloader, lr_scheduler)

    notebook_launcher(train_loop, training_args, num_processes=args.num_processes)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
