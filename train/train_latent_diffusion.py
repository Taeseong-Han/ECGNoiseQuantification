import torch
import argparse

from pathlib import Path
from train.train_utils import train_loop
from torch.utils.data import DataLoader
from accelerate import notebook_launcher
from data.ptbxl_dataset import PTBXLDataset
from models.latent_diffusion import LatentDiffusionModel
from diffusers.optimization import get_cosine_schedule_with_warmup


def get_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for training Latent Diffusion Model on PTB-XL Superlet Dataset.
    Groups arguments into data, model, diffusion, and training settings.
    """
    parser = argparse.ArgumentParser(
        description="Train Latent Diffusion on PTB-XL Superlet Dataset"
    )

    # Data paths
    data_group = parser.add_argument_group('Data')
    data_group.add_argument(
        '--superlet_dir',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'data/database/ptbxl_superlet32',
        help='Directory containing precomputed superlet scalograms.'
    )

    # Logging and checkpoints
    io_group = parser.add_argument_group('I/O')
    io_group.add_argument(
        '--log_dir',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'output/logs',
        help='Directory to save training logs.'
    )
    io_group.add_argument(
        '--save_path',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'output/ldm_model',
        help='Path to save the trained LDM model.'
    )
    io_group.add_argument(
        '--ae_checkpoint_path',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'output/ae_model/ema_100.pth',
        help='Path to pretrained autoencoder weights.'
    )

    # Model & diffusion settings
    model_group = parser.add_argument_group('Model')
    model_group.add_argument(
        '--noise_scheduler_type',
        choices=['ddpm', 'ddim'],
        default='ddpm',
        help="Diffusion scheduler type ('ddpm' or 'ddim')."
    )
    model_group.add_argument(
        '--num_train_timesteps',
        type=int,
        default=1000,
        help='Number of diffusion timesteps for training.'
    )
    model_group.add_argument(
        '--mixed_precision',
        choices=['no', 'fp16', 'bf16'],
        default='no',
        help="Mixed precision mode: disabled, 'fp16', or 'bf16'."
    )

    # Training hyperparameters
    train_group = parser.add_argument_group('Training')
    train_group.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Initial learning rate.'
    )
    train_group.add_argument(
        '--ema_rate',
        type=float,
        default=0.999,
        help='Exponential moving average rate for model weights.'
    )
    train_group.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of training epochs.'
    )
    train_group.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for training.'
    )
    train_group.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of steps to accumulate gradients before updating.'
    )
    train_group.add_argument(
        '--lr_warmup_steps',
        type=int,
        default=0,
        help='Warmup steps for learning rate scheduler.'
    )
    train_group.add_argument(
        '--num_processes',
        type=int,
        default=1,
        help='Number of processes for distributed training.'
    )

    # Preprocessing
    prep_group = parser.add_argument_group('Preprocessing')
    prep_group.add_argument(
        '--ref_min',
        type=float,
        default=0.0,
        help='Minimum reference value for scalogram normalization.'
    )
    prep_group.add_argument(
        '--ref_max',
        type=float,
        default=-8.0,
        help='Maximum reference value for scalogram normalization.'
    )
    prep_group.add_argument(
        '--discretization',
        action='store_true',
        help='Enable discretization of scalograms before training.'
    )

    # Misc
    misc_group = parser.add_argument_group('Misc')
    misc_group.add_argument(
        '--seed',
        type=int,
        default=123,
        help='Random seed for reproducibility.'
    )

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
