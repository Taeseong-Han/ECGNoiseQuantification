#!/usr/bin/env python3
import argparse
from pathlib import Path

from evaluation.eval_utils import noise_quantification_ptbxl


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ECG noise quantification over the PTB-XL dataset."
    )

    # Model checkpoint and data directories
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help="Path to the trained diffusion model checkpoint."
    )
    parser.add_argument(
        '--superlet_dir',
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/database/ptbxl_superlet32",
        help="Directory containing precomputed superlet scalograms."
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="Directory where CSV results will be saved."
    )

    # Diffusion sampling parameters
    parser.add_argument(
        '--noise_scheduler_type',
        choices=['ddpm', 'ddim'],
        default='ddim',
        help="Noise scheduler type for diffusion sampling."
    )
    parser.add_argument(
        '--timestep',
        type=int,
        required=True,
        help="Diffusion timestep for reverse sampling (1–1000)."
    )
    parser.add_argument(
        '--step_interval',
        type=int,
        default=1,
        help="Interval between diffusion timesteps when generating metrics."
    )
    parser.add_argument(
        '--use_ldm',
        action='store_true',
        help="Use latent diffusion model instead of vanilla diffusion."
    )

    # Quantification settings
    parser.add_argument(
        '--ref_min',
        type=float,
        default=-8,
        help="Minimum value for scalogram normalization."
    )
    parser.add_argument(
        '--ref_max',
        type=float,
        default=0,
        help="Maximum value for scalogram normalization."
    )
    parser.add_argument(
        '--discretize',
        action='store_true',
        help="Discretize images before computing metrics."
    )

    # Evaluation options
    parser.add_argument(
        '--include_all_folds',
        action='store_true',
        help=(
            "If set, evaluate both clean and noisy samples across all 10 folds; "
            "otherwise, only clean samples from fold 10 and noisy samples from all folds."
        )
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Batch size for dataset loading."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help="Random seed for reproducibility."
    )

    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    # Validate timestep range
    if not (1 <= args.timestep <= 1000):
        parser.error("--timestep must be between 1 and 1000.")

    noise_quantification_ptbxl(
        checkpoint_path=args.checkpoint,
        ref_min=args.ref_min,
        ref_max=args.ref_max,
        superlet_dir=args.superlet_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        diffusion_timestep=args.timestep,
        noise_scheduler_type=args.noise_scheduler_type,
        use_ldm=args.use_ldm,
        step_interval=args.step_interval,
        include_all_folds=args.include_all_folds,
        discretize=args.discretize,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
