import torch
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Union, List, Set
from torch.utils.data import Dataset
from utils.file_metrics_sort import sort_ptbxl_metric_data
from utils.scalogram_utils import scalogram_to_model_input


class PTBXLDataset(Dataset):
    def __init__(
            self,
            ref_min: float,
            ref_max: float,
            superlet_dir: Union[str, Path],
            discretize: bool,
            train: bool,
            include_all_folds: bool = False,
    ):
        """
        PTB-XL dataset wrapper for scalogram input.

        Args:
            ref_min (float): Minimum reference value for normalization.
            ref_max (float): Maximum reference value for normalization.
            superlet_dir (Path): Directory containing .npz superlet files.
            discretize (bool): Whether to apply discretization to scalogram input.
            train (bool): If True, load training file list; else, evaluation set.
            include_all_folds (bool): If True and not training, use all folds for evaluation.
        """
        if train and include_all_folds:
            raise ValueError(
                "Invalid argument: 'include_all_folds=True' is not allowed when 'train=True'"
            )

        self.superlet_dir = Path(superlet_dir)
        self.ref_min = ref_min
        self.ref_max = ref_max
        self.discretize = discretize

        data_dir = Path(__file__).resolve().parent / 'database'

        if train:
            csv_path = data_dir / "train_clean.csv"
        else:
            csv_path = (
                data_dir / "ptbxl_all_measurement.csv"
                if include_all_folds
                else data_dir / "ptbxl_noise_eval.csv"
            )

        df = pd.read_csv(csv_path)
        self.files = df['filename_hr'].tolist()
        self.indices = df['idx'].tolist()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.FloatTensor:
        file_path = self.superlet_dir / f"{self.files[idx]}_{self.indices[idx]}.npz"
        data = np.load(file_path)
        scalogram = data["scalogram"]

        transformed = scalogram_to_model_input(
            scalogram,
            self.ref_min,
            self.ref_max,
            discretize=self.discretize
        )

        return torch.FloatTensor(transformed).unsqueeze(0)  # shape: [1, H, W]


def burst_static_intersection(
        static_file: str,
        burst_file: str,
        metric: str,
        metadata: str,
        static_percentage: float,
        burst_percentage: float,
) -> List[str]:
    """
    Select top-n % leads from both static and burst metric files and return their intersection.

    Args:
        static_file: Path to CSV with static noise metrics.
        burst_file: Path to CSV with burst noise metrics.
        metric: Metric name ('psnr', 'ssim', 'mse', 'mae').
        metadata: Metadata key to filter filenames.
        static_percentage: Fraction (0< p <=1) for static top-n selection.
        burst_percentage: Fraction (0< p <=1) for burst top-n selection.

    Returns:
        List of filenames present in both top-n static and burst selections.
    """

    # Helper to get top-n percent filenames
    def get_top_n(file_path: str, pct: float, reverse: bool) -> Set[str]:
        container = sort_ptbxl_metric_data(file_path, train=True)
        files = container.get_filenames_by_metadata(metadata)
        n = max(1, int(len(files) * pct))
        # reverse=False => lowest values first; reverse=True => the highest first
        sorted_files = list(files)
        if reverse:
            sorted_files = sorted_files[::-1]
        return set(sorted_files[:n])

    # Determine ordering: higher better for PSNR/SSIM
    higher_better = metric in ('ssim', 'psnr')

    static_set = get_top_n(static_file, static_percentage, reverse=higher_better)
    burst_set = get_top_n(burst_file, burst_percentage, reverse=higher_better)

    intersection = sorted(static_set & burst_set)

    # Summary print
    print(f"Static top {static_percentage * 100:.1f}% count: {len(static_set)}")
    print(f"Burst top {burst_percentage * 100:.1f}% count: {len(burst_set)}")
    print(f"Intersection count: {len(intersection)}")

    return intersection


class PTBXLRefinedDataset(Dataset):
    """
    PyTorch Dataset selecting superlet files that are top-N% for both static and burst noise metrics.
    """

    def __init__(
            self,
            superlet_dir: Union[str, Path],
            static_file: Union[str, Path],
            burst_file: Union[str, Path],
            metric: str,
            static_percentage: float,
            burst_percentage: float,
            metadata: str = "clean",
            ref_min: float = 0.0,
            ref_max: float = -8.0,
            discretize: bool = True
    ):
        """
        Args:
            superlet_dir: Directory containing .npz superlet files.
            static_file: CSV path for static noise metrics.
            burst_file: CSV path for burst noise metrics.
            metric: Metric name for ranking.
            static_percentage: Fraction for static top-N selection.
            burst_percentage: Fraction for burst top-N selection.
            metadata: Metadata key to filter filenames.
            ref_min: Minimum normalization value.
            ref_max: Maximum normalization value.
            discretize: Whether to discretize scalogram input.
        """
        self.superlet_dir = Path(superlet_dir)
        self.metric = metric
        self.metadata = metadata
        self.ref_min = ref_min
        self.ref_max = ref_max
        self.discretize = discretize

        # Determine file list by intersection
        self.files: List[str] = burst_static_intersection(
            static_file, burst_file, metric, metadata,
            static_percentage, burst_percentage
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.FloatTensor:
        filename = self.files[idx]
        file_path = self.superlet_dir / filename
        data = np.load(file_path)
        scalogram = data['scalogram']  # shape: (leads, segments, H, W) or similar

        # Convert to model input
        tensor = scalogram_to_model_input(
            scalogram, self.ref_min, self.ref_max, discretize=self.discretize
        )
        return torch.FloatTensor(tensor).unsqueeze(0)  # shape: [1, H, W]
