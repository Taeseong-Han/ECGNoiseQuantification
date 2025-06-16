import torch
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Union
from torch.utils.data import Dataset
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

