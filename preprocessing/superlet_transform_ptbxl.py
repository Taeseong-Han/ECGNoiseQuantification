import os
import wfdb
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from preprocessing.superlet_transform import compute_superlet_scalogram


def make_dir(path: Path) -> None:
    """Safely create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_labels(label_path: Path) -> pd.DataFrame:
    """Load the PTB-XL parsed label file."""
    return pd.read_csv(label_path)


def process_record(filename: str, raw_path: Path, save_path: Path, freq_len: int, sampling_freq: int) -> None:
    """Process a single ECG record and save per-lead scalogram."""
    record_path = raw_path / filename
    signal = wfdb.rdsamp(record_path)[0].T  # shape: (12, time)

    scalogram = compute_superlet_scalogram(signal, sampling_freq=sampling_freq, freqs_len=freq_len)

    sub_dir = save_path / filename.split('/')[0] / filename.split('/')[1]
    make_dir(sub_dir)

    for lead_idx in range(12):
        np.savez(
            save_path / f"{filename}_{lead_idx}",
            signal=signal[lead_idx],
            scalogram=scalogram[lead_idx],
        )


def superlet_transform_ptbxl(
        ptbxl_raw_path: Path,
        label_csv_path: Path,
        save_dirname: str = 'ptbxl_superlet32',
        sampling_freq: int = 500,
        freq_len: int = 32,
) -> None:
    """Compute superlet scalograms for PTB-XL ECG dataset and save to disk."""
    project_root = Path(__file__).resolve().parents[1]
    save_path = project_root / 'data/database' / save_dirname
    make_dir(save_path)

    df = load_labels(label_csv_path)

    for filename in tqdm(df.filename_hr, desc="Processing PTB-XL records"):
        process_record(filename, ptbxl_raw_path, save_path, freq_len, sampling_freq)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute superlet scalograms for PTB-XL ECG recordings."
    )
    parser.add_argument(
        '--ptbxl_raw_path',
        type=Path,
        required=True,
        help="Path to raw PTB-XL dataset (e.g., ~/Database/physionet.org/files/ptb-xl/1.0.3)."
    )
    parser.add_argument(
        '--label_csv_path',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'data/database/ptbxl_label_dropna.csv',
        help="Path to CSV file with parsed PTB-XL noise labels."
    )
    parser.add_argument(
        '--save_dirname',
        type=str,
        default='ptbxl_superlet32',
        help="Directory name under data/database to save scalograms."
    )
    parser.add_argument(
        '--freq_len',
        type=int,
        default=32,
        help="Number of frequency bins to use for superlet transform."
    )
    parser.add_argument(
        '--sampling_freq',
        type=int,
        default=500,
        help="Sampling frequency of the raw PTB-XL signals (default: 500Hz)."
    )

    return parser.parse_args()


if __name__ == "__main__":
    """
    Superlet scalogram generator for PTB-XL.

    Download raw ECG data from:
    https://physionet.org/content/ptb-xl/1.0.3/

    Expected structure:
    ~/Database/physionet.org/files/ptb-xl/1.0.3/
    """

    args = parse_args()

    superlet_transform_ptbxl(
        ptbxl_raw_path=args.ptbxl_raw_path,
        label_csv_path=args.label_csv_path,
        save_dirname=args.save_dirname,
        freq_len=args.freq_len,
        sampling_freq=args.sampling_freq,
    )
