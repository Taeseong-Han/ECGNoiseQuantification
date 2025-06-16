import re

import numpy as np
import pandas as pd

from typing import List
from pathlib import Path


def noise_label_to_idx(noise_label: str):
    """
    Convert noise label string to a list of lead indices (0–11) indicating noisy channels.
    Returns [np.nan] if unknown label is encountered.
    """
    channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    keywords = ['DRIFT', 'NOISYRECORDING', 'NOISE', 'LEICHT', 'MITTEL', 'STARK']
    pattern = r"V[1-6]"
    noise_idx = []

    if pd.isna(noise_label):
        return noise_idx

    label = (noise_label.strip().replace(' ', '').upper()
             .replace('--', '-').replace('.', ',').replace(';', ','))
    label_parts = [part for part in label.split(',') if part]

    for part in label_parts:
        if part in channels:
            noise_idx.append(channels.index(part))

        elif '-' in part:
            start, end = part.split('-')
            if start in channels and end in channels:
                start_idx = channels.index(start)
                end_idx = channels.index(end)
                noise_idx.extend(range(start_idx, end_idx + 1))

        elif part in {'ALLES', 'ALLE', 'ALLEABL'}:
            return list(range(12))

        elif part in keywords:
            if not noise_idx:
                return list(range(12))

        elif part in {"1", "2", "3", "4", "5", "6"}:
            noise_idx.append(channels.index(f'V{part}'))

        elif re.search(pattern, part):
            for result in re.findall(pattern, part):
                noise_idx.append(channels.index(result))

        elif part in {"KONTAKTPROBLEMEAVL???", "ELEKTRODENVERTAUSCHT???", "STARKEDRIFT"}:
            if not noise_idx:
                return list(range(12))

        elif "???" in part:
            base = part.split('???')[0]
            if base in channels:
                noise_idx.append(channels.index(base))

        elif part in {'ISTARK', 'MITTELI'}:
            noise_idx.append(channels.index("I"))

        else:
            print(f"{part}: This part is an unknown noise label")
            return [np.nan]

    return sorted(noise_idx)


def ptbxl_label_parser(high_res: bool = True) -> None:
    """
    Parse PTB-XL noise annotations and export parsed labels to CSV.

    Args:
        high_res (bool): If True, use high-resolution ECG filenames (500Hz).
                         If False, use low-resolution ECG filenames (100Hz).

    Generates:
        - Full cleaned label CSV: ptbxl_label_dropna.csv
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    Y = pd.read_csv(PROJECT_ROOT / 'data/ptbxl_database.csv')
    filename_col = 'filename_hr' if high_res else 'filename_lr'

    columns = [filename_col, 'scp_codes', 'baseline_drift', 'static_noise',
               'burst_noise', 'electrodes_problems', 'clean', 'strat_fold']
    df = pd.DataFrame(columns=columns)

    for idx, row in Y.iterrows():
        baseline_drift = noise_label_to_idx(row['baseline_drift'])
        static_noise = noise_label_to_idx(row['static_noise'])
        burst_noise = noise_label_to_idx(row['burst_noise'])
        electrodes_problems = noise_label_to_idx(row['electrodes_problems'])

        combined_noise = baseline_drift + static_noise + burst_noise + electrodes_problems
        clean_idx = (np.nan if np.nan in combined_noise
                     else sorted(set(range(12)) - set(combined_noise)))

        df.loc[idx] = [
            row[filename_col],
            row['scp_codes'],
            baseline_drift,
            static_noise,
            burst_noise,
            electrodes_problems,
            clean_idx,
            row['strat_fold']
        ]

    df.dropna(inplace=True)
    df.to_csv(PROJECT_ROOT / 'data/ptbxl_label_dropna.csv', index=False)


if __name__ == "__main__":
    ptbxl_label_parser(high_res=True)
