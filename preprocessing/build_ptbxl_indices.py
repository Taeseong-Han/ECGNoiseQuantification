import ast
import pandas as pd

from typing import List
from pathlib import Path

EXCLUDE_FILES = {'records500/12000/12722_hr'}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE = PROJECT_ROOT / "data/database"


def load_filtered_label(fold: List[int]) -> pd.DataFrame:
    """Load and filter PTB-XL label CSV by fold and known bad samples."""
    df = pd.read_csv(DATABASE / 'ptbxl_label_dropna.csv')
    df = df[~df['filename_hr'].isin(EXCLUDE_FILES)]
    df = df[df['strat_fold'].isin(fold)].reset_index(drop=True)
    return df


def build_train_file(only_norm: bool = False, fold: List[int] = list(range(1, 10))) -> None:
    """Generate training file with clean leads; optionally filter by NORM class."""
    df = load_filtered_label(fold)

    records = []
    for _, row in df.iterrows():
        clean_idx = ast.literal_eval(row['clean'])

        if only_norm:
            if "'NORM': 100" not in row['scp_codes']:
                continue

        for lead in clean_idx:
            records.append([row['filename_hr'], lead])

    filename = 'train_norm.csv' if only_norm else 'train_clean.csv'
    pd.DataFrame(records, columns=['filename_hr', 'idx']).to_csv(DATABASE / filename, index=False)


def build_noise_eval_file(include_all_folds: bool = False) -> None:
    """
    Generate evaluation index for noise measurement.

    If include_all_folds is True:
        - Include all 12 leads from all folds (fold 1 to 10)

    If False:
        - Include:
            - all noisy leads from folds 1–9
            - all [clean+noisy] 12 leads from fold 10 (assumed high-quality clean data for evaluation)

            Wagner, Patrick, et al. "PTB-XL, a large publicly available electrocardiography dataset."
            We propose to use the tenth fold, which is ensured to contain only ECGs that have certainly
            be validated by at least one human cardiologist and are therefore presumably of highest label quality,
            to separate a test set that is only used for the fnal performance evaluation of a proposed algorithm.
    """
    df = pd.read_csv(DATABASE / 'ptbxl_label_dropna.csv')
    records = []

    for _, row in df.iterrows():
        clean_idx = ast.literal_eval(row['clean'])

        if include_all_folds:
            indices = list(range(12))  # use all leads
        elif int(row['strat_fold']) == 10:
            indices = list(range(12))  # clean test set
        else:
            indices = sorted(set(range(12)) - set(clean_idx))  # only noisy leads from folds 1–9

        for lead in indices:
            records.append([row['filename_hr'], lead])

    out_name = 'ptbxl_all_measurement.csv' if include_all_folds else 'ptbxl_noise_eval.csv'
    pd.DataFrame(records, columns=['filename_hr', 'idx']).to_csv(DATABASE / out_name, index=False)


if __name__ == "__main__":
    # All clean leads from folds (1–9)
    build_train_file(only_norm=False)

    # Only leads from recordings labeled as NORM from folds (1–9)
    build_train_file(only_norm=True)

    # Evaluate noise across all leads from all folds (1–10)
    build_noise_eval_file(include_all_folds=True)

    # Evaluate using noisy leads from folds 1–9 and all leads from fold 10
    build_noise_eval_file(include_all_folds=False)
