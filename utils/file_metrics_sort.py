import ast

import pandas as pd

from pathlib import Path
from typing import List, Union
from dataclasses import dataclass, field


@dataclass
class FileData:
    """Container for a single file's metadata and associated metric value."""
    filename: str
    metric_value: float
    metadata: str


@dataclass
class FileDataset:
    """A dataset of files with metric values and metadata, supporting filtering and sorting."""
    files: List[FileData] = field(default_factory=list)

    def add_files(self, new_files: List[FileData]) -> None:
        """Append a list of FileData entries to the dataset."""
        self.files.extend(new_files)

    def sort_by_value(self) -> None:
        """Sort the dataset in-place by metric_value in ascending order."""
        self.files.sort(key=lambda x: x.metric_value)

    def get_filenames(self) -> List[str]:
        """Return list of all filenames in the dataset."""
        return [f.filename for f in self.files]

    def get_values(self) -> List[float]:
        """Return list of all metric values in the dataset."""
        return [f.metric_value for f in self.files]

    def get_filenames_by_metadata(self, keyword: str) -> List[str]:
        """Return filenames where metadata contains the specified keyword."""
        return [f.filename for f in self.files if keyword in f.metadata]

    def get_values_by_metadata(self, keyword: str) -> List[float]:
        """Return metric values where metadata contains the specified keyword."""
        return [f.metric_value for f in self.files if keyword in f.metadata]


def build_metadata(label: str, is_norm: str, lead: int, types: list[str] = None) -> str:
    parts = [label, is_norm, f'lead{lead}']
    if types:
        parts.extend(types)
    return ','.join(parts)


def build_filedata(filename: str, value: float, metadata: str) -> FileData:
    return FileData(
        filename=f"{filename}.npz",
        metric_value=value,
        metadata=metadata
    )


def sort_ptbxl_metric_data(metric_file_path: Union[str, Path], train: bool = False) -> FileDataset:
    """
    Load PTB-XL metric data and return a sorted FileDataset by metric value.

    Args:
        metric_file_path: Path to a CSV file containing lead-level metric values and labels.
        train: If True, include folds 1–9 (clean leads only); if False, use fold 10 clean + all noise.

    Returns:
        FileDataset: Sorted list of FileData entries.
    """
    df = pd.read_csv(metric_file_path)
    if train:
        df = df[df['strat_fold'].isin(range(1, 10))].reset_index(drop=True)

    dataset = FileDataset()

    for _, row in df.iterrows():
        filename = row['filename_hr']
        strat_fold = int(row['strat_fold'])
        metric_values = row.iloc[-12:]

        clean_idx = ast.literal_eval(row['clean'])
        noise_idx = sorted(set(range(12)) - set(clean_idx))

        baseline_idx = set(ast.literal_eval(row['baseline_drift']))
        static_idx = set(ast.literal_eval(row['static_noise']))
        burst_idx = set(ast.literal_eval(row['burst_noise']))
        electrode_idx = set(ast.literal_eval(row['electrodes_problems']))

        is_norm = 'norm' if "'NORM': 100" in row['scp_codes'] else 'abnorm'

        # Clean leads
        if train or strat_fold == 10:
            for i in clean_idx:
                metadata = build_metadata('clean', is_norm, i)
                filedata = build_filedata(f'{filename}_{i}', metric_values[i], metadata)
                dataset.add_files([filedata])

        # Noise leads (always included)
        for i in noise_idx:
            types = []
            if i in baseline_idx:
                types.append('baseline')
            if i in static_idx:
                types.append('static')
            if i in burst_idx:
                types.append('burst')
            if i in electrode_idx:
                types.append('electrode')

            metadata = build_metadata('noise', is_norm, i, types)
            filedata = build_filedata(f'{filename}_{i}', metric_values[i], metadata)
            dataset.add_files([filedata])

    dataset.sort_by_value()
    return dataset


if __name__ == "__main__":
    # Example usage
    file_data_list = [
        FileData(filename="file1.txt", metric_value=10.5, metadata="meta1"),
        FileData(filename="file2.txt", metric_value=5.3, metadata="meta2"),
        FileData(filename="file3.txt", metric_value=8.7, metadata="meta3"),
    ]

    # Create a FileDataset object
    dataset = FileDataset()

    # Add multiple files to the dataset
    dataset.add_files(file_data_list)

    # Sort files by value
    dataset.sort_by_value()

    print(dataset.get_values())
    print(dataset.get_filenames())
    print(dataset.get_filenames_by_metadata('meta1'))
