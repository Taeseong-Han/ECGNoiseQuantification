from typing import List
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
