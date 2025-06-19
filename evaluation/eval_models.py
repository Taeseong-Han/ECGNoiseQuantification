from typing import List
from pathlib import Path

from eval_metrics import compute_w1_distance
from utils.file_metrics_sort import sort_ptbxl_metric_data

def find_result_files(
        directory: Path,
        keyword: str,
        extension: str = "csv"
) -> List[Path]:
    """
    Return list of result file paths containing the given keyword,
    excluding files that contain 'all' in their name.

    Args:
        directory (Path): Directory to search in.
        keyword (str): Substring that should be present in the filename.
        extension (str): File extension to filter (default: 'csv').

    Returns:
        List[Path]: List of matching file paths.
    """
    # Use iterdir for listing and filter by suffix and keywords
    return sorted([
        p for p in directory.iterdir()
        if p.suffix == f".{extension}" and keyword in p.name and "all" not in p.name
    ])

def evaluate_models(
        keyword: str,
        directory: Path = Path("./results")
) -> None:
    """
    Evaluate all result files in `directory` matching `keyword`.
    Computes W1 distances between clean and various noise types.

    Args:
        keyword (str): Keyword to filter result files.
        directory (Path): Directory containing result CSV files.
    """
    files = find_result_files(directory, keyword)
    if not files:
        print(f"No files found in {directory} matching keyword '{keyword}'")
        return

    for file_path in files:
        container = sort_ptbxl_metric_data(file_path, train=False)
        clean_vals = container.get_values_by_metadata('clean')
        # Compute W1 distances for each noise category
        results = {
            noise: compute_w1_distance(clean_vals, container.get_values_by_metadata(noise))
            for noise in ('static', 'burst', 'baseline')
        }
        # Print formatted output
        print(
            f"{file_path.name} | "
            f"W1(clean vs. static): {results['static']:.4f} | "
            f"W1(clean vs. burst): {results['burst']:.4f} | "
            f"W1(clean vs. baseline): {results['baseline']:.4f}"
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute W1 distances for PTBXL noise quantification results"
    )
    parser.add_argument(
        "--keyword", type=str, default="ddpm",
        help="Keyword to filter result files by name"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("./results"),
        help="Directory containing the result CSV files"
    )
    args = parser.parse_args()

    evaluate_models(keyword=args.keyword, directory=args.output_dir)
