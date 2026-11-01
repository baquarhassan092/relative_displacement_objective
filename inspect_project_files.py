
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch


def inspect_csv(path):
    """Display the contents of a CSV file."""
    df = pd.read_csv(path)

    print("\nCSV FILE:", path)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nContents:")
    print(df.to_string(index=False))


def inspect_npz(path):
    """Display arrays stored in an NPZ dataset."""
    with np.load(path, allow_pickle=False) as data:
        print("\nNPZ FILE:", path)
        print("Arrays:", data.files)

        for name in data.files:
            array = data[name]

            print(f"\nArray: {name}")
            print("Shape:", array.shape)
            print("Data type:", array.dtype)

            if array.size and np.issubdtype(
                array.dtype, np.number
            ):
                finite = array[np.isfinite(array)]
                if finite.size:
                    print("Minimum:", finite.min())
                    print("Maximum:", finite.max())
                    print("Mean:", finite.mean())

            # Show only a small preview.
            print("Preview:")
            print(array.reshape(-1)[:10])


def inspect_pt(path):
    """Inspect a PyTorch checkpoint without loading it onto a GPU."""
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    print("\nPT CHECKPOINT:", path)
    print("Checkpoint type:", type(checkpoint).__name__)

    def show_item(name, value):
        if isinstance(value, torch.Tensor):
            print(
                f"{name}: Tensor, "
                f"shape={tuple(value.shape)}, "
                f"dtype={value.dtype}"
            )
        elif isinstance(value, dict):
            print(f"{name}: dictionary ({len(value)} entries)")
            for key, item in value.items():
                show_item(f"{name}.{key}", item)
        elif isinstance(value, (int, float, str, bool)):
            print(f"{name}: {value}")
        else:
            print(f"{name}: {type(value).__name__}")

    if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            show_item(key, value)
    else:
        show_item("checkpoint", checkpoint)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Objective 1 project files"
    )
    parser.add_argument(
        "type",
        choices=["csv", "npz", "pt"],
        help="File type to inspect"
    )
    parser.add_argument("path", help="Path to the file")
    args = parser.parse_args()

    path = Path(args.path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if args.type == "csv":
        inspect_csv(path)
    elif args.type == "npz":
        inspect_npz(path)
    elif args.type == "pt":
        inspect_pt(path)


if __name__ == "__main__":
    main()
