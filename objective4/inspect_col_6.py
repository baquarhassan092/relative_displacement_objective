from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_corrupted"

splits = {
    "train": range(1, 15),
    "validation": range(15, 17),
    "test": range(17, 21),
}

for scenario in ["penetration_10", "penetration_20", "penetration_30"]:
    print(f"\n{scenario}")

    for split_name, seeds in splits.items():
        total_cells = 0
        missing_cells = 0
        gt_nonfinite = 0
        total_windows = 0
        fully_missing_windows = 0

        for seed in seeds:
            path = DATA_DIR / scenario / f"run_seed_{seed}.npz"

            with np.load(path, allow_pickle=False) as data:
                follower_gt = data["X_gt"][:, :, 6]
                missing = data["X_mask"][:, :, 6] == 0

                total_cells += missing.size
                missing_cells += missing.sum()
                gt_nonfinite += (~np.isfinite(follower_gt)).sum()

                total_windows += missing.shape[0]
                fully_missing_windows += missing.all(axis=1).sum()

        print(
            f"  {split_name}: "
            f"missing cells={100 * missing_cells / total_cells:.2f}%, "
            f"fully missing windows="
            f"{100 * fully_missing_windows / total_windows:.2f}%, "
            f"nonfinite ground truth="
            f"{100 * gt_nonfinite / total_cells:.2f}%"
        )