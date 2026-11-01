from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data_corrupted"
SCENARIOS = ["penetration_10", "penetration_20", "penetration_30"]
TRAIN_SEEDS = range(1, 15)


def load_training_mask_bank():
    """Collect column-6 missingness patterns from 20% training data."""
    masks = []

    for seed in TRAIN_SEEDS:
        path = (
            DATA_DIR / "penetration_20" / f"run_seed_{seed}.npz"
        )

        with np.load(path, allow_pickle=False) as data:
            # Shape: (number of windows, 30).
            # True means follower speed is missing.
            missing = data["X_mask"][:, :, 6] == 0
            masks.append(missing)

    return np.concatenate(masks, axis=0)


def apply_col6_masks(X_gt, mask_bank, rng):
    """Hide each follower-speed entry independently at the measured rate."""
    X_masked = X_gt.astype(np.float32).copy()

    # Measure absence probability from the actual 20% training masks.
    # True = missing, so their mean is the missing fraction.
    absence_rate = float(mask_bank.mean())

    # Generate one random number for each window and timestep.
    # Shape: (number_of_windows, 30)
    missing = rng.random(X_masked.shape[:2]) < absence_rate

    # Hide only column 6; retain observed follower-speed values.
    X_masked[:, :, 6] = np.where(
        missing,
        np.nan,
        X_masked[:, :, 6]
    )

    return X_masked, missing


def preview_masking():
    mask_bank = load_training_mask_bank()

    print(f"Mask bank shape: {mask_bank.shape}")
    print(f"Source absence rate: {100 * mask_bank.mean():.6f}%")

    # Fixed seed makes the sampled masks reproducible.
    rng = np.random.default_rng(42)

    for scenario in SCENARIOS:
        missing_count = 0
        total_count = 0

        for seed in TRAIN_SEEDS:
            path = DATA_DIR / scenario / f"run_seed_{seed}.npz"

            with np.load(path, allow_pickle=False) as data:
                X_gt = data["X_gt"]
                X_masked, masks = apply_col6_masks(
                    X_gt, mask_bank, rng
                )

                # Confirm that columns 0–5 were not changed.
                np.testing.assert_allclose(
                    X_masked[:, :, :6],
                    X_gt.astype(np.float32)[:, :, :6],
                    rtol=0,
                    atol=0,
                    equal_nan=True
                )

                # Confirm selected follower-speed entries are missing.
                assert np.isnan(X_masked[:, :, 6][masks]).all()

                missing_count += int(masks.sum())
                total_count += masks.size

        print(
            f"{scenario}: applied absence rate="
            f"{100 * missing_count / total_count:.4f}%"
        )

    print("Checks passed. Original data files were not changed.")


if __name__ == "__main__":
    preview_masking()