from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

OBJ4_DIR = Path(__file__).resolve().parent
ROOT = OBJ4_DIR.parent

sys.path.insert(0, str(ROOT / "sumo"))

from lstm_predictor import load_lstm
import baseline as baselines


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_lstm(
        str(OBJ4_DIR / "lstm_col6_masked.pt"),
        device="cpu",
    ).to(device)
    model.eval()

    print(f"Device: {device}")

    # Load 20%-penetration test runs only.
    data_dir = ROOT / "data_corrupted" / "penetration_20"

    keys = [
        "Y",
        "X_mask",
        "X_corrupted",
        "X_ch1_imputed",
        "X_fully_imputed",
    ]
    collected = {key: [] for key in keys}

    for seed in range(17, 21):
        path = data_dir / f"run_seed_{seed}.npz"

        with np.load(path, allow_pickle=False) as data:
            for key in keys:
                collected[key].append(
                    data[key].astype(np.float32)
                )

    data = {
        key: np.concatenate(parts, axis=0)
        for key, parts in collected.items()
    }

    Y = data["Y"]
    missing_col6 = data["X_mask"][:, :, 6] == 0

    print(f"Test windows: {len(Y):,}")
    print(f"Column-6 absence: {100 * missing_col6.mean():.6f}%")

    conditions = {
        "Baseline 1: Zero-fill":
            baselines.zero_fill(data["X_corrupted"].copy()),

        "Baseline 5: Spline Interpolation":
            baselines.spline_interp(data["X_corrupted"].copy()),

        "Condition B: Channel 1 Only":
            data["X_ch1_imputed"],

        "Condition C: Dual-Channel (Proposed)":
            data["X_fully_imputed"],
    }

    sx_mean = model.scaler_X_mean.cpu().numpy()
    sx_std = model.scaler_X_std.cpu().numpy()

    horizon_rows = []
    summary_rows = []

    with torch.no_grad():
        for condition, inputs in conditions.items():
            X = inputs.astype(np.float32).copy()

            # Preserve actual column-6 absence after imputation.
            X[:, :, 6] = np.where(
                missing_col6, np.nan, X[:, :, 6]
            )

            # Physical zero-fill, then normalize.
            X = np.where(np.isnan(X), 0.0, X)
            X_norm = (X - sx_mean) / sx_std

            if not np.isfinite(X_norm).all():
                raise ValueError(f"Invalid inputs: {condition}")

            error_sums = np.zeros(Y.shape[1], dtype=np.float64)

            for start in range(0, len(Y), 512):
                end = start + 512

                batch = torch.from_numpy(
                    X_norm[start:end].astype(np.float32)
                ).to(device)

                predictions = model.denormalise_y(model(batch))
                predictions = predictions.cpu().numpy()

                # Distance between predicted and true x,y positions.
                errors = np.linalg.norm(
                    predictions - Y[start:end],
                    axis=2,
                )

                if not np.isfinite(errors).all():
                    raise ValueError(f"Invalid errors: {condition}")

                error_sums += errors.sum(axis=0, dtype=np.float64)

            errors_at_t = error_sums / len(Y)

            ade = float(errors_at_t.mean())
            fde = float(errors_at_t[-1])

            for step, error in enumerate(errors_at_t, start=1):
                horizon_rows.append({
                    "Condition": condition,
                    "Timestep": step,
                    "Time (s)": round(step * 0.1, 1),
                    "ADE_at_t (m)": round(float(error), 3),
                })

            summary_rows.append({
                "Condition": condition,
                "ADE (m)": ade,
                "FDE (m)": fde,
            })

            print(f"\n{condition}")
            print(f"  ADE: {ade:.4f} m")
            print(f"  FDE: {fde:.4f} m")

    pd.DataFrame(horizon_rows).to_csv(
        OBJ4_DIR / "objective4_table3.csv",
        index=False,
    )

    pd.DataFrame(summary_rows).to_csv(
        OBJ4_DIR / "objective4_summary.csv",
        index=False,
    )

    print("\nSaved objective4_table3.csv and objective4_summary.csv")


if __name__ == "__main__":
    main()