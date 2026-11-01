from pathlib import Path
import sys
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from objective4_data import (
    DATA_DIR,
    SCENARIOS,
    load_training_mask_bank,
    apply_col6_masks,
)

# Folder containing this training script.
OBJ4_DIR = Path(__file__).resolve().parent

# Shared parent containing sumo, obj4 and data_corrupted.
ROOT = OBJ4_DIR.parent

sys.path.insert(0, str(ROOT / "sumo"))

from lstm_predictor import LSTMPredictor, load_lstm, train_lstm

BASELINE_PATH = ROOT / "sumo" / "lstm_baseline.pt"
OUTPUT_PATH = OBJ4_DIR / "lstm_col6_masked.pt"

RANDOM_SEED = 42


def build_dataset(seeds, mask_bank, scalers, mask_seed):
    sx_mean, sx_std, sy_mean, sy_std = scalers

    rng = np.random.default_rng(mask_seed)
    inputs = []
    targets = []

    missing_count = 0
    entry_count = 0

    for scenario in SCENARIOS:
        for seed in seeds:
            path = DATA_DIR / scenario / f"run_seed_{seed}.npz"

            with np.load(path, allow_pickle=False) as data:
                X_gt = data["X_gt"].astype(np.float32)
                Y = data["Y"].astype(np.float32)

            X_masked, masks = apply_col6_masks(
                X_gt, mask_bank, rng
            )

            missing_count += int(masks.sum())
            entry_count += masks.size

            # Match Table 3: physical zero-fill BEFORE normalization.
            if np.isinf(X_masked).any():
                raise ValueError(f"Infinite input value in {path}")

            X_filled = np.where(
                np.isnan(X_masked), 0.0, X_masked
            )

            X_norm = (X_filled - sx_mean) / sx_std
            Y_norm = (Y - sy_mean) / sy_std

            if not np.isfinite(X_norm).all():
                raise ValueError(f"Invalid normalized inputs: {path}")

            if not np.isfinite(Y_norm).all():
                raise ValueError(f"Invalid normalized targets: {path}")

            inputs.append(X_norm.astype(np.float32))
            targets.append(Y_norm.astype(np.float32))

    X = torch.from_numpy(np.concatenate(inputs, axis=0))
    Y = torch.from_numpy(np.concatenate(targets, axis=0))

    print(f"  Windows: {len(X):,}")
    print(f"  Applied masking rate: {100 * missing_count / entry_count:.6f}%")

    return TensorDataset(X, Y)


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Load the original model only to reuse its normalization scalers.
    baseline = load_lstm(str(BASELINE_PATH), device="cpu")

    scalers = (
        baseline.scaler_X_mean.cpu().numpy().copy(),
        baseline.scaler_X_std.cpu().numpy().copy(),
        baseline.scaler_Y_mean.cpu().numpy().copy(),
        baseline.scaler_Y_std.cpu().numpy().copy(),
    )

    mask_bank = load_training_mask_bank()
    print(f"Source absence rate: {100 * mask_bank.mean():.6f}%")

    print("\nPreparing training data: seeds 1–14")
    train_dataset = build_dataset(
        seeds=range(1, 15),
        mask_bank=mask_bank,
        scalers=scalers,
        mask_seed=42,
    )

    print("\nPreparing validation data: seeds 15–16")
    val_dataset = build_dataset(
        seeds=range(15, 17),
        mask_bank=mask_bank,
        scalers=scalers,
        mask_seed=43,
    )

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    sx_mean, sx_std, sy_mean, sy_std = scalers

    # Fresh model: same architecture, randomly initialized weights.
    model = LSTMPredictor(
        input_dim=7,
        hidden_dim=128,
        num_layers=2,
        output_steps=50,
        output_dim=2,
        scaler_X_mean=torch.from_numpy(sx_mean).float(),
        scaler_X_std=torch.from_numpy(sx_std).float(),
        scaler_Y_mean=torch.from_numpy(sy_mean).float(),
        scaler_Y_std=torch.from_numpy(sy_std).float(),
    )

    train_lstm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        lr=1e-3,
        device=device,
        save_path=str(OUTPUT_PATH),
    )

    print(f"\nSaved masked-training checkpoint: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()