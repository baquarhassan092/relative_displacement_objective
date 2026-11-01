"""Objective 1: train/evaluate a relative-displacement LSTM.

Expected project layout (objective1 is a sibling of sumo and data_corrupted):
project_root/
  sumo/lstm_predictor.py
  data_corrupted/penetration_{10,20,30}/run_seed_*.npz
  objective1/config.json
  objective1/run_objective1.py

The script uses X_gt as input, converts absolute future positions Y into stepwise
Delta x/Delta y labels, predicts the displacement sequence, reconstructs absolute
future positions, and reports ADE/FDE on held-out test seeds.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG_PATH = HERE / "config.json"


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


CFG = load_config()
DATA_DIR = ROOT / CFG["data_folder_name"]
SUMO_DIR = ROOT / CFG["sumo_folder_name"]
RESULTS_DIR = HERE / "results"
CHECKPOINT_DIR = HERE / "checkpoints"
LOGS_DIR = HERE / "logs"
for directory in (RESULTS_DIR, CHECKPOINT_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

if not (SUMO_DIR / "lstm_predictor.py").is_file():
    raise FileNotFoundError(
        f"Cannot find {SUMO_DIR / 'lstm_predictor.py'}. Keep the objective1, sumo, "
        "and data_corrupted folders as siblings in the project root."
    )
sys.path.insert(0, str(SUMO_DIR))
from lstm_predictor import LSTMPredictor, compute_scalers  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCENARIOS = CFG["scenarios"]
TRAIN_SEEDS = {f"run_seed_{i}.npz" for i in range(CFG["train_seed_start"], CFG["train_seed_end"] + 1)}
VAL_SEEDS = {f"run_seed_{i}.npz" for i in range(CFG["validation_seed_start"], CFG["validation_seed_end"] + 1)}
TEST_SEEDS = {f"run_seed_{i}.npz" for i in range(CFG["test_seed_start"], CFG["test_seed_end"] + 1)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_expected_files(seed_names: Iterable[str]) -> list[Path]:
    paths = []
    for scenario in SCENARIOS:
        folder = DATA_DIR / scenario
        for seed_name in sorted(seed_names):
            path = folder / seed_name
            if path.is_file():
                paths.append(path)
    return paths


def load_arrays(seed_names: set[str]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    paths = list_expected_files(seed_names)
    if not paths:
        raise FileNotFoundError(
            f"No dataset files found for seeds {sorted(seed_names)} under {DATA_DIR}. "
            "Check config.json and the data_corrupted folder."
        )
    missing = []
    for scenario in SCENARIOS:
        for seed_name in sorted(seed_names):
            p = DATA_DIR / scenario / seed_name
            if not p.is_file():
                missing.append(str(p.relative_to(ROOT)))
    if missing:
        print(f"WARNING: {len(missing)} expected scenario/seed files are missing; using existing files.")
        for item in missing[:8]:
            print("  missing:", item)
        if len(missing) > 8:
            print("  ...")

    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            if CFG["input_key"] not in data or CFG["target_key"] not in data:
                raise KeyError(f"{path} must contain {CFG['input_key']!r} and {CFG['target_key']!r}.")
            x = data[CFG["input_key"]].astype(np.float32)
            y = data[CFG["target_key"]].astype(np.float32)
        if x.ndim != 3 or x.shape[1:] != (CFG["history_steps"], CFG["input_dim"]):
            raise ValueError(f"Unexpected input shape in {path}: {x.shape}; expected (N, {CFG['history_steps']}, {CFG['input_dim']}).")
        if y.ndim != 3 or y.shape[1:] != (CFG["output_steps"], CFG["output_dim"]):
            raise ValueError(f"Unexpected target shape in {path}: {y.shape}; expected (N, {CFG['output_steps']}, {CFG['output_dim']}).")
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


class RelativeDisplacementDataset(Dataset):
    """Normalize X and train on per-step displacement targets derived from Y."""
    def __init__(self, X: np.ndarray, Y: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray,
                 delta_mean: np.ndarray | None = None, delta_std: np.ndarray | None = None):
        # Features 2 and 3 are ego_x and ego_y; use the last observed position.
        self.last_position = X[:, -1, 2:4].copy()
        previous_positions = np.concatenate([self.last_position[:, None, :], Y[:, :-1, :]], axis=1)
        delta = Y - previous_positions

        self.delta_mean = delta.mean(axis=(0, 1)).astype(np.float32) if delta_mean is None else delta_mean.astype(np.float32)
        self.delta_std = delta.std(axis=(0, 1)).astype(np.float32) if delta_std is None else delta_std.astype(np.float32)
        self.delta_std = np.maximum(self.delta_std, 1e-6)

        X_norm = (X - x_mean) / x_std
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
        delta_norm = (delta - self.delta_mean) / self.delta_std
        if not np.isfinite(delta_norm).all():
            raise ValueError("Non-finite displacement labels found. Inspect Y and dataset files.")

        self.X = torch.from_numpy(X_norm.astype(np.float32))
        self.delta_norm = torch.from_numpy(delta_norm.astype(np.float32))
        self.last_position = torch.from_numpy(self.last_position.astype(np.float32))
        self.Y_absolute = torch.from_numpy(Y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return self.X[index], self.delta_norm[index]


def calculate_metrics(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    distance = np.linalg.norm(predicted - truth, axis=2)
    return float(distance.mean()), float(distance[:, -1].mean())


def main() -> None:
    set_seed(int(CFG["random_seed"]))
    print("=" * 72)
    print(CFG["project_name"])
    print("Project root:", ROOT)
    print("Dataset folder:", DATA_DIR)
    print("SUMO/model folder:", SUMO_DIR)
    print("Device:", DEVICE)
    print("=" * 72)

    # Input scalers are fitted on the training split only, using the existing SUMO utility.
    x_mean, x_std, _, _ = compute_scalers(str(DATA_DIR), TRAIN_SEEDS)
    x_mean = np.asarray(x_mean, dtype=np.float32)
    x_std = np.asarray(x_std, dtype=np.float32)
    x_std = np.where(np.isfinite(x_std) & (x_std > 1e-6), x_std, 1.0).astype(np.float32)
    if not np.isfinite(x_mean).all():
        raise ValueError("Input scaler mean contains non-finite values. Check training data.")

    X_train, Y_train = load_arrays(TRAIN_SEEDS)
    X_val, Y_val = load_arrays(VAL_SEEDS)
    X_test, Y_test = load_arrays(TEST_SEEDS)
    print(f"Samples: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")

    train_ds = RelativeDisplacementDataset(X_train, Y_train, x_mean, x_std)
    delta_mean, delta_std = train_ds.delta_mean, train_ds.delta_std
    val_ds = RelativeDisplacementDataset(X_val, Y_val, x_mean, x_std, delta_mean, delta_std)
    test_ds = RelativeDisplacementDataset(X_test, Y_test, x_mean, x_std, delta_mean, delta_std)

    generator = torch.Generator().manual_seed(int(CFG["random_seed"]))
    train_loader = DataLoader(train_ds, batch_size=int(CFG["batch_size"]), shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=int(CFG["batch_size"]), shuffle=False)

    model = LSTMPredictor(
        input_dim=int(CFG["input_dim"]), hidden_dim=int(CFG["hidden_dim"]),
        num_layers=int(CFG["num_layers"]), output_steps=int(CFG["output_steps"]),
        output_dim=int(CFG["output_dim"]),
        scaler_X_mean=torch.from_numpy(x_mean), scaler_X_std=torch.from_numpy(x_std),
        scaler_Y_mean=torch.from_numpy(delta_mean), scaler_Y_std=torch.from_numpy(delta_std),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(CFG["learning_rate"]))
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    checkpoint_path = CHECKPOINT_DIR / CFG["checkpoint_filename"]
    history_path = RESULTS_DIR / CFG["epoch_history_filename"]
    history = []

    for epoch in range(1, int(CFG["epochs"]) + 1):
        model.train()
        train_losses = []
        for xb, db in train_loader:
            xb, db = xb.to(DEVICE), db.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), db)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, db in val_loader:
                xb, db = xb.to(DEVICE), db.to(DEVICE)
                val_losses.append(float(loss_fn(model(xb), db).item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        print(f"Epoch {epoch:02d}/{CFG['epochs']} | train MSE={train_loss:.6f} | val MSE={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    pd.DataFrame(history).to_csv(history_path, index=False)
    print("Loading best validation checkpoint for held-out test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    predictions, truths = [], []
    with torch.no_grad():
        for start in range(0, len(test_ds), 512):
            end = min(start + 512, len(test_ds))
            xb = test_ds.X[start:end].to(DEVICE)
            predicted_delta_norm = model(xb).cpu().numpy()
            predicted_delta = predicted_delta_norm * delta_std + delta_mean
            last_pos = test_ds.last_position[start:end].numpy()
            predicted_absolute = last_pos[:, None, :] + np.cumsum(predicted_delta, axis=1)
            predictions.append(predicted_absolute)
            truths.append(test_ds.Y_absolute[start:end].numpy())

    predicted = np.concatenate(predictions, axis=0)
    truth = np.concatenate(truths, axis=0)
    ade, fde = calculate_metrics(predicted, truth)
    metrics = {
        "experiment": "Objective 1 - Relative-displacement LSTM",
        "input_key": CFG["input_key"],
        "train_seeds": f"{CFG['train_seed_start']}-{CFG['train_seed_end']}",
        "validation_seeds": f"{CFG['validation_seed_start']}-{CFG['validation_seed_end']}",
        "test_seeds": f"{CFG['test_seed_start']}-{CFG['test_seed_end']}",
        "scenarios": ";".join(SCENARIOS),
        "architecture": f"{CFG['num_layers']}-layer LSTM, hidden_dim={CFG['hidden_dim']}",
        "epochs_configured": int(CFG["epochs"]),
        "best_validation_mse": best_val,
        "ADE_m": ade,
        "FDE_m": fde,
        "test_samples": len(test_ds),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }
    metrics_path = RESULTS_DIR / CFG["results_filename"]
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    with (LOGS_DIR / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nFINAL OBJECTIVE 1 RESULTS")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\nMetrics CSV:", metrics_path)
    print("Training history:", history_path)
    print("Checkpoint:", checkpoint_path)
    print("Note: run a comparable baseline on the same test samples before claiming improvement over baseline.")


if __name__ == "__main__":
    main()
