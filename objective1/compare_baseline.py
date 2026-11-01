"""Compare the original absolute-coordinate LSTM with Objective 1's relative-displacement LSTM.

Run from the project root or any directory:
    python objective1/compare_baseline.py

Uses X_gt and held-out test seeds from objective1/config.json.
Does not train models or modify checkpoints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG_PATH = HERE / "config.json"


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and any(k.startswith("lstm.") for k in value):
                return value
        if any(k.startswith("lstm.") for k in checkpoint):
            return checkpoint
    raise ValueError("Checkpoint does not look like an LSTMPredictor state_dict.")


def make_model(state_dict, device):
    sumo_dir = ROOT / "sumo"
    if not (sumo_dir / "lstm_predictor.py").is_file():
        raise FileNotFoundError(f"Could not find {sumo_dir / 'lstm_predictor.py'}")
    if str(sumo_dir) not in sys.path:
        sys.path.insert(0, str(sumo_dir))
    from lstm_predictor import LSTMPredictor

    model = LSTMPredictor(
        input_dim=7, hidden_dim=128, num_layers=2,
        output_steps=50, output_dim=2,
        scaler_X_mean=torch.zeros(7), scaler_X_std=torch.ones(7),
        scaler_Y_mean=torch.zeros(2), scaler_Y_std=torch.ones(2),
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_test_data(data_dir: Path, scenarios: list[str], seed_start: int, seed_end: int):
    xs, ys, groups = [], [], []
    missing = []
    for scenario in scenarios:
        for seed in range(seed_start, seed_end + 1):
            path = data_dir / scenario / f"run_seed_{seed}.npz"
            if not path.is_file():
                missing.append(str(path))
                continue
            with np.load(path, allow_pickle=True) as data:
                if "X_gt" not in data or "Y" not in data:
                    raise KeyError(f"{path} must contain X_gt and Y.")
                x = np.asarray(data["X_gt"], dtype=np.float32)
                y = np.asarray(data["Y"], dtype=np.float32)
            if x.ndim != 3 or x.shape[1:] != (30, 7):
                raise ValueError(f"Unexpected X_gt shape in {path}: {x.shape}; expected (N,30,7).")
            if y.ndim != 3 or y.shape[1:] != (50, 2):
                raise ValueError(f"Unexpected Y shape in {path}: {y.shape}; expected (N,50,2).")
            if len(x) != len(y):
                raise ValueError(f"X_gt/Y sample count mismatch in {path}.")
            xs.append(x)
            ys.append(y)
            groups.extend([(scenario, seed)] * len(x))
    if missing:
        print(f"WARNING: {len(missing)} expected scenario/seed files were not found.")
        for path in missing[:8]:
            print("  missing:", path)
        if len(missing) > 8:
            print("  ...")
    if not xs:
        raise FileNotFoundError(f"No test data found under {data_dir}.")
    return np.concatenate(xs), np.concatenate(ys), np.asarray(groups, dtype=object)


def normalize_x(x_raw: np.ndarray, model) -> np.ndarray:
    mean = model.scaler_X_mean.detach().cpu().numpy()
    std = model.scaler_X_std.detach().cpu().numpy()
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    x_norm = (x_raw - mean) / std
    return np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def batched_forward(model, x_norm: np.ndarray, device: torch.device, batch_size: int = 512):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x_norm), batch_size):
            xb = torch.from_numpy(x_norm[start:start + batch_size]).to(device)
            outputs.append(model(xb).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def predict_absolute(model, x_raw, device):
    pred_norm = batched_forward(model, normalize_x(x_raw, model), device)
    y_mean = model.scaler_Y_mean.detach().cpu().numpy()
    y_std = model.scaler_Y_std.detach().cpu().numpy()
    return (pred_norm * y_std + y_mean).astype(np.float32)


def predict_relative(model, x_raw, device):
    delta_norm = batched_forward(model, normalize_x(x_raw, model), device)
    delta_mean = model.scaler_Y_mean.detach().cpu().numpy()
    delta_std = model.scaler_Y_std.detach().cpu().numpy()
    delta = delta_norm * delta_std + delta_mean
    # Input features 2 and 3 are ego_x and ego_y; start from the final observed position.
    last_position = x_raw[:, -1, 2:4]
    return (last_position[:, None, :] + np.cumsum(delta, axis=1)).astype(np.float32)


def metrics(pred, truth):
    distances = np.linalg.norm(pred - truth, axis=2)
    return float(distances.mean()), float(distances[:, -1].mean())


def main():
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data_dir = ROOT / config.get("data_folder_name", "data_corrupted")
    scenarios = config.get("scenarios", ["penetration_10", "penetration_20", "penetration_30"])
    seed_start = int(config.get("test_seed_start", 17))
    seed_end = int(config.get("test_seed_end", 20))
    relative_path = HERE / "checkpoints" / config.get("checkpoint_filename", "relative_displacement_lstm.pt")
    baseline_path = ROOT / "sumo" / "lstm_baseline.pt"
    output_dir = HERE / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (relative_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Project root: {ROOT}")
    print(f"Device: {device}")
    print(f"Data folder: {data_dir}")
    print(f"Test seeds: {seed_start}-{seed_end}; scenarios: {', '.join(scenarios)}")

    x_test, y_test, groups = load_test_data(data_dir, scenarios, seed_start, seed_end)
    print(f"Loaded {len(x_test):,} shared test samples.")

    relative_model = make_model(extract_state_dict(safe_torch_load(relative_path, device)), device)
    baseline_model = make_model(extract_state_dict(safe_torch_load(baseline_path, device)), device)
    relative_pred = predict_relative(relative_model, x_test, device)
    baseline_pred = predict_absolute(baseline_model, x_test, device)

    rel_ade, rel_fde = metrics(relative_pred, y_test)
    base_ade, base_fde = metrics(baseline_pred, y_test)
    ade_improvement = (base_ade - rel_ade) / base_ade * 100.0 if base_ade else float("nan")
    fde_improvement = (base_fde - rel_fde) / base_fde * 100.0 if base_fde else float("nan")
    summary = pd.DataFrame([{
        "Input": "X_gt", "Scenarios": ";".join(scenarios),
        "Test_seeds": f"{seed_start}-{seed_end}", "Test_samples": int(len(x_test)),
        "Baseline_checkpoint": str(baseline_path), "Relative_checkpoint": str(relative_path),
        "Baseline_ADE_m": base_ade, "Relative_ADE_m": rel_ade,
        "ADE_improvement_percent_positive_means_relative_better": ade_improvement,
        "Baseline_FDE_m": base_fde, "Relative_FDE_m": rel_fde,
        "FDE_improvement_percent_positive_means_relative_better": fde_improvement,
    }])
    summary_path = output_dir / "baseline_comparison.csv"
    summary.to_csv(summary_path, index=False)

    per_rows = []
    for scenario in scenarios:
        for seed in range(seed_start, seed_end + 1):
            mask = (groups[:, 0] == scenario) & (groups[:, 1].astype(int) == seed)
            if not mask.any():
                continue
            for name, pred in (("Original absolute-coordinate LSTM", baseline_pred),
                               ("Relative-displacement LSTM", relative_pred)):
                ade, fde = metrics(pred[mask], y_test[mask])
                per_rows.append({"Scenario": scenario, "Seed": seed, "Model": name,
                                 "Samples": int(mask.sum()), "ADE_m": ade, "FDE_m": fde})
    per_path = output_dir / "baseline_comparison_per_scenario_seed.csv"
    pd.DataFrame(per_rows).to_csv(per_path, index=False)

    print("\n=== OBJECTIVE 1 BASELINE COMPARISON ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {per_path}")
    print("Positive improvement percentages mean the relative-displacement model has lower error.")


if __name__ == "__main__":
    main()
