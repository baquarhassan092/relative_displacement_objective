"""Objective 4-Relative (scratch-trained): Column-6 bias for the relative model.
This script reproduces the workflow of objective4_relative/ (retrain.py +
evaluation_masked.py + evaluation_original.py + the bias table) for the
RELATIVE-DISPLACEMENT predictor of Objective 1.
Same as objective4/
  1. The column-6 absence rate is MEASURED from the stored masks of the 20%
     penetration training seeds 1-14 (not guessed).
  2. Each follower_speed entry of clean X_gt is hidden independently with that
     probability (train mask seed 42, validation mask seed 43); columns 0-5 and
     the targets are untouched. Masked values become physical 0.0 BEFORE
     normalisation.
  3. A FRESH, randomly initialised LSTMPredictor (2 layers, hidden 128) is
     trained (Adam, lr 1e-3, batch 128, 30 epochs, MSE, best-validation
     checkpoint). Objective 1 is used ONLY for its normalisation scalers.
  4. Original (always-present-trained) and masked-trained models are scored on
     the SAME 20% penetration test windows (seeds 17-20) under four input
     conditions, re-applying the actual stored column-6 mask after imputation.
  5. Delta = original - masked-trained, for ADE and FDE, plus Table-3-style
     ADE-vs-horizon files.
Different from objective4/ (because the model is relative)
  * Targets are step-wise displacements (dx, dy); predictions are turned back
    into positions by a cumulative sum from an ANCHOR position.
Anchor fix (the point of this version)
  * The earlier script used the clean X_gt last position as the anchor in every
    condition. That is information the deployed model would not have (the last
    ego position is missing in ~10% of test windows). Here the PRIMARY results
    anchor on the last OBSERVED ego position of each condition's own input
    (no ground truth). The clean-anchor numbers are still written, clearly
    labelled "oracle", so the effect of the anchor stays visible.
Usage (run from anywhere inside the project tree that contains
objective1/config.json, data_corrupted/ and sumo/):
    python objective4_relative/objective4_relative_scratch.py
    python objective4_relative/objective4_relative_scratch.py --skip-train
(--skip-train evaluates the checkpoint saved by an earlier run.)
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
 # --------------------------------------------------------------------------
 # Paths and configuration
 # --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = next(
    (c for c in (HERE, *HERE.parents) if (c / "objective1" / "config.json").is_file()),
    None,
)
if ROOT is None:
    raise FileNotFoundError(
        "Could not locate objective1/config.json in "
        f"{HERE} or any parent folder."
    )
CFG = json.loads((ROOT / "objective1" / "config.json").read_text(encoding="utf-8"))
DATA_DIR = ROOT / CFG.get("data_folder_name", "data_corrupted")
SUMO_DIR = ROOT / CFG.get("sumo_folder_name", "sumo")
BASE_CKPT = ROOT / "objective1" / "checkpoints" / CFG.get(
    "checkpoint_filename", "relative_displacement_lstm.pt"
)
RESULTS_DIR = HERE / "results"
CHECKPOINT_DIR = HERE / "checkpoints"
LOGS_DIR = HERE / "logs"
for folder in (RESULTS_DIR, CHECKPOINT_DIR, LOGS_DIR):
    folder.mkdir(parents=True, exist_ok=True)
OUT_CKPT = CHECKPOINT_DIR / "relative_displacement_col6_scratch.pt"
for required in ("lstm_predictor.py", "baseline.py"):
    if not (SUMO_DIR / required).is_file():
        raise FileNotFoundError(f"Missing {SUMO_DIR / required}")
sys.path.insert(0, str(SUMO_DIR))
from lstm_predictor import LSTMPredictor   # noqa: E402
import baseline as baselines   # noqa: E402
SCENARIOS = CFG["scenarios"]
TRAIN_SEEDS = range(int(CFG["train_seed_start"]), int(CFG["train_seed_end"]) + 1)
VAL_SEEDS = range(int(CFG["validation_seed_start"]), int(CFG["validation_seed_end"]) + 1)
TEST_SEEDS = range(int(CFG["test_seed_start"]), int(CFG["test_seed_end"]) + 1)
HISTORY_STEPS = int(CFG["history_steps"])
OUTPUT_STEPS = int(CFG["output_steps"])
INPUT_DIM = int(CFG["input_dim"])
OUTPUT_DIM = int(CFG["output_dim"])
HIDDEN_DIM = int(CFG["hidden_dim"])
NUM_LAYERS = int(CFG["num_layers"])
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
TRAIN_MASK_SEED = 42
VAL_MASK_SEED = 43
FEATURE_COL = 6           # follower_speed
EGO_POS = slice(2, 4)     # ego_x, ego_y
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 # Condition labels are the same as in objective4/ so the tables line up.
COND_ZERO = "Baseline 1: Zero-fill"
COND_SPLINE = "Baseline 5: Spline Interpolation"
COND_CH1 = "Condition B: Channel 1 Only"
COND_DUAL = "Condition C: Dual-Channel (Proposed)"
 # --------------------------------------------------------------------------
 # Utilities
 # --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:   # older PyTorch without weights_only
        return torch.load(path, map_location="cpu")
def load_npz(scenario: _str_, seed: _int_, keys: tuple[_str_, ...]) -> _dict_:
    path = DATA_DIR / scenario / f"run_seed_{seed}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = [k for k in keys if k not in archive]
        if missing:
            raise KeyError(f"{path} is missing arrays: {missing}")
        return {k: np.asarray(archive[k]).copy() for k in keys}
 # --------------------------------------------------------------------------
 # Models and scalers
 # --------------------------------------------------------------------------
def load_objective1_scalers() -> _dict_:
    """Read ONLY the normalisation buffers from the Objective 1 checkpoint."""
    if not BASE_CKPT.is_file():
        raise FileNotFoundError(f"Objective 1 checkpoint not found: {BASE_CKPT}")
    state = torch_load(BASE_CKPT)
    names = ("scaler_X_mean", "scaler_X_std", "scaler_Y_mean", "scaler_Y_std")
    missing = [n for n in names if n not in state]
    if missing:
        raise KeyError(f"Objective 1 checkpoint lacks scaler buffers: {missing}")
    scalers = {n: state[n].detach().cpu().numpy().astype(np.float32).copy() for n in names}
    for name in ("scaler_X_std", "scaler_Y_std"):
        s = scalers[name]
        scalers[name] = np.where(np.isfinite(s) & (s > 1e-8), s, 1.0).astype(np.float32)
    return scalers
def make_model(scalers: dict) -> _LSTMPredictor_:
    """New model with the project architecture; scalers are buffers only."""
    return LSTMPredictor(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_steps=OUTPUT_STEPS,
        output_dim=OUTPUT_DIM,
        scaler_X_mean=torch.tensor(scalers["scaler_X_mean"], dtype=torch.float32),
        scaler_X_std=torch.tensor(scalers["scaler_X_std"], dtype=torch.float32),
        scaler_Y_mean=torch.tensor(scalers["scaler_Y_mean"], dtype=torch.float32),
        scaler_Y_std=torch.tensor(scalers["scaler_Y_std"], dtype=torch.float32),
    )
 # --------------------------------------------------------------------------
 # Masking (same method as objective4/objective4_data.py)
 # --------------------------------------------------------------------------
def load_mask_bank() -> np.ndarray:
    """Column-6 absence patterns of the 20% penetration TRAINING seeds."""
    masks = []
    for seed in TRAIN_SEEDS:
        mask = load_npz("penetration_20", seed, ("X_mask",))["X_mask"]
        if mask.ndim != 3 or mask.shape[2] <= FEATURE_COL:
            raise ValueError(f"Unexpected X_mask shape (seed {seed}): {mask.shape}")
        col6 = mask[:, :, FEATURE_COL]
        if not np.isin(col6, (0, 1)).all():
            raise ValueError(f"X_mask column 6 is not binary (seed {seed})")
        masks.append(col6 == 0)   # True = follower speed missing
    return np.concatenate(masks, axis=0)
def apply_col6_masks(x_gt: np.ndarray, absence_rate: _float_, rng: np.random.Generator):
    """Hide each follower-speed entry independently; all other columns kept."""
    x_masked = x_gt.astype(np.float32, copy=True)
    missing = rng.random(x_masked.shape[:2]) < absence_rate
    x_masked[:, :, FEATURE_COL] = np.where(missing, np.nan, x_masked[:, :, FEATURE_COL])
    return x_masked, missing
def zero_fill_and_normalise(x: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    """Physical zero-fill BEFORE normalisation (as in Table 3 / objective4)."""
    if np.isinf(x).any():
        raise ValueError("Infinite value in model input")
    filled = np.where(np.isnan(x), 0.0, x)
    x_norm = (filled - x_mean.reshape(1, 1, -1)) / x_std.reshape(1, 1, -1)
    if not np.isfinite(x_norm).all():
        raise ValueError("Non-finite value after input normalisation")
    return x_norm.astype(np.float32)
def relative_displacement_targets(x_gt: np.ndarray, y_abs: np.ndarray) -> np.ndarray:
    """Objective 1 labels: step-wise (dx, dy) from the last clean position.
    Using clean ground truth is correct for TRAINING LABELS. It must not be
    used as the anchor when scoring corrupted inputs (see last_observed_position).
    """
    anchor = x_gt[:, -1, EGO_POS].astype(np.float32)
    previous = np.concatenate([anchor[:, None, :], y_abs[:, :-1, :]], axis=1)
    return (y_abs - previous).astype(np.float32)
def build_split(seeds, absence_rate, scalers, mask_seed):
    """Masked, normalised inputs and normalised displacement targets."""
    x_mean, x_std = scalers["scaler_X_mean"], scalers["scaler_X_std"]
    d_mean, d_std = scalers["scaler_Y_mean"], scalers["scaler_Y_std"]
    rng = np.random.default_rng(mask_seed)
    inputs, targets, stats = [], [], []
    for scenario in SCENARIOS:
        for seed in seeds:
            arrays = load_npz(scenario, seed, ("X_gt", "Y"))
            x_gt = arrays["X_gt"].astype(np.float32)
            y_abs = arrays["Y"].astype(np.float32)
            if x_gt.shape[1:] != (HISTORY_STEPS, INPUT_DIM):
                raise ValueError(f"Unexpected X_gt shape {x_gt.shape} ({scenario}, seed {seed})")
            if y_abs.shape != (len(x_gt), OUTPUT_STEPS, OUTPUT_DIM):
                raise ValueError(f"Unexpected Y shape {y_abs.shape} ({scenario}, seed {seed})")
            if not (np.isfinite(y_abs).all() and np.isfinite(x_gt[:, -1, EGO_POS]).all()):
                raise ValueError(f"Non-finite label/anchor ({scenario}, seed {seed})")
            x_masked, missing = apply_col6_masks(x_gt, absence_rate, rng)
             # Columns 0-5 must be unchanged by the masking.
            if not np.array_equal(x_masked[:, :, :FEATURE_COL], x_gt[:, :, :FEATURE_COL], equal_nan=True):
                raise AssertionError("Masking changed columns 0-5")
            x_norm = zero_fill_and_normalise(x_masked, x_mean, x_std)
            delta = relative_displacement_targets(x_gt, y_abs)
            delta_norm = (delta - d_mean.reshape(1, 1, -1)) / d_std.reshape(1, 1, -1)
            if not np.isfinite(delta_norm).all():
                raise ValueError(f"Non-finite displacement target ({scenario}, seed {seed})")
            inputs.append(x_norm)
            targets.append(delta_norm.astype(np.float32))
            stats.append({
                "split": "train" if mask_seed == TRAIN_MASK_SEED else "validation",
                "scenario": scenario,
                "seed": seed,
                "windows": len(x_gt),
                "target_absence_probability": absence_rate,
                "realized_absence_rate": float(missing.mean()),
                "masked_entries": int(missing.sum()),
                "total_col6_entries": int(missing.size),
            })
    X = torch.from_numpy(np.concatenate(inputs, axis=0))
    D = torch.from_numpy(np.concatenate(targets, axis=0))
    realised = sum(s["masked_entries"] for s in stats) / sum(s["total_col6_entries"] for s in stats)
    print(f"  Windows: {len(X):,} | realised masking rate: {100 * realised:.6f}%")
    return TensorDataset(X, D), stats
 # --------------------------------------------------------------------------
 # Training (same loop logic as sumo/lstm_predictor.train_lstm, plus a history)
 # --------------------------------------------------------------------------
def train_scratch_model(model, train_loader, val_loader):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_val, best_epoch, history = float("inf"), 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item()
        avg_val = val_loss / len(val_loader)
        improved = avg_val < best_val
        if improved:
            best_val, best_epoch = avg_val, epoch
            torch.save(model.state_dict(), OUT_CKPT)
        history.append({
            "epoch": epoch,
            "train_mse": avg_train,
            "validation_mse": avg_val,
            "best_validation_mse": best_val,
            "checkpoint_saved": improved,
        })
        print(f"Epoch {epoch:02d}/{EPOCHS} | train MSE {avg_train:.6f} | "
              f"val MSE {avg_val:.6f}" + (" | saved" if improved else ""))
    model.load_state_dict(torch_load(OUT_CKPT), strict=True)
    model.eval()
    return model, pd.DataFrame(history), best_val, best_epoch
 # --------------------------------------------------------------------------
 # Evaluation
 # --------------------------------------------------------------------------
def load_test_data_20pct() -> _dict_:
    keys = ("X_gt", "Y", "X_mask", "X_corrupted", "X_ch1_imputed", "X_fully_imputed")
    parts = {k: [] for k in keys}
    for seed in TEST_SEEDS:
        arrays = load_npz("penetration_20", seed, keys)
        for k in keys:
            parts[k].append(arrays[k].astype(np.float32))
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}
def last_observed_position(x: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Last ego (x, y) that is actually present in x (no ground truth).
    x: (N, T, 7) with NaN where the value is missing. If the final step is
    missing, the most recent earlier observed step is used. Windows with no
    observed ego position at all use `fallback` (training-mean position).
    """
    pos = x[:, :, EGO_POS]
    valid = ~np.isnan(pos).any(axis=2)                        # (N, T)
    steps = pos.shape[1]
    last_idx = steps - 1 - np.argmax(valid[:, ::-1], axis=1)   # index of last valid step
    anchor = pos[np.arange(len(x)), last_idx].astype(np.float32)
    none_observed = ~valid.any(axis=1)
    if none_observed.any():
        anchor[none_observed] = fallback
    return anchor
def reconstruct_absolute(delta: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    return anchor[:, None, :] + np.cumsum(delta, axis=1)
def predict_displacements(model, conditions, missing_col6, x_mean, x_std) -> _dict_:
    """Physical-unit displacement predictions for each condition."""
    model.eval()
    outputs = {}
    with torch.no_grad():
        for name, spec in conditions.items():
            x = np.asarray(spec["model_input"], dtype=np.float32).copy()
             # Keep the real column-6 absence after imputation (as objective4/).
            x[:, :, FEATURE_COL] = np.where(missing_col6, np.nan, x[:, :, FEATURE_COL])
            x_norm = zero_fill_and_normalise(x, x_mean, x_std)
            batches = []
            for start in range(0, len(x_norm), 512):
                xb = torch.from_numpy(x_norm[start:start + 512]).to(DEVICE)
                batches.append(model.denormalise_y(model(xb)).cpu().numpy())
            outputs[name] = np.concatenate(batches, axis=0)
    return outputs
def score(pred_deltas: _dict_, anchors: _dict_, y_true: np.ndarray):
    """ADE/FDE and ADE-at-t per condition for a given anchor set."""
    summary, horizon = [], []
    for name, delta in pred_deltas.items():
        pred_abs = reconstruct_absolute(delta, anchors[name])
        errors = np.linalg.norm(pred_abs - y_true, axis=2)     # (N, 50)
        if not np.isfinite(errors).all():
            raise ValueError(f"Non-finite errors: {name}")
        at_t = errors.mean(axis=0)
        summary.append({"Condition": name, "ADE (m)": float(at_t.mean()), "FDE (m)": float(at_t[-1])})
        for step, err in enumerate(at_t, start=1):
            horizon.append({
                "Condition": name,
                "Timestep": step,
                "Time (s)": round(step * 0.1, 1),
                "ADE_at_t (m)": round(float(err), 3),
            })
    return pd.DataFrame(summary), pd.DataFrame(horizon)
def bias_table(original: pd.DataFrame, masked: pd.DataFrame) -> pd.DataFrame:
    """Delta = original - masked-trained (positive = masked training is better)."""
    merged = original.merge(masked, on="Condition", suffixes=("_orig", "_masked"))
    return pd.DataFrame({
        "Condition": merged["Condition"],
        "Original_ADE_m": merged["ADE (m)_orig"],
        "Masked_ADE_m": merged["ADE (m)_masked"],
        "Delta_ADE_m": merged["ADE (m)_orig"] - merged["ADE (m)_masked"],
        "Original_FDE_m": merged["FDE (m)_orig"],
        "Masked_FDE_m": merged["FDE (m)_masked"],
        "Delta_FDE_m": merged["FDE (m)_orig"] - merged["FDE (m)_masked"],
    })
def anchor_diagnostics(conditions, observed, gt_anchor) -> pd.DataFrame:
    rows = []
    for name, spec in conditions.items():
        final_missing = float(np.isnan(spec["anchor_source"][:, -1, EGO_POS]).any(axis=1).mean())
        err = np.linalg.norm(observed[name] - gt_anchor, axis=1)
        rows.append({
            "Condition": name,
            "final_step_ego_position_missing_fraction": final_missing,
            "observed_anchor_mean_error_m": float(err.mean()),
            "observed_anchor_median_error_m": float(np.median(err)),
            "observed_anchor_max_error_m": float(err.max()),
        })
    return pd.DataFrame(rows)
 # --------------------------------------------------------------------------
 # Main
 # --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-train", action="store_true",
                        help="evaluate the existing scratch checkpoint instead of retraining")
    args = parser.parse_args()
    set_seed(RANDOM_SEED)
    print("=" * 76)
    print("OBJECTIVE 4-RELATIVE (SCRATCH) - COLUMN-6 BIAS, RELATIVE-DISPLACEMENT MODEL")
    print("Project root :", ROOT)
    print("Device       :", DEVICE)
    print("Scalers from :", BASE_CKPT)
    print("=" * 76)
    scalers = load_objective1_scalers()
    x_mean, x_std = scalers["scaler_X_mean"], scalers["scaler_X_std"]
    mask_bank = load_mask_bank()
    absence_rate = float(mask_bank.mean())
    print(f"20% training mask bank {mask_bank.shape}; "
          f"measured column-6 absence = {100 * absence_rate:.6f}%")
    best_val = best_epoch = None
    if args.skip_train:
        if not OUT_CKPT.is_file():
            raise FileNotFoundError(f"--skip-train needs an existing checkpoint: {OUT_CKPT}")
        masked_model = make_model(scalers)
        masked_model.load_state_dict(torch_load(OUT_CKPT), strict=True)
        masked_model = masked_model.to(DEVICE)
        print("Loaded existing scratch checkpoint:", OUT_CKPT)
    else:
        print("\nPreparing training data (seeds 1-14, mask seed 42)")
        train_ds, train_stats = build_split(TRAIN_SEEDS, absence_rate, scalers, TRAIN_MASK_SEED)
        print("Preparing validation data (seeds 15-16, mask seed 43)")
        val_ds, val_stats = build_split(VAL_SEEDS, absence_rate, scalers, VAL_MASK_SEED)
        generator = torch.Generator().manual_seed(RANDOM_SEED)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  generator=generator, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        set_seed(RANDOM_SEED)   # reproducible random weights
        masked_model = make_model(scalers)
        print("\nTraining a NEW randomly initialised model (no Objective 1 weights)")
        masked_model, history, best_val, best_epoch = train_scratch_model(
            masked_model, train_loader, val_loader)
        history.to_csv(RESULTS_DIR / "training_history_scratch.csv", index=False)
        pd.DataFrame(train_stats + val_stats).to_csv(
            RESULTS_DIR / "masking_rates_by_scenario_seed_scratch.csv", index=False)
     # Original (always-present-trained) relative model = Objective 1 checkpoint.
    original_model = make_model(scalers)
    original_model.load_state_dict(torch_load(BASE_CKPT), strict=True)
    original_model = original_model.to(DEVICE)
    print("\nLoading 20% penetration test data (seeds 17-20)")
    test = load_test_data_20pct()
    y_true = test["Y"]
    missing_col6 = test["X_mask"][:, :, FEATURE_COL] == 0
    print(f"Test windows: {len(y_true):,} | column-6 absence: {100 * missing_col6.mean():.6f}%")
     # model_input = what the network sees; anchor_source = what the anchor may
     # be read from (observed data only, NaN where missing; never X_gt).
    corrupted = test["X_corrupted"]
    spline = baselines.spline_interp(corrupted.copy())
    conditions = {
        COND_ZERO: {"model_input": baselines.zero_fill(corrupted.copy()),
                    "anchor_source": corrupted},
        COND_SPLINE: {"model_input": spline, "anchor_source": spline},
        COND_CH1: {"model_input": test["X_ch1_imputed"],
                   "anchor_source": test["X_ch1_imputed"]},
        COND_DUAL: {"model_input": test["X_fully_imputed"],
                    "anchor_source": test["X_fully_imputed"]},
    }
    fallback = x_mean[EGO_POS].astype(np.float32)
    observed_anchor = {n: last_observed_position(s["anchor_source"], fallback)
                       for n, s in conditions.items()}
    gt_anchor = test["X_gt"][:, -1, EGO_POS].astype(np.float32)    # ORACLE, reference only
    oracle_anchor = {n: gt_anchor for n in conditions}
    diag = anchor_diagnostics(conditions, observed_anchor, gt_anchor)
    diag.to_csv(RESULTS_DIR / "anchor_diagnostics_scratch.csv", index=False)
    print("\nAnchor diagnostics (observed anchor versus clean position):")
    print(diag.to_string(index=False))
    print("\nPredicting with the original Objective 1 model...")
    pred_original = predict_displacements(original_model, conditions, missing_col6, x_mean, x_std)
    print("Predicting with the scratch-trained masked model...")
    pred_masked = predict_displacements(masked_model, conditions, missing_col6, x_mean, x_std)
     # PRIMARY results: observed (deployable) anchor.
    orig_sum, orig_hor = score(pred_original, observed_anchor, y_true)
    mask_sum, mask_hor = score(pred_masked, observed_anchor, y_true)
    bias = bias_table(orig_sum, mask_sum)
    orig_hor.to_csv(RESULTS_DIR / "original_matched_table3_scratch.csv", index=False)
    orig_sum.to_csv(RESULTS_DIR / "original_matched_summary_scratch.csv", index=False)
    mask_hor.to_csv(RESULTS_DIR / "objective4_scratch_table3.csv", index=False)
    mask_sum.to_csv(RESULTS_DIR / "objective4_scratch_summary.csv", index=False)
    bias.to_csv(RESULTS_DIR / "objective4_relative_bias_scratch.csv", index=False)
     # REFERENCE results: oracle (clean ground-truth) anchor, not deployable.
    o_orig_sum, _ = score(pred_original, oracle_anchor, y_true)
    o_mask_sum, _ = score(pred_masked, oracle_anchor, y_true)
    bias_oracle = bias_table(o_orig_sum, o_mask_sum)
    bias_oracle.to_csv(RESULTS_DIR / "objective4_relative_bias_scratch_ORACLE_anchor.csv", index=False)
    print("\n=== BIAS (observed anchor, PRIMARY): Delta = original - masked-trained ===")
    print(bias.to_string(index=False))
    print("\n=== Same models with the clean ORACLE anchor (reference only) ===")
    print(bias_oracle.to_string(index=False))
    metadata = {
        "method": "Scratch-trained relative-displacement LSTM, realistic column-6 masking",
        "scalers_from_checkpoint": str(BASE_CKPT),
        "training_initialization": "random; Objective 1 weights not loaded",
        "mask_source": "penetration_20, training seeds 1-14",
        "measured_absence_rate": absence_rate,
        "masking": "independent Bernoulli per window/timestep, same p in all scenarios",
        "training_mask_seed": TRAIN_MASK_SEED,
        "validation_mask_seed": VAL_MASK_SEED,
        "missing_value_handling": "physical zero-fill before normalisation",
        "primary_anchor": "last observed ego position of each condition's own input",
        "reference_anchor": "clean X_gt last position (oracle, not deployable)",
        "test": "penetration_20, seeds 17-20, "
                f"{len(y_true)} windows, column-6 absence {float(missing_col6.mean()):.6f}",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_mse": best_val,
        "device": str(DEVICE),
    }
    (LOGS_DIR / "run_summary_scratch.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nSaved results to:", RESULTS_DIR)
    print("Saved checkpoint:", OUT_CKPT)
if __name__ == "__main__":
    main()
