# Objective 4-Relative — Trajectory Prediction Under Column-6 Missingness

## 1. Overview

Objective 4-Relative measures how training with realistic missingness in input
Column 6 (`follower_speed`) affects the Objective 1 relative-displacement
trajectory predictor.

The experiment compares:

- **Original model:** the pretrained Objective 1 model, evaluated without
  additional training.
- **Masked model:** a new, randomly initialized model trained with Column 6
  artificially masked at the measured absence rate.

The Objective 1 checkpoint supplies normalization buffers and the original
baseline model. Its model weights are **not** used to initialize the masked
model. This is scratch training, not fine-tuning.

Both models are evaluated on the same held-out 20% penetration test windows
under four input conditions. The primary evaluation uses an anchor derived
from each condition's own observed input; an additional oracle-anchor result
is provided for reference only.

## 2. Project structure

```text
project_root/
├── data_corrupted/
│   ├── penetration_10/
│   ├── penetration_20/
│   └── penetration_30/
├── objective1/
│   ├── config.json
│   └── checkpoints/
│       └── relative_displacement_lstm.pt
├── objective4_relative/
│   ├── README.md
│   ├── config.json
│   ├── requirements.txt
│   ├── objective4_relative.py
│   ├── checkpoints/
│   ├── logs/
│   └── results/
└── sumo/
    ├── baseline.py
    └── lstm_predictor.py
```

The script searches its parent directories for `objective1/config.json` to
locate the project root. It reads data and shared model utilities from the
project's `data_corrupted/` and `sumo/` directories.

## 3. Data and representation

Each input window contains 30 history steps and seven features:

| Index | Feature |
|---:|---|
| 0 | `ego_speed` |
| 1 | `ego_accel` |
| 2 | `ego_x` |
| 3 | `ego_y` |
| 4 | `gap_to_leader` |
| 5 | `leader_speed` |
| 6 | `follower_speed` |

The input shape is `(samples, 30, 7)`. The model predicts 50 future
displacement steps `(dx, dy)`, with target shape `(samples, 50, 2)`. Predicted
displacements are cumulatively summed from an anchor to reconstruct future
positions.

Dataset files are expected at
`data_corrupted/<scenario>/run_seed_<seed>.npz`.

| Split | Seeds | Scenarios | Use |
|---|---:|---|---|
| Training | 1–14 | 10%, 20%, 30% penetration | Train the masked model |
| Validation | 15–16 | 10%, 20%, 30% penetration | Select the best checkpoint |
| Test | 17–20 | 20% penetration | Compare the two models |

Training and validation load `X_gt` and `Y`. Test evaluation also requires
`X_mask`, `X_corrupted`, `X_ch1_imputed`, and `X_fully_imputed`. The script
expects these arrays to be present in each relevant archive.

## 4. Missingness and training procedure

### Measuring and applying missingness

The Column-6 absence probability is measured from `X_mask` for
`penetration_20`, training seeds 1–14. The measured value depends on the
dataset files. In the run recorded in `logs/run_summary_scratch.json`, it was
approximately 99.002769%.

That single probability is applied to training data from all three penetration
scenarios using independent Bernoulli draws for each sample and history step.
The training-mask seed is 42; the validation-mask seed is 43. Only feature 6
is masked; features 0–5 and the training targets are unchanged.

Masked values are set to physical `0.0` before normalization. As a result,
the normalized value presented to the model is generally not numeric zero.
This procedure does not add a separate missingness-indicator feature.

### Model training

The masked model uses the project's `LSTMPredictor` architecture with input
dimension 7, hidden size 128, two layers, and output dimensions `(50, 2)`.
It is initialized with random weights and trained using:

| Setting | Value |
|---|---:|
| Epochs | Up to 30 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Mean Squared Error |
| Random seed | 42 |

The checkpoint with the lowest validation loss is saved and used for
evaluation. Input and target normalization buffers come from the Objective 1
checkpoint.

## 5. Evaluation methodology

### Test data and input conditions

The original and masked models are evaluated on the same `penetration_20`
windows from seeds 17–20. The script re-applies the actual test Column-6
missingness mask to each condition's model input before normalization.

| Condition | Input |
|---|---|
| Baseline 1: Zero-fill | `X_corrupted`, with missing values zero-filled |
| Baseline 5: Spline Interpolation | Spline-interpolated input |
| Condition B: Channel 1 Only | `X_ch1_imputed` |
| Condition C: Dual-Channel (Proposed) | `X_fully_imputed` |

Because native missingness is re-applied after imputation, imputed values at
Column-6 positions marked missing in the test mask are discarded. Therefore,
these comparisons do not measure recovery of `follower_speed` at those
positions; other differences between the condition inputs may still affect
the results.

### Anchors

The primary results use the last observed ego position in each condition's
own input. If no ego position is observed in a window, the training-mean ego
position is used as a fallback. This avoids using the clean test trajectory
as the anchor for the primary scores.

The script also writes reference results using the final clean `X_gt` position
as an **oracle anchor**. These oracle results are not deployable and should
not be confused with the primary results.

### Metrics and bias

For each test sample, Euclidean position errors are calculated over the
reconstructed trajectory:

- **ADE:** mean error across all 50 future steps and all test windows.
- **FDE:** mean error at the final future step across test windows.
- **Per-horizon ADE:** mean error at each individual future step.

The bias tables use:

```text
Delta = original-model metric - masked-model metric
```

A positive delta means the masked-trained model has lower error for that
metric and condition. A negative delta means the original model has lower
error.

## 6. Running the experiment

Run these commands from the project root in PowerShell.

Install dependencies if they are not already available:

```powershell
python -m pip install -r objective4_relative\requirements.txt
```

Train the masked model and evaluate both models:

```powershell
python objective4_relative\objective4_relative.py
```

Skip training and evaluate an existing masked-model checkpoint:

```powershell
python objective4_relative\objective4_relative.py --skip-train
```

The `--skip-train` option requires
`objective4_relative/checkpoints/relative_displacement_col6_scratch.pt`.
The script creates the `checkpoints/`, `logs/`, and `results/` directories
when needed.

## 7. Generated files

The current script writes the following outputs:

| Path | Description |
|---|---|
| `checkpoints/relative_displacement_col6_scratch.pt` | Best masked-model checkpoint |
| `results/training_history_scratch.csv` | Training and validation loss by epoch |
| `results/masking_rates_by_scenario_seed_scratch.csv` | Requested and realized training/validation masking rates |
| `results/original_matched_summary_scratch.csv` | Original model ADE/FDE using primary observed anchors |
| `results/original_matched_table3_scratch.csv` | Original model per-horizon errors |
| `results/objective4_scratch_summary.csv` | Masked model ADE/FDE using primary observed anchors |
| `results/objective4_scratch_table3.csv` | Masked model per-horizon errors |
| `results/objective4_relative_bias_scratch.csv` | Primary original-minus-masked ADE/FDE differences |
| `results/objective4_relative_bias_scratch_ORACLE_anchor.csv` | Reference bias using clean oracle anchors |
| `results/anchor_diagnostics_scratch.csv` | Anchor missingness and error diagnostics |
| `logs/run_summary_scratch.json` | Run settings, measured absence rate, test size, and training summary |

The `objective4_relative/config.json` file documents settings, but the current
script does not load it. Paths, scenarios, and seed ranges are obtained from
`objective1/config.json`; model and training settings are defined in the
Python script. Editing `objective4_relative/config.json` alone will not
change the run.

## 8. Interpretation and limitations

1. The absence probability is measured from one reference scenario and
   training-seed range, then applied to training data across all three
   scenarios.
2. Artificial masks are independent across sample/timestep positions. They
   may not represent bursty or otherwise structured real-world sensor
   outages.
3. Zero-filling does not distinguish a missing speed from a true physical
   zero-speed observation.
4. Final testing is limited to 20% penetration, seeds 17–20. It does not by
   itself demonstrate generalization to 10% or 30% penetration.
5. Test-time reapplication of the native Column-6 mask means the imputation
   conditions do not retain imputed values at those masked positions.
6. Oracle-anchor results use clean test information and are reference-only.
   Use observed-anchor results for the primary comparison.
7. Results depend on the data, checkpoint, software versions, and random
   seeds. A single run does not establish statistical significance.

For reproducible comparisons, keep the dataset files, Objective 1 checkpoint,
seed splits, software versions, and evaluation preprocessing fixed. To
measure variation across artificial masks, repeat training with multiple mask
seeds and report the resulting distribution of ADE and FDE.
