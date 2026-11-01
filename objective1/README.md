# Objective 1 — Relative-Displacement Trajectory Prediction

## 1. What this objective does

**Objective 1 changes the LSTM prediction target from absolute future coordinates to step-by-step relative displacement (`Δx`, `Δy`).** The model predicts how far the vehicle moves at each future time step, then those predicted movements are accumulated to reconstruct the future trajectory. We compare this model with the existing absolute-coordinate LSTM on the same held-out test samples.

The main question is: **Does predicting step-wise movement instead of absolute future positions improve trajectory-prediction accuracy on our test data?**

This README explains the motivation, dataset, files, implementation, mathematics, evaluation, results, and a simple explanation to present to a teacher.

---

## 2. Project folder layout

The project root on the current computer is:

`C:\Users\baqua\OneDrive\Desktop\relative_traj_displacement`

The important structure is:

```text
relative_traj_displacement/
├── sumo/
│   ├── lstm_predictor.py
│   ├── lstm_baseline.pt
│   └── ... original SUMO/model files
├── data_corrupted/
│   ├── penetration_10/
│   │   └── run_seed_*.npz
│   ├── penetration_20/
│   │   └── run_seed_*.npz
│   └── penetration_30/
│       └── run_seed_*.npz
└── objective1/
    ├── config.json
    ├── run_objective1.py
    ├── compare_baseline.py
    ├── checkpoints/
    │   └── relative_displacement_lstm.pt
    ├── results/
    │   ├── summary_metrics.csv
    │   ├── training_history.csv
    │   ├── baseline_comparison.csv
    │   └── baseline_comparison_per_scenario_seed.csv
    └── logs/
        └── run_summary.json
```

Your folder may contain additional files. The list above highlights the files used in Objective 1.

### What the folders are for

| Folder | Purpose in Objective 1 |
|---|---|
| `sumo/` | Contains the existing LSTM implementation/utilities and the original absolute-coordinate baseline checkpoint. |
| `data_corrupted/` | Contains `.npz` dataset files grouped by scenario and simulation seed. Objective 1 reads `X_gt` and `Y` from these files. |
| `objective1/` | Contains the new relative-displacement experiment, its configuration, trained checkpoint, comparison script, and results. |

**Important distinction:** Although the data directory is named `data_corrupted`, the comparison reported here uses **`X_gt` as the input array**. This is a ground-truth-history input evaluation, not a test of corrupted/missing inputs. The directory contains multiple arrays; the script selects the one named in the configuration.

---

## 3. Dataset: what is loaded from `data_corrupted/`?

Each `run_seed_*.npz` file is a NumPy archive. The relevant arrays are:

| Array | Expected shape | Meaning in this experiment |
|---|---|---|
| `X_gt` | `(N, 30, 7)` | Historical input sequences: `N` samples, 30 time steps, 7 features. |
| `Y` | `(N, 50, 2)` | Ground-truth future trajectories: 50 future positions, each with X and Y coordinates. |

`N` means the number of samples in that file; it can differ by scenario and seed.

### Seven input features

The feature order used by the code is:

| Python index | Feature |
|---:|---|
| 0 | `ego_speed` |
| 1 | `ego_accel` |
| 2 | `ego_x` |
| 3 | `ego_y` |
| 4 | `gap_to_leader` |
| 5 | `leader_speed` |
| 6 | `follower_speed` |

The last observed ego position is extracted from `X_gt[:, -1, 2:4]`. Here `-1` means the final observed time step; indices `2:4` select ego X and ego Y.

### Scenarios and seed split

The scripts use these scenario folders:

- `penetration_10`
- `penetration_20`
- `penetration_30`

The configured split is:

| Split | Seeds | Why it is used |
|---|---|---|
| Training | 1–14 | Learns model weights and training-derived normalization values. |
| Validation | 15–16 | Tracks validation loss and selects the best checkpoint. |
| Test | 17–20 | Held-out evaluation and final baseline comparison. |

This split is applied across the configured scenarios. The reported comparison loaded **8,084 shared test samples** across all three scenarios and test seeds 17–20.

---

## 4. What we reuse from `sumo/`

### `sumo/lstm_predictor.py`

This is existing code. Objective 1 reuses its `LSTMPredictor` network architecture and `compute_scalers` utility. This avoids changing the network architecture while changing the target representation.

The model has:

- 7 input features per time step;
- 30 historical time steps;
- a 2-layer LSTM;
- hidden dimension 128;
- a fully connected output layer;
- 50 future steps, each with 2 output values.

The LSTM encodes the 30-step history. The final hidden representation is passed through a linear layer that produces `50 × 2 = 100` values, reshaped to `(batch, 50, 2)`.

### `sumo/lstm_baseline.pt`

This is the checkpoint for the original absolute-coordinate model. It predicts normalized absolute future coordinates, which are converted back to coordinates using the baseline checkpoint's own target scaler.

**Keep this checkpoint unchanged during Objective 1 comparison.** It is the reference model against which the relative-displacement model is evaluated.

### Why use the same architecture?

The main intended difference is the target representation:

- **Baseline:** predicts absolute future positions directly.
- **Objective 1:** predicts step-wise displacement and reconstructs positions.

Using the same architecture helps isolate the effect of changing the prediction target. It does not guarantee that every other experimental factor is identical, so the data split, preprocessing, and evaluation must also be checked.

---

## 5. What each Objective 1 file does

### `objective1/config.json`

Stores the settings used by the training script, including folder names, scenarios, array keys, train/validation/test seed ranges, model dimensions, training parameters, and output filenames. Check the actual local file when you need to quote an exact setting.

### `objective1/run_objective1.py`

This is the main implementation for training and evaluating the relative-displacement model. Its process is:

1. Reads the configuration and sets random seeds.
2. Locates `data_corrupted/` and imports the model/scaler utility from `sumo/lstm_predictor.py`.
3. Computes input normalization statistics from the training split.
4. Loads train, validation, and test arrays from the configured scenario/seed files.
5. Converts each ground-truth future trajectory `Y` into 50 step-wise displacement labels.
6. Normalizes the input and displacement labels.
7. Trains the LSTM using Mean Squared Error (MSE) loss and Adam.
8. Calculates validation loss after each epoch and saves the checkpoint whenever validation loss improves.
9. Loads the best validation checkpoint.
10. Predicts displacements for the held-out test inputs.
11. Converts normalized predicted displacements back to physical displacement units.
12. Cumulatively reconstructs the predicted absolute future positions.
13. Calculates ADE and FDE and saves metrics, training history, and a JSON summary.

### `objective1/checkpoints/relative_displacement_lstm.pt`

This is the trained Objective 1 model selected by validation loss. The state dictionary includes model weights and scaler buffers. Its output scaler represents **displacements**, not absolute positions.

Do not swap this checkpoint's output scaler with the baseline scaler: the two models predict different quantities.

### `objective1/compare_baseline.py`

This script compares the two existing checkpoints without retraining them. It:

1. Loads the same held-out `X_gt` samples and their `Y` targets.
2. Loads the baseline and relative-displacement checkpoints.
3. Normalizes inputs using the scaler values stored in each model's checkpoint.
4. Runs each model on the same raw test samples.
5. Converts baseline outputs into absolute coordinates.
6. Converts the relative model's output into physical displacements and cumulatively reconstructs absolute positions.
7. Calculates ADE/FDE for both models against the same ground-truth `Y`.
8. Saves an aggregate comparison and per-scenario/per-seed results.

It does not train either model and should not overwrite either checkpoint.

---

## 6. The key idea: absolute coordinates vs. relative displacement

Suppose the last observed position is `(10, 20)` and the vehicle's next positions are `(12, 23)` and `(13, 25)`.

The step-wise displacements are:

- First step: `(12, 23) - (10, 20) = (2, 3)`
- Second step: `(13, 25) - (12, 23) = (1, 2)`

To reconstruct positions from predicted displacements, start at `(10, 20)`:

- First reconstructed position: `(10, 20) + (2, 3) = (12, 23)`
- Second reconstructed position: `(12, 23) + (1, 2) = (13, 25)`

Absolute coordinates answer **“Where is the vehicle?”** Displacement answers **“How far did the vehicle move since the previous step?”**

### Mathematical definition

Let `P_last` be the last observed position and `P_t` the ground-truth position at future step `t`.

First displacement:

\[
\Delta P_0 = P_0 - P_{last}
\]

Later displacements:

\[
\Delta P_t = P_t - P_{t-1}, \quad t \geq 1
\]

The model predicts the displacement sequence:

\[
\widehat{D} = f_\theta(X)
\]

The predicted displacements are inverse-normalized using the relative model's displacement mean and standard deviation. Then absolute positions are reconstructed by cumulative summation:

\[
\hat{P}_t = P_{last} + \sum_{i=0}^{t}\widehat{\Delta P}_i
\]

This produces a 50-step absolute trajectory that can be evaluated against `Y`.

---

## 7. Normalization and why checkpoint scalers matter

The model uses normalized values during training. The input transformation is:

\[
X_{norm} = \frac{X - \mu_X}{\sigma_X}
\]

The relative displacement target is normalized as:

\[
D_{norm} = \frac{D - \mu_D}{\sigma_D}
\]

The input scaler is calculated from training data. The displacement target scaler is calculated from training displacement labels. The same training-derived values are used on validation and test data; test data must not be used to fit these scalers.

After prediction, the displacement output is converted back to physical units:

\[
\widehat{D} = \widehat{D}_{norm} \odot \sigma_D + \mu_D
\]

The baseline checkpoint's output scaler describes absolute coordinates, while the Objective 1 checkpoint's output scaler describes displacement. Therefore, **each checkpoint must use its own saved normalization statistics**. Using the wrong output scaler would make the reconstructed positions incorrect.

The input pipeline converts non-finite normalized input values to zero. This is a preprocessing safeguard; it should not be described as evidence that the model was evaluated under a particular corruption pattern. The selected input array for this experiment is `X_gt`.

---

## 8. Training settings and checkpoint selection

The recorded training setup is:

| Setting | Value |
|---|---|
| Architecture | 2-layer LSTM |
| Hidden dimension | 128 |
| Input shape per sample | `(30, 7)` |
| Output shape per sample | `(50, 2)` displacement values |
| Loss | MSE |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Maximum configured epochs | 30 |
| Device in the reported run | CPU |
| Checkpoint selection | Lowest validation MSE |

The reported best validation MSE was approximately **0.349134**. Training loss and validation loss for each epoch are stored in `objective1/results/training_history.csv`.

MSE is used as a training objective on normalized displacement values. ADE and FDE are different evaluation metrics calculated after reconstructing positions in physical coordinates.

---

## 9. Evaluation metrics

### Average Displacement Error (ADE)

For every sample and each of the 50 future time steps, calculate the Euclidean distance between the predicted position and the true position, then average all distances:

\[
ADE = \frac{1}{NT}\sum_{n=1}^{N}\sum_{t=1}^{T}
\|\hat{P}_{n,t}-P_{n,t}\|_2
\]

ADE measures average trajectory error across the prediction horizon.

### Final Displacement Error (FDE)

Calculate the Euclidean distance at the final predicted time step and average it over samples:

\[
FDE = \frac{1}{N}\sum_{n=1}^{N}
\|\hat{P}_{n,T}-P_{n,T}\|_2
\]

FDE measures the average error at the 50th future step.

Both are reported in meters. **Lower values indicate better prediction accuracy.**

### Percentage improvement

For a metric where lower is better:

\[
Improvement(\%) = \frac{Baseline - Relative}{Baseline}\times100
\]

A positive value means the relative model has lower error than the baseline.

---

## 10. Final baseline comparison results

The comparison script successfully loaded **8,084 shared test samples** using `X_gt`, from scenarios `penetration_10`, `penetration_20`, and `penetration_30`, with test seeds 17–20.

| Metric | Original absolute-coordinate LSTM | Relative-displacement LSTM | Reported improvement |
|---|---:|---:|---:|
| ADE | 1.112899 m | **0.381688 m** | **65.70% lower** |
| FDE | 1.599128 m | **0.933725 m** | **41.61% lower** |
| Test samples | 8,084 | 8,084 | Same samples |

The per-scenario/per-seed CSV contains 12 scenario–seed combinations (3 scenarios × 4 test seeds). In the output shared for this run, the relative-displacement model has lower ADE and FDE in **all 12 combinations**.

### Scenario-level interpretation

The reported pattern is consistent across the three scenarios; the result is not restricted to just one scenario. The overall numbers show that, on this held-out `X_gt` evaluation, predicting step-wise displacements produced lower trajectory error than predicting absolute coordinates directly.

### What this result does and does not establish

**Supported conclusion:** On the tested samples and this evaluation protocol, the relative-displacement model achieved lower ADE and FDE than the original baseline.

**Not established by this experiment:**

- It does not prove robustness to corrupted or missing input features, because the selected input was `X_gt`.
- It does not prove that the method is better on every possible dataset or configuration.
- It does not establish statistical significance by itself; repeated independent training runs or uncertainty estimates may be needed if required by the research plan.
- These aggregate clean-input results should not be compared directly with a corrupted/imputed-input table unless the input condition, scenario scope, and metric aggregation are made comparable.

---

## 11. What the result files represent

| File | Meaning | When to use it |
|---|---|---|
| `objective1/results/summary_metrics.csv` | Metrics for the relative-displacement model's own test evaluation, plus experiment metadata. | Report the model's standalone result. |
| `objective1/results/training_history.csv` | Training and validation MSE for each epoch. | Explain learning progress and inspect possible overfitting. |
| `objective1/results/baseline_comparison.csv` | Aggregate baseline vs. relative-model ADE/FDE and percentage improvements. | Main comparison table for the report. |
| `objective1/results/baseline_comparison_per_scenario_seed.csv` | ADE/FDE for both models by scenario and test seed. | Demonstrate that improvement is consistent across subsets. |
| `objective1/logs/run_summary.json` | Structured record of the Objective 1 run and its metrics. | Keep a machine-readable experiment summary. |
| `objective1/checkpoints/relative_displacement_lstm.pt` | Trained relative-displacement model and scaler buffers. | Load the Objective 1 model for evaluation. |
| `sumo/lstm_baseline.pt` | Existing absolute-coordinate baseline checkpoint. | Reference checkpoint for comparison. |

Keep these files with the project. Do not rename or overwrite the checkpoints without updating the scripts and experiment record.

---

## 12. How to run the scripts in Git Bash

Open Git Bash and move to the project root:

```bash
cd "/c/Users/baqua/OneDrive/Desktop/relative_traj_displacement"
```

### Run training and Objective 1 evaluation

Only run this when intentionally repeating training. It may update the relative checkpoint and result files, so back up the current results first if you need to preserve the exact recorded run.

```bash
python objective1/run_objective1.py
```

### Compare against the original baseline

This uses existing checkpoints and does not retrain the models:

```bash
python -m py_compile objective1/compare_baseline.py
python objective1/compare_baseline.py
```

### Print the per-scenario/per-seed results

```bash
python -c "import pandas as pd; print(pd.read_csv('objective1/results/baseline_comparison_per_scenario_seed.csv').to_string(index=False))"
```

### Open the results directory

```bash
explorer.exe objective1/results
```

---

## 13. Reproducibility checklist

Before presenting the result, confirm:

- [ ] `sumo/lstm_baseline.pt` is the intended original baseline checkpoint.
- [ ] `objective1/checkpoints/relative_displacement_lstm.pt` is the intended Objective 1 checkpoint.
- [ ] Configuration uses train seeds 1–14, validation seeds 15–16, and test seeds 17–20.
- [ ] The same available test samples are used for both models.
- [ ] Each model uses the normalization statistics stored in its own checkpoint.
- [ ] Relative predictions are inverse-normalized before cumulative reconstruction.
- [ ] ADE and FDE are computed on reconstructed absolute positions in meters.
- [ ] The comparison reports 8,084 shared test samples.
- [ ] The report clearly states that this is an `X_gt` evaluation.
- [ ] Configuration, checkpoints, CSVs, and logs are retained for reproducibility.

---

## 14. How to explain Objective 1 to a teacher

### Short explanation

“Our first objective was to reformulate vehicle trajectory prediction using relative displacement. We reused the existing two-layer LSTM architecture from the SUMO code, with hidden dimension 128. The original model predicts absolute future coordinates, whereas our new model predicts the movement in X and Y at each future step. We trained on seeds 1–14, selected the best checkpoint using validation seeds 15–16, and evaluated on held-out seeds 17–20 across three penetration scenarios. After predicting displacement, we converted the outputs back to physical units and accumulated them from the last observed position to reconstruct absolute trajectories. We compared both models on the same 8,084 `X_gt` test samples. The relative-displacement model achieved ADE of 0.381688 m and FDE of 0.933725 m, compared with 1.112899 m and 1.599128 m for the baseline, corresponding to reported reductions of 65.70% in ADE and 41.61% in FDE for this evaluation.”

### If asked, “What did you reuse from SUMO?”

“We reused `LSTMPredictor` and the scaler utility from `sumo/lstm_predictor.py`, along with the existing absolute-coordinate checkpoint `sumo/lstm_baseline.pt`. We kept the new training and comparison code separate in `objective1/`.”

### If asked, “What did you use from `data_corrupted/`?”

“We read the `.npz` files from `penetration_10`, `penetration_20`, and `penetration_30`. For Objective 1, the input was `X_gt` with 30 time steps and 7 features, and the target was `Y` with 50 future positions and 2 coordinates. Seeds 1–14 were used for training, 15–16 for validation, and 17–20 for testing.”

### If asked, “Why use displacement?”

“Displacement represents how far the vehicle moves from one time step to the next. The model predicts those changes, and we add them cumulatively to the last observed position to recover the future trajectory. In our test, this representation gave lower ADE and FDE than the absolute-coordinate baseline.”

### If asked, “How is the comparison fair?”

“Both models were evaluated on the same held-out test samples. Each model used its own stored normalization statistics, and both predictions were converted into absolute positions before ADE and FDE were calculated.”

### If asked, “Does this prove robustness to missing data?”

“No. This experiment uses `X_gt` as input. The effect of missing or corrupted features needs a separate controlled experiment, which is part of the separate column-6 missingness objective.”

---

## 15. Objective 1 status and next objective

**Objective 1's implementation, training/evaluation, and baseline comparison are complete for the current reported run.** The key outputs and per-scenario/per-seed results have been generated, and the relative model improved both metrics in all 12 evaluated scenario–seed combinations.

For final research reporting, retain the exact configuration and result files, and describe the evidence accurately as a clean-history `X_gt` evaluation. The next separate research task is to quantify the effect of column-6 missingness; the current Objective 1 result should not be presented as completing that experiment.
