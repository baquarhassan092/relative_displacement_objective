# Objective 4 — Column-6 Bias Quantification

## Objective

Measure the ADE/FDE difference between the original LSTM trained with follower speed available and a second LSTM trained with realistic follower-speed absence. Both models are evaluated using identical test inputs and missing-value handling.

Column 6 is `follower_speed` (zero-based indexing). Both models predict absolute future positions; the prediction target and network architecture remain unchanged.

## Method

1. Measure column-6 absence from the actual masks in the **20%-penetration training data**, using simulation seeds 1–14.
2. Independently mask each follower-speed entry in clean training inputs with the measured probability: **99.002769%**. Columns 0–5 and future targets are unchanged by masking.
3. Train a fresh predictor with the original architecture and normalization scalers. Missing inputs are replaced with physical zero **before normalization**.
4. Evaluate both checkpoints on **20%-penetration test seeds 17–20** under four input conditions: zero-fill, spline interpolation, Channel 1 and dual-channel.
5. Calculate the error differences: **Delta = original − masked-trained**.

Training uses independent timestep masking, not sampled whole-window patterns. Masks remain fixed during training. Validation uses the same training-derived probability with a separate random seed. Evaluation uses actual stored test masks and reapplies column-6 absence after imputation for both models.

## Experimental settings

| Setting | Value |
|---|---|
| Input / target | 30 × 7 history / 50 × 2 future positions |
| Observation / prediction horizon | 3 seconds / 5 seconds at 10 Hz |
| Training scenarios | 10%, 20%, 30% penetration |
| Train / validation / test seeds | 1–14 / 15–16 / 17–20 |
| Training / validation windows | 27,987 / 3,932 |
| Model | Two-layer LSTM, hidden size 128 |
| Optimizer / loss | Adam / normalized-coordinate MSE |
| Learning rate / batch size / epochs | 0.001 / 128 / 30 |
| Training / validation masking seed | 42 / 43 |
| Reported device | CPU |
| Checkpoint selection | Lowest validation MSE; best recorded loss ≈ 0.00694 |

## Files and setup

| File in `objective4/`    | Purpose                                                     |
| ------------------------ | ----------------------------------------------------------- |
| `inspect_col_6.py`       | Measures missingness across scenarios and splits            |
| `objective4_data.py`     | Calculates the absence probability and prepares masking     |
| `retrain.py`             | Trains and saves the masked predictor                       |
| `evaluation_masked.py`   | Evaluates the masked checkpoint                             |
| `evaluation_original.py` | Evaluates the original checkpoint under matching conditions |
| `lstm_col6_masked.pt`    | Trained masked predictor and scalers                        |
| `objective4_bias.csv`    | Final ADE/FDE comparison and deltas                         |

Required original files are `sumo/lstm_predictor.py`, `sumo/baseline.py` and `sumo/lstm_baseline.pt`. Evaluation requires `Y`, `X_mask`, `X_corrupted`, `X_ch1_imputed` and `X_fully_imputed` in the test archives. Training also requires `X_gt`.

## How to run

Run from the shared project root.

```bash
python -m pip install numpy pandas torch scipy matplotlib
python objective4/inspect_col_6.py
python objective4/objective4_data.py
```

Retrain only when intentionally repeating the experiment; this can overwrite the masked checkpoint:

```bash
python objective4/retrain.py
```

Evaluate saved checkpoints without retraining:

```bash
python objective4/evaluation_masked.py
python objective4/evaluation_original.py
```

Each evaluation saves a Table-3-style CSV and a summary CSV beside the script:

- `objective4_table3.csv` and `objective4_summary.csv`
- `original_matched_table3.csv` and `original_matched_summary.csv`

Use the full-precision summaries to calculate the deltas in `objective4_bias.csv`. The evaluation scripts do not automatically regenerate that comparison file.

## Measuring the absence probability and applying masks

 1.   **input features**
Masking probability is measured from the 20%- penetration training file, training seeds 1-14, rather than choosing manually.

Each `.npz` file contains multiple observation windows from one simulation run, identified by its seed.

| Array             | Shape        | Meaning                                                 |
| ----------------- | ------------ | ------------------------------------------------------- |
| `X_gt`            | `(N, 30, 7)` | Original simulated history before artificial corruption |
| `Y`               | `(N, 50, 2)` | Ground-truth future absolute positions                  |
| `X_mask`          | `(N, 30, 7)` | Stored observation mask: 1 observed, 0 missing          |
| `X_corrupted`     | `(N, 30, 7)` | History after missingness injection                     |
| `X_ch1_imputed`   | `(N, 30, 7)` | History processed by Channel 1                          |
| `X_fully_imputed` | `(N, 30, 7)` | History processed by both channels                      |
| `follower_meta`   | `(N, 30)`    | Follower labels used to verify structural masks         |

`N` varies by file, each window contains 30 observations representing a 3-second history, followed by 50 prediction steps covering 5 seconds.

| Python index | Feature              |
| -----------: | -------------------- |
|            0 | `ego_speed`          |
|            1 | `ego_accel`          |
|            2 | `ego_x`              |
|            3 | `ego_y`              |
|            4 | `gap_to_leader`      |
|            5 | `leader_speed`       |
|        **6** | **`follower_speed`** |

We select column_6 for every window and timestep

2. **measuring probability**

```python
missing = data["X_mask"][:, :, 6] == 0`
```

The resulting boolean are combined into a mask bank of shape (9269,30) where True means missing and from this boolean we find the absence probability.

The measured rate is **99.002769%**, or approximately **0.99002769**. Validation and test data are not used to calculate this probability.

 3. **applying masks**
We copy the clean inputs (`X_gt`) from all three penetration scenarios and apply masking only to column 6 (`follower_speed`).

```python
X_masked = X_gt.astype(np.float32).copy()

absence_rate = float(mask_bank.mean())
missing = rng.random(X_masked.shape[:2]) < absence_rate

X_masked[:, :, 6] = np.where(
    missing, np.nan, X_masked[:, :, 6]
)
```

The measured absence probability is approximately **0.99002769**. For each window and timestep, the code independently generates a random number between 0 and 1. If it is below this probability, the follower-speed value is replaced with `NaN`; otherwise, its original value is retained.

Each follower-speed entry therefore has approximately a **99% chance of being hidden** and a **1% chance of being retained**. Columns 0–5 and future-position targets remain unchanged by masking. The original dataset files are not modified.

The masked inputs are then passed to the training pipeline, where missing values are replaced with physical zero before normalization. A fresh LSTM is trained on these inputs, and the generated masks remain fixed throughout training.

The realised masking rates were **98.993699% for training** and **99.001356% for validation**. Small differences from the measured probability are expected from random sampling. During evaluation, both models use the same actual stored test masks instead of newly generated masks.


## Results

**ADE** averages position error across all 50 future timesteps and test windows. **FDE** averages error at the final timestep (5 seconds). The Table `ADE_at_t (m)` column represents mean error at each individual timestep, not cumulative ADE.

All values below are in metres. Positive deltas indicate lower error after masked retraining. Differences are calculated before display rounding using the summary CSVs.

| Input condition | Original ADE | Masked ADE | Delta ADE | Original FDE | Masked FDE | Delta FDE |
|---|---:|---:|---:|---:|---:|---:|
| Zero-fill | 67.845359 | 39.737308 | 28.108052 | 73.972068 | 45.555203 | 28.416865 |
| Spline | 43.490306 | 12.149469 | 31.340838 | 45.132151 | 13.854305 | 31.277846 |
| Channel 1 | 53.803430 | 32.489971 | 21.313459 | 56.767196 | 37.481758 | 19.285438 |
| **Dual-channel** | **31.369666** | **1.564282** | **29.805385** | **33.015328** | **2.664993** | **30.350335** |

**Main deliverable: dual-channel Delta ADE = 29.805385 m and Delta FDE = 30.350335 m.**

The masked-trained predictor achieved lower ADE and FDE across all four input conditions. Under the dual-channel condition, ADE decreased from **31.370 m to 1.564 m**, while FDE decreased from **33.015 m to 2.665 m**. The measured differences were **29.805 m in ADE** and **30.350 m in FDE**.

These results suggest that training with follower speed almost always available leaves the original predictor poorly prepared for its absence during evaluation. Including the measured column-6 absence during training helped the new predictor perform better when that information was unavailable.


