# relative_displacement_objective

1. PROJECT OVERVIEW
------------------
This project studies vehicle trajectory prediction when connected-vehicle
information is incomplete or missing.

The project contains:
- SUMO-based traffic simulation and data-generation utilities.
- Objective 1: trajectory prediction using relative displacement instead of
  absolute coordinates.
- Objective 4: measurement of prediction bias caused by missing follower speed
  (input feature / column 6).
- Objective 4 Relative: a separate relative-displacement experiment that
  investigates the effect of realistic column-6 missingness.


2. SOFTWARE REQUIREMENTS
------------------------
Recommended environment:
- Windows 10/11, Linux or macOS
- Python 3.10 or newer (the included bytecode suggests Python 3.11/3.12
  has been used; use a compatible Python version)
- pip
- NumPy
- Pandas
- PyTorch
- SciPy and Matplotlib for some analysis / plotting scripts
- SUMO and its Python tools (TraCI) only if you intend to run the traffic
  simulation and SUMO data-generation pipeline

Install the common Python packages:

    python -m pip install --upgrade pip
    python -m pip install numpy pandas torch scipy matplotlib

For Objective 1 and Objective 4 Relative, the supplied requirements files list
the core packages. Install them with:

    python -m pip install -r objective1/requirements.txt
    python -m pip install -r objective4_relative/requirements.txt

If using SUMO scripts, install SUMO separately and configure SUMO_HOME and
the SUMO tools directory according to the official SUMO installation
instructions. SUMO is not required merely to train/evaluate the experiments
using the already generated files in data_corrupted.

A GPU is optional. PyTorch can run on CPU, although training may take longer.
Install the PyTorch build appropriate for your machine from the official
PyTorch installation instructions.

3. MAIN FOLDER STRUCTURE
------------------------
project_root/
|
|-- data_corrupted/
|   |-- penetration_10/
|   |   |-- run_seed_1.npz ... run_seed_20.npz
|   |-- penetration_20/
|   |   |-- run_seed_1.npz ... run_seed_20.npz
|   |-- penetration_30/
|       |-- run_seed_1.npz ... run_seed_20.npz
|
|-- sumo/
|   |-- lstm_predictor.py
|   |-- lstm_baseline.pt
|   |-- run_pipeline.py
|   |-- train_lstm.py
|   |-- train_grud.py
|   |-- simulate_rsu.py
|   |-- inject_corruption.py
|   |-- integrate_ch1.py
|   |-- integrate_dual_channel.py
|   |-- evaluate_e1.py ... evaluate_e5.py
|   |-- highway.net.xml, highway.sumocfg, route and traffic XML files
|   |-- highD/
|       |-- scripts for HighD data extraction, corruption, checks and evaluation
|   |-- CSV result tables, figures, model checkpoints and supporting files
|
|-- objective1/
|   |-- config.json
|   |-- requirements.txt
|   |-- run_objective1.py
|   |-- compare_baseline.py
|   |-- checkpoints/
|   |-- results/
|   |-- logs/
|
|-- objective4/
|   |-- inspect_col_6.py
|   |-- objective4_data.py
|   |-- retrain.py
|   |-- evaluation_masked.py
|   |-- evaluation_original.py
|   |-- lstm_col6_masked.pt
|   |-- results/
|
|-- objective4_relative/
|   |-- config.json
|   |-- requirements.txt
|   |-- objective4_relative.py
|   |-- checkpoints/
|   |-- results/
|   |-- logs/
|
|-- inspect_project_files.py

4. DATASET FORMAT
-----------------
The data_corrupted folder contains NumPy .npz archives. Each archive represents
a simulation run identified by its seed and penetration scenario.

The three scenarios are:
- penetration_10
- penetration_20
- penetration_30

The principal arrays used by the experiments are:

- X_gt:             original / ground-truth input history
- Y:                future ground-truth trajectory
- X_mask:           observation mask (1 = observed, 0 = missing)
- X_corrupted:      input after missingness has been introduced
- X_ch1_imputed:    input after Channel 1 imputation
- X_fully_imputed:  input after the dual-channel imputation process
- follower_meta:    follower information used by some diagnostics

Typical shapes:
- X_gt and other input histories: (number_of_windows, 30, 7)
- Y:                              (number_of_windows, 50, 2)
- X_mask:                         (number_of_windows, 30, 7)

The exact number of windows varies between files. The experiments use 30
historical time steps and predict 50 future time steps.

Input feature order (zero-based Python indexing):
  0 - ego_speed
  1 - ego_accel
  2 - ego_x
  3 - ego_y
  4 - gap_to_leader
  5 - leader_speed
  6 - follower_speed

Column 6 therefore means follower_speed, not the sixth item in
one-based numbering.

The common seed split is:
- Training:   seeds 1-14
- Validation: seeds 15-16
- Testing:    seeds 17-20

Always check the relevant config file and script before changing the split or
data-array names.

5. WHAT EACH SECTION IMPLEMENTS
-------------------------------

A. SUMO
-------
The sumo folder contains the original simulation/model pipeline and supporting
experiments.

Important files:
- lstm_predictor.py: shared LSTM model definition and related scaling utilities.
- lstm_baseline.pt: saved original absolute-coordinate LSTM checkpoint.
- run_pipeline.py: main SUMO / data-processing pipeline orchestrator.
- train_lstm.py and train_grud.py: model-training scripts.
- simulate_rsu.py: RSU-related simulation logic.
- inject_corruption.py: introduces missing/corrupted observations.
- integrate_ch1.py and integrate_dual_channel.py: imputation / channel
  integration steps.
- evaluate_e1.py to evaluate_e5.py: evaluation scripts for the E1-E5
  experiments.
- highD/: scripts for processing and evaluating the HighD dataset, including
  extraction, missingness injection, diagnostics and evaluation.

The SUMO folder also contains network, route, configuration and traffic XML
files, trained checkpoints, CSV tables and generated figures.

B. OBJECTIVE 1 - RELATIVE-DISPLACEMENT TRAJECTORY PREDICTION
------------------------------------------------------------
Purpose:
The original baseline predicts future absolute (x, y) coordinates. Absolute
coordinates can depend on the map origin, making transfer/comparison across
different datasets difficult.

Implementation:
- Reuses the LSTM architecture from sumo/lstm_predictor.py.
- Uses 30 history steps and seven input features.
- Changes the prediction target to step-by-step displacement (dx, dy).
- Predicts 50 future displacement steps.
- Adds the predicted displacements cumulatively to the last observed position
  to reconstruct the future trajectory.
- Trains with training seeds 1-14, validates on 15-16 and tests on 17-20,
  across the configured penetration scenarios.
- Saves the trained model and training/evaluation results.

Main files:
- config.json: data paths, scenarios, seed split and model/training settings.
- run_objective1.py: trains the relative-displacement model.
- compare_baseline.py: compares the trained relative model with the original
  absolute-coordinate baseline.
- checkpoints/: saved trained model.
- results/: summary metrics, training history and baseline comparison CSVs.
- logs/: run summary.

Run Objective 1 from the project root:

    python -m pip install -r objective1/requirements.txt
    python objective1/run_objective1.py
    python objective1/compare_baseline.py

The first command trains the model. The second evaluation command compares
the saved model with the original baseline. Do not delete or rename
sumo/lstm_predictor.py, sumo/lstm_baseline.pt or the dataset files.

C. OBJECTIVE 4 - COLUMN-6 MISSINGNESS / BIAS QUANTIFICATION
-----------------------------------------------------------
Purpose:
Measures how missing follower_speed (feature index 6) affects trajectory
prediction when comparing the original model with a model trained using
artificially masked follower-speed inputs.

Implementation:
- Measures the column-6 absence rate from the stored observation masks.
- Applies missingness to follower_speed in training inputs.
- Trains a separate masked-input predictor.
- Evaluates the original and masked-trained models under matching test
  conditions.
- Compares ADE/FDE and calculates the difference between model results.

Main files:
- inspect_col_6.py: inspects missingness in feature 6.
- objective4_data.py: calculates missingness and prepares masked data.
- retrain.py: trains the masked model.
- evaluation_masked.py: evaluates the masked-trained checkpoint.
- evaluation_original.py: evaluates the original checkpoint.
- results/: CSV summaries and comparison tables.
- lstm_col6_masked.pt: saved masked-trained model.

Run from the project root:

    python -m pip install numpy pandas torch scipy matplotlib
    python objective4/inspect_col_6.py
    python objective4/objective4_data.py
    python objective4/retrain.py
    python objective4/evaluation_masked.py
    python objective4/evaluation_original.py

Retraining may overwrite the existing checkpoint. Skip retraining if you only
want to evaluate the already saved model.

D. OBJECTIVE 4 RELATIVE - COLUMN-6 BIAS WITH RELATIVE DISPLACEMENT
------------------------------------------------------------------
Purpose:
Studies the effect of realistic follower_speed absence on the relative-
displacement trajectory predictor from Objective 1.

Implementation:
- Uses the relative-displacement prediction representation.
- Measures the follower_speed absence probability from the configured
  penetration_20 training data.
- Trains a new model with column 6 randomly masked (scratch training, as
  described in the section README and run configuration).
- Compares the original Objective 1 model and the masked-trained model on
  the same held-out test windows.
- Evaluates multiple input conditions, including zero-filled, spline-
  interpolated, Channel 1 and dual-channel inputs.
- Reports ADE/FDE, bias comparisons, diagnostics and training history.
- Uses observed-position anchors for the primary evaluation; oracle-anchor
  results are reference-only and must not be presented as deployable results.

Main files:
- config.json: experiment settings and paths.
- objective4_relative.py: training and evaluation workflow.
- checkpoints/: saved model checkpoint.
- results/: comparison, bias, diagnostics and summary CSV files.
- logs/: run summary.

Run from the project root:

    python -m pip install -r objective4_relative/requirements.txt
    python objective4_relative/objective4_relative.py

To evaluate without training, if the required checkpoint already exists:

    python objective4_relative/objective4_relative.py --skip-train

Check the script's --help option for any additional supported command-line
options:

    python objective4_relative/objective4_relative.py --help

6. RESULTS AND OUTPUTS
----------------------
CSV files in each objective's results folder contain experiment-specific
metrics and comparisons. Common output types include:
- summary metrics: overall evaluation performance.
- training_history.csv: training and validation loss across epochs.
- baseline_comparison*.csv: baseline-versus-proposed-model comparisons.
- objective4* bias CSV files: measured bias and model differences.
- anchor_diagnostics.csv: diagnostics related to trajectory anchoring.
- masking_rates_by_scenario_seed.csv: measured missingness rates.
- logs/run_summary*.json: configuration and run-level summary information.
- checkpoints/*.pt: trained PyTorch model weights and associated saved data.


