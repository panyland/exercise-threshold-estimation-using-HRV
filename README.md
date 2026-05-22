# HRV-Based Exercise Threshold Estimation

## Overview
This project investigates using **heart rate variability (HRV)** to automatically identify **ventilatory thresholds (VT1 and VT2)** from a graded exercise test — without gas exchange measurements. Given raw RR interval data from a test, the model outputs the estimated power and heart rate at which each threshold occurs.

The approach: classify short RR interval windows into three exercise intensity zones (Sub-VT1, Mid-VT, Supra-VT2) using HRV features, then find where the predicted zone transitions across the test to estimate threshold locations.

## Data
18 young athletes (ages 12–18, fencing / kayak / triathlon) from the [PhysioNet ACTES dataset](https://physionet.org/content/actes-cycloergometer-exercise/1.0.0/). Each subject performed an incremental cycle ergometer test with concurrent RR interval, VO₂, and power output recording. Ground truth VT1 and VT2 are determined from gas exchange analysis.

## Pipeline

```
Raw RR + power data
       ↓
Preprocessing (outlier imputation, threshold marking, zone labelling)
       ↓
Classification dataset (100-beat windows per power step, per subject)
       ↓
HRV feature extraction (SDNN, RMSSD, SampEn, Poincaré SD1/SD2, DFA α1)
       ↓
LOSO cross-validation (Leave-One-Subject-Out)
  ├─ Random Forest (primary)
  └─ XGBoost (comparison)
       ↓
Threshold estimator (power and HR at VT1 and VT2)
       ↓
Final models saved to models/ for inference
```

## Training

Run the full pipeline — LOSO evaluation, result plots, and final model training:

```bash
uv run python src/main.py
```

Outputs:
- `data/cleaned_test_measure.csv` — preprocessed time series
- `data/classification_dataset.csv` — windowed, labelled dataset
- `plots/descriptive/` — per-subject RR and VO₂ plots with threshold markers
- `plots/results/confusion_matrix_{model}.png` — LOSO confusion matrices
- `plots/results/feature_importance_{model}.png` — mean feature importances
- `plots/results/threshold_estimates_{model}.csv` — per-subject VT1/VT2 estimates vs ground truth
- `models/rf_final.pkl` — Random Forest trained on all subjects
- `models/xgboost_final.pkl` — XGBoost trained on all subjects
- `models/feature_cols.json` — feature list used at training time

## Inference

Once models are trained, estimate thresholds for a new graded exercise test.

### Input format

A CSV file with one row per beat:

| column | unit | description |
|--------|------|-------------|
| `time` | seconds | elapsed time since test start |
| `RR` | milliseconds | RR interval |
| `power` | watts | power output at that beat |

Rows where `power == 0` (rest / cooldown) are dropped automatically.
Each power step needs at least 100 consecutive beats (~85 s at 70 bpm).

### Running inference

```bash
# Basic — prints VT1 and VT2 to stdout
uv run python src/inference.py path/to/test.csv

# Save results to a CSV file
uv run python src/inference.py path/to/test.csv --output results.csv

# Use XGBoost instead of Random Forest
uv run python src/inference.py path/to/test.csv --model xgboost

# Custom models directory
uv run python src/inference.py path/to/test.csv --models-dir path/to/models
```

Example output:
```
=== Threshold Estimates ===
VT1:  175 W  |  148 bpm
VT2:  245 W  |  167 bpm
```

### Feature selection

Features can be toggled in `src/main.py` by editing the `FEATURES` dict before re-running training:

```python
FEATURES = {
    'mean_rr':      False,   # mean RR interval (heart rate proxy)
    'sdnn':         True,
    'rmssd':        True,
    'sampen':       True,
    'poincare_sd1': True,
    'poincare_sd2': True,
    'dfa_alpha1':   True,
    'lf_power':     False,   # frequency domain (interpolated)
    'hf_power':     False,
    'lf_hf_ratio':  False,
}
```

After changing features, re-run `main.py` to retrain and overwrite the saved models. The inference script always reads `models/feature_cols.json` to stay consistent with whatever was used at training time.

## What's Next
- Add mean RR interval (heart rate proxy) as a feature
- Per-subject normalization by pre-test resting baseline
- Two-stage classifier (Sub-VT1 vs above → Mid-VT vs Supra-VT2)
- Validation on a larger, better-controlled dataset
