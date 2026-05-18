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
Threshold estimator (power + HR at VT1 and VT2)
```

## Current Results (LOSO cross-validation, 18 subjects)

| Model | Accuracy | Sub-VT1 F1 | Mid-VT F1 | Supra-VT2 F1 | VT1 MAE | VT2 MAE |
|---|---|---|---|---|---|---|
| Random Forest | 74% | 0.88 | 0.59 | 0.68 | ~28W | ~36W |
| XGBoost | 72% | 0.88 | 0.59 | 0.62 | ~26W | ~36W |

The Sub-VT1 zone is classified reliably. Mid-VT is the hardest zone (transitions to both neighbours are ambiguous in HRV space). 4 of 18 subjects have undetected VT2, mostly due to data quality issues in this dataset.

## Features
| Feature | Description |
|---|---|
| SDNN | Standard deviation of NN intervals |
| RMSSD | Root mean square of successive differences |
| SampEn | Sample entropy (complexity) |
| Poincaré SD1 / SD2 | Short- and long-term HRV variability |
| DFA α1 | Detrended fluctuation analysis short-term scaling exponent |

DFA α1 and SampEn carry the most weight in the model.

## Evaluation
Training and evaluation use **Leave-One-Subject-Out (LOSO) cross-validation** — the model is trained on 17 subjects and tested on the held-out subject, repeated for all 18. This is the correct evaluation strategy for generalising to a new person; a simple train/test split would cause subject-level data leakage and inflate accuracy.

Data augmentation (splicing halves of same-class windows) is applied only within each training fold to prevent leakage.

## Running

```bash
uv run python src/main.py
```

Outputs:
- `data/cleaned_test_measure.csv` — preprocessed time series
- `data/classification_dataset.csv` — windowed, labelled dataset
- `data/threshold_estimates_{model}.csv` — per-subject VT1/VT2 estimates vs ground truth
- `plots/descriptive/` — per-subject RR and VO₂ plots with threshold markers
- `plots/results/` — confusion matrices and feature importance plots

## What's Next
- Add mean RR interval (heart rate proxy) as a feature — expected high impact
- Per-subject normalization by pre-test resting baseline — infrastructure ready, needs a dataset with a resting measurement
- Two-stage classifier (Sub-VT1 vs above → Mid-VT vs Supra-VT2)
- Validation on a larger, better-controlled dataset
