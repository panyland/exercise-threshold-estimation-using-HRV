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
```

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
- Add mean RR interval (heart rate proxy) as a feature
- Per-subject normalization by pre-test resting baseline
- Two-stage classifier (Sub-VT1 vs above → Mid-VT vs Supra-VT2)
- Validation on a larger, better-controlled dataset
