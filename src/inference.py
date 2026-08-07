#!/usr/bin/env python3
"""
Estimate VT1 and VT2 from a graded exercise test using a pre-trained model.

Requires trained models in models/ — run main.py first to generate them.

Input CSV must have one row per beat with columns:
  time   — elapsed time in seconds
  RR     — RR interval in milliseconds
  power  — power output in watts at that beat

Usage:
  uv run python src/inference.py path/to/test.csv
  uv run python src/inference.py path/to/test.csv --model xgboost
  uv run python src/inference.py path/to/test.csv --output results.csv
"""

import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import (
    remove_pre_post_periods,
    replace_missing_beats,
    create_classification_dataset,
    extract_resting_intervals,
    create_resting_windows,
)
from feature_extraction import extract_hrv_features

logger = logging.getLogger(__name__)

WINDOW_SIZE = 100


class InferenceError(Exception):
    """Raised when input data can't be turned into threshold estimates (bad columns, too few beats, etc.)."""


def estimate_thresholds(windows, probas, classes):
    classes = list(classes)
    idx_sub   = classes.index('Sub_vt1')
    idx_supra = classes.index('Supra_vt2')

    data = windows.reset_index(drop=True).copy()
    data['_p0'] = probas[:, 0]
    data['_p1'] = probas[:, 1]
    data['_p2'] = probas[:, 2]
    data['_mean_rr'] = [np.mean(rr) for rr in data['RR'].values]

    agg = (
        data.groupby('power')[['_p0', '_p1', '_p2', '_mean_rr']]
        .mean()
        .sort_index()
        .reset_index()
    )
    probs_smooth = (
        pd.DataFrame(agg[['_p0', '_p1', '_p2']].values)
        .rolling(3, min_periods=1, center=True)
        .mean()
        .values
    )
    pred_idx = np.argmax(probs_smooth, axis=1)
    powers   = agg['power'].values
    mean_rrs = agg['_mean_rr'].values

    vt1_power = vt1_hr = None
    for i, cls_idx in enumerate(pred_idx):
        if cls_idx != idx_sub:
            vt1_power = float(powers[i])
            vt1_hr = round(60000 / mean_rrs[i])
            break

    vt2_power = vt2_hr = None
    for i, cls_idx in enumerate(pred_idx):
        if cls_idx == idx_supra:
            vt2_power = float(powers[i])
            vt2_hr = round(60000 / mean_rrs[i])
            break

    return vt1_power, vt1_hr, vt2_power, vt2_hr


def load_model(model_name, models_dir):
    feature_cols_path = os.path.join(models_dir, 'feature_cols.json')
    if not os.path.exists(feature_cols_path):
        raise InferenceError(f"No trained models found in {models_dir}/. Run main.py first.")

    with open(feature_cols_path) as f:
        feature_cols = json.load(f)

    norm_path = os.path.join(models_dir, 'normalization.json')
    norm_stats = None
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm_stats = json.load(f)

    model_path = os.path.join(models_dir, f'{model_name}_final.pkl')
    if not os.path.exists(model_path):
        raise InferenceError(f"Model file not found: {model_path}. Run main.py first.")

    return joblib.load(model_path), feature_cols, norm_stats


def estimate_thresholds_from_data(data, model_name='rf', models_dir='models'):
    """
    Core pipeline: a beat-level DataFrame (columns: time, RR, power) -> (vt1_power, vt1_hr, vt2_power, vt2_hr).

    No file I/O and no printing — raises InferenceError on invalid/insufficient input so
    callers (CLI, API, tests) can decide how to surface it.
    """
    required = {'time', 'RR', 'power'}
    missing = required - set(data.columns)
    if missing:
        raise InferenceError(f"Input data is missing columns: {sorted(missing)}")

    bundle, feature_cols, norm_stats = load_model(model_name, models_dir)

    data = data.copy()
    data['ID'] = 'subject'
    data = replace_missing_beats(data)            # clean before extracting rest
    resting_raw = extract_resting_intervals(data)
    resting_windows = create_resting_windows(resting_raw, n=WINDOW_SIZE)
    data = remove_pre_post_periods(data)

    # Dummy label columns so create_classification_dataset works without ground truth
    for col in ('Sub_vt1', 'Mid_vt', 'Supra_vt2', 'At_vt'):
        data[col] = 0

    windows = create_classification_dataset(data, n=WINDOW_SIZE, stride=None)
    if len(windows) == 0:
        raise InferenceError(
            "No valid windows found. Each power step needs at least "
            f"{WINDOW_SIZE} consecutive beats."
        )

    features = extract_hrv_features(windows, feature_cols)

    # Apply resting-based normalization if the model was trained with it
    if norm_stats is not None:
        if len(resting_windows) == 0:
            logger.warning("No resting beats found (power == 0 before exercise). "
                            "Normalization skipped — results may be less accurate.")
        else:
            rest_feats = extract_hrv_features(resting_windows, feature_cols)
            rest_mean = rest_feats[feature_cols].iloc[0]
            pop_std = pd.Series(norm_stats['pop_std'])
            features = features.copy()
            features[feature_cols] = (features[feature_cols] - rest_mean) / pop_std

    X = features[feature_cols].values

    if model_name == 'rf':
        probas  = bundle.predict_proba(X)
        classes = bundle.classes_
    else:
        probas  = bundle['model'].predict_proba(X)
        classes = bundle['label_encoder'].classes_

    return estimate_thresholds(windows, probas, classes)


def run_inference(csv_path, model_name='rf', output_path=None, models_dir='models'):
    data = pd.read_csv(csv_path)
    try:
        vt1_power, vt1_hr, vt2_power, vt2_hr = estimate_thresholds_from_data(data, model_name, models_dir)
    except InferenceError as e:
        sys.exit(str(e))

    print("\n=== Threshold Estimates ===")
    if vt1_power is not None:
        print(f"VT1:  {vt1_power:.0f} W  |  {vt1_hr} bpm")
    else:
        print("VT1:  not detected")
    if vt2_power is not None:
        print(f"VT2:  {vt2_power:.0f} W  |  {vt2_hr} bpm")
    else:
        print("VT2:  not detected")

    if output_path:
        pd.DataFrame([{
            'vt1_power_W': vt1_power,
            'vt1_hr_bpm':  vt1_hr,
            'vt2_power_W': vt2_power,
            'vt2_hr_bpm':  vt2_hr,
        }]).to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    return vt1_power, vt1_hr, vt2_power, vt2_hr


def main():
    parser = argparse.ArgumentParser(
        description='Estimate VT1 and VT2 from a graded exercise test.'
    )
    parser.add_argument('csv', help='Path to input CSV (columns: time, RR, power)')
    parser.add_argument(
        '--model', choices=['rf', 'xgboost'], default='rf',
        help='Model to use (default: rf)'
    )
    parser.add_argument('--output', default=None, help='Optional path to save results CSV')
    parser.add_argument(
        '--models-dir', default='models',
        help='Directory containing trained models (default: models/)'
    )
    args = parser.parse_args()
    run_inference(args.csv, args.model, args.output, args.models_dir)


if __name__ == '__main__':
    main()
