import json
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from feature_extraction import extract_hrv_features
from preprocessing import (
    remove_pre_post_periods,
    mark_thresholds,
    replace_missing_beats,
    label_rr_intervals,
    create_classification_dataset,
    create_additional_intervals,
    extract_resting_intervals,
    create_resting_windows,
)
from plotting import (
    plot_subject_data,
    plot_confusion_matrix,
    plot_importances,
)

# Set a feature to True to include it in training, False to exclude.
# Disabled features are never computed.
FEATURES = {
    'mean_rr':      False,   # mean RR interval (heart rate proxy)
    'sdnn':         True,
    'rmssd':        True,
    'sampen':       True,
    'poincare_sd1': True,
    'poincare_sd2': True,
    'dfa_alpha1':   True,
    'lf_power':     False,   # frequency domain — interpolated; may be noisy during exercise
    'hf_power':     False,
    'lf_hf_ratio':  False,
}
FEATURE_COLS = [k for k, v in FEATURES.items() if v]

WINDOW_SIZE = 100
STRIDE = 50   # 50% overlapping windows; set to None to revert to one window per power step

MODELS = {
    'RandomForest': RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight='balanced', random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=42
    ),
}


def load_data(measure_path, subject_path):
    data = pd.read_csv(measure_path)
    subjects = pd.read_csv(subject_path)
    return data, subjects


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def compute_baseline_stats(baseline_features, feature_cols):
    """
    Compute mean and std from a set of baseline condition features.

    In deployment with a resting measurement: pass features extracted from the
    pre-test resting period here instead of Sub-VT1 windows.
    """
    mean = baseline_features[feature_cols].mean()
    std = baseline_features[feature_cols].std().clip(lower=1e-6)
    return mean, std


def apply_normalization(features_df, baseline_mean, baseline_std, feature_cols):
    """Z-score normalize feature_cols relative to the provided baseline statistics."""
    result = features_df.copy()
    result[feature_cols] = (features_df[feature_cols] - baseline_mean) / baseline_std
    return result


def normalize_with_resting(features_df, resting_features, pop_std, feature_cols):
    """
    Normalize exercise features using each subject's pre-exercise resting baseline.

    Centering: per-subject resting mean (removes individual differences at rest).
    Scaling: population resting std across all training subjects (consistent scale).
    Augmented samples (NaN ID) fall back to the population resting mean.
    """
    result = features_df.copy()
    rest_by_subject = resting_features.set_index('ID')[feature_cols]
    pop_mean = resting_features[feature_cols].mean()

    for subject_id, group in features_df.dropna(subset=['ID']).groupby('ID'):
        idx = group.index
        subj_mean = rest_by_subject.loc[subject_id] if subject_id in rest_by_subject.index else pop_mean
        result.loc[idx, feature_cols] = (
            (features_df.loc[idx, feature_cols] - subj_mean) / pop_std
        ).values

    nan_id = features_df['ID'].isna()
    if nan_id.sum() > 0:
        result.loc[nan_id, feature_cols] = (
            (features_df.loc[nan_id, feature_cols] - pop_mean) / pop_std
        ).values

    return result


# ---------------------------------------------------------------------------
# Threshold estimation
# ---------------------------------------------------------------------------

def estimate_thresholds(test_data, probas, rf_classes):
    """
    Estimate VT1 and VT2 from per-window classifier probabilities.

    Probabilities are averaged per power level (handles multiple sliding windows per step),
    smoothed with a centered rolling window of 3, then the dominant class sequence is scanned:
      VT1 = first power level where dominant class transitions away from Sub_vt1.
      VT2 = first power level where dominant class is Supra_vt2.

    Returns (vt1_power, vt1_hr, vt2_power, vt2_hr); values are None if undetected.
    """
    classes = list(rf_classes)
    idx_sub   = classes.index('Sub_vt1')
    idx_supra = classes.index('Supra_vt2')

    data = test_data.reset_index(drop=True).copy()
    data['_p0'] = probas[:, 0]
    data['_p1'] = probas[:, 1]
    data['_p2'] = probas[:, 2]
    data['_mean_rr'] = [np.mean(rr) for rr in data['RR'].values]

    # Average across all windows within each power step
    agg = (
        data.groupby('power')[['_p0', '_p1', '_p2', '_mean_rr']]
        .mean()
        .sort_index()
        .reset_index()
    )

    probs_agg = agg[['_p0', '_p1', '_p2']].values
    probs_smooth = (
        pd.DataFrame(probs_agg)
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


# ---------------------------------------------------------------------------
# LOSO cross-validation
# ---------------------------------------------------------------------------

def run_loso(classification_data, subjects, resting_windows):
    subject_ids = sorted(classification_data['ID'].unique())
    base_cols = ['ID', 'power', 'RR', 'Sub_vt1', 'Mid_vt', 'Supra_vt2']

    all_y_true  = {name: [] for name in MODELS}
    all_y_pred  = {name: [] for name in MODELS}
    importances = {name: [] for name in MODELS}
    thresholds  = {name: [] for name in MODELS}

    for test_subject in subject_ids:
        print(f"\nLOSO fold: held-out subject = {test_subject}")

        train_data = classification_data[classification_data['ID'] != test_subject][base_cols].copy()
        test_data  = classification_data[classification_data['ID'] == test_subject][base_cols].copy()

        train_data_aug = create_additional_intervals(train_data)
        train_data_aug['VT_label_3class'] = train_data_aug[['Sub_vt1', 'Mid_vt', 'Supra_vt2']].idxmax(axis=1)
        test_data['VT_label_3class']      = test_data[['Sub_vt1', 'Mid_vt', 'Supra_vt2']].idxmax(axis=1)

        fold_counts = {c: int(train_data_aug[c].sum()) for c in ['Sub_vt1', 'Mid_vt', 'Supra_vt2']}
        print(f"  Training counts (post-augmentation): {fold_counts}")

        # Extract features once per fold
        train_features = extract_hrv_features(train_data_aug, FEATURE_COLS)
        test_features  = extract_hrv_features(test_data, FEATURE_COLS)

        # --- Resting-based normalization (disabled) ---
        # Centering by resting HRV hurt accuracy on this heterogeneous dataset:
        # atypical resting baselines in some subjects inverted the exercise signal.
        # Re-enable when a larger, more homogeneous dataset with resting data is available.
        # train_rest = all_resting_features[all_resting_features['ID'] != test_subject]
        # test_rest  = all_resting_features[all_resting_features['ID'] == test_subject]
        # _, pop_std = compute_baseline_stats(train_rest, FEATURE_COLS)
        # train_features = normalize_with_resting(train_features, train_rest, pop_std, FEATURE_COLS)
        # test_rest_mean = test_rest[FEATURE_COLS].iloc[0]
        # test_features = test_features.copy()
        # test_features[FEATURE_COLS] = (test_features[FEATURE_COLS] - test_rest_mean) / pop_std

        X_train = train_features[FEATURE_COLS].values
        y_train = train_data_aug['VT_label_3class'].values
        X_test  = test_features[FEATURE_COLS].values
        y_test  = test_data['VT_label_3class'].values

        gt = subjects[subjects['ID'] == test_subject].iloc[0]
        vt1_true, vt2_true = float(gt['P_vt1']), float(gt['P_vt2'])

        for name, model in MODELS.items():
            if name == 'XGBoost':
                le = LabelEncoder()
                y_train_enc = le.fit_transform(y_train)
                sw = compute_sample_weight('balanced', y_train_enc)
                model.fit(X_train, y_train_enc, sample_weight=sw)
                y_pred = le.inverse_transform(model.predict(X_test))
                probas = model.predict_proba(X_test)
                classes = le.classes_
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                probas = model.predict_proba(X_test)
                classes = model.classes_

            all_y_true[name].extend(y_test)
            all_y_pred[name].extend(y_pred)
            importances[name].append(model.feature_importances_)

            fold_acc = (y_pred == y_test).mean()

            vt1_pred, vt1_hr, vt2_pred, vt2_hr = estimate_thresholds(test_data, probas, classes)
            vt1_err = abs(vt1_pred - vt1_true) if vt1_pred is not None else float('nan')
            vt2_err = abs(vt2_pred - vt2_true) if vt2_pred is not None else float('nan')

            vt1_str = f"{vt1_pred:.0f}W ({vt1_hr} bpm)" if vt1_pred is not None else "undetected"
            vt2_str = f"{vt2_pred:.0f}W ({vt2_hr} bpm)" if vt2_pred is not None else "undetected"
            print(f"  [{name}] acc={fold_acc:.3f} | "
                  f"VT1: {vt1_str} / true={vt1_true:.0f}W err={vt1_err:.0f}W | "
                  f"VT2: {vt2_str} / true={vt2_true:.0f}W err={vt2_err:.0f}W")

            thresholds[name].append({
                'subject':     test_subject,
                'vt1_true_W':  vt1_true,  'vt1_pred_W':  vt1_pred,
                'vt1_err_W':   vt1_err,   'vt1_pred_HR': vt1_hr,
                'vt2_true_W':  vt2_true,  'vt2_pred_W':  vt2_pred,
                'vt2_err_W':   vt2_err,   'vt2_pred_HR': vt2_hr,
            })

    return all_y_true, all_y_pred, importances, thresholds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data, subjects = load_data('data/test_measure.csv', 'data/subject-info.csv')

    data = replace_missing_beats(data)            # clean all beats first (including resting)
    resting_data = extract_resting_intervals(data)
    resting_windows = create_resting_windows(resting_data, n=WINDOW_SIZE)
    print(f"Resting windows: {len(resting_windows)}/{len(subjects)} subjects")

    data = remove_pre_post_periods(data)
    data = mark_thresholds(data, subjects)
    data.to_csv('data/cleaned_test_measure.csv', index=False)
    plot_subject_data(data)
    data = label_rr_intervals(data)

    classification_data = create_classification_dataset(data, n=WINDOW_SIZE, stride=STRIDE)
    key = classification_data['RR'].apply(lambda x: len(x) > 0 and not any(pd.isna(v) for v in x))
    classification_data = classification_data[key].reset_index(drop=True)

    classification_data['VT_label_3class'] = classification_data[['Sub_vt1', 'Mid_vt', 'Supra_vt2']].idxmax(axis=1)
    classification_data['Supra_vt1'] = (classification_data['Mid_vt'] + classification_data['Supra_vt2']).clip(0, 1)
    classification_data.to_csv('data/classification_dataset.csv', index=False)

    label_counts = {c: int(classification_data[c].sum()) for c in ['Sub_vt1', 'Mid_vt', 'Supra_vt2']}
    print(f"Classification dataset label counts (stride={STRIDE}): {label_counts}")

    all_y_true, all_y_pred, importances, thresholds = run_loso(
        classification_data, subjects, resting_windows
    )

    for name in MODELS:
        y_true = np.array(all_y_true[name])
        y_pred = np.array(all_y_pred[name])
        suffix = f'_{name.lower()}'

        print(f"\n=== LOSO Results: {name} ===")
        print(classification_report(y_true, y_pred))

        plot_confusion_matrix(confusion_matrix(y_true, y_pred), y_true, suffix=suffix)
        plot_importances(np.mean(importances[name], axis=0), FEATURE_COLS, suffix=suffix)

        tdf = pd.DataFrame(thresholds[name])
        print(f"--- Threshold Estimation: {name} ---")
        print(tdf.to_string(index=False))
        print(f"MAE — VT1: {tdf['vt1_err_W'].mean():.1f}W   VT2: {tdf['vt2_err_W'].mean():.1f}W")
        tdf.to_csv(f'plots/results/threshold_estimates_{name.lower()}.csv', index=False)

    save_final_models(classification_data, resting_windows)


# ---------------------------------------------------------------------------
# Final model — trained on all subjects for inference
# ---------------------------------------------------------------------------

def save_final_models(classification_data, resting_windows):
    """Train on all subjects and save models + normalization stats for inference."""
    os.makedirs('models', exist_ok=True)

    base_cols = ['ID', 'power', 'RR', 'Sub_vt1', 'Mid_vt', 'Supra_vt2']
    train_data = classification_data[base_cols].copy()
    train_data_aug = create_additional_intervals(train_data)
    train_data_aug['VT_label_3class'] = (
        train_data_aug[['Sub_vt1', 'Mid_vt', 'Supra_vt2']].idxmax(axis=1)
    )

    train_features = extract_hrv_features(train_data_aug, FEATURE_COLS)

    X = train_features[FEATURE_COLS].values
    y = train_data_aug['VT_label_3class'].values

    with open('models/feature_cols.json', 'w') as f:
        json.dump(FEATURE_COLS, f)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight='balanced', random_state=42
    )
    rf.fit(X, y)
    joblib.dump(rf, 'models/rf_final.pkl')

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    sw = compute_sample_weight('balanced', y_enc)
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=42
    )
    xgb.fit(X, y_enc, sample_weight=sw)
    joblib.dump({'model': xgb, 'label_encoder': le}, 'models/xgboost_final.pkl')

    print(f"Final models saved to models/  (features: {FEATURE_COLS})")


if __name__ == '__main__':
    main()
