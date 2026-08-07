import numpy as np
import pandas as pd
import nolds
import neurokit2 as nk


def calculate_dfa_alpha1(rr, min_window=4, max_window=16):
    return nolds.dfa(rr, nvals=list(range(min_window, max_window + 1)), order=1)


def calculate_sample_entropy(rr, m=2, r=0.2):
    tolerance = r * np.std(rr)
    return nolds.sampen(rr, emb_dim=m, tolerance=tolerance)


def calculate_poincare_sd(rr):
    peaks = nk.intervals_to_peaks(rr.tolist())
    hrv_nonlinear = nk.hrv_nonlinear(peaks, show=False)
    return float(hrv_nonlinear['HRV_SD1'].iloc[0]), float(hrv_nonlinear['HRV_SD2'].iloc[0])


def compute_features(rr, feature_set):
    rr = np.asarray(rr)
    features = {}

    if 'mean_rr' in feature_set:
        features['mean_rr'] = float(np.mean(rr))
    if 'sdnn' in feature_set:
        features['sdnn'] = float(np.std(rr))
    if 'rmssd' in feature_set:
        features['rmssd'] = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    if 'sampen' in feature_set:
        features['sampen'] = calculate_sample_entropy(rr)
    if {'poincare_sd1', 'poincare_sd2'} & feature_set:
        sd1, sd2 = calculate_poincare_sd(rr)
        if 'poincare_sd1' in feature_set:
            features['poincare_sd1'] = sd1
        if 'poincare_sd2' in feature_set:
            features['poincare_sd2'] = sd2
    if 'dfa_alpha1' in feature_set:
        features['dfa_alpha1'] = calculate_dfa_alpha1(rr)

    return features


def extract_hrv_features(classification_data, feature_set):
    feature_set = set(feature_set)
    features = []
    label_cols = [c for c in classification_data.columns if c not in ('RR', 'ID', 'power')]

    for _, row in classification_data.iterrows():
        feats = compute_features(row['RR'], feature_set)
        for label in label_cols:
            feats[label] = row[label]
        if 'ID' in classification_data.columns:
            feats['ID'] = row['ID']
        features.append(feats)

    return pd.DataFrame(features)
