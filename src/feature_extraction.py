import numpy as np
import pandas as pd
from scipy.signal import welch
import nolds
import neurokit2 as nk


def calculate_dfa_alpha1(rr: np.ndarray, min_window: int = 4, max_window: int = 16) -> float:
    alpha1 = nolds.dfa(rr, nvals=list(range(min_window, max_window + 1)), order=1) 
    return alpha1


def calculate_sample_entropy(rr: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    tolerance = r * np.std(rr) 
    sampen_value = nolds.sampen(rr, emb_dim=m, tolerance=tolerance)
    
    return sampen_value


def calculate_poincare_sd(rr: np.ndarray) -> float:
    # NeuroKit2 expects a pandas Series or list
    rr_series = rr.tolist()
    peaks = nk.intervals_to_peaks(rr_series)
    # Compute nonlinear HRV indices
    hrv_nonlinear = nk.hrv_nonlinear(peaks, show=False)
    
    # Extract SD1 and SD2
    sd1 = hrv_nonlinear['HRV_SD1'][0]
    sd2 = hrv_nonlinear['HRV_SD2'][0]
    
    return sd1/sd2


def compute_features(rr: np.ndarray, fs: float = 4.0, include_freq: bool = False) -> dict:
    rr = np.asarray(rr)
    rr_diff = np.diff(rr)

    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(rr_diff ** 2))
    sampen = calculate_sample_entropy(rr)
    poincare = calculate_poincare_sd(rr)
    dfa_alpha1 = calculate_dfa_alpha1(rr)

    features = {
        "sdnn": sdnn,
        "rmssd": rmssd,
        "sampen": sampen,
        "poincare_sd": poincare,
        "dfa_alpha1": dfa_alpha1
    }

    if include_freq:
        f, pxx = welch(rr, fs=fs, nperseg=min(256, len(rr)))

        def band_power(freqs, power, band):
            mask = (freqs >= band[0]) & (freqs < band[1])
            return np.trapezoid(power[mask], freqs[mask]) if np.any(mask) else 0.0

        lf = band_power(f, pxx, (0.04, 0.15))
        hf = band_power(f, pxx, (0.15, 0.4))
        total_power = band_power(f, pxx, (0.0033, 0.4))
        lf_hf = lf / hf if hf > 0 else np.nan

        features.update({
            "lf": lf,
            "hf": hf,
            "lf_hf": lf_hf,
            "total_power": total_power
        })

    return features


def extract_hrv_features(classification_data: pd.DataFrame, include_freq=False) -> pd.DataFrame:
    features = []

    label_cols = [c for c in classification_data.columns if c != "RR"]

    for _, row in classification_data.iterrows():
        rr = row["RR"]
        feats = compute_features(rr, include_freq=include_freq)
        for label in label_cols:
            feats[label] = row[label]
        features.append(feats)

    return pd.DataFrame(features)

