import numpy as np
import pandas as pd
import nolds
import neurokit2 as nk


def calculate_dfa_alpha1(rr: np.ndarray, min_window: int = 4, max_window: int = 16) -> float:
    """
    Calculate the Detrended Fluctuation Analysis (DFA) alpha1 coefficient of an RR interval sequence.
    Parameters:
    - rr: RR interval sequence
    - min_window: Minimum window size for DFA
    - max_window: Maximum window size for DFA
    """
    alpha1 = nolds.dfa(rr, nvals=list(range(min_window, max_window + 1)), order=1) 
    return alpha1


def calculate_sample_entropy(rr: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Calculate sample entropy of an RR interval sequence.
    Parameters:
    - rr: RR interval sequence
    - m: Embedding dimension
    - r: Tolerance as a fraction of the standard deviation of the RR intervals
    """
    tolerance = r * np.std(rr) 
    sampen_value = nolds.sampen(rr, emb_dim=m, tolerance=tolerance)
    
    return sampen_value


def calculate_poincare_sd(rr: np.ndarray) -> float:
    """
    Calculate Poincare standard deviations SD1 and SD2 from an RR interval sequence.
    """
    rr_series = rr.tolist()
    peaks = nk.intervals_to_peaks(rr_series)
    hrv_nonlinear = nk.hrv_nonlinear(peaks, show=False)
    print(hrv_nonlinear)
    sd1 = hrv_nonlinear['HRV_SD1'][0]
    sd2 = hrv_nonlinear['HRV_SD2'][0]
    
    return sd1, sd2


def compute_features(rr: np.ndarray) -> dict:
    """
    Compute standard deviation, root mean square of successive differences, sample entropy, poincare standard deviation and 
    detrended fluctuation analysis coefficient alfa1 from a single RR interval sequence.
    """
    rr = np.asarray(rr)
    rr_diff = np.diff(rr)

    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(rr_diff ** 2))
    sampen = calculate_sample_entropy(rr)
    poincare1, poincare2 = calculate_poincare_sd(rr)
    dfa_alpha1 = calculate_dfa_alpha1(rr)

    features = {
        "sdnn": sdnn,
        "rmssd": rmssd,
        "sampen": sampen,
        "poincare_sd1": poincare1,
        "poincare_sd2": poincare2,
        "dfa_alpha1": dfa_alpha1
    }
    return features


def extract_hrv_features(classification_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from RR interval sequences.
    Parameters:
    - classification_data: DataFrame containing RR interval sequences and their labels
    """
    features = []

    label_cols = [c for c in classification_data.columns if c not in ('RR', 'ID', 'power')]

    for _, row in classification_data.iterrows():
        rr = row["RR"]
        feats = compute_features(rr)
        for label in label_cols:
            feats[label] = row[label]
        if 'ID' in classification_data.columns:
            feats['ID'] = row['ID']
        features.append(feats)

    return pd.DataFrame(features)

