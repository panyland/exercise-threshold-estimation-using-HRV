import pandas as pd
import numpy as np


def remove_pre_post_periods(df: pd.DataFrame, power_column='power') -> pd.DataFrame:
    """
    Remove rows that represent pre or post measurement periods where power output is zero.

    """ 
    result_df = df.copy()
    result_df = result_df[result_df[power_column] != 0]
    
    return result_df.reset_index(drop=True)


def mark_thresholds(df: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    """
    Mark the VT1 and VT2 thresholds in the dataframe based on subject information.

    """
    result_df = df.copy()
    result_df = result_df.merge(subjects[['ID', 'P_vt1', 'P_vt2']], on='ID', how='left')
    result_df['vt1_marker'] = 0
    result_df['vt2_marker'] = 0

    for id_value, group in result_df.groupby('ID'):
        for vt, col in [('P_vt1', 'vt1_marker'), ('P_vt2', 'vt2_marker')]:
            p_vt = group[vt].iloc[0]
            power_values = group['power'].unique()
            power_values.sort()

            lower = power_values[power_values <= p_vt].max() if any(power_values <= p_vt) else None
            upper = power_values[power_values >= p_vt].min() if any(power_values >= p_vt) else None

            mask = (group['power'] == lower) | (group['power'] == upper)
            result_df.loc[mask & (result_df['ID'] == id_value), col] = 1

    return result_df 


def replace_missing_beats(df: pd.DataFrame, rr_column='RR', id_column='ID', median_multiplier=1.2, window_size=10) -> pd.DataFrame:
    """
    Replace outlier RR interval values that deviate significantly from the local median.

    """
    df = df.copy()
    result_frames = []

    for subject_id, group in df.groupby(id_column):
        rr_values = group[rr_column].to_numpy(dtype=float).copy()
        n = len(rr_values)
        
        for i in range(n):
            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)
            
            window = np.concatenate((rr_values[start:i], rr_values[i+1:end]))
            window = window[~np.isnan(window)]
            
            if len(window) == 0:
                continue

            median_val = np.median(window)

            if np.isnan(median_val):
                continue

            if abs(rr_values[i] - median_val) > median_val * (median_multiplier - 1):
                old_val = rr_values[i]
                rr_values[i] = median_val
                print(f"Replaced outlier RR value {old_val} at index {i} for ID {subject_id} with median {median_val}")

        group[rr_column] = rr_values
        result_frames.append(group)

    cleaned_df = pd.concat(result_frames, ignore_index=True)

    return cleaned_df


def label_rr_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label RR interval sequences to above VT2, below VT1 or between VT1 and VT2.

    """
    result_df = df.copy()
    result_df['Sub_vt1'] = (result_df['power'] < result_df['P_vt1']).astype(int)
    result_df['Mid_vt'] = ((result_df['power'] >= result_df['P_vt1']) & (result_df['power'] < result_df['P_vt2'])).astype(int)
    result_df['Supra_vt2'] = (result_df['power'] > result_df['P_vt2']).astype(int)
    result_df['At_vt'] = ((result_df['vt1_marker'] == 1) | (result_df['vt2_marker'] == 1)).astype(int)

    result_df.loc[result_df['At_vt'] == 1, ['Sub_vt1', 'Mid_vt', 'Supra_vt2']] = 0

    return result_df.reset_index(drop=True)


def create_classification_dataset(df: pd.DataFrame, n=100) -> pd.DataFrame:
    """
    Create a dataset suitable for classification by aggregating RR intervals and their labels.
    Parameters:
    - n: Number of RR intervals to include in each sequence

    """
    df = df.copy()
    grouped = (
        df.groupby(['ID', 'power'])
        .agg({
            'RR': list,
            'Sub_vt1': 'max',
            'Mid_vt': 'max',
            'Supra_vt2': 'max',
            'At_vt': 'max'
        })
        .reset_index()
    )
    result_df = grouped.copy()

    result_df = result_df[['RR', 'Sub_vt1', 'Mid_vt', 'Supra_vt2']].reset_index(drop=True)
    result_df = result_df[result_df['RR'].apply(len) >= n].copy()
    result_df['RR'] = result_df['RR'].apply(lambda x: x[-n:])
    result_df = result_df[result_df['RR'].apply(lambda x: not any(pd.isna(v) for v in x))].copy()

    if 'At_vt' in result_df.columns:
        result_df = result_df[result_df.get('At_vt', 0) == 0].reset_index(drop=True)

    return result_df[['RR', 'Sub_vt1', 'Mid_vt', 'Supra_vt2']]


def create_additional_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional RR interval sequences by combining halves of adjacent intervals with identical labels.
    
    """
    df = df.copy()

    df['class_id'] = df[['Sub_vt1', 'Mid_vt', 'Supra_vt2']].idxmax(axis=1)
    df = df.sort_values('class_id').reset_index(drop=True)
    df = df.drop(columns=['class_id'])

    additional_intervals = []

    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i+1]
        
        current_labels = current_row[['Sub_vt1', 'Mid_vt', 'Supra_vt2']].values
        next_labels = next_row[['Sub_vt1', 'Mid_vt', 'Supra_vt2']].values
        
        if np.array_equal(current_labels, next_labels):
            current_rr = current_row['RR']
            next_rr = next_row['RR']
            
            half_length = len(current_rr) // 2
            
            combined_rr = current_rr[-half_length:] + next_rr[:half_length]
            
            new_row = {
                'RR': combined_rr,
                'Sub_vt1': current_labels[0],
                'Mid_vt': current_labels[1], 
                'Supra_vt2': current_labels[2]
            }
            
            additional_intervals.append(new_row)

    additional_df = pd.DataFrame(additional_intervals)
    result_df = pd.concat([df, additional_df], ignore_index=True)

    return result_df

