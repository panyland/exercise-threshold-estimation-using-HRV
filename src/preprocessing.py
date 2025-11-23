import pandas as pd
import numpy as np


def remove_pre_post_periods(df, power_column='power'):
    result_df = df.copy()
    result_df = result_df[result_df[power_column] != 0]
    
    return result_df.reset_index(drop=True)


def mark_thresholds(data, subjects):
    data = data.merge(subjects[['ID', 'P_vt1', 'P_vt2']], on='ID', how='left')
    data['vt1_marker'] = 0
    data['vt2_marker'] = 0

    for id_value, group in data.groupby('ID'):
        for vt, col in [('P_vt1', 'vt1_marker'), ('P_vt2', 'vt2_marker')]:
            p_vt = group[vt].iloc[0]
            power_values = group['power'].unique()
            power_values.sort()

            lower = power_values[power_values <= p_vt].max() if any(power_values <= p_vt) else None
            upper = power_values[power_values >= p_vt].min() if any(power_values >= p_vt) else None

            mask = (group['power'] == lower) | (group['power'] == upper)
            data.loc[mask & (data['ID'] == id_value), col] = 1

    return data 


def replace_missing_beats(df, rr_column='RR', id_column='ID', median_multiplier=1.2, window_size=10):
    df = df.copy()
    result_frames = []

    for subject_id, group in df.groupby(id_column):
        rr_values = group[rr_column].to_numpy(dtype=float).copy()
        n = len(rr_values)
        
        for i in range(n):
            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)
            
            # Exclude the current value from the window
            window = np.concatenate((rr_values[start:i], rr_values[i+1:end]))
            
            # Remove NaNs from the window
            window = window[~np.isnan(window)]
            
            # Skip if no valid values left
            if len(window) == 0:
                continue

            median_val = np.median(window)

            # Skip if median is NaN (shouldn't happen after nan removal)
            if np.isnan(median_val):
                continue

            # Replace if abs difference is larger than threshold
            if abs(rr_values[i] - median_val) > median_val * (median_multiplier - 1):
                old_val = rr_values[i]
                rr_values[i] = median_val
                print(f"Replaced outlier RR value {old_val} at index {i} for ID {subject_id} with median {median_val}")

        group[rr_column] = rr_values
        result_frames.append(group)

    # Combine all processed groups
    cleaned_df = pd.concat(result_frames, ignore_index=True)

    return cleaned_df


def label_rr_intervals(df):
    result_df = df.copy()
    result_df['Sub_vt1'] = (result_df['power'] < result_df['P_vt1']).astype(int)
    result_df['Mid_vt'] = ((result_df['power'] >= result_df['P_vt1']) & (result_df['power'] < result_df['P_vt2'])).astype(int)
    result_df['Supra_vt2'] = (result_df['power'] > result_df['P_vt2']).astype(int)
    result_df['At_vt'] = ((result_df['vt1_marker'] == 1) | (result_df['vt2_marker'] == 1)).astype(int)

    result_df.loc[result_df['At_vt'] == 1, ['Sub_vt1', 'Mid_vt', 'Supra_vt2']] = 0

    return result_df.reset_index(drop=True)


def create_classification_dataset(df, n=100, sampling_rate=4, interpolate=False):
    df = df.copy()

    # --- Case 1: Interpolation Mode ---
    if interpolate:
        grouped = df.groupby(['ID', 'power'])
        results = []

        for (ID, power), group in grouped:
            group = group.sort_values('time')
            t = group['time'].values
            rr = group['RR'].values

            if len(t) < 2:
                continue  # cannot interpolate single-point group

            # Uniform time grid (4 Hz → 0.25 s steps)
            t_uniform = np.arange(t.min(), t.max(), 1 / sampling_rate)
            rr_interp = np.interp(t_uniform, t, rr)

            # Take last n samples (skip too-short)
            if len(rr_interp) >= n:
                rr_interp = rr_interp[-n:]
            else:
                continue

            results.append({
                'RR': rr_interp.tolist(),
                'Sub_vt1': group['Sub_vt1'].max(),
                'Mid_vt': group['Mid_vt'].max(),
                'Supra_vt2': group['Supra_vt2'].max(),
                'At_vt': group['At_vt'].max()
            })

        result_df = pd.DataFrame(results)

    # --- Case 2: Original (non-interpolated) Mode ---
    else:
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

        # Keep only valid non-At_vt samples
        result_df = result_df[result_df['At_vt'] == 0]
        result_df = result_df[['RR', 'Sub_vt1', 'Mid_vt', 'Supra_vt2']].reset_index(drop=True)
        result_df = result_df[result_df['RR'].apply(len) >= n].copy()
        result_df['RR'] = result_df['RR'].apply(lambda x: x[-n:])
        result_df = result_df[result_df['RR'].apply(lambda x: not any(pd.isna(v) for v in x))].copy()

    # Final cleanup — keep only At_vt==0 (if present)
    if 'At_vt' in result_df.columns:
        result_df = result_df[result_df.get('At_vt', 0) == 0].reset_index(drop=True)

    return result_df[['RR', 'Sub_vt1', 'Mid_vt', 'Supra_vt2']]


def create_additional_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    intervals_copy = intervals.copy()

    # Sort
    intervals_copy['class_id'] = intervals_copy[['Sub_vt1', 'Mid_vt', 'Supra_vt2']].idxmax(axis=1) # Helper for sorting
    intervals_copy = intervals_copy.sort_values('class_id').reset_index(drop=True)
    intervals_copy = intervals_copy.drop(columns=['class_id'])

    additional_intervals = []

    for i in range(len(intervals_copy) - 1):
        current_row = intervals_copy.iloc[i]
        next_row = intervals_copy.iloc[i+1]
        
        # Check if both intervals have the same labels
        current_labels = current_row[['Sub_vt1', 'Mid_vt', 'Supra_vt2']].values
        next_labels = next_row[['Sub_vt1', 'Mid_vt', 'Supra_vt2']].values
        
        # Only create additional interval if both have identical labels
        if np.array_equal(current_labels, next_labels):
            # Take last half of current interval and first half of next interval
            current_rr = current_row['RR']
            next_rr = next_row['RR']
            
            half_length = len(current_rr) // 2
            
            # Combine last half of previous with first half of next
            combined_rr = current_rr[-half_length:] + next_rr[:half_length]
            
            # Create new row with combined RR interval and same labels
            new_row = {
                'RR': combined_rr,
                'Sub_vt1': current_labels[0],
                'Mid_vt': current_labels[1], 
                'Supra_vt2': current_labels[2]
            }
            
            additional_intervals.append(new_row)

    additional_df = pd.DataFrame(additional_intervals)
    result_df = pd.concat([intervals, additional_df], ignore_index=True)

    return result_df