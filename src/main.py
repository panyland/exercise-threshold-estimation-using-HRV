import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from feature_extraction import extract_hrv_features
from preprocessing import (
    remove_pre_post_periods,
    mark_thresholds,
    replace_missing_beats,
    label_rr_intervals,
    create_classification_dataset,
    create_additional_intervals
)
from plotting import (
    plot_subject_data,
    plot_confusion_matrix,
    plot_importances,
)


def load_data(measure_path, subject_path):
    data = pd.read_csv(measure_path)
    subjects = pd.read_csv(subject_path)
    return data, subjects 


def main():
    data, subjects = load_data('data/test_measure.csv', 'data/subject-info.csv')
    
    # Preprocessing
    data = remove_pre_post_periods(data)
    data = mark_thresholds(data, subjects)
    data = replace_missing_beats(data)

    data.to_csv('data/cleaned_test_measure.csv', index=False)

    plot_subject_data(data)

    data = label_rr_intervals(data)

    classification_data = create_classification_dataset(data)

    classification_data = classification_data[classification_data['RR'].apply(lambda x: len(x) > 0 and not any(pd.isna(v) for v in x))].reset_index(drop=True)
    
    # Create artificial intervals due to data scarcity
    classification_data = create_additional_intervals(classification_data)

    label_counts = {
        'Sub_vt1': classification_data['Sub_vt1'].sum(),
        'Mid_vt': classification_data['Mid_vt'].sum(),
        'Supra_vt2': classification_data['Supra_vt2'].sum()
    }
    print("Classification dataset label counts:", label_counts)
    
    # Add written labels in addition to one-hot encoding + labels for binary classification by combining labels above VT1
    classification_data['VT_label_3class'] = classification_data[['Sub_vt1','Mid_vt','Supra_vt2']].idxmax(axis=1)
    classification_data['Supra_vt1'] = (classification_data['Mid_vt'] + classification_data['Supra_vt2']).clip(0,1)

    classification_data.to_csv('data/classification_dataset.csv', index=False)

    # Extract features from each RR interval sequence
    features_df = extract_hrv_features(classification_data) 
    features_df.to_csv('data/hrv_features.csv', index=False)

    X = features_df.drop(columns=['Sub_vt1', 'Mid_vt', 'Supra_vt2', 'VT_label_3class', 'Supra_vt1']).values

    mode = '3class'  # '3class' or '2class' 

    if mode == '3class':
        y = classification_data['VT_label_3class'].values
    elif mode == '2class':
        y = classification_data['Supra_vt1'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train Random Forest classifier
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(matrix, y_test)

    importances = rf.feature_importances_
    plot_importances(importances, list(features_df.columns[:-5]))

    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    print(f"\nMean CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


if __name__ == '__main__':
    main()

# Try interpolation and frequency domain features again
# LSTM and raw zero-centered RR intervals?