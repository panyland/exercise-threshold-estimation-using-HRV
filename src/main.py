import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from feature_extraction import extract_hrv_features
from preprocessing import (
    label_rr_intervals,
    mark_thresholds,
    remove_pre_post_periods,
    replace_missing_beats,
    create_classification_dataset,
    create_additional_intervals
)
from plotting import (
    plot_confusion_matrix,
    plot_importances,
    plot_subject_data,
)


def load_data(measure_path, subject_path):
    data = pd.read_csv(measure_path)
    subjects = pd.read_csv(subject_path)
    return data, subjects 


def main():

    # ----------Preparation----------

    data, subjects = load_data('data/test_measure.csv', 'data/subject-info.csv')
    data = remove_pre_post_periods(data)
    
    data = mark_thresholds(data, subjects)
    data.to_csv('data/marked_test_measure.csv', index=False)

    data = replace_missing_beats(data)
    data.to_csv('data/cleaned_test_measure.csv', index=False)

    plot_subject_data(data)

    data = label_rr_intervals(data)
    data.to_csv('data/labeled_test_measure.csv', index=False) 

    classification_data = create_classification_dataset(data, interpolate=True, n=150) # interpolate : bool
    classification_data.to_csv('data/classification_dataset.csv', index=False)

    classification_data = classification_data[classification_data['RR'].apply(lambda x: len(x) > 0 and not any(pd.isna(v) for v in x))].reset_index(drop=True)
    
    classification_data = create_additional_intervals(classification_data)

    label_counts = {
        'Sub_vt1': classification_data['Sub_vt1'].sum(),
        'Mid_vt': classification_data['Mid_vt'].sum(),
        'Supra_vt2': classification_data['Supra_vt2'].sum()
    }
    print("Classification dataset label counts:", label_counts)
    
    # Written labels in addition to one-hot encoding + labels for binary classification
    classification_data['VT_label_3class'] = classification_data[['Sub_vt1','Mid_vt','Supra_vt2']].idxmax(axis=1)
    classification_data['Supra_vt1'] = (classification_data['Mid_vt'] + classification_data['Supra_vt2']).clip(0,1)

    # ----------Features----------
   
    features_df = extract_hrv_features(classification_data, include_freq=False) # include_freq : bool
    features_df.to_csv('data/hrv_features.csv', index=False)

    X = features_df.drop(columns=['Sub_vt1', 'Mid_vt', 'Supra_vt2', 'VT_label_3class', 'Supra_vt1']).values

    mode = '3class'  # '3class' or '2class' 

    if mode == '3class':
        y = classification_data['VT_label_3class'].values
    elif mode == '2class':
        y = classification_data['Supra_vt1'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # ----------Model training and evaluation----------

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


"""     # Hyperparameter tuning visualization (not part of main)
    test_scores = []
    estimators_range = range(1, 301, 10) # Check every 10 trees
    for n in estimators_range:
        # Train a new RF with the limited number of trees
        temp_rf = RandomForestClassifier(n_estimators=n, 
                                        max_depth=None, 
                                        class_weight='balanced', 
                                        random_state=42)
        temp_rf.fit(X_train, y_train)
        
        # Evaluate on the test set
        y_pred_temp = temp_rf.predict(X_test)
        score = cross_val_score(temp_rf, X, y, cv=5, scoring='accuracy').mean()
        test_scores.append(score)

    # 4. Plot the stabilization curve
    plt.figure(figsize=(10, 6))
    plt.plot(estimators_range, test_scores, marker='o', linestyle='-', markersize=4)
    plt.title('Model Stabilization: Test Accuracy vs. Number of Trees')
    plt.xlabel('Number of Estimators ($n_{estimators}$)')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.savefig('plots/results/model_stabilization.png') """


if __name__ == '__main__':
    main()

