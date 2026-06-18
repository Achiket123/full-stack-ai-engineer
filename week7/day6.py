"""
Handling Imbalanced Data
What is Imbalanced Data?
    One class has significantly more samples than the other class.
Challenges:
    Biased towards the mahority class due to its frequency
    Misleading Evaluation metics
    limited information for minority class
    Insufficient samples for the minority class can lead to underfitting
Applications:
    Fraud detection
    Medical diagnosis
Techniques to handle imbalanced data:
    Oversampling
        Increase the number of samples in the minority class, duplicating existing samples or generating synthetic ones
    Undersampling
        Reduce the number of samples in the majority class, either by removing samples or generating synthetic ones
    Algorithms:
        Class weighting
            Assign weights to classes during training to balance their influence
            higher weight for minority class to compensate for its underrepresentation
        Anomaly Detection Models
            Treat the minority class as an anomalies, focusing the model on detecting them rather than the majority class
        Evaluation metrics:
            F1 Score
                harmonic mean of precision and recall , focusing on both false positives and false negatives
            ROC-AUC
                Area under the Receiver Operating Characteristic curve, measures the model's ability to distinguish between classes
                Measures the model's ability to distinguish between classes
            Precision-Recall Curve
                Plots precision and recall at different thresholds, showing the trade-off between false positives and false negatives
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/credit_card.csv")

print(df.head())
print(df.info())
print(df["Class"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    df.drop("Class", axis=1), df["Class"], test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))


print("ROC-AUC:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))  # pyright: ignore[reportCallIssue, reportArgumentType]

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)  # pyright: ignore[reportCallIssue, reportArgumentType, reportAssignmentType]

# Display the new class distribution
print("Class Distribution after SMOTE")
print(pd.Series(y_train_resampled).value_counts())

# Train random forest on resampled
rf_model_smote = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_model_smote.fit(X_train_resampled, y_train_resampled)

print("Classification Report after SMOTE")
y_pred_smote = rf_model_smote.predict(X_test)

print(classification_report(y_test, y_pred_smote))
print("ROC-AUC:", roc_auc_score(y_test, rf_model_smote.predict_proba(X_test)[:, 1]))  # pyright: ignore[reportCallIssue, reportArgumentType]
