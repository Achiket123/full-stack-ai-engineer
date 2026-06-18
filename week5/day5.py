"""
Model Evaluation Metrics for Regression and classification
Model Evaluation Metrics
    1. Mean Squared Error
        Avergage square distance between predicted and actual values
        Sensitive to ouliers due to squaring of errors
    2. Mean Absolute Error
        Average absolute distance between predicted and actual values
        Presents more interpretable measures but less sensitive to outliers
    3. Root Mean Squared Error
        Provides error in the same units as the target variables
Model Evaluation for Classification
    Accuracy
        Proportion of correctly predicted instances
        Useful when the dataset is balanced
    Precision
        Fraction of positive predictions that are correct
        Important for applications for fraud detection, where false positives are costly
    Recall (Sensitivity)
        Fraction of actual positive that are correctly identified
        Useful for cases where missing positive instances are costly
    F1 Score
        Harmonic mean of precision and recall
        Useful when both false positives and false negatives are costly
Cross Validation
    Key Cross Validation Techniques
        K-fold Cross Validation
            Splits the dataset into k folds, trains the model on k-1 folds and tests on the remaining fold, while repeating this process k times
            Average of the k test set scores provides an estimate of the model's generalization performance
        Stratified K-fold Cross Validation
            Splits the dataset into k folds, ensuring that each fold has a similar distribution of target variable classes as the entire dataset
            Useful for imbalanced datasets
        Leave-One-Out Cross Validation (LOOCV)
            Splits the dataset into n folds, where n is the number of instances, and trains the model on n-1 instances and tests on the remaining instance
            Useful for small datasets or when computational resources are limited


Confusion Matrix
    The confusion matrix is a table that summarizes the performance of a classification model by comparing predicted and actual values
    Structure of a confusion matrix
                    | Predicted Positive | Predicted Negative |
    Actual Positive |       TP           |       FP          |
    Actual Negative |       FN           |       TN          |
    Key Metrics:
        True Positive Rate (TPR)
            same as recall
        False Positive Rate (FPR)
            proportion of negative instances incorrectly classified as positive
        Specificity (TNR)
            proportion of negative instances correctly classified as negative


"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split

data = load_iris()
X, y = data.data, data.target  # pyright: ignore[reportAttributeAccessIssue]
# Initialize the classifier

model = RandomForestClassifier(random_state=42)

# Perform KFold Cross Validation
Kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=Kf, scoring="accuracy")

print(f"""

    Cross Validation Scores: {cv_scores}
    Mean CV Score: {cv_scores.mean()}
    """)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train logistic regression model
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

# Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)  # pyright: ignore[reportAttributeAccessIssue]
disp.plot(cmap="Blues")
plt.show()


print(classification_report(y_test, y_pred, target_names=data.target_names))  # pyright: ignore[reportAttributeAccessIssue]
