"""
Cross Validation
It is a method to evaluate the performance of a model by partitioning the data into training and validatiing the dataset multiple times.
It helps ensure that the model's performance is consistent across different subsets of the data.
Prevents Overfitting:
    By evaluating the model on multiple subsets of the data, it can detect overfitting and prevent it from occurring.
Reliable Performance Estimate:
    Reduces the variance of performance metrics compared to a single train-test split.
Optimizes Models Selections:
    Helps in comparing and selecting the best model from a set of candidate models. or hyperparameter tuning.

Types:
    K-fold Cross Validation
        Splits the data into K subsets, uses K-1 subsets for training and 1 subset for validation.
    Stratified K-fold Cross Validation
        Ensures that each fold has a similar distribution of classes as the entire dataset.
        Useful for imbalanced datasets.
        Best for classification task with imbalanced data
    Leave-One-Out Cross Validation LOOCV
        Uses all but one data point for training and the remaining one for validation.
        Repeats the process for each data point.
        Computationally expensive
Choose K based on dataset size
    K=5, or K=10 are commonly used for large datasets.
    Use LOOCV for small datasets
Startification of Imbalanced data
    Always prefer stratified K-fold for imbalanced classification tasks to ensure fair evaluations.
Combine Hyperparameter Tuning
    Use Grid Search or Random Search to find the best hyperparameters for your model.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

df = pd.read_csv("data/credit_card.csv")
print(df["Class"].value_counts())

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the K-Fold
#
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Train
model = RandomForestClassifier(random_state=42)
scores_kf = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")

print(f"K-Fold Cross Validation Scores: {scores_kf}")
print(f"Mean Score: {scores_kf.mean():.4f}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_skf = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")

print(f"K-Fold Cross Validation Scores: {scores_skf}")
print(f"Mean Score: {scores_skf.mean():.4f}")
