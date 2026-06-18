"""
Introduction to cross validation
Techniques used to asses how well a model generalizes to an independent dataset
Type:
    K-Fold Cross Validation
        Splits the dataset into k folds, trains the model on k-1 folds, and validates on the remaining fold
        Repeats this process k times, each time using a different fold as the validation set
    Stratified K-Fold Cross Validation
        Splits the dataset into k folds, trains the model on k-1 folds, and validates on the remaining fold
        Repeats this process k times, each time using a different fold as the validation set
        Stratifies the data to ensure each fold has a similar distribution of classes as the entire dataset
    Leave-One-Out Cross Validation
        Uses single data point for validation and rest for training
        repeats this process for all the data points
        Computationally expensive but provides the most robust evaluation
HyperParameter Tuning
    Hyperparameter are parameters that are not learned by the model during training but are set before training, tuning these parameters is crucial for optimizing model pefromance
    Types:
        Grid Search
            Exhaustively searches through a grid of hyperparameters to find the best combined set of hp
        Random Search
            Randomly searches combinations of hyperparameters from predefined space
            More efficient than grid search when hyperparameter is large
        Importance of hyperparameter tuning
            Underfitting
            Overfitting
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/titanic.csv")

df = df[["Pclass", "Sex", "Age", "Fare", "Survived", "Embarked"]]

# Handle missing values
df.fillna({"Age": df["Age"].median()}, inplace=True)
df.fillna(
    {
        "Embarked": df["Embarked"].mode()[0],
    },
    inplace=True,
)
# df["Age"].fillna(df["Age"].median(), implace=True)
# df["Embarked"].fillna(df["Embarked"].mode()[0], implace=True)

X = df.drop(columns=["Survived"])
y = df["Survived"]

# Apply Feature Scaling and encoding
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Age", "Fare"]),
        ("cat", OneHotEncoder(), ["Pclass", "Sex", "Embarked"]),
    ]
)

X_preprocessed = preprocessor.fit_transform(X)


# Train and eval Logisting Reg
log_model = LogisticRegression()
log_scores = cross_val_score(log_model, X_preprocessed, y, cv=5, scoring="accuracy")
print(f"Logistic Regression CV Accuracy: {log_scores.mean():.4f}")

# Train and eval Random Forest
rf_model = RandomForestClassifier()
rf_scores = cross_val_score(rf_model, X_preprocessed, y, cv=5, scoring="accuracy")
print(f"Random Forest CV Accuracy: {rf_scores.mean():.4f}")

# Define hyperparameter grid for GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}

# grid search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
)

grid_search.fit(X_preprocessed, y)
print(f"Best HyperParameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
