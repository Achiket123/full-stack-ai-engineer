"""
Comprehensive Model Optimization
    Data Preprocessing
    Feature Engineering
    Regularization
    Cross-Validation
    Hyperparameter Tuning
Evaluating and Interpreting Model Performance
    Performance Metrics
        Classification:
            Accuracy, precision, recall, F1-score, ROC-AUC
        Regression:
            Mean Squared Error, Mean Absolute Error, R-squared
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("data/telecom-churn.csv")
df.info()
print(df.describe())
print("Class Distribution")
print(df["Churn"].value_counts())

# Handle missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna({"TotalCharges": df["TotalCharges"].median()}, inplace=True)

label_encoder = LabelEncoder()

for col in df.select_dtypes(include=["object"]).columns:
    if col != "Churn":
        df[col] = label_encoder.fit_transform(df[col])

# Encode target variable
df["Churn"] = label_encoder.fit_transform(df["Churn"])

# Scale Numerical Features
numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Feature and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.2,
)
# Train Initial Model

model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test)
print("Accuracy:Init: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Hyperparameter Tuning
param_dist = {
    "n_estimators": np.arange(50, 200, 10),
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf": np.arange(1, 2, 4),
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    random_state=42,
    n_jobs=-1,
    scoring="accuracy",
)

# Perform Randomized Search
random_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters: ", random_search.best_params_)
print("Best Score: ", random_search.best_score_)

# Evaluate on Test Set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Accuracy: Best: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Accuracy: Train: ", accuracy_score(y_train, best_model.predict(X_train)))
print("Best Score: ", random_search.best_score_)
