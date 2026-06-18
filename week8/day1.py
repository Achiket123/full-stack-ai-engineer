"""
Parameters and Hyperparameters

Parameters:
    Values learned by a machine learning model during training
    Adjusted by the model during training to improve performance, minimize loss function and optimize predictions
    Eg: Co efficients in Lin-Reg, and Weights and Biases in Neural nets
Hyper Parameters:
    Settings defined before training that influences how model learns from the data
    Not learned from the data, but controls the method of learning
    Eg: Learning rate, number of epochs, batch size, Tree Depths

Importance of Hyperparameters tuning:
    Improves model performance by finding the optimal combination of hyperparameters
    Enahnce Efficiency
    Adapt to problem specific needs
Common Hyper parameters in popular models:
    Decision Tree and Random Forests
        Max Depth
        Min Samples Split
        Min Samples Leaf
        Number of Estimators
    Gradient Boosting Models
        Learning Rate
        Number of Estimators
        Max Depth
        Subsample
    Neural Networks
        Learning Rate
        Number of Epochs
        Batch Size
        Number of layers
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

print("Tuned Hyperparameters:")

model_tuned = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=5)
model_tuned.fit(X_train, y_train)
y_pred_tuned = model_tuned.predict(X_test)
print(f"Accuracy (tuned): {accuracy_score(y_test, y_pred_tuned)}")
print(classification_report(y_test, y_pred_tuned))
