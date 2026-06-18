"""
Datascaling and normalization
    Preprocssing techniques used to perform transform numerical feautures to a common range or distribution

    Improves Algorithm performance
    ensures fair comparisons
    stabilizes training
Methods:
    Min-Max Scaling
        transforms features to range [0,1]
        ensures all feature values are within the same range
        usecases kNN or neural networks
        limitations sensitive to outliers as extreme values can distort the scale
    Standardization Z score scaling
        centers the data around zero and scales it to have standard deviation of 1
        Ensures a standard distribution for each feature
        Usecases SVM , logistic regression and PCA
        Handles outliers well by keeping them within the standard deviation range

Algorithms that require scalings
    Distance based algorithms
        Knn, SVM, K-Means
    Gradient Based Models
        Logistic Regression, Neural Networks , linear regression
Algorithms less sensitive to scaling
    tree-based models
        Random Forest, Gradient Boosting, Decision Trees, XGBoost
"""

import pandas as pd
from sklearn.datasets import load_diabetes, load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Display dataset info
print(X.info())

print(data.target_filename)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"Accuracy without scaling: {accuracy_score(y_test, y_pred)}")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train classifier on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = knn_scaled.predict(X_test_scaled)

print(f"Accuracy with scaling: {accuracy_score(y_test_scaled, y_pred_scaled)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = knn_scaled.predict(X_test_scaled)

print(f"Accuracy with StandardScaler: {accuracy_score(y_test_scaled, y_pred_scaled)}")
