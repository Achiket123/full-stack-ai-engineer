"""
Ensemble Learning
    Machine learning technique that combines the predictions of multiple models to produce a final output
    Reduces Variance
    Reduces Bias
    Improves Robustness

Types:
    Bagging (Bootstrap Aggregating)
        Train multiple models on different subsets of the data created through bootstrapping
        Combines predictions by averaging(Regression) or majority voting(Classification)
        Example: Random Forest
    Boosting
        Train models squentially where each model focuses on correcting the errors made by previous models
        Cobines predictions by weighted averaging or voting
        eg:AdaBoost, Gradient Boost, XGBoost, LightGBM
    Stacking
        Combines  predictions from multiple base models (of different types)using a meta-model  to learn how to  best combine their outputs
        Can maximize performance by combining the strengths of different models
Commonly Used Ensemble Methods:
    Random Forest
        Combines multiple decision trees using bagging
        Reduces overfitting common in  individual decision trees,
    Gradient Boosting
        Squentially builds models that minimizes errors in previous models
        Suitable for both regression and classification tasks
    AdaBoost
        Adjusts models weights based on performance
        Forces on misclassified instances
    XGBoost
        Optimized version fo gradient boosting
        Known for speed and accuracy
    Voting Classifier
        Combines predictions from multiple models using voting

"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = [
    (
        "log_reg",
        LogisticRegression(),
    ),
    (
        "knn",
        KNeighborsClassifier(),
    ),
    (
        "decision_tree",
        DecisionTreeClassifier(),
    ),
]

for model in models:
    model[1].fit(X_train, y_train)
    score = model[1].score(X_test, y_test)
    accuracy = accuracy_score(y_test, model[1].predict(X_test))
    print(f"{model[1].__class__.__name__}: {score} (accuracy: {accuracy})")

voting_clf = VotingClassifier(estimators=models, voting="hard")
voting_clf.fit(X_train, y_train)
voting_score = voting_clf.score(X_test, y_test)
voting_accuracy = accuracy_score(y_test, voting_clf.predict(X_test))
print(f"VotingClassifier: {voting_score} (accuracy: {voting_accuracy})")
