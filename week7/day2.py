"""
Bagging
    Ensemble learning technique that combines multiple models to improve performance.
    Regression: average
    Classification: majority voting

    Reduces variance and overfitting by averaging predictions from multiple models.

Random Forest
    Ensemble learning technique that combines multiple decision trees to improve performance.
    Reduces variance and overfitting by averaging predictions from multiple models.
    Boostrap sampling
    Feature Randomness
    Prediction Aggregation

    Key Parameters:
        n_estimators: number of trees in the forest
        max_features: number of features to consider for each split
        max_features: number of features to consider for each split
        max_depth: maximum depth of each tree
        min_samples_split: minimum number of samples required to split a node
        min_samples_leaf: minimum number of samples required at each leaf node


"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Features:", data.feature_names)
print("Target:", data.target_names)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# defing hyperparameters
param_grid = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [
        None,
        10,
        20,
        30,
    ],
    "max_features": ["sqrt", "log2", None],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
