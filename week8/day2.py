"""
Grid Search and Random Search

What is Grid Search:
    Method of hyperparameter tuning that systematically evaluates all possible combinations of hyperparameters within a specific grid.
    How it works:
        Define a range of values for each hyperparameter.
        Evaluate the model for each combination of hyperparameters.
        Select the combination that yields the best performance.
What is Random Search:
    ALternative method where hyperparameters combinations are sampled randomly from specified ranges.
    How it works:
        Define ranges or distributions for each hyperparameter.
        Sample random combinations of hyperparameters.
        Evaluate the model for each combination.
        Select the combination that yields the best performance.
Guidelines
    Use Grid Search when the hyperparameter space is small and well-defined.
    Use Random Search when the hyperparameter space is large or when Grid Search is computationally infeasible.

Start with broad ranges and then narrow down to find the optimal hyperparameters.
Refine with grid search when the optimal hyperparameters are known to be within a specific range.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10],
    "min_samples_split": [
        2,
        5,
        10,
    ],
}
# Initialize the grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2
)
grid_search.fit(X_train, y_train)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_grid = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_grid)
print(f"Best model accuracy: {accuracy:.2f}, best params: {grid_search.best_params_}")

# Define hyperparameter ranges for random search
param_dist = {
    "n_estimators": np.arange(50, 200, 10),
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10, 20],
}
# Initialize the random search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=20,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="accuracy",
)
random_search.fit(X_train, y_train)
# Evaluate the best model from random search
best_model_random = random_search.best_estimator_
y_pred_random = best_model_random.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)
print(
    f"Best model accuracy from random search: {accuracy_random:.2f}, best params: {random_search.best_params_}"
)
