import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.svm import SVC

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
}

# Initilize the grid search
gridCV = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
)
gridCV.fit(X_train, y_train)

# Print the best parameters and score
print(f"Best parameters Grid: {gridCV.best_params_}")
print(f"Best score Grid: {gridCV.best_score_}")


best_grid_model = gridCV.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)
print(f"Accuracy Grid: {accuracy_score(y_test, y_pred_grid)}")
print(classification_report(y_test, y_pred_grid))

# define paramater dist
param_dist = {
    "C": np.logspace(-3, 3, 7),
    "gamma": ["scale", "auto"],
    "kernel": ["rbf", "linear", "poly", "sigmoid"],
}

# Init random search cv
randomCV = RandomizedSearchCV(
    estimator=SVC(random_state=42),
    param_distributions=param_dist,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
)
randomCV.fit(X_train, y_train)

# Print the best parameters and score
print(f"Best parameters Random: {randomCV.best_params_}")
print(f"Best score Random: {randomCV.best_score_}")

best_random_model = randomCV.best_estimator_
y_pred_random = best_random_model.predict(X_test)
print(f"Accuracy Random: {accuracy_score(y_test, y_pred_random)}")
print(classification_report(y_test, y_pred_random))
