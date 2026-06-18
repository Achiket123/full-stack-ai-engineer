"""
Introduction to Bayesian Optimization
What is Bayesian Optimization?
    It is an advance method for hyperparmeter tuning that balances  exploration (searching new regions) and exploitation (refining promising regions)
    Uses a probabilistic model to balance exploration and exploitation for optimial hyperparameters
    How it works:
        Surrogate Models:
            Builds a probabilistic model (Guassian process) of the objective fn based on prior evals
        Acquisition Functions:
            Balances the exploration and exploitation by choosing the next hyperparameters to evaluate based on predicted performance and uncertainity
        Iterative Refinement:
            Updaets the surrogate model after each evaluation , refining the search
    Why Use:
        Efficient for high dimensional and expensive to evaluate function
        reduces the numenr of evaluations required to find new aoptimial hyperparmeters
Popular Libs:
    Hyperopt
        Simplifies the Bayesion optimization for hyperparameter tuning
        It works with fmin to minimize the objective functions over a parameter space
    Scikit-Optimizes
    Optuna:
        Flexible and user friendly library for hyperparameter optimization
        Supports dynamic search spaces and pruning of unpromising trials

Exploration vs Exploitation:

    Exploration:
        Focuses on searching new regions of the hyperparameter space
        Useful for identifying new areas of high potential
    Exploitation:
        Focuses on refining the search around regions with high peroformance
        Useful for fine-tuning near optimal regions
Advantages:
    Balances these approaches using the acquision function to minimize unnecessary evaluations while improving results

"""

import optuna
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
print(f"Test Data Shape: {X_test.shape}")

# train a baseline XGBoost Model
baseline_model = XGBClassifier(eval_metric="logloss", random_state=42)
baseline_model.fit(X_train, y_train)

# evaluate the baseline model
y_pred = baseline_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Baseline Model Accuracy: {accuracy:.4f}")


# Define the objective function for Optunadef objective(trial):
#
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 100),
        "learning_rate": trial.suggest_int("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lamda": trial.suggest_float("reg_lambda", 0.0, 10.0),
    }
    model = XGBClassifier(eval_metric="logloss", random_state=42, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


# Create an Optuna Study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# ?Best Hyper parameters are
print("Best Hyperparameters: ", study.best_params)
print("Best Accuracy: ", study.best_value)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
}

# Train XGBOOST Grid Search
grid_search = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    param_grid=param_grid,
    cv=3,
    verbose=1,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

# Best Hyper parameters are
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Accuracy: ", grid_search.best_score_)

# Train XGBOOST Randomized Search
param_distributions = {
    "n_estimators": [50, 100, 200, 300, 400],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
}

random_search = RandomizedSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    param_distributions=param_distributions,
    cv=3,
    verbose=1,
    n_jobs=-1,
    n_iter=50,
)
random_search.fit(X_train, y_train)

# Best Hyper parameters are
print("Best Hyperparameters: ", random_search.best_params_)
print("Best Accuracy: ", random_search.best_score_)


# Train XGBOOST with Random Search
random_search = RandomizedSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    param_distributions=param_distributions,
    cv=3,
    verbose=1,
    n_jobs=-1,
    n_iter=50,
    scoring="accuracy",
    random_state=42,
)
random_search.fit(X_train, y_train)

# Best Hyper parameters are
print("Best Hyperparameters: ", random_search.best_params_)
print("Best Accuracy: ", random_search.best_score_)
