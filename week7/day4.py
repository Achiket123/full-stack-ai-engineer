"""
XGBoost (Extreme Gradient Boosting)
Advance implementation of Gradient boosting algorithm designed for speed and performance
It introduces various enhancement that makes it faster and more efficient and capable of handling complex datasets.
Improvements:
    Speed: Added parallel computing
    Handle missing values
    Add regularization
    Custom loss functions
    Tree pruning
Key Features:
    Automatically assigns the missing values to the branch that minimizes the loss function
    Reduces the preprocessing steps for datasets with missin values.
    Regularization:
        Includes penalties for overly complex models, reducing overfitting.
        Hyperparameter :
            lambda : L1 regularization terms
            alpha : L2 regularization terms
    Parallel processing
        splits the computation to multiple cores or machines

    Key Parameters:
        Learning rate
        Controls the contribution of each tree to the model
        typical range: 0.01 - 0.3

    Number of trees:
        Determines the number of boosting rounds
        Larger values may increase the performance but may also lead to overfitting
    Tree Depth (max_depth):
        limits the depth of the trees, balancing bias and variance
        Shallower trees generalize better, deeper trees may overfit
    SubSample:
        Fraction of the training data to be used for each tree
        Typical values: 0.5 - 1.0
    Colsample_bytree:
        Fraction of the features to be used for each tree
        Typical values: 0.5 - 1.0
    Regularization parameters:
        lambda : L1 regularization terms
        alpha : L2 regularization terms

"""

import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# convert the datasets to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train xgb
#
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "eta": 0.1,  # learning rate
}
num_rounds = 100

xgb_model = xgb.train(params, dtrain, num_boost_round=num_rounds)
# Predict
y_pred = xgb_model.predict(dtest)
y_pred = [1 if p >= 0.5 else 0 for p in y_pred]

# Evaluate
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# param tuning
params = {
    "learning_rate": [0.01, 0.2],
    "n_estimators": [
        50,
        100,
        200,
    ],
    "max_depth": [
        3,
        5,
        7,
    ],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# Initialize XGBoost Classifier
xgb_cls = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_cls, param_grid=params, cv=5, scoring="accuracy"
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print(grid_search.best_params_)
print(grid_search.best_score_)

# Train gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Predict
y_pred = gb_model.predict(X_test)

# Evaluate
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
