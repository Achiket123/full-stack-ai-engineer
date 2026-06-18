"""
Boosting
    Ensemble learning technique that combines multiple weak learners to create a strong learner.
    Each weak learner is trained sequentially, with each subsequent learner correcting the errors of the previous ones.

    Difference between Boosting and Bagging:
        - Boosting trains weak learners sequentially, while Bagging trains them in parallel.
        - Boosting focuses on correcting errors of the previous learners, while Bagging focuses on reducing variance.
        - Random Forest is an example of Bagging, while Gradient Boosting is an example of Boosting.
Gradient Boosting
    boosting algorithm that builds model sequentially by minimizing a loss function using gradient descent.
    It iteratively adds weak learners to the model[decision trees], with each learner correcting the errors of the previous ones.
How it works:
    initialize models: start with a simple model, often predicting the mean of the target variable.
    compute residuals: calculate the difference between the actual and predicted values.
    fit the weak learner: train a weak learner on the residuals to predict the errors of the previous model.
    update the model: add the weak learner to the model and update the predictions.
    continue: repeat the process until the desired number of learners is reached.
key params:
    learning_rate: controls the step size of the gradient descent.
        Lower values improves the model performance by reducing overfitting but requires more iterations.
        range: 0.1 - 0.03
    n_estimators: number of weak learners to add to the model.
        represents the number of decision trees to add to the model.
        larger value improves model performance but may lead to overfitting.
    max_depth: maximum depth of the decision trees.
        controls the complexity of the model and prevents overfitting.
        range: 3 - 10
    regularization: L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train gb
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)

# predict
y_pred = gb.predict(X_test)

# evaluate
print("accuracy")
print(accuracy_score(y_test, y_pred))
print("classification report")
print(classification_report(y_test, y_pred))

parameters = {
    "learning_rate": [0.1, 0.01, 0.03],
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
}

grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

# best parameters
print("best parameters")
print(grid_search.best_params_)

# best score
print("best score")
print(grid_search.best_score_)

# test score
print("test score")
print(grid_search.score(X_test, y_test))

random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print("random forest accuracy")
print(accuracy_score(y_test, y_pred))

params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

# best parameters
print("best parameters")
print(grid_search.best_params_)

# best score
print("best score")
print(grid_search.best_score_)

# test score
print("test score")
print(grid_search.score(X_test, y_test))
