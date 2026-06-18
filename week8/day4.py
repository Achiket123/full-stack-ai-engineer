"""
Overfitting:
    Occurs when a model learns the noise in th etraining data long with the patterns,leading to poor generalization on unseen data.
    Symptoms:
        Higg training accuracy but low test accuracy
        Large diff b/w training and validation losses
Underfitting:
    Occurs when a model is too simple to capture the underlying patterns in the data, leading to poor performance on both training and test data.
    Symptoms:
        Low training and test accuracy
        High bias in predictions
Regularization:
    Introduces a penalty term to loss function during the model training to prevent overfiiting by dicouraging overly complex models.
    L1 and L2 regularization are common techniques.
    L1 Regularization (Lasso):
        Adds the absolute values of coefficients to the loss function.
        Encourages sparsity by setting some coefficient to zero, effectively selecting features
    L2 Regularization (Ridge):
        Adds the squared values of coefficients to the loss function.
        Shrinks the coefficient towards zero but does not set them to zero
    Elastic net:
        Combines L1 and L2 regularization.
        useful when there are correlated predictors and when feature selection is desired
Applications:
    Penalizes large coefficients , reducing model complexity

Handle Multicollinearity
    Ridge regularization is effective when thr predictors are highly correlated.
Feature Selection
    Lasso regularization can be used for feature selection by setting some coefficients to zero.
"""

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (No regularization): {mse}")
print(f"Model coefficients: {model.coef_}")

model = Ridge(alpha=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Ridge regularization): {mse}")
print(f"Model coefficients: {model.coef_}")


model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Lasso regularization): {mse}")
print(f"Model coefficients: {model.coef_}")
