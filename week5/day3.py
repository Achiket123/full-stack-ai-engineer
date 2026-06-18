# pyright: ignore[reportUnusedImport]
"""
Polynomial Regression
It is an extension of linear regression that allows for polynomial relationships that models non-linear relationships by introducing higher-order terms of the input features.
What is polynomial Regression?
In a typical linear regression model,
we have linear terms,
in polynomial regression we have plynomial terms
y = B0 + B1x + B2x^2 + ... + Bnx^n + e
allows to capture non linear relationships

Steps:
    Transform the input features to polynomial features
        Use PolynomialFeatures to transform the input features to polynomial terms
    Fit a linear regression model on the transformed features
    Evaluate the model using metrics like mean squared error and R-squared
"""

import matplotlib.pyplot as plt
import numpy as np  # pyright: ignore[reportUnusedImport]
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_squared_error,
    r2_score,  # pyright: ignore[reportUnusedImport]
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# np.random.seed(42)

# X = np.random.rand(100, 1) * 10
# Y = 3 * X**2 + 2 * X + np.random.randn(100, 1) * 5
# X_train, X_test, y_train, y_test = train_test_split(
#     X, Y, random_state=42, test_size=0.2
# )

# poly_features = PolynomialFeatures(
#     degree=2,
#     include_bias=False,
# )
# X_poly = poly_features.fit_transform(X)

# # Fit the polynomial reg
# model = LinearRegression()
# model.fit(X_poly, Y)
# Y_pred = model.predict(X_poly)

# plt.scatter(X, Y, color="blue")
# plt.scatter(X, Y_pred, color="red")

# plt.show()

# # Evaluate
# mse = mean_squared_error(Y, Y_pred)
# r2 = r2_score(Y, Y_pred)
# print(f"MSE: {mse}, R2: {r2}")


"""
Introduction to Regularization techniques : Lasso and Ridge Regression
What is Regularization?
    Technique used to prevent overfitting by adding a penalty term to the cost function of a regression model
Types of regularization:
    Ridge Regularization (L2 Regularization)
    Adds the sum of the squared coefficients to the cost function
        J(B) = 1/n * Σ(y_i - (B0 + B1x_i + B2x_i^2 + ... + Bnx_i^n))^2 + ( λ * Σ(B_j^2))
        lambda: regularization strength, controls the strength of the penalty
    Lasso Regularization (L1 Regularization)
        adds the sum of the absolute values of the coefficients to the cost function
        J(B) = 1/n * Σ(y_i - (B0 + B1x_i + B2x_i^2 + ... + Bnx_i^n))^2 + ( λ * Σ(|B_j|))
        lambda: regularization strength, controls the strength of the penalty

    Key Differences:
        Ridge: L2 regularization, adds the sum of the squared coefficients to the cost function
        Lasso: L1 regularization, adds the sum of the absolute values of the coefficients to the cost function
        Ridge shrinks the coefficients but does not eliminate them
        Lasso can shrink some coefficients to zero, removing irrelevant features
Avoiding Overfitting with Regularization:
    Regularization reduces the risk of overfitting by controlling the complexity of the model
    A high lambda value can lead to underfitting, while a low lambda value can lead to overfitting
    The optimal lambda value is often found through cross-validation
"""


# # Ridge regression
# ridge = Ridge(alpha=1)
# ridge.fit(X_train, y_train)
# y_pred_r = ridge.predict(X_test)


# # Lasso regression
# lasso = Lasso(alpha=1)
# lasso.fit(X_train, y_train)
# y_pred_l = lasso.predict(X_test)

# plt.scatter(X_test, y_test, color="blue", label="Actual")
# plt.scatter(X_test, y_pred_r, color="red", label="Ridge")
# plt.scatter(X_test, y_pred_l, color="green", label="Lasso")
# plt.legend()
# plt.show()

# print(f"Ridge MSE: {mean_squared_error(y_test, y_pred_r)}")
# print(f"Lasso MSE: {mean_squared_error(y_test, y_pred_l)}")


data = fetch_california_housing(as_frame=True)
df = data.frame  # pyright: ignore[reportAttributeAccessIssue]
X = df[["MedInc"]]
y = df[["MedHouseVal"]]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)
# Fit Polynomial Regression
model = LinearRegression()


model.fit(X_poly, y)

y_pred = model.predict(X_poly)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Actual")
plt.scatter(X, y_pred, color="red", label="Polynomial")
plt.legend()
plt.show()

# Performnce
#
print(f"Polynomial MSE: {mean_squared_error(y, y_pred)}")


# Lasso and Ridge
ridge = Ridge(alpha=120)
ridge.fit(X_train, y_train)
y_pred_r = ridge.predict(X_test)

print(f"Ridge MSE: {mean_squared_error(y_test, y_pred_r)}")

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_l = lasso.predict(X_test)

print(f"Lasso MSE: {mean_squared_error(y_test, y_pred_l)}")

plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], y_test, color="blue", label="Actual")  # pyright: ignore[reportCallIssue, reportArgumentType]
plt.scatter(X_test[:, 0], y_pred_r, color="red", label="Ridge")  # pyright: ignore[reportCallIssue, reportArgumentType]
plt.scatter(X_test[:, 0], y_pred_l, color="green", label="Lasso")  # pyright: ignore[reportCallIssue, reportArgumentType]
plt.legend()
plt.show()

"""
Vary Regularization Parameters:
    Experiment with different values of a (e.g., 0.1, 1, 10) for Ridge and Lasso regression
    Observe how the model's coefficients and performance change
Use Multiple Features:
    Include more features (e.g., HouseAge, Ave Rooms) and observe the impact on model performance
Feature Importance with Lasso:
    Use Lasso regression to perform feature selection and identify the most relevant predictors
"""
