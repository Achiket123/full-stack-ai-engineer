# Linear Algebra
#   Mathematical Model cap(y) = X.theta(0)
#   X: Feature Matrix (with bias term). theta(0): Parameters (weights and bias) |
#       cap(Y) predicted Value
#
# Calculas:
#   Optimization of theta(0) involves minimizing the loss function
#       J(theta(0)) = 1/2m * sum((cap(y)i - yi)^2) {
#               i = 1 to m
# }
#   Gradient Descent: Update theta(0) to minimize J(theta(0))
# nabla(J(theta(0))) = 1/m X.T.(X.theta(0) - y )
# Statistics:
#   Metrics like MSE (Mean Squared Error ) and R**2 are used to evaluate model performance
#
# Using Gradient Descent for parameter optimization
#
# Algo := theta(0) = theta(0) - alpha * nabla(J(theta(0)))
#   alpha: Learning rate
# Key steps:
#   1. Initialize theta(0)
#   2. Compute gradient: nabla(J(theta(0)))
#   3. Update theta(0): theta(0) = theta(0) - alpha * nabla(J(theta(0))) [rule]
#   4. Repeat until convergence

# Evaluating the model
#  Mean Square Error (MSE):
#   MSE = 1/m * sum((cap(y)i - yi)^2) {
#           i = 1 to m
#   }
# measures the average squared error
#
# R-squared (R^2):
#  R^2 = 1 - SS_residual / SS_total
#  SS_residual = sum((cap(y)i - yi)^2)
#  SS_total = sum((yi - mean(yi))^2)
# How well the regression model fits the data
#
# Task 1 : Implemnt the mathmatical model for linear regression
#
import numpy as np

np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# Add Bias term to feature matrix

X_b = np.c_[np.ones((100, 1)), X]

# Initialize the parameters
theta = np.random.randn(2, 1)
learning_rate = 0.1
epochs = 1000


def predict(X, theta):
    return np.dot(X, theta)


# Task 2: Use Gradient Descent to optimize the model parameters


def gradient_descent(X, y, theta, learning_rate, epochs):
    m = len(y)
    for _ in range(epochs):
        gradients = (1 / m) * np.dot(X.T, (np.dot(X, theta) - y))
        theta -= learning_rate * gradients
    return theta


# Task 3: Calculate Evaluation Metrics


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r_squared(y_true, y_pred):
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_residual / ss_total


def main():
    # Perform Gradient Descent
    theta_opt = gradient_descent(X_b, y, theta, learning_rate, epochs)
    # Predict and eval
    y_pred = predict(X_b, theta_opt)
    mse = mean_squared_error(y, y_pred)
    r2 = r_squared(y, y_pred)
    print(f"""
        OPT THETA : {theta_opt}
        MSE : {mse}
        R^2 : {r2}
        """)


main()
