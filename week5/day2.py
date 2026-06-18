"""
Supervised Learning
Key Characterstics of supervised learning
    Labeled Data
        Supervised learning requires datasets with labeled examples
        Examples: Input|Output
    Objective
        Minimize the error between the predicted output and the actual output
    Types of Supervised Learning
        Regression: Predits continuous outputs
        Classifications: Predicts discrete outputs
    Applications
        Predicting house prices
        Classifying images

Regression Analysis:
    modeling the relationship between input features and output variables
    Linear Regression:
        Assumes a linear relationship between input features and output variables
        Equation: y = mx + b + e
        y: output variable
        m: slope
        b: intercept
        e: error term [diff b/w predicted and actual output]
    Steps:
        Fit the model to the training data
        Predict the output
        Evaluate the model using matrix, mean squared error, r-squared.
    Applications:
        Predicting sales,
        Predicting house prices,
    Cost Function:
        Linear Regression aims to minimize the error between the predicted and actual values of the the target variable using [cost function]
        It measures how far the predicted values are from the actual values
        Most common cost function is mean squared error (mse)
        J(B0, B1)= 1/m . sum((yi - (B0 + B1*xi)) ^ 2) { i: 1 --> m}
        m : number of data points
        B0, B1: parameters to be learned
        i : index of the data point
        yi : actual output
        (B0 + B1*xi) : predicted output
    Optimization With Gradient Descent:
        Iteratively updates the B0 and B1 to minimize the cost function
        Bj := Bj - alpha * d(J)/d(Bj)
        alpha: learning rate controlling the step size
        Convergence:
            Algorithm stops when the updates become very small or a predefined number of iterations is reached
"""

# Implement simple linear regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

np.random.seed(42)

X = np.random.rand(100, 1) * 100
y = 3 * X + np.random.randn(100, 1) * 2

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print(f"""
Slope: {model.coef_[0][0]}
Intercept: {
    model.intercept_[0]  # pyright: ignore[reportIndexIssue]
}
""")
# mse = mean_squared_error(Y_test, y_pred)
# r2 = r2_score()
plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, y_pred, color="red")
plt.show()

# Evaluate the model
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)
print(f"""
MSE: {mse} [lower is better]
R2: {r2} [close to 1 is better]
""")
