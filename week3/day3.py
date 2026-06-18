# Derivatives
# Measures the rate at which function changes with respect to its input
# for a function f(x), the derivative  f'(x) represents the slope of the tangent line at point x
# Role in optimization - optimizing loss function ,
#
import numpy as np
import sympy as sp

x = sp.Symbol("x")
f = sp.sin(x)
der = sp.diff(f, x)
print(der)

# Partial Derivative
# Measures how the fucntion changes with respect to one variable while keeping the other variable constant
#
# Gradients
# Vector of all partial derivative indicating the direction of steepest ascent
x, y = sp.symbols("x y")
f = x**2 + y**2
x_grad = sp.diff(f, x)
y_grad = sp.diff(f, y)
print(x_grad, y_grad)


# Gradient Descent Optimization Algorithm
# Iterative Optimization algorithm used to minimize a function
# Updates the parameters in the direction of negative gradients to find the minimum
# Update Rule: theta(0) = theta(0) - alpha(a).f(theta(0))
# theta(0): parameters of the model
# alpha(a): learning rate of the model
# Importance:
# Its used in training model, minimizing the loss function and find optimal model parameters
# Compute Derivatives of Basic function
#
x = sp.Symbol("x")
f = x**3 - 5 * x + 7
print(sp.diff(f))

# Compute Gradients
x, y = sp.symbols("x y")
f = x**2 + 3 * y**2 - 4 * x * y
x_grad = sp.diff(f, x)
y_grad = sp.diff(f, y)


# Implement Gradient Descent for linear regression
#
# Define gradient Descent function
def gradient_descent(X, Y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        predictions = np.dot(X, theta)
        errors = predictions - Y
        gradients = (1 / m) * np.dot(X.T, errors)
        theta -= learning_rate * gradients
    return theta


X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([2, 2.5, 3.5])
theta = np.array([0.1, 0.1])
learning_rate = 0.1
iterations = 1000

# Performing Gradient Descent
optimized_theta = gradient_descent(X, y, theta, learning_rate, iterations)
print(optimized_theta)

# Compute Second Order Derivatives Hessian Matrix
# Implement Gradient Descent With multiple Learning Rates and Compare Convergence Speed
# Visualize Gradient Descent process on a quadratic function
