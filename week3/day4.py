# INTEGRALS: Compute the Area under the curve, representing accumulating
# the definite integral lim a->b intg{ f(x)dx }
# Probability Distribution
# Cost Function
#
import numpy as np
import sympy as sp

x = sp.Symbol("x")
f = x**2
def_int = sp.integrate(f, (x, 0, 2))
print(def_int)

# Indefinete
indef_int = sp.integrate(f, (x))
print(indef_int)

# Optimization Concepts:
# Local Mini Vs Global Mini
# Local Minima
#   Has the function lower value then the neighbourhood points
# Global Minima
#   Has the absolute lowest point of the function accross it's entire function
#   In ML. we find global Minima
#
# Convex function:
#   Any Local minima is also a global minima
# Non-Convex function:
#   Most of the neural network loss function are non convex
#
# STOCHASTING GRADIENT DESCENT : SDG AND ITS VARIANTS
# AN OPTIMIZATIONS ALGORITHM THAT USES SUBSETS (MINI-BATCHES)OF THE DATA TO COMPUTE
# GRADIENTS AND UPDATE PARAMETERS
# used for faster conversions of large datasets compared to batch gradient descents
# Variants:
# Mini Batch SGD: Updates parameters using small batches instead of single values
# Momentum: adds the fraction of previous update to the current update to increase the speed
# Adam Optimizer: combines momentum with adaptive learning rate for faster conversions.
#
x = sp.Symbol("x")
f = sp.exp(-x)

indef_int = sp.integrate(f, x)
print(indef_int)

def_int = sp.integrate(f, (x, 0, sp.oo))
print(def_int)

# IMPLEMENT SGD FOR A LINEAR MODEL
np.random.seed(23)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add Bias term to x

X_b = np.c_[np.ones((100, 1)), X]


# SGD
def schostic_descent_gradient(X, y, theta, learning_rate, n_epochs):
    m = len(y)
    for epochs in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index : random_index + 1]
            yi = y[random_index : random_index + 1]
            gradients = 2 * xi.T @ (xi @ theta - yi)
            theta -= learning_rate * gradients
    return theta


theta = np.random.randn(2, 1)
learning_rate = 0.01
n_epochs = 50

theta_opt = schostic_descent_gradient(X_b, y, theta, learning_rate, n_epochs)

print(theta_opt)

"""
Additional Practice
Visualize the loss function's surface and the SGD optimization
Implement Mini-Batch SGD and compare it with vanilla SGD
Use Adam optimizer for a more complex dataset
"""
