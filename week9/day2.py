"""
Forward Propagation
Process by which input data flows throhg the layers of neural network to produce an output

    Input layer
        Accepts input features and passes them to the next layer
    Hidden layer
        Compute weighted sums of inputs , apply biases and pass the result through activation functions
    Output layer
        Produces predictions , typically using an activation function suitable for the task
Steps in forward propagation
    Compute weighted sum
        z = W.X + b
            W: weights | X: Inputs | b: bias
    Apply Activation function
        a= sigma(z)
            sigma : Activation Function
    Repeat for Each Layer
        Outputs of one layer become inputs to the next
Common Activation Function
    Sigmoid:
        Usecase: Binary classification in the output layer
        Limitation: Can suffer from vanishing gradients for large positive/ negative z
    Tanh (Hyperbolic tangent):
        Usecase: Hidden layers where zero =cantered outputs are preferred
        Limitation: Also prone to vanishing gradients
    ReLU (Rectified Linear Unit):
        Usecase: Hidden layers where zero-centered outputs are preferred
        Limitation: Can suffer from "dying ReLU" problem where neurons stop learning
    Softmax:
        Usecase: Multiclass classification in the output layer

"""

import matplotlib.pyplot as plt
import numpy as np


# define activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0, keepdims=True)


# forward pass function
def forward_pass(X, weights, biases, activation_function):
    z = np.dot(weights, X) + biases
    a = activation_function(z)
    return a


# Example inputsa
#
X = np.array([[0.5], [0.8]])
weights = np.array([[0.2, 0.4], [0.6, 0.1]])
biases = np.array([[0.1], [0.2]])

# Perform forward pass
activations = {
    "Sigmoid": sigmoid,
    "Tanh": tanh,
    "ReLU": relu,
    "Softmax": softmax,
}
for name, func in activations.items():
    output = forward_pass(X, weights, biases, func)
    print(f"{name} output: {output}")

z = np.linspace(-10, 20, 200)
plt.figure(figsize=(12, 8))
plt.plot(z, sigmoid(z), label="Sigmoid")
plt.plot(z, tanh(z), label="Tanh")
plt.plot(z, relu(z), label="ReLU")
plt.plot(z, softmax(z), label="softmax")
plt.legend()
plt.show()
