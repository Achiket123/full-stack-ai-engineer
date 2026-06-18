"""
Loss Functions:
    Quantify the difference between the predicted output of a model and the actual target values
    Guide the training process by providing a metric to minimize during optimization
    Role in neural network
        Error Measurement
            Evaluate the accuracy of predictions
        Feedback for optimizations
            Provide gradients for weight updates via backpropagation

Common Types of Loss Functions:
    Mean Squared Error (MSE)
        Commonly used for regression tasks
        Penalizes large errors quadratically more heavily than smaller ones
    Cross -Entropy
        Commonly used for classification tasks
        Penalizes incorrect predictions more heavily than correct ones
        Used in conjunction with softmax activation for multi-class classification

Back Propagation:
    Process of computing gradients for each weight and bias in a neural network, enabling optimization algorithms like gradient descent to minimize the loss function
    Steps:
        Forward Pass:
            Compute the output of the network given the input
        Backward Pass:
            Compute the gradients of the loss function with respect to each weight and bias
            Propagate these gradients backward through the network
        Update:
            Use the computed gradients to update the weights and biases via gradient descent
        Repeat:
            Iterate over the dataset multiple times, updating weights and biases after each pass
    Key Concepts:
        Gradients: The rate of change of the loss with respect to a parameter
        Gradient Descent: The optimization algorithm used to update weights and biases

"""

import matplotlib.pyplot as plt
import numpy as np


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Binary cross-entropy loss
def binary_cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Clip to avoid log(0) errors
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Example
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.2, 0.8, 0.7])
print("MSE Loss:", mse_loss(y_true, y_pred))
print("Binary Cross-Entropy Loss:", binary_cross_entropy_loss(y_true, y_pred))


# Derivative of mse loss
#
def mse_gradient(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)


# derivative BCE loss
def bce_gradient(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))


# Calculate the gradients
grad_mse = mse_gradient(y_true, y_pred)
grad_bce = bce_gradient(y_true, y_pred)
print("MSE Gradient:", grad_mse)
print("BCE Gradient:", grad_bce)

# Define predictions and true lables
predictions = np.linspace(0, 1, 100)
true_labels = 1
mse_losses = [(true_labels - p) ** 2 for p in predictions]
bse_losses = [
    -true_labels * np.log(max(p, 1e-15)) - (1 - true_labels) * np.log(max(1 - p, 1e-15))
    for p in predictions
]
plt.figure(figsize=(10, 5))
plt.plot(predictions, mse_losses, label="MSE Loss")
plt.plot(predictions, bse_losses, label="BCE Loss")
plt.xlabel("Prediction")
plt.ylabel("Loss")
plt.legend()
plt.show()
