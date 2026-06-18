"""
Deep Learning
    It is a subset of machine learning that uses artificial neural networks ANN with multiple layers deep architectures to model and learn complex patterns in data.
    Key Features:
        Automaticall extract the relevant features from the raw data eliminating the need for manual feature engineering.
Machine Learning Vs Deep Learning
    Machine Learning:
        Uses traditional algorithms and feature engineering to learn from data.
        Requires manual feature engineering.
        Less capable of learning complex patterns than deep learning.
    Deep Learning:
        Uses deep neural networks with multiple layers to learn complex patterns in data.
        Can automatically learn relevant features from raw data without manual feature engineering.
        Can learn from large amounts of unlabeled data through unsupervised learning.
Overview of ANN
Structure of ANN
    Input Layer:
        accepts input data features
    Hidden Layers:
        performs computations to extract patterns
    output layers:
        produces predictions or classifications

Key Components:
    neurons:
        basic units of computation that takes input, applies weights, and produces output using an activation function
    Weights and biases:
        weights determine the importance of each inputs
        Biases shifts the output of the activation function
    activation function:
        Add non-linearity to the model (e.g. sigmoid, ReLU, Tanh)
How it works:
    Forward Propagation:
        Data flows through the network  to generate  predictions
    Loss Calculation:
        Compare the predictions with actual lable to compute the error
    Backpropagation:
        Computes the gradient of the loss with respect to each weight and bias, and updates them to minimize the loss
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import torch.nn as nn
from tensorflow.keras.datasets import cifar10, mnist

(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()

print(X_train_mnist.shape, y_train_mnist.shape)
print(X_test_mnist.shape, y_test_mnist.shape)

(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar10.load_data()

print(X_train_cifar.shape, y_train_cifar.shape)
print(X_test_cifar.shape, y_test_cifar.shape)
# Basic dense layer
layer = tf.keras.layers.Dense(units=10, activation="relu")

print(f"Layer output shape: {layer}")


# define a basic dense layer in pytorch

layer = nn.Linear(in_features=10, out_features=5)
print(f"Layer output shape: {layer}")

# Visualize the mnist
plt.imshow(X_train_mnist[0], cmap="gray")
plt.title(f"Label: {y_train_mnist[0]}")
plt.show()

plt.imshow(X_train_cifar[0])
plt.title(f"Label: {y_train_cifar[0]}")
plt.show()
