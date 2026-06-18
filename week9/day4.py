"""
What is Gradient Descent
    Optimization algorithm used to minimize the loss function by iterativley adjusting the model 's parameters in the direction of the negative gradient
    variants of gradient descent include:
        - Batch Gradient Descent
            uses the entire dataset to compute the gradient
            computationally expensive but stable
        - Stochastic Gradient Descent
            uses a single data point to compute the gradient
            faster but less stable
        - Mini-Batch Gradient Descent
            uses a subset of the dataset to compute the gradient
            a balance between batch and stochastic gradient descent

Advance optimizations techniques
    Adagrad:
        adaptive learning rate scaling inversely with the sum of gradients squared
    RMSprop:
        modifies adagrad by using an exponentially weighted moving average squared gradients
    Adam (Adaptive Moment Estimation):
        adaptive learning rate scaling based on the first and second moments of the gradients

Learning Rate :
    Determines the step size for parameter updates
Choosing the Right Optimizer
    SGD: works well for simple convex problems
    Adam: Generally performs well across tasks
    RMSprop: often preferred for RNNs and squence based tasks
"""

# import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=no info, 2=no warn, 3=no error
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# import matplotlib.pyplot as plt

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# import tensorflow as tf
# import torch
# import torch.nn as nn
# import torch.optim as optim

# # Gernaate some data
# np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# # plt.scatter(X, y, color="blue")

# # plt.title("Generated Dataset")

# # plt.xlabel("X")
# # plt.ylabel("y")

# # plt.grid()

# # plt.show()

# # Initialize params
# n = 100
# theta = np.random.rand(2, 1)
# learning_rate = 0.1
# iterations = 1000
# # Add bias term to X
# X_b = np.c_[np.ones((n, 1)), X]

# # Gradient Descent
# for iter in range(iterations):
#     gradients = 2 / n * X_b.T.dot(X_b.dot(theta) - y)
#     theta -= learning_rate * gradients

# print("Optmized parameters:", theta)
# # # Plot the results

# X_tensor = tf.constant(X, dtype=tf.float32)
# y_tensor = tf.constant(y, dtype=tf.float32)


# # define model
# class LinearModel(tf.Module):
#     def __init__(self):
#         self.weights = tf.Variable(tf.random.normal([1]))
#         self.bias = tf.Variable(tf.random.normal([1]))

#     def __call__(self, X):
#         return self.weights * X + self.bias


# # Define loss
# def mse_loss(y_true, y_pred):
#     return tf.reduce_mean(tf.square(y_true - y_pred))


# # Train with SGD
# model = LinearModel()
# optimizer = tf.optimizers.SGD(learning_rate=0.1)

# # for epoch in range(1000):
# #     with tf.GradientTape() as tape:
# #         y_pred = model(X_tensor)
# #         loss = mse_loss(y_tensor, y_pred)
# #     gradients = tape.gradient(loss, [model.weights, model.bias])
# #     optimizer.apply_gradients(zip(gradients, [model.weights, model.bias]))
# #     if epoch % 10 == 0:
# #         print(f"Epoch {epoch}, Loss: {loss.numpy()}")

# device = torch.device("cpu")
# # Prepare data
# X_torch = torch.tensor(X, dtype=torch.float32).to(device)
# y_torch = torch.tensor(y, dtype=torch.float32).to(device)

# # Define the model


# # define model
# class LinearModelTorch(nn.Module):
#     def __init__(self):
#         super(LinearModelTorch, self).__init__()
#         self.linear = nn.Linear(1, 1)

#     def forward(self, x):
#         return self.linear(x)


# model_torch = LinearModelTorch().to(device)

# # define loss and optimizer
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model_torch.parameters(), lr=0.1)

# for epoch in range(100):
#     optimizer.zero_grad()
#     outputs = model_torch(X_torch)
#     loss = criterion(outputs, y_torch)
#     loss.backward()
#     optimizer.step()
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item()}")


# ... (rest of your numpy gradient descent code) ...

# Remove the TF LinearModel class, X_tensor, y_tensor, tf.optimizers.SGD entirely

device = torch.device("cpu")
X_torch = torch.tensor(X, dtype=torch.float32).to(device)
y_torch = torch.tensor(y, dtype=torch.float32).to(device)


class LinearModelTorch(nn.Module):
    def __init__(self):
        super(LinearModelTorch, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model_torch = LinearModelTorch().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_torch.parameters(), lr=0.1)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model_torch(X_torch)
    loss = criterion(outputs, y_torch)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
