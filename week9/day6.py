"""
Intro to pytorch: building neural networks with pytorch
What is pytorch?
    PyTorch is an open-source machine learning library for Python, developed by Facebook's AI Research team.
Core Components

    Tensors: multi-dimensional arrays that can be used to represent data.
    Autograd: a system for automatic differentiation, used to compute gradients for backpropagation.
    Neural Networks torch.nn: a collection of layers that can be used to build and train neural networks.


Building a Neural Network

    Define the model
        Use torch.nn.Module to create a neural network with layers and forward propagation
    Define loss function
        Use built in loss functions like cross entropy loss
    Define optimizers
        Use torch.optim to define optimization algorithms like stochastic gradient descent (SGD)

Train the model
    Use the defined model, loss function, and optimizer to train the neural network on a dataset
Evaluate the model
    Use the trained model to make predictions on a test dataset and evaluate its performance
Save the model
    Use torch.save to save the trained model for future use

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define transformations
transformation = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
# Load Datasets]
train_datasets = datasets.MNIST(
    root="data", train=True, transform=transformation, download=True
)
test_datasets = datasets.MNIST(
    root="data", train=False, transform=transformation, download=True
)
# Create Dataloaders
train_loader = DataLoader(train_datasets, batch_size=32, shuffle=True)
test_loader = DataLoader(test_datasets, batch_size=32, shuffle=True)

print(
    f"Training data size: {len(train_datasets)}, Test data size: {len(test_datasets)}"
)


# define the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = NeuralNetwork()

print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Zero gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")


train_model(model, train_loader, criterion, optimizer, epochs=5)


# Evaluation
def evaluate_model(
    model,
    test_loader,
):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%")


evaluate_model(model, test_loader)

# save the model
torch.save(model.state_dict(), "models/week9/mnist_model.pth")

# Load the model
loaded_model = NeuralNetwork()
loaded_model.load_state_dict(torch.load("models/week9/mnist_model.pth"))

# Verify the model
evaluate_model(loaded_model, test_loader)
# # update optimizer with a new learning rate
# optimizer = torch.optim.Adam(loaded_model.parameters(), lr=0.0001)
# train_model(loaded_model, train_loader, criterion, optimizer, epochs=5)
# evaluate_model(loaded_model, test_loader)
