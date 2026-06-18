"""
What is tensorflow
    Opensource library for numerical computation and machine learning
    provides tools for building and training deep learning models
What is keras
    High-level neural networks API, running on top of TensorFlow
    key features:
        User-friendly intuitive syntax for rapid prototyping
        Moudlar: Building blocks for defining layers , optimizers and loss functions
        Integration: Compatible with tensorflow for scalable deep learning tasks
Defining layers:
    Layers are the building blocks of a neural network
    They define the computation performed on the input data
    Common types of layers:

        Dense: Fully connected layer
        Conv2D: Convolutional layer for image data
        LSTM: Long Short-Term Memory layer for sequence data
        Dropout: Dropout layer for regularization
        Flatten: Flatten layer for converting 2D data to 1D
        MaxPooling2D: Max pooling layer for reducing spatial dimensions
        Activation: Activation function layer for introducing non-linearity
Building a Model:
    Keras supports two primary ways to define models
        Sequential: Linear stack of layers
        Functional: Functional API for more complex models
Compiling a model:
    Specifies:
        Optimizer: Algorithm used for updating model weights
        Loss function: Measures the difference between predicted and actual values
        Metrics: Evaluation metrics to monitor during training

Training a model:
    Fits the model to the training data: model.fit()
Evaluating a model:
    Evaluates the model on the test data: model.evaluate()
Saving and loading a model:
    Saves the model to disk: model.save()
    Loads the model from disk: keras.models.load_model()

"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

# Load mnist datasets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize data
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#
print("Training data shape: ", X_train.shape)

model = Sequential(
    [
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(28, 28, 1),
        ),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ]
)

# Display model summary
model.summary()
# compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

# Save the model
model.save("models/week9/mnist_model.h5")

# load models
loaded_model = load_model("models/week9/mnist_model.h5")

# Verify the loaded model performance
loss, accuracy = loaded_model.evaluate(X_test, y_test)
print(f"Loaded model test loss: {loss}, test accuracy: {accuracy}")
