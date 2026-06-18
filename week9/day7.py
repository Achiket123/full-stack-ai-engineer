import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# Preprocess the data Normalize data to [ 0,1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# One Hot encode labels y
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Define the model
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)

# Define and imporve model
improved_model = Sequential(
    [
        Conv2D(64, (5, 5), activation="relu", input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (5, 5), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ]
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
improved_model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)
improved_history = improved_model.fit(
    X_train, y_train, epochs=20, batch_size=64, validation_split=0.2
)

# Evaluate the model
improved_test_loss, improved_test_acc = improved_model.evaluate(X_test, y_test)
print("Test accuracy:", improved_test_acc)
print("Test loss:", improved_test_loss)

# Save the model
improved_model.save("data/week9/improved_model.h5")
model.save("data/week9/model.h5")

# Plot the training history
plt.plot(improved_history.history["accuracy"], label="accuracy")
plt.plot(improved_history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Plot the training and validation loss
plt.plot(improved_history.history["loss"], label="Training loss")
plt.plot(improved_history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
