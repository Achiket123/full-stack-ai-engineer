"""
Machine Learning basics and terminology

Machine Learning : Handles large datasets and extracts meaningful insights which are helpful humans
    types:
        Supervised Learning:
            Trains on labeled data (input-output pairs), learns to predict outputs for new inputs.
            Eg: classification, regression
            Requires labeled data, accuracy depends on the quality of the data.
        Unsupervised Learning:
            Trains on unlabeled data, learns to find patterns and relationships in the data.
            Eg: clustering, dimensionality reduction
            No labeled data required, learns from the data itself.
        Reinforcement Learning:
            an agent interacts with an environment and learns by trial and error to maximize a cumulative reward.
            Eg: game playing, robot control
            The agent learns to take actions that maximize the reward over time.
            Goal oriented, sequential decision making.
    Key Concepts of Machine Learning:
        Features: the input variables used to train the model.
        Target: the output variable to predict by model.
        Training and testing dataset: the data used to train and evaluate the model
        80% and 20%
    Overfitting: the model learns the training data too well, and does not generalize well
    Underfitting: the model does not learn enough from the data, and does not generalize well
    Bias-Variance Tradeoff:
        Bias: the error introduced by assuming a simplified model
        Variance: the error introduced by the model's sensitivity to small changes in training data
        Goal: balance bias and variance to achieve optimal performance
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# Define features and target variables
#
df = pd.read_csv("data/tips.csv")
features = df[["total_bill", "size"]]
target = df["tip"]

print(features.head())
print(target.head())
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
)

print(X_train.shape)  # pyright: ignore[reportAttributeAccessIssue]
print(X_test.shape)  # pyright: ignore[reportAttributeAccessIssue]

# Visualize the data
sns.pairplot(
    df,
    x_vars=["total_bill", "size"],
    y_vars=["tip"],
    height=5,
    aspect=0.8,
    kind="scatter",
)
plt.title("Feature vs Target reln")
plt.show()
