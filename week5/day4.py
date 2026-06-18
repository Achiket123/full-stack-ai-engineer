"""
Classification
    Assigns categories to data points
    Binray Classification
        Yes/No
        True/False
    Multiclass Classification
        Categories
    Multilabel Calssification
        Multiple labels per data point
    Logistic Regression
        Statistical method for binary classification
        Equation:
                applies sigmoid function to a linear equation
                P(y=1|x) = sigmoid(B0 + B1*x + B2*x2 + B3*x3 + B4*xn)
                p(y = 1|x ) = probability of positive class
        Sigmoid function:
            maps the output to a range between 0 and 1
            sigma(x) = 1/(1 + e^ (-x))
            if P(y = 1|x) >= 0.5 the prediction class 1 otherise it is class 0
        Decision Boundary:
            the boundary where the model predicts class 1 vs class 0
            Adjusting the threshold (e.g., from 0.5 to 0.3 or 0.7) can change the decision boundary
        Interpretation of Coefficients:
            B0: intercept, the value of y when all x are 0
            B1: coefficient for x1, the change in y for a unit change in x1
            B2: coefficient for x2, the change in y for a unit change in x2


"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


z = np.linspace(-10, 10, 100)
sigmoid_val = sigmoid(z)
plt.plot(z, sigmoid_val)
plt.show()

"""
Implement Logistic Regression
Logistic Rgression
    It uses the sigmoid function to model the probability of a binary outcome
    It represents the relationship between the independent variables and the probability of the dependent variable
"""

# Generate synthetic data
#
np.random.seed(42)
n_samples = 200
X = np.random.rand(n_samples, 2) * 10
y = (X[:, 0] * 1.5 + X[:, 1] > 15).astype(int)
df = pd.DataFrame(X, columns=["Age", "Salary"])
df["purchases"] = y
X_train, X_test, y_train, y_test = train_test_split(
    df[["Age", "Salary"]], df["purchases"], test_size=0.2, random_state=42
)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"""

    Accuracy: {accuracy_score(y_test, y_pred)}
    Precision: {precision_score(y_test, y_pred)}
    Recall: {recall_score(y_test, y_pred)}
    F1: {f1_score(y_test, y_pred)}
    Classification Report:
    {classification_report(y_test, y_pred)}

    """)

# plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.contourf(xx, yy, z, alpha=0.8)
plt.scatter(X_test["Age"], X_test["Salary"], c=y_test, edgecolors="k", cmap="coolwarm")  # pyright: ignore[reportArgumentType, reportCallIssue]
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Decision Boundary")

plt.show()
