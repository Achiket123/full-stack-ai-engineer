"""
Evaluation metrics for regression
Regression Metrics
    Mean Absolute Error (MAE)
        Measures the average magnitude of errors without considering their directions
        Usecase: suitable when all errors have equal importnce
    Mean Squared Error
        Measure the avg of squared difference between acutal and predicted values
        Usecase: Penalize larger error more than MAE , making it sensitive to outliers
    Root Mean Squared Error (RMSE)
        Measures the square root of the mean squared error
        Usecase: A common metric for inerpretability in real world units
    R- Squared (R2)
        Measures the proportion of variance in the dependent variable that is predictable from the independent variable
        Usecase: A common metric for model evaluation, where higher values indicate better fit
Evaluation Metrics for classification
    Accuracy
        Measures the proportion of correct predictions made by the model
        Usecase: A common metric for model evaluation, where higher values indicate better fit
    Precision
        Measures the proportion of positive predictions that are actually correct
        Usecase: Useful when the cost of false positives is high
    Recall
        Measures the proportion of actual positive instances that are correctly predicted
        Usecase: Useful when the cost of false negatives is high
    F1 Score
        Measures the harmonic mean of precision and recall
        Usecase: Useful when the cost of false positives and false negatives is equal
    Confusion Matrix
        A table that summarizes the performance of a classification model
        Usecase: Useful for understanding the model's performance across different classes
    ROC Curve
        A plot that summarizes the performance of a classification model across different thresholds
        Usecase: Useful for understanding the model's performance across different classes

When to use each metric
    In regression we use
        Mean Absolute Error (MAE) for interpreting the magnitude of errors
        Mean Squared Error (MSE) / Root Mean Squared Error (RMSE) when larger errors need greater penalization
        R- Squared (R2) ot explain variance but not as a sole performance metric
    In classification we use
        Accuracy - for balanced datasets
        Precision/Recall- for imbalanced datasets,depending on the probelem
        F1 Score for a balance evaluation of precision and recall
        ROC Curve for overall model performance evaluation in binary classification
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

data = load_iris()
X = data.data
y = (data.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Not Class 0", "Class 0"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Mtrix")
plt.show()

# Classification matrix
print(f"""
Classification Report:
    {classification_report(y_test, y_pred)}
""")


from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"""
MAE: {mae}
MSE: {mse}
R2: {r2}
""")
