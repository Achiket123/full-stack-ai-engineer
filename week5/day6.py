"""
KNN Algorithm
    Instance Based Learning
        Does not require a training phase instead directly uses the training data to make predictions
        Distance Metric
            Uses Euclidean distance to measure the similarity between instances
            Uses Manhattan distance to measure the similarity between instances
        Classification
            Majority class among the k nearest neighbors
        Regression
            Average value of the k nearest neighbors
    Steps:
        Feature Scaling
        Calculate Distances
        Identify K nearest using sorting

        Make Predictions
            Classification
            Regression
    Choosing the optimal value of k
        Small k - High variance, low bias
        Large k - Low variance, high bias
    common starting point = k = root(n) , where n is the number of training samples

    Computationally expensive
        Predictions require distances computation for all training samples
    Feature Scaling Dependence
        Requires features scaling to avoid feature dominance
    Not Robust to Outliers
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

data = load_iris()
X = data.data  # pyright: ignore[reportAttributeAccessIssue]
y = data.target  # pyright: ignore[reportAttributeAccessIssue]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)
logreg = LogisticRegression()


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


logreg.fit(X_train, y_train)
# Predict
y_pred_lr = logreg.predict(X_test)
accuracylr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression accuracy: {accuracylr}")

# Experiment with different values of k
k = 5
# Initialize KNN
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"k={k}, accuracy={accuracy}")


print("Classification Report")
print("KNN:")
print(classification_report(y_test, y_pred))

print("Logistic Regression: ")
print(classification_report(y_test, y_pred_lr))
