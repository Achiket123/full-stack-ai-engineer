"""
Building an End - to - End Supervised Learning Project

Define the problem
    Identify the Objective
Data Preparation
    Exploratroy Data Analysis
        Understand the structure of the dataset
        Visualize the data distribution and relationships
    preprocessing
        Handle the missing values
        Scale features for algorithms like KNN
        Encode categorical variables
    Model Selection
        Choosing correct model
    Model Valuation
        Evaluate the model's performance on the test set
    Comparison
        Compare the performance of different models
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


def california_housing_linear_regression():
    data = fetch_california_housing(as_frame=True)
    df = data.frame  # pyright: ignore[reportAttributeAccessIssue]
    X = df[["MedInc", "HouseAge", "AveRooms"]]
    y = df["MedHouseVal"]

    # print(df.info())
    # print(df.describe())

    # sns.pairplot(df, vars=["MedInc", "HouseAge", "AveRooms", "MedHouseVal"])
    # plt.show()
    # print("Missing values:")
    # print(df.isnull().sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")


def telecom_churn_linear_regression():
    df = pd.read_csv("data/telecom-churn.csv")

    df = df.drop(columns=["customerID"])

    # One-hot encode categorical features

    le = LabelEncoder()
    df["Churn"] = le.fit_transform(df["Churn"])

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Train Logistic Model
    log_model = LogisticRegression(max_iter=200)
    log_model.fit(X_train, y_train)

    # Train Knn
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Predict
    y_pred_knn = knn_model.predict(X_test)
    y_pred_log = log_model.predict(X_test)

    # Evaluate
    mse_knn = mean_squared_error(y_test, y_pred_knn)
    mse_log = mean_squared_error(y_test, y_pred_log)
    print(f"KNN Mean Squared Error: {mse_knn}")
    print(f"Logistic Mean Squared Error: {mse_log}")

    print(f"""
        CLASSIFICATION REPORT
        Logistic:
        {classification_report(y_test, y_pred_log)}
        KNN:
        {classification_report(y_test, y_pred_knn)}
        """)


telecom_churn_linear_regression()
california_housing_linear_regression()
