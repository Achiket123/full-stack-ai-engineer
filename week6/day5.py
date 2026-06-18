"""
Feature Creation
    Feature Creation involves deriving new features form existing ones to enhance model's ability to capture important patterns in the data
    Examples of feature creations:
        Date-Time features
        Interaction features
        Aggregations features
    Importance:
        adds domain knowledge tothe datasets
        captures hidden patterns and trends not evident in the original features
Feature Transformation
    Feature transformation modifies existing features to better suit the learning algorithm
    Common transformations:
        logarithmic transformation
            reduces skewness in highly skewed distributions
        Square-Root transformation
            moderately reduces skewness often used for count data
        Polynomial transformation
            Adds higher-order terms (x2 , x3 ) to capture the non-linear relations
    Importance

        Enhances the models ability to learn complex patterns
        Improves model performance by providing more informative features

"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("data/bike_sharing_daily.csv")

print(df.info())
print(df.head())
print(df.describe())

df["dteday"] = pd.to_datetime(df["dteday"])

df["day_of_week"] = df["dteday"].dt.day_name()
df["month"] = df["dteday"].dt.month
df["year"] = df["dteday"].dt.year

print(df.info())

# Apply polynomial transformation

X = df[["temp"]]
y = df["cnt"]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# print(f"""
# {pd.DataFrame(X_poly, columns=["temp", "temp^2"]).head()}
#     """)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

model_original = LinearRegression()
model_original.fit(X_train, y_train)

y_pred = model_original.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)
y_pred_poly = model_poly.predict(X_test_poly)
mse_poly = mean_squared_error(y_test_poly, y_pred_poly)
print(f"Mean Squared Error (Poly): {mse_poly}")
