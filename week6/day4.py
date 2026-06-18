"""
Feature Selection techniuqes
Process of identifying and retaining the most relevant features (input variables) in a dataset while discarding irrelevant or redundant ones
Improves model performance, reduces overfitting, and makes the model more interpretable
High Diemensional data
Correlated features
complexity
Techniques :
    Filter Method :
        Evaluate the relevance of features by analyzing their statistical properties in relation to the target variable
        Examples: Correlation | Mutual Information
        When to use: Quick evaluation of features before training a model
    Wrapper Method:
        Iteratively selects features by training an evaluating a model
        Eg: Forward selection | Backward selection
        When to use: useful when feature interactions are important but computationly expensive
    Embedded method:
        Perform feature selection as parrt of the model training process
        Eg: Lasso Regression | Tree -based models
        When to use: effective when training a tree-based models or regularized regression
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import mutual_info_regression

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)

df["target"] = data.target
print(df.head())
print(df.info())

# correlation_matrix = df.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.show()

# select features with high correlations witht target
#
# correlated_features = correlation_matrix["target"].sort_values(ascending=False)
# print(f"""

# Features most correlated with target
# {correlated_features}
# """)

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Calculate mutual info
mutual_info = mutual_info_regression(X, y)

mi_df = pd.DataFrame({"Feature": X.columns, "Mutual Info": mutual_info})
mi_df = mi_df.sort_values(by="Mutual Info", ascending=False)

print("Mutual Information Scores: ")
print(mi_df)

import numpy as np
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X, y)

feature_importances = model.feature_importances_
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances})
importance_df = importance_df.sort_values(by="Importance", ascending=True)

print("Feature Importances: ")
print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.title("Feature Importances")
plt.show()
