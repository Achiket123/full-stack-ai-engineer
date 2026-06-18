"""
Feature Engineering
Process of transforming raw data into meaningful inputs for machine learning models
Importance:
    Imporves Model accuracy
    Reduces model complexity
    Enables Model Interpretability
    Handles data challenges
    Handles Data challenges
Types of features
    Categorical
    Numerical
    Ordinal

Categorical features
    Represent discrete categories or labels
    Encoding techniques
        One-hot encoding
        Label encoding
    Numerical Features
        Represent continuous or discrete numbers
        preprocessing techniques
            Scaling
    Ordinal Features
        Represent categorical data with meaningful order
        Encoding technique
            Ordinal encoding
Scaling
    Ensures all features contribute equally to the model
    techniques: min-max scaling, standardization
Encoding
    converts categorical data into numerical format
    techniques: one-hot encoding, label encoding, ordinal encoding
Transformation:
    Applies mathematical transformations to modify features
    techniques: log transformation, polynomial transformation,
Feature Selection:
    reduces the number of input features to improve model performance
    techniques: statistical methods , recursive features elimination (RFE)

"""

from unicodedata import numeric

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("data/titanic.csv")

print(df.info())
print(df.head())

# separate features
categorical_features = df.select_dtypes(include=["object"]).columns
numerical_features = df.select_dtypes(include=["int64", "float64"]).columns

print(categorical_features.tolist())
print(numerical_features.tolist())

# DISPLAY SUMMARY OF CATEGORICAL FEATURES
for col in categorical_features:
    print(col, df[col].value_counts())
# Display summary numerical features

for i in numerical_features:
    print(i, df[i].describe())
