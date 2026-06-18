import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/placement.csv")

print(df.head())
print(df.info())
print(df.describe())
y = df["Suggested Job Role"].groupby(df["Suggested Job Role"]).count()
X = df["Suggested Job Role"].unique()
plt.bar(
    X,
    y,
)
plt.title("Bar Chart of Suggested Job Roles")
plt.xlabel("Job Role")
plt.xticks(rotation=90)
plt.ylabel("Count")
plt.show()
numerical_features = df.select_dtypes(include=["int64"])
