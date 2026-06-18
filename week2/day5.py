import numpy as np
import pandas as pd

df = pd.read_csv("iris.csv")
# print(df.head())
grouped = df.groupby("species")
for name, group in grouped:
    print(name)
    print(group)
print(grouped.mean())
print(grouped.sum())
print(df.groupby("species")["sepal_width"].mean())
print(df.groupby("species")["sepal_width"].median())
print(df.groupby("species")["sepal_width"].std())

# Aggregation
print(df.groupby("species").agg({"sepal_width": ["mean", "median", "min"]}))

# Pivot Table
pivot = df.pivot_table(values="sepal_width", index="species", aggfunc="mean")
print(pivot)


# Custom Aggregation
def custom_range(x):
    return x.max() - x.min()


print(df.groupby("species")["sepal_width"].agg(custom_range))
# Multi Aggregation
df.groupby("species").agg(
    {
        "sepal_width": ["mean", "min", "max"],
    }
)

# Hands On
df = pd.read_csv("iris.csv")

print(df.groupby("species")["sepal_width"].mean())
# Calculate Summary stats for grouped data
print(
    df.groupby("species")["sepal_width"].mean(),
    df.groupby("species")["sepal_width"].min(),
    df.groupby("species")["sepal_width"].max(),
    df.groupby("species")["sepal_width"].max(),
    df.groupby("species")["sepal_width"].std(),
    df.groupby("species")["sepal_width"].std(),
    df.groupby("species")["sepal_width"].var(),
    df.groupby("species")["sepal_width"].median(),
)
print("Table: ")
print(
    df.pivot_table(
        values="sepal_width", index="petal_width", columns="species", aggfunc="median"
    )
)
