# Data cleaning and manipulation with pandas - affect model performance and analysis
# Methods of handle missing values
#   Drop Missing Values drop_na
#   Fill Missing Values fill_na
#   Interpolate missing values (Linear interpolate)
import numpy as np
import pandas as pd

df = pd.read_csv("student_dirty.csv")
df1 = pd.read_csv("data-2.csv")
df2 = pd.read_csv("iris.csv")
droppeddf = df.dropna(axis=1)
print(droppeddf.head())
filleddf = df.fillna(22)
print(filleddf)
# Forward fill
print(df.ffill())
# Backward fill
print(df.bfill())

df["study_hours_per_day"] = df["study_hours_per_day"].interpolate()

print(df)
#
# Data transformation
#   Renaming columns
#   changin data type -> fload to int, int to double
#   Creating or modifying the columns
#
df = df.rename(columns={"study_hours_per_day": "s_hrs"})
df["s_hrs"] = df["s_hrs"].astype("int")

# df['datetime']=df['datetime'].to_datetime(df['datetime'])

df["col_name"] = df["s_hrs"] * 2
print(df)

# Combining and merging data frames
# Concatenation
comb = pd.concat([df1, df2], axis=0)  # rows
comb = pd.concat([df1, df2], axis=1)  # rows
# Merging
# merged = pd.merge(df1, df2, on="col_name")
# merged = pd.merge(df1, df2, how="left", on="col_name")
# merged = pd.merge(df1, df2, how="left", on="col_name")
# merged = pd.merge(df1, df2, how="right", on="col_name")
# Joining
joined = df.join(df2, how="inner")

data = {
    "Name": ["Alice", "Bob", np.nan, "David"],
    "Age": [10, np.nan, 20, 44],
    "Score": [23, 13, np.nan, 32],
}

df = pd.DataFrame(data)
print(df.head())
print(df.dropna())
print(df.fillna(df["Age"].std()))
df["Age"] = df["Age"].interpolate()
df["Score"] = df["Score"].interpolate()
# df["Name"] = df["Name"].interpolate()
print(df)
