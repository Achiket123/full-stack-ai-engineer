# Data visualization using matplotlib and seaborn
#
# Basic Syntax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

x = [1, 2, 3, 4]
y = [10, 20, 40, 3]
# Line Plot - Used for Trends
# plt.plot(x, y)
# plt.title("Line Plot")
# plt.xlabel("X-Axis")
# plt.ylabel("Y-Axis")
# plt.legend()
# plt.show()

# BAR CHART- Used for Categorical Comparisons
# plt.bar(x, y, color="blue")
# plt.title("Bar Chart")
# plt.show()

# HISTOGRAM - SHOWS DISTRIBUTION OF A DATASET
# val = [2, 2, 4, 13, 5, 1, 3, 4, 1, 3, 5, 3, 2, 4, 2]
# plt.hist(val, bins=4, color="blue", edgecolor="black")

# plt.title("HISTOGRAM")
# plt.show()

# SCATTER PLOT Visualizes the reln b/w two variables
#
# x = [1, 2, 3, 4, 5]
# y = [23.14, 35, 2, 36, 42]
# plt.scatter(x, y, color="red")
# plt.title("Scatter Plot")
# plt.show()


# SEABORN FOR ADVANCED VISUALIZATION

# Heatmap
# Pairplot - pair wise relationship
#

# data = np.random.rand(5, 5)
# sns.heatmap(data, annot=True, cmap="coolwarm")
# plt.title("HEAT MAP")
# plt.show()

#  PAIR PLOT
df = pd.read_csv("./data/iris.csv")
# sns.pairplot(df)
# plt.title("IRIS PAIR PLOT")
# plt.show()
# del df["species"]
# correlation_matrix = df.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.show()

# for species in df["species"].unique():
#     data = df[df["species"] == species]
#     plt.hist(data["sepal_length"], bins=15, alpha=0.5, label=species)

# plt.legend()

# plt.xlabel("sepal_length")
# plt.ylabel("Frequency")
# plt.title("Overlaid Histogram")
# plt.show()


#  BOX PLOT
# df = sns.load_dataset("iris")

# sns.boxplot(data=df, x="species", y="petal_length")

# plt.title("Petal Length Distribution by Species")
# plt.xlabel("Species")
# plt.ylabel("Petal Length")
# plt.show()

# Violin Plot
# sns.violinplot(data=df, x="species", y="petal_length")

# plt.title("Petal Length Distribution by Species")
# plt.xlabel("Species")
# plt.ylabel("Petal Length")
# plt.show()


fig, axes = plt.subplots(2, 2, figsize=(14, 8))
sns.histplot(data=df, x="sepal_length", ax=axes[0, 0])
axes[0, 0].set_title("sepal length")

sns.histplot(data=df, x="sepal_width", ax=axes[0, 1])
axes[0, 1].set_title("sepal width")

sns.histplot(data=df, x="petal_length", ax=axes[1, 0])
axes[1, 0].set_title("Petal Length")

sns.histplot(data=df, x="petal_width", ax=axes[1, 1])
axes[1, 1].set_title("Petal Width")

plt.tight_layout()
plt.show()
