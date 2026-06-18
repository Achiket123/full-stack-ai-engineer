"""
Correlation
    Measures the strength and direction of a relationship between two variables.
    values range from -1 to 1, with 0 indicating no correlation.

    types:
        - Pearson correlation
            - measures linear correlation between two variables
        - Spearman correlation
            - measures monotonic correlation between two variables

Linear Regression Basics
    method to model the relationship between a dependent variable y and one more independent variable x.
    y = B0 + B1.X + E
    - B0 is the intercept
    - B1 is the slope
    - E is the error term
    Key Metrics:
        - B0: intercept
        - B1: slope
        - R-squared: measures the proportion of variance in y explained by x (1 - ss_residual/ss_total)
        - Adjusted R-squared: accounts for the number of predictors in the model
        - p-value: indicates the significance of the regression coefficients
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/iris.csv")
x = df["sepal_length"]
y = df["sepal_width"]
np.random.seed(42)
x = np.random.rand(100, 1) * 10
y = 3 * x + np.random.randn(100, 1) * 2
df = df.drop(columns=["species"])
correlation_matrix = df.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.show()
# Pearson correlation

pearson_corr, _ = pearsonr(x, y)
print("Pearson correlation: ", pearson_corr)

rho, _ = spearmanr(x, y)
print("Spearman correlation: ", rho)

model = LinearRegression()
model.fit(x, y)

slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(x, y)

# Visualize
plt.scatter(x, y, color="blue", label="Data")
plt.plot(x, model.predict(x), color="red", label="regression")
plt.legend()
plt.title("Linear Regression")
plt.show()

"""
- Fit a multiple linear regression model with multiple independent
- variables
- Compare correlation and regression results for non-linear relationships
Use real-world datasets (e.g., housing prices) for regression analysis.
"""
