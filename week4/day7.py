"""
Exploring Exploratory Data Analysis EDA
Steps in EDA:
    Load and inspect the dataset
    Check for missing or iconsistent data
    Visualize these distributions -> can create a report

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/tips.csv")
contingency_table = pd.crosstab(df["smoker"], df["time"])

# perform chisqaure test
chi2, p_val, dof, expected = chi2_contingency(contingency_table)
print(f"""
    ChiSquared : {chi2}
    Pvalue : {p_val}
    Degrees of Freedom: {dof}
    """)
alpha = 0.05
if p_val <= alpha:
    print("Reject The Null Hypothesis, variables are dependent")
else:
    print("Failed to reject null hypthosis, variables are independent")
# sns.histplot(df["total_bill"], kde=True)
# plt.title("distribution of total bill")
# plt.show()
# df = df.drop(
#     columns=[
#         "sex",
#         "smoker",
#         "day",
#         "time",
#     ]
# )

# sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# plt.show()


"""
Conducting Hypothesis Testing
    Steps:
        Formulate null and alternative hypotheses
        Choose and perform an appropriate test
        Interpret p-values and test statistics
"""

# male_tips = df[df["sex"] == "Male"]["tip"]
# female_tips = df[df["sex"] == "Female"]["tip"]

# # Perform t-test
# t_stat, p_value = ttest_ind(male_tips, female_tips)
# print(f"t-statistic: {t_stat}, p-value: {p_value}")

# alpha = 0.05
# if p_value < alpha:
#     print(
#         "Reject the null hypotheses: Significant difference between male and female tips"
#     )
# else:
#     print(
#         "Fail to reject the null hypotheses: No significant difference between male and female tips"
#     )
# Applying Linear Regression
#
model = LinearRegression()
X = df["total_bill"].values.reshape(-1, 1)  # Why ??
y = df["tip"].values
model.fit(X, y)
print(f"""
    Model Slope: {model.coef_[0]}
    Model coefficients: {model.coef_},
    intercept: {model.intercept_}
    R-Square: {model.score(X, y)}
    """)


sns.scatterplot(x=df["total_bill"], y=df["tip"], color="blue")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.legend()
plt.show()


"""
Extend the project by exploring additional relationships (e.g., day of the week vs. tip amount).
Perform multiple linear regression with additional variables (e.g., include smoking status).
Use another real-world dataset (e.g., healthcare or sales data) to apply similar techniques.
"""
