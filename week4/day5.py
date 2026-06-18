# T-Tests
# purpose: Test whether the means of two independent samples/group differ significantly
# Types:
#   One-sample T-test: Tests if the mean of a sample differs from  a known value  or population mean
#   Two-sample T-test (Independent T-Test): Tests if the means of two independent samples differ significantly
#   Paired Sample T-test: Tests if the means of two related samples differ significantly
#
# Usecase:
#   One-sample : testing if the average test score of a class differs from  the national average
#   Two-sample (Independent T-Test): testing if the average test score of two classes differs significantly
#   Paired Sample T-test: testing if the average test score of two related classes differs significantly

# Chi-Square Test:
# purpose: Test for independence or goodness-of-fit in categorical data
# Chi-Square Test of Independence: Test if two categorical variables are independent
# Usecase:
#   Chi-Square Test of Independence: testing if two categorical variables are independent of each other

# Steps:
# Create a contingency table
# Calculate expected frequencies
# Compute chi-square statistic and p-value
#

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, ttest_1samp, ttest_ind, ttest_rel

df = pd.read_csv("data/iris.csv")
# Contingency table
data = [[50, 30], [20, 40]]
# Perform chi-square test
chi2, p_val, ddof, expected = chi2_contingency(data)

print(
    f"Chi-square: {chi2}, p-value: {p_val}, degrees of freedom: {ddof} expected frequencies: {expected}"
)

# ANOVA (Analysis of variance)
# purpose: Compare means of three or more groups
# Hypothesis:
#   Null (H0): The means of all groups are equal
#   Alternative (H1): At least one group mean is different from the others

grp_1 = [12, 14, 15, 16, 17]
grp_2 = [11, 13, 14, 15, 16]
grp_3 = [10, 12, 13, 14, 15]

# Perform ANOVA
f_stat, p_val = f_oneway(grp_1, grp_2, grp_3)
print(f"F-statistic: {f_stat}, p-value: {p_val}")


# ------------------
# One Sample T-test
data = df["sepal_length"]
pop_mean = data.mean()
t_stat, p_val = ttest_1samp(data, pop_mean)
print("One Sample T-test")
print(f"t-statistic: {t_stat}, p-value: {p_val}")

# Two-sample (Independent T-Test)
data1 = df["sepal_length"]
data2 = df["sepal_width"]
t_stat, p_val = ttest_ind(data1, data2)
print("Two-sample (Independent T-Test)")
print(f"t-statistic: {t_stat}, p-value: {p_val}")

# Paired Sample T-test
pre_test = df["sepal_length"]
post_test = df["sepal_width"]
t_stat, p_val = ttest_rel(pre_test, post_test)
print("Paired Sample T-test")
print(f"t-statistic: {t_stat}, p-value: {p_val}")


# Chi-square test
data = [[50, 30, 20], [30, 40, 30]]
chi2, p, dof, expected = chi2_contingency(data)
print(f"""
Chi-square: {chi2}, p-value: {p}, degrees of freedom: {dof}
Expected frequencies: {expected}
    """)

# Conduct ANOVA
f_stat, p_val = f_oneway(grp_1, grp_2, grp_3)
print(f"F-statistic: {f_stat}, p-value: {p_val}")


# Perform a two-way ANOVA to test for interaction effects
# Use real-world datasets (e.g., student scores by gender and class) for hypothesis testing
# Visualize test results using boxplots or bar plots.
