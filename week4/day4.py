"""# Hypothesis Testing
#    Statistical method to determine if there is enough evidence in a sample to infer a conclusion about the population
# Key Components
#   Null Hypothesis: Assumes no effect or no difference
#   Alternative Hypothesis: Assumes there is an effect or difference
#
# Steps
#   Formulate null and alternative hypotheses [Ho and Ha]
#   Choose a significance level (alpha) [common values: 0.05, 0.01,0.1]
#   Calculate test statistic [t-statistic, z-statistic]
#   Determine p-value [compare the p-value to alpha]
#   Make decision to reject or fail to reject null hypothesis

# P-value
#   The probability of observing a results as extreme as the test statistic under Null Hypothesis
#   Smalller p-values indicate stronger evidence against null hypothesis
# Significance level (alpha)
#   Threshold for deciding whether to reject
#   Example : aplpha = 0.05 means 5%risj of rejecting Null Hypothesis when it is true
# Decision Rules
#   If p-value < alpha, reject the null hypothesis
#   If p-value >= alpha, fail to reject the null hypothesis

# Types of Errors
#   Type I Error (alpha): Rejecting the null hypothesis when it is true (false positive)
#   Type II Error (beta): Failing to reject the null hypothesis when it is false (false negative)
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_ind

df = pd.read_csv("data/iris.csv")
data = np.array([12, 14, 15, 16, 17, 18, 19])
alpha = 0.05
mean = 15
# Ttest

t_stat, p_val = ttest_1samp(data, mean)
print(f"t-statistic: {t_stat}, p-value: {p_val}")

if p_val <= alpha:  # pyright: ignore[reportOperatorIssue]
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")

grp_1 = [12, 14, 15, 16, 17, 18, 19]
grp_2 = [11, 13, 14, 15, 16, 17, 18]

# Perform t-test
t_test, p_value = ttest_ind(grp_1, grp_2)
print(f"T-statistic: {t_test}, p-value: {p_value}")

if p_value <= alpha:  # pyright: ignore[reportOperatorIssue]
    print("Reject the null hypothesis")
else:
    print("Failed to reject the null hypothesis")


# Perform a z-test for large sample sizes
# Use the Iris dataset to test if the mean sepal length differs between two
# species
# Perform hypothesis testing on proportions using the binomial distribution
