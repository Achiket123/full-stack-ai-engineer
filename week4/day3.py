"""
Introduction to Statistical Inference
What is Statistical Inference?
    process of making conclusions about a population based on sample data
Population Vs Sample
Goal
    Estimate population parameters and asses the reliability of these estimates
"""

"""
Point Estimation and Interval Estimation
    Point Estimation:
        Single value estimate of a population parameter
    Interval Estimation:
        Range of values that likely contains the true population parameter
        Confidence interval:
            Range of values that contains the true population parameter with a certain level of confidence
            CI = bar(x) ± z * (σ / √n) bar(x)= mean, z = z-score, σ = standard deviation, n = sample size
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t

df = pd.read_csv("data/iris.csv")
data = df["sepal_length"]

mean = np.mean(data)
std = np.std(data, ddof=1)  # ddof is set to 1 to use sample standard deviation

# 95% confidence interval using t-distribution
n = len(data)
z_val = norm.ppf(0.975)
margin_of_error = z_val * (std / np.sqrt(n))
ci = (mean - margin_of_error, mean + margin_of_error)

print(f"""
    95% Confidence Intercval:
    Mean: {mean}
    Confidence Interval: {ci}
    """)


sample = data.sample(30, random_state=40)
mean = sample.mean()
std = sample.std(ddof=1)
n = len(sample)

z_val = norm.ppf(0.975)
margin_of_error = z_val * (std / np.sqrt(n))
ci = (mean - margin_of_error, mean + margin_of_error)

print(f"""
    95% Confidence Intercval:
    Mean: {mean}
    Confidence Interval: {ci}
    """)


# Create confidence intervals for other statistics
# Perform stratified sampling and compare intervals across strata
# Visualize confidence intervals for multiple samples using
# Matplotlib