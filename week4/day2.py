"""
Common Probability Distributions

Gaussian Distribution
    Bell Shaped Curve characterized by mean(mew) and standard deviation(sigma)
    Also known as the Normal Distribution
    PDF: f(x) = 1/(sigma * sqrt(2*pi)) * exp(-(x - mu)^2 / (2 * sigma^2))
    Properties:
        Mean (mu): Center of the distribution
        Standard Deviation (sigma): Spread of the distribution
        Area under the curve = 1
        Symmetric about the mean,
        mean=median=mode
    Applications:

        Common Assumptions in many algorithms [eg. Naive Bayes]
        Used in features scaling [eg. Standardization]

"""

"""
Binomial Distribution
Models The number of successes in n independent Bernoulli trials
Probability Mass Function: f(k) = (n choose k) * p^k * (1-p)^(n-k)
Properties:

    Discrete
    Parameters: n (number of trials), p (probability of success)
Applications:
    Logistic Regression assumes a binomial distribution for binary classification
"""

"""
Poisson Distribution
    Models the number of events occurring in a fixed interval of time or space
    PDF: f(k) = exp(-lambda) * lambda^k / k!
    Properties:
        Mean (lambda): Average number of events
        Discrete
        parameter: lamda
    Applications:
        Modeling the number of phone calls received per hour
        Modeling the number of accidents at a given intersection
"""
"""
Uniform Distribution
Equal probability for all outcomes in a range
Probability Density Function: PDF: f(x) = 1 / (b - a)
Properties:
    Continuous
    Parameters: a (lower bound), b (upper bound)

Applications:

"""

"""
Applications of Distributions:
    Gaussian Distribution
        Used in alogithms like naive bayes and gaussian matrix models
        assumed in statistical tests

    Binomial Distribution
        Foundational for logistic regression and other binary classification models
    Poisson Distributions
        Applied in modeling count data
        eg. number of phone calls received per hour
    Uniform Distribution
        Commonly used in random sampling and Monte Carlo simulations, and initializations of paramters

"""
"""
Visualizing Distributions and understanding their properties
Skewness- measures the asymmetry of distributions
    Positive skew: tail extends to the right
    Negative skew: tail extends to the left
Kurtosis- measures the tailedness / "peakedness" of distributions
    High kurtosis: distribution is leptokurtic (heavy tails)
    Low kurtosis: distribution is platykurtic (light tails)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binom, kurtosis, norm, poisson, skew, uniform

# x = np.linspace(-4, 4, 100)
# plt.plot(
#     x,
#     norm.pdf(x, loc=0, scale=1),
#     label="Normal/Gaussian Distribution loc/mew = 0, scale/sigma=1",
# )
# n, p = 10, 0.5
# x = np.arange(0, n + 1)
# plt.bar(x, binom.pmf(x, n, p), alpha=0.7, label="Binomial (n=10, p=0.5)")

# Poisson
# lam = 3
# x = np.arange(0, 10)
# plt.bar(x, poisson.pmf(x, lam), alpha=0.7, label="Poisson (lambda=3)")

# plt.title("Probability Distributions")
# plt.legend()
# plt.show()
# Uniform
# x = np.random.uniform(low=0, high=1, size=10000)
# sns.histplot(x, kde=True, label="Uniform (0,1)")
# plt.show()


url = "data/iris.csv"
df = pd.read_csv(url)
feature = df["sepal_length"]
print(f"""
Skewness: {skew(feature)}
Kurtosis: {kurtosis(feature)}
    """)
sns.histplot(feature, kde=True)  # pyright: ignore[reportArgumentType]
plt.show()

# Compare the effects of skewness and kurtosis on the distribution
# Simulate random variables from custom distributions