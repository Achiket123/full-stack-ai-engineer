# PROBABILITY THEOREM AND DISTRIBUTION
# PROBABILITY BASICS
# CONDITIONAL PROBABILITY
# THE PROBABILITY OF AN EVENT A GIVEN THAT EVENT B HAS OCCURRED
# P (A|B) = P (A intersection B)/ P(B)
# BAYES' THEOREM
# P (A|B) = (p(B|A).p(A))/P(B)
#  p(A) = prior probability
#  p(B|A) = likelihood
#  p(B) = Evidence
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson  # , bernoulli, binom, norm


def bayes_theorem(prior_prob, likelyhood, evidence):
    return likelyhood * prior_prob / evidence


# COMMON PROBABILITY DISTRIBUTION
# Gaussian Normal Distribution
#   Bell-shaped curve with mean (mew (u)) and standard deviation (sigma )
#  f(x) = (1/root(2pi(sigm^2)) ) e ^ (-(x- mew(u))^2)/(2(sigma^2))

mu, sigma = 0, 1
x = np.linspace(-4, 4, 100)
y = (1 / (np.sqrt(2 * np.pi * (sigma**2)))) * (np.exp(-0.5 * ((x - mu) / sigma) ** 2))

# plt.plot(x, y)
# plt.title("Gaussian Distribution")
# plt.show()

# Bernoulli Distribution
#   Describes the outcomes of a binary experiment
#   P(X=0)=p , P(X=1)= 1-p

# p = 0.6
# plt.bar([0, 1], [1 - p, p], color="blue")
# plt.title("bernoulli distribution")
# plt.xticks([0, 1], labels=["0 (Failure)", "1 (Success)"])
# plt.show()

#  Binomial Distribution
#  P(X=k) = ( n k ) p^k (1-p)^(n-k)

# from scipy.stats import binom, poisson

# lam = 3
# x = np.arange(0, 10)
# y = poisson.pmf(x, lam)
# plt.bar(x, y, color="orange")
# plt.title("Poisson Distribution")
# plt.show()


# APPLICATIONS
# GAUSSIAN DISTRIBUTIONS:
#   USED IN GAUSSIAN NAIVE BAYES AND KERNEL DENSITY ESTIMATION
# BERNOULLI DISTRIBUTION
#   MODELS BINARY CLASSIFICATION PROBLEMS [SPAM/NOT SPAM, HUMAN/NOT HUMAN, ETC.]
# BINOMIAL DISTRIBUTION
#   USED IN LOGISTIC REGRESSION TO MODEL BINARY OUTCOMES
# POISSON DISTRIBUTION
#   MODELS THAT COUNTS DATA
#
# HONs
#

# Calculate probabilites using bayes' theorem
# a disease affects 1% of the population
# a test which is 95% accurate for disease individuals, and 90% accurate for non disease individuals
# find the probability of having the disease given a positive test result


def bayestheorem(prior_probability, sensitivity, specificity):
    evidence = (sensitivity * prior_probability) + (
        (1 - specificity) * (1 - prior_probability)
    )
    posterior = (sensitivity * prior_probability) / evidence
    return posterior


prior_probability = 0.01
sensitivity = 0.95
specificity = 0.90

posterior = bayestheorem(prior_probability, sensitivity, specificity)
print(posterior)


# Gaussian/Normal
# x = np.linspace(-4, 4, 100)
# y = norm.pdf(x, loc=0, scale=1)
# plt.plot(x, y, label="Gaussian/Normal")
# plt.title("Gaussian Distribution")
# plt.show()

#  Binomial Distribution
# #
# n, p = 10, 0.5
# x = np.arange(0, n + 1)
# y = binom.pmf(x, n, p)
# plt.bar(x, y, label="Binomial")
# plt.show()

# Poisson Distribution
lam = 3
x = np.arange(0, 10)
y = poisson.pmf(x, lam)
plt.bar(x, y, label="Poisson")
plt.show()
# Multinomial Distribution for multiclass data
# Gaussian and Unifor Distribution continuous Data
# Use probability distribution to simulate and analyze real-world datasets
#
