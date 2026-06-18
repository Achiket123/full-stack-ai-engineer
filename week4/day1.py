# Probability Theory and Random Variables
# Sample Space
# Events
# Conditional Probability: The probability of an event A occuring, given that B has occured
#
# Independent Events:
#   Events that do not affect each other's probability
#   P(A Intersect B) = P(A)*P(B)
#
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform

sample_space = list(range(1, 7))
even = [2, 4, 6]
p_even = len(even) / len(sample_space)
#
#
# Random Variables
#   A variable that takes on a value from a sample space
# Map outcomes of a random experiment to numerical values
# Types: Discrete , Continuous
#
# Probability Mass Function:
#  A function that assigns a probability to each outcome in a discrete sample space
# Probability distribution of a discrete random variable
#  [1,2,3,4] rolling of dice
# Probability Density Function
# Probability distribution of  a continuous random variable

# Expectation E[x]
#  weighted average of a random variable's possible values
#  E[x] = sum(x * P(x)) Discrete
#  E[x] = integral(x * f(x)) Continuous
# Variance Var[x]
# Var[x] = E[(X-E[X])^2]
# Standard Deviation: sqrt(Var[x])

outcomes = np.array([1, 2, 3, 4, 5, 6])
probabilities = np.array([1 / 6] * 6)
expectations = np.sum(outcomes * probabilities)

variance = np.sum(((outcomes - expectations) ** 2) * probabilities)
std_dev = np.sqrt(variance)

print(f"""
    Expectation: {expectations}
    Variance: {variance}
    Standard Deviation: {std_dev}
    """)

# Simulate 10000 Dice Rolls and calculate probabilites
#
rolls = np.random.randint(1, 7, size=1000000)

P_even = np.sum(rolls % 2 == 0) / len(rolls)
P_greater_than_4 = np.sum(rolls > 4) / len(rolls)

print(f"""
    P(even): {P_even}
    P(greater than 4): {P_greater_than_4}
    """)

# Create and analyze random variables
# outcomes = np.array([1, 2, 3, 4, 5, 6])
# probabilities = np.array([1 / 6] * 6)
# plt.bar(outcomes, probabilities, color="Blue")
# plt.xlabel("Outcomes")
# plt.ylabel("Probabilities")
# plt.title("Probability Distribution of a Dice Roll")
# plt.show()

# Continuos random variable: Unifor distribution
#
X = np.linspace(0, 1, 100)
Y = uniform.pdf(X, loc=0, scale=1)
plt.plot(X, Y, color="Blue")
plt.xlabel("Outcomes")
plt.ylabel("Probabilities")
plt.title("Probability Distribution of a Dice Roll")
plt.show()

