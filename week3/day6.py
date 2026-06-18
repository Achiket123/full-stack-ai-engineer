import scipy.stats as stats

# Statistics Fundamentals
# Measures of central tendency and dispersion
# Central Tendency :
#   Mean = average value
#   Median = middle value
#   Mode = Most frequent
# Dispersion:
#   Variance: Avg squared deviation from the mean
#   Standard Deviation: Square root of variance, indicating the spread of the data

# Variance
#
lst = [10, 20, 30, 40, 50]
mean = sum(lst) / len(lst)
var = sum((x - mean) ** 2 for x in lst) / len(lst)
print(var)
print(std := var**0.5)
# Hypothesis Testing
# Statistical Methiod which tells the output is significant or by chance
# Steps:
#   Formulate Hypothesis
#       Null Hypothesis H(0) | no effect or difference
#       Alternate Hypothesis H(A) | an effect or difference
#   Calculate Test Statistic
#   Determine P-value
#   Interpret the results

# Confidence Intervals
#   Range of values within which the true population parameter is expected to lie.
#   CI = bar(x) +- z . s / root(n)
#   bar(x) = sample mean
#   z = Z-score [eg. 1.96 for 95% confidence]
#   standard deviation
# Statistical Significance
#
sample_mean = mean
z_score = 1.96
CI = (
    sample_mean - z_score * (std / len(lst) ** 0.5),
    sample_mean + z_score * (std / len(lst) ** 0.5),
)

print(CI)

# Perform A T-Test

grp_a = [2.1, 2.5, 2.8, 3.0, 3.2]
grp_b = [1.8, 2.0, 2.4, 2.7, 2.9]

# perform a ttest
t_stat, p_val = stats.ttest_ind(grp_a, grp_b)
print("T- Statistics: ", t_stat)
print("P- Value: ", p_val)

# Interpretation
alpha = 0.05
if p_val < alpha:  # pyright: ignore[reportOperatorIssue]
    print("Reject null hypothesis")
else:
    print("Failed To reject the null hypothesis: no significant difference")

# Visualize the distribution  of data and highlight mean, median and mode using matplotlib
# perform hypothesis
