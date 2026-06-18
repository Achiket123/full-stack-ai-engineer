# Advance Numpy
# Broadcasting -> performing arithmetic operations on arrays of different shapes
import os
import sys
import numpy as np
# Array and scalar boradcasting
a = np.array([1,2,3]) 
print(a+10)

matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
vector = np.array([1,0,1])
print(matrix+vector)

# Aggregate functions

# sum, mean, median, std, var, min, max

arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(np.sum(arr))
print(np.mean(arr)) 
print(np.median(arr))
print(np.std(arr))
print(np.var(arr))
print(np.min(arr))
print(np.max(arr))

# boolean indexing

arr = np.array([[1,2,3],[4,5,6],[7,8,9]])

print(arr[arr > 5])
print(arr[arr % 2 == 0])

# Random Number Generation

random_arr = np.random.rand(3,3) # 0 to 1 random numbers in a 3x3 array
print(random_arr)

rand_int_arr = np.random.randint(1,10,(3,3)) # random integers between 1 and 10 in a 3x3 array
print(rand_int_arr)
# Setting random seed for reproducibility
np.random.seed(42)
random_arr = np.random.rand(3,3)
print(random_arr)

mat = np.array([[1,2,3],[4,5,6],[7,8,9]])

vector = np.array([1,0,-1])

result= (mat + vector)

print(result*2)

# Generate and filter random dataset

data = np.random.randint(1,51,(5,5)) # 100 samples, 5 features

# Filter samples where the first feature is greater than 0.5
data[data>25]=0

print(data)
# Sum
print(np.sum(data)) # sum of each column
# Mean
print(np.mean(data))
# Standard Deviation
print(np.std(data))