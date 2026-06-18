# LINEAR ALGEBRA
# ADVANCE LINEAR ALGEBRA
# CALCULAS FOR ML - DERIVATIVE
# CALCULAS FOR ML - INTEGRALS
# PROBABILITY THEORY AND DISTRIBUTIONS
# STATISTICS FUNDAMENTALS
# MATH DRIVEN MINI PROEJCT - LINEAR REGRESSION FROM SCRATCH
#
import numpy as np

a = np.array([[1, 2], [2, 4]])
b = np.array([[2, 4], [3, 2]])
#  Addition/Subtraction
print(a + b)
print(a - b)
#  Scalar Multiplication
c = 2 * a
print(c)

# Matrix multiplication mxn , pxq, n==p
# Number of col in first mat == number of row in second matrix
z = np.dot(a, b)
print(z)
print(a @ b)
# Special Matrices
# Identity matrices
# [ 1 0 0
#   0 1 0
#   0 0 1
# ]
#
i = np.eye(3)
print(i)
# Zero Matrix
# [ 0 0
#   0 0 ]
# Diagonal Matrix

dg = np.diag([1, 2, 3, 4, 5])
print(dg)
print(a)
v = np.array([1, 0, -1])
a = np.random.randint(1, 20, (3, 3))

print(np.dot(a, v))
print(np.linalg.det(a))
