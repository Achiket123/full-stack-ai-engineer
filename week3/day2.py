import numpy as np

# det(A) == 0 singular matrix, non-invertible
# det(A) != 0 invertible matrix
# for 2x2 matrix the det represents the scaling factor of the area formed by its column vectors
# def (A) = [ [a,b],[c,d]] = ad-bc
a = np.array([[2, 3], [1, 4]])
det = np.linalg.det(a)
print(det)

# Inverse of a matrix =
inv = np.linalg.inv(a)
print(inv)
# EigenValues and EigenVector = Describe transformation  , vector that doesn't change direction during transformation.

#  if A.v == Y.v
# then v is eigen vector
# and Y is eigen value
# eignevectors point in the direction where the matrix transformation stretches or compresses vectors
# eigenvalues indicats the factor of stretching or compress
#
# Properties
#   Matrix of size nxn has n eigen values and n eigenvectors
#   eigenvalues can be real or complex
#   for a symmetric matrix the eigen values are always real


eigenVal, eigenVec = np.linalg.eig(a)
print(eigenVal)
print(eigenVec)

B = np.array([[4, 2], [1, 1]])

eigenVal, eigenVec = np.linalg.eig(B)
print(eigenVal)
print(eigenVec)
# Matrix Decomposition
#   SVD Singular Matrix Decomposition
#   Decomposes matrix into three matrices A = U.M.Vt
#   U: LEFT SINGULAR MATRIX (ORTHOGONAL MATRIX)
#   M: DIAGONAL MATRIX OF SINGULAR VALUES (NON - NEGATIVE)
#   Vt: RIGHT SINGULAR MATRIX (ORTHOGONAL MATRIX)
#
#    Applications-
#       Dimensionality Reduction - Principal Component Analysis PC algorithm
#       Noise Reduction in data science
#       image processing

U, M, Vt = np.linalg.svd(B)
print(B)
print("U: ", U)
print("M: ", M)
print("Vt: ", Vt)
print("---")

# Reconstruct
sigma = np.zeros((2, 2))
np.fill_diagonal(sigma, M)
reconstructed = U @ sigma @ Vt
print(reconstructed)
# PROPERTY OF EIGEN VAL = det(A-YI)=0
print("----")
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
eigenVal, eigenVec = np.linalg.eig(B)
k = eigenVal[0] * np.eye(3, 3)
print(eigenVal[0])
lo = B - k
print(lo)
print(np.linalg.det(lo), np.isclose(np.linalg.det(lo), 0))
