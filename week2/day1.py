import numpy as np

arr = np.array([1, 2,3,4,5])
print(arr)
zeroes = np.zeros((3,4))
print(zeroes)
ones = np.ones((2,3))
print(ones)
eye = np.eye(4)
print(eye)
range_arr = np.arange(0,10,2)
print(range_arr)

#linear space between 0 and 1 with 5 elements
linspace_arr = np.linspace(0,1,5)
print(linspace_arr)

reshape_arr = arr.reshape((5,1))
print(reshape_arr)

arr = np.array([1,2,3])
expanded = arr[ :, np.newaxis]
"""[[1]
 [2]
 [3]]"""
print(expanded)

a = np.array([1,2,3])
b = np.array([4,5,6])

print(a+b)
print(a*b)
print(a/b)
print(np.dot(a,b))
print(np.sqrt(a))
print(np.mean(a))
print(np.median(a))
print(np.std(a))


print("\n\n\n")


arr = np.array([[1,2],[4,5],[7,8]])
print(arr[1,1])

print(arr[0:2, 1:3])

reshape_arr = arr.reshape((2,3))
print(reshape_arr)

#Exercise:

a = np.arange(1,20)
b = np.arange(21,40)
print(a.ndim)
print(b.ndim)
print("addition: ", a+b)
print("subtraction: ", a-b)
print("multiplication: ", a*b)
print("division: ", a/b)
print("dot product: ", np.dot(a,b))


a = np.array([[1,2,3],[4,5,6],[7,8,9]])

b = np.array([[9,8,7],[6,5,4],[3,2,1]])

print(a+b)
print(a-b)
print(a*b)
print(a@b)
print(np.dot(a,b))
print(a)
print(np.sum(a,axis=1))
print((a - a.min()) / (a.max() - a.min()))