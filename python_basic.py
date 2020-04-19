import numpy as np

arr0 = np.arange(12)
arr1 = arr0[:]
print(id(arr0))
print(id(arr1))

arr0[0] = 100
print(arr0)
print(arr1)

brr0 = np.zeros_like(arr0)
brr0[...] = arr0
arr0[0] = 1000
print(brr0)
print(arr0)
