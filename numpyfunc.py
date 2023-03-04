from re import A
import numpy as np
a = np.arange(6)
# print(a)
b = np.array([2,3,4,5,2,5,23,45,42,52,12,12])
# print(b)
# print(b.shape)
b= b.reshape(3,4)
print(b)
b = np.argsort(b, axis=0)
print(b)