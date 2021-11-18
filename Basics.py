import numpy as np

A = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([-3,-2,-1])

print (A*b)

M = np.array(
    [
        [1, 0, 0, 0, 1],
        [0,16, 0, 0, 0],
        [0, 0, 9, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1]
    ]
)

ev = np.argsort(M)

print(ev)

print(M[2,:])

print(M[:,1])