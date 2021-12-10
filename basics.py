# __________________________________________________
# 1.1

import numpy as np
from numpy.core.fromnumeric import sort
from numpy.core.function_base import linspace
from numpy.core.numeric import empty_like
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([-3, -2, -1])

print(A.dot(b))

# __________________________________________________
# 1.2

def sin(x):
    return np.sin(x)

print(sin(np.pi))

print(
    np.around(
        sin(
            np.array([
                0,
                np.pi/2,
                np.pi,
                3*np.pi/2,
                2*np.pi
            ])
        )
    ,3)
)

def area(pt1,pt2,pt3):
    x1,x2,x3,y1,y2,y3 = pt1[0],pt2[0],pt3[0],pt1[1],pt2[1],pt3[1]
    return 1/2 * np.abs(
            (x1 - x3)*(y2 - y1) - (x1 - x2)*(y3 - y1)
        )

print(area([0,0],[0,1],[1,0]))

# __________________________________________________
# 1.3

import matplotlib.pyplot as plt

x = np.linspace(0,500)/100

plt.subplot(131)
plt.plot(x,np.cos(x),marker="v")
plt.legend(["cos(x)"])
plt.subplot(132)
plt.plot(x,np.cos(x)**2,marker="o")
plt.legend(["cos^2(x)"])
plt.subplot(133)
plt.plot(x,np.cos(x**2),marker="x")
plt.legend(["cos(x^2)"])

plt.show()

# __________________________________________________
# 1.4

M = np.array([
    [ 1, 0, 0, 0, 1],
    [ 0,16, 0, 0, 0],
    [ 0, 0, 9, 0, 0],
    [ 0, 0, 0, 0, 0],
    [ 1, 0, 0, 0, 1],
])

eig,_ = np.linalg.eig(M)
print(eig)

def rank(scores):
    sorted = (-scores).argsort()
    rank = empty_like(scores)
    rank[sorted] = linspace(1,len(scores),num=len(scores))
    return rank.astype(int)

print(rank(eig))

print(M[2,:]) # print the third line
print(M[:,1]) # print the second column

# __________________________________________________
# 1.5

M = np.loadtxt("data/matrix.txt")

np.savetxt("results/res.txt", 
    M**2 + 2*M - 3 * np.identity(len(M)),
    fmt='%i'
)