import numpy as np
import matplotlib.pylab as plt

plt.ion()

def getS(A):
    k = getKout(A)

    S = np.zeros([len(A),len(A)])
    for row in range(len(A)):
        for column in range(len(A[row])):
            if k[column] != 0:
                S[row,column] = A[row, column] / k[column]
            else:
                S[row,column] = 1/len(A)
    return S

def getKout(A):
    k = np.zeros(len(A))
    for j in range(len(A)):
        for i in range(len(A[j])):
            k[i] = k[i] + A[j,i]
    return k

def getKin(A):
    k = np.zeros(len(A))
    for j in range(len(A)):
        for i in range(len(A[j])):
            k[j] = k[j] + A[j,i]
    return k

def getP(S, j0, order = 0):
    p = np.zeros(len(S))
    p[j0] = 1

    for n in range(order):
        p = S.dot(p)

    return p

def getG(S, alpha, v):
    G = np.zeros([len(S),len(S)])
    for i in range(len(S)):
        for j in range(len(S[i])):
            G[i,j] = alpha * S[i,j] + (1-alpha) * v[i]

    return G

def surf(G, P, epsilon = 1e-10, plot=False):
    order = 0
    while(diff(G.dot(P),P) >= epsilon):
        P = G.dot(P)
        order += 1
        if plot:
            plt.clf()
            plt.bar(range(len(P)),P,visible=True)
            plt.draw()
            plt.pause(.001)
    return P, order

def diff(a,b):
    res = 0
    for i in range(len(a)):
        res += abs(a[i] - b[i])

    return res

A = np.array([
    [0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 0]
])

S = getS(A)

v = np.ones(len(S)) / len(S)
#j0 = 3                          #np.random.randint(1, len(A))
#val, vec = np.linalg.eig(G)
P = 1/6 * np.array([1,1,1,1,1,1]) # getP(S, j0, 0)
epsilon = 1e-10

# 2.1 - 1
alpha = 0.85
G = getG(S, alpha, v)
print(f"\n2.1)\n   > For a random surfer that start his journey on a random node and with a dumping facotr at {round(alpha,2)}, he have these chances to finish at each node:\n")
plt.subplot(211)
P, step = surf(G,P,epsilon)
for i in range(len(P)):
    print(f"      Node {i} : {P[i]}")
print(f"\n      This stability is reached after {step} steps")

# 2.1 - 2
plt.title(f"Position of the surfer fore alpha={round(alpha,2)}"); plt.xlabel("Node"); plt.ylabel("Probability")
alpha = np.random.rand() / 2 + 0.5
plt.subplot(212)
G = getG(S, alpha, v)
print(f"\n   > Now, if we change the dumping factor to {round(alpha,2)}, the surfer have these chances to finish at each node:\n")
P, step = surf(G,P,epsilon,plot=True)
for i in range(len(P)):
    print(f"      Node {i+1} : {P[i]}")
print(f"\n      This stability is reached after {step} steps")

plt.pause(999)


#fig, ax = plt.subplot([],[])
#fig.
#fig.set_ydata(P)
#fig.set_label("test")
#plt.show()