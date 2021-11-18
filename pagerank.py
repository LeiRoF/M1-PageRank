import enum
import numpy as np
<<<<<<< HEAD
from bokeh.plotting import figure, show
=======
import matplotlib.pylab as plt

plt.ion()
>>>>>>> dea08c74649bf68309af175a11331b53814b4559

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
    evolution = []
    evolution.append(P)
    while(diff(G.dot(P),P) >= epsilon):
        P = G.dot(P)
        order += 1
<<<<<<< HEAD
        evolution.append(P)
    return P, order, evolution
=======
        if plot:
            plt.clf()
            plt.bar(range(len(P)),P,visible=True)
            plt.draw()
            plt.pause(.001)
    return P, order
>>>>>>> dea08c74649bf68309af175a11331b53814b4559

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

colorlist = ["red","orange","yellow","green","blue","purple","pink"]

p = figure(
    title="Centralities",
    sizing_mode="stretch_both",
    x_axis_label="Node",
    y_axis_label="Probability",
)

p1 = figure(
    title="ChaiRank for different P*",
    sizing_mode="stretch_both",
    x_axis_label="Node",
    y_axis_label="Probability",
)

p2 = figure(
    title="Evolution of CheiRank vector over iterations",
    sizing_mode="stretch_both",
    y_axis_label="Probability",
    x_axis_label="Iteration",
)

nodes = []
for i in range(len(A)): nodes.append(i+1)
nodes = np.array(nodes)

v = np.ones(len(S)) / len(S)
#j0 = 3                          #np.random.randint(1, len(A))
#val, vec = np.linalg.eig(G)
P = 1/6 * np.array([1,1,1,1,1,1]) # getP(S, j0, 0)
epsilon = 1e-6

#######################
# PART 2.1 - PAGERANK #
#######################

#__________________________________________________
# 2.1 - 1

alpha = 0.85
G = getG(S, alpha, v)
print("S = \n", S)
print("G = \n", G)
print(f"\n2.1)\n   > For a random surfer that start his journey on a random node and with a dumping facotr at {round(alpha,2)}, he have these chances to finish at each node:\n")
<<<<<<< HEAD
P, step, evolution = surf(G,P,epsilon)
=======
plt.subplot(211)
P, step = surf(G,P,epsilon)
>>>>>>> dea08c74649bf68309af175a11331b53814b4559
for i in range(len(P)):
    print(f"      Node {i+1} : {P[i]}")
p.vbar(x= nodes-0.20, top=P, color="red", width=0.20, legend_label="PageRank centrality")
print(f"\n      This stability is reached after {step} steps")

#__________________________________________________
# 2.1 - 2
<<<<<<< HEAD

=======
plt.title(f"Position of the surfer fore alpha={round(alpha,2)}"); plt.xlabel("Node"); plt.ylabel("Probability")
>>>>>>> dea08c74649bf68309af175a11331b53814b4559
alpha = np.random.rand() / 2 + 0.5
plt.subplot(212)
G = getG(S, alpha, v)
print(f"\n   > Now, if we change the dumping factor to {round(alpha,2)}, the surfer have these chances to finish at each node:\n")
<<<<<<< HEAD
P, step, evolution = surf(G,P,epsilon)
=======
P, step = surf(G,P,epsilon,plot=True)
>>>>>>> dea08c74649bf68309af175a11331b53814b4559
for i in range(len(P)):
    print(f"      Node {i+1} : {P[i]}")
print(f"\n      This stability is reached after {step} steps")

<<<<<<< HEAD
#__________________________________________________
# 2.2 - 3

P = None
P = []
print(f"\n   > In comparison, degree centrelity gives us:\n")
for i in range(len(A)):
    P.append(sum(A[i,:]) + sum(A[:,i])) # Sum of input and output links
P = np.array(P)
P = P/sum(P) # Normalisation
for i in range(len(P)):
    print(f"      Node {i+1} : {P[i]}")
p.vbar(x= nodes, top=P, color="blue", width=0.20, legend_label="Degree centrality")

#######################
# PART 2.2 - CHEIRANK #
#######################

#__________________________________________________
# 2.1 - 1

alpha = 0.85
print(f"\n2.2)\n   > For ChaiRank, alway with alpha = {round(alpha,2)}, we got this:\n")
S = getS(np.transpose(A))
G = getG(S, alpha, v)
P = 1/6 * np.array([1,1,1,1,1,1]) # getP(S, j0, 0)
P, step, evolution = surf(G,P,epsilon)
for i in range(len(P)):
    print(f"      Node {i+1} : {P[i]}")
print(f"\n      This stability is reached after {step} steps")
p.vbar(x= nodes+0.20, top=P, color="green", width=0.20, legend_label="ChaiRank centrality")
p1.circle(nodes, P, size=30, line_color="pink", fill_color="pink", line_width=2, legend_label=f"Arrived at random node")

#__________________________________________________
# 2.1 - 4

x = []
for i in range(len(evolution)):
    x.append(i)

y = []
evolution = np.array(evolution)
for i in range(len(P)):
    y.append(evolution[:,i])
    color = colorlist[i%7]
    p2.line(x, y[i], line_color=color, legend_label=f"Node {i+1}", line_width=2)

#__________________________________________________
# 2.1 - 3

alpha = 0.85
S = getS(np.transpose(A))
G = getG(S, alpha, v)
for i in range(len(S)):
    color = colorlist[i%7]
    P = np.zeros(len(S))
    P[i] = 1
    P, step, evolution = surf(G,P,epsilon)
    p1.circle(nodes, P, size=20-2*i, line_color=color, fill_color=color, line_width=2, legend_label=f"Arrived at node {i+1}")

show(p)
show(p1)
show(p2)


types = {}
with open("nodes.txt","r") as file:
    for i,line in enumerate(file):
        line = line.replace("\n","")
        line = line.split("	")
        if line[-1] in ["","	"]:
            line = line[:-1]
        try:
            types.update({int(line[0]):line[1]})
        except:
            pass


graph = []
with open("links.txt","r") as attacks:
    for i, line in enumerate(attacks):
        line = line.replace("\n","")
        line = line.split("	")

        if len(line) == 4: line = line[:-1]
        keep = True
        try:
            line[0] = int(line[0])
            line[1] = int(line[1])
            line[2] = float(line[2])
        except:
            keep = False
            pass
        if keep: graph.append(line)
matrix = np.ones((17,17))

for i in graph:
    a = i[0]-1
    d = i[1]-1
    r = i[2]
    print(a,d,r)
    matrix[a,d] = r

matrix = np.transpose(matrix)

print(matrix)
=======
plt.pause(999)


#fig, ax = plt.subplot([],[])
#fig.
#fig.set_ydata(P)
#fig.set_label("test")
#plt.show()
>>>>>>> dea08c74649bf68309af175a11331b53814b4559
