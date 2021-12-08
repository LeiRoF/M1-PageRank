import enum
import numpy as np
import networkx as nx
from bokeh.plotting import figure, show, output_file
import os
import matplotlib.pyplot as plt
import time
from pokeplot import *

########################
# FUNCTIONS DEFINITION #
########################

# Computing stochastic matrix
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

# Computing Kout coefficient
def getKout(A):
    k = np.zeros(len(A))
    for j in range(len(A)):
        for i in range(len(A[j])):
            k[i] = k[i] + A[j,i]
    return k

# Computing Kout coefficient
def getKin(A):
    k = np.zeros(len(A))
    for j in range(len(A)):
        for i in range(len(A[j])):
            k[j] = k[j] + A[j,i]
    return k

# Get the google matrix
def getG(S, alpha, v):
    G = np.zeros([len(S),len(S)])
    for i in range(len(S)):
        for j in range(len(S[i])):
            G[i,j] = alpha * S[i,j] + (1-alpha) * v[i]
    return G

# Power iteration method
def surf(G, P, epsilon = 1e-10, plot=False):
    order = 0
    evolution = []
    evolution.append(P)
    while(diff(G.dot(P),P) >= epsilon):
        P = G.dot(P)
        order += 1
        evolution.append(P)
    return P, order, evolution

# Difference between vectors
def diff(a,b):
    res = 0
    for i in range(len(a)):
        res += abs(a[i] - b[i])

    return res

# Write a file containing all results. This function is used in the part 3
def save_result(file):
    if not os.path.isdir("results/"):
        os.makedirs("results/")
    with open("results/" + file,"w+") as file:
        file.write(f"{epsilon},{len(types)}\n\n")
        file.write(f"Rank, ID, Type, Power Iteration, Diagonalization\n")
        rank = nprank(P)
        for i, value in enumerate(P):
            file.write(f"{rank[i]},{i+1},{types[i+1]}, {value}, {D[i]}\n")

# Create a vector that contain the rank of each element of the vector given in parameter
'''
def nprank(scores):
    temp = scores.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(scores))
    return ranks
'''

def nprank(scores):
    rank = np.empty_like(scores)
    i = 0

    while len(scores) > 0:
        index = np.where(scores == np.amin(scores))
        scores = np.delete(scores, index)
        rank[index] = i
        i+=1
    
    return rank


#######################
# VARIABLE DEFINITION #
#######################

# Simple oriented graph
A = np.array([
    [0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 0]
])
S = getS(A)

# Color list for plots
colorlist = ["red","orange","yellow","green","blue","purple","pink"]

# Plot figures
p = figure(title="Centralities",sizing_mode="stretch_both",x_axis_label="Node",y_axis_label="Probability")
p1 = figure(title="ChaiRank for different P*",sizing_mode="stretch_both",x_axis_label="Node",y_axis_label="Probability")
p2 = figure(title="Evolution of CheiRank vector over iterations",sizing_mode="stretch_both",y_axis_label="Probability",x_axis_label="Iteration")
p3 = figure(title="Alpha influence",sizing_mode="stretch_both",x_axis_label="Node",y_axis_label="Probability")
p4 = figure(title="Pokemon ranking",sizing_mode="stretch_both",x_axis_label="Pokemon type ID",y_axis_label="Value")
p5 = figure(title="Test ranking",sizing_mode="stretch_both",x_axis_label="Type",y_axis_label="Value")
output_file("results/plot.html")

# Nodes (= websites)
nodes = []
for i in range(len(A)): nodes.append(i+1)
nodes = np.array(nodes)

v = np.ones(len(S)) / len(S)

# Starting vector
P = 1/6 * np.array([1,1,1,1,1,1])

# Precision
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
P, step, evolution = surf(G,P,epsilon)
for i in range(len(P)):
    print(f"      Node {i+1} : {P[i]}")
p.vbar(x= nodes-0.20, top=P, color="red", width=0.20, legend_label="PageRank centrality")
print(f"\n      This stability is reached after {step} steps")

#__________________________________________________
# 2.1 - 2

for j, alpha in enumerate(np.arange(0.5, 1.05, 0.1)):
    #alpha = np.random.rand() / 2 + 0.5
    G = getG(S, alpha, v)
    print(f"\n   > Now, if we change the dumping factor to {round(alpha,2)}, the surfer have these chances to finish at each node:\n")
    P, step, evolution = surf(G,P,epsilon)
    for i in range(len(P)):
        print(f"      Node {i+1} : {P[i]}")
    print(f"\n      This stability is reached after {step} steps")
    color = colorlist[j%7]
    p3.vbar(x= nodes - 0.25 + (0.1*j), top=P, bottom=0, color=color, width=0.10, legend_label=f"PageRank centrality for alpha={round(alpha,1)}")
p3.legend.location = "top_left"
show(p3)
time.sleep(1) # allow to avoid errors due to asynchroneous plotting method of bokeh

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
P = 1/6 * np.array([1,1,1,1,1,1])
P, step, evolution = surf(G,P,epsilon)
for i in range(len(P)):
    print(f"      Node {i+1} : {P[i]}")
print(f"\n      This stability is reached after {step} steps")
p.vbar(x= nodes+0.20, top=P, color="green", width=0.20, legend_label="ChaiRank centrality")
p1.vbar(nodes-0.30, top=P, color="pink", width=0.1, legend_label="Arrived at random node")

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

show(p2)
time.sleep(1)

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
    p1.vbar(nodes-0.20+(i*0.1), top=P, color=color, width=0.1, legend_label=f"Arrived at node {i+1}")

show(p)
time.sleep(1)
show(p1)

####################
# PART 3 - POKEMON #
####################

# Getting types of pokemon
types = {}
with open("data/nodes.txt","r") as file:
    for i,line in enumerate(file):
        line = line.replace("\n","")
        line = line.split("	")
        if line[-1] in ["","	"]:
            line = line[:-1]
        try:
            types.update({int(line[0]):line[1]})
        except:
            pass

# Reading pokemon interactions file
graph = []
with open("data/links.txt","r") as attacks:
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

# Generating matrix of pokemon interaction
matrix = np.ones((17,17))
for i in graph:
    a = i[0]-1
    d = i[1]-1
    r = i[2]
    matrix[a,d] = r

# Saving attack and defense ratio of each pokemon type
with open("results/ratio_attack.csv", "w+") as attack, open("results/ratio_defense.csv", "w+") as defense:
    for i in range(len(matrix)):
        attack.write(f"{i},{sum(matrix[i,:])/len(matrix)}\n")
        defense.write(f"{i},{sum(matrix[:,i])/len(matrix)}\n")

# Adapting the interaction matrix to the algrithme, according to the definition 3 in the subject.
A = np.transpose(matrix)

#__________________________________________________
# Pagerank

S = getS(A)
alpha = 0.85
v = np.ones(len(S)) / len(S)
G = getG(S, alpha, v)
P, step, evolution = surf(G,v,epsilon)

# Plotting results
plt.subplot(121)
pokeplot(types, A, P)
p4.vbar(x= [x - 0.1 for x in types.keys()], top=P, bottom=0, color="green", width=0.20, legend_label=f"PageRank vector")

D,_ = np.linalg.eig(G)
D = np.absolute(D)
D = D/np.sum(D)

save_result("3-pokemon-pagerank.csv")

#__________________________________________________
# Cheirank

S = getS(matrix)
alpha = 0.85
v = np.ones(len(S)) / len(S)
G = getG(S, alpha, v)
P, step, evolution = surf(G,v,epsilon)

# Plotting results
plt.subplot(122)
pokeplot(types, A, P)
p4.vbar(x= [x + 0.1 for x in types.keys()], top=P, bottom=0, color="blue", width=0.20, legend_label=f"CheiRank vector")

D,_ = np.linalg.eig(G)

D = np.absolute(D)
D = D/np.sum(D)

save_result("3-pokemon-cheirank.csv")


#__________________________________________________
# Plot

print("P = ")
print(P)
print("P.argsort() = ")
print(P.argsort())

p5.vbar([x for x in types.keys()], top=P, color="red", width=0.2, legend_label="")
show(p5)

show(p4)
plt.show()