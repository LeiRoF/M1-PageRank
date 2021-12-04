import enum
import numpy as np
import networkx as nx
from bokeh.plotting import figure, show, output_file
import os
import matplotlib.pyplot as plt
import time

def pokeplot(types, A, P):
    """
    This function allow to plot the network of the pokemon interactions.
    """

    # Types colors
    color = ["grey","red","blue","green","blue","aqua","red","green","grey","grey","purple","green","grey","purple","red","grey","grey"]
    node_colors = {}
    for i in range(len(color)):
        node_colors.update({i+1 : color[i]})

    Graph = nx.Graph()

    # Node creation
    for i, value in enumerate(types.values()):
        Graph.add_node(i, name=value, color=color[i])
    Graph.remove_node(0)

    # Link creation with color associated to the strenght of the interaction
    edge_color = []
    for i, attack in enumerate(A):
        for j, strength in enumerate(attack):
            if strength == 0.5:
                Graph.add_edge(i+1, j+1, color="red")
                edge_color.append("red")
            if strength == 1:
                Graph.add_edge(i+1, j+1, color="black")
                edge_color.append("black")
            if strength == 2:
                Graph.add_edge(i+1, j+1, color="green")
                edge_color.append("green")

    pos = nx.circular_layout(Graph)

    # Size of the displayed node, depending on the final value of the pagerank/cheirank vector.
    size = [(x - min(P)) / (max(P)-min(P)) * 5000 for x in P]

    nx.draw(Graph, pos,labels = types, node_color=color, node_size=size,edge_color=edge_color, font_weight='bold', arrows=True)