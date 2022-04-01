import networkx as nx

def print_graph(graph):
    '''
    A function that displays a graph (a small one though) with
    networkx's built-in method ```networkx.draw```
    '''
    nx.draw(graph)
    