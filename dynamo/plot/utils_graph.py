import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.patches import ConnectionPatch

import numpy as np

def create_edge_patch(posA, 
                    posB, 
                    node_rad=0, 
                    connectionstyle='arc3, rad=0.25', 
                    facecolor='k',
                    head_length=10,
                    head_width=10,
                    tail_width=3, 
                    **kwargs):
    style = "simple,head_length=%d,head_width=%d,tail_width=%d"%(head_length, head_width, tail_width)
    return pat.FancyArrowPatch(posA=posA, posB=posB, arrowstyle=style, connectionstyle=connectionstyle, facecolor=facecolor, shrinkA=node_rad, shrinkB=node_rad, **kwargs)


def create_edge_patches_from_markov_chain(P, X, width=3, node_rad=0, tol=1e-7, connectionstyle='arc3, rad=0.25', facecolor='k', **kwargs):
    arrows = []
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            if P[j, i] > tol:
                arrows.append(create_edge_patch(X[i], X[j], tail_width=P[j, i]*width, node_rad=node_rad, connectionstyle=connectionstyle, facecolor=facecolor, alpha=min(2*P[j, i], 1), **kwargs))
    return arrows


def plot_alternate_function(X, E, arrowstype='-|>', node_rad=5, arrow_size=10, fc='w'):
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if E[i, j] != 0:
                xa = X[i] if E[i, j] > 0 else X[j]
                xb = X[j] if E[i, j] > 0 else X[i]
                con = ConnectionPatch(xa, xb, 'data', 'data',
                      arrowstyle=arrowstype, shrinkA=node_rad, shrinkB=node_rad,
                      mutation_scale=arrow_size, fc=fc)
                plt.gca().add_artist(con)


def arcplot(x, 
            E, 
            node_names=None,
            edge_threshold=0, 
            curve_radius=0.7, 
            node_rad=10, 
            width=1, 
            arrow_head=True,
            curve_alpha=1, 
            arrow_direction=-1,
            node_name_rotation=-45,
            **kwargs):
    X = np.vstack((x, np.zeros(len(x)))).T
    plt.scatter(X[:, 0], X[:, 1], **kwargs)

    # calculate alpha
    alp_scale = np.max(E)

    # arrow style
    curve_radius = np.abs(curve_radius) * arrow_direction
    cstyle = 'arc3, rad=%f'%curve_radius
    head_width = width * 5 if arrow_head else 0

    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if i != j and E[i, j] > edge_threshold:
                posA, posB = X[i], X[j]
                arrow = create_edge_patch(
                        posA, posB, 
                        node_rad=node_rad, 
                        head_width=head_width, 
                        tail_width=width, 
                        connectionstyle=cstyle, 
                        alpha=E[i, j] / alp_scale * curve_alpha)
                plt.gca().add_patch(arrow)
    
    plt.gca().get_yaxis().set_ticks([])

    if node_names is not None:
        plt.xticks(x, node_names, rotation=node_name_rotation)


class ArcPlot:
    def __init__(self,
                x=None, 
                E=None,
                network=None,
                node_names=None,
                **kwargs):
        self.x = x if x is not None else None
        self.E = E if E is not None else None
        self.node_names = node_names if node_names is not None else None
        if network is not None:
            self.set_network(network)
        self.kwargs = kwargs

    def set_network(self, network):
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(f"You need to install the packages `networkx`."
                            f"install networkx via `pip install networkx`.")
        self.E = nx.to_numpy_matrix(network)
        self.node_names = list(network.nodes)

    def compute_node_positions(self, node_order=None):
        if self.E is None:
            raise Exception('The adjacency matrix is not set.')
        
        if node_order is None:
            self.x = np.arange(self.E.shape[0])
        else:
            self.x = np.argsort(node_order)

    def draw(self, node_order=None):
        if self.x is None:
            self.compute_node_positions(node_order=node_order)
        arcplot(self.x, self.E, node_names=self.node_names, **self.kwargs)