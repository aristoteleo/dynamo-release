# include pseudotime and predict cell trajectory
import numpy as np

def plot_directed_pg(adata, principal_g_transition, Y, basis='umap'):
    """

    Parameters
    ----------
    principal_g_transition

    Returns
    -------

    """

    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.from_numpy_matrix(principal_g_transition, create_using=nx.DiGraph())
    W = []
    for n, nbrs in G.adj.items():
        for nbr, eattr in nbrs.items():
            W.append(eattr['weight'])

    options = {
        'width': 30,
        'arrowstyle': '-|>',
        'arrowsize': 10,
    }
    edge_color = 'gray'
    plt.figure(figsize=[10, 10])

    nx.draw(G, pos=Y, with_labels=False, node_color='skyblue', node_size=1,
            edge_color=edge_color, width=W / np.max(W) * 1, edge_cmap=plt.cm.Blues, options=options)

    plt.show()
