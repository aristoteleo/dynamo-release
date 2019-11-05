import numpy as np

def plot_direct_graph(adata, layout=None, figsize=[8, 8]):
    df_mat = adata.uns['df_mat']

    import matplotlib.pyplot as plt
    import networkx as nx

    edge_color = 'gray'

    G = nx.from_pandas_edgelist(df_mat, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph())
    G.nodes()
    W = []
    for n, nbrs in G.adj.items():
        for nbr, eattr in nbrs.items():
            W.append(eattr['weight'])

    options = {
        'width': 300,
        'arrowstyle': '-|>',
        'arrowsize': 1000,
    }

    plt.figure(figsize=figsize)
    if layout is None:
        #     pos : dictionary, optional
        #        A dictionary with nodes as keys and positions as values.
        #        If not specified a spring layout positioning will be computed.
        #        See :py:mod:`networkx.drawing.layout` for functions that
        #        compute node positions.

        nx.draw(G, with_labels=True, node_color='skyblue', node_size=100, edge_color=edge_color,
                width=W / np.max(W) * 5, edge_cmap=plt.cm.Blues, options=options)
    else:
        raise Exception('layout', layout, ' is not supported.')
    plt.show()
