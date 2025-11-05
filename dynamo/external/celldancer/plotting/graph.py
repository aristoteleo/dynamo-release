import os
import networkx as nx
import pandas as pd
import numpy as np
from datashader.layout import forceatlas2_layout
from datashader.bundling import hammer_bundle, connect_edges
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from .colormap import *
from ..utilities import extract_from_df

def PTO_Graph(
        ax,
        cellDancer_df,
        node_layout='forceatlas2',
        PRNG_SEED=None,
        force_iters=2000,
        use_edge_bundling=True,
        node_colors=None,
        node_sizes=5,
        edge_length=None,
        legend='off',
        colorbar='on'):

    """ 
    Graph visualization of selected cells reflecting their orders in
    pseudotime (PseudoTimeOrdered_Graph: PTO_Graph). Embedding and pseudotime 
    of the cells are required. Each cell makes a node and the connections between 
    nodes are based on their separation in the embedding space and the strength 
    of the connection is proportional to the pseudotime difference (the larger 
    the pseudotime difference in absolute values, the weaker the connection).

    Example usage:

    .. code-block:: python

        from celldancer.plotting import graph
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=(10,10))
        graph.PTO_Graph(ax, 
            load_cellDancer, 
            node_layout='forcedirected', 
            use_edge_bundling=True, 
            node_colors='clusters', 
            edge_length=3, 
            node_sizes='pseudotime', 
            colorbar='on',
            legend='on')
        
    In this example, we use a force-directed node layout algorithm (`ForceAtlas2 
    <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679>`_).
    A connection is made between any two cells within 3 (unit in the embedding).
    The resulted edge lengths indicate the time difference between nodes (the
    closer in pseudotime, the shorter the edge length). Edge bundling is applied
    to highlight important edges (trunks). The sizes of the nodes are
    proportional to the pseudotime. The nodes are colored according to their
    cell types (if given by the input data). 

    Arguments
    ---------
    cellDancer_df: `pandas.DataFrame`
        Dataframe of velocity estimation, cell velocity, and pseudotime results. 
        Columns=['cellIndex', 'gene_name', 
        'unsplice', 'splice', 
        'unsplice_predict', 'splice_predict', 
        'alpha', 'beta', 'gamma', 
        'loss', 'cellID', 'clusters', 'embedding1', 'embedding2', 
        'velocity1', 'velocity2', 'pseudotime']

    node_layout: optional, `str` (default: forceatlas2)
         Layout for the graph. Currently only supports the forceatlas2 and
         embedding. 

         - `'forceatlas2'` or `'forcedirected'`: treat connections as forces
         between connected nodes.

         - `'embedding'`: use the embedding as positions of the nodes.

    PRNG_SEED: optional, `int`, or `None` (default: `None`)
        Seed to initialize the pseudo-random number generator.

    force_iters: optional, `int` (default: 2000)
        Number of passes for the force-directed layout calculation.

    use_edge_bundling: optional, `bool` (default: `True`)
        `True` if bundle the edges (computational demanding). 
        Edge bundling allows edges to curve and groups nearby ones together 
        for better visualization of the graph structure. 

    node_colors: optional, `str` (default: `None`)
        The node fill colors. 
        Possible values:

            - *clusters*: color according to the clusters information of the
              respective cells.

            - *pseudotime*: colors according to the pseudotime of the 
              respective cells.

            - A single color format string.

    edge_length: optional, `float` (default: `None`)
        The distance cutoff in the embedding between two nodes to determine 
        whether an edge should be formed (edge is formed when r < *edge_length*).
        By default, the mean of all the cell
        
    node_sizes: optional, `float` or `numeric list-like` or `str` (default: 5)
        The sizes of the nodes. If it is `str`, then the `str` has to be either one of those
        {`pseudotime`, `index`, `x`, `y`} read from the `nodes` dataframe.

    legend: optional, `str` (default: 'off')
        - `'off'`/`'on'`: Exclude/include the cell type legend on the plot. 
        - `'only'`: Negelect the plot and only show the cell type legend.

    colorbar: optional, `str` (default: 'on')
        - `'off'`/`'on'`: Show the colorbar in the case nodes are colored by `pseudotime`.
        

    Returns
    -------
    ax: matplotlib.axes.Axes

    """  

    nodes, edges = create_nodes_edges(cellDancer_df, edge_length)

    if node_layout in ['forceatlas2', 'forcedirected']:
        # Current version of datashader.layout does not support reading a layout (x,y) and perform layout function
        # It does not support other attributes except index.
        forcedirected = forceatlas2_layout(nodes[['index']], edges,
                weight='weight', iterations=force_iters, k=0.1, seed=PRNG_SEED)
        nodes['x'] = forcedirected['x']
        nodes['y'] = forcedirected['y']

    if use_edge_bundling:
        bundle = hammer_bundle(nodes, edges)
    else:
        bundle = connect_edges(nodes, edges)


    # For plotting settings
    def gen_Line2D(label, markerfacecolor, markersize):
        return Line2D([0], [0], color='w', 
            marker='o', 
            label=label,
            markerfacecolor=markerfacecolor,
            markeredgewidth=0,
            markersize=markersize)

    if isinstance(node_sizes, (int, float)) or isinstance(node_sizes, list):
        pass
    elif isinstance(node_sizes, str):
        node_sizes=nodes[node_sizes].to_numpy(dtype=float)*200
    
    if isinstance(node_colors, str):
        # This goes to dict case afterwards
        if node_colors in ['clusters']:
            node_colors = build_colormap(nodes[node_colors])
        if node_colors in ['pseudotime']:
            cmap='viridis'
            c=nodes[node_colors].to_numpy(dtype=float)

    if isinstance(node_colors, dict):
        legend_elements= [gen_Line2D(i, 
                    node_colors[i], 
                    10) 
                    for i in node_colors]

        if legend != 'off':
            lgd=ax.legend(handles=legend_elements,
                bbox_to_anchor=(1.01, 1),
                loc='upper left')
            bbox_extra_artists=(lgd,)
            if legend == 'only':
                return lgd
        else:
            bbox_extra_artists=None

        c=nodes['clusters'].map(node_colors).to_list()
        cmap=ListedColormap(list(node_colors.values()))

    if node_colors is None:
        c = ['Grey']*len(nodes)

    ax.plot(bundle.x, bundle.y, 'y', zorder=1, linewidth=0.3, color='blue', alpha=1)
    im = ax.scatter(nodes.x, nodes.y, c=c, cmap=cmap, s=node_sizes, zorder=2, edgecolors='k', alpha=0.5)
    
    if colorbar == 'on' and isinstance(node_colors, str):
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("top", size="5%", pad="-5%")
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal", shrink=0.1)
        cbar.set_ticks([])
    ax.axis('off')

    return ax



def create_nodes_edges(data, radius):
    def create_KNN_based_graph():
        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(radius = radius)
        neigh.fit(embedding_ds)
        nn_graph = neigh.radius_neighbors_graph(embedding_ds, mode='connectivity')
        nn_array = nn_graph.toarray()

        # nn_array is effectively the edge list
        # Keep track of cells of 0 timeshift.
        node_list = [(i, {'pseudotime': pseudotime_ds[i,0], 'clusters':clusters_ds[i]})
                     for i in range(len(embedding_ds))]

        dtime = pseudotime_ds[:,0] - pseudotime_ds
        INF = 1./np.min(np.abs(dtime[dtime != 0]))

        # upper triangle of the knn array (i<j and nn_array[i,j] = 1)
        edge_filter = np.triu(nn_array, k=1)
        (i,j) = np.where(edge_filter != 0)

        # for forcedirected layouts,
        # edge length is positively correlated with weight.
        # hence 1/dtime here as the weight
        # Created for directed graph
        edge_list = list()
        for a,b,w in zip(i,j, dtime[i,j]):
            if w>0:
                edge_list.append((a, b, 1/w))
            elif w<0:
                edge_list.append((a, b, -1/w))
            else:
                edge_list.append((a, b, INF))

        G = nx.Graph()
        G.add_nodes_from(node_list)
        G.add_weighted_edges_from(edge_list)
        return G

    embedding = extract_from_df(data, ['embedding1', 'embedding2'])
    n_cells = embedding.shape[0]
    sample_cells = data['velocity1'][:n_cells].dropna().index
    clusters = extract_from_df(data, ['clusters'])
    pseudotime = extract_from_df(data, ['pseudotime'])

    embedding_ds = embedding[sample_cells]
    pseudotime_ds = pseudotime[sample_cells]
    clusters_ds = clusters[sample_cells]

    G = create_KNN_based_graph()

    index = np.array(range(len(embedding_ds)), dtype=int)[:,None]
    nodes = pd.DataFrame(np.hstack((embedding_ds, index, pseudotime_ds, clusters_ds)),
                         columns=['x','y','index','pseudotime','clusters'])

    edges = pd.DataFrame([(i[0], i[1], G.edges[i]['weight']) for i in G.edges],
                         columns=['source', 'target', 'weight'])
    return nodes, edges
