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


def kinetic_curves(adata, genes, layer='X', time='pseudotime', color_map='viridis'):
    pass


def kinetic_heatmap(adata, genes, layer='X', time='pseudotime', color_map='viridis', show_col_color=False, \
                    cluster_row_col=(False, False), figsize=(11.5, 6)):
    """Plot the gene expression dynamics over time (pseudotime or inferred real time).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        genes: `list`
            The gene names whose gene expression will be faceted.
        layer: `str` (default: X)
            Which layer of expression value will be used.
        time: `str` (default: `pseudotime`)
            The .obs column that will be used for timing each cell.
        color_map: `str` (default: `viridis`)
            Color map that will be used to color the gene expression.
        show_col_color: `bool` (default: `False`)
            Whether to show the color bar.
        cluster_row_col: `(bool, bool)` (default: `[False, False]`)
            Whether to cluster the row or columns.
        figsize: `str` (default: `(11.5, 6)`
            Size of figure

    Returns
    -------
        Nothing but plots the a heatmap that shows the gene expression dynamics over time.
    """

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    valid_genes = list(set(genes).intersection(adata.var.index))

    time = adata.obs[time].values
    time = time[np.isfinite(time)]

    if layer is 'X':
        exprs = adata[:, adata.var.index.isin(valid_genes)].X.T
    elif layer in adata.layers.keys():
        exprs = adata[:, adata.var.index.isin(valid_genes)].layers[layer].T
    elif layer is 'protein': # update subset here
        exprs = adata[:, adata.var.index.isin(valid_genes)].obsm[layer].T
    else:
        raise Exception(f'The {layer} you passed in is not existed in the adata object.')

    time, all=_ident_switch_point(exprs, time, interpolate=True, spaced_num=100)
    df = pd.DataFrame(all, index=valid_genes)

    sns_heatmap = sns.clustermap(df, col_colors=col_color, col_cluster=cluster_row_col[0], row_cluster=cluster_row_col[1], cmap=color_map, \
                        xticklabels=False, figsize=figsize)
    if not show_col_color: sns_heatmap.set_visible(False)

    plt.show()


def _ident_switch_point(exprs, time, interpolate=False, spaced_num=100):
    """Implement the half-max ordering algorithm from HA Pliner, Molecular Cell, 2018.

    Parameters
    ----------
        exprs: `np.ndarray`
            The gene expression matrix (ngenes x ncells) ordered along time (either pseudotime or inferred real time).
        time: `np.ndarray`
            Pseudotime or inferred real time.
        interpolate: `bool` (default: `False`)
            Whether to interpolate the data when performing the loess fitting.
        spaced_num: `float` (default: `100`)
            The number of points on the loess fitting curve.

    Returns
    -------
        all: `np.ndarray`
            The ordered smoothed, scaled expression matrix.
    """

    from .utilities import Loess
    gene_num = exprs.shape[0]
    cell_num = exprs.shape[1] if interpolate else spaced_num
    if interpolate:
        hm_mat_scaled, hm_mat_scaled_z = np.zeros((gene_num, cell_num)), np.zeros(gene_num, cell_num)
    else:
        hm_mat_scaled, hm_mat_scaled_z = np.zeros_like(exprs), np.zeros_like(exprs)

    transient, trans_max, half_max = np.zero(gene_num), np.zero(gene_num), np.zero(gene_num)
    for i in range(gene_num):
        x = exprs[i]
        x_rng = [np.min(x), np.max(x)]
        norm_x = (x - x_rng[0]) / np.diff(x_rng)
        loess = Loess(time, norm_x)

        tmp = np.zeros(cell_num)

        if interpolate:
            time = np.linspace(np.min(time), np.max(time), spaced_num)
            for j in range(cell_num):
                tmp[j] = loess.estimate(time[j], window=7, use_matrix=False, degree=1)
        else:
            for j in range(cell_num):
                tmp[j] = loess.estimate(time[j], window=7, use_matrix=False, degree=1)

        hm_mat_scaled[i] = tmp
        scale_tmp = (tmp - np.mean(tmp)) / np.std(tmp)
        hm_mat_scaled_z[i] = scale_tmp

        count, current = 0, hm_mat_scaled_z[i, j] < 0
        for j in range(cell_num):
            if not (scale_tmp[j] < 0 == current):
                count = count + 1
                current = scale_tmp[j] < 0

        half_max[i] = np.argmax(np.abs(scale_tmp - 0.5))
        transient[i] = count
        trans_max[i] = np.argsort(-scale_tmp)[0]

    begin = np.arange(max([5, 0.05 * cell_num]))
    end = np.arange(exprs.shape[1] - max([5, 0.05 * cell_num]), cell_num)
    trans_indx = np.logical_and(transient > 1, ~ trans_max in np.concatenate((begin, end)))

    trans, half_max_trans = hm_mat_scaled[trans_indx, :], half_max[trans_indx]
    nt = hm_mat_scaled[~trans_indx, :]
    up, half_max_up = nt[nt[:, 0] < nt[:, nt.shape[1]], :], half_max[nt[:, 0] < nt[:, nt.shape[1]]]
    down, half_max_down = nt[nt[:, 0] > nt[:, nt.shape[1]], :], half_max[nt[:, 0] > nt[:, nt.shape[1]]]

    trans, up, down = trans[np.argsort(half_max_trans), :], up[np.argsort(half_max_up), :], down[np.argsort(half_max_down), :]

    all = np.vstack((up, down, trans))

    return time, all
