# include pseudotime and predict cell trajectory
import numpy as np
from scipy.sparse import issparse


def kinetic_curves(adata, genes, mode='vector_field', basis='X', project_back_to_high_dim=False, layer='X', time='pseudotime', \
                   dist_threshold=1e-10, ncol=4, color=None, c_palette='Set2'):
    """Plot the gene expression dynamics over time (pseudotime or inferred real time) as kinetic curves.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        genes: `list`
            The gene names whose gene expression will be faceted.
        mode: `str` (default: `vector_field`)
            Which data mode will be used, either vector_field or pseudotime. if mode is vector_field, the trajectory predicted by
            vector field function will be used, otherwise pseudotime trajectory (defined by time argument) will be used.
        basis: `str` (default: `X`)
            The embedding data used for drawing the kinetic gene expression curves, only used when mode is `vector_field`.
        project_back_to_high_dim: `bool` (default: `False`)
            Whether to map the coordinates in low dimension back to high dimension to visualize the gene expression curves,
            only used when mode is `vector_field` and basis is not `X`. Currently only works when basis is 'pca' and 'umap'.
        color: `list` or None (default: None)
            A list of attributes of cells (column names in the adata.obs) will be used to color cells.
        layer: `str` (default: X)
            Which layer of expression value will be used. Not used if mode is `vector_field`.
        time: `str` (default: `pseudotime`)
            The .obs column that will be used for timing each cell, only used when mode is `vector_field`.
        dist_threshold: `float` or None (default: 1e-10)
            The threshold for the distance between two points in the gene expression state, i.e, x(t), x(t+1). If below this threshold,
            we assume steady state is achieved and those data points will not be considered.
        ncol: `int` (default: 4)
            Number of columns in each facet grid.
        c_palette: Name of color_palette supported in seaborn color_palette function (default: None)
            The color map function to use.
    Returns
    -------
        Nothing but plots the kinetic curves that shows the gene expression dynamics over time.
    """

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    valid_genes = list(set(genes).intersection(adata.var.index))

    time = adata.obs[time].values if mode is not 'vector_field' else adata.uns['Fate']['t']
    time = time[np.isfinite(time)]

    if mode is not 'vector_field':
        if layer is 'X':
            exprs = adata[:, adata.var.index.isin(valid_genes)].X
        elif layer in adata.layers.keys():
            exprs = adata[:, adata.var.index.isin(valid_genes)].layers[layer]
        elif layer is 'protein': # update subset here
            exprs = adata[:, adata.var.index.isin(valid_genes)].obsm[layer]
        else:
            raise Exception(f'The {layer} you passed in is not existed in the adata object.')
    else:
        if basis is 'X':
            exprs = adata.uns['Fate']['prediction'][:, adata.var.index.isin(valid_genes)]
        else:
            exprs = adata.uns['Fate_' + basis]['prediction']
            if project_back_to_high_dim is False:
                valid_genes = [basis + '_' + str(i) for i in np.arange(exprs.shape[1])]
            else:
                exprs = adata.uns[basis + '_fit'].inverse_transform(exprs)

    Color = np.empty((0, 1))
    if color is not None and mode is not 'vector_field':
        color = list(set(color).intersection(adata.obs.keys()))
        Color = adata.obs[color].values.T.flatten() if len(color) > 0 else np.empty((0, 1))

    exprs = exprs.A if issparse(exprs) else exprs
    # time = np.sort(time)
    # exprs = exprs[np.argsort(time), :]

    if dist_threshold is not None:
        valid_ind = list(np.where(np.sum(np.diff(exprs, axis=0) ** 2, axis=1) > dist_threshold)[0] + 1)
        valid_ind.insert(0, 0)
        exprs = exprs[valid_ind, :]
        time = time[valid_ind]

    exprs_df = pd.DataFrame({'Time': np.repeat(time, len(valid_genes)), 'Expression': exprs.flatten(), \
                             'Gene': np.tile(valid_genes, exprs.shape[0])})


    # https://stackoverflow.com/questions/43920341/python-seaborn-facetgrid-change-titles
    if len(Color) > 0:
        exprs_df['Color'] = np.repeat(Color, len(valid_genes))
        g = sns.lmplot(x="Time", y="Expression", data=exprs_df, order=3, col='Gene', hue='Color', palette=sns.color_palette(c_palette), \
                   col_wrap=ncol, line_kws={"color": "gray"},  scatter_kws={"s": 3})
    else:
        g = sns.lmplot(x="Time", y="Expression", data=exprs_df, order=3, col='Gene', col_wrap=ncol, line_kws={"color": "gray"}, \
                   scatter_kws={"s": 3})

    g.set(xlim=(np.min(time), np.max(time)))

    plt.show()


def kinetic_heatmap(adata, genes, mode='vector_field', basis='X', project_back_to_high_dim=False, layer='X', time='pseudotime',
                    color_map='viridis', half_max_ordering=True, show_col_color=False, dist_threshold=1e-10,
                    cluster_row_col=(False, False), figsize=(11.5, 6), **kwargs):
    """Plot the gene expression dynamics over time (pseudotime or inferred real time) in a heatmap.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        genes: `list`
            The gene names whose gene expression will be faceted.
        mode: `str` (default: `vector_field`)
            Which data mode will be used, either vector_field or pseudotime. if mode is vector_field, the trajectory predicted by
            vector field function will be used, otherwise pseudotime trajectory (defined by time argument) will be used.
        basis: `str` (default: `X`)
            The embedding data used for drawing the kinetic gene expression heatmap, only used when mode is `vector_field`.
        project_back_to_high_dim: `bool` (default: `False`)
            Whether to map the coordinates in low dimension back to high dimension to visualize the gene expression curves,
            only used when mode is `vector_field` and basis is not `X`. Currently only works when basis is 'pca' and 'umap'.
        layer: `str` (default: X)
            Which layer of expression value will be used.
        time: `str` (default: `pseudotime`)
            The .obs column that will be used for timing each cell.
        color_map: `str` (default: `viridis`)
            Color map that will be used to color the gene expression.
        half_max_ordering: `bool` (default: `True`)
            Whether to order genes into up, down and transit groups by the half max ordering algorithm. 
        show_col_color: `bool` (default: `False`)
            Whether to show the color bar.
        dist_threshold: `float` or None (default: 1e-10)
            The threshold for the distance between two points in the gene expression state, i.e, x(t), x(t+1). If below this threshold,
            we assume steady state is achieved and those data points will not be considered.
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

    time = adata.obs[time].values if mode is not 'vector_field' else adata.uns['Fate']['t']
    time = time[np.isfinite(time)]

    if mode is not 'vector_field':
        if layer is 'X':
            exprs = adata[:, adata.var.index.isin(valid_genes)].X.T
        elif layer in adata.layers.keys():
            exprs = adata[:, adata.var.index.isin(valid_genes)].layers[layer].T
        elif layer is 'protein': # update subset here
            exprs = adata[:, adata.var.index.isin(valid_genes)].obsm[layer].T
        else:
            raise Exception(f'The {layer} you passed in is not existed in the adata object.')
    else:
        if basis is 'X':
            exprs = adata.uns['Fate']['prediction'][:, adata.var.index.isin(valid_genes)].T
        else:
            exprs = adata.uns['Fate_' + basis]['prediction'].T
            if project_back_to_high_dim is False:
                valid_genes = [basis + '_' + str(i) for i in np.arange(exprs.shape[1])]
            else:
                exprs = adata.uns[basis + '_fit'].inverse_transform(exprs)

    exprs = exprs.A if issparse(exprs) else exprs

    # time = np.sort(time)
    # exprs = exprs[np.argsort(time), :]
    if dist_threshold is not None:
        valid_ind = list(np.where(np.sum(np.diff(exprs, axis=0) ** 2, axis=1) > dist_threshold)[0] + 1)
        valid_ind.insert(0, 0)
        exprs = exprs[valid_ind, :]

    if half_max_ordering:
        time, all, valid_ind =_half_max_ordering(exprs, time, interpolate=True, spaced_num=100)
        df = pd.DataFrame(all, index=np.array(valid_genes)[valid_ind])
    else:
        df = pd.DataFrame(exprs, index=np.array(valid_genes))

    heatmap_kwargs = dict(xticklabels=False, yticklabels='auto')
    if kwargs is not None:
        heatmap_kwargs.update(kwargs)

    sns_heatmap = sns.clustermap(df, col_cluster=cluster_row_col[0], row_cluster=cluster_row_col[1], cmap=color_map, \
                        figsize=figsize, **heatmap_kwargs)
    # if not show_col_color: sns_heatmap.set_visible(False)

    plt.show()


def _half_max_ordering(exprs, time, interpolate=False, spaced_num=100):
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
        time: `np.ndarray`
            The time at which the loess is evaluated.
        all: `np.ndarray`
            The ordered smoothed, scaled expression matrix, the first group is up, then down, followed by the transient gene groups.
        valid_ind: `np.ndarray`
            The indices of valid genes that Loess smoothed.
    """

    from .utils import Loess
    gene_num = exprs.shape[0]
    cell_num = spaced_num if interpolate else exprs.shape[1]
    if interpolate:
        hm_mat_scaled, hm_mat_scaled_z = np.zeros((gene_num, cell_num)), np.zeros((gene_num, cell_num))
    else:
        hm_mat_scaled, hm_mat_scaled_z = np.zeros_like(exprs), np.zeros_like(exprs)

    transient, trans_max, half_max = np.zeros(gene_num), np.zeros(gene_num), np.zeros(gene_num)
    for i in range(gene_num):
        x = exprs[i]
        x_rng = [np.min(x), np.max(x)]
        norm_x = (x - x_rng[0]) / np.diff(x_rng)
        loess = Loess(time, norm_x)

        tmp = np.zeros(cell_num)

        if interpolate:
            time = np.linspace(np.min(time), np.max(time), spaced_num)
            for j in range(spaced_num):
                tmp[j] = loess.estimate(time[j], window=7, use_matrix=False, degree=1)
        else:
            for j in range(cell_num):
                tmp[j] = loess.estimate(time[j], window=7, use_matrix=False, degree=1)

        hm_mat_scaled[i] = tmp
        scale_tmp = (tmp - np.mean(tmp)) / np.std(tmp)
        hm_mat_scaled_z[i] = scale_tmp

        count, current = 0, hm_mat_scaled_z[i, 0] < 0 # check this
        for j in range(cell_num):
            if not (scale_tmp[j] < 0 == current):
                count = count + 1
                current = scale_tmp[j] < 0

        half_max[i] = np.argmax(np.abs(scale_tmp - 0.5))
        transient[i] = count
        trans_max[i] = np.argsort(-scale_tmp)[0]

    begin = np.arange(max([5, 0.05 * cell_num]))
    end = np.arange(exprs.shape[1] - max([5, 0.05 * cell_num]), cell_num)
    trans_indx = np.logical_and(transient > 1, not [i in np.concatenate((begin, end)) for i in trans_max])

    trans, half_max_trans = hm_mat_scaled[trans_indx, :], half_max[trans_indx]
    nt = hm_mat_scaled[~trans_indx, :]
    up, half_max_up = nt[nt[:, 0] < nt[:, -1], :], half_max[nt[:, 0] < nt[:, -1]]
    down, half_max_down = nt[nt[:, 0] >= nt[:, -1], :], half_max[nt[:, 0] >= nt[:, -1]]

    trans, up, down = trans[np.argsort(half_max_trans), :], up[np.argsort(half_max_up), :], down[np.argsort(half_max_down), :]

    all = np.vstack((up, down, trans))

    return time, all, np.isfinite(nt[:, 0]) & np.isfinite(nt[:, -1])
