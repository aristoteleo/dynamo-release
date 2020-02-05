import numpy as np
import pandas as pd
from scipy.sparse import issparse

from .scatters import scatters
from .utils import quiver_autoscaler
from ..tools.dimension_reduction import reduceDimension
from ..tools.cell_velocities import cell_velocities
from ..tools.Markov import velocity_on_grid
from ..tools.scVectorField import VectorField

from .scatters import docstrings

docstrings.delete_params('scatters.parameters', 'show_legend', 'kwargs')

import scipy as sc
#from licpy.lic import runlic

# moran'I on the velocity genes, etc.

# cellranger data, velocyto, comparison and phase diagram

def _cell_wise_velocity(adata, genes, x=0, y=1, basis='trimap', n_columns=1, color=None, label_on_embedding=True,
                       cmap=None, s_kwargs_dict={}, layer='X', cell_ind='all', quiver_scale=None, figsize=None,
                       **q_kwargs):
    """Plot the velocity vector of each cell.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        genes: `list`
            The gene names whose gene expression will be faceted.
        x: `int` (default: `0`)
            The column index of the low dimensional embedding for the x-axis
        y: `int` (default: `1`)
            The column index of the low dimensional embedding for the y-axis
        basis: `str` (default: `trimap`)
            The reduced dimension embedding of cells to visualize.
        n_columns: `int  (default: 1)
            The number of columns of the resulting plot.
        color: `list` or None (default: None)
            A list of attributes of cells (column names in the adata.obs) will be used to color cells.
        cmap: `plt.cm` or None (default: None)
            The color map function to use.
        s_kwargs_dict: `dict` (default: {})
            The dictionary of the scatter arguments.
        layer: `str` (default: X)
            Which layer of expression value will be used.
        cell_ind: `str` or `list` (default: all)
            the cell index that will be chosen to draw velocity vectors.
        quiver_scale: `float` or None (default: None)
            scale of quiver plot (default: None). Number of data units per arrow length unit, e.g., m/s per plot width;
            a smaller scale parameter makes the arrow longer. If None, we will use quiver_autoscaler to calculate the scale.
        figsize: `None` or `[float, float]` (default: None)
            The width and height of a figure.
        q_kwargs:
            Additional parameters that will be passed to plt.quiver function

    Returns
    -------
        Nothing but a cell wise quiver plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as PathEffects

    if cmap is None and color is None:
        cmap = plt.cm.RdBu_r

    n_cells, n_genes = adata.shape[0], len(genes)
    # {"alpha": 0.5, "s": 8, "edgecolor": "0.8", "lw": 0.15}
    if cell_ind is "all":
        ix_choice = np.arange(adata.shape[0])
    elif cell_ind is 'random':
        ix_choice = np.random.choice(np.range(adata.shape[0]), size=1000, replace=False)
    elif type(cell_ind) is int:
        ix_choice = np.random.choice(np.range(adata.shape[0]), size=cell_ind, replace=False)
    elif type(cell_ind) is list:
        ix_choice = cell_ind

    scatter_kwargs = dict(alpha=0.4, s=8, edgecolor=None, linewidth=0)
    if s_kwargs_dict is not None:
        scatter_kwargs.update(s_kwargs_dict)

    # layer_keys = list(adata.layers.keys())
    # layer_keys.extend(['X', 'protein'])

    if layer is 'X':
        E_vec = adata[:, adata.var.index.isin(genes)].X.T
    elif layer in adata.layers.keys():
        E_vec = adata[:, adata.var.index.isin(genes)].layers[layer].T
    elif layer is 'protein': # update subset here
        E_vec = adata[:, adata.var.index.isin(genes)].obsm[layer].T
    else:
        raise Exception(f'The {layer} you passed in is not existed in the adata object.')

    if color is not None:
        color = list(set(color).intersection(adata.obs.keys()))
        n_genes, genes = len(color), color
        E_vec = adata.obs[color].values.T.flatten() if len(color) > 0 else np.empty((0, 1))

    if ('X_' + basis in adata.obsm.keys()) and ('velocity_' + basis in adata.obsm.keys()):
        X = adata.obsm['X_' + basis][:, [x, y]]
        V = adata.obsm['velocity_' + basis][:, [x, y]]
    else:
        if 'X_' + basis not in adata.obsm.keys():
            reduceDimension(adata, velocity_key='velocity_S', reduction_method=basis)
        if 'kmc' not in adata.uns_keys():
            cell_velocities(adata, vkey='velocity_S', basis=basis, method='analytical')
            X = adata.obsm['X_' + basis][:, [x, y]]
            V = adata.obsm['velocity_' + basis][:, [x, y]]
        else:
            kmc = adata.uns['kmc']
            X = adata.obsm['X_' + basis][:, [x, y]]
            V = kmc.compute_density_corrected_drift(X, kmc.Idx, normalize_vector=True)
            adata.obsm['velocity_' + basis] = V

    if 0 in E_vec.shape:
        raise Exception(f'The gene names {genes} (or cell annotations {color}) provided are not existed in your data.')

    if quiver_scale is None:
        quiver_scale = quiver_autoscaler(X, V)
    quiver_kwargs = {"angles": 'xy', "scale_units": 'xy', 'scale': quiver_scale, "minlength": 1.5, "alpha": 0.4}
    quiver_kwargs.update(q_kwargs)

    n_columns, plot_per_gene = n_columns, 1 # we may also add random velocity results
    nrow, ncol = int(np.ceil(plot_per_gene * n_genes / n_columns)), n_columns
    if figsize is None:
        plt.figure(None, (ncol * 3, nrow * 3), dpi=160)
    else:
        plt.figure(None, (figsize[0]*ncol, figsize[1]*nrow)) # , dpi=160

    E_vec = E_vec.A.flatten() if issparse(E_vec) else E_vec.flatten()
    V = V.A[:, [x, y]] if issparse(V) else V[:, [x, y]]
    # iterate over cell first then a different dimension/gene/column
    df = pd.DataFrame({"x": np.tile(X[:, 0], n_genes), "y": np.tile(X[:, 1], n_genes), "u": np.tile(V[:, 0], n_genes),
                       "v": np.tile(V[:, 1], n_genes), 'gene': np.repeat(np.array(genes), n_cells),
                       "expression": E_vec}, index=range(n_cells * n_genes))

    # the following code is inspired by https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb
    gs = plt.GridSpec(nrow, ncol)
    for i, gn in enumerate(genes):
        ax = plt.subplot(gs[i*plot_per_gene])
        try:
            ix=np.where(adata.var.index == gn)[0][0]
        except:
            ix = gn in adata.obs.columns
            if not ix:
                continue

        cur_pd = df.loc[df.gene == gn, :]

        E_vec = cur_pd.loc[:, 'expression']

        if color is None:
            limit = np.max(np.abs(np.percentile(E_vec, [1, 99])))  # upper and lowe limit / saturation

            E_vec = E_vec + limit  # that is: tmp_colorandum - (-limit)
            E_vec = E_vec / (2 * limit)  # that is: tmp_colorandum / (limit - (-limit))
            E_vec = np.clip(E_vec, 0, 1)

            ax.scatter(cur_pd.iloc[:, 0], cur_pd.iloc[:, 1], c=cmap(E_vec), **scatter_kwargs)
        else:
            import seaborn as sns
            # List of RGB triplets
            color_labels = E_vec.unique()
            rgb_values = sns.color_palette("Set2", len(color_labels))

            # Map label to RGB
            color_map = pd.DataFrame(zip(color_labels, rgb_values), index=color_labels)

            ax.scatter(cur_pd.iloc[:, 0], cur_pd.iloc[:, 1], c=color_map.loc[E_vec, 1].values, **scatter_kwargs)

            if label_on_embedding:
                for i in color_labels:
                    color_cnt = np.median(cur_pd.iloc[np.where(E_vec == i)[0], :2], 0)
                    txt=ax.text(color_cnt[0], color_cnt[1], str(i),
                             fontsize=13) # , bbox={"facecolor": "w", "alpha": 0.6}
                    txt.set_path_effects([
                        PathEffects.Stroke(linewidth=5, foreground="w", alpha=0.1),
                        PathEffects.Normal()])

        ax.quiver(cur_pd.iloc[ix_choice, 0], cur_pd.iloc[ix_choice, 1],
                   cur_pd.iloc[ix_choice, 2], cur_pd.iloc[ix_choice, 3],
                   **quiver_kwargs)
        ax.axis("off")

    plt.show()


def _grid_velocity(adata, genes, x=0, y=1, method='SparseVFC', basis='trimap', n_columns=1, color=None, label_on_embedding=True, cmap=None,
                  s_kwargs_dict={}, layer='X', xy_grid_nums=[30, 30], g_kwargs_dict={}, quiver_scale=None, V_threshold=None,
                  figsize=None, **q_kwargs):
    """Plot the velocity vector of each cell.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        genes: `list`
            The gene names whose gene expression will be faceted.
        x: `int` (default: `0`)
            The column index of the low dimensional embedding for the x-axis
        y: `int` (default: `1`)
            The column index of the low dimensional embedding for the y-axis
        method: `str` (default: `SparseVFC`)
            Method to reconstruct the vector field. Currently it supports either SparseVFC (default) or the empirical method
            Gaussian kernel method from RNA velocity (Gaussian).
        basis: `str` (default: `trimap`)
            The reduced dimension embedding of cells to visualize.
        n_columns: `int  (default: 1)
            The number of columns of the resulting plot.
        color: `list` or None (default: None)
            A list of attributes of cells (column names in the adata.obs) will be used to color cells.
        cmap: `plt.cm` or None (default: None)
            The color map function to use.
        s_kwargs_dict: `dict` (default: {})
            The dictionary of the scatter arguments.
        layer: `str` (default: X)
            Which layer of expression value will be used.
        xy_grid_nums: `tuple` (default: (5, 5))
            the number of grids in either x or y axis.
        g_kwargs_dict: `dict` (default: {})
            A dictionary for the parameters that passed into the velocity_on_grid function.
        quiver_scale: `float` or None (default: None)
            scale of quiver plot (default: None). Number of data units per arrow length unit, e.g., m/s per plot width;
            a smaller scale parameter makes the arrow longer. If None, we will use quiver_autoscaler to calculate the scale.
        V_threshold: None or float (default: None)
            The threshold of velocity value for visualization
        figsize: `None` or `[float, float]` (default: None)
            The width and height of a figure.
        q_kwargs:
            Additional parameters that will be passed to plt.quiver function

    Returns
    -------
        Nothing but a cell wise quiver plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as PathEffects

    if cmap is None and color is None:
        cmap = plt.cm.RdBu_r

    n_cells, n_genes = adata.shape[0], len(genes)
    # {"alpha": 0.5, "s": 8, "edgecolor": "0.8", "lw": 0.15}

    scatter_kwargs = dict(alpha=0.4, s=8, edgecolor=None, linewidth=0)
    if s_kwargs_dict is not None:
        scatter_kwargs.update(s_kwargs_dict)

    if layer is 'X':
        E_vec = adata[:, adata.var.index.isin(genes)].X.T
    elif layer in adata.layers.keys():
        E_vec = adata[:, adata.var.index.isin(genes)].layers[layer].T
    elif layer is 'protein': # update subset here
        E_vec = adata[:, adata.var.index.isin(genes)].obsm[layer].T
    else:
        raise Exception(f'The {layer} you passed in is not existed in the adata object.')

    if color is not None:
        color = list(set(color).intersection(adata.obs.keys()))
        n_genes, genes = len(color), color
        E_vec = adata.obs[color].values.T.flatten() if len(color) > 0 else np.empty((0, 1))

    if ('X_' + basis in adata.obsm.keys()) and ('velocity_' + basis in adata.obsm.keys()):
        X = adata.obsm['X_' + basis][:, [x, y]]
        V = adata.obsm['velocity_' + basis][:, [x, y]]
    else:
        if 'X_' + basis not in adata.obsm.keys():
            reduceDimension(adata, velocity_key='velocity_S', reduction_method=basis)
        if 'kmc' not in adata.uns_keys():
            cell_velocities(adata, vkey='velocity_S', basis=basis, method='analytical')
            X = adata.obsm['X_' + basis][:, [x, y]]
            V = adata.obsm['velocity_' + basis][:, [x, y]]
        else:
            kmc = adata.uns['kmc']
            X = adata.obsm['X_' + basis][:, [x, y]]
            V = kmc.compute_density_corrected_drift(X, kmc.Idx, normalize_vector=True)
            adata.obsm['velocity_' + basis] = V

    if 0 in E_vec.shape:
        raise Exception(f'The gene names {genes} (or cell annotations {color}) provided are not existed in your data.')

    if method is 'SparseVFC' and adata.obsm['X_' + basis].shape[1] == 2:
        if 'VecFld_' + basis not in adata.uns.keys():
            VectorField(adata, basis=basis)
        X_grid, V_grid =  adata.uns['VecFld_' + basis]['grid'], adata.uns['VecFld_' + basis]['grid_V']
        N = int(np.sqrt(V_grid.shape[0]))
        X_grid, V_grid = np.array([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])]), \
                         np.array([V_grid[:, 0].reshape((N, N)), V_grid[:, 1].reshape((N, N))])
    elif 'grid_velocity_' + basis in adata.uns.keys():
        X_grid, V_grid, _ = adata.uns['grid_velocity_' + basis]['X_grid'], adata.uns['grid_velocity_' + basis]['V_grid'], \
                            adata.uns['grid_velocity_' + basis]['D']
    else:
        grid_kwargs_dict = {"density": None, "smooth": None, "n_neighbors": None, "min_mass": None, "autoscale": False,
                            "adjust_for_stream": True, "V_threshold": None}
        grid_kwargs_dict.update(g_kwargs_dict)

        X_grid, V_grid, D = velocity_on_grid(X[:, [x, y]], V[:, [x, y]], xy_grid_nums, **grid_kwargs_dict)

    if quiver_scale is None:
        quiver_scale = quiver_autoscaler(X_grid, V_grid)
    quiver_kwargs = {"angles": 'xy', "scale_units": 'xy', 'scale': quiver_scale, "minlength": 1.5, "alpha": 0.4}
    quiver_kwargs.update(q_kwargs)

    n_columns, plot_per_gene = n_columns, 1 # we may also add random velocity results
    nrow, ncol = int(np.ceil(plot_per_gene * n_genes / n_columns)), n_columns
    if figsize is None:
        plt.figure(None, (3*ncol, 3*nrow), dpi=160) #
    else:
        plt.figure(None, (figsize[0]*ncol, figsize[1]*nrow)) # , dpi=160

    E_vec = E_vec.A.flatten() if issparse(E_vec) else E_vec.flatten()
    V = V.A[:, [x, y]] if issparse(V) else V[:, [x, y]]
    # iterate over cell first then a different dimension/gene/column
    df = pd.DataFrame({"x": np.tile(X[:, 0], n_genes), "y": np.tile(X[:, 1], n_genes), "u": np.tile(V[:, 0], n_genes),
                       "v": np.tile(V[:, 1], n_genes), 'gene': np.repeat(np.array(genes), n_cells),
                       "expression": E_vec}, index=range(n_cells * n_genes))

    # the following code is inspired by https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb
    gs = plt.GridSpec(nrow, ncol)
    for i, gn in enumerate(genes):
        ax = plt.subplot(gs[i*plot_per_gene])
        try:
            ix=np.where(adata.var.index == gn)[0][0]
        except:
            ix = gn in adata.obs.columns
            if not ix:
                continue

        cur_pd = df.loc[df.gene == gn, :]

        E_vec = cur_pd.loc[:, 'expression']

        if color is None:
            limit = np.max(np.abs(np.percentile(E_vec, [1, 99])))  # upper and lowe limit / saturation

            E_vec = E_vec + limit  # that is: tmp_colorandum - (-limit)
            E_vec = E_vec / (2 * limit)  # that is: tmp_colorandum / (limit - (-limit))
            E_vec = np.clip(E_vec, 0, 1)

            ax.scatter(cur_pd.iloc[:, 0], cur_pd.iloc[:, 1], c=cmap(E_vec), **scatter_kwargs)
        else:
            import seaborn as sns
            # List of RGB triplets
            color_labels = E_vec.unique()
            rgb_values = sns.color_palette("Set2", len(color_labels))

            # Map label to RGB
            color_map = pd.DataFrame(zip(color_labels, rgb_values), index=color_labels)

            ax.scatter(cur_pd.iloc[:, 0], cur_pd.iloc[:, 1], c=color_map.loc[E_vec, 1].values, **scatter_kwargs)

            if label_on_embedding:
                for i in color_labels:
                    color_cnt = np.median(cur_pd.iloc[np.where(E_vec == i)[0], :2], 0)
                    txt=ax.text(color_cnt[0], color_cnt[1], str(i),
                             fontsize=13) # , bbox={"facecolor": "w", "alpha": 0.6}
                    txt.set_path_effects([
                        PathEffects.Stroke(linewidth=5, foreground="w", alpha=0.1),
                        PathEffects.Normal()])

        if V_threshold is not None:
            mass = np.sqrt((V_grid ** 2).sum(0))
            if V_threshold is not None:
                V_grid[0][mass.reshape(V_grid[0].shape) < V_threshold] = np.nan

        ax.quiver(X_grid[0], X_grid[1], V_grid[0], V_grid[1], **quiver_kwargs)
        ax.axis("off")

    plt.show()


def _streamline_plot(adata, genes, x=0, y=1, method='sparseVFC', basis='trimap', n_columns=1, color=None, label_on_embedding=True,
                   cmap=None, s_kwargs_dict={}, layer='X', xy_grid_nums=[30, 30], density=1, g_kwargs_dict={},
                   V_threshold=1e-5, figsize=None, show_quiver=True, **streamline_kwargs):
    """Plot the streamline of vector field based on the sampled cells.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        genes: `list`
            The gene names whose gene expression will be faceted.
        x: `int` (default: `0`)
            The column index of the low dimensional embedding for the x-axis
        y: `int` (default: `1`)
            The column index of the low dimensional embedding for the y-axis
        method: `str` (default: `SparseVFC`)
            Method to reconstruct the vector field. Currently it supports either SparseVFC (default) or the empirical method
            Gaussian kernel method from RNA velocity (Gaussian).
        basis: `str` (default: `trimap`)
            The reduced dimension embedding of cells to visualize.
        n_columns: `int  (default: 1)
            The number of columns of the resulting plot.
        color: `list` or None (default: None)
            A list of attributes of cells (column names in the adata.obs) will be used to color cells.
        cmap: `plt.cm` or None (default: None)
            The color map function to use.
        s_kwargs_dict: `dict` (default: {})
            The dictionary of the scatter arguments.
        layer: `str` (default: X)
            Which layer of expression value will be used.
        xy_grid_nums: `tuple` (default: (5, 5))
            the number of grids in either x or y axis.
        density: `float` or None (default: 1)
            density of the plt.streamplot function.
        q_kwargs_dict: `dict` (default: {})
            A dictionary of the parameters for the plt.quiver function.
        V_threshold: `None` or `float`
            The threshold of velocity value for visualization
        figsize: `None` or `[float, float]` (default: None)
            The width and height of a figure.
        show_quiver: `bool` (default: True)
            Whether also show the quiver plot in additional to the streamline plot.
        **streamline_kwargs:
            Additional parameters that will be passed to plt.streamplot function

    Returns
    -------
        Nothing but a cell wise quiver plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as PathEffects

    streamplot_kwargs={"density": density, "linewidth": None, "color": None, "cmap": None, "norm": None, "arrowsize": 1, "arrowstyle": '-|>',
                       "minlength": 0.1, "transform": None, "zorder": None, "start_points": None, "maxlength": 4.0,
                       "integration_direction": 'both'}
    streamplot_kwargs.update(streamline_kwargs)

    if cmap is None and color is None:
        cmap = plt.cm.RdBu_r

    n_cells, n_genes = adata.shape[0], len(genes)
    # {"alpha": 0.5, "s": 8, "edgecolor": "0.8", "lw": 0.15}

    scatter_kwargs = dict(alpha=0.4, s=8, edgecolor=None, linewidth=0)
    if s_kwargs_dict is not None:
        scatter_kwargs.update(s_kwargs_dict)

    if layer is 'X':
        E_vec = adata[:, adata.var.index.isin(genes)].X.T
    elif layer in adata.layers.keys():
        E_vec = adata[:, adata.var.index.isin(genes)].layers[layer].T
    elif layer is 'protein': # update subset here
        E_vec = adata[:, adata.var.index.isin(genes)].obsm[layer].T
    else:
        raise Exception(f'The {layer} you passed in is not existed in the adata object.')

    if color is not None:
        color = list(set(color).intersection(adata.obs.keys()))
        n_genes, genes = len(color), color
        E_vec = adata.obs[color].values.T.flatten() if len(color) > 0 else np.empty((0, 1))

    if ('X_' + basis in adata.obsm.keys()) and ('velocity_' + basis in adata.obsm.keys()):
        X = adata.obsm['X_' + basis][:, [x, y]]
        V = adata.obsm['velocity_' + basis][:, [x, y]]
    else:
        if 'X_' + basis not in adata.obsm.keys():
            reduceDimension(adata, velocity_key='velocity_S', reduction_method=basis)
        if 'kmc' not in adata.uns_keys():
            cell_velocities(adata, vkey='velocity_S', basis=basis, method='analytical')
            X = adata.obsm['X_' + basis][:, [x, y]]
            V = adata.obsm['velocity_' + basis][:, [x, y]]
        else:
            kmc = adata.uns['kmc']
            X = adata.obsm['X_' + basis][:, [x, y]]
            V = kmc.compute_density_corrected_drift(X, kmc.Idx, normalize_vector=True)
            adata.obsm['velocity_' + basis] = V

    if 0 in E_vec.shape:
        raise Exception(f'The gene names {genes} (or cell annotations {color}) provided are not existed in your data.')

    if method is 'SparseVFC' and adata.obsm['X_' + basis].shape[1] == 2:
        if 'VecFld_' + basis not in adata.uns.keys():
            VectorField(adata, basis=basis)
        X_grid, V_grid =  adata.uns['VecFld_' + basis]['grid'], adata.uns['VecFld_' + basis]['grid_V']
        N = int(np.sqrt(V_grid.shape[0]))
        X_grid, V_grid = np.array([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])]), \
                 np.array([V_grid[:, 0].reshape((N, N)), V_grid[:, 1].reshape((N, N))])
    elif 'grid_velocity_' + basis in adata.uns.keys():
        X_grid, V_grid, _ = adata.uns['grid_velocity_' + basis]['X_grid'], adata.uns['grid_velocity_' + basis]['V_grid'], \
                            adata.uns['grid_velocity_' + basis]['D']
    else:
        grid_kwargs_dict = {"density": None, "smooth": None, "n_neighbors": None, "min_mass": None, "autoscale": False,
                            "adjust_for_stream": True, "V_threshold": None}
        grid_kwargs_dict.update(g_kwargs_dict)

        X_grid, V_grid, D = velocity_on_grid(X[:, [x, y]], V[:, [x, y]], xy_grid_nums, **grid_kwargs_dict)


    # if quiver_scale is None:
    #     quiver_scale = quiver_autoscaler(X_grid, V_grid)
    # quiver_kwargs = {"angles": 'xy', "scale_units": 'xy', 'scale': quiver_scale, "minlength": 1.5}
    # quiver_kwargs.update(q_kwargs)

    n_columns, plot_per_gene = n_columns, 1 # we may also add random velocity results
    nrow, ncol = int(np.ceil(plot_per_gene * n_genes / n_columns)), n_columns
    if figsize is None:
        plt.figure(None, (3*ncol, 3*nrow), dpi=160) #
    else:
        plt.figure(None, (figsize[0]*ncol, figsize[1]*nrow)) # , dpi=160

    E_vec = E_vec.A.flatten() if issparse(E_vec) else E_vec.flatten()
    V = V.A[:, [x, y]] if issparse(V) else V[:, [x, y]]
    # iterate over cell first then a different dimension/gene/column
    df = pd.DataFrame({"x": np.tile(X[:, 0], n_genes), "y": np.tile(X[:, 1], n_genes), "u": np.tile(V[:, 0], n_genes),
                       "v": np.tile(V[:, 1], n_genes), 'gene': np.repeat(np.array(genes), n_cells),
                       "expression": E_vec}, index=range(n_cells * n_genes))

    # the following code is inspired by https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb
    gs = plt.GridSpec(nrow, ncol)
    for i, gn in enumerate(genes):
        ax = plt.subplot(gs[i*plot_per_gene])
        try:
            ix=np.where(adata.var.index == gn)[0][0]
        except:
            ix = gn in adata.obs.columns
            if not ix:
                continue

        cur_pd = df.loc[df.gene == gn, :]

        E_vec = cur_pd.loc[:, 'expression']

        if color is None:
            limit = np.max(np.abs(np.percentile(E_vec, [1, 99])))  # upper and lowe limit / saturation

            E_vec = E_vec + limit  # that is: tmp_colorandum - (-limit)
            E_vec = E_vec / (2 * limit)  # that is: tmp_colorandum / (limit - (-limit))
            E_vec = np.clip(E_vec, 0, 1)

            ax.scatter(cur_pd.iloc[:, 0], cur_pd.iloc[:, 1], c=cmap(E_vec), **scatter_kwargs)
        else:
            import seaborn as sns
            # List of RGB triplets
            color_labels = E_vec.unique()
            rgb_values = sns.color_palette("Set2", len(color_labels))

            # Map label to RGB
            color_map = pd.DataFrame(zip(color_labels, rgb_values), index=color_labels)

            ax.scatter(cur_pd.iloc[:, 0], cur_pd.iloc[:, 1], c=color_map.loc[E_vec, 1].values, **scatter_kwargs)

            if label_on_embedding:
                for i in color_labels:
                    color_cnt = np.median(cur_pd.iloc[np.where(E_vec == i)[0], :2], 0)
                    txt=ax.text(color_cnt[0], color_cnt[1], str(i),
                             fontsize=13) # , bbox={"facecolor": "w", "alpha": 0.6}
                    txt.set_path_effects([
                        PathEffects.Stroke(linewidth=5, foreground="w", alpha=0.1),
                        PathEffects.Normal()])
        if show_quiver:
            ax.quiver(X_grid[0], X_grid[1], V_grid[0], V_grid[1], color = 'gray', alpha = 0.7) # , **quiver_kwargs

        mass = np.sqrt((V_grid ** 2).sum(0))
        if V_threshold is not None:
            if V_threshold is not None:
                V_grid[0][mass.reshape(V_grid[0].shape) < V_threshold] = np.nan

        streamplot_kwargs.update({"linewidth": 4 * mass / mass[~np.isnan(mass)].max()})

        ax.streamplot(X_grid[0], X_grid[1], V_grid[0], V_grid[1], **streamplot_kwargs)
        ax.axis("off")

    plt.show()


def cell_wise_velocity_3d():
    pass

def grid_velocity_3d():
    pass

# def velocity(adata, type) # type can be either one of the three, cellwise, velocity on grid, streamline plot.
#	"""
#
#	"""
#


def plot_LIC_gray(tex):
    """GET_P estimates the posterior probability and part of the energy.

    Arguments
    ---------
        Y: 'np.ndarray'
            Original data.
        V: 'np.ndarray'
            Original data.
        sigma2: 'float'
            sigma2 is defined as sum(sum((Y - V)**2)) / (N * D)
        gamma: 'float'
            Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
        a: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.

    Returns
    -------
    P: 'np.ndarray'
        Posterior probability, related to equation 27.
    E: `np.ndarray'
        Energy, related to equation 26.

    """
    import matplotlib.pyplot as plt

    tex = tex[:, ::-1]
    tex = tex.T

    M, N = tex.shape
    texture = np.empty((M, N, 4), np.float32)
    texture[:, :, 0] = tex
    texture[:, :, 1] = tex
    texture[:, :, 2] = tex
    texture[:, :, 3] = 1

#     texture = scipy.ndimage.rotate(texture,-90)
    plt.figure()
    plt.imshow(texture)


def line_integral_conv(adata, basis='trimap', U_grid=None, V_grid=None, method = 'yt', cmap = "viridis", normalize = False,
                       density = 1, lim=(0,1), const_alpha=False, kernellen=100, V_threshold=None, file = None, g_kwargs_dict=None):
    """Visualize vector field with quiver, streamline and line integral convolution (LIC), using velocity estimates on a grid from the associated data.
    A white noise background will be used for texture as default. Adjust the bounds of lim in the range of [0, 1] which applies
    upper and lower bounds to the values of line integral convolution and enhance the visibility of plots. When const_alpha=False,
    alpha will be weighted spatially by the values of line integral convolution; otherwise a constant value of the given alpha is used.

    Arguments
    ---------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains U_grid and V_grid data
        basis: `str` (default: trimap)
            The dimension reduction method to use.
        U_grid: 'np.ndarray' (default: None)
            Original data.
        V_grid: 'np.ndarray' (default: None)
            Original data.
        method: 'float'
            sigma2 is defined as sum(sum((Y - V)**2)) / (N * D)
        cmap: 'float'
            Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
        normalize: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.
        density: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.
        lim: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.
        const_alpha: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.
        kernellen: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.
        V_threshold: `float` or `None` (default: None)
            The threshold of velocity value for visualization

    Returns
    -------
        Nothing, but plot the vector field with quiver, streamline and line integral convolution (LIC).
    """
    X = adata.obsm['X_' + basis][:, :2] if 'X_' + basis in adata.obsm.keys() else None
    V = adata.obsm['velocity_' + basis][:, :2] if 'velocity_' + basis in adata.obsm.keys() else None

    if X is None:
        raise Exception(f'The {basis} dimension reduction is not performed over your data yet.')
    if V is None:
        raise Exception(f'The {basis}_velocity velocity (or velocity) result does not existed in your data.')

    if V_threshold is not None:
        mass = np.sqrt((V_grid ** 2).sum(0))
        if V_threshold is not None:
            V_grid[0][mass.reshape(V_grid[0].shape) < V_threshold] = np.nan

    if 'VecFld_' + basis in adata.uns.keys():
        # first check whether the sparseVFC reconstructed vector field exists
        X_grid_, V_grid = adata.uns['VecFld_' + basis]['grid'], adata.uns['VecFld_' + basis]['grid_V']
        N = int(np.sqrt(V_grid.shape[0]))
        U_grid = np.reshape(V_grid[:, 0], (N, N)).T
        V_grid = np.reshape(V_grid[:, 1], (N, N)).T

    elif 'grid_velocity_' + basis in adata.uns.keys():
        # then check whether the Gaussian Kernel vector field exists
        X_grid_, V_grid_, _ = adata.uns['grid_velocity_' + basis]['X_grid'], adata.uns['grid_velocity_' + basis]['V_grid'], \
                            adata.uns['grid_velocity_' + basis]['D']
        N = int(np.sqrt(V_grid.shape[0]))
        U_grid = np.reshape(V_grid[:, 0], (N, N)).T
        V_grid = np.reshape(V_grid[:, 1], (N, N)).T
    else:
        # if no VF or Gaussian Kernel vector fields, recreate it
        grid_kwargs_dict = {"density": None, "smooth": None, "n_neighbors": None, "min_mass": None, "autoscale": False,
                            "adjust_for_stream": True, "V_threshold": None}
        grid_kwargs_dict.update(g_kwargs_dict)

        N=50
        X_grid_, V_grid_, D = velocity_on_grid(X[:, [1, 2]], V[:, [1, 2]], [N, N], **grid_kwargs_dict)
        U_grid = np.reshape(V_grid[:, 0], (N, N)).T
        V_grid = np.reshape(V_grid[:, 1], (N, N)).T

    U_grid = X_grid_ if U_grid is None else U_grid
    V_grid = V_grid_ if V_grid is None else V_grid

    if method == 'yt':
        import yt

        velocity_x_ori, velocity_y_ori, velocity_z_ori = U_grid, V_grid, np.zeros(U_grid.shape)
        velocity_x = np.repeat(velocity_x_ori[:, :, np.newaxis], V_grid.shape[1], axis=2)
        velocity_y = np.repeat(velocity_y_ori[:, :, np.newaxis], V_grid.shape[1], axis=2)
        velocity_z = np.repeat(velocity_z_ori[np.newaxis, :, :], V_grid.shape[1], axis=0)

        data = {}

        data["velocity_x"] = (velocity_x, "km/s")
        data["velocity_y"] = (velocity_y, "km/s")
        data["velocity_z"] = (velocity_z, "km/s")
        data["velocity_sum"] = (np.sqrt(velocity_x**2 + velocity_y**2), "km/s")

        ds = yt.load_uniform_grid(data, data["velocity_x"][0].shape, length_unit=(1.0,"Mpc"))
        slc = yt.SlicePlot(ds, "z", ["velocity_sum"])
        slc.set_cmap("velocity_sum", cmap)
        slc.set_log("velocity_sum", False)

        slc.annotate_velocity(normalize = normalize)
        slc.annotate_streamlines('velocity_x', 'velocity_y', density=density)
        slc.annotate_line_integral_convolution('velocity_x', 'velocity_y', lim=lim, const_alpha=const_alpha, kernellen=kernellen)

        slc.set_xlabel(basis + '_1')
        slc.set_ylabel(basis + '_2')

        slc.show()

        if file is not None:
            # plt.rc('font', family='serif', serif='Times')
            # plt.rc('text', usetex=True)
            # plt.rc('xtick', labelsize=8)
            # plt.rc('ytick', labelsize=8)
            # plt.rc('axes', labelsize=8)
            slc.save(file, mpl_kwargs = {"figsize": [2, 2]})
    elif method == 'lic':
        velocyto_tex = runlic(V_grid, V_grid, 100)
        plot_LIC_gray(velocyto_tex)


@docstrings.with_indent(4)
def cell_wise_velocity(
        adata,
        basis='umap',
        x=0,
        y=1,
        color=None,
        layer='X',
        highlights=None,
        labels=None,
        values=None,
        theme=None,
        cmap=None,
        color_key=None,
        color_key_cmap=None,
        background=None,
        ncols=1,
        pointsize=None,
        figsize=(7,5),
        show_legend=True,
        use_smoothed=True,
        ax=None,
        cell_ind='all',
        quiver_scale=None,
        s_kwargs_dict={},
        **cell_wise_kwargs):
    """Plot the velocity vector of each cell.

    Parameters
    ----------
        %(scatters.parameters.no_show_legend|kwargs)s
        cell_ind: `str` or `list` (default: all)
            the cell index that will be chosen to draw velocity vectors.
        quiver_scale: `float` or None (default: None)
            scale of quiver plot (default: None). Number of data units per arrow length unit, e.g., m/s per plot width;
            a smaller scale parameter makes the arrow longer. If None, we will use quiver_autoscaler to calculate the scale.
        s_kwargs_dict: `dict` (default: {})
            The dictionary of the scatter arguments.
        cell_wise_kwargs:
            Additional parameters that will be passed to plt.quiver function
    Returns
    -------
        Nothing but a cell wise quiver plot.
    """

    import matplotlib.pyplot as plt

    if ('X_' + basis in adata.obsm.keys()) and ('velocity_' + basis in adata.obsm.keys()):
        X = adata.obsm['X_' + basis][:, [x, y]]
        V = adata.obsm['velocity_' + basis][:, [x, y]]
    else:
        if 'X_' + basis not in adata.obsm.keys():
            reduceDimension(adata, velocity_key='velocity_S', reduction_method=basis)
        if 'kmc' not in adata.uns_keys():
            cell_velocities(adata, vkey='velocity_S', basis=basis, method='analytical')
            X = adata.obsm['X_' + basis][:, [x, y]]
            V = adata.obsm['velocity_' + basis][:, [x, y]]
        else:
            kmc = adata.uns['kmc']
            X = adata.obsm['X_' + basis][:, [x, y]]
            V = kmc.compute_density_corrected_drift(X, kmc.Idx, normalize_vector=True)
            adata.obsm['velocity_' + basis] = V

    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "u": V[:, 0], "v": V[:, 1]})

    if quiver_scale is None:
        quiver_scale = quiver_autoscaler(X, V)
    quiver_kwargs = {"angles": 'xy', "scale_units": 'xy', 'scale': quiver_scale, "minlength": 1.5, "alpha": 0.4}
    quiver_kwargs.update(cell_wise_kwargs)

    axes_list, font_color = scatters(
        adata,
        basis,
        x,
        y,
        color,
        layer,
        highlights,
        labels,
        values,
        theme,
        cmap,
        color_key,
        color_key_cmap,
        background,
        ncols,
        pointsize,
        figsize,
        show_legend,
        use_smoothed,
        ax,
        'return',
        **s_kwargs_dict)

    if cell_ind is "all":
        ix_choice = np.arange(adata.shape[0])
    elif cell_ind is 'random':
        ix_choice = np.random.choice(np.range(adata.shape[0]), size=1000, replace=False)
    elif type(cell_ind) is int:
        ix_choice = np.random.choice(np.range(adata.shape[0]), size=cell_ind, replace=False)
    elif type(cell_ind) is list:
        ix_choice = cell_ind

    for i in range(len(axes_list)):
        axes_list[i].quiver(df.iloc[ix_choice, 0], df.iloc[ix_choice, 1],
                  df.iloc[ix_choice, 2], df.iloc[ix_choice, 3], color = font_color,
                  **quiver_kwargs)

    plt.tight_layout()
    plt.show()


@docstrings.with_indent(4)
def grid_velocity(
        adata,
        basis='umap',
        x=0,
        y=1,
        color=None,
        layer='X',
        highlights=None,
        labels=None,
        values=None,
        theme=None,
        cmap=None,
        color_key=None,
        color_key_cmap=None,
        background=None,
        ncols=1,
        pointsize=None,
        figsize=(7,5),
        show_legend=True,
        use_smoothed=True,
        ax=None,
        method='SparseVFC',
        xy_grid_nums=[30, 30],
        quiver_scale=None,
        s_kwargs_dict={},
        **grid_kwargs):
    """Plot the velocity vector of each cell.

    Parameters
    ----------
        %(scatters.parameters.no_show_legend|kwargs)s
        method: `str` (default: `SparseVFC`)
            Method to reconstruct the vector field. Currently it supports either SparseVFC (default) or the empirical method
            Gaussian kernel method from RNA velocity (Gaussian).
        xy_grid_nums: `tuple` (default: (30, 30))
            the number of grids in either x or y axis.
        quiver_scale: `float` or None (default: None)
            scale of quiver plot (default: None). Number of data units per arrow length unit, e.g., m/s per plot width;
            a smaller scale parameter makes the arrow longer. If None, we will use quiver_autoscaler to calculate the scale.
        s_kwargs_dict: `dict` (default: {})
            The dictionary of the scatter arguments.
        grid_kwargs:
            Additional parameters that will be passed to plt.quiver function

    Returns
    -------
        Nothing but a quiver plot on the grid.
    """

    import matplotlib.pyplot as plt

    if ('X_' + basis in adata.obsm.keys()) and ('velocity_' + basis in adata.obsm.keys()):
        X = adata.obsm['X_' + basis][:, [x, y]]
        V = adata.obsm['velocity_' + basis][:, [x, y]]
    else:
        if 'X_' + basis not in adata.obsm.keys():
            reduceDimension(adata, velocity_key='velocity_S', reduction_method=basis)
        if 'kmc' not in adata.uns_keys():
            cell_velocities(adata, vkey='velocity_S', basis=basis, method='analytical')
            X = adata.obsm['X_' + basis][:, [x, y]]
            V = adata.obsm['velocity_' + basis][:, [x, y]]
        else:
            kmc = adata.uns['kmc']
            X = adata.obsm['X_' + basis][:, [x, y]]
            V = kmc.compute_density_corrected_drift(X, kmc.Idx, normalize_vector=True)
            adata.obsm['velocity_' + basis] = V

    if method == 'SparseVFC' and adata.obsm['X_' + basis].shape[1] == 2:
        if 'VecFld_' + basis not in adata.uns.keys():
            VectorField(adata, basis=basis)
        X_grid, V_grid =  adata.uns['VecFld_' + basis]["VecFld"]['grid'], adata.uns['VecFld_' + basis]["VecFld"]['grid_V']
        N = int(np.sqrt(V_grid.shape[0]))
        X_grid, V_grid = np.array([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])]), \
                         np.array([V_grid[:, 0].reshape((N, N)), V_grid[:, 1].reshape((N, N))])
    elif 'grid_velocity_' + basis in adata.uns.keys():
        X_grid, V_grid, _ = adata.uns['grid_velocity_' + basis]["VecFld"]['X_grid'], adata.uns['grid_velocity_' + basis]["VecFld"]['V_grid'], \
                            adata.uns['grid_velocity_' + basis]["VecFld"]['D']
    else:
        grid_kwargs_dict = {"density": None, "smooth": None, "n_neighbors": None, "min_mass": None, "autoscale": False,
                            "adjust_for_stream": True, "V_threshold": None}
        grid_kwargs_dict.update(grid_kwargs)

        X_grid, V_grid, D = velocity_on_grid(X[:, [x, y]], V[:, [x, y]], xy_grid_nums, **grid_kwargs_dict)

    if quiver_scale is None:
        quiver_scale = quiver_autoscaler(X_grid, V_grid)
    quiver_kwargs = {"angles": 'xy', "scale_units": 'xy', 'scale': quiver_scale, "minlength": 1.5, "alpha": 0.4}
    quiver_kwargs.update(grid_kwargs)

    axes_list, font_color = scatters(
        adata,
        basis,
        x,
        y,
        color,
        layer,
        highlights,
        labels,
        values,
        theme,
        cmap,
        color_key,
        color_key_cmap,
        background,
        ncols,
        pointsize,
        figsize,
        show_legend,
        use_smoothed,
        ax,
        'return',
        **s_kwargs_dict)

    for i in range(len(axes_list)):
        axes_list[i].quiver(X_grid[0], X_grid[1], V_grid[0], V_grid[1], color=font_color, **quiver_kwargs)

    plt.tight_layout()
    plt.show()


@docstrings.with_indent(4)
def streamline_plot(
        adata,
        basis='umap',
        x=0,
        y=1,
        color=None,
        layer='X',
        highlights=None,
        labels=None,
        values=None,
        theme=None,
        cmap=None,
        color_key=None,
        color_key_cmap=None,
        background=None,
        ncols=1,
        pointsize=None,
        figsize=(7, 5),
        show_legend=True,
        use_smoothed=True,
        ax=None,
        method='SparseVFC',
        xy_grid_nums=[30, 30],
        density=1,
        s_kwargs_dict={},
        **streamline_kwargs):
    """Plot the velocity vector of each cell.

    Parameters
    ----------
        %(scatters.parameters.no_show_legend|kwargs)s
        method: `str` (default: `SparseVFC`)
            Method to reconstruct the vector field. Currently it supports either SparseVFC (default) or the empirical method
            Gaussian kernel method from RNA velocity (Gaussian).
        xy_grid_nums: `tuple` (default: (30, 30))
            the number of grids in either x or y axis.
        density: `float` or None (default: 1)
            density of the plt.streamplot function.
        s_kwargs_dict: `dict` (default: {})
            The dictionary of the scatter arguments.
        streamline_kwargs:
            Additional parameters that will be passed to plt.streamplot function
    Returns
    -------
        Nothing but a streamline plot that integrates paths in the vector field.
    """

    import matplotlib.pyplot as plt

    if ('X_' + basis in adata.obsm.keys()) and ('velocity_' + basis in adata.obsm.keys()):
        X = adata.obsm['X_' + basis][:, [x, y]]
        V = adata.obsm['velocity_' + basis][:, [x, y]]
    else:
        if 'X_' + basis not in adata.obsm.keys():
            reduceDimension(adata, velocity_key='velocity_S', reduction_method=basis)
        if 'kmc' not in adata.uns_keys():
            cell_velocities(adata, vkey='velocity_S', basis=basis, method='analytical')
            X = adata.obsm['X_' + basis][:, [x, y]]
            V = adata.obsm['velocity_' + basis][:, [x, y]]
        else:
            kmc = adata.uns['kmc']
            X = adata.obsm['X_' + basis][:, [x, y]]
            V = kmc.compute_density_corrected_drift(X, kmc.Idx, normalize_vector=True)
            adata.obsm['velocity_' + basis] = V

    if method == 'SparseVFC' and adata.obsm['X_' + basis].shape[1] == 2:
        if 'VecFld_' + basis not in adata.uns.keys():
            VectorField(adata, basis=basis)
        X_grid, V_grid =  adata.uns['VecFld_' + basis]["VecFld"]['grid'], adata.uns['VecFld_' + basis]["VecFld"]['grid_V']
        N = int(np.sqrt(V_grid.shape[0]))
        X_grid, V_grid = np.array([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])]), \
                         np.array([V_grid[:, 0].reshape((N, N)), V_grid[:, 1].reshape((N, N))])
    elif 'grid_velocity_' + basis in adata.uns.keys():
        X_grid, V_grid, _ = adata.uns['grid_velocity_' + basis]["VecFld"]['X_grid'], adata.uns['grid_velocity_' + basis]["VecFld"]['V_grid'], \
                            adata.uns['grid_velocity_' + basis]["VecFld"]['D']
    else:
        grid_kwargs_dict = {"density": None, "smooth": None, "n_neighbors": None, "min_mass": None, "autoscale": False,
                            "adjust_for_stream": True, "V_threshold": None}
        grid_kwargs_dict.update(streamline_kwargs)

        X_grid, V_grid, D = velocity_on_grid(X[:, [x, y]], V[:, [x, y]], xy_grid_nums, **grid_kwargs_dict)

    streamplot_kwargs={"density": density, "linewidth": None, "cmap": None, "norm": None, "arrowsize": 1, "arrowstyle": '-|>',
                       "minlength": 0.1, "transform": None, "zorder": None, "start_points": None, "maxlength": 4.0,
                       "integration_direction": 'both'}
    streamplot_kwargs.update(streamline_kwargs)
    mass = np.sqrt((V_grid ** 2).sum(0))
    streamplot_kwargs.update({"linewidth": 4 * mass / mass[~np.isnan(mass)].max()})

    axes_list, font_color = scatters(
        adata,
        basis,
        x,
        y,
        color,
        layer,
        highlights,
        labels,
        values,
        theme,
        cmap,
        color_key,
        color_key_cmap,
        background,
        ncols,
        pointsize,
        figsize,
        show_legend,
        use_smoothed,
        ax,
        'return',
        **s_kwargs_dict)

    for i in range(len(axes_list)):
        axes_list[i].streamplot(X_grid[0], X_grid[1], V_grid[0], V_grid[1], color=font_color, **streamplot_kwargs)

    plt.tight_layout()
    plt.show()

# refactor line_conv_integration
