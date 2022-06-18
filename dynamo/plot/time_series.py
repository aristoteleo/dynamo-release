# include pseudotime and predict cell trajectory
from typing import Optional, Union

import numpy as np
from anndata import AnnData
from scipy.interpolate import interp1d
from scipy.sparse import issparse

from ..docrep import DocstringProcessor
from ..external.hodge import ddhodge
from ..prediction.utils import fetch_exprs
from ..tools.utils import update_dict
from .utils import _to_hex, save_fig

docstrings = DocstringProcessor()


@docstrings.get_sectionsf("kin_curves")
def kinetic_curves(
    adata: AnnData,
    genes: list,
    mode: str = "vector_field",
    basis: Optional[str] = None,
    layer: str = "X",
    project_back_to_high_dim: bool = True,
    tkey: str = "potential",
    dist_threshold: float = 1e-10,
    ncol: int = 4,
    color: Union[list, None] = "ntr",
    c_palette: str = "Set2",
    standard_scale: int = 0,
    traj_ind: int = 0,
    log: bool = True,
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
):
    """Plot the gene expression dynamics over time (pseudotime or inferred real time) as kinetic curves.

    Note that by default `potential` estimated with the diffusion graph built from reconstructed vector field will be
    used as the measure of pseudotime.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        genes: `list`
            The gene names whose gene expression will be faceted.
        mode: `str` (default: `vector_field`)
            Which data mode will be used, either vector_field, lap or pseudotime. if mode is vector_field, the
            trajectory predicted by vector field function will be used; if mode is lap, the trajectory predicted by
            least action path will be used otherwise pseudotime trajectory (defined by time argument) will be used.
            By default `potential` estimated with the diffusion graph built from reconstructed vector field will be used
            as pseudotime when mode is pseudotime.
        basis: `str` or None (default: `None`)
            The embedding data used for drawing the kinetic gene expression curves, only used when mode is
            `vector_field`.
        layer: `str` (default: X)
            Which layer of expression value will be used. Not used if mode is `vector_field`.
        project_back_to_high_dim: `bool` (default: `False`)
            Whether to map the coordinates in low dimension back to high dimension to visualize the gene expression
            curves, only used when mode is `vector_field` and basis is not `X`. Currently only works when basis is 'pca'
            and 'umap'.
        color: `list` or None (default: 'ntr')
            A list of attributes of cells (column names in the adata.obs) will be used to color cells.
        tkey: `str` (default: `potential`)
            The .obs column that will be used for timing each cell, only used when mode is not `vector_field`.
        dist_threshold: `float` (default: 1e-10)
            The threshold for the distance between two points in the gene expression state, i.e, x(t), x(t+1). If below
            this threshold, we assume steady state is achieved and those data points will not be considered. This
            argument is ignored when mode is `pseudotime`.
        ncol: `int` (default: 4)
            Number of columns in each facet grid.
        c_palette: Name of color_palette supported in seaborn color_palette function (default: None)
            The color map function to use.
        standard_scale: `int` (default: 1)
            Either 0 (rows) or 1 (columns). Whether or not to standardize that dimension, meaning for each row or
            column, subtract the minimum and divide each by its maximum.
        traj_ind: `int` (default: 0)
            If the element from the dictionary is a list (obtained from a list of trajectories), the index of trajectory
            that will be selected for visualization.
        log: `bool` (default: True)
            Whether to log1p transform your data before data visualization. If expression data is from adata object,
            it is generally already log1p transformed. When the data is from predicted either from traj simulation or
            LAP, the data is generally in the original gene expression space and needs to be log1p transformed. Note:
            when predicted data is not inverse transformed back to original expression space, no transformation will be
            applied.
        save_show_or_return: {'show', 'save_fig', 'return'} (default: `show`)
            Whether to save_fig, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
            save_fig function will use the {"path": None, "prefix": 'kinetic_curves', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a
            dictionary that properly modify those keys according to your needs.

    Returns
    -------
        Nothing but plots the kinetic curves that shows the gene expression dynamics over time.
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    if mode == "pseudotime" and tkey == "potential" and "potential" not in adata.obs_keys():
        ddhodge(adata)

    exprs, valid_genes, time = fetch_exprs(
        adata,
        basis,
        layer,
        genes,
        tkey,
        mode,
        project_back_to_high_dim,
        traj_ind,
    )

    Color = np.empty((0, 1))
    if color is not None and mode not in ["lap", "vector_field"]:
        color = list(set(color).intersection(adata.obs.keys()))
        Color = adata.obs[color].values.T.flatten() if len(color) > 0 else np.empty((0, 1))

    exprs = exprs.A if issparse(exprs) else exprs
    if len(set(genes).intersection(valid_genes)) > 0:
        # by default, expression values are log1p tranformed if using the expression from adata.
        exprs = np.expm1(exprs) if not log else exprs

    if standard_scale is not None:
        exprs = (exprs - np.min(exprs, axis=standard_scale)) / np.ptp(exprs, axis=standard_scale)

    time = np.sort(time)
    exprs = exprs[np.argsort(time), :]

    if dist_threshold is not None and mode in ["lap", "vector_field"]:
        valid_ind = list(np.where(np.sum(np.diff(exprs, axis=0) ** 2, axis=1) > dist_threshold)[0] + 1)
        valid_ind.insert(0, 0)
        exprs = exprs[valid_ind, :]
        time = time[valid_ind]

    exprs_df = pd.DataFrame(
        {
            "Time": np.repeat(time, len(valid_genes)),
            "Expression": exprs.flatten(),
            "Gene": np.tile(valid_genes, len(time)),
        }
    )

    if exprs_df.shape[0] == 0:
        raise Exception(
            "No genes you provided are detected. Please make sure the genes provided are from the genes "
            "used for vector field reconstructed when layer is set."
        )

    # https://stackoverflow.com/questions/43920341/python-seaborn-facetgrid-change-titles
    if len(Color) > 0:
        exprs_df["Color"] = np.repeat(Color, len(valid_genes))
        g = sns.relplot(
            x="Time",
            y="Expression",
            data=exprs_df,
            col="Gene",
            hue="Color",
            palette=sns.color_palette(c_palette),
            col_wrap=ncol,
            kind="line",
            facet_kws={"sharex": True, "sharey": False},
        )
    else:
        g = sns.relplot(
            x="Time",
            y="Expression",
            data=exprs_df,
            col="Gene",
            col_wrap=ncol,
            kind="line",
            facet_kws={"sharex": True, "sharey": False},
        )

    if save_show_or_return == "save":
        s_kwargs = {
            "path": None,
            "prefix": "kinetic_curves",
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return g


docstrings.delete_params("kin_curves.parameters", "ncol", "color", "c_palette")


@docstrings.with_indent(4)
def kinetic_heatmap(
    adata: AnnData,
    genes: list,
    mode: str = "vector_field",
    basis: Optional[str] = None,
    layer: str = "X",
    project_back_to_high_dim: bool = True,
    tkey: str = "potential",
    dist_threshold: float = 1e-10,
    color_map: int = "BrBG",
    gene_order_method: str = "maximum",
    show_colorbar: bool = False,
    cluster_row_col: list = [False, False],
    figsize: tuple = (11.5, 6),
    standard_scale: int = 1,
    n_convolve: int = 30,
    spaced_num: int = 100,
    traj_ind: int = 0,
    log: bool = True,
    gene_group: Union[None, list] = None,
    gene_group_cmap: Union[None, list] = None,
    cell_group: Union[None, list] = None,
    cell_group_cmap: Union[None, list] = None,
    enforce: bool = False,
    hline_rows: Union[None, list] = None,
    hlines_kwargs: dict = {},
    vline_cols: Union[None, list] = None,
    vlines_kwargs: dict = {},
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
    transpose: bool = False,
    **kwargs,
):
    """Plot the gene expression dynamics over time (pseudotime or inferred real time) in a heatmap.

    Note that by default `potential` estimated with the diffusion graph built from reconstructed vector field will be
    used as the measure of pseudotime.

    Parameters
    ----------
        %(kin_curves.parameters.no_ncol|color|c_palette)s
        color_map: `str` (default: `BrBG`)
            Color map that will be used to color the gene expression. If `half_max_ordering` is True, the
            color map need to be divergent, good examples, include `BrBG`, `RdBu_r` or `coolwarm`, etc.
        gene_order_method: `str` (default: `half_max_ordering`) [`half_max_ordering`, `maximum`, `raw`]
            Supports three different methods for ordering genes when plotting the heatmap: either `half_max_ordering`,
            `maximum` or `raw`. For `half_max_ordering`, it will order genes into up, down and transit groups by the
            half max ordering algorithm (HA Pliner, et. al, Molecular cell 71 (5), 858-871. e8). While for `maximum`,
            it will order by the position of the highest gene expression. `raw` means just use the original order from
            the input gene list.
        show_colorbar: `bool` (default: `False`)
            Whether to show the color bar.
        cluster_row_col: `[bool, bool]` (default: `[False, False]`)
            Whether to cluster the row or columns.
        figsize: `str` (default: `(11.5, 6)`
            Size of figure
        standard_scale: `int` (default: 1)
            Either 0 (rows, cells) or 1 (columns, genes). Whether or not to standardize that dimension, meaning for each
            row or column, subtract the minimum and divide each by its maximum.
        n_convolve: `int` (default: 30)
            Number of cells for convolution.
        traj_ind: `int` (default: 0)
            If the element from the dictionary is a list (obtained from a list of trajectories), the index of trajectory
            that will be selected for visualization.
        log: `bool` (default: True)
            Whether to log1p transform your data before data visualization. If expression data is from adata object,
            it is generally already log1p transformed. When the data is from predicted either from traj simulation or
            LAP, the data is generally in the original gene expression space and needs to be log1p transformed. Note:
            when predicted data is not inverse transformed back to original expression space, no transformation will be
            applied.
        gene_group:
            The key of the gene groups in .var.
        gene_group_cmap:
            The str of the colormap for gene groups.
        cell_group:
             The key of the cell groups in .obs.
        cell_group_cmap:
            The str of the colormap for cell groups.
        enforce:
            Whether to recalculate the dataframe that will be used to create the kinetic heatmap. If this is set to be
            False and the the .uns['kinetic_heatmap'] is in the adata object, we will use data from
            `.uns['kinetic_heatmap']` directly.
        hline_rows:
            The indices of rows that we can place a line on the heatmap.
        hlines_kwargs:
            The dictionary of arguments that will be passed into sns_heatmap.ax_heatmap.hlines.
        vline_cols:
            The indices of column that we can place a line on the heatmap.
        vlines_kwargs:
            The dictionary of arguments that will be passed into sns_heatmap.ax_heatmap.vlines.
        save_show_or_return: {'show', 'save_fig', 'return'} (default: `show`)
            Whether to save_fig, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
            save_fig function will use the {"path": None, "prefix": 'kinetic_heatmap', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a
            dictionary that properly modify those keys according to your needs.
        transpose:
            Whether to transpose the dataframe and swap X-Y in heatmap. In single cell case, `transpose=True` results in gene on the x-axis.
        kwargs:
            All other keyword arguments are passed to heatmap(). Currently `xticklabels=False, yticklabels='auto'` is
            passed to heatmap() by default.

    Returns
    -------
        Nothing but plots a heatmap that shows the gene expression dynamics over time.
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    if enforce or "kinetic_heatmap" not in adata.uns_keys():

        if mode == "pseudotime" and tkey == "potential" and "potential" not in adata.obs_keys():
            ddhodge(adata)

        exprs, valid_genes, time = fetch_exprs(
            adata,
            basis,
            layer,
            genes,
            tkey,
            mode,
            project_back_to_high_dim,
            traj_ind,
        )

        valid_genes = [x for x in genes if x in valid_genes]

        exprs = exprs.A if issparse(exprs) else exprs
        if mode != "pseudotime":
            exprs = np.log1p(exprs) if log else exprs

            spaced_num = None  # don't need to get further smoothed.

        if len(set(genes).intersection(valid_genes)) > 0:
            # by default, expression values are log1p tranformed if using the expression from adata.
            exprs = np.expm1(exprs) if not log else exprs

        if dist_threshold is not None and mode == "vector_field":
            valid_ind = list(np.where(np.sum(np.diff(exprs, axis=0) ** 2, axis=1) > dist_threshold)[0] + 1)
            valid_ind.insert(0, 0)
            exprs = exprs[valid_ind, :]
            time = time[valid_ind]

        if gene_order_method == "half_max_ordering":
            time, all, valid_ind, gene_idx = _half_max_ordering(
                exprs.T, time, mode=mode, interpolate=True, spaced_num=spaced_num
            )
            all, genes = (
                all[np.isfinite(all.sum(1)), :],
                np.array(valid_genes)[gene_idx][np.isfinite(all.sum(1))],
            )

            df = pd.DataFrame(all, index=genes)
        elif gene_order_method in ["maximum", "raw"]:
            exprs = lowess_smoother(time, exprs.T, spaced_num=spaced_num, n_convolve=n_convolve)
            exprs = exprs[np.isfinite(exprs.sum(1)), :]

            if standard_scale is not None:
                exprs = (exprs - np.min(exprs, axis=standard_scale)[:, None]) / np.ptp(exprs, axis=standard_scale)[
                    :, None
                ]
            if gene_order_method == "maximum":
                max_sort = np.argsort(np.argmax(exprs, axis=1))
            else:
                max_sort = np.arange(exprs.shape[0])
            if spaced_num is None and mode == "pseudotime":
                df = pd.DataFrame(
                    exprs[max_sort, :],
                    index=np.array(valid_genes)[max_sort],
                    columns=adata.obs_names,
                )
            else:
                df = pd.DataFrame(exprs[max_sort, :], index=np.array(valid_genes)[max_sort])
        else:
            raise Exception("gene order_method can only be either half_max_ordering or maximum")

        adata.uns["kinetics_heatmap"] = df
    else:
        df = adata.uns["kinetics_heatmap"]

    row_colors, col_colors = None, None
    if gene_group is not None:
        color_key_cmap = "tab20" if gene_group_cmap is None else gene_group_cmap
        uniq_gene_grps = adata.var[gene_group].unique().tolist()
        num_labels = len(uniq_gene_grps)

        color_key = _to_hex(plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels)))
        gene_lut = dict(zip(map(str, uniq_gene_grps), color_key))
        row_colors = adata.var[gene_group].map(gene_lut)
    else:
        uniq_gene_grps, gene_lut = [], {}

    if cell_group is not None:
        color_key_cmap = "tab20" if cell_group_cmap is None else cell_group_cmap
        uniq_cell_grps = adata.obs[cell_group].unique().tolist()
        num_labels = len(uniq_cell_grps)

        color_key = _to_hex(plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels)))
        cell_lut = dict(zip(map(str, uniq_cell_grps), color_key))
        col_colors = adata.obs[cell_group].map(cell_lut)
    else:
        uniq_cell_grps, cell_lut = [], {}

    if transpose:
        row_colors, col_colors = col_colors, row_colors
        cluster_row_col[0], cluster_row_col[1] = cluster_row_col[1], cluster_row_col[0]
        df = df.T

    heatmap_kwargs = dict(
        xticklabels=False,
        yticklabels=1,
        row_colors=row_colors,
        col_colors=col_colors,
        row_linkage=None,
        col_linkage=None,
        method="average",
        metric="euclidean",
        z_score=None,
        standard_scale=None,
    )
    if kwargs is not None:
        heatmap_kwargs = update_dict(heatmap_kwargs, kwargs)

    sns_heatmap = sns.clustermap(
        df,
        col_cluster=cluster_row_col[0],
        row_cluster=cluster_row_col[1],
        cmap=color_map,
        figsize=figsize,
        **heatmap_kwargs,
    )

    if not show_colorbar:
        sns_heatmap.cax.set_visible(False)
    if cell_group is not None or gene_group is not None:
        # https://stackoverflow.com/questions/27988846/how-to-express-classes-on-the-axis-of-a-heatmap-in-seaborn
        # answer from mwaskom
        uniq_grps = uniq_cell_grps + uniq_gene_grps
        lut = cell_lut.copy()
        lut.update(gene_lut)
        for label in uniq_grps:
            sns_heatmap.ax_col_dendrogram.bar(0, 0, color=lut[label], label=label, linewidth=0)
        cell_group_num, gene_group_num = len(cell_lut), len(gene_lut)

        if cell_group_num > 0 and gene_group_num > 0:
            ncol = min([cell_group_num, gene_group_num])
        else:
            ncol = 5

        if cell_group is None:
            title = gene_group
        elif gene_group is None:
            title = cell_group
        else:
            title = gene_group + cell_group

        sns_heatmap.ax_col_dendrogram.legend(title=title, loc="center", ncol=ncol)
        sns_heatmap.cax.set_position([0.15, 0.2, 0.03, 0.45])

    if hline_rows is not None:
        hl_kwargs = update_dict({"linestyles": "dashdot"}, hlines_kwargs)
        sns_heatmap.ax_heatmap.hlines(hline_rows, *sns_heatmap.ax_heatmap.get_xlim(), **hl_kwargs)
    if vline_cols is not None:
        vline_kwargs = update_dict({"linestyles": "dashdot"}, vlines_kwargs)
        sns_heatmap.ax_heatmap.vlines(vline_cols, *sns_heatmap.ax_heatmap.get_ylim(), **vline_kwargs)

    if save_show_or_return == "save":
        s_kwargs = {
            "path": None,
            "prefix": "kinetic_heatmap",
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        if show_colorbar:
            plt.subplots_adjust(right=0.85)
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return sns_heatmap


def _half_max_ordering(exprs, time, mode, interpolate=False, spaced_num=100):
    """Implement the half-max ordering algorithm from HA Pliner, Molecular Cell, 2018.

    Parameters
    ----------
        exprs: `np.ndarray`
            The gene expression matrix (ngenes x ncells) ordered along time (either pseudotime or inferred real time).
        time: `np.ndarray`
            Pseudotime or inferred real time.
        mode: `str` (default: `vector_field`)
            Which data mode will be used, either vector_field or pseudotime. if mode is vector_field, the trajectory
            predicted by vector field function will be used, otherwise pseudotime trajectory (defined by time argument)
            will be used.
        interpolate: `bool` (default: `False`)
            Whether to interpolate the data when performing the loess fitting.
        spaced_num: `float` (default: `100`)
            The number of points on the loess fitting curve.

    Returns
    -------
        time: `np.ndarray`
            The time at which the loess is evaluated.
        all: `np.ndarray`
            The ordered smoothed, scaled expression matrix, the first group is up, then down, followed by the transient
            gene groups.
        valid_ind: `np.ndarray`
            The indices of valid genes that Loess smoothed.
        gene_idx: `np.ndarray`
            The indices of genes that are used for the half-max ordering plot.
    """

    if mode == "vector_field":
        interpolate = False

    gene_num = exprs.shape[0]
    cell_num = spaced_num if interpolate else exprs.shape[1]
    if interpolate:
        hm_mat_scaled, hm_mat_scaled_z = (
            np.zeros((gene_num, cell_num)),
            np.zeros((gene_num, cell_num)),
        )
    else:
        hm_mat_scaled, hm_mat_scaled_z = (
            np.zeros_like(exprs),
            np.zeros_like(exprs),
        )

    transient, trans_max, half_max = (
        np.zeros(gene_num),
        np.zeros(gene_num),
        np.zeros(gene_num),
    )

    tmp = lowess_smoother(time, exprs, spaced_num) if interpolate else exprs

    for i in range(gene_num):
        hm_mat_scaled[i] = tmp[i] - np.min(tmp[i])
        hm_mat_scaled[i] = hm_mat_scaled[i] / np.max(hm_mat_scaled[i])
        scale_tmp = (tmp[i] - np.mean(tmp[i])) / np.std(tmp[i])  # scale in R
        hm_mat_scaled_z[i] = scale_tmp

        count, current = 0, hm_mat_scaled_z[i, 0] < 0  # check this
        for j in range(cell_num):
            if not (scale_tmp[j] < 0 == current):
                count = count + 1
                current = scale_tmp[j] < 0

        half_max[i] = np.argmax(np.abs(scale_tmp - 0.5))
        transient[i] = count
        trans_max[i] = np.argsort(-scale_tmp)[0]

    begin = np.arange(max([5, 0.05 * cell_num]))
    end = np.arange(exprs.shape[1] - max([5, 0.05 * cell_num]), cell_num)
    trans_indx = np.logical_and(
        transient > 1,
        not [i in np.concatenate((begin, end)) for i in trans_max],
    )

    trans_idx, trans, half_max_trans = (
        np.where(trans_indx)[0],
        hm_mat_scaled[trans_indx, :],
        half_max[trans_indx],
    )
    nt_idx, nt = np.where(~trans_indx)[0], hm_mat_scaled[~trans_indx, :]
    up_idx, up, half_max_up = (
        np.where(nt[:, 0] < nt[:, -1])[0],
        nt[nt[:, 0] < nt[:, -1], :],
        half_max[nt[:, 0] < nt[:, -1]],
    )
    down_indx, down, half_max_down = (
        np.where(nt[:, 0] >= nt[:, -1])[0],
        nt[nt[:, 0] >= nt[:, -1], :],
        half_max[nt[:, 0] >= nt[:, -1]],
    )

    trans, up, down = (
        trans[np.argsort(half_max_trans), :],
        up[np.argsort(half_max_up), :],
        down[np.argsort(half_max_down), :],
    )

    all = np.vstack((up, down, trans))
    gene_idx = np.hstack(
        (
            nt_idx[up_idx][np.argsort(half_max_up)],
            nt_idx[down_indx][np.argsort(half_max_down)],
            trans_idx,
        )
    )

    return time, all, np.isfinite(nt[:, 0]) & np.isfinite(nt[:, -1]), gene_idx


def lowess_smoother(time, exprs, spaced_num=None, n_convolve=30):
    gene_num = exprs.shape[0]
    if spaced_num is None:
        res = exprs.copy()

        if exprs.shape[1] < 300:
            return res
    else:
        res = np.zeros((gene_num, spaced_num))

    for i in range(gene_num):
        x = exprs[i]

        if spaced_num is None:
            x_convolved = np.convolve(x[np.argsort(time)], np.ones(30) / 30, mode="same")
            res[i, :] = x_convolved
        else:
            # lowess = sm.nonparametric.lowess
            # tmp = lowess(x, time, frac=.3)
            # # run scipy's interpolation.
            # f = interp1d(tmp[:, 0], tmp[:, 1], bounds_error=False)

            x_convolved = np.convolve(
                x[np.argsort(time)],
                np.ones(n_convolve) / n_convolve,
                mode="same",
            )

            # check: is any difference between interpld and np.convolve?
            if len(time) == len(x_convolved):
                f = interp1d(time[np.argsort(time)], x_convolved, bounds_error=False)

                time_linspace = np.linspace(np.min(time), np.max(time), spaced_num)
                res[i, :] = f(time_linspace)
            else:
                res[i, :] = np.convolve(
                    x[np.argsort(time)],
                    np.ones(spaced_num) / spaced_num,
                    mode="same",
                )

    return res


@docstrings.with_indent(4)
def jacobian_kinetics(
    adata: AnnData,
    basis: str = "umap",
    regulators: Optional[list] = None,
    effectors: Optional[list] = None,
    mode: str = "pseudotime",
    tkey: str = "potential",
    color_map: str = "bwr",
    gene_order_method: str = "raw",
    show_colorbar: bool = False,
    cluster_row_col: list = [False, True],
    figsize: tuple = (11.5, 6),
    standard_scale: int = 1,
    n_convolve: int = 30,
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
    **kwargs,
):
    """Plot the Jacobian dynamics over time (pseudotime or inferred real time) in a heatmap.

    Note that by default `potential` estimated with the diffusion graph built from reconstructed vector field will be
    used as the measure of pseudotime.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        basis: `str`
            The reduced dimension basis.
        regulators: `list` or `None` (default: `None`)
            The list of genes that will be used as regulators for plotting the Jacobian heatmap, only limited to genes
            that have already performed Jacobian analysis.
        effectors: `List` or `None` (default: `None`)
            The list of genes that will be used as targets for plotting the Jacobian heatmap, only limited to genes
            that have already performed Jacobian analysis.
        mode: `str` (default: `vector_field`)
            Which data mode will be used, either vector_field or pseudotime. if mode is vector_field, the trajectory
            predicted by vector field function will be used, otherwise pseudotime trajectory (defined by time argument)
            will be used. By default `potential` estimated with the diffusion graph built reconstructed vector field
            will be used as pseudotime.
        tkey: `str` (default: `potential`)
            The .obs column that will be used for timing each cell, only used when mode is not `vector_field`.
        color_map: `str` (default: `BrBG`)
            Color map that will be used to color the gene expression. If `half_max_ordering` is True, the
            color map need to be divergent, good examples, include `BrBG`, `RdBu_r` or `coolwarm`, etc.
        gene_order_method: `str` (default: `half_max_ordering`) [`half_max_ordering`, `maximum`]
            Supports two different methods for ordering genes when plotting the heatmap: either `half_max_ordering`,
            or `maximum`. For `half_max_ordering`, it will order genes into up, down and transit groups by the half
            max ordering algorithm (HA Pliner, et. al, Molecular cell 71 (5), 858-871. e8). While for `maximum`,
            it will order by the position of the highest gene expression.
        show_colorbar: `bool` (default: `False`)
            Whether to show the color bar.
        cluster_row_col: `[bool, bool]` (default: `[False, False]`)
            Whether to cluster the row or columns.
        figsize: `str` (default: `(11.5, 6)`
            Size of figure
        standard_scale: `int` (default: 1)
            Either 0 (rows, cells) or 1 (columns, genes). Whether or not to standardize that dimension, meaning for each
            row or column, subtract the minimum and divide each by its maximum.
        n_convolve: `int` (default: 30)
            Number of cells for convolution.
        save_show_or_return: {'show', 'save_fig', 'return'} (default: `show`)
            Whether to save_fig, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
            save_fig function will use the {"path": None, "prefix": 'kinetic_curves', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a
            dictionary that properly modify those keys according to your needs.
        kwargs:
            All other keyword arguments are passed to heatmap(). Currently `xticklabels=False, yticklabels='auto'` is
            passed to heatmap() by default.
    Returns
    -------
        Nothing but plots a heatmap that shows the element of Jacobian matrix dynamics over time (potential decreasing).

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> adata = dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.vf.VectorField(adata, basis='pca')
    >>> valid_gene_list = adata[:, adata.var.use_for_transition].var.index[:2]
    >>> dyn.vf.jacobian(adata, regulators=valid_gene_list[0], effectors=valid_gene_list[1])
    >>> dyn.pl.jacobian_kinetics(adata)
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    Jacobian_ = "jacobian" if basis is None else "jacobian_" + basis
    Der, cell_indx, _, regulators_, effectors_ = (
        adata.uns[Jacobian_].get("jacobian"),
        adata.uns[Jacobian_].get("cell_idx"),
        adata.uns[Jacobian_].get("jacobian_gene"),
        adata.uns[Jacobian_].get("regulators"),
        adata.uns[Jacobian_].get("effectors"),
    )
    if tkey == "potential" and "potential" not in adata.obs_keys():
        ddhodge(adata)

    adata_ = adata[cell_indx, :]
    time = adata_.obs[tkey]
    jacobian_mat = Der.reshape((-1, Der.shape[2])) if Der.ndim == 3 else Der[None, :]
    n_source_targets_ = Der.shape[0] * Der.shape[1] if Der.ndim == 3 else 1
    targets_, sources_ = (
        (
            np.repeat(effectors_, Der.shape[1]),
            np.tile(regulators_, Der.shape[0]),
        )
        if Der.ndim == 3
        else (
            np.repeat(effectors_, Der.shape[0]),
            np.repeat(effectors_, Der.shape[0]),
        )
    )
    source_targets_ = [sources_[i] + "->" + targets_[i] for i in range(n_source_targets_)]

    regulators = regulators_ if regulators is None else regulators
    effectors = effectors_ if effectors is None else effectors
    if type(regulators) == str:
        regulators = [regulators]
    if type(effectors) == str:
        effectors = [effectors]
    regulators = list(set(regulators_).intersection(regulators))
    effectors = list(set(effectors_).intersection(effectors))
    if len(regulators) == 0 or len(effectors) == 0:
        raise ValueError(
            f"Jacobian related to source genes {regulators} and target genes {effectors}"
            f"you provided are existed. Available source genes includes {regulators_} while "
            f"available target genes includes {effectors_}"
        )
    n_source_targets = len(regulators) * len(effectors)
    targets, sources = (
        np.repeat(effectors, len(regulators)),
        np.tile(regulators, len(effectors)),
    )
    source_targets = [sources[i] + "->" + targets[i] for i in range(n_source_targets)]

    jacobian_mat = jacobian_mat[:, np.argsort(time)]

    if gene_order_method == "half_max_ordering":
        time, all, valid_ind, gene_idx = _half_max_ordering(
            jacobian_mat, time, mode=mode, interpolate=True, spaced_num=100
        )
        all, source_targets = (
            all[np.isfinite(all.sum(1)), :],
            np.array(source_targets)[gene_idx][np.isfinite(all.sum(1))],
        )

        df = pd.DataFrame(all, index=source_targets_)
    elif gene_order_method == "maximum":
        jacobian_mat = lowess_smoother(time, jacobian_mat, spaced_num=None, n_convolve=n_convolve)
        jacobian_mat = jacobian_mat[np.isfinite(jacobian_mat.sum(1)), :]

        if standard_scale is not None:
            exprs = (jacobian_mat - np.min(jacobian_mat, axis=standard_scale)[:, None]) / np.ptp(
                jacobian_mat, axis=standard_scale
            )[:, None]
        max_sort = np.argsort(np.argmax(exprs, axis=1))
        df = pd.DataFrame(
            exprs[max_sort, :],
            index=np.array(source_targets_)[max_sort],
            columns=adata.obs_names,
        )
    elif gene_order_method == "raw":
        jacobian_mat /= np.abs(jacobian_mat).max(1)[:, None]
        df = pd.DataFrame(
            jacobian_mat,
            index=np.array(source_targets_),
            columns=adata.obs_names,
        )
    else:
        raise Exception("gene order_method can only be either half_max_ordering or maximum")

    heatmap_kwargs = dict(
        xticklabels=False,
        yticklabels=1,
        row_colors=None,
        col_colors=None,
        row_linkage=None,
        col_linkage=None,
        method="average",
        metric="euclidean",
        z_score=None,
        standard_scale=None,
    )
    if kwargs is not None:
        heatmap_kwargs = update_dict(heatmap_kwargs, kwargs)

    sns_heatmap = sns.clustermap(
        df.loc[source_targets, :],
        col_cluster=cluster_row_col[0],
        row_cluster=cluster_row_col[1] if len(source_targets) > 2 else False,
        cmap=color_map,
        figsize=figsize,
        center=0,
        **heatmap_kwargs,
    )
    if not show_colorbar:
        sns_heatmap.cax.set_visible(False)

    if save_show_or_return == "save_fig":
        s_kwargs = {
            "path": None,
            "prefix": "jacobian_kinetics",
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        if show_colorbar:
            plt.subplots_adjust(right=0.85)
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return sns_heatmap


@docstrings.with_indent(4)
def sensitivity_kinetics(
    adata,
    basis="umap",
    regulators=None,
    effectors=None,
    mode="pseudotime",
    tkey="potential",
    color_map="bwr",
    gene_order_method="raw",
    show_colorbar=False,
    cluster_row_col=[False, True],
    figsize=(11.5, 6),
    standard_scale=1,
    n_convolve=30,
    save_show_or_return="show",
    save_kwargs={},
    **kwargs,
):
    """Plot the Sensitivity dynamics over time (pseudotime or inferred real time) in a heatmap.

    Note that by default `potential` estimated with the diffusion graph built from reconstructed vector field will be
    used as the measure of pseudotime.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        basis: `str`
            The reduced dimension basis.
        regulators: `list` or `None` (default: `None`)
            The list of genes that will be used as regulators for plotting the Jacobian heatmap, only limited to genes
            that have already performed Jacobian analysis.
        effectors: `List` or `None` (default: `None`)
            The list of genes that will be used as targets for plotting the Jacobian heatmap, only limited to genes
            that have already performed Jacobian analysis.
        mode: `str` (default: `vector_field`)
            Which data mode will be used, either vector_field or pseudotime. if mode is vector_field, the trajectory
            predicted byvector field function will be used, otherwise pseudotime trajectory (defined by time argument)
            will be used. By default `potential` estimated with the diffusion graph built reconstructed vector field
            will be used as pseudotime.
        tkey: `str` (default: `potential`)
            The .obs column that will be used for timing each cell, only used when mode is not `vector_field`.
        color_map: `str` (default: `BrBG`)
            Color map that will be used to color the gene expression. If `half_max_ordering` is True, the
            color map need to be divergent, good examples, include `BrBG`, `RdBu_r` or `coolwarm`, etc.
        gene_order_method: `str` (default: `half_max_ordering`) [`half_max_ordering`, `maximum`]
            Supports two different methods for ordering genes when plotting the heatmap: either `half_max_ordering`,
            or `maximum`. For `half_max_ordering`, it will order genes into up, down and transit groups by the half
            max ordering algorithm (HA Pliner, et. al, Molecular cell 71 (5), 858-871. e8). While for `maximum`,
            it will order by the position of the highest gene expression.
        show_colorbar: `bool` (default: `False`)
            Whether to show the color bar.
        cluster_row_col: `[bool, bool]` (default: `[False, False]`)
            Whether to cluster the row or columns.
        figsize: `str` (default: `(11.5, 6)`
            Size of figure
        standard_scale: `int` (default: 1)
            Either 0 (rows, cells) or 1 (columns, genes). Whether or not to standardize that dimension, meaning for each
            row or column, subtract the minimum and divide each by its maximum.
        n_convolve: `int` (default: 30)
            Number of cells for convolution.
        save_show_or_return: {'show', 'save_fig', 'return'} (default: `show`)
            Whether to save_fig, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
            save_fig function will use the {"path": None, "prefix": 'kinetic_curves', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a
            dictionary that properly modify those keys according to your needs.
        kwargs:
            All other keyword arguments are passed to heatmap(). Currently `xticklabels=False, yticklabels='auto'` is
            passed to heatmap() by default.
    Returns
    -------
        Nothing but plots a heatmap that shows the element of Jacobian matrix dynamics over time (potential decreasing).

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> adata = dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.vf.VectorField(adata, basis='pca')
    >>> valid_gene_list = adata[:, adata.var.use_for_transition].var.index[:2]
    >>> dyn.vf.sensitivity(adata, regulators=valid_gene_list[0], effectors=valid_gene_list[1])
    >>> dyn.pl.sensitivity_kinetics(adata)
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    Sensitivity_ = "sensitivity" if basis is None else "sensitivity_" + basis
    Der, cell_indx, _, regulators_, effectors_ = (
        adata.uns[Sensitivity_].get("sensitivity"),
        adata.uns[Sensitivity_].get("cell_idx"),
        adata.uns[Sensitivity_].get("sensitivity_gene"),
        adata.uns[Sensitivity_].get("regulators"),
        adata.uns[Sensitivity_].get("effectors"),
    )
    if tkey == "potential" and "potential" not in adata.obs_keys():
        ddhodge(adata)

    adata_ = adata[cell_indx, :]
    time = adata_.obs[tkey]
    sensitivity_mat = Der.reshape((-1, Der.shape[2])) if Der.ndim == 3 else Der[None, :]
    n_source_targets_ = Der.shape[0] * Der.shape[1] if Der.ndim == 3 else 1
    targets_, sources_ = (
        (
            np.repeat(effectors_, Der.shape[1]),
            np.tile(regulators_, Der.shape[0]),
        )
        if Der.ndim == 3
        else (
            np.repeat(effectors_, Der.shape[0]),
            np.repeat(effectors_, Der.shape[0]),
        )
    )
    source_targets_ = [sources_[i] + "->" + targets_[i] for i in range(n_source_targets_)]

    regulators = regulators_ if regulators is None else regulators
    effectors = effectors_ if effectors is None else effectors
    if type(regulators) == str:
        regulators = [regulators]
    if type(effectors) == str:
        effectors = [effectors]
    regulators = list(set(regulators_).intersection(regulators))
    effectors = list(set(effectors_).intersection(effectors))
    if len(regulators) == 0 or len(effectors) == 0:
        raise ValueError(
            f"Sensitivity related to source genes {regulators} and target genes {effectors}"
            f"you provided are existed. Available source genes includes {regulators_} while "
            f"available target genes includes {effectors_}"
        )
    n_source_targets = len(regulators) * len(effectors)
    targets, sources = (
        np.repeat(effectors, len(regulators)),
        np.tile(regulators, len(effectors)),
    )
    source_targets = [sources[i] + "->" + targets[i] for i in range(n_source_targets)]

    sensitivity_mat = sensitivity_mat[:, np.argsort(time)]

    if gene_order_method == "half_max_ordering":
        time, all, valid_ind, gene_idx = _half_max_ordering(
            sensitivity_mat, time, mode=mode, interpolate=True, spaced_num=100
        )
        all, source_targets = (
            all[np.isfinite(all.sum(1)), :],
            np.array(source_targets)[gene_idx][np.isfinite(all.sum(1))],
        )

        df = pd.DataFrame(all, index=source_targets_)
    elif gene_order_method == "maximum":
        sensitivity_mat = lowess_smoother(time, sensitivity_mat, spaced_num=None, n_convolve=n_convolve)
        sensitivity_mat = sensitivity_mat[np.isfinite(sensitivity_mat.sum(1)), :]

        if standard_scale is not None:
            exprs = (sensitivity_mat - np.min(sensitivity_mat, axis=standard_scale)[:, None]) / np.ptp(
                sensitivity_mat, axis=standard_scale
            )[:, None]
        max_sort = np.argsort(np.argmax(exprs, axis=1))
        df = pd.DataFrame(
            exprs[max_sort, :],
            index=np.array(source_targets_)[max_sort],
            columns=adata.obs_names,
        )
    elif gene_order_method == "raw":
        sensitivity_mat /= np.abs(sensitivity_mat).max(1)[:, None]
        df = pd.DataFrame(
            sensitivity_mat,
            index=np.array(source_targets_),
            columns=adata.obs_names,
        )
    else:
        raise Exception("gene order_method can only be either half_max_ordering or maximum")

    heatmap_kwargs = dict(
        xticklabels=False,
        yticklabels=1,
        row_colors=None,
        col_colors=None,
        row_linkage=None,
        col_linkage=None,
        method="average",
        metric="euclidean",
        z_score=None,
        standard_scale=None,
    )
    if kwargs is not None:
        heatmap_kwargs = update_dict(heatmap_kwargs, kwargs)

    sns_heatmap = sns.clustermap(
        df.loc[source_targets, :],
        col_cluster=cluster_row_col[0],
        row_cluster=cluster_row_col[1] if len(source_targets) > 2 else False,
        cmap=color_map,
        figsize=figsize,
        center=0,
        **heatmap_kwargs,
    )
    if not show_colorbar:
        sns_heatmap.cax.set_visible(False)

    if save_show_or_return == "save_fig":
        s_kwargs = {
            "path": None,
            "prefix": "sensitivity_kinetics",
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        if show_colorbar:
            plt.subplots_adjust(right=0.85)
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return sns_heatmap
