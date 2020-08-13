# code adapted from https://github.com/lmcinnes/umap/blob/7e051d8f3c4adca90ca81eb45f6a9d1372c076cf/umap/plot.py
import warnings
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical

from scipy.sparse import issparse
from numbers import Number

import matplotlib.colors
import matplotlib.cm

from ..configuration import _themes, set_figure_params, reset_rcParams
from .utils import (
    despline,
    despline_all,
    deaxis_all,
    set_spine_linewidth,
    scatter_with_colorbar,
    scatter_with_legend,
    _select_font_color,
    arrowed_spines,
    is_gene_name,
    is_cell_anno_column,
    is_list_of_lists,
    is_layer_keys,
    _matplotlib_points,
    _datashade_points,
    save_fig,
)
from ..tools.utils import (
    update_dict,
    get_mapper,
    log1p_,
    flatten,
)
from ..tools.moments import calc_1nd_moment

from ..docrep import DocstringProcessor

docstrings = DocstringProcessor()


def _scatters(
    adata,
    genes,
    x=0,
    y=1,
    theme=None,
    type="expression",
    velocity_key="S",
    ekey="X",
    basis="umap",
    n_columns=1,
    color=None,
    pointsize=None,
    figsize=None,
    legend="on data",
    ax=None,
    normalize=False,
    log1p=True,
    **kwargs
):
    """Scatter plot of cells for phase portrait or for low embedding embedding, colored by gene expression, velocity or cell groups.

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
        theme: `str` (optional, default `None`)
            A color theme to use for plotting. A small set of
            predefined themes are provided which have relatively
            good aesthetics. Available themes are:
               * 'blue'
               * 'red'
               * 'green'
               * 'inferno'
               * 'fire'
               * 'viridis'
               * 'darkblue'
               * 'darkred'
               * 'darkgreen'
        type: `str` (default: `expression`)
            Which plotting type to use, either expression, embedding, velocity or phase.
                * 'embedding': visualize low dimensional embedding of cells, cells can be colored by different annotations,
                  each within its own panel
                * 'expression': visualize low dimensional embedding of cells, cells can be colored by expression of different
                  genes, each within its own panel
                * 'velocity': visualize low dimensional embedding of cells, cells can be colored by velocity values of different
                  genes, each within its own panel.
                * 'phase': visualize S-U (spliced vs. unspliced) or P-S (protein vs. spliced) plot for different genes across cells,
                  each gene has its own panel. Cell can be colored by a single annotation.
        velocity_key: `string` (default: `S`)
            Which (suffix of) the velocity key used for visualizing the magnitude of velocity. Can be either in the layers attribute or the
            keys in the obsm attribute. The full key name can be retrieved by `vkey + '_velocity'`.
        ekey: `str`
            The layer of data to represent the gene expression level.
        basis: `string` (default: umap)
            Which low dimensional embedding will be used to visualize the cell.
        pointsize: `None` or `float` (default: None)
            The scale of the point size. Actual point cell size is calculated as `500.0 / np.sqrt(adata.shape[0]) * pointsize`
        figsize: `None` or `[float, float]` (default: None)
            The width and height of a figure.
        legend: `str` (default: `on data`)
            Where to put the legend.  Legend is drawn by seaborn with “brief” mode, numeric hue and size variables will be
            represented with a sample of evenly spaced values. By default legend is drawn on top of cells.
        normalize: `bool` (default: `True`)
            Whether to normalize the expression / velocity or other continous data so that the value will be scaled between 0 and 1.
        log1p: `bool` (default: `True`)
            Whether to log1p transform the expression so that visualization can be robust to extreme values.
        **kwargs:
            Additional parameters that will be passed to plt.scatter function

    Returns
    -------
        Nothing but a scatter plot of cells.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    mapper = get_mapper()

    point_size = (
        500.0 / np.sqrt(adata.shape[0])
        if pointsize is None
        else 500.0 / np.sqrt(adata.shape[0]) * pointsize
    )
    scatter_kwargs = dict(
        alpha=0.4, s=point_size, edgecolor=None, linewidth=0
    )  # (0, 0, 0, 1)
    if kwargs is not None:
        scatter_kwargs.update(kwargs)

    genes, idx = (
        adata.var.index[adata.var.index.isin(genes)].tolist(),
        np.where(adata.var.index.isin(genes))[0],
    )
    if len(genes) == 0:
        raise Exception(
            "adata has no genes listed in your input gene vector: {}".format(genes)
        )
    if not "X_" + basis in adata.obsm.keys():
        raise Exception("{} is not applied to adata.".format(basis))
    else:
        embedding = pd.DataFrame(
            {
                basis + "_" + str(x): adata.obsm["X_" + basis][:, x],
                basis + "_" + str(y): adata.obsm["X_" + basis][:, y],
            }
        )
        embedding.columns = ["dim_1", "dim_2"]

    if all([i in adata.layers.keys() for i in ["X_new", "X_total"]]):
        mode = "labeling"
    elif all([i in adata.layers.keys() for i in ["X_spliced", "X_unspliced"]]):
        mode = "splicing"
    elif all([i in adata.layers.keys() for i in ["X_uu", "X_ul", "X_su", "X_sl"]]):
        mode = "full"
    else:
        raise Exception(
            "your data should be in one of the proper mode: labelling (has X_new/X_total layers), splicing "
            "(has X_spliced/X_unspliced layers) or full (has X_uu/X_ul/X_su/X_sl layers)"
        )

    layers = list(adata.layers.keys())
    layers.extend(["X", "protein", "X_protein"])
    if ekey in layers:
        if ekey == "X":
            E_vec = (
                adata[:, genes].layers[mapper["X"]]
                if mapper["X"] in adata.layers.keys()
                else adata[:, genes].X
            )
        elif ekey in ["protein", "X_protein"]:
            E_vec = (
                adata[:, genes].layers[mapper[ekey]]
                if (ekey in mapper.keys()) and (mapper[ekey] in adata.obsm_keys())
                else adata[:, genes].obsm[ekey]
            )
        else:
            E_vec = (
                adata[:, genes].layers[mapper[ekey]]
                if (ekey in mapper.keys()) and (mapper[ekey] in adata.layers.keys())
                else adata[:, genes].layers[ekey]
            )
        
        if log1p: E_vec = log1p_(adata, E_vec)

    n_cells, n_genes = adata.shape[0], len(genes)

    color_vec = np.repeat(np.nan, n_cells)
    if color is not None:
        color = list(set(color).intersection(adata.obs.keys()))
        if len(color) > 0 and type != "embedding":
            color_vec = adata.obs[color[0]].values
        else:
            n_genes = len(color)  # set n_genes as the number of obs keys
            color_vec = adata.obs[color[0]].values
            full_color_vec = adata.obs[color].values.flatten()

    if "velocity_" not in velocity_key:
        vkey = "velocity_" + velocity_key

    if type == "embedding":
        df = pd.DataFrame(
            {
                basis + "_0": np.repeat(embedding.iloc[:, 0], n_genes),
                basis + "_1": np.repeat(embedding.iloc[:, 1], n_genes),
                "color": full_color_vec,
                "group": np.tile(color, n_cells),
            }
        )
    else:
        if vkey == "velocity_U":
            V_vec = adata[:, genes].layers["velocity_U"]
            if "velocity_P" in adata.obsm.keys():
                P_vec = adata[:, genes].layer["velocity_P"]
        elif vkey == "velocity_S":
            V_vec = adata[:, genes].layers["velocity_S"]
            if "velocity_P" in adata.obsm.keys():
                P_vec = adata[:, genes].layers["velocity_P"]
        else:
            raise Exception(
                "adata has no vkey {} in either the layers or the obsm attribute".format(
                    vkey
                )
            )

        if issparse(E_vec):
            E_vec, V_vec = E_vec.A, V_vec.A

        if "gamma" in adata.var.columns:
            gamma = adata.var.gamma[genes].values
            velocity_offset = (
                [0] * n_genes
                if not ("gamma_b" in adata.var.columns)
                else adata.var.gamma_b[genes].values
            )
        else:
            raise Exception(
                "adata does not seem to have gamma column. Velocity estimation is required before "
                "running this function."
            )

        if mode == "labelling":
            new_mat, tot_mat = (
                adata[:, genes].layers["X_new"],
                adata[:, genes].layers["X_total"],
            )
            
            if log1p: new_mat, tot_mat = (log1p_(adata, new_mat), log1p_(adata, tot_mat))
            
            new_mat, tot_mat = (
                (new_mat.A, tot_mat.A) if issparse(new_mat) else (new_mat, tot_mat)
            )

            df = pd.DataFrame(
                {
                    "new": new_mat.flatten(),
                    "total": tot_mat.flatten(),
                    "gene": genes * n_cells,
                    "gamma": np.tile(gamma, n_cells),
                    "velocity_offset": np.tile(velocity_offset, n_cells),
                    "expression": E_vec.flatten(),
                    "velocity": V_vec.flatten(),
                    "color": np.repeat(color_vec, n_genes),
                },
                index=range(n_cells * n_genes),
            )

        elif mode == "splicing":
            unspliced_mat, spliced_mat = (
                adata[:, genes].layers["X_unspliced"],
                adata[:, genes].layers["X_spliced"],
            )

            if log1p: unspliced_mat, spliced_mat = (log1p_(adata, unspliced_mat), log1p_(adata, spliced_mat))
            
            unspliced_mat, spliced_mat = (
                (unspliced_mat.A, spliced_mat.A)
                if issparse(unspliced_mat)
                else (unspliced_mat, spliced_mat)
            )

            df = pd.DataFrame(
                {
                    "unspliced": unspliced_mat.flatten(),
                    "spliced": spliced_mat.flatten(),
                    "gene": genes * n_cells,
                    "gamma": np.tile(gamma, n_cells),
                    "velocity_offset": np.tile(velocity_offset, n_cells),
                    "expression": E_vec.flatten(),
                    "velocity": V_vec.flatten(),
                    "color": np.repeat(color_vec, n_genes),
                },
                index=range(n_cells * n_genes),
            )

        elif mode == "full":
            uu, ul, su, sl = (
                adata[:, genes].layers["X_uu"],
                adata[:, genes].layers["X_ul"],
                adata[:, genes].layers["X_su"],
                adata[:, genes].layers["X_sl"],
            )

            if log1p: uu, ul, su, sl = (log1p_(adata, uu), log1p_(adata, ul), log1p_(adata, su), log1p_(adata, sl))
            
            if "protein" in adata.obsm.keys():
                if "delta" in adata.var.columns:
                    gamma_P = adata.var.delta[genes].values
                    velocity_offset_P = (
                        [0] * n_cells
                        if not ("delta_b" in adata.var.columns)
                        else adata.var.delta_b[genes].values
                    )
                else:
                    raise Exception(
                        "adata does not seem to have velocity_gamma column. Velocity estimation is required before "
                        "running this function."
                    )

                P = (
                    adata[:, genes].obsm["X_protein"]
                    if ["X_protein"] in adata.obsm.keys()
                    else adata[:, genes].obsm["protein"]
                )
                uu, ul, su, sl, P = (
                    (uu.A, ul.A, su.A, sl.A, P.A)
                    if issparse(uu)
                    else (uu, ul, su, sl, P)
                )
                if issparse(P_vec):
                    P_vec = P_vec.A

                # df = pd.DataFrame({"uu": uu.flatten(), "ul": ul.flatten(), "su": su.flatten(), "sl": sl.flatten(), "P": P.flatten(),
                #                    'gene': genes * n_cells, 'prediction': np.tile(gamma, n_cells) * uu.flatten() +
                #                     np.tile(velocity_offset, n_cells), "velocity": genes * n_cells}, index=range(n_cells * n_genes))
                df = pd.DataFrame(
                    {
                        "new": (ul + sl).flatten(),
                        "total": (uu + ul + sl + su).flatten(),
                        "S": (sl + su).flatten(),
                        "P": P.flatten(),
                        "gene": genes * n_cells,
                        "gamma": np.tile(gamma, n_cells),
                        "velocity_offset": np.tile(velocity_offset, n_cells),
                        "gamma_P": np.tile(gamma_P, n_cells),
                        "velocity_offset_P": np.tile(velocity_offset_P, n_cells),
                        "expression": E_vec.flatten(),
                        "velocity": V_vec.flatten(),
                        "velocity_protein": P_vec.flatten(),
                        "color": np.repeat(color_vec, n_genes),
                    },
                    index=range(n_cells * n_genes),
                )
            else:
                df = pd.DataFrame(
                    {
                        "new": (ul + sl).flatten(),
                        "total": (uu + ul + sl + su).flatten(),
                        "gene": genes * n_cells,
                        "gamma": np.tile(gamma, n_cells),
                        "velocity_offset": np.tile(velocity_offset, n_cells),
                        "expression": E_vec.flatten(),
                        "velocity": V_vec.flatten(),
                        "color": np.repeat(color_vec, n_genes),
                    },
                    index=range(n_cells * n_genes),
                )
        else:
            raise Exception(
                "Your adata is corrupted. Make sure that your layer has keys new, old for the labelling mode, "
                "spliced, ambiguous, unspliced for the splicing model and uu, ul, su, sl for the full mode"
            )

    if type == "embedding":
        if theme is None:
            if (
                True
            ):  # should also support continous mapping if the color key is not categorical
                import colorcet

                theme = "glasbey_dark"
                # add glasbey_bw_minc_20_maxl_70 theme for cell annotation in dark background
                glasbey_dark_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    "glasbey_dark_cmap", colorcet.glasbey_bw_minc_20_maxl_70
                )
                plt.register_cmap("glasbey_dark", glasbey_dark_cmap)

            else:
                theme = "inferno"
        cmap = _themes[theme]["cmap"]
        color_key_cmap = _themes[theme]["color_key_cmap"]
        background = _themes[theme]["background"]
    else:
        if type == "phase":
            if all(df.color.unique() != np.nan):  # color corresponds to expression
                if theme is None:
                    theme = "inferno"
                cmap = _themes[theme]["cmap"]
                color_key_cmap = _themes[theme]["color_key_cmap"]
                background = _themes[theme]["background"]

                # num_labels = unique_labels.shape[0]
                # color_key = plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
                # legend_elements = [
                #     Patch(facecolor=color_key[i], label=unique_labels[i])
                #     for i, k in enumerate(unique_labels)
                # ]

            else:
                if theme is None:
                    import colorcet

                    theme = "glasbey_dark"
                    # add glasbey_bw_minc_20_maxl_70 theme for cell annotation in dark background
                    glasbey_dark_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                        "glasbey_dark_cmap", colorcet.glasbey_bw_minc_20_maxl_70
                    )
                    plt.register_cmap("glasbey_dark", glasbey_dark_cmap)

                cmap = _themes[theme]["cmap"]
                color_key_cmap = _themes[theme]["color_key_cmap"]
                background = _themes[theme]["background"]
        elif type == "velocity":
            if theme is None:
                import colorcet

                theme = "div_blue_red"
                # add RdBu_r theme for velocity
                div_blue_red_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    "div_blue_red", colorcet.diverging_bwr_55_98_c37
                )
                plt.register_cmap("div_blue_red", div_blue_red_cmap)
            cmap = _themes[theme]["cmap"]
            color_key_cmap = _themes[theme]["color_key_cmap"]
            background = _themes[theme]["background"]
        elif type == "expression":
            if theme is None:
                theme = "inferno"
            cmap = _themes[theme]["cmap"]
            color_key_cmap = _themes[theme]["color_key_cmap"]
            background = _themes[theme]["background"]

    font_color = _select_font_color(background)
    if background == "black":
        # https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mpl-data/stylelib/dark_background.mplstyle
        sns.set(
            rc={
                "axes.facecolor": background,
                "axes.edgecolor": background,
                "figure.facecolor": background,
                "figure.edgecolor": background,
                "axes.grid": False,
                "ytick.color": "w",
                "xtick.color": "w",
                "axes.labelcolor": "w",
                "axes.edgecolor": "w",
                "savefig.facecolor": "k",
                "savefig.edgecolor": "k",
                "grid.color": "w",
                "text.color": "w",
                "lines.color": "w",
                "patch.edgecolor": "w",
                "figure.edgecolor": "w",
            }
        )
    else:
        sns.set(
            rc={
                "axes.facecolor": background,
                "figure.facecolor": background,
                "axes.grid": False,
            }
        )

    n_columns = min(n_genes, n_columns)
    if type == "embedding":
        nrow, ncol = int(np.ceil(len(color) / n_columns)), n_columns

        if figsize is None:
            figsize = plt.rcParams["figsize"]
        fig = plt.figure(
            None, (figsize[0] * ncol, figsize[1] * nrow), facecolor=background
        )

        # the following code is inspired by https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb
        gs = plt.GridSpec(nrow, ncol)

        for i, clr in enumerate(color):
            ax1 = plt.subplot(gs[i])

            cur_pd = df.loc[df.group == clr, :]

            scatter_with_legend(
                fig,
                ax1,
                df,
                font_color,
                embedding.iloc[:, 0],
                embedding.iloc[:, 1],
                cur_pd.loc[:, "color"],
                plt.get_cmap(color_key_cmap),
                legend,
                **scatter_kwargs
            )

            set_spine_linewidth(ax1, 1)
            ax1.set_title(clr)
            ax1.set_xlabel(basis + "_1")
            ax1.set_ylabel(basis + "_2")
    else:
        n_columns = (
            2 * n_columns
            if ("protein" in adata.obsm.keys() and mode == "full")
            else n_columns
        )
        plot_per_gene = 2 if ("protein" in adata.obsm.keys() and mode == "full") else 1
        nrow, ncol = int(np.ceil(plot_per_gene * n_genes / n_columns)), n_columns
        fig = plt.figure(
            None, (figsize[0] * ncol, figsize[1] * nrow), facecolor=background
        )

        # the following code is inspired by https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb
        gs = plt.GridSpec(nrow, ncol)

        for i, gn in enumerate(genes):
            if plot_per_gene == 2:
                ax1, ax2 = plt.subplot(gs[i * 2]), plt.subplot(gs[i * 2 + 1])
            elif plot_per_gene == 1:
                ax1 = plt.subplot(gs[i])
            try:
                ix = np.where(adata.var.index == gn)[0][0]
            except:
                continue
            cur_pd = df.loc[df.gene == gn, :]
            if type == "phase":  # viridis, set2
                if all(cur_pd.color.unique() == np.nan):
                    fig, ax1 = scatter_with_colorbar(
                        fig,
                        ax1,
                        cur_pd.iloc[:, 1],
                        cur_pd.iloc[:, 0],
                        cur_pd.color,
                        cmap,
                        **scatter_kwargs
                    )

                else:
                    fig, ax1 = scatter_with_colorbar(
                        fig,
                        ax1,
                        cur_pd.iloc[:, 1],
                        cur_pd.iloc[:, 0],
                        cur_pd.color,
                        cmap,
                        **scatter_kwargs
                    )

                set_spine_linewidth(ax1, 1)
                ax1.set_title(gn)
                xnew = np.linspace(0, cur_pd.iloc[:, 1].max())
                ax1.plot(
                    xnew,
                    xnew * cur_pd.loc[:, "gamma"].unique()
                    + cur_pd.loc[:, "velocity_offset"].unique(),
                    linewidth=2,
                )
                ax1.set_xlim(0, np.max(cur_pd.iloc[:, 1]) * 1.02)
                ax1.set_ylim(0, np.max(cur_pd.iloc[:, 0]) * 1.02)

                despline(ax1)  # sns.despline()
                set_spine_linewidth(ax1, 1)
                ax1.set_xlabel("spliced")
                ax1.set_ylabel("unspliced")

                if plot_per_gene == 2 and (
                    "protein" in adata.obsm.keys()
                    and mode == "full"
                    and all(
                        [i in adata.layers.keys() for i in ["uu", "ul", "su", "sl"]]
                    )
                ):
                    fig, ax2 = scatter_with_colorbar(
                        fig,
                        ax2,
                        cur_pd.iloc[:, 3],
                        cur_pd.iloc[:, 2],
                        cur_pd.color,
                        cmap,
                        **scatter_kwargs
                    )

                    set_spine_linewidth(ax2, 1)
                    ax2.set_title(gn)
                    xnew = np.linspace(0, cur_pd.iloc[:, 3].max())
                    ax2.plot(
                        xnew,
                        xnew * cur_pd.loc[:, "gamma_P"].unique()
                        + cur_pd.loc[:, "velocity_offset_P"].unique(),
                        linewidth=2,
                    )

                    ax2.set_ylim(0, np.max(cur_pd.iloc[:, 3]) * 1.02)
                    ax2.set_xlim(0, np.max(cur_pd.iloc[:, 2]) * 1.02)

                    despline(ax2)  # sns.despline()
                    ax1.set_xlabel("spliced")
                    ax1.set_ylabel("unspliced")

            elif type == "velocity":
                V_vec = cur_pd.loc[:, "velocity"]
                if normalize:
                    limit = np.nanmax(
                        np.abs(np.nanpercentile(V_vec, [1, 99]))
                    )  # upper and lowe limit / saturation

                    # transform the data so that 0.5 corresponds to 0 in the original data space.
                    V_vec = V_vec + limit  # that is: tmp_colorandum - (-limit)
                    V_vec = V_vec / (
                        2 * limit
                    )  # that is: tmp_colorandum / (limit - (-limit))
                    V_vec = np.clip(V_vec, 0, 1)

                # cmap = plt.cm.RdBu_r # sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
                fig, ax1 = scatter_with_colorbar(
                    fig,
                    ax1,
                    embedding.iloc[:, 0],
                    embedding.iloc[:, 1],
                    V_vec,
                    cmap,
                    **scatter_kwargs
                )

                set_spine_linewidth(ax1, 1)
                ax1.set_title(gn + " (" + vkey + ")")
                ax1.set_xlabel(basis + "_1")
                ax1.set_ylabel(basis + "_2")

                if plot_per_gene == 2:
                    V_vec = cur_pd.loc[:, "velocity_offset_P"]

                    if normalize:
                        limit = np.nanmax(
                            np.abs(np.nanpercentile(V_vec, [1, 99]))
                        )  # upper and lowe limit / saturation

                        V_vec = V_vec + limit  # that is: tmp_colorandum - (-limit)
                        V_vec = V_vec / (
                            2 * limit
                        )  # that is: tmp_colorandum / (limit - (-limit))
                        V_vec = np.clip(V_vec, 0, 1)

                    # cmap = plt.cm.RdBu_r  # sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
                    fig, ax2 = scatter_with_colorbar(
                        fig,
                        ax2,
                        embedding.iloc[:, 0],
                        embedding.iloc[:, 1],
                        V_vec,
                        cmap,
                        **scatter_kwargs
                    )

                    set_spine_linewidth(ax2, 1)
                    ax2.set_title(gn + " (" + vkey + ")")
                    ax2.set_xlabel(basis + "_1")
                    ax2.set_ylabel(basis + "_2")

            elif type == "expression":
                # cmap = plt.cm.Greens # sns.diverging_palette(10, 220, sep=80, as_cmap=True)
                expression = cur_pd.loc[:, "expression"]
                if normalize:
                    expression = np.clip(
                        expression / np.percentile(expression, 99), 0, 1
                    )
                fig, ax1 = scatter_with_colorbar(
                    fig,
                    ax1,
                    embedding.iloc[:, 0],
                    embedding.iloc[:, 1],
                    expression,
                    cmap,
                    **scatter_kwargs
                )

                set_spine_linewidth(ax1, 1)
                ax1.set_title(gn + " (" + ekey + ")")
                ax1.set_xlabel(basis + "_1")
                ax1.set_ylabel(basis + "_2")

                if (
                    "protein" in adata.obsm.keys()
                    and mode == "full"
                    and all(
                        [i in adata.layers.keys() for i in ["uu", "ul", "su", "sl"]]
                    )
                ):
                    expression = cur_pd.loc[:, "P"]
                    if normalize:
                        expression = np.clip(
                            expression / np.percentile(expression, 99), 0, 1
                        )

                    fig, ax2 = scatter_with_colorbar(
                        fig,
                        ax2,
                        embedding.iloc[:, 0],
                        embedding.iloc[:, 1],
                        expression,
                        cmap,
                        **scatter_kwargs
                    )

                    set_spine_linewidth(ax2, 1)
                    ax2.set_title(gn + " (protein expression)")
                    ax2.set_xlabel(basis + "_1")
                    ax2.set_ylabel(basis + "_2")

    plt.tight_layout()
    plt.show()


@docstrings.get_sectionsf("scatters")
def scatters(
    adata,
    basis="umap",
    x=0,
    y=1,
    color='ntr',
    layer="X",
    highlights=None,
    labels=None,
    values=None,
    theme=None,
    cmap=None,
    color_key=None,
    color_key_cmap=None,
    background=None,
    ncols=4,
    pointsize=None,
    figsize=(6, 4),
    show_legend="on data",
    use_smoothed=True,
    aggregate=None,
    show_arrowed_spines=False,
    ax=None,
    sort='raw',
    save_show_or_return="show",
    save_kwargs={},
    return_all=False,
    add_gamma_fit=False,
    frontier=False,
    contour=False,
    ccmap=None,
    calpha=2.3,
    sym_c=False,
    smooth=False,
    **kwargs
):
    """Plot an embedding as points. Currently this only works
    for 2D embeddings. While there are many optional parameters
    to further control and tailor the plotting, you need only
    pass in the trained/fit umap model to get results. This plot
    utility will attempt to do the hard work of avoiding
    overplotting issues, and make it easy to automatically
    colour points by a categorical labelling or numeric values.
    This method is intended to be used within a Jupyter
    notebook with ``%matplotlib inline``.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        basis: `str`
            The reduced dimension.
        x: `int` (default: `0`)
            The column index of the low dimensional embedding for the x-axis.
        y: `int` (default: `1`)
            The column index of the low dimensional embedding for the y-axis.
        color: `string` (default: `ntr`)
            Any column names or gene expression, etc. that will be used for coloring cells.
        layer: `str` (default: `X`)
            The layer of data to use for the scatter plot.
        highlights: `list` (default: None)
            Which color group will be highlighted. if highligts is a list of lists - each list is relate to each color element.
        labels: array, shape (n_samples,) (optional, default None)
            An array of labels (assumed integer or categorical),
            one for each data sample.
            This will be used for coloring the points in
            the plot according to their label. Note that
            this option is mutually exclusive to the ``values``
            option.
        values: array, shape (n_samples,) (optional, default None)
            An array of values (assumed float or continuous),
            one for each sample.
            This will be used for coloring the points in
            the plot according to a colorscale associated
            to the total range of values. Note that this
            option is mutually exclusive to the ``labels``
            option.
        theme: string (optional, default None)
            A color theme to use for plotting. A small set of
            predefined themes are provided which have relatively
            good aesthetics. Available themes are:
               * 'blue'
               * 'red'
               * 'green'
               * 'inferno'
               * 'fire'
               * 'viridis'
               * 'darkblue'
               * 'darkred'
               * 'darkgreen'
        cmap: string (optional, default 'Blues')
            The name of a matplotlib colormap to use for coloring
            or shading points. If no labels or values are passed
            this will be used for shading points according to
            density (largely only of relevance for very large
            datasets). If values are passed this will be used for
            shading according the value. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        color_key: dict or array, shape (n_categories) (optional, default None)
            A way to assign colors to categoricals. This can either be
            an explicit dict mapping labels to colors (as strings of form
            '#RRGGBB'), or an array like object providing one color for
            each distinct category being provided in ``labels``. Either
            way this mapping will be used to color points according to
            the label. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        color_key_cmap: string (optional, default 'Spectral')
            The name of a matplotlib colormap to use for categorical coloring.
            If an explicit ``color_key`` is not given a color mapping for
            categories can be generated from the label list and selecting
            a matching list of colors from the given colormap. Note
            that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        background: string or None (optional, default 'None`)
            The color of the background. Usually this will be either
            'white' or 'black', but any color name will work. Ideally
            one wants to match this appropriately to the colors being
            used for points etc. This is one of the things that themes
            handle for you. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        ncols: int (optional, default `4`)
            Number of columns for the figure.
        pointsize: `None` or `float` (default: None)
            The scale of the point size. Actual point cell size is calculated as `500.0 / np.sqrt(adata.shape[0]) *
            pointsize`
        figsize: `None` or `[float, float]` (default: None)
            The width and height of a figure.
        show_legend: bool (optional, default True)
            Whether to display a legend of the labels
        use_smoothed: bool (optional, default True)
            Whether to use smoothed values (i.e. M_s / M_u instead of spliced / unspliced, etc.).
        aggregate: `str` or `None` (default: `None`)
            The column in adata.obs that will be used to aggregate data points.
        show_arrowed_spines: bool (optional, default False)
            Whether to show a pair of arrowed spines representing the basis of the scatter is currently using.
        ax: `matplotlib.Axis` (optional, default `None`)
            The matplotlib axes object where new plots will be added to. Only applicable to drawing a single component.
        sort: `str` (optional, default `raw`)
            The method to reorder data so that high values points will be on top of background points. Can be one of
            {'raw', 'abs', 'neg'}, i.e. sorted by raw data, sort by absolute values or sort by negative values.
        save_show_or_return: `str` {'save', 'show', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
            will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.
        return_all: `bool` (default: `False`)
            Whether to return all the scatter related variables. Default is False.
        add_gamma_fit: `bool` (default: `False`)
            Whether to add the line of the gamma fitting. This will automatically turn on if `basis` points to gene names
            and those genes have went through gamma fitting.
        frontier: `bool` (default: `False`)
            Whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to show area
            of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib tips & tricks
            cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from scEU-seq paper:
            https://science.sciencemag.org/content/367/6482/1151. If `contour` is set  to be True, `frontier` will be
            ignored as `contour` also add an outlier for data points.
        contour: `bool` (default: `False`)
            Whether to add an countor on top of scatter plots. We use `tricontourf` to plot contour for non-gridded data.
            The shapely package was used to create a polygon of the concave hull of the scatters. With the polygon we
            then check if the mean of the triangulated points are within the polygon and use this as our condition to
            form the mask to create the contour. We also add the polygon shape as a frontier of the data point (similar
            to when setting `frontier = True`). When the color of the data points is continuous, we will use the same cmap
            as for the scatter points by default, when color is categorical, no contour will be drawn but just the
            polygon. cmap can be set with `ccmap` argument. See below.
        ccmap: `str` or `None` (default: `None`)
            The name of a matplotlib colormap to use for coloring or shading points the contour. See above.
        calpha: `float` (default: `2.3`)
            alpha value for identifying the alpha hull to influence the gooeyness of the border. Smaller numbers don't
            fall inward as much as larger numbers. Too large, and you lose everything!
        sym_c: `bool` (default: `False`)
            Whether do you want to make the limits of continuous color to be symmetric, normally this should be used for
            plotting velocity, jacobian, curl, divergence or other types of data with both positive or negative values.
        smooth: `bool` or `int` (default: `False`)
            Whether do you want to further smooth data and how much smoothing do you want. If it is `False`, no smoothing
            will be applied. If `True`, smoothing based on one step diffusion of connectivity matrix (`.uns['moment_cnn']
            will be applied. If a number larger than 1, smoothing will based on `smooth` steps of diffusion.
        kwargs:
            Additional arguments passed to plt.scatters.

    Returns
    -------
        result: matplotlib axis
            The result is a matplotlib axis with the relevant plot displayed.
            If you are using a notbooks and have ``%matplotlib inline`` set
            then this will simply display inline.
    """

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if contour: frontier = False

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
        set_figure_params('dynamo', background=_background)
    else:
        _background = background
        set_figure_params('dynamo', background=_background)

    x, y = [x] if type(x) in [int, str] else x, [y] if type(y) in [int, str] else y
    if all([is_gene_name(adata, i) for i in basis]):
        if x[0] not in ['M_s', 'X_spliced'] and y[0] not in ['M_u', 'X_unspliced']:
            if ('M_s' in adata.layers.keys() and 'M_u' in adata.layers.keys()):
                x, y = ['M_s'], ['M_u']
            elif ('X_spliced' in adata.layers.keys() and 'X_unspliced' in adata.layers.keys()):
                x, y = ['X_spliced'], ['X_unspliced']
            else:
                x, y = ['spliced'], ['unspliced']

    if use_smoothed:
        mapper = get_mapper()

    # check layer, basis -> convert to list

    if type(color) is str:
        color = [color]
    if type(layer) is str:
        layer = [layer]
    if type(basis) is str:
        basis = [basis]
    n_c, n_l, n_b, n_x, n_y = (
        1 if color is None else len(color),
        1 if layer is None else len(layer),
        1 if basis is None else len(basis),
        1 if x is None else len(x),
        1 if y is None else len(y),
    )

    point_size = (
        16000.0 / np.sqrt(adata.shape[0])
        if pointsize is None
        else 16000.0 / np.sqrt(adata.shape[0]) * pointsize
    )

    scatter_kwargs = dict(
        alpha=0.1, s=point_size, edgecolor=None, linewidth=0, rasterized=True
    )  # (0, 0, 0, 1)
    if kwargs is not None:
        scatter_kwargs.update(kwargs)

    font_color = _select_font_color(_background)

    total_panels, ncols = n_c * n_l * n_b * n_x * n_y, min(max([n_c, n_l, n_b, n_x, n_y]), ncols)
    nrow, ncol = int(np.ceil(total_panels / ncols)), ncols
    if figsize is None:
        figsize = plt.rcParams["figsize"]

    if total_panels >= 1 and ax is None:
        plt.figure(None, (figsize[0] * ncol, figsize[1] * nrow), facecolor=_background)
        gs = plt.GridSpec(nrow, ncol, wspace=0.12)

    i = 0
    axes_list, color_list = [], []
    for cur_b in basis:
        for cur_l in layer:
            if use_smoothed:
                cur_l_smoothed = mapper[cur_l]
            prefix = cur_l + "_"

            # if prefix + cur_b in adata.obsm.keys():
            #     if type(x) != str and type(y) != str:
            #         x_, y_ = (
            #             adata.obsm[prefix + cur_b][:, int(x)],
            #             adata.obsm[prefix + cur_b][:, int(y)],
            #         )
            # else:
            #     continue
            for cur_c in color:
                cur_title = cur_c
                if cur_l in ["protein", "X_protein"]:
                    _color = adata.obsm[cur_l].loc[cur_c, :]
                else:
                    _color = adata.obs_vector(cur_c, layer=None) if cur_l == 'X' else adata.obs_vector(cur_c, layer=cur_l)
                for cur_x, cur_y in zip(x, y):
                    if type(cur_x) is int and type(cur_y) is int:
                        points = pd.DataFrame(
                            {
                                cur_b + "_0": adata.obsm[prefix + cur_b][:, cur_x],
                                cur_b + "_1": adata.obsm[prefix + cur_b][:, cur_y],
                            }
                        )
                        points.columns = [cur_b + "_0", cur_b + "_1"]
                    elif is_gene_name(adata, cur_x) and is_gene_name(adata, cur_y):
                        points = pd.DataFrame(
                            {
                                cur_x: adata.obs_vector(k=cur_x, layer=None) if cur_l_smoothed == 'X' else adata.obs_vector(k=cur_x, layer=cur_l_smoothed),
                                cur_y: adata.obs_vector(k=cur_y, layer=None) if cur_l_smoothed == 'X' else adata.obs_vector(k=cur_y, layer=cur_l_smoothed),
                            }
                        )
                        # points = points.loc[(points > 0).sum(1) > 1, :]
                        points.columns = [
                            cur_x + " (" + cur_l_smoothed + ")",
                            cur_y + " (" + cur_l_smoothed + ")",
                        ]
                        cur_title = cur_x + ' VS ' + cur_y
                    elif is_cell_anno_column(adata, cur_x) and is_cell_anno_column(adata, cur_y):
                        points = pd.DataFrame(
                            {
                                cur_x: adata.obs_vector(cur_x),
                                cur_y: adata.obs_vector(cur_y),
                            }
                        )
                        points.columns = [cur_x, cur_y]
                        cur_title = cur_x + ' VS ' + cur_y
                    elif is_cell_anno_column(adata, cur_x) and is_gene_name(adata, cur_y):
                        points = pd.DataFrame(
                            {
                                cur_x: adata.obs_vector(cur_x), 
                                cur_y: adata.obs_vector(k=cur_y, layer=None) if cur_l_smoothed == 'X' else adata.obs_vector(k=cur_y, layer=cur_l_smoothed),
                            }
                        )
                        # points = points.loc[points.iloc[:, 1] > 0, :]
                        points.columns = [cur_x, cur_y + " (" + cur_l_smoothed + ")"]
                        cur_title = cur_y
                    elif is_gene_name(adata, cur_x) and is_cell_anno_column(adata, cur_y):
                        points = pd.DataFrame(
                            {
                                cur_x: adata.obs_vector(k=cur_x, layer=None) if cur_l_smoothed == 'X' else adata.obs_vector(k=cur_x, layer=cur_l_smoothed), 
                                cur_y: adata.obs_vector(cur_y)
                            }
                        )
                        # points = points.loc[points.iloc[:, 0] > 0, :]
                        points.columns = [cur_x + " (" + cur_l_smoothed + ")", cur_y]
                        cur_title = cur_x
                    elif is_layer_keys(adata, cur_x) and is_layer_keys(adata, cur_y):
                        add_gamma_fit = True
                        cur_x_, cur_y_ = adata[:, cur_b].layers[cur_x], adata[:, cur_b].layers[cur_y]
                        points = pd.DataFrame(
                            {cur_x: flatten(cur_x_),
                             cur_y: flatten(cur_y_)}
                        )
                        # points = points.loc[points.iloc[:, 0] > 0, :]
                        points.columns = [cur_x, cur_y]
                        cur_title = cur_b
                    if aggregate is not None:
                        groups, uniq_grp = (
                            adata.obs[aggregate],
                            adata.obs[aggregate].unique().to_list(),
                        )
                        group_color, group_median = (
                            np.zeros((1, len(uniq_grp))).flatten()
                            if isinstance(_color[0], Number)
                            else np.zeros((1, len(uniq_grp))).astype("str").flatten(),
                            np.zeros((len(uniq_grp), 2)),
                        )

                        grp_size = adata.obs[aggregate].value_counts().values
                        scatter_kwargs = (
                            {"s": grp_size}
                            if scatter_kwargs is None
                            else update_dict(scatter_kwargs, {"s": grp_size})
                        )

                        for ind, cur_grp in enumerate(uniq_grp):
                            group_median[ind, :] = np.nanmedian(
                                points.iloc[np.where(groups == cur_grp)[0], :2], 0
                            )
                            if isinstance(_color[0], Number):
                                group_color[ind] = np.nanmedian(
                                    np.array(_color)[np.where(groups == cur_grp)[0]]
                                )
                            else:
                                group_color[ind] = (
                                    pd.Series(_color)[np.where(groups == cur_grp)[0]]
                                    .value_counts()
                                    .index[0]
                                )

                        points, _color = (
                            pd.DataFrame(
                                group_median, index=uniq_grp, columns=points.columns
                            ),
                            group_color,
                        )
                    # https://stackoverflow.com/questions/4187185/how-can-i-check-if-my-python-object-is-a-number
                    # answer from Boris.
                    is_not_continous = (
                        not isinstance(_color[0], Number) or _color.dtype.name == 'category'
                    )

                    if is_not_continous:
                        labels = _color.to_dense() if is_categorical(_color) else _color
                        if theme is None:
                            if _background in ["#ffffff", "black"]:
                                _theme_ = "glasbey_dark"
                            else:
                                _theme_ = "glasbey_white"
                        else:
                            _theme_ = theme
                    else:
                        values = _color
                        if theme is None:
                            if _background in ["#ffffff", "black"]:
                                _theme_ = (
                                    "inferno"
                                    if cur_l != "velocity"
                                    else "div_blue_black_red"
                                )
                            else:
                                _theme_ = (
                                    "viridis" if not cur_l.startswith("velocity") else "div_blue_red"
                                )
                        else:
                            _theme_ = theme

                    _cmap = _themes[_theme_]["cmap"] if cmap is None else cmap
                    _color_key_cmap = (
                        _themes[_theme_]["color_key_cmap"]
                        if color_key_cmap is None
                        else color_key_cmap
                    )
                    _background = (
                        _themes[_theme_]["background"] if _background is None else _background
                    )

                    if labels is not None and values is not None:
                        raise ValueError(
                            "Conflicting options; only one of labels or values should be set"
                        )

                    if total_panels > 1:
                        ax = plt.subplot(gs[i])
                    i += 1

                    # if highligts is a list of lists - each list is relate to each color element
                    if highlights is not None:
                        if is_list_of_lists(highlights):
                            _highlights = highlights[color.index(cur_c)]
                            _highlights = (
                                _highlights
                                if all([i in _color for i in _highlights])
                                else None
                            )
                        else:
                            _highlights = (
                                highlights
                                if all([i in _color for i in highlights])
                                else None
                            )

                    color_out = None

                    if smooth and not is_not_continous:
                        knn = adata.obsp['moments_con']
                        values = calc_1nd_moment(values, knn)[0] if smooth in [1, True] else \
                            calc_1nd_moment(values, knn**smooth)[0]

                    if points.shape[0] <= figsize[0] * figsize[1] * 100000:
                        ax, color_out = _matplotlib_points(
                            points.values,
                            ax,
                            labels,
                            values,
                            highlights,
                            _cmap,
                            color_key,
                            _color_key_cmap,
                            _background,
                            figsize[0],
                            figsize[1],
                            show_legend,
                            sort=sort,
                            frontier=frontier,
                            contour=contour,
                            ccmap=ccmap,
                            calpha=calpha,
                            sym_c=sym_c,
                            **scatter_kwargs
                        )
                    else:
                        ax = _datashade_points(
                            points.values,
                            ax,
                            labels,
                            values,
                            highlights,
                            _cmap,
                            color_key,
                            _color_key_cmap,
                            _background,
                            figsize[0],
                            figsize[1],
                            show_legend,
                            sort=sort,
                            frontier=frontier,
                            contour=contour,
                            ccmap=ccmap,
                            calpha=calpha,
                            sym_c=sym_c,
                            **scatter_kwargs
                        )

                    if i == 1 and show_arrowed_spines:
                        arrowed_spines(ax, points.columns[:2], _background)
                    else:
                        despline_all(ax)
                        deaxis_all(ax)

                    ax.set_title(cur_title)

                    axes_list.append(ax)
                    color_list.append(color_out)

                    labels, values = None, None  # reset labels and values

                    if add_gamma_fit and cur_b in adata.var_names[adata.var.use_for_dynamics]:
                        xnew = np.linspace(0, points.iloc[:, 0].max() * 0.80)
                        k_name = 'gamma_k' if adata.uns['dynamics']['experiment_type'] == 'one-shot' else 'gamma'
                        if k_name in adata.var.columns:
                            if (
                                    not ("gamma_b" in adata.var.columns)
                                    or all(adata.var.gamma_b.isna())
                            ):
                                adata.var.loc[:, "gamma_b"] = 0
                            ax.plot(
                                xnew,
                                xnew * adata[:, cur_b].var.loc[:, k_name].unique()
                                + adata[:, cur_b].var.loc[:, "gamma_b"].unique(),
                                dashes=[6, 2],
                                c=font_color,
                            )
                        else:
                            raise Exception(
                                "adata does not seem to have velocity_gamma column. Velocity estimation is required before "
                                "running this function."
                            )
    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'scatters', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
        if background is not None: reset_rcParams()
    elif save_show_or_return == "show":
        if show_legend:
            plt.subplots_adjust(right=0.85)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()
        
        plt.show()
        if background is not None: reset_rcParams()
    elif save_show_or_return == "return":
        if background is not None: reset_rcParams()

        if return_all:
            return (axes_list, color_list, font_color) if total_panels > 1 else (ax, color_out, font_color)
        else:
            return axes_list if total_panels > 1 else ax
