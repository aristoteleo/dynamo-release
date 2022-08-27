import warnings

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from ..configuration import _themes, reset_rcParams, set_figure_params
from ..tools.utils import get_mapper, update_dict
from .utils import save_fig


def bubble(
    adata,
    genes,
    group,
    gene_order=None,
    group_order=None,
    layer=None,
    theme=None,
    cmap=None,
    color_key=None,
    color_key_cmap="Spectral",
    background="white",
    pointsize=None,
    vmin=0,
    vmax=100,
    sym_c=False,
    alpha=0.8,
    edgecolor=None,
    linewidth=0,
    type="violin",
    sort="diagnoal",
    transpose=False,
    rotate_xlabel="horizontal",
    rotate_ylabel="horizontal",
    figsize=None,
    save_show_or_return="show",
    save_kwargs={},
    **kwargs,
):
    """Bubble plots generalized to velocity, acceleration, curvature.
    It supports either the `dot` or `violin` plot mode. This function is loosely based on
    https://github.com/QuKunLab/COVID-19/blob/master/step3_plot_umap_and_marker_gene_expression.ipynb

    # add sorting
    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        genes: `list`
            The gene list, i.e. marker gene or top acceleration, curvature genes, etc.
        group: `str`
            The column key in `adata.obs` that will be used to group cells.
        gene_order: `None` or `list` (default: `None`)
            The gene groups order that will show up in the resulting bubble plot.
        group_order: `None` or `list` (default: `None`)
            The cells groups order that will show up in the resulting bubble plot.
        layer: `None` or `str` (default: `None`)
            The layer of data to use for the bubble plot.
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
        pointsize: `None` or `float` (default: None)
            The scale of the point size. Actual point cell size is calculated as `500.0 / np.sqrt(adata.shape[0]) *
            pointsize`
        vmin: `float` (default: `0`)
            The percentage of minimal value to consider.
        vmax: `float` (default: `100`)
            The percentage of maximal value to consider.
        sym_c: `bool` (default: `False`)
            Whether do you want to make the limits of continuous color to be symmetric, normally this should be used for
            plotting velocity, jacobian, curl, divergence or other types of data with both positive or negative values.
        alpha: `float` (default: `0.8`)
            alpha value of the plot
        edgecolor: `str` or `None` (default: `None`)
            The color of the edge of the dots when type is to be `dot`.
        linewidth: `str` or `None` (default: `None`)
            The width of the edge of the dots when type is to be `dot`.
        type: `str` (default: `violin`)
            The type of the bubble plot, one of `{'violin', 'dot'}`.
        figsize: `None` or `[float, float]` (default: None)
            The width and height of a figure.
        sort: `str` (default: `diagnol`)
            The method for sorting genes. Not implemented. Need to implement in 2021.
        transpose: `bool` (default: `False`)
            Whether to transpose the row/column of the resulting bubble plot. Gene and cell types are on x/y-axis by
            default.
        rotate_xlabel: `float` (default: `horizontal`)
            The angel to rotate the x-label.
        rotate_ylabel: `float` (default: `horizontal`)
            The angel to rotate the y-label.
        save_show_or_return: `str` {'save', 'show', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
            will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.
        kwargs:
            Additional arguments passed to plt.scatters or sns.violinplot.


    Returns
    -------
        Nothing but plot the bubble plots.

    """
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
        # if save_show_or_return != 'save': set_figure_params('dynamo', background=_background)
    else:
        _background = background
        # if save_show_or_return != 'save': set_figure_params('dynamo', background=_background)

    if theme is None:
        if _background in ["#ffffff", "black"]:
            _theme_ = "glasbey_dark"
        else:
            _theme_ = "glasbey_white"
    else:
        _theme_ = theme
    _cmap = _themes[_theme_]["cmap"] if cmap is None else cmap

    if layer is None:
        mapper = get_mapper()

        has_splicing, has_labeling, splicing_labeling, has_protein = (
            adata.uns["pp"]["has_splicing"],
            adata.uns["pp"]["has_labeling"],
            adata.uns["pp"]["splicing_labeling"],
            adata.uns["pp"]["has_protein"],
        )

        if splicing_labeling:
            layer = mapper["X_total"] if mapper["X_total"] in adata.layers else "X_total"
        elif has_labeling:
            layer = mapper["X_total"] if mapper["X_total"] in adata.layers else "X_total"
        else:
            layer = mapper["X_spliced"] if mapper["X_spliced"] in adata.layers else "X_spliced"

    if group not in adata.obs_keys():
        raise ValueError(f"argument group {group} is not a column name in `adata.obs`")

    genes = adata.var_names.intersection(set(genes)).to_list()
    if len(genes) == 0:
        raise ValueError(f"names from argument genes {genes} don't match any genes from `adata.var_names`.")

    # sort gene/cluster to update the orders
    uniq_groups = adata.obs[group].unique()
    if group_order is None:
        clusters = uniq_groups
    else:
        if not set(group_order).issubset(uniq_groups):
            raise ValueError(
                f"names from argument group_order {group_order} are not subsets of " f"`adata.obs[group].unique()`."
            )
        clusters = group_order

    if gene_order is None:
        genes = genes
    else:
        if not set(gene_order).issubset(genes):
            raise ValueError(
                f"names from argument gene_order {gene_order} is not a subset of "
                f"`adata.var_names.intersection(set(genes)).to_list()`."
            )
        genes = gene_order

    cells_df = adata.obs.get(group)
    gene_df = adata[:, genes].layers[layer]
    gene_df = gene_df.A if issparse(gene_df) else gene_df
    gene_df = pd.DataFrame(gene_df.T, index=genes, columns=adata.obs_names)

    xmin, xmax = gene_df.quantile(vmin / 100, axis=1), gene_df.quantile(vmax / 100, axis=1)
    if sym_c:
        _vmin, _vmax = np.zeros_like(xmin), np.zeros_like(xmax)
        i = 0
        for a, b in zip(xmin, xmax):
            bounds = np.nanmax([np.abs(a), b])
            bounds = bounds * np.array([-1, 1])
            _vmin[i], _vmax[i] = bounds
            i += 1
        xmin, xmax = _vmin, _vmax

    point_size = (
        16000.0 / np.sqrt(adata.shape[0]) if pointsize is None else 16000.0 / (len(genes) * len(clusters)) * pointsize
    )

    if color_key is None:
        cmap_ = matplotlib.cm.get_cmap(color_key_cmap)
        cmap_.set_bad("lightgray")
        unique_labels = np.unique(clusters)
        num_labels = unique_labels.shape[0]
        color_key = plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))

    if figsize is None:
        width = 6 * len(genes) / 14 if transpose else 9 * len(genes) / 14
        height = 4.5 * len(clusters) / 14 if transpose else 4.5 * len(genes) / 14
        figsize = (height, width) if transpose else (width, height)
    else:
        figsize = figsize[::-1] if transpose else figsize

    # scatter_kwargs = dict(
    #     alpha=0.8, s=point_size, edgecolor=None, linewidth=0, rasterized=False
    # )  # (0, 0, 0, 1)

    fig, axes = plt.subplots(
        len(genes) if transpose else 1,
        1 if transpose else len(genes),
        figsize=figsize,
        facecolor=background,
    )
    fig.subplots_adjust(hspace=0, wspace=0)
    clusters_vec = cells_df.loc[gene_df.columns.values].values

    # may also use clusters when transpose
    for igene, gene in enumerate(genes):
        cur_gene_df = pd.DataFrame({gene: gene_df.loc[gene, :].values, "clusters_": clusters_vec})
        cur_gene_df = cur_gene_df.loc[cur_gene_df["clusters_"].isin(clusters)]

        if type == "violin":
            # use sort here
            sns.violinplot(
                data=cur_gene_df,
                x="clusters_" if transpose else gene,
                y=gene if transpose else "clusters_",
                orient="v" if transpose else "h",
                order=clusters,  # genes if transpose else
                linewidth=None,
                palette=color_key,
                inner="box",
                scale="width",
                cut=0,
                ax=axes[igene],
                alpha=alpha,
                **kwargs,
            )
            if transpose:
                axes[igene].set_ylim(xmin[igene], xmax[igene])
                axes[igene].set_yticks([])
                axes[igene].set_ylabel(gene, rotation=rotate_ylabel, ha="right", va="center")
            else:
                axes[igene].set_xlim(xmin[igene], xmax[igene])
                axes[igene].set_xticks([])
                axes[igene].set_xlabel(gene, rotation=rotate_xlabel, ha="right")

        elif type == "dot":
            # use sort here
            avg_perc_cluster = (
                cur_gene_df.groupby("clusters_")
                .expression.apply(lambda x: pd.Series([x.mean(), (x != 0).sum() / len(x)]))
                .unstack()
            )
            avg_perc_cluster.columns = ["avg", "perc"]

            axes[igene].scatter(
                x=clusters if transpose else gene,
                y=gene if transpose else clusters,
                s=avg_perc_cluster.loc[clusters, "perc"] * point_size,
                lw=2,
                c=avg_perc_cluster.loc[clusters, "avg"],
                cmap="viridis" if cmap is None else cmap,
                rasterized=False,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
            )
        if transpose:
            if igene != len(genes) - 1:
                axes[igene].set_xticks([])
            else:
                axes[igene].set_xticklabels(
                    list(map(str, np.array(clusters))),
                    rotation=rotate_xlabel,
                    ha="right",
                )
        else:
            if igene != 0:
                axes[igene].set_yticks([])
            else:
                axes[igene].set_yticklabels(
                    list(map(str, np.array(clusters))),
                    rotation=rotate_ylabel,
                    ha="right",
                    va="center",
                )
        axes[igene].set_xlabel("") if transpose else axes[igene].set_ylabel("")

    if save_show_or_return == "save":
        s_kwargs = {
            "path": None,
            "prefix": "violin",
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
        if background is not None:
            reset_rcParams()
    elif save_show_or_return == "show":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()

        plt.show()
        if background is not None:
            reset_rcParams()
    elif save_show_or_return == "return":
        if background is not None:
            reset_rcParams()

        return fig, axes
