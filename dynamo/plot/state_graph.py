from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes

from .scatters import docstrings, scatters
from .utils import save_show_ret

docstrings.delete_params("scatters.parameters", "aggregate", "kwargs", "save_kwargs")


def create_edge_patch(posA, posB, width=1, node_rad=0, connectionstyle="arc3, rad=0.25", facecolor="k", **kwargs):
    import matplotlib.patches as pat

    style = "simple,head_length=%d,head_width=%d,tail_width=%d" % (
        10,
        10,
        3 * width,
    )
    return pat.FancyArrowPatch(
        posA=posA,
        posB=posB,
        arrowstyle=style,
        connectionstyle=connectionstyle,
        facecolor=facecolor,
        shrinkA=node_rad,
        shrinkB=node_rad,
        **kwargs,
    )


def create_edge_patches_from_markov_chain(
    P,
    X,
    width=3,
    node_rad=0,
    tol=1e-7,
    connectionstyle="arc3, rad=0.25",
    facecolor="k",
    edgecolor="k",
    alpha=0.8,
    **kwargs
):
    """
    create edge patches from a markov chain transition matrix. If P[i, j] > tol, an arrow is created from
    node i to j.
    """
    arrows = []
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            if P[i, j] > tol:
                if type(facecolor) == str:
                    fc = facecolor
                else:
                    if type(facecolor) == pd.DataFrame:
                        fc = facecolor.iloc[i, j]
                    else:
                        fc = facecolor[i, j]

                if type(edgecolor) == str:
                    ec = edgecolor
                else:
                    if type(edgecolor) == pd.DataFrame:
                        ec = edgecolor.iloc[i, j]
                    else:
                        ec = edgecolor[i, j]

                if type(alpha) == float:
                    ac = alpha * min(2 * P[i, j], 1)
                else:
                    if type(alpha) == pd.DataFrame:
                        ac = alpha.iloc[i, j]
                    else:
                        ac = alpha[i, j]

                arrows.append(
                    create_edge_patch(
                        X[i],
                        X[j],
                        width=P[i, j] * width,
                        node_rad=node_rad,
                        connectionstyle=connectionstyle,
                        facecolor=fc,
                        edgecolor=ec,
                        alpha=ac,
                        **kwargs,
                    )
                )
    return arrows


@docstrings.with_indent(4)
def state_graph(
    adata: AnnData,
    group: Optional[str] = None,
    transition_threshold: float = 0.001,
    keep_only_one_direction: bool = True,
    edge_scale: float = 1,
    state_graph: Optional[np.ndarray] = None,
    edgecolor: Union[None, np.ndarray, pd.DataFrame] = None,
    facecolor: Union[None, np.ndarray, pd.DataFrame] = None,
    graph_alpha: Union[None, np.ndarray, pd.DataFrame] = None,
    basis: str = "umap",
    x: int = 0,
    y: int = 1,
    color: str = "ntr",
    layer: str = "X",
    highlights: Optional[list] = None,
    labels: Optional[list] = None,
    values: Optional[list] = None,
    theme: Optional[
        Literal[
            "blue",
            "red",
            "green",
            "inferno",
            "fire",
            "viridis",
            "darkblue",
            "darkred",
            "darkgreen",
        ]
    ] = None,
    cmap: Optional[str] = None,
    color_key: Union[Dict[str, str], List[str], None] = None,
    color_key_cmap: Optional[str] = None,
    background: Optional[str] = None,
    ncols: int = 4,
    pointsize: Optional[float] = None,
    figsize: Tuple[float, float] = (6, 4),
    show_legend: bool = True,
    use_smoothed: bool = True,
    show_arrowed_spines: bool = False,
    ax: Optional[Axes] = None,
    sort: Literal["raw", "abs", "neg"] = "raw",
    frontier: bool = False,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
    s_kwargs_dict: Dict[str, Any] = {"alpha": 1},
    **kwargs
) -> Union[
    Tuple[Axes, List[str], Literal["white", "black"]],
    Tuple[List[Axes], List[str], Literal["white", "black"]],
    None,
]:
    """Plot a summarized cell type (state) transition graph. This function tries to create a model that summarizes the
    possible cell type transitions based on the reconstructed vector field function.

    Args:
        adata: an AnnData object.
        group: the column in adata.obs that will be used to aggregate data points for the purpose of creating a cell
            type transition model. Defaults to None.
        transition_threshold: the threshold of cell fate transition. Transition will be ignored if below this threshold.
            Defaults to 0.001.
        keep_only_one_direction: whether to only keep the higher transition between two cell type. That is if the
            transition rate from A to B is higher than B to A, only edge from A to B will be plotted. Defaults to True.
        edge_scale: the scaler that can be used to scale the edge width of drawn transition graph. Defaults to 1.
        state_graph: the lumped transition graph between cell states (e.g. cell clusters or types). Defaults to None.
        edgecolor: the edge color of the arcs that corresponds to the lumped transition graph between cell states.
            Defaults to None.
        facecolor: the edge color of the arcs that corresponds to the lumped transition graph between cell states.
            Defaults to None.
        graph_alpha: the alpha of the arcs that corresponds to the lumped transition graph between cell states. Defaults
            to None.
        basis: the reduced dimension stored in adata.obsm. The specific basis key will be constructed in the following
            priority if exits: 1) specific layer input +  basis 2) X_ + basis 3) basis. E.g. if basis is PCA, `scatters`
            is going to look for 1) if specific layer is spliced, `spliced_pca` 2) `X_pca` (dynamo convention) 3) `pca`.
            Defaults to "umap".
        x: the column index of the low dimensional embedding for the x-axis. Defaults to 0.
        y: the column index of the low dimensional embedding for the y-axis. Defaults to 1.
        color: any column names or gene expression, etc. that will be used for coloring cells. Defaults to "ntr".
        layer: the layer of data to use for the scatter plot. Defaults to "X".
        highlights: the color group that will be highlighted. If highligts is a list of lists, each list is relate to
            each color element. Defaults to None.
        labels: an array of labels (assumed integer or categorical), one for each data sample. This will be used for
            coloring the points in the plot according to their label. Note that this option is mutually exclusive to the
            `values` option. Defaults to None.
        values: an array of values (assumed float or continuous), one for each sample. This will be used for coloring
            the points in the plot according to a colorscale associated to the total range of values. Note that this
            option is mutually exclusive to the `labels` option. Defaults to None.
        theme: A color theme to use for plotting. A small set of predefined themes are provided which have relatively
            good aesthetics. Available themes are: {'blue', 'red', 'green', 'inferno', 'fire', 'viridis', 'darkblue',
            'darkred', 'darkgreen'}. Defaults to None.
        cmap: The name of a matplotlib colormap to use for coloring or shading points. If no labels or values are passed
            this will be used for shading points according to density (largely only of relevance for very large
            datasets). If values are passed this will be used for shading according the value. Note that if theme is
            passed then this value will be overridden by the corresponding option of the theme. Defaults to None.
        color_key: the method to assign colors to categoricals. This can either be an explicit dict mapping labels to
            colors (as strings of form '#RRGGBB'), or an array like object providing one color for each distinct
            category being provided in `labels`. Either way this mapping will be used to color points according to the
            label. Note that if theme is passed then this value will be overridden by the corresponding option of the
            theme. Defaults to None.
        color_key_cmap: the name of a matplotlib colormap to use for categorical coloring. If an explicit `color_key` is
            not given a color mapping for categories can be generated from the label list and selecting a matching list
            of colors from the given colormap. Note that if theme is passed then this value will be overridden by the
            corresponding option of the theme. Defaults to None.
        background: the color of the background. Usually this will be either 'white' or 'black', but any color name will
            work. Ideally one wants to match this appropriately to the colors being used for points etc. This is one of
            the things that themes handle for you. Note that if theme is passed then this value will be overridden by
            the corresponding option of the theme. Defaults to None.
        ncols: the number of columns for the figure. Defaults to 4.
        pointsize: the scale of the point size. Actual point cell size is calculated as
            `500.0 / np.sqrt(adata.shape[0]) * pointsize`. Defaults to None.
        figsize: the width and height of a figure. Defaults to (6, 4).
        show_legend: whether to display a legend of the labels. Defaults to "on data".
        use_smoothed: whether to use smoothed values (i.e. M_s / M_u instead of spliced / unspliced, etc.). Defaults to
            True.
        show_arrowed_spines: whether to show a pair of arrowed spines representing the basis of the scatter is currently
            using. Defaults to False.
        ax: the matplotlib axes object where new plots will be added to. Only applicable to drawing a single component.
            Defaults to None.
        sort: the method to reorder data so that high values points will be on top of background points. Can be one of
            {'raw', 'abs', 'neg'}, i.e. sorted by raw data, sort by absolute values or sort by negative values. Defaults
            to "raw".
        frontier: whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to
            show area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib
            tips & tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from
            scEU-seq paper: https://science.sciencemag.org/content/367/6482/1151. If `contour` is set  to be True,
            `frontier` will be ignored as `contour` also add an outlier for data points. Defaults to False.
        save_show_or_return: whether to save, show, or return the generated figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'state_graph', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.
        s_kwargs_dict: any other kwargs that would be passed to `dynamo.pl.scatters`. Defaults to {"alpha": 1}.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be `return`, the matplotlib axes
        object of the generated plots, the list of colors used and the font color would be returned.
    """

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    aggregate = group

    points = adata.obsm["X_" + basis][:, [x, y]]
    unique_group_obs = adata.obs[group].unique()
    if type(unique_group_obs) is np.ndarray:
        groups, uniq_grp = adata.obs[group], unique_group_obs.tolist()
    elif type(unique_group_obs) is pd.Series:
        groups, uniq_grp = adata.obs[group], unique_group_obs.to_list()
    else:
        groups, uniq_grp = adata.obs[group], list(unique_group_obs)
    group_median = np.zeros((len(uniq_grp), 2))
    # grp_size = adata.obs[group].value_counts()[uniq_grp].values
    # s_kwargs_dict.update({"s": grp_size})

    if state_graph is None:
        Pl = adata.uns[group + "_graph"]["group_graph"]
        if keep_only_one_direction:
            Pl[Pl - Pl.T < 0] = 0
        if transition_threshold is not None:
            Pl[Pl < transition_threshold] = 0

        Pl /= Pl.sum(1)[:, None] * edge_scale
    else:
        Pl = state_graph

    for i, cur_grp in enumerate(uniq_grp):
        group_median[i, :] = np.nanmedian(points[np.where(groups == cur_grp)[0], :2], 0)

    if background is None:
        _background = rcParams.get("figure.facecolor")
        background = to_hex(_background) if type(_background) is tuple else _background

    plt.figure(facecolor=_background)
    axes_list, color_list, font_color = scatters(
        adata=adata,
        basis=basis,
        x=x,
        y=y,
        color=color,
        layer=layer,
        highlights=highlights,
        labels=labels,
        values=values,
        theme=theme,
        cmap=cmap,
        color_key=color_key,
        color_key_cmap=color_key_cmap,
        background=background,
        ncols=ncols,
        pointsize=pointsize,
        figsize=figsize,
        show_legend=show_legend,
        use_smoothed=use_smoothed,
        aggregate=aggregate,
        show_arrowed_spines=show_arrowed_spines,
        ax=ax,
        sort=sort,
        save_show_or_return="return",
        frontier=frontier,
        **s_kwargs_dict,
        return_all=True,
    )

    edgecolor = "k" if edgecolor is None else edgecolor
    facecolor = "k" if facecolor is None else facecolor
    graph_alpha = 0.8 if graph_alpha is None else graph_alpha

    arrows = create_edge_patches_from_markov_chain(
        Pl, group_median, edgecolor=edgecolor, facecolor=facecolor, alpha=graph_alpha, tol=0.01, node_rad=15
    )
    if type(axes_list) == list:
        for i in range(len(axes_list)):
            for arrow in arrows:
                axes_list[i].add_patch(arrow)
                axes_list[i].set_facecolor(background)
    else:
        for arrow in arrows:
            axes_list.add_patch(arrow)
            axes_list.set_facecolor(background)

    plt.axis("off")

    return save_show_ret("state_graph", save_show_or_return, save_kwargs, (axes_list, color_list, font_color), adjust = show_legend)
