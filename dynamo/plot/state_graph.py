from typing import Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes

from ..tools.utils import update_dict
from .scatters import docstrings, scatters
from .utils import save_fig

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
    state_graph: Union[None, np.ndarray] = None,
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
    theme: Optional[str] = None,
    cmap: Optional[str] = None,
    color_key: Union[dict, list] = None,
    color_key_cmap: Optional[str] = None,
    background: Optional[str] = None,
    ncols: int = 4,
    pointsize: Union[None, float] = None,
    figsize: tuple = (6, 4),
    show_legend: bool = True,
    use_smoothed: bool = True,
    show_arrowed_spines: bool = False,
    ax: Optional[Axes] = None,
    sort: str = "raw",
    frontier: bool = False,
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
    s_kwargs_dict: dict = {"alpha": 1},
    **kwargs
):
    """Plot a summarized cell type (state) transition graph. This function tries to create a model that summarizes
    the possible cell type transitions based on the reconstructed vector field function.

    Parameters
    ----------
        group: `str` or `None` (default: `None`)
            The column in adata.obs that will be used to aggregate data points for the purpose of creating a cell type
            transition model.
        transition_threshold: `float` (default: 0.001)
            The threshold of cell fate transition. Transition will be ignored if below this threshold.
        keep_only_one_direction: `bool` (default: True)
            Whether to only keep the higher transition between two cell type. That is if the transition rate from A to B
            is higher than B to A, only edge from A to B will be plotted.
        edge_scale: `float` (default: 1)
            The scaler that can be used to scale the edge width of drawn transition graph.
        state_graph: `np.ndarray`, `pd.DataFrame` or `None` (default: None)
            The lumped transition graph between cell states (e.g. cell clusters or types).
        edgecolor: `np.ndarray`, `pd.DataFrame` or `None` (default: None)
            The edge color of the arcs that corresponds to the lumped transition graph between cell states.
        facecolor: `np.ndarray`, `pd.DataFrame` or `None` (default: None)
            The edge color of the arcs that corresponds to the lumped transition graph between cell states.
        graph_alpha: `np.ndarray`, `pd.DataFrame` or `None` (default: None)
            The alpha of the arcs that corresponds to the lumped transition graph between cell states.
        %(scatters.parameters.no_aggregate|kwargs|save_kwargs)s
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
            save_fig function will use the {"path": None, "prefix": 'state_graph', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a
            dictionary that properly modify those keys according to your needs.
        s_kwargs_dict: `dict` (default: {"alpha": 1})
            The dictionary of the scatter arguments.
    Returns
    -------
        Plot the a model of cell fate transition that summarizes the possible lineage commitments between different cell
        types.
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

    if save_show_or_return == "save":
        s_kwargs = {
            "path": None,
            "prefix": "state_graph",
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        if show_legend:
            plt.subplots_adjust(right=0.85)
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return axes_list, color_list, font_color
