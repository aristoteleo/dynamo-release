import numpy as np

from ..tools.utils import update_dict
from .scatters import scatters, docstrings
from .utils import save_fig

docstrings.delete_params("scatters.parameters", "aggregate", "kwargs", "save_kwargs")


def create_edge_patch(
    posA,
    posB,
    width=1,
    node_rad=0,
    connectionstyle="arc3, rad=0.25",
    facecolor="k",
    **kwargs
):
    import matplotlib.patches as pat

    style = "simple,head_length=%d,head_width=%d,tail_width=%d" % (10, 10, 3 * width)
    return pat.FancyArrowPatch(
        posA=posA,
        posB=posB,
        arrowstyle=style,
        connectionstyle=connectionstyle,
        facecolor=facecolor,
        shrinkA=node_rad,
        shrinkB=node_rad,
        **kwargs
    )


def create_edge_patches_from_markov_chain(
    P,
    X,
    width=3,
    node_rad=0,
    tol=1e-7,
    connectionstyle="arc3, rad=0.25",
    facecolor="k",
    alpha=0.8,
    **kwargs
):
    arrows = []
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            if P[i, j] > tol:
                arrows.append(
                    create_edge_patch(
                        X[i],
                        X[j],
                        width=P[i, j] * width,
                        node_rad=node_rad,
                        connectionstyle=connectionstyle,
                        facecolor=facecolor,
                        alpha=alpha * min(2 * P[i, j], 1),
                        **kwargs
                    )
                )
    return arrows


@docstrings.with_indent(4)
def state_graph(
    adata,
    group,
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
    ncols=1,
    pointsize=None,
    figsize=(6, 4),
    show_legend=True,
    use_smoothed=True,
    show_arrowed_spines=True,
    ax=None,
    sort='raw',
    frontier=False,
    save_show_or_return="show",
    save_kwargs={},
    s_kwargs_dict={},
    **kwargs
):
    """Plot a summarized cell type (state) transition graph. This function tries to create a model that summarizes
    the possible cell type transitions based on the reconstructed vector field function.

    Parameters
    ----------
        group: `str` or `None` (default: `None`)
            The column in adata.obs that will be used to aggregate data points for the purpose of creating a cell type
            transition model.
        %(scatters.parameters.no_aggregate|kwargs|save_kwargs)s
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
            will use the {"path": None, "prefix": 'state_graph', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.
        s_kwargs_dict: `dict` (default: {})
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
    groups, uniq_grp = adata.obs[group], adata.obs[group].unique().to_list()
    group_median = np.zeros((len(uniq_grp), 2))
    grp_size = adata.obs[group].value_counts().values
    s_kwargs_dict.update({"s": grp_size})

    Pl = adata.uns["Cell type annotation_graph"]["group_graph"]
    Pl[Pl - Pl.T < 0] = 0
    Pl /= Pl.sum(1)[:, None]

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
        save_show_or_return='return',
        frontier=frontier,
        **s_kwargs_dict,
        return_all=True,
    )

    arrows = create_edge_patches_from_markov_chain(
        Pl, group_median, tol=0.01, node_rad=15
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

    plt.show()

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'state_graph', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        if show_legend:
            plt.subplots_adjust(right=0.85)
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return axes_list, color_list, font_color
