from .scatters import scatters, docstrings
import numpy as np

def create_edge_patch(posA, posB, width=1, node_rad=0, connectionstyle='arc3, rad=0.25', facecolor='k', **kwargs):
    import matplotlib.patches as pat

    style = "simple,head_length=%d,head_width=%d,tail_width=%d" % (10, 10, 3 * width)
    return pat.FancyArrowPatch(posA=posA, posB=posB, arrowstyle=style, connectionstyle=connectionstyle,
                               facecolor=facecolor, shrinkA=node_rad, shrinkB=node_rad, **kwargs)


def create_edge_patches_from_markov_chain(P, X, width=3, node_rad=0, tol=1e-7, connectionstyle='arc3, rad=0.25',
                                          facecolor='k', alpha=0.8, **kwargs):
    arrows = []
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            if P[i, j] > tol:
                arrows.append(create_edge_patch(X[i], X[j], width=P[i, j] * width, node_rad=node_rad,
                                                connectionstyle=connectionstyle, facecolor=facecolor,
                                                alpha=alpha * min(2 * P[i, j], 1), **kwargs))
    return arrows


@docstrings.with_indent(4)
def state_graph(adata,
                group,
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
                save_or_show='return',
                s_kwargs_dict={},
                **kwargs):
    """Plot a summarized cell type (state) transition graph. This function tries to create a model that summarize
    the possible cell type transitions based on the reconstructed vector function.

    Parameters
    ----------
        group: `str` or `None` (default: `None`)
            The column in adata.obs that will be used to aggregate data points for the purpose of creating a cell type
            transition model.
        %(scatters.parameters.no_aggregate)s
        s_kwargs_dict: `dict` (default: {})
            The dictionary of the scatter arguments.
    Returns
    -------
        Plot the streamline, fixed points (attractors / saddles), nullcline, separatrices of a recovered dynamic system
        for single cells or return the corresponding axis, depending on the plot argument.
    """

    import matplotlib.pyplot as plt

    aggregate = group

    points = adata.obsm['X_' + basis][:, [x, y]]
    groups, uniq_grp = adata.obs[group], adata.obs[group].unique().to_list()
    group_median = np.zeros((len(uniq_grp), 2))
    grp_size = adata.obs[group].value_counts().values
    s_kwargs_dict.update({'s': grp_size})

    Pl = adata.uns['Cell type annotation_graph']['group_graph']
    Pl[Pl - Pl.T < 0] = 0
    Pl /= Pl.sum(1)[:, None]

    for i, cur_grp in enumerate(uniq_grp):
        group_median[i, :] = np.nanmedian(points[np.where(groups == cur_grp)[0], :2], 0)

    axes_list, color_list, font_color = scatters(
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
        save_or_show,
        aggregate,
        **s_kwargs_dict)


    for i in range(len(axes_list)):
        arrows = create_edge_patches_from_markov_chain(Pl, group_median, tol=0.01, node_rad=15)
        for arrow in arrows:
            axes_list[i].add_patch(arrow)

    plt.axis('off')

    plt.show()
