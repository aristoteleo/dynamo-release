from .utils import _matplotlib_points
import numpy as np

def create_edge_patch(posA, posB, width=1, node_rad=0, connectionstyle='arc3, rad=0.25', facecolor='k', **kwargs):
    import matplotlib.patches as pat

    style = "simple,head_length=%d,head_width=%d,tail_width=%d" % (10, 10, 3 * width)
    return pat.FancyArrowPatch(posA=posA, posB=posB, arrowstyle=style, connectionstyle=connectionstyle,
                               facecolor=facecolor, shrinkA=node_rad, shrinkB=node_rad, **kwargs)


def create_edge_patches_from_markov_chain(P, X, width=3, node_rad=0, tol=1e-7, connectionstyle='arc3, rad=0.25',
                                          facecolor='k', **kwargs):
    arrows = []
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            if P[i, j] > tol:
                arrows.append(create_edge_patch(X[i], X[j], width=P[i, j] * width, node_rad=node_rad,
                                                connectionstyle=connectionstyle, facecolor=facecolor,
                                                alpha=min(2 * P[i, j], 1), **kwargs))
    return arrows


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
                save_or_show='show',
                s_kwargs_dict={},
                **kwargs):

    import matplotlib.pyplot as plt

    points = adata.obsm['X_' + basis][:, [x, y]]
    groups, uniq_grp = adata.obs[group], adata.obs[group].unique().to_list()
    group_median = np.zeros((len(uniq_grp), 2))
    grp_size = adata.obs[group].value_counts().values
    s_kwargs_dict.update({'s': grp_size})

    Pl = adata.uns['Cell type annotation_graph']['group_graph']
    Pl[Pl - Pl.T < 0] = 0
    for i, cur_grp in enumerate(uniq_grp):
        group_median[i, :] = np.nanmedian(points[np.where(groups == cur_grp)[0], :2], 0)

    ax, color = _matplotlib_points(
                    group_median,
                    ax,
                    labels,
                    values,
                    highlights,
                    cmap,
                    color_key,
                    color_key_cmap,
                    background,
                    figsize[0],
                    figsize[1],
                    show_legend,
                    **s_kwargs_dict
                )


    ax.colorbar()
    arrows = create_edge_patches_from_markov_chain(Pl, group_median, tol=0.01, node_rad=15)
    for arrow in arrows:
        ax.add_patch(arrow)

    plt.axis('off')

    plt.show()
