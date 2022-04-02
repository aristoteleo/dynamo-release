import numpy as np

from ..prediction.utils import (
    interp_curvature,
    interp_second_derivative,
    kneedle_difference,
)
from ..tools.utils import update_dict
from ..utils import denormalize, normalize
from .ezplots import plot_X, zscatter
from .scatters import save_fig, scatters
from .utils import map2color


def least_action(
    adata, x=0, y=1, basis="pca", color="ntr", ax=None, save_show_or_return="show", save_kwargs={}, **kwargs
):
    """Draw the least action paths on the low-dimensional embedding.

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
        ax: `matplotlib.Axis` (optional, default `None`)
            The matplotlib axes object where new plots will be added to. Only applicable to drawing a single component.
        save_show_or_return: `str` {'save', 'show', 'return'} (default: `show`)
            Whether to save, show or return the figure. If "both", it will save and plot the figure at the same time. If
            "all", the figure will be saved, displayed and the associated axis and other object will be return.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
            save_fig function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent":
            True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that
            properly modify those keys according to your needs.
        kwargs:
            Additional arguments passed to pl.scatters or plt.scatters.

    Returns
    -------
        result:
            Either None or a matplotlib axis with the relevant plot displayed.
            If you are using a notbooks and have ``%matplotlib inline`` set
            then this will simply display inline.
    """

    import matplotlib.pyplot as plt

    ax = scatters(adata, basis=basis, color=color, save_show_or_return="return", ax=ax, **kwargs)

    LAP_key = "LAP" if basis is None else "LAP_" + basis
    lap_dict = adata.uns[LAP_key]

    for i, j in zip(lap_dict["prediction"], lap_dict["action"]):
        ax.scatter(*i[:, [x, y]].T, c=map2color(j))
        ax.plot(*i[:, [x, y]].T, c="k")

    if save_show_or_return in ["save", "both", "all"]:
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
    elif save_show_or_return in ["show", "both", "all"]:
        # TODO: least_action_path.py:74: UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
        # plt.tight_layout()
        plt.show()
    elif save_show_or_return in ["return", "all"]:
        return ax


def lap_min_time(
    adata,
    basis="pca",
    show_paths=False,
    show_elbow=True,
    show_elbow_func=False,
    color="ntr",
    figsize=(6, 4),
    n_col=3,
    save_show_or_return="show",
    save_kwargs={},
    **kwargs,
):

    import matplotlib.pyplot as plt

    LAP_key = "LAP" if basis is None else "LAP_" + basis
    min_t_dict = adata.uns[LAP_key]["min_t"]
    method = min_t_dict["method"]

    for k in range(len(min_t_dict["A"])):
        A = min_t_dict["A"][k]
        T = min_t_dict["T"][k]
        i_elbow = min_t_dict["i_elbow"][k]
        paths = min_t_dict["paths"][k]

        num_t = len(A)
        if method == "hessian":
            T_, A_ = normalize(T), normalize(A)
            T_, der = interp_second_derivative(T_, A_)
            T_ = denormalize(T_, np.min(T), np.max(T))
        elif method == "curvature":
            T_, A_ = normalize(T), normalize(A)
            T_, der = interp_curvature(T_, A_)
            T_ = denormalize(T_, np.min(T), np.max(T))
        elif method == "kneedle":
            der = kneedle_difference(T, A)
            T_ = T
        else:
            raise NotImplementedError(f"Unsupported method {method}.")

        if show_paths:
            n_row = int(np.ceil((num_t + 1) / n_col))
        else:
            n_row = 1
            n_col = 1

        figsize = (figsize[0] * n_col, figsize[1] * n_row) if figsize is not None else (4 * n_col, 4 * n_row)
        fig, axes = plt.subplots(n_row, n_col, figsize=figsize, sharex=False, sharey=False, squeeze=False)

        for c in range(1 + num_t):
            i, j = c % n_row, c // n_row

            if c == 0:
                axes[i, j].plot(T, A)
                if show_elbow:
                    axes[i, j].plot([T[i_elbow], T[i_elbow]], [np.max(A), np.min(A)], "--")
                axes[i, j].set_xlabel("LAP time")
                axes[i, j].set_ylabel("action")
                # axes[i, j].set_title(f'pair {i}')

                if show_elbow_func:
                    ax2 = axes[i, j].twinx()
                    ax2.plot(T_, der, c="r")
                    ax2.tick_params(axis="y", labelcolor="r")
                    ax2.set_ylabel(method)

            elif show_paths:
                plt.sca(axes[i, j])
                zscatter(adata, basis=basis, color=color)
                plot_X(paths[c - 1], c="k")
                plt.title(f"path {c-1}")
                # scatters(adata, basis=basis, color=color, ax=axes[i, j], **kwargs)
                # axes[i, j].scatter(*i[:, [x, y]].T, c=map2color(j))

        if save_show_or_return in ["save", "both", "all"]:
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
        elif save_show_or_return in ["show", "both", "all"]:
            plt.tight_layout()
            plt.show()
