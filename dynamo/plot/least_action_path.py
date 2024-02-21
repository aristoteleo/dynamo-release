from typing import Any, Dict, Optional, Tuple

import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..prediction.utils import (
    interp_curvature,
    interp_second_derivative,
    kneedle_difference,
)
from ..utils import denormalize, normalize
from .ezplots import plot_X, zscatter
from .scatters import save_show_ret, scatters
from .utils import map2color


def least_action(
    adata: AnnData,
    x: int = 0,
    y: int = 1,
    basis: str = "pca",
    color: str = "ntr",
    ax: Optional[Axes] = None,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Dict[str, Any] = {},
    **kwargs,
) -> Optional[Axes]:
    """Draw the least action paths on the low-dimensional embedding.

    Args:
        adata: an AnnData object.
        x: the column index of the low dimensional embedding for the x-axis. Defaults to 0.
        y: the column index of the low dimensional embedding for the y-axis. Defaults to 1.
        basis: the basis used for dimension reduction. Defaults to "pca".
        color: any column names or gene expression, etc. that will be used for coloring cells. Defaults to "ntr".
        ax: the matplotlib axes object where new plots will be added to. Only applicable to drawing a single component.
            If None, new axis would be created. Defaults to None.
        save_show_or_return: whether the figure should be saved, show, or return. Can be one of "save", "show",
            "return", "both", "all". "both" means that the figure would be shown and saved but not returned. Defaults to
            "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be `"return"` or `"all"`, the matplotlib
        axis of the generated figure would be returned.
    """

    import matplotlib.pyplot as plt

    ax = scatters(adata, basis=basis, color=color, save_show_or_return="return", ax=ax, **kwargs)

    LAP_key = "LAP" if basis is None else "LAP_" + basis
    lap_dict = adata.uns[LAP_key]

    for i, j in zip(lap_dict["prediction"], lap_dict["action"]):
        ax.scatter(*i[:, [x, y]].T, c=map2color(j))
        ax.plot(*i[:, [x, y]].T, c="k")

    return save_show_ret("kinetic_curves", save_show_or_return, save_kwargs, ax)


def lap_min_time(
    adata: AnnData,
    basis: str = "pca",
    show_paths: bool = False,
    show_elbow: bool = True,
    show_elbow_func: bool = False,
    color: str = "ntr",
    figsize: Tuple[float, float] = (6, 4),
    n_col: int = 3,
    save_show_or_return: Literal["save", "show", "both", "all"] = "show",
    save_kwargs: Dict[str, Any] = {},
    **kwargs,
) -> None:
    """Plot minimum time of the least action paths.

    Args:
        adata: an AnnData object.
        basis: the basis used for dimension reduction. Defaults to "pca".
        show_paths: whether to plot the path together with the time. Defaults to False.
        show_elbow: whether to mark the elbow point on time-action curve. Defaults to True.
        show_elbow_func: whether to show the time-action curve that elbow is on. Defaults to False.
        color: any column names or gene expression, etc. that will be used for coloring cells. Defaults to "ntr".
        figsize: the size of the figure. Defaults to (6, 4).
        n_col: the number of subplot columns. Defaults to 3.
        save_show_or_return: whether to save or show the figure. Can be one of "save", "show", "both" or "all". "both"
            and "all" have the same effect. The axis of the plot cannot be returned here. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide
            a dictionary that properly modify those keys according to your needs. Defaults to {}.
        **kwargs: not used here.

    Raises:
        NotImplementedError: unsupported method to find the elbow.
    """

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
                zscatter(adata, basis=basis, color=color, save_show_or_return="return")
                plot_X(paths[c - 1], c="k", save_show_or_return="return")
                plt.title(f"path {c-1}")
                # scatters(adata, basis=basis, color=color, ax=axes[i, j], **kwargs)
                # axes[i, j].scatter(*i[:, [x, y]].T, c=map2color(j))

        return save_show_ret("kinetic_curves", save_show_or_return, save_kwargs)
