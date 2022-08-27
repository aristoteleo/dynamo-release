from typing import List, NamedTuple, Optional, Union

import matplotlib
import pandas as pd
import seaborn as sns
from anndata import AnnData

from ..prediction.fate import fate_bias as fate_bias_pd
from ..tools.utils import update_dict
from .scatters import save_fig, scatters
from .utils import map2color


def fate_bias(
    adata: AnnData,
    group: str,
    basis: Union[str, None] = "umap",
    fate_bias_df: Union[pd.DataFrame, None] = None,
    figsize: tuple = (6, 4),
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
    **cluster_maps_kwargs: dict
):
    """Plot the lineage (fate) bias of cells states whose vector field trajectories are predicted.

    This function internally calls `dyn.tl.fate_bias` to calculate fate bias dataframe. You can also visualize the data
    frame via pandas stlying (https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html), for example:

        >>> df = dyn.vf.fate_bias(adata)
        >>> df.style.background_gradient(cmap='viridis')

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the predicted fate trajectories in the `uns` attribute.
        group: `str`
            The column key that corresponds to the cell type or other group information for quantifying the bias of cell
            state.
        basis: `str` or None (default: `None`)
            The embedding data space that cell fates were predicted and cell fates will be quantified.
        fate_bias_df: `pandas.DataFrame` or None (default: `None`)
            The DataFrame that stores the fate bias information, calculated via fate_bias_df = dyn.tl.fate_bias(adata).
        figsize: `None` or `[float, float]` (default: None)
            The width and height of a figure.
        save_show_or_return: {'show', 'save', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
            save_fig function will use the {"path": None, "prefix": 'fate_bias', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a
            dictionary that properly modify those keys according to your needs.
        cluster_maps_kwargs:
            Additional arguments passed to sns.clustermap.

    Returns
    -------
        Nothing but plot a heatmap shows the fate bias of each cell state to each of the cell group.
    """

    import matplotlib.pyplot as plt

    fate_bias = fate_bias_pd(adata, group=group, basis=basis) if fate_bias_df is None else fate_bias_df

    if "confidence" in fate_bias.keys():
        fate_bias.set_index([fate_bias.index, fate_bias.confidence], inplace=True)

    ax = sns.clustermap(
        fate_bias, col_cluster=True, row_cluster=True, figsize=figsize, yticklabels=False, **cluster_maps_kwargs
    )

    if save_show_or_return == "save":
        s_kwargs = {
            "path": None,
            "prefix": "fate_bias",
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
        return ax


def fate(
    adata: AnnData,
    x: int = 0,
    y: int = 1,
    basis: str = "pca",
    color: str = "ntr",
    ax: matplotlib.axes.Axes = None,
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
    **kwargs: dict
):
    """Draw the predicted integration paths on the low-dimensional embedding.

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

    fate_key = "fate" if basis is None else "fate_" + basis
    lap_dict = adata.uns[fate_key]

    for i, j in zip(lap_dict["prediction"], lap_dict["t"]):
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
        plt.tight_layout()
        plt.show()
    elif save_show_or_return in ["return", "all"]:
        return ax
