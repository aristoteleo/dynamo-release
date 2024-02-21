from typing import Any, Dict, Optional, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes

from ..prediction.fate import fate_bias as fate_bias_pd
from .scatters import scatters
from .utils import map2color, save_show_ret


def fate_bias(
    adata: AnnData,
    group: str,
    basis: str = "umap",
    fate_bias_df: Optional[pd.DataFrame] = None,
    figsize: Tuple[float, float] = (6, 4),
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
    **cluster_maps_kwargs
) -> Optional[Axes]:
    """Plot the lineage (fate) bias of cells states whose vector field trajectories are predicted.

    This function internally calls `dyn.tl.fate_bias` to calculate fate bias dataframe. You can also visualize the data
    frame via pandas stlying (https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html), for example:
        >>> df = dyn.vf.fate_bias(adata)
        >>> df.style.background_gradient(cmap='viridis')

    Args:
        adata: the AnnData object that contains the predicted fate trajectories in the `uns` attribute.
        group: the column key that corresponds to the cell type or other group information for quantifying the bias of
            cell state.
        basis: the embedding data space that cell fates were predicted and cell fates will be quantified. Defaults to
            "umap".
        fate_bias_df: the DataFrame that stores the fate bias information. If None, it would be calculated via
             fate_bias_df = dyn.tl.fate_bias(adata). Defaults to None.
        figsize: the size of the figure. Defaults to (6, 4).
        save_show_or_return: whether to save, show or return the figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the
                {
                    "path": None,
                    "prefix": 'phase_portraits',
                    "dpi": None,
                    "ext": 'pdf',
                    "transparent": True,
                    "close": True,
                    "verbose": True
                }
            as its parameters. Otherwise, you can provide a dictionary that properly modify those keys according to
            your needs. Defaults to {}.
        **cluster_map_kwargs: any other kwargs to be passed to `seaborn.clustermap`.

    Returns:
        None would be returned by default and the heatmap showing the fate bias of each cell state to each of the cell
            group would be shown. If `save_show_or_return` is set to be `return`, the matplotlib axis of the plot would
            be returned.
    """

    fate_bias = fate_bias_pd(adata, group=group, basis=basis) if fate_bias_df is None else fate_bias_df

    if "confidence" in fate_bias.keys():
        fate_bias.set_index([fate_bias.index, fate_bias.confidence], inplace=True)

    fate_bias.fillna(0, inplace=True)

    ax = sns.clustermap(
        fate_bias, col_cluster=True, row_cluster=True, figsize=figsize, yticklabels=False, **cluster_maps_kwargs
    )

    return save_show_ret("fate_bias", save_show_or_return, save_kwargs, ax)


def fate(
    adata: AnnData,
    x: int = 0,
    y: int = 1,
    basis: str = "pca",
    color: str = "ntr",
    ax: Optional[Axes] = None,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
    **kwargs
) -> Optional[Axes]:
    """Draw the predicted integration paths on the low-dimensional embedding.

    Args:
        adata: an Annodata object.
        x: the column index of the low dimensional embedding for the x-axis. Defaults to 0.
        y: the column index of the low dimensional embedding for the y-axis. Defaults to 1.
        basis: the basis used for dimension reduction. Defaults to "pca".
        color: any column names or gene expression, etc. that will be used for coloring cells. Defaults to "ntr".
        ax: the matplotlib axes object where new plots will be added to. Only applicable to drawing a single component.
            If None, new axis would be created. Defaults to None.
        save_show_or_return: whether to save, show or return the figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the
                {
                    "path": None,
                    "prefix": 'phase_portraits',
                    "dpi": None,
                    "ext": 'pdf',
                    "transparent": True,
                    "close": True,
                    "verbose": True
                }
            as its parameters. Otherwise, you can provide a dictionary that properly modify those keys according to
            your needs. Defaults to {}.
        **kwargs: any other kwargs to be passed to `dynamo.pl.scatters`.
    Returns:
        None would be returned by default. If `save_show_or_return` is set to be `return`, the matplotlib axis of the
        plot would be returned.
    """

    ax = scatters(adata, basis=basis, color=color, save_show_or_return="return", ax=ax, **kwargs)

    fate_key = "fate" if basis is None else "fate_" + basis
    lap_dict = adata.uns[fate_key]

    for i, j in zip(lap_dict["prediction"], lap_dict["t"]):
        ax.scatter(*i.T[:, [x, y]].T, c=map2color(j))
        ax.plot(*i.T[:, [x, y]].T, c="k")

    return save_show_ret("kinetic_curves", save_show_or_return, save_kwargs, ax)
