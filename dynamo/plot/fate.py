import numpy as np
import seaborn as sns
import warnings
from .utils import save_fig
from ..prediction.fate import fate_bias as fate_bias_pd
from ..tools.utils import update_dict

def fate_bias(adata,
              group,
              basis='umap',
              fate_bias_df=None,
              figsize=(6, 4),
              save_show_or_return='show',
              save_kwargs={},
              **cluster_maps_kwargs
              ):
    """Plot the lineage (fate) bias of cells states whose vector field trajectories are predicted.

    This function internally calls `dyn.tl.fate_bias` to calculate fate bias dataframe. You can also visualize the data
    frame via pandas stlying (https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html), for example:

        >>> df = dyn.tl.fate_bias(adata)
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
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
            will use the {"path": None, "prefix": 'fate_bias', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.
        cluster_maps_kwargs:
            Additional arguments passed to sns.clustermap.

    Returns
    -------
        Nothing but plot a heatmap shows the fate bias of each cell state to each of the cell group.
    """

    import matplotlib.pyplot as plt

    fate_bias = fate_bias_pd(adata, group=group, basis=basis) if fate_bias_df is None else fate_bias_df

    if 'confidence' in fate_bias.keys():
        fate_bias.set_index([fate_bias.index, fate_bias.confidence], inplace=True)

    ax = sns.clustermap(fate_bias, col_cluster=True, row_cluster=True, figsize=figsize, yticklabels=False,
                        **cluster_maps_kwargs)

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'fate_bias', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return ax

