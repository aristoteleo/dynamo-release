from typing import Any, Dict, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from anndata import AnnData
from matplotlib.axes import Axes

from .utils import save_show_ret


def cell_cycle_scores(
    adata: AnnData,
    cells: Optional[list] = None,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
) -> Optional[Axes]:
    """Plot a heatmap of cells ordered by cell cycle position.

    Args:
        adata: an AnnData object.
        cells: a list of cell ids used to subset the AnnData object. If None, all cells would be used. Defaults to None.
        save_show_or_return: whether to save, show, or return the figure. Available flags are `"save"`, `"show"`, and
            `"return"`. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent":
            True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that
            properly modify those keys according to your needs. Defaults to {}.

    Raises:
        NotImplementedError: unavailable save_show_or_return

    Returns:
        Axes of the plotted figure if `save_show_or_return` is set to `"return"`; otherwise, return `None`.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.pyplot import colorbar
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

    if cells is None:
        cell_cycle_scores = adata.obsm["cell_cycle_scores"].dropna()
    else:
        cell_cycle_scores = adata[cells, :].obsm["cell_cycle_scores"].dropna().dropna()

    cell_cycle_scores.sort_values(
        ["cell_cycle_phase", "cell_cycle_progress"],
        ascending=[True, False],
        inplace=True,
    )

    # based on https://stackoverflow.com/questions/47916205/seaborn-heatmap-move-colorbar-on-top-of-the-plot
    # answwer 4

    # plot heatmap without colorbar
    ax = sns.heatmap(
        cell_cycle_scores[["G1-S", "S", "G2-M", "M", "M-G1"]].transpose(),
        annot=False,
        xticklabels=False,
        linewidths=0,
        cbar=False,
    )  #
    # split axes of heatmap to put colorbar
    ax_divider = make_axes_locatable(ax)
    # define size and padding of axes for colorbar
    cax = ax_divider.append_axes("right", size="2%", pad="0.5%", aspect=4, anchor="NW")
    # make colorbar for heatmap.
    # Heatmap returns an axes obj but you need to get a mappable obj (get_children)
    colorbar(ax.get_children()[0], cax=cax, ticks=[-0.9, 0, 0.9])

    return save_show_ret("plot_direct_graph", save_show_or_return, save_kwargs, ax)
