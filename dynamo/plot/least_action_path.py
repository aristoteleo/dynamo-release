from ..tools.utils import update_dict
from .scatters import scatters, save_fig
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
        plt.tight_layout()
        plt.show()
    elif save_show_or_return in ["return", "all"]:
        return ax
