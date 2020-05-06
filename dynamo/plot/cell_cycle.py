from ..tools.utils import update_dict
from .utils import save_fig


def cell_cycle_scores(adata, cells=None, save_show_or_return='show', save_kwargs={},):
    """Plot a heatmap of cells ordered by cell cycle position

    Parameters
    ----------
        adata: CellPopulation instance
        cells: query string for cell properties (i.e. executes pop.cells.query(cells=cells))
        **kwargs: all other keyword arguments are passed to pop.where
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from mpl_toolkits.axes_grid1.colorbar import colorbar

    if cells is None:
        cell_cycle_scores = adata.obsm['cell_cycle_scores'].dropna()
    else:
        cell_cycle_scores = adata[cells, :].obsm['cell_cycle_scores'].dropna().dropna()

    cell_cycle_scores.sort_values(['cell_cycle_phase', 'cell_cycle_progress'],
                                  ascending=[True, False],
                                  inplace=True)

    # based on https://stackoverflow.com/questions/47916205/seaborn-heatmap-move-colorbar-on-top-of-the-plot
    # answwer 4

    # plot heatmap without colorbar
    ax = sns.heatmap(cell_cycle_scores[['G1-S', 'S', 'G2-M', 'M', 'M-G1']].transpose(),
                annot=False, xticklabels=False, linewidths=0, cbar=False) #
    # split axes of heatmap to put colorbar
    ax_divider = make_axes_locatable(ax)
    # define size and padding of axes for colorbar
    cax = ax_divider.append_axes('right', size='2%', pad='0.5%', aspect=4, anchor='NW')
    # make colorbar for heatmap.
    # Heatmap returns an axes obj but you need to get a mappable obj (get_children)
    colorbar(ax.get_children()[0], cax=cax, ticks=[-0.9, 0, 0.9])

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'plot_direct_graph', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return ax
