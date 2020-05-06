import seaborn as sns

def cell_cycle_position_heatmap(pop, cells=None, **kwargs):
    """Plot a heatmap of cells ordered by cell cycle position

    Args:
        pop: CellPopulation instance
        cells: query string for cell properties (i.e. executes pop.cells.query(cells=cells))
        **kwargs: all other keyword arguments are passed to pop.where
    """
    if cells is None:
        cell_cycle_scores = pop.cells[
            ['G1-S', 'S', 'G2-M', 'M', 'M-G1', 'cell_cycle_phase', 'cell_cycle_progress']].dropna()
    else:
        celllist = pop.where(cells=cells, **kwargs).index
        cell_cycle_scores = pop.cells.loc[celllist][
            ['G1-S', 'S', 'G2-M', 'M', 'M-G1', 'cell_cycle_phase', 'cell_cycle_progress']].dropna()

    cell_cycle_scores.sort_values(['cell_cycle_phase', 'cell_cycle_progress'],
                                  ascending=[True, False],
                                  inplace=True)
    ax = sns.heatmap(cell_cycle_scores[cell_cycle_scores.columns[:-2]].transpose(), annot=False, xticklabels=False,
                     linewidths=0)
