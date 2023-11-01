import matplotlib.pyplot as plt


def filter_fig(fig):
    # remove colorbar from the fig because shiny is incompatible matplotlib>=3.7
    for ax in fig.get_axes():
        if ax.get_subplotspec() is None:
            ax.remove()
    return fig