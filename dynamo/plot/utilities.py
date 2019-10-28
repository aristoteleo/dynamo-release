import numpy as np


# plotting utility functions from https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb

def despline(ax1=None):
    import matplotlib.pyplot as plt

    ax1 = plt.gca() if ax1 is None else ax1
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

def minimal_xticks(start, end):
    import matplotlib.pyplot as plt

    end_ = np.around(end, -int(np.log10(end))+1)
    xlims = np.linspace(start, end_, 5)
    xlims_tx = [""]*len(xlims)
    xlims_tx[0], xlims_tx[-1] = f"{xlims[0]:.0f}", f"{xlims[-1]:.02f}"
    plt.xticks(xlims, xlims_tx)


def minimal_yticks(start, end):
    import matplotlib.pyplot as plt

    end_ = np.around(end, -int(np.log10(end))+1)
    ylims = np.linspace(start, end_, 5)
    ylims_tx = [""]*len(ylims)
    ylims_tx[0], ylims_tx[-1] = f"{ylims[0]:.0f}", f"{ylims[-1]:.02f}"
    plt.yticks(ylims, ylims_tx)


def quiver_autoscaler(X_emb, V_emb):
    """Function to automatically calculate the value for the scale parameter of quiver plot, adapted from scVelo

    Parameters
    ----------
        X_emb: `np.ndarray`
            X, Y-axis coordinates
        V_emb:  `np.ndarray`
            Velocity (U, V) values on the X, Y-axis

    Returns
    -------
        The scale for quiver plot
    """

    import matplotlib.pyplot as plt
    scale_factor = np.ptp(X_emb, 0).mean()
    X_emb = X_emb - X_emb.min(0)

    if len(V_emb.shape) == 3:
        Q = plt.quiver(X_emb[0] / scale_factor, X_emb[1] / scale_factor,
                   V_emb[0], V_emb[1], angles='xy', scale_units='xy', scale=None)
    else:
        Q = plt.quiver(X_emb[:, 0] / scale_factor, X_emb[:, 1] / scale_factor,
                      V_emb[:, 0], V_emb[:, 1], angles='xy', scale_units='xy', scale=None)

    Q._init()
    plt.clf()

    return Q.scale / scale_factor
