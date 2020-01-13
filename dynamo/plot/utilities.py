import numpy as np
import math

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


def set_spine_linewidth(ax, lw):
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(lw)

    return ax

def scatter_with_colorbar(fig, ax, x, y, c, cmap, **scatter_kwargs):
    # https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    g = ax.scatter(x, y, c=c, cmap=cmap, **scatter_kwargs)
    fig.colorbar(g, cax=cax, orientation='vertical')

    return fig, ax


def scatter_with_legend(fig, ax, df, color, font_color, x, y, c, cmap, legend, **scatter_kwargs):
    import seaborn as sns
    import matplotlib.patheffects as PathEffects

    unique_labels = np.unique(c)
    g = sns.scatterplot(x, y, hue=c,
                        palette=cmap, ax=ax, \
                        legend='full', **scatter_kwargs)
    if legend == 'on data':
        for i in unique_labels:
            color_cnt = np.nanmedian(df.iloc[np.where(color == i)[0], :2], 0)
            txt = ax.text(color_cnt[0], color_cnt[1], str(i), fontsize=13, c=font_color, zorder=0)  # c
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground=font_color, alpha=0.1),  # 'w'
                PathEffects.Normal()])
    else:
        # ax1.set_legend(loc=legend, bbox_to_anchor=(0.125, 0.125), ncol=1 if label_len < 15 else 2)
        ax.legend(loc=legend, bbox_to_anchor=(0.125, 0.125), ncol=1 if len(unique_labels) < 15 else 2)

    return fig, ax


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


def tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= -1) & (x <= 1)
    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
    return y


# the following Loess class is taken from:
# https://github.com/joaofig/pyloess/blob/master/pyloess/Loess.py
class Loess(object):

    @staticmethod
    def normalize_array(array):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val), min_val, max_val

    def __init__(self, xx, yy, degree=1):
        self.n_xx, self.min_xx, self.max_xx = self.normalize_array(xx)
        self.n_yy, self.min_yy, self.max_yy = self.normalize_array(yy)
        self.degree = degree

    @staticmethod
    def get_min_range(distances, window):
        min_idx = np.argmin(distances)
        n = len(distances)
        if min_idx == 0:
            return np.arange(0, window)
        if min_idx == n-1:
            return np.arange(n - window, n)

        min_range = [min_idx]
        while len(min_range) < window:
            i0 = min_range[0]
            i1 = min_range[-1]
            if i0 == 0:
                min_range.append(i1 + 1)
            elif i1 == n-1:
                min_range.insert(0, i0 - 1)
            elif distances[i0-1] < distances[i1+1]:
                min_range.insert(0, i0 - 1)
            else:
                min_range.append(i1 + 1)
        return np.array(min_range)

    @staticmethod
    def get_weights(distances, min_range):
        max_distance = np.max(distances[min_range])
        weights = tricubic(distances[min_range] / max_distance)
        return weights

    def normalize_x(self, value):
        return (value - self.min_xx) / (self.max_xx - self.min_xx)

    def denormalize_y(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy

    def estimate(self, x, window, use_matrix=False, degree=1):
        n_x = self.normalize_x(x)
        distances = np.abs(self.n_xx - n_x)
        min_range = self.get_min_range(distances, window)
        weights = self.get_weights(distances, min_range)

        if use_matrix or degree > 1:
            wm = np.multiply(np.eye(window), weights)
            xm = np.ones((window, degree + 1))

            xp = np.array([[math.pow(n_x, p)] for p in range(degree + 1)])
            for i in range(1, degree + 1):
                xm[:, i] = np.power(self.n_xx[min_range], i)

            ym = self.n_yy[min_range]
            xmt_wm = np.transpose(xm) @ wm
            beta = np.linalg.pinv(xmt_wm @ xm) @ xmt_wm @ ym
            y = (beta @ xp)[0]
        else:
            xx = self.n_xx[min_range]
            yy = self.n_yy[min_range]
            sum_weight = np.sum(weights)
            sum_weight_x = np.dot(xx, weights)
            sum_weight_y = np.dot(yy, weights)
            sum_weight_x2 = np.dot(np.multiply(xx, xx), weights)
            sum_weight_xy = np.dot(np.multiply(xx, yy), weights)

            mean_x = sum_weight_x / sum_weight
            mean_y = sum_weight_y / sum_weight

            b = (sum_weight_xy - mean_x * mean_y * sum_weight) / \
                (sum_weight_x2 - mean_x * mean_x * sum_weight)
            a = mean_y - b * mean_x
            y = a + b * n_x
        return self.denormalize_y(y)
