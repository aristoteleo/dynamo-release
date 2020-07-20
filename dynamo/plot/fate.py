import numpy as np
import seaborn as sns
from scipy.integrate import odeint
from .fate_utilities import Animation
from .utils import save_fig
from ..prediction.fate import fate_bias as fate_bias_pd
from ..tools.utils import vector_field_function, update_dict

def fate_bias(adata,
              group,
              basis='umap',
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

    fate_bias = fate_bias_pd(adata, group=group, basis=basis)

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


class StreamFuncAnim(Animation):  #
    def __init__(self, VecFld, dt=0.05, xlim=(-1, 1), ylim=None):
        import matplotlib.pyplot as plt

        self.dt = dt
        # Initialize velocity field and displace *functions*
        self.f = lambda x, t=None: vector_field_function(x, VecFld=VecFld)
        self.displace = lambda x, dt: odeint(self.f, x.flatten(), [0, dt])
        # Save bounds of plot
        self.xlim = xlim
        self.ylim = ylim if ylim is not None else xlim
        # Animation objects must create `fig` and `ax` attributes.
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal")

    def init_background(self):
        """Draw background with streamlines of flow.

        Note: artists drawn here aren't removed or updated between frames.
        """
        u = np.linspace(self.xlim[0], self.xlim[1], 100)
        v = np.linspace(self.ylim[0], self.ylim[1], 100)
        uu, vv = np.meshgrid(u, v)

        # Compute derivatives
        u_vel = np.empty_like(uu)
        v_vel = np.empty_like(vv)
        for i in range(uu.shape[0]):
            for j in range(uu.shape[1]):
                u_vel[i, j], v_vel[i, j] = self.f(np.array([uu[i, j], vv[i, j]]), None)

        streamline = self.ax.streamplot(uu, vv, u_vel, v_vel, color="0.7")

        return streamline

    def update(self):
        """Update locations of "particles" in flow on each frame frame."""
        pts = []
        while True:
            pts = list(pts)
            pts.append((random_xy(self.xlim), random_xy(self.ylim)))
            pts = [
                self.displace(np.array(cur_pts).reshape(-1, 1), self.dt)
                for cur_pts in pts
            ]
            pts = np.asarray(pts)
            pts = remove_particles(pts, self.xlim, self.ylim)
            self.ax.lines = []

            x, y = np.asarray(pts).transpose()
            (lines,) = self.ax.plot(x, y, "ro")
            yield lines,  # return line so that blit works properly


def random_xy(lim):
    return list(np.random.uniform(lim[0], lim[1], 1))


def remove_particles(pts, xlim, ylim):
    if len(pts) == 0:
        return []
    outside_xlim = (pts[:, 0] < xlim[0]) | (pts[:, 0] > xlim[1])
    outside_ylim = (pts[:, 1] < ylim[0]) | (pts[:, 1] > ylim[1])
    keep = ~(outside_xlim | outside_ylim)
    return pts[keep]


class Flow(StreamFuncAnim):
    def init_background(self):
        StreamFuncAnim.init_background(self)


"""
# use example 
VF = lambda x, t=None: vector_field_function(x=x, t=t, VecFld=VecFld)
toggle_flow = Flow(VecFld, xlim=(0, 6), ylim=(0, 6), dt=1e-500)
toggle_flow.animate(blit=False)
plt.show()
"""
