import numpy as np
import seaborn as sns
import warnings
from scipy.integrate import odeint
from matplotlib import animation
from .utils import save_fig
from ..prediction.fate import fate_bias as fate_bias_pd
from ..tools.utils import update_dict
from ..tools.scVectorField import vectorfield
from ..plot.topography import topography

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


class StreamFuncAnim():
    def __init__(self,
                 adata,
                 basis='umap',
                 dims=None,
                 n_steps=100,
                 cell_states=None,
                 color='ntr',
                 ax=None,
                 ln=None,
                 ):
        """Animating cell fate commitment prediction via reconstructed vector field function.

        This class creates necessary components to produce an animation that describes the exact speed of a set of cells
        at each time point, its movement in gene expression and the long range trajectory predicted by the reconstructed
        vector field. Thus it provides intuitive visual understanding of the RNA velocity, speed, acceleration, and cell
        fate commitment in action.

        This function is originally inspired by https://tonysyu.github.io/animating-particles-in-a-flow.html.

        Parameters
        ----------
            adata: :class:`~anndata.AnnData`
                AnnData object that already went through the fate prediction.
            basis: `str` or None (default: `None`)
                The embedding data to use for predicting cell fate. If `basis` is either `umap` or `pca`, the reconstructed
                trajectory will be projected back to high dimensional space via the `inverse_transform` function.
                space.
            dims: `list` or `None` (default: `None')
                The dimensions of low embedding space where cells will be drawn and it should corresponds to the space
                fate prediction take place.
            n_steps: `int` (default: `100`)
                The number of times steps (frames) fate prediction will take.
            cell_states: `int`, `list` or `None` (default: `None`)
                The number of cells state that will be randomly selected (if `int`), the indices of the cells states (if
                `list`) or all cell states which fate prediction executed (if `None`)
            ax: `matplotlib.Axis` (optional, default `None`)
                The matplotlib axes object that will be used as background plot of the vector field animation.
            ln: `tuple` or None (default: `None`)
                An iterable of artists (for example, `matplotlib.lines.Line2D`) used to draw a clear frame.

        Returns
        -------
            A class that contains .fig attribute and .update, .init_background that can be used to produce an animation
            of the prediction of cell fate commitment.

        Examples 1
        ----------
        >>> from matplotlib import animation
        >>> progenitor = adata.obs_names[adata.obs.clusters == 'cluster_1']
        >>> fate_progenitor = progenitor
        >>> info_genes = adata.var_names[adata.var.use_for_velocity]
        >>> dyn.pd.fate(adata, basis='umap', init_cells=fate_progenitor, interpolation_num=100,  direction='forward',
        ...    inverse_transform=False, average=False, arclen_sampling=True)
        >>> instance = dyn.pl.StreamFuncAnim(adata=adata, ax=None, ln=None)
        >>> anim = animation.FuncAnimation(instance.fig, instance.update, init_func=instance.init_background,
        ...                                frames=np.arange(100), interval=100, blit=True)
        >>> from IPython.core.display import display, HTML
        >>> HTML(anim.to_jshtml()) # embedding to jupyter notebook.
        >>> anim.save('fate_ani.gif',writer="imagemagick") # save as gif file.

        Examples 2
        ----------
        >>> from matplotlib import animation
        >>> progenitor = adata.obs_names[adata.obs.clusters == 'cluster_1']
        >>> fate_progenitor = progenitor
        >>> info_genes = adata.var_names[adata.var.use_for_velocity]
        >>> dyn.pd.fate(adata, basis='umap', init_cells=fate_progenitor, interpolation_num=100,  direction='forward',
        ...    inverse_transform=False, average=False, arclen_sampling=True)
        >>> fig, ax = plt.subplots()
        >>> ln, = ax.plot([], [], 'ro')
        >>> ax.set_xlim(xlim)
        >>> ax.set_ylim(ylim)
        >>> instance = dyn.pl.StreamFuncAnim(adata=adata, ax=ax, ln=ln)
        >>> anim = animation.FuncAnimation(fig, instance.update, init_func=instance.init_background,
        ...                                frames=np.arange(100), interval=100, blit=True)
        >>> from IPython.core.display import display, HTML
        >>> HTML(anim.to_jshtml()) # embedding to jupyter notebook.
        >>> anim.save('fate_ani.gif',writer="imagemagick") # save as gif file.

        Examples 3
        ----------
        >>> from matplotlib import animation
        >>> progenitor = adata.obs_names[adata.obs.clusters == 'cluster_1']
        >>> fate_progenitor = progenitor
        >>> info_genes = adata.var_names[adata.var.use_for_velocity]
        >>> dyn.pd.fate(adata, basis='umap', init_cells=fate_progenitor, interpolation_num=100,  direction='forward',
        ...    inverse_transform=False, average=False, arclen_sampling=True)
        >>> dyn.pl.fate_animation(adata)
        """

        import matplotlib.pyplot as plt

        self.adata = adata
        self.basis = basis

        fate_key = 'fate_' + basis
        if fate_key not in adata.uns_keys():
            raise Exception(f"You need to first perform fate prediction before animate the prediction, please run"
                            f"dyn.pd.fate(adata, basis='{basis}' before running this function")

        self.init_states = adata.uns[fate_key]['init_states']
        # self.prediction = adata.uns['fate_umap']['prediction']
        self.t = adata.uns[fate_key]['t']

        flat_list = np.unique([item for sublist in self.t for item in sublist])
        flat_list = np.hstack((0, flat_list))
        flat_list = np.sort(flat_list)
        self.time_vec = flat_list[(np.linspace(0, len(flat_list) - 1, n_steps)).astype(int)]

        # init_states, VecFld, t_end, _valid_genes = fetch_states(
        #     adata, init_states, init_cells, basis, layer, False,
        #     t_end
        # )
        n_states = self.init_states.shape[0]
        if n_states > 50:
            warnings.warn(f'the number of cell states with fate prediction is more than 50. You may want to '
                          f'lower the max number of cell states to draw via cell_states argument.')
        if cell_states is not None:
            if type(cell_states) is int:
                self.init_states = self.init_states[np.random.choice(range(n_states), min(n_states, cell_states))]
            elif type(cell_states) is list:
                self.init_states = self.init_states[cell_states]
            else:
                self.init_states = self.init_states

        # vf = lambda x: vector_field_function(x=x, vf_dict=VecFld)
        vf = vectorfield()
        vf.from_adata(adata, basis=basis)
        # Initialize velocity field and displace *functions*
        self.f = lambda x, _: vf.func(x)  # scale *
        self.displace = lambda x, dt: odeint(self.f, x, [0, dt])

        # Save bounds of plot
        X_data = adata.obsm['X_' + basis][:, :2] if dims is None else adata.obsm['X_' + basis][:, dims]
        m, M = np.min(X_data, 0), np.max(X_data, 0)
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        self.xlim = [m[0], M[0]]
        self.ylim = [m[1], M[1]]

        # Animation objects must create `fig` and `ax` attributes.
        if ax is None or ln is None:
            self.fig, self.ax = plt.subplots()
            self.ln, = self.ax.plot([], [], 'ro')
        else:
            self.ax = ax
            self.ln = ln

        self.ax.set_aspect("equal")
        self.color = color

    def init_background(self):
        self.ax = topography(self.adata, basis=self.basis, color=self.color, ax=self.ax, save_show_or_return='return')

        return self.ln,

    def update(self, frame):
        """Update locations of "particles" in flow on each frame frame."""
        print(frame)
        init_states = self.init_states
        time_vec = self.time_vec

        pts = [i.tolist() for i in init_states]

        if frame == 0:
            x, y = init_states.T

            self.ax.lines = []
            (self.ln,) = self.ax.plot(x, y, "ro", zorder=20)
            return self.ln,  # return line so that blit works properly
        else:
            pts = [
                self.displace(cur_pts, time_vec[frame])[1].tolist()
                for cur_pts in pts
            ]
            pts = np.asarray(pts)

        pts = np.asarray(pts)

        pts = remove_particles(pts, self.xlim, self.ylim)

        x, y = np.asarray(pts).transpose()

        self.ax.lines = []
        (self.ln,) = self.ax.plot(x, y, "ro", zorder=20)

        self.ax.set_title("current vector field time is: {:12.2f}".format(time_vec[frame] - time_vec[frame - 1]))
        # anim.event_source.interval = (time_vec[frame] - time_vec[frame - 1]) / 100

        return self.ln,  # return line so that blit works properly


def remove_particles(pts, xlim, ylim):
    if len(pts) == 0:
        return []
    outside_xlim = (pts[:, 0] < xlim[0]) | (pts[:, 0] > xlim[1])
    outside_ylim = (pts[:, 1] < ylim[0]) | (pts[:, 1] > ylim[1])
    keep = ~(outside_xlim | outside_ylim)
    return pts[keep]


def fate_animation(adata,
                   basis='umap',
                   dims=None,
                   n_steps=100,
                   cell_states=None,
                   color='ntr',
                   ax=None,
                   ln=None,
                   interval=100,
                   blit=True,
                   save_show_or_return='save'):
    """Animating cell fate commitment prediction via reconstructed vector field function.

    This class creates necessary components to produce an animation that describes the exact speed of a set of cells
    at each time point, its movement in gene expression and the long range trajectory predicted by the reconstructed
    vector field. Thus it provides intuitive visual understanding of the RNA velocity, speed, acceleration, and cell
    fate commitment in action.

    This function is originally inspired by https://tonysyu.github.io/animating-particles-in-a-flow.html.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that already went through the fate prediction.
        basis: `str` or None (default: `None`)
            The embedding data to use for predicting cell fate. If `basis` is either `umap` or `pca`, the reconstructed
            trajectory will be projected back to high dimensional space via the `inverse_transform` function.
            space.
        dims: `list` or `None` (default: `None')
            The dimensions of low embedding space where cells will be drawn and it should corresponds to the space
            fate prediction take place.
        n_steps: `int` (default: `100`)
            The number of times steps (frames) fate prediction will take.
        cell_states: `int`, `list` or `None` (default: `None`)
            The number of cells state that will be randomly selected (if `int`), the indices of the cells states (if
            `list`) or all cell states which fate prediction executed (if `None`)
        ax: `matplotlib.Axis` (optional, default `None`)
            The matplotlib axes object that will be used as background plot of the vector field animation.
        ln: `tuple` or None (default: `None`)
            An iterable of artists (for example, `matplotlib.lines.Line2D`) used to draw a clear frame.

        Returns
        -------
            Nothing but produce an animation that will be embedded to jupyter notebook or saved to disk.

        Examples 1
        ----------
        >>> from matplotlib import animation
        >>> progenitor = adata.obs_names[adata.obs.clusters == 'cluster_1']
        >>> fate_progenitor = progenitor
        >>> info_genes = adata.var_names[adata.var.use_for_velocity]
        >>> dyn.pd.fate(adata, basis='umap', init_cells=fate_progenitor, interpolation_num=100,  direction='forward',
        ...    inverse_transform=False, average=False, arclen_sampling=True)
        >>> dyn.pl.fate_animation(adata)
        """

    instance = StreamFuncAnim(adata=adata,
                              basis=basis,
                              dims=dims,
                              n_steps=n_steps,
                              cell_states=cell_states,
                              color=color,
                              ax=ax,
                              ln=ln,
                              )

    anim = animation.FuncAnimation(instance.fig, instance.update, init_func=instance.init_background,
                                   frames=np.arange(n_steps), interval=interval, blit=blit)
    if save_show_or_return == 'save':
        anim.save('fate_ani.gif', writer="imagemagick")  # save as gif file.
    elif save_show_or_return == 'show':
        from IPython.core.display import HTML
        HTML(anim.to_jshtml())  # embedding to jupyter notebook.
    else:
        anim
