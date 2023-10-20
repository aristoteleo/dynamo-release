import warnings
from typing import Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import matplotlib
import numpy as np
from anndata import AnnData
from scipy.integrate import odeint

from ..dynamo_logger import main_info, main_tqdm, main_warning
from ..plot.topography import topography, topography_3D
from ..vectorfield.scVectorField import SvcVectorField
from .utils import remove_particles


class BaseAnim:
    """Base class for animating cell fate commitment prediction via reconstructed vector field function.

    This class creates necessary components to produce an animation that describes the exact speed of a set of cells
    at each time point, its movement in gene expression and the long range trajectory predicted by the reconstructed
    vector field. Thus, it provides intuitive visual understanding of the RNA velocity, speed, acceleration, and cell
    fate commitment in action.
    """
    def __init__(
        self,
        adata: AnnData,
        basis: str = "umap",
        fp_basis: Union[str, None] = None,
        dims: Optional[list] = None,
        n_steps: int = 100,
        cell_states: Union[int, list, None] = None,
        color: str = "ntr",
        logspace: bool = False,
        max_time: Optional[float] = None,
    ):
        """Construct a class that can be used to animate cell fate commitment prediction via reconstructed vector field
        function.

        Args:
            adata: annData object that already went through the fate prediction.
            basis: the embedding data to use for predicting cell fate. If `basis` is either `umap` or `pca`, the
                reconstructed trajectory will be projected back to high dimensional space via the `inverse_transform`
                function space.
            fp_basis: the basis that will be used for identifying or retrieving fixed points. Note that if `fps_basis`
                is different from `basis`, the nearest cells of the fixed point from the `fps_basis` will be found and
                used to visualize the position of the fixed point on `basis` embedding.
            dims: the dimensions of low embedding space where cells will be drawn, and it should correspond to the space
                fate prediction take place.
            n_steps: the number of times steps (frames) fate prediction will take.
            cell_states: the number of cells state that will be randomly selected (if `int`), the indices of the cells
                states (if `list`) or all cell states which fate prediction executed (if `None`)
            color: the key of the data that will be used to color the embedding.
            logspace: `whether or to sample time points linearly on log space. If not, the sorted unique set of all-time
                points from all cell states' fate prediction will be used and then evenly sampled up to `n_steps` time
                points.
            max_time: the maximum time that will be used to scale the time vector.

        Returns:
            A class that contains .fig attribute and .update, .init_background that can be used to produce an animation
            of the prediction of cell fate commitment.
        """

        self.adata = adata
        self.basis = basis
        self.fp_basis = basis if fp_basis is None else fp_basis

        fate_key = "fate_" + basis
        if fate_key not in adata.uns_keys():
            raise Exception(
                f"You need to first perform fate prediction before animate the prediction, please run"
                f"dyn.pd.fate(adata, basis='{basis}' before running this function"
            )

        self.init_states = adata.uns[fate_key]["init_states"]
        # self.prediction = adata.uns['fate_umap']['prediction']
        self.t = adata.uns[fate_key]["t"]

        flat_list = np.unique([item for sublist in self.t for item in sublist])
        flat_list = np.hstack((0, flat_list))
        flat_list = np.sort(flat_list)

        self.logspace = logspace
        if self.logspace:
            self.time_vec = np.logspace(0, np.log10(max(flat_list) + 1), n_steps) - 1
        else:
            self.time_vec = flat_list[(np.linspace(0, len(flat_list) - 1, n_steps)).astype(int)]

        self.time_scaler = None if max_time is None else max_time / (self.time_vec[-1] - self.time_vec[-2])

        # init_states, VecFld, t_end, _valid_genes = fetch_states(
        #     adata, init_states, init_cells, basis, layer, False,
        #     t_end
        # )
        n_states = self.init_states.shape[0]
        if n_states > 50:
            main_warning(
                f"the number of cell states with fate prediction is more than 50. You may want to "
                f"lower the max number of cell states to draw via cell_states argument."
            )
        if cell_states is not None:
            if type(cell_states) is int:
                self.init_states = self.init_states[np.random.choice(range(n_states), min(n_states, cell_states))]
            elif type(cell_states) is list:
                self.init_states = self.init_states[cell_states]
            else:
                self.init_states = self.init_states

        # vf = lambda x: vector_field_function(x=x, vf_dict=VecFld)
        vf = SvcVectorField()
        vf.from_adata(adata, basis=basis)
        # Initialize velocity field and displace *functions*
        self.f = lambda x, _: vf.func(x)  # scale *
        self.displace = lambda x, dt: odeint(self.f, x, [0, dt])

        # Save bounds of plot
        X_data = adata.obsm["X_" + basis][:, :2] if dims is None else adata.obsm["X_" + basis][:, dims]
        m, M = np.min(X_data, 0), np.max(X_data, 0)
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        self.xlim = [m[0], M[0]]
        self.ylim = [m[1], M[1]]
        if X_data.shape[1] == 3:
            self.zlim = [m[2], M[2]]

        # self.ax.set_aspect("equal")
        self.color = color


class StreamFuncAnim(BaseAnim):
    """The class for animating cell fate commitment prediction with matplotlib.

    This function is originally inspired by https://tonysyu.github.io/animating-particles-in-a-flow.html and relies on
    animation module from matplotlib. Note that you may need to install `imagemagick` in order to properly show or save
    the animation. See for example, http://louistiao.me/posts/notebooks/save-matplotlib-animations-as-gifs/ for more
    details.

    Examples 1
        ----------
        >>> from matplotlib import animation
        >>> progenitor = adata.obs_names[adata.obs.clusters == 'cluster_1']
        >>> fate_progenitor = progenitor
        >>> info_genes = adata.var_names[adata.var.use_for_transition]
        >>> dyn.pd.fate(adata, basis='umap', init_cells=fate_progenitor, interpolation_num=100,  direction='forward',
        ...    inverse_transform=False, average=False)
        >>> instance = dyn.mv.StreamFuncAnim(adata=adata, fig=None, ax=None)
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
        >>> info_genes = adata.var_names[adata.var.use_for_transition]
        >>> dyn.pd.fate(adata, basis='umap', init_cells=fate_progenitor, interpolation_num=100,  direction='forward',
        ...    inverse_transform=False, average=False)
        >>> fig, ax = plt.subplots()
        >>> ax = dyn.pl.topography(adata_old, color='time', ax=ax, save_show_or_return='return', color_key_cmap='viridis')
        >>> ax.set_xlim(xlim)
        >>> ax.set_ylim(ylim)
        >>> instance = dyn.mv.StreamFuncAnim(adata=adata, fig=fig, ax=ax)
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
        >>> info_genes = adata.var_names[adata.var.use_for_transition]
        >>> dyn.pd.fate(adata, basis='umap', init_cells=fate_progenitor, interpolation_num=100,  direction='forward',
        ...    inverse_transform=False, average=False)
        >>> dyn.mv.animate_fates(adata)

                    See also:: :func:`animate_fates`
    """

    def __init__(
        self,
        adata: AnnData,
        basis: str = "umap",
        fp_basis: Union[str, None] = None,
        dims: Optional[list] = None,
        n_steps: int = 100,
        cell_states: Union[int, list, None] = None,
        color: str = "ntr",
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: matplotlib.axes.Axes = None,
        logspace: bool = False,
        max_time: Optional[float] = None,
    ):
        """Construct the StreamFuncAnim class.

        Args:
            adata: annData object that already went through the fate prediction.
            basis: the embedding data to use for predicting cell fate. If `basis` is either `umap` or `pca`, the
                reconstructed trajectory will be projected back to high dimensional space via the `inverse_transform`
                function space.
            fp_basis: the basis that will be used for identifying or retrieving fixed points. Note that if `fps_basis`
                is different from `basis`, the nearest cells of the fixed point from the `fps_basis` will be found and
                used to visualize the position of the fixed point on `basis` embedding.
            dims: the dimensions of low embedding space where cells will be drawn, and it should correspond to the space
                fate prediction take place.
            n_steps: the number of times steps (frames) fate prediction will take.
            cell_states: the number of cells state that will be randomly selected (if `int`), the indices of the cells
                states (if `list`) or all cell states which fate prediction executed (if `None`)
            color: the key of the data that will be used to color the embedding.
            fig: the figure that will contain both the background and animated components.
            ax: the matplotlib axes object that will be used as background plot of the vector field animation. If `ax`
                is None, `topography(adata, basis=basis, color=color, ax=ax, save_show_or_return='return')` will be used
                to create an axes.
            logspace: `whether or to sample time points linearly on log space. If not, the sorted unique set of all-time
                points from all cell states' fate prediction will be used and then evenly sampled up to `n_steps` time
                points.
            max_time: the maximum time that will be used to scale the time vector.

        Returns:
            A class that contains .fig attribute and .update, .init_background that can be used to produce an animation
            of the prediction of cell fate commitment.
        """

        import matplotlib.pyplot as plt

        super().__init__(
            adata=adata,
            basis=basis,
            fp_basis=fp_basis,
            dims=dims,
            n_steps=n_steps,
            cell_states=cell_states,
            color=color,
            logspace=logspace,
            max_time=max_time,
        )

        # Animation objects must create `fig` and `ax` attributes.
        if ax is None or fig is None:
            self.fig, self.ax = plt.subplots()
            self.ax = topography(
                self.adata,
                basis=self.basis,
                fps_basis=self.fp_basis,
                color=self.color,
                ax=self.ax,
                save_show_or_return="return",
            )
        else:
            self.fig = fig
            self.ax = ax

        (self.ln,) = self.ax.plot([], [], "ro", zs=[]) if dims is not None and len(dims) == 3 else self.ax.plot([], [], "ro")

    def init_background(self):
        """Initialize background of the animation."""
        return (self.ln,)

    def update(self, frame):
        """Update locations of "particles" in flow on each frame."""
        init_states = self.init_states
        time_vec = self.time_vec

        pts = [i.tolist() for i in init_states]

        if frame == 0:
            x, y = init_states.T

            for line in self.ax.get_lines():
                line.remove()

            (self.ln,) = self.ax.plot(x, y, "ro", zorder=20)
            return (self.ln,)  # return line so that blit works properly
        else:
            pts = [self.displace(cur_pts, time_vec[frame])[1].tolist() for cur_pts in pts]
            pts = np.asarray(pts)

        pts = np.asarray(pts)

        pts = remove_particles(pts, self.xlim, self.ylim)

        x, y = np.asarray(pts).transpose()

        for line in self.ax.get_lines():
            line.remove()

        (self.ln,) = self.ax.plot(x, y, "ro", zorder=20)

        if self.time_scaler is not None:
            vf_time = (time_vec[frame] - time_vec[frame - 1]) * self.time_scaler
            self.ax.set_title("current vector field time is: {:12.2f}".format(vf_time))

        # anim.event_source.interval = (time_vec[frame] - time_vec[frame - 1]) / 100

        return (self.ln,)  # return line so that blit works properly


class StreamFuncAnim3D(StreamFuncAnim):
    """The class of 3D animation instance for matplotlib FuncAnimation function."""
    def update(self, frame):
        """The function to call at each frame. Update the position of the line object in the animation."""
        init_states = self.init_states
        time_vec = self.time_vec

        pts = [i.tolist() for i in init_states]

        if frame == 0:
            x, y, z = init_states.T

            for line in self.ax.get_lines():
                line.remove()

            (self.ln,) = self.ax.plot(x, y, z, "ro", zorder=20)
            return (self.ln,)  # return line so that blit works properly
        else:
            pts = [self.displace(cur_pts, time_vec[frame])[1].tolist() for cur_pts in pts]
            pts = np.asarray(pts)

        pts = np.asarray(pts)

        pts = remove_particles(pts, self.xlim, self.ylim, self.zlim)

        x, y, z = np.asarray(pts).transpose()

        for line in self.ax.get_lines():
            line.remove()

        (self.ln,) = self.ax.plot(x, y, z, "ro", zorder=20)

        if self.time_scaler is not None:
            vf_time = (time_vec[frame] - time_vec[frame - 1]) * self.time_scaler
            self.ax.set_title("current vector field time is: {:12.2f}".format(vf_time))

        # anim.event_source.interval = (time_vec[frame] - time_vec[frame - 1]) / 100

        return (self.ln,)  # return line so that blit works properly


def animate_fates(
    adata,
    basis="umap",
    dims=None,
    n_steps=100,
    cell_states=None,
    color="ntr",
    fig=None,
    ax=None,
    logspace=False,
    max_time=None,
    interval=100,
    blit=True,
    save_show_or_return="show",
    save_kwargs={},
    **kwargs,
):
    """Animating cell fate commitment prediction via reconstructed vector field function.

    This class creates necessary components to produce an animation that describes the exact speed of a set of cells
    at each time point, its movement in gene expression and the long range trajectory predicted by the reconstructed
    vector field. Thus, it provides intuitive visual understanding of the RNA velocity, speed, acceleration, and cell
    fate commitment in action.

    This function is originally inspired by https://tonysyu.github.io/animating-particles-in-a-flow.html and relies on
    animation module from matplotlib. Note that you may need to install `imagemagick` in order to properly show or save
    the animation. See for example, http://louistiao.me/posts/notebooks/save-matplotlib-animations-as-gifs/ for more
    details.

    Args:
        adata: annData object that already went through the fate prediction.
        basis: the embedding data to use for predicting cell fate. If `basis` is either `umap` or `pca`, the
            reconstructed trajectory will be projected back to high dimensional space via the `inverse_transform`
            function space.
        dims: the dimensions of low embedding space where cells will be drawn, and it should correspond to the space
            fate prediction take place.
        n_steps: the number of times steps (frames) fate prediction will take.
        cell_states: the number of cells state that will be randomly selected (if `int`), the indices of the cells
            states (if `list`) or all cell states which fate prediction executed (if `None`)
        color: the key of the data that will be used to color the embedding.
        fig: the figure that will contain both the background and animated components.
        ax: the matplotlib axes object that will be used as background plot of the vector field animation. If `ax`
            is None, `topography(adata, basis=basis, color=color, ax=ax, save_show_or_return='return')` will be used
            to create an axes.
        logspace: `whether or to sample time points linearly on log space. If not, the sorted unique set of all-time
            points from all cell states' fate prediction will be used and then evenly sampled up to `n_steps` time
            points.
        max_time: the maximum time that will be used to scale the time vector.
        interval: delay between frames in milliseconds.
        blit: whether blitting is used to optimize drawing. Note: when using blitting, any animated artists will be drawn
            according to their zorder; however, they will be drawn on top of any previous artists, regardless of their
            zorder.
        save_show_or_return: whether to save, show or return the figure. By default, a gif will be used.
        save_kwargs: a dictionary that will  be passed to the anim.save. By default, it is an empty dictionary and the
            save_fig function will use the {"filename": 'fate_ani.gif', "writer": "imagemagick"} as its parameters.
            Otherwise, you can provide a dictionary that properly modify those keys according to your needs. see
            https://matplotlib.org/api/_as_gen/matplotlib.animation.Animation.save.html for more details.
        kwargs: additional arguments passed to animation.FuncAnimation.

    Returns:
        Nothing but produce an animation that will be embedded to jupyter notebook or saved to disk.

    Examples 1
    ----------
    >>> from matplotlib import animation
    >>> progenitor = adata.obs_names[adata.obs.clusters == 'cluster_1']
    >>> fate_progenitor = progenitor
    >>> info_genes = adata.var_names[adata.var.use_for_transition]
    >>> dyn.pd.fate(adata, basis='umap', init_cells=fate_progenitor, interpolation_num=100,  direction='forward',
    ...    inverse_transform=False, average=False)
    >>> dyn.mv.animate_fates(adata)

        See also:: :func:`StreamFuncAnim`
    """

    from matplotlib import animation

    instance = StreamFuncAnim(
        adata=adata,
        basis=basis,
        dims=dims,
        n_steps=n_steps,
        cell_states=cell_states,
        color=color,
        fig=fig,
        ax=ax,
        logspace=logspace,
        max_time=max_time,
    )

    anim = animation.FuncAnimation(
        instance.fig,
        instance.update,
        init_func=instance.init_background,
        frames=np.arange(n_steps),
        interval=interval,
        blit=blit,
        **kwargs,
    )
    if save_show_or_return == "save":
        save_kwargs_ = {"filename": "fate_ani.gif", "writer": "imagemagick"}
        save_kwargs_.update(save_kwargs)
        anim.save(**save_kwargs_)  # save as gif file.
    elif save_show_or_return == "show":
        from IPython.core.display import HTML

        HTML(anim.to_jshtml())  # embedding to jupyter notebook.
    else:
        anim


class PyvistaAnim(BaseAnim):
    """The class for animating cell fate commitment prediction with pyvista."""
    def __init__(
        self,
        adata: AnnData,
        basis: str = "umap",
        fp_basis: Union[str, None] = None,
        dims: Optional[list] = None,
        n_steps: int = 15,
        cell_states: Union[int, list, None] = None,
        color: str = "ntr",
        point_size: float = 15,
        pl=None,
        logspace: bool = False,
        max_time: Optional[float] = None,
        filename: str = "fate_animation.gif",
    ):
        """Construct the PyvistaAnim class.

        Args:
            adata: annData object that already went through the fate prediction.
            basis: the embedding data to use for predicting cell fate. If `basis` is either `umap` or `pca`, the
                reconstructed trajectory will be projected back to high dimensional space via the `inverse_transform`
                function space.
            fp_basis: the basis that will be used for identifying or retrieving fixed points. Note that if `fps_basis`
                is different from `basis`, the nearest cells of the fixed point from the `fps_basis` will be found and
                used to visualize the position of the fixed point on `basis` embedding.
            dims: the dimensions of low embedding space where cells will be drawn, and it should correspond to the space
                fate prediction take place.
            n_steps: the number of times steps (frames) fate prediction will take.
            cell_states: the number of cells state that will be randomly selected (if `int`), the indices of the cells
                states (if `list`) or all cell states which fate prediction executed (if `None`)
            color: the key of the data that will be used to color the embedding.
            point_size: the size of the points that will be used to draw the cells.
            pl: the pyvista plotter object that will be used to draw the cells. If `pl` is None, `topography_3D(adata,
                basis=basis, fps_basis=fp_basis, color=color, ax=pl, save_show_or_return='return')` will be used to
                create a plotter.
            logspace: `whether or to sample time points linearly on log space. If not, the sorted unique set of all-time
                points from all cell states' fate prediction will be used and then evenly sampled up to `n_steps` time
                points.
            max_time: the maximum time that will be used to scale the time vector.
            filename: the name of the gif file that will be saved to disk.

        Returns:
            A class that contains .animate, that can be used to produce a gif of the prediction of cell fate
            commitment.
        """
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("Please install pyvista first.")

        super().__init__(
            adata=adata,
            basis=basis,
            fp_basis=fp_basis,
            dims=dims,
            n_steps=n_steps,
            cell_states=cell_states,
            color=color,
            logspace=logspace,
            max_time=max_time,
        )

        self.filename = filename

        if pl is None:
            self.pl = topography_3D(
                self.adata,
                basis=self.basis,
                fps_basis=self.fp_basis,
                color=self.color,
                ax=self.pl,
                save_show_or_return="return",
            )
        else:
            self.pl = pl

        self.n_steps = n_steps
        self.point_size = point_size

    def animate(self):
        """Animate the cell fate commitment prediction."""
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("Please install pyvista first.")

        pts = [i.tolist() for i in self.init_states]

        self.pl.open_gif(self.filename)

        pts = [self.displace(cur_pts, self.time_vec[0])[1].tolist() for cur_pts in pts]
        pts = np.asarray(pts)
        pts = remove_particles(pts, self.xlim, self.ylim, self.zlim)

        mesh = pv.PolyData(pts)
        self.pl.add_mesh(mesh, color="red", render_points_as_spheres=True, point_size=self.point_size)

        for frame in range(1, self.n_steps):
            pts = [self.displace(cur_pts, self.time_vec[frame])[1].tolist() for cur_pts in pts]
            pts = np.asarray(pts)
            # pts = remove_particles(pts, self.xlim, self.ylim, self.zlim)
            mesh.points = pv.PolyData(pts).points

            self.pl.write_frame()

        self.pl.close()


class PlotlyAnim(BaseAnim):
    """The class for animating cell fate commitment prediction with plotly."""
    def __init__(
        self,
        adata: AnnData,
        basis: str = "umap",
        fp_basis: Union[str, None] = None,
        dims: Optional[list] = None,
        n_steps: int = 15,
        cell_states: Union[int, list, None] = None,
        color: str = "ntr",
        pl=None,
        logspace: bool = False,
        max_time: Optional[float] = None,
    ):
        """Construct the PlotlyAnim class.

        Args:
            adata: annData object that already went through the fate prediction.
            basis: the embedding data to use for predicting cell fate. If `basis` is either `umap` or `pca`, the
                reconstructed trajectory will be projected back to high dimensional space via the `inverse_transform`
                function space.
            fp_basis: the basis that will be used for identifying or retrieving fixed points. Note that if `fps_basis`
                is different from `basis`, the nearest cells of the fixed point from the `fps_basis` will be found and
                used to visualize the position of the fixed point on `basis` embedding.
            dims: the dimensions of low embedding space where cells will be drawn, and it should correspond to the space
                fate prediction take place.
            n_steps: the number of times steps (frames) fate prediction will take.
            cell_states: the number of cells state that will be randomly selected (if `int`), the indices of the cells
                states (if `list`) or all cell states which fate prediction executed (if `None`)
            color: the key of the data that will be used to color the embedding.
            pl: the plotly figure object that will be used to draw the cells. If `pl` is None, `topography_3D(adata,
                basis=basis, fps_basis=fp_basis, color=color, ax=pl, save_show_or_return='return')` will be used to
                create a plotter.
            logspace: `whether or to sample time points linearly on log space. If not, the sorted unique set of all-time
                points from all cell states' fate prediction will be used and then evenly sampled up to `n_steps` time
                points.
            max_time: the maximum time that will be used to scale the time vector.

        Returns:
            A class that contains .animate, that can be used to produce a gif of the prediction of cell fate
            commitment.
        """
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("Please install pyvista first.")

        super().__init__(
            adata=adata,
            basis=basis,
            fp_basis=fp_basis,
            dims=dims,
            n_steps=n_steps,
            cell_states=cell_states,
            color=color,
            logspace=logspace,
            max_time=max_time,
        )

        if pl is None:
            self.pl = topography_3D(
                self.adata,
                basis=self.basis,
                fps_basis=self.fp_basis,
                color=self.color,
                plot_method="plotly",
                ax=self.pl,
                save_show_or_return="return",
            )
        else:
            self.pl = pl

        self.n_steps = n_steps
        self.pts_history = []

    def calculate_pts_history(self):
        """Calculate the history of the cell states."""
        pts = [i.tolist() for i in self.init_states]

        self.pts_history.append(pts)

        for frame in range(0, self.n_steps):
            pts = [self.displace(cur_pts, self.time_vec[frame])[1].tolist() for cur_pts in pts]
            pts = np.asarray(pts)
            pts = remove_particles(pts, self.xlim, self.ylim, self.zlim)
            self.pts_history.append(np.asarray(pts))

    def animate(self):
        """Animate the cell fate commitment prediction."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Please install plotly first.")

        if len(self.pts_history) == 0:
            self.calculate_pts_history()

        fig = go.Figure(
            data=self.pl,
            layout=go.Layout(title="Moving Frenet Frame Along a Planar Curve",
                             updatemenus=[dict(type="buttons",
                                               buttons=[dict(label="Play",
                                                             method="animate",
                                                             args=[None])])]),
            frames=[
                go.Frame(
                    data=[
                        go.Scatter3d(
                            x=self.pts_history[k][:, 0],
                            y=self.pts_history[k][:, 1],
                            z=self.pts_history[k][:, 2],
                            mode="markers",
                            marker=dict(
                                color="red",
                                size=20,
                            ),
                        )
                    ]
                ) for k in range(1, self.n_steps)
            ]
        )

        fig.show()
