from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

# from scipy.sparse import issparse
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.sparse import spmatrix
from sklearn.preprocessing import normalize

from ..dynamo_logger import main_debug, main_info
from ..tools.cell_velocities import cell_velocities
from ..tools.dimension_reduction import reduceDimension
from ..tools.Markov import (
    grid_velocity_filter,
    prepare_velocity_grid_data,
    velocity_on_grid,
)
from ..tools.utils import update_dict
from ..vectorfield.topography import VectorField
from ..vectorfield.utils import vecfld_from_adata
from .scatters import docstrings, scatters, scatters_interactive
from .utils import (
    _get_adata_color_vec,
    default_quiver_args,
    quiver_autoscaler,
    retrieve_plot_save_path,
    save_show_ret,
    save_plotly_figure,
    save_pyvista_plotter,
    set_arrow_alpha,
    set_stream_line_alpha,
)

docstrings.delete_params("scatters.parameters", "show_legend", "kwargs", "save_kwargs")

# import scipy as sc


# from licpy.lic import runlic

# moran'I on the transition genes, etc.

# cellranger data, velocyto, comparison and phase diagram


def cell_wise_vectors_3d(
    adata: AnnData,
    basis: str = "umap",
    x: int = 0,
    y: int = 1,
    z: int = 2,
    ekey: Optional[str] = None,
    vkey: str = "velocity_S",
    X: Union[np.ndarray, spmatrix] = None,
    V: Union[np.ndarray, spmatrix] = None,
    color: Union[str, List[str]] = None,
    layer: str = "X",
    plot_method: Literal["pv", "matplotlib"] = "pv",
    background: Optional[str] = "white",
    ncols: int = 4,
    figsize: Tuple[float] = (6, 4),
    ax: Optional[Axes] = None,
    inverse: bool = False,
    cell_inds: str = "all",
    vector: str = "velocity",
    save_show_or_return: str = "show",
    save_kwargs: Dict[str, Any] = {},
    quiver_3d_kwargs: Dict[str, Any] = {
         "linewidth": 1,
        "edgecolors": "white",
        "alpha": 1,
        "length": 8,
        "arrow_length_ratio": 1,
        "norm": cm.colors.Normalize(),
        "cmap": cm.PRGn,
    },
    grid_color: Optional[str] = None,
    axis_label_prefix: Optional[str] = None,
    axis_labels: Optional[List[str]] = None,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
    alpha: Optional[float] = None,
    show_magnitude: bool = True,
    titles: Optional[List[str]] = None,
    highlights: Optional[list] = None,
    labels: Optional[list] = None,
    values: Optional[list] = None,
    theme: Optional[
        Literal[
            "blue",
            "red",
            "green",
            "inferno",
            "fire",
            "viridis",
            "darkblue",
            "darkred",
            "darkgreen",
        ]
    ] = None,
    plotly_color: str = "Reds",
    cmap: Optional[str] = None,
    color_key: Union[Dict[str, str], List[str], None] = None,
    color_key_cmap: Optional[str] = None,
    pointsize: Optional[float] = None,
    use_smoothed: bool = True,
    sort: Literal["raw", "abs", "neg"] = "raw",
    aggregate: Optional[str] = None,
    show_arrowed_spines: bool = False,
    frontier: bool = False,
    s_kwargs_dict: Dict[str, Any] = {},
    **cell_wise_kwargs,
) -> np.ndarray:
    """Plot the velocity or acceleration vector of each cell.

    Args:
        adata: an AnnData object.
        basis: the reduced dimension stored in adata.obsm. The specific basis key will be constructed in the following
            priority if exits: 1) specific layer input +  basis 2) X_ + basis 3) basis. E.g. if basis is PCA, `scatters`
            is going to look for 1) if specific layer is spliced, `spliced_pca` 2) `X_pca` (dynamo convention) 3) `pca`.
            Defaults to "umap".
        x: the column index of the low dimensional embedding for the x-axis. Defaults to 0.
        y: the column index of the low dimensional embedding for the y-axis. Defaults to 1.
        z: the column index of the low dimensional embedding for the z-axis. Defaults to 2.
        ekey: the expression key. Defaults to None.
        vkey: the velocity key. Defaults to "velocity_S".
        X: the expression array. If None, the array would be determined by `ekey` provided. Defaults to None.
        V: the velocity array. If None, the array would be determined by `vkey` provided. Defaults to None.
        color: any column names or gene expression, etc. that will be used for coloring cells. Defaults to "ntr".
        layer: the layer of data to use for the scatter plot. Defaults to "X".
        plot_method: the method to plot 3D vectors. Options include `pv` (pyvista) and `matplotlib`.
        background: the background color of the figure. Defaults to "white".
        ncols: the number of sub-plot columns. Defaults to 4.
        figsize: the size of each sub-plot panel. Defaults to (6, 4).
        ax: the axes to plot on. Only work when there is one graph to plot. If None, new axes would be created. Defaults
            to None.
        inverse: whether to inverse the direction of the velocity vectors. Defaults to False.
        cell_inds: the cell index that will be chosen to draw velocity vectors. Can be a list of integers (cell indices)
            or str (Cell names). Defaults to "all".
        vector: which vector type will be used for plotting, one of {'velocity', 'acceleration'} or either velocity
            field or acceleration field will be plotted. Defaults to "velocity".
        save_show_or_return: whether to save, show or return the generated figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'cell_wise_velocity', "dpi": None,
            "ext": 'pdf', "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can
            provide a dictionary that properly modify those keys according to your needs. Defaults to {}.
        quiver_3d_kwargs: any other kwargs to be passed to `pyplot.quiver`. Defaults to { "zorder": 3, "length": 2,
            "linewidth": 5, "arrow_length_ratio": 5, "norm": cm.colors.Normalize(), "cmap": cm.PRGn, }.
        grid_color: the color of the grid lines. Defaults to None.
        axis_label_prefix: the prefix of the axis labels. Defaults to None.
        axis_labels: the axis labels. Defaults to None.
        elev: the elevation angle in degrees rotates the camera above the plane pierced by the vertical axis, with a
            positive angle corresponding to a location above that plane. Defaults to None.
        azim: the azimuthal angle in degrees rotates the camera about the vertical axis, with a positive angle
            corresponding to a right-handed rotation. Defaults to None.
        alpha: the transparency of the colors. Defaults to None.
        show_magnitude: whether to show original values or normalize the data. Defaults to False.
        titles: the titles of the subplots. Defaults to None.
        highlights: the color group that will be highlighted. If highligts is a list of lists, each list is relate to
            each color element. Defaults to None.
        labels: an array of labels (assumed integer or categorical), one for each data sample. This will be used for
            coloring the points in the plot according to their label. Note that this option is mutually exclusive to the
            `values` option. Defaults to None.
        values: an array of values (assumed float or continuous), one for each sample. This will be used for coloring
            the points in the plot according to a colorscale associated to the total range of values. Note that this
            option is mutually exclusive to the `labels` option. Defaults to None.
        theme: A color theme to use for plotting. A small set of predefined themes are provided which have relatively
            good aesthetics. Available themes are: {'blue', 'red', 'green', 'inferno', 'fire', 'viridis', 'darkblue',
            'darkred', 'darkgreen'}. Defaults to None.
        plotly_color: the color of the Plotly Cone plot. It must be an array containing arrays mapping a normalized
            value to a rgb, rgba, hex, hsl, hsv, or named color string.
        cmap: The name of a matplotlib colormap to use for coloring or shading points. If no labels or values are passed
            this will be used for shading points according to density (largely only of relevance for very large
            datasets). If values are passed this will be used for shading according the value. Note that if theme is
            passed then this value will be overridden by the corresponding option of the theme. Defaults to None.
        color_key: the method to assign colors to categoricals. This can either be an explicit dict mapping labels to
            colors (as strings of form '#RRGGBB'), or an array like object providing one color for each distinct
            category being provided in `labels`. Either way this mapping will be used to color points according to the
            label. Note that if theme is passed then this value will be overridden by the corresponding option of the
            theme. Defaults to None.
        color_key_cmap: the name of a matplotlib colormap to use for categorical coloring. If an explicit `color_key` is
            not given a color mapping for categories can be generated from the label list and selecting a matching list
            of colors from the given colormap. Note that if theme is passed then this value will be overridden by the
            corresponding option of the theme. Defaults to None.
        pointsize: the scale of the point size. Actual point cell size is calculated as
            `500.0 / np.sqrt(adata.shape[0]) * pointsize`. Defaults to None.
        use_smoothed: whether to use smoothed values (i.e. M_s / M_u instead of spliced / unspliced, etc.). Defaults to
            True.
        sort: the method to reorder data so that high values points will be on top of background points. Can be one of
            {'raw', 'abs', 'neg'}, i.e. sorted by raw data, sort by absolute values or sort by negative values. Defaults
            to "raw".
        aggregate: the column in adata.obs that will be used to aggregate data points. Defaults to None.
        show_arrowed_spines: whether to show a pair of arrowed spines representing the basis of the scatter is currently
            using. Defaults to False.
        frontier: whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to
            show area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib
            tips & tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from
            scEU-seq paper: https://science.sciencemag.org/content/367/6482/1151. Defaults to False.
        s_kwargs_dict: any other kwargs that will be passed to `dynamo.pl.scatters`. Defaults to {}.

    Raises:
        ValueError: invalid `x`, `y`, or `z`.

    Returns:
        None will be returned by default. If `save_show_or_return` is set to 'return', an array of axes of the subplots
        would be returned.
    """

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    def add_axis_label(ax, labels):
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])

    projection_dim_indexer = [x, y, z]

    # ensure axis_label prefix is not None
    if ekey is not None and axis_label_prefix is None:
        axis_label_prefix = ekey
    elif axis_label_prefix is None:
        axis_label_prefix = "dim"

    # ensure axis_labels is not None
    if axis_labels is None:
        axis_labels = [axis_label_prefix + "_" + str(index) for index in projection_dim_indexer]

    if type(color) is str:
        color = [color]

    if titles is None:
        titles = color

    assert len(color) == len(titles), "#titles does not match #color."

    if grid_color:
        plt.rcParams["grid.color"] = grid_color

    if alpha:
        quiver_3d_kwargs = dict(quiver_3d_kwargs)
        quiver_3d_kwargs["alpha"] = alpha

    if X is not None and V is not None:
        X = X[:, [x, y, z]]
        V = V[:, [x, y, z]]

    elif type(x) == str and type(y) == str and type(z) == str:
        if len(adata.var_names[adata.var.use_for_dynamics].intersection([x, y, z])) != 3:
            raise ValueError(
                "If you want to plot the vector flow of three genes, please make sure those three genes "
                "belongs to dynamics genes or .var.use_for_dynamics is True."
            )
        X = adata[:, projection_dim_indexer].layers[ekey].A
        V = adata[:, projection_dim_indexer].layers[vkey].A
        layer = ekey
    else:
        if ("X_" + basis in adata.obsm.keys()) and (vector + "_" + basis in adata.obsm.keys()):
            X = adata.obsm["X_" + basis][:, projection_dim_indexer]
            V = adata.obsm[vector + "_" + basis][:, projection_dim_indexer]
        else:
            if "X_" + basis not in adata.obsm.keys():
                layer, basis = basis.split("_")
                reduceDimension(adata, layer=layer, reduction_method=basis)
            if "kmc" not in adata.uns_keys():
                cell_velocities(adata, vkey="velocity_S", basis=basis)
                X = adata.obsm["X_" + basis][:, projection_dim_indexer]
                V = adata.obsm[vector + "_" + basis][:, projection_dim_indexer]
            else:
                kmc = adata.uns["kmc"]
                X = adata.obsm["X_" + basis][:, projection_dim_indexer]
                V = kmc.compute_density_corrected_drift(X, kmc.Idx, normalize_vector=True)
                adata.obsm[vector + "_" + basis] = V

    X, V = X.copy(), V.copy()
    if not show_magnitude:
        X = normalize(X, axis=0, norm="max")
        V = normalize(V, axis=0, norm="l2")
        V = normalize(V, axis=1, norm="l2")

    V /= 3 * quiver_autoscaler(X, V)
    if inverse:
        V = -V

    main_info("X shape: " + str(X.shape) + " V shape: " + str(V.shape))
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "z": X[:, 2], "u": V[:, 0], "v": V[:, 1], "w": V[:, 2]})

    if cell_inds == "all":
        ix_choice = np.arange(adata.shape[0])
    elif cell_inds == "random":
        ix_choice = np.random.choice(np.range(adata.shape[0]), size=1000, replace=False)
    elif type(cell_inds) is int:
        ix_choice = np.random.choice(np.range(adata.shape[0]), size=cell_inds, replace=False)
    elif type(cell_inds) is list:
        if type(cell_inds[0]) is str:
            cell_inds = [adata.obs_names.to_list().index(i) for i in cell_inds]
        ix_choice = cell_inds
    else:
        ix_choice = np.arange(adata.shape[0])

    df = df.iloc[ix_choice, :]

    if background is None:
        _background = rcParams.get("figure.facecolor")
        background = to_hex(_background) if type(_background) is tuple else _background

    # single axis output
    x0, x1, x2 = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    v0, v1, v2 = df.iloc[:, 3], df.iloc[:, 4], df.iloc[:, 5]
    nrows = len(color) // ncols
    if nrows * ncols < len(color):
        nrows += 1
    ncols = min(ncols, len(color))

    if plot_method == "pv":
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("Please install pyvista first.")

        pl, colors_list = scatters_interactive(
            adata=adata,
            basis=basis,
            x=x,
            y=y,
            z=z,
            color=color,
            layer=layer,
            labels=labels,
            values=values,
            cmap=cmap,
            theme=theme,
            background=background,
            color_key=color_key,
            color_key_cmap=color_key_cmap,
            use_smoothed=use_smoothed,
            save_show_or_return="return",
            render_points_as_spheres=True,
        )

        point_cloud = pv.PolyData(np.column_stack((x0.values, x1.values, x2.values)))
        point_cloud['vectors'] = np.column_stack((v0.values, v1.values, v2.values))

        r, c = pl.shape[0], pl.shape[1]
        subplot_indices = [[i, j] for i in range(r) for j in range(c)]
        cur_subplot = 0

        for i in range(len(color)):
            point_cloud.point_data["colors"] = np.stack(colors_list[i])

            arrows = point_cloud.glyph(
                orient='vectors',
                factor=3.5,
            )

            if r * c != 1:
                pl.subplot(subplot_indices[cur_subplot][0], subplot_indices[cur_subplot][1])
                cur_subplot += 1

            pl.add_mesh(arrows, scalars="colors", preference='point', rgb=True)

        return save_pyvista_plotter(
            pl=pl,
            save_show_or_return=save_show_or_return,
            save_kwargs=save_kwargs,
        )

    elif plot_method == "plotly":
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Please install plotly first.")

        pl, colors_list = scatters_interactive(
            adata=adata,
            basis=basis,
            x=x,
            y=y,
            z=z,
            color=color,
            layer=layer,
            plot_method="plotly",
            labels=labels,
            values=values,
            cmap=cmap,
            theme=theme,
            background=background,
            color_key=color_key,
            color_key_cmap=color_key_cmap,
            use_smoothed=use_smoothed,
            save_show_or_return="return",
            opacity=0.5,
        )

        r, c = pl._get_subplot_rows_columns()
        subplot_indices = [[i, j] for i in range(list(r)[-1]) for j in range(list(c)[-1])]
        cur_subplot = 0

        for i in range(len(color)):
            # colors = [[index, "rgb({},{},{})".format(int(row[0] * 255), int(row[1] * 255), int(row[2] * 255))] for index, row in enumerate(colors_list[i])]

            pl.add_trace(
                go.Cone(
                    x=x0.values,
                    y=x1.values,
                    z=x2.values,
                    u=v0.values,
                    v=v1.values,
                    w=v2.values,
                    colorscale=plotly_color,
                    # colorscale=colors,
                    sizemode="absolute",
                    sizeref=1,
                ),
                row=subplot_indices[cur_subplot][0] + 1, col=subplot_indices[cur_subplot][1] + 1,
            )  # TODO: implement customized color for individual cone

            cur_subplot += 1

        return save_plotly_figure(
            pl=pl,
            save_show_or_return=save_show_or_return,
            save_kwargs=save_kwargs,
        )

    else:
        axes_list, color_list, _ = scatters(
            adata=adata,
            basis=basis,
            x=x,
            y=y,
            z=z,
            color=color,
            layer=layer,
            highlights=highlights,
            labels=labels,
            values=values,
            theme=theme,
            cmap=cmap,
            color_key=color_key,
            color_key_cmap=color_key_cmap,
            background=background,
            ncols=ncols,
            pointsize=pointsize,
            figsize=figsize,
            show_legend=None,
            use_smoothed=use_smoothed,
            aggregate=aggregate,
            show_arrowed_spines=show_arrowed_spines,
            ax=ax,
            sort=sort,
            save_show_or_return="return",
            frontier=frontier,
            projection="3d",
            **s_kwargs_dict,
            return_all=True,
        )

        if type(axes_list) != list:
            axes_list = [axes_list]
            color_list = [color_list]

        for i in range(len(color)):
            ax = axes_list[i]
            ax.set_title(color[i])
            cmap_3d = [element for element in color_list[i]] + [element for element in color_list[i] for _ in range(2)]
            main_debug("color vec len: " + str(len(cmap_3d)))
            ax.view_init(elev=elev, azim=azim)
            ax.quiver(
                x0,
                x1,
                x2,
                v0,
                v1,
                v2,
                color=cmap_3d,
                # facecolors=color_vec,
                **quiver_3d_kwargs,
            )
            ax.set_title(titles[i])
            ax.set_facecolor(background)
            add_axis_label(ax, axis_labels)

        return save_show_ret("cell_wise_vectors_3d", save_show_or_return, save_kwargs, axes_list, tight=False)


def grid_vectors_3d():
    pass


# def velocity(adata, type) # type can be either one of the three, cellwise, velocity on grid, streamline plot.
# 	"""
#
# 	"""
#


def plot_LIC_gray(tex):
    """GET_P estimates the posterior probability and part of the energy.

    Arguments
    ---------
        Y: 'np.ndarray'
            Original data.
        V: 'np.ndarray'
            Original data.
        sigma2: 'float'
            sigma2 is defined as sum(sum((Y - V)**2)) / (N * D)
        gamma: 'float'
            Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
        a: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of
            outlier's variation space is a.

    Returns
    -------
    P: 'np.ndarray'
        Posterior probability, related to equation 27.
    E: `np.ndarray'
        Energy, related to equation 26.

    """
    import matplotlib.pyplot as plt

    tex = tex[:, ::-1]
    tex = tex.T

    M, N = tex.shape
    texture = np.empty((M, N, 4), np.float32)
    texture[:, :, 0] = tex
    texture[:, :, 1] = tex
    texture[:, :, 2] = tex
    texture[:, :, 3] = 1

    #     texture = scipy.ndimage.rotate(texture,-90)
    plt.figure()
    plt.imshow(texture)


def line_integral_conv(
    adata: AnnData,
    basis: str = "umap",
    U_grid: Optional[np.ndarray] = None,
    V_grid: Optional[np.ndarray] = None,
    xy_grid_nums: Union[Tuple[int], List[int]] = [50, 50],
    method: Literal["yt", "lic"] = "yt",
    cmap: str = "viridis",
    normalize: bool = False,
    density: float = 1,
    lim: Tuple[float, float] = (0, 1),
    const_alpha: bool = False,
    kernellen: float = 100,
    V_threshold: Optional[float] = None,
    vector: str = "velocity",
    file: str = "vectorfield_LIC",
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
    g_kwargs_dict: Dict[str, Any] = {},
):
    """Visualize vector field with quiver, streamline and line integral convolution (LIC), using velocity estimates on a
     grid from the associated data. A white noise background will be used for texture as default. Adjust the bounds of
     lim in the range of [0, 1] which applies upper and lower bounds to the values of line integral convolution and
     enhance the visibility of plots. When const_alpha=False, alpha will be weighted spatially by the values of line
     integral convolution; otherwise a constant value of the given alpha is used.

    Args:
        adata: an AnnData object that contains U_grid and V_grid data.
        basis: the dimension reduction method to use. Defaults to "umap".
        U_grid: original velocity on the first dimension of a 2 d grid. Defaults to None.
        V_grid: original velocity on the second dimension of a 2 d grid. Defaults to None.
        xy_grid_nums: the number of grids in either x or y axis. The number of grids has to be the same on both
            dimensions. Defaults to [50, 50].
        method: the method to visualize the data. Defaults to "yt".
        cmap: the colormap used to plot the figure. Defaults to "viridis".
        normalize: whether to normalize the original data. Defaults to False.
        density: density of the streamlines. Defaults to 1.
        lim: the value of line integral convolution will be clipped to the range of lim, which applies upper and lower
            bounds to the values of line integral convolution and enhance the visibility of plots. Each element should
            be in the range of [0,1].. Defaults to (0, 1).
        const_alpha: whether to prevent the alpha from being weighted spatially by the values of line integral
            convolution; otherwise a constant value of the given alpha is used. Defaults to False.
        kernellen: the lens of kernel for convolution, which is the length over which the convolution will be performed.
            For longer kernellen, longer streamline structure will appear. Defaults to 100.
        V_threshold: the threshold of velocity value for visualization. Defaults to None.
        vector: which vector type will be used for plotting, one of {'velocity', 'acceleration'} or either velocity
            field or acceleration field will be plotted. Defaults to "velocity".
        file: the path to save the slice figure. Defaults to None.
        save_show_or_return: whether to save, show or return the figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'line_integral_conv', "dpi": None,
            "ext": 'pdf', "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can
            provide a dictionary that properly modify those keys according to your needs. Defaults to {}.
        g_kwargs_dict: any other kwargs that would be passed to `dynamo.tl.grid_velocity_filter`. Defaults to {}.

    Raises:
        Exception: _description_
        Exception: _description_

    Returns:
        None would be returned by default. If `save_show_or_return` is set to "return" or "all", the generated 
        `yt.SlicePlot` will be returned.
    """

    import matplotlib.pyplot as plt

    X = adata.obsm["X_" + basis][:, :2] if "X_" + basis in adata.obsm.keys() else None
    V = adata.obsm[vector + "_" + basis][:, :2] if vector + "_" + basis in adata.obsm.keys() else None

    if X is None:
        raise Exception(f"The {basis} dimension reduction is not performed over your data yet.")
    if V is None:
        raise Exception(f"The {basis}_velocity velocity (or velocity) result does not existed in your data.")

    if U_grid is None or V_grid is None:
        if "VecFld_" + basis in adata.uns.keys():
            # first check whether the sparseVFC reconstructed vector field exists
            X_grid_, V_grid = (
                adata.uns["VecFld_" + basis]["grid"],
                adata.uns["VecFld_" + basis]["grid_V"],
            )
            N = int(np.sqrt(V_grid.shape[0]))
            U_grid = np.reshape(V_grid[:, 0], (N, N)).T
            V_grid = np.reshape(V_grid[:, 1], (N, N)).T

        elif "grid_velocity_" + basis in adata.uns.keys():
            # then check whether the Gaussian Kernel vector field exists
            X_grid_, V_grid_, _ = (
                adata.uns["grid_velocity_" + basis]["X_grid"],
                adata.uns["grid_velocity_" + basis]["V_grid"],
                adata.uns["grid_velocity_" + basis]["D"],
            )
            U_grid = V_grid_[0, :, :].T
            V_grid = V_grid_[1, :, :].T
        else:
            # if no VF or Gaussian Kernel vector fields, recreate it
            grid_kwargs_dict = {
                "density": None,
                "smooth": None,
                "n_neighbors": None,
                "min_mass": None,
                "autoscale": False,
                "adjust_for_stream": True,
                "V_threshold": None,
            }
            grid_kwargs_dict.update(g_kwargs_dict)

            X_grid_, V_grid_, _ = velocity_on_grid(X[:, [0, 1]], V[:, [0, 1]], xy_grid_nums, **grid_kwargs_dict)
            U_grid = V_grid_[0, :, :].T
            V_grid = V_grid_[1, :, :].T

    if V_threshold is not None:
        mass = np.sqrt((V_grid**2).sum(0))
        if V_threshold is not None:
            V_grid[0][mass.reshape(V_grid[0].shape) < V_threshold] = np.nan

    if method == "yt":
        try:
            import yt
        except ImportError:
            print(
                "Please first install yt package to use the line integral convolution plot method. "
                "Install instruction is provided here: https://yt-project.org/"
            )

        velocity_x_ori, velocity_y_ori, velocity_z_ori = (
            U_grid,
            V_grid,
            np.zeros(U_grid.shape),
        )
        velocity_x = np.repeat(velocity_x_ori[:, :, np.newaxis], V_grid.shape[1], axis=2)
        velocity_y = np.repeat(velocity_y_ori[:, :, np.newaxis], V_grid.shape[1], axis=2)
        velocity_z = np.repeat(velocity_z_ori[np.newaxis, :, :], V_grid.shape[1], axis=0)

        data = {}

        data["velocity_x"] = (velocity_x, "km/s")
        data["velocity_y"] = (velocity_y, "km/s")
        data["velocity_z"] = (velocity_z, "km/s")
        data["velocity_sum"] = (
            np.sqrt(velocity_x**2 + velocity_y**2),
            "km/s",
        )

        ds = yt.load_uniform_grid(data, data["velocity_x"][0].shape, length_unit=(1.0, "Mpc"))
        slc = yt.SlicePlot(ds, "z", ["velocity_sum"])
        slc.set_cmap("velocity_sum", cmap)
        slc.set_log("velocity_sum", False)

        slc.annotate_velocity(normalize=normalize)
        slc.annotate_streamlines(("gas", "velocity_x"), ("gas", "velocity_y"), density=density)
        slc.annotate_line_integral_convolution(
            ("gas", "velocity_x"),
            ("gas", "velocity_y"),
            lim=lim,
            const_alpha=const_alpha,
            kernellen=kernellen,
        )

        slc.set_xlabel(basis + "_1")
        slc.set_ylabel(basis + "_2")

        if save_show_or_return in ["save", "both", "all"]:
            slc.save(file, mpl_kwargs={"figsize": [2, 2]}, **save_kwargs)
        if save_show_or_return in ["show", "both", "all"]:
            slc.show()
        if save_show_or_return in ["return", "all"]:
            return slc
    elif method == "lic":
        # velocyto_tex = runlic(V_grid, V_grid, 100)
        # plot_LIC_gray(velocyto_tex)
        pass


@docstrings.with_indent(4)
def cell_wise_vectors(
    adata: AnnData,
    basis: str = "umap",
    x: int = 0,
    y: int = 1,
    z: int = 2,
    ekey: str = "M_s",
    vkey: str = "velocity_S",
    color: Union[str, List[str]] = "ntr",
    layer: str = "X",
    highlights: Optional[list] = None,
    labels: Optional[list] = None,
    values: Optional[list] = None,
    theme: Optional[
        Literal[
            "blue",
            "red",
            "green",
            "inferno",
            "fire",
            "viridis",
            "darkblue",
            "darkred",
            "darkgreen",
        ]
    ] = None,
    cmap: Optional[str] = None,
    color_key: Union[Dict[str, str], List[str], None] = None,
    color_key_cmap: Optional[str] = None,
    background: Optional[str] = "white",
    ncols: int = 4,
    pointsize: Optional[float] = None,
    figsize: Tuple[float, float] = (6, 4),
    show_legend: str = "on data",
    use_smoothed: bool = True,
    ax: Optional[Axes] = None,
    sort: Literal["raw", "abs", "neg"] = "raw",
    aggregate: Optional[str] = None,
    show_arrowed_spines: bool = False,
    inverse: bool = False,
    cell_inds: str = "all",
    quiver_size: Optional[float] = 1,
    quiver_length: Optional[float] = None,
    vector: str = "velocity",
    frontier: bool = False,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
    s_kwargs_dict: Dict[str, Any] = {},
    projection: Literal["2d", "3d"] = "2d",
    **cell_wise_kwargs,
) -> Optional[List[Axes]]:
    """Plot the velocity or acceleration vector of each cell.

    Args:
        adata: an AnnData object.
        basis: the reduced dimension stored in adata.obsm. The specific basis key will be constructed in the following
            priority if exits: 1) specific layer input +  basis 2) X_ + basis 3) basis. E.g. if basis is PCA, `scatters`
            is going to look for 1) if specific layer is spliced, `spliced_pca` 2) `X_pca` (dynamo convention) 3) `pca`.
            Defaults to "umap".
        x: the column index of the low dimensional embedding for the x-axis. Defaults to 0.
        y: the column index of the low dimensional embedding for the y-axis. Defaults to 1.
        z: the column index of the low dimensional embedding for the z-axis. Defaults to 2.
        ekey: the expression key. Defaults to "M_s".
        vkey: the velocity key. Defaults to "velocity_S".
        color: any column names or gene expression, etc. that will be used for coloring cells. Defaults to "ntr".
        layer: the layer of data to use for the scatter plot. Defaults to "X".
        highlights: the color group that will be highlighted. If highligts is a list of lists, each list is relate to
            each color element. Defaults to None.
        labels: an array of labels (assumed integer or categorical), one for each data sample. This will be used for
            coloring the points in the plot according to their label. Note that this option is mutually exclusive to the
            `values` option. Defaults to None.
        values: an array of values (assumed float or continuous), one for each sample. This will be used for coloring
            the points in the plot according to a colorscale associated to the total range of values. Note that this
            option is mutually exclusive to the `labels` option. Defaults to None.
        theme: A color theme to use for plotting. A small set of predefined themes are provided which have relatively
            good aesthetics. Available themes are: {'blue', 'red', 'green', 'inferno', 'fire', 'viridis', 'darkblue',
            'darkred', 'darkgreen'}. Defaults to None.
        cmap: The name of a matplotlib colormap to use for coloring or shading points. If no labels or values are passed
            this will be used for shading points according to density (largely only of relevance for very large
            datasets). If values are passed this will be used for shading according the value. Note that if theme is
            passed then this value will be overridden by the corresponding option of the theme. Defaults to None.
        color_key: the method to assign colors to categoricals. This can either be an explicit dict mapping labels to
            colors (as strings of form '#RRGGBB'), or an array like object providing one color for each distinct
            category being provided in `labels`. Either way this mapping will be used to color points according to the
            label. Note that if theme is passed then this value will be overridden by the corresponding option of the
            theme. Defaults to None.
        color_key_cmap: the name of a matplotlib colormap to use for categorical coloring. If an explicit `color_key` is
            not given a color mapping for categories can be generated from the label list and selecting a matching list
            of colors from the given colormap. Note that if theme is passed then this value will be overridden by the
            corresponding option of the theme. Defaults to None.
        background: the color of the background. Usually this will be either 'white' or 'black', but any color name will
            work. Ideally one wants to match this appropriately to the colors being used for points etc. This is one of
            the things that themes handle for you. Note that if theme is passed then this value will be overridden by
            the corresponding option of the theme. Defaults to None.
        ncols: the number of columns for the figure. Defaults to 4.
        pointsize: the scale of the point size. Actual point cell size is calculated as
            `500.0 / np.sqrt(adata.shape[0]) * pointsize`. Defaults to None.
        figsize: the width and height of a figure. Defaults to (6, 4).
        show_legend: whether to display a legend of the labels. Defaults to "on data".
        use_smoothed: whether to use smoothed values (i.e. M_s / M_u instead of spliced / unspliced, etc.). Defaults to
            True.
        ax: the matplotlib axes object where new plots will be added to. Only applicable to drawing a single component.
            Defaults to None.
        sort: the method to reorder data so that high values points will be on top of background points. Can be one of
            {'raw', 'abs', 'neg'}, i.e. sorted by raw data, sort by absolute values or sort by negative values. Defaults
            to "raw".
        aggregate: the column in adata.obs that will be used to aggregate data points. Defaults to None.
        show_arrowed_spines: whether to show a pair of arrowed spines representing the basis of the scatter is currently
            using. Defaults to False.
        inverse: whether to inverse the direction of the velocity vectors. Defaults to False.
        cell_inds: the cell index that will be chosen to draw velocity vectors. Can be a list of integers (cell indices)
            or str (Cell names). Defaults to "all".
        quiver_size: the size of quiver. If None, we will use set quiver_size to be 1. Note that quiver quiver_size is
            used to calculate the head_width (10 x quiver_size), head_length (12 x quiver_size) and headaxislength (8 x
            quiver_size) of the quiver. This is done via the `default_quiver_args` function which also calculate the
            scale of the quiver (1 / quiver_length). Defaults to 1.
        quiver_length: the length of quiver. The quiver length which will be used to calculate scale of quiver. Note
            that befoe applying `default_quiver_args` velocity values are first rescaled via the quiver_autoscaler
            function. Scale of quiver indicates the nuumber of data units per arrow length unit, e.g., m/s per plot
            width; a smaller scale parameter makes the arrow longer. Defaults to None.
        vector: which vector type will be used for plotting, one of {'velocity', 'acceleration'} or either velocity
            field or acceleration field will be plotted. Defaults to "velocity".
        frontier: whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to
            show area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib
            tips & tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from
            scEU-seq paper: https://science.sciencemag.org/content/367/6482/1151. Defaults to False.
        save_show_or_return: whether to save, show, or return the generated figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'cell_wise_velocity', "dpi": None,
            "ext": 'pdf', "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can
            provide a dictionary that properly modify those keys according to your needs. Defaults to {}.
        s_kwargs_dict: any other kwargs that will be passed to `dynamo.pl.scatters`. Defaults to {}.
        projection: the projection property of the matplotlib.Axes. Defaults to "2d".

    Raises:
        ValueError: invalid `x` or `y`.
        NotImplementedError: Invalid `projection`.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be `return`, the matplotlib axes of the
        generated subplots would be returned.
    """

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if projection == "2d":
        projection_dim_indexer = [x, y]
    elif projection == "3d":
        projection_dim_indexer = [x, y, z]
    else:
        projection_dim_indexer = [x, y]

    if type(x) == str and type(y) == str:
        if len(adata.var_names[adata.var.use_for_dynamics].intersection([x, y])) != 2:
            raise ValueError(
                "If you want to plot the vector flow of two genes, please make sure those two genes "
                "belongs to dynamics genes or .var.use_for_dynamics is True."
            )
        X = adata[:, projection_dim_indexer].layers[ekey].A
        V = adata[:, projection_dim_indexer].layers[vkey].A
        layer = ekey
    else:
        if ("X_" + basis in adata.obsm.keys()) and (vector + "_" + basis in adata.obsm.keys()):
            X = adata.obsm["X_" + basis][:, projection_dim_indexer]
            V = adata.obsm[vector + "_" + basis][:, projection_dim_indexer]
        else:
            if "X_" + basis not in adata.obsm.keys():
                layer, basis = basis.split("_")
                reduceDimension(adata, layer=layer, reduction_method=basis)
            if "kmc" not in adata.uns_keys():
                cell_velocities(adata, vkey="velocity_S", basis=basis)
                X = adata.obsm["X_" + basis][:, projection_dim_indexer]
                V = adata.obsm[vector + "_" + basis][:, projection_dim_indexer]
            else:
                kmc = adata.uns["kmc"]
                X = adata.obsm["X_" + basis][:, projection_dim_indexer]
                V = kmc.compute_density_corrected_drift(X, kmc.Idx, normalize_vector=True)
                adata.obsm[vector + "_" + basis] = V

    X, V = X.copy(), V.copy()

    V /= 3 * quiver_autoscaler(X, V)
    if inverse:
        V = -V
    df = None
    main_info("X shape: " + str(X.shape) + " V shape: " + str(V.shape))
    if projection == "2d":
        df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "u": V[:, 0], "v": V[:, 1]})
    elif projection == "3d":
        df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "z": X[:, 2], "u": V[:, 0], "v": V[:, 1], "w": V[:, 2]})
        show_legend = None
    else:
        raise NotImplementedError("Projection method %s is not implemented" % projection)

    if cell_inds == "all":
        ix_choice = np.arange(adata.shape[0])
    elif cell_inds == "random":
        ix_choice = np.random.choice(np.range(adata.shape[0]), size=1000, replace=False)
    elif type(cell_inds) is int:
        ix_choice = np.random.choice(np.range(adata.shape[0]), size=cell_inds, replace=False)
    elif type(cell_inds) is list:
        if type(cell_inds[0]) is str:
            cell_inds = [adata.obs_names.to_list().index(i) for i in cell_inds]
        ix_choice = cell_inds

    df = df.iloc[ix_choice, :]

    if background is None:
        _background = rcParams.get("figure.facecolor")
        background = to_hex(_background) if type(_background) is tuple else _background

    if quiver_size is None:
        quiver_size = 1
    if background == "black":
        edgecolors = "white"
    else:
        edgecolors = "black"

    head_w, head_l, ax_l, scale = default_quiver_args(quiver_size, quiver_length)  #
    quiver_kwargs = {
        "angles": "xy",
        "scale": scale,
        "scale_units": "xy",
        "width": 0.0005,
        "headwidth": head_w,
        "headlength": head_l,
        "headaxislength": ax_l,
        "minshaft": 1,
        "minlength": 1,
        "pivot": "tail",
        "linewidth": 0.1,
        "edgecolors": edgecolors,
        "alpha": 1,
        "zorder": 10,
    }
    quiver_kwargs = update_dict(quiver_kwargs, cell_wise_kwargs)
    quiver_3d_kwargs = {
        "linewidth": 1,
        "edgecolors": "white",
        "alpha": 1,
        "length": 8,
        "arrow_length_ratio": scale,

    }

    axes_list, color_list, _ = scatters(
        adata=adata,
        basis=basis,
        x=x,
        y=y,
        z=z,
        color=color,
        layer=layer,
        highlights=highlights,
        labels=labels,
        values=values,
        theme=theme,
        cmap=cmap,
        color_key=color_key,
        color_key_cmap=color_key_cmap,
        background=background,
        ncols=ncols,
        pointsize=pointsize,
        figsize=figsize,
        show_legend=show_legend,
        use_smoothed=use_smoothed,
        aggregate=aggregate,
        show_arrowed_spines=show_arrowed_spines,
        ax=ax,
        sort=sort,
        save_show_or_return="return",
        frontier=frontier,
        projection=projection,
        **s_kwargs_dict,
        return_all=True,
    )

    # single axis output
    if type(axes_list) != list:
        axes_list = [axes_list]
        color_list = [color_list]
    x0, x1 = df.iloc[:, 0], df.iloc[:, 1]
    v0, v1 = df.iloc[:, 2], df.iloc[:, 3]

    if projection == "3d":
        x0, x1, x2 = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
        v0, v1, v2 = df.iloc[:, 3], df.iloc[:, 4], df.iloc[:, 5]

    for i in range(len(axes_list)):
        ax = axes_list[i]
        if projection == "2d":
            ax.quiver(
                x0,
                x1,
                v0,
                v1,
                color=color_list[i],
                facecolors=color_list[i],
                **quiver_kwargs,
            )
        elif projection == "3d":
            cmap_3d = [element for element in color_list[i]] + [element for element in color_list[i] for _ in range(2)]
            ax.quiver(
                x0,
                x1,
                x2,
                v0,
                v1,
                v2,
                color=cmap_3d,
                # facecolors=color_list[i],
                **quiver_3d_kwargs,
            )
        ax.set_facecolor(background)

    return save_show_ret("cell_wise_vector", save_show_or_return, save_kwargs, axes_list, tight = projection != "3d")


@docstrings.with_indent(4)
def grid_vectors(
    adata: AnnData,
    basis: str = "umap",
    x: int = 0,
    y: int = 1,
    ekey: str = "M_s",
    vkey: str = "velocity_S",
    color: Union[str, List[str]] = "ntr",
    layer: str = "X",
    highlights: Optional[list] = None,
    labels: Optional[list] = None,
    values: Optional[list] = None,
    theme: Optional[
        Literal[
            "blue",
            "red",
            "green",
            "inferno",
            "fire",
            "viridis",
            "darkblue",
            "darkred",
            "darkgreen",
        ]
    ] = None,
    cmap: Optional[str] = None,
    color_key: Union[Dict[str, str], List[str], None] = None,
    color_key_cmap: Optional[str] = None,
    background: Optional[str] = "white",
    ncols: int = 4,
    pointsize: Optional[float] = None,
    figsize: Tuple[float] = (6, 4),
    show_legend: str = "on data",
    use_smoothed: bool = True,
    ax: Optional[Axes] = None,
    sort: Literal["raw", "abs", "neg"] = "raw",
    aggregate: Optional[str] = None,
    show_arrowed_spines: bool = False,
    inverse: bool = False,
    cell_inds: Union[str, list] = "all",
    method: Literal["SparseVFC", "gaussian"] = "gaussian",
    xy_grid_nums: Tuple[int, int] = (50, 50),
    cut_off_velocity: bool = True,
    quiver_size: Optional[float] = None,
    quiver_length: Optional[float] = None,
    vector: str = "velocity",
    frontier: bool = False,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
    s_kwargs_dict: Dict[str, Any] = {},
    q_kwargs_dict: Dict[str, Any] = {},
    **grid_kwargs,
) -> Union[List[Axes], Axes, None]:
    """Plot the velocity or acceleration vector of each cell on a grid.

    Args:
        adata: an AnnData object.
        basis: the reduced dimension stored in adata.obsm. The specific basis key will be constructed in the following
            priority if exits: 1) specific layer input +  basis 2) X_ + basis 3) basis. E.g. if basis is PCA, `scatters`
            is going to look for 1) if specific layer is spliced, `spliced_pca` 2) `X_pca` (dynamo convention) 3) `pca`.
            Defaults to "umap".
        x: the column index of the low dimensional embedding for the x-axis. Defaults to 0.
        y: the column index of the low dimensional embedding for the y-axis. Defaults to 1.
        ekey: the expression key. Defaults to "M_s".
        vkey: the velocity key. Defaults to "velocity_S".
        color: any column names or gene expression, etc. that will be used for coloring cells. Defaults to "ntr".
        layer: the layer of data to use for the scatter plot. Defaults to "X".
        highlights: the color group that will be highlighted. If highligts is a list of lists, each list is relate to
            each color element. Defaults to None.
        labels: an array of labels (assumed integer or categorical), one for each data sample. This will be used for
            coloring the points in the plot according to their label. Note that this option is mutually exclusive to the
            `values` option. Defaults to None.
        values: an array of values (assumed float or continuous), one for each sample. This will be used for coloring
            the points in the plot according to a colorscale associated to the total range of values. Note that this
            option is mutually exclusive to the `labels` option. Defaults to None.
        theme: A color theme to use for plotting. A small set of predefined themes are provided which have relatively
            good aesthetics. Available themes are: {'blue', 'red', 'green', 'inferno', 'fire', 'viridis', 'darkblue',
            'darkred', 'darkgreen'}. Defaults to None.
        cmap: The name of a matplotlib colormap to use for coloring or shading points. If no labels or values are passed
            this will be used for shading points according to density (largely only of relevance for very large
            datasets). If values are passed this will be used for shading according the value. Note that if theme is
            passed then this value will be overridden by the corresponding option of the theme. Defaults to None.
        color_key: the method to assign colors to categoricals. This can either be an explicit dict mapping labels to
            colors (as strings of form '#RRGGBB'), or an array like object providing one color for each distinct
            category being provided in `labels`. Either way this mapping will be used to color points according to the
            label. Note that if theme is passed then this value will be overridden by the corresponding option of the
            theme. Defaults to None.
        color_key_cmap: the name of a matplotlib colormap to use for categorical coloring. If an explicit `color_key` is
            not given a color mapping for categories can be generated from the label list and selecting a matching list
            of colors from the given colormap. Note that if theme is passed then this value will be overridden by the
            corresponding option of the theme. Defaults to None.
        background: the color of the background. Usually this will be either 'white' or 'black', but any color name will
            work. Ideally one wants to match this appropriately to the colors being used for points etc. This is one of
            the things that themes handle for you. Note that if theme is passed then this value will be overridden by
            the corresponding option of the theme. Defaults to None.
        ncols: the number of columns for the figure. Defaults to 4.
        pointsize: the scale of the point size. Actual point cell size is calculated as
            `500.0 / np.sqrt(adata.shape[0]) * pointsize`. Defaults to None.
        figsize: the width and height of a figure. Defaults to (6, 4).
        show_legend: whether to display a legend of the labels. Defaults to "on data".
        use_smoothed: whether to use smoothed values (i.e. M_s / M_u instead of spliced / unspliced, etc.). Defaults to
            True.
        ax: the matplotlib axes object where new plots will be added to. Only applicable to drawing a single component.
            Defaults to None.
        sort: the method to reorder data so that high values points will be on top of background points. Can be one of
            {'raw', 'abs', 'neg'}, i.e. sorted by raw data, sort by absolute values or sort by negative values. Defaults
            to "raw".
        aggregate: the column in adata.obs that will be used to aggregate data points. Defaults to None.
        show_arrowed_spines: whether to show a pair of arrowed spines representing the basis of the scatter is currently
            using. Defaults to False.
        inverse: whether to inverse the direction of the velocity vectors. Defaults to False.
        cell_inds: the cell index that will be chosen to draw velocity vectors. Can be a list of integers (cell integer
            indices) or str (Cell names). Defaults to "all".
        method: method to reconstruct the vector field. Currently it supports either SparseVFC (default) or the
            empirical method Gaussian kernel method from RNA velocity (Gaussian). Defaults to "gaussian".
        xy_grid_nums: the number of grids in either x or y axis. Defaults to (50, 50).
        cut_off_velocity: whether to remove small velocity vectors from the recovered the vector field grid, either
            through the simple Gaussian kernel (applicable to 2D) or the powerful sparseVFC approach. Defaults to True.
        quiver_size: the size of quiver. If None, we will use set quiver_size to be 1. Note that quiver quiver_size is
            used to calculate the head_width (10 x quiver_size), head_length (12 x quiver_size) and headaxislength (8 x
            quiver_size) of the quiver. This is done via the `default_quiver_args` function which also calculate the
            scale of the quiver (1 / quiver_length). Defaults to None.
        quiver_length: the length of quiver. The quiver length which will be used to calculate scale of quiver. Note
            that befoe applying `default_quiver_args` velocity values are first rescaled via the quiver_autoscaler
            function. Scale of quiver indicates the nuumber of data units per arrow length unit, e.g., m/s per plot
            width; a smaller scale parameter makes the arrow longer. Defaults to None.
        vector: which vector type will be used for plotting, one of {'velocity', 'acceleration'} or either velocity
            field or acceleration field will be plotted. Defaults to "velocity".
        frontier: whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to
            show area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib
            tips & tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from
            scEU-seq paper: https://science.sciencemag.org/content/367/6482/1151. Defaults to False.
        save_show_or_return: whether to save, show, or return the generated figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'grid_velocity', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs.. Defaults to {}.
        s_kwargs_dict: any other kwargs that would be passed to `dynamo.pl.scatters`. Defaults to {}.
        q_kwargs_dict: any other kwargs that would be passed to `pyplot.quiver`. Defaults to {}.
        **grid_kwargs: any other kwargs that would be passed to `dynamo.tl.grid_velocity_filter`.

    Raises:
        ValueError: invalid `x` or `y`.
        NotImplementedError: invalid `method`.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be `return`, the matplotlib axes of the
        generated subplots would be returned.
    """

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if type(x) == str and type(y) == str:
        if len(adata.var_names[adata.var.use_for_dynamics].intersection([x, y])) != 2:
            raise ValueError(
                "If you want to plot the vector flow of two genes, please make sure those two genes "
                "belongs to dynamics genes or .var.use_for_dynamics is True."
            )
        X = adata[:, [x, y]].layers[ekey].A
        V = adata[:, [x, y]].layers[vkey].A
        layer = ekey
    else:
        if ("X_" + basis in adata.obsm.keys()) and (vector + "_" + basis in adata.obsm.keys()):
            X = adata.obsm["X_" + basis][:, [x, y]]
            V = adata.obsm[vector + "_" + basis][:, [x, y]]
        else:
            if "X_" + basis not in adata.obsm.keys():
                layer, basis = basis.split("_")
                reduceDimension(adata, layer=layer, reduction_method=basis)
            if "kmc" not in adata.uns_keys():
                cell_velocities(adata, vkey="velocity_S", basis=basis)
                X = adata.obsm["X_" + basis][:, [x, y]]
                V = adata.obsm[vector + "_" + basis][:, [x, y]]
            else:
                kmc = adata.uns["kmc"]
                X = adata.obsm["X_" + basis][:, [x, y]]
                V = kmc.compute_density_corrected_drift(X, kmc.Idx, normalize_vector=True)
                adata.obsm[vector + "_" + basis] = V

    X, V = X.copy(), V.copy()

    if cell_inds == "all":
        ix_choice = np.arange(adata.shape[0])
    elif cell_inds == "random":
        ix_choice = np.random.choice(np.range(adata.shape[0]), size=1000, replace=False)
    elif type(cell_inds) is int:
        ix_choice = np.random.choice(np.range(adata.shape[0]), size=cell_inds, replace=False)
    elif type(cell_inds) is list:
        if type(cell_inds[0]) is str:
            cell_inds = [adata.obs_names.to_list().index(i) for i in cell_inds]
        ix_choice = cell_inds

    X, V = X[ix_choice, :], V[ix_choice, :]  # 0, 0

    grid_kwargs_dict = {
        "density": None,
        "smooth": None,
        "n_neighbors": None,
        "min_mass": None,
        "autoscale": False,
        "adjust_for_stream": True,
        "V_threshold": None,
    }
    grid_kwargs_dict = update_dict(grid_kwargs_dict, grid_kwargs)

    if method.lower() == "sparsevfc":
        if "VecFld_" + basis not in adata.uns.keys():
            VectorField(adata, basis=basis, dims=[x, y])
        X_grid, V_grid = (
            adata.uns["VecFld_" + basis]["grid"],
            adata.uns["VecFld_" + basis]["grid_V"],
        )
        N = int(np.sqrt(V_grid.shape[0]))

        if cut_off_velocity:
            X_grid, p_mass, neighs, weight = prepare_velocity_grid_data(
                X,
                xy_grid_nums,
                density=grid_kwargs_dict["density"],
                smooth=grid_kwargs_dict["smooth"],
                n_neighbors=grid_kwargs_dict["n_neighbors"],
            )
            for i in ["density", "smooth", "n_neighbors"]:
                grid_kwargs_dict.pop(i)

            VecFld, func = vecfld_from_adata(adata, basis)

            V_emb = func(X)
            V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1) / np.maximum(1, p_mass)[:, None]
            X_grid, V_grid = grid_velocity_filter(
                V_emb=V,
                neighs=neighs,
                p_mass=p_mass,
                X_grid=X_grid,
                V_grid=V_grid,
                **grid_kwargs_dict,
            )
        else:
            X_grid, V_grid = (
                np.array([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])]),
                np.array([V_grid[:, 0].reshape((N, N)), V_grid[:, 1].reshape((N, N))]),
            )
    elif method.lower() == "gaussian":
        X_grid, V_grid, D = velocity_on_grid(
            X,
            V,
            xy_grid_nums,
            cut_off_velocity=cut_off_velocity,
            **grid_kwargs_dict,
        )
    elif "grid_velocity_" + basis in adata.uns.keys():
        X_grid, V_grid, _ = (
            adata.uns["grid_velocity_" + basis]["VecFld"]["X_grid"],
            adata.uns["grid_velocity_" + basis]["VecFld"]["V_grid"],
            adata.uns["grid_velocity_" + basis]["VecFld"]["D"],
        )
    else:
        raise NotImplementedError(
            "Vector field learning method {} is not supported or the grid velocity is collected for "
            "the current adata object.".format(method)
        )

    V_grid /= 3 * quiver_autoscaler(X_grid, V_grid)
    if inverse:
        V_grid = -V_grid

    if background is None:
        _background = rcParams.get("figure.facecolor")
        background = to_hex(_background) if type(_background) is tuple else _background
    if quiver_size is None:
        quiver_size = 1
    if background == "black":
        edgecolors = "white"
    else:
        edgecolors = "black"

    head_w, head_l, ax_l, scale = default_quiver_args(quiver_size, quiver_length)

    quiver_kwargs = {
        "angles": "xy",
        "scale": scale,
        "scale_units": "xy",
        "width": 0.0005,
        "headwidth": head_w,
        "headlength": head_l,
        "headaxislength": ax_l,
        "minshaft": 1,
        "minlength": 1,
        "pivot": "tail",
        "edgecolors": edgecolors,
        "linewidth": 0.2,
        "facecolors": edgecolors,
        "color": edgecolors,
        "alpha": 1,
        "zorder": 3,
    }
    quiver_kwargs = update_dict(quiver_kwargs, q_kwargs_dict)

    # if ax is None:
    #     plt.figure(facecolor=background)
    axes_list, _, font_color = scatters(
        adata=adata,
        basis=basis,
        x=x,
        y=y,
        color=color,
        layer=layer,
        highlights=highlights,
        labels=labels,
        values=values,
        theme=theme,
        cmap=cmap,
        color_key=color_key,
        color_key_cmap=color_key_cmap,
        background=background,
        ncols=ncols,
        pointsize=pointsize,
        figsize=figsize,
        show_legend=show_legend,
        use_smoothed=use_smoothed,
        aggregate=aggregate,
        show_arrowed_spines=show_arrowed_spines,
        ax=ax,
        sort=sort,
        save_show_or_return="return",
        frontier=frontier,
        **s_kwargs_dict,
        return_all=True,
    )

    if type(axes_list) == list:
        for i in range(len(axes_list)):
            axes_list[i].quiver(X_grid[0], X_grid[1], V_grid[0], V_grid[1], **quiver_kwargs)
            axes_list[i].set_facecolor(background)
    else:
        axes_list.quiver(X_grid[0], X_grid[1], V_grid[0], V_grid[1], **quiver_kwargs)
        axes_list.set_facecolor(background)

    return save_show_ret("grid_velocity", save_show_or_return, save_kwargs, axes_list)


@docstrings.with_indent(4)
def streamline_plot(
    adata: AnnData,
    basis: str = "umap",
    x: int = 0,
    y: int = 1,
    ekey: str = "M_s",
    vkey: str = "velocity_S",
    color: Union[str, List[str]] = "ntr",
    layer: str = "X",
    highlights: Optional[list] = None,
    labels: Optional[list] = None,
    values: Optional[list] = None,
    theme: Optional[
        Literal[
            "blue",
            "red",
            "green",
            "inferno",
            "fire",
            "viridis",
            "darkblue",
            "darkred",
            "darkgreen",
        ]
    ] = None,
    cmap: Optional[str] = None,
    color_key: Union[Dict[str, str], List[str], None] = None,
    color_key_cmap: Optional[str] = None,
    background: Optional[str] = "white",
    ncols: int = 4,
    pointsize: Optional[float] = None,
    figsize: Tuple[float, float] = (6, 4),
    show_legend: str = "on data",
    use_smoothed: bool = True,
    ax: Optional[Axes] = None,
    sort: Literal["raw", "abs", "neg"] = "raw",
    aggregate: Optional[str] = None,
    show_arrowed_spines: bool = False,
    inverse: bool = False,
    cell_inds: Union[str, list] = "all",
    method: Literal["gaussian", "SparseVFC"] = "gaussian",
    xy_grid_nums: Tuple[int, int] = (50, 50),
    cut_off_velocity: bool = True,
    density: float = 1,
    linewidth: float = 1,
    streamline_alpha: float = 1,
    vector: str = "velocity",
    frontier: bool = False,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
    s_kwargs_dict: Dict[str, Any] = {},
    **streamline_kwargs,
) -> List[Axes]:
    """Plot the velocity vector of each cell.

    Args:
        adata: an AnnData object.
        basis: the reduced dimension stored in adata.obsm. The specific basis key will be constructed in the following
            priority if exits: 1) specific layer input +  basis 2) X_ + basis 3) basis. E.g. if basis is PCA, `scatters`
            is going to look for 1) if specific layer is spliced, `spliced_pca` 2) `X_pca` (dynamo convention) 3) `pca`.
            Defaults to "umap".
        x: the column index of the low dimensional embedding for the x-axis. Defaults to 0.
        y: the column index of the low dimensional embedding for the y-axis. Defaults to 1.
        ekey: the expression key. Defaults to "M_s".
        vkey: the velocity key. Defaults to "velocity_S".
        color: any column names or gene expression, etc. that will be used for coloring cells. Defaults to "ntr".
        layer: the layer of data to use for the scatter plot. Defaults to "X".
        highlights: the color group that will be highlighted. If highligts is a list of lists, each list is relate to
            each color element. Defaults to None.
        labels: an array of labels (assumed integer or categorical), one for each data sample. This will be used for
            coloring the points in the plot according to their label. Note that this option is mutually exclusive to the
            `values` option. Defaults to None.
        values: an array of values (assumed float or continuous), one for each sample. This will be used for coloring
            the points in the plot according to a colorscale associated to the total range of values. Note that this
            option is mutually exclusive to the `labels` option. Defaults to None.
        theme: A color theme to use for plotting. A small set of predefined themes are provided which have relatively
            good aesthetics. Available themes are: {'blue', 'red', 'green', 'inferno', 'fire', 'viridis', 'darkblue',
            'darkred', 'darkgreen'}. Defaults to None.
        cmap: The name of a matplotlib colormap to use for coloring or shading points. If no labels or values are passed
            this will be used for shading points according to density (largely only of relevance for very large
            datasets). If values are passed this will be used for shading according the value. Note that if theme is
            passed then this value will be overridden by the corresponding option of the theme. Defaults to None.
        color_key: the method to assign colors to categoricals. This can either be an explicit dict mapping labels to
            colors (as strings of form '#RRGGBB'), or an array like object providing one color for each distinct
            category being provided in `labels`. Either way this mapping will be used to color points according to the
            label. Note that if theme is passed then this value will be overridden by the corresponding option of the
            theme. Defaults to None.
        color_key_cmap: the name of a matplotlib colormap to use for categorical coloring. If an explicit `color_key` is
            not given a color mapping for categories can be generated from the label list and selecting a matching list
            of colors from the given colormap. Note that if theme is passed then this value will be overridden by the
            corresponding option of the theme. Defaults to None.
        background: the color of the background. Usually this will be either 'white' or 'black', but any color name will
            work. Ideally one wants to match this appropriately to the colors being used for points etc. This is one of
            the things that themes handle for you. Note that if theme is passed then this value will be overridden by
            the corresponding option of the theme. Defaults to None.
        ncols: the number of columns for the figure. Defaults to 4.
        pointsize: the scale of the point size. Actual point cell size is calculated as
            `500.0 / np.sqrt(adata.shape[0]) * pointsize`. Defaults to None.
        figsize: the width and height of a figure. Defaults to (6, 4).
        show_legend: whether to display a legend of the labels. Defaults to "on data".
        use_smoothed: whether to use smoothed values (i.e. M_s / M_u instead of spliced / unspliced, etc.). Defaults to
            True.
        ax: the matplotlib axes object where new plots will be added to. Only applicable to drawing a single component.
            Defaults to None.
        sort: the method to reorder data so that high values points will be on top of background points. Can be one of
            {'raw', 'abs', 'neg'}, i.e. sorted by raw data, sort by absolute values or sort by negative values. Defaults
            to "raw".
        aggregate: the column in adata.obs that will be used to aggregate data points. Defaults to None.
        show_arrowed_spines: whether to show a pair of arrowed spines representing the basis of the scatter is currently
            using. Defaults to False.
        inverse: whether to inverse the direction of the velocity vectors. Defaults to False.
        cell_inds: the cell index that will be chosen to draw velocity vectors. Can be a list of integers (cell integer
            indices) or str (Cell names). Defaults to "all".
        method: the method to reconstruct the vector field. Currently, it supports either SparseVFC (default) or the
            empirical method Gaussian kernel method from RNA velocity (Gaussian). Defaults to "gaussian".
        xy_grid_nums: the number of grids in either x or y axis. Defaults to (50, 50).
        cut_off_velocity: whether to remove small velocity vectors from the recovered the vector field grid, either
            through the simple Gaussian kernel (applicable only to 2D) or the powerful sparseVFC approach. Defaults to True.
        density: density of the `plt.streamplot` function. Defaults to 1.
        linewidth: multiplier of automatically calculated linewidth passed to the `plt.streamplot function`. Defaults
            to 1.
        streamline_alpha: the alpha value applied to the vector field streamlines. Defaults to 1.
        vector: which vector type will be used for plotting, one of {'velocity', 'acceleration'} or either velocity
            field or acceleration field will be plotted. Defaults to "velocity".
        frontier: whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to
            show area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib
            tips & tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from
            scEU-seq paper: https://science.sciencemag.org/content/367/6482/1151. Defaults to False.
        save_show_or_return: whether to save, show, or return the generated figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'streamline_plot', "dpi": None,
            "ext": 'pdf', "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can
            provide a dictionary that properly modify those keys according to your needs.. Defaults to {}.
        s_kwargs_dict: any other kwargs that would be passed to `dynamo.pl.scatters`. Defaults to {}.

    Raises:
        ValueError: invalid `x` or `y`.
        NotImplementedError: invalid `method`.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to 'return', the matplotlib Axes objects of
        the subplots would be returned.
    """

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if type(x) == str and type(y) == str:
        if len(adata.var_names[adata.var.use_for_dynamics].intersection([x, y])) != 2:
            raise ValueError(
                "If you want to plot the vector flow of two genes, please make sure those two genes "
                "belongs to dynamics genes or .var.use_for_dynamics is True."
            )
        X = adata[:, [x, y]].layers[ekey].A
        V = adata[:, [x, y]].layers[vkey].A
        layer = ekey
    else:
        if ("X_" + basis in adata.obsm.keys()) and (vector + "_" + basis in adata.obsm.keys()):
            X = adata.obsm["X_" + basis][:, [x, y]]
            V = adata.obsm[vector + "_" + basis][:, [x, y]]
        else:
            if basis not in adata.obsm.keys() or "X_" + basis not in adata.obsm.keys():
                layer, basis = basis.split("_") if "_" in basis else ("X", basis)
                reduceDimension(adata, layer=layer, reduction_method=basis)
            if "kmc" not in adata.uns_keys():
                cell_velocities(adata, vkey="velocity_S", basis=basis)
                X = adata.obsm["X_" + basis][:, [x, y]]
                V = adata.obsm[vector + "_" + basis][:, [x, y]]
            else:
                kmc = adata.uns["kmc"]
                X = adata.obsm["X_" + basis][:, [x, y]]
                V = kmc.compute_density_corrected_drift(X, kmc.Idx, normalize_vector=True)
                adata.obsm[vector + "_" + basis] = V

    X, V = X.copy(), V.copy()

    if cell_inds == "all":
        ix_choice = np.arange(adata.shape[0])
    elif cell_inds == "random":
        ix_choice = np.random.choice(np.range(adata.shape[0]), size=1000, replace=False)
    elif type(cell_inds) is int:
        ix_choice = np.random.choice(np.range(adata.shape[0]), size=cell_inds, replace=False)
    elif type(cell_inds) is list:
        if type(cell_inds[0]) is str:
            cell_inds = [adata.obs_names.to_list().index(i) for i in cell_inds]
        ix_choice = cell_inds

    X, V = X[ix_choice, :], V[ix_choice, :]  # 0, 0

    grid_kwargs_dict = {
        "density": None,
        "smooth": None,
        "n_neighbors": None,
        "min_mass": None,
        "autoscale": False,
        "adjust_for_stream": True,
        "V_threshold": None,
    }
    grid_kwargs_dict = update_dict(grid_kwargs_dict, streamline_kwargs)

    if method.lower() == "sparsevfc":
        if "VecFld_" + basis not in adata.uns.keys():
            VectorField(adata, basis=basis, dims=[x, y])
        X_grid, V_grid = (
            adata.uns["VecFld_" + basis]["grid"],
            adata.uns["VecFld_" + basis]["grid_V"],
        )
        N = int(np.sqrt(V_grid.shape[0]))

        if cut_off_velocity:
            X_grid, p_mass, neighs, weight = prepare_velocity_grid_data(
                X,
                xy_grid_nums,
                density=grid_kwargs_dict["density"],
                smooth=grid_kwargs_dict["smooth"],
                n_neighbors=grid_kwargs_dict["n_neighbors"],
            )
            for i in ["density", "smooth", "n_neighbors"]:
                grid_kwargs_dict.pop(i)

            VecFld, func = vecfld_from_adata(adata, basis)

            V_emb = func(X)
            V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1) / np.maximum(1, p_mass)[:, None]
            X_grid, V_grid = grid_velocity_filter(
                V_emb=V,
                neighs=neighs,
                p_mass=p_mass,
                X_grid=X_grid,
                V_grid=V_grid,
                **grid_kwargs_dict,
            )
        else:
            X_grid, V_grid = (
                np.array([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])]),
                np.array([V_grid[:, 0].reshape((N, N)), V_grid[:, 1].reshape((N, N))]),
            )
    elif method.lower() == "gaussian":
        X_grid, V_grid, D = velocity_on_grid(
            X,
            V,
            xy_grid_nums,
            cut_off_velocity=cut_off_velocity,
            **grid_kwargs_dict,
        )
    elif "grid_velocity_" + basis in adata.uns.keys():
        X_grid, V_grid, _ = (
            adata.uns["grid_velocity_" + basis]["VecFld"]["X_grid"],
            adata.uns["grid_velocity_" + basis]["VecFld"]["V_grid"],
            adata.uns["grid_velocity_" + basis]["VecFld"]["D"],
        )
    else:
        raise NotImplementedError(
            "Vector field learning method {} is not supported or the grid velocity is collected for "
            "the current adata object.".format(method)
        )

    if inverse:
        V_grid = -V_grid
    streamplot_kwargs = {
        "density": density * 2,
        "linewidth": None,
        "cmap": None,
        "norm": None,
        "arrowsize": 1,
        "arrowstyle": "fancy",
        "minlength": 0.1,
        "transform": None,
        "start_points": None,
        "maxlength": 4.0,
        "integration_direction": "both",
        "zorder": 3,
    }
    mass = np.sqrt((V_grid**2).sum(0))
    linewidth *= 2 * mass / mass[~np.isnan(mass)].max()
    streamplot_kwargs.update({"linewidth": linewidth * streamline_kwargs.pop("linewidth", 1)})

    streamplot_kwargs = update_dict(streamplot_kwargs, streamline_kwargs)

    if background is None:
        _background = rcParams.get("figure.facecolor")
        background = to_hex(_background) if type(_background) is tuple else _background

    if background in ["black", "#ffffff"]:
        streamline_color = "red"
    else:
        streamline_color = "black"

    # if ax is None:
    #     plt.figure(facecolor=background)
    axes_list, _, _ = scatters(
        adata=adata,
        basis=basis,
        x=x,
        y=y,
        color=color,
        layer=layer,
        highlights=highlights,
        labels=labels,
        values=values,
        theme=theme,
        cmap=cmap,
        color_key=color_key,
        color_key_cmap=color_key_cmap,
        background=background,
        ncols=ncols,
        pointsize=pointsize,
        figsize=figsize,
        show_legend=show_legend,
        use_smoothed=use_smoothed,
        aggregate=aggregate,
        show_arrowed_spines=show_arrowed_spines,
        ax=ax,
        sort=sort,
        save_show_or_return="return",
        frontier=frontier,
        **s_kwargs_dict,
        return_all=True,
    )
    # single axis case, convert to list
    if type(axes_list) != list:
        axes_list = [axes_list]

    def streamplot_2d(ax):
        ax.set_facecolor(background)
        s = ax.streamplot(
            X_grid[0],
            X_grid[1],
            V_grid[0],
            V_grid[1],
            color=streamline_color,
            **streamplot_kwargs,
        )
        set_arrow_alpha(ax, streamline_alpha)
        set_stream_line_alpha(s, streamline_alpha)

    if type(axes_list) == list:
        for i in range(len(axes_list)):
            ax = axes_list[i]
            streamplot_2d(ax)

    return save_show_ret("streamline_plot", save_show_or_return, save_kwargs, axes_list)


# refactor line_conv_integration


def plot_energy(
    adata: AnnData,
    basis: Optional[str] = None,
    vecfld_dict: Optional[dict] = None,
    figsize: Optional[Tuple[float, float]] = None,
    fig: Optional[Figure] = None,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
) -> Optional[Figure]:
    """Plot the energy and energy change rate over each optimization iteration.

    Args:
        adata: an Annodata object with vector field function reconstructed.
        basis: the reduced dimension embedding (pca or umap, for example) of cells from which vector field function was
            reconstructed. When basis is None, the velocity vector field function building from the full gene expression
            space is used. Defaults to None.
        vecfld_dict: the dictionary storing the information for the reconstructed velocity vector field function. If
            None, the corresponding dictionary stored in the adata object will be used. Defaults to None.
        figsize: the width and height of the resulting figure when fig is set to be None. Defaults to None.
        fig: the figure object where panels of the energy or energy change rate over iteration plots will be appended
            to. Defaults to None.
        save_show_or_return: whether to save, show or return the figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'energy', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs.. Defaults to {}.

    Raises:
        ValueError: invalid `basis`.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to 'return', the matplotlib Figure object
        of the graph would be returned.
    """

    import matplotlib.pyplot as plt

    if vecfld_dict is None:
        vf_key = "VecFld" if basis is None else "VecFld_" + basis
        if vf_key not in adata.uns.keys():
            raise ValueError(
                f"Your adata doesn't have the key for Vector Field with {basis} basis."
                f"Try firstly running dyn.vf.VectorField(adata, basis={basis})."
            )

        vecfld_dict = adata.uns[vf_key]

    E = vecfld_dict["E_traj"] if "E_traj" in vecfld_dict.keys() else None
    tecr = vecfld_dict["tecr_traj"] if "tecr_traj" in vecfld_dict.keys() else None

    if E is not None and tecr is not None:
        fig = fig or plt.figure(figsize=figsize)

        Iterations = np.arange(0, len(E))
        ax = fig.add_subplot(1, 2, 1)
        E_ = E - np.min(E) + 1
        ax.plot(Iterations, E_, "k")
        ax.plot(E_, "r.")
        ax.set_yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Energy")

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(Iterations, tecr, "k")
        ax.plot(tecr, "r.")
        ax.set_yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Energy change rate")

    return save_show_ret("energy", save_show_or_return, save_kwargs, fig)
