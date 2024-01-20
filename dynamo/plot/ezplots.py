from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from .utils import save_show_ret
from ..tools.utils import flatten, index_gene, velocity_on_grid
from ..utils import areinstance, isarray

# from ..tools.Markov import smoothen_drift_on_grid

SchemeDiverge = {
    "cmap": "Spectral_r",
    "sym_c": True,
    "sort_by_c": "abs",
}

SchemeDivergeBWR = {
    "cmap": "bwr",
    "sym_c": True,
    "sort_by_c": "abs",
}


def plot_X(
    X: np.ndarray,
    dim1: int = 0,
    dim2: int = 1,
    dim3: Optional[int] = None,
    dims: Optional[List[int]] = None,
    create_figure: bool = False,
    figsize: Tuple[float, float] = (6, 6),
    sort_by_c: Literal["neg", "abs", "raw"] = "raw",
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: dict = {},
    **kwargs,
) -> None:
    """Plot scatter graph of the specified dimensions in an array.

    Args:
        X: the array with data to be plotted.
        dim1: the index corresponding to the 1st dimension to be plotted in X. Defaults to 0.
        dim2: the index corresponding to the 2nd dimension to be plotted in X. Defaults to 1.
        dim3: the index corresponding to the 3rd dimension to be plotted in X. Defaults to None.
        dims: a list of indices of the dimensions. Would override dim1/2/3 specified above. Defaults to None.
        create_figure: whether to create a new figure for the plot. Defaults to False.
        figsize: the size of the figure. Defaults to (6, 6).
        sort_by_c: how the colors and corresponding points would be sorted. Can be one of "raw", "neg", and "abs" and
            the data and color would be sorted based on the color scalar's original value, negative value, and absolute
            value, respectively. Defaults to "raw".
        save_show_or_return: whether to save, show or return the figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the
                {
                    "path": None,
                    "prefix": 'phase_portraits',
                    "dpi": None,
                    "ext": 'pdf',
                    "transparent": True,
                    "close": True,
                    "verbose": True
                }
            as its parameters. Otherwise, you can provide a dictionary that properly modify those keys according to
            your needs. Defaults to {}.
        **kwargs: any other kwargs to be passed to `plt.scatter`.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be `return`, the matplotlib axis of the
        plot would be returned.
    """

    if create_figure:
        plt.figure(figsize=figsize)

    if dims is not None:
        dim1 = dims[0]
        dim2 = dims[1]
        if len(dims) > 2:
            dim3 = dims[2]

    x, y = X[:, dim1], X[:, dim2]
    c = kwargs.pop("c", None)
    if c is not None and isarray(c):
        if sort_by_c is not None:
            if sort_by_c == "neg":
                i_sort = np.argsort(-c)
            elif sort_by_c == "abs":
                i_sort = np.argsort(np.abs(c))
            elif sort_by_c == "raw":
                i_sort = np.argsort(c)
            x = x[i_sort]
            y = y[i_sort]
            c = c[i_sort]
            if dim3 is not None:
                z = X[:, dim3][i_sort]
    if dim3 is None:
        plt.scatter(x, y, c=c, **kwargs)
    else:
        from mpl_toolkits.mplot3d import Axes3D

        plt.gcf().add_subplot(111, projection="3d")
        plt.gca().scatter(x, y, z, c=c, **kwargs)

    return save_show_ret("plot_X", save_show_or_return, save_kwargs, plt.gca())


def plot_V(
    X: np.ndarray,
    V: np.ndarray,
    dim1: int = 0,
    dim2: int = 1,
    dims: Optional[List[int]] = None,
    create_figure: bool = False,
    figsize: Tuple[float, float] = (6, 6),
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: dict = {},
    **kwargs,
) -> None:
    """Plot quiver graph (vector arrow graph) with given vectors.

    Args:
        X: the array containing the origins of the vectors.
        V: the array containing the vectors.
        dim1: the column index of the array that would be plotted as X-coordinates. Defaults to 0.
        dim2: the column index of the array that would be plotted as X-coordinates. Defaults to 1.
        dims: a two-item list containing dim1 and dim2. This argument would override dim1 and dim2. Defaults to None.
        create_figure: whether to create a new figure. Defaults to False.
        figsize: the size of the figure. Defaults to (6, 6).
        save_show_or_return: whether to save, show or return the figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the
                {
                    "path": None,
                    "prefix": 'phase_portraits',
                    "dpi": None,
                    "ext": 'pdf',
                    "transparent": True,
                    "close": True,
                    "verbose": True
                }
            as its parameters. Otherwise, you can provide a dictionary that properly modify those keys according to
            your needs. Defaults to {}.
        **kwargs: any other kwargs that would be passed to `plt.quiver`.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be `return`, the matplotlib axis of the
        plot would be returned.
    """

    if create_figure:
        plt.figure(figsize=figsize)
    if dims is not None:
        dim1 = dims[0]
        dim2 = dims[1]
    plt.quiver(X[:, dim1], X[:, dim2], V[:, dim1], V[:, dim2], **kwargs)

    return save_show_ret("plot_V", save_show_or_return, save_kwargs, plt.gca())


def zscatter(
    adata: AnnData,
    basis: str = "umap",
    layer: Optional[str] = "X",
    dim1: int = 0,
    dim2: int = 1,
    dim3: Optional[int] = None,
    color: Union[np.ndarray, str, None] = None,
    c_layer: Optional[str] = None,
    cbar: bool = True,
    cbar_shrink: float = 0.4,
    sym_c: bool = False,
    axis_off: bool = True,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: dict = {},
    **kwargs,
) -> None:
    """Plot scatter graph for a given AnnData object.

    Args:
        adata: an AnnData object.
        basis: the basis used for dimension reduction. It would be used to construct the key to find data in adata.obsm.
            Defaults to "umap".
        layer: the layer key the dimensional-reduced data is related with. It would be used to construct the key to find
            data in adata.obsm. Defaults to "X".
        dim1: the index of the array corresponding to the 1st dimension to be plotted. Defaults to 0.
        dim2: the index of the array corresponding to the 2nd dimension to be plotted. Defaults to 1.
        dim3: the index corresponding to the 3rd dimension to be plottedX. Defaults to None.
        color: specifying how to color the points. If it is a string, it would be considered as a key of adata.var and
            its corresponding value would be used to color the cells. If it is an array, it would be used to color the
            cells directly. If it is None, the color would be determined automatically. Defaults to None.
        c_layer: the layer of the AnnData object to be plotted. If None, adata.X would be used. Defaults to None.
        cbar: whether to show the color bar. Defaults to True.
        cbar_shrink: size factor of the color bar. Defaults to 0.4.
        sym_c: whether to make the color bar symmetric to 0. Defaults to False.
        axis_off: whether to turn of the axis in the graph. Defaults to True.
        save_show_or_return: whether to save, show or return the figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the
                {
                    "path": None,
                    "prefix": 'phase_portraits',
                    "dpi": None,
                    "ext": 'pdf',
                    "transparent": True,
                    "close": True,
                    "verbose": True
                }
            as its parameters. Otherwise, you can provide a dictionary that properly modify those keys according to
            your needs. Defaults to {}.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be `return`, the matplotlib axis of the
        plot would be returned.
    """

    if layer is None or len(layer) == 0:
        emb = basis
    else:
        emb = "%s_%s" % (layer, basis)
    X = adata.obsm[emb]
    title = None
    if not isarray(color):
        if color in adata.var.index:
            title = color
            if c_layer is None:
                color = flatten(index_gene(adata, adata.X, color))
            else:
                color = flatten(index_gene(adata, adata.layers[c_layer], color))
        elif color in adata.obs.keys():
            title = color
            # color = flatten(np.array(adata.obs[color]))
            color = adata.obs[color]

    # categorical data
    if (
        color is not None
        and type(color) is not str
        # and np.any([type(a) is str for a in color])
        and (areinstance(color, [str, bytes], np.any))
    ):
        cat_color = True
        try:
            cat = color.cat.categories
        except:
            cat = np.unique(color)
        value_dict = {c: i for i, c in enumerate(cat)}
        if title + "_colors" in adata.uns.keys():
            color_dict = adata.uns[title + "_colors"]
            if type(color_dict) is dict:
                color_map = ListedColormap([color_dict[c] for c in cat])
            else:
                if areinstance(color_dict, bytes):
                    color_dict = [c.decode() for c in color_dict]
                color_map = ListedColormap(color_dict)
        else:
            color_map = cm.get_cmap("tab20")
        color = np.array([value_dict[i] for i in color])

    else:
        cat_color = False
        color_map = None

    if color_map is None:
        plot_X(X, dim1=dim1, dim2=dim2, dim3=dim3, c=color, save_show_or_return="return", **kwargs)
    else:
        plot_X(
            X,
            dim1=dim1,
            dim2=dim2,
            dim3=dim3,
            c=color,
            cmap=color_map,
            save_show_or_return="return",
            **kwargs,
        )
    if isarray(color):
        if cbar:
            if cat_color:
                cb = plt.colorbar(
                    ticks=[i for i in value_dict.values()],
                    values=[i for i in value_dict.values()],
                    shrink=cbar_shrink,
                )
                cb.ax.set_yticklabels(value_dict.keys())
            else:
                plt.colorbar(shrink=cbar_shrink)
        if sym_c:
            bounds = max(np.abs(np.nanmax(color)), np.abs(np.nanmin(color)))
            bounds = bounds * np.array([-1, 1])
            plt.clim(bounds[0], bounds[1])
    if title is not None:
        plt.title(title)

    if axis_off:
        plt.axis("off")

    return save_show_ret("zscatter", save_show_or_return, save_kwargs, plt.gca())


def zstreamline(
    adata: AnnData,
    basis: str = "umap",
    v_basis: Optional[str] = None,
    x_layer: str = "X",
    v_layer: str = "velocity",
    dim1: int = 0,
    dim2: int = 1,
    dims: Optional[List[int]] = None,
    color: Union[List[str], str, None] = "k",
    create_figure: bool = False,
    figsize: Tuple[float, float] = (6, 4),
    grid_num: int = 50,
    smoothness: float = 1,
    min_vel_mag: Optional[np.ndarray] = None,
    cutoff: float = 1.5,
    return_grid: bool = False,
    linewidth: float = 1,
    constant_lw: bool = False,
    density: float = 1,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: dict = {},
    **streamline_kwargs,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Plot streamline graph with given AnnData object.

    Args:
        adata: an AnnData object.
        basis: the basis used for dimension reduction. It would be used to construct the key to find data in adata.obsm.
            Defaults to "umap".
        v_basis: the basis used for dimension reduction of velocity data if it is different from `basis`. Defaults to
            None.
        x_layer: the layer key the dimensional-reduced data is related with. It would be used to construct the key to
            find data in adata.obsm. Defaults to "X".
        v_layer: the layer key the dimensional-reduced velocity data is related with. It would be used to construct the
            key to find data in adata.obsm. Defaults to "velocity".
        dim1: the index of the array corresponding to the 1st dimension to be plotted. Defaults to 0.
        dim2: the index of the array corresponding to the 2nd dimension to be plotted. Defaults to 1.
        dims: a list containing dim1 and dim2. It would override dim1 and dim2 specified beforehand. Defaults to None.
        color: the color of the streamline. If it is a string, it would be treated as a matplotlib color. If it is an
            array, it should be a series of matplotlib colors corresponding to each grid. If it is None, the color would
            be determined automatically. Defaults to "k".
        create_figure: whether to create a new figure. Defaults to False.
        figsize: the size of the figure. Defaults to (6, 4).
        grid_num: the number of grids. Defaults to 50.
        smoothness: the factor to smooth the streamline. Defaults to 1.
        min_vel_mag: the minimum velocity to be shown in the graph. Defaults to None.
        cutoff: cutoff coefficient for project velocities to grids. Defaults to 1.5.
        return_grid: whether to return the X and V grids. Defaults to False.
        linewidth: the base width of the streamlines. Defaults to 1.
        constant_lw: whether to keep the streamlines having same width or to make the width vary corresponding to local
            velocity. Defaults to False.
        density: density of the stream plot. Refer to `pyplot.streamplot` for more details. Defaults to 1.
        save_show_or_return: whether to save, show or return the figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the
                {
                    "path": None,
                    "prefix": 'phase_portraits',
                    "dpi": None,
                    "ext": 'pdf',
                    "transparent": True,
                    "close": True,
                    "verbose": True
                }
            as its parameters. Otherwise, you can provide a dictionary that properly modify those keys according to
            your needs. Defaults to {}.
        **streamline_kwargs: any other kwargs to be passed to `pyplot.streamplot`.

    Returns:
        None would be returned in default. if `return_grid` is set to be True, the grids of X and V would be returned.
    """

    if x_layer is None or len(x_layer) == 0:
        emb = basis
    else:
        emb = "%s_%s" % (x_layer, basis)
    v_basis = basis if v_basis is None else v_basis
    if v_layer is None or len(v_layer) == 0:
        v_emb = v_basis
    else:
        v_emb = "%s_%s" % (v_layer, v_basis)

    if dims is not None:
        dim1 = dims[0]
        dim2 = dims[1]

    X = adata.obsm[emb][:, [dim1, dim2]]
    V = adata.obsm[v_emb][:, [dim1, dim2]]

    # set up grids
    # if np.isscalar(grid_num):
    #    grid_num = grid_num * np.ones(2)
    # V_grid, X_grid = smoothen_drift_on_grid(X, V, n_grid=grid_num, smoothness=smoothness)
    # V_grid, X_grid = V_grid.T, X_grid.T
    X_grid, V_grid = velocity_on_grid(X, V, n_grids=grid_num, smoothness=smoothness, cutoff_coeff=cutoff)
    V_grid, X_grid = V_grid.T, X_grid.T

    streamplot_kwargs = {
        "density": density * 2,
        "arrowsize": 1,
        "arrowstyle": "fancy",
        "minlength": 0.1,
        "maxlength": 4.0,
        "integration_direction": "both",
        "zorder": 3,
    }

    mass = np.sqrt((V_grid**2).sum(0))
    # velocity filtering
    if min_vel_mag is not None:
        min_vel_mag = np.clip(min_vel_mag, None, np.quantile(mass, 0.4))
        mass[mass < min_vel_mag] = np.nan

    if not constant_lw:
        linewidth *= 2 * mass / mass[~np.isnan(mass)].max()
        linewidth = linewidth.reshape(grid_num, grid_num)
    streamplot_kwargs.update({"linewidth": linewidth})
    streamplot_kwargs.update(streamline_kwargs)

    if np.isscalar(grid_num):
        gnum = grid_num * np.ones(2, dtype=int)
    else:
        gnum = grid_num
    x = X_grid[0].reshape(gnum[0], gnum[1])
    y = X_grid[1].reshape(gnum[0], gnum[1])
    u = V_grid[0].reshape(gnum[0], gnum[1])
    v = V_grid[1].reshape(gnum[0], gnum[1])
    if create_figure:
        plt.figure(figsize=figsize)
    plt.streamplot(x, y, u, v, color=color, **streamplot_kwargs)
    # plt.set_arrow_alpha(axes_list[i], streamline_alpha)
    # set_stream_line_alpha(s, streamline_alpha)
    if return_grid:
        return X_grid.T, V_grid.T

    return save_show_ret("zstreamline", save_show_or_return, save_kwargs, plt.gca())


def multiplot(
    plot_func: Callable,
    arr: Iterable[Any],
    n_row: Optional[int] = None,
    n_col: int = 3,
    fig: Figure = None,
    subplot_size: Tuple[float, float] = (6, 4),
    save_show_or_return: Literal["save", "show", "return"] = "return",
    save_kwargs: dict = {},
) -> List[Axes]:
    """Plot multiple graphs with same plotting function but different inputs.

    Args:
        plot_func: the function to be used to plot.
        arr: the input for the plotting function. If `arr` is a dict, the key should be the names of the arguments and
            the value should be an iterable object and its items would be passed to the function. If each item of it is
            an array, each item would be destructed and passed to the plotting function. Otherwise, each item would be
            directly passed to the function.
        n_row: the number of rows of the subplots. Defaults to None.
        n_col: the number of columns of the subplots. If both `n_row` and `n_col` are None, `n_col` would be set to 3
            and `n_row` would be calculated automatically. If either `n_row` or `n_col` is specified, the other
            parameter would be calculated so that all subplots can be shown. If both are specified, only first
            `n_col x n_row` subplots would be shown. Defaults to 3.
        fig: the figure to plot on. If None, a new figure would be created. Defaults to None.
        subplot_size: the size of each subplot. Defaults to (6, 4).
        save_show_or_return: whether to save, show or return the figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the
                {
                    "path": None,
                    "prefix": 'phase_portraits',
                    "dpi": None,
                    "ext": 'pdf',
                    "transparent": True,
                    "close": True,
                    "verbose": True
                }
            as its parameters. Otherwise, you can provide a dictionary that properly modify those keys according to
            your needs. Defaults to {}.

    Returns:
        The axes of the subplots by default.
    """

    if n_col is None and n_row is None:
        n_col = 3
    n = len(arr[list(arr.keys())[0]]) if type(arr) is dict else len(arr)
    if n_row is None:
        n_row = int(np.ceil(n / n_col))
    elif n_col is None:
        n_col = int(np.ceil(n / n_row))
    else:
        # only the first n plots will be plotted
        n = min(n_row * n_col, n)

    if fig is None:
        figsize = (subplot_size[0] * n_col, subplot_size[1] * n_row)
        fig = plt.figure(figsize=figsize)
    ax_list = []
    for i in range(n):
        ax_list.append(fig.add_subplot(n_row, n_col, i + 1))
        if type(arr) is dict:
            pdict = {key: value[i] for key, value in arr.items()}
            plot_func(**pdict)
        elif isarray(arr[i]):
            plot_func(*arr[i])
        else:
            plot_func(arr[i])
    return save_show_ret("multiplot", save_show_or_return, save_kwargs, ax_list)


def plot_jacobian_gene(
    adata: AnnData,
    jkey: str = "jacobian",
    basis: str = "pca",
    regulators: Optional[Iterable[str]] = None,
    effectors: Optional[Iterable[str]] = None,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: dict = {},
    **kwargs,
) -> None:
    """Plot scatter graphs for gene's jacobians to show relationship between the regulators and effectors.

    Args:
        adata: an AnnData object.
        jkey: the key for jacobian data stored in adata.uns. Defaults to "jacobian".
        basis: the basis of dimension reduction. It would be used to construct the key to find data in adata.uns.
            Defaults to "pca".
        regulators: the regulator genes to be considered. Defaults to None.
        effectors: the effector genes to be considered. Defaults to None.
        save_show_or_return: whether to save, show or return the figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the
                {
                    "path": None,
                    "prefix": 'phase_portraits',
                    "dpi": None,
                    "ext": 'pdf',
                    "transparent": True,
                    "close": True,
                    "verbose": True
                }
            as its parameters. Otherwise, you can provide a dictionary that properly modify those keys according to
            your needs. Defaults to {}.

    Returns:
        None would be returned in default. If `save_show_or_return` is set to be "return", the axes of the subplots.
    """

    jkey = f"{jkey}_{basis}" if basis is not None else jkey
    J_dict = adata.uns[jkey]
    c_arr = []
    ti_arr = []
    for i, reg in enumerate(J_dict["regulators"]):
        if regulators is None or reg in regulators:
            for j, eff in enumerate(J_dict["effectors"]):
                if effectors is None or eff in effectors:
                    c_arr.append(J_dict["jacobian_gene"][j, i, :])
                    ti_arr.append(f"{eff} wrt. {reg}")
    ax_list = multiplot(
        lambda c, ti: [zscatter(adata, color=c, save_show_or_return="return", **kwargs), plt.title(ti)],
        {"c": c_arr, "ti": ti_arr},
        n_col=2,
        subplot_size=(8, 4),
    )
    return save_show_ret("plot_jacobian_gene", save_show_or_return, save_kwargs, ax_list)
