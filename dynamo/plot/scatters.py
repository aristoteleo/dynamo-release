# code adapted from https://github.com/lmcinnes/umap/blob/7e051d8f3c4adca90ca81eb45f6a9d1372c076cf/umap/plot.py
import warnings
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata
import matplotlib.cm
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import patches, rcParams
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.colors import rgb2hex, to_hex
from pandas.api.types import is_categorical_dtype

from ..configuration import _themes
from ..docrep import DocstringProcessor
from ..dynamo_logger import main_debug, main_info, main_warning
from ..preprocessing.utils import affine_transform, gen_rotation_2d
from ..tools.moments import calc_1nd_moment
from ..tools.utils import flatten, get_mapper, get_vel_params, update_dict, update_vel_params
from .utils import (
    _datashade_points,
    _get_adata_color_vec,
    _matplotlib_points,
    _select_font_color,
    arrowed_spines,
    calculate_colors,
    deaxis_all,
    despline_all,
    is_cell_anno_column,
    is_gene_name,
    is_layer_keys,
    is_list_of_lists,
    retrieve_plot_save_path,
    save_show_ret,
    save_plotly_figure,
    save_pyvista_plotter,
)

docstrings = DocstringProcessor()


@docstrings.get_sectionsf("scatters")
def scatters(
    adata: AnnData,
    basis: str = "umap",
    x: int = 0,
    y: int = 1,
    z: int = 2,
    color: str = "ntr",
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
    background: Optional[str] = None,
    ncols: int = 4,
    pointsize: Optional[float] = None,
    figsize: Tuple[float, float] = (6, 4),
    show_legend: str = "on data",
    use_smoothed: bool = True,
    aggregate: Optional[str] = None,
    show_arrowed_spines: bool = False,
    ax: Optional[Axes] = None,
    sort: Literal["raw", "abs", "neg"] = "raw",
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Dict[str, Any] = {},
    return_all: bool = False,
    add_gamma_fit: bool = False,
    frontier: bool = False,
    contour: bool = False,
    ccmap: Optional[str] = None,
    alpha: float = 0.1,
    calpha: float = 0.4,
    sym_c: bool = False,
    smooth: bool = False,
    dpi: int = 100,
    inset_dict: Dict[str, Any] = {},
    marker: Optional[str] = None,
    group: Optional[str] = None,
    add_group_gamma_fit: bool = False,
    affine_transform_degree: Optional[int] = None,
    affine_transform_A: Optional[float] = None,
    affine_transform_b: Optional[float] = None,
    stack_colors: bool = False,
    stack_colors_threshold: float = 0.001,
    stack_colors_title: str = "stacked colors",
    stack_colors_legend_size: float = 2,
    stack_colors_cmaps: Optional[List[str]] = None,
    despline: bool = True,
    deaxis: bool = True,
    despline_sides: Optional[List[str]] = None,
    projection: str = "2d",
    **kwargs,
) -> Union[
    Axes,
    List[Axes],
    Tuple[Axes, List[str], Literal["white", "black"]],
    Tuple[List[Axes], List[str], Literal["white", "black"]],
    None,
]:
    """Plot an embedding as points. Currently this only works for 2D embeddings. While there are many optional
    parameters to further control and tailor the plotting, you need only pass in the trained/fit umap model to get
    results. This plot utility will attempt to do the hard work of avoiding overplotting issues, and make it easy to
    automatically color points by a categorical labelling or numeric values. This method is intended to be used within a
    Jupyter notebook with `%matplotlib inline`.

    Args:
        adata: an AnnData object.
        basis: the reduced dimension stored in adata.obsm. The specific basis key will be constructed in the following
            priority if exits: 1) specific layer input +  basis 2) X_ + basis 3) basis. E.g. if basis is PCA, `scatters`
            is going to look for 1) if specific layer is spliced, `spliced_pca` 2) `X_pca` (dynamo convention) 3) `pca`.
            Defaults to "umap".
        x: the column index of the low dimensional embedding for the x-axis. Defaults to 0.
        y: the column index of the low dimensional embedding for the y-axis. Defaults to 1.
        z: the column index of the low dimensional embedding for the z-axis. Defaults to 2.
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
        aggregate: the column in adata.obs that will be used to aggregate data points. Defaults to None.
        show_arrowed_spines: whether to show a pair of arrowed spines representing the basis of the scatter is currently
            using. Defaults to False.
        ax: the matplotlib axes object where new plots will be added to. Only applicable to drawing a single component.
            Defaults to None.
        sort: the method to reorder data so that high values points will be on top of background points. Can be one of
            {'raw', 'abs', 'neg'}, i.e. sorted by raw data, sort by absolute values or sort by negative values. Defaults
            to "raw".
        save_show_or_return: whether to save, show or return the figure. If "both", it will save and plot the figure at
            the same time. If "all", the figure will be saved, displayed and the associated axis and other object will
            be return. Defaults to "show".
        save_kwargs: A dictionary that will passed to the save_show_ret function. By default it is an empty dictionary and
            the save_show_ret function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.
        return_all: whether to return all the scatter related variables. Defaults to False.
        add_gamma_fit: whether to add the line of the gamma fitting. This will automatically turn on if `basis` points
            to gene names and those genes have went through gamma fitting. Defaults to False.
        frontier: whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to
            show area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib
            tips & tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from
            scEU-seq paper: https://science.sciencemag.org/content/367/6482/1151. If `contour` is set  to be True,
            `frontier` will be ignored as `contour` also add an outlier for data points. Defaults to False.
        contour: whether to add an countor on top of scatter plots. We use tricontourf to plot contour for non-gridded
            data. The shapely package was used to create a polygon of the concave hull of the scatters. With the polygon
            we then check if the mean of the triangulated points are within the polygon and use this as our condition to
            form the mask to create the contour. We also add the polygon shape as a frontier of the data point (similar
            to when setting `frontier = True`). When the color of the data points is continuous, we will use the same
            cmap as for the scatter points by default, when color is categorical, no contour will be drawn but just the
            polygon. cmap can be set with `ccmap` argument. See below. This has recently changed to use seaborn's
            kdeplot. Defaults to False.
        ccmap: the name of a matplotlib colormap to use for coloring or shading points the contour. See above.
            Defaults to None.
        alpha: the point's alpha (transparency) value. Defaults to 0.1.
        calpha: contour alpha value passed into sns.kdeplot. The value should be inbetween [0, 1]. Defaults to 0.4.
        sym_c: whether do you want to make the limits of continuous color to be symmetric, normally this should be used
            for plotting velocity, jacobian, curl, divergence or other types of data with both positive or negative
            values. Defaults to False.
        smooth: whether do you want to further smooth data and how much smoothing do you want. If it is `False`, no
            smoothing will be applied. If `True`, smoothing based on one step diffusion of connectivity matrix
            (`.uns['moment_cnn']`) will be applied. If a number larger than 1, smoothing will based on `smooth` steps of
            diffusion.
        dpi: the resolution of the figure in dots-per-inch. Dots per inches (dpi) determines how many pixels the figure
            comprises. dpi is different from ppi or points per inches. Note that most elements like lines, markers,
            texts have a size given in points so you can convert the points to inches. Matplotlib figures use Points per
            inch (ppi) of 72. A line with thickness 1 point will be 1./72. inch wide. A text with fontsize 12 points
            will be 12./72. inch heigh. Of course if you change the figure size in inches, points will not change, so a
            larger figure in inches still has the same size of the elements.Changing the figure size is thus like taking
            a piece of paper of a different size. Doing so, would of course not change the width of the line drawn with
            the same pen. On the other hand, changing the dpi scales those elements. At 72 dpi, a line of 1 point size
            is one pixel strong. At 144 dpi, this line is 2 pixels strong. A larger dpi will therefore act like a
            magnifying glass. All elements are scaled by the magnifying power of the lens. see more details at answer 2
            by @ImportanceOfBeingErnest:
            https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size. Defaults to 100.
        inset_dict: a  dictionary of parameters in inset_ax. Example, something like {"width": "5%", "height": "50%", "loc":
            'lower left', "bbox_to_anchor": (0.85, 0.90, 0.145, 0.145), "bbox_transform": ax.transAxes, "borderpad": 0}
            See more details at https://matplotlib.org/api/_as_gen/mpl_toolkits.axes_grid1.inset_locator.inset_axes.html
            or https://stackoverflow.com/questions/39803385/what-does-a-4-element-tuple-argument-for-bbox-to-anchor-mean-in-matplotlib.
            Defaults to {}.
        marker: the marker style. marker can be either an instance of the class or the text shorthand for a particular
            marker. See matplotlib.markers for more information about marker styles. Defaults to None.
        group: the key in `adata.obs` corresponding to the cell group data. Defaults to None.
        add_group_gamma_fit: whether to plot the cell group's gamma fit results. Defaults to False.
        affine_transform_degree: transform coordinates of points according to some degree. Defaults to None.
        affine_transform_A: coefficients in affine transformation Ax + b. 2D for now. Defaults to None.
        affine_transform_b: bias in affine transformation Ax + b. Defaults to None.
        stack_colors: whether to stack all color on the same ax passed above. Currently only support 18 sequential
            matplotlib default cmaps assigning to different color groups. (#colors should be smaller than 18, reuse if
            #colors > 18. TODO generate cmaps according to #colors). Defaults to False.
        stack_colors_threshold: a threshold for filtering out points values < threshold when drawing each color. E.g. if
            you do not want points with values < 1 showing up on axis, set threshold to be 1. Defaults to 0.001.
        stack_colors_title: the title for the stack_color plot. Defaults to "stacked colors".
        stack_colors_legend_size: the legend size in stack color plot. Defaults to 2.
        stack_colors_cmaps: a list of cmaps that will be used to map values to color when stacking colors on the same
            subplot. The order corresponds to the order of color. Defaults to None.
        despline: whether to remove splines of the figure. Defaults to True.
        deaxis: whether to remove axis ticks of the figure. Defaults to True.
        despline_sides: which side of splines should be removed. Can be any combination of `["bottom", "right", "top", "left"]`. Defaults to None.
        projection: the projection property of the matplotlib.Axes. Defaults to "2d".
        **kwargs: any other kwargs that would be passed to `pyplot.scatters`.

    Raises:
        ValueError: invalid adata object: lacking of required layers.
        ValueError: `basis` not found in `adata.obsm`.
        ValueError: invalid `x` or `y`.
        ValueError: `labels` and `values` conflicted.
        ValueError: invalid velocity estimation in `adata`.
        ValueError: invalid velocity estimation in `adata`.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return' or 'all', the matplotlib axes
        object of the generated plots would be returned. If `return_all` is set to be true, the list of colors used and
        the font color would also be returned.
    """

    import matplotlib.pyplot as plt

    # 2d is not a projection in matplotlib, default is None (rectilinear)
    if projection == "2d":
        projection = None
    if calpha < 0 or calpha > 1:
        main_warning(
            "calpha=%f is invalid (smaller than 0 or larger than 1) and may cause potential issues. Please check."
            % (calpha)
        )
    group_colors = ["b", "g", "r", "c", "m", "y", "k", "w"]

    if stack_colors and stack_colors_cmaps is None:
        main_info("using default stack colors")
        stack_colors_cmaps = [
            "Greys",
            "Purples",
            "Blues",
            "Greens",
            "Oranges",
            "Reds",
            "YlOrBr",
            "YlOrRd",
            "OrRd",
            "PuRd",
            "RdPu",
            "BuPu",
            "GnBu",
            "PuBu",
            "YlGnBu",
            "PuBuGn",
            "BuGn",
            "YlGn",
        ]
    stack_legend_handles = []
    if stack_colors:
        color_key = None

    if not (affine_transform_degree is None):
        affine_transform_A = gen_rotation_2d(affine_transform_degree)
        affine_transform_b = 0

    if contour:
        frontier = False

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
        # if save_show_or_return != 'save': set_figure_params('dynamo', background=_background)
    else:
        _background = background
        # if save_show_or_return != 'save': set_figure_params('dynamo', background=_background)
    if type(x) in [int, str]:
        x = [x]
    if type(y) in [int, str]:
        y = [y]
    if type(z) in [int, str]:
        z = [z]

    if all([is_gene_name(adata, i) for i in basis]):
        if x[0] not in ["M_s", "X_spliced", "M_t", "X_total", "spliced", "total"] and y[0] not in [
            "M_u",
            "X_unspliced",
            "M_n",
            "X_new",
            "unspliced",
            "new",
        ]:
            if "M_t" in adata.layers.keys() and "M_n" in adata.layers.keys():
                x, y = ["M_t"], ["M_n"]
            elif "X_total" in adata.layers.keys() and "X_new" in adata.layers.keys():
                x, y = ["X_total"], ["X_new"]
            elif "M_s" in adata.layers.keys() and "M_u" in adata.layers.keys():
                x, y = ["M_s"], ["M_u"]
            elif "X_spliced" in adata.layers.keys() and "X_unspliced" in adata.layers.keys():
                x, y = ["X_spliced"], ["X_unspliced"]
            elif "spliced" in adata.layers.keys() and "unspliced" in adata.layers.keys():
                x, y = ["spliced"], ["unspliced"]
            elif "total" in adata.layers.keys() and "new" in adata.layers.keys():
                x, y = ["total"], ["new"]
            else:
                raise ValueError(
                    "your adata object is corrupted. Please make sure it has at least one of the following "
                    "pair of layers:"
                    "'M_s', 'X_spliced', 'M_t', 'X_total', 'spliced', 'total' and "
                    "'M_u', 'X_unspliced', 'M_n', 'X_new', 'unspliced', 'new'. "
                )

    if use_smoothed:
        mapper = get_mapper()

    # check color, layer, basis -> convert to list
    if type(color) is str:
        color = [color]
    if type(layer) is str:
        layer = [layer]
    if type(basis) is str:
        basis = [basis]

    if stack_colors and len(color) > len(stack_colors_cmaps):
        main_warning(
            "#color: %d passed in is greater than #sequential cmaps: %d, will reuse sequential maps"
            % (len(color), len(stack_colors_cmaps))
        )
        main_warning("You should consider decreasing your #color")

    n_c, n_l, n_b, n_x, n_y = (
        1 if color is None else len(color),
        1 if layer is None else len(layer),
        1 if basis is None else len(basis),
        1 if x is None else 1 if type(x) in [anndata._core.views.ArrayView, np.ndarray] else len(x),
        # check whether it is an array
        1 if y is None else 1 if type(y) in [anndata._core.views.ArrayView, np.ndarray] else len(y),
        # check whether it is an array
    )

    if pointsize is None:
        point_size = 16000.0 / np.sqrt(adata.shape[0])
    else:
        point_size = 16000.0 / np.sqrt(adata.shape[0]) * pointsize

    scatter_kwargs = dict(
        alpha=alpha,
        s=point_size,
        edgecolor=None,
        linewidth=0,
        rasterized=True,
        marker=marker,
    )  # (0, 0, 0, 1)
    if kwargs is not None:
        scatter_kwargs.update(kwargs)

    font_color = _select_font_color(_background)

    total_panels, ncols = (
        n_c * n_l * n_b * n_x * n_y,
        min(max([n_c, n_l, n_b, n_x, n_y]), ncols),
    )
    nrow, ncol = int(np.ceil(total_panels / ncols)), ncols
    if figsize is None:
        figsize = plt.rcParams["figsize"]

    figure = None  # possible as argument in future

    # if #total_panel is 1, `_matplotlib_points` will create a figure. No need to create a figure here and generate a blank figure.
    if total_panels > 1 and ax is None:
        figure = plt.figure(
            None,
            (figsize[0] * ncol, figsize[1] * nrow),
            facecolor=_background,
            dpi=dpi,
        )
        gs = plt.GridSpec(nrow, ncol, wspace=0.12)

    ax_index = 0
    axes_list, color_list = [], []
    color_out = None

    def _plot_basis_layer(cur_b, cur_l):
        """a helper function for plotting a specific basis/layer data

        Parameters
        ----------
        cur_b :
            current basis
        cur_l :
            current layer
        """
        nonlocal adata, x, y, z, _background, cmap, color_out, labels, values, ax, sym_c, scatter_kwargs, ax_index

        if cur_l in ["acceleration", "curvature", "divergence", "velocity_S", "velocity_T"]:
            cur_l_smoothed = cur_l
            cmap, sym_c = "bwr", True  # TODO maybe use other divergent color map in the future
        else:
            if use_smoothed:
                cur_l_smoothed = cur_l if cur_l.startswith("M_") | cur_l.startswith("velocity") else mapper[cur_l]
                if cur_l.startswith("velocity"):
                    cmap, sym_c = "bwr", True

        if cur_l + "_" + cur_b in adata.obsm.keys():
            prefix = cur_l + "_"
        elif ("X_" + cur_b) in adata.obsm.keys():
            prefix = "X_"
        elif cur_b in adata.obsm.keys():
            # special case for spatial for compatibility with other packages
            prefix = ""
        else:
            raise ValueError("Please check if basis=%s exists in adata.obsm" % basis)

        basis_key = prefix + cur_b
        main_info("plotting with basis key=%s" % basis_key, indent_level=2)

        # if basis_key in adata.obsm.keys():
        #     if type(x) != str and type(y) != str:
        #         x_, y_ = (
        #             adata.obsm[basis_key][:, int(x)],
        #             adata.obsm[basis_key][:, int(y)],
        #         )
        # else:
        #     continue
        if stack_colors:
            _stack_background_adata_indices = np.ones(len(adata), dtype=bool)

        for cur_c in color:
            main_debug("coloring scatter of cur_c: %s" % str(cur_c))
            if not stack_colors:
                cur_title = cur_c
            else:
                cur_title = stack_colors_title
            _color = _get_adata_color_vec(adata, cur_l, cur_c)

            # select data rows based on stack color thresholding
            is_numeric_color = np.issubdtype(_color.dtype, np.number)
            if not is_numeric_color:
                main_info(
                    "skip filtering %s by stack threshold when stacking color because it is not a numeric type"
                    % (cur_c),
                    indent_level=2,
                )
            _values = values
            if stack_colors and is_numeric_color:
                main_debug("Subsetting adata by stack_colors")
                _adata = adata
                _adata = adata[_color > stack_colors_threshold]
                _stack_background_adata_indices = np.logical_and(
                    _stack_background_adata_indices, (_color < stack_colors_threshold)
                )
                if values:
                    _values = values[_color > stack_colors_threshold]
                _color = _color[_color > stack_colors_threshold]
                main_debug("stack colors: _adata len after thresholding by color value: %d" % (len(_adata)))
                if len(_color) == 0:
                    main_info("skipping color %s because no point of %s is above threshold" % (cur_c, cur_c))
                    continue
            else:
                _adata = adata

            # make x, y, z lists of list, where each list corresponds to one coordinate set
            if (
                type(x) in [anndata._core.views.ArrayView, np.ndarray]
                and type(y) in [anndata._core.views.ArrayView, np.ndarray]
                and len(x) == _adata.n_obs
                and len(y) == _adata.n_obs
            ):
                x, y = [x], [y]
                if projection == "3d":
                    z = [z]
                else:
                    z = [np.nan]

            elif hasattr(x, "__len__") and hasattr(y, "__len__"):
                x, y = list(x), list(y)
                if projection == "3d":
                    z = list(z)
                else:
                    z = [np.nan] * len(x)

            assert len(x) == len(y) and len(x) == len(z), "bug: x, y, z does not have the same shape."
            for cur_x, cur_y, cur_z in zip(x, y, z):  # here x / y are arrays
                main_debug("handling coordinates, cur_x: %s, cur_y: %s" % (cur_x, cur_y))
                if type(cur_x) is int and type(cur_y) is int:
                    x_col_name = cur_b + "_0"
                    y_col_name = cur_b + "_1"
                    z_col_name = cur_b + "_2"
                    points = None
                    points = pd.DataFrame(
                        {
                            x_col_name: _adata.obsm[basis_key][:, cur_x],
                            y_col_name: _adata.obsm[basis_key][:, cur_y],
                        }
                    )
                    points.columns = [x_col_name, y_col_name]

                    if projection == "3d":
                        points = pd.DataFrame(
                            {
                                x_col_name: _adata.obsm[basis_key][:, cur_x],
                                y_col_name: _adata.obsm[basis_key][:, cur_y],
                                z_col_name: _adata.obsm[basis_key][:, cur_z],
                            }
                        )
                        points.columns = [x_col_name, y_col_name, z_col_name]

                elif is_gene_name(_adata, cur_x) and is_gene_name(_adata, cur_y):
                    points = pd.DataFrame(
                        {
                            cur_x: _adata.obs_vector(k=cur_x, layer=None)
                            if cur_l_smoothed == "X"
                            else _adata.obs_vector(k=cur_x, layer=cur_l_smoothed),
                            cur_y: _adata.obs_vector(k=cur_y, layer=None)
                            if cur_l_smoothed == "X"
                            else _adata.obs_vector(k=cur_y, layer=cur_l_smoothed),
                        }
                    )
                    # points = points.loc[(points > 0).sum(1) > 1, :]
                    points.columns = [
                        cur_x + " (" + cur_l_smoothed + ")",
                        cur_y + " (" + cur_l_smoothed + ")",
                    ]
                    cur_title = cur_x + " VS " + cur_y
                elif is_cell_anno_column(_adata, cur_x) and is_cell_anno_column(_adata, cur_y):
                    points = pd.DataFrame(
                        {
                            cur_x: _adata.obs_vector(cur_x),
                            cur_y: _adata.obs_vector(cur_y),
                        }
                    )
                    points.columns = [cur_x, cur_y]
                    cur_title = cur_x + " VS " + cur_y
                elif is_cell_anno_column(_adata, cur_x) and is_gene_name(_adata, cur_y):
                    points = pd.DataFrame(
                        {
                            cur_x: _adata.obs_vector(cur_x),
                            cur_y: _adata.obs_vector(k=cur_y, layer=None)
                            if cur_l_smoothed == "X"
                            else _adata.obs_vector(k=cur_y, layer=cur_l_smoothed),
                        }
                    )
                    # points = points.loc[points.iloc[:, 1] > 0, :]
                    points.columns = [
                        cur_x,
                        cur_y + " (" + cur_l_smoothed + ")",
                    ]
                    cur_title = cur_y
                elif is_gene_name(_adata, cur_x) and is_cell_anno_column(_adata, cur_y):
                    points = pd.DataFrame(
                        {
                            cur_x: _adata.obs_vector(k=cur_x, layer=None)
                            if cur_l_smoothed == "X"
                            else _adata.obs_vector(k=cur_x, layer=cur_l_smoothed),
                            cur_y: _adata.obs_vector(cur_y),
                        }
                    )
                    # points = points.loc[points.iloc[:, 0] > 0, :]
                    points.columns = [
                        cur_x + " (" + cur_l_smoothed + ")",
                        cur_y,
                    ]
                    cur_title = cur_x
                elif is_layer_keys(_adata, cur_x) and is_layer_keys(_adata, cur_y):
                    cur_x_, cur_y_ = (
                        _adata[:, cur_b].layers[cur_x],
                        _adata[:, cur_b].layers[cur_y],
                    )
                    points = pd.DataFrame({cur_x: flatten(cur_x_), cur_y: flatten(cur_y_)})
                    # points = points.loc[points.iloc[:, 0] > 0, :]
                    points.columns = [cur_x, cur_y]
                    cur_title = cur_b
                elif type(cur_x) in [anndata._core.views.ArrayView, np.ndarray] and type(cur_y) in [
                    anndata._core.views.ArrayView,
                    np.ndarray,
                ]:
                    points = pd.DataFrame({"x": flatten(cur_x), "y": flatten(cur_y)})
                    points.columns = ["x", "y"]
                    cur_title = cur_b
                else:
                    raise ValueError("Make sure your `x` and `y` are integers, gene names, column names in .obs, etc.")

                if aggregate is not None:
                    groups, uniq_grp = (
                        _adata.obs[aggregate],
                        list(_adata.obs[aggregate].unique()),
                    )
                    group_color, group_median = (
                        np.zeros((1, len(uniq_grp))).flatten()
                        if isinstance(_color[0], Number)
                        else np.zeros((1, len(uniq_grp))).astype("str").flatten(),
                        np.zeros((len(uniq_grp), 2)),
                    )

                    grp_size = _adata.obs[aggregate].value_counts()[uniq_grp].values
                    scatter_kwargs = (
                        {"s": grp_size} if scatter_kwargs is None else update_dict(scatter_kwargs, {"s": grp_size})
                    )

                    for ind, cur_grp in enumerate(uniq_grp):
                        group_median[ind, :] = np.nanmedian(
                            points.iloc[np.where(groups == cur_grp)[0], :2],
                            0,
                        )
                        if isinstance(_color[0], Number):
                            group_color[ind] = np.nanmedian(np.array(_color)[np.where(groups == cur_grp)[0]])
                        else:
                            group_color[ind] = pd.Series(_color)[np.where(groups == cur_grp)[0]].value_counts().index[0]

                    points, _color = (
                        pd.DataFrame(
                            group_median,
                            index=uniq_grp,
                            columns=points.columns,
                        ),
                        group_color,
                    )
                # https://stackoverflow.com/questions/4187185/how-can-i-check-if-my-python-object-is-a-number
                # answer from Boris.
                is_not_continuous = not isinstance(_color[0], Number) or _color.dtype.name == "category"

                if is_not_continuous:
                    labels = np.asarray(_color) if is_categorical_dtype(_color) else _color
                    if theme is None:
                        if _background in ["#ffffff", "black"]:
                            _theme_ = "glasbey_dark"
                        else:
                            _theme_ = "glasbey_white"
                    else:
                        _theme_ = theme
                else:
                    _values = _color
                    if theme is None:
                        if _background in ["#ffffff", "black"]:
                            _theme_ = "inferno" if cur_l != "velocity" else "div_blue_black_red"
                        else:
                            _theme_ = "viridis" if not cur_l.startswith("velocity") else "div_blue_red"
                    else:
                        _theme_ = theme

                _cmap = _themes[_theme_]["cmap"] if cmap is None else cmap
                if stack_colors:
                    main_debug("stack colors: changing cmap")
                    _cmap = stack_colors_cmaps[ax_index % len(stack_colors_cmaps)]
                    max_color = matplotlib.cm.get_cmap(_cmap)(float("inf"))
                    legend_circle = Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=max_color,
                        label=cur_c,
                        markersize=stack_colors_legend_size,
                    )
                    stack_legend_handles.append(legend_circle)

                _color_key_cmap = _themes[_theme_]["color_key_cmap"] if color_key_cmap is None else color_key_cmap
                _background = _themes[_theme_]["background"] if _background is None else _background

                if labels is not None and values is not None:
                    raise ValueError("Conflicting options; only one of labels or values should be set")

                if total_panels > 1 and not stack_colors:
                    ax = plt.subplot(gs[ax_index], projection=projection)
                ax_index += 1

                # if highligts is a list of lists - each list is relate to each color element
                if highlights is not None:
                    if is_list_of_lists(highlights):
                        _highlights = highlights[color.index(cur_c)]
                        _highlights = _highlights if all([i in _color for i in _highlights]) else None
                    else:
                        _highlights = highlights if all([i in _color for i in highlights]) else None

                if smooth and not is_not_continuous:
                    main_debug("smooth and not continuous")
                    knn = _adata.obsp["moments_con"]
                    values = (
                        calc_1nd_moment(values, knn)[0]
                        if smooth in [1, True]
                        else calc_1nd_moment(values, knn**smooth)[0]
                    )

                if affine_transform_A is None or affine_transform_b is None:
                    point_coords = points.values
                else:
                    point_coords = affine_transform(points.values, affine_transform_A, affine_transform_b)

                if points.shape[0] <= figsize[0] * figsize[1] * 100000:
                    main_debug("drawing with _matplotlib_points function")
                    ax, color_out = _matplotlib_points(
                        # points.values,
                        point_coords,
                        ax,
                        labels,
                        _values,
                        highlights,
                        _cmap,
                        color_key,
                        _color_key_cmap,
                        _background,
                        figsize[0],
                        figsize[1],
                        show_legend,
                        sort=sort,
                        frontier=frontier,
                        contour=contour,
                        ccmap=ccmap,
                        calpha=calpha,
                        sym_c=sym_c,
                        inset_dict=inset_dict,
                        projection=projection,
                        **scatter_kwargs,
                    )
                    if labels is not None:
                        color_dict = {}
                        colors = [rgb2hex(i) for i in color_out]
                        for i, j in zip(labels, colors):
                            color_dict[i] = j

                        adata.uns[cur_title + "_colors"] = color_dict
                else:
                    main_debug("drawing with _datashade_points function")
                    ax = _datashade_points(
                        # points.values,
                        point_coords,
                        ax,
                        labels,
                        values,
                        highlights,
                        _cmap,
                        color_key,
                        _color_key_cmap,
                        _background,
                        figsize[0],
                        figsize[1],
                        show_legend,
                        sort=sort,
                        frontier=frontier,
                        contour=contour,
                        ccmap=ccmap,
                        calpha=calpha,
                        sym_c=sym_c,
                        **scatter_kwargs,
                    )

                if ax_index == 1 and show_arrowed_spines:
                    arrowed_spines(ax, points.columns[:2], _background)
                else:
                    if despline:
                        despline_all(ax, despline_sides)
                    if deaxis:
                        deaxis_all(ax)

                ax.set_title(cur_title)

                axes_list.append(ax)
                color_list.append(color_out)

                labels, values = None, None  # reset labels and values

                if add_gamma_fit and cur_b in _adata.var_names[_adata.var.use_for_dynamics]:
                    xnew = np.linspace(
                        points.iloc[:, 0].min(),
                        points.iloc[:, 0].max() * 0.80,
                    )
                    k_name = "gamma_k" if _adata.uns["dynamics"]["experiment_type"] == "one-shot" else "gamma"
                    vel_params_df = get_vel_params(_adata)
                    if k_name in vel_params_df.columns:
                        if not ("gamma_b" in vel_params_df.columns) or all(vel_params_df.gamma_b.isna()):
                            vel_params_df.loc[:, "gamma_b"] = 0
                            update_vel_params(_adata, params_df=vel_params_df)
                        ax.plot(
                            xnew,
                            xnew * _adata[:, cur_b].var.loc[:, k_name].unique()
                            + _adata[:, cur_b].var.loc[:, "gamma_b"].unique(),
                            dashes=[6, 2],
                            c=font_color,
                        )
                    else:
                        raise ValueError(
                            "_adata does not seem to have %s column. Velocity estimation is required "
                            "before running this function." % k_name
                        )
                if group is not None and add_group_gamma_fit and cur_b in _adata.var_names[_adata.var.use_for_dynamics]:
                    cell_groups = _adata.obs[group]
                    unique_groups = np.unique(cell_groups)
                    k_suffix = "gamma_k" if _adata.uns["dynamics"]["experiment_type"] == "one-shot" else "gamma"
                    for group_idx, cur_group in enumerate(unique_groups):
                        group_k_name = group + "_" + cur_group + "_" + k_suffix
                        group_adata = _adata[_adata.obs[group] == cur_group]
                        group_points = points.iloc[np.array(_adata.obs[group] == cur_group)]
                        group_b_key = group + "_" + cur_group + "_" + "gamma_b"
                        group_xnew = np.linspace(
                            group_points.iloc[:, 0].min(),
                            group_points.iloc[:, 0].max() * 0.90,
                        )
                        group_ynew = (
                            group_xnew * group_adata[:, cur_b].var.loc[:, group_k_name].unique()
                            + group_adata[:, cur_b].var.loc[:, group_b_key].unique()
                        )
                        ax.annotate(group + "_" + cur_group, xy=(group_xnew[-1], group_ynew[-1]))
                        vel_params_df = get_vel_params(group_adata)
                        if group_k_name in vel_params_df.columns:
                            if not (group_b_key in vel_params_df.columns) or all(vel_params_df[group_b_key].isna()):
                                vel_params_df.loc[:, group_b_key] = 0
                                update_vel_params(group_adata, params_df=vel_params_df)
                                main_info("No %s found, setting all bias terms to zero" % group_b_key)
                            ax.plot(
                                group_xnew,
                                group_ynew,
                                dashes=[6, 2],
                                c=group_colors[group_idx % len(group_colors)],
                            )
                        else:
                            raise Exception(
                                "_adata does not seem to have %s column. Velocity estimation is required "
                                "before running this function." % group_k_name
                            )

        # add legends according to colors and cmaps
        # collected during for loop above
        if stack_colors:
            ax.legend(handles=stack_legend_handles, loc="upper right", prop={"size": stack_colors_legend_size})

    for cur_b in basis:
        for cur_l in layer:
            main_debug("Plotting basis:%s, layer: %s" % (str(basis), str(layer)))
            main_debug("colors: %s" % (str(color)))
            _plot_basis_layer(cur_b, cur_l)

    main_debug("show, return or save...")
    return_value = None
    if return_all:
        return_value = (axes_list, color_list, font_color) if total_panels > 1 else (ax, color_out, font_color)
    else:
        return_value = axes_list if total_panels > 1 else ax
    return save_show_ret("scatters", save_show_or_return, save_kwargs, return_value, adjust=show_legend, background=background)


def map_to_points(
    _adata: AnnData,
    axis_x: str,
    axis_y: str,
    axis_z: str,
    basis_key: str,
    cur_c: str,
    cur_b: str,
    cur_l_smoothed: str,
) -> Tuple[pd.DataFrame, str]:
    """A helper function to map the given axis to corresponding coordinates in current embedding space.

    Args:
        _adata: an AnnData object.
        axis_x: the column index of the low dimensional embedding for the x-axis in current space.
        axis_y: the column index of the low dimensional embedding for the y-axis in current space.
        axis_z: the column index of the low dimensional embedding for the z-axis in current space.
        basis_key: the basis key constructed by current basis and layer.
        cur_c: the current key to color the data.
        cur_b: the current basis key representing the reduced dimension.
        cur_l_smoothed: the smoothed layer of data to use.

    Returns:
        The 3D DataFrame with coordinates of each sample and the title of the plot.
    """
    gene_title = []
    anno_title = []

    def _map_cur_axis(cur: str) -> Tuple[np.ndarray, str]:
        """A helper function to map an axis.

        Args:
            cur: the current axis to map.

        Returns:
            The coordinates and the column names.
        """
        nonlocal gene_title, anno_title

        if is_gene_name(_adata, cur):
            points_df_data = (_adata.obs_vector(k=cur, layer=None)
                              if cur_l_smoothed == "X"
                              else _adata.obs_vector(k=cur, layer=cur_l_smoothed))
            points_column = cur + " (" + cur_l_smoothed + ")"
            gene_title.append(cur)
        elif is_cell_anno_column(_adata, cur):
            points_df_data = _adata.obs_vector(cur)
            points_column = cur
            anno_title.append(cur)
        elif is_layer_keys(_adata, cur):
            points_df_data = _adata[:, cur_b].layers[cur]
            points_column = flatten(points_df_data)
        else:
            raise ValueError("Make sure your `x`, `y` and `z` are integers, gene names, column names in .obs, etc.")

        return points_df_data, points_column

    if type(axis_x) is int and type(axis_y) is int and type(axis_z):
        x_col_name = cur_b + "_0"
        y_col_name = cur_b + "_1"
        z_col_name = cur_b + "_2"

        points = pd.DataFrame(
            {
                x_col_name: _adata.obsm[basis_key][:, axis_x],
                y_col_name: _adata.obsm[basis_key][:, axis_y],
                z_col_name: _adata.obsm[basis_key][:, axis_z],
            }
        )
        points.columns = [x_col_name, y_col_name, z_col_name]

        cur_title = cur_c

        return points, cur_title
    elif type(axis_x) in [anndata._core.views.ArrayView, np.ndarray] and type(axis_y) in [
            anndata._core.views.ArrayView,
            np.ndarray,
        ]:
        points = pd.DataFrame({"x": flatten(axis_x), "y": flatten(axis_y), "x": flatten(axis_z)})
        points.columns = ["x", "y", "z"]
    else:
        x_points_df_data, x_points_column = _map_cur_axis(axis_x)
        y_points_df_data, y_points_column = _map_cur_axis(axis_y)
        z_points_df_data, z_points_column = _map_cur_axis(axis_z)
        points = pd.DataFrame({
            axis_x: x_points_df_data,
            axis_y: y_points_df_data,
            axis_z: z_points_df_data,
        })
        points.columns = [x_points_column, y_points_column, z_points_column]

    if len(gene_title) != 0:
        cur_title = " VS ".join(gene_title)
    elif len(anno_title) == 3:
        cur_title = " VS ".join(anno_title)
    else:
        cur_title = cur_b

    return points, cur_title


def scatters_interactive(
    adata: AnnData,
    basis: str = "umap",
    x: Union[int, str] = 0,
    y: Union[int, str] = 1,
    z: Union[int, str] = 2,
    color: str = "ntr",
    layer: str = "X",
    plot_method: str = "pv",
    labels: Optional[list] = None,
    values: Optional[list] = None,
    cmap: Optional[str] = None,
    theme: Optional[str] = None,
    background: Optional[str] = None,
    color_key: Union[Dict[str, str], List[str], None] = None,
    color_key_cmap: Optional[str] = None,
    use_smoothed: bool = True,
    sym_c: bool = False,
    smooth: bool = False,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Dict[str, Any] = {},
    **kwargs,
):
    """Plot an embedding as points with Pyvista. Currently only 3D input is supported. For 2D data, `scatters` is a
    better alternative.

    The function will use the colors from matplotlib to keep consistence with other plotting functions.

    Args:
        adata: an AnnData object.
        basis: the reduced dimension stored in adata.obsm. The specific basis key will be constructed in the following
            priority if exits: 1) specific layer input +  basis 2) X_ + basis 3) basis. E.g. if basis is PCA, `scatters`
            is going to look for 1) if specific layer is spliced, `spliced_pca` 2) `X_pca` (dynamo convention) 3) `pca`.
            Defaults to "umap".
        x: the column index of the low dimensional embedding for the x-axis. Defaults to 0.
        y: the column index of the low dimensional embedding for the y-axis. Defaults to 1.
        z: the column index of the low dimensional embedding for the z-axis. Defaults to 2.
        color: any column names or gene expression, etc. that will be used for coloring cells. Defaults to "ntr".
        layer: the layer of data to use for the scatter plot. Defaults to "X".
        labels: an array of labels (assumed integer or categorical), one for each data sample. This will be used for
            coloring the points in the plot according to their label. Note that this option is mutually exclusive to the
            `values` option. Defaults to None.
        values: an array of values (assumed float or continuous), one for each sample. This will be used for coloring
            the points in the plot according to a colorscale associated to the total range of values. Note that this
            option is mutually exclusive to the `labels` option. Defaults to None.
        theme: A color theme to use for plotting. A small set of predefined themes are provided which have relatively
            good aesthetics. Defaults to None.
        cmap: The name of a matplotlib colormap to use for coloring or shading points. If no labels or values are passed
            this will be used for shading points according to density (largely only of relevance for very large
            datasets). If values are passed this will be used for shading according the value. Note that if theme is
            passed then this value will be overridden by the corresponding option of the theme. Defaults to None.
        background: the color of the background. Usually this will be either 'white' or 'black', but any color name will
            work. Ideally one wants to match this appropriately to the colors being used for points etc. This is one of
            the things that themes handle for you. Note that if theme is passed then this value will be overridden by
            the corresponding option of the theme. Defaults to None.
        color_key: the method to assign colors to categoricals. This can either be an explicit dict mapping labels to
            colors (as strings of form '#RRGGBB'), or an array like object providing one color for each distinct
            category being provided in `labels`. Either way this mapping will be used to color points according to the
            label. Note that if theme is passed then this value will be overridden by the corresponding option of the
            theme. Defaults to None.
        color_key_cmap: the name of a matplotlib colormap to use for categorical coloring. If an explicit `color_key` is
            not given a color mapping for categories can be generated from the label list and selecting a matching list
            of colors from the given colormap. Note that if theme is passed then this value will be overridden by the
            corresponding option of the theme. Defaults to None.
        use_smoothed: whether to use smoothed values (i.e. M_s / M_u instead of spliced / unspliced, etc.). Defaults to
            True.
        sym_c: whether do you want to make the limits of continuous color to be symmetric, normally this should be used
            for plotting velocity, jacobian, curl, divergence or other types of data with both positive or negative
            values. Defaults to False.
        smooth: whether do you want to further smooth data and how much smoothing do you want. If it is `False`, no
            smoothing will be applied. If `True`, smoothing based on one-step diffusion of connectivity matrix
            (`.uns['moment_cnn']`) will be applied. If a number larger than 1, smoothing will be based on `smooth` steps
            of diffusion.
        save_show_or_return: whether to save, show or return the figure. If "both", it will save and plot the figure at
            the same time. If "all", the figure will be saved, displayed and the associated axis and other object will
            be return. Defaults to "show".
        save_kwargs: A dictionary that will be passed to the saving function. By default, it is an empty dictionary
            and the saving function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
            "title": PyVista Export, "raster": True, "painter": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.
        **kwargs: any other kwargs that would be passed to `Plotter.add_points()`.

    Returns:
        If `save_show_or_return` is `save`, `show` or `both`, the function will return nothing but show or save the
        figure. If `save_show_or_return` is `return`, the function will return the axis object(s) that contains the
        figure.
    """

    if plot_method == "pv":
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("Please install pyvista first.")
    elif plot_method == "plotly":
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("Please install plotly first.")
    else:
        raise NotImplementedError("Current plot method not supported.")

    if type(x) in [int, str]:
        x = [x]
    if type(y) in [int, str]:
        y = [y]
    if type(z) in [int, str]:
        z = [z]

    # make x, y, z lists of list, where each list corresponds to one coordinate set
    if (
            type(x) in [anndata._core.views.ArrayView, np.ndarray]
            and type(y) in [anndata._core.views.ArrayView, np.ndarray]
            and type(z) in [anndata._core.views.ArrayView, np.ndarray]
            and len(x) == adata.n_obs
            and len(y) == adata.n_obs
            and len(z) == adata.n_obs
    ):
        x, y, z = [x], [y], [z]

    elif hasattr(x, "__len__") and hasattr(y, "__len__") and hasattr(z, "__len__"):
        x, y, z = list(x), list(y), list(z)

    assert len(x) == len(y) and len(x) == len(z), "bug: x, y, z does not have the same shape."

    if use_smoothed:
        mapper = get_mapper()

    # check color, layer, basis -> convert to list
    if type(color) is str:
        color = [color]
    if type(layer) is str:
        layer = [layer]
    if type(basis) is str:
        basis = [basis]

    n_c, n_l, n_b, n_x, n_y, n_z = (
        1 if color is None else len(color),
        1 if layer is None else len(layer),
        1 if basis is None else len(basis),
        1 if x is None else 1 if type(x) in [anndata._core.views.ArrayView, np.ndarray] else len(x),
        1 if y is None else 1 if type(y) in [anndata._core.views.ArrayView, np.ndarray] else len(y),
        1 if z is None else 1 if type(z) in [anndata._core.views.ArrayView, np.ndarray] else len(z),
    )

    total_panels, ncols = (
        n_c * n_l * n_b * n_x * n_y * n_z,
        max([n_c, n_l, n_b, n_x, n_y, n_z]),
    )

    nrow, ncol = int(np.ceil(total_panels / ncols)), ncols
    subplot_indices = [[i, j] for i in range(nrow) for j in range(ncol)]
    cur_subplot = 0
    colors_list = []

    if total_panels == 1:
        pl = pv.Plotter() if plot_method == "pv" else make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])
    else:
        pl = (
            pv.Plotter(shape=(nrow, ncol))
            if plot_method == "pv"
            else
            make_subplots(rows=nrow, cols=ncol, specs=[[{"type": "scatter3d"} for _ in range(ncol)] for _ in range(nrow)])
        )

    def _plot_basis_layer_pv(cur_b: str, cur_l: str) -> None:
        """A helper function for plotting a specific basis/layer data

        Args:
            cur_b: current basis
            cur_l: current layer
        """
        nonlocal background, adata, cmap, cur_subplot, sym_c

        if cur_l in ["acceleration", "curvature", "divergence", "velocity_S", "velocity_T"]:
            cur_l_smoothed = cur_l
            cmap, sym_c = "bwr", True  # TODO maybe use other divergent color map in the future
        else:
            if use_smoothed:
                cur_l_smoothed = cur_l if cur_l.startswith("M_") | cur_l.startswith("velocity") else mapper[cur_l]
                if cur_l.startswith("velocity"):
                    cmap, sym_c = "bwr", True

        if cur_l + "_" + cur_b in adata.obsm.keys():
            prefix = cur_l + "_"
        elif ("X_" + cur_b) in adata.obsm.keys():
            prefix = "X_"
        elif cur_b in adata.obsm.keys():
            # special case for spatial for compatibility with other packages
            prefix = ""
        else:
            raise ValueError("Please check if basis=%s exists in adata.obsm" % basis)

        basis_key = prefix + cur_b
        main_info("plotting with basis key=%s" % basis_key, indent_level=2)

        for cur_c in color:
            main_debug("coloring scatter of cur_c: %s" % str(cur_c))

            _color = _get_adata_color_vec(adata, cur_l, cur_c)

            # select data rows based on stack color thresholding
            is_numeric_color = np.issubdtype(_color.dtype, np.number)
            if not is_numeric_color:
                main_info(
                    "skip filtering %s by stack threshold when stacking color because it is not a numeric type"
                    % (cur_c),
                    indent_level=2,
                )
            _labels, _values = None, None

            for cur_x, cur_y, cur_z in zip(x, y, z):  # here x / y are arrays
                main_debug("handling coordinates, cur_x: %s, cur_y: %s, cur_z: %s" % (cur_x, cur_y, cur_z))

                points, cur_title = map_to_points(
                    adata,
                    axis_x=cur_x,
                    axis_y=cur_y,
                    axis_z=cur_z,
                    basis_key=basis_key,
                    cur_c=cur_c,
                    cur_b=cur_b,
                    cur_l_smoothed=cur_l_smoothed,
                )

                # https://stackoverflow.com/questions/4187185/how-can-i-check-if-my-python-object-is-a-number
                # answer from Boris.
                is_not_continuous = not isinstance(_color[0], Number) or _color.dtype.name == "category"

                if is_not_continuous:
                    _labels = np.asarray(_color) if is_categorical_dtype(_color) else _color
                    if theme is None:
                        if background in ["#ffffff", "black"]:
                            _theme_ = "glasbey_dark"
                        else:
                            _theme_ = "glasbey_white"
                    else:
                        _theme_ = theme
                else:
                    _values = _color
                    if theme is None:
                        if background in ["#ffffff", "black"]:
                            _theme_ = "inferno" if cur_l != "velocity" else "div_blue_black_red"
                        else:
                            _theme_ = "viridis" if not cur_l.startswith("velocity") else "div_blue_red"
                    else:
                        _theme_ = theme

                _cmap = _themes[_theme_]["cmap"] if cmap is None else cmap

                _color_key_cmap = _themes[_theme_]["color_key_cmap"] if color_key_cmap is None else color_key_cmap
                background = _themes[_theme_]["background"] if background is None else background

                if labels is not None and values is not None:
                    raise ValueError("Conflicting options; only one of labels or values should be set")

                if labels is not None or values is not None:
                    _labels = labels.copy()
                    _values = values.copy()
                    main_info("`Color` will be ignored because labels/values is provided.")

                if smooth and not is_not_continuous:
                    main_debug("smooth and not continuous")
                    knn = adata.obsp["moments_con"]
                    _values = (
                        calc_1nd_moment(_values, knn)[0]
                        if smooth in [1, True]
                        else calc_1nd_moment(_values, knn**smooth)[0]
                    )

                colors, color_type, _ = calculate_colors(
                    points.values,
                    labels=_labels,
                    values=_values,
                    cmap=_cmap,
                    color_key=color_key,
                    color_key_cmap=_color_key_cmap,
                    background=background,
                    sym_c=sym_c,
                )

                colors_list.append(colors)

                if plot_method == "pv":
                    if total_panels > 1:
                        pl.subplot(subplot_indices[cur_subplot][0], subplot_indices[cur_subplot][1])

                    pvdataset = pv.PolyData(points.values)
                    pvdataset.point_data["colors"] = np.stack(colors)
                    pl.add_points(pvdataset, scalars="colors", preference='point', rgb=True, cmap=_cmap, **kwargs)

                    if color_type == "labels":
                        type_color_dict = {cell_type: cell_color for cell_type, cell_color in zip(_labels, colors)}
                        type_color_pair = [[k, v] for k, v in type_color_dict.items()]
                        pl.add_legend(labels=type_color_pair)
                    else:
                        pl.add_scalar_bar()  # TODO: fix the bug that scalar bar only works in the first plot

                    pl.add_text(cur_title)
                    pl.add_axes(xlabel=points.columns[0], ylabel=points.columns[1], zlabel=points.columns[2])
                elif plot_method == "plotly":

                    pl.add_trace(
                        go.Scatter3d(
                            x=points.iloc[:, 0],
                            y=points.iloc[:, 1],
                            z=points.iloc[:, 2],
                            mode="markers",
                            marker=dict(
                                color=colors,
                            ),
                            text=_labels if color_type == "labels" else _values,
                            **kwargs,
                        ),
                        row=subplot_indices[cur_subplot][0] + 1, col=subplot_indices[cur_subplot][1] + 1,
                    )

                    pl.update_layout(
                        scene=dict(
                            xaxis_title=points.columns[0],
                            yaxis_title=points.columns[1],
                            zaxis_title=points.columns[2]
                        ),
                    )

                cur_subplot += 1

    for cur_b in basis:
        for cur_l in layer:
            main_debug("Plotting basis:%s, layer: %s" % (str(basis), str(layer)))
            main_debug("colors: %s" % (str(color)))
            _plot_basis_layer_pv(cur_b, cur_l)

    return save_pyvista_plotter(
        pl=pl,
        colors_list=colors_list,
        save_show_or_return=save_show_or_return,
        save_kwargs=save_kwargs,
    ) if plot_method == "pv" else save_plotly_figure(
        pl=pl,
        colors_list=colors_list,
        save_show_or_return=save_show_or_return,
        save_kwargs=save_kwargs,
    )
