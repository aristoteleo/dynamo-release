# code adapted from https://github.com/lmcinnes/umap/blob/7e051d8f3c4adca90ca81eb45f6a9d1372c076cf/umap/plot.py
import warnings
from numbers import Number
from typing import List, Optional, Union

import anndata
import matplotlib.cm
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from pandas.api.types import is_categorical_dtype

from ..configuration import _themes, reset_rcParams
from ..docrep import DocstringProcessor
from ..dynamo_logger import main_debug, main_info, main_warning
from ..preprocessing.utils import affine_transform, gen_rotation_2d
from ..tools.moments import calc_1nd_moment
from ..tools.utils import flatten, get_mapper, update_dict
from .utils import (
    _datashade_points,
    _get_adata_color_vec,
    _matplotlib_points,
    _select_font_color,
    arrowed_spines,
    deaxis_all,
    despline_all,
    is_cell_anno_column,
    is_gene_name,
    is_layer_keys,
    is_list_of_lists,
    save_fig,
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
    theme: Optional[str] = None,
    cmap: Optional[str] = None,
    color_key: Union[dict, list] = None,
    color_key_cmap: Optional[str] = None,
    background: Optional[str] = None,
    ncols: int = 4,
    pointsize: Union[None, float] = None,
    figsize: tuple = (6, 4),
    show_legend="on data",
    use_smoothed: bool = True,
    aggregate: Optional[str] = None,
    show_arrowed_spines: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    sort: str = "raw",
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
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
    inset_dict: dict = {},
    marker: str = None,
    group: str = None,
    add_group_gamma_fit=False,
    affine_transform_degree: int = None,
    affine_transform_A=None,
    affine_transform_b=None,
    stack_colors=False,
    stack_colors_threshold=0.001,
    stack_colors_title="stacked colors",
    stack_colors_legend_size=2,
    stack_colors_cmaps=None,
    despline: bool = True,
    deaxis: bool = True,
    despline_sides: Union[None, List[str]] = None,
    projection="2d",
    **kwargs,
) -> Union[None, Axes]:
    """Plot an embedding as points. Currently this only works
    for 2D embeddings. While there are many optional parameters
    to further control and tailor the plotting, you need only
    pass in the trained/fit umap model to get results. This plot
    utility will attempt to do the hard work of avoiding
    overplotting issues, and make it easy to automatically
    colour points by a categorical labelling or numeric values.
    This method is intended to be used within a Jupyter
    notebook with ``%matplotlib inline``.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        basis: `str`
            The reduced dimension stored in adata.obsm. The specific basis key will be constructed in the following priority if exits: 1) specific layer input +  basis 2) X_ + basis 3) basis. E.g. if basis is PCA, `scatters` is going to look for 1) if specific layer is spliced, `spliced_pca` 2) `X_pca` (dynamo convention) 3) `pca`
        x: `int` (default: `0`)
            The column index of the low dimensional embedding for the x-axis.
        y: `int` (default: `1`)
            The column index of the low dimensional embedding for the y-axis.
        color: `string` (default: `ntr`)
            Any column names or gene expression, etc. that will be used for coloring cells.
        layer: `str` (default: `X`)
            The layer of data to use for the scatter plot.
        highlights: `list` (default: None)
            Which color group will be highlighted. if highligts is a list of lists - each list is relate to each color
            element.
        labels: array, shape (n_samples,) (optional, default None)
            An array of labels (assumed integer or categorical),
            one for each data sample.
            This will be used for coloring the points in
            the plot according to their label. Note that
            this option is mutually exclusive to the ``values``
            option.
        values: array, shape (n_samples,) (optional, default None)
            An array of values (assumed float or continuous),
            one for each sample.
            This will be used for coloring the points in
            the plot according to a colorscale associated
            to the total range of values. Note that this
            option is mutually exclusive to the ``labels``
            option.
        theme: string (optional, default None)
            A color theme to use for plotting. A small set of
            predefined themes are provided which have relatively
            good aesthetics. Available themes are:
               * 'blue'
               * 'red'
               * 'green'
               * 'inferno'
               * 'fire'
               * 'viridis'
               * 'darkblue'
               * 'darkred'
               * 'darkgreen'
        cmap: string (optional, default 'Blues')
            The name of a matplotlib colormap to use for coloring
            or shading points. If no labels or values are passed
            this will be used for shading points according to
            density (largely only of relevance for very large
            datasets). If values are passed this will be used for
            shading according the value. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        color_key: dict or array, shape (n_categories) (optional, default None)
            A way to assign colors to categoricals. This can either be
            an explicit dict mapping labels to colors (as strings of form
            '#RRGGBB'), or an array like object providing one color for
            each distinct category being provided in ``labels``. Either
            way this mapping will be used to color points according to
            the label. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        color_key_cmap:
            The name of a matplotlib colormap to use for categorical coloring.
            If an explicit ``color_key`` is not given a color mapping for
            categories can be generated from the label list and selecting
            a matching list of colors from the given colormap. Note
            that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        background: string or None (optional, default 'None`)
            The color of the background. Usually this will be either
            'white' or 'black', but any color name will work. Ideally
            one wants to match this appropriately to the colors being
            used for points etc. This is one of the things that themes
            handle for you. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        ncols: int (optional, default `4`)
            Number of columns for the figure.
        pointsize: `None` or `float` (default: None)
            The scale of the point size. Actual point cell size is calculated as `500.0 / np.sqrt(adata.shape[0]) *
            pointsize`
        figsize: `None` or `[float, float]` (default: None)
            The width and height of a figure.
        show_legend: bool (optional, default True)
            Whether to display a legend of the labels
        use_smoothed: bool (optional, default True)
            Whether to use smoothed values (i.e. M_s / M_u instead of spliced / unspliced, etc.).
        aggregate: `str` or `None` (default: `None`)
            The column in adata.obs that will be used to aggregate data points.
        show_arrowed_spines: bool (optional, default False)
            Whether to show a pair of arrowed spines representing the basis of the scatter is currently using.
        ax: `matplotlib.Axis` (optional, default `None`)
            The matplotlib axes object where new plots will be added to. Only applicable to drawing a single component.
        sort: `str` (optional, default `raw`)
            The method to reorder data so that high values points will be on top of background points. Can be one of
            {'raw', 'abs', 'neg'}, i.e. sorted by raw data, sort by absolute values or sort by negative values.
        save_show_or_return: `str` {'save', 'show', 'return'} (default: `show`)
            Whether to save, show or return the figure. If "both", it will save and plot the figure at the same time. If
            "all", the figure will be saved, displayed and the associated axis and other object will be return.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
            save_fig function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent":
            True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that
            properly modify those keys according to your needs.
        return_all: `bool` (default: `False`)
            Whether to return all the scatter related variables. Default is False.
        add_gamma_fit: `bool` (default: `False`)
            Whether to add the line of the gamma fitting. This will automatically turn on if `basis` points to gene
            names and those genes have went through gamma fitting.
        frontier: `bool` (default: `False`)
            Whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to show
            area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib tips &
            tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from scEU-seq
            paper: https://science.sciencemag.org/content/367/6482/1151. If `contour` is set  to be True, `frontier`
            will be ignored as `contour` also add an outlier for data points.
        contour: `bool` (default: `False`)
            Whether to add an countor on top of scatter plots. We use tricontourf to plot contour for non-gridded data.
            The shapely package was used to create a polygon of the concave hull of the scatters. With the polygon we
            then check if the mean of the triangulated points are within the polygon and use this as our condition to
            form the mask to create the contour. We also add the polygon shape as a frontier of the data point (similar
            to when setting `frontier = True`). When the color of the data points is continuous, we will use the same
            cmap as for the scatter points by default, when color is categorical, no contour will be drawn but just the
            polygon. cmap can be set with `ccmap` argument. See below. This has recently changed to use seaborn's
            kdeplot.
        ccmap: `str` or `None` (default: `None`)
            The name of a matplotlib colormap to use for coloring or shading points the contour. See above.
        calpha: `float` (default: `0.4`)
            Contour alpha value passed into sns.kdeplot. The value should be inbetween [0, 1]
        sym_c: `bool` (default: `False`)
            Whether do you want to make the limits of continuous color to be symmetric, normally this should be used for
            plotting velocity, jacobian, curl, divergence or other types of data with both positive or negative values.
        smooth: `bool` or `int` (default: `False`)
            Whether do you want to further smooth data and how much smoothing do you want. If it is `False`, no
            smoothing will be applied. If `True`, smoothing based on one step diffusion of connectivity matrix
            (`.uns['moment_cnn'] will be applied. If a number larger than 1, smoothing will based on `smooth` steps of
            diffusion.
        dpi: `float`, (default: 100.0)
            The resolution of the figure in dots-per-inch. Dots per inches (dpi) determines how many pixels the figure
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
            https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size
        inset_dict: `dict` (default: {})
            A dictionary of parameters in inset_ax. Example, something like {"width": "5%", "height": "50%", "loc":
            'lower left', "bbox_to_anchor": (0.85, 0.90, 0.145, 0.145), "bbox_transform": ax.transAxes, "borderpad": 0}
            See more details at https://matplotlib.org/api/_as_gen/mpl_toolkits.axes_grid1.inset_locator.inset_axes.html
            or https://stackoverflow.com/questions/39803385/what-does-a-4-element-tuple-argument-for-bbox-to-anchor-mean
            -in-matplotlib
        marker: `str` (default: None)
            The marker style. marker can be either an instance of the class or the text shorthand for a particular
            marker. See matplotlib.markers for more information about marker styles.
        affine_transform_degree:
            Transform coordinates of points according to some degree.
        affine_transform_A:
            Coefficients in affine transformation Ax + b. 2D for now.
        affine_transform_b:
            Bias in affine transformation Ax + b.
        stack_colors:
            Whether to stack all color on the same ax passed above.
            Currently only support 18 sequential matplotlib default cmaps assigning to different color groups.
            (#colors should be smaller than 18, reuse if #colors > 18. TODO generate cmaps according to #colors)
        stack_colors_threshold:
            A threshold for filtering out points values < threshold when drawing each color.
            E.g. if you do not want points with values < 1 showing up on axis, set threshold to be 1
        stack_colors_title:
            The title for the stack_color plot.
        stack_colors_legend_size:
            Control the legend size in stack color plot.
        stack_colors_cmaps:
            a list of cmaps that will be used to map values to color when stacking colors on the same subplot. The order corresponds to the order of color.
        despline:
            Whether to remove splines of the figure.
        despline_sides:
            Which side of splines should be removed. Can be any combination of `["bottom", "right", "top", "left"]`.
        deaxis:
            Whether to remove axis ticks of the figure.
        kwargs:
            Additional arguments passed to plt.scatters.

    Returns
    -------
        result:
            Either None or a matplotlib axis with the relevant plot displayed.
            If you are using a notbooks and have ``%matplotlib inline`` set
            then this will simply display inline.
    """

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.colors import rgb2hex, to_hex

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
                    "your adata oject is corrupted. Please make sure it has at least one of the following "
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
            raise Exception("Please check if basis=%s exists in adata.obsm" % basis)

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
                    raise Exception("Make sure your `x` and `y` are integers, gene names, column names in .obs, etc.")

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
                        else calc_1nd_moment(values, knn ** smooth)[0]
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
                    if k_name in _adata.var.columns:
                        if not ("gamma_b" in _adata.var.columns) or all(_adata.var.gamma_b.isna()):
                            _adata.var.loc[:, "gamma_b"] = 0
                        ax.plot(
                            xnew,
                            xnew * _adata[:, cur_b].var.loc[:, k_name].unique()
                            + _adata[:, cur_b].var.loc[:, "gamma_b"].unique(),
                            dashes=[6, 2],
                            c=font_color,
                        )
                    else:
                        raise Exception(
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
                        if group_k_name in group_adata.var.columns:
                            if not (group_b_key in group_adata.var.columns) or all(group_adata.var[group_b_key].isna()):
                                group_adata.var.loc[:, group_b_key] = 0
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
    if save_show_or_return in ["save", "both", "all"]:
        s_kwargs = {
            "path": None,
            "prefix": "scatters",
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
        if background is not None:
            reset_rcParams()
    elif save_show_or_return in ["show", "both", "all"]:
        if show_legend:
            plt.subplots_adjust(right=0.85)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # TODO: tight_layout causing "AttributeError: 'AnchoredSizeLocator' object has no attribute 'get_subplotspec'"
            # plt.tight_layout()

        plt.show()
        if background is not None:
            reset_rcParams()
    elif save_show_or_return in ["return", "all"]:
        if background is not None:
            reset_rcParams()

        if return_all:
            return (axes_list, color_list, font_color) if total_panels > 1 else (ax, color_out, font_color)
        else:
            return axes_list if total_panels > 1 else ax
