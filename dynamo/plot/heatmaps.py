import math
import warnings
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
import scipy.spatial as ss
import seaborn
from anndata import AnnData
from matplotlib.colors import Colormap

from ..dynamo_logger import main_info, main_info_insert_adata, main_warning
from ..estimation.fit_jacobian import (
    fit_hill_grad,
    hill_act_func,
    hill_act_grad,
    hill_inh_func,
    hill_inh_grad,
)
from ..tools.utils import flatten
from ..vectorfield.utils import get_jacobian
from ..vectorfield.vector_calculus import hessian as run_hessian
from .utils import (
    _datashade_points,
    _matplotlib_points,
    _select_font_color,
    arrowed_spines,
    deaxis_all,
    despline_all,
    is_cell_anno_column,
    is_gene_name,
    is_layer_keys,
    is_list_of_lists,
    save_show_ret,
)


def bandwidth_nrd(x):
    x = pd.Series(x)
    h = (x.quantile([0.75]).values - x.quantile([0.25]).values) / 1.34

    res = 4 * 1.06 * min(math.sqrt(np.var(x, ddof=1)), h) * (len(x) ** (-1 / 5))
    return np.asscalar(res) if isinstance(res, np.ndarray) else res


def rep(x, length):
    len_x = len(x)
    n = int(length / len_x)
    r = length % len_x
    re = []
    for i in range(0, n):
        re = re + x
    for i in range(0, r):
        re = re + [x[i]]
    return re


# https://stackoverflow.com/questions/46166933/python-numpy-equivalent-of-r-rep-and-rep-len-functions?rq=1
# def rep2(x, length):
#     x = np.array(x)
#     res = np.repeat(x, length, axis=0)

#     return res


def rep2(x, length_out):
    return np.tile(x, length_out // len(x) + 1)[:length_out]


def dnorm(x, u=0, sig=1):
    return np.exp(-((x - u) ** 2) / (2 * sig**2)) / (math.sqrt(2 * math.pi) * sig)


def kde2d(
    x: List[float], y: List[float], h: Optional[List[float]] = None, n: int = 25, lims: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce kde2d function behavior from MASS package in R.

    Two-dimensional kernel density estimation with an axis-aligned bivariate normal kernel, evaluated on a square grid.

    Args:
        x: x coordinate of the data.
        y: y coordinate of the data.
        h: vector of bandwidths for `x` and `y` directions. if None, it would use normal reference bandwidth
            (see `bandwidth.nrd`). A scalar value will be taken to apply to both directions. Defaults to None.
        n: number of grid points in each direction. Can be scalar or a length-2 integer list. Defaults to 25.
        lims: the limits of the rectangle covered by the grid as `x_l, x_u, y_l, y_u`. Defaults to None.

    Raises:
        ValueError: `x` and `y` have different sizes.
        ValueError: `x` or `y` has non-valid values.
        ValueError: `lims` has non-valid values.
        ValueError: `h` has non-positive values.

    Returns:
        A tuple (gx, gy, z) where `gx` and `gy` are x and y coordinates of grid points, respectively; `z` is a `n[1]`
        by `n[2]` matrix of estimated density: its rows correspond to the value of `x` and columns to the value of `y`.
    """

    nx = len(x)
    if not lims:
        lims = [min(x), max(x), min(y), max(y)]
    if len(y) != nx:
        raise ValueError("data vectors must be the same length")
    elif not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("missing or infinite values in the data are not allowed")
    elif not np.all(np.isfinite(lims)):
        raise ValueError("only finite values are allowed in 'lims'")
    else:
        n = rep(n, length=2) if isinstance(n, list) else rep([n], length=2)
        gx = np.linspace(lims[0], lims[1], n[0])
        gy = np.linspace(lims[2], lims[3], n[1])
        if h is None:
            h = [bandwidth_nrd(x), bandwidth_nrd(y)]
        else:
            h = np.array(rep(h, length=2))

        if np.any(h <= 0):
            raise ValueError("bandwidths must be strictly positive")
        else:
            h /= 4
            ax = pd.DataFrame((gx - x[:, np.newaxis]) / h[0]).T
            ay = pd.DataFrame((gy - y[:, np.newaxis]) / h[1]).T
            z = (np.matrix(dnorm(ax)) * np.matrix(dnorm(ay).T)) / (nx * h[0] * h[1])
    return gx, gy, z


def kde2d_to_mean_and_sigma(
    gx: np.ndarray, gy: np.ndarray, dens: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the mean and sigma of y grids corresponding to x grids from kde2d.

    Args:
        gx: x coordinates of grid points.
        gy: y coordinates of grid points.
        dens: estimated kernel density.

    Returns:
        A tuple (x_grid, y_mean, y_sigm) where x_grid is unique values of `gx`, `y_mean` and `y_sigm` are corresponding
        mean and sigma of y grids.
    """

    x_grid = np.unique(gx)
    y_mean = np.zeros(len(x_grid))
    y_sigm = np.zeros(len(x_grid))
    for i, x in enumerate(x_grid):
        mask = gx == x
        den = dens[mask]
        Y_ = gy[mask]
        mean = np.average(Y_, weights=den)
        sigm = np.sqrt(np.average((Y_ - mean) ** 2, weights=den))
        y_mean[i] = mean
        y_sigm[i] = sigm
    return x_grid, y_mean, y_sigm


def response(
    adata: AnnData,
    pairs_mat: np.ndarray,
    xkey: Optional[str] = None,
    ykey: Optional[str] = None,
    log: bool = True,
    drop_zero_cells: bool = True,
    delay: int = 0,
    grid_num: int = 25,
    n_row: int = 1,
    n_col: Optional[int] = None,
    cmap: Union[str, Colormap, None] = None,
    show_ridge: bool = False,
    show_rug: bool = True,
    zero_indicator: bool = False,
    zero_line_style: str = "w--",
    zero_line_width: float = 2.5,
    mean_style: str = "c*",
    fit_curve: bool = False,
    fit_mode: Literal["hill"] = "hill",
    curve_style: str = "c-",
    curve_lw: float = 2.5,
    no_degradation: bool = True,
    show_extent: bool = False,
    ext_format: Optional[List[str]] = None,
    stacked_fraction: bool = False,
    figsize: Tuple[float, float] = (6, 4),
    save_show_or_return: Literal["save", "show", "both", "all"] = "show",
    save_kwargs: Dict[str, Any] = {},
    return_data: bool = False,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]],
    None,
]:
    """Plot the lagged DREVI plot pairs of genes across pseudotime.

    This plotting function builds on the original idea of DREVI plot but is extended in the context for causal network.
    It considers the time delay between the hypothetical regulators to the target genes which is parametered by `d`.
    Lagged DREVI plot first estimates the joint density (`P(x_{t - d}, y_t)`) for variables `x_{t - d} and y_t`, then it
    divides the joint density by the marginal density `P(x_{t - d})` to get the conditional density estimate
    (`P(x_{t - d}, y_t | x_{x - d})`). We then calculate the z-score normalizing each column of conditional density.
    Note that this plot tries to demonstrate the potential influence between two variables instead of the factual
    influence. A red line corresponding to the point with maximal density on each `x` value is plot which indicates the
    maximal possible point for `y_t` give the value of `x_{t - d}`. The 2-d density is estimated through the kde2d
    function.

    Args:
        adata: an AnnData object.
        pairs_mat: a matrix where each row is the gene pair and the first column is the hypothetical source or regulator
            while the second column represents the hypothetical target. The name in this matrix should match the name in
            the gene_short_name column of the adata object.
        xkey: the key of the layer storing `x` data. Defaults to None.
        ykey: the key of the layer storing `y` data. Defaults to None.
        log: whether to perform log transformation (using `log(expression + 1)`) before calculating density estimates.
            Defaults to True.
        drop_zero_cells: whether to drop cells that with zero expression for either the potential regulator or potential
            target. This can signify the relationship between potential regulators and targets, speed up the
            calculation, but at the risk of ignoring strong inhibition effects from certain regulators to targets.
            Defaults to True.
        delay: the time delay between the source and target gene. Always zero because we don't have real time-series.
            Defaults to 0.
        grid_num: the number of grid when creating the lagged DREVI plot. Defaults to 25.
        n_row: number of rows used to layout the faceted cluster panels. Defaults to 1.
        n_col: number of columns used to layout the faceted cluster panels. Defaults to None.
        cmap: the color map used to plot the heatmap. Could be the name of the color map or a matplotlib color map
            object. If None, the color map would be generated automatically. Defaults to None.
        show_ridge: whether to show the ridge curve. Defaults to False.
        show_rug: whether to plot marginal distributions by drawing ticks along the x and y axes. Defaults to True.
        zero_indicator: whether to plot the zero line in the graph. Defaults to False.
        zero_line_style: line style of the zero line. Defaults to "w--".
        zero_line_width: line width of the zero line. Defaults to 2.5.
        mean_style: the line style for plotting the y mean data. Defaults to "c*".
        fit_curve: whether to fit the curve between `x` and `y`. Defaults to False.
        fit_mode: the fitting mode for fitting the curve between `x` and `y`. Currently, the relationship can only be
            fit into Hill function. Defaults to "hill".
        curve_style: the line style of fitted curve. Defaults to "c-".
        curve_lw: the line width of the fitted curve. Defaults to 2.5.
        no_degradation: whether to consider degradation when fitting Hill equations. Defaults to True.
        show_extent: whether to extend the figure. If False, `show_ridge` and `show_rug` would be set to False
            automatically. Defaults to False.
        ext_format: the string/list of strings (the first is for x and second for y labels) that will be used to format
            the ticks on x or y-axis. If it is None or one of the element in the list is None, the default setting will
            be used. Defaults to None.
        stacked_fraction: If True the jacobian will be represe nted as a stacked fraction in the title, otherwise a
            linear fraction tyle is used. Defaults to False.
        figsize: size of the figure. Defaults to (6, 4).
        save_show_or_return: whether to save or show the figure. If "both", it will save and plot the figure at the same time. If
            "all", the figure will be saved, displayed and the associated axis and other object will be return.
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.
        return_data: whether to return the data used to generate the heatmap. Defaults to False.

    Raises:
        ValueError: no preprocessing data in adata.uns
        ValueError: No layers named as `xkey` or `ykey`.
        ValueError: adata does not contain gene data specified in pairs_mat
        ValueError: `n_col * n_row` is less than number of gene pairs
        NotImplementedError: invalid `fit_mode`

    Returns:
        None would be returned in default. If `return_data` is set to be True, a tuple (flat_res, flat_res_subset,
        ridge_curve_subset) would be returned, where flat_res is a pandas data frame used to create the
        heatmap with four columns (`x`: x-coordinate; `y`: y-coordinate; `den`: estimated density at x/y coordinate;
        `type`: the corresponding gene pair), flat_res_subset is a pandas data frame used to create the heatmap for the
        last gene pair (if multiple gene-pairs are inputted) with four columns (`x`: x-coordinate; `y`: y-coordinate;
        `den`: estimated density at x/y coordinate; `type`: the corresponding gene pair), and ridge_curve_subset is a
        pandas data frame used to create the read ridge line for the last gene pair (if multiple gene-pairs are
        inputted) with four columns (`x`: x-coordinate; `y`: y-coordinate; `type`: the corresponding gene pair). If
        `fit_curve` is True, the hill function fitting result would also be returned at the 4th position.
    """

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if show_extent is False:
        show_ridge = False
        show_rug = False

    all_genes_in_pair = np.unique(pairs_mat)

    if "pp" not in adata.uns_keys():
        raise ValueError("You must first run dyn.pp.recipe_monocle and dyn.tl.moments before running this function.")

    if xkey is None:
        xkey = "M_t" if adata.uns["pp"]["has_labeling"] else "M_s"
    if ykey is None:
        ykey = "M_n" if adata.uns["pp"]["has_labeling"] else "M_u"

    if not set([xkey, ykey]) <= set(adata.layers.keys()).union(set(["jacobian"])):
        raise ValueError(
            f"adata.layers doesn't have {xkey, ykey} layers. Please specify the correct layers or "
            "perform relevant preprocessing and vector field analyses first."
        )
    if cmap is None:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "response", ["#000000", "#000000", "#000000", "#800080", "#FF0000", "#FFFF00"]
        )
    inset_dict = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "50%",  # height : 50%
        "loc": "lower left",
        "bbox_to_anchor": (1.0125, 0.0, 1, 1),
        "borderpad": 0,
    }

    if not (set(all_genes_in_pair) <= set(adata.var_names)):
        raise ValueError(
            "adata doesn't include all genes in gene_pairs_mat. Make sure all genes are included in adata.var_names."
        )

    flat_res = pd.DataFrame(columns=["x", "y", "den", "type"])
    ridge_curve = pd.DataFrame(columns=["x", "y", "type"])
    xy = pd.DataFrame()

    id = 0
    for gene_pairs_ind, gene_pairs in enumerate(pairs_mat):
        f_ini_ind = (grid_num**2) * id
        r_ini_ind = grid_num * id

        gene_pair_name = gene_pairs[0] + "->" + gene_pairs[1]

        if xkey.startswith("jacobian"):
            J_df = get_jacobian(
                adata,
                gene_pairs[0],
                gene_pairs[1],
            )
            jkey = gene_pairs[0] + "->" + gene_pairs[1] + "_jacobian"

            x = flatten(J_df[jkey])
        else:
            x = flatten(adata[:, gene_pairs[0]].layers[xkey])

        if ykey.startswith("jacobian"):
            J_df = get_jacobian(
                adata,
                gene_pairs[0],
                gene_pairs[1],
            )
            jkey = gene_pairs[0] + "->" + gene_pairs[1] + "_jacobian"

            y_ori = flatten(J_df[jkey])
        else:
            y_ori = flatten(adata[:, gene_pairs[1]].layers[ykey])

        if drop_zero_cells:
            finite = np.isfinite(x + y_ori)
            nonzero = np.abs(x) + np.abs(y_ori) > 0

            valid_ids = np.logical_and(finite, nonzero)
        else:
            valid_ids = np.isfinite(x + y_ori)

        x, y_ori = x[valid_ids], y_ori[valid_ids]

        if log:
            x, y_ori = x if sum(x < 0) else np.log(np.array(x) + 1), y_ori if sum(y_ori) < 0 else np.log(
                np.array(y_ori) + 1
            )

        if delay != 0:
            x = x[:-delay]
            y = y_ori[delay:]
        else:
            y = y_ori

        # add LaTex equation in matlibplot

        bandwidth = [bandwidth_nrd(x), bandwidth_nrd(y)]

        if 0 in bandwidth:
            max_vec = [max(x), max(y)]
            bandwidth[bandwidth == 0] = max_vec[bandwidth == 0] / grid_num

        # den_res[0, 0] is at the lower bottom; dens[1, 4]: is the 2nd on x-axis and 5th on y-axis
        x_meshgrid, y_meshgrid, den_res = kde2d(
            x, y, n=[grid_num, grid_num], lims=[min(x), max(x), min(y), max(y)], h=bandwidth
        )
        den_res = np.array(den_res)

        den_x = np.sum(den_res, axis=1)  # condition on each input x, sum over y
        max_ind = 0

        for i in range(len(x_meshgrid)):
            tmp = den_res[i] / den_x[i]  # condition on each input x, normalize over y
            max_val = max(tmp)
            min_val = min(tmp)

            rescaled_val = (tmp - min_val) / (max_val - min_val)
            if np.sum(den_x[i] != 0):
                max_ind = np.argmax(rescaled_val)  # the maximal y ind condition on input x

            res_Row = pd.DataFrame(
                [[x_meshgrid[i], y_meshgrid[max_ind], gene_pair_name]],
                columns=["x", "y", "type"],
                index=[r_ini_ind + i],
            )
            ridge_curve = pd.concat([ridge_curve, res_Row])

            res_row = pd.DataFrame(
                {
                    "x": x_meshgrid[i],
                    "y": y_meshgrid,
                    "den": rescaled_val,
                    "type": gene_pair_name,
                },
                index=[i * len(x_meshgrid) + np.arange(len(y_meshgrid)) + f_ini_ind],
            )

            flat_res = pd.concat([flat_res, res_row])

        cur_data = pd.DataFrame({"x": x, "y": y, "type": gene_pair_name})
        xy = pd.concat([xy, cur_data], axis=0)

        id = id + 1

    gene_pairs_num = len(flat_res.type.unique())

    n_col = gene_pairs_num if n_col is None else n_col

    if n_row * n_col < gene_pairs_num:
        raise ValueError("The number of row or column specified is less than the gene pairs")
    figsize = (figsize[0] * n_col, figsize[1] * n_row) if figsize is not None else (4 * n_col, 4 * n_row)
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize, sharex=False, sharey=False, squeeze=False)

    fit_dict = None
    if fit_curve:
        fit_dict = {}

    def scale_func(x, X, grid_num):
        return grid_num * (x - np.min(X)) / (np.max(X) - np.min(X))

    for x, flat_res_type in enumerate(flat_res.type.unique()):
        gene_pairs = flat_res_type.split("->")

        flat_res_subset = flat_res[flat_res["type"] == flat_res_type]
        ridge_curve_subset = ridge_curve[ridge_curve["type"] == flat_res_type]
        xy_subset = xy[xy["type"] == flat_res_type]

        x_val, y_val = flat_res_subset["x"], flat_res_subset["y"]

        i, j = x % n_row, x // n_row  # %: remainder; //: integer division

        values = flat_res_subset["den"].values.reshape(grid_num, grid_num).T

        axins = inset_axes(axes[i, j], bbox_transform=axes[i, j].transAxes, **inset_dict)

        ext_lim = (min(x_val), max(x_val), min(y_val), max(y_val))
        im = axes[i, j].imshow(
            values,
            interpolation="mitchell",
            origin="lower",
            extent=ext_lim if show_extent else None,
            cmap=cmap,
        )
        cb = fig.colorbar(im, cax=axins)
        cb.set_alpha(1)
        cb.draw_all()
        cb.locator = MaxNLocator(nbins=3, integer=False)
        cb.update_ticks()

        closest_x_ind = np.array([np.searchsorted(x_meshgrid, i) for i in xy_subset["x"].values])
        closest_y_ind = np.array([np.searchsorted(y_meshgrid, i) for i in xy_subset["y"].values])
        valid_ids = np.logical_and(closest_x_ind < grid_num, closest_y_ind < grid_num)
        axes[i, j].scatter(closest_x_ind[valid_ids], closest_y_ind[valid_ids], color="gray", alpha=0.1, s=1)

        if xkey.startswith("jacobian"):
            if stacked_fraction:
                axes[i, j].set_xlabel(r"$\frac{\partial f_{%s}}{\partial x_{%s}}$" % (gene_pairs[1], gene_pairs[0]))
            else:
                axes[i, j].set_xlabel(r"$\partial f_{%s} / {\partial x_{%s}$" % (gene_pairs[1], gene_pairs[0]))
        else:
            axes[i, j].set_xlabel(gene_pairs[0] + rf" (${xkey}$)")
        if ykey.startswith("jacobian"):
            if stacked_fraction:
                axes[i, j].set_ylabel(r"$\frac{\partial f_{%s}}{\partial x_{%s}}$" % (gene_pairs[1], gene_pairs[0]))
                axes[i, j].title.set_text(
                    r"$\rho(\frac{\partial f_{%s}}{\partial x_{%s}})$" % (gene_pairs[1], gene_pairs[0])
                )
            else:
                axes[i, j].set_ylabel(r"$\partial f_{%s} / \partial x_{%s}$" % (gene_pairs[1], gene_pairs[0]))
                axes[i, j].title.set_text(r"$\rho(\partial f_{%s} / \partial x_{%s})$" % (gene_pairs[1], gene_pairs[0]))
        else:
            axes[i, j].set_ylabel(gene_pairs[1] + rf" (${ykey}$)")
            axes[i, j].title.set_text(rf"$\rho_{{{gene_pairs[1]}}}$ (${ykey}$)")

        if show_ridge:
            axes[i, j].plot(ridge_curve_subset["x"].values, ridge_curve_subset["y"].values, color="red")

        if show_rug:
            xy_subset = xy_subset.query("x > @ext_lim[0] & x < @ext_lim[1] & y > @ext_lim[2] & y < @ext_lim[3]")
            seaborn.rugplot(xy_subset["x"].values, height=0.01, axis="x", ax=axes[i, j], c="darkred", alpha=0.1)
            seaborn.rugplot(xy_subset["y"].values, height=0.01, axis="y", ax=axes[i, j], c="darkred", alpha=0.1)

        if not show_extent:
            despline_all(axes[i, j])

        # for some reason,  I have add an extra element at the begining for the ticklabels
        xlabels = list(np.linspace(ext_lim[0], ext_lim[1], 5))
        ylabels = list(np.linspace(ext_lim[2], ext_lim[3], 5))

        # zero indicator
        if zero_indicator:
            axes[i, j].plot(
                scale_func([np.min(xlabels), np.max(xlabels)], xlabels, grid_num),
                scale_func(np.zeros(2), ylabels, grid_num),
                zero_line_style,
                linewidth=zero_line_width,
            )

        # curve fitting
        if fit_curve:
            if fit_mode == "hill":
                if ykey.startswith("jacobian"):
                    x_grid, y_mean, y_sigm = kde2d_to_mean_and_sigma(
                        np.array(x_val), np.array(y_val), flat_res_subset["den"].values
                    )
                    fix_g = 0 if no_degradation else None

                    pdict_act, msd_act = fit_hill_grad(x_grid, y_mean, "act", y_sigm=y_sigm, fix_g=fix_g)
                    pdict_inh, msd_inh = fit_hill_grad(x_grid, y_mean, "inh", y_sigm=y_sigm, fix_g=fix_g, x_shift=1e-4)

                    if msd_act < msd_inh:
                        fit_type = "act"
                        pdict = pdict_act
                        msd = msd_act
                    else:
                        fit_type = "inh"
                        pdict = pdict_inh
                        msd = msd_inh

                    # adata.uns[f'jacobian_response_fit_{gene_pairs[0]}_{gene_pairs[1]}'] = {
                    fit_dict[f"{gene_pairs[0]}_{gene_pairs[1]}"] = {
                        "genes": gene_pairs,
                        "mode": "hill",
                        "type": fit_type,
                        "msd": msd,
                        "param": pdict,
                        "x_grid": x_grid,
                    }

                    xs = np.linspace(x_grid[0], x_grid[-1], 100)
                    func = hill_act_grad if fit_type == "act" else hill_inh_grad

                    xplot = scale_func(xs, xlabels, grid_num)
                    if mean_style is not None:
                        axes[i, j].plot(
                            scale_func(x_grid, xlabels, grid_num), scale_func(y_mean, ylabels, grid_num), mean_style
                        )
                    axes[i, j].plot(
                        xplot,
                        scale_func(func(xs, pdict["A"], pdict["K"], pdict["n"], pdict["g"]), ylabels, grid_num),
                        curve_style,
                        linewidth=curve_lw,
                    )
                else:
                    raise NotImplementedError("The hill function can be applied to the Jacobian response heatmap only.")
            else:
                raise NotImplementedError("Currently, the fit_mode can only be served by the hill function.")

        # set the x/y ticks
        inds = np.linspace(0, grid_num - 1, 5, endpoint=True)
        axes[i, j].set_xticks(inds)
        axes[i, j].set_yticks(inds)

        if ext_format is None:
            if ext_lim[1] < 1e-2:
                xlabels = ["{:.2e}".format(i) for i in xlabels]
            else:
                xlabels = [np.round(i, 2) for i in xlabels]
            if ext_lim[3] < 1e-2:
                ylabels = ["{:.2e}".format(i) for i in ylabels]
            else:
                ylabels = [np.round(i, 2) for i in ylabels]
        else:
            if type(ext_format) == list:
                if ext_format[0] is None:
                    if ext_lim[1] < 1e-2:
                        xlabels = ["{:.2e}".format(i) for i in xlabels]
                    else:
                        xlabels = [np.round(i, 2) for i in xlabels]
                else:
                    xlabels = [ext_format[0].format(i) for i in xlabels]

                if ext_format[1] is None:
                    if ext_lim[3] < 1e-2:
                        ylabels = ["{:.2e}".format(i) for i in ylabels]
                    else:
                        ylabels = [np.round(i, 2) for i in ylabels]
                else:
                    ylabels = [ext_format[1].format(i) for i in ylabels]
            else:
                xlabels = [ext_format.format(i) for i in xlabels]
                ylabels = [ext_format.format(i) for i in ylabels]

        if ext_lim[1] < 1e-2:
            axes[i, j].set_xticklabels(xlabels, rotation=30, ha="right")
        else:
            axes[i, j].set_xticklabels(xlabels)

        axes[i, j].set_yticklabels(ylabels)

    plt.subplots_adjust(left=0.1, right=1, top=0.80, bottom=0.1, wspace=0.1)
    save_show_ret("scatters", save_show_or_return, save_kwargs)

    list_for_return = []

    if return_data:
        if fit_dict is None:
            list_for_return += [flat_res, flat_res_subset, ridge_curve_subset]
        else:
            list_for_return += [flat_res, flat_res_subset, ridge_curve_subset, fit_dict]
    else:
        adata.uns["response"] = {
            "flat_res": flat_res,
            "flat_res_subset": flat_res_subset,
            "ridge_curve_subset": ridge_curve_subset,
        }
        if fit_dict is not None:
            adata.uns["response"]["fit_curve"] = fit_dict

    if list_for_return:
        return tuple(list_for_return)


def plot_hill_function(
    adata: AnnData,
    pairs_mat: Optional[np.ndarray] = None,
    normalize: bool = True,
    n_row: int = 1,
    n_col: Optional[int] = None,
    figsize: Tuple[float, float] = (6, 4),
    linewidth: float = 2,
    save_show_or_return: Literal["save", "show", "both", "all"] = "show",
    save_kwargs: Dict[str, Any] = {},
    **plot_kwargs,
) -> None:
    """Plot the hill function curve generated by `dynamo.pl.response`.

    Args:
        adata: an AnnData object.
        pairs_mat: a matrix where each row is the gene pair and the first column is the hypothetical source or regulator
            while the second column represents the hypothetical target. The name in this matrix should match the name in
            the gene_short_name column of the adata object. If None, all gene pairs in
            `adata.uns["response"]["fit_curve"]` would be used. Defaults to None.
        normalize: whether to normalize the curve. Defaults to True.
        n_row: the number of subplot rows. Defaults to 1.
        n_col: the number of subplot cols. If None, it would be calculated to show all subplots. Defaults to None.
        figsize: the size of the figure. Defaults to (6, 4).
        linewidth: the line width of the curve. Defaults to 2.
        save_show_or_return: whether to save or show the figure. Could be one of "save", "show", "both", or "all".
            "both" and "all" have the same effect. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.
        **plot_kwargs: any other kwargs passed to `pyplot.plot`.

    Raises:
        ValueError: no `"response"` data found in `adata.uns`.
        ValueError: no `"fit_curve"` data found in `adata.uns["response"]`.
        ValueError: adata does not contain gene data specified in pairs_mat
        ValueError: `n_col * n_row` is less than the number of gene pairs.
        ValueError: the gene specified in `pairs_mat` is not found in `adata.uns["response"]["fit_curve"]`.
        NotImplementedError: invalid `fit_type` in `adata.uns["response"]["fit_curve"]`.
        NotImplementedError: invalid `mode` in `adata.uns["response"]["fit_curve"]`.
    """
    import matplotlib.pyplot as plt

    if "response" not in adata.uns.keys():
        raise ValueError("`response` is not found in `.uns`. Run `pl.response` first.")
    if "fit_curve" not in adata.uns["response"].keys():
        raise ValueError("`fit_curve` is not found. Run `pl.response` with `fit_curve=True` first.")

    fit_dict = adata.uns["response"]["fit_curve"]
    if pairs_mat is None:
        pairs_mat = []
        for pairs in fit_dict.keys():
            genes = pairs.split("_")
            pairs_mat.append([genes[0], genes[1]])

    all_genes_in_pair = np.unique(pairs_mat)

    if not (set(all_genes_in_pair) <= set(adata.var_names)):
        raise ValueError(
            "adata doesn't include all genes in gene_pairs_mat. Make sure all genes are included in adata.var_names."
        )

    gene_pairs_num = len(pairs_mat)
    n_col = gene_pairs_num if n_col is None else n_col

    if n_row * n_col < gene_pairs_num:
        raise ValueError("The number of row or column specified is less than the gene pairs")
    figsize = (figsize[0] * n_col, figsize[1] * n_row) if figsize is not None else (4 * n_col, 4 * n_row)
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize, sharex=False, sharey=False, squeeze=False)

    for pair_i, gene_pairs in enumerate(pairs_mat):
        i, j = pair_i % n_row, pair_i // n_row  # %: remainder; //: integer division

        gene_pair_name = gene_pairs[0] + "->" + gene_pairs[1]

        key = f"{gene_pairs[0]}_{gene_pairs[1]}"
        if key not in fit_dict.keys():
            raise ValueError(f"The gene pair {key} is not found in the dictionary.")

        mode = fit_dict[key]["mode"]
        x_grid = fit_dict[key]["x_grid"]
        fit_type = fit_dict[key]["type"]
        A, K, n, g = (
            fit_dict[key]["param"]["A"],
            fit_dict[key]["param"]["K"],
            fit_dict[key]["param"]["n"],
            fit_dict[key]["param"]["g"],
        )
        if normalize:
            A = 1.0

        if mode == "hill":
            xs = np.linspace(x_grid[0], x_grid[-1], 100)
            if fit_type == "act":
                func = hill_act_func
            elif fit_type == "inh":
                func = hill_inh_func
            else:
                raise NotImplementedError(f"Unknown hill function type `{fit_type}`")

            axes[i, j].plot(xs, func(xs, A, K, n, g), linewidth=linewidth, **plot_kwargs)
            axes[i, j].set_xlabel(gene_pairs[0])
            axes[i, j].set_ylabel(r"$f_{%s}$" % gene_pairs[1])
            axes[i, j].set_title(gene_pair_name)
        else:
            raise NotImplementedError(f"The fit mode `{mode}` is not supported.")

    plt.subplots_adjust(left=0.1, right=1, top=0.80, bottom=0.1, wspace=0.1)
    save_show_ret("scatters", save_show_or_return, save_kwargs)


def causality(
    adata: AnnData,
    pairs_mat: np.ndarray,
    hessian_matrix: bool = False,
    xkey: Optional[str] = None,
    ykey: Optional[str] = None,
    zkey: Optional[str] = None,
    log: bool = True,
    drop_zero_cells: bool = False,
    delay: int = 0,
    k: int = 30,
    normalize: bool = True,
    grid_num: int = 25,
    n_row: int = 1,
    n_col: Optional[int] = None,
    cmap: Union[str, Colormap, None] = "viridis",
    show_rug: bool = True,
    show_extent: bool = False,
    ext_format: Optional[List[str]] = None,
    stacked_fraction: bool = False,
    figsize: Tuple[float, float] = (6, 4),
    save_show_or_return: Literal["save", "show", "both", "all"] = "show",
    save_kwargs: Dict[str, Any] = {},
    return_data: bool = False,
    **kwargs,
) -> Optional[pd.DataFrame]:
    """Plot the heatmap for the expected value `z(t)` given `x` and `y` data.

    Args:
        adata: an AnnData object.
        pairs_mat: a matrix where each row is the gene pair and the first column is the hypothetical source or regulator
            while the second column represents the hypothetical target. The name in this matrix should match the name in
            the gene_short_name column of the adata object.
        hessian_matrix: whether to visualize the Hessian matrix results (row, column are the regulatory/co-regulator
            gene expression while the color is the results of the Hessian to the effector). Defaults to False.
        xkey: the layer key of `x` data (regulator gene 1). If None, dynamo's default naming rule would be used.
            Defaults to None.
        ykey: the layer key of `y` data (regulator gene 2).  If None, dynamo's default naming rule would be used.
            Defaults to None.
        zkey: the layer key of `z` data (the hypothetical target).  If None, dynamo's default naming rule would be used.
            Defaults to None.
        log: whether to perform log transformation (using log(expression + 1)) before calculating density estimates.
            Defaults to True.
        drop_zero_cells: whether to drop cells that with zero expression for either the potential regulator or potential
            target. This can signify the relationship between potential regulators and targets, speed up the
            calculation, but at the risk of ignoring strong inhibition effects from certain regulators to targets.
            Defaults to False.
        delay: the time delay between the source and target gene. Always zero because we don't have real time-series.
            Defaults to 0.
        k: number of k-nearest neighbors used in calculating 2-D kernel density. Defaults to 30.
        normalize: whether to row-scale the data. Defaults to True.
        grid_num: the number of grid when creating the lagged DREVI plot. Defaults to 25.
        n_row: the number of rows used to layout the faceted cluster panels. Defaults to 1.
        n_col: the number of columns used to layout the faceted cluster panels. Defaults to None.
        cmap: the color map used to plot the heatmap. Could be the name of the color map or a matplotlib color map
            object. If None, the color map would be generated automatically. Defaults to "viridis".
        show_rug: whether to plot marginal distributions by drawing ticks along the x and y axes. Defaults to True.
        show_extent: whether to extend the figure. If False, `show_ridge` and `show_rug` would be set to False
            automatically. Defaults to False.
        ext_format: the string/list of strings (the first is for x and second for y labels) that will be used to format
            the ticks on x or y-axis. If it is None or one of the element in the list is None, the default setting will
            be used. Defaults to None.
        stacked_fraction: if True the jacobian will be represented as a stacked fraction in the title, otherwise a
            linear fraction style is used. Defaults to False.
        figsize: the size of the figure. Defaults to (6, 4).
        save_show_or_return: whether to save or show the figure. If "both", it will save and plot the figure at the same time. If
            "all", the figure will be saved, displayed and the associated axis and other object will be return.
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}. Defaults to {}.
        return_data: whether to return the calculated causality data. Defaults to False.

    Raises:
        ValueError: no preprocessing data in adata.uns
        ValueError: No layers named as `xkey`, `ykey`, or `zkey`.
        ValueError: adata does not contain gene data specified in pairs_mat
        ValueError: `n_col * n_row` is less than number of gene pairs

    Returns:
        None would be returned by default. If `return_data` is set to be True, the causality data, a DataFrame with
        columns ("x": x coordination, "y": y coordination, "expected_z": expected z coordination, "pair": gene pairs) would
        be returned.
    """

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if show_extent is False:
        show_rug = False

    all_genes_in_pair = np.unique(pairs_mat)

    if "pp" not in adata.uns_keys():
        raise ValueError("You must first run dyn.pp.recipe_monocle and dyn.tl.moments before running this function.")

    if xkey is None:
        xkey = "M_t" if adata.uns["pp"]["has_labeling"] else "M_s"
    if ykey is None:
        ykey = "M_t" if adata.uns["pp"]["has_labeling"] else "M_s"
    if zkey is None:
        if hessian_matrix:
            zkey = "hessian_pca"
        else:
            zkey = "M_n" if adata.uns["pp"]["has_labeling"] else "M_u"
    if cmap is None:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "causality", ["#008000", "#ADFF2F", "#FFFF00", "#FFA500", "#FFC0CB", "#FFFFFE"]
        )

    if not set([xkey, ykey, zkey]) <= set(adata.layers.keys()).union(set(["jacobian", "hessian_pca"])):
        raise ValueError(
            f"adata.layers doesn't have {xkey, ykey, zkey} layers. Please specify the correct layers or "
            "perform relevant preprocessing and vector field analyses first."
        )

    inset_dict = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "50%",  # height : 50%
        "loc": "lower left",
        "bbox_to_anchor": (1.0125, 0.0, 1, 1),
        "borderpad": 0,
    }
    if not (set(all_genes_in_pair) <= set(adata.var_names)):
        raise ValueError(
            "adata doesn't include all genes in gene_pairs_mat. Make sure all genes are included in adata.var_names."
        )

    flat_res = pd.DataFrame(columns=["x", "y", "expected_z", "pair"])
    xy = pd.DataFrame()

    id = 0
    for gene_pairs_ind in range(0, len(pairs_mat)):
        gene_pairs = pairs_mat[gene_pairs_ind, :]
        f_ini_ind = (grid_num**2) * id

        gene_pair_name = reduce(lambda a, b: a + "->" + b, gene_pairs)

        if xkey.startswith("jacobian"):
            J_df = get_jacobian(
                adata,
                gene_pairs[0],
                gene_pairs[1],
            )
            jkey = gene_pairs[0] + "->" + gene_pairs[1] + "_jacobian"

            x = flatten(J_df[jkey])
        else:
            x = flatten(adata[:, gene_pairs[0]].layers[xkey])

        if ykey.startswith("jacobian"):
            J_df = get_jacobian(
                adata,
                gene_pairs[0],
                gene_pairs[1],
            )
            jkey = gene_pairs[0] + "->" + gene_pairs[1] + "_jacobian"

            y_ori = flatten(J_df[jkey])
        else:
            y_ori = flatten(adata[:, gene_pairs[1]].layers[ykey])

        # if only 2 genes, it is causality plot; otherwise it comb_logic plot.
        if zkey.startswith("jacobian"):
            J_df = get_jacobian(
                adata,
                gene_pairs[0],
                gene_pairs[1],
            )

            if len(gene_pairs) == 3:
                main_warning(
                    "your gene_pair_mat has three column, only the genes from first two columns will be used "
                    "to retrieve Jacobian."
                )
            jkey = gene_pairs[0] + "->" + gene_pairs[1] + "_jacobian"

            z_ori = flatten(J_df[jkey])
        else:
            if hessian_matrix:
                res = run_hessian(
                    adata,
                    regulators=gene_pairs[0],
                    coregulators=gene_pairs[1],
                    effector=gene_pairs[2],
                    store_in_adata=False,
                )
                z_ori = np.array(res["hessian_gene"])
            else:
                z_ori = (
                    flatten(adata[:, gene_pairs[2]].layers[zkey])
                    if len(gene_pairs) == 3
                    else flatten(adata[:, gene_pairs[1]].layers[zkey])
                )

        if drop_zero_cells:
            finite = np.isfinite(x + y_ori + z_ori)
            nonzero = np.abs(x) + np.abs(y_ori) + np.abs(z_ori) > 0

            valid_ids = np.logical_and(finite, nonzero)
        else:
            valid_ids = np.isfinite(x + y_ori + z_ori)

        x, y_ori, z_ori = x[valid_ids], y_ori[valid_ids], z_ori[valid_ids]

        if log:
            x = x if sum(x < 0) else np.log(np.array(x) + 1)
            y_ori = y_ori if sum(y_ori) < 0 else np.log(np.array(y_ori) + 1)
            z_ori = z_ori if sum(z_ori) < 0 or hessian_matrix else np.log(np.array(z_ori) + 1)

        if delay != 0:
            x = x[:-delay]
            y = y_ori[delay:]
            z = z_ori[delay - 1 : -1]
        else:
            y = y_ori
            z = z_ori

        # for xy
        cur_data = pd.DataFrame({"x": x, "y": y, "z": z, "pair": gene_pair_name})
        xy = pd.concat([xy, cur_data], axis=0)

        x_meshgrid = np.linspace(min(x), max(x), grid_num, endpoint=True)
        y_meshgrid = np.linspace(min(y), max(y), grid_num, endpoint=True)

        xv, yv = np.meshgrid(x_meshgrid, y_meshgrid)
        xp = xv.reshape((1, -1)).tolist()
        yp = yv.reshape((1, -1)).tolist()
        xy_query = np.array(xp + yp).T
        tree_xy = ss.cKDTree(cur_data[["x", "y"]])
        dist_mat, idx_mat = tree_xy.query(xy_query, k=k + 1)

        for i in range(dist_mat.shape[0]):
            subset_dat = cur_data.iloc[idx_mat[i, 1:], 2]  # get the z value
            u = (
                np.exp(-dist_mat[i, 1:])
                if sum(dist_mat[i] > 0) == 0
                else np.exp(-dist_mat[i, 1:] / np.min(dist_mat[i][dist_mat[i] > 0]))
            )
            w = u / np.sum(u)

            tmp = sum(np.array(w) * np.array(subset_dat))
            res_Row = pd.DataFrame(
                [[xy_query[i, 0], xy_query[i, 1], tmp, gene_pair_name]],
                columns=["x", "y", "expected_z", "pair"],
                index=[f_ini_ind + i],
            )
            flat_res = pd.concat([flat_res, res_Row])
        if normalize:
            vals = flat_res["expected_z"][(f_ini_ind) : (f_ini_ind + len(dist_mat))]
            max_val = max(vals.dropna().values.reshape(1, -1)[0])
            if not np.isfinite(max_val):
                max_val = 1e10

            flat_res.iloc[(f_ini_ind) : (f_ini_ind + len(dist_mat)), 2] = (
                flat_res.iloc[(f_ini_ind) : (f_ini_ind + len(dist_mat)), :]["expected_z"] / max_val
            )

        id = id + 1

    gene_pairs_num = len(flat_res.pair.unique())

    n_col = gene_pairs_num if n_col is None else n_col

    if n_row * n_col < gene_pairs_num:
        raise ValueError("The number of row or column specified is less than the gene pairs")

    figsize = (figsize[0] * n_col, figsize[1] * n_row) if figsize is not None else (4 * n_col, 4 * n_row)
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize, sharex=False, sharey=False, squeeze=False)

    for x, flat_res_type in enumerate(flat_res.pair.unique()):
        gene_pairs = flat_res_type.split("->")

        flat_res_subset = flat_res[flat_res["pair"] == flat_res_type]
        xy_subset = xy[xy["pair"] == flat_res_type]

        x_val, y_val = flat_res_subset["x"], flat_res_subset["y"]

        i, j = x % n_row, x // n_row  # %: remainder; //: integer division

        values = flat_res_subset["expected_z"].values.reshape(xv.shape)

        axins = inset_axes(axes[i, j], bbox_transform=axes[i, j].transAxes, **inset_dict)

        ext_lim = (min(x_val), max(x_val), min(y_val), max(y_val))
        v_min, v_max, v_abs_max = min(values.flatten()), max(values.flatten()), max(abs(values.flatten()))
        im = axes[i, j].imshow(
            values,
            interpolation="mitchell",
            origin="lower",
            extent=ext_lim if show_extent else None,
            cmap=cmap,
            vmin=v_min if v_min >= 0 else -v_abs_max,
            vmax=v_max if v_min >= 0 else v_abs_max,
        )
        cb = fig.colorbar(im, cax=axins)
        cb.set_alpha(1)
        cb.draw_all()
        cb.locator = MaxNLocator(nbins=3, integer=False)
        cb.update_ticks()

        closest_x_ind = np.array([np.searchsorted(x_meshgrid, i) for i in xy_subset["x"].values])
        closest_y_ind = np.array([np.searchsorted(y_meshgrid, i) for i in xy_subset["y"].values])
        valid_ids = np.logical_and(closest_x_ind < grid_num, closest_y_ind < grid_num)
        axes[i, j].scatter(closest_x_ind[valid_ids], closest_y_ind[valid_ids], color="gray", alpha=0.1, s=1)

        if xkey.startswith("jacobian"):
            if stacked_fraction:
                axes[i, j].set_xlabel(r"$\frac{\partial f_{%s}}{\partial x_{%s}}$" % (gene_pairs[1], gene_pairs[0]))
            else:
                axes[i, j].set_xlabel(r"$\partial f_{%s} / \partial x_{%s}$" % (gene_pairs[1], gene_pairs[0]))
        else:
            axes[i, j].set_xlabel(gene_pairs[0] + rf" (${xkey}$)")

        if ykey.startswith("jacobian"):
            if stacked_fraction:
                axes[i, j].set_ylabel(r"$\frac{\partial f_{%s}}{\partial x_{%s}}$" % (gene_pairs[1], gene_pairs[0]))
            else:
                axes[i, j].set_ylabel(r"$\partial f_{%s} / \partial x_{%s}$" % (gene_pairs[1], gene_pairs[0]))
        else:
            axes[i, j].set_ylabel(gene_pairs[1] + rf" (${ykey}$)")

        if zkey.startswith("jacobian"):
            if stacked_fraction:
                axes[i, j].title.set_text(
                    r"$E(\frac{\partial f_{%s}}{\partial x_{%s}})$" % (gene_pairs[1], gene_pairs[0])
                )
            else:
                axes[i, j].title.set_text(r"$E(\partial f_{%s} / \partial x_{%s})$" % (gene_pairs[1], gene_pairs[0]))
        else:
            if hessian_matrix:
                if len(gene_pairs) == 3:
                    axes[i, j].title.set_text(rf"$Hessian_{{{gene_pairs[2]}}}$ (${zkey}$)")
                else:
                    axes[i, j].title.set_text(rf"$Hessian_{{{gene_pairs[1]}}}$ (${zkey}$)")
            else:
                if len(gene_pairs) == 3:
                    axes[i, j].title.set_text(rf"$E_{{{gene_pairs[2]}}}$ (${zkey}$)")
                else:
                    axes[i, j].title.set_text(rf"$E_{{{gene_pairs[1]}}}$ (${zkey}$)")

        if show_rug:
            xy_subset = xy_subset.query("x > @ext_lim[0] & x < @ext_lim[1] & y > @ext_lim[2] & y < @ext_lim[3]")
            seaborn.rugplot(xy_subset["x"].values, height=0.01, axis="x", ax=axes[i, j], c="darkred", alpha=0.1)
            seaborn.rugplot(xy_subset["y"].values, height=0.01, axis="y", ax=axes[i, j], c="darkred", alpha=0.1)

        #         axes[i, j].plot(flat_res_subset['x'], [0.01]*len(flat_res_subset['x']), '|', color='k')
        #         axes[i, j].plot([0.01]*len(flat_res_subset['z']), flat_res_subset['z'], '|', color='k')
        if not show_extent:
            despline_all(axes[i, j])

        # for some reason,  I have add an extra element at the beginingfor the ticklabels
        xlabels = list(np.linspace(ext_lim[0], ext_lim[1], 5))
        ylabels = list(np.linspace(ext_lim[2], ext_lim[3], 5))

        # set the x/y ticks
        inds = np.linspace(0, grid_num - 1, 5, endpoint=True)
        axes[i, j].set_xticks(inds)
        axes[i, j].set_yticks(inds)

        if ext_format is None:
            if ext_lim[1] < 1e-2:
                xlabels = ["{:.2e}".format(i) for i in xlabels]
            else:
                xlabels = [np.round(i, 2) for i in xlabels]
            if ext_lim[3] < 1e-2:
                ylabels = ["{:.2e}".format(i) for i in ylabels]
            else:
                ylabels = [np.round(i, 2) for i in ylabels]
        else:
            if type(ext_format) == list:
                if ext_format[0] is None:
                    if ext_lim[1] < 1e-2:
                        xlabels = ["{:.2e}".format(i) for i in xlabels]
                    else:
                        xlabels = [np.round(i, 2) for i in xlabels]
                else:
                    xlabels = [ext_format[0].format(i) for i in xlabels]

                if ext_format[1] is None:
                    if ext_lim[3] < 1e-2:
                        ylabels = ["{:.2e}".format(i) for i in ylabels]
                    else:
                        ylabels = [np.round(i, 2) for i in ylabels]
                else:
                    ylabels = [ext_format[1].format(i) for i in ylabels]
            else:
                xlabels = [ext_format.format(i) for i in xlabels]
                ylabels = [ext_format.format(i) for i in ylabels]

        if ext_lim[1] < 1e-2:
            axes[i, j].set_xticklabels(xlabels, rotation=30, ha="right")
        else:
            axes[i, j].set_xticklabels(xlabels)

        axes[i, j].set_yticklabels(ylabels)

    # plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    plt.subplots_adjust(left=0.1, right=1, top=0.80, bottom=0.1, wspace=0.1)
    save_show_ret("scatters", save_show_or_return, save_kwargs)

    if return_data:
        return flat_res
    else:
        adata.uns[kwargs.pop("save_key", "causality")] = flat_res


def comb_logic(
    adata: AnnData,
    pairs_mat: np.ndarray,
    xkey: Optional[str] = None,
    ykey: Optional[str] = None,
    zkey: Optional[str] = None,
    log: bool = True,
    drop_zero_cells: bool = False,
    delay: int = 0,
    grid_num: int = 25,
    n_row: int = 1,
    n_col: Optional[int] = None,
    cmap: Union[str, Colormap, None] = "bwr",
    normalize: bool = True,
    k: int = 30,
    show_rug: bool = True,
    show_extent: bool = False,
    ext_format: Optional[List[str]] = None,
    stacked_fraction: bool = False,
    figsize: Tuple[float, float] = (6, 4),
    save_show_or_return: Literal["save", "show", "both", "all"] = "show",
    save_kwargs: Dict[str, Any] = {},
    return_data: bool = False,
) -> Optional[pd.DataFrame]:
    """Plot the combinatorial influence of two genes `x`, `y` to the target `z`.

    This plotting function tries to intuitively visualize the influence from genes `x` and `y` to the target `z`.
    Firstly, we divide the expression space for `x` and `y` based on grid_num and then we estimate the k-nearest
    neighbor for each of the grid. We then use a Gaussian kernel to estimate the expected value for `z`. It is then
    displayed in two dimension with `x` and `y` as two axis and the color represents the value of the expected of `z`.
    This function accepts a matrix where each row is the gene pair and the target genes for this pair. The first column
    is the first hypothetical source or regulator, the second column represents the second hypothetical target while the
    third column represents the hypothetical target gene. The name in this matrix should match the name in the
    gene_short_name column of the cds_subset object.

    Args:
        adata: an AnnData object.
        pairs_mat: a matrix where each row is the gene pair and the first column is the hypothetical source or regulator
            while the second column represents the hypothetical target. The name in this matrix should match the name in
            the gene_short_name column of the adata object.
        xkey: the layer key of `x` data (regulator gene 1). If None, dynamo's default naming rule would be used.
            Defaults to None.
        ykey: the layer key of `y` data (regulator gene 2).  If None, dynamo's default naming rule would be used.
            Defaults to None.
        zkey: the layer key of `z` data (the hypothetical target).  If None, dynamo's default naming rule would be used.
            Defaults to None.
        log: whether to perform log transformation (using log(expression + 1)) before calculating density estimates.
            Defaults to True.
        drop_zero_cells: whether to drop cells that with zero expression for either the potential regulator or potential
            target. This can signify the relationship between potential regulators and targets, speed up the
            calculation, but at the risk of ignoring strong inhibition effects from certain regulators to targets.
            Defaults to False.
        delay: the time delay between the source and target gene. Always zero because we don't have real time-series.
            Defaults to 0.
        grid_num: the number of grid when creating the lagged DREVI plot. Defaults to 25.
        n_row: the number of rows used to layout the faceted cluster panels. Defaults to 1.
        n_col: the number of columns used to layout the faceted cluster panels. Defaults to None.
        cmap: the color map used to plot the heatmap. Could be the name of the color map or a matplotlib color map
            object. If None, the color map would be generated automatically. Defaults to "viridis".
        normalize: whether to row-scale the data. Defaults to True.
        k: number of k-nearest neighbors used in calculating 2-D kernel density. Defaults to 30.
        show_rug: whether to plot marginal distributions by drawing ticks along the x and y axes. Defaults to True.
        show_extent: whether to extend the figure. If False, `show_ridge` and `show_rug` would be set to False
            automatically. Defaults to False.
        ext_format: the string/list of strings (the first is for x and second for y labels) that will be used to format
            the ticks on x or y-axis. If it is None or one of the element in the list is None, the default setting will
            be used. Defaults to None.
        stacked_fraction: if True the jacobian will be represented as a stacked fraction in the title, otherwise a
            linear fraction style is used. Defaults to False.
        figsize: the size of the figure. Defaults to (6, 4).
        save_show_or_return: whether to save or show the figure. If "both", it will save and plot the figure at the same time. If
            "all", the figure will be saved, displayed and the associated axis and other object will be return.
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}. Defaults to {}.
        return_data: whether to return the calculated causality data. Defaults to False.

    Raises:
        ValueError: no preprocessing data in adata.uns
        ValueError: No layers named as `xkey`, `ykey`, or `zkey`.

    Returns:
        None would be returned by default. If `return_data` is set to be True, the causality data, a DataFrame with
        columns ("x": x coordination, "y": y coordination, "expected_z": expected z coordination, "pair": gene pairs) would
        be returned.
    """

    import matplotlib
    from matplotlib.colors import ListedColormap

    if "pp" not in adata.uns_keys():
        raise ValueError("You must first run dyn.pp.recipe_monocle and dyn.tl.moments before running this function.")

    if xkey is None:
        xkey = "M_t" if adata.uns["pp"]["has_labeling"] else "M_s"
    if ykey is None:
        ykey = "M_t" if adata.uns["pp"]["has_labeling"] else "M_s"
    if zkey is None:
        zkey = "velocity_T" if adata.uns["pp"]["has_labeling"] else "velocity_S"

    if not set([xkey, ykey, zkey]) <= set(adata.layers.keys()).union(set(["jacobian"])):
        raise ValueError(
            f"adata.layers doesn't have {xkey, ykey, zkey} layers. Please specify the correct layers or "
            "perform relevant preprocessing and vector field analyses first."
        )

    if cmap is None:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("comb_logic", ["#00CF8D", "#FFFF99", "#FF0000"])

    if return_data:
        flat_res = causality(
            adata,
            pairs_mat,
            xkey=xkey,
            ykey=ykey,
            zkey=zkey,
            log=log,
            drop_zero_cells=drop_zero_cells,
            delay=delay,
            k=k,
            normalize=normalize,
            grid_num=grid_num,
            n_row=n_row,
            n_col=n_col,
            cmap=cmap,
            show_rug=show_rug,
            show_extent=show_extent,
            ext_format=ext_format,
            figsize=figsize,
            return_data=return_data,
        )
        return flat_res
    else:
        causality(
            adata,
            pairs_mat,
            xkey=xkey,
            ykey=ykey,
            zkey=zkey,
            log=log,
            drop_zero_cells=drop_zero_cells,
            delay=delay,
            k=k,
            normalize=normalize,
            grid_num=grid_num,
            n_row=n_row,
            n_col=n_col,
            cmap=cmap,
            show_rug=show_rug,
            show_extent=show_extent,
            ext_format=ext_format,
            stacked_fraction=stacked_fraction,
            figsize=figsize,
            save_show_or_return=save_show_or_return,
            save_kwargs=save_kwargs,
            return_data=return_data,
            save_key="comb_logic",
        )


def hessian(
    adata: AnnData,
    pairs_mat: np.ndarray,
    xkey: Optional[str] = None,
    ykey: Optional[str] = None,
    zkey: Optional[str] = None,
    log: bool = True,
    drop_zero_cells: bool = False,
    delay: int = 0,
    grid_num: int = 25,
    n_row: int = 1,
    n_col: Optional[int] = None,
    cmap: Union[str, Colormap, None] = "bwr",
    normalize: bool = True,
    k: int = 30,
    show_rug: bool = True,
    show_extent: bool = False,
    ext_format: Optional[List[str]] = None,
    stacked_fraction: bool = False,
    figsize: Tuple[float, float] = (6, 4),
    save_show_or_return: Literal["save", "show", "both", "all"] = "show",
    save_kwargs: Dict[str, Any] = {},
    return_data: bool = False,
) -> Optional[pd.DataFrame]:
    """Plot the combinatorial influence of two genes `x`, `y` to the target `z`.

    This plotting function tries to intuitively visualize the influence from genes `x` and `y` to the target `z`.
    Firstly, we divide the expression space for `x` and `y` based on grid_num and then we estimate the k-nearest
    neighbor for each of the grid. We then use a Gaussian kernel to estimate the expected value for `z`. It is then
    displayed in two dimension with `x` and `y` as two axis and the color represents the value of the expected of `z`.
    This function accepts a matrix where each row is the gene pair and the target genes for this pair. The first column
    is the first hypothetical source or regulator, the second column represents the second hypothetical target while the
    third column represents the hypothetical target gene. The name in this matrix should match the name in the
    gene_short_name column of the cds_subset object.

    Args:
        adata: an AnnData object.
        pairs_mat: a matrix where each row is the gene pair and the first column is the hypothetical source or regulator
            while the second column represents the hypothetical target. The name in this matrix should match the name in
            the gene_short_name column of the adata object.
        xkey: the layer key of `x` data (regulator gene 1). If None, dynamo's default naming rule would be used.
            Defaults to None.
        ykey: the layer key of `y` data (regulator gene 2).  If None, dynamo's default naming rule would be used.
            Defaults to None.
        zkey: the layer key of `z` data (the hypothetical target).  If None, dynamo's default naming rule would be used.
            Defaults to None.
        log: whether to perform log transformation (using log(expression + 1)) before calculating density estimates.
            Defaults to True.
        drop_zero_cells: whether to drop cells that with zero expression for either the potential regulator or potential
            target. This can signify the relationship between potential regulators and targets, speed up the
            calculation, but at the risk of ignoring strong inhibition effects from certain regulators to targets.
            Defaults to False.
        delay: the time delay between the source and target gene. Always zero because we don't have real time-series.
            Defaults to 0.
        grid_num: the number of grid when creating the lagged DREVI plot. Defaults to 25.
        n_row: the number of rows used to layout the faceted cluster panels. Defaults to 1.
        n_col: the number of columns used to layout the faceted cluster panels. Defaults to None.
        cmap: the color map used to plot the heatmap. Could be the name of the color map or a matplotlib color map
            object. If None, the color map would be generated automatically. Defaults to "viridis".
        normalize: whether to row-scale the data. Defaults to True.
        k: number of k-nearest neighbors used in calculating 2-D kernel density. Defaults to 30.
        show_rug: whether to plot marginal distributions by drawing ticks along the x and y axes. Defaults to True.
        show_extent: whether to extend the figure. If False, `show_ridge` and `show_rug` would be set to False
            automatically. Defaults to False.
        ext_format: the string/list of strings (the first is for x and second for y labels) that will be used to format
            the ticks on x or y-axis. If it is None or one of the element in the list is None, the default setting will
            be used. Defaults to None.
        stacked_fraction: if True the jacobian will be represented as a stacked fraction in the title, otherwise a
            linear fraction style is used. Defaults to False.
        figsize: the size of the figure. Defaults to (6, 4).
        save_show_or_return: whether to save or show the figure. If "both", it will save and plot the figure at the same time. If
            "all", the figure will be saved, displayed and the associated axis and other object will be return.
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
         and the save_show_ret function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}. Defaults to {}.
        return_data: whether to return the calculated causality data. Defaults to False.

    Returns:
        None would be returned by default. If `return_data` is set to be True, the causality data, a DataFrame with
        columns ("x": x coordination, "y": y coordination, "expected_z": expected z coordination, "pair": gene pairs)
        would be returned.
    """

    import matplotlib
    from matplotlib.colors import ListedColormap

    if cmap is None:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("comb_logic", ["#00CF8D", "#FFFF99", "#FF0000"])

    if return_data:
        flat_res = causality(
            adata,
            pairs_mat,
            hessian_matrix=True,
            xkey=xkey,
            ykey=ykey,
            zkey=zkey,
            log=log,
            drop_zero_cells=drop_zero_cells,
            delay=delay,
            k=k,
            normalize=normalize,
            grid_num=grid_num,
            n_row=n_row,
            n_col=n_col,
            cmap=cmap,
            show_rug=show_rug,
            show_extent=show_extent,
            ext_format=ext_format,
            figsize=figsize,
            return_data=return_data,
        )
        return flat_res
    else:
        causality(
            adata,
            pairs_mat,
            hessian_matrix=True,
            xkey=xkey,
            ykey=ykey,
            zkey=zkey,
            log=log,
            drop_zero_cells=drop_zero_cells,
            delay=delay,
            k=k,
            normalize=normalize,
            grid_num=grid_num,
            n_row=n_row,
            n_col=n_col,
            cmap=cmap,
            show_rug=show_rug,
            show_extent=show_extent,
            ext_format=ext_format,
            stacked_fraction=stacked_fraction,
            figsize=figsize,
            save_show_or_return=save_show_or_return,
            save_kwargs=save_kwargs,
            return_data=return_data,
            save_key="hessian",
        )
