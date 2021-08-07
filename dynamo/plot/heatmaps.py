import numpy as np
import pandas as pd
import math
import scipy.spatial as ss
import seaborn

from ..tools.utils import flatten
from ..plot.utils import set_colorbar


def bandwidth_nrd(x):
    x = pd.Series(x)
    h = (x.quantile([0.75]).values - x.quantile([0.25]).values) / 1.34

    return 4 * 1.06 * min(math.sqrt(np.var(x, ddof=1)), h) * (len(x) ** (-1 / 5))


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
    return np.exp(-((x - u) ** 2) / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)


def kde2d(x, y, h=None, n=25, lims=None):
    """Reproduce kde2d function behavior from MASS package in R.
    Two-dimensional kernel density estimation with an axis-aligned
    bivariate normal kernel, evaluated on a square grid.
    Arguments
    ---------
        x:  `List`
            x coordinate of data
        y:  `List`
            y coordinate of data
        h:  `List` (Default: None)
            vector of bandwidths for :math:`x` and :math:`y` directions.  Defaults to normal reference bandwidth
            (see `bandwidth.nrd`). A scalar value will be taken to apply to both directions.
        n: `int` (Default: 25)
            Number of grid points in each direction.  Can be scalar or a length-2 integer list.
        lims: `List` (Default: None)
            The limits of the rectangle covered by the grid as :math:`x_l, x_u, y_l, y_u`.
    Returns
    -------
        A list of three components
        gx, gy: `List`
            The x and y coordinates of the grid points, lists of length `n`.
        z:  `List`
            An :math:`n[1]` by :math:`n[2]` matrix of the estimated density: rows correspond to the value of :math:`x`,
            columns to the value of :math:`y`.
    """
    nx = len(x)
    if not lims:
        lims = [min(x), max(x), min(y), max(y)]
    if len(y) != nx:
        raise Exception("data vectors must be the same length")
    elif (False in np.isfinite(x)) or (False in np.isfinite(y)):
        raise Exception("missing or infinite values in the data are not allowed")
    elif False in np.isfinite(lims):
        raise Exception("only finite values are allowed in 'lims'")
    else:
        n = rep(n, length=2) if isinstance(n, list) else rep([n], length=2)
        gx = np.linspace(lims[0], lims[1], n[0])
        gy = np.linspace(lims[2], lims[3], n[1])
        if h is None:
            h = [bandwidth_nrd(x), bandwidth_nrd(y)]
        else:
            h = np.array(rep(h, length=2))

        if h[0] <= 0 or h[1] <= 0:
            raise Exception("bandwidths must be strictly positive")
        else:
            h /= 4
            ax = pd.DataFrame()
            ay = pd.DataFrame()
            for i in range(len(x)):
                ax[i] = (gx - x[i]) / h[0]
            for i in range(len(y)):
                ay[i] = (gy - y[i]) / h[1]
            z = (np.matrix(dnorm(ax)) * np.matrix(dnorm(ay).T)) / (nx * h[0] * h[1])
    return gx, gy, z


def response(
    adata,
    pairs_mat,
    xkey=None,
    ykey=None,
    log=False,
    drop_zero_cells=True,
    delay=1,
    k=5,
    grid_num=25,
    n_row=None,
    n_col=1,
    scales="free",
    show_ridge=False,
    show_rug=True,
    return_data=False,
    verbose=False,
):
    """Plot the lagged DREVI plot pairs of genes across pseudotime.
    This plotting function builds on the original idea of DREVI plot but is extended in the context for causal network.
    It considers the time delay between the hypothetical regulators to the target genes which is parametered by :math:`d`.
    Lagged DREVI plot first estimates the joint density (:math:`P(x_{t - d}, y_t)`) for variables :math:`x_{t - d} and y_t`, then it
    divides the joint density by the marginal density :math:`P(x_{t - d})` to get the conditional density estimate
    (:math:`P(x_{t - d}, y_t | x_{x - d})`). We then calculate the z-score normalizing each column of conditional density. Note
    that this plot tries to demonstrate the potential influence between two variables instead of the factual influence.
    A red line corresponding to the point with maximal density on each :math:`x` value is plot which indicates the maximal possible
    point for :math:`y_t` give the value of :math:`x_{t - d}`. The 2-d density is estimated through the kde2d function.
    Arguments
    ---------
        adata: `Anndata`
            Annotated Data Frame, an Anndata object.
        pairs_mat: 'np.ndarray'
            A matrix where each row is the gene pair and the first column is the hypothetical source or regulator while
            the second column represents the hypothetical target. The name in this matrix should match the name in the
            gene_short_name column of the adata object.
        log: `bool` (Default: False)
            A logic argument used to determine whether or not you should perform log transformation (using :math:`log(expression + 1)`)
            before calculating density estimates, default to be TRUE.
        drop_zero_cells: `bool` (Default: True)
            Whether to drop cells that with zero expression for either the potential regulator or potential target. This
            can signify the relationship between potential regulators and targets, speed up the calculation, but at the risk
            of ignoring strong inhibition effects from certain regulators to targets.
        delay: `int` (Default: 1)
            The time delay between the source and target gene.
        k: `int` (Default: 5)
            Number of k-nearest neighbors used in calculating 2-D kernel density
        grid_num: `int` (Default: 25)
            The number of grid when creating the lagged DREVI plot.
        n_row: `int` (Default: None)
            number of columns used to layout the faceted cluster panels.
        n_col: `int` (Default: 1)
            number of columns used to layout the faceted cluster panels.
        scales: `str` (Default: 'free')
            The character string passed to facet function, determines whether or not the scale is fixed or free in
            different dimensions. (not used)
        verbose:
            A logic argument to determine whether or not we should print the detailed running information.
    Returns
    -------
        In addition to figure created by matplotlib, it also returns:
        flat_res: 'pd.core.frame.DataFrame'
            a pandas data frame used to create the heatmap with four columns (`x`: x-coordinate; `y`: y-coordinate; `den`:
            estimated density at x/y coordinate; `type`: the corresponding gene pair).
        flat_res_subset: 'pd.core.frame.DataFrame'
            a pandas data frame used to create the heatmap for the last gene pair (if multiple gene-pairs are inputted) with
            four columns (`x`: x-coordinate; `y`: y-coordinate; `den`: estimated density at x/y coordinate; `type`: the
            corresponding gene pair).
        ridge_curve_subset: 'pd.core.frame.DataFrame'
            a pandas data frame used to create the read ridge line for the last gene pair (if multiple gene-pairs are inputted) with
            four columns (`x`: x-coordinate; `y`: y-coordinate; `type`: the corresponding gene pair).
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import MaxNLocator

    all_genes_in_pair = np.unique(pairs_mat)
    if xkey is None:
        xkey = "M_t" if adata.uns["pp"]["has_labeling"] else "M_s"
    if ykey is None:
        ykey = "M_n" if adata.uns["pp"]["has_labeling"] else "M_u"
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "response", ["#000000", "#000000", "#000000", "#800080", "#FF0000", "#FFFF00"]
    )
    inset_dict = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "50%",  # height : 50%
        "loc": "lower left",
        "bbox_to_anchor": (1.05, 0.0, 1, 1),
        "borderpad": 0,
    }

    if not (set(all_genes_in_pair) <= set(adata.var_names)):
        raise Exception(
            "adata doesn't include all genes in gene_pairs_mat. Make sure all genes are included in adata.var_names."
        )

    flat_res = pd.DataFrame(columns=["x", "y", "den", "type"])
    ridge_curve = pd.DataFrame(columns=["x", "y", "type"])
    xy = pd.DataFrame()

    id = 0
    for gene_pairs_ind, gene_pairs in enumerate(pairs_mat):
        f_ini_ind = (grid_num ** 2) * id - 1
        r_ini_ind = grid_num * id - 1

        gene_pair_name = gene_pairs[0] + "->" + gene_pairs[1]

        x = flatten(adata[:, gene_pairs[0]].layers[xkey])
        y_ori = flatten(adata[:, gene_pairs[1]].layers[ykey])
        if drop_zero_cells:
            finite = np.isfinite(x + y_ori)
            nonzero = np.abs(x) + np.abs(y_ori) > 0

            valid_ids = np.logical_and(finite, nonzero)
        else:
            valid_ids = np.isfinite(x + y_ori)

        x, y_ori = x[valid_ids], y_ori[valid_ids]

        if log:
            x, y_ori = np.log(np.array(x) + 1), np.log(np.array(y_ori) + 1)

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

        x_meshgrid, y_meshgrid, den_res = kde2d(
            x, y, n=[grid_num, grid_num], lims=[min(x), max(x), min(y), max(y)], h=bandwidth
        )
        den_res = np.array(den_res)

        den_x = np.sum(den_res, axis=0)
        max_ind = 1
        den_res = den_res.tolist()

        for i in range(len(x_meshgrid)):
            tmp = den_res[i] / den_x[i]
            max_val = max(tmp)
            min_val = 0

            if np.sum(den_x[i] != 0):
                rescaled_val = (den_res[i] / den_x[i] - min_val) / (max_val - min_val)
                max_ind = np.argmax(rescaled_val)

            res_Row = pd.DataFrame(
                [[x_meshgrid[i], y_meshgrid[max_ind], gene_pair_name]],
                columns=["x", "y", "type"],
                index=[r_ini_ind + i],
            )
            ridge_curve = pd.concat([ridge_curve, res_Row])

            for j in range(len(y_meshgrid)):
                rescaled_val = (den_res[i][j] / den_x[i] - min_val) / (max_val - min_val)
                res_Row = pd.DataFrame(
                    [[x_meshgrid[i], y_meshgrid[j], rescaled_val, gene_pair_name]],
                    columns=["x", "y", "den", "type"],
                    index=[i * len(x_meshgrid) + j + f_ini_ind],
                )
                flat_res = pd.concat([flat_res, res_Row])
        cur_data = pd.DataFrame({"x": x, "y": y, "type": gene_pair_name})
        xy = pd.concat([xy, cur_data], axis=0)

        id = id + 1

    gene_pairs_num = flat_res.type.value_counts().shape[0]

    n_row = gene_pairs_num if n_row is None else n_row

    if n_row * n_col < gene_pairs_num:
        raise Exception("The number of row or column specified is less than the gene pairs")

    fig, axes = plt.subplots(n_row, n_col, figsize=(8, 8), sharex=False, sharey=False, squeeze=False)

    plt.xlabel(r"$x_{t-1}$")
    plt.ylabel(r"$y_{t}$")

    for x, flat_res_type in enumerate(flat_res.type.value_counts().index.values):
        flat_res_subset = flat_res[flat_res["type"] == flat_res_type]
        ridge_curve_subset = ridge_curve[ridge_curve["type"] == flat_res_type]
        xy_subset = xy[xy["type"] == flat_res_type]

        x_val, y_val = flat_res_subset["x"], flat_res_subset["y"]

        i, j = x % n_row, x // n_row  # %: remainder; //: integer division

        values = flat_res_subset["den"].values.reshape(grid_num, grid_num).T
        im = axes[i, j].imshow(
            values,
            interpolation="mitchell",
            origin="lower",
            extent=(min(x_val), max(x_val), min(y_val), max(y_val)),
            cmap=cmap,
        )
        axes[i, j].title.set_text(flat_res_type)

        norm = matplotlib.colors.Normalize(vmin=min(flatten(values)), vmax=min(flatten(values)))

        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(values)
        cb = plt.colorbar(mappable, cax=set_colorbar(axes[i, j], inset_dict), ax=axes[i, j])
        cb.set_alpha(1)
        cb.draw_all()
        cb.locator = MaxNLocator(nbins=3, integer=True)
        cb.update_ticks()

        if show_ridge:
            #         ridge_curve_subset = pd.DataFrame(flat_res_subset).loc[pd.DataFrame(flat_res_subset).groupby('x')['den'].idxmax()]
            axes[i, j].plot(ridge_curve_subset["x"].values, ridge_curve_subset["y"].values, color="red")
            #         axes[i, j].plot(flat_res_subset['x'], [0.01]*len(flat_res_subset['x']), '|', color='white')
            #         axes[i, j].plot([0.01]*len(flat_res_subset['y']), flat_res_subset['y'], '|', color='white')

        if show_rug:
            seaborn.rugplot(xy_subset["x"].values, height=0.05, axis="x", ax=axes[i, j], c="darkred", alpha=0.25)
            seaborn.rugplot(xy_subset["y"].values, height=0.025, axis="y", ax=axes[i, j], c="darkred", alpha=0.25)

    # fig.colorbar(im, ax=axes)
    plt.tight_layout()
    plt.show()

    if return_data:
        return (flat_res, flat_res_subset, ridge_curve_subset)
    else:
        adata.uns["response"] = {
            "flat_res": flat_res,
            "flat_res_subset": flat_res_subset,
            "ridge_curve_subset": ridge_curve_subset,
        }


def causality(
    adata,
    pairs_mat,
    xkey=None,
    ykey=None,
    zkey=None,
    log=False,
    drop_zero_cells=False,
    delay=1,
    k=5,
    normalize=True,
    grid_num=25,
    n_row=None,
    n_col=1,
    cmap=None,
    scales="free",
    show_rug=True,
    return_data=False,
    verbose=False,
):
    """Plot the heatmap for the expected value :math:`y(t)` given :math:`x(t - d)` and :math:`y(t - 1)`.
    This plotting function tries to intuitively visualize the informatioin transfer from :math:`x(t - d)` to :math:`y(t)`
    given :math:`y(t)`'s previous state :math:`y(t - 1)`. Firstly, we divide the expression space for :math:`x(t - d)` to
    :math:`y(t - 1)` based on grid_num and then we estimate the k-nearest neighbor for each of the grid. We then use a
    Gaussian kernel to estimate the expected value for :math:`y(t)`. It is then displayed in two dimension with :math:`x(t - d)`
    and :math:`y(t - 1)` as two axis and the color represents the expected value of :math:`y(t)` give :math:`x(t - d)` and
    :math:`y(t - 1)`. This function accepts a matrix where each row is the gene pair and the first column is the hypothetical
    source or regulator while the second column represents the hypothetical target. The name in this matrix should match
    the name in the gene_short_name column of the cds_subset object.
    Arguments
    ---------
        adata: `Anndata`
            Annotated Data Frame, an Anndata object.
        pairs_mat: 'np.ndarray'
            A matrix where each row is the gene pair and the first column is the hypothetical source or regulator while
            the second column represents the hypothetical target. The name in this matrix should match the name in the
            gene_short_name column of the adata object.
        log: `bool` (Default: False)
            A logic argument used to determine whether or not you should perform log transformation (using log(expression + 1))
            before calculating density estimates, default to be TRUE.
        drop_zero_cells: `bool` (Default: True)
            Whether to drop cells that with zero expression for either the potential regulator or potential target. This
            can signify the relationship between potential regulators and targets, speed up the calculation, but at the risk
            of ignoring strong inhibition effects from certain regulators to targets.
        delay: `int` (Default: 1)
            The time delay between the source and target gene.
        k: `int` (Default: 5)
            Number of k-nearest neighbors used in calculating 2-D kernel density
        grid_num: `int` (Default: 25)
            The number of grid when creating the lagged DREVI plot.
        n_row: `int` (Default: None)
            number of columns used to layout the faceted cluster panels.
        n_col: `int` (Default: 1)
            number of columns used to layout the faceted cluster panels.
        scales: `str` (Default: 'free')
            The character string passed to facet function, determines whether or not the scale is fixed or free in
            different dimensions. (not used)
        verbose:
            A logic argument to determine whether or not we should print the detailed running information.
    Returns
    -------
        A figure created by matplotlib.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import MaxNLocator

    all_genes_in_pair = np.unique(pairs_mat)
    if xkey is None:
        xkey = "M_t" if adata.uns["pp"]["has_labeling"] else "M_s"
    if ykey is None:
        ykey = "M_t" if adata.uns["pp"]["has_labeling"] else "M_s"
    if zkey is None:
        zkey = "M_n" if adata.uns["pp"]["has_labeling"] else "M_u"
    if cmap is None:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "causality", ["#008000", "#ADFF2F", "#FFFF00", "#FFA500", "#FFC0CB", "#FFFFFE"]
        )
    inset_dict = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "50%",  # height : 50%
        "loc": "lower left",
        "bbox_to_anchor": (1.05, 0.0, 1, 1),
        "borderpad": 0,
    }
    if not (set(all_genes_in_pair) <= set(adata.var_names)):
        raise Exception(
            "adata doesn't include all genes in gene_pairs_mat. Make sure all genes are included in adata.var_names."
        )

    if drop_zero_cells:
        # sub_data = sub_data.loc[:, (sub_data > 0).sum(0) == 2]
        pass

    flat_res = pd.DataFrame(columns=["x", "z", "expected_y", "pair"])  ###empty df
    xy = pd.DataFrame()

    id = 0
    for gene_pairs_ind in range(0, len(pairs_mat)):
        # if verbose:
        #     info("current gene pair is ", pairs_mat[gene_pairs_ind, 0], " -> ", pairs_mat[gene_pairs_ind, 1])
        gene_pairs = pairs_mat[gene_pairs_ind, :]
        f_ini_ind = (grid_num ** 2) * id

        gene_pair_name = gene_pairs[0] + "->" + gene_pairs[1]

        x = flatten(adata[:, gene_pairs[0]].layers[xkey])
        y_ori = flatten(adata[:, gene_pairs[1]].layers[ykey])
        z_ori = flatten(adata[:, gene_pairs[2]].layers[zkey])
        if drop_zero_cells:
            finite = np.isfinite(x + y_ori + z_ori)
            nonzero = np.abs(x) + np.abs(y_ori) + np.abs(z_ori) > 0

            valid_ids = np.logical_and(finite, nonzero)
        else:
            valid_ids = np.isfinite(x + y_ori + z_ori)

        x, y_ori, z_ori = x[valid_ids], y_ori[valid_ids], z_ori[valid_ids]

        if log:
            x, y_ori = np.log(np.array(x) + 1), np.log(np.array(y_ori) + 1)

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
        z_meshgrid = np.linspace(min(z), max(z), grid_num, endpoint=True)

        xv, zv = np.meshgrid(x_meshgrid, z_meshgrid)
        xp = xv.reshape((1, -1)).tolist()
        zp = zv.reshape((1, -1)).tolist()
        xz_query = np.array(xp + zp).T
        tree_xz = ss.cKDTree(cur_data[["x", "y"]])
        dist_mat, idx_mat = tree_xz.query(xz_query, k=k + 1)

        for i in range(dist_mat.shape[0]):
            subset_dat = cur_data.iloc[idx_mat[i, 1:], 1]
            u = np.exp(-dist_mat[i, 1:] / np.min(dist_mat[i, 1:]))
            w = u / np.sum(u)

            tmp = sum(np.array(w) * np.array(subset_dat))
            res_Row = pd.DataFrame(
                [[xz_query[i, 0], xz_query[i, 1], tmp, gene_pair_name]],
                columns=["x", "z", "expected_y", "pair"],
                index=[f_ini_ind + i],
            )
            flat_res = pd.concat([flat_res, res_Row])
        if normalize:
            vals = flat_res["expected_y"][(f_ini_ind) : (f_ini_ind + len(dist_mat))]
            max_val = max(vals.dropna().values.reshape(1, -1)[0])
            if not np.isfinite(max_val):
                max_val = 1e10

            flat_res.iloc[(f_ini_ind) : (f_ini_ind + len(dist_mat)), :]["expected_y"] = (
                flat_res.iloc[(f_ini_ind) : (f_ini_ind + len(dist_mat)), :]["expected_y"] / max_val
            )

        id = id + 1

    gene_pairs_num = flat_res.pair.value_counts().shape[0]

    n_row = gene_pairs_num if n_row is None else n_row

    if n_row * n_col < gene_pairs_num:
        raise Exception("The number of row or column specified is less than the gene pairs")

    fig, axes = plt.subplots(n_row, n_col, figsize=(8, 8), sharex=False, sharey=False, squeeze=False)

    plt.xlabel(r"$x_{t-1}$")
    plt.ylabel(r"$y_{t}$")

    for x, flat_res_type in enumerate(flat_res.pair.value_counts().index.values):
        flat_res_subset = flat_res[flat_res["pair"] == flat_res_type]
        xy_subset = xy[xy["pair"] == flat_res_type]

        x_val, z_val = flat_res_subset["x"], flat_res_subset["z"]

        i, j = x % n_row, x // n_row  # %: remainder; //: integer division

        values = flat_res_subset["expected_y"].values.reshape(xv.shape)
        im = axes[i, j].imshow(
            values,
            interpolation="mitchell",
            origin="lower",
            extent=(min(x_val), max(x_val), min(z_val), max(z_val)),
            cmap=cmap,
        )
        norm = matplotlib.colors.Normalize(vmin=min(flatten(values)), vmax=max(flatten(values)))

        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(values)
        cb = plt.colorbar(mappable, cax=set_colorbar(axes[i, j], inset_dict), ax=axes[i, j])
        cb.set_alpha(1)
        cb.draw_all()
        cb.locator = MaxNLocator(nbins=3, integer=True)
        cb.update_ticks()

        #         axes[i, j].plot(flat_res_subset['x'], [0.01]*len(flat_res_subset['x']), '|', color='k')
        #         axes[i, j].plot([0.01]*len(flat_res_subset['z']), flat_res_subset['z'], '|', color='k')
        if show_rug:
            seaborn.rugplot(xy_subset["x"].values, height=0.05, axis="x", ax=axes[i, j], c="darkred", alpha=0.25)
            seaborn.rugplot(xy_subset["z"].values, height=0.025, axis="z", ax=axes[i, j], c="darkred", alpha=0.25)
        axes[i, j].title.set_text(flat_res_type)

    fig.colorbar(im, ax=axes)
    plt.show()

    if return_data:
        return flat_res
    else:
        adata.uns["causality"] = flat_res


def comb_logic(
    adata,
    pairs_mat,
    xkey=None,
    ykey=None,
    zkey=None,
    log=False,
    drop_zero_cells=False,
    delay=1,
    grid_num=25,
    n_row=None,
    n_col=1,
    cmap=None,
    normalize=True,
    scales="free",
    k=5,
    show_rug=True,
    return_data=False,
    verbose=False,
):
    """Plot the combinatorial influence of two genes :math:`x`, :math:`y` to the target :math:`z`.
    This plotting function tries to intuitively visualize the influence from genes :math:`x` and :math:`y` to the target :math:`z`.
    Firstly, we divide the expression space for :math:`x` and :math:`y` based on grid_num and then we estimate the k-nearest neighbor for each of the
    grid. We then use a Gaussian kernel to estimate the expected value for :math:`z`. It is then displayed in two dimension with :math:`x` and :math:`y`
    as two axis and the color represents the value of the expected of :math:`z`. This function accepts a matrix where each row is the gene pair
    and the target genes for this pair. The first column is the first hypothetical source or regulator, the second column represents
    the second hypothetical target while the third column represents the hypothetical target gene. The name in this matrix should match
    the name in the gene_short_name column of the cds_subset object.
    Arguments
    ---------
        adata: `Anndata`
            Annotated Data Frame, an Anndata object.
        pairs_mat: 'np.ndarray'
            A matrix where each row is the gene pair and the first and second columns are the hypothetical source or regulator while
            the third column represents the hypothetical target. The name in this matrix should match the name in the
            gene_short_name column of the adata object.
        log: `bool` (Default: False)
            A logic argument used to determine whether or not you should perform log transformation (using log(expression + 1))
            before calculating density estimates, default to be TRUE.
        drop_zero_cells: `bool` (Default: True)
            Whether to drop cells that with zero expression for either the potential regulator or potential target. This
            can signify the relationship between potential regulators and targets, speed up the calculation, but at the risk
            of ignoring strong inhibition effects from certain regulators to targets.
        delay: `int` (Default: 1)
            The time delay between the source and target gene.
        grid_num: `int` (Default: 25)
            The number of grid when creating the lagged DREVI plot.
        n_row: `int` (Default: None)
            number of columns used to layout the faceted cluster panels.
        normalize: `bool` (Default: True)
            Whether to row-scale the data
        n_col: `int` (Default: 1)
            number of columns used to layout the faceted cluster panels.
        scales: `str` (Default: 'free')
            The character string passed to facet function, determines whether or not the scale is fixed or free in
            different dimensions. (not used)
        verbose:
            A logic argument to determine whether or not we should print the detailed running information.
    Returns
    -------
        A figure created by matplotlib.
    """
    import matplotlib
    from matplotlib.colors import ListedColormap

    if xkey is None:
        xkey = "M_t" if adata.uns["pp"]["has_labeling"] else "M_s"
    if ykey is None:
        ykey = "M_t" if adata.uns["pp"]["has_labeling"] else "M_s"
    if zkey is None:
        zkey = "velocity_T" if adata.uns["pp"]["has_labeling"] else "velocity_S"

    if cmap is None:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("comb_logic", ["#00CF8D", "#FFFF99", "#FF0000"])
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
        scales=scales,
        show_rug=show_rug,
        return_data=return_data,
        verbose=verbose,
    )

    if return_data:
        return flat_res
    else:
        adata.uns["comb_logic"] = adata.uns["causality"]
