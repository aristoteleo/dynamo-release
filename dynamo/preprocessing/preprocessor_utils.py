import warnings
from typing import Callable, List, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata
import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse.base import issparse
from sklearn.svm import SVR
from sklearn.utils import sparsefuncs

from ..configuration import DKM, DynamoAdataKeyManager
from ..dynamo_logger import (
    main_debug,
    main_finish_progress,
    main_info,
    main_info_insert_adata,
    main_info_insert_adata_obs,
    main_info_insert_adata_obsm,
    main_info_insert_adata_uns,
    main_info_insert_adata_var,
    main_log_time,
    main_warning,
)
from ..tools.utils import update_dict
from ..utils import copy_adata
from .preprocess_monocle_utils import top_table
from .utils import (
    Freeman_Tukey,
    add_noise_to_duplicates,
    basic_stats,
    calc_new_to_total_ratio,
    clusters_stats,
    collapse_species_adata,
    compute_gene_exp_fraction,
    convert2symbol,
    convert_layers2csr,
    cook_dist,
    detect_experiment_datatype,
    get_inrange_shared_counts_mask,
    get_svr_filter,
    get_sz_exprs,
    merge_adata_attrs,
    normalize_mat_monocle,
    pca_monocle,
    sz_util,
    unique_var_obs_adata,
)


def is_log1p_transformed_adata(adata: anndata.AnnData) -> bool:
    """check if adata data is log transformed by checking a small subset of adata observations.

    Args:
        adata: an AnnData object

    Returns:
        A flag shows whether the adata object is log transformed.
    """

    chosen_gene_indices = np.random.choice(adata.n_obs, 10)
    _has_log1p_transformed = not np.allclose(
        np.array(adata.X[:, chosen_gene_indices].sum(1)),
        np.array(adata.layers["spliced"][:, chosen_gene_indices].sum(1)),
        atol=1e-4,
    )
    return _has_log1p_transformed


def _infer_labeling_experiment_type(adata: anndata.AnnData, tkey: str) -> Literal["one-shot", "kin", "deg"]:
    """Returns the experiment type of `adata` according to `tkey`s

    Args:
        adata: an AnnData Object.
        tkey: the key for time in `adata.obs`.

    Returns:
        The experiment type, must be one of "one-shot", "kin" or "deg".
    """

    experiment_type = None
    tkey_val = np.array(adata.obs[tkey], dtype="float")
    if len(np.unique(tkey_val)) == 1:
        experiment_type = "one-shot"
    else:
        labeled_frac = adata.layers["new"].T.sum(0) / adata.layers["total"].T.sum(0)
        xx = labeled_frac.A1 if issparse(adata.layers["new"]) else labeled_frac

        yy = tkey_val
        xm, ym = np.mean(xx), np.mean(yy)
        cov = np.mean(xx * yy) - xm * ym
        var_x = np.mean(xx * xx) - xm * xm

        k = cov / var_x

        # total labeled RNA amount will increase (decrease) in kinetic (degradation) experiments over time.
        experiment_type = "kin" if k > 0 else "deg"
    main_info(
        f"\nDynamo detects your labeling data is from a {experiment_type} experiment. If experiment type is not corrent, please correct "
        f"\nthis via supplying the correct experiment_type (one of `one-shot`, `kin`, `deg`) as "
        f"needed."
    )
    return experiment_type


def get_nan_or_inf_data_bool_mask(arr: np.ndarray) -> np.ndarray:
    """Returns the mask of arr with the same shape, indicating whether each index is nan/inf or not.

    Args:
        arr: an array

    Returns:
        A bool array indicating each element is nan/inf or not
    """

    mask = np.isnan(arr) | np.isinf(arr) | np.isneginf(arr)
    return mask


def clip_by_perc(layer_mat):
    """Returns a new matrix by clipping the layer_mat according to percentage."""
    # TODO implement this function (currently not used)
    return


def calc_mean_var_dispersion_general_mat(
    data_mat: Union[np.ndarray, csr_matrix], axis: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """calculate mean, variance, and dispersion of a matrix.

    Args:
        data_mat: the matrix to be evaluated, either a ndarray or a scipy sparse matrix.
        axis: the axis along which calculation is performed. Defaults to 0.

    Returns:
        A tuple (mean, var, dispersion) where mean is the mean of the array along the given axis, var is the variance of
        the array along the given axis, and dispersion is the dispersion of the array along the given axis.
    """

    if not issparse(data_mat):
        return calc_mean_var_dispersion_ndarray(data_mat, axis)
    else:
        return calc_mean_var_dispersion_sparse(data_mat, axis)


def calc_mean_var_dispersion_ndarray(data_mat: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """calculate mean, variance, and dispersion of a non-sparse matrix.

    Args:
        data_mat: the matrix to be evaluated.
        axis: the axis along which calculation is performed. Defaults to 0.

    Returns:
        A tuple (mean, var, dispersion) where mean is the mean of the array along the given axis, var is the variance of
        the array along the given axis, and dispersion is the dispersion of the array along the given axis.
    """

    # per gene mean, var and dispersion
    mean = np.nanmean(data_mat, axis=axis).flatten()

    # <class 'anndata._core.views.ArrayView'> has bug after using operator "==" (e.g. mean == 0), which changes mean.
    mean = np.array(mean)
    mean[mean == 0] += 1e-7  # prevent division by zero
    var = np.nanvar(data_mat, axis=axis)
    dispersion = var / mean
    return mean.flatten(), var.flatten(), dispersion.flatten()


def calc_mean_var_dispersion_sparse(sparse_mat: csr_matrix, axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """calculate mean, variance, and dispersion of a matrix.

    Args:
        sparse_mat: the sparse matrix to be evaluated.
        axis: the axis along which calculation is performed. Defaults to 0.

    Returns:
        A tuple (mean, var, dispersion), where mean is the mean of the array along the given axis, var is the variance
        of the array along the given axis, and dispersion is the dispersion of the array along the given axis.
    """

    sparse_mat = sparse_mat.copy()
    nan_mask = get_nan_or_inf_data_bool_mask(sparse_mat.data)
    temp_val = (sparse_mat != 0).sum(axis)
    sparse_mat.data[nan_mask] = 0
    nan_count = temp_val - (sparse_mat != 0).sum(axis)

    non_nan_count = sparse_mat.shape[axis] - nan_count
    mean = (sparse_mat.sum(axis) / sparse_mat.shape[axis]).A1
    mean[mean == 0] += 1e-7  # prevent division by zero

    # same as numpy var behavior: denominator is N, var=(data_arr-mean)/N
    var = np.power(sparse_mat - mean, 2).sum(axis) / sparse_mat.shape[axis]
    dispersion = var / mean
    return np.array(mean).flatten(), np.array(var).flatten(), np.array(dispersion).flatten()


def seurat_get_mean_var(
    X: Union[csr_matrix, np.ndarray],
    ignore_zeros: bool = False,
    perc: Union[float, List[float], None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Only used in seurat impl to match seurat and scvelo implementation result.

    Args:
        X: a matrix as np.ndarray or a sparse matrix as scipy sparse matrix. Rows are cells while columns are genes.
        ignore_zeros: whether ignore columns with 0 only. Defaults to False.
        perc: clip the gene expression values based on the perc or the min/max boundary of the values. Defaults to None.

    Returns:
        A tuple (mean, var) where mean is the mean of the columns after processing of the matrix and var is the variance
        of the columns after processing of the matrix.
    """

    data = X.data if issparse(X) else X
    mask_nans = np.isnan(data) | np.isinf(data) | np.isneginf(data)

    n_nonzeros = (X != 0).sum(0)
    n_counts = n_nonzeros if ignore_zeros else X.shape[0]

    if mask_nans.sum() > 0:
        if issparse(X):
            data[np.isnan(data) | np.isinf(data) | np.isneginf(data)] = 0
            n_nans = n_nonzeros - (X != 0).sum(0)
        else:
            X[mask_nans] = 0
            n_nans = mask_nans.sum(0)
        n_counts -= n_nans

    if perc is not None:
        if np.size(perc) < 2:
            perc = [perc, 100] if perc < 50 else [0, perc]
        lb, ub = np.percentile(data, perc)
        data = np.clip(data, lb, ub)

    if issparse(X):
        mean = (X.sum(0) / n_counts).A1
        mean_sq = (X.multiply(X).sum(0) / n_counts).A1
    else:
        mean = X.sum(0) / n_counts
        mean_sq = np.multiply(X, X).sum(0) / n_counts
    n_cells = np.clip(X.shape[0], 2, None)  # to avoid division by zero
    var = (mean_sq - mean**2) * (n_cells / (n_cells - 1))

    mean = np.nan_to_num(mean)
    var = np.nan_to_num(var)
    return mean, var


def select_genes_by_dispersion_general(
    adata: AnnData,
    layer: str = DKM.X_LAYER,
    nan_replace_val: Union[float, None] = None,
    n_top_genes: int = 2000,
    recipe: Literal["monocle", "svr", "seurat"] = "monocle",
    seurat_min_disp: Union[float, None] = None,
    seurat_max_disp: Union[float, None] = None,
    seurat_min_mean: Union[float, None] = None,
    seurat_max_mean: Union[float, None] = None,
    monocle_kwargs: dict = {},
    gene_names: Union[List[str], None] = None,
    var_filter_key: str = "pass_basic_filter",
    inplace: bool = False,
) -> None:
    """A general function for feature genes selection.

    Preprocess adata and dispatch to different filtering methods, and eventually set keys in anndata to denote which
    genes are wanted in downstream analysis.

    Args:
        adata: an AnnData object.
        layer: the key of a sparse matrix in adata. Defaults to DKM.X_LAYER.
        nan_replace_val: your choice of value to replace values in layer. Defaults to None.
        n_top_genes: number of genes to select as highly variable genes. Defaults to 2000.
        recipe: a recipe for selecting genes; must be one of "monocle", "svr", or "seurat". Defaults to "monocle".
        seurat_min_disp: seurat dispersion min cutoff. Defaults to None.
        seurat_max_disp: seurat dispersion max cutoff. Defaults to None.
        seurat_min_mean: seurat mean min cutoff. Defaults to None.
        seurat_max_mean: seurat mean max cutoff. Defaults to None.
        monocle_kwargs: kwargs for `select_genes_monocle`. Defaults to {}.
        gene_names: name of genes to be selected. Defaults to None.
        var_filter_key: filter gene names based on the key defined in adata.var before gene selection. Defaults to
            "pass_basic_filter".
        inplace: when inplace is True, subset adata according to selected genes. Defaults to False.

    Raises:
        NotImplementedError: the recipe is invalid/unsupported.
    """

    main_info("filtering genes by dispersion...")
    main_log_time()

    pass_filter_genes = adata.var_names
    if gene_names:
        main_info("select genes on gene names from arguments <gene_names>")
        pass_filter_genes = gene_names
    elif var_filter_key:
        main_info("select genes on var key: %s" % (var_filter_key))
        pass_filter_genes = adata.var_names[adata.var[var_filter_key]]

    if len(pass_filter_genes) != len(set(pass_filter_genes)):
        main_warning("gene names are not unique, please check your preprocessing procedure.")
    subset_adata = adata[:, pass_filter_genes]
    if n_top_genes is None:
        main_info("n_top_genes is None, reserve all genes and add filter gene information")
        n_top_genes = adata.n_vars
    layer_mat = DKM.select_layer_data(subset_adata, layer)
    if nan_replace_val:
        main_info("replacing nan values with: %s" % (nan_replace_val))
        _mask = get_nan_or_inf_data_bool_mask(layer_mat)
        layer_mat[_mask] = nan_replace_val

    main_info("select genes by recipe: " + recipe)
    if recipe == "svr":
        mean, variance, highly_variable_mask, highly_variable_scores = select_genes_by_dispersion_svr(
            subset_adata, layer_mat, n_top_genes
        )
    elif recipe == "seurat":
        mean, variance, highly_variable_mask, highly_variable_scores = select_genes_by_seurat_recipe(
            subset_adata,
            layer_mat,
            min_disp=seurat_min_disp,
            max_disp=seurat_max_disp,
            min_mean=seurat_min_mean,
            max_mean=seurat_max_mean,
            n_top_genes=n_top_genes,
        )
    elif recipe == "monocle":
        # TODO refactor dynamo monocle selection genes part code and make it modular (same as the two functions above)
        # the logics here for dynamo recipe is different from the above recipes
        # Note we do not need to pass subset_adata here because monocle takes care of everything regarding dynamo
        # convention
        select_genes_monocle(adata, **monocle_kwargs)
        adata.var[DKM.VAR_GENE_HIGHLY_VARIABLE_KEY] = adata.var[DKM.VAR_USE_FOR_PCA]
        return
    else:
        raise NotImplementedError("Selected gene seletion recipe not supported.")

    main_info_insert_adata_var(DKM.VAR_GENE_MEAN_KEY)
    main_info_insert_adata_var(DKM.VAR_GENE_VAR_KEY)
    main_info_insert_adata_var(DKM.VAR_GENE_HIGHLY_VARIABLE_KEY)
    main_debug("type of variance:" + str(type(variance)))
    main_debug("shape of variance:" + str(variance.shape))
    adata.var[DKM.VAR_GENE_MEAN_KEY] = np.nan
    adata.var[DKM.VAR_GENE_VAR_KEY] = np.nan
    adata.var[DKM.VAR_GENE_HIGHLY_VARIABLE_KEY] = False
    adata.var[DKM.VAR_USE_FOR_PCA] = False

    adata.var[DKM.VAR_GENE_MEAN_KEY][pass_filter_genes] = mean.flatten()
    adata.var[DKM.VAR_GENE_VAR_KEY][pass_filter_genes] = variance
    adata.var[DKM.VAR_GENE_HIGHLY_VARIABLE_KEY][pass_filter_genes] = highly_variable_mask
    adata.var[DKM.VAR_USE_FOR_PCA][pass_filter_genes] = highly_variable_mask

    main_info("number of selected highly variable genes: " + str(adata.var[DKM.VAR_USE_FOR_PCA].sum()))
    if recipe == "svr":
        # SVR can give highly_variable_scores
        main_info_insert_adata_var(DKM.VAR_GENE_HIGHLY_VARIABLE_SCORES)
        adata.var[DKM.VAR_GENE_HIGHLY_VARIABLE_SCORES] = np.nan
        adata.var[DKM.VAR_GENE_HIGHLY_VARIABLE_SCORES][pass_filter_genes] = highly_variable_scores.flatten()

    if inplace:
        main_info("inplace is True, subset adata according to selected genes.")
        adata = adata[:, adata.var[DKM.VAR_USE_FOR_PCA]]
    main_finish_progress("filter genes by dispersion")


def select_genes_by_seurat_recipe(
    adata: AnnData,
    sparse_layer_mat: csr_matrix,
    n_bins: int = 20,
    log_mean_and_dispersion: bool = True,
    min_disp: float = None,
    max_disp: float = None,
    min_mean: float = None,
    max_mean: float = None,
    n_top_genes: Union[int, None] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, None]:
    """Apply seurat's gene selection recipe by cutoffs.

    Args:
        adata: an AnnData object
        sparse_layer_mat: the sparse matrix used for gene selection.
        n_bins: the number of bins for normalization. Defaults to 20.
        log_mean_and_dispersion: whether log the gene expression values before calculating the dispersion values.
            Defaults to True.
        min_disp: seurat dispersion min cutoff. Defaults to None.
        max_disp: seurat dispersion max cutoff. Defaults to None.
        min_mean: seurat mean min cutoff. Defaults to None.
        max_mean: seurat mean max cutoff. Defaults to None.
        n_top_genes: number of top genes to be evaluated. If set to be None, genes are filtered by mean and dispersion
            norm threshold. Defaults to None.

    Returns:
        A tuple (mean, variance, highly_variable_mask, highly_variable_scores), where mean is the mean of the provided
        sparse matrix, variance is the variance of the provided sparse matrix, highly_variable_mask is a bool array
        indicating whether an element (a gene) is highly variable in the matrix. highly_variable_scores is always none
        since the scores are not applicable to Seurat recipe.
    """

    # default values from Seurat
    if min_disp is None:
        min_disp = 0.5
    if max_disp is None:
        max_disp = np.inf
    if min_mean is None:
        min_mean = 0.0125
    if max_mean is None:
        max_mean = 3

    mean, variance, dispersion = calc_mean_var_dispersion_sparse(sparse_layer_mat)
    sc_mean, sc_var = seurat_get_mean_var(sparse_layer_mat)
    mean, variance = sc_mean, sc_var
    dispersion = variance / mean

    if log_mean_and_dispersion:
        mean = np.log1p(mean)
        dispersion[np.equal(dispersion, 0)] = np.nan
        dispersion = np.log(dispersion)

    temp_df = pd.DataFrame()
    temp_df["mean"], temp_df["dispersion"] = mean, dispersion

    temp_df["mean_bin"] = pd.cut(temp_df["mean"], bins=n_bins)
    disp_grouped = temp_df.groupby("mean_bin")["dispersion"]
    disp_mean_bin = disp_grouped.mean()
    disp_std_bin = disp_grouped.std(ddof=1)

    # handle nan std
    one_gene_per_bin = disp_std_bin.isnull()

    disp_std_bin[one_gene_per_bin] = disp_mean_bin[one_gene_per_bin].values
    disp_mean_bin[one_gene_per_bin] = 0

    # normalized dispersion
    mean = disp_mean_bin[temp_df["mean_bin"].values].values
    std = disp_std_bin[temp_df["mean_bin"].values].values
    variance = std**2
    temp_df["dispersion_norm"] = ((temp_df["dispersion"] - mean) / std).fillna(0)
    dispersion_norm = temp_df["dispersion_norm"].values

    highly_variable_mask = None
    if n_top_genes is not None:
        main_info("choose %d top genes" % (n_top_genes), indent_level=2)
        threshold = temp_df["dispersion_norm"].nlargest(n_top_genes).values[-1]
        highly_variable_mask = temp_df["dispersion_norm"].values >= threshold
    else:
        main_info("choose genes by mean and dispersion norm threshold", indent_level=2)
        highly_variable_mask = np.logical_and.reduce(
            (
                mean > min_mean,
                mean < max_mean,
                dispersion_norm > min_disp,
                dispersion_norm < max_disp,
            )
        )
    return mean, variance, highly_variable_mask, None


def select_genes_by_dispersion_svr(
    adata: AnnData, layer_mat: Union[np.array, csr_matrix], n_top_genes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filters adata's genes according to layer_mat, and set adata's preprocess keys for downstream analysis

    Args:
        adata: an AnnData object.
        layer_mat: the matrix used for select genes with shape of #cells X #genes.
        n_top_genes: the number of genes to use.

    Returns:
        A tuple (mean, variance, highly_variable_mask, highly_variable_scores), where mean is the mean of the provided
        sparse matrix, variance is the variance of the provided sparse matrix, highly_variable_mask is a bool array
        indicating whether an element (a gene) is highly variable in the matrix, and highly_variable_scores is an array
        storing the dispersion score for each gene.
    """

    main_debug("type of layer_mat:" + str(type(layer_mat)))
    if issparse(layer_mat):
        main_info("layer_mat is sparse, dispatch to sparse calc function...")
        mean, variance, dispersion = calc_mean_var_dispersion_sparse(layer_mat)
    else:
        main_info("layer_mat is np, dispatch to sparse calc function...")
        mean, variance, dispersion = calc_mean_var_dispersion_ndarray(layer_mat)

    highly_variable_mask, highly_variable_scores = get_highly_variable_mask_by_dispersion_svr(
        mean, variance, n_top_genes
    )
    variance = np.array(variance).flatten()

    return mean.flatten(), variance, highly_variable_mask, highly_variable_scores


def SVRs(
    adata_ori: anndata.AnnData,
    filter_bool: Union[np.ndarray, None] = None,
    layers: str = "X",
    relative_expr: bool = True,
    total_szfactor: str = "total_Size_Factor",
    min_expr_cells: int = 0,
    min_expr_avg: int = 0,
    max_expr_avg: int = np.inf,
    svr_gamma: Union[float, None] = None,
    winsorize: bool = False,
    winsor_perc: Tuple[float, float] = (1, 99.5),
    sort_inverse: bool = False,
    use_all_genes_cells: bool = False,
) -> anndata.AnnData:
    """Support Vector Regression to identify highly variable genes.

    This function is modified from https://github.com/velocyto-team/velocyto.py/blob/master/velocyto/analysis.py

    Args:
        adata_ori: an AnnData object
        filter_bool: A boolean array from the user to select genes for downstream analysis. Defaults to None.
        layers: The layer(s) to be used for calculating dispersion score via support vector regression (SVR). Defaults
            to "X".
        relative_expr: A logic flag to determine whether we need to divide gene expression values first by size factor
            before run SVR. Defaults to True.
        total_szfactor: The column name in the .obs attribute that corresponds to the size factor for the total mRNA.
            Defaults to "total_Size_Factor".
        min_expr_cells: minimum number of cells that express the gene for it to be considered in the fit. Defaults to 0.
        min_expr_avg: The minimum average of genes across cells required for gene to be selected for SVR analyses.
            Defaults to 0.
        max_expr_avg: The maximum average of genes across cells required for gene to be selected for SVR analyses. Genes
            with average gene expression larger than this value will be treated as house-keeping/outlier genes. Defaults
            to np.inf.
        svr_gamma: the gamma hyper-parameter of the SVR. Defaults to None.
        winsorize: Wether to winsorize the data for the cv vs mean model. Defaults to False.
        winsor_perc: the up and lower bound of the winsorization. Defaults to (1, 99.5).
        sort_inverse: whether to sort genes from less noisy to more noisy (to use for size estimation not for feature
            selection). Defaults to False.
        use_all_genes_cells: A logic flag to determine whether all cells and genes should be used for the size factor
            calculation. Defaults to False.

    Returns:
        An updated annData object with `log_m`, `log_cv`, `score` added to .obs columns and `SVR` added to uns attribute
        as a new key.
    """

    layers = DKM.get_available_layer_keys(adata_ori, layers)

    if use_all_genes_cells:
        # let us ignore the `inplace` parameter in pandas.Categorical.remove_unused_categories  warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adata = adata_ori[:, filter_bool].copy() if filter_bool is not None else adata_ori
    else:
        cell_inds = adata_ori.obs.use_for_pca if "use_for_pca" in adata_ori.obs.columns else adata_ori.obs.index
        filter_list = ["use_for_pca", "pass_basic_filter"]
        filter_checker = [i in adata_ori.var.columns for i in filter_list]
        which_filter = np.where(filter_checker)[0]

        gene_inds = adata_ori.var[filter_list[which_filter[0]]] if len(which_filter) > 0 else adata_ori.var.index

        # let us ignore the `inplace` parameter in pandas.Categorical.remove_unused_categories  warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adata = adata_ori[cell_inds, gene_inds].copy()
        filter_bool = filter_bool[gene_inds]

    for layer in layers:
        if layer == "raw":
            CM = adata.X.copy() if adata.raw is None else adata.raw
            szfactors = (
                adata.obs[layer + "_Size_Factor"].values[:, None]
                if adata.raw.X is not None
                else adata.obs["Size_Factor"].values[:, None]
            )
        elif layer == "X":
            CM = adata.X.copy()
            szfactors = adata.obs["Size_Factor"].values[:, None]
        elif layer == "protein":
            if "protein" in adata.obsm_keys():
                CM = adata.obsm["protein"].copy()
                szfactors = adata.obs[layer + "_Size_Factor"].values[:, None]
            else:
                continue
        else:
            CM = adata.layers[layer].copy()
            szfactors = (
                adata.obs[layer + "_Size_Factor"].values[:, None]
                if layer + "_Size_Factor" in adata.obs.columns
                else None
            )

        if total_szfactor is not None and total_szfactor in adata.obs.keys():
            szfactors = adata.obs[total_szfactor].values[:, None] if total_szfactor in adata.obs.columns else None

        if szfactors is not None and relative_expr:
            if issparse(CM):
                sparsefuncs.inplace_row_scale(CM, 1 / szfactors)
            else:
                CM /= szfactors

        if winsorize:
            if min_expr_cells <= ((100 - winsor_perc[1]) * CM.shape[0] * 0.01):
                min_expr_cells = int(np.ceil((100 - winsor_perc[1]) * CM.shape[1] * 0.01)) + 2

        detected_bool = np.array(
            ((CM > 0).sum(0) >= min_expr_cells) & (CM.mean(0) <= max_expr_avg) & (CM.mean(0) >= min_expr_avg)
        ).flatten()

        valid_CM = CM[:, detected_bool]
        if winsorize:
            down, up = (
                np.percentile(valid_CM.A, winsor_perc, 0)
                if issparse(valid_CM)
                else np.percentile(valid_CM, winsor_perc, 0)
            )
            Sfw = (
                np.clip(valid_CM.A, down[None, :], up[None, :])
                if issparse(valid_CM)
                else np.percentile(valid_CM, winsor_perc, 0)
            )
            mu = Sfw.mean(0)
            sigma = Sfw.std(0, ddof=1)
        else:
            mu = np.array(valid_CM.mean(0)).flatten()
            sigma = (
                np.array(
                    np.sqrt(
                        (valid_CM.multiply(valid_CM).mean(0).A1 - (mu) ** 2)
                        # * (adata.n_obs)
                        # / (adata.n_obs - 1)
                    )
                )
                if issparse(valid_CM)
                else valid_CM.std(0, ddof=1)
            )

        cv = sigma / mu
        log_m = np.array(np.log2(mu)).flatten()
        log_cv = np.array(np.log2(cv)).flatten()
        log_m[mu == 0], log_cv[mu == 0] = 0, 0

        if svr_gamma is None:
            svr_gamma = 150.0 / len(mu)
        # Fit the Support Vector Regression
        clf = SVR(gamma=svr_gamma)
        clf.fit(log_m[:, None], log_cv)
        fitted_fun = clf.predict
        ff = fitted_fun(log_m[:, None])
        score = log_cv - ff
        if sort_inverse:
            score = -score

        prefix = "" if layer == "X" else layer + "_"
        (adata.var[prefix + "log_m"], adata.var[prefix + "log_cv"], adata.var[prefix + "score"],) = (
            np.nan,
            np.nan,
            -np.inf,
        )
        (
            adata.var.loc[detected_bool, prefix + "log_m"],
            adata.var.loc[detected_bool, prefix + "log_cv"],
            adata.var.loc[detected_bool, prefix + "score"],
        ) = (
            np.array(log_m).flatten(),
            np.array(log_cv).flatten(),
            np.array(score).flatten(),
        )

        key = "velocyto_SVR" if layer == "raw" or layer == "X" else layer + "_velocyto_SVR"
        adata_ori.uns[key] = {"SVR": fitted_fun}

    adata_ori = merge_adata_attrs(adata_ori, adata, attr="var")

    return adata_ori


def select_genes_monocle(
    adata: anndata.AnnData,
    layer: str = "X",
    total_szfactor: str = "total_Size_Factor",
    keep_filtered: bool = True,
    sort_by: Literal["SVR", "gini", "dispersion"] = "SVR",
    n_top_genes: int = 2000,
    SVRs_kwargs: dict = {},
    only_bools: bool = False,
    exprs_frac_for_gene_exclusion: float = 1,
    genes_to_exclude: Union[List[str], None] = None,
) -> Union[anndata.AnnData, np.ndarray]:
    """Select genes based on monocle recipe.

    This version is here for modularization of preprocessing, so that users may try combinations of different
    preprocessing procesudres in Preprocessor.

    Args:
        adata: an AnnData object.
        layer: The data from a particular layer (include X) used for feature selection. Defaults to "X".
        total_szfactor: The column name in the .obs attribute that corresponds to the size factor for the total mRNA.
            Defaults to "total_Size_Factor".
        keep_filtered: Whether to keep genes that don't pass the filtering in the adata object. Defaults to True.
        sort_by: the sorting methods, either SVR, dispersion or Gini index, to be used to select genes. Defaults to
            "SVR".
        n_top_genes: the number of top genes based on scoring method (specified by sort_by) will be selected as feature
            genes. Defaults to 2000.
        SVRs_kwargs: kwargs for `SVRs`. Defaults to {}.
        only_bools: Only return a vector of bool values. Defaults to False.
        exprs_frac_for_gene_exclusion: threshold of fractions for high fraction genes. Defaults to 1.
        genes_to_exclude: genes that are excluded from evaluation. Defaults to None.

    Returns:
        The adata object with genes updated if `only_bools` is false. Otherwise, the bool array representing selected
        genes.
    """

    # The following size factor calculation is now a prerequisite for monocle recipe preprocess in preprocessor.
    adata = calc_sz_factor(
        adata,
        total_layers=adata.uns["pp"]["experiment_total_layers"],
        scale_to=None,
        splicing_total_layers=False,
        X_total_layers=False,
        layers=adata.uns["pp"]["experiment_layers"],
        genes_use_for_norm=None,
    )

    filter_bool = (
        adata.var["pass_basic_filter"]
        if "pass_basic_filter" in adata.var.columns
        else np.ones(adata.shape[1], dtype=bool)
    )

    if adata.shape[1] <= n_top_genes:
        filter_bool = np.ones(adata.shape[1], dtype=bool)
    else:
        if sort_by == "dispersion":
            table = top_table(adata, layer, mode="dispersion")
            valid_table = table.query("dispersion_empirical > dispersion_fit")
            valid_table = valid_table.loc[
                set(adata.var.index[filter_bool]).intersection(valid_table.index),
                :,
            ]
            gene_id = np.argsort(-valid_table.loc[:, "dispersion_empirical"])[:n_top_genes]
            gene_id = valid_table.iloc[gene_id, :].index
            filter_bool = adata.var.index.isin(gene_id)
        elif sort_by == "gini":
            table = top_table(adata, layer, mode="gini")
            valid_table = table.loc[filter_bool, :]
            gene_id = np.argsort(-valid_table.loc[:, "gini"])[:n_top_genes]
            gene_id = valid_table.index[gene_id]
            filter_bool = gene_id.isin(adata.var.index)
        elif sort_by == "SVR":
            SVRs_args = {
                "min_expr_cells": 0,
                "min_expr_avg": 0,
                "max_expr_avg": np.inf,
                "svr_gamma": None,
                "winsorize": False,
                "winsor_perc": (1, 99.5),
                "sort_inverse": False,
            }
            SVRs_args = update_dict(SVRs_args, SVRs_kwargs)
            adata = SVRs(
                adata,
                layers=[layer],
                total_szfactor=total_szfactor,
                filter_bool=filter_bool,
                **SVRs_args,
            )

            filter_bool = get_svr_filter(adata, layer=layer, n_top_genes=n_top_genes, return_adata=False)

    # filter genes by gene expression fraction as well
    adata.var["frac"], invalid_ids = compute_gene_exp_fraction(X=adata.X, threshold=exprs_frac_for_gene_exclusion)
    genes_to_exclude = (
        list(adata.var_names[invalid_ids])
        if genes_to_exclude is None
        else genes_to_exclude + list(adata.var_names[invalid_ids])
    )
    if genes_to_exclude is not None and len(genes_to_exclude) > 0:
        adata_exclude_genes = adata.var.index.intersection(genes_to_exclude)
        adata.var.loc[adata_exclude_genes, "use_for_pca"] = False

    if keep_filtered:
        adata.var["use_for_pca"] = filter_bool
    else:
        adata._inplace_subset_var(filter_bool)
        adata.var["use_for_pca"] = True

    return filter_bool if only_bools else adata


def get_highly_variable_mask_by_dispersion_svr(
    mean: np.ndarray,
    var: np.ndarray,
    n_top_genes: int,
    svr_gamma: Union[float, None] = None,
    return_scores: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Returns the mask with shape same as mean and var.

    The mask indicates whether each index is highly variable or not. Each index should represent a gene.

    Args:
        mean: mean of the genes.
        var: variance of the genes.
        n_top_genes: the number of top genes to be inspected.
        svr_gamma: coefficient for support vector regression. Defaults to None.
        return_scores: whether returen the dispersion scores. Defaults to True.

    Returns:
        A tuple (highly_variable_mask, scores) where highly_variable_mask is a bool array indicating whether an element
        (a gene) is highly variable in the matrix and scores is an array recording variable score for each gene. scores
        would only be returned when `return_scores` is True.
    """

    # normally, select svr_gamma based on #features
    if svr_gamma is None:
        svr_gamma = 150.0 / len(mean)
    from sklearn.svm import SVR

    mean_log = np.log2(mean)
    cv_log = np.log2(np.sqrt(var) / mean)
    classifier = SVR(gamma=svr_gamma)
    # fit&preidction will complain about nan values if not take cared here
    is_nan_indices = np.logical_or(np.isnan(mean_log), np.isnan(cv_log))
    if np.sum(is_nan_indices) > 0:
        main_warning(
            (
                "mean and cv_log contain NAN values. We exclude them in SVR training. Please use related gene filtering "
                "methods to filter genes with zero means."
            )
        )

    classifier.fit(mean_log[~is_nan_indices, np.newaxis], cv_log.reshape([-1, 1])[~is_nan_indices])
    scores = np.repeat(np.nan, len(mean_log))
    # TODO handle nan values during prediction here
    scores[~is_nan_indices] = cv_log[~is_nan_indices] - classifier.predict(mean_log[~is_nan_indices, np.newaxis])
    scores = scores.reshape([-1, 1])  # shape should be #genes x 1

    # score threshold based on n top genes
    n_top_genes = min(n_top_genes, len(mean))  # maybe not enough genes there
    score_threshold = np.sort(-scores)[n_top_genes - 1]
    highly_variable_mask = scores >= score_threshold
    highly_variable_mask = np.array(highly_variable_mask).flatten()
    if return_scores:
        return highly_variable_mask, scores
    return highly_variable_mask


def log1p_adata_layer(adata: AnnData, layer: str = DKM.X_LAYER, copy: bool = False) -> AnnData:
    """Calculate log1p of adata's specific layer.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on. Defaults to DKM.X_LAYER.
        copy: whether operate on the original object or on a copied one and return it. Defaults to False.

    Returns:
        The updated AnnData object.
    """

    _adata = adata
    if copy:
        _adata = copy_adata(adata)
    log1p_inplace(_adata, layer=layer)
    return _adata


def log1p_adata(adata: AnnData, layers: list = [DKM.X_LAYER], copy: bool = False) -> AnnData:
    """Perform log1p transform on selected adata layers

    Args:
        adata: an AnnData object.
        layers: the layers to operate on. Defaults to [DKM.X_LAYER].
        copy: whether operate on the original object or on a copied one and return it. Defaults to False.

    Returns:
        The updated AnnData object.
    """

    _adata = adata
    if copy:
        _adata = copy_adata(adata)

    main_info("log1p transform applied to layers: %s" % (str(layers)))
    for layer in layers:
        log1p_adata_layer(_adata, layer=layer)
    return _adata


def _log1p_inplace(data: np.ndarray) -> np.ndarray:
    """Calculate log1p (log(1+x)) of an array and update the array inplace.

    Args:
        data: the array for calculation.

    Returns:
        The updated array.
    """

    return np.log1p(data, out=data)


def log1p_inplace(adata: AnnData, layer: str = DKM.X_LAYER) -> None:
    """Calculate log1p (log(1+x)) for a layer of an AnnData object inplace.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on. Defaults to DKM.X_LAYER.
    """

    mat = DKM.select_layer_data(adata, layer, copy=False)
    if issparse(mat):
        if is_integer_arr(mat.data):
            mat = mat.asfptype()
            DKM.set_layer_data(adata, layer, mat)
        _log1p_inplace(mat.data)
    else:
        mat = mat.astype(np.float)
        _log1p_inplace(mat)


def filter_genes_by_outliers(
    adata: anndata.AnnData,
    filter_bool: Union[np.ndarray, None] = None,
    layer: str = "all",
    min_cell_s: int = 1,
    min_cell_u: int = 1,
    min_cell_p: int = 1,
    min_avg_exp_s: float = 1e-10,
    min_avg_exp_u: float = 0,
    min_avg_exp_p: float = 0,
    max_avg_exp: float = np.inf,
    min_count_s: int = 0,
    min_count_u: int = 0,
    min_count_p: int = 0,
    shared_count: int = 30,
    inplace: bool = False,
) -> Union[anndata.AnnData, pd.DataFrame]:
    """Basic filter of genes based a collection of expression filters.

    Args:
        adata: an AnnData object.
        filter_bool: A boolean array from the user to select genes for downstream analysis. Defaults to None.
        layer: the data from a particular layer (include X) used for feature selection. Defaults to "all".
        min_cell_s: minimal number of cells with expression for the data in the spliced layer (also used for X).
            Defaults to 1.
        min_cell_u: minimal number of cells with expression for the data in the unspliced layer. Defaults to 1.
        min_cell_p: minimal number of cells with expression for the data in the protein layer. Defaults to 1.
        min_avg_exp_s: minimal average expression across cells for the data in the spliced layer (also used for X).
            Defaults to 1e-10.
        min_avg_exp_u: minimal average expression across cells for the data in the unspliced layer. Defaults to 0.
        min_avg_exp_p: minimal average expression across cells for the data in the protein layer. Defaults to 0.
        max_avg_exp: maximal average expression across cells for the data in all layers (also used for X). Defaults to
            np.inf.
        min_count_s: minimal number of counts (UMI/expression) for the data in the spliced layer (also used for X).
            Defaults to 0.
        min_count_u: minimal number of counts (UMI/expression) for the data in the unspliced layer. Defaults to 0.
        min_count_p: minimal number of counts (UMI/expression) for the data in the protein layer. Defaults to 0.
        shared_count: the minimal shared number of counts for each genes across cell between layers. Defaults to 30.
        inplace: whether to update the layer inplace. Defaults to False.

    Returns:
        An updated AnnData object with genes filtered if `inplace` is true. Otherwise, an array containing filtered
        genes.
    """

    detected_bool = np.ones(adata.shape[1], dtype=bool)
    detected_bool = (detected_bool) & np.array(
        ((adata.X > 0).sum(0) >= min_cell_s)
        & (adata.X.mean(0) >= min_avg_exp_s)
        & (adata.X.mean(0) <= max_avg_exp)
        & (adata.X.sum(0) >= min_count_s)
    ).flatten()

    # add our filtering for labeling data below

    # TODO refactor with get_in_range_mask
    if "spliced" in adata.layers.keys() and (layer == "spliced" or layer == "all"):
        detected_bool = (
            detected_bool
            & np.array(
                ((adata.layers["spliced"] > 0).sum(0) >= min_cell_s)
                & (adata.layers["spliced"].mean(0) >= min_avg_exp_s)
                & (adata.layers["spliced"].mean(0) <= max_avg_exp)
                & (adata.layers["spliced"].sum(0) >= min_count_s)
            ).flatten()
        )
    if "unspliced" in adata.layers.keys() and (layer == "unspliced" or layer == "all"):
        detected_bool = (
            detected_bool
            & np.array(
                ((adata.layers["unspliced"] > 0).sum(0) >= min_cell_u)
                & (adata.layers["unspliced"].mean(0) >= min_avg_exp_u)
                & (adata.layers["unspliced"].mean(0) <= max_avg_exp)
                & (adata.layers["unspliced"].sum(0) >= min_count_u)
            ).flatten()
        )
    if shared_count is not None:
        # layers = DKM.get_available_layer_keys(adata, "all", False)
        layers = DKM.get_raw_data_layers(adata)
        tmp = get_inrange_shared_counts_mask(adata, layers, shared_count, "gene")
        if tmp.sum() > 2000:
            detected_bool &= tmp
        else:
            # in case the labeling time is very short for pulse experiment or
            # chase time is very long for degradation experiment.
            tmp = get_inrange_shared_counts_mask(
                adata,
                list(set(layers).difference(["new", "labelled", "labeled"])),
                shared_count,
                "gene",
            )
            detected_bool &= tmp

    # The following code need to be updated
    # just remove genes that are not following the protein criteria
    if "protein" in adata.obsm.keys() and layer == "protein":
        detected_bool = (
            detected_bool
            & np.array(
                ((adata.obsm["protein"] > 0).sum(0) >= min_cell_p)
                & (adata.obsm["protein"].mean(0) >= min_avg_exp_p)
                & (adata.obsm["protein"].mean(0) <= max_avg_exp)
                & (adata.layers["protein"].sum(0) >= min_count_p)  # TODO potential bug confirmation: obsm?
            ).flatten()
        )

    filter_bool = filter_bool & detected_bool if filter_bool is not None else detected_bool

    adata.var["pass_basic_filter"] = np.array(filter_bool).flatten()

    if inplace:
        adata._inplace_subset_var(adata.var["pass_basic_filter"])
        return adata
    return adata.var["pass_basic_filter"]


def get_sum_in_range_mask(
    data_mat: np.ndarray, min_val: float, max_val: float, axis: int = 0, data_min_val_threshold: float = 0
) -> np.ndarray:
    """Check if data_mat's sum is inrange or not along an axis. data_mat's values < data_min_val_threshold are ignored.

    Args:
        data_mat: the array to be inspected.
        min_val: the lower bound of the range.
        max_val: the upper bound of the range.
        axis: the axis to sum. Defaults to 0.
        data_min_val_threshold: the lower threshold for valid data. Defaults to 0.

    Returns:
        A bool array indicating whether the sum is inrage or not.
    """

    return (
        ((data_mat > data_min_val_threshold).sum(axis) >= min_val)
        & ((data_mat > data_min_val_threshold).sum(axis) <= max_val)
    ).flatten()


def filter_cells_by_outliers(
    adata: anndata.AnnData,
    filter_bool: Union[np.ndarray, None] = None,
    layer: str = "all",
    keep_filtered: bool = False,
    min_expr_genes_s: int = 50,
    min_expr_genes_u: int = 25,
    min_expr_genes_p: int = 1,
    max_expr_genes_s: float = np.inf,
    max_expr_genes_u: float = np.inf,
    max_expr_genes_p: float = np.inf,
    shared_count: Union[int, None] = None,
    spliced_key="spliced",
    unspliced_key="unspliced",
    protein_key="protein",
    obs_store_key="pass_basic_filter",
) -> anndata.AnnData:
    """Select valid cells based on a collection of filters including spliced, unspliced and protein min/max vals.

    Args:
        adata: an AnnData object.
        filter_bool: a boolean array from the user to select cells for downstream analysis. Defaults to None.
        layer: the layer (include X) used for feature selection. Defaults to "all".
        keep_filtered: whether to keep cells that don't pass the filtering in the adata object. Defaults to False.
        min_expr_genes_s: minimal number of genes with expression for a cell in the data from the spliced layer (also
            used for X). Defaults to 50.
        min_expr_genes_u: minimal number of genes with expression for a cell in the data from the unspliced layer.
            Defaults to 25.
        min_expr_genes_p: minimal number of genes with expression for a cell in the data from in the protein layer.
            Defaults to 1.
        max_expr_genes_s: maximal number of genes with expression for a cell in the data from the spliced layer (also
            used for X). Defaults to np.inf.
        max_expr_genes_u: maximal number of genes with expression for a cell in the data from the unspliced layer.
            Defaults to np.inf.
        max_expr_genes_p: maximal number of protein with expression for a cell in the data from the protein layer.
            Defaults to np.inf.
        shared_count: the minimal shared number of counts for each cell across genes between layers. Defaults to None.
        spliced_key: name of the layer storing spliced data. Defaults to "spliced".
        unspliced_key: name of the layer storing unspliced data. Defaults to "unspliced".
        protein_key: name of the layer storing protein data. Defaults to "protein".
        obs_store_key: name of the layer to store the filtered data. Defaults to "pass_basic_filter".

    Raises:
        ValueError: the layer provided is invalid.

    Returns:
        An updated AnnData object indicating the selection of cells for downstream analysis. adata will be subsetted
        with only the cells pass filtering if keep_filtered is set to be False.
    """

    predefined_layers_for_filtering = [DKM.X_LAYER, spliced_key, unspliced_key, protein_key]
    predefined_range_dict = {
        DKM.X_LAYER: (min_expr_genes_s, max_expr_genes_s),
        spliced_key: (min_expr_genes_s, max_expr_genes_s),
        unspliced_key: (min_expr_genes_u, max_expr_genes_u),
        protein_key: (min_expr_genes_p, max_expr_genes_p),
    }
    layer_keys_used_for_filtering = []
    if layer == "all":
        layer_keys_used_for_filtering = predefined_layers_for_filtering
    elif isinstance(layer, str) and layer in predefined_layers_for_filtering:
        layer_keys_used_for_filtering = [layer]
    elif isinstance(layer, list) and set(layer) <= set(predefined_layers_for_filtering):
        layer_keys_used_for_filtering = layer
    else:
        raise ValueError(
            "layer should be str or list, and layer should be one of or a subset of "
            + str(predefined_layers_for_filtering)
        )

    detected_bool = get_filter_mask_cells_by_outliers(
        adata, layer_keys_used_for_filtering, predefined_range_dict, shared_count
    )

    if filter_bool is None:
        filter_bool = detected_bool
    else:
        filter_bool = np.array(filter_bool) & detected_bool

    main_info_insert_adata_obs(obs_store_key)
    if keep_filtered:
        main_info("keep filtered cell", indent_level=2)
        adata.obs[obs_store_key] = filter_bool
    else:
        main_info("inplace subsetting adata by filtered cells", indent_level=2)
        adata._inplace_subset_obs(filter_bool)
        adata.obs[obs_store_key] = True

    return adata


def get_filter_mask_cells_by_outliers(
    adata: anndata.AnnData,
    layers: List[str] = None,
    layer2range: dict = None,
    shared_count: Union[int, None] = None,
) -> np.ndarray:
    """Select valid cells based on a collection of filters including spliced, unspliced and protein min/max vals.

    Args:
        adata: an AnnData object.
        layers: a list of layers to be operated on. Defaults to None.
        layer2range: a dict of ranges, layer str to range tuple. Defaults to None.
        shared_count: the minimal shared number of counts for each cell across genes between layers. Defaults to None.

    Returns:
        A bool array indicating valid cells.
    """

    detected_mask = np.full(adata.n_obs, True)
    if layers is None:
        main_info("layers for filtering cells are None, reserve all cells.")
        return detected_mask

    for i, layer in enumerate(layers):
        if layer not in layer2range:
            main_info(
                "skip filtering cells by layer: %s as it is not in the layer2range mapping passed in:" % layer,
                indent_level=2,
            )
            continue
        if not DKM.check_if_layer_exist(adata, layer):
            main_info("skip filtering by layer:%s as it is not in adata." % layer)
            continue

        main_info("filtering cells by layer:%s" % layer, indent_level=2)
        layer_data = DKM.select_layer_data(adata, layer)
        detected_mask = detected_mask & get_sum_in_range_mask(
            layer_data, layer2range[layer][0], layer2range[layer][1], axis=1, data_min_val_threshold=0
        )

    if shared_count is not None:
        main_info("filtering cells by shared counts from all layers", indent_level=2)
        layers = DKM.get_available_layer_keys(adata, layers, False)
        detected_mask = detected_mask & get_inrange_shared_counts_mask(adata, layers, shared_count, "cell")

    detected_mask = np.array(detected_mask).flatten()
    return detected_mask


# TODO It is easy prone to refactor the function below. will refactor together with DKM replacements.
def calc_sz_factor(
    adata_ori: anndata.AnnData,
    layers: Union[str, List[str]] = "all",
    total_layers: Union[List[str], None] = None,
    splicing_total_layers: bool = False,
    X_total_layers: bool = False,
    locfunc: Callable = np.nanmean,
    round_exprs: bool = False,
    method: Literal["mean-geometric-mean-total", "geometric", "median"] = "median",
    scale_to: Union[float, None] = None,
    use_all_genes_cells: bool = True,
    genes_use_for_norm: Union[List[str], None] = None,
) -> anndata.AnnData:
    """Calculate the size factor of the each cell using geometric mean of total UMI across cells for a AnnData object.

    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Args:
        adata_ori: an AnnData object.
        layers: the layer(s) to be normalized. Defaults to "all", including RNA (X, raw) or spliced, unspliced, protein,
            etc.
        total_layers: the layer(s) that can be summed up to get the total mRNA. for example, ["spliced", "unspliced"],
            ["uu", "ul", "su", "sl"] or ["new", "old"], etc. Defaults to None.
        splicing_total_layers: whether to also normalize spliced / unspliced layers by size factor from total RNA.
            Defaults to False.
        X_total_layers: whether to also normalize adata.X by size factor from total RNA. Defaults to False.
        locfunc: the function to normalize the data. Defaults to np.nanmean.
        round_exprs: whether the gene expression should be rounded into integers. Defaults to False.
        method: the method used to calculate the expected total reads / UMI used in size factor calculation. Only
            `mean-geometric-mean-total` / `geometric` and `median` are supported. When `median` is used, `locfunc` will
            be replaced with `np.nanmedian`. Defaults to "median".
        scale_to: the final total expression for each cell that will be scaled to. Defaults to None.
        use_all_genes_cells: whether all cells and genes should be used for the size factor calculation. Defaults to
            True.
        genes_use_for_norm: A list of gene names that will be used to calculate total RNA for each cell and then the
            size factor for normalization. This is often very useful when you want to use only the host genes to
            normalize the dataset in a virus infection experiment (i.e. CMV or SARS-CoV-2 infection). Defaults to None.

    Returns:
        An updated anndata object that are updated with the `Size_Factor` (`layer_` + `Size_Factor`) column(s) in the
        obs attribute.
    """

    if use_all_genes_cells:
        # let us ignore the `inplace` parameter in pandas.Categorical.remove_unused_categories  warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _adata = adata_ori if genes_use_for_norm is None else adata_ori[:, genes_use_for_norm]
    else:
        cell_inds = adata_ori.obs.use_for_pca if "use_for_pca" in adata_ori.obs.columns else adata_ori.obs.index
        filter_list = ["use_for_pca", "pass_basic_filter"]
        filter_checker = [i in adata_ori.var.columns for i in filter_list]
        which_filter = np.where(filter_checker)[0]

        gene_inds = adata_ori.var[filter_list[which_filter[0]]] if len(which_filter) > 0 else adata_ori.var.index

        _adata = adata_ori[cell_inds, :][:, gene_inds]

        if genes_use_for_norm is not None:
            # let us ignore the `inplace` parameter in pandas.Categorical.remove_unused_categories  warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                _adata = _adata[:, _adata.var_names.intersection(genes_use_for_norm)]

    if total_layers is not None:
        if not isinstance(total_layers, list):
            total_layers = [total_layers]
        if len(set(total_layers).difference(_adata.layers.keys())) == 0:
            total = None
            for t_key in total_layers:
                total = _adata.layers[t_key] if total is None else total + _adata.layers[t_key]
            _adata.layers["_total_"] = total
            layers.extend(["_total_"])

    layers = DKM.get_available_layer_keys(_adata, layers)
    if "raw" in layers and _adata.raw is None:
        _adata.raw = _adata.copy()

    excluded_layers = []
    if not X_total_layers:
        excluded_layers.extend(["X"])
    if not splicing_total_layers:
        excluded_layers.extend(["spliced", "unspliced"])

    for layer in layers:
        if layer in excluded_layers:
            sfs, cell_total = sz_util(
                _adata,
                layer,
                round_exprs,
                method,
                locfunc,
                total_layers=None,
                scale_to=scale_to,
            )
        else:
            sfs, cell_total = sz_util(
                _adata,
                layer,
                round_exprs,
                method,
                locfunc,
                total_layers=total_layers,
                scale_to=scale_to,
            )

        sfs[~np.isfinite(sfs)] = 1
        if layer == "raw":
            _adata.obs[layer + "_Size_Factor"] = sfs
            _adata.obs["Size_Factor"] = sfs
            _adata.obs["initial_cell_size"] = cell_total
        elif layer == "X":
            _adata.obs["Size_Factor"] = sfs
            _adata.obs["initial_cell_size"] = cell_total
        elif layer == "_total_":
            _adata.obs["total_Size_Factor"] = sfs
            _adata.obs["initial" + layer + "cell_size"] = cell_total
            del _adata.layers["_total_"]
        else:
            _adata.obs[layer + "_Size_Factor"] = sfs
            _adata.obs["initial_" + layer + "_cell_size"] = cell_total

    adata_ori = merge_adata_attrs(adata_ori, _adata, attr="obs")

    return adata_ori


# TODO refactor the function below
def normalize_cell_expr_by_size_factors(
    adata: anndata.AnnData,
    layers: str = "all",
    total_szfactor: str = "total_Size_Factor",
    splicing_total_layers: bool = False,
    X_total_layers: bool = False,
    norm_method: Union[Callable, None] = None,
    pseudo_expr: int = 1,
    relative_expr: bool = True,
    keep_filtered: bool = True,
    recalc_sz: bool = False,
    sz_method: Literal["mean-geometric-mean-total", "geometric", "median"] = "median",
    scale_to: Union[float, None] = None,
    skip_log: bool = False,
) -> anndata.AnnData:
    """Normalize the gene expression value for the AnnData object.

    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Args:
        adata: an AnnData object.
        layers: the layer(s) to be normalized. Default is all, including RNA (X, raw) or spliced, unspliced, protein,
            etc.
        total_szfactor: the column name in the .obs attribute that corresponds to the size factor for the total mRNA.
            Defaults to "total_Size_Factor".
        splicing_total_layers: whether to also normalize spliced / unspliced layers by size factor from total RNA.
            Defaults to False.
        X_total_layers: whether to also normalize adata.X by size factor from total RNA. Defaults to False.
        norm_method: the method used to normalize data. Can be either function `np.log1p`, `np.log2` or any other
            functions or string `clr`. By default, only .X will be size normalized and log1p transformed while data in
            other layers will only be size normalized. Defaults to None.
        pseudo_expr: a pseudocount added to the gene expression value before log/log2 normalization. Defaults to 1.
        relative_expr: whether we need to divide gene expression values first by size factor before normalization.
            Defaults to True.
        keep_filtered: whether we will only store feature genes in the adata object. If it is False, size factor will be
            recalculated only for the selected feature genes. Defaults to True.
        recalc_sz: whether we need to recalculate size factor based on selected genes before normalization. Defaults to
            False.
        sz_method: the method used to calculate the expected total reads / UMI used in size factor calculation. Only
            `mean-geometric-mean-total` / `geometric` and `median` are supported. When `median` is used, `locfunc` will
            be replaced with `np.nanmedian`. Defaults to "median".
        scale_to: the final total expression for each cell that will be scaled to. Defaults to None.
        skip_log: whether skip log transformation. Defaults to False.

    Returns:
        An updated anndata object that are updated with normalized expression values for different layers.
    """

    if recalc_sz:
        if "use_for_pca" in adata.var.columns and keep_filtered is False:
            adata = adata[:, adata.var.loc[:, "use_for_pca"]]

        adata.obs = adata.obs.loc[:, ~adata.obs.columns.str.contains("Size_Factor")]

    layers = DKM.get_available_layer_keys(adata, layers)

    layer_sz_column_names = [i + "_Size_Factor" for i in set(layers).difference("X")]
    layer_sz_column_names.extend(["Size_Factor"])
    # layers_to_sz = list(set(layer_sz_column_names).difference(adata.obs.keys()))
    layers_to_sz = list(set(layer_sz_column_names))
    if len(layers_to_sz) > 0:
        layers = pd.Series(layers_to_sz).str.split("_Size_Factor", expand=True).iloc[:, 0].tolist()
        if "Size_Factor" in layers:
            layers[np.where(np.array(layers) == "Size_Factor")[0][0]] = "X"
        calc_sz_factor(
            adata,
            layers=layers,
            locfunc=np.nanmean,
            round_exprs=True,
            method=sz_method,
            scale_to=scale_to,
        )

    excluded_layers = []
    if not X_total_layers:
        excluded_layers.extend(["X"])
    if not splicing_total_layers:
        excluded_layers.extend(["spliced", "unspliced"])

    main_info("size factor normalize following layers: " + str(layers))
    for layer in layers:
        if layer in excluded_layers:
            szfactors, CM = get_sz_exprs(adata, layer, total_szfactor=None)
        else:
            szfactors, CM = get_sz_exprs(adata, layer, total_szfactor=total_szfactor)

        # log transforms

        # special default norm case for adata.X in monocle logics
        if norm_method is None and layer == "X":
            _norm_method = np.log1p
        else:
            _norm_method = norm_method

        if skip_log:
            main_info("skipping log transformation as input requires...")
            _norm_method = None

        if _norm_method in [np.log1p, np.log, np.log2, Freeman_Tukey, None] and layer != "protein":
            main_info("applying %s to layer<%s>" % (_norm_method, layer))
            CM = normalize_mat_monocle(CM, szfactors, relative_expr, pseudo_expr, _norm_method)

        elif layer == "protein":  # norm_method == 'clr':
            if _norm_method != "clr":
                main_warning(
                    "For protein data, log transformation is not recommended. Using clr normalization by default."
                )
            """This normalization implements the centered log-ratio (CLR) normalization from Seurat which is computed
            for each gene (M Stoeckius, 2017).
            """
            CM = CM.T
            n_feature = CM.shape[1]

            for i in range(CM.shape[0]):
                x = CM[i].A if issparse(CM) else CM[i]
                res = np.log1p(x / (np.exp(np.nansum(np.log1p(x[x > 0])) / n_feature)))
                res[np.isnan(res)] = 0
                # res[res > 100] = 100
                # no .A is required # https://stackoverflow.com/questions/28427236/set-row-of-csr-matrix
                CM[i] = res

            CM = CM.T
        else:
            main_warning(_norm_method + " is not implemented yet")

        if layer in ["raw", "X"]:
            main_info("set adata <X> to normalized data.")
            adata.X = CM
        elif layer == "protein" and "protein" in adata.obsm_keys():
            main_info_insert_adata_obsm("X_protein")
            adata.obsm["X_protein"] = CM
        else:
            main_info_insert_adata_obsm("X_" + layer)
            adata.layers["X_" + layer] = CM

        main_info_insert_adata_uns("pp.norm_method")
        adata.uns["pp"]["norm_method"] = _norm_method.__name__ if callable(_norm_method) else _norm_method

    return adata


def is_nonnegative(mat: Union[np.ndarray, spmatrix, list]) -> bool:
    """Test whether all elements of an array or sparse array are non-negative.

    Args:
        mat: an array in ndarray or sparse array in scipy spmatrix.

    Returns:
        A flag whether all elements are non-negative.
    """

    if scipy.sparse.issparse(mat):
        return np.all(mat.sign().data >= 0)
    return np.all(np.sign(mat) >= 0)


def is_integer_arr(arr: Union[np.ndarray, spmatrix, list]) -> bool:
    """Test if an array like obj's dtype is integer

    Args:
        arr: an array like object.

    Returns:
        A flag whether the array's dtype is integer.
    """

    return np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, int)


def is_float_integer_arr(arr: Union[np.ndarray, spmatrix, list]) -> bool:
    """Test if an array's elements are integers

    Args:
        arr: an input array.

    Returns:
        A flag whether all elements of the array are integers.
    """

    if issparse(arr):
        arr = arr.data
    return np.all(np.equal(np.mod(arr, 1), 0))


def is_nonnegative_integer_arr(mat: Union[np.ndarray, spmatrix, list]) -> bool:
    """Test if an array's elements are non-negative integers

    Args:
        mat: an input array.

    Returns:
        A flag whether all elements of the array are non-negative integers.
    """

    if (not is_integer_arr(mat)) and (not is_float_integer_arr(mat)):
        return False
    return is_nonnegative(mat)


def pca_selected_genes_wrapper(
    adata: AnnData, pca_input: Union[np.ndarray, None] = None, n_pca_components: int = 30, key: str = "X_pca"
):
    """A wrapper for pca_monocle function to reduce dimensions of the Adata with PCA.

    Args:
        adata: an AnnData object.
        pca_input: an array for nearest neighbor search directly. Defaults to None.
        n_pca_components: number of PCA components. Defaults to 30.
        key: the key to store the calculation result. Defaults to "X_pca".
    """

    adata = pca_monocle(adata, pca_input, n_pca_components=n_pca_components, pca_key=key)
