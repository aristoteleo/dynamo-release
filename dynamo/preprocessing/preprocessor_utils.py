import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

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
from sklearn.utils import sparsefuncs

from ..configuration import DKM, DynamoAdataKeyManager
from ..dynamo_logger import (
    main_debug,
    main_finish_progress,
    main_info,
    main_info_insert_adata,
    main_info_insert_adata_layer,
    main_info_insert_adata_obs,
    main_info_insert_adata_obsm,
    main_info_insert_adata_uns,
    main_log_time,
    main_warning,
)
from ..tools.utils import update_dict
from ..utils import copy_adata
from .utils import (
    _Freeman_Tukey,
    add_noise_to_duplicates,
    basic_stats,
    calc_new_to_total_ratio,
    clusters_stats,
    collapse_species_adata,
    compute_gene_exp_fraction,
    get_inrange_shared_counts_mask,
    get_svr_filter,
    get_sz_exprs,
    merge_adata_attrs,
    pca,
    size_factor_normalize,
    sz_util,
)


def is_log1p_transformed_adata(adata: anndata.AnnData) -> bool:
    """check if adata data is log transformed by checking a small subset of adata observations.

    Args:
        adata: an AnnData object

    Returns:
        A flag shows whether the adata object is log transformed.
    """

    chosen_gene_indices = np.random.choice(adata.n_vars, 10)
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
    main_debug(
        f"\nDynamo has detected that your labeling data is from a kin experiment. \nIf the experiment type is incorrect, "
        f"please provide the correct experiment_type (one-shot, kin, or deg)."
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


def log2_adata_layer(adata: AnnData, layer: str = DKM.X_LAYER, copy: bool = False) -> AnnData:
    """Calculate log2 of adata's specific layer.

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
    log2_inplace(_adata, layer=layer)
    return _adata


def log_adata_layer(adata: AnnData, layer: str = DKM.X_LAYER, copy: bool = False) -> AnnData:
    """Calculate log of adata's specific layer.

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
    log_inplace(_adata, layer=layer)
    return _adata


def Freeman_Tukey_adata_layer(adata: AnnData, layer: str = DKM.X_LAYER, copy: bool = False) -> AnnData:
    """Calculate Freeman_Tukey of adata's specific layer.

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
    Freeman_Tukey_inplace(_adata, layer=layer)
    return _adata


def log1p(adata: AnnData, layers: list = [DKM.X_LAYER], copy: bool = False) -> AnnData:
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

    main_debug("[log1p] transform applied to layers: %s" % (str(layers)))
    for layer in layers:
        log1p_adata_layer(_adata, layer=layer)

    main_info_insert_adata_uns("pp.norm_method")
    adata.uns["pp"]["norm_method"] = "log1p"
    return _adata


def log2(adata: AnnData, layers: list = [DKM.X_LAYER], copy: bool = False) -> AnnData:
    """Perform log2 transform on selected adata layers

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

    main_debug("[log2] transform applied to layers: %s" % (str(layers)))
    for layer in layers:
        log2_adata_layer(_adata, layer=layer)

    main_info_insert_adata_uns("pp.norm_method")
    adata.uns["pp"]["norm_method"] = "log2"
    return _adata


def log(adata: AnnData, layers: list = [DKM.X_LAYER], copy: bool = False) -> AnnData:
    """Perform log transform on selected adata layers

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

    main_debug("[log] transform applied to layers: %s" % (str(layers)))
    for layer in layers:
        log_adata_layer(_adata, layer=layer)

    main_info_insert_adata_uns("pp.norm_method")
    adata.uns["pp"]["norm_method"] = "log"
    return _adata


def Freeman_Tukey(adata: AnnData, layers: list = [DKM.X_LAYER], copy: bool = False) -> AnnData:
    """Perform Freeman_Tukey transform on selected adata layers

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

    main_debug("[Freeman_Tukey] transform applied to layers: %s" % (str(layers)))
    for layer in layers:
        Freeman_Tukey_adata_layer(_adata, layer=layer)

    main_info_insert_adata_uns("pp.norm_method")
    adata.uns["pp"]["norm_method"] = "Freeman_Tukey"
    return _adata


def _log1p_inplace(data: np.ndarray) -> np.ndarray:
    """Calculate log1p (log(1+x)) of an array and update the array inplace.

    Args:
        data: the array for calculation.

    Returns:
        The updated array.
    """

    return np.log1p(data, out=data)


def _log2_inplace(data: np.ndarray) -> np.ndarray:
    """Calculate Base-2 logarithm of `x` of an array and update the array inplace.

    Args:
        data: the array for calculation.

    Returns:
        The updated array.
    """

    return np.log2(data + 1, out=data)


def _log_inplace(data: np.ndarray) -> np.ndarray:
    """Calculate the natural logarithm `log(exp(x)) = x` of an array and update the array inplace.

    Args:
        data: the array for calculation.

    Returns:
        The updated array.
    """

    return np.log(data + 1, out=data)


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


def log2_inplace(adata: AnnData, layer: str = DKM.X_LAYER) -> None:
    """Calculate Base-2 logarithm of `x` for a layer of an AnnData object inplace.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on. Defaults to DKM.X_LAYER.
    """

    mat = DKM.select_layer_data(adata, layer, copy=False)
    if issparse(mat):
        if is_integer_arr(mat.data):
            mat = mat.asfptype()
            DKM.set_layer_data(adata, layer, mat)
        _log2_inplace(mat.data)
    else:
        mat = mat.astype(np.float)
        _log2_inplace(mat)


def log_inplace(adata: AnnData, layer: str = DKM.X_LAYER) -> None:
    """Calculate the natural logarithm `log(exp(x)) = x` for a layer of an AnnData object inplace.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on. Defaults to DKM.X_LAYER.
    """

    mat = DKM.select_layer_data(adata, layer, copy=False)
    if issparse(mat):
        if is_integer_arr(mat.data):
            mat = mat.asfptype()
            DKM.set_layer_data(adata, layer, mat)
        _log_inplace(mat.data)
    else:
        mat = mat.astype(np.float)
        _log_inplace(mat)


def Freeman_Tukey_inplace(adata: AnnData, layer: str = DKM.X_LAYER) -> None:
    """Calculate Freeman-Tukey transform for a layer of an AnnData object inplace.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on. Defaults to DKM.X_LAYER.
    """
    mat = DKM.select_layer_data(adata, layer, copy=False)
    if issparse(mat):
        if is_integer_arr(mat.data):
            mat = mat.asfptype()
            DKM.set_layer_data(adata, layer, mat)
        _Freeman_Tukey(mat.data)
    else:
        mat = mat.astype(np.float)
        _Freeman_Tukey(mat)

    mat.data -= 1
    DKM.set_layer_data(adata, layer, mat)


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
    main_info("filtered out %d outlier genes" % (adata.n_vars - sum(filter_bool)), indent_level=2)

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
    adata: AnnData,
    filter_bool: Union[np.ndarray, None] = None,
    layer: str = "all",
    keep_filtered: bool = False,
    min_expr_genes_s: int = 50,
    min_expr_genes_u: int = 25,
    min_expr_genes_p: int = 1,
    max_expr_genes_s: float = np.inf,
    max_expr_genes_u: float = np.inf,
    max_expr_genes_p: float = np.inf,
    max_pmito_s: Optional[float] = None,
    shared_count: Optional[int] = None,
    spliced_key="spliced",
    unspliced_key="unspliced",
    protein_key="protein",
    obs_store_key="pass_basic_filter",
) -> AnnData:
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
        max_pmito_s: maximal percentage of mitochondrial genes for a cell in the data from the spliced layer.
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

    if max_pmito_s is not None:
        detected_bool = detected_bool & (adata.obs["pMito"] < max_pmito_s)
        main_info(
            "filtered out %d cells by %f%% of mitochondrial genes for a cell."
            % (adata.n_obs - (adata.obs["pMito"] < max_pmito_s).sum(), max_pmito_s),
            indent_level=2,
        )

    filter_bool = detected_bool if filter_bool is None else np.array(filter_bool) & detected_bool

    main_info("filtered out %d outlier cells" % (adata.n_obs - sum(filter_bool)), indent_level=2)
    main_info_insert_adata_obs(obs_store_key)

    adata.obs[obs_store_key] = filter_bool

    if not keep_filtered:
        main_debug("inplace subsetting adata by filtered cells", indent_level=2)
        adata._inplace_subset_obs(filter_bool)

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
            main_debug(
                "skip filtering cells by layer: %s as it is not in the layer2range mapping passed in:" % layer,
                indent_level=2,
            )
            continue
        if not DKM.check_if_layer_exist(adata, layer):
            main_debug("skip filtering by layer:%s as it is not in adata." % layer)
            continue

        main_debug("filtering cells by layer:%s" % layer, indent_level=2)
        layer_data = DKM.select_layer_data(adata, layer)
        detected_mask = detected_mask & get_sum_in_range_mask(
            layer_data, layer2range[layer][0], layer2range[layer][1], axis=1, data_min_val_threshold=0
        )

    if shared_count is not None:
        main_debug("filtering cells by shared counts from all layers", indent_level=2)
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
    """Calculate the size factor of each cell using geometric mean or median of total UMI across cells for a AnnData
    object.

    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Args:
        adata_ori: an AnnData object.
        layers: the layer(s) to be normalized. Defaults to "all", including RNA (X, raw) or spliced, unspliced, protein,
            etc.
        total_layers: the layer(s) that can be summed up to get the total mRNA. For example, ["spliced", "unspliced"],
            ["uu", "ul", "su", "sl"] or ["new", "old"], etc. Defaults to None.
        splicing_total_layers: whether to also normalize spliced / unspliced layers by size factor from total RNA.
            Defaults to False.
        X_total_layers: whether to also normalize adata.X by size factor from total RNA. Defaults to False.
        locfunc: the function to normalize the data. Defaults to np.nanmean.
        round_exprs: whether the gene expression should be rounded into integers. Defaults to False.
        method: the method used to calculate the expected total reads / UMI used in size factor calculation. Only
            `mean-geometric-mean-total` / `geometric` and `median` are supported. When `mean-geometric-mean-total` is
            used, size factors will be calculated using the geometric mean with given mean function. When `median` is
            used, `locfunc` will be replaced with `np.nanmedian`. When `mean` is used, `locfunc` will be replaced with
            `np.nanmean`. Defaults to "median".
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
        total_layers, layers = DKM.aggregate_layers_into_total(
            _adata,
            layers=layers,
            total_layers=total_layers,
        )

    layers = DKM.get_available_layer_keys(_adata, layers)
    if "raw" in layers and _adata.raw is None:
        _adata.raw = _adata.copy()

    excluded_layers = DKM.get_excluded_layers(
        X_total_layers=X_total_layers,
        splicing_total_layers=splicing_total_layers,
    )

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
def normalize(
    adata: anndata.AnnData,
    layers: str = "all",
    total_szfactor: str = "total_Size_Factor",
    splicing_total_layers: bool = False,
    X_total_layers: bool = False,
    keep_filtered: bool = True,
    recalc_sz: bool = False,
    sz_method: Literal["mean-geometric-mean-total", "geometric", "median"] = "median",
    scale_to: Union[float, None] = None,
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
        keep_filtered: whether we will only store feature genes in the adata object. If it is False, size factor will be
            recalculated only for the selected feature genes. Defaults to True.
        recalc_sz: whether we need to recalculate size factor based on selected genes before normalization. Defaults to
            False.
        sz_method: the method used to calculate the expected total reads / UMI used in size factor calculation. Only
            `mean-geometric-mean-total` / `geometric` and `median` are supported. When `mean-geometric-mean-total` is
            used, size factors will be calculated using the geometric mean with given mean function. When `median` is
            used, `locfunc` will be replaced with `np.nanmedian`. When `mean` is used, `locfunc` will be replaced with
            `np.nanmean`. Defaults to "median".
        scale_to: the final total expression for each cell that will be scaled to. Defaults to None.

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
    layers_to_sz = list(set(layer_sz_column_names))

    layers = pd.Series(layers_to_sz).str.split("_Size_Factor", expand=True).iloc[:, 0].tolist()
    layers[np.where(np.array(layers) == "Size_Factor")[0][0]] = "X"
    calc_sz_factor(
        adata,
        layers=layers,
        locfunc=np.nanmean,
        round_exprs=True,
        method=sz_method,
        scale_to=scale_to,
    )

    excluded_layers = DKM.get_excluded_layers(
        X_total_layers=X_total_layers,
        splicing_total_layers=splicing_total_layers,
    )

    main_debug("size factor normalize following layers: " + str(layers))
    for layer in layers:
        if layer in excluded_layers:
            szfactors, CM = get_sz_exprs(adata, layer, total_szfactor=None)
        else:
            szfactors, CM = get_sz_exprs(adata, layer, total_szfactor=total_szfactor)

        if layer == "protein":
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
            CM = size_factor_normalize(CM, szfactors)

        if layer in ["raw", "X"]:
            main_debug("set adata <X> to normalized data.")
            adata.X = CM
        elif layer == "protein" and "protein" in adata.obsm_keys():
            main_info_insert_adata_obsm("X_protein")
            adata.obsm["X_protein"] = CM
        else:
            main_info_insert_adata_layer("X_" + layer)
            adata.layers["X_" + layer] = CM

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
    """A wrapper for pca function to reduce dimensions of the Adata with PCA.

    Args:
        adata: an AnnData object.
        pca_input: an array for nearest neighbor search directly. Defaults to None.
        n_pca_components: number of PCA components. Defaults to 30.
        key: the key to store the calculation result. Defaults to "X_pca".
    """

    adata = pca(adata, pca_input, n_pca_components=n_pca_components, pca_key=key)


def regress_out_parallel(
    adata: AnnData,
    layer: str = DKM.X_LAYER,
    obs_keys: Optional[List[str]] = None,
    gene_selection_key: Optional[str] = None,
    n_cores: Optional[int] = None,
):
    """Perform linear regression to remove the effects of given variables from a target variable.

    Args:
        adata: an AnnData object. Feature matrix of shape (n_samples, n_features).
        layer: the layer to regress out. Defaults to "X".
        obs_keys: List of observation keys to be removed.
        gene_selection_key: the key in adata.var that contains boolean for showing genes` filtering results.
            For example, "use_for_pca" is selected then it will regress out only for the genes that are True
            for "use_for_pca". This input will decrease processing time of regressing out data.
        n_cores: Change this to the number of cores on your system for parallel computing. Default to be None.
        obsm_store_key: the key to store the regress out result. Defaults to "X_regress_out".
    """
    main_debug("regress out %s by multiprocessing..." % obs_keys)
    main_log_time()

    if len(obs_keys) < 1:
        main_warning("No variable to regress out")
        return

    if gene_selection_key is None:
        regressor = DKM.select_layer_data(adata, layer)
    else:
        if gene_selection_key not in adata.var.keys():
            raise ValueError(str(gene_selection_key) + " is not a key in adata.var")

        if not (adata.var[gene_selection_key].dtype == bool):
            raise ValueError(str(gene_selection_key) + " is not a boolean")

        subset_adata = adata[:, adata.var.loc[:, gene_selection_key]]
        regressor = DKM.select_layer_data(subset_adata, layer)

    import itertools

    if issparse(regressor):
        regressor = regressor.toarray()

    if n_cores is None:
        n_cores = 1  # Use no parallel computing as default

    # Split the input data into chunks for parallel processing
    chunk_size = min(1000, regressor.shape[1] // n_cores + 1)
    chunk_len = regressor.shape[1] // chunk_size
    regressor_chunks = np.array_split(regressor, chunk_len, axis=1)

    # Select the variables to remove
    remove = adata.obs[obs_keys].to_numpy()

    res = _parallel_wrapper(regress_out_chunk_helper, zip(itertools.repeat(remove), regressor_chunks), n_cores)

    # Remove the effects of the variables from the target variable
    residuals = regressor - np.hstack(res)

    DKM.set_layer_data(adata, layer, residuals)
    main_finish_progress("regress out")


def regress_out_chunk_helper(args):
    """A helper function for each regressout chunk.

    Args:
        args: list of arguments that is used in _regress_out_chunk.

    Returns:
        numpy array: predicted the effects of the variables calculated by _regress_out_chunk.
    """
    obs_feature, gene_expr = args
    return _regress_out_chunk(obs_feature, gene_expr)


def _regress_out_chunk(
    obs_feature: Union[np.ndarray, spmatrix, list], gene_expr: Union[np.ndarray, spmatrix, list]
) -> Union[np.ndarray, spmatrix, list]:
    """Perform a linear regression to remove the effects of cell features (percentage of mitochondria, etc.)

    Args:
        obs_feature: list of observation keys used to regress out their effect to gene expression.
        gene_expr : the current gene expression values of the target variables.

    Returns:
        numpy array: the residuals that are predicted the effects of the variables.
    """
    from sklearn.linear_model import LinearRegression

    # Fit a linear regression model to the variables to remove
    reg = LinearRegression().fit(obs_feature, gene_expr)

    # Predict the effects of the variables to remove
    return reg.predict(obs_feature)


def _parallel_wrapper(func: Callable, args_list, n_cores: Optional[int] = None):
    """A wrapper for parallel operation to regress out of the input variables.

    Args:
        func: The function to be conducted the multiprocessing.
        args_list: The iterable of arguments to be passed to the function.
        n_cores: The number of CPU cores to be used for parallel processing. Default to be None.

    Returns:
        results: The list of results returned by the function for each element of the iterable.
    """
    import multiprocessing as mp

    ctx = mp.get_context("fork")

    with ctx.Pool(n_cores) as pool:
        results = pool.map(func, args_list)
        pool.close()
        pool.join()

    return results
