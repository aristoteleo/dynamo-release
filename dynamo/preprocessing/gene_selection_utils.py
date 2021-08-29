from anndata import AnnData
import anndata
from ..utils import copy_adata
from ..dynamo_logger import (
    main_finish_progress,
    main_info,
    main_info_insert_adata,
    main_info_insert_adata_var,
    main_log_time,
)
from ..configuration import DynamoAdataKeyManager
from typing import List, Union
import numpy as np
from scipy.sparse.base import issparse
import pandas as pd


def get_nan_or_inf_data_bool_mask(arr):
    mask = np.isnan(arr) | np.isinf(arr) | np.isneginf(arr)
    return mask


def clip_by_perc(layer_mat):
    # TODO
    return


def calc_mean_var_dispersion(data_mat: np.array) -> List[np.ndarray]:
    # per gene mean, var and dispersion
    mean = np.nanmean(data_mat, axis=0)
    var = np.nanvar(data_mat, axis=0)
    dispersion = var / mean
    return mean, var, dispersion


def calc_mean_var_dispersion_sparse(sparse_mat) -> List[np.ndarray]:
    nan_mask = get_nan_or_inf_data_bool_mask(sparse_mat.data)

    non_nan_count = sparse_mat.shape[0] - nan_mask.sum()
    mean = (sparse_mat.sum(0) / non_nan_count).A1
    # same as numpy var behavior: denominator is N, var=(data_arr-mean)/N
    var = np.power(sparse_mat - mean, 2).sum(0) / non_nan_count
    dispersion = var / mean
    return mean, var, dispersion


def filter_genes_by_dispersion_general(
    adata, layer=DynamoAdataKeyManager.X_LAYER, nan_replace_val=None, n_top_genes=None
):
    main_info("filtering genes by dispersion...")
    main_log_time()

    layer_mat = DynamoAdataKeyManager.select_layer_data(adata, layer)
    if nan_replace_val:
        main_info("replacing nan values with: %s" % (nan_replace_val))
        mask = get_nan_or_inf_data_bool_mask(layer_mat)
        layer_mat[mask] = nan_replace_val
    filter_genes_by_dispersion_svr(adata, layer_mat, n_top_genes)

    main_finish_progress("filter genes by dispersion")


def filter_genes_by_dispersion_svr(adata, layer_mat, n_top_genes) -> None:
    mean, var, dispersion = calc_mean_var_dispersion(layer_mat)
    highly_variable_mask = get_highly_variable_mask_by_dispersion_svr(adata, mean, var, n_top_genes)

    main_info_insert_adata_var(DynamoAdataKeyManager.VAR_GENE_MEAN_KEY)
    main_info_insert_adata_var(DynamoAdataKeyManager.VAR_GENE_VAR_KEY)
    main_info_insert_adata_var(DynamoAdataKeyManager.VAR_GENE_HIGHLY_VARIABLE_KEY)
    adata.var[DynamoAdataKeyManager.VAR_GENE_MEAN_KEY] = mean
    adata.var[DynamoAdataKeyManager.VAR_GENE_VAR_KEY] = var
    adata.var[DynamoAdataKeyManager.VAR_GENE_HIGHLY_VARIABLE_KEY] = highly_variable_mask


def get_highly_variable_mask_by_dispersion_svr(mean, var, n_top_genes: int, svr_gamma: float = None):
    # normally, select svr_gamma based on #features
    if svr_gamma is None:
        svr_gamma = 150.0 / len(mean)
    from sklearn.svm import SVR

    mean_log = np.log2(mean)

    mean[mean == 0] += 1e-8  # prevent division by zero
    cv_log = np.log2(np.sqrt(var) / mean)

    clf = SVR(gamma=svr_gamma)
    clf.fit(mean_log[:, None], cv_log)
    score = cv_log - clf.predict(mean_log[:, None])

    # score threshold based on n top genes
    score_threshold = np.sort(-score)[n_top_genes - 1]
    highly_variable_mask = score >= score_threshold
    return highly_variable_mask


def log1p(adata, copy=False) -> AnnData:
    _adata = adata
    if copy:
        _adata = copy_adata(adata)
    log1p_inplace(_adata)
    return _adata


def log1p_inplace(adata):
    if issparse(adata.X):
        log1p_inplace(adata.X.data)
    else:
        log1p_inplace(adata.X)


def log1p_inplace(data):
    np.log1p(data, out=data)


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
    max_avg_exp: float = np.infty,
    min_count_s: int = 0,
    min_count_u: int = 0,
    min_count_p: int = 0,
    shared_count: int = 30,
) -> anndata.AnnData:
    """Basic filter of genes based a collection of expression filters.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object.
        filter_bool: :class:`~numpy.ndarray` (default: None)
            A boolean array from the user to select genes for downstream analysis.
        layer: `str` (default: `X`)
            The data from a particular layer (include X) used for feature selection.
        min_cell_s: `int` (default: `5`)
            Minimal number of cells with expression for the data in the spliced layer (also used for X).
        min_cell_u: `int` (default: `5`)
            Minimal number of cells with expression for the data in the unspliced layer.
        min_cell_p: `int` (default: `5`)
            Minimal number of cells with expression for the data in the protein layer.
        min_avg_exp_s: `float` (default: `1e-2`)
            Minimal average expression across cells for the data in the spliced layer (also used for X).
        min_avg_exp_u: `float` (default: `1e-4`)
            Minimal average expression across cells for the data in the unspliced layer.
        min_avg_exp_p: `float` (default: `1e-4`)
            Minimal average expression across cells for the data in the protein layer.
        max_avg_exp: `float` (default: `100`.)
            Maximal average expression across cells for the data in all layers (also used for X).
        min_cell_s: `int` (default: `5`)
            Minimal number of counts (UMI/expression) for the data in the spliced layer (also used for X).
        min_cell_u: `int` (default: `5`)
            Minimal number of counts (UMI/expression) for the data in the unspliced layer.
        min_cell_p: `int` (default: `5`)
            Minimal number of counts (UMI/expression) for the data in the protein layer.
        shared_count: `int` (default: `30`)
            The minimal shared number of counts for each genes across cell between layers.
    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An updated AnnData object with use_for_pca as a new column in .var attributes to indicate the selection of
            genes for downstream analysis. adata will be subsetted with only the genes pass filter if keep_unflitered is
            set to be False.
    """

    detected_bool = np.ones(adata.shape[1], dtype=bool)
    detected_bool = (detected_bool) & np.array(
        ((adata.X > 0).sum(0) >= min_cell_s)
        & (adata.X.mean(0) >= min_avg_exp_s)
        & (adata.X.mean(0) <= max_avg_exp)
        & (adata.X.sum(0) >= min_count_s)
    ).flatten()

    # add our filtering for labeling data below

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
        layers = DynamoAdataKeyManager.get_layer_keys(adata, "all", False)
        tmp = get_shared_counts(adata, layers, shared_count, "gene")
        if tmp.sum() > 2000:
            detected_bool &= tmp
        else:
            # in case the labeling time is very short for pulse experiment or
            # chase time is very long for degradation experiment.
            tmp = get_shared_counts(
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
                & (adata.layers["protein"].sum(0) >= min_count_p)
            ).flatten()
        )

    filter_bool = filter_bool & detected_bool if filter_bool is not None else detected_bool

    adata.var["pass_basic_filter"] = np.array(filter_bool).flatten()

    return adata
