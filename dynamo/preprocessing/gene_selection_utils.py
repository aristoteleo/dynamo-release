from dynamo.dynamo_logger import (
    main_finish_progress,
    main_info,
    main_info_insert_adata,
    main_info_insert_adata_var,
    main_log_time,
)
from dynamo.configuration import DynamoAdataKeyManager
from typing import List
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
