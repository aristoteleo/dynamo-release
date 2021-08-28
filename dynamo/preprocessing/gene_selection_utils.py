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


def calc_mean_var_dispersion(data_mat) -> List[np.ndarray]:
    # per gene mean, var and dispersion
    # works for both sparse mat and np array
    mean = np.nanmean(data_mat, axis=0)
    var = np.nanvar(data_mat, axis=0)
    dispersion = var / mean
    return mean, var, dispersion


def filter_genes_by_dispersion_seurat(adata, layer="X", nan_replace_val=None):
    layer_data = DynamoAdataKeyManager.select_layer_data(adata, layer)
    if nan_replace_val:
        mask = get_nan_or_inf_data_bool_mask(layer_data)
        layer_data[mask] = nan_replace_val
    # works for both sparse and np array
    # per gene mean, var and dispersion
    mean = np.nanmean(layer_data, axis=0)
    var = np.nanvar(layer_data, axis=0)
    dispersion = var / mean
    return mean, var


def filter_genes_by_dispersion_svr(adata, layer_mat, n_top_genes) -> None:
    mean, var, dispersion = calc_mean_var_dispersion(layer_mat)
    highly_variable_mask = get_highly_variable_mask_by_dispersion_svr(adata, mean, var, n_top_genes)
    adata.var[DynamoAdataKeyManager.VAR_GENE_MEAN_KEY] = mean
    adata.var[DynamoAdataKeyManager.VAR_GENE_VAR_KEY] = var
    adata.var[DynamoAdataKeyManager.VAR_GENE_HIGHLAY_VARIABLE_KEY] = highly_variable_mask


def get_highly_variable_mask_by_dispersion_svr(mean, var, n_top_genes: int, svr_gamma: float = None):
    # normally, select svr_gamma based on #features
    if svr_gamma is None:
        svr_gamma = svr_gamma / len(mean)
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
