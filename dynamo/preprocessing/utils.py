import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
from functools import reduce
from sklearn.decomposition import PCA, TruncatedSVD


# ---------------------------------------------------------------------------------------------------
# implmentation of Cooks' distance (but this is for Poisson distribution fitting)

# https://stackoverflow.com/questions/47686227/poisson-regression-in-statsmodels-and-r

# from __future__ import division, print_function

# https://stats.stackexchange.com/questions/356053/the-identity-link-function-does-not-respect-the-domain-of-the-gamma-family
def _weight_matrix(fitted_model):
    """Calculates weight matrix in Poisson regression

    Parameters
    ----------
    fitted_model : statsmodel object
        Fitted Poisson model

    Returns
    -------
    W : 2d array-like
        Diagonal weight matrix in Poisson regression
    """
    return np.diag(fitted_model.fittedvalues)


def _hessian(X, W):
    """Hessian matrix calculated as -X'*W*X

    Parameters
    ----------
    X : 2d array-like
        Matrix of covariates

    W : 2d array-like
        Weight matrix

    Returns
    -------
    hessian : 2d array-like
        Hessian matrix
    """
    return -np.dot(X.T, np.dot(W, X))


def _hat_matrix(X, W):
    """Calculate hat matrix = W^(1/2) * X * (X'*W*X)^(-1) * X'*W^(1/2)

    Parameters
    ----------
    X : 2d array-like
        Matrix of covariates

    W : 2d array-like
        Diagonal weight matrix

    Returns
    -------
    hat : 2d array-like
        Hat matrix
    """
    # W^(1/2)
    Wsqrt = W ** (0.5)

    # (X'*W*X)^(-1)
    XtWX = -_hessian(X=X, W=W)
    XtWX_inv = np.linalg.inv(XtWX)

    # W^(1/2)*X
    WsqrtX = np.dot(Wsqrt, X)

    # X'*W^(1/2)
    XtWsqrt = np.dot(X.T, Wsqrt)

    return np.dot(WsqrtX, np.dot(XtWX_inv, XtWsqrt))


def cook_dist(model, X, good):
    # Weight matrix
    W = _weight_matrix(model)

    # Hat matrix
    H = _hat_matrix(X, W)
    hii = np.diag(
        H
    )  # Diagonal values of hat matrix # fit.get_influence().hat_matrix_diag

    # Pearson residuals
    r = model.resid_pearson

    # Cook's distance (formula used by R = (res/(1 - hat))^2 * hat/(dispersion * p))
    # Note: dispersion is 1 since we aren't modeling overdispersion

    resid = good.disp - model.predict(good)
    rss = np.sum(resid ** 2)
    MSE = rss / (good.shape[0] - 2)
    # use the formula from: https://www.mathworks.com/help/stats/cooks-distance.html
    cooks_d = (
        r ** 2 / (2 * MSE) * hii / (1 - hii) ** 2
    )  # (r / (1 - hii)) ** 2 *  / (1 * 2)

    return cooks_d


# ---------------------------------------------------------------------------------------------------
# preprocess utilities

def unique_var_obs_adata(adata):
    """Function to make the obs and var attribute's index unique"""
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    return adata


def merge_adata_attrs(adata_ori, adata, attr):
    if attr == 'var':
        _columns = set(adata.var.columns).difference(adata_ori.var.columns)
        var_df = adata_ori.var.merge(adata.var[_columns], how='left',
                                            left_index=True, right_index=True)
        adata_ori.var = var_df.loc[adata_ori.var.index, :]
    elif attr == 'obs':
        _columns = set(adata.obs.columns).difference(adata_ori.obs.columns)
        obs_df = adata_ori.obs.merge(adata.obs[_columns], how='left',
                                            left_index=True, right_index=True)
        adata_ori.obs = obs_df.loc[adata_ori.obs.index, :]

    return adata_ori


def allowed_layer_raw_names():
    only_splicing = ['spliced', 'unspliced']
    only_labeling = ['new', 'total']
    splicing_and_labeling = ['uu', 'ul', 'su', 'sl']

    return only_splicing, only_labeling, splicing_and_labeling


def allowed_X_layer_names():
    only_splicing = ['X_spliced', 'X_unspliced']
    only_labeling = ['X_new', 'X_total']
    splicing_and_labeling = ['X_uu', 'X_ul', 'X_su', 'X_sl']

    return only_splicing, only_labeling, splicing_and_labeling


def get_layer_keys(adata, layers="all", remove_normalized=True, include_protein=True):
    """Get the list of available layers' keys.
    """
    layer_keys = list(adata.layers.keys())
    if remove_normalized:
        layer_keys = [i for i in layer_keys if not i.startswith("X_")]

    if "protein" in adata.obsm.keys() and include_protein:
        layer_keys.extend(["X", "protein"])
    else:
        layer_keys.extend(["X"])
    layers = (
        layer_keys
        if layers is "all"
        else list(set(layer_keys).intersection(list(layers)))
    )

    layers = list(set(layers).difference(["matrix", "ambiguous", "spanning"]))
    return layers


def get_shared_counts(adata, layers, min_shared_count, type="gene"):
    layers = list(set(layers).difference(["X", "matrix", "ambiguous", "spanning"]))
    layers = np.array(layers)[~pd.DataFrame(layers)[0].str.startswith("X_").values]

    _nonzeros, _sum = None, None
    for layer in layers:
        if issparse(adata.layers[layers[0]]):
            _nonzeros = (
                adata.layers[layer] > 0
                if _nonzeros is None
                else _nonzeros.multiply(adata.layers[layer] > 0)
            )
        else:
            _nonzeros = (
                adata.layers[layer] > 0
                if _nonzeros is None
                else _nonzeros * (adata.layers[layer] > 0)
            )

    for layer in layers:
        if issparse(adata.layers[layers[0]]):
            _sum = (
                _nonzeros.multiply(adata.layers[layer])
                if _sum is None
                else _sum + _nonzeros.multiply(adata.layers[layer])
            )
        else:
            _sum = (
                np.multiply(_nonzeros, adata.layers[layer])
                if _sum is None
                else _sum + np.multiply(_nonzeros, adata.layers[layer])
            )

    if type == "gene":
        return (
            np.array(_sum.sum(0).A1 >= min_shared_count)
            if issparse(adata.layers[layers[0]])
            else np.array(_sum.sum(0) >= min_shared_count)
        )
    if type == "cells":
        return (
            np.array(_sum.sum(1).A1 >= min_shared_count)
            if issparse(adata.layers[layers[0]])
            else np.array(_sum.sum(1) >= min_shared_count)
        )


def clusters_stats(U, S, clusters_uid, cluster_ix, size_limit=40):
    """Calculate the averages per cluster

    If the cluster is too small (size<size_limit) the average of the toal is reported instead
    This function is modified from velocyto in order to reproduce velocyto's DentateGyrus notebook.
    """
    U_avgs = np.zeros((S.shape[1], len(clusters_uid)))
    S_avgs = np.zeros((S.shape[1], len(clusters_uid)))
    avgU_div_avgS = np.zeros((S.shape[1], len(clusters_uid)))
    slopes_by_clust = np.zeros((S.shape[1], len(clusters_uid)))

    for i, uid in enumerate(clusters_uid):
        cluster_filter = cluster_ix == i
        n_cells = np.sum(cluster_filter)
        if n_cells > size_limit:
            U_avgs[:, i], S_avgs[:, i] = (
                U[cluster_filter, :].mean(0),
                S[cluster_filter, :].mean(0),
            )
        else:
            U_avgs[:, i], S_avgs[:, i] = U.mean(0), S.mean(0)

    return U_avgs, S_avgs


def get_svr_filter(adata, layer="spliced", n_top_genes=3000, return_adata=False):
    score_name = "score" if layer in ["X", "all"] else layer + "_score"
    valid_idx = np.where(np.isfinite(adata.var.loc[:, score_name]))[0]

    valid_table = adata.var.iloc[valid_idx, :]
    nth_score = np.sort(valid_table.loc[:, score_name])[::-1][
        np.min((n_top_genes - 1, valid_table.shape[0] - 1))
    ]

    feature_gene_idx = np.where(valid_table.loc[:, score_name] >= nth_score)[0][
        :n_top_genes
    ]
    feature_gene_idx = valid_idx[feature_gene_idx]

    if return_adata:
        adata.var.loc[:, "use_for_dynamo"] = False
        adata.var.loc[adata.var.index[feature_gene_idx], "use_for_dynamo"] = True
        res = adata
    else:
        filter_bool = np.zeros(adata.n_vars, dtype=bool)
        filter_bool[feature_gene_idx] = True
        res = filter_bool

    return res

def sz_util(adata, layer, round_exprs, method, locfunc, total_layers=None):
    adata = adata.copy()

    if layer == '_total_' and '_total_' not in adata.layers.keys():
        if total_layers is not None:
            if not isinstance(total_layers, list): total_layers = [total_layers]
            if len(set(total_layers).difference(adata.layers.keys())) == 0:
                total = None
                for t_key in total_layers:
                    total = (
                        adata.layers[t_key] if total is None else total + adata.layers[t_key]
                    )
                adata.layers["_total_"] = total

    if layer is "raw":
        CM = adata.raw.X
    elif layer is "X":
        CM = adata.X
    elif layer is "protein":
        if "protein" in adata.obsm_keys():
            CM = adata.obsm["protein"]
        else:
            return None, None
    else:
        CM = adata.layers[layer]

    if round_exprs:
        if issparse(CM):
            CM.data = np.round(CM.data, 0)
        else:
            CM = CM.round().astype("int")

    cell_total = CM.sum(axis=1).A1 if issparse(CM) else CM.sum(axis=1)
    cell_total += cell_total == 0  # avoid infinity value after log (0)

    if method == "mean-geometric-mean-total":
        sfs = cell_total / np.exp(locfunc(np.log(cell_total)))
    elif method == "median":
        sfs = cell_total / np.nanmedian(cell_total)
    elif method == "mean":
        sfs = cell_total / np.nanmean(cell_total)
    else:
        raise NotImplementedError(f"This method {method} is not supported!")

    return sfs, cell_total

def get_sz_exprs(adata, layer, total_szfactor=None):
    if layer is "raw":
        CM = adata.raw.X
        szfactors = adata.obs[layer + "Size_Factor"][:, None]
    elif layer is "X":
        CM = adata.X
        szfactors = adata.obs["Size_Factor"][:, None]
    elif layer is "protein":
        if "protein" in adata.obsm_keys():
            CM = adata.obsm[layer]
            szfactors = adata.obs["protein_Size_Factor"][:, None]
        else:
            CM, szfactors = None, None
    else:
        CM = adata.layers[layer]
        szfactors = adata.obs[layer + "_Size_Factor"][:, None]

    if total_szfactor is not None and total_szfactor in adata.obs.keys():
        szfactors = adata.obs[total_szfactor][:, None]

    return szfactors, CM

def normalize_util(CM, szfactors, relative_expr, pseudo_expr, norm_method=np.log):
    if relative_expr:
        CM = (
            CM.multiply(csr_matrix(1 / szfactors))
            if issparse(CM)
            else CM / szfactors
        )

    if pseudo_expr is None:
        pseudo_expr = 1
    if issparse(CM):
        CM.data = (
            norm_method(CM.data + pseudo_expr)
            if norm_method is not None
            else CM.data
        )
    else:
        CM = (
            norm_method(CM + pseudo_expr)
            if norm_method is not None
            else CM
        )

    return CM

# ---------------------------------------------------------------------------------------------------
# pca


def pca(adata, CM, n_pca_components=30, pca_key='X'):

    if adata.n_obs < 100000:
        pca = PCA(n_components=min(n_pca_components, CM.shape[1] - 1), svd_solver="arpack", random_state=0)
        fit = pca.fit(CM.toarray()) if issparse(CM) else pca.fit(CM)
        X_pca = fit.transform(CM.toarray()) if issparse(CM) else fit.transform(CM)
        adata.obsm[pca_key] = X_pca
        adata.uns["PCs"] = fit.components_.T

        adata.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_
    else:
        fit = TruncatedSVD(
            n_components=min(n_pca_components + 1, CM.shape[1] - 1), random_state=0
        )  # unscaled PCA
        X_pca = fit.fit_transform(CM)[
            :, 1:
        ]  # first columns is related to the total UMI (or library size)
        adata.obsm[pca_key] = X_pca
        adata.uns["PCs"] = fit.components_.T

        adata.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_[1:]

    return adata, fit, X_pca

# ---------------------------------------------------------------------------------------------------
# labeling related


def collapse_adata(adata):
    """Function to collapse the four species data, will be generalized to handle dual-datasets"""
    only_splicing, only_labeling, splicing_and_labeling = allowed_layer_raw_names()

    if np.all([i in adata.layers.keys() for i in splicing_and_labeling]):
        adata.layers[only_splicing[0]] = adata.layers['su'] + adata.layers['sl']
        adata.layers[only_splicing[1]] = adata.layers['uu'] + adata.layers['ul']
        adata.layers[only_labeling[0]] = adata.layers['ul'] + adata.layers['sl']
        adata.layers[only_labeling[1]] = adata.layers[only_labeling[0]] + adata.layers['uu'] + adata.layers['su']

    return adata

def detect_datatype(adata):
    has_splicing, has_labeling, has_protein = False, False, False

    layers_set = set(adata.layers.keys())
    if len(layers_set.difference(['ul', 'sl', 'uu', 'su'])) == 0:
        has_splicing, has_labeling = True, True
    elif len(layers_set.difference(['unspliced', 'spliced'])) == 0:
        has_splicing = True
    elif len(layers_set.difference(['new', 'total'])) == 0:
        has_labeling = True

    if "protein" in adata.obsm.keys():
        has_protein = True

    return has_splicing, has_labeling, has_protein


def default_layer(adata):
    has_splicing, has_labeling, _ = detect_datatype(adata)

    if has_splicing:
        if has_labeling:
            if len(set(adata.layers.keys()).intersection(['new', 'total', 'spliced', 'unspliced'])) == 4:
                adata = collapse_adata(adata)
            default_layer = "M_t" if "M_t" in adata.layers.keys() else "X_total" if \
                "X_total" in adata.layers.keys() else "total"
        else:
            default_layer = "M_s" if "M_s" in adata.layers.keys() else "X_spliced" if \
                "X_spliced" in adata.layers.keys() else "spliced"
    else:
        default_layer = "M_t" if "M_t" in adata.layers.keys() else "X_total" if \
            "X_total" in adata.layers.keys() else "total"

    return default_layer

def NTR(adata):
    """calculate the new to total ratio across cells. Note that
    NTR for the first time point in degradation approximates gamma/beta."""

    if len(set(['new', 'total']).intersection(adata.layers.keys())) == 2:
        ntr = adata.layers['new'].sum(1) / adata.layers['total'].sum(1)
        ntr = ntr.A1 if issparse(adata.layers['new']) else ntr
    elif len(set(['uu', 'ul', 'su', 'sl']).intersection(adata.layers.keys())) == 4:
        new = adata.layers['ul'].sum(1) + \
              adata.layers['sl'].sum(1)
        total = new + adata.layers['uu'].sum(1) + \
                adata.layers['su'].sum(1)
        ntr = new / total

        ntr = ntr.A1 if issparse(adata.layers['uu']) else ntr
    else:
        ntr = None

    return ntr
