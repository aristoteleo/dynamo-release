import numpy as np
import pandas as pd
from scipy.sparse import issparse
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
    hii = np.diag(H)  # Diagonal values of hat matrix # fit.get_influence().hat_matrix_diag

    # Pearson residuals
    r = model.resid_pearson

    # Cook's distance (formula used by R = (res/(1 - hat))^2 * hat/(dispersion * p))
    # Note: dispersion is 1 since we aren't modeling overdispersion

    resid = good.disp - model.predict(good)
    rss = np.sum(resid ** 2)
    MSE = rss / (good.shape[0] - 2)
    # use the formula from: https://www.mathworks.com/help/stats/cooks-distance.html
    cooks_d = r**2 / (2 * MSE)  * hii / (1 - hii)**2 #(r / (1 - hii)) ** 2 *  / (1 * 2)

    return cooks_d


# ---------------------------------------------------------------------------------------------------
# preprocess utilities

def get_layer_keys(adata, layers='all', remove_normalized=True, include_protein=True):
    """Get the list of available layers' keys.
    """
    layer_keys = list(adata.layers.keys())
    if remove_normalized: layer_keys = [i for i in layer_keys if not i.startswith('X_')]

    if 'protein' in adata.obsm.keys() and include_protein:
        layer_keys.extend(['X', 'protein'])
    else:
        layer_keys.extend(['X'])
    layers = layer_keys if layers is 'all' else list(set(layer_keys).intersection(list(layers)))

    layers = list(set(layers).difference(['matrix', 'ambiguous', 'spanning']))
    return layers


def get_shared_counts(adata, layers, min_shared_count, type='gene'):
    layers = list(set(layers).difference(['X', 'matrix', 'ambiguous', 'spanning']))
    layers = np.array(layers)[~pd.DataFrame(layers)[0].str.startswith('X_').values]

    _nonzeros = reduce(lambda a, b: (adata.layers[a] > 0).multiply(adata.layers[b] > 0), layers) if \
        issparse(adata.layers[layers[0]]) else \
        reduce(lambda a, b: (adata.layers[a] > 0) * (adata.layers[b] > 0), layers)

    _sum = reduce(lambda a, b: _nonzeros.multiply(adata.layers[a]) + _nonzeros.multiply(adata.layers[b]), layers) if \
        issparse(adata.layers[layers[0]]) else \
        reduce(lambda a, b: np.multiply(_nonzeros, adata.layers[a]) + np.multiply(_nonzeros, adata.layers[b]), layers)

    if type == 'gene':
        return np.array(_sum.sum(0).A1 >= min_shared_count) if issparse(adata.layers[layers[0]]) else np.array(_sum.sum(0) >= min_shared_count)
    if type == 'cells':
        return np.array(_sum.sum(1).A1 >= min_shared_count) if issparse(adata.layers[layers[0]]) else np.array(_sum.sum(1) >= min_shared_count)


def clusters_stats(U, S, clusters_uid, cluster_ix, size_limit=40):
    """Calculate the averages per cluster

    If the cluster is too small (size<size_limit) the average of the toal is reported instead
    This function is taken from velocyto in order to reproduce velocyto's DentateGyrus notebook.
    """
    U_avgs = np.zeros((S.shape[0], len(clusters_uid)))
    S_avgs = np.zeros((S.shape[0], len(clusters_uid)))
    avgU_div_avgS = np.zeros((S.shape[0], len(clusters_uid)))
    slopes_by_clust = np.zeros((S.shape[0], len(clusters_uid)))

    for i, uid in enumerate(clusters_uid):
        cluster_filter = cluster_ix == i
        n_cells = np.sum(cluster_filter)
        if n_cells > size_limit:
            U_avgs[:, i], S_avgs[:, i] = U[:, cluster_filter].mean(1), S[:, cluster_filter].mean(1)
        else:
            U_avgs[:, i], S_avgs[:, i] = U.mean(1), S.mean(1)

    return U_avgs, S_avgs

def get_svr_filter(adata, layer='spliced', n_top_genes=3000):
    score_name = 'score' if layer in ['X', 'all'] else layer + '_score'
    valid_idx = np.where(np.isfinite(adata.var.loc[:, score_name]))[0]

    valid_table = adata.var.iloc[valid_idx, :]
    nth_score = np.sort(valid_table.loc[:, score_name])[::-1][n_top_genes]

    feature_gene_idx = np.where(valid_table.loc[:, score_name] >= nth_score)[0][:n_top_genes]

    adata.var.loc[:, 'use_for_dynamo'] = False
    adata.var.loc[adata.var.index[feature_gene_idx], 'use_for_dynamo'] = True

    filter_bool = np.zeros(adata.n_vars, dtype=bool)
    filter_bool[valid_idx[feature_gene_idx]] = True

    return filter_bool
# ---------------------------------------------------------------------------------------------------
# pca

def pca(adata, X, n_pca_components, pca_key):
    cm_genesums = X.sum(axis=0)
    valid_ind = (np.isfinite(cm_genesums)) + (cm_genesums != 0)
    valid_ind = np.array(valid_ind).flatten()
    CM = X[:, valid_ind]

    if adata.n_obs < 100000:
        fit = PCA(n_components=n_pca_components, svd_solver='arpack', random_state=0)
        X_pca = fit.fit_transform(CM.toarray()) if issparse(X) else fit.fit_transform(X)
        adata.obsm[pca_key] = X_pca
    else:
        fit = TruncatedSVD(n_components=n_pca_components + 1, random_state=0)  # unscaled PCA
        X_pca = fit.fit_transform(CM)[:, 1:]  # first columns is related to the total UMI (or library size)
        adata.obsm[pca_key] = X_pca

    return adata, fit, X_pca
