# =================================================================
# Original Code Repository Author: Lause, Berens & Kobak
# Adapted to Dynamo by: dynamo authors
# Created Date: 12/16/2021
# Description: pearson residuals based method for preprocessing single cell expression data
# Original Code Repository: https://github.com/atarashansky/SCTransformPy
# Sctransform Paper: Hafemeister, C., Satija, R. Normalization and variance stabilization of single-cell RNA-seq data using regularized negative binomial regression.
# =================================================================

import os
from multiprocessing import Manager, Pool

import numpy as np
import pandas as pd
import scipy
import scipy as sp
import statsmodels.discrete.discrete_model
import statsmodels.nonparametric.kernel_regression
from anndata import AnnData
from KDEpy import FFTKDE
from scipy import stats

from ..configuration import DKM
from ..dynamo_logger import main_info, main_info_insert_adata_layer

_EPS = np.finfo(float).eps


def robust_scale_binned(y, x, breaks):
    bins = np.digitize(x, breaks)
    binsu = np.unique(bins)
    res = np.zeros(bins.size)
    for i in range(binsu.size):
        yb = y[bins == binsu[i]]
        res[bins == binsu[i]] = (yb - np.median(yb)) / (1.4826 * np.median(np.abs(yb - np.median(yb))) + _EPS)

    return res


def is_outlier(y, x, th=10):
    z = FFTKDE(kernel="gaussian", bw="ISJ").fit(x)
    z.evaluate()
    bin_width = (max(x) - min(x)) * z.bw / 2
    eps = _EPS * 10

    breaks1 = np.arange(min(x), max(x) + bin_width, bin_width)
    breaks2 = np.arange(min(x) - eps - bin_width / 2, max(x) + bin_width, bin_width)
    score1 = robust_scale_binned(y, x, breaks1)
    score2 = robust_scale_binned(y, x, breaks2)
    return np.abs(np.vstack((score1, score2))).min(0) > th


def _parallel_init(igenes_bin_regress, iumi_bin, ign, imm, ips):
    global genes_bin_regress
    global umi_bin
    global gn
    global mm
    global ps
    genes_bin_regress = igenes_bin_regress
    umi_bin = iumi_bin
    gn = ign
    mm = imm
    ps = ips


def _parallel_wrapper(j):
    name = gn[genes_bin_regress[j]]
    y = umi_bin[:, j].A.flatten()
    pr = statsmodels.discrete.discrete_model.Poisson(y, mm)
    res = pr.fit(disp=False)
    mu = res.predict()
    theta = theta_ml(y, mu)
    ps[name] = np.append(res.params, theta)


def gmean(X: scipy.sparse.spmatrix, axis=0, eps=1):
    X = X.copy()
    X = X.asfptype()
    assert np.all(X.sum(0) > 0)
    assert np.all(X.data > 0)
    X.data[:] = np.log(X.data + eps)
    res = np.exp(X.mean(axis).A.flatten()) - eps

    assert np.all(res > 0)
    return res


def theta_ml(y, mu):
    n = y.size
    weights = np.ones(n)
    limit = 10
    eps = (_EPS) ** 0.25

    from scipy.special import polygamma, psi

    def score(n, th, mu, y, w):
        return sum(w * (psi(th + y) - psi(th) + np.log(th) + 1 - np.log(th + mu) - (y + th) / (mu + th)))

    def info(n, th, mu, y, w):
        return sum(w * (-polygamma(1, th + y) + polygamma(1, th) - 1 / th + 2 / (mu + th) - (y + th) / (mu + th) ** 2))

    t0 = n / sum(weights * (y / mu - 1) ** 2)
    it = 0
    de = 1

    while it + 1 < limit and abs(de) > eps:
        it += 1
        t0 = abs(t0)
        i = info(n, t0, mu, y, weights)
        de = score(n, t0, mu, y, weights) / i
        t0 += de
    t0 = max(t0, 0)

    return t0


def sctransform_core(
    adata,
    layer=DKM.X_LAYER,
    min_cells=5,
    gmean_eps=1,
    n_genes=2000,
    n_cells=None,
    bin_size=500,
    bw_adjust=3,
    inplace=True,
):
    """
    A re-implementation of SCTransform from the Satija lab.
    """
    main_info("sctransform adata on layer: %s" % (layer))
    X = DKM.select_layer_data(adata, layer).copy()
    X = sp.sparse.csr_matrix(X)
    X.eliminate_zeros()
    gene_names = np.array(list(adata.var_names))
    cell_names = np.array(list(adata.obs_names))
    genes_cell_count = X.sum(0).A.flatten()
    genes = np.where(genes_cell_count >= min_cells)[0]
    genes_ix = genes.copy()

    X = X[:, genes]
    Xraw = X.copy()
    gene_names = gene_names[genes]
    genes = np.arange(X.shape[1])
    genes_cell_count = X.sum(0).A.flatten()

    genes_log_gmean = np.log10(gmean(X, axis=0, eps=gmean_eps))

    # sample by n_cells, or use all cells
    if n_cells is not None and n_cells < X.shape[0]:
        cells_step1 = np.sort(np.random.choice(X.shape[0], replace=False, size=n_cells))
        genes_cell_count_step1 = X[cells_step1].sum(0).A.flatten()
        genes_step1 = np.where(genes_cell_count_step1 >= min_cells)[0]
        genes_log_gmean_step1 = np.log10(gmean(X[cells_step1][:, genes_step1], axis=0, eps=gmean_eps))
    else:
        cells_step1 = np.arange(X.shape[0])
        genes_step1 = genes
        genes_log_gmean_step1 = genes_log_gmean

    umi = X.sum(1).A.flatten()
    log_umi = np.log10(umi)
    X2 = X.copy()
    X2.data[:] = 1
    gene = X2.sum(1).A.flatten()
    log_gene = np.log10(gene)
    umi_per_gene = umi / gene
    log_umi_per_gene = np.log10(umi_per_gene)

    cell_attrs = pd.DataFrame(
        index=cell_names,
        data=np.vstack((umi, log_umi, gene, log_gene, umi_per_gene, log_umi_per_gene)).T,
        columns=["umi", "log_umi", "gene", "log_gene", "umi_per_gene", "log_umi_per_gene"],
    )

    data_step1 = cell_attrs.iloc[cells_step1]

    if n_genes is not None and n_genes < len(genes_step1):
        log_gmean_dens = stats.gaussian_kde(genes_log_gmean_step1, bw_method="scott")
        xlo = np.linspace(genes_log_gmean_step1.min(), genes_log_gmean_step1.max(), 512)
        ylo = log_gmean_dens.evaluate(xlo)
        xolo = genes_log_gmean_step1
        sampling_prob = 1 / (np.interp(xolo, xlo, ylo) + _EPS)
        genes_step1 = np.sort(
            np.random.choice(genes_step1, size=n_genes, p=sampling_prob / sampling_prob.sum(), replace=False)
        )
        genes_log_gmean_step1 = np.log10(gmean(X[cells_step1, :][:, genes_step1], eps=gmean_eps))

    bin_ind = np.ceil(np.arange(1, genes_step1.size + 1) / bin_size)
    max_bin = max(bin_ind)

    ps = Manager().dict()

    for i in range(1, int(max_bin) + 1):
        genes_bin_regress = genes_step1[bin_ind == i]
        umi_bin = X[cells_step1, :][:, genes_bin_regress]

        mm = np.vstack((np.ones(data_step1.shape[0]), data_step1["log_umi"].values.flatten())).T

        pc_chunksize = umi_bin.shape[1] // os.cpu_count() + 1
        pool = Pool(os.cpu_count(), _parallel_init, [genes_bin_regress, umi_bin, gene_names, mm, ps])
        try:
            pool.map(_parallel_wrapper, range(umi_bin.shape[1]), chunksize=pc_chunksize)
        finally:
            pool.close()
            pool.join()

    ps = ps._getvalue()

    model_pars = pd.DataFrame(
        data=np.vstack([ps[x] for x in gene_names[genes_step1]]),
        columns=["Intercept", "log_umi", "theta"],
        index=gene_names[genes_step1],
    )

    min_theta = 1e-7
    x = model_pars["theta"].values.copy()
    x[x < min_theta] = min_theta
    model_pars["theta"] = x
    dispersion_par = np.log10(1 + 10 ** genes_log_gmean_step1 / model_pars["theta"].values.flatten())

    model_pars_theta = model_pars["theta"]
    model_pars = model_pars.iloc[:, model_pars.columns != "theta"].copy()
    model_pars["dispersion"] = dispersion_par

    outliers = (
        np.vstack(
            ([is_outlier(model_pars.values[:, i], genes_log_gmean_step1) for i in range(model_pars.shape[1])])
        ).sum(0)
        > 0
    )

    filt = np.invert(outliers)
    model_pars = model_pars[filt]
    genes_step1 = genes_step1[filt]
    genes_log_gmean_step1 = genes_log_gmean_step1[filt]

    z = FFTKDE(kernel="gaussian", bw="ISJ").fit(genes_log_gmean_step1)
    z.evaluate()
    bw = z.bw * bw_adjust

    x_points = np.vstack((genes_log_gmean, np.array([min(genes_log_gmean_step1)] * genes_log_gmean.size))).max(0)
    x_points = np.vstack((x_points, np.array([max(genes_log_gmean_step1)] * genes_log_gmean.size))).min(0)

    full_model_pars = pd.DataFrame(
        data=np.zeros((x_points.size, model_pars.shape[1])), index=gene_names, columns=model_pars.columns
    )
    for i in model_pars.columns:
        kr = statsmodels.nonparametric.kernel_regression.KernelReg(
            model_pars[i].values, genes_log_gmean_step1[:, None], ["c"], reg_type="ll", bw=[bw]
        )
        full_model_pars[i] = kr.fit(data_predict=x_points)[0]

    theta = 10 ** genes_log_gmean / (10 ** full_model_pars["dispersion"].values - 1)
    full_model_pars["theta"] = theta
    del full_model_pars["dispersion"]

    model_pars_outliers = outliers

    regressor_data = np.vstack((np.ones(cell_attrs.shape[0]), cell_attrs["log_umi"].values)).T

    d = X.data
    x, y = X.nonzero()
    mud = np.exp(full_model_pars.values[:, 0][y] + full_model_pars.values[:, 1][y] * cell_attrs["log_umi"].values[x])
    vard = mud + mud ** 2 / full_model_pars["theta"].values.flatten()[y]

    X.data[:] = (d - mud) / vard ** 0.5
    X.data[X.data < 0] = 0
    X.eliminate_zeros()

    clip = np.sqrt(X.shape[0] / 30)
    X.data[X.data > clip] = clip

    if inplace:
        adata.raw = adata.copy()

        d = dict(zip(np.arange(X.shape[1]), genes_ix))
        x, y = X.nonzero()
        y = np.array([d[i] for i in y])
        data = X.data
        Xnew = sp.sparse.coo_matrix((data, (x, y)), shape=adata.shape).tocsr()
        if layer == DKM.X_LAYER:
            main_info("set sctransform results to adata.X", indent_level=2)
            DKM.set_layer_data(adata, layer, Xnew)  # TODO: add log1p of corrected umi counts to layers
        else:
            new_X_layer = DKM.gen_layer_X_key(layer)
            main_info_insert_adata_layer(new_X_layer, indent_level=2)
            DKM.set_layer_data(adata, new_X_layer, Xnew)  # TODO: add log1p of corrected umi counts to layers

        # TODO: reformat the following output according to adata key standards in dyn.
        for c in full_model_pars.columns:
            adata.var[c + "_sct"] = full_model_pars[c]

        for c in cell_attrs.columns:
            adata.obs[c + "_sct"] = cell_attrs[c]

        for c in model_pars.columns:
            adata.var[c + "_step1_sct"] = model_pars[c]
        adata.var["model_pars_theta_step1"] = model_pars_theta

        z = pd.Series(index=gene_names, data=np.zeros(gene_names.size, dtype="int"))
        z[gene_names[genes_step1]] = 1

        w = pd.Series(index=gene_names, data=np.zeros(gene_names.size, dtype="int"))
        w[gene_names] = genes_log_gmean
        adata.var["genes_step1_sct"] = z
        adata.var["log10_gmean_sct"] = w

    else:
        adata_new = AnnData(X=X)
        adata_new.var_names = pd.Index(gene_names)
        adata_new.obs_names = adata.obs_names
        adata_new.raw = adata.copy()

        for c in full_model_pars.columns:
            adata_new.var[c + "_sct"] = full_model_pars[c]

        for c in cell_attrs.columns:
            adata_new.obs[c + "_sct"] = cell_attrs[c]

        for c in model_pars.columns:
            adata_new.var[c + "_step1_sct"] = model_pars[c]

        z = pd.Series(index=gene_names, data=np.zeros(gene_names.size, dtype="int"))
        z[gene_names[genes_step1]] = 1
        adata_new.var["genes_step1_sct"] = z
        adata_new.var["log10_gmean_sct"] = genes_log_gmean
        return adata_new


def sctransform(adata: AnnData, layers: str = [DKM.X_LAYER], output_layer: str = None, n_top_genes=2000, **kwargs):
    """a wrapper calls sctransform_core and set dynamo style keys in adata"""
    for layer in layers:
        sctransform_core(adata, layer=layer, n_genes=n_top_genes, **kwargs)
