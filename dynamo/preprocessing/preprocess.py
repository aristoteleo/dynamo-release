import numpy as np
import pandas as pd
import scipy
from scipy.sparse import issparse
# from anndata import AnnData
import warnings
# import statsmodels.api as sm
from sklearn.decomposition import TruncatedSVD, FastICA


def szFactor(adata, layers='all', locfunc=np.nanmean, round_exprs=True, method='mean-geometric-mean-total'):
    """Calculate the size factor of the each cell for a AnnData object.
    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        layers: str (default: all)
            The layer(s) to be normalized. Default is all, including RNA (X, raw) or spliced, unspliced, protein, etc.
        locfunc: `function`
            The function to normalize the data
        round_exprs: `bool`
            A logic flag to determine whether the gene expression should be rounded into integers.
        method: `str`
            The method used to calculate the expected total reads / UMI. Only mean-geometric-mean-total is supported.

    Returns
    -------
        adata: :AnnData
            A updated anndata object that are updated with the Size_Factor in the obs slot.
    """

    layer_keys = list(adata.layers.keys())
    layer_keys.extend('X')
    layers = layer_keys if layers is 'all' else list(set(layer_keys).intersection(layers))

    for layer in layers:
        if layer is 'raw' or layer is 'X':
            CM = adata.raw if adata.raw is not None else adata.X
        else:
            CM = adata.layers[layer]

        if round_exprs:
            CM = CM.astype('int') # will this affect downstream analysis?

        if method == 'mean-geometric-mean-total':
            cell_total = CM.sum(axis=1)
            sfs = cell_total / np.exp(locfunc(np.log(cell_total)))
        else:
            print('This method is supported!')

        sfs[~np.isfinite(sfs)] = 1
        if layer is 'raw' or layer is 'X':
            adata.obs['Size_Factor'] = sfs
        else:
            adata.obs[layer + '_Size_Factor'] = sfs

    return adata


def normalize_expr_data(adata, layers='all', norm_method='log', pseudo_expr=1, relative_expr=True):
    """Normalize the gene expression value for the AnnData object
    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        layers: str (default: all)
            The layer(s) to be normalized. Default is all, including RNA (X, raw) or spliced, unspliced, protein, etc.
        norm_method: `str`
            The method used to normalize data.
        pseudo_expr: `int`
            A pseudocount added to the gene expression value before log2 normalization.
        relative_expr: `bool`
            A logic flag to determine whether we need to divide gene expression values first by size factor before normalization

    Returns
    -------
        adata: :AnnData
            A updated anndata object that are updated with normalized expression values, X.
    """

    layer_keys = list(adata.layers.keys())
    layer_keys.extend('X')
    layers = layer_keys if layers is 'all' else list(set(layer_keys).intersection(layers))

    layer_sz_column_names = [i + '_Size_Factor' for i in set(layers).difference('X')]
    layer_sz_column_names.extend(['Size_Factor'])
    layers_to_sz = set(layer_sz_column_names).difference(adata.obs.keys())

    if len(layers_to_sz) > 0:
        layers = pd.Series(layers_to_sz).str.split('_Size_Factor', expand=True).iloc[:, 0].tolist()
        szFactor(adata, layers=layers, locfunc=np.nanmean, round_exprs=True, method='mean-geometric-mean-total')

    for layer in layers:
        if layer is 'raw' or layer is 'X':
            FM = adata.raw if adata.raw is not None else adata.X
            szfactors = adata.obs['Size_Factor'][:, None]
        else:
            FM = adata.layers[layer]
            szfactors = adata.obs[layer + 'Size_Factor'][:, None]

        if 'use_for_ordering' in adata.var.columns:
            FM = FM[adata.var['use_for_ordering'], :]

        if norm_method == 'log' and layer is not 'protein':
            if relative_expr:
                FM = scipy.sparse.diags((1/szfactors).flatten().tolist(), 0).dot(FM) if issparse(FM) else FM / szfactors

            if pseudo_expr is None:
                pseudo_expr = 1

            if issparse(FM):
                FM.data = np.log2(FM.data + pseudo_expr)
            else:
                FM = np.log2(FM + pseudo_expr)

        elif layer is 'protein': # norm_method == 'clr':
            if norm_method is not 'clr':
                warnings.warn('For protein data, log transformation is not recommended. Using clr normalization by default.')
            """This normalization implements the centered log-ratio (CLR) normalization from Seurat which is computed for
            each gene (M Stoeckius, ‎2017).
            """
            FM = FM.T
            n_feature = FM.shape[1]

            for i in range(FM.shape[0]):
                x = FM[i].A
                res = np.log1p(x / (np.exp(np.nansum(np.log1p(x[x > 0])) / n_feature)))
                res[np.isnan(res)] = 0
                # res[res > 100] = 100
                FM[i] = res

            FM = FM.T
        else:
            warnings.warn(norm_method + ' is not implemented yet')

        if layer in ['raw', 'X']:
            adata.X = FM
        else:
            adata.layers['X_' + layer] = FM

    return adata


def recipe_monocle(adata, layer=None, method='PCA', num_dim=50, norm_method='log', pseudo_expr=1,
                   relative_expr=True, scaling=True, **kwargs):
    """
    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        layers: str (default: None)
            The layer(s) to be normalized. Default is all, including RNA (X, raw) or spliced, unspliced, protein, etc.
        method: `str`
            The linear dimension reduction methods to be used.
        num_dim: `int`
            The number of dimensions reduced to.
        norm_method: `str`
            The method to normalize the data.
        pseudo_expr: `int`
            A pseudocount added to the gene expression value before log2 normalization.
        relative_expr: `bool`
            A logic flag to determine whether we need to divide gene expression values first by size factor before normalization
        scaling: `str`
            A logic flag to determine whether we should scale the data before performing linear dimension reduction method.
        kwargs:
            Other Parameters passed into the function.

    Returns
    -------
        adata: :AnnData
            A updated anndata object that are updated with Size_Factor, normalized expression values, X and reduced dimensions.
    """

    adata = szFactor(adata)

    adata = normalize_expr_data(adata, norm_method=norm_method, pseudo_expr=pseudo_expr, relative_expr=relative_expr)

    if layer is None:
        FM = adata.X if 'spliced' not in adata.layers.keys() else adata.layers['spliced']

    fm_genesums = FM.sum(axis=0)
    valid_ind = (np.isfinite(fm_genesums)) + (fm_genesums != 0)
    valid_ind = np.array(valid_ind).flatten()
    FM = FM[:, valid_ind]

    adata = adata[:, valid_ind]

    if method is 'PCA':
        clf = TruncatedSVD(num_dim, random_state=2019)
        reduce_dim = clf.fit_transform(FM)
        adata.uns['explained_variance_ratio'] = clf.explained_variance_ratio_
    elif method == 'ICA':
        ICA=FastICA(num_dim,
                algorithm='deflation', tol=5e-6, fun='logcosh', max_iter=1000)
        reduce_dim=ICA.fit_transform(FM.toarray())

    adata.obsm['X_' + method.lower()] = reduce_dim

    return adata


def gini(adata):
    """Calculate the Gini coefficient of a numpy array.

    https://github.com/thomasmaxwellnorman/perturbseq_demo/blob/master/perturbseq/util.py
    """

    # From: https://github.com/oliviaguest/gini
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = adata.X.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 # np.min(array[array!=0]) #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0] #number of array elements
    gini = ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

    adata.var['gini'] = gini


def Dispersion(adata, modelFormulaStr="~ 1", relative_expr=True, min_cells_detected=1, removeOutliers=True):
    """
    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
    adata
    modelFormulaStr
    relative_expr
    min_cells_detected
    removeOutliers

    Returns
    -------

    """
    import re

    mu = None
    model_terms = [x.strip() for x in re.compile('~|\\*|\\+').split(modelFormulaStr)]
    model_terms = list(set(model_terms) - set(['']))

    cds_pdata = adata.obs  # .loc[:, model_terms]
    cds_pdata['rowname'] = cds_pdata.index.values
    disp_table = disp_calc_helper_NB(adata[:, :], min_cells_detected)
    # disp_table['disp'] = np.random.uniform(0, 10, 11)
    # disp_table = cds_pdata.apply(disp_calc_helper_NB(adata[:, :], min_cells_detected))

    # cds_pdata <- dplyr::group_by_(dplyr::select_(rownames_to_column(pData(cds)), "rowname", .dots=model_terms), .dots=model_terms)
    # disp_table <- as.data.frame(cds_pdata %>% do(disp_calc_helper_NB(cds[,.$rowname], cds@expressionFamily, min_cells_detected)))

    if disp_table is None:
        raise Exception('Parametric dispersion fitting failed, please set a different lowerDetectionLimit')

    disp_table = disp_table.loc[np.where(disp_table['mu'] != np.nan)[0], :]

    res = parametricDispersionFit(disp_table)
    fit, coefs = res[0], res[1]

    if removeOutliers:
        # influence = fit.get_influence()
        # #CD is the distance and p is p-value
        # (CD, p) = influence.cooks_distance

        CD = cook_dist(fit, 1 / disp_table['mu'][:, None])
        cooksCutoff = 4 / disp_table.shape[0]
        print("Removing ", len(CD[CD > cooksCutoff]), " outliers")
        outliers = CD > cooksCutoff
        # use CD.index.values? remove genes that lost when doing parameter fitting
        lost_gene = set(disp_table.index.values).difference(set(range(len(CD))))
        outliers[lost_gene] = False
        res = parametricDispersionFit(disp_table.loc[~ outliers, :])

        fit, coefs = res[0], res[1]

    def ans(q):
        coefs[0] + coefs[1] / q

    adata.uns['dispFitinfo'] = disp_table, ans
    return adata


def parametricDispersionFit(disp_table, initial_coefs=np.array([1e-6, 1])):
    """

    Parameters
    ----------
    disp_table
    initial_coefs

    Returns
    -------

    """
    coefs = initial_coefs
    iter = 0
    while True:
        residuals = disp_table['disp'] / (coefs[0] + coefs[1] / disp_table['mu'])
        good = disp_table.loc[(residuals > initial_coefs[0]) & (residuals < 10000), :]
        fit = sm.formula.glm("disp ~ I(1 / mu)", data=good,
                             family=sm.families.Gamma(link=sm.genmod.families.links.identity)).fit(start_params=coefs)

        oldcoefs = coefs
        coefs = fit.params

        if (coefs[0] < initial_coefs[0]):
            coefs[0] = initial_coefs[0]
        else:
            raise (
                "Parametric dispersion fit failed. Try a local fit and/or a pooled estimation. (See '?estimateDispersions')")

        if (np.sum(np.log(coefs / oldcoefs) ** 2 < coefs[0])):
            break
        iter += 1

        if (iter > 10):
            warning("Dispersion fit didn't converge")
            break

    return fit, coefs


def disp_calc_helper_NB(adata, min_cells_detected):
    """

    Parameters
    ----------
    adata
    min_cells_detected

    Returns
    -------

    """
    rounded = adata.raw.astype('int') if adata.raw is not None else adata.X
    lowerDetectedLimit = adata.uns['lowerDetectedLimit'] if 'lowerDetectedLimit' in adata.uns.keys() else 1
    nzGenes = (rounded > lowerDetectedLimit).sum(axis=0)
    nzGenes = nzGenes > min_cells_detected

    # maybe we should normalized by Size_Factor anymore if we always normalize the data after calculating size factor?
    # x = rounded[:, nzGenes] / adata.obs['Size_Factor'][:, None] if 'Size_Factor' in adata.obs.columns else adata.X[:, nzGenes]
    x = rounded[:, nzGenes] / adata.obs['Size_Factor'][:, None] if adata.raw is not None else adata.X[:, nzGenes]

    xim = np.mean(1 / adata.obs['Size_Factor']) if 'Size_Factor' in adata.obs.columns else 1

    f_expression_mean = x.mean(axis=0)

    # For NB: Var(Y) = mu * (1 + mu / k)
    # variance formula
    f_expression_var = np.mean((x - f_expression_mean) ** 2, axis=0)

    disp_guess_meth_moments = f_expression_var - xim * f_expression_mean

    disp_guess_meth_moments = disp_guess_meth_moments / np.power(f_expression_mean, 2)

    res = pd.DataFrame({"mu": f_expression_mean.squeeze(), "disp": disp_guess_meth_moments.squeeze()})
    res.loc[res['mu'] == 0, 'mu'] = None
    res.loc[res['mu'] == 0, 'disp'] = None
    res.loc[res['disp'] < 0, 'disp'] = 0

    res['gene_id'] = adata.var_names[nzGenes]

    return res


def dispersionTable(adata):
    """

    Parameters
    ----------
    adata

    Returns
    -------

    """
    if adata.uns["ispFitInfo"]["blind"] is None:
        raise ("Error: no dispersion model found. Please call estimateDispersions() before calling this function")

    disp_df = pd.DataFrame({"gene_id": adata.uns["ispFitInfo"]["blind"]["disp_table"]["gene_id"],
                            "mean_expression": adata.uns["ispFitInfo"]["blind"]["disp_table"]["mu"],
                            "dispersion_fit": adata.uns["ispFitInfo"]["blind"]["disp_table"]["blind"]["mu"],
                            "dispersion_empirical": adata.uns["ispFitInfo"]["blind"]["disp_table"]["disp"]})

    return disp_df


def vstExprs(adata, dispModelName="blind", expr_matrix=None, round_vals=True):
    """

    Parameters
    ----------
    adata
    dispModelName
    expr_matrix
    round_vals

    Returns
    -------

    """
    fitInfo = adata.uns['dispFitInfo'][dispModelName]

    coefs = fitInfo['disp_func']
    if expr_matrix is None:
        ncounts = adata.raw
        ncounts = adata.raw / adata.obs['Size_Factor'] if 'Size_Factor' in adata.obs.columns else adata.X
        if round_vals:
            ncounts.astype(int)
    else:
        ncounts = expr_matrix

    def vst(q):
        np.log((1 + coefs["extraPois"] + 2 * coefs["asymptDisp"] * q +
                2 * np.sqrt(coefs["asymptDisp"] * q * (1 + coefs["extraPois"] + coefs["asymptDisp"] * q)))
               / (4 * coefs["asymptDisp"])) / np.log(2)

    return vst(ncounts)


# implmentation of Cooks' distance

# https://stackoverflow.com/questions/47686227/poisson-regression-in-statsmodels-and-r

# from __future__ import division, print_function


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


def cook_dist(model, X):
    # Weight matrix
    W = _weight_matrix(model)

    # Hat matrix
    H = _hat_matrix(X, W)
    hii = np.diag(H)  # Diagonal values of hat matrix

    # Pearson residuals
    r = model.resid_pearson

    # Cook's distance (formula used by R = (res/(1 - hat))^2 * hat/(dispersion * p))
    # Note: dispersion is 1 since we aren't modeling overdispersion
    cooks_d = (r / (1 - hii)) ** 2 * hii / (1 * 2)

    return cooks_d


# plot the dispersion

# filter cells by mean, dispersion

# add tom's normalization 
