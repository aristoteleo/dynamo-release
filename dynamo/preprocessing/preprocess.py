import numpy as np
import pandas as pd
import scipy
from scipy.sparse import issparse
# from anndata import AnnData
import warnings
import statsmodels.api as sm
from sklearn.decomposition import TruncatedSVD, FastICA
from .utilities import cook_dist

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
                   feature_selection = 'gini', n_top_num = 2000, 
                   relative_expr=True, scaling=True, **kwargs):
    """This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

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

# select feature gene function
# def featureSelection(adata, mode='gini', min_cell=2, min_cell_u, mean_):


def parametricDispersionFit(disp_table, initial_coefs=np.array([1e-6, 1])):
    """fThis function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        disp_table: :class:`~pandas.DataFrame`
            AnnData object
        initial_coefs: :class:`~numpy.ndarray`
            Initial parameters for the gamma fit of the dispersion parameters.

    Returns
    -------
        fit: :class:`~statsmodels.api.formula.glm`
            A statsmodels fitting object.
        coefs: :class:`~numpy.ndarray`
            The two resulting gamma fitting coefficients.
        good: :class:`~pandas.DataFrame`
            The subsetted dispersion table that is subjected to Gamma fitting.
    """
    coefs = initial_coefs
    iter = 0
    while True:
        residuals = disp_table['disp'] / (coefs[0] + coefs[1] / disp_table['mu'])
        good = disp_table.loc[(residuals > initial_coefs[0]) & (residuals < 10000), :]
        # https://stats.stackexchange.com/questions/356053/the-identity-link-function-does-not-respect-the-domain-of-the-gamma-family
        fit = sm.formula.glm("disp ~ I(1 / mu)", data=good,
                             family=sm.families.Gamma(link=sm.genmod.families.links.identity)).fit(start_params=coefs)

        oldcoefs = coefs
        coefs = fit.params

        if(coefs[0] < initial_coefs[0]):
            coefs[0] = initial_coefs[0]
        if coefs[1] < 0:
            warnings.warn("Parametric dispersion fit may be failed.")

        if(np.sum(np.log(coefs / oldcoefs) ** 2 < coefs[0])):
            break
        iter += 1

        if(iter > 10):
            warnings.warn("Dispersion fit didn't converge")
            break
        if not np.all(coefs > 0):
            warnings.warn('Parametric dispersion fit may be failed.')

    return fit, coefs, good


def disp_calc_helper_NB(adata, min_cells_detected, layer='X'):
    """ This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        min_cells_detected: `int` (default: None)
            The mimimal required number of cells with expression for selecting gene for dispersion fitting.
        layer: `str`
            The layer of data used for dispersion fitting.

    Returns
    -------
        res: :class:`~pandas.DataFrame`
            A pandas dataframe with mu, dispersion for each gene that passes filters.
    """
    rounded = adata.raw.astype('int') if adata.raw is not None else adata.X
    lowerDetectedLimit = adata.uns['lowerDetectedLimit'] if 'lowerDetectedLimit' in adata.uns.keys() else 1
    nzGenes = (rounded > lowerDetectedLimit).sum(axis=0)
    nzGenes = nzGenes > min_cells_detected

    nzGenes = np.array(nzGenes).flatten()
    x = adata.X[:, nzGenes]

    xim = np.mean(1 / adata.obs['Size_Factor']) if 'Size_Factor' in adata.obs.columns else 1

    f_expression_mean = x.mean(axis=0)

    # For NB: Var(Y) = mu * (1 + mu / k)
    # variance formula
    f_expression_var = np.mean(np.power(x - f_expression_mean, 2), axis=0) # variance
    # https://scialert.net/fulltext/?doi=ajms.2010.1.15 method of moments
    disp_guess_meth_moments = f_expression_var - xim * f_expression_mean # variance - mu

    disp_guess_meth_moments = disp_guess_meth_moments / np.power(f_expression_mean, 2) # this is dispersion parameter (1/k)

    res = pd.DataFrame({"mu": np.array(f_expression_mean).flatten(), "disp": np.array(disp_guess_meth_moments).flatten()})
    res.loc[res['mu'] == 0, 'mu'] = None
    res.loc[res['mu'] == 0, 'disp'] = None
    res.loc[res['disp'] < 0, 'disp'] = 0

    res['gene_id'] = adata.var_names[nzGenes]

    return res


def dispersionTable(adata):
    """ This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object

    Returns
    -------
        disp_df: :class:`~pandas.DataFrame`
            The data frame with the gene_id, mean_expression, dispersion_fit and dispersion_empirical as the columns.
    """
    if adata.uns["ispFitInfo"]["blind"] is None:
        raise ("Error: no dispersion model found. Please call estimateDispersions() before calling this function")

    disp_df = pd.DataFrame({"gene_id": adata.uns["dispFitinfo"]["disp_table"]["gene_id"],
                            "mean_expression": adata.uns["dispFitinfo"]["disp_table"]["mu"],
                            "dispersion_fit": adata.uns["dispFitinfo"]["disp_table"]["blind"]["mu"],
                            "dispersion_empirical": adata.uns["dispFitinfo"]["disp_table"]["disp"]})

    return disp_df


def vstExprs(adata, expr_matrix=None, round_vals=True):
    """ This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        dispModelName: `str`
            The name of the dispersion function to use for VST.
        expr_matrix: :class:`~numpy.ndarray`
            An matrix of values to transform. Must be normalized (e.g. by size factors) already. This function doesn't do this for you.
        round_vals: `bool`
            Whether to round expression values to the nearest integer before applying the transformation.

    Returns
    -------
        disp_df: :class:`~pandas.DataFrame`
            The name of the dispersion function to use for VST.

    """
    fitInfo = adata.uns['dispFitInfo']

    coefs = fitInfo['disp_func']
    if expr_matrix is None:
        ncounts = adata.X
        if round_vals:
            ncounts.astype(int)
    else:
        ncounts = expr_matrix

    def vst(q): # c( "asymptDisp", "extraPois" )
        np.log((1 + coefs[1] + 2 * coefs[0] * q +
                2 * np.sqrt(coefs[0] * q * (1 + coefs[1] + coefs[0] * q)))
               / (4 * coefs[0])) / np.log(2)

    return vst(ncounts)


def Dispersion(adata, layers=None, modelFormulaStr="~ 1", min_cells_detected=1, removeOutliers=True):
    """ This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        layers: `str` (default: None)
            The layer(s) to be used for calculating dispersion. Default is X if there is no spliced layers.
        modelFormulaStr: `str`
            The model formula used to calculate dispersion parameters. Not used.
        min_cells_detected: `int`
            The minimum number of cells detected for calculating the dispersion.
        removeOutliers: `bool` (default: True)
            Whether to remove outliers when performing dispersion fitting.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            A updated annData object with dispFitinfo added to uns attribute as a new key.
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
    fit, coefs, good = res[0], res[1], res[2]

    if removeOutliers:
        # influence = fit.get_influence()
        # #CD is the distance and p is p-value
        # (CD, p) = influence.cooks_distance

        CD = cook_dist(fit, 1 / good['mu'][:, None])
        cooksCutoff = 4 / good.shape[0]
        print("Removing ", len(CD[CD > cooksCutoff]), " outliers")
        outliers = CD > cooksCutoff
        # use CD.index.values? remove genes that lost when doing parameter fitting
        lost_gene = set(good.index.values).difference(set(range(len(CD))))
        outliers[lost_gene] = False
        res = parametricDispersionFit(good.loc[~ outliers, :])

        fit, coefs = res[0], res[1]

    def ans(q):
        return coefs[0] + coefs[1] / q

    adata.uns['dispFitinfo'] = {"disp_table": good, "disp_func": ans}
    return adata






# plot the dispersion

# filter cells by mean, dispersion

# add tom's normalization 
