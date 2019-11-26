import numpy as np
import pandas as pd
import scipy
# from anndata import AnnData
import warnings
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD, FastICA
from .utilities import cook_dist

def szFactor(adata, layers='all', locfunc=np.nanmean, round_exprs=True, method='mean-geometric-mean-total'):
    """Calculate the size factor of the each cell for a AnnData object.
    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        layers: str or list (default: all)
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
    if 'protein' in adata.obsm.keys():
        layer_keys.extend(['X', 'protein'])
    else:
        layer_keys.extend(['X'])
    layers = layer_keys if layers is 'all' else list(set(layer_keys).intersection(layers))

    for layer in layers:
        if layer is 'raw' or layer is 'X':
            CM = adata.raw if adata.raw is not None else adata.X
        elif layer is 'protein':
            if 'protein' in adata.obsm_keys():
                CM = adata.obsm['protein']
            else:
                continue
        else:
            CM = adata.layers[layer]

        if round_exprs:
            CM = CM.round().astype('int') if not issparse(CM) else CM # will this affect downstream analysis?

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


def normalize_expr_data(adata, layers='all', norm_method='log', pseudo_expr=1, relative_expr=True, keep_unflitered=True):
    """Normalize the gene expression value for the AnnData object
    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        layers: str (default: all)
            The layer(s) to be normalized. Default is all, including RNA (X, raw) or spliced, unspliced, protein, etc.
        norm_method: `str`
            The method used to normalize data. Note that this method only applies to the X data.
        pseudo_expr: `int`
            A pseudocount added to the gene expression value before log2 normalization.
        relative_expr: `bool`
            A logic flag to determine whether we need to divide gene expression values first by size factor before normalization
        keep_unflitered: `bool` (default: `True`)
            A logic flag to determine whether we will only store feature genes in the adata object. If it is False, size factor
            will be recalculated only for the selected feature genes.

    Returns
    -------
        adata: :AnnData
            A updated anndata object that are updated with normalized expression values, X.
    """

    if 'use_for_dynamo' in adata.var.columns and keep_unflitered is False:
        adata = adata[:, adata.var[:, 'use_for_dynamo']]
        adata.obs = adata.obs.loc[:, ~adata.obs.columns.str.contains('Size_Factor')]

    layer_keys = list(adata.layers.keys())
    if 'protein' in adata.obsm.keys():
        layer_keys.extend(['X', 'protein'])
    else:
        layer_keys.extend(['X'])
    layers = layer_keys if layers is 'all' else list(set(layer_keys).intersection(layers))

    layer_sz_column_names = [i + '_Size_Factor' for i in set(layers).difference('X')]
    layer_sz_column_names.extend(['Size_Factor'])
    layers_to_sz = list(set(layer_sz_column_names).difference(adata.obs.keys()))

    if len(layers_to_sz) > 0:
        layers = pd.Series(layers_to_sz).str.split('_Size_Factor', expand=True).iloc[:, 0].tolist()
        szFactor(adata, layers=layers, locfunc=np.nanmean, round_exprs=True, method='mean-geometric-mean-total')

    for layer in layers:
        if layer is 'raw' or layer is 'X':
            FM = adata.raw if adata.raw is not None else adata.X
            szfactors = adata.obs['Size_Factor'][:, None]
        elif layer is 'protein':
            if 'protein' in adata.obsm_keys():
               FM = adata.obsm[layer]
               szfactors = adata.obs[layer + '_Size_Factor'][:, None]
            else:
                continue
        else:
            FM = adata.layers[layer]
            szfactors = adata.obs[layer + '_Size_Factor'][:, None]

        if norm_method == 'log' and layer is not 'protein':
            if relative_expr:
                FM = scipy.sparse.diags((1/szfactors).flatten().tolist(), 0).dot(FM) if issparse(FM) else FM / szfactors

            if pseudo_expr is None:
                pseudo_expr = 1
            if layer is 'X':
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
                x = FM[i].A if issparse(FM) else FM[i]
                res = np.log1p(x / (np.exp(np.nansum(np.log1p(x[x > 0])) / n_feature)))
                res[np.isnan(res)] = 0
                # res[res > 100] = 100
                FM[i] = res # no .A is required

            FM = FM.T
        else:
            warnings.warn(norm_method + ' is not implemented yet')

        if layer in ['raw', 'X']:
            adata.X = FM
        elif layer is 'protein' and 'protein' in adata.obsm_keys():
            adata.obsm['X_' + layer] = FM
        else:
            adata.layers['X_' + layer] = FM

    return adata


def Gini(adata, layers='all'):
    """Calculate the Gini coefficient of a numpy array.
     https://github.com/thomasmaxwellnorman/perturbseq_demo/blob/master/perturbseq/util.py

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        layers: str (default: None)
            The layer(s) to be normalized. Default is all, including RNA (X, raw) or spliced, unspliced, protein, etc.

    Returns
    -------
        adata: :AnnData
            A updated anndata object with gini score for the layers (include .X) in the corresponding var columns (layer + '_gini').
    """

    # From: https://github.com/oliviaguest/gini
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    layer_keys = list(adata.layers.keys())
    if 'protein' in adata.obsm.keys():
        layer_keys.extend(['X', 'protein'])
    else:
        layer_keys.extend(['X'])
    layers = layer_keys if layers is 'all' else list(set(layer_keys).intersection(layers))

    for layer in layers:
        if layer is 'raw' or layer is 'X':
            array = adata.raw if adata.raw is not None else adata.X
        elif layer is 'protein':
            if 'protein' in adata.obsm_keys():
                array = adata.obsm[layer]
            else:
                continue
        else:
            array = adata.layers[layer]

        n_features = adata.shape[1]
        gini = np.zeros(n_features)

        for i in np.arange(n_features):
            cur_array = array[:, i].A if issparse(array) else array[:, i] #all values are treated equally, arrays must be 1d
            if np.amin(array) < 0:
                cur_array -= np.amin(cur_array) #values cannot be negative
            cur_array += 0.0000001 # np.min(array[array!=0]) #values cannot be 0
            cur_array = np.sort(cur_array) #values must be sorted
            index = np.arange(1,cur_array.shape[0]+1) #index per array element
            n = cur_array.shape[0] #number of array elements
            gini[i] = ((np.sum((2 * index - n  - 1) * cur_array)) / (n * np.sum(cur_array))) #Gini coefficient

        if layer in ['raw', 'X']:
            adata.var['gini'] = gini
        else:
            adata.var[layer + '_gini'] = gini

    return adata


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
    import statsmodels.api as sm

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


def disp_calc_helper_NB(adata, layers='X', min_cells_detected=1):
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
    layer_keys = list(adata.layers.keys())
    layer_keys.extend('X')
    layers = layer_keys if layers is 'all' else list(set(layer_keys).intersection(layers))

    res_list = []
    for layer in layers:
        if layer is 'raw' or layer is 'X':
            CM = adata.raw if adata.raw is not None else adata.X
            szfactors = adata.obs['Size_Factor'][:, None]
        else:
            CM = adata.layers[layer]
            szfactors = adata.obs[layer + 'Size_Factor'][:, None]

        rounded = CM.round().astype('int') if not issparse(CM) else CM
        lowerDetectedLimit = adata.uns['lowerDetectedLimit'] if 'lowerDetectedLimit' in adata.uns.keys() else 1
        nzGenes = (rounded > lowerDetectedLimit).sum(axis=0)
        nzGenes = nzGenes > min_cells_detected

        nzGenes = np.array(nzGenes).flatten()
        x = scipy.sparse.diags((1/szfactors).flatten().tolist(), 0).dot(rounded[:, nzGenes]) if issparse(rounded) else rounded[:, nzGenes] / szfactors

        xim = np.mean(1 / szfactors) if szfactors is not None else 1

        f_expression_mean = x.mean(axis=0)

        # For NB: Var(Y) = mu * (1 + mu / k)
        # variance formula
        f_expression_var = x.A.std(axis=0, ddof=1)**2 if issparse(x) else x.std(axis=0, ddof=1)**2 # np.mean(np.power(x - f_expression_mean, 2), axis=0) # variance with n - 1
        # https://scialert.net/fulltext/?doi=ajms.2010.1.15 method of moments
        disp_guess_meth_moments = f_expression_var - xim * f_expression_mean # variance - mu

        disp_guess_meth_moments = disp_guess_meth_moments / np.power(f_expression_mean, 2) # this is dispersion parameter (1/k)

        res = pd.DataFrame({"mu": np.array(f_expression_mean).flatten(), "disp": np.array(disp_guess_meth_moments).flatten()})
        res.loc[res['mu'] == 0, 'mu'] = None
        res.loc[res['mu'] == 0, 'disp'] = None
        res.loc[res['disp'] < 0, 'disp'] = 0

        res['gene_id'] = adata.var_names[nzGenes]

        res_list.append(res)

    return layers, res_list


def topTable(adata, layer='X', mode='dispersion'):
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
    layer_keys = list(adata.layers.keys())
    layer_keys.extend('X')
    layer = list(set(layer_keys).intersection(layer))[0]

    if layer in ['raw', 'X']:
        key = 'dispFitInfo'
    else:
        key = layer + '_dispFitInfo'

    if mode is 'dispersion':
        if adata.uns[key] is None:
            raise KeyError("Error: no dispersion model found. Please call estimateDispersions() before calling this function")

        top_df = pd.DataFrame({"gene_id": adata.uns[key]["disp_table"]["gene_id"],
                                "mean_expression": adata.uns[key]["disp_table"]["mu"],
                                "dispersion_fit": adata.uns[key]['disp_func'](adata.uns[key]["disp_table"]["mu"]),
                                "dispersion_empirical": adata.uns[key]["disp_table"]["disp"]})
    elif mode is 'gini':
        top_df = adata.var[layer + '_gini']

    return top_df


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
        res: :class:`~numpy.ndarray`
            A numpy array of the gene expression after VST.

    """
    fitInfo = adata.uns['dispFitInfo']

    coefs = fitInfo['coefs']
    if expr_matrix is None:
        ncounts = adata.X
        if round_vals:
            ncounts.round().astype('int') if not issparse(ncounts) else ncounts
    else:
        ncounts = expr_matrix

    def vst(q): # c( "asymptDisp", "extraPois" )
        return np.log((1 + coefs[1] + 2 * coefs[0] * q +
                2 * np.sqrt(coefs[0] * q * (1 + coefs[1] + coefs[0] * q)))
               / (4 * coefs[0])) / np.log(2)
    res = vst(ncounts.A) if issparse(ncounts) else vst(ncounts)

    return res


def Dispersion(adata, layers='X', modelFormulaStr="~ 1", min_cells_detected=1, removeOutliers=False):
    """ This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        layers: `str` (default: 'X')
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
            A updated annData object with dispFitInfo added to uns attribute as a new key.
    """
    import re

    mu = None
    model_terms = [x.strip() for x in re.compile('~|\\*|\\+').split(modelFormulaStr)]
    model_terms = list(set(model_terms) - set(['']))

    cds_pdata = adata.obs  # .loc[:, model_terms]
    cds_pdata['rowname'] = cds_pdata.index.values
    layers, disp_tables = disp_calc_helper_NB(adata[:, :], layers, min_cells_detected)
    # disp_table['disp'] = np.random.uniform(0, 10, 11)
    # disp_table = cds_pdata.apply(disp_calc_helper_NB(adata[:, :], min_cells_detected))

    # cds_pdata <- dplyr::group_by_(dplyr::select_(rownames_to_column(pData(cds)), "rowname", .dots=model_terms), .dots=model_terms)
    # disp_table <- as.data.frame(cds_pdata %>% do(disp_calc_helper_NB(cds[,.$rowname], cds@expressionFamily, min_cells_detected)))
    for ind in np.arange(len(layers)):
        layer, disp_table = layers[ind], disp_tables[ind]

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

        if layer in ['raw', 'X']:
            adata.uns['dispFitInfo'] = {"disp_table": good, "disp_func": ans, "coefs": coefs}
        else:
            adata.uns[layer + '_dispFitInfo'] = {"disp_table": good, "disp_func": ans, "coefs": coefs}

    return adata


def filter_cells(adata, filter_bool=None, layer='all', keep_unflitered=False, min_expr_genes_s=50, min_expr_genes_u=25, min_expr_genes_p=1,
                 max_expr_genes_s=np.inf, max_expr_genes_u=np.inf, max_expr_genes_p=np.inf):
    """Select valid cells basedon a collection of filters.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        filter_bool: :class:`~numpy.ndarray` (default: None)
            A boolean array from the user to select cells for downstream analysis.
        layer: `str` (default: `all`)
            The data from a particular layer (include X) used for feature selection.
        keep_unflitered: `bool` (default: True)
            Whether to keep cells that don't pass the filtering in the adata object.
        min_expr_genes_s: `int` (default: 50)
            Minimal number of genes with expression for the data in the spliced layer (also used for X)
        min_expr_genes_u: `int` (default: 25)
            Minimal number of genes with expression for the data in the unspliced layer
        min_expr_genes_p: `int` (default: 1)
            Minimal number of genes with expression for the data in the protein layer
        max_expr_genes_s: `float` (default: np.inf)
            Maximal number of genes with expression for the data in the spliced layer (also used for X)
        max_expr_genes_u: `float` (default: np.inf)
            Maximal average expression across cells for the data in the unspliced layer
        max_expr_genes_p: `float` (default: np.inf)
            Maximal average expression across cells for the data in the protein layer

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An updated AnnData object with use_for_dynamo as a new column in obs to indicate the selection of cells for
            downstream analysis. adata will be subsetted with only the cells pass filter if keep_unflitered is set to be
            False.
    """

    detected_bool = np.ones(adata.X.shape[0], dtype=bool)
    detected_bool = (detected_bool) & (((adata.X > 0).sum(1) > min_expr_genes_s) & ((adata.X > 0).sum(1) < max_expr_genes_s)).flatten()

    if "spliced" in adata.layers.keys() & (layer is 'spliced' or layer is 'all'):
        detected_bool = detected_bool & (((adata.layers['spliced'] > 0).sum(1) > min_expr_genes_s) & ((adata.layers['spliced'] > 0).sum(1) < max_expr_genes_s)).flatten()
    if "unspliced" in adata.layers.keys() & (layer is 'unspliced' or layer is 'all'):
        detected_bool = detected_bool & (((adata.layers['unspliced'] > 0).sum(1) > min_expr_genes_u) & ((adata.layers['unspliced'] > 0).sum(1) < max_expr_genes_u)).flatten()
    if "protein" in adata.obsm.keys() & (layer is 'protein' or layer is 'all'):
        detected_bool = detected_bool & (((adata.obsm['protein'] > 0).sum(1) > min_expr_genes_p) & ((adata.obsm['protein'] > 0).sum(1) < max_expr_genes_p)).flatten()

    filter_bool = filter_bool & detected_bool if filter_bool is not None else detected_bool

    if keep_unflitered:
        adata.obs['use_for_dynamo'] = np.array(filter_bool).flatten()
    else:
        adata = adata[np.array(filter_bool).flatten(), :]
        adata.obs['use_for_dynamo'] = True


    return adata


def filter_genes(adata, filter_bool=None, layer='X', keep_unflitered=True, min_cell_s=5, min_cell_u=5, min_cell_p=5,
                 min_avg_exp_s=1e-2, min_avg_exp_u=1e-4, min_avg_exp_p=1e-4, max_avg_exp=100., sort_by='dispersion',
                 n_top_genes=3000):
    """Select feature genes basedon a collection of filters.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        filter_bool: :class:`~numpy.ndarray` (default: None)
            A boolean array from the user to select genes for downstream analysis.
        layer: `str` (default: `X`)
            The data from a particular layer (include X) used for feature selection.
        keep_unflitered: `bool` (default: True)
            Whether to keep genes that don't pass the filtering in the adata object.
        min_cell_s: `int` (default: 5)
            Minimal number of cells with expression for the data in the spliced layer (also used for X)
        min_cell_u: `int` (default: 5)
            Minimal number of cells with expression for the data in the unspliced layer
        min_cell_p: `int` (default: 5)
            Minimal number of cells with expression for the data in the protein layer
        min_avg_exp_s: `float` (default: 1e-2)
            Minimal average expression across cells for the data in the spliced layer (also used for X)
        min_avg_exp_u: `float` (default: 1e-4)
            Minimal average expression across cells for the data in the unspliced layer
        min_avg_exp_p: `float` (default: 1e-4)
            Minimal average expression across cells for the data in the protein layer
        max_avg_exp: `float` (default: 100.)
            Maximal average expression across cells for the data in all layers (also used for X)
        sort_by: `str` (default: dispersion)
            Which soring datatype, either dispersion or Gini index, to be used to select genes.
        n_top_genes
            How many top genes based on scoring method (specified by sort_by) will be selected as feature genes.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An updated AnnData object with use_for_dynamo as a new column in var to indicate the selection of genes for
            downstream analysis. adata will be subsetted with only the genes pass filter if keep_unflitered is set to be
            False.
    """

    detected_bool = np.ones(adata.X.shape[1], dtype=bool)
    detected_bool = (detected_bool) & np.array(((adata.X > 0).sum(0) > min_cell_s) & (adata.X.mean(0) > min_avg_exp_s) & (adata.X.mean(0) < max_avg_exp)).flatten()

    if "spliced" in adata.layers.keys() and layer is 'spliced':
        detected_bool = detected_bool & np.array(((adata.layers['spliced'] > 0).sum(0) > min_cell_s) & (adata.layers['spliced'].mean(0) < min_avg_exp_s) & (adata.layers['spliced'].mean(0) < max_avg_exp)).flatten()
    if "unspliced" in adata.layers.keys() and layer is 'unspliced':
        detected_bool = detected_bool & np.array(((adata.layers['unspliced'] > 0).sum(0) > min_cell_u) & (adata.layers['unspliced'].mean(0) < min_avg_exp_u) & (adata.layers['unspliced'].mean(0) < max_avg_exp)).flatten()
    ############################## The following code need to be updated ##############################
    if "protein" in adata.obsm.keys() and layer is 'protein':
        detected_bool = detected_bool & np.array(((adata.obsm['protein'] > 0).sum(0) > min_cell_p) & (adata.obsm['protein'].mean(0) < min_avg_exp_p) & (adata.obsm['protein'].mean(0) < max_avg_exp)).flatten()

    filter_bool = filter_bool & detected_bool if filter_bool is not None else detected_bool

    adata.var['pass_basic_filter'] = np.array(filter_bool).flatten()
    ### check this
    if sort_by is 'dispersion':
        table = topTable(adata, layer, mode=sort_by)
        table = table.set_index(['gene_id'])
        valid_table = valid_table = table.query("dispersion_empirical > dispersion_fit")
        valid_table = valid_table.loc[set(adata.var.index[filter_bool]).intersection(valid_table.index), :]
        gene_id = np.argsort(-valid_table.loc[:, 'dispersion_empirical'])[:n_top_genes]
        gene_id = valid_table.iloc[gene_id, :].index
        filter_bool = adata.var.index.isin(gene_id)

    elif sort_by is 'gini':
        table = topTable(adata, layer, mode='gini')
        valid_table = table.loc[filter_bool, :]
        gene_id = np.argsort(-valid_table.loc[:, 'gini'])[:n_top_genes]
        gene_id = valid_table.index[gene_id]
        filter_bool = gene_id.isin(adata.var.index)

    if keep_unflitered:
        adata.var['use_for_dynamo'] = np.array(filter_bool).flatten()
    else:
        adata = adata[:, np.array(filter_bool).flatten()]
        adata.var['use_for_dynamo'] = True


    return adata


def recipe_monocle(adata, layer=None, gene_to_use=None, method='pca', num_dim=50, norm_method='log', pseudo_expr=1,
                   feature_selection = 'dispersion', n_top_genes = 2000,
                   relative_expr=True, keep_unflitered=True, **kwargs):
    """This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        layers: str (default: None)
            The layer(s) to be normalized. Default is all, including RNA (X, raw) or spliced, unspliced, protein, etc.
        gene_to_use: `list` (default: None)
            A list genes of gene names that will be used to set as the feature genes for downstream analysis.
        method: `str`
            The linear dimension reduction methods to be used.
        num_dim: `int`
            The number of dimensions reduced to.
        norm_method: `str`
            The method to normalize the data.
        pseudo_expr: `int`
            A pseudocount added to the gene expression value before log2 normalization.
        feature_selection: `str`
            Which soring datatype, either dispersion or Gini index, to be used to select genes.
        n_top_genes: `int`
            How many top genes based on scoring method (specified by sort_by) will be selected as feature genes.
        relative_expr: `bool`
            A logic flag to determine whether we need to divide gene expression values first by size factor before normalization
        keep_unflitered: `bool` (default: True)
            Whether to keep genes that don't pass the filtering in the adata object.
        kwargs:
            Other Parameters passed into the function.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            A updated anndata object that are updated with Size_Factor, normalized expression values, X and reduced dimensions.
    """

    adata = szFactor(adata)
    adata = Dispersion(adata)
    # set use_for_dynamo
    if gene_to_use is None:
        adata = filter_genes(adata, sort_by=feature_selection, n_top_genes=n_top_genes, keep_unflitered=keep_unflitered, **kwargs)
    else:
        adata.var['use_for_dynamo'] = adata.var.index.isin(gene_to_use)
    # normalize on all genes
    adata = normalize_expr_data(adata, norm_method=norm_method, pseudo_expr=pseudo_expr,
                                relative_expr=relative_expr, keep_unflitered=keep_unflitered)
    # only use genes pass filter (based on use_for_dynamo) to perform dimension reduction.
    if layer is None:
        FM = adata.X[:, adata.var.use_for_dynamo.values] if 'spliced' not in adata.layers.keys() else adata.layers['spliced'][:, adata.var.use_for_dynamo.values]
    else:
        if layer is 'X':
            FM = adata.X[:, adata.var.use_for_dynamo.values]
        else:
            adata.layers[layer][:, adata.var.use_for_dynamo.values]

    fm_genesums = FM.sum(axis=0)
    valid_ind = (np.isfinite(fm_genesums)) + (fm_genesums != 0)
    valid_ind = np.array(valid_ind).flatten()
    FM = FM[:, valid_ind]

    if method is 'pca':
        fit = TruncatedSVD(n_components=num_dim, random_state=2019)
        reduce_dim = fit.fit_transform(FM)
        adata.uns['explained_variance_ratio_'] = fit.explained_variance_ratio_
    elif method == 'ica':
        fit=FastICA(num_dim,
                algorithm='deflation', tol=5e-6, fun='logcosh', max_iter=1000)
        reduce_dim=fit.fit_transform(FM.toarray())

    adata.obsm['X_' + method.lower()] = reduce_dim
    adata.uns[method+'_fit'] = fit

    return adata

