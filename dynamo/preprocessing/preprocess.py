from collections.abc import Iterable
import numpy as np
import pandas as pd
import warnings
from scipy.sparse import issparse, csr_matrix
from sklearn.decomposition import FastICA
from sklearn.utils import sparsefuncs
import anndata
from anndata import AnnData
from typing import Optional, Union, Callable

from .cell_cycle import cell_cycle_scores
from ..tools.utils import update_dict
from .utils import (
    convert2symbol,
    pca,
    clusters_stats,
    cook_dist,
    get_shared_counts,
    get_svr_filter,
    Freeman_Tukey,
    merge_adata_attrs,
    sz_util,
    normalize_util,
    get_sz_exprs,
    unique_var_obs_adata,
    layers2csr,
    collapse_adata,
    NTR,
    detect_experiment_datatype,
    basic_stats,
    add_noise_to_duplicates,
    gene_exp_fraction,
)
from ..dynamo_logger import (
    main_info,
    main_critical,
    main_warning,
    LoggerManager,
)
from ..utils import copy_adata
from ..configuration import DynamoAdataConfig, DynamoAdataKeyManager
from .gene_selection_utils import _infer_labeling_experiment_type, filter_genes_by_outliers


def szFactor(
    adata_ori: anndata.AnnData,
    layers: Union[str, list] = "all",
    total_layers: Union[list, None] = None,
    splicing_total_layers: bool = False,
    X_total_layers: bool = False,
    locfunc: Callable = np.nanmean,
    round_exprs: bool = False,
    method: str = "median",
    scale_to: Union[float, None] = None,
    use_all_genes_cells: bool = True,
    genes_use_for_norm: Union[list, None] = None,
) -> anndata.AnnData:
    """Calculate the size factor of the each cell using geometric mean of total UMI across cells for a AnnData object.
    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        adata_ori: :class:`~anndata.AnnData`.
            AnnData object.
        layers: str or list (default: `all`)
            The layer(s) to be normalized. Default is `all`, including RNA (X, raw) or spliced, unspliced, protein, etc.
        total_layers: list or None (default `None`)
            The layer(s) that can be summed up to get the total mRNA. for example, ["spliced", "unspliced"], ["uu", "ul"
            , "su", "sl"] or ["new", "old"], etc.
        splicing_total_layers: bool (default `False`)
            Whether to also normalize spliced / unspliced layers by size factor from total RNA.
        X_total_layers: bool (default `False`)
            Whether to also normalize adata.X by size factor from total RNA.
        locfunc: `function` (default: `np.nanmean`)
            The function to normalize the data.
        round_exprs: `bool` (default: `False`)
            A logic flag to determine whether the gene expression should be rounded into integers.
        method: `str` (default: `mean-geometric-mean-total`)
            The method used to calculate the expected total reads / UMI used in size factor calculation.
            Only `mean-geometric-mean-total` / `geometric` and `median` are supported. When `median` is used, `locfunc`
            will be replaced with `np.nanmedian`.
        scale_to: `float` or None (default: `None`)
            The final total expression for each cell that will be scaled to.
        use_all_genes_cells: `bool` (default: `True`)
            A logic flag to determine whether all cells and genes should be used for the size factor calculation.
        genes_use_for_norm: `list` (default: `None`)
            A list of gene names that will be used to calculate total RNA for each cell and then the size factor for
            normalization. This is often very useful when you want to use only the host genes to normalize the dataset
            in a virus infection experiment (i.e. CMV or SARS-CoV-2 infection).

    Returns
    -------
        adata: :AnnData
            An updated anndata object that are updated with the `Size_Factor` (`layer_` + `Size_Factor`) column(s) in
            the obs attribute.
    """
    if use_all_genes_cells:
        # let us ignore the `inplace` parameter in pandas.Categorical.remove_unused_categories  warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adata = adata_ori if genes_use_for_norm is None else adata_ori[:, genes_use_for_norm]
    else:
        cell_inds = adata_ori.obs.use_for_pca if "use_for_pca" in adata_ori.obs.columns else adata_ori.obs.index
        filter_list = ["use_for_pca", "pass_basic_filter"]
        filter_checker = [i in adata_ori.var.columns for i in filter_list]
        which_filter = np.where(filter_checker)[0]

        gene_inds = adata_ori.var[filter_list[which_filter[0]]] if len(which_filter) > 0 else adata_ori.var.index

        adata = adata_ori[cell_inds, :][:, gene_inds]

        if genes_use_for_norm is not None:
            # let us ignore the `inplace` parameter in pandas.Categorical.remove_unused_categories  warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                adata = adata[:, adata.var_names.intersection(genes_use_for_norm)]

    if total_layers is not None:
        if not isinstance(total_layers, list):
            total_layers = [total_layers]
        if len(set(total_layers).difference(adata.layers.keys())) == 0:
            total = None
            for t_key in total_layers:
                total = adata.layers[t_key] if total is None else total + adata.layers[t_key]
            adata.layers["_total_"] = total
            layers.extend(["_total_"])

    layers = DynamoAdataKeyManager.get_layer_keys(adata, layers)
    if "raw" in layers and adata.raw is None:
        adata.raw = adata.copy()

    excluded_layers = []
    if not X_total_layers:
        excluded_layers.extend(["X"])
    if not splicing_total_layers:
        excluded_layers.extend(["spliced", "unspliced"])

    for layer in layers:
        if layer in excluded_layers:
            sfs, cell_total = sz_util(
                adata,
                layer,
                round_exprs,
                method,
                locfunc,
                total_layers=None,
                scale_to=scale_to,
            )
        else:
            sfs, cell_total = sz_util(
                adata,
                layer,
                round_exprs,
                method,
                locfunc,
                total_layers=total_layers,
                scale_to=scale_to,
            )

        sfs[~np.isfinite(sfs)] = 1
        if layer == "raw":
            adata.obs[layer + "_Size_Factor"] = sfs
            adata.obs["Size_Factor"] = sfs
            adata.obs["initial_cell_size"] = cell_total
        elif layer == "X":
            adata.obs["Size_Factor"] = sfs
            adata.obs["initial_cell_size"] = cell_total
        elif layer == "_total_":
            adata.obs["total_Size_Factor"] = sfs
            adata.obs["initial" + layer + "cell_size"] = cell_total
            del adata.layers["_total_"]
        else:
            adata.obs[layer + "_Size_Factor"] = sfs
            adata.obs["initial_" + layer + "_cell_size"] = cell_total

    adata_ori = merge_adata_attrs(adata_ori, adata, attr="obs")

    return adata_ori


def normalize_expr_data(
    adata: anndata.AnnData,
    layers: str = "all",
    total_szfactor: str = "total_Size_Factor",
    splicing_total_layers: str = False,
    X_total_layers: str = False,
    norm_method: Union[Callable, None] = None,
    pseudo_expr: int = 1,
    relative_expr: bool = True,
    keep_filtered: bool = True,
    recalc_sz: bool = False,
    sz_method: str = "median",
    scale_to: Union[float, None] = None,
) -> anndata.AnnData:
    """Normalize the gene expression value for the AnnData object
    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object.
        layers: `str` (default: `all`)
            The layer(s) to be normalized. Default is all, including RNA (X, raw) or spliced, unspliced, protein, etc.
        total_szfactor: `str` (default: `total_Size_Factor`)
            The column name in the .obs attribute that corresponds to the size factor for the total mRNA.
        splicing_total_layers: bool (default `False`)
            Whether to also normalize spliced / unspliced layers by size factor from total RNA.
        X_total_layers: bool (default `False`)
            Whether to also normalize adata.X by size factor from total RNA.
        norm_method: `function` or None (default: `None`)
            The method used to normalize data. Can be either function `np.log1p, np.log2 or any other functions or
            string `clr`. By default, only .X will be size normalized and log1p transformed while data in other layers
            will only be size normalized.
        pseudo_expr: `int` (default: `1`)
            A pseudocount added to the gene expression value before log/log2 normalization.
        relative_expr: `bool` (default: `True`)
            A logic flag to determine whether we need to divide gene expression values first by size factor before
            normalization.
        keep_filtered: `bool` (default: `True`)
            A logic flag to determine whether we will only store feature genes in the adata object. If it is False, size
            factor will be recalculated only for the selected feature genes.
        recalc_sz: `bool` (default: `False`)
            A logic flag to determine whether we need to recalculate size factor based on selected genes before
            normalization.
        sz_method: `str` (default: `mean-geometric-mean-total`)
            The method used to calculate the expected total reads / UMI used in size factor calculation.
            Only `mean-geometric-mean-total` / `geometric` and `median` are supported. When `median` is used, `locfunc`
            will be replaced with `np.nanmedian`.
        scale_to: `float` or None (default: `None`)
            The final total expression for each cell that will be scaled to.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An updated anndata object that are updated with normalized expression values for different layers.
    """

    if recalc_sz:
        if "use_for_pca" in adata.var.columns and keep_filtered is False:
            adata = adata[:, adata.var.loc[:, "use_for_pca"]]

        adata.obs = adata.obs.loc[:, ~adata.obs.columns.str.contains("Size_Factor")]

    layers = DynamoAdataKeyManager.get_layer_keys(adata, layers)

    layer_sz_column_names = [i + "_Size_Factor" for i in set(layers).difference("X")]
    layer_sz_column_names.extend(["Size_Factor"])
    layers_to_sz = list(set(layer_sz_column_names).difference(adata.obs.keys()))

    if len(layers_to_sz) > 0:
        layers = pd.Series(layers_to_sz).str.split("_Size_Factor", expand=True).iloc[:, 0].tolist()
        if "Size_Factor" in layers:
            layers[np.where(np.array(layers) == "Size_Factor")[0][0]] = "X"
        szFactor(
            adata,
            layers=layers,
            locfunc=np.nanmean,
            round_exprs=True,
            method=sz_method,
            scale_to=scale_to,
        )

    excluded_layers = []
    if not X_total_layers:
        excluded_layers.extend(["X"])
    if not splicing_total_layers:
        excluded_layers.extend(["spliced", "unspliced"])

    for layer in layers:
        if layer in excluded_layers:
            szfactors, CM = get_sz_exprs(adata, layer, total_szfactor=None)
        else:
            szfactors, CM = get_sz_exprs(adata, layer, total_szfactor=total_szfactor)

        if norm_method is None and layer == "X":
            CM = normalize_util(CM, szfactors, relative_expr, pseudo_expr, np.log1p)
        elif norm_method in [np.log1p, np.log, np.log2, Freeman_Tukey, None] and layer != "protein":
            CM = normalize_util(CM, szfactors, relative_expr, pseudo_expr, norm_method)

        elif layer == "protein":  # norm_method == 'clr':
            if norm_method != "clr":
                main_warning(
                    "For protein data, log transformation is not recommended. Using clr normalization by default."
                )
            """This normalization implements the centered log-ratio (CLR) normalization from Seurat which is computed
            for each gene (M Stoeckius, ‎2017).
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
            main_warning(norm_method + " is not implemented yet")

        if layer in ["raw", "X"]:
            adata.X = CM
        elif layer == "protein" and "protein" in adata.obsm_keys():
            adata.obsm["X_protein"] = CM
        else:
            adata.layers["X_" + layer] = CM

        adata.uns["pp"]["norm_method"] = norm_method.__name__ if callable(norm_method) else norm_method

    return adata


def Gini(adata, layers="all"):
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
        adata: :class:`~anndata.AnnData`
            An updated anndata object with gini score for the layers (include .X) in the corresponding var columns
            (layer + '_gini').
    """

    # From: https://github.com/oliviaguest/gini
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    layers = DynamoAdataKeyManager.get_layer_keys(adata, layers)

    for layer in layers:
        if layer == "raw":
            CM = adata.raw.X
        elif layer == "X":
            CM = adata.X
        elif layer == "protein":
            if "protein" in adata.obsm_keys():
                CM = adata.obsm[layer]
            else:
                continue
        else:
            CM = adata.layers[layer]

        n_features = adata.shape[1]
        gini = np.zeros(n_features)

        for i in np.arange(n_features):
            # all values are treated equally, arrays must be 1d
            cur_cm = CM[:, i].A if issparse(CM) else CM[:, i]
            if np.amin(CM) < 0:
                cur_cm -= np.amin(cur_cm)  # values cannot be negative
            cur_cm += 0.0000001  # np.min(array[array!=0]) #values cannot be 0
            cur_cm = np.sort(cur_cm)  # values must be sorted
            # index per array element
            index = np.arange(1, cur_cm.shape[0] + 1)
            n = cur_cm.shape[0]  # number of array elements
            gini[i] = (np.sum((2 * index - n - 1) * cur_cm)) / (n * np.sum(cur_cm))  # Gini coefficient

        if layer in ["raw", "X"]:
            adata.var["gini"] = gini
        else:
            adata.var[layer + "_gini"] = gini

    return adata


def parametric_dispersion_fit(disp_table: pd.DataFrame, initial_coefs: np.ndarray = np.array([1e-6, 1])):
    """This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

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
        residuals = disp_table["disp"] / (coefs[0] + coefs[1] / disp_table["mu"])
        good = disp_table.loc[(residuals > initial_coefs[0]) & (residuals < 10000), :]
        # https://stats.stackexchange.com/questions/356053/the-identity-link-function-does-not-respect-the-domain-of-the
        # -gamma-family
        fit = sm.formula.glm(
            "disp ~ I(1 / mu)",
            data=good,
            family=sm.families.Gamma(link=sm.genmod.families.links.identity),
        ).train(start_params=coefs)

        oldcoefs = coefs
        coefs = fit.params

        if coefs[0] < initial_coefs[0]:
            coefs[0] = initial_coefs[0]
        if coefs[1] < 0:
            main_warning("Parametric dispersion fit may be failed.")

        if np.sum(np.log(coefs / oldcoefs) ** 2 < coefs[0]):
            break
        iter += 1

        if iter > 10:
            main_warning("Dispersion fit didn't converge")
            break
        if not np.all(coefs > 0):
            main_warning("Parametric dispersion fit may be failed.")

    return fit, coefs, good


def disp_calc_helper_NB(adata: anndata.AnnData, layers: str = "X", min_cells_detected: int = 1) -> pd.DataFrame:
    """This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

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
    layers = DynamoAdataKeyManager.get_layer_keys(adata, layers=layers, include_protein=False)

    res_list = []
    for layer in layers:
        if layer == "raw":
            CM = adata.raw.X
            szfactors = adata.obs[layer + "Size_Factor"][:, None]
        elif layer == "X":
            CM = adata.X
            szfactors = adata.obs["Size_Factor"][:, None]
        else:
            CM = adata.layers[layer]
            szfactors = adata.obs[layer + "Size_Factor"][:, None]

        if issparse(CM):
            CM.data = np.round(CM.data, 0)
            rounded = CM
        else:
            rounded = CM.round().astype("int")

        lowerDetectedLimit = adata.uns["lowerDetectedLimit"] if "lowerDetectedLimit" in adata.uns.keys() else 1
        nzGenes = (rounded > lowerDetectedLimit).sum(axis=0)
        nzGenes = nzGenes > min_cells_detected

        nzGenes = nzGenes.A1 if issparse(rounded) else nzGenes
        if layer.startswith("X_"):
            x = rounded[:, nzGenes]
        else:
            x = (
                rounded[:, nzGenes].multiply(csr_matrix(1 / szfactors))
                if issparse(rounded)
                else rounded[:, nzGenes] / szfactors
            )

        xim = np.mean(1 / szfactors) if szfactors is not None else 1

        f_expression_mean = x.mean(axis=0)

        # For NB: Var(Y) = mu * (1 + mu / k)
        # x.A.var(axis=0, ddof=1)
        f_expression_var = (
            (x.multiply(x).mean(0).A1 - f_expression_mean.A1 ** 2) * x.shape[0] / (x.shape[0] - 1)
            if issparse(x)
            else x.var(axis=0, ddof=0) ** 2
        )  # np.mean(np.power(x - f_expression_mean, 2), axis=0) # variance with n - 1
        # https://scialert.net/fulltext/?doi=ajms.2010.1.15 method of moments
        disp_guess_meth_moments = f_expression_var - xim * f_expression_mean  # variance - mu

        disp_guess_meth_moments = disp_guess_meth_moments / np.power(
            f_expression_mean, 2
        )  # this is dispersion parameter (1/k)

        res = pd.DataFrame(
            {
                "mu": np.array(f_expression_mean).flatten(),
                "disp": np.array(disp_guess_meth_moments).flatten(),
            }
        )
        res.loc[res["mu"] == 0, "mu"] = None
        res.loc[res["mu"] == 0, "disp"] = None
        res.loc[res["disp"] < 0, "disp"] = 0

        res["gene_id"] = adata.var_names[nzGenes]

        res_list.append(res)

    return layers, res_list


def top_table(adata: anndata.AnnData, layer: str = "X", mode: str = "dispersion") -> pd.DataFrame:
    """This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object

    Returns
    -------
        disp_df: :class:`~pandas.DataFrame`
            The data frame with the gene_id, mean_expression, dispersion_fit and dispersion_empirical as the columns.
    """
    layer = DynamoAdataKeyManager.get_layer_keys(adata, layers=layer, include_protein=False)[0]

    if layer in ["X"]:
        key = "dispFitInfo"
    else:
        key = layer + "_dispFitInfo"

    if mode == "dispersion":
        if adata.uns[key] is None:
            estimate_dispersion(adata, layers=[layer])

        if adata.uns[key] is None:
            raise KeyError(
                "Error: for adata.uns.key=%s, no dispersion model found. Please call estimate_dispersion() before calling this function"
                % key
            )

        top_df = pd.DataFrame(
            {
                "gene_id": adata.uns[key]["disp_table"]["gene_id"],
                "mean_expression": adata.uns[key]["disp_table"]["mu"],
                "dispersion_fit": adata.uns[key]["disp_func"](adata.uns[key]["disp_table"]["mu"]),
                "dispersion_empirical": adata.uns[key]["disp_table"]["disp"],
            }
        )
        top_df = top_df.set_index("gene_id")

    elif mode == "gini":
        top_df = adata.var[layer + "_gini"]

    return top_df


def vstExprs(
    adata: anndata.AnnData,
    expr_matrix: Union[np.ndarray, None] = None,
    round_vals: bool = True,
) -> np.ndarray:
    """This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        dispModelName: `str`
            The name of the dispersion function to use for VST.
        expr_matrix: :class:`~numpy.ndarray` or `None` (default: `None`)
            An matrix of values to transform. Must be normalized (e.g. by size factors) already. This function doesn't
            do this for you.
        round_vals: `bool`
            Whether to round expression values to the nearest integer before applying the transformation.

    Returns
    -------
        res: :class:`~numpy.ndarray`
            A numpy array of the gene expression after VST.

    """
    fitInfo = adata.uns["dispFitInfo"]

    coefs = fitInfo["coefs"]
    if expr_matrix is None:
        ncounts = adata.X
        if round_vals:
            if issparse(ncounts):
                ncounts.data = np.round(ncounts.data, 0)
            else:
                ncounts = ncounts.round().astype("int")
    else:
        ncounts = expr_matrix

    def vst(q):  # c( "asymptDisp", "extraPois" )
        return np.log(
            (1 + coefs[1] + 2 * coefs[0] * q + 2 * np.sqrt(coefs[0] * q * (1 + coefs[1] + coefs[0] * q)))
            / (4 * coefs[0])
        ) / np.log(2)

    res = vst(ncounts.toarray()) if issparse(ncounts) else vst(ncounts)

    return res


def estimate_dispersion(
    adata: anndata.AnnData,
    layers: str = "X",
    modelFormulaStr: str = "~ 1",
    min_cells_detected: int = 1,
    removeOutliers: bool = False,
) -> anndata.AnnData:
    """This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

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
            An updated annData object with dispFitInfo added to uns attribute as a new key.
    """
    import re

    logger = LoggerManager.gen_logger("dynamo-preprocessing")
    # mu = None
    model_terms = [x.strip() for x in re.compile("~|\\*|\\+").split(modelFormulaStr)]
    model_terms = list(set(model_terms) - set([""]))

    cds_pdata = adata.obs  # .loc[:, model_terms]
    cds_pdata["rowname"] = cds_pdata.index.values
    layers, disp_tables = disp_calc_helper_NB(adata[:, :], layers, min_cells_detected)
    # disp_table['disp'] = np.random.uniform(0, 10, 11)
    # disp_table = cds_pdata.apply(disp_calc_helper_NB(adata[:, :], min_cells_detected))

    # cds_pdata <- dplyr::group_by_(dplyr::select_(rownames_to_column(pData(cds)), "rowname", .dots=model_terms), .dots
    # =model_terms)
    # disp_table <- as.data.frame(cds_pdata %>% do(disp_calc_helper_NB(cds[,.$rowname], cds@expressionFamily, min_cells_
    # detected)))
    for ind in range(len(layers)):
        layer, disp_table = layers[ind], disp_tables[ind]

        if disp_table is None:
            raise Exception("Parametric dispersion fitting failed, please set a different lowerDetectionLimit")

        disp_table = disp_table.loc[np.where(disp_table["mu"] != np.nan)[0], :]

        res = parametric_dispersion_fit(disp_table)
        fit, coefs, good = res[0], res[1], res[2]

        if removeOutliers:
            # influence = fit.get_influence().cooks_distance()
            # #CD is the distance and p is p-value
            # (CD, p) = influence.cooks_distance

            CD = cook_dist(fit, 1 / good["mu"][:, None], good)
            cooksCutoff = 4 / good.shape[0]
            print("Removing ", len(CD[CD > cooksCutoff]), " outliers")
            outliers = CD > cooksCutoff
            # use CD.index.values? remove genes that lost when doing parameter fitting
            lost_gene = set(good.index.values).difference(set(range(len(CD))))
            outliers[lost_gene] = True
            res = parametric_dispersion_fit(good.loc[~outliers, :])

            fit, coefs = res[0], res[1]

        def ans(q):
            return coefs[0] + coefs[1] / q

        if layer == "X":
            logger.info_insert_adata("dispFitInfo", "uns")
            adata.uns["dispFitInfo"] = {
                "disp_table": good,
                "disp_func": ans,
                "coefs": coefs,
            }
        else:
            logger.info_insert_adata(layer + "_dispFitInfo", "uns")
            adata.uns[layer + "_dispFitInfo"] = {
                "disp_table": good,
                "disp_func": ans,
                "coefs": coefs,
            }

    return adata


def SVRs(
    adata_ori: anndata.AnnData,
    filter_bool: Union[np.ndarray, None] = None,
    layers: str = "X",
    relative_expr: bool = True,
    total_szfactor: str = "total_Size_Factor",
    min_expr_cells: int = 0,
    min_expr_avg: int = 0,
    max_expr_avg: int = 0,
    svr_gamma: Union[float, None] = None,
    winsorize: bool = False,
    winsor_perc: tuple = (1, 99.5),
    sort_inverse: bool = False,
    use_all_genes_cells: bool = False,
) -> anndata.AnnData:
    """This function is modified from https://github.com/velocyto-team/velocyto.py/blob/master/velocyto/analysis.py

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object.
        filter_bool: :class:`~numpy.ndarray` (default: None)
            A boolean array from the user to select genes for downstream analysis.
        layers: `str` (default: 'X')
            The layer(s) to be used for calculating dispersion score via support vector regression (SVR). Default is X
            if there is no spliced layers.
        relative_expr: `bool` (default: `True`)
            A logic flag to determine whether we need to divide gene expression values first by size factor before run
            SVR.
        total_szfactor: `str` (default: `total_Size_Factor`)
            The column name in the .obs attribute that corresponds to the size factor for the total mRNA.
        min_expr_cells: `int` (default: `2`)
            minimum number of cells that express that gene for it to be considered in the fit.
        min_expr_avg: `int` (default: `0`)
            The minimum average of genes across cells accepted.
        max_expr_avg: `float` (defaul: `20`)
            The maximum average of genes across cells accepted before treating house-keeping/outliers for removal.
        svr_gamma: `float` or None (default: `None`)
            the gamma hyper-parameter of the SVR.
        winsorize: `bool` (default: `False`)
            Wether to winsorize the data for the cv vs mean model.
        winsor_perc: `tuple` (default: `(1, 99.5)`)
            the up and lower bound of the winsorization.
        sort_inverse: `bool` (default: `False`)
            if True it sorts genes from less noisy to more noisy (to use for size estimation not for feature selection).
        use_all_genes_cells: `bool` (default: `False`)
            A logic flag to determine whether all cells and genes should be used for the size factor calculation.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An updated annData object with `log_m`, `log_cv`, `score` added to .obs columns and `SVR` added to uns
            attribute as a new key.
    """
    from sklearn.svm import SVR

    layers = DynamoAdataKeyManager.get_layer_keys(adata_ori, layers)

    if use_all_genes_cells:
        # let us ignore the `inplace` parameter in pandas.Categorical.remove_unused_categories  warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adata = adata_ori[:, filter_bool].copy() if filter_bool is not None else adata_ori
    else:
        cell_inds = adata_ori.obs.use_for_pca if "use_for_pca" in adata_ori.obs.columns else adata_ori.obs.index
        filter_list = ["use_for_pca", "pass_basic_filter"]
        filter_checker = [i in adata_ori.var.columns for i in filter_list]
        which_filter = np.where(filter_checker)[0]

        gene_inds = adata_ori.var[filter_list[which_filter[0]]] if len(which_filter) > 0 else adata_ori.var.index

        # let us ignore the `inplace` parameter in pandas.Categorical.remove_unused_categories  warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adata = adata_ori[cell_inds, gene_inds].copy()
        filter_bool = filter_bool[gene_inds]

    for layer in layers:
        if layer == "raw":
            CM = adata.X.copy() if adata.raw is None else adata.raw
            szfactors = (
                adata.obs[layer + "_Size_Factor"].values[:, None]
                if adata.raw.X is not None
                else adata.obs["Size_Factor"].values[:, None]
            )
        elif layer == "X":
            CM = adata.X.copy()
            szfactors = adata.obs["Size_Factor"].values[:, None]
        elif layer == "protein":
            if "protein" in adata.obsm_keys():
                CM = adata.obsm["protein"].copy()
                szfactors = adata.obs[layer + "_Size_Factor"].values[:, None]
            else:
                continue
        else:
            CM = adata.layers[layer].copy()
            szfactors = (
                adata.obs[layer + "_Size_Factor"].values[:, None]
                if layer + "_Size_Factor" in adata.obs.columns
                else None
            )

        if total_szfactor is not None and total_szfactor in adata.obs.keys():
            szfactors = adata.obs[total_szfactor].values[:, None] if total_szfactor in adata.obs.columns else None

        if szfactors is not None and relative_expr:
            if issparse(CM):
                sparsefuncs.inplace_row_scale(CM, 1 / szfactors)
            else:
                CM /= szfactors

        if winsorize:
            if min_expr_cells <= ((100 - winsor_perc[1]) * CM.shape[0] * 0.01):
                min_expr_cells = int(np.ceil((100 - winsor_perc[1]) * CM.shape[1] * 0.01)) + 2

        detected_bool = np.array(
            ((CM > 0).sum(0) >= min_expr_cells) & (CM.mean(0) <= max_expr_avg) & (CM.mean(0) >= min_expr_avg)
        ).flatten()

        valid_CM = CM[:, detected_bool]
        if winsorize:
            down, up = (
                np.percentile(valid_CM.A, winsor_perc, 0)
                if issparse(valid_CM)
                else np.percentile(valid_CM, winsor_perc, 0)
            )
            Sfw = (
                np.clip(valid_CM.A, down[None, :], up[None, :])
                if issparse(valid_CM)
                else np.percentile(valid_CM, winsor_perc, 0)
            )
            mu = Sfw.mean(0)
            sigma = Sfw.std(0, ddof=1)
        else:
            mu = np.array(valid_CM.mean(0)).flatten()
            sigma = (
                np.array(
                    np.sqrt(
                        (valid_CM.multiply(valid_CM).mean(0).A1 - (mu) ** 2)
                        # * (adata.n_obs)
                        # / (adata.n_obs - 1)
                    )
                )
                if issparse(valid_CM)
                else valid_CM.std(0, ddof=1)
            )

        cv = sigma / mu
        log_m = np.array(np.log2(mu)).flatten()
        log_cv = np.array(np.log2(cv)).flatten()
        log_m[mu == 0], log_cv[mu == 0] = 0, 0

        if svr_gamma is None:
            svr_gamma = 150.0 / len(mu)
        # Fit the Support Vector Regression
        clf = SVR(gamma=svr_gamma)
        clf.fit(log_m[:, None], log_cv)
        fitted_fun = clf.predict
        ff = fitted_fun(log_m[:, None])
        score = log_cv - ff
        if sort_inverse:
            score = -score

        prefix = "" if layer == "X" else layer + "_"
        (adata.var[prefix + "log_m"], adata.var[prefix + "log_cv"], adata.var[prefix + "score"],) = (
            np.nan,
            np.nan,
            -np.inf,
        )
        (
            adata.var.loc[detected_bool, prefix + "log_m"],
            adata.var.loc[detected_bool, prefix + "log_cv"],
            adata.var.loc[detected_bool, prefix + "score"],
        ) = (
            np.array(log_m).flatten(),
            np.array(log_cv).flatten(),
            np.array(score).flatten(),
        )

        key = "velocyto_SVR" if layer == "raw" or layer == "X" else layer + "_velocyto_SVR"
        adata_ori.uns[key] = {"SVR": fitted_fun}

    adata_ori = merge_adata_attrs(adata_ori, adata, attr="var")

    return adata_ori


def filter_cells(
    adata: anndata.AnnData,
    filter_bool: Union[np.ndarray, None] = None,
    layer: str = "all",
    keep_filtered: bool = False,
    min_expr_genes_s: int = 50,
    min_expr_genes_u: int = 25,
    min_expr_genes_p: int = 1,
    max_expr_genes_s: float = np.inf,
    max_expr_genes_u: float = np.inf,
    max_expr_genes_p: float = np.inf,
    shared_count: Union[int, None] = None,
) -> anndata.AnnData:
    """Select valid cells based on a collection of filters.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object.
        filter_bool: :class:`~numpy.ndarray` (default: `None`)
            A boolean array from the user to select cells for downstream analysis.
        layer: `str` (default: `all`)
            The data from a particular layer (include X) used for feature selection.
        keep_filtered: `bool` (default: `False`)
            Whether to keep cells that don't pass the filtering in the adata object.
        min_expr_genes_s: `int` (default: `50`)
            Minimal number of genes with expression for a cell in the data from the spliced layer (also used for X).
        min_expr_genes_u: `int` (default: `25`)
            Minimal number of genes with expression for a cell in the data from the unspliced layer.
        min_expr_genes_p: `int` (default: `1`)
            Minimal number of genes with expression for a cell in the data from in the protein layer.
        max_expr_genes_s: `float` (default: `np.inf`)
            Maximal number of genes with expression for a cell in the data from the spliced layer (also used for X).
        max_expr_genes_u: `float` (default: `np.inf`)
            Maximal number of genes with expression for a cell in the data from the unspliced layer.
        max_expr_genes_p: `float` (default: `np.inf`)
            Maximal number of protein with expression for a cell in the data from the protein layer.
        shared_count: `int` or `None` (default: `None`)
            The minimal shared number of counts for each cell across genes between layers.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An updated AnnData object with use_for_pca as a new column in obs to indicate the selection of cells for
            downstream analysis. adata will be subsetted with only the cells pass filtering if keep_filtered is set to
            be False.
    """

    detected_bool = np.ones(adata.X.shape[0], dtype=bool)
    detected_bool = (detected_bool) & (
        ((adata.X > 0).sum(1) >= min_expr_genes_s) & ((adata.X > 0).sum(1) <= max_expr_genes_s)
    ).flatten()

    if ("spliced" in adata.layers.keys()) & (layer == "spliced" or layer == "all"):
        detected_bool = (
            detected_bool
            & (
                ((adata.layers["spliced"] > 0).sum(1) >= min_expr_genes_s)
                & ((adata.layers["spliced"] > 0).sum(1) <= max_expr_genes_s)
            ).flatten()
        )
    if ("unspliced" in adata.layers.keys()) & (layer == "unspliced" or layer == "all"):
        detected_bool = (
            detected_bool
            & (
                ((adata.layers["unspliced"] > 0).sum(1) >= min_expr_genes_u)
                & ((adata.layers["unspliced"] > 0).sum(1) <= max_expr_genes_u)
            ).flatten()
        )
    if ("protein" in adata.obsm.keys()) & (layer == "protein" or layer == "all"):
        detected_bool = (
            detected_bool
            & (
                ((adata.obsm["protein"] > 0).sum(1) >= min_expr_genes_p)
                & ((adata.obsm["protein"] > 0).sum(1) <= max_expr_genes_p)
            ).flatten()
        )

    if shared_count is not None:
        layers = DynamoAdataKeyManager.get_layer_keys(adata, layer, False)
        detected_bool = detected_bool & get_shared_counts(adata, layers, shared_count, "cell")

    filter_bool = filter_bool & detected_bool if filter_bool is not None else detected_bool

    filter_bool = np.array(filter_bool).flatten()
    if keep_filtered:
        adata.obs["pass_basic_filter"] = filter_bool
    else:
        adata._inplace_subset_obs(filter_bool)
        adata.obs["pass_basic_filter"] = True

    return adata


def filter_genes_by_clusters_(
    adata: anndata.AnnData,
    cluster: str,
    min_avg_U: float = 0.02,
    min_avg_S: float = 0.08,
    size_limit: int = 40,
):
    """Prepare filtering genes on the basis of cluster-wise expression threshold
    This function is taken from velocyto in order to reproduce velocyto's DentateGyrus notebook.

    Arguments
    ---------
        adata: :class:`~anndata.AnnData`
            AnnData object.
        cluster: `str`
            A column in the adata.obs attribute which will be used for cluster specific expression filtering.
        min_avg_U: float
            Include genes that have unspliced average bigger than `min_avg_U` in at least one of the clusters
        min_avg_S: float
            Include genes that have spliced average bigger than `min_avg_U` in at least one of the clusters
        Note: the two conditions are combined by and "&" logical operator.

    Returns
    -------
    Nothing but it creates the attribute
    clu_avg_selected: np.ndarray bool
        The gene cluster that is selected
    To perform the filtering use the method `filter_genes`
    """
    U, S, cluster_uid = (
        adata.layers["unspliced"],
        adata.layers["spliced"],
        adata.obs[cluster],
    )
    cluster_uid, cluster_ix = np.unique(cluster_uid, return_inverse=True)

    U_avgs, S_avgs = clusters_stats(U, S, cluster_uid, cluster_ix, size_limit=size_limit)
    clu_avg_selected = (U_avgs.max(1) > min_avg_U) & (S_avgs.max(1) > min_avg_S)

    return clu_avg_selected


def select_genes(
    adata: anndata.AnnData,
    layer: str = "X",
    total_szfactor: str = "total_Size_Factor",
    keep_filtered: bool = True,
    sort_by: str = "SVR",
    n_top_genes: int = 2000,
    SVRs_kwargs: dict = {},
    only_bools: bool = False,
) -> anndata.AnnData:
    """Select feature genes based on a collection of filters.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object.
        layer: `str` (default: `X`)
            The data from a particular layer (include X) used for feature selection.
        total_szfactor: `str` (default: `total_Size_Factor`)
            The column name in the .obs attribute that corresponds to the size factor for the total mRNA.
        keep_filtered: `bool` (default: `True`)
            Whether to keep genes that don't pass the filtering in the adata object.
        sort_by: `str` (default: `SVR`)
            Which soring method, either SVR, dispersion or Gini index, to be used to select genes.
        n_top_genes: `int` (default: `int`)
            How many top genes based on scoring method (specified by sort_by) will be selected as feature genes.
        only_bools: `bool` (default: `False`)
            Only return a vector of bool values.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An updated AnnData object with use_for_pca as a new column in .var attributes to indicate the selection of
            genes for downstream analysis. adata will be subsetted with only the genes pass filter if keep_unflitered is
            set to be False.
    """

    filter_bool = (
        adata.var["pass_basic_filter"]
        if "pass_basic_filter" in adata.var.columns
        else np.ones(adata.shape[1], dtype=bool)
    )

    if adata.shape[1] <= n_top_genes:
        filter_bool = np.ones(adata.shape[1], dtype=bool)
    else:
        if sort_by == "dispersion":
            table = top_table(adata, layer, mode="dispersion")
            valid_table = table.query("dispersion_empirical > dispersion_fit")
            valid_table = valid_table.loc[
                set(adata.var.index[filter_bool]).intersection(valid_table.index),
                :,
            ]
            gene_id = np.argsort(-valid_table.loc[:, "dispersion_empirical"])[:n_top_genes]
            gene_id = valid_table.iloc[gene_id, :].index
            filter_bool = adata.var.index.isin(gene_id)
        elif sort_by == "gini":
            table = top_table(adata, layer, mode="gini")
            valid_table = table.loc[filter_bool, :]
            gene_id = np.argsort(-valid_table.loc[:, "gini"])[:n_top_genes]
            gene_id = valid_table.index[gene_id]
            filter_bool = gene_id.isin(adata.var.index)
        elif sort_by == "SVR":
            SVRs_args = {
                "min_expr_cells": 0,
                "min_expr_avg": 0,
                "max_expr_avg": np.inf,
                "svr_gamma": None,
                "winsorize": False,
                "winsor_perc": (1, 99.5),
                "sort_inverse": False,
            }
            SVRs_args = update_dict(SVRs_args, SVRs_kwargs)
            adata = SVRs(
                adata,
                layers=[layer],
                total_szfactor=total_szfactor,
                filter_bool=filter_bool,
                **SVRs_args,
            )

            filter_bool = get_svr_filter(adata, layer=layer, n_top_genes=n_top_genes, return_adata=False)

    if keep_filtered:
        adata.var["use_for_pca"] = filter_bool
    else:
        adata._inplace_subset_var(filter_bool)
        adata.var["use_for_pca"] = True

    return filter_bool if only_bools else adata


def recipe_monocle(
    adata: anndata.AnnData,
    reset_X: bool = False,
    tkey: Union[str, None] = None,
    t_label_keys: Union[str, list, None] = None,
    experiment_type: Union[str, None] = None,
    normalized: Union[bool, None] = None,
    layer: Union[str, None] = None,
    total_layers: Union[bool, list, None] = None,
    splicing_total_layers: bool = False,
    X_total_layers: bool = False,
    genes_use_for_norm: Union[list, None] = None,
    genes_to_use: Union[list, None] = None,
    genes_to_append: Union[list, None] = None,
    genes_to_exclude: Union[list, None] = None,
    exprs_frac_max: float = 1,
    method: str = "pca",
    num_dim: int = 30,
    sz_method: str = "median",
    scale_to: Union[float, None] = None,
    norm_method: Union[str, None] = None,
    pseudo_expr: int = 1,
    feature_selection: str = "SVR",
    n_top_genes: int = 2000,
    maintain_n_top_genes: bool = True,
    relative_expr: bool = True,
    keep_filtered_cells: Optional[bool] = None,
    keep_filtered_genes: Optional[bool] = None,
    keep_raw_layers: Optional[bool] = None,
    scopes: Union[str, Iterable, None] = None,
    fc_kwargs: Union[dict, None] = None,
    fg_kwargs: Union[dict, None] = None,
    sg_kwargs: Union[dict, None] = None,
    copy: bool = False,
    feature_selection_layer: Union[list, np.ndarray, np.array, str] = "X",
) -> Union[anndata.AnnData, None]:
    """This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object.
        tkey: `str` or None (default: None)
            The column key for the labeling time  of cells in .obs. Used for labeling based scRNA-seq data (will also
            support for conventional scRNA-seq data). Note that `tkey` will be saved to adata.uns['pp']['tkey'] and used
            in `dyn.tl.dynamics` in which when `group` is None, `tkey` will also be used for calculating  1st/2st moment
            or covariance. We recommend to use hour as the unit of `time`.
        t_label_keys: `str`, `list` or None (default: None)
            The column key(s) for the labeling time label of cells in .obs. Used for either "conventional" or "labeling
            based" scRNA-seq data. Not used for now and `tkey` is implicitly assumed as `t_label_key` (however, `tkey`
            should just be the time of the experiment).
        experiment_type: `str` {`deg`, `kin`, `one-shot`, `mix_std_stm`, 'mixture'} or None, (default: `None`)
            experiment type for labeling single cell RNA-seq. Available options are:
            (1) 'conventional': conventional single-cell RNA-seq experiment, if `experiment_type` is `None` and there is
            only splicing data, this will be set to `conventional`;
            (2) 'deg': chase/degradation experiment. Cells are first labeled with an extended period, followed by chase;
            (3) 'kin': pulse/synthesis/kinetics experiment. Cells are labeled for different duration in a time-series;
            (4) 'one-shot': one-shot kinetic experiment. Cells are only labeled for a short pulse duration;
            Other possible experiments include:
            (5) 'mix_pulse_chase' or 'mix_kin_deg': This is a mixture chase experiment in which the entire experiment
            lasts for a certain period of time which an initial pulse followed by washing out at different time point
            but chasing cells at the same time point. This type of labeling strategy was adopted in scEU-seq paper. For
            kind of experiment, we need to assume a non-steady state dynamics.
            (4) 'mix_std_stm';
        reset_X: bool (default: `False`)
            Whether do you want to let dynamo reset `adata.X` data based on layers stored in your experiment. One
            critical functionality of dynamo is about visualizing RNA velocity vector flows which requires proper data
            into which the high dimensional RNA velocity vectors will be projected.
            (1) For `kinetics` experiment, we recommend the use of `total` layer as `adata.X`;
            (2) For `degradation/conventional` experiment scRNA-seq, we recommend using `splicing` layer as `adata.X`.
            Set `reset_X` to `True` to set those default values if you are not sure.
        normalized: `None` or `bool` (default: `None`)
            If you already normalized your data (or run recipe_monocle already), set this to be `True` to avoid
            renormalizing your data. By default it is set to be `None` and the first 20 values of adata.X (if adata.X is
            sparse) or its first column will be checked to determine whether you already normalized your data. This only
            works for UMI based or read-counts data.
        layer: str (default: `None`)
            The layer(s) to be normalized. Default is all, including RNA (X, raw) or spliced, unspliced, protein, etc.
        total_layers: bool, list or None (default `None`)
            The layer(s) that can be summed up to get the total mRNA. for example, ["spliced", "unspliced"], ["uu", "ul"
            , "su", "sl"] or ["total"], etc. If total_layers is `True`, total_layers will be set to be `total` or ["uu",
            "ul", "su", "sl"] depends on whether you have labeling but no splicing or labeling and splicing data.
        splicing_total_layers: bool (default `False`)
            Whether to also normalize spliced / unspliced layers by size factor from total RNA.
        X_total_layers: bool (default `False`)
            Whether to also normalize adata.X by size factor from total RNA.
        genes_use_for_norm: `list` (default: `None`)
            A list of gene names that will be used to calculate total RNA for each cell and then the size factor for
            normalization. This is often very useful when you want to use only the host genes to normalize the dataset
            in a virus infection experiment (i.e. CMV or SARS-CoV-2 infection).
        genes_to_use: `list` (default: `None`)
            A list of gene names that will be used to set as the feature genes for downstream analysis.
        genes_to_append: `list` (default: `None`)
            A list of gene names that will be appended to the feature genes list for downstream analysis.
        genes_to_exclude: `list` (default: `None`)
            A list of gene names that will be excluded to the feature genes list for downstream analysis.
        exprs_frac_max: `float` (default: `1`)
            The minimal fraction of gene counts to the total counts across cells that will used to filter genes. By
            default it is 1 which means we don't filter any genes, but we need to change it to 0.005 or something in
            order to remove some highly expressed housekeeping genes.
        method: `str` (default: `pca`)
            The linear dimension reduction methods to be used.
        num_dim: `int` (default: `30`)
            The number of linear dimensions reduced to.
        sz_method: `str` (default: `mean-geometric-mean-total`)
            The method used to calculate the expected total reads / UMI used in size factor calculation.
            Only `mean-geometric-mean-total` / `geometric` and `median` are supported. When `median` is used, `locfunc`
            will be replaced with `np.nanmedian`.
        scale_to: `float` or None (default: `None`)
            The final total expression for each cell that will be scaled to.
        norm_method: `function` or None (default: function `None`)
            The method to normalize the data. Can be any numpy function or `Freeman_Tukey`. By default, only .X will be
            size normalized and log1p transformed while data in other layers will only be size factor normalized.
        pseudo_expr: `int` (default: `1`)
            A pseudocount added to the gene expression value before log/log2 normalization.
        feature_selection: `str` (default: `SVR`)
            Which soring method, either dispersion, SVR or Gini index, to be used to select genes.
        n_top_genes: `int` (default: `2000`)
            How many top genes based on scoring method (specified by sort_by) will be selected as feature genes.
        maintain_n_top_genes: `bool` (default: `True`)
            Whether to ensure 2000 feature genes selected no matter what genes_to_use, genes_to_append, etc. are
            specified. The only exception is that if `genes_to_use` is supplied with `n_top_genes`.
        relative_expr: `bool` (default: `True`)
            A logic flag to determine whether we need to divide gene expression values first by size factor before
            normalization.
        keep_filtered_cells: `bool` (default: `True`)
            Whether to keep genes that don't pass the filtering in the returned adata object.
        keep_filtered_genes: `bool` (default: `True`)
            Whether to keep genes that don't pass the filtering in the returned adata object.
        keep_raw_layers: `bool` (default: `True`)
            Whether to keep layers with raw measurements in the returned adata object.
        scopes: `str`, `list-like` or `None` (default: `None`)
            Scopes are needed when you use non-official gene name as your gene indices (or adata.var_name). This
            arugument corresponds to type of types of identifiers, either a list or a comma-separated fields to specify
            type of input qterms, e.g. “entrezgene”, “entrezgene,symbol”, [“ensemblgene”, “symbol”]. Refer to official
            MyGene.info docs (https://docs.mygene.info/en/latest/doc/query_service.html#available_fields) for full list
            of fields.
        fc_kwargs: `dict` or None (default: `None`)
            Other Parameters passed into the filter_cells function.
        fg_kwargs: `dict` or None (default: `None`)
            Other Parameters passed into the filter_genes function.
        sg_kwargs: `dict` or None (default: `None`)
            Other Parameters passed into the select_genes function.
        copy:
            Whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An new or updated anndata object, based on copy parameter, that are updated with Size_Factor, normalized
            expression values, X and reduced dimensions, etc.
    """
    logger = LoggerManager.gen_logger("dynamo-preprocessing")
    logger.log_time()
    keep_filtered_cells = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_cells, DynamoAdataConfig.RECIPE_MONOCLE_KEEP_FILTERED_CELLS_KEY
    )
    keep_filtered_genes = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_genes, DynamoAdataConfig.RECIPE_MONOCLE_KEEP_FILTERED_GENES_KEY
    )
    keep_raw_layers = DynamoAdataConfig.use_default_var_if_none(
        keep_raw_layers, DynamoAdataConfig.RECIPE_MONOCLE_KEEP_RAW_LAYERS_KEY
    )

    adata = copy_adata(adata) if copy else adata

    logger.info("apply Monocole recipe to adata...", indent_level=1)
    if "use_for_pca" in adata.var.columns:
        del adata.var["use_for_pca"]  # avoid use_for_pca was set previously.

    adata = convert2symbol(adata, scopes=scopes)
    n_cells, n_genes = adata.n_obs, adata.n_vars

    # Since convert2symbol may subset adata and generate a new AnnData object,
    # we should create all following data after convert2symbol (gene names)
    adata.uns["pp"] = {}
    if norm_method == "Freeman_Tukey":
        norm_method = Freeman_Tukey

    basic_stats(adata)
    (
        has_splicing,
        has_labeling,
        splicing_labeling,
        has_protein,
    ) = detect_experiment_datatype(adata)
    logger.info_insert_adata("pp", "uns")
    logger.info_insert_adata("has_splicing", "uns['pp']", indent_level=2)
    logger.info_insert_adata("has_labling", "uns['pp']", indent_level=2)
    logger.info_insert_adata("splicing_labeling", "uns['pp']", indent_level=2)
    logger.info_insert_adata("has_protein", "uns['pp']", indent_level=2)
    (
        adata.uns["pp"]["has_splicing"],
        adata.uns["pp"]["has_labeling"],
        adata.uns["pp"]["splicing_labeling"],
        adata.uns["pp"]["has_protein"],
    ) = (has_splicing, has_labeling, splicing_labeling, has_protein)

    if has_splicing and has_labeling and splicing_labeling:
        layer = (
            [
                "X",
                "uu",
                "ul",
                "su",
                "sl",
                "spliced",
                "unspliced",
                "new",
                "total",
            ]
            if layer is None
            else layer
        )

        if type(total_layers) != list:
            total_layers = ["uu", "ul", "su", "sl"] if total_layers else None
    if has_splicing and has_labeling and not splicing_labeling:
        layer = ["X", "spliced", "unspliced", "new", "total"] if layer is None else layer

        if type(total_layers) != list:
            total_layers = ["total"] if total_layers else None
    elif has_labeling and not has_splicing:
        layer = ["X", "total", "new"] if layer is None else layer

        if type(total_layers) != list:
            total_layers = ["total"] if total_layers else None
    elif has_splicing and not has_labeling:
        layer = ["X", "spliced", "unspliced"] if layer is None else layer

    logger.info("ensure all cell and variable names unique.", indent_level=1)
    adata = unique_var_obs_adata(adata)
    logger.info(
        "ensure all data in different layers in csr sparse matrix format.",
        indent_level=1,
    )
    adata = layers2csr(adata)
    logger.info("ensure all labeling data properly collapased", indent_level=1)
    adata = collapse_adata(adata)

    # reset adata.X
    if has_labeling:
        if tkey is None:
            main_warning(
                "\nWhen analyzing labeling based scRNA-seq without providing `tkey`, dynamo will try to use "
                "\n `time` as the key for labeling time. Please correct this via supplying the correct `tkey`"
                "\nif needed."
            )
            tkey = "time"
        if tkey not in adata.obs.keys():
            raise ValueError(f"`tkey` {tkey} that encodes the labeling time is not existed in your adata.")
        if experiment_type is None:
            experiment_type = _infer_labeling_experiment_type(adata, tkey)

        main_info("detected experiment type: %s" % experiment_type)

        valid_experiment_types = [
            "one-shot",
            "kin",
            "mixture",
            "mix_std_stm",
            "kinetics",
            "mix_pulse_chase",
            "mix_kin_deg",
            "deg",
        ]
        if experiment_type not in valid_experiment_types:
            raise ValueError(
                "expriment_type can only be one of ['one-shot', 'kin', 'mixture', 'mix_std_stm', "
                "'kinetics', 'mix_pulse_chase','mix_kin_deg', 'deg']"
            )
        elif experiment_type == "kinetics":
            experiment_type = "kin"
        elif experiment_type == "degradation":
            experiment_type = "deg"

    if reset_X:
        if has_labeling:
            if experiment_type.lower() in [
                "one-shot",
                "kin",
                "mixture",
                "mix_std_stm",
                "kinetics",
                "mix_pulse_chase",
                "mix_kin_deg",
            ]:
                adata.X = adata.layers["total"].copy()
            if experiment_type.lower() in ["deg", "degradation"] and has_splicing:
                adata.X = adata.layers["spliced"].copy()
            if experiment_type.lower() in ["deg", "degradation"] and not has_splicing:
                main_warning(
                    "It is not possible to calculate RNA velocity from a degradation experiment which has no "
                    "splicing information."
                )
                adata.X = adata.layers["total"].copy()
            else:
                adata.X = adata.layers["total"].copy()
        else:
            adata.X = adata.layers["spliced"].copy()

    if tkey is not None:
        if adata.obs[tkey].max() > 60:
            main_warning(
                "Looks like you are using minutes as the time unit. For the purpose of numeric stability, "
                "we recommend using hour as the time unit."
            )

    logger.info_insert_adata("tkey", "uns['pp']", indent_level=2)
    logger.info_insert_adata("experiment_type", "uns['pp']", indent_level=2)
    adata.uns["pp"]["tkey"] = tkey
    adata.uns["pp"]["experiment_type"] = "conventional" if experiment_type is None else experiment_type

    _szFactor, _logged = (True, True) if normalized else (False, False)
    if normalized is None and not has_labeling:
        if "raw_data" in adata.uns_keys():
            _szFactor, _logged = (
                not adata.uns["raw_data"],
                not adata.uns["raw_data"],
            )
        else:
            # automatically detect whether the data is size-factor normalized -- no integers (only works for readcounts
            # / UMI based data).
            _szFactor = not np.allclose(
                (adata.X.data[:20] if issparse(adata.X) else adata.X[:, 0]) % 1,
                0,
                atol=1e-3,
            )
            # check whether total UMI is the same -- if not the same, logged
            if _szFactor:
                _logged = not np.allclose(
                    np.sum(adata.X.sum(1)[np.random.choice(adata.n_obs, 10)] - adata.X.sum(1)[0]),
                    0,
                    atol=1e-1,
                )

        if _szFactor or _logged:
            main_warning(
                "dynamo detects your data is size factor normalized and/or log transformed. If this is not "
                "right, plese set `normalized = False."
            )

    # filter bad cells
    filter_cells_kwargs = {
        "filter_bool": None,
        "layer": "all",
        "min_expr_genes_s": min(50, 0.01 * n_genes),
        "min_expr_genes_u": min(25, 0.01 * n_genes),
        "min_expr_genes_p": min(2, 0.01 * n_genes),
        "max_expr_genes_s": np.inf,
        "max_expr_genes_u": np.inf,
        "max_expr_genes_p": np.inf,
        "shared_count": None,
    }
    if fc_kwargs is not None:
        filter_cells_kwargs.update(fc_kwargs)

    logger.info("filtering cells...")
    logger.info_insert_adata("pass_basic_filter", "obs")
    adata = filter_cells(adata, keep_filtered=keep_filtered_cells, **filter_cells_kwargs)
    logger.info(f"{adata.obs.pass_basic_filter.sum()} cells passed basic filters.")

    filter_genes_kwargs = {
        "filter_bool": None,
        "layer": "all",
        "min_cell_s": max(5, 0.01 * n_cells),
        "min_cell_u": max(5, 0.005 * n_cells),
        "min_cell_p": max(5, 0.005 * n_cells),
        "min_avg_exp_s": 0,
        "min_avg_exp_u": 0,
        "min_avg_exp_p": 0,
        "max_avg_exp": np.inf,
        "min_count_s": 0,
        "min_count_u": 0,
        "min_count_p": 0,
        "shared_count": 30,
    }
    if fg_kwargs is not None:
        filter_genes_kwargs.update(fg_kwargs)

    # set pass_basic_filter for genes
    logger.info("filtering gene...")
    logger.info_insert_adata("pass_basic_filter", "var")
    adata = filter_genes_by_outliers(
        adata,
        **filter_genes_kwargs,
    )
    logger.info(f"{adata.var.pass_basic_filter.sum()} genes passed basic filters.")

    if adata.var.pass_basic_filter.sum() == 0:
        logger.error(
            "No genes pass basic filters. Please check your data, for example, layer names, etc or other " "arguments."
        )
        raise Exception()
    if adata.obs.pass_basic_filter.sum() == 0:
        logger.error("No cells pass basic filters. Please check your data or arguments, for example, fc_kwargs.")
        raise Exception()

    # calculate sz factor
    logger.info("calculating size factor...")
    if not _szFactor or "Size_Factor" not in adata.obs_keys():
        adata = szFactor(
            adata,
            total_layers=total_layers,
            scale_to=scale_to,
            splicing_total_layers=splicing_total_layers,
            X_total_layers=X_total_layers,
            layers=layer if type(layer) is list else "all",
            genes_use_for_norm=genes_use_for_norm,
        )

    # if feature_selection.lower() == "dispersion":
    #     adata = estimate_dispersion(adata)

    # set use_for_pca (use basic_filtered data)
    select_genes_dict = {
        "min_expr_cells": 0,
        "min_expr_avg": 0,
        "max_expr_avg": np.inf,
        "svr_gamma": None,
        "winsorize": False,
        "winsor_perc": (1, 99.5),
        "sort_inverse": False,
    }
    if sg_kwargs is not None:
        select_genes_dict.update(sg_kwargs)

    if genes_to_use is None:
        pass_basic_filter_num = adata.var.pass_basic_filter.sum()
        if pass_basic_filter_num < n_top_genes:
            logger.warning(
                f"only {pass_basic_filter_num} genes passed basic filtering, but you requested {n_top_genes} "
                f"genes for feature selection. Try lowering the gene selection stringency: "
                f"{select_genes_dict}",
            )
        logger.info("selecting genes in layer: %s, sort method: %s..." % (feature_selection_layer, feature_selection))
        adata = select_genes(
            adata,
            layer=feature_selection_layer,
            sort_by=feature_selection,
            n_top_genes=n_top_genes,
            keep_filtered=True,
            SVRs_kwargs=select_genes_dict,
        )
    else:
        if len(adata.var_names.intersection(genes_to_use)) == 0:
            logger.error(
                "No genes from genes_to_use matches with the gene names from adata. Please ensure you use gene short "
                "names!"
            )
            raise Exception()
        logger.info_insert_adata("use_for_pca", "var")
        adata.var["use_for_pca"] = adata.var.index.isin(genes_to_use)

    logger.info_insert_adata("frac", "var")
    adata.var["frac"], invalid_ids = gene_exp_fraction(X=adata.X, threshold=exprs_frac_max)
    genes_to_exclude = (
        list(adata.var_names[invalid_ids])
        if genes_to_exclude is None
        else genes_to_exclude + list(adata.var_names[invalid_ids])
    )

    if genes_to_append is not None:
        valid_genes = adata.var.index.intersection(genes_to_append)
        if len(valid_genes) > 0:
            adata.var.loc[valid_genes, "use_for_pca"] = True

    if genes_to_exclude is not None:
        valid_genes = adata.var.index.intersection(genes_to_exclude)
        if len(valid_genes) > 0:
            adata.var.loc[valid_genes, "use_for_pca"] = False

        if adata.var.use_for_pca.sum() < 50 and not maintain_n_top_genes:
            main_warning(
                "You only have less than 50 feature gene selected. Are you sure you want to exclude all "
                "genes passed to the genes_to_exclude argument?"
            )

    if maintain_n_top_genes:
        if genes_to_append is not None:
            n_top_genes = n_top_genes - len(genes_to_append)
            valid_ids = adata.var.index.difference(genes_to_exclude + genes_to_append)
        else:
            valid_ids = adata.var.index.difference(genes_to_exclude)

        if n_top_genes > 0:
            # let us ignore the `inplace` parameter in pandas.Categorical.remove_unused_categories  warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                filter_bool = select_genes(
                    adata[:, valid_ids],
                    sort_by=feature_selection,
                    n_top_genes=n_top_genes,
                    keep_filtered=True,  # no effect to adata
                    SVRs_kwargs=select_genes_dict,
                    only_bools=True,
                )

            adata.var.loc[valid_ids, "use_for_pca"] = filter_bool

    if not keep_filtered_genes:
        logger.info("Discarding genes that failed the filtering...")
        adata._inplace_subset_var(adata.var["use_for_pca"])

    # normalized data based on sz factor
    if not _logged:
        total_szfactor = "total_Size_Factor" if total_layers is not None else None
        logger.info("size factor normalizing the data, followed by log1p transformation.")
        adata = normalize_expr_data(
            adata,
            layers=layer if type(layer) is list else "all",
            total_szfactor=total_szfactor,
            splicing_total_layers=splicing_total_layers,
            X_total_layers=X_total_layers,
            norm_method=norm_method,
            pseudo_expr=pseudo_expr,
            relative_expr=relative_expr,
            keep_filtered=keep_filtered_genes,
            sz_method=sz_method,
            scale_to=scale_to,
        )
    else:
        layers = DynamoAdataKeyManager.get_layer_keys(adata, "all")
        for layer in layers:
            if layer != "X":
                logger.info_insert_adata("X_" + layer, "layers")
                adata.layers["X_" + layer] = adata.layers[layer].copy()
        logger.info_insert_adata("norm_method", "uns['pp']", indent_level=2)
        adata.uns["pp"]["norm_method"] = None

    # only use genes pass filter (based on use_for_pca) to perform dimension reduction.
    if layer is None:
        CM = adata.X[:, adata.var.use_for_pca.values]
    else:
        if "X" in layer:
            CM = adata.X[:, adata.var.use_for_pca.values]
        elif "total" in layer:
            CM = adata.layers["X_total"][:, adata.var.use_for_pca.values]
        elif "spliced" in layer:
            CM = adata.layers["X_spliced"][:, adata.var.use_for_pca.values]
        elif "protein" in layer:
            CM = adata.obsm["X_protein"]
        elif type(layer) is str:
            CM = adata.layers["X_" + layer][:, adata.var.use_for_pca.values]
        else:
            raise ValueError(
                f"your input layer argument should be either a `str` or a list that includes one of `X`, "
                f"`total`, `protein` element. `Layer` currently is {layer}."
            )

    cm_genesums = CM.sum(axis=0)
    valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
    valid_ind = np.array(valid_ind).flatten()

    bad_genes = np.where(adata.var.use_for_pca)[0][~valid_ind]
    if genes_to_append is not None and len(adata.var.index[bad_genes].intersection(genes_to_append)) > 0:
        raise ValueError(
            f"The gene list passed to argument genes_to_append contains genes with no expression "
            f"across cells or non finite values. Please check those genes:"
            f"{set(bad_genes).intersection(genes_to_append)}!"
        )

    adata.var.iloc[bad_genes, adata.var.columns.tolist().index("use_for_pca")] = False
    CM = CM[:, valid_ind]
    logger.info("applying %s ..." % (method.upper()))
    if method == "pca":
        adata = pca(adata, CM, num_dim, "X_" + method.lower())
        adata.obsm["X"] = adata.obsm["X_" + method.lower()]

    elif method == "ica":
        fit = FastICA(
            num_dim,
            algorithm="deflation",
            tol=5e-6,
            fun="logcosh",
            max_iter=1000,
        )
        reduce_dim = fit.fit_transform(CM.toarray())

        adata.obsm["X_" + method.lower()] = reduce_dim
        adata.obsm["X"] = adata.obsm["X_" + method.lower()]

    logger.info_insert_adata(method + "_fit", "uns")
    adata.uns[method + "_fit"], adata.uns["feature_selection"] = (
        {},
        feature_selection,
    )
    # calculate NTR for every cell:
    ntr, var_ntr = NTR(adata)
    if ntr is not None:
        logger.info_insert_adata("ntr", "obs")
        logger.info_insert_adata("ntr", "var")
        adata.obs["ntr"] = ntr
        adata.var["ntr"] = var_ntr

    logger.info("cell cycle scoring...")
    try:
        cell_cycle_scores(adata)
    except Exception:
        logger.warning(
            "\nDynamo is not able to perform cell cycle staging for you automatically. \n"
            "Since dyn.pl.phase_diagram in dynamo by default colors cells by its cell-cycle stage, \n"
            "you need to set color argument accordingly if confronting errors related to this."
        )

    if "raw_data" in adata.uns_keys():
        logger.info_insert_adata("raw_data", "uns")
        adata.uns["raw_data"] = False

    if not keep_raw_layers:
        layers = list(adata.layers.keys())
        for layer in layers:
            if not layer.startswith("X_"):
                del adata.layers[layer]

    logger.finish_progress(progress_name="recipe_monocle preprocess")

    if copy:
        return adata
    return None


def recipe_velocyto(
    adata: anndata.AnnData,
    total_layers=None,
    method="pca",
    num_dim=30,
    norm_method=None,
    pseudo_expr=1,
    feature_selection="SVR",
    n_top_genes=2000,
    cluster="Clusters",
    relative_expr=True,
    keep_filtered_genes=None,
):
    """This function is adapted from the velocyto's DentateGyrus notebook.
    .

        Parameters
        ----------
            adata: :class:`~anndata.AnnData`
                AnnData object.
            total_layers: list or None (default `None`)
                The layer(s) that can be summed up to get the total mRNA. for example, ["spliced", "unspliced"], ["uu",
                "ul", "su", "sl"] or ["new", "old"], etc.
            method: `str` (default: `log`)
                The linear dimension reduction methods to be used.
            num_dim: `int` (default: `50`)
                The number of linear dimensions reduced to.
            norm_method: `function`, `str` or `None` (default: function `None`)
                The method to normalize the data.
            pseudo_expr: `int` (default: `1`)
                A pseudocount added to the gene expression value before log/log2 normalization.
            feature_selection: `str` (default: `SVR`)
                Which sorting method, either dispersion, SVR or Gini index, to be used to select genes.
            n_top_genes: `int` (default: `2000`)
                How many top genes based on scoring method (specified by sort_by) will be selected as feature genes.
            cluster: `str`
                A column in the adata.obs attribute which will be used for cluster specific expression filtering.
            relative_expr: `bool` (default: `True`)
                A logic flag to determine whether we need to divide gene expression values first by size factor before
                normalization.
            keep_filtered_genes: `bool` (default: `True`)
                Whether to keep genes that don't pass the filtering in the adata object.

        Returns
        -------
            adata: :class:`~anndata.AnnData`
                An updated anndata object that are updated with Size_Factor, normalized expression values, X and reduced
                dimensions, etc.
    """

    keep_filtered_genes = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_genes, DynamoAdataConfig.RECIPE_KEEP_FILTERED_GENES_KEY
    )

    adata = szFactor(adata, method="mean", total_layers=total_layers)
    initial_Ucell_size = adata.layers["unspliced"].sum(1)

    filter_bool = initial_Ucell_size > np.percentile(initial_Ucell_size, 0.4)

    adata = filter_cells(adata, filter_bool=np.array(filter_bool).flatten())

    filter_bool = filter_genes_by_outliers(adata, min_cell_s=30, min_count_s=40, shared_count=None)

    adata = adata[:, filter_bool]

    adata = SVRs(
        adata,
        layers=["spliced"],
        min_expr_cells=2,
        max_expr_avg=35,
        min_expr_avg=0,
    )

    filter_bool = get_svr_filter(adata, layer="spliced", n_top_genes=n_top_genes)

    adata = adata[:, filter_bool]
    filter_bool_gene = filter_genes_by_outliers(
        adata,
        min_cell_s=0,
        min_count_s=0,
        min_count_u=25,
        min_cell_u=20,
        shared_count=None,
    )
    filter_bool_cluster = filter_genes_by_clusters_(adata, min_avg_S=0.08, min_avg_U=0.01, cluster=cluster)

    adata = adata[:, filter_bool_gene & filter_bool_cluster]

    adata = normalize_expr_data(
        adata,
        total_szfactor=None,
        norm_method=norm_method,
        pseudo_expr=pseudo_expr,
        relative_expr=relative_expr,
        keep_filtered=keep_filtered_genes,
    )
    CM = adata.X
    cm_genesums = CM.sum(axis=0)
    valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
    valid_ind = np.array(valid_ind).flatten()
    adata.var.use_for_pca[np.where(adata.var.use_for_pca)[0][~valid_ind]] = False

    CM = CM[:, valid_ind]

    if method == "pca":
        adata, fit, _ = pca(adata, CM, num_dim, "X_" + method.lower(), return_all=True)
        # adata.obsm['X_' + method.lower()] = reduce_dim

    elif method == "ica":
        cm_genesums = CM.sum(axis=0)
        valid_ind = (np.isfinite(cm_genesums)) + (cm_genesums != 0)
        valid_ind = np.array(valid_ind).flatten()
        CM = CM[:, valid_ind]

        fit = FastICA(
            num_dim,
            algorithm="deflation",
            tol=5e-6,
            fun="logcosh",
            max_iter=1000,
        )
        reduce_dim = fit.fit_transform(CM.toarray())

        adata.obsm["X_" + method.lower()] = reduce_dim

    add_noise_to_duplicates(adata, method.lower())
    adata.uns[method + "_fit"], adata.uns["feature_selection"] = (
        fit,
        feature_selection,
    )

    # calculate NTR for every cell:
    ntr = NTR(adata)
    if ntr is not None:
        adata.obs["ntr"] = ntr

    return adata


def highest_frac_genes(
    adata: AnnData,
    store_key: str = "highest_frac_genes",
    n_top: int = 30,
    gene_prefix_list: list = None,
    show_individual_prefix_gene: bool = False,
    layer: Union[str, None] = None,
):
    """
    Compute top genes df and store results in `adata.uns`


    Parameters
    ----------
    adata : AnnData
        [description]
    store_key : str, optional
        [description], by default "highest_frac_genes"
    n_top : int, optional
        [description], by default 30
    gene_prefix_list : list, optional
        [description], by default None
    show_individual_prefix_gene : bool, optional
        [description], by default False
    layer : Union[str, None], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    gene_mat = adata.X
    if layer is not None:
        gene_mat = adata.layers[layer]
    # compute gene percents at each cell row
    cell_expression_sum = gene_mat.sum(axis=1).flatten()
    # get rid of cells that have all zero counts
    not_all_zero = cell_expression_sum != 0
    adata = adata[not_all_zero, :]
    cell_expression_sum = cell_expression_sum[not_all_zero]
    main_info("%d rows(cells or subsets) are not zero. zero total RNA cells are removed." % np.sum(not_all_zero))

    valid_gene_set = set()
    prefix_to_genes = {}
    _adata = adata
    if gene_prefix_list is not None:
        prefix_to_genes = {prefix: [] for prefix in gene_prefix_list}
        for name in adata.var_names:
            for prefix in gene_prefix_list:
                length = len(prefix)
                if name[:length] == prefix:
                    valid_gene_set.add(name)
                    prefix_to_genes[prefix].append(name)
                    break
        if len(valid_gene_set) == 0:
            main_critical("NO VALID GENES FOUND WITH REQUIRED GENE PREFIX LIST, GIVING UP PLOTTING")
            return None
        if not show_individual_prefix_gene:
            # gathering gene prefix set data
            df = pd.DataFrame(index=adata.obs.index)
            for prefix in prefix_to_genes:
                if len(prefix_to_genes[prefix]) == 0:
                    main_info("There is no %s gene prefix in adata." % prefix)
                    continue
                df[prefix] = adata[:, prefix_to_genes[prefix]].X.sum(axis=1)
            # adata = adata[:, list(valid_gene_set)]

            _adata = AnnData(X=df)
            gene_mat = _adata.X

    # compute gene's total percents in the dataset
    gene_percents = np.array(gene_mat.sum(axis=0))
    gene_percents = (gene_percents / gene_mat.shape[1]).flatten()
    # obtain top genes
    sorted_indices = np.argsort(-gene_percents)
    selected_indices = sorted_indices[:n_top]
    gene_names = _adata.var_names[selected_indices]

    gene_X_percents = gene_mat / cell_expression_sum.reshape([-1, 1])

    # assemble a dataframe
    selected_gene_X_percents = np.array(gene_X_percents)[:, selected_indices]
    selected_gene_X_percents = np.squeeze(selected_gene_X_percents)

    top_genes_df = pd.DataFrame(
        selected_gene_X_percents,
        index=adata.obs_names,
        columns=gene_names,
    )

    adata.uns[store_key] = {
        "top_genes_df": top_genes_df,
        "gene_mat": gene_mat,
        "layer": layer,
        "selected_indices": selected_indices,
        "gene_prefix_list": gene_prefix_list,
        "show_individual_prefix_gene": show_individual_prefix_gene,
        "gene_percents": gene_percents,
    }

    return adata
