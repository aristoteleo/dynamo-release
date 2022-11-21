import warnings
from collections.abc import Iterable
from typing import Callable, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import FastICA
from sklearn.utils import sparsefuncs

from ..configuration import DKM, DynamoAdataConfig, DynamoAdataKeyManager
from ..dynamo_logger import (
    LoggerManager,
    main_critical,
    main_info,
    main_info_insert_adata_obsm,
    main_info_insert_adata_uns,
    main_warning,
)
from ..tools.utils import update_dict
from ..utils import copy_adata
from .cell_cycle import cell_cycle_scores
from .preprocess_monocle_utils import top_table
from .preprocessor_utils import (
    SVRs,
    _infer_labeling_experiment_type,
    filter_cells_by_outliers,
    filter_genes_by_outliers,
    normalize_cell_expr_by_size_factors,
    select_genes_monocle,
)
from .utils import (
    Freeman_Tukey,
    add_noise_to_duplicates,
    basic_stats,
    calc_new_to_total_ratio,
    clusters_stats,
    collapse_species_adata,
    compute_gene_exp_fraction,
    convert2symbol,
    convert_layers2csr,
    cook_dist,
    detect_experiment_datatype,
    get_inrange_shared_counts_mask,
    get_svr_filter,
    get_sz_exprs,
    merge_adata_attrs,
    normalize_mat_monocle,
    pca_monocle,
    sz_util,
    unique_var_obs_adata,
)


def calc_sz_factor_legacy(
    adata_ori: anndata.AnnData,
    layers: Union[str, list] = "all",
    total_layers: Union[list, None] = None,
    splicing_total_layers: bool = False,
    X_total_layers: bool = False,
    locfunc: Callable = np.nanmean,
    round_exprs: bool = False,
    method: Literal["mean-geometric-mean-total", "geometric", "median"] = "median",
    scale_to: Union[float, None] = None,
    use_all_genes_cells: bool = True,
    genes_use_for_norm: Union[list, None] = None,
) -> anndata.AnnData:
    """Calculate the size factor of the each cell using geometric mean of total UMI across cells for a AnnData object.

    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Args:
        adata_ori: an AnnData object.
        layers: the layer(s) to be normalized, including RNA (X, raw) or spliced, unspliced, protein, etc. Defaults to
            "all".
        total_layers: the layer(s) that can be summed up to get the total mRNA. For example, ["spliced", "unspliced"],
            ["uu", "ul", "su", "sl"] or ["new", "old"], etc. Defaults to None.
        splicing_total_layers: whether to also normalize spliced / unspliced layers by size factor from total RNA.
            Defaults to False.
        X_total_layers: whether to also normalize adata.X by size factor from total RNA. Defaults to False.
        locfunc: the function to normalize the data. Defaults to np.nanmean.
        round_exprs: whether the gene expression should be rounded into integers. Defaults to False.
        method: the method used to calculate the expected total reads / UMI used in size factor calculation. Only
            `mean-geometric-mean-total` / `geometric` and `median` are supported. When `median` is used, `locfunc` will
            be replaced with `np.nanmedian`. Defaults to "median".
        scale_to: the final total expression for each cell that will be scaled to. Defaults to None.
        use_all_genes_cells: whether all cells and genes should be used for the size factor calculation. Defaults to
            True.
        genes_use_for_norm: a list of gene names that will be used to calculate total RNA for each cell and then the
            size factor for normalization. This is often very useful when you want to use only the host genes to
            normalize the dataset in a virus infection experiment (i.e. CMV or SARS-CoV-2 infection). Defaults to None.

    Returns:
        An updated anndata object that are updated with the `Size_Factor` (`layer_` + `Size_Factor`) column(s) in the
        obs attribute.
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
            if type(layers) is str:
                layers = [layers]
            layers.extend(["_total_"])

    layers = DynamoAdataKeyManager.get_available_layer_keys(adata, layers)
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


def normalize_cell_expr_by_size_factors_legacy(
    adata: anndata.AnnData,
    layers: str = "all",
    total_szfactor: str = "total_Size_Factor",
    splicing_total_layers: bool = False,
    X_total_layers: bool = False,
    norm_method: Union[Callable, Literal["clr"], None] = None,
    pseudo_expr: int = 1,
    relative_expr: bool = True,
    keep_filtered: bool = True,
    recalc_sz: bool = False,
    sz_method: str = "median",
    scale_to: Union[float, None] = None,
) -> anndata.AnnData:
    """Normalize the gene expression value for the AnnData object

    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Args:
        adata: an AnnData object
        layers: the layer(s) to be normalized. Default is all, including RNA (X, raw) or spliced, unspliced, protein,
            etc. Defaults to "all".
        total_szfactor: the column name in the .obs attribute that corresponds to the size factor for the total mRNA.
            Defaults to "total_Size_Factor".
        splicing_total_layers: whether to also normalize spliced / unspliced layers by size factor from total RNA.
            Defaults to False.
        X_total_layers: whether to also normalize adata.X by size factor from total RNA. Defaults to False.
        norm_method: the method used to normalize data. Can be either function `np.log1p`, `np.log2` or any other
            functions or string `clr`. By default, only .X will be size normalized and log1p transformed while data in
            other layers will only be size normalized. Defaults to None.
        pseudo_expr: a pseudocount added to the gene expression value before log/log2 normalization. Defaults to 1.
        relative_expr:  a logic flag to determine whether we need to divide gene expression values first by size factor
            before normalization. Defaults to True.
        keep_filtered: a logic flag to determine whether we will only store feature genes in the adata object. If it is
            False, size factor will be recalculated only for the selected feature genes. Defaults to True.
        recalc_sz: a logic flag to determine whether we need to recalculate size factor based on selected genes before
            normalization. Defaults to False.
        sz_method: the method used to calculate the expected total reads / UMI used in size factor calculation. Only
            `mean-geometric-mean-total` / `geometric` and `median` are supported. When `median` is used, `locfunc` will
            be replaced with `np.nanmedian`. Defaults to "median".
        scale_to: the final total expression for each cell that will be scaled to. Defaults to None.

    Returns:
        An updated anndata object that are updated with normalized expression values for different layers.
    """

    if recalc_sz:
        if "use_for_pca" in adata.var.columns and keep_filtered is False:
            adata = adata[:, adata.var.loc[:, "use_for_pca"]]

        adata.obs = adata.obs.loc[:, ~adata.obs.columns.str.contains("Size_Factor")]

    layers = DynamoAdataKeyManager.get_available_layer_keys(adata, layers)
    layer_sz_column_names = [i + "_Size_Factor" for i in set(layers).difference("X")]
    layer_sz_column_names.extend(["Size_Factor"])
    layers_to_sz = list(set(layer_sz_column_names).difference(adata.obs.keys()))

    if len(layers_to_sz) > 0:
        layers = pd.Series(layers_to_sz).str.split("_Size_Factor", expand=True).iloc[:, 0].tolist()
        if "Size_Factor" in layers:
            layers[np.where(np.array(layers) == "Size_Factor")[0][0]] = "X"
        calc_sz_factor_legacy(
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
            CM = normalize_mat_monocle(CM, szfactors, relative_expr, pseudo_expr, np.log1p)
        elif norm_method in [np.log1p, np.log, np.log2, Freeman_Tukey, None] and layer != "protein":
            CM = normalize_mat_monocle(CM, szfactors, relative_expr, pseudo_expr, norm_method)
        elif layer == "protein":  # norm_method == 'clr':
            if norm_method != "clr":
                main_warning(
                    "For protein data, log transformation is not recommended. Using clr normalization by default."
                )
            """This normalization implements the centered log-ratio (CLR) normalization from Seurat which is computed
            for each gene (M Stoeckius, 2017).
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
            main_info("Set <adata.X> to normalized data")
            adata.X = CM
        elif layer == "protein" and "protein" in adata.obsm_keys():
            main_info_insert_adata_obsm("X_protein")
            adata.obsm["X_protein"] = CM
        else:
            adata.layers["X_" + layer] = CM

        adata.uns["pp"]["norm_method"] = norm_method.__name__ if callable(norm_method) else norm_method

    return adata


def Gini(adata: anndata.AnnData, layers: Union[Literal["all"], List[str]] = "all") -> anndata.AnnData:
    """Calculate the Gini coefficient of a numpy array. https://github.com/thomasmaxwellnorman/perturbseq_demo/blob/master/perturbseq/util.py

    Args:
        adata: an AnnData object
        layers: the layer(s) to be normalized. Defaults to "all".

    Returns:
        An updated anndata object with gini score for the layers (include .X) in the corresponding var columns (layer + '_gini').
    """

    # From: https://github.com/oliviaguest/gini
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    layers = DynamoAdataKeyManager.get_available_layer_keys(adata, layers)

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


def disp_calc_helper_NB(
    adata: anndata.AnnData, layers: str = "X", min_cells_detected: int = 1
) -> Tuple[List[str], List[pd.DataFrame]]:
    """Calculate the dispersion parameter of the negative binomial distribution.

    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Args:
        adata: an adata object
        layers: the layer of data used for dispersion fitting. Defaults to "X".
        min_cells_detected: the minimal required number of cells with expression for selecting gene for dispersion
            fitting. Defaults to 1.

    Returns:
        A tuple (layers, res_list), where layers is a list of layers available and res_list is a list of pd.DataFrame's
        with mu, dispersion for each gene that passes filters.
    """

    layers = DynamoAdataKeyManager.get_available_layer_keys(adata, layers=layers, include_protein=False)

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
            (x.multiply(x).mean(0).A1 - f_expression_mean.A1**2) * x.shape[0] / (x.shape[0] - 1)
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


def vstExprs(
    adata: anndata.AnnData,
    expr_matrix: Union[np.ndarray, None] = None,
    round_vals: bool = True,
) -> np.ndarray:
    """Variance stabilization transformation of the gene expression.

    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Args:
        adata: an AnnData object.
        expr_matrix: an matrix of values to transform. Must be normalized (e.g. by size factors) already. Defaults to
            None.
        round_vals: whether to round expression values to the nearest integer before applying the transformation.
            Defaults to True.

    Returns:
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


def filter_cells_legacy(
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

    Args:
        adata: an AnnData object
        filter_bool: a boolean array from the user to select cells for downstream analysis. Defaults to None.
        layer: the data from a particular layer (include X) used for feature selection. Defaults to "all".
        keep_filtered: whether to keep cells that don't pass the filtering in the adata object. Defaults to False.
        min_expr_genes_s: minimal number of genes with expression for a cell in the data from the spliced layer (also
            used for X). Defaults to 50.
        min_expr_genes_u: minimal number of genes with expression for a cell in the data from the unspliced layer.
            Defaults to 25.
        min_expr_genes_p: minimal number of genes with expression for a cell in the data from the protein layer.
            Defaults to 1.
        max_expr_genes_s:  maximal number of genes with expression for a cell in the data from the spliced layer (also
            used for X). Defaults to np.inf.
        max_expr_genes_u: maximal number of genes with expression for a cell in the data from the unspliced layer.
            Defaults to np.inf.
        max_expr_genes_p: maximal number of protein with expression for a cell in the data from the protein layer.
            Defaults to np.inf.
        shared_count: the minimal shared number of counts for each cell across genes between layers. Defaults to None.

    Returns:
        An updated AnnData object with `pass_basic_filter` as a new column in .var attribute to indicate the selection
        of cells for downstream analysis. adata will be subsetted with only the cells pass filtering if keep_filtered is
        set to be False.
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
        layers = DynamoAdataKeyManager.get_available_layer_keys(adata, layer, False)
        detected_bool = detected_bool & get_inrange_shared_counts_mask(adata, layers, shared_count, "cell")

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
) -> np.ndarray:
    """Prepare filtering genes on the basis of cluster-wise expression threshold.

    This function is taken from velocyto in order to reproduce velocyto's DentateGyrus notebook.

    Args:
        adata: an Anndata object.
        cluster: a column name in the adata.obs attribute which will be used for cluster specific expression filtering.
        min_avg_U: include genes that have unspliced average bigger than `min_avg_U` in at least one of the clusters.
            Defaults to 0.02.
        min_avg_S: include genes that have spliced average bigger than `min_avg_U` in at least one of the clusters.
            Defaults to 0.08.
        size_limit: the max number of members to be considered in a cluster during calculation. Defaults to 40.

    Returns:
        A boolean array corresponding to genes selected.
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


def filter_genes_by_outliers_legacy(
    adata: anndata.AnnData,
    filter_bool: Union[np.ndarray, None] = None,
    layer: str = "all",
    min_cell_s: int = 1,
    min_cell_u: int = 1,
    min_cell_p: int = 1,
    min_avg_exp_s: float = 1e-10,
    min_avg_exp_u: float = 0,
    min_avg_exp_p: float = 0,
    max_avg_exp: float = np.inf,
    min_count_s: int = 0,
    min_count_u: int = 0,
    min_count_p: int = 0,
    shared_count: int = 30,
) -> anndata.AnnData:
    """Basic filter of genes based a collection of expression filters.

    Args:
        adata: an Anndata object
        filter_bool: a boolean array from the user to select genes for downstream analysis. Defaults to None.
        layer: the data from a particular layer (include X) used for feature selection. Defaults to "all".
        min_cell_s: minimal number of cells with expression for the data in the spliced layer (also used for X).
            Defaults to 1.
        min_cell_u: minimal number of cells with expression for the data in the unspliced layer. Defaults to 1.
        min_cell_p: minimal number of cells with expression for the data in the protein layer. Defaults to 1.
        min_avg_exp_s: minimal average expression across cells for the data in the spliced layer (also used for X).
            Defaults to 1e-10.
        min_avg_exp_u: minimal average expression across cells for the data in the unspliced layer. Defaults to 0.
        min_avg_exp_p: minimal average expression across cells for the data in the protein layer. Defaults to 0.
        max_avg_exp: maximal average expression across cells for the data in all layers (also used for X). Defaults to
            np.inf.
        min_count_s: minimal number of counts (UMI/expression) for the data in the spliced layer (also used for X).
            Defaults to 0.
        min_count_u: minimal number of counts (UMI/expression) for the data in the unspliced layer. Defaults to 0.
        min_count_p: minimal number of counts (UMI/expression) for the data in the protein layer. Defaults to 0.
        shared_count: the minimal shared number of counts for each genes across cell between layers. Defaults to 30.

    Returns:
        An updated AnnData object with use_for_pca as a new column in .var attributes to indicate the selection of genes
        for downstream analysis. adata will be subsetted with only the genes pass filter if keep_unflitered is set to be
        False.
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
        layers = DynamoAdataKeyManager.get_available_layer_keys(adata, "all", False)
        tmp = get_inrange_shared_counts_mask(adata, layers, shared_count, "gene")
        if tmp.sum() > 2000:
            detected_bool &= tmp
        else:
            # in case the labeling time is very short for pulse experiment or
            # chase time is very long for degradation experiment.
            tmp = get_inrange_shared_counts_mask(
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
                & (adata.layers["protein"].sum(0) >= min_count_p)  # TODO potential bug confirmation: obsm?
            ).flatten()
        )

    filter_bool = filter_bool & detected_bool if filter_bool is not None else detected_bool

    adata.var["pass_basic_filter"] = np.array(filter_bool).flatten()

    return adata


def recipe_monocle(
    adata: anndata.AnnData,
    reset_X: bool = False,
    tkey: Union[str, None] = None,
    t_label_keys: Union[str, List[str], None] = None,
    experiment_type: Optional[str] = None,
    normalized: Union[bool, None] = None,
    layer: Union[str, None] = None,
    total_layers: Union[bool, List[str], None] = None,
    splicing_total_layers: bool = False,
    X_total_layers: bool = False,
    genes_use_for_norm: Union[List[str], None] = None,
    genes_to_use: Union[List[str], None] = None,
    genes_to_append: Union[List[str], None] = None,
    genes_to_exclude: Union[List[str], None] = None,
    exprs_frac_for_gene_exclusion: float = 1,
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
    keep_filtered_cells: Union[bool, None] = None,
    keep_filtered_genes: Union[bool, None] = None,
    keep_raw_layers: Union[bool, None] = None,
    scopes: Union[str, Iterable, None] = None,
    fc_kwargs: Union[dict, None] = None,
    fg_kwargs: Union[dict, None] = None,
    sg_kwargs: Union[dict, None] = None,
    copy: bool = False,
    feature_selection_layer: Union[List[str], np.ndarray, np.array, str] = DKM.X_LAYER,
) -> Union[anndata.AnnData, None]:
    """The monocle style preprocessing recipe.

    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Args:
        adata: an AnnData object.
        reset_X: whether do you want to let dynamo reset `adata.X` data based on layers stored in your experiment. One
            critical functionality of dynamo is about visualizing RNA velocity vector flows which requires proper data
            into which the high dimensional RNA velocity vectors will be projected.
                (1) For `kinetics` experiment, we recommend the use of `total` layer as `adata.X`;
                (2) For `degradation/conventional` experiment scRNA-seq, we recommend using `splicing` layer as `adata.X`.
            Set `reset_X` to `True` to set those default values if you are not sure. Defaults to False.
        tkey: the column key for the labeling time  of cells in .obs. Used for labeling based scRNA-seq data (will also
            support for conventional scRNA-seq data). Note that `tkey` will be saved to adata.uns['pp']['tkey'] and used
            in `dyn.tl.dynamics` in which when `group` is None, `tkey` will also be used for calculating  1st/2st moment
            or covariance. We recommend to use hour as the unit of `time`. Defaults to None.
        t_label_keys: the column key(s) for the labeling time label of cells in .obs. Used for either "conventional" or
            "labeling based" scRNA-seq data. Not used for now and `tkey` is implicitly assumed as `t_label_key`
            (however, `tkey` should just be the time of the experiment). Defaults to None.
        experiment_type: experiment type for labeling single cell RNA-seq. Available options are:
            (1) 'conventional': conventional single-cell RNA-seq experiment, if `experiment_type` is `None` and there is
                only splicing data, this will be set to `conventional`;
            (2) 'deg': chase/degradation experiment. Cells are first labeled with an extended period, followed by chase;
            (3) 'kin': pulse/synthesis/kinetics experiment. Cells are labeled for different duration in a time-series;
            (4) 'one-shot': one-shot kinetic experiment. Cells are only labeled for a short pulse duration;
            Other possible experiments include:
            (5) 'mix_pulse_chase' or 'mix_kin_deg': This is a mixture chase experiment in which the entire experiment
            lasts for a certain period of time with an initial pulse followed by washing out at different time point but
            chasing cells at the same time point. This type of labeling strategy was adopted in scEU-seq paper. For this
            kind of experiment, we need to assume a non-steady state dynamics.
            (6) 'mix_std_stm';. Defaults to None.
        normalized: if you already normalized your data (or run recipe_monocle already), set this to be `True` to avoid
            renormalizing your data. By default it is set to be `None` and the first 20 values of adata.X (if adata.X is
            sparse) or its first column will be checked to determine whether you already normalized your data. This only
            works for UMI based or read-counts data. Defaults to None.
        layer: the layer(s) to be normalized. if not supplied, all layers would be used, including RNA (X, raw) or
            spliced, unspliced, protein, etc. Defaults to None.
        total_layers: the layer(s) that can be summed up to get the total mRNA. for example, ["spliced", "unspliced"],
            ["uu", "ul", "su", "sl"] or ["total"], etc. If total_layers is `True`, total_layers will be set to be
            `total` or ["uu", "ul", "su", "sl"] depends on whether you have labeling but no splicing or labeling and
            splicing data. Defaults to None.
        splicing_total_layers: whether to also normalize spliced / unspliced layers by size factor fromtotal RNA.
            Defaults to False.
        X_total_layers: whether to also normalize adata.X by size factor from total RNA. Defaults to False.
        genes_use_for_norm: a list of gene names that will be used to calculate total RNA for each cell and then the
            size factor for normalization. This is often very useful when you want to use only the host genes to
            normalize the dataset in a virus infection experiment (i.e. CMV or SARS-CoV-2 infection). Defaults to None.
        genes_to_use: a list of gene names that will be used to set as the feature genes for downstream analysis.
            Defaults to None.
        genes_to_append: a list of gene names that will be appended to the feature genes list for downstream analysis.
            Defaults to None.
        genes_to_exclude: a list of gene names that will be excluded to the feature genes list for downstream analysis.
            Defaults to None.
        exprs_frac_for_gene_exclusion: the minimal fraction of gene counts to the total counts across cells that will
            used to filter genes. By default it is 1 which means we don't filter any genes, but we need to change it to
            0.005 or something in order to remove some highly expressed housekeeping genes. Defaults to 1.
        method: the linear dimension reduction methods to be used. Defaults to "pca".
        num_dim: the number of linear dimensions reduced to. Defaults to 30.
        sz_method: the method used to calculate the expected total reads / UMI used in size factor calculation. Only
            `mean-geometric-mean-total` / `geometric` and `median` are supported. When `median` is used, `locfunc` will
            be replaced with `np.nanmedian`. Defaults to "median".
        scale_to: the final total expression for each cell that will be scaled to. Defaults to None.
        norm_method: the method to normalize the data. Can be any numpy function or `Freeman_Tukey`. By default, only
            .X will be size normalized and log1p transformed while data in other layers will only be size factor
            normalized. Defaults to None.
        pseudo_expr: a pseudocount added to the gene expression value before log/log2 normalization. Defaults to 1.
        feature_selection: Which sorting method, either dispersion, SVR or Gini index, to be used to select genes.
            Defaults to "SVR".
        n_top_genes:  how many top genes based on scoring method (specified by sort_by) will be selected as feature
            genes. Defaults to 2000.
        maintain_n_top_genes: whether to ensure 2000 feature genes selected no matter what genes_to_use,
            genes_to_append, etc. are specified. The only exception is that if `genes_to_use` is supplied with
            `n_top_genes`. Defaults to True.
        relative_expr: whether we need to divide gene expression values first by size factor before normalization.
            Defaults to True.
        keep_filtered_cells: whether to keep genes that don't pass the filtering in the returned adata object.
            Defaults to None.
        keep_filtered_genes: whether to keep genes that don't pass the filtering in the returned adata object.
            Defaults to None.
        keep_raw_layers: whether to keep layers with raw measurements in the returned adata object. Defaults to None.
        scopes: scopes are needed when you use non-official gene name as your gene indices (or adata.var_name). This
            argument corresponds to types of identifiers, either a list or a comma-separated fields to specify type of
            input qterms, e.g. “entrezgene”, “entrezgene,symbol”, [“ensemblgene”, “symbol”]. Refer to official
            MyGene.info docs (https://docs.mygene.info/en/latest/doc/query_service.html#available_fields) for the full
            list of fields. Defaults to None.
        fc_kwargs: other Parameters passed into the filter_cells function. Defaults to None.
        fg_kwargs: other Parameters passed into the filter_genes function. Defaults to None.
        sg_kwargs: other Parameters passed into the select_genes function. Defaults to None.
        copy: whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments.
            Defaults to False.
        feature_selection_layer: name of layers to apply feature selection. Defaults to DKM.X_LAYER.

    Raises:
        ValueError: time key does not existed in adata.obs.
        ValueError: provided experiment type is invalid.
        Exception: no genes pass basic filter.
        Exception: no cells pass basic filter.
        Exception: genes_to_use contains genes that are not found in adata.
        ValueError: provided layer(s) is invalid.
        ValueError: genes_to_append contains invalid genes.

    Returns:
        A new updated anndata object if `copy` arg is `True`. In the object, Size_Factor, normalized expression values,
        X and reduced dimensions, etc., are updated. Otherwise, return None.
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

    # new/total, splicing/unsplicing
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
    adata = convert_layers2csr(adata)
    logger.info("ensure all labeling data properly collapased", indent_level=1)
    adata = collapse_species_adata(adata)

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

    _has_szFactor_normalized, _has_log1p_transformed = (True, True) if normalized else (False, False)
    if normalized is None and not has_labeling:
        # if has been flagged as preprocessed or not
        if "raw_data" in adata.uns_keys():
            _has_szFactor_normalized, _has_log1p_transformed = (
                not adata.uns["raw_data"],
                not adata.uns["raw_data"],
            )
        else:
            # automatically detect whether the data is size-factor normalized -- no integers (only works for readcounts
            # / UMI based data).
            _has_szFactor_normalized = not np.allclose(
                (adata.X.data[:20] if issparse(adata.X) else adata.X[:, 0]) % 1,
                0,
                atol=1e-3,
            )
            # check whether total UMI is the same -- if not the same, logged
            if _has_szFactor_normalized:
                _has_log1p_transformed = not np.allclose(
                    np.sum(adata.X.sum(1)[np.random.choice(adata.n_obs, 10)] - adata.X.sum(1)[0]),
                    0,
                    atol=1e-1,
                )

        if _has_szFactor_normalized or _has_log1p_transformed:
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
    adata = filter_cells_legacy(adata, keep_filtered=keep_filtered_cells, **filter_cells_kwargs)
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
    adata = filter_genes_by_outliers_legacy(
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
    if not _has_szFactor_normalized or "Size_Factor" not in adata.obs_keys():
        adata = calc_sz_factor_legacy(
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
        adata = select_genes_monocle_legacy(
            adata,
            layer=feature_selection_layer,
            sort_by=feature_selection,
            n_top_genes=n_top_genes,
            keep_filtered=True,  # TODO double check if should comply with the argument keep_filtered_genes
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
    adata.var["frac"], invalid_ids = compute_gene_exp_fraction(X=adata.X, threshold=exprs_frac_for_gene_exclusion)
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
        exclude_genes = adata.var.index.intersection(genes_to_exclude)
        if len(exclude_genes) > 0:
            adata.var.loc[exclude_genes, "use_for_pca"] = False

        if adata.var.use_for_pca.sum() < 50 and not maintain_n_top_genes:
            main_warning(
                "You only have less than 50 feature gene selected. Are you sure you want to exclude all "
                "genes passed to the genes_to_exclude argument?"
            )

    if maintain_n_top_genes:
        extra_n_top_genes = n_top_genes
        if genes_to_append is not None:
            extra_n_top_genes = n_top_genes - len(genes_to_append)
            valid_ids = adata.var.index.difference(genes_to_exclude + genes_to_append)
        else:
            valid_ids = adata.var.index.difference(genes_to_exclude)

        if extra_n_top_genes > 0:
            # let us ignore the `inplace` parameter in pandas.Categorical.remove_unused_categories  warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                filter_bool = select_genes_monocle_legacy(
                    adata[:, valid_ids],
                    sort_by=feature_selection,
                    n_top_genes=extra_n_top_genes,
                    keep_filtered=True,  # no effect to adata
                    SVRs_kwargs=select_genes_dict,
                    only_bools=True,
                )

            adata.var.loc[valid_ids, "use_for_pca"] = filter_bool

    if not keep_filtered_genes:
        logger.info("Discarding genes that failed the filtering...")
        adata._inplace_subset_var(adata.var["use_for_pca"])

    # normalized data based on sz factor
    if not _has_log1p_transformed:
        total_szfactor = "total_Size_Factor" if total_layers is not None else None
        logger.info("size factor normalizing the data, followed by log1p transformation.")
        adata = normalize_cell_expr_by_size_factors_legacy(
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
        layers = DynamoAdataKeyManager.get_available_layer_keys(adata, "all")
        for layer in layers:
            if layer != "X":
                logger.info_insert_adata("X_" + layer, "layers")
                adata.layers["X_" + layer] = adata.layers[layer].copy()
        logger.info_insert_adata("norm_method", "uns['pp']", indent_level=2)
        adata.uns["pp"]["norm_method"] = None

    # only use genes pass filter (based on use_for_pca) to perform dimension reduction.
    if layer is None:
        pca_input = adata.X[:, adata.var.use_for_pca.values]
    else:
        if "X" in layer:
            pca_input = adata.X[:, adata.var.use_for_pca.values]
        elif "total" in layer:
            pca_input = adata.layers["X_total"][:, adata.var.use_for_pca.values]
        elif "spliced" in layer:
            pca_input = adata.layers["X_spliced"][:, adata.var.use_for_pca.values]
        elif "protein" in layer:
            pca_input = adata.obsm["X_protein"]
        elif type(layer) is str:
            pca_input = adata.layers["X_" + layer][:, adata.var.use_for_pca.values]
        else:
            raise ValueError(
                f"your input layer argument should be either a `str` or a list that includes one of `X`, "
                f"`total`, `protein` element. `Layer` currently is {layer}."
            )

    pca_input_genesums = pca_input.sum(axis=0)
    valid_ind = np.logical_and(np.isfinite(pca_input_genesums), pca_input_genesums != 0)
    valid_ind = np.array(valid_ind).flatten()

    bad_genes = np.where(adata.var.use_for_pca)[0][~valid_ind]
    if genes_to_append is not None and len(adata.var.index[bad_genes].intersection(genes_to_append)) > 0:
        raise ValueError(
            f"The gene list passed to argument genes_to_append contains genes with no expression "
            f"across cells or non finite values. Please check those genes:"
            f"{set(bad_genes).intersection(genes_to_append)}!"
        )

    adata.var.iloc[bad_genes, adata.var.columns.tolist().index("use_for_pca")] = False
    pca_input = pca_input[:, valid_ind]
    logger.info("applying %s ..." % (method.upper()))

    if method == "pca":
        adata = pca_monocle(adata, pca_input, num_dim, "X_" + method.lower())
        # TODO remove adata.obsm["X"] in future, use adata.obsm.X_pca instead
        adata.obsm["X"] = adata.obsm["X_" + method.lower()]

    elif method == "ica":
        fit = FastICA(
            num_dim,
            algorithm="deflation",
            tol=5e-6,
            fun="logcosh",
            max_iter=1000,
        )
        reduce_dim = fit.fit_transform(pca_input.toarray())

        adata.obsm["X_" + method.lower()] = reduce_dim
        adata.obsm["X"] = adata.obsm["X_" + method.lower()]

    logger.info_insert_adata(method + "_fit", "uns")
    adata.uns[method + "_fit"], adata.uns["feature_selection"] = (
        {},
        feature_selection,
    )
    # calculate NTR for every cell:
    ntr, var_ntr = calc_new_to_total_ratio(adata)
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

    # flag adata as preprocessed by recipe_monocle
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
    total_layers: Union[List[str], None] = None,
    method: str = "pca",
    num_dim: int = 30,
    norm_method: Union[Callable, str, None] = None,
    pseudo_expr: int = 1,
    feature_selection: str = "SVR",
    n_top_genes: int = 2000,
    cluster: str = "Clusters",
    relative_expr: bool = True,
    keep_filtered_genes: Union[bool, None] = None,
) -> anndata.AnnData:
    """Velocyto's preprocess recipe.

    This function is adapted from the velocyto's DentateGyrus notebook.

    Args:
        adata: an AnnData object
        total_layers: the layer(s) that can be summed up to get the total mRNA. for example, ["spliced", "unspliced"],
            ["uu", "ul", "su", "sl"] or ["new", "old"], etc. Defaults to None.
        method: the linear dimension reduction methods to be used. Defaults to "pca".
        num_dim: the number of linear dimensions reduced to. Defaults to 30.
        norm_method: the method to normalize the data. Defaults to None.
        pseudo_expr: a pseudocount added to the gene expression value before log/log2 normalization. Defaults to 1.
        feature_selection: which sorting method, either dispersion, SVR or Gini index, to be used to select genes.
            Defaults to "SVR".
        n_top_genes: how many top genes based on scoring method (specified by sort_by) will be selected as feature
            genes. Defaults to 2000.
        cluster: a column in the adata.obs attribute which will be used for cluster specific expression filtering.
            Defaults to "Clusters".
        relative_expr: a logic flag to determine whether we need to divide gene expression values first by size factor
            before normalization. Defaults to True.
        keep_filtered_genes: whether to keep genes that don't pass the filtering in the adata object. Defaults to None.

    Returns:
        An updated anndata object that are updated with Size_Factor, normalized expression values, X and reduced
        dimensions, etc., according to the velocyto preprocessing recipe.
    """

    keep_filtered_genes = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_genes, DynamoAdataConfig.RECIPE_KEEP_FILTERED_GENES_KEY
    )

    adata = calc_sz_factor_legacy(adata, method="mean", total_layers=total_layers)
    initial_Ucell_size = adata.layers["unspliced"].sum(1)

    filter_bool = initial_Ucell_size > np.percentile(initial_Ucell_size, 0.4)

    adata = filter_cells_legacy(adata, filter_bool=np.array(filter_bool).flatten())

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

    adata = normalize_cell_expr_by_size_factors_legacy(
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
        adata, fit, _ = pca_monocle(adata, CM, num_dim, "X_" + method.lower(), return_all=True)
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
    ntr = calc_new_to_total_ratio(adata)
    if ntr is not None:
        adata.obs["ntr"] = ntr

    return adata


def highest_frac_genes(
    adata: AnnData,
    store_key: str = "highest_frac_genes",
    n_top: int = 30,
    gene_prefix_list: List[str] = None,
    gene_prefix_only: bool = False,
    layer: Union[str, None] = None,
) -> anndata.AnnData:
    """Compute top genes df and store results in `adata.uns`

    Args:
        adata: an AnnData object
        store_key: key for storing expression percent results. Defaults to "highest_frac_genes".
        n_top: number of top genes to show. Defaults to 30.
        gene_prefix_list: a list of gene name prefixes used for gathering/calculating expression percents from genes
            with these prefixes. Defaults to None.
        gene_prefix_only: whether to calculate percentages for gene groups with the specified prefixes only. It only
            takes effect if gene prefix list is provided. Defaults to False.
        layer: layer on which the gene percents will be computed. Defaults to None.

    Returns:
        An updated adata with top genes df stored in `adata.uns`
    """

    gene_mat = adata.X
    if layer is not None:
        gene_mat = DKM.select_layer_data(layer)
    # compute gene percents at each cell row
    cell_expression_sum = gene_mat.sum(axis=1).flatten()
    # get rid of cells that have all zero counts
    not_all_zero = cell_expression_sum != 0
    filtered_adata = adata[not_all_zero, :]
    cell_expression_sum = cell_expression_sum[not_all_zero]
    main_info("%d rows(cells or subsets) are not zero. zero total RNA cells are removed." % np.sum(not_all_zero))

    valid_gene_set = set()
    prefix_to_genes = {}
    _adata = filtered_adata
    if gene_prefix_list is not None:
        prefix_to_genes = {prefix: [] for prefix in gene_prefix_list}
        for name in _adata.var_names:
            for prefix in gene_prefix_list:
                length = len(prefix)
                if name[:length] == prefix:
                    valid_gene_set.add(name)
                    prefix_to_genes[prefix].append(name)
                    break
        if len(valid_gene_set) == 0:
            main_critical("NO VALID GENES FOUND WITH REQUIRED GENE PREFIX LIST, GIVING UP PLOTTING")
            return None
        if gene_prefix_only:
            # gathering gene prefix set data
            df = pd.DataFrame(index=_adata.obs.index)
            for prefix in prefix_to_genes:
                if len(prefix_to_genes[prefix]) == 0:
                    main_info("There is no %s gene prefix in adata." % prefix)
                    continue
                df[prefix] = _adata[:, prefix_to_genes[prefix]].X.sum(axis=1)
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
    gene_percents_df = pd.Series(gene_percents, index=_adata.var_names)

    main_info_insert_adata_uns(store_key)
    adata.uns[store_key] = {
        "top_genes_df": top_genes_df,
        "gene_mat": gene_mat,
        "layer": layer,
        "selected_indices": selected_indices,
        "gene_prefix_list": gene_prefix_list,
        "show_individual_prefix_gene": gene_prefix_only,
        "gene_percents": gene_percents_df,
    }

    return adata


def select_genes_monocle_legacy(
    adata: anndata.AnnData,
    layer: str = "X",
    total_szfactor: str = "total_Size_Factor",
    keep_filtered: bool = True,
    sort_by: str = "SVR",
    n_top_genes: int = 2000,
    SVRs_kwargs: dict = {},
    only_bools: bool = False,
    exprs_frac_for_gene_exclusion: float = 1,
    genes_to_exclude: Union[list, None] = None,
) -> anndata.AnnData:
    """Select feature genes based on a collection of filters.

    Args:
        adata: an AnnData object
        layer: the layer (include X) used for feature selection. Defaults to "X".
        total_szfactor: the column name in the .obs attribute that corresponds to the size factor for the total mRNA.
            Defaults to "total_Size_Factor".
        keep_filtered: whether to keep genes that don't pass the filtering in the adata object. Defaults to True.
        sort_by: which soring method, either SVR, dispersion or Gini index, to be used to select genes. Defaults to
            "SVR".
        n_top_genes: the number of top genes based on scoring method (specified by sort_by) will be selected as feature
            genes. Defaults to 2000.
        SVRs_kwargs: extra kwargs for `SVR` function. Defaults to {}.
        only_bools: whether only return a vector of bool values. Defaults to False.
        exprs_frac_for_gene_exclusion: threshold of fractions for high fraction genes. Defaults to 1.
        genes_to_exclude: list of gene names that are excluded from calculation. Defaults to None.

    Returns:
        An updated AnnData object with use_for_pca as a new column in .var attributes to indicate the selection of genes
        for downstream analysis. adata will be subsetted with only the genes pass filter if keep_unflitered is set to be
        False.
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

    # filter genes by gene expression fraction as well
    adata.var["frac"], invalid_ids = compute_gene_exp_fraction(X=adata.X, threshold=exprs_frac_for_gene_exclusion)
    genes_to_exclude = (
        list(adata.var_names[invalid_ids])
        if genes_to_exclude is None
        else genes_to_exclude + list(adata.var_names[invalid_ids])
    )
    if genes_to_exclude is not None and len(genes_to_exclude) > 0:
        adata_exclude_genes = adata.var.index.intersection(genes_to_exclude)
        adata.var.loc[adata_exclude_genes, "use_for_pca"] = False

    if keep_filtered:
        adata.var["use_for_pca"] = filter_bool
    else:
        adata._inplace_subset_var(filter_bool)
        adata.var["use_for_pca"] = True

    return filter_bool if only_bools else adata
