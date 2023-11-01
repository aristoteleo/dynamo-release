import warnings
from typing import Dict, List, Optional, Tuple, Union

from numpy import ndarray

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata
import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

from ..configuration import DKM
from ..dynamo_logger import (
    LoggerManager,
    main_critical,
    main_debug,
    main_info,
    main_info_insert_adata_uns,
    main_info_insert_adata_var,
    main_warning,
)
from .pca import pca
from .utils import (
    compute_gene_exp_fraction,
    get_gene_selection_filter,
    get_nan_or_inf_data_bool_mask,
    get_svr_filter,
    merge_adata_attrs,
    seurat_get_mean_var,
)


def calc_Gini(adata: AnnData, layers: Union[Literal["all"], List[str]] = "all") -> AnnData:
    """Calculate the Gini coefficient of a numpy array.
    https://github.com/thomasmaxwellnorman/perturbseq_demo/blob/master/perturbseq/util.py

    Args:
        adata: an AnnData object
        layers: the layer(s) to be normalized. Defaults to "all".

    Returns:
        An updated anndata object with gini score for the layers (include .X) in the corresponding var columns
        (layer + '_gini').
    """

    # From: https://github.com/oliviaguest/gini
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    layers = DKM.get_available_layer_keys(adata, layers)

    def _compute_gini(CM):
        # convert to dense array if sparse
        if issparse(CM):
            CM = CM.A

        # shift all values to be non-negative
        CM -= np.min(CM)

        # add small constant to avoid zeros
        CM = CM.astype(float) + 0.0000001  # values cannot be 0

        # sort values along axis 0
        CM = np.sort(CM, axis=0)

        # compute index array
        n = CM.shape[0]
        index = 2 * (np.arange(1, n + 1)) - n - 1

        # compute Gini coefficient for each feature
        gini = (np.sum(index[:, np.newaxis] * CM, axis=0)) / (n * np.sum(CM, axis=0))

        return gini

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

        var_gini = _compute_gini(CM)
        adata.var[layer + "_gini"] = var_gini

    return adata


def select_genes_monocle(
    adata: AnnData,
    layer: str = DKM.X_LAYER,
    keep_filtered: bool = True,
    n_top_genes: int = 2000,
    sort_by: Literal["gini", "cv_dispersion", "fano_dispersion"] = "cv_dispersion",
    exprs_frac_for_gene_exclusion: float = 1,
    genes_to_exclude: Union[List[str], None] = None,
    SVRs_kwargs: dict = {},
):
    """Select genes based on monocle recipe.

    This version is here for modularization of preprocessing, so that users may try combinations of different
    preprocessing procedures in Preprocessor.

    Args:
        adata: an AnnData object.
        layer: The data from a particular layer (include X) used for feature selection. Defaults to "X".
        keep_filtered: Whether to keep genes that don't pass the filtering in the adata object. Defaults to True.
        n_top_genes: the number of top genes based on scoring method (specified by sort_by) will be selected as feature
            genes. Defaults to 2000.
        sort_by: the sorting methods to be used to select genes. Should be one of the gini index or
            dispersion of coefficient variation or fano. Defaults to cv_dispersion.
        exprs_frac_for_gene_exclusion: threshold of fractions for high fraction genes. Defaults to 1.
        genes_to_exclude: genes that are excluded from evaluation. Defaults to None.
        SVRs_kwargs: kwargs for `SVRs`. Defaults to {}.

    Raises:
        NotImplementedError: the 'sort_by' algorithm is invalid/unsupported.
    """

    filter_bool = (
        adata.var["pass_basic_filter"]
        if "pass_basic_filter" in adata.var.columns
        else np.ones(adata.shape[1], dtype=bool)
    )

    if adata.shape[1] <= n_top_genes:
        filter_bool = np.ones(adata.shape[1], dtype=bool)
    else:
        if sort_by == "gini":
            if layer + "_gini" is not adata.var.keys():
                calc_Gini(adata)
            filter_bool = get_gene_selection_filter(
                adata.var[layer + "_gini"][filter_bool], n_top_genes=n_top_genes, basic_filter=filter_bool
            )
        elif sort_by == "cv_dispersion" or sort_by == "fano_dispersion":
            if not any("velocyto_SVR" in key for key in adata.uns.keys()):
                calc_dispersion_by_svr(
                    adata,
                    layers=layer,
                    filter_bool=filter_bool,
                    algorithm=sort_by,
                    **SVRs_kwargs,
                )
            filter_bool = get_svr_filter(adata, layer=layer, n_top_genes=n_top_genes, return_adata=False)
        else:
            raise NotImplementedError(f"The algorithm {sort_by} is invalid/unsupported")

    invalid_ids = []
    # filter genes by gene expression fraction as well
    if "frac" not in adata.var.keys():
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

    adata.uns["feature_selection"] = sort_by


def calc_dispersion_by_svr(
    adata_ori: AnnData,
    filter_bool: Union[np.ndarray, None] = None,
    layers: str = "X",
    algorithm: Literal["cv_dispersion", "fano_dispersion"] = "cv_dispersion",
    use_all_genes_cells: bool = False,
    **SVRs_kwargs,
) -> AnnData:
    """Support Vector Regression to identify highly variable genes.

    This function is modified from https://github.com/velocyto-team/velocyto.py/blob/master/velocyto/analysis.py

    Args:
        adata_ori: an AnnData object
        filter_bool: A boolean array from the user to select genes for downstream analysis. Defaults to None.
        layers: The layer(s) to be used for calculating dispersion score via support vector regression (SVR). Defaults
            to "X".
        algorithm: Method of calculating mean and coefficient of variation, either "cv_dispersion" or "fano_dispersion"
        sort_inverse: whether to sort genes from less noisy to more noisy (to use for size estimation not for feature
            selection). Defaults to False.
        use_all_genes_cells: A logic flag to determine whether all cells and genes should be used for the size factor
            calculation. Defaults to False.

    Returns:
        An updated annData object with `log_m`, `log_cv`, `score` added to .obs columns and `SVR` added to uns attribute
        as a new key.
    """

    layers = DKM.get_available_layer_keys(adata_ori, layers)
    winsorize = SVRs_kwargs.get("winsorize", False)
    winsor_perc = SVRs_kwargs.get("winsor_perc", (1, 99.5))
    svr_gamma = SVRs_kwargs.pop("svr_gamma", None)
    sort_inverse = SVRs_kwargs.pop("sort_inverse", False)

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

    for layer in layers:
        valid_CM, detected_bool = get_vaild_CM(adata, layer, **SVRs_kwargs)
        if valid_CM is None:
            continue

        mean, cv = get_mean_cv(adata, valid_CM, algorithm, winsorize, winsor_perc)
        fitted_fun, svr_gamma = get_prediction_by_svr(mean, cv, svr_gamma)
        score = cv - fitted_fun(mean)
        if sort_inverse:
            score = -score

        # Now we can get "SVR" from get_prediction_by_svr
        key = "velocyto_SVR" if layer == "raw" or layer == "X" else layer + "_velocyto_SVR"
        adata_ori.uns[key] = {"mean": mean, "cv": cv, "svr_gamma": svr_gamma}

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
            np.array(mean).flatten(),
            np.array(cv).flatten(),
            np.array(score).flatten(),
        )

    adata_ori = merge_adata_attrs(adata_ori, adata, attr="var")

    return adata_ori


def get_vaild_CM(
    adata: AnnData,
    layer: str = "X",
    relative_expr: bool = True,
    total_szfactor: str = "total_Size_Factor",
    min_expr_cells: int = 0,
    min_expr_avg: int = 0,
    max_expr_avg: int = np.inf,
    winsorize: bool = False,
    winsor_perc: Tuple[float, float] = (1, 99.5),
):
    """Find a valid CM that is the data of the layer corresponding to the size factor.

    Args:
        adata: an AnnData object.
        layer: The data from a particular layer (include X) used for feature selection. Defaults to "X".
        relative_expr: A logic flag to determine whether we need to divide gene expression values first by size factor
            before run SVR. Defaults to True.
        total_szfactor: The column name in the .obs attribute that corresponds to the size factor for the total mRNA.
            Defaults to "total_Size_Factor".
        min_expr_cells: minimum number of cells that express the gene for it to be considered in the fit. Defaults to 0.
        min_expr_avg: The minimum average of genes across cells required for gene to be selected for SVR analyses.
            Defaults to 0.
        max_expr_avg: The maximum average of genes across cells required for gene to be selected for SVR analyses. Genes
            with average gene expression larger than this value will be treated as house-keeping/outlier genes. Defaults
            to np.inf.
        winsorize: Weather to winsorize the data for the cv vs mean model. Defaults to False.
        winsor_perc: the up and lower bound of the winsorization. Defaults to (1, 99.5).

    Returns:
        An updated annData object with `log_m`, `log_cv`, `score` added to .obs columns and `SVR` added to uns attribute
        as a new key.
    """

    CM = None
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
        CM = adata.layers[layer].copy()
        szfactors = (
            adata.obs[layer + "_Size_Factor"].values[:, None] if layer + "_Size_Factor" in adata.obs.columns else None
        )

    if total_szfactor is not None and total_szfactor in adata.obs.keys():
        szfactors = adata.obs[total_szfactor].values[:, None] if total_szfactor in adata.obs.columns else None

    if szfactors is not None and relative_expr:
        if issparse(CM):
            from sklearn.utils import sparsefuncs

            sparsefuncs.inplace_row_scale(CM, 1 / szfactors)
        else:
            CM /= szfactors

    if winsorize:
        if min_expr_cells <= ((100 - winsor_perc[1]) * CM.shape[0] * 0.01):
            min_expr_cells = int(np.ceil((100 - winsor_perc[1]) * CM.shape[1] * 0.01)) + 2

    detected_bool = np.array(
        ((CM > 0).sum(0) >= min_expr_cells) & (CM.mean(0) <= max_expr_avg) & (CM.mean(0) >= min_expr_avg)
    ).flatten()

    return CM[:, detected_bool], detected_bool


def get_mean_cv(
    adata: AnnData,
    valid_CM: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, scipy.sparse.coo_matrix],
    algorithm: Literal["cv_dispersion", "fano_dispersion"] = "cv_dispersion",
    winsorize: bool = False,
    winsor_perc: Tuple[float, float] = (1, 99.5),
) -> AnnData:
    """Find the mean and coefficient of variation of gene expression.

    Args:
        adata: an AnnData object
        algorithm: Method of calculating mean and coefficient of variation, either fano_dispersion or cv_dispersion.
        valid_CM: Gene expression matrix to be used in a downstream analysis.
        winsorize: Whether to winsorize the data for the cv vs mean model. Defaults to False.
        winsor_perc: The up and lower bound of the winsorization. Defaults to (1, 99.5).

    Returns:
        mean: the array dataset that contains mean values of gene expression.
        cv: the array dataset with coefficient of variation of gene expression.
    """

    if algorithm == "fano_dispersion":
        (gene_counts_stats, gene_fano_parameters) = get_highvar_genes_sparse(adata, valid_CM)
        mean = np.array(gene_counts_stats["mean"]).flatten()[:, None]
        cv = np.array(gene_counts_stats["fano"]).flatten()
        return mean, cv
    elif algorithm == "cv_dispersion":
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
                        (valid_CM.multiply(valid_CM).mean(0).A1 - mu**2)
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
        return log_m[:, None], log_cv
    else:
        raise ValueError(f"The algorithm {algorithm} is not existed")


def get_prediction_by_svr(ground: np.ndarray, target: np.ndarray, svr_gamma: Optional[float] = None):
    """This function will return the base class for estimators that use libsvm as backing library.

    Args:
        ground: the training array dataset that contains mean values of gene expression.
        target: the target array dataset with coefficient of variation of gene expression.
        mean: the mean value to estimate a value of svr_gamma.
        svr_gamma: the gamma hyperparameter of the SVR. Defaults to None.

    Returns:
        A fitted SVM model according to the given training and target data.
    """
    from sklearn.svm import SVR

    if svr_gamma is None:
        svr_gamma = 150.0 / len(ground)

    # Fit the Support Vector Regression
    clf = SVR(gamma=svr_gamma)
    clf.fit(ground, target)
    return clf.predict, svr_gamma


# Highly variable gene selection function:
def get_highvar_genes_sparse(
    adata: AnnData,
    expression: Union[
        np.ndarray,
        scipy.sparse.csr_matrix,
        scipy.sparse.csc_matrix,
        scipy.sparse.coo_matrix,
    ],
    expected_fano_threshold: Optional[float] = None,
    numgenes: Optional[int] = None,
    minimal_mean: float = 0.5,
    save_key: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Find highly-variable genes in sparse single-cell data matrices.

    Args:
        adata: an AnnData object
        expression: Gene expression matrix
        expected_fano_threshold: Optionally can be used to set a manual dispersion threshold (for definition of
            "highly-variable")
        numgenes: Optionally can be used to find the n most variable genes
        minimal_mean: Sets a threshold on the minimum mean expression to consider
        save_key: the key to store the fano calculation results

    Returns:
        gene_counts_stats: Results dataframe containing pertinent information for each gene
        gene_fano_parameters: Additional informative dictionary (w/ records of dispersion for each gene, threshold,
        etc.)
    """
    gene_mean = np.array(expression.mean(axis=0)).astype(float).reshape(-1)
    E2 = expression.copy()
    E2.data **= 2
    gene2_mean = np.array(E2.mean(axis=0)).reshape(-1)
    gene_var = pd.Series(gene2_mean - (gene_mean**2))
    del E2
    gene_mean = pd.Series(gene_mean)
    gene_fano = gene_var / gene_mean

    # Find parameters for expected fano line -- this line can be non-linear...
    top_genes = gene_mean.sort_values(ascending=False)[:20].index
    A = (np.sqrt(gene_var) / gene_mean)[top_genes].min()

    w_mean_low, w_mean_high = gene_mean.quantile([0.10, 0.90])
    w_fano_low, w_fano_high = gene_fano.quantile([0.10, 0.90])
    winsor_box = (
        (gene_fano > w_fano_low) & (gene_fano < w_fano_high) & (gene_mean > w_mean_low) & (gene_mean < w_mean_high)
    )
    fano_median = gene_fano[winsor_box].median()
    B = np.sqrt(fano_median)

    gene_expected_fano = (A**2) * gene_mean + (B**2)
    fano_ratio = gene_fano / gene_expected_fano

    # Identify high var genes
    if numgenes is not None:
        highvargenes = fano_ratio.sort_values(ascending=False).index[:numgenes]
        high_var_genes_ind = fano_ratio.index.isin(highvargenes)
        T = None
    else:
        if not expected_fano_threshold:
            T = 1.0 + gene_fano[winsor_box].std()
        else:
            T = expected_fano_threshold

        high_var_genes_ind = (fano_ratio > T) & (gene_mean > minimal_mean)

    gene_counts_stats = pd.DataFrame(
        {
            "mean": gene_mean,
            "var": gene_var,
            "fano": gene_fano,
            "expected_fano": gene_expected_fano,
            "high_var": high_var_genes_ind,
            "fano_ratio": fano_ratio,
        }
    )
    gene_fano_parameters = {
        "A": A,
        "B": B,
        "T": T,
        "minimal_mean": minimal_mean,
    }

    if save_key is not None:
        LoggerManager.main_logger.info_insert_adata(save_key, "varm")
        gene_counts_stats.set_index(adata.var.index, inplace=True)
        adata.varm[save_key] = gene_counts_stats
    return gene_counts_stats, gene_fano_parameters


def select_genes_by_seurat_recipe(
    adata: AnnData,
    layer: str = DKM.X_LAYER,
    nan_replace_val: Union[float, None] = None,
    n_top_genes: int = 2000,
    algorithm: Literal["seurat_dispersion", "fano_dispersion"] = "seurat_dispersion",
    seurat_min_disp: Union[float, None] = None,
    seurat_max_disp: Union[float, None] = None,
    seurat_min_mean: Union[float, None] = None,
    seurat_max_mean: Union[float, None] = None,
    gene_names: Union[List[str], None] = None,
    var_filter_key: str = "pass_basic_filter",
    inplace: bool = False,
) -> None:
    """A general function for feature genes selection.

    Preprocess adata and dispatch to different filtering methods, and eventually set keys in anndata to denote which
    genes are wanted in downstream analysis.

    Args:
        adata: an AnnData object.
        layer: the key of a sparse matrix in adata. Defaults to DKM.X_LAYER.
        nan_replace_val: your choice of value to replace values in layer. Defaults to None.
        n_top_genes: number of genes to select as highly variable genes. Defaults to 2000.
        algorithm: a method for selecting genes; must be one of "seurat_dispersion" or "fano".
        seurat_min_disp: seurat dispersion min cutoff. Defaults to None.
        seurat_max_disp: seurat dispersion max cutoff. Defaults to None.
        seurat_min_mean: seurat mean min cutoff. Defaults to None.
        seurat_max_mean: seurat mean max cutoff. Defaults to None.
        gene_names: name of genes to be selected. Defaults to None.
        var_filter_key: filter gene names based on the key defined in adata.var before gene selection. Defaults to
            "pass_basic_filter".
        inplace: when inplace is True, subset adata according to selected genes. Defaults to False.

    Raises:
        NotImplementedError: the recipe is invalid/unsupported.
    """

    pass_filter_genes = adata.var_names
    if gene_names:
        main_info("select genes on gene names from arguments <gene_names>")
        pass_filter_genes = gene_names
    elif var_filter_key:
        main_info("select genes on var key: %s" % (var_filter_key))
        pass_filter_genes = adata.var_names[adata.var[var_filter_key]]

    if len(pass_filter_genes) != len(set(pass_filter_genes)):
        main_warning("gene names are not unique, please check your preprocessing procedure.")
    subset_adata = adata[:, pass_filter_genes]
    if n_top_genes is None:
        main_info("n_top_genes is None, reserve all genes and add filter gene information")
        n_top_genes = adata.n_vars
    layer_mat = DKM.select_layer_data(subset_adata, layer)
    if nan_replace_val:
        main_info("replacing nan values with: %s" % (nan_replace_val))
        _mask = get_nan_or_inf_data_bool_mask(layer_mat)
        layer_mat[_mask] = nan_replace_val

    if algorithm == "seurat_dispersion":
        mean, variance, highly_variable_mask = select_genes_by_seurat_dispersion(
            layer_mat,
            min_disp=seurat_min_disp,
            max_disp=seurat_max_disp,
            min_mean=seurat_min_mean,
            max_mean=seurat_max_mean,
            n_top_genes=n_top_genes,
        )
        main_info_insert_adata_var(DKM.VAR_GENE_MEAN_KEY)
        main_info_insert_adata_var(DKM.VAR_GENE_VAR_KEY)
        main_info_insert_adata_var(DKM.VAR_GENE_HIGHLY_VARIABLE_KEY)
        main_debug("type of variance:" + str(type(variance)))
        main_debug("shape of variance:" + str(variance.shape))
        adata.var[DKM.VAR_GENE_MEAN_KEY] = np.nan
        adata.var[DKM.VAR_GENE_VAR_KEY] = np.nan
        adata.var[DKM.VAR_GENE_HIGHLY_VARIABLE_KEY] = False
        adata.var[DKM.VAR_USE_FOR_PCA] = False

        adata.var[DKM.VAR_GENE_MEAN_KEY][pass_filter_genes] = mean.flatten()
        adata.var[DKM.VAR_GENE_VAR_KEY][pass_filter_genes] = variance
        adata.var[DKM.VAR_GENE_HIGHLY_VARIABLE_KEY][pass_filter_genes] = highly_variable_mask
        adata.var[DKM.VAR_USE_FOR_PCA][pass_filter_genes] = highly_variable_mask

    elif algorithm == "fano_dispersion":
        select_genes_monocle(adata, layer=layer, sort_by=algorithm)
        # adata = select_genes_by_svr(
        #     adata,
        #     layers=layer,
        #     algorithm=algorithm,
        # )
        # filter_bool = get_svr_filter(adata, layer=layer, n_top_genes=n_top_genes, return_adata=False)
    else:
        raise ValueError(f"The algorithm {algorithm} is not existed")

    main_info("number of selected highly variable genes: " + str(adata.var[DKM.VAR_USE_FOR_PCA].sum()))
    if inplace:
        main_info("inplace is True, subset adata according to selected genes.")
        adata = adata[:, adata.var[DKM.VAR_USE_FOR_PCA]]


def select_genes_by_seurat_dispersion(
    sparse_layer_mat: csr_matrix,
    n_bins: int = 20,
    log_mean_and_dispersion: bool = True,
    min_disp: float = None,
    max_disp: float = None,
    min_mean: float = None,
    max_mean: float = None,
    n_top_genes: Union[int, None] = None,
) -> Tuple[ndarray, ndarray, Union[bool, ndarray]]:
    """Apply seurat's gene selection recipe by cutoffs.

    Args:
        sparse_layer_mat: the sparse matrix used for gene selection.
        n_bins: the number of bins for normalization. Defaults to 20.
        log_mean_and_dispersion: whether log the gene expression values before calculating the dispersion values.
            Defaults to True.
        min_disp: seurat dispersion min cutoff. Defaults to None.
        max_disp: seurat dispersion max cutoff. Defaults to None.
        min_mean: seurat mean min cutoff. Defaults to None.
        max_mean: seurat mean max cutoff. Defaults to None.
        n_top_genes: number of top genes to be evaluated. If set to be None, genes are filtered by mean and dispersion
            norm threshold. Defaults to None.

    Returns:
        A tuple (mean, variance, highly_variable_mask, highly_variable_scores), where mean is the mean of the provided
        sparse matrix, variance is the variance of the provided sparse matrix, highly_variable_mask is a bool array
        indicating whether an element (a gene) is highly variable in the matrix. highly_variable_scores is always none
        since the scores are not applicable to Seurat recipe.
    """

    # default values from Seurat
    if min_disp is None:
        min_disp = 0.5
    if max_disp is None:
        max_disp = np.inf
    if min_mean is None:
        min_mean = 0.0125
    if max_mean is None:
        max_mean = 3

    # mean, variance, dispersion = calc_mean_var_dispersion_sparse(sparse_layer_mat) # Dead
    sc_mean, sc_var = seurat_get_mean_var(sparse_layer_mat)
    mean, variance = sc_mean, sc_var
    dispersion = variance / mean

    if log_mean_and_dispersion:
        mean = np.log1p(mean)
        dispersion[np.equal(dispersion, 0)] = np.nan
        dispersion = np.log(dispersion)

    temp_df = pd.DataFrame()
    temp_df["mean"], temp_df["dispersion"] = mean, dispersion

    temp_df["mean_bin"] = pd.cut(temp_df["mean"], bins=n_bins)
    disp_grouped = temp_df.groupby("mean_bin")["dispersion"]
    disp_mean_bin = disp_grouped.mean()
    disp_std_bin = disp_grouped.std(ddof=1)

    # handle nan std
    one_gene_per_bin = disp_std_bin.isnull()

    disp_std_bin[one_gene_per_bin] = disp_mean_bin[one_gene_per_bin].values
    disp_mean_bin[one_gene_per_bin] = 0

    # normalized dispersion
    mean = disp_mean_bin[temp_df["mean_bin"].values].values
    std = disp_std_bin[temp_df["mean_bin"].values].values
    variance = std**2
    temp_df["dispersion_norm"] = ((temp_df["dispersion"] - mean) / std).fillna(0)
    dispersion_norm = temp_df["dispersion_norm"].values

    highly_variable_mask = None
    if n_top_genes is not None:
        main_info("choose %d top genes" % n_top_genes, indent_level=2)
        threshold = temp_df["dispersion_norm"].nlargest(n_top_genes).values[-1]
        highly_variable_mask = temp_df["dispersion_norm"].values >= threshold
    else:
        main_info("choose genes by mean and dispersion norm threshold", indent_level=2)
        highly_variable_mask = np.logical_and.reduce(
            (
                mean > min_mean,
                mean < max_mean,
                dispersion_norm > min_disp,
                dispersion_norm < max_disp,
            )
        )

    return mean, variance, highly_variable_mask


def get_highly_variable_mask_by_dispersion_svr(
    mean: np.ndarray,
    var: np.ndarray,
    n_top_genes: int,
    svr_gamma: Optional[float] = None,
    return_scores: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Returns the mask with shape same as mean and var.

    The mask indicates whether each index is highly variable or not. Each index should represent a gene.

    Args:
        mean: mean of the genes.
        var: variance of the genes.
        n_top_genes: the number of top genes to be inspected.
        svr_gamma: coefficient for support vector regression. Defaults to None.
        return_scores: whether return the dispersion scores. Defaults to True.

    Returns:
        A tuple (highly_variable_mask, scores) where highly_variable_mask is a bool array indicating whether an element
        (a gene) is highly variable in the matrix and scores is an array recording variable score for each gene. scores
        would only be returned when `return_scores` is True.
    """

    # normally, select svr_gamma based on #features
    if svr_gamma is None:
        svr_gamma = 150.0 / len(mean)
    from sklearn.svm import SVR

    mean_log = np.log2(mean)
    cv_log = np.log2(np.sqrt(var) / mean)
    classifier = SVR(gamma=svr_gamma)
    # fit & prediction will complain about nan values if not take cared here
    is_nan_indices = np.logical_or(np.isnan(mean_log), np.isnan(cv_log))
    if np.sum(is_nan_indices) > 0:
        main_warning(
            (
                "mean and cv_log contain NAN values. We exclude them in SVR training. Please use related gene filtering"
                " methods to filter genes with zero means."
            )
        )

    classifier.fit(mean_log[~is_nan_indices, np.newaxis], cv_log.reshape([-1, 1])[~is_nan_indices])
    scores = np.repeat(np.nan, len(mean_log))
    # TODO handle nan values during prediction here
    scores[~is_nan_indices] = cv_log[~is_nan_indices] - classifier.predict(mean_log[~is_nan_indices, np.newaxis])
    scores = scores.reshape([-1, 1])  # shape should be #genes x 1

    # score threshold based on n top genes
    n_top_genes = min(n_top_genes, len(mean))  # maybe not enough genes there
    score_threshold = np.sort(-scores)[n_top_genes - 1]
    highly_variable_mask = scores >= score_threshold
    highly_variable_mask = np.array(highly_variable_mask).flatten()
    if return_scores:
        return highly_variable_mask, scores
    return highly_variable_mask


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
        gene_mat = DKM.select_layer_data(adata, layer)
    # compute gene percents at each cell row
    cell_expression_sum = gene_mat.sum(axis=1).flatten()
    # get rid of cells that have all zero counts
    not_all_zero = cell_expression_sum != 0
    filtered_adata = adata[not_all_zero, :]
    cell_expression_sum = cell_expression_sum[not_all_zero]
    main_debug("%d rows(cells or subsets) are not zero. zero total RNA cells are removed." % np.sum(not_all_zero))

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
    if issparse(gene_X_percents):
        selected_gene_X_percents = gene_X_percents.toarray()[:, selected_indices]
    elif type(gene_X_percents) == np.ndarray:
        selected_gene_X_percents = gene_X_percents[:, selected_indices]
    else:
        selected_gene_X_percents = np.array(gene_X_percents)[:, selected_indices]

    selected_gene_X_percents = np.squeeze(selected_gene_X_percents)

    top_genes_df = pd.DataFrame(
        selected_gene_X_percents,
        index=adata.obs_names,
        columns=gene_names,
    )
    gene_percents_df = pd.DataFrame(
        gene_percents, index=_adata.var_names, columns=["percent"]
    )  # Series is not appropriate for h5ad format.

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


def pca_selected_genes_wrapper(
    adata: AnnData, pca_input: Union[np.ndarray, None] = None, n_pca_components: int = 30, key: str = "X_pca"
):
    """A wrapper for pca function to reduce dimensions of the Adata with PCA.

    Args:
        adata: an AnnData object.
        pca_input: an array for nearest neighbor search directly. Defaults to None.
        n_pca_components: number of PCA components. Defaults to 30.
        key: the key to store the calculation result. Defaults to "X_pca".
    """

    adata = pca(adata, pca_input, n_pca_components=n_pca_components, pca_key=key)
