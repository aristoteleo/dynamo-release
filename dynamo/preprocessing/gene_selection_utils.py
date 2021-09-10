from typing import List, Tuple, Union
import numpy as np
from scipy.sparse.base import issparse
import pandas as pd
from anndata import AnnData
import anndata
import scipy.sparse
from ..utils import copy_adata
from ..dynamo_logger import (
    main_debug,
    main_finish_progress,
    main_info,
    main_info_insert_adata,
    main_info_insert_adata_obs,
    main_info_insert_adata_var,
    main_log_time,
    main_warning,
)
from ..configuration import DynamoAdataKeyManager
from .utils import get_inrange_shared_counts_mask


def _infer_labeling_experiment_type(adata, tkey):
    """Returns the experiment type of `adata` according to `tkey`s"""
    experiment_type = None
    tkey_val = np.array(adata.obs[tkey], dtype="float")
    if len(np.unique(tkey_val)) == 1:
        experiment_type = "one-shot"
    else:
        labeled_frac = adata.layers["new"].T.sum(0) / adata.layers["total"].T.sum(0)
        xx = labeled_frac.A1 if issparse(adata.layers["new"]) else labeled_frac

        yy = tkey_val
        xm, ym = np.mean(xx), np.mean(yy)
        cov = np.mean(xx * yy) - xm * ym
        var_x = np.mean(xx * xx) - xm * xm

        k = cov / var_x

        # total labeled RNA amount will increase (decrease) in kinetic (degradation) experiments over time.
        experiment_type = "kin" if k > 0 else "deg"
    main_warning(
        f"\nDynamo detects your labeling data is from a {experiment_type} experiment, please correct "
        f"\nthis via supplying the correct experiment_type (one of `one-shot`, `kin`, `deg`) as "
        f"needed."
    )
    return experiment_type


def get_nan_or_inf_data_bool_mask(arr: np.ndarray):
    """Returns the mask of arr with the same shape, indicating whether each index is nan/inf or not."""
    mask = np.isnan(arr) | np.isinf(arr) | np.isneginf(arr)
    return mask


def clip_by_perc(layer_mat):
    """Returns a new matrix by clipping the layer_mat according to percentage."""
    # TODO implement this function (currently not used)
    return


def calc_mean_var_dispersion(data_mat: np.array, axis=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate mean, variance and dispersion of data_mat, a numpy array."""
    # per gene mean, var and dispersion
    mean = np.nanmean(data_mat, axis=axis)
    mean[mean == 0] += 1e-8  # prevent division by zero
    var = np.nanvar(data_mat, axis=axis)
    dispersion = var / mean
    return mean.flatten(), var.flatten(), dispersion.flatten()


def calc_mean_var_dispersion_sparse(
    sparse_mat: scipy.sparse.csr_matrix, axis=0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate mean, variance and dispersion of data_mat, a scipy sparse matrix."""
    nan_mask = get_nan_or_inf_data_bool_mask(sparse_mat.data)

    non_nan_count = sparse_mat.shape[axis] - nan_mask.sum()
    mean = (sparse_mat.sum(axis) / non_nan_count).A1
    mean[mean == 0] += 1e-8  # prevent division by zero
    # same as numpy var behavior: denominator is N, var=(data_arr-mean)/N
    var = np.power(sparse_mat - mean, 2).sum(axis) / non_nan_count
    dispersion = var / mean
    return mean.flatten(), var.flatten(), dispersion.flatten()


def select_genes_by_dispersion_general(
    adata: AnnData, layer: str = DynamoAdataKeyManager.X_LAYER, nan_replace_val: float = None, n_top_genes: int = None
):
    """A general function for filter genes family. Preprocess adata and dispatch to different filtering methods."""
    main_info("filtering genes by dispersion...")
    main_log_time()
    if n_top_genes is None:
        main_info("n_top_genes is None, reservie all genes and add filter gene information")
        n_top_genes = adata.n_vars
    layer_mat = DynamoAdataKeyManager.select_layer_data(adata, layer)
    if nan_replace_val:
        main_info("replacing nan values with: %s" % (nan_replace_val))
        mask = get_nan_or_inf_data_bool_mask(layer_mat)
        layer_mat[mask] = nan_replace_val

    select_genes_by_dispersion_svr(adata, layer_mat, n_top_genes)

    main_finish_progress("filter genes by dispersion")


def select_genes_by_dispersion_svr(
    adata: AnnData, layer_mat: Union[np.array, scipy.sparse.csr_matrix], n_top_genes: int
) -> None:
    """Filters adata's genes according to layer_mat, and set adata's preprocess keys for downstream analysis

    Parameters
    ----------
    adata : AnnData
    layer_mat :
        The specific layer matrix used for filtering genes. It can be any matrix with shape of #cells X #genes.
    n_top_genes : int
        The number of genes to use.
    """
    main_debug("type of layer_mat:" + str(type(layer_mat)))
    if issparse(layer_mat):
        main_info("layer_mat is sparse, dispatch to sparse calc function...")
        mean, variance, dispersion = calc_mean_var_dispersion_sparse(layer_mat)
    else:
        mean, variance, dispersion = calc_mean_var_dispersion(layer_mat)

    highly_variable_mask, highly_variable_scores = get_highly_variable_mask_by_dispersion_svr(
        mean, variance, n_top_genes
    )
    variance = np.array(variance).flatten()
    main_info_insert_adata_var(DynamoAdataKeyManager.VAR_GENE_MEAN_KEY)
    main_info_insert_adata_var(DynamoAdataKeyManager.VAR_GENE_VAR_KEY)
    main_info_insert_adata_var(DynamoAdataKeyManager.VAR_GENE_HIGHLY_VARIABLE_KEY)
    main_debug("type of variance:" + str(type(variance)))
    main_debug("shape of variance:" + str(variance.shape))
    adata.var[DynamoAdataKeyManager.VAR_GENE_MEAN_KEY] = mean.flatten()
    adata.var[DynamoAdataKeyManager.VAR_GENE_VAR_KEY] = variance
    adata.var[DynamoAdataKeyManager.VAR_GENE_HIGHLY_VARIABLE_KEY] = highly_variable_mask
    adata.var[DynamoAdataKeyManager.VAR_GENE_HIGHLY_VARIABLE_SCORES] = highly_variable_scores
    adata.var[DynamoAdataKeyManager.VAR_USE_FOR_PCA] = highly_variable_mask


def get_highly_variable_mask_by_dispersion_svr(
    mean: np.ndarray, var: np.ndarray, n_top_genes: int, svr_gamma: float = None, return_scores=True
):
    """Returns the mask with shape same as mean and var, indicating whether each index is highly variable or not. Each index should represent a gene."""
    # normally, select svr_gamma based on #features
    if svr_gamma is None:
        svr_gamma = 150.0 / len(mean)
    from sklearn.svm import SVR

    mean_log = np.log2(mean)
    cv_log = np.log2(np.sqrt(var) / mean)
    classifier = SVR(gamma=svr_gamma)
    classifier.fit(mean_log[:, np.newaxis], cv_log.reshape([-1, 1]))
    scores = cv_log - classifier.predict(mean_log[:, np.newaxis])
    scores = scores.reshape([-1, 1])  # shape should be #genes x 1

    # score threshold based on n top genes
    score_threshold = np.sort(-scores)[n_top_genes - 1]
    highly_variable_mask = scores >= score_threshold
    highly_variable_mask = np.array(highly_variable_mask).flatten()
    if return_scores:
        return highly_variable_mask, scores
    return highly_variable_mask


def log1p_adata(adata: AnnData, copy: bool = False) -> AnnData:
    """returns log1p  of adata's data. If copy is true, operates on a copy of adata and returns the copy."""
    _adata = adata
    if copy:
        _adata = copy_adata(adata)
    log1p_inplace(_adata)
    return _adata


def _log1p_inplace(data):
    np.log1p(data, out=data)


def log1p_inplace(adata: AnnData):
    if issparse(adata.X):
        _log1p_inplace(adata.X.data)
    else:
        _log1p_inplace(adata.X)


def filter_genes_by_outliers(
    adata: anndata.AnnData,
    filter_bool: Union[np.ndarray, None] = None,
    layer: str = "all",
    min_cell_s: int = 1,
    min_cell_u: int = 1,
    min_cell_p: int = 1,
    min_avg_exp_s: float = 1e-10,
    min_avg_exp_u: float = 0,
    min_avg_exp_p: float = 0,
    max_avg_exp: float = np.infty,
    min_count_s: int = 0,
    min_count_u: int = 0,
    min_count_p: int = 0,
    shared_count: int = 30,
) -> anndata.AnnData:
    """Basic filter of genes based a collection of expression filters.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object.
        filter_bool: :class:`~numpy.ndarray` (default: None)
            A boolean array from the user to select genes for downstream analysis.
        layer: `str` (default: `X`)
            The data from a particular layer (include X) used for feature selection.
        min_cell_s: `int` (default: `5`)
            Minimal number of cells with expression for the data in the spliced layer (also used for X).
        min_cell_u: `int` (default: `5`)
            Minimal number of cells with expression for the data in the unspliced layer.
        min_cell_p: `int` (default: `5`)
            Minimal number of cells with expression for the data in the protein layer.
        min_avg_exp_s: `float` (default: `1e-2`)
            Minimal average expression across cells for the data in the spliced layer (also used for X).
        min_avg_exp_u: `float` (default: `1e-4`)
            Minimal average expression across cells for the data in the unspliced layer.
        min_avg_exp_p: `float` (default: `1e-4`)
            Minimal average expression across cells for the data in the protein layer.
        max_avg_exp: `float` (default: `100`.)
            Maximal average expression across cells for the data in all layers (also used for X).
        min_cell_s: `int` (default: `5`)
            Minimal number of counts (UMI/expression) for the data in the spliced layer (also used for X).
        min_cell_u: `int` (default: `5`)
            Minimal number of counts (UMI/expression) for the data in the unspliced layer.
        min_cell_p: `int` (default: `5`)
            Minimal number of counts (UMI/expression) for the data in the protein layer.
        shared_count: `int` (default: `30`)
            The minimal shared number of counts for each genes across cell between layers.
    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An updated AnnData object with use_for_pca as a new column in .var attributes to indicate the selection of
            genes for downstream analysis. adata will be subsetted with only the genes pass filter if keep_unflitered is
            set to be False.
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


def get_in_range_mask(data_mat, min_val, max_val, axis=0, sum_min_val_threshold=0):
    return (
        ((data_mat > sum_min_val_threshold).sum(axis) >= min_val)
        & ((data_mat > sum_min_val_threshold).sum(axis) <= max_val)
    ).flatten()


def filter_cells_by_outliers(
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
    spliced_key="spliced",
    unspliced_key="unspliced",
    protein_key="protein",
    obs_store_key="pass_basic_filter",
) -> anndata.AnnData:
    """Select valid cells based on a collection of filters including spliced, unspliced and protein min/max vals.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object.
        filter_bool: :class:`~numpy.ndarray` (default: `None`)
            A boolean array from the user to select cells for downstream analysis.
        layer: `str` (default: `all`)
            The data from a particular layer (include X) used for feature selection. Use 'all' or a set/list to filter by shared counts of a set/list of layers.
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

    predefined_layers_for_filtering = [DynamoAdataKeyManager.X_LAYER, spliced_key, unspliced_key, protein_key]
    predefined_range_dict = {
        DynamoAdataKeyManager.X_LAYER: (min_expr_genes_s, max_expr_genes_s),
        spliced_key: (min_expr_genes_s, max_expr_genes_s),
        unspliced_key: (min_expr_genes_u, max_expr_genes_u),
        protein_key: (min_expr_genes_p, max_expr_genes_p),
    }
    layer_keys_used_for_filtering = []
    if layer == "all":
        layer_keys_used_for_filtering = predefined_layers_for_filtering
    elif type(layer) == str and layer in predefined_layers_for_filtering:
        layer_keys_used_for_filtering = [layer]

    detected_bool = get_filter_mask_cells_by_outliers(
        adata, layer_keys_used_for_filtering, predefined_range_dict, shared_count
    )

    if filter_bool is None:
        filter_bool = detected_bool
    else:
        filter_bool = np.array(filter_bool) & detected_bool

    main_info_insert_adata_obs(obs_store_key)
    if keep_filtered:
        main_info("keep filtered genes", indent_level=2)
        adata.obs[obs_store_key] = filter_bool
    else:
        main_info("inplace subsetting adata by filtered genes", indent_level=2)
        adata._inplace_subset_obs(filter_bool)
        adata.obs[obs_store_key] = True

    return adata


def get_filter_mask_cells_by_outliers(
    adata: anndata.AnnData,
    layers: list = None,
    layer2range: dict = None,
    shared_count: Union[int, None] = None,
) -> anndata.AnnData:
    """Select valid cells based on a collection of filters including spliced, unspliced and protein min/max vals.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object.
        layers:
            a list of layers
        ranges:
            a dict of ranges, with shape of #layers x 2
        shared_count: `int` or `None` (default: `None`)
            The minimal shared number of counts for each cell across genes between layers.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An updated AnnData object with use_for_pca as a new column in obs to indicate the selection of cells for
            downstream analysis. adata will be subsetted with only the cells pass filtering if keep_filtered is set to
            be False.
    """
    detected_mask = np.full(adata.n_obs, True)
    if layers is None:
        main_info("layers for filtering cells are None, reserve all cells.")
        return detected_mask

    for i, layer in enumerate(layers):
        if layer not in layer2range:
            main_info(
                "skip filtering cells by layer: %s as it is not in predefined range list:" % layer, indent_level=2
            )
            continue
        if not DynamoAdataKeyManager.check_if_layer_exist(adata, layer):
            main_info("skip filtering by layer:%s as it is not in adata." % layer)
            continue
        main_info("filtering cells by layer:%s" % layer, indent_level=2)
        layer_data = DynamoAdataKeyManager.select_layer_data(adata, layer)
        detected_mask = detected_mask & get_in_range_mask(
            layer_data, layer2range[layer][0], layer2range[layer][1], axis=1, sum_min_val_threshold=0
        )

    if shared_count is not None:
        main_info("filtering cells by shared counts from all layers", indent_level=2)
        layers = DynamoAdataKeyManager.get_available_layer_keys(adata, layers, False)
        detected_mask = detected_mask & get_inrange_shared_counts_mask(adata, layers, shared_count, "cell")

    detected_mask = np.array(detected_mask).flatten()
    return detected_mask
