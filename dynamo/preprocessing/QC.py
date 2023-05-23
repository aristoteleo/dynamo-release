import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse, spmatrix

from ..configuration import DKM
from ..dynamo_logger import (
    LoggerManager,
    main_debug,
    main_exception,
    main_finish_progress,
    main_info,
    main_info_insert_adata_obs,
    main_log_time,
    main_warning,
)
from .utils import get_inrange_shared_counts_mask


def _parallel_wrapper(func: Callable, args_list, n_cores: Optional[int] = None):
    """A wrapper for parallel operation to regress out of the input variables.

    Args:
        func: The function to be conducted the multiprocessing.
        args_list: The iterable of arguments to be passed to the function.
        n_cores: The number of CPU cores to be used for parallel processing. Default to be None.

    Returns:
        results: The list of results returned by the function for each element of the iterable.
    """
    import multiprocessing as mp
    import sys

    if sys.platform != "win32":
        ctx = mp.get_context("fork")  # this fixes loop on MacOS
    else:
        ctx = mp.get_context("spawn")

    with ctx.Pool(n_cores) as pool:
        results = pool.map(func, args_list)
        pool.close()
        pool.join()

    return results


def _regress_out_chunk(
    obs_feature: Union[np.ndarray, spmatrix, list], gene_expr: Union[np.ndarray, spmatrix, list]
) -> Union[np.ndarray, spmatrix, list]:
    """Perform a linear regression to remove the effects of cell features (percentage of mitochondria, etc.)

    Args:
        obs_feature: list of observation keys used to regress out their effect to gene expression.
        gene_expr : the current gene expression values of the target variables.

    Returns:
        numpy array: the residuals that are predicted the effects of the variables.
    """
    from sklearn.linear_model import LinearRegression

    # Fit a linear regression model to the variables to remove
    reg = LinearRegression().fit(obs_feature, gene_expr)

    # Predict the effects of the variables to remove
    return reg.predict(obs_feature)


def basic_stats(adata: anndata.AnnData) -> None:
    """Generate basic stats of the adata, including number of genes, number of cells, and number of mitochondria genes.

    Args:
        adata: an AnnData object.
    """

    adata.obs["nGenes"], adata.obs["nCounts"] = np.array((adata.X > 0).sum(1)), np.array((adata.X).sum(1))
    adata.var["nCells"], adata.var["nCounts"] = np.array((adata.X > 0).sum(0).T), np.array((adata.X).sum(0).T)
    if adata.var_names.inferred_type == "bytes":
        adata.var_names = adata.var_names.astype("str")
    mito_genes = adata.var_names.str.upper().str.startswith("MT-")

    if sum(mito_genes) > 0:
        try:
            adata.obs["pMito"] = np.array(adata.X[:, mito_genes].sum(1) / adata.obs["nCounts"].values.reshape((-1, 1)))
        except:  # noqa E722
            main_exception(
                "no mitochondria genes detected; looks like your var_names may be corrupted (i.e. "
                "include nan values). If you don't believe so, please report to us on github or "
                "via xqiu@wi.mit.edu"
            )
    else:
        adata.obs["pMito"] = 0


def clusters_stats(
    U: pd.DataFrame, S: pd.DataFrame, clusters_uid: np.ndarray, cluster_ix: np.ndarray, size_limit: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the averages per cluster for unspliced and spliced data.

    Args:
        U: the unspliced DataFrame.
        S: the spliced DataFrame.
        clusters_uid: the uid of the clusters.
        cluster_ix: the indices of the clusters in adata.obs.
        size_limit: the max number of members to be considered in a cluster during calculation. Defaults to 40.

    Returns:
        U_avgs: the average of clusters for unspliced data.
        S_avgs: the average of clusters for spliced data.
    """

    U_avgs = np.zeros((S.shape[1], len(clusters_uid)))
    S_avgs = np.zeros((S.shape[1], len(clusters_uid)))
    # avgU_div_avgS = np.zeros((S.shape[1], len(clusters_uid)))
    # slopes_by_clust = np.zeros((S.shape[1], len(clusters_uid)))

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


def filter_genes_by_clusters(
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


def filter_cells_by_outliers(
    adata: AnnData,
    filter_bool: Union[np.ndarray, None] = None,
    layer: str = "all",
    keep_filtered: bool = False,
    min_expr_genes_s: int = 50,
    min_expr_genes_u: int = 25,
    min_expr_genes_p: int = 1,
    max_expr_genes_s: float = np.inf,
    max_expr_genes_u: float = np.inf,
    max_expr_genes_p: float = np.inf,
    max_pmito_s: Optional[float] = None,
    shared_count: Optional[int] = None,
    spliced_key="spliced",
    unspliced_key="unspliced",
    protein_key="protein",
    obs_store_key="pass_basic_filter",
) -> AnnData:
    """Select valid cells based on a collection of filters including spliced, unspliced and protein min/max vals.

    Args:
        adata: an AnnData object.
        filter_bool: a boolean array from the user to select cells for downstream analysis. Defaults to None.
        layer: the layer (include X) used for feature selection. Defaults to "all".
        keep_filtered: whether to keep cells that don't pass the filtering in the adata object. Defaults to False.
        min_expr_genes_s: minimal number of genes with expression for a cell in the data from the spliced layer (also
            used for X). Defaults to 50.
        min_expr_genes_u: minimal number of genes with expression for a cell in the data from the unspliced layer.
            Defaults to 25.
        min_expr_genes_p: minimal number of genes with expression for a cell in the data from in the protein layer.
            Defaults to 1.
        max_expr_genes_s: maximal number of genes with expression for a cell in the data from the spliced layer (also
            used for X). Defaults to np.inf.
        max_expr_genes_u: maximal number of genes with expression for a cell in the data from the unspliced layer.
            Defaults to np.inf.
        max_expr_genes_p: maximal number of protein with expression for a cell in the data from the protein layer.
            Defaults to np.inf.
        max_pmito_s: maximal percentage of mitochondrial genes for a cell in the data from the spliced layer.
        shared_count: the minimal shared number of counts for each cell across genes between layers. Defaults to None.
        spliced_key: name of the layer storing spliced data. Defaults to "spliced".
        unspliced_key: name of the layer storing unspliced data. Defaults to "unspliced".
        protein_key: name of the layer storing protein data. Defaults to "protein".
        obs_store_key: name of the layer to store the filtered data. Defaults to "pass_basic_filter".

    Raises:
        ValueError: the layer provided is invalid.

    Returns:
        An updated AnnData object indicating the selection of cells for downstream analysis. adata will be subsetted
        with only the cells pass filtering if keep_filtered is set to be False.
    """

    predefined_layers_for_filtering = [DKM.X_LAYER, spliced_key, unspliced_key, protein_key]
    predefined_range_dict = {
        DKM.X_LAYER: (min_expr_genes_s, max_expr_genes_s),
        spliced_key: (min_expr_genes_s, max_expr_genes_s),
        unspliced_key: (min_expr_genes_u, max_expr_genes_u),
        protein_key: (min_expr_genes_p, max_expr_genes_p),
    }
    layer_keys_used_for_filtering = []
    if layer == "all":
        layer_keys_used_for_filtering = predefined_layers_for_filtering
    elif isinstance(layer, str) and layer in predefined_layers_for_filtering:
        layer_keys_used_for_filtering = [layer]
    elif isinstance(layer, list) and set(layer) <= set(predefined_layers_for_filtering):
        layer_keys_used_for_filtering = layer
    else:
        raise ValueError(
            "layer should be str or list, and layer should be one of or a subset of "
            + str(predefined_layers_for_filtering)
        )

    detected_bool = get_filter_mask_cells_by_outliers(
        adata, layer_keys_used_for_filtering, predefined_range_dict, shared_count
    )

    if max_pmito_s is not None:
        detected_bool = detected_bool & (adata.obs["pMito"] < max_pmito_s)
        main_info(
            "filtered out %d cells by %f%% of mitochondrial genes for a cell."
            % (adata.n_obs - (adata.obs["pMito"] < max_pmito_s).sum(), max_pmito_s),
            indent_level=2,
        )

    filter_bool = detected_bool if filter_bool is None else np.array(filter_bool) & detected_bool

    main_info("filtered out %d outlier cells" % (adata.n_obs - sum(filter_bool)), indent_level=2)
    main_info_insert_adata_obs(obs_store_key)

    adata.obs[obs_store_key] = filter_bool

    if not keep_filtered:
        main_debug("inplace subsetting adata by filtered cells", indent_level=2)
        adata._inplace_subset_obs(filter_bool)

    return adata


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
    max_avg_exp: float = np.inf,
    min_count_s: int = 0,
    min_count_u: int = 0,
    min_count_p: int = 0,
    shared_count: int = 30,
    inplace: bool = False,
) -> Union[anndata.AnnData, pd.DataFrame]:
    """Basic filter of genes based a collection of expression filters.

    Args:
        adata: an AnnData object.
        filter_bool: A boolean array from the user to select genes for downstream analysis. Defaults to None.
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
        inplace: whether to update the layer inplace. Defaults to False.

    Returns:
        An updated AnnData object with genes filtered if `inplace` is true. Otherwise, an array containing filtered
        genes.
    """

    detected_bool = np.ones(adata.shape[1], dtype=bool)
    detected_bool = (detected_bool) & np.array(
        ((adata.X > 0).sum(0) >= min_cell_s)
        & (adata.X.mean(0) >= min_avg_exp_s)
        & (adata.X.mean(0) <= max_avg_exp)
        & (adata.X.sum(0) >= min_count_s)
    ).flatten()

    # add our filtering for labeling data below

    # TODO refactor with get_in_range_mask
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
        # layers = DKM.get_available_layer_keys(adata, "all", False)
        layers = DKM.get_raw_data_layers(adata)
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
    main_info("filtered out %d outlier genes" % (adata.n_vars - sum(filter_bool)), indent_level=2)

    if inplace:
        adata._inplace_subset_var(adata.var["pass_basic_filter"])
        return adata
    return adata.var["pass_basic_filter"]


def filter_genes_by_pattern(
    adata: anndata.AnnData,
    patterns: Tuple[str] = ("MT-", "RPS", "RPL", "MRPS", "MRPL", "ERCC-"),
    drop_genes: bool = False,
) -> Union[List[bool], None]:
    """Utility function to filter mitochondria, ribsome protein and ERCC spike-in genes, etc.

    Args:
        adata: an AnnData object.
        patterns: the patterns used to filter genes. Defaults to ("MT-", "RPS", "RPL", "MRPS", "MRPL", "ERCC-").
        drop_genes: whether inplace drop the genes from the AnnData object. Defaults to False.

    Returns:
        A list of indices of matched genes if `drop_genes` is False. Otherwise, returns none.
    """

    logger = LoggerManager.gen_logger("dynamo-utils")

    matched_genes = pd.Series(adata.var_names).str.startswith(patterns).to_list()
    logger.info(
        "total matched genes is " + str(sum(matched_genes)),
        indent_level=1,
    )
    if sum(matched_genes) > 0:
        if drop_genes:
            gene_bools = np.ones(adata.n_vars, dtype=bool)
            gene_bools[matched_genes] = False
            logger.info(
                "inplace subset matched genes ... ",
                indent_level=1,
            )
            # let us ignore the `inplace` parameter in pandas.Categorical.remove_unused_categories  warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                adata._inplace_subset_var(gene_bools)

            logger.finish_progress(progress_name="filter_genes_by_pattern")
            return None
        else:
            logger.finish_progress(progress_name="filter_genes_by_pattern")
            return matched_genes


def get_filter_mask_cells_by_outliers(
    adata: anndata.AnnData,
    layers: List[str] = None,
    layer2range: dict = None,
    shared_count: Union[int, None] = None,
) -> np.ndarray:
    """Select valid cells based on a collection of filters including spliced, unspliced and protein min/max vals.

    Args:
        adata: an AnnData object.
        layers: a list of layers to be operated on. Defaults to None.
        layer2range: a dict of ranges, layer str to range tuple. Defaults to None.
        shared_count: the minimal shared number of counts for each cell across genes between layers. Defaults to None.

    Returns:
        A bool array indicating valid cells.
    """

    detected_mask = np.full(adata.n_obs, True)
    if layers is None:
        main_info("layers for filtering cells are None, reserve all cells.")
        return detected_mask

    for i, layer in enumerate(layers):
        if layer not in layer2range:
            main_debug(
                "skip filtering cells by layer: %s as it is not in the layer2range mapping passed in:" % layer,
                indent_level=2,
            )
            continue
        if not DKM.check_if_layer_exist(adata, layer):
            main_debug("skip filtering by layer:%s as it is not in adata." % layer)
            continue

        main_debug("filtering cells by layer:%s" % layer, indent_level=2)
        layer_data = DKM.select_layer_data(adata, layer)
        detected_mask = detected_mask & get_sum_in_range_mask(
            layer_data, layer2range[layer][0], layer2range[layer][1], axis=1, data_min_val_threshold=0
        )

    if shared_count is not None:
        main_debug("filtering cells by shared counts from all layers", indent_level=2)
        layers = DKM.get_available_layer_keys(adata, layers, False)
        detected_mask = detected_mask & get_inrange_shared_counts_mask(adata, layers, shared_count, "cell")

    detected_mask = np.array(detected_mask).flatten()
    return detected_mask


def get_sum_in_range_mask(
    data_mat: np.ndarray, min_val: float, max_val: float, axis: int = 0, data_min_val_threshold: float = 0
) -> np.ndarray:
    """Check if data_mat's sum is inrange or not along an axis. data_mat's values < data_min_val_threshold are ignored.

    Args:
        data_mat: the array to be inspected.
        min_val: the lower bound of the range.
        max_val: the upper bound of the range.
        axis: the axis to sum. Defaults to 0.
        data_min_val_threshold: the lower threshold for valid data. Defaults to 0.

    Returns:
        A bool array indicating whether the sum is inrage or not.
    """

    return (
        ((data_mat > data_min_val_threshold).sum(axis) >= min_val)
        & ((data_mat > data_min_val_threshold).sum(axis) <= max_val)
    ).flatten()


def regress_out_chunk_helper(args):
    """A helper function for each regressout chunk.

    Args:
        args: list of arguments that is used in _regress_out_chunk.

    Returns:
        numpy array: predicted the effects of the variables calculated by _regress_out_chunk.
    """
    obs_feature, gene_expr = args
    return _regress_out_chunk(obs_feature, gene_expr)


def regress_out_parallel(
    adata: AnnData,
    layer: str = DKM.X_LAYER,
    obs_keys: Optional[List[str]] = None,
    gene_selection_key: Optional[str] = None,
    n_cores: Optional[int] = None,
):
    """Perform linear regression to remove the effects of given variables from a target variable.

    Args:
        adata: an AnnData object. Feature matrix of shape (n_samples, n_features).
        layer: the layer to regress out. Defaults to "X".
        obs_keys: List of observation keys to be removed.
        gene_selection_key: the key in adata.var that contains boolean for showing genes` filtering results.
            For example, "use_for_pca" is selected then it will regress out only for the genes that are True
            for "use_for_pca". This input will decrease processing time of regressing out data.
        n_cores: Change this to the number of cores on your system for parallel computing. Default to be None.
        obsm_store_key: the key to store the regress out result. Defaults to "X_regress_out".
    """
    main_debug("regress out %s by multiprocessing..." % obs_keys)
    main_log_time()

    if len(obs_keys) < 1:
        main_warning("No variable to regress out")
        return

    if gene_selection_key is None:
        regressor = DKM.select_layer_data(adata, layer)
    else:
        if gene_selection_key not in adata.var.keys():
            raise ValueError(str(gene_selection_key) + " is not a key in adata.var")

        if not (adata.var[gene_selection_key].dtype == bool):
            raise ValueError(str(gene_selection_key) + " is not a boolean")

        subset_adata = adata[:, adata.var.loc[:, gene_selection_key]]
        regressor = DKM.select_layer_data(subset_adata, layer)

    import itertools

    if issparse(regressor):
        regressor = regressor.toarray()

    if n_cores is None:
        n_cores = 1  # Use no parallel computing as default

    # Split the input data into chunks for parallel processing
    chunk_size = min(1000, regressor.shape[1] // n_cores + 1)
    chunk_len = regressor.shape[1] // chunk_size
    regressor_chunks = np.array_split(regressor, chunk_len, axis=1)

    # Select the variables to remove
    remove = adata.obs[obs_keys].to_numpy()

    res = _parallel_wrapper(regress_out_chunk_helper, zip(itertools.repeat(remove), regressor_chunks), n_cores)

    # Remove the effects of the variables from the target variable
    residuals = regressor - np.hstack(res)

    DKM.set_layer_data(adata, layer, residuals)
    main_finish_progress("regress out")
