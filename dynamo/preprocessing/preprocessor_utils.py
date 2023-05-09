import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata
import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse.base import issparse
from sklearn.utils import sparsefuncs

from ..configuration import DKM, DynamoAdataKeyManager
from ..dynamo_logger import (
    main_debug,
    main_finish_progress,
    main_info,
    main_info_insert_adata,
    main_info_insert_adata_layer,
    main_info_insert_adata_obs,
    main_info_insert_adata_obsm,
    main_info_insert_adata_uns,
    main_log_time,
    main_warning,
)
from ..utils import copy_adata
from .pca import pca
from .utils import (
    add_noise_to_duplicates,
    calc_new_to_total_ratio,
    collapse_species_adata,
    compute_gene_exp_fraction,
    get_inrange_shared_counts_mask,
    get_svr_filter,
    merge_adata_attrs,
)


def _infer_labeling_experiment_type(adata: anndata.AnnData, tkey: str) -> Literal["one-shot", "kin", "deg"]:
    """Returns the experiment type of `adata` according to `tkey`s

    Args:
        adata: an AnnData Object.
        tkey: the key for time in `adata.obs`.

    Returns:
        The experiment type, must be one of "one-shot", "kin" or "deg".
    """

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
    main_debug(
        f"\nDynamo has detected that your labeling data is from a kin experiment. \nIf the experiment type is incorrect, "
        f"please provide the correct experiment_type (one-shot, kin, or deg)."
    )
    return experiment_type


def get_nan_or_inf_data_bool_mask(arr: np.ndarray) -> np.ndarray:
    """Returns the mask of arr with the same shape, indicating whether each index is nan/inf or not.

    Args:
        arr: an array

    Returns:
        A bool array indicating each element is nan/inf or not
    """

    mask = np.isnan(arr) | np.isinf(arr) | np.isneginf(arr)
    return mask


def clip_by_perc(layer_mat):
    """Returns a new matrix by clipping the layer_mat according to percentage."""
    # TODO implement this function (currently not used)
    return


def seurat_get_mean_var(
    X: Union[csr_matrix, np.ndarray],
    ignore_zeros: bool = False,
    perc: Union[float, List[float], None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Only used in seurat impl to match seurat and scvelo implementation result.

    Args:
        X: a matrix as np.ndarray or a sparse matrix as scipy sparse matrix. Rows are cells while columns are genes.
        ignore_zeros: whether ignore columns with 0 only. Defaults to False.
        perc: clip the gene expression values based on the perc or the min/max boundary of the values. Defaults to None.

    Returns:
        A tuple (mean, var) where mean is the mean of the columns after processing of the matrix and var is the variance
        of the columns after processing of the matrix.
    """

    data = X.data if issparse(X) else X
    mask_nans = np.isnan(data) | np.isinf(data) | np.isneginf(data)

    n_nonzeros = (X != 0).sum(0)
    n_counts = n_nonzeros if ignore_zeros else X.shape[0]

    if mask_nans.sum() > 0:
        if issparse(X):
            data[np.isnan(data) | np.isinf(data) | np.isneginf(data)] = 0
            n_nans = n_nonzeros - (X != 0).sum(0)
        else:
            X[mask_nans] = 0
            n_nans = mask_nans.sum(0)
        n_counts -= n_nans

    if perc is not None:
        if np.size(perc) < 2:
            perc = [perc, 100] if perc < 50 else [0, perc]
        lb, ub = np.percentile(data, perc)
        data = np.clip(data, lb, ub)

    if issparse(X):
        mean = (X.sum(0) / n_counts).A1
        mean_sq = (X.multiply(X).sum(0) / n_counts).A1
    else:
        mean = X.sum(0) / n_counts
        mean_sq = np.multiply(X, X).sum(0) / n_counts
    n_cells = np.clip(X.shape[0], 2, None)  # to avoid division by zero
    var = (mean_sq - mean**2) * (n_cells / (n_cells - 1))

    mean = np.nan_to_num(mean)
    var = np.nan_to_num(var)
    return mean, var


def is_nonnegative(mat: Union[np.ndarray, spmatrix, list]) -> bool:
    """Test whether all elements of an array or sparse array are non-negative.

    Args:
        mat: an array in ndarray or sparse array in scipy spmatrix.

    Returns:
        A flag whether all elements are non-negative.
    """

    if scipy.sparse.issparse(mat):
        return np.all(mat.sign().data >= 0)
    return np.all(np.sign(mat) >= 0)


def is_integer_arr(arr: Union[np.ndarray, spmatrix, list]) -> bool:
    """Test if an array like obj's dtype is integer

    Args:
        arr: an array like object.

    Returns:
        A flag whether the array's dtype is integer.
    """

    return np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, int)


def is_float_integer_arr(arr: Union[np.ndarray, spmatrix, list]) -> bool:
    """Test if an array's elements are integers

    Args:
        arr: an input array.

    Returns:
        A flag whether all elements of the array are integers.
    """

    if issparse(arr):
        arr = arr.data
    return np.all(np.equal(np.mod(arr, 1), 0))


def is_nonnegative_integer_arr(mat: Union[np.ndarray, spmatrix, list]) -> bool:
    """Test if an array's elements are non-negative integers

    Args:
        mat: an input array.

    Returns:
        A flag whether all elements of the array are non-negative integers.
    """

    if (not is_integer_arr(mat)) and (not is_float_integer_arr(mat)):
        return False
    return is_nonnegative(mat)


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


def regress_out_chunk_helper(args):
    """A helper function for each regressout chunk.

    Args:
        args: list of arguments that is used in _regress_out_chunk.

    Returns:
        numpy array: predicted the effects of the variables calculated by _regress_out_chunk.
    """
    obs_feature, gene_expr = args
    return _regress_out_chunk(obs_feature, gene_expr)


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

    ctx = mp.get_context("fork")

    with ctx.Pool(n_cores) as pool:
        results = pool.map(func, args_list)
        pool.close()
        pool.join()

    return results
