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
