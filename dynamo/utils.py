"""General utility functions
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import anndata
import numpy as np
import scipy.sparse as sp

from .dynamo_logger import LoggerManager


def isarray(arr: Any) -> bool:
    """Check if a variable is an array. Essentially the variable has the attribute 'len' and it is not a string."""
    return hasattr(arr, "__len__") and (not isinstance(arr, str) and (not isinstance(arr, type)))


def ismatrix(arr: Any) -> bool:
    """Check if a variable is a matrix. Essentially the variable is `np.matrix` or `sp.issparse`."""
    return type(arr) is np.matrix or sp.issparse(arr)


def areinstance(arr: Any, dtype: Union[type, List[type]], logic_func: Callable = all) -> bool:
    """Check if elements of an array are all (by default) of 'dtype'."""
    if not isarray(dtype):
        dtype = [dtype]
    ret = None
    for dt in dtype:
        if ret is None:
            ret = [isinstance(a, dt) for a in arr]
        else:
            ret = np.logical_or(ret, [isinstance(a, dt) for a in arr])
    return logic_func(ret)


def copy_adata(adata: anndata.AnnData, logger: Optional[LoggerManager] = None) -> anndata.AnnData:
    """Wrapper for deep copy adata and log copy operation since it is memory intensive.

    Args:
        adata: An adata object that will be deep copied.
        logger : [bool], optional
            Whether to report logging info

    Returns:
        The deep copied adata object.

    Examples:
        >>> import dynamo as dyn
        >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
        >>> original_adata = copy_adata(adata)
        >>> # now after this statement, adata "points" to a new object, copy of the original
        >>> adata = copy_adata(adata)
        >>> adata.X[0, 1] = -999
        >>> # original_adata unchanged
        >>> print(original_adata.X[0, 1])
        >>> # we can use adata = copy_adata(adata) inside a dynammo function when we want to create a adata copy
        >>> # without worrying about changing the original copy.
    """
    if logger is None:
        logger = LoggerManager.get_main_logger()
    logger.info(
        "Deep copying AnnData object and working on the new copy. Original AnnData object will not be modified.",
        indent_level=1,
    )
    data = adata.copy()
    return data


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize the input array to [0, 1]."""
    x_min = np.min(x)
    return (x - x_min) / (np.max(x) - x_min)


def denormalize(y: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Denormalize the input array from [0, 1] to [x_min, x_max]."""
    return y * (x_max - x_min) + x_min


# ---------------------------------------------------------------------------------------------------
# trajectory related
def pca_to_expr(
    X: Union[np.ndarray, sp.csr_matrix],
    PCs: np.ndarray,
    mean: Union[int, np.ndarray] = 0,
    func: Optional[Callable] = None,
) -> np.ndarray:
    """Inverse transform the data with given principal components.

    Args:
        X: Raw data to transform.
        PCs: The principal components.
        mean: The mean used to fit the PCA.
        func: Additional function to transform the output.

    Returns:
        The inverse transformed data.
    """
    # reverse project from PCA back to raw expression space
    X = X.toarray() if sp.issparse(X) else X
    if PCs.shape[1] == X.shape[1]:
        exprs = X @ PCs.T + mean
        if func is not None:
            exprs = func(exprs)
    else:
        raise Exception("PCs dim 1 (%d) does not match X dim 1 (%d)." % (PCs.shape[1], X.shape[1]))
    return exprs


def expr_to_pca(
    expr: Union[np.ndarray, sp.csr_matrix],
    PCs: np.ndarray,
    mean: Union[int, np.ndarray] = 0,
    func: Optional[Callable] = None,
) -> np.ndarray:
    """Transform the data with given principal components.

    Args:
        expr: Raw data to transform.
        PCs: The principal components.
        mean: The mean of expr.
        func: Additional function to transform the output.

    Returns:
        The transformed data.
    """
    # project from raw expression space to PCA
    expr = expr.toarray() if sp.issparse(expr) else expr
    if PCs.shape[0] == expr.shape[1]:
        X = (expr - mean) @ PCs
        if func is not None:
            X = func(X)
    else:
        raise Exception("PCs dim 1 (%d) does not match X dim 1 (%d)." % (PCs.shape[0], expr.shape[1]))
    return X
