"""General utility functions
"""
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from .dynamo_logger import LoggerManager


def isarray(arr):
    """
    Check if a variable is an array. Essentially the variable has the attribute 'len'
    and it is not a string.
    """
    return hasattr(arr, "__len__") and (not isinstance(arr, str) and (not isinstance(arr, type)))


def ismatrix(arr):
    """
    Check if a variable is an array. Essentially the variable has the attribute 'len'
    and it is not a string.
    """
    return type(arr) is np.matrix or sp.issparse(arr)


def areinstance(arr, dtype, logic_func=all):
    """
    Check if elements of an array are all (by default) of 'dtype'.
    """
    if not isarray(dtype):
        dtype = [dtype]
    ret = None
    for dt in dtype:
        if ret is None:
            ret = [isinstance(a, dt) for a in arr]
        else:
            ret = np.logical_or(ret, [isinstance(a, dt) for a in arr])
    return logic_func(ret)


def copy_adata(adata: AnnData, logger=None) -> AnnData:
    """wrapper for deep copy adata and log copy operation since it is memory intensive.

    Parameters
    ----------
    adata :
        [description]
    logger : [type], optional
        [description], by default None

    Examples
    --------
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


def normalize(x):
    x_min = np.min(x)
    return (x - x_min) / (np.max(x) - x_min)


def denormalize(y, x_min, x_max):
    return y * (x_max - x_min) + x_min
