"""General utilility functions
"""

from .dynamo_logger import LoggerManager


def copy_adata(adata, logger=None):
    """[summary]

    Parameters
    ----------
    adata : [type]
        [description]
    logger : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]

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
    >>> # therefore, we can use adata = copy_adata(adata) inside a dynammo function when we want to create a adata copy
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
