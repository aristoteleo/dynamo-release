"""General utilility functions
"""

from anndata import AnnData
from .dynamo_logger import LoggerManager


def copy_annData(adata, logger=None):
    if logger is None:
        logger = LoggerManager.get_main_logger()
    logger.info(
        "Deep copying AnnData object and working on the new copy. Original AnnData object will not be modified.",
        indent_level=1,
    )
    adata = adata.copy()
    return adata
