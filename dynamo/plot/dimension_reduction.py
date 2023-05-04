"""plotting utilities that are built based on scSLAM-seq paper"""

from typing import Optional, Union

from anndata import AnnData
from matplotlib.axes import Axes

from .scatters import docstrings, scatters

docstrings.delete_params("scatters.parameters", "adata", "basis")


def pca(adata: AnnData, *args, **kwargs) -> Optional[Axes]:
    """Scatter plot with pca basis.

    Args:
        adata: an AnnData object.
        *args: any other positional arguments passed to `dynamo.pl.scatters`.
        **kwargs: any other keyword arguments passed to `dynamo.pl.scatters`.

    Returns:
        None would be returned in default and the plotted figure would be shown directly. If set
        `save_show_or_return='return'` as a kwarg, the axes of the plot would be returned.
    """

    return scatters(adata, "pca", *args, **kwargs)


def umap(adata: AnnData, *args, **kwargs) -> Optional[Axes]:
    """Scatter plot with umap basis.

    Args:
        adata: an AnnData object.
        *args: any other positional arguments passed to `dynamo.pl.scatters`.
        **kwargs: any other keyword arguments passed to `dynamo.pl.scatters`.

    Returns:
        None would be returned in default and the plotted figure would be shown directly. If set
        `save_show_or_return='return'` as a kwarg, the axes of the plot would be returned.
    """

    return scatters(adata, "umap", *args, **kwargs)


def trimap(adata: AnnData, *args, **kwargs):
    """Scatter plot with trimap basis.

    Args:
        adata: an AnnData object.
        *args: any other positional arguments passed to `dynamo.pl.scatters`.
        **kwargs: any other keyword arguments passed to `dynamo.pl.scatters`.

    Returns:
        None would be returned in default and the plotted figure would be shown directly. If set
        `save_show_or_return='return'` as a kwarg, the axes of the plot would be returned.
    """

    return scatters(adata, "trimap", *args, **kwargs)


def tsne(adata: AnnData, *args, **kwargs):
    """Scatter plot with tsne basis.

    Args:
        adata: an AnnData object.
        *args: any other positional arguments passed to `dynamo.pl.scatters`.
        **kwargs: any other keyword arguments passed to `dynamo.pl.scatters`.

    Returns:
        None would be returned in default and the plotted figure would be shown directly. If set
        `save_show_or_return='return'` as a kwarg, the axes of the plot would be returned.
    """

    return scatters(adata, "tsne", *args, **kwargs)


# add leidan, louvain, etc.
