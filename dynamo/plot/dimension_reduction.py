"""plotting utilities that are built based on scSLAM-seq paper"""

from typing import Optional, Union

from anndata import AnnData

from .scatters import docstrings, scatters

docstrings.delete_params("scatters.parameters", "adata", "basis")


@docstrings.with_indent(4)
def pca(adata: AnnData, *args, **kwargs):
    """\
    Scatter plot with pca basis.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        %(scatters.parameters.no_adata|basis)s

    Returns
    -------
    Nothing but plots the pca embedding of the adata object.

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.pl.pca(
    ...     adata,
    ...     color='ntr'
    ... )
    """

    scatters(adata, "pca", *args, **kwargs)


@docstrings.with_indent(4)
def umap(adata: AnnData, *args, **kwargs):
    """\
    Scatter plot with umap basis.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        %(scatters.parameters.no_adata|basis)s

    Returns
    -------
    Nothing but plots the umap embedding of the adata object.

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.pl.umap(
    ...     adata,
    ...     color='ntr'
    ... )
    """

    return scatters(adata, "umap", *args, **kwargs)


@docstrings.with_indent(4)
def trimap(adata: AnnData, *args, **kwargs):
    """\
    Scatter plot with trimap basis.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        %(scatters.parameters.no_adata|basis)s

    Returns
    -------
    Nothing but plots the pca embedding of the adata object.

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.pl.trimap(
    ...     adata,
    ...     color='ntr'
    ... )
    """
    return scatters(adata, "trimap", *args, **kwargs)


@docstrings.with_indent(4)
def tsne(adata: AnnData, *args, **kwargs):
    """\
    Scatter plot with tsne basis.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        %(scatters.parameters.no_adata|basis)s

    Returns
    -------
    Nothing but plots the tsne embedding of the adata object.

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.pl.tsne(
    ...     adata,
    ...     color='ntr'
    ... )
    """
    return scatters(adata, "tsne", *args, **kwargs)


# add leidan, louvain, etc.
