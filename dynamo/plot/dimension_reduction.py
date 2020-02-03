# """ plotting utilities that are built based on scSLAM-seq paper
from .scatters import scatters
from .scatters import docstrings

docstrings.delete_params('scatters.parameters', 'basis')
@docstrings.with_indent(4)
def pca(
        adata,
        *args,
        **kwargs):
    """\
    Scatter plot with pca basis.

    Parameters
    ----------
    %(scatters.parameters.no_basis)s

    Returns
    -------
    Nothing but plots the pca embedding of the adata object.

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> adata = dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.pl.pca(
    ...     adata,
    ...     color='Clusters'
    ... )
    """

    scatters(
        adata,
        'pca',
        *args,
        **kwargs)


@docstrings.with_indent(4)
def umap(
        adata,
        *args,
        **kwargs):
    """\
    Scatter plot with umap basis.

    Parameters
    ----------
    %(scatters.parameters.no_basis)s

    Returns
    -------
    Nothing but plots the umap embedding of the adata object.

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> adata = dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.pl.phate(
    ...     adata,
    ...     color='Clusters'
    ... )
    """

    scatters(
        adata,
        'umap',
        *args,
        **kwargs)


@docstrings.with_indent(4)
def trimap(
        adata,
        *args,
        **kwargs):
    """\
    Scatter plot with trimap basis.

    Parameters
    ----------
    %(scatters.parameters.no_basis)s

    Returns
    -------
    Nothing but plots the pca embedding of the adata object.

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> adata = dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.pl.phate(
    ...     adata,
    ...     color='Clusters'
    ... )
    """
    scatters(
        adata,
        'trip',
        *args,
        **kwargs)


@docstrings.with_indent(4)
def tsne(
        adata,
        *args,
        **kwargs):
    """\
    Scatter plot with tsne basis.

    Parameters
    ----------
    %(scatters.parameters.no_basis)s

    Returns
    -------
    Nothing but plots the tsne embedding of the adata object.

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> adata = dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.pl.phate(
    ...     adata,
    ...     color='Clusters'
    ... )
    """
    scatters(
        adata,
        'tsne',
        *args,
        **kwargs)

# add leidan, louvain, etc.
