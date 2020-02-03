# """ plotting utilities that are built based on scSLAM-seq paper
from .scatters import scatters
from .scatters import docstrings

docstrings.delete_params('scatters.parameters', 'pca')
@docstrings.with_indent(4)
def pca(
        adata,
        *args,
        **kwargs):
    """\
    Scatter plot with pca basis.

    Parameters
    ----------
    %(scatters.parameters.no_pca)s

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
        x=0,
        y=1,
        color=None,
        basis='umap',
        layer='X',
        highlights=None,
        labels=None,
        values=None,
        theme=None,
        cmap=None,
        color_key=None,
        color_key_cmap=None,
        background="black",
        ncols=1,
        pointsize=None,
        figsize=(7,5),
        show_legend=True,
        use_smoothed=True,
        ax=None,
        **kwargs):
    """\
    Scatter plot with umap basis.

    Parameters
    ----------
    %(scatters.parameters.no_pca)s

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
        x,
        y,
        color,
        layer,
        highlights,
        labels,
        values,
        theme,
        cmap,
        color_key,
        color_key_cmap,
        background,
        ncols,
        pointsize,
        figsize,
        show_legend,
        use_smoothed,
        ax,
        **kwargs)


@docstrings.with_indent(4)
def trimap(
        adata,
        basis='trimap',
        x=0,
        y=1,
        color=None,
        layer='X',
        highlights=None,
        labels=None,
        values=None,
        theme=None,
        cmap=None,
        color_key=None,
        color_key_cmap=None,
        background="black",
        ncols=1,
        pointsize=None,
        figsize=(7,5),
        show_legend=True,
        use_smoothed=True,
        ax=None,
        **kwargs):
    """\
    Scatter plot with trimap basis.

    Parameters
    ----------
    %(scatters.parameters.no_pca)s

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
        'trimap',
        x,
        y,
        color,
        layer,
        highlights,
        labels,
        values,
        theme,
        cmap,
        color_key,
        color_key_cmap,
        background,
        ncols,
        pointsize,
        figsize,
        show_legend,
        use_smoothed,
        ax,
        **kwargs)


@docstrings.with_indent(4)
def tsne(
        adata,
        basis='tSNE',
        x=0,
        y=1,
        color=None,
        layer='X',
        highlights=None,
        labels=None,
        values=None,
        theme=None,
        cmap=None,
        color_key=None,
        color_key_cmap=None,
        background="black",
        ncols=1,
        pointsize=None,
        figsize=(7,5),
        show_legend=True,
        use_smoothed=True,
        ax=None,
        **kwargs):
    """\
    Scatter plot with tsne basis.

    Parameters
    ----------
    %(scatters.parameters.no_pca)s

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
        'tSNE',
        x,
        y,
        color,
        layer,
        highlights,
        labels,
        values,
        theme,
        cmap,
        color_key,
        color_key_cmap,
        background,
        ncols,
        pointsize,
        figsize,
        show_legend,
        use_smoothed,
        ax,
        **kwargs)

# add leidan, louvain, etc.
