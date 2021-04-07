from .scatters import scatters
from .scatters import docstrings

docstrings.delete_params("scatters.parameters", "adata", "basis")


@docstrings.with_indent(4)
def louvain(adata, basis, key="louvain_community", *args, **kwargs):
    """\
    Scatter plot with umap basis.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        %(scatters.parameters.no_adata|basis)s

    Returns
    -------
        plots louvain clustering of the adata object.
    """

    return scatters(adata, basis=basis, color=key, *args, **kwargs)


@docstrings.with_indent(4)
def leiden(adata, basis, key="leiden_community", *args, **kwargs):
    """\
    Scatter plot with umap basis.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        %(scatters.parameters.no_adata|basis)s

    Returns
    -------
        plots leiden clustering of the adata object.
    """

    return scatters(adata, basis=basis, color=key, *args, **kwargs)
