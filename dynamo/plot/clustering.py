import numpy as np
from anndata import AnnData

from .scatters import docstrings, scatters
from .utils import _to_hex

docstrings.delete_params("scatters.parameters", "adata", "basis")


@docstrings.with_indent(4)
def hdbscan(adata: AnnData, basis: str = "umap", color: str = "hdbscan", *args, **kwargs):
    """\
    Scatter plot for hdbscan clustering in selected basis.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        %(scatters.parameters.no_adata|basis)s

    Returns
    -------
        plots hdbscan clustering of the adata object.
    """

    return scatters(adata, basis=basis, color=color, *args, **kwargs)


@docstrings.with_indent(4)
def leiden(adata: AnnData, basis: str = "umap", color: str = "leiden", *args, **kwargs):
    """\
   Scatter plot for leiden community detection in selected basis.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        %(scatters.parameters.no_adata|basis)s

    Returns
    -------
        plots leiden clustering of the adata object.
    """

    return scatters(adata, basis=basis, color=color, *args, **kwargs)


@docstrings.with_indent(4)
def louvain(adata: AnnData, basis: str = "umap", color: str = "louvain", color_key_cmap="Spectral", *args, **kwargs):
    """\
    Scatter plot for louvain community detection in selected basis.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        %(scatters.parameters.no_adata|basis)s

    Returns
    -------
        plots louvain clustering of the adata object.
    """

    return scatters(adata, basis=basis, color=color, *args, **kwargs)


@docstrings.with_indent(4)
def infomap(adata: AnnData, basis: str = "umap", color: str = "infomap", *args, **kwargs):
    """\
    Scatter plot for infomap community detection in selected basis.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        %(scatters.parameters.no_adata|basis)s

    Returns
    -------
        plots leiden clustering of the adata object.
    """

    return scatters(adata, basis=basis, color=color, *args, **kwargs)


@docstrings.with_indent(4)
def streamline_clusters(
    adata: AnnData, basis: str = "umap", clusters="clusters", color_key_cmap: str = "Spectral", *args, **kwargs
):
    """\
    Scatter plot for infomap community detection in selected basis.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        %(scatters.parameters.no_adata|basis)s

    Returns
    -------
        plots leiden clustering of the adata object.
    """
    import matplotlib.pyplot as plt

    if "streamline_clusters_" + basis not in adata.uns_keys():
        from ..vectorfield.clustering import streamline_clusters

        streamline_clusters(adata, basis=basis)

    segments = adata.uns["streamline_clusters_" + basis]["segments"]
    clusters = adata.uns["streamline_clusters_" + basis][clusters].astype(int)
    clusters[np.isnan(clusters)] = -1
    num_labels = len(np.unique(clusters))

    color_key = _to_hex(plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels)))

    fig, ax = plt.subplots(1, 1)
    for key, values in segments.items():
        ax.plot(*values.T, color=color_key[int(clusters[key])])
    plt.show()
