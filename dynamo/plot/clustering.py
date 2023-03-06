from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes

from .scatters import scatters
from .utils import _to_hex


def hdbscan(adata: AnnData, basis: str = "umap", color: str = "hdbscan", *args, **kwargs) -> Optional[Axes]:
    """Scatter plot for hdbscan clustering in selected basis.

    Args:
        adata: an AnnData object.
        basis: the reduced dimension stored in adata.obsm. The specific basis key will be constructed in the following
            priority if exits: 1) specific layer input + basis 2) X_ + basis 3) basis. E.g. if basis is PCA, `scatters`
            is going to look for 1) if specific layer is spliced, `spliced_pca` 2) `X_pca` (dynamo convention) 3) `pca`.
            Defaults to "umap".
        color: any column names or gene expression, etc. that will be used for coloring cells. Defaults to "hdbscan".
        *args: any other positional arguments passed to `dynamo.pl.scatters`.
        **kwargs: any other keyword arguments passed to `dynamo.pl.scatters`.

    Returns:
        None would be returned in default and the plotted figure would be shown directly. If set
        `save_show_or_return='return'` as a kwarg, the axes of the plot would be returned.
    """

    return scatters(adata, basis=basis, color=color, *args, **kwargs)


def leiden(adata: AnnData, basis: str = "umap", color: str = "leiden", *args, **kwargs) -> Optional[Axes]:
    """Scatter plot for leiden community detection in selected basis.

    Args:
        adata: an AnnData object.
        basis: the reduced dimension stored in adata.obsm. The specific basis key will be constructed in the following
            priority if exits: 1) specific layer input + basis 2) X_ + basis 3) basis. E.g. if basis is PCA, `scatters`
            is going to look for 1) if specific layer is spliced, `spliced_pca` 2) `X_pca` (dynamo convention) 3) `pca`.
            Defaults to "umap".
        color: any column names or gene expression, etc. that will be used for coloring cells. Defaults to "leiden".
        *args: any other positional arguments passed to `dynamo.pl.scatters`.
        **kwargs: any other keyword arguments passed to `dynamo.pl.scatters`.

    Returns:
        None would be returned in default and the plotted figure would be shown directly. If set
        `save_show_or_return='return'` as a kwarg, the axes of the plot would be returned.
    """

    return scatters(adata, basis=basis, color=color, *args, **kwargs)


def louvain(
    adata: AnnData, basis: str = "umap", color: str = "louvain", color_key_cmap="Spectral", *args, **kwargs
) -> Optional[Axes]:
    """Scatter plot for louvain community detection in selected basis.

    Args:
        adata: an AnnData object.
        basis: the reduced dimension stored in adata.obsm. The specific basis key will be constructed in the following
            priority if exits: 1) specific layer input + basis 2) X_ + basis 3) basis. E.g. if basis is PCA, `scatters`
            is going to look for 1) if specific layer is spliced, `spliced_pca` 2) `X_pca` (dynamo convention) 3) `pca`.
            Defaults to "umap".
        color: any column names or gene expression, etc. that will be used for coloring cells. Defaults to "louvain".
        color_key_cmap: not used. Left here for backward compatibility. Defaults to "Spectral".
        *args: any other positional arguments passed to `dynamo.pl.scatters`.
        **kwargs: any other keyword arguments passed to `dynamo.pl.scatters`.

    Returns:
        None would be returned in default and the plotted figure would be shown directly. If set
        `save_show_or_return='return'` as a kwarg, the axes of the plot would be returned.
    """

    return scatters(adata, basis=basis, color=color, *args, **kwargs)


def infomap(adata: AnnData, basis: str = "umap", color: str = "infomap", *args, **kwargs) -> Optional[Axes]:
    """Scatter plot for infomap community detection in selected basis.

    Args:
        adata: an Annodata object.
        basis: the reduced dimension stored in adata.obsm. The specific basis key will be constructed in the following
            priority if exits: 1) specific layer input + basis 2) X_ + basis 3) basis. E.g. if basis is PCA, `scatters`
            is going to look for 1) if specific layer is spliced, `spliced_pca` 2) `X_pca` (dynamo convention) 3) `pca`.
            Defaults to "umap".
        color: any column names or gene expression, etc. that will be used for coloring cells. Defaults to "infomap".

    Returns:
        None would be returned in default and the plotted figure would be shown directly. If set
        `save_show_or_return='return'` as a kwarg, the axes of the plot would be returned.
    """

    return scatters(adata, basis=basis, color=color, *args, **kwargs)


def streamline_clusters(
    adata: AnnData, basis: str = "umap", clusters="clusters", color_key_cmap: str = "Spectral", *args, **kwargs
) -> None:
    """Scatter plot for visualizing streamline clusters in selected basis.

    Args:
        adata: an Annodata object.
        basis: the reduced dimension stored in adata.uns. The specific basis key will be constructed as
            `"streamline_clusters_"+basis`. Defaults to "umap".
        clusters: the key of cluster data stored in the basis data. Defaults to "clusters".
        color_key_cmap: name of the pyplot color map used for plotting. Defaults to "Spectral".
        *args: not used. Left here for backward compatibility.
        **kwargs: not used. Left here for backward compatibility.
    """

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
