from hdbscan import HDBSCAN
from .utils_reduceDimension import prepare_dim_reduction, run_reduce_dim
from .utils import update_dict

def hdbscan(adata,
            X_data=None,
            genes=None,
            layer=None,
            basis='pca',
            dims=None,
            n_pca_components=30,
            n_components=2,
            **hdbscan_kwargs):
    """Apply hdbscan to cluster cells in the space defined by basis.

    HDBSCAN is a clustering algorithm developed by Campello, Moulavi, and Sander
    (https://doi.org/10.1007/978-3-642-37456-2_14) which extends DBSCAN by converting
    it into a hierarchical clustering algorithm, followed by using a technique to extract
    a flat clustering based in the stability of clusters. Here you can use hdbscan to
    cluster your data in any space specified by `basis`. The data that used to produced
    from this space can be specified by `layer`. Thus, you are able to use either the
    unspliced or new RNA data for dimension reduction and clustering. HDBSCAN is a density
    based method, it thus requires you to perform clustering on relatively low dimension,
    for example top 30 PCs or top 5 umap dimension with at least several thousands of cells.
    In practice, HDBSCAN will assign -1 for cells that have low local density and thus not
    able to confidentially assign to any clusters.

    The hdbscan package from Leland McInnes, John Healy, Steve Astels Revision is used.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        AnnData object.
    X_data: `np.ndarray` (default: `None`)
        The user supplied data that will be used for clustering directly.
    genes: `list` or None (default: `None`)
        The list of genes that will be used to subset the data for dimension reduction and clustering. If `None`, all
        genes will be used.
    layer: `str` or None (default: `None`)
        The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is used.
    basis: `str` or None (default: `None`)
        The space that will be used for clustering. Valid names includes, for example, `pca`, `umap`, `velocity_pca`
        (that is, you can use velocity for clustering), etc.
    dims: `list` or None (default: `None`)
        The list of dimensions that will be selected for clustering. If `None`, all dimensions will be used.
    n_pca_components: `int` (default: `30`)
        The number of pca components that will be used.
    n_components: `int` (default: `2`)
        The number of dimension that non-linear dimension reduction will be projected to.
    hdbscan_kwargs: `dict`
        Additional parameters that will be passed to hdbscan function.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            A updated AnnData object with the clustering updated. `Cluster` and `C_prob` are two newly added columns
            from .obs, corresponding to either the Cluster results or the probability of each cell belong to a cluster.
            `hdbscan` key in .uns corresponds to a dictionary that includes additional results returned from hdbscan run.
    """

    if X_data is None:
        _, n_components, has_basis, basis = prepare_dim_reduction(adata,
                              genes=genes,
                              layer=layer,
                              basis=basis,
                              dims=dims,
                              n_pca_components=n_pca_components,
                              n_components=n_components, )

    if has_basis:
        X_data = adata.obsm[basis]
    else:
        reduction_method = basis.split('_')[-1]
        embedding_key = (
            "X_" + reduction_method if layer is None else layer + "_" + reduction_method
        )
        neighbor_key = "neighbors" if layer is None else layer + "_neighbors"

        adata = run_reduce_dim(adata, X_data, n_components, n_pca_components, reduction_method,
                               embedding_key=embedding_key, n_neighbors=30, neighbor_key=neighbor_key, cores=1)

    X_data = X_data if dims is None else X_data[:, dims]

    if hdbscan_kwargs is not None and 'metric' in hdbscan_kwargs.keys():
        if hdbscan_kwargs['metric'] == 'cosine':
            from sklearn.preprocessing import normalize
            X_data = normalize(X_data, norm='l2')

    h_kwargs = {"min_cluster_size": 5,
                "min_samples": None,
                "metric": "euclidean",
                "p": None,
                "alpha": 1.0,
                "cluster_selection_epsilon": 0.0,
                "algorithm": 'best',
                "leaf_size": 40,
                "approx_min_span_tree": True,
                "gen_min_span_tree": False,
                "core_dist_n_jobs": 1,
                "cluster_selection_method": 'eom',
                "allow_single_cluster": False,
                "prediction_data": False,
                "match_reference_implementation": False,
                }

    h_kwargs = update_dict(h_kwargs, hdbscan_kwargs)
    cluster = HDBSCAN(**h_kwargs)
    cluster.fit(X_data)
    adata.obs['Cluster'] = cluster.labels_.astype('str')
    adata.obs['C_prob'] = cluster.probabilities_
    adata.uns['hdbscan'] = {'Cluster': cluster.labels_.astype('str'),
                            "probabilities_": cluster.probabilities_,
                            "cluster_persistence_": cluster.cluster_persistence_,
                            "outlier_scores_": cluster.outlier_scores_,
                            "exemplars_": cluster.exemplars_,
                            }

    return adata

