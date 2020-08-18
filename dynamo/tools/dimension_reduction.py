import warnings
from .utils_reduceDimension import prepare_dim_reduction, run_reduce_dim
from .connectivity import neighbors


def reduceDimension(
    adata,
    X_data=None,
    genes=None,
    layer=None,
    basis='pca',
    dims=None,
    n_pca_components=30,
    n_components=2,
    n_neighbors=30,
    reduction_method="umap",
    enforce=False,
    cores=1,
    **kwargs
):

    """Compute a low dimension reduction projection of an annodata object first with PCA, followed by non-linear
    dimension reduction methods

    Arguments
    ---------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        X_data: `np.ndarray` (default: `None`)
            The user supplied data that will be used for dimension reduction directly.
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
        n_pca_components: 'int' (optional, default 30)
            Number of PCA components.
        n_components: 'int' (optional, default 2)
            The dimension of the space to embed into.
        n_neighbors: 'int' (optional, default 30)
            Number of nearest neighbors when constructing adjacency matrix.
        reduction_method: 'str' (optional, default umap)
            Non-linear dimension reduction method to further reduce dimension based on the top n_pca_components PCA
            components. Currently, PSL
            (probablistic structure learning, a new dimension reduction by us), tSNE (fitsne instead of traditional tSNE
            used) or umap are supported.
        cores: `int` (optional, default `1`)
            Number of cores. Used only when the tSNE reduction_method is used.

    Returns
    -------
        Returns an updated `adata` with reduced dimension data for data from different layers.
    """

    if X_data is None:
        X_data, n_components, has_basis, _ = prepare_dim_reduction(adata,
                              genes=genes,
                              layer=layer,
                              basis=reduction_method,
                              dims=dims,
                              n_pca_components=n_pca_components,
                              n_components=n_components, )
    else:
        has_basis = False

    if has_basis and not enforce:
        warnings.warn(f"adata already have basis {reduction_method}. dimension reduction {reduction_method} will be skipped! \n"
                      f"set enforce=True to re-performing dimension reduction.")

    embedding_key = (
        "X_" + reduction_method if layer is None else layer + "_" + reduction_method
    )
    neighbor_key = "neighbors" if layer is None else layer + "_neighbors"

    if enforce or not has_basis:
       adata = run_reduce_dim(adata, X_data, n_components, n_pca_components, reduction_method, embedding_key,
                              n_neighbors, neighbor_key, cores, kwargs)
    if neighbor_key not in adata.uns_keys():
        neighbors(adata)

    return adata


# @docstrings.with_indent(4)
# def run_umap(X,
#         n_neighbors=30,
#         n_components=2,
#         metric="euclidean",
#         min_dist=0.1,
#         spread=1.0,
#         n_epochs=None,
#         alpha=1.0,
#         gamma=1.0,
#         negative_sample_rate=5,
#         init_pos='spectral',
#         random_state=0,
#         verbose=False, **umap_kwargs):
#     """Perform umap analysis.
#
#     Parameters
#     ----------
#     %(umap_ann.parameters)s
#
#     Returns
#     -------
#         graph, knn_indices, knn_dists, embedding_, mapper
#             A tuple of kNN graph (`graph`), indices of nearest neighbors of each cell (knn_indicies), distances of nearest
#             neighbors (knn_dists), the low dimensional embedding (embedding_) and finally the fit mapper from umap which
#             can be used to transform new high dimensional data to low dimensional space or perofrm inverse transform of
#             new data back to high dimension.
#     """
#
#     _umap_kwargs={"angular_rp_forest": False,  "local_connectivity": 1.0, "metric_kwds": None,
#                  "set_op_mix_ratio": 1.0, "target_metric": 'categorical', "target_metric_kwds": None,
#                  "target_n_neighbors": -1, "target_weight": 0.5, "transform_queue_size": 4.0,
#                  "transform_seed": 42}
#     umap_kwargs = update_dict(_umap_kwargs, umap_kwargs)
#
#     mapper = umap.UMAP(n_neighbors=n_neighbors,
#                        n_components=n_components,
#                        metric=metric,
#                        min_dist=min_dist,
#                        spread=spread,
#                        n_epochs=n_epochs,
#                        learning_rate=alpha,
#                        repulsion_strength=gamma,
#                        negative_sample_rate=negative_sample_rate,
#                        init=init_pos,
#                        random_state = random_state,
#                        verbose=verbose,
#                        **umap_kwargs
#     ).fit(X)
#
#     dmat = pairwise_distances(X, metric=metric)
#     graph = fuzzy_simplicial_set(
#         X=dmat,
#         n_neighbors=n_neighbors,
#         random_state=random_state,
#         metric="precomputed",
#         verbose=verbose
#     )
#     # extract knn_indices, knn_dist
#     g_tmp = deepcopy(graph)
#     g_tmp[graph.nonzero()] = dmat[graph.nonzero()]
#     knn_indices, knn_dists = extract_indices_dist_from_graph(g_tmp, n_neighbors=n_neighbors)
#
#     knn_indices, knn_dists = extract_indices_dist_from_graph(mapper.graph_, n_neighbors=n_neighbors)
#
#     return mapper.graph_, knn_dists, knn_indices, mapper.transform(X), mapper
