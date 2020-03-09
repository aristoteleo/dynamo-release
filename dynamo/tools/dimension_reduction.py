import warnings
from .psl import *

from .utils import update_dict
from .connectivity import (
    umap_conn_indices_dist_embedding,
    extract_indices_dist_from_graph,
)
from ..preprocessing.utils import pca


def reduceDimension(
    adata,
    layer="X",
    n_pca_components=30,
    n_components=2,
    n_neighbors=30,
    reduction_method="umap",
    cores=1,
    **kwargs
):

    """Compute a low dimension reduction projection of an annodata object first with PCA, followed by non-linear dimension reduction methods

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    layer: str (default: X)
            The layer where the dimension reduction will be performed.
    n_pca_components: 'int' (optional, default 30)
        Number of PCA components.  
    n_components: 'int' (optional, default 2)
        The dimension of the space to embed into.
    n_neighbors: 'int' (optional, default 30)
        Number of nearest neighbors when constructing adjacency matrix. 
    reduction_method: 'str' (optional, default umap)
        Non-linear dimension reduction method to further reduce dimension based on the top n_pca_components PCA components. Currently, PSL 
        (probablistic structure learning, a new dimension reduction by us), tSNE (fitsne instead of traditional tSNE used) or umap are supported.
    cores: `int` (optional, default `1`)
        Number of cores. Used only when the tSNE reduction_method is used.

    Returns
    -------
    Returns an updated `adata` with reduced dimension data for spliced counts, projected future transcript counts 'Y_dim'.
    """

    layer = layer if layer.startswith("X") else "X_" + layer
    if layer is not "X" and layer not in adata.layers.keys():
        raise Exception(
            "The layer {} you provided is not existed in adata.".format(layer)
        )
    pca_key = "X_pca" if layer == "X" else layer + "_pca"
    embedding_key = (
        "X_" + reduction_method if layer == "X" else layer + "_" + reduction_method
    )
    neighbor_key = "neighbors" if layer == "X" else layer + "_neighbors"

    if "use_for_dynamo" in adata.var.keys():
        if layer == "X":
            X = adata.X[:, adata.var.use_for_dynamo.values]
        else:
            X = adata[:, adata.var.use_for_dynamo.values].layers[layer]
    else:
        if layer == "X":
            X = adata.X
        else:
            X = adata.layers[layer]

    if layer == "X":
        if (
            (pca_key not in adata.obsm.keys()) or "pca_fit" not in adata.uns.keys()
        ) or reduction_method is "pca":
            adata, _, X_pca = pca(adata, X, n_pca_components, pca_key)
        else:
            X_pca = adata.obsm[pca_key][:, :n_pca_components]
            adata.obsm[pca_key] = X_pca
    else:
        if (pca_key not in adata.obsm.keys()) or reduction_method is "pca":
            adata, _, X_pca = pca(adata, X, n_pca_components, pca_key)
        else:
            X_pca = adata.obsm[pca_key][:, :n_pca_components]
            adata.obsm[pca_key] = X_pca

    if reduction_method == "trimap":
        import trimap

        triplemap = trimap.TRIMAP(
            n_inliers=20,
            n_outliers=10,
            n_random=10,
            distance="euclidean",  # cosine
            weight_adj=1000.0,
            apply_pca=False,
        )
        X_dim = triplemap.fit_transform(X_pca)

        adata.obsm[embedding_key] = X_dim
        adata.uns[neighbor_key] = {
            "params": {"n_neighbors": n_neighbors, "method": reduction_method},
            "connectivities": None,
            "distances": None,
            "indices": None,
        }
    elif reduction_method == "diffusion_map":
        pass
    elif reduction_method == "tSNE":
        try:
            from fitsne import FItSNE
        except ImportError:
            print(
                "Please first install fitsne to perform accelerated tSNE method. Install instruction is provided here: https://pypi.org/project/fitsne/"
            )

        X_dim = FItSNE(X_pca, nthreads=cores)  # use FitSNE

        # bh_tsne = TSNE(n_components = n_components)
        # X_dim = bh_tsne.fit_transform(X_pca)
        adata.obsm[embedding_key] = X_dim
        adata.uns[neighbor_key] = {
            "params": {"n_neighbors": n_neighbors, "method": reduction_method},
            "connectivities": None,
            "distances": None,
            "indices": None,
        }
    elif reduction_method == "umap":
        _umap_kwargs = {
            "n_components": 2,
            "metric": "euclidean",
            "min_dist": 0.1,
            "spread": 1.0,
            "n_epochs": 0,
            "alpha": 1.0,
            "gamma": 1.0,
            "negative_sample_rate": 5,
            "init_pos": "spectral",
            "random_state": 0,
            "verbose": False,
        }
        umap_kwargs = update_dict(_umap_kwargs, kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (
                mapper,
                graph,
                knn_indices,
                knn_dists,
                X_dim,
            ) = umap_conn_indices_dist_embedding(
                X_pca, n_neighbors, **umap_kwargs
            )  # X_pca

        adata.obsm[embedding_key] = X_dim
        adata.uns[neighbor_key] = {
            "params": {"n_neighbors": n_neighbors, "method": reduction_method},
            "connectivities": graph,
            "distances": knn_dists,
            "indices": knn_indices,
        }
        adata.uns["umap_fit"] = {"fit": mapper, "n_pca_components": n_pca_components}
    elif reduction_method is "psl":
        adj_mat, X_dim = psl_py(
            X_pca, d=n_components, K=n_neighbors
        )  # this need to be updated
        adata.obsm[embedding_key] = X_dim
        adata.uns[neighbor_key] = adj_mat

    else:
        raise Exception(
            "reduction_method {} is not supported.".format(reduction_method)
        )

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
