from typing import Union

import anndata
import numpy as np

from ..dynamo_logger import LoggerManager
from ..utils import copy_adata
from .connectivity import _gen_neighbor_keys, neighbors
from .utils_reduceDimension import prepare_dim_reduction, run_reduce_dim


def reduceDimension(
    adata: anndata.AnnData,
    X_data: np.ndarray = None,
    genes: Union[list, None] = None,
    layer: Union[str, None] = None,
    basis: Union[str, None] = "pca",
    dims: Union[list, None] = None,
    n_pca_components: int = 30,
    n_components: int = 2,
    n_neighbors: int = 30,
    reduction_method: str = "umap",
    embedding_key: Union[str, None] = None,
    neighbor_key: Union[str, None] = None,
    enforce: bool = False,
    cores: int = 1,
    copy: bool = False,
    **kwargs,
) -> Union[anndata.AnnData, None]:
    """Compute a low dimension reduction projection of an annodata object first with PCA, followed by non-linear
    dimension reduction methods

    Arguments
    ---------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        X_data:
            The user supplied data that will be used for dimension reduction directly.
        genes:
            The list of genes that will be used to subset the data for dimension reduction and clustering. If `None`,
            all genes will be used.
        layer:
            The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is used.
        basis:
            The space that will be used for clustering. Valid names includes, for example, `pca`, `umap`, `velocity_pca`
            (that is, you can use velocity for clustering), etc.
        dims:
            The list of dimensions that will be selected for clustering. If `None`, all dimensions will be used.
        n_pca_components:
            Number of input PCs (principle components) that will be used for further non-linear dimension reduction.. If n_pca_components is larger than the existing #PC in adata.obsm['X_pca'] or input layer's corresponding pca space (layer_pca), pca will be rerun with n_pca_components PCs requested.
        n_components:
            The dimension of the space to embed into.
        n_neighbors:
            Number of nearest neighbors when constructing adjacency matrix.
        reduction_method:
            Non-linear dimension reduction method to further reduce dimension based on the top n_pca_components PCA
            components. Currently, PSL
            (probablistic structure learning, a new dimension reduction by us), tSNE (fitsne instead of traditional tSNE
            used) or umap are supported.
        embedding_key:
            The str in .obsm that will be used as the key to save the reduced embedding space. By default it is None and
            embedding key is set as layer + reduction_method. If layer is None, it will be "X_neighbors".
        neighbor_key:
            The str in .uns that will be used as the key to save the nearest neighbor graph. By default it is None and
            neighbor_key key is set as layer + "_neighbors". If layer is None, it will be "X_neighbors".
        cores:
            Number of cores. Used only when the tSNE reduction_method is used.

    Returns
    -------
       adata: :class:`~anndata.AnnData`
            An new or updated anndata object, based on copy parameter, that are updated with reduced dimension data for
            data from different layers.
    """

    logger = LoggerManager.gen_logger("dynamo-dimension-reduction")
    logger.log_time()

    adata = copy_adata(adata) if copy else adata

    logger.info("retrive data for non-linear dimension reduction...", indent_level=1)
    if X_data is None:
        X_data, n_components, basis = prepare_dim_reduction(
            adata,
            genes=genes,
            layer=layer,
            basis=basis,
            dims=dims,
            n_pca_components=n_pca_components,
            n_components=n_components,
        )
    if basis[:2] + reduction_method in adata.obsm_keys():
        has_basis = True
    else:
        has_basis = False

    if has_basis and not enforce:
        logger.warning(
            f"adata already have basis {reduction_method}. dimension reduction {reduction_method} will be skipped! \n"
            f"set enforce=True to re-performing dimension reduction."
        )

    if embedding_key is None:
        embedding_key = "X_" + reduction_method if layer is None else layer + "_" + reduction_method
    if neighbor_key is None:
        neighbor_result_prefix = "" if layer is None else layer
        conn_key, dist_key, neighbor_key = _gen_neighbor_keys(neighbor_result_prefix)

    if enforce or not has_basis:
        logger.info(f"perform {reduction_method}...", indent_level=1)
        adata = run_reduce_dim(
            adata,
            X_data,
            n_components,
            n_pca_components,
            reduction_method,
            embedding_key,
            n_neighbors,
            neighbor_key,
            cores,
            **kwargs,
        )
    if neighbor_key not in adata.uns_keys():
        neighbors(adata)

    logger.finish_progress(progress_name="dimension_reduction projection")

    if copy:
        return adata
    return None


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
#             A tuple of kNN graph (`graph`), indices of nearest neighbors of each cell (knn_indicies), distances of
#             nearest
#             neighbors (knn_dists), the low dimensional embedding (embedding_) and finally the fit mapper from umap
#             which
#             can be used to transform new high dimensional data to low dimensional space or perofrm inverse transform
#             of
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
