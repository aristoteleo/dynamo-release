import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
# from sklearn.manifold import TSNE
import scipy
from scipy.stats import norm

from copy import deepcopy

from .psl import *


def extract_indices_dist_from_graph(graph, n_neighbors):
    """Extract the matrices for index, distance from the associate kNN sparse graph

    Arguments
    ---------
    graph: sparse matrix (`.X`, dtype `float32`)
        Sparse matrix of the kNN graph (n_cell x n_cell). The element in the matrix corresponds to the distance between cells.
    n_neighbors: 'int' (optional, default 15)
        The number of nearest neighbors of the kNN graph.

    Returns
    -------
    ind_mat: :class:`~numpy.ndarray`
        The matrix (n_cell x n_neighbors) that stores the indices for the each cell's n_neighbors nearest neighbors.
    dist_mat: :class:`~numpy.ndarray`
        The matrix (n_cell x n_neighbors) that stores the distances for the each cell's n_neighbors nearest neighbors.
    """

    n_cells = graph.shape[0]
    ind_mat = np.zeros((n_cells, n_neighbors), dtype=int)
    dist_mat = np.zeros((n_cells, n_neighbors), dtype=graph.dtype)

    for cur_cell in range(n_cells):
        cur_neighbors = graph[cur_cell, :].nonzero()  #returns the coordinate tuple for non-zero items

        # set itself as the nearest neighbor
        ind_mat[cur_cell, 0] = cur_cell
        dist_mat[cur_cell, 0] = 0

        # there could be more or less than n_neighbors because of an approximate search
        if len(cur_neighbors[1]) != n_neighbors - 1:
            sorted_indices = np.argsort(graph[cur_cell][:, cur_neighbors[1]].A)[0][:(n_neighbors - 1)]
            ind_mat[cur_cell, 1:] = cur_neighbors[1][sorted_indices]
            dist_mat[cur_cell, 1:] = graph[cur_cell][0, cur_neighbors[1][sorted_indices]].A
        else:
            ind_mat[cur_cell, 1:] = cur_neighbors[1] # could not broadcast input array from shape (13) into shape (14)
            dist_mat[cur_cell, 1:] = graph[cur_cell][:, cur_neighbors[1]].A

    return ind_mat, dist_mat

def umap_conn_indices_dist_embedding(X,
        n_neighbors=15,
        n_components=2,
        metric="euclidean",
        min_dist=0.1,
        random_state=0,
        verbose=False):
    """Compute connectivity graph, matrices for kNN neighbor indices, distance and low dimension embedding with UMAP.
    This code is adapted from umap-learn (https://github.com/lmcinnes/umap/blob/97d33f57459de796774ab2d7fcf73c639835676d/umap/umap_.py)

    Arguments
    ---------
    X: sparse matrix (`.X`, dtype `float32`)
        expression matrix (n_cell x n_genes)
    n_neighbors: 'int' (optional, default 15)
        The number of nearest neighbors to compute for each sample in ``X``.
    n_components: 'int' (optional, default 2)
        The dimension of the space to embed into.
    metric: 'str' or `callable` (optional, default cosine)
        The metric to use for the computation.
    min_dist: 'float' (optional, default 0.1)
        The effective minimum distance between embedded points. Smaller values will result in a more clustered/clumped
        embedding where nearby points on the manifold are drawn closer together, while larger values will result on a
        more even dispersal of points. The value should be set relative to the ``spread`` value, which determines the
        scale at which embedded points will be spread out.
    random_state: `int`, `RandomState` instance or `None`, optional (default: None)
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator is the RandomState instance used by `numpy.random`.
    verbose: `bool` (optional, default False)
        Controls verbosity of logging.

    Returns
    -------
    Returns an updated `adata` with reduced dimension data for spliced counts, projected future transcript counts 'Y_dim' and adjacency matrix when possible.
    """

    from sklearn.utils import check_random_state
    from sklearn.metrics import pairwise_distances
    from umap.umap_ import nearest_neighbors, fuzzy_simplicial_set, simplicial_set_embedding, find_ab_params

    import umap.sparse as sparse
    import umap.distances as dist

    from umap.utils import tau_rand_int, deheap_sort
    from umap.rp_tree import rptree_leaf_array, make_forest
    # https://github.com/lmcinnes/umap/blob/97d33f57459de796774ab2d7fcf73c639835676d/umap/nndescent.py
    from umap.nndescent import (
        make_nn_descent,
        make_initialisations,
        make_initialized_nnd_search,
        initialise_search,
    )
    from umap.spectral import spectral_layout

    random_state = check_random_state(42)

    _raw_data = X

    if X.shape[0] < 4096: #1
        dmat = pairwise_distances(X, metric=metric)
        graph = fuzzy_simplicial_set(
            X=dmat,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric="precomputed",
            verbose=verbose
        )
        # extract knn_indices, knn_dist
        g_tmp = deepcopy(graph)
        g_tmp[graph.nonzero()] = dmat[graph.nonzero()]
        knn_indices, knn_dists = extract_indices_dist_from_graph(g_tmp, n_neighbors=n_neighbors)

    else:
        # Standard case
        (knn_indices, knn_dists, rp_forest) = nearest_neighbors(
            X=X,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds={},
            angular=False,
            random_state=random_state,
            verbose=verbose
        )

        graph = fuzzy_simplicial_set(
            X=X,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric=metric,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
            angular=rp_forest,
            verbose=verbose
        )

        _raw_data = X
        _transform_available = True
        _search_graph = scipy.sparse.lil_matrix(
            (X.shape[0], X.shape[0]), dtype=np.int8
        )
        _search_graph.rows = knn_indices # An array (self.rows) of rows, each of which is a sorted list of column indices of non-zero elements.
        _search_graph.data = (knn_dists != 0).astype(np.int8) # The corresponding nonzero values are stored in similar fashion in self.data.
        _search_graph = _search_graph.maximum( # Element-wise maximum between this and another matrix.
            _search_graph.transpose()
        ).tocsr()

    if verbose:
        print("Construct embedding")

    a, b = find_ab_params(1, min_dist)

    embedding_ = simplicial_set_embedding(
        data=_raw_data,
        graph=graph,
        n_components=n_components,
        initial_alpha=1.0, # learning_rate
        a=a,
        b=b,
        gamma=1.0,
        negative_sample_rate=5,
        n_epochs=0,
        init="spectral",
        random_state=random_state,
        metric=metric,
        metric_kwds={},
        verbose=verbose
    )

    return graph, knn_indices, knn_dists, embedding_


def reduceDimension(adata, n_pca_components=25, n_components=2, n_neighbors=10, reduction_method='trimap', velocity_key='velocity_S', cores=1):
    """Compute a low dimension reduction projection of an annodata object first with PCA, followed by non-linear dimension reduction methods

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        an Annodata object 
    n_pca_components: 'int' (optional, default 50)
        Number of PCA components.  
    n_components: 'int' (optional, default 50)
        The dimension of the space to embed into.
    n_neighbors: 'int' (optional, default 10)
        Number of nearest neighbors when constructing adjacency matrix. 
    reduction_method: 'str' (optional, default trimap)
        Non-linear dimension reduction method to further reduce dimension based on the top n_pca_components PCA components. Currently, PSL 
        (probablistic structure learning, a new dimension reduction by us), tSNE (fitsne instead of traditional tSNE used) or umap are supported.
    velocity_key: 'str' (optional, default velocity_S)
        The dictionary key that corresponds to the estimated velocity values. 
    cores: `int` (optional, default `1`)
        Number of cores. Used only when the tSNE reduction_method is used.

    Returns
    -------
    Returns an updated `adata` with reduced dimension data for spliced counts, projected future transcript counts 'Y_dim' and adjacency matrix when possible.
    """

    n_obs = adata.shape[0]

    if 'use_for_dynamo' in adata.var.keys():
        X = adata.X[:, adata.var.use_for_dynamo.values]
        if velocity_key is not None:
            X_t = adata.X[:, adata.var.use_for_dynamo.values] + adata.layers[velocity_key][:, adata.var.use_for_dynamo.values]
    else:
        X = adata.X
        if velocity_key is not None:
            X_t = adata.X + adata.layers[velocity_key]

    if((not 'X_pca' in adata.obsm.keys()) or 'pca_fit' not in adata.uns.keys()):
        transformer = TruncatedSVD(n_components=n_pca_components, random_state=0)
        X_fit = transformer.fit(X)
        X_pca = X_fit.transform(X)
        adata.obsm['X_pca'] = X_pca
        if velocity_key is not None and "velocity_pca" not in adata.obsm.keys():
            X_t_pca = X_fit.transform(X_t)
            adata.obsm['velocity_pca'] = X_t_pca - X_pca
    else:
        X_pca = adata.obsm['X_pca']
        if velocity_key is not None and "velocity_pca" not in adata.obsm.keys():
            X_t_pca = adata.uns['pca_fit'].fit_transform(X_t)
            adata.obsm['velocity_pca'] = X_t_pca - X_pca

    if reduction_method is "trimap":
        import trimap
        triplemap = trimap.TRIMAP(n_inliers=20,
                                  n_outliers=10,
                                  n_random=10,
                                  weight_adj=1000.0,
                                  apply_pca=False)
        X_dim = triplemap.fit_transform(X_pca)

        adata.obsm['X_trimap'] = X_dim
        adata.uns['neighbors'] = {'params': {'n_neighbors': n_neighbors, 'method': reduction_method}, 'connectivities': None, \
                                  'distances': None, 'indices': None}
    elif reduction_method is 'tSNE':
        from fitsne import FItSNE
        X_dim=FItSNE(X_pca, nthreads=cores) # use FitSNE

        # bh_tsne = TSNE(n_components = n_components)
        # X_dim = bh_tsne.fit_transform(X_pca)
        adata.obsm['X_tSNE'] = X_dim
        adata.uns['neighbors'] = {'params': {'n_neighbors': n_neighbors, 'method': reduction_method}, 'connectivities': None, \
                                  'distances': None, 'indices': None}
    elif reduction_method is 'umap':
        graph, knn_indices, knn_dists, X_dim = umap_conn_indices_dist_embedding(X_pca) # X_pca
        adata.obsm['X_umap'] = X_dim
        adata.uns['neighbors'] = {'params': {'n_neighbors': n_neighbors, 'method': reduction_method}, 'connectivities': graph, \
                                  'distances': knn_dists, 'indices': knn_indices}
    elif reduction_method is 'psl':
        adj_mat, X_dim = psl_py(X_pca, d = n_components, K = n_neighbors) # this need to be updated
        adata.obsm['X_psl'] = X_dim
        adata.uns['PSL_adj_mat'] = adj_mat

    return adata

if __name__ == '__main__':
    import anndata
    adata = anndata.read_h5ad('/Users/xqiu/data/tmp.h5ad')

    dyn.tl.reduceDimension(tmp, velocity_key='velocity_S')
