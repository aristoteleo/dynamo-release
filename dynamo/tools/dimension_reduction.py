import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
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

        # there could be more than n_neighbors because of an approximate search
        if len(cur_neighbors[1]) > n_neighbors - 1:
            sorted_indices = np.argsort(graph[cur_cell][:, cur_neighbors[1]].A)[0][:(n_neighbors - 1)]
            ind_mat[cur_cell, 1:] = cur_neighbors[1][sorted_indices]
            dist_mat[cur_cell, 1:] = graph[cur_cell][0, cur_neighbors[1][sorted_indices]].A
        else:
            ind_mat[cur_cell, 1:] = cur_neighbors[1]
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
    from umap.nndescent import (
        make_nn_descent,
        make_initialisations,
        make_initialized_nnd_search,
        initialise_search,
    )
    from umap.spectral import spectral_layout

    INT32_MIN = np.iinfo(np.int32).min + 1
    INT32_MAX = np.iinfo(np.int32).max - 1
    random_state = check_random_state(42)

    _raw_data = X

    if X.shape[0] < 4096: #
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
        knn_indices, knn_dists = extract_indices_dist_from_graph(g_tmp, n_neighbors = n_neighbors)

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
        _search_graph.rows = knn_indices
        _search_graph.data = (knn_dists != 0).astype(np.int8)
        _search_graph = _search_graph.maximum(
            _search_graph.transpose()
        ).tocsr()

        if callable(metric):
            _distance_func = metric
        elif metric in dist.named_distances:
            _distance_func = dist.named_distances[metric]
        else:
            raise ValueError(
                "Metric is neither callable, " + "nor a recognised string"
            )
        _dist_args = tuple({}.values())

        _random_init, _tree_init = make_initialisations(
            _distance_func, _dist_args
        )
        _search = make_initialized_nnd_search(
            _distance_func, _dist_args
        )

    if verbose:
        print("Construct embedding")

    a, b = find_ab_params(1, min_dist)
    # rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

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


def reduceDimension(adata, n_pca_components = 25, n_components = 2, velocity_method = None, n_neighbors = 10, reduction_method='UMAP', velocity_key = 'velocity'): # c("UMAP", 'tSNE', "DDRTree", "ICA", 'none')
    """Compute a low dimension reduction projection of an annodata object first with PCA, followed by non-linear dimension reduction methods

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        an Annodata object 
    n_pca_components: 'int' (optional, default 50)
        Number of PCA components.  
    n_components: 'int' (optional, default 50)
        The dimension of the space to embed into.
    velocity_method: 'str' (optional, default None)
        Which method to learn the 2D velocity projection.
    n_neighbors: 'int' (optional, default 10)
        Number of nearest neighbors when constructing adjacency matrix. 
    reduction_method: 'str' (optional, default PSL)
        Non-linear dimension reduction method to further reduce dimension based on the top n_pca_components PCA components. Currently, PSL 
        (probablistic structure learning, a new dimension reduction by us), tSNE or UMAP are supported. 
    velocity_key: 'str' (optional, default velocity)
        The dictionary key that corresponds to the estimated velocity values. 

    Returns
    -------
    Returns an updated `adata` with reduced dimension data for spliced counts, projected future transcript counts 'Y_dim' and adjacency matrix when possible.
    """

    n_obs = adata.shape[0]

    X = adata.X

    if(not 'X_pca' in adata.obsm.keys()):
        transformer = TruncatedSVD(n_components=n_pca_components, random_state=0)
        X_pca = transformer.fit(X.T).components_.T
        adata.obsm['X_pca'] = X_pca
    else:
        X_pca = adata.obsm['X_pca']

    if reduction_method is 'tSNE':
        bh_tsne = TSNE(n_components = n_components)
        X_dim = bh_tsne.fit_transform(X_pca)
        adata.obsm['X_tSNE'] = X_dim
    elif reduction_method is 'UMAP':
        graph, knn_indices, knn_dists, X_dim = umap_conn_indices_dist_embedding(X) # X_pca
        adata.obsm['X_umap'] = X_dim.copy()
        adata.uns['neighbors']['connectivities'] = graph
        adata.uns['neighbors']['distances'] = knn_dists
        adata.uns['neighbors']['indices'] = knn_indices
    elif reduction_method is 'PSL':
        adj_mat, X_dim = psl_py(X_pca, d = n_components, K = n_neighbors) # this need to be updated
        adata.obsm['X_psl'] = X_dim
        adata.uns['PSL_adj_mat'] = adj_mat

    # use both existing data and predicted future states in dimension reduction to get the velocity plot in 2D
    # use only the existing data for dimension reduction and then project new data in this reduced dimension
    if velocity_method is not 'UMAP':
        if n_neighbors is None: n_neighbors = int(n_obs / 50)
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        nn.fit(adata.X)
        tmp = adata.X + adata.layers[velocity_key]
        dists, neighs = nn.kneighbors(tmp)
        scale = np.median(dists, axis=1)
        weight = norm.pdf(x = dists, scale=scale[:, None])
        p_mass = weight.sum(1)
        weight = weight / p_mass[:, None]

        # calculate the embedding for the predicted future states of cells using a Gaussian kernel
        Y_dim = (X_dim[neighs] * (weight[:, :, None])).sum(1)
        adata.obsm['Y_dim'] = Y_dim
    elif velocity_method is 'UMAP':
        tmp = adata.X + adata.layers[velocity_key]
        tmp[tmp < 0] = 0

        test_embedding = X_umap.transform(tmp) # use umap's transformer to get the embedding points of future states
        adata.obsm['Y_dim'] = test_embedding

    return adata

