import numpy as np
import scipy
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
import warnings
from copy import deepcopy
from inspect import signature
from sklearn.utils import sparsefuncs
from ..preprocessing.utils import get_layer_keys
from .utils import get_mapper

def extract_indices_dist_from_graph(graph, n_neighbors):
    """Extract the matrices for index, distance from the associated kNN sparse graph

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
        cur_neighbors = graph[cur_cell, :].nonzero()  # returns the coordinate tuple for non-zero items

        # set itself as the nearest neighbor
        ind_mat[cur_cell, :] = cur_cell
        dist_mat[cur_cell, :] = 0

        # there could be more or less than n_neighbors because of an approximate search
        cur_n_neighbors = len(cur_neighbors[1])
        if cur_n_neighbors > n_neighbors - 1:
            sorted_indices = np.argsort(graph[cur_cell][:, cur_neighbors[1]].A)[0][:(n_neighbors - 1)]
            ind_mat[cur_cell, 1:] = cur_neighbors[1][sorted_indices]
            dist_mat[cur_cell, 1:] = graph[cur_cell][0, cur_neighbors[1][sorted_indices]].A
        else:
            ind_mat[cur_cell, 1:(cur_n_neighbors + 1)] = cur_neighbors[1]
            dist_mat[cur_cell, 1:(cur_n_neighbors + 1)] = graph[cur_cell][:, cur_neighbors[1]].A

    return ind_mat, dist_mat


def umap_conn_indices_dist_embedding(X,
        n_neighbors=30,
        n_components=2,
        metric="euclidean",
        min_dist=0.1,
        random_state=0,
        verbose=False):
    """Compute connectivity graph, matrices for kNN neighbor indices, distance matrix and low dimension embedding with UMAP.
    This code is adapted from umap-learn (https://github.com/lmcinnes/umap/blob/97d33f57459de796774ab2d7fcf73c639835676d/umap/umap_.py)

    Arguments
    ---------
        X: sparse matrix (`.X`, dtype `float32`)
            expression matrix (n_cell x n_genes)
        n_neighbors: 'int' (optional, default 15)
            The number of nearest neighbors to compute for each sample in ``X``.
        n_components: 'int' (optional, default 2)
            The dimension of the space to embed into.
        metric: 'str' or `callable` (optional, default `cosine`)
            The metric to use for the computation.
        min_dist: 'float' (optional, default `0.1`)
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
        graph, knn_indices, knn_dists, embedding_
            A tuple of kNN graph (`graph`), indices of nearest neighbors of each cell (knn_indicies), distances of nearest
            neighbors (knn_dists) and finally the low dimensional embedding (embedding_).
    """

    from sklearn.utils import check_random_state
    from sklearn.metrics import pairwise_distances
    from umap.umap_ import nearest_neighbors, fuzzy_simplicial_set, simplicial_set_embedding, find_ab_params

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


def mnn_from_list(knn_graph_list):
    """Apply reduce function to calculate the mutual kNN.
    """
    import functools
    mnn = functools.reduce(scipy.sparse.csr.csr_matrix.minimum, knn_graph_list) if issparse(knn_graph_list[0]) else \
        functools.reduce(scipy.minimum, knn_graph_list)

    return mnn


def normalize_knn_graph(knn):
    """normalize the knn graph so that each row will be sum up to 1.
    """
    knn.setdiag(1)
    knn = knn.astype('float32')
    sparsefuncs.inplace_row_scale(knn, 1 / knn.sum(axis=1).A1)

    return knn


def mnn(adata, n_pca_components=25, n_neighbors=250, layers='all', use_pca_fit=True, save_all_to_adata=False):
    """ Function to calculate mutual nearest neighbor graph across specific data layers.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        n_pca_components: 'int' (optional, default `25`)
            Number of PCA components.
        layers: str or list (default: `all`)
            The layer(s) to be normalized. Default is `all`, including RNA (X, raw) or spliced, unspliced, protein, etc.
        use_pca_fit: `bool` (default: `True`)
            Whether to use the precomputed pca model to transform different data layers or calculate pca for each data layer
            separately.
        save_all_to_adata: `bool` (default: `False`)
            Whether to save all calculated data to adata object.

    Returns
    -------
        adata: :AnnData
            A updated anndata object that are updated with the `mnn` or other relevant data that are calculated during mnn
            calculation.
    """
    if use_pca_fit:
        if 'pca_fit' in adata.uns.keys():
            fiter = adata.uns['pca_fit']
        else:
            raise Exception('use_pca_fit is set to be True, but there is no pca fit results in .uns attribute.')

    layers = get_layer_keys(adata, layers, False, False)
    layers = [layer for layer in layers if layer.startswith('X_') and (not layer.endswith('_matrix') and
                                                                       not layer.endswith('_ambiguous'))]
    knn_graph_list = []
    for layer in layers:
        layer_X = adata.layers[layer]
        if use_pca_fit:
            layer_pca = fiter.fit_transform(layer_X)[:, 1:]
        else:
            transformer = TruncatedSVD(n_components=n_pca_components + 1, random_state=0)
            layer_pca = transformer.fit_transform(layer_X)[:, 1:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph, knn_indices, knn_dists, X_dim = umap_conn_indices_dist_embedding(layer_pca, n_neighbors=n_neighbors)

        if save_all_to_adata:
            adata.obsm[layer + '_pca'], adata.obsm[layer + '_umap'] = layer_pca, X_dim
            n_neighbors = signature(umap_conn_indices_dist_embedding).parameters['n_neighbors']

            adata.uns[layer + '_neighbors'] = {'params': {'n_neighbors': eval(n_neighbors), 'method': 'umap'},
                                      'connectivities': graph, 'distances': knn_dists, 'indices': knn_indices}

        knn_graph_list.append(graph > 0)

    mnn = mnn_from_list(knn_graph_list)
    adata.uns['mnn'] = normalize_knn_graph(mnn)

    return adata

def smoother(adata, use_mnn=False, layers='all'):
    mapper = get_mapper()

    if use_mnn:
        if 'mnn' not in adata.uns.keys():
            adata = mnn(adata, n_pca_components=30, layers='all', use_pca_fit=True, save_all_to_adata=False)
        kNN = adata.uns['mnn']
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kNN, _, _, _ = umap_conn_indices_dist_embedding(adata.obsm['X_pca'], n_neighbors=30)

    conn = normalize_knn_graph(kNN > 0)

    layers = get_layer_keys(adata, layers, False, False)
    layers = [layer for layer in layers if layer.startswith('X_') and (not layer.endswith('_matrix') and
                                                               not layer.endswith('_ambiguous'))]

    for layer in layers:
        layer_X = adata.layers[layer].copy()

        if issparse(layer_X):
            layer_X.data = 2**layer_X.data - 1 if adata.uns['pp_log'] == 'log2' else np.exp(layer_X.data) - 1
        else:
            layer_X = 2** layer_X - 1 if adata.uns['pp_log'] == 'log2' else np.exp(layer_X) - 1

        adata.layers[mapper[layer]] = conn.dot(layer_X)

    if 'X_protein' in adata.obsm.keys(): # may need to update with mnn or just use knn from protein layer itself.
        adata.obsm[mapper['X_protein']] = conn.dot(adata.obsm['X_protein'])

    return adata

