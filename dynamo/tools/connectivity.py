import numpy as np
import scipy
from scipy.sparse import issparse, csr_matrix
from sklearn.decomposition import TruncatedSVD
import warnings
from copy import deepcopy
from inspect import signature
from sklearn.utils import sparsefuncs
from ..preprocessing.utils import get_layer_keys
from .utils import (
    log1p_,
    fetch_X_data,
)

from ..docrep import DocstringProcessor

docstrings = DocstringProcessor()


def adj_to_knn(adj, n_neighbors):
    """convert the adjacency matrix of a nearest neighbor graph to the indices
        and weights for a knn graph.

    Arguments
    ---------
        adj: matrix (`.X`, dtype `float32`)
            Adjacency matrix (n x n) of the nearest neighbor graph.
        n_neighbors: 'int' (optional, default 15)
            The number of nearest neighbors of the kNN graph.

    Returns
    -------
        idx: :class:`~numpy.ndarray`
            The matrix (n x n_neighbors) that stores the indices for each node's 
            n_neighbors nearest neighbors.
        wgt: :class:`~numpy.ndarray`
            The matrix (n x n_neighbors) that stores the weights on the edges 
            for each node's n_neighbors nearest neighbors.
    """

    n_cells = adj.shape[0]
    idx = np.zeros((n_cells, n_neighbors), dtype=int)
    wgt = np.zeros((n_cells, n_neighbors), dtype=adj.dtype)

    for cur_cell in range(n_cells):
        cur_neighbors = adj[cur_cell, :].nonzero()  # returns the coordinate tuple for non-zero items

        # set itself as the nearest neighbor
        idx[cur_cell, :] = cur_cell
        wgt[cur_cell, :] = 0

        # there could be more or less than n_neighbors because of an approximate search
        cur_n_neighbors = len(cur_neighbors[1])
        if cur_n_neighbors > n_neighbors - 1:
            sorted_indices = np.argsort(adj[cur_cell][:, cur_neighbors[1]].A)[0][
                : (n_neighbors - 1)
            ]
            idx[cur_cell, 1:] = cur_neighbors[1][sorted_indices]
            wgt[cur_cell, 1:] = adj[cur_cell][
                0, cur_neighbors[1][sorted_indices]
            ].A
        else:
            idx[cur_cell, 1 : (cur_n_neighbors + 1)] = cur_neighbors[1]
            wgt[cur_cell, 1 : (cur_n_neighbors + 1)] = adj[cur_cell][
                :, cur_neighbors[1]
            ].A

    return idx, wgt


def knn_to_adj(knn_indices, knn_weights):
    adj = csr_matrix((knn_weights.flatten(),
                    (np.repeat(knn_indices[:, 0], knn_indices.shape[1]),
                     knn_indices.flatten())))
    adj.eliminate_zeros()

    return adj


def get_conn_dist_graph(knn, distances):
    n_obs, n_neighbors = knn.shape
    distances = csr_matrix((distances.flatten(), (np.repeat(np.arange(n_obs), n_neighbors), knn.flatten())),
                           shape=(n_obs, n_obs))
    connectivities = distances.copy()
    connectivities.data[connectivities.data > 0] = 1

    distances.eliminate_zeros()
    connectivities.eliminate_zeros()

    return distances, connectivities


@docstrings.get_sectionsf("umap_ann")
def umap_conn_indices_dist_embedding(
    X,
    n_neighbors=30,
    n_components=2,
    metric="euclidean",
    min_dist=0.1,
    spread=1.0,
    n_epochs=0,
    alpha=1.0,
    gamma=1.0,
    negative_sample_rate=5,
    init_pos="spectral",
    random_state=0,
    return_mapper=True,
    verbose=False,
    **umap_kwargs
):
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
        spread: `float` (optional, default 1.0)
            The effective scale of embedded points. In combination with min_dist this determines how clustered/clumped the
            embedded points are.
        n_epochs: 'int' (optional, default 0)
            The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in
            more accurate embeddings. If None is specified a value will be selected based on the size of the input dataset
            (200 for large datasets, 500 for small).
        alpha: `float` (optional, default 1.0)
            Initial learning rate for the SGD.
        gamma: `float` (optional, default 1.0)
            Weight to apply to negative samples. Values higher than one will result in greater weight being given to
            negative samples.
        negative_sample_rate: `float` (optional, default 5)
            The number of negative samples to select per positive sample in the optimization process. Increasing this value
             will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
             The number of negative edge/1-simplex samples to use per positive edge/1-simplex sample in optimizing the low
             dimensional embedding.
        init_pos: 'spectral':
            How to initialize the low dimensional embedding. Use a spectral embedding of the fuzzy 1-skeleton
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
    from umap.umap_ import (
        nearest_neighbors,
        fuzzy_simplicial_set,
        simplicial_set_embedding,
        find_ab_params,
    )

    random_state = check_random_state(random_state)

    _raw_data = X

    if X.shape[0] < 4096:  # 1
        dmat = pairwise_distances(X, metric=metric)
        graph = fuzzy_simplicial_set(
            X=dmat,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric="precomputed",
            verbose=verbose,
        )
        if type(graph) == tuple: graph = graph[0]

        # extract knn_indices, knn_dist
        g_tmp = deepcopy(graph)
        g_tmp[graph.nonzero()] = dmat[graph.nonzero()]
        knn_indices, knn_dists = adj_to_knn(
            g_tmp, n_neighbors=n_neighbors
        )
    else:
        # Standard case
        (knn_indices, knn_dists, rp_forest) = nearest_neighbors(
            X=X,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds={},
            angular=False,
            random_state=random_state,
            verbose=verbose,
        )

        graph = fuzzy_simplicial_set(
            X=X,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric=metric,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
            angular=rp_forest,
            verbose=verbose,
        )

        _raw_data = X
        _transform_available = True
        # The corresponding nonzero values are stored in similar fashion in self.data.
        _search_graph, _ = get_conn_dist_graph(knn_indices, knn_dists)
        _search_graph = _search_graph.maximum(  # Element-wise maximum between this and another matrix.
            _search_graph.transpose()
        ).tocsr()

    if verbose:
        print("Construct embedding")

    a, b = find_ab_params(spread, min_dist)
    if type(graph) == tuple: graph = graph[0]
    embedding_ = simplicial_set_embedding(
        data=_raw_data,
        graph=graph,
        n_components=n_components,
        initial_alpha=alpha,  # learning_rate
        a=a,
        b=b,
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        n_epochs=n_epochs,
        init=init_pos,
        random_state=random_state,
        metric=metric,
        metric_kwds={},
        verbose=verbose,
    )

    if return_mapper:
        import umap
        from .utils import update_dict

        if n_epochs == 0:
            n_epochs = None

        _umap_kwargs = {
            "angular_rp_forest": False,
            "local_connectivity": 1.0,
            "metric_kwds": None,
            "set_op_mix_ratio": 1.0,
            "target_metric": "categorical",
            "target_metric_kwds": None,
            "target_n_neighbors": -1,
            "target_weight": 0.5,
            "transform_queue_size": 4.0,
            "transform_seed": 42,
        }
        umap_kwargs = update_dict(_umap_kwargs, umap_kwargs)

        mapper = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=metric,
            min_dist=min_dist,
            spread=spread,
            n_epochs=n_epochs,
            learning_rate=alpha,
            repulsion_strength=gamma,
            negative_sample_rate=negative_sample_rate,
            init=init_pos,
            random_state=random_state,
            verbose=verbose,
            **umap_kwargs
        ).fit(X)

        return mapper, graph, knn_indices, knn_dists, embedding_
    else:
        return graph, knn_indices, knn_dists, embedding_


def mnn_from_list(knn_graph_list):
    """Apply reduce function to calculate the mutual kNN.
    """
    import functools

    mnn = (
        functools.reduce(scipy.sparse.csr.csr_matrix.minimum, knn_graph_list)
        if issparse(knn_graph_list[0])
        else functools.reduce(scipy.minimum, knn_graph_list)
    )

    return mnn


def normalize_knn_graph(knn):
    """normalize the knn graph so that each row will be sum up to 1.
    """
    knn.setdiag(1)
    knn = knn.astype("float32")
    sparsefuncs.inplace_row_scale(knn, 1 / knn.sum(axis=1).A1)

    return knn


def mnn(
    adata,
    n_pca_components=30,
    n_neighbors=250,
    layers="all",
    use_pca_fit=True,
    save_all_to_adata=False,
):
    """ Function to calculate mutual nearest neighbor graph across specific data layers.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        n_pca_components: 'int' (optional, default `30`)
            Number of PCA components.
        layers: str or list (default: `all`)
            The layer(s) to be normalized. Default is `all`, including RNA (X, raw) or spliced, unspliced, protein, etc.
        use_pca_fit: `bool` (default: `True`)
            Whether to use the precomputed pca model to transform different data layers or calculate pca for each data layer
            separately.
        save_all_to_adata: `bool` (default: `False`)
            Whether to save_fig all calculated data to adata object.

    Returns
    -------
        adata: :AnnData
            A updated anndata object that are updated with the `mnn` or other relevant data that are calculated during mnn
            calculation.
    """
    if use_pca_fit:
        if "pca_fit" in adata.uns.keys():
            fiter = adata.uns["pca_fit"]
        else:
            raise Exception(
                "use_pca_fit is set to be True, but there is no pca fit results in .uns attribute."
            )

    layers = get_layer_keys(adata, layers, False, False)
    layers = [
        layer
        for layer in layers
        if layer.startswith("X_")
        and (not layer.endswith("_matrix") and not layer.endswith("_ambiguous"))
    ]
    knn_graph_list = []
    for layer in layers:
        layer_X = adata.layers[layer]
        layer_X = log1p_(adata, layer_X)
        if use_pca_fit:
            layer_pca = fiter.fit_transform(layer_X)[:, 1:]
        else:
            transformer = TruncatedSVD(
                n_components=n_pca_components + 1, random_state=0
            )
            layer_pca = transformer.fit_transform(layer_X)[:, 1:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph, knn_indices, knn_dists, X_dim = umap_conn_indices_dist_embedding(
                layer_pca, n_neighbors=n_neighbors, return_mapper=False
            )

        if save_all_to_adata:
            adata.obsm[layer + "_pca"], adata.obsm[layer + "_umap"] = layer_pca, X_dim
            n_neighbors = signature(umap_conn_indices_dist_embedding).parameters[
                "n_neighbors"
            ]

            adata.uns[layer + "_neighbors"] = {
                "params": {"n_neighbors": eval(n_neighbors), "method": "umap"},
                # "connectivities": None,
                # "distances": None,
                "indices": knn_indices,
            }
            adata.obsp[layer + "_connectivities"], adata.obsp[layer + "_distances"] = graph, knn_dists

        knn_graph_list.append(graph > 0)

    mnn = mnn_from_list(knn_graph_list)
    adata.uns["mnn"] = normalize_knn_graph(mnn)

    return adata


def neighbors(
    adata,
    X_data=None,
    genes=None,
    basis='pca',
    layer=None,
    n_pca_components=30,
    n_neighbors=30,
    method=None,
    metric="euclidean",
    metric_kwads=None,
    cores=1,
    seed=19491001,
    **kwargs

):
    """Function to search nearest neighbors of the adata object.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        X_data: `np.ndarray` (default: `None`)
            The user supplied data that will be used for nearest neighbor search directly.
        genes: `list` or None (default: `None`)
            The list of genes that will be used to subset the data for nearest neighbor search. If `None`, all genes
            will be used.
        basis: `str` (default: `pca`)
            The space that will be used for nearest neighbor search. Valid names includes, for example, `pca`, `umap`,
            `velocity_pca` or `X` (that is, you can use velocity for clustering), etc.
        layers: str or list (default: `all`)
            The layer to be used for nearest neighbor search.
        n_pca_components: 'int' (optional, default `30`)
            Number of PCA components. Applicable only if you will use pca `basis` for nearest neighbor search.
        n_neighbors: `int` (optional, default `30`)
            Number of nearest neighbors.
        method: `str` or `None` (default: `None`)
            The methoed used for nearest neighbor search. If `umap` or `pynn`, it relies on `pynndescent` package's
            NNDescent for fast nearest neighbor search.
        metric: `str` or callable, default='euclidean'
            The distance metric to use for the tree.  The default metric is , and with p=2 is equivalent to the standard
            Euclidean metric. See the documentation of :class:`DistanceMetric` for a list of available metrics. If metric
            is "precomputed", X is assumed to be a distance matrix and must be square during fit. X may be a
            :term:`sparse graph`, in which case only "nonzero" elements may be considered neighbors.
        metric_params : dict, default=None
            Additional keyword arguments for the metric function.
        cores: `int` (default: 1)
            The number of parallel jobs to run for neighbors search. ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.
        seed: `int` (default `19491001`)
            Random seed to ensure the reproducibility of each run.
        kwargs:
            Additional arguments that will be passed to each nearest neighbor search algorithm.

    Returns
    -------
        adata: :AnnData
            A updated anndata object that are updated with the `indices`, `connectivity`, `distance` to the .obsp, as well
            as a new `neighbors` key in .uns.
    """

    if X_data is None:
        if basis == 'pca' and 'X_pca' not in adata.obsm_keys():
            from ..preprocessing.utils import pca
            CM = adata.X if genes is None else adata[:, genes].X
            cm_genesums = CM.sum(axis=0)
            valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
            valid_ind = np.array(valid_ind).flatten()
            CM = CM[:, valid_ind]
            adata, _, _ = pca(adata, CM, n_pca_components=n_pca_components)

            X_data = adata.obsm['X_pca']
        else:
            genes, X_data = fetch_X_data(adata, genes, layer, basis)

    if method is None:
        if X_data.shape[0] > 200000 and X_data.shape[1] > 2:
            from pynndescent import NNDescent
            method = 'pynn'
        elif X_data.shape[1] > 10:
            method = 'ball_tree' 
        else:
            method = 'kd_tree'

    # may distinguish between umap and pynndescent -- treat them equal for now
    if method.lower() in ['pynn', 'umap']:
        index = NNDescent(X_data, metric=metric, metric_kwads=metric_kwads, n_neighbors=n_neighbors, n_jobs=cores,
                          random_state=seed, **kwargs)
        knn, distances = index.query(X_data, k=n_neighbors)
    elif method in ['ball_tree', 'kd_tree']:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, metric_params=metric_kwads, algorithm=method,
                                n_jobs=cores, **kwargs).fit(X_data)
        distances, knn = nbrs.kneighbors(X_data)
    else:
        raise ImportError(f'nearest neighbor search method {method} is not supported')


    adata.obsp["connectivities"], adata.obsp["distances"] = get_conn_dist_graph(knn, distances)

    adata.uns['neighbors'] = {}
    adata.uns['neighbors']["indices"] =  knn
    adata.uns["neighbors"]["params"] = {
        "n_neighbors": n_neighbors,
        "method": method,
        "metric": metric,
        "n_pcs": n_pca_components,
    }

    return adata
