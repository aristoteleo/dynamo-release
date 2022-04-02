import warnings
from copy import deepcopy
from inspect import signature

import numpy as np
import scipy
from anndata import AnnData
from pynndescent.distances import true_angular
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import sparsefuncs

from ..configuration import DynamoAdataKeyManager
from ..docrep import DocstringProcessor
from ..dynamo_logger import LoggerManager, main_info, main_warning
from .utils import fetch_X_data, log1p_

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
        # returns the coordinate tuple for non-zero items
        cur_neighbors = adj[cur_cell, :].nonzero()

        # set itself as the nearest neighbor
        idx[cur_cell, :] = cur_cell
        wgt[cur_cell, :] = 0

        # there could be more or less than n_neighbors because of an approximate search
        cur_n_neighbors = len(cur_neighbors[1])

        if cur_n_neighbors > n_neighbors - 1:
            sorted_indices = np.argsort(adj[cur_cell][:, cur_neighbors[1]].A)[0][: (n_neighbors - 1)]
            idx[cur_cell, 1:] = cur_neighbors[1][sorted_indices]
            wgt[cur_cell, 1:] = adj[cur_cell][0, cur_neighbors[1][sorted_indices]].A
        else:
            idx_ = np.arange(1, (cur_n_neighbors + 1))
            idx[cur_cell, idx_] = cur_neighbors[1]
            wgt[cur_cell, idx_] = adj[cur_cell][:, cur_neighbors[1]].A

    return idx, wgt


def knn_to_adj(knn_indices, knn_weights):
    adj = csr_matrix(
        (
            knn_weights.flatten(),
            (
                np.repeat(knn_indices[:, 0], knn_indices.shape[1]),
                knn_indices.flatten(),
            ),
        )
    )
    adj.eliminate_zeros()

    return adj


def get_conn_dist_graph(knn, distances):
    """Compute connection and distance sparse matrices

    Parameters
    ----------
        knn:
            n_obs x n_neighbors, k nearest neighbor graph
        distances:
            KNN dists

    Returns
    -------
        distance and connectivity matrices
    """
    n_obs, n_neighbors = knn.shape
    distances = csr_matrix(
        (
            distances.flatten(),
            (np.repeat(np.arange(n_obs), n_neighbors), knn.flatten()),
        ),
        shape=(n_obs, n_obs),
    )
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
    max_iter=None,
    alpha=1.0,
    gamma=1.0,
    negative_sample_rate=5,
    init_pos="spectral",
    random_state=0,
    densmap=False,
    dens_lambda=2.0,
    dens_frac=0.3,
    dens_var_shift=0.1,
    output_dens=False,
    return_mapper=True,
    verbose=False,
    **umap_kwargs,
):
    """Compute connectivity graph, matrices for kNN neighbor indices, distance matrix and low dimension embedding with
    UMAP. This code is adapted from umap-learn:
    (https://github.com/lmcinnes/umap/blob/97d33f57459de796774ab2d7fcf73c639835676d/umap/umap_.py)

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
            The effective minimum distance between embedded points. Smaller values will result in a more
            clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger
            values will result on a more even dispersal of points. The value should be set relative to the ``spread``
            value, which determines the scale at which embedded points will be spread out.
        spread: `float` (optional, default 1.0)
            The effective scale of embedded points. In combination with min_dist this determines how clustered/clumped
            the embedded points are.
        max_iter: 'int' or None (optional, default None)
            The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result
            in more accurate embeddings. If None is specified a value will be selected based on the size of the input
            dataset (200 for large datasets, 500 for small). This argument was refactored from n_epochs from UMAP-learn
            to account for recent API changes in UMAP-learn 0.5.2.
        alpha: `float` (optional, default 1.0)
            Initial learning rate for the SGD.
        gamma: `float` (optional, default 1.0)
            Weight to apply to negative samples. Values higher than one will result in greater weight being given to
            negative samples.
        negative_sample_rate: `float` (optional, default 5)
            The number of negative samples to select per positive sample in the optimization process. Increasing this
            value will result in greater repulsive force being applied, greater optimization cost, but slightly more
            accuracy. The number of negative edge/1-simplex samples to use per positive edge/1-simplex sample in
             optimizing the low dimensional embedding.
        init_pos: 'spectral':
            How to initialize the low dimensional embedding. Use a spectral embedding of the fuzzy 1-skeleton
        random_state: `int`, `RandomState` instance or `None`, optional (default: None)
            If int, random_state is the seed used by the random number generator; If RandomState instance, random_state
            is the random number generator; If None, the random number generator is the RandomState instance used by
            `numpy.random`.
        dens_lambda: float (optional, default 2.0)
            Controls the regularization weight of the density correlation term
            in densMAP. Higher values prioritize density preservation over the
            UMAP objective, and vice versa for values closer to zero. Setting this
            parameter to zero is equivalent to running the original UMAP algorithm.
        dens_frac: float (optional, default 0.3)
            Controls the fraction of epochs (between 0 and 1) where the
            density-augmented objective is used in densMAP. The first
            (1 - dens_frac) fraction of epochs optimize the original UMAP objective
            before introducing the density correlation term.
        dens_var_shift: float (optional, default 0.1)
            A small constant added to the variance of local radii in the
            embedding when calculating the density correlation objective to
            prevent numerical instability from dividing by a small number
        output_dens: float (optional, default False)
            Determines whether the local radii of the final embedding (an inverse
            measure of local density) are computed and returned in addition to
            the embedding. If set to True, local radii of the original data
            are also included in the output for comparison; the output is a tuple
            (embedding, original local radii, embedding local radii). This option
            can also be used when densmap=False to calculate the densities for
            UMAP embeddings.
        verbose: `bool` (optional, default False)
                Controls verbosity of logging.

    Returns
    -------
        graph, knn_indices, knn_dists, embedding_
            A tuple of kNN graph (`graph`), indices of nearest neighbors of each cell (knn_indicies), distances of
            nearest neighbors (knn_dists) and finally the low dimensional embedding (embedding_).
    """

    from sklearn.metrics import pairwise_distances
    from sklearn.utils import check_random_state
    from umap.umap_ import (
        find_ab_params,
        fuzzy_simplicial_set,
        nearest_neighbors,
        simplicial_set_embedding,
    )

    # also see github issue at: https://github.com/lmcinnes/umap/issues/798
    default_epochs = 500 if X.shape[0] <= 10000 else 200
    max_iter = default_epochs if max_iter is None else max_iter

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
        if type(graph) == tuple:
            graph = graph[0]

        # extract knn_indices, knn_dist
        g_tmp = deepcopy(graph)
        g_tmp[graph.nonzero()] = dmat[graph.nonzero()]
        knn_indices, knn_dists = adj_to_knn(g_tmp, n_neighbors=n_neighbors)
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
        # _transform_available = True
        # The corresponding nonzero values are stored in similar fashion in self.data.
        _search_graph, _ = get_conn_dist_graph(knn_indices, knn_dists)
        _search_graph = _search_graph.maximum(  # Element-wise maximum between this and another matrix.
            _search_graph.transpose()
        ).tocsr()

    if verbose:
        print("Construct embedding")

    a, b = find_ab_params(spread, min_dist)
    if type(graph) == tuple:
        graph = graph[0]

    dens_lambda = dens_lambda if densmap else 0.0
    dens_frac = dens_frac if densmap else 0.0

    if dens_lambda < 0.0:
        raise ValueError("dens_lambda cannot be negative")
    if dens_frac < 0.0 or dens_frac > 1.0:
        raise ValueError("dens_frac must be between 0.0 and 1.0")
    if dens_var_shift < 0.0:
        raise ValueError("dens_var_shift cannot be negative")

    densmap_kwds = {
        "lambda": dens_lambda,
        "frac": dens_frac,
        "var_shift": dens_var_shift,
        "n_neighbors": n_neighbors,
    }
    embedding_, aux_data = simplicial_set_embedding(
        data=_raw_data,
        graph=graph,
        n_components=n_components,
        initial_alpha=alpha,  # learning_rate
        a=a,
        b=b,
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        n_epochs=max_iter,
        init=init_pos,
        random_state=random_state,
        metric=metric,
        metric_kwds={},
        verbose=verbose,
        densmap=densmap,
        densmap_kwds=densmap_kwds,
        output_dens=output_dens,
    )

    if return_mapper:
        import umap.umap_ as umap

        from .utils import update_dict

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
            n_epochs=max_iter,
            learning_rate=alpha,
            repulsion_strength=gamma,
            negative_sample_rate=negative_sample_rate,
            init=init_pos,
            random_state=random_state,
            verbose=verbose,
            **umap_kwargs,
        ).fit(X)

        return mapper, graph, knn_indices, knn_dists, embedding_
    else:
        return graph, knn_indices, knn_dists, embedding_


def mnn_from_list(knn_graph_list):
    """Apply reduce function to calculate the mutual kNN."""
    import functools

    mnn = (
        functools.reduce(scipy.sparse.csr.csr_matrix.minimum, knn_graph_list)
        if issparse(knn_graph_list[0])
        else functools.reduce(scipy.minimum, knn_graph_list)
    )

    return mnn


def normalize_knn_graph(knn):
    """normalize the knn graph so that each row will be sum up to 1."""
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
    """Function to calculate mutual nearest neighbor graph across specific data layers.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        n_pca_components: 'int' (optional, default `30`)
            Number of PCA components.
        layers: str or list (default: `all`)
            The layer(s) to be normalized. Default is `all`, including RNA (X, raw) or spliced, unspliced, protein, etc.
        use_pca_fit: `bool` (default: `True`)
            Whether to use the precomputed pca model to transform different data layers or calculate pca for each data
            layer separately.
        save_all_to_adata: `bool` (default: `False`)
            Whether to save_fig all calculated data to adata object.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An updated anndata object that are updated with the `mnn` or other relevant data that are calculated during
            mnn calculation.
    """
    if use_pca_fit:
        if "pca_fit" in adata.uns.keys():
            fiter = adata.uns["pca_fit"]
        else:
            raise Exception("use_pca_fit is set to be True, but there is no pca fit results in .uns attribute.")

    layers = DynamoAdataKeyManager.get_available_layer_keys(adata, layers, False, False)
    layers = [
        layer
        for layer in layers
        if layer.startswith("X_") and (not layer.endswith("_matrix") and not layer.endswith("_ambiguous"))
    ]
    knn_graph_list = []
    for layer in layers:
        layer_X = adata.layers[layer]
        layer_X = log1p_(adata, layer_X)
        if use_pca_fit:
            layer_pca = fiter.fit_transform(layer_X)[:, 1:]
        else:
            transformer = TruncatedSVD(n_components=n_pca_components + 1, random_state=0)
            layer_pca = transformer.fit_transform(layer_X)[:, 1:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (
                graph,
                knn_indices,
                knn_dists,
                X_dim,
            ) = umap_conn_indices_dist_embedding(layer_pca, n_neighbors=n_neighbors, return_mapper=False)

        if save_all_to_adata:
            adata.obsm[layer + "_pca"], adata.obsm[layer + "_umap"] = (
                layer_pca,
                X_dim,
            )
            n_neighbors = signature(umap_conn_indices_dist_embedding).parameters["n_neighbors"]

            adata.uns[layer + "_neighbors"] = {
                "params": {"n_neighbors": eval(n_neighbors), "method": "umap"},
                # "connectivities": None,
                # "distances": None,
                "indices": knn_indices,
            }
            (
                adata.obsp[layer + "_connectivities"],
                adata.obsp[layer + "_distances"],
            ) = (graph, knn_dists)

        knn_graph_list.append(graph > 0)

    mnn = mnn_from_list(knn_graph_list)
    adata.uns["mnn"] = normalize_knn_graph(mnn)

    return adata


def _gen_neighbor_keys(result_prefix="") -> tuple:
    """Generate neighbor keys for other functions to store/access info in adata.

    Parameters
    ----------
        result_prefix : str, optional
            generate keys based on this prefix, by default ""

    Returns
    -------
        tuple:
            A tuple consisting of (conn_key, dist_key, neighbor_key)

    """
    if result_prefix:
        result_prefix = result_prefix if result_prefix.endswith("_") else result_prefix + "_"
    if result_prefix is None:
        result_prefix = ""

    conn_key, dist_key, neighbor_key = (
        result_prefix + "connectivities",
        result_prefix + "distances",
        result_prefix + "neighbors",
    )
    return conn_key, dist_key, neighbor_key


def neighbors(
    adata,
    X_data=None,
    genes=None,
    basis="pca",
    layer=None,
    n_pca_components=30,
    n_neighbors=30,
    method=None,
    metric="euclidean",
    metric_kwads=None,
    cores=1,
    seed=19491001,
    result_prefix="",
    **kwargs,
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
        layers: str or None (default: `None`)
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
            Euclidean metric. See the documentation of :class:`DistanceMetric` for a list of available metrics. If
            metric is "precomputed", X is assumed to be a distance matrix and must be square during fit. X may be a
            :term:`sparse graph`, in which case only "nonzero" elements may be considered neighbors.
        metric_params : dict, default=None
            Additional keyword arguments for the metric function.
        cores: `int` (default: 1)
            The number of parallel jobs to run for neighbors search. ``None`` means 1 unless in a
            :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        seed: `int` (default `19491001`)
            Random seed to ensure the reproducibility of each run.
        result_prefix: `str` (default: `''`)
            The key that will be used as the prefix of the connectivity, distance and neighbor keys in the returning
            adata.
        kwargs:
            Additional arguments that will be passed to each nearest neighbor search algorithm.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An updated anndata object that are updated with the `indices`, `connectivity`, `distance` to the .obsp, as
            well as a new `neighbors` key in .uns.
    """
    logger = LoggerManager.gen_logger("neighbors")
    logger.info("Start computing neighbor graph...")
    logger.log_time()

    if X_data is None:
        logger.info("X_data is None, fetching or recomputing...", indent_level=2)
        if basis == "pca" and "X_pca" not in adata.obsm_keys():
            logger.info("PCA as basis not X_pca not found, doing PCAs", indent_level=2)
            from ..preprocessing.utils import pca_monocle

            CM = adata.X if genes is None else adata[:, genes].X
            cm_genesums = CM.sum(axis=0)
            valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
            valid_ind = np.array(valid_ind).flatten()
            CM = CM[:, valid_ind]
            adata, _, _ = pca_monocle(adata, CM, pca_key="X_pca", n_pca_components=n_pca_components, return_all=True)

            X_data = adata.obsm["X_pca"]
        else:
            logger.info("fetching X data from layer:%s, basis:%s" % (str(layer), str(basis)))
            genes, X_data = fetch_X_data(adata, genes, layer, basis)

    if method is None:
        logger.info("method arg is None, choosing methods automatically...")
        if X_data.shape[0] > 200000 and X_data.shape[1] > 2:

            from pynndescent import NNDescent

            method = "pynn"
        elif X_data.shape[1] > 10:
            method = "ball_tree"
        else:
            method = "kd_tree"
        logger.info("method %s selected" % (method), indent_level=2)

    # may distinguish between umap and pynndescent -- treat them equal for now
    if method.lower() in ["pynn", "umap"]:
        index = NNDescent(
            X_data,
            metric=metric,
            n_neighbors=n_neighbors,
            n_jobs=cores,
            random_state=seed,
            **kwargs,
        )
        knn, distances = index.query(X_data, k=n_neighbors)
    elif method in ["ball_tree", "kd_tree"]:
        from sklearn.neighbors import NearestNeighbors

        # print("***debug X_data:", X_data)
        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            metric_params=metric_kwads,
            algorithm=method,
            n_jobs=cores,
            **kwargs,
        ).fit(X_data)
        distances, knn = nbrs.kneighbors(X_data)
    else:
        raise ImportError(f"nearest neighbor search method {method} is not supported")

    conn_key, dist_key, neighbor_key = _gen_neighbor_keys(result_prefix)
    logger.info_insert_adata(conn_key, adata_attr="obsp")
    logger.info_insert_adata(dist_key, adata_attr="obsp")
    adata.obsp[dist_key], adata.obsp[conn_key] = get_conn_dist_graph(knn, distances)

    logger.info_insert_adata(neighbor_key, adata_attr="uns")
    logger.info_insert_adata(neighbor_key + ".indices", adata_attr="uns")
    logger.info_insert_adata(neighbor_key + ".params", adata_attr="uns")
    adata.uns[neighbor_key] = {}
    adata.uns[neighbor_key]["indices"] = knn
    adata.uns[neighbor_key]["params"] = {
        "n_neighbors": n_neighbors,
        "method": method,
        "metric": metric,
        "n_pcs": n_pca_components,
    }

    return adata


def check_neighbors_completeness(
    adata: AnnData,
    conn_key="connectivities",
    dist_key="distances",
    result_prefix="",
    check_nonzero_row=True,
    check_nonzero_col=False,
) -> bool:
    """Check if neighbor graph in adata is valid.

    Parameters
    ----------
        adata : AnnData
        conn_key : str, optional
            connectivity key, by default "connectivities"
        dist_key : str, optional
            distance key, by default "distances"
        result_prefix : str, optional
            The result prefix in adata.uns for neighbor graph related data, by default ""
        check_nonzero_row:
            Whether to check if row sums of neighbor graph distance or connectivity matrix are nonzero. Row sums correspond to out-degrees by convention.
        check_nonzero_col:
            Whether to check if column sums of neighbor graph distance or connectivity matrix are nonzero. Column sums correspond to in-degrees by convention.

    Returns
    -------
        bool
            whether the neighbor graph is valid or not. (If valid, return True)
    """
    is_valid = True
    conn_key, dist_key, neighbor_key = _gen_neighbor_keys(result_prefix)
    keys = [conn_key, dist_key, neighbor_key]

    # Old anndata version version
    # conn_mat = adata.uns[neighbor_key]["connectivities"]
    # dist_mat = adata.uns[neighbor_key]["distances"]
    if (conn_key not in adata.obsp) or (dist_key not in adata.obsp) or ("indices" not in adata.uns[neighbor_key]):
        main_info(
            "incomplete neighbor graph info detected: %s and %s do not exist in adata.obsp, indices not in adata.uns.%s."
            % (conn_key, dist_key, neighbor_key)
        )
        return False
    # New anndata stores connectivities and distances in obsp
    conn_mat = adata.obsp[conn_key]
    dist_mat = adata.obsp[dist_key]
    n_obs = adata.n_obs

    # check if connection matrix and distance matrix shapes are compatible with adata shape
    is_valid = is_valid and tuple(conn_mat.shape) == tuple([n_obs, n_obs])
    is_valid = is_valid and tuple(dist_mat.shape) == tuple([n_obs, n_obs])
    if not is_valid:
        main_info("Connection matrix or dist matrix has some invalid shape.")
        return False

    if neighbor_key not in adata.uns:
        main_info("%s not in adata.uns" % (neighbor_key))
        return False

    # check if indices in nearest neighbor matrix are valid
    neighbor_mat = adata.uns[neighbor_key]["indices"]
    is_indices_valid = np.all(neighbor_mat < n_obs)
    if not is_indices_valid:
        main_warning(
            "Some indices in %s are larger than the number of observations and thus not valid." % (neighbor_key)
        )
        return False
    is_valid = is_valid and is_indices_valid

    def _check_nonzero_sum(mat, axis):
        sums = np.sum(mat, axis=axis)
        return np.all(sums > 0)

    if check_nonzero_row:
        is_row_valid = _check_nonzero_sum(dist_mat, 1) and _check_nonzero_sum(conn_mat, 1)
        if not is_row_valid:
            main_warning("Some row sums(out degree) in adata's neighbor graph are zero.")
        is_valid = is_valid and is_row_valid
    if check_nonzero_col:
        is_col_valid = _check_nonzero_sum(dist_mat, 0) and _check_nonzero_sum(conn_mat, 0)
        if not is_col_valid:
            main_warning("Some column sums(in degree) in adata's neighbor graph are zero.")
        is_valid = is_valid and is_col_valid

    return is_valid


def check_and_recompute_neighbors(adata: AnnData, result_prefix: str = ""):
    """Check if adata's neighbor graph is valid and recompute neighbor graph if necessary.

    Parameters
    ----------
        adata:
        result_prefix : str, optional
            The result prefix in adata.uns for neighbor graph related data, by default ""
    """
    if result_prefix is None:
        result_prefix = ""
    conn_key, dist_key, neighbor_key = _gen_neighbor_keys(result_prefix)

    if not check_neighbors_completeness(adata, conn_key=conn_key, dist_key=dist_key, result_prefix=result_prefix):
        main_info("Neighbor graph is broken, recomputing....")
        neighbors(adata, result_prefix=result_prefix)
