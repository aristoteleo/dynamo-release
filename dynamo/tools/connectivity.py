from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import warnings
from copy import deepcopy
from inspect import signature

import numpy as np
import scipy
from anndata import AnnData
from pynndescent.distances import true_angular
from scipy.sparse import coo_matrix, csr_matrix, issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import sparsefuncs
from umap import UMAP

from ..configuration import DynamoAdataKeyManager
from ..docrep import DocstringProcessor
from ..dynamo_logger import LoggerManager, main_info, main_warning
from .utils import fetch_X_data, log1p_

docstrings = DocstringProcessor()


def adj_to_knn(adj: np.ndarray, n_neighbors: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the adjacency matrix of a nearest neighbor graph to the indices and weights for a knn graph.

    Args:
        adj: adjacency matrix (n x n) of the nearest neighbor graph.
        n_neighbors: the number of nearest neighbors of the kNN graph. Defaults to 15.

    Returns:
        A tuple (idx, wgt) where idx is the matrix (n x n_neighbors) storing the indices for each node's n_neighbors
        nearest neighbors and wgt is the matrix (n x n_neighbors) storing the wights on the edges for each node's
        n_neighbors nearest neighbors.
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


def knn_to_adj(knn_indices: np.ndarray, knn_weights: np.ndarray) -> csr_matrix:
    """Convert a knn graph's indices and weights to an adjacency matrix of the corresponding nearest neighbor graph.

    Args:
        knn_indices: the matrix (n x n_neighbors) storing the indices for each node's n_neighbors nearest neighbors in
            the knn graph.
        knn_weights: the matrix (n x n_neighbors) storing the wights on the edges for each node's n_neighbors nearest
            neighbors in the knn graph.

    Returns:
        The converted adjacency matrix (n x n) of the corresponding nearest neighbor graph.
    """

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


def get_conn_dist_graph(knn: np.ndarray, distances: np.ndarray) -> Tuple[csr_matrix, csr_matrix]:
    """Compute connection and distance sparse matrix.

    Args:
        knn: a matrix (n x n_neighbors) storing the indices for each node's n_neighbors nearest neighbors in knn graph.
        distances: the distances to the n_neighbors the closest points in knn graph.

    Returns:
        A tuple (distances, connectivities), where distance is the distance sparse matrix and connectivities is the
        connectivity sparse matrix.
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
    X: np.ndarray,
    n_neighbors: int = 30,
    n_components: int = 2,
    metric: Union[str, Callable] = "euclidean",
    min_dist: float = 0.1,
    spread: float = 1.0,
    max_iter: Optional[int] = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: float = 5,
    init_pos: Union[Literal["spectral", "random"], np.ndarray] = "spectral",
    random_state: Union[int, np.random.RandomState, None] = 0,
    densmap: bool = False,
    dens_lambda: float = 2.0,
    dens_frac: float = 0.3,
    dens_var_shift: float = 0.1,
    output_dens: bool = False,
    return_mapper: bool = True,
    verbose: bool = False,
    **umap_kwargs,
) -> Union[
    Tuple[UMAP, coo_matrix, np.ndarray, np.ndarray, np.ndarray],
    Tuple[coo_matrix, np.ndarray, np.ndarray, np.ndarray],
]:
    """Compute connectivity graph, matrices for kNN neighbor indices, distance matrix and low dimension embedding with
    UMAP.

    This code is adapted from umap-learn:
    (https://github.com/lmcinnes/umap/blob/97d33f57459de796774ab2d7fcf73c639835676d/umap/umap_.py)


    Args:
        X: the expression matrix (n_cell x n_genes).
        n_neighbors: the number of nearest neighbors to compute for each sample in `X`. Defaults to 30.
        n_components: the dimension of the space to embed into. Defaults to 2.
        metric: the metric to use for the computation. Defaults to "euclidean".
        min_dist: the effective minimum distance between embedded points. Smaller values will result in a more
            clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger
            values will result on a more even dispersal of points. The value should be set relative to the `spread`
            value, which determines the scale at which embedded points will be spread out. Defaults to 0.1.
        spread: the effective scale of embedded points. In combination with min_dist this determines how
            clustered/clumped the embedded points are. Defaults to 1.0.
        max_iter: the number of training epochs to be used in optimizing the low dimensional embedding. Larger values
            result in more accurate embeddings. If None is specified a value will be selected based on the size of the
            input dataset (200 for large datasets, 500 for small). This argument was refactored from n_epochs from
            UMAP-learn to account for recent API changes in UMAP-learn 0.5.2. Defaults to None.
        alpha: initial learning rate for the SGD. Defaults to 1.0.
        gamma: weight to apply to negative samples. Values higher than one will result in greater weight being given to
            negative samples. Defaults to 1.0.
        negative_sample_rate: the number of negative samples to select per positive sample in the optimization process.
            Increasing this value will result in greater repulsive force being applied, greater optimization cost, but
            slightly more accuracy. The number of negative edge/1-simplex samples to use per positive edge/1-simplex
            sample in optimizing the low dimensional embedding. Defaults to 5.
        init_pos: the method to initialize the low dimensional embedding. Where:
            "spectral": use a spectral embedding of the fuzzy 1-skeleton.
            "random": assign initial embedding positions at random.
            np.ndarray: the array to define the initial position.
            Defaults to "spectral".
        random_state: the method to generate random numbers. If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random number generator; If None, the random number
            generator is the RandomState instance used by `numpy.random`. Defaults to 0.
        densmap: whether to use the density-augmented objective function to optimize the embedding according to the
            densMAP algorithm. Defaults to False.
        dens_lambda: controls the regularization weight of the density correlation term in densMAP. Higher values
            prioritize density preservation over the UMAP objective, and vice versa for values closer to zero. Setting
            this parameter to zero is equivalent to running the original UMAP algorithm. Defaults to 2.0.
        dens_frac: controls the fraction of epochs (between 0 and 1) where the density-augmented objective is used in
            densMAP. The first (1 - dens_frac) fraction of epochs optimize the original UMAP objective before
            introducing the density correlation term. Defaults to 0.3.
        dens_var_shift: a small constant added to the variance of local radii in the embedding when calculating the
            density correlation objective to prevent numerical instability from dividing by a small number Defaults to
            0.1.
        output_dens: whether the local radii of the final embedding (an inverse measure of local density) are computed
            and returned in addition to the embedding. If set to True, local radii of the original data are also
            included in the output for comparison; the output is a tuple (embedding, original local radii, embedding
            local radii). This option can also be used when densmap=False to calculate the densities for UMAP
            embeddings. Defaults to False.
        return_mapper: whether to return the data mapped onto the UMAP space. Defaults to True.
        verbose: whether to log verbosely. Defaults to False.

    Raises:
        ValueError: `dense_lambda` is negative and thus invalid.
        ValueError: `dense_frac` out of range (0.0 ~ 1.0)
        ValueError: `dense_var_shift` is negative and thus invalid.

    Returns:
        A tuple ([mapper,] graph, knn_indices, knn_dists, embedding_). `mapper` is the data mapped onto umap space and
        will be returned only if `return_mapper` is true. graph is the sparse matrix representing the graph,
        `knn_indices` is the matrix storing indices of nearest neighbors of each cell, `knn_dists` is the distances to
        the n_neighbors closest points in knn graph, and `embedding_` is the low dimensional embedding.
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


CsrOrNdarray = TypeVar("CsrOrNdarray", csr_matrix, np.ndarray)


def mnn_from_list(knn_graph_list: List[CsrOrNdarray]) -> CsrOrNdarray:
    """Apply reduce function to calculate the mutual kNN.

    Args:
        knn_graph_list: a list of ndarray or csr_matrix representing a series of knn graphs.

    Returns:
        The calculated mutual knn, in same type as the input (ndarray of csr_matrix).
    """

    import functools

    mnn = (
        functools.reduce(scipy.sparse.csr.csr_matrix.minimum, knn_graph_list)
        if issparse(knn_graph_list[0])
        else functools.reduce(scipy.minimum, knn_graph_list)
    )

    return mnn


def normalize_knn_graph(knn: csr_matrix) -> csr_matrix:
    """Normalize the knn graph so that each row will be sum up to 1.

    Args:
        knn: the sparse matrix containing the indices of nearest neighbors of each cell.

    Returns:
        The normalized matrix.
    """

    """normalize the knn graph so that each row will be sum up to 1."""
    knn.setdiag(1)
    knn = knn.astype("float32")
    sparsefuncs.inplace_row_scale(knn, 1 / knn.sum(axis=1).A1)

    return knn


def mnn(
    adata: AnnData,
    n_pca_components: int = 30,
    n_neighbors: int = 250,
    layers: Union[str, List[str]] = "all",
    use_pca_fit: bool = True,
    save_all_to_adata: bool = False,
) -> AnnData:
    """Calculate mutual nearest neighbor graph across specific data layers.

    Args:
        adata: an AnnData object.
        n_pca_components: the number of PCA components. Defaults to 30.
        n_neighbors: the number of nearest neighbors to compute for each sample. Defaults to 250.
        layers: the layer(s) to be normalized. When set to `'all'`, it will include RNA (X, raw) or spliced, unspliced,
            protein, etc. Defaults to "all".
        use_pca_fit: whether to use the precomputed pca model to transform different data layers or calculate pca for
            each data layer separately. Defaults to True.
        save_all_to_adata: whether to save_fig all calculated data to adata object. Defaults to False.

    Raises:
        Exception: no PCA fit result in .uns.

    Returns:
        An updated anndata object that are updated with the `mnn` or other relevant data that are calculated during mnn
        calculation.
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


def _gen_neighbor_keys(result_prefix: str = "") -> Tuple[str, str, str]:
    """Generate neighbor keys for other functions to store/access info in adata.

    Args:
        result_prefix: the prefix for keys. Defaults to "".

    Returns:
        A tuple (conn_key, dist_key, neighbor_key) for key of connectivity matrix, distance matrix, neighbor matrix,
        respectively.
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
    adata: AnnData,
    X_data: np.ndarray = None,
    genes: Optional[List[str]] = None,
    basis: str = "pca",
    layer: Optional[str] = None,
    n_pca_components: int = 30,
    n_neighbors: int = 30,
    method: Optional[str] = None,
    metric: Union[str, Callable] = "euclidean",
    metric_kwads: Dict[str, Any] = None,
    cores: int = 1,
    seed: int = 19491001,
    result_prefix: str = "",
    **kwargs,
) -> AnnData:
    """Search nearest neighbors of the adata object.

    Args:
        adata: an AnnData object.
        X_data: the user supplied data that will be used for nearest neighbor search directly. Defaults to None.
        genes: the list of genes that will be used to subset the data for nearest neighbor search. If `None`, all genes
            will be used. Defaults to None.
        basis: The space that will be used for nearest neighbor search. Valid names includes, for example, `pca`,
            `umap`, `velocity_pca` or `X` (that is, you can use velocity for clustering), etc. Defaults to "pca".
        layer: the layer to be used for nearest neighbor search. Defaults to None.
        n_pca_components: number of PCA components. Applicable only if you will use pca `basis` for nearest neighbor
            search. Defaults to 30.
        n_neighbors: number of nearest neighbors. Defaults to 30.
        method: the methoed used for nearest neighbor search. If `umap` or `pynn`, it relies on `pynndescent` package's
            NNDescent for fast nearest neighbor search. Defaults to None.
        metric: the distance metric to use for the tree. The default metric is euclidean, and with p=2 is equivalent to
            the standard Euclidean metric. See the documentation of `DistanceMetric` for a list of available metrics. If
            metric is "precomputed", X is assumed to be a distance matrix and must be square during fit. X may be a
            `sparse graph`, in which case only "nonzero" elements may be considered neighbors. Defaults to "euclidean".
        metric_kwads: additional keyword arguments for the metric function. Defaults to None.
        cores: the number of parallel jobs to run for neighbors search. `None` means 1 unless in a
            `joblib.parallel_backend` context. `-1` means using all processors. Defaults to 1.
        seed: random seed to ensure the reproducibility of each run. Defaults to 19491001.
        result_prefix: the key that will be used as the prefix of the connectivity, distance and neighbor keys in the
            returning adata. Defaults to "".
        kwargs: additional arguments that will be passed to each nearest neighbor search algorithm.

    Raises:
        ImportError: `method` is invalid.

    Returns:
        An updated anndata object that are updated with the `indices`, `connectivity`, `distance` to the .obsp, as well
        as a new `neighbors` key in .uns.
    """

    logger = LoggerManager.gen_logger("neighbors")
    logger.info("Start computing neighbor graph...")
    logger.log_time()

    if X_data is None:
        logger.info("X_data is None, fetching or recomputing...", indent_level=2)
        if basis == "pca" and "X_pca" not in adata.obsm_keys():
            logger.info("PCA as basis not X_pca not found, doing PCAs", indent_level=2)
            from ..preprocessing.pca import pca

            CM = adata.X if genes is None else adata[:, genes].X
            cm_genesums = CM.sum(axis=0)
            valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
            valid_ind = np.array(valid_ind).flatten()
            CM = CM[:, valid_ind]
            adata, _, _ = pca(adata, CM, pca_key="X_pca", n_pca_components=n_pca_components, return_all=True)

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
    conn_key: str = "connectivities",
    dist_key: str = "distances",
    result_prefix: str = "",
    check_nonzero_row: bool = True,
    check_nonzero_col: bool = False,
) -> bool:
    """Check if neighbor graph in the AnnData object is valid.

    Args:
        adata: an AnnData object.
        conn_key: the key for connectivity matrix. Defaults to "connectivities".
        dist_key: the key for distance matrix. Defaults to "distances".
        result_prefix: the result prefix in adata.uns for neighbor graph related data. Defaults to "".
        check_nonzero_row: whether to check if row sums of neighbor graph distance or connectivity matrix are nonzero.
            Row sums correspond to out-degrees by convention. Defaults to True.
        check_nonzero_col: whether to check if column sums of neighbor graph distance or connectivity matrix are
            nonzero. Column sums correspond to in-degrees by convention. Defaults to False.

    Returns:
        Whether the neighbor graph is valid or not.
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


def check_and_recompute_neighbors(adata: AnnData, result_prefix: str = "") -> None:
    """Check if adata's neighbor graph is valid and recompute neighbor graph if necessary.

    Args:
        adata: an AnnData object.
        result_prefix: the result prefix in adata.uns for neighbor graph related data. Defaults to "".
    """

    if result_prefix is None:
        result_prefix = ""
    conn_key, dist_key, neighbor_key = _gen_neighbor_keys(result_prefix)

    if not check_neighbors_completeness(adata, conn_key=conn_key, dist_key=dist_key, result_prefix=result_prefix):
        main_info("Neighbor graph is broken, recomputing....")
        neighbors(adata, result_prefix=result_prefix)
