from hdbscan import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np
import anndata
import pandas as pd
from ..dynamo_logger import main_info
from .connectivity import neighbors
from .utils_reduceDimension import prepare_dim_reduction, run_reduce_dim
from .utils import update_dict
from ..utils import LoggerManager, copy_adata


def hdbscan(
    adata,
    X_data=None,
    genes=None,
    layer=None,
    basis="pca",
    dims=None,
    n_pca_components=30,
    n_components=2,
    result_key=None,
    copy=False,
    **hdbscan_kwargs
):
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
    copy:
        Whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments.
    hdbscan_kwargs: `dict`
        Additional parameters that will be passed to hdbscan function.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An updated AnnData object with the clustering updated. `hdbscan` and `hdbscan_prob` are two newly added
            columns from .obs, corresponding to either the Cluster results or the probability of each cell belong to a
            cluster. `hdbscan` key in .uns corresponds to a dictionary that includes additional results returned from
            hdbscan run.
    """

    logger = LoggerManager.gen_logger("dynamo-hdbscan")
    logger.log_time()
    adata = copy_adata(adata) if copy else adata

    if X_data is None:
        _, n_components, has_basis, basis = prepare_dim_reduction(
            adata,
            genes=genes,
            layer=layer,
            basis=basis,
            dims=dims,
            n_pca_components=n_pca_components,
            n_components=n_components,
        )

    if has_basis:
        X_data = adata.obsm[basis]
    else:
        reduction_method = basis.split("_")[-1]
        embedding_key = "X_" + reduction_method if layer is None else layer + "_" + reduction_method
        neighbor_key = "neighbors" if layer is None else layer + "_neighbors"

        adata = run_reduce_dim(
            adata,
            X_data,
            n_components,
            n_pca_components,
            reduction_method,
            embedding_key=embedding_key,
            n_neighbors=30,
            neighbor_key=neighbor_key,
            cores=1,
        )

    X_data = X_data if dims is None else X_data[:, dims]

    if hdbscan_kwargs is not None and "metric" in hdbscan_kwargs.keys():
        if hdbscan_kwargs["metric"] == "cosine":
            from sklearn.preprocessing import normalize

            X_data = normalize(X_data, norm="l2")

    h_kwargs = {
        "min_cluster_size": 5,
        "min_samples": None,
        "metric": "euclidean",
        "p": None,
        "alpha": 1.0,
        "cluster_selection_epsilon": 0.0,
        "algorithm": "best",
        "leaf_size": 40,
        "approx_min_span_tree": True,
        "gen_min_span_tree": False,
        "core_dist_n_jobs": 1,
        "cluster_selection_method": "eom",
        "allow_single_cluster": False,
        "prediction_data": False,
        "match_reference_implementation": False,
    }

    h_kwargs = update_dict(h_kwargs, hdbscan_kwargs)
    cluster = HDBSCAN(**h_kwargs)
    cluster.fit(X_data)

    if result_key is None:
        key = "hdbscan"
    adata.obs[key] = cluster.labels_.astype("str")
    adata.obs[key + "_prob"] = cluster.probabilities_
    adata.uns[key] = {
        "hdbscan": cluster.labels_.astype("str"),
        "probabilities_": cluster.probabilities_,
        "cluster_persistence_": cluster.cluster_persistence_,
        "outlier_scores_": cluster.outlier_scores_,
        "exemplars_": cluster.exemplars_,
    }

    logger.finish_progress(progress_name="hdbscan density-based-clustering")

    if copy:
        return adata
    return None


def cluster_field(
    adata, basis="pca", embedding_basis=None, normalize=True, method="leiden", cores=1, copy=False, **kwargs
):
    """Cluster cells based on vector field features.

    We would like to see whether the vector field can be used to better define cell state/types. This can be accessed
    via characterizing critical points (attractor/saddle/repressor, etc.) and characteristic curves (nullcline,
    separatrix). However, the calculation of those is not easy, for example, a strict definition of an attractor is
    states where velocity is 0 and the eigenvalue of the jacobian matrix at that point is all negative. Under this
    strict definition, we may sometimes find the attractors are very far away from our sampled cell states which makes
    them less meaningful although this can be largely avoided when we decide to remove the density correction during the
    velocity projection. This is not unexpected as the vector field we learned is defined via a set of basis functions
    based on gaussian kernels and thus it is hard to satisfy that strict definition.

    Fortunately, we can handle this better with the help of a different set of ideas. Instead of using critical points
    by the classical dynamic system methods, we can use some machine learning approaches that are based on extracting
    geometric features of streamline to "cluster vector field space" for define cell states/type. This requires
    calculating, potential (ordered pseudotime), speed, curliness, divergence, acceleration, curvature, etc. Thanks to
    the fact that we can analytically calculate Jacobian matrix matrix, those quantities of the vector field function
    can be conveniently and efficiently calculated.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`.
        adata object that includes both newly synthesized and total gene expression of cells. Alternatively,
        the object should include both unspliced and spliced gene expression of cells.
    basis: `str` or None (default: `None`)
        The space that will be used for calculating vector field features. Valid names includes, for example, `pca`,
        `umap`, etc.
    embedding_basis: `str` or None (default: `None`)
        The embedding basis that will be combined with the vector field feature space for clustering.
    normalize: `bool` (default: `True`)
        Whether to mean center and scale the feature across all cells so that the mean
    method: `str` (default: `leiden`)
        The method that will be used for clustering, one of `{'kmeans'', 'hdbscan', 'louvain', 'leiden'}`. If `louvain`
        or `leiden` used, you need to have `cdlib` installed.
    cores: `int` (default: 1)
        The number of parallel jobs to run for neighbors search. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    copy:
        Whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments.
    kwargs:
        Any additional arguments that will be passed to either kmeans, hdbscan, louvain or leiden clustering algorithms.

    Returns
    -------

    """

    logger = LoggerManager.gen_logger("dynamo-cluster_field")
    logger.log_time()
    adata = copy_adata(adata) if copy else adata

    if method in ["louvain", "leiden"]:
        try:
            from cdlib import algorithms

            "leiden" in dir(algorithms)

        except ImportError:
            raise ImportError(
                "You need to install the excellent package `cdlib` if you want to use louvain or leiden "
                "for clustering."
            )

    feature_key = [
        "speed_" + basis,
        basis + "_ddhodge_potential",
        "divergence_" + basis,
        "acceleration_" + basis,
        "curvature_" + basis,
    ]

    if feature_key[0] not in adata.obs.keys():
        from ..vectorfield import speed

        speed(adata, basis=basis)
    if feature_key[1] not in adata.obs.keys():
        from ..ext import ddhodge

        ddhodge(adata, basis=basis)
    if feature_key[2] not in adata.obs.keys():
        from ..vectorfield import divergence

        divergence(adata, basis=basis)
    if feature_key[3] not in adata.obs.keys():
        from ..vectorfield import acceleration

        acceleration(adata, basis=basis)
    if feature_key[4] not in adata.obs.keys():
        from ..vectorfield import curvature

        curvature(adata, basis=basis)

    feature_data = adata.obs.loc[:, feature_key].values
    if embedding_basis is None:
        embedding_basis = basis
    X = np.hstack((feature_data, adata.obsm["X_" + embedding_basis]))

    if normalize:
        # X = (X - X.min(0)) / X.ptp(0)
        X = (X - X.mean(0)) / X.std(0)

    if method in ["hdbscan", "kmeans"]:
        if method == "hdbscan":
            key = "field_hdbscan"
            hdbscan(adata, X_data=X, result_key=key, **kwargs)
        elif method == "kmeans":
            from sklearn.cluster import KMeans

            key = "field_kmeans"

            kmeans = KMeans(random_state=0, **kwargs).fit(X)
            adata.obs[key] = kmeans.labels_.astype("str")

        # clusters need to be categorical variables
        adata.obs.obs[key] = adata.obs.obs[key].astype("category")

    elif method in ["louvain", "leiden"]:
        if X.shape[0] > 200000 and X.shape[1] > 2:
            from pynndescent import NNDescent

            nbrs = NNDescent(
                X,
                metric="euclidean",
                n_neighbors=31,
                n_jobs=cores,
                random_state=19491001,
            )
            nbrs_idx, dist = nbrs.query(X, k=31)
        else:
            nbrs = NearestNeighbors(n_neighbors=31, n_jobs=cores).fit(X)
            dist, nbrs_idx = nbrs.kneighbors(X)

        row = np.repeat(nbrs_idx[:, 0], 30)
        col = nbrs_idx[:, 1:].flatten()
        graph = csr_matrix(
            (np.repeat(1, len(col)), (row, col)),
            shape=(adata.n_obs, adata.n_obs),
        )
        adata.obsp["vf_feature_knn"] = graph

        if method == "leiden":
            leiden(
                adata,
                adj_matrix_key="vf_feature_knn",
                result_key="field_leiden",
            )
        elif method == "louvain":
            louvain(
                adata,
                adj_matrix_key="vf_feature_knn",
                result_key="field_louvain",
            )
        elif method == "infomap":
            infomap(
                adata,
                adj_matrix_key="vf_feature_knn",
                result_key="field_infomap",
            )

    logger.finish_progress(progress_name="clustering_field")

    if copy:
        return adata
    return None


def leiden(
    adata,
    use_weight=True,
    weight=None,
    initial_membership=None,
    adj_matrix=None,
    adj_matrix_key=None,
    result_key=None,
    layer=None,
    obsm_key=None,
    selected_cluster_subset: list = None,
    selected_cell_subset=None,
    directed=False,
    copy=False,
    **kwargs
) -> anndata.AnnData:
    kwargs.update(
        {
            "weight": weight,
            "initial_membership": initial_membership,
        }
    )

    return cluster_community(
        adata,
        method="leiden",
        use_weight=use_weight,
        result_key=result_key,
        adj_matrix=adj_matrix,
        adj_matrix_key=adj_matrix_key,
        layer=layer,
        obsm_key=obsm_key,
        cluster_and_subsets=selected_cluster_subset,
        cell_subsets=selected_cell_subset,
        directed=directed,
        copy=copy,
        **kwargs
    )


def louvain(
    adata,
    resolution=1.0,
    use_weight=True,
    adj_matrix=None,
    adj_matrix_key=None,
    randomize=False,
    result_key=None,
    layer=None,
    obsm_key=None,
    selected_cluster_subset: list = None,
    selected_cell_subset=None,
    directed=False,
    copy=False,
    **kwargs
) -> anndata.AnnData:
    """[summary]

    Parameters
    ----------
    adata : [type]
        [description]
    resolution : float, optional
        [description], by default 1.0
    adj_matrix : [type], optional
        [description], by default None
    use_weight : bool, optional
        [description], by default True
    randomize : bool, optional
        randomize – boolean, from CDlib: "randomize the node evaluation order and the community evaluation order to get
        different partitions at each call, by default False"
    result_key : [type], optional
        [description], by default None

    Returns
    -------
    AnnData
        [description]
    """
    if directed:
        raise ValueError("CDlib does not support directed graph for Louvain community detection for now.")
    kwargs.update(
        {
            "resolution": resolution,
            "randomize": randomize,
        }
    )

    return cluster_community(
        adata,
        method="louvain",
        use_weight=use_weight,
        adj_matrix=adj_matrix,
        adj_matrix_key=adj_matrix_key,
        result_key=result_key,
        layer=layer,
        obsm_key=obsm_key,
        cluster_and_subsets=selected_cluster_subset,
        cell_subsets=selected_cell_subset,
        directed=directed,
        copy=copy,
        **kwargs
    )


def infomap(
    adata,
    use_weight=True,
    adj_matrix=None,
    adj_matrix_key=None,
    result_key=None,
    layer=None,
    obsm_key=None,
    selected_cluster_subset: list = None,
    selected_cell_subset=None,
    directed=False,
    copy=False,
    **kwargs
) -> anndata.AnnData:
    kwargs.update({})

    return cluster_community(
        adata,
        method="infomap",
        use_weight=use_weight,
        result_key=result_key,
        adj_matrix=adj_matrix,
        adj_matrix_key=adj_matrix_key,
        layer=layer,
        obsm_key=obsm_key,
        cluster_and_subsets=selected_cluster_subset,
        cell_subsets=selected_cell_subset,
        directed=directed,
        copy=copy,
        **kwargs
    )


def cluster_community(
    adata,
    method="leiden",
    result_key=None,
    adj_matrix=None,
    adj_matrix_key=None,
    use_weight=True,
    no_community_label=-1,
    layer=None,
    conn_type="connectivities",
    obsm_key=None,
    cell_subsets=None,
    cluster_and_subsets: list = None,
    directed=True,
    copy=False,
    **kwargs
):
    """Detect communities and insert data into adata.


    Parameters
    ----------
    adata : [type]
        [description]
    method : str, optional
        [description], by default "leiden"
    result_key : [type], optional
        [description], by default None
    adj_matrix : [type], optional
        [description], by default None
    adj_matrix_key : str, optional
        [description], by default "connectivities"
    use_weight : bool, optional
        [description], by default False
    no_community_label : int, optional
        [description], by default -1
    layer: str or None:
        [description], by default None
    conn_type: str:
        [description], by default "connectivities"
    obsm_key: str or None:
        [description], by default None
    cell_subsets : [type], optional
        [description], by default None
    cluster_and_subsets : list, optional
        [description], by default None
    directed : bool, optional
        [description], by default False
    copy : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """

    adata = copy_adata(adata) if copy else adata
    if (layer is not None) and (adj_matrix_key is not None):
        raise ValueError("Please supply one of adj_matrix_key and layer")
    if adj_matrix_key is None:
        if layer is None:
            if obsm_key is None:
                adj_matrix_key = "connectivities"
            else:
                adj_matrix_key = obsm_key + "_" + "connectivities"
        else:
            adj_matrix_key = layer + "_" + conn_type

    # try generating required adj_matrix according to
    # user inputs through "neighbors" interface
    if adj_matrix is None:
        if not (adj_matrix_key in adata.obsp):
            if layer is None:
                if obsm_key is None:
                    neighbors(adata)
                else:
                    X_data = adata.obsm[obsm_key]
                    neighbors(adata, X_data=X_data, result_prefix=obsm_key)
            else:
                X_data = adata[:, adata.var.use_for_pca].layers[layer]
                neighbors(adata, X_data=X_data, result_prefix=layer)

        if not (adj_matrix_key in adata.obsp):
            raise ValueError("%s not exist in adata.obsp" % adj_matrix_key)

        graph_sparse_matrix = adata.obsp[adj_matrix_key]
    else:
        graph_sparse_matrix = adj_matrix

    if result_key is None:
        if all((cell_subsets is None, cluster_and_subsets is None)):
            result_key = "%s" % (method) if layer is None else layer + "_" + method
        else:
            result_key = "subset_" + method if layer is None else layer + "_subset_" + method

    valid_indices = None
    if cell_subsets is not None:
        if type(cell_subsets[0]) == str:
            valid_indices = [adata.obs_names.get_loc(i) for i in cell_subsets]
        else:
            valid_indices = cell_subsets

        graph_sparse_matrix = graph_sparse_matrix[valid_indices, :][:, valid_indices]

    if cluster_and_subsets is not None:
        cluster_col, allowed_clusters = (
            cluster_and_subsets[0],
            cluster_and_subsets[1],
        )
        valid_indices_bools = np.isin(adata.obs[cluster_col], allowed_clusters)
        valid_indices = np.argwhere(valid_indices_bools).flatten()
        graph_sparse_matrix = graph_sparse_matrix[valid_indices, :][:, valid_indices]

    if not use_weight:
        graph_sparse_matrix.data = 1

    community_result = cluster_community_from_graph(
        method=method, graph_sparse_matrix=graph_sparse_matrix, directed=directed, **kwargs
    )

    labels = np.zeros(len(adata), dtype=int) + no_community_label

    # No subset required case, use all indices
    if valid_indices is None:
        valid_indices = np.arange(0, len(adata))
    for i, community in enumerate(community_result.communities):
        labels[valid_indices[community]] = i
    # clusters need to be categorical strings
    adata.obs[result_key] = pd.Categorical(labels.astype(str))

    adata.uns[result_key] = {
        "method": method,
        "adj_matrix_key": adj_matrix_key,
        "use_weight": use_weight,
        "layer": layer,
        "layer_conn_type": conn_type,
        "cell_subsets": cell_subsets,
        "cluster_and_subsets": cluster_and_subsets,
        "directed": directed,
    }

    return adata


def cluster_community_from_graph(graph=None, graph_sparse_matrix=None, method="louvain", directed=False, **kwargs):
    """Detect communities based on graph inputs and selected methods with arguments passed in kwargs.

    Parameters
    ----------
    graph : [type], optional
        [description], by default None
    graph_sparse_matrix : [type], optional
        [description], by default None
    method : str, optional
        [description], by default "louvain"

    Returns
    -------
    [type]
        NodeClustering Object from CDlib

    Raises
    ------
    ImportError
        [description]
    ValueError
        [description]
    NotImplementedError
        [description]
    """
    logger = LoggerManager.get_main_logger()
    logger.info("Detecting communities on graph...")
    try:
        import networkx as nx
        from cdlib import algorithms
    except ImportError:
        raise ImportError(
            "You need to install the excellent package `cdlib` if you want to use louvain or leiden " "for clustering."
        )
    if graph is not None:
        # highest priority
        pass
    elif graph_sparse_matrix is not None:
        logger.info("Converting graph_sparse_matrix to networkx object", indent_level=2)
        # if graph matrix is with weight, then edge attr "weight" stores weight of edges
        graph = nx.convert_matrix.from_scipy_sparse_matrix(graph_sparse_matrix, edge_attribute="weight")
        for i in range(graph_sparse_matrix.shape[0]):
            if not (i in graph.nodes):
                graph.add_node(i)
    else:
        raise ValueError("Expected graph inputs are invalid")

    if directed:
        graph = graph.to_directed()
    else:
        graph = graph.to_undirected()

    if method == "leiden":
        initial_membership, weights = None, None
        if "initial_membership" in kwargs:
            logger.info("Detecting community with initial_membership input from caller")
            initial_membership = kwargs["initial_membership"]
        if "weights" in kwargs:
            weights = kwargs["weights"]

        if initial_membership is not None:
            main_info(
                "Currently initial_membership for leiden has some issue and thus we ignore it. "
                "We will support it in future."
            )
            initial_membership = None

        coms = algorithms.leiden(graph, weights=weights, initial_membership=initial_membership)
    elif method == "louvain":
        if "resolution" not in kwargs:
            raise KeyError("resolution not in louvain input parameters")
        # if "weight" not in kwargs:
        #     raise KeyError("weight not in louvain input parameters")
        if "randomize" not in kwargs:
            raise KeyError("randomize not in louvain input parameters")

        resolution = kwargs["resolution"]
        weight = "weight"
        randomize = kwargs["randomize"]
        coms = algorithms.louvain(graph, weight=weight, resolution=resolution, randomize=randomize)
    elif method == "infomap":
        coms = algorithms.infomap(graph)
    else:
        raise NotImplementedError("clustering algorithm not implemented yet")

    logger.finish_progress(progress_name="Community clustering with %s" % (method))

    return coms
