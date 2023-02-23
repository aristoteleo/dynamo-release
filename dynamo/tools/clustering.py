from typing import Any, Iterable, List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from ..configuration import DKM
from ..dynamo_logger import main_info
from ..preprocessing.preprocessor_utils import filter_genes_by_outliers as filter_genes
from ..preprocessing.preprocessor_utils import log1p_adata as log1p
from ..preprocessing.preprocessor_utils import normalize_cell_expr_by_size_factors
from ..preprocessing.utils import pca_monocle
from ..utils import LoggerManager, copy_adata
from .connectivity import _gen_neighbor_keys, neighbors
from .utils import update_dict
from .utils_reduceDimension import prepare_dim_reduction, run_reduce_dim


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

    try:
        from hdbscan import HDBSCAN
    except ImportError:
        raise ImportError("You need to install the package `hdbscan`." "install hdbscan via `pip install hdbscan`")

    logger = LoggerManager.gen_logger("dynamo-hdbscan")
    logger.log_time()
    adata = copy_adata(adata) if copy else adata

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

        if basis in adata.obsm_keys():
            X_data = adata.obsm[basis]
        else:
            reduction_method = basis.split("_")[-1]
            embedding_key = "X_" + reduction_method if layer is None else layer + "_" + reduction_method
            neighbor_result_prefix = "" if layer is None else layer
            conn_key, dist_key, neighbor_key = _gen_neighbor_keys(neighbor_result_prefix)

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

            X_data = adata.obsm[basis]

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


def leiden(
    adata: AnnData,
    resolution: float = 1.0,
    use_weight: bool = False,
    weight: Optional[Union[str, Iterable]] = None,
    initial_membership: Optional[List[int]] = None,
    adj_matrix: Optional[csr_matrix] = None,
    adj_matrix_key: Optional[str] = None,
    randomize: Optional[int] = None,
    result_key: Optional[str] = None,
    layer: Optional[str] = None,
    obsm_key: Optional[str] = None,
    selected_cluster_subset: Optional[Tuple[str, List[int]]] = None,
    selected_cell_subset: Optional[List[int]] = None,
    directed: bool = False,
    copy: bool = False,
    **kwargs
) -> anndata.AnnData:
    """Apply leiden clustering to the input adata.

    For other general community detection related parameters, please refer to ``dynamo's``
    :py:meth:`~dynamo.tl.cluster_community` function.

    The Leiden algorithm is an improvement of the Louvain algorithm. Based on the cdlib package, the Leiden algorithm
    consists of three phases:
    (1) local moving of nodes,
    (2) refinement of the partition,
    (3) aggregation of the network based on the refined partition, using the non-refined partition to create an initial
    partition for the aggregate network.

    Args:
        adata: an adata object
        resolution: the resolution of the clustering that determines the level of detail in the clustering process.
            An increase in this value will result in the generation of a greater number of clusters.
        use_weight: whether to use the weight of the edges in the clustering process. Default False.
        weight: weights of edges. Can be either an iterable (list of double) or an edge attribute.
        initial_membership: list of int. Initial membership for the partition.
            If None then defaults to a singleton partition.
        adj_matrix: the adjacency matrix to use for the cluster_community function.
        adj_matrix_key: the key of the adjacency matrix in adata.obsp used for the cluster_community function.
        randomize: seed for the random number generator. By default uses a random seed if nothing is specified.
        result_key: the key to use for saving clustering results which will be included in both adata.obs and adata.uns.
        layer: the adata layer where cluster algorithms will work on.
        obsm_key: the key of the obsm that points to the expression embedding to be used for dyn.tl.neighbors to
            calculate the nearest neighbor graph.
        selected_cluster_subset: a tuple of 2 elements (cluster_key, allowed_clusters) filtering cells in adata based on
            cluster_key in adata.obs and only reserves cells in the allowed clusters.
        selected_cell_subset: a list of cell indices to cluster.
        directed: whether the graph is directed.
        copy: return a copy instead of writing to adata.
        **kwargs: additional arguments to pass to the cluster_community function.

    Returns:
        adata: An updated AnnData object with the leiden clustering results added. The adata is updated up with the
        `result_key` key to use for saving clustering results which will be included in both adata.obs and adata.uns.
        adata.obs[result_key] saves the clustering identify of each cell where the adata.uns[result_key] saves the
        relevant parameters for the leiden clustering .
    """

    kwargs.update(
        {
            "resolution_parameter": resolution,
            "weight": weight,
            "initial_membership": initial_membership,
            "randomize": randomize,
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
    adata: AnnData,
    resolution: float = 1.0,
    use_weight: bool = False,
    weight: Optional[Union[str, Iterable]] = None,
    initial_membership: Optional[List[int]] = None,
    adj_matrix: Optional[csr_matrix] = None,
    adj_matrix_key: Optional[str] = None,
    randomize: Optional[int] = None,
    result_key: Optional[str] = None,
    layer: Optional[str] = None,
    obsm_key: Optional[str] = None,
    selected_cluster_subset: Optional[Tuple[str, List[int]]] = None,
    selected_cell_subset: Optional[List[int]] = None,
    directed: bool = False,
    copy: bool = False,
    **kwargs
) -> anndata.AnnData:
    """Apply louvain clustering to adata.

    For other general community detection related parameters,
    please refer to ``dynamo's`` :py:meth:`~dynamo.tl.cluster_community` function.

    Based on the cdlib package, the Louvain algorithm optimises the modularity in two elementary phases:
    (1) local moving of nodes;
    (2) aggregation of the network.
    In the local moving phase, individual nodes are moved to the community that yields the largest increase in the
    quality function. In the aggregation phase, an aggregate network is created based on the partition obtained in the
    local moving phase. Each community in this partition becomes a node in the aggregate network. The two phases are
    repeated until the quality function cannot be increased further.

    Args:
        adata: an adata object
        resolution: The resolution parameter that determines clustering level of detail.
            Please note that in louvain-igraph, increasing the parameter creates fewer clusters.
            In our code, the resolution parameter in louvain is inverted (1/resolution) to match the effect of leiden,
            As a result, increasing resolution creates more clusters and decreasing it generates fewer.
        use_weight: whether to use the weight of the edges in the clustering process. Default False
        weight: weights of edges. Can be either an iterable (list of double) or an edge attribute.
        initial_membership: list of int. Initial membership for the partition.
            If None then defaults to a singleton partition.
        adj_matrix: the adjacency matrix to use for the cluster_community function. Default None
        adj_matrix_key: adj_matrix_key in adata.obsp used for the cluster_community function. Default None
        randomize: randomState instance or None, optional (default=None).
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.
            according to louvain algorithm's documentation (cdlib.algorithms.louvain)
        result_key: the key to use for saving clustering results which will be included in both adata.obs and adata.uns.
        layer: the adata layer where cluster algorithms will work on.
        obsm_key: the key of the obsm that points to the expression embedding to be used for dyn.tl.neighbors to
            calculate the nearest neighbor graph.
        selected_cluster_subset: a tuple of 2 elements (cluster_key, allowed_clusters) filtering cells in adata based on
            cluster_key in adata.obs and only reserves cells in the allowed clusters.
        selected_cell_subset: a list of cell indices to cluster.
        directed: whether the graph is directed.
        copy: return a copy instead of writing to adata.
        **kwargs: additional arguments to pass to the clustering function.

    Returns:
        adata: An updated AnnData object with the leiden clustering results added. The adata is updated up with the
        `result_key` key to use for saving clustering results which will be included in both adata.obs and adata.uns.
        adata.obs[result_key] saves the clustering identify of each cell where the adata.uns[result_key] saves the
        relevant parameters for the leiden clustering .
    """
    if directed:
        raise ValueError("CDlib does not support directed graph for Louvain community detection for now.")

    kwargs.update(
        {
            "resolution_parameter": resolution,
            "weight": weight,
            "initial_membership": initial_membership,
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
    """Apply infomap community detection algorithm to cluster adata.

    For other community detection general parameters, please refer to ``dynamo's`` :py:meth:`~dynamo.tl.cluster_community` function.
    "Infomap is based on ideas of information theory. The algorithm uses the probability flow of random walks on a network as a proxy for information flows in the real system and it decomposes the network into modules by compressing a description of the probability flow." - cdlib
    """
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
    adata: AnnData,
    method: str = "leiden",
    result_key: Optional[str] = None,
    adj_matrix: Optional[Union[list, np.array, csr_matrix]] = None,
    adj_matrix_key: Optional[str] = None,
    use_weight: bool = False,
    no_community_label: int = -1,
    layer: Optional[str] = None,
    obsm_key: Optional[str] = None,
    cell_subsets: Optional[List[int]] = None,
    cluster_and_subsets: Optional[Tuple[str, List[int]]] = None,
    directed: bool = True,
    copy: bool = False,
    **kwargs
) -> Union[AnnData, None]:
    """A base function for detecting communities and inserting results into adata with algorithms specified parameters
    passed in. Adjacent matrix retrieval priority: adj_matrix > adj_matrix_key > others

    Args:
        adata: adata object
        method: community detection method, by default "leiden"
        result_key: the key where the results are stored in obs, by default None
        adj_matrix: adj_matrix used for clustering, by default None
        adj_matrix_key: adj_matrix_key in adata.obsp used for clustering
        use_weight: if using graph weight or not, by default False meaning using connectivities only (0/1 integer
            values)
        no_community_label: the label value used for nodes not contained in any community, by default -1
        layer: some adata layer which cluster algorithms will work on, by default None
        cell_subsets: cluster only a subset of cells in adata, by default None
        cluster_and_subsets: A tuple of 2 elements (cluster_key, allowed_clusters).filtering cells in adata based on
            cluster_key in adata.obs and only reserve cells in the allowed clusters, by default None
        directed: if the edges in the graph should be directed, by default False
    """

    adata = copy_adata(adata) if copy else adata
    if (layer is not None) and (adj_matrix_key is not None):
        raise ValueError("Please supply one of adj_matrix_key and layer")
    if use_weight:
        conn_type = DKM.OBSP_ADJ_MAT_DIST
    else:
        conn_type = DKM.OBSP_ADJ_MAT_CONNECTIVITY

    # build adj_matrix_key
    if adj_matrix_key is None:
        if layer is None:
            if obsm_key is None:
                adj_matrix_key = conn_type
            else:
                adj_matrix_key = obsm_key + "_" + conn_type
        else:
            adj_matrix_key = layer + "_" + conn_type

    # try generating required adj_matrix according to
    # user inputs through "neighbors" interface
    if adj_matrix is None:
        main_info("accessing adj_matrix_key=%s built from args for clustering..." % (adj_matrix_key))
        if not (adj_matrix_key in adata.obsp):
            if layer is None:
                if obsm_key is None:
                    neighbors(adata)
                else:
                    X_data = adata.obsm[obsm_key]
                    neighbors(adata, X_data=X_data, result_prefix=obsm_key)
            else:
                main_info("using PCA genes for clustering based on adata.var.use_for_pca...")
                X_data = adata[:, adata.var.use_for_pca].layers[layer]
                neighbors(adata, X_data=X_data, result_prefix=layer)

        if not (adj_matrix_key in adata.obsp):
            raise ValueError("%s does not exist in adata.obsp" % adj_matrix_key)

        graph_sparse_matrix = adata.obsp[adj_matrix_key]
    else:
        main_info("using adj_matrix from arg for clustering...")
        graph_sparse_matrix = adj_matrix

    # build result_key for storing results
    if result_key is None:
        if all((cell_subsets is None, cluster_and_subsets is None)):
            result_key = "%s" % (method) if layer is None else layer + "_" + method
        else:
            result_key = "subset_" + method if layer is None else layer + "_subset_" + method

    valid_indices = None
    if cell_subsets is not None:
        if type(cell_subsets[0]) == str:
            valid_indices = [adata.obs_names.get_loc(cur_cell) for cur_cell in cell_subsets]
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

    community_result = cluster_community_from_graph(
        method=method, graph_sparse_matrix=graph_sparse_matrix, directed=directed, **kwargs
    )

    labels = np.zeros(len(adata), dtype=int) + no_community_label

    # No subset required case, use all indices
    if valid_indices is None:
        valid_indices = np.arange(0, len(adata))

    if hasattr(community_result, "membership"):
        labels[valid_indices] = community_result.membership
    else:
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
    if copy:
        return adata


def cluster_community_from_graph(
    graph: Any = None,
    graph_sparse_matrix: Optional[csr_matrix] = None,
    method: str = "leiden",
    directed: bool = False,
    **kwargs
) -> Any:
    """A function takes a graph as input and clusters its nodes into communities using one of three algorithms:
    Leiden, Louvain, or Infomap.

        Args:
            graph: a graph object, by default None.
            graph_sparse_matrix: a sparse matrix that stores the weights of the edges in the graph.
            method: one of three clustering algorithms (Leiden, Louvain, or Infomap).
            directed: if the edges in the graph should be directed, by default False.

        Returns:
            NodeClustering: a NodeClustering object that contains the communities detected by the chosen algorithm.
    """
    logger = LoggerManager.get_main_logger()
    logger.info("Detecting communities on graph...")

    try:
        import igraph
        import leidenalg
    except ImportError:
        raise ImportError(
            "Please install networkx, igraph, leidenalg via "
            "`pip install networkx or igraph or leidenalg` for clustering on graph."
        )

    initial_membership, weights, seed = None, None, None
    if "initial_membership" in kwargs:
        logger.info("Detecting community with initial_membership input from caller")
        initial_membership = kwargs["initial_membership"]
        kwargs.pop("initial_membership")
    if "weight" in kwargs:
        weights = kwargs["weight"]
        kwargs.pop("weight")
    if "randomize" in kwargs:
        seed = kwargs["randomize"]
        kwargs.pop("randomize")

    if graph is not None:
        # highest priority
        pass
    elif graph_sparse_matrix is not None:
        logger.info("Converting graph_sparse_matrix to igraph object", indent_level=2)
        # if graph matrix is with weight, then edge attr "weight" stores weight of edges
        graph = igraph.Graph.Adjacency((graph_sparse_matrix > 0), directed=directed)
    else:
        raise ValueError("Expected graph inputs are invalid")

    if method == "leiden":
        if initial_membership is not None:
            main_info(
                "Currently initial_membership for leiden has some issue and thus we ignore it. "
                "We will support it in future."
            )
            initial_membership = None

        # ModularityVertexPartition does not accept a resolution_parameter, instead RBConfigurationVertexPartition.
        if kwargs["resolution_parameter"] != 1:
            partition_type = leidenalg.RBConfigurationVertexPartition
        else:
            partition_type = leidenalg.ModularityVertexPartition
            kwargs.pop("resolution_parameter")

        coms = leidenalg.find_partition(
            graph, partition_type, initial_membership=initial_membership, weights=weights, seed=seed, **kwargs
        )

    elif method == "louvain":
        try:
            import louvain
        except ImportError:
            raise ImportError("Please install louvain via `pip install python-louvain==0.14` for clustering on graph.")

        # convert louvain's resolution so that the effect between leiden and louvain is the same.
        coms = louvain.find_partition(graph, louvain.RBConfigurationVertexPartition, seed=seed, **kwargs)
    elif method == "infomap":
        try:
            import cdlib as algorithms
        except ImportError:
            raise ImportError("Please install cdlib via `pip install cdlib` for clustering on graph.")
        coms = algorithms.infomap(graph)
    else:
        raise NotImplementedError("clustering algorithm not implemented yet")

    logger.finish_progress(progress_name="Community clustering with %s" % (method))

    return coms


def scc(
    adata: anndata.AnnData,
    min_cells: int = 100,
    spatial_key: str = "spatial",
    e_neigh: int = 30,
    s_neigh: int = 6,
    resolution: Optional[float] = None,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """Spatially constrained clustering (scc) to identify continuous tissue domains.

    Args:
        adata: an Anndata object, after normalization.
        min_cells: minimal number of cells the gene expressed.
        spatial_key: the key in `.obsm` that corresponds to the spatial coordinate of each bucket.
        e_neigh: the number of nearest neighbor in gene expression space.
        s_neigh: the number of nearest neighbor in physical space.
        resolution: the resolution parameter of the leiden clustering algorithm.
        copy: Whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments.
            Defaults to False.

    Returns:
        Depends on the argument `copy` return either an `~anndata.AnnData` object with cluster info in "scc_e_{a}_s{b}"
        or None.
    """

    filter_genes(adata, min_cell_s=min_cells)
    adata.uns["pp"] = {}
    normalize_cell_expr_by_size_factors(adata, layers="X")
    log1p(adata)
    pca_monocle(adata, n_pca_components=30, pca_key="X_pca")

    neighbors(adata, n_neighbors=e_neigh)
    if "X_" + spatial_key not in adata.obsm.keys():
        adata.obsm["X_" + spatial_key] = adata.obsm[spatial_key].copy()

    neighbors(adata, n_neighbors=s_neigh, basis=spatial_key, result_prefix="spatial")
    conn = adata.obsp["connectivities"].copy()
    conn.data[conn.data > 0] = 1
    adj = conn + adata.obsp["spatial_connectivities"]
    adj.data[adj.data > 0] = 1
    leiden(adata, adj_matrix=adj, resolution=resolution, result_key="scc_e" + str(e_neigh) + "_s" + str(s_neigh))

    if copy:
        return adata
    return None


def purity(
    adata,
    neighbor: int = 30,
    resolution: Optional[float] = None,
    spatial_key: str = "spatial",
    neighbors_key: str = "spatial_connectivities",
    cluster_key: str = "leiden",
) -> float:
    """Calculate the puriority of the scc's clustering results.

    Args:
        adata: an adata object
        neighbor: the number of nearest neighbor in physical space.
        resolution: the resolution parameter of the leiden clustering algorithm.
        spatial_key: the key in `.obsm` that corresponds to the spatial coordinate of each bucket.
        neighbors_key: the key in `.obsp` that corresponds to the spatial nearest neighbor graph.
        cluster_key: the key in `.obsm` that corresponds to the clustering identity.

    Returns:
        purity_score: the average purity score across cells.
    """

    if neighbors_key not in adata.obsp.keys():
        neighbors(adata, n_neighbors=neighbor, basis=spatial_key, result_prefix=neighbors_key.split("_")[0])

    neighbor_graph = adata.obsp[neighbors_key]

    if cluster_key not in adata.obs.columns:
        leiden(adata, adj_matrix=neighbor_graph, resolution=resolution, result_key=cluster_key)

    cluster = adata.obs[cluster_key]

    purity_score = np.zeros(adata.n_obs)
    for i in np.arange(adata.n_obs):
        cur_cluster = cluster[i]
        other_cluster = neighbor_graph[0].nonzero()[1]
        other_cluster = cluster[other_cluster]
        other_cluster = other_cluster[: min([neighbor, len(other_cluster)])]

        purity_score[i] = sum(other_cluster == cur_cluster) / len(other_cluster)

    purity_score = purity_score.mean()

    return purity_score
