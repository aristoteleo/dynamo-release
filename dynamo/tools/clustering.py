from hdbscan import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np
from .utils_reduceDimension import prepare_dim_reduction, run_reduce_dim
from .utils import update_dict


def hdbscan(adata,
            X_data=None,
            genes=None,
            layer=None,
            basis='pca',
            dims=None,
            n_pca_components=30,
            n_components=2,
            **hdbscan_kwargs):
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
    hdbscan_kwargs: `dict`
        Additional parameters that will be passed to hdbscan function.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            A updated AnnData object with the clustering updated. `hdbscan` and `hdbscan_prob` are two newly added columns
            from .obs, corresponding to either the Cluster results or the probability of each cell belong to a cluster.
            `hdbscan` key in .uns corresponds to a dictionary that includes additional results returned from hdbscan run.
    """

    if X_data is None:
        _, n_components, has_basis, basis = prepare_dim_reduction(adata,
                              genes=genes,
                              layer=layer,
                              basis=basis,
                              dims=dims,
                              n_pca_components=n_pca_components,
                              n_components=n_components, )

    if has_basis:
        X_data = adata.obsm[basis]
    else:
        reduction_method = basis.split('_')[-1]
        embedding_key = (
            "X_" + reduction_method if layer is None else layer + "_" + reduction_method
        )
        neighbor_key = "neighbors" if layer is None else layer + "_neighbors"

        adata = run_reduce_dim(adata, X_data, n_components, n_pca_components, reduction_method,
                               embedding_key=embedding_key, n_neighbors=30, neighbor_key=neighbor_key, cores=1)

    X_data = X_data if dims is None else X_data[:, dims]

    if hdbscan_kwargs is not None and 'metric' in hdbscan_kwargs.keys():
        if hdbscan_kwargs['metric'] == 'cosine':
            from sklearn.preprocessing import normalize
            X_data = normalize(X_data, norm='l2')

    h_kwargs = {"min_cluster_size": 5,
                "min_samples": None,
                "metric": "euclidean",
                "p": None,
                "alpha": 1.0,
                "cluster_selection_epsilon": 0.0,
                "algorithm": 'best',
                "leaf_size": 40,
                "approx_min_span_tree": True,
                "gen_min_span_tree": False,
                "core_dist_n_jobs": 1,
                "cluster_selection_method": 'eom',
                "allow_single_cluster": False,
                "prediction_data": False,
                "match_reference_implementation": False,
                }

    h_kwargs = update_dict(h_kwargs, hdbscan_kwargs)
    cluster = HDBSCAN(**h_kwargs)
    cluster.fit(X_data)
    adata.obs['hdbscan'] = cluster.labels_.astype('str')
    adata.obs['hdbscan_prob'] = cluster.probabilities_
    adata.uns['hdbscan'] = {'hdbscan': cluster.labels_.astype('str'),
                            "probabilities_": cluster.probabilities_,
                            "cluster_persistence_": cluster.cluster_persistence_,
                            "outlier_scores_": cluster.outlier_scores_,
                            "exemplars_": cluster.exemplars_,
                            }

    return adata


def cluster_field(adata,
                  basis='pca',
                  embedding_basis=None,
                  normalize=True,
                  method='louvain',
                  cores=1,
                  **kwargs):
    """Cluster cells based on vector field features.

    We would like to see whether the vector field can be used to better define cell state/types. This can be accessed via
    characterizing critical points (attractor/saddle/repressor, etc.) and characteristic curves (nullcline, separatrix).
    However, the calculation of those is not easy, for example, a strict definition of an attractor is states where
    velocity is 0 and the eigenvalue of the jacobian matrix at that point is all negative. Under this strict definition,
    we may sometimes find the attractors are very far away from our sampled cell states which makes them less meaningful.
    This is not unexpected as the vector field we learned is defined via a set of basis functions based on gaussian
    kernels and thus it is hard to satisfy that strict definition.

    Fortunately, we can handle this better with the help of a different set of ideas. Instead of using critical points
    by the classical dynamic system methods, we can use some machine learning approaches that are based on extracting
    geometric features of streamline to "cluster vector field space" for define cell states/type. This requires calculating,
    potential (ordered pseudotime), speed, curliness, divergence, acceleration, curvature, etc. Thanks to the fact that we
    can analytically calculate Jacobian matrix matrix, those quantities of the vector field function can be conveniently
    and efficiently calculated.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`.
        adata object that includes both newly synthesized and total gene expression of cells. Alternatively,
        the object should include both unspliced and spliced gene expression of cells.
    basis: `str` or None (default: `None`)
        The space that will be used for calculating vector field features. Valid names includes, for example, `pca`, `umap`, etc.
    embedding_basis: `str` or None (default: `None`)
        The embedding basis that will be combined with the vector field feature space for clustering.
    normalize: `bool` (default: `True`)
        Whether to mean center and scale the feature across all cells so that the mean
    method: `str` (default: `louvain`)
        The method that will be used for clustering, one of `{'kmeans'', 'hdbscan', 'louvain', 'leiden'}`. If `louvain`
        or `leiden` used, you need to have `scanpy` installed.
    cores: `int` (default: 1)
        The number of parallel jobs to run for neighbors search. ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    kwargs:
        Any additional arguments that will be passed to either kmeans, hdbscan, louvain or leiden clustering algorithms.

    Returns
    -------

    """

    if method in ['louvain', 'leiden']:
        try:
            import scanpy as sc
        except ImportError:
            raise ImportError("You need to install the excellent package `scanpy` if you want to use louvain or leiden "
                              "for clustering.")

    feature_key = ['speed_' + basis, basis + '_ddhodge_potential', 'divergence_' + basis, 'acceleration_' + basis,
                   'curvature_' + basis]

    if feature_key[0] not in adata.obs.keys():
        from .vector_calculus import speed
        speed(adata, basis=basis)
    if feature_key[1] not in adata.obs.keys():
        from ..ext import ddhodge
        ddhodge(adata, basis=basis)
    if feature_key[2] not in adata.obs.keys():
        from .vector_calculus import divergence
        divergence(adata, basis=basis)
    if feature_key[3] not in adata.obs.keys():
        from .vector_calculus import acceleration
        acceleration(adata, basis=basis)
    if feature_key[4] not in adata.obs.keys():
        from .vector_calculus import curvature
        curvature(adata, basis=basis)

    feature_data = adata.obs.loc[:, feature_key].values
    if embedding_basis is None: embedding_basis = basis
    X = np.hstack((feature_data, adata.obsm['X_' + embedding_basis]))

    if normalize:
        # X = (X - X.min(0)) / X.ptp(0)
        X = (X - X.mean(0)) / X.std(0)

    if method in ['hdbscan', 'kmeans']:
        if method == 'hdbscan':
            hdbscan(adata, X_data=X, **kwargs)
        elif method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(random_state=0, **kwargs).fit(X)
            adata.obs['kmeans'] = kmeans.labels_.astype('str')

    elif method in ['louvain', 'leiden']:
        if X.shape[0] > 200000 and X.shape[1] > 2: 
            from pynndescent import NNDescent

            nbrs = NNDescent(X, metric='euclidean', n_neighbors=31, n_jobs=cores,
                              random_state=19491001)
            nbrs_idx, dist = nbrs.query(X, k=31)
        else:
            nbrs = NearestNeighbors(n_neighbors=31, n_jobs=cores).fit(X)
            dist, nbrs_idx = nbrs.kneighbors(X)

        row = np.repeat(nbrs_idx[:, 0], 30)
        col = nbrs_idx[:, 1:].flatten()
        g = csr_matrix((np.repeat(1, len(col)), (row, col)), shape=(adata.n_obs, adata.n_obs))
        adata.obsp['feature_knn'] = g

        if method == 'louvain':
            sc.tl.louvain(adata, obsp='feature_knn', **kwargs)
        elif method == 'leiden':
            sc.tl.leiden(adata, obsp='feature_knn', **kwargs)
