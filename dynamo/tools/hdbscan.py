import hdbscan
from .utils_reduceDimension import prepare_dim_reduction, run_reduce_dim
from .utils import update_dict

def hdbscan(adata,
            X_data=None,
            dims=None,
            genes=None,
            layer=None,
            basis='pca',
            n_pca_components=30,
            n_components=2,
            **hdbscan_kwargs):
    """

    Parameters
    ----------
    adata
    X_data
    dims
    genes
    layer
    basis
    n_pca_components
    n_components
    hdbscan_kwargs

    Returns
    -------

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
                "alpha": 1,
                "cluster_selection_epsilon": 0,
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
    cluster = hdbscan.HDBSCAN(*h_kwargs)
    cluster.fit(X_data)
    adata.obs['Cluster'] = cluster.labels_
    adata.uns['hdbscan'] = {'Cluster': cluster.labels_,
                            "probabilities_": cluster.probabilities_,
                            "cluster_persistence_": cluster.cluster_persistence_,
                            "outlier_scores_": cluster.outlier_scores_,
                            "exemplars_": cluster.exemplars_,
                            "relative_validity_": cluster.relative_validity_,
                            }

    return adata

