import numpy as np
from tqdm import tqdm
from scipy.sparse import issparse
from .connectivity import umap_conn_indices_dist_embedding, mnn_from_list
from .utils import get_finite_inds, inverse_norm, einsum_correlation


def cell_wise_confidence(adata, X_data=None, V_data=None, ekey="M_s", vkey="velocity_S", method="jaccard"):
    """ Calculate the cell-wise velocity confidence metric.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        ekey: `str` (optional, default `M_s`)
            The dictionary key that corresponds to the gene expression in the layer attribute. By default, it is the
            smoothed expression `M_s`.
        vkey: 'str' (optional, default `velocity_S`)
            The dictionary key that corresponds to the estimated velocity values in layers attribute.
        method: `str` (optional, default `jaccard`)
            Which method will be used for calculating the cell wise velocity confidence metric.
            By default it uses
            `jaccard` index, which measures how well each velocity vector meets the geometric constraints defined by the
            local neighborhood structure. Jaccard index is calculated as the fraction of the number of the intersected
            set of nearest neighbors from each cell at current expression state (X) and that from the future expression
            state (X + V) over the number of the union of these two sets. The `cosine` or `correlation` method is similar
            to that used by scVelo (https://github.com/theislab/scvelo).

    Returns
    -------
        Adata: :class:`~anndata.AnnData`
            Returns an updated `~anndata.AnnData` with `.obs.confidence` as the cell-wise velocity confidence.
    """

    if ekey is "X": 
        X, V = (adata.X if X_data is None else X_data, adata.layers[vkey] if V_data is None else V_data)
        norm_method = adata.uns["pp_norm_method"].copy()
        adata.uns["pp_norm_method"] = 'log1p'
        X = inverse_norm(adata, X) if X_data is None else X_data
        adata.uns["pp_norm_method"] = norm_method
    else:
        X, V = (adata.layers[ekey] if X_data is None else X_data, adata.layers[vkey] if V_data is None else V_data)
        X = inverse_norm(adata, X) if X_data is None else X_data

    n_neigh, X_neighbors = (
        adata.uns["neighbors"]["params"]["n_neighbors"],
        adata.obsp["connectivities"],
    )
    n_neigh = n_neigh[0] if type(n_neigh) == np.ndarray else n_neigh
    n_pca_components = adata.obsm["X"].shape[1]

    finite_inds = get_finite_inds(V, 0)
    X, V = X[:, finite_inds], V[:, finite_inds]
    if method == "jaccard":
        jac, _, _ = jaccard(X, V, n_pca_components, n_neigh, X_neighbors)
        confidence = jac

    elif method == "hybrid":
        # this is inspired from the locality preservation paper
        jac, intersect_, _ = jaccard(X, V, n_pca_components, n_neigh, X_neighbors)

        confidence = np.zeros(adata.n_obs)
        for i in tqdm(range(adata.n_obs), desc='calculating hybrid method (jaccard + consensus) based cell wise confidence'):
            neigh_ids = np.where(intersect_[i].A)[0] if issparse(intersect_) else np.where(intersect_[i])[0]
            confidence[i] = jac[i] * np.mean(
                [consensus(V[i].A.flatten(), V[j].A.flatten()) for j in neigh_ids]
            )  if issparse(V) else jac[i] * np.mean(
                [consensus(V[i].flatten(), V[j].flatten()) for j in neigh_ids]
            )

    elif method == "cosine":
        indices = adata.uns["neighbors"]["indices"]
        confidence = np.zeros(adata.n_obs)
        for i in tqdm(range(adata.n_obs), desc='calculating cosine based cell wise confidence'):
            neigh_ids = indices[i]
            confidence[i] = np.mean(
                [einsum_correlation(V[i].A, V[j].A.flatten(), type="cosine")[0, 0] for j in neigh_ids]
            ) if issparse(V) else np.mean(
                [einsum_correlation(V[i][None, :], V[j].flatten(), type="cosine")[0, 0] for j in neigh_ids]
            )

    elif method == "consensus":
        indices = adata.uns["neighbors"]["indices"]
        confidence = np.zeros(adata.n_obs)
        for i in tqdm(range(adata.n_obs), desc='calculating consensus based cell wise confidence'):
            neigh_ids = indices[i]
            confidence[i] = np.mean(
                [consensus(V[i].A.flatten(), V[j].A.flatten()) for j in neigh_ids]
            ) if issparse(V) else np.mean(
                [consensus(V[i], V[j].flatten()) for j in neigh_ids]
            )

    elif method == "correlation":
        # this is equivalent to scVelo
        indices = adata.uns["neighbors"]["indices"]
        confidence = np.zeros(adata.n_obs)
        for i in tqdm(range(adata.n_obs), desc='calculating correlation based cell wise confidence'):
            neigh_ids = indices[i]
            confidence[i] = np.mean(
                [einsum_correlation(V[i].A, V[j].A.flatten(), type="pearson")[0, 0] for j in neigh_ids]
            ) if issparse(V) else np.mean(
                [einsum_correlation(V[i][None, :], V[j].flatten(), type="pearson")[0, 0] for j in neigh_ids]
            )

    elif method == "divergence":
        pass

    else:
        raise Exception(
            "The input {} method for cell-wise velocity confidence calculation is not implemented"
            " yet".format(method)
        )

    adata.obs[method + "_velocity_confidence"] = confidence

    return adata


def jaccard(X, V, n_pca_components, n_neigh, X_neighbors):
    from sklearn.decomposition import TruncatedSVD

    transformer = TruncatedSVD(n_components=n_pca_components + 1, random_state=0)
    Xt = X + V
    if issparse(Xt):
        Xt.data[Xt.data < 0] = 0
        Xt.data = np.log2(Xt.data + 1)
    else:
        Xt = np.log2(Xt + 1)
    X_fit = transformer.fit(Xt)
    Xt_pca = X_fit.transform(Xt)[:, 1:]

    V_neighbors, _, _, _ = umap_conn_indices_dist_embedding(
        Xt_pca, n_neighbors=n_neigh, return_mapper=False
    )
    X_neighbors_, V_neighbors_ = (
        X_neighbors.dot(X_neighbors),
        V_neighbors.dot(V_neighbors),
    )
    union_ = X_neighbors_ + V_neighbors_ > 0
    intersect_ = mnn_from_list([X_neighbors_, V_neighbors_]) > 0

    jaccard = (
        (intersect_.sum(1) / union_.sum(1)).A1
        if issparse(X)
        else intersect_.sum(1) / union_.sum(1)
    )

    return jaccard, intersect_, union_


def consensus(x, y):
    x_norm, y_norm = np.linalg.norm(x), np.linalg.norm(y)
    consensus = einsum_correlation(x[None, :], y, type="cosine")[0, 0] * \
                np.min([x_norm, y_norm]) / np.max([x_norm, y_norm])

    return consensus
