import numpy as np
import scipy as sp
import numdifftools as nd
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from scipy.sparse import issparse
from .connectivity import umap_conn_indices_dist_embedding, mnn_from_list
from .utils import get_finite_inds


def cell_wise_confidence(adata, ekey='M_s', vkey='velocity_S', method='jaccard'):
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

    X, V = (adata.X, adata.layers[vkey]) if ekey is 'X' else (adata.layers[ekey], adata.layers[vkey])
    n_neigh, X_neighbors = adata.uns['neighbors']['params']['n_neighbors'], adata.uns['neighbors']['connectivities']
    n_pca_components = adata.obsm['X_pca'].shape[1]

    finite_inds = get_finite_inds(V, 0)
    X, V = X[:, finite_inds], V[:, finite_inds]
    if method == 'jaccard':
        jac, _, _ = jaccard(X, V, n_pca_components, n_neigh, X_neighbors)
        confidence = jac

    elif method == 'hybrid':
        # this is inspired from the localcity preservation paper
        jac, intersect_, _ = jaccard(X, V, n_pca_components, n_neigh, X_neighbors)

        confidence = np.zeros(adata.n_obs)
        for i in range(adata.n_obs):
            neigh_ids = np.where(intersect_[i].A)[0]
            confidence[i] = jac[i] * np.mean([consensus(V[i].A.flatten(), V[j].A.flatten()) for j in neigh_ids])  # X

    elif method == 'cosine':
        indices = adata.uns['neighbors']['indices']
        confidence = np.zeros(adata.n_obs)
        for i in range(adata.n_obs):
            neigh_ids = indices[i]
            confidence[i] = np.mean([cosine(V[i].A.flatten(), V[j].A.flatten()) for j in neigh_ids])  # X

    elif method == 'consensus':
        indices = adata.uns['neighbors']['indices']
        confidence = np.zeros(adata.n_obs)
        for i in range(adata.n_obs):
            neigh_ids = indices[i]
            confidence[i] = np.mean([consensus(V[i].A.flatten(), V[j].A.flatten()) for j in neigh_ids])  # X

    elif method == 'correlation':
        # this is equivalent to scVelo
        indices = adata.uns['neighbors']['indices']
        confidence = np.zeros(adata.n_obs)
        for i in range(adata.n_obs):
            neigh_ids = indices[i]
            confidence[i] = np.mean([pearsonr(V[i].A.flatten(), V[j].A.flatten())[0] for j in neigh_ids]) # X

    elif method == 'divergence':
        pass

    else:
        raise Exception('The input {} method for cell-wise velocity confidence calculation is not implemented'
                        ' yet'.format(method))

    adata.obs[method + '_velocity_confidence'] = confidence

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

    V_neighbors, _, _, _ = umap_conn_indices_dist_embedding(Xt_pca, n_neighbors=n_neigh)
    X_neighbors_, V_neighbors_ = X_neighbors.dot(X_neighbors), V_neighbors.dot(V_neighbors)
    union_ = X_neighbors_ + V_neighbors_ > 0
    intersect_ = mnn_from_list([X_neighbors_, V_neighbors_]) > 0

    jaccard = (intersect_.sum(1) / union_.sum(1)).A1 if issparse(X) else intersect_.sum(1) / union_.sum(1)

    return jaccard, intersect_, union_

def consensus(x, y):
    x_norm, y_norm = np.linalg.norm(x), np.linalg.norm(y)
    consensus = cosine(x, y) * np.min([x_norm, y_norm]) / np.max([x_norm, y_norm])

    return consensus

def curl(f,x):
    jac = nd.Jacobian(f)(x)
    return sp.array([jac[1,0]-jac[0,1]]) # 2D curl


def divergence(f):
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # plt.pcolormesh(x, y, g)
    # plt.colorbar()
    # plt.savefig( 'Div' + str(NY) +'.png', format = 'png')
    # plt.show()
