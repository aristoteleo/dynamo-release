import numpy as np
import scipy as scp
from scipy.sparse import coo_matrix


def cell_velocities(adata, vkey = 'velocity', basis = 'umap'):
    """Compute transition probability and project high dimension velocity vector to existing low dimension embedding.

    We may consider using umap transform function or applying a neuron net model to project the velocity vectors.

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    vkey: 'int' (optional, default velocity)
        The dictionary key that corresponds to the estimated velocity values in layers slot.
    basis: 'int' (optional, default umap)
        The dictionary key that corresponds to the reduced dimension in obsm slot.

    Returns
    -------
    Returns an updated `~anndata.AnnData` with transition_matrix and projected embedding of high dimension velocity vector
    in the existing embeding of current cell state, calculated using method from (La Manno et al. 2018).
    """

    neighbors, dist, indices = adata.uns['neighbors']['connectivities'], adata.uns['neighbors']['distances'], adata.uns['neighbors']['indices']
    V_mat = adata.layers[vkey] if vkey in adata.layers.keys() else None
    X_pca, X_embedding = adata.obsm['X_pca'], adata.obsm['X_'+basis]

    n, knn = neighbors.shape[0], indices.shape[1] - 1 #remove the first one in kNN
    rows = np.zeros((n * knn, 1))
    cols = np.zeros((n * knn, 1))
    vals = np.zeros((n * knn, 1))

    delta_X = np.zeros((n, X_embedding.shape[1]))

    idx = 0
    for i in range(n):
        i_vals = np.zeros((knn, 1))
        velocity = V_mat[i, :] # project V_mat to pca space
        diff_velocity = np.sign(velocity) * np.sqrt(np.abs(velocity))

        for j in np.arange(1, knn):
            neighbor_ind_j = indices[i, j]
            diff = X_pca[neighbor_ind_j, :] - X_pca[i, :]
            diff_rho = np.sign(diff) * np.sqrt(np.abs(diff))
            pearson_corr = np.corrcoef(diff_rho, diff_velocity)[0, 1]

            rows[idx] = i
            cols[idx] = neighbor_ind_j
            i_vals[j] = pearson_corr
            idx = idx + 1

        sigma = max(abs(i_vals))
        exp_i_vals = np.exp(i_vals / sigma)
        denominator = sum(exp_i_vals)
        i_prob = exp_i_vals / denominator
        vals[i*knn:(i+1)*knn] = i_prob

        j_vec = indices[i, 1:]
        numerator = np.array([X_embedding[j, :] - X_embedding[i, :] for j in j_vec])
        denominator = np.array([[scp.linalg.norm(numerator[j]) for j in range(knn)]]).T

        delta_X[i, :] = (i_prob - 1 / knn).T.dot(numerator / np.hstack((denominator, denominator)))

    T = coo_matrix((vals.flatten(), (rows.flatten(), cols.flatten())), shape=neighbors.shape)

    adata.uns['transition_matrix'] = T
    adata.obsm['Velocity_' + basis] = delta_X

    return adata
