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

    n, knn = neighbors.shape[0], indices.shape[1]
    rows = np.zeros(n * knn, 1) # check the first one in kNN
    cols = np.zeros(n * knn, 1)
    vals = np.zeros(n * knn, 1)

    delta_X = np.zeros(n, X_embedding.shape[1])

    idx = 0
    for i in range(n):
        i_vals = np.zeros(knn, 1)
        velocity = V_mat[i, :]

        for j in range(knn):
            diff = X_pca[i, :] - X_pca[j, :]
            diff_rho = np.sign(diff) * np.sqrt(np.abs(diff))
            diff_velocity = np.sign(velocity) * np.sqrt(np.abs(velocity))
            pearson_corr = np.corrcoef(diff_rho, diff_velocity)[0, 1]

            rows[idx] = i
            cols[idx] = j
            i_vals[j] = pearson_corr
            idx = idx + 1

        sigma = max(abs(i_vals))
        exp_i_vals = np.exp(i_vals / sigma)
        denominator = sum(exp_i_vals)
        i_prob = exp_i_vals / denominator
        vals[(i-1)*knn:i*knn] = i_prob

        j_vec = indices[i, :]
        numerator = [X_embedding[j, :] - X_embedding[i, :] for j in j_vec]
        denominator = [scp.linalg.norm(numerator[j]) for j in numerator]

        delta_X[i, :] = i_prob.dot(numerator / denominator)

    T = coo_matrix((vals, rows, cols), shape=neighbors.shape)

    adata.uns['transition_matrix'] = T
    adata.obsm['Velocity_' + basis] = delta_X

    return adata
