import numpy as np
import scipy as scp
from scipy.sparse import coo_matrix, issparse
import scipy.linalg


def cell_velocities(adata, vkey='pca', basis='umap', method='analytical', neg_cells_trick=False):
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
    method: `string` (optimal, default 'new')
        The method to calculate the transition matrix and project high dimensional vector to low dimension, either new or
        empirical. "Empiricial" is the method used in the original RNA velocity paper.
    neg_cells_trick: 'bool' (optional, default False)
        Whether we should handle cells having negative correlations in gene expression difference with high dimensional
        velocity vector separately. This option is inspired from scVelo package (https://github.com/theislab/scvelo).

    Returns
    -------
    Returns an updated `~anndata.AnnData` with transition_matrix and projected embedding of high dimension velocity vector
    in the existing embeding of current cell state, calculated using method from (La Manno et al. 2018).
    """

    neighbors, dist, indices = adata.uns['neighbors']['connectivities'], adata.uns['neighbors']['distances'], adata.uns['neighbors']['indices']
    V_mat = adata.obsm['velocity_' + vkey] if 'velocity_' + vkey in adata.obsm.keys() else None
    X_pca, X_embedding = adata.obsm['X_pca'], adata.obsm['X_'+basis]

    n, knn = neighbors.shape[0], indices.shape[1] - 1 #remove the first one in kNN
    rows = np.zeros((n * knn, 1))
    cols = np.zeros((n * knn, 1))
    vals = np.zeros((n * knn, 1))

    delta_X = np.zeros((n, X_embedding.shape[1]))

    if method == 'analytical':
        Q = np.zeros((n, knn))
        U = np.zeros((n, 2))
        for i in range(n):
            y = X_pca[i]
            v = V_mat[i]
            Y = X_pca[indices[i, 1:]]
            q, u = markov_combination(y, v, Y)
            Q[i] = q.T
            U[i] = (X_embedding[indices[i, 1:]] - X_embedding[i]).T.dot(np.array(q)).T # project in two dimension

            delta_X[i, :] = X_embedding[i, :] + U[i]

        T = makeTransitionMatrix(Q, indices[:, 1:], 1e-4)
    elif method == 'empiricial':
        idx = 0
        for i in range(n):
            i_vals = np.zeros((knn, 1))
            velocity = V_mat[i, :] # project V_mat to pca space
            diff_velocity = np.sign(velocity) * np.sqrt(np.abs(velocity))

            for j in np.arange(1, knn + 1):
                neighbor_ind_j = indices[i, j]
                diff = X_pca[neighbor_ind_j, :] - X_pca[i, :]
                diff_rho = np.sign(diff) * np.sqrt(np.abs(diff))
                pearson_corr = np.corrcoef(diff_rho, diff_velocity)[0, 1]

                rows[idx] = i
                cols[idx] = neighbor_ind_j
                i_vals[j - 1] = pearson_corr
                idx = idx + 1

            if neg_cells_trick:
                val_ind_vec = np.array(range(i*knn,(i+1)*knn))
                for sig in [-1, 1]:
                    cur_ind = np.where(np.sign(i_vals) == sig)[0]
                    if len(cur_ind) == 0:
                        continue

                    cur_i_vals = i_vals[cur_ind]
                    sigma = max(abs(cur_i_vals))
                    exp_i_vals = np.exp(np.abs(cur_i_vals) / sigma)
                    denominator = sum(exp_i_vals)
                    i_prob = exp_i_vals / denominator
                    vals[val_ind_vec[cur_ind]] = sig * i_prob

                    j_vec = indices[i, 1:][cur_ind]
                    numerator = sig * np.array([X_embedding[j, :] - X_embedding[i, :] for j in j_vec])
                    denominator = np.array([[scp.linalg.norm(numerator[j]) for j in range(len(j_vec))]]).T

                    delta_X[i, :] += 0.5 * (i_prob - 1 / len(cur_ind)).T.dot(numerator / np.hstack((denominator, denominator))).flatten()
            else:
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
    adata.obsm['velocity_' + basis] = delta_X

    return adata


def markov_combination(x, v, X):
    from cvxopt import matrix, solvers

    n = X.shape[0]
    R = matrix((X - x).astype('double')).T
    H = R.T * R
    f = matrix((v).astype('double')).T * R
    G = np.vstack((-np.eye(n),
                   np.ones(n)))
    h = np.zeros(n+1)
    h[-1] = 1
    solvers.options['show_progress'] = False

    p = solvers.qp(H, -f.T, G=matrix(G), h=matrix(h))['x']
    u = R * p
    return p, u


def makeTransitionMatrix(Q, I, tol=0.):
    n = Q.shape[0]
    M = np.zeros((n, n))

    for i in range(n):
        q = Q[i]
        q[q < tol] = 0
        M[I[i], i] = q
        M[i, i] = 1 - np.sum(q)
    return M


def diffusion(M, P0=None, steps=None, backward=False):
    """Find the state distribution of a Markov process.

    Parameters
    ----------
        M: `numpy.ndarray` (dimension n x n, where n is the cell number)
            The transition matrix.
        P0: `numpy.ndarray` (default None; dimension is n, )
            The initial cell state.
        steps: `int` (default None)
            The random walk steps on the Markov transitioin matrix.
        backward: `bool` (default False)
            Whether the backward transition will be considered.

    Returns
    -------
        Mu: `numpy.ndarray`
            The state distribution of the Markov process.
    """

    if backward is True:
        M = M.T

    if steps is None:
        # code inspired from  https://github.com/prob140/prob140/blob/master/prob140/markov_chains.py#L284
        from scipy.sparse.linalg import eigs
        eigenvalue, eigenvector = scp.linalg.eig(M, left=True, right=False) if not issparse(M) else eigs(M) # source is on the row

        eigenvector = np.real(eigenvector) if not issparse(M) else np.real(eigenvector.T)
        eigenvalue_1_ind = np.isclose(eigenvalue, 1)
        mu = eigenvector[:, eigenvalue_1_ind] / np.sum(b[:, eigenvalue_1_ind])

        # Zero out floating poing errors that are negative.
        indices = np.logical_and(np.isclose(mu, 0),
                                 mu < 0)
        mu[indices] = 0 # steady state distribution

    else:
        mu = np.nanmean(M.dot(np.linalg.matrix_power(M, steps)), 0) if P0 is None else P0.dot(np.linalg.matrix_power(M, steps))

    return mu


def expected_return_time(M, backward=False):
    """Find the expected returning time.

    Parameters
    ----------
        M: `numpy.ndarray` (dimension n x n, where n is the cell number)
            The transition matrix.
        backward: `bool` (default False)
            Whether the backward transition will be considered.

    Returns
    -------
        T: `numpy.ndarray`
            The expected return time (1 / steady_state_probability).

    """
    steady_state = diffusion(M, P0=None, steps=None, backward=backward)

    T = 1 / steady_state
    return T

