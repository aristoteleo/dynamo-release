import scipy as scp
from scipy.sparse import csr_matrix, issparse
from .Markov import *
from numba import jit

def cell_velocities(adata, ekey='M_s', vkey='velocity_S', basis='umap', method='analytical', neg_cells_trick=False, calc_rnd_vel=False,
                    xy_grid_nums=(50, 50), sample_fraction=None, random_seed=19491001, **kmc_kwargs):
    """Compute transition probability and project high dimension velocity vector to existing low dimension embedding.

    It is powered by the Itô kernel that not only considers the correlation between the vector from any cell to its
    nearest neighbors and its velocity vector but also the corresponding distances. We expect this new kernel will enable
    us to visualize more intricate vector flow or steady states in low dimension. We also expect it will improve the
    calculation of the stationary distribution or source states of sampled cells. The original "correlation" velocity projection
    method is also supported.

    Arguments
    ---------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        ekey: `str` (optional, default `M_s`)
            The dictionary key that corresponds to the gene expression in the layer attribute. By default, it is the smoothed expression
            `M_s`.
        vkey: 'str' (optional, default `velocity`)
            The dictionary key that corresponds to the estimated velocity values in layers attribute.
        basis: 'int' (optional, default `umap`)
            The dictionary key that corresponds to the reduced dimension in `.obsm` attribute.
        method: `string` (optimal, default `analytical`)
            The method to calculate the transition matrix and project high dimensional vector to low dimension, either `analytical`
            or `empirical`. "analytical" is our new approach to learn the transition matrix via diffusion approximation or an Itô
            kernel. "empirical" is the method used in the original RNA velocity paper via correlation. "analytical" option is
            better than "empirical" as it not only considers the correlation but also the distance of the nearest neighbors to
            the high dimensional velocity vector.
        neg_cells_trick: 'bool' (optional, default `False`)
            Whether we should handle cells having negative correlations in gene expression difference with high dimensional
            velocity vector separately. This option is inspired from scVelo package (https://github.com/theislab/scvelo). Not
            required if method is set to be "analytical".
        calc_rnd_vel: `bool` (default: `False`)
            A logic flag to determine whether we will calculate the random velocity vectors which can be plotted downstream
            as a negative control and used to adjust the quiver scale of the velocity field.
        xy_grid_nums: `tuple` (default: `(50, 50)`).
            A tuple of number of grids on each dimension.
        sample_fraction: `None` or `float` (default: None)
            The downsampled fraction of kNN for the purpose of acceleration.
        random_seed: `int` (default: 19491001)
            The random seed for numba to ensure consistency of the random velocity vectors. Default value 19491001 is a special
            day for those who care.

    Returns
    -------
        Adata: :class:`~anndata.AnnData`
            Returns an updated `~anndata.AnnData` with transition_matrix and projected embedding of high dimension velocity vector
            in the existing embedding of current cell state, calculated using either the Itô kernel method (default) or the diffusion
            approximation or the method from (La Manno et al. 2018).
    """

    if calc_rnd_vel:
        numba_random_seed(random_seed)

    neighbors, dist, indices = adata.uns['neighbors']['connectivities'], adata.uns['neighbors']['distances'], adata.uns['neighbors']['indices']
    if 'use_for_dynamo' in adata.var.keys():
        X = adata[:, adata.var.use_for_dynamo.values].layers[ekey]
        V_mat = adata[:, adata.var.use_for_dynamo.values].layers[vkey] if vkey in adata.layers.keys() else None
    else:
        X = adata.layers[ekey]
        V_mat = adata.layers[vkey] if vkey in adata.layers.keys() else None

    X_embedding = adata.obsm['X_'+basis][:, :2]
    V_mat = V_mat.A if issparse(V_mat) else V_mat
    X = X.A if issparse(X) else X

    # add both source and sink distribution
    if method == 'analytical':
        kmc = KernelMarkovChain()
        ndims = X.shape[1]
        kmc_args = {"M_diff": 0.25 * np.eye(ndims), "epsilon": 10**2, "adaptive_local_kernel": True, "tol": 1e-7}
        kmc_args.update(kmc_kwargs)

        # number of kNN in neighbor_idx may be too small
        kmc.fit(X, V_mat, neighbor_idx=indices, k=min(500, X.shape[0] - 1), sample_fraction=sample_fraction, **kmc_args)
        T = kmc.P
        delta_X = kmc.compute_density_corrected_drift(X_embedding, kmc.Idx, normalize_vector=True) # indices, k = 500
        # P = kmc.compute_stationary_distribution()
        # adata.obs['stationary_distribution'] = P
        X_grid, V_grid, D = velocity_on_grid(X_embedding, delta_X, xy_grid_nums=xy_grid_nums)

        if calc_rnd_vel:
            kmc = KernelMarkovChain()
            ndims = X.shape[1]
            permute_rows_nsign(V_mat)
            kmc.fit(X, V_mat, k=min(500, X.shape[0] - 1), M_diff=4 * np.eye(ndims),
                    epsilon=None,
                    adaptive_local_kernel=True, tol=1e-7)  # neighbor_idx=indices,
            T_rnd = kmc.P
            delta_X_rnd = kmc.compute_density_corrected_drift(X_embedding, kmc.Idx,
                                                          normalize_vector=True)  # indices, k = 500
            # P_rnd = kmc.compute_stationary_distribution()
            # adata.obs['stationary_distribution_rnd'] = P_rnd
            X_grid_rnd, V_grid_rnd, D_rnd = velocity_on_grid(X_embedding, delta_X_rnd, xy_grid_nums=xy_grid_nums)

        adata.uns['transition_matrix'] = kmc
    elif method == 'empirical': # add random velocity vectors calculation below
        T, delta_X, X_grid, V_grid, D = _empirical_vec(X, X_embedding, V_mat, indices, neg_cells_trick, xy_grid_nums, neighbors)

        if calc_rnd_vel:
            permute_rows_nsign(V_mat)
            T_rnd, delta_X_rnd, X_grid_rnd, V_grid_rnd, D_rnd = _empirical_vec(X, X_embedding, V_mat, indices, neg_cells_trick, xy_grid_nums, neighbors)

    adata.uns['transition_matrix'] = T
    adata.obsm['velocity_' + basis] = delta_X
    adata.uns['grid_velocity_' + basis] = {'X_grid': X_grid, "V_grid": V_grid, "D": D}

    if calc_rnd_vel:
        adata.uns['transition_matrix_rnd'] = T_rnd
        adata.obsm['velocity_' + basis + '_rnd'] = delta_X_rnd
        adata.uns['grid_velocity_' + basis + '_rnd'] = {'X_grid': X_grid_rnd, "V_grid": V_grid_rnd, "D": D_rnd}

    return adata


def stationary_distribution(adata, method='kmc', direction='both', calc_rnd=True):
    """Compute stationary distribution of cells using the transition matrix.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        method: `str` (default: `kmc`)
            The method to calculate the stationary distribution.
        direction: `str` (default: `both`)
            The direction of diffusion for calculating the stationary distribution, can be one of `both`, `forward`, `backward`.
        calc_rnd: `bool` (default: `True`)
            Whether to also calculate the stationary distribution from the control randomized transition matrix.
    Returns
    -------
        Adata: :class:`~anndata.AnnData`
            Returns an updated `~anndata.AnnData` with source, sink stationary distributions and the randomized results,
            depending on the direction and calc_rnd arguments.
    """

    T = adata.uns['transition_matrix'] # row is the source and columns are targets

    if method is 'kmc':
        kmc = KernelMarkovChain()
        kmc.P = T
        if direction is 'both':
            adata.obs['sink_steady_state_distribution'] = kmc.compute_stationary_distribution()
            kmc.P = T.T / T.T.sum(0)
            adata.obs['source_steady_state_distribution'] = kmc.compute_stationary_distribution()

            if calc_rnd:
                T_rnd = adata.uns['transition_matrix_rnd']
                kmc.P = T_rnd
                adata.obs['sink_steady_state_distribution_rnd'] = kmc.compute_stationary_distribution()
                kmc.P = T_rnd.T / T_rnd.T.sum(0)
                adata.obs['source_steady_state_distribution_rnd'] = kmc.compute_stationary_distribution()

        elif direction is 'forward':
            adata.obs['sink_steady_state_distribution'] = kmc.compute_stationary_distribution()

            if calc_rnd:
                T_rnd = adata.uns['transition_matrix_rnd']
                kmc.P = T_rnd
                adata.obs['sink_steady_state_distribution_rnd'] = kmc.compute_stationary_distribution()
        elif direction is 'backward':
            kmc.P = T.T / T.T.sum(0)
            adata.obs['source_steady_state_distribution'] = kmc.compute_stationary_distribution()

            if calc_rnd:
                T_rnd = adata.uns['transition_matrix_rnd']
                kmc.P = T_rnd.T / T_rnd.T.sum(0)
                adata.obs['sink_steady_state_distribution_rnd'] = kmc.compute_stationary_distribution()

    else:
        T = T.T
        if direction is 'both':
            adata.obs['source_steady_state_distribution'] = diffusion(T, backward=True)
            adata.obs['sink_steady_state_distribution'] = diffusion(T)
            if calc_rnd:
                T_rnd = adata.uns['transition_matrix_rnd']
                adata.obs['source_steady_state_distribution_rnd'] = diffusion(T_rnd, backward=True)
                adata.obs['sink_steady_state_distribution_rnd'] = diffusion(T_rnd)
        elif direction is 'forward':
            adata.obs['sink_steady_state_distribution'] = diffusion(T)
            if calc_rnd:
                T_rnd = adata.uns['transition_matrix_rnd']
                adata.obs['sink_steady_state_distribution_rnd'] = diffusion(T_rnd)
        elif direction is 'backward':
            adata.obs['source_steady_state_distribution'] = diffusion(T, backward=True)
            if calc_rnd:
                T_rnd = adata.uns['transition_matrix_rnd']
                adata.obs['source_steady_state_distribution_rnd'] = diffusion(T_rnd, backward=True)


def generalized_diffusion_map(adata, **kwargs):
    """Apply the diffusion map algorithm on the transition matrix build from Itô kernel.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the constructed transition matrix.'
        kwargs:
            Additional parameters that will be passed to the diffusion_map_embedding function.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that updated with X_diffusion_map embedding in obsm attribute.
    """

    kmc = KernelMarkovChain()
    kmc.P = adata.uns['transition_matrix']
    dm_args = {"n_dims": 2, "t": 1}
    dm_args.update(kwargs)
    dm = kmc.diffusion_map_embedding(*dm_args)

    adata.obsm['X_diffusion_map'] = dm


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
        M = M / M.sum(1)

    if steps is None:
        # code inspired from  https://github.com/prob140/prob140/blob/master/prob140/markov_chains.py#L284
        from scipy.sparse.linalg import eigs
        eigenvalue, eigenvector = scp.linalg.eig(M, left=True, right=False) # if not issparse(M) else eigs(M) # source is on the row

        eigenvector = np.real(eigenvector) if not issparse(M) else np.real(eigenvector.T)
        eigenvalue_1_ind = np.isclose(eigenvalue, 1)
        mu = eigenvector[:, eigenvalue_1_ind] / np.sum(eigenvector[:, eigenvalue_1_ind])

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


def _empirical_vec(X_pca, X_embedding, V_mat, indices, neg_cells_trick, xy_grid_nums, neighbors):
    """utility function for calculating the transition matrix or low dimensional velocity embedding via the original correlation kernel."""
    n = X_pca.shape[0]
    if indices is not None:
        knn = indices.shape[1] - 1 #remove the first one in kNN
        rows = np.zeros((n * knn, 1))
        cols = np.zeros((n * knn, 1))
        vals = np.zeros((n * knn, 1))

    delta_X = np.zeros((n, X_embedding.shape[1]))
    idx = 0
    for i in range(n):
        i_vals = np.zeros((knn, 1))
        velocity = V_mat[i, :]  # project V_mat to pca space
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
            val_ind_vec = np.array(range(i * knn, (i + 1) * knn))
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

                delta_X[i, :] += 0.5 * (i_prob - 1 / len(cur_ind)).T.dot(
                    numerator / np.hstack((denominator, denominator))).flatten()
        else:
            sigma = max(abs(i_vals))
            exp_i_vals = np.exp(i_vals / sigma)
            denominator = sum(exp_i_vals)
            i_prob = exp_i_vals / denominator
            vals[i * knn:(i + 1) * knn] = i_prob

            j_vec = indices[i, 1:]
            numerator = np.array([X_embedding[j, :] - X_embedding[i, :] for j in j_vec])
            denominator = np.array([[scp.linalg.norm(numerator[j]) for j in range(knn)]]).T

            delta_X[i, :] = (i_prob - 1 / knn).T.dot(numerator / np.hstack((denominator, denominator)))

        X_grid, V_grid, D = velocity_on_grid(X_embedding, X_embedding + delta_X, xy_grid_nums=xy_grid_nums)

    T = csr_matrix((vals.flatten(), (rows.flatten(), cols.flatten())), shape=neighbors.shape)

    return T, delta_X, X_grid, V_grid, D


# utility functions for calculating the random cell velocities
@jit(nopython=True)
def numba_random_seed(seed):
    """Same as np.random.seed but for numba. Function adapated from velocyto.


    Parameters
    ----------
        seed: `int`
            Random seed value

    Returns
    -------
        Nothing but set the random seed in numba.

    """
    np.random.seed(seed)


@jit(nopython=True)
def permute_rows_nsign(A):
    """Permute in place the entries and randomly switch the sign for each row of a matrix independently. Function adapted
    from velocyto

    Parameters
    ----------
        A: `np.array`
            A numpy array that will be permuted.

    Returns
    -------
        Nothing but permute entries and signs for each row of the input matrix in place.
    """

    plmi = np.array([+1, -1])
    for i in range(A.shape[0]):
        np.random.shuffle(A[i, :])
        A[i, :] = A[i, :] * np.random.choice(plmi, size=A.shape[1])
