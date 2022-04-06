# create by Yan Zhang, minor adjusted by Xiaojie Qiu
import numpy as np
import scipy.sparse as sp
from numba import jit
from scipy.linalg import eig, null_space
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from ..dynamo_logger import LoggerManager
from ..simulation.utils import directMethod
from .utils import append_iterative_neighbor_indices, flatten


def markov_combination(x, v, X):
    from cvxopt import matrix, solvers

    n = X.shape[0]
    R = matrix(X - x).T
    H = R.T * R
    f = matrix(v).T * R
    G = np.vstack((-np.eye(n), np.ones(n)))
    h = np.zeros(n + 1)
    h[-1] = 1
    p = solvers.qp(H, -f.T, G=matrix(G), h=matrix(h))["x"]
    u = R * p
    return p, u


def compute_markov_trans_prob(x, v, X, s=None, cont_time=False):
    from cvxopt import matrix, solvers

    n = X.shape[0]
    R = X - x
    # normalize R, v, and s
    Rn = np.array(R, copy=True)
    vn = np.array(v, copy=True)
    scale = np.abs(np.max(R, 0) - np.min(R, 0))
    Rn = Rn / scale
    vn = vn / scale
    if s is not None:
        sn = np.array(s, copy=True)
        sn = sn / scale
        A = np.hstack((Rn, 0.5 * Rn * Rn))
        b = np.hstack((vn, 0.5 * sn * sn))
    else:
        A = Rn
        b = vn

    H = A.dot(A.T)
    f = b.dot(A.T)
    if cont_time:
        G = -np.eye(n)
        h = np.zeros(n)
    else:
        G = np.vstack((-np.eye(n), np.ones(n)))
        h = np.zeros(n + 1)
        h[-1] = 1
    p = solvers.qp(matrix(H), matrix(-f), G=matrix(G), h=matrix(h))["x"]
    p = np.array(p).flatten()
    return p


@jit(nopython=True)
def compute_kernel_trans_prob(x, v, X, inv_s, cont_time=False):
    n = X.shape[0]
    p = np.zeros(n)
    for i in range(n):
        d = X[i] - x
        p[i] = np.exp(-0.25 * (d - v) @ inv_s @ (d - v).T)
    p /= np.sum(p)
    return p


# @jit(nopython=True)
def compute_drift_kernel(x, v, X, inv_s):
    n = X.shape[0]
    k = np.zeros(n)
    for i in range(n):
        d = X[i] - x
        if np.isscalar(inv_s):
            k[i] = np.exp(-0.25 * inv_s * (d - v).dot(d - v))
        else:
            k[i] = np.exp(-0.25 * (d - v) @ inv_s @ (d - v).T)
    return k


"""def compute_drift_local_kernel(x, v, X, inv_s):
    n = X.shape[0]
    k = np.zeros(n)
    # compute tau
    D = X - x
    dists = np.zeros(n)
    vds = np.zeros(n)
    for (i, d) in enumerate(D):
        dists[i] = np.linalg.norm(d)
        if dists[i] > 0:
            vds[i] = v.dot(d) / dists[i]
    i_dir = np.logical_and(vds >= np.quantile(vds, 0.7), vds > 0)
    tau = np.mean(dists[i_dir] / vds[i_dir])
    if np.isnan(tau): tau = 1
    if tau > 1e2: tau = 1e2

    tau_v = tau * v
    tau_invs = (1 / (tau * np.linalg.norm(v))) * inv_s
    for i in range(n):
        d = D[i]
        k[i] = np.exp(-0.25 * (d-tau_v) @ tau_invs @ (d-tau_v).T)
    return k, tau_invs"""


# @jit(nopython=True)
def compute_drift_local_kernel(x, v, X, inv_s):
    n = X.shape[0]
    k = np.zeros(n)
    # compute tau
    D = X - x
    dists = np.zeros(n)
    vds = np.zeros(n)
    for (i, d) in enumerate(D):
        dists[i] = np.linalg.norm(d)
        if dists[i] > 0:
            vds[i] = v.dot(d) / dists[i]
    i_dir = np.logical_and(vds >= np.quantile(vds, 0.7), vds > 0)
    if np.any(i_dir):
        tau = np.mean(dists[i_dir] / vds[i_dir])
        if tau > 1e2:
            tau = 1e2
        tau_v = tau * v
        tau_invs = (1 / (tau * v.dot(v))) * inv_s
    else:
        tau_v = 0
        tau_invs = (1 / (1e2 * v.dot(v))) * inv_s
    for i in range(n):
        d = D[i]
        if np.isscalar(tau_invs):
            k[i] = np.exp(-0.25 * tau_invs * (d - tau_v).dot(d - tau_v))
        else:
            k[i] = np.exp(-0.25 * (d - tau_v) @ tau_invs @ (d - tau_v).T)
    return k


@jit(nopython=True)
def compute_density_kernel(x, X, inv_eps):
    n = X.shape[0]
    k = np.zeros(n)
    for i in range(n):
        d = X[i] - x
        k[i] = np.exp(-0.25 * inv_eps * d.dot(d))
    return k


@jit(nopython=True)
def makeTransitionMatrix(Qnn, I_vec, tol=0.0):  # Qnn, I, tol=0.0
    n = Qnn.shape[0]
    M = np.zeros((n, n))

    for i in range(n):
        q = Qnn[i]
        q[q < tol] = 0
        M[I_vec[i], i] = q
        M[i, i] = 1 - np.sum(q)
    return M


@jit(nopython=True)
def compute_tau(X, V, k=100, nbr_idx=None):
    if nbr_idx is None:
        if X.shape[0] > 200000 and X.shape[1] > 2:
            from pynndescent import NNDescent

            nbrs = NNDescent(
                X,
                metric="euclidean",
                n_neighbors=k,
                n_jobs=-1,
                random_state=19491001,
            )
            _, dist = nbrs.query(X, k=k)
        else:
            alg = "ball_tree" if X.shape[1] > 10 else "kd_tree"
            nbrs = NearestNeighbors(n_neighbors=k, algorithm=alg, n_jobs=-1).fit(X)
            dists, _ = nbrs.kneighbors(X)

    else:
        dists = np.zeros(nbr_idx.shape)
        for i in range(nbr_idx.shape[0]):
            for j in range(nbr_idx.shape[1]):
                x = X[i]
                y = X[nbr_idx[i, j]]
                dists[i, j] = np.sqrt((x - y).dot(x - y))
    d = np.mean(dists[:, 1:], 1)
    v = np.linalg.norm(V, axis=1)
    tau = d / v
    return tau, v


def prepare_velocity_grid_data(
    X_emb,
    xy_grid_nums,
    density=None,
    smooth=None,
    n_neighbors=None,
):
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs, scale = [], 0
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, xy_grid_nums[dim_i] * density)
        scale += gr[1] - gr[0]
        grs.append(gr)

    scale = scale / n_dim * smooth

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # estimate grid velocities
    if n_neighbors is None:
        n_neighbors = np.max([10, int(n_obs / 50)])

    if X_emb.shape[0] > 200000 and X_emb.shape[1] > 2:
        from pynndescent import NNDescent

        nn = NNDescent(
            X_emb,
            metric="euclidean",
            n_neighbors=n_neighbors,
            n_jobs=-1,
            random_state=19491001,
        )
        neighs, dists = nn.query(X_grid, k=n_neighbors)
    else:
        alg = "ball_tree" if X_emb.shape[1] > 10 else "kd_tree"
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1, algorithm=alg)
        nn.fit(X_emb)
        dists, neighs = nn.kneighbors(X_grid)

    weight = norm.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    return X_grid, p_mass, neighs, weight


def grid_velocity_filter(
    V_emb,
    neighs,
    p_mass,
    X_grid,
    V_grid,
    min_mass=None,
    autoscale=False,
    adjust_for_stream=True,
    V_threshold=None,
):
    if adjust_for_stream:
        X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(np.sqrt(V_grid.shape[0]))
        V_grid = V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((V_grid ** 2).sum(0))
        if V_threshold is not None:
            V_grid[0][mass.reshape(V_grid[0].shape) < V_threshold] = np.nan
        else:
            if min_mass is None:
                min_mass = 1e-5
            min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)
            cutoff = mass.reshape(V_grid[0].shape) < min_mass

            if neighs is not None:
                length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T.reshape(ns, ns)
                cutoff |= length < np.percentile(length, 5)

            V_grid[0][cutoff] = np.nan
    else:
        from ..plot.utils import quiver_autoscaler

        if p_mass is None:
            p_mass = np.sqrt((V_grid ** 2).sum(1))
            if min_mass is None:
                min_mass = np.clip(np.percentile(p_mass, 5), 1e-5, None)
        else:
            if min_mass is None:
                min_mass = np.clip(np.percentile(p_mass, 99) / 100, 1e-5, None)
        X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]

        if autoscale:
            V_grid /= 3 * quiver_autoscaler(X_grid, V_grid)

    return X_grid, V_grid


def velocity_on_grid(
    X_emb,
    V_emb,
    xy_grid_nums,
    density=None,
    smooth=None,
    n_neighbors=None,
    min_mass=None,
    autoscale=False,
    adjust_for_stream=True,
    V_threshold=None,
    cut_off_velocity=True,
):
    """Function to calculate the velocity vectors on a grid for grid vector field quiver plot and streamplot, adapted
    from scVelo"""
    from ..vectorfield.stochastic_process import diffusionMatrix2D

    valid_idx = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb, V_emb = X_emb[valid_idx], V_emb[valid_idx]

    X_grid, p_mass, neighs, weight = prepare_velocity_grid_data(
        X_emb,
        xy_grid_nums,
        density=density,
        smooth=smooth,
        n_neighbors=n_neighbors,
    )

    # V_grid = (V_emb[neighs] * (weight / p_mass[:, None])[:, :, None]).sum(1) # / np.maximum(1, p_mass)[:, None]
    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1) / np.maximum(1, p_mass)[:, None]
    # calculate diffusion matrix D
    D = diffusionMatrix2D(V_emb[neighs])

    if cut_off_velocity:
        X_grid, V_grid = grid_velocity_filter(
            V_emb,
            neighs,
            p_mass,
            X_grid,
            V_grid,
            min_mass=min_mass,
            autoscale=autoscale,
            adjust_for_stream=adjust_for_stream,
            V_threshold=V_threshold,
        )
    else:
        X_grid, V_grid = (
            np.array([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])]),
            np.array(
                [
                    V_grid[:, 0].reshape(xy_grid_nums),
                    V_grid[:, 1].reshape(xy_grid_nums),
                ]
            ),
        )

    return X_grid, V_grid, D


# we might need a separate module/file for discrete vector field and markovian methods in the future
def graphize_velocity(V, X, nbrs_idx=None, k=30, normalize_v=False, E_func=None):
    """
        The function generates a graph based on the velocity data. The flow from i- to j-th
        node is returned as the edge matrix E[i, j], and E[i, j] = -E[j, i].

    Arguments
    ---------
        V: :class:`~numpy.ndarray`
            The velocities for all cells.
        X: :class:`~numpy.ndarray`
            The coordinates for all cells.
        nbrs_idx: list (optional, default None)
            a list of neighbor indices for each cell. If None a KNN will be performed instead.
        k: int (optional, default 30)
            The number of neighbors for the KNN search.
        normalize_v: bool (optional, default False)
            Whether or not normalizing the velocity vectors.
        E_func: str, function, or None (optional, default None)
            A variance stabilizing function for reducing the variance of the flows.
            If a string is passed, there are two options:
                'sqrt': the numpy.sqrt square root function;
                'exp': the numpy.exp exponential function.

    Returns
    -------
        E: :class:`~numpy.ndarray`
            The edge matrix.
        nbrs_idx: list
            Neighbor indices.
    """
    n, d = X.shape

    nbrs = None
    if nbrs_idx is None:
        if n > 200000 and d > 2:
            from pynndescent import NNDescent

            nbrs = NNDescent(
                X,
                metric="euclidean",
                n_neighbors=k + 1,
                n_jobs=-1,
                random_state=19491001,
            )
            nbrs_idx, _ = nbrs.query(X, k=k + 1)
        else:
            alg = "ball_tree" if d > 10 else "kd_tree"
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm=alg, n_jobs=-1).fit(X)
            _, nbrs_idx = nbrs.kneighbors(X)

    if type(E_func) is str:
        if E_func == "sqrt":
            E_func = np.sqrt
        elif E_func == "exp":
            E_func = np.exp
        else:
            raise NotImplementedError("The specified edge function is not implemented.")

    # E = sp.csr_matrix((n, n))      # Making E a csr_matrix will slow down this process. Try lil_matrix maybe?
    E = np.zeros((n, n))
    for i in range(n):
        x = flatten(X[i])
        idx = nbrs_idx[i]
        if len(idx) > 0 and idx[0] == i:  # excluding the node itself from the neighbors
            idx = idx[1:]
        vi = flatten(V[i])
        if normalize_v:
            vi_norm = np.linalg.norm(vi)
            if vi_norm > 0:
                vi /= vi_norm

        # normalized differences
        U = X[idx] - x
        U_norm = np.linalg.norm(U, axis=1)
        U_norm[U_norm == 0] = 1
        U /= U_norm[:, None]

        for jj, j in enumerate(idx):
            vj = flatten(V[j])
            if normalize_v:
                vj_norm = np.linalg.norm(vj)
                if vj_norm > 0:
                    vj /= vj_norm
            u = flatten(U[jj])
            v = np.mean((vi.dot(u), vj.dot(u)))

            if E_func is not None:
                v = np.sign(v) * E_func(np.abs(v))
            E[i, j] = v
            E[j, i] = -v

    return E, nbrs_idx


def calc_Laplacian(E, convention="graph"):
    A = np.abs(np.sign(E))
    L = np.diag(np.sum(A, 0)) - A

    if convention == "diffusion":
        L = -L

    return L


def fp_operator(E, D):
    # drift
    Mu = E.T.copy()
    Mu[Mu < 0] = 0
    Mu = np.diag(np.sum(Mu, 0)) - Mu
    # diffusion
    L = calc_Laplacian(E, convention="diffusion")

    return -Mu + D * L


def divergence(E, tol=1e-5):
    n = E.shape[0]
    div = np.zeros(n)
    # optimize for sparse matrices later...
    for i in range(n):
        for j in range(i + 1, n):
            if np.abs(E[i, j]) > tol:
                div[i] += E[i, j] - E[j, i]

    return div


class MarkovChain:
    def __init__(self, P=None, eignum=None, check_norm=True, sumto=1, tol=1e-3):
        if check_norm and not self.is_normalized(P, axis=0, sumto=sumto, tol=tol):
            if self.is_normalized(P, axis=1, sumto=sumto, tol=tol):
                LoggerManager.main_logger.info(
                    f"Column sums of the input matrix are not {sumto} but row sums are. "
                    "Transposing the transition matrix"
                )
                P = P.T
            else:
                raise Exception(f"Neither the row nor the column sums of the input matrix are {sumto}.")
        self.P = P
        self.D = None  # eigenvalues
        self.U = None  # left eigenvectors
        self.W = None  # right eigenvectors
        self.W_inv = None
        self.eignum = eignum  # if None all eigs are solved

    def eigsys(self, eignum=None):
        if eignum is not None:
            self.eignum = eignum
        if self.eignum is None:
            D, U, W = eig(self.P, left=True, right=True)
            idx = D.argsort()[::-1]
            self.D = D[idx]
            self.U = U[:, idx]
            self.W = W[:, idx]
            try:
                self.W_inv = np.linalg.inv(self.W)
            except np.linalg.LinAlgError:
                self.W_inv = np.linalg.pinv(self.W)
        else:
            self.D, self.W = sp.linalg.eigs(self.P, k=self.eignum, which="LR")
            self.U = self.right_eigvecs_to_left(self.W, self.W[:, 0])
            self.W_inv = self.U.T.copy()
            self.U /= np.linalg.norm(self.U, axis=0)

    def right_eigvecs_to_left(self, W, p_st):
        U = np.diag(1 / np.abs(p_st)) @ W
        f = np.mean(np.diag(U.T @ U))
        return U / np.sqrt(f)

    def get_num_states(self):
        return self.P.shape[0]

    def is_normalized(self, P=None, tol=1e-3, sumto=1, axis=0, ignore_nan=True):
        """
        check if the matrix is properly normalized up to `tol`.

        Parameters
        ----------
            P: None or :class:`~numpy.ndarray` (default `None`)
                The transition matrix. If None, self.P is checked instead.
            tol: float (default 1e-3)
                The numerical tolerance.
            sumto: int (default: 1)
                The value that each column/row should sum to.
            axis: int (default: 0)
                0 - check if the matrix is column normalized;
                1 - check if the matrix is row normalized.
        """
        P = self.P if P is None else P
        sumfunc = np.sum if not ignore_nan else np.nansum
        return np.all(np.abs(sumfunc(P, axis=axis) - sumto) < tol)

    def __reset__(self):
        self.D = None
        self.U = None
        self.W = None
        self.W_inv = None


class KernelMarkovChain(MarkovChain):
    def __init__(self, P=None, Idx=None, n_recurse_neighbors=None):
        super().__init__(P)
        self.Kd = None
        if n_recurse_neighbors is not None and Idx is not None:
            self.Idx = append_iterative_neighbor_indices(Idx, n_recurse_neighbors)
        else:
            self.Idx = Idx

    def fit(
        self,
        X,
        V,
        M_diff,
        neighbor_idx=None,
        n_recurse_neighbors=None,
        k=30,
        epsilon=None,
        adaptive_local_kernel=False,
        tol=1e-4,
        sparse_construct=True,
        sample_fraction=None,
    ):
        # compute connectivity
        if neighbor_idx is None:
            if X.shape[0] > 200000 and X.shape[1] > 2:
                from pynndescent import NNDescent

                nbrs = NNDescent(
                    X,
                    metric="euclidean",
                    n_neighbors=k,
                    n_jobs=-1,
                    random_state=19491001,
                )
                neighbor_idx, _ = nbrs.query(X, k=k)
            else:
                alg = "ball_tree" if X.shape[1] > 10 else "kd_tree"
                nbrs = NearestNeighbors(n_neighbors=k, algorithm=alg, n_jobs=-1).fit(X)
                _, neighbor_idx = nbrs.kneighbors(X)

        if n_recurse_neighbors is not None:
            self.Idx = append_iterative_neighbor_indices(neighbor_idx, n_recurse_neighbors)
        else:
            self.Idx = neighbor_idx

        # apply kNN downsampling to accelerate calculation (adapted from velocyto)
        if sample_fraction is not None:
            neighbor_idx = self.Idx
            p = np.linspace(0.5, 1, neighbor_idx.shape[1])
            p = p / p.sum()

            sampling_ixs = np.stack(
                (
                    np.random.choice(
                        np.arange(1, neighbor_idx.shape[1] - 1),
                        size=int(sample_fraction * (neighbor_idx.shape[1] + 1)),
                        replace=False,
                        p=p,
                    )
                    for i in range(neighbor_idx.shape[0])
                ),
                0,
            )
            self.Idx = self.Idx[np.arange(neighbor_idx.shape[0])[:, None], sampling_ixs]

        n = X.shape[0]
        if sparse_construct:
            self.P = sp.lil_matrix((n, n))
        else:
            self.P = np.zeros((n, n))

        # compute density kernel
        if epsilon is not None:
            if sparse_construct:
                self.Kd = sp.lil_matrix((n, n))
            else:
                self.Kd = np.zeros((n, n))
            inv_eps = 1 / epsilon
            for i in range(n):
                self.Kd[i, self.Idx[i]] = compute_density_kernel(X[i], X[self.Idx[i]], inv_eps)
            self.Kd = sp.csc_matrix(self.Kd)
            D = np.sum(self.Kd, 0)

        # compute transition prob.
        if np.isscalar(M_diff):
            inv_s = 1 / M_diff
        else:
            inv_s = np.linalg.inv(M_diff)
        for i in tqdm(range(n), desc="compute transiton matrix"):
            y = X[i]
            v = V[i]
            Y = X[self.Idx[i]]
            if adaptive_local_kernel:
                k = compute_drift_local_kernel(y, v, Y, inv_s)
            else:
                k = compute_drift_kernel(y, v, Y, inv_s)
            if epsilon is not None:
                k = k / D[0, self.Idx[i]]
            else:
                k = np.matrix(k)
            p = k / np.sum(k) if np.sum(k) > 0 else np.ones_like(k) / n
            p[p <= tol] = 0  # tolerance check
            p = p / np.sum(p)
            self.P[self.Idx[i], i] = p.A[0]

        self.P = sp.csc_matrix(self.P)

    def propagate_P(self, num_prop):
        ret = sp.csc_matrix(self.P, copy=True)
        for i in range(num_prop - 1):
            ret = self.P * ret  # sparse matrix (ret) is a `np.matrix`
        return ret

    def compute_drift(self, X, num_prop=1, scale=True):
        n = self.get_num_states()
        V = np.zeros_like(X)
        P = self.propagate_P(int(num_prop))
        for i in tqdm(range(n), desc="compute drift"):
            V[i] = (X - X[i]).T.dot(P[:, i].A.flatten())
        return V * 1 / V.max() if scale else V

    def compute_density_corrected_drift(
        self,
        X,
        neighbor_idx=None,
        k=None,
        num_prop=1,
        normalize_vector=False,
        correct_by_mean=True,
        scale=True,
    ):
        n = self.get_num_states()
        V = np.zeros_like(X)
        P = self.propagate_P(num_prop)
        if neighbor_idx is None:
            neighbor_idx = self.Idx
        for i in tqdm(range(n), desc="calculating density corrected drift"):
            Idx = neighbor_idx[i]
            D = X[Idx] - X[i]
            if normalize_vector:
                D = D / np.linalg.norm(D, 1)
            p = P[Idx, i].A.flatten()
            if k is None:
                if not correct_by_mean:
                    k_inv = 1 / len(Idx)
                else:
                    k_inv = np.mean(p)
            else:
                k_inv = 1 / k
            p -= k_inv
            V[i] = D.T.dot(p)
        return V * 1 / V.max() if scale else V

    def compute_stationary_distribution(self):
        # if self.W is None:
        # self.eigsys()
        _, vecs = sp.linalg.eigs(self.P, k=1, which="LR")
        p = np.abs(np.real(vecs[:, 0]))
        p = p / np.sum(p)
        return p

    def diffusion_map_embedding(self, n_dims=2, t=1):
        # if self.W is None:
        #    self.eigsys()
        vals, vecs = sp.linalg.eigs(self.P.T, k=n_dims + 1, which="LR")
        ind = np.range(1, n_dims + 1)
        Y = np.real(vals[ind] ** t) * np.real(vecs[:, ind])
        return Y

    def compute_theta(self, p_st=None):
        p_st = self.compute_stationary_distribution() if p_st is None else p_st
        Pi = sp.csr_matrix(np.diag(np.sqrt(p_st)))
        Pi_right = sp.csc_matrix(np.diag(np.sqrt(p_st)))
        # Pi_inv = sp.csr_matrix(np.linalg.pinv(Pi))
        # Pi_inv_right = sp.csc_matrix(np.linalg.pinv(Pi))
        Pi_inv = sp.csr_matrix(np.diag(np.sqrt(1 / p_st)))
        Pi_inv_right = sp.csc_matrix(np.diag(np.sqrt(1 / p_st)))
        Theta = 0.5 * (Pi @ self.P @ Pi_inv_right + Pi_inv @ self.P.T @ Pi_right)

        return Theta


class DiscreteTimeMarkovChain(MarkovChain):
    def __init__(self, P=None, eignum=None, sumto=1, **kwargs):
        super().__init__(P, eignum=eignum, sumto=sumto, **kwargs)
        # self.Kd = None

    """def fit(self, X, V, k, s=None, method="qp", eps=None, tol=1e-4):  # pass index
        # the parameter k will be replaced by a connectivity matrix in the future.
        self.__reset__()
        # knn clustering
        if X.shape[0] > 200000 and X.shape[1] > 2:
            from pynndescent import NNDescent

            nbrs = NNDescent(
                X,
                metric="euclidean",
                n_neighbors=k,
                n_jobs=-1,
                random_state=19491001,
            )
            Idx, _ = nbrs.query(X, k=k)
        else:
            alg = "ball_tree" if X.shape[1] > 10 else "kd_tree"
            nbrs = NearestNeighbors(n_neighbors=k, algorithm=alg, n_jobs=-1).fit(X)
            _, Idx = nbrs.kneighbors(X)
        # compute transition prob.
        n = X.shape[0]
        self.P = np.zeros((n, n))
        if method == "kernel":
            inv_s = np.linalg.inv(s)
            # compute density kernel
            if eps is not None:
                self.Kd = np.zeros((n, n))
                inv_eps = 1 / eps
                for i in range(n):
                    self.Kd[i, Idx[i]] = compute_density_kernel(X[i], X[Idx[i]], inv_eps)
                D = np.sum(self.Kd, 0)
        for i in range(n):
            y = X[i]
            v = V[i]
            if method == "qp":
                Y = X[Idx[i, 1:]]
                p = compute_markov_trans_prob(y, v, Y, s)
                p[p <= tol] = 0  # tolerance check
                self.P[Idx[i, 1:], i] = p
                self.P[i, i] = 1 - np.sum(p)
            else:
                Y = X[Idx[i]]
                # p = compute_kernel_trans_prob(y, v, Y, inv_s)
                k = compute_drift_kernel(y, v, Y, inv_s)
                if eps is not None:
                    k /= D[Idx[i]]
                p = k / np.sum(k)
                p[p <= tol] = 0  # tolerance check
                p = p / np.sum(p)
                self.P[Idx[i], i] = p"""

    def propagate_P(self, num_prop):
        ret = np.array(self.P, copy=True)
        for _ in range(num_prop - 1):
            ret = self.P @ ret
        return ret

    def compute_drift(self, X, num_prop=1):
        n = self.get_num_states()
        V = np.zeros_like(X)
        P = self.propagate_P(num_prop)
        for i in range(n):
            V[i] = (X - X[i]).T.dot(P[:, i])
        return V

    def compute_density_corrected_drift(self, X, k=None, normalize_vector=False):
        n = self.get_num_states()
        if k is None:
            k = n
        V = np.zeros_like(X)
        for i in range(n):
            d = X - X[i]  # no self.nbrs_idx[i] is here.... may be wrong?
            if normalize_vector:
                d /= np.linalg.norm(d)
            V[i] = d.T.dot(self.P[:, i] - 1 / k)
        return V

    def solve_distribution(self, p0, n, method="naive"):
        if method == "naive":
            p = p0
            for _ in range(n):
                p = self.P.dot(p)
        else:
            if self.D is None:
                self.eigsys()
            p = np.real(self.W @ np.diag(self.D ** n) @ np.linalg.inv(self.W)).dot(p0)
        return p

    def compute_stationary_distribution(self, method="eig"):
        if method == "solve":
            p = np.real(null_space(self.P - np.eye(self.P.shape[0])[:, 0]).flatten())
        else:
            if self.W is None:
                self.eigsys()
            p = np.abs(np.real(self.W[:, 0]))
        p = p / np.sum(p)
        return p

    def lump(self, labels, M_weight=None):
        """
        Markov chain lumping based on:
        K. Hoffmanna and P. Salamon, Bounding the lumping error in Markov chain dynamics, Appl Math Lett, (2009)
        """
        k = len(labels)
        M_part = np.zeros((k, self.get_num_states()))

        for i in range(len(labels)):
            M_part[labels[i], i] = 1

        n_node = self.get_num_states()
        if M_weight is None:
            p_st = self.compute_stationary_distribution()
            M_weight = np.multiply(M_part, p_st)
            M_weight = np.divide(M_weight.T, M_weight @ np.ones(n_node))
        P_lumped = M_part @ self.P @ M_weight

        return P_lumped

    def naive_lump(self, x, grp):
        k = len(np.unique(grp))
        y = np.zeros((k, k))
        for i in range(len(y)):
            for j in range(len(y)):
                y[i, j] = x[grp == i, :][:, grp == j].mean()

        return y

    def diffusion_map_embedding(self, n_dims=2, t=1):
        if self.W is None:
            self.eigsys()

        ind = np.arange(1, n_dims + 1)
        Y = np.real(self.D[ind] ** t) * np.real(self.U[:, ind])
        return Y


class ContinuousTimeMarkovChain(MarkovChain):
    def __init__(self, P=None, eignum=None, **kwargs):
        super().__init__(P, eignum=eignum, sumto=0, **kwargs)
        self.Q = None  # embedded markov chain transition matrix
        self.Kd = None
        self.p_st = None

    def check_transition_rate_matrix(self, P, tol=1e-6):
        if np.any(flatten(np.abs(np.sum(P, 0))) <= tol):
            return P
        elif np.any(flatten(np.abs(np.sum(P, 1))) <= tol):
            return P.T
        else:
            raise ValueError("The input transition rate matrix must have a zero row- or column-sum.")

    """def fit(self, X, V, k, s=None, tol=1e-4):
        self.__reset__()
        # knn clustering
        if self.nbrs_idx is None:
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
            _, Idx = nbrs.kneighbors(X)
            self.nbrs_idx = Idx
        else:
            Idx = self.nbrs_idx
        # compute transition prob.
        n = X.shape[0]
        self.P = np.zeros((n, n))
        for i in range(n):
            y = X[i]
            v = V[i]
            Y = X[Idx[i, 1:]]
            p = compute_markov_trans_prob(y, v, Y, s, cont_time=True)
            p[p<=tol] = 0       # tolerance check
            self.P[Idx[i, 1:], i] = p
            self.P[i, i] = - np.sum(p)"""

    def compute_drift(self, X, t, n_top=5, normalize_vector=False):
        n = self.get_num_states()
        V = np.zeros_like(X)
        P = self.compute_transition_matrix(t)

        for i in range(n):
            if n_top is None:
                d = (X - X[i]).T
                if normalize_vector:
                    d = d / np.linalg.norm(d, axis=0)
                V[i] = d.dot(P[:, i])
            else:
                idx = np.argsort(P[:, i])[-n_top:]
                d = (X[idx] - X[i]).T
                if normalize_vector:
                    d = d / np.linalg.norm(d, axis=0)
                V[i] = d.dot(P[idx, i])
                # q = P[idx, i] / np.sum(P[idx, i])
                # V[i] = d.dot(q)
        return V

    def compute_density_corrected_drift(self, X, t, k=None, normalize_vector=False):
        n = self.get_num_states()
        V = np.zeros_like(X)
        P = self.compute_transition_matrix(t)
        for i in range(n):
            P[i, i] = 0
            P[:, i] /= np.sum(P[:, i])
            d = X - X[i]
            if normalize_vector:
                d /= np.linalg.norm(d)
            correction = 1 / k if k is not None else np.mean(P[:, i])
            V[i] = d.T.dot(P[:, i] - correction)
        return V

    def compute_transition_matrix(self, t):
        if self.D is None:
            self.eigsys()
        P = np.real(self.W @ np.diag(np.exp(self.D * t)) @ self.W_inv)
        return P

    def compute_embedded_transition_matrix(self):
        self.Q = np.array(self.P, copy=True)
        for i in range(self.Q.shape[1]):
            self.Q[i, i] = 0
            self.Q[:, i] /= np.sum(self.Q[:, i])
        return self.Q

    def solve_distribution(self, p0, t):
        P = self.compute_transition_matrix(t)
        p = P @ p0
        p = p / np.sum(p)
        return p

    def compute_stationary_distribution(self, method="eig"):
        if self.p_st is None:
            if method == "null":
                p = np.abs(np.real(null_space(self.P)[:, 0].flatten()))
                p = p / np.sum(p)
                self.p_st = p
            else:
                if self.D is None:
                    self.eigsys()
                p = np.abs(np.real(self.W[:, 0]))
                p = p / np.sum(p)
                self.p_st = p
        return self.p_st

    def simulate_random_walk(self, init_idx, tspan):
        P = self.P.copy()

        def prop_func(c):
            a = P[:, c[0]]
            a[c[0]] = 0
            return a

        def update_func(c, mu):
            return np.array([mu])

        T, C = directMethod(prop_func, update_func, tspan=tspan, C0=np.array([init_idx]))

        return C, T

    def compute_hitting_time(self, p_st=None, return_Z=False):
        p_st = self.compute_stationary_distribution() if p_st is None else p_st
        n_nodes = len(p_st)
        W = np.ones((n_nodes, 1)) * p_st
        Z = np.linalg.inv(-self.P.T + W)
        H = np.ones((n_nodes, 1)) * np.diag(Z).T - Z
        H = H / W
        H = H.T
        if return_Z:
            return H, Z
        else:
            return H

    def diffusion_map_embedding(self, n_dims=2, t=1, n_pca_dims=None):
        if self.D is None:
            self.eigsys()

        ind = np.arange(1, n_dims + 1)
        Y = np.real(np.exp(self.D[ind] ** t)) * np.real(self.U[:, ind])
        if n_pca_dims is not None:
            pca = PCA(n_components=n_pca_dims)
            Y = pca.fit_transform(Y)
        return Y
