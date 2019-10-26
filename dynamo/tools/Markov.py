import numpy as np
import scipy.sparse as sp
from cvxopt import matrix, solvers
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import eig, null_space
from numba import jit


def markov_combination(x, v, X):
    n = X.shape[0]
    R = matrix(X - x).T
    H = R.T * R
    f = matrix(v).T * R
    G = np.vstack((-np.eye(n),
                   np.ones(n)))
    h = np.zeros(n + 1)
    h[-1] = 1
    p = solvers.qp(H, -f.T, G=matrix(G), h=matrix(h))['x']
    u = R * p
    return p, u


def compute_markov_trans_prob(x, v, X, s=None, cont_time=False):
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
        G = np.vstack((-np.eye(n),
                       np.ones(n)))
        h = np.zeros(n + 1)
        h[-1] = 1
    p = solvers.qp(matrix(H), matrix(-f), G=matrix(G), h=matrix(h))['x']
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


@jit(nopython=True)
def compute_drift_kernel(x, v, X, inv_s):
    n = X.shape[0]
    k = np.zeros(n)
    for i in range(n):
        d = X[i] - x
        k[i] = np.exp(-0.25 * (d - v) @ inv_s @ (d - v).T)
    return k


'''def compute_drift_local_kernel(x, v, X, inv_s):
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
    return k, tau_invs'''


@jit(nopython=True)
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
        if tau > 1e2: tau = 1e2
        tau_v = tau * v
        tau_invs = (1 / (tau * np.linalg.norm(v))) * inv_s
    else:
        tau_v = 0
        tau_invs = (1 / (1e2 * np.linalg.norm(v))) * inv_s
    for i in range(n):
        d = D[i]
        k[i] = np.exp(-0.25 * (d - tau_v) @ tau_invs @ (d - tau_v).T)
    return k, tau_invs


@jit(nopython=True)
def compute_density_kernel(x, X, inv_eps):
    n = X.shape[0]
    k = np.zeros(n)
    for i in range(n):
        d = X[i] - x
        k[i] = np.exp(-0.25 * inv_eps * d.dot(d))
    return k


@jit(nopython=True)
def makeTransitionMatrix(Qnn, I, tol=0.):
    n = Qnn.shape[0]
    M = np.zeros((n, n))

    for i in range(n):
        q = Qnn[i]
        q[q < tol] = 0
        M[I[i], i] = q
        M[i, i] = 1 - np.sum(q)
    return M


@jit(nopython=True)
def compute_tau(X, V, k=100, nbr_idx=None):
    if nbr_idx is None:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
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


@jit(nopython=True)
def smoothen_drift_on_grid(X, V, n_grid, nbrs=None, k=None, smoothness=1):
    # These codes are borrowed from velocyto. Need to be rewritten later.
    # Prepare the grid
    grs = []
    for dim_i in range(X.shape[1]):
        m, M = np.min(X[:, dim_i]), np.max(X[:, dim_i])
        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        gr = np.linspace(m, M, n_grid)
        grs.append(gr)
    meshes_tuple = np.meshgrid(*grs)
    gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T

    if nbrs is None:
        if k is None: k = 100
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    dists, neighs = nbrs.kneighbors(gridpoints_coordinates)

    from scipy.stats import norm as normal
    std = np.mean([(g[1] - g[0]) for g in grs])
    # isotropic gaussian kernel
    gaussian_w = normal.pdf(loc=0, scale=smoothness * std, x=dists[:, :k])
    total_p_mass = gaussian_w.sum(1)

    U = (V[neighs[:, :k]] * gaussian_w[:, :, None]).sum(1) / np.maximum(1, total_p_mass)[:, None]  # weighed average
    return U, gridpoints_coordinates


class MarkovChain:
    def __init__(self, P=None):
        self.P = P
        self.D = None  # eigenvalues
        self.U = None  # left eigenvectors
        self.W = None  # right eigenvectors

    def eigsys(self):
        D, U, W = eig(self.P, left=True, right=True)
        idx = D.argsort()[::-1]
        self.D = D[idx]
        self.U = U[:, idx]
        self.W = W[:, idx]

    def get_num_states(self):
        return self.P.shape[0]

    def __reset__(self):
        self.D = None
        self.U = None
        self.W = None


class KernelMarkovChain(MarkovChain):
    def __init__(self, P=None):
        super().__init__(P)
        self.Kd = None
        self.Idx = None

    def fit(self, X, V, M_diff, neighbor_idx=None, k=200, epsilon=None, adaptive_local_kernel=False, tol=1e-4,
            sparse_construct=True):
        # compute connectivity
        if neighbor_idx is None:
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
            _, self.Idx = nbrs.kneighbors(X)
        else:
            self.Idx = neighbor_idx
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
        inv_s = np.linalg.inv(M_diff)
        for i in range(n):
            y = X[i]
            v = V[i]
            Y = X[self.Idx[i]]
            if adaptive_local_kernel:
                k, tau = compute_drift_local_kernel(y, v, Y, inv_s)
            else:
                k = compute_drift_kernel(y, v, Y, inv_s)
            if epsilon is not None:
                k = k / D[0, self.Idx[i]]
            else:
                k = np.matrix(k)
            if np.sum(k) == 0:
                print(i)
                print(tau)
            p = k / np.sum(k)
            p[p <= tol] = 0  # tolerance check
            p = p / np.sum(p)
            self.P[self.Idx[i], i] = p.A[0] ###################why p.A[0] here? ###########################

        self.P = sp.csc_matrix(self.P)

    def propagate_P(self, num_prop):
        ret = sp.csc_matrix(self.P, copy=True)
        for i in range(num_prop - 1):
            ret = self.P * ret
        return ret

    def compute_drift(self, X, num_prop=1):
        n = self.get_num_states()
        V = np.zeros_like(X)
        P = self.propagate_P(num_prop)
        for i in range(n):
            V[i] = (X - X[i]).T.dot(P[:, i].A.flatten())
        return V

    def compute_density_corrected_drift(self, X, neighbor_idx, k=None, num_prop=1, normalize_vector=False):
        n = self.get_num_states()
        V = np.zeros_like(X)
        P = self.propagate_P(num_prop)
        if k is None:
            k = neighbor_idx.shape[1]
        for i in range(n):
            Idx = neighbor_idx[i]
            D = X[Idx] - X[i]
            if normalize_vector:
                D = D / np.linalg.norm(D, 1)
            p = P[Idx, i].A.flatten() - 1 / k
            V[i] = D.T.dot(p)
        return V

    def compute_stationary_distribution(self):
        # if self.W is None:
        # self.eigsys()
        _, vecs = sp.linalg.eigs(self.P, k=1, which='LR')
        p = np.abs(np.real(vecs[:, 0]))
        p = p / np.sum(p)
        return p

    def diffusion_map_embedding(self, n_dims=2, t=1):
        # if self.W is None:
        #    self.eigsys()
        vals, vecs = sp.linalg.eigs(self.P.T, k=n_dims + 1, which='LR')
        Y = np.real(vals[1:n_dims + 1] ** t) * np.real(vecs[:, 1:n_dims + 1])
        return Y


class DiscreteTimeMarkovChain(MarkovChain):
    def __init__(self, P=None):
        super().__init__(P)
        self.Kd = None

    def fit(self, X, V, k, s=None, method='qp', eps=None, tol=1e-4):
        # the parameter k will be replaced by a connectivity matrix in the future.
        self.__reset__()
        # knn clustering
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
        _, Idx = nbrs.kneighbors(X)
        # compute transition prob.
        n = X.shape[0]
        self.P = np.zeros((n, n))
        if method == 'kernel':
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
            if method == 'qp':
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
                self.P[Idx[i], i] = p

    def propagate_P(self, num_prop):
        ret = np.array(self.P, copy=True)
        for i in range(num_prop - 1):
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
            d = X - X[i] ###############################no self.nbrs_idx[i] is here.... may be wrong?
            if normalize_vector:
                d /= np.linalg.norm(d)
            V[i] = d.T.dot(self.P[:, i] - 1 / k)
        return V

    def solve_distribution(self, p0, n, method='naive'):
        if method == 'naive':
            p = p0
            for _ in range(n):
                p = self.P.dot(p)
        else:
            if self.D is None:
                self.eigsys()
            p = np.real(self.W @ np.diag(self.D ** n) @ np.linalg.inv(self.W)).dot(p0)
        return p

    def compute_stationary_distribution(self, method='eig'):
        if method == 'solve':
            p = np.real(null_space(self.P - np.eye(self.P.shape[0])[:, 0]).flatten())
        else:
            if self.W is None:
                self.eigsys()
            p = np.abs(np.real(self.W[:, 0]))
        p = p / np.sum(p)
        return p

    def diffusion_map_embedding(self, n_dims=2, t=1):
        if self.W is None:
            self.eigsys()
        Y = np.real(self.D[1:n_dims + 1] ** t) * np.real(self.U[:, 1:n_dims + 1])
        return Y


class ContinuousTimeMarkovChain(MarkovChain):
    def __init__(self, P=None, nbrs_idx=None):
        super().__init__(P)
        self.Kd = None
        self.nbrs_idx = nbrs_idx

    def fit(self, X, V, k, s=None, tol=1e-4):
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
            p[p <= tol] = 0  # tolerance check
            self.P[Idx[i, 1:], i] = p
            self.P[i, i] = - np.sum(p)

    def compute_drift(self, X):
        n = self.get_num_states()
        V = np.zeros_like(X)
        for i in range(n):
            V[i] = (X - X[i]).T.dot(self.P[:, i])
        return V

    def compute_density_corrected_drift(self, X, k=None, normalize_vector=False):
        n = self.get_num_states()
        if k is None:
            k = n
        V = np.zeros_like(X)
        for i in range(n):
            d = X[self.nbrs_idx[i]] - X[i]
            if normalize_vector:
                d /= np.linalg.norm(d)
            V[i] = d.T.dot(self.P[self.nbrs_idx[i], i] - 1 / k)
        return V

    def solve_distribution(self, p0, t):
        if self.D is None:
            self.eigsys()
        p = np.real(self.W @ np.diag(np.exp(self.D * t)) @ np.linalg.inv(self.W)).dot(p0)
        return p

    def compute_stationary_distribution(self):
        p = np.real(null_space(self.P)[:, 0].flatten())
        p = p / np.sum(p)
        return p
