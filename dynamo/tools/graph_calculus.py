import numpy as np
import scipy.sparse as sp
from .utils import k_nearest_neighbors, nbrs_to_dists, flatten, index_condensed_matrix


def graphize_velocity(V, X, nbrs_idx=None, dists=None, k=30, normalize_v=False, E_func=None, use_sparse=False):
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
    n = X.shape[0]

    if nbrs_idx is None:
        nbrs_idx, dists = k_nearest_neighbors(X, k, exclude_self=True)

    if dists is None:
        dists = nbrs_to_dists(X, nbrs_idx)

    if type(E_func) is str:
        if E_func == "sqrt":
            E_func = np.sqrt
        elif E_func == "exp":
            E_func = np.exp
        else:
            raise NotImplementedError("The specified edge function is not implemented.")

    if normalize_v:
        V_norm = np.linalg.norm(V, axis=1)
        V = np.array(V, copy=True)
        V[V_norm > 0] = V[V_norm > 0] / V_norm[V_norm > 0]

    if use_sparse:
        E = sp.lil_matrix((n, n))
    else:
        E = np.zeros((n, n))

    for i in range(n):
        x = flatten(X[i])
        idx = nbrs_idx[i]
        dist = dists[i]
        if len(idx) > 0 and idx[0] == i:  # excluding the node itself from the neighbors
            idx = idx[1:]
            dist = dist[1:]
        vi = flatten(V[i])

        # normalized differences
        U = X[idx] - x
        dist[dist == 0] = 1
        U /= dist[:, None]

        for jj, j in enumerate(idx):
            vj = flatten(V[j])
            u = flatten(U[jj])
            v = np.mean((vi.dot(u), vj.dot(u)))

            if E_func is not None:
                v = np.sign(v) * E_func(np.abs(v))
            E[i, j] = v
            E[j, i] = -v

    return E, nbrs_idx, dists


def calc_gaussian_weight(nbrs_idx, dists, sig, format="squareform"):
    n = len(nbrs_idx)
    if format == "sparse":
        W = sp.lil_matrix((n, n))
    elif format == "squareform":
        W = np.zeros((n, n))
    elif format == "condense":
        W = np.zeros(int(n * (n - 1) / 2))
    else:
        raise NotImplementedError(f"Unidentified format `{format}`")

    sig2 = sig ** 2
    for i in range(n):
        w = np.exp(-0.5 * dists[i] * dists[i] / sig2)

        if format == "condense":
            idx = np.array([index_condensed_matrix(n, i, j) for j in nbrs_idx[i]])
            W[idx] = w
        else:
            W[i, nbrs_idx[i]] = w
            W[nbrs_idx[i], i] = w

    return W


def calc_laplacian(W, weight_mode="symmetric", convention="graph"):
    if weight_mode == "naive":
        A = np.abs(np.sign(W))
    elif weight_mode == "symmetric":
        A = W
    elif weight_mode == "asymmetric":
        A = np.array(W, copy=True)
        A = 0.5 * (A + A.T)
    else:
        raise NotImplementedError(f"Unidentified weight mode: `{weight_mode}`")

    L = np.diag(np.sum(A, 0)) - A

    if convention == "diffusion":
        L = -L

    return L


def fp_operator(E, D, W=None, drift_weight=False, weight_mode="symmetric"):
    # drift
    Mu = E.T.copy()
    Mu[Mu < 0] = 0
    if W is not None and drift_weight:
        Mu *= W.T
    Mu = np.diag(np.sum(Mu, 0)) - Mu
    # diffusion
    if W is None:
        L = calc_laplacian(E, convention="diffusion", weight_mode="naive")
    else:
        L = calc_laplacian(W, convention="diffusion", weight_mode=weight_mode)

    # return - Mu + D * L
    return -0.5 * Mu + D * L


def divergence(E, tol=1e-5):
    n = E.shape[0]
    div = np.zeros(n)
    # optimize for sparse matrices later...
    for i in range(n):
        for j in range(i + 1, n):
            if np.abs(E[i, j]) > tol:
                div[i] += E[i, j] - E[j, i]

    return div
