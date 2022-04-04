from typing import Callable, Union

import numpy as np
import scipy.sparse as sp
from scipy.linalg import qr
from scipy.optimize import lsq_linear

from ..dynamo_logger import main_info, main_warning
from .utils import flatten, index_condensed_matrix, k_nearest_neighbors, nbrs_to_dists


def graphize_velocity(
    V,
    X,
    nbrs_idx: Union[np.array, list] = None,
    dists: np.array = None,
    k: int = 30,
    normalize_v: bool = False,
    scale_by_dist: bool = False,
    E_func: Union[str, Callable] = None,
    use_sparse: bool = False,
    return_nbrs: bool = False,
) -> tuple:
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
        return_nbrs:
            returns a neighbor object if this arg is true. A neighbor object is from k_nearest_neighbors and may be from NNDescent (pynndescent) or NearestNeighbors.

    Returns
    -------
        E: :class:`~numpy.ndarray`
            The edge matrix.
        nbrs_idx: :class:`~numpy.ndarray`
            Neighbor indices.
    """
    n = X.shape[0]

    if (nbrs_idx is not None) and return_nbrs:
        main_warning(
            "nbrs_idx argument is ignored and recomputed because nbrs_idx is not None and return_nbrs=True",
            indent_level=2,
        )

    if nbrs_idx is None or return_nbrs:
        main_info("calculating neighbor indices...", indent_level=2)
        nbrs_idx, dists, nbrs = k_nearest_neighbors(X, k, exclude_self=True, return_nbrs=True)

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
        V_norm[V_norm == 0] = 1
        V = np.array(V, copy=True)
        V = (V.T / V_norm).T

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
            if scale_by_dist:
                v /= dist[jj]

            if E_func is not None:
                v = np.sign(v) * E_func(np.abs(v))
            E[i, j] = v
            E[j, i] = -v

    if return_nbrs:
        return E, nbrs_idx, dists, nbrs
    return E, nbrs_idx, dists


def calc_gaussian_weight(nbrs_idx, dists, sig=None, auto_sig_func=None, auto_sig_multiplier=2, format="squareform"):
    n = len(nbrs_idx)
    if format == "sparse":
        W = sp.lil_matrix((n, n))
    elif format == "squareform":
        W = np.zeros((n, n))
    elif format == "condense":
        W = np.zeros(int(n * (n - 1) / 2))
    else:
        raise NotImplementedError(f"Unidentified format `{format}`")

    if sig is None:
        if auto_sig_func is None:
            auto_sig_func = np.median
        sig = auto_sig_func(dists) * auto_sig_multiplier

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


def divergence(E, W=None, method="operator"):
    # support weight in the future
    if method == "direct":
        n = E.shape[0]
        div = np.zeros(n)
        for i in range(n):
            div[i] += np.sum(E[i, :]) - np.sum(E[:, i])
        div *= 0.5
    elif method == "operator":
        W = np.abs(np.sign(E)) if W is None else W
        div = divop(W) @ E[W.nonzero()]
    else:
        raise NotImplementedError(f"Unsupported method `{method}`")

    return div


def gradop(W):
    e = np.array(W.nonzero())
    ne = e.shape[1]
    nv = W.shape[0]
    i, j, x = np.tile(range(ne), 2), e.flatten(), np.repeat([-1, 1], ne)

    return sp.csr_matrix((x, (i, j)), shape=(ne, nv))


def divop(W):
    return -0.5 * gradop(W).T


def potential(E, W=None, div=None, method="lsq"):
    """potential is related to the intrinsic time. Note that the returned value from this function is the negative of
    potential. Thus small potential is related to smaller intrinsic time and vice versa."""

    W = np.abs(np.sign(E)) if W is None else W
    div_neg = -divergence(E, W=W) if div is None else -div
    L = calc_laplacian(W, weight_mode="naive")

    if method == "inv":
        p = np.linalg.inv(L).dot(div_neg)
    elif method == "pinv":
        p = np.linalg.pinv(L).dot(div_neg)
    elif method == "qr_pinv":
        Q, R = qr(L)
        L_inv = np.linalg.pinv(R).dot(Q.T)
        p = L_inv.dot(div_neg)
    elif method == "lsq":
        res = lsq_linear(L, div_neg)
        p = res["x"]

    p -= p.min()
    return p
