from typing import Callable, Union

import numpy as np
import scipy.sparse as sp
from scipy.linalg import qr
from scipy.optimize import lsq_linear, minimize

from ..dynamo_logger import main_info, main_warning
from ..tools.utils import projection_with_transition_matrix
from .utils import (
    elem_prod,
    flatten,
    index_condensed_matrix,
    k_nearest_neighbors,
    nbrs_to_dists,
    symmetrize_symmetric_matrix,
)


# test
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
            A list of neighbor indices for each cell. If None a KNN will be performed instead.
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
            returns a neighbor object if this arg is True. A neighbor object is from k_nearest_neighbors and may be
            from NNDescent (pynndescent) or NearestNeighbors.

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


def graphize_velocity_coopt(
    X: np.ndarray,
    V: np.ndarray,
    C: np.ndarray,
    nbrs: list,
    a: float = 1.0,
    b: float = 1.0,
    r: float = 1.0,
    loss_func: str = "log",
    nonneg: bool = False,
    norm_dist: bool = False,
):
    # TODO: merge with graphize_velocity
    """
    The function generates a graph based on the velocity data by minimizing the loss function:
                    L(w_i) = a |v_ - v|^2 - b cos(u, v_) + lambda * \sum_j |w_ij|^2
    where v_ = \sum_j w_ij*d_ij. The flow from i- to j-th node is returned as the edge matrix E[i, j],
    and E[i, j] = -E[j, i].

    Arguments
    ---------
        X: :class:`~numpy.ndarray`
            The coordinates of cells in the expression space.
        V: :class:`~numpy.ndarray`
            The velocity vectors in the expression space.
        C: :class:`~numpy.ndarray`
            The transition matrix of cells based on the correlation/cosine kernel.
        nbrs: list
            List of neighbor indices for each cell.
        a: float (default 1.0)
            The weight for preserving the velocity length.
        b: float (default 1.0)
            The weight for the cosine similarity.
        r: float (default 1.0)
            The weight for the regularization.
        nonneg: bool (default False)
            Whether to ensure the resultant transition matrix to have non-negative values.
        norm_dist: bool (defauilt False)
            norm_dist should be False so that the resulting graph vector field penalizes long jumps.

    Returns
    -------
        E: :class:`~numpy.ndarray`
            The edge matrix.
    """
    E = np.zeros((X.shape[0], X.shape[0]))

    for i, x in enumerate(X):
        v, c, idx = V[i], C[i], nbrs[i]
        c = C[idx]

        # normalized differences
        D = X[idx] - x
        if norm_dist:
            dist = np.linalg.norm(D, axis=1)
            dist[dist == 0] = 1
            D /= dist[:, None]

        # co-optimization
        c_norm = np.linalg.norm(c)

        def func(w):
            v_ = w @ D

            # cosine similarity between w and c
            if b == 0:
                sim = 0
            else:
                cw = c_norm * np.linalg.norm(w)
                if cw > 0:
                    sim = c.dot(w) / cw
                else:
                    sim = 0

            # reconstruction error between v_ and v
            rec = v_ - v
            rec = rec.dot(rec)
            if loss_func is None or loss_func == "linear":
                rec = rec
            elif loss_func == "log":
                rec = np.log(rec)
            else:
                raise NotImplementedError(
                    f"The function {loss_func} is not supported. Choose either `linear` or `log`."
                )

            # regularization
            reg = 0 if r == 0 else w.dot(w)

            ret = a * rec - b * sim + r * reg
            return ret

        def fjac(w):
            v_ = w @ D

            # reconstruction error
            jac_con = 2 * a * D @ (v_ - v)

            if loss_func is None or loss_func == "linear":
                jac_con = jac_con
            elif loss_func == "log":
                jac_con = jac_con / (v_ - v).dot(v_ - v)

            # cosine similarity
            w_norm = np.linalg.norm(w)
            if w_norm == 0 or b == 0:
                jac_sim = 0
            else:
                jac_sim = b * (c / (w_norm * c_norm) - w.dot(c) / (w_norm**3 * c_norm) * w)

            # regularization
            if r == 0:
                jac_reg = 0
            else:
                jac_reg = 2 * r * w

            return jac_con - jac_sim + jac_reg

        if nonneg:
            bounds = [(0, np.inf)] * D.shape[0]
        else:
            bounds = None

        res = minimize(func, x0=D @ v, jac=fjac, bounds=bounds)
        E[i][idx] = res["x"]
    return E


def _graphize_velocity_coopt(
    X: np.ndarray,
    V: np.ndarray,
    U: np.ndarray,
    nbrs: list,
    a: float = 1.0,
    b: float = 1.0,
    r: float = 1.0,
    loss_func: str = "log",
    nonneg: bool = False,
    norm_dist: bool = False,
):
    # TODO: merge with graphize_velocity
    """
    The function generates a graph based on the velocity data by minimizing the loss function:
                    L(w_i) = a |v_ - v|^2 - b cos(u, v_) + lambda * \sum_j |w_ij|^2
    where v_ = \sum_j w_ij*d_ij. The flow from i- to j-th node is returned as the edge matrix E[i, j],
    and E[i, j] = -E[j, i].

    Arguments
    ---------
        X: :class:`~numpy.ndarray`
            The coordinates of cells in the expression space.
        V: :class:`~numpy.ndarray`
            The velocity vectors in the expression space.
        U: :class:`~numpy.ndarray`
            The correlation kernel-projected velocity vectors, must be in the original expression space, can be
            calculated as `adata.obsm['velocity_pca'] @ adata.uns['PCs'].T` where velocity_pca is the kernel projected
            velocity vector in PCA space.
        nbrs: list
            List of neighbor indices for each cell.
        a: float (default 1.0)
            The weight for preserving the velocity length.
        b: float (default 1.0)
            The weight for the cosine similarity.
        r: float (default 1.0)
            The weight for the regularization.
        nonneg: bool (default False)
            Whether to ensure the resultant transition matrix to have non-negative values.
        norm_dist: bool (defauilt False)
            norm_dist should be False so that the resulting graph vector field penalizes long jumps.



    Returns
    -------
        E: :class:`~numpy.ndarray`
            The edge matrix.
    """
    E = np.zeros((X.shape[0], X.shape[0]))

    for i, x in enumerate(X):
        v, u, idx = V[i], U[i], nbrs[i]

        # normalized differences
        D = X[idx] - x
        if norm_dist:
            dist = np.linalg.norm(D, axis=1)
            dist[dist == 0] = 1
            D /= dist[:, None]

        # co-optimization
        u_norm = np.linalg.norm(u)

        def func(w):
            v_ = w @ D

            # cosine similarity between v_ and u
            if b == 0:
                sim = 0
            else:
                uv = u_norm * np.linalg.norm(v_)
                if uv > 0:
                    sim = u.dot(v_) / uv
                else:
                    sim = 0

            # reconstruction error between v_ and v
            rec = v_ - v
            rec = rec.dot(rec)
            if loss_func is None or loss_func == "linear":
                rec = rec
            elif loss_func == "log":
                rec = np.log(rec)
            else:
                raise NotImplementedError(
                    f"The function {loss_func} is not supported. Choose either `linear` or `log`."
                )

            # regularization
            reg = 0 if r == 0 else w.dot(w)

            ret = a * rec - b * sim + r * reg
            return ret

        def fjac(w):
            v_ = w @ D
            v_norm = np.linalg.norm(v_)
            u_ = u / u_norm

            # reconstruction error
            jac_con = 2 * a * D @ (v_ - v)

            if loss_func is None or loss_func == "linear":
                jac_con = jac_con
            elif loss_func == "log":
                jac_con = jac_con / (v_ - v).dot(v_ - v)

            # cosine similarity
            if v_norm == 0 or b == 0:
                jac_sim = 0
            else:
                jac_sim = b / v_norm**2 * (v_norm * D @ u_ - v_.dot(u_) * v_ @ D.T / v_norm)

            # regularization
            if r == 0:
                jac_reg = 0
            else:
                jac_reg = 2 * r * w

            return jac_con - jac_sim + jac_reg

        if nonneg:
            bounds = [(0, np.inf)] * D.shape[0]
        else:
            bounds = None

        res = minimize(func, x0=D @ v, jac=fjac, bounds=bounds)
        E[i][idx] = res["x"]
    return E


def symmetrize_discrete_vector_field(F, mode="asym"):
    E_ = F.copy()
    for i in range(F.shape[0]):
        for j in range(i + 1, F.shape[1]):
            if mode == "asym":
                flux = 0.5 * (F[i, j] - F[j, i])
                E_[i, j], E_[j, i] = flux, -flux
            elif mode == "sym":
                flux = 0.5 * (F[i, j] + F[j, i])
                E_[i, j], E_[j, i] = flux, flux
    return E_


def dist_mat_to_gaussian_weight(dist, sigma):
    dist = symmetrize_symmetric_matrix(dist)
    W = elem_prod(dist, dist) / sigma**2
    W[W.nonzero()] = np.exp(-0.5 * W.data)

    return W


def calc_gaussian_weight(nbrs_idx, dists, sig=None, auto_sig_func=None, auto_sig_multiplier=2, format="squareform"):
    # TODO: deprecate this function
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

    sig2 = sig**2
    for i in range(n):
        w = np.exp(-0.5 * dists[i] * dists[i] / sig2)

        if format == "condense":
            idx = np.array([index_condensed_matrix(n, i, j) for j in nbrs_idx[i]])
            W[idx] = w
        else:
            W[i, nbrs_idx[i]] = w
            W[nbrs_idx[i], i] = w

    return W


def calc_laplacian(W, E=None, weight_mode="asymmetric", convention="graph"):
    """
    W: the weights stored for each edge e_ij.
    E: length of edges. If None, all edges are assumed to have lengths of one.
    """
    if weight_mode == "naive":
        A = np.abs(np.sign(W))
    elif weight_mode == "asymmetric":
        A = np.array(W, copy=True)
    elif weight_mode == "symmetric":
        A = np.array(W, copy=True)
        A = 0.5 * (A + A.T)
    else:
        raise NotImplementedError(f"Unidentified weight mode: `{weight_mode}`")

    if E is not None:
        E_ = np.zeros(E.shape)
        # E_[np.where(E > 0)] = 1 / E[np.where(E > 0)] ** 2
        E_[E.nonzero()] = 1 / E[E.nonzero()] ** 2
        # A *= E_
        A = elem_prod(A, E_)

    L = np.diag(np.sum(A, 1)) - A

    if convention == "diffusion":
        L = -L

    return L


def fp_operator(
    F, D, E=None, W=None, symmetrize_E=True, drift_weight=False, weight_mode="asymmetric", renormalize=False
):
    """
    The output of this function is a transition rate matrix Q, encoding the transition rate
    from node i to j in Q_ji

    F: graph vector field. F_ij encodes the flow on edge e_ij (from vertex i to j)
    D: diffusion coefficient
    W: edge weight. W_ij is the weight on edge e_ij
    """
    # drift
    if symmetrize_E:
        F = symmetrize_discrete_vector_field(F, mode="asym")

    Mu = F.T.copy()
    Mu[Mu < 0] = 0
    if W is not None and drift_weight:
        Mu *= W.T
    Mu = np.diag(np.sum(Mu, 0)) - Mu
    # diffusion
    if W is None:
        if E is not None:
            L = calc_laplacian(E, E=E, convention="diffusion", weight_mode="naive")
        else:
            L = calc_laplacian(F, E=E, convention="diffusion", weight_mode="naive")
    else:
        L = calc_laplacian(W, E=E, convention="diffusion", weight_mode=weight_mode)

    # return - Mu + D * L
    # TODO: make sure the 0.5 factor here is needed when there's already 0.5 in symmetrize dvf
    Q = -0.5 * Mu + D * L.T
    if renormalize:
        # TODO: automate this and give a warning.
        col_sum = np.sum(Q, 0)
        Q -= np.diag(col_sum)
    return Q


def divergence(E, W=None, method="operator"):
    # TODO: support weight in the future
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


def gradop(adj):
    e = np.array(adj.nonzero())
    ne = e.shape[1]
    nv = adj.shape[0]
    i, j, x = np.tile(range(ne), 2), e.flatten(), np.repeat([-1, 1], ne)

    return sp.csr_matrix((x, (i, j)), shape=(ne, nv))


def gradient(E, p):
    adj = np.abs(np.sign(E))
    F_pot = np.array(adj, copy=True, dtype=float)
    F_pot[F_pot.nonzero()] = gradop(adj) * p / E[E.nonzero()] ** 2
    return F_pot


def divop(W):
    return -0.5 * gradop(W).T


def potential(F, E=None, W=None, div=None, method="lsq"):
    """potential is related to the intrinsic time. Note that the returned value from this function is the negative of
    potential. Thus small potential is related to smaller intrinsic time and vice versa."""

    W = np.abs(np.sign(F)) if W is None else W
    div_neg = -divergence(F, W=W) if div is None else -div
    L = calc_laplacian(W, E=E, weight_mode="naive")

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


class GraphVectorField:
    def __init__(self, F, E=None, W=None, E_tol=1e-5) -> None:
        # TODO: sparse matrix support
        self.F = F
        self._sym = None
        self._asym = None
        self._div = None

        if W is None and E is None:
            main_info("Neither edge weight nor length matrix is provided. Inferring adjacency matrix from `F`.")
            self.adj = np.abs(np.sign(self.F))
            self.adj = symmetrize_symmetric_matrix(self.adj).A
            self.W = self.adj
            self.E = self.adj
        else:
            if E is not None and E.shape != F.shape:
                raise Exception("Edge length and graph vector field dimensions do not match.")
            if W is not None and W.shape != F.shape:
                raise Exception("Edge weight and graph vector field dimensions do not match.")

            self.adj = np.abs(np.sign(E)) if E is not None else np.abs(np.sign(W))
            self.adj = symmetrize_symmetric_matrix(self.adj).A
            self.W = W if W is not None else self.adj

            if E is not None:
                E_ = symmetrize_symmetric_matrix(E)
                if E_tol is not None:  # prevent numerical instability due to extremely small e_ij
                    E_[E_.nonzero()] = np.clip(E_[E_.nonzero()], 1e-5, None)
                self.E = E_.A
            else:
                self.E = self.adj

    def sym(self):
        """
        Return the symmetric components of the graph vector field.
        """
        if self._sym is None:
            self._sym = symmetrize_discrete_vector_field(self.F, mode="sym")
        return self._sym

    def asym(self):
        """
        Return the asymmetric components of the graph vector field.
        """
        if self._asym is None:
            self._asym = symmetrize_discrete_vector_field(self.F, mode="asym")
        return self._asym

    def divergence(self, **kwargs):
        if self._div is None:
            self._div = divergence(self.asym(), W=self.W, **kwargs)
        return self._div

    def potential(self, mode="asym", **kwargs):
        if mode == "raw":
            F_ = self.F
        elif mode == "asym":
            F_ = self.asym()
        return potential(F_, E=self.E, W=self.W, div=self.divergence(), **kwargs)

    def fp_operator(self, D, **kwargs):
        return fp_operator(self.asym(), D, E=self.E, W=self.W, symmetrize_E=False, **kwargs)

    def project_velocity(self, X_emb, mode="raw", correct_density=False, norm_dist=False, **kwargs):
        if mode == "raw":
            F_ = self.F
        elif mode == "asym":
            F_ = self.asym()

        return projection_with_transition_matrix(
            F_, X_emb, correct_density=correct_density, norm_dist=norm_dist, **kwargs
        )
