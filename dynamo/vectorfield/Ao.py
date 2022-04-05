import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares
from tqdm import tqdm

from ..tools.utils import condensed_idx_to_squareform_idx, squareform, timeit

# from scPotential import show_landscape


def f_left(X, F):
    """An auxiliary function for fast computation of F.X - (F.X)^T"""
    R = F.dot(X)
    return R - R.T


def f_left_jac(q, F):
    """
    Analytical Jacobian of f(Q) = F.Q - (F.Q)^T, where Q is
    an anti-symmetric matrix s.t. Q^T = -Q.
    """
    J = np.zeros((np.prod(F.shape), len(q)))
    for i in range(len(q)):
        jac = np.zeros(F.shape)
        a, b = condensed_idx_to_squareform_idx(len(q), i)
        jac[:, b] = F[:, a]
        jac[:, a] = -F[:, b]
        jac[b, :] -= F[:, a]
        jac[a, :] -= -F[:, b]
        J[:, i] = jac.flatten()
    return J


@timeit
def solveQ(D, F, q0=None, debug=False, precompute_jac=True, **kwargs):
    """Function to solve for the anti-symmetric Q matrix in the equation:
        F.Q - (F.Q)^T = F.D - (F.D)^T
    using least squares.

    Parameters
    ----------
        D: :class:`~numpy.ndarray`
            A symmetric diffusion matrix.
        F: :class:`~numpy.ndarray`
            Jacobian of the vector field function evaluated at a particular point.
        debug: bool
            Whether additional info of the solution is returned.
        precompute_jac: bool
            Whether the analytical Jacobian is precomputed for the optimizer.

    Returns
    -------
        Q: :class:`~numpy.ndarray`
            The solved anti-symmetric Q matrix.
        C: :class:`~numpy.ndarray`
            The right-hand side of the equation to be solved.
    """

    n = D.shape[0]
    m = int(n * (n - 1) / 2)
    C = f_left(D, F)
    f_obj = lambda q: (f_left(squareform(q, True), F) - C).flatten()
    q0 = np.ones(m, dtype=float) if q0 is None else q0
    if precompute_jac:
        J = f_left_jac(q0, F)
        f_jac = lambda q: J
    else:
        f_jac = "2-point"
    sol = least_squares(f_obj, q0, jac=f_jac, **kwargs)
    Q = squareform(sol.x, True)
    if debug:
        C_left = f_left(Q, F)
        return Q, C, C_left, sol.cost
    else:
        return Q, C


def Ao_pot_map(vecFunc, X, D=None, **kwargs):
    """Mapping potential landscape with the algorithm developed by Ao method.
    References: Potential in stochastic differential equations: novel construction. Journal of physics A: mathematical and
        general, Ao Ping, 2004

    Parameters
    ----------
        vecFunc: `function`
            The vector field function
        X: :class:`~numpy.ndarray`
            A n_cell x n_dim matrix of coordinates where the potential function is evaluated.
        D: None or :class:`~numpy.ndarray`
            Diffusion matrix. It must be a square matrix with size corresponds to the number of columns (features) in the X matrix.

    Returns
    -------
        X: :class:`~numpy.ndarray`
            A matrix storing the x-coordinates on the two-dimesional grid.
        U: :class:`~numpy.ndarray`
            A matrix storing the potential value at each position.
        P: :class:`~numpy.ndarray`
            Steady state distribution or the Boltzmann-Gibbs distribution for the state variable.
        vecMat: list
            List velocity vector at each position from X.
        S: list
            List of constant symmetric and semi-positive matrix or friction matrix, corresponding to the divergence part,
            at each position from X.
        A: list
            List of constant antisymmetric matrix or transverse matrix, corresponding to the curl part, at each position
            from X.
    """

    import numdifftools as nda

    nobs, ndim = X.shape
    D = 0.1 * np.eye(ndim) if D is None else D
    U = np.zeros((nobs, 1))
    vecMat, S, A = [None] * nobs, [None] * nobs, [None] * nobs

    for i in range(nobs):
        X_s = X[i, :]
        F = nda.Jacobian(vecFunc)(X_s)
        Q, _ = solveQ(D, F, **kwargs)
        H = np.linalg.inv(D + Q).dot(F)
        U[i] = -0.5 * X_s.dot(H).dot(X_s)

        vecMat[i] = vecFunc(X_s)
        S[i], A[i] = (
            (np.linalg.inv(D + Q) + np.linalg.inv((D + Q).T)) / 2,
            (np.linalg.inv(D + Q) - np.linalg.inv((D + Q).T)) / 2,
        )

    P = np.exp(-U)
    P = P / np.sum(P)

    return X, U, P, vecMat, S, A


def Ao_pot_map_jac(fjac, X, D=None, **kwargs):
    nobs, ndim = X.shape
    if D is None:
        D = 0.1 * np.eye(ndim)
    elif np.isscalar(D):
        D = D * np.eye(ndim)
    U = np.zeros((nobs, 1))

    m = int(ndim * (ndim - 1) / 2)
    q0 = np.ones(m) * np.mean(np.diag(D)) * 1000
    for i in tqdm(range(nobs), "Calc Ao potential"):
        X_s = X[i, :]
        F = fjac(X_s)
        Q, _ = solveQ(D, F, q0=q0, **kwargs)
        H = np.linalg.inv(D + Q).dot(F)
        U[i] = -0.5 * X_s.dot(H).dot(X_s)

    return U.flatten()
