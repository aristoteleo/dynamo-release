import numpy as np
from scipy.optimize import least_squares

# from scPotential import show_landscape


def constructQ(q):
    """Construct the Q matrix from the vector q, estimated by the least square optimizer

    Parameters
    ----------
        q: `list`
            the list corresponds the elements in the Q matrix, estimated by the least square optimizer

    Returns
    -------
        Q: `numpy.ndarray`
            The Q matrix constructed
    """

    m = len(q)
    n = int((1 + np.sqrt(1 + 8 * m)) / 2)

    Q = np.zeros((n, n), dtype=float)
    c = 0
    for i in range(n):
        for j in range(n):
            if j > i:
                Q[i, j] = q[c]
                c += 1
            elif j < i:
                Q[i, j] = -Q[j, i]
    return Q


def solveQ(D, F, debug=False):
    """Function to calculate Q matrix by a least square method

    Parameters
    ----------
    D:  `numpy.ndarray`
        Diffusion matrix.
    F: `numpy.ndarray`
        Jacobian of the vector field function at specific location.
    debug: `bool`
        A flag to determine whether the debug mode should be used.

    Returns
    -------
        Depends on whether
    """

    n = D.shape[0]
    m = int(n * (n - 1) / 2)
    C = F.dot(D) - D.dot(F.T)
    f_left = lambda X, F: X.dot(F.T) + F.dot(X)
    # f_obj = @(q)(sum(sum((constructQ(q) * F' + F * constructQ(q) - C).^2)));
    f_obj = lambda q: np.sum((f_left(constructQ(q), F) - C) ** 2)

    sol = least_squares(f_obj, np.ones(m, dtype=float))
    Q = constructQ(sol.x)
    if debug:
        C_left = f_left(Q, F)
        return Q, C, C_left, sol.cost
    else:
        return Q, C


def Ao_pot_map(vecFunc, X, D=None):
    """Mapping potential landscape with the algorithm developed by Ao method.
    References: Potential in stochastic differential equations: novel construction. Journal of physics A: mathematical and
        general, Ao Ping, 2004

    Parameters
    ----------
        vecFunc: `function`
            The vector field function
        X: `numpy.ndarray`
            A matrix of coordinates to calculate potential values for. Rows are observations (cells), columns are features (genes)
        D: None or `numpy.ndarray`
            Diffusion matrix. It must be a square matrix with size corresponds to the number of columns (features) in the X matrix.

    Returns
    -------
        X: `numpy.ndarray`
            A matrix storing the x-coordinates on the two-dimesional grid.
        U: `numpy.ndarray`
            A matrix storing the potential value at each position.
        P: `numpy.ndarray`
            Steady state distribution or the Boltzmann-Gibbs distribution for the state variable.
        vecMat: `list`
            List velocity vector at each position from X.
        S: `list`
            List of constant symmetric and semi-positive matrix or friction matrix, corresponding to the divergence part,
            at each position from X.
        A: `list`
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
        Q, _ = solveQ(D, F)
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
