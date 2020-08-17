import numpy as np
from scipy import optimize


def Wang_action(X_input, F, D, dim, N, lamada_=1):
    """Calculate action by path integral by Wang's method.
    Quantifying the Waddington landscape and biological paths for development and differentiation. Jin Wang, Kun Zhang,
    Li Xu, and Erkang Wang, PNAS, 2011

    Parameters
    ----------
        X_input: `numpy.ndarray`
            The initial guess of the least action path. Default is a straight line connecting the starting and end path.
        F: `Function`
            The reconstructed vector field function. This is assumed to be time-independent.
        D: `float`
            The diffusion constant. Note that this can be a space-dependent matrix.
        dim: `int`
            The feature numbers of the input data.
        N: `int`
            Number of waypoints along the least action path.
        lamada_: `float`
            Regularization parameter

    Returns
    -------
        The action function calculated by the Hamilton-Jacobian method.
    """

    X_input = X_input.reshape((int(dim), -1)) if len(X_input.shape) == 1 else X_input

    delta, delta_l = delta_delta_l(X_input)

    V_m = np.zeros((N, 1))
    F_l = np.zeros((N, 1))
    E_eff = np.zeros((N, 1))

    for i in range(N - 1):
        F_m = F(X_input[:, i]).reshape((1, -1))
        V_m[i] = V(F, D, X_input[:, i])
        E_eff[i] = np.sum(F(X_input[:, i]) ** 2) / (4 * D) - V_m[i]
        F_l[i] = F_m.dot(delta[:, i]) / delta_l[i]

    P = np.sum((delta_l - np.linalg.norm(X_input[:, N - 1] - X_input[:, 0]) / N) ** 2)
    S_HJ = (
        np.sum((np.sqrt((E_eff + V_m[:N]) / D) - 1 / (2 * D) * F_l) * delta_l)
        + lamada_ * P
    )

    print(S_HJ)
    return S_HJ


def V_jacobina(F, X):
    import numdifftools as nda

    V_jacobina = nda.Jacobian(F)

    return V_jacobina(X)


def V(F, D, X):
    """Calculate V

    Parameters
    ----------
        F: `Function`
            The reconstructed vector field function
        D: `float`
            The diffusion constant
        X: `nummpy.ndarray`
            The input coordinates corresponding to the cell states.

    Returns
    -------
        Returns V
    """

    V = 1 / (4 * D) * np.sum(F(X) ** 2) + 1 / 2 * np.trace(V_jacobina(F, X))

    return V


def delta_delta_l(X_input):
    """Calculate delta_L

    Parameters
    ----------
    X_input: `numpy.ndarray`

    Returns
    -------
    Return delta_L
    """

    delta = np.diff(X_input, 1, 1)
    delta_l = np.sqrt(np.sum(delta ** 2, 0))

    return delta, delta_l


def Wang_LAP(F, n_points, point_start, point_end, D=0.1, lambda_=1):
    """Calculating least action path based methods from Jin Wang and colleagues (http://www.pnas.org/cgi/doi/10.1073/pnas.1017017108)

    Parameters
    ----------
        F: `Function`
            The reconstructed vector field function
        n_points: 'int'
            The number of points along the least action path.
        point_start: 'np.ndarray'
            The matrix for storing the coordinates (gene expression configuration) of the start point (initial cell state).
        point_end: 'np.ndarray'
            The matrix for storing the coordinates (gene expression configuration) of the end point (terminal cell state).
        D: `float`
            The diffusion constant. Note that this can be a space-dependent matrix.
        lamada_: `float`
            Regularization parameter

    Returns
    -------
        The least action path and the action way of the inferred path.
    """
    initpath = point_start.dot(np.ones((1, n_points + 1))) + (
        point_end - point_start
    ).dot(np.linspace(0, 1, n_points + 1, endpoint=True).reshape(1, -1))

    dim, N = initpath.shape
    # update this optimization method
    res = optimize.basinhopping(
        Wang_action, x0=initpath, minimizer_kwargs={"args": (F, D, dim, N, lambda_)}
    )

    return res


def transition_rate(X_input, F, D=0.1, lambda_=1):
    """Calculate the rate to convert from one cell state to another cell state by taking the optimal path.

     In the small noise limit (D -> 0) the Wentzell-Freidlin theory states that the transition rate from one basin to
     another one to a leading order is related to the minimum action corresponding zero energy path (Eeff = 0) connecting
     the starting fixed point and the saddle point x_{sd} by k \approx exp(−S0 (x_{sd})). To take into account that for finite
     noise, the actual optimal path bypasses the saddle point, in Eqn. 2 of the main text a transition rate is
     actually estimated by the action of the whole path connecting the two fixed points, giving that the portion of the path
     following the vector field contributes zero action. Here we have neglected some pre-exponential factor (see Eq. 5.24 of
     reference [15]), which is expected to be on the order of 1 [12]. (Reference: Epigenetic state network approach for
     describing cell phenotypic transitions. Ping Wang, Chaoming Song, Hang Zhang, Zhanghan Wu, Xiao-Jun Tian and Jianhua Xing)

    Parameters
    ----------
        X_input: `numpy.ndarray`
            The initial guess of the least action path. Default is a straight line connecting the starting and end path.
        F: `Function`
            The reconstructed vector field function
        D: `float`
            The diffusion constant. Note that this can be a space-dependent matrix.
        lamada_: `float`
            Regularization parameter

    Returns
    -------
        The transition to convert from one cell state to another.
    """

    res = Wang_LAP(X_input, F, D=D, lambda_=lambda_)
    r = np.exp(-res)

    return r


def MFPT(X_input, F, D=0.1, lambda_=1):
    """Calculate the MFPT (mean first passage time) to convert from one cell state to another cell state by taking the optimal path.

     The mean first-passage time (MFPT) defines an average timescale for a stochastic event to first occur. The MFPT maps
     a multi-step kinetic process to a coarse-grained timescale for reaching a final state, having started at some initial
     state. The inverse of the MFPT is an effective rate of the overall reaction. (reference: Mean First-Passage Times in Biology
     Nicholas F. Polizzi,a Michael J. Therien,b and David N. Beratan)

    Parameters
    ----------
        X_input: `numpy.ndarray`
            The initial guess of the least action path. Default is a straight line connecting the starting and end path.
        F: `Function`
            The reconstructed vector field function
        D: `float`
            The diffusion constant. Note that this can be a space-dependent matrix.
        lamada_: `float`
            Regularization parameter

    Returns
    -------
        The transition to convert from one cell state to another.
    """

    r = transition_rate(X_input, F, D=D, lambda_=lambda_)
    t = 1 / r

    return t
