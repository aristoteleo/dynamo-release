import numpy as np
from scipy import optimize
import numdifftools as nda


def Wang_action(X_input, dim, F, D, N, lamada_=1):
    """Calculate action by path integral by Wang's method.
    Quantifying the Waddington landscape and biological paths for development and differentiation. Jin Wang, Kun Zhang,
    Li Xu, and Erkang Wang, PNAS, 2011

    Parameters
    ----------
        X_input: `numpy.ndarray`
            The initial guess of the least action path. Default is a straight line connecting the starting and end path.
        dim: `int`
            The feature numbers of the input data.
        F: `Function`
            The reconstructed vector field function
        D: `float`
            The diffusion constant. Note that this can be a space-dependent matrix.
        N: `int`
            Number of waypoints along the least action path.
        lamada_: `float`
            Regularization parameter

    Returns
    -------
        The action function calculated by the Hamilton-Jacobian method.
    """

    X_input = X_input.reshape((dim, -1)) if len(X_input.shape) == 1 else X_input
    print(X_input.shape)
    a = np.arange(1.5, 0-1.5/N, -1.5/N)
    E_eff = - V(F, D, X_input[:, X_input.shape[1] - 1])

    delta, delta_l = delta_delta_l(X_input)

    V_m = np.zeros((N, 1))
    F_l = np.zeros((N, 1))

    for i in range(N):
        F_m = F(X_input[:, i])
        V_m[i] = V(F, D, X_input[:, i])
        F_l[i] = F_m.dot(delta[:, i]) / delta_l[i]

    P = np.sum((delta_l - np.linalg.norm(X_input[:, N] - X_input[:, 0]) / N)**2)
    S_HJ = np.sum((np.sqrt((E_eff + V_m[:N])/D) - 1/(2 * D)* F_l) * delta_l) + lamada_ * P

    return S_HJ


def V_jacobina(F, X):
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

    V = 1 / (4 * D) * np.sum(F(X)**2)  + 1/2 * np.sum(np.diagonal(V_jacobina(F, X)))

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
    delta_l = np.sqrt(np.sum(delta**2, 0))

    return delta, delta_l


def Wang_LAP(X_input, F, D=0.1, lambda_=1):
    """Calculating least action path based methods from Jin Wang and colleagues (http://www.pnas.org/cgi/doi/10.1073/pnas.1017017108)

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
        The least action path and the action way of the inferred path.
    """

    dim, N = X_input.shape
    res = optimize.basinhopping(Wang_action, x0=X_input, minimizer_kwargs={'args': (dim, F, D, N, lambda_)})

    return res
