import numpy as np
from scipy.optimize import least_squares
import numdifftools as nda

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
    n = int((1 + np.sqrt(1 + 8*m))/2)

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
    m = int(n*(n-1)/2)
    C = F.dot(D) - D.dot(F.T)
    f_left = lambda X, F: X.dot(F.T) + F.dot(X)
    #f_obj = @(q)(sum(sum((constructQ(q) * F' + F * constructQ(q) - C).^2)));
    f_obj = lambda q: np.sum((f_left(constructQ(q), F) - C)**2)

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
        general, Ao Ping, 2014

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
    """

    nobs, ndim = X.shape
    D = 0.1 * np.eye(ndim) if D is None else D
    U = np.zeros((nobs, 1))

    for i in range(nobs):
        X_s = X[i, :]
        F = nda.Jacobian(vecFunc)(X_s)
        Q, _ = solveQ(D, F)
        H = np.linalg.inv(D + Q).dot(F)
        U[i] = - 0.5 * X_s.dot(H).dot(X_s)

    return X, U

# #
# # # Xgrid, Ygrid, U = Ao_pot_map(vecFunc=two_gene_model, test=1, xlim=[5], ylim=[5], N=50)
# # # show_landscape(Xgrid, Ygrid, U)
# # # test on real dataset
# import scipy.io
# VecFld = scipy.io.loadmat('/Volumes/xqiu/proj/dynamo/data/VecFld.mat')
#
# def vector_field_function(x, VecFld = VecFld):
#     '''Learn an analytical function of vector field from sparse single cell samples on the entire space robustly.
#     Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition
#     '''
#     x = x.reshape((1, 2))
#     K= dyn.tl.con_K(x, VecFld['X'], VecFld['beta'])
#
#     K = K.dot(VecFld['C'])
#
#     return K.T
#
# def vector_field_function_auto(x, VecFld = VecFld, autograd = False):
#     '''Learn an analytical function of vector field from sparse single cell samples on the entire space robustly.
#     Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition
#     '''
#     if(len(x.shape) == 1):
#         x = x[None, :]
#     K= auto_con_K(x, VecFld['X'], VecFld['beta'])
#
#     K = K.dot(VecFld['C'])
#
#     return K
#
# import dynamo as dyn
#
# xlim, ylim = [-25, 25] , [-25, 25]
# N = 100
# x_space = np.diff(xlim)[0] / N
# y_space = np.diff(ylim)[0] / N
#
# Xgrid, Ygrid = np.meshgrid(np.arange(xlim[0], xlim[1], x_space), np.arange(xlim[0], xlim[1], y_space))
#
# U = Ao_pot_map(vecFunc=vector_field_function, X=np.vstack((Xgrid.flatten(), Ygrid.flatten())).T, D=None)
# show_landscape(Xgrid, Ygrid, U.reshape(Xgrid.shape))
#
# import plotly.graph_objects as go
#
# import pandas as pd
#
# # Read data from a csv
# z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
#
# fig = go.Figure(data=[go.Surface(z=U)])
#
# fig.update_layout(title='Mt Bruno Elevation', autosize=False,
#                   width=500, height=500,
#                   margin=dict(l=65, r=50, b=65, t=90))
#
# fig.show()
