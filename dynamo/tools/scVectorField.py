from tqdm import tqdm
import numpy.matlib
from numpy import format_float_scientific as scinot
import numpy as np
from scipy.linalg import lstsq
import numdifftools as nda
import warnings
import time
from numpy import format_float_scientific as scinot
from sklearn.neighbors import NearestNeighbors
from .utils import update_dict, update_n_merge_dict, linear_least_squares, timeit
from scipy.spatial.distance import cdist
from .sampling import sample_by_velocity

def norm(X, V, T):
    """Normalizes the X, Y (X + V) matrix to have zero means and unit covariance.
        We use the mean of X, Y's center (mean) and scale parameters (standard deviation) to normalize T.

    Arguments
    ---------
        X: 'np.ndarray'
            Current state. This corresponds to, for example, the spliced transcriptomic state.
        V: 'np.ndarray'
            Velocity estimates in delta t. This corresponds to, for example, the inferred spliced transcriptomic velocity estimated calculated by dynamo or velocyto, scvelo.
        T: 'np.ndarray'
            Current state on a grid which is often used to visualize the vector field. This corresponds to, for example, the spliced transcriptomic state.

    Returns
    -------
        A tuple of updated X, V, T and norm_dict which includes the mean and scale values for original X, V data used
        in normalization.
    """

    Y = X + V
    n, m = X.shape[0], V.shape[0]

    xm = np.mean(X, 0)
    ym = np.mean(Y, 0)

    x, y, t = (
        X - xm[None, :],
        Y - ym[None, :],
        T - (1 / 2 * (xm[None, :] + ym[None, :])),
    )

    xscale, yscale = (
        np.sqrt(np.sum(np.sum(x ** 2, 1)) / n),
        np.sqrt(np.sum(np.sum(y ** 2, 1)) / m),
    )

    X, Y, T = x / xscale, y / yscale, t / (1 / 2 * (xscale + yscale))

    X, V, T = X, Y - X, T
    norm_dict = {"xm": xm, "ym": ym, "xscale": xscale, "yscale": yscale}

    return X, V, T, norm_dict

def bandwidth_rule_of_thumb(X, return_sigma=False):
    '''
        This function computes a rule-of-thumb bandwidth for a Gaussian kernel based on:
        https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
    '''
    sig = sig = np.sqrt(np.mean(np.diag(np.cov(X.T))))
    h = 1.06 * sig/(len(X)**(-1/5))
    if return_sigma:
        return h, sig
    else:
        return h

def bandwidth_selector(X):
    '''
        This function computes a empirical bandwidth for a Gaussian kernel.
    '''
    n = len(X)
    nbrs = NearestNeighbors(n_neighbors=int(0.2*n), algorithm='ball_tree').fit(X)
    distances, _ = nbrs.kneighbors(X)
    d = np.mean(distances[:, 1:]) / 1.5
    return np.sqrt(2) * d

def denorm(VecFld, X_old, V_old, norm_dict):
    """Denormalize data back to the original scale.

    Parameters
    ----------
        VecFld:  `dict`
            The dictionary that stores the information for the reconstructed vector field function.
        X_old: `np.ndarray`
            The original data for current state.
        V_old: `np.ndarray`
            The original velocity data.
        norm_dict: `dict`
            norm_dict to the class which includes the mean and scale values for X, Y used in normalizing the data.

    Returns
    -------
        An updated VecFld function that includes denormalized X, Y, X_ctrl, grid, grid_V, V and the norm_dict key.
    """

    Y_old = X_old + V_old
    X, Y, V, xm, ym, x_scale, y_scale = VecFld['X'], VecFld['Y'], VecFld['V'], norm_dict['xm'], norm_dict['ym'], \
                                        norm_dict['xscale'], norm_dict['yscale']
    grid, grid_V = VecFld['grid'], VecFld['grid_V']
    xy_m, xy_scale = (xm + ym) / 2, (x_scale + y_scale) / 2

    VecFld['X'] = X_old
    VecFld['Y'] = Y_old
    VecFld['X_ctrl'] = X * x_scale + np.matlib.tile(xm, [X.shape[0], 1])
    VecFld['grid'] = grid * xy_scale + np.matlib.tile(xy_m, [X.shape[0], 1])
    VecFld['grid_V'] = (grid + grid_V) * xy_scale + np.matlib.tile(xy_m, [Y.shape[0], 1]) - grid
    VecFld['V'] = (V + X) * y_scale + np.matlib.tile(ym, [Y.shape[0], 1]) - X_old
    VecFld['norm_dict'] = norm_dict

    return VecFld


@timeit
def lstsq_solver(lhs, rhs, method='drouin'):
    if method == 'scipy':
        C = lstsq(lhs, rhs)[0]
    elif method == 'drouin':
        C = linear_least_squares(lhs, rhs)
    else:
        warnings.warn('Invalid linear least squares solver. Use Drouin\'s method instead.')
        C = linear_least_squares(lhs, rhs)
    return C


def get_P(Y, V, sigma2, gamma, a, div_cur_free_kernels=False):
    """GET_P estimates the posterior probability and part of the energy.

    Arguments
    ---------
        Y: 'np.ndarray'
            Velocities from the data.
        V: 'np.ndarray'
            The estimated velocity: V=f(X), f being the vector field function.
        sigma2: 'float'
            sigma2 is defined as sum(sum((Y - V)**2)) / (N * D)
        gamma: 'float'
            Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
        a: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.
        div_cur_free_kernels: `bool` (default: False)
            A logic flag to determine whether the divergence-free or curl-free kernels will be used for learning the vector
            field.

    Returns
    -------
    P: 'np.ndarray'
        Posterior probability, related to equation 27.
    E: `np.ndarray'
        Energy, related to equation 26.

    """

    if div_cur_free_kernels:
        Y = Y.reshape((2, int(Y.shape[0] / 2)), order='F').T
        V = V.reshape((2, int(V.shape[0] / 2)), order='F').T

    D = Y.shape[1]
    temp1 = np.exp(-np.sum((Y - V) ** 2, 1) / (2 * sigma2))
    temp2 = (2 * np.pi * sigma2) ** (D / 2) * (1 - gamma) / (gamma * a)
    temp1[temp1 == 0] = np.min(temp1[temp1 != 0])
    P = temp1 / (temp1 + temp2)
    E = (
        P.T.dot(np.sum((Y - V) ** 2, 1)) / (2 * sigma2)
        + np.sum(P) * np.log(sigma2) * D / 2
    )

    return (P[:, None], E) if P.ndim == 1 else (P, E)

@timeit
def con_K(x, y, beta, method='cdist'):
    """con_K constructs the kernel K, where K(i, j) = k(x, y) = exp(-beta * ||x - y||^2).

    Arguments
    ---------
        x: 'np.ndarray'
            Original training data points.
        y: 'np.ndarray'
            Control points used to build kernel basis functions.
        beta: 'float' (default: 0.1)
            Paramerter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2),

    Returns
    -------
    K: 'np.ndarray'
    the kernel to represent the vector field function.
    """
    if method == 'cdist':
        K = cdist(x, y, 'sqeuclidean')
        if len(K) == 1:
            K = K.flatten()
    else:
        n = x.shape[0]
        m = y.shape[0]

        # https://stackoverflow.com/questions/1721802/what-is-the-equivalent-of-matlabs-repmat-in-numpy
        # https://stackoverflow.com/questions/12787475/matlabs-permute-in-python
        K = np.matlib.tile(x[:, :, None], [1, 1, m]) - np.transpose(
            np.matlib.tile(y[:, :, None], [1, 1, n]), [2, 1, 0])
        K = np.squeeze(np.sum(K ** 2, 1))
    K = -beta * K
    K = np.exp(K)

    return K

@timeit
def con_K_div_cur_free(x, y, sigma=0.8, eta=0.5):
    """Construct a convex combination of the divergence-free kernel T_df and curl-free kernel T_cf with a bandwidth sigma
    and a combination coefficient gamma.

    Arguments
    ---------
        x: 'np.ndarray'
            Original training data points.
        y: 'np.ndarray'
            Control points used to build kernel basis functions
        sigma: 'int' (default: `0.8`)
            Bandwidth parameter.
        eta: 'int' (default: `0.5`)
            Combination coefficient for the divergence-free or the curl-free kernels.

    Returns
    -------
        A tuple of G (the combined kernel function), divergence-free kernel and curl-free kernel.

    See also:: :func:`sparseVFC`.
    """
    m, d = x.shape
    n, d = y.shape
    sigma2 = sigma ** 2
    G_tmp = np.matlib.tile(x[:, :, None], [1, 1, n]) - np.transpose(
        np.matlib.tile(y[:, :, None], [1, 1, m]), [2, 1, 0]
    )
    G_tmp = np.squeeze(np.sum(G_tmp ** 2, 1))
    G_tmp3 = -G_tmp / sigma2
    G_tmp = -G_tmp / (2 * sigma2)
    G_tmp = np.exp(G_tmp) / sigma2
    G_tmp = np.kron(G_tmp, np.ones((d, d)))

    x_tmp = np.matlib.tile(x, [n, 1])
    y_tmp = np.matlib.tile(y, [1, m]).T
    y_tmp = y_tmp.reshape((d, m * n), order='F').T
    xminusy = x_tmp - y_tmp
    G_tmp2 = np.zeros((d * m, d * n))

    tmp4_ = np.zeros((d, d))
    for i in tqdm(range(d), desc="Iterating each dimension in con_K_div_cur_free:"):
        for j in np.arange(i, d):
            tmp1 = xminusy[:, i].reshape((m, n), order='F')
            tmp2 = xminusy[:, j].reshape((m, n), order='F')
            tmp3 = tmp1 * tmp2
            tmp4 = tmp4_.copy()
            tmp4[i, j] = 1
            tmp4[j, i] = 1
            G_tmp2 = G_tmp2 + np.kron(tmp3, tmp4)

    G_tmp2 = G_tmp2 / sigma2
    G_tmp3 = np.kron((G_tmp3 + d - 1), np.eye(d))
    G_tmp4 = np.kron(np.ones((m, n)), np.eye(d)) - G_tmp2
    df_kernel, cf_kernel = (1 - eta) * G_tmp * (G_tmp2 + G_tmp3), eta * G_tmp * G_tmp4
    G = df_kernel + cf_kernel

    return G, df_kernel, cf_kernel


@timeit
def vector_field_function(x, VecFld, dim=None, kernel='full', **kernel_kwargs):
    """Learn an analytical function of vector field from sparse single cell samples on the entire space robustly.
    Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition
    """
    # x=np.array(x).reshape((1, -1))
    if "div_cur_free_kernels" in VecFld.keys():
        has_div_cur_free_kernels = True
    else:
        has_div_cur_free_kernels = False

    #x = np.array(x)
    if x.ndim == 1:
        x = x[None, :]

    if has_div_cur_free_kernels:
        if kernel == 'full':
            kernel_ind = 0
        elif kernel == 'df_kernel':
            kernel_ind = 1
        elif kernel == 'cf_kernel':
            kernel_ind = 2
        else:
            raise ValueError(f"the kernel can only be one of {'full', 'df_kernel', 'cf_kernel'}!")

        K = con_K_div_cur_free(x, VecFld["X_ctrl"], VecFld["sigma"], VecFld["eta"], **kernel_kwargs)[kernel_ind]
    else:
        K = con_K(x, VecFld["X_ctrl"], VecFld["beta"], **kernel_kwargs)

    if dim is None:
        K = K.dot(VecFld["C"])
    else:
        K = K.dot(VecFld["C"]) if con_K_div_cur_free else K.dot(VecFld["C"][:, dim])
    return K


@timeit
def compute_divergence(func, X):
    f_jac = nda.Jacobian(func)
    div = np.zeros(len(X))
    for i in tqdm(range(len(X)), desc="Calculating divergence"):
        J = f_jac(X[i])
        div[i] = np.trace(J)

    return div


def SparseVFC(
    X,
    Y,
    Grid,
    M=100,
    a=5,
    beta=None,
    ecr=1e-5,
    gamma=0.9,
    lambda_=3,
    minP=1e-5,
    MaxIter=500,
    theta=0.75,
    div_cur_free_kernels=False,
    velocity_based_sampling=True,
    sigma=0.8,
    eta=0.5,
    lstsq_method='drouin',
    verbose=1
):
    """Apply sparseVFC (vector field consensus) algorithm to learn a functional form of the vector field from random
    samples with outlier on the entire space robustly and efficiently. (Ma, Jiayi, etc. al, Pattern Recognition, 2013)

    Arguments
    ---------
        X: 'np.ndarray'
            Current state. This corresponds to, for example, the spliced transcriptomic state.
        Y: 'np.ndarray'
            Velocity estimates in delta t. This corresponds to, for example, the inferred spliced transcriptomic velocity
            or total RNA velocity based on metabolic labeling data estimated calculated by dynamo.
        Grid: 'np.ndarray'
            Current state on a grid which is often used to visualize the vector field. This corresponds to, for example,
            the spliced transcriptomic state or total RNA state.
        M: 'int' (default: 100)
            The number of basis functions to approximate the vector field.
        a: 'float' (default: 10)
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of
            outlier's variation space is `a`.
        beta: 'float' (default: 0.1)
            Paramerter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2).
            If None, a rule-of-thumb bandwidth will be computed automatically.
        ecr: 'float' (default: 1e-5)
            The minimum limitation of energy change rate in the iteration process.
        gamma: 'float' (default: 0.9)
            Percentage of inliers in the samples. This is an initial value for EM iteration, and it is not important.
        lambda_: 'float' (default: 0.3)
            Represents the trade-off between the goodness of data fit and regularization. Larger Lambda_ put more weights
            on regularization.
        minP: 'float' (default: 1e-5)
            The posterior probability Matrix P may be singular for matrix inversion. We set the minimum value of P as
            minP.
        MaxIter: 'int' (default: 500)
            Maximum iteration times.
        theta: 'float' (default: 0.75)
            Define how could be an inlier. If the posterior probability of a sample is an inlier is larger than theta,
            then it is regarded as an inlier.
        div_cur_free_kernels: `bool` (default: False)
            A logic flag to determine whether the divergence-free or curl-free kernels will be used for learning the vector
            field.
        sigma: 'int' (default: `0.8`)
            Bandwidth parameter.
        eta: 'int' (default: `0.5`)
            Combination coefficient for the divergence-free or the curl-free kernels.
        lstsq_method: 'str' (default: `drouin`)
           The name of the linear least square solver, can be either 'scipy` or `douin`.
        verbose: `int` (default: `1`)
            The level of printing running information.

    Returns
    -------
    VecFld: 'dict'
        A dictionary which contains:
            X: Current state.
            X_ctrl: Sample control points of current state.
            Y: Velocity estimates in delta t.
            beta: Parameter of the Gaussian Kernel for the kernel matrix (Gram matrix).
            V: Prediction of velocity of X.
            C: Finite set of the coefficients for the
            P: Posterior probability Matrix of inliers.
            VFCIndex: Indexes of inliers found by sparseVFC.
            sigma2: Energy change rate.
            grid: Grid of current state.
            grid_V: Prediction of velocity of the grid.
            iteration: Number of the last iteration.
            tecr_vec: Vector of relative energy changes rate comparing to previous step.
            E_traj: Vector of energy at each iteration,
        where V = f(X), P is the posterior probability and VFCIndex is the indexes of inliers found by sparseVFC.
        Note that V = `con_K(Grid, X_ctrl, beta).dot(C)` gives the prediction of velocity on Grid (but can also be any
        point in the gene expression state space).

    """

    timeit_ = True if verbose > 1 else False

    Y[~np.isfinite(Y)] = 0  # set nan velocity to 0.
    N, D = Y.shape
    grid_U = None

    # Construct kernel matrix K
    tmp_X, uid = np.unique(X, axis=0, return_index=True)  # return unique rows
    M = min(M, tmp_X.shape[0])
    if velocity_based_sampling:
        if verbose > 1:
            print('Sampling control points based on data velocity magnitude...')
        idx = sample_by_velocity(Y[uid], M)
    else:
        idx = np.random.RandomState(seed=0).permutation(
            tmp_X.shape[0]
        )  # rand select some initial points
        idx = idx[range(M)]
    ctrl_pts = tmp_X[idx, :]

    if beta is None:
        h = bandwidth_selector(ctrl_pts)
        beta = 1/h**2

    K = (
        con_K(ctrl_pts, ctrl_pts, beta, timeit=timeit_)
        if div_cur_free_kernels is False
        else con_K_div_cur_free(ctrl_pts, ctrl_pts, sigma, eta, timeit=timeit_)[0]
    )
    U = (
        con_K(X, ctrl_pts, beta, timeit=timeit_)
        if div_cur_free_kernels is False
        else con_K_div_cur_free(X, ctrl_pts, sigma, eta, timeit=timeit_)[0]
    )
    if Grid is not None:
        grid_U = (
            con_K(Grid, ctrl_pts, beta, timeit=timeit_)
            if div_cur_free_kernels is False
            else con_K_div_cur_free(Grid, ctrl_pts, sigma, eta, timeit=timeit_)[0]
        )
    M = ctrl_pts.shape[0]*D if div_cur_free_kernels else ctrl_pts.shape[0]

    if div_cur_free_kernels:
        X = X.flatten()[:, None]
        Y = Y.flatten()[:, None]

    # Initialization
    V = X.copy() if div_cur_free_kernels else np.zeros((N, D))
    C = np.zeros((M, 1)) if div_cur_free_kernels else np.zeros((M, D))
    i, tecr, E = 0, 1, 1
    sigma2 = sum(sum((Y - X) ** 2)) / (N * D) if div_cur_free_kernels else sum(sum((Y - V) ** 2)) / (N * D)  ## test this
    # sigma2 = 1e-7 if sigma2 > 1e-8 else sigma2
    tecr_vec = np.ones(MaxIter) * np.nan
    E_vec = np.ones(MaxIter) * np.nan

    while i < MaxIter and tecr > ecr and sigma2 > 1e-8:
        # E_step
        E_old = E
        P, E = get_P(Y, V, sigma2, gamma, a, div_cur_free_kernels)

        E = E + lambda_ / 2 * np.trace(C.T.dot(K).dot(C))
        E_vec[i] = E
        tecr = abs((E - E_old) / E)
        tecr_vec[i] = tecr

        if verbose > 1:
            print('\niterate: %d, gamma: %.3f, energy change rate: %s, sigma2=%s'
                %(i, gamma, scinot(tecr, 3), scinot(sigma2, 3)))
        elif verbose > 0:
            print('\niteration %d'%i)

        # M-step. Solve linear system for C.
        if timeit_:
            st = time.time()

        P = np.maximum(P, minP)
        if div_cur_free_kernels:
            P = np.kron(P, np.ones((int(U.shape[0] / P.shape[0]), 1))) # np.kron(P, np.ones((D, 1)))
            lhs = (U.T * np.matlib.tile(P.T, [M, 1])).dot(U) + lambda_ * sigma2 * K
            rhs = (U.T * np.matlib.tile(P.T, [M, 1])).dot(Y)
        else:
            UP = U.T * numpy.matlib.repmat(P.T, M, 1)
            lhs = UP.dot(U) + lambda_ * sigma2 * K
            rhs = UP.dot(Y)

        if timeit_:
            print('Time elapsed for computing lhs and rhs: %f s'%(time.time()-st))
        
        C = lstsq_solver(lhs, rhs, method=lstsq_method, timeit=timeit_)
            
        # Update V and sigma**2
        V = U.dot(C)
        Sp = sum(P) / 2 if div_cur_free_kernels else sum(P)
        sigma2 = (sum(P.T.dot(np.sum((Y - V) ** 2, 1))) / np.dot(Sp, D))[0]

        # Update gamma
        numcorr = len(np.where(P > theta)[0])
        gamma = numcorr / X.shape[0]

        if gamma > 0.95:
            gamma = 0.95
        elif gamma < 0.05:
            gamma = 0.05

        i += 1

    grid_V = None
    if Grid is not None:
        grid_V = np.dot(grid_U, C)

    VecFld = {
        "X": X.reshape((N, D)) if div_cur_free_kernels else X,
        "X_ctrl": ctrl_pts,
        "Y": Y.reshape((N, D)) if div_cur_free_kernels else Y,
        "beta": beta,
        "V": V.reshape((N, D)) if div_cur_free_kernels else V,
        "C": C,
        "P": P,
        "VFCIndex": np.where(P > theta)[0],
        "sigma2": sigma2,
        "grid": Grid,
        "grid_V": grid_V,
        "iteration": i - 1,
        "tecr_traj": tecr_vec[:i],
        "E_traj": E_vec[:i]
    }
    if div_cur_free_kernels:
        VecFld['div_cur_free_kernels'], VecFld['sigma'], VecFld['eta'] = True, sigma, eta
        _, VecFld['df_kernel'], VecFld['cf_kernel'], = con_K_div_cur_free(X, ctrl_pts, sigma, eta, timeit=timeit_)

    return VecFld


class vectorfield:
    def __init__(
        self,
        X=None,
        V=None,
        Grid=None,
        **kwargs
    ):
        """Initialize the VectorField class.

        Parameters
        ----------
        X: 'np.ndarray' (dimension: n_obs x n_features)
                Original data.
        V: 'np.ndarray' (dimension: n_obs x n_features)
                Velocities of cells in the same order and dimension of X.
        Grid: 'np.ndarray'
                The function that returns diffusion matrix which can be dependent on the variables (for example, genes)
        M: 'int' (default: 100)
                The number of basis functions to approximate the vector field.
        a: `float` (default 5)
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's
            variation space is a.
        beta: `float` (default: None)
             Paramerter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2).
             If None, a rule-of-thumb bandwidth will be computed automatically.
        ecr: `float` (default: 1e-5)
            The minimum limitation of energy change rate in the iteration process.
        gamma: `float` (default:  0.9)
            Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
            Default value is 0.9.
        lambda_: `float` (default: 3)
            Represents the trade-off between the goodness of data fit and regularization.
        minP: `float` (default: 1e-5)
            The posterior probability Matrix P may be singular for matrix inversion. We set the minimum value of P as minP.
        MaxIter: `int` (default: 500)
            Maximum iterition times.
        theta: `float` (default 0.75)
            Define how could be an inlier. If the posterior probability of a sample is an inlier is larger than theta, then
            it is regarded as an inlier.
        div_cur_free_kernels: `bool` (default: False)
            A logic flag to determine whether the divergence-free or curl-free kernels will be used for learning the vector
            field.
        sigma: 'int'
            Bandwidth parameter.
        eta: 'int'
            Combination coefficient for the divergence-free or the curl-free kernels.
        """

        self.data = {"X": X, "V": V, "Grid": Grid}
        self.parameters = kwargs

        self.parameters = update_n_merge_dict(self.parameters, {
            "M": kwargs.pop('M', None) or int(0.05 * len(X)) + 1,
            "a": kwargs.pop('a', 5),
            "beta": kwargs.pop('beta', None),
            "ecr": kwargs.pop('ecr', 1e-5),
            "gamma": kwargs.pop('gamma', 0.9),
            "lambda_": kwargs.pop('lambda_', 3),
            "minP": kwargs.pop('minP', 1e-5),
            "MaxIter": kwargs.pop('MaxIter', 500),
            "theta": kwargs.pop('theta', 0.75),
            "div_cur_free_kernels": kwargs.pop('div_cur_free_kernels', False),
            "velocity_based_sampling": kwargs.pop('velocity_based_sampling', True),
            "sigma": kwargs.pop('sigma', 0.8),
            "eta": kwargs.pop('eta', 0.5),
        })

        self.norm_dict = {}
        self.vf_dict = {}
        self.func = None

    def fit(self, normalize=False, method="SparseVFC", **kwargs):
        """Learn an function of vector field from sparse single cell samples in the entire space robustly.
        Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition

        Arguments
        ---------
            normalize: 'bool' (default: False)
                Logic flag to determine whether to normalize the data to have zero means and unit covariance. This is often
                required for raw dataset (for example, raw UMI counts and RNA velocity values in high dimension). But it is
                normally not required for low dimensional embeddings by PCA or other non-linear dimension reduction methods.
            method: 'string'
                Method that is used to reconstruct the vector field functionally. Currently only SparseVFC supported but other
                improved approaches are under development.

        Returns
        -------
            VecFld: `dict'
                A dictionary which contains X, Y, beta, V, C, P, VFCIndex. Where V = f(X), P is the posterior probability and
                VFCIndex is the indexes of inliers which found by VFC.
        """

        if normalize:
            X_old, V_old, T_old, norm_dict = norm(self.data["X"], self.data["V"], self.data["Grid"])
            self.data["X"], self.data["V"], self.data["Grid"], self.norm_dict = (
                X_old,
                V_old,
                T_old,
                norm_dict,
            )

        verbose = kwargs.pop('verbose', 0)
        lstsq_method = kwargs.pop('lstsq_method', 'drouin')
        if method == "SparseVFC":
            VecFld = SparseVFC(
                self.data["X"],
                self.data["V"],
                self.data["Grid"],
                **self.parameters,
                verbose=verbose,
                lstsq_method=lstsq_method,
            )
            if normalize:
                VecFld = denorm(VecFld, X_old, V_old, self.norm_dict)

        self.parameters = update_dict(self.parameters, VecFld)

        self.vf_dict = {
            "VecFld": VecFld,
            "parameters": self.parameters
        }

        self.func = lambda x: vector_field_function(x, VecFld)
        return self.vf_dict


    def plot_energy(self, figsize=None, fig=None):
        from ..plot.scVectorField import plot_energy
        plot_energy(self.vf_dict, figsize, fig)


    def compute_divergence(self, X, timeit=False):
        return compute_divergence(self.func, X, timeit=timeit)


    def get_Jacobian(self):
        return nda.Jacobian(self.func)


    def evaluate(self, CorrectIndex, VFCIndex, siz):
        """Evaluate the precision, recall, corrRate of the sparseVFC algorithm.

        Arguments
        ---------
            CorrectIndex: 'List'
                Ground truth indexes of the correct vector field samples.
            VFCIndex: 'List'
                Indexes of the correct vector field samples learned by VFC.
            siz: 'int'
                Number of initial matches.

        Returns
        -------
        A tuple of precision, recall, corrRate:
        Precision, recall, corrRate: Precision and recall of VFC, percentage of initial correct matches.

        See also:: :func:`sparseVFC`.
        """

        if len(VFCIndex) == 0:
            VFCIndex = range(siz)

        VFCCorrect = np.intersect1d(VFCIndex, CorrectIndex)
        NumCorrectIndex = len(CorrectIndex)
        NumVFCIndex = len(VFCIndex)
        NumVFCCorrect = len(VFCCorrect)

        corrRate = NumCorrectIndex / siz
        precision = NumVFCCorrect / NumVFCIndex
        recall = NumVFCCorrect / NumCorrectIndex

        print(
            "correct correspondence rate in the original data: %d/%d = %f"
            % (NumCorrectIndex, siz, corrRate)
        )
        print("precision rate: %d/%d = %f" % (NumVFCCorrect, NumVFCIndex, precision))
        print("recall rate: %d/%d = %f" % (NumVFCCorrect, NumCorrectIndex, recall))

        return corrRate, precision, recall

