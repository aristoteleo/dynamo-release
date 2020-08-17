from tqdm import tqdm
import numpy.matlib
from numpy import format_float_scientific as scinot
import numpy as np
import scipy.sparse as sp
from scipy.linalg import lstsq
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from multiprocessing.dummy import Pool as ThreadPool
import itertools, functools
import warnings
import time
from ..tools.sampling import sample_by_velocity
from ..tools.utils import (
    update_dict,
    update_n_merge_dict,
    linear_least_squares,
    timeit,
    index_condensed_matrix,
)
from .utils_vecCalc import (
    vector_field_function,
    con_K_div_cur_free,
    con_K,
    Jacobian_numerical,
    compute_divergence,
    compute_curl,
    compute_acceleration,
    compute_curvature,
    compute_torsion,
    Jacobian_rkhs_gaussian,
    Jacobian_rkhs_gaussian_parallel,
    vecfld_from_adata,
)

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
        This function computes an empirical bandwidth for a Gaussian kernel.
    '''
    n, m = X.shape
    if n > 200000 and m > 2: 
        from pynndescent import NNDescent

        nbrs = NNDescent(X, metric='euclidean', n_neighbors=max(2, int(0.2*n)), n_jobs=-1, random_state=19491001)
        _, distances = nbrs.query(X, k=max(2, int(0.2*n)))
    else:
        alg = 'ball_tree' if X.shape[1] > 10 else 'kd_tree'
        nbrs = NearestNeighbors(n_neighbors=max(2, int(0.2*n)), algorithm=alg, n_jobs=-1).fit(X)
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
def graphize_vecfld(func, X, nbrs_idx=None, dist=None, k=30, distance_free=True, n_int_steps=20, cores=1):
    n, d = X.shape

    nbrs = None
    if nbrs_idx is None:
        if X.shape[0] > 200000 and X.shape[1] > 2: 
            from pynndescent import NNDescent

            nbrs = NNDescent(X, metric='euclidean', n_neighbors=k+1, n_jobs=-1, random_state=19491001)
            nbrs_idx, dist = nbrs.query(X, k=k+1)
        else:
            alg = 'ball_tree' if X.shape[1] > 10 else 'kd_tree'
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm=alg, n_jobs=-1).fit(X)
            dist, nbrs_idx = nbrs.kneighbors(X)

    if dist is None and not distance_free:
        D = pdist(X)
    else:
        D = None

    V = sp.csr_matrix((n, n))
    if cores == 1:
        for i, idx in tqdm(enumerate(nbrs_idx), desc='Constructing diffusion graph from reconstructed vector field'):
            V += construct_v(X, i, idx, n_int_steps, func, distance_free, dist, D, n)

    else:
        pool = ThreadPool(cores)
        res = pool.starmap(construct_v, zip(itertools.repeat(X), np.arange(len(nbrs_idx)), nbrs_idx, itertools.repeat(n_int_steps),
                                            itertools.repeat(func), itertools.repeat(distance_free),
                                            itertools.repeat(dist), itertools.repeat(D), itertools.repeat(n)))
        pool.close()
        pool.join()
        V = functools.reduce((lambda a, b: a + b), res)

    return V, nbrs


def construct_v(X, i, idx, n_int_steps, func, distance_free, dist, D, n):
    """helper function for parallism"""

    V = sp.csr_matrix((n, n))
    x = X[i].A if sp.issparse(X) else X[i]
    Y = X[idx[1:]].A if sp.issparse(X) else X[idx[1:]]
    for j, y in enumerate(Y):
        pts = np.linspace(x, y, n_int_steps)
        v = func(pts)

        u = (y - x) / np.linalg.norm(y - x)
        v = np.mean(v.dot(u))
        if not distance_free:
            if dist is None:
                d = D[index_condensed_matrix(n, i, idx[j + 1])]
            else:
                d = dist[i][j + 1]
            v *= d
        V[i, idx[j + 1]] = v
        V[idx[j + 1], i] = -v

    return V


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
    seed=0,
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
            Parameter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of
            outlier's variation space is `a`.
        beta: 'float' (default: 0.1)
            Parameter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2).
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
        seed : int or 1-d array_like, optional (default: `0`)
            Seed for RandomState. Must be convertible to 32 bit unsigned integers. Used in sampling control points. Default
            is to be 0 for ensure consistency between different runs.
        lstsq_method: 'str' (default: `drouin`)
           The name of the linear least square solver, can be either 'scipy` or `douin`.
        verbose: `int` (default: `1`)
            The level of printing running information.

    Returns
    -------
    VecFld: 'dict'
        A dictionary which contains:
            X: Current state.
            valid_ind: The indices of cells that have finite velocity values.
            X_ctrl: Sample control points of current state.
            ctrl_idx: Indices for the sampled control points.
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

    X_ori, Y_ori = X.copy(), Y.copy()
    valid_ind = np.where(np.isfinite(Y.sum(1)))[0]
    X, Y = X[valid_ind], Y[valid_ind]
    N, D = Y.shape
    grid_U = None

    # Construct kernel matrix K
    tmp_X, uid = np.unique(X, axis=0, return_index=True)  # return unique rows
    M = min(M, tmp_X.shape[0])
    if velocity_based_sampling:
        np.random.seed(seed)
        if verbose > 1:
            print('Sampling control points based on data velocity magnitude...')
        idx = sample_by_velocity(Y[uid], M)
    else:
        idx = np.random.RandomState(seed=seed).permutation(
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
        "X": X_ori,
        "valid_ind": valid_ind,
        "X_ctrl": ctrl_pts,
        "ctrl_idx": idx,
        "Y": Y_ori,
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
        M: 'int' (default: None)
            The number of basis functions to approximate the vector field. By default it is calculated as
            `min(len(X), int(1500 * np.log(len(X)) / (np.log(len(X)) + np.log(100))))`. So that any datasets with less
            than  about 900 data points (cells) will use full data for vector field reconstruction while any dataset
            larger than that will at most use 1500 data points.
        a: `float` (default 5)
            Parameter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's
            variation space is a.
        beta: `float` (default: None)
             Parameter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2).
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
            Maximum iteration times.
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
        seed : int or 1-d array_like, optional (default: `0`)
            Seed for RandomState. Must be convertible to 32 bit unsigned integers. Used in sampling control points. Default
            is to be 0 for ensure consistency between different runs.
        """

        
        self.data = {"X": X, "V": V, "Grid": Grid}
        if X is not None and V is not None:
            self.parameters = kwargs
            self.parameters = update_n_merge_dict(self.parameters, {
                "M": kwargs.pop('M', None) or max(min([50, len(X)]), int(0.05 * len(X)) + 1), # min(len(X), int(1500 * np.log(len(X)) / (np.log(len(X)) + np.log(100)))),
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
                "seed": kwargs.pop('seed', 0),
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
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
        self.vf_dict['VecFld']['V'] = self.func(self.data["X"])

        return self.vf_dict


    def plot_energy(self, figsize=None, fig=None):
        from ..plot.scVectorField import plot_energy
        plot_energy(None, vecfld_dict=self.vf_dict, figsize=figsize, fig=fig)


    def integrate(self,
                  init_states,
                  VecFld_true=None,
                  dims=None,
                  scale=1,
                  t_end=None,
                  step_size=None,
                  args=(),
                  integration_direction='forward',
                  interpolation_num=250,
                  average=True,
                  sampling='arc_length',
                  verbose=False,
                  disable=False):

        from ..prediction.utils import (
            getTend,
            getTseq,
            integrate_vf_ivp,
        )

        if np.isscalar(dims):
            init_states = init_states[:, :dims]
        elif dims is not None:
            init_states = init_states[:, dims]

        VecFld = self.vf_dict
        vf = lambda x: scale * vector_field_function(x=x, vf_dict=VecFld,
                                                     dim=dims) if VecFld_true is None else VecFld_true
        if t_end is None: t_end = getTend(self.get_X(), self.get_V())
        t_linspace = getTseq(init_states, t_end, step_size)
        t, prediction = integrate_vf_ivp(init_states,
                               t=t_linspace,
                               args=args,
                               integration_direction=integration_direction,
                               f=vf,
                               interpolation_num=interpolation_num,
                               average=average,
                               sampling=sampling,
                               verbose=verbose,
                               disable=disable,
        )

        return t, prediction


    def compute_divergence(self, X=None, method='analytical', **kwargs):
        X = self.data['X'] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_divergence(f_jac, X, **kwargs)


    def compute_curl(self, X=None, method='analytical', dim1=0, dim2=1, dim3=2, **kwargs):
        X = self.data['X'] if X is None else X
        if dim3 is None or X.shape[1] < 3:
            X = X[:, [dim1, dim2]]
        else:
            X = X[:, [dim1, dim2, dim3]]
        f_jac = self.get_Jacobian(method=method, **kwargs)
        return compute_curl(f_jac, X, **kwargs)


    def compute_acceleration(self, X=None, method='analytical', **kwargs):
        X = self.data['X'] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_acceleration(self.func, f_jac, X, **kwargs)


    def compute_curvature(self, X=None, method='analytical', **kwargs):
        X = self.data['X'] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_curvature(self.func, f_jac, X, **kwargs)


    def compute_torsion(self, X=None, method='analytical', **kwargs):
        X = self.data['X'] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_torsion(self.func, f_jac, X, **kwargs)


    def get_Jacobian(self, method='analytical', input_vector_convention='row', **kwargs):
        '''
            Get the Jacobian of the vector field function.
            If method is 'analytical': 
            The analytical Jacobian will be returned and it always
            take row vectors as input no matter what input_vector_convention is.

            If method is 'numerical':
            If the input_vector_convention is 'row', it means that fjac takes row vectors
            as input, otherwise the input should be an array of column vectors. Note that
            the returned Jacobian would behave exactly the same if the input is an 1d array.

            The column vector convention is slightly faster than the row vector convention.
            So the matrix of row vector convention is converted into column vector convention
            under the hood.

            No matter the method and input vector convention, the returned Jacobian is of the 
            following format:
                    df_1/dx_1   df_1/dx_2   df_1/dx_3   ...
                    df_2/dx_1   df_2/dx_2   df_2/dx_3   ...
                    df_3/dx_1   df_3/dx_2   df_3/dx_3   ...
                    ...         ...         ...         ...
        '''
        if method == 'numerical':
            return Jacobian_numerical(self.func, input_vector_convention, **kwargs)
        elif method == 'parallel':
            return lambda x: Jacobian_rkhs_gaussian_parallel(x, self.vf_dict['VecFld'], **kwargs)
        elif method == 'analytical':
            return lambda x: Jacobian_rkhs_gaussian(x, self.vf_dict['VecFld'], **kwargs)
        else:
            raise NotImplementedError(f"The method {method} is not implemented. Currently only "
                                      f"supports 'analytical', 'numerical', and 'parallel'.")


    def construct_graph(self, X=None, **kwargs):
        X = self.data['X'] if X is None else X
        return graphize_vecfld(self.func, X, **kwargs)


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


    def from_adata(self, adata, basis='', vf_key='VecFld'):
        vf_dict, func = vecfld_from_adata(adata, basis=basis, vf_key=vf_key)
        self.data['X'] = vf_dict['X']
        self.data['V'] = vf_dict['V']
        self.vf_dict['VecFld'] = vf_dict
        self.func = func


    def get_X(self):
        return self.data['X']


    def get_V(self):
        return self.data['V']


    def get_data(self):
        return self.data['X'], self.data['V']
