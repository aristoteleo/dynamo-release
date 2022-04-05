import functools
import itertools
import warnings
from multiprocessing.dummy import Pool as ThreadPool
from typing import Callable, Union

import numpy as np
import numpy.matlib
import scipy.sparse as sp
from numpy import format_float_scientific as scinot
from scipy.linalg import lstsq
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

from ..dynamo_logger import LoggerManager, main_warning
from ..tools.sampling import sample, sample_by_velocity
from ..tools.utils import (
    index_condensed_matrix,
    linear_least_squares,
    nearest_neighbors,
    starmap_with_kwargs,
    timeit,
    update_dict,
    update_n_merge_dict,
)
from .utils import (
    FixedPoints,
    Jacobian_kovf,
    Jacobian_numerical,
    Jacobian_rkhs_gaussian,
    Jacobian_rkhs_gaussian_parallel,
    compute_acceleration,
    compute_curl,
    compute_curvature,
    compute_divergence,
    compute_sensitivity,
    compute_torsion,
    con_K,
    con_K_div_cur_free,
    find_fixed_points,
    remove_redundant_points,
    vecfld_from_adata,
    vector_field_function,
    vector_transformation,
)


def norm(X, V, T, fix_velocity=True):
    """Normalizes the X, Y (X + V) matrix to have zero means and unit covariance.
        We use the mean of X, Y's center (mean) and scale parameters (standard deviation) to normalize T.

    Arguments
    ---------
        X: :class:`~numpy.ndarray`
            Current state. This corresponds to, for example, the spliced transcriptomic state.
        V: :class:`~numpy.ndarray`
            Velocity estimates in delta t. This corresponds to, for example, the inferred spliced transcriptomic
            velocity estimated calculated by dynamo or velocyto, scvelo.
        T: :class:`~numpy.ndarray`
            Current state on a grid which is often used to visualize the vector field. This corresponds to, for example,
            the spliced transcriptomic state.
        fix_velocity: bool (default: `True`)
            Whether to fix velocity and don't transform it.

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
        T - (1 / 2 * (xm[None, :] + ym[None, :])) if T is not None else None,
    )

    xscale, yscale = (
        np.sqrt(np.sum(np.sum(x ** 2, 1)) / n),
        np.sqrt(np.sum(np.sum(y ** 2, 1)) / m),
    )

    X, Y, T = x / xscale, y / yscale, t / (1 / 2 * (xscale + yscale)) if T is not None else None

    X, V, T = X, V if fix_velocity else Y - X, T
    norm_dict = {"xm": xm, "ym": ym, "xscale": xscale, "yscale": yscale, "fix_velocity": fix_velocity}

    return X, V, T, norm_dict


def bandwidth_rule_of_thumb(X, return_sigma=False):
    """
    This function computes a rule-of-thumb bandwidth for a Gaussian kernel based on:
    https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
    """
    sig = sig = np.sqrt(np.mean(np.diag(np.cov(X.T))))
    h = 1.06 * sig / (len(X) ** (-1 / 5))
    if return_sigma:
        return h, sig
    else:
        return h


def bandwidth_selector(X):
    """
    This function computes an empirical bandwidth for a Gaussian kernel.
    """
    n, m = X.shape
    if n > 200000 and m > 2:
        from pynndescent import NNDescent

        nbrs = NNDescent(
            X,
            metric="euclidean",
            n_neighbors=max(2, int(0.2 * n)),
            n_jobs=-1,
            random_state=19491001,
        )
        _, distances = nbrs.query(X, k=max(2, int(0.2 * n)))
    else:
        alg = "ball_tree" if X.shape[1] > 10 else "kd_tree"
        nbrs = NearestNeighbors(n_neighbors=max(2, int(0.2 * n)), algorithm=alg, n_jobs=-1).fit(X)
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
    X, Y, V, xm, ym, x_scale, y_scale, fix_velocity = (
        VecFld["X"],
        VecFld["Y"],
        VecFld["V"],
        norm_dict["xm"],
        norm_dict["ym"],
        norm_dict["xscale"],
        norm_dict["yscale"],
        norm_dict["fix_velocity"],
    )
    grid, grid_V = VecFld["grid"], VecFld["grid_V"]
    xy_m, xy_scale = (xm + ym) / 2, (x_scale + y_scale) / 2

    VecFld["X"] = X_old
    VecFld["Y"] = Y_old
    # VecFld["X_ctrl"] = X * x_scale + np.matlib.tile(xm, [X.shape[0], 1])
    VecFld["grid"] = grid * xy_scale + np.matlib.tile(xy_m, [grid.shape[0], 1]) if grid is not None else None
    VecFld["grid_V"] = (
        (grid + grid_V) * xy_scale + np.matlib.tile(xy_m, [grid_V.shape[0], 1]) - grid if grid_V is not None else None
    )
    VecFld["V"] = V if fix_velocity else (V + X) * y_scale + np.matlib.tile(ym, [V.shape[0], 1]) - X_old
    VecFld["norm_dict"] = norm_dict

    return VecFld


@timeit
def lstsq_solver(lhs, rhs, method="drouin"):
    if method == "scipy":
        C = lstsq(lhs, rhs)[0]
    elif method == "drouin":
        C = linear_least_squares(lhs, rhs)
    else:
        main_warning("Invalid linear least squares solver. Use Drouin's method instead.")
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
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of
            outlier's variation space is a.
        div_cur_free_kernels: `bool` (default: False)
            A logic flag to determine whether the divergence-free or curl-free kernels will be used for learning the
            vector field.

    Returns
    -------
    P: 'np.ndarray'
        Posterior probability, related to equation 27.
    E: `np.ndarray'
        Energy, related to equation 26.

    """

    if div_cur_free_kernels:
        Y = Y.reshape((2, int(Y.shape[0] / 2)), order="F").T
        V = V.reshape((2, int(V.shape[0] / 2)), order="F").T

    D = Y.shape[1]
    temp1 = np.exp(-np.sum((Y - V) ** 2, 1) / (2 * sigma2))
    temp2 = (2 * np.pi * sigma2) ** (D / 2) * (1 - gamma) / (gamma * a)
    temp1[temp1 == 0] = np.min(temp1[temp1 != 0])
    P = temp1 / (temp1 + temp2)
    E = P.T.dot(np.sum((Y - V) ** 2, 1)) / (2 * sigma2) + np.sum(P) * np.log(sigma2) * D / 2

    return (P[:, None], E) if P.ndim == 1 else (P, E)


@timeit
def graphize_vecfld(
    func,
    X,
    nbrs_idx=None,
    dist=None,
    k=30,
    distance_free=True,
    n_int_steps=20,
    cores=1,
):
    n, d = X.shape

    nbrs = None
    if nbrs_idx is None:
        if X.shape[0] > 200000 and X.shape[1] > 2:
            from pynndescent import NNDescent

            nbrs = NNDescent(
                X,
                metric="euclidean",
                n_neighbors=k + 1,
                n_jobs=-1,
                random_state=19491001,
            )
            nbrs_idx, dist = nbrs.query(X, k=k + 1)
        else:
            alg = "ball_tree" if X.shape[1] > 10 else "kd_tree"
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm=alg, n_jobs=-1).fit(X)
            dist, nbrs_idx = nbrs.kneighbors(X)

    if dist is None and not distance_free:
        D = pdist(X)
    else:
        D = None

    V = sp.csr_matrix((n, n))
    if cores == 1:
        for i, idx in enumerate(LoggerManager.progress_logger(nbrs_idx, progress_name="graphize_vecfld")):
            V += construct_v(X, i, idx, n_int_steps, func, distance_free, dist, D, n)

    else:
        pool = ThreadPool(cores)
        res = pool.starmap(
            construct_v,
            zip(
                itertools.repeat(X),
                np.arange(len(nbrs_idx)),
                nbrs_idx,
                itertools.repeat(n_int_steps),
                itertools.repeat(func),
                itertools.repeat(distance_free),
                itertools.repeat(dist),
                itertools.repeat(D),
                itertools.repeat(n),
            ),
        )
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

        lxy = np.linalg.norm(y - x)
        if lxy > 0:
            u = (y - x) / np.linalg.norm(y - x)
        else:
            u = y - x
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
    X: np.ndarray,
    Y: np.ndarray,
    Grid: np.ndarray,
    M: int = 100,
    a: float = 5,
    beta: float = None,
    ecr: float = 1e-5,
    gamma: float = 0.9,
    lambda_: float = 3,
    minP: float = 1e-5,
    MaxIter: int = 500,
    theta: float = 0.75,
    div_cur_free_kernels: bool = False,
    velocity_based_sampling: bool = True,
    sigma: float = 0.8,
    eta: float = 0.5,
    seed=0,
    lstsq_method: str = "drouin",
    verbose: int = 1,
) -> dict:
    """Apply sparseVFC (vector field consensus) algorithm to learn a functional form of the vector field from random
    samples with outlier on the entire space robustly and efficiently. (Ma, Jiayi, etc. al, Pattern Recognition, 2013)

    Arguments
    ---------
        X: 'np.ndarray'
            Current state. This corresponds to, for example, the spliced transcriptomic state.
        Y: 'np.ndarray'
            Velocity estimates in delta t. This corresponds to, for example, the inferred spliced transcriptomic
            velocity or total RNA velocity based on metabolic labeling data estimated calculated by dynamo.
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
            Represents the trade-off between the goodness of data fit and regularization. Larger Lambda_ put more
            weights on regularization.
        minP: 'float' (default: 1e-5)
            The posterior probability Matrix P may be singular for matrix inversion. We set the minimum value of P as
            minP.
        MaxIter: 'int' (default: 500)
            Maximum iteration times.
        theta: 'float' (default: 0.75)
            Define how could be an inlier. If the posterior probability of a sample is an inlier is larger than theta,
            then it is regarded as an inlier.
        div_cur_free_kernels: `bool` (default: False)
            A logic flag to determine whether the divergence-free or curl-free kernels will be used for learning the
            vector field.
        sigma: 'int' (default: `0.8`)
            Bandwidth parameter.
        eta: 'int' (default: `0.5`)
            Combination coefficient for the divergence-free or the curl-free kernels.
        seed : int or 1-d array_like, optional (default: `0`)
            Seed for RandomState. Must be convertible to 32 bit unsigned integers. Used in sampling control points.
            Default is to be 0 for ensure consistency between different runs.
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
    logger = LoggerManager.gen_logger("SparseVFC")
    temp_logger = LoggerManager.get_temp_timer_logger()
    logger.info("[SparseVFC] begins...")
    logger.log_time()

    need_utility_time_measure = verbose > 1
    X_ori, Y_ori = X.copy(), Y.copy()
    valid_ind = np.where(np.isfinite(Y.sum(1)))[0]
    X, Y = X[valid_ind], Y[valid_ind]
    N, D = Y.shape
    grid_U = None

    # Construct kernel matrix K
    tmp_X, uid = np.unique(X, axis=0, return_index=True)  # return unique rows
    M = min(M, tmp_X.shape[0])
    if velocity_based_sampling:
        logger.info("Sampling control points based on data velocity magnitude...")
        idx = sample_by_velocity(Y[uid], M, seed=seed)
    else:
        idx = np.random.RandomState(seed=seed).permutation(tmp_X.shape[0])  # rand select some initial points
        idx = idx[range(M)]
    ctrl_pts = tmp_X[idx, :]

    if beta is None:
        h = bandwidth_selector(ctrl_pts)
        beta = 1 / h ** 2

    K = (
        con_K(ctrl_pts, ctrl_pts, beta, timeit=need_utility_time_measure)
        if div_cur_free_kernels is False
        else con_K_div_cur_free(ctrl_pts, ctrl_pts, sigma, eta, timeit=need_utility_time_measure)[0]
    )
    U = (
        con_K(X, ctrl_pts, beta, timeit=need_utility_time_measure)
        if div_cur_free_kernels is False
        else con_K_div_cur_free(X, ctrl_pts, sigma, eta, timeit=need_utility_time_measure)[0]
    )
    if Grid is not None:
        grid_U = (
            con_K(Grid, ctrl_pts, beta, timeit=need_utility_time_measure)
            if div_cur_free_kernels is False
            else con_K_div_cur_free(Grid, ctrl_pts, sigma, eta, timeit=need_utility_time_measure)[0]
        )
    M = ctrl_pts.shape[0] * D if div_cur_free_kernels else ctrl_pts.shape[0]

    if div_cur_free_kernels:
        X = X.flatten()[:, None]
        Y = Y.flatten()[:, None]

    # Initialization
    V = X.copy() if div_cur_free_kernels else np.zeros((N, D))
    C = np.zeros((M, 1)) if div_cur_free_kernels else np.zeros((M, D))
    i, tecr, E = 0, 1, 1
    # test this
    sigma2 = sum(sum((Y - X) ** 2)) / (N * D) if div_cur_free_kernels else sum(sum((Y - V) ** 2)) / (N * D)
    sigma2 = 1e-7 if sigma2 < 1e-8 else sigma2
    tecr_vec = np.ones(MaxIter) * np.nan
    E_vec = np.ones(MaxIter) * np.nan
    P = None
    while i < MaxIter and tecr > ecr and sigma2 > 1e-8:
        # E_step
        E_old = E
        P, E = get_P(Y, V, sigma2, gamma, a, div_cur_free_kernels)

        E = E + lambda_ / 2 * np.trace(C.T.dot(K).dot(C))
        E_vec[i] = E
        tecr = abs((E - E_old) / E)
        tecr_vec[i] = tecr

        # logger.report_progress(count=i, total=MaxIter, progress_name="E-step iteration")
        if need_utility_time_measure:
            logger.info(
                "iterate: %d, gamma: %.3f, energy change rate: %s, sigma2=%s"
                % (i, gamma, scinot(tecr, 3), scinot(sigma2, 3))
            )

        # M-step. Solve linear system for C.
        temp_logger.log_time()
        P = np.maximum(P, minP)
        if div_cur_free_kernels:
            P = np.kron(P, np.ones((int(U.shape[0] / P.shape[0]), 1)))  # np.kron(P, np.ones((D, 1)))
            lhs = (U.T * np.matlib.tile(P.T, [M, 1])).dot(U) + lambda_ * sigma2 * K
            rhs = (U.T * np.matlib.tile(P.T, [M, 1])).dot(Y)
        else:
            UP = U.T * numpy.matlib.repmat(P.T, M, 1)
            lhs = UP.dot(U) + lambda_ * sigma2 * K
            rhs = UP.dot(Y)
        if need_utility_time_measure:
            temp_logger.finish_progress(progress_name="computing lhs and rhs")
        temp_logger.log_time()

        C = lstsq_solver(lhs, rhs, method=lstsq_method, timeit=need_utility_time_measure)

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
    if i == 0 and not (tecr > ecr and sigma2 > 1e-8):
        raise Exception(
            "please check your input parameters, "
            f"tecr: {tecr}, ecr {ecr} and sigma2 {sigma2},"
            f"tecr must larger than ecr and sigma2 must larger than 1e-8"
        )

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
        "E_traj": E_vec[:i],
    }
    if div_cur_free_kernels:
        VecFld["div_cur_free_kernels"], VecFld["sigma"], VecFld["eta"] = (
            True,
            sigma,
            eta,
        )
        temp_logger.log_time()
        (
            _,
            VecFld["df_kernel"],
            VecFld["cf_kernel"],
        ) = con_K_div_cur_free(X, ctrl_pts, sigma, eta, timeit=need_utility_time_measure)
        temp_logger.finish_progress(progress_name="con_K_div_cur_free")

    logger.finish_progress(progress_name="SparseVFC")
    return VecFld


class BaseVectorField:
    def __init__(
        self,
        X=None,
        V=None,
        Grid=None,
        *args,
        **kwargs,
    ):
        self.data = {"X": X, "V": V, "Grid": Grid}
        self.vf_dict = kwargs.pop("vf_dict", {})
        self.func = kwargs.pop("func", None)
        self.fixed_points = kwargs.pop("fixed_points", None)
        super().__init__(**kwargs)

    def construct_graph(self, X=None, **kwargs):
        X = self.data["X"] if X is None else X
        return graphize_vecfld(self.func, X, **kwargs)

    def from_adata(self, adata, basis="", vf_key="VecFld"):
        vf_dict, func = vecfld_from_adata(adata, basis=basis, vf_key=vf_key)
        self.data["X"] = vf_dict["X"]
        self.data["V"] = vf_dict["Y"]  # use the raw velocity
        self.vf_dict = vf_dict
        self.func = func

    def get_X(self, idx=None):
        if idx is None:
            return self.data["X"]
        else:
            return self.data["X"][idx]

    def get_V(self, idx=None):
        if idx is None:
            return self.data["V"]
        else:
            return self.data["V"][idx]

    def get_data(self):
        return self.data["X"], self.data["V"]

    def find_fixed_points(self, n_x0=100, X0=None, domain=None, sampling_method="random", **kwargs):
        """
        Search for fixed points of the vector field function.

        """
        if self.data is None and X0 is None:
            raise Exception(
                "The initial points `X0` are not provided, "
                "and no data is stored in the vector field for the sampling of initial points."
            )
        elif X0 is None:
            indices = sample(np.arange(len(self.data["X"])), n_x0, method=sampling_method)
            X0 = self.data["X"][indices]

        if domain is None and self.data is not None:
            domain = np.vstack((np.min(self.data["X"], axis=0), np.max(self.data["X"], axis=0))).T

        X, J, _ = find_fixed_points(X0, self.func, domain=domain, **kwargs)
        self.fixed_points = FixedPoints(X, J)

    def get_fixed_points(self, **kwargs):
        """
        Get fixed points of the vector field function.

        Returns
        -------
            Xss: :class:`~numpy.ndarray`
                Coordinates of the fixed points.
            ftype: :class:`~numpy.ndarray`
                Types of the fixed points:
                -1 -- stable,
                 0 -- saddle,
                 1 -- unstable
        """
        if self.fixed_points is None:
            self.find_fixed_points(**kwargs)

        Xss = self.fixed_points.get_X()
        ftype = self.fixed_points.get_fixed_point_types()
        return Xss, ftype

    def assign_fixed_points(self, domain=None, cores=1, **kwargs):
        """assign each cell to the associated fixed points"""
        if domain is None and self.data is not None:
            domain = np.vstack((np.min(self.data["X"], axis=0), np.max(self.data["X"], axis=0))).T

        if cores == 1:
            X, J, _ = find_fixed_points(
                self.data["X"],
                self.func,
                domain=domain,
                return_all=True,
                **kwargs,
            )
        else:
            pool = ThreadPool(cores)

            args_iter = zip(
                [i[None, :] for i in self.data["X"]],
                itertools.repeat(self.func),
                itertools.repeat(domain),
                itertools.repeat(True),
            )
            kwargs_iter = itertools.repeat(kwargs)
            res = starmap_with_kwargs(pool, find_fixed_points, args_iter, kwargs_iter)

            pool.close()
            pool.join()

            (X, J, _) = zip(*res)
            X = np.vstack([[i] * self.data["X"].shape[1] if i is None else i for i in X]).astype(float)
            J = np.array(
                [np.zeros((self.data["X"].shape[1], self.data["X"].shape[1])) * np.nan if i is None else i for i in J]
            )

        self.fixed_points = FixedPoints(X, J)
        fps_assignment = self.fixed_points.get_X()
        fps_type_assignment = self.fixed_points.get_fixed_point_types()

        valid_fps_assignment, valid_fps_type_assignment = (
            fps_assignment[np.abs(fps_assignment).sum(1) > 0, :],
            fps_type_assignment[np.abs(fps_assignment).sum(1) > 0],
        )
        X, discard = remove_redundant_points(valid_fps_assignment, output_discard=True)

        assignment_id = np.zeros(len(fps_assignment))
        for i, cur_fps in enumerate(fps_assignment):
            if np.isnan(cur_fps).any():
                assignment_id[i] = np.nan
            else:
                assignment_id[i] = int(nearest_neighbors(cur_fps, X, 1))

        return X, valid_fps_type_assignment[discard], assignment_id

    def integrate(
        self,
        init_states,
        VecFld_true=None,
        dims=None,
        scale=1,
        t_end=None,
        step_size=None,
        args=(),
        integration_direction="forward",
        interpolation_num=250,
        average=True,
        sampling="arc_length",
        verbose=False,
        disable=False,
    ):

        from ..prediction.utils import integrate_vf_ivp
        from ..tools.utils import getTend, getTseq

        if np.isscalar(dims):
            init_states = init_states[:, :dims]
        elif dims is not None:
            init_states = init_states[:, dims]

        if self.func is None:
            VecFld = self.vf_dict
            self.func = (
                lambda x: scale * vector_field_function(x=x, vf_dict=VecFld, dim=dims)
                if VecFld_true is None
                else VecFld_true
            )
        if t_end is None:
            t_end = getTend(self.get_X(), self.get_V())
        t_linspace = getTseq(init_states, t_end, step_size)
        t, prediction = integrate_vf_ivp(
            init_states,
            t=t_linspace,
            integration_direction=integration_direction,
            f=self.func,
            args=args,
            interpolation_num=interpolation_num,
            average=average,
            sampling=sampling,
            verbose=verbose,
            disable=disable,
        )

        return t, prediction


class DifferentiableVectorField(BaseVectorField):
    def get_Jacobian(self, method=None):
        # subclasses must implement this function.
        pass

    def compute_divergence(self, X=None, method="analytical", **kwargs):
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_divergence(f_jac, X, **kwargs)

    def compute_curl(self, X=None, method="analytical", dim1=0, dim2=1, dim3=2, **kwargs):
        X = self.data["X"] if X is None else X
        if dim3 is None or X.shape[1] < 3:
            X = X[:, [dim1, dim2]]
        else:
            X = X[:, [dim1, dim2, dim3]]
        f_jac = self.get_Jacobian(method=method, **kwargs)
        return compute_curl(f_jac, X, **kwargs)

    def compute_acceleration(self, X=None, method="analytical", **kwargs):
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_acceleration(self.func, f_jac, X, **kwargs)

    def compute_curvature(self, X=None, method="analytical", formula=2, **kwargs):
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_curvature(self.func, f_jac, X, formula=formula, **kwargs)

    def compute_torsion(self, X=None, method="analytical", **kwargs):
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_torsion(self.func, f_jac, X, **kwargs)

    def compute_sensitivity(self, X=None, method="analytical", **kwargs):
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_sensitivity(f_jac, X, **kwargs)


class SvcVectorField(DifferentiableVectorField):
    def __init__(self, X=None, V=None, Grid=None, *args, **kwargs):
        """Initialize the VectorField class.

        Parameters
        ----------
        X: :class:`~numpy.ndarray` (dimension: n_obs x n_features)
                Original data.
        V: :class:`~numpy.ndarray` (dimension: n_obs x n_features)
                Velocities of cells in the same order and dimension of X.
        Grid: :class:`~numpy.ndarray`
                The function that returns diffusion matrix which can be dependent on the variables (for example, genes)
        M: `int` (default: None)
            The number of basis functions to approximate the vector field. By default it is calculated as
            `min(len(X), int(1500 * np.log(len(X)) / (np.log(len(X)) + np.log(100))))`. So that any datasets with less
            than  about 900 data points (cells) will use full data for vector field reconstruction while any dataset
            larger than that will at most use 1500 data points.
        a: `float` (default 5)
            Parameter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of
            outlier's variation space is a.
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
            The posterior probability Matrix P may be singular for matrix inversion. We set the minimum value of P as
            minP.
        MaxIter: `int` (default: 500)
            Maximum iteration times.
        theta: `float` (default 0.75)
            Define how could be an inlier. If the posterior probability of a sample is an inlier is larger than theta,
            then it is regarded as an inlier.
        div_cur_free_kernels: `bool` (default: False)
            A logic flag to determine whether the divergence-free or curl-free kernels will be used for learning the
            vector field.
        sigma: `int`
            Bandwidth parameter.
        eta: `int`
            Combination coefficient for the divergence-free or the curl-free kernels.
        seed : int or 1-d array_like, optional (default: `0`)
            Seed for RandomState. Must be convertible to 32 bit unsigned integers. Used in sampling control points.
            Default is to be 0 for ensure consistency between different runs.
        """

        super().__init__(X, V, Grid)
        if X is not None and V is not None:
            self.parameters = kwargs
            self.parameters = update_n_merge_dict(
                self.parameters,
                {
                    "M": kwargs.pop("M", None) or max(min([50, len(X)]), int(0.05 * len(X)) + 1),
                    # min(len(X), int(1500 * np.log(len(X)) / (np.log(len(X)) + np.log(100)))),
                    "a": kwargs.pop("a", 5),
                    "beta": kwargs.pop("beta", None),
                    "ecr": kwargs.pop("ecr", 1e-5),
                    "gamma": kwargs.pop("gamma", 0.9),
                    "lambda_": kwargs.pop("lambda_", 3),
                    "minP": kwargs.pop("minP", 1e-5),
                    "MaxIter": kwargs.pop("MaxIter", 500),
                    "theta": kwargs.pop("theta", 0.75),
                    "div_cur_free_kernels": kwargs.pop("div_cur_free_kernels", False),
                    "velocity_based_sampling": kwargs.pop("velocity_based_sampling", True),
                    "sigma": kwargs.pop("sigma", 0.8),
                    "eta": kwargs.pop("eta", 0.5),
                    "seed": kwargs.pop("seed", 0),
                },
            )

        self.norm_dict = {}

    def train(self, normalize=False, **kwargs):
        """Learn an function of vector field from sparse single cell samples in the entire space robustly.
        Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al,
        Pattern Recognition

        Arguments
        ---------
            normalize: 'bool' (default: False)
                Logic flag to determine whether to normalize the data to have zero means and unit covariance. This is
                often required for raw dataset (for example, raw UMI counts and RNA velocity values in high dimension).
                But it is normally not required for low dimensional embeddings by PCA or other non-linear dimension
                reduction methods.
            method: 'string'
                Method that is used to reconstruct the vector field functionally. Currently only SparseVFC supported but
                other improved approaches are under development.

        Returns
        -------
            VecFld: `dict'
                A dictionary which contains X, Y, beta, V, C, P, VFCIndex. Where V = f(X), P is the posterior
                probability and VFCIndex is the indexes of inliers which found by VFC.
        """

        if normalize:
            X_norm, V_norm, T_norm, norm_dict = norm(self.data["X"], self.data["V"], self.data["Grid"])
            (self.data["X"], self.data["V"], self.data["Grid"], self.norm_dict,) = (
                X_norm,
                V_norm,
                T_norm,
                norm_dict,
            )

        verbose = kwargs.pop("verbose", 0)
        lstsq_method = kwargs.pop("lstsq_method", "drouin")
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
            VecFld = denorm(VecFld, X_norm, V_norm, self.norm_dict)

        self.parameters = update_dict(self.parameters, VecFld)

        self.vf_dict = VecFld

        self.func = lambda x: vector_field_function(x, VecFld)
        self.vf_dict["V"] = self.func(self.data["X"])
        self.vf_dict["normalize"] = normalize

        return self.vf_dict

    def plot_energy(self, figsize=None, fig=None):
        from ..plot.scVectorField import plot_energy

        plot_energy(None, vecfld_dict=self.vf_dict, figsize=figsize, fig=fig)

    def get_Jacobian(self, method="analytical", input_vector_convention="row", **kwargs):
        """
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
        """
        if method == "numerical":
            return Jacobian_numerical(self.func, input_vector_convention, **kwargs)
        elif method == "parallel":
            return lambda x: Jacobian_rkhs_gaussian_parallel(x, self.vf_dict, **kwargs)
        elif method == "analytical":
            return lambda x: Jacobian_rkhs_gaussian(x, self.vf_dict, **kwargs)
        else:
            raise NotImplementedError(
                f"The method {method} is not implemented. Currently only "
                f"supports 'analytical', 'numerical', and 'parallel'."
            )

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

        print("correct correspondence rate in the original data: %d/%d = %f" % (NumCorrectIndex, siz, corrRate))
        print("precision rate: %d/%d = %f" % (NumVFCCorrect, NumVFCIndex, precision))
        print("recall rate: %d/%d = %f" % (NumVFCCorrect, NumCorrectIndex, recall))

        return corrRate, precision, recall


class KOVectorField(DifferentiableVectorField):
    def __init__(
        self, X=None, V=None, Grid=None, K=None, func_base=None, fjac_base=None, PCs=None, mean=None, *args, **kwargs
    ):
        super().__init__(X, V, Grid=Grid, *args, **kwargs)

        if K.ndim == 2:
            K = np.diag(K)
        self.K = K
        self.PCs = PCs
        self.mean = mean
        self.func_base = func_base
        self.fjac_base = fjac_base

        if self.K is not None and self.PCs is not None and self.mean is not None and self.func_base is not None:
            self.setup_perturbed_func()

    def setup_perturbed_func(self):
        def vf_func_perturb(x):
            x_gene = np.abs(x @ self.PCs.T + self.mean)
            v_gene = vector_transformation(self.func_base(x), self.PCs)
            v_gene = v_gene - self.K * x_gene
            return v_gene @ self.PCs

        self.func = vf_func_perturb

    def get_Jacobian(self, method="analytical", **kwargs):
        """
        Get the Jacobian of the vector field function.
        If method is 'analytical':
        The analytical Jacobian will be returned and it always
        take row vectors as input no matter what input_vector_convention is.

        No matter the method and input vector convention, the returned Jacobian is of the
        following format:
                df_1/dx_1   df_1/dx_2   df_1/dx_3   ...
                df_2/dx_1   df_2/dx_2   df_2/dx_3   ...
                df_3/dx_1   df_3/dx_2   df_3/dx_3   ...
                ...         ...         ...         ...
        """
        if method == "analytical":
            exact = kwargs.pop("exact", False)
            mu = kwargs.pop("mu", None)
            if exact:
                if mu is None:
                    mu = self.mean
                return lambda x: Jacobian_kovf(x, self.fjac_base, self.K, self.PCs, exact=True, mu=mu, **kwargs)
            else:
                return lambda x: Jacobian_kovf(x, self.fjac_base, self.K, self.PCs, **kwargs)
        elif method == "numerical":
            if self.func is not None:
                return Jacobian_numerical(self.func, **kwargs)
            else:
                raise Exception("The perturbed vector field function has not been set up.")
        else:
            raise NotImplementedError(
                f"The method {method} is not implemented. Currently only " f"supports 'analytical'."
            )


try:
    from dynode.vectorfield import Dynode

    use_dynode = True
except ImportError:
    use_dynode = False

if use_dynode:

    class dynode_vectorfield(BaseVectorField, Dynode):  #
        def __init__(self, X=None, V=None, Grid=None, *args, **kwargs):
            self.norm_dict = {}

            if X is not None and V is not None:
                self.parameters = update_n_merge_dict(kwargs, {"X": X, "V": V, "Grid": Grid})

                import tempfile

                from dynode.vectorfield import networkModels
                from dynode.vectorfield.losses_weighted import (  # MAD, BinomialChannel, WassersteinDistance, CosineDistance
                    MSE,
                )
                from dynode.vectorfield.samplers import VelocityDataSampler

                good_ind = np.where(~np.isnan(V.sum(1)))[0]
                good_V = V[good_ind, :]
                good_X = X[good_ind, :]

                self.valid_ind = good_ind

                velocity_data_sampler = VelocityDataSampler(
                    adata={"X": good_X, "V": good_V},
                    normalize_velocity=kwargs.get("normalize_velocity", False),
                )

                vf_kwargs = {
                    "X": X,
                    "V": V,
                    "Grid": Grid,
                    "model": networkModels,
                    "sirens": False,
                    "enforce_positivity": False,
                    "velocity_data_sampler": velocity_data_sampler,
                    "time_course_data_sampler": None,
                    "network_dim": X.shape[1],
                    "velocity_loss_function": MSE(),  # CosineDistance(), # #MSE(), MAD()
                    "time_course_loss_function": None,  # BinomialChannel(p=0.1, alpha=1)
                    "velocity_x_initialize": X,
                    "time_course_x0_initialize": None,
                    "smoothing_factor": None,
                    "stability_factor": None,
                    "load_model_from_buffer": False,
                    "buffer_path": tempfile.mkdtemp(),
                    "hidden_features": 256,
                    "hidden_layers": 3,
                    "first_omega_0": 30.0,
                    "hidden_omega_0": 30.0,
                }
                vf_kwargs = update_dict(vf_kwargs, self.parameters)
                super().__init__(**vf_kwargs)

        def train(self, **kwargs):
            if len(kwargs) > 0:
                self.parameters = update_n_merge_dict(self.parameters, kwargs)
            max_iter = 2 * 100000 * np.log(self.data["X"].shape[0]) / (250 + np.log(self.data["X"].shape[0]))
            train_kwargs = {
                "max_iter": int(max_iter),
                "velocity_batch_size": 50,
                "time_course_batch_size": 100,
                "autoencoder_batch_size": 50,
                "velocity_lr": 1e-4,
                "velocity_x_lr": 0,
                "time_course_lr": 1e-4,
                "time_course_x0_lr": 1e4,
                "autoencoder_lr": 1e-4,
                "velocity_sample_fraction": 1,
                "time_course_sample_fraction": 1,
                "iter_per_sample_update": None,
            }
            train_kwargs = update_dict(train_kwargs, kwargs)

            super().train(**train_kwargs)

            self.func = self.predict_velocity
            self.vf_dict = {
                "X": self.data["X"],
                "valid_ind": self.valid_ind,
                "Y": self.data["V"],
                "V": self.func(self.data["X"]),
                "grid": self.data["Grid"],
                "grid_V": self.func(self.data["Grid"]),
                "iteration": self.parameters.pop("max_iter", int(max_iter)),
                "velocity_loss_traj": self.velocity_loss_traj,
                "time_course_loss_traj": self.time_course_loss_traj,
                "autoencoder_loss_traj": self.autoencoder_loss_traj,
                "parameters": self.parameters,
            }

            return self.vf_dict


def vector_field_function_knockout(
    adata,
    vecfld: Union[Callable, BaseVectorField],
    ko_genes,
    k_deg=None,
    pca_genes="use_for_pca",
    PCs="PCs",
    mean="pca_mean",
    return_vector_field_class=True,
):

    if type(pca_genes) is str:
        pca_genes = adata.var[adata.var[pca_genes]].index

    g_mask = np.zeros(len(pca_genes), dtype=bool)
    for i, g in enumerate(pca_genes):
        if g in ko_genes:
            g_mask[i] = True
    if g_mask.sum() != len(ko_genes):
        raise ValueError(f"the ko_genes {ko_genes} you provided don't all belong to {pca_genes}.")

    k = np.zeros(len(pca_genes))
    if k_deg is None:
        k_deg = np.ones(len(ko_genes))
    k[g_mask] = k_deg

    if type(PCs) is str:
        if PCs not in adata.uns.keys():
            raise Exception(f"The key {PCs} is not in `.uns`.")
        PCs = adata.uns[PCs]

    if type(mean) is str:
        if mean not in adata.uns.keys():
            raise Exception(f"The key {mean} is not in `.uns`.")
        mean = adata.uns[mean]

    if not callable(vecfld):
        vf_func = vecfld.func
    else:
        vf_func = vecfld

    """def vf_func_perturb(x):
        x_gene = np.abs(x @ PCs.T + mean)
        v_gene = vector_transformation(vf_func(x), PCs)
        v_gene = v_gene - k * x_gene
        return v_gene @ PCs"""

    vf = KOVectorField(K=k, func_base=vf_func, fjac_base=vecfld.get_Jacobian(), PCs=PCs, mean=mean)
    if not callable(vecfld):
        vf.data["X"] = vecfld.data["X"]
        vf.data["V"] = vf.func(vf.data["X"])
    if return_vector_field_class:
        # vf = ko_vectorfield(K=k, func_base=vf_func, fjac_base=vecfld.get_Jacobian(), Q=PCs, mean=mean)
        # vf.func = vf_func_perturb
        return vf
    else:
        return vf.func
