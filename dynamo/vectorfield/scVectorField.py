import functools
import itertools
import sys
import warnings
from abc import abstractmethod
from multiprocessing.dummy import Pool as ThreadPool
from typing import Callable, Dict, List, Optional, Tuple, Union

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import matplotlib
import numpy as np
import numpy.matlib
import scipy.sparse as sp
from anndata import AnnData
from numpy import format_float_scientific as scinot
from pynndescent import NNDescent
from scipy.linalg import lstsq
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

from ..dynamo_logger import LoggerManager, main_info, main_warning
from ..simulation.ODE import jacobian_bifur2genes, ode_bifur2genes
from ..tools.connectivity import k_nearest_neighbors
from ..tools.sampling import lhsclassic, sample, sample_by_velocity
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
    Hessian_rkhs_gaussian,
    Jacobian_kovf,
    Jacobian_numerical,
    Jacobian_rkhs_gaussian,
    Jacobian_rkhs_gaussian_parallel,
    Laplacian,
    NormDict,
    VecFldDict,
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


def norm(
    X: np.ndarray, V: np.ndarray, T: Optional[np.ndarray] = None, fix_velocity: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Normalizes the X, Y (X + V) matrix to have zero means and unit covariance.
    We use the mean of X, Y's center (mean) and scale parameters (standard deviation) to normalize T.

    Args:
        X: Current state. This corresponds to, for example, the spliced transcriptomic state.
        V: Velocity estimates in delta t. This corresponds to, for example, the inferred spliced transcriptomic
            velocity estimated calculated by dynamo or velocyto, scvelo.
        T: Current state on a grid which is often used to visualize the vector field. This corresponds to, for example,
            the spliced transcriptomic state.
        fix_velocity: Whether to fix velocity and don't transform it. Default is True.

    Returns:
        A tuple of updated X, V, T and norm_dict which includes the mean and scale values for original X, V data used
        in normalization.
    """
    Y = X + V

    xm = np.mean(X, axis=0)
    ym = np.mean(Y, axis=0)

    x, y, t = (X - xm[None, :], Y - ym[None, :], T - (1 / 2 * (xm[None, :] + ym[None, :])) if T is not None else None)

    xscale, yscale = (np.sqrt(np.mean(x**2, axis=0))[None, :], np.sqrt(np.mean(y**2, axis=0))[None, :])

    X, Y, T = x / xscale, y / yscale, t / (1 / 2 * (xscale + yscale)) if T is not None else None

    X, V, T = X, V if fix_velocity else Y - X, T
    norm_dict = {"xm": xm, "ym": ym, "xscale": xscale, "yscale": yscale, "fix_velocity": fix_velocity}

    return X, V, T, norm_dict


def bandwidth_rule_of_thumb(X: np.ndarray, return_sigma: Optional[bool] = False) -> Union[Tuple[float, float], float]:
    """
    This function computes a rule-of-thumb bandwidth for a Gaussian kernel based on:
    https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator

    The bandwidth is a free parameter that controls the level of smoothing in the estimated distribution.

    Args:
        X: Current state. This corresponds to, for example, the spliced transcriptomic state.
        return_sigma: Determines whether the standard deviation will be returned in addition to the bandwidth parameter

    Returns:
        Either a tuple with the bandwith and standard deviation or just the bandwidth
    """
    sig = np.sqrt(np.mean(np.diag(np.cov(X.T))))
    h = 1.06 * sig / (len(X) ** (-1 / 5))
    if return_sigma:
        return h, sig
    else:
        return h


def bandwidth_selector(X: np.ndarray) -> float:
    """
    This function computes an empirical bandwidth for a Gaussian kernel.
    """
    n, m = X.shape

    _, distances = k_nearest_neighbors(
        X,
        k=max(2, int(0.2 * n)) - 1,
        exclude_self=False,
        pynn_rand_state=19491001,
    )

    d = np.mean(distances[:, 1:]) / 1.5
    return np.sqrt(2) * d

def denorm(
    VecFld: Dict[str, Union[np.ndarray, None]],
    X_old: np.ndarray,
    V_old: np.ndarray,
    norm_dict: Dict[str, Union[np.ndarray, bool]],
) -> Dict[str, Union[np.ndarray, None]]:
    """Denormalize data back to the original scale.

    Args:
        VecFld: The dictionary that stores the information for the reconstructed vector field function.
        X_old: The original data for current state.
        V_old: The original velocity data.
        norm_dict: The norm_dict dictionary that includes the mean and scale values for X, Y used in normalizing the
            data.

    Returns:
        An updated VecFld dictionary that includes denormalized X, Y, X_ctrl, grid, grid_V, V, and the norm_dict key.
    """
    Y_old = X_old + V_old
    xm, ym = norm_dict["xm"], norm_dict["ym"]
    x_scale, y_scale = norm_dict["xscale"], norm_dict["yscale"]
    xy_m, xy_scale = (xm + ym) / 2, (x_scale + y_scale) / 2

    X = VecFld["X"]
    X_denorm = X_old
    Y = VecFld["Y"]
    Y_denorm = Y_old
    V = VecFld["V"]
    V_denorm = V_old if norm_dict["fix_velocity"] else (V + X) * y_scale + np.tile(ym, [V.shape[0], 1]) - X_denorm
    grid = VecFld["grid"]
    grid_denorm = grid * xy_scale + np.tile(xy_m, [grid.shape[0], 1]) if grid is not None else None
    grid_V = VecFld["grid_V"]
    grid_V_denorm = (
        (grid + grid_V) * xy_scale + np.tile(xy_m, [grid_V.shape[0], 1]) - grid if grid_V is not None else None
    )
    VecFld_denorm = {
        "X": X_denorm,
        "Y": Y_denorm,
        "V": V_denorm,
        "grid": grid_denorm,
        "grid_V": grid_V_denorm,
        "norm_dict": norm_dict,
    }

    return VecFld_denorm


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


def get_P(
    Y: np.ndarray, V: np.ndarray, sigma2: float, gamma: float, a: float, div_cur_free_kernels: Optional[bool] = False
) -> Tuple[np.ndarray, np.ndarray]:
    """GET_P estimates the posterior probability and part of the energy.

    Args:
        Y: Velocities from the data.
        V: The estimated velocity: V=f(X), f being the vector field function.
        sigma2: sigma2 is defined as sum(sum((Y - V)**2)) / (N * D)
        gamma: Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
        a: Parameter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of
            outlier's variation space is a.
        div_cur_free_kernels: A logic flag to determine whether the divergence-free or curl-free kernels will be used for learning the vector field.

    Returns:
        Tuple of (posterior probability, energy) related to equations 27 and 26 in the SparseVFC paper.

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
    func: Callable,
    X: np.ndarray,
    nbrs_idx=None,
    dist=None,
    k: int = 30,
    distance_free: bool = True,
    n_int_steps: int = 20,
    cores: int = 1,
) -> Tuple[np.ndarray, Union[NNDescent, NearestNeighbors]]:
    n, d = X.shape

    nbrs = None
    if nbrs_idx is None:
        nbrs_idx, dist, nbrs, _ = k_nearest_neighbors(
            X,
            k=k,
            exclude_self=False,
            pynn_rand_state=19491001,
            return_nbrs=True,
        )

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
    seed: Union[int, np.ndarray] = 0,
    lstsq_method: str = "drouin",
    verbose: int = 1,
) -> VecFldDict:
    """Apply sparseVFC (vector field consensus) algorithm to learn a functional form of the vector field from random
    samples with outlier on the entire space robustly and efficiently. (Ma, Jiayi, etc. al, Pattern Recognition, 2013)

    Args:
        X: Current state. This corresponds to, for example, the spliced transcriptomic state.
        Y: Velocity estimates in delta t. This corresponds to, for example, the inferred spliced transcriptomic
            velocity or total RNA velocity based on metabolic labeling data estimated calculated by dynamo.
        Grid: Current state on a grid which is often used to visualize the vector field. This corresponds to, for example,
            the spliced transcriptomic state or total RNA state.
        M: The number of basis functions to approximate the vector field.
        a: Parameter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of
            outlier's variation space is `a`.
        beta: Parameter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2).
            If None, a rule-of-thumb bandwidth will be computed automatically.
        ecr: The minimum limitation of energy change rate in the iteration process.
        gamma: Percentage of inliers in the samples. This is an initial value for EM iteration, and it is not important.
        lambda_: Represents the trade-off between the goodness of data fit and regularization. Larger Lambda_ put more
            weights on regularization.
        minP: The posterior probability Matrix P may be singular for matrix inversion. We set the minimum value of P as
            minP.
        MaxIter: Maximum iteration times.
        theta: Define how could be an inlier. If the posterior probability of a sample is an inlier is larger than theta,
            then it is regarded as an inlier.
        div_cur_free_kernels: A logic flag to determine whether the divergence-free or curl-free kernels will be used for learning the
            vector field.
        sigma: Bandwidth parameter.
        eta: Combination coefficient for the divergence-free or the curl-free kernels.
        seed: int or 1-d array_like, optional (default: `0`)
            Seed for RandomState. Must be convertible to 32 bit unsigned integers. Used in sampling control points.
            Default is to be 0 for ensure consistency between different runs.
        lstsq_method: The name of the linear least square solver, can be either 'scipy` or `douin`.
        verbose: The level of printing running information.

    Returns:
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
            tecr_traj: Vector of relative energy changes rate comparing to previous step.
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
        beta = 1 / h**2

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
    """The BaseVectorField class is a base class for storing and manipulating vector fields. A vector field is a function that associates a vector to each point in a certain space.

    The BaseVectorField class has a number of methods that allow you to work with vector fields. The __init__ method initializes the object, taking in a number of optional arguments such as X, V, and Grid, which correspond to the coordinates of the points in the vector field, the vector values at those points, and a grid used for evaluating the vector field, respectively.

    The construct_graph method takes in a set of coordinates X and returns a tuple consisting of a matrix of pairwise distances between the points in X and an object for performing nearest neighbor searches. The from_adata method takes in an AnnData object and a basis string, and extracts the coordinates and vector values of the vector field stored in the AnnData object.

    The get_X, get_V, and get_data methods return the coordinates, vector values, and both the coordinates and vector values of the vector field, respectively. The find_fixed_points method searches for fixed points of the vector field function, which are points where the velocity of the vector field is zero. The get_fixed_points method returns the fixed points and their types (stable or unstable). The plot method generates a plot of the vector field.
    """

    def __init__(
        self,
        X: Optional[np.ndarray] = None,
        V: Optional[np.ndarray] = None,
        Grid: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ):
        self.data = {"X": X, "V": V, "Grid": Grid}
        self.vf_dict = kwargs.pop("vf_dict", {})
        self.func = kwargs.pop("func", None)
        self.fixed_points = kwargs.pop("fixed_points", None)
        super().__init__(**kwargs)

    def construct_graph(
        self,
        X: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Union[NNDescent, NearestNeighbors]]:
        X = self.data["X"] if X is None else X
        return graphize_vecfld(self.func, X, **kwargs)

    def from_adata(self, adata: AnnData, basis: str = "", vf_key: str = "VecFld"):
        vf_dict, func = vecfld_from_adata(adata, basis=basis, vf_key=vf_key)
        self.data["X"] = vf_dict["X"]
        self.data["V"] = vf_dict["Y"]  # use the raw velocity
        self.vf_dict = vf_dict
        self.func = func

    def get_X(self, idx: Optional[int] = None) -> np.ndarray:
        if idx is None:
            return self.data["X"]
        else:
            return self.data["X"][idx]

    def get_V(self, idx: Optional[int] = None) -> np.ndarray:
        if idx is None:
            return self.data["V"]
        else:
            return self.data["V"][idx]

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data["X"], self.data["V"]

    def find_fixed_points(
        self,
        n_x0: int = 100,
        X0: Optional[np.ndarray] = None,
        domain: Optional[np.ndarray] = None,
        sampling_method: Literal["random", "velocity", "trn", "kmeans"] = "random",
        **kwargs,
    ) -> None:
        """
        Search for fixed points of the vector field function.

        Args:
            n_x0: Number of sampling points
            X0: An array of shape (n_samples, n_dim)
            domain: Domain in which to search for fixed points
            sampling_method: Method for sampling initial points. Can be one of `random`, `velocity`, `trn`, or `kmeans`.
        """
        if domain is not None:
            domain = np.atleast_2d(domain)

        if self.data is None and X0 is None:
            if domain is None:
                raise Exception(
                    "The initial points `X0` are not provided, "
                    "no data is stored in the vector field, and no domain is provided for the sampling of initial points."
                )
            else:
                main_info(f"Sampling {n_x0} initial points in the provided domain using the Latin Hypercube method.")
                X0 = lhsclassic(n_x0, domain.shape[0], bounds=domain)

        elif X0 is None:
            indices = sample(np.arange(len(self.data["X"])), n_x0, method=sampling_method)
            X0 = self.data["X"][indices]

        if domain is None and self.data is not None:
            domain = np.vstack((np.min(self.data["X"], axis=0), np.max(self.data["X"], axis=0))).T

        X, J, _ = find_fixed_points(X0, self.func, domain=domain, **kwargs)
        self.fixed_points = FixedPoints(X, J)

    def get_fixed_points(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get fixed points of the vector field function.

        Returns:
            Tuple storing the coordinates of the fixed points and the types of the fixed points.

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

    def assign_fixed_points(
        self, domain: Optional[np.ndarray] = None, cores: int = 1, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """assign each cell to the associated fixed points

        Args:
            domain: Array of shape (n_dim, 2), which stores the domain for each dimension
            cores: Defaults to 1.

        Returns:
            Tuple of fixed points, type assignments, and assignment IDs
        """
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
        init_states: np.ndarray,
        dims: Optional[Union[int, list, np.ndarray]] = None,
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate along a path through the vector field field function to predict the state after a certain amount of time t has elapsed.

        Args:
            init_states: Initial state provided to scipy's ivp_solver with shape (num_cells, num_dim)
            dims: Dimensions of state to be used
            scale: Scale the vector field function by this factor. Defaults to 1.
            t_end: Integrates up till when t=t_end, Defaults to None.
            step_size: Defaults to None.
            args: Additional arguments provided to scipy's ivp_solver Defaults to ().
            integration_direction: Defaults to "forward".
            interpolation_num: Defaults to 250.
            average: Defaults to True.
            sampling: Defaults to "arc_length".
            verbose: Defaults to False.
            disable: Defaults to False.

        Returns:
            Tuple storing times and predictions
        """

        from ..prediction.utils import integrate_vf_ivp
        from ..tools.utils import getTend, getTseq

        if np.isscalar(dims):
            init_states = init_states[:, :dims]
        elif dims is not None:
            init_states = init_states[:, dims]

        if self.func is None:
            VecFld = self.vf_dict
            self.func = lambda x: scale * vector_field_function(x=x, vf_dict=VecFld, dim=dims)
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
    @abstractmethod
    def get_Jacobian(self, method=None):
        raise NotImplementedError

    def compute_divergence(self, X: Optional[np.ndarray] = None, method: str = "analytical", **kwargs) -> np.ndarray:
        """Takes the trace of the jacobian matrix to calculate the divergence.

        Args:
            X: Current state. Defaults to None, initialized from self.data
            method: Method for calculating the Jacobian, one of numerical, analytical, parallel

        Returns:
            The divergence of the Jacobian matrix.
        """
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_divergence(f_jac, X, **kwargs)

    def compute_curl(
        self,
        X: Optional[np.ndarray] = None,
        method: str = "analytical",
        dim1: int = 0,
        dim2: int = 1,
        dim3: int = 2,
        **kwargs,
    ) -> np.ndarray:
        """Curl computation for many samples for 2/3 D systems.

        Args:
            X: Current state. Defaults to None, initialized from self.data
            method: Method for calculating the Jacobian, one of numerical, analytical, parallel
            dim1: index of first dimension
            dim2: index of second dimension
            dim3: index of third dimension

        Returns:
            np.ndarray storing curl
        """
        X = self.data["X"] if X is None else X
        if dim3 is None or X.shape[1] < 3:
            X = X[:, [dim1, dim2]]
        else:
            X = X[:, [dim1, dim2, dim3]]
        f_jac = self.get_Jacobian(method=method, **kwargs)
        return compute_curl(f_jac, X, **kwargs)

    def compute_acceleration(self, X: Optional[np.ndarray] = None, method: str = "analytical", **kwargs) -> np.ndarray:
        """Calculate acceleration for many samples via

        .. math::
        a = || J \cdot v ||.

        Args:
            X: Current state. Defaults to None, initialized from self.data
            method: Method for calculating the Jacobian, one of numerical, analytical, parallel

        Returns:
            np.ndarray storing the vector norm of acceleration (across all genes) for each cell
        """
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_acceleration(self.func, f_jac, X, **kwargs)

    def compute_curvature(
        self, X: Optional[np.ndarray] = None, method: str = "analytical", formula: int = 2, **kwargs
    ) -> np.ndarray:
        """Calculate curvature for many samples via

        Formula 1:
        .. math::
        \kappa = \frac{||\mathbf{v} \times \mathbf{a}||}{||\mathbf{V}||^3}

        Formula 2:
        .. math::
        \kappa = \frac{||\mathbf{Jv} (\mathbf{v} \cdot \mathbf{v}) -  ||\mathbf{v} (\mathbf{v} \cdot \mathbf{Jv})}{||\mathbf{V}||^4}

        Args:
            X: Current state. Defaults to None, initialized from self.data
            method: Method for calculating the Jacobian, one of numerical, analytical, parallel
            formula: Choose between formulas 1 and 2 to compute the curvature. Defaults to 2.

        Returns:
            np.ndarray storing the vector norm of curvature (across all genes) for each cell
        """
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_curvature(self.func, f_jac, X, formula=formula, **kwargs)

    def compute_torsion(self, X: Optional[np.ndarray] = None, method: str = "analytical", **kwargs) -> np.ndarray:
        """Calculate torsion for many samples via

        .. math::
        \tau = \frac{(\mathbf{v} \times \mathbf{a}) \cdot (\mathbf{J} \cdot \mathbf{a})}{||\mathbf{V} \times \mathbf{a}||^2}

        Args:
            X: Current state. Defaults to None, initialized from self.data
            method: Method for calculating the Jacobian, one of numerical, analytical, parallel

        Returns:
            np.ndarray storing torsion for each sample
        """
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_torsion(self.func, f_jac, X, **kwargs)

    def compute_sensitivity(self, X: Optional[np.ndarray] = None, method: str = "analytical", **kwargs) -> np.ndarray:
        """Calculate sensitivity for many samples via

        .. math::
        S = (I - J)^{-1} D(\frac{1}{{I-J}^{-1}})

        Args:
            X: Current state. Defaults to None, initialized from self.data
            method: Method for calculating the Jacobian, one of numerical, analytical, parallel Defaults to "analytical".

        Returns:
            np.ndarray storing sensitivity matrix
        """
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_sensitivity(f_jac, X, **kwargs)


class SvcVectorField(DifferentiableVectorField):
    def __init__(
        self,
        X: Optional[np.ndarray] = None,
        V: Optional[np.ndarray] = None,
        Grid: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ):
        """Initialize the VectorField class.

        Args:
            X: (dimension: n_obs x n_features), Original data.
            V: (dimension: n_obs x n_features), Velocities of cells in the same order and dimension of X.
            Grid: The function that returns diffusion matrix which can be dependent on the variables (for example, genes)
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

    def train(self, normalize: bool = False, **kwargs) -> VecFldDict:
        """Learn an function of vector field from sparse single cell samples in the entire space robustly.
        Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al,
        Pattern Recognition

        Args:
            normalize: Logic flag to determine whether to normalize the data to have zero means and unit covariance. This is
                often required for raw dataset (for example, raw UMI counts and RNA velocity values in high dimension).
                But it is normally not required for low dimensional embeddings by PCA or other non-linear dimension
                reduction methods.

        Returns:
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

    def plot_energy(
        self, figsize: Optional[Tuple[float, float]] = None, fig: Optional[matplotlib.figure.Figure] = None
    ):
        from ..plot.scVectorField import plot_energy

        plot_energy(None, vecfld_dict=self.vf_dict, figsize=figsize, fig=fig)

    def get_Jacobian(self, method: str = "analytical", input_vector_convention: str = "row", **kwargs) -> np.ndarray:
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

    def get_Hessian(self, method: str = "analytical", **kwargs) -> np.ndarray:
        """
        Get the Hessian of the vector field function.
        If method is 'analytical':
        The analytical Hessian will be returned and it always
        take row vectors as input no matter what input_vector_convention is.

        No matter the method and input vector convention, the returned Hessian is of the
        following format:
                df^2/dx_1^2        df_1^2/(dx_1 dx_2)   df_1^2/(dx_1 dx_3)   ...
                df^2/(dx_2 dx_1)   df^2/dx_2^2          df^2/(dx_2 dx_3)     ...
                df^2/(dx_3 dx_1)   df^2/(dx_3 dx_2)     df^2/dx_3^2          ...
                ...                ...                  ...                  ...
        """
        if method == "analytical":
            return lambda x: Hessian_rkhs_gaussian(x, self.vf_dict, **kwargs)
        elif method == "numerical":
            if self.func is not None:
                raise Exception("numerical Hessian for vector field is not defined.")
            else:
                raise Exception("The perturbed vector field function has not been set up.")
        else:
            raise NotImplementedError(f"The method {method} is not implemented. Currently only supports 'analytical'.")

    def get_Laplacian(self, method: str = "analytical", **kwargs) -> np.ndarray:
        """
        Get the Laplacian of the vector field. Laplacian is defined as the sum of the diagonal of the Hessian matrix.
        Because Hessian is originally defined for scalar function and here we extend it to vector functions. We will
        calculate the summation of the diagonal of each output (target) dimension.

        A Laplacian filter is an edge detector used to compute the second derivatives of an image, measuring the rate
        at which the first derivatives change (so it is the derivative of the Jacobian). This determines if a change in
        adjacent pixel values is from an edge or continuous progression.
        """
        if method == "analytical":
            return lambda x: Laplacian(H=x)
        elif method == "numerical":
            if self.func is not None:
                raise Exception("Numerical Laplacian for vector field is not defined.")
            else:
                raise Exception("The perturbed vector field function has not been set up.")
        else:
            raise NotImplementedError(f"The method {method} is not implemented. Currently only supports 'analytical'.")

    def evaluate(self, CorrectIndex: List, VFCIndex: List, siz: int) -> Tuple[float, float, float]:
        """Evaluate the precision, recall, corrRate of the sparseVFC algorithm.

        Args:
            CorrectIndex: Ground truth indexes of the correct vector field samples.
            VFCIndex: Indexes of the correct vector field samples learned by VFC.
            siz: Number of initial matches.

        Returns:
            A tuple of precision, recall, corrRate, where Precision, recall, corrRate are Precision and recall of VFC, percentage of initial correct matches, respectively.

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
        self,
        X: Optional[np.ndarray] = None,
        V: Optional[np.ndarray] = None,
        Grid=None,
        K=None,
        func_base: Optional[Callable] = None,
        fjac_base: Optional[Callable] = None,
        PCs: Optional[np.ndarray] = None,
        mean: float = None,
        *args,
        **kwargs,
    ):
        """_summary_

        Args:
            X: (dimension: n_obs x n_features), Original data. Defaults to None.
            V: (dimension: n_obs x n_features), Velocities of cells in the same order and dimension of X. Defaults to None.
            Grid: Grid of the current state. Defaults to None.
            K: _description_. Defaults to None.
            func_base: _description_. Defaults to None.
            fjac_base: Callable passed to `Jacobian_kovf` to generate the Jacobian. Defaults to None.
            PCs: The PCA loading matrix of dimensions d x k, where d is the number of dimensions of the original space. Defaults to None.
            mean: _description_. Defaults to None.
        """
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
        """
        Reference "In silico perturbation to predict gene-wise perturbation effects and cell fate diversions" in the methods section
        """

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
        def __init__(self, X=None, V=None, Grid=None, dynode_object=None, *args, **kwargs):

            self.norm_dict = {}

            assert dynode_object is not None, "dynode_object argument is required."

            valid_ind = None
            if X is not None and V is not None:
                pass
            elif dynode_object.Velocity["sampler"] is not None:
                X = dynode_object.Velocity["sampler"].X_raw
                V = dynode_object.Velocity["sampler"].V_raw
                Grid = (
                    dynode_object.Velocity["sampler"].Grid
                    if hasattr(dynode_object.Velocity["sampler"], "Grid")
                    else None
                )
                # V = dynode_object.predict_velocity(dynode_object.Velocity["sampler"].X_raw)
                valid_ind = dynode_object.Velocity["sampler"].valid_ind
            else:
                raise

            self.parameters = update_n_merge_dict(kwargs, {"X": X, "V": V, "Grid": Grid})

            self.valid_ind = np.where(~np.isnan(V.sum(1)))[0] if valid_ind is None else valid_ind

            vf_kwargs = {
                "X": X,
                "V": V,
                "Grid": Grid,
                "NNmodel": dynode_object.NNmodel,
                "Velocity_sampler": dynode_object.Velocity["sampler"],
                "TimeCourse_sampler": dynode_object.TimeCourse["sampler"],
                "Velocity_ChannelModel": dynode_object.Velocity["channel_model"],
                "TimeCourse_ChannelModel": dynode_object.TimeCourse["channel_model"],
                "Velocity_x_initialize": dynode_object.Velocity["x_variable"],
                "TimeCourse_x0_initialize": dynode_object.TimeCourse["x0_variable"],
                "NNmodel_save_path": dynode_object.NNmodel_save_path,
                "device": dynode_object.device,
            }

            vf_kwargs = update_dict(vf_kwargs, self.parameters)
            super().__init__(**vf_kwargs)

            self.func = self.predict_velocity

            self.vf_dict = {
                "X": self.data["X"],
                "valid_ind": self.valid_ind,
                "Y": self.data["V"],
                "V": self.func(self.data["X"]),
                "grid": self.data["Grid"],
                "grid_V": self.func(self.data["Grid"]),
                "iteration": int(dynode_object.max_iter),
                "velocity_loss_traj": dynode_object.Velocity["loss_trajectory"],
                "time_course_loss_traj": dynode_object.TimeCourse["loss_trajectory"],
                "autoencoder_loss_traj": dynode_object.AutoEncoder["loss_trajectory"],
                "parameters": self.parameters,
            }

        @classmethod
        def fromDynode(cls, dynode_object: Dynode) -> "dynode_vectorfield":
            return cls(X=None, V=None, Grid=None, dynode_object=dynode_object)


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


class BifurcationTwoGenesVectorField(DifferentiableVectorField):
    def __init__(
        self,
        param_dict,
        X: Optional[np.ndarray] = None,
        V: Optional[np.ndarray] = None,
        Grid: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ):
        super().__init__(X, V, Grid, *args, **kwargs)
        param_dict_ = param_dict.copy()
        for k in param_dict_.keys():
            if k not in ["a", "b", "S", "K", "m", "n", "gamma"]:
                del param_dict_[k]
                main_warning(f"The parameter {k} is not used for the vector field.")
        self.vf_dict["params"] = param_dict_
        self.func = lambda x: ode_bifur2genes(x, **param_dict_)

    def get_Jacobian(self, method=None):
        return lambda x: jacobian_bifur2genes(x, **self.vf_dict["params"])

    def find_fixed_points(self, n_x0: int = 10, **kwargs):
        a = self.vf_dict["params"]["a"]
        b = self.vf_dict["params"]["b"]
        gamma = self.vf_dict["params"]["gamma"]
        xss = (a + b) / gamma
        margin = 10
        domain = np.array([[0, xss[0] + margin], [0, xss[1] + margin]])
        return super().find_fixed_points(n_x0, X0=None, domain=domain, **kwargs)

    # TODO: nullcline calculation
