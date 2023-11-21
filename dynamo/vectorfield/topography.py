# create by Yan Zhang, minor adjusted by Xiaojie Qiu
import datetime
import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import anndata
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from scipy.integrate import odeint
from scipy.linalg import eig
from scipy.optimize import fsolve
from sklearn.neighbors import NearestNeighbors

from ..dynamo_logger import LoggerManager, main_info, main_warning
from ..tools.connectivity import k_nearest_neighbors
from ..tools.utils import gaussian_1d, inverse_norm, nearest_neighbors, update_dict
from ..utils import copy_adata
from .FixedPoints import FixedPoints
from .scVectorField import BaseVectorField, SvcVectorField
from .utils import (
    VecFldDict,
    angle,
    dynode_vector_field_function,
    find_fixed_points,
    is_outside,
    remove_redundant_points,
    vecfld_from_adata,
    vector_field_function,
)


def pac_onestep(x0: np.ndarray, func: Callable, v0: np.ndarray, ds: float = 0.01):
    """One step of the predictor-corrector method

    Args:
        x0: current value
        func: function to be integrated
        v0: tangent predictor
        ds: step size, Defaults to 0.01.

    Returns:
        x1: next value
    """
    x01 = x0 + v0 * ds

    def F(x):
        return np.array([func(x), (x - x0).dot(v0) - ds])

    x1 = fsolve(F, x01)
    return x1


def continuation(
    x0: np.ndarray,
    func: Callable,
    s_max: float,
    ds: float = 0.01,
    v0: Optional[np.ndarray] = None,
    param_axis: int = 0,
    param_direction: int = 1,
) -> np.ndarray:
    """Continually integrate the ODE `func` from x0

    Args:
        x0: initial value
        func: function to be integrated
        s_max: maximum integration length
        ds: step size, Defaults to 0.01.
        v0: initial tangent vector, Defaults to None.
        param_axis: axis of the parameter, Defaults to 0.
        param_direction: direction of the parameter, Defaults to 1.

    Returns:
        np.ndarray of values along the curve
    """
    ret = [x0]
    if v0 is None:  # initialize tangent predictor
        v = np.zeros_like(x0)
        v[param_axis] = param_direction
    else:
        v = v0
    s = 0
    while s <= s_max:
        x1 = ret[-1]
        x = pac_onestep(x1, func, v, ds)
        ret.append(x)
        s += ds

        # compute tangent predictor
        v = x - x1
        v /= np.linalg.norm(v)
    return np.array(ret)


def clip_curves(
    curves: Union[List[List], List[np.ndarray]], domain: np.ndarray, tol_discont=None
) -> Union[List[List], List[np.ndarray]]:
    """Clip curves to the domain

    Args:
        curves: list of curves
        domain: domain of the curves of dimension n x 2
        tol_discont: tolerance for discontinuity, Defaults to None.

    Returns:
        list of clipped curves joined together
    """
    ret = []
    for cur in curves:
        clip_away = np.zeros(len(cur), dtype=bool)
        for i, p in enumerate(cur):
            for j in range(len(domain)):
                if p[j] < domain[j][0] or p[j] > domain[j][1]:
                    clip_away[i] = True
                    break
            if tol_discont is not None and i > 0:
                d = np.linalg.norm(p - cur[i - 1])
                if d > tol_discont:
                    clip_away[i] = True
        # clip curve and assemble
        i_start = 0
        while i_start < len(cur) - 1:
            if not clip_away[i_start]:
                for i_end in range(i_start, len(cur)):
                    if clip_away[i_end]:
                        break
                # a tiny bit of the end could be chopped off
                ret.append(cur[i_start:i_end])
                i_start = i_end
            else:
                i_start += 1
    return ret


def compute_nullclines_2d(
    X0: Union[List, np.ndarray],
    fdx: Callable,
    fdy: Callable,
    x_range: List,
    y_range: List,
    s_max: Optional[float] = None,
    ds: Optional[float] = None,
) -> Tuple[List]:
    """Compute nullclines of a 2D vector field. Nullclines are curves along which vector field is zero in either the x or y direction.

    Args:
        X0: initial value
        fdx: differential equation for x
        fdy: differential equation for y
        x_range: range of x
        y_range: range of y
        s_max: maximum integration length, Defaults to None.
        ds: step size, Defaults to None.

    Returns:
        Tuple of nullclines in x and y
    """
    if s_max is None:
        s_max = 5 * ((x_range[1] - x_range[0]) + (y_range[1] - y_range[0]))
    if ds is None:
        ds = s_max / 1e3

    NCx = []
    NCy = []
    for x0 in X0:
        # initialize tangent predictor
        theta = np.random.rand() * 2 * np.pi
        v0 = [np.cos(theta), np.sin(theta)]
        v0 /= np.linalg.norm(v0)
        # nullcline continuation
        NCx.append(continuation(x0, fdx, s_max, ds, v0=v0))
        NCx.append(continuation(x0, fdx, s_max, ds, v0=-v0))
        NCy.append(continuation(x0, fdy, s_max, ds, v0=v0))
        NCy.append(continuation(x0, fdy, s_max, ds, v0=-v0))
    NCx = clip_curves(NCx, [x_range, y_range], ds * 10)
    NCy = clip_curves(NCy, [x_range, y_range], ds * 10)
    return NCx, NCy


def compute_separatrices(
    Xss: np.ndarray,
    Js: np.ndarray,
    func: Callable,
    x_range: List,
    y_range: List,
    t: int = 50,
    n_sample: int = 500,
    eps: float = 1e-6,
) -> List:
    """Compute separatrix based on jacobians at points in `Xss`

    Args:
        Xss: list of steady states
        Js: list of jacobians at steady states
        func: function to be integrated
        x_range: range of x
        y_range: range of y
        t: integration time, Defaults to 50.
        n_sample: number of samples, Defaults to 500.
        eps: tolerance for discontinuity, Defaults to 1e-6.

    Returns:
        list of separatrices
    """
    ret = []
    for i, x in enumerate(Xss):
        print(x)
        J = Js[i]
        w, v = eig(J)
        I_stable = np.where(np.real(w) < 0)[0]
        print(I_stable)
        for j in I_stable:  # I_unstable
            u = np.real(v[j])
            u = u / np.linalg.norm(u)
            print("u=%f, %f" % (u[0], u[1]))

            # Parameters for building separatrix
            T = np.linspace(0, t, n_sample)
            # all_sep_a, all_sep_b = None, None
            # Build upper right branch of separatrix
            ab_upper = odeint(lambda x, _: -func(x), x + eps * u, T)
            # Build lower left branch of separatrix
            ab_lower = odeint(lambda x, _: -func(x), x - eps * u, T)

            sep = np.vstack((ab_lower[::-1], ab_upper))
            ret.append(sep)
    ret = clip_curves(ret, [x_range, y_range])
    return ret


def set_test_points_on_curve(curve: List[np.ndarray], interval: float) -> np.ndarray:
    """Generates an np.ndarray of test points that are spaced out by `interval` distance

    Args:
        curve: list of points
        interval: distance for separation

    Returns:
        np.ndarray of test points
    """
    P = [curve[0]]
    dist = 0
    for i in range(1, len(curve)):
        dist += np.linalg.norm(curve[i] - curve[i - 1])
        if dist >= interval:
            P.append(curve[i])
            dist = 0
    return np.array(P)


def find_intersection_2d(curve1: List[np.ndarray], curve2: List[np.ndarray], tol_redundant: float = 1e-4) -> np.ndarray:
    """Compute intersections between curve 1 and curve2

    Args:
        curve1: list of points
        curve2: list of points
        tol_redundant: Defaults to 1e-4.

    Returns:
        np.ndarray of intersection points between curve1 and curve2
    """
    P = []
    for i in range(len(curve1) - 1):
        for j in range(len(curve2) - 1):
            p1 = curve1[i]
            p2 = curve1[i + 1]
            p3 = curve2[j]
            p4 = curve2[j + 1]
            denom = np.linalg.det([p1 - p2, p3 - p4])
            if denom != 0:
                t = np.linalg.det([p1 - p3, p3 - p4]) / denom
                u = -np.linalg.det([p1 - p2, p1 - p3]) / denom
                if t >= 0 and t <= 1 and u >= 0 and u <= 1:
                    P.append(p1 + t * (p2 - p1))
    if tol_redundant is not None:
        remove_redundant_points(P, tol=tol_redundant)
    return np.array(P)


def find_fixed_points_nullcline(
    func: Callable,
    NCx: List[List[np.ndarray]],
    NCy: List[List[np.ndarray]],
    sample_interval: float = 0.5,
    tol_redundant: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find fixed points by computing the intersections of x and y nullclines using `find_intersection_2d` and passing these intersection points as samppling points to `find_fixed_points`.

    Args:
        func: Callable passed to `find_fixed_points` along with the intersection points of the two nullclines
        NCx: List of x nullcline
        NCy: List of y nullcline
        sample_interval: Interval for sampling test points along x and y nullclines. Defaults to 0.5.
        tol_redundant: Defaults to 1e-4.

    Returns:
        A tuple with solutions for where func(x) = 0 and the Jacobian matrix
    """
    test_Px = []
    for i in range(len(NCx)):
        test_Px.append(set_test_points_on_curve(NCx[i], sample_interval))

    test_Py = []
    for i in range(len(NCy)):
        test_Py.append(set_test_points_on_curve(NCy[i], sample_interval))

    int_P = []
    for i in range(len(test_Px)):
        for j in range(len(test_Py)):
            p = find_intersection_2d(test_Px[i], test_Py[j], tol_redundant)
            for k in range(len(p)):
                int_P.append(p[k])
    int_P = np.array(int_P)
    P, J, _ = find_fixed_points(int_P, func, tol_redundant=tol_redundant)
    return P, J


def calc_fft(x):
    out = np.fft.rfft(x)
    n = len(x)
    xFFT = abs(out) / n * 2
    freq = np.arange(int(n / 2)) / n
    return xFFT[: int(n / 2)], freq


def dup_osc_idx(x: np.ndarray, n_dom: int = 3, tol: float = 0.05):
    """
    Find the index of the end of the first division in an array where the oscillatory patterns of two consecutive divisions are similar within a given tolerance.

    Args:
        x: An array-like object containing the data to be analyzed.
        n_dom: An integer specifying the number of divisions to make in the array. Defaults to 3.
        tol: A float specifying the tolerance for considering the oscillatory patterns of two divisions to be similar. Defaults to 0.05.

    Returns:
        A tuple containing the index of the end of the first division and the difference between the FFTs of the two divisions. If the oscillatory patterns of the two divisions are not similar within the given tolerance, returns (None, None).
    """
    l_int = int(np.floor(len(x) / n_dom))
    ind_a, ind_b = np.arange((n_dom - 2) * l_int, (n_dom - 1) * l_int), np.arange((n_dom - 1) * l_int, n_dom * l_int)
    y1 = x[ind_a]
    y2 = x[ind_b]

    def calc_fft_k(x):
        ret = []
        for k in range(x.shape[1]):
            xFFT, _ = calc_fft(x[:, k])
            ret.append(xFFT[1:])
        return np.hstack(ret)

    try:
        xFFt1 = calc_fft_k(y1)
        xFFt2 = calc_fft_k(y2)
    except ValueError:
        print("calc_fft_k run failed...")
        return None, None

    diff = np.linalg.norm(xFFt1 - xFFt2) / len(xFFt1)
    if diff <= tol:
        idx = (n_dom - 1) * l_int
    else:
        idx = None
    return idx, diff


def dup_osc_idx_iter(x: np.ndarray, max_iter: int = 5, **kwargs) -> Tuple[int, np.ndarray]:
    """
    Find the index of the end of the first division in an array where the oscillatory patterns of two consecutive divisions are similar within a given tolerance, using iterative search.

    Args:
        x: An array-like object containing the data to be analyzed.
        max_iter: An integer specifying the maximum number of iterations to perform. Defaults to 5.

    Returns:
        A tuple containing the index of the end of the first division and an array of differences between the FFTs of consecutive divisions. If the oscillatory patterns of the two divisions are not similar within the given tolerance after the maximum number of iterations, returns the index and array from the final iteration.
    """
    stop = False
    idx = len(x)
    j = 0
    D = []
    while not stop:
        i, d = dup_osc_idx(x[:idx], **kwargs)
        D.append(d)
        if i is None:
            stop = True
        else:
            idx = i
        j += 1
        if j >= max_iter or idx == 0:
            stop = True
    D = np.array(D)
    return idx, D


# TODO: This should be inherited from the BaseVectorField/DifferentiatiableVectorField class,
#       and BifurcationTwoGenes should be inherited from this class.
class VectorField2D:
    """
    The VectorField2D class is a class that represents a 2D vector field, which is a type of mathematical object that assigns a 2D vector to each point in a 2D space. This vector field can be defined using a function that returns the vector at each point, or by separate functions for the x and y components of the vector.

    The class also has several methods for finding fixed points (points where the vector is zero) in the vector field, as well as for querying the fixed points that have been found. The `find_fixed_points_by_sampling` method uses sampling to find fixed points within a specified range in the x and y dimensions. It does this by generating a set of random or Latin Hypercube Sampled (LHS) points within the specified range, and then using the `find_fixed_points` function to find the fixed points that are closest to these points. The `find_fixed_points function` uses an iterative method to find fixed points, starting from an initial guess and using the Jacobian matrix at each point to update the guess until the fixed point is found to within a certain tolerance.

    The `get_Xss_confidence` method estimates the confidence of the fixed points by computing the mean distance of each fixed point to its nearest
    neighbors in the data used to define the vector field. It returns an array of confidence values for each fixed point, with higher values indicating higher confidence.
    """

    def __init__(
        self,
        func: Callable,
        func_vx: Optional[Callable] = None,
        func_vy: Optional[Callable] = None,
        X_data: Optional[np.ndarray] = None,
    ):
        """
        Args:
            func: a function that takes an (n, 2) array of coordinates and returns an (n, 2) array of vectors
            func_vx: a function that takes an (n, 2) array of coordinates and returns an (n,) array of x components of the vectors, Defaults to None.
            func_vy: a function that takes an (n, 2) array of coordinates and returns an (n,) array of y components of the vectors, Defaults to None.
            X_data: Defaults to None.
        """
        self.func = func

        def func_dim(x, func, dim):
            y = func(x)
            if y.ndim == 1:
                y = y[dim]
            else:
                y = y[:, dim].flatten()
            return y

        if func_vx is None:
            self.fx = lambda x: func_dim(x, self.func, 0)
        else:
            self.fx = func_vx
        if func_vy is None:
            self.fy = lambda x: func_dim(x, self.func, 1)
        else:
            self.fy = func_vy
        self.Xss = FixedPoints()
        self.X_data = X_data
        self.NCx = None
        self.NCy = None

    def get_num_fixed_points(self) -> int:
        """
        Get the number of fixed points stored in the `Xss` attribute.

        Returns:
            int: the number of fixed points
        """
        return len(self.Xss.get_X())

    def get_fixed_points(self, get_types: Optional[bool] = True) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Get the fixed points stored in the `Xss` attribute, along with their types (stable, saddle, or unstable) if `get_types` is `True`.

        Args:
            get_types: whether to include the types of the fixed points. Defaults to `True`.

        Returns:
            tuple: a tuple containing:
                - X (np.array): an (n, 2) array of coordinates of the fixed points
                - ftype (np.array): an (n,) array of the types of the fixed points (-1 for stable, 0 for saddle, 1 for unstable). Only returned if `get_types` is `True`.
        """
        X = self.Xss.get_X()
        if not get_types:
            return X
        else:
            is_saddle, is_stable = self.Xss.is_saddle()
            # -1 -- stable, 0 -- saddle, 1 -- unstable
            ftype = np.ones(len(X))
            for i in range(len(ftype)):
                if is_saddle[i]:
                    ftype[i] = 0
                elif is_stable[i]:
                    ftype[i] = -1
            return X, ftype

    def get_Xss_confidence(self, k: Optional[int] = 50) -> np.ndarray:
        """Get the confidence of each fixed point stored in the `Xss` attribute.

        Args:
            k: the number of nearest neighbors to consider for each fixed point. Defaults to 50.

        Returns:
            an (n,) array of confidences for the fixed points
        """
        X = self.X_data
        X = X.A if sp.issparse(X) else X
        Xss = self.Xss.get_X()
        Xref = np.median(X, 0)
        Xss = np.vstack((Xss, Xref))

        _, dist = k_nearest_neighbors(
            X,
            query_X=Xss,
            k=min(k, X.shape[0] - 1) - 1,
            exclude_self=False,
            pynn_rand_state=19491001,
        )

        dist_m = dist.mean(1)
        # confidence = 1 - dist_m / dist_m.max()
        sigma = 0.1 * 0.5 * (np.max(X[:, 0]) - np.min(X[:, 0]) + np.max(X[:, 1]) - np.min(X[:, 1]))
        confidence = gaussian_1d(dist_m, sigma=sigma)
        confidence /= np.max(confidence)
        return confidence[:-1]

    def find_fixed_points_by_sampling(
        self,
        n: int,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        lhs: Optional[bool] = True,
        tol_redundant: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find fixed points by sampling the vector field within a specified range of coordinates.

        Args:
            n: the number of samples to take
            x_range: a tuple of two floats specifying the range of x coordinates to sample
            y_range: a tuple of two floats specifying the range of y coordinates to sample
            lhs: whether to use Latin Hypercube Sampling to generate the samples. Defaults to `True`.
            tol_redundant: the tolerance for removing redundant fixed points. Defaults to 1e-4.
        """
        if lhs:
            from ..tools.sampling import lhsclassic

            X0 = lhsclassic(n, 2)
        else:
            X0 = np.random.rand(n, 2)
        X0[:, 0] = X0[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
        X0[:, 1] = X0[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
        X, J, _ = find_fixed_points(
            X0,
            self.func,
            domain=[x_range, y_range],
            tol_redundant=tol_redundant,
        )
        if X is None:
            raise ValueError(f"No fixed points found. Try to increase the number of samples n.")
        self.Xss.add_fixed_points(X, J, tol_redundant)

    def find_nearest_fixed_point(
        self, x: np.ndarray, x_range: Tuple[float, float], y_range: Tuple[float, float], tol_redundant: float = 1e-4
    ):
        """Find the fixed point closest to a given initial guess within a given range.

        Args:
            x: an array specifying the initial guess
            x_range: a tuple of two floats specifying the range of x coordinates
            y_range: a tuple of two floats specifying the range of y coordinates
                tol_redundant: the tolerance for removing redundant fixed points. Defaults to 1e-4.
        """
        X, J, _ = find_fixed_points(x, self.func, domain=[x_range, y_range], tol_redundant=tol_redundant)
        if len(X) > 0:
            self.Xss.add_fixed_points(X, J, tol_redundant)

    def compute_nullclines(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        find_new_fixed_points: Optional[bool] = False,
        tol_redundant: Optional[float] = 1e-4,
    ):
        """Compute nullclines. Nullclines are curves along which vector field is zero along a particular dimension.

        Args:
            x_range: range of x
            y_range: range of y
            find_new_fixed_points: whether to find new fixed points along the nullclines and add to `self.Xss`. Defaults to False.
            s_max: maximum integration length, Defaults to None.
            ds: step size, Defaults to None.
        """
        # compute arguments
        s_max = 5 * ((x_range[1] - x_range[0]) + (y_range[1] - y_range[0]))
        ds = s_max / 1e3
        self.NCx, self.NCy = compute_nullclines_2d(
            self.Xss.get_X(),
            self.fx,
            self.fy,
            x_range,
            y_range,
            s_max=s_max,
            ds=ds,
        )
        if find_new_fixed_points:
            sample_interval = ds * 10
            X, J = find_fixed_points_nullcline(self.func, self.NCx, self.NCy, sample_interval, tol_redundant)
            outside = is_outside(X, [x_range, y_range])
            self.Xss.add_fixed_points(X[~outside], J[~outside], tol_redundant)

    # TODO Refactor dict_vf

    def output_to_dict(self, dict_vf):
        dict_vf["NCx"] = self.NCx
        dict_vf["NCy"] = self.NCy
        dict_vf["Xss"] = self.Xss.get_X()
        dict_vf["confidence"] = self.get_Xss_confidence()
        dict_vf["J"] = self.Xss.get_J()
        return dict_vf


class VectorField3D(VectorField2D):
    """A class that represents a 3D vector field, which is a type of mathematical object that assigns a 3D vector to
    each point in a 3D space.

    The class is derived from the VectorField2D class. This vector field can be defined using a function that returns
    the vector at each point, or by separate functions for the x and y components of the vector. Nullclines calculation
    are not supported for 3D vector space because of the computational complexity.
    """
    def __init__(
        self,
        func: Callable,
        func_vx: Optional[Callable] = None,
        func_vy: Optional[Callable] = None,
        func_vz: Optional[Callable] = None,
        X_data: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the VectorField3D object.

        Args:
            func: a function that takes an (n, 3) array of coordinates and returns an (n, 3) array of vectors
            func_vx: a function that takes an (n, 3) array of coordinates and returns an (n,) array of x components of
                the vectors, Defaults to None.
            func_vy: a function that takes an (n, 3) array of coordinates and returns an (n,) array of y components of
                the vectors, Defaults to None.
            func_vz: a function that takes an (n, 3) array of coordinates and returns an (n,) array of z components of
                the vectors, Defaults to None.
            X_data: Defaults to None.
        """
        super().__init__(func, func_vx, func_vy, X_data)

        def func_dim(x, func, dim):
            y = func(x)
            if y.ndim == 1:
                y = y[dim]
            else:
                y = y[:, dim].flatten()
            return y

        if func_vz is None:
            self.fz = lambda x: func_dim(x, self.func, 2)
        else:
            self.fz = func_vz

        self.NCz = None

    def find_fixed_points_by_sampling(
        self,
        n: int,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        lhs: Optional[bool] = True,
        tol_redundant: float = 1e-4,
    ) -> None:
        """Find fixed points by sampling the vector field within a specified range of coordinates.

        Args:
            n: the number of samples to take.
            x_range: a tuple of two floats specifying the range of x coordinates to sample.
            y_range: a tuple of two floats specifying the range of y coordinates to sample.
            z_range: a tuple of two floats specifying the range of z coordinates to sample.
            lhs: whether to use Latin Hypercube Sampling to generate the samples. Defaults to `True`.
            tol_redundant: the tolerance for removing redundant fixed points. Defaults to 1e-4.
        """
        if lhs:
            from ..tools.sampling import lhsclassic

            X0 = lhsclassic(n, 3)
        else:
            X0 = np.random.rand(n, 3)
        X0[:, 0] = X0[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
        X0[:, 1] = X0[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
        X0[:, 2] = X0[:, 2] * (z_range[1] - z_range[0]) + z_range[0]
        X, J, _ = find_fixed_points(
            X0,
            self.func,
            domain=[x_range, y_range, z_range],
            tol_redundant=tol_redundant,
        )
        if X is None:
            raise ValueError(f"No fixed points found. Try to increase the number of samples n.")
        self.Xss.add_fixed_points(X, J, tol_redundant)


    def output_to_dict(self, dict_vf) -> Dict:
        """Output the vector field as a dictionary.

        Returns:
            A dictionary containing nullclines, fixed points, confidence and jacobians.
        """
        dict_vf["NCx"] = self.NCx
        dict_vf["NCy"] = self.NCy
        dict_vf["NCz"] = self.NCz
        dict_vf["Xss"] = self.Xss.get_X()
        dict_vf["confidence"] = self.get_Xss_confidence()
        dict_vf["J"] = self.Xss.get_J()
        return dict_vf


def util_topology(
    adata: AnnData,
    basis: str,
    dims: Tuple[int, int],
    func: Callable,
    VecFld: VecFldDict,
    X: Optional[np.ndarray] = None,
    n: Optional[int] = 25,
    **kwargs,
):
    """A function that computes nullclines and fixed points defined by the function func.

    Args:
        adata: `AnnData` object containing cell state information.
        basis: A string specifying the reduced dimension embedding  to use for the computation.
        dims: A tuple of two integers specifying the dimensions of X to consider.
        func: A vector-valued function taking in coordinates and returning the vector field.
        VecFld: `VecFldDict` TypedDict storing information about the vector field and SparseVFC-related parameters and
            computations.
        X: an alternative to providing an `AnnData` object. Provide a np.ndarray from which `dims` are accessed,
            Defaults to None.
        n: An optional integer specifying the number of points to use for computing fixed points. Defaults to 25.

    Returns:
        A tuple consisting of the following elements:
            - X_basis: an array of shape (n, 2) where n is the number of points in X. This is the subset of X consisting
                of the first two dimensions specified by dims. If X is not provided, X_basis is taken from the obsm
                attribute of adata using the key "X_" + basis.
            - xlim, ylim, zlim: a tuple of floats specifying the limits of the x, y and z axes, respectively. These are
                computed based on the minimum and maximum values of X_basis.
            - confidence: an array of shape (n, ) containing the confidence scores of the fixed points.
            - NCx, NCy: arrays of shape (n, ) containing the x and y coordinates of the nullclines (lines where the
                derivative of the system is zero), respectively.
            - Xss: an array of shape (n, k) where k is the number of dimensions of the system, containing the fixed
                points.
            - ftype: an array of shape (n, ) containing the types of fixed points (attractor, repeller, or saddle).
            - an array of shape (n, ) containing the indices of the fixed points in the original data.
    """
    X_basis = adata.obsm["X_" + basis][:, dims] if X is None else X[:, dims]

    if X_basis.shape[1] == 2:
        fp_ind = None
        min_, max_ = X_basis.min(0), X_basis.max(0)

        xlim = [
            min_[0] - (max_[0] - min_[0]) * 0.1,
            max_[0] + (max_[0] - min_[0]) * 0.1,
        ]
        ylim = [
            min_[1] - (max_[1] - min_[1]) * 0.1,
            max_[1] + (max_[1] - min_[1]) * 0.1,
        ]
        zlim = None

        vecfld = VectorField2D(func, X_data=X_basis)
        vecfld.find_fixed_points_by_sampling(n, xlim, ylim)
        if vecfld.get_num_fixed_points() > 0:
            vecfld.compute_nullclines(xlim, ylim, find_new_fixed_points=True)
            NCx, NCy = vecfld.NCx, vecfld.NCy

        Xss, ftype = vecfld.get_fixed_points(get_types=True)
        confidence = vecfld.get_Xss_confidence()
    elif X_basis.shape[1] == 3:
        fp_ind = None
        min_, max_ = X_basis.min(0), X_basis.max(0)

        xlim = [
            min_[0] - (max_[0] - min_[0]) * 0.1,
            max_[0] + (max_[0] - min_[0]) * 0.1,
        ]
        ylim = [
            min_[1] - (max_[1] - min_[1]) * 0.1,
            max_[1] + (max_[1] - min_[1]) * 0.1,
        ]
        zlim = [
            min_[2] - (max_[2] - min_[2]) * 0.1,
            max_[2] + (max_[2] - min_[2]) * 0.1,
        ]

        vecfld = VectorField3D(func, X_data=X_basis)
        vecfld.find_fixed_points_by_sampling(n, xlim, ylim, zlim)

        NCx, NCy = None, None

        Xss, ftype = vecfld.get_fixed_points(get_types=True)
        confidence = vecfld.get_Xss_confidence()
    else:
        fp_ind = None
        xlim, ylim, zlim, confidence, NCx, NCy = None, None, None, None, None, None
        vecfld = BaseVectorField(
            X=VecFld["X"][VecFld["valid_ind"], :],
            V=VecFld["Y"][VecFld["valid_ind"], :],
            func=func,
        )

        Xss, ftype = vecfld.get_fixed_points(n_x0=n, **kwargs)
        if Xss.ndim > 1 and Xss.shape[1] > 2:
            fp_ind = nearest_neighbors(Xss, vecfld.data["X"], 1).flatten()
            Xss = vecfld.data["X"][fp_ind]

    return X_basis, xlim, ylim, zlim, confidence, NCx, NCy, Xss, ftype, fp_ind


def topography(
    adata: AnnData,
    basis: Optional[str] = "umap",
    layer: Optional[str] = None,
    X: Optional[np.ndarray] = None,
    dims: Optional[list] = None,
    n: Optional[int] = 25,
    VecFld: Optional[VecFldDict] = None,
    **kwargs,
) -> AnnData:
    """Map the topography of the single cell vector field in (first) two or three dimensions.

    Args:
        adata: an AnnData object.
        basis: The reduced dimension embedding of cells to visualize.
        layer: Which layer of the data will be used for vector field function reconstruction. This will be used in
            conjunction with X.
        X: Original data. Not used
        dims: The dimensions that will be used for vector field reconstruction.
        n: Number of samples for calculating the fixed points.
        VecFld: The reconstructed vector field function.
        kwargs: Key word arguments passed to the find_fixed_point function of the vector field class for high dimension
        fixed point identification.

    Returns:
        `AnnData` object that is updated with the `VecFld` or 'VecFld_' + basis dictionary in the `uns` attribute.
        The `VecFld2D` key stores an instance of the VectorField2D class which presumably has fixed points, nullcline,
            separatrix, computed and stored.
    """

    if VecFld is None:
        VecFld, func = vecfld_from_adata(adata, basis)
    else:
        if "velocity_loss_traj" in VecFld.keys():

            def func(x):
                return dynode_vector_field_function(x, VecFld)

        else:

            def func(x):
                return vector_field_function(x, VecFld)

    if dims is None:
        dims = np.arange(adata.obsm["X_" + basis].shape[1])

    (
        X_basis,
        xlim,
        ylim,
        zlim,
        confidence,
        NCx,
        NCy,
        Xss,
        ftype,
        fp_ind,
    ) = util_topology(adata=adata, basis=basis, X=X, dims=dims, func=func, VecFld=VecFld, n=n, *kwargs)

    # commented for now, will go back to this later.
    # sep = compute_separatrices(vecfld.Xss.get_X(), vecfld.Xss.get_J(), vecfld.func, xlim, ylim)

    if layer is None:
        vf_key = "VecFld_" + basis
    else:
        vf_key = "VecFld" if layer == "X" else "VecFld_" + layer

    if vf_key in adata.uns_keys():
        adata.uns[vf_key].update(
            {
                "xlim": xlim,
                "ylim": ylim,
                "zlim": zlim,
                "X_data": X_basis,
                "Xss": Xss,
                "ftype": ftype,
                "confidence": confidence,
                "NCx": {str(index): array for index, array in enumerate(NCx)},
                "NCy": {str(index): array for index, array in enumerate(NCy)},
                "separatrices": None,
                "fp_ind": fp_ind,
            }
        )
    else:
        adata.uns[vf_key] = {
            "xlim": xlim,
            "ylim": ylim,
            "zlim": zlim,
            "X_data": X_basis,
            "Xss": Xss,
            "ftype": ftype,
            "confidence": confidence,
            "NCx": {str(index): array for index, array in enumerate(NCx)},
            "NCy": {str(index): array for index, array in enumerate(NCy)},
            "separatrices": None,
            "fp_ind": fp_ind,
        }

    return adata


def VectorField(
    adata: anndata.AnnData,
    basis: Optional[str] = None,
    layer: Optional[str] = None,
    dims: Optional[Union[int, list]] = None,
    genes: Optional[list] = None,
    normalize: Optional[bool] = False,
    grid_velocity: bool = False,
    grid_num: int = 50,
    velocity_key: str = "velocity_S",
    method: str = "SparseVFC",
    min_vel_corr: float = 0.6,
    restart_num: int = 5,
    restart_seed: Optional[list] = [0, 100, 200, 300, 400],
    model_buffer_path: Optional[str] = None,
    return_vf_object: bool = False,
    map_topography: bool = False,
    pot_curl_div: bool = False,
    cores: int = 1,
    result_key: Optional[str] = None,
    copy: bool = False,
    n: int = 25,
    **kwargs,
) -> Union[anndata.AnnData, BaseVectorField]:
    """Learn a function of high dimensional vector field from sparse single cell samples in the entire space robustly.

    Args:
        adata: AnnData object that contains embedding and velocity data
        basis: The embedding data to use. The vector field function will be learned on the low  dimensional embedding and can be then projected
            back to the high dimensional space.
        layer: Which layer of the data will be used for vector field function reconstruction. The layer once provided, will override the `basis`
            argument and then learn the vector field function in high dimensional space.
        dims: The dimensions that will be used for reconstructing vector field functions. If it is an `int` all     dimension from the first
            dimension to `dims` will be used; if it is a list, the dimensions in the list will be used.
        genes: The gene names whose gene expression will be used for vector field reconstruction. By default (when genes is set to None), the genes
            used for velocity embedding (var.use_for_transition) will be used for vector field reconstruction. Note that the genes to be used need to have velocity calculated.
        normalize: Logic flag to determine whether to normalize the data to have zero means and unit covariance. This is often required for raw
            dataset (for example, raw UMI counts and RNA velocity values in high dimension). But it is normally not required for low dimensional embeddings by PCA or other non-linear dimension reduction methods.
        grid_velocity: Whether to generate grid velocity. Note that by default it is set to be False, but for datasets with embedding dimension
            less than 4, the grid velocity will still be generated. Please note that number of total grids in the space increases exponentially as the number of dimensions increases. So it may quickly lead to lack of memory, for example, it cannot allocate the array with grid_num set to be 50 and dimension is 6 (50^6 total grids) on 32 G memory computer. Although grid velocity may not be generated, the vector field function can still be learned for thousands of dimensions and we can still predict the transcriptomic cell states over long time period.
        grid_num: The number of grids in each dimension for generating the grid velocity.
        velocity_key: The key from the adata layer that corresponds to the velocity matrix.
        method: Method that is used to reconstruct the vector field functionally. Currently only SparseVFC supported but other improved approaches
            are under development.
        min_vel_corr: The minimal threshold for the cosine correlation between input velocities and learned velocities to consider as a successful
            vector field reconstruction procedure. If the cosine correlation is less than this threshold and restart_num > 1, `restart_num` trials will be attempted with different seeds to reconstruct the vector field function. This can avoid some reconstructions to be trapped in some local optimal.
        restart_num: The number of retrials for vector field reconstructions.
        restart_seed: A list of seeds for each retrial. Must be the same length as `restart_num` or None.
        buffer_path: The directory address keeping all the saved/to-be-saved torch variables and NN modules. When `method` is set to be `dynode`,
            buffer_path will set to be
        return_vf_object: Whether or not to include an instance of a vectorfield class in the the `VecFld` dictionary in the `uns`attribute.
        map_topography: Whether to quantify the topography of vector field. Note that for higher than 2D vector     field, we can only identify
            fixed points as high-dimensional nullcline and separatrices are mathematically difficult to be identified. Nullcline and separatrices will also be a surface or manifold in high-dimensional vector field.
        pot_curl_div: Whether to calculate potential, curl or divergence for each cell. Potential can be calculated for any basis while curl and
            divergence is by default only applied to 2D basis. However, divergence is applicable for any dimension while curl is generally only defined for 2/3 D systems.
        cores: Number of cores to run the ddhodge function. If cores is set to be > 1, multiprocessing will be used to
            parallel the ddhodge calculation.
        result_key:
            The key that will be used as prefix for the vector field key in .uns
        copy: Whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments and
            returning `None`.
        n: Number of samples for calculating the fixed points.
        kwargs: Other additional parameters passed to the vectorfield class.

    Returns:
        If `copy` and `return_vf_object` arguments are set to False, `annData` object is updated with the `VecFld`dictionary in the `uns` attribute.
        If `return_vf_object` is set to True, then a vector field class object is returned.
        If `copy` is set to True, a deep copy of the original `adata` object is returned.
    """
    logger = LoggerManager.gen_logger("dynamo-topography")
    logger.info("VectorField reconstruction begins...", indent_level=1)
    logger.log_time()
    adata = copy_adata(adata) if copy else adata

    if basis is not None:
        logger.info(
            "Retrieve X and V based on basis: %s. \n "
            "       Vector field will be learned in the %s space." % (basis.upper(), basis.upper())
        )
        X = adata.obsm["X_" + basis].copy()
        V = adata.obsm["velocity_" + basis].copy()

        if np.isscalar(dims):
            X, V = X[:, :dims], V[:, :dims]
        elif type(dims) is list:
            X, V = X[:, dims], V[:, dims]
    else:
        logger.info(
            "Retrieve X and V based on `genes`, layer: %s. \n "
            "       Vector field will be learned in the gene expression space." % layer
        )
        valid_genes = (
            list(set(genes).intersection(adata.var.index))
            if genes is not None
            else adata.var_names[adata.var.use_for_transition]
        )
        if layer == "X":
            X = adata[:, valid_genes].X.copy()
            X = np.expm1(X)
        else:
            X = inverse_norm(adata, adata.layers[layer])

        V = adata[:, valid_genes].layers[velocity_key].copy()

        if sp.issparse(X):
            X, V = X.A, V.A

        # keep only genes with finite velocity and expression values, useful when learning vector field in the original
        # gene expression space.
        finite_genes = np.logical_and(np.isfinite(X).all(axis=0), np.isfinite(V).all(axis=0))
        X, V = X[:, finite_genes], V[:, finite_genes]
        valid_genes = np.array(valid_genes)[np.where(finite_genes)[0]].tolist()
        if sum(finite_genes) < len(finite_genes):
            logger.warning(
                f"There are {(len(finite_genes) - sum(finite_genes))} genes with infinite expression or velocity "
                f"values. These genes will be excluded from vector field reconstruction. Please make sure the genes you "
                f"selected has no non-infinite values"
            )

    Grid = None
    if X.shape[1] < 4 or grid_velocity:
        logger.info("Generating high dimensional grids and convert into a row matrix.")
        # smart way for generating high dimensional grids and convert into a row matrix
        min_vec, max_vec = (
            X.min(0),
            X.max(0),
        )
        min_vec = min_vec - 0.01 * np.abs(max_vec - min_vec)
        max_vec = max_vec + 0.01 * np.abs(max_vec - min_vec)

        Grid_list = np.meshgrid(*[np.linspace(i, j, grid_num) for i, j in zip(min_vec, max_vec)])
        Grid = np.array([i.flatten() for i in Grid_list]).T

    if X is None:
        raise Exception(f"X is None. Make sure you passed the correct X or {basis} dimension reduction method.")
    elif V is None:
        raise Exception("V is None. Make sure you passed the correct V.")

    logger.info("Learning vector field with method: %s." % (method.lower()))
    if method.lower() == "sparsevfc":
        vf_kwargs = {
            "M": None,
            "a": 5,
            "beta": None,
            "ecr": 1e-5,
            "gamma": 0.9,
            "lambda_": 3,
            "minP": 1e-5,
            "MaxIter": 30,
            "theta": 0.75,
            "div_cur_free_kernels": False,
            "velocity_based_sampling": True,
            "sigma": 0.8,
            "eta": 0.5,
            "seed": 0,
        }
    elif method.lower() == "dynode_old":
        try:
            from dynode.vectorfield import networkModels

            # from dynode.vectorfield.losses_weighted import MAD, BinomialChannel, WassersteinDistance, CosineDistance
            from dynode.vectorfield.losses_weighted import MSE
            from dynode.vectorfield.samplers import VelocityDataSampler

            from .scVectorField import dynode_vectorfield
        except ImportError:
            raise ImportError("You need to install the package `dynode`." "install dynode via `pip install dynode`")

        velocity_data_sampler = VelocityDataSampler(adata={"X": X, "V": V}, normalize_velocity=normalize)
        max_iter = 2 * 100000 * np.log(X.shape[0]) / (250 + np.log(X.shape[0]))

        cwd, cwt = os.getcwd(), datetime.datetime.now()

        if model_buffer_path is None:
            model_buffer_path = cwd + "/" + basis + "_" + str(cwt.year) + "_" + str(cwt.month) + "_" + str(cwt.day)
            main_warning("the buffer path saving the dynode model is in %s" % (model_buffer_path))

        vf_kwargs = {
            "model": networkModels,
            "sirens": False,
            "enforce_positivity": False,
            "velocity_data_sampler": velocity_data_sampler,
            "time_course_data_sampler": None,
            "network_dim": X.shape[1],
            "velocity_loss_function": MSE(),  # CosineDistance(), # #MSE(), MAD()
            # BinomialChannel(p=0.1, alpha=1)
            "time_course_loss_function": None,
            "velocity_x_initialize": X,
            "time_course_x0_initialize": None,
            "smoothing_factor": None,
            "stability_factor": None,
            "load_model_from_buffer": False,
            "buffer_path": model_buffer_path,
            "hidden_features": 256,
            "hidden_layers": 3,
            "first_omega_0": 30.0,
            "hidden_omega_0": 30.0,
        }
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
    elif method.lower() == "dynode":
        try:
            from dynode.vectorfield import Dynode  # networkModels,

            # from dynode.vectorfield.losses_weighted import MAD, BinomialChannel, WassersteinDistance, CosineDistance
            # from dynode.vectorfield.losses_weighted import MSE
            # from dynode.vectorfield.samplers import VelocityDataSampler
            from .scVectorField import dynode_vectorfield
        except ImportError:
            raise ImportError("You need to install the package `dynode`." "install dynode via `pip install dynode`")

        if not ("Dynode" in kwargs and type(kwargs["Dynode"]) == Dynode):
            velocity_data_sampler = VelocityDataSampler(adata={"X": X, "V": V}, normalize_velocity=normalize)
            max_iter = 2 * 100000 * np.log(X.shape[0]) / (250 + np.log(X.shape[0]))

            cwd, cwt = os.getcwd(), datetime.datetime.now()

            if model_buffer_path is None:
                model_buffer_path = cwd + "/" + basis + "_" + str(cwt.year) + "_" + str(cwt.month) + "_" + str(cwt.day)
                main_warning("the buffer path saving the dynode model is in %s" % (model_buffer_path))

            vf_kwargs = {
                "model": networkModels,
                "sirens": False,
                "enforce_positivity": False,
                "velocity_data_sampler": velocity_data_sampler,
                "time_course_data_sampler": None,
                "network_dim": X.shape[1],
                "velocity_loss_function": MSE(),  # CosineDistance(), # #MSE(), MAD()
                # BinomialChannel(p=0.1, alpha=1)
                "time_course_loss_function": None,
                "velocity_x_initialize": X,
                "time_course_x0_initialize": None,
                "smoothing_factor": None,
                "stability_factor": None,
                "load_model_from_buffer": False,
                "buffer_path": model_buffer_path,
                "hidden_features": 256,
                "hidden_layers": 3,
                "first_omega_0": 30.0,
                "hidden_omega_0": 30.0,
            }
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
        else:
            vf_kwargs, train_kwargs = {}, {}
    else:
        raise ValueError("current only support two methods, SparseVFC and dynode")

    vf_kwargs = update_dict(vf_kwargs, kwargs)

    if restart_num > 0:
        if len(restart_seed) != restart_num:
            main_warning(
                f"the length of {restart_seed} is different from {restart_num}, " f"using `np.range(restart_num) * 100"
            )
            restart_seed = np.arange(restart_num) * 100
        restart_counter, cur_vf_list, res_list = 0, [], []
        while True:
            if method.lower() == "sparsevfc":
                kwargs.update({"seed": restart_seed[restart_counter]})
                VecFld = SvcVectorField(X, V, Grid, **vf_kwargs)
                cur_vf_dict = VecFld.train(normalize=normalize, **kwargs)
            elif method.lower() == "dynode_old":
                train_kwargs = update_dict(train_kwargs, kwargs)
                VecFld = dynode_vectorfield(X, V, Grid, **vf_kwargs)
                # {"VecFld": VecFld.train(**kwargs)}
                cur_vf_dict = VecFld.train(**train_kwargs)
            elif method.lower() == "dynode":
                if not ("Dynode" in kwargs and type(kwargs["Dynode"]) == Dynode):
                    train_kwargs = update_dict(train_kwargs, kwargs)
                    VecFld = dynode_vectorfield(X, V, Grid, **vf_kwargs)
                    # {"VecFld": VecFld.train(**kwargs)}
                    cur_vf_dict = VecFld.train(**train_kwargs)
                else:
                    Dynode_obj = kwargs["Dynode"]
                    VecFld = dynode_vectorfield.fromDynode(Dynode_obj)
                    X, Y = Dynode_obj.Velocity["sampler"].X_raw, Dynode_obj.Velocity["sampler"].V_raw
                    cur_vf_dict = {
                        "X": X,
                        "Y": Y,
                        "V": Dynode_obj.predict_velocity(Dynode_obj.Velocity["sampler"].X_raw),
                        "grid_V": Dynode_obj.predict_velocity(Dynode_obj.Velocity["sampler"].Grid),
                        "valid_ind": Dynode_obj.Velocity["sampler"].valid_ind
                        if hasattr(Dynode_obj.Velocity["sampler"], "valid_ind")
                        else np.arange(X.shape[0]),
                        "parameters": Dynode_obj.Velocity,
                        "dynode_object": VecFld,
                    }

            # consider refactor with .simulation.evaluation.py
            reference, prediction = (
                cur_vf_dict["Y"][cur_vf_dict["valid_ind"]],
                cur_vf_dict["V"][cur_vf_dict["valid_ind"]],
            )
            true_normalized = reference / (np.linalg.norm(reference, axis=1).reshape(-1, 1) + 1e-20)
            predict_normalized = prediction / (np.linalg.norm(prediction, axis=1).reshape(-1, 1) + 1e-20)
            res = np.mean(true_normalized * predict_normalized) * prediction.shape[1]

            cur_vf_list += [cur_vf_dict]
            res_list += [res]
            if res < min_vel_corr:
                restart_counter += 1
                main_info(
                    f"current cosine correlation between input velocities and learned velocities is less than "
                    f"{min_vel_corr}. Make a {restart_counter}-th vector field reconstruction trial.",
                    indent_level=2,
                )
            else:
                vf_dict = cur_vf_dict
                break

            if restart_counter > restart_num - 1:
                main_warning(
                    f"Cosine correlation between input velocities and learned velocities is less than"
                    f" {min_vel_corr} after {restart_num} trials of vector field reconstruction."
                )
                vf_dict = cur_vf_list[np.argmax(np.array(res_list))]

                break
    else:
        if method.lower() == "sparsevfc":
            VecFld = SvcVectorField(X, V, Grid, **vf_kwargs)
            vf_dict = VecFld.train(normalize=normalize, **kwargs)
        elif method.lower() == "dynode":
            train_kwargs = update_dict(train_kwargs, kwargs)
            VecFld = dynode_vectorfield(X, V, Grid, **vf_kwargs)
            # {"VecFld": VecFld.train(**kwargs)}
            vf_dict = VecFld.train(**train_kwargs)

    if result_key is None:
        vf_key = "VecFld" if basis is None else "VecFld_" + basis
    else:
        vf_key = result_key if basis is None else result_key + "_" + basis

    vf_dict["method"] = method
    if basis is not None:
        key = "velocity_" + basis + "_" + method
        X_copy_key = "X_" + basis + "_" + method

        logger.info_insert_adata(key, adata_attr="obsm")
        logger.info_insert_adata(X_copy_key, adata_attr="obsm")
        adata.obsm[key] = vf_dict["V"]
        adata.obsm[X_copy_key] = vf_dict["X"]

        vf_dict["dims"] = dims

        logger.info_insert_adata(vf_key, adata_attr="uns")
        adata.uns[vf_key] = vf_dict
    else:
        key = velocity_key + "_" + method

        logger.info_insert_adata(key, adata_attr="layers")
        adata.layers[key] = sp.csr_matrix((adata.shape))
        adata.layers[key][:, [adata.var_names.get_loc(i) for i in valid_genes]] = vf_dict["V"]

        vf_dict["layer"] = layer
        vf_dict["genes"] = genes
        vf_dict["velocity_key"] = velocity_key

        logger.info_insert_adata(vf_key, adata_attr="uns")
        adata.uns[vf_key] = vf_dict

    if map_topography:
        tp_kwargs = {"n": n}
        tp_kwargs = update_dict(tp_kwargs, kwargs)

        logger.info("Mapping topography...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            adata = topography(
                adata,
                basis=basis,
                X=X,
                layer=layer,
                dims=None,
                VecFld=vf_dict,
                **tp_kwargs,
            )
    if pot_curl_div:
        from .vector_calculus import curl, divergence

        logger.info(f"Running ddhodge to estimate vector field based pseudotime in {basis} basis...")
        from ..external.hodge import ddhodge

        ddhodge(adata, basis=basis, cores=cores)
        if X.shape[1] == 2:
            logger.info("Computing curl...")
            curl(adata, basis=basis)

        logger.info("Computing divergence...")
        divergence(adata, basis=basis)

    control_point, inlier_prob, valid_ids = (
        "control_point_" + basis if basis is not None else "control_point",
        "inlier_prob_" + basis if basis is not None else "inlier_prob",
        vf_dict["valid_ind"],
    )
    if method.lower() == "sparsevfc":
        logger.info_insert_adata(control_point, adata_attr="obs")
        logger.info_insert_adata(inlier_prob, adata_attr="obs")

        adata.obs[control_point], adata.obs[inlier_prob] = False, np.nan
        adata.obs.loc[adata.obs_names[vf_dict["ctrl_idx"]], control_point] = True
        adata.obs.loc[adata.obs_names[valid_ids], inlier_prob] = vf_dict["P"].flatten()

    # angles between observed velocity and that predicted by vector field across cells:
    cell_angles = np.zeros(adata.n_obs, dtype=float)
    for i, u, v in zip(valid_ids, V[valid_ids], vf_dict["V"]):
        # fix the u, v norm == 0 in angle function
        cell_angles[i] = angle(u.astype("float64"), v.astype("float64"))

    if basis is not None:
        temp_key = "obs_vf_angle_" + basis

        logger.info_insert_adata(temp_key, adata_attr="obs")
        adata.obs[temp_key] = cell_angles
    else:
        temp_key = "obs_vf_angle"
        logger.info_insert_adata(temp_key, adata_attr="obs")
        adata.obs[temp_key] = cell_angles

    logger.finish_progress("VectorField")
    if return_vf_object:
        return VecFld
    elif copy:
        return adata
    return None


def assign_fixedpoints(
    adata: AnnData,
    basis: str = "pca",
    cores: int = 1,
    copy: bool = False,
) -> Optional[AnnData]:
    """Assign each cell in our data to a fixed point.

    Args:
        adata: AnnData object that contains reconstructed vector field in the `basis` space.
        basis: The vector field function for the `basis` that will be used to assign fixed points for each cell.
        cores: Number of cores to run the fixed-point search for each cell.
        copy: Whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments and
            returning `None`.

    Returns:
        adata: :class:`Union[None, anndata.AnnData]`
            If `copy` is set to False, return None but the adata object will updated with a `fps_assignment` in .obs as
            well as the `'fps_assignment_' + basis` in the .uns.
            If `copy` is set to True, a deep copy of the original `adata` object is returned.
    """
    logger = LoggerManager.gen_logger("dynamo-assign_fixedpoints")
    logger.info("assign_fixedpoints begins...", indent_level=1)
    logger.log_time()
    adata = copy_adata(adata) if copy else adata

    VecFld, func = vecfld_from_adata(adata, basis=basis)

    vecfld_class = BaseVectorField(
        X=VecFld["X"],
        V=VecFld["Y"],
        func=func,
    )

    (
        X,
        valid_fps_type_assignment,
        assignment_id,
    ) = vecfld_class.assign_fixed_points(cores=cores)
    assignment_id = [str(int(i)) if np.isfinite(i) else None for i in assignment_id]
    adata.obs["fps_assignment"] = assignment_id
    adata.uns["fps_assignment_" + basis] = {
        "X": X,
        "valid_fps_type_assignment": valid_fps_type_assignment,
        "assignment_id": assignment_id,
    }

    logger.finish_progress("assign_fixedpoints")
    if copy:
        return adata
    return None
