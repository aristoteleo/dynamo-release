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


class Topography2D:
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
        X = X.toarray() if sp.issparse(X) else X
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


class Topography3D(Topography2D):
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

        vecfld = Topography2D(func, X_data=X_basis)
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

        vecfld = Topography3D(func, X_data=X_basis)
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
                "NCx": {str(index): array for index, array in enumerate(NCx)} if NCx is not None else None,
                "NCy": {str(index): array for index, array in enumerate(NCy)} if NCy is not None else None,
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
            "NCx": {str(index): array for index, array in enumerate(NCx)} if NCx is not None else None,
            "NCy": {str(index): array for index, array in enumerate(NCy)} if NCy is not None else None,
            "separatrices": None,
            "fp_ind": fp_ind,
        }

    return adata


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
