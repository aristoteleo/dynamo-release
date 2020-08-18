# create by Yan Zhang, minor adjusted by Xiaojie Qiu
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.optimize import fsolve
from scipy.spatial.distance import pdist
from scipy.linalg import eig
from scipy.integrate import odeint
from sklearn.neighbors import NearestNeighbors

from .scVectorField import vectorfield
from ..tools.utils import (
    update_dict,
    form_triu_matrix,
    index_condensed_matrix,
    inverse_norm,
)

from .utils_vecCalc import vector_field_function, vecfld_from_adata

from ..external.hodge import ddhodge
from .vector_calculus import curl, divergence

def remove_redundant_points(X, tol=1e-4, output_discard=False):
    X = np.atleast_2d(X)
    discard = np.zeros(len(X), dtype=bool)
    if X.shape[0] > 1:
        dist = pdist(X)
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                if dist[index_condensed_matrix(len(X), i, j)] < tol:
                    discard[j] = True
        X = X[~discard]
    if output_discard:
        return X, discard
    else:
        return X


def find_fixed_points(X0, func_vf, tol_redundant=1e-4, full_output=False):
    X = []
    J = []
    fval = []
    for x0 in X0:
        if full_output:
            x, info_dict, _, _ = fsolve(func_vf, x0, full_output=True)
            fval.append(info_dict["fvec"])
            # compute Jacobian
            Q = info_dict["fjac"]
            R = form_triu_matrix(info_dict["r"])
            J.append(Q.T @ R)
        else:
            x = fsolve(func_vf, x0)
        X.append(x)
    X = np.array(X)
    if full_output:
        J = np.array(J)
        fval = np.array(fval)

    if tol_redundant is not None:
        if full_output:
            X, discard = remove_redundant_points(X, tol_redundant, output_discard=True)
            J = J[~discard]
            fval = fval[~discard]
        else:
            X = remove_redundant_points(X, tol_redundant)

    if full_output:
        return X, J, fval
    else:
        return X


def pac_onestep(x0, func, v0, ds=0.01):
    x01 = x0 + v0 * ds
    F = lambda x: np.array([func(x), (x - x0).dot(v0) - ds])
    x1 = fsolve(F, x01)
    return x1


def continuation(x0, func, s_max, ds=0.01, v0=None, param_axis=0, param_direction=1):
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


def clip_curves(curves, domain, tol_discont=None):
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
                ret.append(
                    cur[i_start:i_end]
                )  # a tiny bit of the end could be chopped off
                i_start = i_end
            else:
                i_start += 1
    return ret


def compute_nullclines_2d(X0, fdx, fdy, x_range, y_range, s_max=None, ds=None):
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


def compute_separatrices(Xss, Js, func, x_range, y_range, t=50, n_sample=500, eps=1e-6):
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


def set_test_points_on_curve(curve, interval):
    P = [curve[0]]
    dist = 0
    for i in range(1, len(curve)):
        dist += np.linalg.norm(curve[i] - curve[i - 1])
        if dist >= interval:
            P.append(curve[i])
            dist = 0
    return np.array(P)


def find_intersection_2d(curve1, curve2, tol_redundant=1e-4):
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
    func, NCx, NCy, sample_interval=0.5, tol_redundant=1e-4, full_output=False
):
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
    if full_output:
        P, J, _ = find_fixed_points(int_P, func, tol_redundant, full_output=True)
        return P, J
    else:
        P = find_fixed_points(int_P, func, tol_redundant)
        return P


def is_outside(X, domain):
    is_outside = np.zeros(X.shape[0], dtype=bool)
    for k in range(X.shape[1]):
        o = np.logical_or(X[:, k] < domain[k][0], X[:, k] > domain[k][1])
        is_outside = np.logical_or(is_outside, o)
    return is_outside

def calc_fft(x):
    out = np.fft.rfft(x)
    n = len(x)
    xFFT = abs(out)/n*2
    freq = np.arange(int(n/2))/n
    return xFFT[:int(n/2)], freq

def dup_osc_idx(x, n_dom=3, tol=0.05):
    l = int(np.floor(len(x) / n_dom))
    y1 = x[(n_dom-2)*l : (n_dom-1)*l]
    y2 = x[(n_dom-1)*l : n_dom*l]
    
    def calc_fft_k(x):
        ret = []
        for k in range(x.shape[1]):
            xFFT, _ = calc_fft(x[:, k])
            ret.append(xFFT[1:])
        return np.hstack(ret)
    
    xFFt1 = calc_fft_k(y1)
    xFFt2 = calc_fft_k(y2)
    
    diff = np.linalg.norm(xFFt1 - xFFt2)/len(xFFt1)
    if diff <= tol:
        idx = (n_dom-1)*l
    else:
        idx = None
    return idx, diff

def dup_osc_idx_iter(x, max_iter=5, **kwargs):
    stop = False
    idx = len(x)
    j = 0
    D = []
    while (not stop):
        i, d = dup_osc_idx(x[:idx], **kwargs)
        D.append(d)
        if i is None:
            stop = True
        else:
            idx = i
        j += 1
        if j >= max_iter:
            stop = True
    D = np.array(D)
    return idx, D

class FixedPoints:
    def __init__(self, X=None, J=None):
        self.X = X or []
        self.J = J or []
        self.eigvals = []

    def get_X(self):
        return np.array(self.X)

    def get_J(self):
        return np.array(self.J)

    def add_fixed_points(self, X, J, tol_redundant=1e-4):
        for i, x in enumerate(X):
            redundant = False
            if tol_redundant is not None and len(self.X) > 0:
                for y in self.X:
                    if np.linalg.norm(x - y) <= tol_redundant:
                        redundant = True
            if not redundant:
                self.X.append(x)
                self.J.append(J[i])

    def compute_eigvals(self):
        self.eigvals = []
        for i in range(len(self.J)):
            w, _ = eig(self.J[i])
            self.eigvals.append(w)

    def is_stable(self):
        if len(self.eigvals) != len(self.X):
            self.compute_eigvals()

        stable = np.ones(len(self.eigvals), dtype=bool)
        for i, w in enumerate(self.eigvals):
            if np.any(np.real(w) >= 0):
                stable[i] = False
        return stable

    def is_saddle(self):
        is_stable = self.is_stable()
        saddle = np.zeros(len(self.eigvals), dtype=bool)
        for i, w in enumerate(self.eigvals):
            if not is_stable[i] and np.any(np.real(w) < 0):
                saddle[i] = True
        return saddle, is_stable


class VectorField2D:
    def __init__(self, func, func_vx=None, func_vy=None, X_data=None, k=50):
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
        self.k = k
        self.NCx = None
        self.NCy = None

    def get_num_fixed_points(self):
        return len(self.Xss.get_X())

    def get_fixed_points(self, get_types=True):
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

    def get_Xss_confidence(self):
        X = self.X_data
        X = X.A if sp.issparse(X) else X
        Xss = self.Xss.get_X()
        alg = 'ball_tree' if Xss.shape[1] > 10 else 'kd_tree'

        if X.shape[0] > 200000 and X.shape[1] > 2: 
            from pynndescent import NNDescent

            nbrs = NNDescent(X, metric='euclidean', n_neighbors=min(self.k, X.shape[0] - 1), n_jobs=-1, random_state=19491001)
            _, dist = nbrs.query(Xss, k=min(self.k, X.shape[0] - 1))
        else:
            alg = 'ball_tree' if X.shape[1] > 10 else 'kd_tree'
            nbrs = NearestNeighbors(n_neighbors=min(self.k, X.shape[0] - 1), algorithm=alg, n_jobs=-1).fit(X)
            dist, _ = nbrs.kneighbors(Xss)

        dist_m = dist.mean(1)
        confidence = 1 - dist_m / dist_m.max()

        return confidence

    def find_fixed_points_by_sampling(
        self, n, x_range, y_range, lhs=True, tol_redundant=1e-4
    ):
        if lhs:
            from ..tools.sampling import lhsclassic

            X0 = lhsclassic(n, 2)
        else:
            X0 = np.random.rand(n, 2)
        X0[:, 0] = X0[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
        X0[:, 1] = X0[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
        X, J, _ = find_fixed_points(
            X0, self.func, tol_redundant=tol_redundant, full_output=True
        )
        # remove points that are outside the domain
        outside = is_outside(X, [x_range, y_range])
        self.Xss.add_fixed_points(X[~outside], J[~outside], tol_redundant)

    def find_nearest_fixed_point(self, x, x_range, y_range, tol_redundant=1e-4):
        X, J, _ = find_fixed_points(
            x, self.func, tol_redundant=tol_redundant, full_output=True
        )
        # remove point if outside the domain
        outside = is_outside(X, [x_range, y_range])[0]
        if not outside:
            self.Xss.add_fixed_points(X, J, tol_redundant)

    def compute_nullclines(
        self, x_range, y_range, find_new_fixed_points=False, tol_redundant=1e-4
    ):
        # compute arguments
        s_max = 5 * ((x_range[1] - x_range[0]) + (y_range[1] - y_range[0]))
        ds = s_max / 1e3
        self.NCx, self.NCy = compute_nullclines_2d(
            self.Xss.get_X(), self.fx, self.fy, x_range, y_range, s_max=s_max, ds=ds
        )
        if find_new_fixed_points:
            sample_interval = ds * 10
            X, J = find_fixed_points_nullcline(
                self.func, self.NCx, self.NCy, sample_interval, tol_redundant, True
            )
            outside = is_outside(X, [x_range, y_range])
            self.Xss.add_fixed_points(X[~outside], J[~outside], tol_redundant)

    def output_to_dict(self, dict_vf):
        dict_vf["NCx"] = self.NCx
        dict_vf["NCy"] = self.NCy
        dict_vf["Xss"] = self.Xss.get_X()
        dict_vf["confidence"] = self.get_Xss_confidence()
        dict_vf["J"] = self.Xss.get_J()
        return dict_vf


def topography(adata, basis="umap", layer=None, X=None, dims=None, n=25, VecFld=None):
    """Map the topography of the single cell vector field in (first) two dimensions.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        basis: `str` (default: `trimap`)
            The reduced dimension embedding of cells to visualize.
        layer: `str` or None (default: None)
            Which layer of the data will be used for vector field function reconstruction. This will be used in conjunction
            with X.
        X: 'np.ndarray' (dimension: n_obs x n_features)
                Original data.
        dims: `list` or `None` (default: `None`)
            The dimensions that will be used for vector field reconstruction.
        n: `int` (default: `10`)
            Number of samples for calculating the fixed points.
        VecFld: `dictionary` or None (default: None)
            The reconstructed vector field function.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            `AnnData` object that is updated with the `VecFld` or 'VecFld_' + basis dictionary in the `uns` attribute.
            The `VecFld2D` key stores an instance of the VectorField2D class which presumably has fixed points, nullcline,
             separatrix, computed and stored.
    """

    if VecFld is None:
        VecFld, func = vecfld_from_adata(adata, basis)
    else:
        func = lambda x: vector_field_function(x, VecFld)

    if dims is None:
        dims = [0, 1]
    X_basis = adata.obsm["X_" + basis][:, dims] if X is None else X[:, dims]
    min_, max_ = X_basis.min(0), X_basis.max(0)

    xlim = [min_[0] - (max_[0] - min_[0]) * 0.1, max_[0] + (max_[0] - min_[0]) * 0.1]
    ylim = [min_[1] - (max_[1] - min_[1]) * 0.1, max_[1] + (max_[1] - min_[1]) * 0.1]

    vecfld = VectorField2D(func, X_data=X_basis)
    vecfld.find_fixed_points_by_sampling(n, xlim, ylim)
    if vecfld.get_num_fixed_points() > 0:
        vecfld.compute_nullclines(xlim, ylim, find_new_fixed_points=True)
    # sep = compute_separatrices(vecfld.Xss.get_X(), vecfld.Xss.get_J(), vecfld.func, xlim, ylim)
    #

    if layer is None:
        if "VecFld_" + basis in adata.uns_keys():
            adata.uns["VecFld_" + basis].update(
                {"VecFld": VecFld, "VecFld2D": vecfld, "xlim": xlim, "ylim": ylim}
            )
        else:
            adata.uns["VecFld_" + basis] = {
                "VecFld": VecFld,
                "VecFld2D": vecfld,
                "xlim": xlim,
                "ylim": ylim,
            }
    else:
        vf_key = "VecFld" if layer == "X" else "VecFld_" + layer
        if "VecFld" in adata.uns_keys():
            adata.uns[vf_key].update(
                {"VecFld": VecFld, "VecFld2D": vecfld, "xlim": xlim, "ylim": ylim}
            )
        else:
            adata.uns[vf_key] = {
                "VecFld": VecFld,
                "VecFld2D": vecfld,
                "xlim": xlim,
                "ylim": ylim,
            }

    return adata


def VectorField(
    adata,
    basis=None,
    layer="X",
    dims=None,
    genes=None,
    normalize=False,
    grid_velocity=False,
    grid_num=50,
    velocity_key="velocity_S",
    method="SparseVFC",
    return_vf_object=False,
    map_topography=True,
    pot_curl_div=False,
    cores=1,
    **kwargs,
):
    """Learn a function of high dimensional vector field from sparse single cell samples in the entire space robustly.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains embedding and velocity data
        basis: `str` or None (default: `None`)
            The embedding data to use. The vector field function will be learned on the low dimensional embedding and can be then
            projected back to the high dimensional space.
        layer: `str` or None (default: `X`)
            Which layer of the data will be used for vector field function reconstruction. The layer once provided, will override
            the `basis` argument and then learn the vector field function in high dimensional space.
        dims: `int`, `list` or None (default: None)
            The dimensions that will be used for reconstructing vector field functions. If it is an `int` all dimension from
            the first dimension to `dims` will be used; if it is a list, the dimensions in the list will be used.
        genes: `list` or None (default: None)
            The gene names whose gene expression will be used for vector field reconstruction. By default (when genes is
            set to None), the genes used for velocity embedding (var.use_for_velocity) will be used for vector field reconstruction.
            Note that the genes to be used need to have velocity calculated.
        normalize: 'bool' (default: False)
            Logic flag to determine whether to normalize the data to have zero means and unit covariance. This is often
            required for raw dataset (for example, raw UMI counts and RNA velocity values in high dimension). But it is
            normally not required for low dimensional embeddings by PCA or other non-linear dimension reduction methods.
        grid_velocity: `bool` (default: False)
            Whether to generate grid velocity. Note that by default it is set to be False, but for datasets with embedding
            dimension less than 4, the grid velocity will still be generated. Please note that number of total grids in
            the space increases exponentially as the number of dimensions increases. So it may quickly lead to lack of
            memory, for example, it cannot allocate the array with grid_num set to be 50 and dimension is 6 (50^6 total
            grids) on 32 G memory computer. Although grid velocity may not be generated, the vector field function can still
            be learned for thousands of dimensions and we can still predict the transcriptomic cell states over long time period.
        grid_num: `int` (default: 50)
            The number of grids in each dimension for generating the grid velocity.
        velocity_key: `str` (default: `velocity_S`)
            The key from the adata layer that corresponds to the velocity matrix.
        method: `str` (default: `sparseVFC`)
            Method that is used to reconstruct the vector field functionally. Currently only SparseVFC supported but other
            improved approaches are under development.
        return_vf_object: `bool` (default: `False`)
            Whether or not to include an instance of a vectorfield class in the the `VecFld` dictionary in the `uns`
            attribute.
        map_topography: `bool` (default: `True`)
            Whether to quantify the topography of the 2D vector field.
        pot_curl_div: `bool` (default: `False`)
            Whether to calculate potential, curl or divergence for each cell. Potential can be calculated for any basis
            while curl and divergence is by default only applied to 2D basis. However, divergence is applicable for any
            dimension while curl is generally only defined for 2/3 D systems.
        cores: `int` (default: 1):
            Number of cores to run the ddhodge function. If cores is set to be > 1, multiprocessing will be used to parallel
            the ddhodge calculation.
        kwargs:
            Other additional parameters passed to the vectorfield class.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            `AnnData` object that is updated with the `VecFld` dictionary in the `uns` attribute.
    """

    if basis is not None:
        X = adata.obsm["X_" + basis].copy()
        V = adata.obsm["velocity_" + basis].copy()

        if np.isscalar(dims):
            X, V = X[:, :dims], V[:, :dims]
        elif type(dims) is list:
            X, V = X[:, dims], V[:, dims]
    else:
        valid_genes = (
            list(set(genes).intersection(adata.var.index))
            if genes is not None
            else adata.var_names[adata.var.use_for_velocity]
        )
        if layer == "X":
            X = adata[:, valid_genes].X.copy()
            X = np.expm1(X)
        else:
            X = inverse_norm(adata, adata.layers[layer])

        V = adata[:, valid_genes].layers[velocity_key].copy()

        if sp.issparse(X):
            X, V = X.A, V.A

    Grid = None
    if X.shape[1] < 4 or grid_velocity:
        # smart way for generating high dimensional grids and convert into a row matrix
        min_vec, max_vec = (
            X.min(0),
            X.max(0),
        )
        min_vec = min_vec - 0.01 * np.abs(max_vec - min_vec)
        max_vec = max_vec + 0.01 * np.abs(max_vec - min_vec)

        Grid_list = np.meshgrid(
            *[np.linspace(i, j, grid_num) for i, j in zip(min_vec, max_vec)]
        )
        Grid = np.array([i.flatten() for i in Grid_list]).T

    if X is None:
        raise Exception(
            f"X is None. Make sure you passed the correct X or {basis} dimension reduction method."
        )
    elif V is None:
        raise Exception("V is None. Make sure you passed the correct V.")

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
    vf_kwargs = update_dict(vf_kwargs, kwargs)

    VecFld = vectorfield(X, V, Grid, **vf_kwargs)
    vf_dict = VecFld.fit(normalize=normalize, method=method, **kwargs)

    vf_key = "VecFld" if basis is None else "VecFld_" + basis

    if basis is not None:
        key = "velocity_" + basis + '_' + method
        adata.obsm[key] = vf_dict['VecFld']['V']
        adata.obsm['X_' + basis + '_' + method] = vf_dict['VecFld']['X']

        vf_dict['dims'] = dims
        adata.uns[vf_key] = vf_dict
    else:
        key = velocity_key + '_' + method
        adata.layers[key] = sp.csr_matrix((adata.shape))
        adata.layers[key][:, np.where(adata.var.use_for_velocity)[0]] = vf_dict['VecFld']['V']

        vf_dict['layer'] = layer
        vf_dict['genes'] = genes
        vf_dict['velocity_key'] = velocity_key
        adata.uns[vf_key] = vf_dict

    if X.shape[1] == 2 and map_topography:
        tp_kwargs = {"n": 25}
        tp_kwargs = update_dict(tp_kwargs, kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            adata = topography(
                adata, basis=basis, X=X, layer=layer, dims=[0, 1], VecFld=vf_dict['VecFld'], **tp_kwargs
            )
    if pot_curl_div:
        if basis in ["pca", 'umap', 'tsne', 'diffusion_map', 'trimap']:
            ddhodge(adata, basis=basis, cores=cores)
            if X.shape[1] == 2: curl(adata, basis=basis)
            divergence(adata, basis=basis)

    if return_vf_object:
        return VecFld


