# create by Yan Zhang, minor adjusted by Xiaojie Qiu
import datetime
import os
import warnings
from typing import Union

import anndata
import numpy as np
import scipy.sparse as sp
from scipy.integrate import odeint
from scipy.linalg import eig
from scipy.optimize import fsolve
from sklearn.neighbors import NearestNeighbors

from ..dynamo_logger import LoggerManager, main_info, main_warning
from ..tools.utils import gaussian_1d, inverse_norm, nearest_neighbors, update_dict
from ..utils import copy_adata
from .FixedPoints import FixedPoints
from .scVectorField import BaseVectorField, SvcVectorField
from .utils import (
    angle,
    dynode_vector_field_function,
    find_fixed_points,
    is_outside,
    remove_redundant_points,
    vecfld_from_adata,
    vector_field_function,
)
from .vector_calculus import curl, divergence


def pac_onestep(x0, func, v0, ds=0.01):
    x01 = x0 + v0 * ds

    def F(x):
        return np.array([func(x), (x - x0).dot(v0) - ds])

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
                # a tiny bit of the end could be chopped off
                ret.append(cur[i_start:i_end])
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


def find_fixed_points_nullcline(func, NCx, NCy, sample_interval=0.5, tol_redundant=1e-4):
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


def dup_osc_idx(x, n_dom=3, tol=0.05):
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


def dup_osc_idx_iter(x, max_iter=5, **kwargs):
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


class VectorField2D:
    def __init__(self, func, func_vx=None, func_vy=None, X_data=None):
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

    def get_Xss_confidence(self, k=50):
        X = self.X_data
        X = X.A if sp.issparse(X) else X
        Xss = self.Xss.get_X()
        Xref = np.median(X, 0)
        Xss = np.vstack((Xss, Xref))

        if X.shape[0] > 200000 and X.shape[1] > 2:
            from pynndescent import NNDescent

            nbrs = NNDescent(
                X,
                metric="euclidean",
                n_neighbors=min(k, X.shape[0] - 1),
                n_jobs=-1,
                random_state=19491001,
            )
            _, dist = nbrs.query(Xss, k=min(k, X.shape[0] - 1))
        else:
            alg = "ball_tree" if X.shape[1] > 10 else "kd_tree"
            nbrs = NearestNeighbors(n_neighbors=min(k, X.shape[0] - 1), algorithm=alg, n_jobs=-1).fit(X)
            dist, _ = nbrs.kneighbors(Xss)

        dist_m = dist.mean(1)
        # confidence = 1 - dist_m / dist_m.max()
        sigma = 0.1 * 0.5 * (np.max(X[:, 0]) - np.min(X[:, 0]) + np.max(X[:, 1]) - np.min(X[:, 1]))
        confidence = gaussian_1d(dist_m, sigma=sigma)
        confidence /= np.max(confidence)
        return confidence[:-1]

    def find_fixed_points_by_sampling(self, n, x_range, y_range, lhs=True, tol_redundant=1e-4):
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
        if len(X) > 0:
            self.Xss.add_fixed_points(X, J, tol_redundant)

    def find_nearest_fixed_point(self, x, x_range, y_range, tol_redundant=1e-4):
        X, J, _ = find_fixed_points(x, self.func, domain=[x_range, y_range], tol_redundant=tol_redundant)
        if len(X) > 0:
            self.Xss.add_fixed_points(X, J, tol_redundant)

    def compute_nullclines(self, x_range, y_range, find_new_fixed_points=False, tol_redundant=1e-4):
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

    def output_to_dict(self, dict_vf):
        dict_vf["NCx"] = self.NCx
        dict_vf["NCy"] = self.NCy
        dict_vf["Xss"] = self.Xss.get_X()
        dict_vf["confidence"] = self.get_Xss_confidence()
        dict_vf["J"] = self.Xss.get_J()
        return dict_vf


def util_topology(adata, basis, X, dims, func, VecFld, n=25, **kwargs):
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

        vecfld = VectorField2D(func, X_data=X_basis)
        vecfld.find_fixed_points_by_sampling(n, xlim, ylim)
        if vecfld.get_num_fixed_points() > 0:
            vecfld.compute_nullclines(xlim, ylim, find_new_fixed_points=True)
            NCx, NCy = vecfld.NCx, vecfld.NCy

        Xss, ftype = vecfld.get_fixed_points(get_types=True)
        confidence = vecfld.get_Xss_confidence()
    else:
        fp_ind = None
        xlim, ylim, confidence, NCx, NCy = None, None, None, None, None
        vecfld = BaseVectorField(
            X=VecFld["X"][VecFld["valid_ind"], :],
            V=VecFld["Y"][VecFld["valid_ind"], :],
            func=func,
        )

        Xss, ftype = vecfld.get_fixed_points(n_x0=n, **kwargs)
        if Xss.ndim > 1 and Xss.shape[1] > 2:
            fp_ind = nearest_neighbors(Xss, vecfld.data["X"], 1).flatten()
            Xss = vecfld.data["X"][fp_ind]

    return X_basis, xlim, ylim, confidence, NCx, NCy, Xss, ftype, fp_ind


def topography(
    adata,
    basis="umap",
    layer=None,
    X=None,
    dims=None,
    n=25,
    VecFld=None,
    **kwargs,
):
    """Map the topography of the single cell vector field in (first) two dimensions.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        basis: `str` (default: `trimap`)
            The reduced dimension embedding of cells to visualize.
        layer: `str` or None (default: None)
            Which layer of the data will be used for vector field function reconstruction. This will be used in
            conjunction with X.
        X: 'np.ndarray' (dimension: n_obs x n_features)
                Original data. Not used
        dims: `list` or `None` (default: `None`)
            The dimensions that will be used for vector field reconstruction.
        n: `int` (default: `10`)
            Number of samples for calculating the fixed points.
        VecFld: `dictionary` or None (default: None)
            The reconstructed vector field function.
        kwargs:
            Key word arguments passed to the find_fixed_point function of the vector field class for high dimension
            fixed point identification.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            `AnnData` object that is updated with the `VecFld` or 'VecFld_' + basis dictionary in the `uns` attribute.
            The `VecFld2D` key stores an instance of the VectorField2D class which presumably has fixed points,
            nullcline, separatrix, computed and stored.
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
        confidence,
        NCx,
        NCy,
        Xss,
        ftype,
        fp_ind,
    ) = util_topology(adata, basis, X, dims, func, VecFld, n=n, *kwargs)

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
                "X_data": X_basis,
                "Xss": Xss,
                "ftype": ftype,
                "confidence": confidence,
                "nullcline": [NCx, NCy],
                "separatrices": None,
                "fp_ind": fp_ind,
            }
        )
    else:
        adata.uns[vf_key] = {
            "xlim": xlim,
            "ylim": ylim,
            "X_data": X_basis,
            "Xss": Xss,
            "ftype": ftype,
            "confidence": confidence,
            "nullcline": [NCx, NCy],
            "separatrices": None,
            "fp_ind": fp_ind,
        }

    return adata


def VectorField(
    adata: anndata.AnnData,
    basis: Union[None, str] = None,
    layer: Union[None, str] = None,
    dims: Union[int, list, None] = None,
    genes: Union[list, None] = None,
    normalize: bool = False,
    grid_velocity: bool = False,
    grid_num: int = 50,
    velocity_key: str = "velocity_S",
    method: str = "SparseVFC",
    min_vel_corr: float = 0.6,
    restart_num: int = 5,
    restart_seed: Union[None, list] = [0, 100, 200, 300, 400],
    model_buffer_path: Union[str, None] = None,
    return_vf_object: bool = False,
    map_topography: bool = False,
    pot_curl_div: bool = False,
    cores: int = 1,
    result_key: Union[str, None] = None,
    copy: bool = False,
    **kwargs,
) -> Union[anndata.AnnData, BaseVectorField]:
    """Learn a function of high dimensional vector field from sparse single cell samples in the entire space robustly.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains embedding and velocity data
        basis:
            The embedding data to use. The vector field function will be learned on the low dimensional embedding and
            can be then projected back to the high dimensional space.
        layer:
            Which layer of the data will be used for vector field function reconstruction. The layer once provided, will
            override the `basis` argument and then learn the vector field function in high dimensional space.
        dims:
            The dimensions that will be used for reconstructing vector field functions. If it is an `int` all dimension
            from the first dimension to `dims` will be used; if it is a list, the dimensions in the list will be used.
        genes:
            The gene names whose gene expression will be used for vector field reconstruction. By default (when genes is
            set to None), the genes used for velocity embedding (var.use_for_transition) will be used for vector field
            reconstruction. Note that the genes to be used need to have velocity calculated.
        normalize:
            Logic flag to determine whether to normalize the data to have zero means and unit covariance. This is often
            required for raw dataset (for example, raw UMI counts and RNA velocity values in high dimension). But it is
            normally not required for low dimensional embeddings by PCA or other non-linear dimension reduction methods.
        grid_velocity:
            Whether to generate grid velocity. Note that by default it is set to be False, but for datasets with
            embedding dimension less than 4, the grid velocity will still be generated. Please note that number of total
            grids in the space increases exponentially as the number of dimensions increases. So it may quickly lead to
            lack of memory, for example, it cannot allocate the array with grid_num set to be 50 and dimension is 6
            (50^6 total grids) on 32 G memory computer. Although grid velocity may not be generated, the vector field
            function can still be learned for thousands of dimensions and we can still predict the transcriptomic cell
            states over long time period.
        grid_num:
            The number of grids in each dimension for generating the grid velocity.
        velocity_key:
            The key from the adata layer that corresponds to the velocity matrix.
        method:
            Method that is used to reconstruct the vector field functionally. Currently only SparseVFC supported but
            other improved approaches are under development.
        min_vel_corr:
            The minimal threshold for the cosine correlation between input velocities and learned velocities to consider
            as a successful vector field reconstruction procedure. If the cosine correlation is less than this
            threshold and restart_num > 1, `restart_num` trials will be attempted with different seeds to reconstruct
            the vector field function. This can avoid some reconstructions to be trapped in some local optimal.
        restart_num:
            The number of retrials for vector field reconstructions.
        restart_seed:
            A list of seeds for each retrial. Must be the same length as `restart_num` or None.
        buffer_path:
               The directory address keeping all the saved/to-be-saved torch variables and NN modules. When `method` is
               set to be `dynode`, buffer_path will set to be
        return_vf_object:
            Whether or not to include an instance of a vectorfield class in the the `VecFld` dictionary in the `uns`
            attribute.
        map_topography:
            Whether to quantify the topography of vector field. Note that for higher than 2D vector field, we can only
            identify fixed points as high-dimensional nullcline and separatrices are mathematically difficult to be
            identified. Nullcline and separatrices will also be a surface or manifold in high-dimensional vector field.
        pot_curl_div:
            Whether to calculate potential, curl or divergence for each cell. Potential can be calculated for any basis
            while curl and divergence is by default only applied to 2D basis. However, divergence is applicable for any
            dimension while curl is generally only defined for 2/3 D systems.
        cores:
            Number of cores to run the ddhodge function. If cores is set to be > 1, multiprocessing will be used to
            parallel the ddhodge calculation.
        result_key:
            The key that will be used as prefix for the vector field key in .uns
        copy:
            Whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments and
            returning `None`.
        kwargs:
            Other additional parameters passed to the vectorfield class.

    Returns
    -------
        adata: :class:`Union[anndata.AnnData, base_vectorfield]`
            If `copy` and `return_vf_object` arguments are set to False, `annData` object is updated with the `VecFld`
            dictionary in the `uns` attribute.
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
    elif method.lower() == "dynode":
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
            elif method.lower() == "dynode":
                train_kwargs = update_dict(train_kwargs, kwargs)
                VecFld = dynode_vectorfield(X, V, Grid, **vf_kwargs)
                # {"VecFld": VecFld.train(**kwargs)}
                cur_vf_dict = VecFld.train(**train_kwargs)

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
        tp_kwargs = {"n": 25}
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
    adata: anndata.AnnData,
    basis: str = "pca",
    cores: int = 1,
    copy: bool = False,
) -> Union[None, anndata.AnnData]:
    """Assign each cell in our data to a fixed point.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains reconstructed vector field in the `basis` space.
        basis:
            The vector field function for the `basis` that will be used to assign fixed points for each cell.
        cores:
            Number of cores to run the fixed-point search for each cell.
        copy:
            Whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments and
            returning `None`.

    Returns
    -------
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
