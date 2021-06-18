from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.sparse import issparse
from scipy.optimize import fsolve
from scipy.linalg import eig
import numdifftools as nd
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mp
import itertools, functools
import inspect
from numba import njit
from ..tools.utils import (
    form_triu_matrix,
    index_condensed_matrix,
    timeit,
    subset_dict_with_key_list,
)
from ..dynamo_logger import LoggerManager


def is_outside_domain(x, domain):
    x = x[None, :] if x.ndim == 1 else x
    return np.any(np.logical_or(x < domain[0], x > domain[1]), axis=1)


def grad(f, x):
    """Gradient of scalar-valued function f evaluated at x"""
    return nd.Gradient(f)(x)


def laplacian(f, x):
    """Laplacian of scalar field f evaluated at x"""
    hes = nd.Hessdiag(f)(x)
    return sum(hes)


# ---------------------------------------------------------------------------------------------------
# vector field function
@timeit
def vector_field_function(x, vf_dict, dim=None, kernel="full", X_ctrl_ind=None, **kernel_kwargs):
    """vector field function constructed by sparseVFC.
    Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition
    """
    # x=np.array(x).reshape((1, -1))
    if "div_cur_free_kernels" in vf_dict.keys():
        has_div_cur_free_kernels = True
    else:
        has_div_cur_free_kernels = False

    # x = np.array(x)
    if x.ndim == 1:
        x = x[None, :]

    if has_div_cur_free_kernels:
        if kernel == "full":
            kernel_ind = 0
        elif kernel == "df_kernel":
            kernel_ind = 1
        elif kernel == "cf_kernel":
            kernel_ind = 2
        else:
            raise ValueError(f"the kernel can only be one of {'full', 'df_kernel', 'cf_kernel'}!")

        K = con_K_div_cur_free(
            x,
            vf_dict["X_ctrl"],
            vf_dict["sigma"],
            vf_dict["eta"],
            **kernel_kwargs,
        )[kernel_ind]
    else:
        Xc = vf_dict["X_ctrl"]
        K = con_K(x, Xc, vf_dict["beta"], **kernel_kwargs)

    if X_ctrl_ind is not None:
        C = np.zeros_like(vf_dict["C"])
        C[X_ctrl_ind, :] = vf_dict["C"][X_ctrl_ind, :]
    else:
        C = vf_dict["C"]

    K = K.dot(C)

    if dim is not None and not has_div_cur_free_kernels:
        if np.isscalar(dim):
            K = K[:, :dim]
        elif dim is not None:
            K = K[:, dim]

    return K


def dynode_vector_field_function(x, vf_dict, dim=None, **kwargs):
    try:
        import dynode
        from dynode.vectorfield import Dynode
    except ImportError:
        raise ImportError("You need to install the package `dynode`." "install dynode via `pip install dynode`")
    vf_dict["parameters"]["load_model_from_buffer"] = True
    dynode_inspect = inspect.getfullargspec(Dynode)
    dynode_dict = subset_dict_with_key_list(vf_dict["parameters"], dynode_inspect.args)

    nn = Dynode(**dynode_dict)

    to_flatten = False
    if x.ndim == 1:
        to_flatten = True
        x = x[None, :]

    res = nn.predict_velocity(input_x=x)

    if dim is not None:
        if np.isscalar(dim):
            res = res[:, :dim]
        elif dim is not None:
            res = res[:, dim]

    if to_flatten:
        res = res.flatten()

    return res


@timeit
def con_K(x, y, beta, method="cdist", return_d=False):
    """con_K constructs the kernel K, where K(i, j) = k(x, y) = exp(-beta * ||x - y||^2).

    Arguments
    ---------
        x: :class:`~numpy.ndarray`
            Original training data points.
        y: :class:`~numpy.ndarray`
            Control points used to build kernel basis functions.
        beta: float (default: 0.1)
            Paramerter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2),
        return_d: bool
            If True the intermediate 3D matrix x - y will be returned for analytical Jacobian.

    Returns
    -------
    K: :class:`~numpy.ndarray`
    the kernel to represent the vector field function.
    """
    if method == "cdist" and not return_d:
        K = cdist(x, y, "sqeuclidean")
        if len(K) == 1:
            K = K.flatten()
    else:
        n = x.shape[0]
        m = y.shape[0]

        # https://stackoverflow.com/questions/1721802/what-is-the-equivalent-of-matlabs-repmat-in-numpy
        # https://stackoverflow.com/questions/12787475/matlabs-permute-in-python
        D = np.matlib.tile(x[:, :, None], [1, 1, m]) - np.transpose(np.matlib.tile(y[:, :, None], [1, 1, n]), [2, 1, 0])
        K = np.squeeze(np.sum(D ** 2, 1))
    K = -beta * K
    K = np.exp(K)

    if return_d:
        return K, D
    else:
        return K


@timeit
def con_K_div_cur_free(x, y, sigma=0.8, eta=0.5):
    """Construct a convex combination of the divergence-free kernel T_df and curl-free kernel T_cf with a bandwidth sigma
    and a combination coefficient gamma.

    Arguments
    ---------
        x: :class:`~numpy.ndarray`
            Original training data points.
        y: :class:`~numpy.ndarray`
            Control points used to build kernel basis functions
        sigma: int (default: `0.8`)
            Bandwidth parameter.
        eta: int (default: `0.5`)
            Combination coefficient for the divergence-free or the curl-free kernels.

    Returns
    -------
        A tuple of G (the combined kernel function), divergence-free kernel and curl-free kernel.

    See also: :func:`sparseVFC`.
    """
    m, d = x.shape
    n, d = y.shape
    sigma2 = sigma ** 2
    G_tmp = np.matlib.tile(x[:, :, None], [1, 1, n]) - np.transpose(np.matlib.tile(y[:, :, None], [1, 1, m]), [2, 1, 0])
    G_tmp = np.squeeze(np.sum(G_tmp ** 2, 1))
    G_tmp3 = -G_tmp / sigma2
    G_tmp = -G_tmp / (2 * sigma2)
    G_tmp = np.exp(G_tmp) / sigma2
    G_tmp = np.kron(G_tmp, np.ones((d, d)))

    x_tmp = np.matlib.tile(x, [n, 1])
    y_tmp = np.matlib.tile(y, [1, m]).T
    y_tmp = y_tmp.reshape((d, m * n), order="F").T
    xminusy = x_tmp - y_tmp
    G_tmp2 = np.zeros((d * m, d * n))

    tmp4_ = np.zeros((d, d))
    for i in tqdm(range(d), desc="Iterating each dimension in con_K_div_cur_free:"):
        for j in np.arange(i, d):
            tmp1 = xminusy[:, i].reshape((m, n), order="F")
            tmp2 = xminusy[:, j].reshape((m, n), order="F")
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


def get_vf_dict(adata, basis="", vf_key="VecFld"):
    if basis is not None or len(basis) > 0:
        vf_key = "%s_%s" % (vf_key, basis)

    if vf_key not in adata.uns.keys():
        raise ValueError(
            f"Vector field function {vf_key} is not included in the adata object! "
            f"Try firstly running dyn.vf.VectorField(adata, basis='{basis}')"
        )

    vf_dict = adata.uns[vf_key]
    return vf_dict


def vecfld_from_adata(adata, basis="", vf_key="VecFld"):
    vf_dict = get_vf_dict(adata, basis=basis, vf_key=vf_key)

    method = vf_dict["method"]
    if method.lower() == "sparsevfc":
        func = lambda x: vector_field_function(x, vf_dict)
    elif method.lower() == "dynode":
        func = lambda x: dynode_vector_field_function(x, vf_dict)
    else:
        raise ValueError(f"current only support two methods, SparseVFC and dynode")

    return vf_dict, func


def vector_transformation(V, Q):
    """Transform vectors from PCA space to the original space using the formula:
                    :math:`\hat{v} = v Q^T`,
    where `Q, v, \hat{v}` are the PCA loading matrix, low dimensional vector and the
    transformed high dimensional vector.

    Parameters
    ----------
        V: :class:`~numpy.ndarray`
            The n x k array of vectors to be transformed, where n is the number of vectors,
            k the dimension.
        Q: :class:`~numpy.ndarray`
            PCA loading matrix with dimension d x k, where d is the dimension of the original space,
            and k the number of leading PCs.

    Returns
    -------
        ret: :class:`~numpy.ndarray`
            The array of transformed vectors.

    """
    return V @ Q.T


def vector_field_function_transformation(vf_func, Q):
    """Transform vector field function from PCA space to the original space.
    The formula used for transformation:
                                            :math:`\hat{f} = f Q^T`,
    where `Q, f, \hat{f}` are the PCA loading matrix, low dimensional vector field function and the
    transformed high dimensional vector field function.

    Parameters
    ----------
        vf_func: callable
            The vector field function.
        Q: :class:`~numpy.ndarray`
            PCA loading matrix with dimension d x k, where d is the dimension of the original space,
            and k the number of leading PCs.

    Returns
    -------
        ret: callable
            The transformed vector field function.

    """
    return lambda x: vf_func.func(x) @ Q.T


# ---------------------------------------------------------------------------------------------------
# jacobian
def Jacobian_rkhs_gaussian(x, vf_dict, vectorize=False):
    """analytical Jacobian for RKHS vector field functions with Gaussian kernel.

    Arguments
    ---------
    x: :class:`~numpy.ndarray`
        Coordinates where the Jacobian is evaluated.
    vf_dict: dict
        A dictionary containing RKHS vector field control points, Gaussian bandwidth,
        and RKHS coefficients.
        Essential keys: 'X_ctrl', 'beta', 'C'

    Returns
    -------
    J: :class:`~numpy.ndarray`
        Jacobian matrices stored as d-by-d-by-n numpy arrays evaluated at x.
        d is the number of dimensions and n the number of coordinates in x.
    """
    if x.ndim == 1:
        K, D = con_K(x[None, :], vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
        J = (vf_dict["C"].T * K) @ D[0].T
    elif not vectorize:
        n, d = x.shape
        J = np.zeros((d, d, n))
        for i, xi in enumerate(x):
            K, D = con_K(xi[None, :], vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
            J[:, :, i] = (vf_dict["C"].T * K) @ D[0].T
    else:
        K, D = con_K(x, vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
        if K.ndim == 1:
            K = K[None, :]
        J = np.einsum("nm, mi, njm -> ijn", K, vf_dict["C"], D)

    return -2 * vf_dict["beta"] * J


def Jacobian_rkhs_gaussian_parallel(x, vf_dict, cores=None):
    n = len(x)
    if cores is None:
        cores = mp.cpu_count()
    n_j_per_core = int(np.ceil(n / cores))
    xx = []
    for i in range(0, n, n_j_per_core):
        xx.append(x[i : i + n_j_per_core])
    # with mp.Pool(cores) as p:
    #    ret = p.starmap(Jacobian_rkhs_gaussian, zip(xx, itertools.repeat(vf_dict)))
    with ThreadPool(cores) as p:
        ret = p.starmap(Jacobian_rkhs_gaussian, zip(xx, itertools.repeat(vf_dict)))
    ret = [np.transpose(r, axes=(2, 0, 1)) for r in ret]
    ret = np.transpose(np.vstack(ret), axes=(1, 2, 0))
    return ret


def Jacobian_numerical(f, input_vector_convention="row"):
    """
    Get the numerical Jacobian of the vector field function.
    If the input_vector_convention is 'row', it means that fjac takes row vectors
    as input, otherwise the input should be an array of column vectors. Note that
    the returned Jacobian would behave exactly the same if the input is an 1d array.

    The column vector convention is slightly faster than the row vector convention.
    So the matrix of row vector convention is converted into column vector convention
    under the hood.

    No matter the input vector convention, the returned Jacobian is of the following
    format:
            df_1/dx_1   df_1/dx_2   df_1/dx_3   ...
            df_2/dx_1   df_2/dx_2   df_2/dx_3   ...
            df_3/dx_1   df_3/dx_2   df_3/dx_3   ...
            ...         ...         ...         ...
    """
    fjac = nd.Jacobian(lambda x: f(x.T).T)
    if input_vector_convention == "row" or input_vector_convention == 0:

        def f_aux(x):
            x = x.T
            return fjac(x)

        return f_aux
    else:
        return fjac


@timeit
def elementwise_jacobian_transformation(Js, qi, qj):
    """Inverse transform low dimensional k x k Jacobian matrix (:math:`\partial F_i / \partial x_j`) back to the
    d-dimensional gene expression space. The formula used to inverse transform Jacobian matrix calculated from
    low dimension (PCs) is:
                                            :math:`Jac = Q J Q^T`,
    where `Q, J, Jac` are the PCA loading matrix, low dimensional Jacobian matrix and the inverse transformed high
    dimensional Jacobian matrix. This function takes only one row from Q to form qi or qj.

    Parameters
    ----------
        Js: :class:`~numpy.ndarray`
            k x k x n matrices of n k-by-k Jacobians.
        qi: :class:`~numpy.ndarray`
            The i-th row of the PC loading matrix Q with dimension d x k, corresponding to the regulator gene i.
        qj: :class:`~numpy.ndarray`
            The j-th row of the PC loading matrix Q with dimension d x k, corresponding to the effector gene j.

    Returns
    -------
        ret: :class:`~numpy.ndarray`
            The calculated vector of Jacobian matrix (:math:`\partial F_i / \partial x_j`) for each cell.
    """

    Js = np.atleast_3d(Js)
    n = Js.shape[2]
    ret = np.zeros(n)
    for i in tqdm(range(n), "calculating Jacobian for each cell"):
        ret[i] = qi @ Js[:, :, i] @ qj

    return ret


@timeit
def subset_jacobian_transformation(Js, Qi, Qj, cores=1):
    """Transform Jacobian matrix (:math:`\partial F_i / \partial x_j`) from PCA space to the original space.
    The formula used for transformation:
                                            :math:`\hat{J} = Q J Q^T`,
    where `Q, J, \hat{J}` are the PCA loading matrix, low dimensional Jacobian matrix and the inverse transformed high
    dimensional Jacobian matrix. This function takes multiple rows from Q to form Qi or Qj.

    Parameters
    ----------
        fjac: callable
            The function for calculating numerical Jacobian matrix.
        X: :class:`~numpy.ndarray`
            The samples coordinates with dimension n_obs x n_PCs, from which Jacobian will be calculated.
        Qi: :class:`~numpy.ndarray`
            Sampled genes' PCA loading matrix with dimension n' x n_PCs, from which local dimension Jacobian matrix (k x k)
            will be inverse transformed back to high dimension.
        Qj: :class:`~numpy.ndarray`
            Sampled genes' (sample genes can be the same as those in Qi or different) PCs loading matrix with dimension
            n' x n_PCs, from which local dimension Jacobian matrix (k x k) will be inverse transformed back to high dimension.
        cores: int (default: 1):
            Number of cores to calculate Jacobian. If cores is set to be > 1, multiprocessing will be used to
            parallel the Jacobian calculation.
        return_J: bool (default: False)
            Whether to return the raw tensor of Jacobian matrix of each cell before transformation.

    Returns
    -------
        ret: :class:`~numpy.ndarray`
            The calculated Jacobian matrix (n_gene x n_gene x n_obs) for each cell.
    """

    Js = np.atleast_3d(Js)
    Qi = np.atleast_2d(Qi)
    Qj = np.atleast_2d(Qj)
    d1, d2, n = Qi.shape[0], Qj.shape[0], Js.shape[2]

    ret = np.zeros((d1, d2, n))

    if cores == 1:
        ret = transform_jacobian(Js, Qi, Qj, pbar=True)
    else:
        if cores is None:
            cores = mp.cpu_count()
        n_j_per_core = int(np.ceil(n / cores))
        JJ = []
        for i in range(0, n, n_j_per_core):
            JJ.append(Js[:, :, i : i + n_j_per_core])
        with ThreadPool(cores) as p:
            ret = p.starmap(
                transform_jacobian,
                zip(JJ, itertools.repeat(Qi), itertools.repeat(Qj)),
            )
        ret = [np.transpose(r, axes=(2, 0, 1)) for r in ret]
        ret = np.transpose(np.vstack(ret), axes=(1, 2, 0))

    return ret


def transform_jacobian(Js, Qi, Qj, pbar=False):
    d1, d2, n = Qi.shape[0], Qj.shape[0], Js.shape[2]
    ret = np.zeros((d1, d2, n), dtype=np.float32)
    if pbar:
        iterj = tqdm(range(n), desc="Transforming subset Jacobian")
    else:
        iterj = range(n)
    for i in iterj:
        J = Js[:, :, i]
        ret[:, :, i] = Qi @ J @ Qj.T
    return ret


def average_jacobian_by_group(Js, group_labels):
    """
    Returns a dictionary of averaged jacobians with group names as the keys.
    No vectorized indexing was used due to its high memory cost.
    """
    d1, d2, _ = Js.shape
    groups = np.unique(group_labels)

    J_mean = {}
    N = {}
    for i, g in enumerate(group_labels):
        if g in J_mean.keys():
            J_mean[g] += Js[:, :, i]
            N[g] += 1
        else:
            J_mean[g] = Js[:, :, i]
            N[g] = 1
    for g in groups:
        J_mean[g] /= N[g]
    return J_mean


# ---------------------------------------------------------------------------------------------------
# dynamical properties
def _divergence(f, x):
    """Divergence of the reconstructed vector field function f evaluated at x"""
    jac = nd.Jacobian(f)(x)
    return np.trace(jac)


@timeit
def compute_divergence(f_jac, X, vectorize_size=1):
    """Calculate divergence for many samples by taking the trace of a Jacobian matrix.

    vectorize_size is used to control the number of samples computed in each vectorized batch.
        If vectorize_size = 1, there's no vectorization whatsoever.
        If vectorize_size = None, all samples are vectorized.
    """
    n = len(X)
    if vectorize_size is None:
        vectorize_size = n

    div = np.zeros(n)
    for i in tqdm(range(0, n, vectorize_size), desc="Calculating divergence"):
        J = f_jac(X[i : i + vectorize_size])
        div[i : i + vectorize_size] = np.trace(J)
    return div


def acceleration_(v, J):
    if v.ndim == 1:
        v = v[:, None]
    return J.dot(v)


def curvature_method1(a, v):
    """https://link.springer.com/article/10.1007/s12650-018-0474-6"""
    if v.ndim == 1:
        v = v[:, None]
    kappa = np.linalg.norm(np.outer(v, a)) / np.linalg.norm(v) ** 3

    return kappa


def curvature_method2(a, v):
    """https://dl.acm.org/doi/10.5555/319351.319441"""
    # if v.ndim == 1: v = v[:, None]
    kappa = (np.multiply(a, np.dot(v, v)) - np.multiply(v, np.dot(v, a))) / np.linalg.norm(v) ** 4

    return kappa


def torsion_(v, J, a):
    """only works in 3D"""
    if v.ndim == 1:
        v = v[:, None]
    tau = np.outer(v, a).dot(J.dot(a)) / np.linalg.norm(np.outer(v, a)) ** 2

    return tau


@timeit
def compute_acceleration(vf, f_jac, X, return_all=False):
    """Calculate acceleration for many samples via

    .. math::
    a = J \cdot v.

    """
    n = len(X)
    acce = np.zeros(n)
    acce_mat = np.zeros((n, X.shape[1]))

    v_ = vf(X)
    J_ = f_jac(X)
    temp_logger = LoggerManager.get_temp_timer_logger()
    for i in LoggerManager.progress_logger(range(n), temp_logger, progress_name="Calculating acceleration"):
        v = v_[i]
        J = J_[:, :, i]
        acce_mat[i] = acceleration_(v, J).flatten()
        acce[i] = np.linalg.norm(acce_mat[i])

    if return_all:
        return v_, J_, acce, acce_mat
    else:
        return acce, acce_mat


@timeit
def compute_curvature(vf, f_jac, X, formula=2):
    """Calculate curvature for many samples via

    Formula 1:
    .. math::
    \kappa = \frac{||\mathbf{v} \times \mathbf{a}||}{||\mathbf{V}||^3}

    Formula 2:
    .. math::
    \kappa = \frac{||\mathbf{Jv} (\mathbf{v} \cdot \mathbf{v}) -  ||\mathbf{v} (\mathbf{v} \cdot \mathbf{Jv})}{||\mathbf{V}||^4}
    """
    n = len(X)

    curv = np.zeros(n)
    v, _, _, a = compute_acceleration(vf, f_jac, X, return_all=True)
    cur_mat = np.zeros((n, X.shape[1])) if formula == 2 else None

    for i in LoggerManager.progress_logger(range(n), progress_name="Calculating curvature"):
        if formula == 1:
            curv[i] = curvature_method1(a[i], v[i])
        elif formula == 2:
            cur_mat[i] = curvature_method2(a[i], v[i])
            curv[i] = np.linalg.norm(cur_mat[i])

    return curv, cur_mat


@timeit
def compute_torsion(vf, f_jac, X):
    """Calculate torsion for many samples via

    .. math::
    \tau = \frac{(\mathbf{v} \times \mathbf{a}) \cdot (\mathbf{J} \cdot \mathbf{a})}{||\mathbf{V} \times \mathbf{a}||^2}
    """
    if X.shape[1] != 3:
        raise Exception(f"torsion is only defined in 3 dimension.")

    n = len(X)

    tor = np.zeros((n, X.shape[1], X.shape[1]))
    v, J, a = compute_acceleration(vf, f_jac, X, return_all=True)

    for i in tqdm(range(n), desc="Calculating torsion"):
        tor[i] = torsion_(v[i], J[:, :, i], a[i])

    return tor


@timeit
def compute_sensitivity(f_jac, X):
    """Calculate sensitivity for many samples via

    .. math::
    S = (I - J)^{-1} D(\frac{1}{{I-J}^{-1}})
    """
    J = f_jac(X)

    n_genes, n_genes_, n_cells = J.shape
    S = np.zeros_like(J)

    I = np.eye(n_genes)
    for i in tqdm(
        np.arange(n_cells),
        desc="Calculating sensitivity matrix with precomputed component-wise Jacobians",
    ):
        s = np.linalg.inv(I - J[:, :, i])  # np.transpose(J)
        S[:, :, i] = s.dot(np.diag(1 / np.diag(s)))
        # tmp = np.transpose(J[:, :, i])
        # s = np.linalg.inv(I - tmp)
        # S[:, :, i] = s * (1 / np.diag(s)[None, :])

    return S


def _curl(f, x, method="analytical", VecFld=None, jac=None):
    """Curl of the reconstructed vector field f evaluated at x in 3D"""
    if jac is None:
        if method == "analytical" and VecFld is not None:
            jac = Jacobian_rkhs_gaussian(x, VecFld)
        else:
            jac = nd.Jacobian(f)(x)

    return np.array([jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]])


def curl2d(f, x, method="analytical", VecFld=None, jac=None):
    """Curl of the reconstructed vector field f evaluated at x in 2D"""
    if jac is None:
        if method == "analytical" and VecFld is not None:
            jac = Jacobian_rkhs_gaussian(x, VecFld)
        else:
            jac = nd.Jacobian(f)(x)

    curl = jac[1, 0] - jac[0, 1]

    return curl


@timeit
def compute_curl(f_jac, X):
    """Calculate curl for many samples for 2/3 D systems."""
    if X.shape[1] > 3:
        raise Exception(f"curl is only defined in 2/3 dimension.")

    n = len(X)

    if X.shape[1] == 2:
        curl = np.zeros(n)
        f = curl2d
    else:
        curl = np.zeros((n, 2, 2))
        f = _curl

    for i in tqdm(range(n), desc=f"Calculating {X.shape[1]}-D curl"):
        J = f_jac(X[i])
        curl[i] = f(None, None, method="analytical", VecFld=None, jac=J)

    return curl


# ---------------------------------------------------------------------------------------------------
# ranking related utilies
def get_metric_gene_in_rank(mat, genes, neg=False):
    metric_in_rank = mat.mean(0).A1 if issparse(mat) else mat.mean(0)
    rank = metric_in_rank.argsort() if neg else metric_in_rank.argsort()[::-1]
    metric_in_rank, genes_in_rank = metric_in_rank[rank], genes[rank]

    return metric_in_rank, genes_in_rank


def get_metric_gene_in_rank_by_group(mat, genes, groups, grp, neg=False):
    mask = groups == grp
    if type(mask) == pd.Series:
        mask = mask.values

    gene_wise_metrics, group_wise_metrics = (
        mat[mask, :].mean(0).A1 if issparse(mat) else mat[mask, :].mean(0),
        mat[mask, :].mean(0).A1 if issparse(mat) else mat[mask, :].mean(0),
    )
    rank = gene_wise_metrics.argsort() if neg else gene_wise_metrics.argsort()[::-1]
    gene_wise_metrics, genes_in_rank = gene_wise_metrics[rank], genes[rank]

    return gene_wise_metrics, group_wise_metrics, genes_in_rank


def get_sorted_metric_genes_df(df, genes, neg=False):
    sorted_metric = pd.DataFrame(
        {
            key: (sorted(values, reverse=False) if neg else sorted(values, reverse=True))
            for key, values in df.transpose().iterrows()
        }
    )
    sorted_genes = pd.DataFrame(
        {
            key: (genes[values.argsort()] if neg else genes[values.argsort()[::-1]])
            for key, values in df.transpose().iterrows()
        }
    )
    return sorted_metric, sorted_genes


def rank_vector_calculus_metrics(mat, genes, group, groups, uniq_group):
    if issparse(mat):
        mask = mat.data > 0
        pos_mat, neg_mat = mat.copy(), mat.copy()
        pos_mat.data[~mask], neg_mat.data[mask] = 0, 0
        pos_mat.eliminate_zeros()
        neg_mat.eliminate_zeros()
    else:
        mask = mat > 0
        pos_mat, neg_mat = mat.copy(), mat.copy()
        pos_mat[~mask], neg_mat[mask] = 0, 0

    if group is None:
        metric_in_rank, genes_in_rank = get_metric_gene_in_rank(abs(mat), genes)

        pos_metric_in_rank, pos_genes_in_rank = get_metric_gene_in_rank(pos_mat, genes)

        neg_metric_in_rank, neg_genes_in_rank = get_metric_gene_in_rank(neg_mat, genes, neg=True)

        return (
            metric_in_rank,
            genes_in_rank,
            pos_metric_in_rank,
            pos_genes_in_rank,
            neg_metric_in_rank,
            neg_genes_in_rank,
        )
    else:
        (
            gene_wise_metrics,
            gene_wise_genes,
            gene_wise_pos_metrics,
            gene_wise_pos_genes,
            gene_wise_neg_metrics,
            gene_wise_neg_genes,
        ) = ({}, {}, {}, {}, {}, {})
        (
            group_wise_metrics,
            group_wise_genes,
            group_wise_pos_metrics,
            group_wise_pos_genes,
            group_wise_neg_metrics,
            group_wise_neg_genes,
        ) = ({}, {}, {}, {}, {}, {})
        for i, grp in tqdm(enumerate(uniq_group), desc="ranking genes across gropus"):
            (
                gene_wise_metrics[grp],
                group_wise_metrics[grp],
                gene_wise_genes[grp],
            ) = (None, None, None)
            (
                gene_wise_metrics[grp],
                group_wise_metrics[grp],
                gene_wise_genes[grp],
            ) = get_metric_gene_in_rank_by_group(abs(mat), genes, groups, grp)

            (
                gene_wise_pos_metrics[grp],
                group_wise_pos_metrics[grp],
                gene_wise_pos_genes[grp],
            ) = (None, None, None)
            (
                gene_wise_pos_metrics[grp],
                group_wise_pos_metrics[grp],
                gene_wise_pos_genes[grp],
            ) = get_metric_gene_in_rank_by_group(pos_mat, genes, groups, grp)

            (
                gene_wise_neg_metrics[grp],
                group_wise_neg_metrics[grp],
                gene_wise_neg_genes[grp],
            ) = (None, None, None)
            (
                gene_wise_neg_metrics[grp],
                group_wise_neg_metrics[grp],
                gene_wise_neg_genes[grp],
            ) = get_metric_gene_in_rank_by_group(neg_mat, genes, groups, grp, neg=True)

        (
            metric_in_group_rank_by_gene,
            genes_in_group_rank_by_gene,
        ) = get_sorted_metric_genes_df(pd.DataFrame(group_wise_metrics), genes)
        (
            pos_metric_gene_rank_by_group,
            pos_genes_group_rank_by_gene,
        ) = get_sorted_metric_genes_df(pd.DataFrame(group_wise_pos_metrics), genes)
        (
            neg_metric_in_group_rank_by_gene,
            neg_genes_in_group_rank_by_gene,
        ) = get_sorted_metric_genes_df(pd.DataFrame(group_wise_neg_metrics), genes, neg=True)

        metric_in_gene_rank_by_group, genes_in_gene_rank_by_group = (
            pd.DataFrame(gene_wise_metrics),
            pd.DataFrame(gene_wise_genes),
        )
        pos_metric_in_gene_rank_by_group, pos_genes_in_gene_rank_by_group = (
            pd.DataFrame(gene_wise_pos_metrics),
            pd.DataFrame(gene_wise_pos_genes),
        )
        neg_metric_in_gene_rank_by_group, neg_genes_in_gene_rank_by_group = (
            pd.DataFrame(gene_wise_neg_metrics),
            pd.DataFrame(gene_wise_neg_genes),
        )

        return (
            metric_in_gene_rank_by_group,
            genes_in_gene_rank_by_group,
            pos_metric_in_gene_rank_by_group,
            pos_genes_in_gene_rank_by_group,
            neg_metric_in_gene_rank_by_group,
            neg_genes_in_gene_rank_by_group,
            metric_in_group_rank_by_gene,
            genes_in_group_rank_by_gene,
            pos_metric_gene_rank_by_group,
            pos_genes_group_rank_by_gene,
            neg_metric_in_group_rank_by_gene,
            neg_genes_in_group_rank_by_gene,
        )


# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python#answer-13849249
# answer from crizCraig
@njit(cache=True, nogil=True)
def angle(vector1, vector2):
    """ Returns the angle in radians between given vectors"""
    v1_norm, v1_u = unit_vector(vector1)
    v2_norm, v2_u = unit_vector(vector2)

    if v1_norm == 0 or v2_norm == 0:
        return np.nan
    else:
        minor = np.linalg.det(np.stack((v1_u[-2:], v2_u[-2:])))
        if minor == 0:
            sign = 1
        else:
            sign = -np.sign(minor)
        dot_p = np.dot(v1_u, v2_u)
        dot_p = min(max(dot_p, -1.0), 1.0)
        return sign * np.arccos(dot_p)


@njit(cache=True, nogil=True)
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    vec_norm = np.linalg.norm(vector)
    if vec_norm == 0:
        return vec_norm, vector
    else:
        return vec_norm, vector / vec_norm


def normalize_vectors(vectors, axis=1, **kwargs):
    """ Returns the unit vectors of the vectors.  """
    vec = np.array(vectors, copy=True)
    vec = np.atleast_2d(vec)
    vec_norm = np.linalg.norm(vec, axis=axis, **kwargs)

    vec_norm[vec_norm == 0] = 1
    vec = (vec.T / vec_norm).T
    return vec


# ---------------------------------------------------------------------------------------------------
# topology related utilies


def is_outside(X, domain):
    is_outside = np.zeros(X.shape[0], dtype=bool)
    for k in range(X.shape[1]):
        o = np.logical_or(X[:, k] < domain[k][0], X[:, k] > domain[k][1])
        is_outside = np.logical_or(is_outside, o)
    return is_outside


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


def find_fixed_points(X0, func_vf, domain=None, tol_redundant=1e-4, return_all=False):
    X = []
    J = []
    fval = []
    for x0 in X0:
        x, info_dict, _, _ = fsolve(func_vf, x0, full_output=True)

        outside = is_outside(x[None, :], domain)[0] if domain is not None else False
        if not outside:
            fval.append(info_dict["fvec"])
            # compute Jacobian
            Q = info_dict["fjac"]
            R = form_triu_matrix(info_dict["r"])
            J.append(Q.T @ R)
            X.append(x)
        elif return_all:
            X.append(np.zeros_like(x) * np.nan)
            J.append(np.zeros((len(x), len(x))) * np.nan)

    X = np.array(X)
    J = np.array(J)
    fval = np.array(fval)

    if return_all:
        return X, J, fval
    else:
        if X.size != 0:
            if tol_redundant is not None:
                X, discard = remove_redundant_points(X, tol_redundant, output_discard=True)
                J = J[~discard]
                fval = fval[~discard]

            return X, J, fval
        else:
            return None, None, None


class FixedPoints:
    def __init__(self, X=None, J=None):
        self.X = X if X is not None else []
        self.J = J if J is not None else []
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
            if self.J[i] is None or np.isnan(self.J[i]).any():
                w = np.nan
            else:
                w, _ = eig(self.J[i])
            self.eigvals.append(w)

    def is_stable(self):
        if len(self.eigvals) != len(self.X):
            self.compute_eigvals()

        stable = np.ones(len(self.eigvals), dtype=bool)
        for i, w in enumerate(self.eigvals):
            if w is None or np.isnan(w).any():
                stable[i] = np.nan
            else:
                if np.any(np.real(w) >= 0):
                    stable[i] = False
        return stable

    def is_saddle(self):
        is_stable = self.is_stable()
        saddle = np.zeros(len(self.eigvals), dtype=bool)
        for i, w in enumerate(self.eigvals):
            if w is None or np.isnan(w).any():
                saddle[i] = np.nan
            else:
                if not is_stable[i] and np.any(np.real(w) < 0):
                    saddle[i] = True
        return saddle, is_stable

    def get_fixed_point_types(self):
        is_saddle, is_stable = self.is_saddle()
        # -1 -- stable, 0 -- saddle, 1 -- unstable
        ftype = np.ones(len(self.X))
        for i in range(len(ftype)):
            if self.X[i] is None or np.isnan((self.X[i])).any():
                ftype[i] = np.nan
            else:
                if is_saddle[i]:
                    ftype[i] = 0
                elif is_stable[i]:
                    ftype[i] = -1
        return ftype


# ---------------------------------------------------------------------------------------------------
# data retrieval related utilies
def intersect_sources_targets(regulators, regulators_, effectors, effectors_, Der):
    regulators = regulators_ if regulators is None else regulators
    effectors = effectors_ if effectors is None else effectors
    if type(regulators) == str:
        regulators = [regulators]
    if type(effectors) == str:
        effectors = [effectors]
    regulators = list(set(regulators_).intersection(regulators))
    effectors = list(set(effectors_).intersection(effectors))
    if len(regulators) == 0 or len(effectors) == 0:
        raise ValueError(
            f"Jacobian related to source genes {regulators} and target genes {effectors}"
            f"you provided are existed. Available source genes includes {regulators_} while "
            f"available target genes includes {effectors_}"
        )
    # subset Der with correct index of selected source / target genes
    valid_source_idx = [i for i, e in enumerate(regulators_) if e in regulators]
    valid_target_idx = [i for i, e in enumerate(effectors_) if e in effectors]
    Der = Der[valid_target_idx, :, :][:, valid_source_idx, :] if len(regulators_) + len(effectors_) > 2 else Der
    regulators, effectors = (
        np.array(regulators_)[valid_source_idx],
        np.array(effectors_)[valid_target_idx],
    )

    return Der, regulators, effectors
