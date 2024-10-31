import itertools
import multiprocessing as mp
import sys
from multiprocessing.dummy import Pool as ThreadPool
from typing import Callable, Dict, List, Optional, Tuple, Union

import numdifftools as nd
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.optimize import fsolve
from scipy.sparse import issparse
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm

from ..dynamo_logger import LoggerManager, main_info
from ..tools.utils import form_triu_matrix, index_condensed_matrix, timeit
from .FixedPoints import FixedPoints

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class NormDict(TypedDict):
    xm: np.ndarray
    ym: np.ndarray
    xscale: float
    yscale: float
    fix_velocity: bool


class VecFldDict(TypedDict):
    X: np.ndarray
    valid_ind: float
    X_ctrl: np.ndarray
    ctrl_idx: float
    Y: np.ndarray
    beta: float
    V: np.ndarray
    C: np.ndarray
    P: np.ndarray
    VFCIndex: np.ndarray
    sigma2: float
    grid: np.ndarray
    grid_V: np.ndarray
    iteration: int
    tecr_traj: np.ndarray
    E_traj: np.ndarray
    norm_dict: NormDict


def is_outside_domain(x: np.ndarray, domain: Tuple[float, float]) -> np.ndarray:
    x = x[None, :] if x.ndim == 1 else x
    return np.any(np.logical_or(x < domain[0], x > domain[1]), axis=1)


def grad(f: Callable, x: np.ndarray) -> nd.Gradient:
    """Gradient of scalar-valued function f evaluated at x"""
    return nd.Gradient(f)(x)


def laplacian(f: Callable, x: np.ndarray) -> float:
    """Laplacian of scalar field f evaluated at x"""
    hes = nd.Hessdiag(f)(x)
    return sum(hes)


# ---------------------------------------------------------------------------------------------------
# vector field function
@timeit
def vector_field_function(
    x: np.ndarray,
    vf_dict: VecFldDict,
    dim: Optional[Union[int, np.ndarray]] = None,
    kernel: str = "full",
    X_ctrl_ind: Optional[List] = None,
    **kernel_kwargs,
) -> np.ndarray:
    """vector field function constructed by sparseVFC.
    Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition

    Args:
        x: Set of cell expression state samples
        vf_dict: VecFldDict with stored parameters necessary for reconstruction
        dim: Index or indices of dimensions of the K gram matrix to return. Defaults to None.
        kernel: one of {"full", "df_kernel", "cf_kernel"}. Defaults to "full".
        X_ctrl_ind: Indices of control points at which kernels will be centered. Defaults to None.

    Raises:
        ValueError: If the kernel value specified is not one of "full", "df_kernel", or "cf_kernel"

    Returns:
        np.ndarray storing the `dim` dimensions of m x m gram matrix K storing the kernel evaluated at each pair of control points
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


def dynode_vector_field_function(
    x: np.ndarray, vf_dict: VecFldDict, dim: Optional[Union[int, np.ndarray]] = None, **kwargs
) -> np.ndarray:
    # try:
    #     import dynode
    #     from dynode.vectorfield import Dynode
    # except ImportError:
    #     raise ImportError("You need to install the package `dynode`." "install dynode via `pip install dynode`")
    # vf_dict["parameters"]["load_model_from_buffer"] = True
    # dynode_inspect = inspect.getfullargspec(Dynode)
    # dynode_dict = subset_dict_with_key_list(vf_dict["parameters"], dynode_inspect.args)
    #
    # nn = Dynode(**dynode_dict)
    dynode_obj = vf_dict["dynode_object"]

    to_flatten = False
    if x.ndim == 1:
        to_flatten = True
        x = x[None, :]

    res = dynode_obj.predict_velocity(X=x)  # input_x=

    if dim is not None:
        if np.isscalar(dim):
            res = res[:, :dim]
        elif dim is not None:
            res = res[:, dim]

    if to_flatten:
        res = res.flatten()

    return res


@timeit
def con_K(
    x: np.ndarray, y: np.ndarray, beta: float = 0.1, method: str = "cdist", return_d: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """con_K constructs the kernel K, where K(i, j) = k(x, y) = exp(-beta * ||x - y||^2).

    Args:
        x: Original training data points.
        y: Control points used to build kernel basis functions.
        beta: Paramerter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2),
        return_d: If True the intermediate 3D matrix x - y will be returned for analytical Jacobian.

    Returns:
        Tuple(K: the kernel to represent the vector field function, D:
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
        K = np.squeeze(np.sum(D**2, 1))
    K = -beta * K
    K = np.exp(K)

    if return_d:
        return K, D
    else:
        return K


@timeit
def con_K_div_cur_free(
    x: np.ndarray, y: np.ndarray, sigma: int = 0.8, eta: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a convex combination of the divergence-free kernel T_df and curl-free kernel T_cf with a bandwidth sigma
    and a combination coefficient gamma.

    Args:
        x: Original training data points.
        y: Control points used to build kernel basis functions
        sigma: Bandwidth parameter.
        eta: Combination coefficient for the divergence-free or the curl-free kernels.

    Returns:
        A tuple of G (the combined kernel function), divergence-free kernel and curl-free kernel.

    See also: :func:`sparseVFC`.
    """
    m, d = x.shape
    n, d = y.shape
    sigma2 = sigma**2
    G_tmp = np.matlib.tile(x[:, :, None], [1, 1, n]) - np.transpose(np.matlib.tile(y[:, :, None], [1, 1, m]), [2, 1, 0])
    G_tmp = np.squeeze(np.sum(G_tmp**2, 1))
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


def get_vf_dict(adata: AnnData, basis: str = "", vf_key: str = "VecFld") -> VecFldDict:
    """Get vector field dictionary from the `.uns` attribute of the AnnData object.

    Args:
        adata: `AnnData` object
        basis: string indicating the embedding data to use for calculating velocities. Defaults to "".
        vf_key: _description_. Defaults to "VecFld".

    Raises:
        ValueError: if vf_key or vfkey_basis is not included in the adata object.

    Returns:
        vector field dictionary
    """
    if basis is not None:
        if len(basis) > 0:
            vf_key = "%s_%s" % (vf_key, basis)

    if vf_key not in adata.uns.keys():
        raise ValueError(
            f"Vector field function {vf_key} is not included in the adata object! "
            f"Try firstly running dyn.vf.VectorField(adata, basis='{basis}')"
        )

    vf_dict = adata.uns[vf_key]
    return vf_dict


def vecfld_from_adata(adata: AnnData, basis: str = "", vf_key: str = "VecFld") -> Tuple[VecFldDict, Callable]:
    vf_dict = get_vf_dict(adata, basis=basis, vf_key=vf_key)

    method = vf_dict["method"]
    if method.lower() == "sparsevfc":
        func = lambda x: vector_field_function(x, vf_dict)
    elif method.lower() == "dynode":
        func = lambda x: dynode_vector_field_function(x, vf_dict)
    else:
        raise ValueError(f"current only support two methods, SparseVFC and dynode")

    return vf_dict, func


def vector_transformation(V: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Transform vectors from PCA space to the original space using the formula:
                    :math:`\hat{v} = v Q^T`,
    where `Q, v, \hat{v}` are the PCA loading matrix, low dimensional vector and the
    transformed high dimensional vector.

    Args:
        V: The n x k array of vectors to be transformed, where n is the number of vectors,
            k the dimension.
        Q: PCA loading matrix with dimension d x k, where d is the dimension of the original space,
            and k the number of leading PCs.

    Returns:
        ret: The array of transformed vectors.
    """
    return V @ Q.T


def vector_field_function_transformation(vf_func: Callable, Q: np.ndarray, func_inv_x: Callable) -> Callable:
    """Transform vector field function from PCA space to the original space.
    The formula used for transformation:
                                            :math:`\hat{f} = f Q^T`,
    where `Q, f, \hat{f}` are the PCA loading matrix, low dimensional vector field function and the
    transformed high dimensional vector field function.

    Args:
        vf_func: The vector field function.
        Q: PCA loading matrix with dimension d x k, where d is the dimension of the original space,
            and k the number of leading PCs.
        func_inv_x: The function that transform x back into the PCA space.

    Returns:
        The transformed vector field function.

    """
    return lambda x: vf_func(func_inv_x(x)) @ Q.T


# ---------------------------------------------------------------------------------------------------
# jacobian
def Jacobian_rkhs_gaussian(x: np.ndarray, vf_dict: VecFldDict, vectorize: bool = False) -> np.ndarray:
    """analytical Jacobian for RKHS vector field functions with Gaussian kernel.

    Args:
    x: Coordinates where the Jacobian is evaluated.
    vf_dict: A dictionary containing RKHS vector field control points, Gaussian bandwidth,
        and RKHS coefficients.
        Essential keys: 'X_ctrl', 'beta', 'C'

    Returns:
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


def Jacobian_rkhs_gaussian_parallel(x: np.ndarray, vf_dict: VecFldDict, cores: Optional[int] = None) -> np.ndarray:
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


def Jacobian_numerical(f: Callable, input_vector_convention: str = "row") -> Union[Callable, nd.Jacobian]:
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
def elementwise_jacobian_transformation(Js: np.ndarray, qi: np.ndarray, qj: np.ndarray) -> np.ndarray:
    """Inverse transform low dimensional k x k Jacobian matrix (:math:`\partial F_i / \partial x_j`) back to the
    d-dimensional gene expression space. The formula used to inverse transform Jacobian matrix calculated from
    low dimension (PCs) is:
                                            :math:`Jac = Q J Q^T`,
    where `Q, J, Jac` are the PCA loading matrix, low dimensional Jacobian matrix and the inverse transformed high
    dimensional Jacobian matrix. This function takes only one row from Q to form qi or qj.

    Args:
        Js: k x k x n matrices of n k-by-k Jacobians.
        qi: The i-th row of the PC loading matrix Q with dimension d x k, corresponding to the effector gene i.
        qj: The j-th row of the PC loading matrix Q with dimension d x k, corresponding to the regulator gene j.

    Returns:
        The calculated Jacobian elements (:math:`\partial F_i / \partial x_j`) for each cell.
    """

    Js = np.atleast_3d(Js)
    n = Js.shape[2]
    ret = np.zeros(n)
    for i in tqdm(range(n), "calculating Jacobian for each cell"):
        ret[i] = qi @ Js[:, :, i] @ qj

    return ret


def Jacobian_kovf(
    x: np.ndarray, fjac_base: Callable, K: np.ndarray, Q: np.ndarray, exact: bool = False, mu: Optional[float] = None
) -> np.ndarray:
    """analytical Jacobian for RKHS vector field functions with Gaussian kernel.

    Args:
        x: Coordinates where the Jacobian is evaluated.
        vf_dict:
            A dictionary containing RKHS vector field control points, Gaussian bandwidth,
            and RKHS coefficients.
            Essential keys: 'X_ctrl', 'beta', 'C'

    Returns:
        Jacobian matrices stored as d-by-d-by-n numpy arrays evaluated at x.
            d is the number of dimensions and n the number of coordinates in x.
    """
    if K.ndim == 1:
        K = np.diag(K)

    if exact:
        if mu is None:
            raise Exception("For exact calculations of the Jacobian, the mean of the PCA transformation is needed.")

        s = np.sign(x @ Q.T + mu)
        if x.ndim > 1:
            G = np.zeros((Q.shape[1], Q.shape[1], x.shape[0]))
            KQ = K @ Q
            # KQ = (np.diag(K) * Q.T).T
            for i in range(x.shape[0]):
                G[:, :, i] = s[i] * Q.T @ KQ
        else:
            G = s * Q.T @ K @ Q
    else:
        G = Q.T @ K @ Q
        if x.ndim > 1:
            G = np.repeat(G[:, :, None], x.shape[0], axis=2)

    return fjac_base(x) - G


@timeit
def subset_jacobian_transformation(Js: np.ndarray, Qi: np.ndarray, Qj: np.ndarray, cores: int = 1) -> np.ndarray:
    """Transform Jacobian matrix (:math:`\partial F_i / \partial x_j`) from PCA space to the original space.
    The formula used for transformation:
                                            :math:`\hat{J} = Q J Q^T`,
    where `Q, J, \hat{J}` are the PCA loading matrix, low dimensional Jacobian matrix and the inverse transformed high
    dimensional Jacobian matrix. This function takes multiple rows from Q to form Qi or Qj.

    Args:
        Js: Original (k x k) dimension Jacobian matrix
        Qi: PCA loading matrix with dimension n' x n_PCs of the effector genes, from which local dimension Jacobian matrix (k x k)
            will be inverse transformed back to high dimension.
        Qj: PCs loading matrix with dimension n' x n_PCs of the regulator genes, from which local dimension Jacobian matrix (k x k)
            will be inverse transformed back to high dimension.
        cores: Number of cores to calculate Jacobian. If cores is set to be > 1, multiprocessing will be used to
            parallel the Jacobian calculation.

    Returns:
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


def transform_jacobian(Js: np.ndarray, Qi: np.ndarray, Qj: np.ndarray, pbar=False) -> np.ndarray:
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


def average_jacobian_by_group(Js: np.ndarray, group_labels: List[str]) -> Dict[str, np.ndarray]:
    """
    Returns a dictionary of averaged jacobians with group names as the keys.
    No vectorized indexing was used due to its high memory cost.

    Args:
        Js: List of Jacobian matrices
        group_labels: list of group labels

    Returns:
        dictionary with group labels as keys and average Jacobians as values
    """
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
# Hessian


def Hessian_rkhs_gaussian(x: np.ndarray, vf_dict: VecFldDict) -> np.ndarray:
    """analytical Hessian for RKHS vector field functions with Gaussian kernel.

    Args:
        x: Coordinates where the Hessian is evaluated. Note that x has to be 1D.
        vf_dict: A dictionary containing RKHS vector field control points, Gaussian bandwidth,
            and RKHS coefficients.
            Essential keys: 'X_ctrl', 'beta', 'C'

    Returns:
        H: Hessian matrix stored as d-by-d-by-d numpy arrays evaluated at x.
            d is the number of dimensions.
    """
    x = np.atleast_2d(x)

    C = vf_dict["C"]
    beta = vf_dict["beta"]
    K, D = con_K(x, vf_dict["X_ctrl"], beta, return_d=True)

    K = K * C.T

    D = D.T
    D = np.eye(x.shape[1]) - 2 * beta * D @ np.transpose(D, axes=(0, 2, 1))

    H = -2 * beta * np.einsum("ij, jlm -> ilm", K, D)

    return H


def hessian_transformation(H: np.ndarray, qi: np.ndarray, Qj: np.ndarray, Qk: np.ndarray) -> np.ndarray:
    """Inverse transform low dimensional k x k x k Hessian matrix (:math:`\partial^2 F_i / \partial x_j \partial x_k`)
    back to the d-dimensional gene expression space. The formula used to inverse transform Hessian matrix calculated
    from low dimension (PCs) is:
                                            :math:`h = \sum_i\sum_j\sum_k q_i q_j q_k H_ijk`,
    where `q, H, h` are the PCA loading matrix, low dimensional Hessian matrix and the inverse transformed element from
    the high dimensional Hessian matrix.

    Args:
        H: k x k x k matrix of the Hessian.
        qi: The i-th row of the PC loading matrix Q with dimension d x k, corresponding to the effector i.
        Qj: The submatrix of the PC loading matrix Q with dimension d x k, corresponding to regulators j.
        Qk: The submatrix of the PC loading matrix Q with dimension d x k, corresponding to co-regulators k.

    Returns:
        h: The calculated Hessian matrix for the effector i w.r.t regulators j and co-regulators k.
    """

    h = np.einsum("ijk, di -> djk", H, qi)
    Qj, Qk = np.atleast_2d(Qj), np.atleast_2d(Qk)
    h = Qj @ h @ Qk.T

    return h


def elementwise_hessian_transformation(H: np.ndarray, qi: np.ndarray, qj: np.ndarray, qk: np.ndarray) -> np.ndarray:
    """Inverse transform low dimensional k x k x k Hessian matrix (:math:`\partial^2 F_i / \partial x_j \partial x_k`) back to the
    d-dimensional gene expression space. The formula used to inverse transform Hessian matrix calculated from
    low dimension (PCs) is:
                                            :math:`Jac = Q J Q^T`,
    where `Q, J, Jac` are the PCA loading matrix, low dimensional Jacobian matrix and the inverse transformed high
    dimensional Jacobian matrix. This function takes only one row from Q to form qi or qj.

    Args:
        H: k x k x k matrix of the Hessian.
        qi: The i-th row of the PC loading matrix Q with dimension d x k, corresponding to the effector i.
        qj: The j-th row of the PC loading matrix Q with dimension d x k, corresponding to the regulator j.
        qk: The k-th row of the PC loading matrix Q with dimension d x k, corresponding to the co-regulator k.

    Returns:
        h: The calculated Hessian elements for each cell.
    """

    h = np.einsum("ijk, i -> jk", H, qi)
    h = qj @ h @ qk

    return h


# ---------------------------------------------------------------------------------------------------
def Laplacian(H: np.ndarray) -> np.ndarray:
    """
    Computes the Laplacian of the Hessian matrix by summing the diagonal elements of the Hessian matrix (summing the unmixed second partial derivatives)
                                            :math: `\Delta f = \sum_{i=1}^{n} \frac{\partial^2 f}{\partial x_i^2}`
    Args:
        H: Hessian matrix
    """
    # when H has four dimensions, H is calculated across all cells
    if H.ndim == 4:
        L = np.zeros([H.shape[2], H.shape[3]])
        for sample_indx in range(H.shape[3]):
            for out_indx in range(L.shape[0]):
                L[out_indx, sample_indx] = np.diag(H[:, :, out_indx, sample_indx]).sum()
    else:
        # when H has three dimensions, H is calculated only on one single cell
        L = np.zeros([H.shape[2], 1])
        for out_indx in range(L.shape[0]):
            L[out_indx, 0] = np.diag(H[:, :, out_indx]).sum()

    return L


# ---------------------------------------------------------------------------------------------------
# dynamical properties
def _divergence(f: Callable, x: np.ndarray) -> float:
    """Divergence of the reconstructed vector field function f evaluated at x"""
    jac = nd.Jacobian(f)(x)
    return np.trace(jac)


@timeit
def compute_divergence(
    f_jac: Callable, X: np.ndarray, Js: Optional[np.ndarray] = None, vectorize_size: int = 1000
) -> np.ndarray:
    """Calculate divergence for many samples by taking the trace of a Jacobian matrix.

    vectorize_size is used to control the number of samples computed in each vectorized batch.
        If vectorize_size = 1, there's no vectorization whatsoever.
        If vectorize_size = None, all samples are vectorized.

    Args:
        f_jac: function for calculating Jacobian from cell states
        X: cell states
        Js: Jacobian matrices for each sample, if X is not provided
        vectorize_size: number of Jacobian matrices to process at once in the vectorization

    Returns:
        divergence np.ndarray across Jacobians for many samples
    """
    n = len(X)
    if vectorize_size is None:
        vectorize_size = n

    div = np.zeros(n)
    for i in tqdm(range(0, n, vectorize_size), desc="Calculating divergence"):
        J = f_jac(X[i : i + vectorize_size]) if Js is None else Js[:, :, i : i + vectorize_size]
        div[i : i + vectorize_size] = np.trace(J)
    return div


def acceleration_(v: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Calculate acceleration by dotting the Jacobian and the velocity vector.

    Args:
        v: velocity vector
        J: Jacobian matrix

    Returns:
        Acceleration vector, with one element for the acceleration of each component
    """
    if v.ndim == 1:
        v = v[:, None]
    return J.dot(v)


def curvature_method1(a: np.array, v: np.array) -> float:
    """https://link.springer.com/article/10.1007/s12650-018-0474-6"""
    if v.ndim == 1:
        v = v[:, None]
    kappa = np.linalg.norm(np.outer(v, a)) / np.linalg.norm(v) ** 3

    return kappa


def curvature_method2(a: np.array, v: np.array) -> float:
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
def compute_acceleration(vf, f_jac, X, Js=None, return_all=False):
    """Calculate acceleration for many samples via

    .. math::
    a = J \cdot v.

    """
    n = len(X)
    acce = np.zeros(n)
    acce_mat = np.zeros((n, X.shape[1]))

    v_ = vf(X)
    J_ = f_jac(X) if Js is None else Js
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
def compute_curvature(vf, f_jac, X, Js=None, formula=2):
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
    v, _, _, a = compute_acceleration(vf, f_jac, X, Js=Js, return_all=True)
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
    v, J, a_, a = compute_acceleration(vf, f_jac, X, return_all=True)

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


def curl3d(f, x, method="analytical", VecFld=None, jac=None):
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
        curl = np.zeros((n, 3, 3))
        f = curl3d

    for i in tqdm(range(n), desc=f"Calculating {X.shape[1]}-D curl"):
        J = f_jac(X[i])
        curl[i] = f(None, None, method="analytical", VecFld=None, jac=J)

    return curl


# ---------------------------------------------------------------------------------------------------
# ranking related utilies
def get_metric_gene_in_rank(mat: np.asmatrix, genes: list, neg: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate ranking of genes based on mean value of matrix.

    Args:
        mat: A matrix with shape (n, m) where n is the number of samples and m is the number of genes.
        genes: A list of m gene names.
        neg: A boolean flag indicating whether the ranking should be in ascending (True) or descending (False) order. Defaults to False (descending order).

    Returns:
        metric_in_rank: A list of m gene scores, sorted in ascending or descending order depending on the value of neg.
        genes_in_rank: A list of m gene names, sorted in the same order as metric_in_rank.
    """
    metric_in_rank = mat.mean(0).A1 if issparse(mat) else mat.mean(0)
    rank = metric_in_rank.argsort() if neg else metric_in_rank.argsort()[::-1]
    metric_in_rank, genes_in_rank = metric_in_rank[rank], genes[rank]

    return metric_in_rank, genes_in_rank


def get_metric_gene_in_rank_by_group(
    mat: np.asmatrix, genes: list, groups: np.array, selected_group, neg: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate ranking of genes based on mean value of matrix, grouped by selected group.

    Args:
        mat: A matrix with shape (n, m) where n is the number of samples and m is the number of genes.
        genes: A list of m gene names.
        groups: A numpy array with shape (n,) indicating the group membership for each sample.
        selected_group: The group for which the ranking should be calculated.
        neg: A boolean flag indicating whether the ranking should be in ascending (True) or descending (False) order. Defaults to False (descending order).

    Returns:
        gene_wise_metrics: A list of m gene scores, sorted in ascending or descending order depending on the value of neg.
        group_wise_metrics: A list of m gene scores, calculated as the mean value of mat for samples belonging to the selected group.
        genes_in_rank: A list of m gene names, sorted in the same order as gene_wise_metrics.
    """
    mask = groups == selected_group
    if type(mask) == pd.Series:
        mask = mask.values

    gene_wise_metrics, group_wise_metrics = (
        mat[mask, :].mean(0).A1 if issparse(mat) else mat[mask, :].mean(0),
        mat[mask, :].mean(0).A1 if issparse(mat) else mat[mask, :].mean(0),
    )
    rank = gene_wise_metrics.argsort() if neg else gene_wise_metrics.argsort()[::-1]
    gene_wise_metrics, genes_in_rank = gene_wise_metrics[rank], genes[rank]

    return gene_wise_metrics, group_wise_metrics, genes_in_rank


def get_sorted_metric_genes_df(df: pd.DataFrame, genes: list, neg: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sort metric and gene lists based on values in dataframe.

    Args:
        df: A dataframe with shape (m, n) where m is the number of genes and n is the number of samples.
        genes: A list of m gene names.
        neg: A boolean flag indicating whether the ranking should be in ascending (True) or descending (False) order. Defaults to False (descending order).

    Returns:
        sorted_metric: A dataframe with shape (m, n) containing the sorted metric values.
        sorted_genes: A dataframe with shape (m, n) containing the sorted gene names.
    """
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


def rank_vector_calculus_metrics(mat: np.asmatrix, genes: list, group, groups: list, uniq_group: list) -> Tuple:
    """Calculate ranking of genes based on vector calculus metric.

    Args:
        mat: A matrix with shape (n, m) where n is the number of samples and m is the number of genes.
        genes: A list of m gene names.
        group: The group for which the ranking should be calculated. If group is None, the ranking is calculated for all samples.
        groups: A list of n group labels, indicating the group membership for each sample.
        uniq_group: A list of unique group labels.

    Returns:
        If group is None:
            metric_in_rank: A list of m gene scores, sorted by the mean of the the absolute values of the elements in each column of mat.
            genes_in_rank: A list of m gene names, sorted in the same order as metric_in_rank.
            pos_metric_in_rank: A list of m gene scores, sorted by the mean of the positive elements in each column of mat.
            pos_genes_in_rank: A list of m gene names, sorted in the same order as pos_metric_in_rank.
            neg_metric_in_rank: A list of m gene scores, calculated as the mean of the negative elements in each column of mat.
            neg_genes_in_rank: A list of m gene names, sorted in the same order as neg_metric_in_rank.
        If group is not None:
            gene_wise_metrics: A dictionary with keys equal to the unique group labels in uniq_group, and values equal to lists of m gene scores, calculated as the mean value of mat for samples belonging to each group.
            group_wise_metrics: A dictionary with keys equal to the unique group labels in uniq_group, and values equal to lists of m gene scores, calculated as the mean value of mat for samples belonging to each group.
            gene_wise_genes: A dictionary with keys equal to the unique group labels in uniq_group, and values equal to lists of m gene names, sorted in the same order as the corresponding values in gene_wise_metrics.
            gene_wise_pos_metrics: A dictionary with keys equal to the unique group labels in uniq_group, and values equal to lists of m gene scores, calculated as the mean value of the positive elements in mat for samples belonging to each group.
            group_wise_pos_metrics: A dictionary with keys equal to the unique group labels in uniq_group, and values equal to lists of m gene scores, calculated as the mean value of the positive elements in mat for samples belonging to each group.
            gene_wise_pos_genes: A dictionary with keys equal to the unique group labels in uniq_group, and values equal to lists of m gene names, sorted in the same order as the corresponding values in gene_wise_pos_metrics.
            gene_wise_neg_metrics: A dictionary with keys equal to the unique group labels in uniq_group, and values equal to lists of m gene scores, calculated as the mean value of the negative elements in mat for samples belonging to each group.
            group_wise_neg_metrics: A dictionary with keys equal to the unique labels in uniq_group, and values equal to lists of m gene scores, calculated as the mean value of the negative elements in mat for samples belonging to each group.
    """
    main_info("split mat to a positive matrix and a negative matrix.")
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
        main_info("ranking vector calculus in group: %s" % (group))
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
        for i, grp in tqdm(enumerate(uniq_group), desc="ranking genes across groups"):
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
# @njit(cache=True, nogil=True) # causing numba error_write issue
def angle(vector1, vector2):
    """Returns the angle in radians between given vectors"""
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


# @njit(cache=True, nogil=True) # causing numba error_write issue
def unit_vector(vector):
    """Returns the unit vector of the vector."""
    vec_norm = np.linalg.norm(vector)
    if vec_norm == 0:
        return vec_norm, vector
    else:
        return vec_norm, vector / vec_norm


def normalize_vectors(vectors, axis=1, **kwargs):
    """Returns the unit vectors of the vectors."""
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


def find_fixed_points(
    x0_list: Union[list, np.ndarray],
    func_vf: Callable,
    domain: Optional[np.ndarray] = None,
    tol_redundant: float = 1e-4,
    return_all: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given sampling points, a function, and a domain, finds points for which func_vf(x) = 0.

    Args:
        x0_list: Array-like structure with sampling points
        func_vf: Function for which to find fixed points
        domain: Finds fixed points within the given domain of shape (n_dim, 2)
        tol_redundant: Margin outside of which points are considered distinct
        return_all: If set to true, always return a tuple of three arrays as output

    Returns:
        A tuple with the solutions, Jacobian matrix, and function values at the solutions.

    """

    def vf_aux(x):
        """auxillary function unifying dimensionality"""
        v = func_vf(x)
        if x.ndim == 1:
            v = v.flatten()
        return v

    X = []
    J = []
    fval = []
    for x0 in x0_list:
        x, info_dict, _, _ = fsolve(vf_aux, x0, full_output=True)

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
    if len(regulators_) + len(effectors_) > 2:
        Der = Der[valid_target_idx, :, :][:, valid_source_idx, :]

    # reshape Der: special case for getting Jacobian of two identical genes
    if len(np.array(Der).shape) == 1:
        Der = np.array(Der).reshape([1, 1, -1])

    regulators, effectors = (
        np.array(regulators_)[valid_source_idx],
        np.array(effectors_)[valid_target_idx],
    )

    return Der, regulators, effectors


# ---------------------------------------------------------------------------------------------------
# vector field ranking related utilies
def parse_int_df(
    df: pd.DataFrame,
    self_int: bool = False,
    genes: bool = None,
) -> pd.DataFrame:
    """parse the dataframe produced from vector field ranking for gene interactions or switch gene pairs

    Args:
        df: The dataframe that returned from performing the `int` or `switch` mode ranking via dyn.vf.rank_jacobian_genes.
        self_int: Whether to keep self-interactions pairs.
        genes: List of genes that are used to filter for gene interactions.

    Returns:
        res: The parsed interaction dataframe.
    """

    df_shape, columns = df.shape, df.columns
    # first we have second column name ends with "_values", it means the data frame include ranking values.
    if columns[1].endswith("_values"):
        col_step = 2
    else:
        col_step = 1

    res = {}
    if genes is not None:
        genes_set = set(genes)
    for col in columns[::col_step]:
        cur_col = df[col]
        gene_pairs = cur_col.str.split(" - ", expand=True)

        if not self_int:
            good_int = gene_pairs[0] != gene_pairs[1]
        else:
            good_int = np.ones(df_shape[0], dtype=bool)

        if genes is not None:
            good_int &= np.logical_and([i in genes_set for i in gene_pairs[0]], [i in genes_set for i in gene_pairs[1]])

        if col_step == 1:
            res[col] = cur_col.loc[good_int].values
        else:
            res[col] = cur_col.loc[good_int].values
            res[col + "_values"] = df[col + "_values"].loc[good_int].values

    return pd.DataFrame(res)


# ---------------------------------------------------------------------------------------------------
# jacobian retrival related utilies
def get_jacobian(
    adata: AnnData,
    regulators: np.ndarray,
    effectors: np.ndarray,
    jkey: str = "jacobian",
    j_basis: str = "pca",
) -> pd.DataFrame:
    """Return dataframe with Jacobian values (where regulators are in the denominator and effectors in the numerator)

    Args:
        adata: AnnData object
        regulators: string labels capturing regulator genes of interest
        effectors: string labels capturing effector genes of interest
        jkey: Defaults to "jacobian".
        j_basis: Defaults to "pca".

    Returns:
        dataframe with Jacobian values (where regulators are in the denominator and effectors in the numerator)
    """

    regulators, effectors = (
        list(np.unique(regulators)) if regulators is not None else None,
        list(np.unique(effectors)) if effectors is not None else None,
    )

    Jacobian_ = jkey if j_basis is None else jkey + "_" + j_basis
    Der, cell_indx, jacobian_gene, regulators_, effectors_ = (
        adata.uns[Jacobian_].get(jkey.split("_")[-1]),
        adata.uns[Jacobian_].get("cell_idx"),
        adata.uns[Jacobian_].get(jkey.split("_")[-1] + "_gene"),
        adata.uns[Jacobian_].get("regulators"),
        adata.uns[Jacobian_].get("effectors"),
    )

    adata_ = adata[cell_indx, :]

    if regulators is None and effectors is not None:
        regulators = effectors
    elif effectors is None and regulators is not None:
        effectors = regulators
    # test the simulation data here
    if regulators_ is None or effectors_ is None:
        if Der.shape[0] != adata_.n_vars:
            source_genes = [j_basis + "_" + str(i) for i in range(Der.shape[0])]
            target_genes = [j_basis + "_" + str(i) for i in range(Der.shape[1])]
        else:
            source_genes, target_genes = adata_.var_names, adata_.var_names
    else:
        Der, source_genes, target_genes = intersect_sources_targets(
            regulators,
            regulators_,
            effectors,
            effectors_,
            Der if jacobian_gene is None else jacobian_gene,
        )

    df = pd.DataFrame(index=adata.obs_names[cell_indx])
    for i, source in enumerate(source_genes):
        for j, target in enumerate(target_genes):
            J = Der[j, i, :]  # dim 0: target; dim 1: source
            key = source + "->" + target + "_jacobian"
            df[key] = np.nan
            df.loc[:, key] = J

    return df


# ---------------------------------------------------------------------------------------------------
# jacobian subset related utilies
def subset_jacobian(adata: AnnData, cells: np.ndarray, basis: str = "pca"):
    """Subset adata object while also subset the jacobian

    Args:
        adata: AnnData object
        cells: vector of cell indices
        basis: string specifying jacobian layer to subset

    Returns:
        subset of adata
    """

    adata_subset = adata[cells]

    jkey = "jacobian_" + basis
    adata_subset.uns[jkey].keys()

    # assume all cells are used to calculate Jacobian for now
    adata_subset.uns[jkey]["cell_idx"] = np.arange(len(cells))

    adata_subset.uns[jkey]["jacobian_gene"] = adata_subset.uns[jkey]["jacobian_gene"][:, :, cells]
    adata_subset.uns[jkey]["jacobian"] = adata_subset.uns[jkey]["jacobian"][:, :, cells]

    return adata_subset
