from tqdm import tqdm
import numpy as np
import numdifftools as nd
from .utils import timeit

def grad(f, x):
    """Gradient of scalar-valued function f evaluated at x"""
    return nd.Gradient(f)(x)


def laplacian(f, x):
    """Laplacian of scalar field f evaluated at x"""
    hes = nd.Hessdiag(f)(x)
    return sum(hes)


def get_fjac(f, input_vector_convention='row'):
    '''
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
    '''
    fjac = nd.Jacobian(lambda x: f(x.T).T)
    if input_vector_convention == 'row' or input_vector_convention == 0:
        def f_aux(x):
            x = x.T
            return fjac(x)

        return f_aux
    else:
        return fjac


@timeit
def elementwise_jacobian_transformation(fjac, X, qi, qj):
    """Inverse transform low dimension Jacobian matrix (:math:`\partial F_i / \partial x_j`) back to original space.
    The formula used to inverse transform Jacobian matrix calculated from low dimension (PCs) is:
                                            :math:`Jac = Q J Q^T`,
    where `Q, J, Jac` are the PCA loading matrix, low dimensional Jacobian matrix and the inverse transformed high
    dimensional Jacobian matrix. This function only take one element from Q to form qi or qj.

    Parameters
    ----------
        fjac: `function`:
            The function for calculating numerical Jacobian matrix.
        X: `np.ndarray`:
            The samples coordinates with dimension n_obs x n_PCs, from which Jacobian will be calculated.
        Qi: `np.ndarray`:
            One sampled gene's PCs loading matrix with dimension n' x n_PCs, from which local dimension Jacobian matrix
            (k x k) will be inverse transformed back to high dimension.
        Qj: `np.ndarray`
            Another gene's (can be the same as those in Qi or different) PCs loading matrix with dimension  n' x n_PCs,
            from which local dimension Jacobian matrix (k x k) will be inverse transformed back to high dimension.

    Returns
    -------
        ret `np.ndarray`
            The calculated vector of Jacobian matrix (:math:`\partial F_i / \partial x_j`) for each cell.
    """

    Js = fjac(X)
    ret = np.zeros(len(X))
    for i in range(len(X)):
        J = Js[:, :, i]
        ret[i] = qi @ J @ qj
    return ret


@timeit
def subset_jacobian_transformation(fjac, X, Qi, Qj):
    """Inverse transform low dimension Jacobian matrix (:math:`\partial F_i / \partial x_j`) back to original space.
    The formula used to inverse transform Jacobian matrix calculated from low dimension (PCs) is:
                                            :math:`Jac = Q J Q^T`,
    where `Q, J, Jac` are the PCA loading matrix, low dimensional Jacobian matrix and the inverse transformed high
    dimensional Jacobian matrix. This function only take multiple elements from Q to form Qi or Qj.

    Parameters
    ----------
        fjac: `function`:
            The function for calculating numerical Jacobian matrix.
        X: `np.ndarray`:
            The samples coordinates with dimension n_obs x n_PCs, from which Jacobian will be calculated.
        Qi: `np.ndarray`:
            Sampled genes' PCs loading matrix with dimension n' x n_PCs, from which local dimension Jacobian matrix (k x k)
            will be inverse transformed back to high dimension.
        Qj: `np.ndarray`
            Sampled genes' (sample genes can be the same as those in Qi or different) PCs loading matrix with dimension
            n' x n_PCs, from which local dimension Jacobian matrix (k x k) will be inverse transformed back to high dimension.

    Returns
    -------
        ret `np.ndarray`
            The calculated Jacobian matrix (n_gene x n_gene x n_obs) for each cell.

    """
    X = np.atleast_2d(X)
    Qi = np.atleast_2d(Qi)
    Qj = np.atleast_2d(Qj)
    d1, d2, n = Qi.shape[0], Qj.shape[0], X.shape[0]

    Js = fjac(X)
    ret = np.zeros((d1, d2, n))
    for i in tqdm(range(n), desc='Transforming subset Jacobian'):
        J = Js[:, :, i]
        ret[:, :, i] = Qi @ J @ Qj.T
    return ret


def divergence(f, x):
    """Divergence of the reconstructed vector field function f evaluated at x"""
    jac = nd.Jacobian(f)(x)
    return np.trace(jac)


@timeit
def compute_divergence(f_jac, X, vectorize=True):
    """calculate divergence for many samples by taking the trace of a Jacobian matrix"""
    if vectorize:
        J = f_jac(X)
        div = np.trace(J)
    else:
        div = np.zeros(len(X))
        for i in tqdm(range(len(X)), desc="Calculating divergence"):
            J = f_jac(X[i])
            div[i] = np.trace(J)

    return div


def curl(f, x):
    """Curl of the reconstructed vector field f evaluated at x in 3D"""
    jac = nd.Jacobian(f)(x)
    return np.array([jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]])


def curl2d(f, x):
    """Curl of the reconstructed vector field f evaluated at x in 2D"""
    jac = nd.Jacobian(f)(x)
    curl = jac[0, 0] - jac[1, 1] + jac[0, 1] - jac[1, 1]

    return curl


def Curl(adata,
         basis='umap',
         vecfld_dict=None,
         ):
    """Calculate Curl for each cell with the reconstructed vector field function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: `str` or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        vecfld_dict: `dict`
            The true ODE function, useful when the data is generated through simulation.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `curl` key in the .obs.
    """

    if vecfld_dict is None:
        vf_key = 'VecFld' if basis is None else 'VecFld_' + basis
        if vf_key not in adata.uns.keys():
            raise ValueError(f"Your adata doesn't have the key for Vector Field with {basis} basis. "
                             f"Try firstly running dyn.tl.VectorField(adata, basis={basis}).")

        vecfld_dict = adata.uns[vf_key]

    X_data = adata.obsm["X_" + basis]

    curl = np.zeros((adata.n_obs, 1))
    func = vecfld_dict['func']

    for i, x in tqdm(enumerate(X_data), f"Calculating curl with the reconstructed vector field on the {basis} basis. "):
        curl[i] = curl2d(func, x.flatten())

    adata.obs['curl'] = curl


def Divergence(adata,
         basis='umap',
         vecfld_dict=None,
         ):
    """Calculate divergence for each cell with the reconstructed vector field function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: `str` or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        vecfld_dict: `dict`
            The true ODE function, useful when the data is generated through simulation.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `divergence` key in the .obs.
    """

    if vecfld_dict is None:
        vf_key = 'VecFld' if basis is None else 'VecFld_' + basis
        if vf_key not in adata.uns.keys():
            raise ValueError(f"Your adata doesn't have the key for Vector Field with {basis} basis."
                             f"Try firstly running dyn.tl.VectorField(adata, basis={basis}).")

        vecfld_dict = adata.uns[vf_key]

    X_data = adata.obsm["X_" + basis]

    func = vecfld_dict['func']

    div = compute_divergence(get_fjac(func), X_data, vectorize=True)

    adata.obs['divergence'] = div
