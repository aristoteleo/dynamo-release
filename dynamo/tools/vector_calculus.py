from tqdm import tqdm
import numpy as np
import numdifftools as nd
from .scVectorField import vector_field_function


def grad(f, x):
    """Gradient of scalar-valued function f evaluated at x"""
    return nd.Gradient(f)(x)


def laplacian(f, x):
    """Laplacian of scalar field f evaluated at x"""
    hes = nd.Hessdiag(f)(x)
    return sum(hes)


def divergence(f, x):
    """Divergence of the reconstructed vector field function f evaluated at x"""
    jac = nd.Jacobian(f)(x)
    return np.trace(jac)


def curl(f, x):
    """Curl of the reconstructed vector field f evaluated at x"""
    jac = nd.Jacobian(f)(x)
    return np.array([jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]])


def curl2d(f, x):
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
    func = lambda x: vector_field_function(x, vecfld_dict['VecFld'])

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

    div = np.zeros((adata.n_obs, 1))
    func = lambda x: vector_field_function(x, vecfld_dict['VecFld'])

    for i, x in tqdm(enumerate(X_data), f"Calculating divergence with the reconstructed vector field on the {basis} basis. "):
        div[i] = divergence(func, x.flatten())

    adata.obs['divergence'] = div
