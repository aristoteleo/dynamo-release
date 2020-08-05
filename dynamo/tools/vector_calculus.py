from tqdm import tqdm
import multiprocessing as mp
import itertools, functools
import numpy as np
from .utils import timeit, get_pd_row_column_idx
from .utils_vecCalc import (
    vector_field_function, 
    vecfld_from_adata, 
    curl2d, 
    elementwise_jacobian_transformation, 
    subset_jacobian_transformation
    )
from .scVectorField import vectorfield
from .sampling import sample


def speed(adata,
          basis='umap',
          VecFld=None,
          method='analytical',
          ):
    """Calculate the speed for each cell with the reconstructed vector field function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: `str` or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        VecFld: `dict`
            The true ODE function, useful when the data is generated through simulation.
        method: `str` (default: `analytical`)
            The method that will be used for calculating speed, either `analytical` or `numeric`. `analytical`
            method will use the analytical form of the reconstructed vector field for calculating Jacobian. Otherwise,
            raw velocity vectors are used.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `speed` key in the .obs.
    """

    if VecFld is None:
        VecFld, func = vecfld_from_adata(adata, basis)
    else:
        func = lambda x: vector_field_function(x, VecFld)

    X_data = adata.obsm["X_" + basis]

    vec_mat = func(X_data) if method == 'analytical' else adata.obsm["velocity_" + basis]
    speed = np.array([np.linalg.norm(i) for i in vec_mat])

    speed_key = "speed" if basis is None else "speed_" + basis

    adata.obs[speed_key] = speed


def jacobian(adata,
             regulators=None,
             effectors=None,
             cell_idx=None,
             sampling=None,
             sample_ncells=1000,
             basis='pca',
             vector_field_class=None,
             method='analytical',
             store_in_adata=True,
             **kwargs
             ):
    """Calculate Jacobian for each cell with the reconstructed vector field function.

    If the vector field was reconstructed from the reduced PCA space, the Jacobian matrix will then be inverse
    transformed back to high dimension. Note that this should also be possible for reduced UMAP space and will be
    supported shortly. Note that we use analytical formula to calculate Jacobian matrix which computationally efficient.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        regulators: `list`
            The list of genes that will be used as regulators when calculating the cell-wise Jacobian matrix. The Jacobian
            is the matrix consisting of partial derivatives of the vector field wrt gene expressions. It can be used to 
            evaluate the change in velocities of effectors (see below) as the expressions of regulators increase. The 
            regulators are the denominators of the partial derivatives. 
        effectors: `List` or `None` (default: `None`)
            The list of genes that will be used as effectors when calculating the cell-wise Jacobian matrix. The effectors
            are the numerators of the partial derivatives.
        basis: `str` or None (default: `pca`)
            The embedding data in which the vector field was reconstructed. If `None`, use the vector field function that
            was reconstructed directly from the original unreduced gene expression space.
        vector_field_class: :class:`~scVectorField.vectorfield`
            If not None, the divergene will be computed using this class instead of the vector field stored in adata.
        method: `str` (default: `analytical`)
            The method that will be used for calculating Jacobian, either `analytical` or `numeric`. `analytical`
            method will use the analytical form of the reconstructed vector field for calculating Jacobian while
            `numeric` method will use numdifftools for calculation. `analytical` method is much more efficient.
        cores: `int` (default: 1):
            Number of cores to calculate Jacobian. If cores is set to be > 1, multiprocessing will be used to
            parallel the Jacobian calculation.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `Jacobian` key in the .uns. This is a 3-dimensional tensor with
            dimensions n_obs x n_regulators x n_effectors.
    """

    if vector_field_class is None:
        vector_field_class = vectorfield()
        vector_field_class.from_adata(adata, basis=basis)

    if basis == 'umap': cell_idx = np.arange(adata.n_obs)

    X, V = vector_field_class.get_data()
    if cell_idx is None:
        if sampling is None or sampling == 'all':
            cell_idx = np.arange(adata.n_obs)
        else:
            cell_idx = sample(np.arange(adata.n_obs), sample_ncells, sampling, X, V)

    Jac_func = vector_field_class.get_Jacobian(method=method)
    Js = Jac_func(X[cell_idx])
    if regulators is not None and effectors is not None:
        if type(regulators) is str: regulators = [regulators]
        if type(effectors) is str: effectors = [effectors]
        var_df = adata[:, adata.var.use_for_dynamics].var
        regulators = var_df.index.intersection(regulators)
        effectors = var_df.index.intersection(effectors)

        reg_idx, eff_idx = get_pd_row_column_idx(var_df, regulators, "row"), \
                                get_pd_row_column_idx(var_df, effectors, "row")
        if len(regulators) == 0 or len(effectors) == 0:
            raise ValueError(f"Either the regulator or the effector gene list provided is not in the velocity gene list!")

        PCs_ = "PCs" if basis == 'pca' else "PCs_" + basis
        Q = adata.uns[PCs_][:, :X.shape[1]]
        if len(regulators) == 1 and len(effectors) == 1:
            Jacobian = elementwise_jacobian_transformation(Js, 
                    Q[eff_idx, :].flatten(), Q[reg_idx, :].flatten(), **kwargs)
        else:
            Jacobian = subset_jacobian_transformation(Js, Q[eff_idx, :], Q[reg_idx, :], **kwargs)
    else:
        Jacobian = None

    ret_dict = {"Jacobian": Js, "cell_idx": cell_idx}
    if Jacobian is not None: ret_dict['Jacobian_gene'] = Jacobian
    if regulators is not None: ret_dict['regulators'] = regulators
    if effectors is not None: ret_dict['effectors'] = effectors

    if store_in_adata:
        jkey = "jacobian" if basis is None else "jacobian_" + basis
        adata.uns[jkey] = ret_dict
    return ret_dict


def curl(adata,
         basis='umap',
         vector_field_class=None,
         **kwargs
         ):
    """Calculate Curl for each cell with the reconstructed vector field function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: `str` or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        vector_field_class: :class:`~scVectorField.vectorfield`
            If not None, the divergene will be computed using this class instead of the vector field stored in adata.
        method: `str` (default: `analytical`)
            The method that will be used for calculating divergence, either `analytical` or `numeric`. `analytical`
            method will use the analytical form of the reconstructed vector field for calculating curl while
            `numeric` method will use numdifftools for calculation. `analytical` method is much more efficient.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `curl` key in the .obs.
    """

    if vector_field_class is None:
        vector_field_class = vectorfield()
        vector_field_class.from_adata(adata, basis=basis)
    '''
    X_data = adata.obsm["X_" + basis][:, :2]

    curl = np.zeros((adata.n_obs, 1))
    
    Jacobian_ = "jacobian" if basis is None else "jacobian_" + basis

    if Jacobian_ in adata.uns_keys():
        Js = adata.uns[Jacobian_]['Jacobian_raw']
        for i in tqdm(range(X_data.shape[0]), f"Calculating curl with the reconstructed vector field on the {basis} basis. "):
            curl[i] = curl2d(func, None, method=method, VecFld=None, jac=Js[:, :, i])
    else:
        for i, x in tqdm(enumerate(X_data), f"Calculating curl with the reconstructed vector field on the {basis} basis. "):
            curl[i] = vector_field_class.compute_curl(X=x, **kwargs)
    '''
    curl = vector_field_class.compute_curl(**kwargs)
    curl_key = "curl" if basis is None else "curl_" + basis

    adata.obs[curl_key] = curl


def divergence(adata,
               cell_idx=None,
               sampling=None,
               sample_ncells=1000,
               basis='pca',
               vector_field_class=None,
               method='analytical',
               store_in_adata=True,
               **kwargs
               ):
    """Calculate divergence for each cell with the reconstructed vector field function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: `str` or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        vector_field_class: :class:`~scVectorField.vectorfield`
            If not None, the divergene will be computed using this class instead of the vector field stored in adata.
        method: `str` (default: `analytical`)
            The method that will be used for calculating divergence, either `analytical` or `numeric`. `analytical`
            method will use the analytical form of the reconstructed vector field for calculating Jacobian while
            `numeric` method will use numdifftools for calculation. `analytical` method is much more efficient.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `divergence` key in the .obs.
    """

    if vector_field_class is None:
        vector_field_class = vectorfield()
        vector_field_class.from_adata(adata, basis=basis)

    if basis == 'umap': cell_idx = np.arange(adata.n_obs)

    X, V = vector_field_class.get_data()
    if cell_idx is None:
        if sampling is None or sampling == 'all':
            cell_idx = np.arange(adata.n_obs)
        else:
            cell_idx = sample(np.arange(adata.n_obs), sample_ncells, sampling, X, V)

    jkey = "jacobian" if basis is None else "jacobian_" + basis

    div = np.zeros(len(cell_idx))
    calculated = np.zeros(len(cell_idx), dtype=bool)
    if jkey in adata.uns_keys():
        Js = adata.uns[jkey]['Jacobian']
        cidx = adata.uns[jkey]['cell_idx']
        for i, c in tqdm(enumerate(cell_idx), desc="Calculating divergence with precomputed Jacobians"):
            if c in cidx:
                calculated[i] = True
                div[i] = np.trace(Js[:, :, i]) if Js.shape[2] == len(cell_idx) else np.trace(Js[:, :, c])

    div[~calculated] = vector_field_class.compute_divergence(X[cell_idx[~calculated]], **kwargs)

    if store_in_adata:
        div_key = "divergence" if basis is None else "divergence_" + basis
        Div = np.array(adata.obs[div_key]) if div_key in adata.obs.keys() else np.ones(adata.n_obs) * np.nan
        Div[cell_idx] = div
        adata.obs[div_key] = Div
    return div


def acceleration(adata,
         basis='umap',
         vector_field_class=None,
         **kwargs
         ):
    """Calculate acceleration for each cell with the reconstructed vector field function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: `str` or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        vector_field_class: :class:`~scVectorField.vectorfield`
            If not None, the divergene will be computed using this class instead of the vector field stored in adata.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `acceleration` key in the .obs as well as .obsm. If basis is `pca`,
            acceleration matrix will be inverse transformed back to original high dimension space.
    """

    if vector_field_class is None:
        vector_field_class = vectorfield()
        vector_field_class.from_adata(adata, basis=basis)

    acce_mat = vector_field_class.compute_acceleration(**kwargs)
    acce = np.array([np.linalg.norm(i) for i in acce_mat])

    acce_key = "acceleration" if basis is None else "acceleration_" + basis

    adata.obs[acce_key] = acce
    adata.obsm[acce_key] = acce_mat

    if basis == 'pca':
        adata.layers['acceleration'] = adata.layers['velocity_S'].copy()
        adata.layers['acceleration'][:, np.where(adata.var.use_for_dynamics)[0]] = acce_mat @ adata.uns['PCs'].T


def curvature(adata,
         basis='umap',
         vector_field_class=None,
         **kwargs
         ):
    """Calculate curvature for each cell with the reconstructed vector field function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: `str` or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        vector_field_class: :class:`~scVectorField.vectorfield`
            If not None, the divergene will be computed using this class instead of the vector field stored in adata.
        method: `str` (default: `analytical`)
            The method that will be used for calculating divergence, either `analytical` or `numeric`. `analytical`
            method will use the analytical form of the reconstructed vector field for calculating curvature while
            `numeric` method will use numdifftools for calculation. `analytical` method is much more efficient.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `curvature` key in the .obs.
    """

    if vector_field_class is None:
        vector_field_class = vectorfield()
        vector_field_class.from_adata(adata, basis=basis)

    curv = vector_field_class.compute_curvature(**kwargs)

    curv_key = "curvature" if basis is None else "curvature_" + basis

    adata.obs[curv_key] = curv


def torsion(adata,
         basis='umap',
         vector_field_class=None,
         **kwargs
         ):
    """Calculate torsion for each cell with the reconstructed vector field function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: `str` or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        vector_field_class: `dict`
            The true ODE function, useful when the data is generated through simulation.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `torsion` key in the .obs.
    """

    if vector_field_class is None:
        vector_field_class = vectorfield()
        vector_field_class.from_adata(adata, basis=basis)

    torsion_mat = vector_field_class.compute_torsion(**kwargs)
    torsion = np.array([np.linalg.norm(i) for i in torsion_mat])

    torsion_key = "torsion" if basis is None else "torsion_" + basis

    adata.obs[torsion_key] = torsion
    adata.uns[torsion_key] = torsion_mat

