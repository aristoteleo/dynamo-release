from tqdm import tqdm
from anndata._core.views import ArrayView
import itertools, functools
import scipy.sparse as sp
import numpy as np
import pandas as pd
from ..tools.utils import (
    timeit,
    get_pd_row_column_idx,
    elem_prod,
    fetch_states
)
from .utils import (
    vector_field_function,
    vecfld_from_adata,
    curl2d,
    vector_transformation,
    elementwise_jacobian_transformation,
    subset_jacobian_transformation,
    get_metric_gene_in_rank,
    get_metric_gene_in_rank_by_group,
    get_sorted_metric_genes_df,
    rank_vector_calculus_metrics,
    average_jacobian_by_group,
    intersect_sources_targets,
    )
from .scVectorField import vectorfield
from ..tools.sampling import sample
from ..tools.utils import (
    isarray,
    ismatrix,
    areinstance,
    list_top_genes,
    create_layer,
    index_gene,
    table_top_genes,
    list_top_interactions,
)


def velocities(adata,
               init_cells,
               init_states=None,
               basis=None,
               VecFld=None,
               layer="X",
               dims=None,
               ):
    """Calculate the velocities for any cell state with the reconstructed vector field function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        init_cells: list (default: None)
            Cell name or indices of the initial cell states for the historical or future cell state prediction with
            numerical integration. If the names in init_cells are not find in the adata.obs_name, it will be treated as
            cell indices and must be integers.
        init_states: :class:`~numpy.ndarray` or None (default: None)
            Initial cell states for the historical or future cell state prediction with numerical integration.
        basis: str or None (default: None)
            The embedding data to use for calculating velocities. If `basis` is either `umap` or `pca`, the reconstructed
            trajectory will be projected back to high dimensional space via the `inverse_transform` function.
        VecFld: dict
            The true ODE function, useful when the data is generated through simulation.
        layer: str or None (default: 'X')
            Which layer of the data will be used for predicting cell fate with the reconstructed vector field function.
            The layer once provided, will override the `basis` argument and then predicting cell fate in high dimensional
            space.
        dims: int, list, or None (default: None)
            The dimensions that will be selected for velocity calculation.


    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `"velocities"` related key in the `.uns`.
    """

    if VecFld is None:
        VecFld, func = vecfld_from_adata(adata, basis)
    else:
        func = lambda x: vector_field_function(x, VecFld)

    init_states, _, _, _ = fetch_states(
        adata, init_states, init_cells, basis, layer, False, None
    )

    vec_mat = func(init_states)
    vec_key = "velocities" if basis is None else "velocities_" + basis

    if np.isscalar(dims):
        vec_mat = vec_mat[:, :dims]
    elif dims is not None:
        vec_mat = vec_mat[:, dims]

    adata.uns[vec_key] = vec_mat


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
        basis: str or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        VecFld: dict
            The true ODE function, useful when the data is generated through simulation.
        method: str (default: `analytical`)
            The method that will be used for calculating speed, either `analytical` or `numeric`. `analytical`
            method will use the analytical form of the reconstructed vector field for calculating Jacobian. Otherwise,
            raw velocity vectors are used.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `'speed'` key in the `.obs`.
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
             Qkey='PCs',
             vector_field_class=None,
             method='analytical',
             store_in_adata=True,
             **kwargs
             ):
    """Calculate Jacobian for each cell with the reconstructed vector field.

    If the vector field was reconstructed from the reduced PCA space, the Jacobian matrix will then be inverse
    transformed back to high dimension. Note that this should also be possible for reduced UMAP space and will be
    supported shortly. Note that we compute the Jacobian for the RKHS kernel vector field analytically, 
    which is much more computationally efficient than the numerical method.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field in `.uns`.
        regulators: list
            The list of genes that will be used as regulators when calculating the cell-wise Jacobian matrix. The Jacobian
            is the matrix consisting of partial derivatives of the vector field wrt gene expressions. It can be used to 
            evaluate the change in velocities of effectors (see below) as the expressions of regulators increase. The 
            regulators are the denominators of the partial derivatives. 
        effectors: list or None (default: None)
            The list of genes that will be used as effectors when calculating the cell-wise Jacobian matrix. The effectors
            are the numerators of the partial derivatives.
        cell_idx: list or None (default: None)
            A list of cell index (or boolean flags) for which the jacobian is calculated. 
            If `None`, all or a subset of sampled cells are used.
        sampling: {None, 'random', 'velocity', 'trn'}, (default: None)
            See specific information on these methods in `.tl.sample`.
            If `None`, all cells are used.
        sample_ncells: int (default: 1000)
            The number of cells to be sampled. If `sampling` is None, this parameter is ignored.
        basis: str (default: 'pca')
            The embedding data in which the vector field was reconstructed. If `None`, use the vector field function that
            was reconstructed directly from the original unreduced gene expression space.
        Qkey: str (default: 'PCs')
            The key of the PCA loading matrix in `.uns`.
        vector_field_class: :class:`~scVectorField.vectorfield`
            If not `None`, the jacobian will be computed using this class instead of the vector field stored in adata.
        method: str (default: 'analytical')
            The method that will be used for calculating Jacobian, either `'analytical'` or `'numerical'`. `'analytical'`
            method uses the analytical expressions for calculating Jacobian while `'numerical'` method uses numdifftools,
            a numerical differentiation tool, for computing Jacobian. `'analytical'` method is much more efficient.
        cores: int (default: 1)
            Number of cores to calculate Jacobian. If cores is set to be > 1, multiprocessing will be used to
            parallel the Jacobian calculation.
        kwargs:
            Any additional keys that will be passed to elementwise_jacobian_transformation function.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `'jacobian'` key in the `.uns`. This is a 3-dimensional tensor with
            dimensions n_obs x n_regulators x n_effectors.
    """

    regulators, effectors = list(np.unique(regulators)) if regulators is not None else None, \
                            list(np.unique(effectors)) if effectors is not None else None
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

    if regulators is None and effectors is not None:
        regulators = effectors
    elif effectors is None and regulators is not None:
        effectors = regulators

    if regulators is not None and effectors is not None:
        if type(regulators) is str:
            if regulators in adata.var.keys():
                regulators = adata.var.index[adata.var[regulators]]
            else:
                regulators = [regulators]
        if type(effectors) is str:
            if effectors in adata.var.keys():
                effectors = adata.var.index[adata.var[effectors]]
            else:
                effectors = [effectors]
        var_df = adata[:, adata.var.use_for_dynamics].var
        regulators = var_df.index.intersection(regulators)
        effectors = var_df.index.intersection(effectors)

        reg_idx, eff_idx = get_pd_row_column_idx(var_df, regulators, "row"), \
                                get_pd_row_column_idx(var_df, effectors, "row")
        if len(regulators) == 0 or len(effectors) == 0:
            raise ValueError(f"Either the regulator or the effector gene list provided is not in the dynamics gene list!")
        
        if Qkey in adata.uns.keys():
            Q = adata.uns[Qkey]
        elif Qkey in adata.varm.keys():
            Q = adata.varm[Qkey]
        else:
            raise Exception(f'No PC matrix {Qkey} found in neither .uns nor .varm.')
        Q = Q[:, :X.shape[1]]
        if len(regulators) == 1 and len(effectors) == 1:
            Jacobian = elementwise_jacobian_transformation(Js,
                    Q[eff_idx, :].flatten(), Q[reg_idx, :].flatten(), **kwargs)
        else:
            Jacobian = subset_jacobian_transformation(Js, Q[eff_idx, :], Q[reg_idx, :], **kwargs)
    else:
        Jacobian = None

    ret_dict = {"jacobian": Js, "cell_idx": cell_idx}
    # use 'str_key' in dict.keys() to check if these items are computed, or use dict.get('str_key')
    if Jacobian is not None: ret_dict['jacobian_gene'] = Jacobian
    if regulators is not None: ret_dict['regulators'] = regulators.to_list()
    if effectors is not None: ret_dict['effectors'] = effectors.to_list()

    Js_det = [np.linalg.det(Js[:, :, i]) for i in np.arange(Js.shape[2])]
    adata.obs['jacobian_det_' + basis] = np.nan
    adata.obs['jacobian_det_' + basis][cell_idx] = Js_det
    if store_in_adata:
        jkey = "jacobian" if basis is None else "jacobian_" + basis
        adata.uns[jkey] = ret_dict
        return adata
    else:
        return ret_dict


def sensitivity(adata,
                regulators=None,
                effectors=None,
                cell_idx=None,
                sampling=None,
                sample_ncells=1000,
                basis='pca',
                Qkey='PCs',
                vector_field_class=None,
                method='analytical',
                projection_method='from_jacobian',
                store_in_adata=True,
                **kwargs
                ):
    """Calculate Sensitivity matrix for each cell with the reconstructed vector field.

    If the vector field was reconstructed from the reduced PCA space, the Sensitivity matrix will then be inverse
    transformed back to high dimension. Note that this should also be possible for reduced UMAP space and will be
    supported shortly. Note that we compute the Sensitivity for the RKHS kernel vector field analytically,
    which is much more computationally efficient than the numerical method.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field in `.uns`.
        regulators: list
            The list of genes that will be used as regulators when calculating the cell-wise Jacobian matrix. The Jacobian
            is the matrix consisting of partial derivatives of the vector field wrt gene expressions. It can be used to
            evaluate the change in velocities of effectors (see below) as the expressions of regulators increase. The
            regulators are the denominators of the partial derivatives.
        effectors: list or None (default: None)
            The list of genes that will be used as effectors when calculating the cell-wise Jacobian matrix. The effectors
            are the numerators of the partial derivatives.
        cell_idx: list or None (default: None)
            A list of cell index (or boolean flags) for which the jacobian is calculated.
            If `None`, all or a subset of sampled cells are used.
        sampling: {None, 'random', 'velocity', 'trn'}, (default: None)
            See specific information on these methods in `.tl.sample`.
            If `None`, all cells are used.
        sample_ncells: int (default: 1000)
            The number of cells to be sampled. If `sampling` is None, this parameter is ignored.
        basis: str (default: 'pca')
            The embedding data in which the vector field was reconstructed. If `None`, use the vector field function that
            was reconstructed directly from the original unreduced gene expression space.
        Qkey: str (default: 'PCs')
            The key of the PCA loading matrix in `.uns`.
        vector_field_class: :class:`~scVectorField.vectorfield`
            If not `None`, the jacobian will be computed using this class instead of the vector field stored in adata.
        method: str (default: 'analytical')
            The method that will be used for calculating Jacobian, either `'analytical'` or `'numerical'`. `'analytical'`
            method uses the analytical expressions for calculating Jacobian while `'numerical'` method uses numdifftools,
            a numerical differentiation tool, for computing Jacobian. `'analytical'` method is much more efficient.
        projection_method: str (dfault: 'from_jacobian')
            The method that will be used to project back to original gene expression space for calculating gene-wise
            sensitivity matrix:
                (1) 'from_jacobian': first calculate jacobian matrix and then calculate sensitivity matrix. This method
                    will take the combined regulator + effectors gene set for calculating a square Jacobian matrix
                    required for the sensitivyt matrix calculation.
                (2) 'direct': The sensitivity matrix on low dimension will first calculated and then projected back to
                    original gene expression space in a way that is similar to the gene-wise jacobian calculation.
        cores: int (default: 1)
            Number of cores to calculate Jacobian. If cores is set to be > 1, multiprocessing will be used to
            parallel the Jacobian calculation.
        kwargs:
            Any additional keys that will be passed to elementwise_jacobian_transformation function.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `'sensitivity'` key in the `.uns`. This is a 3-dimensional tensor with
            dimensions n_obs x n_regulators x n_effectors.
    """

    regulators, effectors = list(np.unique(regulators)) if regulators is not None else None, \
                            list(np.unique(effectors)) if effectors is not None else None
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

    S = vector_field_class.compute_sensitivity(method=method)

    if regulators is None and effectors is not None:
        regulators = effectors
    elif effectors is None and regulators is not None:
        effectors = regulators

    if regulators is not None and effectors is not None:
        if type(regulators) is str:
            if regulators in adata.var.keys():
                regulators = adata.var.index[adata.var[regulators]]
            else:
                regulators = [regulators]
        if type(effectors) is str:
            if effectors in adata.var.keys():
                effectors = adata.var.index[adata.var[effectors]]
            else:
                effectors = [effectors]
        var_df = adata[:, adata.var.use_for_dynamics].var
        regulators = var_df.index.intersection(regulators)
        effectors = var_df.index.intersection(effectors)

        if projection_method == 'direct':
                reg_idx, eff_idx = get_pd_row_column_idx(var_df, regulators, "row"), \
                                   get_pd_row_column_idx(var_df, effectors, "row")
                if len(regulators) == 0 or len(effectors) == 0:
                    raise ValueError(
                        f"Either the regulator or the effector gene list provided is not in the dynamics gene list!")

                Q = adata.uns[Qkey][:, :X.shape[1]]
                if len(regulators) == 1 and len(effectors) == 1:
                    Sensitivity = elementwise_jacobian_transformation(S,
                                                                   Q[eff_idx, :].flatten(), Q[reg_idx, :].flatten(), **kwargs)
                else:
                    Sensitivity = subset_jacobian_transformation(S, Q[eff_idx, :], Q[reg_idx, :], **kwargs)
        elif projection_method == 'from_jacobian':
            Js = jacobian(adata,
                        regulators=list(regulators) + list(effectors),
                        effectors=list(regulators) + list(effectors),
                        cell_idx=cell_idx,
                        sampling=sampling,
                        sample_ncells=sample_ncells,
                        basis=basis,
                        Qkey=Qkey,
                        vector_field_class=vector_field_class,
                        method=method,
                        store_in_adata=False,
                        **kwargs
                        )

            J, regulators, effectors = Js.get('jacobian_gene'), Js.get('regulators'), Js.get('effectors')
            Sensitivity = np.zeros_like(J)
            n_genes, n_genes_, n_cells = J.shape
            I = np.eye(n_genes)
            for i in tqdm(np.arange(n_cells), desc="Calculating sensitivity matrix with precomputed gene-wise Jacobians"):
                s = np.linalg.inv(I - J[:, :, i])  # np.transpose(J)
                Sensitivity[:, :, i] = s.dot(np.diag(1 / np.diag(s)))
        else:
            raise ValueError(f"`projection_method` can only be `from_jacoian` or `direct`!")
    else:
        Sensitivity = None

    ret_dict = {"sensitivity": S, "cell_idx": cell_idx}
    # use 'str_key' in dict.keys() to check if these items are computed, or use dict.get('str_key')
    if Sensitivity is not None: ret_dict['sensitivity_gene'] = Sensitivity
    if regulators is not None: ret_dict['regulators'] = regulators if type(regulators) == list else regulators.to_list()
    if effectors is not None: ret_dict['effectors'] = effectors if type(effectors) == list else effectors.to_list()

    S_det = [np.linalg.det(S[:, :, i]) for i in np.arange(S.shape[2])]
    adata.obs['sensitivity_det_' + basis] = np.nan
    adata.obs['sensitivity_det_' + basis][cell_idx] = S_det
    if store_in_adata:
        skey = "sensitivity" if basis is None else "sensitivity_" + basis
        adata.uns[skey] = ret_dict
        return adata
    else:
        return ret_dict


def acceleration(adata,
         basis='umap',
         vector_field_class=None,
         Qkey='PCs',
         method='analytical',
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
        Qkey: str (default: 'PCs')
            The key of the PCA loading matrix in `.uns`.
        method: str (default: 'analytical')
            The method that will be used for calculating acceleration field, either `'analytical'` or `'numerical'`.
            `'analytical'` method uses the analytical expressions for calculating acceleration field while `'numerical'`
            method uses numdifftools, a numerical differentiation tool, for computing acceleration. `'analytical'` method
            is much more efficient.
        kwargs:
            Any additional keys that will be passed to vector_field_class.compute_acceleration function.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `'acceleration'` key in the `.obs` as well as .obsm. If basis is `pca`,
            acceleration matrix will be inverse transformed back to original high dimension space.
    """

    if vector_field_class is None:
        vector_field_class = vectorfield()
        vector_field_class.from_adata(adata, basis=basis)

    acce_norm, acce = vector_field_class.compute_acceleration(method=method, **kwargs)

    acce_key = "acceleration" if basis is None else "acceleration_" + basis
    adata.obsm[acce_key] = acce
    adata.obs[acce_key] = acce_norm
    if basis == 'pca':
        if Qkey in adata.uns.keys():
            Q = adata.uns[Qkey]
        elif Qkey in adata.varm.keys():
            Q = adata.varm[Qkey]
        else:
            raise Exception(f'No PC matrix {Qkey} found in neither .uns nor .varm.')
        acce_hi = vector_transformation(acce, Q)
        create_layer(adata, acce_hi, layer_key='acceleration', genes=adata.var.use_for_pca)


def curvature(adata,
         basis='pca',
         vector_field_class=None,
         formula=2,
         Qkey='PCs',
         method="analytical",
         **kwargs
         ):
    """Calculate curvature for each cell with the reconstructed vector field function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: str or None (default: `pca`)
            The embedding data in which the vector field was reconstructed.
        vector_field_class: :class:`~scVectorField.vectorfield`
            If not None, the divergene will be computed using this class instead of the vector field stored in adata.
        formula: int (default: 2)
            Which formula of curvature will be used, there are two formulas, so formula can be either `{1, 2}`. By
            default it is 2 and returns both the curvature vectors and the norm of the curvature. The formula one only
            gives the norm of the curvature.
        Qkey: str (default: 'PCs')
            The key of the PCA loading matrix in `.uns`.
        method: str (default: 'analytical')
            The method that will be used for calculating curvature field, either `'analytical'` or `'numerical'`. `'analytical'`
            method uses the analytical expressions for calculating curvature while `'numerical'` method uses numdifftools,
            a numerical differentiation tool, for computing curvature. `'analytical'` method is much more efficient.
        kwargs:
            Any additional keys that will be passed to vector_field_class.compute_curvature function.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `curvature` key in the `.obs`.
    """

    if vector_field_class is None:
        vector_field_class = vectorfield()
        vector_field_class.from_adata(adata, basis=basis)

    if formula not in [1, 2]:
        raise ValueError(f"There are only two available formulas (formula can be either `{1, 2}`) to calculate "
                         f"curvature, but your formula argument is {formula}.")

    curv, curv_mat = vector_field_class.compute_curvature(formula=formula, method=method, **kwargs)

    curv_key = "curvature" if basis is None else "curvature_" + basis

    adata.obs[curv_key] = curv
    adata.obsm[curv_key] = curv_mat
    if basis == 'pca':
        curv_hi = vector_transformation(curv_mat, adata.uns[Qkey])
        create_layer(adata, curv_hi, layer_key='curvature', genes=adata.var.use_for_pca)


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
        basis: str or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        vector_field_class: dict
            The true ODE function, useful when the data is generated through simulation.
        kwargs:
            Any additional keys that will be passed to vector_field_class.compute_torsion function.

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


def curl(adata,
         basis='umap',
         vector_field_class=None,
         method='analytical',
         **kwargs
         ):
    """Calculate Curl for each cell with the reconstructed vector field function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: str or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        vector_field_class: :class:`~.scVectorField.vectorfield`
            If not None, the divergene will be computed using this class instead of the vector field stored in adata.
        method: str (default: `analytical`)
            The method that will be used for calculating curl, either `analytical` or `numeric`. `analytical`
            method will use the analytical form of the reconstructed vector field for calculating curl while
            `numeric` method will use numdifftools for calculation. `analytical` method is much more efficient.
        kwargs:
            Any additional keys that will be passed to vector_field_class.compute_curl function.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `'curl'` key in the `.obs`.
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
    curl = vector_field_class.compute_curl(method=method, **kwargs)
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
        basis: str or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        vector_field_class: :class:`scVectorField.vectorfield`
            If not None, the divergene will be computed using this class instead of the vector field stored in adata.
        method: str (default: `analytical`)
            The method that will be used for calculating divergence, either `analytical` or `numeric`. `analytical`
            method will use the analytical form of the reconstructed vector field for calculating divergence while
            `numeric` method will use numdifftools for calculation. `analytical` method is much more efficient.
        store_in_adata: bool (default: `True`)
            Whether to store the divergence result in adata.
        kwargs:
            Any additional keys that will be passed to vector_field_class.compute_divergence function.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `'divergence'` key in the `.obs`.
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
        Js = adata.uns[jkey]['jacobian']
        cidx = adata.uns[jkey]['cell_idx']
        for i, c in tqdm(enumerate(cell_idx), desc="Calculating divergence with precomputed Jacobians"):
            if c in cidx:
                calculated[i] = True
                div[i] = np.trace(Js[:, :, i]) if Js.shape[2] == len(cell_idx) else np.trace(Js[:, :, c])

    div[~calculated] = vector_field_class.compute_divergence(X[cell_idx[~calculated]], method=method, **kwargs)

    if store_in_adata:
        div_key = "divergence" if basis is None else "divergence_" + basis
        Div = np.array(adata.obs[div_key]) if div_key in adata.obs.keys() else np.ones(adata.n_obs) * np.nan
        Div[cell_idx] = div
        adata.obs[div_key] = Div
    else:
        return div


def rank_genes(adata,
               arr_key,
               groups=None,
               genes=None,
               abs=False,
               fcn_pool=lambda x: np.mean(x, axis=0),
               dtype=None,
               output_values=False
               ):
    """Rank gene's absolute, positive, negative vector field metrics by different cell groups.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the array to be sorted in `.var` or `.layer`.
        arr_key: str or :class:`~numpy.ndarray`
            The key of the to-be-ranked array stored in `.var` or or `.layer`.
            If the array is found in `.var`, the `groups` argument will be ignored.
            If a numpy array is passed, it is used as the array to be ranked and must 
            be either an 1d array of length `.n_var`, or a `.n_obs`-by-`.n_var` 2d array.
        groups: str or None (default: None)
            Cell groups used to group the array.
        genes: list or None (default: None)
            The gene list that speed will be ranked. If provided, they must overlap the dynamics genes.
        abs: bool (default: False)
            When pooling the values in the array (see below), whether to take the absolute values.
        fcn_pool: callable (default: numpy.mean(x, axis=0))
            The function used to pool values in the to-be-ranked array if the array is 2d.
    Returns
    -------
        ret_dict: dict
            A dictionary of gene names and values based on which the genes are sorted for each cell group.
    """

    dynamics_genes = adata.var.use_for_dynamics \
        if 'use_for_dynamics' in adata.var.keys() \
        else np.ones(adata.n_vars, dtype=bool)
    if genes is not None:
        if type(genes) is str:
            genes = adata.var[genes].to_list()
            genes = np.logical_and(genes, dynamics_genes.to_list())
        elif areinstance(genes, str):
            genes_ = adata.var_names[dynamics_genes].intersection(genes).to_list()
            genes = adata.var_names.isin(genes_)
        elif areinstance(genes, bool) or areinstance(genes, np.bool_):
            genes = np.array(genes)
            genes = np.logical_and(genes, dynamics_genes.to_list())
        else:
            raise TypeError(f"The provided genes should either be a key of adata.var, "
                                f"an array of gene names, or of booleans.")
    else:
        genes = dynamics_genes

    if not np.any(genes):
        raise ValueError(f"The list of genes provided does not contain any dynamics genes.")

    if type(arr_key) is str:
        if arr_key in adata.layers.keys():
            arr = index_gene(adata, adata.layers[arr_key], genes)
        elif arr_key in adata.var.keys():
            arr = index_gene(adata, adata.var[arr_key], genes)
        else:
            raise Exception(f'Key {arr_key} not found in neither .layers nor .var.')
    else:
        arr = index_gene(adata, arr_key, genes)

    if type(arr) == ArrayView: arr = np.array(arr)
    if sp.issparse(arr): arr = arr.A
    arr[np.isnan(arr)] = 0

    if dtype is not None:
        arr = np.array(arr, dtype=dtype)
    if abs:
        arr = np.abs(arr)

    if arr.ndim > 1:
        if groups is not None:
            if type(groups) is str and groups in adata.obs.keys():
                grps = np.array(adata.obs[groups])
            elif isarray(groups):
                grps = np.array(groups)
            else:
                raise Exception(f'The group information {groups} you provided is not in your adata object.')
            arr_dict = {}
            for g in np.unique(grps):
                arr_dict[g] = fcn_pool(arr[grps==g])
        else:
            arr_dict = {'all': fcn_pool(arr)}
    else:
        arr_dict = {'all': arr}

    ret_dict = {}
    var_names = np.array(index_gene(adata, adata.var_names, genes))
    for g, arr in arr_dict.items():
        if ismatrix(arr):
            arr = arr.A.flatten()
        glst, sarr = list_top_genes(arr, var_names, None, return_sorted_array=True)
        #ret_dict[g] = {glst[i]: sarr[i] for i in range(len(glst))}
        ret_dict[g] = glst
        if output_values:
            ret_dict[g+'_values'] = sarr
    return pd.DataFrame(data=ret_dict)


def rank_cells(adata,
               arr_key,
               groups=None,
               genes=None,
               abs=False,
               fcn_pool=lambda x: np.mean(x, axis=0),
               dtype=None,
               output_values=False
               ):
    """Rank cell's absolute, positive, negative vector field metrics by different gene groups.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the array to be sorted in `.var` or `.layer`.
        arr_key: str or :class:`~numpy.ndarray`
            The key of the to-be-ranked array stored in `.var` or or `.layer`.
            If the array is found in `.var`, the `groups` argument will be ignored.
            If a numpy array is passed, it is used as the array to be ranked and must
            be either an 1d array of length `.n_var`, or a `.n_obs`-by-`.n_var` 2d array.
        groups: str or None (default: None)
            Gene groups used to group the array.
        genes: list or None (default: None)
            The gene list that speed will be ranked. If provided, they must overlap the dynamics genes.
        abs: bool (default: False)
            When pooling the values in the array (see below), whether to take the absolute values.
        fcn_pool: callable (default: numpy.mean(x, axis=0))
            The function used to pool values in the to-be-ranked array if the array is 2d.
    Returns
    -------
        ret_dict: dict
            A dictionary of cells names and values based on which the genes are sorted for each gene group.
    """

    dynamics_genes = adata.var.use_for_dynamics \
        if 'use_for_dynamics' in adata.var.keys() \
        else np.ones(adata.n_vars, dtype=bool)
    if genes is not None:
        if type(genes) is str:
            genes = adata.var[genes].to_list()
            genes = np.logical_and(genes, dynamics_genes.to_list())
        elif areinstance(genes, str):
            genes_ = adata.var_names[dynamics_genes].intersection(genes).to_list()
            genes = adata.var_names.isin(genes_)
        elif areinstance(genes, bool) or areinstance(genes, np.bool_):
            genes = np.array(genes)
            genes = np.logical_and(genes, dynamics_genes.to_list())
        else:
            raise TypeError(f"The provided genes should either be a key of adata.var, "
                                f"an array of gene names, or of booleans.")
    else:
        genes = dynamics_genes

    if not np.any(genes):
        raise ValueError(f"The list of genes provided does not contain any dynamics genes.")

    if type(arr_key) is str:
        if arr_key in adata.layers.keys():
            arr = index_gene(adata, adata.layers[arr_key], genes)
        elif arr_key in adata.var.keys():
            arr = index_gene(adata, adata.var[arr_key], genes)
        else:
            raise Exception(f'Key {arr_key} not found in neither .layers nor .var.')
    else:
        arr = index_gene(adata, arr_key, genes)

    if type(arr) == ArrayView: arr = np.array(arr)
    if sp.issparse(arr): arr = arr.A
    arr[np.isnan(arr)] = 0

    if dtype is not None:
        arr = np.array(arr, dtype=dtype)
    if abs:
        arr = np.abs(arr)
    arr = arr.T # genes x cells

    if arr.ndim > 1:
        if groups is not None:
            if type(groups) is str and groups in adata.var.keys():
                grps = np.array(adata.var[groups])
            elif isarray(groups):
                grps = np.array(groups)
            else:
                raise Exception(f'The group information {groups} you provided is not in your adata object.')
            arr_dict = {}
            for g in np.unique(grps):
                arr_dict[g] = fcn_pool(arr[grps==g])
        else:
            arr_dict = {'all': fcn_pool(arr)}
    else:
        arr_dict = {'all': arr}

    ret_dict = {}
    cell_names = np.array(adata.obs_names)
    for g, arr in arr_dict.items():
        if ismatrix(arr):
            arr = arr.A.flatten()
        glst, sarr = list_top_genes(arr, cell_names, None, return_sorted_array=True)
        #ret_dict[g] = {glst[i]: sarr[i] for i in range(len(glst))}
        ret_dict[g] = glst
        if output_values:
            ret_dict[g+'_values'] = sarr
    return pd.DataFrame(data=ret_dict)


def rank_velocity_genes(adata, vkey='velocity_S', prefix_store='rank', **kwargs):
    """Rank genes based on their raw and absolute velocities for each cell group.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the gene-wise velocities.
        vkey: str (default: 'velocity_S')
            The velocity key.
        prefix_store: str (default: 'rank')
            The prefix added to the key for storing the returned in adata.
        kwargs:
            Keyword arguments passed to `vf.rank_genes`.
    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object which has the rank dictionary for velocities in `.uns`.
    """
    rdict = rank_genes(adata, vkey, **kwargs)
    rdict_abs = rank_genes(adata, vkey, abs=True, **kwargs)
    adata.uns[prefix_store + '_' + vkey] = rdict
    adata.uns[prefix_store + '_abs_' + vkey] = rdict_abs
    return adata


def rank_divergence_genes(adata,
                         jkey='jacobian_pca',
                         genes=None,
                         prefix_store='rank_div_gene',
                         **kwargs
                         ):
    """Rank genes based on their diagonal Jacobian for each cell group. 
        Be aware that this 'divergence' refers to the diagonal elements of a gene-wise
        Jacobian, rather than its trace, which is the common definition of the divergence.

        Run .vf.jacobian and set store_in_adata=True before using this function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field in the `.uns` attribute.
        jkey: str (default: 'jacobian_pca')
            The key in .uns of the cell-wise Jacobian matrix.
        genes: list or None (default: None)
            A list of names for genes of interest.
        prefix_store: str (default: 'rank')
            The prefix added to the key for storing the returned ranking info in adata.
        kwargs:
            Keyword arguments passed to `vf.rank_genes`.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object which has the rank dictionary for diagonal jacobians in `.uns`.
    """

    if jkey not in adata.uns_keys():
        raise Exception(f'The provided dictionary key {jkey} is not in .uns.')

    reg = [x for x in adata.uns[jkey]['regulators']]
    eff = [x for x in adata.uns[jkey]['effectors']]
    if reg != eff:
        raise Exception(f'The Jacobian should have the same regulators and effectors.')
    else:
        Genes = adata.uns[jkey]['regulators']
    cell_idx = adata.uns[jkey]['cell_idx']
    div = np.einsum('iij->ji', adata.uns[jkey]['jacobian_gene'])
    Div = create_layer(adata, div, genes=Genes, cells=cell_idx, dtype=np.float32)

    if genes is not None:
        Genes = list(set(Genes).intersection(genes))

    rdict = rank_genes(adata, Div, fcn_pool=lambda x: np.nanmean(x, axis=0), genes=Genes, **kwargs)
    adata.uns[prefix_store + '_' + jkey] = rdict
    return rdict


def rank_s_divergence_genes(adata,
                           skey='sensitivity_pca',
                           genes=None,
                           prefix_store='rank_s_div_gene',
                           **kwargs
                           ):
    """Rank genes based on their diagonal Sensitivity for each cell group.
        Be aware that this 'divergence' refers to the diagonal elements of a gene-wise
        Sensitivity, rather than its trace, which is the common definition of the divergence.

        Run .vf.sensitivity and set store_in_adata=True before using this function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field in the `.uns` attribute.
        skey: str (default: 'sensitivity_pca')
            The key in .uns of the cell-wise sensitivity matrix.
        genes: list or None (default: None)
            A list of names for genes of interest.
        prefix_store: str (default: 'rank')
            The prefix added to the key for storing the returned ranking info in adata.
        kwargs:
            Keyword arguments passed to `vf.rank_genes`.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object which has the rank dictionary for diagonal sensitivity in `.uns`.
    """

    if skey not in adata.uns_keys():
        raise Exception(f'The provided dictionary key {skey} is not in .uns.')

    reg = [x for x in adata.uns[skey]['regulators']]
    eff = [x for x in adata.uns[skey]['effectors']]
    if reg != eff:
        raise Exception(f'The Jacobian should have the same regulators and effectors.')
    else:
        Genes = adata.uns[skey]['regulators']
    cell_idx = adata.uns[skey]['cell_idx']
    div = np.einsum('iij->ji', adata.uns[skey]['sensitivity_gene'])
    Div = create_layer(adata, div, genes=Genes, cells=cell_idx, dtype=np.float32)

    if genes is not None:
        Genes = list(set(Genes).intersection(genes))

    rdict = rank_genes(adata, Div, fcn_pool=lambda x: np.nanmean(x, axis=0), genes=Genes, **kwargs)
    adata.uns[prefix_store + '_' + skey] = rdict
    return rdict


def rank_acceleration_genes(adata,
                            akey='acceleration',
                            prefix_store='rank',
                            **kwargs):
    """Rank genes based on their absolute, positive, negative accelerations for each cell group.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        group: str or None (default: None)
            The cell group that speed ranking will be grouped-by.
        akey: str (default: 'acceleration')
            The acceleration key.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `'rank_acceleration'` information in the `.uns`.
    """

    rdict = rank_genes(adata, akey, **kwargs)
    rdict_abs = rank_genes(adata, akey, abs=True, **kwargs)
    adata.uns[prefix_store + '_' + akey] = rdict
    adata.uns[prefix_store + '_abs_' + akey] = rdict_abs
    return adata


def rank_curvature_genes(adata,
                         ckey='curvature',
                         prefix_store='rank',
                         **kwargs):
    """Rank gene's absolute, positive, negative curvature by different cell groups.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `.uns` attribute.
        groups: `str` or None (default: `None`)
            The cell group that speed ranking will be grouped-by.
        ckey: `str` (default: `curvature`)
            The curvature key.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `'rank_curvature'` related information in the .uns.
    """

    rdict = rank_genes(adata, ckey, **kwargs)
    rdict_abs = rank_genes(adata, ckey, abs=True, **kwargs)
    adata.uns[prefix_store + '_' + ckey] = rdict
    adata.uns[prefix_store + '_abs_' + ckey] = rdict_abs
    return adata


def rank_jacobian_genes(adata,
                        groups=None,
                        jkey='jacobian_pca',
                        abs=False,
                        mode='full reg',
                        exclude_diagonal=False,
                        **kwargs
                        ):
    """Rank genes or gene-gene interactions based on their Jacobian elements for each cell group. 

        Run .vf.jacobian and set store_in_adata=True before using this function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field in the `.uns` attribute.
        groups: str or None (default: None)
            Cell groups used to group the Jacobians.
        jkey: str (default: 'jacobian_pca')
            The key of the stored Jacobians in `.uns`.
        abs: bool (default: False)
            Whether or not to take the absolute value of the Jacobian.
        mode: {'full reg', 'full eff', 'reg', 'eff', 'int'} (default: 'full_reg')
            The mode of ranking:
            (1) `'full reg'`: top regulators are ranked for each effector for each cell group;
            (2) `'full eff'`: top effectors are ranked for each regulator for each cell group;
            (3) '`reg`': top regulators in each cell group;
            (4) '`eff`': top effectors in each cell group;
            (5) '`int`': top effector-regulator pairs in each cell group.
        kwargs:
            Keyword arguments passed to ranking functions.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object which has the rank dictionary in `.uns`.
    """
    J_dict = adata.uns[jkey]
    J = J_dict['jacobian_gene']
    if abs:
        J = np.abs(J)
    if groups is None:
        J_mean = {'all': np.mean(J, axis=2)}
    else:
        if type(groups) is str and groups in adata.obs.keys():
            grps = np.array(adata.obs[groups])
        elif isarray(groups):
            grps = np.array(groups)
        else:
            raise Exception(f'The group information {groups} you provided is not in your adata object.')
        J_mean = average_jacobian_by_group(J, grps[J_dict['cell_idx']])

    eff = np.array([x for x in J_dict['effectors']])
    reg = np.array([x for x in J_dict['regulators']])
    rank_dict= {}
    if mode in ['full reg', 'full_reg']:
        for k, J in J_mean.items():
            rank_dict[k] = table_top_genes(J, eff, reg, n_top_genes=None, **kwargs)
    elif mode in ['full eff', 'full_eff']:
        for k, J in J_mean.items():
            rank_dict[k] = table_top_genes(J.T, reg, eff, n_top_genes=None, **kwargs)
    elif mode == 'reg':
        ov = kwargs.pop('output_values', False)
        for k, J in J_mean.items():
            if exclude_diagonal:
                for i, ef in enumerate(eff):
                    ii = np.where(reg==ef)[0]
                    if len(ii) > 0:
                        J[i, ii] = np.nan
            j = np.nanmean(J, axis=0)
            if ov:
                rank_dict[k], rank_dict[k+'_values'] = list_top_genes(j, reg, None, return_sorted_array=True, **kwargs)
            else:
                rank_dict[k] = list_top_genes(j, reg, None, **kwargs)
        rank_dict = pd.DataFrame(data=rank_dict)
    elif mode == 'eff':
        ov = kwargs.pop('output_values', False)
        for k, J in J_mean.items():
            if exclude_diagonal:
                for i, re in enumerate(reg):
                    ii = np.where(eff==re)[0]
                    if len(ii) > 0:
                        J[ii, i] = np.nan
            j = np.nanmean(J, axis=1)
            if ov:
                rank_dict[k], rank_dict[k+'_values'] = list_top_genes(j, eff, None, return_sorted_array=True, **kwargs)
            else:
                rank_dict[k] = list_top_genes(j, eff, None, **kwargs)
        rank_dict = pd.DataFrame(data=rank_dict)
    elif mode == 'int':
        ov = kwargs.pop('output_values', False)
        for k, J in J_mean.items():
            ints, vals = list_top_interactions(J, eff, reg, **kwargs)
            rank_dict[k] = []
            if ov: rank_dict[k+'_values'] = []
            for l, i in enumerate(ints):
                if not (exclude_diagonal and i[0] == i[1]):
                    rank_dict[k].append(i[0] + ' - ' + i[1])
                    if ov: rank_dict[k+'_values'].append(vals[l])
        rank_dict = pd.DataFrame(data=rank_dict)
    else:
        raise ValueError(f'No such mode as {mode}.')
    return rank_dict


def rank_sensitivity_genes(adata,
                        groups=None,
                        skey='sensitivity_pca',
                        abs=False,
                        mode='full reg',
                        exclude_diagonal=False,
                        **kwargs
                        ):
    """Rank genes or gene-gene interactions based on their sensitivity elements for each cell group.

        Run .vf.sensitivity and set store_in_adata=True before using this function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field in the `.uns` attribute.
        groups: str or None (default: None)
            Cell groups used to group the sensitivity.
        skey: str (default: 'sensitivity_pca')
            The key of the stored sensitivity in `.uns`.
        abs: bool (default: False)
            Whether or not to take the absolute value of the Jacobian.
        mode: {'full reg', 'full eff', 'reg', 'eff', 'int'} (default: 'full_reg')
            The mode of ranking:
            (1) `'full reg'`: top regulators are ranked for each effector for each cell group;
            (2) `'full eff'`: top effectors are ranked for each regulator for each cell group;
            (3) '`reg`': top regulators in each cell group;
            (4) '`eff`': top effectors in each cell group;
            (5) '`int`': top effector-regulator pairs in each cell group.
        kwargs:
            Keyword arguments passed to ranking functions.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object which has the rank dictionary in `.uns`.
    """
    S_dict = adata.uns[skey]
    S = S_dict['sensitivity_gene']
    if abs:
        S = np.abs(S)
    if groups is None:
        S_mean = {'all': np.mean(S, axis=2)}
    else:
        if type(groups) is str and groups in adata.obs.keys():
            grps = np.array(adata.obs[groups])
        elif isarray(groups):
            grps = np.array(groups)
        else:
            raise Exception(f'The group information {groups} you provided is not in your adata object.')
        S_mean = average_jacobian_by_group(S, grps[S_dict['cell_idx']])

    eff = np.array([x for x in S_dict['effectors']])
    reg = np.array([x for x in S_dict['regulators']])
    rank_dict= {}
    if mode in ['full reg', 'full_reg']:
        for k, S in S_mean.items():
            rank_dict[k] = table_top_genes(S, eff, reg, n_top_genes=None, **kwargs)
    elif mode in ['full eff', 'full_eff']:
        for k, S in S_mean.items():
            rank_dict[k] = table_top_genes(S.T, reg, eff, n_top_genes=None, **kwargs)
    elif mode == 'reg':
        ov = kwargs.pop('output_values', False)
        for k, S in S_mean.items():
            if exclude_diagonal:
                for i, ef in enumerate(eff):
                    ii = np.where(reg==ef)[0]
                    if len(ii) > 0:
                        S[i, ii] = np.nan
            j = np.nanmean(S, axis=0)
            if ov:
                rank_dict[k], rank_dict[k+'_values'] = list_top_genes(j, reg, None, return_sorted_array=True, **kwargs)
            else:
                rank_dict[k] = list_top_genes(j, reg, None, **kwargs)
        rank_dict = pd.DataFrame(data=rank_dict)
    elif mode == 'eff':
        ov = kwargs.pop('output_values', False)
        for k, S in S_mean.items():
            if exclude_diagonal:
                for i, re in enumerate(reg):
                    ii = np.where(eff == re)[0]
                    if len(ii) > 0:
                        S[ii, i] = np.nan
            j = np.nanmean(S, axis=1)
            if ov:
                rank_dict[k], rank_dict[k+'_values'] = list_top_genes(j, eff, None, return_sorted_array=True, **kwargs)
            else:
                rank_dict[k] = list_top_genes(j, eff, None, **kwargs)
        rank_dict = pd.DataFrame(data=rank_dict)
    elif mode == 'int':
        ov = kwargs.pop('output_values', False)
        for k, S in S_mean.items():
            ints, vals = list_top_interactions(S, eff, reg, **kwargs)
            rank_dict[k] = []
            if ov: rank_dict[k+'_values'] = []
            for l, i in enumerate(ints):
                if not (exclude_diagonal and i[0] == i[1]):
                    rank_dict[k].append(i[0] + ' - ' + i[1])
                    if ov: rank_dict[k+'_values'].append(vals[l])
        rank_dict = pd.DataFrame(data=rank_dict)
    else:
        raise ValueError(f'No such mode as {mode}.')
    return rank_dict


# ---------------------------------------------------------------------------------------------------
# aggregate regulators or targets
def aggregateRegEffs(adata,
                     data_dict=None,
                     reg_dict=None,
                     eff_dict=None,
                     key="jacobian",
                     basis='pca',
                     store_in_adata=True):
    """Aggregate multiple genes' Jacobian or sensitivity.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field in `.uns`.
        data_dict: `dict`
            A dictionary corresponds to the Jacobian or sensitivity information, must be calculated with either:
            `dyn.vf.jacobian(adata, basis='pca', regulators=genes, effectors=genes)` or
            `dyn.vf.sensitivity(adata, basis='pca', regulators=genes, effectors=genes)`
        reg_dict: `dict`
            A dictionary in which keys correspond to regulator-groups (i.e. TFs for specific cell type) while values
            a list of genes that must have at least one overlapped genes with that from the Jacobian or sensitivity dict.
        eff_dict: `dict`
            A dictionary in which keys correspond to effector-groups (i.e. markers for specific cell type) while values
            a list of genes that must have at least one overlapped genes with that from the Jacobian or sensitivity dict.
        key: `str`
            The key in .uns that corresponds to the Jacobian or sensitivity matrix information.
        basis: str (default: 'pca')
            The embedding data in which the vector field was reconstructed. If `None`, use the vector field function that
            was reconstructed directly from the original unreduced gene expression space.
        store_in_adata: bool (default: `True`)
            Whether to store the divergence result in adata.


    Returns
    -------
        adata: :class:`~anndata.AnnData`
            Depending on `store_in_adata`, it will either return a dictionary that include the aggregated Jacobian or
            sensitivity information or the updated AnnData object that is updated with the `'aggregation'` key in the
            `.uns`. This dictionary contains a 3-dimensional tensor with dimensions n_obs x n_regulators x n_effectors
            as well as other information.
    """

    key_ = key if basis is None else key + '_' + basis
    data_dict = adata.uns[key_] if data_dict is None else data_dict

    tensor, cell_idx, tensor_gene, regulators_, effectors_ = data_dict.get(key), \
                                                                data_dict.get('cell_idx'), \
                                                                data_dict.get(key + '_gene'), \
                                                                data_dict.get('regulators'), \
                                                                data_dict.get('effectors')

    Aggregation = np.zeros((len(eff_dict), len(reg_dict), len(cell_idx)))
    reg_ind = 0
    for reg_key, reg_val in reg_dict.items():
        eff_ind = 0
        for eff_key, eff_val in eff_dict.items():
            reg_val, eff_val = list(np.unique(reg_val)) if reg_val is not None else None, \
                                    list(np.unique(eff_val)) if eff_val is not None else None

            Der, source_genes, target_genes = intersect_sources_targets(reg_val,
                                                                        regulators_,
                                                                        eff_val,
                                                                        effectors_,
                                                                        tensor if tensor_gene is None else tensor_gene)
            if len(source_genes) + len(target_genes) > 0:
                Aggregation[eff_ind, reg_ind, :] = Der.sum(axis=(0, 1)) # dim 0: target; dim 1: source
            else:
                Aggregation[eff_ind, reg_ind, :] = np.nan
            eff_ind += 1
        reg_ind += 0

    ret_dict = {"aggregation": None, "cell_idx": cell_idx}
    # use 'str_key' in dict.keys() to check if these items are computed, or use dict.get('str_key')
    if Aggregation is not None: ret_dict['aggregation_gene'] = Aggregation
    if reg_dict.keys() is not None: ret_dict['regulators'] = list(reg_dict.keys())
    if eff_dict.keys() is not None: ret_dict['effectors'] = list(eff_dict.keys())

    det = [np.linalg.det(Aggregation[:, :, i]) for i in np.arange(Aggregation.shape[2])]
    key = key + "_aggregation" if basis is None else key + "_aggregation_" + basis
    adata.obs[key + '_det'] = np.nan
    adata.obs[key + '_det'][cell_idx] = det
    if store_in_adata:
        adata.uns[key] = ret_dict
        return adata
    else:
        return ret_dict
