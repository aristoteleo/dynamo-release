# from tqdm import tqdm

# from anndata._core.views import ArrayView
# import scipy.sparse as sp
from typing import Dict, List, Optional, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
from anndata._core.anndata import AnnData

from ..dynamo_logger import (
    LoggerManager,
    main_info,
    main_info_insert_adata,
    main_info_insert_adata_uns,
    main_warning,
)
from ..tools.sampling import sample
from ..tools.utils import (
    create_layer,
    fetch_states,
    get_pd_row_column_idx,
    get_rank_array,
    index_gene,
    list_top_genes,
    list_top_interactions,
    table_top_genes,
)
from ..utils import isarray, ismatrix
from ..vectorfield import scVectorField
from .scVectorField import SvcVectorField
from .utils import (
    average_jacobian_by_group,
    elementwise_hessian_transformation,
    elementwise_jacobian_transformation,
    get_vf_dict,
    hessian_transformation,
    intersect_sources_targets,
    subset_jacobian_transformation,
    vecfld_from_adata,
    vector_field_function,
    vector_transformation,
)

try:
    import dynode

    use_dynode = "vectorfield" in dir(dynode)
except ImportError:
    use_dynode = False

if use_dynode:
    from .scVectorField import dynode_vectorfield
    from .utils import dynode_vector_field_function


def get_vf_class(adata: AnnData, basis: str = "pca") -> SvcVectorField:
    """Get the corresponding vector field class according to different methods.

        Args:
            adata: AnnData object that contains the reconstructed vector field in the `uns` attribute.
            basis: The embedding data in which the vector field was reconstructed.

        Returns:
            SvcVectorField object that is extracted from the `uns` attribute of adata.
    """
    vf_dict = get_vf_dict(adata, basis=basis)
    if "method" not in vf_dict.keys():
        vf_dict["method"] = "sparsevfc"
    if vf_dict["method"].lower() == "sparsevfc":
        vector_field_class = SvcVectorField()
        vector_field_class.from_adata(adata, basis=basis)
    elif vf_dict["method"].lower() == "dynode":
        vf_dict["parameters"]["load_model_from_buffer"] = True
        vector_field_class = vf_dict["dynode_object"]  # dynode_vectorfield(**vf_dict["parameters"])
    else:
        raise ValueError("current only support two methods, SparseVFC and dynode")
    return vector_field_class


def velocities(
    adata: AnnData,
    init_cells: Optional[List] = None,
    init_states: Optional[list] = None,
    basis: Optional[str] = None,
    vector_field_class: Optional[scVectorField.BaseVectorField] = None,
    layer: Optional[str] = "X",
    dims: Optional[Union[int, list]] = None,
    Qkey: str = "PCs",
) -> AnnData:
    """Calculate the velocities for any cell state with the reconstructed vector field function.

    Args:
        adata: AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        init_cells: Cell name or indices of the initial cell states for the historical or future cell state prediction with
            numerical integration. If the names in init_cells are not found in the adata.obs_name, they will be treated as
            cell indices and must be integers.
        init_states: Initial cell states for the historical or future cell state prediction with numerical integration.
        basis: The embedding data to use for calculating velocities. If `basis` is either `umap` or `pca`, the
            reconstructed trajectory will be projected back to high dimensional space via the `inverse_transform`
            function.
        vector_field_class: If not None, the speed will be computed using this class instead of the vector field stored in adata. You
            can set up the class with a known ODE function, useful when the data is generated through simulation.
        layer: Which layer of the data will be used for predicting cell fate with the reconstructed vector field function.
            The layer once provided, will override the `basis` argument and this function will then predict cell fate in high
            dimensional space.
        dims: The dimensions that will be selected for velocity calculation.
        Qkey: The key of the PCA loading matrix in `.uns`. Only used when basis is `pca`.

    Returns:
        AnnData object that is updated with the `"velocities"` related key in the `.uns`.
    """

    if vector_field_class is None:
        vector_field_class = get_vf_class(adata, basis=basis)

    init_states, _, _, _ = fetch_states(adata, init_states, init_cells, basis, layer, False, None)

    if vector_field_class.vf_dict["normalize"]:
        xm, xscale = vector_field_class.norm_dict["xm"][None, :], vector_field_class.norm_dict["xscale"]
        init_states = (init_states - xm) / xscale
    vec_mat = vector_field_class.func(init_states)
    vec_key = "velocities" if basis is None else "velocities_" + basis

    if np.isscalar(dims):
        vec_mat = vec_mat[:, :dims]
    elif dims is not None:
        vec_mat = vec_mat[:, dims]

    if basis == "pca":
        adata.uns["velocities_pca"] = vec_mat
        Qkey = "PCs" if Qkey is None else Qkey

        if Qkey in adata.uns.keys():
            Q = adata.uns[Qkey]
        elif Qkey in adata.varm.keys():
            Q = adata.varm[Qkey]
        else:
            raise Exception(f"No PC matrix {Qkey} found in neither .uns nor .varm.")

        vel = adata.uns["velocities_pca"].copy()
        vel_hi = vector_transformation(vel, Q)
        create_layer(
            adata,
            vel_hi,
            layer_key="velocity_VecFld",
            genes=adata.var.use_for_pca,
        )

    adata.uns[vec_key] = vec_mat


def speed(
    adata: AnnData,
    basis: Optional[str] = "umap",
    vector_field_class: Optional[scVectorField.BaseVectorField] = None,
    method: str = "analytical",
) -> AnnData:
    """Calculate the speed for each cell with the reconstructed vector field function.

    Args:
        adata: AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: The embedding data in which the vector field was reconstructed.
        vector_field_class: If not None, the speed will be computed using this class instead of the vector field stored in adata. You
            can set up the class with a known ODE function, useful when the data is generated through simulation.
        method: The method that will be used for calculating speed, either `analytical` or `numeric`. `analytical`
            method will use the analytical form of the reconstructed vector field for calculating Jacobian. Otherwise,
            raw velocity vectors are used.

    Returns:
        AnnData object that is updated with the `'speed'` key in the `.obs`.
    """

    if vector_field_class is None:
        vector_field_class = get_vf_class(adata, basis=basis)

    X, V = vector_field_class.get_data()

    if method == "analytical":
        vec_mat = vector_field_class.func(X)
    else:
        vec_mat = adata.obsm["velocity_" + basis] if basis is not None else vector_field_class.vf_dict["Y"]

    speed = np.array([np.linalg.norm(i) for i in vec_mat])

    speed_key = "speed" if basis is None else "speed_" + basis

    adata.obs[speed_key] = speed


def jacobian(
    adata: AnnData,
    regulators: Optional[List] = None,
    effectors: Optional[List] = None,
    cell_idx: Optional[List] = None,
    sampling: Optional[Literal['random', 'velocity', 'trn']] = None,
    sample_ncells: int = 1000,
    basis: str = "pca",
    Qkey: str = "PCs",
    vector_field_class: Optional[scVectorField.BaseVectorField] = None,
    method: str = "analytical",
    store_in_adata: bool = True,
    **kwargs,
):
    """Calculate Jacobian for each cell with the reconstructed vector field.

    If the vector field was reconstructed from the reduced PCA space, the Jacobian matrix will then be inverse
    transformed back to high dimension. Note that this should also be possible for reduced UMAP space and will be
    supported shortly. Note that we compute the Jacobian for the RKHS kernel vector field analytically,
    which is much more computationally efficient than the numerical method.

    Args:
        adata: AnnData object that contains the reconstructed vector field in `.uns`.
        regulators: The list of genes that will be used as regulators when calculating the cell-wise Jacobian matrix. The
            Jacobian is the matrix consisting of partial derivatives of the vector field wrt gene expressions. It can be
            used to evaluate the change in velocities of effectors (see below) as the expressions of regulators
            increase. The regulators are the denominators of the partial derivatives.
        effectors: The list of genes that will be used as effectors when calculating the cell-wise Jacobian matrix. The
            effectors are the numerators of the partial derivatives.
        cell_idx: A list of cell index (or boolean flags) for which the jacobian is calculated.
            If `None`, all or a subset of sampled cells are used.
        sampling: {None, 'random', 'velocity', 'trn'}, (default: None)
            See specific information on these methods in `.tl.sample`.
            If `None`, all cells are used.
        sample_ncells: The number of cells to be sampled. If `sampling` is None, this parameter is ignored.
        basis: The embedding data in which the vector field was reconstructed. If `None`, use the vector field function
            that was reconstructed directly from the original unreduced gene expression space.
        Qkey: The key of the PCA loading matrix in `.uns`.
        vector_field_class: If not `None`, the jacobian will be computed using this class instead of the vector field stored in adata.
        method: The method that will be used for calculating Jacobian, either `'analytical'` or `'numerical'`.
            `'analytical'` method uses the analytical expressions for calculating Jacobian while `'numerical'` method
            uses numdifftools, a numerical differentiation tool, for computing Jacobian. `'analytical'` method is much
            more efficient.
        cores: Number of cores to calculate Jacobian. If cores is set to be > 1, multiprocessing will be used to
            parallel the Jacobian calculation.
        kwargs: Any additional keys that will be passed to `elementwise_jacobian_transformation` function.

    Returns:
        AnnData object that is updated with the `'jacobian'` key in the `.uns`. This is a 3-dimensional tensor with
            dimensions n_effectors x n_regulators x n_obs.
    """

    if vector_field_class is None:
        vector_field_class = get_vf_class(adata, basis=basis)

    if basis == "umap":
        cell_idx = np.arange(adata.n_obs)

    X, V = vector_field_class.get_data()
    if cell_idx is None:
        if sampling is None or sampling == "all":
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

        regulators = np.unique(regulators)
        effectors = np.unique(effectors)

        var_df = adata[:, adata.var.use_for_dynamics].var
        regulators = var_df.index.intersection(regulators)
        effectors = var_df.index.intersection(effectors)

        reg_idx, eff_idx = (
            get_pd_row_column_idx(var_df, regulators, "row"),
            get_pd_row_column_idx(var_df, effectors, "row"),
        )
        if len(regulators) == 0 or len(effectors) == 0:
            raise ValueError(
                "Either the regulator or the effector gene list provided is not in the dynamics gene list!"
            )

        if basis == "pca":
            if Qkey in adata.uns.keys():
                Q = adata.uns[Qkey]
            elif Qkey in adata.varm.keys():
                Q = adata.varm[Qkey]
            else:
                raise Exception(f"No PC matrix {Qkey} found in neither .uns nor .varm.")
            Q = Q[:, : X.shape[1]]
            if len(regulators) == 1 and len(effectors) == 1:
                Jacobian = elementwise_jacobian_transformation(
                    Js, Q[eff_idx, :].flatten(), Q[reg_idx, :].flatten(), **kwargs
                )
            else:
                Jacobian = subset_jacobian_transformation(Js, Q[eff_idx, :], Q[reg_idx, :], **kwargs)
        else:
            Jacobian = Js.copy()
    else:
        Jacobian = None

    ret_dict = {"jacobian": Js, "cell_idx": cell_idx}
    # use 'str_key' in dict.keys() to check if these items are computed, or use dict.get('str_key')
    if Jacobian is not None:
        ret_dict["jacobian_gene"] = Jacobian
    if regulators is not None:
        ret_dict["regulators"] = regulators.to_list()
    if effectors is not None:
        ret_dict["effectors"] = effectors.to_list()

    Js_det = [np.linalg.det(Js[:, :, i]) for i in np.arange(Js.shape[2])]
    jacobian_det_key = "jacobian_det" if basis is None else "jacobian_det_" + basis
    adata.obs[jacobian_det_key] = np.nan
    adata.obs.loc[adata.obs_names[cell_idx], jacobian_det_key] = Js_det

    if store_in_adata:
        jkey = "jacobian" if basis is None else "jacobian_" + basis
        adata.uns[jkey] = ret_dict
        return adata
    else:
        return ret_dict


def hessian(
    adata: AnnData,
    regulators: List,
    coregulators: List,
    effector: Optional[List] = None,
    cell_idx: Optional[List] = None,
    sampling: Optional[Literal['random', 'velocity', 'trn']] = None,
    sample_ncells: int = 1000,
    basis: str = "pca",
    Qkey: str = "PCs",
    vector_field_class: Optional[scVectorField.BaseVectorField] = None,
    method: str = "analytical",
    store_in_adata: bool = True,
    **kwargs,
):
    """Calculate Hessian for each cell with the reconstructed vector field.

    If the vector field was reconstructed from the reduced PCA space, the Hessian matrix will then be inverse
    transformed back to high dimension. Note that this should also be possible for reduced UMAP space and will be
    supported shortly. Note that we compute the Hessian for the RKHS kernel vector field analytically, which is much
    more computationally efficient than the numerical method.

    Args:
        adata: AnnData object that contains the reconstructed vector field in `.uns`.
        regulators: The list of genes that will be used as regulators when calculating the cell-wise Hessian matrix. The
            Hessian is the matrix consisting of secondary partial derivatives of the vector field wrt gene expressions.
            It can be used to evaluate the change in velocities of effectors (see below) as the expressions of
            regulators and co-regulators increase. The regulators/co-regulators are the denominators of the partial
            derivatives.
        coregulators: The list of genes that will be used as regulators when calculating the cell-wise Hessian matrix. The
            Hessian is the matrix consisting of secondary partial derivatives of the vector field wrt gene expressions.
            It can be used to evaluate the change in velocities of effectors (see below) as the expressions of
            regulators and co-regulators increase. The regulators/co-regulators are the denominators of the partial
            derivatives.
        effector: The target gene that will be used as effector when calculating the cell-wise Hessian matrix. Effector
            must be a single gene. The effector is the numerator of the partial derivatives.
        cell_idx: A list of cell index (or boolean flags) for which the Hessian is calculated.
            If `None`, all or a subset of sampled cells are used.
        sampling: {None, 'random', 'velocity', 'trn'}, (default: None)
            See specific information on these methods in `.tl.sample`.
            If `None`, all cells are used.
        sample_ncells: The number of cells to be sampled. If `sampling` is None, this parameter is ignored.
        basis: The embedding data in which the vector field was reconstructed. If `None`, use the vector field function
            that was reconstructed directly from the original unreduced gene expression space.
        Qkey: The key of the PCA loading matrix in `.uns`.
        vector_field_class: If not `None`, the Hessian will be computed using this class instead of the vector field stored in adata.
        method: The method that will be used for calculating Hessian, either `'analytical'` or `'numerical'`.
            `'analytical'` method uses the analytical expressions for calculating Hessian while `'numerical'` method
            uses numdifftools, a numerical differentiation tool, for computing Hessian. `'analytical'` method is much
            more efficient.
        cores: Number of cores to calculate Hessian. Currently note used.
        kwargs: Any additional keys that will be passed to elementwise_hessian_transformation function.

    Returns:
        AnnData object that is updated with the `'Hessian'` key in the `.uns`. This is a 4-dimensional tensor with
            dimensions 1 (n_effector) x n_regulators x n_coregulators x n_obs.
    """

    if vector_field_class is None:
        vector_field_class = get_vf_class(adata, basis=basis)

    if basis == "umap":
        cell_idx = np.arange(adata.n_obs)

    X, V = vector_field_class.get_data()
    if cell_idx is None:
        if sampling is None or sampling == "all":
            cell_idx = np.arange(adata.n_obs)
        else:
            cell_idx = sample(np.arange(adata.n_obs), sample_ncells, sampling, X, V)

    Hessian_func = vector_field_class.get_Hessian(method=method)
    Hs = np.zeros([X.shape[1], X.shape[1], X.shape[1], X.shape[0]])
    for ind, i in enumerate(cell_idx):
        Hs[:, :, :, ind] = Hessian_func(X[i])

    if regulators is not None and coregulators is not None and effector is not None:
        if type(regulators) is str:
            if regulators in adata.var.keys():
                regulators = adata.var.index[adata.var[regulators]]
            else:
                regulators = [regulators]
        if type(coregulators) is str:
            if coregulators in adata.var.keys():
                coregulators = adata.var.index[adata.var[coregulators]]
            else:
                coregulators = [coregulators]
        if type(effector) is str:
            if effector in adata.var.keys():
                effector = adata.var.index[adata.var[effector]]
            else:
                effector = [effector]

            if len(effector) > 1:
                raise Exception(f"effector must be a single gene but you have {effector}. ")

        regulators = np.unique(regulators)
        coregulators = np.unique(coregulators)
        effector = np.unique(effector)

        var_df = adata[:, adata.var.use_for_dynamics].var
        regulators = var_df.index.intersection(regulators)
        coregulators = var_df.index.intersection(coregulators)
        effector = var_df.index.intersection(effector)

        reg_idx, coreg_idx, eff_idx = (
            get_pd_row_column_idx(var_df, regulators, "row"),
            get_pd_row_column_idx(var_df, coregulators, "row"),
            get_pd_row_column_idx(var_df, effector, "row"),
        )
        if len(regulators) == 0 or len(coregulators) == 0 or len(effector) == 0:
            raise ValueError(
                "Either the regulator, coregulator or the effector gene list provided is not in the dynamics gene list!"
            )

        if basis == "pca":
            if Qkey in adata.uns.keys():
                Q = adata.uns[Qkey]
            elif Qkey in adata.varm.keys():
                Q = adata.varm[Qkey]
            else:
                raise Exception(f"No PC matrix {Qkey} found in neither .uns nor .varm.")
            Q = Q[:, : X.shape[1]]
            if len(regulators) == 1 and len(coregulators) == 1 and len(effector) == 1:
                Hessian = [
                    elementwise_hessian_transformation(
                        Hs[:, :, :, i],
                        Q[eff_idx, :].flatten(),
                        Q[reg_idx, :].flatten(),
                        Q[coreg_idx, :].flatten(),
                        **kwargs,
                    )
                    for i in np.arange(Hs.shape[-1])
                ]
            else:
                Hessian = [
                    hessian_transformation(Hs[:, :, :, i], Q[eff_idx, :], Q[reg_idx, :], Q[coreg_idx, :], **kwargs)
                    for i in np.arange(Hs.shape[-1])
                ]
        else:
            Hessian = Hs.copy()
    else:
        Hessian = None

    ret_dict = {"hessian": Hs, "cell_idx": cell_idx}
    # use 'str_key' in dict.keys() to check if these items are computed, or use dict.get('str_key')
    if Hessian is not None:
        ret_dict["hessian_gene"] = Hessian
    if regulators is not None:
        ret_dict["regulators"] = regulators.to_list() if type(regulators) != list else regulators
    if coregulators is not None:
        ret_dict["coregulators"] = coregulators.to_list() if type(coregulators) != list else coregulators
    if effector is not None:
        ret_dict["effectors"] = effector.to_list() if type(effector) != list else effector

    if store_in_adata:
        hkey = "hessian" if basis is None else "hessian_" + basis
        adata.uns[hkey] = ret_dict
        return adata
    else:
        return ret_dict


def laplacian(
    adata: AnnData,
    hkey: str = "hessian_pca",
    basis: str = "pca",
    Qkey: str = "PCs",
    vector_field_class: Optional[scVectorField.BaseVectorField] = None,
    method: str = "analytical",
    **kwargs,
):
    """Calculate Laplacian for each target gene in each cell with the reconstructed vector field.

    If the vector field was reconstructed from the reduced PCA space, the Lapalacian matrix will then be inverse
    transformed back to high dimension. Note that this should also be possible for reduced UMAP space and will be
    supported shortly. Note we compute the Lapalacian for the RKHS kernel vector field analytically, which is much
    more computationally efficient than the numerical method.

    Args:
        adata: AnnData object that contains the reconstructed vector field in `.uns` and the `hkey` (the hessian matrix).
        basis: The embedding data in which the vector field was reconstructed. If `None`, use the vector field function
            that was reconstructed directly from the original unreduced gene expression space.
        Qkey: The key of the PCA loading matrix in `.uns`.
        vector_field_class: If not `None`, the Hessian will be computed using this class instead of the vector field stored in adata.
        method: The method that will be used for calculating Laplacian, either `'analytical'` or `'numerical'`.
            `'analytical'` method uses the analytical expressions for calculating Laplacian while `'numerical'` method
            uses numdifftools, a numerical differentiation tool, for computing Laplacian. `'analytical'` method is much
            more efficient.
        kwargs: Any additional keys that will be passed to elementwise_hessian_transformation function.

    Returns:
        AnnData object that is updated with the `'Laplacian'` key in the `.obs` and `obsm`. The first one is the
            norm of the Laplacian for all target genes in a cell while the second one is the vector of Laplacian for all
            target genes in each cell.
    """

    if hkey not in adata.uns_keys():
        raise Exception(
            f"{hkey} is not in adata.uns_keys(). Please first run dyn.vf.hessian(adata) properly before "
            f"calculating Laplacian. This can be done by calculating Hessian between any three dynamical "
            f"genes which will generate the Hessian matrix."
        )
    else:
        H = adata.uns[hkey]["hessian"]

    if vector_field_class is None:
        vector_field_class = get_vf_class(adata, basis=basis)

    Laplacian_func = vector_field_class.get_Laplacian(method=method)
    Ls = Laplacian_func(H)

    L_key = "Laplacian" if basis is None else "Laplacian_" + basis
    adata.obsm[L_key] = Ls.T
    adata.obs[L_key] = np.linalg.norm(Ls, axis=0)
    if basis == "pca":
        if Qkey in adata.uns.keys():
            Q = adata.uns[Qkey]
        elif Qkey in adata.varm.keys():
            Q = adata.varm[Qkey]
        else:
            raise Exception(f"No PC matrix {Qkey} found in neither .uns nor .varm.")
        Ls_hi = vector_transformation(Ls.T, Q)
        create_layer(
            adata,
            Ls_hi,
            layer_key="laplacian",
            genes=adata.var.use_for_pca,
        )
    elif basis is None:
        create_layer(
            adata,
            Ls.T,
            layer_key="laplacian",
            genes=adata.var.use_for_pca,
        )


def sensitivity(
    adata: AnnData,
    regulators: Optional[List] = None,
    effectors: Optional[List] = None,
    cell_idx: Optional[List] = None,
    sampling: Optional[Literal['random', 'velocity', 'trn']] = None,
    sample_ncells: int = 1000,
    basis: str = "pca",
    Qkey: str = "PCs",
    vector_field_class: Optional[scVectorField.BaseVectorField] = None,
    method: str = "analytical",
    projection_method: str = "from_jacobian",
    store_in_adata: bool = True,
    **kwargs,
) -> Union[AnnData, Dict]:
    """Calculate Sensitivity matrix for each cell with the reconstructed vector field.

    If the vector field was reconstructed from the reduced PCA space, the Sensitivity matrix will then be inverse
    transformed back to high dimension. Note that this should also be possible for reduced UMAP space and will be
    supported shortly. Note that we compute the Sensitivity for the RKHS kernel vector field analytically,
    which is much more computationally efficient than the numerical method.

    Args:
        adata: AnnData object that contains the reconstructed vector field in `.uns`.
        regulators: The list of genes that will be used as regulators when calculating the cell-wise Jacobian matrix. The
            Jacobian is the matrix consisting of partial derivatives of the vector field wrt gene expressions. It can be
            used to evaluate the change in velocities of effectors (see below) as the expressions of regulators
            increase. The regulators are the denominators of the partial derivatives.
        effectors: The list of genes that will be used as effectors when calculating the cell-wise Jacobian matrix. The
            effectors are the numerators of the partial derivatives.
        cell_idx: A list of cell index (or boolean flags) for which the jacobian is calculated.
            If `None`, all or a subset of sampled cells are used.
        sampling: {None, 'random', 'velocity', 'trn'}, (default: None)
            See specific information on these methods in `.tl.sample`.
            If `None`, all cells are used.
        sample_ncells: The number of cells to be sampled. If `sampling` is None, this parameter is ignored.
        basis: The embedding data in which the vector field was reconstructed. If `None`, use the vector field function
            that was reconstructed directly from the original unreduced gene expression space.
        Qkey: The key of the PCA loading matrix in `.uns`.
        vector_field_class: If not `None`, the jacobian will be computed using this class instead of the vector field stored in adata.
        method: The method that will be used for calculating Jacobian, either `'analytical'` or `'numerical'`.
            `'analytical'` method uses the analytical expressions for calculating Jacobian while `'numerical'` method
            uses numdifftools, a numerical differentiation tool, for computing Jacobian. `'analytical'` method is much
            more efficient.
        projection_method: The method that will be used to project back to original gene expression space for calculating gene-wise
            sensitivity matrix:
                (1) 'from_jacobian': first calculate jacobian matrix and then calculate sensitivity matrix. This method
                    will take the combined regulator + effectors gene set for calculating a square Jacobian matrix
                    required for the sensitivity matrix calculation.
                (2) 'direct': The sensitivity matrix on low dimension will first calculated and then projected back to
                    original gene expression space in a way that is similar to the gene-wise jacobian calculation.
        cores: Number of cores to calculate Jacobian. If cores is set to be > 1, multiprocessing will be used to
            parallel the Jacobian calculation.
        kwargs: Any additional keys that will be passed to elementwise_jacobian_transformation function.

    Returns:
        adata: AnnData object that is updated with the `'sensitivity'` key in the `.uns`. This is a 3-dimensional tensor
            with dimensions n_obs x n_regulators x n_effectors.
    """

    regulators, effectors = (
        list(np.unique(regulators)) if regulators is not None else None,
        list(np.unique(effectors)) if effectors is not None else None,
    )
    if vector_field_class is None:
        vector_field_class = get_vf_class(adata, basis=basis)

    if basis == "umap":
        cell_idx = np.arange(adata.n_obs)

    X, V = vector_field_class.get_data()
    if cell_idx is None:
        if sampling is None or sampling == "all":
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

        if projection_method == "direct":
            reg_idx, eff_idx = (
                get_pd_row_column_idx(var_df, regulators, "row"),
                get_pd_row_column_idx(var_df, effectors, "row"),
            )
            if len(regulators) == 0 or len(effectors) == 0:
                raise ValueError(
                    "Either the regulator or the effector gene list provided is not in the dynamics gene list!"
                )

            Q = adata.uns[Qkey][:, : X.shape[1]]
            if len(regulators) == 1 and len(effectors) == 1:
                Sensitivity = elementwise_jacobian_transformation(
                    S,
                    Q[eff_idx, :].flatten(),
                    Q[reg_idx, :].flatten(),
                    **kwargs,
                )
            else:
                Sensitivity = subset_jacobian_transformation(S, Q[eff_idx, :], Q[reg_idx, :], **kwargs)
        elif projection_method == "from_jacobian":
            Js = jacobian(
                adata,
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
                **kwargs,
            )

            J, regulators, effectors = (
                Js.get("jacobian_gene"),
                Js.get("regulators"),
                Js.get("effectors"),
            )
            Sensitivity = np.zeros_like(J)
            n_genes, n_genes_, n_cells = J.shape
            idenity = np.eye(n_genes)
            for i in LoggerManager.progress_logger(
                np.arange(n_cells), progress_name="Calculating sensitivity matrix with precomputed gene-wise Jacobians"
            ):
                s = np.linalg.inv(idenity - J[:, :, i])  # np.transpose(J)
                Sensitivity[:, :, i] = s.dot(np.diag(1 / np.diag(s)))
        else:
            raise ValueError("`projection_method` can only be `from_jacoian` or `direct`!")
    else:
        Sensitivity = None

    ret_dict = {"sensitivity": S, "cell_idx": cell_idx}
    # use 'str_key' in dict.keys() to check if these items are computed, or use dict.get('str_key')
    if Sensitivity is not None:
        ret_dict["sensitivity_gene"] = Sensitivity
    if regulators is not None:
        ret_dict["regulators"] = regulators if type(regulators) == list else regulators.to_list()
    if effectors is not None:
        ret_dict["effectors"] = effectors if type(effectors) == list else effectors.to_list()

    S_det = [np.linalg.det(S[:, :, i]) for i in np.arange(S.shape[2])]
    adata.obs["sensitivity_det_" + basis] = np.nan
    adata.obs["sensitivity_det_" + basis][cell_idx] = S_det
    if store_in_adata:
        skey = "sensitivity" if basis is None else "sensitivity_" + basis
        adata.uns[skey] = ret_dict
        return adata
    else:
        return ret_dict


def acceleration(
    adata: AnnData,
    basis: str = "umap",
    vector_field_class: Optional[scVectorField.BaseVectorField] = None,
    Qkey: str = "PCs",
    method: str = "analytical",
    **kwargs,
):
    """Calculate acceleration for each cell with the reconstructed vector field function. AnnData object is updated with the `'acceleration'` key in the `.obs` as well as .obsm. If basis is `pca`, acceleration matrix will be inverse transformed back to original high dimension space.

    Args:
        adata: AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: The embedding data in which the vector field was reconstructed.
        vector_field_class: If not None, the divergene will be computed using this class instead of the vector field stored in adata.
        Qkey: The key of the PCA loading matrix in `.uns`.
        method: The method that will be used for calculating acceleration field, either `'analytical'` or `'numerical'`.
            `'analytical'` method uses the analytical expressions for calculating acceleration field while `'numerical'`
            method uses numdifftools, a numerical differentiation tool, for computing acceleration. `'analytical'`
            method is much more efficient.
        kwargs: Any additional keys that will be passed to vector_field_class.compute_acceleration function.
    """

    if vector_field_class is None:
        vector_field_class = get_vf_class(adata, basis=basis)

    X, V = vector_field_class.get_data()

    acce_norm, acce = vector_field_class.compute_acceleration(X=X, method=method, **kwargs)

    acce_key = "acceleration" if basis is None else "acceleration_" + basis
    adata.obsm[acce_key] = acce
    adata.obs[acce_key] = acce_norm
    if basis == "pca":
        if Qkey in adata.uns.keys():
            Q = adata.uns[Qkey]
        elif Qkey in adata.varm.keys():
            Q = adata.varm[Qkey]
        else:
            raise Exception(f"No PC matrix {Qkey} found in neither .uns nor .varm.")
        acce_hi = vector_transformation(acce, Q)
        create_layer(
            adata,
            acce_hi,
            layer_key="acceleration",
            genes=adata.var.use_for_pca,
        )
    elif basis is None:
        create_layer(
            adata,
            acce,
            layer_key="acceleration",
            genes=adata.var.use_for_pca,
        )


def curvature(
    adata: AnnData,
    basis: str = "pca",
    vector_field_class: Optional[scVectorField.BaseVectorField] = None,
    formula: int = 2,
    Qkey: str = "PCs",
    method: str = "analytical",
    **kwargs,
):
    """Calculate curvature for each cell with the reconstructed vector field function. AnnData object that is updated with the `curvature` key in the `.obs`.

    Args:
        adata: AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: The embedding data in which the vector field was reconstructed.
        vector_field_class: If not None, the divergene will be computed using this class instead of the vector field stored in adata.
        formula: Which formula of curvature will be used, there are two formulas, so formula can be either `{1, 2}`. By
            default it is 2 and returns both the curvature vectors and the norm of the curvature. The formula one only
            gives the norm of the curvature.
        Qkey: The key of the PCA loading matrix in `.uns`.
        method: The method that will be used for calculating curvature field, either `'analytical'` or `'numerical'`.
            `'analytical'` method uses the analytical expressions for calculating curvature while `'numerical'` method
            uses numdifftools, a numerical differentiation tool, for computing curvature. `'analytical'` method is much
            more efficient.
        kwargs: Any additional keys that will be passed to vector_field_class.compute_curvature function.
    """

    if vector_field_class is None:
        vector_field_class = get_vf_class(adata, basis=basis)

    if formula not in [1, 2]:
        raise ValueError(
            f"There are only two available formulas (formula can be either `{1, 2}`) to calculate "
            f"curvature, but your formula argument is {formula}."
        )

    X, V = vector_field_class.get_data()

    curv, curv_mat = vector_field_class.compute_curvature(X=X, formula=formula, method=method, **kwargs)

    curv_key = "curvature" if basis is None else "curvature_" + basis

    main_info_insert_adata(curv_key, adata_attr="obs", indent_level=1)
    adata.obs[curv_key] = curv

    main_info_insert_adata(curv_key, adata_attr="obsm", indent_level=1)
    adata.obsm[curv_key] = curv_mat
    if basis == "pca":
        curv_hi = vector_transformation(curv_mat, adata.uns[Qkey])
        create_layer(adata, curv_hi, layer_key="curvature", genes=adata.var.use_for_pca)
    elif basis is None:
        create_layer(
            adata,
            curv_mat,
            layer_key="curvature",
            genes=adata.var.use_for_pca,
        )


def torsion(
    adata: AnnData, basis: str = "umap", vector_field_class: Optional[scVectorField.BaseVectorField] = None, **kwargs
):
    """Calculate torsion for each cell with the reconstructed vector field function. AnnData object that is updated with the `torsion` key in the .obs.

    Args:
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: str or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        vector_field_class: dict
            The true ODE function, useful when the data is generated through simulation.
        kwargs:
            Any additional keys that will be passed to vector_field_class.compute_torsion function.

    Examples
    --------
    >>> adata = dyn.sample_data.hematopoiesis()
    >>> dyn.tl.reduceDimension(adata, n_components=3, enforce=True, embedding_key='X_umap_3d')
    >>> adata
    >>> dyn.tl.cell_velocities(adata,
    >>>                        X=adata.layers["M_t"],
    >>>                        V=adata.layers["velocity_alpha_minus_gamma_s"],
    >>>                        basis='umap_3d',
    >>>                        )
    >>> dyn.vf.VectorField(adata, basis='umap_3d')
    >>> dyn.vf.torsion(adata, basis='umap_3d')
    >>> dyn.pl.streamline_plot(adata, color='torsion_umap_3d', basis='umap_3d')
    >>> dyn.pl.streamline_plot(adata, color='torsion_umap_3d')
    """

    if vector_field_class is None:
        vector_field_class = get_vf_class(adata, basis=basis)

    X, V = vector_field_class.get_data()
    torsion_mat = vector_field_class.compute_torsion(X=X, **kwargs)
    torsion = np.array([np.linalg.norm(i) for i in torsion_mat])

    torsion_key = "torsion" if basis is None else "torsion_" + basis

    adata.obs[torsion_key] = torsion
    adata.uns[torsion_key] = torsion_mat


def curl(
    adata: AnnData,
    basis: str = "umap",
    vector_field_class: Optional[scVectorField.BaseVectorField] = None,
    method: str = "analytical",
    **kwargs,
):
    """Calculate Curl for each cell with the reconstructed vector field function. AnnData object is updated with the `'curl'` information in the `.
    obs`. When vector field has three dimension, adata.obs['curl'] (magnitude of curl) and adata.obsm['curl'] (curl vector) will be added; when vector field has two dimension, only adata.obs['curl'] (magnitude of curl) will be provided.

    Args:
        adata: AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: The embedding data in which the vector field was reconstructed.
        vector_field_class: If not None, the divergene will be computed using this class instead of the vector field stored in adata.
        method: The method that will be used for calculating curl, either `analytical` or `numeric`. `analytical`
            method will use the analytical form of the reconstructed vector field for calculating curl while
            `numeric` method will use numdifftools for calculation. `analytical` method is much more efficient.
        kwargs: Any additional keys that will be passed to vector_field_class.compute_curl function.
    """

    if vector_field_class is None:
        vector_field_class = get_vf_class(adata, basis=basis)

    X, V = vector_field_class.get_data()
    curl = vector_field_class.compute_curl(X=X, method=method, **kwargs)
    curl_key = "curl" if basis is None else "curl_" + basis

    if X.shape[1] == 3:
        curl_mag = np.array([np.linalg.norm(i) for i in curl])
        adata.obs[curl_key] = curl_mag
        adata.obsm[curl_key] = curl
    else:
        adata.obs[curl_key] = curl


def divergence(
    adata: AnnData,
    cell_idx: Optional[List] = None,
    sampling: Optional[Literal['random', 'velocity', 'trn']] = None,
    sample_ncells: int = 1000,
    basis: str = "pca",
    vector_field_class=None,
    method: str = "analytical",
    store_in_adata: bool = True,
    **kwargs,
) -> Optional[np.ndarray]:
    """Calculate divergence for each cell with the reconstructed vector field function. Either AnnData object is updated with the `'divergence'` key in the `.obs` or the divergence is returned as a numpy array.

    Args:
        adata: AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        cell_idx: A list of cell index (or boolean flags) for which the jacobian is calculated.
        sampling: {None, 'random', 'velocity', 'trn'}, (default: None)
            See specific information on these methods in `.tl.sample`.
            If `None`, all cells are used.
        sample_ncells: The number of cells to be sampled. If `sampling` is None, this parameter is ignored.
        basis: The embedding data in which the vector field was reconstructed.
        vector_field_class: If not None, the divergene will be computed using this class instead of the vector field stored in adata.
        method: The method that will be used for calculating divergence, either `analytical` or `numeric`. `analytical`
            method will use the analytical form of the reconstructed vector field for calculating divergence while
            `numeric` method will use numdifftools for calculation. `analytical` method is much more efficient.
        store_in_adata: Whether to store the divergence result in adata.
        kwargs: Any additional keys that will be passed to vector_field_class.compute_divergence function.

    Returns:
        the divergence is returned as an np.ndarray if store_in_adata is False.
    """

    if vector_field_class is None:
        vector_field_class = get_vf_class(adata, basis=basis)

    if basis == "umap":
        cell_idx = np.arange(adata.n_obs)

    X, V = vector_field_class.get_data()
    if cell_idx is None:
        if sampling is None or sampling == "all":
            cell_idx = np.arange(adata.n_obs)
        else:
            cell_idx = sample(np.arange(adata.n_obs), sample_ncells, sampling, X, V)

    jkey = "jacobian" if basis is None else "jacobian_" + basis

    div = np.zeros(len(cell_idx))
    calculated = np.zeros(len(cell_idx), dtype=bool)
    if jkey in adata.uns_keys():
        Js = adata.uns[jkey]["jacobian"]
        cidx = adata.uns[jkey]["cell_idx"]
        for i, c in enumerate(
            LoggerManager.progress_logger(cell_idx, progress_name="Calculating divergence with precomputed Jacobians")
        ):
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
