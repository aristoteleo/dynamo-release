from typing import Callable, Optional, Union

import anndata
import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix

from ..dynamo_logger import LoggerManager
from ..tools.cell_velocities import cell_velocities
from ..utils import expr_to_pca, pca_to_expr
from ..vectorfield import SvcVectorField
from ..vectorfield.scVectorField import KOVectorField, vector_field_function_knockout
from ..vectorfield.vector_calculus import (
    jacobian,
    vecfld_from_adata,
    vector_transformation,
)

from ..vectorfield.rank_vf import rank_cell_groups, rank_genes
from .utils import z_score, z_score_inv


def KO(
    adata: anndata.AnnData,
    KO_genes: Union[str, list],
    vecfld: Union[None, Callable] = None,
    vf_key: str = "VecFld",
    basis: str = "pca",
    emb_basis: str = "umap",
    velocity_ko_wt_difference: bool = False,
    add_ko_basis_key: Union[str, None] = None,
    add_embedding_key: Union[str, None] = None,
    store_vf_ko: bool = False,
    add_vf_ko_key: Union[str, None] = None,
    return_vector_field_class: bool = True,
) -> Optional[KOVectorField]:
    """In silico knockout genes (and thus the vector field function) and prediction of cell fate after knockout.

    Args:
        adata: an Anndata object with the vector field function for PCA learned.
        KO_genes: The gene or list of genes that will be used to perform in-silico knockout.
        vecfld: The vector field function.
        vf_key: A key to the vector field functions in adata.uns.
        basis: The basis in which the vector field function is created.
        emb_basis: The embedding basis where the perturbed (KO) vector field function will be projected to.
        velocity_ko_wt_difference: Whether to use the difference from perturbed (KO) vector field to wildtype vector field in embedding space
            instead of raw perturbation (KO) vector field. Using the difference may reveal the perturbation (KO) effects more
            clearly.
        add_ko_basis_key: The key name for the velocity corresponds to the `basis` name whose associated vector field is perturbed
            (KO).
        add_embedding_key: The key name for the velocity corresponds to the `embedding` name to which the high dimensional perturbed
            (KO) vector field will be projected to.
        store_vf_ko: Whether to store the perturbed (KO) vector field function. By default it is False.
        add_vf_ko_key: The key to store the perturbed (KO) vector field function in adata.uns.
        return_vector_field_class: Whether to return the perturbed (KO) vector field class. By default it is True.

    Returns:
        If return_vector_field_class is True, return the perturbed (KO) vector field class and update objected with
        perturbed (KO) vector field in both the PCA and low dimension space. If return_vector_field_class is False,
        return nothing but updates the adata object.
    """

    logger = LoggerManager.gen_logger("dynamo-KO")

    if basis != "pca":
        logger.error("Currently we can only perturb (KO) PCA space based vector field function.")
        raise ValueError()

    if vecfld is None:
        vf = SvcVectorField()
        vf.from_adata(adata, basis=basis, vf_key=vf_key)
    else:
        vf = vecfld

    logger.info(f"In silico knockout {KO_genes}")
    KO_genes = [KO_genes] if type(KO_genes) is str else KO_genes
    vf_ko = vector_field_function_knockout(adata, vf, KO_genes)

    if add_ko_basis_key is None:
        x_basis_key, v_basis_key = "X_" + basis + "_KO", "velocity_" + basis + "_KO"
    else:
        if not add_ko_basis_key.startswith("velocity_"):
            raise ValueError(f"add_ko_basis_key {add_ko_basis_key} must starts with `velocity_`")
        x_basis_key, v_basis_key = "X_" + add_ko_basis_key.split("velocity_")[1], add_ko_basis_key

    if add_embedding_key is None:
        x_emb_key, v_emb_key = "X_" + emb_basis + "_KO", "velocity_" + emb_basis + "_KO"
    else:
        if not add_embedding_key.startswith("velocity_"):
            raise ValueError(f"add_embedding_key {add_embedding_key} must starts with `velocity_`")
        x_emb_key, v_emb_key = "X_" + add_embedding_key.split("velocity_")[1], add_embedding_key

    logger.info_insert_adata(x_basis_key, "obsm")
    adata.obsm[x_basis_key] = adata.obsm["X_" + basis].copy()
    logger.info_insert_adata(v_basis_key, "obsm")
    adata.obsm[v_basis_key] = vf_ko.get_V()

    logger.info_insert_adata(x_emb_key, "obsm")
    adata.obsm[x_emb_key] = adata.obsm["X_" + emb_basis].copy()
    logger.info(f"Project the high dimensional vector field after KO to {emb_basis}.")
    cell_velocities(
        adata,
        X=adata.obsm["X_" + basis],
        V=adata.obsm["velocity_" + basis + "_KO"],
        basis=emb_basis + "_KO",
        enforce=True,
        add_velocity_key=v_emb_key,
    )
    if velocity_ko_wt_difference:
        adata.obsm[v_emb_key] -= adata.obsm["velocity_" + emb_basis]
    if store_vf_ko:
        if add_vf_ko_key is None:
            add_vf_ko_key = "vf_KO"
        logger.info_insert_adata(add_vf_ko_key, "uns")
        adata.uns[add_vf_ko_key] = vf_ko
    if return_vector_field_class:
        return vf_ko


def perturbation(
    adata: anndata.AnnData,
    genes: Union[str, list],
    expression: Union[float, list] = 10,
    perturb_mode: str = "raw",
    cells: Optional[Union[list, np.ndarray]] = None,
    zero_perturb_genes_vel: bool = False,
    pca_key: Optional[Union[str, np.ndarray]] = None,
    PCs_key: Optional[Union[str, np.ndarray]] = None,
    pca_mean_key: Optional[Union[str, np.ndarray]] = None,
    basis: str = "pca",
    emb_basis: str = "umap",
    jac_key: str = "jacobian_pca",
    X_pca: Optional[np.ndarray] = None,
    delta_Y: Optional[np.ndarray] = None,
    projection_method: str = "fp",
    pertubation_method: str = "j_delta_x",
    J_jv_delta_t: float = 1,
    delta_t: float = 1,
    add_delta_Y_key: Optional[str] = None,
    add_transition_key: Optional[str] = None,
    add_velocity_key: Optional[str] = None,
    add_embedding_key: Optional[str] = None,
):
    """In silico perturbation of single-cells and prediction of cell fate after perturbation.

    To simulate genetic perturbation and its downstream effects, we take advantage of the analytical Jacobian from our
    vector field function. In particular, we first calculate the perturbation velocity vector:

    .. math::
        \\delta Y = J \\dot \\delta X

    where the J is the analytical Jacobian, \\delta X is the perturbation vector (that is,

    if overexpress gene i to expression 10 but downexpress gene j to -10 but keep others not changed, we have
    delta X = [0, 0, 0, delta x_i = 10, 0, 0, .., x_j = -10, 0, 0, 0]). Because Jacobian encodes the instantaneous
    changes of velocity of any genes after increasing any other gene, J \\dot \\delta X will produce the perturbation
    effect vector after propagating the genetic perturbation (\\delta_X) through the gene regulatory network. We then
    use X_pca and \\delta_Y as a pair (just like M_s and velocity_S) to project the perturbation vector to low
    dimensional space. The \\delta_Y can be also used to identify the strongest responders of the genetic perturbation.

    Args:
        adata: an Annodata object.
        genes: The gene or list of genes that will be used to perform in-silico perturbation.
        expression: The numerical value or list of values that will be used to encode the genetic perturbation. High positive
             values indicates up-regulation while low negative value repression.
        perturb_mode: The mode for perturbing the gene expression vector, either `raw` or `z_score`.
        cells: The list of the cell indices that we will perform the perturbation.
        zero_perturb_genes_vel: Whether to set the peturbed genes' perturbation velocity vector values to be zero.
        pca_key: The key that corresponds to pca embedding. Can also be the actual embedding matrix.
        PCs_key: The key that corresponds to PC loading embedding. Can also be the actual loading matrix.
        pca_mean_key: The key that corresponds to means values that used for pca projection. Can also be the actual means matrix.
        basis: The key that corresponds to the basis from which the vector field is reconstructed.
        jac_key: The key to the jacobian matrix.
        X_pca: The pca embedding matrix.
        delta_Y: The actual perturbation matrix. This argument enables more customized perturbation schemes.
        projection_method: The approach that will be used to project the high dimensional perturbation effect vector to low dimensional
            space.
        pertubation_method: The approach that will be used to calculate the perturbation effect vector after in-silico genetic
            perturbation. Can only be one of `"j_delta_x", "j_x_prime", "j_jv", "f_x_prime", "f_x_prime_minus_f_x_0"`
        J_jv_delta_t: If pertubation_method is `j_jv`, this will be used to determine the $\\delta x = jv \\delta t_{jv}$
        delta_t: This will be used to determine the $\\delta Y = jv \\delta t$
        add_delta_Y_key: The key that will be used to store the perturbation effect matrix. Both the pca dimension matrix (stored in
            obsm) or the matrix of the original gene expression space (stored in .layers) will use this key. By default
            it is None and is set to be `method + '_perturbation'`.
        add_transition_key: The dictionary key that will be used for storing the transition matrix in .obsp.
        add_velocity_key: The dictionary key that will be used for storing the low dimensional velocity projection matrix in .obsm.
        add_embedding_key: The dictionary key that will be used for storing the low dimensional velocity projection matrix in .obsm.

    Returns:
        adata: Returns an updated :class:`~anndata.AnnData` with perturbation effect matrix, projected perturbation vectors
            , and a cell transition matrix based on the perturbation vectors.

    """

    if pertubation_method.lower() not in ["j_delta_x", "j_x_prime", "j_jv", "f_x_prime", "f_x_prime_minus_f_x_0"]:
        raise ValueError(
            f"your method is set to be {pertubation_method.lower()} but must be one of `j_delta_x`, `j_x_prime`, "
            "`j_jv`,`f_x_prime`, `f_x_prime_minus_f_x_0`"
        )

    logger = LoggerManager.get_main_logger()
    logger.info(
        "In silico perturbation of single-cells and prediction of cell fate after perturbation...",
    )
    if type(genes) == str:
        genes = [genes]
    if type(expression) in [int, float]:
        expression = [expression]

    pca_genes = adata.var_names[adata.var.use_for_pca]
    valid_genes = pca_genes.intersection(genes)

    if len(valid_genes) == 0:
        raise ValueError("genes to perturb must be pca genes (genes used to perform the pca dimension reduction).")
    if len(expression) > 1:
        if len(expression) != len(valid_genes):
            raise ValueError(
                "if you want to set different values for different genes, you need to ensure those genes "
                "are included in the pca gene list and the length of those genes is the same as that of the"
                "expression."
            )

    if X_pca is None:
        logger.info("Retrive X_pca, PCs, pca_mean...")

        pca_key = "X_pca" if pca_key is None else pca_key
        PCs_key = "PCs" if PCs_key is None else PCs_key
        pca_mean_key = "pca_mean" if pca_mean_key is None else pca_mean_key

        X_pca = adata.obsm[pca_key]

    if delta_Y is None:
        logger.info("Calculate perturbation effect matrix via \\delta Y = J \\dot \\delta X....")

        if type(PCs_key) == np.ndarray:
            PCs = PCs_key
        else:
            PCs = adata.uns[PCs_key]

        if type(pca_mean_key) == np.ndarray:
            means = pca_mean_key
        else:
            means = adata.uns[pca_mean_key]

        # project pca gene expression back to original gene expression:
        X = pca_to_expr(X_pca, PCs, means)

        # get gene position
        gene_loc = [adata.var_names[adata.var.use_for_pca].get_loc(i) for i in valid_genes]

        # in-silico perturbation
        X_perturb = X.copy()

        if cells is None:
            cells = np.arange(adata.n_obs)

        for i, gene in enumerate(gene_loc):
            if perturb_mode == "z_score":
                x = X_perturb[:, gene]
                _, m, s = z_score(x, 0)
                X_perturb[cells, gene] = z_score_inv(expression[i], m, s)
            elif perturb_mode == "raw":
                X_perturb[cells, gene] = expression[i]
            else:
                raise NotImplementedError(f"The perturbation mode {perturb_mode} is not supported.")

        # project gene expression back to pca space
        X_perturb_pca = expr_to_pca(X_perturb, PCs, means)

        # calculate Jacobian
        if jac_key not in adata.uns_keys():
            jacobian(adata, regulators=valid_genes, effectors=valid_genes)

        Js = adata.uns[jac_key]["jacobian"]  # pcs x pcs x cells

        # calculate perturbation velocity vector: \delta Y = J \dot \delta X:
        delta_Y = np.zeros_like(X_pca)

        # get the actual delta_X:
        if pertubation_method.lower() in ["j_delta_x", "j_x_prime", "j_jv"]:
            if pertubation_method.lower() == "j_delta_x":
                delta_X = X_perturb_pca - X_pca
            elif pertubation_method.lower() == "j_x_prime":
                delta_X = X_perturb_pca
            elif pertubation_method.lower() == "j_jv":
                tmp = X_perturb_pca - X_pca
                delta_X = np.zeros_like(X_pca)
                for i in np.arange(adata.n_obs):
                    delta_X[i, :] = Js[:, :, i].dot(tmp[i] * J_jv_delta_t)

            for i in np.arange(adata.n_obs):
                delta_Y[i, :] = Js[:, :, i].dot(delta_X[i] * delta_t)

    if add_delta_Y_key is None:
        add_delta_Y_key = pertubation_method + "_perturbation"
    logger.info_insert_adata(add_delta_Y_key, "obsm", indent_level=1)

    if pertubation_method.lower() == "f_x_prime":
        _, func = vecfld_from_adata(adata, basis)
        vec_mat = func(X_perturb_pca)
        delta_Y = vec_mat
    elif pertubation_method.lower() == "f_x_prime_minus_f_x_0":
        _, func = vecfld_from_adata(adata, basis)
        vec_mat = func(X_perturb_pca) - func(X_pca)
        delta_Y = vec_mat

    adata.obsm[add_delta_Y_key] = delta_Y

    perturbation_csc = vector_transformation(delta_Y, PCs)

    adata.layers[add_delta_Y_key] = csr_matrix(adata.shape, dtype=np.float64)
    adata.layers[add_delta_Y_key][:, adata.var.use_for_pca] = perturbation_csc
    if zero_perturb_genes_vel:
        adata.layers[add_delta_Y_key][:, gene_loc] = 0

    logger.info(
        "project the pca perturbation vector to low dimensional space....",
    )

    if add_transition_key is None:
        transition_key = "perturbation_transition_matrix"
    else:
        transition_key = add_transition_key

    if add_velocity_key is None:
        velocity_key, embedding_key = "velocity_" + emb_basis + "_perturbation", "X_" + emb_basis + "_perturbation"
    else:
        velocity_key, embedding_key = add_velocity_key, add_embedding_key

    cell_velocities(
        adata,
        X=X_pca,
        V=delta_Y,
        basis=emb_basis,
        enforce=True,
        method=projection_method,
        add_transition_key=transition_key,
        add_velocity_key=velocity_key,
    )

    logger.info_insert_adata("X_" + emb_basis + "_perturbation", "obsm", indent_level=1)

    logger.info(
        f"you can use dyn.pl.streamline_plot(adata, basis='{emb_basis}_perturbation') to visualize the "
        f"perturbation vector"
    )
    adata.obsm[embedding_key] = adata.obsm["X_" + emb_basis].copy()


def rank_perturbation_genes(
    adata: AnnData, pkey: str = "j_delta_x_perturbation", prefix_store: str = "rank", **kwargs
) -> AnnData:
    """Rank genes based on their raw and absolute perturbation effects for each cell group.

    Args:
        adata: AnnData object that contains the gene-wise perturbation effect vectors.
        pkey: The perturbation key.
        prefix_store: The prefix added to the key for storing the returned ranking information in adata.
        kwargs: Keyword arguments passed to `vf.rank_genes`.

    Returns:
        adata: AnnData object which has the rank dictionary for perturbation effects in `.uns`.
    """
    rdict = rank_genes(adata, pkey, **kwargs)
    rdict_abs = rank_genes(adata, pkey, abs=True, **kwargs)
    adata.uns[prefix_store + "_" + pkey] = rdict
    adata.uns[prefix_store + "_abs_" + pkey] = rdict_abs
    return adata


def rank_perturbation_cells(
    adata: AnnData, pkey: str = "j_delta_x_perturbation", prefix_store: str = "rank", **kwargs
) -> AnnData:
    """Rank cells based on their raw and absolute perturbation for each cell group.

    Args:
        adata: AnnData object that contains the gene-wise velocities.
        pkey: The perturbation key.
        prefix_store: The prefix added to the key for storing the returned in adata.
        kwargs: Keyword arguments passed to `vf.rank_cells`.

    Returns:
        adata: AnnData object which has the rank dictionary for perturbation effects in `.uns`.
    """
    rdict = rank_cell_groups(adata, pkey, **kwargs)
    rdict_abs = rank_cell_groups(adata, pkey, abs=True, **kwargs)
    adata.uns[prefix_store + "_" + pkey + "_cells"] = rdict
    adata.uns[prefix_store + "_abs_" + pkey + "_cells"] = rdict_abs
    return adata


def rank_perturbation_cell_clusters(
    adata: AnnData, pkey: str = "j_delta_x_perturbation", prefix_store: str = "rank", **kwargs
) -> AnnData:
    """Rank cells based on their raw and absolute perturbation for each cell group.

    Args:
        adata: AnnData object that contains the gene-wise velocities.
        pkey: The perturbation key.
        prefix_store: The prefix added to the key for storing the returned in adata.
        kwargs: Keyword arguments passed to `vf.rank_cells`.

    Returns:
        adata: AnnData object which has the rank dictionary for perturbation effects in `.uns`.
    """
    rdict = rank_cell_groups(adata, pkey, **kwargs)
    rdict_abs = rank_cell_groups(adata, pkey, abs=True, **kwargs)
    adata.uns[prefix_store + "_" + pkey + "_cell_groups"] = rdict
    adata.uns[prefix_store + "_abs_" + pkey + "_cells_groups"] = rdict_abs
    return adata
