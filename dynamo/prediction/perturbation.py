import numpy as np
import anndata
from typing import Union

from ..tools.cell_velocities import cell_velocities
from ..vectorfield.vector_calculus import jacobian
from .utils import expr_to_pca, pca_to_expr

from ..dynamo_logger import LoggerManager


def perturbation(
    adata: anndata.AnnData,
    genes: Union[str, list],
    expression: Union[float, list] = 10,
    pca_key: Union[str, np.ndarray, None] = None,
    PCs_key: Union[str, np.ndarray, None] = None,
    pca_mean_key: Union[str, np.ndarray, None] = None,
    basis: Union[str, None] = "umap",
    X_pca: Union[np.ndarray, None] = None,
    delta_Y: Union[np.ndarray, None] = None,
    add_delta_Y_key: str = "perturbation_vector",
    add_transition_key: str = None,
    add_velocity_key: str = None,
    add_embedding_key: str = None,
):
    """In silico perturbation of single-cells and prediction of cell fate after perturbation.

    To simulate genetic perturbation and its downstream effects, we take advantage of the analytical Jacobian from our
    vector field function. In particular, we first calculate the perturbation velocity vector:
                        \delta Y = J \dot \delta X.
    where the J is the analytical Jacobian, \delta X is the perturbation vector (that is, if upregulate gene i to
    expression 10 but downregulate gene j to -10 while keep other genes not changed, we have
    delta X = [0, 0, 0, delta x_i = 10, 0, 0, .., x_j = -10, 0, 0, 0]). Because Jacobian encodes the instantaneous
    changes of velocity of any genes after increasing any other gene, J \dot \delta X will produce the perturbation
    effect vector after propogating the genetic perturbation (\delta_X) through the gene regulatory network. We can then
    use X_pca and \delta_Y as a pair (just like M_s and velocity_S) to project the perturbation vector to low
    dimensional space. The \delta_Y can be also used to identify the strongest responders of the genetic perturbation.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        genes:
            The gene or list of genes that will be used to perform in-silico perturbation.
        expression:
             The numerical value or list of values that will be used to encode the genetic perturbation. High positive
             values indicates up-regulation while low negative value repression.
        pca_key
            The key that corresponds to pca embedding. Can also be the actual embedding matrix.
        PCs_key
            The key that corresponds to PC loading embedding. Can also be the actual loading matrix.
        pca_mean_key
            The key that corresponds to means values that used for pca projection. Can also be the actual means matrix.
        basis
            The key that corresponds to perturbation vector projection embedding.
        X_pca:
            The pca embedding matrix.
        delta_Y:
            The actual perturbation matrix. This argument enables more customized perturbation schemes.
        add_delta_Y_key:
            The key that will be used to store the perturbation effect matrix.
        add_transition_key: str or None (default: None)
            The dictionary key that will be used for storing the transition matrix in .obsp.
        add_velocity_key: str or None (default: None)
            The dictionary key that will be used for storing the low dimensional velocity projection matrix in .obsm.
        add_embedding_key: str or None (default: None)
            The dictionary key that will be used for storing the low dimensional velocity projection matrix in .obsm.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            Returns an updated :class:`~anndata.AnnData` with perturbation effect matrix, projected perturbation vectors
            , and a cell transition matrix based on the perturbation vectors.

    """
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
        logger.info(
            "Retrive X_pca, PCs, pca_mean...",
        )

        pca_key = "X_pca" if pca_key is None else pca_key
        PCs_key = "PCs" if PCs_key is None else PCs_key
        pca_mean_key = "pca_mean" if pca_mean_key is None else pca_mean_key

        X_pca = adata.obsm[pca_key]

    if delta_Y is None:
        logger.info(
            "Calculate perturbation effect matrix via \delta Y = J \dot \delta X....",
        )

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

        if len(expression) > 1:
            for i, gene in enumerate(gene_loc):
                X_perturb[:, gene] = expression[i]
        else:
            X_perturb[:, gene_loc] = expression

        # project gene expression back to pca space
        X_perturb_pca = expr_to_pca(X_perturb, PCs, means)

        # calculate Jacobian
        if "jacobian_pca" not in adata.uns_keys():
            jacobian(adata, regulators=valid_genes, effectors=valid_genes)

        Js = adata.uns["jacobian_pca"]["jacobian"]  # pcs x pcs x cells

        # calculate perturbation velocity vector: \delta Y = J \dot \delta X:
        delta_Y = np.zeros_like(X_pca)

        for i in np.arange(adata.n_obs):
            delta_Y[i, :] = Js[:, :, i].dot(X_perturb_pca[i])

    logger.info_insert_adata(add_delta_Y_key, "obsm", indent_level=1)

    adata.obsm[add_delta_Y_key] = delta_Y

    logger.info(
        "project the pca perturbation vector to low dimensional space....",
    )

    if add_transition_key is None:
        transition_key = "perturbation_transition_matrix"
    else:
        transition_key = add_transition_key

    if add_velocity_key is None:
        velocity_key, embedding_key = "velocity_" + basis + "_perturbation", "X_" + basis + "_perturbation"
    else:
        velocity_key, embedding_key = add_velocity_key, "grid_" + add_velocity_key, add_embedding_key

    cell_velocities(
        adata,
        X=X_pca,
        V=delta_Y,
        basis=basis,
        enforce=True,
        add_transition_key=transition_key,
        add_velocity_key=velocity_key,
    )

    logger.info_insert_adata("X_" + basis + "_perturbation", "obsm", indent_level=1)

    logger.info(
        f"so that you can use dyn.pl.streamline_plot(adata, basis={basis} + '_' + {perturbation}) to visualize the "
        f"perturbation vector"
    )
    adata.obsm[embedding_key] = adata.obsm["X_" + basis].copy()
