from typing import Optional

import anndata
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from .DDRTree_py import DDRTree


def cal_ncenter(ncells: int, ncells_limit: int = 100) -> int:
    """Calculate number of centers to be considered.

    Args:
        ncells: The number of cells.
        ncells_limit: The limitation of number of cells to be considered. Defaults to 100.

    Returns:
        The number of centers to be considered
    """
    return np.round(2 * ncells_limit * np.log(ncells) / (np.log(ncells) + np.log(ncells_limit)))


def directed_pg(
    adata: anndata.AnnData,
    basis: str = "umap",
    transition_method: str = "pearson",
    maxIter: int = 10,
    sigma: float = 0.001,
    Lambda: Optional[float] = None,
    gamma: float = 10,
    ncenter: Optional[int] = None,
    raw_embedding: bool = True,
) -> anndata.AnnData:
    """A function that learns a direct principal graph by integrating the transition matrix between and DDRTree.

    Args:
        adata: An AnnData object,
        basis: The dimension reduction method utilized. Defaults to "umap".
        transition_method: The method to calculate the transition matrix and project high dimensional vector to low
            dimension.
        maxIter: The max iteration numbers. Defaults to 10.
        sigma: The bandwidth parameter. Defaults to 0.001.
        Lambda: The regularization parameter for inverse graph embedding. Defaults to None.
        gamma: The regularization parameter for k-means. Defaults to 10.
        ncenter: The number of centers to be considered. If None, number of centers would be calculated automatically.
            Defaults to None.
        raw_embedding: Whether to project the nodes on the principal graph into the original embedding. Defaults to
            True.

    Raises:
        Exception: invalid `basis`.
        Exception: adata.uns["transition_matrix"] not defined.

    Returns:
        An updated AnnData object that is updated with principal_g_transition, X__DDRTree and X_DDRTree_pg keys.
    """

    X = adata.obsm["X_" + basis] if "X_" + basis in adata.obsm.keys() else None
    if X is None:
        raise Exception("{} is not a key of obsm ({} dimension reduction is not performed yet.).".format(basis, basis))

    transition_matrix = (
        adata.obsp[transition_method + "_transition_matrix"]
        if transition_method + "_transition_matrix" in adata.obsp.keys()
        else None
    )
    if transition_matrix is None:
        raise Exception("transition_matrix is not a key of uns. Please first run cell_velocity.")

    Lambda = 5 * X.shape[1] if Lambda is None else Lambda
    ncenter = 250 if cal_ncenter(X.shape[1]) is None else ncenter

    Z, Y, principal_g, cell_membership, W, Q, C, objs = DDRTree(
        X,
        maxIter=maxIter,
        Lambda=Lambda,
        sigma=sigma,
        gamma=gamma,
        ncenter=ncenter,
    )

    X = csr_matrix(principal_g)
    Tcsr = minimum_spanning_tree(X)
    principal_g = Tcsr.toarray().astype(int)

    # here we can also identify siginificant links using methods related to PAGA
    transition_matrix = transition_matrix.toarray()
    principal_g_transition = cell_membership.T.dot(transition_matrix).dot(cell_membership) * principal_g

    adata.uns["principal_g_transition"] = principal_g_transition
    adata.obsm["X_DDRTree"] = X.T if raw_embedding else Z
    cell_membership = csr_matrix(cell_membership)
    adata.uns["X_DDRTree_pg"] = cell_membership.dot(X.T) if raw_embedding else Y

    return adata
