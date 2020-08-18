import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from .DDRTree_py import DDRTree_py


def cal_ncenter(ncells, ncells_limit=100):
    return np.round(
        2 * ncells_limit * np.log(ncells) / (np.log(ncells) + np.log(ncells_limit))
    )


def directed_pg(
    adata,
    basis="umap",
    maxIter=10,
    sigma=0.001,
    Lambda=None,
    gamma=10,
    ncenter=None,
    raw_embedding=True,
):
    """A function that learns a direct principal graph by integrating the transition matrix between and DDRTree.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        maxIter: `int` (default: 10)
            maximum iterations
        sigma: `float` (default: 0.001)
            bandwidth parameter
        Lambda: None or `float` (default: None)
            regularization parameter for inverse graph embedding
        ncenter: None or `int` (default: None)
            number of nodes allowed in the regularization graph
        gamma: `float` (default: 10)
            regularization parameter for k-means.
        raw_embedding: `bool` (default: True)
            Whether to project the nodes on the principal graph into the original embedding.

    Returns
    -------
        An updated AnnData object that is updated with principal_g_transition, X__DDRTree and and X_DDRTree_pg keys.

    """
    X = adata.obsm["X_" + basis].T if basis in adata.obsm.keys() else None
    if X is None:
        raise Exception(
            "{} is not a key of obsm ({} dimension reduction is not performed yet.).".format(
                basis, basis
            )
        )

    transition_matrix = (
        adata.uns["transition_matrix"]
        if "transition_matrix" in adata.uns.keys()
        else None
    )
    if transition_matrix is None:
        raise Exception(
            "transition_matrix is not a key of uns. Please first run cell_velocity."
        )

    Lambda = 5 * X.shape[1] if Lambda is None else Lambda
    ncenter = 250 if cal_ncenter(X.shape[1]) is None else ncenter

    DDRTree_res = DDRTree_py(
        X, maxIter=maxIter, Lambda=Lambda, sigma=sigma, gamma=gamma, ncenter=ncenter
    )
    principal_g, cell_membership = (
        DDRTree_res.loc[maxIter - 1, "stree"],
        DDRTree_res.loc[maxIter - 1, "R"],
    )

    X = csr_matrix(principal_g)
    Tcsr = minimum_spanning_tree(X)
    principal_g = Tcsr.toarray().astype(int)

    # here we can also identify siginificant links using methods related to PAGA
    principal_g_transition = (
        cell_membership.T.dot(transition_matrix).dot(cell_membership) * principal_g
    )

    adata.uns["principal_g_transition"] = principal_g_transition
    adata.obsm["X_DDRTree"] = (
        X.T if raw_embedding else DDRTree_res.loc[maxIter - 1, "Z"]
    )
    adata.uns["X_DDRTree_pg"] = (
        cell_membership.dot(X.T) if raw_embedding else DDRTree_res.loc[maxIter - 1, "Y"]
    )

    return adata
