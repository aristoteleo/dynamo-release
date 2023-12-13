try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Union

import anndata
import numpy as np
from scipy.sparse import csr_matrix, diags, issparse

from ..dynamo_logger import LoggerManager
from .connectivity import adj_to_knn, knn_to_adj
from .graph_operators import build_graph, gradop
from .utils import projection_with_transition_matrix


def pseudotime_velocity(
    adata: anndata.AnnData,
    pseudotime: str = "pseudotime",
    basis: str = "umap",
    adj_key: str = "distances",
    ekey: str = "M_s",
    vkey: str = "velocity_S",
    add_tkey: str = "pseudotime_transition_matrix",
    add_ukey: str = "M_u_pseudo",
    method: Literal["hodge", "naive", "gradient"] = "hodge",
    dynamics_info: bool = False,
    unspliced_RNA: bool = False,
) -> None:
    """Embrace RNA velocity and velocity vector field analysis for pseudotime.

    The AnnData object will be updated, inplace, with low-dimensional velocity, pseudotime based transition matrix as
    well as the pseudotime based RNA velocity.

    When you don't have unspliced/spliced RNA but still want to utilize the velocity/vector field and downstream
    differential geometry analysis, we can use `pseudotime_velocity` to convert pseudotime to RNA velocity. Essentially
    this function computes the gradient of pseudotime and use that to calculate a transition graph (a directed weighted
    graph) between each cell and use that to learn either the velocity on low dimensional embedding as well as the
    gene-wise RNA velocity.

    Args:
        adata: An AnnData object.
        pseudotime: The key in the `adata.obs` that corresponds to the pseudotime values. Defaults to "pseudotime".
        basis: The dictionary key that corresponds to the reduced dimension in `.obsm` attribute. Can be
            `X_spliced_umap` or `X_total_umap`, etc. Defaults to "umap".
        adj_key: The dictionary key that corresponds to the adjacency matrix in `.obsp` attribute. If method is
            `gradient`, the weight will be ignored; while it is `exponent` the weight will be used. Defaults to
            "distances".
        ekey: The dictionary key that corresponds to the gene expression in the layer attribute. This will be used to
            calculate RNA velocity. Defaults to "M_s".
        vkey: The dictionary key that will be used to save the estimated velocity values in the layers attribute.
            Defaults to "velocity_S".
        add_tkey: The dictionary key that will be used to keep the pseudotime-based transition matrix. Defaults to
            "pseudotime_transition_matrix".
        add_ukey: The dictionary key that will be used to save the estimated "unspliced mRNA". Since we assume gamma is
            0, we thus have M_u_pseudo essentially the estimated high dimensional velocity vector. Defaults to
            "M_u_pseudo".
        method: Which pseudotime to vector field method to be used. There are three different methods, `hodge`, `naive`,
            `gradient`. By default, the `hodge` method will be used. Defaults to "hodge".
        dynamics_info: Whether to add dynamics info (a dictionary (with `dynamics` key to the .uns) to your adata object
            which is required for downstream velocity and vector field analysis. Defaults to False.
        unspliced_RNA: Whether to add an unspliced layer to your adata object which is required for downstream velocity
            and vector field analysis. Defaults to False.

    Raises:
        Exception: `method` is invalid.
    """

    logger = LoggerManager.get_main_logger()
    logger.info(
        "Embrace RNA velocity and velocity vector field analysis for pseudotime...",
    )

    logger.info(
        "Retrieve neighbor graph and pseudotime...",
    )
    embedding_key, velocity_key = "X_" + basis, "velocity_" + basis
    E = adata.obsp[adj_key]
    pseudotime_vec = adata.obs[pseudotime].values

    logger.info(
        f"Computing transition graph via calculating pseudotime gradient with {method} method...",
    )

    if method == "hodge":
        grad_ddhodge = gradop(build_graph(E)).dot(pseudotime_vec)

        T = csr_matrix((grad_ddhodge, (E.nonzero())), shape=E.shape)

    elif method == "gradient":
        T = pseudotime_transition(E, pseudotime_vec, laplace_weight=10)

    elif method == "naive":
        knn, dist = adj_to_knn(E, n_neighbors=31)
        T = np.zeros((knn.shape[0], 31))

        logger.log_time()
        for neighbors, distances, i in zip(knn, dist, np.arange(knn.shape[0])):
            logger.report_progress(count=i, total=knn.shape[0])
            meanDis = np.mean(distances[1:])
            weights = distances[1:] / meanDis
            weights_exp = np.exp(weights)

            pseudotime_diff = pseudotime_vec[neighbors[1:]] - pseudotime_vec[neighbors[0]]
            sumW = np.sum(weights_exp)
            weights_scale = weights_exp / sumW
            weights_scale *= np.sign(pseudotime_diff)
            T[i, 1:] = weights_scale
        logger.finish_progress(progress_name="Iterating through each cell...")

        T = knn_to_adj(knn, T)
    else:
        raise Exception(f"{method} method is not supported. only `hodge`, `gradient` and `naive` method is supported!")

    logger.info("Use pseudotime transition matrix to learn low dimensional velocity projection.")
    delta_x = projection_with_transition_matrix(T, X_embedding=adata.obsm[embedding_key], correct_density=True)
    logger.info_insert_adata(velocity_key, "obsm")
    adata.obsm[velocity_key] = delta_x

    logger.info("Use pseudotime transition matrix to learn gene-wise velocity vectors.")
    delta_X = projection_with_transition_matrix(T, adata.layers[ekey].A, True)
    logger.info_insert_adata(vkey, "layers")
    adata.layers[vkey] = csr_matrix(delta_X)

    logger.info_insert_adata(add_tkey, "obsp")
    adata.obsp[add_tkey] = T

    if dynamics_info:
        if "dynamics" not in adata.uns_keys():
            logger.info_insert_adata("dynamics", "uns")
            adata.uns["dynamics"] = {}

        logger.info_insert_adata("has_labeling, has_splicing, splicing_labeling", "uns['dynamics']", indent_level=2)
        adata.uns["dynamics"]["has_labeling"] = False
        adata.uns["dynamics"]["has_splicing"] = True
        adata.uns["dynamics"]["splicing_labeling"] = False

        logger.info_insert_adata(
            "experiment_type, use_smoothed, NTR_vel, est_method", "uns['dynamics']", indent_level=2
        )
        adata.uns["dynamics"]["experiment_type"] = "conventional"
        adata.uns["dynamics"]["use_smoothed"] = True
        adata.uns["dynamics"]["NTR_vel"] = True
        adata.uns["dynamics"]["est_method"] = "conventional"

    if unspliced_RNA:
        logger.info("set velocity_S to be the unspliced RNA.")

        logger.info_insert_adata(add_ukey, "layers", indent_level=2)
        adata.layers[add_ukey] = adata.layers["velocity_S"].copy()

        logger.info("set gamma to be 0 in .var. so that velocity_S = unspliced RNA.")
        logger.info_insert_adata("gamma", "var", indent_level=2)
        adata.varm["pseudotime_vel_params"] = np.zeros((adata.n_vars, 2))
        adata.uns["pseudotime_vel_params_names"] = ["gamma", "gamma_b"]


def pseudotime_transition(E: np.ndarray, pseudotime: np.ndarray, laplace_weight: float = 10) -> csr_matrix:
    """Calculate the transition graph with pseudotime gradient.

    Args:
        E: The adjacency matrix.
        pseudotime: The pseudo time value matrix.
        laplace_weight: The weight of adding laplacian to gradient during calculation of transition graph. Defaults to
            10.

    Returns:
        The pseudo-based transition matrix.
    """
    grad = gradient(E, pseudotime)
    lap = laplacian(E, convention="diffusion")
    T = grad + laplace_weight * lap
    return T


def gradient(E: Union[csr_matrix, np.ndarray], f: np.ndarray, tol: float = 1e-5) -> csr_matrix:
    """Calculate the graph's gradient.

    Args:
        E: The adjacency matrix of the graph.
        f: The pseudotime matrix.
        tol: The tolerance of considering a value to be non-zero. Defaults to 1e-5.

    Returns:
        The gradient of the graph.
    """
    if issparse(E):
        row, col = E.nonzero()
        val = E.data
    else:
        row, col = np.where(E != 0)
        val = E[E != 0]

    G_i, G_j, G_val = np.zeros_like(row), np.zeros_like(col), np.zeros_like(val)

    for ind, i, j, k in zip(np.arange(len(row)), list(row), list(col), list(val)):
        if i != j and np.abs(k) > tol:
            G_i[ind], G_j[ind] = i, j
            G_val[ind] = f[j] - f[i]

    valid_ind = G_val != 0
    G = csr_matrix((G_val[valid_ind], (G_i[valid_ind], G_j[valid_ind])), shape=E.shape)
    G.eliminate_zeros()

    return G


def laplacian(E: Union[csr_matrix, np.ndarray], convention: Literal["graph", "diffusion"] = "graph") -> csr_matrix:
    """Calculate the laplacian of the given graph (here the adjacency matrix).

    Args:
        E: The adjacency matrix.
        convention: The convention of results. Could be either "graph" or "diffusion". If "diffusion" is specified, the
            negative of graph laplacian would be returned. Defaults to "graph".

    Returns:
        The laplacian matrix.

    Raises:
        NotImplementedError: invalid `convention`.
    """
    if issparse(E):
        A = E.copy()
        A.data = np.ones_like(A.data)
        L = diags(A.sum(0).A1, 0) - A
    else:
        A = np.sign(E)
        L = np.diag(np.sum(A, 0)) - A
    if convention == "graph":
        pass
    elif convention == "diffusion":
        L = -L
    else:
        raise NotImplementedError("The convention is not implemented. ")

    L = csr_matrix(L)

    return L
