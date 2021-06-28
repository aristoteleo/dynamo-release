# YEAR: 2019
# COPYRIGHT HOLDER: ddhodge

# Code adapted from https://github.com/kazumits/ddhodge.

import numpy as np
from scipy.sparse import csr_matrix
from anndata import AnnData
from typing import Union
from ..vectorfield.scVectorField import graphize_vecfld
from ..vectorfield.utils import (
    vecfld_from_adata,
    vector_field_function,
)
from ..tools.sampling import (
    trn,
    sample_by_velocity,
)
from ..tools.graph_operators import (
    build_graph,
    div,
    potential,
)
from ..tools.connectivity import _gen_neighbor_keys, check_and_recompute_neighbors


def ddhodge(
    adata: AnnData,
    X_data: Union[np.ndarray, None] = None,
    layer: Union[str, None] = None,
    basis: str = "pca",
    n: int = 30,
    VecFld: Union[dict, None] = None,
    adjmethod: str = "graphize_vecfld",
    distance_free: bool = False,
    n_downsamples: int = 5000,
    up_sampling: bool = True,
    sampling_method: str = "velocity",
    seed: int = 19491001,
    enforce: bool = False,
    cores: int = 1,
):
    """Modeling Latent Flow Structure using Hodge Decomposition based on the creation of sparse diffusion graph from the
    reconstructed vector field function. This method is relevant to the curl-free/divergence-free vector field
    reconstruction.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        X_data:
            The user supplied expression (embedding) data that will be used for graph hodege decomposition directly.
        layer:
            Which layer of the data will be used for graph Hodge decomposition.
        basis:
            Which basis of the data will be used for graph Hodge decomposition.
        n:
            Number of nearest neighbors when the nearest neighbor graph is not included.
        VecFld:
            The reconstructed vector field function.
        adjmethod:
            The method to build the ajacency matrix that will be used to create the sparse diffusion graph, can be
            either "naive" or "graphize_vecfld". If "naive" used, the transition_matrix that created during vector field
            projection will be used; if "graphize_vecfld" used, a method that guarantees the preservance of divergence
            will be used.
        n_downsamples:
            Number of cells to downsample to if the cell number is large than this value. Three downsampling methods are
            available, see `sampling_method`.
        up_sampling:
            Whether to assign calculated potential, curl and divergence to cells not sampled based on values from their
            nearest sampled cells.
        sampling_method:
            Methods to downsample datasets to facilitate calculation. Can be one of {`random`, `velocity`, `trn`}, each
            corresponds to random sampling, velocity magnitude based and topology representing network based sampling.
        seed:
            Seed for RandomState. Must be convertible to 32 bit unsigned integers. Used in sampling control points.
            Default is to be 0 for ensure consistency between different runs.
        enforce:
            Whether to enforce the calculation of adjacency matrix for estimating potential, curl, divergence for each
            cell.
        cores:
            Number of cores to run the graphize_vecfld function. If cores is set to be > 1, multiprocessing will be used
            to parallel the graphize_vecfld calculation.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            `AnnData` object that is updated with the `ddhodge` key in the `obsp` attribute which to adjacency matrix
            that corresponds to the sparse diffusion graph. Two columns `potential` and `divergence` corresponds to the
            potential and divergence for each cell will also be added."""

    prefix = "" if basis is None else basis + "_"
    to_downsample = adata.n_obs > n_downsamples

    if VecFld is None:
        VecFld, func = vecfld_from_adata(adata, basis)
    else:

        def func(x):
            return vector_field_function(x, VecFld)

    if X_data is None:
        X_data_full = VecFld["X"].copy()
    else:
        if X_data.shape[0] != adata.n_obs:
            raise ValueError(f"The X_data you provided doesn't correspond to exactly {adata.n_obs} cells")
        X_data_full = X_data.copy()

    if to_downsample:
        if sampling_method == "trn":
            cell_idx = trn(X_data_full, n_downsamples)
        elif sampling_method == "velocity":
            np.random.seed(seed)
            cell_idx = sample_by_velocity(func(X_data_full), n_downsamples)
        elif sampling_method == "random":
            np.random.seed(seed)
            cell_idx = np.random.choice(np.arange(adata.n_obs), n_downsamples)
        else:
            raise ImportError(
                f"sampling method {sampling_method} is not available. Only `random`, `velocity`, `trn` are"
                f"available."
            )
    else:
        cell_idx = np.arange(adata.n_obs)

    X_data = X_data_full[cell_idx, :]
    adata_ = adata[cell_idx].copy()

    if prefix + "ddhodge" in adata_.obsp.keys() and not enforce and not to_downsample:
        adj_mat = adata_.obsp[prefix + "ddhodge"]
    else:
        if adjmethod == "graphize_vecfld":
            neighbor_result_prefix = "" if layer is None else layer
            conn_key, dist_key, neighbor_key = _gen_neighbor_keys(neighbor_result_prefix)
            if neighbor_key not in adata_.uns_keys() or to_downsample:
                Idx = None
            else:
                check_and_recompute_neighbors(adata, result_prefix=neighbor_result_prefix)
                neighbors = adata_.obsp[conn_key]
                Idx = neighbors.tolil().rows

            adj_mat, nbrs = graphize_vecfld(
                func,
                X_data,
                nbrs_idx=Idx,
                k=n,
                distance_free=distance_free,
                n_int_steps=20,
                cores=cores,
            )
        elif adjmethod == "naive":
            if "transition_matrix" not in adata_.uns.keys():
                raise Exception(
                    "Your adata doesn't have transition matrix created. You need to first "
                    "run dyn.tl.cell_velocity(adata) to get the transition before running"
                    " this function."
                )

            adj_mat = adata_.uns["transition_matrix"][cell_idx, cell_idx]
        else:
            raise ValueError(f"adjmethod can be only one of {'naive', 'graphize_vecfld'}")

    # if not all cells are used in the graphize_vecfld function, set diagnoal to be 1
    if len(np.unique(np.hstack(adj_mat.nonzero()))) != adata.n_obs:
        adj_mat.setdiag(1)

    g = build_graph(adj_mat)

    if (prefix + "ddhodge" not in adata.obsp.keys() or enforce) and not to_downsample:
        adata.obsp[prefix + "ddhodge"] = adj_mat

    ddhodge_div = div(g)
    potential_ = potential(g, -ddhodge_div)

    if up_sampling and to_downsample:
        query_idx = list(set(np.arange(adata.n_obs)).difference(cell_idx))
        query_data = X_data_full[query_idx, :]

        if hasattr(nbrs, "kneighbors"):
            dist, nbrs_idx = nbrs.kneighbors(query_data)
        elif hasattr(nbrs, "query"):
            nbrs_idx, dist = nbrs.query(query_data, k=nbrs.n_neighbors)

        k = nbrs_idx.shape[1]
        row, col = np.repeat(np.arange(len(query_idx)), k), nbrs_idx.flatten()
        W = csr_matrix(
            (np.repeat(1 / k, len(row)), (row, col)),
            shape=(len(query_idx), len(cell_idx)),
        )

        query_data_div, query_data_potential = (
            W.dot(ddhodge_div),
            W.dot(potential_),
        )
        (adata.obs[prefix + "ddhodge_sampled"], adata.obs[prefix + "ddhodge_div"], adata.obs[prefix + "potential"],) = (
            False,
            0,
            0,
        )
        adata.obs.loc[adata.obs_names[cell_idx], prefix + "ddhodge_sampled"] = True
        adata.obs.loc[adata.obs_names[cell_idx], prefix + "ddhodge_div"] = ddhodge_div
        adata.obs.loc[adata.obs_names[cell_idx], prefix + "ddhodge_potential"] = potential_
        adata.obs.loc[adata.obs_names[query_idx], prefix + "ddhodge_div"] = query_data_div
        adata.obs.loc[adata.obs_names[query_idx], prefix + "ddhodge_potential"] = query_data_potential
    else:
        adata.obs[prefix + "ddhodge_div"] = ddhodge_div
        adata.obs[prefix + "ddhodge_potential"] = potential_
