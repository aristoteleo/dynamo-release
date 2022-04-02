# YEAR: 2019
# COPYRIGHT HOLDER: ddhodge

# Code adapted from https://github.com/kazumits/ddhodge.

from typing import Union

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

from ..dynamo_logger import main_finish_progress, main_info, main_log_time

# from ..vectorfield.scVectorField import graphize_vecfld
from ..tools.graph_calculus import divergence, graphize_velocity, potential
from ..tools.sampling import sample_by_velocity, trn
from ..vectorfield.utils import vecfld_from_adata, vector_field_function

"""from ..tools.graph_operators import (
    build_graph,
    div,
    potential,
)"""
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
    **kwargs,
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

    main_log_time()
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
        main_info("fetch computation results from adata.obsp[%s]..." % (prefix + "ddhodge"))
        adj_mat = adata_.obsp[prefix + "ddhodge"]
    else:
        if adjmethod == "graphize_vecfld":
            main_info("graphizing vectorfield...")
            V_data = func(X_data)
            neighbor_result_prefix = "" if layer is None else layer
            conn_key, dist_key, neighbor_key = _gen_neighbor_keys(neighbor_result_prefix)
            if neighbor_key not in adata_.uns_keys() or to_downsample:
                existing_nbrs_idx = None
            else:
                check_and_recompute_neighbors(adata, result_prefix=neighbor_result_prefix)
                neighbors = adata_.obsp[conn_key]
                existing_nbrs_idx = neighbors.tolil().rows

            adj_mat, nbrs_idx, dists, nbrs = graphize_velocity(
                V_data,
                X_data,
                nbrs_idx=existing_nbrs_idx,
                k=n,
                return_nbrs=True,
            )
            """adj_mat, nbrs = graphize_vecfld(
                func,
                X_data,
                nbrs_idx=existing_nbrs_idx,
                k=n,
                distance_free=distance_free,
                n_int_steps=20,
                cores=cores,
            )"""
        elif adjmethod == "naive":
            main_info(
                'method=naive, get adj_mat from transition matrix in adata directly (adata.uns["transition_matrix"'
            )
            if "transition_matrix" not in adata_.uns.keys():
                raise Exception(
                    "Your adata doesn't have transition matrix created. You need to first "
                    "run dyn.tl.cell_velocity(adata) to get the transition before running"
                    " this function."
                )

            adj_mat = adata_.uns["transition_matrix"][cell_idx, cell_idx]
        else:
            raise ValueError(f"adjmethod can be only one of {'naive', 'graphize_vecfld'}")

    # TODO transform the type of adj_mat here so that we can maintain one set of API (either sparse or numpy)
    # if not issparse(adj_mat):
    #     main_info("adj_mat:%s is not sparse, transforming it to a sparse matrix..." %(str(type(adj_mat))))
    #     adj_mat = csr_matrix(adj_mat)

    # TODO temp fix; refactor to make adj_mat sparse and adjust all the function call
    if issparse(adj_mat):
        adj_mat = adj_mat.toarray()

    # if not all cells are used in the graphize_vecfld function, set diagnoal to be 1
    if len(np.unique(np.hstack(adj_mat.nonzero()))) != adata.n_obs:
        main_info("not all cells are used, set diag to 1...", indent_level=2)
        # temporary fix for github issue #263
        # https://github.com/aristoteleo/dynamo-release/issues/263
        # support numpy and sparse matrices here
        if issparse(adj_mat):
            adj_mat.setdiag(1)
        else:
            np.fill_diagonal(adj_mat, 1)

    # g = build_graph(adj_mat)
    # TODO the following line does not work on sparse matrix.
    A = np.abs(np.sign(adj_mat))

    if (prefix + "ddhodge" not in adata.obsp.keys() or enforce) and not to_downsample:
        adata.obsp[prefix + "ddhodge"] = adj_mat

    # ddhodge_div = div(g)
    # potential_ = potential(g, -ddhodge_div)
    ddhodge_div = divergence(adj_mat, W=A)
    potential_ = potential(adj_mat, W=A, div=ddhodge_div, **kwargs)

    if up_sampling and to_downsample:
        main_info("Constructing W matrix according upsampling=True and downsampling=True options...", indent_level=2)

        query_idx = np.array(list(set(np.arange(adata.n_obs)).difference(cell_idx)))
        query_data = X_data_full[query_idx, :]

        # construct nbrs of query points based on two types of nbrs: from NNDescent (pynndescent) or NearestNeighbors
        if hasattr(nbrs, "kneighbors"):
            dist, query_nbrs_idx = nbrs.kneighbors(query_data)
        elif hasattr(nbrs, "query"):
            query_nbrs_idx, dist = nbrs.query(query_data, k=nbrs.n_neighbors)

        k = query_nbrs_idx.shape[1]
        query_W_row = np.repeat(np.arange(len(query_idx)), k)
        query_W_col = query_nbrs_idx.flatten()

        W = csr_matrix(
            (np.repeat(1 / k, len(query_W_row)), (query_W_row, query_W_col)),
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

    main_finish_progress("ddhodge completed")
