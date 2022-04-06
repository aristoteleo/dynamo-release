from typing import List, Union

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree

from ..dynamo_logger import LoggerManager, main_info, main_warning
from ..prediction.fate import _fate
from ..tools.clustering import neighbors
from ..tools.Markov import DiscreteTimeMarkovChain
from ..tools.utils import fetch_states
from ..vectorfield import vector_field_function
from .utils import (
    arclength_sampling,
    integrate_streamline,
    remove_redundant_points_trajectory,
)

# from sklearn.preprocessing import OrdinalEncoder


def classify_clone_cell_type(adata, clone, clone_column, cell_type_column, cell_type_to_excluded):
    """find the dominant cell type of all the cells that are from the same clone"""
    cell_ids = np.where(adata.obs[clone_column] == clone)[0]

    to_check = adata[cell_ids].obs[cell_type_column].value_counts().index.isin(list(cell_type_to_excluded))

    cell_type = np.where(to_check)[0]

    return cell_type


def prune_transition(
    adata: anndata.AnnData,
    group: str,
    basis: str = "umap",
    n_neighbors: int = 30,
    neighbor_key: Union[str, None] = None,
    graph_mat: np.ndarray = None,
    state_graph_method: str = "vf",
):
    """This function prune a cell group transiton graph based on cell similarity graph (kNN graph).

    The pruning algorithm is as following: assuming the vf based cell-type transition graph is `m` (cell type x cell
    type matrix); the `M` matrix as the cell to cell-type assignment matrix (row is the cell and column the cell type;
    if i-th cell is j-th cell type, the `M_{ij}` is 1). the knn graph between cells based on the umap embedding (or
    others) is `n` (number of cells x number of cells matrix). We compute `t(M) n M` to get a cell-type by cell type
    connectivity graph M' (basically this propagates the cell type to cell matrix to the cell-cell knn graph and then
    lump the transition down to cell-type). Lastly, `g * M'`  will give pruned graph, where `g` is the vector field
    based cell-type transition graph. As you can see the resultant graph considers both vector field based connection
    and the similarity relationship of cells in expression space.

    Parameters
    ----------
    adata:
        AnnData object.
    group:
        Cell graph that will be used to build transition graph and lineage tree.
    basis:
         The basis that will be used to build the k-nearest neighbor graph when neighbor_key is not set.
    n_neighbors:
        The number of neighbors that will be used to build the k-nn graph, passed to `dyn.tl.neighbors` function. Not
        used when neighbor_key provided.
    neighbor_key:
         The nearest neighbor graph key in `adata.obsp`. This nearest neighbor graph will be used to build a
         gene-expression space based cell-type level connectivity graph.
    state_graph_method:
         Method that will be used to build the initial state graph.

    Returns
    -------
    M:
        The pruned cell state transition graph.
    """

    logger = LoggerManager.gen_logger("dynamo-prune_transition")
    logger.log_time()
    from patsy import dmatrix

    if group not in adata.obs.columns:
        raise Exception(f"group has to be in adata.obs.columns, but you have {group}. ")

    data = adata.obs
    groups = data[group]
    uniq_grps, data[group] = groups.unique(), list(groups)
    sorted_grps = np.sort(uniq_grps)

    if graph_mat is not None:
        if graph_mat.shape != (len(uniq_grps), len(uniq_grps)):
            raise Exception(f"the input graph_mat has to have the same shape as ({len(uniq_grps), len(uniq_grps)})")

        group_graph = graph_mat
    else:
        if group + "_graph" not in adata.uns_keys():
            main_info(f"build state graph `g` via {state_graph_method}")
            state_graph(adata, group=group, basis=basis, method=state_graph_method)  # the markov method
        group_graph = adata.uns[group + "_graph"]["group_graph"]

    if neighbor_key is None:
        main_info(f"build knn graph with {n_neighbors} neighbors in {basis} basis.")
        neighbors(adata, basis=basis, result_prefix=basis + "_knn", n_neighbors=n_neighbors)
        transition_matrix = adata.obsp[basis + "_knn_distances"]
    else:
        main_info(f"retrieve knn graph via {neighbor_key} ley.")
        transition_matrix = adata.obsp[neighbor_key]

    main_info("build cell to cell graph assignment matrix via `dmatrix` from `pasty`")
    cell_membership = csr_matrix(dmatrix(f"~{group}+0", data=data))

    main_info("build lumped cell group to cell group connectivity matrix via `t(M) n M`.")
    membership_matrix = cell_membership.T.dot(transition_matrix).dot(cell_membership)

    main_info("prune vf based cell graph transition graph via g' = `M' g")
    # note that dmatrix will first sort the unique group names and then construct the design matrix, so this is needed.
    membership_df = pd.DataFrame(membership_matrix.A > 0, index=sorted_grps, columns=sorted_grps)

    M = (group_graph * (membership_df.loc[uniq_grps, uniq_grps].values > 0) > 0).astype(float)

    logger.finish_progress(progress_name="prune_transition")

    return M


def state_graph(
    adata,
    group,
    method="vf",
    transition_mat_key="pearson_transition_matrix",
    approx=False,
    eignum=5,
    basis="umap",
    layer=None,
    arc_sample=False,
    sample_num=100,
    prune_graph=False,
    **kwargs,
):
    """Estimate the transition probability between cell types using method of vector field integrations or Markov chain
    lumping.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that will be used to calculate a cell type (group) transition graph.
        group: `str`
            The attribute to group cells (column names in the adata.obs).
        method: `str` (default: 'vf')
            The method that will be used to construct lumped cell state graph. Must be one of {`vf` or `markov`}
        transition_mat_key: `str` (default: 'pearson_transition_matrix')
            The key that corresponds to the transition graph used in the KernelMarkovChain class for lumping.
        approx: `bool` (default: False)
            Whether to use streamplot to get the integration lines from each cell.
        eignum: `int` (default: 5)
            The number of eigen-vectors when performing the eigen-decomposition to obtain the stationary
            distribution. 5 should be sufficient as the stationary distribution will be the first eigenvector. This also
            accelerates the calculation.
        basis: `str` or None (default: `umap`)
            The embedding data to use for predicting cell fate. If `basis` is either `umap` or `pca`, the reconstructed
            trajectory will be projected back to high dimensional space via the `inverse_transform` function.
        layer: `str` or None (default: `None`)
            Which layer of the data will be used for predicting cell fate with the reconstructed vector field function.
            The layer once provided, will override the `basis` argument and then predicting cell fate in high
            dimensional space.
        sample_num: `int` (default: 100)
            The number of cells to sample in each group that will be used for calculating the transitoin graph between
            cell groups. This is required for facilitating the calculation.
        prune_graph: `bool` (default: `False`)
            Whether to prune the transition graph based on cell similarities in `basis` bases.
        kwargs:
            Additional parameters that will be passed to `prune_transition` function.

    Returns
    -------
        An updated adata object that is added with the `group + '_graph'` key, including the transition graph
        and the average transition time.
    """
    logger = LoggerManager.get_main_logger()
    timer_logger = LoggerManager.get_temp_timer_logger()
    timer_logger.log_time()

    logger.info("Estimating the transition probability between cell types...")
    groups, uniq_grp = adata.obs[group], list(adata.obs[group].unique())

    if method.lower() in ["naive", "markov"]:
        logger.info("Applying kernel Markov chain")
        T = adata.obsp[transition_mat_key]
        if np.isclose(T.sum(1), 1).sum() > np.isclose(T.sum(0), 1).sum():
            logger.info("KernelMarkovChain assuming column sum to be 1. Transposing transition matrix")
            T = T.T
        if sp.issparse(T):
            T = T.A
        dtmc = DiscreteTimeMarkovChain(P=T, eignum=eignum, check_norm=False)

        # ord_enc = OrdinalEncoder()
        # labels = ord_enc.fit_transform(adata.obs[[group]])
        # labels = labels.flatten().astype(int)
        labels = np.zeros(len(groups), dtype=int)
        for i, grp in enumerate(uniq_grp):
            labels[groups == grp] = i

        grp_graph = dtmc.lump(labels).T if method == "markov" else dtmc.naive_lump(T.A, labels).T
        label_len, grp_avg_time = len(np.unique(labels)), None
        grp_graph = grp_graph[:label_len, :label_len]

    elif method == "vf":
        logger.info("Applying vector field")
        grp_graph = np.zeros((len(uniq_grp), len(uniq_grp)))
        grp_avg_time = np.zeros((len(uniq_grp), len(uniq_grp)))

        all_X, VecFld, t_end, _ = fetch_states(
            adata,
            init_states=None,
            init_cells=adata.obs_names,
            basis=basis,
            layer=layer,
            average=False,
            t_end=None,
        )
        logger.report_progress(percent=0, progress_name="KDTree parameter preparation computation")
        logger.log_time()
        kdt = cKDTree(all_X, leafsize=30)
        logger.finish_progress(progress_name="KDTree computation")
        vf_dict = adata.uns["VecFld_" + basis]

        for i, cur_grp in enumerate(LoggerManager.progress_logger(uniq_grp, progress_name="iterate groups")):
            init_cells = adata.obs_names[groups == cur_grp]
            if sample_num is not None:
                cell_num = np.min((sample_num, len(init_cells)))
                ind = np.random.choice(len(init_cells), cell_num, replace=False)
                init_cells = init_cells[ind]

            init_states, _, _, _ = fetch_states(
                adata,
                init_states=None,
                init_cells=init_cells,
                basis=basis,
                layer=layer,
                average=False,
                t_end=None,
            )
            if approx and basis != "pca" and layer is None:
                X_grid, V_grid = (
                    vf_dict["grid"],
                    vf_dict["grid_V"],
                )
                N = int(np.sqrt(V_grid.shape[0]))
                X_grid, V_grid = (
                    np.array([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])]),
                    np.array(
                        [
                            V_grid[:, 0].reshape((N, N)),
                            V_grid[:, 1].reshape((N, N)),
                        ]
                    ),
                )

                t, X = integrate_streamline(
                    X_grid[0],
                    X_grid[1],
                    V_grid[0],
                    V_grid[1],
                    integration_direction="forward",
                    init_states=init_states,
                    interpolation_num=250,
                    average=False,
                )
            else:
                t, X = _fate(
                    lambda x: vector_field_function(x=x, vf_dict=vf_dict),
                    init_states,
                    t_end=t_end,
                    step_size=None,
                    direction="forward",
                    interpolation_num=250,
                    average=False,
                )
                # t, X = np.hstack(t), np.hstack(X).T

            len_per_cell = None if type(t) == list else len(t)
            cell_num = len(t) if type(X) == list else int(X.shape[0] / len(t))

            knn_dist_, knn_ind_ = kdt.query(init_states, k=2)
            dist_min, dist_threshold = (
                np.max([knn_dist_[:, 1].min(), 1e-3]),
                np.mean(knn_dist_[:, 1]),
            )

            for j in np.arange(cell_num):
                if len_per_cell is not None:
                    cur_ind = np.arange(j * len_per_cell, (j + 1) * len_per_cell)
                    Y, arclength, T_bool = remove_redundant_points_trajectory(
                        X[cur_ind], tol=dist_min, output_discard=True
                    )

                    if arc_sample:
                        Y, arclength, T = arclength_sampling(Y, arclength / 1000, t=t[~T_bool])
                    else:
                        T = t[~T_bool]
                else:
                    Y, T = X[j].T, t[j] if type(t[j]) == np.ndarray else np.array(t[j])

                knn_dist, knn_ind = kdt.query(Y, k=1)

                # set up a dataframe with group and time
                pass_t = np.where(knn_dist < dist_threshold)[0]
                pass_df = pd.DataFrame({"group": adata[knn_ind[pass_t]].obs[group], "t": T[pass_t]})
                # only consider trajectory that pass at least 10 cells in group as confident pass
                pass_group_counter = pass_df.group.value_counts()
                pass_groups, confident_pass_check = (
                    pass_group_counter.index.tolist(),
                    np.where(pass_group_counter > 10)[0],
                )
                # assign the transition matrix and average transition time
                if len(confident_pass_check) > 0:
                    ind_other_cell_type = [uniq_grp.index(k) for k in np.array(pass_groups)[confident_pass_check]]
                    grp_graph[i, ind_other_cell_type] += 1
                    grp_avg_time[i, ind_other_cell_type] += (
                        pass_df.groupby("group")["t"].mean()[confident_pass_check].values
                    )

            # average across cells
            grp_avg_time[i, :] /= grp_graph[i, :]
            grp_graph[i, :] /= cell_num

    else:
        raise NotImplementedError("Only vector field (vf) or Markov chain (markov) based lumping are supported.")

    if prune_graph:
        grp_graph = prune_transition(
            adata,
            group,
            basis,
            graph_mat=grp_graph,
            **kwargs,
        )
    adata.uns[group + "_graph"] = {"group_graph": grp_graph, "group_avg_time": grp_avg_time, "group_names": uniq_grp}
    timer_logger.finish_progress(progress_name="State graph estimation")
    return adata


def tree_model(
    adata: anndata.AnnData,
    group: str,
    progenitor: str,
    terminators: List[str],
    basis: str = "umap",
    n_neighbors: int = 30,
    neighbor_key: Union[str, None] = None,
    graph_mat: np.ndarray = None,
    state_graph_method: str = "vf",
    prune_graph: bool = True,
    row_norm: bool = True,
) -> pd.DataFrame:
    """This function learns a tree model of cell states (types).

    It is based on the shortest path from the source to target cells of the pruned vector field based cell-type
    transition graph. The pruning was done by restricting cell state transition that are only between cell states that
    are nearby in gene expression space (often low gene expression space).

    Parameters
    ----------
    adata:
        AnnData object.
    group:
        Cell graph that will be used to build transition graph and lineage tree.
    progenitor:
        The source cell type name of the lineage tree.
    terminators:
         The terminal cell type names of the lineage tree.
    basis:
         The basis that will be used to build the k-nearest neighbor graph when neighbor_key is not set.
    n_neighbors:
        The number of neighbors that will be used to build the k-nn graph, passed to `dyn.tl.neighbors` function. Not
        used when neighbor_key provided.
    neighbor_key:
         The nearest neighbor graph key in `adata.obsp`. This nearest neighbor graph will be used to build a
         gene-expression space based cell-type level connectivity graph.
    state_graph_method:
         Method that will be used to build the initial state graph.
    prune_graph: `bool` (default: `True`)
        Whether to prune the transition graph based on cell similarities in `basis` bases first before learning tree
        model.
    row_norm: `bool` (default: `True`)
        Whether to normalize each row so that each row sum up to be 1. Note that row, columns in transition matrix
        correspond to source and targets in dynamo by default.

    Returns
    -------
    res:
        The final tree model of cell groups. See following example on how to visualize the tree via dynamo.

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.pancreatic_endocrinogenesis()
    >>> dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.tl.cell_velocities(adata)
    >>> dyn.vf.VectorField(adata, basis='umap', pot_curl_div=False)
    >>> dyn.pd.state_graph(adata, group='clusters', basis='umap')
    >>> res = dyn.pd.tree_model(adata, group='clusters', basis='umap')
    >>> # in the following we first copy the state_graph result to a new key and then replace the `group_graph` key of
    >>> # the state_graph result and visualize tree model via dynamo.
    >>> adata.obs['clusters2'] = adata.obs['clusters'].copy()
    >>> adata.uns['clusters2_graph'] = adata.uns['clusters_graph'].copy()
    >>> adata.uns['clusters2_graph']['group_graph'] = res
    >>> dyn.pl.state_graph(adata, group='clusters2', keep_only_one_direction=False, transition_threshold=None,
    >>> color='clusters2', basis='umap', show_legend='on data')
    """

    logger = LoggerManager.gen_logger("dynamo-tree_model")
    logger.log_time()

    data = adata.obs
    groups = data[group]
    uniq_grps, data[group] = groups.unique(), list(groups)

    progenitor = progenitor[0] if type(progenitor) is not str else progenitor
    if progenitor not in uniq_grps:
        raise Exception(f"progenitor has to be in adata.obs[{group}], but you have {progenitor}. ")
    else:
        progenitor = list(uniq_grps).index(progenitor)

    if not set(terminators) <= set(uniq_grps):
        raise Exception(f"all terminators have to be in adata.obs[{group}], but you have {terminators}.")
    else:
        terminators = [list(uniq_grps).index(i) for i in terminators]

    if prune_graph:
        M = prune_transition(
            adata,
            group,
            basis,
            n_neighbors,
            neighbor_key,
            graph_mat,
            state_graph_method,
        )
    else:
        M = graph_mat

    if np.any(M < 0):
        main_warning("the transition graph have negative values.")
        M[M < 0] = 0
        M += 1e-5 - 1e-5  # ensure no -0 values existed

    if row_norm:
        M /= M.sum(1)

    M[M > 0] = 1 - M[M > 0]  # because it is shortest path, so we need to use 1 - M[M > 0]

    D, Pr = shortest_path(np.copy(M, order="c"), directed=False, method="FW", return_predecessors=True)
    res = np.zeros(M.shape)

    # this builds the tree based on each shortest path connecting the source to each target cell type
    main_info("builds the tree model based on each shortest path connecting the source to each target cell type in g'.")
    for j in terminators:
        p = j
        while Pr[progenitor, p] != -9999:
            res[Pr[progenitor, p], p] = 1
            p = Pr[progenitor, p]
    res = pd.DataFrame(res, index=uniq_grps, columns=uniq_grps)

    logger.finish_progress(progress_name="tree_model building")

    return res
