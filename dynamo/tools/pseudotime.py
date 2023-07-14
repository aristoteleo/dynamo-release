from typing import Optional

import anndata
import numpy as np
import pandas as pd
from scipy.spatial import distance

from .DDRTree_py import DDRTree
from .utils import log1p_


def _find_nearest_vertex(data_matrix, target_points, block_size=50000, process_targets_in_blocks=False):
    closest_vertex = np.array([])

    if not process_targets_in_blocks:
        num_blocks = np.ceil(data_matrix.shape[1] / block_size).astype(int)
        if num_blocks < 1:
            print('bad loop: num_blocks < 1')
        for i in range(1, num_blocks + 1):
            if i < num_blocks:
                block = data_matrix[:, ((i - 1) * block_size):(i * block_size)]
            else:
                block = data_matrix[:, ((i - 1) * block_size):]
            distances_Z_to_Y = distance.cdist(block.T, target_points.T)
            closest_vertex_for_block = np.argmin(distances_Z_to_Y, axis=1)
            closest_vertex = np.append(closest_vertex, closest_vertex_for_block)
    else:
        num_blocks = np.ceil(target_points.shape[1] / block_size).astype(int)
        dist_to_closest_vertex = np.full(data_matrix.shape[1], np.inf)
        closest_vertex = np.full(data_matrix.shape[1], np.nan)
        if num_blocks < 1:
            print('bad loop: num_blocks < 1')
        for i in range(1, num_blocks + 1):
            if i < num_blocks:
                block = target_points[:, ((i - 1) * block_size):(i * block_size)]
            else:
                block = target_points[:, ((i - 1) * block_size):]
            distances_Z_to_Y = distance.cdist(data_matrix.T, block.T)
            closest_vertex_for_block = np.argmin(distances_Z_to_Y, axis=1)
            if distances_Z_to_Y.shape[0] < 1:
                print('bad loop: nrow(distances_Z_to_Y) < 1')
            new_block_distances = distances_Z_to_Y[np.arange(distances_Z_to_Y.shape[0]), closest_vertex_for_block]
            updated_nearest_idx = np.where(new_block_distances < dist_to_closest_vertex)
            closest_vertex[updated_nearest_idx] = closest_vertex_for_block[updated_nearest_idx] + (i - 1) * block_size
            dist_to_closest_vertex[updated_nearest_idx] = new_block_distances[updated_nearest_idx]

    assert len(closest_vertex) == data_matrix.shape[1], "length(closest_vertex) != ncol(data_matrix)"
    return closest_vertex


def _check_and_replace_nan_weights(graph):
    weights = np.array(graph.es['weight'])
    nan_indices = np.isnan(weights)
    weights[nan_indices] = 1
    graph.es['weight'] = weights
    return graph


def get_order_from_DDRTree(adata, principal_graph, root_cell):
    import igraph as ig
    from scipy.sparse.csgraph import minimum_spanning_tree

    dp = adata.obsp["distances"]
    mst = minimum_spanning_tree(principal_graph)
    # n = mst.shape[0]
    # edges = [(i, j) for i in range(n) for j in range(n) if mst[i, j] != 0]
    # weights = mst[np.nonzero(mst)]
    dp_mst = ig.Graph.Weighted_Adjacency(matrix=mst, attr='weight', loops=False)
    # dp_mst.es['weight'] = weights
    curr_state = 0
    states = [0 for _ in range(dp.shape[1])]
    pseudotimes = [0 for _ in range(dp.shape[1])]
    parents = [None for _ in range(dp.shape[1])]
    ordering_df = pd.DataFrame(columns=['cell_index', 'cell_pseudo_state', 'pseudo_time', 'parent'])

    orders, pres = dp_mst.dfs(root=root_cell, mode=ig.ALL, unreachable=False, return_fathers=True)
    pres = pres.astype(int)

    for i in range(len(orders)):
        curr_node = orders[i]

        if pd.notna(pres[curr_node]):
            parent_node = pres[curr_node]
            parent_node_pseudotime = pseudotimes[parent_node]
            # parent_node_state = states[parent_node]
            curr_node_pseudotime = parent_node_pseudotime + dp[curr_node, parent_node]

            if dp_mst.degree(parent_node) > 2:
                curr_state += 1
        else:
            parent_node = None
            curr_node_pseudotime = 0

        pseudotimes[curr_node] = curr_node_pseudotime
        states[curr_node] = curr_state
        parents[curr_node] = parent_node

        ordering_df = ordering_df.append({
            'cell_index': curr_node,
            'cell_pseudo_state': states[curr_node],
            'pseudo_time': pseudotimes[curr_node],
            'parent': parent_node
        }, ignore_index=True)

    ordering_df.set_index('cell_index', inplace=True)
    return ordering_df


def Pseudotime(
    adata: anndata.AnnData, layer: str = "X", basis: Optional[str] = None, method: str = "DDRTree", **kwargs
) -> anndata.AnnData:

    """

    Parameters
    ----------
    adata
    layer
    method
    kwargs

    Returns
    -------

    """

    if basis is None:
        X = adata.layers["X_" + layer].T if layer != "X" else adata.X.T
        X = log1p_(adata, X)
    else:
        X = adata.obsm["X_" + basis]

    DDRTree_kwargs = {
        "maxIter": 10,
        "sigma": 0.001,
        "gamma": 10,
        "eps": 0,
        "dim": 2,
        "Lambda": 5 * X.shape[1],
        "ncenter": _cal_ncenter(X.shape[1]),
    }
    DDRTree_kwargs.update(kwargs)

    if method == "DDRTree":
        Z, Y, stree, R, W, Q, C, objs = DDRTree(X, **DDRTree_kwargs)

    transition_matrix, cell_membership, principal_g = (
        adata.uns["transition_matrix"],
        R,
        stree,
    )

    final_g = compute_partition(transition_matrix, cell_membership, principal_g)
    direct_g = final_g.copy()
    tmp = final_g - final_g.T
    direct_g[np.where(tmp < 0)] = 0

    adata.uns["directed_principal_g"] = direct_g

    # identify branch points and tip points

    # calculate net flow between branchpoints and tip points

    # assign direction of flow and reset direct_g

    return adata


def _cal_ncenter(ncells, ncells_limit=100):
    if ncells <= ncells_limit:
        return None
    else:
        return np.round(2 * ncells_limit * np.log(ncells) / (np.log(ncells) + np.log(ncells_limit)))


# make this function to also calculate the directed graph between clusters:
def compute_partition(adata, transition_matrix, cell_membership, principal_g, group=None):
    """

    Parameters
    ----------
    transition_matrix
    cell_membership
    principal_g

    Returns
    -------

    """

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    # http://active-analytics.com/blog/rvspythonwhyrisstillthekingofstatisticalcomputing/
    if group is not None and group in adata.obs.columns:
        from patsy import dmatrix  # dmatrices, dmatrix, demo_data

        data = adata.obs
        data.columns[data.columns == group] = "group_"

        cell_membership = csr_matrix(dmatrix("~group_+0", data=data))

    X = csr_matrix(principal_g > 0)
    Tcsr = minimum_spanning_tree(X)
    principal_g = Tcsr.toarray().astype(int)

    membership_matrix = cell_membership.T.dot(transition_matrix).dot(cell_membership)

    direct_principal_g = principal_g * membership_matrix

    # get the data:
    # edges_per_module < - Matrix::rowSums(num_links)
    # total_edges < - sum(num_links)
    #
    # theta < - (as.matrix(edges_per_module) / total_edges) % * %
    # Matrix::t(edges_per_module / total_edges)
    #
    # var_null_num_links < - theta * (1 - theta) / total_edges
    # num_links_ij < - num_links / total_edges - theta
    # cluster_mat < - pnorm_over_mat(as.matrix(num_links_ij), var_null_num_links)
    #
    # num_links < - num_links_ij / total_edges
    #
    # cluster_mat < - matrix(stats::p.adjust(cluster_mat),
    #                               nrow = length(louvain_modules),
    #                                      ncol = length(louvain_modules))
    #
    # sig_links < - as.matrix(num_links)
    # sig_links[cluster_mat > qval_thresh] = 0
    # diag(sig_links) < - 0

    return direct_principal_g


# if __name__ == '__main__':
#     # import anndata
#     # adata=anndata.read_h5ad('')
#     # Pseudotime(adata)
