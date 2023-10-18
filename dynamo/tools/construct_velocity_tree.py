import re
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy
from anndata import AnnData
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.csgraph import shortest_path

from .DDRTree_py import DDRTree

from ..dynamo_logger import main_info, main_info_insert_adata_uns


def remove_velocity_points(G: np.ndarray, n: int) -> np.ndarray:
    """Modify a tree graph to remove the nodes themselves and recalculate the weights.

    Args:
        G: a smooth tree graph embedded in the low dimension space.
        n: the number of genes (column num of the original data)

    Returns:
        The tree graph with a node itself removed and weight recalculated.
    """
    for nodeid in range(n, 2 * n):
        nb_ids = []
        for nb_id in range(len(G[0])):
            if G[nodeid][nb_id] != 0:
                nb_ids = nb_ids + [nb_id]
        num_nbs = len(nb_ids)

        if num_nbs == 1:
            G[nodeid][nb_ids[0]] = 0
            G[nb_ids[0]][nodeid] = 0
        else:
            min_val = np.inf
            for i in range(len(G[0])):
                if G[nodeid][i] != 0:
                    if G[nodeid][i] < min_val:
                        min_val = G[nodeid][i]
                        min_ind = i
            for i in nb_ids:
                if i != min_ind:
                    new_weight = G[nodeid][i] + min_val
                    G[i][min_ind] = new_weight
                    G[min_ind][i] = new_weight
            # print('Add ege %s, %s\n',G.Nodes.Name {nb_ids(i)}, G.Nodes.Name {nb_ids(min_ind)});
            G[nodeid][nb_ids[0]] = 0
            G[nb_ids[0]][nodeid] = 0

    return G


def calculate_angle(o: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    """Calculate the angle between two vectors.

    Args:
        o: coordination of the origin.
        y: end point of the first vector.
        x: end point of the second vector.

    Returns:
        The angle between the two vectors.
    """

    yo = y - o
    norm_yo = yo / scipy.linalg.norm(yo)
    xo = x - o
    norm_xo = xo / scipy.linalg.norm(xo)
    angle = np.arccos(norm_yo.T * norm_xo)
    return angle


def _compute_transition_matrix(transition_matrix: Union[csr_matrix, np.ndarray], R: np.ndarray) -> np.ndarray:
    """Calculate the transition matrix for DDRTree centers.

    Args:
        transition_matrix: the array representing the transition matrix of cells.
        R: the matrix that assigns cells to the centers.

    Returns:
        The transition matrix for centers.
    """
    if issparse(transition_matrix):
        transition_matrix = transition_matrix.toarray()

    highest_probability = np.max(R, axis=1)
    assignment = np.argmax(R, axis=1)
    num_clusters = R.shape[1]
    clusters = {i: np.where(assignment == i)[0] for i in range(num_clusters)}

    transition = np.zeros((num_clusters, num_clusters))
    totals = np.zeros((num_clusters,))

    for a in range(num_clusters):
        for b in range(num_clusters):
            if a == b:
                continue
            indices_a = clusters[a]
            indices_b = clusters[b]
            q = np.sum(
                R[indices_a, a][:, np.newaxis] *
                R[indices_b, b].T[np.newaxis, :] *
                transition_matrix[indices_a[:, None], indices_b]
            ) if (indices_a.shape[0] > 0 and indices_b.shape[0] > 0) else 0
            totals[a] += q
            transition[a, b] = q

    totals = totals.reshape(-1, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = transition / totals
        res[np.isinf(res)] = 0
        res = np.nan_to_num(res)
    return res


def _calculate_segment_probability(transition_matrix: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Calculate the probability of	the	segment	by first order Markov assumption.

    Args:
        transition_matrix: the transition matrix for DDRTree centers.
        segments: the segments of the minimum spanning tree.

    Returns:
        The probability for each segment.
    """
    transition_matrix = transition_matrix.toarray() if issparse(transition_matrix) else transition_matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        log_transition_matrix = np.log(transition_matrix)
        log_transition_matrix[np.isinf(log_transition_matrix)] = 0
        log_transition_matrix = np.nan_to_num(log_transition_matrix)

    return np.cumsum(log_transition_matrix[[s[0] for s in segments], [s[1] for s in segments]])


def _get_edges(orders: Union[np.ndarray, List], parents: Optional[Union[np.ndarray, List]] = None) -> Tuple:
    """Get m segments pairs from the minimum spanning tree.

    Args:
        orders: the order to traverse the minimum spanning tree.
        parents: the parent node for each node. If not provided, will construct the segments with orders[i-1] and
            orders[i].

    Returns:
        A tuple that contains segments pairs from 1 to m and from m to 1.
    """
    if parents:
        segments = [(p, o) for p, o in zip(parents, orders) if p != -1]
    else:
        segments = [(orders[i-1], orders[i]) for i in range(1, len(orders))]
    return segments


def _get_path(
    parents_dict: Dict,
    start: int,
    end_nodes: List,
):
    if parents_dict[start] == -1:
        return None
    cur = parents_dict[start]
    path = [start, parents_dict[start]]
    while cur not in end_nodes:
        cur = parents_dict[cur]
        path.append(cur)
    return path


def _get_all_segments(orders: Union[np.ndarray, List], parents: Union[np.ndarray, List]):
    from collections import Counter

    leaf_nodes = [node for node in orders if node not in parents]

    if len(leaf_nodes) == 1:
        return [orders]

    parents_dict = {}
    for child, parent in zip(orders, parents):
        parents_dict[child] = parent

    element_counts = Counter(parents)
    bifurcation_nodes = [
        node for node, count in element_counts.items()
        if count > 1 and node != -1 and not (count == 2 and parents_dict == -1)
    ]
    root_nodes = [node for node in orders if parents_dict[node] == -1]
    start_nodes = leaf_nodes + bifurcation_nodes
    end_nodes = bifurcation_nodes
    for node in root_nodes:
        if node not in bifurcation_nodes:
            end_nodes.append(node)

    parents_dict = {}
    for child, parent in zip(orders, parents):
        parents_dict[child] = parent

    segments = []
    for node in start_nodes:
        path = _get_path(parents_dict=parents_dict, start=node, end_nodes=end_nodes)
        if path is not None:
            segments.append(path)

    return segments


def construct_velocity_tree(adata: AnnData, transition_matrix_key: str = "pearson"):
    """Integrate pseudotime ordering with velocity to automatically assign the direction of the learned trajectory.

    Args:
        adata: the anndata object containing the single-cell data.
        transition_matrix_key (str, optional): key to the transition matrix in the `adata.obsp` object that represents
            the transition probabilities between cells. Defaults to "pearson".

    Raises:
        KeyError: If the transition matrix or cell order information is not found in the `adata` object.

    Returns:
        A directed velocity tree represented as a NumPy array.
    """
    if transition_matrix_key + "_transition_matrix" not in adata.obsp.keys():
        raise KeyError("Transition matrix not found in anndata. Please call cell_velocities() before constructing "
                       "velocity tree")

    if "cell_order" not in adata.uns.keys():
        raise KeyError("Cell order information not found in anndata. Please call order_cells() before constructing "
                       "velocity tree.")

    main_info("Constructing velocity tree...")

    transition_matrix = adata.obsp[transition_matrix_key + "_transition_matrix"]
    R = adata.uns["cell_order"]["R"]
    orders = np.argsort(adata.uns["cell_order"]["centers_order"])
    parents = [adata.uns["cell_order"]["centers_parent"][node] for node in orders]
    velocity_tree = adata.uns["cell_order"]["centers_minSpanningTree"]
    cell_proj_closest_vertex = adata.uns["cell_order"]["pr_graph_cell_proj_closest_vertex"]
    directed_velocity_tree = velocity_tree.copy()

    segments = _get_all_segments(orders, parents)
    center_transition_matrix = _compute_transition_matrix(transition_matrix, R)

    for segment in segments:
        edge_pairs = _get_edges(segment)
        edge_pairs_reversed = _get_edges(segment[::-1])
        segment_p = _calculate_segment_probability(center_transition_matrix, edge_pairs)
        segment_p_reveresed = _calculate_segment_probability(center_transition_matrix, edge_pairs_reversed)
        if segment_p[-1] >= segment_p_reveresed[-1]:
            for i, (r, c) in enumerate(edge_pairs):
                directed_velocity_tree[r, c] = max(velocity_tree[r, c], velocity_tree[c, r])
                directed_velocity_tree[c, r] = 0
        else:
            for i, (r, c) in enumerate(edge_pairs):
                directed_velocity_tree[c, r] = max(velocity_tree[r, c], velocity_tree[c, r])
                directed_velocity_tree[r, c] = 0

    adata.uns["directed_velocity_tree"] = velocity_tree
    main_info_insert_adata_uns("directed_velocity_tree")
    return velocity_tree


def construct_velocity_tree_py(X1: np.ndarray, X2: np.ndarray) -> None:
    """Save a velocity tree graph with given data.

    Args:
        X1: epxression matrix.
        X2: velocity matrix.
    """
    if issparse(X1):
        X1 = X1.toarray()
    if issparse(X2):
        X2 = X2.toarray()
    n = X1.shape[1]

    # merge two data with a given time
    t = 0.5
    X_all = np.hstack((X1, X1 + t * X2))

    # parameter settings
    maxIter = 20
    eps = 1e-3
    sigma = 0.001
    gamma = 10

    # run DDRTree algorithm
    Z, Y, stree, R, W, Q, C, objs = DDRTree(X_all, maxIter=maxIter, eps=eps, sigma=sigma, gamma=gamma)

    # draw velocity figure

    # quiver(Z(1, 1: 100), Z(2, 1: 100), Z(1, 101: 200)-Z(1, 1: 100), Z(2, 101: 200)-Z(2, 1: 100));
    # plot(Z(1, 1: 100), Z(2, 1: 100), 'ob');
    # plot(Z(1, 101: 200), Z(2, 101: 200), 'sr');
    G = stree

    sG = remove_velocity_points(G, n)
    tree = sG
    row = []
    col = []
    val = []
    for i in range(sG.shape[0]):
        for j in range(sG.shape[1]):
            if sG[i][j] != 0:
                row = row + [i]
                col = col + [j]
                val = val + [sG[1][j]]
    tree_fname = "tree.csv"
    # write sG data to tree.csv
    #######
    branch_fname = "branch.txt"
    cmd = "python extract_branches.py" + tree_fname + branch_fname

    branch_cell = []
    fid = open(branch_fname, "r")
    tline = next(fid)
    while isinstance(tline, str):
        path = re.regexp(tline, "\d*", "Match")  ############
        branch_cell = branch_cell + [path]  #################
        tline = next(fid)
    fid.close()

    dG = np.zeros((n, n))
    for p in range(len(branch_cell)):
        path = branch_cell[p]
        pos_direct = 0
        for bp in range(len(path)):
            u = path(bp)
            v = u + n

            # find the shorest path on graph G(works for trees)
            nodeid = u
            ve_nodeid = v
            shortest_mat = shortest_path(
                csgraph=G,
                directed=False,
                indices=nodeid,
                return_predecessors=True,
            )
            velocity_path = []
            while ve_nodeid != nodeid:
                velocity_path = [shortest_mat[nodeid][ve_nodeid]] + velocity_path
                ve_nodeid = shortest_mat[nodeid][ve_nodeid]
            velocity_path = [shortest_mat[nodeid][ve_nodeid]] + velocity_path
            ###v_path = G.Nodes.Name(velocity_path)

            # check direction consistency between path and v_path
            valid_idx = []
            for i in velocity_path:
                if i <= n:
                    valid_idx = valid_idx + [i]
            if len(valid_idx) == 1:
                # compute direction matching
                if bp < len(path):
                    tree_next_point = Z[:, path(bp)]
                    v_point = Z[:, v]
                    u_point = Z[:, u]
                    angle = calculate_angle(u_point, tree_next_point, v_point)
                    angle = angle / 3.14 * 180
                    if angle < 90:
                        pos_direct = pos_direct + 1

                else:
                    tree_pre_point = Z[:, path(bp - 1)]
                    v_point = Z[:, v]
                    u_point = Z[:, u]
                    angle = calculate_angle(u_point, tree_pre_point, v_point)
                    angle = angle / 3.14 * 180
                    if angle > 90:
                        pos_direct = pos_direct + 1

            else:

                if bp < len(path):
                    if path[bp + 1] == valid_idx[2]:
                        pos_direct = pos_direct + 1

                else:
                    if path[bp - 1] != valid_idx[2]:
                        pos_direct = pos_direct + 1

        neg_direct = len(path) - pos_direct
        print(
            "branch="
            + str(p)
            + ", ("
            + path[0]
            + "->"
            + path[-1]
            + "), pos="
            + pos_direct
            + ", neg="
            + neg_direct
            + "\n"
        )
        print(path)
        print("\n")

        if pos_direct > neg_direct:
            for bp in range(len(path) - 1):
                dG[path[bp], path[bp + 1]] = 1

        else:
            for bp in range(len(path) - 1):
                dG[path(bp + 1), path(bp)] = 1

    # figure;
    # plot(digraph(dG));
    # title('directed graph') figure; hold on;
    row = []
    col = []
    for i in range(dG.shape[0]):
        for j in range(dG.shape[1]):
            if dG[i][j] != 0:
                row = row + [i]
                col = col + [j]
    for tn in range(len(row)):
        p1 = Y[:, row[tn]]
        p2 = Y[:, col[tn]]
        dp = p2 - p1
        h = plt.quiver(p1(1), p1(2), dp(1), dp(2), "LineWidth", 5)  ###############need to plot it
        set(h, "MaxHeadSize", 1e3, "AutoScaleFactor", 1)  #############

    for i in range(n):
        plt.text(Y(1, i), Y(2, i), str(i))  ##############
    plt.savefig("./results/t01_figure3.fig")  ##################
