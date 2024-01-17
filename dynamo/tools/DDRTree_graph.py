from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from .DDRTree import cal_ncenter, DDRTree
from ..dynamo_logger import main_info, main_info_insert_adata_uns


def construct_velocity_tree(adata: AnnData, transition_matrix_key: str = "pearson"):
    """Integrate pseudotime ordering with velocity to automatically assign the direction of the learned trajectory.

    Args:
        adata: The anndata object containing the single-cell data.
        transition_matrix_key (str, optional): Key to the transition matrix in the `adata.obsp` object that represents
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
    directed_velocity_tree = velocity_tree.copy()

    segments = _get_all_segments(orders, parents)
    center_transition_matrix = _compute_center_transition_matrix(transition_matrix, R)

    for segment in segments:
        edge_pairs = _get_edges(segment)
        edge_pairs_reversed = _get_edges(segment[::-1])
        segment_p = _calculate_segment_probability(center_transition_matrix, edge_pairs)
        segment_p_reveresed = _calculate_segment_probability(center_transition_matrix, edge_pairs_reversed)
        if segment_p[-1] > segment_p_reveresed[-1]:
            for i, (r, c) in enumerate(edge_pairs):
                directed_velocity_tree[r, c] = max(velocity_tree[r, c], velocity_tree[c, r])
                directed_velocity_tree[c, r] = 0
        elif segment_p[-1] < segment_p_reveresed[-1]:
            for i, (r, c) in enumerate(edge_pairs):
                directed_velocity_tree[c, r] = max(velocity_tree[r, c], velocity_tree[c, r])
                directed_velocity_tree[r, c] = 0
        else:
            for i, (r, c) in enumerate(edge_pairs):
                directed_velocity_tree[c, r] = velocity_tree[c, r]
                directed_velocity_tree[r, c] = velocity_tree[r, c]

    adata.uns["directed_velocity_tree"] = directed_velocity_tree
    main_info_insert_adata_uns("directed_velocity_tree")
    return directed_velocity_tree


def directed_pg(
    adata: AnnData,
    basis: str = "umap",
    transition_method: str = "pearson",
    maxIter: int = 10,
    sigma: float = 0.001,
    Lambda: Optional[float] = None,
    gamma: float = 10,
    ncenter: Optional[int] = None,
    raw_embedding: bool = True,
) -> AnnData:
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


def _compute_center_transition_matrix(transition_matrix: Union[csr_matrix, np.ndarray], R: np.ndarray) -> np.ndarray:
    """Calculate the transition matrix for DDRTree centers.

    Args:
        transition_matrix: The array representing the transition matrix of cells.
        R: The matrix that assigns cells to the centers.

    Returns:
        The transition matrix for centers.
    """
    if issparse(transition_matrix):
        transition_matrix = transition_matrix.toarray()

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
        transition_matrix: The transition matrix for DDRTree centers.
        segments: The segments of the minimum spanning tree.

    Returns:
        The probability for each segment.
    """

    with np.errstate(divide='ignore', invalid='ignore'):
        log_transition_matrix = np.log1p(transition_matrix)
        log_transition_matrix[np.isinf(log_transition_matrix)] = 0
        log_transition_matrix = np.nan_to_num(log_transition_matrix)

    return np.cumsum(log_transition_matrix[[s[0] for s in segments], [s[1] for s in segments]])


def _get_edges(orders: Union[np.ndarray, List], parents: Optional[Union[np.ndarray, List]] = None) -> Tuple:
    """Get m segments pairs from the minimum spanning tree.

    Args:
        orders: The order to traverse the minimum spanning tree.
        parents: The parent node for each node. If not provided, will construct the segments with orders[i-1] and
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
) -> List:
    """Get the path from the start node to the end node.

    Args:
        parents_dict: The dictionary that maps each node to its parent node.
        start: The start node.
        end_nodes: The end nodes.

    Returns:
        The path from the start node to the end node.
    """
    if parents_dict[start] == -1:
        return None
    cur = parents_dict[start]
    path = [start, parents_dict[start]]
    while cur not in end_nodes:
        cur = parents_dict[cur]
        path.append(cur)
    return path


def _get_all_segments(orders: Union[np.ndarray, List], parents: Union[np.ndarray, List]) -> List:
    """Get all segments from the minimum spanning tree.

    Segments is defined as a path without any bifurcations.

    Args:
        orders: The order to traverse the minimum spanning tree.
        parents: The parent node for each node.

    Returns:
        A list of segments.
    """
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
