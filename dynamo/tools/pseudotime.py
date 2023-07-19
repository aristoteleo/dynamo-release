from typing import Callable, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.sparse.csgraph import minimum_spanning_tree

from .DDRTree_py import DDRTree
from .utils import log1p_

from ..dynamo_logger import main_info, main_info_insert_adata_obs


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


def get_order_from_DDRTree(dp: np.ndarray, mst: np.ndarray, root_cell: int) -> pd.DataFrame:
    """Calculates the order of cells based on a minimum spanning tree and a distance matrix.

    Args:
        dp: the distance matrix representing the pairwise distances between cells.
        mst: the minimum spanning tree matrix.
        root_cell: the index of the root cell.

    Returns:
        A pandas DataFrame containing the cell ordering information.
    """
    import igraph as ig

    dp_mst = ig.Graph.Weighted_Adjacency(matrix=mst)
    curr_state = 0
    pseudotimes = [0 for _ in range(dp.shape[1])]
    ordering_dict = {
        'cell_index': [],
        'cell_pseudo_state': [],
        'pseudo_time': [],
        'parent': []
    }

    orders, pres = dp_mst.dfs(vid=root_cell, mode="all")

    for i in range(len(orders)):
        curr_node = orders[i]

        if pres[i] > 0:
            parent_node = pres[i]
            parent_node_pseudotime = pseudotimes[parent_node]
            curr_node_pseudotime = parent_node_pseudotime + dp[curr_node, parent_node]

            if dp_mst.degree(parent_node) > 2:
                curr_state += 1
        else:
            parent_node = -1
            curr_node_pseudotime = 0

        pseudotimes[curr_node] = curr_node_pseudotime

        ordering_dict["cell_index"].append(curr_node)
        ordering_dict["cell_pseudo_state"].append(curr_state)
        ordering_dict["pseudo_time"].append(pseudotimes[curr_node])
        ordering_dict["parent"].append(parent_node)

    ordering_df = pd.DataFrame.from_dict(ordering_dict)
    ordering_df.reset_index(inplace=True)
    ordering_df = ordering_df.rename(columns={'index': 'orders'})
    ordering_df.set_index('cell_index', inplace=True)
    ordering_df = ordering_df.sort_index()
    return ordering_df


def find_cell_proj_closest_vertex(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Find the closest vertex of each cell's projection to the nearest in the graph.

    Args:
        Z: a matrix representing cell projection points.
        Y: a matrix representing target points in the graph.

    Returns:
        Array of indices indicating the closest vertex.
    """
    distances_Z_to_Y = distance.cdist(Z.T, Y.T)
    return np.apply_along_axis(lambda z: np.where(z == np.min(z))[0][0], axis=1, arr=distances_Z_to_Y)

def project_point_to_line_segment(p: np.ndarray, df: np.ndarray) -> np.ndarray:
    """Optimized version of `project_point_to_line`, which projects a point onto a line segment defined by two points A
    and B.

    Args:
        p: The point to be projected onto the line segment.
        df: An array representing the line segment.

    Returns:
        np.ndarray: The closest point on the line segment to the given point.
    """
    # Returns q, the closest point to p on the line segment from A to B
    A = df[:, 0]
    B = df[:, 1]
    # Vector from A to B
    AB = B - A
    # Squared distance from A to B
    AB_squared = np.sum(AB**2)

    if AB_squared == 0:
        # A and B are the same point
        q = A
    else:
        # Vector from A to p
        Ap = p - A
        # Calculate the projection parameter t
        t = np.dot(Ap, AB) / AB_squared

        if t < 0.0:
            # "Before" A on the line, just return A
            q = A
        elif t > 1.0:
            # "After" B on the line, just return B
            q = B
        else:
            # Projection lies "inbetween" A and B on the line
            q = A + t * AB

    return q


def proj_point_on_line(point: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Default method to project a point onto a line defined by two points.

    Args:
        point: the point to be projected onto the line.
        line: an array representing the line.

    Returns:
        The projected point on the line.
    """
    ap = point - line[:, 0]
    ab = line[:, 1] - line[:, 0]

    res = line[:, 0] + (np.dot(ap, ab) / np.dot(ab, ab)) * ab
    return res


def project2MST(mst: np.ndarray, Z: np.ndarray, Y: np.ndarray, Projection_Method: Callable) -> Tuple:
    """Project cell projection points onto the minimum spanning tree (MST) of the principal graph.

    Args:
        mst: the adjacency matrix representing the minimum spanning tree (MST).
        Z: the matrix representing points to project.
        Y: the matrix representing target centers on the principal graph.
        Projection_Method: callable function for projection method.

    Returns:
        A tuple containing the following elements:
            - cellPairwiseDistances: Pairwise distances between cells after projection.
            - P: Projected cell points.
            - closest_vertex: Array of indices indicating the closest vertex for each cell.
            - dp_mst: Minimum spanning tree (MST) of the projected cell points.
    """
    import igraph as ig

    closest_vertex = find_cell_proj_closest_vertex(Z=Z, Y=Y)

    dp_mst = ig.Graph.Weighted_Adjacency(matrix=mst)
    tip_leaves = [v.index for v in dp_mst.vs.select(_degree_eq=1)]

    if not callable(Projection_Method):
        P = Y.iloc[:, closest_vertex]
    else:
        P = np.zeros_like(Z)  # Initialize P with zeros
        for i in range(len(closest_vertex)):
            neighbors = dp_mst.neighbors(closest_vertex[i], mode="all")
            projection = None
            dist = []
            Z_i = Z[:, i]

            for neighbor in neighbors:
                if closest_vertex[i] in tip_leaves:
                    tmp = proj_point_on_line(Z_i, Y[:, [closest_vertex[i], neighbor]])
                else:
                    tmp = Projection_Method(Z_i, Y[:, [closest_vertex[i], neighbor]])
                projection = np.vstack((projection, tmp)) if projection is not None else tmp[None, :]
                dist.append(distance.euclidean(Z_i, tmp))

            P[:, i] = projection[np.where(dist == np.min(dist))[0][0], :]

    dp = distance.squareform(distance.pdist(P.T))
    min_dist = np.min(dp[np.nonzero(dp)])
    dp += min_dist
    np.fill_diagonal(dp, 0)

    cellPairwiseDistances = dp
    dp_mst = minimum_spanning_tree(dp)
    # gp = ig.Graph.Weighted_Adjacency(matrix=dp, mode="undirected")
    # dp_mst = gp.spanning_tree(weights=gp.es['weight'])

    return cellPairwiseDistances, P, closest_vertex, dp_mst


def select_root_cell(
    adata: anndata.AnnData,
    Z: np.ndarray,
    root_state: Optional[int] = None,
    reverse: bool = False,
) -> int:
    """Selects the root cell for ordering based on the diameter of the minimum spanning tree, with the option to specify
    a root state as an additional constraint.

    Args:
        adata: the anndata object.
        Z: a matrix representing cell projection points.
        root_state: the specific state for selecting the root cell.
        reverse: whether to reverse the selection of the root cell.

    Raises:
        ValueError: If the state has not yet been set or if there are no cells for the specified state.
        ValueError: If no spanning tree is found for the object.

    Returns:
        The index of the selected root cell.
    """
    import igraph as ig

    if root_state is not None:
        if 'cell_pseudo_state' not in adata.obs.keys():
            raise ValueError("State has not yet been set. Please call order_cells() without specifying root_state.")

        root_cell_candidates = np.where(adata.obs["cell_pseudo_state"] == root_state)[0]
        if root_cell_candidates.shape[0] == 0:
            raise ValueError("No cells for State =", root_state)

        reduced_dim_subset = Z[:, root_cell_candidates].T
        dp = distance.cdist(reduced_dim_subset, reduced_dim_subset, metric="euclidean")
        gp = ig.Graph.Weighted_Adjacency(dp, mode="undirected")
        dp_mst = gp.spanning_tree(weights=gp.es['weight'])
        diameter = dp_mst.get_diameter(directed=False)

        if len(diameter) == 0:
            raise ValueError("No valid root cells for State =", root_state)

        root_cell_candidates = root_cell_candidates[diameter]
        if adata.uns['cell_order']['root_cell'] is not None and \
                adata.obs["cell_pseudo_state"][adata.uns['cell_order']['root_cell']] == root_state:
            root_cell = root_cell_candidates[np.argmin(adata[root_cell_candidates].obs['Pseudotime'].values)]
        else:
            root_cell = root_cell_candidates[np.argmax(adata[root_cell_candidates].obs['Pseudotime'].values)]
        if isinstance(root_cell, list):
            root_cell = root_cell[0]

        root_cell = adata.uns['cell_order']['pr_graph_cell_proj_closest_vertex'][root_cell]

    else:
        if 'minSpanningTree' not in adata.uns['cell_order'].keys():
            raise ValueError("No spanning tree found for adata object.")

        graph = ig.Graph.Weighted_Adjacency(adata.uns['cell_order']['minSpanningTree'], mode="undirected")
        diameter = graph.get_diameter(directed=False)
        if reverse:
            root_cell = diameter[-1]
        else:
            root_cell = diameter[0]

    return root_cell


def order_cells(
    adata: anndata.AnnData,
    layer: str = "X",
    basis: Optional[str] = None,
    root_state: Optional[int] = None,
    reverse: bool = False,
    maxIter: int = 10,
    sigma: float = 0.001,
    gamma: float = 10.0,
    eps: int = 0,
    dim: int = 2,
    Lambda: Optional[float] = None,
    ncenter: Optional[int] = None,
    **kwargs,
) -> anndata.AnnData:
    """Order the cells based on the calculated pseudotime derived from the principal graph.

    Learns a "trajectory" describing the biological process the cells are going through, and calculates where each cell
    falls within that trajectory. The trajectory will be composed of segments. The cells from a segment will share the
    same value of state. One of these segments will be selected as the root of the trajectory. The most distal cell on
    that segment will be chosen as the "first" cell in the trajectory, and will have a pseudotime value of zero. Then
    the function will then "walk" along the trajectory, and as it encounters additional cells, it will assign them
    increasingly large values of pseudotime based on distance.

    Args:
        adata: the anndata object.
        layer: the layer used to order the cells.
        basis: the basis that indicates the data after dimension reduction.
        root_state: the specific state for selecting the root cell.
        reverse: whether to reverse the selection of the root cell.
        maxIter: the max number of iterations.
        sigma: the bandwidth parameter.
        gamma: regularization parameter for k-means.
        eps: the threshold of convergency to stop the iteration. Defaults to 0.
        dim: the number of dimensions reduced to. Defaults to 2.
        Lambda: regularization parameter for inverse praph embedding. Defaults to 1.0.
        ncenter: the number of center genes to be considered. If None, all genes would be considered. Defaults to None.
        kwargs: additional keyword arguments.

    Returns:
        The anndata object updated with pseudotime, cell order state and other necessary information.
    """
    import igraph as ig

    main_info("Ordering cells based on pseudotime...")
    if basis is None:
        X = adata.layers["X_" + layer].T if layer != "X" else adata.X.T
        X = log1p_(adata, X)
    else:
        X = adata.obsm["X_" + basis]

    if "cell_order" not in adata.uns.keys():
        adata.uns["cell_order"] = {}
        adata.uns["cell_order"]["root_cell"] = None

    DDRTree_kwargs = {
        "maxIter": maxIter,
        "sigma": sigma,
        "gamma": gamma,
        "eps": eps,
        "dim": dim,
        "Lambda": Lambda if Lambda else 5 * X.shape[1],
        "ncenter": ncenter if ncenter else _cal_ncenter(X.shape[1]),
    }
    DDRTree_kwargs.update(kwargs)

    Z, Y, stree, R, W, Q, C, objs = DDRTree(X, **DDRTree_kwargs)
    adata.uns["cell_order"]["cell_order_method"] = "DDRTree"
    adata.uns["cell_order"]["Z"] = Z
    adata.uns["cell_order"]["Y"] = Y
    adata.uns["cell_order"]["stree"] = stree
    adata.uns["cell_order"]["R"] = R
    adata.uns["cell_order"]["W"] = W

    principal_graph = stree
    dp = distance.squareform(distance.pdist(Y.T))
    mst = minimum_spanning_tree(principal_graph)
    adata.uns["cell_order"]["minSpanningTree"] = mst
    adata.uns["cell_order"]["center_minSpanningTree"] = mst

    root_cell = select_root_cell(adata, Z=Z, root_state=root_state, reverse=reverse)
    cc_ordering = get_order_from_DDRTree(dp=dp, mst=mst, root_cell=root_cell)

    adata.uns["cell_order"]["root_cell"] = root_cell
    adata.uns["cell_order"]["centers_order"] = cc_ordering["orders"].values

    old_mst_graph = ig.Graph.Weighted_Adjacency(matrix=mst)
    cellPairwiseDistances, pr_graph_cell_proj_dist, pr_graph_cell_proj_closest_vertex, pr_graph_cell_proj_tree = project2MST(mst, Z, Y, project_point_to_line_segment)  # project_point_to_line_segment can be changed to other states

    adata.uns["cell_order"]["minSpanningTree"] = pr_graph_cell_proj_tree
    adata.uns["cell_order"]["pr_graph_cell_proj_closest_vertex"] = pr_graph_cell_proj_closest_vertex

    cells_mapped_to_graph_root = np.where(pr_graph_cell_proj_closest_vertex == root_cell)[0]
    # avoid the issue of multiple cells projected to the same point on the principal graph
    if len(cells_mapped_to_graph_root) == 0:
        cells_mapped_to_graph_root = root_cell

    pr_graph_cell_proj_tree_graph = ig.Graph.Weighted_Adjacency(matrix=pr_graph_cell_proj_tree)
    tip_leaves = [v.index for v in pr_graph_cell_proj_tree_graph.vs.select(_degree=1)]
    root_cell = cells_mapped_to_graph_root[np.isin(cells_mapped_to_graph_root, tip_leaves)][0]
    if np.isnan(root_cell):
        root_cell = select_root_cell(adata, Z=Z, root_state=root_state, reverse=reverse)
        adata.uns["cell_order"]["root_cell"] = root_cell

    adata.uns["cell_order"]["root_cell"] = root_cell

    cc_ordering_new_pseudotime = get_order_from_DDRTree(dp=cellPairwiseDistances, mst=pr_graph_cell_proj_tree, root_cell=root_cell)  # re-calculate the pseudotime again

    adata.obs["Pseudotime"] = cc_ordering_new_pseudotime["pseudo_time"].values
    adata.uns["cell_order"]["parent"] = cc_ordering_new_pseudotime["parent"]
    main_info_insert_adata_obs("Pseudotime")
    if root_state is None:
        closest_vertex = pr_graph_cell_proj_closest_vertex
        adata.obs["cell_pseudo_state"] = cc_ordering.loc[closest_vertex, "cell_pseudo_state"].values
        main_info_insert_adata_obs("cell_pseudo_state")

    adata.uns["cell_order"]["branch_points"] = np.array(pr_graph_cell_proj_tree_graph.vs.select(_degree_gt=2))
    return adata


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
    """Calculate the number of centers genes to be considered.

    Parameters:
        ncells: total number of cells.
        ncells_limit: upper limit of number of cells to be considered. Default is 100.

    Returns:
        Number of centers to use. Returns None if `ncells` is less than or equal to `ncells_limit`.
    """
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
