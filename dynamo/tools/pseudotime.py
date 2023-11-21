from typing import Callable, List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.sparse.csgraph import minimum_spanning_tree

from .DDRTree import DDRTree
from .utils import log1p_

from ..dynamo_logger import main_info, main_info_insert_adata_obs


def order_cells(
    adata: anndata.AnnData,
    layer: str = "X",
    basis: Optional[str] = None,
    root_state: Optional[int] = None,
    init_cells: Optional[Union[List, np.ndarray, pd.Index]] = None,
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
        adata: The anndata object.
        layer: The layer used to order the cells.
        basis: The basis that indicates the data after dimension reduction.
        root_state: The specific state for selecting the root cell.
        init_cells: The index to search for root cells. If provided, root_state will be ignored.
        reverse: Whether to reverse the selection of the root cell.
        maxIter: The max number of iterations.
        sigma: The bandwidth parameter.
        gamma: Regularization parameter for k-means.
        eps: The threshold of convergency to stop the iteration. Defaults to 0.
        dim: The number of dimensions reduced to. Defaults to 2.
        Lambda: Regularization parameter for inverse praph embedding. Defaults to 1.0.
        ncenter: The number of center genes to be considered. If None, all genes would be considered. Defaults to None.
        kwargs: Additional keyword arguments.

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

    principal_graph = stree
    dp = distance.squareform(distance.pdist(Y.T))
    mst = minimum_spanning_tree(principal_graph)

    adata.uns["cell_order"]["cell_order_method"] = "DDRTree"
    adata.uns["cell_order"]["Z"] = Z
    adata.uns["cell_order"]["Y"] = Y
    adata.uns["cell_order"]["stree"] = stree
    adata.uns["cell_order"]["R"] = R
    adata.uns["cell_order"]["W"] = W
    adata.uns["cell_order"]["minSpanningTree"] = mst
    adata.uns["cell_order"]["centers_minSpanningTree"] = mst

    root_cell = select_root_cell(adata, Z=Z, root_state=root_state, init_cells=init_cells, reverse=reverse)
    cc_ordering = get_order_from_DDRTree(dp=dp, mst=mst, root_cell=root_cell)

    (
        cellPairwiseDistances,
        pr_graph_cell_proj_dist,
        pr_graph_cell_proj_closest_vertex,
        pr_graph_cell_proj_tree
    ) = project2MST(mst, Z, Y, project_point_to_line_segment)

    adata.uns["cell_order"]["root_cell"] = root_cell
    adata.uns["cell_order"]["centers_order"] = cc_ordering["orders"].values
    adata.uns["cell_order"]["centers_parent"] = cc_ordering["parent"].values
    adata.uns["cell_order"]["minSpanningTree"] = pr_graph_cell_proj_tree
    adata.uns["cell_order"]["pr_graph_cell_proj_closest_vertex"] = pr_graph_cell_proj_closest_vertex

    cells_mapped_to_graph_root = np.where(pr_graph_cell_proj_closest_vertex == root_cell)[0]
    # avoid the issue of multiple cells projected to the same point on the principal graph
    if len(cells_mapped_to_graph_root) == 0:
        cells_mapped_to_graph_root = [root_cell]

    pr_graph_cell_proj_tree_graph = ig.Graph.Weighted_Adjacency(matrix=pr_graph_cell_proj_tree)
    tip_leaves = [v.index for v in pr_graph_cell_proj_tree_graph.vs.select(_degree=1)]
    root_cell_candidates = np.intersect1d(cells_mapped_to_graph_root, tip_leaves)

    if len(root_cell_candidates) == 0:
        root_cell = select_root_cell(adata, Z=Z, root_state=root_state, init_cells=init_cells, reverse=reverse, map_to_tree=False)
    else:
        root_cell = root_cell_candidates[0]

    cc_ordering_new_pseudotime = get_order_from_DDRTree(dp=cellPairwiseDistances, mst=pr_graph_cell_proj_tree, root_cell=root_cell)  # re-calculate the pseudotime again

    adata.uns["cell_order"]["root_cell"] = root_cell
    adata.obs["Pseudotime"] = cc_ordering_new_pseudotime["pseudo_time"].values
    adata.uns["cell_order"]["parent"] = cc_ordering_new_pseudotime["parent"]
    adata.uns["cell_order"]["branch_points"] = np.array(pr_graph_cell_proj_tree_graph.vs.select(_degree_gt=2))
    main_info_insert_adata_obs("Pseudotime")

    if root_state is None:
        closest_vertex = pr_graph_cell_proj_closest_vertex
        adata.obs["cell_pseudo_state"] = cc_ordering.loc[closest_vertex, "cell_pseudo_state"].values
        main_info_insert_adata_obs("cell_pseudo_state")

    return adata


def get_order_from_DDRTree(dp: np.ndarray, mst: np.ndarray, root_cell: int) -> pd.DataFrame:
    """Calculates the order of cells based on a minimum spanning tree and a distance matrix.

    Args:
        dp: The distance matrix representing the pairwise distances between cells.
        mst: The minimum spanning tree matrix.
        root_cell: The index of the root cell.

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

        if pres[i] >= 0:
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
        Z: A matrix representing cell projection points.
        Y: A matrix representing target points in the graph.

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
        point: The point to be projected onto the line.
        line: An array representing the line.

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
        mst: The adjacency matrix representing the minimum spanning tree (MST).
        Z: The matrix representing points to project.
        Y: The matrix representing target centers on the principal graph.
        Projection_Method: Callable function for projection method.

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
    init_cells: Optional[Union[List, np.ndarray, pd.Index]] = None,
    reverse: bool = False,
    map_to_tree: bool = True,
) -> int:
    """Selects the root cell for ordering based on the diameter of the minimum spanning tree, with the option to specify
    a root state as an additional constraint.

    Args:
        adata: The anndata object.
        Z: A matrix representing cell projection points.
        root_state: The specific state for selecting the root cell.
        init_cells: The index to search for root cells. If provided, root_state will be ignored.
        reverse: Whether to reverse the selection of the root cell.
        map_to_tree: Whether to map the root in all cells to the tree after dimension reduction.

    Raises:
        ValueError: If the state has not yet been set or if there are no cells for the specified state.
        ValueError: If no spanning tree is found for the object.

    Returns:
        The index of the selected root cell.
    """
    import igraph as ig

    if init_cells is not None:

        root_cell_candidates = [adata.obs_names.get_loc(init_cell) for init_cell in init_cells]
        reduced_dim_subset = Z[:, root_cell_candidates].T
        centroid = np.mean(reduced_dim_subset, axis=0)
        distances = np.linalg.norm(reduced_dim_subset - centroid, axis=1)
        index_of_closest_sample = np.argmin(distances)
        root_cell = root_cell_candidates[index_of_closest_sample]

        if map_to_tree:
            cell_proj_closest_vertex = np.argmax(adata.uns['cell_order']['R'], axis=1)
            root_cell = cell_proj_closest_vertex[root_cell]

    elif root_state is not None:
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

        if map_to_tree:
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


def _cal_ncenter(ncells, ncells_limit=100):
    """Calculate the number of centers genes to be considered.

    Parameters:
        ncells: Total number of cells.
        ncells_limit: Upper limit of number of cells to be considered. Default is 100.

    Returns:
        Number of centers to use. Returns None if `ncells` is less than or equal to `ncells_limit`.
    """
    if ncells <= ncells_limit:
        return None
    else:
        return np.round(2 * ncells_limit * np.log(ncells) / (np.log(ncells) + np.log(ncells_limit)))
