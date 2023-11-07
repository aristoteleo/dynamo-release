import math
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix

from .utils import get_color_map_from_labels, save_show_ret


def _calculate_cells_mapping(
    adata: AnnData,
    group_key: str,
    cell_proj_closest_vertex: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Calculate the distribution of cells in each node.

    Args:
        adata: the anndata object.
        group_key: the key to locate the groups of each cell in adata.
        cell_proj_closest_vertex: the mapping from each cell to the corresponding node.

    Returns:
        The size of each node, the percentage of each group in every node, and the color mapping of each group.
    """
    cells_mapping_size = np.bincount(cell_proj_closest_vertex)
    centroids_index = range(len(cells_mapping_size))

    cell_type_info = pd.DataFrame({
        "class": adata.obs[group_key].values,
        "centroid": cell_proj_closest_vertex,
    })

    cell_color_map = get_color_map_from_labels(adata.obs[group_key].values)

    cell_type_info = cell_type_info.groupby(['centroid', 'class']).size().unstack()
    cell_type_info = cell_type_info.reindex(centroids_index, fill_value=0)
    cells_mapping_percentage = cell_type_info.div(cells_mapping_size, axis=0)
    cells_mapping_percentage = np.nan_to_num(cells_mapping_percentage.values)

    cells_mapping_size = (cells_mapping_size / len(cell_proj_closest_vertex))
    cells_mapping_size = [0.05 if s < 0.05 else s for s in cells_mapping_size]

    return cells_mapping_size, cells_mapping_percentage, cell_color_map


def _scale_positions(positions: np.ndarray, variance_scale: int = 1.5) -> np.ndarray:
    """Scale an array representing to the matplotlib coordinates system and scale the variance if needed.

    Args:
        positions: the array representing the positions of the data to plot.
        variance_scale: the value to scale the variance of data.

    Returns:
        The positions after scaling.
    """
    min_value = np.min(positions)
    max_value = np.max(positions)
    pos = (positions - min_value) / (max_value - min_value)
    mean = np.mean(pos, axis=0)
    pos = (pos - mean) * variance_scale
    return pos


def plot_dim_reduced_direct_graph(
    adata: AnnData,
    group_key: Optional[str] = "Cell_type",
    graph: Optional[Union[csr_matrix, np.ndarray]] = None,
    cell_proj_closest_vertex: Optional[np.ndarray] = None,
    center_coordinates: Optional[np.ndarray] = None,
    display_piechart: bool = True,
    variance_scale: int = 1.5,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
) -> Optional[plt.Axes]:
    """Plot the directed graph constructed velocity-guided pseudotime.

    Args:
        adata: the anndata object.
        group_key:  the key to locate the groups of each cell in adata.
        graph: the directed graph to plot.
        cell_proj_closest_vertex: the mapping from each cell to the corresponding node.
        center_coordinates: the array representing the positions of the center nodes in the low dimensions. Only need
            this when display_piechart is True.
        display_piechart: whether to display piechart for each node.
        variance_scale: the value to scale the variance of data. This function is employed to space out the pie charts
            when they are positioned too closely to each other.
        save_show_or_return: whether to save, show or return the plot.
        save_kwargs: additional keyword arguments of plot saving.

    Returns:
        The plot of the directed graph or `None`.
    """

    try:
        if graph is None:
            graph = adata.uns["directed_velocity_tree"]

        if cell_proj_closest_vertex is None:
            cell_proj_closest_vertex = adata.uns["cell_order"]["pr_graph_cell_proj_closest_vertex"]
    except KeyError:
        raise KeyError("Cell order data is missing. Please run `tl.order_cells()` first!")

    cells_size, cells_percentage, cells_color_map = _calculate_cells_mapping(
        adata=adata,
        group_key=group_key,
        cell_proj_closest_vertex=cell_proj_closest_vertex,
    )

    cells_colors = np.array([v for v in cells_color_map.values()])

    fig, ax = plt.subplots(figsize=(6, 6))

    G = nx.from_numpy_array(graph, create_using=nx.DiGraph)

    center_coordinates = adata.uns["cell_order"]["Y"].T.copy() if center_coordinates is None else center_coordinates
    pos = _scale_positions(center_coordinates, variance_scale=variance_scale)
    pos_dict = {}
    for i in range(len(pos)):
        pos_dict[i] = pos[i]

    if display_piechart:

        for node in G.nodes:
            attributes = cells_percentage[node]

            if np.all(attributes == 0):
                plt.pie(
                    [1],
                    center=pos[node],
                    colors=[[0, 0, 0, 1]],
                    radius=cells_size[node],
                )
            else:
                valid_indices = np.where(attributes != 0)[0]
                plt.pie(
                    attributes[valid_indices],
                    center=pos[node],
                    colors=cells_colors[valid_indices],
                    radius=cells_size[node],
                )
        g = nx.draw_networkx_edges(
            G,
            pos=pos_dict,
            node_size=[s * len(cells_size) * 300 for s in cells_size],
            arrows=True,
            arrowstyle="->",
            arrowsize=20,
            ax=ax,
        )

    else:
        dominate_colors = []

        for node in G.nodes:
            attributes = cells_percentage[node]
            if np.all(attributes == 0):
                dominate_colors.append([0, 0, 0, 1])
            else:
                max_idx = np.argmax(attributes)
                dominate_colors.append(cells_colors[max_idx])

        nx.draw_networkx_nodes(G, pos=pos_dict, node_color=dominate_colors, node_size=[s * len(cells_size) * 300 for s in cells_size], ax=ax)
        g = nx.draw_networkx_edges(
            G,
            pos=pos_dict,
            node_size=[s * len(cells_size) * 300 for s in cells_size],
            arrows=True,
            arrowstyle="->",
            arrowsize=20,
            ax=ax,
        )

    cells_color_map["None"] = np.array([0, 0, 0, 1])
    plt.legend(handles=[plt.Line2D([0], [0], marker="o", color='w', label=label,
                                   markerfacecolor=color) for label, color in cells_color_map.items()],
               loc="best",
               fontsize="medium",
               )

    return save_show_ret("plot_dim_reduced_direct_graph", save_show_or_return, save_kwargs, g)


def plot_direct_graph(
    adata: AnnData,
    layout: None = None,
    figsize: Tuple[float, float] = (6, 4),
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
) -> None:
    """Not implemented."""

    df_mat = adata.uns["df_mat"]

    import matplotlib.pyplot as plt
    import networkx as nx

    edge_color = "gray"

    G = nx.from_pandas_edgelist(
        df_mat,
        source="source",
        target="target",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )
    G.nodes()
    W = []
    for n, nbrs in G.adj.items():
        for nbr, eattr in nbrs.items():
            W.append(eattr["weight"])

    options = {
        "width": 300,
        "arrowstyle": "-|>",
        "arrowsize": 1000,
    }

    plt.figure(figsize=figsize)
    if layout is None:
        #     pos : dictionary, optional
        #        A dictionary with nodes as keys and positions as values.
        #        If not specified a spring layout positioning will be computed.
        #        See :py:mod:`networkx.drawing.layout` for functions that
        #        compute node positions.

        g = nx.draw(
            G,
            with_labels=True,
            node_color="skyblue",
            node_size=100,
            edge_color=edge_color,
            width=W / np.max(W) * 5,
            edge_cmap=plt.cm.Blues,
            options=options,
        )
    else:
        raise Exception("layout", layout, " is not supported.")

    return save_show_ret("plot_direct_graph", save_show_or_return, save_kwargs, g)
