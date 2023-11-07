"""
code are largely adapted from 
https://github.com/lmcinnes/umap/blob/7e051d8f3c4adca90ca81eb45f6a9d1372c076cf/umap/plot.py
The code base will be extended extensively to consider the following cases:
    1. nneighbors: kNN graph constructed from umap/scKDTree/annoy, etc
    2. mutual kNN shared between spliced or unspliced layer
    3. principal graph that learnt from DDRTree, L1graph or other principal graph algorithms
    4. regulatory network learnt from Scribe
    5. spatial kNN graph
    6. others
"""


from typing import Any, Dict, List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import logging
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..configuration import _themes
from ..docrep import DocstringProcessor
from ..tools.connectivity import check_and_recompute_neighbors
from .utils import is_list_of_lists  # is_gene_name
from .utils import (
    _datashade_points,
    _embed_datashader_in_an_axis,
    _get_extent,
    _select_font_color,
    save_show_ret,
)

docstrings = DocstringProcessor()


def _plt_connectivity(coord: dict, connectivity: scipy.sparse.csr_matrix) -> None:
    """Plot connectivity graph via networkx and matplotlib.

    Args:
        coord: a dictionary where the keys are the graph node names and values are the corresponding coordinates of the
            node.
        connectivity: a csr sparse matrix of the cell connectivities.
    """

    import networkx as nx

    if_symmetric = (abs(connectivity - connectivity.T) > 1e-10).nnz == 0

    G = (
        nx.from_scipy_sparse_matrix(connectivity, create_using=nx.Graph())
        if if_symmetric
        else nx.from_scipy_sparse_matrix(connectivity, create_using=nx.DiGraph())
    )
    W = []
    for n, nbrs in G.adj.items():
        for nbr, eattr in nbrs.items():
            W.append(eattr["weight"])

    options = {
        "width": 30,
        "arrowstyle": "-|>",
        "arrowsize": 10,
    }
    edge_color = "gray"
    plt.figure(figsize=[6, 4])

    nx.draw(
        G,
        pos=coord,
        with_labels=False,
        node_color="skyblue",
        node_size=1,
        edge_color=edge_color,
        width=W / np.max(W) * 1,
        edge_cmap=plt.cm.Blues,
        options=options,
    )

    plt.show()


@docstrings.get_sectionsf("con_base")
def connectivity_base(
    x: int,
    y: int,
    edge_df: pd.DataFrame,
    highlights: Optional[List[str]] = None,
    edge_bundling: Optional[Literal["hammer"]] = None,
    edge_cmap: str = "gray_r",
    show_points: bool = True,
    labels: Optional[list] = None,
    values: Optional[list] = None,
    theme: Optional[
        Literal[
            "blue",
            "red",
            "green",
            "inferno",
            "fire",
            "viridis",
            "darkblue",
            "darkgreen",
            "darkred",
        ]
    ] = None,
    cmap: str = "Blues",
    color_key: Union[dict, list, None] = None,
    color_key_cmap: str = "Spectral",
    background: str = "black",
    figsize: tuple = (7, 5),
    ax: Optional[Axes] = None,
    sort: Literal["raw", "abs"] = "raw",
    save_show_or_return: Literal["save", "show", "return"] = "return",
    save_kwargs: Dict[str, Any] = {},
) -> Optional[Axes]:
    """Plot connectivity relationships of the underlying UMAP simplicial set data structure.

    Internally UMAP will make use of what can be viewed as a weighted graph. This graph can be plotted using the layout
    provided by UMAP as a potential diagnostic view of the embedding. Currently this only works for 2D embeddings. While
    there are many optional parameters to further control and tailor the plotting, you only need pass in the trained/fit
    umap model to get results. This plot utility will attempt to do the hard work of avoiding over-plotting issues and
    provide options for plotting the points as well as using edge bundling for graph visualization.

    Args:
        x: the first component of the embedding.
        y: the second component of the embedding.
        edge_df: the dataframe denotes the graph edge pairs. The three columns include 'source', 'target' and 'weight'.
        highlights: the list that cells will be restricted to. Defaults to None.
        edge_bundling: the edge bundling method to use. Currently supported are None or 'hammer'. See the datashader
            docs on graph visualization for more details. Defaults to None.
        edge_cmap: the name of a matplotlib colormap to use for shading/coloring the edges of the connectivity graph.
            Note that the `theme`, if specified, will override this. Defaults to "gray_r".
        show_points: whether to display the points over top of the edge connectivity. Further options allow for
            coloring/shading the points accordingly. Defaults to True.
        labels: An array of labels (assumed integer or categorical), one for each data sample. This will be used for
            coloring the points in the plot according to their label. Note that this option is mutually exclusive to the
            `values` option. Defaults to None.
        values: an array of values (assumed float or continuous), one for each sample. This will be used for coloring
            the points in the plot according to a colorscale associated to the total range of values. Note that this
            option is mutually exclusive to the `labels` option. Defaults to None.
        theme: a color theme to use for plotting. A small set of predefined themes are provided which have relatively
            good aesthetics. Available themes are:
               * 'blue'
               * 'red'
               * 'green'
               * 'inferno'
               * 'fire'
               * 'viridis'
               * 'darkblue'
               * 'darkred'
               * 'darkgreen'.
            Defaults to None.
        cmap: the name of a matplotlib colormap to use for coloring or shading points. If no labels or values are passed
            this will be used for shading points according to density (largely only of relevance for very large
            datasets). If values are passed this will be used for shading according the value. Note that if theme is
            passed then this value will be overridden by the corresponding option of the theme. Defaults to "Blues".
        color_key: a way to assign colors to categorical. This can either be an explicit dict mapping labels to colors
            (as strings of form '#RRGGBB'), or an array like object providing one color for each distinct category being
            provided in `labels`. Either way this mapping will be used to color points according to the label. Note that
            if theme is passed then this value will be overridden by the corresponding option of the theme. Defaults to
            None.
        color_key_cmap: the name of a matplotlib colormap to use for categorical coloring. If an explicit `color_key` is
            not given a color mapping for categories can be generated from the label list and selecting a matching list
            of colors from the given colormap. Note that if theme is passed then this value will be overridden by the
            corresponding option of the theme. Defaults to "Spectral".
        background: the color of the background. Usually this will be either 'white' or 'black', but any color name will
            work. Ideally one wants to match this appropriately to the colors being used for points etc. This is one of
            the things that themes handle for you. Note that if theme is passed then this value will be overridden by
            the corresponding option of the theme. Defaults to "black".
        figsize: the desired size of the figure. Defaults to (7, 5).
        ax: the axis on which the subplot would be shown. If set to be `None`, a new axis would be created. Defaults to
            None.
        sort: the method to reorder data so that high values points will be on top of background points. Can be one of
            {'raw', 'abs'}, i.e. sorted by raw data or sort by absolute values. Defaults to "raw".
        save_show_or_return: whether to save, show or return the figure. Defaults to "return".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the
                {
                    "path": None,
                    "prefix": 'connectivity_base',
                    "dpi": None,
                    "ext": 'pdf',
                    "transparent": True,
                    "close": True,
                    "verbose": True
                }
            as its parameters. Otherwise, you can provide a dictionary that properly modify those keys according to your
            needs. Defaults to {}.

    Raises:
        ImportError: `datashader` is not installed.
        NotImplementedError: invalid `theme`.
        ValueError: invalid `edge_bundling`.

    Returns:
        The matplotlib axis with the relevant plot displayed by default. If `save_show_or_return` is set to be `"show"`
        or `"save"`, nothing would be returned.
    """

    try:
        import datashader as ds
        import datashader.bundling as bd
        import datashader.transfer_functions as tf
    except ImportError as e:
        logging.critical('"datashader" package is required for this function', exc_info=True)
        raise e

    dpi = plt.rcParams["figure.dpi"]

    available_themes = [
        "blue",
        "red",
        "green",
        "inferno",
        "fire",
        "viridis",
        "darkblue",
        "darkgreen",
        "darkred",
    ]
    if theme is None:
        pass
    else:
        if theme in available_themes:
            cmap = _themes[theme]["cmap"]
            color_key_cmap = _themes[theme]["color_key_cmap"]
            edge_cmap = _themes[theme]["edge_cmap"]
            background = _themes[theme]["background"]
        else:
            raise NotImplementedError('Invalid value for "theme".')

    points = np.array([x, y]).T
    point_df = pd.DataFrame(points, columns=("x", "y"))

    point_size = 500.0 / np.sqrt(points.shape[0])
    if point_size > 1:
        px_size = int(np.round(point_size))
    else:
        px_size = 1

    if show_points:
        edge_how = "log"
    else:
        edge_how = "eq_hist"

    extent = _get_extent(points)
    canvas = ds.Canvas(
        plot_width=int(figsize[0] * dpi),
        plot_height=int(figsize[1] * dpi),
        x_range=(extent[0], extent[1]),
        y_range=(extent[2], extent[3]),
    )

    if edge_bundling is None:
        edges = bd.directly_connect_edges(point_df, edge_df, weight="weight")
    elif edge_bundling == "hammer":
        warn("Hammer edge bundling is expensive for large graphs!\n" "This may take a long time to compute!")
        edges = bd.hammer_bundle(point_df, edge_df, weight="weight")
    else:
        raise ValueError("{} is not a recognised bundling method".format(edge_bundling))

    edge_img = tf.shade(
        canvas.line(edges, "x", "y", agg=ds.sum("weight")),
        cmap=plt.get_cmap(edge_cmap),
        how=edge_how,
    )
    edge_img = tf.set_background(edge_img, background)

    if show_points:
        point_img = _datashade_points(
            points,
            None,
            labels,
            values,
            highlights,
            cmap,
            color_key,
            color_key_cmap,
            None,
            figsize[0] * dpi,
            figsize[1] * dpi,
            True,
            sort=sort,
        )
        if px_size > 1:
            point_img = tf.dynspread(point_img, threshold=0.5, max_px=px_size)
        result = tf.stack(edge_img, point_img, how="over")
    else:
        result = edge_img

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    _embed_datashader_in_an_axis(result, ax)

    ax.set(xticks=[], yticks=[])

    return save_show_ret("connectivity_base", save_show_or_return, save_kwargs, ax)


docstrings.delete_params("con_base.parameters", "edge_df", "save_show_or_return", "save_kwargs")


@docstrings.with_indent(4)
def nneighbors(
    adata: AnnData,
    x: int = 0,
    y: int = 1,
    color: List[str] = ["ntr"],
    basis: List[str] = ["umap"],
    layer: List[str] = ["X"],
    highlights: Optional[list] = None,
    ncols: int = 1,
    edge_bundling: Optional[Literal["hammer"]] = None,
    edge_cmap: str = "gray_r",
    show_points: bool = True,
    labels: Optional[list] = None,
    values: Optional[list] = None,
    theme: Optional[
        Literal[
            "blue",
            "red",
            "green",
            "inferno",
            "fire",
            "viridis",
            "darkblue",
            "darkgreen",
            "darkred",
        ]
    ] = None,
    cmap: str = "Blues",
    color_key: Union[dict, list, None] = None,
    color_key_cmap: str = "Spectral",
    background: str = "black",
    figsize: tuple = (6, 4),
    ax: Optional[Axes] = None,
    save_show_or_return: Literal["save", "show", "return"] = "return",
    save_kwargs: dict = {},
) -> Optional[Figure]:
    """Plot nearest neighbor graph of cells used to embed data into low dimension space.

    Args:
        adata: an Annodata object that include the umap embedding and simplicial graph.
        x: the first component of the embedding. Defaults to 0.
        y: the second component of the embedding. Defaults to 1.
        color: gene name(s) or cell annotation column(s) used for coloring the graph. Defaults to ["ntr"].
        basis: the low dimensional embedding to be used to visualize the cell. Defaults to ["umap"].
        layer: the layers of data representing the gene expression level. Defaults to ["X"].
        highlights: the list that cells will be restricted to. Defaults to None.
        ncols: the number of columns to be plotted. Defaults to 1.
        edge_bundling: the edge bundling method to use. Currently supported are None or 'hammer'. See the datashader
            docs on graph visualization for more details. Defaults to None.
        edge_cmap: the name of a matplotlib colormap to use for shading/coloring the edges of the connectivity graph.
            Note that the `theme`, if specified, will override this. Defaults to "gray_r".
        show_points: whether to display the points over top of the edge connectivity. Further options allow for
            coloring/shading the points accordingly. Defaults to True.
        labels: an array of labels (assumed integer or categorical), one for each data sample. This will be used for
            coloring the points in the plot according to their label. Note that this option is mutually exclusive to the
            `values` option. Defaults to None.
        values: an array of values (assumed float or continuous), one for each sample. This will be used for coloring
            the points in the plot according to a colorscale associated to the total range of values. Note that this
            option is mutually exclusive to the `labels` option. Defaults to None.
        theme: a color theme to use for plotting. A small set of predefined themes are provided which have relatively
            good aesthetics. Available themes are:
               * 'blue'
               * 'red'
               * 'green'
               * 'inferno'
               * 'fire'
               * 'viridis'
               * 'darkblue'
               * 'darkred'
               * 'darkgreen'.
            Defaults to None.
        cmap: the name of a matplotlib colormap to use for coloring or shading points. If no labels or values are passed
            this will be used for shading points according to density (largely only of relevance for very large
            datasets). If values are passed this will be used for shading according the value. Note that if theme is
            passed then this value will be overridden by the corresponding option of the theme. Defaults to "Blues".
        color_key: a way to assign colors to categoricals. This can either be an explicit dict mapping labels to colors
            (as strings of form '#RRGGBB'), or an array like object providing one color for each distinct category being
            provided in `labels`. Either way this mapping will be used to color points according to the label. Note that
            if theme is passed then this value will be overridden by the corresponding option of the theme. Defaults to
            None.
        color_key_cmap: the name of a matplotlib colormap to use for categorical coloring. If an explicit `color_key` is
            not given a color mapping for categories can be generated from the label list and selecting a matching list
            of colors from the given colormap. Note that if theme is passed then this value will be overridden by the
            corresponding option of the theme. Defaults to "Spectral".
        background: the color of the background. Usually this will be either 'white' or 'black', but any color name will
            work. Ideally one wants to match this appropriately to the colors being used for points etc. This is one of
            the things that themes handle for you. Note that if theme is passed then this value will be overridden by
            the corresponding option of the theme. Defaults to "black".
        figsize: the desired size of the figure. Defaults to (6, 4).
        ax: the axis on which the subplot would be shown. If set to be `None`, a new axis would be created. Defaults to
            None.
        save_show_or_return: whether to save, show or return the figure. Defaults to "return".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the
                {
                    "path": None,
                    "prefix": 'connectivity_base',
                    "dpi": None,
                    "ext": 'pdf',
                    "transparent": True,
                    "close": True,
                    "verbose": True
                }
            as its parameters. Otherwise you can provide a dictionary that properly modify those keys according to your
            needs. Defaults to {}.

    Raises:
        TypeError: wrong type of `x` and `y`.

    Returns:
        The matplotlib axis with the plotted knn graph by default. If `save_show_or_return` is set to be `"show"`
        or `"save"`, nothing would be returned.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    if type(x) is not int or type(y) is not int:
        raise TypeError(
            "x, y have to be integers (components in the a particular embedding {}) for nneighbor "
            "function".format(basis)
        )

    n_c, n_l, n_b = (
        0 if color is None else len(color),
        0 if layer is None else len(layer),
        0 if basis is None else len(basis),
    )
    # c_is_gene_name = [is_gene_name(adata, i) for i in list(color)] if n_c > 0 else [False] * n_c
    # cnt, gene_num = 0, sum(c_is_gene_name)

    check_and_recompute_neighbors(adata, result_prefix="")
    # coo_graph = adata.uns["neighbors"]["connectivities"].tocoo()
    coo_graph = adata.obsp["connectivities"].tocoo()

    edge_df = pd.DataFrame(
        np.vstack([coo_graph.row, coo_graph.col, coo_graph.data]).T,
        columns=("source", "target", "weight"),
    )
    edge_df["source"] = edge_df.source.astype(np.int32)
    edge_df["target"] = edge_df.target.astype(np.int32)

    total_panels, ncols = n_c * n_l * n_b, min(n_c, ncols)
    nrow, ncol = int(np.ceil(total_panels / ncols)), ncols
    if figsize is None:
        figsize = plt.rcParams["figsize"]

    font_color = _select_font_color(background)
    if background == "black":
        # https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mpl-data/stylelib/dark_background.mplstyle
        sns.set(
            rc={
                "axes.facecolor": background,
                "axes.edgecolor": background,
                "figure.facecolor": background,
                "figure.edgecolor": background,
                "axes.grid": False,
                "ytick.color": font_color,
                "xtick.color": font_color,
                "axes.labelcolor": font_color,
                "axes.edgecolor": font_color,
                "savefig.facecolor": "k",
                "savefig.edgecolor": "k",
                "grid.color": font_color,
                "text.color": font_color,
                "lines.color": font_color,
                "patch.edgecolor": font_color,
                "figure.edgecolor": font_color,
            }
        )
    else:
        sns.set(
            rc={
                "axes.facecolor": background,
                "figure.facecolor": background,
                "axes.grid": False,
            }
        )

    if total_panels > 1:
        g = plt.figure(None, (figsize[0] * ncol, figsize[1] * nrow), facecolor=background)
        gs = plt.GridSpec(nrow, ncol, wspace=0.12)

    i = 0
    for cur_b in basis:
        for cur_l in layer:
            prefix = cur_l + "_"
            if prefix + cur_b in adata.obsm.keys():
                x_, y_ = (
                    adata.obsm[prefix + cur_b][:, int(x)],
                    adata.obsm[prefix + cur_b][:, int(y)],
                )
            else:
                continue
            for cur_c in color:
                _color = adata.obs_vector(cur_c, layer=cur_l)
                is_not_continous = _color.dtype.name == "category"
                if is_not_continous:
                    labels = _color
                    if theme is None:
                        theme = "glasbey_dark"
                else:
                    values = _color
                    if theme is None:
                        theme = "inferno" if cur_l != "velocity" else "div_blue_red"

                if total_panels > 1:
                    ax = plt.subplot(gs[i])
                i += 1

                # if highligts is a list of lists - each list is relate to each color element
                if is_list_of_lists(highlights):
                    _highlights = highlights[color.index(cur_c)]
                    _highlights = _highlights if all([i in _color for i in _highlights]) else None
                else:
                    _highlights = highlights if all([i in _color for i in highlights]) else None

                connectivity_base(
                    x_,
                    y_,
                    edge_df,
                    edge_bundling,
                    edge_cmap,
                    show_points,
                    labels,
                    values,
                    _highlights,
                    theme,
                    cmap,
                    color_key,
                    color_key_cmap,
                    background,
                    figsize,
                    ax,
                )

                ax.set_xlabel(
                    cur_b + "_1",
                )
                ax.set_ylabel(cur_b + "_2")
                ax.set_title(cur_c)

    return save_show_ret("nneighbors", save_show_or_return, save_kwargs, g)


def pgraph():
    """Plot principal graph of cells that learnt from graph embedding algorithms.

    return:
    """
    pass


def cgroups():
    """Plot transition matrix graph of groups of cells that produced from clustering or other grouping procedures.

    :return:
    """
    pass


def causal_net():
    """Plot causal regulatory networks of genes learnt with Scribe (https://github.com/aristoteleo/Scribe-py).

    :return:
    """
    pass
