"""
# code are largely adapted from https://github.com/lmcinnes/umap/blob/7e051d8f3c4adca90ca81eb45f6a9d1372c076cf/umap/plot.py
The code base will be extended extensively to consider the following cases:
    1. nneighbors: kNN graph constructed from umap/scKDTree/annoy, etc
    2. mutual kNN shared between spliced or unspliced layer
    3. principal graph that learnt from DDRTree, L1graph or other principal graph algorithms
    4. regulatory network learnt from Scribe
    5. spatial kNN graph
    6. others
"""

from .utils import _select_font_color, _get_extent, _embed_datashader_in_an_axis, _to_hex
from .utils import is_gene_name, is_list_of_lists
from ..configuration import _themes

import pandas as pd
import numpy as np
from warnings import warn
import datashader as ds
import datashader.transfer_functions as tf
import datashader.bundling as bd


def _plt_connectivity(coord, connectivity):
    """Plot connectivity graph via networkx and matplotlib.

    Parameters
    ----------
        coord: `dict`
            A dictionary where the keys are the graph node names and values are the corresponding coordinates of the node.
        connectivity: `scipy.sparse.csr_matrix`
            A csr sparse matrix of the cell connectivities.

    Returns
    -------
        Nothing but a connectivity graph plot built upon networkx and matplotlib.
    """

    import networkx as nx
    import matplotlib.pyplot as plt

    if_symmetric = (abs(connectivity-connectivity.T) > 1e-10).nnz == 0

    G = nx.from_scipy_sparse_matrix(connectivity, create_using=nx.Graph()) if if_symmetric else \
        nx.from_scipy_sparse_matrix(connectivity, create_using=nx.DiGraph())
    W = []
    for n, nbrs in G.adj.items():
        for nbr, eattr in nbrs.items():
            W.append(eattr['weight'])

    options = {
        'width': 30,
        'arrowstyle': '-|>',
        'arrowsize': 10,
    }
    edge_color = 'gray'
    plt.figure(figsize=[10, 10])

    nx.draw(G, pos=coord, with_labels=False, node_color='skyblue', node_size=1,
            edge_color=edge_color, width=W / np.max(W) * 1, edge_cmap=plt.cm.Blues, options=options)

    plt.show()


def _datashade_points(
    points,
    ax=None,
    labels=None,
    values=None,
    highlights=None,
    cmap="blue",
    color_key=None,
    color_key_cmap="Spectral",
    background="black",
    width=700,
    height=500,
    show_legend=True,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    """Use datashader to plot points"""
    extent = _get_extent(points)
    canvas = ds.Canvas(
        plot_width=int(width),
        plot_height=int(height),
        x_range=(extent[0], extent[1]),
        y_range=(extent[2], extent[3]),
    )
    data = pd.DataFrame(points, columns=("x", "y"))

    legend_elements = None

    # Color by labels
    if labels is not None:
        if labels.shape[0] != points.shape[0]:
            raise ValueError(
                "Labels must have a label for "
                "each sample (size mismatch: {} {})".format(
                    labels.shape[0], points.shape[0]
                )
            )

        labels = np.array(labels, dtype='str')
        data["label"] = pd.Categorical(labels)
        if color_key is None and color_key_cmap is None:
            aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("label"))
            result = tf.shade(aggregation, how="eq_hist")
        elif color_key is None:
            if highlights is None:
                aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("label"))
                unique_labels = np.unique(labels)
                num_labels = unique_labels.shape[0]
                color_key = _to_hex(
                    plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
                )
            else:
                highlights.append('other')
                unique_labels = np.array(highlights)
                num_labels = unique_labels.shape[0]
                color_key = _to_hex(
                    plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
                )
                color_key[-1] = '#bdbdbd' # lightgray hex code https://www.color-hex.com/color/d3d3d3

                labels[[i not in highlights for i in labels]] = 'other'
                data["label"] = pd.Categorical(labels)
                aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("label"))

            legend_elements = [
                Patch(facecolor=color_key[i], label=k)
                for i, k in enumerate(unique_labels)
            ]
            result = tf.shade(aggregation, color_key=color_key, how="eq_hist")
        else:
            aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("label"))

            legend_elements = [
                Patch(facecolor=color_key[k], label=k) for k in color_key.keys()
            ]
            result = tf.shade(aggregation, color_key=color_key, how="eq_hist")

    # Color by values
    elif values is not None:
        if values.shape[0] != points.shape[0]:
            raise ValueError(
                "Values must have a value for "
                "each sample (size mismatch: {} {})".format(
                    values.shape[0], points.shape[0]
                )
            )
        unique_values = np.unique(values)
        if unique_values.shape[0] >= 256:
            min_val, max_val = np.min(values), np.max(values)
            bin_size = (max_val - min_val) / 255.0
            data["val_cat"] = pd.Categorical(
                np.round((values - min_val) / bin_size).astype(np.int16)
            )
            aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("val_cat"))
            color_key = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, 256)))
            result = tf.shade(aggregation, color_key=color_key, how="eq_hist")
        else:
            data["val_cat"] = pd.Categorical(values)
            aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("val_cat"))
            color_key_cols = _to_hex(
                plt.get_cmap(cmap)(np.linspace(0, 1, unique_values.shape[0]))
            )
            color_key = dict(zip(unique_values, color_key_cols))
            result = tf.shade(aggregation, color_key=color_key, how="eq_hist")

    # Color by density (default datashader option)
    else:
        aggregation = canvas.points(data, "x", "y", agg=ds.count())
        result = tf.shade(aggregation, cmap=plt.get_cmap(cmap))

    if background is not None:
        result = tf.set_background(result, background)

    if ax is not None:
        _embed_datashader_in_an_axis(result, ax)
        if show_legend and legend_elements is not None:
            ax.legend(handles=legend_elements)
        return ax
    else:
        return result


def connectivity_base(
        x,
        y,
        edge_df,
        edge_bundling=None,
        edge_cmap="gray_r",
        show_points=True,
        labels=None,
        values=None,
        highlights=None,
        theme=None,
        cmap="blue",
        color_key=None,
        color_key_cmap="Spectral",
        background="black",
        figsize=(7,5),
        ax=None,
):
    """Plot connectivity relationships of the underlying UMAP
    simplicial set data structure. Internally UMAP will make
    use of what can be viewed as a weighted graph. This graph
    can be plotted using the layout provided by UMAP as a
    potential diagnostic view of the embedding. Currently this only works
    for 2D embeddings. While there are many optional parameters
    to further control and tailor the plotting, you need only
    pass in the trained/fit umap model to get results. This plot
    utility will attempt to do the hard work of avoiding
    overplotting issues and provide options for plotting the
    points as well as using edge bundling for graph visualization.

    Parameters
    ----------
    x: `int`
        The first component of the embedding.
    y: `int`
        The second component of the embedding.
    edge_bundling: string or None (optional, default None)
        The edge bundling method to use. Currently supported
        are None or 'hammer'. See the datashader docs
        on graph visualization for more details.
    edge_cmap: string (default 'gray_r')
        The name of a matplotlib colormap to use for shading/
        coloring the edges of the connectivity graph. Note that
        the ``theme``, if specified, will override this.
    show_points: bool (optional False)
        Whether to display the points over top of the edge
        connectivity. Further options allow for coloring/
        shading the points accordingly.
    labels: array, shape (n_samples,) (optional, default None)
        An array of labels (assumed integer or categorical),
        one for each data sample.
        This will be used for coloring the points in
        the plot according to their label. Note that
        this option is mutually exclusive to the ``values``
        option.
    values: array, shape (n_samples,) (optional, default None)
        An array of values (assumed float or continuous),
        one for each sample.
        This will be used for coloring the points in
        the plot according to a colorscale associated
        to the total range of values. Note that this
        option is mutually exclusive to the ``labels``
        option.
    theme: string (optional, default None)
        A color theme to use for plotting. A small set of
        predefined themes are provided which have relatively
        good aesthetics. Available themes are:
           * 'blue'
           * 'red'
           * 'green'
           * 'inferno'
           * 'fire'
           * 'viridis'
           * 'darkblue'
           * 'darkred'
           * 'darkgreen'
    cmap: string (optional, default 'Blues')
        The name of a matplotlib colormap to use for coloring
        or shading points. If no labels or values are passed
        this will be used for shading points according to
        density (largely only of relevance for very large
        datasets). If values are passed this will be used for
        shading according the value. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.
    color_key: dict or array, shape (n_categories) (optional, default None)
        A way to assign colors to categoricals. This can either be
        an explicit dict mapping labels to colors (as strings of form
        '#RRGGBB'), or an array like object providing one color for
        each distinct category being provided in ``labels``. Either
        way this mapping will be used to color points according to
        the label. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.
    color_key_cmap: string (optional, default 'Spectral')
        The name of a matplotlib colormap to use for categorical coloring.
        If an explicit ``color_key`` is not given a color mapping for
        categories can be generated from the label list and selecting
        a matching list of colors from the given colormap. Note
        that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.
    background: string (optional, default 'white)
        The color of the background. Usually this will be either
        'white' or 'black', but any color name will work. Ideally
        one wants to match this appropriately to the colors being
        used for points etc. This is one of the things that themes
        handle for you. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.
    width: int (optional, default 800)
        The desired width of the plot in pixels.
    height: int (optional, default 800)
        The desired height of the plot in pixels
    Returns
    -------
    result: matplotlib axis
        The result is a matplotlib axis with the relevant plot displayed.
        If you are using a notbooks and have ``%matplotlib inline`` set
        then this will simply display inline.
    """

    import matplotlib.pyplot as plt
    dpi = plt.rcParams["figure.dpi"]

    if theme is not None:
        cmap = _themes[theme]["cmap"]
        color_key_cmap = _themes[theme]["color_key_cmap"]
        edge_cmap = _themes[theme]["edge_cmap"]
        background = _themes[theme]["background"]

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
        warn(
            "Hammer edge bundling is expensive for large graphs!\n"
            "This may take a long time to compute!"
        )
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

    return ax


def nneighbors(adata,
        x=0,
        y=1,
        color=None,
        basis='umap',
        layer='X',
        highlights=None,
        edge_bundling=None,
        edge_cmap="gray_r",
        show_points=True,
        labels=None,
        values=None,
        theme=None,
        cmap=None,
        color_key=None,
        color_key_cmap=None,
        background="black",
        ncols=1,
        figsize=(7,5),
        ax=None):
    """
    Plot nearest neighbor graph of cells used to embed data into low dimension space.
    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object that include the umap embedding and simplicial graph.
        x: `int`
            The first component of the embedding.
        y: `int`
            The second component of the embedding.
        color: `str` or list of `str` or None (default: None)
            Gene name(s) or cell annotation column(s)
        basis: `str` or list of `str` (default: `X`)
            Which low dimensional embedding will be used to visualize the cell.
        layer: `str` or list of `str` (default: `X`)
            The layers of data to represent the gene expression level.
        highlights: `list`, `list of list` or None (default: `None`)
            The list that cells will be restricted to.
        edge_bundling: string or None (optional, default None)
            The edge bundling method to use. Currently supported
            are None or 'hammer'. See the datashader docs
            on graph visualization for more details.
        edge_cmap: string (default 'gray_r')
            The name of a matplotlib colormap to use for shading/
            coloring the edges of the connectivity graph. Note that
            the ``theme``, if specified, will override this.
        show_points: bool (optional False)
            Whether to display the points over top of the edge
            connectivity. Further options allow for coloring/
            shading the points accordingly.
        labels: array, shape (n_samples,) (optional, default None)
            An array of labels (assumed integer or categorical),
            one for each data sample.
            This will be used for coloring the points in
            the plot according to their label. Note that
            this option is mutually exclusive to the ``values``
            option.
        values: array, shape (n_samples,) (optional, default None)
            An array of values (assumed float or continuous),
            one for each sample.
            This will be used for coloring the points in
            the plot according to a colorscale associated
            to the total range of values. Note that this
            option is mutually exclusive to the ``labels``
            option.
        theme: string (optional, default None)
            A color theme to use for plotting. A small set of
            predefined themes are provided which have relatively
            good aesthetics. Available themes are:
               * 'blue'
               * 'red'
               * 'green'
               * 'inferno'
               * 'fire'
               * 'viridis'
               * 'darkblue'
               * 'darkred'
               * 'darkgreen'
        cmap: string (optional, default 'Blues')
            The name of a matplotlib colormap to use for coloring
            or shading points. If no labels or values are passed
            this will be used for shading points according to
            density (largely only of relevance for very large
            datasets). If values are passed this will be used for
            shading according the value. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        color_key: dict or array, shape (n_categories) (optional, default None)
            A way to assign colors to categoricals. This can either be
            an explicit dict mapping labels to colors (as strings of form
            '#RRGGBB'), or an array like object providing one color for
            each distinct category being provided in ``labels``. Either
            way this mapping will be used to color points according to
            the label. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        color_key_cmap: string (optional, default 'Spectral')
            The name of a matplotlib colormap to use for categorical coloring.
            If an explicit ``color_key`` is not given a color mapping for
            categories can be generated from the label list and selecting
            a matching list of colors from the given colormap. Note
            that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        background: string (optional, default 'white)
            The color of the background. Usually this will be either
            'white' or 'black', but any color name will work. Ideally
            one wants to match this appropriately to the colors being
            used for points etc. This is one of the things that themes
            handle for you. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
            ncols: `int`
                The number of columns of the plot.
            figsize: `list` or `tuple` (default: (7, 5))
                The width and height of a figure.
            ax: `matplotlib.axes._subplots.AxesSubplot`
                The axis that the connectivity plot will attached to. Only workable if the
                argument combination involves a single ax.

    Returns
    -------

    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if type(x) is not int or type(y) is not int:
        raise Exception('x, y have to be integers (components in the a particular embedding {}) for nneighbor '
                        'function'.format(basis))

    n_c, n_l, n_b = 0 if color is None else len(color), 0 if layer is None else len(layer), 0 if basis is None else len(basis)
    c_is_gene_name = [is_gene_name(adata, i) for i in list(color)] if n_c > 0 else None
    cnt, gene_num = 0, sum(c_is_gene_name)

    coo_graph = adata.uns['neighbors']['connectivities'].tocoo()
    edge_df = pd.DataFrame(
        np.vstack([coo_graph.row, coo_graph.col, coo_graph.data]).T,
        columns=("source", "target", "weight"),
    )
    edge_df["source"] = edge_df.source.astype(np.int32)
    edge_df["target"] = edge_df.target.astype(np.int32)

    total_panels, ncols = n_c * n_l * n_b, min(gene_num, ncols)
    nrow, ncol = int(np.ceil(total_panels / ncols)), ncols
    if figsize is None: figsize = plt.rcParams['figsize']

    font_color = _select_font_color(background)
    if background == 'black':
        # https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mpl-data/stylelib/dark_background.mplstyle
        sns.set(rc={'axes.facecolor': background, 'axes.edgecolor': background, 'figure.facecolor': background, 'figure.edgecolor': background,
                    'axes.grid': False, "ytick.color": font_color, "xtick.color": font_color, "axes.labelcolor": font_color, "axes.edgecolor": font_color,
                    "savefig.facecolor": 'k', "savefig.edgecolor": 'k', "grid.color": font_color, "text.color": font_color,
                    "lines.color": font_color, "patch.edgecolor": font_color, 'figure.edgecolor': font_color,
                    })
    else:
        sns.set(rc={'axes.facecolor': background, 'figure.facecolor': background, 'axes.grid': False})

    if total_panels > 1:
        plt.figure(None, (figsize[0] * ncol, figsize[1] * nrow), facecolor=background)
        gs = plt.GridSpec(nrow, ncol)

    i = 0
    for cur_b in basis:
        for cur_l in layer:
            prefix = cur_l + '_'
            if prefix + cur_b in adata.obsm.keys():
                x_, y_ = adata.obsm[prefix + cur_b][:, int(x)], adata.obsm[prefix + cur_b][:, int(y)]
            else:
                continue
            for cur_c in color:
                _color = adata.obs_vector(cur_c, layer=cur_l)
                is_not_continous = np.allclose(_color[_color > 0][:20] % 1, 0, atol=1e-3)
                if is_not_continous:
                    labels = _color
                    if theme is None: theme = 'glasbey_dark'
                else:
                    values = _color
                    if theme is None: theme = 'inferno' if cur_l is not 'velocity' else 'div_blue_red'

                if total_panels > 1:
                    ax = plt.subplot(gs[i])
                i += 1

                # if highligts is a list of lists
                if is_list_of_lists(highlights):
                    _highlights = highlights[color.index(cur_c)]
                    _highlights = _highlights if all([i in _color for i in _highlights]) else None
                else:
                    _highlights = highlights if all([i in _color for i in highlights]) else None

                connectivity_base(
                    x_, y_, edge_df,
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
                    ax
                )

                ax.set_xlabel(cur_b + '_1', )
                ax.set_ylabel(cur_b + '_2')
                ax.set_title(cur_c)

    plt.tight_layout()
    plt.show()


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



