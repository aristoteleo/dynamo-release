# code adapted from https://github.com/lmcinnes/umap/blob/7e051d8f3c4adca90ca81eb45f6a9d1372c076cf/umap/plot.py
from ..configuration import _themes

import numpy as np
import pandas as pd
import numba
from warnings import warn

import matplotlib.colors
import matplotlib.cm
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

import datashader as ds
import datashader.transfer_functions as tf
import datashader.bundling as bd

import bokeh.plotting as bpl
import bokeh.transform as btr
from bokeh.plotting import output_notebook, output_file, show

import holoviews as hv
import holoviews.operation.datashader as hd


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


# https://stackoverflow.com/questions/8468855/convert-a-rgb-colour-value-to-decimal
"""Convert RGB color to decimal RGB integers are typically treated as three distinct bytes where the left-most (highest-order) 
byte is red, the middle byte is green and the right-most (lowest-order) byte is blue. """

@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _red(x):
    return (x & 0xFF0000) >> 16


@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _green(x):
    return (x & 0x00FF00) >> 8


@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _blue(x):
    return x & 0x0000FF


def _embed_datashader_in_an_axis(datashader_image, ax):
    img_rev = datashader_image.data[::-1]
    mpl_img = np.dstack([_blue(img_rev), _green(img_rev), _red(img_rev)])
    ax.imshow(mpl_img)
    return ax


def _get_extent(points):
    """Compute bounds on a space with appropriate padding"""
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    extent = (
        np.round(min_x - 0.05 * (max_x - min_x)),
        np.round(max_x + 0.05 * (max_x - min_x)),
        np.round(min_y - 0.05 * (max_y - min_y)),
        np.round(max_y + 0.05 * (max_y - min_y)),
    )

    return extent


def _select_font_color(background):
    if background == "black":
        font_color = "white"
    elif background.startswith("#"):
        mean_val = np.mean(
            [int("0x" + c) for c in (background[1:3], background[3:5], background[5:7])]
        )
        if mean_val > 126:
            font_color = "black"
        else:
            font_color = "white"

    else:
        font_color = "black"

    return font_color


def _datashade_points(
    points,
    ax=None,
    labels=None,
    values=None,
    cmap="Blues",
    color_key=None,
    color_key_cmap="Spectral",
    background="white",
    width=800,
    height=800,
    show_legend=True,
):

    """Use datashader to plot points"""
    extent = _get_extent(points)
    canvas = ds.Canvas(
        plot_width=width,
        plot_height=height,
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

        data["label"] = pd.Categorical(labels)
        aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("label"))
        if color_key is None and color_key_cmap is None:
            result = tf.shade(aggregation, how="eq_hist")
        elif color_key is None:
            unique_labels = np.unique(labels)
            num_labels = unique_labels.shape[0]
            color_key = _to_hex(
                plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
            )
            legend_elements = [
                Patch(facecolor=color_key[i], label=k)
                for i, k in enumerate(unique_labels)
            ]
            result = tf.shade(aggregation, color_key=color_key, how="eq_hist")
        else:
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


def _matplotlib_points(
    points,
    ax=None,
    labels=None,
    values=None,
    cmap="Blues",
    color_key=None,
    color_key_cmap="Spectral",
    background="white",
    width=800,
    height=800,
    show_legend=True,
):
    """Use matplotlib to plot points"""
    point_size = 100.0 / np.sqrt(points.shape[0])

    legend_elements = None

    if ax is None:
        dpi = plt.rcParams["figure.dpi"]
        fig = plt.figure(figsize=(width / dpi, height / dpi))
        ax = fig.add_subplot(111)

    ax.set_facecolor(background)

    # Color by labels
    if labels is not None:
        if labels.shape[0] != points.shape[0]:
            raise ValueError(
                "Labels must have a label for "
                "each sample (size mismatch: {} {})".format(
                    labels.shape[0], points.shape[0]
                )
            )
        if color_key is None:
            unique_labels = np.unique(labels)
            num_labels = unique_labels.shape[0]
            color_key = plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
            legend_elements = [
                Patch(facecolor=color_key[i], label=unique_labels[i])
                for i, k in enumerate(unique_labels)
            ]

        if isinstance(color_key, dict):
            colors = pd.Series(labels).map(color_key)
            unique_labels = np.unique(labels)
            legend_elements = [
                Patch(facecolor=color_key[k], label=k) for k in unique_labels
            ]
        else:
            unique_labels = np.unique(labels)
            if len(color_key) < unique_labels.shape[0]:
                raise ValueError(
                    "Color key must have enough colors for the number of labels"
                )

            new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
            legend_elements = [
                Patch(facecolor=color_key[i], label=k)
                for i, k in enumerate(unique_labels)
            ]
            colors = pd.Series(labels).map(new_color_key)

        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=colors)

    # Color by values
    elif values is not None:
        if values.shape[0] != points.shape[0]:
            raise ValueError(
                "Values must have a value for "
                "each sample (size mismatch: {} {})".format(
                    values.shape[0], points.shape[0]
                )
            )
        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=values, cmap=cmap)

    # No color (just pick the midpoint of the cmap)
    else:

        color = plt.get_cmap(cmap)(0.5)
        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=color)

    if show_legend and legend_elements is not None:
        ax.legend(handles=legend_elements)

    return ax


def points(
    adata,
    basis,
    labels=None,
    values=None,
    theme=None,
    cmap="fire",
    color_key=None,
    color_key_cmap="Spectral",
    background="black",
    width=800,
    height=800,
    show_legend=True,
):
    """Plot an embedding as points. Currently this only works
    for 2D embeddings. While there are many optional parameters
    to further control and tailor the plotting, you need only
    pass in the trained/fit umap model to get results. This plot
    utility will attempt to do the hard work of avoiding
    overplotting issues, and make it easy to automatically
    colour points by a categorical labelling or numeric values.
    This method is intended to be used within a Jupyter
    notebook with ``%matplotlib inline``.
    Parameters
    ----------
    adata: an anndata object.
    basis: `str`
        The reduced dimension.
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
    show_legend: bool (optional, default True)
        Whether to display a legend of the labels
    Returns
    -------
    result: matplotlib axis
        The result is a matplotlib axis with the relevant plot displayed.
        If you are using a notbooks and have ``%matplotlib inline`` set
        then this will simply display inline.
    """
    if not hasattr(umap_object, "embedding_"):
        raise ValueError(
            "UMAP object must perform fit on data before it can be visualized"
        )

    if theme is not None:
        cmap = _themes[theme]["cmap"]
        color_key_cmap = _themes[theme]["color_key_cmap"]
        background = _themes[theme]["background"]

    if labels is not None and values is not None:
        raise ValueError(
            "Conflicting options; only one of labels or values should be set"
        )

    points = adata.obsm['X_' + basis]

    if points.shape[1] != 2:
        raise ValueError("Plotting is currently only implemented for 2D embeddings")

    font_color = _select_font_color(background)

    dpi = plt.rcParams["figure.dpi"]
    fig = plt.figure(figsize=(width / dpi, height / dpi))
    ax = fig.add_subplot(111)

    if points.shape[0] <= width * height // 10:
        ax = _matplotlib_points(
            points,
            ax,
            labels,
            values,
            cmap,
            color_key,
            color_key_cmap,
            background,
            width,
            height,
            show_legend,
        )
    else:
        ax = _datashade_points(
            points,
            ax,
            labels,
            values,
            cmap,
            color_key,
            color_key_cmap,
            background,
            width,
            height,
            show_legend,
        )

    ax.set(xticks=[], yticks=[])
    if umap_object.metric != "euclidean":
        ax.text(
            0.99,
            0.01,
            "UMAP: metric={}, n_neighbors={}, min_dist={}".format(
                umap_object.metric, umap_object.n_neighbors, umap_object.min_dist
            ),
            transform=ax.transAxes,
            horizontalalignment="right",
            color=font_color,
        )
    else:
        ax.text(
            0.99,
            0.01,
            "UMAP: n_neighbors={}, min_dist={}".format(
                umap_object.n_neighbors, umap_object.min_dist
            ),
            transform=ax.transAxes,
            horizontalalignment="right",
            color=font_color,
        )

    return ax


def connectivity(
        umap_object,
        edge_bundling=None,
        edge_cmap="gray_r",
        show_points=False,
        labels=None,
        values=None,
        theme=None,
        cmap="Blues",
        color_key=None,
        color_key_cmap="Spectral",
        background="white",
        width=800,
        height=800,
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
    umap_object: trained UMAP object
        A trained UMAP object that has a 2D embedding.
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
    if theme is not None:
        cmap = _themes[theme]["cmap"]
        color_key_cmap = _themes[theme]["color_key_cmap"]
        edge_cmap = _themes[theme]["edge_cmap"]
        background = _themes[theme]["background"]

    points = umap_object.embedding_
    point_df = pd.DataFrame(points, columns=("x", "y"))

    point_size = 100.0 / np.sqrt(points.shape[0])
    if point_size > 1:
        px_size = int(np.round(point_size))
    else:
        px_size = 1

    if show_points:
        edge_how = "log"
    else:
        edge_how = "eq_hist"

    coo_graph = umap_object.graph_.tocoo()
    edge_df = pd.DataFrame(
        np.vstack([coo_graph.row, coo_graph.col, coo_graph.data]).T,
        columns=("source", "target", "weight"),
    )
    edge_df["source"] = edge_df.source.astype(np.int32)
    edge_df["target"] = edge_df.target.astype(np.int32)

    extent = _get_extent(points)
    canvas = ds.Canvas(
        plot_width=width,
        plot_height=height,
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
            cmap,
            color_key,
            color_key_cmap,
            None,
            width,
            height,
            show_legend,
        )
        if px_size > 1:
            point_img = tf.dynspread(point_img, threshold=0.5, max_px=px_size)
        result = tf.stack(edge_img, point_img, how="over")
    else:
        result = edge_img

    font_color = _select_font_color(background)

    dpi = plt.rcParams["figure.dpi"]
    fig = plt.figure(figsize=(width / dpi, height / dpi))
    ax = fig.add_subplot(111)

    _embed_datashader_in_an_axis(result, ax)

    ax.set(xticks=[], yticks=[])
    ax.text(
        0.99,
        0.01,
        "UMAP: n_neighbors={}, min_dist={}".format(
            umap_object.n_neighbors, umap_object.min_dist
        ),
        transform=ax.transAxes,
        horizontalalignment="right",
        color=font_color,
    )

    return ax


def interactive(
    umap_object,
    labels=None,
    values=None,
    hover_data=None,
    theme=None,
    cmap="Blues",
    color_key=None,
    color_key_cmap="Spectral",
    background="white",
    width=800,
    height=800,
    point_size=None,
):
    """Create an interactive bokeh plot of a UMAP embedding.
    While static plots are useful, sometimes a plot that
    supports interactive zooming, and hover tooltips for
    individual points is much more desireable. This function
    provides a simple interface for creating such plots. The
    result is a bokeh plot that will be displayed in a notebook.
    Note that more complex tooltips etc. will require custom
    code -- this is merely meant to provide fast and easy
    access to interactive plotting.
    Parameters
    ----------
    umap_object: trained UMAP object
        A trained UMAP object that has a 2D embedding.
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
    hover_data: DataFrame, shape (n_samples, n_tooltip_features)
    (optional, default None)
        A dataframe of tooltip data. Each column of the dataframe
        should be a Series of length ``n_samples`` providing a value
        for each data point. Column names will be used for
        identifying information within the tooltip.
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
    """
    if theme is not None:
        cmap = _themes[theme]["cmap"]
        color_key_cmap = _themes[theme]["color_key_cmap"]
        background = _themes[theme]["background"]

    if labels is not None and values is not None:
        raise ValueError(
            "Conflicting options; only one of labels or values should be set"
        )

    points = umap_object.embedding_

    if points.shape[1] != 2:
        raise ValueError("Plotting is currently only implemented for 2D embeddings")

    if point_size is None:
        point_size = 100.0 / np.sqrt(points.shape[0])

    data = pd.DataFrame(umap_object.embedding_, columns=("x", "y"))

    if labels is not None:
        data["label"] = labels

        if color_key is None:
            unique_labels = np.unique(labels)
            num_labels = unique_labels.shape[0]
            color_key = _to_hex(
                plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
            )

        if isinstance(color_key, dict):
            data["color"] = pd.Series(labels).map(color_key)
        else:
            unique_labels = np.unique(labels)
            if len(color_key) < unique_labels.shape[0]:
                raise ValueError(
                    "Color key must have enough colors for the number of labels"
                )

            new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
            data["color"] = pd.Series(labels).map(new_color_key)

        colors = "color"

    elif values is not None:
        data["value"] = values
        palette = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, 256)))
        colors = btr.linear_cmap(
            "value", palette, low=np.min(values), high=np.max(values)
        )

    else:
        colors = matplotlib.colors.rgb2hex(plt.get_cmap(cmap)(0.5))

    if points.shape[0] <= width * height // 10:

        if hover_data is not None:
            tooltip_dict = {}
            for col_name in hover_data:
                data[col_name] = hover_data[col_name]
                tooltip_dict[col_name] = "@" + col_name
            tooltips = list(tooltip_dict.items())
        else:
            tooltips = None

        # bpl.output_notebook(hide_banner=True) # this doesn't work for non-notebook use
        data_source = bpl.ColumnDataSource(data)

        plot = bpl.figure(
            width=width,
            height=height,
            tooltips=tooltips,
            background_fill_color=background,
        )
        plot.circle(x="x", y="y", source=data_source, color=colors, size=point_size)

        plot.grid.visible = False
        plot.axis.visible = False

        # bpl.show(plot)
    else:
        if hover_data is not None:
            warn(
                "Too many points for hover data -- tooltips will not"
                "be displayed. Sorry; try subssampling your data."
            )
        hv.extension("bokeh")
        hv.output(size=300)
        hv.opts('RGB [bgcolor="{}", xaxis=None, yaxis=None]'.format(background))
        if labels is not None:
            point_plot = hv.Points(data, kdims=["x", "y"], vdims=["color"])
            plot = hd.datashade(
                point_plot,
                aggregator=ds.count_cat("color"),
                cmap=plt.get_cmap(cmap),
                width=width,
                height=height,
            )
        elif values is not None:
            min_val = data.values.min()
            val_range = data.values.max() - min_val
            data["val_cat"] = pd.Categorical(
                (data.values - min_val) // (val_range // 256)
            )
            point_plot = hv.Points(data, kdims=["x", "y"], vdims=["val_cat"])
            plot = hd.datashade(
                point_plot,
                aggregator=ds.count_cat("val_cat"),
                cmap=plt.get_cmap(cmap),
                width=width,
                height=height,
            )
        else:
            point_plot = hv.Points(data, kdims=["x", "y"])
            plot = hd.datashade(
                point_plot,
                aggregator=ds.count(),
                cmap=plt.get_cmap(cmap),
                width=width,
                height=height,
            )

    return plot


def scatters(adata, gene_names, color, dims=[0, 1], current_layer='spliced', use_raw=False, Vkey='S', Ekey='spliced',
             basis='umap', mode='expression', label_on_embedding=True, cmap=None, gs=None, **kwargs):
    """Scatter

    Parameters
    ----------
        adata
        gene_names
        color
        dims
        current_layer
        use_raw
        Vkey
        Ekey
        basis
        mode
        cmap
        gs
        kwargs

    Returns
    -------

    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patheffects as PathEffects

    color = adata.obs.loc[:, color]
    if mode is 'expression' and cmap is None:
        cmap = plt.cm.Greens  # qualitative
    elif mode is 'velocity' and cmap is None:
        cmap = plt.cm.RdBu_r  # diverging
    elif mode is 'phase':
        from matplotlib import cm
        viridis = cm.get_cmap('viridis', len(np.unique(color)))
        cmap = viridis

    # plt.figure(None, (17, 2.8), dpi=80)
    # gs = plt.GridSpec(1, 6)

    gene_names = list(set(gene_names).intersection(adata.var.index))
    ix = np.where(adata.var.index.isin(gene_names))[0][0]

    # do the RNA/protein phase diagram on the same plot
    if mode is 'phase':
        gamma, q = adata.var.velocity_parameter_gamma[ix], adata.var.velocity_parameter_q[ix]
        if use_raw:
            x, y = adata.layers['spliced'][:, ix], adata.layers['unspliced'][: ix] if current_layer is 'spliced' else \
                adata.layers['protein'][:, ix], adata.layers['spliced'][: ix]
        else:
            x, y = adata.layers['X_spliced'][:, ix], adata.layers['X_unspliced'][
                                                     : ix] if current_layer is 'spliced' else \
                adata.layers['X_protein'][:, ix], adata.layers['X_spliced'][: ix]

        plt.scatter(x, y, s=5, alpha=0.4, rasterized=True)  # , c=vlm.colorandum
        plt.title(gene_names)
        xnew = np.linspace(0, x.max())
        plt.plot(xnew, gamma * xnew + q, c="k")
        plt.ylim(0, np.max(y) * 1.02)
        plt.xlim(0, np.max(x) * 1.02)
        minimal_yticks(0, y * 1.02)
        minimal_xticks(0, x * 1.02)
        despline()
    elif mode is 'velocity':
        kwarg_plot = {"alpha": 0.5, "s": 8, "edgecolor": "0.8", "lw": 0.15}
        kwarg_plot.update(kwargs)

        x, y = adata.obsm['X_' + basis][:, dims[0]], adata.obsm['X_' + basis][:, dims[1]]

        if gs is None:
            fig = plt.figure(figsize=(10, 10))
            plt.subplot(111)
        else:
            plt.subplot(gs)

        if Vkey is 'U':
            V_vec = adata.layer['velocity_U'][:, ix]
        elif Vkey is 'S':
            V_vec = adata.layer['velocity_S'][:, ix]
        elif Vkey is 'P':
            V_vec = adata.layer['velocity_P'][:, ix]

        if (np.abs(V_vec) > 0.00005).sum() < 10:  # If S vs U scatterplot it is flat
            print("S vs U scatterplot it is flat")
            return
        limit = np.max(np.abs(np.percentile(V_vec, [1, 99])))  # upper and lowe limit / saturation
        V_vec = V_vec + limit  # that is: tmp_colorandum - (-limit)
        V_vec = V_vec / (2 * limit)  # that is: tmp_colorandum / (limit - (-limit))
        V_vec = np.clip(V_vec, 0, 1)

        plt.scatter(x, y, rasterized=True, c=np.reshape(cmap(V_vec), x.shape), **kwarg_plot)
        plt.axis("off")
        plt.title(f"{gene_names}")
    elif mode is 'expression':
        kwarg_plot = {"alpha": 0.15, "s": 8, "edgecolor": "0.8", "lw": 0.15}
        kwarg_plot.update(kwargs)
        if gs is None:
            fig = plt.figure(figsize=(10, 10))
            plt.subplot(111)
        else:
            plt.subplot(gs)

        x, y = adata.obsm['X_' + basis][:, dims[0]], adata.obsm['X_' + basis][:, dims[1]]
        if use_raw:
            if Ekey == 'X':
                E_vec = adata.X[:, ix]
            else:
                E_vec = adata.layers[Ekey][:, ix]
        else:
            if Ekey == 'X':
                E_vec = adata.X[:, ix]
            else:
                E_vec = adata.layers['X_' + Ekey][:, ix]

        E_vec = E_vec / np.percentile(E_vec, 99)
        # tmp_colorandum = np.log2(tmp_colorandum+1)
        E_vec = np.clip(E_vec, 0, 1)

        color_labels = color.unique()
        flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        #         sns.palplot(sns.color_palette(flatui))

        rgb_values = sns.color_palette(flatui, len(color_labels))  # viridis

        # Map label to RGB
        color_map = pd.DataFrame(zip(color_labels, rgb_values), index=color_labels)

        # ax.scatter(cur_pd.iloc[:, 0], cur_pd.iloc[:, 1], c=color_map.loc[E_vec, 1].values, **scatter_kwargs)
        df = pd.DataFrame(
            {"x": x, "y": y},  # 'gene': np.repeat(np.array(genes), n_cells), "expression": E_vec},
            index=range(x.shape[0]))

        ax = plt.scatter(x, y, rasterized=True, c=color_map.loc[color, 1].values,
                         **kwarg_plot)  # , label=color.tolist()
        plt.legend(loc='best')  # bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.

        if label_on_embedding:
            for i in color_labels:
                color_cnt = np.nanmedian(df.iloc[np.where(color == i)[0], :2], 0)
                txt = plt.text(color_cnt[0], color_cnt[1], str(i),
                               fontsize=13, c=color_map.loc[i, 1], bbox={"facecolor": "w", "alpha": 0.6})  #
                txt.set_path_effects([
                    PathEffects.Stroke(linewidth=5, foreground="w", alpha=0.1),
                    PathEffects.Normal()])

        #         plt.legend((p1[0]), (header[0]), fontsize=12, ncol=1, framealpha=0, fancybox=True)
        #         plt.axis("off")
        plt.title(f"{gene_names}")

    plt.show()
