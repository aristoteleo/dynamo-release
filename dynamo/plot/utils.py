import os
import numpy as np
import pandas as pd
import math
import numba
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
import matplotlib.tri as tri
from scipy.spatial import Delaunay
from warnings import warn

from ..configuration import _themes
from ..tools.utils import integrate_vf  # integrate_vf_ivp


# ---------------------------------------------------------------------------------------------------
# variable checking utilities
def is_gene_name(adata, var):
    return var in adata.var.index


def is_cell_anno_column(adata, var):
    return var in adata.obs.columns


def is_layer_keys(adata, var):
    return var in adata.layers.keys()


def is_list_of_lists(list_of_lists):
    all(isinstance(elem, list) for elem in list_of_lists)


# ---------------------------------------------------------------------------------------------------
# plotting utilities that borrowed from umap
# link: https://github.com/lmcinnes/umap/blob/7e051d8f3c4adca90ca81eb45f6a9d1372c076cf/umap/plot.py

def map2color(val, min=None, max=None, cmap='viridis'):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    minima = np.min(val) if min is None else min
    maxima = np.max(val) if max is None else max

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))

    cols = [mapper.to_rgba(v) for v in val]

    return cols


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
    if background in ["k", "black"]:
        font_color = "white"
    elif background in ["w", "white"]:
        font_color = "black"
    elif background.startswith("#"):
        mean_val = np.mean(
            # specify 0 as the base in order to invoke this prefix-guessing behavior;
            # omitting it means to assume base-10
            [
                int("0x" + c, 0)
                for c in (background[1:3], background[3:5], background[5:7])
            ]
        )
        if mean_val > 126:
            font_color = "black"
        else:
            font_color = "white"

    else:
        font_color = "black"

    return font_color


def _matplotlib_points(
        points,
        ax=None,
        labels=None,
        values=None,
        highlights=None,
        cmap="Blues",
        color_key=None,
        color_key_cmap="Spectral",
        background="white",
        width=7,
        height=5,
        show_legend=True,
        vmin=2,
        vmax=98,
        sort='raw',
        frontier=False,
        contour=False,
        ccmap=None,
        calpha=2.3,
        sym_c=False,
        inset_dict={},
        **kwargs,
):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    dpi = plt.rcParams["figure.dpi"]
    width, height = width * dpi, height * dpi

    # """Use matplotlib to plot points"""
    # point_size = 500.0 / np.sqrt(points.shape[0])

    legend_elements = None

    if ax is None:
        dpi = plt.rcParams["figure.dpi"]
        fig = plt.figure(figsize=(width / dpi, height / dpi))
        ax = fig.add_subplot(111)

    ax.set_facecolor(background)

    # Color by labels
    unique_labels = []

    if labels is not None:
        if labels.shape[0] != points.shape[0]:
            raise ValueError(
                "Labels must have a label for "
                "each sample (size mismatch: {} {})".format(
                    labels.shape[0], points.shape[0]
                )
            )
        if color_key is None:
            if highlights is None:
                unique_labels = np.unique(labels)
                num_labels = unique_labels.shape[0]
                color_key = plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
            else:
                if type(highlights) is str: highlights = [highlights]
                highlights.append("other")
                unique_labels = np.array(highlights)
                num_labels = unique_labels.shape[0]
                color_key = _to_hex(
                    plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
                )
                color_key[
                    -1
                ] = "#bdbdbd"  # lightgray hex code https://www.color-hex.com/color/d3d3d3

                labels[[i not in highlights for i in labels]] = "other"
                points = pd.DataFrame(points)
                points["label"] = pd.Categorical(labels)

                # reorder data so that highlighting points will be on top of background points
                highlight_ids, background_ids = (
                    points["label"] != "other",
                    points["label"] == "other",
                )
                reorder_data = points.copy(deep=True)
                (
                    reorder_data.loc[:sum(background_ids), :],
                    reorder_data.loc[sum(background_ids):, :],
                ) = (points.loc[background_ids, :], points.loc[highlight_ids, :])

                points = reorder_data.values

        if isinstance(color_key, dict):
            colors = pd.Series(labels).map(color_key).values
            unique_labels = np.unique(labels)
            legend_elements = [
                # Patch(facecolor=color_key[k], label=k) for k in unique_labels
                Line2D(
                    [0], [0], marker="o", color=color_key[k], label=k, linestyle="None"
                )
                for k in unique_labels
            ]
        else:
            unique_labels = np.unique(labels)
            if len(color_key) < unique_labels.shape[0]:
                raise ValueError(
                    "Color key must have enough colors for the number of labels"
                )

            new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
            legend_elements = [
                # Patch(facecolor=color_key[i], label=k)
                Line2D(
                    [0], [0], marker="o", color=color_key[i], label=k, linestyle="None"
                )
                for i, k in enumerate(unique_labels)
            ]
            colors = pd.Series(labels).map(new_color_key)

        if frontier:
            rasterized = kwargs['rasterized'] if 'rasterized' in kwargs.keys() else None
            ax.scatter(points[:, 0], points[:, 1], kwargs['s'] * 2, "0.0", lw=2, rasterized=rasterized)
            ax.scatter(points[:, 0], points[:, 1], kwargs['s'] * 2, "1.0", lw=0, rasterized=rasterized)
            ax.scatter(points[:, 0], points[:, 1], c=colors, **kwargs)
        elif contour:
            try:
                from shapely.geometry import Polygon, MultiPoint, Point
            except ImportError:
                raise ImportError(
                    "If you want to use the tricontourf in plotting function, you need to install `shapely` "
                    "package via `pip install shapely` see more details at https://pypi.org/project/Shapely/,"
                )

            x, y = points[:, :2].T
            triang = tri.Triangulation(x, y)
            concave_hull, edge_points = alpha_shape(x, y, alpha=calpha)
            ax = plot_polygon(concave_hull, ax=ax)

            # Use the mean distance between the triangulated x & y poitns
            x2 = x[triang.triangles].mean(axis=1)
            y2 = y[triang.triangles].mean(axis=1)
            ##note the very obscure mean command, which, if not present causes an error.
            ##now we need some masking condition.

            # Create an empty set to fill with zeros and ones
            cond = np.empty(len(x2))
            # iterate through points checking if the point lies within the polygon
            for i in range(len(x2)):
                cond[i] = concave_hull.contains(Point(x2[i], y2[i]))

            mask = np.where(cond, 0, 1)
            # apply masking
            triang.set_mask(mask)

            ccmap = 'viridis' if ccmap is None else ccmap
            # ax.tricontourf(triang, values, cmap=ccmap)
            ax.scatter(x, y,
                       c=values,
                       cmap=cmap,
                       **kwargs, )
        else:
            ax.scatter(points[:, 0], points[:, 1], c=colors, **kwargs)

    # Color by values
    elif values is not None:
        if values.shape[0] != points.shape[0]:
            raise ValueError(
                "Values must have a value for "
                "each sample (size mismatch: {} {})".format(
                    values.shape[0], points.shape[0]
                )
            )
        # reorder data so that high values points will be on top of background points
        sorted_id = np.argsort(abs(values)) if sort == 'abs' else np.argsort(- values) if sort == 'neg' \
            else np.argsort(values)
        values, points = values[sorted_id], points[sorted_id, :]

        # if there are very few cells have expression, set the vmin/vmax only based on positive values
        if np.nanmin(values) == 0:
            n_pos_cells = sum(values > 0)
            if 0 < n_pos_cells / len(values) < 0.02:
                vmin = 0 if n_pos_cells == 1 else np.percentile(values[values > 0], 2)
                vmax = np.nanmax(values) if n_pos_cells == 1 else np.percentile(values[values > 0], 98)
                if vmin + vmax in [1, 100]:
                    vmin += 1e-12
                    vmax += 1e-12

        # if None: min/max from data
        # if positive and sum up to 1, take fraction
        # if positive and sum up to 100, take percentage
        # otherwise take the data
        _vmin = np.nanmin(values) if vmin is None else np.nanpercentile(values, vmin * 100) if \
            (vmin + vmax == 1 and 0 <= vmin < vmax) else np.nanpercentile(values, vmin) \
            if (vmin + vmax == 100 and 0 <= vmin < vmax) else vmin
        _vmax = np.nanmax(values) if vmax is None else np.nanpercentile(values, vmax * 100) if \
            (vmin + vmax == 1 and 0 <= vmin < vmax) else np.nanpercentile(values, vmax) \
            if (vmin + vmax == 100 and 0 <= vmin < vmax) else vmax

        if sym_c and _vmin < 0 and _vmax > 0:
            bounds = np.nanmax([np.abs(_vmin), _vmax])
            bounds = bounds * np.array([-1, 1])
            _vmin, _vmax = bounds

        if frontier == True:
            rasterized = kwargs['rasterized'] if 'rasterized' in kwargs.keys() else None
            ax.scatter(points[:, 0], points[:, 1], kwargs['s'] * 2, "0.0", lw=2, rasterized=rasterized)
            ax.scatter(points[:, 0], points[:, 1], kwargs['s'] * 2, "1.0", lw=0, rasterized=rasterized)
            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=values,
                cmap=cmap,
                vmin=_vmin,
                vmax=_vmax,
                **kwargs,
            )
        elif contour:
            try:
                from shapely.geometry import Polygon, MultiPoint, Point
            except ImportError:
                raise ImportError(
                    "If you want to use the tricontourf in plotting function, you need to install `shapely` "
                    "package via `pip install shapely` see more details at https://pypi.org/project/Shapely/,"
                    )

            x, y = points[:, :2].T
            triang = tri.Triangulation(x, y)
            concave_hull, edge_points = alpha_shape(x, y, alpha=calpha)
            ax = plot_polygon(concave_hull, ax=ax)

            # Use the mean distance between the triangulated x & y poitns
            x2 = x[triang.triangles].mean(axis=1)
            y2 = y[triang.triangles].mean(axis=1)
            ##note the very obscure mean command, which, if not present causes an error.
            ##now we need some masking condition.

            # Create an empty set to fill with zeros and ones
            cond = np.empty(len(x2))
            # iterate through points checking if the point lies within the polygon
            for i in range(len(x2)):
                cond[i] = concave_hull.contains(Point(x2[i], y2[i]))

            mask = np.where(cond, 0, 1)
            # apply masking
            triang.set_mask(mask)

            ccmap = cmap if ccmap is None else ccmap

            ax.tricontourf(triang, values, cmap=ccmap)
            ax.scatter(x, y,
                       c=values,
                       cmap=cmap,
                       vmin=_vmin,
                       vmax=_vmax,
                       **kwargs,)
        else:
            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=values,
                cmap=cmap,
                vmin=_vmin,
                vmax=_vmax,
                **kwargs,
            )

        if 'norm' in kwargs:
            norm = kwargs['norm']
        else:
            norm = matplotlib.colors.Normalize(vmin=_vmin, vmax=_vmax)

        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(values)
        cb = plt.colorbar(mappable, cax=set_colorbar(ax, inset_dict), ax=ax)
        cb.set_alpha(1)
        cb.draw_all()
        cb.locator = MaxNLocator(nbins=3, integer=True)
        cb.update_ticks()

        cmap = matplotlib.cm.get_cmap(cmap)
        colors = cmap(values)

    # No color (just pick the midpoint of the cmap)
    else:
        colors = plt.get_cmap(cmap)(0.5)
        ax.scatter(points[:, 0], points[:, 1], c=colors, **kwargs)

    if show_legend and legend_elements is not None:
        if len(unique_labels) > 1 and show_legend == "on data":
            font_color = "white" if background in ["black", "#ffffff"] else "black"
            for i in unique_labels:
                if i == 'other':
                    continue
                color_cnt = np.nanmedian(points[np.where(labels == i)[0], :2].astype('float'), 0)
                txt = plt.text(
                    color_cnt[0],
                    color_cnt[1],
                    str(i),
                    color=_select_font_color(font_color),
                    zorder=1000,
                    verticalalignment="center",
                    horizontalalignment="center",
                    weight="bold",
                )  #
                txt.set_path_effects(
                    [
                        PathEffects.Stroke(linewidth=1.5, foreground=font_color, alpha=0.8),
                        PathEffects.Normal(),
                    ]
                )
        else:
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.04, 1),
                loc=show_legend,
                ncol=len(unique_labels) // 15 + 1,
            )

    return ax, colors


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
        width=7,
        height=5,
        show_legend=True,
        vmin=2,
        vmax=98,
        sort='raw',
        **kwargs,
):
    import matplotlib.pyplot as plt

    import datashader.transfer_functions as tf
    import datashader as ds

    dpi = plt.rcParams["figure.dpi"]
    width, height = width * dpi, height * dpi

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

        labels = np.array(labels, dtype="str")
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
                highlights.append("other")
                unique_labels = np.array(highlights)
                num_labels = unique_labels.shape[0]
                color_key = _to_hex(
                    plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
                )
                color_key[
                    -1
                ] = "#bdbdbd"  # lightgray hex code https://www.color-hex.com/color/d3d3d3

                labels[[i not in highlights for i in labels]] = "other"
                data["label"] = pd.Categorical(labels)

                # reorder data so that highlighting points will be on top of background points
                highlight_ids, background_ids = (
                    data["label"] != "other",
                    data["label"] == "other",
                )
                reorder_data = data.copy(deep=True)
                (
                    reorder_data.iloc[: sum(background_ids), :],
                    reorder_data.iloc[sum(background_ids):, :],
                ) = (data.iloc[background_ids, :], data.iloc[highlight_ids, :])
                aggregation = canvas.points(
                    reorder_data, "x", "y", agg=ds.count_cat("label")
                )

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
        # reorder data so that high values data will be on top of background data
        sorted_id = np.argsort(abs(values)) if sort == 'abs' else np.argsort(values)
        values, data = values[sorted_id], data.iloc[sorted_id, :]

        values[np.isnan(values)] = 0
        _vmin = np.min(values) if vmin is None else np.percentile(values, vmin)
        _vmax = np.min(values) if vmin is None else np.percentile(values, vmax)

        values = np.clip(values, _vmin, _vmax)

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
            if len(unique_labels) > 1 and show_legend == "on data":
                font_color = "white" if background == "black" else "black"
                for i in unique_labels:
                    color_cnt = np.nanmedian(
                        points.iloc[np.where(labels == i)[0], :2], 0
                    )
                    txt = plt.text(
                        color_cnt[0],
                        color_cnt[1],
                        str(i),
                        color=_select_font_color(font_color),
                        zorder=1000,
                        verticalalignment="center",
                        horizontalalignment="center",
                        weight="bold",
                    )  #
                    txt.set_path_effects(
                        [
                            PathEffects.Stroke(linewidth=1.5, foreground=font_color, alpha=0.8),
                            PathEffects.Normal(),
                        ]
                    )
            else:
                if type(show_legend) == "str":
                    ax.legend(
                        handles=legend_elements,
                        loc=show_legend,
                        ncol=len(unique_labels) // 15 + 1,
                    )
                else:
                    ax.legend(
                        handles=legend_elements,
                        loc="best",
                        ncol=len(unique_labels) // 15 + 1,
                    )
        return ax
    else:
        return result


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
        width=7,
        height=5,
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
    import bokeh.plotting as bpl
    import bokeh.transform as btr
    from bokeh.plotting import output_notebook, output_file, show
    import datashader as ds

    import holoviews as hv
    import holoviews.operation.datashader as hd
    import matplotlib.pyplot as plt

    dpi = plt.rcParams["figure.dpi"]
    width, height = width * dpi, height * dpi

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


# ---------------------------------------------------------------------------------------------------
# plotting utilities borrow from velocyto
# link - https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb


def despline(ax=None):
    import matplotlib.pyplot as plt

    ax = plt.gca() if ax is None else ax
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


def despline_all(ax=None):
    # removing the default axis on all sides:
    import matplotlib.pyplot as plt

    ax = plt.gca() if ax is None else ax

    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)


def deaxis_all(ax=None):
    # removing the axis ticks
    import matplotlib.pyplot as plt

    ax = plt.gca() if ax is None else ax

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def minimal_xticks(start, end):
    import matplotlib.pyplot as plt

    end_ = np.around(end, -int(np.log10(end)) + 1)
    xlims = np.linspace(start, end_, 5)
    xlims_tx = [""] * len(xlims)
    xlims_tx[0], xlims_tx[-1] = f"{xlims[0]:.0f}", f"{xlims[-1]:.02f}"
    plt.xticks(xlims, xlims_tx)


def minimal_yticks(start, end):
    import matplotlib.pyplot as plt

    end_ = np.around(end, -int(np.log10(end)) + 1)
    ylims = np.linspace(start, end_, 5)
    ylims_tx = [""] * len(ylims)
    ylims_tx[0], ylims_tx[-1] = f"{ylims[0]:.0f}", f"{ylims[-1]:.02f}"
    plt.yticks(ylims, ylims_tx)


def set_spine_linewidth(ax, lw):
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(lw)

    return ax


# ---------------------------------------------------------------------------------------------------
# scatter plot utilities


def scatter_with_colorbar(fig, ax, x, y, c, cmap, **scatter_kwargs):
    # https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    g = ax.scatter(x, y, c=c, cmap=cmap, **scatter_kwargs)
    fig.colorbar(g, cax=cax, orientation="vertical")

    return fig, ax


def scatter_with_legend(
        fig, ax, df, font_color, x, y, c, cmap, legend, **scatter_kwargs
):
    import seaborn as sns
    import matplotlib.patheffects as PathEffects

    unique_labels = np.unique(c)

    if legend == "on data":
        g = sns.scatterplot(
            x, y, hue=c, palette=cmap, ax=ax, legend=False, **scatter_kwargs
        )

        for i in unique_labels:
            color_cnt = np.nanmedian(df.iloc[np.where(c == i)[0], :2], 0)
            txt = ax.text(
                color_cnt[0],
                color_cnt[1],
                str(i),
                color=font_color,
                zorder=1000,
                verticalalignment="center",
                horizontalalignment="center",
                weight="bold",
            )  # c
            txt.set_path_effects(
                [
                    PathEffects.Stroke(
                        linewidth=1.5, foreground=font_color, alpha=0.8
                    ),  # 'w'
                    PathEffects.Normal(),
                ]
            )
    else:
        g = sns.scatterplot(
            x, y, hue=c, palette=cmap, ax=ax, legend="full", **scatter_kwargs
        )
        ax.legend(loc=legend, ncol=unique_labels // 15)

    return fig, ax

def set_colorbar(ax, inset_dict={}):
    """https://matplotlib.org/3.1.0/gallery/axes_grid1/demo_colorbar_with_inset_locator.html"""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if len(inset_dict) == 0:
        # see more at https://matplotlib.org/gallery/axes_grid1/inset_locator_demo.html
        axins = inset_axes(ax,
                           width="12%",  # width = 5% of parent_bbox width
                           height="100%",  # height : 50%
                           loc='upper right',
                           bbox_to_anchor=(0.85, 0.97, 0.145, 0.17),
                           bbox_transform=ax.transAxes,
                           borderpad=1.85,
                           )
    else:
        axins = inset_axes(ax, bbox_transform=ax.transAxes, **inset_dict)

    return axins


def arrowed_spines(ax, columns, background='white'):
    """https://stackoverflow.com/questions/33737736/matplotlib-axis-arrow-tip
        modified based on Answer 6
    """
    if type(columns) == str: columns = [columns.upper() + ' 0', columns.upper() + ' 1']
    import matplotlib.pyplot as plt
    fig = plt.gcf()

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    despline_all(ax)

    # removing the axis ticks
    deaxis_all(ax)

    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length (x-axis)
    hw = 1./20.*(ymax-ymin)
    hl = 1./20.*(xmax-xmin)
    lw = 1. # axis line width
    ohg = 0.2 # arrow overhang

    # compute matching arrowhead length and width (y-axis)
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    fc, ec = ("w", "w") if background in ['black', "#ffffff"] else ('k', 'k')
    ax.arrow(xmin, ymin, hl * 5/2, 0, fc=fc, ec=ec,
             lw=lw,
             head_width=hw/2, head_length=hl/2,
             overhang=ohg/2,
             length_includes_head=True, clip_on=False)
    ax.arrow(xmin, ymin, 0, hw * 5/2, fc=fc, ec=ec,
             lw=lw,
             head_width=yhw/2, head_length=yhl/2,
             overhang=ohg/2,
             length_includes_head=True, clip_on=False)

    ax.text(xmin + hl * 2.5/2, ymin - 1.5 * hw/2, columns[0], ha="center", va="center", rotation=0,
            # size=hl * 5 / (2 * len(str(columns[0]))) * 20,
            # size=matplotlib.rcParams['axes.titlesize'],
            size=np.clip((hl + yhw) * 8 / 2, 6, 18)
            )
    ax.text(xmin - 1.5 * yhw/2, ymin + hw * 2.5/2, columns[1], ha="center", va="center", rotation=90,
            # size=hw * 5 / (2 * len(str(columns[1]))) * 20,
            # size=matplotlib.rcParams['axes.titlesize'],
            size=np.clip((hl + yhw) * 8 / 2, 6, 18)
            )

    return ax

# ---------------------------------------------------------------------------------------------------
# vector field plot related utilities


def quiver_autoscaler(X_emb, V_emb):
    """Function to automatically calculate the value for the scale parameter of quiver plot, adapted from scVelo

    Parameters
    ----------
        X_emb: `np.ndarray`
            X, Y-axis coordinates
        V_emb:  `np.ndarray`
            Velocity (U, V) values on the X, Y-axis

    Returns
    -------
        The scale for quiver plot
    """

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    scale_factor = np.ptp(X_emb, 0).mean()
    X_emb = X_emb - X_emb.min(0)

    if len(V_emb.shape) == 3:
        Q = ax.quiver(
            X_emb[0] / scale_factor,
            X_emb[1] / scale_factor,
            V_emb[0],
            V_emb[1],
            angles="xy",
            scale_units="xy",
            scale=None,
        )
    else:
        Q = ax.quiver(
            X_emb[:, 0] / scale_factor,
            X_emb[:, 1] / scale_factor,
            V_emb[:, 0],
            V_emb[:, 1],
            angles="xy",
            scale_units="xy",
            scale=None,
        )

    Q._init()
    fig.clf()
    plt.close(fig)

    return Q.scale / scale_factor * 2


def default_quiver_args(arrow_size, arrow_len=None):
    if isinstance(arrow_size, (list, tuple)) and len(arrow_size) == 3:
        head_w, head_l, ax_l = arrow_size
    elif type(arrow_size) in [int, float]:
        head_w, head_l, ax_l = 10 * arrow_size, 12 * arrow_size, 8 * arrow_size
    else:
        head_w, head_l, ax_l = 10, 12, 8

    scale = 1 / arrow_len if arrow_len is not None else 1 / arrow_size

    return head_w, head_l, ax_l, scale


# ---------------------------------------------------------------------------------------------------
def _plot_traj(y0, t, args, integration_direction, ax, color, lw, f):
    _, y = integrate_vf(y0, t, args, integration_direction, f)  # integrate_vf_ivp

    ax.plot(*y.transpose(), color=color, lw=lw, linestyle="dashed", alpha=0.5)

    ax.scatter(*y0.transpose(), color=color, marker="*")

    return ax


# ---------------------------------------------------------------------------------------------------
# streamline related aesthetics
# ---------------------------------------------------------------------------------------------------

def set_arrow_alpha(ax=None, alpha=1):
    from matplotlib import patches
    ax = plt.gca() if ax is None else ax

    # iterate through the children of ax
    for art in ax.get_children():
        # we are only interested in FancyArrowPatches
        if not isinstance(art, patches.FancyArrowPatch):
            continue
        art.set_alpha(alpha)

def set_stream_line_alpha(s=None, alpha=1):
    """s has to be a StreamplotSet"""
    s.lines.set_alpha(alpha)

# ---------------------------------------------------------------------------------------------------
# save_fig figure related
# ---------------------------------------------------------------------------------------------------

def save_fig(path=None, prefix=None, dpi=None, ext='pdf', transparent=True, close=True, verbose=True):
    """Save a figure from pyplot.
    code adapated from http://www.jesshamrick.com/2012/09/03/saving-figures-from-pyplot/

    Parameters
    ----------
         path: `string`
            The path (and filename, without the extension) to save_fig the
            figure to.
        prefix: `str` or `None`
            The prefix added to the figure name. This will be automatically set
            accordingly to the plotting function used.
        dpi: [ None | scalar > 0 | 'figure' ]
            The resolution in dots per inch. If None, defaults to rcParams["savefig.dpi"].
            If 'figure', uses the figure's dpi value.
        ext: `string` (default='pdf')
            The file extension. This must be supported by the active
            matplotlib backend (see matplotlib.backends module).  Most
            backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
        close: `boolean` (default=True)
            Whether to close the figure after saving.  If you want to save_fig
            the figure multiple times (e.g., to multiple formats), you
            should NOT close it in between saves or you will have to
            re-plot it.
        verbose: boolean (default=True)
            Whether to print information about when and where the image
            has been saved.
    """
    import matplotlib.pyplot as plt

    if path is None: path = os.getcwd() + '/'

    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = os.path.split(path)[1]
    if directory == '':
        directory = '.'
    if filename == '':
        filename = 'dyn_savefig'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save_fig to
    savepath = os.path.join(directory, filename + '.' + ext) if prefix is None \
        else os.path.join(directory, prefix + '_' + filename + '.' + ext)

    if verbose:
        print(f"Saving figure to {savepath}...")

    # Actually save the figure
    plt.savefig(savepath, dpi=dpi, transparent=transparent, format=ext)

    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")


# ---------------------------------------------------------------------------------------------------
def alpha_shape(x, y, alpha):
    # Start Using SHAPELY
    try:
        import shapely.geometry as geometry
        from shapely.geometry import Polygon, MultiPoint, Point
        from shapely.ops import triangulate
        from shapely.ops import cascaded_union, polygonize
    except ImportError:
        raise ImportError("If you want to use the tricontourf in plotting function, you need to install `shapely` "
                          "package via `pip install shapely` see more details at https://pypi.org/project/Shapely/,"
                          )

    crds = np.array([x.flatten(), y.flatten()]).transpose()
    points = MultiPoint(crds)

    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    coords = np.array([point.coords[0]
                       for point in points])

    tri = Delaunay(coords)
    edges = set()
    edge_points = []

    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

        # Semiperimeter of triangle
        s = (a + b + c) / 2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)

        # Here's the radius filter.
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))

    return cascaded_union(triangles), edge_points


# View the polygon and adjust alpha if needed
def plot_polygon(polygon,
                 margin=1,
                 fc='#999999',
                 ec='#000000',
                 fill=True,
                 ax=None,
                 **kwargs):
    try:
        from descartes.patch import PolygonPatch
    except ImportError:
        raise ImportError("If you want to use the tricontourf in plotting function, you need to install `descartes` "
                          "package via `pip install descartes` see more details at https://pypi.org/project/descartes/,"
                          )


    from descartes.patch import PolygonPatch
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    margin = margin
    x_min, y_min, x_max, y_max = polygon.bounds
    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    patch = PolygonPatch(polygon, fc=fc, ec=ec, fill=fill, zorder=-1, lw=3, alpha=0.4, **kwargs)
    ax.add_patch(patch)

    return ax

# ---------------------------------------------------------------------------------------------------
# the following Loess class is taken from:
# link: https://github.com/joaofig/pyloess/blob/master/pyloess/Loess.py


def tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= -1) & (x <= 1)
    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
    return y


class Loess(object):
    @staticmethod
    def normalize_array(array):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val), min_val, max_val

    def __init__(self, xx, yy, degree=1):
        self.n_xx, self.min_xx, self.max_xx = self.normalize_array(xx)
        self.n_yy, self.min_yy, self.max_yy = self.normalize_array(yy)
        self.degree = degree

    @staticmethod
    def get_min_range(distances, window):
        min_idx = np.argmin(distances)
        n = len(distances)
        if min_idx == 0:
            return np.arange(0, window)
        if min_idx == n - 1:
            return np.arange(n - window, n)

        min_range = [min_idx]
        while len(min_range) < window:
            i0 = min_range[0]
            i1 = min_range[-1]
            if i0 == 0:
                min_range.append(i1 + 1)
            elif i1 == n - 1:
                min_range.insert(0, i0 - 1)
            elif distances[i0 - 1] < distances[i1 + 1]:
                min_range.insert(0, i0 - 1)
            else:
                min_range.append(i1 + 1)
        return np.array(min_range)

    @staticmethod
    def get_weights(distances, min_range):
        max_distance = np.max(distances[min_range])
        weights = tricubic(distances[min_range] / max_distance)
        return weights

    def normalize_x(self, value):
        return (value - self.min_xx) / (self.max_xx - self.min_xx)

    def denormalize_y(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy

    def estimate(self, x, window, use_matrix=False, degree=1):
        n_x = self.normalize_x(x)
        distances = np.abs(self.n_xx - n_x)
        min_range = self.get_min_range(distances, window)
        weights = self.get_weights(distances, min_range)

        if use_matrix or degree > 1:
            wm = np.multiply(np.eye(window), weights)
            xm = np.ones((window, degree + 1))

            xp = np.array([[math.pow(n_x, p)] for p in range(degree + 1)])
            for i in range(1, degree + 1):
                xm[:, i] = np.power(self.n_xx[min_range], i)

            ym = self.n_yy[min_range]
            xmt_wm = np.transpose(xm) @ wm
            beta = np.linalg.pinv(xmt_wm @ xm) @ xmt_wm @ ym
            y = (beta @ xp)[0]
        else:
            xx = self.n_xx[min_range]
            yy = self.n_yy[min_range]
            sum_weight = np.sum(weights)
            sum_weight_x = np.dot(xx, weights)
            sum_weight_y = np.dot(yy, weights)
            sum_weight_x2 = np.dot(np.multiply(xx, xx), weights)
            sum_weight_xy = np.dot(np.multiply(xx, yy), weights)

            mean_x = sum_weight_x / sum_weight
            mean_y = sum_weight_y / sum_weight

            b = (sum_weight_xy - mean_x * mean_y * sum_weight) / (
                    sum_weight_x2 - mean_x * mean_x * sum_weight
            )
            a = mean_y - b * mean_x
            y = a + b * n_x
        return self.denormalize_y(y)
