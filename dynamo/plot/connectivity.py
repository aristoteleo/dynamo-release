# code adapted from https://github.com/lmcinnes/umap/blob/7e051d8f3c4adca90ca81eb45f6a9d1372c076cf/umap/plot.py
# need to integrate with DDRTree and possibly GRN learnt from Scribe.


def _plt_connectivity(Y, connectivity):
        """

        Parameters
        ----------
        principal_g_transition

        Returns
        -------

        """

        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.from_numpy_matrix(connectivity, create_using=nx.DiGraph())
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

        nx.draw(G, pos=Y, with_labels=False, node_color='skyblue', node_size=1,
                edge_color=edge_color, width=W / np.max(W) * 1, edge_cmap=plt.cm.Blues, options=options)

        plt.show()


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
    import datashader as ds
    import datashader.transfer_functions as tf
    import datashader.bundling as bd

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


def connectivity(
        adata,
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


