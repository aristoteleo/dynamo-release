import numpy as np, pandas as pd
from ..tools.utils import update_dict, index_gene, flatten
from .utils import save_fig
from .utils_graph import ArcPlot


def nxvizPlot(adata,
              cluster,
              cluster_name,
              edges_list,
              plot='arcplot',
              network=None,
              weight_scale=5e3,
              figsize=(6, 6),
              save_show_or_return='show',
              save_kwargs={},
              **kwargs,
              ):
    """Arc or circos plot of gene regulatory network for a particular cell cluster.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`.
            AnnData object.
        cluster: `str`
            The group key that points to the columns of `adata.obs`.
        cluster_name: `str` (default: `None`)
            The group whose network and arcplot will be constructed and created.
        edges_list: `dict` of `pandas.DataFrame`
            A dictionary of dataframe of interactions between input genes for each group of cells based on ranking
            information of Jacobian analysis. Each composite dataframe has `regulator`, `target` and `weight` three columns.
        plot: `str` (default: `arcplot`)
            Which nxviz plot to use, one of {'arcplot', 'circosplot'}.
        network: class:`~networkx.classes.digraph.DiGraph`
            A direct network for this cluster constructed based on Jacobian analysis.
        weight_scale: `float` (default: `1e3`)
            Because values in Jacobian matrix is often small, the value will be multiplied by the weight_scale so that
            the edge will have proper width in display.
        figsize: `None` or `[float, float]` (default: (6, 6)
            The width and height of each panel in the figure.
        save_show_or_return: `str` {'save', 'show', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig
            function will use the {"path": None, "prefix": 'arcplot', "dpi": None, "ext": 'pdf', "transparent": True,
            "close": True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly
            modify those keys according to your needs.
        **kwargs:
            Additional parameters that will pass to ArcPlot or CircosPlot

    Returns
    -------
        Nothing but plot an ArcPlot of the input direct network.
    """

    import matplotlib.pyplot as plt
    try:
        import networkx as nx
        import nxviz as nv
    except ImportError:
        raise ImportError(f"You need to install the packages `networkx, nxviz`."
                          f"install networkx via `pip install networkx`."
                          f"install nxviz via `pip install nxviz`.")

    if edges_list is not None:
        network = nx.from_pandas_edgelist(edges_list[cluster], 'regulator', 'target', edge_attr='weight',
                                          create_using=nx.DiGraph())

    # Iterate over all the nodes in G, including the metadata
    if type(cluster_name) is str: cluster_names = [cluster_name]
    for n, d in network.nodes(data=True):
        # Calculate the degree of each node: G.node[n]['degree']
        network.nodes[n]['degree'] = nx.degree(network, n)
        # data has to be float
        network.nodes[n]['size'] = adata[adata.obs[cluster].isin(cluster_names), n].layers['M_s'].A.mean().astype(float)
        network.nodes[n]['label'] = n
    for e in network.edges():
        network.edges[e]['weight'] *= weight_scale

    if plot.lower() == 'arcplot':
        prefix = 'arcPlot'
        # Create the customized ArcPlot object: a2
        nv_ax = nv.ArcPlot(network,
                           node_order=kwargs.pop('node_order', 'degree'),
                           node_size=kwargs.pop('node_size', None),
                           node_grouping=kwargs.pop('node_grouping', None),
                           group_order=kwargs.pop('group_order', 'alphabetically'),
                           node_color=kwargs.pop('node_color', 'size'),
                           node_labels=kwargs.pop('node_labels', True),
                           edge_width=kwargs.pop('edge_width', 'weight'),
                           edge_color=kwargs.pop('edge_color', None),
                           data_types=kwargs.pop('data_types', None),
                           nodeprops=kwargs.pop('nodeprops',
                                                {'facecolor': 'None', 'alpha': 0.2, 'cmap': 'viridis', 'label': 'label'}),
                           edgeprops=kwargs.pop('edgeprops', {'facecolor': 'None', 'alpha': 0.2}),
                           node_label_color=kwargs.pop('node_label_color', False),
                           group_label_position=kwargs.pop('group_label_position', None),
                           group_label_color=kwargs.pop('group_label_color', False),
                           fontsize=kwargs.pop('fontsize', 10),
                           fontfamily=kwargs.pop('fontfamily', "serif"),
                           figsize=figsize,
                           )
    elif plot.lower() == 'circosplot':
        prefix = 'circosPlot'
        # Create the customized CircosPlot object: a2
        nv_ax = nv.CircosPlot(network,
                              node_order=kwargs.pop('node_order', 'degree'),
                              node_size=kwargs.pop('node_size', None),
                              node_grouping=kwargs.pop('node_grouping', None),
                              group_order=kwargs.pop('group_order', 'alphabetically'),
                              node_color=kwargs.pop('node_color', 'size'),
                              node_labels=kwargs.pop('node_labels', True),
                              edge_width=kwargs.pop('edge_width', 'weight'),
                              edge_color=kwargs.pop('edge_color', None),
                              data_types=kwargs.pop('data_types', None),
                              nodeprops=kwargs.pop('nodeprops', None),
                              edgeprops=kwargs.pop('edgeprops', {'facecolor': 'None', 'alpha': 0.2}),
                              node_label_color=kwargs.pop('node_label_color', False),
                              group_label_position=kwargs.pop('group_label_position', None),
                              group_label_color=kwargs.pop('group_label_color', False),
                              fontsize=kwargs.pop('fontsize', 10),
                              fontfamily=kwargs.pop('fontfamily', "serif"),
                              figsize=figsize,
                              )

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": prefix, "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        # Draw a to the screen
        nv_ax.draw()
        plt.autoscale()
        # Display the plot
        plt.show()
        # plt.savefig('./unknown_arcplot.pdf', dpi=300)
    elif save_show_or_return == "return":
        return nv_ax


def arcPlot(adata,
            cluster,
            cluster_name,
            edges_list,
            network=None,
            color=None,
            node_size=100,
            cbar=True,
            cbar_shrink=0.6,
            cbar_text=None,
            figsize=(6, 6),
            save_show_or_return='show',
            save_kwargs={},
            **kwargs,
            ):
    """Arc plot of gene regulatory network for a particular cell cluster.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`.
            AnnData object.
        cluster: `str`
            The group key that points to the columns of `adata.obs`.
        cluster_name: `str` (default: `None`)
            The group whose network and arcplot will be constructed and created.
        edges_list: `dict` of `pandas.DataFrame`
            A dictionary of dataframe of interactions between input genes for each group of cells based on ranking
            information of Jacobian analysis. Each composite dataframe has `regulator`, `target` and `weight` three columns.
        network: class:`~networkx.classes.digraph.DiGraph`
            A direct network for this cluster constructed based on Jacobian analysis.
        weight_scale: `float` (default: `1e3`)
            Because values in Jacobian matrix is often small, the value will be multiplied by the weight_scale so that
            the edge will have proper width in display.
        figsize: `None` or `[float, float]` (default: (6, 6)
            The width and height of each panel in the figure.
        save_show_or_return: `str` {'save', 'show', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig
            function will use the {"path": None, "prefix": 'arcplot', "dpi": None, "ext": 'pdf', "transparent": True,
            "close": True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly
            modify those keys according to your needs.
        **kwargs:
            Additional parameters that will eventually pass to ArcPlot.

    Returns
    -------
        Nothing but plot an ArcPlot of the input direct network.
    """
    '''nxvizPlot(adata,
            cluster,
            cluster_name,
            edges_list,
            plot='arcplot',
            network=network,
            weight_scale=weight_scale,
            figsize=figsize,
            save_show_or_return=save_show_or_return,
            save_kwargs=save_kwargs,
            **kwargs,
            )'''
    import matplotlib.pyplot as plt
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(f"You need to install the package `networkx`."
                          f"install networkx via `pip install networkx`.")

    if edges_list is not None:
        network = nx.from_pandas_edgelist(edges_list[cluster], 'regulator', 'target', edge_attr='weight',
                                          create_using=nx.DiGraph())

    # Iterate over all the nodes in G, including the metadata
    if type(cluster_name) is str: cluster_names = [cluster_name]
    if type(color) is str and color in adata.layers.keys():
        data = adata[adata.obs[cluster].isin(cluster_names), :].layers[color]
        color = []
        for gene in network.nodes:
            c = np.mean(flatten(index_gene(adata, data, [gene])))
            color.append(c)
    
    ap = ArcPlot(network=network, c=color, s=node_size, **kwargs)
    ap.draw()
    
    if cbar:
        cbar = plt.colorbar(shrink=cbar_shrink)
        if cbar_text is not None:
            cbar.ax.set_ylabel(cbar_text, va='top')




def circosPlot(adata,
               cluster,
               cluster_name,
               edges_list,
               network=None,
               weight_scale=5e3,
               figsize=(12, 6),
               save_show_or_return='show',
               save_kwargs={},
               **kwargs,
               ):
    """Circos plot of gene regulatory network for a particular cell cluster.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`.
            AnnData object.
        cluster: `str`
            The group key that points to the columns of `adata.obs`.
        cluster_name: `str` (default: `None`)
            The group whose network and arcplot will be constructed and created.
        edges_list: `dict` of `pandas.DataFrame`
            A dictionary of dataframe of interactions between input genes for each group of cells based on ranking
            information of Jacobian analysis. Each composite dataframe has `regulator`, `target` and `weight` three columns.
        network: class:`~networkx.classes.digraph.DiGraph`
            A direct network for this cluster constructed based on Jacobian analysis.
        weight_scale: `float` (default: `1e3`)
            Because values in Jacobian matrix is often small, the value will be multiplied by the weight_scale so that
            the edge will have proper width in display.
        figsize: `None` or `[float, float]` (default: (12, 6)
            The width and height of each panel in the figure.
        save_show_or_return: `str` {'save', 'show', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig
            function will use the {"path": None, "prefix": 'arcplot', "dpi": None, "ext": 'pdf', "transparent": True,
            "close": True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly
            modify those keys according to your needs.
        **kwargs:
            Additional parameters that will eventually pass to CircosPlot.

    Returns
    -------
        Nothing but plot an CircosPlot of the input direct network.
    """
    nxvizPlot(adata,
              cluster,
              cluster_name,
              edges_list,
              plot='circosplot',
              network=network,
              weight_scale=weight_scale,
              figsize=figsize,
              save_show_or_return=save_show_or_return,
              save_kwargs=save_kwargs,
              **kwargs,
              )


def hivePlot(adata,
             edges_list,
             cluster,
             cluster_names=None,
             figsize=(6, 6),
             save_show_or_return='show',
             save_kwargs={},
             ):
    """Hive plot of cell cluster specific gene regulatory networks.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`.
            AnnData object.
        edges_list: `dict` of `pandas.DataFrame`
            A dictionary of dataframe of interactions between input genes for each group of cells based on ranking
            information of Jacobian analysis. Each composite dataframe has `regulator`, `target` and `weight` three columns.
        cluster: `str`
            The group key that points to the columns of `adata.obs`.
        cluster_names: `str` (default: `None`)
            The group whose network and arcplot will be constructed and created.
        figsize: `None` or `[float, float]` (default: (6, 6)
            The width and height of each panel in the figure.
        save_show_or_return: `str` {'save', 'show', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig
            function will use the {"path": None, "prefix": 'hiveplot', "dpi": None, "ext": 'pdf', "transparent": True,
            "close": True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly
            modify those keys according to your needs.

    Returns
    -------
        Nothing but plot a hive plot of the input cell cluster specific direct network.
    """

    # from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    try:
        import networkx as nx
        from hiveplotlib import Axis, Node, HivePlot
        from hiveplotlib.viz import axes_viz_mpl, node_viz_mpl, edge_viz_mpl
    except ImportError:
        raise ImportError(f"You need to install the package `networkx, hiveplotlib`."
                          f"install hiveplotlib via `pip install hiveplotlib`"
                          f"install networkx via `pip install nxviz`.")

    reg_groups = adata.obs[cluster].unique().to_list()
    if not set(edges_list.keys()).issubset(reg_groups):
        raise ValueError(f"the edges_list's keys are not equal or subset of the clusters from the "
                         f"adata.obs[{cluster}]")
    if cluster_names is not None:
        reg_groups = list(set(reg_groups).intersection(cluster_names))
        if len(reg_groups) == 0:
            raise ValueError(f"the clusters argument {cluster_names} provided doesn't match up with any clusters from the "
                             f"adata.")

    combined_edges, G, edges_dict = None, {}, {}
    for i, grp in enumerate(edges_list.keys()):
        G[grp] = nx.from_pandas_edgelist(edges_list[grp], 'regulator', 'target', edge_attr='weight', create_using=nx.DiGraph())
        edges_dict[grp] = np.array(G[grp].edges)
        combined_edges = edges_list[grp] if combined_edges is None else pd.concat((combined_edges, edges_list[grp]))

    # pull out degree information from nodes for later use
    combined_G = nx.from_pandas_edgelist(combined_edges, 'regulator', 'target',
                                         edge_attr='weight', create_using=nx.DiGraph())
    edges = np.array(combined_G.edges)
    node_ids, degrees = np.unique(edges, return_counts=True)

    nodes = []
    for node_id, degree in zip(node_ids, degrees):
        # store the index number as a way to align the nodes on axes
        combined_G.nodes.data()[node_id]['loc'] = node_id
        # also store the degree of each node as another way to
        #  align nodes on axes
        combined_G.nodes.data()[node_id]['degree'] = degree
        temp_node = Node(unique_id=node_id,
                         data=combined_G.nodes.data()[node_id])
        nodes.append(temp_node)

    hp = HivePlot()

    ### nodes ###
    hp.add_nodes(nodes)

    ### axes ###
    angles = np.linspace(0, 360, len(reg_groups) + 1)
    axes = []
    for i, grp in enumerate(reg_groups):
        axis = Axis(axis_id=grp, start=1, end=5, angle=angles[i],
                    long_name=grp)
        axes.append(axis)

    hp.add_axes(axes)

    ### node assignments ###
    nodes = [node.unique_id for node in nodes]

    # assign nodes and sorting procedure to position nodes on axis
    for i, grp in enumerate(reg_groups):
        hp.place_nodes_on_axis(axis_id=grp, unique_ids=nodes,
                                      sorting_feature_to_use="degree")
    for i, grp in enumerate(reg_groups):
        ### edges ###
        nex_grp = reg_groups[i + 1] if i < len(reg_groups) - 1 else reg_groups[0]
        hp.connect_axes(edges=edges_dict[grp], axis_id_1=grp,
                               axis_id_2=nex_grp, c="C" + str(i)) ### different color for each lineage

    # plot axes
    fig, ax = axes_viz_mpl(hp, figsize=figsize,
                           axes_labels_buffer=1.4)
    # plot nodes
    node_viz_mpl(hp,
                 fig=fig, ax=ax, s=80, c="black")
    # plot edges
    edge_viz_mpl(hive_plot=hp, fig=fig, ax=ax, alpha=0.7,
                 zorder=-1)

    # ax.set_title("Hive Plot", fontsize=20, y=0.9)
    # custom_lines = [Line2D([0], [0], color=f'C{i}', lw=3, linestyle='-') for i in range(len(reg_groups))]
    # ax.legend(custom_lines, reg_groups, loc='upper left', bbox_to_anchor=(0.37, 0.35),
    #           title="Regulatory network based on Jacobian analysis")

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'hiveplot', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return ax
