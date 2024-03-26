# import utils
import copy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

import dynamo as dyn


@pytest.mark.skip(reason="need additional dependency")
def test_circosPlot_deprecated(adata):
    # genes from top acceleration rank
    selected_genes = ["hmgn2", "hmgb2a", "si:ch211-222l21.1", "mbpb", "h2afvb"]
    edges_list = dyn.vf.build_network_per_cluster(
        adata,
        cluster="Cell_type",
        cluster_names=None,
        genes=selected_genes,
        n_top_genes=1000,
        abs=True,
    )

    print(edges_list["Unknown"])
    network = nx.from_pandas_edgelist(
        edges_list["Unknown"].drop_duplicates().query("weight > 1e-5"),
        "regulator",
        "target",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )
    _network = copy.deepcopy(network)
    dyn.pl.circosPlotDeprecated(
        adata,
        cluster="Cell_type",
        cluster_name="Unknown",
        edges_list=None,
        network=network,
        color="M_s",
        save_show_or_return="return",
    )

    for e in network.edges():
        assert network.edges[e]["weight"] == _network.edges[e]["weight"]
    dyn.pl.circosPlotDeprecated(
        adata,
        cluster="Cell_type",
        cluster_name="Unknown",
        edges_list=None,
        network=network,
        color="M_s",
        save_show_or_return="return",
    )
    pass


@pytest.mark.skip(reason="require viral data")
def test_scatter_group_gamma(viral_adata, gene_list_df: list):
    dyn.pl.scatters(
        viral_adata,
        basis=viral_adata.var_names.intersection(gene_list_df.index)[:5],
        x="M_s",
        y="M_u",
        color="coarse_cluster",
        group="coarse_cluster",
        add_group_gamma_fit=True,
    )


@pytest.mark.skip(reason="need additional dependency")
def test_nxviz7_circosplot(adata):
    selected_genes = ["hmgn2", "hmgb2a", "si:ch211-222l21.1", "mbpb", "h2afvb"]
    edges_list = dyn.vf.build_network_per_cluster(
        adata,
        cluster="Cell_type",
        cluster_names=None,
        genes=selected_genes,
        n_top_genes=1000,
        abs=True,
    )

    print(edges_list["Unknown"])
    network = nx.from_pandas_edgelist(
        edges_list["Unknown"].drop_duplicates().query("weight > 1e-5"),
        "regulator",
        "target",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )
    adata_layer_key = "M_s"

    for node in network.nodes:
        network.nodes[node][adata_layer_key] = adata[:, node].layers[adata_layer_key].mean()
    dyn.pl.circosPlot(network, node_color_key="M_s", show_colorbar=False, edge_alpha_scale=1, edge_lw_scale=1)
    dyn.pl.circosPlot(network, node_color_key="M_s", show_colorbar=True, edge_alpha_scale=0.5, edge_lw_scale=0.4)
    # plt.show() # show via command line run.


def test_scatters_markers_ezplots():
    adata = dyn.sample_data.hematopoiesis()

    ax = dyn.pl.cell_cycle_scores(adata, save_show_or_return="return")
    assert isinstance(ax, plt.Axes)

    ax = dyn.pl.pca(adata, color="cell_type", save_show_or_return="return")
    assert isinstance(ax, plt.Axes)

    ax = dyn.pl.umap(adata, color="cell_type", save_show_or_return="return")
    assert isinstance(ax, plt.Axes)

    ax = dyn.pl.cell_wise_vectors(adata, color=["cell_type"], basis="umap", save_show_or_return="return")
    assert isinstance(ax, list)

    ax = dyn.pl.streamline_plot(adata, color=["cell_type"], basis="umap", save_show_or_return="return")
    assert isinstance(ax, list)

    ax = dyn.pl.topography(adata, basis="umap", color=["ntr", "cell_type"], save_show_or_return="return")
    assert isinstance(ax, list)

    ax = dyn.pl.plot_X(adata.obsm["X_umap"], save_show_or_return="return")
    assert isinstance(ax, plt.Axes)

    ax = dyn.pl.plot_V(adata.obsm["X_pca"], adata.obsm["velocity_umap"], save_show_or_return="return")
    assert isinstance(ax, plt.Axes)

    ax = dyn.pl.zscatter(adata, save_show_or_return="return")
    assert isinstance(ax, plt.Axes)

    ax = dyn.pl.zstreamline(adata, save_show_or_return="return")
    assert isinstance(ax, plt.Axes)

    ax = dyn.pl.plot_jacobian_gene(adata, save_show_or_return="return")
    assert isinstance(ax, list)

    ax = dyn.pl.bubble(adata, genes=adata.var_names[:4], group="cell_type", save_show_or_return="return")
    assert isinstance(ax, tuple)


def test_lap_plots():
    import numpy as np
    import seaborn as sns

    adata = dyn.sample_data.hematopoiesis()

    progenitor = adata.obs_names[adata.obs.cell_type.isin(['HSC'])]

    dyn.pd.fate(adata, basis='umap', init_cells=progenitor, interpolation_num=25, direction='forward',
                inverse_transform=False, average=False)
    ax = dyn.pl.fate_bias(adata, group="cell_type", basis="umap", save_show_or_return='return')
    assert isinstance(ax, sns.matrix.ClusterGrid)

    ax = dyn.pl.fate(adata, basis="umap", save_show_or_return='return')
    assert isinstance(ax, plt.Axes)

    # genes = adata.var_names[adata.var.use_for_dynamics]
    # integer_pairs = [
    #     (3, 21),
    #     (2, 11),
    # ]
    # pair_matrix = [[genes[x[0]], genes[x[1]]] for x in integer_pairs]
    # ax = dyn.pl.response(adata, pairs_mat=pair_matrix, return_data=True)
    # assert isinstance(ax, tuple)

    # ax = dyn.plot.connectivity.plot_connectivity(adata, graph=adata.obsp["perturbation_transition_matrix"],
    #                                         color=["cell_type"], save_show_or_return='return')
    # assert isinstance(ax, plt.Figure)

    from dynamo.tools.utils import nearest_neighbors

    fixed_points = np.array(
        [
            [8.45201833, 9.37697661],
            [14.00630381, 2.53853712],
        ]
    )

    HSC_cells_indices = nearest_neighbors(fixed_points[0], adata.obsm["X_umap"])
    Meg_cells_indices = nearest_neighbors(fixed_points[1], adata.obsm["X_umap"])
    dyn.pd.least_action(
        adata,
        [adata.obs_names[HSC_cells_indices[0]][0]],
        [adata.obs_names[Meg_cells_indices[0]][0]],
        basis="umap",
        adj_key="X_umap_distances",
        min_lap_t=True,
        EM_steps=2,
    )
    ax = dyn.pl.least_action(adata, basis="umap", save_show_or_return="return")
    assert isinstance(ax, plt.Axes)
    ax = dyn.pl.lap_min_time(adata, basis="umap", save_show_or_return="return")
    assert isinstance(ax, plt.Figure)


def test_heatmaps():
    import numpy as np
    import pandas as pd

    adata = dyn.sample_data.hematopoiesis()
    genes = adata.var_names[adata.var.use_for_dynamics]
    integer_pairs = [
        (3, 21),
        (2, 11),
    ]
    pair_matrix = [[genes[x[0]], genes[x[1]]] for x in integer_pairs]
    ax = dyn.pl.response(adata, pairs_mat=pair_matrix, return_data=True)
    assert isinstance(ax, tuple)
    ax = dyn.pl.causality(adata, pairs_mat=np.array(pair_matrix), return_data=True)
    assert isinstance(ax, pd.DataFrame)

    integer_pairs = [
        (3, 21, 23),
        (2, 11, 7),
    ]
    pair_matrix = [[genes[x[0]], genes[x[1]], genes[x[2]]] for x in integer_pairs]
    ax = dyn.pl.hessian(adata, pairs_mat=np.array(pair_matrix), return_data=True)
    assert isinstance(ax, pd.DataFrame)

    ax = dyn.pl.comb_logic(
        adata, pairs_mat=np.array(pair_matrix), xkey="M_n", ykey="M_t", zkey="velocity_alpha_minus_gamma_s", return_data=True
    )
    assert isinstance(ax, pd.DataFrame)


def test_preprocess(adata):
    import seaborn as sns

    ax = dyn.pl.basic_stats(adata, save_show_or_return="return")
    assert isinstance(ax, sns.axisgrid.FacetGrid)

    ax = dyn.pl.show_fraction(adata, save_show_or_return="return")
    assert isinstance(ax, sns.axisgrid.FacetGrid)

    ax = dyn.pl.variance_explained(adata, save_show_or_return="return")
    assert isinstance(ax, plt.Axes)

    ax = dyn.pl.biplot(adata, save_show_or_return="return")
    assert isinstance(ax, plt.Axes)

    ax = dyn.pl.loading(adata, n_pcs=4, ncol=2, save_show_or_return="return")
    assert isinstance(ax, np.ndarray)

    ax = dyn.pl.feature_genes(adata, save_show_or_return="return")
    assert isinstance(ax, plt.Figure)

    ax = dyn.pl.exp_by_groups(adata, genes=adata.var_names[:3], save_show_or_return="return")
    assert isinstance(ax, sns.axisgrid.FacetGrid)

    ax = dyn.pl.highest_frac_genes(adata)
    assert isinstance(ax, plt.Axes)


def test_time_series_plot(adata):
    import seaborn as sns

    adata = adata.copy()
    adata.uns["umap_fit"]["umap_kwargs"]["max_iter"] = None
    progenitor = adata.obs_names[adata.obs.Cell_type.isin(['Proliferating Progenitor', 'Pigment Progenitor'])]

    dyn.pd.fate(adata, basis='umap', init_cells=progenitor, interpolation_num=25, direction='forward',
                inverse_transform=True, average=False)

    ax = dyn.pl.kinetic_curves(adata, basis="umap", genes=adata.var_names[:4], save_show_or_return="return")
    assert isinstance(ax, sns.axisgrid.FacetGrid)
    ax = dyn.pl.kinetic_heatmap(adata, basis="umap", genes=adata.var_names[:4], save_show_or_return="return")
    assert isinstance(ax, sns.matrix.ClusterGrid)

    dyn.tl.order_cells(adata, basis="umap")
    progenitor = adata.obs_names[adata.obs.Cell_type.isin(['Proliferating Progenitor', 'Pigment Progenitor'])]

    dyn.vf.jacobian(adata, basis="umap", regulators=["ptmaa", "rpl5b"], effectors=["ptmaa", "rpl5b"],
                    cell_idx=progenitor)
    ax = dyn.pl.jacobian_kinetics(
        adata,
        basis="umap",
        genes=adata.var_names[:4],
        regulators=["ptmaa", "rpl5b"],
        effectors=["ptmaa", "rpl5b"],
        tkey="Pseudotime",
        save_show_or_return="return",
    )
    assert isinstance(ax, sns.matrix.ClusterGrid)

    dyn.vf.sensitivity(adata, basis="umap", regulators=["rpl5b"], effectors=["ptmaa"], cell_idx=progenitor)
    ax = dyn.pl.sensitivity_kinetics(
        adata,
        basis="umap",
        genes=adata.var_names[:4],
        regulators=["rpl5b"],
        effectors=["ptmaa"],
        tkey="Pseudotime",
        save_show_or_return="return",
    )
    assert isinstance(ax, sns.matrix.ClusterGrid)
