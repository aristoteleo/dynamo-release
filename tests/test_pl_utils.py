# import utils
import copy

import matplotlib.pyplot as plt
import networkx as nx
import pytest

import dynamo as dyn


def test_scatter_contour(processed_zebra_adata):
    dyn.pl.scatters(processed_zebra_adata, layer="curvature", save_show_or_return="return", contour=True)
    dyn.pl.scatters(processed_zebra_adata, layer="curvature", save_show_or_return="return", contour=True, calpha=1)


@pytest.mark.skip(reason="nxviz old version")
def test_circosPlot_deprecated(processed_zebra_adata):
    # genes from top acceleration rank
    selected_genes = ["hmgn2", "hmgb2a", "si:ch211-222l21.1", "mbpb", "h2afvb"]
    edges_list = dyn.vf.build_network_per_cluster(
        processed_zebra_adata,
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
        processed_zebra_adata,
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
        processed_zebra_adata,
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


@pytest.mark.skip(reason="breaks build")
def test_nxviz7_circosplot(processed_zebra_adata):
    selected_genes = ["hmgn2", "hmgb2a", "si:ch211-222l21.1", "mbpb", "h2afvb"]
    edges_list = dyn.vf.build_network_per_cluster(
        processed_zebra_adata,
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
        network.nodes[node][adata_layer_key] = processed_zebra_adata[:, node].layers[adata_layer_key].mean()
    dyn.pl.circosPlot(network, node_color_key="M_s", show_colorbar=False, edge_alpha_scale=1, edge_lw_scale=1)
    dyn.pl.circosPlot(network, node_color_key="M_s", show_colorbar=True, edge_alpha_scale=0.5, edge_lw_scale=0.4)
    # plt.show() # show via command line run.


if __name__ == "__main__":
    # generate data if needed
    # processed_zebra_adata = utils.gen_or_read_zebrafish_data()

    # TODO use a fixture in future
    # test_space_simple1(processed_zebra_adata)
    # test_scatter_contour(processed_zebra_adata)
    # test_circosPlot_deprecated(processed_zebra_adata)
    # test_nxviz7_circosplot()
    pass
