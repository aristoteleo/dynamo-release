from utils import *
import networkx as nx
import dynamo as dyn
import matplotlib.pyplot as plt

logger = LoggerManager.get_main_logger()


def test_scatter_contour(adata):
    dyn.pl.scatters(adata, layer="curvature", save_show_or_return="show", contour=True)
    dyn.pl.scatters(adata, layer="curvature", save_show_or_return="show", contour=True, calpha=1)


def test_circosPlot(adata):
    # genes from top acceleration rank
    selected_genes = ["hmgn2", "hmgb2a", "si:ch211-222l21.1", "mbpb", "h2afvb"]
    full_reg_rank = dyn.vf.rank_jacobian_genes(adata, groups="Cell_type", mode="full_reg", abs=True, output_values=True)
    full_eff_rank = dyn.vf.rank_jacobian_genes(adata, groups="Cell_type", mode="full_eff", abs=True, output_values=True)
    edges_list = dyn.vf.build_network_per_cluster(
        adata,
        cluster="Cell_type",
        cluster_names=None,
        full_reg_rank=full_reg_rank,
        full_eff_rank=full_eff_rank,
        genes=selected_genes,
        n_top_genes=1000,
    )

    print(edges_list["Unknown"])
    network = nx.from_pandas_edgelist(
        edges_list["Unknown"].drop_duplicates().query("weight > 1e-5"),
        "regulator",
        "target",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )
    print(network.nodes)
    print(network.edges.data())
    dyn.pl.circosPlot(
        adata,
        cluster="Cell_type",
        cluster_name="Unknown",
        edges_list=None,
        network=network,
        color="M_s",
    )
    plt.clf()
    plt.cla()
    plt.close()
    print("====after first plot=====")
    print(network.nodes)
    print(network.edges.data())
    dyn.pl.circosPlot(
        adata,
        cluster="Cell_type",
        cluster_name="Unknown",
        edges_list=None,
        network=network,
        color="M_s",
    )
    pass


if __name__ == "__main__":
    # generate data if needed
    adata = gen_or_read_zebrafish_data()

    # To-do: use a fixture in future
    # test_space_simple1(adata)
    # test_scatter_contour(adata)
    print("adata shape:", adata.shape)
    test_circosPlot(adata)
