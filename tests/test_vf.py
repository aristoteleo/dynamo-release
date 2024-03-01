import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import dynamo as dyn


def test_vector_calculus_rank_vf(adata):
    import matplotlib
    import matplotlib.pyplot as plt

    adata = adata.copy()

    progenitor = adata.obs_names[adata.obs.Cell_type.isin(['Proliferating Progenitor', 'Pigment Progenitor'])]
    dyn.vf.velocities(adata, basis="umap", init_cells=progenitor)
    assert "velocities_umap" in adata.uns.keys()

    dyn.vf.speed(adata)
    assert "speed_umap" in adata.obs.keys()

    ax = dyn.pl.speed(adata, basis="umap", save_show_or_return="return")
    assert isinstance(ax, plt.Axes)

    dyn.vf.jacobian(adata, basis="umap", regulators=["ptmaa", "rpl5b"], effectors=["ptmaa", "rpl5b"], cell_idx=progenitor)
    assert "jacobian_umap" in adata.uns.keys()

    ax = dyn.pl.jacobian(adata, basis="umap", j_basis="umap", save_show_or_return="return")
    assert isinstance(ax, matplotlib.gridspec.GridSpec)

    ax = dyn.pl.jacobian_heatmap(
        adata, basis="umap", regulators=["ptmaa", "rpl5b"], effectors=["ptmaa", "rpl5b"], cell_idx=[2, 3], save_show_or_return="return")
    assert isinstance(ax, matplotlib.gridspec.GridSpec)

    dyn.vf.hessian(adata, basis="umap", regulators=["rpl5b"], coregulators=["ptmaa"], effector=["ptmaa"], cell_idx=progenitor)
    assert "hessian_umap" in adata.uns.keys()

    dyn.vf.laplacian(adata, hkey="hessian_umap", basis="umap")
    assert "Laplacian_umap" in adata.obs.keys()

    dyn.vf.sensitivity(adata, basis="umap", regulators=["rpl5b"], effectors=["ptmaa"], cell_idx=progenitor)
    assert "sensitivity_umap" in adata.uns.keys()

    ax = dyn.pl.sensitivity(adata, basis="umap", s_basis="umap", save_show_or_return="return")
    assert isinstance(ax, matplotlib.gridspec.GridSpec)

    ax = dyn.pl.sensitivity_heatmap(
        adata, basis="umap", regulators=["rpl5b"], effectors=["ptmaa"], cell_idx=[2, 3], save_show_or_return="return")
    assert isinstance(ax, matplotlib.gridspec.GridSpec)

    dyn.vf.acceleration(adata, basis="pca")
    assert "acceleration_pca" in adata.obs.keys()

    ax = dyn.pl.acceleration(adata, basis="pca", save_show_or_return="return")
    assert isinstance(ax, plt.Axes)

    dyn.vf.curvature(adata, basis="pca")
    assert "curvature_pca" in adata.obsm.keys()

    ax = dyn.pl.curvature(adata, basis="pca", save_show_or_return="return")
    assert isinstance(ax, plt.Axes)

    # Need 3D vector field data
    # dyn.vf.torsion(adata, basis="umap")
    # assert "torsion_umap" in adata.obs.keys()

    dyn.vf.curl(adata, basis="umap")
    assert "curl_umap" in adata.obs.keys()

    ax = dyn.pl.curl(adata, basis="umap", save_show_or_return="return")
    assert isinstance(ax, plt.Axes)

    dyn.vf.divergence(adata, basis="umap", cell_idx=progenitor)
    assert "divergence_umap" in adata.obs.keys()

    ax = dyn.pl.divergence(adata, basis="umap", save_show_or_return="return")
    assert isinstance(ax, plt.Axes)

    rank = dyn.vf.rank_genes(adata, arr_key="velocity_S")
    assert len(rank) == adata.n_vars

    rank = dyn.vf.rank_cell_groups(adata, arr_key="velocity_S")
    assert len(rank) == adata.n_obs

    dyn.vf.rank_expression_genes(adata)
    assert "rank_M_s" in adata.uns.keys()

    dyn.vf.rank_velocity_genes(adata)
    assert "rank_velocity_S" in adata.uns.keys()

    rank = dyn.vf.rank_divergence_genes(adata, jkey="jacobian_umap")
    assert len(rank) == len(adata.uns["jacobian_umap"]["regulators"])

    rank = dyn.vf.rank_s_divergence_genes(adata, skey="sensitivity_umap")
    assert len(rank) == len(adata.uns["sensitivity_umap"]["regulators"])

    dyn.vf.rank_acceleration_genes(adata, akey="acceleration")
    assert "rank_acceleration" in adata.uns.keys()

    dyn.vf.rank_curvature_genes(adata, ckey="curvature")
    assert "rank_curvature" in adata.uns.keys()

    rank = dyn.vf.rank_jacobian_genes(adata, jkey="jacobian_umap", return_df=True)
    assert len(rank["all"]) == len(adata.uns["jacobian_umap"]["regulators"])

    rank = dyn.vf.rank_sensitivity_genes(adata, skey="sensitivity_umap")
    assert len(rank["all"]) == len(adata.uns["sensitivity_umap"]["regulators"])

    reg_dict = {"ptmaa": "ptmaa", "rpl5b": "rpl5b"}
    eff_dict = {"ptmaa": "ptmaa", "rpl5b": "rpl5b"}
    rank = dyn.vf.aggregateRegEffs(adata, basis="umap", reg_dict=reg_dict, eff_dict=eff_dict, store_in_adata=False)
    assert rank["aggregation_gene"].shape[2] == adata.n_obs


def test_cell_vectors(adata):
    adata = adata.copy()
    dyn.vf.cell_accelerations(adata)
    dyn.vf.cell_curvatures(adata)
    assert "acceleration_pca" in adata.obsm.keys()
    assert "curvature_pca" in adata.obsm.keys()


def test_potential(adata):
    import matplotlib.pyplot as plt

    adata = adata.copy()
    dyn.vf.Potential(adata, basis="umap")
    assert "grid_Pot_umap" in adata.uns.keys()
    adata.uns.pop("grid_Pot_umap")

    dyn.vf.Potential(adata, basis="umap", method="Bhattacharya")
    assert "grid_Pot_umap" in adata.uns.keys()

    # too time-consuming
    # dyn.vf.Potential(adata, basis="umap", boundary=[0, 10], method="Tang")
    # assert "grid_Pot_umap" in adata.uns.keys()

    ax = dyn.pl.show_landscape(adata, basis="umap", save_show_or_return="return")
    assert isinstance(ax, plt.Axes)


def test_networks(adata):
    import matplotlib.pyplot as plt
    import networkx as nx

    adata = adata.copy()

    dyn.vf.jacobian(adata, basis="pca", regulators=["ptmaa", "rpl5b"], effectors=["ptmaa", "rpl5b"], cell_idx=np.arange(adata.n_obs))

    edges_list = dyn.vf.build_network_per_cluster(adata, cluster="Cell_type", genes=adata.var_names)
    assert isinstance(edges_list, dict)

    network = nx.from_pandas_edgelist(edges_list['Unknown'], 'regulator', 'target', edge_attr='weight',
                                      create_using=nx.DiGraph())
    ax = dyn.pl.arcPlot(adata, cluster="Cell_type", cluster_name="Unknown", edges_list=None, network=network, color="M_s", save_show_or_return="return")
    assert isinstance(ax, dyn.plot.networks.ArcPlot)

    res = dyn.vf.adj_list_to_matrix(adj_list=edges_list["Neuron"])
    assert isinstance(res, pd.DataFrame)
