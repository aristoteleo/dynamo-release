from dynamo import LoggerManager
import dynamo.tools
import dynamo as dyn
import pytest
import time
import numpy as np
import os


test_data_path = "test_clustering_zebrafish.h5ad"
logger = LoggerManager.get_main_logger()


def gen_test_data():
    adata = dyn.sample_data.zebrafish()
    # adata = adata[:3000]
    dyn.pp.recipe_monocle(adata, num_dim=20, exprs_frac_max=0.005)
    dyn.tl.dynamics(adata, model="stochastic", cores=8)
    dyn.tl.reduceDimension(adata, n_pca_components=30, enforce=True)
    dyn.tl.cell_velocities(adata, basis="pca")
    dyn.vf.VectorField(adata, basis="pca", M=100)
    dyn.vf.curvature(adata, basis="pca")
    dyn.vf.acceleration(adata, basis="pca")
    dyn.cleanup(adata)
    adata.write_h5ad(test_data_path)


def test_simple_cluster_community_adata(adata):
    dyn.tl.louvain(adata)
    dyn.tl.leiden(adata)
    initial_membership = np.random.randint(
        low=0, high=100, size=len(adata), dtype=int
    )
    dyn.tl.leiden(adata, initial_membership=initial_membership)
    dyn.tl.infomap(adata)

    try:
        dyn.tl.louvain(adata, directed=True)
    except ValueError as e:
        print("################################################")
        print("PASSED: Value error captured as EXPECTED, Exception info:")
        print(e)
        print("################################################")

    dyn.tl.leiden(adata, directed=True)
    initial_membership = np.random.randint(
        low=0, high=100, size=len(adata), dtype=int
    )
    dyn.tl.leiden(adata, directed=True, initial_membership=initial_membership)
    dyn.tl.infomap(adata, directed=True)
    assert np.all(adata.obs["louvain"] != -1)
    assert np.all(adata.obs["leiden"] != -1)
    assert np.all(adata.obs["infomap"] != -1)

    # dyn.pl.louvain(adata, basis="pca")
    # dyn.pl.leiden(adata, basis="pca")
    # dyn.pl.infomap(adata, basis="pca")


def test_simple_cluster_subset(adata):
    print(adata.obs["Cluster"])
    adata = dyn.tl.infomap(
        adata,
        directed=True,
        copy=True,
        selected_cluster_subset=["Cluster", [0, 1, 2]],
    )
    print(adata.obs["infomap"])
    adata = dyn.tl.infomap(
        adata, directed=True, copy=True, selected_cell_subset=np.arange(0, 10)
    )
    print(adata.obs["infomap"])


def test_simple_cluster_field(adata):
    dyn.tl.cluster_field(adata, method="louvain")
    dyn.tl.cluster_field(adata, method="leiden")


def test_simple_cluster_keys(adata):
    adata = dyn.tl.infomap(adata, directed=True, copy=True, layer="curvature")
    # adata = dyn.tl.infomap(
    #     adata,
    #     directed=True,
    #     copy=True,
    #     layer="acceleration"
    # )


if __name__ == "__main__":
    # generate data if needed
    if not os.path.exists(test_data_path):
        print("generating test data...")
        gen_test_data()

    print("reading test data...")
    # To-do: use a fixture in future
    adata = dyn.read_h5ad(test_data_path)
    print("******acc layer: ", adata.layers["curvature"])
    print(adata)

    print("tests begin...")
    ######### testing begins here #########
    test_simple_cluster_community_adata(adata)
    test_simple_cluster_subset(adata)
    test_simple_cluster_keys(adata)
    # test_simple_cluster_field(adata)
