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
    adata = adata[:1000]
    dyn.pp.recipe_monocle(adata, num_dim=20, exprs_frac_max=0.005)
    dyn.tl.dynamics(adata, model="stochastic", cores=8)
    dyn.tl.reduceDimension(adata, n_pca_components=5, enforce=True)
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

    dyn.tl.leiden(adata, **{"initial_membership": initial_membership})
    dyn.tl.infomap(adata)
    # dyn.tl.cluster_community(adata, method="louvain")
    # dyn.tl.cluster_community(adata, method="leiden")
    assert np.all(adata.obs["louvain_community"] != -1)
    assert np.all(adata.obs["leiden_community"] != -1)
    assert np.all(adata.obs["infomap_community"] != -1)
    # dyn.pl.louvain(adata, basis="pca")
    # dyn.pl.leiden(adata, basis="pca")


def test_simple_cluster_field(adata):
    dyn.tl.cluster_field(adata, method="louvain")
    dyn.tl.cluster_field(adata, method="leiden")


if __name__ == "__main__":
    # generate data if needed
    if not os.path.exists(test_data_path):
        print("generating test data...")
        gen_test_data()

    print("reading test data...")
    # To-do: use a fixture in future
    adata = dyn.read_h5ad(test_data_path)
    # select a subset of adata for testing

    print("tests begin...")
    ######### testing begins here #########
    test_simple_cluster_community_adata(adata)
    # test_simple_cluster_field(adata)
