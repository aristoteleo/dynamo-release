import os
import time

import numpy as np
import pytest

import dynamo as dyn
import dynamo.tools
from dynamo import LoggerManager

# import utils

logger = LoggerManager.get_main_logger()


def test_simple_cluster_community_adata(processed_zebra_adata):
    dyn.tl.louvain(processed_zebra_adata)
    dyn.tl.leiden(processed_zebra_adata)
    dyn.tl.infomap(processed_zebra_adata)

    try:
        dyn.tl.louvain(processed_zebra_adata, directed=True)
    except ValueError as e:
        print("################################################")
        print("PASSED: Value error captured as EXPECTED, Exception info:")
        print(e)
        print("################################################")

    dyn.tl.leiden(processed_zebra_adata, directed=True)
    dyn.tl.infomap(processed_zebra_adata, directed=True)
    assert np.all(processed_zebra_adata.obs["louvain"] != -1)
    assert np.all(processed_zebra_adata.obs["leiden"] != -1)
    assert np.all(processed_zebra_adata.obs["infomap"] != -1)

    # dyn.pl.louvain(processed_zebra_adata, basis="pca")
    # dyn.pl.leiden(processed_zebra_adata, basis="pca")
    # dyn.pl.infomap(processed_zebra_adata, basis="pca")


def test_simple_cluster_subset(processed_zebra_adata):
    print(processed_zebra_adata.obs["Cluster"])
    result = dyn.tl.infomap(
        processed_zebra_adata,
        directed=True,
        copy=True,
        selected_cluster_subset=["Cluster", [0, 1, 2]],
    )
    print(result.obs["subset_infomap"])
    result = dyn.tl.infomap(processed_zebra_adata, directed=True, copy=True, selected_cell_subset=np.arange(0, 10))
    print(result.obs["subset_infomap"])


def test_simple_cluster_field(processed_zebra_adata):
    dyn.tl.cluster_field(processed_zebra_adata, method="louvain")
    dyn.tl.cluster_field(processed_zebra_adata, method="leiden")


def test_simple_cluster_keys(processed_zebra_adata):
    processed_zebra_adata = dyn.tl.infomap(processed_zebra_adata, directed=True, copy=True, layer="curvature")
    # processed_zebra_adata = dyn.tl.infomap(
    #     processed_zebra_adata,
    #     directed=True,
    #     copy=True,
    #     layer="acceleration"
    # )


def test_leiden_membership_input(processed_zebra_adata):
    # TODO fix the following test cases
    # somehow this initial member ship works before, but not now
    initial_membership = np.random.randint(
        low=0, high=min(100, len(processed_zebra_adata)), size=len(processed_zebra_adata), dtype=int
    )
    dyn.tl.leiden(processed_zebra_adata, initial_membership=initial_membership)

    initial_membership = np.random.randint(low=0, high=100, size=len(processed_zebra_adata), dtype=int)
    dyn.tl.leiden(processed_zebra_adata, directed=True, initial_membership=initial_membership)


if __name__ == "__main__":
    # processed_zebra_adata = utils.gen_or_read_zebrafish_data()
    # print("tests begin...")

    # ######### testing begins here #########
    # test_leiden_membership_input(processed_zebra_adata)
    # test_simple_cluster_community_adata(processed_zebra_adata)
    # test_simple_cluster_subset(processed_zebra_adata)
    # test_simple_cluster_keys(processed_zebra_adata)
    pass
