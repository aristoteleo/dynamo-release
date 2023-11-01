import os
import time

import numpy as np
import pytest

import dynamo as dyn
import dynamo.tools
from dynamo import LoggerManager

# import utils

logger = LoggerManager.get_main_logger()


@pytest.mark.skip(reason="dependency not installed")
def test_simple_cluster_community_adata(adata):
    adata = adata.copy()
    dyn.tl.louvain(adata)
    dyn.tl.leiden(adata)

    try:
        dyn.tl.louvain(adata, directed=True)
    except ValueError as e:
        print("################################################")
        print("PASSED: Value error captured as EXPECTED, Exception info:")
        print(e)
        print("################################################")

    dyn.tl.leiden(adata, directed=True)
    assert np.all(adata.obs["louvain"] != -1)
    assert np.all(adata.obs["leiden"] != -1)

    # dyn.pl.louvain(adata, basis="pca")
    # dyn.pl.leiden(adata, basis="pca")
    # dyn.pl.infomap(adata, basis="pca")


@pytest.mark.skip(reason="umap compatability issue with numpy, pynndescent and pytest")
def test_simple_cluster_field(adata):
    adata = adata.copy()
    dyn.tl.reduceDimension(adata, basis="umap", n_pca_components=30, enforce=True)
    dyn.tl.cell_velocities(adata, basis="umap")
    dyn.vf.VectorField(adata, basis="umap", M=100)
    dyn.vf.cluster_field(adata, basis="umap", method="louvain")
    dyn.vf.cluster_field(adata, basis="umap", method="leiden")


@pytest.mark.skip(reason="dependency not installed")
def test_leiden_membership_input(adata):
    adata = adata.copy()
    # somehow this initial member ship works before, but not now
    initial_membership = np.random.randint(low=0, high=min(100, len(adata)), size=len(adata), dtype=int)
    dyn.tl.leiden(adata, initial_membership=initial_membership)

    initial_membership = np.random.randint(low=0, high=100, size=len(adata), dtype=int)
    dyn.tl.leiden(adata, directed=True, initial_membership=initial_membership)
