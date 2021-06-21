from dynamo.tools.connectivity import check_and_recompute_neighbors, check_neighbors_completeness
from utils import *
import networkx as nx
import dynamo as dyn
import matplotlib.pyplot as plt
import numpy as np


def test_neighbors_subset(adata):
    dyn.tl.neighbors(adata)
    assert check_neighbors_completeness(adata)
    indices = np.random.randint(0, len(adata), size=100)
    _adata = adata[indices].copy()
    assert not check_neighbors_completeness(_adata)

    # check obsp keys subsetting by AnnData Obj
    neighbor_key = "neighbors"
    expected_conn_mat = adata.obsp["connectivities"][indices][:, indices]
    expected_dist_mat = adata.obsp["distances"][indices][:, indices]

    print("expected_conn_mat:", expected_conn_mat.shape)
    conn_mat = _adata.obsp["connectivities"]
    dist_mat = _adata.obsp["distances"]

    assert np.all(np.abs(expected_conn_mat - conn_mat.toarray()) < 1e-7)
    assert np.all(expected_dist_mat == dist_mat.toarray())

    # recompute and neighbor graph should be fine
    dyn.tl.neighbors(_adata)
    assert check_neighbors_completeness(_adata)


def test_broken_neighbors_check_recompute(adata):
    dyn.tl.neighbors(adata)
    assert check_neighbors_completeness(adata)
    indices = np.random.randint(0, len(adata), size=100)
    _adata = adata[indices].copy()
    assert not check_neighbors_completeness(_adata)
    check_and_recompute_neighbors(_adata)
    assert check_neighbors_completeness(_adata)


if __name__ == "__main__":
    # generate data if needed
    adata = gen_or_read_zebrafish_data()
    test_neighbors_subset(adata)
    test_broken_neighbors_check_recompute(adata)
