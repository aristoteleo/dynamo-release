import matplotlib.pyplot as plt

# import utils
import networkx as nx
import numpy as np

import dynamo as dyn
from dynamo.tools.connectivity import (
    _gen_neighbor_keys,
    check_and_recompute_neighbors,
    check_neighbors_completeness,
)


def test_neighbors_subset(processed_zebra_adata):
    dyn.tl.neighbors(processed_zebra_adata)
    assert check_neighbors_completeness(processed_zebra_adata)
    indices = np.random.randint(0, len(processed_zebra_adata), size=100)
    _adata = processed_zebra_adata[indices].copy()
    assert not check_neighbors_completeness(_adata)

    # check obsp keys subsetting by AnnData Obj
    neighbor_result_prefix = ""
    conn_key, dist_key, neighbor_key = _gen_neighbor_keys(neighbor_result_prefix)
    check_and_recompute_neighbors(processed_zebra_adata, result_prefix=neighbor_result_prefix)
    expected_conn_mat = processed_zebra_adata.obsp[conn_key][indices][:, indices]
    expected_dist_mat = processed_zebra_adata.obsp[dist_key][indices][:, indices]

    print("expected_conn_mat:", expected_conn_mat.shape)
    conn_mat = _adata.obsp[conn_key]
    dist_mat = _adata.obsp[dist_key]

    assert np.all(np.abs(expected_conn_mat - conn_mat.toarray()) < 1e-7)
    assert np.all(expected_dist_mat == dist_mat.toarray())

    # recompute and neighbor graph should be fine
    dyn.tl.neighbors(_adata)
    assert check_neighbors_completeness(_adata)


def test_broken_neighbors_check_recompute(processed_zebra_adata):
    dyn.tl.neighbors(processed_zebra_adata)
    assert check_neighbors_completeness(processed_zebra_adata)
    indices = np.random.randint(0, len(processed_zebra_adata), size=100)
    _adata = processed_zebra_adata[indices].copy()
    assert not check_neighbors_completeness(_adata)
    check_and_recompute_neighbors(_adata)
    assert check_neighbors_completeness(_adata)


def test_neighbors_no_pca_key(raw_zebra_adata):
    dyn.tl.neighbors(raw_zebra_adata)


if __name__ == "__main__":
    # generate data if needed
    # adata = utils.gen_or_read_zebrafish_data()
    # test_neighbors_subset(adata)
    # test_broken_neighbors_check_recompute(adata)
    # test_neighbors_no_pca_key()
    pass
