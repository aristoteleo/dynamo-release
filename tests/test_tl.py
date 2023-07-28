import numpy as np
import pandas as pd
import pytest

import dynamo as dyn


def test_calc_1nd_moment():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    result = dyn.tl.calc_1nd_moment(X, W, normalize_W=False)
    expected_result = np.array([[3, 4], [6, 8], [3, 4]])
    assert np.array_equal(result, expected_result)

    result, normalized_W = dyn.tl.calc_1nd_moment(X, W, normalize_W=True)
    expected_result = np.array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])
    assert np.array_equal(result, expected_result)
    assert np.array_equal(normalized_W,
                          np.array([[0.0, 1, 0.0], [0.5, 0.0, 0.5], [0.0, 1, 0.0]]))


def test_calc_2nd_moment():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    Y = np.array([[2, 3], [4, 5], [6, 7]])
    W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    result = dyn.tl.calc_2nd_moment(X, Y, W, normalize_W=False, center=False)
    expected_result = np.array([[12, 20], [32, 48], [12, 20]])
    assert np.array_equal(result, expected_result)

    result = dyn.tl.calc_2nd_moment(X, Y, W, normalize_W=True, center=False)
    expected_result = np.array([[12., 20.], [16., 24.], [12., 20.]])
    assert np.array_equal(result, expected_result)


def test_cell_growth_rate(processed_zebra_adata):
    adata = processed_zebra_adata.copy()
    dyn.tl.cell_growth_rate(adata, group="Cell_type")
    assert "growth_rate" in adata.obs.keys()


def test_top_n_markers(processed_zebra_adata):
    adata = processed_zebra_adata.copy()
    dyn.tl.find_group_markers(adata, group="Cell_type")
    top_n_df = dyn.tl.top_n_markers(adata)
    assert type(top_n_df) == pd.DataFrame
    assert len(top_n_df) == (len(adata.obs["Cell_type"].unique()) - 1) * 5


def test_sampling():
    arr = np.random.rand(20, 3)
    n = 2
    samples = dyn.tl.sample(arr, n)
    assert samples.shape[0] == n
    assert samples[0] in arr
    assert samples[1] in arr

    V = np.random.rand(20, 3)
    samples = dyn.tl.sample(arr, n, method="velocity", V=V)
    assert samples.shape[0] == n
    assert samples[0] in arr
    assert samples[1] in arr

    samples = dyn.tl.sample(arr, n, method="trn", X=arr)
    assert samples.shape[0] == n
    assert samples[0] in arr
    assert samples[1] in arr

    samples = dyn.tl.sample(arr, n, method="kmeans", X=arr)
    assert samples.shape[0] == n
    assert samples[0] in arr
    assert samples[1] in arr


def test_score_cells(processed_zebra_adata):
    adata = processed_zebra_adata.copy()
    scores = dyn.tl.score_cells(adata)
    assert scores.shape[0] == adata.n_obs
