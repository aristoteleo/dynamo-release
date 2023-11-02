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


def test_cell_growth_rate(adata):
    dyn.tl.cell_growth_rate(
        adata,
        group="Cell_type",
        source="Unknown",
        target="Unknown",
        clone_column="batch",
    )
    assert "growth_rate" in adata.obs.keys()


@pytest.mark.skip(reason="umap compatability issue with numpy, pynndescent and pytest")
def test_dynamics():
    adata = dyn.sample_data.scNT_seq_neuron_labeling()
    adata.obs['label_time'] = 2  # this is the labeling time

    adata = adata[:, adata.var.activity_genes]
    adata.obs['time'] = adata.obs['time'] / 60

    adata1 = adata.copy()
    preprocessor = dyn.pp.Preprocessor(cell_cycle_score_enable=True)
    preprocessor.preprocess_adata(adata1, recipe='monocle', tkey='label_time', experiment_type='one-shot')
    dyn.tl.dynamics(adata1)
    assert "velocity_N" in adata.layers.keys()

    adata2 = adata.copy()
    preprocessor = dyn.pp.Preprocessor(cell_cycle_score_enable=True)
    preprocessor.preprocess_adata(adata2, recipe='monocle', tkey='label_time', experiment_type='kin')
    dyn.tl.dynamics(adata2)
    assert "velocity_N" in adata.layers.keys()


def test_top_n_markers(adata):
    dyn.tl.find_group_markers(adata, group="Cell_type")
    top_n_df = dyn.tl.top_n_markers(adata, top_n_genes=1)
    assert type(top_n_df) == pd.DataFrame
    assert len(top_n_df) == len(adata.obs["Cell_type"].unique()) - 1


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


def test_score_cells(adata):
    scores = dyn.tl.score_cells(adata)
    assert scores.shape[0] == adata.n_obs
