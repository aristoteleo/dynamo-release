"""Offline tests for the protocol LAP plotting API."""

import copy
import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.collections import LineCollection

import dynamo as dyn
from dynamo.plot.utils import map2color


@pytest.fixture
def lap_data():
    grid = np.linspace(-1, 1, 12)
    x, y = np.meshgrid(grid, grid)
    coords = np.column_stack([x.ravel(), y.ravel()])
    adata = AnnData(np.zeros((len(coords), 1)))
    adata.obsm["X_umap"] = coords
    adata.obsm["velocity_umap"] = np.column_stack([-coords[:, 1], coords[:, 0]])
    adata.obs["cell_type"] = pd.Categorical(np.where(coords[:, 0] < 0, "HSC", "Meg"))
    graph = {
        "HSC->Meg": {
            "LAP_umap": {
                "prediction": [np.array([[-0.8, 0.0], [0.0, 0.5], [0.8, 0.0]])],
                "action": [np.array([0.0, 0.2, 0.8])],
            }
        },
        "Meg->HSC": {
            "LAP_umap": {
                "prediction": [np.array([[0.8, 0.0], [0.0, -0.5], [-0.8, 0.0]])],
                "action": [np.array([0.0, 0.3, 0.9])],
            }
        },
    }
    yield adata, graph
    plt.close("all")


@pytest.fixture
def background(monkeypatch):
    calls = []

    def draw(adata, **kwargs):
        calls.append(kwargs)
        return [kwargs["ax"]]

    module = importlib.import_module("dynamo.plot.scVectorField")
    monkeypatch.setattr(module, "streamline_plot", draw)
    return calls


def test_public_api_selected_paths_and_colors(lap_data, background):
    adata, graph = lap_data
    original = copy.deepcopy(graph)
    palette = {"HSC": "red", "Meg": "blue"}
    fig, ax = dyn.pd.plot_lap(
        adata,
        graph,
        ["Meg->HSC", "HSC->Meg"],
        title="Selected LAPs",
        color_key=palette,
        figsize=(4, 2),
        density=0.5,
    )
    assert dyn.pd.plot_lap is dyn.prediction.plot_lap
    assert ax.figure is fig
    np.testing.assert_allclose(fig.get_size_inches(), [4, 2])
    assert ax.get_title() == "Selected LAPs"
    assert ax.get_xlabel() == "X_umap1"
    assert ax.get_ylabel() == "X_umap2"
    assert background[0]["color_key"] == palette
    assert background[0]["save_show_or_return"] == "return"
    assert background[0]["density"] == 0.5
    assert len(ax.lines) == len(ax.collections) == 2
    for index, key in enumerate(["Meg->HSC", "HSC->Meg"]):
        lap = graph[key]["LAP_umap"]
        np.testing.assert_allclose(ax.lines[index].get_xydata(), lap["prediction"][0])
        np.testing.assert_allclose(ax.collections[index].get_facecolors(), map2color(lap["action"][0], cmap="RdBu_r"))
        np.testing.assert_array_equal(lap["prediction"][0], original[key]["LAP_umap"]["prediction"][0])
        np.testing.assert_array_equal(lap["action"][0], original[key]["LAP_umap"]["action"][0])
    assert "LAP_umap" not in adata.uns


def test_single_path_existing_axes_and_other_basis(lap_data, background):
    adata, graph = lap_data
    adata.obsm["X_pca"] = adata.obsm["X_umap"].copy()
    graph["HSC->Meg"]["LAP_pca"] = graph["HSC->Meg"]["LAP_umap"]
    fig, axes = plt.subplots(1, 2)
    result = dyn.pd.plot_lap(adata, graph, "HSC->Meg", basis="pca", ax=axes[1])
    assert result == (fig, axes[1])
    assert len(axes[0].lines) == 0
    assert len(axes[1].lines) == 1
    assert background[0]["basis"] == "pca"
    assert axes[1].get_xlabel() == "X_pca1"


@pytest.mark.parametrize("case", ["empty", "unknown", "missing_basis", "missing_action", "short_action"])
def test_invalid_selection_fails_before_drawing(lap_data, background, case):
    adata, graph = lap_data
    paths = ["HSC->Meg"]
    error = ValueError
    if case == "empty":
        paths = []
    elif case == "unknown":
        paths = ["missing"]
        error = KeyError
    elif case == "missing_basis":
        del graph["HSC->Meg"]["LAP_umap"]
        error = KeyError
    elif case == "missing_action":
        graph["HSC->Meg"]["LAP_umap"]["action"] = []
    else:
        graph["HSC->Meg"]["LAP_umap"]["action"] = [np.array([0.0])]
    figures = plt.get_fignums()
    with pytest.raises(error):
        dyn.pd.plot_lap(adata, graph, paths)
    assert not background
    assert plt.get_fignums() == figures


def test_real_streamline_background_and_save(lap_data, tmp_path):
    adata, graph = lap_data
    fig, ax = dyn.pd.plot_lap(
        adata,
        graph,
        ["HSC->Meg", "Meg->HSC"],
        title="LAP smoke test",
        frontier=False,
        show_legend=False,
        xy_grid_nums=(15, 15),
        n_neighbors=10,
    )
    assert len(ax.lines) == 2
    assert any(isinstance(collection, LineCollection) for collection in ax.collections)
    assert not any(text.get_text().startswith("X_umap") for text in ax.texts)
    np.testing.assert_allclose(ax.lines[0].get_xydata(), graph["HSC->Meg"]["LAP_umap"]["prediction"][0])
    output = tmp_path / "laps.png"
    fig.savefig(output)
    assert output.stat().st_size > 1000
