import warnings

warnings.filterwarnings("ignore")

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from anndata import AnnData
from scipy.sparse import csr_matrix

import dynamo as dyn


@pytest.mark.skip(reason="excessive running time")
def test_run_rpe1_tutorial():
    rpe1 = dyn.sample_data.scEU_seq_rpe1()
    # show results
    rpe1.obs.exp_type.value_counts()
    rpe1[rpe1.obs.exp_type == "Chase", :].obs.time.value_counts()
    rpe1[rpe1.obs.exp_type == "Pulse", :].obs.time.value_counts()

    # create rpe1 kinectics
    rpe1_kinetics = rpe1[rpe1.obs.exp_type == "Pulse", :]
    rpe1_kinetics.obs["time"] = rpe1_kinetics.obs["time"].astype(str)
    rpe1_kinetics.obs.loc[rpe1_kinetics.obs["time"] == "dmso", "time"] = -1
    rpe1_kinetics.obs["time"] = rpe1_kinetics.obs["time"].astype(float)
    rpe1_kinetics = rpe1_kinetics[rpe1_kinetics.obs.time != -1, :]

    rpe1_kinetics.layers["new"], rpe1_kinetics.layers["total"] = (
        rpe1_kinetics.layers["ul"] + rpe1_kinetics.layers["sl"],
        rpe1_kinetics.layers["su"]
        + rpe1_kinetics.layers["sl"]
        + rpe1_kinetics.layers["uu"]
        + rpe1_kinetics.layers["ul"],
    )

    del rpe1_kinetics.layers["uu"], rpe1_kinetics.layers["ul"], rpe1_kinetics.layers["su"], rpe1_kinetics.layers["sl"]
    dyn.pl.basic_stats(rpe1_kinetics, save_show_or_return="return")
    rpe1_genes = ["UNG", "PCNA", "PLK1", "HPRT1"]
    rpe1_kinetics.obs.time

    assert np.sum(rpe1_kinetics.var_names.isnull()) == 0

    rpe1_kinetics.obs.time = rpe1_kinetics.obs.time.astype("float")
    rpe1_kinetics.obs.time = rpe1_kinetics.obs.time / 60
    rpe1_kinetics.obs.time.value_counts()
    # rpe1_kinetics = dyn.pp.recipe_monocle(rpe1_kinetics, n_top_genes=1000, total_layers=False, copy=True)
    dyn.pp.recipe_monocle(rpe1_kinetics, n_top_genes=1000, total_layers=False)

    dyn.tl.dynamics(rpe1_kinetics, model="deterministic", tkey="time", est_method="twostep", cores=16)
    dyn.tl.reduceDimension(rpe1_kinetics, reduction_method="umap")
    dyn.tl.cell_velocities(rpe1_kinetics, enforce=True, vkey="velocity_T", ekey="M_t")

    rpe1_kinetics.obsm["X_RFP_GFP"] = rpe1_kinetics.obs.loc[
        :, ["RFP_log10_corrected", "GFP_log10_corrected"]
    ].values.astype("float")
    rpe1_kinetics.layers["velocity_S"] = rpe1_kinetics.layers["velocity_T"].copy()
    dyn.tl.cell_velocities(rpe1_kinetics, enforce=True, vkey="velocity_S", ekey="M_t", basis="RFP_GFP")

    rpe1_kinetics.obs.Cell_cycle_relativePos = rpe1_kinetics.obs.Cell_cycle_relativePos.astype(float)
    rpe1_kinetics.obs.Cell_cycle_possition = rpe1_kinetics.obs.Cell_cycle_possition.astype(float)

    dyn.pl.streamline_plot(
        rpe1_kinetics,
        color=["Cell_cycle_possition", "Cell_cycle_relativePos"],
        basis="RFP_GFP",
        save_show_or_return="return",
    )
    dyn.pl.streamline_plot(rpe1_kinetics, color=["cell_cycle_phase"], basis="RFP_GFP", save_show_or_return="return")
    dyn.vf.VectorField(rpe1_kinetics, basis="RFP_GFP")
    progenitor = rpe1_kinetics.obs_names[rpe1_kinetics.obs.Cell_cycle_relativePos < 0.1]

    np.random.seed(19491001)

    from matplotlib import animation

    info_genes = rpe1_kinetics.var_names[rpe1_kinetics.var.use_for_transition]
    dyn.pd.fate(
        rpe1_kinetics,
        basis="RFP_GFP",
        init_cells=progenitor,
        interpolation_num=100,
        direction="forward",
        inverse_transform=False,
        average=False,
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax = dyn.pl.topography(
        rpe1_kinetics, basis="RFP_GFP", color="Cell_cycle_relativePos", ax=ax, save_show_or_return="return"
    )
    ax.set_aspect(0.8)

    instance = dyn.mv.StreamFuncAnim(adata=rpe1_kinetics, basis="RFP_GFP", color="Cell_cycle_relativePos", ax=ax)


if __name__ == "__main__":
    test_run_rpe1_tutorial()
