import os

import numpy as np

import dynamo
import dynamo as dyn
import pytest
# import utils


@pytest.mark.skip(reason="excessive memory usage")
def test_save_rank_info(processed_zebra_adata):
    adata = processed_zebra_adata.copy()
    dyn.vf.acceleration(adata, basis="pca")
    dyn.vf.rank_acceleration_genes(adata, groups="Cell_type", akey="acceleration", prefix_store="rank")
    dyn.export_rank_xlsx(adata)


@pytest.mark.skip(reason="excessive memory usage")
def test_scEU_seq():
    dynamo.sample_data.scEU_seq_rpe1()
    assert os.path.exists("./data/rpe1.h5ad")


@pytest.mark.skip(reason="excessive memory usage")
def test_save_adata():
    import numpy as np

    import dynamo as dyn
    from dynamo.preprocessing import Preprocessor

    adata = dyn.read("./data/hematopoiesis_raw.h5ad")  # dyn.sample_data.hematopoiesis_raw()
    # dyn.pp.recipe_monocle(adata, n_top_genes=200, fg_kwargs={"shared_count": 30})

    dyn.pl.basic_stats(adata)
    dyn.pl.highest_frac_genes(adata)

    preprocessor = Preprocessor(select_genes_kwargs={"n_top_genes": 200})

    preprocessor.preprocess_adata(adata, recipe="monocle")

    adata.write_h5ad("debug1.h5ad")

    dyn.tl.reduceDimension(adata, enforce=True)
    adata.write_h5ad("debug2.h5ad")

    dyn.tl.leiden(adata)
    adata.write_h5ad("debug3.h5ad")

    dyn.tl.find_group_markers(adata, group="leiden")  # DEG , n_genes=1000)
    adata.write_h5ad("debug4.h5ad")

    dyn.tl.moments(adata, group="time")
    adata.uns["pp"]["has_splicing"] = False
    dyn.tl.dynamics(adata, group="time", one_shot_method="sci_fate", model="deterministic")
    # dyn.tl.dynamics(adata)

    adata.write_h5ad("debug5.h5ad")  # Numeriacl  -> var

    # dyn.tl.cell_velocities(adata, method="transform")
    dyn.tl.cell_velocities(adata, basis="umap")
    adata.write_h5ad("debug6.h5ad")

    pca_genes = adata.var.use_for_pca
    new_expr = adata[:, pca_genes].layers["M_n"]
    time_3_gamma = adata[:, pca_genes].var.time_3_gamma.astype(float)
    time_5_gamma = adata[:, pca_genes].var.time_5_gamma.astype(float)

    t = adata.obs.time.astype(float)
    M_s = adata.layers["M_s"][:, pca_genes]

    time_3_cells = adata.obs.time == 3
    time_5_cells = adata.obs.time == 5

    def alpha_minus_gamma_s(new, gamma, t, M_s):
        # equation: alpha = new / (1 - e^{-rt}) * r
        alpha = new.A.T / (1 - np.exp(-gamma.values[:, None] * t.values[None, :])) * gamma.values[:, None]

        gamma_s = gamma.values[:, None] * M_s.A.T
        alpha_minus_gamma_s = alpha - gamma_s
        return alpha_minus_gamma_s

    time_3_velocity_n = alpha_minus_gamma_s(
        new_expr[time_3_cells, :], time_3_gamma, t[time_3_cells], M_s[time_3_cells, :]
    )
    time_5_velocity_n = alpha_minus_gamma_s(
        new_expr[time_5_cells, :], time_5_gamma, t[time_5_cells], M_s[time_5_cells, :]
    )

    velocity_n = adata.layers["velocity_N"].copy()

    valid_velocity_n = velocity_n[:, pca_genes].copy()
    valid_velocity_n[time_3_cells, :] = time_3_velocity_n.T
    valid_velocity_n[time_5_cells, :] = time_5_velocity_n.T
    velocity_n[:, pca_genes] = valid_velocity_n.copy()

    adata.layers["velocity_alpha_minus_gamma_s"] = velocity_n.copy()

    dyn.tl.cell_velocities(adata, basis="pca", X=adata.layers["M_t"], V=adata.layers["velocity_alpha_minus_gamma_s"])

    adata.write_h5ad("debug7.h5ad")

    dyn.vf.VectorField(adata, basis="pca")
    adata.write_h5ad("debug8.h5ad")

    dyn.vf.VectorField(adata, basis="umap")
    adata.write_h5ad("debug9.h5ad")

    dyn.vf.jacobian(adata, regulators=adata.var_names)
    adata.write_h5ad("debug10.h5ad")

    dyn.vf.rank_jacobian_genes(adata, groups="leiden")
    adata.write_h5ad("debug11.h5ad")
