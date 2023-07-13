import os

import numpy as np
import pandas as pd

import dynamo
import dynamo as dyn

# import utils


def test_save_rank_info(zebra_adata):
    dyn.export_rank_xlsx(zebra_adata)


def test_zebrafish():
    assert os.path.exists("./data/zebrafish.h5ad")


def test_zebrafish():
    assert os.path.exists("./data/zebrafish.h5ad")


def test_scEU_seq():
    assert os.path.exists("./data/rpe1.h5ad")


def test_save_adata(raw_hemato_adata):
    import numpy as np

    import dynamo as dyn
    from dynamo.preprocessing import Preprocessor

    # raw_hemato_adata = dyn.read("./data/hematopoiesis_raw.h5ad")  # dyn.sample_data.hematopoiesis_raw()
    # dyn.pp.recipe_monocle(raw_hemato_adata, n_top_genes=200, fg_kwargs={"shared_count": 30})

    dyn.pl.basic_stats(raw_hemato_adata)
    dyn.pl.highest_frac_genes(raw_hemato_adata)

    preprocessor = Preprocessor(select_genes_kwargs={"n_top_genes": 200})

    preprocessor.preprocess_adata(raw_hemato_adata, recipe="monocle")

    raw_hemato_adata.write_h5ad("debug1.h5ad")

    dyn.tl.reduceDimension(raw_hemato_adata, enforce=True)
    raw_hemato_adata.write_h5ad("debug2.h5ad")

    dyn.tl.leiden(raw_hemato_adata)
    raw_hemato_adata.write_h5ad("debug3.h5ad")

    dyn.tl.find_group_markers(raw_hemato_adata, group="leiden")  # DEG , n_genes=1000)
    raw_hemato_adata.write_h5ad("debug4.h5ad")

    dyn.tl.moments(raw_hemato_adata, group="time")
    raw_hemato_adata.uns["pp"]["has_splicing"] = False
    dyn.tl.dynamics(raw_hemato_adata, group="time", one_shot_method="sci_fate", model="deterministic")
    # dyn.tl.dynamics(raw_hemato_adata)

    raw_hemato_adata.write_h5ad("debug5.h5ad")  # Numeriacl  -> var

    # dyn.tl.cell_velocities(raw_hemato_adata, method="transform")
    dyn.tl.cell_velocities(raw_hemato_adata, basis="umap")
    raw_hemato_adata.write_h5ad("debug6.h5ad")

    pca_genes = raw_hemato_adata.var.use_for_pca
    new_expr = raw_hemato_adata[:, pca_genes].layers["M_n"]
    time_3_gamma = raw_hemato_adata[:, pca_genes].var.time_3_gamma.astype(float)
    time_5_gamma = raw_hemato_adata[:, pca_genes].var.time_5_gamma.astype(float)

    t = raw_hemato_adata.obs.time.astype(float)
    M_s = raw_hemato_adata.layers["M_s"][:, pca_genes]

    time_3_cells = raw_hemato_adata.obs.time == 3
    time_5_cells = raw_hemato_adata.obs.time == 5

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

    velocity_n = raw_hemato_adata.layers["velocity_N"].copy()

    valid_velocity_n = velocity_n[:, pca_genes].copy()
    valid_velocity_n[time_3_cells, :] = time_3_velocity_n.T
    valid_velocity_n[time_5_cells, :] = time_5_velocity_n.T
    velocity_n[:, pca_genes] = valid_velocity_n.copy()

    raw_hemato_adata.layers["velocity_alpha_minus_gamma_s"] = velocity_n.copy()

    dyn.tl.cell_velocities(
        raw_hemato_adata,
        basis="pca",
        X=raw_hemato_adata.layers["M_t"],
        V=raw_hemato_adata.layers["velocity_alpha_minus_gamma_s"],
    )

    raw_hemato_adata.write_h5ad("debug7.h5ad")

    dyn.vf.VectorField(raw_hemato_adata, basis="pca")
    raw_hemato_adata.write_h5ad("debug8.h5ad")

    dyn.vf.VectorField(raw_hemato_adata, basis="umap")
    raw_hemato_adata.write_h5ad("debug9.h5ad")

    dyn.vf.jacobian(raw_hemato_adata, regulators=raw_hemato_adata.var_names)
    raw_hemato_adata.write_h5ad("debug10.h5ad")

    dyn.vf.rank_jacobian_genes(raw_hemato_adata, groups="leiden")
    raw_hemato_adata.write_h5ad("debug11.h5ad")


if __name__ == "__main__":
    # test_scEU_seq()
    # test_zebrafish()
    # raw_hemato_adata = utils.gen_or_read_zebrafish_data()
    # test_save_rank_info(raw_hemato_adata)
    test_save_adata()
    pass
