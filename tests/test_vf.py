import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import dynamo as dyn


def test_vector_calculus_rank_vf(adata):
    adata = adata.copy()

    progenitor = adata.obs_names[adata.obs.Cell_type.isin(['Proliferating Progenitor', 'Pigment Progenitor'])]
    dyn.vf.velocities(adata, basis="umap", init_cells=progenitor)
    assert "velocities_umap" in adata.uns.keys()

    dyn.vf.speed(adata)
    assert "speed_umap" in adata.obs.keys()

    dyn.vf.jacobian(adata, basis="umap", regulators=["ptmaa", "tmsb4x"], effectors=["ptmaa", "tmsb4x"], cell_idx=progenitor)
    assert "jacobian_umap" in adata.uns.keys()

    dyn.vf.hessian(adata, basis="umap", regulators=["tmsb4x"], coregulators=["ptmaa"], effector=["ptmaa"], cell_idx=progenitor)
    assert "hessian_umap" in adata.uns.keys()

    dyn.vf.laplacian(adata, hkey="hessian_umap", basis="umap")
    assert "Laplacian_umap" in adata.obs.keys()

    dyn.vf.sensitivity(adata, basis="umap", regulators=["tmsb4x"], effectors=["ptmaa"], cell_idx=progenitor)
    assert "sensitivity_umap" in adata.uns.keys()

    dyn.vf.acceleration(adata, basis="pca")
    assert "acceleration_pca" in adata.obs.keys()

    dyn.vf.curvature(adata, basis="pca")
    assert "curvature_pca" in adata.obsm.keys()

    # Need 3D vector field data
    # dyn.vf.torsion(adata, basis="umap")
    # assert "torsion_umap" in adata.obs.keys()

    dyn.vf.curl(adata, basis="umap")
    assert "curl_umap" in adata.obs.keys()

    dyn.vf.divergence(adata, basis="umap", cell_idx=progenitor)
    assert "divergence_umap" in adata.obs.keys()

    rank = dyn.vf.rank_genes(adata, arr_key="velocity_S")
    assert len(rank) == adata.n_vars

    rank = dyn.vf.rank_cell_groups(adata, arr_key="velocity_S")
    assert len(rank) == adata.n_obs

    dyn.vf.rank_expression_genes(adata)
    assert "rank_M_s" in adata.uns.keys()

    dyn.vf.rank_velocity_genes(adata)
    assert "rank_velocity_S" in adata.uns.keys()

    rank = dyn.vf.rank_divergence_genes(adata, jkey="jacobian_umap")
    assert len(rank) == len(adata.uns["jacobian_umap"]["regulators"])

    rank = dyn.vf.rank_s_divergence_genes(adata, skey="sensitivity_umap")
    assert len(rank) == len(adata.uns["sensitivity_umap"]["regulators"])

    dyn.vf.rank_acceleration_genes(adata, akey="acceleration")
    assert "rank_acceleration" in adata.uns.keys()

    dyn.vf.rank_curvature_genes(adata, ckey="curvature")
    assert "rank_curvature" in adata.uns.keys()

    rank = dyn.vf.rank_jacobian_genes(adata, jkey="jacobian_umap", return_df=True)
    assert len(rank["all"]) == len(adata.uns["jacobian_umap"]["regulators"])

    rank = dyn.vf.rank_sensitivity_genes(adata, skey="sensitivity_umap")
    assert len(rank["all"]) == len(adata.uns["sensitivity_umap"]["regulators"])

    reg_dict = {"ptmaa": "ptmaa", "tmsb4x": "tmsb4x"}
    eff_dict = {"ptmaa": "ptmaa", "tmsb4x": "tmsb4x"}
    rank = dyn.vf.aggregateRegEffs(adata, basis="umap", reg_dict=reg_dict, eff_dict=eff_dict, store_in_adata=False)
    assert rank["aggregation_gene"].shape[2] == adata.n_obs


def test_cell_vectors(adata):
    adata = adata.copy()
    dyn.vf.cell_accelerations(adata)
    dyn.vf.cell_curvatures(adata)
    assert "acceleration_pca" in adata.obsm.keys()
    assert "curvature_pca" in adata.obsm.keys()
