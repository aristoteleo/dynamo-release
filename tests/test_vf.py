import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import dynamo as dyn


def test_vector_calculus(adata):
    progenitor = adata.obs_names[adata.obs.Cell_type.isin(['Proliferating Progenitor', 'Pigment Progenitor'])]
    dyn.vf.velocities(adata, basis="umap", init_cells=progenitor)
    assert "velocities_umap" in adata.uns.keys()

    dyn.vf.speed(adata)
    assert "speed_umap" in adata.obs.keys()

    dyn.vf.jacobian(adata, basis="umap", regulators=['tmsb4x'], effectors=['ptmaa'], cell_idx=progenitor)
    assert "jacobian_umap" in adata.uns.keys()

    dyn.vf.hessian(adata, basis="umap", regulators=['tmsb4x'], coregulators=['ptmaa'], cell_idx=progenitor)
    assert "hessian_umap" in adata.uns.keys()

    dyn.vf.laplacian(adata, hkey="hessian_umap", basis="umap")
    assert "Laplacian_umap" in adata.obs.keys()

    dyn.vf.sensitivity(adata, basis="umap", regulators=['tmsb4x'], effectors=['ptmaa'], cell_idx=progenitor)
    assert "sensitivity_umap" in adata.uns.keys()

    dyn.vf.acceleration(adata, basis="umap")
    assert "acceleration_umap" in adata.obs.keys()

    dyn.vf.curvature(adata, basis="umap")
    assert "curvature_umap" in adata.obsm.keys()

    # Need 3D vector field data
    # dyn.vf.torsion(adata, basis="umap")
    # assert "torsion_umap" in adata.obs.keys()

    dyn.vf.curl(adata, basis="umap")
    assert "curl_umap" in adata.obs.keys()

    dyn.vf.divergence(adata, basis="umap", cell_idx=progenitor)
    assert "divergence_umap" in adata.obs.keys()
