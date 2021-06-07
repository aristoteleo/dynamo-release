from dynamo import dynamo_logger
import dynamo
from dynamo import LoggerManager
import dynamo.preprocessing
import dynamo as dyn
import pytest
import time
import numpy as np
import os

LoggerManager.main_logger.setLevel(LoggerManager.DEBUG)

test_zebrafish_data_path = "./test_data/test_zebrafish.h5ad"
test_spatial_genomics_path = "./test_data/allstage_processed.h5ad"


def gen_zebrafish_test_data(basis="pca"):
    adata = dyn.sample_data.zebrafish()
    # adata = adata[:3000]
    dyn.pp.recipe_monocle(adata, num_dim=20, exprs_frac_max=0.005)
    dyn.tl.dynamics(adata, model="stochastic", cores=8)
    dyn.tl.reduceDimension(adata, basis=basis, n_pca_components=30, enforce=True)
    dyn.tl.cell_velocities(adata, basis=basis)
    dyn.vf.VectorField(adata, basis=basis, M=100)
    dyn.vf.curvature(adata, basis=basis)
    dyn.vf.acceleration(adata, basis=basis)

    dyn.vf.rank_acceleration_genes(adata, groups="Cell_type", akey="acceleration", prefix_store="rank")
    dyn.vf.rank_curvature_genes(adata, groups="Cell_type", ckey="curvature", prefix_store="rank")
    dyn.vf.rank_velocity_genes(adata, groups="Cell_type", vkey="velocity_S", prefix_store="rank")

    dyn.pp.top_pca_genes(adata, n_top_genes=100)
    top_pca_genes = adata.var.index[adata.var.top_pca_genes]
    dyn.vf.jacobian(adata, regulators=top_pca_genes, effectors=top_pca_genes)
    dyn.cleanup(adata)
    adata.write_h5ad(test_zebrafish_data_path)


def gen_or_read_zebrafish_data():
    # generate data if needed
    if not os.path.exists(test_zebrafish_data_path):
        print("generating test data...")
        gen_zebrafish_test_data()

    print("reading test data...")
    # To-do: use a fixture in future
    adata = dyn.read_h5ad(test_zebrafish_data_path)
    return adata


def read_test_spatial_genomics_data():
    return dyn.read_h5ad(test_spatial_genomics_path)
