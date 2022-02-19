from pathlib import Path
from typing import Union
from dynamo import dynamo_logger
import dynamo
from dynamo import LoggerManager
from dynamo.dynamo_logger import main_info
import dynamo.preprocessing
import dynamo as dyn
import pytest
import time
import numpy as np
import os

LoggerManager.main_logger.setLevel(LoggerManager.DEBUG)

test_data_dir = Path("./test_data/")
test_zebrafish_data_path = test_data_dir / "test_zebrafish.h5ad"
test_spatial_genomics_path = test_data_dir / "allstage_processed.h5ad"


def mkdirs_wrapper(path: Union[str, Path], abort=True):
    if os.path.exists(path):
        main_info(str(path) + " : exists")
        if abort:
            exit(0)
        elif os.path.isdir(path):
            main_info(str(path) + " : is a directory, continue using the old one")
            return False
        else:
            main_info(str(path) + " : is not a directory, creating one")
            os.makedirs(path)
            return True
    else:
        os.makedirs(path)
        return True


def gen_zebrafish_test_data(basis="pca"):
    adata = dyn.sample_data.zebrafish()
    # adata = adata[:3000]
    dyn.pp.recipe_monocle(adata, num_dim=20, exprs_frac_for_gene_exclusion=0.005)
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
    mkdirs_wrapper(test_data_dir, abort=False)
    adata.write_h5ad(test_zebrafish_data_path)


def gen_or_read_zebrafish_data():
    # generate data if needed
    if not os.path.exists(test_zebrafish_data_path):
        print("generating test data...")
        gen_zebrafish_test_data()

    print("reading test data...")
    # TODO use a fixture in future
    adata = dyn.read_h5ad(test_zebrafish_data_path)
    return adata


def read_test_spatial_genomics_data():
    return dyn.read_h5ad(test_spatial_genomics_path)
