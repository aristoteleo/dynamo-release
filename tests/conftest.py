import os
import time
from pathlib import Path
from typing import Union

import numpy as np
import pytest

import dynamo
import dynamo as dyn
import dynamo.preprocessing
from dynamo import LoggerManager, dynamo_logger
from dynamo.dynamo_logger import main_info

LoggerManager.main_logger.setLevel(LoggerManager.DEBUG)

test_data_dir = Path("./test_data/")
test_zebrafish_data_path = test_data_dir / "test_zebrafish.h5ad"
test_spatial_genomics_path = test_data_dir / "allstage_processed.h5ad"
import pytest


class TestUtils:
    LoggerManager = LoggerManager

    @staticmethod
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

    def read_test_spatial_genomics_data():
        return dyn.read_h5ad(test_spatial_genomics_path)


@pytest.fixture
def utils():
    return TestUtils


ZEBRAFISH_ADATA = None

# refactored gen_zebrafish_test_data() from TestUtils into a Pytest fixture
@pytest.fixture(scope="session")
def processed_zebra_adata():
    adata = dyn.sample_data.zebrafish()
    dyn.pp.recipe_monocle(adata, num_dim=20, exprs_frac_for_gene_exclusion=0.005)
    dyn.tl.dynamics(adata, model="stochastic", cores=8)
    dyn.tl.reduceDimension(adata, basis="pca", n_pca_components=30, enforce=True)
    dyn.tl.cell_velocities(adata, basis="pca")
    dyn.vf.VectorField(adata, basis="pca", M=100)
    dyn.vf.curvature(adata, basis="pca")
    dyn.vf.acceleration(adata, basis="pca")

    dyn.vf.rank_acceleration_genes(adata, groups="Cell_type", akey="acceleration", prefix_store="rank")
    dyn.vf.rank_curvature_genes(adata, groups="Cell_type", ckey="curvature", prefix_store="rank")
    dyn.vf.rank_velocity_genes(adata, groups="Cell_type", vkey="velocity_S", prefix_store="rank")

    dyn.pp.top_pca_genes(adata, n_top_genes=100)
    top_pca_genes = adata.var.index[adata.var.top_pca_genes]
    dyn.vf.jacobian(adata, regulators=top_pca_genes, effectors=top_pca_genes)
    dyn.cleanup(adata)
    return adata


@pytest.fixture(scope="session")
def raw_zebra_adata():
    adata = dyn.sample_data.zebrafish()


@pytest.fixture(scope="session")
def raw_hemato_adata():
    adata = dyn.sample_data.hematopoiesis()
    return adata


@pytest.fixture(scope="session")
def raw_pancreas_adata():
    adata = dyn.sample_data.pancreatic_endocrinogenesis()
    return adata


@pytest.fixture(scope="session")
def raw_rpe1_adata():
    adata = dyn.sample_data.scEU_seq_rpe1()
    return adata
