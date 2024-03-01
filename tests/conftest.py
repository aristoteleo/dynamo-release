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

    def gen_zebrafish_test_data():
        raw_adata = dyn.sample_data.zebrafish()
        adata = raw_adata[:300, :1000].copy()
        del raw_adata

        preprocessor = dyn.pp.Preprocessor(cell_cycle_score_enable=True)
        preprocessor.config_monocle_recipe(adata, n_top_genes=40)
        preprocessor.filter_genes_by_outliers_kwargs["inplace"] = True
        preprocessor.select_genes_kwargs["keep_filtered"] = False
        preprocessor.pca_kwargs["n_pca_components"] = 5
        preprocessor.preprocess_adata_monocle(adata)

        dyn.tl.dynamics(adata, model="stochastic")
        dyn.tl.reduceDimension(adata)
        dyn.tl.cell_velocities(adata)
        dyn.vf.VectorField(adata, basis="umap")

        dyn.tl.cell_velocities(adata, basis="pca")
        dyn.vf.VectorField(adata, basis="pca")

        TestUtils.mkdirs_wrapper(test_data_dir, abort=False)
        adata.write_h5ad(test_zebrafish_data_path)

    def gen_or_read_zebrafish_data():
        # generate data if needed
        if not os.path.exists(test_zebrafish_data_path):
            print("generating test data...")
            TestUtils.gen_zebrafish_test_data()

        print("reading test data...")
        adata = dyn.read_h5ad(test_zebrafish_data_path)
        return adata

    # def read_test_spatial_genomics_data():
    #     return dyn.read_h5ad(test_spatial_genomics_path)


@pytest.fixture
def utils():
    return TestUtils


ZEBRAFISH_ADATA = None


@pytest.fixture
def adata(utils):
    global ZEBRAFISH_ADATA
    if ZEBRAFISH_ADATA:
        return ZEBRAFISH_ADATA
    ZEBRAFISH_ADATA = utils.gen_or_read_zebrafish_data()
    return ZEBRAFISH_ADATA
