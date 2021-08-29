import dynamo
import numpy as np
import dynamo as dyn
import os

from utils import *


def test_save_rank_info(adata):
    dyn.export_rank_xlsx(adata)


def test_scEU_seq():
    dynamo.sample_data.scEU_seq()
    assert os.path.exists("./data/rpe1.h5ad")


def test_zebrafish():
    dynamo.sample_data.zebrafish()


if __name__ == "__main__":
    # test_scEU_seq()
    test_zebrafish()
    adata = gen_or_read_zebrafish_data()
    test_save_rank_info(adata)
