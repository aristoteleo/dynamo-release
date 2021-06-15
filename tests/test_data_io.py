import dynamo
import numpy as np
import dynamo as dyn
import os

from utils import *


def test_save_rank_info(adata):
    dyn.export_rank_xlsx(adata)


def test_scEU_seq():
    dyn.sample_data.scEU_seq()
    os.path.exists("./data/rpe1.h5ad")


if __name__ == "__main__":
    adata = gen_or_read_zebrafish_data()
    test_scEU_seq()
    test_save_rank_info(adata)
