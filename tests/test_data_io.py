import os

import numpy as np

import dynamo
import dynamo as dyn
import pytest
# import utils


def test_save_rank_info(processed_zebra_adata):
    dyn.export_rank_xlsx(processed_zebra_adata)


@pytest.mark.skip(reason="excessive memory usage")
def test_scEU_seq():
    dynamo.sample_data.scEU_seq_rpe1()
    assert os.path.exists("./data/rpe1.h5ad")


if __name__ == "__main__":
    # test_scEU_seq()
    # adata = utils.gen_or_read_zebrafish_data()
    # test_save_rank_info(adata)
    pass
