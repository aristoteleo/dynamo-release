from pathlib import Path

import dynamo as dyn
from memory_profiler import profile

test_data_dir = Path("./test_data/")
test_zebrafish_data_path = test_data_dir / "test_zebrafish.h5ad"


@profile()
def test_processed_zebra_adata_adata():
    raw_adata = dyn.sample_data.zebrafish()
    adata = raw_adata[:, :5000].copy()
    del raw_adata

    preprocessor = dyn.pp.Preprocessor(cell_cycle_score_enable=True)
    preprocessor.config_monocle_recipe(adata, n_top_genes=100)
    preprocessor.filter_genes_by_outliers_kwargs["inplace"] = True
    preprocessor.select_genes_kwargs["keep_filtered"] = False
    preprocessor.preprocess_adata_monocle(adata)

    adata.write_h5ad(test_zebrafish_data_path)


@profile()
def test_dynamcis():
    adata = dyn.read_h5ad(test_zebrafish_data_path)
    adata.uns["pp"]["tkey"] = None
    dyn.tl.dynamics(adata, model="stochastic")
    dyn.tl.reduceDimension(adata)
    dyn.tl.cell_velocities(adata)
    dyn.vf.VectorField(adata, basis="umap", M=100)
