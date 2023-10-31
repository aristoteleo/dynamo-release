from pathlib import Path

import dynamo as dyn

test_data_dir = Path("./test_data/")
test_zebrafish_data_path = test_data_dir / "test_zebrafish.h5ad"


def test_dynamcis(adata):
    adata.uns["pp"]["tkey"] = None
    dyn.tl.dynamics(adata, model="stochastic")
    dyn.tl.reduceDimension(adata)
    dyn.tl.cell_velocities(adata)
    dyn.vf.VectorField(adata, basis="umap", M=100)
