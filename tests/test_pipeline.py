import dynamo as dyn


def test_dynamcis(adata):
    adata.uns["pp"]["tkey"] = None
    dyn.tl.dynamics(adata, model="stochastic")
    dyn.tl.reduceDimension(adata)
    dyn.tl.cell_velocities(adata)
    dyn.vf.VectorField(adata, basis="umap", M=100)