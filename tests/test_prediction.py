import dynamo as dyn


def test_fate(adata):
    progenitor = adata.obs_names[adata.obs.Cell_type.isin(['Proliferating Progenitor', 'Pigment Progenitor'])]
    dyn.pd.fate(adata, basis='umap', init_cells=progenitor, direction='backward')
    assert "fate_umap" in adata.uns.keys()
    assert adata.uns["fate_umap"]["prediction"][0].shape == (2, 250)

    dyn.pd.fate(adata, basis='umap', init_cells=progenitor, direction='both')
    assert "fate_umap" in adata.uns.keys()
    assert adata.uns["fate_umap"]["prediction"][0].shape == (2, 500)

    dyn.pd.fate(adata, basis='umap', init_cells=progenitor, direction='forward')
    assert "fate_umap" in adata.uns.keys()
    assert adata.uns["fate_umap"]["prediction"][0].shape == (2, 250)

    bias = dyn.pd.fate_bias(adata, group="Cell_type")
    assert len(bias) == len(adata.uns["fate_umap"]["prediction"])

    dyn.pd.andecestor(adata, init_cells=adata.obs_names[adata.obs.Cell_type.isin(['Iridophore'])], direction='backward')
    assert "ancestor" in adata.obs.keys()

    dyn.pd.andecestor(adata, init_cells=progenitor)
    assert "descendant" in adata.obs.keys()

    dyn.pd.andecestor(adata, init_cells=progenitor, direction="both")
    assert "lineage" in adata.obs.keys()


def test_perturbation(adata):
    dyn.pd.perturbation(adata, basis='umap', genes=adata.var_names[0], expression=-10)
    assert "X_umap_perturbation" in adata.obsm.keys()

    vf_ko = dyn.pd.KO(adata, basis='pca', KO_genes=adata.var_names[0])
    assert vf_ko.K.shape[0] == adata.n_vars

    dyn.pd.rank_perturbation_genes(adata)
    assert "rank_j_delta_x_perturbation" in adata.uns.keys()

    dyn.pd.rank_perturbation_cells(adata)
    assert "rank_j_delta_x_perturbation_cells" in adata.uns.keys()

    dyn.pd.rank_perturbation_cell_clusters(adata)
    assert "rank_j_delta_x_perturbation_cell_groups" in adata.uns.keys()


def test_state_graph(adata):
    # TODO: add this function to __init__ if we need it
    # res = dyn.pd.classify_clone_cell_type(
    #     adata,
    #     clone="hypo_trunk_2",
    #     clone_column="sample",
    #     cell_type_column="Cell_type",
    #     cell_type_to_excluded=["Unknown"],
    # )

    dyn.pd.state_graph(adata, group='Cell_type')
    assert "Cell_type_graph" in adata.uns.keys()

    res = dyn.pd.tree_model(
        adata,
        group='Cell_type',
        basis='umap',
        progenitor='Proliferating Progenitor',
        terminators=['Iridophore'],
    )
    assert len(res) == len(adata.obs["Cell_type"].unique())
