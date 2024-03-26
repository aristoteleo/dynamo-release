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
    import matplotlib.pyplot as plt

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

    ax = dyn.pl.state_graph(adata, group='Cell_type', save_show_or_return='return')
    assert isinstance(ax, tuple)

    res = dyn.pd.tree_model(
        adata,
        group='Cell_type',
        basis='umap',
        progenitor='Proliferating Progenitor',
        terminators=['Iridophore'],
    )
    assert len(res) == len(adata.obs["Cell_type"].unique())


def test_least_action_path():
    import pandas as pd

    adata = dyn.sample_data.hematopoiesis()
    adata = adata[:2000, :2000].copy()

    HSC_cells = dyn.tl.select_cell(adata, "cell_type", "HSC")
    Meg_cells = dyn.tl.select_cell(adata, "cell_type", "Meg")

    HSC_cells_indices = dyn.tools.utils.nearest_neighbors(
        adata.obsm["X_umap"][HSC_cells[15]],
        adata.obsm["X_umap"],
    )
    Meg_cells_indices = dyn.tools.utils.nearest_neighbors(
        adata.obsm["X_umap"][Meg_cells[1]],
        adata.obsm["X_umap"],
    )

    dyn.tl.neighbors(adata, basis="umap", result_prefix="umap")

    dyn.pd.least_action(
        adata,
        [adata.obs_names[HSC_cells_indices[0]][0]],
        [adata.obs_names[Meg_cells_indices[0]][0]],
        basis="umap",
        adj_key="X_umap_distances",
        min_lap_t=False,
        EM_steps=2,
    )
    ax = dyn.pl.least_action(adata, basis="umap", save_show_or_return="return")
    assert ax is not None

    lap = dyn.pd.least_action(
        adata,
        [adata.obs_names[HSC_cells_indices[0]][0]],
        [adata.obs_names[Meg_cells_indices[0]][0]],
        basis="pca",
        adj_key="cosine_transition_matrix",
        min_lap_t=False,
        EM_steps=2,
    )

    gtraj = dyn.pd.GeneTrajectory(adata)
    gtraj.from_pca(lap.X, t=lap.t)
    gtraj.calc_msd()
    ranking = dyn.vf.rank_genes(adata, "traj_msd")

    assert type(ranking) == pd.DataFrame
    assert ranking.shape[0] == adata.n_vars


def test_trajectoy_analysis():
    adata = dyn.sample_data.hematopoiesis()
    adata = adata[:1000, :1000].copy()
    adata.obs["trajectory"] = [i for i in range(adata.n_obs)]
    mfpt = dyn.pd.mean_first_passage_time(adata, sink_states=[0, 1, 2], init_states=[3, 4, 5], target_states=[6, 7, 8])
    assert mfpt is not None
