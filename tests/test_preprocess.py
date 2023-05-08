import timeit

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

# from utils import *
import dynamo as dyn
from dynamo.preprocessing import Preprocessor
from dynamo.preprocessing.cell_cycle import get_cell_phase
from dynamo.preprocessing.preprocessor_utils import (
    calc_mean_var_dispersion_sparse,
    is_float_integer_arr,
    is_integer_arr,
    is_log1p_transformed_adata,
    is_nonnegative,
    is_nonnegative_integer_arr,
    log1p,
    normalize,
)
from dynamo.preprocessing.utils import convert_layers2csr

SHOW_FIG = False


def test_highest_frac_genes_plot(adata, is_X_sparse=True):
    dyn.pl.highest_frac_genes(
        adata,
        show=SHOW_FIG,
        log=False,
        save_path="./test_simple_highest_frac_genes.png",
    )
    dyn.pl.highest_frac_genes(
        adata,
        log=False,
        show=SHOW_FIG,
        save_path="test_simple_highest_frac_genes.png",
    )
    dyn.pl.highest_frac_genes(
        adata,
        log=False,
        show=SHOW_FIG,
        save_path="test_simple_highest_frac_genes.png",
    )
    dyn.pl.highest_frac_genes(
        adata,
        log=False,
        show=SHOW_FIG,
        save_path="test_simple_highest_frac_genes.png",
        orient="h",
    )
    dyn.pl.highest_frac_genes(
        adata,
        log=False,
        show=SHOW_FIG,
        save_path="test_simple_highest_frac_genes.png",
        orient="h",
    )
    dyn.pl.highest_frac_genes(
        adata,
        log=False,
        show=SHOW_FIG,
        save_path="test_simple_highest_frac_genes.png",
        layer="M_s",
    )

    if is_X_sparse:
        adata.X = adata.X.toarray()
        dyn.pl.highest_frac_genes(adata, show=SHOW_FIG)


def test_highest_frac_genes_plot_prefix_list(adata, is_X_sparse=True):
    sample_list = ["MT-", "RPS", "RPL", "MRPS", "MRPL", "ERCC-"]
    dyn.pl.highest_frac_genes(adata, show=SHOW_FIG, gene_prefix_list=sample_list)
    dyn.pl.highest_frac_genes(adata, show=SHOW_FIG, gene_prefix_list=["RPL", "MRPL"])

    dyn.pl.highest_frac_genes(
        adata,
        gene_prefix_list=["someGenePrefixNotExisting"],
        show=SHOW_FIG,
    )


def test_recipe_monocle_feature_selection_layer_simple0():
    rpe1 = dyn.sample_data.scEU_seq_rpe1()
    # show results
    rpe1.obs.exp_type.value_counts()

    # create rpe1 kinectics
    rpe1_kinetics = rpe1[rpe1.obs.exp_type == "Pulse", :]
    rpe1_kinetics.obs["time"] = rpe1_kinetics.obs["time"].astype(str)
    rpe1_kinetics.obs.loc[rpe1_kinetics.obs["time"] == "dmso", "time"] = -1
    rpe1_kinetics.obs["time"] = rpe1_kinetics.obs["time"].astype(float)
    rpe1_kinetics = rpe1_kinetics[rpe1_kinetics.obs.time != -1, :]

    rpe1_kinetics.layers["new"], rpe1_kinetics.layers["total"] = (
        rpe1_kinetics.layers["ul"] + rpe1_kinetics.layers["sl"],
        rpe1_kinetics.layers["su"]
        + rpe1_kinetics.layers["sl"]
        + rpe1_kinetics.layers["uu"]
        + rpe1_kinetics.layers["ul"],
    )

    del rpe1_kinetics.layers["uu"], rpe1_kinetics.layers["ul"], rpe1_kinetics.layers["su"], rpe1_kinetics.layers["sl"]
    dyn.pl.basic_stats(rpe1_kinetics, save_show_or_return="return")
    rpe1_genes = ["UNG", "PCNA", "PLK1", "HPRT1"]

    # rpe1_kinetics = dyn.pp.recipe_monocle(rpe1_kinetics, n_top_genes=1000, total_layers=False, copy=True)
    dyn.pp.recipe_monocle(rpe1_kinetics, n_top_genes=1000, total_layers=False, feature_selection_layer="new")


def test_calc_dispersion_sparse():
    # TODO add randomize tests
    sparse_mat = csr_matrix([[1, 2, 0, 1, 5], [0, 0, 3, 1, 299], [4, 0, 5, 1, 399]])
    mean, var, dispersion = calc_mean_var_dispersion_sparse(sparse_mat)
    expected_mean = np.mean(sparse_mat.toarray(), axis=0)
    expected_var = np.var(sparse_mat.toarray(), axis=0)
    expected_dispersion = expected_var / expected_mean
    print("mean:", mean)
    print("expected mean:", expected_mean)
    print("var:", mean)
    print("expected var:", expected_mean)
    assert np.all(np.isclose(mean, expected_mean))
    assert np.all(np.isclose(var, expected_var))
    assert np.all(np.isclose(dispersion, expected_dispersion))

    # TODO adapt to seurat_get_mean_var test
    # sc_mean, sc_var = dyn.preprocessing.preprocessor_utils.seurat_get_mean_var(sparse_mat)
    # print("sc_mean:", sc_mean)
    # print("expected mean:", sc_mean)
    # print("sc_var:", sc_var)
    # print("expected var:", expected_var)
    # assert np.all(np.isclose(sc_mean, expected_mean))
    # assert np.all(np.isclose(sc_var, expected_var))


def test_Preprocessor_simple_run(adata):
    preprocess_worker = Preprocessor()
    preprocess_worker.preprocess_adata_monocle(adata)


def test_is_log_transformed():
    adata = dyn.sample_data.zebrafish()
    assert not is_log1p_transformed_adata(adata)
    log1p(adata)
    assert is_log1p_transformed_adata(adata)


def test_layers2csr_matrix():
    adata = dyn.sample_data.zebrafish()
    adata = adata[100:]
    convert_layers2csr(adata)
    for key in adata.layers.keys():
        print("layer:", key, "type:", type(adata.layers[key]))
        assert type(adata.layers[key]) is anndata._core.views.SparseCSRView


def test_compute_gene_exp_fraction():
    # TODO fix compute_gene_exp_fraction: discuss with Xiaojie
    # df = pd.DataFrame([[1, 2], [1, 1]]) # input cannot be dataframe
    df = csr_matrix([[1, 2], [1, 1]])
    frac, indices = dyn.preprocessing.compute_gene_exp_fraction(df)
    print("frac:", list(frac))
    assert np.all(np.isclose(frac.flatten(), [2 / 5, 3 / 5]))


def test_pca():
    adata = dyn.sample_data.zebrafish()
    preprocessor = Preprocessor()
    preprocessor.preprocess_adata_seurat_wo_pca(adata)
    adata = dyn.pp.pca(adata, n_pca_components=30)

    assert adata.obsm["X_pca"].shape[1] == 30
    assert adata.uns["PCs"].shape[1] == 30
    assert adata.uns["explained_variance_ratio_"].shape[0] == 30

    X_filterd = adata.X[:, adata.var.use_for_pca.values].copy()
    pca = PCA(n_components=30, random_state=0)
    X_pca_sklearn = pca.fit_transform(X_filterd.toarray())

    assert np.linalg.norm(X_pca_sklearn[:, :10] - adata.obsm["X_pca"][:, :10]) < 1e-1
    assert np.linalg.norm(pca.components_.T[:, :10] - adata.uns["PCs"][:, :10]) < 1e-1
    assert np.linalg.norm(pca.explained_variance_ratio_[:10] - adata.uns["explained_variance_ratio_"][:10]) < 1e-1



def test_preprocessor_seurat(adata):
    adata = dyn.sample_data.zebrafish()
    preprocessor = dyn.pp.Preprocessor()
    preprocessor.preprocess_adata(adata, recipe="seurat")
    # TODO add assert comparison later. Now checked by notebooks only.


def test_is_nonnegative():
    test_mat = csr_matrix([[1, 2, 0, 1, 5], [0, 0, 3, 1, 299], [4, 0, 5, 1, 399]])
    assert is_integer_arr(test_mat)
    assert is_nonnegative(test_mat)
    assert is_nonnegative_integer_arr(test_mat)
    test_mat = test_mat.toarray()
    assert is_integer_arr(test_mat)
    assert is_nonnegative(test_mat)
    assert is_nonnegative_integer_arr(test_mat)

    test_mat = csr_matrix([[-1, 2, 0, 1, 5], [0, 0, 3, 1, 299], [4, 0, 5, 1, 399]])
    assert is_integer_arr(test_mat)
    assert not is_nonnegative(test_mat)
    test_mat = test_mat.toarray()
    assert is_integer_arr(test_mat)
    assert not is_nonnegative(test_mat)

    test_mat = csr_matrix([[0, 2, 0, 1, 5], [0, 0, -3, 1, 299], [4, 0, 5, -1, 399]])
    assert is_integer_arr(test_mat)
    assert not is_nonnegative(test_mat)
    test_mat = test_mat.toarray()
    assert is_integer_arr(test_mat)
    assert not is_nonnegative(test_mat)

    test_mat = csr_matrix([[0, 2, 0, 1, 5], [0, 0, 5, 1, 299], [4, 0, 5, 5, 399]], dtype=float)
    assert is_float_integer_arr(test_mat)
    assert is_nonnegative_integer_arr(test_mat)
    test_mat = test_mat.toarray()
    assert is_float_integer_arr(test_mat)
    assert is_nonnegative_integer_arr(test_mat)

    test_mat = csr_matrix([[0, 2, 0, 1, 5], [0, 0, -3, 1, 299], [4, 0, 5, -1, 399.1]], dtype=float)
    assert not is_nonnegative_integer_arr(test_mat)
    test_mat = test_mat.toarray()
    assert not is_nonnegative_integer_arr(test_mat)


def test_filter_genes_by_clusters_():
    # Create test data
    n_cells = 1000
    n_genes = 500
    data = np.random.rand(n_cells, n_genes)
    layers = {
        "spliced": csr_matrix(data),
        "unspliced": csr_matrix(data),
    }
    adata = anndata.AnnData(X=data, layers=layers)

    # Add cluster information
    clusters = np.random.randint(low=0, high=3, size=n_cells)
    adata.obs['clusters'] = clusters

    # Filter genes by cluster
    clu_avg_selected = dyn.pp.filter_genes_by_clusters_(adata, 'clusters')

    # Check that the output is a numpy array
    assert type(clu_avg_selected) == np.ndarray

    # Check that the output has the correct shape
    assert clu_avg_selected.shape == (n_genes,)

    # Check that all genes with U and S average > min_avg_U and min_avg_S respectively are selected
    U, S = adata.layers['unspliced'], adata.layers['spliced']
    U_avgs = np.array([np.mean(U[clusters == i], axis=0) for i in range(3)])
    S_avgs = np.array([np.mean(S[clusters == i], axis=0) for i in range(3)])
    expected_clu_avg_selected = np.any((U_avgs.max(1) > 0.02) & (S_avgs.max(1) > 0.08), axis=0)
    assert np.array_equal(clu_avg_selected, expected_clu_avg_selected)


def test_filter_genes_by_outliers():
    # create a small test AnnData object
    data = np.array([[1, 0, 3, 0], [0, 1, 1, 0], [0, 1, 2, 1], [0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 0, 1]])
    adata = anndata.AnnData(data)
    adata.obs_names = ["cell1", "cell2", "cell3", "cell4", "cell5", "cell6"]
    adata.var_names = ["gene1", "gene2", "gene3", "gene4"]

    filtered_adata = dyn.pp.filter_genes_by_outliers(
        adata,
        min_avg_exp_s=0.5,
        min_cell_s=2,
        max_avg_exp=2.5,
        min_count_s=2,
        inplace=False,
    )

    # check that the filtered object contains the correct values
    assert np.all(filtered_adata.values == [False, True, True, True])

    # check that the original object is unchanged
    assert np.all(adata.var_names.values == ["gene1", "gene2", "gene3", "gene4"])

    dyn.pp.filter_genes_by_outliers(adata,
                                    min_avg_exp_s=0.5,
                                    min_cell_s=2,
                                    max_avg_exp=2.5,
                                    min_count_s=2,
                                    inplace=True)

    # check that the adata has been updated
    assert adata.shape == (6, 3)
    assert np.all(adata.var_names.values == ["gene2", "gene3", "gene4"])


def test_filter_cells_by_outliers():
    # Create a test AnnData object with some example data
    adata = anndata.AnnData(
        X=np.array([[1, 0, 3], [4 ,0 ,0], [7, 8, 9], [10, 11, 12]]))
    adata.var_names = ["gene1", "gene2", "gene3"]
    adata.obs_names = ["cell1", "cell2", "cell3", "cell4"]

    # Test the function with custom range values
    dyn.pp.filter_cells_by_outliers(
        adata, min_expr_genes_s=2, max_expr_genes_s=6)

    assert np.array_equal(
        adata.obs_names.values,
        ["cell1", "cell3", "cell4"],
    )

    # Test the function with invalid layer value
    try:
        dyn.pp.filter_cells_by_outliers(adata, layer="invalid_layer")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_get_cell_phase():
    from collections import OrderedDict

    # create a mock anndata object with mock data
    adata = anndata.AnnData(
        X=pd.DataFrame(
            [[1, 2, 7, 4, 1], [4, 3, 3, 5, 16], [5, 26, 7, 18, 9], [8, 39, 1, 1, 12]],
            columns=["arglu1", "dynll1", "cdca5", "cdca8", "ckap2"],
        )
    )

    # expected output
    expected_output = pd.DataFrame(
        {
            "G1-S": [0.52330,-0.28244,-0.38155,-0.13393],
            "S": [-0.77308, 0.98018, 0.39221, -0.38089],
            "G2-M": [-0.33656, -0.27547, 0.70090, 0.35216],
            "M": [0.07714, -0.36019, -0.67685, 0.77044],
            "M-G1": [0.50919, -0.06209, -0.03472, -0.60778],
        },
    )

    # test the function output against the expected output
    np.allclose(get_cell_phase(adata).iloc[:, :5], expected_output)


def test_gene_selection_method():
    adata = dyn.sample_data.zebrafish()
    dyn.pl.basic_stats(adata)
    dyn.pl.highest_frac_genes(adata)

    # Drawing for the downstream analysis.
    # df = adata.obs.loc[:, ["nCounts", "pMito", "nGenes"]]
    # g = sns.PairGrid(df, y_vars=["pMito", "nGenes"], x_vars=["nCounts"], height=4)
    # g.map(sns.regplot, color=".3")
    # # g.set(ylim=(-1, 11), yticks=[0, 5, 10])
    # g.add_legend()
    # plt.show()

    bdata = adata.copy()
    cdata = adata.copy()
    ddata = adata.copy()
    edata = adata.copy()
    preprocessor = Preprocessor()

    starttime = timeit.default_timer()
    preprocessor.preprocess_adata(edata, recipe="monocle", gene_selection_method="gini")
    monocle_gini_result = edata.var.use_for_pca

    preprocessor.preprocess_adata(adata, recipe="monocle", gene_selection_method="cv_dispersion")
    monocle_cv_dispersion_result_1 = adata.var.use_for_pca

    preprocessor.preprocess_adata(bdata, recipe="monocle", gene_selection_method="fano_dispersion")
    monocle_fano_dispersion_result_2 = bdata.var.use_for_pca

    preprocessor.preprocess_adata(cdata, recipe="seurat", gene_selection_method="fano_dispersion")
    seurat_fano_dispersion_result_3 = cdata.var.use_for_pca

    preprocessor.preprocess_adata(ddata, recipe="seurat", gene_selection_method="seurat_dispersion")
    seurat_seurat_dispersion_result_4 = ddata.var.use_for_pca

    diff_count = sum(1 for x, y in zip(monocle_cv_dispersion_result_1, monocle_gini_result) if x != y)
    print(diff_count / len(monocle_cv_dispersion_result_1) * 100)

    diff_count = sum(1 for x, y in zip(monocle_cv_dispersion_result_1, monocle_fano_dispersion_result_2) if x != y)
    print(diff_count / len(monocle_cv_dispersion_result_1) * 100)

    diff_count = sum(1 for x, y in zip(monocle_fano_dispersion_result_2, seurat_fano_dispersion_result_3) if x != y)
    print(diff_count / len(monocle_fano_dispersion_result_2) * 100)

    diff_count = sum(1 for x, y in zip(seurat_fano_dispersion_result_3, seurat_seurat_dispersion_result_4) if x != y)
    print(diff_count / len(seurat_fano_dispersion_result_3) * 100)

    diff_count = sum(1 for x, y in zip(monocle_cv_dispersion_result_1, seurat_seurat_dispersion_result_4) if x != y)
    print(diff_count / len(monocle_cv_dispersion_result_1) * 100)

    print("The preprocess_adata() time difference is :", timeit.default_timer() - starttime)


def test_normalize():
    # Set up test data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    layers = {
        "spliced": csr_matrix(X),
        "unspliced": csr_matrix(X),
    }
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "batch": ["batch1", "batch2", "batch2"],
                "use_for_pca": [True, True, True],
            }
        ),
        var=pd.DataFrame(index=["gene1", "gene2"]),
        layers=layers,
    )
    adata.uns["pp"] = dict()

    # Call the function
    normalized = normalize(
        adata=adata,
        # norm_method=np.log1p,
    )

    # Assert that the output is a valid AnnData object
    assert isinstance(normalized, anndata.AnnData)

    # Assert that the shape of the expression matrix is the same
    assert normalized.X.shape == (3, 2)
    assert normalized.layers["X_spliced"].shape == (3, 2)

    # Assert that the normalization was applied correctly
    assert np.allclose(normalized.X, (X / adata.obs["Size_Factor"].values[:, None]))
    assert np.allclose(normalized.layers["X_spliced"].toarray(), (X / adata.obs["spliced_Size_Factor"].values[:, None]))


def test_regress_out():
    starttime = timeit.default_timer()
    celltype_key = "Cell_type"
    figsize = (10, 10)
    adata = dyn.read("./data/zebrafish.h5ad")  # dyn.sample_data.hematopoiesis_raw()
    dyn.pl.basic_stats(adata)
    dyn.pl.highest_frac_genes(adata)

    preprocessor = Preprocessor(regress_out_kwargs={"obs_keys": ["nCounts", "pMito"]})

    preprocessor.preprocess_adata(adata, recipe="monocle")
    dyn.tl.reduceDimension(adata, basis="pca")

    dyn.pl.umap(adata, color=celltype_key, figsize=figsize)
    print("The preprocess_adata() time difference is :", timeit.default_timer() - starttime)


if __name__ == "__main__":
    dyn.dynamo_logger.main_set_level("DEBUG")
    # test_is_nonnegative()

    # test_calc_dispersion_sparse()
    # test_select_genes_seurat(gen_or_read_zebrafish_data())

    # test_compute_gene_exp_fraction()
    # test_layers2csr_matrix()

    # # generate data if needed
    # adata = utils.gen_or_read_zebrafish_data()
    # test_is_log_transformed()
    # test_pca()
    # test_Preprocessor_simple_run(dyn.sample_data.zebrafish())

    # test_calc_dispersion_sparse()
    # # TODO use a fixture in future
    # test_highest_frac_genes_plot(adata.copy())
    # test_highest_frac_genes_plot_prefix_list(adata.copy())
    # test_recipe_monocle_feature_selection_layer_simple0()
    # test_filter_genes_by_clusters_()
    # test_filter_genes_by_outliers()
    # test_filter_cells_by_outliers()
    # test_get_cell_phase()
    # test_gene_selection_method()
    # test_normalize()
    # test_regress_out()
    pass
