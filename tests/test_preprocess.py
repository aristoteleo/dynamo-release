import timeit

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import PCA

# from utils import *
import dynamo as dyn
from dynamo.preprocessing import Preprocessor
from dynamo.preprocessing.cell_cycle import get_cell_phase
from dynamo.preprocessing.deprecated import _calc_mean_var_dispersion_sparse_legacy
from dynamo.preprocessing.normalization import calc_sz_factor, normalize
from dynamo.preprocessing.transform import log, log1p, log2, Freeman_Tukey, is_log1p_transformed_adata
from dynamo.preprocessing.utils import (
    convert_layers2csr,
    is_float_integer_arr,
    is_integer_arr,
    is_nonnegative,
    is_nonnegative_integer_arr,
)

SHOW_FIG = False


@pytest.mark.skip(reason="will be moved to plot tests")
def test_highest_frac_genes_plot(processed_zebra_adata, is_X_sparse=True):
    dyn.pl.highest_frac_genes(
        processed_zebra_adata,
        show=SHOW_FIG,
        log=False,
        save_path="./test_simple_highest_frac_genes.png",
    )
    dyn.pl.highest_frac_genes(
        processed_zebra_adata,
        log=False,
        show=SHOW_FIG,
        save_path="test_simple_highest_frac_genes.png",
    )
    dyn.pl.highest_frac_genes(
        processed_zebra_adata,
        log=False,
        show=SHOW_FIG,
        save_path="test_simple_highest_frac_genes.png",
    )
    dyn.pl.highest_frac_genes(
        processed_zebra_adata,
        log=False,
        show=SHOW_FIG,
        save_path="test_simple_highest_frac_genes.png",
        orient="h",
    )
    dyn.pl.highest_frac_genes(
        processed_zebra_adata,
        log=False,
        show=SHOW_FIG,
        save_path="test_simple_highest_frac_genes.png",
        orient="h",
    )
    dyn.pl.highest_frac_genes(
        processed_zebra_adata,
        log=False,
        show=SHOW_FIG,
        save_path="test_simple_highest_frac_genes.png",
        layer="M_s",
    )

    if is_X_sparse:
        processed_zebra_adata.X = processed_zebra_adata.X.toarray()
        dyn.pl.highest_frac_genes(processed_zebra_adata, show=SHOW_FIG)


@pytest.mark.skip(reason="need full test data")
def test_highest_frac_genes_plot_prefix_list(processed_zebra_adata):
    sample_list = ["MT-", "RPS", "RPL", "MRPS", "MRPL", "ERCC-"]
    dyn.pl.highest_frac_genes(processed_zebra_adata, show=SHOW_FIG, gene_prefix_list=sample_list)
    dyn.pl.highest_frac_genes(processed_zebra_adata, show=SHOW_FIG, gene_prefix_list=["RPL", "MRPL"])

    try:
        dyn.pl.highest_frac_genes(
            processed_zebra_adata,
            gene_prefix_list=["someGenePrefixNotExisting"],
            show=SHOW_FIG,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError to be raised")


@pytest.mark.skip(reason="optional dependency mygene not installed")
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

    del rpe1, rpe1_kinetics.layers["uu"], rpe1_kinetics.layers["ul"], rpe1_kinetics.layers["su"], rpe1_kinetics.layers["sl"]
    rpe1_kinetics = rpe1_kinetics[:100, :100]
    dyn.pl.basic_stats(rpe1_kinetics, save_show_or_return="return")
    rpe1_genes = ["UNG", "PCNA", "PLK1", "HPRT1"]

    # rpe1_kinetics = dyn.pp.recipe_monocle(rpe1_kinetics, n_top_genes=1000, total_layers=False, copy=True)
    dyn.pp.recipe_monocle(rpe1_kinetics, n_top_genes=20, total_layers=False, feature_selection_layer="new")
    assert np.all(rpe1_kinetics.X.A == rpe1_kinetics.X.A)
    assert not np.all(rpe1_kinetics.layers["new"].A != rpe1_kinetics.layers["new"].A)


def test_calc_dispersion_sparse():
    # TODO add randomize tests
    sparse_mat = csr_matrix([[1, 2, 0, 1, 5], [0, 0, 3, 1, 299], [4, 0, 5, 1, 399]])
    mean, var, dispersion = _calc_mean_var_dispersion_sparse_legacy(sparse_mat)
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
    # sc_mean, sc_var = dyn.preprocessing.utils.seurat_get_mean_var(sparse_mat)
    # print("sc_mean:", sc_mean)
    # print("expected mean:", sc_mean)
    # print("sc_var:", sc_var)
    # print("expected var:", expected_var)
    # assert np.all(np.isclose(sc_mean, expected_mean))
    # assert np.all(np.isclose(sc_var, expected_var))



def test_Preprocessor_monocle_recipe():
    raw_zebra_adata = dyn.sample_data.zebrafish()
    adata = raw_zebra_adata[:1000, :1000].copy()
    del raw_zebra_adata
    preprocess_worker = Preprocessor(cell_cycle_score_enable=True)
    preprocess_worker.preprocess_adata(adata, recipe="monocle")
    assert "X_pca" in adata.obsm.keys()


def test_Preprocessor_seurat_recipe():
    raw_zebra_adata = dyn.sample_data.zebrafish()
    adata = raw_zebra_adata[:1000, :1000].copy()
    del raw_zebra_adata
    preprocess_worker = Preprocessor(cell_cycle_score_enable=True)
    preprocess_worker.preprocess_adata(adata, recipe="seurat")
    assert "X_pca" in adata.obsm.keys()


def test_Preprocessor_pearson_residuals_recipe():
    raw_zebra_adata = dyn.sample_data.zebrafish()
    adata = raw_zebra_adata[:1000, :1000].copy()
    del raw_zebra_adata
    preprocess_worker = Preprocessor(cell_cycle_score_enable=True)
    preprocess_worker.preprocess_adata(adata, recipe="pearson_residuals")
    assert "X_pca" in adata.obsm.keys()


def test_Preprocessor_monocle_pearson_residuals_recipe():
    raw_zebra_adata = dyn.sample_data.zebrafish()
    adata = raw_zebra_adata[:1000, :1000].copy()
    del raw_zebra_adata
    preprocess_worker = Preprocessor(cell_cycle_score_enable=True)
    preprocess_worker.preprocess_adata(adata, recipe="monocle_pearson_residuals")
    assert "X_pca" in adata.obsm.keys()


@pytest.mark.skip(reason="optional dependency KDEpy not installed")
def test_Preprocessor_sctransform_recipe():
    raw_zebra_adata = dyn.sample_data.zebrafish()
    adata = raw_zebra_adata[:1000, :1000].copy()
    del raw_zebra_adata
    preprocess_worker = Preprocessor(cell_cycle_score_enable=True)
    preprocess_worker.preprocess_adata(adata, recipe="sctransform")
    assert "X_pca" in adata.obsm.keys()


def test_is_log_transformed():
    adata = dyn.sample_data.zebrafish()
    adata.uns["pp"] = {}
    assert not is_log1p_transformed_adata(adata)
    log1p(adata)
    assert is_log1p_transformed_adata(adata)


def test_layers2csr_matrix():
    data = np.array([[1, 2], [3, 4]])
    adata = anndata.AnnData(
        X=data,
        obs={'obs1': ['cell1', 'cell2']},
        var={'var1': ['gene1', 'gene2']},
    )
    layer = csr_matrix([[1, 2], [3, 4]]).transpose()  # Transpose the matrix
    adata.layers['layer1'] = layer

    result = dyn.preprocessing.utils.convert_layers2csr(adata)

    assert issparse(result.layers['layer1'])
    assert result.layers['layer1'].shape == layer.shape
    assert (result.layers['layer1'].toarray() == layer.toarray()).all()


def test_compute_gene_exp_fraction():
    # TODO fix compute_gene_exp_fraction: discuss with Xiaojie
    # df = pd.DataFrame([[1, 2], [1, 1]]) # input cannot be dataframe
    df = csr_matrix([[1, 2], [1, 1]])
    frac, indices = dyn.preprocessing.compute_gene_exp_fraction(df)
    print("frac:", list(frac))
    assert np.all(np.isclose(frac.flatten(), [2 / 5, 3 / 5]))


def test_pca():
    adata = dyn.sample_data.zebrafish()
    adata = adata[:2000, :5000].copy()
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


@pytest.mark.skip(reason="unhelpful test")
def test_preprocessor_seurat(raw_zebra_adata):
    adata = raw_zebra_adata.copy()
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
    clu_avg_selected = dyn.pp.filter_genes_by_clusters(adata, 'clusters')

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


def test_filter_genes_by_patterns():
    adata = anndata.AnnData(
        X=np.array([[1, 0, 3], [4, 0, 0], [7, 8, 9], [10, 11, 12]]))
    adata.var_names = ["MT-1", "RPS", "GATA1"]
    adata.obs_names = ["cell1", "cell2", "cell3", "cell4"]

    matched_genes = dyn.pp.filter_genes_by_pattern(adata, drop_genes=False)
    dyn.pp.filter_genes_by_pattern(adata, drop_genes=True)

    assert matched_genes == [True, True, False]
    assert np.array_equal(
        adata.var_names.values,
        ["GATA1"],
    )


@pytest.mark.skip(reason="skip this temporarily, waiting for debugging in master branch")
def test_lambda_correction():
    adata = anndata.AnnData(
        X=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        obs={"lambda": [0.1, 0.2, 0.3]},
        layers={
            "ul_layer": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            "un_layer": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            "sl_layer": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            "sn_layer": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            "unspliced": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            "spliced": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        },
    )
    # adata = dyn.sample_data.zebrafish()
    # adata.obs["lambda"] = [0.1 for i in range(adata.shape[0])]

    dyn.pp.lambda_correction(adata, lambda_key="lambda", inplace=False)

    assert "ul_layer_corrected" in adata.layers.keys()


def test_top_pca_genes():
    adata = anndata.AnnData(
        X=np.random.rand(10, 5),  # 10 cells, 5 genes
        uns={"PCs": np.random.rand(5, 5)},  # Random PC matrix for testing
        varm={"PCs": np.random.rand(5, 5)},
        var={"gene_names": ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]},
    )

    dyn.pp.top_pca_genes(adata, pc_key="PCs", n_top_genes=3, pc_components=2, adata_store_key="top_pca_genes")

    assert "top_pca_genes" in adata.var.keys()
    assert sum(adata.var["top_pca_genes"]) >= 3


def test_vst_exprs():
    adata = anndata.AnnData(
        X=np.random.rand(10, 5),  # 10 cells, 5 genes
        uns={"dispFitInfo": {"coefs": np.array([0.1, 0.2])}},  # Random dispersion coefficients for testing
    )

    result_exprs = dyn.preprocessing.transform.vstExprs(adata)

    assert result_exprs.shape == (10, 5)
    assert np.all(np.isfinite(result_exprs))


def test_cell_cycle_scores():
    adata = anndata.AnnData(
        X=pd.DataFrame(
            [[1, 2, 7, 4, 1], [4, 3, 3, 5, 16], [5, 26, 7, 18, 9], [8, 39, 1, 1, 12]],
            columns=["arglu1", "dynll1", "cdca5", "cdca8", "ckap2"],
        )
    )

    dyn.pp.cell_cycle_scores(adata)
    assert "cell_cycle_scores" in adata.obsm.keys()
    assert "cell_cycle_phase" in adata.obs.keys()
    assert np.all(list(adata.obsm["cell_cycle_scores"].iloc[:, :5].columns) == ["G1-S", "S", "G2-M", "M", "M-G1"])


def test_gene_selection_methods_in_preprocessor():
    raw_adata = dyn.sample_data.zebrafish()
    adata = raw_adata[:100, :500].copy()
    dyn.pl.basic_stats(adata)
    dyn.pl.highest_frac_genes(adata)

    preprocessor = Preprocessor()

    preprocessor.config_monocle_recipe(adata)
    preprocessor.select_genes_kwargs = {"sort_by": "gini", "n_top_genes": 50}
    preprocessor.preprocess_adata_monocle(adata)
    result = adata.var.use_for_pca

    assert result.shape[0] == 500
    assert np.count_nonzero(result) <= 50

    adata = raw_adata[:100, :500].copy()
    preprocessor.config_monocle_recipe(adata)
    preprocessor.select_genes_kwargs = {"sort_by": "cv_dispersion", "n_top_genes": 50}
    preprocessor.preprocess_adata_monocle(adata)
    result = adata.var.use_for_pca

    assert result.shape[0] == 500
    assert np.count_nonzero(result) <= 50

    adata = raw_adata[:100, :500].copy()
    preprocessor.config_monocle_recipe(adata)
    preprocessor.select_genes_kwargs = {"sort_by": "fano_dispersion", "n_top_genes": 50}
    preprocessor.preprocess_adata_monocle(adata)
    result = adata.var.use_for_pca

    assert result.shape[0] == 500
    assert np.count_nonzero(result) <= 50

    adata = raw_adata[:100, :500].copy()
    preprocessor.config_monocle_recipe(adata)
    preprocessor.select_genes_kwargs["n_top_genes"] = 50
    preprocessor.preprocess_adata_seurat(adata)
    result = adata.var.use_for_pca

    assert result.shape[0] == 500
    assert np.count_nonzero(result) <= 50


def test_transform():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layers = {
        "spliced": csr_matrix(X),
    }
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "batch": ["batch1", "batch2", "batch2"],
            }
        ),
        var=pd.DataFrame(index=["gene1", "gene2"]),
        layers=layers,
    )
    adata.uns["pp"] = dict()

    adata1 = log1p(adata.copy())
    assert not np.all(adata1.X == adata.X)
    assert np.all(adata1.layers["spliced"].A == adata.layers["spliced"].A)

    adata2 = log(adata.copy())
    assert not np.all(adata2.X == adata.X)
    assert np.all(adata2.layers["spliced"].A == adata.layers["spliced"].A)

    adata3 = log2(adata.copy())
    assert not np.all(adata3.X == adata.X)
    assert np.all(adata3.layers["spliced"].A == adata.layers["spliced"].A)

    adata4 = Freeman_Tukey(adata.copy())
    assert not np.all(adata3.X == adata.X)
    assert np.all(adata4.layers["spliced"].A == adata.layers["spliced"].A)


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
    calc_sz_factor(adata)
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
    adata = dyn.sample_data.zebrafish()  # dyn.sample_data.hematopoiesis_raw()
    # dyn.pl.basic_stats(adata)
    # dyn.pl.highest_frac_genes(adata)

    preprocessor = Preprocessor(regress_out_kwargs={"obs_keys": ["nCounts", "pMito"]})

    preprocessor.preprocess_adata(adata, recipe="monocle")
    dyn.tl.reduceDimension(adata, basis="pca")

    dyn.pl.umap(adata, color=celltype_key, figsize=figsize)
    print("The preprocess_adata() time difference is :", timeit.default_timer() - starttime)
