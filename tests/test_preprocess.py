import anndata
import numpy as np
import pandas as pd
import scipy
import scipy.sparse
from scipy import sparse
from scipy.sparse.csr import csr_matrix

# from utils import *
import dynamo as dyn
from dynamo.preprocessing import Preprocessor
from dynamo.preprocessing.preprocessor_utils import (
    calc_mean_var_dispersion_sparse,
    is_float_integer_arr,
    is_integer_arr,
    is_log1p_transformed_adata,
    is_nonnegative,
    is_nonnegative_integer_arr,
    log1p_adata,
    select_genes_by_dispersion_general,
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
    log1p_adata(adata)
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
    df = scipy.sparse.csr_matrix([[1, 2], [1, 1]])
    frac, indices = dyn.preprocessing.compute_gene_exp_fraction(df)
    print("frac:", list(frac))
    assert np.all(np.isclose(frac.flatten(), [2 / 5, 3 / 5]))


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


if __name__ == "__main__":
    # test_is_nonnegative()

    # test_calc_dispersion_sparse()
    # test_select_genes_seurat(gen_or_read_zebrafish_data())

    # test_compute_gene_exp_fraction()
    # test_layers2csr_matrix()

    # # generate data if needed
    # adata = utils.gen_or_read_zebrafish_data()
    # test_is_log_transformed()
    # test_Preprocessor_simple_run(dyn.sample_data.zebrafish())

    # test_calc_dispersion_sparse()
    # # TODO use a fixture in future
    # test_highest_frac_genes_plot(adata.copy())
    # test_highest_frac_genes_plot_prefix_list(adata.copy())
    # test_recipe_monocle_feature_selection_layer_simple0()
    pass
