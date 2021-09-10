from dynamo.preprocessing import PreprocessWorker
from scipy.sparse.csr import csr_matrix
from dynamo.preprocessing.gene_selection_utils import calc_mean_var_dispersion_sparse
from utils import *
import dynamo as dyn

logger = LoggerManager.get_main_logger()
SHOW_FIG = True


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
    rpe1 = dyn.sample_data.scEU_seq()
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


def test_filter_by_dispersion_sparse():
    # TODO add randomize tests
    sparse_mat = csr_matrix([[1, 2, 0, 1], [0, 0, 3, 1], [4, 0, 5, 1]])
    mean, var, dispersion = calc_mean_var_dispersion_sparse(sparse_mat)
    expected_mean = np.mean(sparse_mat.toarray(), axis=0)
    expected_var = np.var(sparse_mat.toarray(), axis=0)
    expected_dispersion = expected_var / expected_mean
    assert np.all((mean - expected_mean) == 0)
    assert np.all((var - expected_var) == 0)
    assert np.all((dispersion - expected_dispersion) == 0)


def test_PreprocessWorker_simple_run(adata):
    preprocess_worker = PreprocessWorker()
    preprocess_worker.preprocess_adata(adata)


if __name__ == "__main__":
    # # generate data if needed
    adata = gen_or_read_zebrafish_data()
    test_PreprocessWorker_simple_run(adata)
    # test_filter_by_dispersion_sparse()
    # # # TODO use a fixture in future
    # test_highest_frac_genes_plot(adata.copy())
    # test_highest_frac_genes_plot_prefix_list(adata.copy())
    # test_recipe_monocle_feature_selection_layer_simple0()
