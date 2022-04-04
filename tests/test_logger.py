import time

import pytest

import dynamo as dyn
import dynamo.tools
from dynamo import LoggerManager
from dynamo.dynamo_logger import main_critical, main_info, main_tqdm, main_warning


@pytest.fixture
def test_logger():
    return LoggerManager.get_main_logger()


def test_logger_simple_output_1(test_logger):
    print()  # skip the first pytest default log line with script name
    test_logger.info("someInfoMessage")
    test_logger.warning("someWarningMessage", indent_level=2)
    test_logger.critical("someCriticalMessage", indent_level=3)
    test_logger.critical("someERRORMessage", indent_level=2)


def test_logger_simple_progress_naive(test_logger):
    total = 10
    test_logger.log_time()
    for i in range(total):
        # test_logger.report_progress(i / total * 100)
        test_logger.report_progress(count=i, total=total)
        time.sleep(0.1)
    test_logger.finish_progress(progress_name="pytest simple progress logger test")


def test_logger_simple_progress_logger(test_logger):
    total = 10
    test_logger.log_time()
    for _ in LoggerManager.progress_logger(
        range(total),
        test_logger,
        progress_name="progress logger test looooooooooooooooooooooooooooooong",
    ):
        time.sleep(0.1)


def test_tqdm_style_loops():
    for i in enumerate(main_tqdm(range(1, 11), desc="using TQDM style logging")):
        time.sleep(0.1)
    for i in main_tqdm(range(1, 11), desc="using TQDM style logging"):
        time.sleep(0.1)
    for i in LoggerManager.progress_logger(range(1, 11), progress_name="using LoggerManager's progress_logger"):
        time.sleep(0.1)


@pytest.mark.skip(reason="excessive running time")
def test_vectorField_logger():
    adata = dyn.sample_data.zebrafish()
    adata = adata[:500]
    dyn.pp.recipe_monocle(adata, num_dim=20, exprs_frac_for_gene_exclusion=0.005)
    dyn.tl.dynamics(adata, model="stochastic", cores=8)
    dyn.tl.reduceDimension(adata, n_pca_components=5, enforce=True)
    dyn.tl.cell_velocities(adata, basis="pca")
    dyn.vf.VectorField(adata, basis="pca", M=100)
    dyn.vf.VectorField(adata, basis="pca", M=100)
    dyn.vf.VectorField(adata, basis="pca", M=100)
    dyn.vf.curvature(adata, basis="pca")
    dyn.vf.acceleration(adata, basis="pca")
    dyn.vf.rank_acceleration_genes(adata, groups="Cell_type")
    dyn.pp.top_pca_genes(adata)
    top_pca_genes = adata.var.index[adata.var.top_pca_genes]
    dyn.vf.jacobian(adata, regulators=top_pca_genes, effectors=top_pca_genes)


@pytest.mark.skip(reason="excessive running time")
def test_sparseVFC_logger():
    adata = dyn.sample_data.zebrafish()
    adata = adata[:500]
    dyn.pp.recipe_monocle(adata, num_dim=20, exprs_frac_for_gene_exclusion=0.005)
    dyn.tl.dynamics(adata, model="stochastic", cores=8)
    dyn.tl.reduceDimension(adata, n_pca_components=5, enforce=True)
    dyn.tl.cell_velocities(adata, basis="pca")
    dyn.vf.VectorField(adata, basis="pca", M=100, method="SparseVFC", verbose=1)


@pytest.mark.skip(reason="need refactor: follow latest differential geometry notebook")
# TODO: refactor
def test_zebrafish_topography_tutorial_logger():
    adata = dyn.sample_data.zebrafish()
    adata = adata[:500]
    dyn.pp.recipe_monocle(adata, num_dim=20, exprs_frac_for_gene_exclusion=0.005)
    dyn.tl.dynamics(adata, model="stochastic", cores=8)
    dyn.tl.reduceDimension(adata, n_pca_components=5, enforce=True)
    dyn.tl.cell_velocities(adata, basis="pca")
    dyn.vf.VectorField(adata, basis="pca", M=100)
    dyn.vf.curvature(adata, basis="pca")
    dyn.vf.acceleration(adata, basis="pca")
    dyn.vf.rank_acceleration_genes(adata, groups="Cell_type")
    dyn.pp.top_pca_genes(adata)
    top_pca_genes = adata.var.index[adata.var.top_pca_genes]
    dyn.vf.jacobian(adata, regulators=top_pca_genes, effectors=top_pca_genes)
    dyn.pd.state_graph(adata, group="Cell_type", basis="pca", method="vf")


def test_cell_cycle_score_logger_pancreatic_endocrinogenesis():
    adata = dyn.sample_data.pancreatic_endocrinogenesis()
    adata = adata[:1000]
    dyn.pp.recipe_monocle(
        adata,
        n_top_genes=1000,
        fg_kwargs={"shared_count": 20},
        # genes_to_append=['Xkr4', 'Gm28672', 'Gm20837'],
        genes_to_exclude=["Sbspon", "Adgrb3", "Eif2s3y"],
    )
    dyn.pp.cell_cycle_scores(adata)


if __name__ == "__main__":
    test_tqdm_style_loops()

    test_logger_simple_output_1(LoggerManager.get_main_logger())
    test_logger_simple_progress_naive(LoggerManager.get_main_logger())
    test_logger_simple_progress_logger(LoggerManager.get_main_logger())
    test_logger_simple_progress_logger(LoggerManager.get_temp_timer_logger())

    test_vectorField_logger()
    test_zebrafish_topography_tutorial_logger()
    test_cell_cycle_score_logger_pancreatic_endocrinogenesis()
    test_sparseVFC_logger()
