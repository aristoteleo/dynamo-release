from dynamo import LoggerManager
import dynamo.tools
import dynamo as dyn
import pytest
import time


@pytest.fixture
def test_logger():
    return LoggerManager.get_main_logger()


def test_logger_simple_output_1(test_logger):
    print()  # skip the first pytest default log line with script name
    test_logger.info("someInfoMessage")
    test_logger.warning("someWarningMessage", indent_level=2)
    test_logger.critical("someCriticalMessage", indent_level=3)
    test_logger.critical("someERRORMessage", indent_level=2)


def test_logger_simple_progress_logger(test_logger):
    total = 10
    test_logger.log_time()
    for i in range(total):
        test_logger.report_progress(i / total * 100)
        time.sleep(0.1)
    test_logger.finish_progress()


# To-do:
# following test does not work with pytest but can be run in main directly
# the reason seems to be compatibility of pytest and numba
def test_vectorField_logger():
    adata = dyn.sample_data.zebrafish()
    dyn.pp.recipe_monocle(adata, num_dim=20, exprs_frac_max=0.005)
    dyn.tl.dynamics(adata, model="stochastic", cores=8)
    dyn.tl.reduceDimension(adata, n_pca_components=5, enforce=True)
    dyn.tl.cell_velocities(adata, basis="pca")
    dyn.vf.VectorField(adata, basis="pca", M=100)
    dyn.vf.VectorField(adata, basis="pca", M=100)
    dyn.vf.VectorField(adata, basis="pca", M=100)


if __name__ == "__main__":
    # test_logger_simple_output_1(LoggerManager.get_main_logger())
    # test_logger_simple_progress_logger(LoggerManager.get_main_logger())
    test_vectorField_logger()
