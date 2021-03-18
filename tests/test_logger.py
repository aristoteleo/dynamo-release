from numba import jit
import dynamo.tools
from dynamo import LoggerManager
import dynamo as dyn
import pytest


@pytest.fixture
def test_logger():
    return LoggerManager.get_main_logger()


def test_logger_simple_output_1(test_logger):
    print()  # skip the first pytest default log line with script name
    test_logger.info("someInfoMessage")
    test_logger.warning("someWarningMessage", indent_level=2)
    test_logger.critical("someCriticalMessage", indent_level=3)
    test_logger.critical("someERRORMessage", indent_level=2)


def test_vectorField_logger(test_logger):
    adata = dyn.sample_data.zebrafish()
    dyn.pp.recipe_monocle(adata, num_dim=20, exprs_frac_max=0.005)
    dyn.tl.dynamics(adata, model="stochastic", cores=8)
    dyn.tl.cell_velocities(adata, basis="pca")
    dyn.vf.VectorField(adata, basis="pca", M=1000)


if __name__ == "__main__":
    test_vectorField_logger(test_logger)
