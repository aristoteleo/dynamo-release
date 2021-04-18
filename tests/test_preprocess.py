from test_utils import *
import dynamo as dyn

logger = LoggerManager.get_main_logger()


def test_highly_expr_genes_plot(adata, is_X_sparse=True):
    dyn.pl.highest_expr_genes(adata)

    if is_X_sparse:
        adata.X = adata.X.toarray()
        dyn.pl.highest_expr_genes(adata)


if __name__ == "__main__":
    # generate data if needed
    if not os.path.exists(test_zebrafish_data_path):
        print("generating test data...")
        gen_zebrafish_test_data()

    print("reading test data...")
    # To-do: use a fixture in future
    adata = dyn.read_h5ad(test_zebrafish_data_path)
    test_highly_expr_genes_plot(adata)
