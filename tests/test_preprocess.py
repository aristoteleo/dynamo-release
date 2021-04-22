from test_utils import *
import dynamo as dyn

logger = LoggerManager.get_main_logger()


def test_highest_frac_genes_plot(adata, is_X_sparse=True):
    dyn.pl.highest_frac_genes(
        adata, log=False, save_path="./test_simple_highest_frac_genes.png"
    )

    if is_X_sparse:
        adata.X = adata.X.toarray()
        dyn.pl.highest_frac_genes(adata)


def test_highest_frac_genes_plot_prefix_list(adata, is_X_sparse=True):
    dyn.pl.highest_frac_genes(adata, gene_prefix_list=["RPL", "MRPL"])
    dyn.pl.highest_frac_genes(
        adata, gene_prefix_list=["someGenePrefixNotExisting"], show=False
    )

    if is_X_sparse:
        adata.X = adata.X.toarray()
        dyn.pl.highest_frac_genes(adata)


if __name__ == "__main__":
    # generate data if needed
    adata = gen_or_read_zebrafish_data()

    # To-do: use a fixture in future
    adata = dyn.read_h5ad(test_zebrafish_data_path)
    test_highest_frac_genes_plot(adata.copy())
    test_highest_frac_genes_plot_prefix_list(adata.copy())
