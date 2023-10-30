# import utils

import pytest

import dynamo as dyn
from dynamo.pl import space


def test_space_plot_simple1(processed_zebra_adata, color=["Cell_type"]):
    processed_zebra_adata.obsm["spatial"] = processed_zebra_adata.obsm["X_umap"]
    processed_zebra_adata.obsm["X_spatial"] = processed_zebra_adata.obsm["X_umap"]
    space(processed_zebra_adata, marker="p", save_show_or_return="show")
    space(processed_zebra_adata, color=color, marker="*", save_show_or_return="show")


@pytest.mark.skip(reason="todo: add test data for spatial genomics")
def test_space_stack_color(adata, utils):
    adata = utils.read_test_spatial_genomics_data()
    genes = adata.var_names[:18]
    # space(adata, genes=genes, marker="*", save_show_or_return="show", stack_genes=False)
    space(
        adata,
        genes=genes,
        marker=".",
        save_show_or_return="save",
        stack_genes=True,
        alpha=0.5,
        show_colorbar=False,
    )


@pytest.mark.skip(reason="todo: add test data for spatial genomics")
def test_space_data():
    adata = dyn.read_h5ad("allstage_splice.h5ad")


if __name__ == "__main__":
    # generate data if needed
    # adata = utils.gen_or_read_zebrafish_data()

    # TODO use a fixture in future
    # test_space_plot_simple1(adata)
    # test_space_stack_color(adata)
    pass
