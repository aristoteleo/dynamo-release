# import utils

import pytest

import dynamo as dyn
from dynamo.pl import space


def test_space_plot_simple1(color=["clusters"]):
    adata = dyn.sample_data.pancreatic_endocrinogenesis()
    adata = adata[:1000, :1000].copy()
    adata.obsm["spatial"] = adata.obsm["X_umap"]
    adata.obsm["X_spatial"] = adata.obsm["X_umap"]
    space(adata, marker="p", save_show_or_return="show")
    space(adata, color=color, marker="*", save_show_or_return="show")


@pytest.mark.skip(reason="todo: add test data for spatial genomics")
def test_space_stack_color(utils):
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
