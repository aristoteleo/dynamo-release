from anndata import AnnData
from umap import UMAP
from sklearn.decomposition import PCA
import numpy as np
from scvelo import logging as logg


def process_kinetic_rates(adata, mode=["total_map"], seed=0):
    if "cell_specific_alpha" in adata.layers:
        all_rates = np.concatenate(
            [
                adata.layers["cell_specific_beta"],
                adata.layers["cell_specific_gamma"],
                adata.layers["cell_specific_alpha"],
            ],
            axis=1,
        )
    else:
        all_rates = np.concatenate(
            [
                adata.layers["cell_specific_beta"],
                adata.layers["cell_specific_gamma"],
            ],
            axis=1,
        )

    # pca and umap of all rates
    if "total_map" in mode and "X_rates_umap" not in adata.obsm:
        rates_pca = PCA(n_components=30, random_state=seed).fit_transform(all_rates)
        adata.obsm["X_rates_pca"] = rates_pca
        logg.info("Added `X_rates_pca` (adata.obsm)")

        rates_umap = UMAP(
            n_neighbors=60,
            min_dist=0.6,
            spread=0.9,
            random_state=seed,
        ).fit_transform(rates_pca)
        adata.obsm["X_rates_umap"] = rates_umap

        logg.info("Added `X_rates_umap` (adata.obsm)")

    # # pca and umap of gene-wise rates
    # if "gene_wise_map" in mode:
    #     rates_pca_gene_wise = PCA(n_components=30, random_state=seed).fit_transform(
    #         adata.layers["Ms"].T
    #     )
    #     adata.varm["rates_pca"] = rates_pca_gene_wise

    #     rates_umap_gene_wise = UMAP(
    #         n_neighbors=60,
    #         min_dist=0.6,
    #         spread=0.9,
    #         random_state=seed,
    #     ).fit_transform(rates_pca_gene_wise)
    #     adata.varm["rates_umap"] = rates_umap_gene_wise
