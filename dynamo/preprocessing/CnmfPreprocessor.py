import os
from shutil import rmtree

import numpy as np
from anndata import AnnData

from ..configuration import DKM
from ..data_io import make_dir, read_h5ad
from .Preprocessor import Preprocessor


class CnmfPreprocessor(Preprocessor):
    def __init__(self, **kwargs) -> None:
        """A specialized preprocessor based on cNMF. Args used are the same as normal Preprocessor."""

        super().__init__(**kwargs)
        self.selected_K = 7
        self.n_iter = 200
        self.n_top_genes = 2000
        self.output_dir = "./cnmf_dyn_preprocess_temp"
        self.seed = 0
        self.density_threshold = 2.00
        self.run_name = "temp"
        self.adata_h5ad_path = os.path.join(self.output_dir, "temp_adata.h5ad")

        self.tkey = None
        self.experiment_type = None
        # TODO: enable parallel computing in the future. Currently cNMF only provides cmd interfaces for factorization.
        self.num_worker = 1

    def preprocess_adata(self, adata: AnnData) -> AnnData:
        """Preprocess the AnnData object with cNMF.

        Args:
            adata: an AnnData object.

        Returns:
            The preprocessed AnnData object.
        """

        try:
            from cnmf import cNMF
        except Exception as e:
            print("Exception when importing CNMF")
            print("detailed exception:", str(e))

        make_dir(self.output_dir)
        counts_fn = self.adata_h5ad_path
        self.standardize_adata(adata, tkey=self.tkey, experiment_type=self.experiment_type)

        adata.write_h5ad(counts_fn)
        cnmf_obj = cNMF(output_dir=self.output_dir, name=self.run_name)
        cnmf_obj.prepare(
            counts_fn=counts_fn,
            components=np.arange(5, 11),
            n_iter=self.n_iter,
            seed=self.seed,
            num_highvar_genes=self.n_top_genes,
        )
        cnmf_obj.factorize(worker_i=0, total_workers=1)
        cnmf_obj.combine()
        cnmf_obj.consensus(
            k=self.selected_K,
            density_threshold=self.density_threshold,
            show_clustering=True,
            close_clustergram_fig=False,
        )
        adata = read_h5ad(counts_fn)
        hvg_path = os.path.join(self.output_dir, self.run_name, self.run_name + ".overdispersed_genes.txt")
        hvgs = open(hvg_path).read().split("\n")

        self.force_gene_list = hvgs
        self._force_gene_list(adata)
        adata = adata[:, adata.var[DKM.VAR_USE_FOR_PCA]]
        self._normalize_by_cells(adata)
        self._pca(adata)

        self.cnmf_obj = cnmf_obj
        return adata

    def k_selection_plot(self) -> None:
        """Plot the K selection curve of cNMF and save to the output folder."""

        self.cnmf_obj.k_selection_plot(close_fig=False)

    def cleanup_cnmf(self) -> None:
        """Remove the tmp folder to store data used for cNMF."""

        rmtree(self.output_dir, ignore_errors=True)
