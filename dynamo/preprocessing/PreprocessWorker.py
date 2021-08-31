from typing import Callable
from anndata import AnnData

from ..tools.connectivity import neighbors as default_neighbors
from ..dynamo_logger import main_info
from .gene_selection_utils import (
    filter_genes_by_dispersion_general,
    filter_genes_by_outliers as default_filter_genes_by_outliers,
    log1p,
)


class PreprocessWorker:
    def __init__(
        self,
        filter_genes_by_outliers_function: Callable = default_filter_genes_by_outliers,
        normalize_by_cells_function: Callable = None,
        filter_genes_function: Callable = filter_genes_by_dispersion_general,
        use_log1p: bool = True,
    ) -> None:
        self.filter_genes_by_outliers = filter_genes_by_outliers_function
        self.normalize_by_cells = normalize_by_cells_function
        self.filter_genes = filter_genes_function
        self.use_log1p = use_log1p
        self.log1p = log1p

    def preprocess_adata(self, adata: AnnData):
        main_info("Running preprocessing pipeline")
        if self.filter_genes_by_outliers:
            main_info("filtering outlier genes...")
            self.filter_genes_by_outliers(adata)

        if self.normalize_by_cells:
            main_info("applying normalizing by cells function...")
            self.normalize_by_cells(adata)

        if self.use_log1p:
            main_info("applying log1p transformation on data...")
            log1p(adata)

        if self.filter_genes:
            main_info("applying filter genes function...")
            self.filter_genes(adata)
