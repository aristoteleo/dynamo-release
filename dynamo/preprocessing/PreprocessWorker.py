from dynamo.dynamo_logger import main_info
from dynamo.preprocessing.gene_selection_utils import log1p
from anndata import AnnData

from dynamo.tools.connectivity import neighbors as default_neighbors


class PreprocessWorker:
    def __init__(
        self, normalize_by_cells_function=None, filter_genes_function=None, use_log1p=True, neighbors=default_neighbors
    ) -> None:
        self.normalize_by_cells = normalize_by_cells_function
        self.filter_genes = filter_genes_function
        self.use_log1p = use_log1p
        self.log1p = log1p
        self.neighbors = neighbors

    def preprocess_adata(self, adata: AnnData):
        main_info("Running preprocessing pipeline")
        if self.normalize_by_cells:
            main_info("applying normalizing by cells function...")
            self.normalize_by_cells(adata)

        if self.use_log1p:
            main_info("applying log1p transformation on data...")
            log1p(adata)

        if self.neighbors:
            main_info("searching for neighbors...")
            self.neighbors(adata)

        if self.filter_genes:
            main_info("applying filter genes function...")
            self.filter_genes(adata)
