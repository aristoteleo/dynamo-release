from typing import Callable
from anndata import AnnData


from .gene_selection_utils import (
    _infer_labeling_experiment_type,
    is_log1p_transformed_adata,
    select_genes_by_dispersion_general,
    filter_genes_by_outliers as default_filter_genes_by_outliers,
    log1p_adata,
    filter_cells_by_outliers as default_filter_cells_by_outliers,
)
from .utils import collapse_species_adata, detect_experiment_datatype, convert2symbol
from ..tools.connectivity import neighbors as default_neighbors
from ..dynamo_logger import main_info, main_info_insert_adata, main_warning


class PreprocessWorker:
    def __init__(
        self,
        collapse_speicies_adata_function: Callable = collapse_species_adata,
        convert_gene_name_function: Callable = convert2symbol,
        filter_cells_by_outliers_function: Callable = default_filter_cells_by_outliers,
        filter_genes_by_outliers_function: Callable = default_filter_genes_by_outliers,
        normalize_by_cells_function: Callable = None,
        select_genes_function: Callable = select_genes_by_dispersion_general,
        use_log1p: bool = True,
        n_top_genes=2000,
    ) -> None:
        """Initialize the worker.

        Parameters
        ----------
        filter_genes_by_outliers_function : Callable, optional
            The first function in pipeline to filter gene outliers.
        normalize_by_cells_function : Callable, optional
            Normalize data according to cells (typically rows), by default None
        filter_genes_function : Callable, optional
            A function to filter genes after previous steps.The function should accept `n_top_genes`  as an argument or wildcard argument match.
        use_log1p : bool, optional
            Whether to apply log1p function to all data in adata, by default True
        n_top_genes : int, optional
            keep at most `n_top_genes` in adata for downstream analysis, by default 2000
        """
        self.filter_cells_by_outliers = filter_cells_by_outliers_function
        self.filter_genes_by_outliers = filter_genes_by_outliers_function
        self.normalize_by_cells = normalize_by_cells_function
        self.filter_genes = select_genes_function
        self.use_log1p = use_log1p
        self.log1p = log1p_adata
        self.n_top_genes = n_top_genes
        self.convert_gene_name = convert_gene_name_function
        self.collapse_species_adata = collapse_speicies_adata_function

    def preprocess_adata(self, adata: AnnData, tkey: str = "time", experiment_type: str = None):
        main_info("Running preprocessing pipeline...")
        if tkey is not None and adata.obs[tkey].max() > 60:
            main_warning(
                "Looks like you are using minutes as the time unit. For the purpose of numeric stability, "
                "we recommend using hour as the time unit."
            )

        adata.uns["pp"] = {}
        main_info_insert_adata("%s" % adata.uns["pp"], "uns['pp']", indent_level=2)

        (
            has_splicing,
            has_labeling,
            splicing_labeling,
            has_protein,
        ) = detect_experiment_datatype(adata)
        adata.uns["pp"]["tkey"] = tkey
        adata.uns["pp"]["has_splicing"] = has_splicing
        adata.uns["pp"]["has_labeling"] = has_labeling
        adata.uns["pp"]["has_protein"] = has_protein
        adata.uns["pp"]["splicing_labeling"] = splicing_labeling
        # infer and set experiment type
        if experiment_type is None and has_labeling:
            experiment_type = _infer_labeling_experiment_type(adata, tkey)
        if experiment_type is None:
            experiment_type = "conventional"
        adata.uns["pp"]["experiment_type"] = experiment_type

        main_info_insert_adata("tkey=%s" % tkey, "uns['pp']", indent_level=2)
        main_info_insert_adata("experiment_type=%s" % experiment_type, "uns['pp']", indent_level=2)

        if self.collapse_species_adata:
            main_info("applying collapse species adata...")
            self.convert_gene_name(adata)

        if self.convert_gene_name:
            main_info("applying filter genes function...")
            self.convert_gene_name(adata)

        if self.filter_cells_by_outliers:
            main_info("filtering outlier cells...")
            self.filter_cells_by_outliers(adata)

        if self.filter_genes_by_outliers:
            main_info("filtering outlier genes...")
            self.filter_genes_by_outliers(adata)

        if self.normalize_by_cells:
            main_info("applying normalizing by cells function...")
            self.normalize_by_cells(adata)

        if self.use_log1p:
            if is_log1p_transformed_adata(adata):
                main_warning(
                    "Your adata.X maybe log1p transformed before. If you are sure that your adata is not log1p transformed, please ignore this warning. Dynamo will do log1p transformation still."
                )
            main_info("applying log1p transformation on expression matrix data (adata.X)...")
            log1p_adata(adata)

        if self.filter_genes:
            main_info("applying filter genes function...")
            self.filter_genes(adata, n_top_genes=self.n_top_genes)
