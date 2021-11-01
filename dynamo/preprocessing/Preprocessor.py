from typing import Callable, List, Optional
from anndata import AnnData
import numpy as np

from .preprocessor_utils import (
    _infer_labeling_experiment_type,
    is_log1p_transformed_adata,
    normalize_cell_expr_by_size_factors,
    select_genes_by_dispersion_general,
    filter_genes_by_outliers as default_filter_genes_by_outliers,
    log1p_adata,
    filter_cells_by_outliers as default_filter_cells_by_outliers,
)

from .preprocess import normalize_cell_expr_by_size_factors_legacy, pca_monocle
from .utils import collapse_species_adata, detect_experiment_datatype, convert2symbol, unique_var_obs_adata
from ..tools.connectivity import neighbors as default_neighbors
from ..dynamo_logger import main_info, main_info_insert_adata, main_warning
from ..configuration import DKM


class Preprocessor:
    def __init__(
        self,
        collapse_speicies_adata_function: Callable = collapse_species_adata,
        convert_gene_name_function: Callable = convert2symbol,
        filter_cells_by_outliers_function: Callable = default_filter_cells_by_outliers,
        filter_cells_by_outliers_kwargs: Callable = {},
        filter_genes_by_outliers_function: Callable = default_filter_genes_by_outliers,
        filter_genes_by_outliers_kwargs: dict = {},
        normalize_by_cells_function: Callable = normalize_cell_expr_by_size_factors,
        normalize_by_cells_function_kwargs: Callable = {},
        select_genes_function: Callable = select_genes_by_dispersion_general,
        select_genes_kwargs: dict = {},
        normalize_selected_genes_function: Callable = None,
        use_log1p: bool = True,
        pca_function: bool = pca_monocle,
        pca_kwargs: dict = {},
        gene_append_list: List = [],
        gene_exclude_list: List = [],
        force_gene_list: Optional[List] = None,
        # n_top_genes=2000,
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
        """
        self.filter_cells_by_outliers = filter_cells_by_outliers_function
        self.filter_genes_by_outliers = filter_genes_by_outliers_function
        self.normalize_by_cells = normalize_by_cells_function
        self.select_genes = select_genes_function
        self.normalize_selected_genes = normalize_selected_genes_function
        self.use_log1p = use_log1p
        self.log1p = log1p_adata
        self.pca = pca_function
        self.pca_kwargs = pca_kwargs

        # self.n_top_genes = n_top_genes
        self.convert_gene_name = convert_gene_name_function
        self.collapse_species_adata = collapse_speicies_adata_function
        self.gene_append_list = gene_append_list
        self.gene_exclude_list = gene_exclude_list
        self.force_gene_list = force_gene_list
        self.filter_genes_by_outliers_kwargs = filter_genes_by_outliers_kwargs
        self.normalize_by_cells_function_kwargs = normalize_by_cells_function_kwargs
        self.filter_cells_by_outliers_kwargs = filter_cells_by_outliers_kwargs
        self.select_genes_kwargs = select_genes_kwargs

    def get_monocle_filter_genes_outliers_kwargs(adata: AnnData):
        n_obs = adata.n_obs
        default_filter_genes_by_outliers_kwargs = {
            "filter_bool": None,
            "layer": "all",
            "min_cell_s": max(5, 0.01 * n_obs),
            "min_cell_u": max(5, 0.005 * n_obs),
            "min_cell_p": max(5, 0.005 * n_obs),
            "min_avg_exp_s": 0,
            "min_avg_exp_u": 0,
            "min_avg_exp_p": 0,
            "max_avg_exp": np.inf,
            "min_count_s": 0,
            "min_count_u": 0,
            "min_count_p": 0,
            "shared_count": 30,
        }
        return default_filter_genes_by_outliers_kwargs

    def add_experiment_info(self, adata: AnnData, tkey: Optional[str] = None, experiment_type: str = None):
        if DKM.UNS_PP_KEY not in adata.uns.keys():
            adata.uns[DKM.UNS_PP_KEY] = {}
        main_info_insert_adata("%s" % adata.uns["pp"], "uns['pp']", indent_level=2)

        (
            has_splicing,
            has_labeling,
            splicing_labeling,
            has_protein,
        ) = detect_experiment_datatype(adata)
        # check whether tkey info exists if has_labeling
        if has_labeling:
            main_info("data contains labeling info, checking tkey:" + str(tkey))
            if tkey not in adata.obs.keys():
                raise ValueError("tkey:%s encoding the labeling time is not existed in your adata." % (str(tkey)))
            if tkey is not None and adata.obs[tkey].max() > 60:
                main_warning(
                    "Looks like you are using minutes as the time unit. For the purpose of numeric stability, "
                    "we recommend using hour as the time unit."
                )

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

        # infer new/total, splicing/unsplicing layers we have in experiment data
        layers, total_layers = None, None
        if has_splicing and has_labeling and splicing_labeling:
            layers = [
                "X",
                "uu",
                "ul",
                "su",
                "sl",
                "spliced",
                "unspliced",
                "new",
                "total",
            ]
            total_layers = ["uu", "ul", "su", "sl"]
        if has_splicing and has_labeling and not splicing_labeling:
            layers = ["X", "spliced", "unspliced", "new", "total"]
            total_layers = ["total"]
        elif has_labeling and not has_splicing:
            layers = ["X", "total", "new"]
            total_layers = ["total"]
        elif has_splicing and not has_labeling:
            layers = ["X", "spliced", "unspliced"]
        adata.uns["pp"]["experiment_layers"] = layers
        adata.uns["pp"]["experiment_total_layers"] = total_layers

    def config_pearson_residual_recipe(self):
        raise NotImplementedError("test in progress")  # TODO uncomment after integrate
        # self.select_genes=pearson_residual_normalization_recipe.select_genes_by_pearson_residual,
        # self.normalize_selected_genes=pearson_residual_normalization_recipe.normalize_layers_pearson_residuals,
        # self.use_log1p=False

    def config_monocle_recipe(self, adata: AnnData, n_top_genes: int = 2000, gene_selection_method: str = "SVR"):
        # TODO
        self.add_experiment_info(adata)
        self.use_log1p = False
        self.filter_cells_by_outliers_kwargs = {"keep_filtered": True}
        self.select_genes_kwargs = {
            "dynamo_monocle_kwargs": {
                "sort_by": gene_selection_method,
                "n_top_genes": n_top_genes,
                "keep_filtered": True,
                "SVRs_kwargs": {
                    "min_expr_cells": 0,
                    "min_expr_avg": 0,
                    "max_expr_avg": np.inf,
                    "svr_gamma": None,
                    "winsorize": False,
                    "winsor_perc": (1, 99.5),
                    "sort_inverse": False,
                },
                "only_bools": True,
            }
        }
        self.select_genes = select_genes_by_dispersion_general
        self.normalize_selected_genes = None
        self.normalize_by_cells = normalize_cell_expr_by_size_factors
        self.pca = pca_monocle
        self.pca_kwargs = {"pca_key": "X_pca"}

    def preprocess_adata(self, adata: AnnData, tkey: Optional[str] = None, experiment_type: str = None):
        main_info("Running preprocessing pipeline...")

        adata.uns["pp"] = {}
        self.add_experiment_info(adata, tkey, experiment_type)

        main_info_insert_adata("tkey=%s" % tkey, "uns['pp']", indent_level=2)
        main_info_insert_adata("experiment_type=%s" % experiment_type, "uns['pp']", indent_level=2)

        main_info("making adata observation index unique...")
        adata = unique_var_obs_adata(adata)
        if self.collapse_species_adata:
            main_info("applying collapse species adata...")
            self.collapse_species_adata(adata)

        if self.convert_gene_name:
            main_info("applying convert_gene_name function...")
            self.convert_gene_name(adata)

        if self.filter_cells_by_outliers:
            main_info("filtering outlier cells...")
            self.filter_cells_by_outliers(adata, **self.filter_cells_by_outliers_kwargs)

        if self.filter_genes_by_outliers:
            main_info("filtering outlier genes...")
            main_info("extra kwargs:" + str(self.filter_genes_by_outliers_kwargs))
            self.filter_genes_by_outliers(adata, **self.filter_genes_by_outliers_kwargs)

        if self.select_genes:
            main_info("selecting genes...")
            self.select_genes(adata, **self.select_genes_kwargs)

        # gene selection has been completed above. Now we need to append/delete/force gene list required by users.
        if self.gene_append_list is not None:
            append_genes = adata.var.index.intersection(self.gene_append_list)
            adata.var.loc[append_genes, DKM.VAR_USE_FOR_PCA] = True
            main_info("appended %d extra genes as required..." % len(append_genes))

        if self.gene_exclude_list is not None:
            exclude_genes = adata.var.index.intersection(self.gene_exclude_list)
            adata.var.loc[exclude_genes, DKM.VAR_USE_FOR_PCA] = False
            main_info("excluded %d genes as required..." % len(exclude_genes))

        if self.force_gene_list is not None:
            adata.var.loc[:, DKM.VAR_USE_FOR_PCA] = False
            forced_genes = adata.var.index.intersection(self.force_gene_list)
            adata.var.loc[forced_genes, DKM.VAR_USE_FOR_PCA] = True
            main_info(
                "OVERWRITE all gene selection results above according to user gene list inputs. %d genes in use."
                % len(forced_genes)
            )

        if self.normalize_selected_genes:
            main_info("normalizing selected genes...")
            self.normalize_selected_genes(adata)

        if self.normalize_by_cells:
            main_info("applying normalize by cells function...")
            self.normalize_by_cells(adata, **self.normalize_by_cells_function_kwargs)

        if self.use_log1p:
            if is_log1p_transformed_adata(adata):
                main_warning(
                    "Your adata.X maybe log1p transformed before. If you are sure that your adata is not log1p transformed, please ignore this warning. Dynamo will do log1p transformation still."
                )
            main_info("applying log1p transformation on expression matrix data (adata.X)...")
            log1p_adata(adata)

        import pandas as pd

        if self.pca:
            main_info("reducing dimension by PCA...")
            self.pca(adata, **self.pca_kwargs)
