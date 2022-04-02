from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from anndata import AnnData

from ..configuration import DKM
from ..dynamo_logger import (
    LoggerManager,
    main_info,
    main_info_insert_adata,
    main_warning,
)
from ..external import (
    normalize_layers_pearson_residuals,
    sctransform,
    select_genes_by_pearson_residuals,
)
from ..tools.connectivity import neighbors as default_neighbors
from .preprocess import normalize_cell_expr_by_size_factors_legacy, pca_monocle
from .preprocessor_utils import _infer_labeling_experiment_type
from .preprocessor_utils import (
    filter_cells_by_outliers as monocle_filter_cells_by_outliers,
)
from .preprocessor_utils import (
    filter_genes_by_outliers as monocle_filter_genes_by_outliers,
)
from .preprocessor_utils import (
    is_log1p_transformed_adata,
    log1p_adata,
    normalize_cell_expr_by_size_factors,
    select_genes_by_dispersion_general,
)
from .utils import (
    collapse_species_adata,
    convert2symbol,
    convert_layers2csr,
    detect_experiment_datatype,
    unique_var_obs_adata,
)


class Preprocessor:
    def __init__(
        self,
        collapse_speicies_adata_function: Callable = collapse_species_adata,
        convert_gene_name_function: Callable = convert2symbol,
        filter_cells_by_outliers_function: Callable = monocle_filter_cells_by_outliers,
        filter_cells_by_outliers_kwargs: Callable = {},
        filter_genes_by_outliers_function: Callable = monocle_filter_genes_by_outliers,
        filter_genes_by_outliers_kwargs: dict = {},
        normalize_by_cells_function: Callable = normalize_cell_expr_by_size_factors,
        normalize_by_cells_function_kwargs: Callable = {},
        select_genes_function: Callable = select_genes_by_dispersion_general,
        select_genes_kwargs: dict = {},
        normalize_selected_genes_function: Callable = None,
        normalize_selected_genes_kwargs: dict = {},
        use_log1p: bool = True,
        log1p_kwargs: dict = {},
        pca_function: bool = pca_monocle,
        pca_kwargs: dict = {},
        gene_append_list: List = [],
        gene_exclude_list: List = [],
        force_gene_list: Optional[List] = None,
        sctransform_kwargs={},
    ) -> None:
        """Preprocessor constructor
        The default preprocess functions are those of monocle recipe by default.
        You can pass your own Callable objects (functions) to this constructor directly, which wil be used in the preprocess steps later.
        These functions parameters are saved into Preprocessor instances. You can set these attributes directly to your own implementation.

        Parameters
        ----------
        collapse_speicies_adata_function :
            function for collapsing the species data, by default collapse_species_adata
        convert_gene_name_function :
            transform gene names, by default convert2symbol, which transforms unofficial gene names to official gene names
        filter_cells_by_outliers_function :
            filter cells by thresholds, by default monocle_filter_cells_by_outliers
        filter_cells_by_outliers_kwargs :
            arguments that will be passed to filter_cells_by_outliers, by default {}
        filter_genes_by_outliers_function :
            filter genes by thresholds, by default monocle_filter_genes_by_outliers
        filter_genes_by_outliers_kwargs : dict, optional
            arguments that will be passed to filter_genes_by_outliers, by default {}
        normalize_by_cells_function :
            function for performing cell-wise normalization, by default normalize_cell_expr_by_size_factors
        normalize_by_cells_function_kwargs :
            arguments that will be passed to normalize_by_cells_function, by default {}
        select_genes_function :
            function for selecting gene features, by default select_genes_by_dispersion_general
        select_genes_kwargs : dict, optional
            arguments that will be passed to select_genes, by default {}
        normalize_selected_genes_function :
            function for normalize selected genes, by default None
        normalize_selected_genes_kwargs :
            arguments that will be passed to  normalize_selected_genes, by default {}
        use_log1p : bool, optional
            whether to use log1p to normalize layers in adata, by default True
        log1p_kwargs :
            arguments passed to use_log1p, e.g. `layers` that will be normalized, by default {}
        pca_function :
            function to perform pca, by default pca_monocle
        pca_kwargs :
            arguments that will be passed pca, by default {}
        gene_append_list :
            ensure that a list of genes show up in selected genes in monocle recipe pipeline, by default []
        gene_exclude_list :
            exclude a list of genes in monocle recipe pipeline, by default []
        force_gene_list :
            use this gene list as selected genes in monocle recipe pipeline, by default None
        sctransform_kwargs :
            arguments passed into sctransform function, by default {}
        """
        self.convert_layers2csr = convert_layers2csr
        self.unique_var_obs_adata = unique_var_obs_adata
        self.log1p = log1p_adata
        self.log1p_kwargs = log1p_kwargs
        self.sctransform = sctransform

        self.filter_cells_by_outliers = filter_cells_by_outliers_function
        self.filter_genes_by_outliers = filter_genes_by_outliers_function
        self.normalize_by_cells = normalize_by_cells_function
        self.select_genes = select_genes_function
        self.normalize_selected_genes = normalize_selected_genes_function
        self.use_log1p = use_log1p

        self.pca = pca_function
        self.pca_kwargs = pca_kwargs

        # self.n_top_genes = n_top_genes
        self.convert_gene_name = convert_gene_name_function
        self.collapse_species_adata = collapse_speicies_adata_function
        self.gene_append_list = gene_append_list
        self.gene_exclude_list = gene_exclude_list
        self.force_gene_list = force_gene_list

        # kwargs pass to the functions above
        self.filter_genes_by_outliers_kwargs = filter_genes_by_outliers_kwargs
        self.normalize_by_cells_function_kwargs = normalize_by_cells_function_kwargs
        self.filter_cells_by_outliers_kwargs = filter_cells_by_outliers_kwargs
        self.select_genes_kwargs = select_genes_kwargs
        self.sctransform_kwargs = sctransform_kwargs
        self.normalize_selected_genes_kwargs = normalize_selected_genes_kwargs

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

    def standardize_adata(self, adata: AnnData, tkey: str, experiment_type: str):
        adata.uns["pp"] = {}
        adata.uns["pp"]["norm_method"] = None
        self.add_experiment_info(adata, tkey, experiment_type)
        main_info_insert_adata("tkey=%s" % tkey, "uns['pp']", indent_level=2)
        main_info_insert_adata("experiment_type=%s" % experiment_type, "uns['pp']", indent_level=2)
        main_info("making adata observation index unique...")
        self.unique_var_obs_adata(adata)
        self.convert_layers2csr(adata)

        if self.collapse_species_adata:
            main_info("applying collapse species adata...")
            self.collapse_species_adata(adata)

        if self.convert_gene_name:
            main_info("applying convert_gene_name function...")
            self.convert_gene_name(adata)
            main_info("making adata observation index unique after gene name conversion...")
            self.unique_var_obs_adata(adata)

    def _filter_cells_by_outliers(self, adata: AnnData):
        if self.filter_cells_by_outliers:
            main_info("filtering outlier cells...")
            main_info("cell filter kwargs:" + str(self.filter_cells_by_outliers_kwargs))
            self.filter_cells_by_outliers(adata, **self.filter_cells_by_outliers_kwargs)

    def _filter_genes_by_outliers(self, adata: AnnData):
        if self.filter_genes_by_outliers:
            main_info("filtering outlier genes...")
            main_info("gene filter kwargs:" + str(self.filter_genes_by_outliers_kwargs))
            self.filter_genes_by_outliers(adata, **self.filter_genes_by_outliers_kwargs)

    def _select_genes(self, adata: AnnData):
        if self.select_genes:
            main_info("selecting genes...")
            main_info("select_genes kwargs:" + str(self.select_genes_kwargs))
            self.select_genes(adata, **self.select_genes_kwargs)

    def _append_gene_list(self, adata: AnnData):
        if self.gene_append_list is not None:
            append_genes = adata.var.index.intersection(self.gene_append_list)
            adata.var.loc[append_genes, DKM.VAR_USE_FOR_PCA] = True
            main_info("appended %d extra genes as required..." % len(append_genes))

    def _exclude_gene_list(self, adata: AnnData):
        if self.gene_exclude_list is not None:
            exclude_genes = adata.var.index.intersection(self.gene_exclude_list)
            adata.var.loc[exclude_genes, DKM.VAR_USE_FOR_PCA] = False
            main_info("excluded %d genes as required..." % len(exclude_genes))

    def _force_gene_list(self, adata: AnnData):
        if self.force_gene_list is not None:
            adata.var.loc[:, DKM.VAR_USE_FOR_PCA] = False
            forced_genes = adata.var.index.intersection(self.force_gene_list)
            adata.var.loc[forced_genes, DKM.VAR_USE_FOR_PCA] = True
            main_info(
                "OVERWRITE all gene selection results above according to user gene list inputs. %d genes in use."
                % len(forced_genes)
            )

    def _normalize_selected_genes(self, adata: AnnData):
        if not callable(self.normalize_selected_genes):
            main_info(
                "skipping normalize by selected genes as preprocessor normalize_selected_genes is not callable..."
            )
            return

        main_info("normalizing selected genes...")
        self.normalize_selected_genes(adata, **self.normalize_selected_genes_kwargs)

    def _normalize_by_cells(self, adata: AnnData):
        if not callable(self.normalize_by_cells):
            main_info("skipping normalize by cells as preprocessor normalize_by_cells is not callable...")
            return

        main_info("applying normalize by cells function...")
        self.normalize_by_cells(adata, **self.normalize_by_cells_function_kwargs)

    def _log1p(self, adata: AnnData):
        if self.use_log1p:
            if is_log1p_transformed_adata(adata):
                main_warning(
                    "Your adata.X maybe log1p transformed before. If you are sure that your adata is not log1p transformed, please ignore this warning. Dynamo will do log1p transformation still."
                )
            # TODO: the following line is for monocle recipe and later dynamics matrix recovery
            # refactor with dynamics module
            adata.uns["pp"]["norm_method"] = "log1p"
            main_info("applying log1p transformation on expression matrix data (adata.X)...")
            self.log1p(adata, **self.log1p_kwargs)

    def _pca(self, adata):
        if self.pca:
            main_info("reducing dimension by PCA...")
            self.pca(adata, **self.pca_kwargs)

    def config_monocle_recipe(self, adata: AnnData, n_top_genes: int = 2000, gene_selection_method: str = "SVR"):
        n_obs, n_genes = adata.n_obs, adata.n_vars
        n_cells = n_obs
        self.use_log1p = False
        self.filter_cells_by_outliers = monocle_filter_cells_by_outliers
        self.filter_cells_by_outliers_kwargs = {
            "filter_bool": None,
            "layer": "all",
            "min_expr_genes_s": min(50, 0.01 * n_genes),
            "min_expr_genes_u": min(25, 0.01 * n_genes),
            "min_expr_genes_p": min(2, 0.01 * n_genes),
            "max_expr_genes_s": np.inf,
            "max_expr_genes_u": np.inf,
            "max_expr_genes_p": np.inf,
            "shared_count": None,
        }
        self.filter_genes_by_outliers = monocle_filter_genes_by_outliers
        self.filter_genes_by_outliers_kwargs = {
            "filter_bool": None,
            "layer": "all",
            "min_cell_s": max(5, 0.01 * n_cells),
            "min_cell_u": max(5, 0.005 * n_cells),
            "min_cell_p": max(5, 0.005 * n_cells),
            "min_avg_exp_s": 0,
            "min_avg_exp_u": 0,
            "min_avg_exp_p": 0,
            "max_avg_exp": np.inf,
            "min_count_s": 0,
            "min_count_u": 0,
            "min_count_p": 0,
            "shared_count": 30,
        }
        self.select_genes = select_genes_by_dispersion_general
        self.select_genes_kwargs = {
            "recipe": "monocle",
            "monocle_kwargs": {
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
            },
        }
        self.normalize_selected_genes = None
        self.normalize_by_cells = normalize_cell_expr_by_size_factors

        # recipe monocle log1p all raw data in normalize_by_cells (dynamo version), so we do not need extra log1p transform.
        self.use_log1p = False
        self.pca = pca_monocle
        self.pca_kwargs = {"pca_key": "X_pca"}

    def preprocess_adata_monocle(self, adata: AnnData, tkey: Optional[str] = None, experiment_type: str = None):
        main_info("Running preprocessing pipeline...")
        temp_logger = LoggerManager.gen_logger("preprocessor-monocle")
        temp_logger.log_time()

        self.standardize_adata(adata, tkey, experiment_type)

        self._filter_cells_by_outliers(adata)
        self._filter_genes_by_outliers(adata)
        self._filter_cells_by_outliers(adata)
        self._select_genes(adata)

        # gene selection has been completed above. Now we need to append/delete/force selected gene list required by users.
        self._append_gene_list(adata)
        self._exclude_gene_list(adata)
        self._force_gene_list(adata)

        self._normalize_selected_genes(adata)
        self._normalize_by_cells(adata)

        self._log1p(adata)
        self._pca(adata)

        temp_logger.finish_progress(progress_name="preprocess")

    def config_seurat_recipe(self, adata: AnnData):
        self.config_monocle_recipe(adata)
        self.select_genes = select_genes_by_dispersion_general
        self.select_genes_kwargs = {"recipe": "seurat", "n_top_genes": 2000}
        self.normalize_by_cells_function_kwargs = {"skip_log": True}
        self.pca_kwargs = {"pca_key": "X_pca"}
        self.filter_genes_by_outliers_kwargs = {"shared_count": 20}
        self.use_log1p = True
        self.log1p_kwargs = {"layers": ["X"]}

    def preprocess_adata_seurat(self, adata: AnnData, tkey: Optional[str] = None, experiment_type: str = None):
        """
        The preprocess pipeline in Seurat based on dispersion, implemented by dynamo authors.
        Stuart and Butler et al. Comprehensive Integration of Single-Cell Data. Cell (2019)
        Butler et al. Integrating single-cell transcriptomic data across different conditions, technologies, and species. Nat Biotechnol

        Parameters
        ----------
        adata : AnnData
        tkey : Optional[str], optional
            time key, by default None
        experiment_type : str, optional
            experiment type of data, by default None
        """
        temp_logger = LoggerManager.gen_logger("preprocessor-seurat")
        temp_logger.log_time()
        main_info("Applying Seurat recipe preprocessing...")

        self.standardize_adata(adata, tkey, experiment_type)
        self._filter_genes_by_outliers(adata)
        self._normalize_by_cells(adata)
        self._select_genes(adata)
        self._log1p(adata)
        self._pca(adata)
        temp_logger.finish_progress(progress_name="preprocess by seurat recipe")

    def config_sctransform_recipe(self, adata: AnnData):
        self.use_log1p = False
        raw_layers = DKM.get_raw_data_layers(adata)
        self.filter_cells_by_outliers_kwargs = {"keep_filtered": False}
        self.filter_genes_by_outliers_kwargs = {
            "inplace": True,
            "min_cell_s": 5,
            "min_count_s": 1,
            "min_cell_u": 5,
            "min_count_u": 1,
        }
        self.select_genes_kwargs = {"inplace": True}
        self.sctransform_kwargs = {"layers": raw_layers, "n_top_genes": 2000}
        self.pca_kwargs = {"pca_key": "X_pca", "n_pca_components": 50}

    def preprocess_adata_sctransform(self, adata: AnnData, tkey: Optional[str] = None, experiment_type: str = None):
        """
        Python implementation of https://github.com/satijalab/sctransform.
        Hao and Hao et al. Integrated analysis of multimodal single-cell data. Cell (2021)

        Parameters
        ----------
        adata : AnnData
        tkey : Optional[str], optional
            time key, by default None
        experiment_type : str, optional
            experiment type of data, by default None
        """
        temp_logger = LoggerManager.gen_logger("preprocessor-sctransform")
        temp_logger.log_time()
        main_info("Applying Sctransform recipe preprocessing...")

        self.standardize_adata(adata, tkey, experiment_type)

        self._filter_cells_by_outliers(adata)
        self._filter_genes_by_outliers(adata)
        self._select_genes(adata)
        # TODO: if inplace in select_genes is True, the following subset is unnecessary.
        adata._inplace_subset_var(adata.var["use_for_pca"])
        self.sctransform(adata, **self.sctransform_kwargs)
        self._pca(adata)

        temp_logger.finish_progress(progress_name="preprocess by sctransform recipe")

    def config_pearson_residuals_recipe(self, adata: AnnData):
        self.filter_cells_by_outliers = None
        self.filter_genes_by_outliers = None
        self.normalize_by_cells = None
        self.select_genes = select_genes_by_pearson_residuals
        self.select_genes_kwargs = {"n_top_genes": 2000}
        self.normalize_selected_genes = normalize_layers_pearson_residuals
        # select layers in adata to be normalized
        normalize_layers = DKM.get_raw_data_layers(adata)
        self.normalize_selected_genes_kwargs = {"layers": normalize_layers, "copy": False}
        self.pca_kwargs = {"pca_key": "X_pca", "n_pca_components": 50}
        self.use_log1p = False

    def preprocess_adata_pearson_residuals(
        self, adata: AnnData, tkey: Optional[str] = None, experiment_type: Optional[str] = None
    ):
        """A pipeline proposed in Pearson residuals (Lause, Berens & Kobak, 2021).
        Lause, J., Berens, P. & Kobak, D. Analytic Pearson residuals for normalization of single-cell RNA-seq UMI data. Genome Biol 22, 258 (2021). https://doi.org/10.1186/s13059-021-02451-7

        Parameters
        ----------
        adata : AnnData
        tkey : Optional[str], optional
            time key, by default None
        experiment_type : str, optional
            experiment type of data, by default None
        """
        temp_logger = LoggerManager.gen_logger("preprocessor-sctransform")
        temp_logger.log_time()
        self.standardize_adata(adata, tkey, experiment_type)

        self._select_genes(adata)
        self._normalize_selected_genes(adata)
        self._pca(adata)

        temp_logger.finish_progress(progress_name="preprocess by pearson residual recipe")

    def config_monocle_pearson_residuals_recipe(self, adata: AnnData):
        self.config_monocle_recipe(adata)
        # self.filter_cells_by_outliers = None
        # self.filter_genes_by_outliers = None
        self.normalize_by_cells = normalize_cell_expr_by_size_factors
        self.select_genes = select_genes_by_pearson_residuals
        self.select_genes_kwargs = {"n_top_genes": 2000}
        self.normalize_selected_genes = normalize_layers_pearson_residuals

        self.normalize_selected_genes_kwargs = {"layers": ["X"], "copy": False}

        self.pca_kwargs = {"pca_key": "X_pca", "n_pca_components": 50}
        self.use_log1p = False

    def preprocess_adata_monocle_pearson_residuals(
        self, adata: AnnData, tkey: Optional[str] = None, experiment_type: Optional[str] = None
    ):
        """
        A combined pipeline of monocle and pearson_residuals.
        pearson_residuals results can contain negative values, an undesired attributes of later dyanmics analysis. This function uses monocle recipe to generate X_spliced, X_unspliced, X_new, X_total or other data values for dynamics and other downstream methods.

        Parameters
        ----------
        adata : AnnData
        tkey : Optional[str], optional
            time key, by default None
        experiment_type : str, optional
            experiment type of data, by default None

        """
        temp_logger = LoggerManager.gen_logger("preprocessor-monocle-pearson-residual")
        temp_logger.log_time()
        self.standardize_adata(adata, tkey, experiment_type)
        self._select_genes(adata)
        X_copy = adata.X.copy()
        self._normalize_by_cells(adata)
        adata.X = X_copy
        self._normalize_selected_genes(adata)
        # use monocle to pprocess adata
        # self.config_monocle_recipe(adata_copy)
        # self.pca = None # do not do pca in this monocle
        # self.preprocess_adata_monocle(adata_copy)
        # for layer in adata_copy.layers:
        #     if DKM.is_layer_X_key(layer):
        #         adata.layers[layer] = adata.

        self.pca(adata, **self.pca_kwargs)
        temp_logger.finish_progress(progress_name="preprocess by monocle pearson residual recipe")

    def preprocess_adata(self, adata: AnnData, recipe: str = "monocle", tkey: Optional[str] = None):
        """A wrapper and interface entry for all recipes."""
        if recipe == "monocle":
            self.config_monocle_recipe(adata)
            self.preprocess_adata_monocle(adata, tkey=tkey)
        elif recipe == "seurat":
            self.config_seurat_recipe(adata)
            self.preprocess_adata_seurat(adata, tkey=tkey)
        elif recipe == "sctransform":
            self.config_sctransform_recipe(adata)
            self.preprocess_adata_sctransform(adata, tkey=tkey)
        elif recipe == "pearson_residuals":
            self.config_pearson_residuals_recipe(adata)
            self.preprocess_adata_pearson_residuals(adata, tkey=tkey)
        elif recipe == "monocle_pearson_residuals":
            self.config_monocle_pearson_residuals_recipe(adata)
            self.preprocess_adata_monocle_pearson_residuals(adata, tkey=tkey)
        else:
            raise NotImplementedError("preprocess recipe chosen not implemented: %s" % (recipe))
