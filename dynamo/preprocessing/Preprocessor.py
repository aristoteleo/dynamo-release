from typing import Any, Callable, Dict, List, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from anndata import AnnData

from ..configuration import DKM
from ..dynamo_logger import (
    LoggerManager,
    main_debug,
    main_info,
    main_info_insert_adata,
    main_warning,
)
from ..tools.connectivity import neighbors as default_neighbors
from ..tools.utils import update_dict
from .cell_cycle import cell_cycle_scores
from .external import (
    normalize_layers_pearson_residuals,
    sctransform,
    select_genes_by_pearson_residuals,
)
from .gene_selection import select_genes_by_seurat_recipe, select_genes_monocle
from .normalization import calc_sz_factor, normalize
from .pca import pca
from .QC import basic_stats
from .QC import filter_cells_by_outliers as monocle_filter_cells_by_outliers
from .QC import filter_genes_by_outliers as monocle_filter_genes_by_outliers
from .QC import regress_out_parallel
from .transform import Freeman_Tukey, log, log1p, log2
from .utils import (
    _infer_labeling_experiment_type,
    calc_new_to_total_ratio,
    collapse_species_adata,
    convert2symbol,
    convert_layers2csr,
    detect_experiment_datatype,
    unique_var_obs_adata,
)


class Preprocessor:
    def __init__(
        self,
        collapse_species_adata_function: Callable = collapse_species_adata,
        convert_gene_name_function: Callable = convert2symbol,
        filter_cells_by_outliers_function: Callable = monocle_filter_cells_by_outliers,
        filter_cells_by_outliers_kwargs: Dict[str, Any] = {},
        filter_genes_by_outliers_function: Callable = monocle_filter_genes_by_outliers,
        filter_genes_by_outliers_kwargs: Dict[str, Any] = {},
        normalize_by_cells_function: Callable = normalize,
        normalize_by_cells_function_kwargs: Dict[str, Any] = {},
        size_factor_function: Callable = calc_sz_factor,
        size_factor_kwargs: Dict[str, Any] = {},
        select_genes_function: Callable = select_genes_monocle,
        select_genes_kwargs: Dict[str, Any] = {},
        normalize_selected_genes_function: Callable = None,
        normalize_selected_genes_kwargs: Dict[str, Any] = {},
        norm_method: Callable = log1p,
        norm_method_kwargs: Dict[str, Any] = {},
        pca_function: Callable = pca,
        pca_kwargs: Dict[str, Any] = {},
        gene_append_list: List[str] = [],
        gene_exclude_list: List[str] = [],
        force_gene_list: Optional[List[str]] = None,
        sctransform_kwargs: Dict[str, Any] = {},
        regress_out_kwargs: Dict[List[str], Any] = {},
        cell_cycle_score_enable: bool = False,
        cell_cycle_score_kwargs: Dict[str, Any] = {},
    ) -> None:
        """Preprocessor constructor.

        The default preprocess functions are those of monocle recipe by default.
        You can pass your own Callable objects (functions) to this constructor directly, which wil be used in
        the preprocess steps later. These functions parameters are saved into Preprocessor instances.
        You can set these attributes directly to your own implementation.

        Args:
            collapse_species_adata_function: function for collapsing the species data. Defaults to
                collapse_species_adata.
            convert_gene_name_function: transform gene names, by default convert2symbol, which transforms unofficial
                gene names to official gene names. Defaults to convert2symbol.
            filter_cells_by_outliers_function: filter cells by thresholds. Defaults to monocle_filter_cells_by_outliers.
            filter_cells_by_outliers_kwargs: arguments that will be passed to filter_cells_by_outliers. Defaults to {}.
            filter_genes_by_outliers_function: filter genes by thresholds. Defaults to monocle_filter_genes_by_outliers.
            filter_genes_by_outliers_kwargs: arguments that will be passed to filter_genes_by_outliers. Defaults to {}.
            normalize_by_cells_function: function for performing cell-wise normalization. Defaults to
                normalize_cell_expr_by_size_factors.
            normalize_by_cells_function_kwargs: arguments that will be passed to normalize_by_cells_function. Defaults
                to {}.
            select_genes_function: function for selecting gene features. Defaults to select_genes_monocle.
            select_genes_kwargs: arguments that will be passed to select_genes. Defaults to {}.
            normalize_selected_genes_function: function for normalize selected genes. Defaults to None.
            normalize_selected_genes_kwargs: arguments that will be passed to normalize_selected_genes. Defaults to {}.
            norm_method: whether to use a method to normalize layers in adata. Defaults to True.
            norm_method_kwargs: arguments passed to norm_method. Defaults to {}.
            pca_function: function to perform pca. Defaults to pca in utils.py.
            pca_kwargs: arguments that will be passed pca. Defaults to {}.
            gene_append_list: ensure that a list of genes show up in selected genes across all the recipe pipeline.
                Defaults to [].
            gene_exclude_list: exclude a list of genes across all the recipe pipeline. Defaults to [].
            force_gene_list: use this gene list as selected genes across all the recipe pipeline. Defaults to None.
            sctransform_kwargs: arguments passed into sctransform function. Defaults to {}.
            regress_out_kwargs: arguments passed into regress_out function. Defaults to {}.
        """

        self.basic_stats = basic_stats
        self.convert_layers2csr = convert_layers2csr
        self.unique_var_obs_adata = unique_var_obs_adata
        self.norm_method = norm_method
        self.norm_method_kwargs = norm_method_kwargs
        self.sctransform = sctransform

        self.filter_cells_by_outliers = filter_cells_by_outliers_function
        self.filter_genes_by_outliers = filter_genes_by_outliers_function
        self.normalize_by_cells = normalize_by_cells_function
        self.calc_size_factor = size_factor_function
        self.calc_new_to_total_ratio = calc_new_to_total_ratio
        self.select_genes = select_genes_function
        self.normalize_selected_genes = normalize_selected_genes_function
        self.regress_out = regress_out_parallel
        self.pca = pca_function
        self.pca_kwargs = pca_kwargs

        # self.n_top_genes = n_top_genes
        self.convert_gene_name = convert_gene_name_function
        self.collapse_species_adata = collapse_species_adata_function
        self.gene_append_list = gene_append_list
        self.gene_exclude_list = gene_exclude_list
        self.force_gene_list = force_gene_list

        # kwargs pass to the functions above
        self.filter_genes_by_outliers_kwargs = filter_genes_by_outliers_kwargs
        self.normalize_by_cells_function_kwargs = normalize_by_cells_function_kwargs
        self.filter_cells_by_outliers_kwargs = filter_cells_by_outliers_kwargs
        self.size_factor_kwargs = size_factor_kwargs
        self.select_genes_kwargs = select_genes_kwargs
        self.sctransform_kwargs = sctransform_kwargs
        self.normalize_selected_genes_kwargs = normalize_selected_genes_kwargs
        self.cell_cycle_score_enable = cell_cycle_score_enable
        self.cell_cycle_score = cell_cycle_scores
        self.cell_cycle_score_kwargs = cell_cycle_score_kwargs
        self.regress_out_kwargs = regress_out_kwargs

    def add_experiment_info(
        self, adata: AnnData, tkey: Optional[str] = None, experiment_type: Optional[str] = None
    ) -> None:
        """Infer the experiment type and experiment layers stored in the AnnData
        object and record the info in unstructured metadata (.uns).

        Args:
            adata: an AnnData object.
            tkey: the key for time information (labeling time period for the cells) in .obs. Defaults to None.
            experiment_type: the experiment type. If set to None, the experiment type would be inferred from the data.

        Raises:
            ValueError: the tkey is invalid.
        """

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
            main_debug("data contains labeling info, checking tkey:" + str(tkey))
            if tkey is not None and adata.obs[tkey].max() > 60:
                main_warning(
                    "Looks like you are using minutes as the time unit. For the purpose of numeric stability, "
                    "we recommend using hour as the time unit."
                )
            if tkey not in adata.obs.keys():
                if (tkey is None) and (DKM.UNS_PP_TKEY in adata.obs.keys()):
                    tkey = DKM.UNS_PP_TKEY
                    main_warning(
                        "No 'tkey' value was given despite 'tkey' information in the adata, "
                        "so we will use 'time' in the adata as the default."
                    )
                else:
                    raise ValueError("tkey:%s encoding the labeling time is not existed in your adata." % (str(tkey)))

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

    def standardize_adata(self, adata: AnnData, tkey: str, experiment_type: str) -> None:
        """Process the AnnData object to make it meet the standards of dynamo.

        The index of the observations would be ensured to be unique. The layers with sparse matrix would be converted to
        compressed csr_matrix. DKM.allowed_layer_raw_names() will be used to define only_splicing, only_labeling and
        splicing_labeling keys. The genes would be renamed to their official name.

        Args:
            adata: an AnnData object.
            tkey: the key for time information (labeling time period for the cells) in .obs.
            experiment_type: the experiment type.
        """

        adata.uns["pp"] = {}
        adata.uns["pp"]["X_norm_method"] = None
        adata.uns["pp"]["layers_norm_method"] = None

        main_debug("applying convert_gene_name function...")
        self.convert_gene_name(adata)


        self.basic_stats(adata)
        self.add_experiment_info(adata, tkey, experiment_type)
        main_info_insert_adata("tkey=%s" % tkey, "uns['pp']", indent_level=2)
        main_info_insert_adata("experiment_type=%s" % adata.uns["pp"]["experiment_type"], "uns['pp']", indent_level=2)

        self.convert_layers2csr(adata)
        self.collapse_species_adata(adata)

        main_debug("making adata observation index unique after gene name conversion...")
        self.unique_var_obs_adata(adata)

    def _filter_cells_by_outliers(self, adata: AnnData) -> None:
        """Select valid cells based on the method specified as the preprocessor's `filter_cells_by_outliers`.

        Args:
            adata: an AnnData object.
        """

        if self.filter_cells_by_outliers:
            main_debug("filtering outlier cells...")
            main_debug("cell filter kwargs:" + str(self.filter_cells_by_outliers_kwargs))
            self.filter_cells_by_outliers(adata, **self.filter_cells_by_outliers_kwargs)

    def _filter_genes_by_outliers(self, adata: AnnData) -> None:
        """Select valid genes based on the method specified as the preprocessor's `filter_genes_by_outliers`.

        Args:
            adata: an AnnData object.
        """

        if self.filter_genes_by_outliers:
            main_debug("filtering outlier genes...")
            main_debug("gene filter kwargs:" + str(self.filter_genes_by_outliers_kwargs))
            self.filter_genes_by_outliers(adata, **self.filter_genes_by_outliers_kwargs)

    def _calc_size_factor(self, adata: AnnData) -> None:
        """Calculate the size factor of each cell based on method specified as the preprocessor's `calc_size_factor`.

        Args:
            adata: an AnnData object.
        """

        if self.calc_size_factor:
            main_debug("size factor calculation...")
            main_debug("size_factor_kwargs kwargs:" + str(self.size_factor_kwargs))
            self.calc_size_factor(
                adata,
                total_layers=adata.uns["pp"]["experiment_total_layers"],
                layers=adata.uns["pp"]["experiment_layers"],
                **self.size_factor_kwargs
            )

    def _select_genes(self, adata: AnnData) -> None:
        """selecting gene by features, based on method specified as the preprocessor's `select_genes`.

        Args:
            adata: an AnnData object.
        """

        if self.select_genes:
            main_debug("selecting genes...")
            main_debug("select_genes kwargs:" + str(self.select_genes_kwargs))
            self.select_genes(adata, **self.select_genes_kwargs)

    def _append_gene_list(self, adata: AnnData) -> None:
        """Add genes to the feature gene list detected by the preprocessing.

        Args:
            adata: an AnnData object.
        """

        if len(self.gene_append_list) > 0:
            append_genes = adata.var.index.intersection(self.gene_append_list)
            adata.var.loc[append_genes, DKM.VAR_USE_FOR_PCA] = True
            main_info("appended %d extra genes as required..." % len(append_genes))

    def _exclude_gene_list(self, adata: AnnData) -> None:
        """Remove genes from the feature gene list detected by the preprocessing.

        Args:
            adata: an AnnData object.
        """

        if len(self.gene_exclude_list) > 0:
            exclude_genes = adata.var.index.intersection(self.gene_exclude_list)
            adata.var.loc[exclude_genes, DKM.VAR_USE_FOR_PCA] = False
            main_info("excluded %d genes as required..." % len(exclude_genes))

    def _force_gene_list(self, adata: AnnData) -> None:
        """Use the provided gene list as the feature gene list, overwrite the gene list detected by the preprocessing.

        Args:
            adata: an AnnData object.
        """

        if self.force_gene_list is not None:
            adata.var.loc[:, DKM.VAR_USE_FOR_PCA] = False
            forced_genes = adata.var.index.intersection(self.force_gene_list)
            adata.var.loc[forced_genes, DKM.VAR_USE_FOR_PCA] = True
            main_info(
                "OVERWRITE all gene selection results above according to user gene list inputs. %d genes in use."
                % len(forced_genes)
            )

    def _normalize_selected_genes(self, adata: AnnData) -> None:
        """Normalize selected genes with method specified in the preprocessor's `normalize_selected_genes`

        Args:
            adata: an AnnData object.
        """

        if callable(self.normalize_selected_genes):
            main_debug("normalizing selected genes...")
            self.normalize_selected_genes(adata, **self.normalize_selected_genes_kwargs)

    def _normalize_by_cells(self, adata: AnnData) -> None:
        """Performing cell-wise normalization based on method specified as the preprocessor's `normalize_by_cells`.

        Args:
            adata: an AnnData object.
        """

        if callable(self.normalize_by_cells):
            main_debug("applying normalize by cells function...")
            self.normalize_by_cells(adata, **self.normalize_by_cells_function_kwargs)

    def _norm_method(self, adata: AnnData) -> None:
        """Perform a normalization method on the data with args specified in the preprocessor's `norm_method_kwargs`.

        Args:
            adata: an AnnData object.
        """

        if callable(self.norm_method):
            main_debug("applying a normalization method transformation on expression matrix data...")
            self.norm_method(adata, **self.norm_method_kwargs)

    def _regress_out(self, adata: AnnData) -> None:
        """Perform regressing out with args specified in the preprocessor's `regress_out_kwargs`.

        Args:
            adata: an AnnData object.
        """

        if self.regress_out:
            main_info("regressing out...")
            self.regress_out(adata, **self.regress_out_kwargs)

    def _pca(self, adata: AnnData) -> None:
        """Perform principal component analysis reduction with args specified in the preprocessor's `pca_kwargs`.

        Args:
            adata: an AnnData object.
        """

        if self.pca:
            main_info("PCA dimension reduction")
            self.pca(adata, **self.pca_kwargs)

    def _calc_ntr(self, adata: AnnData) -> None:
        """Calculate the size factor of each cell based on method specified as the preprocessor's `calc_size_factor`.

        Args:
            adata: an AnnData object.
        """

        if self.calc_new_to_total_ratio:
            main_debug("ntr calculation...")
            # calculate NTR for every cell:
            ntr, var_ntr = self.calc_new_to_total_ratio(adata)
            adata.obs["ntr"] = ntr
            adata.var["ntr"] = var_ntr

    def _cell_cycle_score(self, adata: AnnData) -> None:
        """Estimate cell cycle stage of each cell based on its gene expression pattern.

        Args:
            adata: an AnnData object.
        """

        if self.cell_cycle_score_enable:
            main_debug("cell cycle scoring...")
            try:
                self.cell_cycle_score(adata, **self.cell_cycle_score_kwargs)
            except Exception:
                main_warning(
                    "\nDynamo is not able to perform cell cycle staging for you automatically. \n"
                    "Since dyn.pl.phase_diagram in dynamo by default colors cells by its cell-cycle stage, \n"
                    "you need to set color argument accordingly if confronting errors related to this."
                )

    def preprocess_adata_seurat_wo_pca(
        self, adata: AnnData, tkey: Optional[str] = None, experiment_type: Optional[str] = None
    ) -> None:
        """Preprocess the anndata object according to standard preprocessing in Seurat recipe without PCA.
        This can be used to test different dimension reduction methods.
        """
        main_info("Running preprocessing pipeline...")
        temp_logger = LoggerManager.gen_logger("preprocessor-seurat_wo_pca")
        temp_logger.log_time()

        self.standardize_adata(adata, tkey, experiment_type)
        self._filter_cells_by_outliers(adata)
        self._filter_genes_by_outliers(adata)
        self._calc_size_factor(adata)
        self._normalize_by_cells(adata)
        self._select_genes(adata)
        self._norm_method(adata)

        temp_logger.finish_progress(progress_name="preprocess by seurat wo pca recipe")

    def config_monocle_recipe(self, adata: AnnData, n_top_genes: int = 2000) -> None:
        """Automatically configure the preprocessor for monocle recipe.

        Args:
            adata: an AnnData object.
            n_top_genes: Number of top feature genes to select in the preprocessing step. Defaults to 2000.
        """

        n_obs, n_genes = adata.n_obs, adata.n_vars
        n_cells = n_obs

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
        self.select_genes = select_genes_monocle
        self.select_genes_kwargs = {"n_top_genes": n_top_genes, "SVRs_kwargs": {"relative_expr": False}}
        self.normalize_selected_genes = None
        self.normalize_by_cells = normalize
        self.norm_method = log1p

        self.regress_out_kwargs = update_dict({"obs_keys": []}, self.regress_out_kwargs)

        self.pca = pca
        self.pca_kwargs = {"pca_key": "X_pca"}

        self.cell_cycle_score_kwargs = {
            "layer": None,
            "gene_list": None,
            "refine": True,
            "threshold": 0.3,
            "copy": False,
        }

    def preprocess_adata_monocle(
        self, adata: AnnData, tkey: Optional[str] = None, experiment_type: Optional[str] = None
    ) -> None:
        """Preprocess the AnnData object based on Monocle style preprocessing recipe.

        Args:
            adata: an AnnData object.
            tkey: the key for time information (labeling time period for the cells) in .obs. Defaults to None.
            experiment_type: the experiment type of the data. If not provided, would be inferred from the data.
        """

        main_info("Running monocle preprocessing pipeline...")
        temp_logger = LoggerManager.gen_logger("preprocessor-monocle")
        temp_logger.log_time()

        self.standardize_adata(adata, tkey, experiment_type)
        self._filter_cells_by_outliers(adata)
        self._filter_genes_by_outliers(adata)

        # The following size factor calculation is a prerequisite for monocle recipe preprocess in preprocessor.
        self._calc_size_factor(adata)
        self._normalize_by_cells(adata)
        self._select_genes(adata)

        # append/delete/force selected gene list required by users.
        self._append_gene_list(adata)
        self._exclude_gene_list(adata)
        self._force_gene_list(adata)

        self._norm_method(adata)

        if len(self.regress_out_kwargs["obs_keys"]) > 0:
            self._regress_out(adata)

        self._pca(adata)
        self._calc_ntr(adata)
        self._cell_cycle_score(adata)

        temp_logger.finish_progress(progress_name="Preprocessor-monocle")

    def config_seurat_recipe(self, adata: AnnData) -> None:
        """Automatically configure the preprocessor for using the seurat style recipe.

        Args:
            adata: an AnnData object.
        """

        self.config_monocle_recipe(adata)
        self.select_genes = select_genes_by_seurat_recipe
        self.select_genes_kwargs = {
            "algorithm": "seurat_dispersion",
            "n_top_genes": 2000,
        }
        self.pca_kwargs = {"pca_key": "X_pca"}
        self.filter_genes_by_outliers_kwargs = {"shared_count": 20}
        self.regress_out_kwargs = update_dict({"obs_keys": []}, self.regress_out_kwargs)

    def preprocess_adata_seurat(
        self, adata: AnnData, tkey: Optional[str] = None, experiment_type: Optional[str] = None
    ) -> None:
        """The preprocess pipeline in Seurat based on dispersion, implemented by dynamo authors.

        Stuart and Butler et al. Comprehensive Integration of Single-Cell Data.
        Cell (2019) Butler et al. Integrating single-cell transcriptomic data
        across different conditions, technologies, and species. Nat Biotechnol

        Args:
            adata: an AnnData object
            tkey: the key for time information (labeling time period for the cells) in .obs. Defaults to None.
            experiment_type: the experiment type of the data. If not provided, would be inferred from the data.
        """

        temp_logger = LoggerManager.gen_logger("preprocessor-seurat")
        temp_logger.log_time()
        main_info("Running Seurat recipe preprocessing...")

        self.standardize_adata(adata, tkey, experiment_type)
        self._filter_cells_by_outliers(adata)
        self._filter_genes_by_outliers(adata)

        self._calc_size_factor(adata)
        self._normalize_by_cells(adata)
        self._select_genes(adata)

        # append/delete/force selected gene list required by users.
        self._append_gene_list(adata)
        self._exclude_gene_list(adata)
        self._force_gene_list(adata)

        self._norm_method(adata)

        if len(self.regress_out_kwargs["obs_keys"]) > 0:
            self._regress_out(adata)

        self._pca(adata)
        temp_logger.finish_progress(progress_name="Preprocessor-seurat")

    def config_sctransform_recipe(self, adata: AnnData) -> None:
        """Automatically configure the preprocessor for using the sctransform style recipe.

        Args:
            adata: an AnnData object.
        """

        raw_layers = DKM.get_raw_data_layers(adata)
        raw_layers = [layer for layer in raw_layers if layer != DKM.X_LAYER]
        self.filter_cells_by_outliers_kwargs = {"keep_filtered": False}
        self.filter_genes_by_outliers_kwargs = {
            "inplace": True,
            "min_cell_s": 5,
            "min_count_s": 1,
            "min_cell_u": 5,
            "min_count_u": 1,
        }
        self.select_genes_kwargs = {"n_top_genes": 3000}
        self.sctransform_kwargs = {"n_top_genes": 2000}
        self.normalize_by_cells_function_kwargs = {"layers": raw_layers}
        self.normalize_by_cells = normalize
        self.regress_out_kwargs = update_dict({"obs_keys": []}, self.regress_out_kwargs)
        self.pca_kwargs = {"pca_key": "X_pca", "n_pca_components": 50}

    def preprocess_adata_sctransform(
        self, adata: AnnData, tkey: Optional[str] = None, experiment_type: Optional[str] = None
    ) -> None:
        """Python implementation of https://github.com/satijalab/sctransform.

        Hao and Hao et al. Integrated analysis of multimodal single-cell data. Cell (2021)

        Args:
            adata: an AnnData object
            tkey: the key for time information (labeling time period for the
                cells) in .obs. Defaults to None.
            experiment_type: the experiment type of the data. If not provided,
                would be inferred from the data. Defaults to None.
        """

        temp_logger = LoggerManager.gen_logger("preprocessor-sctransform")
        temp_logger.log_time()
        main_info("Running Sctransform recipe preprocessing...")

        self.standardize_adata(adata, tkey, experiment_type)
        self._filter_cells_by_outliers(adata)
        self._filter_genes_by_outliers(adata)

        main_warning(
            "Sctransform recipe will subset the data first with default gene selection function for "
            "efficiency. If you want to disable this, please perform sctransform without recipe."
        )
        self._calc_size_factor(adata)
        self._select_genes(adata)
        # TODO: if inplace in select_genes is True, the following subset is unnecessary.
        adata._inplace_subset_var(adata.var["use_for_pca"])

        # append/delete/force selected gene list required by users.
        self._append_gene_list(adata)
        self._exclude_gene_list(adata)
        self._force_gene_list(adata)

        self.sctransform(adata, **self.sctransform_kwargs)
        self._normalize_by_cells(adata)
        if len(self.regress_out_kwargs["obs_keys"]) > 0:
            self._regress_out(adata)
        self._pca(adata)

        temp_logger.finish_progress(progress_name="Preprocessor-sctransform")

    def config_pearson_residuals_recipe(self, adata: AnnData) -> None:
        """Automatically configure the preprocessor for using the Pearson residuals style recipe.

        Args:
            adata: an AnnData object.
        """

        self.filter_cells_by_outliers = None
        self.filter_genes_by_outliers = None
        self.normalize_by_cells = None
        self.select_genes = select_genes_by_pearson_residuals
        self.select_genes_kwargs = {"n_top_genes": 2000}
        self.normalize_selected_genes = normalize_layers_pearson_residuals
        # select layers in adata to be normalized
        normalize_layers = DKM.X_LAYER
        self.normalize_selected_genes_kwargs = {"layers": normalize_layers, "copy": False}
        self.regress_out_kwargs = update_dict({"obs_keys": []}, self.regress_out_kwargs)
        self.pca_kwargs = {"pca_key": "X_pca", "n_pca_components": 50}

    def preprocess_adata_pearson_residuals(
        self, adata: AnnData, tkey: Optional[str] = None, experiment_type: Optional[str] = None
    ) -> None:
        """A pipeline proposed in Pearson residuals (Lause, Berens & Kobak, 2021).

        Lause, J., Berens, P. & Kobak, D. Analytic Pearson residuals for normalization of single-cell RNA-seq UMI data.
        Genome Biol 22, 258 (2021). https://doi.org/10.1186/s13059-021-02451-7

        Args:
            adata: an AnnData object
            tkey: the key for time information (labeling time period for the
                cells) in .obs. Defaults to None.
            experiment_type: the experiment type of the data. If not provided,
                would be inferred from the data. Defaults to None.
        """

        temp_logger = LoggerManager.gen_logger("Preprocessor-pearson residual")
        temp_logger.log_time()
        self.standardize_adata(adata, tkey, experiment_type)
        self._filter_cells_by_outliers(adata)
        self._filter_genes_by_outliers(adata)

        self._select_genes(adata)
        # append/delete/force selected gene list required by users.
        self._append_gene_list(adata)
        self._exclude_gene_list(adata)
        self._force_gene_list(adata)

        self._normalize_selected_genes(adata)
        if len(self.regress_out_kwargs["obs_keys"]) > 0:
            self._regress_out(adata)

        self._pca(adata)

        temp_logger.finish_progress(progress_name="Preprocessor-pearson residual")

    def config_monocle_pearson_residuals_recipe(self, adata: AnnData) -> None:
        """Automatically configure the preprocessor for using the Monocle-Pearson-residuals style recipe.

        Useful when you want to use Pearson residual to obtain feature genes and perform PCA but also using the standard
        size-factor normalization and log1p analyses to normalize data for RNA velocity and vector field analyses.

        Args:
            adata: an AnnData object.
        """

        self.config_monocle_recipe(adata)
        # self.filter_cells_by_outliers = None
        # self.filter_genes_by_outliers = None
        self.normalize_by_cells = normalize
        self.select_genes = select_genes_by_pearson_residuals
        self.select_genes_kwargs = {"n_top_genes": 2000}
        self.normalize_selected_genes = normalize_layers_pearson_residuals
        self.normalize_selected_genes_kwargs = {"layers": ["X"], "copy": False}
        self.regress_out_kwargs = update_dict({"obs_keys": []}, self.regress_out_kwargs)
        self.pca_kwargs = {"pca_key": "X_pca", "n_pca_components": 50}

    def preprocess_adata_monocle_pearson_residuals(
        self, adata: AnnData, tkey: Optional[str] = None, experiment_type: Optional[str] = None
    ) -> None:
        """A combined pipeline of monocle and pearson_residuals.

        Results after running pearson_residuals can contain negative values, an undesired feature for later RNA velocity
        analysis. This function combine pearson_residual and monocle recipes so that it uses Pearson residual to obtain
        feature genes and perform PCA but also uses monocle recipe to generate X_spliced, X_unspliced, X_new, X_total or
        other data values for RNA velocity and downstream vector field analyses.

        Args:
            adata: an AnnData object
            tkey: the key for time information (labeling time period for the cells) in .obs. Defaults to None.
            experiment_type: the experiment type of the data. If not provided, would be inferred from the data.
        """

        temp_logger = LoggerManager.gen_logger("preprocessor-monocle-pearson-residual")
        temp_logger.log_time()
        self.standardize_adata(adata, tkey, experiment_type)
        self._filter_cells_by_outliers(adata)
        self._filter_genes_by_outliers(adata)
        self._select_genes(adata)

        # append/delete/force selected gene list required by users.
        self._append_gene_list(adata)
        self._exclude_gene_list(adata)
        self._force_gene_list(adata)

        X_copy = adata.X.copy()
        self._calc_size_factor(adata)
        self._normalize_by_cells(adata)
        adata.X = X_copy
        self._normalize_selected_genes(adata)
        if len(self.regress_out_kwargs["obs_keys"]) > 0:
            self._regress_out(adata)

        self.pca(adata, **self.pca_kwargs)
        temp_logger.finish_progress(progress_name="Preprocessor-monocle-pearson-residual")

    def preprocess_adata(
        self,
        adata: AnnData,
        recipe: Literal[
            "monocle", "seurat", "sctransform", "pearson_residuals", "monocle_pearson_residuals"
        ] = "monocle",
        tkey: Optional[str] = None,
        experiment_type: Optional[str] = None,
    ) -> None:
        """Preprocess the AnnData object with the recipe specified.

        Args:
            adata: An AnnData object.
            recipe: The recipe used to preprocess the data. Defaults to "monocle".
            tkey: the key for time information (labeling time period for the cells) in .obs. Defaults to None.
            experiment_type: the experiment type of the data. If not provided, would be inferred from the data.

        Raises:
            NotImplementedError: the recipe is invalid.
        """

        if recipe == "monocle":
            self.config_monocle_recipe(adata)
            self.preprocess_adata_monocle(adata, tkey=tkey, experiment_type=experiment_type)
        elif recipe == "seurat":
            self.config_seurat_recipe(adata)
            self.preprocess_adata_seurat(adata, tkey=tkey, experiment_type=experiment_type)
        elif recipe == "sctransform":
            self.config_sctransform_recipe(adata)
            self.preprocess_adata_sctransform(adata, tkey=tkey, experiment_type=experiment_type)
        elif recipe == "pearson_residuals":
            self.config_pearson_residuals_recipe(adata)
            self.preprocess_adata_pearson_residuals(adata, tkey=tkey, experiment_type=experiment_type)
        elif recipe == "monocle_pearson_residuals":
            self.config_monocle_pearson_residuals_recipe(adata)
            self.preprocess_adata_monocle_pearson_residuals(adata, tkey=tkey, experiment_type=experiment_type)
        else:
            raise NotImplementedError("preprocess recipe chosen not implemented: %s" % recipe)
