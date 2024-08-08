"""Mapping Vector Field of Single Cells
"""

from .cell_cycle import cell_cycle_scores
from .deprecated import (
    calc_sz_factor_legacy,
    cook_dist,
    filter_cells_legacy,
    normalize_cell_expr_by_size_factors,
    recipe_monocle,
    recipe_velocyto,
)
from .dynast import lambda_correction
from .external import (
    harmony_debatch,
    integrate,
    normalize_layers_pearson_residuals,
    sctransform,
    select_genes_by_pearson_residuals,
)
from .normalization import calc_sz_factor, normalize
from .pca import pca, top_pca_genes
from .QC import (
    basic_stats,
    filter_cells_by_highly_variable_genes,
    filter_cells_by_outliers,
    filter_genes_by_clusters,
    filter_genes_by_outliers,
    filter_genes_by_pattern,
)
from .transform import log1p, log1p_adata_layer
from .utils import (
    compute_gene_exp_fraction,
    convert2symbol,
    decode,
    get_svr_filter,
    relative2abs,
    scale,
)

filter_cells = filter_cells_by_outliers
filter_genes = filter_genes_by_outliers
log1p = log1p
normalize_cells = normalize

from .CnmfPreprocessor import CnmfPreprocessor
from .gene_selection import (
    calc_dispersion_by_svr,
    calc_Gini,
    highest_frac_genes,
    select_genes_monocle,
)
from .Preprocessor import Preprocessor

__all__ = [
    "calc_sz_factor",
    "filter_cells",
    "filter_genes",
    "normalize_cells",
    "lambda_correction",
    "calc_sz_factor_legacy",
    "normalize_layers_pearson_residuals",
    "normalize",
    "recipe_monocle",
    "recipe_velocyto",
    "calc_Gini",
    "filter_cells_by_outliers",
    "filter_cells_by_highly_variable_genes",
    "select_genes_monocle",
    "select_genes_by_pearson_residuals",
    "filter_genes",
    "filter_genes_by_outliers",
    "filter_genes_by_clusters",
    "calc_dispersion_by_svr",
    "get_svr_filter",
    "highest_frac_genes",
    "cell_cycle_scores",
    "basic_stats",
    "cook_dist",
    "pca",
    "top_pca_genes",
    "relative2abs",
    "scale",
    "sctransform",
    "convert2symbol",
    "filter_genes_by_pattern",
    "decode",
    "Preprocessor",
    "CnmfPreprocessor",
    "log1p",
    "log1p_adata_layer",
    "harmony_debatch",
    "integrate",
    "normalize_cell_expr_by_size_factors",
    "filter_cells_legacy",
]
