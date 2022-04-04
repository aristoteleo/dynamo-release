"""Mapping Vector Field of Single Cells
"""

from .cell_cycle import cell_cycle_scores
from .dynast import lambda_correction
from .preprocess import (
    Gini,
    SVRs,
    calc_sz_factor_legacy,
    filter_cells_by_outliers,
    filter_cells_legacy,
    filter_genes_by_clusters_,
    filter_genes_by_outliers,
    get_svr_filter,
    highest_frac_genes,
    normalize_cell_expr_by_size_factors_legacy,
    recipe_monocle,
    recipe_velocyto,
    select_genes_monocle,
)
from .preprocessor_utils import *
from .utils import (
    basic_stats,
    compute_gene_exp_fraction,
    convert2symbol,
    cook_dist,
    decode,
    filter_genes_by_pattern,
    pca_monocle,
    relative2abs,
    scale,
    top_pca_genes,
)

filter_cells = filter_cells_by_outliers
filter_genes = filter_genes_by_outliers
log1p = log1p_adata
normalize_cells = normalize_cell_expr_by_size_factors

from .preprocess_monocle_utils import estimate_dispersion, top_table
from .Preprocessor import Preprocessor

__all__ = [
    "lambda_correction",
    "calc_sz_factor_legacy",
    "normalize_cell_expr_by_size_factors",
    "recipe_monocle",
    "recipe_velocyto",
    "Gini",
    "top_table",
    "estimate_dispersion",
    "filter_cells_by_outliers",
    "select_genes_monocle",
    "filter_genes",
    "filter_genes_by_outliers",
    "filter_genes_by_clusters_",
    "SVRs",
    "get_svr_filter",
    "highest_frac_genes",
    "cell_cycle_scores",
    "basic_stats",
    "cook_dist",
    "pca_monocle",
    "top_pca_genes",
    "relative2abs",
    "scale",
    "convert2symbol",
    "filter_genes_by_pattern",
    "decode",
    "Preprocessor",
    "log1p",
    "log1p_adata",
    "log1p_adata_layer",
]
