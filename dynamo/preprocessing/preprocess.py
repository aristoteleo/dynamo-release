import warnings
from collections.abc import Iterable
from typing import Callable, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata
import numpy as np
import pandas as pd
import scipy
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import FastICA

from ..configuration import DKM, DynamoAdataConfig, DynamoAdataKeyManager
from ..dynamo_logger import (
    LoggerManager,
    main_critical,
    main_debug,
    main_info,
    main_info_insert_adata_obsm,
    main_info_insert_adata_uns,
    main_warning,
)
from ..tools.utils import update_dict
from ..utils import copy_adata
from .deprecated import _top_table
from .cell_cycle import cell_cycle_scores
from .gene_selection import calc_dispersion_by_svr
from .preprocessor_utils import (
    _infer_labeling_experiment_type,
)
from .pca import pca
from .QC import (
    basic_stats,
    filter_genes_by_clusters,
    filter_cells_by_outliers,
    filter_genes_by_outliers,
)
from .utils import (
    _Freeman_Tukey,
    add_noise_to_duplicates,
    calc_new_to_total_ratio,
    collapse_species_adata,
    compute_gene_exp_fraction,
    convert2symbol,
    convert_layers2csr,
    detect_experiment_datatype,
    get_inrange_shared_counts_mask,
    get_svr_filter,
    merge_adata_attrs,
    unique_var_obs_adata,
)


def vstExprs(
    adata: anndata.AnnData,
    expr_matrix: Union[np.ndarray, None] = None,
    round_vals: bool = True,
) -> np.ndarray:
    """Variance stabilization transformation of the gene expression.

    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Args:
        adata: an AnnData object.
        expr_matrix: an matrix of values to transform. Must be normalized (e.g. by size factors) already. Defaults to
            None.
        round_vals: whether to round expression values to the nearest integer before applying the transformation.
            Defaults to True.

    Returns:
        A numpy array of the gene expression after VST.
    """

    fitInfo = adata.uns["dispFitInfo"]

    coefs = fitInfo["coefs"]
    if expr_matrix is None:
        ncounts = adata.X
        if round_vals:
            if issparse(ncounts):
                ncounts.data = np.round(ncounts.data, 0)
            else:
                ncounts = ncounts.round().astype("int")
    else:
        ncounts = expr_matrix

    def vst(q):  # c( "asymptDisp", "extraPois" )
        return np.log(
            (1 + coefs[1] + 2 * coefs[0] * q + 2 * np.sqrt(coefs[0] * q * (1 + coefs[1] + coefs[0] * q)))
            / (4 * coefs[0])
        ) / np.log(2)

    res = vst(ncounts.toarray()) if issparse(ncounts) else vst(ncounts)

    return res


def highest_frac_genes(
    adata: AnnData,
    store_key: str = "highest_frac_genes",
    n_top: int = 30,
    gene_prefix_list: List[str] = None,
    gene_prefix_only: bool = False,
    layer: Union[str, None] = None,
) -> anndata.AnnData:
    """Compute top genes df and store results in `adata.uns`

    Args:
        adata: an AnnData object
        store_key: key for storing expression percent results. Defaults to "highest_frac_genes".
        n_top: number of top genes to show. Defaults to 30.
        gene_prefix_list: a list of gene name prefixes used for gathering/calculating expression percents from genes
            with these prefixes. Defaults to None.
        gene_prefix_only: whether to calculate percentages for gene groups with the specified prefixes only. It only
            takes effect if gene prefix list is provided. Defaults to False.
        layer: layer on which the gene percents will be computed. Defaults to None.

    Returns:
        An updated adata with top genes df stored in `adata.uns`
    """

    gene_mat = adata.X
    if layer is not None:
        gene_mat = DKM.select_layer_data(layer)
    # compute gene percents at each cell row
    cell_expression_sum = gene_mat.sum(axis=1).flatten()
    # get rid of cells that have all zero counts
    not_all_zero = cell_expression_sum != 0
    filtered_adata = adata[not_all_zero, :]
    cell_expression_sum = cell_expression_sum[not_all_zero]
    main_debug("%d rows(cells or subsets) are not zero. zero total RNA cells are removed." % np.sum(not_all_zero))

    valid_gene_set = set()
    prefix_to_genes = {}
    _adata = filtered_adata
    if gene_prefix_list is not None:
        prefix_to_genes = {prefix: [] for prefix in gene_prefix_list}
        for name in _adata.var_names:
            for prefix in gene_prefix_list:
                length = len(prefix)
                if name[:length] == prefix:
                    valid_gene_set.add(name)
                    prefix_to_genes[prefix].append(name)
                    break
        if len(valid_gene_set) == 0:
            main_critical("NO VALID GENES FOUND WITH REQUIRED GENE PREFIX LIST, GIVING UP PLOTTING")
            return None
        if gene_prefix_only:
            # gathering gene prefix set data
            df = pd.DataFrame(index=_adata.obs.index)
            for prefix in prefix_to_genes:
                if len(prefix_to_genes[prefix]) == 0:
                    main_info("There is no %s gene prefix in adata." % prefix)
                    continue
                df[prefix] = _adata[:, prefix_to_genes[prefix]].X.sum(axis=1)
            # adata = adata[:, list(valid_gene_set)]

            _adata = AnnData(X=df)
            gene_mat = _adata.X

    # compute gene's total percents in the dataset
    gene_percents = np.array(gene_mat.sum(axis=0))
    gene_percents = (gene_percents / gene_mat.shape[1]).flatten()
    # obtain top genes
    sorted_indices = np.argsort(-gene_percents)
    selected_indices = sorted_indices[:n_top]
    gene_names = _adata.var_names[selected_indices]

    gene_X_percents = gene_mat / cell_expression_sum.reshape([-1, 1])

    # assemble a dataframe
    selected_gene_X_percents = np.array(gene_X_percents)[:, selected_indices]
    selected_gene_X_percents = np.squeeze(selected_gene_X_percents)

    top_genes_df = pd.DataFrame(
        selected_gene_X_percents,
        index=adata.obs_names,
        columns=gene_names,
    )
    gene_percents_df = pd.DataFrame(
        gene_percents, index=_adata.var_names, columns=["percent"]
    )  # Series is not appropriate for h5ad format.

    main_info_insert_adata_uns(store_key)
    adata.uns[store_key] = {
        "top_genes_df": top_genes_df,
        "gene_mat": gene_mat,
        "layer": layer,
        "selected_indices": selected_indices,
        "gene_prefix_list": gene_prefix_list,
        "show_individual_prefix_gene": gene_prefix_only,
        "gene_percents": gene_percents_df,
    }

    return adata
