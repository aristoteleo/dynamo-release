# =================================================================
# Original Code Repository Author: Lause, J., Berens, P. & Kobak, D.
# Adapted to Dynamo by: dynamo authors
# Created Date: 12/16/2021
# Description: Sctrasnform for preprocessing single cell expression data adapated from R counterpart implemented in Seurat
# Reference: Lause, J., Berens, P. & Kobak, D. Analytic Pearson residuals for normalization of single-cell RNA-seq UMI data. Genome Biol 22, 258 (2021). https://doi.org/10.1186/s13059-021-02451-7
# =================================================================

import warnings
from typing import Dict, Optional, Tuple
from warnings import warn

import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
from anndata import AnnData
from scipy.sparse import issparse

from ...configuration import DKM
from ...dynamo_logger import (
    LoggerManager,
    main_info,
    main_info_insert_adata_layer,
    main_warning,
)
from ...preprocessing.utils import is_nonnegative_integer_arr, seurat_get_mean_var
from ..QC import filter_genes_by_outliers

main_logger = LoggerManager.main_logger

# TODO: Use compute_pearson_residuals function to calculate residuals
def _highly_variable_pearson_residuals(
    adata: AnnData,
    theta: float = 100,
    clip: Optional[float] = None,
    n_top_genes: int = 1000,
    batch_key: Optional[str] = None,
    chunksize: int = 1000,
    check_values: bool = True,
    layer: Optional[str] = None,
    subset: bool = False,
    inplace: bool = True,
) -> Optional[pd.DataFrame]:
    """Compute and annotate highly variable genes based on analytic Pearson residuals [Lause21]_.

    For [Lause21]_, Pearson residuals of a negative binomial offset model (with overdispersion theta shared across
    genes) are computed. By default, overdispersion theta=100 is used and residuals are clipped to sqrt(n). Finally,
    genes are ranked by residual variance. Expects raw count input.

    Args:
        adata: an annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
        theta: the negative binomial overdispersion parameter theta for Pearson residuals. Higher values correspond to
            less overdispersion (var = mean + mean^2/theta), and `theta=np.Inf` corresponds to a Poisson model.
        clip: the threshold to determine if and how residuals are clipped. If `None`, residuals are clipped to the
            interval [-sqrt(n), sqrt(n)] where n is the number of cells in the dataset (default behavior). If any
            scalar c, residuals are clipped to the interval [-c, c]. Set `clip=np.Inf` for no clipping.
        n_top_genes: the number of highly-variable genes to keep.
        batch_key: the key to indicate how highly-variable genes are selected within each batch separately and merged
            later. This simple process avoids the selection of batch-specific genes and acts as a lightweight batch
            correction method. Genes are first sorted by how many batches they are an HVG. Ties are broken by the median
            rank (across batches) based on within-batch residual variance.
        chunksize: the number of genes are processed at once while computing the Pearson residual variance. Choosing a
            smaller value will reduce the required memory.
        check_values: whether to check if counts in selected layer are integers. A Warning is returned if set to True.
        layer: the layer to perform gene selection on.
        subset: whether to inplace subset to highly-variable genes. If `True` otherwise merely indicate highly variable
            genes.
        inplace: whether to place calculated metrics in `.var` or return them.

    Returns:
        Depending on `inplace` returns calculated metrics (:class:`~pandas.DataFrame`) or updates `.var` with
        the following fields:
            highly_variable: boolean indicator of highly-variable genes.
            means: means per gene.
            variances: variance per gene.
            residual_variances: For `recipe='pearson_residuals'`, residual variance per gene. Averaged in the case of
                multiple batches.
            highly_variable_rank: For `recipe='pearson_residuals'`, rank of the gene according to residual variance,
                median rank in the case of multiple batches.
            highly_variable_nbatches: If `batch_key` given, denotes in how many batches genes are detected as HVG.
            highly_variable_intersection: If `batch_key` given, denotes the genes that are highly variable in all
                batches.
    """

    # view_to_actual(adata)
    X = DKM.select_layer_data(adata, layer)
    _computed_on_prompt_str = layer if layer else "adata.X"

    # Check for raw counts
    if check_values and (is_nonnegative_integer_arr(X) is False):
        warnings.warn(
            "`flavor='pearson_residuals'` expects raw count data, but non-integers were found.",
            UserWarning,
        )
    # check theta
    if theta <= 0:
        # TODO: would "underdispersion" with negative theta make sense?
        # then only theta=0 were undefined..
        raise ValueError("Pearson residuals require theta > 0")
    # prepare clipping

    if batch_key is None:
        batch_info = np.zeros(adata.shape[0], dtype=int)
    else:
        batch_info = adata.obs[batch_key].values
    n_batches = len(np.unique(batch_info))

    # Get pearson residuals for each batch separately
    residual_gene_vars_by_batch = []
    for batch in np.unique(batch_info):

        adata_subset = adata[batch_info == batch]

        # Filter out zero genes

        nonzero_genes = filter_genes_by_outliers(
            adata_subset, min_cell_s=1, min_cell_u=0, min_cell_p=0, shared_count=None
        )
        # nonzero_genes = filter_genes(adata_subset, min_cells=1, inplace=False)[0]
        adata_subset = adata_subset[:, nonzero_genes]

        if layer is not None:
            X_batch = adata_subset.layers[layer]
        else:
            X_batch = adata_subset.X

        # Prepare clipping
        if clip is None:
            n = X_batch.shape[0]
            clip = np.sqrt(n)
        if clip < 0:
            raise ValueError("Pearson residuals normalization requires `clip>=0` or `clip=None`.")

        if sp_sparse.issparse(X_batch):
            sums_genes = np.sum(X_batch, axis=0)
            sums_cells = np.sum(X_batch, axis=1)
            sum_total = np.sum(sums_genes).squeeze()
        else:
            sums_genes = np.sum(X_batch, axis=0, keepdims=True)
            sums_cells = np.sum(X_batch, axis=1, keepdims=True)
            sum_total = np.sum(sums_genes)

        # Compute pearson residuals in chunks
        residual_gene_var = np.empty((X_batch.shape[1]))
        for start in np.arange(0, X_batch.shape[1], chunksize):
            stop = start + chunksize
            mu = np.array(sums_cells @ sums_genes[:, start:stop] / sum_total)
            X_dense = X_batch[:, start:stop].toarray()
            residuals = (X_dense - mu) / np.sqrt(mu + mu**2 / theta)
            residuals = np.clip(residuals, a_min=-clip, a_max=clip)
            residual_gene_var[start:stop] = np.var(residuals, axis=0)

        # Add 0 values for genes that were filtered out
        unmasked_residual_gene_var = np.zeros(len(nonzero_genes))
        unmasked_residual_gene_var[nonzero_genes] = residual_gene_var
        residual_gene_vars_by_batch.append(unmasked_residual_gene_var.reshape(1, -1))

    residual_gene_vars_by_batch = np.concatenate(residual_gene_vars_by_batch, axis=0)

    # Get rank per gene within each batch
    # argsort twice gives ranks, small rank means most variable
    ranks_residual_var = np.argsort(np.argsort(-residual_gene_vars_by_batch, axis=1), axis=1)
    ranks_residual_var = ranks_residual_var.astype(np.float32)
    # count in how many batches a genes was among the n_top_genes
    highly_variable_nbatches = np.sum((ranks_residual_var < n_top_genes).astype(int), axis=0)
    # set non-top genes within each batch to nan
    ranks_residual_var[ranks_residual_var >= n_top_genes] = np.nan
    ranks_masked_array = np.ma.masked_invalid(ranks_residual_var)
    # Median rank across batches, ignoring batches in which gene was not selected
    medianrank_residual_var = np.ma.median(ranks_masked_array, axis=0).filled(np.nan)

    means, variances = seurat_get_mean_var(X)
    df = pd.DataFrame.from_dict(
        dict(
            means=means,
            variances=variances,
            # mean of each batch
            residual_variances=np.mean(residual_gene_vars_by_batch, axis=0),
            highly_variable_rank=medianrank_residual_var,
            highly_variable_nbatches=highly_variable_nbatches.astype(np.int64),
            highly_variable_intersection=highly_variable_nbatches == n_batches,
        )
    )
    df = df.set_index(adata.var_names)

    # Sort genes by how often they selected as hvg within each batch and
    # break ties with median rank of residual variance across batches
    df.sort_values(
        ["highly_variable_nbatches", "highly_variable_rank"],
        ascending=[False, True],
        na_position="last",
        inplace=True,
    )

    high_var = np.zeros(df.shape[0])
    high_var[:n_top_genes] = True
    df[DKM.VAR_GENE_HIGHLY_VARIABLE_KEY] = high_var.astype(bool)
    df = df.loc[adata.var_names, :]

    if inplace:
        adata.uns[DKM.UNS_PP_KEY]["hvg"] = {"flavor": "pearson_residuals", "computed_on": _computed_on_prompt_str}
        main_logger.debug(
            "added\n"
            "    'highly_variable', boolean vector (adata.var)\n"
            "    'highly_variable_rank', float vector (adata.var)\n"
            "    'highly_variable_nbatches', int vector (adata.var)\n"
            "    'highly_variable_intersection', boolean vector (adata.var)\n"
            "    'means', float vector (adata.var)\n"
            "    'variances', float vector (adata.var)\n"
            "    'residual_variances', float vector (adata.var)"
        )
        adata.var["means"] = df["means"].values
        adata.var["variances"] = df["variances"].values
        adata.var["residual_variances"] = df["residual_variances"]
        adata.var["highly_variable_rank"] = df["highly_variable_rank"].values
        if batch_key is not None:
            adata.var["highly_variable_nbatches"] = df["highly_variable_nbatches"].values
            adata.var["highly_variable_intersection"] = df["highly_variable_intersection"].values
        adata.var[DKM.VAR_GENE_HIGHLY_VARIABLE_KEY] = df[DKM.VAR_GENE_HIGHLY_VARIABLE_KEY].values
        adata.var[DKM.VAR_USE_FOR_PCA] = df[
            DKM.VAR_GENE_HIGHLY_VARIABLE_KEY
        ].values  # set use_for_pca for down stream analysis in dynamo

        if subset:
            adata._inplace_subset_var(df[DKM.VAR_GENE_HIGHLY_VARIABLE_KEY].values)

    else:
        if batch_key is None:
            df = df.drop(["highly_variable_nbatches", "highly_variable_intersection"], axis=1)
        if subset:
            df = df.iloc[df.highly_variable.values, :]

        return df


# TODO: Move this function to a higher level. Now this function is called by
# pearson_residual_recipe, but this function aims to support different recipe in
# the future.
def compute_highly_variable_genes(
    adata: AnnData,
    *,
    theta: float = 100,
    clip: Optional[float] = None,
    n_top_genes: Optional[int] = None,
    batch_key: Optional[str] = None,
    chunksize: int = 1000,
    recipe: str = "pearson_residuals",
    check_values: bool = True,
    layer: Optional[str] = None,
    subset: bool = False,
    inplace: bool = True,
) -> Optional[pd.DataFrame]:
    """A wrapper calls corresponding recipe to identify highly variable genes. Currently only 'pearson_residuals'
    is supported.
    """

    main_logger.info("extracting highly variable genes")

    if not isinstance(adata, AnnData):
        raise ValueError(
            "`pp.highly_variable_genes` expects an `AnnData` argument, "
            "pass `inplace=False` if you want to return a `pd.DataFrame`."
        )

    if recipe == "pearson_residuals":
        if n_top_genes is None:
            raise ValueError(
                "`pp.highly_variable_genes` requires the argument `n_top_genes`" " for `flavor='pearson_residuals'`"
            )
        return _highly_variable_pearson_residuals(
            adata,
            layer=layer,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            theta=theta,
            clip=clip,
            chunksize=chunksize,
            subset=subset,
            check_values=check_values,
            inplace=inplace,
        )


def compute_pearson_residuals(
    X: np.ndarray,
    theta: float = 100,
    clip: Optional[float] = None,
    check_values: bool = True,
    copy: bool = False,
) -> np.ndarray:
    """Compute Pearson residuals from count data.

    Pearson residuals are a measure of the deviation of observed counts from expected counts under a Poisson or negative
    binomial model.

    Args:
        X: array_like count matrix, shape (n_cells, n_genes).
        theta: the dispersion parameter for the negative binomial model. Must be positive.
        clip: The maximum absolute value of the residuals. Residuals with absolute value larger than `clip` are clipped
            to `clip`. If `None`,`clip` is set to the square root of the number of cells in `X`.
        check_values: whether to check if `X` contains non-negative integers. If `True` and non-integer values are
            found, a `UserWarning` is issued.
        copy: whether to make a copy of `X`.

    Returns:
        The Pearson residuals.
    """
    X = X.copy() if copy else X

    # check theta
    if theta <= 0:
        # TODO: would "underdispersion" with negative theta make sense?
        # then only theta=0 were undefined..
        raise ValueError("Pearson residuals require theta > 0")
    # prepare clipping
    if clip is None:
        n = X.shape[0]
        clip = np.sqrt(n)
    if clip < 0:
        raise ValueError("Pearson residuals require `clip>=0` or `clip=None`.")

    if check_values and not is_nonnegative_integer_arr(X):
        warn(
            "`normalize_pearson_residuals()` expects raw count data, but non-integers were found.",
            UserWarning,
        )

    if issparse(X):
        sums_genes = np.sum(X, axis=0)
        sums_cells = np.sum(X, axis=1)
        sum_total = np.sum(sums_genes).squeeze()
    else:
        sums_genes = np.sum(X, axis=0, keepdims=True)
        sums_cells = np.sum(X, axis=1, keepdims=True)
        sum_total = np.sum(sums_genes)

    mu = np.array(sums_cells @ sums_genes / sum_total)
    diff = np.array(X - mu)
    residuals = diff / np.sqrt(mu + mu**2 / theta)

    # clip
    residuals = np.clip(residuals, a_min=-clip, a_max=clip)

    return residuals


# TODO: Read pearson residuals if they exist instead of calculating them again.
def _normalize_single_layer_pearson_residuals(
    adata: AnnData,
    *,
    theta: float = 100,
    clip: Optional[float] = None,
    check_values: bool = True,
    layer: Optional[str] = None,
    var_select_genes_key: np.array = None,
    copy: bool = False,
) -> Optional[AnnData]:
    """Applies analytic Pearson residual normalization, based on [Lause21]_.

    The residuals are based on a negative binomial offset model with overdispersion `theta` shared across genes. By
    default, residuals are clipped to sqrt(n) and overdispersion `theta=100` is used. Expects raw count input.

    Args:
        adata: the annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
        theta: the negative binomial overdispersion parameter theta for Pearson residuals. Higher values correspond to
            less overdispersion (var = mean + mean^2/theta), and `theta=np.Inf` corresponds to a Poisson model.
        clip: the threshold to determine if and how residuals are clipped. If `None`, residuals are clipped to the
            interval [-sqrt(n), sqrt(n)] where n is the number of cells in the dataset (default behavior). If any
            scalar c, residuals are clipped to the interval [-c, c]. Set `clip=np.Inf` for no clipping.
        check_values: whether to check if counts in selected layer are integers. A Warning is returned if set to True.
        layer: the Layer to normalize instead of `X`.
        copy: Whether to modify copied input object.

    Returns:
        If `copy` is True, a new AnnData object with normalized layers will be returned. If `copy` is False, modifies
        the given AnnData object in place and returns None.
    """

    msg = "applying Pearson residuals to layer <%s>" % layer
    main_logger.info(msg)
    main_logger.log_time()

    if layer is None:
        layer = DKM.X_LAYER

    if layer != DKM.X_LAYER:
        main_warning(
            f"Pearson residual is only recommended for X layer while you are applying on layer: {layer}, "
            f"This will overwrite existing pearson residual params and create negative values in layers, "
            f"which will cause error in the velocities calculation. Please run the pearson residual recipe by default "
            f"if you plan to perform downstream analysis."
        )
        copy = True  # residuals for spliced/unspliced layers will be saved in X_splice/X_unspliced.

    if copy:
        adata = adata.copy()

    pp_pearson_store_key = DKM.gen_layer_pearson_residual_key(layer)

    selected_genes_bools = np.ones(adata.n_vars, dtype=bool)
    if var_select_genes_key:
        selected_genes_bools = adata.var[var_select_genes_key]

    adata_selected_genes = adata[:, selected_genes_bools]
    X = DKM.select_layer_data(adata_selected_genes, layer=layer)

    residuals = compute_pearson_residuals(X, theta, clip, check_values, copy=copy)
    pearson_residual_params_dict = dict(theta=theta, clip=clip, layer=layer)

    main_logger.info("replacing layer <%s> with pearson residual normalized data." % layer)
    DKM.set_layer_data(adata, layer, residuals, selected_genes_bools)
    adata.uns["pp"][pp_pearson_store_key] = pearson_residual_params_dict

    main_logger.finish_progress(progress_name=f"pearson residual normalization for {layer}")

    if copy:
        return adata


def normalize_layers_pearson_residuals(
    adata: AnnData,
    layers: list = ["X"],
    select_genes_layer="X",
    select_genes_key="use_for_pca",
    copy=False,
    **normalize_pearson_residual_args,
) -> None:
    """Normalize the given layers of the AnnData object using Pearson residuals.

    Args:
        adata: AnnData object to normalize.
        layers: the list of layers to normalize.
        select_genes_layer: the layer to select highly variable genes.
        select_genes_key: the key to use for selecting highly variable genes.
        copy: Whether to create a copy of the AnnData object before editing it.
        **normalize_pearson_residual_args: Additional arguments to pass to the
            `_normalize_single_layer_pearson_residuals` function.

    Returns:
        None. Anndata object will be updated.
    """
    if len(layers) == 0:
        main_warning("layers arg has zero length. return and do nothing in normalize_layers_pearson_residuals.")
    if not select_genes_layer in layers:
        main_warning(
            "select_genes_layer: %s not in layers, using layer: %s instead to select genes instead."
            % (select_genes_layer, layers[0])
        )
        select_genes_layer = layers[0]

    for layer in layers:
        temp_select_genes_key = None

        # if the current layer is used for selecting genes
        if select_genes_layer == layer:
            temp_select_genes_key = select_genes_key

        temp_adata = _normalize_single_layer_pearson_residuals(
            adata, layer=layer, var_select_genes_key=temp_select_genes_key, copy=copy, **normalize_pearson_residual_args
        )

        if layer != DKM.X_LAYER:  # update 'X_' layers
            new_X_key = DKM.gen_layer_X_key(layer)
            main_info_insert_adata_layer(new_X_key, indent_level=2)
            adata.layers[new_X_key] = DKM.select_layer_data(temp_adata, layer)


# TODO: Combine this function with compute_highly_variable_genes.
def select_genes_by_pearson_residuals(
    adata: AnnData,
    layer: str = None,
    theta: float = 100,
    clip: Optional[float] = None,
    n_top_genes: int = 2000,
    batch_key: Optional[str] = None,
    chunksize: int = 1000,
    check_values: bool = True,
    inplace: bool = True,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Gene selection and normalization based on [Lause21]_.

    This function applies gene selection based on Pearson residuals. Expects raw count input on the resulting subset.

    Args:
        adata: an annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
        layer: the layer to perform gene selection on.
        theta: the negative binomial overdispersion parameter theta for Pearson residuals. Higher values correspond to
            less overdispersion (var = mean + mean^2/theta), and `theta=np.Inf` corresponds to a Poisson model.
        clip: the threshold to determine if and how residuals are clipped. If `None`, residuals are clipped to the
            interval [-sqrt(n), sqrt(n)] where n is the number of cells in the dataset (default behavior). If any
            scalar c, residuals are clipped to the interval [-c, c]. Set `clip=np.Inf` for no clipping.
        n_top_genes: the number of highly-variable genes to keep.
        batch_key: the key to indicate how highly-variable genes are selected within each batch separately and merged
            later. This simple process avoids the selection of batch-specific genes and acts as a lightweight batch
            correction method. Genes are first sorted by how many batches they are an HVG. Ties are broken by the median
            rank (across batches) based on within-batch residual variance.
        chunksize: the number of genes are processed at once while computing the Pearson residual variance. Choosing a
            smaller value will reduce the required memory.
        check_values: whether to check if counts in selected layer are integers. A Warning is returned if set to True.
        inplace: whether to place results in `adata` or return them.

    Returns:
        If inplace is 'True', the 'adata' will be updated without return values. Otherwise, the 'adata' object and
        selected highly-variable genes will be returned.
    """
    if layer is None:
        layer = DKM.X_LAYER
    main_info("gene selection on layer: " + layer)
    if DKM.UNS_PP_KEY not in adata.uns:
        DKM.init_uns_pp_namespace(adata)

    # highly variable genes calculation args
    hvg_params = dict(
        recipe="pearson_residuals",
        n_top_genes=n_top_genes,
        batch_key=batch_key,
        theta=theta,
        clip=clip,
        chunksize=chunksize,
        check_values=check_values,
    )

    if inplace:
        compute_highly_variable_genes(adata, **hvg_params, inplace=True)
    else:
        hvg = compute_highly_variable_genes(adata, **hvg_params, inplace=False)

    if inplace:
        return None
    else:
        return adata, hvg


def pearson_residuals(
    adata: AnnData,
    n_top_genes: Optional[int] = 3000,
    subset: bool = False,
    theta: float = 100,
    clip: Optional[float] = None,
    check_values: bool = True,
) -> None:
    """Preprocess UMI count data with analytic Pearson residuals.

    Pearson residuals transform raw UMI counts into a representation where three aims are achieved:
        1.Remove the technical variation that comes from differences in total counts between cells.
        2.Stabilize the mean-variance relationship across genes, i.e. ensure that biological signal from both low and
            high expression genes can contribute similarly to downstream processing.
        3.Genes that are homogeneously expressed (like housekeeping genes) have small variance, while genes that are
            differentially expressed (like marker genes) have high variance.

    Args:
        adata: An anndata object.
        n_top_genes: Number of highly-variable genes to keep.
        subset: Inplace subset to highly-variable genes if `True` otherwise merely indicate highly variable genes.
        theta: The negative binomial overdispersion parameter theta for Pearson residuals. Higher values correspond to
            less overdispersion (var = mean + mean^2/theta), and `theta=np.Inf` corresponds to a Poisson model.
        clip: Determines if and how residuals are clipped:
                * If `None`, residuals are clipped to the interval [-sqrt(n), sqrt(n)], where n is the number of cells
                    in the dataset (default behavior).
                * If any scalar c, residuals are clipped to the interval [-c, c]. Set `clip=np.Inf` for no clipping.
        check_values: Check if counts in selected layer are integers. A Warning is returned if set to True.

    Returns:
        Updates adata with the field ``adata.obsm["pearson_residuals"]``, containing pearson_residuals.
    """
    if not (n_top_genes is None):
        compute_highly_variable_genes(
            adata, n_top_genes=n_top_genes, recipe="pearson_residuals", inplace=True, subset=subset
        )

    X = adata.X.copy()
    residuals = compute_pearson_residuals(X, theta=theta, clip=clip, check_values=check_values)
    adata.obsm["pearson_residuals"] = residuals
