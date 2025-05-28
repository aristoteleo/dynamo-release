import anndata as ad
from anndata import AnnData
from .MultiConfiguration import MDKM




import numpy as np
import pandas as pd

from scipy.sparse import (
    issparse
)
from typing import (
    List,
    Literal,
    Optional,
    Union,
)

# Define several Literals - might move to MDKM
ModalityType = Literal['atac', 'cite', 'hic', 'rna']
ObsKeyType = Literal['n_genes_by_counts', 'total_counts']
VarKeyType = Literal['n_cells_by_counts']

# Imports from dynamo
from ..dynamo_logger import (
    LoggerManager,
    main_debug,
    main_exception,
    main_finish_progress,
    main_info,
    main_info_insert_adata,
    main_warning,
)

def modality_basic_stats(
        mdata                     ,
        modality: ModalityType = None
):
    from mudata import MuData
    """Generate basic stats of the adata, including number of genes, number of cells, and number of mitochondria genes.

    Args:
        adata: an AnnData object.

    Returns:
        An updated AnnData object with a number of QC metrics computed: 'n_cells_by_counts', 'n_features_by_counts', and
        'total_counts'.  (Note: since most modalities do not have direct information about related genes, fractions of
        mitochondrial genes cannot be computed.)
    """
    from muon import atac as ac
    modality_adata = mdata.mod.get(modality, None)
    if modality_adata is None:
        raise ValueError(f'Modality {modality} not found in MuData object.')

    # Compute QC metrics via functionality in scanpy
    import scanpy as sc
    sc.pp.calculate_qc_metrics(modality_adata,  percent_top=None, log1p=False, inplace=True)

    # Compute modality specific QC metrics
    if modality == 'atac':
        ac.tl.nucleosome_signal(mdata, n=1e6)


def modality_filter_cells_by_outliers(
        mdata,
        modality:   ModalityType = 'atac',
        obs_key:    VarKeyType = 'n_cells_by_counts',
        quantiles:  Optional[Union[List[float], float]] = [0.01, 0.99],
        thresholds: Optional[Union[List[float], float]] = None
):
    from mudata import MuData
    import muon as mu
    modality_adata = mdata.mod.get(modality, None)
    if modality_adata is None:
        raise ValueError(f'Modality {modality} not found in MuData object.')

    if quantiles is not None:
        # Thresholds were specified as quantiles
        qc_parameter_series = modality_adata.obs[obs_key]

        if isinstance(quantiles, list):
            if len(quantiles) > 2:
                raise ValueError(f'More than 2 quantiles were specified {len(quantiles)}.')

            min_feature_thresh, max_feature_thresh = qc_parameter_series.quantile(quantiles).tolist()
        else:
            min_feature_thresh, max_feature_thresh = qc_parameter_series.quantile(quantiles), np.inf
    else:
        # Thresholds were specified as absolute thresholds
        if isinstance(thresholds, list):
            if len(thresholds) > 2:
                raise ValueError(f'More than 2 thresholds were specified {len(thresholds)}.')

            min_feature_thresh, max_feature_thresh = thresholds
        else:
            min_feature_thresh, max_feature_thresh = thresholds, np.inf

    # Carry out the actual filtering
    pre_filter_n_cells = modality_adata.n_obs
    mu.pp.filter_obs(modality_adata, obs_key, lambda x: (x >= min_feature_thresh) & (x <= max_feature_thresh))
    post_filter_n_cells = modality_adata.n_obs
    main_info(f'filtered out {pre_filter_n_cells - post_filter_n_cells} outlier cells', indent_level=2)


def modality_filter_features_by_outliers(
        mdata,
        modality:   ModalityType = 'atac',
        quantiles:  Optional[Union[List[float], float]] = [0.01, 0.99],
        thresholds: Optional[Union[List[float], float]] = None,
        var_key:    ObsKeyType = 'n_cells_by_counts'
):
    from mudata import MuData
    import muon as mu
    modality_adata = mdata.mod.get(modality, None)
    if modality_adata is None:
        raise ValueError(f'Modality {modality} not found in MuData object.')

    if quantiles is not None:
        # Thresholds were specified as quantiles
        qc_parameter_series = modality_adata.var[var_key]

        if isinstance(quantiles, list):
            if len(quantiles) > 2:
                raise ValueError(f'More than 2 quantiles were specified {len(quantiles)}.')

            min_feature_thresh, max_feature_thresh = qc_parameter_series.quantile(quantiles).tolist()
        else:
            min_feature_thresh, max_feature_thresh = qc_parameter_series.quantile(quantiles), np.inf
    else:
        # Thresholds were specified as absolute thresholds
        if isinstance(thresholds, list):
            if len(thresholds) > 2:
                raise ValueError(f'More than 2 thresholds were specified {len(thresholds)}.')

            min_feature_thresh, max_feature_thresh = thresholds
        else:
            min_feature_thresh, max_feature_thresh = thresholds, np.inf

    # Carry out the actual filtering
    pre_filter_n_cells = modality_adata.n_obs
    mu.pp.filter_var(modality_adata, var_key, lambda x: (x >= min_feature_thresh) & (x <= max_feature_thresh))
    post_filter_n_cells = modality_adata.n_obs
    main_info(f'filtered out {pre_filter_n_cells - post_filter_n_cells} outlier features', indent_level=2)
