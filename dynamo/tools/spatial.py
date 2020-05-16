from tqdm import tqdm
import numpy as np
import pandas as pd
import pysal
from scipy.sparse import issparse
import warnings
from .utils import fdr


def Moran_I(adata,
            X_data = None,
            genes=None,
            layer=None,
            weighted=True,
            assumption='permutation',
            local_moran=False):
    """Identify genes with strong spatial autocorrelation with Moran's I test. This can be used to identify genes that are
    potentially related to critical dynamic process. Moran's I test is first introduced in single cell genomics analysis
    in (Cao, et al, 2019).

    Global Moran's I test is based on pysal. More details can be found at:
    http://geodacenter.github.io/workbook/5a_global_auto/lab5a.html#morans-i

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        X_data: `np.ndarray` (default: `None`)
            The user supplied data that will be used for clustering directly.
        genes: `list` or None (default: `None`)
            The list of genes that will be used to subset the data for dimension reduction and clustering. If `None`, all
            genes will be used.
        layer: `str` or None (default: `None`)
            The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is used.
        weighted: `bool` (default: `True`)
            Whether to consider edge weights in the spatial weights graph.
        assumption: `str` (default: `permutation`)
            Assumption of the Moran's I test, can be one of {'permutation', 'normality', 'randomization'}.
            Inference for Moran’s I is based on a null hypothesis of spatial randomness. The distribution of the
            statistic under the null can be derived using either an assumption of normality (independent normal random
            variates), or so-called randomization (i.e., each value is equally likely to occur at any location). An
            alternative to an analytical derivation is a computational approach based on permutation. This calculates a
            reference distribution for the statistic under the null hypothesis of spatial randomness by randomly permuting
            the observed values over the locations. The statistic is computed for each of these randomly reshuffled data sets,
            which yields a reference distribution.
        local_moran: `bool` (default: `False`)
            Whether to also calculate local Moran's I.

    Returns
    -------
        Returns an updated `~anndata.AnnData` with a new property `'Moran_' + type` in the .uns attribute.
    """
    if X_data is None:
        if genes is not None:
            genes = adata.var_name.intersection(genes).to_list()
            if len(genes) == 0:
                raise ValueError(f'No genes from your genes list appear in your adata object.')

        if layer == None:
            if genes is not None:
                X_data = adata[:, genes].X
            else:
                X_data = adata.X if 'use_for_dynamo' not in adata.var.keys() \
                    else adata[:, adata.var.use_for_dynamo].X
                genes = adata.var_names[adata.var.use_for_dynamo]
        else:
            if genes is not None:
                X_data = adata[:, genes].layers[layer]
            else:
                X_data = adata.layers[layer] if 'use_for_dynamo' not in adata.var.keys() \
                    else adata[:, adata.var.use_for_dynamo].layers[layer]
                genes = adata.var_names[adata.var.use_for_dynamo]

    cell_num, gene_num = X_data.shape

    embedding_key = (
        "X_umap" if layer is None else layer + "_umap"
    )
    neighbor_key = "neighbors" if layer is None else layer + "_neighbors"
    if neighbor_key not in adata.uns.keys():
        warnings.warn(f"Neighbor graph is required for Moran's I test. No neighbor_key {neighbor_key} exists in the data. "
                      f"Running reduceDimension which will generate the neighbor graph and the associated low dimension"
                      f"embedding {embedding_key}. ")
        from .dimension_reduction import reduceDimension
        adata = reduceDimension(adata, X_data=X_data, layer=layer)

    neighbor_graph = adata.uns["neighbors"]["connectivities"]

    # convert a sparse adjacency matrix to a dictionary
    adj_dict = {
        i: row.indices
        for i, row in enumerate(neighbor_graph)
    }
    if weighted :
        weight_dict = {
            i: row.data
            for i, row in enumerate(neighbor_graph)
        }
        W = pysal.lib.weights.W(adj_dict, weight_dict)
    else:
        W = pysal.lib.weights.W(adj_dict)

    sparse = issparse(X_data)

    Moran_I, p_value, statistics = [None] * gene_num, [None] * gene_num, [None] * gene_num
    if local_moran:
        l_moran = np.zeros(X_data.shape)
    for cur_g in tqdm(range(gene_num), desc="Moran’s I Global Autocorrelation Statistic"):
        cur_X = X_data[:, cur_g].A if sparse else X_data[:, cur_g]
        mbi = pysal.explore.esda.moran.Moran(cur_X, W, two_tailed=False)

        Moran_I[cur_g] = mbi.I
        p_value[cur_g] = mbi.p_sim if assumption == 'permutation' else \
            mbi.p_norm if assumption == 'normality' else mbi.p_rand
        statistics[cur_g] = mbi.z_sim if assumption == 'permutation' else \
            mbi.z_norm if assumption == 'normality' else mbi.z_sim
        if local_moran:
            l_moran[:, cur_g] = pysal.explore.esda.moran.Moran_Local(cur_X, W).Is

    Moran_res = pd.DataFrame(
        {"Moran_I": Moran_I,
         "Moran_p_val": p_value,
         "Moran_q_val": fdr(np.array(p_value)),
         "Moran_z": statistics,
         }, index=genes
    )

    adata.var = adata.var.merge(Moran_res, left_index=True, right_index=True, how='left')
    if local_moran:
        adata.uns['local_moran'] = l_moran

    return adata


def find_group_markers(adata, group, genes, layer):
    # mean expression
    # specifity
    # percentage of expression in a group
    # number of cells in each group
    # logFC
    pass
