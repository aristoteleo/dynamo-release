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
            local_moran=False):
    """Identify genes with strong spatial autocorrelation with Moran's I test. This can be used to identify genes that are
    potentially related to critical dynamic process. Moran's I test is first introduced in single cell genomics analysis
    in (Cao, et al, 2019).

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
        i: np.nonzero(row.toarray().squeeze())[0].tolist()
        for i, row in enumerate(neighbor_graph)
    }

    W = pysal.lib.weights.W(adj_dict)
    sparse = issparse(X_data)

    Moran_I, p_value = [None] * gene_num, [None] * gene_num
    if local_moran:
        l_moran = np.zeros_like(X_data)
    for cur_g in tqdm(range(gene_num), desc="Moran’s I Global Autocorrelation Statistic"):
        cur_X = X_data[:, cur_g].A if sparse else X_data[:, cur_g]
        if layer is not None and layer.startwith('velocity '):
            mbi = pysal.explore.esda.moran.Moran_Rate(cur_X, W, two_tailed=False)
        else:
            mbi = pysal.explore.esda.moran.Moran(cur_X, W, two_tailed=False)
        Moran_I[cur_g] = mbi.I
        p_value[cur_g] = mbi.p_norm
        if local_moran:
            if layer is not None and layer.startwith('velocity '):
                l_moran[cur_g, :] = pysal.explore.esda.moran.Moran_Local_Rate(cur_X, W, two_tailed=False)
            else:
                l_moran[cur_g, :] = pysal.explore.esda.moran.Moran_Local(cur_X, W).Is

    Moran_res = pd.DataFrame(
        {"Moran_I": Moran_I,
         "Moran_p_val": p_value,
         "Moran_q_val": fdr(np.array(p_value))}, index=genes
    )

    adata.var = adata.var.merge(Moran_res, left_index=True, right_index=True, how='left')
    if local_moran:
        adata.uns['local_moran'] = l_moran

    return adata
