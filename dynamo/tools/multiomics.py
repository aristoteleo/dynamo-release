from anndata import AnnData
import numpy as np

# 1. concatenate RNA/protein data
# 2. filter gene/protein for velocity
# 3. use steady state assumption to calculate protein velocity
# 4. use the PRL paper to estimate the parameters


def AddAssay(adata, data, key, slot="obsm"):
    """Add a new data as a key to the specified slot

    Parameters
    ----------
        adata: :AnnData
            AnnData object
        data: `pd.DataFrame`
            The data (in pandas DataFrame format) that will be added to adata.
        key: `str`
            The key name to be used for the new data.
        slot: `str`
            The slot of adata to store the new data.

    Returns
    -------
        adata: :AnnData
            A updated anndata object that are updated with a new data as a key to the specified slot.
    """

    if slot == "uns":
        adata.uns[key] = data.loc[
            adata.obs.index, set(adata.var.index).intersection(data.columns)
        ]
    elif slot == "obsm":
        adata.obsm[key] = data.loc[
            adata.obs.index, set(adata.var.index).intersection(data.columns)
        ]

    return adata


def getAssay(adata, key, slot="obsm"):
    """Retrieve a key named data from the specified slot

    Parameters
    ----------
        adata: :AnnData
            AnnData object
        key: `str`
            The key name to be used for the new data.
        slot: `str`
            The slot of adata to store the new data.

    Returns
    -------
        data: `pd.DataFrame`
            The data (in pandas DataFrame format) that will be retrieved from adata.
    """

    if slot == "uns":
        data = adata.uns[key]
    elif slot == "obsm":
        data = adata.obsm[key]

    return data
