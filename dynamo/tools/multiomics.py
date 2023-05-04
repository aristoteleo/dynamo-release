import anndata
import pandas as pd

# 1. concatenate RNA/protein data
# 2. filter gene/protein for velocity
# 3. use steady state assumption to calculate protein velocity
# 4. use the PRL paper to estimate the parameters


def AddAssay(adata: anndata.AnnData, data: pd.DataFrame, key: str, slot: str = "obsm") -> anndata.AnnData:
    """Add a new data as a key to the specified slot.

    Args:
        adata: an AnnData object.
        data: the data (in pandas DataFrame format) that will be added to adata.
        key: the key name to be used for the new data.
        slot: the slot of adata to store the new data. Defaults to "obsm".

    Returns:
        An updated anndata object that are updated with a new data as a key to the specified slot.
    """

    if slot == "uns":
        adata.uns[key] = data.loc[adata.obs.index, set(adata.var.index).intersection(data.columns)]
    elif slot == "obsm":
        adata.obsm[key] = data.loc[adata.obs.index, set(adata.var.index).intersection(data.columns)]

    return adata


def getAssay(adata: anndata.AnnData, key: str, slot: str = "obsm") -> pd.DataFrame:
    """Retrieve a key named data from the specified slot.

    Args:
        adata: an AnnData object.
        key: the key name of the data to be retrieved. .
        slot: the slot of adata to be retrieved from. Defaults to "obsm".

    Returns:
        The data (in pd.DataFrame) that will be retrieved from adata.
    """

    if slot == "uns":
        data = adata.uns[key]
    elif slot == "obsm":
        data = adata.obsm[key]

    return data
