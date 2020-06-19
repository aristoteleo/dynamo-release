from anndata import (
    read,
    read_loom,
    read_csv,
    read_excel,
    read_h5ad,
    read_hdf,
    read_mtx,
    read_umi_tools,
    read_zarr,
    read_text,
)


def load_NASC_seq(dir, delimiter="_", colnames=None):
    """Function to create an anndata object from NASC-seq pipeline

    Parameters
    ----------
        dir: `str`
            The directory that points to the NASC-seq pipeline result output folder (something like /Experimentdir/outfiles).
        delimiter: `str` (default: `_`)
            delimiter pattern for splitting the cells names (columns of each count table)
        colnames: `list` or none
            The list of column names after splitting the cell names.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object
    """

    import os
    from anndata import AnnData
    import pandas as pd

    included_extensions = ["newTable.csv", "oldTable.csv", "readCounts.csv"]
    file_names = [
        fn
        for fn in os.listdir(dir)
        if any(fn.endswith(ext) for ext in included_extensions)
    ]
    if len(file_names) == 3:
        new_RNA = pd.read_csv(dir + file_names[0], index_col=0, delimiter=",")
        old_RNA = pd.read_csv(dir + file_names[1], index_col=0, delimiter=",")
        tot_RNA = pd.read_csv(dir + file_names[2], index_col=0, delimiter=",")
    else:
        raise Exception(
            "The directory you provided doesn't contain files end with newTable.csv, oldcounts.csv and \
        readcounts.csv that returned from NASC-seq pipeline."
        )

    new_RNA.fillna(0, inplace=True)
    old_RNA.fillna(0, inplace=True)
    tot_RNA.fillna(0, inplace=True)

    col_num = len(colnames) if colnames is not None else 2
    split_array = [
        new_RNA.columns.str.split("_", n=col_num)[i]
        for i in range(len(new_RNA.columns.str.split(delimiter, n=col_num)))
    ]
    split_df = pd.DataFrame(split_array, columns=colnames)

    adata = AnnData(
        tot_RNA.values.T,
        var=pd.DataFrame(gene_name=tot_RNA.index, index=tot_RNA.index),
        obs=split_df,
        layers=dict(new=new_RNA.values.T, total=tot_RNA.values.T),
    )

    return adata


def cleanup(adata):
    """clean up adata before saving it to a file"""

    adata.uns['pca_fit'] = None
    adata.uns['velocyto_SVR']['SVR'] = None
    adata.uns['umap_fit']['fit'] = None
    adata.uns['velocity_pca_fit'] = None
    adata.uns['kmc'] = None
    VF_keys = [i if i.startswith('VecFld') else None for i in adata.uns_keys()]
    for i in VF_keys:
        if i is not None:
            del adata.uns[i]

    return adata
