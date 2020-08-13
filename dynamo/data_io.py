from tqdm import tqdm
import numpy as np
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


def convert2float(adata, columns, var=False):
    """This helper function can convert the category columns (undesiredly converted) when saving adata object into h5ad
    file back to float type."""

    columns = list(adata.var.columns.intersection(columns)) if var else list(adata.obs.columns.intersection(columns))
    if len(columns) == 0:
        raise ValueError(f"the columns {columns} you provided doesn't match with any columns from the adata object.")

    for i in columns:
        data = adata.var[i] if var else adata.obs[i]
        data[data == 'None'] = None
        data = data.astype(float)
        if var:
            adata.var[i] = data.copy()
        else:
            adata.obs[i] = data.copy()


def load_NASC_seq(dir, type='TPM', delimiter="_", colnames=None, dropna=False):
    """Function to create an anndata object from NASC-seq pipeline

    Parameters
    ----------
        dir: `str`
            The directory that points to the NASC-seq pipeline analysis folder (something like /Experimentdir).
        type: `str` (default: `TPM`)
            The data type that will be used as the gene expression. One of `{'TPM', 'FPKM', 'Reads'}`.
        delimiter: `str` (default: `_`)
            delimiter pattern for splitting the cells names (columns of each count table)
        colnames: `list` or none
            The list of column names after splitting the cell names.
        dropna: `bool`
            Whether to drop all genes that have any np.nan values across all cells. If not, all na values will be filled
            as 0.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object with the `new` and `total` layers.
    """

    import os
    from anndata import AnnData
    import glob
    from scipy.sparse import csr_matrix
    import pandas as pd, numpy as np

    if type == 'TMM':
        delimiter = '_'
        tot_RNA = pd.read_csv(dir + '/rmse/RSEM.isoform.TMM.EXPR.matrix', sep='\t', index_col=0).T
        cells_raw = tot_RNA.index
        cells = [i.split(delimiter)[1] for i in tot_RNA.index]
        tot_RNA.index = cells
        pi_g = pd.read_csv(dir + '/outfiles/_mode.csv', index_col=0)
        pi_g.index = pd.Series(pi_g.index).str.split(delimiter, expand=True)[1].values
        print(pi_g.head(2))

        new_RNA, old_RNA = pd.DataFrame(0., columns=tot_RNA.columns, index=cells), \
                           pd.DataFrame(0., columns=tot_RNA.columns, index=cells)
        valid_index, valid_columns = tot_RNA.index.intersection(pi_g.index), tot_RNA.columns.intersection(pi_g.columns)
        new_, old_ = tot_RNA.loc[valid_index, valid_columns] * pi_g.loc[valid_index, valid_columns], \
                     tot_RNA.loc[valid_index, valid_columns] * (1 - pi_g.loc[valid_index, valid_columns])
        new_RNA.loc[new_.index, new_.columns], old_RNA.loc[new_.index, new_.columns] = new_.values, old_.values

    elif type in ['TPM', 'FPKM']:
        files = glob.glob(dir + '/rmse/*genes.results')
        tot_RNA = None
        cells_raw, cells = None, None

        for f in tqdm(files, desc=f'reading rmse output files:'):
            tmp = pd.read_csv(f, index_col=0, sep='\t')

            if tot_RNA is None:
                tot_RNA = tmp.loc[:, [type]]
                cells_raw = [os.path.basename(f)]
                cells = [cells_raw[-1].split(delimiter)[1]]
            else:
                tot_RNA = pd.merge(tot_RNA, tmp.loc[:, [type]], left_index=True, right_index=True, how='outer')
                cells_raw.append(os.path.basename(f))

                cells.append(cells_raw[-1].split(delimiter)[1])
        tot_RNA.columns, tot_RNA.index = cells, list(tot_RNA.index)

        pi_g = pd.read_csv(dir + '/outfiles/_mode.csv', index_col=0)
        pi_g.index = pd.Series(pi_g.index).str.split(delimiter, expand=True)[1].values

        new_RNA, old_RNA = pd.DataFrame(0., columns=tot_RNA.index, index=cells), \
                           pd.DataFrame(0., columns=tot_RNA.index, index=cells)
        new_, old_ = tot_RNA.loc[pi_g.columns, pi_g.index].T * pi_g, \
                     tot_RNA.loc[pi_g.columns, pi_g.index].T * (1 - pi_g)
        new_RNA.loc[new_.index, new_.columns], old_RNA.loc[new_.index, new_.columns] = new_.values, old_.values

        tot_RNA = tot_RNA.T
        if colnames is not None: colnames = ['plate', 'well', 'sample']
    elif type == 'Reads':
        included_extensions = ["newTable.csv", "oldTable.csv", "readCounts.csv"]
        file_names = [
            fn
            for fn in os.listdir(dir + '/outfiles/')
            if any(fn.endswith(ext) for ext in included_extensions)
        ]

        if len(file_names) == 3:
            new_RNA = pd.read_csv(dir + '/outfiles/' + file_names[0], index_col=0, delimiter=",")
            old_RNA = pd.read_csv(dir + '/outfiles/' + file_names[1], index_col=0, delimiter=",")
            tot_RNA = pd.read_csv(dir + '/outfiles/' + file_names[2], index_col=0, delimiter=",")
        else:
            raise Exception(
                "The directory you provided doesn't contain files end with newTable.csv, oldcounts.csv and \
            readcounts.csv that returned from NASC-seq pipeline."
            )

        cells_raw = new_RNA.index
    else:
        raise ValueError(f"The data type {type} requested is not supported. Available data types include:"
                         f"{'TPM', 'FPKM', 'Reads'}")

    split_df = pd.Series(cells_raw).str.split(delimiter, expand=True)
    split_df.index = split_df.iloc[:, 1].values
    if colnames is not None: split_df.columns = colnames

    if dropna:
        valid_ids = np.isnan((new_RNA + old_RNA + tot_RNA).sum(0, skipna=False))
        new_RNA, old_RNA, tot_RNA = new_RNA.iloc[:, valid_ids], old_RNA.iloc[:, valid_ids], tot_RNA.iloc[:, valid_ids]
    else:
        new_RNA.fillna(0, inplace=True)
        old_RNA.fillna(0, inplace=True)
        tot_RNA.fillna(0, inplace=True)

    adata = AnnData(csr_matrix(tot_RNA.values),
                    var=pd.DataFrame({"gene_name": tot_RNA.columns}, index=tot_RNA.columns),
                    obs=split_df,
                    layers=dict(new=csr_matrix(new_RNA.values), total=csr_matrix(tot_RNA.values)),
                    )

    adata = adata[:, adata.X.sum(0).A > 0]
    adata.uns['raw_data'] = True


def cleanup(adata, del_prediction=False):
    """clean up adata before saving it to a file"""

    if 'pca_fit' in adata.uns_keys(): adata.uns['pca_fit'] = None
    if 'velocyto_SVR' in adata.uns_keys(): adata.uns['velocyto_SVR']['SVR'] = None
    if 'umap_fit' in adata.uns_keys(): adata.uns['umap_fit']['fit'] = None
    if 'velocity_pca_fit' in adata.uns_keys(): adata.uns['velocity_pca_fit'] = None
    if 'kmc' in adata.uns_keys(): adata.uns['kmc'] = None

    VF_keys = [i if i.startswith('VecFld') else None for i in adata.uns_keys()]
    for i in VF_keys:
        if i is not None and 'VecFld2D' in adata.uns[i].keys():
            del adata.uns[i]['VecFld2D']

    fate_keys = [i if i.startswith('fate') else None for i in adata.uns_keys()]
    for i in fate_keys:
        if i is not None:
            if adata.uns[i]['init_cells'] is not None:
                adata.uns[i]['init_cells'] = list(adata.uns[i]['init_cells'])
            if 'prediction' in adata.uns[i].keys():
                if del_prediction: del adata.uns[i]['prediction']
            if 'VecFld_true' in adata.uns[i].keys():
                if adata.uns[i]['VecFld_true'] is not None: del adata.uns[i]['VecFld_true']

    return adata
