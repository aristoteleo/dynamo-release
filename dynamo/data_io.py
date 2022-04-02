# Please import relevant packages in corresponding functions to avoid conflicts with dynamo's modules (e.g. dyn.pd.**)

from functools import reduce

from anndata import (
    AnnData,
    read,
    read_csv,
    read_excel,
    read_h5ad,
    read_hdf,
    read_loom,
    read_mtx,
    read_text,
    read_umi_tools,
    read_zarr,
)
from tqdm import tqdm

from .dynamo_logger import main_info


def convert2float(adata, columns, var=False):
    """This helper function can convert the category columns (undesiredly converted) when saving adata object into h5ad
    file back to float type."""

    columns = list(adata.var.columns.intersection(columns)) if var else list(adata.obs.columns.intersection(columns))
    if len(columns) == 0:
        raise ValueError(f"the columns {columns} you provided doesn't match with any columns from the adata object.")

    for i in columns:
        if i.startswith("use_for") or i == "pass_basic_filter":
            data = adata.var[i] if var else adata.obs[i]
            data[data == "None"] = None
            data = data.astype(bool)
            if var:
                adata.var[i] = data.copy()
            else:
                adata.obs[i] = data.copy()
        else:
            data = adata.var[i] if var else adata.obs[i]
            data[data == "None"] = None
            data = data.astype(str).astype(float)
            if var:
                adata.var[i] = data.copy()
            else:
                adata.obs[i] = data.copy()


def load_NASC_seq(dir, type="TPM", delimiter="_", colnames=None, dropna=False):
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

    import glob
    import os

    import numpy as np
    import pandas as pd
    from anndata import AnnData
    from scipy.sparse import csr_matrix

    if type == "TMM":
        delimiter = "_"
        tot_RNA = pd.read_csv(dir + "/rmse/RSEM.isoform.TMM.EXPR.matrix", sep="\t", index_col=0).T
        cells_raw = tot_RNA.index
        cells = [i.split(delimiter)[1] for i in tot_RNA.index]
        tot_RNA.index = cells
        pi_g = pd.read_csv(dir + "/outfiles/_mode.csv", index_col=0)
        pi_g.index = pd.Series(pi_g.index).str.split(delimiter, expand=True)[1].values
        print(pi_g.head(2))

        new_RNA, old_RNA = (
            pd.DataFrame(0.0, columns=tot_RNA.columns, index=cells),
            pd.DataFrame(0.0, columns=tot_RNA.columns, index=cells),
        )
        valid_index, valid_columns = (
            tot_RNA.index.intersection(pi_g.index),
            tot_RNA.columns.intersection(pi_g.columns),
        )
        new_, old_ = (
            tot_RNA.loc[valid_index, valid_columns] * pi_g.loc[valid_index, valid_columns],
            tot_RNA.loc[valid_index, valid_columns] * (1 - pi_g.loc[valid_index, valid_columns]),
        )
        (
            new_RNA.loc[new_.index, new_.columns],
            old_RNA.loc[new_.index, new_.columns],
        ) = (new_.values, old_.values)

    elif type in ["TPM", "FPKM"]:
        files = glob.glob(dir + "/rmse/*genes.results")
        tot_RNA = None
        cells_raw, cells = None, None

        for f in tqdm(files, desc=f"reading rmse output files:"):
            tmp = pd.read_csv(f, index_col=0, sep="\t")

            if tot_RNA is None:
                tot_RNA = tmp.loc[:, [type]]
                cells_raw = [os.path.basename(f)]
                cells = [cells_raw[-1].split(delimiter)[1]]
            else:
                tot_RNA = pd.merge(
                    tot_RNA,
                    tmp.loc[:, [type]],
                    left_index=True,
                    right_index=True,
                    how="outer",
                )
                cells_raw.append(os.path.basename(f))

                cells.append(cells_raw[-1].split(delimiter)[1])
        tot_RNA.columns, tot_RNA.index = cells, list(tot_RNA.index)

        pi_g = pd.read_csv(dir + "/outfiles/_mode.csv", index_col=0)
        pi_g.index = pd.Series(pi_g.index).str.split(delimiter, expand=True)[1].values

        new_RNA, old_RNA = (
            pd.DataFrame(0.0, columns=tot_RNA.index, index=cells),
            pd.DataFrame(0.0, columns=tot_RNA.index, index=cells),
        )
        new_, old_ = (
            tot_RNA.loc[pi_g.columns, pi_g.index].T * pi_g,
            tot_RNA.loc[pi_g.columns, pi_g.index].T * (1 - pi_g),
        )
        (
            new_RNA.loc[new_.index, new_.columns],
            old_RNA.loc[new_.index, new_.columns],
        ) = (new_.values, old_.values)

        tot_RNA = tot_RNA.T
        if colnames is not None:
            colnames = ["plate", "well", "sample"]
    elif type == "Reads":
        included_extensions = ["newTable.csv", "oldTable.csv", "readCounts.csv"]
        file_names = [
            fn for fn in os.listdir(dir + "/outfiles/") if any(fn.endswith(ext) for ext in included_extensions)
        ]

        if len(file_names) == 3:
            new_RNA = pd.read_csv(dir + "/outfiles/" + file_names[0], index_col=0, delimiter=",")
            old_RNA = pd.read_csv(dir + "/outfiles/" + file_names[1], index_col=0, delimiter=",")
            tot_RNA = pd.read_csv(dir + "/outfiles/" + file_names[2], index_col=0, delimiter=",")
        else:
            raise Exception(
                "The directory you provided doesn't contain files end with newTable.csv, oldcounts.csv and \
            readcounts.csv that returned from NASC-seq pipeline."
            )

        cells_raw = new_RNA.index
    else:
        raise ValueError(
            f"The data type {type} requested is not supported. Available data types include:"
            f"{'TPM', 'FPKM', 'Reads'}"
        )

    split_df = pd.Series(cells_raw).str.split(delimiter, expand=True)
    split_df.index = split_df.iloc[:, 1].values
    if colnames is not None:
        split_df.columns = colnames

    if dropna:
        valid_ids = np.isnan((new_RNA + old_RNA + tot_RNA).sum(0, skipna=False))
        new_RNA, old_RNA, tot_RNA = (
            new_RNA.iloc[:, valid_ids],
            old_RNA.iloc[:, valid_ids],
            tot_RNA.iloc[:, valid_ids],
        )
    else:
        new_RNA.fillna(0, inplace=True)
        old_RNA.fillna(0, inplace=True)
        tot_RNA.fillna(0, inplace=True)

    adata = AnnData(
        csr_matrix(tot_RNA.values),
        var=pd.DataFrame({"gene_name": tot_RNA.columns}, index=tot_RNA.columns),
        obs=split_df,
        layers=dict(new=csr_matrix(new_RNA.values), total=csr_matrix(tot_RNA.values)),
    )

    adata = adata[:, adata.X.sum(0).A > 0]
    adata.uns["raw_data"] = True


def aggregate_adata(file_list: list) -> AnnData:
    """Aggregate gene expression from adata.X or layer for a list of adata based on the same cell and gene names.

    Parameters
    ----------
        file_list:
            A list of strings specifies the link to the anndata object.

    Returns
    -------
        agg_adata:
            Aggregated adata object.
    """

    import anndata
    from anndata import AnnData

    if type(file_list[0]) == anndata._core.anndata.AnnData:
        adata_list = file_list
    elif type(file_list[0]) == str:
        adata_list = [anndata.read(i) for i in file_list]

    valid_cells = reduce(lambda a, b: a.intersection(b), [i.obs_names for i in adata_list])
    valid_genes = reduce(lambda a, b: a.intersection(b), [i.var_names for i in adata_list])

    if len(valid_cells) == 0 or len(valid_genes) == 0:
        raise Exception(
            f"we don't find any gene or cell names shared across different adata objects." f"Please check your data. "
        )

    layer_dict = {}
    for i in adata_list[0].layers.keys():
        layer_dict[i] = reduce(
            lambda a, b: a + b,
            [adata[valid_cells, valid_genes].layers[i] for adata in adata_list],
        )

    agg_adata = anndata.AnnData(
        X=reduce(
            lambda a, b: a + b,
            [adata[valid_cells, valid_genes].X for adata in adata_list],
        ),
        obs=adata_list[0][valid_cells, valid_genes].obs,
        var=adata_list[0][valid_cells, valid_genes].var,
        layers=layer_dict,
    )

    return agg_adata


def cleanup(adata, del_prediction=False, del_2nd_moments=False):
    """clean up adata before saving it to a file"""

    if "pca_fit" in adata.uns_keys():
        adata.uns["pca_fit"] = None
    if "velocyto_SVR" in adata.uns_keys():
        adata.uns["velocyto_SVR"]["SVR"] = None
    if "umap_fit" in adata.uns_keys():
        adata.uns["umap_fit"]["fit"] = None
    if "velocity_pca_fit" in adata.uns_keys():
        adata.uns["velocity_pca_fit"] = None
    if "kmc" in adata.uns_keys():
        adata.uns["kmc"] = None
    if "kinetics_heatmap" in adata.uns_keys():
        adata.uns.pop("kinetics_heatmap")
    if "hdbscan" in adata.uns_keys():
        adata.uns.pop("hdbscan")

    VF_keys = [i if i.startswith("VecFld") else None for i in adata.uns_keys()]
    for i in VF_keys:
        if i is not None and "VecFld2D" in adata.uns[i].keys():
            del adata.uns[i]["VecFld2D"]

    fate_keys = [i if i.startswith("fate") else None for i in adata.uns_keys()]
    for i in fate_keys:
        if i is not None:
            if adata.uns[i]["init_cells"] is not None:
                adata.uns[i]["init_cells"] = list(adata.uns[i]["init_cells"])
            if "prediction" in adata.uns[i].keys():
                if del_prediction:
                    del adata.uns[i]["prediction"]
            if "VecFld_true" in adata.uns[i].keys():
                if adata.uns[i]["VecFld_true"] is not None:
                    del adata.uns[i]["VecFld_true"]

    if del_2nd_moments:
        from .tools.utils import remove_2nd_moments

        remove_2nd_moments(adata)

    return adata


def export_rank_xlsx(adata, path="rank_info.xlsx", ext="excel", rank_prefix="rank"):
    import pandas as pd

    with pd.ExcelWriter(path) as writer:
        for key in adata.uns.keys():
            if key[: len(rank_prefix)] == rank_prefix:
                main_info("saving sheet: " + str(key))
                adata.uns[key].to_excel(writer, sheet_name=str(key))
