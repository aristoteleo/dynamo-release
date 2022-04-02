import ntpath
import os
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from anndata import read_h5ad, read_loom

from .dynamo_logger import LoggerManager, main_info, main_log_time


def download_data(url, file_path=None, dir="./data"):
    file_path = ntpath.basename(url) if file_path is None else file_path
    file_path = os.path.join(dir, file_path)
    main_info("Downloading data to " + file_path)

    if not os.path.exists(file_path):
        if not os.path.exists("./data/"):
            os.mkdir("data")

        # download the data
        urlretrieve(url, file_path, reporthook=LoggerManager.get_main_logger().request_report_hook)

    return file_path


def get_adata(url, filename=None):
    """Download example data to local folder.

    Parameters
    ----------
        url:
        filename

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
    """

    file_path = download_data(url, filename)
    if Path(file_path).suffixes[-1][1:] == "loom":
        adata = read_loom(filename=file_path)
    elif Path(file_path).suffixes[-1][1:] == "h5ad":
        adata = read_h5ad(filename=file_path)

    adata.var_names_make_unique()

    return adata


# add our toy sample data
def Gillespie():
    pass


def HL60():
    pass


def NASCseq():
    pass


def scSLAMseq():
    pass


def scifate():
    pass


def scNT_seq_neuron_splicing(
    url="https://www.dropbox.com/s/5wk3q2xhgqai2xq/neuron_splicing_4_11.h5ad?dl=1",
    filename="neuron_splicing.h5ad",
):
    """The neuron splicing data is from Qiu, et al (2020).
    This data consists of 44,021 genes across 13,476 cells.

    Returns
    -------
        Returns `adata` object
    """
    adata = get_adata(url, filename)

    return adata


def scNT_seq_neuron_labeling(
    url="https://www.dropbox.com/s/5wk3q2xhgqai2xq/neuron_labeling.h5ad?dl=1",
    filename="neuron_labeling.h5ad",
):
    """The neuron splicing data is from Qiu, et al (2020).
    This data consists of 37,007 genes across 3,060 cells.

    Returns
    -------
        Returns `adata` object
    """
    adata = get_adata(url, filename)

    return adata


def cite_seq():
    pass


def zebrafish(
    url="https://pitt.box.com/shared/static/w81022hta7lymss36i8m5ughppjjagqw.h5ad",
    filename="zebrafish.h5ad",
):
    """The zebrafish is from Saunders, et al (2019).
    This data consists of 16,940 genes across 4,181 cells.

    Returns
    -------
        Returns `adata` object
    """
    adata = get_adata(url, filename)

    return adata


def DentateGyrus(
    url="http://pklab.med.harvard.edu/velocyto/DentateGyrus/DentateGyrus.loom",
    filename=None,
):
    """The Dentate Gyrus dataset used in https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb.
    This data consists of 27,998 genes across 18,213 cells.

    Note this one http://pklab.med.harvard.edu/velocyto/DG1/10X43_1.loom: a subset of the above data.

    Returns
    -------
        Returns `adata` object
    """
    adata = get_adata(url, filename)

    return adata


def Haber(
    url="http://pklab.med.harvard.edu/velocyto/Haber_et_al/Haber_et_al.loom",
    filename=None,
):
    """The Haber dataset used in https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/Haber_et_al.ipynb
    This data consists of 27,998 genes across 7,216 cells.

    Returns
    -------
        Returns `adata` object
    """
    adata = get_adata(url, filename)
    urlretrieve(
        "http://pklab.med.harvard.edu/velocyto/Haber_et_al/goatools_cellcycle_genes.txt",
        "data/goatools_cellcycle_genes.txt",
    )
    cell_cycle_genes = open("data/goatools_cellcycle_genes.txt").read().split()
    adata.var.loc[:, "cell_cycle_genes"] = adata.var.index.isin(cell_cycle_genes)

    return adata


def hgForebrainGlutamatergic(
    url="http://pklab.med.harvard.edu/velocyto/hgForebrainGlut/hgForebrainGlut.loom",
    filename=None,
):
    """The hgForebrainGlutamatergic dataset used in https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/hgForebrainGlutamatergic.ipynb
    This data consists of 32,738 genes across 1,720 cells.

    Returns
    -------
        Returns `adata` object
    """
    adata = get_adata(url, filename)
    urlretrieve(
        "http://pklab.med.harvard.edu/velocyto/Haber_et_al/goatools_cellcycle_genes.txt",
        "data/goatools_cellcycle_genes.txt",
    )
    cell_cycle_genes = open("data/goatools_cellcycle_genes.txt").read().split()
    adata.var.loc[:, "cell_cycle_genes"] = adata.var.index.isin(cell_cycle_genes)

    return adata


def chromaffin(
    url="https://www.dropbox.com/s/awevuz836tlclvw/onefilepercell_A1_unique_and_others_J2CH1.loom?dl=1",
    filename="onefilepercell_A1_unique_and_others_J2CH1.loom",
):  #
    """The chromaffin dataset used in http://pklab.med.harvard.edu/velocyto/notebooks/R/chromaffin2.nb.html
    This data consists of 32,738 genes across 1,720 cells.

    Returns
    -------
        Returns `adata` object
    """

    adata = get_adata(url, filename)

    adata.var_names_make_unique()
    return adata


def BM(
    url="http://pklab.med.harvard.edu/velocyto/mouseBM/SCG71.loom",
    filename=None,
):
    """The BM dataset used in http://pklab.med.harvard.edu/velocyto/notebooks/R/SCG71.nb.html
    This data consists of 24,421genes across 6,667 cells.

    Returns
    -------
        Returns `adata` object
    """

    adata = get_adata(url, filename)

    return adata


def pancreatic_endocrinogenesis(
    url="https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad",
    filename=None,
):
    """Pancreatic endocrinogenesis. Data from scvelo

        Pancreatic epithelial and Ngn3-Venus fusion (NVF) cells during secondary transition / embryonic day 15.5.
        https://dev.biologists.org/content/146/12/dev173849

    Returns
    -------
        Returns `adata` object
    """

    adata = get_adata(url, filename)

    return adata


def DentateGyrus_scvelo(
    url="https://www.dropbox.com/s/3w1wzb0b68fhdsw/dentategyrus_scv.h5ad?dl=1",
    filename="dentategyrus_scv.h5ad",
):
    """The Dentate Gyrus dataset used in https://github.com/theislab/scvelo_notebooks/tree/master/data/DentateGyrus.
    This data consists of 13,913 genes across 2,930 cells.

    Note this dataset is the same processed dataset from the excellent scVelo package, which is a subset of the DentateGyrus dataset.

    Returns
    -------
        Returns `adata` object
    """
    adata = get_adata(url, filename)

    return adata


def scEU_seq_rpe1(
    url: str = "https://www.dropbox.com/s/25enev458c8egn7/rpe1.h5ad?dl=1",
    filename: str = "rpe1.h5ad",
):
    """
    Download rpe1 dataset from Battich, et al (2020) via Dropbox link.
    This data consists of 13,913 genes across 2,930 cells.

    Returns
    -------
        Returns `adata` object
    """
    main_info("Downloading scEU_seq data")
    adata = get_adata(url, filename)
    return adata


def scEU_seq_organoid(
    url: str = "https://www.dropbox.com/s/25enev458c8egn7/organoid.h5ad?dl=1",
    filename: str = "organoid.h5ad",
):
    """
    Download organoid dataset from Battich, et al (2020) via Dropbox link.
    This data consists of 9,157 genes across 3,831 cells.

    Returns
    -------
        Returns `adata` object
    """
    main_info("Downloading scEU_seq data")
    adata = get_adata(url, filename)
    return adata


def hematopoiesis(
    url: str = "https://pitt.box.com/shared/static/kyh3s4wrxdywupn9wk9r2j27vzlvk8vf.h5ad",
    # url: str = "https://pitt.box.com/shared/static/efqa8icu1m6d1ghfcc3s9tj0j91pky1h.h5ad", # v0: umap_ori version
    filename: str = "hematopoiesis.h5ad",
):
    """https://pitt.box.com/v/hematopoiesis-processed"""
    main_info("Downloading processed hematopoiesis adata")
    adata = get_adata(url, filename)
    return adata


def human_tfs(url="https://pitt.box.com/shared/static/spr7mi9rl2s7kgstrvrpidg138quuo4c.txt", filename="human_tfs.txt"):
    file_path = download_data(url, filename)
    tfs = pd.read_csv(file_path, sep="\t")
    return tfs


if __name__ == "__main__":
    DentateGyrus()
