from anndata import read_h5ad, read_loom
from urllib.request import urlretrieve
from pathlib import Path
import os
import ntpath


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

    filename = ntpath.basename(url) if filename is None else filename

    filename = "./data/" + filename
    if not os.path.exists(filename):
        if not os.path.exists("./data/"):
            os.mkdir("data")

        urlretrieve(url, filename)  # download the data

    if Path(filename).suffixes[-1][1:] == "loom":
        adata = read_loom(filename=filename)
    elif Path(filename).suffixes[-1][1:] == "h5ad":
        adata = read_h5ad(filename=filename)

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


def scNT_seq():
    pass


def cite_seq():
    pass


def DentateGyrus(
    url="http://pklab.med.harvard.edu/velocyto/DentateGyrus/DentateGyrus.loom",
    filename=None,
):
    """The Dentate Gyrus dataset used in https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb.
    This data consists of 27, 998 genes across 18, 213 cells.

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
    This data consists of 27, 998 genes across 7, 216 cells.

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
    This data consists of 32, 738 genes across 1, 720 cells.

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
    This data consists of 32, 738 genes across 1, 720 cells.

    Returns
    -------
        Returns `adata` object
    """

    adata = get_adata(url, filename)

    adata.var_names_make_unique()
    return adata


def BM(url="http://pklab.med.harvard.edu/velocyto/mouseBM/SCG71.loom", filename=None):
    """The BM dataset used in http://pklab.med.harvard.edu/velocyto/notebooks/R/SCG71.nb.html
    This data consists of 24, 421genes across 6, 667 cells.

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
    This data consists of 13, 913 genes across 2, 930 cells.

    Note this dataset is the same processed dataset from the excellent scVelo package, which is a subset of the DentateGyrus dataset.

    Returns
    -------
        Returns `adata` object
    """
    adata = get_adata(url, filename)

    return adata


if __name__ == "__main__":
    DentateGyrus()
