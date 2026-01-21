import ntpath
import os
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import os
import requests
from typing import Optional
from tqdm import tqdm

import pandas as pd
from anndata import AnnData, read_h5ad, read_loom

from .dynamo_logger import LoggerManager, main_info, main_log_time

DATA_DOWNLOAD_FIGSHARE_DICT = {
    "neuron_splicing": "https://figshare.com/ndownloader/files/47439605",
    "neuron_labeling": "https://figshare.com/ndownloader/files/47439629",
    "zebrafish": "https://figshare.com/ndownloader/files/47420257",
    "bone_marrow": "https://figshare.com/ndownloader/files/35826944",
    "human_tfs": "https://figshare.com/ndownloader/files/47439617",
    "onefilepercell_A1_unique_and_others_J2CH1": "https://figshare.com/ndownloader/files/47439620",
    "10X_multiome_mouse_brain": "https://figshare.com/ndownloader/files/54153947",
    "cell_annotations": "https://figshare.com/ndownloader/files/54154376",
    "dentategyrus_scv": "https://figshare.com/ndownloader/files/47439623",
    "hematopoiesis_raw": "https://figshare.com/ndownloader/files/47439626",
    "rpe1": "https://figshare.com/ndownloader/files/47439641",
    "organoid": "https://figshare.com/ndownloader/files/47439632",
    "hematopoiesis": "https://figshare.com/ndownloader/files/47439635",
}

DATA_DOWNLOAD_STANFORD_DICT = {
    "neuron_splicing": "https://stacks.stanford.edu/file/sh696dv4420/neuron_splicing.h5ad",
    "neuron_labeling": "https://stacks.stanford.edu/file/sh696dv4420/neuron_labeling.h5ad",
    "zebrafish": "https://stacks.stanford.edu/file/sh696dv4420/zebrafish.h5ad",
    "bone_marrow": "https://stacks.stanford.edu/file/sh696dv4420/setty_bone_marrow.h5ad",
    "human_tfs": "https://stacks.stanford.edu/file/sh696dv4420/human_tfs.txt",
    "onefilepercell_A1_unique_and_others_J2CH1": "https://stacks.stanford.edu/file/sh696dv4420/onefilepercell_A1_unique_and_others_J2CH1.loom",
    "10X_multiome_mouse_brain": "https://stacks.stanford.edu/file/sh696dv4420/10X_multiome_mouse_brain.loom",
    "cell_annotations": "https://stacks.stanford.edu/file/sh696dv4420/cell_annotations.tsv",
    "dentategyrus_scv": "https://stacks.stanford.edu/file/sh696dv4420/dentategyrus_scv.h5ad",
    "hematopoiesis_raw": "https://stacks.stanford.edu/file/sh696dv4420/hematopoiesis_raw.h5ad",
    "rpe1": "https://stacks.stanford.edu/file/sh696dv4420/rpe1.h5ad",
    "organoid": "https://stacks.stanford.edu/file/sh696dv4420/organoid.h5ad",
    "hematopoiesis": "https://stacks.stanford.edu/file/sh696dv4420/hematopoiesis.h5ad",
}

FIGSHARE_URL_TO_KEY = {value: key for key, value in DATA_DOWNLOAD_FIGSHARE_DICT.items()}


def _get_stanford_url(figshare_url: str) -> Optional[str]:
    key = FIGSHARE_URL_TO_KEY.get(figshare_url)
    if not key:
        return None
    stanford_url = DATA_DOWNLOAD_STANFORD_DICT.get(key, "")
    return stanford_url if stanford_url else None



def download_data(url: str, file_path: Optional[str] = None, dir: str = "./data") -> str:
    """Download example data to local folder."""
    file_path = ntpath.basename(url) if file_path is None else file_path
    file_path = os.path.join(dir, file_path)
    main_info("Downloading data to " + file_path)

    if not os.path.exists(file_path):
        if not os.path.exists("./data/"):
            os.mkdir("data")

        # download the data
        urlretrieve(url, file_path, reporthook=LoggerManager.get_main_logger().request_report_hook)
    else:
        main_info("File " + file_path + " already exists.")

    return file_path


def get_adata(url: str, filename: Optional[str] = None) -> Optional[AnnData]:
    """Download example data to local folder.

    Args:
        url: the url of the data.
        filename: the name of the file to be saved.

    Returns:
        An Annodata object.
    """

    try:
        file_path = None
        stanford_url = _get_stanford_url(url)
        if stanford_url:
            try:
                file_path = download_data_requests(stanford_url, filename)
            except Exception as e:
                main_info(f"Stanford download failed, fallback to figshare. Error: {e}")
                file_path = None
        if file_path is None:
            file_path = download_data(url, filename)
        if Path(file_path).suffixes[-1][1:] == "loom":
            adata = read_loom(filename=file_path)
        elif Path(file_path).suffixes[-1][1:] == "h5ad":
            adata = read_h5ad(filename=file_path)
        else:
            main_info("REPORT THIS: Unknown filetype (" + file_path + ")")
            return None

        adata.var_names_make_unique()
    except OSError:
        # Usually occurs when download is stopped before completion then attempted again.
        if filename is not None:
            file_path_to_remove = os.path.join('./data', filename)
        else:
            file_path_to_remove = file_path  # Use the file_path that was returned from download_data
        main_info("Corrupted file. Deleting " + file_path_to_remove + " then redownloading...")
        # Half-downloaded file cannot be read due to corruption so it's better to delete it.
        # Potential issue: user have a file with duplicate name but is not sample data (this will overwrite file).
        try:
            os.remove(file_path_to_remove)
        except:
            pass
        adata = get_adata(url, filename)
    except Exception as e:
        main_info("REPORT THIS: " + e)
        adata = None

    return adata


# add our toy sample data
def Gillespie():
    #TODO: add data here
    pass


def HL60():
    #TODO: add data here
    pass


def NASCseq():
    #TODO: add data here
    pass


def scSLAMseq():
    #TODO: add data here
    pass


def scifate():
    #TODO: add data here
    pass


def scNT_seq_neuron_splicing(
    url: str = "https://figshare.com/ndownloader/files/47439605",
    filename: str = "neuron_splicing.h5ad",
) -> AnnData:
    """The neuron splicing data is from Qiu, et al (2020).

    This data consists of 44,021 genes across 13,476 cells.
    """
    adata = get_adata(url, filename)

    return adata


def scNT_seq_neuron_labeling(
    url: str = "https://figshare.com/ndownloader/files/47439629",
    filename: str = "neuron_labeling.h5ad",
) -> AnnData:
    """The neuron splicing data is from Qiu, et al (2020).

    This data consists of 24, 078 genes across 3,060 cells.
    """
    adata = get_adata(url, filename)

    return adata


def cite_seq():
    pass


def zebrafish(
    url: str = "https://figshare.com/ndownloader/files/47420257",
    filename: str = "zebrafish.h5ad",
) -> AnnData:
    """The zebrafish is from Saunders, et al (2019).

    This data consists of 16,940 genes across 4,181 cells.
    """
    adata = get_adata(url, filename)

    return adata


def DentateGyrus(
    url: str = "http://pklab.med.harvard.edu/velocyto/DentateGyrus/DentateGyrus.loom",
    filename: Optional[str] = None,
) -> AnnData:
    """The Dentate Gyrus dataset used in https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb.

    This data consists of 27,998 genes across 18,213 cells.
    Note this one http://pklab.med.harvard.edu/velocyto/DG1/10X43_1.loom: a subset of the above data.
    """
    adata = get_adata(url, filename)

    return adata

def bone_marrow(
    url: str = "https://figshare.com/ndownloader/files/35826944",
    filename: str = "bone_marrow.h5ad",
) -> AnnData:
    """The bone marrow dataset used in

    This data consists of 27,876 genes across 5,780 cells.
    """
    adata = get_adata(url, filename)

    return adata

def Haber(
    url: str = "http://pklab.med.harvard.edu/velocyto/Haber_et_al/Haber_et_al.loom",
    filename: Optional[str] = None,
) -> AnnData:
    """The Haber dataset used in https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/Haber_et_al.ipynb

    This data consists of 27,998 genes across 7,216 cells.
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
    url: str = "http://pklab.med.harvard.edu/velocyto/hgForebrainGlut/hgForebrainGlut.loom",
    filename: Optional[str] = None,
) -> AnnData:
    """The hgForebrainGlutamatergic dataset used in https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/hgForebrainGlutamatergic.ipynb

    This data consists of 32,738 genes across 1,720 cells.
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
    url: str = "https://figshare.com/ndownloader/files/47439620",
    filename: str = "onefilepercell_A1_unique_and_others_J2CH1.loom",
) -> AnnData:  #
    """The chromaffin dataset used in http://pklab.med.harvard.edu/velocyto/notebooks/R/chromaffin2.nb.html

    This data consists of 32,738 genes across 1,720 cells.
    """

    adata = get_adata(url, filename)

    adata.var_names_make_unique()
    return adata


def BM(
    url: str = "http://pklab.med.harvard.edu/velocyto/mouseBM/SCG71.loom",
    filename: Optional[str] = None,
) -> AnnData:
    """The BM dataset used in http://pklab.med.harvard.edu/velocyto/notebooks/R/SCG71.nb.html

    This data consists of 24,421genes across 6,667 cells.
    """

    adata = get_adata(url, filename)

    return adata


def pancreatic_endocrinogenesis(
    url: str = "https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad",
    filename: Optional[str] = None,
) -> AnnData:
    """Pancreatic endocrinogenesis. Data from scvelo.

    Pancreatic epithelial and Ngn3-Venus fusion (NVF) cells during secondary transition / embryonic day 15.5.
    https://dev.biologists.org/content/146/12/dev173849
    """

    adata = get_adata(url, filename)

    return adata


def DentateGyrus_scvelo(
    url: str = "https://figshare.com/ndownloader/files/47439623",
    filename: str = "dentategyrus_scv.h5ad",
) -> AnnData:
    """The Dentate Gyrus dataset used in https://github.com/theislab/scvelo_notebooks/tree/master/data/DentateGyrus.

    This data consists of 13,913 genes across 2,930 cells. Note this dataset is the same processed dataset from the
    excellent scVelo package, which is a subset of the DentateGyrus dataset.
    """
    adata = get_adata(url, filename)

    return adata


def scEU_seq_rpe1(
    url: str = "https://figshare.com/ndownloader/files/47439641",
    filename: str = "rpe1.h5ad",
):
    """Download rpe1 dataset from Battich, et al (2020) via a figshare link.

    This data consists of 13,913 genes across 2,930 cells.
    """
    main_info("Downloading scEU_seq data")
    adata = get_adata(url, filename)
    return adata


def scEU_seq_organoid(
    url: str = "https://figshare.com/ndownloader/files/47439632",
    filename: str = "organoid.h5ad",
):
    """Download organoid dataset from Battich, et al (2020) via a figshare link.

    This data consists of 9,157 genes across 3,831 cells.
    """
    main_info("Downloading scEU_seq data")
    adata = get_adata(url, filename)
    return adata


def hematopoiesis(
    url: str = "https://figshare.com/ndownloader/files/47439635",
    # url: str = "https://pitt.box.com/shared/static/kyh3s4wrxdywupn9wk9r2j27vzlvk8vf.h5ad", # with box
    # url: str = "https://pitt.box.com/shared/static/efqa8icu1m6d1ghfcc3s9tj0j91pky1h.h5ad", # v0: umap_ori version
    filename: str = "hematopoiesis.h5ad",
) -> AnnData:
    """Processed dataset originally from https://pitt.box.com/v/hematopoiesis-processed."""
    main_info("Downloading processed hematopoiesis adata")
    adata = get_adata(url, filename)
    return adata



def download_data_requests(url: str, file_path: Optional[str] = None, dir: str = "./data") -> str:
    """Download data with headers to bypass 403 errors."""
    if not os.path.exists(dir):
        os.makedirs(dir)

    file_name = os.path.basename(url) if file_path is None else file_path
    file_path = os.path.join(dir, file_name)

    if os.path.exists(file_path):
        main_info(f"File {file_path} already exists.")
        return file_path

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://cf.10xgenomics.com/",
    }

    logger = LoggerManager.get_main_logger()

    def _download_with_requests(download_url: str) -> None:
        main_info(f"Downloading data to {file_path}...")
        with requests.get(download_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('Content-Length', 0))
            chunk_size = 8192
            downloaded_size = 0
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        # Pass downloaded_size as bn and 1 as rs, so rs*bn = downloaded_size
                        logger.request_report_hook(downloaded_size, 1, total_size)

    stanford_url = _get_stanford_url(url)
    if stanford_url:
        try:
            _download_with_requests(stanford_url)
            return file_path
        except Exception as e:
            main_info(f"Stanford download failed, fallback to figshare. Error: {e}")

    try:
        _download_with_requests(url)
    except Exception as e:
        main_info(f"Download failed: {e}")
        raise

    return file_path


def multi_brain_5k(
        
):
    """Processed dataset originally from https://pitt.box.com/v/hematopoiesis-processed."""
    main_info("Downloading raw Fresh Embryonic E18 Mouse Brain (5k)\nEpi Multiome ATAC + Gene Expression dataset adata")

    h5_url='https://cf.10xgenomics.com/samples/cell-arc/1.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_filtered_feature_bc_matrix.h5'
    fragment_url='https://cf.10xgenomics.com/samples/cell-arc/1.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz'
    fragment_tbi_url='https://cf.10xgenomics.com/samples/cell-arc/1.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz.tbi'
    peak_annotation_url='https://cf.10xgenomics.com/samples/cell-arc/1.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_atac_peak_annotation.tsv'
    velocyto_url='https://figshare.com/ndownloader/files/54153947'
    anontation_url='https://figshare.com/ndownloader/files/54154376'

    

    h5_path = download_data_requests(h5_url, 'filtered_feature_bc_matrix.h5', dir='./data/multi_brain_5k')
    fragment_path = download_data_requests(fragment_url, 'fragments.tsv.gz', dir='./data/multi_brain_5k')
    fragment_tbi_path = download_data_requests(fragment_tbi_url, 'fragments.tsv.gz.tbi', dir='./data/multi_brain_5k')
    peak_annotation_path = download_data_requests(peak_annotation_url, 'peak_annotation.tsv', dir='./data/multi_brain_5k')
    velocyto_path = download_data_requests(velocyto_url, '10X_multiome_mouse_brain.loom', dir='./data/multi_brain_5k/velocyto')
    annotation_path = download_data_requests(anontation_url, 'cell_annotations.tsv', dir='./data/multi_brain_5k')

    analysis_url='https://cf.10xgenomics.com/samples/cell-arc/1.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_analysis.tar.gz'
    analysis_path = download_data_requests(analysis_url, 'e18_mouse_brain_fresh_5k_analysis.tar.gz', dir='./data/multi_brain_5k')
    # Extract the tar.gz file
    import tarfile
    with tarfile.open(analysis_path, "r:gz") as tar:
        tar.extractall(path='./data/multi_brain_5k/')
    # Remove the tar.gz file after extraction
    os.remove(analysis_path)

    from .multi import read_10x_multiome_h5
    mdata=read_10x_multiome_h5(multiome_base_path='./data/multi_brain_5k',
                                                       rna_splicing_loom='velocyto/10X_multiome_mouse_brain.loom',
                                                      cellranger_path_structure=False)
    cell_annot = pd.read_csv('./data/multi_brain_5k/cell_annotations.tsv', sep='\t', index_col=0)
    cell_annot.index=[i.split('-')[0] for i in cell_annot.index]
    ret_index=list(set(cell_annot.index) & set(mdata.obs.index))
    cell_annot=cell_annot.loc[ret_index]
    mdata.update()
    mdata = mdata[ret_index]
    mdata['rna'].obs['celltype'] = cell_annot['celltype'].tolist()
    return mdata



def hematopoiesis_raw(
    url: str = "https://figshare.com/ndownloader/files/47439626",
    # url: str = "https://pitt.box.com/shared/static/bv7q0kgxjncc5uoget5wvmi700xwntje.h5ad", # with box
    filename: str = "hematopoiesis_raw.h5ad",
) -> AnnData:
    """Processed dataset originally from https://pitt.box.com/v/hematopoiesis-processed."""
    main_info("Downloading raw hematopoiesis adata")
    adata = get_adata(url, filename)
    return adata


def human_tfs(
    url: str = "https://figshare.com/ndownloader/files/47439617",
    filename: str = "human_tfs.txt",
) -> pd.DataFrame:
    """Download human transcription factors."""
    file_path = download_data_requests(url, filename)
    tfs = pd.read_csv(file_path, sep="\t")
    return tfs




if __name__ == "__main__":
    pass
