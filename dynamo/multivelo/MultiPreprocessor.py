# Imports from external modules
from anndata import AnnData
from .MultiConfiguration import MDKM

import numpy as np
import os
import pandas as pd

from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix, diags

from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

# Imports from dynamo
from ..dynamo_logger import (
    LoggerManager,
    main_debug,
    main_exception,
    main_info,
    main_info_insert_adata,
    main_warning,
)
from ..preprocessing.gene_selection import (
    select_genes_monocle
)
from ..preprocessing.normalization import (
    calc_sz_factor,
    normalize
)
from ..preprocessing.pca import (
    pca
)
from ..preprocessing.Preprocessor import (
    Preprocessor
)
from ..preprocessing.QC import (
    filter_cells_by_highly_variable_genes,
    filter_cells_by_outliers as monocle_filter_cells_by_outliers,
    filter_genes_by_outliers as monocle_filter_genes_by_outliers
)
from ..preprocessing.transform import (
    log1p
)
from ..preprocessing.utils import (
    collapse_species_adata,
    convert2symbol
)

# Imports from MultiDynamo
from .ATACseqTools import (
    tfidf_normalize
)
from .MultiQC import (
    modality_basic_stats,
    modality_filter_cells_by_outliers,
    modality_filter_features_by_outliers
)

# Define a custom type for the recipe dictionary using TypedDict
ATACType = Literal['archR', 'cicero', 'muon', 'signac']
CITEType = Literal['seurat']
HiCType = Literal['periwal']
ModalityType = Literal['atac', 'cite', 'hic', 'rna']
RNAType = Literal['monocle', 'seurat', 'sctransform', 'pearson_residuals', 'monocle_pearson_residuals']

class RecipeDataType(TypedDict, total=False): # total=False allows partial dictionary to be valid
    atac: ATACType
    cite: CITEType
    hic: HiCType
    rna: RNAType


# The Multiomic Preprocessor class, MultiPreprocessor
class MultiPreprocessor(Preprocessor):
    def __init__(
            self,
            cell_cycle_score_enable:                        bool=False,
            cell_cycle_score_kwargs:                        Dict[str, Any] = {},
            collapse_species_adata_function:                Callable = collapse_species_adata,
            convert_gene_name_function:                     Callable=convert2symbol,
            filter_cells_by_highly_variable_genes_function: Callable = filter_cells_by_highly_variable_genes,
            filter_cells_by_highly_variable_genes_kwargs:   Dict[str, Any] = {},
            filter_cells_by_outliers_function:              Callable=monocle_filter_cells_by_outliers,
            filter_cells_by_outliers_kwargs:                Dict[str, Any] = {},
            filter_genes_by_outliers_function:              Callable=monocle_filter_genes_by_outliers,
            filter_genes_by_outliers_kwargs:                Dict[str, Any] = {},
            force_gene_list:                                Optional[List[str]]=None,
            gene_append_list:                               List[str] = [],
            gene_exclude_list:                              List[str] = {},
            norm_method:                                    Callable=log1p,
            norm_method_kwargs:                             Dict[str, Any] = {},
            normalize_by_cells_function:                    Callable=normalize,
            normalize_by_cells_function_kwargs:             Dict[str, Any] = {},
            normalize_selected_genes_function:              Callable=None,
            normalize_selected_genes_kwargs:                Dict[str, Any] = {},
            pca_function:                                   Callable=pca,
            pca_kwargs:                                     Dict[str, Any] = {},
            regress_out_kwargs:                             Dict[List[str], Any] = {},
            sctransform_kwargs:                             Dict[str, Any] = {},
            select_genes_function:                          Callable = select_genes_monocle,
            select_genes_kwargs:                            Dict[str, Any] = {},
            size_factor_function:                           Callable=calc_sz_factor,
            size_factor_kwargs:                             Dict[str, Any] = {}) -> None:
        super().__init__(
            collapse_species_adata_function = collapse_species_adata_function,
            convert_gene_name_function = convert_gene_name_function,
            filter_cells_by_outliers_function = filter_cells_by_outliers_function,
            filter_cells_by_outliers_kwargs = filter_cells_by_outliers_kwargs,
            filter_genes_by_outliers_function = filter_genes_by_outliers_function,
            filter_genes_by_outliers_kwargs = filter_genes_by_outliers_kwargs,
            filter_cells_by_highly_variable_genes_function = filter_cells_by_highly_variable_genes_function,
            filter_cells_by_highly_variable_genes_kwargs = filter_cells_by_highly_variable_genes_kwargs,
            normalize_by_cells_function = normalize_by_cells_function,
            normalize_by_cells_function_kwargs = normalize_by_cells_function_kwargs,
            size_factor_function = size_factor_function,
            size_factor_kwargs = size_factor_kwargs,
            select_genes_function = select_genes_function,
            select_genes_kwargs = select_genes_kwargs,
            normalize_selected_genes_function = normalize_selected_genes_function,
            normalize_selected_genes_kwargs = normalize_selected_genes_kwargs,
            norm_method = norm_method,
            norm_method_kwargs = norm_method_kwargs,
            pca_function = pca_function,
            pca_kwargs = pca_kwargs,
            gene_append_list = gene_append_list,
            gene_exclude_list = gene_exclude_list,
            force_gene_list = force_gene_list,
            sctransform_kwargs = sctransform_kwargs,
            regress_out_kwargs = regress_out_kwargs,
            cell_cycle_score_enable = cell_cycle_score_enable,
            cell_cycle_score_kwargs = cell_cycle_score_kwargs
        )

    def preprocess_atac(
            self,
            mdata                     ,
            recipe:          ATACType = 'muon',
            tkey:            Optional[str] = None,
            experiment_type: Optional[str] = None
    ):
        from mudata import MuData
        if recipe == 'archR':
            self.preprocess_atac_archr(mdata,
                                       tkey=tkey,
                                       experiment_type=experiment_type)
        elif recipe == 'cicero':
            self.preprocess_atac_cicero(mdata,
                                        tkey=tkey,
                                        experiment_type=experiment_type)
        elif recipe == 'muon':
            self.preprocess_atac_muon(mdata,
                                      tkey=tkey,
                                      experiment_type=experiment_type)
        elif recipe == 'signac':
            self.preprocess_atac_signac(mdata,
                                        tkey=tkey,
                                        experiment_type=experiment_type)
        else:
            raise NotImplementedError("preprocess recipe chosen not implemented: %s" % recipe)

    def preprocess_atac_archr(
            self,
            mdata                     ,
            tkey: Optional[str] = None,
            experiment_type: Optional[str] = None
    ):
        from mudata import MuData
        pass

    def preprocess_atac_cicero(
            self,
            mdata                     ,
            tkey: Optional[str] = None,
            experiment_type: Optional[str] = None
    ):
        from mudata import MuData
        pass

    def preprocess_atac_muon(
            self,
            mdata                     ,
            tkey:            Optional[str] = None,
            experiment_type: Optional[str] = None
    ):
        from muon import atac as ac
        import scanpy as sc
        main_info('Running muon preprocessing pipeline for scATAC-seq data ...')
        preprocess_logger = LoggerManager.gen_logger('preprocess_atac_muon')
        preprocess_logger.log_time()

        # Standardize MuData object
        self.standardize_mdata(mdata, tkey, experiment_type)

        # Filter peaks
        modality_filter_features_by_outliers(mdata,
                                             modality='atac',
                                             quantiles=[0.01, 0.99],
                                             var_key='n_cells_by_counts')

        # Filter cells
        modality_filter_cells_by_outliers(mdata,
                                          modality='atac',
                                          quantiles=[0.01, 0.99],
                                          obs_key='n_genes_by_counts')

        modality_filter_cells_by_outliers(mdata,
                                          modality='atac',
                                          quantiles=[0.01, 0.99],
                                          obs_key='total_counts')

        # Extract chromatin accessibility and transcriptome
        atac_adata, rna_adata = mdata.mod['atac'], mdata.mod['rna']

        # ... store counts layer used for SCVI's variational autoencoders
        atac_adata.layers[MDKM.ATAC_COUNTS_LAYER] = atac_adata.X
        rna_adata.layers[MDKM.RNA_COUNTS_LAYER] = rna_adata.X

        # ... compute TF-IDF
        main_info(f'computing TF-IDF', indent_level=1)
        atac_adata = tfidf_normalize(atac_adata=atac_adata, mv_algorithm=False)

        # Normalize
        main_info(f'normalizing', indent_level=1)
        sc.pp.normalize_total(atac_adata, target_sum=1e4)
        sc.pp.log1p(atac_adata)

        # Feature selection
        main_info(f'feature selection', indent_level=1)
        sc.pp.highly_variable_genes(atac_adata, min_mean=0.05, max_mean=1.5, min_disp=0.5)
        main_info(f'identified {np.sum(atac_adata.var.highly_variable)} highly variable features', indent_level=2)

        # Store current AnnData object in raw
        atac_adata.raw = atac_adata

        # Latent sematic indexing
        main_info(f'computing latent sematic indexing', indent_level=1)
        ac.tl.lsi(atac_adata)

        # ... drop first component (size related)
        main_info(f'<insert> X_lsi key in .obsm', indent_level=2)
        atac_adata.obsm[MDKM.ATAC_OBSM_LSI_KEY] = atac_adata.obsm[MDKM.ATAC_OBSM_LSI_KEY][:, 1:]
        main_info(f'<insert> LSI key in .varm', indent_level=2)
        atac_adata.varm[MDKM.ATAC_VARM_LSI_KEY] = atac_adata.varm[MDKM.ATAC_VARM_LSI_KEY][:, 1:]
        main_info(f'<insert> [lsi][stdev] key in .uns', indent_level=2)
        atac_adata.uns['lsi']['stdev'] = atac_adata.uns['lsi']['stdev'][1:]

        # ... perhaps gratuitous deep copy
        mdata.mod['atac'] = atac_adata.copy()

        preprocess_logger.finish_progress(progress_name='preprocess_atac_muon')

    def preprocess_atac_signac(
            self,
            mdata                     ,
            recipe:          ATACType = 'muon',
            tkey:            Optional[str] = None,
            experiment_type: Optional[str] = None
    ):
        from mudata import MuData
        pass

    def preprocess_cite(
            self,
            mdata                     ,
            recipe: CITEType
    ):
        from mudata import MuData
        pass

    def preprocess_hic(
            self,
            mdata                     ,
            recipe: HiCType
    ):
        from mudata import MuData
        pass

    def preprocess_mdata(
            self,
            mdata                     ,
            recipe_dict:     RecipeDataType = None,
            tkey:            Optional[str] = None,
            experiment_type: Optional[str] = None,
    ):
        from mudata import MuData
        """Preprocess the MuData object with the recipe specified.

        Args:
            mdata: An AnnData object.
            recipe_dict: The recipe used to preprocess the data. Current modalities are scATAC-seq, CITE-seq, scHi-C
                         and scRNA-seq
            tkey: the key for time information (labeling time period for the cells) in .obs. Defaults to None.
            experiment_type: the experiment type of the data. If not provided, would be inferred from the data.

        Raises:
            NotImplementedError: the recipe is invalid.
        """

        if recipe_dict is None:
            # Default recipe
            recipe_dict = {'atac': 'signac', 'rna': 'seurat'}

        for mod, recipe in recipe_dict.items():
            if mod not in mdata.mod:
                main_exception((f'Modality {mod} not found in MuData object'))

            if mod == 'atac':
                self.preprocess_atac(mdata=mdata,
                                     recipe=recipe,
                                     tkey=tkey,
                                     experiment_type=experiment_type)

            elif mod == 'cite':
                self.preprocess_cite(mdata=mdata,
                                     recipe=recipe,
                                     tkey=tkey,
                                     experiment_type=experiment_type)
            elif mod == 'hic':
                self.preprocess_hic(mdata=mdata,
                                    recipe=recipe,
                                    tkey=tkey,
                                    experiment_type=experiment_type)
            elif mod == 'rna':
                #rna_adata = mdata.mod.get('rna', None)
                self.preprocess_adata(adata=mdata['rna'],
                                      recipe=recipe,
                                      tkey=tkey,
                                      experiment_type=experiment_type)
            else:
                raise NotImplementedError(f'Preprocess recipe not implemented for modality: {mod}')

        # Integrate modalities - at this point have filtered out poor quality cells for individual
        # modalities.  Next we need to

    def standardize_mdata(
            self,
            mdata                     ,
            tkey:            str,
            experiment_type: str
):
        from mudata import MuData
        """Process the scATAC-seq modality within MuData to make it meet the standards of dynamo.

        The index of the observations would be ensured to be unique. The layers with sparse matrix would be converted to
        compressed csr_matrix. MDKM.allowed_layer_raw_names() will be used to define only_splicing, only_labeling and
        splicing_labeling keys.

        Args:
            mdata: an AnnData object.
            tkey: the key for time information (labeling time period for the cells) in .obs.
            experiment_type: the experiment type.
        """

        for modality, modality_adata in mdata.mod.items():
            if modality == 'rna':
                # Handled by dynamo
                continue

            # Compute basic QC metrics
            modality_basic_stats(mdata=mdata, modality=modality)

            self.add_experiment_info(modality_adata, tkey, experiment_type)
            main_info_insert_adata("tkey=%s" % tkey, "uns['pp']", indent_level=2)
            main_info_insert_adata("experiment_type=%s" % modality_adata.uns["pp"]["experiment_type"],
                                   "uns['pp']",
                                   indent_level=2)

            self.convert_layers2csr(modality_adata)


def aggregate_peaks_10x(adata_atac, peak_annot_file, linkage_file,
                        peak_dist=10000, min_corr=0.5, gene_body=False,
                        return_dict=False, parallel=False, n_jobs=1):

    """Peak to gene aggregation.

    This function aggregates promoter and enhancer peaks to genes based on the
    10X linkage file.

    Parameters
    ----------
    adata_atac: :class:`~anndata.AnnData`
        ATAC anndata object which stores raw peak counts.
    peak_annot_file: `str`
        Peak annotation file from 10X CellRanger ARC.
    linkage_file: `str`
        Peak-gene linkage file from 10X CellRanger ARC. This file stores highly
        correlated peak-peak and peak-gene pair information.
    peak_dist: `int` (default: 10000)
        Maximum distance for peaks to be included for a gene.
    min_corr: `float` (default: 0.5)
        Minimum correlation for a peak to be considered as enhancer.
    gene_body: `bool` (default: `False`)
        Whether to add gene body peaks to the associated promoters.
    return_dict: `bool` (default: `False`)
        Whether to return promoter and enhancer dictionaries.

    Returns
    -------
    A new ATAC anndata object which stores gene aggreagted peak counts.
    Additionally, if `return_dict==True`:
        A dictionary which stores genes and promoter peaks.
        And a dictionary which stores genes and enhancer peaks.
    """
    promoter_dict = {}
    distal_dict = {}
    gene_body_dict = {}
    corr_dict = {}

    # read annotations
    with open(peak_annot_file) as f:
        header = next(f)
        tmp = header.split('\t')
        if len(tmp) == 4:
            cellranger_version = 1
        elif len(tmp) == 6:
            cellranger_version = 2
        else:
            raise ValueError('Peak annotation file should contain 4 columns '
                             '(CellRanger ARC 1.0.0) or 6 columns (CellRanger '
                             'ARC 2.0.0)')

        main_info(f'CellRanger ARC identified as {cellranger_version}.0.0',
                    indent_level=1)

        if cellranger_version == 1:
            for line in f:
                tmp = line.rstrip().split('\t')
                tmp1 = tmp[0].split('_')
                peak = f'{tmp1[0]}:{tmp1[1]}-{tmp1[2]}'
                if tmp[1] != '':
                    genes = tmp[1].split(';')
                    dists = tmp[2].split(';')
                    types = tmp[3].split(';')
                    for i, gene in enumerate(genes):
                        dist = dists[i]
                        annot = types[i]
                        if annot == 'promoter':
                            if gene not in promoter_dict:
                                promoter_dict[gene] = [peak]
                            else:
                                promoter_dict[gene].append(peak)
                        elif annot == 'distal':
                            if dist == '0':
                                if gene not in gene_body_dict:
                                    gene_body_dict[gene] = [peak]
                                else:
                                    gene_body_dict[gene].append(peak)
                            else:
                                if gene not in distal_dict:
                                    distal_dict[gene] = [peak]
                                else:
                                    distal_dict[gene].append(peak)
        else:
            for line in f:
                tmp = line.rstrip().split('\t')
                peak = f'{tmp[0]}:{tmp[1]}-{tmp[2]}'
                gene = tmp[3]
                dist = tmp[4]
                annot = tmp[5]
                if annot == 'promoter':
                    if gene not in promoter_dict:
                        promoter_dict[gene] = [peak]
                    else:
                        promoter_dict[gene].append(peak)
                elif annot == 'distal':
                    if dist == '0':
                        if gene not in gene_body_dict:
                            gene_body_dict[gene] = [peak]
                        else:
                            gene_body_dict[gene].append(peak)
                    else:
                        if gene not in distal_dict:
                            distal_dict[gene] = [peak]
                        else:
                            distal_dict[gene].append(peak)

    # read linkages
    with open(linkage_file) as f:
        for line in f:
            tmp = line.rstrip().split('\t')
            if tmp[12] == "peak-peak":
                peak1 = f'{tmp[0]}:{tmp[1]}-{tmp[2]}'
                peak2 = f'{tmp[3]}:{tmp[4]}-{tmp[5]}'
                tmp2 = tmp[6].split('><')[0][1:].split(';')
                tmp3 = tmp[6].split('><')[1][:-1].split(';')
                corr = float(tmp[7])
                for t2 in tmp2:
                    gene1 = t2.split('_')
                    for t3 in tmp3:
                        gene2 = t3.split('_')
                        # one of the peaks is in promoter, peaks belong to the
                        # same gene or are close in distance
                        if (((gene1[1] == "promoter") !=
                            (gene2[1] == "promoter")) and
                            ((gene1[0] == gene2[0]) or
                             (float(tmp[11]) < peak_dist))):

                            if gene1[1] == "promoter":
                                gene = gene1[0]
                            else:
                                gene = gene2[0]
                            if gene in corr_dict:
                                # peak 1 is in promoter, peak 2 is not in gene
                                # body -> peak 2 is added to gene 1
                                if (peak2 not in corr_dict[gene] and
                                    gene1[1] == "promoter" and
                                    (gene2[0] not in gene_body_dict or
                                     peak2 not in gene_body_dict[gene2[0]])):

                                    corr_dict[gene][0].append(peak2)
                                    corr_dict[gene][1].append(corr)
                                # peak 2 is in promoter, peak 1 is not in gene
                                # body -> peak 1 is added to gene 2
                                if (peak1 not in corr_dict[gene] and
                                    gene2[1] == "promoter" and
                                    (gene1[0] not in gene_body_dict or
                                     peak1 not in gene_body_dict[gene1[0]])):

                                    corr_dict[gene][0].append(peak1)
                                    corr_dict[gene][1].append(corr)
                            else:
                                # peak 1 is in promoter, peak 2 is not in gene
                                # body -> peak 2 is added to gene 1
                                if (gene1[1] == "promoter" and
                                    (gene2[0] not in
                                     gene_body_dict
                                     or peak2 not in
                                     gene_body_dict[gene2[0]])):

                                    corr_dict[gene] = [[peak2], [corr]]
                                # peak 2 is in promoter, peak 1 is not in gene
                                # body -> peak 1 is added to gene 2
                                if (gene2[1] == "promoter" and
                                    (gene1[0] not in
                                     gene_body_dict
                                     or peak1 not in
                                     gene_body_dict[gene1[0]])):

                                    corr_dict[gene] = [[peak1], [corr]]
            elif tmp[12] == "peak-gene":
                peak1 = f'{tmp[0]}:{tmp[1]}-{tmp[2]}'
                tmp2 = tmp[6].split('><')[0][1:].split(';')
                gene2 = tmp[6].split('><')[1][:-1]
                corr = float(tmp[7])
                for t2 in tmp2:
                    gene1 = t2.split('_')
                    # peak 1 belongs to gene 2 or are close in distance
                    # -> peak 1 is added to gene 2
                    if ((gene1[0] == gene2) or (float(tmp[11]) < peak_dist)):
                        gene = gene1[0]
                        if gene in corr_dict:
                            if (peak1 not in corr_dict[gene] and
                                gene1[1] != "promoter" and
                                (gene1[0] not in gene_body_dict or
                                 peak1 not in gene_body_dict[gene1[0]])):

                                corr_dict[gene][0].append(peak1)
                                corr_dict[gene][1].append(corr)
                        else:
                            if (gene1[1] != "promoter" and
                                (gene1[0] not in gene_body_dict or
                                 peak1 not in gene_body_dict[gene1[0]])):
                                corr_dict[gene] = [[peak1], [corr]]
            elif tmp[12] == "gene-peak":
                peak2 = f'{tmp[3]}:{tmp[4]}-{tmp[5]}'
                gene1 = tmp[6].split('><')[0][1:]
                tmp3 = tmp[6].split('><')[1][:-1].split(';')
                corr = float(tmp[7])
                for t3 in tmp3:
                    gene2 = t3.split('_')
                    # peak 2 belongs to gene 1 or are close in distance
                    # -> peak 2 is added to gene 1
                    if ((gene1 == gene2[0]) or (float(tmp[11]) < peak_dist)):
                        gene = gene1
                        if gene in corr_dict:
                            if (peak2 not in corr_dict[gene] and
                                gene2[1] != "promoter" and
                                (gene2[0] not in gene_body_dict or
                                 peak2 not in gene_body_dict[gene2[0]])):

                                corr_dict[gene][0].append(peak2)
                                corr_dict[gene][1].append(corr)
                        else:
                            if (gene2[1] != "promoter" and
                                (gene2[0] not in gene_body_dict or
                                 peak2 not in gene_body_dict[gene2[0]])):

                                corr_dict[gene] = [[peak2], [corr]]

    gene_dict = promoter_dict
    enhancer_dict = {}
    promoter_genes = list(promoter_dict.keys())
    main_info(f'Found {len(promoter_genes)} genes with promoter peaks', indent_level=1)
    for gene in promoter_genes:
        if gene_body:  # add gene-body peaks
            if gene in gene_body_dict:
                for peak in gene_body_dict[gene]:
                    if peak not in gene_dict[gene]:
                        gene_dict[gene].append(peak)
        enhancer_dict[gene] = []
        if gene in corr_dict:  # add enhancer peaks
            for j, peak in enumerate(corr_dict[gene][0]):
                corr = corr_dict[gene][1][j]
                if corr > min_corr:
                    if peak not in gene_dict[gene]:
                        gene_dict[gene].append(peak)
                        enhancer_dict[gene].append(peak)

    # aggregate to genes
    import scipy as sp
    if sp.__version__ < '1.14.0':
        adata_atac_X_copy = adata_atac.X.A
    else:
        adata_atac_X_copy = adata_atac.X.toarray()
    gene_mat = np.zeros((adata_atac.shape[0], len(promoter_genes)))
    var_names = adata_atac.var_names.to_numpy()
    var_dict = {}

    for i, name in enumerate(var_names):
        var_dict.update({name: i})

    # if we only want to run one job at a time, then no parallelization
    # is necessary
    if n_jobs == 1:
        parallel = False

    if parallel:
        from joblib import Parallel, delayed
        # if we want to run in parallel, modify the gene_mat variable with
        # multiple cores, calling prepare_gene_mat with joblib.Parallel()
        Parallel(n_jobs=n_jobs,
                 require='sharedmem')(
                 delayed(prepare_gene_mat)(var_dict,
                                           gene_dict[promoter_genes[i]],
                                           gene_mat,
                                           adata_atac_X_copy,
                                           i)for i in tqdm(range(
                                               len(promoter_genes))))

    else:
        # if we aren't running in parallel, just call prepare_gene_mat
        # from a for loop
        for i, gene in tqdm(enumerate(promoter_genes),
                            total=len(promoter_genes)):
            prepare_gene_mat(var_dict,
                             gene_dict[promoter_genes[i]],
                             gene_mat,
                             adata_atac_X_copy,
                             i)

    gene_mat[gene_mat < 0] = 0
    gene_mat = AnnData(X=csr_matrix(gene_mat))
    gene_mat.obs_names = pd.Index(list(adata_atac.obs_names))
    gene_mat.var_names = pd.Index(promoter_genes)
    gene_mat = gene_mat[:, gene_mat.X.sum(0) > 0]
    if return_dict:
        return gene_mat, promoter_dict, enhancer_dict
    else:
        return gene_mat

def prepare_gene_mat(var_dict, peaks, gene_mat, adata_atac_X_copy, i):

    for peak in peaks:
        if peak in var_dict:
            peak_index = var_dict[peak]

            gene_mat[:, i] += adata_atac_X_copy[:, peak_index]



def knn_smooth_chrom(adata_atac, nn_idx=None, nn_dist=None, conn=None,
                     n_neighbors=None):
    """KNN smoothing.

    This function smooth (impute) the count matrix with k nearest neighbors.
    The inputs can be either KNN index and distance matrices or a pre-computed
    connectivities matrix (for example in adata_rna object).

    Parameters
    ----------
    adata_atac: :class:`~anndata.AnnData`
        ATAC anndata object.
    nn_idx: `np.darray` (default: `None`)
        KNN index matrix of size (cells, k).
    nn_dist: `np.darray` (default: `None`)
        KNN distance matrix of size (cells, k).
    conn: `csr_matrix` (default: `None`)
        Pre-computed connectivities matrix.
    n_neighbors: `int` (default: `None`)
        Top N neighbors to extract for each cell in the connectivities matrix.

    Returns
    -------
    `.layers['Mc']` stores imputed values.
    """
    if nn_idx is not None and nn_dist is not None:
        if nn_idx.shape[0] != adata_atac.shape[0]:
            raise ValueError('Number of rows of KNN indices does not equal to '
                             'number of observations.')
        if nn_dist.shape[0] != adata_atac.shape[0]:
            raise ValueError('Number of rows of KNN distances does not equal '
                             'to number of observations.')
        X = coo_matrix(([], ([], [])), shape=(nn_idx.shape[0], 1))
        from umap.umap_ import fuzzy_simplicial_set
        conn, sigma, rho, dists = fuzzy_simplicial_set(X, nn_idx.shape[1],
                                                       None, None,
                                                       knn_indices=nn_idx-1,
                                                       knn_dists=nn_dist,
                                                       return_dists=True)
    elif conn is not None:
        pass
    else:
        raise ValueError('Please input nearest neighbor indices and distances,'
                         ' or a connectivities matrix of size n x n, with '
                         'columns being neighbors.'
                         ' For example, RNA connectivities can usually be '
                         'found in adata.obsp.')

    conn = conn.tocsr().copy()
    n_counts = (conn > 0).sum(1).A1
    if n_neighbors is not None and n_neighbors < n_counts.min():
        from .sparse_matrix_utils import top_n_sparse
        conn = top_n_sparse(conn, n_neighbors)
    conn.setdiag(1)
    conn_norm = conn.multiply(1.0 / conn.sum(1)).tocsr()
    adata_atac.layers['Mc'] = csr_matrix.dot(conn_norm, adata_atac.X)
    adata_atac.obsp['connectivities'] = conn

