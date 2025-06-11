from anndata import (
    AnnData,
    read_loom
)
from .MultiConfiguration import MDKM



import numpy as np
import os
from os import PathLike
import pandas as pd
from pathlib import Path
import re
from typing import (
    Dict,
    Literal,
    Union
)

# Imports from dynamo
from ..dynamo_logger import (
    LoggerManager,
    main_exception,
    main_info,
)

# Imports from MultiDynamo
from .old_MultiVelocity import MultiVelocity
from .MultiPreprocessor import aggregate_peaks_10x


def add_splicing_data(
        mdata                     ,
        multiome_base_path:        Union[PathLike, str],
        rna_splicing_loom:         Union[PathLike, str] = 'multiome.loom',
        cellranger_path_structure: bool = True
):
    from mudata import MuData
    # Extract accessibility and transcriptomic counts
    atac_adata, rna_adata = mdata.mod['atac'], mdata.mod['rna']

    # Read in spicing data
    splicing_data_path = os.path.join(multiome_base_path,
                                      'velocyto' if cellranger_path_structure else '',
                                      rna_splicing_loom)
    ldata = read_loom(filename=Path(splicing_data_path))

    # Merge splicing data with transcriptomic data
    rna_adata.var_names_make_unique()
    ldata.var_names_make_unique()

    common_obs = pd.unique(rna_adata.obs_names.intersection(ldata.obs_names))
    common_vars = pd.unique(rna_adata.var_names.intersection(ldata.var_names))

    if len(common_obs) == 0:
        # Try cleaning cell indices, if intersection of indices is vacuous
        clean_obs_names(rna_adata)
        clean_obs_names(ldata)
        common_obs = rna_adata.obs_names.intersection(ldata.obs_names)

    # Restrict to common cell indices and genes
    rna_adata = rna_adata[common_obs, common_vars].copy()
    ldata = ldata[common_obs, common_vars].copy()

    # Transfer layers from ldata
    for key, data in ldata.layers.items():
        if key not in rna_adata.layers:
            rna_adata.layers[key] = data.copy()

    # Copy over the loom counts to a counts layer
    rna_adata.layers[MDKM.RNA_COUNTS_LAYER] = rna_adata.layers[MDKM.RNA_COUNTS_LAYER_FROM_LOOM].copy()

    mdata = MuData({'atac': atac_adata, 'rna': rna_adata})

    return mdata


# These are convenience functions pattern after (but not identical to) ones in scvelo
def clean_obs_names(
        adata:     AnnData,
        alphabet:  Literal['[AGTCBDHKMNRSVWY]'] = '[AGTCBDHKMNRSVWY]',
        batch_key: str = MDKM.INFERRED_BATCH_KEY,
        id_length: int = 16
) -> AnnData:
    if adata.obs_names.map(len).unique().size == 1:
        # Here if all cell indices have the same numbers of characters
        # ... find (first) instance of id_length valid nucleotides in the first cell index
        start, end = re.search(alphabet * id_length, adata.obs_names[0]).span()

        # ... truncate the cell indices to the valid nucleotides
        new_obs_names = [obs_name[start:end] for obs_name in adata.obs_names]

        # ... any characters prior to the characters that define the new cell index
        #     might specify the batch, so save it as tuple with the new cell index
        prefixes = [
            obs_name.replace(new_obs_name, "")
            for obs_name, new_obs_name in zip(adata.obs_names, new_obs_names)
        ]
    else:
        # Here if cell indices have different lengths
        prefixes, new_obs_names = [], []
        for obs_name in adata.obs_names:
            # ... loop over the cell indices individually; find the (first) instance
            #     of id_length valid nucleotides in each cell index
            start, end = re.search(alphabet * id_length, adata.obs_names[0]).span()

            # ... truncate the cell indices to the valid nucleotides
            new_obs_names.append(obs_name[start:end])

            # ... any characters prior to the characters that define the new cell index
            #     might specify the batch, so save it as tuple with the new cell index
            prefixes.append(obs_name.replace(obs_name[start:end], ""))

    adata.obs_names = new_obs_names
    adata.obs_names_make_unique()

    if len(prefixes[0]) > 0 and len(np.unique(prefixes)) > 1:
        # If non-trival list of prefices (non-trivial length and more than one different),
        # then add MDKM.INFERRED_BATCH_KEY to cell metadata
        adata.obs[batch_key] = (
            pd.Categorical(prefixes)
            if len(np.unique(prefixes)) < adata.n_obs
            else prefixes
        )

    return adata


def homogenize_mudata_obs_names(
        mdata                     ,
        alphabet:                 Literal['[AGTCBDHKMNRSVWY]'] = '[AGTCBDHKMNRSVWY]',
        batch_key:                str = MDKM.INFERRED_BATCH_KEY,
        id_length:                int = 16
):
    from mudata import MuData
    cleaned_modality_dict = {}
    for modality, modality_adata in mdata.mod.items():
        cleaned_modality_adata = clean_obs_names(adata=modality_adata,
                                                 alphabet=alphabet,
                                                 batch_key=batch_key,
                                                 id_length=id_length)
        cleaned_modality_dict[modality] = cleaned_modality_adata.copy()
    return MuData(cleaned_modality_dict)


def read(path_dict: Dict) -> MultiVelocity:
    pass # Can significantly simply

# ... from unmatched scRNA-seq and scATAC-seq data
def read_10x_atac_rna_h5_old(
        atac_path:                 Union[PathLike, str],
        rna_path:                  Union[PathLike, str],
        atac_counts_matrix:        Union[PathLike, str] = 'filtered_peak_bc_matrix',
        rna_h5_fn:                 Union[PathLike, str] = 'filtered_feature_bc_matrix.h5',
        rna_splicing_loom:         Union[PathLike, str] = 'multiome.loom',
        alphabet:                  Literal['[AGTCBDHKMNRSVWY]'] = '[AGTCBDHKMNRSVWY]',
        batch_key:                 str = MDKM.INFERRED_BATCH_KEY,
        cellranger_path_structure: bool = True,
        id_length:                 int = 16
):
    from mudata import MuData
    from muon import atac as ac
    import muon as mu
    import scvi
    main_info('Deserializing UNMATCHED scATAC-seq and scRNA-seq data ...')
    temp_logger = LoggerManager.gen_logger('read_10x_atac_rna_h5')
    temp_logger.log_time()

    # Read scATAC-seq h5 file
    # ... counts
    main_info(f'reading scATAC-seq data', indent_level=2)
    atac_matrix_path = os.path.join(atac_path,
                                    'outs' if cellranger_path_structure else '',
                                    atac_counts_matrix)

    atac_adata = scvi.data.read_10x_atac(atac_matrix_path)

    # Read scRNA-seq h5 file
    main_info(f'reading scRNA-seq data', indent_level=2)
    rna_h5_path = os.path.join(rna_path,
                               'outs' if cellranger_path_structure else '',
                               rna_h5_fn)

    rna_adata = mu.read_10x_h5(filename=Path(rna_h5_path)).mod['rna']

    # Assemble MuData object
    main_info(f'combining scATAC-seq data and scRNA-seq data into MuData object ...', indent_level=2)
    mdata = MuData({'atac': atac_adata, 'rna': rna_adata})

    # Flag the scATAC-seq data as unmatched to the scRNA-seq data
    main_info(f'<insert> .uns[{MDKM.MATCHED_ATAC_RNA_DATA_KEY}] = False', indent_level=3)
    mdata.mod['atac'].uns[MDKM.MATCHED_ATAC_RNA_DATA_KEY] = False

    # Add path to fragment file
    main_info(f"<insert> path to fragments file in .uns['files']", indent_level=3)
    mdata.mod['atac'].uns['files'] = {'fragments': os.path.join(atac_path,
                                                                'outs' if cellranger_path_structure else '',
                                                                'fragments.tsv.gz')}

    # Add 'outs' paths
    # ... atac
    mdata.mod['atac'].uns['base_data_path'] = atac_path

    # ... rna
    mdata.mod['rna'].uns['base_data_path'] = rna_path

    # Add peak annotation
    main_info(f'adding peak annotation ...', indent_level=2)
    ac.tl.add_peak_annotation(data=mdata, annotation=os.path.join(atac_path,
                                                                  'outs' if cellranger_path_structure else '',
                                                                  'peak_annotation.tsv'))

    # Homogenize cell indices across modalities
    main_info(f'homogenizing cell indices ...', indent_level=2)
    mdata = homogenize_mudata_obs_names(mdata=mdata,
                                        alphabet=alphabet,
                                        batch_key=batch_key,
                                        id_length=id_length)

    # Add transcriptomic splicing data
    main_info(f'adding splicing data ...', indent_level=2)
    mdata = add_splicing_data(mdata=mdata,
                              multiome_base_path=rna_path,
                              rna_splicing_loom=rna_splicing_loom,
                              cellranger_path_structure=cellranger_path_structure)

    temp_logger.finish_progress(progress_name='read_10x_atac_rna_h5')

    return mdata


# ... from matched 10X multiome
def read_10x_multiome_h5_old(
        multiome_base_path:        Union[PathLike, str],
        multiome_h5_fn:            Union[PathLike, str] = 'filtered_feature_bc_matrix.h5',
        rna_splicing_loom:         Union[PathLike, str] = 'multiome.loom',
        alphabet:                  Literal['[AGTCBDHKMNRSVWY]'] = '[AGTCBDHKMNRSVWY]',
        batch_key:                 str = MDKM.INFERRED_BATCH_KEY,
        cellranger_path_structure: bool = True,
        id_length: int = 16
):
    from mudata import MuData
    import muon as mu
    from muon import atac as ac

    main_info('Deserializing MATCHED scATAC-seq and scRNA-seq data ...')
    temp_logger = LoggerManager.gen_logger('read_10x_multiome_h5')
    temp_logger.log_time()

    # Assemble absolute path to multiomic data
    full_multiome_path = os.path.join(multiome_base_path,
                                      'outs' if cellranger_path_structure else '',
                                      multiome_h5_fn)

    # Read the multiome h5 file
    main_info(f'reading the multiome h5 file ...', indent_level=2)
    mdata = mu.read_10x_h5(Path(full_multiome_path), extended=True)

    # Flag the scATAC-seq data as matched to the scRNA-seq data
    main_info(f'<insert> .uns[{MDKM.MATCHED_ATAC_RNA_DATA_KEY}] = True', indent_level=3)
    mdata.mod['atac'].uns[MDKM.MATCHED_ATAC_RNA_DATA_KEY] = True

    # Add 'outs' paths - Note: for multiome they are identical
    # ... atac
    mdata.mod['atac'].uns['base_data_path'] = multiome_base_path

    # ... rna
    mdata.mod['rna'].uns['base_data_path'] = multiome_base_path

    # Add path to fragment file
    main_info(f"<insert> path to fragments file in .uns['files'] ...", indent_level=3)
    mdata.mod['atac'].uns['files'] = {'fragments': os.path.join(multiome_base_path,
                                                                'outs' if cellranger_path_structure else '',
                                                                'fragments.tsv.gz')}

    # Add peak annotation
    main_info(f'adding peak annotation ...', indent_level=2)
    ac.tl.add_peak_annotation(data=mdata, annotation=os.path.join(multiome_base_path,
                                                                  'outs' if cellranger_path_structure else '',
                                                                  'peak_annotation.tsv'))

    # Homogenize cell indices across modalities
    main_info(f'homogenizing cell indices ...', indent_level=2)
    mdata = homogenize_mudata_obs_names(mdata=mdata,
                                        alphabet=alphabet,
                                        batch_key=batch_key,
                                        id_length=id_length)

    # Add transcriptomic splicing data
    main_info(f'adding splicing data ...', indent_level=2)
    mdata = add_splicing_data(mdata=mdata,
                              multiome_base_path=multiome_base_path,
                              rna_splicing_loom=rna_splicing_loom,
                              cellranger_path_structure=cellranger_path_structure)

    temp_logger.finish_progress(progress_name='read_10x_multiome_h5')

    return mdata


def read_10x_multiome_h5(
        multiome_base_path:        Union[PathLike, str],
        multiome_h5_fn:            Union[PathLike, str] = 'filtered_feature_bc_matrix.h5',
        rna_splicing_loom:         Union[PathLike, str] = 'multiome.loom',
        alphabet:                  Literal['[AGTCBDHKMNRSVWY]'] = '[AGTCBDHKMNRSVWY]',
        batch_key:                 str = MDKM.INFERRED_BATCH_KEY,
        cellranger_path_structure: bool = True,
        id_length: int = 16,
        gtf_path: Union[PathLike, str] = None,
):
    import muon as mu
    from muon import atac as ac

    main_info('Deserializing MATCHED scATAC-seq and scRNA-seq data ...')
    temp_logger = LoggerManager.gen_logger('read_10x_multiome_h5')
    temp_logger.log_time()

    # Assemble absolute path to multiomic data
    full_multiome_path = os.path.join(multiome_base_path,
                                      'outs' if cellranger_path_structure else '',
                                      multiome_h5_fn)

    # Read the multiome h5 file
    main_info(f'reading the multiome h5 file ...', indent_level=2)
    mdata = mu.read_10x_h5(Path(full_multiome_path), extended=True)

    # Flag the scATAC-seq data as matched to the scRNA-seq data
    main_info(f'<insert> .uns[{MDKM.MATCHED_ATAC_RNA_DATA_KEY}] = True', indent_level=3)
    mdata.mod['atac'].uns[MDKM.MATCHED_ATAC_RNA_DATA_KEY] = True

    # Add 'outs' paths - Note: for multiome they are identical
    # ... atac
    mdata.mod['atac'].uns['base_data_path'] = multiome_base_path

    # ... rna
    mdata.mod['rna'].uns['base_data_path'] = multiome_base_path

    #Add path of fragments file if exist
    fragments_path = os.path.join(multiome_base_path,
                                  'outs' if cellranger_path_structure else '',
                                  'fragments.tsv.gz')
    if os.path.exists(fragments_path):
        main_info(f"<insert> path to fragments file in .uns['files'] ...", indent_level=3)
        mdata.mod['atac'].uns['files'] = {'fragments': fragments_path}
    else:
        main_info(f"fragments file not found in {fragments_path}", indent_level=3)

    # Add peak annotation file if exist
    peak_annotation_path = os.path.join(multiome_base_path,
                                        'outs' if cellranger_path_structure else '',
                                        'peak_annotation.tsv')
    if os.path.exists(peak_annotation_path):
        main_info(f'adding peak annotation ...', indent_level=2)
        ac.tl.add_peak_annotation(data=mdata, annotation=peak_annotation_path)

    elif gtf_path is not None:
        main_info(f'adding peak annotation from gtf file ...', indent_level=2)
        import Epiverse as ev
        atac_anno=ev.utils.Annotation(gtf_path)
        atac_anno.tss_init(upstream=1000,
                   downstream=100)
        atac_anno.distal_init(upstream=[1000,200000],
                            downstream=[1000,200000])
        atac_anno.body_init()        

        import pandas as pd
        k=0
        for chr in mdata['atac'].var['seqnames'].unique():
            if k==0:
                merge_pd=atac_anno.query_multi(query_list=mdata['atac'].var.loc[mdata['atac'].var['seqnames']==chr].index.tolist(),
                                        chrom=chr,batch=4,ncpus=8)
            else:
                merge_pd1=atac_anno.query_multi(query_list=mdata['atac'].var.loc[mdata['atac'].var['seqnames']==chr].index.tolist(),
                                        chrom=chr,batch=4,ncpus=8)
                merge_pd=pd.concat([merge_pd,merge_pd1])
            k+=1
        merge_pd=atac_anno.merge_info(merge_pd)
        atac_anno.add_gene_info(mdata['atac'],merge_pd,
                        columns=['peaktype','neargene','neargene_tss'])
    else:
        main_info(f"peak annotation file not found in {peak_annotation_path} and gtf file not provided", indent_level=3)

    # Homogenize cell indices across modalities
    main_info(f'homogenizing cell indices ...', indent_level=2)
    mdata = homogenize_mudata_obs_names(mdata=mdata,
                                        alphabet=alphabet,
                                        batch_key=batch_key,
                                        id_length=id_length)
    
    # Add transcriptomic splicing data if exist
    rna_splicing_loom_path = os.path.join(multiome_base_path,
                                          'velocyto' if cellranger_path_structure else '',
                                          rna_splicing_loom)
    if os.path.exists(rna_splicing_loom_path):
        main_info(f'adding splicing data ...', indent_level=2)
        mdata = add_splicing_data(mdata=mdata,
                                  multiome_base_path=multiome_base_path,
                                  rna_splicing_loom=rna_splicing_loom,
                                  cellranger_path_structure=cellranger_path_structure)
    else:
        main_info(f"splicing data file not found in {rna_splicing_loom_path}", indent_level=3)

    # Aggregate_peaks_10x
    main_info(f'aggregating peaks ...', indent_level=2)
    feature_linkage_path=os.path.join(multiome_base_path,
                                        'outs' if cellranger_path_structure else '',
                                        'analysis/feature_linkage/feature_linkage.bedpe')
    adata_aggr = aggregate_peaks_10x(mdata['atac'],
                                    peak_annotation_path,
                                    feature_linkage_path)
    
    mdata.mod['aggr']=adata_aggr


    temp_logger.finish_progress(progress_name='read_10x_multiome_h5')

    return mdata