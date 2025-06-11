import anndata as ad
from anndata import AnnData

from concurrent.futures import as_completed, ThreadPoolExecutor


import numpy as np
from os import PathLike
import pandas as pd


from scipy.sparse import coo_matrix, csr_matrix, diags, hstack
from tqdm import tqdm
from typing import (
    Literal,
    Union
)

# Imports from dynamo
from ..dynamo_logger import (
    LoggerManager,
    main_info,
)


# Imports from MultiDynamo
from .MultiConfiguration import MDKM


def extend_gene_coordinates(
        bedtool,
        upstream: int = 2000,
        downstream: int = 0
):
    from pybedtools import BedTool
    extended_genes = []
    for feature in bedtool:
        if feature[2] == 'gene':
            start = max(0, int(feature.start) - upstream)
            end = int(feature.end) + downstream
            extended_genes.append((feature.chrom, start, end, feature.name))
    return BedTool(extended_genes)


def annotate_integrated_mdata(mdata,
                              celltypist_model: str = 'Immune_All_Low.pkl'
                              ) :
    import celltypist
    from celltypist import models
    import scanpy as sc
    from mudata import MuData
    # Extract the RNA data
    rna_adata = mdata.mod['rna'].copy()

    # ... revert to counts
    rna_adata.X = rna_adata.layers['counts'].copy()

    # ... normalize counts so total number per cell is 10,000 (required by celltypist)
    sc.pp.normalize_total(rna_adata,
                          target_sum=1e4)

    # ... pseudo-log transform (x -> log(1 + x)) for better dynamical range (and required by celltypist)
    sc.pp.log1p(rna_adata)

    # ... rerun PCA - CellTypist can need larger number than we already computed
    sc.pp.pca(rna_adata, n_comps=50)

    # ... recompute the neighborhood graph for majority voting
    sc.pp.neighbors(rna_adata,
                    n_neighbors=50,
                    n_pcs=50)

    # Download celltypist models for annotation
    models.download_models(force_update=True)

    # Select the low resolution immun cell model
    model = models.Model.load(model=celltypist_model)

    # Compute cell type labels
    predictions = celltypist.annotate(rna_adata,
                                      model=celltypist_model,
                                      majority_voting=True)

    # Transfer the predictions back to the RNA AnnData object
    rna_adata = predictions.to_adata()

    # Create dictionary from cell indices to cell types
    cellindex_to_celltype_dict = rna_adata.obs['majority_voting'].to_dict()

    # Apply the index map to both RNA and ATAC AnnData objects
    atac_adata, rna_adata = mdata.mod['atac'].copy(), mdata.mod['rna'].copy()
    atac_adata.obs['cell_type'] = atac_adata.obs.index.map(
        lambda cell_idx: cellindex_to_celltype_dict.get(cell_idx, 'Undefined'))
    rna_adata.obs['cell_type'] = rna_adata.obs.index.map(
        lambda cell_idx: cellindex_to_celltype_dict.get(cell_idx, 'Undefined'))

    return MuData({'atac': atac_adata.copy(), 'rna': rna_adata.copy()})


def gene_activity(
        atac_adata: AnnData,
        gtf_path: PathLike,
        upstream: int = 2000,
        downstream: int = 0
) -> Union[AnnData, None]:
    from pybedtools import BedTool
    # Drop UCSC convention for naming of chromosomes - This assumes we are using ENSEMBL-format of GTF
    atac_adata.var.index = [c.lstrip('chr') for c in atac_adata.var.index]

    # Read GTF to annotate genes
    main_info('reading GTF', indent_level=3)
    gene_annotations = BedTool(gtf_path)

    # Extend gene coordinates
    main_info('extending genes to estimate regulatory regions', indent_level=3)
    peak_extension_logger = LoggerManager.gen_logger('extend_genes')
    peak_extension_logger.log_time()

    extended_genes = extend_gene_coordinates(gene_annotations,
                                             upstream=upstream,
                                             downstream=downstream)

    peak_extension_logger.finish_progress(progress_name='extend_genes', indent_level=3)

    # Extract ATAC-seq peak coordinates
    chrom_list = atac_adata.var_names.str.split(':').str[0]  # .astype(int)
    start_list = atac_adata.var_names.str.split(':').str[1].str.split('-').str[0]  # .astype(int).astype(int)
    end_list = atac_adata.var_names.str.split(':').str[1].str.split('-').str[1]  # .astype(int).astype(int)

    # Convert ATAC-seq peak data to BedTool format
    atac_peaks = BedTool.from_dataframe(pd.DataFrame({
        'chrom': chrom_list,
        'start': start_list,
        'end': end_list
    }))

    # Find overlaps between peaks and extended genes
    main_info('overlapping peaks and extended genes', indent_level=3)
    linked_peaks = atac_peaks.intersect(extended_genes, wa=True, wb=True)

    # Create a DataFrame from the linked peaks
    linked_peaks_df = linked_peaks.to_dataframe(
        names=['chrom', 'peak_start', 'peak_end', 'chrom_gene', 'gene_start', 'gene_end', 'gene_name'])

    # Create a dictionary to map peak indices to gene names
    main_info('building dictionaries', indent_level=3)
    peak_to_gene = linked_peaks_df.set_index(['chrom', 'peak_start', 'peak_end'])['gene_name'].to_dict()
    peak_to_gene = {f'{chrom}:{start}-{end}': gene_name for (chrom, start, end), gene_name in peak_to_gene.items()}

    # Get the list of peaks from the ATAC-seq data
    peaks = atac_adata.var.index
    gene_names = np.array([peak_to_gene.get(peak, '') for peak in peaks])

    # Get the unique genes
    unique_genes = np.unique(gene_names)

    # Initialize a sparse matrix for gene activity scores
    n_cells, n_genes = atac_adata.n_obs, len(unique_genes)

    # Create a mapping from gene names to column indices in the sparse matrix
    gene_to_idx = {gene: idx for idx, gene in enumerate(unique_genes)}

    def process_peak(i):
        gene = gene_names[i]
        return gene_to_idx.get(gene, -1), atac_adata[:, i].X

    # Fill the sparse matrix with aggregated counts in parallel
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_peak, i) for i in range(len(peaks))]
        with tqdm(total=len(peaks), desc="Processing peaks") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                pbar.update(1)

    # Aggregate results in batches to minimize overhead
    main_info('aggregating results', indent_level=3)
    aggregation_logger = LoggerManager.gen_logger('aggregating_results')
    aggregation_logger.log_time()

    data = []
    rows = []
    cols = []

    # Loop through the results to gather data for COO matrix
    for col_idx, sparse_col_vector in results:
        # Extract row indices and data from the sparse column vector
        coo = sparse_col_vector.tocoo()
        data.extend(coo.data)
        rows.extend(coo.row)
        cols.extend([col_idx] * len(coo.row))

    # Create a COO matrix from collected data
    coo_matrix_all = coo_matrix((data, (rows, cols)), shape=(n_cells, n_genes))

    # Convert COO matrix to CSR format
    gene_activity_matrix = coo_matrix_all.tocsr()

    aggregation_logger.finish_progress(progress_name='aggregating_results', indent_level=3)

    # Add the sparse gene activity matrix as a new .obsm
    atac_adata.obsm[MDKM.ATAC_GENE_ACTIVITY_KEY] = gene_activity_matrix
    atac_adata.uns[MDKM.ATAC_GENE_ACTIVITY_GENES_KEY] = pd.Index(unique_genes)

    return atac_adata


def integrate(mdata                   ,
              integration_method:      Literal['moscot', 'multivi'] = 'multivi',
              alpha:                   float = 0.5,
              entropic_regularization: float = 0.01,
              gtf_path:                Union[PathLike, str] = None,
              max_epochs:              int = 500,
              lr:                      float = 0.0001,
              ):
    from mudata import MuData
    # Split into scATAC-seq and scRNA-seq AnnData objects
    atac_adata, rna_adata = mdata.mod['atac'].copy(), mdata.mod['rna'].copy()
    atac_adata.obs['modality'], rna_adata.obs['modality'] = 'atac', 'rna'

    if atac_adata.uns[MDKM.MATCHED_ATAC_RNA_DATA_KEY]:
        main_info('Integration: matched multiome, so just filtering cells')

        # Restrict to cells common to both AnnData objects
        shared_cells = pd.Index(np.intersect1d(rna_adata.obs_names, atac_adata.obs_names))
        atac_adata_filtered = atac_adata[shared_cells, :].copy()
        rna_adata_filtered = rna_adata[shared_cells, :].copy()

        return MuData({'atac': atac_adata_filtered, 'rna': rna_adata_filtered})
    elif integration_method == 'moscot':
        return integrate_via_moscot(mdata=mdata,
                                    alpha=alpha,
                                    entropic_regularization=entropic_regularization)
    elif integration_method == 'multivi':
        return integrate_via_multivi(mdata=mdata,
                                     gtf_path=gtf_path,
                                     lr=lr,
                                     max_epochs=max_epochs)
    else:
        raise ValueError(f'Unknown integration method {integration_method} requested.')


def integrate_via_moscot(mdata                   ,
                         alpha:                   float = 0.7,
                         entropic_regularization: float = 0.01,
                         gtf_path:                Union[PathLike, str] = None,
                         ):
    from mudata import MuData
    pass


def integrate_via_multivi(mdata      ,
                          gtf_path:   Union[PathLike, str] = None,
                          lr:         float = 0.0001,
                          max_epochs: int = 500,
                          ):
    import scvi
    main_info('Integration via MULTIVI ...')
    integration_logger = LoggerManager.gen_logger('integration_via_multivi')
    integration_logger.log_time()

    # Split into scATAC-seq and scRNA-seq AnnData objects
    atac_adata, rna_adata = mdata.mod['atac'].copy(), mdata.mod['rna'].copy()
    atac_adata.obs['modality'], rna_adata.obs['modality'] = 'atac', 'rna'

    # Check whether cell indices need to be prepended by 'atac' and 'rna'
    if ':' not in atac_adata.obs_names[0]:
        atac_adata.obs_names = atac_adata.obs_names.map(lambda x: f'atac:{x}')
    num_atac_cells, num_atac_peaks = atac_adata.n_obs, atac_adata.n_vars

    if ':' not in rna_adata.obs_names[0]:
        rna_adata.obs_names = rna_adata.obs_names.map(lambda x: f'rna:{x}')
    num_rna_cells = rna_adata.n_obs

    # Check whether gene activity was pre-computed
    if MDKM.ATAC_GENE_ACTIVITY_KEY not in atac_adata.obsm.keys():
        main_info('Computing gene activities', indent_level=2)
        atac_adata = gene_activity(atac_adata=atac_adata,
                                   gtf_path=gtf_path)
    gene_activity_matrix = atac_adata.obsm[MDKM.ATAC_GENE_ACTIVITY_KEY]

    # Restrict to gene names common to gene activity matrix from atac-seq data and
    # counts matrix from rna-seq data
    gene_names_atac = atac_adata.uns[MDKM.ATAC_GENE_ACTIVITY_GENES_KEY]
    gene_names_rna = rna_adata.var_names
    common_genes = gene_names_rna.intersection(gene_names_atac)
    num_genes = len(common_genes)

    # Filter gene activity and scATAC-seq data into a single AnnData object, with a
    # batch label indicating the origin
    main_info('Preparing ATAC-seq data for MULTIVI', indent_level=2)
    gene_activity_filtered = gene_activity_matrix[:, [gene_names_atac.get_loc(gene) for gene in common_genes]]

    # Assemble multi-ome for the ATAC-seq data
    # ... X
    atac_multiome_X = hstack([gene_activity_filtered, atac_adata.X])

    # ... obs
    atac_multiome_obs = atac_adata.obs[['modality']].copy()

    # ... var
    multiome_var = pd.concat((rna_adata.var.loc[common_genes].copy(), atac_adata.var.copy()), axis=1)

    atac_multiome = AnnData(X=csr_matrix(atac_multiome_X),
                            obs=atac_multiome_obs,
                            var=multiome_var)

    # Assemble multi-ome for RNA-seq data
    main_info('Preparing RNA-seq data for MULTIVI', indent_level=2)
    rna_adata_filtered = rna_adata[:, common_genes].copy()

    # ... X
    rna_multiome_X = hstack([rna_adata_filtered.X.copy(), csr_matrix((num_rna_cells, num_atac_peaks))])

    # ... obs
    rna_multiome_obs = rna_adata_filtered.obs[['modality']].copy()

    # ... var - NTD

    rna_multiome = AnnData(X=csr_matrix(rna_multiome_X),
                           obs=rna_multiome_obs,
                           var=multiome_var)

    # Concatenate the data
    combined_adata = ad.concat([atac_multiome, rna_multiome], axis=0)

    # Setup AnnData object for scvi-tools
    main_info('Setting up combined data for MULTIVI', indent_level=2)
    scvi.model.MULTIVI.setup_anndata(combined_adata, batch_key='modality')

    # Instantiate the SCVI model
    main_info('Instantiating MULTIVI model', indent_level=2)
    multivi_model = scvi.model.MULTIVI(adata=combined_adata, n_genes=num_genes, n_regions=num_atac_peaks)

    # Train the model
    main_info('Training MULTIVI model', indent_level=2)
    multivi_model.train(max_epochs=max_epochs, lr=lr)

    # Extract the latent representation
    combined_adata.obsm['latent'] = multivi_model.get_latent_representation()

    # Impute counts from latent space
    # ... X
    main_info('Imputing RNA expression', indent_level=2)
    imputed_rna_X = multivi_model.get_normalized_expression()

    # ... obs
    multiome_obs = pd.concat((atac_multiome_obs, rna_multiome_obs))

    # ... var
    rna_multiome_var = rna_adata.var.loc[common_genes].copy()

    imputed_rna_adata = AnnData(X=imputed_rna_X,
                                obs=multiome_obs,
                                var=rna_multiome_var,
                                )

    # ... X
    main_info('Imputing accessibility', indent_level=2)
    imputed_atac_X = multivi_model.get_accessibility_estimates()

    # ... obs - NTD

    # ... var
    atac_multiome_var = atac_adata.var.copy()

    imputed_atac_adata = AnnData(X=imputed_atac_X,
                                 obs=multiome_obs,
                                 var=atac_multiome_var,
                                 )

    # Knit together into one harmonized MuData object
    from mudata import MuData
    harmonized_mdata = MuData({'atac': imputed_atac_adata, 'rna': imputed_rna_adata})

    integration_logger.finish_progress(progress_name='integration_via_multivi', indent_level=3)

    return harmonized_mdata


def tfidf_normalize(
        atac_adata:   AnnData,
        log_tf:       bool = True,
        log_idf:      bool = True,
        log_tfidf:    bool = False,
        mv_algorithm: bool = True,
        scale_factor: float = 1e4,
) -> None:
    import muon as mu
    # This computes the term frequency / inverse domain frequency normalization.
    if mv_algorithm:
        # MultiVelo's method
        npeaks = atac_adata.X.sum(1)
        npeaks_inv = csr_matrix(1.0 / npeaks)
        tf = atac_adata.X.multiply(npeaks_inv)
        idf = diags(np.ravel(atac_adata.X.shape[0] / atac_adata.X.sum(0))).log1p()
        tf_idf = tf.dot(idf) * scale_factor
        atac_adata.layers[MDKM.ATAC_TFIDF_LAYER] = np.log1p(tf_idf)
    else:
        atac_adata = mu.atac.pp.tfidf(data=atac_adata,
                                      log_tf=log_tf,
                                      log_idf=log_idf,
                                      log_tfidf=log_tfidf,
                                      scale_factor=scale_factor,
                                      from_layer='counts',
                                      to_layer=MDKM.ATAC_TFIDF_LAYER,
                                      copy=True)

    return atac_adata
