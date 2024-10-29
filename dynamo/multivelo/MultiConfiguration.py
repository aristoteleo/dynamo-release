from ..configuration import DynamoAdataKeyManager

class MultiDynamoMdataKeyManager(DynamoAdataKeyManager):
    # A class to manage the keys used in MuData object used for MultiDynamo
    # Universal keys - independent of modality
    INFERRED_BATCH_KEY = 'inferred_batch'

    # .mod
    # ... 'atac'
    # ... ... layers
    ATAC_COUNTS_LAYER = 'counts'
    ATAC_FIRST_MOMENT_CHROM_LAYER = 'M_c'
    ATAC_TFIDF_LAYER = 'X_tfidf' # Also X?
    ATAC_CHROMATIN_VELOCITY_LAYER = 'lifted_velo_c'

    # ... ... .obs

    # ... ... .obsm
    ATAC_GENE_ACTIVITY_KEY = 'gene_activity' # Computed gene activity matrix - for unmatched data only
    ATAC_OBSM_LSI_KEY = 'X_lsi'
    ATAC_OBSM_PC_KEY = 'X_pca'

    # ... ... .obsp

    # ... ... .uns
    ATAC_GENE_ACTIVITY_GENES_KEY = 'gene_activity_genes' # Genes for gene activity matrix
    MATCHED_ATAC_RNA_DATA_KEY = 'matched_atac_rna_data' # Indicates whether ATAC- and RNA-seq data are matched

    # ... ... .var (atac:*)

    # ... ... .varm
    ATAC_VARM_LSI_KEY = 'LSI'

    # ... 'cite'
    # ... ... layers

    # ... ... .obs

    # ... ... .obsm

    # ... ... .obsp

    # ... ... .uns
    MATCHED_CITE_RNA_DATA_KEY = 'matched_cite_rna_data' # Indicates whether CITE- and RNA-seq data are matched

    # ... ... .var (cite:*)

    # ... ... .varm

    # ... 'hic'
    # ... ... layers

    # ... ... .obs

    # ... ... .obsm

    # ... ... .obsp

    # ... ... .uns
    MATCHED_HIC_RNA_DATA_KEY = 'matched_hic_rna_data' # Indicates whether HiC- and RNA-seq data are matched

    # ... ... .var (hic:*)

    # ... ... .varm

    # ... 'rna'
    # Most things are handled by DynamoAdataKeyManager; these are in addition to thos defined in dynamo
    # ... ... layers
    RNA_COUNTS_LAYER = 'counts'
    RNA_COUNTS_LAYER_FROM_LOOM = 'matrix'
    RNA_FIRST_MOMENT_CHROM_LAYER = 'M_c'
    RNA_FIRST_MOMENT_SPLICED_LAYER = 'M_s'
    RNA_FIRST_MOMENT_UNSPLICED_LAYER = 'M_u'
    RNA_SECOND_MOMENT_SS_LAYER = 'M_ss'
    RNA_SECOND_MOMENT_US_LAYER = 'M_us'
    RNA_SECOND_MOMENT_UU_LAYER = 'M_uu'
    RNA_SPLICED_LAYER = 'spliced'
    RNA_SPLICED_VELOCITY_LAYER = 'velocity_S'
    RNA_UNSPLICED_LAYER = 'unspliced'

    # ... ... .obs

    # ... ... .obsm
    RNA_OBSM_PC_KEY = 'X_pca'

    # ... ... .obsp

    # ... ... .uns

    # ... ... .var (rna:*)

    # ... ... .varm

    def bogus_function(self):
        pass

MDKM = MultiDynamoMdataKeyManager
