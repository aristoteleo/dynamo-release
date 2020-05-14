from scipy.sparse import issparse
import pandas as pd
from .utils import normalize_data

def scribe(adata,
           genes=None,
           gene_filter_rate=0.1,
           cell_filter_UMI=10000,
           motif_ref='https://www.dropbox.com/s/s8em539ojl55kgf/motifAnnotations_hgnc.csv?dl=1',
           nt_layers=['new', 'total']):
    """Apply Scribe to calculate causal network from spliced/unspliced, metabolic labeling based and other "real" time
    series datasets. Note that this function can be applied to both of the metabolic labeling based single-cell assays with
    newly synthesized and total RNA as well as the regular single cell assays with both the unspliced and spliced
    transcripts. Furthermore, you can also replace the either the new or unspliced RNA with dynamo estimated cell-wise
    velocity, transcription, splicing and degradation rates for each gene (similarly, replacing the expression values
    of transcription factors with RNA binding, ribosome, epigenetics or epitranscriptomic factors, etc.) to infer the
    tottal regulatory effects, transcription, splicing and post-transcriptional regulation of different factors.


    Parameters
    ----------
        adata: :class:`~anndata.AnnData`.
            adata object that includes both newly synthesized and total gene expression of cells. Alternatively,
            the object should include both unspliced and spliced gene expression of cells.
        genes: `List` (default: None)
            The list of gene names that will be used for casual network inference. By default, it is `None` and thus will
            use all genes.
        gene_filter_rate: `float` (default: 0.1)
            minimum percentage of expressed cells for gene filtering.
        cell_filter_UMI: `int` (default: 10000)
             minimum number of UMIs for cell filtering.
        motif_ref: `str` (default: 'https://www.dropbox.com/s/bjuope41pte7mf4/df_gene_TF_link_ENCODE.csv?dl=1')
            It provides the list of TFs gene names and is used to parse the data to get the list of TFs and Targets
            for the causal network inference from those TFs to Targets. But currently the motif based filtering is not implemented.
            By default it is a dropbox link that store the data from us. Other motif reference can bed downloaded from RcisTarget:
            https://resources.aertslab.org/cistarget/. For human motif matrix, it can be downloaded from June's shared folder:
            https://shendure-web.gs.washington.edu/content/members/cao1025/public/nobackup/sci_fate/data/hg19-tss-centered-10kb-7species.mc9nr.feather
       nt_layers:
            The two keys for layers that will be used for the network inference. Note that the layers can be changed
            flexibly. See the description of this function above. The first key corresponds to the transcriptome of the
            next time point, for example unspliced RNAs (or estimated velocitym, see Fig 6 of the Scribe preprint:
            https://www.biorxiv.org/content/10.1101/426981v1) from RNA velocity, old RNA from scSLAM-seq data, etc.
            The second key corresponds to the transcriptome of the initial time point, for example spliced RNAs from RNA
            velocity, old RNA from scSLAM-seq data.

    Returns
    -------
        An updated adata object with a new key `causal_net` in .uns attribute, which stores the inferred causal network.
    """

    try:
        import Scribe
    except ImportError:
        raise ImportError("You need to install the package `Scribe`."
                          "Plelease install from https://github.com/aristoteleo/Scribe-py."
                          "Also check our pape: "
                          "https://www.sciencedirect.com/science/article/abs/pii/S2405471220300363")

    from Scribe.Scribe import causal_net_dynamics_coupling

    motifAnnotations_hgnc = pd.read_csv(motif_ref, sep='\t')
    TF_list = motifAnnotations_hgnc.loc[:, 'TF']

    adata_ = adata.copy()
    n_obs, n_var = adata_.n_obs, adata_.n_vars

    # filter genes
    print(f"Original gene number: {n_var}")

    gene_filter_new = (adata.layers[nt_layers[0]] > 0).sum(0) > (gene_filter_rate * n_obs)
    gene_filter_tot = (adata.layers[nt_layers[1]] > 0).sum(0) > (gene_filter_rate * n_obs)
    if issparse(adata.layers[nt_layers[0]]): gene_filter_new = gene_filter_new.A1
    if issparse(adata.layers[nt_layers[1]]): gene_filter_tot = gene_filter_tot.A1
    adata = adata[:, gene_filter_new * gene_filter_tot]

    print(f"Gene number after filtering: {sum(gene_filter_new * gene_filter_tot)}")

    # filter cells
    print(f"Original cell number: {n_obs}")

    cell_filter = adata.layers[nt_layers[1]].sum(1) > cell_filter_UMI
    if issparse(adata.layers[nt_layers[1]]): cell_filter = cell_filter.A1
    adata = adata[cell_filter, :]

    print(f"Cell number after filtering: {adata.n_obs}")

    # generate the expression matrix for downstream analysis
    new = adata.layers[nt_layers[0]]
    total = adata.layers[nt_layers[1]]

    # recalculate size factor
    from ..preprocessing import szFactor
    adata = szFactor(adata, method='mean-geometric-mean-total', round_exprs=True, total_layers=['total'])
    szfactors = adata.obs["Size_Factor"][:, None]

    # normalize data (size factor correction, log transform and the scaling)
    adata.layers[nt_layers[0]] = normalize_data(new, szfactors, pseudo_expr=0.1)
    adata.layers[nt_layers[1]] = normalize_data(total, szfactors, pseudo_expr=0.1)

    if nt_layers[1] == 'old' and 'old' not in adata.layers.keys():
        adata.layers['old'] = adata.layers['total'] - adata.layers['new'] \
            if 'velocity' not in adata.layers.keys() \
            else adata.layers['total'] - adata.layers['velocity']

    TFs = adata.var_names[adata.var.gene_short_name.isin(TF_list)].to_list()
    Targets = adata.var_names.difference(TFs).to_list()

    if genes is not None:
        TFs = list(set(genes).intersection(TFs))
        Targets = list(set(genes).intersection(Targets))

    causal_net_dynamics_coupling(adata, TFs, Targets, t0_key=nt_layers[1], t1_key=nt_layers[0], normalize=False)
    adata_.uns['causal_net'] = adata.uns['causal_net']

    return adata_
