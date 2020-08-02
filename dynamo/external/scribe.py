from scipy.sparse import issparse
import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
import itertools
from tqdm import tqdm
from .utils import normalize_data, TF_link_gene_chip
from ..tools.utils import flatten, einsum_correlation

def scribe(adata,
           genes=None,
           TFs=None,
           Targets=None,
           gene_filter_rate=0.1,
           cell_filter_UMI=10000,
           motif_ref='https://www.dropbox.com/s/s8em539ojl55kgf/motifAnnotations_hgnc.csv?dl=1',
           nt_layers=['X_new', 'X_total'],
           normalize=True,
           do_CLR=True,
           drop_zero_cells=True,
           TF_link_ENCODE_ref='https://www.dropbox.com/s/bjuope41pte7mf4/df_gene_TF_link_ENCODE.csv?dl=1',
           ):
    """Apply Scribe to calculate causal network from spliced/unspliced, metabolic labeling based and other "real" time
    series datasets. Note that this function can be applied to both of the metabolic labeling based single-cell assays with
    newly synthesized and total RNA as well as the regular single cell assays with both the unspliced and spliced
    transcripts. Furthermore, you can also replace the either the new or unspliced RNA with dynamo estimated cell-wise
    velocity, transcription, splicing and degradation rates for each gene (similarly, replacing the expression values
    of transcription factors with RNA binding, ribosome, epigenetics or epitranscriptomic factors, etc.) to infer the
    total regulatory effects, transcription, splicing and post-transcriptional regulation of different factors.


    Parameters
    ----------
        adata: :class:`~anndata.AnnData`.
            adata object that includes both newly synthesized and total gene expression of cells. Alternatively,
            the object should include both unspliced and spliced gene expression of cells.
        genes: `List` (default: None)
            The list of gene names that will be used for casual network inference. By default, it is `None` and thus will
            use all genes.
        TFs: `List` or `None` (default: None)
            The list of transcription factors that will be used for casual network inference. When it is `None` gene list
            included in the file linked by `motif_ref` will be used.
        Targets: `List` or `None` (default: None)
            The list of target genes that will be used for casual network inference. When it is `None` gene list not
            included in the file linked by `motif_ref` will be used.
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
        nt_layers: `List` (Default: ['X_new', 'X_total'])
            The two keys for layers that will be used for the network inference. Note that the layers can be changed
            flexibly. See the description of this function above. The first key corresponds to the transcriptome of the
            next time point, for example unspliced RNAs (or estimated velocitym, see Fig 6 of the Scribe preprint:
            https://www.biorxiv.org/content/10.1101/426981v1) from RNA velocity, old RNA from scSLAM-seq data, etc.
            The second key corresponds to the transcriptome of the initial time point, for example spliced RNAs from RNA
            velocity, old RNA from scSLAM-seq data.
        drop_zero_cells: `bool` (Default: False)
            Whether to drop cells that with zero expression for either the potential regulator or potential target. This
            can signify the relationship between potential regulators and targets, speed up the calculation, but at the risk
            of ignoring strong inhibition effects from certain regulators to targets.
        do_CLR: `bool` (Default: True)
            Whether to perform context likelihood relatedness analysis on the reconstructed causal network
        TF_link_ENCODE_ref: `str` (default: 'https://www.dropbox.com/s/s8em539ojl55kgf/motifAnnotations_hgnc.csv?dl=1')
            The path to the TF chip-seq data. By default it is a dropbox link from us that stores the data. Other data can
            be downloaded from: https://amp.pharm.mssm.edu/Harmonizome/dataset/ENCODE+Transcription+Factor+Targets.

    Returns
    -------
        An updated adata object with a new key `causal_net` in .uns attribute, which stores the inferred causal network.
    """

    try:
        import Scribe
    except ImportError:
        raise ImportError("You need to install the package `Scribe`."
                          "Plelease install from https://github.com/aristoteleo/Scribe-py."
                          "Also check our paper: "
                          "https://www.sciencedirect.com/science/article/abs/pii/S2405471220300363")

    from Scribe.Scribe import causal_net_dynamics_coupling, CLR

    # detect format of the gene name:
    str_format = "upper" if adata.var_names[0].isupper() else 'lower' \
        if adata.var_names[0].islower() else "title" \
        if adata.var_names[0].istitle() else "other"

    motifAnnotations_hgnc = pd.read_csv(motif_ref, sep='\t')
    TF_list = motifAnnotations_hgnc.loc[:, 'TF'].values
    if str_format == "title":
        TF_list = [i.capitalize() for i in TF_list]
    elif str_format == 'lower':
        TF_list = [i.lower() for i in TF_list]

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
    if adata.n_obs == 0:
        raise Exception('No cells remaining after filtering, try relaxing `cell_filtering_UMI`.')

    print(f"Cell number after filtering: {adata.n_obs}")

    # generate the expression matrix for downstream analysis
    if nt_layers[1] == 'old' and 'old' not in adata.layers.keys():
        adata.layers['old'] = adata.layers['total'] - adata.layers['new'] \
            if 'velocity' not in adata.layers.keys() \
            else adata.layers['total'] - adata.layers['velocity']

    new = adata.layers[nt_layers[0]]
    total = adata.layers[nt_layers[1]]

    if normalize:
        # recalculate size factor
        from ..preprocessing import szFactor
        adata = szFactor(adata, method='mean-geometric-mean-total', round_exprs=True, total_layers=['total'])
        szfactors = adata.obs["Size_Factor"][:, None]

        # normalize data (size factor correction, log transform and the scaling)
        adata.layers[nt_layers[0]] = normalize_data(new, szfactors, pseudo_expr=0.1)
        adata.layers[nt_layers[1]] = normalize_data(total, szfactors, pseudo_expr=0.1)

    TFs = adata.var_names[adata.var.index.isin(TF_list)].to_list() if TFs is None else np.unique(TFs)

    Targets = adata.var_names.difference(TFs).to_list() if Targets is None else np.unique(Targets)

    if genes is not None:
        TFs = list(set(genes).intersection(TFs))
        Targets = list(set(genes).intersection(Targets))

    if len(TFs) == 0 or len(Targets) == 0:
        raise Exception('The TFs or Targets are empty! Something (input TFs/Targets list, gene_filter_rate, etc.) is wrong.')

    print(f"Potential TFs are: {len(TFs)}")
    print(f"Potential Targets are: {len(Targets)}")

    causal_net_dynamics_coupling(adata, TFs, Targets, t0_key=nt_layers[1], t1_key=nt_layers[0], normalize=False,
                                 drop_zero_cells=drop_zero_cells)
    res_dict = {"RDI": adata.uns['causal_net']["RDI"]}
    if do_CLR: res_dict.update({"CLR": CLR(res_dict['RDI'])})

    if TF_link_ENCODE_ref is not None:
        df_gene_TF_link_ENCODE = pd.read_csv(TF_link_ENCODE_ref, sep='\t')
        df_gene_TF_link_ENCODE['id_gene'] = df_gene_TF_link_ENCODE['id'].astype('str') + '_' + \
                                            df_gene_TF_link_ENCODE['linked_gene_name'].astype('str')

        df_gene = pd.DataFrame(adata.var.index, index=adata.var.index)
        df_gene.columns = ['linked_gene']

        net = res_dict[list(res_dict.keys())[-1]]
        net = net.reset_index().melt(id_vars='index', id_names='id', var_name='linked_gene', value_name='corcoef')
        net_var = net.merge(df_gene)
        net_var['id_gene'] = net_var['id'].astype('str') + '_' + \
                             net_var['linked_gene_name'].astype('str')

        filtered = TF_link_gene_chip(net_var, df_gene_TF_link_ENCODE, adata.var, cor_thresh=0.02)
        res_dict.update({"filtered": filtered})

    adata_.uns['causal_net'] = res_dict

    return adata_


def mutual_inform(adata, genes, layer_x, layer_y, cores=1):
    """Calculate mutual information (as well as pearson correlation) of genes between two different layers.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`.
            adata object that will be used for mutual information calculation.
        genes: `List` (default: None)
            Gene names from the adata object that will be used for mutual information calculation.
        layer_x
            The first key of the layer from the adata object that will be used for mutual information calculation.
        layer_y
            The second key of the layer from the adata object that will be used for mutual information calculation.
        cores: `int` (default: 1)
            Number of cores to run the MI calculation. If cores is set to be > 1, multiprocessing will be used to
            parallel the calculation.

    Returns
    -------
        An updated adata object that updated with a new columns (`mi`, `pearson`) in .var contains the mutual information
        of input genes.
    """

    try:
        import Scribe
    except ImportError:
        raise ImportError("You need to install the package `Scribe`."
                          "Plelease install from https://github.com/aristoteleo/Scribe-py."
                          "Also check our paper: "
                          "https://www.sciencedirect.com/science/article/abs/pii/S2405471220300363")

    from Scribe.information_estimators import mi

    adata.var['mi'], adata.var['pearson'] = np.nan, np.nan

    mi_vec, pearson = np.zeros(len(genes)), np.zeros(len(genes))
    X, Y = adata[:, genes].layers[layer_x], adata[:, genes].layers[layer_y]
    X, Y = X.A if issparse(X) else X, Y.A if issparse(Y) else Y

    k = min(5, int(adata.n_obs / 5 + 1))
    if cores == 1:
        for i in tqdm(range(len(genes)), desc=f'calculating mutual information between {layer_x} and {layer_y} data'):
            x, y = X[i], Y[i]
            mask = np.logical_and(np.isfinite(x), np.isfinite(y))
            pearson[i] = einsum_correlation(x[None, mask], y[mask], type="pearson")
            x, y = [[i] for i in x[mask]], [[i] for i in y[mask]]

            mi_vec[i] = mi(x, y, k=k)
    else:
        for i in tqdm(range(len(genes)), desc=f'calculating mutual information between {layer_x} and {layer_y} data'):
            x, y = X[i], Y[i]
            mask = np.logical_and(np.isfinite(x), np.isfinite(y))
            pearson[i] = einsum_correlation(x[None, mask], y[mask], type="pearson")

        def pool_mi(x, y, k):
            mask = np.logical_and(np.isfinite(x), np.isfinite(y))
            x, y = [[i] for i in x[mask]], [[i] for i in y[mask]]

            return mi(x, y, k)

        pool = ThreadPool(cores)
        res = pool.starmap(pool_mi, zip(X, Y, itertools.repeat(k)))
        pool.close()
        pool.join()
        mi_vec = np.array(res)

    adata.var.loc[genes, 'mi'] = mi_vec
    adata.var.loc[genes, 'pearson'] = pearson



