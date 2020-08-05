# integrate with Scribe (Qiu, et. al, Cell Systems, 2020) next
# the following code is based on Cao, et. al, Nature Biotechnology, 2020 and
# https://github.com/JunyueC/sci-fate_analysis

from tqdm import tqdm
import pandas as pd
import numpy as np
from anndata.utils import make_index_unique
from scipy.sparse import issparse
from .utils import normalize_data, TF_link_gene_chip
from ..tools.utils import einsum_correlation

def scifate_glmnet(adata,
                   gene_filter_rate=0.1,
                   cell_filter_UMI=10000,
                   core_n_lasso=1,
                   core_n_filtering=1,
                   motif_ref='https://www.dropbox.com/s/s8em539ojl55kgf/motifAnnotations_hgnc.csv?dl=1',
                   TF_link_ENCODE_ref='https://www.dropbox.com/s/bjuope41pte7mf4/df_gene_TF_link_ENCODE.csv?dl=1',
                   nt_layers=['X_new', 'X_total']):
    """Reconstruction of regulatory network (Cao, et. al, Nature Biotechnology, 2020) from TFs to other target
     genes via LASSO regression between the total expression of known transcription factors and the newly synthesized
     RNA of potential targets. The inferred regulatory relationships between TF and targets are further filtered based
     on evidence of promoter motifs (not implemented currently) and the ENCODE chip-seq peaks. The python wrapper for the
     glmnet FORTRON code, glm-python (https://github.com/bbalasub1/glmnet_python) was used. More details on lasso regression
     with glm-python can be found here (https://github.com/bbalasub1/glmnet_python/blob/master/test/glmnet_examples.ipynb).
     Note that this function can be applied to both of the metabolic labeling single-cell assays with newly synthesized
     and total RNA as well as the regular single cell assays with both the unspliced and spliced transcripts. Furthermore,
     you can also replace the either the new or unspliced RNA with dynamo estimated cell-wise velocity, transcription,
     splicing and degradation rates for each gene (similarly, replacing the expression values of transcription factors with RNA binding,
     ribosome, epigenetics or epitranscriptomic factors, etc.) to infer the tottal regulatory effects, transcription, splicing
     and post-transcriptional regulation of different factors. In addition, this approach will be fully integrated with
     Scribe (Qiu, et. al, 2020) which employs restricted directed information to determine causality by estimating the
     strength of information transferred from a potential regulator to its downstream target. In contrast of lasso regression,
     Scribe can learn both linear and non-linear causality in deterministic and stochastic systems. It also incorporates
     rigorous procedures to alleviate sampling bias and builds upon novel estimators and regularization techniques to
     facilitate inference of large-scale causal networks.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`.
            adata object that includes both newly synthesized and total gene expression of cells. Alternatively,
            the object should include both unspliced and spliced gene expression of cells.
        gene_filter_rate: `float` (default: 0.1)
            minimum percentage of expressed cells for gene filtering.
        cell_filter_UMI: `int` (default: 10000)
            minimum number of UMIs for cell filtering.
        core_n_lasso: `int` (default: 1)
            number of cores for lasso regression in linkage analysis. By default, it is 1 and parallel is turned off.
            Parallel computing can significantly speed up the computation process, especially for datasets involve
            many cells or genes. But for smaller datasets or genes, it could result in a reduction in speed due to the
            additional overhead. User discretion is advised.
        core_n_filtering: `int` (default: 1)
            number of cores for filtering TF-gene links. Not used currently.
        motif_ref: `str` (default: 'https://www.dropbox.com/s/bjuope41pte7mf4/df_gene_TF_link_ENCODE.csv?dl=1')
            The path to the TF binding motif data as described above. It provides the list of TFs gene names and
            is used to process adata object to generate the TF expression and target new expression matrix for glmnet
            based TF-target synthesis rate linkage analysis. But currently it is not used for motif based filtering.
            By default it is a dropbox link that store the data from us. Other motif reference can bed downloaded from RcisTarget:
            https://resources.aertslab.org/cistarget/. For human motif matrix, it can be downloaded from June's shared folder:
            https://shendure-web.gs.washington.edu/content/members/cao1025/public/nobackup/sci_fate/data/hg19-tss-centered-10kb-7species.mc9nr.feather
        TF_link_ENCODE_ref: `str` (default: 'https://www.dropbox.com/s/s8em539ojl55kgf/motifAnnotations_hgnc.csv?dl=1')
            The path to the TF chip-seq data. By default it is a dropbox link from us that stores the data. Other data can
            be downloaded from: https://amp.pharm.mssm.edu/Harmonizome/dataset/ENCODE+Transcription+Factor+Targets.
        nt_layers: `list([str, str])` (default: `['X_new', 'X_total']`)
            The layers that will be used for the network inference. Note that the layers can be changed flexibly. See
            the description of this function above.

        Note that if your internet connection is slow, we recommend to download the `motif_ref` and `TF_link_ENCODE_ref` and
        supplies those two arguments with the local paths where the downloaded datasets are saved.

    Returns
    -------
        An updated adata object with a new key `scifate` in .uns attribute, which stores the raw lasso regression results
        and the filter results after applying the Fisher exact test of the ChIP-seq peaks.
    """

    try:
        import glmnet_python
        global cvglmnet, cvglmnetCoef
        from glmnet_python import cvglmnet, cvglmnetCoef
    except ImportError:
        raise ImportError("You need to install the package `glmnet_python`."
                          "The original version https://github.com/bbalasub1/glmnet_python is not updated."
                          "Plelease install johnlees's fork https://github.com/johnlees/glmnet_python."
                          "Also check this pull request for more details: "
                          "https://github.com/bbalasub1/glmnet_python/pull/47")

    df_gene_TF_link_ENCODE = pd.read_csv(TF_link_ENCODE_ref, sep='\t')
    motifAnnotations_hgnc = pd.read_csv(motif_ref, sep='\t')
    TF_list = motifAnnotations_hgnc.loc[:, 'TF']

    _, new_mat, obs, var, var_TF, TF_matrix = adata_processing_TF_link(adata, nt_layers, TF_list, gene_filter_rate, cell_filter_UMI)

    # Apply LASSO for TF-gene linkage analysis
    new_size_factor = obs.loc[:, 'Size_Factor']
    new_size_factor = (new_size_factor - np.mean(new_size_factor)) / np.std(new_size_factor)

    TF_matrix.loc['new_size_factor'] = new_size_factor
    labeling_rate = obs['labeling_rate']
    if (np.std(labeling_rate) != 0):
        labeling_ratio = (labeling_rate - np.mean(labeling_rate)) / np.std(labeling_rate)
        TF_matrix.loc['labeling_ratio', :] = labeling_ratio

    # link TF and genes based on covariance
    link_result = link_TF_gene_analysis(TF_matrix, new_mat, var_TF, core_num=core_n_lasso)
    link_result = pd.concat([i if i != 'unknown' else None for i in link_result], axis=0)

    # filtering the links using TF-gene binding data and store the result in the target folder
    # note that currently the motif filtering is not implement
    df_gene_TF_link_chip = TF_gene_filter_links(link_result, var, core_n_filtering, motif_ref, df_gene_TF_link_ENCODE)

    adata.uns['scifate'] = {'glmnet_res': link_result, "glmnet_chip_filter": df_gene_TF_link_chip}

    return adata


def adata_processing_TF_link(adata, nt_layers, TF_list, gene_filter_rate=0.1, cell_filter_UMI=10000):
    """preprocess adata and get ready for TF-target gene analysis"""

    n_obs, n_var = adata.n_obs, adata.n_vars

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
    if issparse(new): new = new.A
    if issparse(total): total = total.A
    new_mat = normalize_data(new, szfactors, pseudo_expr=0.1)
    tot_mat = normalize_data(total, szfactors, pseudo_expr=0.1)
    new_mat = pd.DataFrame(new_mat, index=adata.obs_names, columns=adata.var_names)
    tot_mat = pd.DataFrame(tot_mat, index=adata.obs_names, columns=adata.var_names)

    # compute the labeling reads rate in each cell
    obs = adata.obs
    var = adata.var
    var.loc[:, 'gene_short_name'] = make_index_unique(var.loc[:, 'gene_short_name'].astype('str'))
    obs.loc[:, 'labeling_rate'] = adata.layers['new'].sum(1) / adata.layers['total'].sum(1)

    # extract the TF matrix
    var_TF = var.query("gene_short_name in @TF_list")
    print(f"\nNumber of TFs found in the list: {var_TF.shape[0]}")

    TF_matrix = tot_mat.loc[:, var_TF.loc[:, 'gene_id']]
    TF_matrix.columns = var_TF.loc[:, 'gene_short_name']

    return (tot_mat.T, new_mat.T, obs, var, var_TF, TF_matrix.T)


def link_TF_gene_analysis(TF_matrix, gene_matrix, var_TF, core_num=1, cor_thresh=0.03, seed=123456):
    """Perform lasso regression for each gene."""

    gene_list = gene_matrix.index
    link_result = [None] * len(gene_list)

    for i, linked_gene in tqdm(enumerate(gene_list), desc=f"Perform lasso regression for each gene:"):
        gene_name, TFs = var_TF.loc[linked_gene, 'gene_short_name'] if linked_gene in var_TF.index else None, TF_matrix.index
        valid_gene_ids = TFs if gene_name is None else list(TFs.difference(set(gene_name)))
        input_TF_matrix = TF_matrix.loc[valid_gene_ids, :]

        result = TF_gene_link(input_TF_matrix, linked_gene, gene_matrix.loc[linked_gene, :], core_num=core_num,
                              cor_thresh=cor_thresh, seed=seed)
        link_result[i] = result

    return link_result


def TF_gene_link(TF_matrix, linked_gene, linked_gene_expr_vector, core_num=1, cor_thresh=0.03, seed=123456):
    """Estimate the regulatory weight of each TF to its potential targets via lasso regression for each gene."""

    expr_cor = einsum_correlation(TF_matrix.values, linked_gene_expr_vector.values)[0]
    select_sites = abs(expr_cor) > cor_thresh
    TF_matrix = TF_matrix.iloc[select_sites, :]

    if select_sites.sum() > 1:
        TF_matrix= TF_matrix.T
        result = lasso_regression_expression(TF_matrix, linked_gene, linked_gene_expr_vector, seed, core_num)

        return (result)
    else:
        return "unknown"


def lasso_regression_expression(x1, linked_gene, y, seed, parallel=1):
    """Lasso linear model with iterative fitting along a regularization path. Select best model is by cross-validation."""

    x1 = x1.loc[:, ~x1.columns.duplicated(keep='first')]

    # the following code should match up that from cv.glmnet in R if the foldid is hard coded to be the same.
    # https://github.com/bbalasub1/glmnet_python/issues/45

    np.random.seed(seed)
    seq = np.linspace(np.log(0.001), np.log(10), 100)

    # alpha = 1 is required for lasso regression.
    cv_out = cvglmnet(x=x1.values, y=y.values, ptype='mse', alpha=1, lambdau=np.exp(seq), parallel=parallel)
    r2 = r2_glmnet(cv_out, y)

    bestlam = cv_out['lambda_1se']

    # this rescaling ensures returning very similar results to R's cv.glm
    cor_list = cvglmnetCoef(cv_out, s=bestlam).flatten() / np.sqrt(x1.shape[0])
    df_cor = pd.DataFrame({"id": x1.columns,
                           "corcoef": cor_list[1:], # ignore the first intercept term
                           "r_squre": r2,
                           "linked_gene": linked_gene})

    return df_cor


def r2_glmnet(cv_out, y):
    """calculate r2 using the lambda_1se. This value is for the most regularized model whose mean squared error is
    within one standard error of the minimal."""

    # https://stackoverflow.com/questions/50610895/how-to-calculate-r-squared-value-for-lasso-regression-using-glmnet-in-r
    bestlam = cv_out['lambda_1se']
    i = np.where(cv_out['lambdau'] == bestlam)[0]
    e = cv_out['cvm'][i]
    r2 = 1 - e / np.var(y)
    if r2 < 0:
       r2 = [0]

    return r2[0]


def TF_gene_filter_links(raw_glmnet_res,
                         var,
                         core_n,
                         motif_ref,
                         df_gene_TF_link_ENCODE):
    """prepare data for TF-target gene link filtering"""

    var_ori = var.copy()
    df_gene = var.loc[:, ['gene_id', 'gene_short_name', 'gene_type']]
    df_gene.columns = ['linked_gene', 'linked_gene_name', 'gene_type']

    raw_glmnet_res_var = raw_glmnet_res.merge(df_gene)
    raw_glmnet_res_var = raw_glmnet_res_var.query("id != 'labeling_ratio' and id != 'new_size_factor'")

    # filter the link by ChIP-seq analysis
    df_gene_TF_link_chip = TF_link_gene_chip(raw_glmnet_res_var, df_gene_TF_link_ENCODE, var_ori, cor_thresh=0.04)

    return df_gene_TF_link_chip

