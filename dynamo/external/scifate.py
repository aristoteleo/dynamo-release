# integrate with Scribe next
# the following code is based on Cao, et. al, Nature Biotechnology, 2020

import pandas as pd
import numpy as np
from anndata.utils import make_index_unique
import scipy.stats as stats
from scipy.sparse import issparse
from glmnet_python import cvglmnet, cvglmnetCoef
from ..tools.utils import einsum_correlation


def scifate_glmnet(adata, TF_list, gene_filter_rate, cell_filter_UMI, output_folder,
                   core_n_lasso, core_n_filtering, motif_ref, df_gene_TF_link_ENCODE):
    """
    ln /usr/local/opt/gcc/lib/gcc/5/libgfortran.3.dylib /Library/Frameworks/R.framework/Versions/3.6/Resources/lib/libgfortran.3.dylib
    Parameters
    ----------
    cds_new:
        monocle2 cds object for newly synthesized gene expression of cells
    TF_list:
        gene names of TFs for linkage analysis
    output_links_new_RNA:
        output folder
    gene_filter_rate:
        minimum percentage of expressed cells for gene filtering
    cell_filter_UMI:
        minimum number of UMIs for cell filtering
    core_n_lasso:
        number of cores for lasso regression in linkage analysis
    core_n_filtering:
        number of cores for filtering TF-gene links
    motif_ref:
        TF binding motif data as described above
    df_gene_TF_link_ENCODE:
        TF chip-seq data as described above process the cds files to generate the TF expression and gene expression matrix for linkage analysis

    Returns
    -------

    """

    input_data = cds_processing_TF_link(adata, TF_list, gene_filter_rate, cell_filter_UMI)
    # Apply LASSO for TF-gene linkage analysis
    mm_1 = input_data[0]
    mm_2 = input_data[1]
    df_cell = input_data[2]
    df_gene = input_data[3]
    df_gene_TF = input_data[4]
    TF_matrix = input_data[5]
    new_size_factor = df_cell.loc[:, 'Size_Factor']
    new_size_factor = (new_size_factor - np.mean(new_size_factor)) / np.std(new_size_factor)

    tmp = TF_matrix.T
    tmp.columns = tmp.columns.astype('str')
    tmp.loc[:, 'new_size_factor'] = new_size_factor
    labeling_rate = df_cell.loc[:, 'labeling_rate']
    if (np.std(labeling_rate) != 0):
        labeling_ratio = (labeling_rate - np.mean(labeling_rate)) / np.std(labeling_rate)
        tmp.loc[:, 'labeling_ratio']= labeling_ratio

    tmp = tmp.T

    # link TF and genes based on covariance
    link_result = link_TF_gene_analysis(tmp, mm_2, df_gene_TF, core_num=core_n_lasso)
    link_result = pd.concat(link_result, axis=0)

    # filtering the links using TF-gene binding data and store the result in the target folder
    # note that currently the motif filtering is not implement
    df_gene_TF_link_chip = TF_gene_filter_links(link_result, df_gene, output_folder, core_n_filtering, motif_ref, df_gene_TF_link_ENCODE)

    return df_gene_TF_link_chip

def cds_processing_TF_link(adata, TF_list, gene_filter_rate=0.1, cell_filter_UMI=10000):
    n_obs, n_var = adata.n_obs, adata.n_vars
    gene_filter_1 = (adata.layers['total'] > 0).sum(0) > (gene_filter_rate * n_obs)
    gene_filter_2 = (adata.layers['new'] > 0).sum(0) > (gene_filter_rate * n_obs)
    if issparse(adata.layers['total']): gene_filter_1 = gene_filter_1.A1
    if issparse(adata.layers['new']): gene_filter_2 = gene_filter_2.A1
    print(f"Original gene number: {n_var}")
    print(f"Gene number after filtering: {sum(gene_filter_1 * gene_filter_2)}")

    print(f"Original cell number: {n_obs}")
    adata = adata[:, gene_filter_2 * gene_filter_1]
    cell_filter = (adata.layers['total'].sum(1) > cell_filter_UMI)
    if issparse(adata.layers['total']): cell_filter = cell_filter.A1
    adata = adata[cell_filter, :]
    cds_all = adata.layers['total']
    cds_new = adata.layers['new']
    print(f"Cell number after filtering: {cds_all.shape[0]}")

    # generate the expression matrix for downstream analysis
    from ..preprocessing import szFactor
    adata = szFactor(adata, method='mean-geometric-mean-total', round_exprs=True)
    szfactors = adata.obs["Size_Factor"][:, None]

    if issparse(cds_all): cds_all = cds_all.A
    if issparse(cds_new): cds_new = cds_new.A
    mm_1 = normalize_data(cds_all, szfactors, pseudo_expr=0.1)
    mm_2 = normalize_data(cds_new, szfactors, pseudo_expr=0.1)
    mm_1 = pd.DataFrame(mm_1, index=adata.obs_names, columns=adata.var_names)
    mm_2 = pd.DataFrame(mm_2, index=adata.obs_names, columns=adata.var_names)

    # compute the labeling reads rate in each cell
    df_cell = adata.obs
    df_gene = adata.var
    df_gene.loc[:, 'gene_short_name'] = make_index_unique(df_gene.loc[:, 'gene_short_name'].astype('str'))
    df_cell.loc[:, 'labeling_rate'] = adata.layers['new'].sum(1) / adata.layers['total'].sum(1)

    # extract the TF matrix
    df_gene_TF = df_gene.query("gene_short_name in @TF_list")
    print(f"\nNumber of TFs found in the list: {df_gene_TF.shape[0]}")

    TF_matrix = mm_1.loc[:, df_gene_TF.loc[:, 'gene_id']]
    TF_matrix.columns = df_gene_TF.loc[:, 'gene_short_name']

    return (mm_1.T, mm_2.T, df_cell, df_gene, df_gene_TF, TF_matrix.T)


def link_TF_gene_analysis(TF_matrix, gene_matrix, df_gene_TF, core_num = 10, cor_thresh = 0.03, seed = 123456):
    gene_list = gene_matrix.index
    link_result = [None] * len(gene_list)

    for i, linked_gene in enumerate(gene_list):
        print('i, x is ', i, linked_gene)
        gene_name, TFs = df_gene_TF.loc[linked_gene, 'gene_id'] if linked_gene in df_gene_TF.index else None, TF_matrix.index
        valid_gene_ids = TFs if gene_name is None else list(TFs.difference(set(gene_name)))
        input_TF_matrix = TF_matrix.loc[valid_gene_ids, :]

        result = TF_gene_link(input_TF_matrix, linked_gene, gene_matrix.loc[linked_gene, :], cor_thresh = cor_thresh, seed = seed)
        link_result[i] = result

    return link_result


def TF_gene_link(TF_matrix, linked_gene, gene_expr_vector, cor_thresh = 0.03, seed = 123456):
    TF_MM = TF_matrix
    expr_cor = einsum_correlation(TF_MM.values, gene_expr_vector.values)[0]
    select_sites = abs(expr_cor) > cor_thresh
    TF_MM.iloc[~ select_sites, :] = 0

    find_target = False

    if select_sites.sum() > 1:
        find_target = True
        TF_MM= TF_MM.T
        TF_MM = TF_MM.loc[:, select_sites]
        target_num = sum(select_sites)
        # TF_MM['expression'] = gene_expr_vector
        result = lasso_regression_expression(TF_MM, linked_gene, gene_expr_vector, seed)

        return (result)
    # cat("\n number of distal vector: ", distal.num)
    else:
        return "unknown"


def lasso_regression_expression(x1, linked_gene, y, seed, parallel=1):
    # cat("dimension for ml_matrix: ", dim(ml_matrix))
    # there are duplicated points
    x1 = x1.loc[:, ~x1.columns.duplicated(keep='first')]
    np.random.seed(seed)
    seq = np.linspace(np.log(0.001), np.log(10), 100)
    # the following result doesn't match cv.glmnet in R
    cv_out1 = cvglmnet(x=x1.values, y=y.values, ptype='mse', alpha=1, lambdau=np.exp(seq), parallel=parallel)
    r2_1 = r2_glmnet(cv_out1, y)

    bestlam = cv_out1['lambda_1se']
    print('lambda_1se is', bestlam)
    # this ensures returning very similar results to R's cv.glm
    cor_list = cvglmnetCoef(cv_out1, s=bestlam).flatten() / np.sqrt(x1.shape[0])
    df_cor = pd.DataFrame({"id": x1.columns,
                           "corcoef": cor_list[1:],
                           "r_squre": r2_1,
                           "linked_gene": linked_gene})
    return df_cor


def r2_glmnet(cv_out, y):
    # https://stackoverflow.com/questions/50610895/how-to-calculate-r-squared-value-for-lasso-regression-using-glmnet-in-r
    bestlam = cv_out['lambda_1se']
    i = np.where(cv_out['lambdau'] == bestlam)[0]
    e = cv_out['cvm'][i]
    r2 = 1 - e / np.var(y)
    if r2 < 0:
       r2 = [0]

    return r2[0]


def TF_gene_filter_links(df_link_file, df_gene, output_folder, core_n = 1,
                                 motif_ref = "~/Projects/processed_data/181103_sciMM/data/Main_A549/Gene_prediction/hg19-tss-centered-10kb-7species.mc9nr.feather",
                                 df_gene_TF_link_ENCODE = "~/Projects/processed_data/181103_sciMM/data/Main_A549/Gene_prediction/df_TF_gene_ENCODE.RData"):
    cds = df_gene
    # df_gene_TF_link_ENCODE  = pd.read_csv(df_gene_TF_link_ENCODE)
    link_new_2 = df_link_file
    df_gene = cds.loc[:, ['gene_id', 'gene_short_name']]
    df_gene.columns = ['linked_gene', 'linked_gene_name']

    df_link_new_2 = link_new_2.merge(df_gene)
    df_link_new_2 = df_link_new_2.query("id != 'labeling_ratio' and id != 'new_size_factor'")
    df_gene = cds.loc[:, ['gene_id', 'gene_type']] #
    df_gene.columns= ['linked_gene', 'gene_type']
    df_link_new_2 = df_link_new_2.merge(df_gene)

    df_link = df_link_new_2

    # filter the link by TF motif analysis
    df_gene_TF_link_chip = TF_link_gene_chip(df_link, df_gene_TF_link_ENCODE, cds, cor_thresh=0.04)
    # saveRDS(df_gene_TF_link_chip, file=file.path(output_folder, "df_gene_TF_link_Chip.RDS"))

    return df_gene_TF_link_chip


def link_TF_gene_summary(link_result):
    gene_name = link_result[0]
    tmp_linked = [None] * len(gene_name)
    for x, cur_gene_name in enumerate(gene_name):
        print('x, cur_gene_name is ', x, cur_gene_name)
        if len((link_result[1])[x]) == 1:
            tmp_linked[x] = None
        else:
            df_tmp = (link_result[1])[x][1].copy()
            df_tmp['r_squre'] = (link_result[1])[x][0]
            df_tmp['linked_gene'] = gene_name[x]
            tmp_linked[x] = df_tmp

    return tmp_linked


def TF_link_gene_chip(df_link, df_gene_TF_link_ENCODE, df_gene, cor_thresh = 0.04):
    df_link_new = df_link.query("abs(corcoef) > @cor_thresh")
    print(f"\n Number of possible links: df_link_new.shape[0]")
    # df_gene_TF_link_ENCODE['id_gene'] = df_gene_TF_link_ENCODE[['id', 'linked_gene_name']].agg('_'.join, axis=1)
    # df_gene_TF_link_ENCODE[['id', 'linked_gene_name']].apply(lambda x: ''.join(x), axis=1)
    df_gene_TF_link_ENCODE['id_gene'] = df_gene_TF_link_ENCODE['id'] + '_' + df_gene_TF_link_ENCODE['linked_gene_name']
    # df_link_new.loc[:, 'id_gene'] = df_link_new[['id', 'linked_gene_name']].apply(lambda x: '_'.join(x), axis=1)
    df_link_new['id_gene'] = df_link_new['id'].astype('str') + '_' + df_link_new.loc[:, 'linked_gene_name'].astype('str')

    df_gene_TF_link_ENCODE['link_new'] = df_gene_TF_link_ENCODE['id_gene'].isin(df_link_new['id_gene'])
    df_tb = pd.crosstab(df_gene_TF_link_ENCODE['link_new'], df_gene_TF_link_ENCODE['peak'])
    # tmp_fisher = fisher.test(df_tb)
    # print(tmp_fisher)
    # oddsratio, pvalue = stats.fisher_exact(df_tb)
    unique_TF_new = df_link_new.id[df_link_new.id.isin(df_gene_TF_link_ENCODE['id'])].unique()

    unique_TF_pvalue = [None] * len(unique_TF_new)
    for i, tf in enumerate(unique_TF_new):
        df_tmp = df_gene_TF_link_ENCODE.query("id == @tf")
        df_tb = pd.crosstab(df_tmp['link_new'], df_tmp['peak'])
        if df_tb.shape == (2, 2):
            oddsratio, pvalue = stats.fisher_exact(df_tb)
            unique_TF_pvalue[i] = pvalue
        else:
            unique_TF_pvalue[i] = 1

    df_unique_TF_new = pd.DataFrame({"id": unique_TF_new, "pvalue": unique_TF_pvalue})
    df_unique_TF_new['qval'] =  fdr(df_unique_TF_new['pvalue'])
    df_unique_TF_new = df_unique_TF_new.query("qval < 0.05")

    print(f"Number of positive TFs: {df_unique_TF_new.shape[0]}")

    validated_TF = df_unique_TF_new['id'].unique()
    df_gene_TF_link_chip = df_gene_TF_link_ENCODE.loc[df_gene_TF_link_ENCODE.id.isin(validated_TF)]
    df_gene_TF_link_chip = df_gene_TF_link_chip.merge(df_link_new[['id_gene', 'corcoef', 'r_squre']])
    df_gene = df_gene[['gene_id', 'gene_short_name', 'gene_type']]
    df_gene.columns = ['linked_gene_id', 'linked_gene_name', df_gene.columns[2]]

    df_gene_TF_link_chip = df_gene_TF_link_chip.merge(df_gene)
    df_gene_TF_link_chip['Conf'] = "Chip_peak"
    df_gene_TF_link_chip['group'] = ["New_RNA" if i else "no_link" for i in df_gene_TF_link_chip['link_new']]

    df_gene_TF_link_chip = df_gene_TF_link_chip.query("group == 'New_RNA'")
    df_gene_TF_link_chip = df_gene_TF_link_chip[['id', 'linked_gene_name', 'Conf', 'linked_gene_id',
                                                 'gene_type', 'corcoef', 'r_squre', 'id_gene']]
    df_gene_TF_link_chip.columns = ['TF', 'linked_gene', 'Conf', 'linked_gene_id', 'linked_gene_type',
                                    'corcoef', 'r_2', 'TF_link']


def fdr(p_vals):

    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr

def normalize_data(mm, szfactors, pseudo_expr=0.1):
    mm = mm / szfactors
    mm = np.log(mm + pseudo_expr)

    mm = (mm - np.mean(mm, 0)) / np.std(mm, 0, ddof=1)

    return mm
