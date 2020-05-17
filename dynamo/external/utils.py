import numpy as np
import pandas as pd
from scipy.sparse import issparse
import scipy.stats as stats
from ..tools.utils_markers import fdr

def normalize_data(mm, szfactors, pseudo_expr=0.1):
    """normalize data via size factor and scaling."""

    mm = mm / szfactors

    if issparse(mm):
        from ..tools.utils import elem_prod
        mm.data = np.log(mm.data + pseudo_expr)
        mm1, mm2 = mm.mean(0), elem_prod(mm, mm).mean(0)
        var_ = mm2 - elem_prod(mm1, mm1)
        mm = (mm - np.mean(mm, 0)) / np.sqrt(var_)
        mm = mm.A
    else:
        mm = np.log(mm + pseudo_expr)

        mm = (mm - np.mean(mm, 0)) / np.std(mm, 0, ddof=1)

    return mm

def TF_link_gene_chip(raw_glmnet_res_var, df_gene_TF_link_ENCODE, var, cor_thresh=0.02):
    """Filter the raw lasso regression links via chip-seq data based on a Fisher exact test"""

    glmnet_res_var_filtered = raw_glmnet_res_var.query("abs(corcoef) > @cor_thresh")
    print(f"\n Number of possible links: glmnet_res_var_filtered.shape[0]")

    # source - target gene pairs
    df_gene_TF_link_ENCODE['id_gene'] = df_gene_TF_link_ENCODE['id'].astype('str') + '_' + \
                                        df_gene_TF_link_ENCODE['linked_gene_name'].astype('str')
    glmnet_res_var_filtered['id_gene'] = glmnet_res_var_filtered['id'].astype('str') + '_' + \
                                       glmnet_res_var_filtered['linked_gene_name'].astype('str')

    df_gene_TF_link_ENCODE['glmnet_chip_links'] = df_gene_TF_link_ENCODE['id_gene'].isin(glmnet_res_var_filtered['id_gene'])
    unique_TFs = glmnet_res_var_filtered.id[glmnet_res_var_filtered.id.isin(df_gene_TF_link_ENCODE['id'])].unique()

    df_tb = pd.crosstab(df_gene_TF_link_ENCODE['glmnet_chip_links'], df_gene_TF_link_ENCODE['peak'])
    oddsratio, pvalue = stats.fisher_exact(df_tb)
    print(f'odd ratio and pvalue of Fisher exact test for regression identified regulations were validated by target '
          f'TF-binding sites near gene promoters from ENCODE, limiting to TFs from lasso regression and also '
          f'characterized in ENCODE are: {oddsratio}, {pvalue}')

    # Only gene sets with significant enrichment of the correct TF ChIP–seq binding sites were retained
    unique_TF_pvalue = [None] * len(unique_TFs)
    for i, tf in enumerate(unique_TFs):
        df_tmp = df_gene_TF_link_ENCODE.query("id == @tf")
        df_tb = pd.crosstab(df_tmp['glmnet_chip_links'], df_tmp['peak'])
        if df_tb.shape == (2, 2):
            _, pvalue = stats.fisher_exact(df_tb)
            unique_TF_pvalue[i] = pvalue
        else:
            unique_TF_pvalue[i] = 1

    df_unique_TF = pd.DataFrame({"id": unique_TFs, "pvalue": unique_TF_pvalue})
    df_unique_TF['qval'] = fdr(df_unique_TF['pvalue'])
    df_unique_TF = df_unique_TF.query("qval < 0.05")

    print(f"Number of positive TFs: {df_unique_TF.shape[0]}")

    validated_TF = df_unique_TF['id'].unique()
    df_gene_TF_link_chip = df_gene_TF_link_ENCODE.loc[df_gene_TF_link_ENCODE.id.isin(validated_TF)]
    df_gene_TF_link_chip = df_gene_TF_link_chip.merge(glmnet_res_var_filtered[['id_gene', 'corcoef', 'r_squre']])
    df_gene = var[['gene_id', 'gene_short_name', 'gene_type']]
    df_gene.columns = ['linked_gene_id', 'linked_gene_name', df_gene.columns[2]]

    df_gene_TF_link_chip = df_gene_TF_link_chip.merge(df_gene)
    df_gene_TF_link_chip['Conf'] = "Chip_peak"
    df_gene_TF_link_chip['group'] = ["has_link" if i else "no_link" for i in df_gene_TF_link_chip['glmnet_chip_links']]

    df_gene_TF_link_chip = df_gene_TF_link_chip.query("group == 'has_link'")
    df_gene_TF_link_chip = df_gene_TF_link_chip[['id', 'linked_gene_name', 'Conf', 'linked_gene_id',
                                                 'gene_type', 'corcoef', 'r_squre', 'id_gene']]
    df_gene_TF_link_chip.columns = ['TF', 'linked_gene', 'Conf', 'linked_gene_id', 'linked_gene_type',
                                    'corcoef', 'r_2', 'TF_link']

    return df_gene_TF_link_chip



