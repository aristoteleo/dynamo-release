from tqdm import tqdm
import numpy as np
import pandas as pd
import pysal
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
import patsy
from patsy import dmatrix, bs, cr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from collections import Counter
import warnings
from .utils_markers import fetch_X_data, specificity, fdr
from ..preprocessing.utils import Freeman_Tukey


def moran_i(adata,
            X_data = None,
            genes=None,
            layer=None,
            weighted=True,
            assumption='permutation',
            local_moran=False):
    """Identify genes with strong spatial autocorrelation with Moran's I test. This can be used to identify genes that are
    potentially related to critical dynamic process. Moran's I test is first introduced in single cell genomics analysis
    in (Cao, et al, 2019). Note that moran_i supports performing spatial autocorrelation analysis for any layer or
    normalized data in your adata object. That is you can either use the total, new, unspliced or velocity, etc. for the
    Moran's I analysis.

    Global Moran's I test is based on pysal. More details can be found at:
    http://geodacenter.github.io/workbook/5a_global_auto/lab5a.html#morans-i

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        X_data: `np.ndarray` (default: `None`)
            The user supplied data that will be used for clustering directly.
        genes: `list` or None (default: `None`)
            The list of genes that will be used to subset the data for dimension reduction and clustering. If `None`, all
            genes will be used.
        layer: `str` or None (default: `None`)
            The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is used.
        weighted: `bool` (default: `True`)
            Whether to consider edge weights in the spatial weights graph.
        assumption: `str` (default: `permutation`)
            Assumption of the Moran's I test, can be one of {'permutation', 'normality', 'randomization'}.
            Inference for Moran’s I is based on a null hypothesis of spatial randomness. The distribution of the
            statistic under the null can be derived using either an assumption of normality (independent normal random
            variates), or so-called randomization (i.e., each value is equally likely to occur at any location). An
            alternative to an analytical derivation is a computational approach based on permutation. This calculates a
            reference distribution for the statistic under the null hypothesis of spatial randomness by randomly permuting
            the observed values over the locations. The statistic is computed for each of these randomly reshuffled data sets,
            which yields a reference distribution.
        local_moran: `bool` (default: `False`)
            Whether to also calculate local Moran's I.

    Returns
    -------
        Returns an updated `~anndata.AnnData` with a new key `'Moran_' + type` in the .uns attribute, storing the Moran' I
        test results.
    """

    if X_data is None:
        genes, X_data = fetch_X_data(adata, genes, layer)
    else:
        if genes is None or len(genes) != X_data.shape[1]:
            raise ValueError(f"When providing X_data, a list of genes name that corresponds to the columns of X_data "
                             f"must be provided")

    cell_num, gene_num = X_data.shape

    embedding_key = (
        "X_umap" if layer is None else layer + "_umap"
    )
    neighbor_key = "neighbors" if layer is None else layer + "_neighbors"
    if neighbor_key not in adata.uns.keys():
        warnings.warn(f"Neighbor graph is required for Moran's I test. No neighbor_key {neighbor_key} exists in the data. "
                      f"Running reduceDimension which will generate the neighbor graph and the associated low dimension"
                      f"embedding {embedding_key}. ")
        from .dimension_reduction import reduceDimension
        adata = reduceDimension(adata, X_data=X_data, layer=layer)

    neighbor_graph = adata.uns["neighbors"]["connectivities"]

    # convert a sparse adjacency matrix to a dictionary
    adj_dict = {
        i: row.indices
        for i, row in enumerate(neighbor_graph)
    }
    if weighted :
        weight_dict = {
            i: row.data
            for i, row in enumerate(neighbor_graph)
        }
        W = pysal.lib.weights.W(adj_dict, weight_dict)
    else:
        W = pysal.lib.weights.W(adj_dict)

    sparse = issparse(X_data)

    Moran_I, p_value, statistics = [None] * gene_num, [None] * gene_num, [None] * gene_num
    if local_moran:
        l_moran = np.zeros(X_data.shape)
    for cur_g in tqdm(range(gene_num), desc="Moran’s I Global Autocorrelation Statistic"):
        cur_X = X_data[:, cur_g].A if sparse else X_data[:, cur_g]
        mbi = pysal.explore.esda.moran.Moran(cur_X, W, two_tailed=False)

        Moran_I[cur_g] = mbi.I
        p_value[cur_g] = mbi.p_sim if assumption == 'permutation' else \
            mbi.p_norm if assumption == 'normality' else mbi.p_rand
        statistics[cur_g] = mbi.z_sim if assumption == 'permutation' else \
            mbi.z_norm if assumption == 'normality' else mbi.z_sim
        if local_moran:
            l_moran[:, cur_g] = pysal.explore.esda.moran.Moran_Local(cur_X, W).Is

    Moran_res = pd.DataFrame(
        {"moran_i": Moran_I,
         "moran_p_val": p_value,
         "moran_q_val": fdr(np.array(p_value)),
         "moran_z": statistics,
         }, index=genes
    )

    adata.var = adata.var.merge(Moran_res, left_index=True, right_index=True, how='left')
    if local_moran:
        adata.uns['local_moran'] = l_moran

    return adata


def find_group_markers(adata,
                       group,
                       genes=None,
                       layer=None,
                       exp_frac_thresh=0.1,
                       log2_fc_thresh=1,
                       qval_thresh=0.05,
                       de_frequency=1,
                       ):
    """Find marker genes for each group of cells based on gene expression or velocity values as specified by the layer.

    Tests each gene for differential expression between cells in one group to cells from all other groups via Mann-Whitney U
    test. It also calculates the fraction of cells with non-zero expression, log 2 fold changes as well as the specificity
    (calculated as 1 - Jessen-Shannon distance between the distribution of percentage of cells with expression across all
    groups to the hypothetical perfect distribution in which only the test group of cells has expression). In addition,
    Rank-biserial correlation (rbc) and qval are calculated. The rank biserial correlation is used to assess the relationship
    between a dichotomous categorical variable and an ordinal variable. The rank biserial test is very similar to the
    non-parametric Mann-Whitney U test that is used to compare two independent groups on an ordinal variable. Mann-Whitney
    U tests are preferable to rank biserial correlations when comparing independent groups. Rank biserial correlations can
    only be used with dichotomous (two levels) categorical variables. qval is calculated using Benjamini-Hochberg adjustment.

    Note that this function is designed in a general way so that you can either use the total, new, unspliced or velocity,
    etc. to identify differentially expressed genes.

    This function is adapted from https://github.com/KarlssonG/nabo/blob/master/nabo/_marker.py and Monocle 3alpha.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        group: `str` or None (default: `None`)
            The column key/name that identifies the grouping information (for example, clusters that correspond to
            different cell types or different time points) of cells. This will be used for calculating group-specific genes.
        genes: `list` or None (default: `None`)
            The list of genes that will be used to subset the data for dimension reduction and clustering. If `None`, all
            genes will be used.
        layer: `str` or None (default: `None`)
            The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is used.
        exp_frac_thresh: `float` (default: 0.1)
            The minimum percentage of cells with expression for a gene to proceed differential expression test.
        log2_fc_thresh: `float` (default: 0.1)
            The minimal threshold of log2 fold change for a gene to proceed differential expression test.
        qval_thresh: `float` (default: 0.05)
            The minimial threshold of qval to be considered as significant genes.
        de_frequency:
            Minimum number of clusters against a gene should be significantly differentially expressed for it to qualify
            as a marker.

    Returns
    -------
        Returns an updated `~anndata.AnnData` with a new property `cluster_markers` in the .uns attribute, which includes
        a concated pandas DataFrame of the differential expression analysis result for all groups and a dictionary where keys are
        cluster numbers and values are lists of marker genes for the corresponding clusters.
    """

    genes, X_data = fetch_X_data(adata, genes, layer)
    if len(genes) == 0:
        raise ValueError(f'No genes from your genes list appear in your adata object.')

    if group not in adata.obs.keys():
        raise ValueError(f"group {group} is not a valid key for .obs in your adata object.")
    else:
        adata.obs[group] = adata.obs[group].astype('str')
        cluster_set = adata.obs[group].unique()

    de_tables = [None] * len(cluster_set)
    de_genes = {}

    for i, test_group in enumerate(cluster_set):
        control_groups = sorted(set(cluster_set).difference([test_group]))

        de = two_groups_degs(adata, genes, layer, group, test_group, control_groups, X_data, exp_frac_thresh,
                             log2_fc_thresh, qval_thresh, )

        de_tables[i] = de.copy()
        de_genes[i] = [k for k, v in Counter(de['gene']).items()
                       if v >= de_frequency]
    de_table = pd.concat(de_tables).reset_index().drop(columns=['index'])

    adata.uns['cluster_markers'] = {'deg_table': de_table, 'de_genes': de_genes}

    return adata


def two_groups_degs(adata,
                    genes,
                    layer,
                    group,
                    test_group,
                    control_groups,
                    X_data,
                    exp_frac_thresh=0.1,
                    log2_fc_thresh=1,
                    qval_thresh=0.05,
                    ):
    """Find marker genes between two groups of cells based on gene expression or velocity values as specified by the layer.

    Tests each gene for differential expression between cells in one group to cells from another groups via Mann-Whitney U
    test. It also calculates the fraction of cells with non-zero expression, log 2 fold changes as well as the specificity
    (calculated as 1 - Jessen-Shannon distance between the distribution of percentage of cells with expression across all
    groups to the hypothetical perfect distribution in which only the current group of cells have expression). In addition,
    Rank-biserial correlation (rbc) and qval are calculated. The rank biserial correlation is used to assess the relationship
    between a dichotomous categorical variable and an ordinal variable. The rank biserial test is very similar to the
    non-parametric Mann-Whitney U test that is used to compare two independent groups on an ordinal variable. Mann-Whitney
    U tests are preferable to rank biserial correlations when comparing independent groups. Rank biserial correlations can
    only be used with dichotomous (two levels) categorical variables. qval is calculated using Benjamini-Hochberg adjustment.


    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        genes: `list` or None (default: `None`)
            The list of genes that will be used to subset the data for dimension reduction and clustering. If `None`, all
            genes will be used.
        layer: `str` or None (default: `None`)
            The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is used.
        group: `str` or None (default: `None`)
            The column key/name that identifies the grouping information (for example, clusters that correspond to
            different cell types or different time points) of cells. This will be used for calculating group-specific genes.
        test_group: `str` or None (default: `None`)
            The group name from `group` for which markers has to be found.
        control_groups: `list`
            The list of group name(s) from `group` for which markers has to be tested against.
        X_data: `np.ndarray` (default: `None`)
            The user supplied data that will be used for marker gene detection directly.
        exp_frac_thresh: `float` (default: 0.1)
            The minimum percentage of cells with expression for a gene to proceed differential expression test.
        log2_fc_thresh: `float` (default: 0.1)
            The minimal threshold of log2 fold change for a gene to proceed differential expression test.
        qval_thresh: `float` (default: 0.05)
            The maximal threshold of qval to be considered as significant genes.


    Returns
    -------
        A pandas DataFrame of the differential expression analysis result between the two groups.
    """

    if X_data is None:
        genes, X_data = fetch_X_data(adata, genes, layer)
    else:
        if genes is None or len(genes) != X_data.shape[1]:
            raise ValueError(f"When providing X_data, a list of genes name that corresponds to the columns of X_data "
                             f"must be provided")

    n_cells, n_genes = X_data.shape
    sparse = issparse(X_data)

    test_cells, control_cells = adata.obs[group] == test_group, \
                                adata.obs[group].isin(control_groups)

    num_test_cells = test_cells.sum()
    num_groups = len(control_groups)
    min_n = [min(num_test_cells, sum(adata.obs[group] == x)) for x in control_groups]
    n1n2 = [num_test_cells * x for x in min_n]

    de = []
    for i_gene, gene in tqdm(enumerate(genes), desc="identifying top markers for each group"):
        rbc, specificity_, mw_p, log_fc, ncells = 0, 0, 1, 0, 0

        all_vals = X_data[:, i_gene].A if sparse else X_data[:, i_gene]
        test_vals = all_vals[test_cells]
        perc, ef = [len(test_vals.nonzero()[0]) / n_cells], len(test_vals.nonzero()[0]) / num_test_cells
        if ef < exp_frac_thresh:
            continue

        log_mean_test_vals = np.log2(test_vals.mean())
        perc.extend([len(all_vals[adata.obs[group] == x].nonzero()[0]) / n_cells for x in control_groups])

        for i in range(num_groups):
            control_vals = all_vals[adata.obs[group] == control_groups[i]]
            control_vals.sort()
            control_vals = control_vals[-min_n[i]:]

            mean_control_vals = control_vals.mean()
            if mean_control_vals == 0:
                log_fc = np.inf
            else:
                log_fc = log_mean_test_vals - np.log2(mean_control_vals)
            if log_fc < log2_fc_thresh:
                continue
            try:
                u, mw_p = mannwhitneyu(test_vals, control_vals)
            except ValueError:
                pass
            else:
                rbc = 1 - ((2 * u) / n1n2[i])

            perfect_specificity = np.repeat(0.0, num_groups + 1)
            perfect_specificity[i + 1] = 1.0

            specificity_ = specificity(perc, perfect_specificity)
            de.append((gene, control_groups[i], ef, rbc, log_fc, mw_p, specificity_))

    de = pd.DataFrame(de,
                      columns=['gene', 'versus_group', 'exp_frac', 'rbc', 'log2_fc', 'pval', 'specificity'])
    de = de[de.iloc[:, 2:].sum(1) > 0]

    if de.shape[0] > 1:
        de['qval'] = multipletests(de['pval'].values, method='fdr_bh')[1]
    else:
        de['qval'] = [np.nan for _ in range(de.shape[0])]
    de['test_group'] = [test_group for _ in range(de.shape[0])]
    out_order = ['gene', 'test_group', 'versus_group', 'specificity', 'exp_frac',
                 'rbc', 'log2_fc', 'pval', 'qval']
    de = de[out_order].sort_values(by='qval')
    res = de[(de.qval < qval_thresh)].reset_index().drop(columns=['index'])

    return res


def top_n_markers(adata,
                  with_moran_i=False,
                  group_by='test_group',
                  sort_by='specificity',
                  sort_order='decreasing',
                  top_n_genes=5,
                  exp_frac_thresh=0.1,
                  log2_fc_thresh=1,
                  qval_thresh=0.05,
                  specificity_thresh=0.3,
                  only_gene_list=False,
                  display=True):
    """Filter cluster deg (Moran's I test) results and retrieve top markers for each cluster.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        with_moran_i: `bool` (default: `False`)
            Whether or not to include Moran's I test results for selecting top marker genes.
        group_by: `str` or `list` (default: `test_group`)
            Column name or names to group by.
        sort_by: `str` or `list`
            Column name or names to sort by.
        sort_order: `str` (default: `decreasing`)
            Whether to sort the data frame with `increasing` or `decreasing` order.
        top_n_genes: `int`
            The number of top sorted markers.
        exp_frac_thresh: `float` (default: 0.1)
            The minimum percentage of cells with expression for a gene to proceed selection of top markers.
        log2_fc_thresh: `float` (default: 0.1)
            The minimal threshold of log2 fold change for a gene to proceed selection of top markers.
        qval_thresh: `float` (default: 0.05)
            The maximal threshold of qval to be considered as top markers.
        only_gene_list: `bool`
            Whether to only return the gene list for each cluster.
        display: `bool`
            Whether to print the data frame for the top marker genes after the filtering.

    Returns
    -------
        A data frame that stores the top marker for each group or just a list for those markers, depending on
        whether `only_gene_list` is set to be True. In addition, it will display the data frame depending on whether
        `display` is set to be True.
    """

    if "cluster_markers" not in adata.uns.keys():
        warnings.warn(f'No info of cluster markers stored in your adata. '
                      f'Running `find_group_markers` with default parameters.')
        adata = find_group_markers(adata, group='clusters')

    deg_table = adata.uns['cluster_markers']['deg_table']
    deg_table = deg_table.query("exp_frac > @exp_frac_thresh and "
                                "log2_fc > @log2_fc_thresh and "
                                "qval < @qval_thresh and "
                                "specificity > @specificity_thresh")
    if deg_table.shape[0] == 0:
        raise ValueError(f'Looks like your filter threshold is too extreme. No gene detected. '
                         f'Please try relaxing the thresholds you specified: '
                         f'exp_frac_thresh: {exp_frac_thresh}'
                         f'log2_fc_thresh: {log2_fc_thresh}'
                         f'qval_thresh: {qval_thresh}'
                         f'specificity_thresh: {specificity_thresh}')
    if with_moran_i:
        moran_i_columns = ['moran_i', 'moran_p_val', 'moran_q_val', 'moran_z']
        if len(adata.var.columns.intersection(moran_i_columns)) != 4:
            warnings.warn(f'No info of cluster markers stored in your adata. '
                          f'Running `find_group_markers` with default parameters.')
            adata = moran_i(adata)

        moran_i_df = adata.var[moran_i_columns]
        deg_table = deg_table.merge(moran_i_df, left_on='gene', right_index=True, how='left')

    top_n_df = deg_table.groupby(group_by).apply(lambda grp: grp.nlargest(top_n_genes, sort_by)) if sort_order == 'decreasing'\
        else deg_table.groupby(group_by).apply(lambda grp: grp.nsmallest(top_n_genes, sort_by))

    if display:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(top_n_df)

    top_n_groups = top_n_df.loc[:, group_by].unique()
    de_genes = [None] * len(top_n_groups)

    if only_gene_list:
        for i in top_n_groups:
            de_genes[i] = top_n_df[top_n_df[group_by] == i].loc[:, 'gene'].to_list()
        return de_genes
    else:
        return top_n_df


def glm_degs(adata,
             X_data=None,
             genes=None,
             layer=None,
             fullModelFormulaStr="~cr(integral_time, df=3)",
             reducedModelFormulaStr="~1",
             family='NB2',
             ):
    """Differential genes expression tests using generalized linear regressions.

    Tests each gene for differential expression as a function of integral time (the time estimated via the reconstructed
    vector field function) or pseudotime using generalized additive models with nature spline basis. This function can also
    use other covariates as specified in the full and reduced model formula to identify differentially expression genes
    across different categories, group, etc.

    glm_degs relies on statsmodels package and is adapted from the `differentialGeneTest` in Monocle. Note that glm_degs
    supports performing deg analysis for any layer or normalized data in your adata object. That is you can either use
    the total, new, unspliced or velocity, etc. for the differential expression analysis.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        X_data: `np.ndarray` (default: `None`)
            The user supplied data that will be used for clustering directly.
        genes: `list` or None (default: `None`)
            The list of genes that will be used to subset the data for dimension reduction and clustering. If `None`, all
            genes will be used.
        layer: `str` or None (default: `None`)
            The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is used.
        fullModelFormulaStr: `str` (default: `~cr(time, df=3)`)
            fullModelFormulaStr a formula string specifying the full model in differential expression tests (i.e.
            likelihood ratio tests) for each gene/feature.
        reducedModelFormulaStr: `str` (default: `~1`)
            reducedModelFormulaStr a formula string specifying the reduced model in differential expression tests (i.e.
            likelihood ratio tests) for each gene/feature.
        family: `str` (default: `NB2`)
            The distribution family used for the expression responses in statsmodels. Currently always uses `NB2` and this
            is ignored.

    Returns
    -------
        Returns an updated `~anndata.AnnData` with a new key `glm_degs` in the .uns attribute, storing the differential
        expression test results after the GLM test.
    """

    if X_data is None:
        genes, X_data = fetch_X_data(adata, genes, layer)
    else:
        if genes is None or len(genes) != X_data.shape[1]:
            raise ValueError(f"When providing X_data, a list of genes name that corresponds to the columns of X_data "
                             f"must be provided")
    if layer is None:
        if issparse(X_data):
            X_data.data = (
                2 ** X_data.data - 1
                if adata.uns["pp_norm_method"] == "log2"
                else np.exp(X_data.data) - 1
                if adata.uns["pp_norm_method"] == "log"
                else Freeman_Tukey(X_data.data + 1, inverse=True)
                if adata.uns["pp_norm_method"] == "Freeman_Tukey"
                else X_data.data
            )
        else:
            X_data = (
                2 ** X_data - 1
                if adata.uns["pp_norm_method"] == "log2"
                else np.exp(X_data) - 1
                if adata.uns["pp_norm_method"] == "log"
                else Freeman_Tukey(X_data, inverse=True)
                if adata.uns["pp_norm_method"] == "Freeman_Tukey"
                else X_data
            )

    factors = get_all_variables(fullModelFormulaStr)
    factors = ['Pseudotime' if i == 'cr(Pseudotime, df=3)' else i for i in factors]
    if len(set(factors).difference(adata.obs.columns)) == 0:
        df_factors = adata.obs[factors]
    else:
        raise Exception(f"adata object doesn't include the factors from the model formula "
                        f"{fullModelFormulaStr} you provided.")

    sparse = issparse(X_data)
    deg_df = pd.DataFrame(index=genes, columns=['status', 'family', 'pval'])
    for i, gene in tqdm(enumerate(genes), "Detecting time dependent genes via Generalized Additive Models (GAMs)"):
        expression = X_data[:, i].A if sparse else X_data[:, i]
        df_factors['expression'] = expression
        deg_df.iloc[i, :] = diff_test_helper(df_factors, fullModelFormulaStr, reducedModelFormulaStr)

    deg_df['qval'] = multipletests(deg_df['pval'], method='fdr_bh')[1]

    adata.uns['glm_degs'] = deg_df


def diff_test_helper(data,
                     fullModelFormulaStr="~cr(time, df=3)",
                     reducedModelFormulaStr="~1",
                     ):
    # Dividing data into train and validation datasets
    transformed_x = dmatrix(fullModelFormulaStr, data, return_type='dataframe')
    transformed_x_null = dmatrix(reducedModelFormulaStr, data, return_type='dataframe')

    expression = data['expression']
    poisson_training_results = sm.GLM(expression, transformed_x, family=sm.families.Poisson()).fit()
    poisson_df = pd.DataFrame({'mu': poisson_training_results.mu, 'expression': expression})
    poisson_df['AUX_OLS_DEP'] = poisson_df.apply(lambda x: ((x['expression'] - x['mu']) ** 2
                                                            - x['expression']) / x['mu'], axis=1)
    ols_expr = """AUX_OLS_DEP ~ mu - 1"""
    aux_olsr_results = smf.ols(ols_expr, poisson_df).fit()

    nb2_family = sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])

    try:
        nb2_full = sm.GLM(expression, transformed_x, family=nb2_family).fit()
        nb2_null = sm.GLM(expression, transformed_x_null, family=nb2_family).fit()
    except:
        return ('fail', 'NB2', 1)

    pval = lrt(nb2_full, nb2_null)
    return ('ok', 'NB2', pval)


def get_all_variables(formula):
    md = patsy.ModelDesc.from_formula(formula)
    termlist = md.rhs_termlist + md.lhs_termlist

    factors = []
    for term in termlist:
        for factor in term.factors:
            factors.append(factor.name())

    return factors


def lrt(full, restr):
    llf_full = full.llf
    llf_restr = restr.llf
    df_full = full.df_resid
    df_restr = restr.df_resid
    lrdf = (df_restr - df_full)
    lrstat = -2 * (llf_restr - llf_full)
    lr_pvalue = stats.chi2.sf(lrstat, df=lrdf)

    return  lr_pvalue
