from tqdm import tqdm
import numpy as np
import pandas as pd
import pysal
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests
from collections import Counter
import warnings
from .utils_markers import fetch_X_data, specificity, fdr

def moran_i(adata,
            X_data = None,
            genes=None,
            layer=None,
            weighted=True,
            assumption='permutation',
            local_moran=False):
    """Identify genes with strong spatial autocorrelation with Moran's I test. This can be used to identify genes that are
    potentially related to critical dynamic process. Moran's I test is first introduced in single cell genomics analysis
    in (Cao, et al, 2019).

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
        Returns an updated `~anndata.AnnData` with a new property `'Moran_' + type` in the .uns attribute.
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
         "Moran_p_val": p_value,
         "Moran_q_val": fdr(np.array(p_value)),
         "Moran_z": statistics,
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
    """Find marker genes for each group of cells.

    Tests each gene for differential expression between cells in one group to cells from all other groups via Mann-Whitney U
    test. It also calculates the fraction of cells with non-zero expression, log 2 fold changes as well as the specificity
    (calculate via the Jessen-Shannon distance between the distribution of percentage of cells with expression across all
    groups to the hypothetical perfect distribution in which only the current group of cells have expression). In addition,
    Rank-biserial correlation (rbc) and qval are calculated. The rank biserial correlation is used to assess the relationship
    between a dichotomous categorical variable and an ordinal variable. The rank biserial test is very similar to the
    non-parametric Mann-Whitney U test that is used to compare two independent groups on an ordinal variable. Mann-Whitney
    U tests are preferable to rank biserial correlations when comparing independent groups. Rank biserial correlations can
    only be used with dichotomous (two levels) categorical variables. qval is calculated using Benjamini-Hochberg adjustment.

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
    ------- ['cluster_markers'] = {'deg_table': de_table, 'de_genes': de_genes}
        Returns an updated `~anndata.AnnData` with a new property `cluster_markers` in the .uns attribute, which include
        a pandas DataFrame of the differential expression analysis result for all groups and a dictionary where keys are
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

        de = top_markers(adata, genes, layer, group, test_group, control_groups, X_data, exp_frac_thresh,
                         log2_fc_thresh, qval_thresh, )

        de_tables[i] = de.copy()
        de_genes[i] = [k for k, v in Counter(de['gene']).items()
                       if v >= de_frequency]
    de_table = pd.concat(de_tables).reset_index().drop(columns=['index'])

    adata.uns['cluster_markers'] = {'deg_table': de_table, 'de_genes': de_genes}

    return adata


def top_markers(adata,
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
    """Find marker genes for each group of cells between any two groups of cells.

    Tests each gene for differential expression between cells in one group to cells from another groups via Mann-Whitney U
    test. It also calculates the fraction of cells with non-zero expression, log 2 fold changes as well as the specificity
    (calculate via the Jessen-Shannon distance between the distribution of percentage of cells with expression across all
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
        rbc, specifity_, mw_p, log_fc, ncells = 0, 0, 1, 0, 0

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

            specifity_ = specificity(perc, perfect_specificity)
            de.append((gene, control_groups[i], ef, rbc, log_fc, mw_p, specifity_))

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
