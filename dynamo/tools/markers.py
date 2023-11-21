from typing import List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import warnings
from collections import Counter

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from anndata import AnnData
from patsy import bs, cr, dmatrix
from scipy import stats
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from statsmodels.sandbox.stats.multicomp import multipletests
from tqdm import tqdm

from ..dynamo_logger import (
    main_critical,
    main_exception,
    main_info,
    main_tqdm,
    main_warning,
)
from ..preprocessing.transform import _Freeman_Tukey
from ..tools.connectivity import generate_neighbor_keys, check_and_recompute_neighbors
from .utils import fdr, fetch_X_data


def moran_i(
    adata: AnnData,
    X_data: Optional[np.ndarray] = None,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    weighted: bool = True,
    assumption: Literal["permutation", "normality", "randomization"] = "permutation",
    local_moran: bool = False,
) -> AnnData:
    """Identify genes with strong spatial autocorrelation with Moran's I test.

    This can be used to identify genes that are potentially related to critical dynamic process. Moran's I test is first
    introduced in single cell genomics analysis in (Cao, et al. 2019). Note that moran_i supports performing spatial
    autocorrelation analysis for any layer or normalized data in your adata object. That is you can either use the
    total, new, unspliced or velocity, etc. for the Moran's I analysis.

    Args:
        adata: An AnnData object.
        X_data: The user supplied data that will be used for Moran's I calculation directly. Defaults to None.
        genes: The list of genes that will be used to subset the data for dimension reduction and clustering. If `None`,
            all genes will be used. Defaults to None.
        layer: The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is
            used. Defaults to None.
        weighted: Whether to consider edge weights in the spatial weights graph. Defaults to True.
        assumption: Assumption of the Moran's I test. Inference for Moran's I is based on a null hypothesis of spatial
            randomness. The distribution of the statistic under the null can be derived using either an assumption of
            normality (independent normal random variates), or so-called randomization (i.e., each value is equally
            likely to occur at any location). An alternative to an analytical derivation is a computational approach
            based on permutation. This calculates a reference distribution for the statistic under the null hypothesis
            of spatial randomness by randomly permuting the observed values over the locations. The statistic is
            computed for each of these randomly reshuffled data sets, which yields a reference distribution. Defaults to
            "permutation".
        local_moran: Whether to also calculate local Moran's I. Defaults to False.

    Raises:
        ImportError: Package pysal not installed.
        ValueError: `X_data` is provided but `genes` does not correspond to its columns.

    Returns:
        The updated AnnData object with a new key `'Moran_' + type` in the .uns attribute, storing the Moran' I
        test results.
    """

    try:
        from pysal import explore, lib
    except ImportError:
        raise ImportError(
            "You need to install the package `pysal`."
            "Please install via pip install pysal. See more details at https://pypi.org/project/pysal/"
        )

    if X_data is None:
        genes, X_data = fetch_X_data(adata, genes, layer)
    else:
        if genes is None or len(genes) != X_data.shape[1]:
            raise ValueError(
                f"When providing X_data, a list of genes name that corresponds to the columns of X_data "
                f"must be provided"
            )

    cell_num, gene_num = X_data.shape

    embedding_key = "X_umap" if layer is None else layer + "_umap"
    neighbor_result_prefix = "" if layer is None else layer
    conn_key, dist_key, neighbor_key = generate_neighbor_keys(neighbor_result_prefix)

    if neighbor_key not in adata.uns.keys():
        main_warning(
            f"Neighbor graph is required for Moran's I test. No neighbor_key {neighbor_key} exists in the data. "
            f"Running reduceDimension which will generate the neighbor graph and the associated low dimension"
            f"embedding {embedding_key}. "
        )
        from .dimension_reduction import reduceDimension

        adata = reduceDimension(adata, X_data=X_data, layer=layer)

    check_and_recompute_neighbors(adata, result_prefix=neighbor_result_prefix)
    neighbor_graph = adata.obsp[conn_key]

    # convert a sparse adjacency matrix to a dictionary
    adj_dict = {i: row.indices for i, row in enumerate(neighbor_graph)}
    if weighted:
        weight_dict = {i: row.data for i, row in enumerate(neighbor_graph)}
        W = lib.weights.W(adj_dict, weight_dict)
    else:
        W = lib.weights.W(adj_dict)

    sparse = issparse(X_data)

    Moran_I, p_value, statistics = (
        [None] * gene_num,
        [None] * gene_num,
        [None] * gene_num,
    )
    if local_moran:
        l_moran = np.zeros(X_data.shape)
    for cur_g in tqdm(range(gene_num), desc="Moran’s I Global Autocorrelation Statistic"):
        cur_X = X_data[:, cur_g].A if sparse else X_data[:, cur_g]
        mbi = explore.esda.moran.Moran(cur_X, W, two_tailed=False)

        Moran_I[cur_g] = mbi.I
        p_value[cur_g] = (
            mbi.p_sim if assumption == "permutation" else mbi.p_norm if assumption == "normality" else mbi.p_rand
        )
        statistics[cur_g] = (
            mbi.z_sim if assumption == "permutation" else mbi.z_norm if assumption == "normality" else mbi.z_sim
        )
        if local_moran:
            l_moran[:, cur_g] = explore.esda.moran.Moran_Local(cur_X, W).Is

    Moran_res = pd.DataFrame(
        {
            "moran_i": Moran_I,
            "moran_p_val": p_value,
            "moran_q_val": fdr(np.array(p_value)),
            "moran_z": statistics,
        },
        index=genes,
    )

    adata.var = adata.var.merge(Moran_res, left_index=True, right_index=True, how="left")
    if local_moran:
        adata.uns["local_moran"] = l_moran

    return adata


def find_group_markers(
    adata: AnnData,
    group: str,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    exp_frac_thresh: Optional[float] = None,
    log2_fc_thresh: Optional[float] = None,
    qval_thresh: float = 0.05,
    de_frequency: int = 1,
    subset_control_vals: Optional[bool] = None,
) -> AnnData:
    """Find marker genes for each group of cells based on gene expression or velocity values as specified by the layer.

    Tests each gene for differential expression between cells in one group to cells from all other groups via
    Mann-Whitney U test. It also calculates the fraction of cells with non-zero expression, log 2-fold changes as well
    as the specificity (calculated as 1 - Jessen-Shannon distance between the distribution of percentage of cells with
    expression across all groups to the hypothetical perfect distribution in which only the test group of cells has
    expression). In addition, Rank-biserial correlation (rbc) and qval are calculated. The rank biserial correlation is
    used to assess the relationship between a dichotomous categorical variable and an ordinal variable. The rank
    biserial test is very similar to the non-parametric Mann-Whitney U test that is used to compare two independent
    groups on an ordinal variable. Mann-Whitney U tests are preferable to rank biserial correlations when comparing
    independent groups. Rank biserial correlations can only be used with dichotomous (two levels) categorical variables.
    qval is calculated using Benjamini-Hochberg adjustment.

    Note that this function is designed in a general way so that you can either use the total, new, unspliced or
    velocity, etc. to identify differentially expressed genes.

    This function is adapted from https://github.com/KarlssonG/nabo/blob/master/nabo/_marker.py and Monocle 3alpha.

    Args:
        adata: An AnnData object.
        group: The column key/name that identifies the grouping information (for example, clusters that correspond to
            different cell types or different time points) of cells. This will be used for calculating group-specific
            genes.
        genes: The list of genes that will be used to subset the data for dimension reduction and clustering. If `None`,
            all genes will be used. Defaults to None.
        layer: The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is
            used. Defaults to None.
        exp_frac_thresh: The minimum percentage of cells with expression for a gene to proceed differential expression
            test. If `layer` is not `velocity` related (i.e. `velocity_S`), `exp_frac_thresh` by default is set to be
            0.1, otherwise 0. Defaults to None.
        log2_fc_thresh: The minimal threshold of log2 fold change for a gene to proceed differential expression test. If
            `layer` is not `velocity` related (i.e. `velocity_S`), `log2_fc_thresh` by default is set to be 1, otherwise
            0. Defaults to None.
        qval_thresh: The minimal threshold of qval to be considered as significant genes. Defaults to 0.05.
        de_frequency: Minimum number of clusters against a gene should be significantly differentially expressed for it
            to qualify as a marker. Defaults to 1.
        subset_control_vals: Whether to subset the top ranked control values. When `subset_control_vals = None`, this is
            subset to be `True` when layer is not related to either `velocity` related or `acceleration` or `curvature`
            related layers and `False` otherwise. When layer is not related to either `velocity` related or
            `acceleration` or `curvature` related layers used, the control values will be sorted by absolute values.
            Defaults to None.

    Raises:
        ValueError: Gene list does not overlap with genes in adata.
        ValueError: `group` is invalid.
        ValueError: .obs[group] does not contain enough number of groups.

    Returns:
        An updated `~anndata.AnnData` with a new property `cluster_markers` in the .uns attribute, which includes a
        concated pandas DataFrame of the differential expression analysis result for all groups and a dictionary where
        keys are cluster numbers and values are lists of marker genes for the corresponding clusters.
    """

    if layer is None or not (layer.startswith("velocity") or layer in ["acceleration", "curvature"]):
        exp_frac_thresh = 0.1 if exp_frac_thresh is None else exp_frac_thresh
        log2_fc_thresh = 1 if log2_fc_thresh is None else log2_fc_thresh
        subset_control_vals = True if subset_control_vals is None else subset_control_vals
    else:
        exp_frac_thresh = 0 if exp_frac_thresh is None else exp_frac_thresh
        log2_fc_thresh = None
        subset_control_vals = False if subset_control_vals is None else subset_control_vals

    genes, X_data = fetch_X_data(adata, genes, layer)
    if len(genes) == 0:
        raise ValueError(f"No genes from your genes list appear in your adata object.")

    if group not in adata.obs.keys():
        raise ValueError(f"group {group} is not a valid key for .obs in your adata object.")
    else:
        adata.obs[group] = adata.obs[group].astype("str")
        cluster_set = adata.obs[group].unique()

        if len(cluster_set) < 2:
            raise ValueError(f"the number of groups for the argument {group} must be at least two.")

    de_tables = [None] * len(cluster_set)
    de_genes = {}

    if len(cluster_set) > 2:
        for i, test_group in enumerate(cluster_set):
            control_groups = sorted(set(cluster_set).difference([test_group]))

            de = two_groups_degs(
                adata,
                genes,
                layer,
                group,
                test_group,
                control_groups,
                X_data,
                exp_frac_thresh,
                log2_fc_thresh,
                qval_thresh,
                subset_control_vals,
            )

            de_tables[i] = de.copy()
            de_genes[i] = [k for k, v in Counter(de["gene"]).items() if v >= de_frequency]
    else:
        de = two_groups_degs(
            adata,
            genes,
            layer,
            group,
            cluster_set[0],
            cluster_set[1],
            X_data,
            exp_frac_thresh,
            log2_fc_thresh,
            qval_thresh,
            subset_control_vals,
        )

        de_tables[0] = de.copy()
        de_genes[0] = [k for k, v in Counter(de["gene"]).items() if v >= de_frequency]

    de_table = pd.concat(de_tables).reset_index().drop(columns=["index"])
    de_table["log2_fc"] = de_table["log2_fc"].astype("float")

    adata.uns["cluster_markers"] = {"deg_table": de_table}
    for key, value in de_genes.items():
        adata.uns["cluster_markers"]["de_genes" + str(key)] = value

    return adata


def two_groups_degs(
    adata: AnnData,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    group: Optional[str] = None,
    test_group: Optional[str] = None,
    control_groups: List[str] = None,
    X_data: Optional[np.ndarray] = None,
    exp_frac_thresh: Optional[float] = None,
    log2_fc_thresh: Optional[str] = None,
    qval_thresh: float = 0.05,
    subset_control_vals: Optional[bool] = None,
) -> pd.DataFrame:
    """Find marker genes between two groups of cells based on gene expression or velocity as specified by the layer.

    Tests each gene for differential expression between cells in one group to cells from another groups via Mann-Whitney
    U test. It also calculates the fraction of cells with non-zero expression, log 2-fold changes as well as the
    specificity (calculated as 1 - Jessen-Shannon distance between the distribution of percentage of cells with
    expression across all groups to the hypothetical perfect distribution in which only the current group of cells has
    expression). In addition, Rank-biserial correlation (rbc) and qval are calculated. The rank biserial correlation is
    used to assess the relationship between a dichotomous categorical variable and an ordinal variable. The rank
    biserial test is very similar to the non-parametric Mann-Whitney U test that is used to compare two independent
    groups on an ordinal variable. Mann-Whitney U tests are preferable to rank biserial correlations when comparing
    independent groups. Rank biserial correlations can only be used with dichotomous (two levels) categorical variables.
    qval is calculated using Benjamini-Hochberg adjustment.

    Args:
        adata: An AnnData object.
        genes: The list of genes that will be used to subset the data for dimension reduction and clustering. If `None`,
            all genes will be used.
        layer: The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is
            used.
        group: The column key/name that identifies the grouping information (for example, clusters that correspond to
            different cell types or different time points) of cells. This will be used for calculating group-specific
            genes.
        test_group: The group name from `group` for which markers has to be found.
        control_groups: The list of group name(s) from `group` for which markers has to be tested against.
        X_data: the user supplied data that will be used for marker gene detection directly. Defaults to None.
        exp_frac_thresh: the minimum percentage of cells with expression for a gene to proceed differential expression
            test. If `layer` is not `velocity` related (i.e. `velocity_S`), `exp_frac_thresh` by default is set to be
            0.1, otherwise 0. Defaults to None.
        log2_fc_thresh: The minimal threshold of log2 fold change for a gene to proceed differential expression test. If
            `layer` is not `velocity` related (i.e. `velocity_S`), `log2_fc_thresh` by default is set to be 1, otherwise
            0. Defaults to None.
        qval_thresh: The maximal threshold of qval to be considered as significant genes. Defaults to 0.05.
        subset_control_vals: Whether to subset the top ranked control values. When `subset_control_vals = None`, this is
            subset to be `True` when layer is not related to either `velocity` related or `acceleration` or `curvature`
            related layers and `False` otherwise. When layer is not related to either `velocity` related or
            `acceleration` or `curvature` related layers used, the control values will be sorted by absolute values.
            Defaults to None.

    Raises:
        ValueError: `X_data` is provided but `genes` does not correspond to its columns.

    Returns:
        A pandas DataFrame of the differential expression analysis result between the two groups.
    """

    if layer is None or not (layer.startswith("velocity") or layer in ["acceleration", "curvature"]):
        exp_frac_thresh = 0.1 if exp_frac_thresh is None else exp_frac_thresh
        log2_fc_thresh = 1 if log2_fc_thresh is None else log2_fc_thresh
        subset_control_vals = True if subset_control_vals is None else subset_control_vals
    else:
        exp_frac_thresh = 0 if exp_frac_thresh is None else exp_frac_thresh
        log2_fc_thresh = None
        subset_control_vals = False if subset_control_vals is None else subset_control_vals

    if X_data is None:
        genes, X_data = fetch_X_data(adata, genes, layer)
    else:
        if genes is None or len(genes) != X_data.shape[1]:
            raise ValueError(
                f"When providing X_data, a list of genes name that corresponds to the columns of X_data "
                f"must be provided"
            )

    n_cells, n_genes = X_data.shape
    sparse = issparse(X_data)

    if type(control_groups) == str:
        control_groups = [control_groups]
    test_cells, control_cells = (
        adata.obs[group] == test_group,
        adata.obs[group].isin(control_groups),
    )

    num_test_cells = test_cells.sum()
    num_groups = len(control_groups)

    if subset_control_vals:
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

            if subset_control_vals:
                if layer is None or not (layer.startswith("velocity") or layer in ["acceleration", "curvature"]):
                    control_vals.sort()
                else:
                    control_vals = control_vals[np.argsort(np.absolute(control_vals))]

                control_vals = control_vals[-min_n[i] :]
                cur_n1n2 = n1n2[i]
            else:
                cur_n1n2 = num_test_cells * len(control_vals)

            if log2_fc_thresh is not None:
                mean_control_vals = control_vals.mean()
                if mean_control_vals == 0:
                    log_fc = np.inf
                else:
                    log_fc = log_mean_test_vals - np.log2(mean_control_vals)

                if abs(log_fc) < log2_fc_thresh:
                    continue
            else:
                log_fc = test_vals.mean() - control_vals.mean()  # for curvature, acceleration, log fc is meaningless

            try:
                u, mw_p = map(float, mannwhitneyu(test_vals, control_vals))
            except ValueError:
                pass
            else:
                rbc = 1 - ((2 * u) / cur_n1n2)

            perfect_specificity = np.repeat(0.0, num_groups + 1)
            perfect_specificity[i + 1] = 1.0

            specificity_ = specificity(perc, perfect_specificity)

            tmp0, tmp1 = sum(np.sign(test_vals) > 0), sum(np.sign(control_vals) > 0)
            tmp0 = tmp0 if np.isscalar(tmp0) else tmp0[0]
            tmp1 = tmp0 if np.isscalar(tmp1) else tmp1[0]

            diff_ratio_pos = tmp0 / len(test_vals) - tmp1 / len(control_vals)

            de.append(
                (
                    gene,
                    control_groups[i],
                    ef,
                    rbc,
                    log_fc,
                    mw_p,
                    specificity_,
                    diff_ratio_pos,
                )
            )

    de = pd.DataFrame(
        de,
        columns=[
            "gene",
            "versus_group",
            "exp_frac",
            "rbc",
            "log2_fc",
            "pval",
            "specificity",
            "diff_ratio_pos",
        ],
    )
    de = de[de.iloc[:, 2:].sum(1) > 0]

    if de.shape[0] > 1:
        de["qval"] = multipletests(de["pval"].values, method="fdr_bh")[1]
    else:
        de["qval"] = [np.nan for _ in range(de.shape[0])]
    de["test_group"] = [test_group for _ in range(de.shape[0])]
    out_order = [
        "gene",
        "test_group",
        "versus_group",
        "specificity",
        "exp_frac",
        "diff_ratio_pos",
        "rbc",
        "log2_fc",
        "pval",
        "qval",
    ]
    de = de[out_order].sort_values(by="qval")
    res = de[(de.qval < qval_thresh)].reset_index().drop(columns=["index"])

    return res


def top_n_markers(
    adata: AnnData,
    with_moran_i: bool = False,
    group_by: str = "test_group",
    sort_by: Union[str, List[str]] = "specificity",
    sort_order: Literal["increasing", "decreasing"] = "decreasing",
    top_n_genes: int = 5,
    exp_frac_thresh: float = 0.1,
    log2_fc_thresh: Optional[float] = None,
    qval_thresh: float = 0.05,
    specificity_thresh: float = 0.3,
    only_gene_list: bool = False,
    display: bool = True,
) -> Union[List[List[str]], pd.DataFrame]:
    """Filter cluster deg (Moran's I test) results and retrieve top markers for each cluster.

    Args:
        adata: An AnnData object.
        with_moran_i: Whether to include Moran's I test results for selecting top marker genes. Defaults to False.
        group_by: Column name or names to group by. Defaults to "test_group".
        sort_by: Column name or names to sort by. Defaults to "specificity".
        sort_order: Whether to sort the data frame with `increasing` or `decreasing` order. Defaults to "decreasing".
        top_n_genes: The number of top sorted markers. Defaults to 5.
        exp_frac_thresh: The minimum percentage of cells with expression for a gene to proceed selection of top markers.
            Defaults to 0.1.
        log2_fc_thresh: The minimal threshold of log2 fold change for a gene to proceed selection of top markers.
            Applicable to none `velocity`, `acceleration` or `curvature` layers based DEGs. Defaults to None.
        qval_thresh: The maximal threshold of qval to be considered as top markers. Defaults to 0.05.
        specificity_thresh: The minimum threshold of specificity to be considered as top markers. Defaults to 0.3.
        only_gene_list: Whether to only return the gene list for each cluster. Defaults to False.
        display: Whether to print the data frame for the top marker genes after the filtering. Defaults to True.

    Raises:
        ValueError: Threshold too extreme that no genes passed the filter.

    Returns:
        If `only_gene_list` is false, a data frame that stores the top marker for each group would be returned.
        Otherwise, a list containing lists of genes in each cluster would be returned. In addition, it will display the
        data frame depending on whether `display` is set to be True.
    """

    if "cluster_markers" not in adata.uns.keys():
        main_warning(
            f"No info of cluster markers stored in your adata. "
            f"Running `find_group_markers` with default parameters."
        )
        adata = find_group_markers(adata, group="clusters")

    deg_table = adata.uns["cluster_markers"]["deg_table"]

    if len(deg_table["log2_fc"].unique()) > 1:
        if np.abs(deg_table["log2_fc"]).max() < 1:
            log2_fc_thresh = -np.inf if log2_fc_thresh is None else log2_fc_thresh
        else:
            log2_fc_thresh = 1 if log2_fc_thresh is None else log2_fc_thresh
        deg_table = deg_table.query(
            "exp_frac > @exp_frac_thresh and "
            "log2_fc > @log2_fc_thresh and "
            "qval < @qval_thresh and "
            "specificity > @specificity_thresh"
        )
    else:
        deg_table = deg_table.query(
            "exp_frac > @exp_frac_thresh and " "qval < @qval_thresh and " "specificity > @specificity_thresh"
        )

    if deg_table.shape[0] == 0:
        raise ValueError(
            f"Looks like your filter threshold is too extreme. No gene detected. "
            f"Please try relaxing the thresholds you specified: "
            f"exp_frac_thresh: {exp_frac_thresh}"
            f"log2_fc_thresh: {log2_fc_thresh}"
            f"qval_thresh: {qval_thresh}"
            f"specificity_thresh: {specificity_thresh}"
        )
    if with_moran_i:
        moran_i_columns = ["moran_i", "moran_p_val", "moran_q_val", "moran_z"]
        if len(adata.var.columns.intersection(moran_i_columns)) != 4:
            main_warning(
                f"No info of cluster markers stored in your adata. "
                f"Running `find_group_markers` with default parameters."
            )
            adata = moran_i(adata)

        moran_i_df = adata.var[moran_i_columns]
        deg_table = deg_table.merge(moran_i_df, left_on="gene", right_index=True, how="left")

    top_n_df = (
        deg_table.groupby(group_by).apply(lambda grp: grp.nlargest(top_n_genes, sort_by))
        if sort_order == "decreasing"
        else deg_table.groupby(group_by).apply(lambda grp: grp.nsmallest(top_n_genes, sort_by))
    )

    if display:
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            1000,
        ):
            print(top_n_df)

    top_n_groups = top_n_df.loc[:, group_by].unique()
    de_genes = [None] * len(top_n_groups)

    if only_gene_list:
        for i in top_n_groups:
            de_genes[i] = top_n_df[top_n_df[group_by] == i].loc[:, "gene"].to_list()
        return de_genes
    else:
        return top_n_df


def glm_degs(
    adata: AnnData,
    X_data: Optional[np.ndarray] = None,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    fullModelFormulaStr: str = "~cr(integral_time, df=3)",
    reducedModelFormulaStr: str = "~1",
    family: Literal["NB2"] = "NB2",
) -> None:
    """Differential genes expression tests using generalized linear regressions.

    The results would be stored in the adata's .uns["glm_degs"] annotation and the update is inplace.

    Tests each gene for differential expression as a function of integral time (the time estimated via the reconstructed
    vector field function) or pseudotime using generalized additive models with natural spline basis. This function can
    also use other covariates as specified in the full (i.e `~clusters`) and reduced model formula to identify
    differentially expression genes across different categories, group, etc.

    glm_degs relies on statsmodels package and is adapted from the `differentialGeneTest` function in Monocle. Note that
    glm_degs supports performing deg analysis for any layer or normalized data in your adata object. That is you can
    either use the total, new, unspliced or velocity, etc. for the differential expression analysis.

    Args:
        adata: An AnnData object.
        X_data: The user supplied data that will be used for differential expression analysis directly. Defaults to
            None.
        genes: The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is
            used. Defaults to None.
        layer: The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is
            used. Defaults to None.
        fullModelFormulaStr: A formula string specifying the full model in differential expression tests (i.e.
            likelihood ratio tests) for each gene/feature. Defaults to "~cr(integral_time, df=3)".
        reducedModelFormulaStr: A formula string specifying the reduced model in differential expression tests (i.e.
            likelihood ratio tests) for each gene/feature. Defaults to "~1".
        family: The distribution family used for the expression responses in statsmodels. Currently, always uses `NB2`
            and this is ignored. NB model requires us to define a parameter alpha which it uses to express the
            variance in terms of the mean as follows: variance = mean + alpha mean^p. When p=2, it corresponds to
            the NB2 model. In order to obtain the correct parameter alpha (sm.genmod.families.family.NegativeBinomial
            (link=None, alpha=1.0), by default it is 1), we use the auxiliary OLS regression without a constant from
            Messrs Cameron and Trivedi. More details can be found here:
            https://towardsdatascience.com/negative-binomial-regression-f99031bb25b4. Defaults to "NB2".

    Raises:
        ValueError: `X_data` is provided but `genes` does not correspond to its columns.
        Exception: Factors from the model formula `fullModelFormulaStr` invalid.
    """

    if X_data is None:
        genes, X_data = fetch_X_data(adata, genes, layer)
    else:
        if genes is None or len(genes) != X_data.shape[1]:
            raise ValueError(
                f"When providing X_data, a list of genes name that corresponds to the columns of X_data "
                f"must be provided"
            )

    norm_method_key = "X_norm_method" if layer is None or layer == "X" else "layers_norm_method"
    if layer is None:
        if issparse(X_data):
            X_data.data = (
                2**X_data.data - 1
                if adata.uns["pp"][norm_method_key] == "log2"
                else np.exp(X_data.data) - 1
                if adata.uns["pp"][norm_method_key] == "log"
                else _Freeman_Tukey(X_data.data + 1, inverse=True) - 1
                if adata.uns["pp"][norm_method_key] == "Freeman_Tukey"
                else X_data.data
            )
        else:
            X_data = (
                2**X_data - 1
                if adata.uns["pp"][norm_method_key] == "log2"
                else np.exp(X_data) - 1
                if adata.uns["pp"][norm_method_key] == "log"
                else _Freeman_Tukey(X_data, inverse=True)
                if adata.uns["pp"][norm_method_key] == "Freeman_Tukey"
                else X_data
            )

    factors = get_all_variables(fullModelFormulaStr)
    factors = ["Pseudotime" if i == "cr(Pseudotime, df=3)" else i for i in factors]
    if len(set(factors).difference(adata.obs.columns)) == 0:
        df_factors = adata.obs[factors]
    else:
        raise Exception(
            f"adata object doesn't include the factors from the model formula " f"{fullModelFormulaStr} you provided."
        )

    sparse = issparse(X_data)
    deg_df = pd.DataFrame(index=genes, columns=["status", "family", "pval"])
    for i, gene in tqdm(
        enumerate(genes),
        "Detecting time dependent genes via Generalized Additive Models (GAMs)",
    ):
        expression = X_data[:, i].A if sparse else X_data[:, i]
        df_factors["expression"] = expression
        deg_df.iloc[i, :] = diff_test_helper(df_factors, fullModelFormulaStr, reducedModelFormulaStr)

    deg_df["qval"] = multipletests(deg_df["pval"], method="fdr_bh")[1]

    adata.uns["glm_degs"] = deg_df


def diff_test_helper(
    data: pd.DataFrame,
    fullModelFormulaStr: str = "~cr(time, df=3)",
    reducedModelFormulaStr: str = "~1",
) -> Union[Tuple[Literal["fail"], Literal["NB2"], Literal[1]], Tuple[Literal["ok"], Literal["NB2"], np.ndarray],]:
    """A helper function to generate required data fields for differential gene expression test.

    Args:
        data: The original dataframe containing expression data.
        fullModelFormulaStr: A formula string specifying the full model in differential expression tests (i.e.
            likelihood ratio tests) for each gene/feature. Defaults to "~cr(integral_time, df=3)".
        reducedModelFormulaStr: A formula string specifying the reduced model in differential expression tests (i.e.
            likelihood ratio tests) for each gene/feature. Defaults to "~1".
    Returns:
        A tuple [parseResult, family, pval], where `parseResult` should be "ok" or "fail", showing whether the provided
        dataframe is successfully parsed or not. `family` is the distribution family used for the expression responses
        in statsmodels, currently only "NB2" is supported. `pval` is the survival probability (1-cumulative probability)
        to observe the likelihood ratio for the constrained model to be true. If parsing dataframe failed, this value is
        set to be 1.
    """
    # Dividing data into train and validation datasets
    transformed_x = dmatrix(fullModelFormulaStr, data, return_type="dataframe")
    transformed_x_null = dmatrix(reducedModelFormulaStr, data, return_type="dataframe")

    expression = data["expression"]

    try:
        # poisson_training_results = sm.GLM(expression, transformed_x, family=sm.families.Poisson()).fit()
        # poisson_df = pd.DataFrame({"mu": poisson_training_results.mu, "expression": expression})
        # poisson_df["AUX_OLS_DEP"] = poisson_df.apply(
        #     lambda x: ((x["expression"] - x["mu"]) ** 2 - x["expression"]) / x["mu"],
        #     axis=1,
        # )
        # ols_expr = """AUX_OLS_DEP ~ mu - 1"""
        # aux_olsr_results = smf.ols(ols_expr, poisson_df).fit()

        nb2_family = sm.families.NegativeBinomial()  # (alpha=aux_olsr_results.params[0])

        nb2_full = sm.GLM(expression, transformed_x, family=nb2_family).fit()
        nb2_null = sm.GLM(expression, transformed_x_null, family=nb2_family).fit()
    except:
        return ("fail", "NB2", 1)

    pval = lrt(nb2_full, nb2_null)
    return ("ok", "NB2", pval)


def get_all_variables(formula: str) -> List[str]:
    """A helper function to get all variable names for a formula string.

    Args:
        formula: The formula string specifying the model in differential expression data.

    Returns:
        A list of variable names.
    """
    md = patsy.ModelDesc.from_formula(formula)
    termlist = md.rhs_termlist + md.lhs_termlist

    factors = []
    for term in termlist:
        for factor in term.factors:
            factors.append(factor.name())

    return factors


def lrt(full: GLMResultsWrapper, restr: GLMResultsWrapper) -> np.float64:
    """Perform likelihood-ratio test on the full model and constrained model.

    Args:
        full: The regression model without constraint.
        restr: The regression model after constraint.

    Returns:
        The survival probability (1-cumulative probability) to observe the likelihood ratio for the constrained
        model to be true.
    """

    llf_full = full.llf
    llf_restr = restr.llf
    df_full = full.df_resid
    df_restr = restr.df_resid
    lrdf = df_restr - df_full
    lrstat = -2 * (llf_restr - llf_full)
    lr_pvalue = stats.chi2.sf(lrstat, df=lrdf)

    return lr_pvalue


def specificity(percentage: np.ndarray, perfect_specificity: np.ndarray) -> float:
    """Calculate specificity"""

    spec = 1 - JSdistVec(makeprobsvec(percentage), perfect_specificity)

    return spec


def makeprobsvec(p: np.ndarray) -> np.ndarray:
    """Calculate the probability matrix for a relative abundance matrix"""

    phat = p / np.sum(p)
    phat[np.isnan((phat))] = 0

    return phat


def shannon_entropy(p: np.ndarray) -> float:
    """Calculate the Shannon entropy based on the probability vector"""

    if np.min(p) < 0 or np.sum(p) <= 0:
        return np.inf
    p_norm = p[p > 0] / np.sum(p)

    return -np.sum(np.log(p_norm) * p_norm)


def JSdistVec(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate the Jessen-Shannon distance for two probability distribution"""

    Jsdiv = shannon_entropy((p + q) / 2) - (shannon_entropy(p) + shannon_entropy(q)) / 2
    if np.isinf(Jsdiv):
        Jsdiv = 1
    if Jsdiv < 0:
        Jsdiv = 0
    JSdist = np.sqrt(Jsdiv)

    return JSdist
