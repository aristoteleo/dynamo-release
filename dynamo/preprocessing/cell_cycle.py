# This file is adapted from the perturbseq library by Thomas Norman
# https://github.com/thomasmaxwellnorman/perturbseq_demo/blob/master/perturbseq/cell_cycle.py

from collections import OrderedDict
from typing import List, Tuple, Union

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from ..tools.utils import einsum_correlation, log1p_
from ..utils import LoggerManager, copy_adata


def group_corr(adata: anndata.AnnData, layer: Union[str, None], gene_list: list) -> Tuple[np.ndarray, np.ndarray]:
    """Measures the correlation of all genes within a list to the average expression of all genes within that
    list (used for staging cell cycle phase of each cell).

    Args:
        adata: an anndata object.
        layer: The layer of data to use for calculating correlation. If None, use adata.X.
        gene_list: list of gene names

    Raises:
        Exception: if gene_list is empty, raise this exception

    Returns:
        A tuple (valid_gene_list, corr). valid_gene_list contains valid gene names, e.g. gene names that intersect
        between input gene_list and the adata.var_names. corr contains the correlation coefficient of each gene with
        the mean expression of all genes in the list.
    """

    # returns list of correlations of each gene within a list of genes with the total expression of the group
    tmp = adata.var_names.intersection(gene_list)
    # get the location of gene names
    intersect_genes = [adata.var.index.get_loc(i) for i in tmp]

    if len(intersect_genes) == 0:
        raise Exception(f"your adata doesn't have any gene from the gene_list {gene_list}.")

    if layer is None:
        expression_matrix = adata.X[:, intersect_genes]
    else:
        expression_matrix = adata.layers[layer][:, intersect_genes]
        expression_matrix = log1p_(adata, expression_matrix)

    avg_exp = expression_matrix.mean(axis=1)
    cor = (
        einsum_correlation(
            np.array(expression_matrix.A.T, dtype="float"),
            np.array(avg_exp.A1, dtype="float"),
        )
        if issparse(expression_matrix)
        else einsum_correlation(
            np.array(expression_matrix.T, dtype="float"),
            np.array(avg_exp, dtype="float"),
        )
    )

    # get back to gene names again
    return np.array(adata.var.index[intersect_genes]), cor.flatten()


def refine_gene_list(
    adata: anndata.AnnData,
    layer: Union[str, None],
    gene_list: List[str],
    threshold: Union[float, None],
    return_corrs: bool = False,
) -> list:
    """Refines a list of genes by removing those that don't correlate well with the average expression of
    those genes.

    Args:
        adata: an anndata object.
        layer: The layer of data to use for calculating correlation. If None, use adata.X.
        gene_list: list of gene names.
        threshold: threshold on correlation coefficient used to discard genes (expression of each gene is compared to
            the bulk expression of the group and any gene with a correlation coefficient less than this is discarded).
        return_corrs: whether to return the correlations along with the gene names. Defaults to False.

    Returns:
        A refined list of genes that are well correlated with the average expression trend.
    """

    gene_list, corrs = group_corr(adata, layer, gene_list)
    if return_corrs:
        return corrs[corrs >= threshold]
    else:
        return gene_list[corrs >= threshold]


def group_score(adata: anndata.AnnData, layer: Union[str, None], gene_list: List[str]) -> pd.Series:
    """Scores cells within population for expression of a set of genes. Raw expression data are first
    log transformed, then the values are summed, and then scores are Z-normalized across all cells.

    Args:
        adata: an anndata object.
        layer: The layer of data to use for calculating correlation. If None, use adata.X.
        gene_list: list of gene names.

    Raises:
        Exception: if gene_list is empty, raise this exception

    Returns:
        The Z-scored expression data.
    """

    tmp = adata.var_names.intersection(gene_list)
    # use indices
    intersect_genes = [adata.var_names.get_loc(i) for i in tmp]

    if len(intersect_genes) == 0:
        raise Exception(f"your adata doesn't have any gene from the gene_list {gene_list}.")

    if layer is None:
        expression_matrix = adata.X[:, intersect_genes]
    else:
        expression_matrix = adata.layers[layer][:, intersect_genes]
        expression_matrix = log1p_(adata, expression_matrix)

    # TODO FutureWarning: Index.is_all_dates is deprecated, will be removed in a future version.
    # check index.inferred_type instead
    if layer is None or layer.startswith("X_"):
        scores = expression_matrix.sum(1).A1 if issparse(expression_matrix) else expression_matrix.sum(1)
    else:
        if issparse(expression_matrix):
            expression_matrix.data = np.log1p(expression_matrix.data)
            scores = expression_matrix.sum(1).A1
        else:
            scores = np.log1p(expression_matrix).sum(1)

    scores = (scores - scores.mean()) / scores.std()

    return scores


def batch_group_score(adata: anndata.AnnData, layer: Union[str, None], gene_lists: List[str]) -> OrderedDict:
    """Scores cells within population for expression of sets of genes. Raw expression data are first log transformed,
    then the values are summed, and then scores are Z-normalized across all cells. Returns an OrderedDict of each score.

    Args:
        adata: an anndata object.
        layer: The layer of data to use for calculating correlation. If None, use adata.X.
        gene_lists: list of lists of gene names.

    Returns:
        An OrderedDict of each score.
    """

    batch_scores = OrderedDict()
    for gene_list in gene_lists:
        batch_scores[gene_list] = group_score(adata, layer, gene_lists[gene_list])
    return batch_scores


def get_cell_phase_genes(
    adata: anndata.AnnData,
    layer: Union[str, None],
    refine: bool = True,
    threshold: Union[float, None] = 0.3,
) -> OrderedDict:
    """Returns a list of cell-cycle-regulated marker genes, filtered for coherence

    Args:
        adata (anndata.AnnData): an anndata object.
        layer (Union[str, None]): The layer of data to use for calculating correlation. If None, use adata.X.
        refine (bool, optional): whether to refine the gene lists based on how consistent the expression is among the
            groups. Defaults to True.
        threshold (Union[float, None], optional): threshold on correlation coefficient used to discard genes (expression
            of each gene is compared to the bulk expression of the group and any gene with a correlation coefficient
            less than this is discarded). Defaults to 0.3.

    Returns:
        A list of cell-cycle-regulated marker genes that show strong co-expression.
    """

    cell_phase_genes = OrderedDict()
    cell_phase_genes["G1-S"] = pd.Series(
        [
            "ARGLU1",
            "BRD7",
            "CDC6",
            "CLSPN",
            "ESD",
            "GINS2",
            "GMNN",
            "LUC7L3",
            "MCM5",
            "MCM6",
            "NASP",
            "PCNA",
            "PNN",
            "SLBP",
            "SRSF7",
            "SSR3",
            "ZRANB2",
        ]
    )
    cell_phase_genes["S"] = pd.Series(
        [
            "ASF1B",
            "CALM2",
            "CDC45",
            "CDCA5",
            "CENPM",
            "DHFR",
            "EZH2",
            "FEN1",
            "HIST1H2AC",
            "HIST1H4C",
            "NEAT1",
            "PKMYT1",
            "PRIM1",
            "RFC2",
            "RPA2",
            "RRM2",
            "RSRC2",
            "SRSF5",
            "SVIP",
            "TOP2A",
            "TYMS",
            "UBE2T",
            "ZWINT",
        ]
    )
    cell_phase_genes["G2-M"] = pd.Series(
        [
            "AURKB",
            "BUB3",
            "CCNA2",
            "CCNF",
            "CDCA2",
            "CDCA3",
            "CDCA8",
            "CDK1",
            "CKAP2",
            "DCAF7",
            "HMGB2",
            "HN1",
            "KIF5B",
            "KIF20B",
            "KIF22",
            "KIF23",
            "KIFC1",
            "KPNA2",
            "LBR",
            "MAD2L1",
            "MALAT1",
            "MND1",
            "NDC80",
            "NUCKS1",
            "NUSAP1",
            "PIF1",
            "PSMD11",
            "PSRC1",
            "SMC4",
            "TIMP1",
            "TMEM99",
            "TOP2A",
            "TUBB",
            "TUBB4B",
            "VPS25",
        ]
    )
    cell_phase_genes["M"] = pd.Series(
        [
            "ANP32B",
            "ANP32E",
            "ARL6IP1",
            "AURKA",
            "BIRC5",
            "BUB1",
            "CCNA2",
            "CCNB2",
            "CDC20",
            "CDC27",
            "CDC42EP1",
            "CDCA3",
            "CENPA",
            "CENPE",
            "CENPF",
            "CKAP2",
            "CKAP5",
            "CKS1B",
            "CKS2",
            "DEPDC1",
            "DLGAP5",
            "DNAJA1",
            "DNAJB1",
            "GRK6",
            "GTSE1",
            "HMG20B",
            "HMGB3",
            "HMMR",
            "HN1",
            "HSPA8",
            "KIF2C",
            "KIF5B",
            "KIF20B",
            "LBR",
            "MKI67",
            "MZT1",
            "NUF2",
            "NUSAP1",
            "PBK",
            "PLK1",
            "PRR11",
            "PSMG3",
            "PWP1",
            "RAD51C",
            "RBM8A",
            "RNF126",
            "RNPS1",
            "RRP1",
            "SFPQ",
            "SGOL2",
            "SMARCB1",
            "SRSF3",
            "TACC3",
            "THRAP3",
            "TPX2",
            "TUBB4B",
            "UBE2D3",
            "USP16",
            "WIBG",
            "YWHAH",
            "ZNF207",
        ]
    )
    cell_phase_genes["M-G1"] = pd.Series(
        [
            "AMD1",
            "ANP32E",
            "CBX3",
            "CDC42",
            "CNIH4",
            "CWC15",
            "DKC1",
            "DNAJB6",
            "DYNLL1",
            "EIF4E",
            "FXR1",
            "GRPEL1",
            "GSPT1",
            "HMG20B",
            "HSPA8",
            "ILF2",
            "KIF5B",
            "KPNB1",
            "LARP1",
            "LYAR",
            "MORF4L2",
            "MRPL19",
            "MRPS2",
            "MRPS18B",
            "NUCKS1",
            "PRC1",
            "PTMS",
            "PTTG1",
            "RAN",
            "RHEB",
            "RPL13A",
            "SRSF3",
            "SYNCRIP",
            "TAF9",
            "TMEM138",
            "TOP1",
            "TROAP",
            "UBE2D3",
            "ZNF593",
        ]
    )

    if refine:
        for phase in cell_phase_genes:
            cur_cell_phase_genes = None
            if adata.var_names[0].isupper():
                cur_cell_phase_genes = cell_phase_genes[phase]
            elif adata.var_names[0][0].isupper() and adata.var_names[0][1:].islower():
                cur_cell_phase_genes = [gene.capitalize() for gene in cell_phase_genes[phase]]
            else:
                cur_cell_phase_genes = [gene.lower() for gene in cell_phase_genes[phase]]

            cell_phase_genes[phase] = refine_gene_list(adata, layer, cur_cell_phase_genes, threshold)

    return cell_phase_genes


def get_cell_phase(
    adata: anndata.AnnData,
    layer: str = None,
    gene_list: Union[OrderedDict, None] = None,
    refine: bool = True,
    threshold: Union[float, None] = 0.3,
) -> pd.DataFrame:
    """Compute cell cycle phase scores for cells in the population

    Args:
        adata: an anndata object.
        layer: the layer of adata to use for calculating correlation. If None, use adata.X. Defaults to None.
        gene_list: OrderedDict of marker genes to use for cell cycle phases. If None, the default list will be used.
            Defaults to None.
        refine: whether to refine the gene lists based on how consistent the expression is among the groups. Defaults to
            True.
        threshold: threshold on correlation coefficient used to discard genes (expression of each gene is compared to
            the bulk expression of the group and any gene with a correlation coefficient less than this is discarded).
            Defaults to 0.3.

    Returns:
        Cell cycle scores indicating the likelihood a given cell is in a given cell cycle phase.
    """

    # get list of genes if one is not provided
    if gene_list is None:
        cell_phase_genes = get_cell_phase_genes(adata, layer, refine=refine, threshold=threshold)
    else:
        cell_phase_genes = gene_list

    adata.uns["cell_phase_genes"] = cell_phase_genes
    # score each cell cycle phase and Z-normalize
    phase_scores = pd.DataFrame(batch_group_score(adata, layer, cell_phase_genes))
    normalized_phase_scores = phase_scores.sub(phase_scores.mean(axis=1), axis=0).div(phase_scores.std(axis=1), axis=0)

    normalized_phase_scores_corr = normalized_phase_scores.transpose()
    normalized_phase_scores_corr["G1-S"] = [1, 0, 0, 0, 0]
    normalized_phase_scores_corr["S"] = [0, 1, 0, 0, 0]
    normalized_phase_scores_corr["G2-M"] = [0, 0, 1, 0, 0]
    normalized_phase_scores_corr["M"] = [0, 0, 0, 1, 0]
    normalized_phase_scores_corr["M-G1"] = [0, 0, 0, 0, 1]

    phase_list = ["G1-S", "S", "G2-M", "M", "M-G1"]

    # final scores for each phaase are correlation of expression profile with vectors defined above
    cell_cycle_scores = normalized_phase_scores_corr.corr()
    tmp = -len(phase_list)
    cell_cycle_scores = cell_cycle_scores[tmp:].transpose()[: -len(phase_list)]

    # pick maximal score as the phase for that cell
    cell_cycle_scores["cell_cycle_phase"] = cell_cycle_scores.idxmax(axis=1)
    cell_cycle_scores["cell_cycle_phase"] = cell_cycle_scores["cell_cycle_phase"].astype("category")
    cell_cycle_scores["cell_cycle_phase"].cat.set_categories(phase_list, inplace=True)

    def progress_ratio(x, phase_list):
        ind = phase_list.index(x["cell_cycle_phase"])
        return x[phase_list[(ind - 1) % len(phase_list)]] - x[phase_list[(ind + 1) % len(phase_list)]]

    # interpolate position within given cell cycle phase
    cell_cycle_scores["cell_cycle_progress"] = cell_cycle_scores.apply(
        lambda x: progress_ratio(x, list(phase_list)), axis=1
    )
    cell_cycle_scores.sort_values(
        ["cell_cycle_phase", "cell_cycle_progress"],
        ascending=[True, False],
        inplace=True,
    )

    # order of cell within cell cycle phase
    cell_cycle_scores["cell_cycle_order"] = cell_cycle_scores.groupby("cell_cycle_phase").cumcount()
    cell_cycle_scores["cell_cycle_order"] = cell_cycle_scores.groupby("cell_cycle_phase")["cell_cycle_order"].apply(
        lambda x: x / (len(x) - 1)
    )

    return cell_cycle_scores


def cell_cycle_scores(
    adata: anndata.AnnData,
    layer: Union[str, None] = None,
    gene_list: Union[OrderedDict, None] = None,
    refine: bool = True,
    threshold: float = 0.3,
    copy: bool = False,
) -> Union[anndata.AnnData, None]:
    """Estimate cell cycle stage of each cell based on its gene expression pattern.

    If more direct control is desired, use get_cell_phase.


    Args:
        adata: an anndata object.
        layer: The layer of data to use for calculating correlation. If None, use adata.X. Defaults to None.
        gene_list: OrderedDict of marker genes to use for cell cycle phases. If None, the default list will be used.
            Defaults to None.
        refine: whether to refine the gene lists based on how consistent the expression is among the groups. Defaults to
            True.
        threshold: threshold on correlation coefficient used to discard genes (expression of each gene is compared to
            the bulk expression of the group and any gene with a correlation coefficient less than this is discarded).
            Defaults to 0.3.
        copy: whether copy the original AnnData object and return it. Defaults to False.

    Returns:
        Returns an updated adata object with cell_cycle_phase as new column in .obs and a new data frame with
        `cell_cycle_scores` key to .obsm where the cell cycle scores indicating the likelihood a given cell is in a
        given cell cycle phase if `copy` is true. Otherwise, return None.
    """

    logger = LoggerManager.gen_logger("dynamo-cell-cycle-score")
    adata = copy_adata(adata, logger=logger) if copy else adata

    temp_timer_logger = LoggerManager.get_temp_timer_logger()
    temp_timer_logger.info("computing cell phase...")
    cell_cycle_scores = get_cell_phase(
        adata,
        layer=layer,
        refine=refine,
        gene_list=gene_list,
        threshold=threshold,
    )
    temp_timer_logger.finish_progress(progress_name="cell phase estimation")

    cell_cycle_scores.index = adata.obs_names[cell_cycle_scores.index.values.astype("int")]

    logger.info_insert_adata("cell_cycle_phase", adata_attr="obs")
    adata.obs["cell_cycle_phase"] = cell_cycle_scores["cell_cycle_phase"].astype("category")

    # adata.obsm['cell_cycle_scores'] = cell_cycle_scores.set_index(adata.obs_names)
    # .values
    logger.info_insert_adata("cell_cycle_scores", adata_attr="obsm")
    adata.obsm["cell_cycle_scores"] = cell_cycle_scores.loc[adata.obs_names, :]
    logger.finish_progress(progress_name="Cell Cycle Scores Estimation")

    # return the deep-copied adata if 'copy' is true
    if copy:
        return adata
