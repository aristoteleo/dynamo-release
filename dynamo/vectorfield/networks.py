from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData

from ..dynamo_logger import main_debug, main_info, main_tqdm
from .rank_vf import rank_jacobian_genes


def get_interaction_in_cluster(
    rank_df_dict: Dict[str, pd.DataFrame],
    group: str,
    genes: List,
    n_top_genes: int = 100,
    rank_regulators: bool = False,
    negative_values: bool = False,
) -> pd.DataFrame:
    """Retrieve interactions among input genes given the ranking dataframe.

    Args:
        rank_df_dict: The dictionary of pandas data frame storing the gene ranking information for each cluster.
        group: The group name that points to the key for the rank_df.
        genes: The list of input genes, from which the network will be constructed.
        n_top_genes: Number of top genes that will be selected from to build the network.
        rank_regulators: Whether the input dictionary is about ranking top regulators of each gene per cluster.

    Returns:
        A dataframe of interactions between input genes for the specified group of cells based on ranking information
        of Jacobian analysis. It has `regulator`, `target` and `weight` three columns.

    """

    subset_rank_df = rank_df_dict[group].head(n_top_genes)
    if negative_values:
        subset_rank_df = pd.concat([subset_rank_df, rank_df_dict[group].tail(n_top_genes)])
    valid_genes = subset_rank_df.columns.intersection(genes).to_list()
    edges = None

    if len(valid_genes) > 0:
        top_n_genes_df = subset_rank_df.loc[:, valid_genes]
        valid_genes_values = [i + "_values" for i in valid_genes]
        top_n_genes_values_df = subset_rank_df.loc[:, valid_genes_values]

        for cur_gene in valid_genes:
            targets = list(set(top_n_genes_df[cur_gene].values).intersection(valid_genes))
            t_n = len(targets)

            if t_n > 0:
                targets_inds = [list(top_n_genes_df[cur_gene].values).index(i) for i in targets]

                targets_values = top_n_genes_values_df.loc[:, cur_gene + "_values"].iloc[targets_inds].values

                if rank_regulators:
                    tmp = pd.DataFrame(
                        {
                            "regulator": targets,
                            "target": np.repeat(cur_gene, t_n),
                            "weight": targets_values,
                        }
                    )
                else:
                    tmp = pd.DataFrame(
                        {
                            "regulator": np.repeat(cur_gene, t_n),
                            "target": targets,
                            "weight": targets_values,
                        }
                    )

                edges = tmp if edges is None else pd.concat((edges, tmp), axis=0)

    return edges


def build_network_per_cluster(
    adata: AnnData,
    cluster: str,
    cluster_names: Optional[str] = None,
    full_reg_rank: Optional[Dict] = None,
    full_eff_rank: Optional[Dict] = None,
    genes: Optional[List] = None,
    n_top_genes: int = 100,
    abs: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Build a cluster specific network between input genes based on ranking information of Jacobian analysis.

    Args:
        adata: AnnData object, must at least have gene-wise Jacobian matrix calculated for each or selected cell.
        cluster: The group key that points to the columns of `adata.obs`.
        cluster_names: The groups whose networks will be constructed, must overlap with names in adata.obs and / or keys from the
            ranking dictionaries.
        full_reg_rank: The dictionary stores the regulator ranking information per cluster based on cell-wise Jacobian matrix. If
            None, we will call `rank_jacobian_genes(adata, groups=cluster, mode='full reg', abs=True,
            output_values=True)` to first obtain this dictionary.
        full_eff_rank: The dictionary stores the effector ranking information per cluster based on cell-wise Jacobian matrix. If
            None, we will call `rank_jacobian_genes(adata, , groups=cluster, mode='full eff', abs=True,
            output_values=True)` to first obtain this dictionary.
        genes: The list of input genes, from which the network will be constructed.
        n_top_genes: Number of top genes that will be selected from to build the network.

    Returns:
        A dictionary of dataframe of interactions between input genes for each group of cells based on ranking
        information of Jacobian analysis. Each composite dataframe has `regulator`, `target` and `weight` three columns.
    """

    genes = np.unique(genes)
    if full_reg_rank is None:
        full_reg_rank = rank_jacobian_genes(
            adata, groups=cluster, mode="full reg", abs=abs, output_values=True, return_df=True
        )
    if full_eff_rank is None:
        full_eff_rank = rank_jacobian_genes(
            adata, groups=cluster, mode="full eff", abs=abs, output_values=True, return_df=True
        )

    edges_list = {}

    reg_groups, eff_groups = full_reg_rank.keys(), full_eff_rank.keys()
    if reg_groups != eff_groups:
        raise Exception(f"the regulators ranking and effector ranking dataframe must have the same keys.")

    if cluster_names is not None:
        reg_groups = list(set(reg_groups).intersection(cluster_names))
        if len(reg_groups) == 0:
            raise ValueError(
                f"the clusters argument {cluster_names} provided doesn't match up with any clusters from the " f"adata."
            )

    for c in main_tqdm(reg_groups, desc="iterating reg_groups"):
        if genes is None:
            reg_valid_genes, eff_valid_genes = (
                full_reg_rank[c].columns.values,
                full_eff_rank[c].columns.values,
            )
            reg_valid_genes = reg_valid_genes[np.arange(0, len(reg_valid_genes), 2)].tolist()
            eff_valid_genes = eff_valid_genes[np.arange(0, len(eff_valid_genes), 2)].tolist()
        else:
            reg_valid_genes, eff_valid_genes = (
                full_reg_rank[c].columns.intersection(genes),
                full_eff_rank[c].columns.intersection(genes),
            )
        if len(reg_valid_genes) > 0:
            reg_df = get_interaction_in_cluster(
                full_reg_rank,
                c,
                reg_valid_genes,
                n_top_genes=n_top_genes,
                rank_regulators=True,
                negative_values=not abs,
            )
        if len(eff_valid_genes) > 0:
            eff_df = get_interaction_in_cluster(
                full_eff_rank,
                c,
                eff_valid_genes,
                n_top_genes=n_top_genes,
                rank_regulators=False,
                negative_values=not abs,
            )

        if len(reg_valid_genes) > 0 and len(eff_valid_genes) == 0:
            edges_list[c] = reg_df
        elif len(reg_valid_genes) == 0 and len(eff_valid_genes) > 0:
            edges_list[c] = eff_df
        elif len(reg_valid_genes) > 0 and len(eff_valid_genes) > 0:
            edges_list[c] = pd.concat((reg_df, eff_df), axis=0)

    return edges_list


def adj_list_to_matrix(
    adj_list: pd.DataFrame, only_one_edge: bool = False, clr: bool = False, graph: bool = False
) -> Union[pd.DataFrame, nx.Graph]:
    """Convert a pandas adjacency list (with regulator, target, weight columns) to a processed adjacency matrix (or
    network).

    Args:
        adj_list: A pandas adjacency dataframe with regulator, target, weight columns for representing a network graph.
        only_one_edge: Whether or not to only keep the edges with higher weight for any two gene pair.
        clr: Whether to post-process the direct network via the context likelihood relatedness.
        graph: Whether a direct, weighted graph based on networkx should be returned.

    Returns:
        A pandas adjacency matrix or a direct, weighted graph constructed via networkx.
    """

    uniq_genes = list(set(adj_list.regulator) | set(adj_list.target))

    adj_matrix = pd.DataFrame(0, index=uniq_genes, columns=uniq_genes)

    for i, row in adj_list.iterrows():
        adj_matrix.loc[row.regulator, row.target] = row.weight

    if only_one_edge:
        adj_matrix[adj_matrix - adj_matrix.T < 0] = 0
    if clr:
        adj_matrix = clr_directed(adj_matrix)
    if graph:
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                f"You need to install the package `networkx`." f"install networkx via `pip install networkx`."
            )

        network = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())
        return network
    else:
        return adj_matrix


def clr_directed(adj_mat):
    """clr on directed graph"""
    col_means = adj_mat.mean(axis=0)
    col_sd = adj_mat.std(axis=0)
    col_sd[col_sd == 0] = -1e-4

    updated_adj_mat = adj_mat.copy()
    for i, row_i in adj_mat.iterrows():
        row_mean, row_sd = np.mean(row_i), np.std(row_i)
        if row_sd == 0:
            row_sd = -1e-4

        s_i_vec = np.maximum(0, (row_i - row_mean) / row_sd)
        s_j_vec = np.maximum(0, (row_i - col_means) / col_sd)
        updated_adj_mat.loc[i, :] = np.sqrt(s_i_vec**2 + s_j_vec**2)

    return updated_adj_mat
