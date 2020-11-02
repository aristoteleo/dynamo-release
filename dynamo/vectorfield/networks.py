import numpy as np, pandas as pd
from .vector_calculus import rank_jacobian_genes

def get_interaction_in_cluster(rank_df_dict,
                               group,
                               genes,
                               n_top_genes=100,
                               rank_regulators=False):
    """Retrieve interactions among input genes given the ranking dataframe.

    Parameters
    ----------
        rank_df_dict: `dict` of `pandas.DataFrame`
            The dictionary of pandas data frame storing the gene ranking information for each cluster.
        group: `str`
            The group name that points to the key for the rank_df.
        genes: `list`
            The list of input genes, from which the network will be constructed.
        n_top_genes: `int`
            Number of top genes that will be selected from to build the network.
        rank_regulators
            Whether the input dictionary is about ranking top regulators of each gene per cluster.

    Returns
    -------
        A dataframe of interactions between input genes for the specified group of cells based on ranking information
        of Jacobian analysis. It has `regulator`, `target` and `weight` three columns.

    """

    subset_rank_df = rank_df_dict[group].head(n_top_genes)
    valid_genes = subset_rank_df.columns.intersection(genes).to_list()
    edges = None

    if len(valid_genes) > 0:
        top_n_genes_df = subset_rank_df.loc[:, valid_genes]
        valid_genes_values = [i + '_values' for i in valid_genes]
        top_n_genes_values_df = subset_rank_df.loc[:, valid_genes_values]

        for cur_gene in valid_genes:
            targets = list(set(top_n_genes_df[cur_gene].values).intersection(valid_genes))
            t_n = len(targets)

            if t_n > 0:
                targets_inds = [list(top_n_genes_df[cur_gene].values).index(i) for i in targets]

                targets_values = top_n_genes_values_df.loc[:, cur_gene + '_values'].iloc[targets_inds].values

                if rank_regulators:
                    tmp = pd.DataFrame({'regulator': targets,
                                        "target": np.repeat(cur_gene, t_n),
                                        "weight": targets_values})
                else:
                    tmp = pd.DataFrame({'regulator': np.repeat(cur_gene, t_n),
                                        "target": targets,
                                        "weight": targets_values})

                edges = tmp if edges is None else pd.concat((edges, tmp), axis=0)

    return edges


def build_network_per_cluster(adata,
                              cluster,
                              cluster_names=None,
                              full_reg_rank=None,
                              full_eff_rank=None,
                              genes=None,
                              n_top_genes=100):
    """Build a cluster specific network between input genes based on ranking information of Jacobian analysis.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`.
            AnnData object, must at least have gene-wise Jacobian matrix calculated for each or selected cell.
        cluster: `str`
            The group key that points to the columns of `adata.obs`.
        cluster_names: `str` or `list` (default: `None`)
            The groups whose networks will be constructed, must overlap with names in adata.obs and / or keys from the
            ranking dictionaries.
        full_reg_rank: `dict` (default: `None`)
            The dictionary stores the regulator ranking information per cluster based on cell-wise Jacobian matrix. If
            None, we will call `rank_jacobian_genes(adata, groups=cluster, mode='full reg', abs=True,
            output_values=True)` to first obtain this dictionary.
        full_eff_rank (default: `None`)
            The dictionary stores the effector ranking information per cluster based on cell-wise Jacobian matrix. If
            None, we will call `rank_jacobian_genes(adata, , groups=cluster, mode='full eff', abs=True,
            output_values=True)` to first obtain this dictionary.
        genes: `list` (default: `None`)
            The list of input genes, from which the network will be constructed.
        n_top_genes: `int` (default: `100`)
            Number of top genes that will be selected from to build the network.

    Returns
    -------
        A dictionary of dataframe of interactions between input genes for each group of cells based on ranking
        information of Jacobian analysis. Each composite dataframe has `regulator`, `target` and `weight` three columns.
    """

    genes = np.unique(genes)
    if full_reg_rank is None:
        full_reg_rank = rank_jacobian_genes(adata, groups=cluster, mode='full reg', abs=True, output_values=True)
    if full_eff_rank is None:
        full_eff_rank = rank_jacobian_genes(adata, groups=cluster, mode='full eff', abs=True, output_values=True)

    edges_list = {}

    reg_groups, eff_groups = full_reg_rank.keys(), full_eff_rank.keys()
    if reg_groups != eff_groups:
        raise Exception(f"the regulators ranking and effector ranking dataframe must have the same keys.")

    if cluster_names is not None:
        reg_groups = list(set(reg_groups).intersection(cluster_names))
        if len(reg_groups) == 0:
            raise ValueError(f"the clusters argument {cluster_names} provided doesn't match up with any clusters from the "
                             f"adata.")

    for c in reg_groups:
        if genes is None:
            reg_valid_genes, eff_valid_genes = full_reg_rank[c].columns.values, \
                                               full_eff_rank[c].columns.values
            reg_valid_genes = reg_valid_genes[np.arange(0, len(reg_valid_genes), 2)].tolist()
            eff_valid_genes = eff_valid_genes[np.arange(0, len(eff_valid_genes), 2)].tolist()
        else:
            reg_valid_genes, eff_valid_genes = full_reg_rank[c].columns.intersection(genes), \
                                               full_eff_rank[c].columns.intersection(genes)
        if len(reg_valid_genes) > 0:
            reg_df = get_interaction_in_cluster(full_reg_rank, c, reg_valid_genes, n_top_genes=n_top_genes,
                                                rank_regulators=True)
        if len(eff_valid_genes) > 0:
            eff_df = get_interaction_in_cluster(full_eff_rank, c, eff_valid_genes, n_top_genes=n_top_genes,
                                                rank_regulators=False)

        if len(reg_valid_genes) > 0 and len(eff_valid_genes) == 0:
            edges_list[c] = reg_df
        elif len(reg_valid_genes) == 0 and len(eff_valid_genes) > 0:
            edges_list[c] = eff_df
        elif len(reg_valid_genes) > 0 and len(eff_valid_genes) > 0:
            edges_list[c] = pd.concat((reg_df, eff_df), axis=0)

    return edges_list
