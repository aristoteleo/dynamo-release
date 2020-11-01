import numpy as np, pandas as pd
from .vector_calculus import rank_jacobian_genes

def get_group_interaction(rank_df,
                          group,
                          genes,
                          n_top_genes=100,
                          rank_regulators=False):
    subset_rank_df = rank_df[group].head(n_top_genes)
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


def build_cluster_graph(adata,
                        cluster,
                        cluster_names=None,
                        full_reg_rank=None,
                        full_eff_rank=None,
                        genes=None,
                        top_n=100):
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
            reg_df = get_group_interaction(full_reg_rank, c, reg_valid_genes, n_top_genes=top_n,
                                           rank_regulators=True)
        if len(eff_valid_genes) > 0:
            eff_df = get_group_interaction(full_eff_rank, c, eff_valid_genes, n_top_genes=top_n,
                                           rank_regulators=False)

        if len(reg_valid_genes) > 0 and len(eff_valid_genes) == 0:
            edges_list[c] = reg_df
        elif len(reg_valid_genes) == 0 and len(eff_valid_genes) > 0:
            edges_list[c] = eff_df
        elif len(reg_valid_genes) > 0 and len(eff_valid_genes) > 0:
            edges_list[c] = pd.concat((reg_df, eff_df), axis=0)

    return edges_list
