import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix

from ..preprocessing.preprocess import topTable
from ..preprocessing.utils import get_layer_keys
from .utils import save_fig
from ..tools.utils import update_dict, get_mapper
from ..preprocessing.utils import detect_datatype

def basic_stats(adata,
                  group=None,
                  figsize=(4, 3),
                  save_show_or_return='show',
                  save_kwargs={},):
    """Plot the basic statics (nGenes, nCounts and pMito) of each category of adata.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    group: `string` (default: None)
        Which group to facets the data into subplots. Default is None, or no faceting will be used.
    figsize: `string` (default: (4, 3))
        Figure size of each facet.
    save_show_or_return: {'show', 'save', 'return'} (default: `show`)
        Whether to save, show or return the figure.
    save_kwargs: `dict` (default: `{}`)
        A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
        will use the {"path": None, "prefix": 'basic_stats', "dpi": None, "ext": 'pdf', "transparent": True, "close":
        True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
        according to your needs.

    Returns
    -------
        A violin plot that shows the fraction of each category, produced by seaborn.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    if len(adata.obs.columns.intersection(['nGenes', 'nCounts', 'pMito'])) != 3:
        from ..preprocessing.utils import basic_stats
        basic_stats(adata)

    df = pd.DataFrame(
            {"nGenes": adata.obs['nGenes'], "nCounts": adata.obs['nCounts'],
             "pMito": adata.obs['pMito']}, index=adata.obs.index,
    )

    if group is not None and group in adata.obs.columns:
        df["group"] = adata.obs.loc[:, group]
        res = (
            df.melt(
                value_vars=["nGenes", "nCounts", "pMito"], id_vars=["group"]
            )
        )
    else:
        res = (
            df.melt(value_vars=["nGenes", "nCounts", "pMito"])
        )

    # https://wckdouglas.github.io/2016/12/seaborn_annoying_title
    g = sns.FacetGrid(res, col="variable", sharex=False, sharey=False, margin_titles=True, hue="variable",
                      height=figsize[1], aspect=figsize[0]/figsize[1])

    if group is None:
        g.map_dataframe(sns.violinplot, x="variable", y="value")
        g.set_xticklabels([])
        g.set(xticks=[])
    else:
        if res['group'].dtype.name == 'category':
            xticks = res['group'].cat.categories
        else:
            xticks = np.sort(res['group'].unique())
        kws = dict(order=xticks)

        g.map_dataframe(sns.violinplot, x="group", y="value", **kws)
        g.set_xticklabels(rotation=-30)

    [plt.setp(ax.texts, text="") for ax in g.axes.flat]  # remove the original texts
    # important to add this before setting titles
    g.set_titles(row_template='{row_name}', col_template='{col_name}')

    g.set_xlabels("")
    g.set_ylabels("")
    g.set(ylim=(0, None))

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'basic_stats', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return g


def show_fraction(adata,
                  genes=None,
                  group=None,
                  figsize=(4, 3),
                  save_show_or_return='show',
                  save_kwargs={},):
    """Plot the fraction of each category of data used in the velocity estimation.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    genes: `list` like:
        The list of gene names from which the fraction will be calculated.
    group: `string` (default: None)
        Which group to facets the data into subplots. Default is None, or no faceting will be used.
    figsize: `string` (default: (4, 3))
        Figure size of each facet.
    save_show_or_return: {'show', 'save', 'return'} (default: `show`)
        Whether to save, show or return the figure.
    save_kwargs: `dict` (default: `{}`)
        A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
        will use the {"path": None, "prefix": 'show_fraction', "dpi": None, "ext": 'pdf', "transparent": True, "close":
        True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
        according to your needs.

    Returns
    -------
        A violin plot that shows the fraction of each category, produced by seaborn.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    if genes is not None:
        genes = list(adata.var_names.intersection(genes))

        if len(genes) == 0:
            raise Exception("The gene list you provided doesn't much any genes from the adata object.")

    mode = None
    if pd.Series(["spliced", "unspliced"]).isin(adata.layers.keys()).all():
        mode = "splicing"
    elif pd.Series(["new", "total"]).isin(adata.layers.keys()).all():
        mode = "labelling"
    elif pd.Series(["uu", "ul", "su", "sl"]).isin(adata.layers.keys()).all():
        mode = "full"

    if not (mode in ["labelling", "splicing", "full"]):
        raise Exception(
            "your data doesn't seem to have either splicing or labeling or both information"
        )

    if mode == "labelling":
        new_mat, total_mat = (adata.layers["new"], adata.layers["total"]) if genes is None else \
            (adata[:, genes].layers["new"], adata[:, genes].layers["total"])

        new_cell_sum, tot_cell_sum = (
            (np.sum(new_mat, 1), np.sum(total_mat, 1))
            if not issparse(new_mat)
            else (new_mat.sum(1).A1, total_mat.sum(1).A1)
        )

        new_frac_cell = new_cell_sum / tot_cell_sum
        old_frac_cell = 1 - new_frac_cell
        df = pd.DataFrame(
            {"new_frac_cell": new_frac_cell, "old_frac_cell": old_frac_cell},
            index=adata.obs.index,
        )

        if group is not None and group in adata.obs.keys():
            df["group"] = adata.obs[group]
            res = df.melt(
                value_vars=["new_frac_cell", "old_frac_cell"], id_vars=["group"]
            )
        else:
            res = df.melt(value_vars=["new_frac_cell", "old_frac_cell"])

    elif mode == "splicing":
        if "ambiguous" in adata.layers.keys():
            ambiguous = adata.layers["ambiguous"] if genes is None else adata[:, genes].layers["ambiguous"]
        else:
            ambiguous = (
                csr_matrix(np.array([[0]]))
                if issparse(adata.layers["unspliced"])
                else np.array([[0]])
            )

        unspliced_mat, spliced_mat, ambiguous_mat = (
            adata.layers["unspliced"] if genes is None else adata[:, genes].layers["unspliced"],
            adata.layers["spliced"] if genes is None else adata[:, genes].layers["spliced"],
            ambiguous,
        )
        un_cell_sum, sp_cell_sum = (
            (np.sum(unspliced_mat, 1), np.sum(spliced_mat, 1))
            if not issparse(unspliced_mat)
            else (unspliced_mat.sum(1).A1, spliced_mat.sum(1).A1)
        )

        if "ambiguous" in adata.layers.keys():
            am_cell_sum = (
                ambiguous_mat.sum(1).A1
                if issparse(unspliced_mat)
                else np.sum(ambiguous_mat, 1)
            )
            tot_cell_sum = un_cell_sum + sp_cell_sum + am_cell_sum
            un_frac_cell, sp_frac_cell, am_frac_cell = (
                un_cell_sum / tot_cell_sum,
                sp_cell_sum / tot_cell_sum,
                am_cell_sum / tot_cell_sum,
            )
            df = pd.DataFrame(
                {
                    "unspliced": un_frac_cell,
                    "spliced": sp_frac_cell,
                    "ambiguous": am_frac_cell,
                },
                index=adata.obs.index,
            )
        else:
            tot_cell_sum = un_cell_sum + sp_cell_sum
            un_frac_cell, sp_frac_cell = (
                un_cell_sum / tot_cell_sum,
                sp_cell_sum / tot_cell_sum,
            )
            df = pd.DataFrame(
                {"unspliced": un_frac_cell, "spliced": sp_frac_cell},
                index=adata.obs.index,
            )

        if group is not None and group in adata.obs.columns:
            df["group"] = adata.obs.loc[:, group]
            res = (
                df.melt(
                    value_vars=["unspliced", "spliced", "ambiguous"], id_vars=["group"]
                )
                if "ambiguous" in adata.layers.keys()
                else df.melt(value_vars=["unspliced", "spliced"], id_vars=["group"])
            )
        else:
            res = (
                df.melt(value_vars=["unspliced", "spliced", "ambiguous"])
                if "ambiguous" in adata.layers.keys()
                else df.melt(value_vars=["unspliced", "spliced"])
            )

    elif mode == "full":
        uu, ul, su, sl = (
            adata.layers["uu"] if genes is None else adata[:, genes].layers["uu"],
            adata.layers["ul"] if genes is None else adata[:, genes].layers["ul"],
            adata.layers["su"] if genes is None else adata[:, genes].layers["su"],
            adata.layers["sl"] if genes is None else adata[:, genes].layers["sl"],
        )
        uu_sum, ul_sum, su_sum, sl_sum = (
            np.sum(uu, 1),
            np.sum(ul, 1),
            np.sum(su, 1),
            np.sum(sl, 1) if not issparse(uu) else uu.sum(1).A1,
            ul.sum(1).A1,
            su.sum(1).A1,
            sl.sum(1).A1,
        )

        tot_cell_sum = uu + ul + su + sl
        uu_frac, ul_frac, su_frac, sl_frac = (
            uu_sum / tot_cell_sum,
            ul_sum / tot_cell_sum,
            su / tot_cell_sum,
            sl / tot_cell_sum,
        )
        df = pd.DataFrame(
            {
                "uu_frac": uu_frac,
                "ul_frac": ul_frac,
                "su_frac": su_frac,
                "sl_frac": sl_frac,
            },
            index=adata.obs.index,
        )

        if group is not None and group in adata.obs.keys():
            df["group"] = adata.obs[group]
            res = df.melt(
                value_vars=["uu_frac", "ul_frac", "su_frac", "sl_frac"],
                id_vars=["group"],
            )
        else:
            res = df.melt(value_vars=["uu_frac", "ul_frac", "su_frac", "sl_frac"])

    g = sns.FacetGrid(res, col="variable", sharex=False, sharey=False, margin_titles=True, hue="variable",
                      height=figsize[1], aspect=figsize[0]/figsize[1])
    if group is None:
        g.map_dataframe(sns.violinplot, x="variable", y="value")
        g.set_xticklabels([])
        g.set(xticks=[])
    else:
        if res['group'].dtype.name == 'category':
            xticks = res['group'].cat.categories
        else:
            xticks = np.sort(res['group'].unique())
        kws = dict(order=xticks)

        g.map_dataframe(sns.violinplot, x="group", y="value", **kws)
        g.set_xticklabels(rotation=-30)

    [plt.setp(ax.texts, text="") for ax in g.axes.flat]  # remove the original texts
    # important to add this before setting titles
    g.set_titles(row_template='{row_name}', col_template='{col_name}')

    g.set_xlabels("")
    g.set_ylabels("Fraction")
    g.set(ylim=(0, None))

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'show_fraction', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return g


def variance_explained(adata,
                       threshold=0.002,
                       n_pcs=None,
                       figsize=(4, 3),
                       save_show_or_return='show',
                       save_kwargs={},
                       ):
    """Plot the accumulative variance explained by the principal components.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
        threshold: `float` (default: `0.002`)
            The threshold for the second derivative of the cumulative sum of the variance for each principal component.
            This threshold is used to determine the number of principal component used for downstream non-linear dimension
            reduction.
        n_pcs: `int` (default: `None`)
            Number of principal components.
        figsize: `string` (default: (4, 3))
            Figure size of each facet.
        save_show_or_return: {'show', 'save', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
            will use the {"path": None, "prefix": 'variance_explained', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.

    Returns
    -------
        Nothing but make a matplotlib based plot for showing the cumulative variance explained by each PC.
    """

    import matplotlib.pyplot as plt

    var_ = adata.uns["explained_variance_ratio_"]
    _, ax = plt.subplots(figsize=figsize)
    ax.plot(var_, c="r")
    tmp = np.diff(np.diff(np.cumsum(var_)) > threshold)
    n_comps = n_pcs if n_pcs is not None else np.where(tmp)[0][0] if np.any(tmp) else 20
    ax.axvline(n_comps, c="r")
    ax.set_xlabel("PCs")
    ax.set_ylabel("Variance explained")
    ax.set_xticks(list(ax.get_xticks()) + [n_comps])
    ax.set_xlim(0, len(var_))

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'variance_explained', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return ax


def feature_genes(adata,
                  layer="X",
                  mode=None,
                  figsize=(4, 3),
                  save_show_or_return='show',
                  save_kwargs={},
):
    """Plot selected feature genes on top of the mean vs. dispersion scatterplot.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        layer: `str` (default: `X`)
            The data from a particular layer (include X) used for making the feature gene plot.
        mode: None or `str` (default: `None`)
            The method to select the feature genes (can be either `dispersion`, `gini` or `SVR`).
        figsize: `string` (default: (4, 3))
            Figure size of each facet.
        save_show_or_return: {'show', 'save', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
            will use the {"path": None, "prefix": 'feature_genes', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.

    Returns
    -------
        Nothing but plots the selected feature genes via the mean, CV plot.
    """

    import matplotlib.pyplot as plt

    mode = adata.uns["feature_selection"] if mode is None else mode

    layer = get_layer_keys(adata, layer, include_protein=False)[0]

    if mode == "dispersion":
        key = "dispFitInfo" if layer in ["raw", "X"] else layer + "_dispFitInfo"

        table = topTable(adata, layer)
        x_min, x_max = (
            np.nanmin(table["mean_expression"]),
            np.nanmax(table["mean_expression"]),
        )
    elif mode == "SVR":
        prefix = "" if layer == "X" else layer + "_"
        key = (
            "velocyto_SVR"
            if layer == "raw" or layer == "X"
            else layer + "_velocyto_SVR"
        )

        if not np.all(
            pd.Series([prefix + "log_m", prefix + "score"]).isin(adata.var.columns)
        ):
            raise Exception(
                "Looks like you have not run support vector machine regression yet, try run SVRs first."
            )
        else:
            table = adata.var.loc[:
                , [prefix + "log_m", prefix + "log_cv", prefix + "score"]
            ]
            table = table.loc[
                np.isfinite(table[prefix + "log_m"])
                & np.isfinite(table[prefix + "log_cv"]),
                :,
            ]
            x_min, x_max = (
                np.nanmin(table[prefix + "log_m"]),
                np.nanmax(table[prefix + "log_m"]),
            )

    ordering_genes = (
        adata.var["use_for_pca"] if "use_for_pca" in adata.var.columns else None
    )

    mu_linspace = np.linspace(x_min, x_max, num=1000)
    fit = (
        adata.uns[key]["disp_func"](mu_linspace)
        if mode == "dispersion"
        else adata.uns[key]["SVR"](mu_linspace.reshape(-1, 1))
    )

    plt.figure(figsize=figsize)
    plt.plot(mu_linspace, fit, alpha=0.4, color="r")
    valid_ind = (
        table.index.isin(ordering_genes.index[ordering_genes])
        if ordering_genes is not None
        else np.ones(table.shape[0], dtype=bool)
    )

    valid_disp_table = table.iloc[valid_ind, :]
    if mode == "dispersion":
        ax = plt.scatter(
            valid_disp_table["mean_expression"],
            valid_disp_table["dispersion_empirical"],
            s=3,
            alpha=1,
            color="xkcd:red",
        )
    elif mode == "SVR":
        ax = plt.scatter(
            valid_disp_table[prefix + "log_m"],
            valid_disp_table[prefix + "log_cv"],
            s=3,
            alpha=1,
            color="xkcd:red",
        )

    neg_disp_table = table.iloc[~valid_ind, :]

    if mode == "dispersion":
        ax = plt.scatter(
            neg_disp_table["mean_expression"],
            neg_disp_table["dispersion_empirical"],
            s=3,
            alpha=0.5,
            color="xkcd:grey",
        )
    elif mode == "SVR":
        ax = plt.scatter(
            neg_disp_table[prefix + "log_m"],
            neg_disp_table[prefix + "log_cv"],
            s=3,
            alpha=0.5,
            color="xkcd:grey",
        )

    # plt.xlim((0, 100))
    if mode == "dispersion":
        plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mean (log)")
    plt.ylabel("Dispersion (log)") if mode == "dispersion" else plt.ylabel("CV (log)")

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'feature_genes', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return ax


def exp_by_groups(adata,
                    genes,
                    layer=None,
                    group=None,
                    use_ratio=False,
                    use_smoothed=True,
                    log=True,
                    angle=0,
                    figsize=(4, 3),
                    save_show_or_return='show',
                    save_kwargs={},
                  ):
    """Plot the (labeled) expression values of genes across different groups (time points).

    This function can be used as a sanity check about the labeled species to see whether they increase or decrease across
    time for a kinetic or degradation experiment, etc.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        genes: `list`
            The list of genes that you want to plot the gene expression.
        group: `string` (default: None)
            Which group information to plot aganist (as elements on x-axis). Default is None, or no groups will be used.
            Normally you should supply the column that indicates the time related to the labeling experiment. For example,
            it can be either the labeling time for a kinetic experiment or the chase time for a degradation experiment.
        use_ratio: `bool` (default: False)
            Whether to plot the fraction of expression (for example NTR, new to total ratio) over groups.
        use_smoothed: `bool` (default: 'True')
            Whether to use the smoothed data as gene expression.
        log: `bool` (default: `True`)
            Whether to log1p transform the expression data.
        figsize: `string` (default: (4, 3))
            Figure size of each facet.
        save_show_or_return: {'show', 'save', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
            will use the {"path": None, "prefix": 'exp_by_groups', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.

    Returns
    -------
        A violin plot that shows each gene's expression (row) across different groups (time), produced by seaborn.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    valid_genes = adata.var_names.intersection(genes)
    if len(valid_genes) == 0:
        raise ValueError(f"The adata object doesn't include any gene from the list you provided!")
    if group is not None and group not in adata.obs.keys():
        raise ValueError(f"The group {group} is not existed in your adata object!")

    has_splicing, has_labeling, has_protein = detect_datatype(adata)
    if (has_splicing + has_labeling) == 0:
        layer = 'X' if layer is None else layer
    elif has_splicing and not has_labeling:
        layer = 'X_spliced' if layer is None else layer
    elif not has_splicing and has_labeling:
        layer = 'X_new' if layer is None else layer
    elif has_splicing and has_labeling:
        layer = 'X_new' if layer is None else layer

    if use_smoothed:
        mapper = get_mapper()
        layer = mapper[layer]

    if layer != 'X' and layer not in adata.layers.keys():
        raise ValueError(f"The layer {layer} is not existed in your adata object!")

    exprs = adata[:, valid_genes].X if layer == 'X' else adata[:, valid_genes].layers[layer]
    exprs = exprs.A if issparse(exprs) else exprs
    if use_ratio:
        has_splicing, has_labeling, has_protein = detect_datatype(adata)
        if has_labeling:
            if layer.startswith('X_') or layer.startswith('M_'):
                tot = adata[:, valid_genes].layers[mapper['X_total']] if use_smoothed \
                    else adata[:, valid_genes].layers['X_total']
                tot = tot.A if issparse(tot) else tot
                exprs = exprs / tot
            else:
                exprs = exprs
        else:
            if layer.startswith('X_') or layer.startswith('M_'):
                tot = adata[:, valid_genes].layers[mapper['X_unspliced']] + \
                        adata[:, valid_genes].layers[mapper['X_spliced']] if use_smoothed \
                    else adata[:, valid_genes].layers['X_unspliced'] + \
                         adata[:, valid_genes].layers['X_spliced']
                tot = tot.A if issparse(tot) else tot
                exprs = exprs / tot
            else:
                exprs = exprs

    df = pd.DataFrame(np.log1p(exprs), index=adata.obs_names, columns=valid_genes) if log else \
        pd.DataFrame(np.log1p(exprs), index=adata.obs_names, columns=valid_genes)

    if group is not None and group in adata.obs.columns:
        df["group"] = adata.obs[group]
        res = (
            df.melt(id_vars=["group"])
        )
    else:
        df["group"] =1
        res = df.melt(id_vars=["group"])

    if res['group'].dtype.name == 'category':
        xticks = res['group'].cat.categories
    else:
        xticks = np.sort(res['group'].unique())
    kws = dict(order=xticks)

    # https://wckdouglas.github.io/2016/12/seaborn_annoying_title
    g = sns.FacetGrid(res, row="variable", sharex=False, sharey=False, margin_titles=True, hue="variable",
                      height=figsize[1], aspect=figsize[0]/figsize[1])
    g.map_dataframe(sns.violinplot, x="group", y="value",  **kws)
    g.map_dataframe(sns.pointplot, x="group", y="value", color='k',  **kws)
    if group is None:
        g.set_xticklabels([])
        g.set(xticks=[])
    else:
        g.set_xticklabels(rotation=angle)

    [plt.setp(ax.texts, text="") for ax in g.axes.flat]  # remove the original texts
    # important to add this before setting titles
    g.set_titles(row_template='{row_name}', col_template='{col_name}')

    if log:
        g.set_ylabels("log(Expression + 1)")
    else:
        g.set_ylabels("Expression")

    g.set_xlabels("")
    g.set(ylim=(0, None))

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'exp_by_groups', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return g
