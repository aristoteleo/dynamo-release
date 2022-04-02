from typing import Optional, Sequence, Union

import matplotlib
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes
from scipy.sparse import csr_matrix, issparse

from ..configuration import DynamoAdataKeyManager
from ..dynamo_logger import main_warning
from ..preprocessing import preprocess as pp
from ..preprocessing.preprocess_monocle_utils import top_table
from ..preprocessing.utils import detect_experiment_datatype
from ..tools.utils import get_mapper, update_dict
from .utils import save_fig


def basic_stats(
    adata: AnnData,
    group: Optional[str] = None,
    figsize: tuple = (4, 3),
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
):
    """Plot the basic statics (nGenes, nCounts and pMito) of each category of adata.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    group: `string` (default: None)
        Which group to facets the data into subplots. Default is None, or no faceting will be used.
    figsize:
        Figure size of each facet.
    save_show_or_return: {'show', 'save', 'return'} (default: `show`)
        Whether to save, show or return the figure.
    save_kwargs: `dict` (default: `{}`)
        A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig
        function will use the {"path": None, "prefix": 'basic_stats', "dpi": None, "ext": 'pdf', "transparent": True,
        "close": True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify
        those keys according to your needs.

    Returns
    -------
        A violin plot that shows the fraction of each category, produced by seaborn.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    if len(adata.obs.columns.intersection(["nGenes", "nCounts", "pMito"])) != 3:
        from ..preprocessing.utils import basic_stats

        basic_stats(adata)

    df = pd.DataFrame(
        {
            "nGenes": adata.obs["nGenes"],
            "nCounts": adata.obs["nCounts"],
            "pMito": adata.obs["pMito"],
        },
        index=adata.obs.index,
    )

    if group is not None and group in adata.obs.columns:
        df["group"] = adata.obs.loc[:, group]
        res = df.melt(value_vars=["nGenes", "nCounts", "pMito"], id_vars=["group"])
    else:
        res = df.melt(value_vars=["nGenes", "nCounts", "pMito"])

    # https://wckdouglas.github.io/2016/12/seaborn_annoying_title
    g = sns.FacetGrid(
        res,
        col="variable",
        sharex=False,
        sharey=False,
        margin_titles=True,
        hue="variable",
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
    )

    if group is None:
        g.map_dataframe(sns.violinplot, x="variable", y="value")
        g.set_xticklabels([])
        g.set(xticks=[])
    else:
        if res["group"].dtype.name == "category":
            xticks = res["group"].cat.categories
        else:
            xticks = np.sort(res["group"].unique())
        kws = dict(order=xticks)

        g.map_dataframe(sns.violinplot, x="group", y="value", **kws)
        g.set_xticklabels(rotation=-30)

    [plt.setp(ax.texts, text="") for ax in g.axes.flat]  # remove the original texts
    # important to add this before setting titles
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    g.set_xlabels("")
    g.set_ylabels("")
    g.set(ylim=(0, None))

    if save_show_or_return == "save":
        s_kwargs = {
            "path": None,
            "prefix": "basic_stats",
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)
        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return g


def show_fraction(
    adata: AnnData,
    genes: Optional[list] = None,
    group: Optional[str] = None,
    figsize: tuple = (4, 3),
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
):
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
        A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig
        function will use the {"path": None, "prefix": 'show_fraction', "dpi": None, "ext": 'pdf', "transparent": True,
        "close": True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify
        those keys according to your needs.

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
        raise Exception("your data doesn't seem to have either splicing or labeling or both information")

    if mode == "labelling":
        new_mat, total_mat = (
            (adata.layers["new"], adata.layers["total"])
            if genes is None
            else (
                adata[:, genes].layers["new"],
                adata[:, genes].layers["total"],
            )
        )

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
            res = df.melt(value_vars=["new_frac_cell", "old_frac_cell"], id_vars=["group"])
        else:
            res = df.melt(value_vars=["new_frac_cell", "old_frac_cell"])

    elif mode == "splicing":
        if "ambiguous" in adata.layers.keys():
            ambiguous = adata.layers["ambiguous"] if genes is None else adata[:, genes].layers["ambiguous"]
        else:
            ambiguous = csr_matrix(np.array([[0]])) if issparse(adata.layers["unspliced"]) else np.array([[0]])

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
            am_cell_sum = ambiguous_mat.sum(1).A1 if issparse(unspliced_mat) else np.sum(ambiguous_mat, 1)
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
                    value_vars=["unspliced", "spliced", "ambiguous"],
                    id_vars=["group"],
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
            (np.sum(uu, 1), np.sum(ul, 1), np.sum(su, 1), np.sum(sl, 1))
            if not issparse(uu)
            else (
                uu.sum(1).A1,
                ul.sum(1).A1,
                su.sum(1).A1,
                sl.sum(1).A1,
            )
        )

        tot_cell_sum = uu_sum + ul_sum + su_sum + sl_sum
        uu_frac, ul_frac, su_frac, sl_frac = (
            uu_sum / tot_cell_sum,
            ul_sum / tot_cell_sum,
            su_sum / tot_cell_sum,
            sl_sum / tot_cell_sum,
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

    g = sns.FacetGrid(
        res,
        col="variable",
        sharex=False,
        sharey=False,
        margin_titles=True,
        hue="variable",
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
    )
    if group is None:
        g.map_dataframe(sns.violinplot, x="variable", y="value")
        g.set_xticklabels([])
        g.set(xticks=[])
    else:
        if res["group"].dtype.name == "category":
            xticks = res["group"].cat.categories
        else:
            xticks = np.sort(res["group"].unique())
        kws = dict(order=xticks)

        g.map_dataframe(sns.violinplot, x="group", y="value", **kws)
        g.set_xticklabels(rotation=-30)

    [plt.setp(ax.texts, text="") for ax in g.axes.flat]  # remove the original texts
    # important to add this before setting titles
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    g.set_xlabels("")
    g.set_ylabels("Fraction")
    g.set(ylim=(0, None))

    if save_show_or_return == "save":
        s_kwargs = {
            "path": None,
            "prefix": "show_fraction",
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return g


def variance_explained(
    adata: AnnData,
    threshold: float = 0.002,
    n_pcs: Optional[int] = None,
    figsize: tuple = (4, 3),
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
):
    """Plot the accumulative variance explained by the principal components.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
        threshold: `float` (default: `0.002`)
            The threshold for the second derivative of the cumulative sum of the variance for each principal component.
            This threshold is used to determine the number of principal component used for downstream non-linear
            dimension reduction.
        n_pcs: `int` (default: `None`)
            Number of principal components.
        figsize: `string` (default: (4, 3))
            Figure size of each facet.
        save_show_or_return: {'show', 'save', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
            save_fig function will use the {"path": None, "prefix": 'variance_explained', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a
            dictionary that properly modify those keys according to your needs.

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
        s_kwargs = {
            "path": None,
            "prefix": "variance_explained",
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return ax


def biplot(
    adata: AnnData,
    pca_components: Sequence[int] = [0, 1],
    pca_key: str = "X_pca",
    loading_key: str = "PCs",
    figsize: tuple = (6, 4),
    scale_pca_embedding: bool = False,
    draw_pca_embedding: bool = False,
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
    ax: Union[matplotlib.axes._subplots.SubplotBase, None] = None,
):
    """A biplot overlays a score plot and a loadings plot in a single graph. In such a plot, points are the projected
    observations; vectors are the projected variables. If the data are well-approximated by the first two principal
    components, a biplot enables you to visualize high-dimensional data by using a two-dimensional graph. See more at:
    https://blogs.sas.com/content/iml/2019/11/06/what-are-biplots.html

    In general, the score plot and the loadings plot will have different scales. Consequently, you need to rescale the
    vectors or observations (or both) when you overlay the score and loadings plots. There are four common choices of
    scaling. Each scaling emphasizes certain geometric relationships between pairs of observations (such as distances),
    between pairs of variables (such as angles), or between observations and variables. This article discusses the
    geometry behind two-dimensional biplots and shows how biplots enable you to understand relationships in multivariate
    data.

    Parameters
    ----------
        adata:
            An Annodata object that has pca and loading information prepared.
        pca_components:
            The pca components that will be used to draw the biplot.
        pca_key:
            A key to the pca embedding matrix, in `.obsm`.
        loading_key:
            A key to the pca loading matrix, in either `.uns` or `.obsm`.
        figsize:
            The figure size.
        scale_pca_embedding:
            Whether to scale the pca embedding.
        draw_pca_embedding:
            Whether to draw the pca embedding.
        save_show_or_return: {'show', 'save', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
            save_fig function will use the {"path": None, "prefix": 'biplot', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a
            dictionary that properly modify those keys according to your needs.
        ax
            An ax where the biplot will be appended to.

    Returns
    -------
        If save_show_or_return is not `return`, return nothing but plot or save the biplot; otherwise return an axes
        with the biplot in it.
    """

    import matplotlib.pyplot as plt

    if loading_key in adata.uns.keys():
        PCs = adata.uns[loading_key]
    elif loading_key in adata.varm.keys():
        PCs = adata.varm[loading_key]
    else:
        raise Exception(f"No PC matrix {loading_key} found in neither .uns nor .varm.")

    # rotation matrix
    xvector = PCs[:, pca_components[0]]
    yvector = PCs[:, pca_components[1]]

    # pca components
    xs = adata.obsm[pca_key][:, pca_components[0]]
    ys = adata.obsm[pca_key][:, pca_components[1]]

    # scale pca component
    if scale_pca_embedding:
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
    else:
        scalex, scaley = 1, 1

    genes = adata.var_names[adata.var.use_for_pca]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i in range(len(xvector)):
        # arrows project features, e.g. genes, as vectors onto PC axes
        ax.arrow(0, 0, xvector[i] * max(xs), yvector[i] * max(ys), color="r", width=0.0005, head_width=0.0025)
        ax.text(xvector[i] * max(xs) * 1.01, yvector[i] * max(ys) * 1.01, genes[i], color="r")

    ax.set_xlabel("PC" + str(pca_components[0]))
    ax.set_ylabel("PC" + str(pca_components[1]))
    if draw_pca_embedding:
        for i in range(len(xs)):
            # circles project cells
            ax.plot(xs[i] * scalex, ys[i] * scaley, "b", alpha=0.1)
            ax.text(xs[i] * scalex * 1.01, ys[i] * scaley * 1.01, list(adata.obs.cluster)[i], color="b", alpha=0.1)

    if save_show_or_return == "save":
        s_kwargs = {
            "path": None,
            "prefix": "biplot",
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    else:
        return ax


def loading(
    adata: AnnData,
    n_pcs: int = 10,
    loading_key: str = "PCs",
    n_top_genes: int = 10,
    ncol: int = 5,
    figsize: tuple = (6, 4),
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
):
    """Plot the top absolute pca loading genes.

    Red text are positive loading genes while black negative loading genes.

    Parameters
    ----------
        adata:
            An Annodata object that has pca and loading information prepared.
        n_pcs:
            Number of pca.
        loading_key:
            A key to the pca loading matrix, in either `.uns` or `.obsm`.
        n_top_genes:
            Number of top genes with highest absolute loading score.
        ncol:
            Number of panels on the resultant figure.
        figsize:
            Figure size.
        save_show_or_return: {'show', 'save', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
            save_fig function will use the {"path": None, "prefix": 'biplot', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a
            dictionary that properly modify those keys according to your needs.

    Returns
    -------
        If save_show_or_return is not `return`, return nothing but plot or save the biplot; otherwise return an axes
        with the loading plot in it.
    """

    import matplotlib.pyplot as plt

    if loading_key in adata.uns.keys():
        PCs = adata.uns[loading_key]
    elif loading_key in adata.varm.keys():
        PCs = adata.varm[loading_key]
    else:
        raise Exception(f"No PC matrix {loading_key} found in neither .uns nor .varm.")

    if n_pcs is None:
        n_pcs = PCs.shape[1]

    x = np.arange(n_top_genes)
    genes = adata.var_names[adata.var.use_for_pca]

    nrow, ncol = int(n_pcs / ncol), min([ncol, n_pcs])
    fig, axes = plt.subplots(nrow, ncol, figsize=(figsize[0] * ncol, figsize[1] * nrow))

    for i in np.arange(n_pcs):
        cur_row, cur_col = int(i / ncol), i % ncol

        cur_pc = PCs[:, i]
        cur_sign = np.sign(cur_pc)
        cur_pc = np.abs(cur_pc)
        sort_ind, sort_val = np.argsort(cur_pc)[::-1], np.sort(cur_pc)[::-1]
        axes[cur_row, cur_col].scatter(x, sort_val[: len(x)])
        for j in x:
            axes[cur_row, cur_col].text(
                x[j], sort_val[j] * 1.01, genes[sort_ind[j]], color="r" if cur_sign[sort_ind[j]] > 0 else "k"
            )

        axes[cur_row, cur_col].set_title("PC " + str(i))

    if save_show_or_return == "save":
        s_kwargs = {
            "path": None,
            "prefix": "loading",
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    else:
        return axes


def feature_genes(
    adata: AnnData,
    layer: str = "X",
    mode: Union[None, str] = None,
    figsize: tuple = (4, 3),
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
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
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
            save_fig function will use the {"path": None, "prefix": 'feature_genes', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a
            dictionary that properly modify those keys according to your needs.

    Returns
    -------
        Nothing but plots the selected feature genes via the mean, CV plot.
    """

    import matplotlib.pyplot as plt

    mode = adata.uns["feature_selection"] if mode is None else mode

    layer = DynamoAdataKeyManager.get_available_layer_keys(adata, layer, include_protein=False)[0]

    uns_store_key = None
    if mode == "dispersion":
        uns_store_key = "dispFitInfo" if layer in ["raw", "X"] else layer + "_dispFitInfo"

        table = top_table(adata, layer)
        x_min, x_max = (
            np.nanmin(table["mean_expression"]),
            np.nanmax(table["mean_expression"]),
        )
    elif mode == "SVR":
        prefix = "" if layer == "X" else layer + "_"
        uns_store_key = "velocyto_SVR" if layer == "raw" or layer == "X" else layer + "_velocyto_SVR"

        if not np.all(pd.Series([prefix + "log_m", prefix + "score"]).isin(adata.var.columns)):
            raise Exception("Looks like you have not run support vector machine regression yet, try run SVRs first.")
        else:
            table = adata.var.loc[:, [prefix + "log_m", prefix + "log_cv", prefix + "score"]]
            table = table.loc[
                np.isfinite(table[prefix + "log_m"]) & np.isfinite(table[prefix + "log_cv"]),
                :,
            ]
            x_min, x_max = (
                np.nanmin(table[prefix + "log_m"]),
                np.nanmax(table[prefix + "log_m"]),
            )

    ordering_genes = adata.var["use_for_pca"] if "use_for_pca" in adata.var.columns else None

    mu_linspace = np.linspace(x_min, x_max, num=1000)
    fit = (
        adata.uns[uns_store_key]["disp_func"](mu_linspace)
        if mode == "dispersion"
        else adata.uns[uns_store_key]["SVR"](mu_linspace.reshape(-1, 1))
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
        s_kwargs = {
            "path": None,
            "prefix": "feature_genes",
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return ax


def exp_by_groups(
    adata: AnnData,
    genes: list,
    layer: Optional[str] = None,
    group: Optional[str] = None,
    use_ratio: bool = False,
    use_smoothed: bool = True,
    log: bool = True,
    angle: int = 0,
    re_order: bool = True,
    figsize: tuple = (4, 3),
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
):
    """Plot the (labeled) expression values of genes across different groups (time points).

    This function can be used as a sanity check about the labeled species to see whether they increase or decrease
    across time for a kinetic or degradation experiment, etc.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        genes: `list`
            The list of genes that you want to plot the gene expression.
        group: `string` (default: None)
            Which group information to plot aganist (as elements on x-axis). Default is None, or no groups will be used.
            Normally you should supply the column that indicates the time related to the labeling experiment. For
            example, it can be either the labeling time for a kinetic experiment or the chase time for a degradation
            experiment.
        use_ratio: `bool` (default: False)
            Whether to plot the fraction of expression (for example NTR, new to total ratio) over groups.
        use_smoothed: `bool` (default: 'True')
            Whether to use the smoothed data as gene expression.
        log: `bool` (default: `True`)
            Whether to log1p transform the expression data.
        angle: `float` (default: `0`)
            The angle to rotate the xtick labels for the purpose of avoiding overlapping between text.
        re_order: `bool` (default: `True`)
            Whether to reorder categories before drawing groups on the x-axis.
        figsize: `string` (default: (4, 3))
            Figure size of each facet.
        save_show_or_return: {'show', 'save', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
            save_fig function will use the {"path": None, "prefix": 'exp_by_groups', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a
            dictionary that properly modify those keys according to your needs.

    Returns
    -------
        A violin plot that shows each gene's expression (row) across different groups (time), produced by seaborn.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    valid_genes = adata.var_names.intersection(genes)
    if len(valid_genes) == 0:
        raise ValueError("The adata object doesn't include any gene from the list you provided!")
    if group is not None and group not in adata.obs.keys():
        raise ValueError(f"The group {group} is not existed in your adata object!")

    (
        has_splicing,
        has_labeling,
        splicing_labeling,
        has_protein,
    ) = detect_experiment_datatype(adata)
    if (has_splicing + has_labeling) == 0:
        layer = "X" if layer is None else layer
    elif has_splicing and not has_labeling:
        layer = "X_spliced" if layer is None else layer
    elif not has_splicing and has_labeling:
        layer = "X_new" if layer is None else layer
    elif has_splicing and has_labeling:
        layer = "X_new" if layer is None else layer

    if use_smoothed:
        mapper = get_mapper()
        layer = mapper[layer]

    if layer != "X" and layer not in adata.layers.keys():
        raise ValueError(f"The layer {layer} is not existed in your adata object!")

    exprs = adata[:, valid_genes].X if layer == "X" else adata[:, valid_genes].layers[layer]
    exprs = exprs.A if issparse(exprs) else exprs
    if use_ratio:
        (
            has_splicing,
            has_labeling,
            splicing_labeling,
            has_protein,
        ) = detect_experiment_datatype(adata)
        if has_labeling:
            if layer.startswith("X_") or layer.startswith("M_"):
                tot = (
                    adata[:, valid_genes].layers[mapper["X_total"]]
                    if use_smoothed
                    else adata[:, valid_genes].layers["X_total"]
                )
                tot = tot.A if issparse(tot) else tot
                exprs = exprs / tot
            else:
                exprs = exprs
        else:
            if layer.startswith("X_") or layer.startswith("M_"):
                tot = (
                    adata[:, valid_genes].layers[mapper["X_unspliced"]]
                    + adata[:, valid_genes].layers[mapper["X_spliced"]]
                    if use_smoothed
                    else adata[:, valid_genes].layers["X_unspliced"] + adata[:, valid_genes].layers["X_spliced"]
                )
                tot = tot.A if issparse(tot) else tot
                exprs = exprs / tot
            else:
                exprs = exprs

    df = (
        pd.DataFrame(np.log1p(exprs), index=adata.obs_names, columns=valid_genes)
        if log
        else pd.DataFrame(np.log1p(exprs), index=adata.obs_names, columns=valid_genes)
    )

    if group is not None and group in adata.obs.columns:
        df["group"] = adata.obs[group]
        res = df.melt(id_vars=["group"])
    else:
        df["group"] = 1
        res = df.melt(id_vars=["group"])

    if res["group"].dtype.name == "category":
        xticks = res["group"].cat.categories.sort_values() if re_order else res["group"].cat.categories
    else:
        xticks = np.sort(res["group"].unique())

    kws = dict(order=xticks)

    # https://wckdouglas.github.io/2016/12/seaborn_annoying_title
    g = sns.FacetGrid(
        res,
        row="variable",
        sharex=False,
        sharey=False,
        margin_titles=True,
        hue="variable",
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
    )
    g.map_dataframe(sns.violinplot, x="group", y="value", **kws)
    g.map_dataframe(sns.pointplot, x="group", y="value", color="k", **kws)
    if group is None:
        g.set_xticklabels([])
        g.set(xticks=[])
    else:
        g.set_xticklabels(rotation=angle)

    [plt.setp(ax.texts, text="") for ax in g.axes.flat]  # remove the original texts
    # important to add this before setting titles
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    if log:
        g.set_ylabels("log(Expression + 1)")
    else:
        g.set_ylabels("Expression")

    g.set_xlabels("")
    g.set(ylim=(0, None))

    if save_show_or_return == "save":
        s_kwargs = {
            "path": None,
            "prefix": "exp_by_groups",
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return g


def highest_frac_genes(
    adata: AnnData,
    n_top: int = 30,
    gene_prefix_list: list = None,
    show_individual_prefix_gene: bool = False,
    show: Optional[bool] = True,
    save_path: str = None,
    ax: Optional[Axes] = None,
    gene_annotations: Optional[list] = None,
    gene_annotation_key: str = "use_for_pca",
    log: bool = False,
    store_key: str = "highest_frac_genes",
    orient: str = "v",
    figsize: Union[list, None] = None,
    layer: Union[str, None] = None,
    title: Union[str, None] = None,
    v_rotation: float = 35,
    **kwargs,
):
    """[summary]

    Parameters
    ----------
    adata : AnnData
        [description]
    n_top : int, optional
        [description], by default 30
    gene_prefix_list : list, optional
        A list of gene name prefix, by default None
    show_individual_prefix_gene: bool, optional
        [description], by default False
    show : Optional[bool], optional
        [description], by default True
    save_path : str, optional
        [description], by default None
    ax : Optional[Axes], optional
        [description], by default None
    gene_annotations : Optional[list], optional
        Annotations for genes, or annotations for gene prefix subsets, by default None
    gene_annotation_key : str, optional
        gene annotations key in adata.var, by default "use_for_pca".
        This option is not available for gene_prefix_list and thus users should
        pass gene_annotations argument for the prefix list.
    log : bool, optional
        [description], by default False
    store_key : str, optional
        [description], by default "expr_percent"

    Returns
    -------
    [type]
        [description]
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        length = n_top * 0.4
        if figsize is None:
            if orient == "v":
                fig, ax = plt.subplots(figsize=(length, 5))
            else:
                fig, ax = plt.subplots(figsize=(7, length))
        else:
            fig, ax = plt.subplots(figsize=figsize)
    if log:
        ax.set_xscale("log")

    adata = pp.highest_frac_genes(
        adata,
        store_key=store_key,
        n_top=n_top,
        layer=layer,
        gene_prefix_list=gene_prefix_list,
        show_individual_prefix_gene=show_individual_prefix_gene,
    )
    if adata is None:
        # something wrong with user input or compute_top_genes_df
        return
    top_genes_df, selected_indices = (
        adata.uns[store_key]["top_genes_df"],
        adata.uns[store_key]["selected_indices"],
    )

    # TODO use top genes_df dataframe; however this logic currently
    # does not fit subset logics and may fail tests.

    # main_info("Using prexisting top_genes_df in .uns.")
    # top_genes_df = adata.uns["top_genes_df"]

    # draw plots
    sns.boxplot(
        data=top_genes_df,
        orient=orient,
        ax=ax,
        fliersize=1,
        showmeans=True,
        **kwargs,
    )

    if gene_annotations is None:
        if gene_annotation_key in adata.var:
            gene_annotations = adata.var[gene_annotation_key][selected_indices]

        else:
            main_warning(
                "%s not in adata.var, ignoring the gene annotation key when plotting",
                indent_level=2,
            )

    if orient == "v":
        ax.set_xticklabels(ax.get_xticklabels(), rotation=v_rotation, ha="right")
        ax.set_xlabel("genes")
        ax.set_ylabel("fractions of total counts")

        if gene_annotations is not None:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_ylim())
            ax2.set_xticks(ax.get_yticks())
            ax2.set_xticks(list(range(len(gene_annotations))))
            ax2.set_xticklabels(gene_annotations, rotation=v_rotation, ha="left")
            ax2.set_xlabel(gene_annotation_key)
    elif orient == "h":
        ax.set_xlabel("fractions of total counts")
        ax.set_ylabel("genes")
        if gene_annotations is not None:
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(ax.get_yticks())
            ax2.set_yticks(list(range(len(gene_annotations))))
            ax2.set_yticklabels(gene_annotations)
            ax2.set_ylabel(gene_annotation_key)
    else:
        raise NotImplementedError()

    if title is None:
        if layer is None:
            ax.set_title("Rank by gene expression fraction")
        else:
            ax.set_title("Rank by %s fraction" % layer)
    if show:
        plt.show()

    if save_path:
        s_kwargs = {
            "path": save_path,
            "prefix": "plot_highest_gene",
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        save_fig(**s_kwargs)

    return ax
    # if save_show_or_return == "save":
    #     s_kwargs = {
    #         "path": save_path,
    #         "prefix": "plot_highest_gene",
    #         "dpi": None,
    #         "ext": "pdf",
    #         "transparent": True,
    #         "close": True,
    #         "verbose": True,
    #     }
    #     s_kwargs.update(kwargs)
    #     save_fig(save_path, **s_kargs)
    # elif save_show_or_return == "show":
    #     plt.show()
    # else:
    #     return ax
