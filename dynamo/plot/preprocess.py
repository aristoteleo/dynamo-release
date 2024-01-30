from typing import Any, Dict, List, Optional, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from scipy.sparse import csr_matrix, issparse

from ..configuration import DynamoAdataKeyManager
from ..dynamo_logger import main_warning
from ..preprocessing import gene_selection
from ..preprocessing.gene_selection import get_prediction_by_svr
from ..preprocessing.utils import detect_experiment_datatype
from ..tools.utils import get_mapper
from .utils import save_fig, save_show_ret


def basic_stats(
    adata: AnnData,
    group: Optional[str] = None,
    figsize: Tuple[float, float] = (4, 3),
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
) -> Optional[sns.FacetGrid]:
    """Plot the basic statics (nGenes, nCounts and pMito) of each category of adata.

    Args:
        adata: an AnnData object.
        group: the column key of `adata.obs` to facet the data into subplots. If None, no faceting will be used.
            Defaults to None.
        figsize: the size of each panel in the figure. Defaults to (4, 3).
        save_show_or_return: whether to save, show, or return the plots. Could be one of 'save', 'show', or 'return'.
            Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'basic_stats', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to 'return', the generated figure
        (seaborn.FacetGrid) would be returned.
    """

    if len(adata.obs.columns.intersection(["nGenes", "nCounts", "pMito"])) != 3:
        from ..preprocessing.QC import basic_stats

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

    return save_show_ret("basic_stats", save_show_or_return, save_kwargs, g)


def show_fraction(
    adata: AnnData,
    genes: Optional[List[str]] = None,
    group: Optional[str] = None,
    figsize: Tuple[float, float] = (4, 3),
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
) -> Optional[sns.FacetGrid]:
    """Plot the fraction of each category of data used in the velocity estimation.

    Args:
        adata: an AnnData object.
        genes: the list of gene names from which the fraction will be calculated. Defaults to None.
        group: the column key of `adata.obs` to facet the data into subplots. If None, no faceting will be used.
            Defaults to None.
        figsize: the size of each panel in the figure. Defaults to (4, 3).
        save_show_or_return: whether to save, show, or return the plots. Could be one of 'save', 'show', or 'return'.
            Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'show_fraction', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.

    Raises:
        ValueError: `genes` does not contain any genes from the adata object.
        ValueError: `adata` does not have proper splicing or labeling data.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to 'return', the generated figure
        (seaborn.FacetGrid) would be returned.
    """

    if genes is not None:
        genes = list(adata.var_names.intersection(genes))

        if len(genes) == 0:
            raise ValueError("The gene list you provided doesn't much any genes from the adata object.")

    mode = None
    if pd.Series(["spliced", "unspliced"]).isin(adata.layers.keys()).all():
        mode = "splicing"
    elif pd.Series(["new", "total"]).isin(adata.layers.keys()).all():
        mode = "labelling"
    elif pd.Series(["uu", "ul", "su", "sl"]).isin(adata.layers.keys()).all():
        mode = "full"

    if not (mode in ["labelling", "splicing", "full"]):
        raise ValueError("your data doesn't seem to have either splicing or labeling or both information")

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

    return save_show_ret("show_fraction", save_show_or_return, save_kwargs, g)


def variance_explained(
    adata: AnnData,
    threshold: float = 0.002,
    n_pcs: Optional[int] = None,
    figsize: Tuple[float, float] = (4, 3),
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
) -> Axes:
    """Plot the accumulative variance explained by the principal components.

    Args:
        adata: an AnnDate object.
        threshold: the threshold for the second derivative of the cumulative sum of the variance for each principal
            component. This threshold is used to determine the number of principle components used for downstream non-
            linear dimension reduction. Defaults to 0.002.
        n_pcs: the number of principal components. If None, the number of components would be inferred automatically.
            Defaults to None.
        figsize: the size of each panel of the figure. Defaults to (4, 3).
        save_show_or_return: whether to save, show, or return the generated figure. Can be one of 'save', 'show', or
            'return'. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'variance_explained', "dpi": None,
            "ext": 'pdf', "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can
            provide a dictionary that properly modify those keys according to your needs. Defaults to {}.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the matplotlib Axes of the
        figure would be returned.
    """

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

    return save_show_ret("variance_explained", save_show_or_return, save_kwargs, ax)


def biplot(
    adata: AnnData,
    pca_components: Tuple[int, int] = [0, 1],
    pca_key: str = "X_pca",
    loading_key: str = "PCs",
    figsize: Tuple[float, float] = (6, 4),
    scale_pca_embedding: bool = False,
    draw_pca_embedding: bool = False,
    show_text: bool = False,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
    ax: Optional[Axes] = None,
) -> Axes:
    """A biplot overlays a score plot and a loadings plot in a single graph.

    In such a plot, points are the projected observations; vectors are the projected variables. If the data are well-
    approximated by the first two principal components, a biplot enables you to visualize high-dimensional data by using
    a two-dimensional graph. See more at: https://blogs.sas.com/content/iml/2019/11/06/what-are-biplots.html

    In general, the score plot and the loadings plot will have different scales. Consequently, you need to rescale the
    vectors or observations (or both) when you overlay the score and loadings plots. There are four common choices of
    scaling. Each scaling emphasizes certain geometric relationships between pairs of observations (such as distances),
    between pairs of variables (such as angles), or between observations and variables. This article discusses the
    geometry behind two-dimensional biplots and shows how biplots enable you to understand relationships in multivariate
    data.

    Args:
        adata: an AnnData object that has pca and loading information prepared.
        pca_components: the index of the pca components in loading matrix. Defaults to [0, 1].
        pca_key: the key to the pca embedding matrix in `adata.obsm`. Defaults to "X_pca".
        loading_key: the key to the pca loading matrix in either `adata.uns` or `adata.varm`. Defaults to "PCs".
        figsize: the size of each subplot. Defaults to (6, 4).
        scale_pca_embedding: whether to scale the pca embedding. Defaults to False.
        draw_pca_embedding: whether to draw the pca embedding. Defaults to False.
        show_text: whether to show the text labels on plot. Defaults to False.
        save_show_or_return: whether to save, show, or return the generated figure. Can be one of 'save', 'show', or
            'return'. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the function will use the {"path": None, "prefix": 'variance_explained', "dpi": None,
            "ext": 'pdf', "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can
            provide a dictionary that properly modify those keys according to your needs. Defaults to {}.

        ax: the axes object on which the graph would be plotted. If None, a new axis would be created. Defaults to None.

    Raises:
        ValueError: invalid `loading_key`.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the matplotlib Axes of the
        figure would be returned.
    """

    if loading_key in adata.uns.keys():
        PCs = adata.uns[loading_key]
    elif loading_key in adata.varm.keys():
        PCs = adata.varm[loading_key]
    else:
        raise ValueError(f"No PC matrix {loading_key} found in neither .uns nor .varm.")

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

    cells = adata.obs_names

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i in range(len(xvector)):
        # arrows project features, e.g. genes, as vectors onto PC axes
        ax.arrow(0, 0, xvector[i] * max(xs), yvector[i] * max(ys), color="r", width=0.0005, head_width=0.0025)

    if show_text:
        for i in range(len(xvector)):
            ax.text(xvector[i] * max(xs) * 1.01, yvector[i] * max(ys) * 1.01, cells[i], color="r")

    ax.set_xlabel("PC" + str(pca_components[0]))
    ax.set_ylabel("PC" + str(pca_components[1]))
    if draw_pca_embedding:
        for i in range(len(xs)):
            # circles project cells
            ax.plot(xs[i] * scalex, ys[i] * scaley, "b", alpha=0.1)
        if show_text:
            for i in range(len(xs)):
                ax.text(xs[i] * scalex * 1.01, ys[i] * scaley * 1.01, list(adata.obs.cluster)[i], color="b", alpha=0.1)

    return save_show_ret("biplot", save_show_or_return, save_kwargs, ax)


def loading(
    adata: AnnData,
    n_pcs: int = 10,
    loading_key: str = "PCs",
    n_top_genes: int = 10,
    ncol: int = 5,
    figsize: Tuple[float] = (6, 4),
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
) -> List[List[Axes]]:
    """Plot the top absolute pca loading genes.

    Red text are positive loading genes while black negative loading genes.

    Args:
        adata: an AnnData object that has pca and loading information prepared.
        n_pcs: the number of pca components. Defaults to 10.
        loading_key: the key to the pca loading matrix. Defaults to "PCs".
        n_top_genes: the number of top genes with the highest absolute loading score. Defaults to 10.
        ncol: the number of columns of the subplots. Defaults to 5.
        figsize: the size of each panel of the figure. Defaults to (6, 4).
        save_show_or_return: whether to save, show, or return the generated figure. Can be one of 'save', 'show', or
            'return'. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the function will use the {"path": None, "prefix": 'biplot', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.

    Raises:
        ValueError: invalid `loading_key`

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the matplotlib Axes of the
        figure would be returned.
    """

    if loading_key in adata.uns.keys():
        PCs = adata.uns[loading_key]
    elif loading_key in adata.varm.keys():
        PCs = adata.varm[loading_key]
    else:
        raise ValueError(f"No PC matrix {loading_key} found in neither .uns nor .varm.")

    if n_pcs is None:
        n_pcs = PCs.shape[1]

    x = np.arange(n_top_genes)
    cells = adata.obs_names

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
                x[j], sort_val[j] * 1.01, cells[sort_ind[j]], color="r" if cur_sign[sort_ind[j]] > 0 else "k"
            )

        axes[cur_row, cur_col].set_title("PC " + str(i))

    return save_show_ret("loading", save_show_or_return, save_kwargs, axes)


def feature_genes(
    adata: AnnData,
    layer: str = "X",
    mode: Optional[Literal["cv_dispersion", "fano_dispersion", "seurat_dispersion", "gini"]] = None,
    figsize: tuple = (4, 3),
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
) -> Optional[PathCollection]:
    """Plot selected feature genes on top of the mean vs. dispersion scatterplot.

    Args:
        adata: an AnnData object.
        layer: the data from a particular layer (include X) used for making the feature gene plot. Defaults to "X".
        mode: the method to select the feature genes (can be either one kind of `dispersion` or `gini`). Defaults to None.
        figsize: the size of each panel of the figure. Defaults to (4, 3).
        save_show_or_return: whether to save, show, or return the generated figure. Can be one of 'save', 'show', or
            'return'. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'feature_genes', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.

    Raises:
        ValueError: vector machine regression result not available in the AnnData object.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the `PathCollection`
        generated with `pyplot.scatter` would be returned.
    """

    mode = adata.uns["feature_selection"] if mode is None else mode
    layer = DynamoAdataKeyManager.get_available_layer_keys(adata, layer, include_protein=False)[0]
    uns_store_key = None

    ordering_genes = adata.var["use_for_pca"] if "use_for_pca" in adata.var.columns else None

    if "_dispersion" not in mode and mode != "gini":
        raise ValueError("Invalid mode!.")

    plt.figure(figsize=figsize)

    if mode == "gini":
        mean_key = layer + "_m"
        variance_key = layer + "_gini"

        if variance_key not in adata.var.columns:
            raise ValueError(
                "Looks like you have not run gene selection yet, try run necessary preprocessing first.")

        mean = DynamoAdataKeyManager.select_layer_data(adata, layer).mean(0)[0]
        table = adata.var.loc[:, [variance_key]]
        table[mean_key] = mean.T
    else:
        if mode == "seurat_dispersion":
            mean_key = "pp_gene_mean"
            variance_key = "pp_gene_variance"
        else:
            prefix = "" if layer == "X" else layer + "_"
            mean_key = prefix + "log_m"
            variance_key = prefix + "log_cv"
            uns_store_key = "velocyto_SVR" if layer == "raw" or layer == "X" else layer + "_velocyto_SVR"

        if not np.all(pd.Series([mean_key, variance_key]).isin(adata.var.columns)):
            raise ValueError("Looks like you have not run gene selection yet, try run necessary preprocessing first.")
        else:
            table = adata.var.loc[:, [mean_key, variance_key]]

    table = table.loc[
        np.isfinite(table[mean_key]) & np.isfinite(table[variance_key])
    ]
    x_min, x_max = (
        np.nanmin(table[mean_key]),
        np.nanmax(table[mean_key]),
    )

    if mode == "cv_dispersion" or mode == "fano_dispersion":
        mu_linspace = np.linspace(x_min, x_max, num=1000)
        mean = adata.uns[uns_store_key]["mean"]
        cv = adata.uns[uns_store_key]["cv"]
        svr_gamma = adata.uns[uns_store_key]["svr_gamma"]
        fit, _ = get_prediction_by_svr(mean, cv, svr_gamma)
        fit = fit(mu_linspace.reshape(-1, 1))

        plt.plot(mu_linspace, fit, alpha=0.4, color="r")

    valid_ind = (
        table.index.isin(ordering_genes.index[ordering_genes])
        if ordering_genes is not None
        else np.ones(table.shape[0], dtype=bool)
    )

    valid_disp_table = table.iloc[valid_ind, :]

    plt.scatter(
        valid_disp_table[mean_key],
        valid_disp_table[variance_key],
        s=3,
        alpha=1,
        color="xkcd:red",
    )

    neg_disp_table = table.iloc[~valid_ind, :]

    plt.scatter(
        neg_disp_table[mean_key],
        neg_disp_table[variance_key],
        s=3,
        alpha=0.5,
        color="xkcd:grey",
    )

    if mode == "gini":
        plt.xlabel("Mean")
        plt.ylabel("Variance")
    else:
        plt.yscale("log")
        plt.xlabel("Mean (log)")
        plt.ylabel("CV (log)")

    return save_show_ret("feature_genes", save_show_or_return, save_kwargs, plt.gcf())


def exp_by_groups(
    adata: AnnData,
    genes: List[str],
    layer: Optional[str] = None,
    group: Optional[str] = None,
    use_ratio: bool = False,
    use_smoothed: bool = True,
    log: bool = True,
    angle: int = 0,
    re_order: bool = True,
    figsize: Tuple[float] = (4, 3),
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
) -> Optional[sns.FacetGrid]:
    """Plot the (labeled) expression values of genes across different groups (time points).

    Args:
        adata: an AnnData object,
        genes: the list of genes that you want to plot the gene expression.
        layer: the layer key containing the expression data. If None, the layer used would be inferred automatically.
            Defaults to None.
        group: the key of group information in `adata.obs` that will be plotted against to. If None, no groups will be
            used. Normally you should supply the column that indicates the time related to the labeling experiment. For
            example, it can be either the labeling time for a kinetic experiment or the chase time for a degradation
            experiment. Defaults to None.
        use_ratio: whether to plot the fraction of expression (for example NTR, new to total ratio) over groups.
            Defaults to False.
        use_smoothed: whether to use the smoothed data as gene expression. Defaults to True.
        log: whether to log1p transform the expression data. Defaults to True.
        angle: the angle to rotate the xtick labels for the purpose of avoiding overlapping between text. Defaults to 0.
        re_order: whether to reorder categories before drawing groups on the x-axis. Defaults to True.
        figsize: the size of each panel of the figure. Defaults to (4, 3).
        save_show_or_return: whether to save, show, or return the generated figure. Can be one of 'save', 'show', or
            'return'. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'exp_by_groups', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.

    Raises:
        ValueError: invalid `genes`.
        ValueError: invalid `group`.
        ValueError: invalid `layer`.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the `FacetGrid` with violin
        plot generated with seaborn would be returned.
    """

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

    return save_show_ret("exp_by_groups", save_show_or_return, save_kwargs, g)


def highest_frac_genes(
    adata: AnnData,
    n_top: int = 30,
    gene_prefix_list: Optional[List[str]] = None,
    gene_prefix_only: bool = True,
    show: bool = True,
    save_path: str = None,
    ax: Optional[Axes] = None,
    gene_annotations: Optional[List[str]] = None,
    gene_annotation_key: str = "use_for_pca",
    log: bool = False,
    store_key: str = "highest_frac_genes",
    orient: Literal["v", "h"] = "v",
    figsize: Optional[Tuple[float]] = None,
    layer: Optional[str] = None,
    title: Optional[str] = None,
    v_rotation: float = 35,
    **kwargs,
) -> Axes:
    """Plot the top genes.

    Args:
        adata: an AnnData object.
        n_top: the number of top genes to show. Defaults to 30.
        gene_prefix_list: a list of gene name prefix. Defaults to None.
        gene_prefix_only: whether to show prefix of genes only. It only takes effect if gene prefix is provided.
            Defaults to True.
        show: whether to show the plots. Defaults to True.
        save_path: the path to save the figure. Defaults to None.
        ax: the axis on which the graph will be plotted. If None, a new axis would be created. Defaults to None.
        gene_annotations: annotations for genes, or annotations for gene prefix subsets. Defaults to None.
        gene_annotation_key:  gene annotations key in adata.var. Defaults to "use_for_pca".
        log: whether to use log scale. Defaults to False.
        store_key: the key for storing expression percent results. Defaults to "highest_frac_genes".
        orient: the orientation of the graph. Can be one of 'v' or 'h'. 'v' for genes in x-axis and 'h' for genes on
            y-axis. Defaults to "v".
        figsize: the size of each panel of the figure. Defaults to None.
        layer: layer on which the gene percents will be computed. Defaults to None.
        title: the title of the figure. Defaults to None.
        v_rotation: rotation of text sticks when the direction is vertica. Defaults to 35.

    Raises:
        ValueError: invalid AnnData object.
        NotImplementedError: invalid `orient`.

    Returns:
        The matplotlib Axes of the figure.
    """

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

    adata = gene_selection.highest_frac_genes(
        adata,
        store_key=store_key,
        n_top=n_top,
        layer=layer,
        gene_prefix_list=gene_prefix_list,
        gene_prefix_only=gene_prefix_only,
    )
    if adata is None:
        # something wrong with user input or compute_top_genes_df
        raise ValueError("Invalid adata. ")
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
                "%s not in adata.var, ignoring the gene annotation key when plotting" % gene_annotation_key,
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
        raise NotImplementedError("Invalid orient option")

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
