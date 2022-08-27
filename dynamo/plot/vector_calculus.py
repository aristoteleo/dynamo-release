"""plotting utilities that are used to visualize the curl, divergence."""

from typing import List, Optional, Union

import anndata
import numpy as np
import pandas as pd
from anndata import AnnData

from ..tools.utils import flatten, update_dict
from ..vectorfield.utils import intersect_sources_targets
from .scatters import docstrings, scatters
from .utils import (
    _matplotlib_points,
    arrowed_spines,
    deaxis_all,
    despline_all,
    is_cell_anno_column,
    is_gene_name,
    is_layer_keys,
    save_fig,
)

docstrings.delete_params("scatters.parameters", "adata", "color", "cmap", "frontier", "sym_c")
docstrings.delete_params("scatters.parameters", "adata", "color", "cmap", "frontier")


@docstrings.with_indent(4)
def speed(
    adata: AnnData,
    basis: str = "pca",
    color: Union[str, list, None] = None,
    frontier: bool = True,
    *args,
    **kwargs,
):
    """\
    Scatter plot with cells colored by the estimated velocity speed (and other information if provided).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object with speed estimated.
        basis: `str` or None (default: `pca`)
            The embedding data in which the vector field was reconstructed and RNA speed was estimated.
        color: `str`, `list` or None:
            Any column names or gene names, etc. in addition to the `curl` to be used for coloring cells.
        frontier: `bool` (default: `False`)
            Whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to show area
            of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib tips & tricks
            cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from scEU-seq paper:
            https://science.sciencemag.org/content/367/6482/1151.
        %(scatters.parameters.no_adata|color|cmap|frontier)s

    Returns
    -------
    Nothing but plots scatterplots with cells colored by the estimated speed (and other information if provided).

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> adata = dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.tl.reduceDimension(adata)
    >>> dyn.tl.cell_velocities(adata, basis='pca')
    >>> dyn.vf.VectorField(adata, basis='pca')
    >>> dyn.vf.speed(adata)
    >>> dyn.pl.speed(adata)

    See also:: :func:`..external.ddhodge.curl` for calculating curl with a diffusion graph built from reconstructed vector
    field.
    """

    speed_key = "speed" if basis is None else "speed_" + basis
    color_ = [speed_key]
    if not np.any(adata.obs.columns.isin(color_)):
        raise Exception(f"{speed_key} is not existed in .obs, try run dyn.tl.speed(adata, basis='{basis}') first.")

    if color is not None:
        color = [color] if type(color) == str else color
        color_.extend(color)

    return scatters(adata, color=color_, frontier=frontier, *args, **kwargs)


@docstrings.with_indent(4)
def curl(
    adata: AnnData,
    basis: str = "umap",
    color: Union[str, list, None] = None,
    cmap: str = "bwr",
    frontier: bool = True,
    sym_c: bool = True,
    *args,
    **kwargs,
):
    """\
    Scatter plot with cells colored by the estimated curl (and other information if provided).

    Cells with negative or positive curl correspond to cells with clock-wise rotation vectors or counter-clock-wise
    ration vectors. Currently only support for 2D vector field. But in principal could be generated to high dimension
    space.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object with curl estimated.
        basis: `str` or None (default: `umap`)
            The embedding data in which the vector field was reconstructed and RNA curl was estimated.
        color: `str`, `list` or None:
            Any column names or gene names, etc. in addition to the `curl` to be used for coloring cells.
        frontier: `bool` (default: `False`)
            Whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to show area
            of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib tips & tricks
            cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from scEU-seq paper:
            https://science.sciencemag.org/content/367/6482/1151.
        sym_c: `bool` (default: `False`)
            Whether do you want to make the limits of continuous color to be symmetric, normally this should be used for
            plotting velocity, curl, divergence or other types of data with both positive or negative values.
        %(scatters.parameters.no_adata|color|cmap|frontier|sym_c)s

    Returns
    -------
    Nothing but plots scatterplots with cells colored by the estimated curl (and other information if provided).

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> adata = dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.tl.reduceDimension(adata)
    >>> dyn.tl.cell_velocities(adata, basis='umap')
    >>> dyn.vf.VectorField(adata, basis='umap')
    >>> dyn.vf.curl(adata, basis='umap')
    >>> dyn.pl.curl(adata, basis='umap')

    See also:: :func:`..external.ddhodge.curl` for calculating curl with a diffusion graph built from reconstructed vector
    field.
    """

    curl_key = "curl" if basis is None else "curl_" + basis
    color_ = [curl_key]
    if not np.any(adata.obs.columns.isin(color_)):
        raise Exception(f"{curl_key} is not existed in .obs, try run dyn.tl.curl(adata, basis='{basis}') first.")

    if color is not None:
        color = [color] if type(color) == str else color
        color_.extend(color)

    # adata.obs[curl_key] = adata.obs[curl_key].astype('float')
    # adata_ = adata[~ adata.obs[curl_key].isna(), :]

    return scatters(
        adata,
        color=color_,
        cmap=cmap,
        frontier=frontier,
        sym_c=sym_c,
        *args,
        **kwargs,
    )


@docstrings.with_indent(4)
def divergence(
    adata: AnnData,
    basis: str = "pca",
    color: Union[str, list, None] = None,
    cmap: str = "bwr",
    frontier: bool = True,
    sym_c: bool = True,
    *args,
    **kwargs,
):
    """\
    Scatter plot with cells colored by the estimated divergence (and other information if provided).

    Cells with negative or positive divergence correspond to possible sink (stable cell types) or possible source
    (unstable metastable states or progenitors)

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object with divergence estimated.
        basis: `str` or None (default: `pca`)
            The embedding data in which the vector field was reconstructed and RNA divergence was estimated.
        color: `str`, `list` or None:
            Any column names or gene names, etc. in addition to the `divergence` to be used for coloring cells.
        frontier: `bool` (default: `False`)
            Whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to show area
            of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib tips & tricks
            cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from scEU-seq paper:
            https://science.sciencemag.org/content/367/6482/1151.
        sym_c: `bool` (default: `False`)
            Whether do you want to make the limits of continuous color to be symmetric, normally this should be used for
            plotting velocity, divergence or other types of data with both positive or negative values.
        %(scatters.parameters.no_adata|color|cmap|frontier|sym_c)s

    Returns
    -------
    Nothing but plots scatterplots with cells colored by the estimated divergence (and other information if provided).

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.tl.cell_velocities(adata, basis='pca')
    >>> dyn.vf.VectorField(adata, basis='pca')
    >>> dyn.vf.divergence(adata)
    >>> dyn.pl.divergence(adata)

    See also:: :func:`..external.ddhodge.divergence` for calculating divergence with a diffusion graph built from reconstructed
    vector field.
    """

    div_key = "divergence" if basis is None else "divergence_" + basis
    color_ = [div_key]
    if not np.any(adata.obs.columns.isin(color_)):
        raise Exception(f"{div_key} is not existed in .obs, try run dyn.tl.divergence(adata, basis='{basis}') first.")

    # adata.obs[div_key] = adata.obs[div_key].astype('float')
    # adata_ = adata[~ adata.obs[div_key].isna(), :]

    if color is not None:
        color = [color] if type(color) == str else color
        color_.extend(color)

    return scatters(
        adata,
        color=color_,
        cmap=cmap,
        frontier=frontier,
        sym_c=sym_c,
        *args,
        **kwargs,
    )


@docstrings.with_indent(4)
def acceleration(
    adata: AnnData,
    basis: str = "pca",
    color: Union[str, list, None] = None,
    frontier: bool = True,
    *args,
    **kwargs,
):
    """\
    Scatter plot with cells colored by the estimated acceleration (and other information if provided).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object with curvature estimated.
        basis: `str` or None (default: `pca`)
            The embedding data in which the vector field was reconstructed and RNA curvature was estimated.
        color: `str`, `list` or None:
            Any column names or gene names, etc. in addition to the `acceleration` to be used for coloring cells.
        frontier: `bool` (default: `False`)
            Whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to show area
            of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib tips & tricks
            cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from scEU-seq paper:
            https://science.sciencemag.org/content/367/6482/1151.
        %(scatters.parameters.no_adata|color|cmap|frontier)s

    Returns
    -------
    Nothing but plots scatterplots with cells colored by the estimated curvature (and other information if provided).

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.tl.cell_velocities(adata, basis='pca')
    >>> dyn.vf.VectorField(adata, basis='pca')
    >>> dyn.vf.acceleration(adata)
    >>> dyn.pl.acceleration(adata)
    """

    acc_key = "acceleration" if basis is None else "acceleration_" + basis
    color_ = [acc_key]
    if not np.any(adata.obs.columns.isin(color_)):
        raise Exception(
            f"{acc_key} is not existed in .obs, try run dyn.tl.acceleration(adata, basis='{acc_key}') first."
        )

    adata.obs[acc_key] = adata.obs[acc_key].astype("float")
    adata_ = adata[~adata.obs[acc_key].isna(), :]

    if color is not None:
        color = [color] if type(color) == str else color
        color_.extend(color)

    return scatters(adata_, color=color_, frontier=frontier, *args, **kwargs)


@docstrings.with_indent(4)
def curvature(
    adata: AnnData,
    basis: str = "pca",
    color: Union[str, list, None] = None,
    frontier: bool = True,
    *args,
    **kwargs,
):
    """\
    Scatter plot with cells colored by the estimated curvature (and other information if provided).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object with curvature estimated.
        basis: `str` or None (default: `pca`)
            The embedding data in which the vector field was reconstructed and RNA curvature was estimated.
        color: `str`, `list` or None:
            Any column names or gene names, etc. in addition to the `curvature` to be used for coloring cells.
        frontier: `bool` (default: `False`)
            Whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to show area
            of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib tips & tricks
            cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from scEU-seq paper:
            https://science.sciencemag.org/content/367/6482/1151.
        %(scatters.parameters.no_adata|color|cmap|frontier)s

    Returns
    -------
    Nothing but plots scatterplots with cells colored by the estimated curvature (and other information if provided).

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.tl.cell_velocities(adata, basis='pca')
    >>> dyn.vf.VectorField(adata, basis='pca')
    >>> dyn.vf.curvature(adata)
    >>> dyn.pl.curvature(adata)
    """

    curv_key = "curvature" if basis is None else "curvature_" + basis
    color_ = [curv_key]
    if not np.any(adata.obs.columns.isin(color_)):
        raise Exception(
            f"{curv_key} is not existed in .obs, try run dyn.tl.curvature(adata, basis='{curv_key}') first."
        )

    adata.obs[curv_key] = adata.obs[curv_key].astype("float")
    adata_ = adata[~adata.obs[curv_key].isna(), :]

    if color is not None:
        color = [color] if type(color) == str else color
        color_.extend(color)

    return scatters(adata_, color=color_, frontier=frontier, *args, **kwargs)


@docstrings.with_indent(4)
def jacobian(
    adata: AnnData,
    regulators: Optional[List] = None,
    effectors: Optional[List] = None,
    basis: str = "umap",
    jkey: str = "jacobian",
    j_basis: str = "pca",
    x: int = 0,
    y: int = 1,
    layer: str = "M_s",
    highlights: list = None,
    cmap: str = "bwr",
    background: Optional[str] = None,
    pointsize: Union[None, float] = None,
    figsize: tuple = (6, 4),
    show_legend: bool = True,
    frontier: bool = True,
    sym_c: bool = True,
    sort: str = "abs",
    show_arrowed_spines: bool = False,
    stacked_fraction: bool = False,
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
    **kwargs,
):
    """\
    Scatter plot of Jacobian values across cells.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object with Jacobian matrix estimated.
        regulators: `list` or `None` (default: `None`)
            The list of genes that will be used as regulators for plotting the Jacobian heatmap, only limited to genes
            that have already performed Jacobian analysis.
        effectors: `List` or `None` (default: `None`)
            The list of genes that will be used as targets for plotting the Jacobian heatmap, only limited to genes
            that have already performed Jacobian analysis.
        basis: `str` (default: `umap`)
            The reduced dimension basis.
        jkey: `str` (default: `jacobian`)
            The key to the jacobian dictionary in .uns.
        j_basis: `str` (default: `pca`)
            The reduced dimension space that will be used to calculate the jacobian matrix.
        x: `int` (default: `0`)
            The column index of the low dimensional embedding for the x-axis.
        y: `int` (default: `1`)
            The column index of the low dimensional embedding for the y-axis.
        highlights: `list` (default: None)
            Which color group will be highlighted. if highligts is a list of lists - each list is relate to each color element.
        cmap: string (optional, default 'Blues')
            The name of a matplotlib colormap to use for coloring
            or shading points. If no labels or values are passed
            this will be used for shading points according to
            density (largely only of relevance for very large
            datasets). If values are passed this will be used for
            shading according the value. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        background: string or None (optional, default 'None`)
            The color of the background. Usually this will be either
            'white' or 'black', but any color name will work. Ideally
            one wants to match this appropriately to the colors being
            used for points etc. This is one of the things that themes
            handle for you. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        figsize: `None` or `[float, float]` (default: (6, 4))
                The width and height of each panel in the figure.
        show_legend: bool (optional, default True)
            Whether to display a legend of the labels
        frontier: `bool` (default: `False`)
            Whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to show area
            of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib tips & tricks
            cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from scEU-seq paper:
            https://science.sciencemag.org/content/367/6482/1151.
        sym_c: `bool` (default: `True`)
            Whether do you want to make the limits of continuous color to be symmetric, normally this should be used for
            plotting velocity, jacobian, curl, divergence or other types of data with both positive or negative values.
        sort: `str` (optional, default `abs`)
            The method to reorder data so that high values points will be on top of background points. Can be one of
            {'raw', 'abs', 'neg'}, i.e. sorted by raw data, sort by absolute values or sort by negative values.
        show_arrowed_spines: bool (optional, default False)
            Whether to show a pair of arrowed spines representing the basis of the scatter is currently using.
        stacked_fraction: bool (default: False)
            If True the jacobian will be represented as a stacked fraction in the title, otherwise a linear fraction
            style is used.
        save_show_or_return: `str` {'save', 'show', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig
            function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True,
            "close": True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly
            modify those keys according to your needs.
        kwargs:
            Additional arguments passed to plt._matplotlib_points.

    Returns
    -------
    Nothing but plots the n_source x n_targets scatter plots of low dimensional embedding of the adata object, each
    corresponds to one element in the Jacobian matrix for all sampled cells.

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.tl.cell_velocities(adata, basis='pca')
    >>> dyn.vf.VectorField(adata, basis='pca')
    >>> valid_gene_list = adata[:, adata.var.use_for_transition].var.index[:2]
    >>> dyn.vf.jacobian(adata, regulators=valid_gene_list[0], effectors=valid_gene_list[1])
    >>> dyn.pl.jacobian(adata)
    """

    regulators, effectors = (
        list(np.unique(regulators)) if regulators is not None else None,
        list(np.unique(effectors)) if effectors is not None else None,
    )
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
    else:
        _background = background

    Jacobian_ = jkey if j_basis is None else jkey + "_" + j_basis
    Der, cell_indx, jacobian_gene, regulators_, effectors_ = (
        adata.uns[Jacobian_].get(jkey.split("_")[-1]),
        adata.uns[Jacobian_].get("cell_idx"),
        adata.uns[Jacobian_].get(jkey.split("_")[-1] + "_gene"),
        adata.uns[Jacobian_].get("regulators"),
        adata.uns[Jacobian_].get("effectors"),
    )

    adata_ = adata[cell_indx, :]

    if regulators is None and effectors is not None:
        regulators = effectors
    elif effectors is None and regulators is not None:
        effectors = regulators
    # test the simulation data here
    if regulators_ is None or effectors_ is None:
        if Der.shape[0] != adata_.n_vars:
            source_genes = [j_basis + "_" + str(i) for i in range(Der.shape[0])]
            target_genes = [j_basis + "_" + str(i) for i in range(Der.shape[1])]
        else:
            source_genes, target_genes = adata_.var_names, adata_.var_names
    else:
        Der, source_genes, target_genes = intersect_sources_targets(
            regulators,
            regulators_,
            effectors,
            effectors_,
            Der if jacobian_gene is None else jacobian_gene,
        )

    ## integrate this with the code in scatter ##

    if type(x) is int and type(y) is int:
        prefix = "X_"
        cur_pd = pd.DataFrame(
            {
                basis + "_" + str(x): adata_.obsm[prefix + basis][:, x],
                basis + "_" + str(y): adata_.obsm[prefix + basis][:, y],
            }
        )
    elif is_gene_name(adata_, x) and is_gene_name(adata_, y):
        cur_pd = pd.DataFrame(
            {
                x: adata_.obs_vector(k=x, layer=None) if layer == "X" else adata_.obs_vector(k=x, layer=layer),
                y: adata_.obs_vector(k=y, layer=None) if layer == "X" else adata_.obs_vector(k=y, layer=layer),
            }
        )
        # cur_pd = cur_pd.loc[(cur_pd > 0).sum(1) > 1, :]
        cur_pd.columns = [
            x + " (" + layer + ")",
            y + " (" + layer + ")",
        ]
    elif is_cell_anno_column(adata_, x) and is_cell_anno_column(adata_, y):
        cur_pd = pd.DataFrame(
            {
                x: adata_.obs_vector(x),
                y: adata_.obs_vector(y),
            }
        )
        cur_pd.columns = [x, y]
    elif is_cell_anno_column(adata_, x) and is_gene_name(adata_, y):
        cur_pd = pd.DataFrame(
            {
                x: adata_.obs_vector(x),
                y: adata_.obs_vector(k=y, layer=None) if layer == "X" else adata_.obs_vector(k=y, layer=layer),
            }
        )
        cur_pd.columns = [x, y + " (" + layer + ")"]
    elif is_gene_name(adata_, x) and is_cell_anno_column(adata_, y):
        cur_pd = pd.DataFrame(
            {
                x: adata_.obs_vector(k=x, layer=None) if layer == "X" else adata_.obs_vector(k=x, layer=layer),
                y: adata_.obs_vector(y),
            }
        )
        # cur_pd = cur_pd.loc[cur_pd.iloc[:, 0] > 0, :]
        cur_pd.columns = [x + " (" + layer + ")", y]
    elif is_layer_keys(adata_, x) and is_layer_keys(adata_, y):
        x_, y_ = adata_[:, basis].layers[x], adata_[:, basis].layers[y]
        cur_pd = pd.DataFrame({x: flatten(x_), y: flatten(y_)})
        # cur_pd = cur_pd.loc[cur_pd.iloc[:, 0] > 0, :]
        cur_pd.columns = [x, y]
    elif type(x) in [anndata._core.views.ArrayView, np.ndarray] and type(y) in [
        anndata._core.views.ArrayView,
        np.ndarray,
    ]:
        cur_pd = pd.DataFrame({"x": flatten(x), "y": flatten(y)})
        cur_pd.columns = ["x", "y"]

    point_size = 500.0 / np.sqrt(adata_.shape[0]) if pointsize is None else 500.0 / np.sqrt(adata_.shape[0]) * pointsize
    point_size = 4 * point_size

    scatter_kwargs = dict(
        alpha=0.2,
        s=point_size,
        edgecolor=None,
        linewidth=0,
    )  # (0, 0, 0, 1)
    if kwargs is not None:
        scatter_kwargs.update(kwargs)

    nrow, ncol = len(source_genes), len(target_genes)
    if figsize is None:
        g = plt.figure(None, (3 * ncol, 3 * nrow), facecolor=_background)  # , dpi=160
    else:
        g = plt.figure(None, (figsize[0] * ncol, figsize[1] * nrow), facecolor=_background)  # , dpi=160

    gs = plt.GridSpec(nrow, ncol, wspace=0.12)

    for i, source in enumerate(source_genes):
        for j, target in enumerate(target_genes):
            ax = plt.subplot(gs[i * ncol + j])
            J = Der[j, i, :]  # dim 0: target; dim 1: source
            cur_pd["jacobian"] = J

            # cur_pd.loc[:, "jacobian"] = np.array([scinot(i) for i in cur_pd.loc[:, "jacobian"].values])
            v_max = np.max(np.abs(J))
            scatter_kwargs.update({"vmin": -v_max, "vmax": v_max})
            ax, color = _matplotlib_points(
                cur_pd.iloc[:, [0, 1]].values,
                ax=ax,
                labels=None,
                values=J,
                highlights=highlights,
                cmap=cmap,
                color_key=None,
                color_key_cmap=None,
                background=_background,
                width=figsize[0],
                height=figsize[1],
                show_legend=show_legend,
                frontier=frontier,
                sort=sort,
                sym_c=sym_c,
                **scatter_kwargs,
            )
            if stacked_fraction:
                ax.set_title(r"$\frac{\partial f_{%s}}{\partial x_{%s}}$" % (target, source))
            else:
                ax.set_title(r"$\partial f_{%s} / \partial x_{%s}$" % (target, source))
            if i + j == 0 and show_arrowed_spines:
                arrowed_spines(ax, basis, background)
            else:
                despline_all(ax)
                deaxis_all(ax)

    if save_show_or_return == "save":
        s_kwargs = {
            "path": None,
            "prefix": jkey,
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
        return gs


def jacobian_heatmap(
    adata: AnnData,
    cell_idx: Union[int, List],
    jkey: str = "jacobian",
    basis: str = "umap",
    regulators: Optional[List] = None,
    effectors: Optional[List] = None,
    figsize: tuple = (7, 5),
    ncols: int = 1,
    cmap: str = "bwr",
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
    **kwargs,
):
    """\
    Plot the Jacobian matrix for each cell as a heatmap.

    Note that Jacobian matrix can be understood as a regulatory activity matrix between genes directly computed from the
    reconstructed vector fields.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object with Jacobian matrix estimated.
        cell_idx: `int` or `list`
            The numeric indices of the cells that you want to draw the jacobian matrix to reveal the regulatory activity.
        jkey: `str` (default: `jacobian`)
            The key to the jacobian dictionary in .uns.
        basis: `str`
            The reduced dimension basis.
        regulators: `list` or `None` (default: `None`)
            The list of genes that will be used as regulators for plotting the Jacobian heatmap, only limited to genes
            that have already performed Jacobian analysis.
        effectors: `List` or `None` (default: `None`)
            The list of genes that will be used as targets for plotting the Jacobian heatmap, only limited to genes
            that have already performed Jacobian analysis.
        figsize: `None` or `[float, float]` (default: None)
                The width and height of each panel in the figure.
        ncols: `int` (default: `1`)
            The number of columns for drawing the heatmaps.
        cmap: `str` (default: `bwr`)
            The mapping from data values to color space. If not provided, the default will depend on whether center is set.
        save_show_or_return: `str` {'save', 'show', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
            will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.
        kwargs:
            Additional arguments passed to sns.heatmap.

    Returns
    -------
        Nothing but plots the n_cell_idx heatmaps of the corresponding Jacobian matrix for each selected cell.

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.tl.cell_velocities(adata, basis='pca')
    >>> dyn.vf.VectorField(adata, basis='pca')
    >>> valid_gene_list = adata[:, adata.var.use_for_transition].var.index[:2]
    >>> dyn.vf.jacobian(adata, regulators=valid_gene_list[0], effectors=valid_gene_list[1])
    >>> dyn.pl.jacobian_heatmap(adata)
    """

    regulators, effectors = (
        list(np.unique(regulators)) if regulators is not None else None,
        list(np.unique(effectors)) if effectors is not None else None,
    )
    import matplotlib.pyplot as plt
    import seaborn as sns

    Jacobian_ = jkey if basis is None else jkey + "_" + basis
    if type(cell_idx) == int:
        cell_idx = [cell_idx]
    Der, cell_indx, jacobian_gene, regulators_, effectors_ = (
        adata.uns[Jacobian_].get(jkey.split("_")[-1]),
        adata.uns[Jacobian_].get("cell_idx"),
        adata.uns[Jacobian_].get(jkey.split("_")[-1] + "_gene"),
        adata.uns[Jacobian_].get("regulators"),
        adata.uns[Jacobian_].get("effectors"),
    )

    Der, regulators, effectors = intersect_sources_targets(regulators, regulators_, effectors, effectors_, Der)

    adata_ = adata[cell_indx, :]
    valid_cell_idx = list(set(cell_idx).intersection(cell_indx))
    if len(valid_cell_idx) == 0:
        raise ValueError(
            f"Jacobian matrix was not calculated for the cells you provided {cell_indx}."
            f"Check adata.uns[{Jacobian_}].values() for available cells that have Jacobian matrix calculated."
            f"Note that limiting calculation of Jacobian matrix only for a subset of cells are required for "
            f"speeding up calculations."
        )
    else:
        cell_names = adata.obs_names[valid_cell_idx]

    total_panels, ncols = len(valid_cell_idx), ncols
    nrow, ncol = int(np.ceil(total_panels / ncols)), ncols

    if figsize is None:
        g = plt.figure(None, (3 * ncol, 3 * nrow))  # , dpi=160
    else:
        g = plt.figure(None, (figsize[0] * ncol, figsize[1] * nrow))  # , dpi=160

    gs = plt.GridSpec(nrow, ncol)
    heatmap_kwargs = dict(xticklabels=1, yticklabels=1)
    heatmap_kwargs = update_dict(heatmap_kwargs, kwargs)
    for i, name in enumerate(cell_names):
        ind = np.where(adata_.obs_names == name)[0]
        J = Der[:, :, ind][:, :, 0].T  # dim 0: target; dim 1: source
        J = pd.DataFrame(J, index=regulators, columns=effectors)
        ax = plt.subplot(gs[i])
        sns.heatmap(
            J,
            annot=True,
            ax=ax,
            cmap=cmap,
            cbar=False,
            center=0,
            **heatmap_kwargs,
        )
        plt.title(name)

    if save_show_or_return == "save":
        s_kwargs = {
            "path": None,
            "prefix": jkey + "_heatmap",
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
        return gs


@docstrings.with_indent(4)
def sensitivity(
    adata,
    regulators=None,
    effectors=None,
    basis="umap",
    skey="sensitivity",
    s_basis="pca",
    x=0,
    y=1,
    layer="M_s",
    highlights=None,
    cmap="bwr",
    background=None,
    pointsize=None,
    figsize=(6, 4),
    show_legend=True,
    frontier=True,
    sym_c=True,
    sort="abs",
    show_arrowed_spines=False,
    stacked_fraction=False,
    save_show_or_return="show",
    save_kwargs={},
    **kwargs,
):
    """\
    Scatter plot of Sensitivity value across cells.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object with Jacobian matrix estimated.
        regulators: `list` or `None` (default: `None`)
            The list of genes that will be used as regulators for plotting the Jacobian heatmap, only limited to genes
            that have already performed Jacobian analysis.
        effectors: `List` or `None` (default: `None`)
            The list of genes that will be used as targets for plotting the Jacobian heatmap, only limited to genes
            that have already performed Jacobian analysis.
        basis: `str` (default: `umap`)
            The reduced dimension basis.
        skey: `str` (default: `sensitivity`)
            The key to the sensitivity dictionary in .uns.
        s_basis: `str` (default: `pca`)
            The reduced dimension space that will be used to calculate the jacobian matrix.
        x: `int` (default: `0`)
            The column index of the low dimensional embedding for the x-axis.
        y: `int` (default: `1`)
            The column index of the low dimensional embedding for the y-axis.
        highlights: `list` (default: None)
            Which color group will be highlighted. if highligts is a list of lists - each list is relate to each color element.
        cmap: string (optional, default 'Blues')
            The name of a matplotlib colormap to use for coloring
            or shading points. If no labels or values are passed
            this will be used for shading points according to
            density (largely only of relevance for very large
            datasets). If values are passed this will be used for
            shading according the value. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        background: string or None (optional, default 'None`)
            The color of the background. Usually this will be either
            'white' or 'black', but any color name will work. Ideally
            one wants to match this appropriately to the colors being
            used for points etc. This is one of the things that themes
            handle for you. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        figsize: `None` or `[float, float]` (default: (6, 4))
                The width and height of each panel in the figure.
        show_legend: bool (optional, default True)
            Whether to display a legend of the labels
        frontier: `bool` (default: `False`)
            Whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to show area
            of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib tips & tricks
            cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from scEU-seq paper:
            https://science.sciencemag.org/content/367/6482/1151.
        sym_c: `bool` (default: `True`)
            Whether do you want to make the limits of continuous color to be symmetric, normally this should be used for
            plotting velocity, jacobian, curl, divergence or other types of data with both positive or negative values.
        sort: `str` (optional, default `abs`)
            The method to reorder data so that high values points will be on top of background points. Can be one of
            {'raw', 'abs', 'neg'}, i.e. sorted by raw data, sort by absolute values or sort by negative values.
        show_arrowed_spines: bool (optional, default False)
            Whether to show a pair of arrowed spines representing the basis of the scatter is currently using.
        stacked_fraction: bool (default: False)
            If True the jacobian will be represented as a stacked fraction in the title, otherwise a linear fraction
            style is used.
        save_show_or_return: `str` {'save', 'show', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig
            function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True,
            "close": True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly
            modify those keys according to your needs.
        kwargs:
            Additional arguments passed to plt._matplotlib_points.

    Returns
    -------
    Nothing but plots the n_source x n_targets scatter plots of low dimensional embedding of the adata object, each
    corresponds to one element in the Jacobian matrix for all sampled cells.

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.tl.cell_velocities(adata, basis='pca')
    >>> dyn.vf.VectorField(adata, basis='pca')
    >>> valid_gene_list = adata[:, adata.var.use_for_transition].var.index[:2]
    >>> dyn.vf.sensitivity(adata, regulators=valid_gene_list[0], effectors=valid_gene_list[1])
    >>> dyn.pl.sensitivity(adata)
    """

    regulators, effectors = (
        list(np.unique(regulators)) if regulators is not None else None,
        list(np.unique(effectors)) if effectors is not None else None,
    )

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
    else:
        _background = background

    Sensitivity_ = skey if s_basis is None else skey + "_" + s_basis
    Der, cell_indx, sensitivity_gene, regulators_, effectors_ = (
        adata.uns[Sensitivity_].get(skey.split("_")[-1]),
        adata.uns[Sensitivity_].get("cell_idx"),
        adata.uns[Sensitivity_].get(skey.split("_")[-1] + "_gene"),
        adata.uns[Sensitivity_].get("regulators"),
        adata.uns[Sensitivity_].get("effectors"),
    )

    adata_ = adata[cell_indx, :]

    # test the simulation data here
    if regulators_ is None or effectors_ is None:
        if Der.shape[0] != adata_.n_vars:
            source_genes = [s_basis + "_" + str(i) for i in range(Der.shape[0])]
            target_genes = [s_basis + "_" + str(i) for i in range(Der.shape[1])]
        else:
            source_genes, target_genes = adata_.var_names, adata_.var_names
    else:
        Der, source_genes, target_genes = intersect_sources_targets(
            regulators,
            regulators_,
            effectors,
            effectors_,
            Der if sensitivity_gene is None else sensitivity_gene,
        )

    ## integrate this with the code in scatter ##

    if type(x) is int and type(y) is int:
        prefix = "X_"
        cur_pd = pd.DataFrame(
            {
                basis + "_" + str(x): adata_.obsm[prefix + basis][:, x],
                basis + "_" + str(y): adata_.obsm[prefix + basis][:, y],
            }
        )
    elif is_gene_name(adata_, x) and is_gene_name(adata_, y):
        cur_pd = pd.DataFrame(
            {
                x: adata_.obs_vector(k=x, layer=None) if layer == "X" else adata_.obs_vector(k=x, layer=layer),
                y: adata_.obs_vector(k=y, layer=None) if layer == "X" else adata_.obs_vector(k=y, layer=layer),
            }
        )
        # cur_pd = cur_pd.loc[(cur_pd > 0).sum(1) > 1, :]
        cur_pd.columns = [
            x + " (" + layer + ")",
            y + " (" + layer + ")",
        ]
    elif is_cell_anno_column(adata_, x) and is_cell_anno_column(adata_, y):
        cur_pd = pd.DataFrame(
            {
                x: adata_.obs_vector(x),
                y: adata_.obs_vector(y),
            }
        )
        cur_pd.columns = [x, y]
    elif is_cell_anno_column(adata_, x) and is_gene_name(adata_, y):
        cur_pd = pd.DataFrame(
            {
                x: adata_.obs_vector(x),
                y: adata_.obs_vector(k=y, layer=None) if layer == "X" else adata_.obs_vector(k=y, layer=layer),
            }
        )
        cur_pd.columns = [x, y + " (" + layer + ")"]
    elif is_gene_name(adata_, x) and is_cell_anno_column(adata_, y):
        cur_pd = pd.DataFrame(
            {
                x: adata_.obs_vector(k=x, layer=None) if layer == "X" else adata_.obs_vector(k=x, layer=layer),
                y: adata_.obs_vector(y),
            }
        )
        # cur_pd = cur_pd.loc[cur_pd.iloc[:, 0] > 0, :]
        cur_pd.columns = [x + " (" + layer + ")", y]
    elif is_layer_keys(adata_, x) and is_layer_keys(adata_, y):
        x_, y_ = adata_[:, basis].layers[x], adata_[:, basis].layers[y]
        cur_pd = pd.DataFrame({x: flatten(x_), y: flatten(y_)})
        # cur_pd = cur_pd.loc[cur_pd.iloc[:, 0] > 0, :]
        cur_pd.columns = [x, y]
    elif type(x) in [anndata._core.views.ArrayView, np.ndarray] and type(y) in [
        anndata._core.views.ArrayView,
        np.ndarray,
    ]:
        cur_pd = pd.DataFrame({"x": flatten(x), "y": flatten(y)})
        cur_pd.columns = ["x", "y"]

    point_size = 500.0 / np.sqrt(adata_.shape[0]) if pointsize is None else 500.0 / np.sqrt(adata_.shape[0]) * pointsize
    point_size = 4 * point_size

    scatter_kwargs = dict(
        alpha=0.2,
        s=point_size,
        edgecolor=None,
        linewidth=0,
    )  # (0, 0, 0, 1)
    if kwargs is not None:
        scatter_kwargs.update(kwargs)

    nrow, ncol = len(source_genes), len(target_genes)
    if figsize is None:
        g = plt.figure(None, (3 * ncol, 3 * nrow), facecolor=_background)  # , dpi=160
    else:
        g = plt.figure(None, (figsize[0] * ncol, figsize[1] * nrow), facecolor=_background)  # , dpi=160

    gs = plt.GridSpec(nrow, ncol, wspace=0.12)

    for i, source in enumerate(source_genes):
        for j, target in enumerate(target_genes):
            ax = plt.subplot(gs[i * ncol + j])
            S = Der[j, i, :]  # dim 0: target; dim 1: source
            cur_pd["sensitivity"] = S

            # cur_pd.loc[:, "sensitivity"] = np.array([scinot(i) for i in cur_pd.loc[:, "jacobian"].values])
            v_max = np.max(np.abs(S))
            scatter_kwargs.update({"vmin": -v_max, "vmax": v_max})
            ax, color = _matplotlib_points(
                cur_pd.iloc[:, [0, 1]].values,
                ax=ax,
                labels=None,
                values=S,
                highlights=highlights,
                cmap=cmap,
                color_key=None,
                color_key_cmap=None,
                background=_background,
                width=figsize[0],
                height=figsize[1],
                show_legend=show_legend,
                frontier=frontier,
                sort=sort,
                sym_c=sym_c,
                **scatter_kwargs,
            )
            if stacked_fraction:
                ax.set_title(r"$\frac{d x_{%s}}{d x_{%s}}$" % (target, source))
            else:
                ax.set_title(r"$d x_{%s} / d x_{%s}$" % (target, source))
            if i + j == 0 and show_arrowed_spines:
                arrowed_spines(ax, basis, background)
            else:
                despline_all(ax)
                deaxis_all(ax)

    if save_show_or_return == "save":
        s_kwargs = {
            "path": None,
            "prefix": skey,
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
        return gs


def sensitivity_heatmap(
    adata,
    cell_idx,
    skey="sensitivity",
    basis="pca",
    regulators=None,
    effectors=None,
    figsize=(7, 5),
    ncols=1,
    cmap="bwr",
    save_show_or_return="show",
    save_kwargs={},
    **kwargs,
):
    """\
    Plot the Jacobian matrix for each cell as a heatmap.

    Note that Jacobian matrix can be understood as a regulatory activity matrix between genes directly computed from the
    reconstructed vector fields.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object with Jacobian matrix estimated.
        cell_idx: `int` or `list`
            The numeric indices of the cells that you want to draw the sensitivity matrix to reveal the regulatory activity.
        skey: `str` (default: `sensitivity`)
            The key to the sensitivity dictionary in .uns.
        basis: `str`
            The reduced dimension basis.
        regulators: `list` or `None` (default: `None`)
            The list of genes that will be used as regulators for plotting the Jacobian heatmap, only limited to genes
            that have already performed Jacobian analysis.
        effectors: `List` or `None` (default: `None`)
            The list of genes that will be used as targets for plotting the Jacobian heatmap, only limited to genes
            that have already performed Jacobian analysis.
        figsize: `None` or `[float, float]` (default: None)
                The width and height of each panel in the figure.
        ncols: `int` (default: `1`)
            The number of columns for drawing the heatmaps.
        cmap: `str` (default: `bwr`)
            The mapping from data values to color space. If not provided, the default will depend on whether center is set.
        save_show_or_return: `str` {'save', 'show', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
            will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.
        kwargs:
            Additional arguments passed to sns.heatmap.

    Returns
    -------
        Nothing but plots the n_cell_idx heatmaps of the corresponding Jacobian matrix for each selected cell.

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.tl.reduceDimension(adata)
    >>> dyn.tl.cell_velocities(adata, basis='pca')
    >>> dyn.vf.VectorField(adata, basis='pca')
    >>> valid_gene_list = adata[:, adata.var.use_for_transition].var.index[:2]
    >>> dyn.vf.sensitivity(adata, regulators=valid_gene_list[0], effectors=valid_gene_list[1])
    >>> dyn.pl.sensitivity_heatmap(adata)
    """

    regulators, effectors = (
        list(np.unique(regulators)) if regulators is not None else None,
        list(np.unique(effectors)) if effectors is not None else None,
    )

    import matplotlib.pyplot as plt
    import seaborn as sns

    Sensitivity_ = skey if basis is None else skey + "_" + basis
    if type(cell_idx) == int:
        cell_idx = [cell_idx]
    Der, cell_indx, sensitivity_gene, regulators_, effectors_ = (
        adata.uns[Sensitivity_].get(skey.split("_")[-1]),
        adata.uns[Sensitivity_].get("cell_idx"),
        adata.uns[Sensitivity_].get(skey.split("_")[-1] + "_gene"),
        adata.uns[Sensitivity_].get("regulators"),
        adata.uns[Sensitivity_].get("effectors"),
    )

    Der, regulators, effectors = intersect_sources_targets(regulators, regulators_, effectors, effectors_, Der)

    adata_ = adata[cell_indx, :]
    valid_cell_idx = list(set(cell_idx).intersection(cell_indx))
    if len(valid_cell_idx) == 0:
        raise ValueError(
            f"Sensitivity matrix was not calculated for the cells you provided {cell_indx}."
            f"Check adata.uns[{Sensitivity_}].values() for available cells that have Sensitivity matrix calculated."
            f"Note that limiting calculation of Sensitivity matrix only for a subset of cells are required for "
            f"speeding up calculations."
        )
    else:
        cell_names = adata.obs_names[valid_cell_idx]

    total_panels, ncols = len(valid_cell_idx), ncols
    nrow, ncol = int(np.ceil(total_panels / ncols)), ncols

    if figsize is None:
        g = plt.figure(None, (3 * ncol, 3 * nrow))  # , dpi=160
    else:
        g = plt.figure(None, (figsize[0] * ncol, figsize[1] * nrow))  # , dpi=160

    gs = plt.GridSpec(nrow, ncol)
    heatmap_kwargs = dict(xticklabels=1, yticklabels=1)
    heatmap_kwargs = update_dict(heatmap_kwargs, kwargs)
    for i, name in enumerate(cell_names):
        ind = np.where(adata_.obs_names == name)[0]
        J = Der[:, :, ind][:, :, 0].T  # dim 0: target; dim 1: source
        J = pd.DataFrame(J, index=regulators, columns=effectors)
        ax = plt.subplot(gs[i])
        sns.heatmap(
            J,
            annot=True,
            ax=ax,
            cmap=cmap,
            cbar=False,
            center=0,
            **heatmap_kwargs,
        )
        plt.title(name)

    if save_show_or_return == "save":
        s_kwargs = {
            "path": None,
            "prefix": skey + "_heatmap",
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
        return gs
