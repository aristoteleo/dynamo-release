"""plotting utilities that are used to visualize the curl, divergence."""

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

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
    save_show_ret,
)

docstrings.delete_params("scatters.parameters", "adata", "color", "cmap", "frontier", "sym_c")
docstrings.delete_params("scatters.parameters", "adata", "color", "cmap", "frontier")


@docstrings.with_indent(4)
def speed(
    adata: AnnData,
    basis: str = "pca",
    color: Union[str, List[str], None] = None,
    frontier: bool = True,
    *args,
    **kwargs,
) -> Union[
    Axes,
    List[Axes],
    Tuple[Axes, List[str], Literal["white", "black"]],
    Tuple[List[Axes], List[str], Literal["white", "black"]],
    None,
]:
    """Scatter plot with cells colored by the estimated velocity speed (and other information if provided).

    Args:
        adata: an Annodata object with speed estimated.
        basis: the embedding data in which the vector field was reconstructed and RNA speed was estimated. Defaults to
            "pca".
        color: any column names or gene names, etc. in addition to the `curl` to be used for coloring cells. Defaults to
            None.
        frontier: whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to
            show area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib
            tips & tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from
            scEU-seq paper: https://science.sciencemag.org/content/367/6482/1151. Defaults to True.

    Raises:
        Exception: speed information is not found in `adata`.

    Returns:
        None would be returned by default. If in kwargs `save_show_or_return` is set to be 'return' or 'all', the
        matplotlib axes object of the generated plots would be returned. If `return_all` is set to be true, the list of
        colors used and the font color would also be returned. See docs of `dynamo.pl.scatters` for more information.
    """

    speed_key = "speed" if basis is None else "speed_" + basis
    color_ = [speed_key]
    if not np.any(adata.obs.columns.isin(color_)):
        raise ValueError(f"{speed_key} is not existed in .obs, try run dyn.tl.speed(adata, basis='{basis}') first.")

    if color is not None:
        color = [color] if type(color) == str else color
        color_.extend(color)

    return scatters(adata, color=color_, frontier=frontier, *args, **kwargs)


@docstrings.with_indent(4)
def curl(
    adata: AnnData,
    basis: str = "umap",
    color: Union[str, List[str], None] = None,
    cmap: str = "bwr",
    frontier: bool = True,
    sym_c: bool = True,
    *args,
    **kwargs,
) -> Union[
    Axes,
    List[Axes],
    Tuple[Axes, List[str], Literal["white", "black"]],
    Tuple[List[Axes], List[str], Literal["white", "black"]],
    None,
]:
    """Scatter plot with cells colored by the estimated curl (and other information if provided).

    Cells with negative or positive curl correspond to cells with clock-wise rotation vectors or counter-clock-wise
    ration vectors. Currently only support for 2D vector field. But in principal could be generated to high dimension
    space.

    Args:
        adata: an Annodata object with curl estimated.
        basis: the embedding data in which the vector field was reconstructed and RNA curl was estimated. Defaults to
            "umap".
        color: any column names or gene names, etc. in addition to the `curl` to be used for coloring cells. Defaults to
            None.
        cmap: the color map used for the plot. Defaults to "bwr".
        frontier: whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to
            show area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib
            tips & tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from
            scEU-seq paper: https://science.sciencemag.org/content/367/6482/1151. Defaults to True.
        sym_c: whether do you want to make the limits of continuous color to be symmetric, normally this should be used
            for plotting velocity, curl, divergence or other types of data with both positive or negative values.
            Defaults to True.
        *args: any other positional arguments to be passed to `dynamo.pl.scatters`.
        **kwargs: any other kwargs to be passed to `dynamo.pl.scatters`.

    Raises:
        ValueError: curl information not found in `adata`.

    Returns:
        None would be returned by default. If in kwargs `save_show_or_return` is set to be 'return' or 'all', the
        matplotlib axes object of the generated plots would be returned. If `return_all` is set to be true, the list of
        colors used and the font color would also be returned. See docs of `dynamo.pl.scatters` for more information.
    """

    curl_key = "curl" if basis is None else "curl_" + basis
    color_ = [curl_key]
    if not np.any(adata.obs.columns.isin(color_)):
        raise ValueError(f"{curl_key} is not existed in .obs, try run dyn.tl.curl(adata, basis='{basis}') first.")

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
    color: Union[str, List[str], None] = None,
    cmap: str = "bwr",
    frontier: bool = True,
    sym_c: bool = True,
    *args,
    **kwargs,
):
    """Scatter plot with cells colored by the estimated divergence (and other information if provided).

    Cells with negative or positive divergence correspond to possible sink (stable cell types) or possible source
    (unstable metastable states or progenitors).

    Args:
        adata: an Annodata object with divergence estimated.
        basis: the embedding data in which the vector field was reconstructed and RNA divergence was estimated. Defaults
            to "pca".
        color: any column names or gene names, etc. in addition to the `divergence` to be used for coloring cells.
            Defaults to None.
        cmap: The name of a matplotlib colormap to use for coloring or shading points. Defaults to "bwr".
        frontier: whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to
            show area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib
            tips & tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from
            scEU-seq paper: https://science.sciencemag.org/content/367/6482/1151. Defaults to True.
        sym_c: whether do you want to make the limits of continuous color to be symmetric, normally this should be used
            for plotting velocity, curl, divergence or other types of data with both positive or negative values.
            Defaults to True.

    Raises:
        ValueError: divergence information not found in `adata`.

    Returns:
        None would be returned by default. If in kwargs `save_show_or_return` is set to be 'return' or 'all', the
        matplotlib axes object of the generated plots would be returned. If `return_all` is set to be true, the list of
        colors used and the font color would also be returned. See docs of `dynamo.pl.scatters` for more information.
    """

    div_key = "divergence" if basis is None else "divergence_" + basis
    color_ = [div_key]
    if not np.any(adata.obs.columns.isin(color_)):
        raise ValueError(f"{div_key} is not existed in .obs, try run dyn.tl.divergence(adata, basis='{basis}') first.")

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
    color: Union[str, List[str], None] = None,
    frontier: bool = True,
    *args,
    **kwargs,
) -> Union[
    Axes,
    List[Axes],
    Tuple[Axes, List[str], Literal["white", "black"]],
    Tuple[List[Axes], List[str], Literal["white", "black"]],
    None,
]:
    """Scatter plot with cells colored by the estimated acceleration (and other information if provided).

    Args:
        adata: an Annodata object with curvature estimated.
        basis: the embedding data in which the vector field was reconstructed and RNA curvature was estimated. Defaults
            to "pca".
        color: any column names or gene names, etc. in addition to the `acceleration` to be used for coloring cells.
            Defaults to None.
        frontier: whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to
            show area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib
            tips & tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from
            scEU-seq paper: https://science.sciencemag.org/content/367/6482/1151. Defaults to True.
        *args: any other positional arguments to be passed to `dynamo.pl.scatters`.
        **kwargs: any other kwargs to be passed to `dynamo.pl.scatters`.

    Raises:
        ValueError: acceleration estimation information is not found in `adata`.

    Returns:
        None would be returned by default. If in kwargs `save_show_or_return` is set to be 'return' or 'all', the
        matplotlib axes object of the generated plots would be returned. If `return_all` is set to be true, the list of
        colors used and the font color would also be returned. See docs of `dynamo.pl.scatters` for more information.
    """

    acc_key = "acceleration" if basis is None else "acceleration_" + basis
    color_ = [acc_key]
    if not np.any(adata.obs.columns.isin(color_)):
        raise ValueError(
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
    color: Union[str, List[str], None] = None,
    frontier: bool = True,
    *args,
    **kwargs,
) -> Union[
    Axes,
    List[Axes],
    Tuple[Axes, List[str], Literal["white", "black"]],
    Tuple[List[Axes], List[str], Literal["white", "black"]],
    None,
]:
    """Scatter plot with cells colored by the estimated curvature (and other information if provided).

    Args:
        adata: an Annodata object with curvature estimated.
        basis: the embedding data in which the vector field was reconstructed and RNA curvature was estimated. Defaults
            to "pca".
        color: any column names or gene names, etc. in addition to the `curvature` to be used for coloring cells.
            Defaults to None.
        frontier: whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to
            show area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib
            tips & tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from
            scEU-seq paper: https://science.sciencemag.org/content/367/6482/1151. Defaults to True.
        *args: any other positional arguments to be passed to `dynamo.pl.scatters`.
        **kwargs: any other kwargs to be passed to `dynamo.pl.scatters`.

    Raises:
        ValueError: curvature information is not found in `adata`.

    Returns:
        None would be returned by default. If in kwargs `save_show_or_return` is set to be 'return' or 'all', the
        matplotlib axes object of the generated plots would be returned. If `return_all` is set to be true, the list of
        colors used and the font color would also be returned. See docs of `dynamo.pl.scatters` for more information.
    """

    curv_key = "curvature" if basis is None else "curvature_" + basis
    color_ = [curv_key]
    if not np.any(adata.obs.columns.isin(color_)):
        raise ValueError(
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
    regulators: Optional[List[str]] = None,
    effectors: Optional[List[str]] = None,
    basis: str = "umap",
    jkey: str = "jacobian",
    j_basis: str = "pca",
    x: int = 0,
    y: int = 1,
    layer: str = "M_s",
    highlights: Optional[list] = None,
    cmap: str = "bwr",
    background: Optional[str] = None,
    pointsize: Optional[float] = None,
    figsize: Tuple[float, float] = (6, 4),
    show_legend: bool = True,
    frontier: bool = True,
    sym_c: bool = True,
    sort: Literal["raw", "abs", "neg"] = "abs",
    show_arrowed_spines: bool = False,
    stacked_fraction: bool = False,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
    **kwargs,
) -> Optional[GridSpec]:
    """Scatter plot of Jacobian values across cells.

    Args:
        adata: an Annodata object with Jacobian matrix estimated.
        regulators: the list of genes that will be used as regulators for plotting the Jacobian heatmap, only limited to
            genes that have already performed Jacobian analysis. Defaults to None.
        effectors: the list of genes that will be used as targets for plotting the Jacobian heatmap, only limited to
            genes that have already performed Jacobian analysis. Defaults to None.
        basis: the reduced dimension basis. Defaults to "umap".
        jkey: the key to the jacobian dictionary in .uns. Defaults to "jacobian".
        j_basis: the reduced dimension space that will be used to calculate the jacobian matrix. Defaults to "pca".
        x: the column index of the low dimensional embedding for the x-axis. Defaults to 0.
        y: the column index of the low dimensional embedding for the y-axis. Defaults to 1.
        layer: the layer key with jacobian data. Defaults to "M_s".
        highlights: which color group will be highlighted. if highligts is a list of lists - each list is relate to each
            color element. Defaults to None.
        cmap: the name of a matplotlib colormap to use for coloring or shading points. If no labels or values are passed
            this will be used for shading points according to density (largely only of relevance for very large
            datasets). If values are passed this will be used for shading according the value. Note that if theme is
            passed then this value will be overridden by the corresponding option of the theme. Defaults to "bwr".
        background: the color of the background. Usually this will be either 'white' or 'black', but any color name will
            work. Ideally one wants to match this appropriately to the colors being used for points etc. This is one of
            the things that themes handle for you. Note that if theme is passed then this value will be overridden by
            the corresponding option of the theme. Defaults to None.
        pointsize: the size of plotted points. Defaults to None.
        figsize: the size of the figure. Defaults to (6, 4).
        show_legend: whether to display a legend of the labels. Defaults to True.
        frontier: whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to
            show area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib
            tips & tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from
            scEU-seq paper: https://science.sciencemag.org/content/367/6482/1151. Defaults to True.
        sym_c: whether do you want to make the limits of continuous color to be symmetric, normally this should be used
            for plotting velocity, jacobian, curl, divergence or other types of data with both positive or negative
            values. Defaults to True.
        sort: the method to reorder data so that high values points will be on top of background points. Can be one of
            {'raw', 'abs', 'neg'}, i.e. sorted by raw data, sort by absolute values or sort by negative values. Defaults
            to "abs".
        show_arrowed_spines: whether to show a pair of arrowed spines representing the basis of the scatter is currently
            using. Defaults to False.
        stacked_fraction: whether the jacobian will be represented as a stacked fraction in the title or a linear
            fraction style will be used. Defaults to False.
        save_show_or_return: whether to save, show, or return the figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs.. Defaults to {}.
        **kwargs: any other kwargs that would be passed to `plt._matplotlib_points`.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the matplotlib `GridSpec` of
        the figure would be returned.
        
    Examples:
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

    return save_show_ret(jkey, save_show_or_return, save_kwargs, gs)


def jacobian_heatmap(
    adata: AnnData,
    cell_idx: Union[int, List[int]],
    average: bool = False,
    jkey: str = "jacobian",
    basis: str = "umap",
    regulators: Optional[List[str]] = None,
    effectors: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (7, 5),
    ncols: int = 1,
    cmap: str = "bwr",
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: dict = {},
    **kwargs,
) -> Optional[GridSpec]:
    """Plot the Jacobian matrix for each cell or the average Jacobian matrix of the cells from input indices as a heatmap.

    Note that Jacobian matrix can be understood as a regulatory activity matrix between genes directly computed from the
    reconstructed vector fields.

    Args:
        adata: an Annodata object with Jacobian matrix estimated.
        cell_idx: the numeric indices of the cells that you want to draw the jacobian matrix to reveal the regulatory
            activity.
        average: whether to average the Jacobian matrix of the cells from the input indices.
        jkey: the key to the jacobian dictionary in .uns. Defaults to "jacobian".
        basis: the reduced dimension basis. Defaults to "umap".
        regulators: the list of genes that will be used as regulators for plotting the Jacobian heatmap, only limited to
            genes that have already performed Jacobian analysis. Defaults to None.
        effectors: the list of genes that will be used as targets for plotting the Jacobian heatmap, only limited to
            genes that have already performed Jacobian analysis. Defaults to None.
        figsize: the size of the subplots. Defaults to (7, 5).
        ncols: the number of columns for drawing the heatmaps. Defaults to 1.
        cmap: the mapping from data values to color space. If not provided, the default will depend on whether center is
            set. Defaults to "bwr".
        save_show_or_return: whether to save, show, or return the generated figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.
        **kwargs: any other kwargs passed to `sns.heatmap`.

    Raises:
        ValueError: jacobian information is not found in `adata`

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the matplotlib `GridSpec` of
        the figure would be returned.

    Examples:
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
            f"Jacobian matrix was not calculated for the cells you provided {cell_idx}."
            f"Check adata.uns[{Jacobian_}].values() for available cells that have Jacobian matrix calculated."
            f"Note that limiting calculation of Jacobian matrix only for a subset of cells are required for "
            f"speeding up calculations."
        )
    else:
        cell_names = adata.obs_names[valid_cell_idx]

    total_panels, ncols = len(valid_cell_idx) if not average else 1, ncols
    nrow, ncol = int(np.ceil(total_panels / ncols)), ncols

    if figsize is None:
        g = plt.figure(None, (3 * ncol, 3 * nrow))  # , dpi=160
    else:
        g = plt.figure(None, (figsize[0] * ncol, figsize[1] * nrow))  # , dpi=160

    gs = plt.GridSpec(nrow, ncol)
    heatmap_kwargs = dict(xticklabels=1, yticklabels=1)
    heatmap_kwargs = update_dict(heatmap_kwargs, kwargs)

    if average:
        J = np.mean(Der[:, :, valid_cell_idx], axis=2).T
        J = pd.DataFrame(J, index=regulators, columns=effectors)
        ax = plt.subplot(gs[0, 0])
        sns.heatmap(
            J,
            annot=True,
            ax=ax,
            cmap=cmap,
            cbar=False,
            center=0,
            **heatmap_kwargs,
        )
        ax.set_title("Average Jacobian Matrix")
    else:
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
            ax.title(name)

    return save_show_ret(jkey + "_heatmap", save_show_or_return, save_kwargs, gs)


@docstrings.with_indent(4)
def sensitivity(
    adata: AnnData,
    regulators: Optional[List[str]] = None,
    effectors: Optional[List[str]] = None,
    basis: str = "umap",
    skey: str = "sensitivity",
    s_basis: str = "pca",
    x: int = 0,
    y: int = 1,
    layer: str = "M_s",
    highlights: Optional[list] = None,
    cmap: str = "bwr",
    background: Optional[str] = None,
    pointsize: Optional[float] = None,
    figsize: Tuple[float, float] = (6, 4),
    show_legend: bool = True,
    frontier: bool = True,
    sym_c: bool = True,
    sort: Literal["abs", "neg", "raw"] = "abs",
    show_arrowed_spines: bool = False,
    stacked_fraction: bool = False,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
    **kwargs,
) -> Optional[GridSpec]:
    """Scatter plot of Sensitivity value across cells.

    Args:
        adata: an Annodata object with Jacobian matrix estimated.
        regulators: the list of genes that will be used as regulators for plotting the Jacobian heatmap, only limited to
            genes that have already performed Jacobian analysis. Defaults to None.
        effectors: the list of genes that will be used as targets for plotting the Jacobian heatmap, only limited to
            genes that have already performed Jacobian analysis. Defaults to None.
        basis: the reduced dimension basis. Defaults to "umap".
        skey: the key to the sensitivity dictionary in .uns. Defaults to "sensitivity".
        s_basis: the reduced dimension space that will be used to calculate the jacobian matrix. Defaults to "pca".
        x: the column index of the low dimensional embedding for the x-axis. Defaults to 0.
        y: the column index of the low dimensional embedding for the y-axis. Defaults to 1.
        layer: _description_. Defaults to "M_s".
        highlights: the layer key for the data. Defaults to None.
        cmap: the name of a matplotlib colormap to use for coloring or shading points. If no labels or values are passed
            this will be used for shading points according to density (largely only of relevance for very large
            datasets). If values are passed this will be used for shading according the value. Note that if theme is
            passed then this value will be overridden by the corresponding option of the theme. Defaults to "bwr".
        background: the color of the background. Usually this will be either 'white' or 'black', but any color name will
            work. Ideally one wants to match this appropriately to the colors being used for points etc. This is one of
            the things that themes handle for you. Note that if theme is passed then this value will be overridden by
            the corresponding option of the theme. Defaults to None.
        pointsize: the size of the plotted points. Defaults to None.
        figsize: the size of each subplot. Defaults to (6, 4).
        show_legend: whether to display a legend of the labels. Defaults to True.
        frontier: whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to
            show area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib
            tips & tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from
            scEU-seq paper: https://science.sciencemag.org/content/367/6482/1151. Defaults to True.
        sym_c: whether do you want to make the limits of continuous color to be symmetric, normally this should be used
            for plotting velocity, jacobian, curl, divergence or other types of data with both positive or negative
            values. Defaults to True.
        sort: the method to reorder data so that high values points will be on top of background points. Can be one of
            {'raw', 'abs', 'neg'}, i.e. sorted by raw data, sort by absolute values or sort by negative values. Defaults
            to "abs".
        show_arrowed_spines: whether to show a pair of arrowed spines representing the basis of the scatter is currently
            using. Defaults to False.
        stacked_fraction: whether to represent the jacobianas a stacked fraction in the title or a linear fraction style
            will be used. Defaults to False.
        save_show_or_return: whether to save, show, or return the fugure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.
        **kwargs: any other kwargs passed to `plt._matplotlib_points`.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the matplotlib `GridSpec` of
        the figure would be returned.
    
    Examples:
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

    return save_show_ret(skey, save_show_or_return, save_kwargs, gs)


def sensitivity_heatmap(
    adata: AnnData,
    cell_idx: Union[List[int], int],
    skey: str = "sensitivity",
    basis: str = "pca",
    regulators: Optional[List[str]] = None,
    effectors: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (7, 5),
    ncols: int = 1,
    cmap: str = "bwr",
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
    **kwargs,
) -> Optional[GridSpec]:
    """Plot the Jacobian matrix for each cell as a heatmap.

    Note that Jacobian matrix can be understood as a regulatory activity matrix between genes directly computed from the
    reconstructed vector fields.

    Args:
        adata: an Annodata object with Jacobian matrix estimated.
        cell_idx: the numeric indices of the cells that you want to draw the sensitivity matrix to reveal the regulatory
            activity.
        skey: the key to the sensitivity dictionary in .uns. Defaults to "sensitivity".
        basis: the reduced dimension basis. Defaults to "pca".
        regulators: the list of genes that will be used as regulators for plotting the Jacobian heatmap, only limited to
            genes that have already performed Jacobian analysis. Defaults to None.
        effectors: the list of genes that will be used as targets for plotting the Jacobian heatmap, only limited to
            genes that have already performed Jacobian analysis. Defaults to None.
        figsize: the size of the subplots. Defaults to (7, 5).
        ncols: the number of columns for drawing the heatmaps. Defaults to 1.
        cmap: the mapping from data values to color space. If not provided, the default will depend on whether center is
            set. Defaults to "bwr".
        save_show_or_return: whether to save, show, or return the figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs.. Defaults to {}.
        **kwargs: any other kwargs passed to `sns.heatmap`.

    Raises:
        ValueError: sensitivity data is not found in `adata`.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the matplotlib `GridSpec` of
        the figure would be returned.
    
    Examples:
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

    return save_show_ret(skey + "_heatmap", save_show_or_return, save_kwargs, gs)
