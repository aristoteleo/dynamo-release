"""plotting utilities that are used to visualize the curl, divergence."""

import numpy as np, pandas as pd

from .scatters import scatters
from .scatters import docstrings
from .utils import (
    _matplotlib_points,
    save_fig,
    arrowed_spines,
    deaxis_all,
    despline_all,
)

from ..tools.utils import update_dict


docstrings.delete_params("scatters.parameters", "adata", "color", "cmap", "frontier", "sym_c")
docstrings.delete_params("scatters.parameters", "adata", "color", "cmap", "frontier")

@docstrings.with_indent(4)
def speed(adata, basis='pca', color=None, frontier=True, *args, **kwargs):
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
    >>> dyn.tl.VectorField(adata, basis='pca')
    >>> dyn.tl.speed(adata)
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
def curl(adata, basis='umap', color=None, cmap='bwr', frontier=True, sym_c=True, *args, **kwargs):
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
    >>> dyn.tl.VectorField(adata, basis='umap')
    >>> dyn.tl.curl(adata, basis='umap')
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

    return scatters(adata, color=color_, cmap=cmap, frontier=frontier, sym_c=sym_c, *args, **kwargs)


@docstrings.with_indent(4)
def divergence(adata, basis='pca', color=None, cmap='bwr', frontier=True, sym_c=True, *args, **kwargs):
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
    >>> adata = dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.tl.VectorField(adata, basis='pca')
    >>> dyn.tl.divergence(adata)
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

    return scatters(adata, color=color_, cmap=cmap, frontier=frontier, sym_c=sym_c, *args, **kwargs)


@docstrings.with_indent(4)
def curvature(adata, basis='pca', color=None, frontier=True, *args, **kwargs):
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
    >>> adata = dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.tl.VectorField(adata, basis='pca')
    >>> dyn.tl.curvature(adata)
    >>> dyn.pl.curvature(adata)
    """

    curv_key = "curvature" if basis is None else "curvature_" + basis
    color_ = [curv_key]
    if not np.any(adata.obs.columns.isin(color_)):
        raise Exception(f"{curv_key} is not existed in .obs, try run dyn.tl.curvature(adata, basis='{curv_key}') first.")

    adata.obs[curv_key] = adata.obs[curv_key].astype('float')
    adata_ = adata[~ adata.obs[curv_key].isna(), :]

    if color is not None:
        color = [color] if type(color) == str else color
        color_.extend(color)

    return scatters(adata_, color=color_, frontier=frontier, *args, **kwargs)


@docstrings.with_indent(4)
def jacobian(adata,
             source_genes=None,
             target_genes=None,
             basis="umap",
             x=0,
             y=1,
             highlights=None,
             cmap='bwr',
             background=None,
             pointsize=None,
             figsize=(6, 4),
             show_legend=True,
             frontier=True,
             sym_c=True,
             save_show_or_return="show",
             save_kwargs={},
             **kwargs):
    """\
    Scatter plot with pca basis.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object with Jacobian matrix estimated.
        source_genes: `list` or `None` (default: `None`)
            The list of genes that will be used as regulators for plotting the Jacobian heatmap, only limited to genes
            that have already performed Jacobian analysis.
        target_genes: `List` or `None` (default: `None`)
            The list of genes that will be used as targets for plotting the Jacobian heatmap, only limited to genes
            that have already performed Jacobian analysis.
        basis: `str`
            The reduced dimension.
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
        figsize: `None` or `[float, float]` (default: None)
                The width and height of each panel in the figure.
        show_legend: bool (optional, default True)
            Whether to display a legend of the labels
        frontier: `bool` (default: `False`)
            Whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to show area
            of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib tips & tricks
            cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from scEU-seq paper:
            https://science.sciencemag.org/content/367/6482/1151.
        sym_c: `bool` (default: `False`)
            Whether do you want to make the limits of continuous color to be symmetric, normally this should be used for
            plotting velocity, jacobian, curl, divergence or other types of data with both positive or negative values.
        save_show_or_return: `str` {'save', 'show', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
            will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.
        kwargs:
            Additional arguments passed to plt.scatters.

    Returns
    -------
    Nothing but plots the n_source x n_targets scatter plots of low dimensional embedding of the adata object, each
    corresponds to one element in the Jacobian matrix for all sampled cells.

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> adata = dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.tl.VectorField(adata, basis='pca')
    >>> valid_gene_list = adata[:, adata.var.use_for_velocity].var.index[:2]
    >>> dyn.tl.jacobian(adata, source_genes=valid_gene_list[0], target_genes=valid_gene_list[1])
    >>> dyn.pl.jacobian(adata)
    """

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
    else:
        _background = background

    Jacobian_ = "jacobian" if basis is None else "jacobian_" + basis
    Der, source_genes_, target_genes_, cell_indx, _  =  adata.uns[Jacobian_].values()
    adata_ = adata[cell_indx, :]

    Der, source_genes, target_genes = intersect_sources_targets(source_genes,
                              source_genes_,
                              target_genes,
                              target_genes_,
                              Der)

    cur_pd = pd.DataFrame(
        {
            basis + "_0": adata_.obsm["X_" + basis][:, x],
            basis + "_1": adata_.obsm["X_" + basis][:, y],
        }
    )

    point_size = (
        500.0 / np.sqrt(adata_.shape[0])
        if pointsize is None
        else 500.0 / np.sqrt(adata_.shape[0]) * pointsize
    )
    point_size = 4 * point_size

    scatter_kwargs = dict(
        alpha=0.2, s=point_size, edgecolor=None, linewidth=0,
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
            J = Der if nrow == 1 and ncol == 1 else Der[j, i, :] # dim 0: target; dim 1: source
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
                sym_c=sym_c,
                **scatter_kwargs
            )
            ax.set_title(r'$\frac{{\partial f_{{{}}} }}{{\partial {}}}$'.format(target, source))
            if i + j == 0:
                arrowed_spines(ax, basis, background)
            else:
                despline_all(ax)
                deaxis_all(ax)

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'jacobian', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return gs


def jacobian_heatmap(adata,
                     cell_idx,
                     source_genes=None,
                     target_genes=None,
                     figsize=(7, 5),
                     ncols=1,
                     cmap='bwr',
                     save_show_or_return="show",
                     save_kwargs={},
                     **kwargs):
    """\
    Plot the Jacobian matrix for each cell as a heatmap.

    Note that Jacobian matrix can be understood as a regulatory activity matrix between genes directly computed from the
    reconstructed vector fields.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object with Jacobian matrix estimated.
        source_genes: `list` or `None` (default: `None`)
            The list of genes that will be used as regulators for plotting the Jacobian heatmap, only limited to genes
            that have already performed Jacobian analysis.
        target_genes: `List` or `None` (default: `None`)
            The list of genes that will be used as targets for plotting the Jacobian heatmap, only limited to genes
            that have already performed Jacobian analysis.
        cell_idx: `int` or `list`
            The numeric indices of the cells that you want to draw the jacobian matrix to reveal the regulatory activity.
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
    >>> adata = dyn.pp.recipe_monocle(adata)
    >>> dyn.tl.dynamics(adata)
    >>> dyn.tl.VectorField(adata, basis='pca')
    >>> valid_gene_list = adata[:, adata.var.use_for_velocity].var.index[:2]
    >>> dyn.tl.jacobian(adata, source_genes=valid_gene_list[0], target_genes=valid_gene_list[1])
    >>> dyn.pl.jacobian_heatmap(adata)
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    Jacobian_ = "jacobian" #f basis is None else "jacobian_" + basis
    if type(cell_idx) == int: cell_idx = [cell_idx]
    Der, source_genes_, target_genes_, cell_indx, _  =  adata.uns[Jacobian_].values()
    Der, source_genes, target_genes = intersect_sources_targets(source_genes,
                              source_genes_,
                              target_genes,
                              target_genes_,
                              Der)

    adata_ = adata[cell_indx, :]
    valid_cell_idx = list(set(cell_idx).intersection(cell_indx))
    if len(valid_cell_idx) == 0:
        raise ValueError(f"Jacobian matrix was not calculated for the cells you provided {cell_indx}."
                         f"Check adata.uns[{Jacobian_}].values() for available cells that have Jacobian matrix calculated."
                         f"Note that limiting calculation of Jacobian matrix only for a subset of cells are required for "
                         f"speeding up calculations.")
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
        J = Der[:, :, ind][:, :, 0].T # dim 0: target; dim 1: source
        J = pd.DataFrame(J, index=source_genes, columns=target_genes)
        ax = plt.subplot(gs[i])
        sns.heatmap(J, annot=True, ax=ax, cmap=cmap, cbar=False, center=0, **heatmap_kwargs)
        plt.title(name)

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'jacobian_heatmap', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return gs


def intersect_sources_targets(source_genes,
                              source_genes_,
                              target_genes,
                              target_genes_,
                              Der):
    source_genes = source_genes_ if source_genes is None else source_genes
    target_genes = target_genes_ if target_genes is None else target_genes
    if type(source_genes) == str: source_genes = [source_genes]
    if type(target_genes) == str: target_genes = [target_genes]
    source_genes = list(set(source_genes_).intersection(source_genes))
    target_genes = list(set(target_genes_).intersection(target_genes))
    if len(source_genes) == 0 or len(target_genes) == 0:
        raise ValueError(f"Jacobian related to source genes {source_genes} and target genes {target_genes}"
                         f"you provided are existed. Available source genes includes {source_genes_} while "
                         f"available target genes includes {target_genes_}")
    # subset Der with correct index of selected source / target genes
    valid_source_idx = [i for i, e in enumerate(source_genes_) if e in source_genes]
    valid_target_idx = [i for i, e in enumerate(target_genes_) if e in target_genes]
    Der = Der[valid_target_idx,  :, :][:, valid_source_idx, :] if len(source_genes_) + len(target_genes_) > 2 else Der
    source_genes, target_genes = source_genes_[valid_source_idx], target_genes_[valid_target_idx]

    return Der, source_genes, target_genes
