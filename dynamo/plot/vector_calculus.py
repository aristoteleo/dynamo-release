"""plotting utilities that are used to visualize the curl, divergence."""

import numpy as np, pandas as pd
from ..configuration import set_figure_params

from .scatters import scatters
from .scatters import docstrings
from .utils import _matplotlib_points, save_fig

from ..tools.utils import update_dict


docstrings.delete_params("scatters.parameters", "adata", "color", "cmap")

@docstrings.with_indent(4)
def curl(adata, color=None, cmap='bwr', *args, **kwargs):
    """\
    Scatter plot with cells colored by the estimated curl (and other information if provided).

    Cells with negative or positive curl correspond to cells with clock-wise rotation vectors or counter-clock-wise
    ration vectors. Currently only support for 2D vector field. But in principal could be generated to high dimension
    space.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object with curl estimated.
        color: `str`, `list` or None:
            Any column names or gene names, etc. in addition to the `curl` to be used for coloring cells.
        %(scatters.parameters.no_adata|color|cmap)s

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
    >>> dyn.tl.curl(adata)
    >>> dyn.pl.curl(adata)

    See also:: :func:`..external.ddhodge.curl` for calculating curl with a diffusion graph built from reconstructed vector
    field.
    """

    color_ = ['curl']
    if not np.any(adata.obs.columns.isin(color_)):
        raise Exception(f"curl is not existed in .obs, try run dyn.tl.curl(adata) first.")

    if color is not None:
        color = [color] if type(color) == str else color
        color_.extend(color)

    scatters(adata, color=color_, cmap=cmap, *args, **kwargs)


@docstrings.with_indent(4)
def divergence(adata, color=None, cmap='bwr', *args, **kwargs):
    """\
    Scatter plot with cells colored by the estimated divergence (and other information if provided).

    Cells with negative or positive divergence correspond to possible sink (stable cell types) or possible source
    (unstable metastable states or progenitors)

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object with divergence estimated.
        %(scatters.parameters.no_adata|color|cmap)s

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

    color_ = ['divergence']
    if not np.any(adata.obs.columns.isin(color_)):
        raise Exception(f"divergence is not existed in .obs, try run dyn.tl.divergence(adata) first.")

    adata.obs.divergence = adata.obs.divergence.astype('float')
    adata_ = adata[~ adata.obs.divergence.isna(), :]

    if color is not None:
        color = [color] if type(color) == str else color
        color_.extend(color)

    scatters(adata_, color=color_, cmap=cmap, *args, **kwargs)


@docstrings.with_indent(4)
def jacobian(adata,
             basis="umap",
             x=0,
             y=1,
             highlights=None,
             cmap='bwr',
             background=None,
             pointsize=None,
             figsize=(7, 5),
             show_legend=True,
             save_show_or_return="show",
             save_kwargs={},
             **kwargs):
    """\
    Scatter plot with pca basis.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object with Jacobian matrix estimated.
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

    if background is not None:
        set_figure_params(background=background)
    else:
        _background = rcParams.get("figure.facecolor")
        background = to_hex(_background) if type(_background) is tuple else _background

    Jacobian_ = "jacobian" #f basis is None else "jacobian_" + basis
    Der, source_gene, target_gene, cell_indx, _  =  adata.uns[Jacobian_].values()
    adata_ = adata[cell_indx, :]

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
    scatter_kwargs = dict(
        alpha=0.2, s=point_size, edgecolor=None, linewidth=0
    )  # (0, 0, 0, 1)
    if kwargs is not None:
        scatter_kwargs.update(kwargs)

    nrow, ncol = len(source_gene), len(target_gene)
    # if figsize is None:
    #     g = plt.figure(None, (3 * ncol, 3 * nrow))  # , dpi=160
    # else:
    #     g = plt.figure(None, (figsize[0] * ncol, figsize[1] * nrow))  # , dpi=160

    gs = plt.GridSpec(nrow, ncol)

    for i, source in enumerate(source_gene):
        for j, target in enumerate(target_gene):
            ax = plt.subplot(gs[i * j + j])
            J = Der[i, j, :]
            cur_pd["jacobian"] = J

            ax, color = _matplotlib_points(
                cur_pd.iloc[:, [0, 1]].values,
                ax=ax,
                labels=None,
                values=cur_pd.loc[:, "jacobian"].values,
                highlights=highlights,
                cmap=cmap,
                color_key=None,
                color_key_cmap=None,
                background=background,
                width=figsize[0],
                height=figsize[1],
                show_legend=show_legend,
                **scatter_kwargs
            )
            ax.title('Jacobian for %s wrt. %s' % (source, target))

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'phase_portraits', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return gs

