from typing import Union

import anndata
import numpy as np

from ..dynamo_logger import (
    main_critical,
    main_finish_progress,
    main_info,
    main_log_time,
    main_warning,
)
from ..tl import compute_smallest_distance
from .scatters import docstrings, scatters

docstrings.delete_params("scatters.parameters", "adata", "basis", "figsize")


@docstrings.with_indent(4)
def space(
    adata: anndata.AnnData,
    color: Union[list, str, None] = None,
    genes: Union[list, None] = [],
    gene_cmaps=None,
    space_key: str = "spatial",
    width: float = 6,
    marker: str = ".",
    pointsize: Union[float, None] = None,
    dpi: int = 100,
    ps_sample_num: int = 1000,
    alpha: float = 0.8,
    stack_genes: bool = False,
    stack_genes_threshold: float = 0.01,
    stack_colors_legend_size: int = 10,
    figsize=None,
    *args,
    **kwargs
):
    """\
    Scatter plot for physical coordinates of each cell.

    Parameters
    ----------
        adata:
            an Annodata object that contain the physical coordinates for each bin/cell, etc.
        genes:
            The gene list that will be used to plot the gene expression on the same scatter plot. Each gene will have a
            different color. Can be a single gene name string and we will convert it to a list.
        gene_cmaps:
            A list of cmaps for mapping each gene's values according to a type of cmap when stacking gene colors on the same subplot. The order of each gene's cmap corresponds to the order in genes.
        color: `string` (default: `ntr`)
            Any or any list of column names or gene names, etc. that will be used for coloring cells. If `color` is not None, stack_genes will be disabled automatically because `color` can contain non numerical values.
        space_key: `str`
            The key to space coordinates.
        stack_genes:
            whether to show all gene plots on the same plot
        stack_genes_threshold:
            lower bound of gene values that will be drawn on the plot.
        stack_colors_legend_size:
            control the size of legend when stacking genes
        alpha: `float`
            The alpha value of the scatter points.
        width: `int`
        marker:
            a string representing some marker from matplotlib
            https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
        pointsize: `float`
            The size of the points on the scatter plot.
        dpi: `float`, (default: 100.0)
            The resolution of the figure in dots-per-inch. Dots per inches (dpi) determines how many pixels the figure
            comprises. dpi is different from ppi or points per inches. Note that most elements like lines, markers,
            texts have a size given in points so you can convert the points to inches. Matplotlib figures use Points per
            inch (ppi) of 72. A line with thickness 1 point will be 1./72. inch wide. A text with fontsize 12 points
            will be 12./72. inch heigh. Of course if you change the figure size in inches, points will not change, so a
            larger figure in inches still has the same size of the elements.Changing the figure size is thus like taking
            a piece of paper of a different size. Doing so, would of course not change the width of the line drawn with
            the same pen. On the other hand, changing the dpi scales those elements. At 72 dpi, a line of 1 point size
            is one pixel strong. At 144 dpi, this line is 2 pixels strong. A larger dpi will therefore act like a
            magnifying glass. All elements are scaled by the magnifying power of the lens. see more details at answer 2
            by @ImportanceOfBeingErnest:
            https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size
        ps_sample_num: `int`
            The number of bins / cells that will be sampled to estimate the distance between different bin / cells.

        %(scatters.parameters.no_adata|basis|figsize)s

    Returns
    -------
        plots gene or cell feature of the adata object on the physical spatial coordinates.
    """
    main_info("Plotting spatial info on adata")
    main_log_time()
    if color is not None and stack_genes:
        main_warning(
            "Set `stack_genes` to False because `color` argument cannot be used with stack_genes. If you would like to stack genes (or other numeical values), please pass gene expression like column names into `gene` argument."
        )
        stack_genes = False

    genes = [genes] if type(genes) is str else list(genes)
    # concatenate genes and colors for scatters plot
    if color is not None and genes is not None:
        color = [color] if type(color) is str else list(color)
        genes.extend(color)

    show_colorbar = True
    if stack_genes:
        main_warning("disable side colorbar due to colorbar scale (numeric tick) related issue.")
        show_colorbar = False

    if genes is None or (len(genes) == 0):
        if color is not None:
            genes = color
        else:
            main_critical("No genes provided. Please check your argument passed in.")
            return
    ptp_vec = adata.obsm[space_key].ptp(0)
    # calculate the figure size based on the width and the ratio between width and height
    # from the physical coordinate.
    if figsize is None:
        figsize = (width, ptp_vec[1] / ptp_vec[0] * width + 0.3)

    # calculate point size based on minimum radius
    if pointsize is None:
        pointsize = compute_smallest_distance(adata.obsm[space_key], sample_num=ps_sample_num)
        # here we will scale the point size by the dpi and the figure size in inch.
        pointsize *= figsize[0] / ptp_vec[0] * dpi
        # meaning of s in scatters:
        # https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size/47403507#47403507
        # Note that np.sqrt(adata.shape[0]) / 16000.0 is used in pl.scatters
        pointsize = pointsize ** 2 * np.sqrt(adata.shape[0]) / 16000.0

        main_info("estimated point size for plotting each cell in space: %f" % (pointsize))

    # here we should pass different point size, type (square or hexogon, etc), etc.
    res = scatters(
        adata,
        marker=marker,
        basis=space_key,
        color=genes,
        figsize=figsize,
        pointsize=pointsize,
        dpi=dpi,
        alpha=alpha,
        stack_colors=stack_genes,
        stack_colors_threshold=stack_genes_threshold,
        stack_colors_title="stacked spatial genes",
        show_colorbar=show_colorbar,
        stack_colors_legend_size=stack_colors_legend_size,
        stack_colors_cmaps=gene_cmaps,
        *args,
        **kwargs,
    )

    main_finish_progress("space plot")
    return res
