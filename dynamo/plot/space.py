from typing import List, Optional, Tuple, Union

import anndata
import numpy as np
from matplotlib.axes import Axes

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
    color: Union[List[str], str, None] = None,
    genes: Optional[List[str]] = [],
    gene_cmaps: Optional[List[str]] = None,
    space_key: str = "spatial",
    width: float = 6,
    marker: str = ".",
    pointsize: Optional[float] = None,
    dpi: int = 100,
    ps_sample_num: int = 1000,
    alpha: float = 0.8,
    stack_genes: bool = False,
    stack_genes_threshold: float = 0.01,
    stack_colors_legend_size: int = 10,
    figsize: Tuple[float, float] = None,
    *args,
    **kwargs
) -> Union[Axes, List[Axes]]:
    """Scatter plot for physical coordinates of each cell.

    Args:
        adata: an Annodata object that contain the physical coordinates for each bin/cell, etc.
        color: any or any list of column names or gene names, etc. that will be used for coloring cells. If `color` is
            not None, stack_genes will be disabled automatically because `color` can contain non numerical values.
            Defaults to None.
        genes: the gene list that will be used to plot the gene expression on the same scatter plot. Each gene will have
            a different color. Can be a single gene name string and we will convert it to a list. Defaults to [].
        gene_cmaps: a list of cmaps for mapping each gene's values according to a type of cmap when stacking gene colors
            on the same subplot. The order of each gene's cmap corresponds to the order in genes. Defaults to None.
        space_key: the key to space coordinates. Defaults to "spatial".
        width: the width of the figure. Would be used when `figsize` is not specified. Defaults to 6.
        marker: a string representing some marker from matplotlib
            https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers. Defaults to ".".
        pointsize: the size of the points on the scatter plot. Defaults to None.
        dpi: the resolution of the figure in dots-per-inch. Dots per inches (dpi) determines how many pixels the figure
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
            https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size. Defaults to 100.
        ps_sample_num: the number of bins / cells that will be sampled to estimate the distance between different
            bin / cells. Defaults to 1000.
        alpha: the point's alpha (transparency) value. Defaults to 0.8.
        stack_genes: whether to stack all genes on the same ax passed above. Defaults to False.
        stack_genes_threshold: a threshold for filtering out points values < threshold when drawing each gene. Defaults
            to 0.01.
        stack_colors_legend_size: the legend size in stack gene plot. Defaults to 10.
        figsize: the size of each subplot. Defaults to None.

    Returns:
        The matplotlib axes of the generated subplots.
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
        pointsize = pointsize**2 * np.sqrt(adata.shape[0]) / 16000.0

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
