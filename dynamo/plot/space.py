import numpy as np
from typing import Union
import anndata
from .scatters import (
    scatters,
    docstrings,
)

from ..tl import compute_smallest_distance
from ..dynamo_logger import main_critical, main_info, main_finish_progress, main_log_time

docstrings.delete_params("scatters.parameters", "adata", "basis", "figsize")


@docstrings.with_indent(4)
def space(
    adata: anndata.AnnData,
    genes: Union[list, None] = None,
    space: str = "spatial",
    width: float = 6,
    marker: str = ".",
    pointsize: Union[float, None] = None,
    dpi: int = 100,
    ps_sample_num: int = 1000,
    alpha: float = 0.8,
    stack_genes: bool = False,
    stack_genes_threshold: float = 0.01,
    figsize=None,
    *args,
    **kwargs
):
    """\
    Scatter plot for physical coordinates of each cell.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object that contain the physical coordinates for each bin/cell, etc.
        genes:
            The gene list that will be used to plot the gene expression on the same scatter plot. Each gene will have a
            different color.
        space: `str`
            The key to space coordinates.
        width: `int`
            an Annodata object.
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
        alpha: `float`
            The alpha value of the scatter points.
        %(scatters.parameters.no_adata|basis|figsize)s

    Returns
    -------
        plots gene or cell feature of the adata object on the physical spatial coordinates.
    """
    main_info("Plotting spatial info on adata")
    main_log_time()
    if genes is None or (len(genes) == 0):
        main_critical("No genes provided. Please check your argument passed in.")
        return
    if "X_" + space in adata.obsm_keys():
        space_key = space
    elif space in adata.obsm_keys():
        if space.startswith("X_"):
            space_key = space.split("X_")[1]
        else:
            # scatters currently will append "X_" to the basis, so we need to create the `X_{space}` key.
            # In future, extend scatters to directly plot coordinates in space key without append "X_"
            if "X_" + space not in adata.obsm_keys():
                adata.obsm["X_" + space] = adata.obsm[space]
                space_key = space

    ptp_vec = adata.obsm["X_" + space_key].ptp(0)
    # calculate the figure size based on the width and the ratio between width and height
    # from the physical coordinate.
    if figsize is None:
        figsize = (width, ptp_vec[1] / ptp_vec[0] * width + 0.3)

    # calculate point size based on minimum radius
    if pointsize is None:
        pointsize = compute_smallest_distance(adata.obsm["X_" + space_key], sample_num=ps_sample_num)
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
        *args,
        **kwargs,
    )

    main_finish_progress("space plot")
    return res
