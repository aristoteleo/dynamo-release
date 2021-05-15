import matplotlib
import numpy as np
from .scatters import (
    scatters,
    docstrings,
)

from ..tl import compute_smallest_distance
from ..dynamo_logger import *

docstrings.delete_params("scatters.parameters", "adata", "basis", "figsize")


@docstrings.with_indent(4)
def space(
    adata,
    genes="all",
    space="spatial",
    width=6,
    marker="p",
    pointsize=None,
    pointsize_estimation_num=1000,
    *args,
    **kwargs
):
    """\
    Scatter plot for physical coordinates of each cell.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        space: `str`
            The key to space coordinates.
        width: `int`
            an Annodata object.
        %(scatters.parameters.no_adata|basis|figsize)s
        marker:
            a string representing some marker from matplotlib
            https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers

    Returns
    -------
        plots gene or cell feature of the adata object on the physical spatial coordinates.
    """
    main_info("Plotting spatial info on adata")
    main_log_time()
    if space in adata.obsm_keys():
        space_key = space
    elif "X_" + space in adata.obsm_keys():
        space_key = "X_" + space

    ptp_vec = adata.obsm[space_key].ptp(0)
    # calculate the figure size based on the width and the ratio between width and height
    # from the physical coordinate.
    figsize = (width, ptp_vec[1] / ptp_vec[0] * width)

    # calculate point size based on minimum radius
    if pointsize is None:
        selected_estimation_indices = np.random.choice(
            len(adata), size=min(len(adata), pointsize_estimation_num), replace=False
        )
        pointsize = compute_smallest_distance(adata.obsm[space_key][selected_estimation_indices, :])
        main_info("estimated point size for plotting each cell in space: %f" % (pointsize))
    main_finish_progress("space plot")
    # here we should pass different point size, type (square or hexogon, etc), etc.
    return scatters(
        adata,
        marker=marker,
        basis=space_key,
        figsize=figsize,
        pointsize=pointsize,
        *args,
        **kwargs,
    )
