import matplotlib
from .scatters import (
    scatters,
    docstrings,
)

from ..tl import compute_smallest_distance

docstrings.delete_params("scatters.parameters", "adata", "basis", "figsize")


@docstrings.with_indent(4)
def space(adata, space="spatial", width=6, marker="p", *args, **kwargs):
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

    if space in adata.obsm_keys():
        space_key = space
    elif "X_" + space in adata.obsm_keys():
        space_key = "X_" + space

    ptp_vec = adata.obsm[space_key].ptp(0)
    # calculate the figure size based on the width and the ratio between width and height
    # from the physical coordinate.
    figsize = (width, ptp_vec[1] / ptp_vec[0] * width)

    # calculate point size based on minimum radius
    pointsize = compute_smallest_distance(adata[space_key])

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
