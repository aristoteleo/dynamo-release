from .scatters import (
    scatters,
    docstrings,
)

docstrings.delete_params("scatters.parameters", "adata", "basis", "figsize")


@docstrings.with_indent(4)
def space(adata, space="spatial", width=6, *args, **kwargs):
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

    Returns
    -------
        plots gene or cell feature of the adata object on the physical spatial coordinates.
    """

    if space in adata.obsm_keys():
        space_key = "X_" + space
    elif "X_" + space in adata.obsm_keys():
        space_key = space

    ptp_vec = adata.obsm[space_key].ptp(0)
    # calculate the figure size based on the width and the ratio between width and height
    # from the physical coordinate.
    figsize = (width, ptp_vec[1] / ptp_vec[0] * width)

    # here we should pass different point size, type (square or hexogon, etc), etc.
    return scatters(
        adata,
        basis=space_key,
        figsize=figsize,
        *args,
        **kwargs,
    )
