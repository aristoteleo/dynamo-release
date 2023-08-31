from typing import Optional, Union

import numpy as np


def remove_particles(
    pts: np.ndarray,
    xlim: Union[tuple, list],
    ylim: Union[tuple, list],
    zlim: Optional[Union[tuple, list]] = None,
) -> np.ndarray:
    """Remove particles that fall outside specified coordinate ranges.

    Args:
        pts: an array of points.
        xlim: X-coordinate limits specified as a tuple or list of two values: (min_x, max_x).
        ylim: Y-coordinate limits specified as a tuple or list of two values: (min_y, max_y).
        zlim: Z-coordinate limits specified as a tuple or list of two values: (min_z, max_z). If not provided (default),
            only 2D filtering based on xlim and ylim is performed.

    Returns:
        An array of points that fall within the specified coordinate ranges.
    """
    if len(pts) == 0:
        return []
    outside_xlim = (pts[:, 0] < xlim[0]) | (pts[:, 0] > xlim[1])
    outside_ylim = (pts[:, 1] < ylim[0]) | (pts[:, 1] > ylim[1])
    outside_zlim = np.full(outside_xlim.shape, False) if zlim is None else (pts[:, 2] < ylim[0]) | (pts[:, 2] > ylim[1])
    keep = ~(outside_xlim | outside_ylim | outside_zlim)
    return pts[keep]
