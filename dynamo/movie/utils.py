from typing import Optional, Union

import numpy as np


def remove_particles(
    pts: list,
    xlim: Union[tuple, list],
    ylim: Union[tuple, list],
    zlim: Optional[Union[tuple, list]] = None,
):
    if len(pts) == 0:
        return []
    outside_xlim = (pts[:, 0] < xlim[0]) | (pts[:, 0] > xlim[1])
    outside_ylim = (pts[:, 1] < ylim[0]) | (pts[:, 1] > ylim[1])
    outside_zlim = np.full(outside_xlim.shape, False) if zlim is None else (pts[:, 2] < ylim[0]) | (pts[:, 2] > ylim[1])
    keep = ~(outside_xlim | outside_ylim | outside_zlim)
    return pts[keep]
