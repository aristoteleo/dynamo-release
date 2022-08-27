from typing import Union


def remove_particles(pts: list, xlim: Union[tuple, list], ylim: Union[tuple, list]):
    if len(pts) == 0:
        return []
    outside_xlim = (pts[:, 0] < xlim[0]) | (pts[:, 0] > xlim[1])
    outside_ylim = (pts[:, 1] < ylim[0]) | (pts[:, 1] > ylim[1])
    keep = ~(outside_xlim | outside_ylim)
    return pts[keep]
