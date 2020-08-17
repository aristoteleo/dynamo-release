# code adapted from answer 2 by e-malito
# https://stackoverflow.com/questions/43150872/number-of-arrowheads-on-matplotlib-streamplot

import numpy as np


def curve_coord(line=None):
    """ return curvilinear coordinate """
    x = line[:, 0]
    y = line[:, 1]
    s = np.zeros(x.shape)
    s[1:] = np.sqrt((x[1:] - x[0:-1]) ** 2 + (y[1:] - y[0:-1]) ** 2)
    s = np.cumsum(s)
    return s


def curve_extract(line, spacing, offset=None):
    """ Extract points at equidistant space along a curve"""
    x = line[:, 0]
    y = line[:, 1]
    if offset is None:
        offset = spacing / 2
    # Computing curvilinear length
    s = curve_coord(line)
    offset = np.mod(offset, s[-1])  # making sure we always get one point
    # New (equidistant) curvilinear coordinate
    sExtract = np.arange(offset, s[-1], spacing)
    # Interpolating based on new curvilinear coordinate
    xx = np.interp(sExtract, s, x);
    yy = np.interp(sExtract, s, y);
    return np.array([xx, yy]).T


def seg_to_lines(seg):
    """ Convert a list of segments to a list of lines """

    def extract_continuous(i):
        x = []
        y = []
        # Special case, we have only 1 segment remaining:
        if i == len(seg) - 1:
            x.append(seg[i][0, 0])
            y.append(seg[i][0, 1])
            x.append(seg[i][1, 0])
            y.append(seg[i][1, 1])
            return i, x, y
        # Looping on continuous segment
        while i < len(seg) - 1:
            # Adding our start point
            x.append(seg[i][0, 0])
            y.append(seg[i][0, 1])
            # Checking whether next segment continues our line
            Continuous = all(seg[i][1, :] == seg[i + 1][0, :])
            if not Continuous:
                # We add our end point then
                x.append(seg[i][1, 0])
                y.append(seg[i][1, 1])
                break
            elif i == len(seg) - 2:
                # we add the last segment
                x.append(seg[i + 1][0, 0])
                y.append(seg[i + 1][0, 1])
                x.append(seg[i + 1][1, 0])
                y.append(seg[i + 1][1, 1])
            i = i + 1
        return i, x, y

    lines = []
    i = 0
    while i < len(seg):
        iEnd, x, y = extract_continuous(i)
        lines.append(np.array([x, y]).T)
        i = iEnd + 1
    return lines


def lines_to_arrows(lines, n=5, spacing=None, normalize=True):
    """ Extract "streamlines" arrows from a set of lines
    Either: `n` arrows per line
        or an arrow every `spacing` distance
    If `normalize` is true, the arrows have a unit length
    """
    if spacing is None:
        # if n is provided we estimate the spacing based on each curve lenght)
        spacing = [curve_coord(l)[-1] / n for l in lines]
    try:
        len(spacing)
    except:
        spacing = [spacing] * len(lines)

    lines_s = [curve_extract(l, spacing=sp, offset=sp / 2) for l, sp in zip(lines, spacing)]
    lines_e = [curve_extract(l, spacing=sp, offset=sp / 2 + 0.01 * sp) for l, sp in zip(lines, spacing)]
    arrow_x = [l[i, 0] for l in lines_s for i in range(len(l))]
    arrow_y = [l[i, 1] for l in lines_s for i in range(len(l))]
    arrow_dx = [le[i, 0] - ls[i, 0] for ls, le in zip(lines_s, lines_e) for i in range(len(ls))]
    arrow_dy = [le[i, 1] - ls[i, 1] for ls, le in zip(lines_s, lines_e) for i in range(len(ls))]

    if normalize:
        dn = [np.sqrt(ddx ** 2 + ddy ** 2) for ddx, ddy in zip(arrow_dx, arrow_dy)]
        arrow_dx = [ddx / ddn for ddx, ddn in zip(arrow_dx, dn)]
        arrow_dy = [ddy / ddn for ddy, ddn in zip(arrow_dy, dn)]
    return arrow_x, arrow_y, arrow_dx, arrow_dy


if __name__ == '__main__':
    # --- Main body of streamQuiver
    # Extracting lines
    import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # sp = ax.streamplot(Grid[:, 0].reshape((50, 50)),
    #                    Grid[:, 1].reshape((50, 50)),
    #                    VF[:, 0].reshape((50, 50)),
    #                    VF[:, 1].reshape((50, 50)), arrowstyle='-', density=10)
    #
    # streamQuiver(ax, sp, n=3)
