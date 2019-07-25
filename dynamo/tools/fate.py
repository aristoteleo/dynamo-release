"""
Functions to predict the history and future of cell states.
"""
from typing import List, Any, Tuple

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal
# from matplotlib import rcParams
# import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np

# __all__ = ['fate'] # what is this?

# can we vectorize the function so that we can do the analysis for multiple genes and cells at once
# integrate this method with the method below (fate)
def fate_(adata, time, direction = 'forward'):
    from .moments import *
    gene_exprs = adata.X
    cell_num, gene_num = gene_exprs.shape


    for i in range(gene_num):
        params = {'a': adata.uns['dynamo'][i, "a"], \
                  'b': adata.uns['dynamo'][i, "b"], \
                  'la': adata.uns['dynamo'][i, "la"], \
                  'alpha_a': adata.uns['dynamo'][i, "alpha_a"], \
                  'alpha_i': adata.uns['dynamo'][i, "alpha_i"], \
                  'sigma': adata.uns['dynamo'][i, "sigma"], \
                  'beta': adata.uns['dynamo'][i, "beta"], \
                  'gamma': adata.uns['dynamo'][i, "gamma"]}
        mom = moments_simple(**params)
        for j in range(cell_num):
            x0 = gene_exprs[i, j]
            mom.set_initial_condition(*x0)
            if direction == "forward":
                gene_exprs[i, j] = mom.solve([0, time])
            elif direction == "backward":
                gene_exprs[i, j] = mom.solve([0, - time])

    adata.uns['prediction'] = gene_exprs
    return adata

# support forward / backward time, plot velocity as color on the curve, calculate time traveled
# is it possible to predict the history? -- seems like you can do it with steady states
# scan the entire space and then assign the space to different steady states -- this gives us basin of attraction
#
def fate(adata, dim_type = 'umap', xy_grid_nums=(50, 50), density=1, linewidth=None, color=None,
         cmap=None, norm=None, arrowsize=1, arrowstyle='-|>',
         minlength=0.1, transform=None, zorder=None, start_cells = None, start_points=None,
         maxlength=4.0, integration_direction='both', show_plot=True, size_x = 8, size_y = 8):
    """
    Predict history and future of cells based on measured vector field. This function builds upon the streamplot from matplotlib.
    Parameters
    ----------
    dim_type : str
        The result of dimension reduction used for embedding both of the current expression and associated velocity for each cell
    xy_grid_nums : list of integers
        Number of grids on x or y axes to evenly separate the state space.
    density : float or 2-tuple
        Controls the closeness of streamlines. When ``density = 1``, the domain
        is divided into a 30x30 grid---*density* linearly scales this grid.
        Each cell in the grid can have, at most, one traversing streamline.
        For different densities in each direction, use [density_x, density_y].
    linewidth : numeric or 2d array
        Vary linewidth when given a 2d array with the same shape as velocities.
    color : matplotlib color code, or 2d array
        Streamline color. When given an array with the same shape as
        velocities, *color* values are converted to colors using *cmap*.
    cmap : `~matplotlib.colors.Colormap`
        Colormap used to plot streamlines and arrows. Only necessary when using
        an array input for *color*.
    norm : `~matplotlib.colors.Normalize`
        Normalize object used to scale luminance data to 0, 1. If ``None``,
        stretch (min, max) to (0, 1). Only necessary when *color* is an array.
    arrowsize : float
        Factor scale arrow size.
    arrowstyle : str
        Arrow style specification.
        See `~matplotlib.patches.FancyArrowPatch`.
    minlength : float
        Minimum length of streamline in axes coordinates.
    start_points : Nx2 array
        Coordinates of starting points for the streamlines.
        In data coordinates, the same as the *x* and *y* arrays.
    zorder : int
        Any number.
    maxlength : float
        Maximum length of streamline in axes coordinates.
    integration_direction : ['future' | 'history' | 'both']
        Integrate the streamline in future, history or both directions.
        default is ``'both'``.
    show_plot : logic
        A logic flag to determine whether the streamline will be drawn
    Returns -----> update
    -------
    stream_container : StreamplotSet
        Container object with attributes
        - lines: `matplotlib.collections.LineCollection` of streamlines
        - arrows: collection of `matplotlib.patches.FancyArrowPatch`
          objects representing arrows half-way along stream
          lines.
        This container will probably change in the future to allow changes
        to the colormap, alpha, etc. for both lines and arrows, but these
        changes should be history compatible.
    """

    vlm_embedding, delta_embedding = adata.obsm['X_' + dim_type], adata.obsm['velocity_' + dim_type]  # adata.layers['velocity'].shape
    x_grid, v_grid, d_mat = compute_velocity_on_grid(vlm_embedding, delta_embedding, xy_grid_nums)
    x, y, u, v = x_grid[0], x_grid[1], v_grid[0], v_grid[1]

    if start_cells is not None:
        start_points = adata.obsm['X_umap'][adata.obs.index.isin(start_cells)]
        nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        grs = []
        for dim_i in range(2):
            m, M = np.min(vlm_embedding[:, dim_i]), np.max(vlm_embedding[:, dim_i])
            m = m - .01 * np.abs(M - m)
            M = M + .01 * np.abs(M - m)
            gr = np.linspace(m, M, xy_grid_nums[dim_i] * 1)
            grs.append(gr)

        meshes_tuple = np.meshgrid(*grs)
        x_grid2 = np.vstack([i.flat for i in meshes_tuple]).T

        nn.fit(x_grid2)
        _, neighs = nn.kneighbors(start_points)

        start_points = x_grid2[np.unique(neighs.squeeze())]

    grid = Grid(x, y)
    mask = StreamMask(density)
    dmap = DomainMap(grid, mask)

    if show_plot:
        _, axes = plt.subplots(figsize = (size_x, size_y))

        if zorder is None:
            zorder = mlines.Line2D.zorder

        # default to data coordinates
        if transform is None:
            transform = axes.transData

        if color is None:
            color = axes._get_lines.get_next_color()

        speed = np.sqrt(u*u + v*v)
        linewidth = 5 * (speed + 0.025 * speed.max()) / speed.max() #(speed - np.min(speed) + 0.1 * np.max(speed)) / (np.max(speed) - np.min(speed))
        print('range is ', np.max(linewidth), '--', np.min(linewidth))
        print('speed is ', np.max(speed), '--', np.min(speed))
        if linewidth is None:
            linewidth = matplotlib.rcParams['lines.linewidth']

        line_kw = {}
        arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)

    if integration_direction not in ['both', 'future', 'history']:
        errstr = ("Integration direction '%s' not recognised. "
                  "Expected 'both', 'future' or 'history'." %
                  integration_direction)
        raise ValueError(errstr)

    if integration_direction == 'both':
        maxlength /= 2.

    if show_plot:
        use_multicolor_lines = isinstance(color, np.ndarray)
        if use_multicolor_lines:
            if color.shape != grid.shape:
                raise ValueError(
                    "If 'color' is given, must have the shape of 'Grid(x,y)'")
            line_colors = []
            color = np.ma.masked_invalid(color)
        else:
            line_kw['color'] = color
            arrow_kw['color'] = color

        if isinstance(linewidth, np.ndarray):
            if linewidth.shape != grid.shape:
                raise ValueError(
                    "If 'linewidth' is given, must have the shape of 'Grid(x,y)'")
            line_kw['linewidth'] = []
        else:
            line_kw['linewidth'] = linewidth
            arrow_kw['linewidth'] = linewidth

        line_kw['zorder'] = zorder
        arrow_kw['zorder'] = zorder

    ## Sanity checks.
    if u.shape != grid.shape or v.shape != grid.shape:
        raise ValueError("'u' and 'v' must be of shape 'Grid(x,y)'")

    u = np.ma.masked_invalid(u)
    v = np.ma.masked_invalid(v)

    integrate = get_integrator(u, v, dmap, minlength, maxlength,
                               integration_direction)

    trajectories: List[Tuple[List[Any], List[Any]]] = [] # complete the type definition
    if start_points is None:
        for xm, ym in _gen_starting_points(mask.shape):
            if mask[ym, xm] == 0:
                xg, yg = dmap.mask2grid(xm, ym)
                t = integrate(xg, yg)
                if t is not None:
                    trajectories.append(t)
    else:
        sp2 = np.asanyarray(start_points, dtype=float).copy()

        # Check if start_points are outside the data boundaries
        for xs, ys in sp2:
            if not (grid.x_origin <= xs <= grid.x_origin + grid.width
                    and grid.y_origin <= ys <= grid.y_origin + grid.height):
                raise ValueError("Starting point ({}, {}) outside of data "
                                 "boundaries".format(xs, ys))

        # Convert start_points from data to array coords
        # Shift the seed points from the bottom left of the data so that
        # data2grid works properly.
        sp2[:, 0] -= grid.x_origin
        sp2[:, 1] -= grid.y_origin

        valid_start_points = [False] * len(sp2)
        i = 0
        for xs, ys in sp2:
            xg, yg = dmap.data2grid(xs, ys)
            t = integrate(xg, yg)
            if t is not None:
                trajectories.append(t)
                valid_start_points[i] = True

            i += 1

        print('Total failed integration number is ', len(sp2) - len(trajectories))
        print('Total successful trajectory number is ', len(trajectories))

    if show_plot:
        clusters = list(map(int, adata.obs.louvain.tolist()))
        clusters = clusters / np.max(clusters)
        clusters_c = cm.get_cmap('viridis', len(set(clusters)))

        color_dim = clusters_c(clusters)
        color_dim[:, 3] = 0.1

        axes.scatter(vlm_embedding[:, 0], vlm_embedding[:, 1], s = 5, c = color_dim)

        if start_points is not None:
            axes.plot(start_points[valid_start_points, 0], start_points[valid_start_points, 1], 'bo')

        axes.quiver(x_grid[0], x_grid[1], v_grid[0], v_grid[1])

        if use_multicolor_lines:
            if norm is None:
                norm = mcolors.Normalize(color.min(), color.max())
            if cmap is None:
                cmap = cm.get_cmap(matplotlib.rcParams['image.cmap'])
            else:
                cmap = cm.get_cmap(cmap)

        streamlines = []
        arrows = []

        for t in trajectories:
            tgx = np.array(t[0])
            tgy = np.array(t[1])
            # Rescale from grid-coordinates to data-coordinates.
            tx, ty = dmap.grid2data(*np.array(t))
            tx += grid.x_origin
            ty += grid.y_origin

            points = np.transpose([tx, ty]).reshape(-1, 1, 2)
            streamlines.extend(np.hstack([points[:-1], points[1:]]))

            # Add arrows half way along each trajectory.
            s = np.cumsum(np.hypot(np.diff(tx), np.diff(ty)))
            n = np.searchsorted(s, s[-1] / 2.)
            arrow_tail = (tx[n], ty[n])
            arrow_head = (np.mean(tx[n:n + 2]), np.mean(ty[n:n + 2]))

            if isinstance(linewidth, np.ndarray):
                line_widths = interpgrid(linewidth, tgx, tgy)[:-1]
                line_kw['linewidth'].extend(line_widths)
                arrow_kw['linewidth'] = line_widths[n]

            if use_multicolor_lines:
                color_values = interpgrid(color, tgx, tgy)[:-1]
                line_colors.append(color_values)
                arrow_kw['color'] = cmap(norm(color_values[n]))

            p = patches.FancyArrowPatch(
                arrow_tail, arrow_head, transform=transform, **arrow_kw)
            axes.add_patch(p)
            arrows.append(p)

        lc = mcollections.LineCollection(
            streamlines, transform=transform, **line_kw)
        lc.sticky_edges.x[:] = [grid.x_origin, grid.x_origin + grid.width]
        lc.sticky_edges.y[:] = [grid.y_origin, grid.y_origin + grid.height]
        if use_multicolor_lines:
            lc.set_array(np.ma.hstack(line_colors))
            lc.set_cmap(cmap)
            lc.set_norm(norm)
        axes.add_collection(lc)
        axes.autoscale_view()

        ac = matplotlib.collections.PatchCollection(arrows)
        stream_container = StreamplotSet(lc, ac)
        return stream_container, trajectories
    else:
        return trajectories


class StreamplotSet(object):

    def __init__(self, lines, arrows, **kwargs):
        self.lines = lines
        self.arrows = arrows


# Coordinate definitions
# ========================

class DomainMap(object):
    """Map representing different coordinate systems.
    Coordinate definitions:
    * axes-coordinates goes from 0 to 1 in the domain.
    * data-coordinates are specified by the input x-y coordinates.
    * grid-coordinates goes from 0 to N and 0 to M for an N x M grid,
      where N and M match the shape of the input data.
    * mask-coordinates goes from 0 to N and 0 to M for an N x M mask,
      where N and M are user-specified to control the density of streamlines.
    This class also has methods for adding trajectories to the StreamMask.
    Before adding a trajectory, run `start_trajectory` to keep track of regions
    crossed by a given trajectory. Later, if you decide the trajectory is bad
    (e.g., if the trajectory is very short) just call `undo_trajectory`.
    """

    def __init__(self, grid, mask):
        self.grid = grid
        self.mask = mask
        # Constants for conversion between grid- and mask-coordinates
        self.x_grid2mask = (mask.nx - 1) / grid.nx
        self.y_grid2mask = (mask.ny - 1) / grid.ny

        self.x_mask2grid = 1. / self.x_grid2mask
        self.y_mask2grid = 1. / self.y_grid2mask

        self.x_data2grid = 1. / grid.dx
        self.y_data2grid = 1. / grid.dy

    def grid2mask(self, xi, yi):
        """Return nearest space in mask-coords from given grid-coords."""
        return (int(xi * self.x_grid2mask + 0.5),
                int(yi * self.y_grid2mask + 0.5))

    def mask2grid(self, xm, ym):
        return xm * self.x_mask2grid, ym * self.y_mask2grid

    def data2grid(self, xd, yd):
        return xd * self.x_data2grid, yd * self.y_data2grid

    def grid2data(self, xg, yg):
        return xg / self.x_data2grid, yg / self.y_data2grid

    def start_trajectory(self, xg, yg):
        xm, ym = self.grid2mask(xg, yg)
        self.mask._start_trajectory(xm, ym)

    def reset_start_point(self, xg, yg):
        xm, ym = self.grid2mask(xg, yg)
        self.mask._current_xy = (xm, ym)

    def update_trajectory(self, xg, yg):
        if not self.grid.within_grid(xg, yg):
            raise InvalidIndexError
        xm, ym = self.grid2mask(xg, yg)
        self.mask._update_trajectory(xm, ym)

    def undo_trajectory(self):
        self.mask._undo_trajectory()


class Grid(object):
    """Grid of data."""

    def __init__(self, x, y):

        if x.ndim == 1:
            pass
        elif x.ndim == 2:
            x_row = x[0, :]
            if not np.allclose(x_row, x):
                raise ValueError("The rows of 'x' must be equal")
            x = x_row
        else:
            raise ValueError("'x' can have at maximum 2 dimensions")

        if y.ndim == 1:
            pass
        elif y.ndim == 2:
            y_col = y[:, 0]
            if not np.allclose(y_col, y.T):
                raise ValueError("The columns of 'y' must be equal")
            y = y_col
        else:
            raise ValueError("'y' can have at maximum 2 dimensions")

        self.nx = len(x)
        self.ny = len(y)

        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]

        self.x_origin = x[0]
        self.y_origin = y[0]

        self.width = x[-1] - x[0]
        self.height = y[-1] - y[0]

        if not np.allclose(np.diff(x), self.width / (self.nx - 1)):
            raise ValueError("'x' values must be equally spaced")
        if not np.allclose(np.diff(y), self.height / (self.ny - 1)):
            raise ValueError("'y' values must be equally spaced")

    @property
    def shape(self):
        return self.ny, self.nx

    def within_grid(self, xi, yi):
        """Return True if point is a valid index of grid."""
        # Note that xi/yi can be floats; so, for example, we can't simply check
        # `xi < self.nx` since `xi` can be `self.nx - 1 < xi < self.nx`
        return xi >= 0 and xi <= self.nx - 1 and yi >= 0 and yi <= self.ny - 1


class StreamMask(object):
    """Mask to keep track of discrete regions crossed by streamlines.
    The resolution of this grid determines the approximate spacing between
    trajectories. Streamlines are only allowed to pass through zeroed cells:
    When a streamline enters a cell, that cell is set to 1, and no new
    streamlines are allowed to enter.
    """

    def __init__(self, density):
        try:
            self.nx, self.ny = (30 * np.broadcast_to(density, 2)).astype(int)
        except ValueError:
            raise ValueError("'density' must be a scalar or be of length 2")
        if self.nx < 0 or self.ny < 0:
            raise ValueError("'density' must be positive")
        self._mask = np.zeros((self.ny, self.nx))
        self.shape = self._mask.shape

        self._current_xy = None

    def __getitem__(self, *args):
        return self._mask.__getitem__(*args)

    def _start_trajectory(self, xm, ym):
        """Start recording streamline trajectory"""
        self._traj = []
        self._update_trajectory(xm, ym)

    def _undo_trajectory(self):
        """Remove current trajectory from mask"""
        for t in self._traj:
            self._mask.__setitem__(t, 0)

    def _update_trajectory(self, xm, ym):
        """Update current trajectory position in mask.
        If the new position has already been filled, raise `InvalidIndexError`.
        """
        if self._current_xy != (xm, ym):
            if self[ym, xm] == 0:
                self._traj.append((ym, xm))
                self._mask[ym, xm] = 1
                self._current_xy = (xm, ym)
            else:
                raise InvalidIndexError


class InvalidIndexError(Exception):
    pass


class TerminateTrajectory(Exception):
    pass


# Integrator definitions
# ========================

def get_integrator(u, v, dmap, minlength, maxlength, integration_direction):
    # rescale velocity onto grid-coordinates for integrations.
    u, v = dmap.data2grid(u, v)

    # speed (path length) will be in axes-coordinates
    u_ax = u / dmap.grid.nx
    v_ax = v / dmap.grid.ny
    speed = np.ma.sqrt(u_ax ** 2 + v_ax ** 2)

    def forward_time(xi, yi):
        ds_dt = interpgrid(speed, xi, yi)
        if ds_dt == 0:
            raise TerminateTrajectory()
        dt_ds = 1. / ds_dt
        ui = interpgrid(u, xi, yi)
        vi = interpgrid(v, xi, yi)
        return ui * dt_ds, vi * dt_ds

    def backward_time(xi, yi):
        dxi, dyi = forward_time(xi, yi)
        return -dxi, -dyi

    def integrate(x0, y0):
        """Return x, y grid-coordinates of trajectory based on starting point.
        Integrate both forward and backward in time from starting point in
        grid coordinates.
        Integration is terminated when a trajectory reaches a domain boundary
        or when it crosses into an already occupied cell in the StreamMask. The
        resulting trajectory is None if it is shorter than `minlength`.
        """

        stotal, x_traj, y_traj = 0., [], []

        try:
            dmap.start_trajectory(x0, y0)
        except InvalidIndexError:
            return None
        if integration_direction in ['both', 'history']:
            s, xt, yt = _integrate_rk12(x0, y0, dmap, backward_time, maxlength)
            stotal += s
            x_traj += xt[::-1]
            y_traj += yt[::-1]

        if integration_direction in ['both', 'future']:
            dmap.reset_start_point(x0, y0)
            s, xt, yt = _integrate_rk12(x0, y0, dmap, forward_time, maxlength)
            if len(x_traj) > 0:
                xt = xt[1:]
                yt = yt[1:]
            stotal += s
            x_traj += xt
            y_traj += yt

        if stotal > minlength:
            return x_traj, y_traj
        else:  # reject short trajectories
            dmap.undo_trajectory()
            return None

    return integrate


def _integrate_rk12(x0, y0, dmap, f, maxlength):
    """2nd-order Runge-Kutta algorithm with adaptive step size.
    This method is also referred to as the improved Euler's method, or Heun's
    method. This method is favored over higher-order methods because:
    1. To get decent looking trajectories and to sample every mask cell
       on the trajectory we need a small timestep, so a lower order
       solver doesn't hurt us unless the data is *very* high resolution.
       In fact, for cases where the user inputs
       data smaller or of similar grid size to the mask grid, the higher
       order corrections are negligible because of the very fast linear
       interpolation used in `interpgrid`.
    2. For high resolution input data (i.e. beyond the mask
       resolution), we must reduce the timestep. Therefore, an adaptive
       timestep is more suited to the problem as this would be very hard
       to judge automatically otherwise.
    This integrator is about 1.5 - 2x as fast as both the RK4 and RK45
    solvers in most setups on my machine. I would recommend removing the
    other two to keep things simple.
    """
    # This error is below that needed to match the RK4 integrator. It
    # is set for visual reasons -- too low and corners start
    # appearing ugly and jagged. Can be tuned.
    maxerror = 0.003

    # This limit is important (for all integrators) to avoid the
    # trajectory skipping some mask cells. We could relax this
    # condition if we use the code which is commented out below to
    # increment the location gradually. However, due to the efficient
    # nature of the interpolation, this doesn't boost speed by much
    # for quite a bit of complexity.
    maxds = min(1. / dmap.mask.nx, 1. / dmap.mask.ny, 0.1)

    ds = maxds
    stotal = 0
    xi = x0
    yi = y0
    xf_traj = []
    yf_traj = []

    while dmap.grid.within_grid(xi, yi):
        xf_traj.append(xi)
        yf_traj.append(yi)
        try:
            k1x, k1y = f(xi, yi)
            k2x, k2y = f(xi + ds * k1x,
                         yi + ds * k1y)
        except IndexError:
            # Out of the domain on one of the intermediate integration steps.
            # Take an Euler step to the boundary to improve neatness.
            ds, xf_traj, yf_traj = _euler_step(xf_traj, yf_traj, dmap, f)
            stotal += ds
            break
        except TerminateTrajectory:
            break

        dx1 = ds * k1x
        dy1 = ds * k1y
        dx2 = ds * 0.5 * (k1x + k2x)
        dy2 = ds * 0.5 * (k1y + k2y)

        nx, ny = dmap.grid.shape
        # Error is normalized to the axes coordinates
        error = np.hypot((dx2 - dx1) / nx, (dy2 - dy1) / ny)

        # Only save step if within error tolerance
        if error < maxerror:
            xi += dx2
            yi += dy2
            try:
                dmap.update_trajectory(xi, yi)
            except InvalidIndexError:
                break
            if stotal + ds > maxlength:
                break
            stotal += ds

        # recalculate stepsize based on step error
        if error == 0:
            ds = maxds
        else:
            ds = min(maxds, 0.85 * ds * (maxerror / error) ** 0.5)

    return stotal, xf_traj, yf_traj


def _euler_step(xf_traj, yf_traj, dmap, f):
    """Simple Euler integration step that extends streamline to boundary."""
    ny, nx = dmap.grid.shape
    xi = xf_traj[-1]
    yi = yf_traj[-1]
    cx, cy = f(xi, yi)
    if cx == 0:
        dsx = np.inf
    elif cx < 0:
        dsx = xi / -cx
    else:
        dsx = (nx - 1 - xi) / cx
    if cy == 0:
        dsy = np.inf
    elif cy < 0:
        dsy = yi / -cy
    else:
        dsy = (ny - 1 - yi) / cy
    ds = min(dsx, dsy)
    xf_traj.append(xi + cx * ds)
    yf_traj.append(yi + cy * ds)
    return ds, xf_traj, yf_traj


# Utility functions
# ========================

def interpgrid(a, xi, yi):
    """Fast 2D, linear interpolation on an integer grid"""

    Ny, Nx = np.shape(a)
    if isinstance(xi, np.ndarray):
        x = xi.astype(int)
        y = yi.astype(int)
        # Check that xn, yn don't exceed max index
        xn = np.clip(x + 1, 0, Nx - 1)
        yn = np.clip(y + 1, 0, Ny - 1)
    else:
        x = int(xi)
        y = int(yi)
        # conditional is faster than clipping for integers
        if x == (Nx - 1):
            xn = x
        else:
            xn = x + 1
        if y == (Ny - 1):
            yn = y
        else:
            yn = y + 1

    a00 = a[y, x]
    a01 = a[y, xn]
    a10 = a[yn, x]
    a11 = a[yn, xn]
    xt = xi - x
    yt = yi - y
    a0 = a00 * (1 - xt) + a01 * xt
    a1 = a10 * (1 - xt) + a11 * xt
    ai = a0 * (1 - yt) + a1 * yt

    if not isinstance(xi, np.ndarray):
        if np.ma.is_masked(ai):
            raise TerminateTrajectory

    return ai


def _gen_starting_points(shape):
    """Yield starting points for streamlines.
    Trying points on the boundary first gives higher quality streamlines.
    This algorithm starts with a point on the mask corner and spirals inward.
    This algorithm is inefficient, but fast compared to rest of streamplot.
    """
    ny, nx = shape
    xfirst = 0
    yfirst = 1
    xlast = nx - 1
    ylast = ny - 1
    x, y = 0, 0
    direction = 'right'
    for i in range(nx * ny):

        yield x, y

        if direction == 'right':
            x += 1
            if x >= xlast:
                xlast -= 1
                direction = 'up'
        elif direction == 'up':
            y += 1
            if y >= ylast:
                ylast -= 1
                direction = 'left'
        elif direction == 'left':
            x -= 1
            if x <= xfirst:
                xfirst += 1
                direction = 'down'
        elif direction == 'down':
            y -= 1
            if y <= yfirst:
                yfirst += 1
                direction = 'right'


def diffusionMatrix(V_mat):
    D = np.zeros((V_mat.shape[0], 2, 2))  # this one works for two dimension -- generalize it to high dimensions
    D[:, 0, 0] = np.mean((V_mat[:, :, 0] - np.mean(V_mat[:, :, 0], axis=1)[:, None]) ** 2, axis=1)
    D[:, 1, 1] = np.mean((V_mat[:, :, 1] - np.mean(V_mat[:, :, 1], axis=1)[:, None]) ** 2, axis=1)
    D[:, 0, 1] = np.mean((V_mat[:, :, 0] - np.mean(V_mat[:, :, 0], axis=1)[:, None]) * (
                V_mat[:, :, 0] - np.mean(V_mat[:, :, 0], axis=1)[:, None]), axis=1)
    D[:, 1, 0] = D[:, 0, 1]

    return D / 2


def compute_velocity_on_grid(X_emb, V_emb, xy_grid_nums, density=None, smooth=None, n_neighbors=None, min_mass=None, autoscale=False,
                             adjust_for_stream=True):
    # prepare grid
    n_obs, n_dim = X_emb.shape
    print('n_obs, n_dim', n_obs, n_dim)
    density = 1 if density is None else density
    smooth = .5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - .01 * np.abs(M - m)
        M = M + .01 * np.abs(M - m)
        gr = np.linspace(m, M, xy_grid_nums[dim_i] * density)
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # estimate grid velocities
    if n_neighbors is None: n_neighbors = int(n_obs / 50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1) / np.maximum(1, p_mass)[:, None]

    # calculate diffusion matrix D
    D = diffusionMatrix(V_emb[neighs])

    if adjust_for_stream:
        X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(np.sqrt(len(V_grid[:, 0])))
        V_grid = V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((V_grid ** 2).sum(0))
        # V_grid[0][mass.reshape(V_grid[0].shape) < 1e-5] = np.nan
    else:
        if min_mass is None: min_mass = np.clip(np.percentile(p_mass, 99) / 100, 1e-2, 1)
        X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]

        if autoscale: V_grid /= 3 * quiver_autoscale(X_grid, V_grid)

    return X_grid, V_grid, D


def quiver_autoscale(X_emb, V_emb):
    import matplotlib.pyplot as pl
    scale_factor = X_emb.max()  # just so that it handles very large values
    Q = pl.quiver(X_emb[:, 0] / scale_factor, X_emb[:, 1] / scale_factor,
                  V_emb[:, 0], V_emb[:, 1], angles='xy', scale_units='xy', scale=None)
    Q._init()
    pl.clf()
    return Q.scale / scale_factor

def high_dimension_streamlines(adata, source, n_neighbors = None, reduce_dim_type = 'pca', direction = 'both'):
    V_source = value_at(adata, source, n_neighbors, reduce_dim_type)

    # travel to next position using rk4
    sf, f_traj = rk4(adata, source, V_source, f, n_neighbors, reduce_dim_type)
    sb, b_traj = rk4(adata, source, V_source, g, n_neighbors, reduce_dim_type)
    s_total = sf + sb
    traj = b_traj[::-1] + f_traj[1:]

    adata.uns['traj'] = traj

    return adata

def value_at(adata, source, n_neighbors = None, reduce_dim_type = 'pca'):
    velocity = adata.layers['velocity']
    n_obs, n_vars = adata.shape

    embedding = adata.X # obsm['X_' + reduce_dim_type]

    if n_neighbors is None: n_neighbors = int(n_obs / 50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(embedding)
    dists, neighs = nn.kneighbors(source)

    scale = np.mean(dists)
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    V_source = (velocity[neighs] * weight[:, :, None]).sum(1) / np.maximum(1, p_mass)[:, None]

    return V_source

## function to move forward / backward
def f(Vi):
    return Vi / np.sum(Vi ** 2)

def g(Vi):
    return - Vi / np.sum(Vi ** 2)
# def f(Vi):
#     return Vi
# def g(Vi):
#     return - Vi

## Integrator function
def rk4(adata, source, V_source, f, n_neighbors = None, reduce_dim_type = 'pca'):
    ds = 0.01 #min(1./NGX, 1./NGY, 0.01)
    stotal = 0
    f_traj = []
    Xi = source
    Vi = V_source

    while True:
        f_traj.append(Xi)
        k1 = f(Vi)
        k2 = f(0.5 * ds * k1)
        k3 = f(0.5 * ds * k2)
        k4 = f(ds * k3)

        Xi += ds * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        Vi = value_at(adata, Xi, n_neighbors, reduce_dim_type)

        stotal += ds
        print('stotal is ', stotal)

        if stotal > 2:
            break

    return stotal, f_traj

# test on two dimension first
# then validate in high dimension with real ODE results

