import numpy as np
import scipy

from ..tools.scVectorField import vector_field_function
from ..tools.scPotential import gen_fixed_points


def plot_flow_field(ax, f, u_range, v_range, start_points=None, args=(), n_grid=100):
    """Plots the flow field with line thickness proportional to speed.
    code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html

    Parameters
    ----------
    ax : Matplotlib Axis instance
        Axis on which to make the plot
    f : function for form f(y, t, *args)
        The right-hand-side (rhs) of the dynamical system.
        Must return a 2-array.
    u_range : array_like, shape (2,)
        Range of values for u-axis.
    v_range : array_like, shape (2,)
        Range of values for v-axis.
    args : tuple, default ()
        Additional arguments to be passed to f
    n_grid : int, default 100
        Number of grid points to use in computing
        derivatives on phase portrait.
        
    Returns
    -------
    output : Matplotlib Axis instance
        Axis with streamplot included.
    """
    
    # Set up u,v space
    u = np.linspace(u_range[0], u_range[1], n_grid)
    v = np.linspace(v_range[0], v_range[1], n_grid)
    uu, vv = np.meshgrid(u, v)

    # Compute derivatives
    u_vel = np.empty_like(uu)
    v_vel = np.empty_like(vv)
    for i in range(uu.shape[0]):
        for j in range(uu.shape[1]):
            u_vel[i,j], v_vel[i,j] = f(np.array([uu[i,j], vv[i,j]]), None, *args)

    # Compute speed
    speed = np.sqrt(u_vel**2 + v_vel**2)

    # Make linewidths proportional to speed,
    # with minimal line width of 0.5 and max of 3
    lw = 0.5 + 2.5 * speed / speed.max()

    # Make stream plot
    if start_points is None:
        ax.streamplot(uu, vv, u_vel, v_vel, linewidth=lw, arrowsize=1.2,
                      density=1, color='thistle')
    else:
        ax.streamplot(uu, vv, u_vel, v_vel, linewidth=lw * 2, arrowsize=1.2, start_points = start_points,
                      density=1, color='tomato')

    return ax


def plot_null_clines(ax, f, a_range, b_range, fixed_points=None, method='point_wise', colors=['#1f77b4', '#1f77b4'], lw=3):
    """Add nullclines to ax.
        code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html
    """
    # from scipy.optimize import Bounds
    # from scipy import optimize
    # a_bounds, b_bounds = Bounds(a_range[0], a_range[1]), Bounds(b_range[0], b_range[1])

    # a/b-nullcline
    nca_a = np.linspace(a_range[0], a_range[1], 200)
    nca_b = np.linspace(b_range[0], b_range[1], 200)
    res_a, res_b = np.zeros_like(nca_a), np.zeros_like(nca_a)

    if method is 'point_wise':
        for i in range(200):
            nullc_a = lambda a: f(np.array([a[0], nca_b[i]]), t=None)[0] # given y-coordinates, calculate x
            nullc_b = lambda b: f(np.array([nca_a[i], b[0]]), t=None)[1] # given x-coordinates, calculate y
            res_a[i] = scipy.optimize.fsolve(nullc_a, res_a[i - 1] + abs(np.random.normal(0, 1e-1, 1))) if i > 0 \
                else scipy.optimize.fsolve(nullc_a, nca_a[i] + abs(np.random.normal(0, 1e-1, 1)))
            res_b[i] = scipy.optimize.fsolve(nullc_b, res_b[i - 1] - abs(np.random.normal(0, 1e-1, 1))) if i > 0 \
                else scipy.optimize.fsolve(nullc_b, nca_b[i] - abs(np.random.normal(0, 1e-1, 1)))
            # res_a[i] = optimize.shgo(nullc_a, a_bounds)
            # res_b[i] = optimize.shgo(nullc_b, b_bounds)
    # elif method is 'path_like':
    #     # initial path as a line connecting start point and end point point_start*ones(1,n_points+1)+(point_end-point_start)*(0:tmax/n_points:tmax)/tmax;
    #     initpath = point_start.dot(np.ones((1, n_points + 1))) + (point_end - point_start).dot(
    #         np.linspace(0, tmax, n_points + 1, endpoint=True).reshape(1, -1)) / tmax
    #
    #     Bounds = scipy.optimize.Bounds((np.ones((1, (n_points - 1) * 2)) * boundary[0]).flatten(),
    #                                    (np.ones((1, (n_points - 1) * 2)) * boundary[1]).flatten(), keep_feasible=True)
    #
    #     # sc.optimize.least_squares(lambda_f, initpath[:, 1:n_points].flatten())
    #     res = sc.optimize.minimize(lambda_f, initpath[:, 1:n_points],
    #                                tol=1e-12)  # , bounds=Bounds , options={"maxiter": 250}
    #     fval, output_path = res['fun'], np.hstack((point_start, res['x'].reshape((2, -1)), point_end))

    # Plot
    ax.plot(res_a, nca_b, lw=lw, color=colors[0])
    ax.plot(nca_a, res_b, lw=lw, color=colors[1])
    
    return ax


# if multiple points are very close, maybe combine them together?
def plot_fixed_points(ax, f, dim_range=[0, 6], saddle=None, stable=None, **fix_points_kwargs):
    """Add fixed points to plot."""
    # Compute fixed points

    if saddle is None and stable is None:
        fix_points_dict = {"auto_func": None, "dim_range": dim_range, "RandNum": 5000, "EqNum": 2, "x_ini": None}
        if fix_points_kwargs is not None:
            fix_points_dict.update(fix_points_kwargs)

        stable, saddle = gen_fixed_points(func=f, **fix_points_dict)
        stable = stable[:, np.logical_and(dim_range[0] >= stable[0], stable[0] <= dim_range[1]) &
                           np.logical_and(dim_range[0] >= stable[1], stable[1] <= dim_range[1])]
        saddle = saddle[:, np.logical_and(dim_range[0] >= saddle[0], saddle[0] <= dim_range[1]) &
                           np.logical_and(dim_range[0] >= saddle[1], saddle[1] <= dim_range[1])]
    # Plot
    for i in range(stable.shape[1]): # attractors
        ax.plot(*stable[:, i], '.', color='black', markersize=20)
    for i in range(saddle.shape[1]): # saddle points
        ax.plot(*saddle[:, i], '.', markerfacecolor='white', markeredgecolor='black',
                markeredgewidth=2, markersize=20)

    return ax


def plot_traj(ax, f, y0, t, args=(), color='black', lw=2):
    """Plots a trajectory on a phase portrait.
    code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html

    Parameters
    ----------
    ax : Matplotlib Axis instance
        Axis on which to make the plot
    f : function for form f(y, t, *args)
        The right-hand-side of the dynamical system.
        Must return a 2-array.
    y0 : array_like, shape (2,)
        Initial condition.
    t : array_like
        Time points for trajectory.
    args : tuple, default ()
        Additional arguments to be passed to f
    n_grid : int, default 100
        Number of grid points to use in computing
        derivatives on phase portrait.
        
    Returns
    -------
    output : Matplotlib Axis instance
        Axis with streamplot included.
    """
    
    y = scipy.integrate.odeint(f, y0, t, args=args)
    ax.plot(*y.transpose(), color=color, lw=lw)
    return ax


def plot_separatrix(ax, f, saddle, a_range, b_range, t, eps=1e-6,
                           color='tomato', lw=3, **fix_points_kwargs):
    """Plot separatrix on phase portrait.
        code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html

    """
    # Compute fixed points

    if saddle is None:
        fix_points_dict = {"auto_func": None, "dim_range": [min(a_range[0], b_range[0]), max(a_range[1], b_range[1])], "RandNum": 5000, "EqNum": 2, "x_ini": None}
        if fix_points_kwargs is not None:
            fix_points_dict.update(fix_points_kwargs)

        stable, saddle = gen_fixed_points(func=f, **fix_points_dict)

        saddle = saddle[:, np.logical_and(saddle[0] >= a_range[0], saddle[0] <= a_range[1]) &
                           np.logical_and(saddle[1] >= b_range[0], saddle[1] <= b_range[1])]

    # If only no saddle point, no separatrix
    if len(saddle) < 1:
        return ax
    
    # Negative time function to integrate to compute separatrix
    def rhs(ab, t):
        # Unpack variables
        a, b = ab
    
        # Stop integrating if we get the edge of where we want to integrate
        if a_range[0] < a < a_range[1] and b_range[0] < b < b_range[1]:
            return -f(ab, t)
        else:
            return np.array([0, 0])

    # Parameters for building separatrix
    # t = np.linspace(0, t_max, 400)

    all_sep_a, all_sep_b = None, None
    for i in range(saddle.shape[1]):
        fps = saddle[:, i]
        # Build upper right branch of separatrix
        ab0 = fps + eps
        ab_upper = scipy.integrate.odeint(rhs, ab0, t)

        # Build lower left branch of separatrix
        ab0 = fps - eps
        ab_lower = scipy.integrate.odeint(rhs, ab0, t)

        # Concatenate, reversing lower so points are sequential
        sep_a = np.concatenate((ab_lower[::-1,0], ab_upper[:,0]))
        sep_b = np.concatenate((ab_lower[::-1,1], ab_upper[:,1]))

        # Plot
        ax.plot(sep_a, sep_b, '-', color=color, lw=lw)

        all_sep_a = sep_a if all_sep_a is None else np.concatenate((all_sep_a, sep_a))
        all_sep_b = sep_b if all_sep_b is None else np.concatenate((all_sep_b, sep_b))

    return ax


def topography(adata, basis, xlim, ylim, t, terms=['streamline', 'nullcline', 'fixed_points', 'separatrices', 'trajectory'], init_state=None, VF=None, plot=True, **fixed_points_kwargs):
    """ Plot the streamline, fixed points (attractor / saddles), nullcline, separatrices of a recovered dynamic system
    for single cells. The plot is created on two dimensional space.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        basis: `str` (default: `trimap`)
            The reduced dimension embedding of cells to visualize.
        t:  t_end: `float` (default 1)
            The length of the time period from which to predict cell state forward or backward over time. This is used
            by the odeint function.
        xlim: `numpy.ndarray`
            The range of x-coordinate
        ylim: `numpy.ndarray`
            The range of y-coordinate
        terms: `list` (default: ['streamline', 'nullcline', 'fixed_points', 'separatrices', 'trajectory'])
            A list of plotting items to include in the final topography figure.
        init_state: `numpy.ndarray` (default: None)
            Initial cell states for the historical or future cell state prediction with numerical integration.
        VF: `function` or None (default: None)
            The true vector field function if known and want to demonstrate.
        plot: `bool` (default: True)
            Whether or not to plot the topography plot or just return the axis object.
        fixed_points_kwargs: `dict`
            A dictionary of parameters for calculating the fixed points.

    Returns
    -------
        Nothing but plots the streamline, fixed points (attractors / saddles), nullcline, separatrices of a recovered dynamic system
        for single cells or return the corresponding axis, depending on the plot argument.
    """

    import matplotlib.pyplot as plt

    VecFld = adata.uns['VecFld'] if basis is 'X' else adata.uns['VecFld_' + basis]

    if VF is None:
        VF = lambda x, t=None: vector_field_function(x=x, t=t, VecFld=VecFld)

    # Set up the figure
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel(basis + '_1')
    ax.set_ylabel(basis + '_2')
    ax.set_aspect('equal')

    # Build the plot
    xlim = [0, 6] if xlim is None else xlim
    ylim = [0, 6] if ylim is None else ylim
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if t is None:
        t = np.linspace(0, max(max(np.diff(xlim), np.diff(ylim)) / np.percentile(VecFld['grid_V'].__abs__(), 5)), 1e7)

    if 'streamline' in terms:
        ax = plot_flow_field(ax, VF, xlim, ylim)

    if 'nullcline' in terms  or 'fixed_points' in terms or 'separatrices' in terms:
        fix_points_dict = {"auto_func": None, "dim_range": xlim, "RandNum": 5000, "EqNum": 2, "x_ini": None}

        if fixed_points_kwargs is not None:
            fix_points_dict.update(fixed_points_kwargs)

        stable, saddle = gen_fixed_points(func=VF, **fix_points_dict)
        stable = stable[:, np.logical_and(stable[0] >= xlim[0], stable[0] <= xlim[1]) &
                           np.logical_and(stable[1] >= ylim[0], stable[1] <= ylim[1])]
        saddle = saddle[:, np.logical_and(saddle[0] >= xlim[0], saddle[0] <= xlim[1]) &
                           np.logical_and(saddle[1] >= ylim[0], saddle[1] <= ylim[1])]

    if 'nullcline' in terms:
        points = np.hstack((stable, saddle))

        ax = plot_null_clines(ax, VF, xlim, ylim)

    if 'fixed_points' in terms:
        ax = plot_fixed_points(ax, VF, dim_range=xlim, saddle=saddle, stable=stable)
    if 'separatrices' in terms:
        if stable.shape[1] > 2:
            ax = plot_flow_field(ax, VF, xlim, ylim, start_points=saddle.T - 0.01)
            ax = plot_flow_field(ax, VF, xlim, ylim, start_points=saddle.T + 0.01)
        else:
            ax = plot_separatrix(ax, VF, saddle, xlim, ylim, t=t)

    if init_state is not None and 'trajectory' in terms:
        ax = plot_traj(ax, VF, init_state, t)

    if plot:
        plt.show()
    else:
        return ax

