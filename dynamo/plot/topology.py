import numpy as np
import scipy

from ..tools.scVectorField import vector_field_function
from ..tools.scPotential import gen_fixed_points

def plot_flow_field(ax, f, u_range, v_range, args=(), n_grid=100):
    """Plots the flow field with line thickness proportional to speed.
    code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html

    Parameters
    ----------
    ax : Matplotlib Axis instance
        Axis on which to make the plot
    f : function for form f(y, t, *args)
        The right-hand-side of the dynamical system.
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
    ax.streamplot(uu, vv, u_vel, v_vel, linewidth=lw, arrowsize=1.2, 
                  density=1, color='thistle')

    return ax


def plot_null_clines(ax, f, a_range, b_range, colors=['#1f77b4', '#1f77b4'], lw=3):
    """Add nullclines to ax.
        code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html
    """
    
    # a/b-nullcline
    nca_a = np.linspace(a_range[0], a_range[1], 200)
    nca_b = np.linspace(b_range[0], b_range[1], 200)
    res_a, res_b = np.zeros_like(nca_a), np.zeros_like(nca_a)

    for i in range(200):
        nullc_a = lambda a: f(np.array([a, nca_b[i]]), t=None)[0]
        nullc_b = lambda b: f(np.array([nca_a[i], b]), t=None)[1]
        res_a[i]= scipy.optimize.fsolve(nullc_a, [0])
        res_b[i] = scipy.optimize.fsolve(nullc_b, [0])

    # # b-nullcline
    # ncb_a = np.linspace(a_range[0], a_range[1], 200)
    # ncb_b = beta / (1 + ncb_a**n)

    # Plot
    ax.plot(nca_a, res_a, lw=lw, color=colors[0])
    ax.plot(nca_b, res_b, lw=lw, color=colors[1])
    
    return ax


# if multiple points are very close, maybe combine them together?
def plot_fixed_points(ax, f, **fix_points_kwargs):
    """Add fixed points to plot."""
    # Compute fixed points
    fix_points_dict = {"auto_func": None, "dim_range": [-10, 10], "RandNum": 5000, "EqNum": 2, "x_ini": None}
    if fix_points_kwargs is not None:
        fix_points_dict.update(fix_points_kwargs)

    stable, saddle = gen_fixed_points(func=f, **fix_points_dict)

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


def plot_separatrix(ax, f, saddle, a_range, b_range, t_max=30, eps=1e-6,
                           color='tomato', lw=3):
    """Plot separatrix on phase portrait.
        code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html

    """
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
    t = np.linspace(0, t_max, 400)

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

        all_sep_a = sep_a if all_sep_a is None else np.concatenate((all_sep_a, sep_a))
        all_sep_b = sep_b if all_sep_b is None else np.concatenate((all_sep_b, sep_b))

    # Plot
    ax.plot(all_sep_a, all_sep_b, '-', color=color, lw=lw)
    
    return ax


def plot_topology(adata, basis, y0, t, a_range, b_range, VecFld_true=None, plot=True, **fixed_points_kwargs):
    import matplotlib.pyplot as plt

    VecFld = adata.uns['VecFld'] if basis is 'X' else adata.uns['VecFld_' + basis]
    f = lambda x, t: vector_field_function(x=x, t=t, VecFld=VecFld) if VecFld_true is None else VecFld_true

    # Set up the figure
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_aspect('equal')

    # Build the plot
    a_range = [0, 6] if a_range else None
    b_range = [0, 6] if b_range else None
    ax.set_xlim(a_range)
    ax.set_ylim(b_range)

    ax = plot_flow_field(ax, f, a_range, b_range)
    ax = plot_null_clines(ax, f, a_range, b_range)
    ax = plot_fixed_points(ax, f)
    fix_points_dict = {"auto_func": None, "dim_range": [-10, 10], "RandNum": 5000, "EqNum": 2, "x_ini": None}

    if fixed_points_kwargs is not None:
        fix_points_dict.update(fixed_points_kwargs)

    stable, saddle = gen_fixed_points(func=VecFld, **fix_points_dict)
    ax = plot_separatrix(ax, f, saddle, a_range, b_range)
    ax = plot_traj(ax, f, np.array([0.5, 0.5]), np.linspace(0, 2, 250))

    if plot:
        plt.show()
    else:
        return ax


if __name__ is '__main__':
    import matplotlib.pyplot as plt
    import dynamo as dyn
    VecFld = dyn.tl.ODE
    f = lambda x, t: VecFld(x=x)

    # Set up the figure
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_aspect('equal')

    # Build the plot
    a_range = [-3, 3]
    b_range = [-3, 3]
    ax.set_xlim(a_range)
    ax.set_ylim(b_range)

    ax = plot_flow_field(ax, f, a_range, b_range)
    ax = plot_null_clines(ax, f, a_range, b_range)
    ax = plot_fixed_points(ax, f)
    fix_points_dict = {"auto_func": None, "dim_range": [-10, 10], "RandNum": 5000, "EqNum": 2, "x_ini": None}

    stable, saddle = dyn.tl.gen_fixed_points(func=VecFld, **fix_points_dict)
    ax = plot_separatrix(ax, f, saddle, a_range, b_range)
    ax = plot_traj(ax, f, np.array([0.5, 0.5]), np.linspace(0, 2, 250))

    plt.show()
