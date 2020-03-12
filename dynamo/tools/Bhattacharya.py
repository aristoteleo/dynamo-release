import numpy as np

# from scipy.integrate import odeint
from scipy.interpolate import griddata


def path_integral(
    VecFnc, x_lim, y_lim, xyGridSpacing, dt=1e-2, tol=1e-2, numTimeSteps=1400
):
    """A deterministic map of Waddington’s epigenetic landscape for cell fate specification
    Sudin Bhattacharya, Qiang Zhang and Melvin E. Andersen

    Parameters
    ----------
    VecFnc
    x_lim: `list`
        Lower or upper limit of x-axis.
    y_lim: `list`
        Lower or upper limit of y-axis
    xyGridSpacing: `float`
        Grid spacing for "starting points" for each "path" on the pot. surface
    dt: `float`
        Time step for the path integral.
    tol: `float` (default: 1.0e-2)
        Tolerance to test for convergence.
    numTimeSteps: `int`
        A high-enough number for convergence with given dt.

    Returns
    -------
    numAttractors: `int`
        Number of attractors identified by the path integral approach.
    attractors_num_X_Y: `numpy.ndarray`
        Attractor number and the corresponding x, y coordinates.
    sepx_old_new_pathNum: `numpy.ndarray`
        The IDs of the two attractors for each separaxis per row.
    numPaths_att `numpy.ndarray`
        Number of paths per attractor
    numPaths: `int`
        Total Number of paths for defined grid spacing.
    numTimeSteps: `int`
        A high-enough number for convergence with given dt.
    pot_path: `numpy.ndarray` (dimension: numPaths x numTimeSteps)
        Potential along the path.
    path_tag: `numpy.ndarray` (dimension: numPaths x 1)
        Tag for given path (to denote basin of attraction).
    attractors_pot: `numpy.ndarray`
        Potential value of each identified attractors by the path integral approach.
    x_path: `numpy.ndarray`
        x-coord. along path.
    y_path: `numpy.ndarray`
        y-coord. along path.
    """

    # -- First, generate potential surface from deterministic rate equations –

    # Define grid spacing for "starting points" for each "path" on the pot. surface

    # Define grid spacing for "starting points" for each "path" on the pot. surface

    # No. of time steps for integrating along each path (to ensure uniform arrays)

    # Time step and tolerance to test for convergence

    # Calculate total no. of paths for defined grid spacing
    numPaths = int(np.diff(x_lim) / xyGridSpacing + 1) ** 2

    # Initialize "path" variable matrices
    x_path = np.zeros((numPaths, numTimeSteps))  # x-coord. along path
    y_path = np.zeros((numPaths, numTimeSteps))  # y-coord. along path
    pot_path = np.zeros((numPaths, numTimeSteps))  # pot. along path

    path_tag = np.ones(
        (numPaths, 1), dtype="int"
    )  # tag for given path (to denote basin of attraction)
    # ** initialized to 1 for all paths **

    # Initialize "Path counter" to 1
    path_counter = 0

    # Initialize no. of attractors and separatrices (basin boundaries)
    num_attractors = 0
    num_sepx = 0

    # Assign array to keep track of attractors and their coordinates; and pot.
    attractors_num_X_Y = None
    attractors_pot = None

    # Assign array to keep track of no. of paths per attractor
    numPaths_att = None

    # Assign array to keep track of separatrices
    sepx_old_new_pathNum = None

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Loop over x-y grid
    for i in np.arange(x_lim[0], x_lim[1] + xyGridSpacing, xyGridSpacing):
        for j in np.arange(y_lim[0], y_lim[1] + xyGridSpacing, xyGridSpacing):
            # *** Init conds for given (x,y) ***
            # Initialize coords.
            x0 = i
            y0 = j

            # ** Set initial value of "potential" to 0 **
            p0 = 0  # (to facilitate comparison of "potential drop")

            # Initialize "path" variables
            x_p = x0
            y_p = y0

            # Initialize accumulators for "potential" along path
            Pot = p0
            Pot_old = 1.0e7  # initialize to large number

            # Initialize global arrays (time t = 0 counts as "time step #1")
            x_path[path_counter, 0] = x_p
            y_path[path_counter, 0] = y_p
            pot_path[path_counter, 0] = Pot

            # Evaluate potential (Integrate) over trajectory from init cond to  stable steady state
            for n_steps in np.arange(1, numTimeSteps):
                # record "old" values of variables
                # x_old = x_p;
                # y_old = y_p;
                # v_old = v;
                Pot_old = Pot

                # update dxdt, dydt

                dxdt, dydt = VecFnc([x_p, y_p])

                # update x, y
                dx = dxdt * dt
                dy = dydt * dt

                x_p = x_p + dx
                y_p = y_p + dy

                x_path[path_counter, n_steps] = x_p
                y_path[path_counter, n_steps] = y_p

                # update "potential"
                dPot = (
                    -(dxdt) * dx - (dydt) * dy
                )  # signs ensure that "potential" decreases as "velocity" increases
                Pot = Pot_old + dPot
                pot_path[path_counter, n_steps] = Pot

            # ################################################################################################################
            # # just use odeint for integration:
            # Pot_func = lambda x_p, y_p: -VecFnc([x_p, y_p])**2
            # y_path=odeint(Pot_func, x_p, y_p, t=0)
            # ################################################################################################################
            # check for convergence
            if abs(Pot - Pot_old) > tol:
                print(1, "Warning: not converged!\n")

            # --- assign path tag (to track multiple basins of attraction) ---
            if path_counter == 0:
                # record attractor of first path and its coords
                num_attractors = num_attractors + 1
                current_att_num_X_Y = np.array(
                    [num_attractors, x_p, y_p]
                )  # create array
                attractors_num_X_Y = (
                    np.vstack((attractors_num_X_Y, current_att_num_X_Y))
                    if attractors_num_X_Y is not None
                    else np.array([current_att_num_X_Y])
                )  # append array (vertically)
                attractors_pot = (
                    np.vstack((attractors_pot, Pot))
                    if attractors_pot is not None
                    else np.array([Pot])
                )  # append attractor potentials to array (vertically)
                path_tag[path_counter] = num_attractors - 1  # initialize path tag
                numPaths_att = (
                    np.vstack((numPaths_att, 1))
                    if numPaths_att is not None
                    else np.array([1])
                )  # append to array (vertically)

            else:
                # i.e. if path counter > 1
                # set path tag to that of previous path (default)
                path_tag[path_counter] = path_tag[path_counter - 1]

                # record info of previous path
                x0_lastPath = x_path[(path_counter - 1), 0]
                y0_lastPath = y_path[(path_counter - 1), 0]

                xp_lastPath = x_path[(path_counter - 1), numTimeSteps - 1]
                yp_lastPath = y_path[(path_counter - 1), numTimeSteps - 1]

                pot_p_lastPath = pot_path[(path_counter - 1), numTimeSteps - 1]

                # calculate distance between "start points" of current and previous paths
                startPt_dist_sqr = (x0 - x0_lastPath) ** 2 + (y0 - y0_lastPath) ** 2

                # calculate distance between "end points" of current and previous paths
                endPt_dist_sqr = (x_p - xp_lastPath) ** 2 + (y_p - yp_lastPath) ** 2

                # check if the current path *ended* in a different point compared to previous path (x-y grid spacing used
                # as a "tolerance" for distance)
                if endPt_dist_sqr > (2 * (xyGridSpacing ** 2)):

                    # --- check if this "different" attractor has been identified before
                    new_attr_found = 1

                    for k in range(num_attractors):
                        x_att = attractors_num_X_Y[k, 1]
                        y_att = attractors_num_X_Y[k, 2]
                        if (abs(x_p - x_att) < xyGridSpacing) and (
                            abs(y_p - y_att) < xyGridSpacing
                        ):
                            # this attractor has been identified before
                            new_attr_found = 0
                            path_tag[path_counter] = k  # DOUBLE CHECK ***
                            numPaths_att[k] = numPaths_att[k] + 1
                            break  # exit for-loop

                    if new_attr_found == 1:
                        num_attractors = num_attractors + 1
                        current_att_num_X_Y = [num_attractors, x_p, y_p]  # create array
                        attractors_num_X_Y = np.vstack(
                            (attractors_num_X_Y, current_att_num_X_Y)
                        )  # append array (vertically)
                        path_tag[path_counter] = num_attractors - 1  # DOUBLE CHECK **
                        numPaths_att = np.vstack(
                            (numPaths_att, 1)
                        )  # append to array (vertically)
                        attractors_pot = np.vstack(
                            (attractors_pot, Pot)
                        )  # append attractor potentials to array (vertically)

                        # check if start points of current and previous paths are "adjacent" - if so, assign separatrix
                        if startPt_dist_sqr < (2 * (xyGridSpacing ** 2)):
                            curr_sepx = [
                                path_tag[path_counter - 1],
                                path_tag[path_counter],
                                (path_counter - 1),
                            ]  # create array
                            sepx_old_new_pathNum = (
                                np.vstack((sepx_old_new_pathNum, curr_sepx))
                                if sepx_old_new_pathNum is not None
                                else np.array([curr_sepx])
                            )  # append array (vertically)
                            # attractors_pot = np.vstack((attractors_pot, Pot)) # append attractor potentials to array (vertically) #????????????????????????????????????????????????????????????????????????????????????
                            num_sepx = num_sepx + 1  # increment no. of separatrices
                    else:
                        # --- check if the attractor of the *previous* path
                        #     has been encountered in a separatrix before ---
                        #     (note that current path tag has already been set
                        #     above)

                        prev_attr_new = 1

                        for k in range(num_sepx):
                            attr1 = sepx_old_new_pathNum[k, 0]
                            attr2 = sepx_old_new_pathNum[k, 1]

                            if (path_tag[path_counter - 1] == attr1) or (
                                path_tag[path_counter - 1] == attr2
                            ):
                                # this attractor has been identified before
                                prev_attr_new = 0
                                break  # exit for-loop

                        if prev_attr_new == 1:
                            # check if start points of current and previous paths are "adjacent"  - if so, assign separatrix
                            if startPt_dist_sqr < (2 * (xyGridSpacing ** 2)):
                                curr_sepx = [
                                    path_tag[path_counter - 1],
                                    path_tag[path_counter],
                                    (path_counter - 1),
                                ]  # create array
                                sepx_old_new_pathNum = (
                                    np.vstack((sepx_old_new_pathNum, curr_sepx))
                                    if sepx_old_new_pathNum is not None
                                    else np.array([curr_sepx])
                                )  # append array (vertically)
                                # attractors_pot = np.vstack((attractors_pot, pot_p_lastPath)) # append attractor potentials to array vertically) #????????????????????????????????????????????????????????????????????????????????????
                                num_sepx = num_sepx + 1  # increment no. of separatrices

                else:
                    # i.e. current path converged at same pt. as previous path

                    # update path tag
                    # path_tag(path_counter) = path_tag(path_counter - 1);

                    # update no. of paths for current attractor
                    # (path tag already updated at start of path-counter loop)
                    tag = path_tag[path_counter]
                    numPaths_att[tag - 1] = numPaths_att[tag - 1] + 1

            # increment "path counter"
            path_counter = path_counter + 1

    return (
        attractors_num_X_Y,
        sepx_old_new_pathNum,
        numPaths_att,
        num_attractors,
        numPaths,
        numTimeSteps,
        pot_path,
        path_tag,
        attractors_pot,
        x_path,
        y_path,
    )


def alignment(
    numPaths,
    numTimeSteps,
    pot_path,
    path_tag,
    attractors_pot,
    x_path,
    y_path,
    grid=100,
    interpolation_method="linear",
):
    """ Align potential values so all path-potentials end up at same global min and then generate potential surface with
    interpolation on a grid.

    Parameters
    ----------
    numPaths: `int`
        Total Number of paths for defined grid spacing.
    numTimeSteps: `int`
        A high-enough number for convergence with given dt.
    pot_path: `numpy.ndarray` (dimension: numPaths x numTimeSteps)
        Potential along the path.
    path_tag: `numpy.ndarray` (dimension: numPaths x 1)
        Tag for given path (to denote basin of attraction).
    attractors_pot: `numpy.ndarray`
        Potential value of each identified attractors by the path integral approach.
    x_path: `numpy.ndarray`
        x-coord. along path.
    y_path: `numpy.ndarray`
        y-coord. along path.
    grid: `int`
        No. of grid lines in x- and y- directions
    interpolation_method: `string`
        Method of interpolation in griddata function. One of

        ``nearest``
          return the value at the data point closest to
          the point of interpolation.  See `NearestNDInterpolator` for
          more details.

        ``linear``
          tessellate the input point set to n-dimensional
          simplices, and interpolate linearly on each simplex.  See
          `LinearNDInterpolator` for more details.

        ``cubic`` (1-D)
          return the value determined from a cubic
          spline.

        ``cubic`` (2-D)
          return the value determined from a
          piecewise cubic, continuously differentiable (C1), and
          approximately curvature-minimizing polynomial surface. See
          `CloughTocher2DInterpolator` for more details.

    Returns
    -------
    Xgrid: `numpy.ndarray`
        x-coordinates of the Grid produced from the meshgrid function.
    Ygrid: `numpy.ndarray`
            y-coordinates of the Grid produced from the meshgrid function.
    Zgrid: `numpy.ndarray`
            z-coordinates or potential at each of the x/y coordinate.
    """

    # -- need 1-D "lists" (vectors) to plot all x,y, Pot values along paths --
    list_size = numPaths * numTimeSteps

    x_p_list = np.zeros((list_size, 1))
    y_p_list = np.zeros((list_size, 1))
    pot_p_list = np.zeros((list_size, 1))

    n_list = 0

    # "Align" potential values so all path-potentials end up at same global min.
    for n_path in range(numPaths):
        tag = path_tag[n_path]
        # print(tag)
        del_pot = pot_path[n_path, numTimeSteps - 1] - attractors_pot[tag]

        # align pot. at each time step along path
        for n_steps in range(numTimeSteps):
            pot_old = pot_path[n_path, n_steps]
            pot_path[n_path, n_steps] = pot_old - del_pot

            # add data point to list
            x_p_list[n_list] = x_path[n_path, n_steps]
            y_p_list[n_list] = y_path[n_path, n_steps]
            pot_p_list[n_list] = pot_path[n_path, n_steps]

            n_list = n_list + 1  # increment n_list

    # Generate surface interpolation grid

    # % To generate log-log surface
    x_p_list = x_p_list + 0.1
    y_p_list = y_p_list + 0.1

    # --- Create X,Y grid to interpolate "potential surface" ---
    xlin = np.linspace(min(x_p_list), max(x_p_list), grid)
    ylin = np.linspace(min(y_p_list), max(y_p_list), grid)
    Xgrid, Ygrid = np.meshgrid(xlin, ylin)

    Zgrid = griddata(
        np.hstack((x_p_list, y_p_list)),
        pot_p_list,
        np.vstack((Xgrid.flatten(), Ygrid.flatten())).T,
        method=interpolation_method,
    )
    Zgrid = Zgrid.reshape(Xgrid.shape)

    # print('Ran surface grid-interpolation okay!\n')

    return Xgrid, Ygrid, Zgrid
