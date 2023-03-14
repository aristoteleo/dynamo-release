from typing import Callable, List, Union

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

# dynamo logger related
from ..dynamo_logger import (
    LoggerManager,
    main_critical,
    main_exception,
    main_info,
    main_tqdm,
    main_warning,
)
from ..tools.utils import AnnDataPredicate, select
from ..utils import areinstance, isarray
from .trajectory import Trajectory


def calc_mean_exit_time(trajectories: List[Trajectory], in_init_state: Callable, in_sink_state: Callable) -> float:
    """
    Calculates the mean exit time (MET) from the initial state to the sink state for a list of trajectories.

    Args:
        trajectories: A list of trajectories.
        in_init_state: A callable that takes a state as an argument and returns a boolean indicating whether
            the state is in the initial state.
        in_sink_state: A callable that takes a state as an argument and returns a boolean indicating whether
            the state is in the sink state.

    Returns:
        The mean exit time from the initial state to the sink state for the list of trajectories.

    Raises:
        ValueError: If no trajectory reaches the sink state.

    """
    met = []
    for traj in trajectories:
        t_init = -1
        for j, c in enumerate(traj.X):
            t = traj.t[j]
            if in_init_state(c):
                t_init = t
            if t_init > 0 and in_sink_state(c):
                met.append(t - t_init)
                break
    return np.mean(met)


def calc_mean_first_passage_time(
    trajectories: List[Trajectory], in_init_state: Callable, in_target_state: Callable, in_sink_state: Callable
) -> float:
    """
    Calculates the mean first-passage time (MFPT) from the initial state to the target state for a list of trajectories.

    Args:
        trajectories: A list of trajectories.
        in_init_state: A callable that takes a state as an argument and returns a boolean indicating whether
            the state is in the initial state.
        in_target_state: A callable that takes a state as an argument and returns a boolean indicating whether
            the state is in the target state.
        in_sink_state: A callable that takes a state as an argument and returns a boolean indicating whether
            the state is in the sink state.

    Returns:
        The mean first-passage time from the initial state to the target state for the list of trajectories.

    Raises:
        ValueError: If no trajectory reaches the target state.
    """
    mfpt = []
    for traj in trajectories:
        t_init = -1
        for j, c in enumerate(traj.X):
            t = traj.t[j]
            if in_init_state(c):
                t_init = t
            if t_init > 0:
                if in_target_state(c) and not in_sink_state(c):
                    mfpt.append(t - t_init)
                    break
                elif in_sink_state(c):
                    break
    return np.mean(mfpt)


def is_in_state(x: np.ndarray, centers: np.ndarray, radius: float) -> bool:
    """Checks whether a point is within a given radius of any center point in a set of centers.

    Args:
        x: The point to check.
        centers: The set of center points.
        radius: The radius within which to consider a point to be in a state.

    Returns:
        True if the point is within the given radius of any center point, False otherwise.
    """
    in_state = False
    if np.min(np.linalg.norm(x - np.atleast_2d(centers), axis=1)) <= radius:
        in_state = True
    return in_state


def mean_first_passage_time(
    adata: AnnData,
    sink_states: Union[list, np.ndarray, Callable],
    init_states: Union[None, list, np.ndarray, Callable] = None,
    target_states: Union[None, list, np.ndarray, Callable] = None,
    xkey: Union[None, str] = None,
    tkey: str = "time",
    traj_key: str = "trajectory",
    init_T_quantile: float = 0.01,
    init_state_radius: float = 0.1,
    sink_state_radius: float = 0.1,
    target_state_radius: float = 0.1,
) -> float:
    """Calculate the mean first passage time or mean exit time of a set of trajectories.

    Args:
        adata: The annotated data object containing the trajectories.
        sink_states: A sink state is a state that once entered, the trajectory can never leave. If it is a callable, it should take a point in the state space as input and return True if it is in a sink state and False otherwise.
        init_states: Specifies the initial states. If it is a callable, it should take
            a point in the state space as input and return True if it is in an initial state and False otherwise. If
            not specified, the initial states will be defined as the states with time points less than or equal to
            the `init_T_quantile` percentile of all time points.
        target_states: Specifies the target states. A target state is a state that
            once entered, the trajectory cannot return to the initial state. If it is a callable, it should take a
            point in the state space as input and return True if it is in a target state and False otherwise. If not
            specified, the mean exit time will be calculated instead of the mean first passage time.
        xkey: The key for the matrix of cell states. If None, the X attribute of the `adata` object
            will be used. Otherwise, the X attribute of the layer or the X attribute of the obsm slot of `adata` with
            the corresponding key will be used.
        tkey: Specifies the key for the time points of the trajectory. Default is "time".
        traj_key: The key for the trajectory index of each cell. Default is "trajectory".
        init_T_quantile: A float specifying the quantile threshold for defining initial states based on time points.
            Only used if `init_states` is not specified. Default is 0.01.
        init_state_radius: A float specifying the radius of the initial states. Default is 0.1.
        sink_state_radius: A float specifying the radius of the sink states. Default is 0.1.
        target_state_radius: A float specifying the radius of the target states. Default is 0.1.

    Returns:
        The mean first passage time or mean exit time, as a float.
    """

    if xkey is None:
        X = adata.X
    elif xkey in adata.layers.keys():
        X = adata.layers[xkey]
    elif xkey in adata.obsm.keys():
        X = adata.obsm[xkey]
    else:
        raise Exception(f"Cannot find `{xkey}` in neither `.layers` nor `.obsm`.")

    T = adata.obs[tkey]
    traj_id = adata.obs[traj_key]

    if sp.issparse(X):
        X = X.A

    trajs = []
    for traj_i in np.unique(traj_id):
        cells = select(adata.obs, AnnDataPredicate(traj_key, traj_i))
        traj = Trajectory(X[cells], T[cells])
        trajs.append(traj)

    def in_state_func(states, radius):
        if type(states) == Callable:
            # an in_state function
            in_state = states
        elif isarray(states):
            if np.array(states).ndim == 1:
                # an 1d array of cell indices
                in_state = lambda x: is_in_state(x, X[states], radius)
            else:
                # a 2d array of coordiantes
                in_state = lambda x: is_in_state(x, states, radius)
        return in_state

    in_sink_state = in_state_func(sink_states, sink_state_radius)

    if init_states is None:
        init_states = np.where(T <= np.quantile(T, init_T_quantile))[0]
    in_init_state = in_state_func(init_states, init_state_radius)

    if target_states is not None:
        in_target_state = in_state_func(target_states, target_state_radius)

    if target_states is None:
        main_info("No target states is provided. Calculating mean exit time...")
        mfpt = calc_mean_exit_time(trajs, in_init_state=in_init_state, in_sink_state=in_sink_state)
    else:
        main_info("Calculating mean first passage time to the target states...")
        mfpt = calc_mean_first_passage_time(
            trajs, in_init_state=in_init_state, in_target_state=in_target_state, in_sink_state=in_sink_state
        )

    return mfpt
