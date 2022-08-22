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


def is_in_state(x, centers, radius):
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
    init_T_quantile=0.01,
    init_state_radius=0.1,
    sink_state_radius=0.1,
    target_state_radius=0.1,
) -> float:

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
