from typing import Callable, List, Optional, Tuple, Union

# from anndata._core.views import ArrayView
import numpy as np
from anndata import AnnData
from scipy import interpolate
from scipy.integrate import solve_ivp
from tqdm import tqdm

from ..dynamo_logger import main_warning
from ..utils import isarray, normalize

# import scipy.sparse as sp
from ..vectorfield.topography import dup_osc_idx_iter
from .trajectory import Trajectory

# ---------------------------------------------------------------------------------------------------
# initial state related


def init_r0_pulse(
    r: Union[float, np.ndarray], l: Union[float, np.ndarray], k: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """calculate initial total RNA via ODE formula of RNA kinetics for one-shot/kinetics experiment

    Args:
        r: total RNA at current time point.
        l: labeled RNA at current time point.
        k: $k = 1 - e^{-\gamma t}$

    Returns:
        r0: The intial total RNA at the beginning of the one-shot or kinetics experiment.
    """
    r0 = (r - l) / (1 - k)

    return r0


def init_l0_chase(
    l: Union[float, np.ndarray], gamma: Union[float, np.ndarray], t: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """calculate initial labeled RNA (l0) via ODE formula of RNA kinetics for degradation experiment

    Note that this approach only estimate the initial labeled RNA based on first-order decay model. To get the intial r0
    we can also assume cells with extreme total RNA as steady state cells and use that to estimate transcription rate.

    Args:
        l: labeled RNA(s)
        gamma: degradation rate(s)
        t:labeling time(s)

    Returns:
        l0: The initial labeled RNA at the beginning of a degradation experiment.
    """
    l0 = l / np.exp(-gamma * t)

    return l0


# ---------------------------------------------------------------------------------------------------
# integration related
def integrate_vf_ivp(
    init_states: np.ndarray,
    t: np.ndarray,
    integration_direction: str,
    f: Callable,
    args: Optional[Tuple] = None,
    interpolation_num: int = 250,
    average: bool = True,
    sampling: str = "arc_length",
    verbose: bool = False,
    disable: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrating along vector field function using the initial value problem solver from scipy.integrate.

    Args:
        init_states: Initial states of the system.
        t: Time points to integrate the system over.
        integration_direction: The direction of integration.
        f: The vector field function of the system.
        args: Additional arguments to pass to the vector field function.
        interpolation_num: Number of time points to interpolate the trajectories over.
        average: Whether to average the trajectories.
        sampling: The method of sampling points along a trajectory.
        verbose: Whether to print the integration time.
        disable: Whether to disable the progress bar.

    Returns:
        The time and trajectories of the system.
    """

    # TODO: rewrite this function with the Trajectory class
    if init_states.ndim == 1:
        init_states = init_states[None, :]
    n_cell, n_feature = init_states.shape
    max_step = np.abs(t[-1] - t[0]) / interpolation_num

    T, Y, SOL = [], [], []

    if interpolation_num is not None and integration_direction == "both":
        interpolation_num = interpolation_num * 2

    for i in tqdm(range(n_cell), desc="integration with ivp solver", disable=disable):
        y0 = init_states[i, :]
        ivp_f, ivp_f_event = (
            lambda t, x: f(x),
            lambda t, x: np.all(abs(f(x)) < 1e-5) - 1,
            # np.linalg.norm(np.abs(f(x))) - 1e-5 if velocity on all dimension is less than 1e-5
        )
        ivp_f_event.terminal = True

        if verbose:
            print("\nintegrating cell ", i, "; Initial state: ", init_states[i, :])
        if integration_direction == "forward":
            y_ivp = solve_ivp(
                ivp_f,
                [t[0], t[-1]],
                y0,
                events=ivp_f_event,
                args=args,
                max_step=max_step,
                dense_output=True,
            )
            y, t_trans, sol = y_ivp.y, y_ivp.t, y_ivp.sol
        elif integration_direction == "backward":
            y_ivp = solve_ivp(
                ivp_f,
                [-t[0], -t[-1]],
                y0,
                events=ivp_f_event,
                args=args,
                max_step=max_step,
                dense_output=True,
            )
            y, t_trans, sol = y_ivp.y, y_ivp.t, y_ivp.sol
        elif integration_direction == "both":
            y_ivp_f = solve_ivp(
                ivp_f,
                [t[0], t[-1]],
                y0,
                events=ivp_f_event,
                args=args,
                max_step=max_step,
                dense_output=True,
            )
            y_ivp_b = solve_ivp(
                ivp_f,
                [-t[0], -t[-1]],
                y0,
                events=ivp_f_event,
                args=args,
                max_step=max_step,
                dense_output=True,
            )
            y, t_trans = (
                np.hstack((y_ivp_b.y[::-1, :], y_ivp_f.y)),
                np.hstack((y_ivp_b.t[::-1], y_ivp_f.t)),
            )
            sol = [y_ivp_b.sol, y_ivp_f.sol]
        else:
            raise Exception("both, forward, backward are the only valid direction argument strings")

        T.append(t_trans)
        Y.append(y)
        SOL.append(sol)

        if verbose:
            print("\nintegration time: ", len(t_trans))

    trajs = [Trajectory(X=Y[i], t=T[i], sort=False) for i in range(n_cell)]

    if sampling == "arc_length":
        for i in tqdm(
            range(n_cell),
            desc="uniformly sampling points along a trajectory",
            disable=disable,
        ):
            trajs[i].archlength_sampling(
                SOL[i],
                interpolation_num=interpolation_num,
                integration_direction=integration_direction,
            )

        t, Y = [traj.t for traj in trajs], [traj.X for traj in trajs]
    elif sampling == "logspace":
        for i in tqdm(
            range(n_cell),
            desc="sampling points along a trajectory in logspace",
            disable=disable,
        ):
            trajs[i].logspace_sampling(
                SOL[i],
                interpolation_num=interpolation_num,
                integration_direction=integration_direction,
            )

        t, Y = [traj.t for traj in trajs], [traj.X for traj in trajs]
    elif sampling == "uniform_indices":
        t_uniq = np.unique(np.hstack(T))
        if len(t_uniq) > interpolation_num:
            valid_t_trans = np.hstack([0, np.sort(t_uniq)])[
                (np.linspace(0, len(t_uniq), interpolation_num)).astype(int)
            ]

            if len(valid_t_trans) != interpolation_num:
                n_missed = interpolation_num - len(valid_t_trans)
                tmp = np.zeros(n_missed)

                for i in range(n_missed):
                    tmp[i] = (valid_t_trans[i] + valid_t_trans[i + 1]) / 2

                valid_t_trans = np.sort(np.hstack([tmp, valid_t_trans]))
        else:
            neg_tau, pos_tau = t_uniq[t_uniq < 0], t_uniq[t_uniq >= 0]
            t_0, t_1 = (
                -np.linspace(min(t_uniq), 0, interpolation_num),
                np.linspace(0, max(t_uniq), interpolation_num),
            )

            valid_t_trans = np.hstack((t_0, t_1))

        _Y = None
        if integration_direction == "both":
            neg_t_len = sum(valid_t_trans < 0)
        for i in tqdm(
            range(n_cell),
            desc="calculate solutions on the sampled time points in logspace",
            disable=disable,
        ):
            cur_Y = (
                SOL[i](valid_t_trans)
                if integration_direction != "both"
                else np.hstack(
                    (
                        SOL[i][0](valid_t_trans[:neg_t_len]),
                        SOL[i][1](valid_t_trans[neg_t_len:]),
                    )
                )
            )
            _Y = cur_Y if _Y is None else np.hstack((_Y, cur_Y))

        t, Y = valid_t_trans, _Y

        # TODO: this part is buggy, need to fix
        if n_cell > 1 and average:
            t_len = int(len(t) / n_cell)
            avg = np.zeros((n_feature, t_len))

            for i in range(t_len):
                avg[:, i] = np.mean(Y[:, np.arange(n_cell) * t_len + i], 1)
            Y = avg

        t = [t] * n_cell
        subarray_width = Y.shape[1] // n_cell
        Y = [Y[:, i * subarray_width : (i + 1) * subarray_width] for i in range(n_cell)]

    return t, Y


def integrate_sde(
    init_states: Union[np.ndarray, list],
    t: Union[float, np.ndarray],
    f: Callable,
    sigma: Union[float, np.ndarray, Callable],
    num_t: int = 100,
    **interp_kwargs,
) -> np.ndarray:
    """Calculate the trajectories by integrating a system of stochastic differential equations (SDEs) using the sdeint
    package.

    Args:
        init_states: Initial states of the system.
        t: Time points to integrate the system over.
        f: The vector field function of the system.
        sigma: The diffusion matrix of the system.
        num_t: Number of time points to interpolate the trajectories over.
        interp_kwargs: Additional keyword arguments to pass to the interpolation function.

    Returns:
        The trajectories of the system.
    """
    try:
        from sdeint import itoint
    except:
        raise ImportError("Please install sdeint using `pip install sdeint`")

    init_states = np.atleast_2d(init_states)
    n, d = init_states.shape

    if isarray(t):
        if len(t) == 1:
            t = np.linspace(0, t[0], num_t)
        elif len(t) == 2:
            t = np.linspace(t[0], t[-1], num_t)
    else:
        t = np.linspace(0, t, num_t)

    if callable(sigma):
        D_func = sigma
    elif isarray(sigma):
        if sigma.ndim == 1:
            sigma = np.diag(sigma)
        D_func = lambda x, t: sigma
    else:
        sigma = sigma * np.eye(d)
        D_func = lambda x, t: sigma

    trajs = []
    for y0 in init_states:
        y = itoint(lambda x, t: f(x), D_func, y0, t)
        trajs.append(y)
    if n == 1:
        trajs = trajs[0]

    return np.array(trajs)


def estimate_sigma(
    X: np.ndarray,
    V: np.ndarray,
    diff_multiplier: int = 1.0,
    num_nbrs: int = 30,
    nbr_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate the diffusion matrix of the system using the vector field and the data.

    Args:
        X: The array representing cell states.
        V: The array representing velocity.
        diff_multiplier: The multiplier for the diffusion matrix.
        num_nbrs: The number of nearest neighbors to use for the estimation.
        nbr_idx: The indices of the nearest neighbors.

    Returns:
        The estimated diffusion matrix.
    """
    from ..tools.utils import nearest_neighbors
    
    if nbr_idx is None:
        nbr_idx = nearest_neighbors(X, X, k=num_nbrs)

    MSD = np.zeros(X.shape)
    for i, idx in enumerate(nbr_idx):
        MSD[i] = np.std(X[idx], 0)

    msd = np.mean(MSD, 0)
    vm = np.abs(V).mean(0)
    tau = msd / vm
    sigma = msd / np.sqrt(tau) * np.sqrt(diff_multiplier)

    return sigma


def integrate_streamline(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    integration_direction: str,
    init_states: np.ndarray,
    interpolation_num: int = 100,
    average: bool = True,
) -> np.ndarray:
    """Use streamline's integrator to alleviate stacking of the solve_ivp. Need to update with the correct time.

    Args:
        X: The x-coordinates of the grid.
        Y: The y-coordinates of the grid.
        U: The x-components of the velocity.
        V: The y-components of the velocity.
        integration_direction: The direction of integration.
        init_states: The initial states of the system.
        interpolation_num: The number of time points to interpolate the trajectories over.
        average: Whether to average the trajectories.

    Returns:
        The time and trajectories of the system.
    """
    import matplotlib.pyplot as plt

    n_cell = init_states.shape[0]

    res = np.zeros((n_cell * interpolation_num, 2))
    j = -1  # this index will become 0 when the first trajectory found

    for i in tqdm(range(n_cell), "integration with streamline"):
        strm = plt.streamplot(
            X,
            Y,
            U,
            V,
            start_points=init_states[i, None],
            integration_direction=integration_direction,
            density=100,
        )
        strm_res = np.array(strm.lines.get_segments()).reshape((-1, 2))

        if len(strm_res) == 0:
            continue
        else:
            j += 1
        t = np.arange(strm_res.shape[0])
        t_linspace = np.linspace(t[0], t[-1], interpolation_num)
        f = interpolate.interp1d(t, strm_res.T)

        cur_rng = np.arange(j * interpolation_num, (j + 1) * interpolation_num)
        res[cur_rng, :] = f(t_linspace).T

    res = res[: cur_rng[-1], :]  # remove all empty trajectories
    n_cell = int(res.shape[0] / interpolation_num)

    if n_cell > 1 and average:
        t_len = len(t_linspace)
        avg = np.zeros((t_len, 2))

        for i in range(t_len):
            cur_rng = np.arange(n_cell) * t_len + i
            avg[i, :] = np.mean(res[cur_rng, :], 0)

        res = avg

    plt.close()

    return t_linspace, res


# ---------------------------------------------------------------------------------------------------
# fate related
def fetch_exprs(
    adata: AnnData,
    basis: str,
    layer: str,
    genes: Union[str, list],
    time: str,
    mode: str,
    project_back_to_high_dim: bool,
    traj_ind: int,
) -> Tuple:
    """Fetch the expression data for the given genes and time points.

    Args:
        adata: The AnnData object.
        basis: Target basis to fetch.
        layer: Target layer to fetch.
        genes: Target genes to consider.
        time: The time information.
        mode: The mode of the trajectory.
        project_back_to_high_dim: Whether to project the data back to high dimension.
        traj_ind: The index of the trajectory.

    Returns:
        The expression data for the given genes and time points.
    """
    from ..tools.utils import log1p_
    if type(genes) != list:
        genes = list(genes)

    prefix = "LAP_" if mode.lower() == "lap" else "fate_"
    if basis is not None:
        traj_key = prefix + basis
    else:
        traj_key = prefix if layer == "X" else prefix + layer

    time = adata.obs[time].values if mode == "pseudotime" else adata.uns[traj_key]["t"]
    if type(time) == list:
        time = time[traj_ind]

    if mode.lower() not in ["vector_field", "lap"]:
        valid_genes = list(sorted(set(genes).intersection(adata.var.index), key=genes.index))

        if layer == "X":
            exprs = adata[np.isfinite(time), :][:, valid_genes].X
        elif layer in adata.layers.keys():
            exprs = adata[np.isfinite(time), :][:, valid_genes].layers[layer]
            exprs = log1p_(adata, exprs)
        elif layer == "protein":  # update subset here
            exprs = adata[np.isfinite(time), :][:, valid_genes].obsm[layer]
        else:
            raise Exception(f"The {layer} you passed in is not existed in the adata object.")
    else:
        fate_genes = adata.uns[traj_key]["genes"]
        valid_genes = list(sorted(set(genes).intersection(fate_genes), key=genes.index))

        if basis is not None:
            if project_back_to_high_dim:
                exprs = adata.uns[traj_key]["exprs"]
                if type(exprs) == list:
                    exprs = exprs[traj_ind]
                exprs = exprs[np.isfinite(time), :][:, fate_genes.get_indexer(valid_genes)]
            else:
                exprs = adata.uns[traj_key]["prediction"]
                if type(exprs) == list:
                    exprs = exprs[traj_ind]
                exprs = exprs.T[np.isfinite(time), :]
                valid_genes = [basis + "_" + str(i) for i in np.arange(exprs.shape[1])]
        else:
            exprs = adata.uns[traj_key]["prediction"]
            if type(exprs) == list:
                exprs = exprs[traj_ind]
            exprs = exprs.T[np.isfinite(time), adata.var.index.get_indexer(valid_genes)]

    time = np.array(time)[np.isfinite(time)]

    return exprs, valid_genes, time


# ---------------------------------------------------------------------------------------------------
# perturbation related


def z_score(X: np.ndarray, axis: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the z-score of the given data.

    Args:
        X: The data to calculate the z-score for.
        axis: The axis to calculate the z-score over.

    Returns:
        The z-score of the data.
    """
    s = X.std(axis)
    m = X.mean(axis)
    Z = ((X.T - m) / s).T
    return Z, m, s


def z_score_inv(Z: Union[float, List, np.ndarray], m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """The inverse operation of z-score calculation.

    Args:
        Z: The z-scored data.
        m: The mean of the original data.
        s: The standard deviation of the original data.

    Returns:
        The original data reconstructed from the z-scored data.
    """
    if isarray(Z):
        X = (Z.T * s + m).T
    else:
        X = Z * s + m
    return X


# ---------------------------------------------------------------------------------------------------
# state graph related


def get_path(Pr: np.ndarray, i: int, j: int) -> List:
    """Retrieve the shortest path from node i to node j in given graph.

    Args:
        Pr: The graph.
        i: The start node.
        j: The end node.

    Returns:
        The shortest path from node i to node j.
    """
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]


# ---------------------------------------------------------------------------------------------------
# least action path related


def interp_second_derivative(
    t: np.ndarray,
    f: np.ndarray,
    num: int = 5e2,
    interp_kind="cubic",
    **interp_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate f(t) and calculate the discrete second derivative using:
        d^2 f / dt^2 = (f(x+h1) - 2f(x) + f(x-h2)) / (h1 * h2)

    Args:
        t: The time points.
        f: The function values corresponding to the time points.
        num: The number of points to interpolate to.
        interp_kind: The kind of interpolation to use.
        interp_kwargs: Additional keyword arguments to pass to the interpolation function.

    Returns:
        The interpolated time points and the discrete second derivative.
    """
    t_ = np.linspace(t[0], t[-1], int(num))
    f_ = interpolate.interp1d(t, f, kind=interp_kind, **interp_kwargs)(t_)

    dt = np.diff(t_)
    df = np.diff(f_)
    t_ = t_[1:-1]

    d2fdt2 = np.zeros(len(t_))
    for i in range(len(t_)):
        d2fdt2[i] = (df[i + 1] - df[i]) / (dt[i + 1] * dt[i])

    return t_, d2fdt2


def interp_curvature(
    t: np.ndarray,
    f: np.ndarray,
    num: int = 5e2,
    interp_kind="cubic",
    **interp_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate f(t) and calculate the curvature.

    Args:
        t: The time points.
        f: The function values corresponding to the time points.
        num: The number of points to interpolate to.
        interp_kind: The kind of interpolation to use.
        interp_kwargs: Additional keyword arguments to pass to the interpolation function.

    Returns:
        The interpolated time points and the curvature.
    """
    t_ = np.linspace(t[0], t[-1], int(num))
    f_ = interpolate.interp1d(t, f, kind=interp_kind, **interp_kwargs)(t_)

    dt = np.diff(t_)
    df = np.diff(f_)
    dfdt_ = df / dt

    t_ = t_[1:-1]
    d2fdt2 = np.zeros(len(t_))
    dfdt = np.zeros(len(t_))
    for i in range(len(t_)):
        dfdt[i] = (dfdt_[i] + dfdt_[i + 1]) / 2
        d2fdt2[i] = (df[i + 1] - df[i]) / (dt[i + 1] * dt[i])

    cur = d2fdt2 / (1 + dfdt * dfdt) ** 1.5

    return t_, cur


def kneedle_difference(t: np.ndarray, f: np.ndarray, type: str = "decrease") -> np.ndarray:
    """Calculate the difference between the function and the diagonal line.

    Args:
        t: The time points.
        f: The function values corresponding to the time points.
        type: The type of function to use.

    Returns:
        The difference between the function and the diagonal line.
    """
    if type == "decrease":
        diag_line = lambda x: -x + 1
    elif type == "increase":
        diag_line = lambda x: x
    else:
        raise NotImplementedError(f"Unsupported function type {type}")

    t_ = normalize(t)
    f_ = normalize(f)
    res = np.abs(f_ - diag_line(t_))
    return res


def find_elbow(
    T: np.ndarray,
    F: np.ndarray,
    method: str = "kneedle",
    order: int = 1,
    **kwargs,
) -> int:
    """Find the elbow of the given function.

    Args:
        T: The time points.
        F: The function values corresponding to the time points.
        method: The method to use for finding the elbow.
        order: The order of the elbow.
        kwargs: Additional keyword arguments to pass to the elbow finding function.

    Returns:
        The index of the elbow.
    """
    i_elbow = None
    if method == "hessian":
        T_ = normalize(T)
        F_ = normalize(F)
        tol = kwargs.pop("tol", 2)
        t_, der = interp_second_derivative(T_, F_, **kwargs)

        found = False
        for i, t in enumerate(t_[::order]):
            if der[::order][i] > tol:
                i_elbow = np.argmin(np.abs(T_ - t))
                found = True
                break

        if not found:
            main_warning("The elbow was not found.")

    elif method == "curvature":
        T_ = normalize(T)
        F_ = normalize(F)
        t_, cur = interp_curvature(T_, F_, **kwargs)

        i_elbow = np.argmax(cur)

    elif method == "kneedle":
        type = "decrease" if order == -1 else "increase"
        res = kneedle_difference(T, F, type=type)
        i_elbow = np.argmax(res)
    else:
        raise NotImplementedError(f"The method {method} is not supported.")

    return i_elbow
