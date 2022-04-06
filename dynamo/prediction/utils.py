from typing import Callable

# from anndata._core.views import ArrayView
import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp
from tqdm import tqdm

from ..dynamo_logger import main_warning
from ..tools.utils import log1p_, nearest_neighbors
from ..utils import isarray, normalize

# import scipy.sparse as sp
from ..vectorfield.topography import dup_osc_idx_iter

# ---------------------------------------------------------------------------------------------------
# initial state related


def init_r0_pulse(r, l, k):
    """calculate initial total RNA via ODE formula of RNA kinetics for one-shot/kinetics experiment

    Parameters
    ----------
        r:
            total RNA at current time point.
        l:
            labeled RNA at current time point.
        k:
            $k = 1 - e^{-\gamma t}$

    Returns
    -------
        r0:
            The intial total RNA at the beginning of the one-shot or kinetics experiment.
    """
    r0 = (r - l) / (1 - k)

    return r0


def init_l0_chase(l, gamma, t):
    """calculate initial total RNA via ODE formula of RNA kinetics for degradation experiment

    Note that this approach only estimate the initial labeled RNA based on first-order decay model. To get the intial r0
    we can also assume cells with extreme total RNA as steady state cells and use that to estimate transcription rate.

    Parameters
    ----------
        l:
            labeled RNA(s)
        gamma:
            degradation rate(s)
        t:
            labeling time(s)

    Returns
    -------
        l0:
            The initial labeled RNA at the beginning of a degradation experiment.
    """
    l0 = l / np.exp(-gamma * t)

    return l0


# ---------------------------------------------------------------------------------------------------
# integration related


def integrate_vf_ivp(
    init_states,
    t,
    integration_direction,
    f: Callable,
    args=None,
    interpolation_num=250,
    average=True,
    sampling="arc_length",
    verbose=False,
    disable=False,
):
    """integrating along vector field function using the initial value problem solver from scipy.integrate"""

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

    if sampling == "arc_length":
        Y_, t_ = [None] * n_cell, [None] * n_cell
        for i in tqdm(
            range(n_cell),
            desc="uniformly sampling points along a trajectory",
            disable=disable,
        ):
            tau, x = T[i], Y[i].T

            idx = dup_osc_idx_iter(x, max_iter=100, tol=x.ptp(0).mean() / 1000)[0]

            # idx = dup_osc_idx_iter(x)
            x = x[:idx]
            _, arclen, _ = remove_redundant_points_trajectory(x, tol=1e-4, output_discard=True)
            arc_stepsize = arclen / interpolation_num
            cur_Y, alen, t_[i] = arclength_sampling(x, step_length=arc_stepsize, t=tau[:idx])

            if integration_direction == "both":
                neg_t_len = sum(np.array(t_[i]) < 0)

            odeint_cur_Y = (
                SOL[i](t_[i])
                if integration_direction != "both"
                else np.hstack(
                    (
                        SOL[i][0](t_[i][:neg_t_len]),
                        SOL[i][1](t_[i][neg_t_len:]),
                    )
                )
            )
            Y_[i] = odeint_cur_Y

        Y, t = Y_, t_
    elif sampling == "logspace":
        Y_, t_ = [None] * n_cell, [None] * n_cell
        for i in tqdm(
            range(n_cell),
            desc="sampling points along a trajectory in logspace",
            disable=disable,
        ):
            tau, x = T[i], Y[i].T
            neg_tau, pos_tau = tau[tau < 0], tau[tau >= 0]

            if len(neg_tau) > 0:
                t_0, t_1 = (
                    -(
                        np.logspace(
                            0,
                            np.log10(abs(min(neg_tau)) + 1),
                            interpolation_num,
                        )
                    )
                    - 1,
                    np.logspace(0, np.log10(max(pos_tau) + 1), interpolation_num) - 1,
                )
                t_[i] = np.hstack((t_0[::-1], t_1))
            else:
                t_[i] = np.logspace(0, np.log10(max(tau) + 1), interpolation_num) - 1

            if integration_direction == "both":
                neg_t_len = sum(np.array(t_[i]) < 0)

            odeint_cur_Y = (
                SOL[i](t_[i])
                if integration_direction != "both"
                else np.hstack(
                    (
                        SOL[i][0](t_[i][:neg_t_len]),
                        SOL[i][1](t_[i][neg_t_len:]),
                    )
                )
            )
            Y_[i] = odeint_cur_Y

        Y, t = Y_, t_
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

        # this part is buggy, need to fix
        if n_cell > 1 and average:
            t_len = int(len(t) / n_cell)
            avg = np.zeros((n_feature, t_len))

            for i in range(t_len):
                avg[:, i] = np.mean(Y[:, np.arange(n_cell) * t_len + i], 1)
            Y = avg

        Y = Y.T

    return t, Y


def integrate_sde(init_states, t, f, sigma, num_t=100, **interp_kwargs):
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


def estimate_sigma(X, V, diff_multiplier=1.0, num_nbrs=30, nbr_idx=None):
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
    X,
    Y,
    U,
    V,
    integration_direction,
    init_states,
    interpolation_num=100,
    average=True,
):
    """use streamline's integrator to alleviate stacking of the solve_ivp. Need to update with the correct time."""
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
# arc curve related
def remove_redundant_points_trajectory(X, tol=1e-4, output_discard=False):
    """remove consecutive data points that are too close to each other."""
    X = np.atleast_2d(X)
    discard = np.zeros(len(X), dtype=bool)
    if X.shape[0] > 1:
        for i in range(len(X) - 1):
            dist = np.linalg.norm(X[i + 1] - X[i])
            if dist < tol:
                discard[i + 1] = True
        X = X[~discard]

    arclength = 0

    x0 = X[0]
    for i in range(1, len(X)):
        tangent = X[i] - x0 if i == 1 else X[i] - X[i - 1]
        d = np.linalg.norm(tangent)

        arclength += d

    if output_discard:
        return (X, arclength, discard)
    else:
        return (X, arclength)


def arclength_sampling(X, step_length, t=None):
    """uniformly sample data points on an arc curve that generated from vector field predictions."""
    Y = []
    x0 = X[0]
    T = [] if t is not None else None
    t0 = t[0] if t is not None else None
    i = 1
    terminate = False
    arclength = 0

    while i < len(X) - 1 and not terminate:
        L = 0
        for j in range(i, len(X)):
            tangent = X[j] - x0 if j == i else X[j] - X[j - 1]
            d = np.linalg.norm(tangent)
            if L + d >= step_length:
                x = x0 if j == i else X[j - 1]
                y = x + (step_length - L) * tangent / d
                if t is not None:
                    tau = t0 if j == i else t[j - 1]
                    tau += (step_length - L) / d * (t[j] - tau)
                    T.append(tau)
                    t0 = tau
                Y.append(y)
                x0 = y
                i = j
                break
            else:
                L += d
        if j == len(X) - 1:
            i += 1
        arclength += step_length
        if L + d < step_length:
            terminate = True

    if T is not None:
        return np.array(Y), arclength, T
    else:
        return np.array(Y), arclength


def arclength_sampling_n(X, num, t=None):
    arclen = np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))
    arclen = np.hstack((0, arclen))

    z = np.linspace(arclen[0], arclen[-1], num)
    X_ = interpolate.interp1d(arclen, X, axis=0)(z)
    if t is not None:
        t_ = interpolate.interp1d(arclen, t)(z)
        return X_, arclen[-1], t_
    else:
        return X_, arclen[-1]


# ---------------------------------------------------------------------------------------------------
# trajectory related
def pca_to_expr(X, PCs, mean=0, func=None):
    # reverse project from PCA back to raw expression space
    if PCs.shape[1] == X.shape[1]:
        exprs = X @ PCs.T + mean
        if func is not None:
            exprs = func(exprs)
    else:
        raise Exception("PCs dim 1 (%d) does not match X dim 1 (%d)." % (PCs.shape[1], X.shape[1]))
    return exprs


def expr_to_pca(expr, PCs, mean=0, func=None):
    # project from raw expression space to PCA
    if PCs.shape[0] == expr.shape[1]:
        X = (expr - mean) @ PCs
        if func is not None:
            X = func(X)
    else:
        raise Exception("PCs dim 1 (%d) does not match X dim 1 (%d)." % (PCs.shape[0], expr.shape[1]))
    return X


# ---------------------------------------------------------------------------------------------------
# fate related
def fetch_exprs(adata, basis, layer, genes, time, mode, project_back_to_high_dim, traj_ind):
    import pandas as pd

    prefix = "LAP_" if mode.lower() == "lap" else "fate_"
    if basis is not None:
        traj_key = prefix + basis
    else:
        traj_key = prefix if layer == "X" else prefix + layer

    time = adata.obs[time].values if mode == "pseudotime" else adata.uns[traj_key]["t"]
    if type(time) == list:
        time = time[traj_ind]

    if mode.lower() not in ["vector_field", "lap"]:
        valid_genes = list(set(genes).intersection(adata.var.index))

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
        valid_genes = list(set(genes).intersection(fate_genes))

        if basis is not None:
            if project_back_to_high_dim:
                exprs = adata.uns[traj_key]["exprs"]
                if type(exprs) == list:
                    exprs = exprs[traj_ind]
                exprs = exprs[np.isfinite(time), :][:, pd.Series(fate_genes).isin(valid_genes)]
            else:
                exprs = adata.uns[traj_key]["prediction"]
                if type(exprs) == list:
                    exprs = exprs[traj_ind]
                exprs = exprs[np.isfinite(time), :]
                valid_genes = [basis + "_" + str(i) for i in np.arange(exprs.shape[1])]
        else:
            exprs = adata.uns[traj_key]["prediction"]
            if type(exprs) == list:
                exprs = exprs[traj_ind]
            exprs = exprs[np.isfinite(time), pd.Series(fate_genes).isin(valid_genes)]

    time = time[np.isfinite(time)]

    return exprs, valid_genes, time


# ---------------------------------------------------------------------------------------------------
# perturbation related


def z_score(X, axis=1):
    s = X.std(axis)
    m = X.mean(axis)
    Z = ((X.T - m) / s).T
    return Z, m, s


def z_score_inv(Z, m, s):
    if isarray(Z):
        X = (Z.T * s + m).T
    else:
        X = Z * s + m
    return X


# ---------------------------------------------------------------------------------------------------
# state graph related


def get_path(Pr, i, j):
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]


# ---------------------------------------------------------------------------------------------------
# least action path related


def interp_second_derivative(t, f, num=5e2, interp_kind="cubic", **interp_kwargs):
    """
    interpolate f(t) and calculate the discrete second derivative using:
        d^2 f / dt^2 = (f(x+h1) - 2f(x) + f(x-h2)) / (h1 * h2)
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


def interp_curvature(t, f, num=5e2, interp_kind="cubic", **interp_kwargs):
    """"""
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


def kneedle_difference(t, f, type="decrease"):
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


def find_elbow(T, F, method="kneedle", order=1, **kwargs):
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
