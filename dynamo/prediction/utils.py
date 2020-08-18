from tqdm import tqdm
import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp
from scipy.sparse import issparse
from ..vectorfield.topography import dup_osc_idx_iter
from ..tools.utils import log1p_

def integrate_vf_ivp(
    init_states, t, args, integration_direction, f, interpolation_num=250, average=True, sampling='arc_length',
    verbose=False, disable=False,
):
    """integrating along vector field function using the initial value problem solver from scipy.integrate"""

    if init_states.ndim == 1: init_states = init_states[None, :]
    n_cell, n_feature = init_states.shape
    max_step = np.abs(t[-1] - t[0]) / interpolation_num

    T, Y, SOL = [], [], []

    for i in tqdm(range(n_cell), desc="integration with ivp solver", disable=disable):
        y0 = init_states[i, :]
        ivp_f, ivp_f_event = (
            lambda t, x: f(x),
            lambda t, x: np.all(abs(f(x)) < 1e-5) - 1, # np.linalg.norm(np.abs(f(x))) - 1e-5 if velocity on all dimension is less than 1e-5
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

            if interpolation_num is not None:
                interpolation_num = interpolation_num * 2
        else:
            raise Exception(
                "both, forward, backward are the only valid direction argument strings"
            )

        T.append(t_trans)
        Y.append(y)
        SOL.append(sol)

        if verbose:
            print("\nintegration time: ", len(t_trans))

    if sampling == 'arc_length':
        Y_, t_ = [None] * n_cell, [None] * n_cell
        for i in tqdm(range(n_cell), desc="uniformly sampling points along a trajectory", disable=disable):
            tau, x = T[i], Y[i].T
            idx = dup_osc_idx_iter(x, max_iter=100, tol=x.ptp(0).mean() / 1000)[0]
            # idx = dup_osc_idx_iter(x)
            x = x[:idx]
            _, arclen, _ = remove_redundant_points_trajectory(x, tol=1e-4, output_discard=True)
            arc_stepsize = arclen / interpolation_num
            cur_Y, alen, t_[i] = arclength_sampling(x, step_length=arc_stepsize, t=tau[:idx])

            if integration_direction == "both":
                neg_t_len = sum(np.array(t_[i]) < 0)

            odeint_cur_Y = SOL[i](t_[i]) if integration_direction != "both" \
                else np.hstack(
                (
                    SOL[i][0](t_[i][:neg_t_len]),
                    SOL[i][1](t_[i][neg_t_len:]),
                )
            )
            Y_[i] = odeint_cur_Y

        Y, t = Y_, t_
    elif sampling == 'logspace':
        Y_, t_ = [None] * n_cell, [None] * n_cell
        for i in tqdm(range(n_cell), desc="uniformly sampling points along a trajectory", disable=disable):
            tau, x = T[i], Y[i].T
            t_[i] = np.logspace(0, np.log10(max(tau) + 1), interpolation_num) - 1

            if integration_direction == "both":
                neg_t_len = sum(np.array(t_[i]) < 0)

            odeint_cur_Y = SOL[i](t_[i]) if integration_direction != "both" \
                else np.hstack(
                (
                    SOL[i][0](t_[i][:neg_t_len]),
                    SOL[i][1](t_[i][neg_t_len:]),
                )
            )
            Y_[i] = odeint_cur_Y

        Y, t = Y_, t_
    elif sampling == 'uniform_indices':
        t_uniq = np.unique(np.hstack(T))
        if len(t_uniq) > interpolation_num:
            valid_t_trans = np.hstack([0, np.sort(t_uniq)])[(np.linspace(0, len(t_uniq), interpolation_num)).astype(int)]

            if len(valid_t_trans) != interpolation_num:
                n_missed = interpolation_num - len(valid_t_trans)
                tmp = np.zeros(n_missed)

                for i in range(n_missed):
                    tmp[i] = (valid_t_trans[i] + valid_t_trans[i + 1]) / 2

                valid_t_trans = np.sort(np.hstack([tmp, valid_t_trans]))
        else:
            valid_t_trans = np.logspace(0, np.log10(max(t_uniq) + 1), interpolation_num) - 1

        _Y = None
        if integration_direction == "both":
            neg_t_len = sum(valid_t_trans < 0)
        for i in tqdm(range(n_cell), desc="calculate solutions on the sampled time points", disable=disable):
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

        if n_cell > 1 and average:
            t_len = int(len(t) / n_cell)
            avg = np.zeros((n_feature, t_len))

            for i in range(t_len):
                avg[:, i] = np.mean(Y[:, np.arange(n_cell) * t_len + i], 1)
            Y = avg

        Y = Y.T

    return t, Y

def integrate_streamline(
    X, Y, U, V, integration_direction, init_states, interpolation_num=100, average=True
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
        l = 0
        for j in range(i, len(X)):
            tangent = X[j] - x0 if j == i else X[j] - X[j - 1]
            d = np.linalg.norm(tangent)
            if l + d >= step_length:
                x = x0 if j == i else X[j - 1]
                y = x + (step_length - l) * tangent / d
                if t is not None:
                    tau = t0 if j == i else t[j - 1]
                    tau += (step_length - l) / d * (t[j] - tau)
                    T.append(tau)
                    t0 = tau
                Y.append(y)
                x0 = y
                i = j
                break
            else:
                l += d
        if j == len(X) - 1: i += 1
        arclength += step_length
        if l + d < step_length:
            terminate = True

    if T is not None:
        return np.array(Y), arclength, T
    else:
        return np.array(Y), arclength


# ---------------------------------------------------------------------------------------------------
# fate related
def fetch_exprs(adata, basis, layer, genes, time, mode, project_back_to_high_dim):
    import pandas as pd

    if basis is not None:
        fate_key = "fate_" + basis
    else:
        fate_key = "fate" if layer == "X" else "fate_" + layer

    time = (
        adata.obs[time].values
        if mode != "vector_field"
        else adata.uns[fate_key]["t"]
    )

    if mode != "vector_field":
        valid_genes = list(set(genes).intersection(adata.var.index))

        if layer == "X":
            exprs = adata[np.isfinite(time), :][:, valid_genes].X
        elif layer in adata.layers.keys():
            exprs = adata[np.isfinite(time), :][:, valid_genes].layers[layer]
            exprs = log1p_(adata, exprs)
        elif layer == "protein":  # update subset here
            exprs = adata[np.isfinite(time), :][:, valid_genes].obsm[layer]
        else:
            raise Exception(
                f"The {layer} you passed in is not existed in the adata object."
            )
    else:
        fate_genes = adata.uns[fate_key]["genes"]
        valid_genes = list(set(genes).intersection(fate_genes))

        if basis is not None:
            if project_back_to_high_dim:
                exprs = adata.uns[fate_key]["high_prediction"]
                exprs = exprs[np.isfinite(time), pd.Series(fate_genes).isin(valid_genes)]
            else:
                exprs = adata.uns[fate_key]["prediction"][np.isfinite(time), :]
                valid_genes = [basis + "_" + str(i) for i in np.arange(exprs.shape[1])]
        else:
            exprs = adata.uns[fate_key]["prediction"][np.isfinite(time), pd.Series(fate_genes).isin(valid_genes)]

    time = time[np.isfinite(time)]

    return exprs, valid_genes, time


def fetch_states(adata, init_states, init_cells, basis, layer, average, t_end):
    if basis is not None:
        vf_key = "VecFld_" + basis
    else:
        vf_key = "VecFld"
    VecFld = adata.uns[vf_key]['VecFld']
    X = VecFld['X']
    valid_genes = None

    if init_states is None and init_cells is None:
        raise Exception("Either init_state or init_cells should be provided.")
    elif init_states is None and init_cells is not None:
        if type(init_cells) == str:
            init_cells = [init_cells]
        intersect_cell_names = sorted(
            set(init_cells).intersection(adata.obs_names),
            key=lambda x: list(init_cells).index(x),
        )
        _cell_names = (
            init_cells if len(intersect_cell_names) == 0 else intersect_cell_names
        )

        if basis is not None:
            init_states = adata[_cell_names].obsm["X_" + basis].copy()
            if len(_cell_names) == 1:
                init_states = init_states.reshape((1, -1))
            VecFld = adata.uns["VecFld_" + basis]["VecFld"]
            X = adata.obsm["X_" + basis]

            valid_genes = [
                basis + "_" + str(i) for i in np.arange(init_states.shape[1])
            ]
        else:
            # valid_genes = list(set(genes).intersection(adata.var_names[adata.var.use_for_velocity]) if genes is not None \
            #     else adata.var_names[adata.var.use_for_velocity]
            # ----------- enable the function to only only a subset genes -----------

            vf_key = "VecFld" if layer == "X" else "VecFld_" + layer
            valid_genes = adata.uns[vf_key]["genes"]
            init_states = (
                adata[_cell_names, :][:, valid_genes].X
                if layer == "X"
                else log1p_(adata, adata[_cell_names, :][:, valid_genes].layers[layer])
            )
            if issparse(init_states):
                init_states = init_states.A
            if len(_cell_names) == 1:
                init_states = init_states.reshape((1, -1))

            if layer == "X":
                VecFld = adata.uns["VecFld"]["VecFld"]
                X = adata[:, valid_genes].X
            else:
                VecFld = adata.uns["VecFld_" + layer]["VecFld"]
                X = log1p_(adata, adata[:, valid_genes].layers[layer])

    if init_states.shape[0] > 1 and average in ["origin", 'trajectory', True]:
        init_states = init_states.mean(0).reshape((1, -1))

    if t_end is None:
        t_end = getTend(X, VecFld["V"])

    if issparse(init_states):
        init_states = init_states.A

    return init_states, VecFld, t_end, valid_genes


def getTend(X, V):
    xmin, xmax = X.min(0), X.max(0)
    V_abs = np.abs(V)
    t_end = np.max(xmax - xmin) / np.percentile(V_abs[V_abs > 0], 1)

    return t_end


def getTseq(init_states, t_end, step_size=None):
    if step_size is None:
        max_steps = (
            int(max(7 / (init_states.shape[1] / 300), 4))
            if init_states.shape[1] > 300
            else 7
        )
        t_linspace = np.linspace(
            0, t_end, 10 ** (np.min([int(np.log10(t_end)), max_steps]))
        )
    else:
        t_linspace = np.arange(0, t_end + step_size, step_size)

    return t_linspace
