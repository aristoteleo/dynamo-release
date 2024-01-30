import re
import warnings
from typing import Callable, Iterable, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import functools

import matplotlib.pyplot as plt
import numpy as np
import scipy
from anndata import AnnData
from numpy import *
from scipy.integrate import odeint
from scipy.optimize import curve_fit, least_squares
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm

from ..dynamo_logger import main_info, main_warning
from .DDRTree import DDRTree
from .moments import calc_1nd_moment, strat_mom
from .utils import (
    get_data_for_kin_params_estimation,
    get_mapper,
    get_U_S_for_velocity_estimation,
    get_valid_bools,
    set_param_kinetic,
    set_param_ss,
    set_velocity,
)
from ..estimation.tsc.utils_moments import moments
from ..estimation.csc.velocity import ss_estimation, Velocity


def deprecated(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is deprecated and will be removed in a future release. "
            f"Please update your code to use the new replacement function.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------------------------------
# deprecated _dynamics_deprecated.py
@deprecated
def _dynamics(*args, **kwargs):
    _dynamics_legacy(*args, **kwargs)


def _dynamics_legacy(
    adata,
    tkey=None,
    filter_gene_mode="final",
    mode="moment",
    use_smoothed=True,
    group=None,
    protein_names=None,
    experiment_type=None,
    assumption_mRNA=None,
    assumption_protein="ss",
    NTR_vel=True,
    concat_data=False,
    log_unnormalized=True,
    one_shot_method="combined",
):
    """Inclusive model of expression dynamics considers splicing, metabolic labeling and protein translation. It supports
    learning high-dimensional velocity vector samples for droplet based (10x, inDrop, drop-seq, etc), scSLAM-seq, NASC-seq
    sci-fate, scNT-seq or cite-seq datasets.

    Args:
        adata: :class:`~anndata.AnnData`
            AnnData object.
        tkey: `str` or None (default: None)
            The column key for the time label of cells in .obs. Used for either "steady_state" or non-"steady_state" mode or `moment`
            mode  with labeled data.
        filter_gene_mode: `str` (default: `final`)
            The string for indicating which mode (one of, {'final', 'basic', 'no'}) of gene filter will be used.
        mode: `str` (default: `deterministic`)
            String indicates which estimation mode will be used. This parameter should be used in conjunction with assumption_mRNA.
            * Available options when the `assumption_mRNA` is 'ss' include:
            (1) 'linear_regression': The canonical method from the seminar RNA velocity paper based on deterministic ordinary
            differential equations;
            (2) 'gmm': The new generalized methods of moments from us that is based on master equations, similar to the
            "moment" mode in the excellent scvelo package;
            (3) 'negbin': The new method from us that models steady state RNA expression as a negative binomial distribution,
            also built upons on master equations.
            Note that all those methods require using extreme data points (except negbin) for the estimation. Extreme data points
            are defined as the data from cells where the expression of unspliced / spliced or new / total RNA, etc. are in the
            top or bottom, 5%, for example. `linear_regression` only considers the mean of RNA species (based on the deterministic
            ordinary different equations) while moment based methods (`gmm`, `negbin`) considers both first moment (mean) and
            second moment (uncentered variance) of RNA species (based on the stochastic master equations).
            * Available options when the `assumption_mRNA` is 'kinetic' include:
            (1) 'deterministic': The method based on deterministic ordinary differential equations;
            (2) 'stochastic' or `moment`: The new method from us that is based on master equations;
            Note that `kinetic` model implicitly assumes the `experiment_type` is not `conventional`. Thus `deterministic`,
            `stochastic` (equivalent to `moment`) models are only possible for the labeling experiments.
            A "model_selection" mode will be supported soon in which alpha, beta and gamma will be modeled as a function of time.
        use_smoothed: `bool` (default: `True`)
            Whether to use the smoothed data when calculating velocity for each gene. `use_smoothed` is only relevant when
            mode is `linear_regression` (and experiment_type and assumption_mRNA correspond to `conventional` and `ss` implicitly).
        group: `str` or None (default: `None`)
            The column key/name that identifies the grouping information (for example, clusters that correspond to different cell types)
            of cells. This will be used to estimate group-specific (i.e cell-type specific) kinetic parameters.
        protein_names: `List`
            A list of gene names corresponds to the rows of the measured proteins in the `X_protein` of the `obsm` attribute.
            The names have to be included in the adata.var.index.
        experiment_type: `str`
            single cell RNA-seq experiment type. Available options are:
            (1) 'conventional': conventional single-cell RNA-seq experiment;
            (2) 'deg': chase/degradation experiment;
            (3) 'kin': pulse/synthesis/kinetics experiment;
            (4) 'one-shot': one-shot kinetic experiment.
        assumption_mRNA: `str`
            Parameter estimation assumption for mRNA. Available options are:
            (1) 'ss': pseudo steady state;
            (2) 'kinetic' or None: degradation and kinetic data without steady state assumption.
            If no labelling data exists, assumption_mRNA will automatically set to be 'ss'. For one-shot experiment, assumption_mRNA
            is set to be None. However we will use steady state assumption to estimate parameters alpha and gamma either by a deterministic
            linear regression or the first order decay approach in line of the sci-fate paper.
        assumption_protein: `str`
            Parameter estimation assumption for protein. Available options are:
            (1) 'ss': pseudo steady state;
        NTR_vel: `bool` (default: `True`)
            Whether to use NTR (new/total ratio) velocity for labeling datasets.
        concat_data: `bool` (default: `False`)
            Whether to concatenate data before estimation. If your data is a list of matrices for each time point, this need to be set as True.
        log_unnormalized: `bool` (default: `True`)
            Whether to log transform the unnormalized data.

    Returns:
        adata: :class:`~anndata.AnnData`
            An updated AnnData object with estimated kinetic parameters and inferred velocity included.
    """

    if "use_for_dynamics" not in adata.var.columns and "pass_basic_filter" not in adata.var.columns:
        filter_gene_mode = "no"

    valid_ind = get_valid_bools(adata, filter_gene_mode)

    if mode == "moment" or (use_smoothed and len([i for i in adata.layers.keys() if i.startswith("M_")]) < 2):
        if experiment_type == "kin":
            use_smoothed = False
        else:
            moments(adata)

    valid_adata = adata[:, valid_ind].copy()
    if group is not None and group in adata.obs[group]:
        _group = adata.obs[group].unique()
    else:
        _group = ["_all_cells"]

    for cur_grp in _group:
        if cur_grp == "_all_cells":
            kin_param_pre = ""
            cur_cells_bools = np.ones(valid_adata.shape[0], dtype=bool)
            subset_adata = valid_adata[cur_cells_bools]
        else:
            kin_param_pre = group + "_" + cur_grp + "_"
            cur_cells_bools = (valid_adata.obs[group] == cur_grp).values
            subset_adata = valid_adata[cur_cells_bools]

        (
            U,
            Ul,
            S,
            Sl,
            P,
            US,
            S2,
            t,
            normalized,
            has_splicing,
            has_labeling,
            has_protein,
            ind_for_proteins,
            assumption_mRNA,
            exp_type,
        ) = get_data_for_kin_params_estimation(
            subset_adata,
            mode,
            use_smoothed,
            tkey,
            protein_names,
            experiment_type,
            log_unnormalized,
            NTR_vel,
        )

        if exp_type is not None:
            if experiment_type != exp_type:
                main_warning(
                    "dynamo detects the experiment type of your data as {}, but your input experiment_type "
                    "is {}".format(exp_type, experiment_type)
                )

            experiment_type = exp_type
            assumption_mRNA = "ss" if exp_type == "conventional" and mode == "deterministic" else None
            NTR_vel = False

        if mode == "moment" and experiment_type not in ["conventional", "kin"]:
            """
            # temporially convert to deterministic mode as moment mode for one-shot,
            degradation and other types of labeling experiment is ongoing."""

            mode = "deterministic"

        if mode == "deterministic" or (experiment_type != "kin" and mode == "moment"):
            est = ss_estimation(
                U=U,
                Ul=Ul,
                S=S,
                Sl=Sl,
                P=P,
                US=US,
                S2=S2,
                t=t,
                ind_for_proteins=ind_for_proteins,
                experiment_type=experiment_type,
                assumption_mRNA=assumption_mRNA,
                assumption_protein=assumption_protein,
                concat_data=concat_data,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if experiment_type in ["one-shot", "one_shot"]:
                    est.train(one_shot_method=one_shot_method)
                else:
                    est.train()

            alpha, beta, gamma, eta, delta = est.parameters.values()

            U, S = get_U_S_for_velocity_estimation(
                subset_adata,
                use_smoothed,
                has_splicing,
                has_labeling,
                log_unnormalized,
                NTR_vel,
            )
            vel = Velocity(estimation=est)
            vel_U = vel.vel_u(U)
            vel_S = vel.vel_s(U, S)
            vel_P = vel.vel_p(S, P)

            adata = set_velocity(
                adata,
                vel_U,
                vel_S,
                vel_P,
                _group,
                cur_grp,
                cur_cells_bools,
                valid_ind,
                ind_for_proteins,
            )

            adata = set_param_ss(
                adata,
                est,
                alpha,
                beta,
                gamma,
                eta,
                delta,
                experiment_type,
                _group,
                cur_grp,
                kin_param_pre,
                valid_ind,
                ind_for_proteins,
            )

        elif mode == "moment":
            adata, Est, t_ind = moment_model(adata, subset_adata, _group, cur_grp, log_unnormalized, tkey)
            t_ind += 1

            params, costs = Est.train()
            a, b, alpha_a, alpha_i, beta, gamma = (
                params[:, 0],
                params[:, 1],
                params[:, 2],
                params[:, 3],
                params[:, 4],
                params[:, 5],
            )

            def fbar(x_a, x_i, a, b):
                return b / (a + b) * x_a + a / (a + b) * x_i

            alpha = fbar(alpha_a, alpha_i, a, b)[:, None]

            params = {"alpha": alpha, "beta": beta, "gamma": gamma, "t": t}
            vel = Velocity(**params)

            U, S = get_U_S_for_velocity_estimation(
                subset_adata,
                use_smoothed,
                has_splicing,
                has_labeling,
                log_unnormalized,
                NTR_vel,
            )
            vel_U = vel.vel_u(U)
            vel_S = vel.vel_s(U, S)
            vel_P = vel.vel_p(S, P)

            adata = set_velocity(
                adata,
                vel_U,
                vel_S,
                vel_P,
                _group,
                cur_grp,
                cur_cells_bools,
                valid_ind,
                ind_for_proteins,
            )

            adata = set_param_kinetic(
                adata,
                alpha,
                a,
                b,
                alpha_a,
                alpha_i,
                beta,
                gamma,
                kin_param_pre,
                _group,
                cur_grp,
                valid_ind,
            )
            # add protein related parameters in the moment model below:
        elif mode == "model_selection":
            main_warning("Not implemented yet.")

    if group is not None and group in adata.obs[group]:
        uns_key = group + "_dynamics"
    else:
        uns_key = "dynamics"

    if has_splicing and has_labeling:
        adata.layers["X_U"], adata.layers["X_S"] = (
            adata.layers["X_uu"] + adata.layers["X_ul"],
            adata.layers["X_su"] + adata.layers["X_sl"],
        )

    adata.uns[uns_key] = {
        "t": t,
        "group": group,
        "asspt_mRNA": assumption_mRNA,
        "experiment_type": experiment_type,
        "normalized": normalized,
        "mode": mode,
        "has_splicing": has_splicing,
        "has_labeling": has_labeling,
        "has_protein": has_protein,
        "use_smoothed": use_smoothed,
        "NTR_vel": NTR_vel,
        "log_unnormalized": log_unnormalized,
    }

    return adata


# ---------------------------------------------------------------------------------------------------
# deprecated dynamo_bk.py
@deprecated
def sol_u(*args, **kwargs):
    _sol_u_legacy(*args, **kwargs)


def _sol_u_legacy(t, u0, alpha, beta):
    return u0 * np.exp(-beta * t) + alpha / beta * (1 - np.exp(-beta * t))


@deprecated
def sol_s_dynamo_bk(*args, **kwargs):
    _sol_s_dynamo_bk_legacy(*args, **kwargs)


def _sol_s_dynamo_bk_legacy(t, s0, u0, alpha, beta, gamma):
    exp_gt = np.exp(-gamma * t)
    return (
        s0 * exp_gt + alpha / gamma * (1 - exp_gt) + (alpha + u0 * beta) / (gamma - beta) * (exp_gt - np.exp(-beta * t))
    )


@deprecated
def fit_gamma_labelling_dynamo_bk(*args, **kwargs):
    _fit_gamma_labelling_dynamo_bk_legacy(*args, **kwargs)


def _fit_gamma_labelling_dynamo_bk_legacy(t, l, mode=None, lbound=None):
    n = l.size
    tau = t - np.min(t)
    tm = np.mean(tau)

    # prepare y
    if lbound is not None:
        l[l < lbound] = lbound
    y = np.log(l)
    ym = np.mean(y)

    # calculate slope
    var_t = np.mean(tau**2) - tm**2
    cov = np.sum(y.dot(tau)) / n - ym * tm
    k = cov / var_t

    # calculate intercept
    b = np.exp(ym - k * tm) if mode != "fast" else None

    return -k, b


@deprecated
def fit_alpha_dynamo_bk_labelling(*args, **kwargs):
    _fit_alpha_labelling_dynamo_bk_legacy(*args, **kwargs)


def _fit_alpha_labelling_dynamo_bk_legacy(t, u, gamma, mode=None):
    n = u.size
    tau = t - np.min(t)
    expt = np.exp(gamma * tau)

    # prepare x
    x = expt - 1
    xm = np.mean(x)

    # prepare y
    y = u * expt
    ym = np.mean(y)

    # calculate slope
    var_x = np.mean(x**2) - xm**2
    cov = np.sum(y.dot(x)) / n - ym * xm
    k = cov / var_x

    # calculate intercept
    b = ym - k * xm if mode != "fast" else None

    return k * gamma, b


@deprecated
def fit_gamma_splicing_dynamo_bk(*args, **kwargs):
    _fit_gamma_splicing_dynamo_bk_legacy(*args, **kwargs)


def _fit_gamma_splicing_dynamo_bk_legacy(t, s, beta, u0, bounds=(0, np.inf)):
    tau = t - np.min(t)
    s0 = np.mean(s[tau == 0])
    g0 = beta * u0 / s0

    f_lsq = lambda g: _sol_s_dynamo_bk_legacy(tau, u0, s0, 0, beta, g) - s
    ret = least_squares(f_lsq, g0, bounds=bounds)
    return ret.x, s0


@deprecated
def fit_gamma_dynamo_bk(*args, **kwargs):
    _fit_gamma_dynamo_bk_legacy(*args, **kwargs)


def _fit_gamma_dynamo_bk_legacy(u, s):
    cov = u.dot(s) / len(u) - np.mean(u) * np.mean(s)
    var_s = s.dot(s) / len(s) - np.mean(s) ** 2
    gamma = cov / var_s
    return gamma


# ---------------------------------------------------------------------------------------------------
# deprecated dynamo_fitting.py
@deprecated
def sol_s_dynamo_fitting(*args, **kwargs):
    _sol_s_dynamo_fitting_legacy(*args, **kwargs)


def _sol_s_dynamo_fitting_legacy(t, s0, u0, alpha, beta, gamma):
    exp_gt = np.exp(-gamma * t)
    if beta == gamma:
        s = s0 * exp_gt + (beta * u0 - alpha) * t * exp_gt + alpha / gamma * (1 - exp_gt)
    else:
        s = (
            s0 * exp_gt
            + alpha / gamma * (1 - exp_gt)
            + (alpha - u0 * beta) / (gamma - beta) * (exp_gt - np.exp(-beta * t))
        )
    return s


@deprecated
def sol_p_dynamo_fitting(*args, **kwargs):
    _sol_p_dynamo_fitting_legacy(*args, **kwargs)


def _sol_p_dynamo_fitting_legacy(t, p0, s0, u0, alpha, beta, gamma, eta, gamma_p):
    u = _sol_u_legacy(t, u0, alpha, beta)
    s = _sol_s_dynamo_fitting_legacy(t, s0, u0, alpha, beta, gamma)
    exp_gt = np.exp(-gamma_p * t)
    p = p0 * exp_gt + eta / (gamma_p - gamma) * (
        s - s0 * exp_gt - beta / (gamma_p - beta) * (u - u0 * exp_gt - alpha / gamma_p * (1 - exp_gt))
    )
    return p, s, u


@deprecated
def sol_ode_dynamo_fitting(*args, **kwargs):
    _sol_ode_dynamo_fitting_legacy(*args, **kwargs)


def _sol_ode_dynamo_fitting_legacy(x, t, alpha, beta, gamma, eta, gamma_p):
    dx = np.zeros(x.shape)
    dx[0] = alpha - beta * x[0]
    dx[1] = beta * x[0] - gamma * x[1]
    dx[2] = eta * x[1] - gamma_p * x[2]
    return dx


@deprecated
def sol_num_dynamo_fitting(args, kwargs):
    _sol_num_dynamo_fitting_legacy(*args, **kwargs)


def _sol_num_dynamo_fitting_legacy(t, p0, s0, u0, alpha, beta, gamma, eta, gamma_p):
    sol = odeint(
        lambda x, t: _sol_ode_dynamo_fitting_legacy(x, t, alpha, beta, gamma, eta, gamma_p),
        np.array([u0, s0, p0]),
        t,
    )
    return sol


@deprecated
def fit_gamma_labelling_dynamo_fitting(*args, **kwargs):
    _fit_gamma_labelling_dynamo_fitting_legacy(*args, **kwargs)


def _fit_gamma_labelling_dynamo_fitting_legacy(t, l, mode=None, lbound=None):
    t = np.array(t, dtype=float)
    l = np.array(l, dtype=float)
    if l.ndim == 1:
        # l is a vector
        n_rep = 1
    else:
        n_rep = l.shape[0]
    t = np.tile(t, n_rep)
    l = l.flatten()

    # remove low counts based on lbound
    if lbound is not None:
        t[l < lbound] = np.nan
        l[l < lbound] = np.nan

    n = np.sum(~np.isnan(t))
    tau = t - np.nanmin(t)
    tm = np.nanmean(tau)

    # prepare y
    y = np.log(l)
    ym = np.nanmean(y)

    # calculate slope
    var_t = np.nanmean(tau**2) - tm**2
    cov = np.nansum(y * tau) / n - ym * tm
    k = cov / var_t

    # calculate intercept
    b = np.exp(ym - k * tm) if mode != "fast" else None

    gamma = -k
    u0 = b

    return gamma, u0


@deprecated
def fit_beta_lsq_dynamo_fitting(*args, **kwargs):
    _fit_beta_lsq_dynamo_fitting_legacy(*args, **kwargs)


def _fit_beta_lsq_dynamo_fitting_legacy(t, l, bounds=(0, np.inf), fix_l0=False, beta_0=None):
    tau = t - np.min(t)
    l0 = np.mean(l[:, tau == 0])
    if beta_0 is None:
        beta_0 = 1

    if fix_l0:
        f_lsq = lambda b: (_sol_u_legacy(tau, l0, 0, b) - l).flatten()
        ret = least_squares(f_lsq, beta_0, bounds=bounds)
        beta = ret.x
    else:
        f_lsq = lambda p: (_sol_u_legacy(tau, p[1], 0, p[0]) - l).flatten()
        ret = least_squares(f_lsq, np.array([beta_0, l0]), bounds=bounds)
        beta = ret.x[0]
        l0 = ret.x[1]
    return beta, l0


@deprecated
def fit_alpha_labelling_dynamo_fitting(*args, **kwargs):
    _fit_alpha_labelling_dynamo_fitting_legacy(*args, **kwargs)


def _fit_alpha_labelling_dynamo_fitting_legacy(t, u, gamma, mode=None):
    n = u.size
    tau = t - np.min(t)
    expt = np.exp(gamma * tau)

    # prepare x
    x = expt - 1
    xm = np.mean(x)

    # prepare y
    y = u * expt
    ym = np.mean(y)

    # calculate slope
    var_x = np.mean(x**2) - xm**2
    cov = np.sum(y.dot(x)) / n - ym * xm
    k = cov / var_x

    # calculate intercept
    b = ym - k * xm if mode != "fast" else None

    return k * gamma, b


@deprecated
def fit_alpha_synthesis_dynamo_fitting(*args, **kwargs):
    _fit_alpha_synthesis_dynamo_fitting_legacy(*args, **kwargs)


def _fit_alpha_synthesis_dynamo_fitting_legacy(t, u, beta, mode=None):
    tau = t - np.min(t)
    expt = np.exp(-beta * tau)

    # prepare x
    x = 1 - expt

    return beta * np.mean(u) / np.mean(x)


@deprecated
def fit_gamma_splicing_dynamo_fitting(*args, **kwargs):
    _fit_gamma_splicing_dynamo_fitting_legacy(*args, **kwargs)


def _fit_gamma_splicing_dynamo_fitting_legacy(t, s, beta, u0, bounds=(0, np.inf), fix_s0=False):
    tau = t - np.min(t)
    s0 = np.mean(s[:, tau == 0])
    g0 = beta * u0 / s0

    if fix_s0:
        f_lsq = lambda g: (_sol_s_dynamo_fitting_legacy(tau, s0, u0, 0, beta, g) - s).flatten()
        ret = least_squares(f_lsq, g0, bounds=bounds)
        gamma = ret.x
    else:
        f_lsq = lambda p: (_sol_s_dynamo_fitting_legacy(tau, p[1], u0, 0, beta, p[0]) - s).flatten()
        ret = least_squares(f_lsq, np.array([g0, s0]), bounds=bounds)
        gamma = ret.x[0]
        s0 = ret.x[1]
    return gamma, s0


@deprecated
def fit_gamma_dynamo_fitting(*args, **kwargs):
    _fit_gamma_dynamo_fitting_legacy(*args, **kwargs)


def _fit_gamma_dynamo_fitting_legacy(u, s):
    cov = u.dot(s) / len(u) - np.mean(u) * np.mean(s)
    var_s = s.dot(s) / len(s) - np.mean(s) ** 2
    gamma = cov / var_s
    return gamma


# ---------------------------------------------------------------------------------------------------
# deprecated utils_moments_deprecated.py
class moments:
    def __init__(
        self,
        a=None,
        b=None,
        la=None,
        alpha_a=None,
        alpha_i=None,
        sigma=None,
        beta=None,
        gamma=None,
    ):
        # species
        self.ua = 0
        self.ui = 1
        self.wa = 2
        self.wi = 3
        self.xa = 4
        self.xi = 5
        self.ya = 6
        self.yi = 7
        self.uu = 8
        self.ww = 9
        self.xx = 10
        self.yy = 11
        self.uw = 12
        self.ux = 13
        self.uy = 14
        self.wy = 15

        self.n_species = 16

        # solution
        self.t = None
        self.x = None
        self.x0 = zeros(self.n_species)
        self.K = None
        self.p = None

        # parameters
        if not (
            a is None
            or b is None
            or la is None
            or alpha_a is None
            or alpha_i is None
            or sigma is None
            or beta is None
            or gamma is None
        ):
            self.set_params(a, b, la, alpha_a, alpha_i, sigma, beta, gamma)

    def ode_moments(self, x, t):
        dx = zeros(len(x))
        # parameters
        a = self.a
        b = self.b
        la = self.la
        aa = self.aa
        ai = self.ai
        si = self.si
        be = self.be
        ga = self.ga

        # first moments
        dx[self.ua] = la * aa - be * x[self.ua] + a * (x[self.ui] - x[self.ua])
        dx[self.ui] = la * ai - be * x[self.ui] - b * (x[self.ui] - x[self.ua])
        dx[self.wa] = (1 - la) * aa - be * x[self.wa] + a * (x[self.wi] - x[self.wa])
        dx[self.wi] = (1 - la) * ai - be * x[self.wi] - b * (x[self.wi] - x[self.wa])
        dx[self.xa] = be * (1 - si) * x[self.ua] - ga * x[self.xa] + a * (x[self.xi] - x[self.xa])
        dx[self.xi] = be * (1 - si) * x[self.ui] - ga * x[self.xi] - b * (x[self.xi] - x[self.xa])
        dx[self.ya] = be * si * x[self.ua] + be * x[self.wa] - ga * x[self.ya] + a * (x[self.yi] - x[self.ya])
        dx[self.yi] = be * si * x[self.ui] + be * x[self.wi] - ga * x[self.yi] - b * (x[self.yi] - x[self.ya])

        # second moments
        dx[self.uu] = 2 * la * self.fbar(aa * x[self.ua], ai * x[self.ui]) - 2 * be * x[self.uu]
        dx[self.ww] = 2 * (1 - la) * self.fbar(self.aa * x[self.wa], ai * x[self.wi]) - 2 * be * x[self.ww]
        dx[self.xx] = 2 * be * (1 - si) * x[self.ux] - 2 * ga * x[self.xx]
        dx[self.yy] = 2 * si * be * x[self.uy] + 2 * be * x[self.wy] - 2 * ga * x[self.yy]
        dx[self.uw] = (
            la * self.fbar(aa * x[self.wa], ai * x[self.wi])
            + (1 - la) * self.fbar(aa * x[self.ua], ai * x[self.ui])
            - 2 * be * x[self.uw]
        )
        dx[self.ux] = (
            la * self.fbar(aa * x[self.xa], ai * x[self.xi]) + be * (1 - si) * x[self.uu] - (be + ga) * x[self.ux]
        )
        dx[self.uy] = (
            la * self.fbar(aa * x[self.ya], ai * x[self.yi])
            + si * be * x[self.uu]
            + be * x[self.uw]
            - (be + ga) * x[self.uy]
        )
        dx[self.wy] = (
            (1 - la) * self.fbar(aa * x[self.ya], ai * x[self.yi])
            + si * be * x[self.uw]
            + be * x[self.ww]
            - (be + ga) * x[self.wy]
        )

        return dx

    def integrate(self, t, x0=None):
        if x0 is None:
            x0 = self.x0
        else:
            self.x0 = x0
        sol = odeint(self.ode_moments, x0, t)
        self.x = sol
        self.t = t
        return sol

    def fbar(self, x_a, x_i):
        return self.b / (self.a + self.b) * x_a + self.a / (self.a + self.b) * x_i

    def set_params(self, a, b, la, alpha_a, alpha_i, sigma, beta, gamma):
        self.a = a
        self.b = b
        self.la = la
        self.aa = alpha_a
        self.ai = alpha_i
        self.si = sigma
        self.be = beta
        self.ga = gamma

        # reset solutions
        self.t = None
        self.x = None
        self.K = None
        self.p = None

    def get_all_central_moments(self):
        ret = zeros((8, len(self.t)))
        ret[0] = self.get_nu()
        ret[1] = self.get_nw()
        ret[2] = self.get_nx()
        ret[3] = self.get_ny()
        ret[4] = self.get_var_nu()
        ret[5] = self.get_var_nw()
        ret[6] = self.get_var_nx()
        ret[7] = self.get_var_ny()
        return ret

    def get_nosplice_central_moments(self):
        ret = zeros((4, len(self.t)))
        ret[0] = self.get_n_labeled()
        ret[1] = self.get_n_unlabeled()
        ret[2] = self.get_var_labeled()
        ret[3] = self.get_var_unlabeled()
        return ret

    def get_central_moments(self, keys=None):
        if keys is None:
            ret = self.get_all_centeral_moments()
        else:
            ret = zeros((len(keys) * 2, len(self.t)))
            i = 0
            if "ul" in keys:
                ret[i] = self.get_nu()
                ret[i + 1] = self.get_var_nu()
                i += 2
            if "uu" in keys:
                ret[i] = self.get_nw()
                ret[i + 1] = self.get_var_nw()
                i += 2
            if "sl" in keys:
                ret[i] = self.get_nx()
                ret[i + 1] = self.get_var_nx()
                i += 2
            if "su" in keys:
                ret[i] = self.get_ny()
                ret[i + 1] = self.get_var_ny()
                i += 2
        return ret

    def get_nu(self):
        return self.fbar(self.x[:, self.ua], self.x[:, self.ui])

    def get_nw(self):
        return self.fbar(self.x[:, self.wa], self.x[:, self.wi])

    def get_nx(self):
        return self.fbar(self.x[:, self.xa], self.x[:, self.xi])

    def get_ny(self):
        return self.fbar(self.x[:, self.ya], self.x[:, self.yi])

    def get_n_labeled(self):
        return self.get_nu() + self.get_nx()

    def get_n_unlabeled(self):
        return self.get_nw() + self.get_ny()

    def get_var_nu(self):
        c = self.get_nu()
        return self.x[:, self.uu] + c - c**2

    def get_var_nw(self):
        c = self.get_nw()
        return self.x[:, self.ww] + c - c**2

    def get_var_nx(self):
        c = self.get_nx()
        return self.x[:, self.xx] + c - c**2

    def get_var_ny(self):
        c = self.get_ny()
        return self.x[:, self.yy] + c - c**2

    def get_cov_ux(self):
        cu = self.get_nu()
        cx = self.get_nx()
        return self.x[:, self.ux] - cu * cx

    def get_cov_wy(self):
        cw = self.get_nw()
        cy = self.get_ny()
        return self.x[:, self.wy] - cw * cy

    def get_var_labeled(self):
        return self.get_var_nu() + self.get_var_nx() + 2 * self.get_cov_ux()

    def get_var_unlabeled(self):
        return self.get_var_nw() + self.get_var_ny() + 2 * self.get_cov_wy()

    def computeKnp(self):
        # parameters
        a = self.a
        b = self.b
        la = self.la
        aa = self.aa
        ai = self.ai
        si = self.si
        be = self.be
        ga = self.ga

        K = zeros((self.n_species, self.n_species))
        # E1
        K[self.ua, self.ua] = -be - a
        K[self.ua, self.ui] = a
        K[self.ui, self.ua] = b
        K[self.ui, self.ui] = -be - b
        K[self.wa, self.wa] = -be - a
        K[self.wa, self.wi] = a
        K[self.wi, self.wa] = b
        K[self.wi, self.wi] = -be - b

        # E2
        K[self.xa, self.xa] = -ga - a
        K[self.xa, self.xi] = a
        K[self.xi, self.xa] = b
        K[self.xi, self.xi] = -ga - b
        K[self.ya, self.ya] = -ga - a
        K[self.ya, self.yi] = a
        K[self.yi, self.ya] = b
        K[self.yi, self.yi] = -ga - b

        # E3
        K[self.uu, self.uu] = -2 * be
        K[self.ww, self.ww] = -2 * be
        K[self.xx, self.xx] = -2 * ga
        K[self.yy, self.yy] = -2 * ga

        # E4
        K[self.uw, self.uw] = -2 * be
        K[self.ux, self.ux] = -be - ga
        K[self.uy, self.uy] = -be - ga
        K[self.wy, self.wy] = -be - ga
        K[self.uy, self.uw] = be
        K[self.wy, self.uw] = si * be

        # F21
        K[self.xa, self.ua] = (1 - si) * be
        K[self.xi, self.ui] = (1 - si) * be
        K[self.ya, self.wa] = be
        K[self.ya, self.ua] = si * be
        K[self.yi, self.wi] = be
        K[self.yi, self.ui] = si * be

        # F31
        K[self.uu, self.ua] = 2 * la * aa * b / (a + b)
        K[self.uu, self.ui] = 2 * la * ai * a / (a + b)
        K[self.ww, self.wa] = 2 * (1 - la) * aa * b / (a + b)
        K[self.ww, self.wi] = 2 * (1 - la) * ai * a / (a + b)

        # F34
        K[self.xx, self.ux] = 2 * (1 - si) * be
        K[self.yy, self.uy] = 2 * si * be
        K[self.yy, self.wy] = 2 * be

        # F41
        K[self.uw, self.ua] = (1 - la) * aa * b / (a + b)
        K[self.uw, self.ui] = (1 - la) * ai * a / (a + b)
        K[self.uw, self.wa] = la * aa * b / (a + b)
        K[self.uw, self.wi] = la * ai * a / (a + b)

        # F42
        K[self.ux, self.xa] = la * aa * b / (a + b)
        K[self.ux, self.xi] = la * ai * a / (a + b)
        K[self.uy, self.ya] = la * aa * b / (a + b)
        K[self.uy, self.yi] = la * ai * a / (a + b)
        K[self.wy, self.ya] = (1 - la) * aa * b / (a + b)
        K[self.wy, self.yi] = (1 - la) * ai * a / (a + b)

        # F43
        K[self.ux, self.uu] = (1 - si) * be
        K[self.uy, self.uu] = si * be
        K[self.wy, self.ww] = be

        p = zeros(self.n_species)
        p[self.ua] = la * aa
        p[self.ui] = la * ai
        p[self.wa] = (1 - la) * aa
        p[self.wi] = (1 - la) * ai

        return K, p

    def solve(self, t, x0=None):
        t0 = t[0]
        if x0 is None:
            x0 = self.x0
        else:
            self.x0 = x0

        if self.K is None or self.p is None:
            K, p = self.computeKnp()
            self.K = K
            self.p = p
        else:
            K = self.K
            p = self.p
        x_ss = linalg.solve(K, p)
        # x_ss = linalg.inv(K).dot(p)
        y0 = x0 + x_ss

        D, U = linalg.eig(K)
        V = linalg.inv(U)
        D, U, V = map(real, (D, U, V))
        expD = exp(D)
        x = zeros((len(t), self.n_species))
        x[0] = x0
        for i in range(1, len(t)):
            x[i] = U.dot(diag(expD ** (t[i] - t0))).dot(V).dot(y0) - x_ss
        self.x = x
        self.t = t
        return x


class moments_simple:
    def __init__(
        self,
        a=None,
        b=None,
        la=None,
        alpha_a=None,
        alpha_i=None,
        sigma=None,
        beta=None,
        gamma=None,
    ):
        # species
        self._u = 0
        self._w = 1
        self._x = 2
        self._y = 3

        self.n_species = 4

        # solution
        self.t = None
        self.x = None
        self.x0 = zeros(self.n_species)
        self.K = None
        self.p = None

        # parameters
        if not (
            a is None
            or b is None
            or la is None
            or alpha_a is None
            or alpha_i is None
            or sigma is None
            or beta is None
            or gamma is None
        ):
            self.set_params(a, b, la, alpha_a, alpha_i, sigma, beta, gamma)

    def set_initial_condition(self, nu0, nw0, nx0, ny0):
        x = zeros(self.n_species)
        x[self._u] = nu0
        x[self._w] = nw0
        x[self._x] = nx0
        x[self._y] = ny0

        self.x0 = x
        return x

    def get_x_velocity(self, nu0, nx0):
        return self.be * (1 - self.si) * nu0 - self.ga * nx0

    def get_y_velocity(self, nu0, nw0, ny0):
        return self.be * self.si * nu0 + self.be * nw0 - self.ga * ny0

    def fbar(self, x_a, x_i):
        return self.b / (self.a + self.b) * x_a + self.a / (self.a + self.b) * x_i

    def set_params(self, a, b, la, alpha_a, alpha_i, sigma, beta, gamma):
        self.a = a
        self.b = b
        self.la = la
        self.aa = alpha_a
        self.ai = alpha_i
        self.si = sigma
        self.be = beta
        self.ga = gamma

        # reset solutions
        self.t = None
        self.x = None
        self.K = None
        self.p = None

    def get_total(self):
        return sum(self.x, 1)

    def computeKnp(self):
        # parameters
        la = self.la
        aa = self.aa
        ai = self.ai
        si = self.si
        be = self.be
        ga = self.ga

        K = zeros((self.n_species, self.n_species))

        # Diagonal
        K[self._u, self._u] = -be
        K[self._w, self._w] = -be
        K[self._x, self._x] = -ga
        K[self._y, self._y] = -ga

        # off-diagonal
        K[self._x, self._u] = be * (1 - si)
        K[self._y, self._u] = si * be
        K[self._y, self._w] = be

        p = zeros(self.n_species)
        p[self._u] = la * self.fbar(aa, ai)
        p[self._w] = (1 - la) * self.fbar(aa, ai)

        return K, p

    def solve(self, t, x0=None):
        t0 = t[0]
        if x0 is None:
            x0 = self.x0
        else:
            self.x0 = x0

        if self.K is None or self.p is None:
            K, p = self.computeKnp()
            self.K = K
            self.p = p
        else:
            K = self.K
            p = self.p
        x_ss = linalg.solve(K, p)
        y0 = x0 + x_ss

        D, U = linalg.eig(K)
        V = linalg.inv(U)
        D, U, V = map(real, (D, U, V))
        expD = exp(D)
        x = zeros((len(t), self.n_species))
        x[0] = x0
        for i in range(1, len(t)):
            x[i] = U.dot(diag(expD ** (t[i] - t0))).dot(V).dot(y0) - x_ss
        self.x = x
        self.t = t
        return x


class estimation:
    def __init__(self, ranges, x0=None):
        self.ranges = ranges
        self.n_params = len(ranges)
        self.simulator = moments()
        if not x0 is None:
            self.simulator.x0 = x0

    def sample_p0(self, samples=1, method="lhs"):
        ret = zeros((samples, self.n_params))
        if method == "lhs":
            ret = self._lhsclassic(samples)
            for i in range(self.n_params):
                ret[:, i] = ret[:, i] * (self.ranges[i][1] - self.ranges[i][0]) + self.ranges[i][0]
        else:
            for n in range(samples):
                for i in range(self.n_params):
                    r = random.rand()
                    ret[n, i] = r * (self.ranges[i][1] - self.ranges[i][0]) + self.ranges[i][0]
        return ret

    def _lhsclassic(self, samples):
        # From PyDOE
        # Generate the intervals
        cut = linspace(0, 1, samples + 1)

        # Fill points uniformly in each interval
        u = random.rand(samples, self.n_params)
        a = cut[:samples]
        b = cut[1 : samples + 1]
        rdpoints = zeros_like(u)
        for j in range(self.n_params):
            rdpoints[:, j] = u[:, j] * (b - a) + a

        # Make the random pairings
        H = zeros_like(rdpoints)
        for j in range(self.n_params):
            order = random.permutation(range(samples))
            H[:, j] = rdpoints[order, j]

        return H

    def get_bound(self, index):
        ret = zeros(self.n_params)
        for i in range(self.n_params):
            ret[i] = self.ranges[i][index]
        return ret

    def normalize_data(self, X):
        # ret = zeros(X.shape)
        # for i in range(len(X)):
        #     x = X[i]
        #     #ret[i] = x / max(x)
        #     ret[i] = log10(x + 1)
        return log10(X + 1)

    def f_curve_fit(self, t, *params):
        self.simulator.set_params(*params)
        self.simulator.integrate(t, self.simulator.x0)
        ret = self.simulator.get_all_central_moments()
        ret = self.normalize_data(ret)
        return ret.flatten()

    def f_lsq(self, params, t, x_data_norm, method="analytical", experiment_type=None):
        self.simulator.set_params(*params)
        if method == "numerical":
            self.simulator.integrate(t, self.simulator.x0)
        elif method == "analytical":
            self.simulator.solve(t, self.simulator.x0)
        if experiment_type is None:
            ret = self.simulator.get_all_central_moments()
        elif experiment_type == "nosplice":
            ret = self.simulator.get_nosplice_central_moments()
        elif experiment_type == "label":
            ret = self.simulator.get_central_moments(["ul", "sl"])
        ret = self.normalize_data(ret).flatten()
        ret[isnan(ret)] = 0
        return ret - x_data_norm

    def fit(self, t, x_data, p0=None, bounds=None):
        if p0 is None:
            p0 = self.sample_p0()
        x_data_norm = self.normalize_data(x_data)
        if bounds is None:
            bounds = (self.get_bound(0), self.get_bound(1))
        popt, pcov = curve_fit(self.f_curve_fit, t, x_data_norm.flatten(), p0=p0, bounds=bounds)
        return popt, pcov

    def fit_lsq(
        self,
        t,
        x_data,
        p0=None,
        n_p0=1,
        bounds=None,
        sample_method="lhs",
        method="analytical",
        experiment_type=None,
    ):
        if p0 is None:
            p0 = self.sample_p0(n_p0, sample_method)
        else:
            if p0.ndim == 1:
                p0 = [p0]
            n_p0 = len(p0)
        x_data_norm = self.normalize_data(x_data)
        if bounds is None:
            bounds = (self.get_bound(0), self.get_bound(1))

        costs = zeros(n_p0)
        X = []
        for i in range(n_p0):
            ret = least_squares(
                lambda p: self.f_lsq(p, t, x_data_norm.flatten(), method, experiment_type),
                p0[i],
                bounds=bounds,
            )
            costs[i] = ret.cost
            X.append(ret.x)
        i_min = argmin(costs)
        return X[i_min], costs[i_min]


# ---------------------------------------------------------------------------------------------------
# deprecated construct_velocity_tree.py
@deprecated
def remove_velocity_points(*args, **kwargs):
    return _remove_velocity_points_legacy(*args, **kwargs)


def _remove_velocity_points_legacy(G: np.ndarray, n: int) -> np.ndarray:
    """Modify a tree graph to remove the nodes themselves and recalculate the weights.

    Args:
        G: A smooth tree graph embedded in the low dimension space.
        n: The number of genes (column num of the original data)

    Returns:
        The tree graph with a node itself removed and weight recalculated.
    """
    for nodeid in range(n, 2 * n):
        nb_ids = []
        for nb_id in range(len(G[0])):
            if G[nodeid][nb_id] != 0:
                nb_ids = nb_ids + [nb_id]
        num_nbs = len(nb_ids)

        if num_nbs == 1:
            G[nodeid][nb_ids[0]] = 0
            G[nb_ids[0]][nodeid] = 0
        else:
            min_val = np.inf
            for i in range(len(G[0])):
                if G[nodeid][i] != 0:
                    if G[nodeid][i] < min_val:
                        min_val = G[nodeid][i]
                        min_ind = i
            for i in nb_ids:
                if i != min_ind:
                    new_weight = G[nodeid][i] + min_val
                    G[i][min_ind] = new_weight
                    G[min_ind][i] = new_weight
            # print('Add ege %s, %s\n',G.Nodes.Name {nb_ids(i)}, G.Nodes.Name {nb_ids(min_ind)});
            G[nodeid][nb_ids[0]] = 0
            G[nb_ids[0]][nodeid] = 0

    return G


@deprecated
def calculate_angle(*args, **kwargs):
    return _calculate_angle_legacy(*args, **kwargs)


def _calculate_angle_legacy(o: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    """Calculate the angle between two vectors.

    Args:
        o: Coordination of the origin.
        y: End point of the first vector.
        x: End point of the second vector.

    Returns:
        The angle between the two vectors.
    """

    yo = y - o
    norm_yo = yo / scipy.linalg.norm(yo)
    xo = x - o
    norm_xo = xo / scipy.linalg.norm(xo)
    angle = np.arccos(norm_yo.T * norm_xo)
    return angle


@deprecated
def construct_velocity_tree_py(*args, **kwargs):
    return _construct_velocity_tree_py_legacy(*args, **kwargs)


def _construct_velocity_tree_py_legacy(X1: np.ndarray, X2: np.ndarray) -> None:
    """Save a velocity tree graph with given data.

    Args:
        X1: Expression matrix.
        X2: Velocity matrix.
    """
    if issparse(X1):
        X1 = X1.toarray()
    if issparse(X2):
        X2 = X2.toarray()
    n = X1.shape[1]

    # merge two data with a given time
    t = 0.5
    X_all = np.hstack((X1, X1 + t * X2))

    # parameter settings
    maxIter = 20
    eps = 1e-3
    sigma = 0.001
    gamma = 10

    # run DDRTree algorithm
    Z, Y, stree, R, W, Q, C, objs = DDRTree(X_all, maxIter=maxIter, eps=eps, sigma=sigma, gamma=gamma)

    # draw velocity figure

    # quiver(Z(1, 1: 100), Z(2, 1: 100), Z(1, 101: 200)-Z(1, 1: 100), Z(2, 101: 200)-Z(2, 1: 100));
    # plot(Z(1, 1: 100), Z(2, 1: 100), 'ob');
    # plot(Z(1, 101: 200), Z(2, 101: 200), 'sr');
    G = stree

    sG = _remove_velocity_points_legacy(G, n)
    tree = sG
    row = []
    col = []
    val = []
    for i in range(sG.shape[0]):
        for j in range(sG.shape[1]):
            if sG[i][j] != 0:
                row = row + [i]
                col = col + [j]
                val = val + [sG[1][j]]
    tree_fname = "tree.csv"
    # write sG data to tree.csv
    #######
    branch_fname = "branch.txt"
    cmd = "python extract_branches.py" + tree_fname + branch_fname

    branch_cell = []
    fid = open(branch_fname, "r")
    tline = next(fid)
    while isinstance(tline, str):
        path = re.regexp(tline, "\d*", "Match")  ############
        branch_cell = branch_cell + [path]  #################
        tline = next(fid)
    fid.close()

    dG = np.zeros((n, n))
    for p in range(len(branch_cell)):
        path = branch_cell[p]
        pos_direct = 0
        for bp in range(len(path)):
            u = path(bp)
            v = u + n

            # find the shorest path on graph G(works for trees)
            nodeid = u
            ve_nodeid = v
            shortest_mat = shortest_path(
                csgraph=G,
                directed=False,
                indices=nodeid,
                return_predecessors=True,
            )
            velocity_path = []
            while ve_nodeid != nodeid:
                velocity_path = [shortest_mat[nodeid][ve_nodeid]] + velocity_path
                ve_nodeid = shortest_mat[nodeid][ve_nodeid]
            velocity_path = [shortest_mat[nodeid][ve_nodeid]] + velocity_path
            ###v_path = G.Nodes.Name(velocity_path)

            # check direction consistency between path and v_path
            valid_idx = []
            for i in velocity_path:
                if i <= n:
                    valid_idx = valid_idx + [i]
            if len(valid_idx) == 1:
                # compute direction matching
                if bp < len(path):
                    tree_next_point = Z[:, path(bp)]
                    v_point = Z[:, v]
                    u_point = Z[:, u]
                    angle = _calculate_angle_legacy(u_point, tree_next_point, v_point)
                    angle = angle / 3.14 * 180
                    if angle < 90:
                        pos_direct = pos_direct + 1

                else:
                    tree_pre_point = Z[:, path(bp - 1)]
                    v_point = Z[:, v]
                    u_point = Z[:, u]
                    angle = _calculate_angle_legacy(u_point, tree_pre_point, v_point)
                    angle = angle / 3.14 * 180
                    if angle > 90:
                        pos_direct = pos_direct + 1

            else:

                if bp < len(path):
                    if path[bp + 1] == valid_idx[2]:
                        pos_direct = pos_direct + 1

                else:
                    if path[bp - 1] != valid_idx[2]:
                        pos_direct = pos_direct + 1

        neg_direct = len(path) - pos_direct
        print(
            "branch="
            + str(p)
            + ", ("
            + path[0]
            + "->"
            + path[-1]
            + "), pos="
            + pos_direct
            + ", neg="
            + neg_direct
            + "\n"
        )
        print(path)
        print("\n")

        if pos_direct > neg_direct:
            for bp in range(len(path) - 1):
                dG[path[bp], path[bp + 1]] = 1

        else:
            for bp in range(len(path) - 1):
                dG[path(bp + 1), path(bp)] = 1

    # figure;
    # plot(digraph(dG));
    # title('directed graph') figure; hold on;
    row = []
    col = []
    for i in range(dG.shape[0]):
        for j in range(dG.shape[1]):
            if dG[i][j] != 0:
                row = row + [i]
                col = col + [j]
    for tn in range(len(row)):
        p1 = Y[:, row[tn]]
        p2 = Y[:, col[tn]]
        dp = p2 - p1
        h = plt.quiver(p1(1), p1(2), dp(1), dp(2), "LineWidth", 5)  ###############need to plot it
        set(h, "MaxHeadSize", 1e3, "AutoScaleFactor", 1)  #############

    for i in range(n):
        plt.text(Y(1, i), Y(2, i), str(i))  ##############
    plt.savefig("./results/t01_figure3.fig")  ##################


# ---------------------------------------------------------------------------------------------------
# deprecated pseudotime.py
@deprecated
def compute_partition(*args, **kwargs):
    return _compute_partition_legacy(*args, **kwargs)


def _compute_partition_legacy(adata, transition_matrix, cell_membership, principal_g, group=None):
    """Compute a partition of cells based on a minimum spanning tree and cell membership.

    Args:
        adata: The anndata object containing the single-cell data.
        transition_matrix: The matrix representing the transition probabilities between cells.
        cell_membership: The matrix representing the cell membership information.
        principal_g: The principal graph information saved as array.
        group: The name of a categorical group in `adata.obs`. If provided, it is used to construct the
            `cell_membership` matrix based on the specified group membership. Defaults to None.

    Returns:
        A partition of cells represented as a matrix.
    """

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    # http://active-analytics.com/blog/rvspythonwhyrisstillthekingofstatisticalcomputing/
    if group is not None and group in adata.obs.columns:
        from patsy import dmatrix  # dmatrices, dmatrix, demo_data

        data = adata.obs
        data.columns[data.columns == group] = "group_"

        cell_membership = csr_matrix(dmatrix("~group_+0", data=data))

    X = csr_matrix(principal_g > 0)
    Tcsr = minimum_spanning_tree(X)
    principal_g = Tcsr.toarray().astype(int)

    membership_matrix = cell_membership.T.dot(transition_matrix).dot(cell_membership)

    direct_principal_g = principal_g * membership_matrix

    # get the data:
    # edges_per_module < - Matrix::rowSums(num_links)
    # total_edges < - sum(num_links)
    #
    # theta < - (as.matrix(edges_per_module) / total_edges) % * %
    # Matrix::t(edges_per_module / total_edges)
    #
    # var_null_num_links < - theta * (1 - theta) / total_edges
    # num_links_ij < - num_links / total_edges - theta
    # cluster_mat < - pnorm_over_mat(as.matrix(num_links_ij), var_null_num_links)
    #
    # num_links < - num_links_ij / total_edges
    #
    # cluster_mat < - matrix(stats::p.adjust(cluster_mat),
    #                               nrow = length(louvain_modules),
    #                                      ncol = length(louvain_modules))
    #
    # sig_links < - as.matrix(num_links)
    # sig_links[cluster_mat > qval_thresh] = 0
    # diag(sig_links) < - 0

    return direct_principal_g


# ---------------------------------------------------------------------------------------------------
# deprecated moments.py
@deprecated
def _calc_1nd_moment(*args, **kwargs):
    return _calc_1nd_moment_legacy(*args, **kwargs)


def _calc_1nd_moment_legacy(X, W, normalize_W=True):
    """deprecated"""
    if normalize_W:
        d = np.sum(W, 1)
        W = np.diag(1 / d) @ W
    return W @ X


@deprecated
def _calc_2nd_moment(*args, **kwargs):
    return _calc_2nd_moment_legacy(*args, **kwargs)


def _calc_2nd_moment_legacy(X, Y, W, normalize_W=True, center=False, mX=None, mY=None):
    """deprecated"""
    if normalize_W:
        d = np.sum(W, 1)
        W = np.diag(1 / d) @ W
    XY = np.multiply(W @ Y, X)
    if center:
        mX = calc_1nd_moment(X, W, False) if mX is None else mX
        mY = calc_1nd_moment(Y, W, False) if mY is None else mY
        XY = XY - np.multiply(mX, mY)
    return XY


# old moment estimation code
class MomData(AnnData):
    """deprecated"""

    def __init__(self, adata, time_key="Time", has_nan=False):
        # self.data = adata
        self.__dict__ = adata.__dict__
        # calculate first and second moments from data
        self.times = np.array(self.obs[time_key].values, dtype=float)
        self.uniq_times = np.unique(self.times)
        nT = self.get_n_times()
        ng = self.get_n_genes()
        self.M = np.zeros((ng, nT))  # first moments (data)
        self.V = np.zeros((ng, nT))  # second moments (data)
        for g in tqdm(range(ng), desc="calculating 1/2 moments"):
            tmp = self[:, g].layers["new"]
            L = (
                np.array(tmp.A, dtype=float) if issparse(tmp) else np.array(tmp, dtype=float)
            )  # consider using the `adata.obs_vector`, `adata.var_vector` methods or accessing the array directly.
            if has_nan:
                self.M[g] = strat_mom(L, self.times, np.nanmean)
                self.V[g] = strat_mom(L, self.times, np.nanvar)
            else:
                self.M[g] = strat_mom(L, self.times, np.mean)
                self.V[g] = strat_mom(L, self.times, np.var)

    def get_n_genes(self):
        return self.var.shape[0]

    def get_n_cell(self):
        return self.obs.shape[0]

    def get_n_times(self):
        return len(self.uniq_times)


class Estimation:
    """deprecated"""

    def __init__(
        self,
        adata,
        adata_u=None,
        time_key="Time",
        normalize=True,
        param_ranges=None,
        has_nan=False,
    ):
        # initialize Estimation
        self.data = MomData(adata, time_key, has_nan)
        self.data_u = MomData(adata_u, time_key, has_nan) if adata_u is not None else None
        if param_ranges is None:
            param_ranges = {
                "a": [0, 10],
                "b": [0, 10],
                "alpha_a": [10, 1000],
                "alpha_i": [0, 10],
                "beta": [0, 10],
                "gamma": [0, 10],
            }
        self.normalize = normalize
        self.param_ranges = param_ranges
        self.n_params = len(param_ranges)

    def param_array2dict(self, parr):
        if parr.ndim == 1:
            return {
                "a": parr[0],
                "b": parr[1],
                "alpha_a": parr[2],
                "alpha_i": parr[3],
                "beta": parr[4],
                "gamma": parr[5],
            }
        else:
            return {
                "a": parr[:, 0],
                "b": parr[:, 1],
                "alpha_a": parr[:, 2],
                "alpha_i": parr[:, 3],
                "beta": parr[:, 4],
                "gamma": parr[:, 5],
            }

    def fit_gene(self, gene_no, n_p0=10):
        from ..estimation.tsc.utils_moments import estimation

        estm = estimation(list(self.param_ranges.values()))
        if self.data_u is None:
            m = self.data.M[gene_no, :].T
            v = self.data.V[gene_no, :].T
            x_data = np.vstack((m, v))
            popt, cost = estm.fit_lsq(
                self.data.uniq_times,
                x_data,
                p0=None,
                n_p0=n_p0,
                normalize=self.normalize,
                experiment_type="nosplice",
            )
        else:
            mu = self.data_u.M[gene_no, :].T
            ms = self.data.M[gene_no, :].T
            vu = self.data_u.V[gene_no, :].T
            vs = self.data.V[gene_no, :].T
            x_data = np.vstack((mu, ms, vu, vs))
            popt, cost = estm.fit_lsq(
                self.data.uniq_times,
                x_data,
                p0=None,
                n_p0=n_p0,
                normalize=self.normalize,
                experiment_type=None,
            )
        return popt, cost

    def fit(self, n_p0=10):
        ng = self.data.get_n_genes()
        params = np.zeros((ng, self.n_params))
        costs = np.zeros(ng)
        for i in tqdm(range(ng), desc="fitting genes"):
            params[i], costs[i] = self.fit_gene(i, n_p0)
        return params, costs


# use for kinetic assumption with full data, deprecated
def moment_model(adata, subset_adata, _group, cur_grp, log_unnormalized, tkey):
    """deprecated"""
    # a few hard code to set up data for moment mode:
    if "uu" in subset_adata.layers.keys() or "X_uu" in subset_adata.layers.keys():
        if log_unnormalized and "X_uu" not in subset_adata.layers.keys():
            if issparse(subset_adata.layers["uu"]):
                (
                    subset_adata.layers["uu"].data,
                    subset_adata.layers["ul"].data,
                    subset_adata.layers["su"].data,
                    subset_adata.layers["sl"].data,
                ) = (
                    np.log1p(subset_adata.layers["uu"].data),
                    np.log1p(subset_adata.layers["ul"].data),
                    np.log1p(subset_adata.layers["su"].data),
                    np.log1p(subset_adata.layers["sl"].data),
                )
            else:
                (
                    subset_adata.layers["uu"],
                    subset_adata.layers["ul"],
                    subset_adata.layers["su"],
                    subset_adata.layers["sl"],
                ) = (
                    np.log1p(subset_adata.layers["uu"]),
                    np.log1p(subset_adata.layers["ul"]),
                    np.log1p(subset_adata.layers["su"]),
                    np.log1p(subset_adata.layers["sl"]),
                )

        subset_adata_u, subset_adata_s = (
            subset_adata.copy(),
            subset_adata.copy(),
        )
        del (
            subset_adata_u.layers["su"],
            subset_adata_u.layers["sl"],
            subset_adata_s.layers["uu"],
            subset_adata_s.layers["ul"],
        )
        (
            subset_adata_u.layers["new"],
            subset_adata_u.layers["old"],
            subset_adata_s.layers["new"],
            subset_adata_s.layers["old"],
        ) = (
            subset_adata_u.layers.pop("ul"),
            subset_adata_u.layers.pop("uu"),
            subset_adata_s.layers.pop("sl"),
            subset_adata_s.layers.pop("su"),
        )
        Moment, Moment_ = MomData(subset_adata_s, tkey), MomData(subset_adata_u, tkey)
        if cur_grp == _group[0]:
            t_ind = 0
            g_len, t_len = len(_group), len(np.unique(adata.obs[tkey]))
            (adata.uns["M_sl"], adata.uns["V_sl"], adata.uns["M_ul"], adata.uns["V_ul"]) = (
                np.zeros((Moment.M.shape[0], g_len * t_len)),
                np.zeros((Moment.M.shape[0], g_len * t_len)),
                np.zeros((Moment.M.shape[0], g_len * t_len)),
                np.zeros((Moment.M.shape[0], g_len * t_len)),
            )

        inds = np.arange((t_len * t_ind), (t_len * (t_ind + 1)))
        (
            adata.uns["M_sl"][:, inds],
            adata.uns["V_sl"][:, inds],
            adata.uns["M_ul"][:, inds],
            adata.uns["V_ul"][:, inds],
        ) = (Moment.M, Moment.V, Moment_.M, Moment_.V)

        del Moment_
        Est = Estimation(Moment, adata_u=subset_adata_u, time_key=tkey, normalize=True)  # # data is already normalized
    else:
        if log_unnormalized and "X_total" not in subset_adata.layers.keys():
            if issparse(subset_adata.layers["total"]):
                (subset_adata.layers["new"].data, subset_adata.layers["total"].data,) = (
                    np.log1p(subset_adata.layers["new"].data),
                    np.log1p(subset_adata.layers["total"].data),
                )
            else:
                subset_adata.layers["total"], subset_adata.layers["total"] = (
                    np.log1p(subset_adata.layers["new"]),
                    np.log1p(subset_adata.layers["total"]),
                )

        Moment = MomData(subset_adata, tkey)
        if cur_grp == _group[0]:
            t_ind = 0
            g_len, t_len = len(_group), len(np.unique(adata.obs[tkey]))
            adata.uns["M"], adata.uns["V"] = (
                np.zeros((adata.shape[1], g_len * t_len)),
                np.zeros((adata.shape[1], g_len * t_len)),
            )

        inds = np.arange((t_len * t_ind), (t_len * (t_ind + 1)))
        (
            adata.uns["M"][:, inds],
            adata.uns["V"][:, inds],
        ) = (Moment.M, Moment.V)
        Est = Estimation(Moment, time_key=tkey, normalize=True)  # # data is already normalized

    return adata, Est, t_ind


#---------------------------------------------------------------------------------------------------
# deprecated clustering.py
def infomap(
    adata: AnnData,
    use_weight: bool = True,
    adj_matrix: Union[np.ndarray, csr_matrix, None] = None,
    adj_matrix_key: Optional[str] = None,
    result_key: Optional[str] = None,
    layer: Optional[str] = None,
    obsm_key: Optional[str] = None,
    selected_cluster_subset: Optional[Tuple[str, str]] = None,
    selected_cell_subset: Union[List[int], List[str], None] = None,
    directed: bool = False,
    copy: bool = False,
    **kwargs
) -> AnnData:
    """Apply infomap community detection algorithm to cluster adata.

    For other community detection general parameters, please refer to `dynamo`'s `tl.cluster_community` function.
    "Infomap is based on ideas of information theory. The algorithm uses the probability flow of random walks on a
    network as a proxy for information flows in the real system and it decomposes the network into modules by
    compressing a description of the probability flow." - cdlib

    Args:
        adata: an AnnData object.
        use_weight: whether to use graph weight or not. False means to use connectivities only (0/1 integer values).
            Defaults to True.
        adj_matrix: adj_matrix used for clustering. Defaults to None.
        adj_matrix_key: the key for adj_matrix stored in adata.obsp. Defaults to None.
        result_key: the key where the results will be stored in obs. Defaults to None.
        layer: the adata layer on which cluster algorithms will work. Defaults to None.
        obsm_key: the key in obsm corresponding to the data that would be used for finding neighbors. Defaults to None.
        selected_cluster_subset: a tuple of (cluster_key, allowed_clusters).Filtering cells in adata based on
            cluster_key in adata.obs and only reserve cells in the allowed clusters. Defaults to None.
        selected_cell_subset: a subset of cells in adata that would be clustered. Could be a list of indices or a list
            of cell names. Defaults to None.
        directed: whether the edges in the graph should be directed. Defaults to False.
        copy: whether to return a new updated AnnData object or updated the original one inplace. Defaults to False.

    Returns:
        An updated AnnData object if `copy` is set to be true.
    """

    raise NotImplementedError("infomap algorithm has been deprecated.")