import warnings
from typing import Callable, Iterable, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import functools

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares


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
