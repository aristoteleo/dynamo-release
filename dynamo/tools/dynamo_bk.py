import numpy as np
from scipy.optimize import least_squares
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal


def sol_u(t, u0, alpha, beta):
    return u0 * np.exp(-beta * t) + alpha / beta * (1 - np.exp(-beta * t))


def sol_s(t, s0, u0, alpha, beta, gamma):
    exp_gt = np.exp(-gamma * t)
    return (
        s0 * exp_gt
        + alpha / gamma * (1 - exp_gt)
        + (alpha + u0 * beta) / (gamma - beta) * (exp_gt - np.exp(-beta * t))
    )


def fit_gamma_labelling(t, l, mode=None, lbound=None):
    n = l.size
    tau = t - np.min(t)
    tm = np.mean(tau)

    # prepare y
    if lbound is not None:
        l[l < lbound] = lbound
    y = np.log(l)
    ym = np.mean(y)

    # calculate slope
    var_t = np.mean(tau ** 2) - tm ** 2
    cov = np.sum(y.dot(tau)) / n - ym * tm
    k = cov / var_t

    # calculate intercept
    b = np.exp(ym - k * tm) if mode != "fast" else None

    return -k, b


def fit_alpha_labelling(t, u, gamma, mode=None):
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
    var_x = np.mean(x ** 2) - xm ** 2
    cov = np.sum(y.dot(x)) / n - ym * xm
    k = cov / var_x

    # calculate intercept
    b = ym - k * xm if mode != "fast" else None

    return k * gamma, b


def fit_gamma_splicing(t, s, beta, u0, bounds=(0, np.inf)):
    tau = t - np.min(t)
    s0 = np.mean(s[tau == 0])
    g0 = beta * u0 / s0

    f_lsq = lambda g: sol_s(tau, u0, s0, 0, beta, g) - s
    ret = least_squares(f_lsq, g0, bounds=bounds)
    return ret.x, s0


def fit_gamma(u, s):
    cov = u.dot(s) / len(u) - np.mean(u) * np.mean(s)
    var_s = s.dot(s) / len(s) - np.mean(s) ** 2
    gamma = cov / var_s
    return gamma
