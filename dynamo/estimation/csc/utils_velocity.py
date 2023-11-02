from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import statsmodels.api as sm
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, issparse
from sklearn.linear_model import LinearRegression, RANSACRegressor

from ...tools.moments import strat_mom
from ...tools.utils import elem_prod, find_extreme


def sol_u(t: np.ndarray, u0: float, alpha: float, beta: float) -> np.ndarray:
    """The analytical solution of unspliced mRNA kinetics.

    Args:
        t: A vector of time points.
        u0: Initial value of u.
        alpha: Transcription rate.
        beta: Splicing rate constant.

    Returns:
        Unspliced mRNA counts at given time points.
    """
    return u0 * np.exp(-beta * t) + alpha / beta * (1 - np.exp(-beta * t))


def sol_u_2p(t: np.ndarray, u0: float, t1: np.ndarray, alpha0: float, alpha1: float, beta: float) -> np.ndarray:
    """The combined 2-piece analytical solution of unspliced mRNA kinetics.

    Args:
        t: A vector of time points for both steady state and stimulation labeling.
        u0: Initial value of u.
        t1: The time point when the cells switch from steady state to stimulation.
        alpha0: Transcription rate for steady state labeling.
        alpha1: Transcription rate for stimulation based labeling.
        beta: Splicing rate constant.

    Returns:
        Unspliced mRNA counts at given time points.
    """
    u1 = sol_u(t1, u0, alpha0, beta)
    u_pre = sol_u(t[t <= t1], u0, alpha0, beta)
    u_aft = sol_u(t[t > t1] - t1, u1, alpha1, beta)

    return np.concatenate((u_pre, u_aft))


def sol_s(t: np.ndarray, s0: float, u0: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """The analytical solution of spliced mRNA kinetics.

    Args:
        t: A vector of time points.
        s0: Initial value of s.
        u0: Initial value of u.
        alpha: Transcription rate.
        beta: Splicing rate constant.
        gamma: Degradation rate constant for spliced mRNA.

    Returns:
        Spliced mRNA counts at given time points.
    """
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


def sol_p(
    t: np.ndarray,
    p0: float,
    s0: float,
    u0: float,
    alpha: float,
    beta: float,
    gamma: float,
    eta: float,
    delta: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The analytical solution of protein kinetics.

    Args:
        t: A vector of time points.
        p0: Initial value of p.
        s0: Initial value of s.
        u0: Initial value of u.
        alpha: Transcription rate.
        beta: Splicing rate constant.
        gamma: Degradation rate constant for spliced mRNA.
        eta: Synthesis rate constant for protein.
        delta: Degradation rate constant for protein.

    Returns:
        Protein counts at given time points (p), spliced mRNA counts at given time points (s) and unspliced mRNA counts
        at given time points (u).
    """
    u = sol_u(t, u0, alpha, beta)
    s = sol_s(t, s0, u0, alpha, beta, gamma)
    exp_gt = np.exp(-delta * t)
    p = p0 * exp_gt + eta / (delta - gamma) * (
        s - s0 * exp_gt - beta / (delta - beta) * (u - u0 * exp_gt - alpha / delta * (1 - exp_gt))
    )
    return p, s, u


def solve_gamma(t: float, old: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Analytical solution to calculate gamma (degradation rate) using first-order degradation kinetics.

    Args:
        t: Metabolic labeling time period.
        old: A vector of old RNA amount in each cell
        total: A vector of total RNA amount in each cell

    Returns:
        The degradation rate (gamma).
    """

    old, total = np.mean(old), np.mean(total)
    gamma = -1 / t * np.log(old / total)

    return gamma


def solve_alpha_2p(t0: float, t1: float, alpha0: float, beta: float, u1: Union[csr_matrix, np.ndarray]) -> float:
    """Given known steady state alpha and beta, solve stimulation alpha for a mixed steady state and stimulation
    labeling experiment.

    Args:
        t0: Time period for steady state labeling.
        t1: Time period for stimulation labeling.
        alpha0: Steady state transcription rate calculated from one-shot experiment mode.
        beta: Steady state (and simulation) splicing rate calculated from one-shot experiment mode.
        u1: A vector of labeled RNA amount in each cell observed at time t0 + t1.

    Returns:
        The transcription rate (alpha1) for the stimulation period in the data.
    """

    u1 = np.mean(u1)

    u0 = alpha0 / beta * (1 - np.exp(-beta * t0))
    if t0 == 0:
        u0 = 0
    alpha1 = beta * (u1 - u0 * np.exp(-beta * t1)) / (1 - np.exp(-beta * t1))
    if t0 == 0:
        alpha1 = 0

    return alpha1


def solve_alpha_2p_mat(
    t0: np.ndarray,
    t1: np.ndarray,
    alpha0: np.ndarray,
    beta: np.ndarray,
    u1: np.ndarray,
) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Given known steady state alpha and beta, solve stimulation alpha for a mixed steady state and stimulation
    labeling experiment in a matrix form.

    Args:
        t0: Time period for steady state labeling for each cell.
        t1: Time period for stimulation labeling for each cell.
        alpha0: Steady state transcription rate calculated from one-shot experiment mode for each gene.
        beta: Steady state (and simulation) splicing rate calculated from one-shot experiment mode for each gene.
        u1: A vector of labeled RNA amount in each cell observed at time t0 + t1.

    Returns:
        The transcription rate (alpha1) for the stimulation period in the data.
    """

    alpha0 = np.repeat(alpha0.reshape((-1, 1)), u1.shape[1], axis=1)
    beta = np.repeat(beta.reshape((-1, 1)), u1.shape[1], axis=1)
    t0 = np.repeat(t0.reshape((-1, 1)), u1.shape[0], axis=1).T
    t1 = np.repeat(t1.reshape((-1, 1)), u1.shape[0], axis=1).T

    u0 = alpha0 / np.multiply(beta, (1 - np.exp(-np.multiply(beta, t0))))
    u0[t0 == 0] = 0

    u_new = u1 - np.multiply(u0, np.exp(-np.multiply(beta, t1)))
    u_new[t1 == 0] = 0

    alpha1 = np.multiply(beta, u_new / (1 - np.exp(-np.multiply(beta, t1))))
    alpha1[t1 == 0] = 0

    return csr_matrix(u0), csr_matrix(u_new), csr_matrix(alpha1)


def fit_linreg(
    x: Union[csr_matrix, np.ndarray],
    y: Union[csr_matrix, np.ndarray],
    mask: Optional[Union[List, np.ndarray]] = None,
    intercept: bool = False,
    r2: bool = True,
) -> Union[Tuple[float, float, float, float], Tuple[float, float]]:
    """Simple linear regression: y = kx + b.

    Args:
        x: A vector of independent variables.
        y: A vector of dependent variables.
        mask: The index to filter the data before regression.
        intercept: If using steady state assumption for fitting, then:
            True -- the linear regression is performed with an unfixed intercept;
            False -- the linear regression is performed with a fixed zero intercept.
        r2: The R squared value calculated from the regression.

    Returns:
        The estimated slope (k) , the estimated intercept (b), coefficient of determination or r square calculated with
        the extreme data points (r2), the r2 calculated using all data points (all_r2). If argument r2 is False, r2 and
        all_r2 will not be returned.
    """
    x = x.A if issparse(x) else x
    y = y.A if issparse(y) else y

    _mask = np.logical_and(~np.isnan(x), ~np.isnan(y))
    if mask is not None:
        _mask &= mask
    xx = x[_mask]
    yy = y[_mask]

    if intercept:
        ym = np.mean(yy)
        xm = np.mean(xx)

        cov = np.mean(xx * yy) - xm * ym
        var_x = np.mean(xx * xx) - xm * xm
        k = cov / var_x
        b = ym - k * xm
        # # assume b is always positive
        # if b_positive and b < 0:
        #     k, b = np.mean(xx * yy) / np.mean(xx * xx), 0
    else:
        # use uncentered cov and var_x
        cov = np.mean(xx * yy)
        var_x = np.mean(xx * xx)
        k = cov / var_x
        b = 0

    if r2:
        SS_tot_n, all_SS_tot_n = np.var(yy), np.var(y)
        SS_res_n, all_SS_res_n = (
            np.mean((yy - k * xx - b) ** 2),
            np.mean((y - k * x - b) ** 2),
        )
        r2, all_r2 = 1 - SS_res_n / SS_tot_n, 1 - all_SS_res_n / all_SS_tot_n

        return k, b, r2, all_r2
    else:
        return k, b


def fit_linreg_robust(
    x: Union[csr_matrix, np.ndarray],
    y: Union[csr_matrix, np.ndarray],
    mask: Optional[Union[List, np.ndarray]] = None,
    intercept: bool = False,
    r2: bool = True,
    est_method: str = "rlm",
) -> Union[Tuple[float, float, float, float], Tuple[float, float]]:
    """Apply robust linear regression of y w.r.t x.

    Args:
        x: A vector of independent variables.
        y: A vector of dependent variables.
        intercept: If using steady state assumption for fitting, then:
            True -- the linear regression is performed with an unfixed intercept;
            False -- the linear regression is performed with a fixed zero intercept.
        est_method: The linear regression estimation method that will be used.

    Returns:
        The estimated slope (k), the estimated intercept (b), coefficient of determination or r square calculated with
        the extreme data points (r2), the r2 calculated using all data points (all_r2).If argument r2 is False, r2 and
        all_r2 will not be returned.
    """

    x = x.A if issparse(x) else x
    y = y.A if issparse(y) else y

    _mask = np.logical_and(~np.isnan(x), ~np.isnan(y))
    if mask is not None:
        _mask &= mask
    xx = x[_mask]
    yy = y[_mask]

    try:
        if est_method.lower() == "rlm":
            xx_ = sm.add_constant(xx) if intercept else xx
            res = sm.RLM(yy, xx_).fit()
            k, b = res.params[::-1] if intercept else (res.params[0], 0)
        elif est_method.lower() == "ransac":
            reg = RANSACRegressor(LinearRegression(fit_intercept=intercept), random_state=0)
            reg.fit(xx.reshape(-1, 1), yy.reshape(-1, 1))
            k, b = reg.estimator_.coef_[0, 0], (reg.estimator_.intercept_[0] if intercept else 0)
        else:
            raise ImportError(
                f"estimation method {est_method} is not implemented. "
                f"Currently supported linear regression methods include `rlm` and `ransac`."
            )
    except:
        if intercept:
            ym = np.mean(yy)
            xm = np.mean(xx)

            cov = np.mean(xx * yy) - xm * ym
            var_x = np.mean(xx * xx) - xm * xm
            k = cov / var_x
            b = ym - k * xm
            # # assume b is always positive
            # if b < 0:
            #     k, b = np.mean(xx * yy) / np.mean(xx * xx), 0
        else:
            # use uncentered cov and var_x
            cov = np.mean(xx * yy)
            var_x = np.mean(xx * xx)
            k = cov / var_x
            b = 0

    if r2:
        SS_tot_n, all_SS_tot_n = np.var(yy), np.var(y)
        SS_res_n, all_SS_res_n = (
            np.mean((yy - k * xx - b) ** 2),
            np.mean((y - k * x - b) ** 2),
        )
        r2, all_r2 = 1 - SS_res_n / SS_tot_n, 1 - all_SS_res_n / all_SS_tot_n

        return k, b, r2, all_r2
    else:
        return k, b


def fit_stochastic_linreg(
    u: Union[csr_matrix, np.ndarray],
    s: Union[csr_matrix, np.ndarray],
    us: Union[csr_matrix, np.ndarray],
    ss: Union[csr_matrix, np.ndarray],
    fit_2_gammas: bool = True,
    err_cov: bool = False,
) -> float:
    """Generalized method of moments: [u, 2*us + u] = gamma * [s, 2*ss - s].

    Args:
        u: A vector of first moments (mean) of unspliced (or new) RNA expression.
        s: A vector of first moments (mean) of spliced (or total) RNA expression.
        us: A vector of second moments (uncentered co-variance) of unspliced/spliced (or new/total) RNA expression.
        ss: A vector of second moments (uncentered variance) of spliced (or total) RNA expression.

    Returns:
        The estimated gamma.
    """
    y = np.vstack((u.flatten(), (u + 2 * us).flatten()))
    x = np.vstack((s.flatten(), (2 * ss - s).flatten()))

    # construct the error covariance matrix
    if fit_2_gammas:
        k1 = fit_linreg(x[0], y[0])[0]
        k2 = fit_linreg(x[1], y[1])[0]
        k = np.array([k1, k2])
        E = y - k[:, None] * x
    else:
        k = np.mean(np.sum(elem_prod(y, x), 0)) / np.mean(np.sum(elem_prod(x, x), 0))
        E = y - k * x

    if err_cov:
        # cov = E @ E.T
        cov = np.cov(E)
    else:
        cov = np.diag(E.var(1))
    cov_inv = np.linalg.pinv(cov)

    # generalized least squares
    xy, xx = 0, 0
    for i in range(x.shape[1]):
        xy += y[:, i].T @ cov_inv @ x[:, i]
        xx += x[:, i].T @ cov_inv @ x[:, i]
    if xx == 0:
        gamma = np.nan
    else:
        gamma = xy / xx
    return gamma


def fit_first_order_deg_lsq(
    t: np.ndarray,
    l: Union[csr_matrix, np.ndarray],
    bounds: Tuple = (0, np.inf),
    fix_l0: bool = False,
    beta_0: Union[float, int] = 1,
) -> Tuple[float, float]:
    """Estimate beta with degradation data using least squares method.

    Args:
        t: A vector of time points.
        l: A vector of unspliced, labeled mRNA counts for each time point.
        bounds: The bound for beta. The default is beta > 0.
        fix_l0: Whether to used fixed l0. If ture, l0 will be calculated by averaging the first column of l;
            If false, l0 is a parameter that will be estimated all together with beta using lsq.
        beta_0: Initial guess for beta.

    Returns:
        The estimated value for beta (beta) and the estimated value for the initial spliced, labeled mRNA count (l0).
    """
    l = l.A.flatten() if issparse(l) else l

    tau = t - np.min(t)
    l0 = np.nanmean(l[tau == 0])

    if fix_l0:
        f_lsq = lambda b: sol_u(tau, l0, 0, b) - l
        ret = least_squares(f_lsq, beta_0, bounds=bounds)
        beta = ret.x
    else:
        f_lsq = lambda p: sol_u(tau, p[1], 0, p[0]) - l
        ret = least_squares(f_lsq, np.array([beta_0, l0]), bounds=bounds)
        beta = ret.x[0]
        l0 = ret.x[1]
    return beta, l0


def solve_first_order_deg(t: np.ndarray, l: Union[csr_matrix, np.ndarray]) -> Tuple[float, float, float]:
    """Solve for the initial amount and the rate constant of a species (for example, labeled mRNA) with time-series data
    under first-order degradation kinetics model.

    Args:
        t: A vector of time points.
        l: A vector of labeled mRNA counts for each time point.

    Returns:
        The initial counts of the species (for example, labeled mRNA), degradation rate constant and half-life the species.
    """

    x = l.A.flatten() if issparse(l) else l

    t_uniq = np.unique(t)
    x_stra = strat_mom(x, t, np.nanmean)
    x_stra_ = x_stra[1:] / x_stra[0]
    k = np.mean(-np.log(x_stra_) / t_uniq[1:])

    l0, half_life = np.mean(x_stra / np.exp(-k * t_uniq)), np.log(2) / k

    return l0, k, half_life


def fit_gamma_lsq(
    t: np.ndarray,
    s: Union[csr_matrix, np.ndarray],
    beta: float,
    u0: float,
    bounds: Tuple = (0, np.inf),
    fix_s0: bool = False,
) -> Tuple[float, float]:
    """Estimate gamma with degradation data using least squares method.

    Args:
        t: A vector of time points.
        s: A vector of spliced, labeled mRNA counts for each time point.
        beta: The value of beta.
        u0: Initial number of unspliced mRNA.
        bounds: The bound for gamma. The default is gamma > 0.
        fix_s0: Whether to use fixed s0. If true, s0 will be calculated by averaging the first column of s;
            If false, s0 is a parameter that will be estimated all together with gamma using lsq.

    Returns:
        The estimated value for gamma and the estimated value for the initial spliced mRNA count.
    """
    s = s.A.flatten() if issparse(s) else s

    tau = t - np.min(t)
    s0 = np.mean(s[tau == 0])
    g0 = beta * u0 / s0

    if fix_s0:
        f_lsq = lambda g: sol_s(tau, s0, u0, 0, beta, g) - s
        ret = least_squares(f_lsq, g0, bounds=bounds)
        gamma = ret.x
    else:
        if np.isfinite(g0):
            f_lsq = lambda p: sol_s(tau, p[1], u0, 0, beta, p[0]) - s
            ret = least_squares(f_lsq, np.array([g0, s0]), bounds=bounds)
            gamma = ret.x[0]
            s0 = ret.x[1]
        else:
            gamma, s0 = np.nan, 0
    return gamma, s0


def fit_alpha_synthesis(t: np.ndarray, u: Union[csr_matrix, np.ndarray], beta: float) -> float:
    """Estimate alpha with synthesis data using linear regression with fixed zero intercept.
    It is assumed that u(0) = 0.

    Args:
        t: A vector of time points.
        u: A matrix of unspliced mRNA counts. Dimension: cells x time points.
        beta: The value of beta.

    Returns:
        The estimated value for alpha.
    """
    u = u.A if issparse(u) else u

    # fit alpha assuming u=0 at t=0
    expt = np.exp(-beta * t)

    # prepare x
    x = 1 - expt

    return beta * np.mean(u) / np.mean(x)


def fit_alpha_degradation(
    t: np.ndarray,
    u: Union[csr_matrix, np.ndarray],
    beta: float,
    intercept: bool = False,
) -> Union[float, float, float]:
    """Estimate alpha with degradation data using linear regression. This is a lsq version of the following function that
    constrains u0 to be larger than 0.

    Args:
        t: A vector of time points.
        u: A matrix of unspliced mRNA counts. Dimension: cells x time points.
        beta: The value of beta.
        intercept: Whether to steady state assumption for fitting, then:
            True -- the linear regression is performed with an unfixed intercept;
            False -- the linear regression is performed with a fixed zero intercept.

    Returns:
        The estimated value for alpha (alpha), the initial unspliced mRNA count (u0), and coefficient of determination
    or r square (r2).
    """
    x = u.A if issparse(u) else u

    tau = t - np.min(t)

    f_lsq = lambda p: sol_u(tau, p[0], p[1], beta) - x
    ret = least_squares(f_lsq, np.array([1, 1]), bounds=(0, np.inf))
    alpha, u0 = ret.x[0], ret.x[1]

    # calculate r-squared
    SS_tot_n = np.var(x)
    SS_res_n = np.mean((f_lsq([alpha, u0])) ** 2)
    r2 = 1 - SS_res_n / SS_tot_n

    return alpha, u0, r2


def solve_alpha_degradation(
    t: np.ndarray,
    u: Union[csr_matrix, np.ndarray],
    beta: float,
    intercept: bool = False,
) -> Union[float, float, float]:
    """Estimate alpha with degradation data using linear regression.

    Args:
        t: A vector of time points.
        u: A matrix of unspliced mRNA counts. Dimension: cells x time points.
        beta: The value of beta.
        intercept: If using steady state assumption for fitting, then:
            True -- the linear regression is performed with an unfixed intercept;
            False -- the linear regression is performed with a fixed zero intercept.

    Returns:
        The estimated value for alpha (alpha), the initial unspliced mRNA count (b), coefficient of determination or r
        square.
    """
    u = u.A if issparse(u) else u

    n = u.size
    tau = t - np.min(t)
    expt = np.exp(beta * tau)

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
    b = ym - k * xm if intercept else 0
    SS_tot_n = np.var(y)
    SS_res_n = np.mean((y - k * x - b) ** 2) if b is not None else np.mean((y - k * x) ** 2)
    r2 = 1 - SS_res_n / SS_tot_n

    return k * beta, b, r2


def fit_alpha_beta_synthesis(
    t: np.ndarray,
    l: Union[csr_matrix, np.ndarray],
    bounds: Tuple = (0, np.inf),
    alpha_0: Union[float, int] = 1,
    beta_0: Union[float, int] = 1,
) -> Tuple[float, float]:
    """Estimate alpha and beta with synthesis data using the least square method.
    It is assumed that u(0) = 0.

    Args:
        t: A vector of time points.
        l: A matrix of labeled mRNA counts. Dimension: cells x time points.
        bounds: The bound for alpha and beta. The default is alpha / beta > 0.
        alpha_0: Initial guess for alpha.
        beta_0: Initial guess for beta.

    Returns:
        The estimated value for alpha and the estimated value for beta.
    """
    l = l.A if issparse(l) else l

    tau = np.hstack((0, t))
    x = np.hstack((0, l))

    f_lsq = lambda p: sol_u(tau, 0, p[0], p[1]) - x
    ret = least_squares(f_lsq, np.array([alpha_0, beta_0]), bounds=bounds)
    return ret.x[0], ret.x[1]


def fit_all_synthesis(
    t: np.ndarray,
    l: Union[csr_matrix, np.ndarray],
    bounds: Tuple = (0, np.inf),
    alpha_0: Union[float, int] = 1,
    beta_0: Union[float, int] = 1,
    gamma_0: Union[float, int] = 1,
) -> Tuple[float, float, float]:
    """Estimate alpha, beta and gamma with synthesis data using the least square method.
    It is assumed that u(0) = 0 and s(0) = 0.

    Args:
        t: A vector of time points.
        l: A matrix of labeled mRNA counts. Dimension: cells x time points.
        bounds: The bound for alpha and beta. The default is alpha / beta > 0.
        alpha_0: Initial guess for alpha.
        beta_0: Initial guess for beta.
        gamma_0: Initial guess for gamma.

    Returns:
        The estimated value for alpha, beta and gamma.
    """
    l = l.A if issparse(l) else l

    tau = np.hstack((0, t))
    x = np.hstack((0, l))

    f_lsq = lambda p: sol_u(tau, 0, p[0], p[1]) + sol_s(tau, 0, 0, p[0], p[1], p[2]) - x
    ret = least_squares(f_lsq, np.array([alpha_0, beta_0, gamma_0]), bounds=bounds)
    return ret.x[0], ret.x[1], ret.x[2]


def concat_time_series_matrices(
    mats: np.ndarray,
    t: Optional[Union[List, np.ndarray]] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Concatenate a list of gene x cell matrices into a single matrix.

    Args:
        mats: A list of gene x cell matrices. The length of the list equals the number of time points
        t: A vector or list of time points.

    Returns:
        Concatenated gene x cell matrix (ret_mat) and a vector of time point for each cell (ret_t).
    """
    ret_mat = np.concatenate(mats, axis=1)
    if t is not None:
        ret_t = np.concatenate([[t[i]] * mats[i].shape[1] for i in range(len(t))])
        return ret_mat, ret_t
    else:
        return ret_mat


# ---------------------------------------------------------------------------------------------------
# negbin method related
def compute_dispersion(mX: Union[np.ndarray, csr_matrix], varX: Union[np.ndarray, csr_matrix]) -> float:
    """Compute the reciprocal of the dispersion parameter of NB distribution from the following relationship:
        variance = mean + phi * mean^2

    Args:
        mX: The mean of variable X.
        varX: The variance of variable X.

    Returns:
        The dispersion parameter.
    """
    phi = fit_linreg(mX**2, varX - mX, intercept=False)[0]
    return phi


def fit_k_negative_binomial(
    n: Union[np.ndarray, csr_matrix],
    r: Union[np.ndarray, csr_matrix],
    var: float,
    phi: Optional[float] = None,
    k0: Optional[float] = None,
    return_k0: bool = False,
) -> Union[Tuple[float, float], float]:
    """Fit parameter k for the negative binomial distribution of one gene using the least squares optimizer:
        k * r = n
        k^2 * variance(r) = k * n - phi * n * 2

    Args:
        n: Unspliced mRNA or new/labeled mRNA of one gene.
        r: Spliced mRNA or total mRNA of one gene.
        var: The variance of r.
        phi: The dispersion parameter of the negative binomial distribution of one gene.
        k0: The initial guess for k.
        return_k0: Whether to return k0.

    Returns:
        The resolution from the least squares optimizer.
    """
    k0 = fit_linreg(r, n, intercept=False)[0] if k0 is None else k0
    phi = compute_dispersion(r, var) if phi is None else phi

    g1 = lambda k: k * r - n
    g2 = lambda k: k * k * var - k * n - phi * n * n
    g = lambda k: np.hstack((g1(k) / max(g1(k0).std(0), 1e-3), g2(k) / max(g2(k0).std(0), 1e-3)))
    ret = least_squares(g, k0, bounds=(0, np.inf))
    if return_k0:
        return ret.x[0], k0
    else:
        return ret.x[0]


def fit_K_negbin(
    N: Union[np.ndarray, csr_matrix],
    R: Union[np.ndarray, csr_matrix],
    varR: Union[np.ndarray, csr_matrix],
    perc_left: Optional[Union[float, int]] = None,
    perc_right: Optional[Union[float, int]] = None,
    return_phi: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Fit parameter K of negative binomial distribution to each gene.

    Args:
        N: Unspliced mRNA or new/labeled mRNA.
        R: Spliced mRNA or total mRNA.
        varR: The variance of R.
        perc_left: Left percentile for selecting extreme data points.
        perc_right: Right percentile for selecting extreme data points.
        return_phi: Whether to return dispersion parameter Phi.

    Returns:
        Fitted parameter K for each gene.
    """
    n_gene = N.shape[1]
    K = np.zeros(n_gene)
    Phi = np.zeros(n_gene)
    for i in range(n_gene):
        n = N[:, i].flatten()
        r = R[:, i].flatten()
        var = varR[:, i].flatten()
        Phi[i] = compute_dispersion(r, var)
        if perc_left is None and perc_right is None:
            K[i] = fit_k_negative_binomial(n, r, var, Phi[i])
        else:
            eind = find_extreme(n, r, perc_left=perc_left, perc_right=perc_right)
            K[i] = fit_k_negative_binomial(n[eind], r[eind], var[eind], Phi[i])
    if return_phi:
        return K, Phi
    else:
        return K


def compute_velocity_labeling(
    N: Union[np.ndarray, csr_matrix],
    R: Union[np.ndarray, csr_matrix],
    K: Union[np.ndarray, csr_matrix],
    tau: Union[np.ndarray, csr_matrix],
) -> Union[np.ndarray, csr_matrix]:
    """Compute velocity for labeling data by: velocity = gamma / k * new - gamma * total.

    Args:
        N: New or labeled mRNA.
        R: Total mRNA.
        K: Fitted slope k.
        tau: Time information.

    Returns:
        Computed velocity.
    """
    Kc = np.clip(K, 0, 1 - 1e-3)
    if np.isscalar(tau):
        Beta_or_gamma = -np.log(1 - Kc) / tau
    else:
        Beta_or_gamma = -(np.log(1 - Kc)[None, :].T / tau).T
    return elem_prod(Beta_or_gamma, N) / K - elem_prod(Beta_or_gamma, R)


def compute_bursting_properties(
    M_t: Union[float, np.ndarray],
    phi: float,
    gamma: float,
) -> Tuple[Union[float, np.ndarray], np.ndarray]:
    """Compute bursting frequency and size for the negative binomial regression model. The equations come from:
        Anton J.M. Larsson et al. Genomic encoding of transcriptional burst kinetics, Nature volume 565, pages251–254(2019)

    Args:
        M_t: The first moment of the number of total mRNA. If an array is passed, a cell-wise bursting size will be
            calculated.
        phi: The reciprocal dispersion parameter.
        gamma: Degradation rate constant.

    Returns:
        Bursting size and bursting frequency
    """
    bs = M_t * phi
    bf = gamma / phi
    return bs, bf
