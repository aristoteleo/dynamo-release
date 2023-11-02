from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, issparse
from tqdm import tqdm

from ...tools.utils import calc_norm_loglikelihood, calc_R2, elem_prod, find_extreme
from ..csc.utils_velocity import fit_linreg, fit_stochastic_linreg


def fit_slope_stochastic(
    S: Union[csc_matrix, csr_matrix, np.ndarray],
    U: Union[csc_matrix, csr_matrix, np.ndarray],
    US: Union[csc_matrix, csr_matrix, np.ndarray],
    S2: Union[csc_matrix, csr_matrix, np.ndarray],
    perc_left: Optional[int] = None,
    perc_right: int = 5,
) -> Tuple:
    """Estimate the slope of unspliced and spliced RNA with the GMM method: [u, 2*us + u] = gamma * [s, 2*ss - s]. The
    answer is denoted as gamma_k most of the time, which equals gamma/beta under steady state.

    Args:
        S: A matrix of the first moments of the spliced RNA.
        U: A matrix of the first moments of the unspliced RNA.
        US: A matrix of the cross moments of unspliced/spliced RNA.
        S2: A matrix of the second moments of spliced RNA.
        perc_left: The left percentile limitation to find extreme data points.
        perc_right: The right percentile limitation to find extreme data points.

    Returns:
        The slope, intercept, R squared and log likelihood.
    """
    n_var = S.shape[0]
    k, all_r2, all_logLL = np.zeros(n_var), np.zeros(n_var), np.zeros(n_var)

    for i, s, u, us, s2 in tqdm(
        zip(np.arange(n_var), S, U, US, S2),
        "Estimate slope k via linear regression.",
    ):
        u = u.A.flatten() if issparse(u) else u.flatten()
        s = s.A.flatten() if issparse(s) else s.flatten()
        us = us.A.flatten() if issparse(us) else us.flatten()
        s2 = s2.A.flatten() if issparse(s2) else s2.flatten()

        mask = find_extreme(u, s, perc_left=perc_left, perc_right=perc_right)
        k[i] = fit_stochastic_linreg(u[mask], s[mask], us[mask], s2[mask])

        all_r2[i] = calc_R2(s, u, k[i])
        all_logLL[i] = calc_norm_loglikelihood(s, u, k[i])

    return k, 0, all_r2, all_logLL


def fit_labeling_synthesis(
    new: Union[csc_matrix, csr_matrix, np.ndarray],
    total: Union[csc_matrix, csr_matrix, np.ndarray],
    t: Union[List, csc_matrix, csr_matrix, np.ndarray],
    intercept: bool = False,
    perc_left: Optional[int] = None,
    perc_right: Optional[int] = None,
):
    """Calculate the slope of total and new RNA under steady-state assumption.

    Args:
        new: A matrix representing new RNA. Can be expression or the first moments.
        total: A matrix representing total RNA. Can be expression or the first moments.
        t: A matrix of time information.
        intercept: Whether to perform the linear regression with intercept.
        perc_left: The left percentile limitation to find extreme data points.
        perc_right: The right percentile limitation to find extreme data points.

    Returns:
        The slope K and R squared of linear regression.
    """
    T = np.unique(t)
    K = np.zeros(len(T))
    R2 = np.zeros(len(T))
    for i in range(len(T)):
        n = new[t == T[i]]
        r = total[t == T[i]]
        eind = find_extreme(n, r, perc_left=perc_left, perc_right=perc_right)
        K[i], _, R2[i], _ = fit_linreg(r[eind], n[eind], intercept=intercept)
    return K, R2


def compute_gamma_synthesis(
    K: Union[csc_matrix, csr_matrix, np.ndarray],
    T: Union[csc_matrix, csr_matrix, np.ndarray],
) -> Tuple:
    """Calculate gamma as the linear regression results of given time and log(1 - slope k).

    Args:
        K: A matrix of the slope k.
        T: A matrix of time information.

    Returns:
        The gamma and R squared of linear regression.
    """
    gamma, _, r2, _ = fit_linreg(T, -np.log(1 - K))
    return gamma, r2


def compute_velocity_synthesis(
    N: Union[csc_matrix, csr_matrix, np.ndarray],
    R: Union[csc_matrix, csr_matrix, np.ndarray],
    gamma: Union[csc_matrix, csr_matrix, np.ndarray],
    t: Union[List, csc_matrix, csr_matrix, np.ndarray],
) -> Union[csc_matrix, csr_matrix, np.ndarray]:
    """Calculate the velocity of total RNA with a physical time unit: velocity = (gamma / k) N - gamma * R.

    Args:
        N: A matrix representing new RNA.
        R: A matrix representing total RNA.
        gamma: A matrix of degradation rate.
        t: A matrix of time information.

    Returns:
        The velocity.
    """
    k = 1 - np.exp(-np.einsum("i,j->ij", t, gamma))
    V = elem_prod(gamma, N) / k - elem_prod(gamma, R)
    return V


def lin_reg_gamma_synthesis(
    R: Union[csc_matrix, csr_matrix, np.ndarray],
    N: Union[csc_matrix, csr_matrix, np.ndarray],
    time: Union[List, csc_matrix, csr_matrix, np.ndarray],
    perc_right: int = 100,
) -> Tuple:
    """Under the steady-state assumption, alpha / gamma equals the total RNA. Gamma can be calculated from the slope of
    total RNA and new RNA. The relationship can be expressed as:
        l(t) = (1 - exp(- gamma * t)) alpha / gamma

    Args:
        R: A matrix representing total RNA. Can be expression or the first moments.
        N: A matrix representing new RNA. Can be expression or the first moments.
        time: A matrix with time information.
        perc_right: The percentile limitation to find extreme data points.

    Returns:
        Gamma, R squared, the slope k, the mean of R squared and the fitted k by time and gamma.
    """
    n_var = R.shape[0]
    mean_R2, gamma, r2 = np.zeros(n_var), np.zeros(n_var), np.zeros(n_var)
    K_list, K_fit_list = [None] * n_var, [None] * n_var
    for i, r, n in tqdm(
        zip(np.arange(n_var), R, N),
        "Estimate gamma via linear regression of t vs. -ln(1-K)",
    ):
        r = r.A.flatten() if issparse(r) else r.flatten()
        n = n.A.flatten() if issparse(n) else n.flatten()

        K_list[i], R2 = fit_labeling_synthesis(n, r, time, perc_right=perc_right)
        gamma[i], r2[i] = compute_gamma_synthesis(K_list[i], np.unique(time))
        K_fit_list[i] = np.unique(time) * gamma[i]
        mean_R2[i] = np.mean(R2)

    return gamma, r2, K_list, mean_R2, K_fit_list
