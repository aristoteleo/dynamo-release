from typing import Tuple, Union, Optional
from anndata import AnnData

from scipy.sparse import (
    csr_matrix,
    issparse,
    SparseEfficiencyWarning,
)
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.special import gammaln
from scipy.optimize import root, fsolve

from dynamo.tools.utils import find_extreme


def mle_cell_specific_poisson_ss(
        R: Union[np.ndarray, csr_matrix],
        N: Union[np.ndarray, csr_matrix],
        time: np.ndarray,
        gamma_init: np.ndarray,
        cell_total: np.ndarray,
        Total_smoothed,
        New_smoothed,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """"Infer parameters based on the cell specific Poisson model using maximum likelihood estimation under the
    steady-state assumption

    Args:
        R: The number of total mRNA counts for each gene in each cell. shape: (n_var, n_obs).
        N: The number of new mRNA counts for each gene in each cell. shape: (n_var, n_obs).
        time: The time point of each cell. shape: (n_obs,).
        gamma_init: The initial value of gamma. shape: (n_var,).
        cell_total: The total counts of reads for each cell. shape: (n_obs,).

    Returns:
        gamma: The estimated total mRNA degradation rate gamma. shape: (n_var,).
        gamma_r2: The R2 of gamma. shape: (n_var,).
        gamma_r2_raw: The R2 of gamma without correction. shape: (n_var,).
        alpha: The estimated gene specific transcription rate alpha. shape: (n_var,).

    """
    n_var = N.shape[0]
    n_obs = N.shape[1]
    cell_capture_rate = cell_total / np.median(cell_total)

    # When there is only one labeling duration we can obtain the analytical solution directly but cannot define the
    # goodness-of-fit.
    if len(np.unique(time)) == 1:
        gamma = np.zeros(n_var)
        gamma_r2 = np.ones(n_var)  # As goodness of fit could not be defined, all were set to 1.
        gamma_r2_raw = np.ones(n_var)
        alpha = np.zeros(n_var)
        for i, r, n, r_smooth, n_smooth in tqdm(
                zip(np.arange(n_var), R, N, Total_smoothed, New_smoothed),
                "Infer parameters via maximum likelihood estimation based on the CSP model under the steady-state assumption"
        ):
            n = n.A.flatten() if issparse(n) else n.flatten()
            r = r.A.flatten() if issparse(r) else r.flatten()
            n_smooth = n_smooth.A.flatten() if issparse(n_smooth) else n_smooth.flatten()
            r_smooth = r_smooth.A.flatten() if issparse(r_smooth) else r_smooth.flatten()
            t_unique = np.unique(time)
            mask = find_extreme(n_smooth, r_smooth, perc_left=None, perc_right=50)
            gamma[i] = - np.log(1 - np.mean(n[mask]) / np.mean(r[mask])) / t_unique
            alpha[i] = gamma[i]*np.mean(r[mask])/np.mean(cell_capture_rate[mask])
    else:
        gamma = np.zeros(n_var)
        gamma_r2 = np.zeros(n_var)
        gamma_r2_raw = np.zeros(n_var)
        alphadivgamma = np.zeros(n_var)
        for i, r, n in tqdm(
                zip(np.arange(n_var), R, N),
                "Infer parameters via maximum likelihood estimation based on the CSP model under the steady-state assumption"
        ):
            n = n.A.flatten() if issparse(n) else n.flatten()
            r = r.A.flatten() if issparse(r) else r.flatten()

            def loss_func_ss(parameters):
                # Loss function of cell specific Poisson model under the steady-state assumption
                parameter_alpha_div_gamma, parameter_gamma = parameters
                mu_new = parameter_alpha_div_gamma * (1 - np.exp(-parameter_gamma * time)) * cell_capture_rate
                loss_new = -np.sum(n * np.log(mu_new) - mu_new)
                mu_total = parameter_alpha_div_gamma * cell_capture_rate
                loss_total = -np.sum(r * np.log(mu_total) - mu_total)
                loss = loss_new + loss_total
                return loss

            # Initialize and add boundary conditions
            alpha_div_gamma_init = np.mean(n) / np.mean(cell_capture_rate * (1 - np.exp(-gamma_init[i] * time)))
            b1 = (0, 10 * alpha_div_gamma_init)
            b2 = (0, 10 * gamma_init[i])
            bnds = (b1, b2)
            parameters_init = np.array([alpha_div_gamma_init, gamma_init[i]])

            # Solve
            res = minimize(loss_func_ss, parameters_init, method='SLSQP', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
            # res = minimize(loss_func_ss, parameters_init, method='Nelder-Mead', tol=1e-2, options={'maxiter': 1000})
            # res = minimize(loss_func_ss, parameters_init, method='COBYLA', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
            parameters = res.x
            loss = res.fun
            success = res.success
            alphadivgamma[i], gamma[i] = parameters

            # Calculate deviance R2 as goodness of fit

            def null_loss_func_ss(parameters_null):
                # Loss function of null model under the steady-state assumption
                parameters_a0_new, parameters_a0_total = parameters_null
                mu_new = parameters_a0_new * cell_capture_rate
                loss0_new = -np.sum(n * np.log(mu_new) - mu_new)
                mu_total = parameters_a0_total * cell_capture_rate
                loss0_total = -np.sum(r * np.log(mu_total) - mu_total)
                loss0 = loss0_new + loss0_total
                return loss0

            def saturated_loss_func_ss():
                # Loss function of saturated model under the steady-state assumption
                loss_saturated_new = -np.sum(n[n > 0] * np.log(n[n > 0]) - n[n > 0])
                loss_saturated_total = -np.sum(r[r > 0] * np.log(r[r > 0]) - r[r > 0])
                loss_saturated = loss_saturated_new + loss_saturated_total
                return loss_saturated

            a0_new = np.mean(n) / np.mean(cell_capture_rate)
            a0_total = np.mean(r) / np.mean(cell_capture_rate)
            loss0 = null_loss_func_ss((a0_new, a0_total))

            loss_saturated = saturated_loss_func_ss()
            null_devanice = 2 * (loss0 - loss_saturated)
            devanice = 2 * (loss - loss_saturated)
            gamma_r2_raw[i] = 1 - (devanice / (2*n_obs - 2)) / (null_devanice / (2*n_obs - 2))

        # Top 40% genes were selected by goodness of fit
        gamma_r2 = gamma_r2_raw.copy()
        number_selected_genes = int(n_var * 0.4)
        gamma_r2[gamma < 0.01] = 0
        sort_index = np.argsort(-gamma_r2)
        gamma_r2[sort_index[:number_selected_genes]] = 1
        gamma_r2[sort_index[number_selected_genes + 1:]] = 0

        alpha = alphadivgamma*gamma

    return gamma, gamma_r2, gamma_r2_raw, alpha


def mle_cell_specific_poisson(
        N: Union[np.ndarray, csr_matrix],
        time: np.ndarray,
        gamma_init: np.ndarray,
        cell_total: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """"Infer parameters based on cell specific Poisson distributions using maximum likelihood estimation

    Args:
        N: The number of new mRNA counts for each gene in each cell. shape: (n_var, n_obs).
        time: The time point of each cell. shape: (n_obs,).
        gamma_init: The initial value of gamma. shape: (n_var,).
        cell_total: The total counts of reads for each cell. shape: (n_obs,).

    Returns:
        gamma: The estimated total mRNA degradation rate gamma. shape: (n_var,).
        gamma_r2: The R2 of gamma. shape: (n_var,).
        gamma_r2_raw: The R2 of gamma without correction. shape: (n_var,).
        alpha: The estimated gene specific transcription rate alpha. shape: (n_var,).
    """
    n_var = N.shape[0]
    n_obs = N.shape[1]
    gamma = np.zeros(n_var)
    gamma_r2 = np.zeros(n_var)
    gamma_r2_raw = np.zeros(n_var)
    alphadivgamma = np.zeros(n_var)
    for i, n in tqdm(
            zip(np.arange(n_var), N),
            "Infer parameters via maximum likelihood estimation based on the CSP model"
    ):
        n = n.A.flatten() if issparse(n) else n.flatten()
        cell_capture_rate = cell_total / np.median(cell_total)

        def loss_func(parameters):
            # Loss function of cell specific Poisson model
            parameter_alpha_div_gamma, parameter_gamma = parameters
            mu = parameter_alpha_div_gamma * (1 - np.exp(-parameter_gamma * time)) * cell_capture_rate
            loss = -np.sum(n * np.log(mu) - mu)
            return loss

        # Initialize and add boundary conditions
        alpha_div_gamma_init = np.mean(n) / np.mean(cell_capture_rate * (1 - np.exp(-gamma_init[i] * time)))
        b1 = (0, 10 * alpha_div_gamma_init)
        b2 = (0, 10 * gamma_init[i])
        bnds = (b1, b2)
        parameters_init = np.array([alpha_div_gamma_init, gamma_init[i]])

        # Solve
        res = minimize(loss_func, parameters_init, method='SLSQP', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
        # res = minimize(loss_func, parameters_init, method='Nelder-Mead', tol=1e-2, options={'maxiter': 1000})
        # res = minimize(loss_func, parameters_init, method='COBYLA', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
        parameters = res.x
        loss = res.fun
        success = res.success
        alphadivgamma[i], gamma[i] = parameters

        # Calculate deviance R2 as goodness of fit

        def null_loss_func(parameters_null):
            # Loss function of null model
            parameters_a0 = parameters_null
            mu = parameters_a0 * cell_capture_rate
            loss0 = -np.sum(n * np.log(mu) - mu)
            return loss0

        def saturated_loss_func():
            # Loss function of saturated model
            loss_saturated = -np.sum(n[n > 0] * np.log(n[n > 0]) - n[n > 0])
            return loss_saturated

        a0 = np.mean(n) / np.mean(cell_capture_rate)
        loss0 = null_loss_func(a0)

        loss_saturated = saturated_loss_func()
        null_devanice = 2 * (loss0 - loss_saturated)
        devanice = 2 * (loss - loss_saturated)
        gamma_r2_raw[i] = 1 - (devanice / (n_obs - 2)) / (null_devanice / (n_obs - 1))

    # Top 40% genes were selected by goodness of fit
    gamma_r2 = gamma_r2_raw.copy()
    number_selected_genes = int(n_var * 0.4)
    gamma_r2[gamma < 0.01] = 0
    sort_index = np.argsort(-gamma_r2)
    gamma_r2[sort_index[:number_selected_genes]] = 1
    gamma_r2[sort_index[number_selected_genes + 1:]] = 0

    return gamma, gamma_r2, gamma_r2_raw, alphadivgamma*gamma


def mle_cell_specific_zero_inflated_poisson(
        N: Union[np.ndarray, csr_matrix],
        time: np.ndarray,
        gamma_init: np.ndarray,
        cell_total: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """"Infer parameters based on cell specific zero-inflated Poisson distributions using maximum likelihood estimation

        Args:
        N: The number of new mRNA counts for each gene in each cell. shape: (n_var, n_obs).
        time: The time point of each cell. shape: (n_obs,).
        gamma_init: The initial value of gamma. shape: (n_var,).
        cell_total: The total counts of reads for each cell. shape: (n_obs,).

    Returns:
        gamma: The estimated total mRNA degradation rate gamma. shape: (n_var,).
        prob_off: The estimated probability of gene expression being in the off state $p_{off}$. shape: (n_var,).
        gamma_r2: The R2 of gamma. shape: (n_var,).
        gamma_r2_raw: The R2 of gamma without correction. shape: (n_var,).
        alpha: The estimated gene specific transcription rate alpha. shape: (n_var,).
    """
    n_var = N.shape[0]
    n_obs = N.shape[1]
    gamma = np.zeros(n_var)
    gamma_r2 = np.zeros(n_var)
    gamma_r2_raw = np.zeros(n_var)
    prob_off = np.zeros(n_var)
    alphadivgamma = np.zeros(n_var)

    for i, n in tqdm(
            zip(np.arange(n_var), N),
            "Infer parameters via maximum likelihood estimation based on the CSZIP model"
    ):
        n = n.A.flatten() if issparse(n) else n.flatten()
        cell_capture_rate = cell_total / np.median(cell_total)

        def loss_func(parameters):
            # Loss function of cell specific zero-inflated Poisson model
            parameter_alpha_div_gamma, parameter_gamma, parameter_prob_off = parameters
            mu = parameter_alpha_div_gamma * (1 - np.exp(-parameter_gamma * time)) * cell_capture_rate
            n_eq_0_index = n < 0.001
            n_over_0_index = n > 0.001
            loss_eq0 = -np.sum(np.log(parameter_prob_off + (1 - parameter_prob_off) * np.exp(-mu[n_eq_0_index])))
            loss_over0 = -np.sum(np.log(1 - parameter_prob_off) + (-mu[n_over_0_index]) + n[n_over_0_index] * np.log(
                mu[n_over_0_index]))
            loss = loss_eq0 + loss_over0
            return loss

        # Initialize and add boundary conditions
        mean_n = np.mean(n)
        s2_n = np.mean(np.power(n, 2))
        temp = np.mean(cell_capture_rate * (1 - np.exp(-gamma_init[i] * time)))
        prob_off_init = 1 - mean_n * mean_n * np.mean(
            np.power(cell_capture_rate * (1 - np.exp(-gamma_init[i] * time)), 2)) / (
                                temp * temp * (s2_n - mean_n))  # Use moment estimation as the initial value of prob_off
        alphadivgamma_init = mean_n / ((1 - prob_off_init) * temp)
        b1 = (0, 10 * alphadivgamma_init)
        b2 = (0, 10 * gamma_init[i])
        b3 = (0, (np.sum(n < 0.001) / np.sum(n > -1)))
        bnds = (b1, b2, b3)
        parameters_init = np.array([alphadivgamma_init, gamma_init[i], prob_off_init])

        # Slove
        res = minimize(loss_func, parameters_init, method='SLSQP', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
        # res = minimize(loss_func, parameters_init, method='Nelder-Mead', tol=1e-2, options={'maxiter': 1000})
        # res = minimize(loss_func, parameters_init, method='COBYLA', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
        parameters = res.x
        alphadivgamma[i], gamma[i], prob_off[i] = parameters
        loss = res.fun
        success = res.success

        # Calculate deviance R2 as goodness of fit

        def null_Loss_func(parameters_null):
            # Loss function of null model
            parameters_null_lambda, parameters_null_prob_off = parameters_null
            mu = parameters_null_lambda * cell_capture_rate
            n_eq_0_index = n < 0.0001
            n_over_0_index = n > 0.0001
            null_loss_eq0 = -np.sum(
                np.log(parameters_null_prob_off + (1 - parameters_null_prob_off) * np.exp(-mu[n_eq_0_index])))
            null_loss_over0 = -np.sum(
                np.log(1 - parameters_null_prob_off) + (-mu[n_over_0_index]) + n[n_over_0_index] * np.log(
                    mu[n_over_0_index]))
            null_loss = null_loss_eq0 + null_loss_over0
            return null_loss

        mean_cell_capture_rate = np.mean(cell_capture_rate)
        prob_off_init_null = 1 - mean_n * mean_n * np.mean(np.power(cell_capture_rate, 2)) / (
                mean_cell_capture_rate * mean_cell_capture_rate * (s2_n - mean_n))
        lambda_init_null = mean_n / ((1 - prob_off_init_null) * mean_cell_capture_rate)
        b1_null = (0, 10 * lambda_init_null)
        b2_null = (0, (np.sum(n < 0.001) / np.sum(n > -1)))
        bnds_null = (b1_null, b2_null)
        parameters_init_null = np.array([lambda_init_null, prob_off_init_null])
        res_null = minimize(null_Loss_func, parameters_init_null, method='SLSQP', bounds=bnds_null, tol=1e-2,
                            options={'maxiter': 1000})
        loss0 = res_null.fun

        def saturated_loss_func():
            loss_saturated = -np.sum(n[n > 0] * np.log(n[n > 0]) - n[n > 0])
            return loss_saturated

        loss_saturated = saturated_loss_func()
        null_devanice = 2 * (loss0 - loss_saturated)
        devanice = 2 * (loss - loss_saturated)

        gamma_r2_raw[i] = 1 - (devanice / (n_obs - 2)) / (null_devanice / (n_obs - 1))

    # Top 40% genes were selected by goodness of fit
    gamma_r2 = gamma_r2_raw.copy()
    number_selected_genes = int(n_var * 0.4)
    gamma_r2[gamma < 0.01] = 0
    sort_index = np.argsort(-gamma_r2)
    gamma_r2[sort_index[:number_selected_genes]] = 1
    gamma_r2[sort_index[number_selected_genes + 1:]] = 0

    return gamma, prob_off, gamma_r2, gamma_r2_raw, gamma*alphadivgamma


def mle_independent_cell_specific_poisson(
        UL: Union[np.ndarray, csr_matrix],
        SL: Union[np.ndarray, csr_matrix],
        time: np.ndarray,
        gamma_init: np.ndarray,
        beta_init: np.ndarray,
        cell_total: np.ndarray,
        Total_smoothed: Union[np.ndarray, csr_matrix],
        S_smoothed: Union[np.ndarray, csr_matrix]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """"Infer parameters based on independent cell specific Poisson distributions using maximum likelihood estimation

    Args:
        UL: The number of unspliced labeled mRNA counts for each gene in each cell. shape: (n_var, n_obs).
        SL: The number of spliced labeled mRNA counts for each gene in each cell. shape: (n_var, n_obs).
        time: The time point of each cell. shape: (n_obs,).
        gamma_init: The initial value of gamma. shape: (n_var,).
        beta_init: The initial value of beta. shape: (n_var,).
        cell_total: The total counts of reads for each cell. shape: (n_obs,).
        Total_smoothed: The number of total mRNA expression after normalization and smoothing for each gene in each cell. shape: (n_var, n_obs).
        S_smoothed: The number of spliced mRNA expression after normalization and smoothing for each gene in each cell. shape: (n_var, n_obs).

    Returns:
        gamma_s: The estimated spliced mRNA degradation rate gamma_s. shape: (n_var,).
        gamma_r2: The R2 of gamma. shape: (n_var,).
        beta: The estimated gene specific splicing rate beta. shape: (n_var,).
        gamma_t: The estimated total mRNA degradation rate gamma_t. shape: (n_var,).
        gamma_r2_raw: The R2 of gamma without correction. shape: (n_var,).
        alpha: The estimated gene specific transcription rate alpha. shape: (n_var,).
    """
    n_var = UL.shape[0]
    n_obs = UL.shape[1]
    gamma_s = np.zeros(n_var)
    gamma_r2 = np.zeros(n_var)
    gamma_r2_raw = np.zeros(n_var)
    beta = np.zeros(n_var)
    alpha = np.zeros(n_var)
    gamma_t = np.zeros(n_var)

    for i, ul, sl, r, s in tqdm(
            zip(np.arange(n_var), UL, SL, Total_smoothed, S_smoothed),
            "Estimate gamma via maximum likelihood estimation based on the ICSP model "
    ):
        sl = sl.A.flatten() if issparse(sl) else sl.flatten()
        ul = ul.A.flatten() if issparse(ul) else ul.flatten()
        r = r.A.flatten() if issparse(r) else r.flatten()
        s = s.A.flatten() if issparse(s) else s.flatten()

        cell_capture_rate = cell_total / np.median(cell_total)

        def loss_func(parameters):
            # Loss function of independent cell specific Poisson model
            parameter_alpha, parameter_beta, parameter_gamma_s = parameters
            mu_u = parameter_alpha / parameter_beta * (1 - np.exp(-parameter_beta * time)) * cell_capture_rate
            mu_s = (parameter_alpha / parameter_gamma_s * (1 - np.exp(-parameter_gamma_s * time)) + parameter_alpha /
                    (parameter_gamma_s - parameter_beta) * (np.exp(-parameter_gamma_s * time) - np.exp(
                        -parameter_beta * time))) * cell_capture_rate
            loss_u = -np.sum(ul * np.log(mu_u) - mu_u)
            loss_s = -np.sum(sl * np.log(mu_s) - mu_s)
            loss = loss_u + loss_s
            return loss

        # The initial values of gamma_s, beta and alpha are obtained from the initial values of gamma_t.
        gamma_s_init = gamma_init[i] * np.sum(r * s) / np.sum(np.power(s, 2))
        beta_init_new = beta_init[i] * gamma_s_init / gamma_init[i]
        alpha_init = np.mean(ul + sl) / np.mean(cell_capture_rate * (
                (1 - np.exp(-beta_init_new * time)) / beta_init_new + (1 - np.exp(-gamma_s_init * time)) / gamma_s_init
                + (np.exp(-gamma_s_init * time) - np.exp(-beta_init_new * time)) / (gamma_s_init - beta_init_new)))

        # Initialize and add boundary conditions
        b1 = (0, 10 * alpha_init)
        b2 = (0, 10 * beta_init_new)
        b3 = (0, 10 * gamma_s_init)
        bnds = (b1, b2, b3)
        parameters_init = np.array([alpha_init, beta_init_new, gamma_s_init])

        # Solve
        res = minimize(loss_func, parameters_init, method='SLSQP', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
        # res = minimize(loss_func, parameters_init, method='Nelder-Mead', tol=1e-2, options={'maxiter': 1000})
        # res = minimize(loss_func, parameters_init, method='COBYLA', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
        parameters = res.x
        loss = res.fun
        success = res.success
        alpha[i], beta[i], gamma_s[i] = parameters

        # Calculate deviance R2 as goodness of fit

        def null_loss_func(parameters_null):
            # Loss function of null model
            parameters_a0, parameters_b0 = parameters_null
            mu_u = parameters_a0 * cell_capture_rate
            mu_s = parameters_b0 * cell_capture_rate
            loss0_u = -np.sum(ul * np.log(mu_u) - mu_u)
            loss0_s = -np.sum(sl * np.log(mu_s) - mu_s)
            loss0 = loss0_u + loss0_s
            return loss0

        b0 = np.mean(ul) / np.mean(cell_capture_rate)
        c0 = np.mean(sl) / np.mean(cell_capture_rate)
        loss0 = null_loss_func((b0, c0))

        def saturated_loss_func():
            # Loss function of saturated model
            loss_saturated_u = -np.sum(ul[ul > 0] * np.log(ul[ul > 0]) - ul[ul > 0])
            loss_saturated_s = -np.sum(sl[sl > 0] * np.log(sl[sl > 0]) - sl[sl > 0])
            loss_saturated = loss_saturated_u + loss_saturated_s
            return loss_saturated

        loss_saturated = saturated_loss_func()
        null_devanice = 2 * (loss0 - loss_saturated)
        devanice = 2 * (loss - loss_saturated)
        gamma_r2_raw[i] = 1 - (devanice / (2 * n_obs - 3)) / (null_devanice / (2 * n_obs - 2))  # + 0.82

        gamma_t[i] = gamma_s[i] * np.sum(np.power(s, 2)) / np.sum(r * s)

    # Top 40% genes were selected by goodness of fit
    gamma_r2 = gamma_r2_raw.copy()
    number_selected_genes = int(n_var * 0.4)
    gamma_r2[gamma_s < 0.01] = 0
    sort_index = np.argsort(-gamma_r2)
    gamma_r2[sort_index[:number_selected_genes]] = 1
    gamma_r2[sort_index[number_selected_genes + 1:]] = 0

    return gamma_s, gamma_r2, beta, gamma_t, gamma_r2_raw, alpha


def cell_specific_alpha_beta(
        UL_smoothed_CSP: Union[np.ndarray, csr_matrix],
        SL_smoothed_CSP: Union[np.ndarray, csr_matrix],
        time: np.ndarray,
        gamma_init: np.ndarray,
        beta_init: np.ndarray
) -> Tuple[csr_matrix, csr_matrix]:
    """"Infer cell specific transcription rate and splicing rate based on ICSP model

    Args:
        UL_smoothed_CSP: The number of unspliced labeled mRNA expression after smoothing based on CSP type model for
        each gene in each cell. shape: (n_var, n_obs).
        SL_smoothed_CSP: The number of spliced labeled mRNA expression after smoothing based on CSP type model for
        each gene in each cell. shape: (n_var, n_obs).
        time: The time point of each cell. shape: (n_obs,).
        gamma_init: The gene wise initial value of gamma. shape: (n_var,).
        beta_init: The gene wise initial value of beta. shape: (n_var,).

    Returns: alpha_cs, beta_cs
        alpha_cs: The transcription rate for each gene in each cell. shape: (n_var, n_obs).
        beta_cs: The splicing rate for each gene in each cell. shape: (n_var, n_obs).
    """
    beta_cs = np.zeros_like(UL_smoothed_CSP.A) if issparse(UL_smoothed_CSP) else np.zeros_like(UL_smoothed_CSP)

    n_var = UL_smoothed_CSP.shape[0]
    n_obs = UL_smoothed_CSP.shape[1]

    for i, ul, sl, gamma_i, beta_i in tqdm(
            zip(np.arange(n_var), UL_smoothed_CSP, SL_smoothed_CSP, gamma_init, beta_init),
            "Estimate cell specific alpha and beta"
    ):
        sl = sl.A.flatten() if issparse(sl) else sl.flatten()
        ul = ul.A.flatten() if issparse(ul) else ul.flatten()

        for j in range(n_obs):
            sl_j = sl[j]
            ul_j = ul[j]
            sl_div_ul_j = sl_j / ul_j
            time_j = time[j]

            def solve_beta_func(beta_j):
                # Equation for solving cell specific beta
                return sl_div_ul_j - (1 - np.exp(-gamma_i * time_j)) / gamma_i * beta_j / (1 - np.exp(-beta_j * time_j)) \
                       - beta_j / (gamma_i - beta_j) * (np.exp(-gamma_i * time_j) - np.exp(-beta_j * time_j)) / \
                       (1 - np.exp(-beta_j * time_j))

            beta_j_solve = root(solve_beta_func, beta_i)
            # beta_j_solve = fsolve(solve_beta_func, beta_i)

            beta_cs[i, j] = beta_j_solve.x

    k = 1 - np.exp(-beta_cs * (np.tile(time, (n_var, 1))))
    beta_cs = csr_matrix(beta_cs)
    alpha_cs = beta_cs.multiply(UL_smoothed_CSP).multiply(1 / k)
    return alpha_cs, beta_cs


def visualize_CSP_loss_landscape(
        adata: AnnData,
        gene_name_list: list,
        figsize: tuple = (3, 3),
        dpi: int = 75,
        save_name: Optional[str] = None):
    """"Draw the landscape of CSP model-based loss function for the given genes.

    Args:
        adata: class:`~anndata.AnnData`
            an Annodata object
        gene_name_list: A list of gene names that are going to be visualized.
        figsize: The width and height of each panel in the figure.
        dpi: The dot per inch of the figure.
        save_name: The save path for visualization results. save_name = None means that only show but not save the
        results.

    Returns:
    -------
        A matplotlib plot that shows the landscape of CSP model-based loss function for the given genes.
    """

    def _traverse_CSP(n, time, gamma_init, cell_total):
        """Traverse the CSP loss function to draw the landscape"""
        n = n.A.flatten() if issparse(n) else n.flatten()
        cell_capture_rate = cell_total / np.median(cell_total)

        def loss_func(parameters):
            # Loss function of cell specific Poisson model
            parameter_alpha_div_gamma, parameter_gamma = parameters
            mu = parameter_alpha_div_gamma * (1 - np.exp(-parameter_gamma * time)) * cell_capture_rate
            loss = -np.sum(n * np.log(mu) - mu - gammaln(n + 1))
            return loss

        def dldalpha_eq0(gamma):
            # Analytic solution to the equation that the derivative of the loss with respect to alpha is equal to 0
            alpha_div_gamma_dldalpha_eq0 = np.mean(n) / np.mean(cell_capture_rate * (1 - np.exp(-gamma * time)))
            return alpha_div_gamma_dldalpha_eq0

        def alpha_constant(gamma):
            # When gamma is sufficiently small, alpha is approximated as a constant.
            alpha_div_gamma_constant = np.mean(n) / np.mean(cell_capture_rate * (gamma * time))
            return alpha_div_gamma_constant

        # Determine the scope of the traversal
        alpha_div_gamma_init = np.mean(n / (1 - np.exp(-gamma_init * time)))
        gamma_range = gamma_init * np.logspace(-2, 1, base=5, num=200)
        alpha_div_gamma_range = alpha_div_gamma_init * np.logspace(-2, 1, base=5, num=200)

        # Iterate over the value of the loss function in the given range
        loss_all = np.zeros((len(gamma_range), len(alpha_div_gamma_range)))
        for s in range(len(gamma_range)):
            for t in range(len(alpha_div_gamma_range)):
                gamma_temp = gamma_range[s]
                alpha_div_gamma_temp = alpha_div_gamma_range[t]
                loss_all[s, t] = loss_func((alpha_div_gamma_temp, gamma_temp))

        # Create grid data for drawing
        X, Y = np.meshgrid(gamma_range, alpha_div_gamma_range)
        Z = np.transpose(loss_all)

        # Calculate the loss value where dl/dalpha is equal to 0 and alpha is equal to a constant
        alpha_div_gamma_dldalpha_eq0_range = np.zeros_like(gamma_range)
        alpha_div_gamma_constant_range = np.zeros_like(gamma_range)
        loss_dldalpha_eq0_range = np.zeros_like(gamma_range)
        loss_constant_range = np.zeros_like(gamma_range)
        for s in range(len(gamma_range)):
            alpha_div_gamma_dldalpha_eq0_range[s] = dldalpha_eq0(gamma_range[s])
            alpha_div_gamma_constant_range[s] = alpha_constant(gamma_range[s])
            loss_dldalpha_eq0_range[s] = loss_func((alpha_div_gamma_dldalpha_eq0_range[s], gamma_range[s]))
            loss_constant_range[s] = loss_func((alpha_div_gamma_constant_range[s], gamma_range[s]))

        return X, Y, Z, gamma_range, alpha_div_gamma_dldalpha_eq0_range, \
               alpha_div_gamma_constant_range, loss_dldalpha_eq0_range, loss_constant_range

    def _plot_landscape(X, Y, Z, gamma, alpha_div_gamma_dldalpha_eq0, alpha_div_gamma_constant,
                        loss_dldalpha_eq0, loss_constant, figsize, dpi, gene_name, save_name):
        """Function to draw the landscape, dl/d$\alpha$ and $\alpha_cons$."""

        # Adjust the range of the parameter to make the results clearer
        index1 = np.where(np.logical_and(gamma > np.min(X), gamma < np.max(X)))
        index2_dldgeq0 = np.where(
            np.logical_and(alpha_div_gamma_dldalpha_eq0 > np.min(Y), alpha_div_gamma_dldalpha_eq0 < np.max(Y)))
        index_dldgeq0 = np.intersect1d(index1, index2_dldgeq0)
        index2_constant = np.where(
            np.logical_and(alpha_div_gamma_constant > np.min(Y), alpha_div_gamma_constant < np.max(Y)))
        index_constant = np.intersect1d(index1, index2_constant)

        # Create figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        plt.tick_params(pad=-2)

        # Create plot
        surf = ax.plot_surface(X, Y, Z, cmap='rainbow', rstride=1, cstride=1, alpha=0.75)
        ax.plot(gamma[index_dldgeq0], alpha_div_gamma_dldalpha_eq0[index_dldgeq0], loss_dldalpha_eq0[index_dldgeq0],
                color='black',
                linewidth=1, label='$\\frac{\partial \ell}{\partial \\alpha}(\\alpha, \gamma_{t})=0$')
        ax.plot(gamma[index_constant], alpha_div_gamma_constant[index_constant], loss_constant[index_constant],
                color='red',
                linewidth=1, label='$\\alpha=\\alpha_{cons}$')
        plt.legend()

        cax = fig.add_axes([0.005, 0.15, 0.025, 0.75])  # left down right up
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, cax=cax)

        # Add labels
        ax.set_xlabel('$\gamma_{t}$', labelpad=-7)
        ax.set_ylabel('$\\alpha/\gamma_{t}$', labelpad=-7)
        ax.set_zlabel('$-\ell(\\alpha,\gamma_{t})$', labelpad=-7)
        ax.set_zlim(np.min(Z), np.max(Z))
        ax.set_title(f'Loss function landscape of for gene {gene_name}')
        ax.zaxis.get_major_formatter().set_powerlimits((0, 1))

        # ax.view_init(azim=-50)
        fig.tight_layout()
        plt.grid(False)
        if save_name:
            plt.savefig(save_name)
        plt.show()

    sub_adata = adata[:, gene_name_list]
    cell_total = sub_adata.obs['initial_cell_size'].astype("float").values
    time = sub_adata.obs['time']
    N = sub_adata.layers['new'].T
    gamma_init = sub_adata.var['gamma']
    n_var = len(gene_name_list)
    for i, n, gene, gamma_init_i in tqdm(
            zip(np.arange(n_var), N, gene_name_list, gamma_init),
            'Visualize the landscape of the CSP model loss function'
    ):
        X, Y, Z, gamma, alpha_div_gamma_dldalpha_eq0, alpha_div_gamma_constant, loss_dldalpha_eq0, loss_constant = \
            _traverse_CSP(n, time, gamma_init_i, cell_total)
        _plot_landscape(X, Y, Z, gamma, alpha_div_gamma_dldalpha_eq0, alpha_div_gamma_constant, loss_dldalpha_eq0,
                        loss_constant, figsize, dpi, gene, save_name)


def robustness_measure_CSP(
        adata: AnnData,
        gene_name_list: list,
) -> np.ndarray:
    """Calculate the robustness measure based on CSP model inference of the given genes

    Args:
        adata: class:`~anndata.AnnData`
            an Annodata object
        gene_name_list: A list of gene names that are going to be calculated robustness measure based on CSP model.

    Returns:
        robustness_measure: The robustness measure based on CSP model inference of the given genes.
        shape: (len(gene_name_list),).
    """
    sub_adata = adata[:, gene_name_list]
    cell_total = sub_adata.obs['initial_cell_size'].astype("float").values
    time = sub_adata.obs['time']
    N = sub_adata.layers['new'].T
    robustness_measure = calculate_robustness_measure_CSP(N, time, cell_total)
    return robustness_measure


def calculate_robustness_measure_CSP(
        N: Union[np.ndarray, csr_matrix],
        time: np.ndarray,
        cell_total: np.ndarray
) -> np.ndarray:
    """Calculate the robustness measure based on CSP model inference

    Args:
        N: The number of new mRNA counts for each gene in each cell. shape: (n_var, n_obs).
        time: The time point of each cell. shape: (n_obs,).
        cell_total: The total counts of reads for each cell. shape: (n_obs,).

    Returns:
        robustness_measure: The robustness measure based on CSP model inference for each gene. shape: (n_var,).
    """
    n_var = N.shape[0]
    robustness_measure = np.zeros(n_var)
    for i, n in tqdm(
            zip(np.arange(n_var), N),
            "Calculate the robustness measure"
    ):
        n = n.A.flatten() if issparse(n) else n.flatten()
        cell_capture_rate = cell_total / np.median(cell_total)

        def partial_loss_partial_gamma(parameters):
            # Partial derivative of loss with respect to gamma.
            parameter_gamma = parameters
            optimal_alphadivgamma = np.mean(n) / np.mean(cell_capture_rate * (1 - np.exp(-parameter_gamma * time)))
            pLoss_pgamma = np.sum(-n * time * np.exp(-parameter_gamma * time) / (1 - np.exp(
                -parameter_gamma * time)) + cell_capture_rate * optimal_alphadivgamma * time * np.exp(
                -parameter_gamma * time))
            return pLoss_pgamma

        def loss_func(parameters):
            # Loss function of cell specific Poisson model
            parameter_alpha_div_gamma, parameter_gamma = parameters
            mu = parameter_alpha_div_gamma * (1 - np.exp(-parameter_gamma * time)) * cell_capture_rate
            loss = -np.sum(n * np.log(mu) - mu - gammaln(n + 1))
            return loss

        gamma_range = np.arange(0.01, 1.51, 0.01)
        loss = np.zeros_like(gamma_range)
        p_loss_p_gamma = np.zeros_like(gamma_range)
        for s in range(len(gamma_range)):
            gamma_temp = gamma_range[s]
            alpha_div_gamma_temp = np.mean(n) / np.mean(cell_capture_rate * (1 - np.exp(-gamma_temp * time)))
            p_loss_p_gamma[s] = partial_loss_partial_gamma(gamma_temp)
            loss[s] = loss_func((gamma_temp, alpha_div_gamma_temp))

        # robust_measure[i] = np.mean(np.abs(p_loss_p_gamma))
        robustness_measure[i] = np.sum(np.abs(loss[1:] - loss[0:-1]))

    return robustness_measure
