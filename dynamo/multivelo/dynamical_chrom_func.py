from dynamo.multivelo import settings

import os
import sys
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.optimize import minimize
from scipy.spatial import KDTree
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture


import pandas as pd
import seaborn as sns
from numba import njit
import numba
from numba.typed import List
from tqdm.auto import tqdm
import scipy as sp

import math
import torch
from torch import nn

current_path = os.path.dirname(__file__)
src_path = os.path.join(current_path, "..")
sys.path.append(src_path)

from ..dynamo_logger import (
    LoggerManager,
    main_exception,
    main_info,
)



# a funciton to check for invalid values of different parameters
def check_params(alpha_c,
                 alpha,
                 beta,
                 gamma,
                 c0=None,
                 u0=None,
                 s0=None):

    new_alpha_c = alpha_c
    new_alpha = alpha
    new_beta = beta
    new_gamma = gamma

    new_c0 = c0
    new_u0 = u0
    new_s0 = s0

    inf_fix = 1e10
    zero_fix = 1e-10

    # check if any of our parameters are infinite
    if c0 is not None and math.isinf(c0):
        main_info("c0 is infinite.", indent_level=1)
        new_c0 = inf_fix
    if u0 is not None and math.isinf(u0):
        main_info("u0 is infinite.", indent_level=1)
        new_u0 = inf_fix
    if s0 is not None and math.isinf(s0):
        main_info("s0 is infinite.", indent_level=1)
        new_s0 = inf_fix
    if math.isinf(alpha_c):
        new_alpha_c = inf_fix
        main_info("alpha_c is infinite.", indent_level=1)
    if math.isinf(alpha):
        new_alpha = inf_fix
        main_info("alpha is infinite.", indent_level=1)
    if math.isinf(beta):
        new_beta = inf_fix
        main_info("beta is infinite.", indent_level=1)
    if math.isinf(gamma):
        new_gamma = inf_fix
        main_info("gamma is infinite.", indent_level=1)

    # check if any of our parameters are nan
    if c0 is not None and math.isnan(c0):
        main_info("c0 is Nan.", indent_level=1)
        new_c0 = zero_fix
    if u0 is not None and math.isnan(u0):
        main_info("u0 is Nan.", indent_level=1)
        new_u0 = zero_fix
    if s0 is not None and math.isnan(s0):
        main_info("s0 is Nan.", indent_level=1)
        new_s0 = zero_fix
    if math.isnan(alpha_c):
        new_alpha_c = zero_fix
        main_info("alpha_c is Nan.", indent_level=1)
    if math.isnan(alpha):
        new_alpha = zero_fix
        main_info("alpha is Nan.", indent_level=1)
    if math.isnan(beta):
        new_beta = zero_fix
        main_info("beta is Nan.", indent_level=1)
    if math.isnan(gamma):
        new_gamma = zero_fix
        main_info("gamma is Nan.", indent_level=1)

    # check if any of our rate parameters are 0
    if alpha_c < 1e-7:
        new_alpha_c = zero_fix
        main_info("alpha_c is zero.", indent_level=1)
    if alpha < 1e-7:
        new_alpha = zero_fix
        main_info("alpha is zero.", indent_level=1)
    if beta < 1e-7:
        new_beta = zero_fix
        main_info("beta is zero.", indent_level=1)
    if gamma < 1e-7:
        new_gamma = zero_fix
        main_info("gamma is zero.", indent_level=1)

    if beta == alpha_c:
        new_beta += zero_fix
        main_info("alpha_c and beta are equal, leading to divide by zero",
                   indent_level=1)
    if beta == gamma:
        new_gamma += zero_fix
        main_info("gamma and beta are equal, leading to divide by zero",
                   indent_level=1)
    if alpha_c == gamma:
        new_gamma += zero_fix
        main_info("gamma and alpha_c are equal, leading to divide by zero",
                   indent_level=1)

    if c0 is not None and u0 is not None and s0 is not None:
        return new_alpha_c, new_alpha, new_beta, new_gamma, new_c0, new_u0, \
               new_s0

    return new_alpha_c, new_alpha, new_beta, new_gamma


@njit(
    locals={
            "res": numba.types.float64[:, ::1],
            "eat": numba.types.float64[::1],
            "ebt": numba.types.float64[::1],
            "egt": numba.types.float64[::1],
    },
    fastmath=True)
def predict_exp(tau,
                c0,
                u0,
                s0,
                alpha_c,
                alpha,
                beta,
                gamma,
                scale_cc=1,
                pred_r=True,
                chrom_open=True,
                backward=False,
                rna_only=False):

    if len(tau) == 0:
        return np.empty((0, 3))
    if backward:
        tau = -tau
    res = np.empty((len(tau), 3))
    eat = np.exp(-alpha_c * tau)
    ebt = np.exp(-beta * tau)
    egt = np.exp(-gamma * tau)
    if rna_only:
        kc = 1
        c0 = 1
    else:
        if chrom_open:
            kc = 1
        else:
            kc = 0
            alpha_c *= scale_cc

    const = (kc - c0) * alpha / (beta - alpha_c)

    res[:, 0] = kc - (kc - c0) * eat

    if pred_r:

        res[:, 1] = u0 * ebt + (alpha * kc / beta) * (1 - ebt)
        res[:, 1] += const * (ebt - eat)

        res[:, 2] = s0 * egt + (alpha * kc / gamma) * (1 - egt)
        res[:, 2] += ((beta / (gamma - beta)) *
                      ((alpha * kc / beta) - u0 - const) * (egt - ebt))
        res[:, 2] += (beta / (gamma - alpha_c)) * const * (egt - eat)

    else:
        res[:, 1] = np.zeros(len(tau))
        res[:, 2] = np.zeros(len(tau))
    return res


@njit(locals={
            "exp_sw1": numba.types.float64[:, ::1],
            "exp_sw2": numba.types.float64[:, ::1],
            "exp_sw3": numba.types.float64[:, ::1],
            "exp1": numba.types.float64[:, ::1],
            "exp2": numba.types.float64[:, ::1],
            "exp3": numba.types.float64[:, ::1],
            "exp4": numba.types.float64[:, ::1],
            "tau_sw1": numba.types.float64[::1],
            "tau_sw2": numba.types.float64[::1],
            "tau_sw3": numba.types.float64[::1],
            "tau1": numba.types.float64[::1],
            "tau2": numba.types.float64[::1],
            "tau3": numba.types.float64[::1],
            "tau4": numba.types.float64[::1]
    },
    fastmath=True)
def generate_exp(tau_list,
                 t_sw_array,
                 alpha_c,
                 alpha,
                 beta,
                 gamma,
                 scale_cc=1,
                 model=1,
                 rna_only=False):

    if beta == alpha_c:
        beta += 1e-3
    if gamma == beta or gamma == alpha_c:
        gamma += 1e-3
    switch = len(t_sw_array)
    if switch >= 1:
        tau_sw1 = np.array([t_sw_array[0]])
        if switch >= 2:
            tau_sw2 = np.array([t_sw_array[1] - t_sw_array[0]])
            if switch == 3:
                tau_sw3 = np.array([t_sw_array[2] - t_sw_array[1]])
    exp_sw1, exp_sw2, exp_sw3 = (np.empty((0, 3)),
                                 np.empty((0, 3)),
                                 np.empty((0, 3)))
    if tau_list is None:
        if model == 0:
            if switch >= 1:
                exp_sw1 = predict_exp(tau_sw1, 0, 0, 0, alpha_c, alpha, beta,
                                      gamma, pred_r=False, scale_cc=scale_cc,
                                      rna_only=rna_only)
                if switch >= 2:
                    exp_sw2 = predict_exp(tau_sw2, exp_sw1[0, 0],
                                          exp_sw1[0, 1], exp_sw1[0, 2],
                                          alpha_c, alpha, beta, gamma,
                                          pred_r=False, chrom_open=False,
                                          scale_cc=scale_cc, rna_only=rna_only)
                    if switch >= 3:
                        exp_sw3 = predict_exp(tau_sw3, exp_sw2[0, 0],
                                              exp_sw2[0, 1], exp_sw2[0, 2],
                                              alpha_c, alpha, beta, gamma,
                                              chrom_open=False,
                                              scale_cc=scale_cc,
                                              rna_only=rna_only)
        elif model == 1:
            if switch >= 1:
                exp_sw1 = predict_exp(tau_sw1, 0, 0, 0, alpha_c, alpha, beta,
                                      gamma, pred_r=False, scale_cc=scale_cc,
                                      rna_only=rna_only)
                if switch >= 2:
                    exp_sw2 = predict_exp(tau_sw2, exp_sw1[0, 0],
                                          exp_sw1[0, 1], exp_sw1[0, 2],
                                          alpha_c, alpha, beta, gamma,
                                          scale_cc=scale_cc, rna_only=rna_only)
                    if switch >= 3:
                        exp_sw3 = predict_exp(tau_sw3, exp_sw2[0, 0],
                                              exp_sw2[0, 1], exp_sw2[0, 2],
                                              alpha_c, alpha, beta, gamma,
                                              chrom_open=False,
                                              scale_cc=scale_cc,
                                              rna_only=rna_only)
        elif model == 2:
            if switch >= 1:
                exp_sw1 = predict_exp(tau_sw1, 0, 0, 0, alpha_c, alpha, beta,
                                      gamma, pred_r=False, scale_cc=scale_cc,
                                      rna_only=rna_only)
                if switch >= 2:
                    exp_sw2 = predict_exp(tau_sw2, exp_sw1[0, 0],
                                          exp_sw1[0, 1], exp_sw1[0, 2],
                                          alpha_c, alpha, beta, gamma,
                                          scale_cc=scale_cc, rna_only=rna_only)
                    if switch >= 3:
                        exp_sw3 = predict_exp(tau_sw3, exp_sw2[0, 0],
                                              exp_sw2[0, 1], exp_sw2[0, 2],
                                              alpha_c, 0, beta, gamma,
                                              scale_cc=scale_cc,
                                              rna_only=rna_only)

        return (np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)),
                np.empty((0, 3))), (exp_sw1, exp_sw2, exp_sw3)

    tau1 = tau_list[0]
    if switch >= 1:
        tau2 = tau_list[1]
        if switch >= 2:
            tau3 = tau_list[2]
            if switch == 3:
                tau4 = tau_list[3]
    exp1, exp2, exp3, exp4 = (np.empty((0, 3)), np.empty((0, 3)),
                              np.empty((0, 3)), np.empty((0, 3)))
    if model == 0:
        exp1 = predict_exp(tau1, 0, 0, 0, alpha_c, alpha, beta, gamma,
                           pred_r=False, scale_cc=scale_cc, rna_only=rna_only)
        if switch >= 1:
            exp_sw1 = predict_exp(tau_sw1, 0, 0, 0, alpha_c, alpha, beta,
                                  gamma, pred_r=False, scale_cc=scale_cc,
                                  rna_only=rna_only)
            exp2 = predict_exp(tau2, exp_sw1[0, 0], exp_sw1[0, 1],
                               exp_sw1[0, 2], alpha_c, alpha, beta, gamma,
                               pred_r=False, chrom_open=False,
                               scale_cc=scale_cc, rna_only=rna_only)
            if switch >= 2:
                exp_sw2 = predict_exp(tau_sw2, exp_sw1[0, 0], exp_sw1[0, 1],
                                      exp_sw1[0, 2], alpha_c, alpha, beta,
                                      gamma, pred_r=False, chrom_open=False,
                                      scale_cc=scale_cc, rna_only=rna_only)
                exp3 = predict_exp(tau3, exp_sw2[0, 0], exp_sw2[0, 1],
                                   exp_sw2[0, 2], alpha_c, alpha, beta, gamma,
                                   chrom_open=False, scale_cc=scale_cc,
                                   rna_only=rna_only)
                if switch == 3:
                    exp_sw3 = predict_exp(tau_sw3, exp_sw2[0, 0],
                                          exp_sw2[0, 1], exp_sw2[0, 2],
                                          alpha_c, alpha, beta, gamma,
                                          chrom_open=False, scale_cc=scale_cc,
                                          rna_only=rna_only)
                    exp4 = predict_exp(tau4, exp_sw3[0, 0], exp_sw3[0, 1],
                                       exp_sw3[0, 2], alpha_c, 0, beta, gamma,
                                       chrom_open=False, scale_cc=scale_cc,
                                       rna_only=rna_only)
    elif model == 1:
        exp1 = predict_exp(tau1, 0, 0, 0, alpha_c, alpha, beta, gamma,
                           pred_r=False, scale_cc=scale_cc, rna_only=rna_only)
        if switch >= 1:
            exp_sw1 = predict_exp(tau_sw1, 0, 0, 0, alpha_c, alpha, beta,
                                  gamma, pred_r=False, scale_cc=scale_cc,
                                  rna_only=rna_only)
            exp2 = predict_exp(tau2, exp_sw1[0, 0], exp_sw1[0, 1],
                               exp_sw1[0, 2], alpha_c, alpha, beta, gamma,
                               scale_cc=scale_cc, rna_only=rna_only)
            if switch >= 2:
                exp_sw2 = predict_exp(tau_sw2, exp_sw1[0, 0], exp_sw1[0, 1],
                                      exp_sw1[0, 2], alpha_c, alpha, beta,
                                      gamma, scale_cc=scale_cc,
                                      rna_only=rna_only)
                exp3 = predict_exp(tau3, exp_sw2[0, 0], exp_sw2[0, 1],
                                   exp_sw2[0, 2], alpha_c, alpha, beta, gamma,
                                   chrom_open=False, scale_cc=scale_cc,
                                   rna_only=rna_only)
                if switch == 3:
                    exp_sw3 = predict_exp(tau_sw3, exp_sw2[0, 0],
                                          exp_sw2[0, 1], exp_sw2[0, 2],
                                          alpha_c, alpha, beta, gamma,
                                          chrom_open=False, scale_cc=scale_cc,
                                          rna_only=rna_only)
                    exp4 = predict_exp(tau4, exp_sw3[0, 0], exp_sw3[0, 1],
                                       exp_sw3[0, 2], alpha_c, 0, beta, gamma,
                                       chrom_open=False, scale_cc=scale_cc,
                                       rna_only=rna_only)
    elif model == 2:
        exp1 = predict_exp(tau1, 0, 0, 0, alpha_c, alpha, beta, gamma,
                           pred_r=False, scale_cc=scale_cc, rna_only=rna_only)
        if switch >= 1:
            exp_sw1 = predict_exp(tau_sw1, 0, 0, 0, alpha_c, alpha, beta,
                                  gamma, pred_r=False, scale_cc=scale_cc,
                                  rna_only=rna_only)
            exp2 = predict_exp(tau2, exp_sw1[0, 0], exp_sw1[0, 1],
                               exp_sw1[0, 2], alpha_c, alpha, beta, gamma,
                               scale_cc=scale_cc, rna_only=rna_only)
            if switch >= 2:
                exp_sw2 = predict_exp(tau_sw2, exp_sw1[0, 0], exp_sw1[0, 1],
                                      exp_sw1[0, 2], alpha_c, alpha, beta,
                                      gamma, scale_cc=scale_cc,
                                      rna_only=rna_only)
                exp3 = predict_exp(tau3, exp_sw2[0, 0], exp_sw2[0, 1],
                                   exp_sw2[0, 2], alpha_c, 0, beta, gamma,
                                   scale_cc=scale_cc, rna_only=rna_only)
                if switch == 3:
                    exp_sw3 = predict_exp(tau_sw3, exp_sw2[0, 0],
                                          exp_sw2[0, 1], exp_sw2[0, 2],
                                          alpha_c, 0, beta, gamma,
                                          scale_cc=scale_cc, rna_only=rna_only)
                    exp4 = predict_exp(tau4, exp_sw3[0, 0], exp_sw3[0, 1],
                                       exp_sw3[0, 2], alpha_c, 0, beta, gamma,
                                       chrom_open=False, scale_cc=scale_cc,
                                       rna_only=rna_only)
    return (exp1, exp2, exp3, exp4), (exp_sw1, exp_sw2, exp_sw3)


@njit(locals={
            "exp_sw1": numba.types.float64[:, ::1],
            "exp_sw2": numba.types.float64[:, ::1],
            "exp_sw3": numba.types.float64[:, ::1],
            "exp1": numba.types.float64[:, ::1],
            "exp2": numba.types.float64[:, ::1],
            "exp3": numba.types.float64[:, ::1],
            "exp4": numba.types.float64[:, ::1],
            "tau_sw1": numba.types.float64[::1],
            "tau_sw2": numba.types.float64[::1],
            "tau_sw3": numba.types.float64[::1],
            "tau1": numba.types.float64[::1],
            "tau2": numba.types.float64[::1],
            "tau3": numba.types.float64[::1],
            "tau4": numba.types.float64[::1]
    },
    fastmath=True)
def generate_exp_backward(tau_list, t_sw_array, alpha_c, alpha, beta, gamma,
                          scale_cc=1, model=1, t=None):
    if beta == alpha_c:
        beta += 1e-3
    if gamma == beta or gamma == alpha_c:
        gamma += 1e-3
    switch = len(t_sw_array)
    if switch >= 1:
        tau_sw1 = np.array([t_sw_array[0]])
        if switch >= 2:
            tau_sw2 = np.array([t_sw_array[1] - t_sw_array[0]])
    else:
        tau_sw1 = np.array([t_sw_array[0]])
        tau_sw2 = np.array([t_sw_array[1] - t_sw_array[0]])

    if t is None:
        if model == 0:
            exp_sw1 = predict_exp(tau_sw1, 1e-3, 1e-3, 1e-3, alpha_c, 0, beta,
                                  gamma, scale_cc=scale_cc, chrom_open=False,
                                  backward=True)
            exp_sw2 = predict_exp(tau_sw2, exp_sw1[0, 0], exp_sw1[0, 1],
                                  exp_sw1[0, 2], alpha_c, alpha, beta, gamma,
                                  scale_cc=scale_cc, chrom_open=False,
                                  backward=True)
        elif model == 1:
            exp_sw1 = predict_exp(tau_sw1, 1e-3, 1e-3, 1e-3, alpha_c, 0, beta,
                                  gamma, scale_cc=scale_cc, chrom_open=False,
                                  backward=True)
            exp_sw2 = predict_exp(tau_sw2, exp_sw1[0, 0], exp_sw1[0, 1],
                                  exp_sw1[0, 2], alpha_c, alpha, beta, gamma,
                                  scale_cc=scale_cc, chrom_open=False,
                                  backward=True)
        elif model == 2:
            exp_sw1 = predict_exp(tau_sw1, 1e-3, 1e-3, 1e-3, alpha_c, 0, beta,
                                  gamma, scale_cc=scale_cc, chrom_open=False,
                                  backward=True)
            exp_sw2 = predict_exp(tau_sw2, exp_sw1[0, 0], exp_sw1[0, 1],
                                  exp_sw1[0, 2], alpha_c, 0, beta, gamma,
                                  scale_cc=scale_cc, backward=True)
        return (np.empty((0, 0)),
                np.empty((0, 0)),
                np.empty((0, 0))), (exp_sw1, exp_sw2)

    tau1 = tau_list[0]
    if switch >= 1:
        tau2 = tau_list[1]
        if switch >= 2:
            tau3 = tau_list[2]

    exp1, exp2, exp3 = np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
    if model == 0:
        exp1 = predict_exp(tau1, 1e-3, 1e-3, 1e-3, alpha_c, 0, beta, gamma,
                           scale_cc=scale_cc, chrom_open=False, backward=True)
        if switch >= 1:
            exp_sw1 = predict_exp(tau_sw1, 1e-3, 1e-3, 1e-3, alpha_c, 0, beta,
                                  gamma, scale_cc=scale_cc, chrom_open=False,
                                  backward=True)
            exp2 = predict_exp(tau2, exp_sw1[0, 0], exp_sw1[0, 1],
                               exp_sw1[0, 2], alpha_c, alpha, beta, gamma,
                               scale_cc=scale_cc, chrom_open=False,
                               backward=True)
            if switch >= 2:
                exp_sw2 = predict_exp(tau_sw2, exp_sw1[0, 0], exp_sw1[0, 1],
                                      exp_sw1[0, 2], alpha_c, alpha, beta,
                                      gamma, scale_cc=scale_cc,
                                      chrom_open=False, backward=True)
                exp3 = predict_exp(tau_sw2, exp_sw1[0, 0], exp_sw1[0, 1],
                                   exp_sw1[0, 2], alpha_c, alpha, beta, gamma,
                                   scale_cc=scale_cc, chrom_open=False,
                                   backward=True)
    elif model == 1:
        exp1 = predict_exp(tau1, 1e-3, 1e-3, 1e-3, alpha_c, 0, beta, gamma,
                           scale_cc=scale_cc, chrom_open=False, backward=True)
        if switch >= 1:
            exp_sw1 = predict_exp(tau_sw1, 1e-3, 1e-3, 1e-3, alpha_c, 0, beta,
                                  gamma, scale_cc=scale_cc, chrom_open=False,
                                  backward=True)
            exp2 = predict_exp(tau2, exp_sw1[0, 0], exp_sw1[0, 1],
                               exp_sw1[0, 2], alpha_c, alpha, beta, gamma,
                               scale_cc=scale_cc, chrom_open=False,
                               backward=True)
            if switch >= 2:
                exp_sw2 = predict_exp(tau_sw2, exp_sw1[0, 0], exp_sw1[0, 1],
                                      exp_sw1[0, 2], alpha_c, alpha, beta,
                                      gamma, scale_cc=scale_cc,
                                      chrom_open=False, backward=True)
                exp3 = predict_exp(tau3, exp_sw2[0, 0], exp_sw2[0, 1],
                                   exp_sw2[0, 2], alpha_c, alpha, beta, gamma,
                                   scale_cc=scale_cc, backward=True)
    elif model == 2:
        exp1 = predict_exp(tau1, 1e-3, 1e-3, 1e-3, alpha_c, 0, beta, gamma,
                           scale_cc=scale_cc, chrom_open=False, backward=True)
        if switch >= 1:
            exp_sw1 = predict_exp(tau_sw1, 1e-3, 1e-3, 1e-3, alpha_c, alpha,
                                  beta, gamma, scale_cc=scale_cc,
                                  chrom_open=False, backward=True)
            exp2 = predict_exp(tau2, exp_sw1[0, 0], exp_sw1[0, 1],
                               exp_sw1[0, 2], alpha_c, 0, beta, gamma,
                               scale_cc=scale_cc, backward=True)
            if switch >= 2:
                exp_sw2 = predict_exp(tau_sw2, exp_sw1[0, 0], exp_sw1[0, 1],
                                      exp_sw1[0, 2], alpha_c, 0, beta, gamma,
                                      scale_cc=scale_cc, backward=True)
                exp3 = predict_exp(tau3, exp_sw2[0, 0], exp_sw2[0, 1],
                                   exp_sw2[0, 2], alpha_c, alpha, beta, gamma,
                                   scale_cc=scale_cc, backward=True)
    return (exp1, exp2, exp3), (exp_sw1, exp_sw2)


@njit(locals={
            "res": numba.types.float64[:, ::1],
    },
    fastmath=True)
def ss_exp(alpha_c, alpha, beta, gamma, pred_r=True, chrom_open=True):
    res = np.empty((1, 3))
    if not chrom_open:
        res[0, 0] = 0
        res[0, 1] = 0
        res[0, 2] = 0
    else:
        res[0, 0] = 1
        if pred_r:
            res[0, 1] = alpha / beta
            res[0, 2] = alpha / gamma
        else:
            res[0, 1] = 0
            res[0, 2] = 0
    return res


@njit(locals={
            "ss1": numba.types.float64[:, ::1],
            "ss2": numba.types.float64[:, ::1],
            "ss3": numba.types.float64[:, ::1],
            "ss4": numba.types.float64[:, ::1]
    },
    fastmath=True)
def compute_ss_exp(alpha_c, alpha, beta, gamma, model=0):
    if model == 0:
        ss1 = ss_exp(alpha_c, alpha, beta, gamma, pred_r=False)
        ss2 = ss_exp(alpha_c, alpha, beta, gamma, pred_r=False,
                     chrom_open=False)
        ss3 = ss_exp(alpha_c, alpha, beta, gamma, chrom_open=False)
        ss4 = ss_exp(alpha_c, 0, beta, gamma, chrom_open=False)
    elif model == 1:
        ss1 = ss_exp(alpha_c, alpha, beta, gamma, pred_r=False)
        ss2 = ss_exp(alpha_c, alpha, beta, gamma)
        ss3 = ss_exp(alpha_c, alpha, beta, gamma, chrom_open=False)
        ss4 = ss_exp(alpha_c, 0, beta, gamma, chrom_open=False)
    elif model == 2:
        ss1 = ss_exp(alpha_c, alpha, beta, gamma, pred_r=False)
        ss2 = ss_exp(alpha_c, alpha, beta, gamma)
        ss3 = ss_exp(alpha_c, 0, beta, gamma)
        ss4 = ss_exp(alpha_c, 0, beta, gamma, chrom_open=False)
    return np.vstack((ss1, ss2, ss3, ss4))


@njit(fastmath=True)
def velocity_equations(c, u, s, alpha_c, alpha, beta, gamma, scale_cc=1,
                       pred_r=True, chrom_open=True, rna_only=False):
    if rna_only:
        c = np.full(len(u), 1.0)
    if not chrom_open:
        alpha_c *= scale_cc
        if pred_r:
            return -alpha_c * c, alpha * c - beta * u, beta * u - gamma * s
        else:
            return -alpha_c * c, np.zeros(len(u)), np.zeros(len(u))
    else:
        if pred_r:
            return (alpha_c - alpha_c * c), (alpha * c - beta * u), (beta * u
                                                                     - gamma
                                                                     * s)
        else:
            return alpha_c - alpha_c * c, np.zeros(len(u)), np.zeros(len(u))


@njit(locals={
            "state0": numba.types.boolean[::1],
            "state1": numba.types.boolean[::1],
            "state2": numba.types.boolean[::1],
            "state3": numba.types.boolean[::1],
            "tau1": numba.types.float64[::1],
            "tau2": numba.types.float64[::1],
            "tau3": numba.types.float64[::1],
            "tau4": numba.types.float64[::1],
            "exp_list": numba.types.Tuple((numba.types.float64[:, ::1],
                                           numba.types.float64[:, ::1],
                                           numba.types.float64[:, ::1],
                                           numba.types.float64[:, ::1])),
            "exp_sw_list": numba.types.Tuple((numba.types.float64[:, ::1],
                                              numba.types.float64[:, ::1],
                                              numba.types.float64[:, ::1])),
            "c": numba.types.float64[::1],
            "u": numba.types.float64[::1],
            "s": numba.types.float64[::1],
            "vc_vec": numba.types.float64[::1],
            "vu_vec": numba.types.float64[::1],
            "vs_vec": numba.types.float64[::1]
    },
    fastmath=True)
def compute_velocity(t,
                     t_sw_array,
                     state,
                     alpha_c,
                     alpha,
                     beta,
                     gamma,
                     rescale_c,
                     rescale_u,
                     scale_cc=1,
                     model=1,
                     total_h=20,
                     rna_only=False):

    if state is None:
        state0 = t <= t_sw_array[0]
        state1 = (t_sw_array[0] < t) & (t <= t_sw_array[1])
        state2 = (t_sw_array[1] < t) & (t <= t_sw_array[2])
        state3 = t_sw_array[2] < t
    else:
        state0 = np.equal(state, 0)
        state1 = np.equal(state, 1)
        state2 = np.equal(state, 2)
        state3 = np.equal(state, 3)

    tau1 = t[state0]
    tau2 = t[state1] - t_sw_array[0]
    tau3 = t[state2] - t_sw_array[1]
    tau4 = t[state3] - t_sw_array[2]
    tau_list = [tau1, tau2, tau3, tau4]
    switch = np.sum(t_sw_array < total_h)
    typed_tau_list = List()
    [typed_tau_list.append(x) for x in tau_list]
    exp_list, exp_sw_list = generate_exp(typed_tau_list,
                                         t_sw_array[:switch],
                                         alpha_c,
                                         alpha,
                                         beta,
                                         gamma,
                                         model=model,
                                         scale_cc=scale_cc,
                                         rna_only=rna_only)

    c = np.empty(len(t))
    u = np.empty(len(t))
    s = np.empty(len(t))
    for i, ii in enumerate([state0, state1, state2, state3]):
        if np.any(ii):
            c[ii] = exp_list[i][:, 0]
            u[ii] = exp_list[i][:, 1]
            s[ii] = exp_list[i][:, 2]

    vc_vec = np.zeros(len(u))
    vu_vec = np.zeros(len(u))
    vs_vec = np.zeros(len(u))

    if model == 0:
        if np.any(state0):
            vc_vec[state0], vu_vec[state0], vs_vec[state0] = \
                velocity_equations(c[state0], u[state0], s[state0], alpha_c,
                                   alpha, beta, gamma, pred_r=False,
                                   scale_cc=scale_cc, rna_only=rna_only)
        if np.any(state1):
            vc_vec[state1], vu_vec[state1], vs_vec[state1] = \
                velocity_equations(c[state1], u[state1], s[state1], alpha_c,
                                   alpha, beta, gamma, pred_r=False,
                                   chrom_open=False, scale_cc=scale_cc,
                                   rna_only=rna_only)
        if np.any(state2):
            vc_vec[state2], vu_vec[state2], vs_vec[state2] = \
                velocity_equations(c[state2], u[state2], s[state2], alpha_c,
                                   alpha, beta, gamma, chrom_open=False,
                                   scale_cc=scale_cc, rna_only=rna_only)
        if np.any(state3):
            vc_vec[state3], vu_vec[state3], vs_vec[state3] = \
                velocity_equations(c[state3], u[state3], s[state3], alpha_c, 0,
                                   beta, gamma, chrom_open=False,
                                   scale_cc=scale_cc, rna_only=rna_only)
    elif model == 1:
        if np.any(state0):
            vc_vec[state0], vu_vec[state0], vs_vec[state0] = \
                velocity_equations(c[state0], u[state0], s[state0], alpha_c,
                                   alpha, beta, gamma, pred_r=False,
                                   scale_cc=scale_cc, rna_only=rna_only)
        if np.any(state1):
            vc_vec[state1], vu_vec[state1], vs_vec[state1] = \
                velocity_equations(c[state1], u[state1], s[state1], alpha_c,
                                   alpha, beta, gamma, scale_cc=scale_cc,
                                   rna_only=rna_only)
        if np.any(state2):
            vc_vec[state2], vu_vec[state2], vs_vec[state2] = \
                velocity_equations(c[state2], u[state2], s[state2], alpha_c,
                                   alpha, beta, gamma, chrom_open=False,
                                   scale_cc=scale_cc, rna_only=rna_only)
        if np.any(state3):
            vc_vec[state3], vu_vec[state3], vs_vec[state3] = \
                velocity_equations(c[state3], u[state3], s[state3], alpha_c, 0,
                                   beta, gamma, chrom_open=False,
                                   scale_cc=scale_cc, rna_only=rna_only)
    elif model == 2:
        if np.any(state0):
            vc_vec[state0], vu_vec[state0], vs_vec[state0] = \
                velocity_equations(c[state0], u[state0], s[state0], alpha_c,
                                   alpha, beta, gamma, pred_r=False,
                                   scale_cc=scale_cc, rna_only=rna_only)
        if np.any(state1):
            vc_vec[state1], vu_vec[state1], vs_vec[state1] = \
                velocity_equations(c[state1], u[state1], s[state1], alpha_c,
                                   alpha, beta, gamma, scale_cc=scale_cc,
                                   rna_only=rna_only)
        if np.any(state2):
            vc_vec[state2], vu_vec[state2], vs_vec[state2] = \
                velocity_equations(c[state2], u[state2], s[state2], alpha_c,
                                   0, beta, gamma, scale_cc=scale_cc,
                                   rna_only=rna_only)
        if np.any(state3):
            vc_vec[state3], vu_vec[state3], vs_vec[state3] = \
                velocity_equations(c[state3], u[state3], s[state3], alpha_c, 0,
                                   beta, gamma, chrom_open=False,
                                   scale_cc=scale_cc, rna_only=rna_only)
    return vc_vec * rescale_c, vu_vec * rescale_u, vs_vec


def log_valid(x):
    return np.log(np.clip(x, 1e-3, 1 - 1e-3))


def approx_tau(u, s, u0, s0, alpha, beta, gamma):
    if gamma == beta:
        gamma -= 1e-3
    u_inf = alpha / beta
    if beta > gamma:
        b_new = beta / (gamma - beta)
        s_inf = alpha / gamma
        s_inf_new = s_inf - b_new * u_inf
        s_new = s - b_new * u
        s0_new = s0 - b_new * u0
        tau = -1.0 / gamma * log_valid((s_new - s_inf_new) /
                                       (s0_new - s_inf_new))
    else:
        tau = -1.0 / beta * log_valid((u - u_inf) / (u0 - u_inf))
    return tau


def anchor_points(t_sw_array, total_h=20, t=1000, mode='uniform',
                  return_time=False):
    t_ = np.linspace(0, total_h, t)
    tau1 = t_[t_ <= t_sw_array[0]]
    tau2 = t_[(t_sw_array[0] < t_) & (t_ <= t_sw_array[1])] - t_sw_array[0]
    tau3 = t_[(t_sw_array[1] < t_) & (t_ <= t_sw_array[2])] - t_sw_array[1]
    tau4 = t_[t_sw_array[2] < t_] - t_sw_array[2]

    if mode == 'log':
        if len(tau1) > 0:
            tau1 = np.expm1(tau1)
            tau1 = tau1 / np.max(tau1) * (t_sw_array[0])
        if len(tau2) > 0:
            tau2 = np.expm1(tau2)
            tau2 = tau2 / np.max(tau2) * (t_sw_array[1] - t_sw_array[0])
        if len(tau3) > 0:
            tau3 = np.expm1(tau3)
            tau3 = tau3 / np.max(tau3) * (t_sw_array[2] - t_sw_array[1])
        if len(tau4) > 0:
            tau4 = np.expm1(tau4)
            tau4 = tau4 / np.max(tau4) * (total_h - t_sw_array[2])

    tau_list = [tau1, tau2, tau3, tau4]
    if return_time:
        return t_, tau_list
    else:
        return tau_list


# @jit(nopython=True, fastmath=True, debug=True)
def pairwise_distance_square(X, Y):
    res = np.empty((X.shape[0], Y.shape[0]), dtype=X.dtype)
    for a in range(X.shape[0]):
        for b in range(Y.shape[0]):
            val = 0.0
            for i in range(X.shape[1]):
                tmp = X[a, i] - Y[b, i]
                val += tmp**2
            res[a, b] = val
    return res


def calculate_dist_and_time(c, u, s,
                            t_sw_array,
                            alpha_c, alpha, beta, gamma,
                            rescale_c, rescale_u,
                            scale_cc=1,
                            scale_factor=None,
                            model=1,
                            conn=None,
                            t=1000, k=1,
                            direction='complete',
                            total_h=20,
                            rna_only=False,
                            penalize_gap=True,
                            all_cells=True):

    n = len(u)
    if scale_factor is None:
        scale_factor = np.array([np.std(c), np.std(u), np.std(s)])
    tau_list = anchor_points(t_sw_array, total_h, t)
    switch = np.sum(t_sw_array < total_h)
    typed_tau_list = List()
    [typed_tau_list.append(x) for x in tau_list]
    alpha_c, alpha, beta, gamma = check_params(alpha_c, alpha, beta, gamma)
    exp_list, exp_sw_list = generate_exp(typed_tau_list,
                                         t_sw_array[:switch],
                                         alpha_c,
                                         alpha,
                                         beta,
                                         gamma,
                                         model=model,
                                         scale_cc=scale_cc,
                                         rna_only=rna_only)
    rescale_factor = np.array([rescale_c, rescale_u, 1.0])
    exp_list = [x*rescale_factor for x in exp_list]
    exp_sw_list = [x*rescale_factor for x in exp_sw_list]
    max_c = 0
    max_u = 0
    max_s = 0
    if rna_only:
        exp_mat = (np.hstack((np.reshape(u, (-1, 1)), np.reshape(s, (-1, 1))))
                   / scale_factor[1:])
    else:
        exp_mat = np.hstack((np.reshape(c, (-1, 1)), np.reshape(u, (-1, 1)),
                             np.reshape(s, (-1, 1)))) / scale_factor

    dists = np.full((n, 4), np.inf)
    taus = np.zeros((n, 4), dtype=u.dtype)
    ts = np.zeros((n, 4), dtype=u.dtype)
    anchor_exp, anchor_t = None, None

    for i in range(switch+1):
        if not all_cells:
            max_ci = (np.max(exp_list[i][:, 0]) if exp_list[i].shape[0] > 0
                      else 0)
            max_c = max_ci if max_ci > max_c else max_c
        max_ui = np.max(exp_list[i][:, 1]) if exp_list[i].shape[0] > 0 else 0
        max_u = max_ui if max_ui > max_u else max_u
        max_si = np.max(exp_list[i][:, 2]) if exp_list[i].shape[0] > 0 else 0
        max_s = max_si if max_si > max_s else max_s

        skip_phase = False
        if direction == 'off':
            if (model in [1, 2]) and (i < 2):
                skip_phase = True
        elif direction == 'on':
            if (model in [1, 2]) and (i >= 2):
                skip_phase = True
        if rna_only and i == 0:
            skip_phase = True

        if not skip_phase:
            if rna_only:
                tmp = exp_list[i][:, 1:] / scale_factor[1:]
            else:
                tmp = exp_list[i] / scale_factor
            if anchor_exp is None:
                anchor_exp = exp_list[i]
                anchor_t = (tau_list[i] + t_sw_array[i-1] if i >= 1
                            else tau_list[i])
            else:
                anchor_exp = np.vstack((anchor_exp, exp_list[i]))
                anchor_t = np.hstack((anchor_t, tau_list[i] + t_sw_array[i-1]
                                      if i >= 1 else tau_list[i]))

            if not all_cells:
                anchor_dist = np.diff(tmp, axis=0, prepend=np.zeros((1, 2))
                                      if rna_only else np.zeros((1, 3)))
                anchor_dist = np.sqrt((anchor_dist**2).sum(axis=1))
                remove_cand = anchor_dist < (0.01*np.max(exp_mat[1])
                                             if rna_only
                                             else 0.01*np.max(exp_mat[2]))
                step_idx = np.arange(0, len(anchor_dist), 1) % 3 > 0
                remove_cand &= step_idx
                keep_idx = np.where(~remove_cand)[0]
                tmp = tmp[keep_idx, :]

            tree = KDTree(tmp)
            dd, ii = tree.query(exp_mat, k=k)
            dd = dd**2
            if k > 1:
                dd = np.mean(dd, axis=1)
            if conn is not None:
                dd = conn.dot(dd)
            dists[:, i] = dd

            if not all_cells:
                ii = keep_idx[ii]
            if k == 1:
                taus[:, i] = tau_list[i][ii]
            else:
                for j in range(n):
                    taus[j, i] = tau_list[i][ii[j, :]]
            ts[:, i] = taus[:, i] + t_sw_array[i-1] if i >= 1 else taus[:, i]

    min_dist = np.min(dists, axis=1)
    state_pred = np.argmin(dists, axis=1)
    t_pred = ts[np.arange(n), state_pred]

    anchor_t1_list = []
    anchor_t2_list = []
    t_sw_adjust = np.zeros(3, dtype=u.dtype)

    if direction == 'complete':
        t_sorted = np.sort(t_pred)
        dt = np.diff(t_sorted, prepend=0)
        gap_thresh = 3*np.percentile(dt, 99)
        idx = np.where(dt > gap_thresh)[0]
        for i in idx:
            t1 = t_sorted[i-1] if i > 0 else 0
            t2 = t_sorted[i]
            anchor_t1 = anchor_exp[np.argmin(np.abs(anchor_t - t1)), :]
            anchor_t2 = anchor_exp[np.argmin(np.abs(anchor_t - t2)), :]
            if all_cells:
                anchor_t1_list.append(np.ravel(anchor_t1))
                anchor_t2_list.append(np.ravel(anchor_t2))
            if not all_cells:
                for j in range(1, switch):
                    crit1 = ((t1 > t_sw_array[j-1]) and (t2 > t_sw_array[j-1])
                             and (t1 <= t_sw_array[j])
                             and (t2 <= t_sw_array[j]))
                    crit2 = ((np.abs(anchor_t1[2] - exp_sw_list[j][0, 2])
                             < 0.02 * max_s) and
                             (np.abs(anchor_t2[2] - exp_sw_list[j][0, 2])
                             < 0.01 * max_s))
                    crit3 = ((np.abs(anchor_t1[1] - exp_sw_list[j][0, 1])
                             < 0.02 * max_u) and
                             (np.abs(anchor_t2[1] - exp_sw_list[j][0, 1])
                             < 0.01 * max_u))
                    crit4 = ((np.abs(anchor_t1[0] - exp_sw_list[j][0, 0])
                             < 0.02 * max_c) and
                             (np.abs(anchor_t2[0] - exp_sw_list[j][0, 0])
                             < 0.01 * max_c))
                    if crit1 and crit2 and crit3 and crit4:
                        t_sw_adjust[j] += t2 - t1
            if penalize_gap:
                dist_gap = np.sum(((anchor_t1[1:] - anchor_t2[1:]) /
                                   scale_factor[1:])**2)
                idx_to_adjust = t_pred >= t2
                t_sw_array_ = np.append(t_sw_array, total_h)
                state_to_adjust = np.where(t_sw_array_ > t2)[0]
                dists[np.ix_(idx_to_adjust, state_to_adjust)] += dist_gap
        min_dist = np.min(dists, axis=1)
        state_pred = np.argmin(dists, axis=1)
        if all_cells:
            t_pred = ts[np.arange(n), state_pred]

    if all_cells:
        exp_ss_mat = compute_ss_exp(alpha_c, alpha, beta, gamma, model=model)
        if rna_only:
            exp_ss_mat[:, 0] = 1
        dists_ss = pairwise_distance_square(exp_mat, exp_ss_mat *
                                            rescale_factor / scale_factor)

        reach_ss = np.full((n, 4), False)
        for i in range(n):
            for j in range(4):
                if min_dist[i] > dists_ss[i, j]:
                    reach_ss[i, j] = True
        late_phase = np.full(n, -1)
        for i in range(3):
            late_phase[np.abs(t_pred - t_sw_array[i]) < 0.1] = i
        return min_dist, t_pred, state_pred, reach_ss, late_phase, max_u, \
            max_s, anchor_t1_list, anchor_t2_list
    else:
        return min_dist, state_pred, max_u, max_s, t_sw_adjust


def t_of_c(alpha_c, k_c, c_o, c, rescale_factor, sw_t):

    coef = -float(1)/alpha_c

    c_val = np.clip(c / rescale_factor, a_min=0, a_max=1)

    in_log = (float(k_c) - c_val) / float((k_c) - (c_o))

    epsilon = 1e-9

    return_val = coef * np.log(in_log + epsilon)

    if k_c == 0:
        return_val += sw_t

    return return_val


def make_X(c, u, s,
           max_u,
           max_s,
           alpha_c, alpha, beta, gamma,
           gene_sw_t,
           c0, c_sw1, c_sw2, c_sw3,
           u0, u_sw1, u_sw2, u_sw3,
           s0, s_sw1, s_sw2, s_sw3,
           model, direction, state):

    if direction == "complete":
        dire = 0
    elif direction == "on":
        dire = 1
    elif direction == "off":
        dire = 2

    n = c.shape[0]

    epsilon = 1e-5

    if dire == 0:
        x = np.concatenate((np.array([c,
                                      np.log(u + epsilon),
                                      np.log(s + epsilon)]),
                            np.full((n, 17), [np.log(alpha_c + epsilon),
                                              np.log(alpha + epsilon),
                                              np.log(beta + epsilon),
                                              np.log(gamma + epsilon),
                                              c_sw1, c_sw2, c_sw3,
                                              np.log(u_sw2 + epsilon),
                                              np.log(u_sw3 + epsilon),
                                              np.log(s_sw2 + epsilon),
                                              np.log(s_sw3 + epsilon),
                                              np.log(max_u),
                                              np.log(max_s),
                                              gene_sw_t[0],
                                              gene_sw_t[1],
                                              gene_sw_t[2],
                                              model]).T,
                            np.full((n, 1), state).T
                            )).T.astype(np.float32)

    elif dire == 1:
        x = np.concatenate((np.array([c,
                                      np.log(u + epsilon),
                                      np.log(s + epsilon)]),
                            np.full((n, 12), [np.log(alpha_c + epsilon),
                                              np.log(alpha + epsilon),
                                              np.log(beta + epsilon),
                                              np.log(gamma + epsilon),
                                              c_sw1, c_sw2,
                                              np.log(u_sw1 + epsilon),
                                              np.log(u_sw2 + epsilon),
                                              np.log(s_sw1 + epsilon),
                                              np.log(s_sw2 + epsilon),
                                              gene_sw_t[0],
                                              model]).T,
                            np.full((n, 1), state).T
                            )).T.astype(np.float32)

    elif dire == 2:
        if model == 1:

            max_u_t = -(float(1)/alpha_c)*np.log((max_u*beta)
                                                 / (alpha*c0[2]))

            x = np.concatenate((np.array([np.log(c + epsilon),
                                          np.log(u + epsilon),
                                          np.log(s + epsilon)]),
                                np.full((n, 14), [np.log(alpha_c + epsilon),
                                                  np.log(alpha + epsilon),
                                                  np.log(beta + epsilon),
                                                  np.log(gamma + epsilon),
                                                  c_sw2, c_sw3,
                                                  np.log(u_sw2 + epsilon),
                                                  np.log(u_sw3 + epsilon),
                                                  np.log(s_sw2 + epsilon),
                                                  np.log(s_sw3 + epsilon),
                                                  max_u_t,
                                                  np.log(max_u),
                                                  np.log(max_s),
                                                  gene_sw_t[2]]).T,
                                np.full((n, 1), state).T
                                )).T.astype(np.float32)
        elif model == 2:
            x = np.concatenate((np.array([c,
                                          np.log(u + epsilon),
                                          np.log(s + epsilon)]),
                                np.full((n, 12), [np.log(alpha_c + epsilon),
                                                  np.log(alpha + epsilon),
                                                  np.log(beta + epsilon),
                                                  np.log(gamma + epsilon),
                                                  c_sw2, c_sw3,
                                                  np.log(u_sw2 + epsilon),
                                                  np.log(u_sw3 + epsilon),
                                                  np.log(s_sw2 + epsilon),
                                                  np.log(s_sw3 + epsilon),
                                                  np.log(max_u),
                                                  gene_sw_t[2]]).T,
                                np.full((n, 1), state).T
                                )).T.astype(np.float32)

    return x


def calculate_dist_and_time_nn(c, u, s,
                               max_u, max_s,
                               t_sw_array,
                               alpha_c, alpha, beta, gamma,
                               rescale_c, rescale_u,
                               ode_model_0, ode_model_1,
                               ode_model_2_m1, ode_model_2_m2,
                               device,
                               scale_cc=1,
                               scale_factor=None,
                               model=1,
                               conn=None,
                               t=1000, k=1,
                               direction='complete',
                               total_h=20,
                               rna_only=False,
                               penalize_gap=True,
                               all_cells=True):

    rescale_factor = np.array([rescale_c, rescale_u, 1.0])

    exp_list_net, exp_sw_list_net = generate_exp(None,
                                                 t_sw_array,
                                                 alpha_c,
                                                 alpha,
                                                 beta,
                                                 gamma,
                                                 model=model,
                                                 scale_cc=scale_cc,
                                                 rna_only=rna_only)

    N = len(c)
    N_list = np.arange(N)

    if scale_factor is None:
        cur_scale_factor = np.array([np.std(c),
                                     np.std(u),
                                     np.std(s)])
    else:
        cur_scale_factor = scale_factor

    t_pred_per_state = []
    dists_per_state = []

    dire = 0

    if direction == "on":
        states = [0, 1]
        dire = 1

    elif direction == "off":
        states = [2, 3]
        dire = 2

    else:
        states = [0, 1, 2, 3]
        dire = 0

    dists_per_state = np.zeros((N, len(states)))
    t_pred_per_state = np.zeros((N, len(states)))
    u_pred_per_state = np.zeros((N, len(states)))
    s_pred_per_state = np.zeros((N, len(states)))

    increment = 0

    # determine when we can consider u and s close to zero
    zero_us = np.logical_and((u < 0.1 * max_u), (s < 0.1 * max_s))

    t_pred = np.zeros(N)
    dists = None

    # pass all the data through the neural net as each valid state
    for state in states:

        # when u and s = 0, it's better to use the inverse c equation
        # instead of the neural network, which happens for part of
        # state 3 and all of state 0
        inverse_c = np.logical_or(state == 0,
                                  np.logical_and(state == 3, zero_us))

        not_inverse_c = np.logical_not(inverse_c)

        # if we want to use the inverse c equation...
        if np.any(inverse_c):

            # find out at what switch time chromatin closes
            c_sw_t = t_sw_array[int(model)]

            # figure out whether chromatin is opening/closing and what
            # the initial c value is
            if state <= model:
                k_c = 1
                c_0_for_t_guess = 0
            elif state > model:
                k_c = 0
                c_0_for_t_guess = exp_sw_list_net[int(model)][0, 0]

            # calculate predicted time from the inverse c equation
            t_pred[inverse_c] = t_of_c(alpha_c,
                                       k_c, c_0_for_t_guess,
                                       c[inverse_c],
                                       rescale_factor[0],
                                       c_sw_t)

        # if there are points where we want to use the neural network...
        if np.any(not_inverse_c):

            # create an input matrix from the data
            x = make_X(c[not_inverse_c] / rescale_factor[0],
                       u[not_inverse_c] / rescale_factor[1],
                       s[not_inverse_c] / rescale_factor[2],
                       max_u,
                       max_s,
                       alpha_c*(scale_cc if state > model else 1),
                       alpha, beta, gamma,
                       t_sw_array,
                       0,
                       exp_sw_list_net[0][0, 0],
                       exp_sw_list_net[1][0, 0],
                       exp_sw_list_net[2][0, 0],
                       0,
                       exp_sw_list_net[0][0, 1],
                       exp_sw_list_net[1][0, 1],
                       exp_sw_list_net[2][0, 1],
                       0,
                       exp_sw_list_net[0][0, 2],
                       exp_sw_list_net[1][0, 2],
                       exp_sw_list_net[2][0, 2],
                       model, direction, state)

            # do a forward pass
            if dire == 0:
                t_pred_ten = ode_model_0(torch.tensor(x,
                                                      dtype=torch.float,
                                                      device=device)
                                         .reshape(-1, x.shape[1]))

            elif dire == 1:
                t_pred_ten = ode_model_1(torch.tensor(x,
                                                      dtype=torch.float,
                                                      device=device)
                                         .reshape(-1, x.shape[1]))

            elif dire == 2:
                if model == 1:
                    t_pred_ten = ode_model_2_m1(torch.tensor(x,
                                                             dtype=torch.float,
                                                             device=device)
                                                .reshape(-1, x.shape[1]))
                elif model == 2:
                    t_pred_ten = ode_model_2_m2(torch.tensor(x,
                                                             dtype=torch.float,
                                                             device=device)
                                                .reshape(-1, x.shape[1]))

            # make a numpy array out of our tensor of predicted time points
            t_pred[not_inverse_c] = (t_pred_ten.cpu().detach().numpy()
                                     .flatten()*21) - 1

        # calculate tau values from our predicted time points
        if state == 0:
            t_pred = np.clip(t_pred, a_min=0, a_max=t_sw_array[0])
            tau1 = t_pred
            tau2 = []
            tau3 = []
            tau4 = []
        elif state == 1:
            tau1 = []
            t_pred = np.clip(t_pred, a_min=t_sw_array[0], a_max=t_sw_array[1])
            tau2 = t_pred - t_sw_array[0]
            tau3 = []
            tau4 = []
        elif state == 2:
            tau1 = []
            tau2 = []
            t_pred = np.clip(t_pred, a_min=t_sw_array[1], a_max=t_sw_array[2])
            tau3 = t_pred - t_sw_array[1]
            tau4 = []
        elif state == 3:
            tau1 = []
            tau2 = []
            tau3 = []
            t_pred = np.clip(t_pred, a_min=t_sw_array[2], a_max=20)
            tau4 = t_pred - t_sw_array[2]

        tau_list = [tau1, tau2, tau3, tau4]

        valid_vals = []

        for i in range(len(tau_list)):
            if len(tau_list[i]) == 0:
                tau_list[i] = np.array([0.0])
            else:
                valid_vals.append(i)

        # take the time points and get predicted c/u/s values from them
        exp_list, exp_sw_list_2 = generate_exp(tau_list,
                                               t_sw_array,
                                               alpha_c,
                                               alpha,
                                               beta,
                                               gamma,
                                               model=model,
                                               scale_cc=scale_cc,
                                               rna_only=rna_only)

        pred_c = np.concatenate([exp_list[x][:, 0] * rescale_factor[0]
                                 for x in valid_vals])
        pred_u = np.concatenate([exp_list[x][:, 1] * rescale_factor[1]
                                 for x in valid_vals])
        pred_s = np.concatenate([exp_list[x][:, 2] * rescale_factor[2]
                                 for x in valid_vals])

        # calculate distance between predicted and real values
        c_diff = (c - pred_c) / cur_scale_factor[0]
        u_diff = (u - pred_u) / cur_scale_factor[1]
        s_diff = (s - pred_s) / cur_scale_factor[2]

        dists = (c_diff*c_diff) + (u_diff*u_diff) + (s_diff*s_diff)

        if conn is not None:
            dists = conn.dot(dists)

        # store the distances, times, and predicted u and s values for
        # each state
        dists_per_state[:, increment] = dists
        t_pred_per_state[:, increment] = t_pred
        u_pred_per_state[:, increment] = pred_u
        s_pred_per_state[:, increment] = pred_s

        increment += 1

    # whichever state has the smallest distance for a given data point
    # is our predicted state
    state_pred = np.argmin(dists_per_state, axis=1)

    # slice dists and predicted time over the correct state
    dists = dists_per_state[N_list, state_pred]
    t_pred = t_pred_per_state[N_list, state_pred]

    max_t = t_pred.max()
    min_t = t_pred.min()

    penalty = 0

    # for induction and complete genes, add a penalty to ensure that not
    # all points are in state 0
    if direction == "on" or direction == "complete":

        if t_sw_array[0] >= max_t:
            penalty += (t_sw_array[0] - max_t) + 10

    # for induction genes, add a penalty to ensure that predicted time
    # points are not "out of bounds" by being greater than the
    # second switch time
    if direction == "on":

        if min_t > t_sw_array[1]:
            penalty += (min_t - t_sw_array[1]) + 10

    # for repression genes, add a penalty to ensure that predicted time
    # points are not "out of bounds" by being smaller than the
    # second switch time
    if direction == "off":

        if t_sw_array[1] >= max_t:
            penalty += (t_sw_array[1] - max_t) + 10

    # add penalty to ensure that the time points aren't concentrated to
    # one spot
    if np.abs(max_t - min_t) <= 1e-2:
        penalty += np.abs(max_t - min_t) + 10

    # because the indices chosen by np.argmin are just indices,
    # we need to increment by two to get the true state number for
    # our "off" genes (e.g. so that they're in the domain of [2,3] instead
    # of [0,1])
    if direction == "off":
        state_pred += 2

    if all_cells:
        return dists, t_pred, state_pred, max_u, max_s, penalty
    else:
        return dists, state_pred, max_u, max_s, penalty


# @jit(nopython=True, fastmath=True)
def compute_likelihood(c, u, s,
                       t_sw_array,
                       alpha_c, alpha, beta, gamma,
                       rescale_c, rescale_u,
                       t_pred,
                       state_pred,
                       scale_cc=1,
                       scale_factor=None,
                       model=1,
                       weight=None,
                       total_h=20,
                       rna_only=False):

    if weight is None:
        weight = np.full(c.shape, True)
    c_ = c[weight]
    u_ = u[weight]
    s_ = s[weight]
    t_pred_ = t_pred[weight]
    state_pred_ = state_pred[weight]

    n = len(u_)
    if scale_factor is None:
        scale_factor = np.ones(3)
    tau1 = t_pred_[state_pred_ == 0]
    tau2 = t_pred_[state_pred_ == 1] - t_sw_array[0]
    tau3 = t_pred_[state_pred_ == 2] - t_sw_array[1]
    tau4 = t_pred_[state_pred_ == 3] - t_sw_array[2]
    tau_list = [tau1, tau2, tau3, tau4]
    switch = np.sum(t_sw_array < total_h)
    typed_tau_list = List()
    [typed_tau_list.append(x) for x in tau_list]
    alpha_c, alpha, beta, gamma = check_params(alpha_c, alpha, beta, gamma)
    exp_list, _ = generate_exp(typed_tau_list,
                               t_sw_array[:switch],
                               alpha_c,
                               alpha,
                               beta,
                               gamma,
                               model=model,
                               scale_cc=scale_cc,
                               rna_only=rna_only)
    rescale_factor = np.array([rescale_c, rescale_u, 1.0])
    exp_list = [x*rescale_factor*scale_factor for x in exp_list]
    exp_mat = np.hstack((np.reshape(c_, (-1, 1)), np.reshape(u_, (-1, 1)),
                         np.reshape(s_, (-1, 1)))) * scale_factor
    diffs = np.empty((n, 3), dtype=u.dtype)
    likelihood_c = 0
    likelihood_u = 0
    likelihood_s = 0
    ssd_c, var_c = 0, 0
    for i in range(switch+1):
        index = state_pred_ == i
        if np.sum(index) > 0:
            diff = exp_mat[index, :] - exp_list[i]
            diffs[index, :] = diff
    if rna_only:
        diff_u = np.ravel(diffs[:, 0])
        diff_s = np.ravel(diffs[:, 1])
        dist_us = diff_u ** 2 + diff_s ** 2
        var_us = np.var(np.sign(diff_s) * np.sqrt(dist_us))
        nll = (0.5 * np.log(2 * np.pi * var_us) + 0.5 / n /
               var_us * np.sum(dist_us))
    else:
        diff_c = np.ravel(diffs[:, 0])
        diff_u = np.ravel(diffs[:, 1])
        diff_s = np.ravel(diffs[:, 2])
        dist_c = diff_c ** 2
        dist_u = diff_u ** 2
        dist_s = diff_s ** 2
        var_c = np.var(diff_c)
        var_u = np.var(diff_u)
        var_s = np.var(diff_s)
        ssd_c = np.sum(dist_c)
        nll_c = (0.5 * np.log(2 * np.pi * var_c) + 0.5 / n /
                 var_c * np.sum(dist_c))
        nll_u = (0.5 * np.log(2 * np.pi * var_u) + 0.5 / n /
                 var_u * np.sum(dist_u))
        nll_s = (0.5 * np.log(2 * np.pi * var_s) + 0.5 / n /
                 var_s * np.sum(dist_s))
        nll = nll_c + nll_u + nll_s
        likelihood_c = np.exp(-nll_c)
        likelihood_u = np.exp(-nll_u)
        likelihood_s = np.exp(-nll_s)
    likelihood = np.exp(-nll)
    return likelihood, likelihood_c, ssd_c, var_c, likelihood_u, likelihood_s


class ChromatinDynamical:
    def __init__(self, c, u, s,
                 gene=None,
                 model=None,
                 max_iter=10,
                 init_mode="grid",
                 device="cpu",
                 neural_net=False,
                 adam=False,
                 adam_lr=None,
                 adam_beta1=None,
                 adam_beta2=None,
                 batch_size=None,
                 local_std=None,
                 embed_coord=None,
                 connectivities=None,
                 plot=False,
                 save_plot=False,
                 plot_dir=None,
                 fit_args=None,
                 partial=None,
                 direction=None,
                 rna_only=False,
                 fit_decoupling=True,
                 extra_color=None,
                 rescale_u=None,
                 alpha=None,
                 beta=None,
                 gamma=None,
                 t_=None
                 ):

        self.device = device
        self.gene = gene
        self.local_std = local_std
        self.conn = connectivities

        self.neural_net = neural_net
        self.adam = adam
        self.adam_lr = adam_lr
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.batch_size = batch_size

        self.torch_type = type(u[0].item())

        # fitting arguments
        self.init_mode = init_mode
        self.rna_only = rna_only
        self.fit_decoupling = fit_decoupling
        self.max_iter = max_iter
        self.n_anchors = np.clip(int(fit_args['t']), 201, 2000)
        self.k_dist = np.clip(int(fit_args['k']), 1, 20)
        self.tm = np.clip(fit_args['thresh_multiplier'], 0.4, 2)
        self.weight_c = np.clip(fit_args['weight_c'], 0.1, 5)
        self.outlier = np.clip(fit_args['outlier'], 80, 100)
        self.model = int(model) if isinstance(model, float) else model
        self.model_ = None
        if self.model == 0 and self.init_mode == 'invert':
            self.init_mode = 'grid'

        # plot parameters
        self.plot = plot
        self.save_plot = save_plot
        self.extra_color = extra_color
        self.fig_size = fit_args['fig_size']
        self.point_size = fit_args['point_size']
        if plot_dir is None:
            self.plot_path = 'rna_plots' if self.rna_only else 'plots'
        else:
            self.plot_path = plot_dir
        self.color = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']
        self.fig = None
        self.ax = None

        # input
        self.total_n = len(u)
        
        if sp.__version__ < '1.14.0':
            if sparse.issparse(c):
                c = c.A
            if sparse.issparse(u):
                u = u.A
            if sparse.issparse(s):
                s = s.A
        else:
            if sparse.issparse(c):
                c = c.toarray()
            if sparse.issparse(u):
                u = u.toarray()
            if sparse.issparse(s):
                s = s.toarray()
        
        self.c_all = np.ravel(np.array(c, dtype=np.float64))
        self.u_all = np.ravel(np.array(u, dtype=np.float64))
        self.s_all = np.ravel(np.array(s, dtype=np.float64))

        # adjust offset
        self.offset_c, self.offset_u, self.offset_s = np.min(self.c_all), \
            np.min(self.u_all), np.min(self.s_all)
        self.offset_c = 0 if self.rna_only else self.offset_c
        self.c_all -= self.offset_c
        self.u_all -= self.offset_u
        self.s_all -= self.offset_s
        # remove zero counts
        self.non_zero = (np.ravel(self.c_all > 0) | np.ravel(self.u_all > 0) |
                         np.ravel(self.s_all > 0))
        # remove outliers
        self.non_outlier = np.ravel(self.c_all <= np.percentile(self.c_all,
                                                                self.outlier))
        self.non_outlier &= np.ravel(self.u_all <= np.percentile(self.u_all,
                                                                 self.outlier))
        self.non_outlier &= np.ravel(self.s_all <= np.percentile(self.s_all,
                                                                 self.outlier))
        self.c = self.c_all[self.non_zero & self.non_outlier]
        self.u = self.u_all[self.non_zero & self.non_outlier]
        self.s = self.s_all[self.non_zero & self.non_outlier]
        self.low_quality = len(self.u) < 10
        # scale modalities
        self.std_c, self.std_u, self.std_s = (np.std(self.c_all)
                                              if not self.rna_only
                                              else 1.0, np.std(self.u_all),
                                              np.std(self.s_all))
        if self.std_u == 0 or self.std_s == 0:
            self.low_quality = True
        self.scale_c, self.scale_u, self.scale_s = np.max(self.c_all) \
            if not self.rna_only else 1.0, self.std_u/self.std_s, 1.0

        # if we're on neural net mode, check to see if c is way bigger than
        # u or s, which would be very hard for the neural net to fit
        if not self.low_quality and neural_net:
            max_c_orig = np.max(self.c)
            if max_c_orig / np.max(self.u) > 500:
                self.low_quality = True

            if not self.low_quality:
                if max_c_orig / np.max(self.s) > 500:
                    self.low_quality = True

        self.c_all /= self.scale_c
        self.u_all /= self.scale_u
        self.s_all /= self.scale_s
        self.c /= self.scale_c
        self.u /= self.scale_u
        self.s /= self.scale_s
        self.scale_factor = np.array([np.std(self.c_all) / self.std_s /
                                      self.weight_c, 1.0, 1.0])
        self.scale_factor[0] = 1 if self.rna_only else self.scale_factor[0]
        self.max_u, self.max_s = np.max(self.u), np.max(self.s)
        self.max_u_all, self.max_s_all = np.max(self.u_all), np.max(self.s_all)
        if self.conn is not None:
            self.conn_sub = self.conn[np.ix_(self.non_zero & self.non_outlier,
                                             self.non_zero & self.non_outlier)]
        else:
            self.conn_sub = None

        #main_info(f'{len(self.u)} cells passed filter and will be used to '
        #            'compute trajectories.', indent_level=2)
        self.known_pars = (True
                           if None not in [rescale_u, alpha, beta, gamma, t_]
                           else False)
        if self.known_pars:
            main_info(f'known parameters for gene {self.gene} are '
                        f'scaling={rescale_u}, alpha={alpha}, beta={beta},'
                        f' gamma={gamma}, t_={t_}.', indent_level=1)

        # define neural networks
        self.ode_model_0 = nn.Sequential(
            nn.Linear(21, 150),
            nn.ReLU(),
            nn.Linear(150, 112),
            nn.ReLU(),
            nn.Linear(112, 75),
            nn.ReLU(),
            nn.Linear(75, 1),
            nn.Sigmoid()
        )

        self.ode_model_1 = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.ode_model_2_m1 = nn.Sequential(
            nn.Linear(18, 220),
            nn.ReLU(),
            nn.Linear(220, 165),
            nn.ReLU(),
            nn.Linear(165, 110),
            nn.ReLU(),
            nn.Linear(110, 1),
            nn.Sigmoid()
        )

        self.ode_model_2_m2 = nn.Sequential(
            nn.Linear(16, 150),
            nn.ReLU(),
            nn.Linear(150, 112),
            nn.ReLU(),
            nn.Linear(112, 75),
            nn.ReLU(),
            nn.Linear(75, 1),
            nn.Sigmoid()
        )

        self.ode_model_0.to(torch.device(self.device))
        self.ode_model_1.to(torch.device(self.device))
        self.ode_model_2_m1.to(torch.device(self.device))
        self.ode_model_2_m2.to(torch.device(self.device))

        # load in neural network
        net_path = os.path.dirname(os.path.abspath(__file__)) + \
            "/neural_nets/"

        self.ode_model_0.load_state_dict(torch.load(net_path+"dir0.pt"))
        self.ode_model_1.load_state_dict(torch.load(net_path+"dir1.pt"))
        self.ode_model_2_m1.load_state_dict(torch.load(net_path+"dir2_m1.pt"))
        self.ode_model_2_m2.load_state_dict(torch.load(net_path+"dir2_m2.pt"))

        # 4 rate parameters
        self.alpha_c = 0.1
        self.alpha = alpha if alpha is not None else 0.0
        self.beta = beta if beta is not None else 0.0
        self.gamma = gamma if gamma is not None else 0.0
        # 3 possible switch time points
        self.t_sw_1 = 0.1 if t_ is not None else 0.0
        self.t_sw_2 = t_+0.1 if t_ is not None else 0.0
        self.t_sw_3 = 20.0 if t_ is not None else 0.0
        # 2 rescale factors
        self.rescale_c = 1.0
        self.rescale_u = rescale_u if rescale_u is not None else 1.0
        self.rates = None
        self.t_sw_array = None
        self.fit_rescale = True if rescale_u is None else False
        self.params = None

        # other parameters or results
        self.t = None
        self.state = None
        self.loss = [np.inf]
        self.likelihood = -1.0
        self.l_c = 0
        self.ssd_c, self.var_c = 0, 0
        self.scale_cc = 1.0
        self.fitting_flag_ = 0
        self.velocity = None
        self.anchor_t1_list, self.anchor_t2_list = None, None
        self.anchor_exp = None
        self.anchor_exp_sw = None
        self.anchor_min_idx, self.anchor_max_idx, self.anchor_velo_min_idx, \
            self.anchor_velo_max_idx = None, None, None, None
        self.anchor_velo = None
        self.c0 = self.u0 = self.s0 = 0.0
        self.realign_ratio = 1.0
        self.partial = False
        self.direction = 'complete'
        self.steady_state_func = None

        # for fit and update
        self.cur_iter = 0
        self.cur_loss = None
        self.cur_state_pred = None
        self.cur_t_sw_adjust = None

        # partial checking and model examination
        determine_model = model is None
        if partial is None and direction is None:
            if embed_coord is not None:
                self.embed_coord = embed_coord[self.non_zero &
                                               self.non_outlier]
            else:
                self.embed_coord = None
            self.check_partial_trajectory(determine_model=determine_model)
        elif direction is not None:
            self.direction = direction
            if direction in ['on', 'off']:
                self.partial = True
            else:
                self.partial = False
            self.check_partial_trajectory(fit_gmm=False, fit_slope=False,
                                          determine_model=determine_model)
        elif partial is not None:
            self.partial = partial
            self.check_partial_trajectory(fit_gmm=False,
                                          determine_model=determine_model)
        else:
            self.check_partial_trajectory(fit_gmm=False, fit_slope=False,
                                          determine_model=determine_model)

        # intialize steady state parameters
        if not self.known_pars and not self.low_quality:
            self.initialize_steady_state_params(model_mismatch=self.model
                                                != self.model_)
        if self.known_pars:
            self.params = np.array([self.t_sw_1,
                                    self.t_sw_2-self.t_sw_1,
                                    self.t_sw_3-self.t_sw_2,
                                    self.alpha_c,
                                    self.alpha,
                                    self.beta,
                                    self.gamma,
                                    self.scale_cc,
                                    self.rescale_c,
                                    self.rescale_u])

    # the torch tensor version of the anchor points function
    def anchor_points_ten(self, t_sw_array, total_h=20, t=1000, mode='uniform',
                          return_time=False):

        t_ = torch.linspace(0, total_h, t, device=self.device,
                            dtype=self.torch_type)
        tau1 = t_[t_ <= t_sw_array[0]]
        tau2 = t_[(t_sw_array[0] < t_) & (t_ <= t_sw_array[1])] - t_sw_array[0]
        tau3 = t_[(t_sw_array[1] < t_) & (t_ <= t_sw_array[2])] - t_sw_array[1]
        tau4 = t_[t_sw_array[2] < t_] - t_sw_array[2]

        if mode == 'log':
            if len(tau1) > 0:
                tau1 = torch.expm1(tau1)
                tau1 = tau1 / torch.max(tau1) * (t_sw_array[0])
            if len(tau2) > 0:
                tau2 = torch.expm1(tau2)
                tau2 = tau2 / torch.max(tau2) * (t_sw_array[1] - t_sw_array[0])
            if len(tau3) > 0:
                tau3 = torch.expm1(tau3)
                tau3 = tau3 / torch.max(tau3) * (t_sw_array[2] - t_sw_array[1])
            if len(tau4) > 0:
                tau4 = torch.expm1(tau4)
                tau4 = tau4 / torch.max(tau4) * (total_h - t_sw_array[2])

        tau_list = [tau1, tau2, tau3, tau4]
        if return_time:
            return t_, tau_list
        else:
            return tau_list

    # the torch version of the predict_exp function
    def predict_exp_ten(self,
                        tau,
                        c0,
                        u0,
                        s0,
                        alpha_c,
                        alpha,
                        beta,
                        gamma,
                        scale_cc=None,
                        pred_r=True,
                        chrom_open=True,
                        backward=False,
                        rna_only=False):

        if scale_cc is None:
            scale_cc = torch.tensor(1.0, requires_grad=True,
                                    device=self.device,
                                    dtype=self.torch_type)

        if len(tau) == 0:
            return torch.empty((0, 3),
                               requires_grad=True,
                               device=self.device,
                               dtype=self.torch_type)
        if backward:
            tau = -tau

        eat = torch.exp(-alpha_c * tau)
        ebt = torch.exp(-beta * tau)
        egt = torch.exp(-gamma * tau)
        if rna_only:
            kc = 1
            c0 = 1
        else:
            if chrom_open:
                kc = 1
            else:
                kc = 0
                alpha_c = alpha_c * scale_cc

        const = (kc - c0) * alpha / (beta - alpha_c)

        res0 = kc - (kc - c0) * eat

        if pred_r:

            res1 = u0 * ebt + (alpha * kc / beta) * (1 - ebt)
            res1 += const * (ebt - eat)

            res2 = s0 * egt + (alpha * kc / gamma) * (1 - egt)
            res2 += ((beta / (gamma - beta)) *
                     ((alpha * kc / beta) - u0 - const) * (egt - ebt))
            res2 += (beta / (gamma - alpha_c)) * const * (egt - eat)

        else:
            res1 = torch.zeros(len(tau), device=self.device,
                               requires_grad=True,
                               dtype=self.torch_type)
            res2 = torch.zeros(len(tau), device=self.device,
                               requires_grad=True,
                               dtype=self.torch_type)

        res = torch.stack((res0, res1, res2), 1)

        return res

    # the torch tensor version of the generate_exp function
    def generate_exp_tens(self,
                          tau_list,
                          t_sw_array,
                          alpha_c,
                          alpha,
                          beta,
                          gamma,
                          scale_cc=None,
                          model=1,
                          rna_only=False):

        if scale_cc is None:
            scale_cc = torch.tensor(1.0, requires_grad=True,
                                    device=self.device,
                                    dtype=self.torch_type)

        if beta == alpha_c:
            beta += 1e-3
        if gamma == beta or gamma == alpha_c:
            gamma += 1e-3
        switch = int(t_sw_array.size(dim=0))
        if switch >= 1:
            tau_sw1 = torch.tensor([t_sw_array[0]], requires_grad=True,
                                   device=self.device,
                                   dtype=self.torch_type)
            if switch >= 2:
                tau_sw2 = torch.tensor([t_sw_array[1] - t_sw_array[0]],
                                       requires_grad=True,
                                       device=self.device,
                                       dtype=self.torch_type)
                if switch == 3:
                    tau_sw3 = torch.tensor([t_sw_array[2] - t_sw_array[1]],
                                           requires_grad=True,
                                           device=self.device,
                                           dtype=self.torch_type)
        exp_sw1, exp_sw2, exp_sw3 = (torch.empty((0, 3),
                                                 requires_grad=True,
                                                 device=self.device,
                                                 dtype=self.torch_type),
                                     torch.empty((0, 3),
                                                 requires_grad=True,
                                                 device=self.device,
                                                 dtype=self.torch_type),
                                     torch.empty((0, 3),
                                                 requires_grad=True,
                                                 device=self.device,
                                                 dtype=self.torch_type))
        if tau_list is None:
            if model == 0:
                if switch >= 1:
                    exp_sw1 = self.predict_exp_ten(tau_sw1, 0, 0, 0, alpha_c,
                                                   alpha, beta, gamma,
                                                   pred_r=False,
                                                   scale_cc=scale_cc,
                                                   rna_only=rna_only)
                    if switch >= 2:
                        exp_sw2 = self.predict_exp_ten(tau_sw2, exp_sw1[0, 0],
                                                       exp_sw1[0, 1],
                                                       exp_sw1[0, 2],
                                                       alpha_c, alpha, beta,
                                                       gamma, pred_r=False,
                                                       chrom_open=False,
                                                       scale_cc=scale_cc,
                                                       rna_only=rna_only)
                        if switch >= 3:
                            exp_sw3 = self.predict_exp_ten(tau_sw3,
                                                           exp_sw2[0, 0],
                                                           exp_sw2[0, 1],
                                                           exp_sw2[0, 2],
                                                           alpha_c, alpha,
                                                           beta, gamma,
                                                           chrom_open=False,
                                                           scale_cc=scale_cc,
                                                           rna_only=rna_only)
            elif model == 1:
                if switch >= 1:
                    exp_sw1 = self.predict_exp_ten(tau_sw1, 0, 0, 0, alpha_c,
                                                   alpha, beta, gamma,
                                                   pred_r=False,
                                                   scale_cc=scale_cc,
                                                   rna_only=rna_only)
                    if switch >= 2:
                        exp_sw2 = self.predict_exp_ten(tau_sw2, exp_sw1[0, 0],
                                                       exp_sw1[0, 1],
                                                       exp_sw1[0, 2],
                                                       alpha_c, alpha,
                                                       beta, gamma,
                                                       scale_cc=scale_cc,
                                                       rna_only=rna_only)
                        if switch >= 3:
                            exp_sw3 = self.predict_exp_ten(tau_sw3,
                                                           exp_sw2[0, 0],
                                                           exp_sw2[0, 1],
                                                           exp_sw2[0, 2],
                                                           alpha_c, alpha,
                                                           beta, gamma,
                                                           chrom_open=False,
                                                           scale_cc=scale_cc,
                                                           rna_only=rna_only)
            elif model == 2:
                if switch >= 1:
                    exp_sw1 = self.predict_exp_ten(tau_sw1, 0, 0, 0, alpha_c,
                                                   alpha, beta, gamma,
                                                   pred_r=False,
                                                   scale_cc=scale_cc,
                                                   rna_only=rna_only)
                    if switch >= 2:
                        exp_sw2 = self.predict_exp_ten(tau_sw2, exp_sw1[0, 0],
                                                       exp_sw1[0, 1],
                                                       exp_sw1[0, 2], alpha_c,
                                                       alpha, beta, gamma,
                                                       scale_cc=scale_cc,
                                                       rna_only=rna_only)
                        if switch >= 3:
                            exp_sw3 = self.predict_exp_ten(tau_sw3,
                                                           exp_sw2[0, 0],
                                                           exp_sw2[0, 1],
                                                           exp_sw2[0, 2],
                                                           alpha_c, 0, beta,
                                                           gamma,
                                                           scale_cc=scale_cc,
                                                           rna_only=rna_only)

            return [torch.empty((0, 3), requires_grad=True,
                                device=self.device,
                                dtype=self.torch_type),
                    torch.empty((0, 3), requires_grad=True,
                                device=self.device,
                                dtype=self.torch_type),
                    torch.empty((0, 3), requires_grad=True,
                                device=self.device,
                                dtype=self.torch_type),
                    torch.empty((0, 3), requires_grad=True,
                                device=self.device,
                                dtype=self.torch_type)], \
                   [exp_sw1, exp_sw2, exp_sw3]

        tau1 = tau_list[0]
        if switch >= 1:
            tau2 = tau_list[1]
            if switch >= 2:
                tau3 = tau_list[2]
                if switch == 3:
                    tau4 = tau_list[3]
        exp1, exp2, exp3, exp4 = (torch.empty((0, 3), requires_grad=True,
                                              device=self.device,
                                              dtype=self.torch_type),
                                  torch.empty((0, 3), requires_grad=True,
                                              device=self.device,
                                              dtype=self.torch_type),
                                  torch.empty((0, 3), requires_grad=True,
                                              device=self.device,
                                              dtype=self.torch_type),
                                  torch.empty((0, 3), requires_grad=True,
                                              device=self.device,
                                              dtype=self.torch_type))
        if model == 0:
            exp1 = self.predict_exp_ten(tau1, 0, 0, 0, alpha_c, alpha, beta,
                                        gamma, pred_r=False, scale_cc=scale_cc,
                                        rna_only=rna_only)
            if switch >= 1:
                exp_sw1 = self.predict_exp_ten(tau_sw1, 0, 0, 0, alpha_c,
                                               alpha, beta, gamma,
                                               pred_r=False, scale_cc=scale_cc,
                                               rna_only=rna_only)
                exp2 = self.predict_exp_ten(tau2, exp_sw1[0, 0], exp_sw1[0, 1],
                                            exp_sw1[0, 2], alpha_c, alpha,
                                            beta, gamma, pred_r=False,
                                            chrom_open=False,
                                            scale_cc=scale_cc,
                                            rna_only=rna_only)
                if switch >= 2:
                    exp_sw2 = self.predict_exp_ten(tau_sw2, exp_sw1[0, 0],
                                                   exp_sw1[0, 1],
                                                   exp_sw1[0, 2],
                                                   alpha_c, alpha, beta, gamma,
                                                   pred_r=False,
                                                   chrom_open=False,
                                                   scale_cc=scale_cc,
                                                   rna_only=rna_only)
                    exp3 = self.predict_exp_ten(tau3, exp_sw2[0, 0],
                                                exp_sw2[0, 1], exp_sw2[0, 2],
                                                alpha_c, alpha, beta, gamma,
                                                chrom_open=False,
                                                scale_cc=scale_cc,
                                                rna_only=rna_only)
                    if switch == 3:
                        exp_sw3 = self.predict_exp_ten(tau_sw3, exp_sw2[0, 0],
                                                       exp_sw2[0, 1],
                                                       exp_sw2[0, 2],
                                                       alpha_c, alpha, beta,
                                                       gamma,
                                                       chrom_open=False,
                                                       scale_cc=scale_cc,
                                                       rna_only=rna_only)
                        exp4 = self.predict_exp_ten(tau4, exp_sw3[0, 0],
                                                    exp_sw3[0, 1],
                                                    exp_sw3[0, 2],
                                                    alpha_c, 0, beta, gamma,
                                                    chrom_open=False,
                                                    scale_cc=scale_cc,
                                                    rna_only=rna_only)
        elif model == 1:
            exp1 = self.predict_exp_ten(tau1, 0, 0, 0, alpha_c, alpha, beta,
                                        gamma, pred_r=False, scale_cc=scale_cc,
                                        rna_only=rna_only)
            if switch >= 1:
                exp_sw1 = self.predict_exp_ten(tau_sw1, 0, 0, 0, alpha_c,
                                               alpha, beta, gamma,
                                               pred_r=False, scale_cc=scale_cc,
                                               rna_only=rna_only)
                exp2 = self.predict_exp_ten(tau2, exp_sw1[0, 0], exp_sw1[0, 1],
                                            exp_sw1[0, 2], alpha_c, alpha,
                                            beta, gamma, scale_cc=scale_cc,
                                            rna_only=rna_only)
                if switch >= 2:
                    exp_sw2 = self.predict_exp_ten(tau_sw2, exp_sw1[0, 0],
                                                   exp_sw1[0, 1],
                                                   exp_sw1[0, 2], alpha_c,
                                                   alpha, beta, gamma,
                                                   scale_cc=scale_cc,
                                                   rna_only=rna_only)
                    exp3 = self.predict_exp_ten(tau3, exp_sw2[0, 0],
                                                exp_sw2[0, 1], exp_sw2[0, 2],
                                                alpha_c, alpha, beta, gamma,
                                                chrom_open=False,
                                                scale_cc=scale_cc,
                                                rna_only=rna_only)
                    if switch == 3:
                        exp_sw3 = self.predict_exp_ten(tau_sw3, exp_sw2[0, 0],
                                                       exp_sw2[0, 1],
                                                       exp_sw2[0, 2],
                                                       alpha_c, alpha, beta,
                                                       gamma,
                                                       chrom_open=False,
                                                       scale_cc=scale_cc,
                                                       rna_only=rna_only)
                        exp4 = self.predict_exp_ten(tau4, exp_sw3[0, 0],
                                                    exp_sw3[0, 1],
                                                    exp_sw3[0, 2], alpha_c, 0,
                                                    beta, gamma,
                                                    chrom_open=False,
                                                    scale_cc=scale_cc,
                                                    rna_only=rna_only)
        elif model == 2:
            exp1 = self.predict_exp_ten(tau1, 0, 0, 0, alpha_c, alpha, beta,
                                        gamma, pred_r=False, scale_cc=scale_cc,
                                        rna_only=rna_only)
            if switch >= 1:
                exp_sw1 = self.predict_exp_ten(tau_sw1, 0, 0, 0, alpha_c,
                                               alpha, beta, gamma,
                                               pred_r=False, scale_cc=scale_cc,
                                               rna_only=rna_only)
                exp2 = self.predict_exp_ten(tau2, exp_sw1[0, 0], exp_sw1[0, 1],
                                            exp_sw1[0, 2], alpha_c, alpha,
                                            beta, gamma, scale_cc=scale_cc,
                                            rna_only=rna_only)
                if switch >= 2:
                    exp_sw2 = self.predict_exp_ten(tau_sw2, exp_sw1[0, 0],
                                                   exp_sw1[0, 1],
                                                   exp_sw1[0, 2], alpha_c,
                                                   alpha, beta, gamma,
                                                   scale_cc=scale_cc,
                                                   rna_only=rna_only)
                    exp3 = self.predict_exp_ten(tau3, exp_sw2[0, 0],
                                                exp_sw2[0, 1],
                                                exp_sw2[0, 2], alpha_c, 0,
                                                beta, gamma, scale_cc=scale_cc,
                                                rna_only=rna_only)
                    if switch == 3:
                        exp_sw3 = self.predict_exp_ten(tau_sw3, exp_sw2[0, 0],
                                                       exp_sw2[0, 1],
                                                       exp_sw2[0, 2],
                                                       alpha_c, 0, beta, gamma,
                                                       scale_cc=scale_cc,
                                                       rna_only=rna_only)
                        exp4 = self.predict_exp_ten(tau4, exp_sw3[0, 0],
                                                    exp_sw3[0, 1],
                                                    exp_sw3[0, 2],
                                                    alpha_c, 0, beta, gamma,
                                                    chrom_open=False,
                                                    scale_cc=scale_cc,
                                                    rna_only=rna_only)
        return [exp1, exp2, exp3, exp4], [exp_sw1, exp_sw2, exp_sw3]

    def check_partial_trajectory(self, fit_gmm=True, fit_slope=True,
                                 determine_model=True):
        w_non_zero = ((self.c >= 0.1 * np.max(self.c)) &
                      (self.u >= 0.1 * np.max(self.u)) &
                      (self.s >= 0.1 * np.max(self.s)))
        u_non_zero = self.u[w_non_zero]
        s_non_zero = self.s[w_non_zero]
        if len(u_non_zero) < 10:
            self.low_quality = True
            return

        # GMM
        w_low = ((np.percentile(s_non_zero, 30) <= s_non_zero) &
                 (s_non_zero <= np.percentile(s_non_zero, 40)))
        if np.sum(w_low) < 10:
            fit_gmm = False
            self.partial = True
        #if self.local_std is None:
        #    main_info('local standard deviation not provided. '
        #              'Skipping GMM..', indent_level=2)
        #if self.embed_coord is None:
        #    main_info('Warning: embedded coordinates not provided. '
        #              'Skipping GMM..')
        if (fit_gmm and self.local_std is not None and self.embed_coord
                is not None):

            pdist = pairwise_distances(
                self.embed_coord[w_non_zero, :][w_low, :])
            dists = (np.ravel(pdist[np.triu_indices_from(pdist, k=1)])
                     .reshape(-1, 1))
            model = GaussianMixture(n_components=2, covariance_type='tied',
                                    random_state=2021).fit(dists)
            mean_diff = np.abs(model.means_[1][0] - model.means_[0][0])
            criterion1 = mean_diff > self.local_std / self.tm
            #main_info(f'GMM: difference between means = {mean_diff}, '
            #            f'threshold = {self.local_std / self.tm}.', indent_level=2)
            criterion2 = np.all(model.weights_[1] > 0.2 / self.tm)
            #main_info('GMM: weight of the second Gaussian ='
            #            f' {model.weights_[1]}.', indent_level=2)
            if criterion1 and criterion2:
                self.partial = False
            else:
                self.partial = True
            #main_info(f'GMM decides {"" if self.partial else "not "}'
            #            'partial.', indent_level=2)

        # steady-state slope
        wu = self.u >= np.percentile(u_non_zero, 95)
        ws = self.s >= np.percentile(s_non_zero, 95)
        ss_u = self.u[wu | ws]
        ss_s = self.s[wu | ws]
        if np.all(ss_u == 0) or np.all(ss_s == 0):
            self.low_quality = True
            return
        gamma = np.dot(ss_u, ss_s) / np.dot(ss_s, ss_s)
        self.steady_state_func = lambda x: gamma*x

        # thickness of phase portrait
        u_norm = u_non_zero / np.max(self.u)
        s_norm = s_non_zero / np.max(self.s)
        exp = np.hstack((np.reshape(u_norm, (-1, 1)),
                         np.reshape(s_norm, (-1, 1))))
        U, S, Vh = np.linalg.svd(exp)
        self.thickness = S[1]

        # slope-based direction decision
        with np.errstate(divide='ignore', invalid='ignore'):
            slope = self.u / self.s
        non_nan = ~np.isnan(slope)
        slope = slope[non_nan]
        on = slope >= gamma
        off = slope < gamma
        if len(ss_u) < 10 or len(u_non_zero) < 10:
            fit_slope = False
            self.direction = 'complete'
        if fit_slope:
            slope_ = u_non_zero / s_non_zero
            on_ = slope_ >= gamma
            off_ = slope_ < gamma
            on_dist = np.sum((u_non_zero[on_] - gamma * s_non_zero[on_])**2)
            off_dist = np.sum((gamma * s_non_zero[off_] - u_non_zero[off_])**2)
            #main_info(f'Slope: SSE on induction phase = {on_dist},'
            #            f' SSE on repression phase = {off_dist}.', indent_level=2)
            if self.thickness < 1.5 / np.sqrt(self.tm):
                narrow = True
            else:
                narrow = False
            #main_info(f'Thickness of trajectory = {self.thickness}. '
            #            f'Trajectory is {"narrow" if narrow else "normal"}.',
            #            indent_level=2)
            if on_dist > 10 * self.tm**2 * off_dist:
                self.direction = 'on'
                self.partial = True
            elif off_dist > 10 * self.tm**2 * on_dist:
                self.direction = 'off'
                self.partial = True
            else:
                if self.partial is True:
                    if on_dist > 3 * self.tm * off_dist:
                        self.direction = 'on'
                    elif off_dist > 3 * self.tm * on_dist:
                        self.direction = 'off'
                    else:
                        if narrow:
                            self.direction = 'on'
                        else:
                            self.direction = 'complete'
                            self.partial = False
                else:
                    if narrow:
                        self.direction = ('off'
                                          if off_dist > 2 * self.tm * on_dist
                                          else 'on')
                        self.partial = True
                    else:
                        self.direction = 'complete'

        # model pre-determination
        if self.direction == 'on':
            self.model_ = 1
        elif self.direction == 'off':
            self.model_ = 2
        else:
            c_high = self.c >= np.mean(self.c) + 2 * np.std(self.c)
            c_high = c_high[non_nan]
            if np.sum(c_high) < 10:
                c_high = self.c >= np.mean(self.c) + np.std(self.c)
                c_high = c_high[non_nan]
            if np.sum(c_high) < 10:
                c_high = self.c >= np.percentile(self.c, 90)
                c_high = c_high[non_nan]
            if np.sum(self.c[non_nan][c_high] == 0) > 0.5*np.sum(c_high):
                self.low_quality = True
                return
            c_high_on = np.sum(c_high & on)
            c_high_off = np.sum(c_high & off)
            if c_high_on > c_high_off:
                self.model_ = 1
            else:
                self.model_ = 2
        if determine_model:
            self.model = self.model_

        '''
        if not self.known_pars:
            if fit_gmm or fit_slope:
                main_info(f'predicted partial trajectory: {self.partial}',
                            indent_level=1)
                main_info('predicted trajectory direction:'
                            f'{self.direction}', indent_level=1)
            if determine_model:
                main_info(f'predicted model: {self.model}', indent_level=1)
        '''

    def initialize_steady_state_params(self, model_mismatch=False):
        self.scale_cc = 1.0
        self.rescale_c = 1.0
        # estimate rescale factor for u
        s_norm = self.s / self.max_s
        u_mid = (self.u >= 0.4 * self.max_u) & (self.u <= 0.6 * self.max_u)
        if np.sum(u_mid) < 10:
            self.rescale_u = self.thickness / 5
        else:
            s_low, s_high = np.percentile(s_norm[u_mid], [2, 98])
            s_dist = s_high - s_low
            self.rescale_u = s_dist
        if self.rescale_u == 0:
            self.low_quality = True
            return

        c = self.c / self.rescale_c
        u = self.u / self.rescale_u
        s = self.s

        # some extreme values
        wu = u >= np.percentile(u, 97)
        ws = s >= np.percentile(s, 97)
        ss_u = u[wu | ws]
        ss_s = s[wu | ws]
        c_upper = np.mean(c[wu | ws])

        c_high = c >= np.mean(c)
        # _r stands for repressed state
        c0_r = np.mean(c[c_high])
        u0_r = np.mean(ss_u)
        s0_r = np.mean(ss_s)
        if c0_r < c_upper:
            c0_r = c_upper + 0.1

        # adjust chromatin level for reasonable initialization
        if model_mismatch or not self.fit_decoupling:
            c_indu = np.mean(c[self.u > self.steady_state_func(self.s)])
            c_repr = np.mean(c[self.u < self.steady_state_func(self.s)])
            if c_indu == np.nan or c_repr == np.nan:
                self.low_quality = True
                return
            c0_r = np.mean(c[c >= np.min([c_indu, c_repr])])

        # initialize rates
        self.alpha_c = 0.1
        self.beta = 1.0
        self.gamma = np.dot(ss_u, ss_s) / np.dot(ss_s, ss_s)
        alpha = u0_r
        self.alpha = u0_r
        self.rates = np.array([self.alpha_c, self.alpha, self.beta,
                               self.gamma])

        # RNA-only
        if self.rna_only:
            t_sw_1 = 0.1
            t_sw_3 = 20.0
            if self.init_mode == 'grid':
                
                # arange returns sequence [2,6,10,14,18]
                for t_sw_2 in np.arange(2, 20, 4, dtype=np.float64):
                    self.update(self.params, initialize=True, adjust_time=False,
                                plot=False)

            elif self.init_mode == 'simple':
                t_sw_2 = 10
                self.params = np.array([t_sw_1,
                                        t_sw_2-t_sw_1,
                                        t_sw_3-t_sw_2,
                                        self.alpha_c,
                                        self.alpha,
                                        self.beta,
                                        self.gamma,
                                        self.scale_cc,
                                        self.rescale_c,
                                        self.rescale_u])

            elif self.init_mode == 'invert':
                t_sw_2 = approx_tau(u0_r, s0_r, 0, 0, alpha, self.beta,
                                    self.gamma)
                if t_sw_2 <= 0.2:
                    t_sw_2 = 1.0
                elif t_sw_2 >= 19.9:
                    t_sw_2 = 19.0
                self.params = np.array([t_sw_1,
                                        t_sw_2-t_sw_1,
                                        t_sw_3-t_sw_2,
                                        self.alpha_c,
                                        self.alpha,
                                        self.beta,
                                        self.gamma,
                                        self.scale_cc,
                                        self.rescale_c,
                                        self.rescale_u])

        # chromatin-RNA
        else:
            if self.init_mode == 'grid':
                # arange returns sequence [1,5,9,13,17]
                for t_sw_1 in np.arange(1, 18, 4, dtype=np.float64):
                    # arange returns sequence 2,6,10,14,18
                    for t_sw_2 in np.arange(t_sw_1+1, 19, 4, dtype=np.float64):
                        # arange returns sequence [3,7,11,15,19]
                        for t_sw_3 in np.arange(t_sw_2+1, 20, 4,
                                                dtype=np.float64):
                            if not self.fit_decoupling:
                                t_sw_3 = t_sw_2 + 30 / self.n_anchors
                            params = np.array([t_sw_1,
                                               t_sw_2-t_sw_1,
                                               t_sw_3-t_sw_2,
                                               self.alpha_c,
                                               self.alpha,
                                               self.beta,
                                               self.gamma,
                                               self.scale_cc,
                                               self.rescale_c,
                                               self.rescale_u])
                            self.update(params, initialize=True,
                                        adjust_time=False, plot=False)
                            if not self.fit_decoupling:
                                break

            elif self.init_mode == 'simple':
                t_sw_1, t_sw_2, t_sw_3 = 5, 10, 15 \
                    if not self.fit_decoupling \
                    else 10.1
                self.params = np.array([t_sw_1,
                                        t_sw_2-t_sw_1,
                                        t_sw_3-t_sw_2,
                                        self.alpha_c,
                                        self.alpha,
                                        self.beta,
                                        self.gamma,
                                        self.scale_cc,
                                        self.rescale_c,
                                        self.rescale_u])

            elif self.init_mode == 'invert':
                self.alpha = u0_r / c_upper
                if model_mismatch or not self.fit_decoupling:
                    self.alpha = u0_r / c0_r
                rna_interval = approx_tau(u0_r, s0_r, 0, 0, alpha, self.beta,
                                          self.gamma)
                rna_interval = np.clip(rna_interval, 3, 12)
                if self.model == 1:
                    for t_sw_1 in np.arange(1, rna_interval-1, 2,
                                            dtype=np.float64):
                        t_sw_3 = rna_interval + t_sw_1
                        for t_sw_2 in np.arange(t_sw_1+1, rna_interval, 2,
                                                dtype=np.float64):
                            if not self.fit_decoupling:
                                t_sw_2 = t_sw_3 - 30 / self.n_anchors

                            alpha_c = -np.log(1 - c0_r) / t_sw_2
                            params = np.array([t_sw_1,
                                               t_sw_2-t_sw_1,
                                               t_sw_3-t_sw_2,
                                               alpha_c,
                                               self.alpha,
                                               self.beta,
                                               self.gamma,
                                               self.scale_cc,
                                               self.rescale_c,
                                               self.rescale_u])
                            self.update(params, initialize=True,
                                        adjust_time=False, plot=False)
                            if not self.fit_decoupling:
                                break

                elif self.model == 2:
                    for t_sw_1 in np.arange(1, rna_interval, 2,
                                            dtype=np.float64):
                        t_sw_2 = rna_interval + t_sw_1
                        for t_sw_3 in np.arange(t_sw_2+1, t_sw_2+6, 2,
                                                dtype=np.float64):
                            if not self.fit_decoupling:
                                t_sw_3 = t_sw_2 + 30 / self.n_anchors

                            alpha_c = -np.log(1 - c0_r) / t_sw_3
                            params = np.array([t_sw_1,
                                               t_sw_2-t_sw_1,
                                               t_sw_3-t_sw_2,
                                               alpha_c,
                                               self.alpha,
                                               self.beta,
                                               self.gamma,
                                               self.scale_cc,
                                               self.rescale_c,
                                               self.rescale_u])
                            self.update(params, initialize=True,
                                        adjust_time=False, plot=False)
                            if not self.fit_decoupling:
                                break

        self.loss = [self.mse(self.params)]
        self.t_sw_array = np.array([self.params[0],
                                    self.params[0]+self.params[1],
                                    self.params[0]+self.params[1]
                                    + self.params[2]])
        self.t_sw_1, self.t_sw_2, self.t_sw_3 = self.t_sw_array

        #main_info(f'initial params:\nswitch time array = {self.t_sw_array},'
         #           '\n'
         #           f'rates = {self.rates},\ncc scale = {self.scale_cc},\n'
         #           f'c rescale factor = {self.rescale_c},\n'
         #           f'u rescale factor = {self.rescale_u}', indent_level=1)
        #main_info(f'initial loss: {self.loss[-1]}', indent_level=1)

    def fit(self):
        if self.low_quality:
            return self.loss

        if self.plot:
            plt.ion()
            self.fig = plt.figure(figsize=self.fig_size)
            if self.rna_only:
                self.ax = self.fig.add_subplot(111)
            else:
                self.ax = self.fig.add_subplot(111, projection='3d')

        if not self.known_pars:
            self.fit_dyn()

        self.update(self.params, perform_update=True, fit_outlier=True,
                    plot=True)

        # remove long gaps in the last observed state
        t_sorted = np.sort(self.t)
        dt = np.diff(t_sorted, prepend=0)
        mean_dt = np.mean(dt)
        std_dt = np.std(dt)
        gap_thresh = np.clip(mean_dt+3*std_dt, 3*20/self.n_anchors, None)
        if gap_thresh > 0:
            idx = np.where(dt > gap_thresh)[0]
            gap_sum = 0
            last_t_sw = np.max(self.t_sw_array[self.t_sw_array < 20])
            for i in idx:
                t1 = t_sorted[i-1] if i > 0 else 0
                t2 = t_sorted[i]
                if t1 > last_t_sw and t2 <= 20:
                    gap_sum += np.clip(t2 - t1 - mean_dt, 0, None)
            if last_t_sw > np.max(self.t):
                gap_sum += 20 - last_t_sw
            realign_ratio = np.clip(20/(20 - gap_sum), None, 20/last_t_sw)
            #main_info(f'removing gaps and realigning by {realign_ratio}..',
            #            indent_level=1)
            self.rates /= realign_ratio
            self.alpha_c, self.alpha, self.beta, self.gamma = self.rates
            self.params[:3] *= realign_ratio
            self.params[3:7] = self.rates
            self.t_sw_array = np.array([self.params[0],
                                        self.params[0]+self.params[1],
                                        self.params[0]+self.params[1]
                                        + self.params[2]])
            self.t_sw_1, self.t_sw_2, self.t_sw_3 = self.t_sw_array
            self.update(self.params, perform_update=True, fit_outlier=True,
                        plot=True)

        if self.plot:
            plt.ioff()
            plt.show(block=True)

        # likelihood
        #main_info('computing likelihood..', indent_level=1)
        keep = self.non_zero & self.non_outlier & \
            (self.u_all > 0.2 * np.percentile(self.u_all, 99.5)) & \
            (self.s_all > 0.2 * np.percentile(self.s_all, 99.5))
        scale_factor = np.array([self.scale_c / self.std_c,
                                 self.scale_u / self.std_u,
                                 self.scale_s / self.std_s])
        if np.sum(keep) >= 10:
            self.likelihood, self.l_c, self.ssd_c, self.var_c, l_u, l_s = \
                compute_likelihood(self.c_all,
                                   self.u_all,
                                   self.s_all,
                                   self.t_sw_array,
                                   self.alpha_c,
                                   self.alpha,
                                   self.beta,
                                   self.gamma,
                                   self.rescale_c,
                                   self.rescale_u,
                                   self.t,
                                   self.state,
                                   scale_cc=self.scale_cc,
                                   scale_factor=scale_factor,
                                   model=self.model,
                                   weight=keep,
                                   rna_only=self.rna_only)
        else:
            self.likelihood, self.l_c, self.ssd_c, self.var_c, l_u = \
                0, 0, 0, 0, 0
            # TODO: Keep? Remove??
            l_s = 0

        #if not self.rna_only:
            #main_info(f'likelihood of c: {self.l_c}, likelihood of u: {l_u},'
            #            f' likelihood of s: {l_s}', indent_level=1)

        # velocity
        #main_info('computing velocities..', indent_level=1)
        self.velocity = np.empty((len(self.u_all), 3))
        if self.conn is not None:
            new_time = self.conn.dot(self.t)
            new_time[new_time > 20] = 20
            new_state = self.state.copy()
            new_state[new_time <= self.t_sw_1] = 0
            new_state[(self.t_sw_1 < new_time) & (new_time <= self.t_sw_2)] = 1
            new_state[(self.t_sw_2 < new_time) & (new_time <= self.t_sw_3)] = 2
            new_state[self.t_sw_3 < new_time] = 3

        else:
            new_time = self.t
            new_state = self.state

        self.alpha_c, self.alpha, self.beta, self.gamma = \
            check_params(self.alpha_c, self.alpha, self.beta, self.gamma)
        vc, vu, vs = compute_velocity(new_time,
                                      self.t_sw_array,
                                      new_state,
                                      self.alpha_c,
                                      self.alpha,
                                      self.beta,
                                      self.gamma,
                                      self.rescale_c,
                                      self.rescale_u,
                                      scale_cc=self.scale_cc,
                                      model=self.model,
                                      rna_only=self.rna_only)

        self.velocity[:, 0] = vc * self.scale_c
        self.velocity[:, 1] = vu * self.scale_u
        self.velocity[:, 2] = vs * self.scale_s

        # anchor expression and velocity
        anchor_time, tau_list = anchor_points(self.t_sw_array, 20,
                                              self.n_anchors, return_time=True)
        switch = np.sum(self.t_sw_array < 20)
        typed_tau_list = List()
        [typed_tau_list.append(x) for x in tau_list]
        self.alpha_c, self.alpha, self.beta, self.gamma, \
            self.c0, self.u0, self.s0 = \
            check_params(self.alpha_c, self.alpha, self.beta, self.gamma,
                         c0=self.c0, u0=self.u0, s0=self.s0)
        exp_list, exp_sw_list = generate_exp(typed_tau_list,
                                             self.t_sw_array[:switch],
                                             self.alpha_c,
                                             self.alpha,
                                             self.beta,
                                             self.gamma,
                                             scale_cc=self.scale_cc,
                                             model=self.model,
                                             rna_only=self.rna_only)
        rescale_factor = np.array([self.rescale_c, self.rescale_u, 1.0])
        exp_list = [x*rescale_factor for x in exp_list]
        exp_sw_list = [x*rescale_factor for x in exp_sw_list]
        c = np.ravel(np.concatenate([exp_list[x][:, 0]
                                     for x in range(switch+1)]))
        u = np.ravel(np.concatenate([exp_list[x][:, 1]
                                     for x in range(switch+1)]))
        s = np.ravel(np.concatenate([exp_list[x][:, 2]
                                     for x in range(switch+1)]))
        c_sw = np.ravel(np.concatenate([exp_sw_list[x][:, 0]
                                        for x in range(switch)]))
        u_sw = np.ravel(np.concatenate([exp_sw_list[x][:, 1]
                                        for x in range(switch)]))
        s_sw = np.ravel(np.concatenate([exp_sw_list[x][:, 2]
                                        for x in range(switch)]))
        self.alpha_c, self.alpha, self.beta, self.gamma = \
            check_params(self.alpha_c, self.alpha, self.beta, self.gamma)
        vc, vu, vs = compute_velocity(anchor_time,
                                      self.t_sw_array,
                                      None,
                                      self.alpha_c,
                                      self.alpha,
                                      self.beta,
                                      self.gamma,
                                      self.rescale_c,
                                      self.rescale_u,
                                      scale_cc=self.scale_cc,
                                      model=self.model,
                                      rna_only=self.rna_only)

        # scale and shift back to original scale
        c_ = c * self.scale_c + self.offset_c
        u_ = u * self.scale_u + self.offset_u
        s_ = s * self.scale_s + self.offset_s
        c_sw_ = c_sw * self.scale_c + self.offset_c
        u_sw_ = u_sw * self.scale_u + self.offset_u
        s_sw_ = s_sw * self.scale_s + self.offset_s
        vc = vc * self.scale_c
        vu = vu * self.scale_u
        vs = vs * self.scale_s

        self.anchor_exp = np.empty((len(u_), 3))
        self.anchor_exp[:, 0], self.anchor_exp[:, 1], self.anchor_exp[:, 2] = \
            c_, u_, s_
        self.anchor_exp_sw = np.empty((len(u_sw_), 3))
        self.anchor_exp_sw[:, 0], self.anchor_exp_sw[:, 1], \
            self.anchor_exp_sw[:, 2] = c_sw_, u_sw_, s_sw_
        self.anchor_velo = np.empty((len(u_), 3))
        self.anchor_velo[:, 0] = vc
        self.anchor_velo[:, 1] = vu
        self.anchor_velo[:, 2] = vs
        self.anchor_velo_min_idx = np.sum(anchor_time < np.min(new_time))
        self.anchor_velo_max_idx = np.sum(anchor_time < np.max(new_time)) - 1

        if self.save_plot:
            main_info('saving plots..', indent_level=1)
            self.save_dyn_plot(c_, u_, s_, c_sw_, u_sw_, s_sw_, tau_list)

        self.realign_time_and_velocity(c, u, s, anchor_time)

        #main_info(f'final params:\nswitch time array = {self.t_sw_array},\n'
       #             f'rates = {self.rates},\ncc scale = {self.scale_cc},\n'
        #            f'c rescale factor = {self.rescale_c},\n'
        #            f'u rescale factor = {self.rescale_u}',
        #            indent_level=1)
        #main_info(f'final loss: {self.loss[-1]}', indent_level=1)
        #main_info(f'final likelihood: {self.likelihood}', indent_level=1)

        return self.loss

    # the adam algorithm
    # NOTE: The starting point for this function was an excample on the
    # GeeksForGeeks website. The particular article is linked below:
    # www.geeksforgeeks.org/how-to-implement-adam-gradient-descent-from-scratch-using-python/
    def AdamMin(self, x, n_iter, tol, eps=1e-8):

        n = len(x)

        x_ten = torch.tensor(x, requires_grad=True, device=self.device,
                             dtype=self.torch_type)

        # record lowest loss as a benchmark
        # (right now the lowest loss is the current loss)
        lowest_loss = torch.tensor(np.array(self.loss[-1], dtype=self.u.dtype),
                                   device=self.device,
                                   dtype=self.torch_type)

        # record the tensor of the parameters that cause the lowest loss
        lowest_x_ten = x_ten

        # the m and v variables used in the adam calculations
        m = torch.zeros(n, device=self.device, requires_grad=True,
                        dtype=self.torch_type)
        v = torch.zeros(n, device=self.device, requires_grad=True,
                        dtype=self.torch_type)

        # the update amount to add to the x tensor after the appropriate
        # calculations are made
        u = torch.ones(n, device=self.device, requires_grad=True,
                       dtype=self.torch_type) * float("inf")

        # how many times the new loss is lower than the lowest loss
        update_count = 0

        iterations = 0

        # run the gradient descent updates
        for t in range(n_iter):

            iterations += 1

            # calculate the loss
            loss = self.mse_ten(x_ten)

            # if the loss is lower than the lowest loss...
            if loss < lowest_loss:

                # record the new best tensor
                lowest_x_ten = x_ten
                update_count += 1

                # if the percentage difference in x tensors and loss values
                # is less than the tolerance parameter and we've update the
                # loss 3 times by now...
                if torch.all((torch.abs(u) / lowest_x_ten) < tol) and \
                    (torch.abs(loss - lowest_loss) / lowest_loss) < tol and \
                        update_count >= 3:

                    # ...we've updated enough. Break!
                    break

                # record the new lowest loss
                lowest_loss = loss

            # take the gradient of mse w/r/t our current parameter values
            loss.backward(inputs=x_ten)
            g = x_ten.grad

            # calculate the new update value using the Adam formula
            m = (self.adam_beta1 * m) + ((1.0 - self.adam_beta1) * g)
            v = (self.adam_beta2 * v) + ((1.0 - self.adam_beta2) * g * g)

            mhat = m / (1.0 - (self.adam_beta1**(t+1)))
            vhat = v / (1.0 - (self.adam_beta2**(t+1)))

            u = -(self.adam_lr * mhat) / (torch.sqrt(vhat) + eps)

            # update the x tensor
            x_ten = x_ten + u

        # as long as we've found at least one better x tensor...
        if update_count > 1:

            # record the final lowest loss
            if loss < lowest_loss:
                lowest_loss = loss

            # set the new loss for the gene to the new lowest loss
            self.cur_loss = lowest_loss.item()

            # use the update() function so the gene's parameters
            # are the new best one we found
            updated = self.update(lowest_x_ten.cpu().detach().numpy())

        # if we never found a better x tensor, then the return value should
        # state that we did not update it
        else:
            updated = False

        # return whether we updated the x tensor or not
        return updated

    def fit_dyn(self):

        while self.cur_iter < self.max_iter:
            self.cur_iter += 1

            # RNA-only
            if self.rna_only:
                #main_info('Nelder Mead on t_sw_2 and alpha..', indent_level=2)
                self.fitting_flag_ = 0
                if self.cur_iter == 1:
                    var_test = (self.alpha +
                                np.array([-2, -1, -0.5, 0.5, 1, 2]) * 0.1
                                * self.alpha)
                    new_params = self.params.copy()
                    for var in var_test:
                        new_params[4] = var
                        self.update(new_params, adjust_time=False,
                                    penalize_gap=False)
                res = minimize(self.mse, x0=[self.params[1], self.params[4]],
                               method='Nelder-Mead', tol=1e-2,
                               callback=self.update, options={'maxiter': 3})

                if self.fit_rescale:
                    #main_info('Nelder Mead on t_sw_2, beta, and rescale u..',
                    #            indent_level=2)
                    res = minimize(self.mse, x0=[self.params[1],
                                                 self.params[5],
                                                 self.params[9]],
                                   method='Nelder-Mead', tol=1e-2,
                                   callback=self.update,
                                   options={'maxiter': 5})

                #main_info('Nelder Mead on alpha and gamma..', indent_level=2)
                self.fitting_flag_ = 1
                res = minimize(self.mse, x0=[self.params[4], self.params[6]],
                               method='Nelder-Mead', tol=1e-2,
                               callback=self.update, options={'maxiter': 3})

                #main_info('Nelder Mead on t_sw_2..', indent_level=2)
                res = minimize(self.mse, x0=[self.params[1]],
                               method='Nelder-Mead', tol=1e-2,
                               callback=self.update, options={'maxiter': 2})

                #main_info('Full Nelder Mead..', indent_level=2)
                res = minimize(self.mse, x0=[self.params[1], self.params[4],
                                             self.params[5], self.params[6]],
                               method='Nelder-Mead', tol=1e-2,
                               callback=self.update, options={'maxiter': 5})

            # chromatin-RNA
            else:

                if not self.adam:
                    #main_info('Nelder Mead on t_sw_1, chromatin switch time,'
                    #            'and alpha_c..', indent_level=2)
                    self.fitting_flag_ = 1
                    if self.cur_iter == 1:
                        var_test = (self.gamma + np.array([-1, -0.5, 0.5, 1])
                                    * 0.1 * self.gamma)
                        new_params = self.params.copy()
                        for var in var_test:
                            new_params[6] = var
                            self.update(new_params, adjust_time=False)
                    if self.model == 0 or self.model == 1:
                        res = minimize(self.mse, x0=[self.params[0],
                                                     self.params[1],
                                                     self.params[3]],
                                       method='Nelder-Mead', tol=1e-2,
                                       callback=self.update,
                                       options={'maxiter': 20})
                    elif self.model == 2:
                        res = minimize(self.mse, x0=[self.params[0],
                                                     self.params[2],
                                                     self.params[3]],
                                       method='Nelder-Mead', tol=1e-2,
                                       callback=self.update,
                                       options={'maxiter': 20})

                    #main_info('Nelder Mead on chromatin switch time,'
                    #            'chromatin closing rate scaling, and rescale'
                    #            'c..', indent_level=2)
                    self.fitting_flag_ = 2
                    if self.model == 0 or self.model == 1:
                        res = minimize(self.mse, x0=[self.params[1],
                                                     self.params[7],
                                                     self.params[8]],
                                       method='Nelder-Mead', tol=1e-2,
                                       callback=self.update,
                                       options={'maxiter': 20})
                    elif self.model == 2:
                        res = minimize(self.mse, x0=[self.params[2],
                                                     self.params[7],
                                                     self.params[8]],
                                       method='Nelder-Mead', tol=1e-2,
                                       callback=self.update,
                                       options={'maxiter': 20})

                    #main_info('Nelder Mead on rna switch time and alpha..',
                    #            indent_level=2)
                    self.fitting_flag_ = 1
                    if self.model == 0 or self.model == 1:
                        res = minimize(self.mse, x0=[self.params[2],
                                                     self.params[4]],
                                       method='Nelder-Mead', tol=1e-2,
                                       callback=self.update,
                                       options={'maxiter': 10})
                    elif self.model == 2:
                        res = minimize(self.mse, x0=[self.params[1],
                                                     self.params[4]],
                                       method='Nelder-Mead', tol=1e-2,
                                       callback=self.update,
                                       options={'maxiter': 10})

                    #main_info('Nelder Mead on rna switch time, beta, and '
                    #            'rescale u..', indent_level=2)
                    self.fitting_flag_ = 3
                    if self.model == 0 or self.model == 1:
                        res = minimize(self.mse, x0=[self.params[2],
                                                     self.params[5],
                                                     self.params[9]],
                                       method='Nelder-Mead', tol=1e-2,
                                       callback=self.update,
                                       options={'maxiter': 20})
                    elif self.model == 2:
                        res = minimize(self.mse, x0=[self.params[1],
                                                     self.params[5],
                                                     self.params[9]],
                                       method='Nelder-Mead', tol=1e-2,
                                       callback=self.update,
                                       options={'maxiter': 20})

                    #main_info('Nelder Mead on alpha and gamma..', indent_level=2)
                    self.fitting_flag_ = 2
                    res = minimize(self.mse, x0=[self.params[4],
                                                 self.params[6]],
                                   method='Nelder-Mead', tol=1e-2,
                                   callback=self.update,
                                   options={'maxiter': 10})

                    #main_info('Nelder Mead on t_sw..', indent_level=2)
                    self.fitting_flag_ = 4
                    res = minimize(self.mse, x0=self.params[:3],
                                   method='Nelder-Mead', tol=1e-2,
                                   callback=self.update,
                                   options={'maxiter': 20})

                else:

                    #main_info('Adam on all parameters', indent_level=2)
                    self.AdamMin(np.array(self.params, dtype=self.u.dtype), 20,
                                 tol=1e-2)

                    #main_info('Nelder Mead on t_sw..', indent_level=2)
                    self.fitting_flag_ = 4
                    res = minimize(self.mse, x0=self.params[:3],
                                   method='Nelder-Mead', tol=1e-2,
                                   callback=self.update,
                                   options={'maxiter': 15})

            #main_info(f'iteration {self.cur_iter} finished', indent_level=2)

    def _variables(self, x):
        scale_cc = self.scale_cc
        rescale_c = self.rescale_c
        rescale_u = self.rescale_u

        # RNA-only
        if self.rna_only:
            if len(x) == 1:  # fit t_sw_2
                t3 = np.array([self.t_sw_1, x[0],
                               self.t_sw_3 - self.t_sw_1 - x[0]])
                r4 = self.rates

            elif len(x) == 2:
                if self.fitting_flag_:  # fit alpha and gamma
                    t3 = self.params[:3]
                    r4 = np.array([self.alpha_c, x[0], self.beta, x[1]])
                else:  # fit t_sw_2 and alpha
                    t3 = np.array([self.t_sw_1, x[0],
                                   self.t_sw_3 - self.t_sw_1 - x[0]])
                    r4 = np.array([self.alpha_c, x[1], self.beta, self.gamma])

            elif len(x) == 3:  # fit t_sw_2, beta, and rescale u
                t3 = np.array([self.t_sw_1,
                               x[0], self.t_sw_3 - self.t_sw_1 - x[0]])
                r4 = np.array([self.alpha_c, self.alpha, x[1], self.gamma])
                rescale_u = x[2]

            elif len(x) == 4:  # fit all
                t3 = np.array([self.t_sw_1, x[0], self.t_sw_3 - self.t_sw_1
                               - x[0]])
                r4 = np.array([self.alpha_c, x[1], x[2], x[3]])

            elif len(x) == 10:  # all available
                t3 = x[:3]
                r4 = x[3:7]
                scale_cc = x[7]
                rescale_c = x[8]
                rescale_u = x[9]

            else:
                return

        # chromatin-RNA
        else:

            if len(x) == 2:
                if self.fitting_flag_ == 1:  # fit rna switch time and alpha
                    if self.model == 0 or self.model == 1:
                        t3 = np.array([self.t_sw_1, self.params[1], x[0]])
                    elif self.model == 2:
                        t3 = np.array([self.t_sw_1, x[0],
                                       self.t_sw_3 - self.t_sw_1 - x[0]])
                    r4 = np.array([self.alpha_c, x[1], self.beta, self.gamma])
                elif self.fitting_flag_ == 2:  # fit alpha and gamma
                    t3 = self.params[:3]
                    r4 = np.array([self.alpha_c, x[0], self.beta, x[1]])

            elif len(x) == 3:
                # fit t_sw_1, chromatin switch time, and alpha_c
                if self.fitting_flag_ == 1:
                    if self.model == 0 or self.model == 1:
                        t3 = np.array([x[0], x[1], self.t_sw_3 - x[0] - x[1]])
                    elif self.model == 2:
                        t3 = np.array([x[0], self.t_sw_2 - x[0], x[1]])
                    r4 = np.array([x[2], self.alpha, self.beta, self.gamma])
                # fit chromatin switch time, chromatin closing rate scaling,
                # and rescale c
                elif self.fitting_flag_ == 2:
                    if self.model == 0 or self.model == 1:
                        t3 = np.array([self.t_sw_1, x[0],
                                       self.t_sw_3 - self.t_sw_1 - x[0]])
                    elif self.model == 2:
                        t3 = np.array([self.t_sw_1, self.params[1], x[0]])
                    r4 = self.rates
                    scale_cc = x[1]
                    rescale_c = x[2]
                # fit rna switch time, beta, and rescale u
                elif self.fitting_flag_ == 3:
                    if self.model == 0 or self.model == 1:
                        t3 = np.array([self.t_sw_1, self.params[1], x[0]])
                    elif self.model == 2:
                        t3 = np.array([self.t_sw_1, x[0],
                                       self.t_sw_3 - self.t_sw_1 - x[0]])
                    r4 = np.array([self.alpha_c, self.alpha, x[1], self.gamma])
                    rescale_u = x[2]
                # fit three switch times
                elif self.fitting_flag_ == 4:
                    t3 = x
                    r4 = self.rates

            elif len(x) == 7:
                t3 = x[:3]
                r4 = x[3:]

            elif len(x) == 10:
                t3 = x[:3]
                r4 = x[3:7]
                scale_cc = x[7]
                rescale_c = x[8]
                rescale_u = x[9]

            else:
                return

        # clip to meaningful values
        if self.fitting_flag_ and not self.adam:
            scale_cc = np.clip(scale_cc,
                               np.max([0.5*self.scale_cc, 0.25]),
                               np.min([2*self.scale_cc, 4]))

        if not self.known_pars:
            if self.fit_decoupling:
                t3 = np.clip(t3, 0.1, None)
            else:
                t3[2] = 30 / self.n_anchors
                t3[:2] = np.clip(t3[:2], 0.1, None)
            r4 = np.clip(r4, 0.001, 1000)
            rescale_c = np.clip(rescale_c, 0.75, 1.5)
            rescale_u = np.clip(rescale_u, 0.2, 3)

        return t3, r4, scale_cc, rescale_c, rescale_u

    # the tensor version of the calculate_dist_and_time function
    def calculate_dist_and_time_ten(self,
                                    c, u, s,
                                    t_sw_array,
                                    alpha_c, alpha, beta, gamma,
                                    rescale_c, rescale_u,
                                    scale_cc=1,
                                    scale_factor=None,
                                    model=1,
                                    conn=None,
                                    t=1000, k=1,
                                    direction='complete',
                                    total_h=20,
                                    rna_only=False,
                                    penalize_gap=True,
                                    all_cells=True):

        conn = torch.tensor(conn.todense(),
                            device=self.device,
                            dtype=self.torch_type)

        c_ten = torch.tensor(c, device=self.device, dtype=self.torch_type)
        u_ten = torch.tensor(u, device=self.device, dtype=self.torch_type)
        s_ten = torch.tensor(s, device=self.device, dtype=self.torch_type)

        n = len(u)
        if scale_factor is None:
            scale_factor_ten = torch.stack((torch.std(c_ten), torch.std(u_ten),
                                            torch.std(s_ten)))
        else:
            scale_factor_ten = torch.tensor(scale_factor, device=self.device,
                                            dtype=self.torch_type)

        tau_list = self.anchor_points_ten(t_sw_array, total_h, t)

        switch = torch.sum(t_sw_array < total_h)

        exp_list, exp_sw_list = self.generate_exp_tens(tau_list,
                                                       t_sw_array[:switch],
                                                       alpha_c,
                                                       alpha,
                                                       beta,
                                                       gamma,
                                                       model=model,
                                                       scale_cc=scale_cc,
                                                       rna_only=rna_only)

        rescale_factor = torch.stack((rescale_c, rescale_u,
                                     torch.tensor(1.0, device=self.device,
                                                  requires_grad=True,
                                                  dtype=self.torch_type)))

        for i in range(len(exp_list)):
            exp_list[i] = exp_list[i]*rescale_factor

            if i < len(exp_list)-1:
                exp_sw_list[i] = exp_sw_list[i]*rescale_factor

        max_c = 0
        max_u = 0
        max_s = 0

        if rna_only:
            exp_mat = (torch.hstack((torch.reshape(u_ten, (-1, 1)),
                                     torch.reshape(s_ten, (-1, 1))))
                       / scale_factor_ten[1:])
        else:
            exp_mat = torch.hstack((torch.reshape(c_ten, (-1, 1)),
                                    torch.reshape(u_ten, (-1, 1)),
                                    torch.reshape(s_ten, (-1, 1))))\
                                    / scale_factor_ten

        taus = torch.zeros((1, n), device=self.device,
                           requires_grad=True,
                           dtype=self.torch_type)
        anchor_exp, anchor_t = None, None

        dists0 = torch.full((1, n), 0.0 if direction == "on"
                            or direction == "complete" else np.inf,
                            device=self.device,
                            requires_grad=True,
                            dtype=self.torch_type)
        dists1 = torch.full((1, n), 0.0 if direction == "on"
                            or direction == "complete" else np.inf,
                            device=self.device,
                            requires_grad=True,
                            dtype=self.torch_type)
        dists2 = torch.full((1, n), 0.0 if direction == "off"
                            or direction == "complete" else np.inf,
                            device=self.device,
                            requires_grad=True,
                            dtype=self.torch_type)
        dists3 = torch.full((1, n), 0.0 if direction == "off"
                            or direction == "complete" else np.inf,
                            device=self.device,
                            requires_grad=True,
                            dtype=self.torch_type)

        ts0 = torch.zeros((1, n), device=self.device,
                          requires_grad=True,
                          dtype=self.torch_type)
        ts1 = torch.zeros((1, n), device=self.device,
                          requires_grad=True,
                          dtype=self.torch_type)
        ts2 = torch.zeros((1, n), device=self.device,
                          requires_grad=True,
                          dtype=self.torch_type)
        ts3 = torch.zeros((1, n), device=self.device,
                          requires_grad=True,
                          dtype=self.torch_type)

        for i in range(switch+1):

            if not all_cells:
                max_ci = (torch.max(exp_list[i][:, 0])
                          if exp_list[i].shape[0] > 0
                          else 0)
                max_c = max_ci if max_ci > max_c else max_c
            max_ui = torch.max(exp_list[i][:, 1]) if exp_list[i].shape[0] > 0 \
                else 0
            max_u = max_ui if max_ui > max_u else max_u
            max_si = torch.max(exp_list[i][:, 2]) if exp_list[i].shape[0] > 0 \
                else 0
            max_s = max_si if max_si > max_s else max_s

            skip_phase = False
            if direction == 'off':
                if (model in [1, 2]) and (i < 2):
                    skip_phase = True
            elif direction == 'on':
                if (model in [1, 2]) and (i >= 2):
                    skip_phase = True
            if rna_only and i == 0:
                skip_phase = True

            if not skip_phase:
                if rna_only:
                    tmp = exp_list[i][:, 1:] / scale_factor_ten[1:]
                else:
                    tmp = exp_list[i] / scale_factor_ten
                if anchor_exp is None:
                    anchor_exp = exp_list[i]
                    anchor_t = (tau_list[i] + t_sw_array[i-1] if i >= 1
                                else tau_list[i])
                else:
                    anchor_exp = torch.vstack((anchor_exp, exp_list[i]))
                    anchor_t = torch.hstack((anchor_t,
                                             tau_list[i] + t_sw_array[i-1]
                                             if i >= 1 else tau_list[i]))

                if not all_cells:
                    anchor_prepend_rna = torch.zeros((1, 2),
                                                     device=self.device,
                                                     dtype=self.torch_type)
                    anchor_prepend_chrom = torch.zeros((1, 3),
                                                       device=self.device,
                                                       dtype=self.torch_type)
                    anchor_dist = torch.diff(tmp, dim=0,
                                             prepend=anchor_prepend_rna
                                             if rna_only
                                             else anchor_prepend_chrom)

                    anchor_dist = torch.sqrt((anchor_dist*anchor_dist)
                                             .sum(axis=1))
                    remove_cand = anchor_dist < (0.01*torch.max(exp_mat[1])
                                                 if rna_only
                                                 else
                                                 0.01*torch.max(exp_mat[2]))
                    step_idx = torch.arange(0, anchor_dist.size()[0], 1,
                                            device=self.device,
                                            dtype=self.torch_type) % 3 > 0
                    remove_cand &= step_idx
                    keep_idx = torch.where(~remove_cand)[0]

                    tmp = tmp[keep_idx, :]
                from sklearn.neighbors import NearestNeighbors
                model = NearestNeighbors(n_neighbors=k, output_type="numpy")
                model.fit(tmp.detach())
                dd, ii = model.kneighbors(exp_mat.detach())
                ii = ii.T[0]

                new_dd = ((exp_mat[:, 0] - tmp[ii, 0])
                          * (exp_mat[:, 0] - tmp[ii, 0])
                          + (exp_mat[:, 1] - tmp[ii, 1])
                          * (exp_mat[:, 1] - tmp[ii, 1])
                          + (exp_mat[:, 2] - tmp[ii, 2])
                          * (exp_mat[:, 2] - tmp[ii, 2]))

                if k > 1:
                    new_dd = torch.mean(new_dd, dim=1)
                if conn is not None:
                    new_dd = torch.matmul(conn, new_dd)

                if i == 0:
                    dists0 = dists0 + new_dd
                elif i == 1:
                    dists1 = dists1 + new_dd
                elif i == 2:
                    dists2 = dists2 + new_dd
                elif i == 3:
                    dists3 = dists3 + new_dd

                if not all_cells:
                    ii = keep_idx[ii]
                if k == 1:
                    taus = tau_list[i][ii]
                else:
                    for j in range(n):
                        taus[j] = tau_list[i][ii[j, :]]

                if i == 0:
                    ts0 = ts0 + taus
                elif i == 1:
                    ts1 = ts1 + taus + t_sw_array[0]
                elif i == 2:
                    ts2 = ts2 + taus + t_sw_array[1]
                elif i == 3:
                    ts3 = ts3 + taus + t_sw_array[2]

        dists = torch.cat((dists0, dists1, dists2, dists3), 0)

        ts = torch.cat((ts0, ts1, ts2, ts3), 0)

        state_pred = torch.argmin(dists, axis=0)

        t_pred = ts[state_pred, torch.arange(n, device=self.device)]

        anchor_t1_list = []
        anchor_t2_list = []

        t_sw_adjust = torch.zeros(3, device=self.device, dtype=self.torch_type)

        if direction == 'complete':

            dist_gap_add = torch.zeros((1, n), device=self.device,
                                       dtype=self.torch_type)

            t_sorted = torch.clone(t_pred)
            t_sorted, t_sorted_indices = torch.sort(t_sorted)

            dt = torch.diff(t_sorted, dim=0,
                            prepend=torch.zeros(1, device=self.device,
                                                dtype=self.torch_type))

            gap_thresh = 3*torch.quantile(dt, 0.99)

            idx = torch.where(dt > gap_thresh)[0]

            if len(idx) > 0 and penalize_gap:
                h_tens = torch.tensor([total_h], device=self.device,
                                      dtype=self.torch_type)

            for i in idx:

                t1 = t_sorted[i-1] if i > 0 else 0
                t2 = t_sorted[i]
                anchor_t1 = anchor_exp[torch.argmin(torch.abs(anchor_t - t1)),
                                       :]
                anchor_t2 = anchor_exp[torch.argmin(torch.abs(anchor_t - t2)),
                                       :]
                if all_cells:
                    anchor_t1_list.append(torch.ravel(anchor_t1))
                    anchor_t2_list.append(torch.ravel(anchor_t2))
                if not all_cells:
                    for j in range(1, switch):
                        crit1 = ((t1 > t_sw_array[j-1])
                                 and (t2 > t_sw_array[j-1])
                                 and (t1 <= t_sw_array[j])
                                 and (t2 <= t_sw_array[j]))
                        crit2 = ((torch.abs(anchor_t1[2]
                                            - exp_sw_list[j][0, 2])
                                  < 0.02 * max_s) and
                                 (torch.abs(anchor_t2[2]
                                            - exp_sw_list[j][0, 2])
                                 < 0.01 * max_s))
                        crit3 = ((torch.abs(anchor_t1[1]
                                            - exp_sw_list[j][0, 1])
                                 < 0.02 * max_u) and
                                 (torch.abs(anchor_t2[1]
                                            - exp_sw_list[j][0, 1])
                                 < 0.01 * max_u))
                        crit4 = ((torch.abs(anchor_t1[0]
                                            - exp_sw_list[j][0, 0])
                                 < 0.02 * max_c) and
                                 (torch.abs(anchor_t2[0]
                                            - exp_sw_list[j][0, 0])
                                 < 0.01 * max_c))
                        if crit1 and crit2 and crit3 and crit4:
                            t_sw_adjust[j] += t2 - t1
                if penalize_gap:
                    dist_gap = torch.sum(((anchor_t1[1:] - anchor_t2[1:]) /
                                          scale_factor_ten[1:])**2)

                    idx_to_adjust = torch.tensor(t_pred >= t2,
                                                 device=self.device)

                    idx_to_adjust = torch.reshape(idx_to_adjust,
                                                  (1, idx_to_adjust.size()[0]))

                    true_tensor = torch.tensor([True], device=self.device)
                    false_tensor = torch.tensor([False], device=self.device)

                    t_sw_array_ = torch.cat((t_sw_array, h_tens), dim=0)
                    state_to_adjust = torch.where(t_sw_array_ > t2,
                                                  true_tensor, false_tensor)

                    dist_gap_add[idx_to_adjust] += dist_gap

                    if state_to_adjust[0].item():
                        dists0 += dist_gap_add
                    if state_to_adjust[1].item():
                        dists1 += dist_gap_add
                    if state_to_adjust[2].item():
                        dists2 += dist_gap_add
                    if state_to_adjust[3].item():
                        dists3 += dist_gap_add

                    dist_gap_add[idx_to_adjust] -= dist_gap

            dists = torch.cat((dists0, dists1, dists2, dists3), 0)

            state_pred = torch.argmin(dists, dim=0)

            if all_cells:
                t_pred = ts[torch.arange(n, device=self.device), state_pred]

        min_dist = torch.min(dists, dim=0).values

        if all_cells:
            exp_ss_mat = compute_ss_exp(alpha_c, alpha, beta, gamma,
                                        model=model)
            if rna_only:
                exp_ss_mat[:, 0] = 1
            dists_ss = pairwise_distance_square(exp_mat, exp_ss_mat *
                                                rescale_factor / scale_factor)

            reach_ss = np.full((n, 4), False)
            for i in range(n):
                for j in range(4):
                    if min_dist[i] > dists_ss[i, j]:
                        reach_ss[i, j] = True
            late_phase = np.full(n, -1)
            for i in range(3):
                late_phase[torch.abs(t_pred - t_sw_array[i]) < 0.1] = i

            return min_dist, t_pred, state_pred.cpu().detach().numpy(), \
                reach_ss, late_phase, max_u, max_s, anchor_t1_list, \
                anchor_t2_list

        else:
            return min_dist, state_pred.cpu().detach().numpy(), max_u, max_s, \
                   t_sw_adjust.cpu().detach().numpy()

    # the torch tensor version of the mse function
    def mse_ten(self, x, fit_outlier=False,
                penalize_gap=True):

        t3 = x[:3]
        r4 = x[3:7]
        scale_cc = x[7]
        rescale_c = x[8]
        rescale_u = x[9]

        if not self.known_pars:
            if self.fit_decoupling:
                t3 = torch.clip(t3, 0.1, None)
            else:
                t3[2] = 30 / self.n_anchors
                t3[:2] = torch.clip(t3[:2], 0.1, None)
            r4 = torch.clip(r4, 0.001, 1000)
            rescale_c = torch.clip(rescale_c, 0.75, 1.5)
            rescale_u = torch.clip(rescale_u, 0.2, 3)

        t_sw_array = torch.cumsum(t3, dim=0)

        if self.rna_only:
            t_sw_array[2] = 20

        # conditions for minimum switch time and rate params
        penalty = 0
        if any(t3 < 0.2) or any(r4 < 0.005):
            penalty = (torch.sum(0.2 - t3[t3 < 0.2]) if self.fit_decoupling
                       else torch.sum(0.2 - t3[:2][t3[:2] < 0.2]))
            penalty += torch.sum(0.005 - r4[r4 < 0.005]) * 1e2

        # condition for all params
        if any(x > 500):
            penalty = torch.sum(x[x > 500] - 500) * 1e-2

        c_array = self.c_all if fit_outlier else self.c
        u_array = self.u_all if fit_outlier else self.u
        s_array = self.s_all if fit_outlier else self.s

        if self.batch_size is not None and self.batch_size < len(c_array):

            subset_choice = np.random.choice(len(c_array), self.batch_size,
                                             replace=False)

            c_array = c_array[subset_choice]
            u_array = u_array[subset_choice]
            s_array = s_array[subset_choice]

            if fit_outlier:
                conn_for_calc = self.conn[subset_choice]
            if not fit_outlier:
                conn_for_calc = self.conn_sub[subset_choice]

            conn_for_calc = ((conn_for_calc.T)[subset_choice]).T

        else:

            if fit_outlier:
                conn_for_calc = self.conn
            if not fit_outlier:
                conn_for_calc = self.conn_sub

        scale_factor_func = np.array(self.scale_factor, dtype=self.u.dtype)

        # distances and time assignments
        res = self.calculate_dist_and_time_ten(c_array,
                                               u_array,
                                               s_array,
                                               t_sw_array,
                                               r4[0],
                                               r4[1],
                                               r4[2],
                                               r4[3],
                                               rescale_c,
                                               rescale_u,
                                               scale_cc=scale_cc,
                                               scale_factor=scale_factor_func,
                                               model=self.model,
                                               direction=self.direction,
                                               conn=conn_for_calc,
                                               k=self.k_dist,
                                               t=self.n_anchors,
                                               rna_only=self.rna_only,
                                               penalize_gap=penalize_gap,
                                               all_cells=fit_outlier)

        if fit_outlier:
            min_dist, t_pred, state_pred, reach_ss, late_phase, max_u, max_s, \
                self.anchor_t1_list, self.anchor_t2_list = res
        else:
            min_dist, state_pred, max_u, max_s, t_sw_adjust = res

        loss = torch.mean(min_dist)

        # avoid exceeding maximum expressions
        reg = torch.max(torch.tensor([0, max_s - torch.tensor(self.max_s)],
                                     requires_grad=True,
                                     dtype=self.torch_type))\
            + torch.max(torch.tensor([0, max_u - torch.tensor(self.max_u)],
                                     requires_grad=True,
                                     dtype=self.torch_type))

        loss += reg

        loss += 1e-1 * penalty

        self.cur_loss = loss.item()
        self.cur_state_pred = state_pred

        if fit_outlier:
            return loss, t_pred
        else:
            self.cur_t_sw_adjust = t_sw_adjust

        return loss

    def mse(self, x, fit_outlier=False, penalize_gap=True):
        x = np.array(x)

        t3, r4, scale_cc, rescale_c, rescale_u = self._variables(x)

        t_sw_array = np.array([t3[0], t3[0]+t3[1], t3[0]+t3[1]+t3[2]])
        if self.rna_only:
            t_sw_array[2] = 20

        # conditions for minimum switch time and rate params
        penalty = 0
        if any(t3 < 0.2) or any(r4 < 0.005):
            penalty = (np.sum(0.2 - t3[t3 < 0.2]) if self.fit_decoupling
                       else np.sum(0.2 - t3[:2][t3[:2] < 0.2]))
            penalty += np.sum(0.005 - r4[r4 < 0.005]) * 1e2

        # condition for all params
        if any(x > 500):
            penalty = np.sum(x[x > 500] - 500) * 1e-2

        c_array = self.c_all if fit_outlier else self.c
        u_array = self.u_all if fit_outlier else self.u
        s_array = self.s_all if fit_outlier else self.s

        if self.neural_net:

            res = calculate_dist_and_time_nn(c_array,
                                             u_array,
                                             s_array,
                                             self.max_u_all if fit_outlier
                                             else self.max_u,
                                             self.max_s_all if fit_outlier
                                             else self.max_s,
                                             t_sw_array,
                                             r4[0],
                                             r4[1],
                                             r4[2],
                                             r4[3],
                                             rescale_c,
                                             rescale_u,
                                             self.ode_model_0,
                                             self.ode_model_1,
                                             self.ode_model_2_m1,
                                             self.ode_model_2_m2,
                                             self.device,
                                             scale_cc=scale_cc,
                                             scale_factor=self.scale_factor,
                                             model=self.model,
                                             direction=self.direction,
                                             conn=self.conn if fit_outlier
                                             else self.conn_sub,
                                             k=self.k_dist,
                                             t=self.n_anchors,
                                             rna_only=self.rna_only,
                                             penalize_gap=penalize_gap,
                                             all_cells=fit_outlier)

            if fit_outlier:
                min_dist, t_pred, state_pred, max_u, max_s, nn_penalty = res
            else:
                min_dist, state_pred, max_u, max_s, nn_penalty = res

            penalty += nn_penalty

            t_sw_adjust = [0, 0, 0]

        else:

            # distances and time assignments
            res = calculate_dist_and_time(c_array,
                                          u_array,
                                          s_array,
                                          t_sw_array,
                                          r4[0],
                                          r4[1],
                                          r4[2],
                                          r4[3],
                                          rescale_c,
                                          rescale_u,
                                          scale_cc=scale_cc,
                                          scale_factor=self.scale_factor,
                                          model=self.model,
                                          direction=self.direction,
                                          conn=self.conn if fit_outlier
                                          else self.conn_sub,
                                          k=self.k_dist,
                                          t=self.n_anchors,
                                          rna_only=self.rna_only,
                                          penalize_gap=penalize_gap,
                                          all_cells=fit_outlier)

            if fit_outlier:
                min_dist, t_pred, state_pred, reach_ss, late_phase, max_u, \
                    max_s, self.anchor_t1_list, self.anchor_t2_list = res
            else:
                min_dist, state_pred, max_u, max_s, t_sw_adjust = res

        loss = np.mean(min_dist)

        # avoid exceeding maximum expressions
        reg = np.max([0, max_s - self.max_s]) + np.max([0, max_u - self.max_u])
        loss += reg

        loss += 1e-1 * penalty
        self.cur_loss = loss
        self.cur_state_pred = state_pred

        if fit_outlier:
            return loss, t_pred
        else:
            self.cur_t_sw_adjust = t_sw_adjust

        return loss

    def update(self, x, perform_update=False, initialize=False,
               fit_outlier=False, adjust_time=True, penalize_gap=True,
               plot=True):
        t3, r4, scale_cc, rescale_c, rescale_u = self._variables(x)
        t_sw_array = np.array([t3[0], t3[0]+t3[1], t3[0]+t3[1]+t3[2]])

        # read results
        if initialize:
            new_loss = self.mse(x, penalize_gap=penalize_gap)
        elif fit_outlier:
            new_loss, t_pred = self.mse(x, fit_outlier=True,
                                        penalize_gap=penalize_gap)
        else:
            new_loss = self.cur_loss
            t_sw_adjust = self.cur_t_sw_adjust
        state_pred = self.cur_state_pred

        if new_loss < self.loss[-1] or perform_update:
            perform_update = True

            self.loss.append(new_loss)
            self.alpha_c, self.alpha, self.beta, self.gamma = r4
            self.rates = r4
            self.scale_cc = scale_cc
            self.rescale_c = rescale_c
            self.rescale_u = rescale_u

            # adjust overcrowded anchors
            if not fit_outlier and adjust_time:
                t_sw_array -= np.cumsum(t_sw_adjust)
                if self.rna_only:
                    t_sw_array[2] = 20

            self.t_sw_1, self.t_sw_2, self.t_sw_3 = t_sw_array
            self.t_sw_array = t_sw_array
            self.params = np.array([self.t_sw_1,
                                    self.t_sw_2-self.t_sw_1,
                                    self.t_sw_3-self.t_sw_2,
                                    self.alpha_c,
                                    self.alpha,
                                    self.beta,
                                    self.gamma,
                                    self.scale_cc,
                                    self.rescale_c,
                                    self.rescale_u])
            if not initialize:
                self.state = state_pred
            if fit_outlier:
                self.t = t_pred

            #main_info(f'params updated as: {self.t_sw_array} {self.rates} '
            #            f'{self.scale_cc} {self.rescale_c} {self.rescale_u}',
            #            indent_level=2)

            # interactive plot
            if self.plot and plot:
                tau_list = anchor_points(self.t_sw_array, 20, self.n_anchors)
                switch = np.sum(self.t_sw_array < 20)
                typed_tau_list = List()
                [typed_tau_list.append(x) for x in tau_list]
                self.alpha_c, self.alpha, self.beta, self.gamma, \
                    self.c0, self.u0, self.s0 = \
                    check_params(self.alpha_c, self.alpha, self.beta,
                                 self.gamma, c0=self.c0, u0=self.u0,
                                 s0=self.s0)
                exp_list, exp_sw_list = generate_exp(typed_tau_list,
                                                     self.t_sw_array[:switch],
                                                     self.alpha_c,
                                                     self.alpha,
                                                     self.beta,
                                                     self.gamma,
                                                     scale_cc=self.scale_cc,
                                                     model=self.model,
                                                     rna_only=self.rna_only)
                rescale_factor = np.array([self.rescale_c,
                                           self.rescale_u,
                                           1.0])
                exp_list = [x*rescale_factor for x in exp_list]
                exp_sw_list = [x*rescale_factor for x in exp_sw_list]
                c = np.ravel(np.concatenate([exp_list[x][:, 0] for x in
                                             range(switch+1)]))
                u = np.ravel(np.concatenate([exp_list[x][:, 1] for x in
                                             range(switch+1)]))
                s = np.ravel(np.concatenate([exp_list[x][:, 2] for x in
                                             range(switch+1)]))
                c_ = self.c_all if fit_outlier else self.c
                u_ = self.u_all if fit_outlier else self.u
                s_ = self.s_all if fit_outlier else self.s
                self.ax.clear()
                plt.pause(0.1)
                if self.rna_only:
                    self.ax.scatter(s, u, s=self.point_size*1.5, c='black',
                                    alpha=0.6, zorder=2)
                    if switch >= 1:
                        c_sw1, u_sw1, s_sw1 = exp_sw_list[0][0]
                        self.ax.plot([s_sw1], [u_sw1], "om",
                                     markersize=self.point_size, zorder=5)
                    if switch >= 2:
                        c_sw2, u_sw2, s_sw2 = exp_sw_list[1][0]
                        self.ax.plot([s_sw2], [u_sw2], "Xm",
                                     markersize=self.point_size, zorder=5)
                    if switch == 3:
                        c_sw3, u_sw3, s_sw3 = exp_sw_list[2][0]
                        self.ax.plot([s_sw3], [u_sw3], "Dm",
                                     markersize=self.point_size, zorder=5)
                    if np.max(self.t) == 20:
                        self.ax.plot([s[-1]], [u[-1]], "*m",
                                     markersize=self.point_size, zorder=5)
                    for i in range(4):
                        if any(self.state == i):
                            self.ax.scatter(s_[(self.state == i)],
                                            u_[(self.state == i)],
                                            s=self.point_size, c=self.color[i])
                    self.ax.set_xlabel('s')
                    self.ax.set_ylabel('u')

                else:
                    self.ax.scatter(s, u, c, s=self.point_size*1.5,
                                    c='black', alpha=0.6, zorder=2)
                    if switch >= 1:
                        c_sw1, u_sw1, s_sw1 = exp_sw_list[0][0]
                        self.ax.plot([s_sw1], [u_sw1], [c_sw1], "om",
                                     markersize=self.point_size, zorder=5)
                    if switch >= 2:
                        c_sw2, u_sw2, s_sw2 = exp_sw_list[1][0]
                        self.ax.plot([s_sw2], [u_sw2], [c_sw2], "Xm",
                                     markersize=self.point_size, zorder=5)
                    if switch == 3:
                        c_sw3, u_sw3, s_sw3 = exp_sw_list[2][0]
                        self.ax.plot([s_sw3], [u_sw3], [c_sw3], "Dm",
                                     markersize=self.point_size, zorder=5)
                    if np.max(self.t) == 20:
                        self.ax.plot([s[-1]], [u[-1]], [c[-1]], "*m",
                                     markersize=self.point_size, zorder=5)
                    for i in range(4):
                        if any(self.state == i):
                            self.ax.scatter(s_[(self.state == i)],
                                            u_[(self.state == i)],
                                            c_[(self.state == i)],
                                            s=self.point_size, c=self.color[i])
                    self.ax.set_xlabel('s')
                    self.ax.set_ylabel('u')
                    self.ax.set_zlabel('c')
                self.fig.canvas.draw()
                plt.pause(0.1)
        return perform_update

    def save_dyn_plot(self, c, u, s, c_sw, u_sw, s_sw, tau_list,
                      show_all=False):
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
            #main_info(f'{self.plot_path} directory created.', indent_level=2)

        switch = np.sum(self.t_sw_array < 20)
        scale_back = np.array([self.scale_c, self.scale_u, self.scale_s])
        shift_back = np.array([self.offset_c, self.offset_u, self.offset_s])
        if switch >= 1:
            c_sw1, u_sw1, s_sw1 = c_sw[0], u_sw[0], s_sw[0]
        if switch >= 2:
            c_sw2, u_sw2, s_sw2 = c_sw[1], u_sw[1], s_sw[1]
        if switch == 3:
            c_sw3, u_sw3, s_sw3 = c_sw[2], u_sw[2], s_sw[2]

        if not show_all:
            n_anchors = len(u)
            t_lower = np.min(self.t)
            t_upper = np.max(self.t)
            t_ = np.concatenate((tau_list[0], tau_list[1] + self.t_sw_array[0],
                                 tau_list[2] + self.t_sw_array[1],
                                 tau_list[3] + self.t_sw_array[2]))
            c_pre = c[t_[:n_anchors] <= t_lower]
            u_pre = u[t_[:n_anchors] <= t_lower]
            s_pre = s[t_[:n_anchors] <= t_lower]
            c = c[(t_lower < t_[:n_anchors]) & (t_[:n_anchors] < t_upper)]
            u = u[(t_lower < t_[:n_anchors]) & (t_[:n_anchors] < t_upper)]
            s = s[(t_lower < t_[:n_anchors]) & (t_[:n_anchors] < t_upper)]

        c_all = self.c_all * self.scale_c + self.offset_c
        u_all = self.u_all * self.scale_u + self.offset_u
        s_all = self.s_all * self.scale_s + self.offset_s

        fig = plt.figure(figsize=self.fig_size)
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111, facecolor='white')
        if not show_all and len(u_pre) > 0:
            ax.scatter(s_pre, u_pre, s=self.point_size/2, c='black',
                       alpha=0.4, zorder=2)
        ax.scatter(s, u, s=self.point_size*1.5, c='black', alpha=0.6, zorder=2)
        for i in range(4):
            if any(self.state == i):
                ax.scatter(s_all[(self.state == i) & (self.non_outlier)],
                           u_all[(self.state == i) & (self.non_outlier)],
                           s=self.point_size, c=self.color[i])
        ax.scatter(s_all[~self.non_outlier], u_all[~self.non_outlier],
                   s=self.point_size/2, c='grey')
        if show_all or t_lower <= self.t_sw_array[0]:
            ax.plot([s_sw1], [u_sw1], "om", markersize=self.point_size,
                    zorder=5)
        if switch >= 2 and (show_all or (t_lower <= self.t_sw_array[1] and
                                         t_upper >= self.t_sw_array[1])):
            ax.plot([s_sw2], [u_sw2], "Xm", markersize=self.point_size,
                    zorder=5)
        if switch >= 3 and (show_all or (t_lower <= self.t_sw_array[2] and
                                         t_upper >= self.t_sw_array[2])):
            ax.plot([s_sw3], [u_sw3], "Dm", markersize=self.point_size,
                    zorder=5)
        if np.max(self.t) == 20:
            ax.plot([s[-1]], [u[-1]], "*m", markersize=self.point_size,
                    zorder=5)
        if (self.anchor_t1_list is not None and len(self.anchor_t1_list) > 0
                and show_all):
            for i in range(len(self.anchor_t1_list)):
                exp_t1 = self.anchor_t1_list[i] * scale_back + shift_back
                exp_t2 = self.anchor_t2_list[i] * scale_back + shift_back
                ax.plot([exp_t1[2]], [exp_t1[1]], "|y",
                        markersize=self.point_size*1.5)
                ax.plot([exp_t2[2]], [exp_t2[1]], "|c",
                        markersize=self.point_size*1.5)
        ax.plot(s_all,
                self.steady_state_func(self.s_all) * self.scale_u
                + self.offset_u, c='grey', ls=':', lw=self.point_size/4,
                alpha=0.7)
        ax.set_xlabel('s')
        ax.set_ylabel('u')
        ax.set_title(f'{self.gene}-{self.model}')
        plt.tight_layout()
        fig.savefig(f'{self.plot_path}/{self.gene}-{self.model}-us.png',
                    dpi=fig.dpi, facecolor=fig.get_facecolor(),
                    transparent=False, edgecolor='none')
        plt.close(fig)
        plt.pause(0.2)

        if self.extra_color is not None:
            fig = plt.figure(figsize=self.fig_size)
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111, facecolor='white')
            if not show_all and len(u_pre) > 0:
                ax.scatter(s_pre, u_pre, s=self.point_size/2, c='black',
                           alpha=0.4, zorder=2)
            ax.scatter(s, u, s=self.point_size*1.5, c='black', alpha=0.6,
                       zorder=2)
            ax.scatter(s_all, u_all, s=self.point_size, c=self.extra_color)
            if show_all or t_lower <= self.t_sw_array[0]:
                ax.plot([s_sw1], [u_sw1], "om", markersize=self.point_size,
                        zorder=5)
            if switch >= 2 and (show_all or (t_lower <= self.t_sw_array[1] and
                                             t_upper >= self.t_sw_array[1])):
                ax.plot([s_sw2], [u_sw2], "Xm", markersize=self.point_size,
                        zorder=5)
            if switch >= 3 and (show_all or (t_lower <= self.t_sw_array[2] and
                                             t_upper >= self.t_sw_array[2])):
                ax.plot([s_sw3], [u_sw3], "Dm", markersize=self.point_size,
                        zorder=5)
            if np.max(self.t) == 20:
                ax.plot([s[-1]], [u[-1]], "*m", markersize=self.point_size,
                        zorder=5)
            if (self.anchor_t1_list is not None and
                    len(self.anchor_t1_list) > 0 and show_all):
                for i in range(len(self.anchor_t1_list)):
                    exp_t1 = self.anchor_t1_list[i] * scale_back + shift_back
                    exp_t2 = self.anchor_t2_list[i] * scale_back + shift_back
                    ax.plot([exp_t1[2]], [exp_t1[1]], "|y",
                            markersize=self.point_size*1.5)
                    ax.plot([exp_t2[2]], [exp_t2[1]], "|c",
                            markersize=self.point_size*1.5)
            ax.plot(s_all, self.steady_state_func(self.s_all) * self.scale_u
                    + self.offset_u, c='grey', ls=':', lw=self.point_size/4,
                    alpha=0.7)
            ax.set_xlabel('s')
            ax.set_ylabel('u')
            ax.set_title(f'{self.gene}-{self.model}')
            plt.tight_layout()
            fig.savefig(f'{self.plot_path}/{self.gene}-{self.model}-'
                        'us_colorby_extra.png', dpi=fig.dpi,
                        facecolor=fig.get_facecolor(), transparent=False,
                        edgecolor='none')
            plt.close(fig)
            plt.pause(0.2)

            if not self.rna_only:
                fig = plt.figure(figsize=self.fig_size)
                fig.patch.set_facecolor('white')
                ax = fig.add_subplot(111, facecolor='white')
                if not show_all and len(u_pre) > 0:
                    ax.scatter(u_pre, c_pre, s=self.point_size/2, c='black',
                               alpha=0.4, zorder=2)
                ax.scatter(u, c, s=self.point_size*1.5, c='black', alpha=0.6,
                           zorder=2)
                ax.scatter(u_all, c_all, s=self.point_size, c=self.extra_color)
                if show_all or t_lower <= self.t_sw_array[0]:
                    ax.plot([u_sw1], [c_sw1], "om", markersize=self.point_size,
                            zorder=5)
                if switch >= 2 and (show_all or (t_lower <= self.t_sw_array[1]
                                                 and t_upper >=
                                                 self.t_sw_array[1])):
                    ax.plot([u_sw2], [c_sw2], "Xm", markersize=self.point_size,
                            zorder=5)
                if switch >= 3 and (show_all or (t_lower <= self.t_sw_array[2]
                                                 and t_upper >=
                                                 self.t_sw_array[2])):
                    ax.plot([u_sw3], [c_sw3], "Dm", markersize=self.point_size,
                            zorder=5)
                if np.max(self.t) == 20:
                    ax.plot([u[-1]], [c[-1]], "*m", markersize=self.point_size,
                            zorder=5)
                ax.set_xlabel('u')
                ax.set_ylabel('c')
                ax.set_title(f'{self.gene}-{self.model}')
                plt.tight_layout()
                fig.savefig(f'{self.plot_path}/{self.gene}-{self.model}-'
                            'cu_colorby_extra.png', dpi=fig.dpi,
                            facecolor=fig.get_facecolor(), transparent=False,
                            edgecolor='none')
                plt.close(fig)
                plt.pause(0.2)

        if not self.rna_only:
            fig = plt.figure(figsize=self.fig_size)
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111, projection='3d', facecolor='white')
            if not show_all and len(u_pre) > 0:
                ax.scatter(s_pre, u_pre, c_pre, s=self.point_size/2, c='black',
                           alpha=0.4, zorder=2)
            ax.scatter(s, u, c, s=self.point_size*1.5, c='black', alpha=0.6,
                       zorder=2)
            for i in range(4):
                if any(self.state == i):
                    ax.scatter(s_all[(self.state == i) & (self.non_outlier)],
                               u_all[(self.state == i) & (self.non_outlier)],
                               c_all[(self.state == i) & (self.non_outlier)],
                               s=self.point_size, c=self.color[i])
            ax.scatter(s_all[~self.non_outlier], u_all[~self.non_outlier],
                       c_all[~self.non_outlier], s=self.point_size/2, c='grey')
            if show_all or t_lower <= self.t_sw_array[0]:
                ax.plot([s_sw1], [u_sw1], [c_sw1], "om",
                        markersize=self.point_size, zorder=5)
            if switch >= 2 and (show_all or (t_lower <= self.t_sw_array[1] and
                                             t_upper >= self.t_sw_array[1])):
                ax.plot([s_sw2], [u_sw2], [c_sw2], "Xm",
                        markersize=self.point_size, zorder=5)
            if switch >= 3 and (show_all or (t_lower <= self.t_sw_array[2] and
                                             t_upper >= self.t_sw_array[2])):
                ax.plot([s_sw3], [u_sw3], [c_sw3], "Dm",
                        markersize=self.point_size, zorder=5)
            if np.max(self.t) == 20:
                ax.plot([s[-1]], [u[-1]], [c[-1]], "*m",
                        markersize=self.point_size, zorder=5)
            ax.set_xlabel('s')
            ax.set_ylabel('u')
            ax.set_zlabel('c')
            ax.set_title(f'{self.gene}-{self.model}')
            plt.tight_layout()
            fig.savefig(f'{self.plot_path}/{self.gene}-{self.model}-cus.png',
                        dpi=fig.dpi, facecolor=fig.get_facecolor(),
                        transparent=False, edgecolor='none')
            plt.close(fig)
            plt.pause(0.2)

            fig = plt.figure(figsize=self.fig_size)
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111, facecolor='white')
            if not show_all and len(u_pre) > 0:
                ax.scatter(s_pre, u_pre, s=self.point_size/2, c='black',
                           alpha=0.4, zorder=2)
            ax.scatter(s, u, s=self.point_size*1.5, c='black', alpha=0.6,
                       zorder=2)
            ax.scatter(s_all, u_all, s=self.point_size, c=np.log1p(self.c_all),
                       cmap='coolwarm')
            if show_all or t_lower <= self.t_sw_array[0]:
                ax.plot([s_sw1], [u_sw1], "om", markersize=self.point_size,
                        zorder=5)
            if switch >= 2 and (show_all or (t_lower <= self.t_sw_array[1] and
                                             t_upper >= self.t_sw_array[1])):
                ax.plot([s_sw2], [u_sw2], "Xm", markersize=self.point_size,
                        zorder=5)
            if switch >= 3 and (show_all or (t_lower <= self.t_sw_array[2] and
                                             t_upper >= self.t_sw_array[2])):
                ax.plot([s_sw3], [u_sw3], "Dm", markersize=self.point_size,
                        zorder=5)
            if np.max(self.t) == 20:
                ax.plot([s[-1]], [u[-1]], "*m", markersize=self.point_size,
                        zorder=5)
            ax.plot(s_all, self.steady_state_func(self.s_all) * self.scale_u +
                    self.offset_u, c='grey', ls=':', lw=self.point_size/4,
                    alpha=0.7)
            ax.set_xlabel('s')
            ax.set_ylabel('u')
            ax.set_title(f'{self.gene}-{self.model}')
            plt.tight_layout()
            fig.savefig(f'{self.plot_path}/{self.gene}-{self.model}-'
                        'us_colorby_c.png', dpi=fig.dpi,
                        facecolor=fig.get_facecolor(), transparent=False,
                        edgecolor='none')
            plt.close(fig)
            plt.pause(0.2)

            fig = plt.figure(figsize=self.fig_size)
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111, facecolor='white')
            if not show_all and len(u_pre) > 0:
                ax.scatter(u_pre, c_pre, s=self.point_size/2, c='black',
                           alpha=0.4, zorder=2)
            ax.scatter(u, c, s=self.point_size*1.5, c='black', alpha=0.6,
                       zorder=2)
            for i in range(4):
                if any(self.state == i):
                    ax.scatter(u_all[(self.state == i) & (self.non_outlier)],
                               c_all[(self.state == i) & (self.non_outlier)],
                               s=self.point_size, c=self.color[i])
            ax.scatter(u_all[~self.non_outlier], c_all[~self.non_outlier],
                       s=self.point_size/2, c='grey')
            if show_all or t_lower <= self.t_sw_array[0]:
                ax.plot([u_sw1], [c_sw1], "om", markersize=self.point_size,
                        zorder=5)
            if switch >= 2 and (show_all or (t_lower <= self.t_sw_array[1] and
                                             t_upper >= self.t_sw_array[1])):
                ax.plot([u_sw2], [c_sw2], "Xm", markersize=self.point_size,
                        zorder=5)
            if switch >= 3 and (show_all or (t_lower <= self.t_sw_array[2] and
                                             t_upper >= self.t_sw_array[2])):
                ax.plot([u_sw3], [c_sw3], "Dm", markersize=self.point_size,
                        zorder=5)
            if np.max(self.t) == 20:
                ax.plot([u[-1]], [c[-1]], "*m", markersize=self.point_size,
                        zorder=5)
            ax.set_xlabel('u')
            ax.set_ylabel('c')
            ax.set_title(f'{self.gene}-{self.model}')
            plt.tight_layout()
            fig.savefig(f'{self.plot_path}/{self.gene}-{self.model}-cu.png',
                        dpi=fig.dpi, facecolor=fig.get_facecolor(),
                        transparent=False, edgecolor='none')
            plt.close(fig)
            plt.pause(0.2)

    def get_loss(self):
        return self.loss

    def get_model(self):
        return self.model

    def get_params(self):
        return self.t_sw_array, self.rates, self.scale_cc, self.rescale_c, \
            self.rescale_u, self.realign_ratio

    def is_partial(self):
        return self.partial

    def get_direction(self):
        return self.direction

    def realign_time_and_velocity(self, c, u, s, anchor_time):
        # realign time to range (0,20)
        self.anchor_min_idx = np.sum(anchor_time < (np.min(self.t)-1e-5))
        self.anchor_max_idx = np.sum(anchor_time < (np.max(self.t)-1e-5))
        self.c0 = c[self.anchor_min_idx]
        self.u0 = u[self.anchor_min_idx]
        self.s0 = s[self.anchor_min_idx]
        self.realign_ratio = 20 / (np.max(self.t) - np.min(self.t))
        #main_info(f'fitted params:\nswitch time array = {self.t_sw_array},\n'
        #            f'rates = {self.rates},\ncc scale = {self.scale_cc},\n'
        #            f'c rescale factor = {self.rescale_c},\n'
        #            f'u rescale factor = {self.rescale_u}',
        #            indent_level=1)
        #main_info(f'aligning to range (0,20) by {self.realign_ratio}..',
         #           indent_level=1)
        self.rates /= self.realign_ratio
        self.alpha_c, self.alpha, self.beta, self.gamma = self.rates
        self.params[3:7] = self.rates
        self.t_sw_array = ((self.t_sw_array - np.min(self.t))
                           * self.realign_ratio)
        self.t_sw_1, self.t_sw_2, self.t_sw_3 = self.t_sw_array
        self.params[:3] = np.array([self.t_sw_1, self.t_sw_2 - self.t_sw_1,
                                    self.t_sw_3 - self.t_sw_2])
        self.t -= np.min(self.t)
        self.t = self.t * 20 / np.max(self.t)
        self.velocity /= self.realign_ratio
        self.velocity[:, 0] = np.clip(self.velocity[:, 0], -self.c_all
                                      * self.scale_c, None)
        self.velocity[:, 1] = np.clip(self.velocity[:, 1], -self.u_all
                                      * self.scale_u, None)
        self.velocity[:, 2] = np.clip(self.velocity[:, 2], -self.s_all
                                      * self.scale_s, None)
        self.anchor_velo /= self.realign_ratio
        self.anchor_velo[:, 0] = np.clip(self.anchor_velo[:, 0],
                                         -np.max(self.c_all * self.scale_c),
                                         None)
        self.anchor_velo[:, 1] = np.clip(self.anchor_velo[:, 1],
                                         -np.max(self.u_all * self.scale_u),
                                         None)
        self.anchor_velo[:, 2] = np.clip(self.anchor_velo[:, 2],
                                         -np.max(self.s_all * self.scale_s),
                                         None)

    def get_initial_exp(self):
        return np.array([self.c0, self.u0, self.s0])

    def get_time_assignment(self):
        if self.low_quality:
            return np.zeros(len(self.u_all))
        return self.t

    def get_state_assignment(self):
        if self.low_quality:
            return np.zeros(len(self.u_all))
        return self.state

    def get_velocity(self):
        if self.low_quality:
            return np.zeros((len(self.u_all), 3))
        return self.velocity

    def get_likelihood(self):
        return self.likelihood, self.l_c, self.ssd_c, self.var_c

    def get_anchors(self):
        if self.low_quality:
            return (np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)),
                    0, 0, 0, 0)
        return self.anchor_exp, self.anchor_exp_sw, self.anchor_velo, \
            self.anchor_min_idx, self.anchor_max_idx, \
            self.anchor_velo_min_idx, self.anchor_velo_max_idx


def regress_func(c, u, s, m, mi, im, dev, nn, ad, lr, b1, b2, bs, gpdist,
                 embed, conn, pl, sp, pdir, fa, gene, pa, di, ro, fit, fd,
                 extra, ru, alpha, beta, gamma, t_, verbosity, log_folder,
                 log_filename):

    settings.VERBOSITY = verbosity
    settings.LOG_FOLDER = log_folder
    settings.LOG_FILENAME = log_filename
    settings.GENE = gene

    #if m is not None:
        #main_info('#########################################################'
        #            '######################################', indent_level=1)
       # main_info(f'testing model {m}', indent_level=1)

    c_90 = np.percentile(c, 90)
    u_90 = np.percentile(u, 90)
    s_90 = np.percentile(s, 90)
    low_quality = (u_90 == 0 or s_90 == 0) if ro else (c_90 == 0 or u_90 == 0
                                                       or s_90 == 0)
    if low_quality:
        main_info(f'low quality gene {gene}, skipping', indent_level=1)
        return (np.inf, np.nan, '', (np.zeros(3), np.zeros(4), 0, 0, 0, 0),
                np.zeros(3), np.zeros(len(u)), np.zeros(len(u)),
                np.zeros((len(u), 3)), (-1.0, 0, 0, 0),
                (np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), 0, 0,
                 0, 0))

    if gpdist is not None:
        subset_cells = s > 0.1 * np.percentile(s, 99)
        subset_cells = np.where(subset_cells)[0]
        if len(subset_cells) > 3000:
            rng = np.random.default_rng(2021)
            subset_cells = rng.choice(subset_cells, 3000, replace=False)
        local_pdist = gpdist[np.ix_(subset_cells, subset_cells)]
        dists = (np.ravel(local_pdist[np.triu_indices_from(local_pdist, k=1)])
                 .reshape(-1, 1))
        local_std = np.std(dists)
    else:
        local_std = None

    cdc = ChromatinDynamical(c,
                             u,
                             s,
                             model=m,
                             max_iter=mi,
                             init_mode=im,
                             device=dev,
                             neural_net=nn,
                             adam=ad,
                             adam_lr=lr,
                             adam_beta1=b1,
                             adam_beta2=b2,
                             batch_size=bs,
                             local_std=local_std,
                             embed_coord=embed,
                             connectivities=conn,
                             plot=pl,
                             save_plot=sp,
                             plot_dir=pdir,
                             fit_args=fa,
                             gene=gene,
                             partial=pa,
                             direction=di,
                             rna_only=ro,
                             fit_decoupling=fd,
                             extra_color=extra,
                             rescale_u=ru,
                             alpha=alpha,
                             beta=beta,
                             gamma=gamma,
                             t_=t_)
    if fit:
        loss = cdc.fit()
        if loss[-1] == np.inf:
            main_info(f'low quality gene {gene}, skipping..', indent_level=1)
    loss = cdc.get_loss()
    model = cdc.get_model()
    direction = cdc.get_direction()
    parameters = cdc.get_params()
    initial_exp = cdc.get_initial_exp()
    velocity = cdc.get_velocity()
    likelihood = cdc.get_likelihood()
    time = cdc.get_time_assignment()
    state = cdc.get_state_assignment()
    anchors = cdc.get_anchors()
    return loss[-1], model, direction, parameters, initial_exp, time, state, \
        velocity, likelihood, anchors


def multimodel_helper(c, u, s,
                      model_to_run,
                      max_iter,
                      init_mode,
                      device,
                      neural_net,
                      adam,
                      adam_lr,
                      adam_beta1,
                      adam_beta2,
                      batch_size,
                      global_pdist,
                      embed_coord,
                      conn,
                      plot,
                      save_plot,
                      plot_dir,
                      fit_args,
                      gene,
                      partial,
                      direction,
                      rna_only,
                      fit,
                      fit_decoupling,
                      extra_color,
                      rescale_u,
                      alpha,
                      beta,
                      gamma,
                      t_,
                      verbosity, log_folder, log_filename):

    loss, param_cand, initial_cand, time_cand = [], [], [], []
    state_cand, velo_cand, likelihood_cand, anch_cand = [], [], [], []

    for model in model_to_run:
        (loss_m, _, direction_, parameters, initial_exp,
         time, state, velocity, likelihood, anchors) = \
         regress_func(c, u, s, model, max_iter, init_mode, device, neural_net,
                      adam, adam_lr, adam_beta1, adam_beta2, batch_size,
                      global_pdist, embed_coord, conn, plot, save_plot,
                      plot_dir, fit_args, gene, partial, direction, rna_only,
                      fit, fit_decoupling, extra_color, rescale_u, alpha, beta,
                      gamma, t_)
        loss.append(loss_m)
        param_cand.append(parameters)
        initial_cand.append(initial_exp)
        time_cand.append(time)
        state_cand.append(state)
        velo_cand.append(velocity)
        likelihood_cand.append(likelihood)
        anch_cand.append(anchors)

    best_model = np.argmin(loss)
    model = np.nan if rna_only else model_to_run[best_model]
    parameters = param_cand[best_model]
    initial_exp = initial_cand[best_model]
    time = time_cand[best_model]
    state = state_cand[best_model]
    velocity = velo_cand[best_model]
    likelihood = likelihood_cand[best_model]
    anchors = anch_cand[best_model]
    return loss, model, direction_, parameters, initial_exp, time, state, \
        velocity, likelihood, anchors


def recover_dynamics_chrom(adata_rna,
                           adata_atac=None,
                           gene_list=None,
                           max_iter=5,
                           init_mode='invert',
                           device="cpu",
                           neural_net=False,
                           adam=False,
                           adam_lr=None,
                           adam_beta1=None,
                           adam_beta2=None,
                           batch_size=None,
                           model_to_run=None,
                           plot=False,
                           parallel=True,
                           n_jobs=None,
                           save_plot=False,
                           plot_dir=None,
                           rna_only=False,
                           fit=True,
                           fit_decoupling=True,
                           extra_color_key=None,
                           embedding='X_umap',
                           n_anchors=500,
                           k_dist=1,
                           thresh_multiplier=1.0,
                           weight_c=0.6,
                           outlier=99.8,
                           n_pcs=30,
                           n_neighbors=30,
                           fig_size=(8, 6),
                           point_size=7,
                           partial=None,
                           direction=None,
                           rescale_u=None,
                           alpha=None,
                           beta=None,
                           gamma=None,
                           t_sw=None
                           ):

    """Multi-omic dynamics recovery.

    This function optimizes the joint chromatin and RNA model parameters in
    ODE solutions.

    Parameters
    ----------
    adata_rna: :class:`~anndata.AnnData`
        RNA anndata object. Required fields: `Mu`, `Ms`, and `connectivities`.
    adata_atac: :class:`~anndata.AnnData` (default: `None`)
        ATAC anndata object. Required fields: `Mc`.
    gene_list: `str`,  list of `str` (default: highly variable genes)
        Genes to use for model fitting.
    max_iter: `int` (default: `5`)
        Iterations to run for parameter optimization.
    init_mode: `str` (default: `'invert'`)
        Initialization method for switch times.
        `'invert'`: initial RNA switch time will be computed with scVelo time
        inversion method.
        `'grid'`: grid search the best set of switch times.
        `'simple'`: simply initialize switch times to be 5, 10, and 15.
    device: `str` (default: `'cpu'`)
        The CUDA device that pytorch tensor calculations will be run on. Only
        to be used with Adam or Neural Network mode.
    neural_net: `bool` (default: `False`)
        Whether to run time predictions with a neural network or not. Shortens
        runtime at the expense of accuracy. If False, uses the usual method of
        assigning each data point to an anchor time point as outlined in the
        Multivelo paper.
    adam: `bool` (default: `False`)
        Whether MSE minimization is handled by the Adam algorithm or not. When
        set to the default of False, function uses Nelder-Mead instead.
    adam_lr: `float` (default: `None`)
        The learning rate to use the Adam algorithm. If adam is False, this
        value is ignored.
    adam_beta1: `float` (default: `None`)
        The beta1 parameter for the Adam algorithm. If adam is False, this
        value is ignored.
    adam_beta2: `float` (default: `None`)
        The beta2 parameter for the Adam algorithm. If adam is False, this
        value is ignored.
    batch_size: `int` (default: `None`)
        Speeds up performance using minibatch training. Specifies number of
        cells to use per run of MSE when running the Adam algorithm. Ignored
        if Adam is set to False.
    model_to_run: `int` or list of `int` (default: `None`)
        User specified models for each genes. Possible values are 1 are 2. If
        `None`, the model
        for each gene will be inferred based on expression patterns. If more
        than one value is given,
        the best model will be decided based on loss of fit.
    plot: `bool` or `None` (default: `False`)
        Whether to interactively plot the 3D gene portraits. Ignored if
        parallel is True.
    parallel: `bool` (default: `True`)
        Whether to fit genes in a parallel fashion (recommended).
    n_jobs: `int` (default: available threads)
        Number of parallel jobs.
    save_plot: `bool` (default: `False`)
        Whether to save the fitted gene portrait figures as files. This will
        take some disk space.
    plot_dir: `str` (default: `plots` for multiome and `rna_plots` for
    RNA-only)
        Directory to save the plots.
    rna_only: `bool` (default: `False`)
        Whether to only use RNA for fitting (RNA velocity).
    fit: `bool` (default: `True`)
        Whether to fit the models. If False, only pre-determination and
        initialization will be run.
    fit_decoupling: `bool` (default: `True`)
        Whether to fit decoupling phase (Model 1 vs Model 2 distinction).
    n_anchors: `int` (default: 500)
        Number of anchor time-points to generate as a representation of the
        trajectory.
    k_dist: `int` (default: 1)
        Number of anchors to use to determine a cell's gene time. If more than
        1, time will be averaged.
    thresh_multiplier: `float` (default: 1.0)
        Multiplier for the heuristic threshold of partial versus complete
        trajectory pre-determination.
    weight_c: `float` (default: 0.6)
        Weighting of scaled chromatin distances when performing 3D residual
        calculation.
    outlier: `float` (default: 99.8)
        The percentile to mark as outlier that will be excluded when fitting
        the model.
    n_pcs: `int` (default: 30)
        Number of principal components to compute distance smoothing neighbors.
        This can be different from the one used for expression smoothing.
    n_neighbors: `int` (default: 30)
        Number of nearest neighbors for distance smoothing.
        This can be different from the one used for expression smoothing.
    fig_size: `tuple` (default: (8,6))
        Size of each figure when saved.
    point_size: `float` (default: 7)
        Marker point size for plotting.
    extra_color_key: `str` (default: `None`)
        Extra color key used for plotting. Common choices are `leiden`,
        `celltype`, etc.
        The colors for each category must be present in one of anndatas, which
        can be pre-computed
        with `scanpy.pl.scatter` function.
    embedding: `str` (default: `X_umap`)
        2D coordinates of the low-dimensional embedding of cells.
    partial: `bool` or list of `bool` (default: `None`)
        User specified trajectory completeness for each gene.
    direction: `str` or list of `str` (default: `None`)
        User specified trajectory directionality for each gene.
    rescale_u: `float` or list of `float` (default: `None`)
        Known scaling factors for unspliced. Can be computed from scVelo
        `fit_scaling` values
        as `rescale_u = fit_scaling / std(u) * std(s)`.
    alpha: `float` or list of `float` (default: `None`)
        Known trascription rates. Can be computed from scVelo `fit_alpha`
        values
        as `alpha = fit_alpha * fit_alignment_scaling`.
    beta: `float` or list of `float` (default: `None`)
        Known splicing rates. Can be computed from scVelo `fit_alpha` values
        as `beta = fit_beta * fit_alignment_scaling`.
    gamma: `float` or list of `float` (default: `None`)
        Known degradation rates. Can be computed from scVelo `fit_gamma` values
        as `gamma = fit_gamma * fit_alignment_scaling`.
    t_sw: `float` or list of `float` (default: `None`)
        Known RNA switch time. Can be computed from scVelo `fit_t_` values
        as `t_sw = fit_t_ / fit_alignment_scaling`.

    Returns
    -------
    fit_alpha_c, fit_alpha, fit_beta, fit_gamma: `.var`
        inferred chromatin opening, transcription, splicing, and degradation
        (nuclear export) rates
    fit_t_sw1, fit_t_sw2, fit_t_sw3: `.var`
        inferred switching time points
    fit_rescale_c, fit_rescale_u: `.var`
        inferred scaling factor for chromatin and unspliced counts
    fit_scale_cc: `.var`
        inferred scaling value for chromatin closing rate compared to opening
        rate
    fit_alignment_scaling: `.var`
        ratio used to realign observed time range to 0-20
    fit_c0, fit_u0, fit_s0: `.var`
        initial expression values at earliest observed time
    fit_model: `.var`
        inferred gene model
    fit_direction: `.var`
        inferred gene direction
    fit_loss: `.var`
        loss of model fit
    fit_likelihood: `.var`
        likelihood of model fit
    fit_likelihood_c: `.var`
        likelihood of chromatin fit
    fit_anchor_c, fit_anchor_u, fit_anchor_s: `.varm`
        anchor expressions
    fit_anchor_c_sw, fit_anchor_u_sw, fit_anchor_s_sw: `.varm`
        switch time-point expressions
    fit_anchor_c_velo, fit_anchor_u_velo, fit_anchor_s_velo: `.varm`
        velocities of anchors
    fit_anchor_min_idx: `.var`
        first anchor mapped to observations
    fit_anchor_max_idx: `.var`
        last anchor mapped to observations
    fit_anchor_velo_min_idx: `.var`
        first velocity anchor mapped to observations
    fit_anchor_velo_max_idx: `.var`
        last velocity anchor mapped to observations
    fit_t: `.layers`
        inferred gene time
    fit_state: `.layers`
        inferred state assignments
    velo_s, velo_u, velo_chrom: `.layers`
        velocities in spliced, unspliced, and chromatin space
    velo_s_genes, velo_u_genes, velo_chrom_genes: `.var`
        velocity genes
    velo_s_params, velo_u_params, velo_chrom_params: `.var`
        fitting arguments used
    ATAC: `.layers`
        KNN smoothed chromatin accessibilities copied from adata_atac
    """

    fit_args = {}
    fit_args['max_iter'] = max_iter
    fit_args['init_mode'] = init_mode
    fit_args['fit_decoupling'] = fit_decoupling
    n_anchors = np.clip(int(n_anchors), 201, 2000)
    fit_args['t'] = n_anchors
    fit_args['k'] = k_dist
    fit_args['thresh_multiplier'] = thresh_multiplier
    fit_args['weight_c'] = weight_c
    fit_args['outlier'] = outlier
    fit_args['n_pcs'] = n_pcs
    fit_args['n_neighbors'] = n_neighbors
    fit_args['fig_size'] = list(fig_size)
    fit_args['point_size'] = point_size

    if adam and neural_net:
        raise Exception("ADAM and Neural Net mode can not be run concurently."
                        " Please choose one to run on.")

    if not adam and not neural_net and not device == "cpu":
        raise Exception("Multivelo only uses non-CPU devices for Adam or"
                        " Neural Network mode. Please use one of those or"
                        "set the device to \"cpu\"")
    if device[0:5] == "cuda:":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    if adam and not device[0:5] == "cuda:":
        raise Exception("ADAM and Neural Net mode are only possible on a cuda "
                        "device. Please try again.")
    if not adam and batch_size is not None:
        raise Exception("Batch training is for ADAM only, please set "
                        "batch_size to None")

    if adam:
        from cuml.neighbors import NearestNeighbors

    all_genes = adata_rna.var_names
    if adata_atac is None:
        import anndata as ad
        rna_only = True
        adata_atac = ad.AnnData(X=np.ones(adata_rna.shape), obs=adata_rna.obs,
                                var=adata_rna.var)
        adata_atac.layers['Mc'] = np.ones(adata_rna.shape)
    if adata_rna.shape != adata_atac.shape:
        raise ValueError('Shape of RNA and ATAC adata objects do not match: '
                         f'{adata_rna.shape} {adata_atac.shape}')
    if not np.all(adata_rna.obs_names == adata_atac.obs_names):
        raise ValueError('obs_names of RNA and ATAC adata objects do not '
                         'match, please check if they are consistent')
    if not np.all(all_genes == adata_atac.var_names):
        raise ValueError('var_names of RNA and ATAC adata objects do not '
                         'match, please check if they are consistent')
    if 'connectivities' not in adata_rna.obsp.keys():
        raise ValueError('Missing connectivities entry in RNA adata object')
    if extra_color_key is None:
        extra_color = None
    elif (isinstance(extra_color_key, str) and extra_color_key in adata_rna.obs
            and adata_rna.obs[extra_color_key].dtype.name == 'category'):
        ngroups = len(adata_rna.obs[extra_color_key].cat.categories)
        extra_color = adata_rna.obs[extra_color_key].cat.rename_categories(
            adata_rna.uns[extra_color_key+'_colors'][:ngroups]).to_numpy()
    elif (isinstance(extra_color_key, str) and extra_color_key in
          adata_atac.obs and
          adata_rna.obs[extra_color_key].dtype.name == 'category'):
        ngroups = len(adata_atac.obs[extra_color_key].cat.categories)
        extra_color = adata_atac.obs[extra_color_key].cat.rename_categories(
            adata_atac.uns[extra_color_key+'_colors'][:ngroups]).to_numpy()
    else:
        raise ValueError('Currently, extra_color_key must be a single string '
                         'of categories and available in adata obs, and its '
                         'colors can be found in adata uns')
    if ('connectivities' not in adata_rna.obsp.keys() or
            (adata_rna.obsp['connectivities'] > 0).sum(1).min()
            > (n_neighbors-1)):
        from scanpy import Neighbors
        neighbors = Neighbors(adata_rna)
        neighbors.compute_neighbors(n_neighbors=n_neighbors, knn=True,
                                    n_pcs=n_pcs)
        rna_conn = neighbors.connectivities
    else:
        rna_conn = adata_rna.obsp['connectivities'].copy()
    rna_conn.setdiag(1)
    rna_conn = rna_conn.multiply(1.0 / rna_conn.sum(1)).tocsr()
    if not rna_only:
        if 'connectivities' not in adata_atac.obsp.keys():
            main_info('Missing connectivities in ATAC adata object, using '
                        'RNA connectivities instead', indent_level=1)
            atac_conn = rna_conn
        else:
            atac_conn = adata_atac.obsp['connectivities'].copy()
            atac_conn.setdiag(1)
        atac_conn = atac_conn.multiply(1.0 / atac_conn.sum(1)).tocsr()
    if gene_list is None:
        if 'highly_variable' in adata_rna.var:
            gene_list = adata_rna.var_names[adata_rna.var['highly_variable']]\
                .values
        else:
            gene_list = adata_rna.var_names.values[
                (~np.isnan(np.asarray(adata_rna.layers['Mu'].sum(0))
                             .reshape(-1)
                           if sparse.issparse(adata_rna.layers['Mu'])
                           else np.sum(adata_rna.layers['Mu'], axis=0)))
                & (~np.isnan(np.asarray(adata_rna.layers['Ms'].sum(0))
                             .reshape(-1)
                             if sparse.issparse(adata_rna.layers['Ms'])
                             else np.sum(adata_rna.layers['Ms'], axis=0)))
                & (~np.isnan(np.asarray(adata_atac.layers['Mc'].sum(0))
                             .reshape(-1)
                             if sparse.issparse(adata_atac.layers['Mc'])
                             else np.sum(adata_atac.layers['Mc'], axis=0)))]
    elif isinstance(gene_list, (list, np.ndarray, pd.Index, pd.Series)):
        gene_list = np.array([x for x in gene_list if x in all_genes])
    elif isinstance(gene_list, str):
        gene_list = np.array([gene_list]) if gene_list in all_genes else []
    else:
        raise ValueError('Invalid gene list, must be one of (str, np.ndarray,'
                         'pd.Index, pd.Series)')
    gn = len(gene_list)
    if gn == 0:
        raise ValueError('None of the genes specified are in the adata object')
    main_info(f'{gn} genes will be fitted', indent_level=1)

    models = np.zeros(gn)
    t_sws = np.zeros((gn, 3))
    rates = np.zeros((gn, 4))
    scale_ccs = np.zeros(gn)
    rescale_cs = np.zeros(gn)
    rescale_us = np.zeros(gn)
    realign_ratios = np.zeros(gn)
    initial_exps = np.zeros((gn, 3))
    times = np.zeros((adata_rna.n_obs, gn))
    states = np.zeros((adata_rna.n_obs, gn))
    if not rna_only:
        velo_c = np.zeros((adata_rna.n_obs, gn))
    velo_u = np.zeros((adata_rna.n_obs, gn))
    velo_s = np.zeros((adata_rna.n_obs, gn))
    likelihoods = np.zeros(gn)
    l_cs = np.zeros(gn)
    ssd_cs = np.zeros(gn)
    var_cs = np.zeros(gn)
    directions = []
    anchor_c = np.zeros((n_anchors, gn))
    anchor_u = np.zeros((n_anchors, gn))
    anchor_s = np.zeros((n_anchors, gn))
    anchor_c_sw = np.zeros((3, gn))
    anchor_u_sw = np.zeros((3, gn))
    anchor_s_sw = np.zeros((3, gn))
    anchor_vc = np.zeros((n_anchors, gn))
    anchor_vu = np.zeros((n_anchors, gn))
    anchor_vs = np.zeros((n_anchors, gn))
    anchor_min_idx = np.zeros(gn)
    anchor_max_idx = np.zeros(gn)
    anchor_velo_min_idx = np.zeros(gn)
    anchor_velo_max_idx = np.zeros(gn)

    if rna_only:
        model_to_run = [2]
        main_info('Skipping model checking for RNA-only, running model 2',
                    indent_level=1)

    m_per_g = False
    if model_to_run is not None:
        if isinstance(model_to_run, (list, np.ndarray, pd.Index, pd.Series)):
            model_to_run = [int(x) for x in model_to_run]
            if np.any(~np.isin(model_to_run, [0, 1, 2])):
                raise ValueError('Invalid model number (must be values in'
                                 ' [0,1,2])')
            if len(model_to_run) == gn:
                losses = np.zeros((gn, 1))
                m_per_g = True
                func_to_call = regress_func
            else:
                losses = np.zeros((gn, len(model_to_run)))
                func_to_call = multimodel_helper
        elif isinstance(model_to_run, (int, float)):
            model_to_run = int(model_to_run)
            if not np.isin(model_to_run, [0, 1, 2]):
                raise ValueError('Invalid model number (must be values in '
                                 '[0,1,2])')
            model_to_run = [model_to_run]
            losses = np.zeros((gn, 1))
            func_to_call = multimodel_helper
        else:
            raise ValueError('Invalid model number (must be values in '
                             '[0,1,2])')
    else:
        losses = np.zeros((gn, 1))
        func_to_call = regress_func

    p_per_g = False
    if partial is not None:
        if isinstance(partial, (list, np.ndarray, pd.Index, pd.Series)):
            if np.any(~np.isin(partial, [True, False])):
                raise ValueError('Invalid partial argument (must be values in'
                                 ' [True,False])')
            if len(partial) == gn:
                p_per_g = True
            else:
                raise ValueError('Incorrect partial argument length')
        elif isinstance(partial, bool):
            if not np.isin(partial, [True, False]):
                raise ValueError('Invalid partial argument (must be values in'
                                 ' [True,False])')
        else:
            raise ValueError('Invalid partial argument (must be values in'
                             ' [True,False])')

    d_per_g = False
    if direction is not None:
        if isinstance(direction, (list, np.ndarray, pd.Index, pd.Series)):
            if np.any(~np.isin(direction, ['on', 'off', 'complete'])):
                raise ValueError('Invalid direction argument (must be values'
                                 ' in ["on","off","complete"])')
            if len(direction) == gn:
                d_per_g = True
            else:
                raise ValueError('Incorrect direction argument length')
        elif isinstance(direction, str):
            if not np.isin(direction, ['on', 'off', 'complete']):
                raise ValueError('Invalid direction argument (must be values'
                                 ' in ["on","off","complete"])')
        else:
            raise ValueError('Invalid direction argument (must be values in'
                             ' ["on","off","complete"])')

    known_pars = [rescale_u, alpha, beta, gamma, t_sw]
    for x in known_pars:
        if x is not None:
            if isinstance(x, (list, np.ndarray)):
                if np.sum(np.isnan(x)) + np.sum(np.isinf(x)) > 0:
                    raise ValueError('Known parameters cannot contain NaN or'
                                     ' Inf')
            elif isinstance(x, (int, float)):
                if x == np.nan or x == np.inf:
                    raise ValueError('Known parameters cannot contain NaN or'
                                     ' Inf')
            else:
                raise ValueError('Invalid known parameters type')

    if ((embedding not in adata_rna.obsm) and
            (embedding not in adata_atac.obsm)):
        raise ValueError(f'{embedding} is not found in obsm')
    embed_coord = adata_rna.obsm[embedding] if embedding in adata_rna.obsm \
        else adata_atac.obsm[embedding]
    global_pdist = pairwise_distances(embed_coord)

    if sp.__version__ < '1.14.0':

        u_mat = adata_rna[:, gene_list].layers['Mu'].A \
            if sparse.issparse(adata_rna.layers['Mu']) \
            else adata_rna[:, gene_list].layers['Mu']
        s_mat = adata_rna[:, gene_list].layers['Ms'].A \
            if sparse.issparse(adata_rna.layers['Ms']) \
            else adata_rna[:, gene_list].layers['Ms']
        c_mat = adata_atac[:, gene_list].layers['Mc'].A \
            if sparse.issparse(adata_atac.layers['Mc']) \
            else adata_atac[:, gene_list].layers['Mc']
    else:
        u_mat = adata_rna[:, gene_list].layers['Mu'].toarray() \
            if sparse.issparse(adata_rna.layers['Mu']) \
            else adata_rna[:, gene_list].layers['Mu']
        s_mat = adata_rna[:, gene_list].layers['Ms'].toarray() \
            if sparse.issparse(adata_rna.layers['Ms']) \
            else adata_rna[:, gene_list].layers['Ms']
        c_mat = adata_atac[:, gene_list].layers['Mc'].toarray() \
            if sparse.issparse(adata_atac.layers['Mc']) \
            else adata_atac[:, gene_list].layers['Mc']

    ru = rescale_u if rescale_u is not None else None

    if parallel:
        if (n_jobs is None or not isinstance(n_jobs, int) or n_jobs < 0 or
                n_jobs > os.cpu_count()):
            n_jobs = os.cpu_count()
        if n_jobs > gn:
            n_jobs = gn
        batches = -(-gn // n_jobs)
        if n_jobs > 1:
            main_info(f'running {n_jobs} jobs in parallel', indent_level=1)
    else:
        n_jobs = 1
        batches = gn
    if n_jobs == 1:
        parallel = False

    pbar = tqdm(total=gn)
    for group in range(batches):
        gene_indices = range(group * n_jobs, np.min([gn, (group+1) * n_jobs]))
        if parallel:
            from joblib import Parallel, delayed
            verb = 51 if settings.VERBOSITY >= 2 else 0
            plot = False

            # clear the settings file if it exists
            open("settings.txt", "w").close()

            # write our current settings to the file
            with open("settings.txt", "a") as sfile:
                sfile.write(str(settings.VERBOSITY) + "\n")
                sfile.write(str(settings.CWD) + "\n")
                sfile.write(str(settings.LOG_FOLDER) + "\n")
                sfile.write(str(settings.LOG_FILENAME) + "\n")

            res = Parallel(n_jobs=n_jobs, backend='loky', verbose=verb)(
                delayed(func_to_call)(
                    c_mat[:, i],
                    u_mat[:, i],
                    s_mat[:, i],
                    model_to_run[i] if m_per_g else model_to_run,
                    max_iter,
                    init_mode,
                    device,
                    neural_net,
                    adam,
                    adam_lr,
                    adam_beta1,
                    adam_beta2,
                    batch_size,
                    global_pdist,
                    embed_coord,
                    rna_conn,
                    plot,
                    save_plot,
                    plot_dir,
                    fit_args,
                    gene_list[i],
                    partial[i] if p_per_g else partial,
                    direction[i] if d_per_g else direction,
                    rna_only,
                    fit,
                    fit_decoupling,
                    extra_color,
                    ru[i] if isinstance(ru, (list, np.ndarray)) else ru,
                    alpha[i] if isinstance(alpha, (list, np.ndarray))
                    else alpha,
                    beta[i] if isinstance(beta, (list, np.ndarray))
                    else beta,
                    gamma[i] if isinstance(gamma, (list, np.ndarray))
                    else gamma,
                    t_sw[i] if isinstance(t_sw, (list, np.ndarray)) else t_sw,
                    settings.VERBOSITY,
                    settings.LOG_FOLDER,
                    settings.LOG_FILENAME)
                for i in gene_indices)

            for i, r in zip(gene_indices, res):
                (loss, model, direct_out, parameters, initial_exp,
                 time, state, velocity, likelihood, anchors) = r
                switch, rate, scale_cc, rescale_c, rescale_u, realign_ratio = \
                    parameters
                likelihood, l_c, ssd_c, var_c = likelihood
                losses[i, :] = loss
                models[i] = model
                directions.append(direct_out)
                t_sws[i, :] = switch
                rates[i, :] = rate
                scale_ccs[i] = scale_cc
                rescale_cs[i] = rescale_c
                rescale_us[i] = rescale_u
                realign_ratios[i] = realign_ratio
                likelihoods[i] = likelihood
                l_cs[i] = l_c
                ssd_cs[i] = ssd_c
                var_cs[i] = var_c
                if fit:
                    initial_exps[i, :] = initial_exp
                    times[:, i] = time
                    states[:, i] = state
                    n_anchors_ = anchors[0].shape[0]
                    n_switch = anchors[1].shape[0]
                    if not rna_only:
                        velo_c[:, i] = smooth_scale(atac_conn, velocity[:, 0])
                        anchor_c[:n_anchors_, i] = anchors[0][:, 0]
                        anchor_c_sw[:n_switch, i] = anchors[1][:, 0]
                        anchor_vc[:n_anchors_, i] = anchors[2][:, 0]
                    velo_u[:, i] = smooth_scale(rna_conn, velocity[:, 1])
                    velo_s[:, i] = smooth_scale(rna_conn, velocity[:, 2])
                    anchor_u[:n_anchors_, i] = anchors[0][:, 1]
                    anchor_s[:n_anchors_, i] = anchors[0][:, 2]
                    anchor_u_sw[:n_switch, i] = anchors[1][:, 1]
                    anchor_s_sw[:n_switch, i] = anchors[1][:, 2]
                    anchor_vu[:n_anchors_, i] = anchors[2][:, 1]
                    anchor_vs[:n_anchors_, i] = anchors[2][:, 2]
                    anchor_min_idx[i] = anchors[3]
                    anchor_max_idx[i] = anchors[4]
                    anchor_velo_min_idx[i] = anchors[5]
                    anchor_velo_max_idx[i] = anchors[6]
        else:
            i = group
            gene = gene_list[i]
            main_info(f'@@@@@fitting {gene}', indent_level=1)
            (loss, model, direct_out,
             parameters, initial_exp,
             time, state, velocity,
             likelihood, anchors) = \
                func_to_call(c_mat[:, i], u_mat[:, i], s_mat[:, i],
                             model_to_run[i] if m_per_g else model_to_run,
                             max_iter, init_mode,
                             device,
                             neural_net,
                             adam,
                             adam_lr,
                             adam_beta1,
                             adam_beta2,
                             batch_size,
                             global_pdist, embed_coord,
                             rna_conn, plot, save_plot, plot_dir,
                             fit_args, gene,
                             partial[i] if p_per_g else partial,
                             direction[i] if d_per_g else direction,
                             rna_only, fit, fit_decoupling, extra_color,
                             ru[i] if isinstance(ru, (list, np.ndarray))
                             else ru,
                             alpha[i] if isinstance(alpha, (list, np.ndarray))
                             else alpha,
                             beta[i] if isinstance(beta, (list, np.ndarray))
                             else beta,
                             gamma[i] if isinstance(gamma, (list, np.ndarray))
                             else gamma,
                             t_sw[i] if isinstance(t_sw, (list, np.ndarray))
                             else t_sw,
                             settings.VERBOSITY,
                             settings.LOG_FOLDER,
                             settings.LOG_FILENAME)
            switch, rate, scale_cc, rescale_c, rescale_u, realign_ratio = \
                parameters
            likelihood, l_c, ssd_c, var_c = likelihood
            losses[i, :] = loss
            models[i] = model
            directions.append(direct_out)
            t_sws[i, :] = switch
            rates[i, :] = rate
            scale_ccs[i] = scale_cc
            rescale_cs[i] = rescale_c
            rescale_us[i] = rescale_u
            realign_ratios[i] = realign_ratio
            likelihoods[i] = likelihood
            l_cs[i] = l_c
            ssd_cs[i] = ssd_c
            var_cs[i] = var_c
            if fit:
                initial_exps[i, :] = initial_exp
                times[:, i] = time
                states[:, i] = state
                n_anchors_ = anchors[0].shape[0]
                n_switch = anchors[1].shape[0]
                if not rna_only:
                    velo_c[:, i] = smooth_scale(atac_conn, velocity[:, 0])
                    anchor_c[:n_anchors_, i] = anchors[0][:, 0]
                    anchor_c_sw[:n_switch, i] = anchors[1][:, 0]
                    anchor_vc[:n_anchors_, i] = anchors[2][:, 0]
                velo_u[:, i] = smooth_scale(rna_conn, velocity[:, 1])
                velo_s[:, i] = smooth_scale(rna_conn, velocity[:, 2])
                anchor_u[:n_anchors_, i] = anchors[0][:, 1]
                anchor_s[:n_anchors_, i] = anchors[0][:, 2]
                anchor_u_sw[:n_switch, i] = anchors[1][:, 1]
                anchor_s_sw[:n_switch, i] = anchors[1][:, 2]
                anchor_vu[:n_anchors_, i] = anchors[2][:, 1]
                anchor_vs[:n_anchors_, i] = anchors[2][:, 2]
                anchor_min_idx[i] = anchors[3]
                anchor_max_idx[i] = anchors[4]
                anchor_velo_min_idx[i] = anchors[5]
                anchor_velo_max_idx[i] = anchors[6]
        pbar.update(len(gene_indices))
    pbar.close()
    directions = np.array(directions)

    filt = np.sum(losses != np.inf, 1) >= 1
    if np.sum(filt) == 0:
        raise ValueError('None of the genes were fitted due to low quality,'
                         ' not returning')
    adata_copy = adata_rna[:, gene_list[filt]].copy()
    adata_copy.layers['ATAC'] = c_mat[:, filt]
    adata_copy.var['fit_alpha_c'] = rates[filt, 0]
    adata_copy.var['fit_alpha'] = rates[filt, 1]
    adata_copy.var['fit_beta'] = rates[filt, 2]
    adata_copy.var['fit_gamma'] = rates[filt, 3]
    adata_copy.var['fit_t_sw1'] = t_sws[filt, 0]
    adata_copy.var['fit_t_sw2'] = t_sws[filt, 1]
    adata_copy.var['fit_t_sw3'] = t_sws[filt, 2]
    adata_copy.var['fit_scale_cc'] = scale_ccs[filt]
    adata_copy.var['fit_rescale_c'] = rescale_cs[filt]
    adata_copy.var['fit_rescale_u'] = rescale_us[filt]
    adata_copy.var['fit_alignment_scaling'] = realign_ratios[filt]
    adata_copy.var['fit_model'] = models[filt]
    adata_copy.var['fit_direction'] = directions[filt]
    if model_to_run is not None and not m_per_g and not rna_only:
        for i, m in enumerate(model_to_run):
            adata_copy.var[f'fit_loss_M{m}'] = losses[filt, i]
    else:
        adata_copy.var['fit_loss'] = losses[filt, 0]
    adata_copy.var['fit_likelihood'] = likelihoods[filt]
    adata_copy.var['fit_likelihood_c'] = l_cs[filt]
    adata_copy.var['fit_ssd_c'] = ssd_cs[filt]
    adata_copy.var['fit_var_c'] = var_cs[filt]
    if fit:
        adata_copy.layers['fit_t'] = times[:, filt]
        adata_copy.layers['fit_state'] = states[:, filt]
        adata_copy.layers['velo_s'] = velo_s[:, filt]
        adata_copy.layers['velo_u'] = velo_u[:, filt]
        if not rna_only:
            adata_copy.layers['velo_chrom'] = velo_c[:, filt]
        adata_copy.var['fit_c0'] = initial_exps[filt, 0]
        adata_copy.var['fit_u0'] = initial_exps[filt, 1]
        adata_copy.var['fit_s0'] = initial_exps[filt, 2]
        adata_copy.var['fit_anchor_min_idx'] = anchor_min_idx[filt]
        adata_copy.var['fit_anchor_max_idx'] = anchor_max_idx[filt]
        adata_copy.var['fit_anchor_velo_min_idx'] = anchor_velo_min_idx[filt]
        adata_copy.var['fit_anchor_velo_max_idx'] = anchor_velo_max_idx[filt]
        adata_copy.varm['fit_anchor_c'] = np.transpose(anchor_c[:, filt])
        adata_copy.varm['fit_anchor_u'] = np.transpose(anchor_u[:, filt])
        adata_copy.varm['fit_anchor_s'] = np.transpose(anchor_s[:, filt])
        adata_copy.varm['fit_anchor_c_sw'] = np.transpose(anchor_c_sw[:, filt])
        adata_copy.varm['fit_anchor_u_sw'] = np.transpose(anchor_u_sw[:, filt])
        adata_copy.varm['fit_anchor_s_sw'] = np.transpose(anchor_s_sw[:, filt])
        adata_copy.varm['fit_anchor_c_velo'] = np.transpose(anchor_vc[:, filt])
        adata_copy.varm['fit_anchor_u_velo'] = np.transpose(anchor_vu[:, filt])
        adata_copy.varm['fit_anchor_s_velo'] = np.transpose(anchor_vs[:, filt])
    v_genes = adata_copy.var['fit_likelihood'] >= 0.05
    adata_copy.var['velo_s_genes'] = adata_copy.var['velo_u_genes'] = \
        adata_copy.var['velo_chrom_genes'] = v_genes
    adata_copy.uns['velo_s_params'] = adata_copy.uns['velo_u_params'] = \
        adata_copy.uns['velo_chrom_params'] = {'mode': 'dynamical'}
    adata_copy.uns['velo_s_params'].update(fit_args)
    adata_copy.uns['velo_u_params'].update(fit_args)
    adata_copy.uns['velo_chrom_params'].update(fit_args)
    adata_copy.obsp['_RNA_conn'] = rna_conn
    if not rna_only:
        adata_copy.obsp['_ATAC_conn'] = atac_conn
    return adata_copy


def smooth_scale(conn, vector):
    max_to = np.max(vector)
    min_to = np.min(vector)
    v = conn.dot(vector.T).T
    max_from = np.max(v)
    min_from = np.min(v)
    res = ((v - min_from) * (max_to - min_to) / (max_from - min_from)) + min_to
    return res


def top_n_sparse(conn, n):
    conn_ll = conn.tolil()
    for i in range(conn_ll.shape[0]):
        row_data = np.array(conn_ll.data[i])
        row_idx = np.array(conn_ll.rows[i])
        new_idx = row_data.argsort()[-n:]
        top_val = row_data[new_idx]
        top_idx = row_idx[new_idx]
        conn_ll.data[i] = top_val.tolist()
        conn_ll.rows[i] = top_idx.tolist()
    conn = conn_ll.tocsr()
    idx1 = conn > 0
    idx2 = conn > 0.25
    idx3 = conn > 0.5
    conn[idx1] = 0.25
    conn[idx2] = 0.5
    conn[idx3] = 1
    conn.eliminate_zeros()
    return conn


def set_velocity_genes(adata,
                       likelihood_lower=0.05,
                       rescale_u_upper=None,
                       rescale_u_lower=None,
                       rescale_c_upper=None,
                       rescale_c_lower=None,
                       primed_upper=None,
                       primed_lower=None,
                       decoupled_upper=None,
                       decoupled_lower=None,
                       alpha_c_upper=None,
                       alpha_c_lower=None,
                       alpha_upper=None,
                       alpha_lower=None,
                       beta_upper=None,
                       beta_lower=None,
                       gamma_upper=None,
                       gamma_lower=None,
                       scale_cc_upper=None,
                       scale_cc_lower=None
                       ):
    """Reset velocity genes.

    This function resets velocity genes based on criteria of variables.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        Anndata result from dynamics recovery.
    likelihood_lower: `float` (default: 0.05)
        Minimum ikelihood.
    rescale_u_upper: `float` (default: `None`)
        Maximum rescale_u.
    rescale_u_lower: `float` (default: `None`)
        Minimum rescale_u.
    rescale_c_upper: `float` (default: `None`)
        Maximum rescale_c.
    rescale_c_lower: `float` (default: `None`)
        Minimum rescale_c.
    primed_upper: `float` (default: `None`)
        Maximum primed interval.
    primed_lower: `float` (default: `None`)
        Minimum primed interval.
    decoupled_upper: `float` (default: `None`)
        Maximum decoupled interval.
    decoupled_lower: `float` (default: `None`)
        Minimum decoupled interval.
    alpha_c_upper: `float` (default: `None`)
        Maximum alpha_c.
    alpha_c_lower: `float` (default: `None`)
        Minimum alpha_c.
    alpha_upper: `float` (default: `None`)
        Maximum alpha.
    alpha_lower: `float` (default: `None`)
        Minimum alpha.
    beta_upper: `float` (default: `None`)
        Maximum beta.
    beta_lower: `float` (default: `None`)
        Minimum beta.
    gamma_upper: `float` (default: `None`)
        Maximum gamma.
    gamma_lower: `float` (default: `None`)
        Minimum gamma.
    scale_cc_upper: `float` (default: `None`)
        Maximum scale_cc.
    scale_cc_lower: `float` (default: `None`)
        Minimum scale_cc.

    Returns
    -------
    velo_s_genes, velo_u_genes, velo_chrom_genes: `.var`
        new velocity genes for each modalities.
    """

    v_genes = (adata.var['fit_likelihood'] >= likelihood_lower)
    if rescale_u_upper is not None:
        v_genes &= adata.var['fit_rescale_u'] <= rescale_u_upper
    if rescale_u_lower is not None:
        v_genes &= adata.var['fit_rescale_u'] >= rescale_u_lower
    if rescale_c_upper is not None:
        v_genes &= adata.var['fit_rescale_c'] <= rescale_c_upper
    if rescale_c_lower is not None:
        v_genes &= adata.var['fit_rescale_c'] >= rescale_c_lower
    t_sw1 = adata.var['fit_t_sw1'] + 20 / adata.uns['velo_s_params']['t'] * \
        adata.var['fit_anchor_min_idx'] * adata.var['fit_alignment_scaling']
    if primed_upper is not None:
        v_genes &= t_sw1 <= primed_upper
    if primed_lower is not None:
        v_genes &= t_sw1 >= primed_lower
    t_sw2 = np.clip(adata.var['fit_t_sw2'], None, 20)
    t_sw3 = np.clip(adata.var['fit_t_sw3'], None, 20)
    t_interval3 = t_sw3 - t_sw2
    if decoupled_upper is not None:
        v_genes &= t_interval3 <= decoupled_upper
    if decoupled_lower is not None:
        v_genes &= t_interval3 >= decoupled_lower
    if alpha_c_upper is not None:
        v_genes &= adata.var['fit_alpha_c'] <= alpha_c_upper
    if alpha_c_lower is not None:
        v_genes &= adata.var['fit_alpha_c'] >= alpha_c_lower
    if alpha_upper is not None:
        v_genes &= adata.var['fit_alpha'] <= alpha_upper
    if alpha_lower is not None:
        v_genes &= adata.var['fit_alpha'] >= alpha_lower
    if beta_upper is not None:
        v_genes &= adata.var['fit_beta'] <= beta_upper
    if beta_lower is not None:
        v_genes &= adata.var['fit_beta'] >= beta_lower
    if gamma_upper is not None:
        v_genes &= adata.var['fit_gamma'] <= gamma_upper
    if gamma_lower is not None:
        v_genes &= adata.var['fit_gamma'] >= gamma_lower
    if scale_cc_upper is not None:
        v_genes &= adata.var['fit_scale_cc'] <= scale_cc_upper
    if scale_cc_lower is not None:
        v_genes &= adata.var['fit_scale_cc'] >= scale_cc_lower
    main_info(f'{np.sum(v_genes)} velocity genes were selected', indent_level=1)
    adata.var['velo_s_genes'] = adata.var['velo_u_genes'] = \
        adata.var['velo_chrom_genes'] = v_genes


def velocity_graph(adata, vkey='velo_s', xkey='Ms', **kwargs):
    """Computes velocity graph.

    This function normalizes the velocity matrix and computes velocity graph
    with `scvelo.tl.velocity_graph`.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        Anndata result from dynamics recovery.
    vkey: `str` (default: `velo_s`)
        Default to use spliced velocities.
    xkey: `str` (default: `Ms`)
        Default to use smoothed spliced counts.
    Additional parameters passed to `scvelo.tl.velocity_graph`.

    Returns
    -------
    Normalized velocity matrix and associated velocity genes and params.
    Outputs of `scvelo.tl.velocity_graph`.
    """
    import scvelo as scv
    if vkey not in adata.layers.keys():
        raise ValueError('Velocity matrix is not found. Please run multivelo'
                         '.recover_dynamics_chrom function first.')
    if vkey+'_norm' not in adata.layers.keys():
        adata.layers[vkey+'_norm'] = adata.layers[vkey] / np.sum(
            np.abs(adata.layers[vkey]), 0)
        adata.layers[vkey+'_norm'] /= np.mean(adata.layers[vkey+'_norm'])
        adata.uns[vkey+'_norm_params'] = adata.uns[vkey+'_params']
    if vkey+'_norm_genes' not in adata.var.columns:
        adata.var[vkey+'_norm_genes'] = adata.var[vkey+'_genes']
    scv.tl.velocity_graph(adata, vkey=vkey+'_norm', xkey=xkey, **kwargs)


def velocity_embedding_stream(adata, vkey='velo_s', show=True, **kwargs):
    """Plots velocity stream.

    This function plots velocity streamplot with
    `scvelo.pl.velocity_embedding_stream`.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        Anndata result from dynamics recovery.
    vkey: `str` (default: `velo_s`)
        Default to use spliced velocities. The normalized matrix will be used.
    show: `bool` (default: `True`)
        Whether to show the plot.
    Additional parameters passed to `scvelo.tl.velocity_graph`.

    Returns
    -------
    If `show==False`, a matplotlib axis object.
    """
    import scvelo as scv
    if vkey not in adata.layers:
        raise ValueError('Velocity matrix is not found. Please run multivelo.'
                         'recover_dynamics_chrom function first.')
    if vkey+'_norm' not in adata.layers.keys():
        adata.layers[vkey+'_norm'] = adata.layers[vkey] / np.sum(
            np.abs(adata.layers[vkey]), 0)
        adata.uns[vkey+'_norm_params'] = adata.uns[vkey+'_params']
    if vkey+'_norm_genes' not in adata.var.columns:
        adata.var[vkey+'_norm_genes'] = adata.var[vkey+'_genes']
    if vkey+'_norm_graph' not in adata.uns.keys():
        velocity_graph(adata, vkey=vkey, **kwargs)
    out = scv.pl.velocity_embedding_stream(adata, vkey=vkey+'_norm', show=show,
                                           **kwargs)
    if not show:
        return out


def latent_time(adata, vkey='velo_s', **kwargs):
    """Computes latent time.

    This function computes latent time with `scvelo.tl.latent_time`.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        Anndata result from dynamics recovery.
    vkey: `str` (default: `velo_s`)
        Default to use spliced velocities. The normalized matrix will be used.
    Additional parameters passed to `scvelo.tl.velocity_graph`.

    Returns
    -------
    Outputs of `scvelo.tl.latent_time`.
    """
    import scvelo as scv
    if vkey not in adata.layers.keys() or 'fit_t' not in adata.layers.keys():
        raise ValueError('Velocity or time matrix is not found. Please run '
                         'multivelo.recover_dynamics_chrom function first.')
    if vkey+'_norm' not in adata.layers.keys():
        raise ValueError('Normalized velocity matrix is not found. Please '
                         'run multivelo.velocity_graph function first.')
    if vkey+'_norm_graph' not in adata.uns.keys():
        velocity_graph(adata, vkey=vkey, **kwargs)
    scv.tl.latent_time(adata, vkey=vkey+'_norm', **kwargs)


def LRT_decoupling(adata_rna, adata_atac, **kwargs):
    """Computes likelihood ratio test for decoupling state.

    This function computes whether keeping decoupling state improves fit
    Likelihood.

    Parameters
    ----------
    adata_rna: :class:`~anndata.AnnData`
        RNA anndata object
    adata_atac: :class:`~anndata.AnnData`
        ATAC anndata object.
    Additional parameters passed to `recover_dynamics_chrom`.

    Returns
    -------
    adata_result_w_decoupled: class:`~anndata.AnnData`
        fit result with decoupling state
    adata_result_w_decoupled: class:`~anndata.AnnData`
        fit result without decoupling state
    res: `pandas.DataFrame`
        LRT statistics
    """
    from scipy.stats.distributions import chi2
    main_info('fitting models with decoupling intervals', v=0)
    adata_result_w_decoupled = recover_dynamics_chrom(adata_rna, adata_atac,
                                                      fit_decoupling=True,
                                                      **kwargs)
    main_info('fitting models without decoupling intervals', v=0)
    adata_result_wo_decoupled = recover_dynamics_chrom(adata_rna, adata_atac,
                                                       fit_decoupling=False,
                                                       **kwargs)
    main_info('testing likelihood ratio', v=0)
    shared_genes = pd.Index(np.intersect1d(adata_result_w_decoupled.var_names,
                                           adata_result_wo_decoupled.var_names)
                            )
    l_c_w_decoupled = adata_result_w_decoupled[:, shared_genes].\
        var['fit_likelihood_c'].values
    l_c_wo_decoupled = adata_result_wo_decoupled[:, shared_genes].\
        var['fit_likelihood_c'].values
    n_obs = adata_rna.n_obs
    LRT_c = -2 * n_obs * (np.log(l_c_wo_decoupled) - np.log(l_c_w_decoupled))
    p_c = chi2.sf(LRT_c, 1)
    l_w_decoupled = adata_result_w_decoupled[:, shared_genes].\
        var['fit_likelihood'].values
    l_wo_decoupled = adata_result_wo_decoupled[:, shared_genes].\
        var['fit_likelihood'].values
    LRT = -2 * n_obs * (np.log(l_wo_decoupled) - np.log(l_w_decoupled))
    p = chi2.sf(LRT, 1)
    res = pd.DataFrame({'likelihood_c_w_decoupled': l_c_w_decoupled,
                        'likelihood_c_wo_decoupled': l_c_wo_decoupled,
                        'LRT_c': LRT_c,
                        'pval_c': p_c,
                        'likelihood_w_decoupled': l_w_decoupled,
                        'likelihood_wo_decoupled': l_wo_decoupled,
                        'LRT': LRT,
                        'pval': p,
                        }, index=shared_genes)
    return adata_result_w_decoupled, adata_result_wo_decoupled, res


def transition_matrix_s(s_mat, velo_s, knn):
    if sp.__version__ < '1.14.0':
        knn = knn.astype(int)
        tm_val, tm_col, tm_row = [], [], []
        for i in range(knn.shape[0]):
            two_step_knn = knn[i, :]
            for j in knn[i, :]:
                two_step_knn = np.append(two_step_knn, knn[j, :])
            two_step_knn = np.unique(two_step_knn)
            for j in two_step_knn:
                s = s_mat[i, :]
                sn = s_mat[j, :]
                ds = s - sn
                dx = np.ravel(ds.A)
                velo = velo_s[i, :]
                cos_sim = np.dot(dx, velo)/(norm(dx)*norm(velo))
                tm_val.append(cos_sim)
                tm_col.append(j)
                tm_row.append(i)
    else:
        knn = knn.astype(int)
        tm_val, tm_col, tm_row = [], [], []
        for i in range(knn.shape[0]):
            two_step_knn = knn[i, :]
            for j in knn[i, :]:
                two_step_knn = np.append(two_step_knn, knn[j, :])
            two_step_knn = np.unique(two_step_knn)
            for j in two_step_knn:
                s = s_mat[i, :]
                sn = s_mat[j, :]
                ds = s - sn
                dx = np.ravel(ds.toarray())
                velo = velo_s[i, :]
                cos_sim = np.dot(dx, velo)/(norm(dx)*norm(velo))
                tm_val.append(cos_sim)
                tm_col.append(j)
                tm_row.append(i)
    tm = coo_matrix((tm_val, (tm_row, tm_col)), shape=(s_mat.shape[0],
                    s_mat.shape[0])).tocsr()
    tm.setdiag(0)
    tm_neg = tm.copy()
    tm.data = np.clip(tm.data, 0, 1)
    tm_neg.data = np.clip(tm_neg.data, -1, 0)
    tm.eliminate_zeros()
    tm_neg.eliminate_zeros()
    return tm, tm_neg


def transition_matrix_chrom(c_mat, u_mat, s_mat, velo_c, velo_u, velo_s, knn):
    if sp.__version__ < '1.14.0':
        knn = knn.astype(int)
        tm_val, tm_col, tm_row = [], [], []
        for i in range(knn.shape[0]):
            two_step_knn = knn[i, :]
            for j in knn[i, :]:
                two_step_knn = np.append(two_step_knn, knn[j, :])
            two_step_knn = np.unique(two_step_knn)
            for j in two_step_knn:
                u = u_mat[i, :].A
                s = s_mat[i, :].A
                c = c_mat[i, :].A
                un = u_mat[j, :]
                sn = s_mat[j, :]
                cn = c_mat[j, :]
                dc = (c - cn) / np.std(c)
                du = (u - un) / np.std(u)
                ds = (s - sn) / np.std(s)
                dx = np.ravel(np.hstack((dc.A, du.A, ds.A)))
                velo = np.hstack((velo_c[i, :], velo_u[i, :], velo_s[i, :]))
                cos_sim = np.dot(dx, velo)/(norm(dx)*norm(velo))
                tm_val.append(cos_sim)
                tm_col.append(j)
                tm_row.append(i)
    else:
        knn = knn.astype(int)
        tm_val, tm_col, tm_row = [], [], []
        for i in range(knn.shape[0]):
            two_step_knn = knn[i, :]
            for j in knn[i, :]:
                two_step_knn = np.append(two_step_knn, knn[j, :])
            two_step_knn = np.unique(two_step_knn)
            for j in two_step_knn:
                u = u_mat[i, :].toarray()
                s = s_mat[i, :].toarray()
                c = c_mat[i, :].toarray()
                un = u_mat[j, :]
                sn = s_mat[j, :]
                cn = c_mat[j, :]
                dc = (c - cn) / np.std(c)
                du = (u - un) / np.std(u)
                ds = (s - sn) / np.std(s)
                dx = np.ravel(np.hstack((dc.toarray(), du.toarray(), ds.toarray())))
                velo = np.hstack((velo_c[i, :], velo_u[i, :], velo_s[i, :]))
                cos_sim = np.dot(dx, velo)/(norm(dx)*norm(velo))
                tm_val.append(cos_sim)
                tm_col.append(j)
                tm_row.append(i)
    tm = coo_matrix((tm_val, (tm_row, tm_col)), shape=(c_mat.shape[0],
                    c_mat.shape[0])).tocsr()
    tm.setdiag(0)
    tm_neg = tm.copy()
    tm.data = np.clip(tm.data, 0, 1)
    tm_neg.data = np.clip(tm_neg.data, -1, 0)
    tm.eliminate_zeros()
    tm_neg.eliminate_zeros()
    return tm, tm_neg


def likelihood_plot(adata,
                    genes=None,
                    figsize=(14, 10),
                    bins=50,
                    pointsize=4
                    ):
    """Likelihood plots.

    This function plots likelihood and variable distributions.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        Anndata result from dynamics recovery.
    genes: `str`,  list of `str` (default: `None`)
        If `None`, will use all fitted genes.
    figsize: `tuple` (default: (14,10))
        Figure size.
    bins: `int` (default: 50)
        Number of bins for histograms.
    pointsize: `float` (default: 4)
        Point size for scatter plots.
    """
    if genes is None:
        var = adata.var
    else:
        genes = np.array(genes)
        var = adata[:, genes].var
    likelihood = var[['fit_likelihood']].values
    rescale_u = var[['fit_rescale_u']].values
    rescale_c = var[['fit_rescale_c']].values
    t_interval1 = var['fit_t_sw1'] + 20 / adata.uns['velo_s_params']['t'] \
        * var['fit_anchor_min_idx'] * var['fit_alignment_scaling']
    t_sw2 = np.clip(var['fit_t_sw2'], None, 20)
    t_sw3 = np.clip(var['fit_t_sw3'], None, 20)
    t_interval3 = t_sw3 - t_sw2
    log_s = np.log1p(np.sum(adata.layers['Ms'], axis=0))
    alpha_c = var[['fit_alpha_c']].values
    alpha = var[['fit_alpha']].values
    beta = var[['fit_beta']].values
    gamma = var[['fit_gamma']].values
    scale_cc = var[['fit_scale_cc']].values

    fig, axes = plt.subplots(4, 5, figsize=figsize)
    axes[0, 0].hist(likelihood, bins=bins)
    axes[0, 0].set_title('likelihood')
    axes[0, 1].hist(rescale_u, bins=bins)
    axes[0, 1].set_title('rescale u')
    axes[0, 2].hist(rescale_c, bins=bins)
    axes[0, 2].set_title('rescale c')
    axes[0, 3].hist(t_interval1.values, bins=bins)
    axes[0, 3].set_title('primed interval')
    axes[0, 4].hist(t_interval3, bins=bins)
    axes[0, 4].set_title('decoupled interval')

    axes[1, 0].scatter(log_s, likelihood, s=pointsize)
    axes[1, 0].set_xlabel('log spliced')
    axes[1, 0].set_ylabel('likelihood')
    axes[1, 1].scatter(rescale_u, likelihood, s=pointsize)
    axes[1, 1].set_xlabel('rescale u')
    axes[1, 2].scatter(rescale_c, likelihood, s=pointsize)
    axes[1, 2].set_xlabel('rescale c')
    axes[1, 3].scatter(t_interval1.values, likelihood, s=pointsize)
    axes[1, 3].set_xlabel('primed interval')
    axes[1, 4].scatter(t_interval3, likelihood, s=pointsize)
    axes[1, 4].set_xlabel('decoupled interval')

    axes[2, 0].hist(alpha_c, bins=bins)
    axes[2, 0].set_title('alpha c')
    axes[2, 1].hist(alpha, bins=bins)
    axes[2, 1].set_title('alpha')
    axes[2, 2].hist(beta, bins=bins)
    axes[2, 2].set_title('beta')
    axes[2, 3].hist(gamma, bins=bins)
    axes[2, 3].set_title('gamma')
    axes[2, 4].hist(scale_cc, bins=bins)
    axes[2, 4].set_title('scale cc')

    axes[3, 0].scatter(alpha_c, likelihood, s=pointsize)
    axes[3, 0].set_xlabel('alpha c')
    axes[3, 0].set_ylabel('likelihood')
    axes[3, 1].scatter(alpha, likelihood, s=pointsize)
    axes[3, 1].set_xlabel('alpha')
    axes[3, 2].scatter(beta, likelihood, s=pointsize)
    axes[3, 2].set_xlabel('beta')
    axes[3, 3].scatter(gamma, likelihood, s=pointsize)
    axes[3, 3].set_xlabel('gamma')
    axes[3, 4].scatter(scale_cc, likelihood, s=pointsize)
    axes[3, 4].set_xlabel('scale cc')
    fig.tight_layout()


def pie_summary(adata, genes=None):
    """Summary of directions and models.

    This function plots a pie chart for (pre-determined or specified)
    directions and models.
    `induction`: induction-only genes.
    `repression`: repression-only genes.
    `Model 1`: model 1 complete genes.
    `Model 2`: model 2 complete genes.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        Anndata result from dynamics recovery.
    genes: `str`,  list of `str` (default: `None`)
        If `None`, will use all fitted genes.
    """
    if genes is None:
        genes = adata.var_names
    fit_model = adata[:, (adata.var['fit_direction'] == 'complete') &
                      np.isin(adata.var_names, genes)].var['fit_model'].values
    fit_direction = adata[:, genes].var['fit_direction'].values
    data = [np.sum(fit_direction == 'on'), np.sum(fit_direction == 'off'),
            np.sum(fit_model == 1), np.sum(fit_model == 2)]
    index = ['induction', 'repression', 'Model 1', 'Model 2']
    index = [x for i, x in enumerate(index) if data[i] > 0]
    data = [x for x in data if x > 0]
    df = pd.DataFrame({'data': data}, index=index)
    df.plot.pie(y='data', autopct='%1.1f%%', legend=False, startangle=30,
                ylabel='')
    circle = plt.Circle((0, 0), 0.8, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(circle)


def switch_time_summary(adata, genes=None):
    """Summary of switch times.

    This function plots a box plot for observed switch times.
    `primed`: primed intervals.
    `coupled-on`: coupled induction intervals.
    `decoupled`: decoupled intervals.
    `coupled-off`: coupled repression intervals.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        Anndata result from dynamics recovery.
    genes: `str`,  list of `str` (default: `None`)
        If `None`, will use velocity genes.
    """
    t_sw = adata[:, adata.var['velo_s_genes']
                 if genes is None
                 else genes] \
        .var[['fit_t_sw1', 'fit_t_sw2', 'fit_t_sw3']].copy()
    t_sw = t_sw.mask(t_sw > 20, 20)
    t_sw = t_sw.mask(t_sw < 0)
    t_sw['interval 1'] = t_sw['fit_t_sw1']
    t_sw['t_sw2 - t_sw1'] = t_sw['fit_t_sw2'] - t_sw['fit_t_sw1']
    t_sw['t_sw3 - t_sw2'] = t_sw['fit_t_sw3'] - t_sw['fit_t_sw2']
    t_sw['20 - t_sw3'] = 20 - t_sw['fit_t_sw3']
    t_sw = t_sw.mask(t_sw <= 0)
    t_sw = t_sw.mask(t_sw > 20)
    t_sw.columns = pd.Index(['time 1', 'time 2', 'time 3', 'primed',
                             'coupled-on', 'decoupled', 'coupled-off'])
    t_sw = t_sw[['primed', 'coupled-on', 'decoupled', 'coupled-off']]
    t_sw = t_sw / 20
    fig, ax = plt.subplots(figsize=(4, 5))
    ax = sns.boxplot(data=t_sw, width=0.5, palette='Set2', ax=ax)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_title('Switch Intervals')


def dynamic_plot(adata,
                 genes,
                 by='expression',
                 color_by='state',
                 gene_time=True,
                 axis_on=True,
                 frame_on=True,
                 show_anchors=True,
                 show_switches=True,
                 downsample=1,
                 full_range=False,
                 figsize=None,
                 pointsize=2,
                 linewidth=1.5,
                 cmap='coolwarm'
                 ):
    """Gene dynamics plot.

    This function plots accessibility, expression, or velocity by time.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        Anndata result from dynamics recovery.
    genes: `str`,  list of `str`
        List of genes to plot.
    by: `str` (default: `expression`)
        Plot accessibilities and expressions if `expression`. Plot velocities
        if `velocity`.
    color_by: `str` (default: `state`)
        Color by the four potential states if `state`. Other common values are
        leiden, louvain, celltype, etc.
        If not `state`, the color field must be present in `.uns`, which can
        be pre-computed with `scanpy.pl.scatter`.
        For `state`, red, orange, green, and blue represent state 1, 2, 3, and
        4, respectively.
    gene_time: `bool` (default: `True`)
        Whether to use individual gene fitted time, or shared global latent
        time.
        Mean values of 20 equal sized windows will be connected and shown if
        `gene_time==False`.
    axis_on: `bool` (default: `True`)
        Whether to show axis labels.
    frame_on: `bool` (default: `True`)
        Whether to show plot frames.
    show_anchors: `bool` (default: `True`)
        Whether to display anchors.
    show_switches: `bool` (default: `True`)
        Whether to show switch times. The switch times are indicated by
        vertical dotted line.
    downsample: `int` (default: 1)
        How much to downsample the cells. The remaining number will be
        `1/downsample` of original.
    full_range: `bool` (default: `False`)
        Whether to show the full time range of velocities before smoothing or
        subset to only smoothed range.
    figsize: `tuple` (default: `None`)
        Total figure size.
    pointsize: `float` (default: 2)
        Point size for scatter plots.
    linewidth: `float` (default: 1.5)
        Line width for anchor line or mean line.
    cmap: `str` (default: `coolwarm`)
        Color map for continuous color key.
    """
    from pandas.api.types import is_numeric_dtype, is_categorical_dtype
    if by not in ['expression', 'velocity']:
        raise ValueError('"by" must be either "expression" or "velocity".')
    if by == 'velocity':
        show_switches = False
    if color_by == 'state':
        types = [0, 1, 2, 3]
        colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']
    elif color_by in adata.obs and is_numeric_dtype(adata.obs[color_by]):
        types = None
        colors = adata.obs[color_by].values
    elif color_by in adata.obs and is_categorical_dtype(adata.obs[color_by]) \
            and color_by+'_colors' in adata.uns.keys():
        types = adata.obs[color_by].cat.categories
        colors = adata.uns[f'{color_by}_colors']
    else:
        raise ValueError('Currently, color key must be a single string of '
                         'either numerical or categorical available in adata '
                         'obs, and the colors of categories can be found in '
                         'adata uns.')

    downsample = np.clip(int(downsample), 1, 10)
    genes = np.array(genes)
    missing_genes = genes[~np.isin(genes, adata.var_names)]
    if len(missing_genes) > 0:
        main_info(f'{missing_genes} not found', v=0)
    genes = genes[np.isin(genes, adata.var_names)]
    gn = len(genes)
    if gn == 0:
        return
    if not gene_time:
        show_anchors = False
        latent_time = np.array(adata.obs['latent_time'])
        time_window = latent_time // 0.05
        time_window = time_window.astype(int)
        time_window[time_window == 20] = 19
    if 'velo_s_params' in adata.uns.keys() and 'outlier' \
            in adata.uns['velo_s_params']:
        outlier = adata.uns['velo_s_params']['outlier']
    else:
        outlier = 99

    fig, axs = plt.subplots(gn, 3, squeeze=False, figsize=(10, 2.3*gn)
                            if figsize is None else figsize)
    fig.patch.set_facecolor('white')
    for row, gene in enumerate(genes):
        u = adata[:, gene].layers['Mu' if by == 'expression' else 'velo_u']
        s = adata[:, gene].layers['Ms' if by == 'expression' else 'velo_s']
        c = adata[:, gene].layers['ATAC' if by == 'expression'
                                  else 'velo_chrom']
        if sp.__version__ < '1.14.0':
            c = c.A if sparse.issparse(c) else c
            u = u.A if sparse.issparse(u) else u
            s = s.A if sparse.issparse(s) else s
        else:
            c = c.toarray() if sparse.issparse(c) else c
            u = u.toarray() if sparse.issparse(u) else u
            s = s.toarray() if sparse.issparse(s) else s
        c, u, s = np.ravel(c), np.ravel(u), np.ravel(s)
        non_outlier = c <= np.percentile(c, outlier)
        non_outlier &= u <= np.percentile(u, outlier)
        non_outlier &= s <= np.percentile(s, outlier)
        c, u, s = c[non_outlier], u[non_outlier], s[non_outlier]
        time = np.array(adata[:, gene].layers['fit_t'] if gene_time
                        else latent_time)
        if by == 'velocity':
            time = np.reshape(time, (-1, 1))
            time = np.ravel(adata.obsp['_RNA_conn'].dot(time))
        time = time[non_outlier]
        if types is not None:
            for i in range(len(types)):
                if color_by == 'state':
                    filt = adata[non_outlier, gene].layers['fit_state'] \
                           == types[i]
                else:
                    filt = adata[non_outlier, :].obs[color_by] == types[i]
                filt = np.ravel(filt)
                if np.sum(filt) > 0:
                    axs[row, 0].scatter(time[filt][::downsample],
                                        c[filt][::downsample], s=pointsize,
                                        c=colors[i], alpha=0.6)
                    axs[row, 1].scatter(time[filt][::downsample],
                                        u[filt][::downsample],
                                        s=pointsize, c=colors[i], alpha=0.6)
                    axs[row, 2].scatter(time[filt][::downsample],
                                        s[filt][::downsample], s=pointsize,
                                        c=colors[i], alpha=0.6)
        else:
            axs[row, 0].scatter(time[::downsample], c[::downsample],
                                s=pointsize,
                                c=colors[non_outlier][::downsample],
                                alpha=0.6, cmap=cmap)
            axs[row, 1].scatter(time[::downsample], u[::downsample],
                                s=pointsize,
                                c=colors[non_outlier][::downsample],
                                alpha=0.6, cmap=cmap)
            axs[row, 2].scatter(time[::downsample], s[::downsample],
                                s=pointsize,
                                c=colors[non_outlier][::downsample],
                                alpha=0.6, cmap=cmap)

        if not gene_time:
            window_count = np.zeros(20)
            window_mean_c = np.zeros(20)
            window_mean_u = np.zeros(20)
            window_mean_s = np.zeros(20)
            for i in np.unique(time_window[non_outlier]):
                idx = time_window[non_outlier] == i
                window_count[i] = np.sum(idx)
                window_mean_c[i] = np.mean(c[idx])
                window_mean_u[i] = np.mean(u[idx])
                window_mean_s[i] = np.mean(s[idx])
            window_idx = np.where(window_count > 20)[0]
            axs[row, 0].plot(window_idx*0.05+0.025, window_mean_c[window_idx],
                             linewidth=linewidth, color='black', alpha=0.5)
            axs[row, 1].plot(window_idx*0.05+0.025, window_mean_u[window_idx],
                             linewidth=linewidth, color='black', alpha=0.5)
            axs[row, 2].plot(window_idx*0.05+0.025, window_mean_s[window_idx],
                             linewidth=linewidth, color='black', alpha=0.5)

        if show_anchors:
            n_anchors = adata.uns['velo_s_params']['t']
            t_sw_array = np.array([adata[:, gene].var['fit_t_sw1'],
                                   adata[:, gene].var['fit_t_sw2'],
                                   adata[:, gene].var['fit_t_sw3']])
            t_sw_array = t_sw_array[t_sw_array < 20]
            min_idx = int(adata[:, gene].var['fit_anchor_min_idx'])
            max_idx = int(adata[:, gene].var['fit_anchor_max_idx'])
            old_t = np.linspace(0, 20, n_anchors)[min_idx:max_idx+1]
            new_t = old_t - np.min(old_t)
            new_t = new_t * 20 / np.max(new_t)
            if by == 'velocity' and not full_range:
                anchor_interval = 20 / (max_idx + 1 - min_idx)
                min_idx = int(adata[:, gene].var['fit_anchor_velo_min_idx'])
                max_idx = int(adata[:, gene].var['fit_anchor_velo_max_idx'])
                start = 0 + (min_idx -
                             adata[:, gene].var['fit_anchor_min_idx']) \
                    * anchor_interval
                end = 20 + (max_idx -
                            adata[:, gene].var['fit_anchor_max_idx']) \
                    * anchor_interval
                new_t = np.linspace(start, end, max_idx + 1 - min_idx)
            ax = axs[row, 0]
            a_c = adata[:, gene].varm['fit_anchor_c' if by == 'expression'
                                      else 'fit_anchor_c_velo']\
                                .ravel()[min_idx:max_idx+1]
            if show_switches:
                for t_sw in t_sw_array:
                    if t_sw > 0:
                        ax.vlines(t_sw, np.min(c), np.max(c), colors='black',
                                  linestyles='dashed', alpha=0.5)
            ax.plot(new_t[0:new_t.shape[0]], a_c, linewidth=linewidth,
                    color='black', alpha=0.5)
            ax = axs[row, 1]
            a_u = adata[:, gene].varm['fit_anchor_u' if by == 'expression'
                                      else 'fit_anchor_u_velo']\
                                .ravel()[min_idx:max_idx+1]
            if show_switches:
                for t_sw in t_sw_array:
                    if t_sw > 0:
                        ax.vlines(t_sw, np.min(u), np.max(u), colors='black',
                                  linestyles='dashed', alpha=0.5)
            ax.plot(new_t[0:new_t.shape[0]], a_u, linewidth=linewidth,
                    color='black', alpha=0.5)
            ax = axs[row, 2]
            a_s = adata[:, gene].varm['fit_anchor_s' if by == 'expression'
                                      else 'fit_anchor_s_velo']\
                                .ravel()[min_idx:max_idx+1]
            if show_switches:
                for t_sw in t_sw_array:
                    if t_sw > 0:
                        ax.vlines(t_sw, np.min(s), np.max(s), colors='black',
                                  linestyles='dashed', alpha=0.5)
            ax.plot(new_t[0:new_t.shape[0]], a_s, linewidth=linewidth,
                    color='black', alpha=0.5)

        axs[row, 0].set_title(f'{gene} ATAC' if by == 'expression'
                              else f'{gene} chromatin velocity')
        axs[row, 0].set_xlabel('t' if by == 'expression' else '~t')
        axs[row, 0].set_ylabel('c' if by == 'expression' else 'dc/dt')
        axs[row, 1].set_title(f'{gene} unspliced' + ('' if by == 'expression'
                              else ' velocity'))
        axs[row, 1].set_xlabel('t' if by == 'expression' else '~t')
        axs[row, 1].set_ylabel('u' if by == 'expression' else 'du/dt')
        axs[row, 2].set_title(f'{gene} spliced' + ('' if by == 'expression'
                              else ' velocity'))
        axs[row, 2].set_xlabel('t' if by == 'expression' else '~t')
        axs[row, 2].set_ylabel('s' if by == 'expression' else 'ds/dt')

        for j in range(3):
            ax = axs[row, j]
            if not axis_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            if not frame_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.set_frame_on(False)
    fig.tight_layout()


def scatter_plot(adata,
                 genes,
                 by='us',
                 color_by='state',
                 n_cols=5,
                 axis_on=True,
                 frame_on=True,
                 show_anchors=True,
                 show_switches=True,
                 show_all_anchors=False,
                 title_more_info=False,
                 velocity_arrows=False,
                 downsample=1,
                 figsize=None,
                 pointsize=2,
                 markersize=5,
                 linewidth=2,
                 cmap='coolwarm',
                 view_3d_elev=None,
                 view_3d_azim=None,
                 full_name=False
                 ):
    """Gene scatter plot.

    This function plots phase portraits of the specified plane.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        Anndata result from dynamics recovery.
    genes: `str`,  list of `str`
        List of genes to plot.
    by: `str` (default: `us`)
        Plot unspliced-spliced plane if `us`. Plot chromatin-unspliced plane
        if `cu`.
        Plot 3D phase portraits if `cus`.
    color_by: `str` (default: `state`)
        Color by the four potential states if `state`. Other common values are
        leiden, louvain, celltype, etc.
        If not `state`, the color field must be present in `.uns`, which can be
        pre-computed with `scanpy.pl.scatter`.
        For `state`, red, orange, green, and blue represent state 1, 2, 3, and
        4, respectively.
        When `by=='us'`, `color_by` can also be `c`, which displays the log
        accessibility on U-S phase portraits.
    n_cols: `int` (default: 5)
        Number of columns to plot on each row.
    axis_on: `bool` (default: `True`)
        Whether to show axis labels.
    frame_on: `bool` (default: `True`)
        Whether to show plot frames.
    show_anchors: `bool` (default: `True`)
        Whether to display anchors.
    show_switches: `bool` (default: `True`)
        Whether to show switch times. The three switch times and the end of
        trajectory are indicated by
        circle, cross, dismond, and star, respectively.
    show_all_anchors: `bool` (default: `False`)
        Whether to display full range of (predicted) anchors even for
        repression-only genes.
    title_more_info: `bool` (default: `False`)
        Whether to display model, direction, and likelihood information for
        the gene in title.
    velocity_arrows: `bool` (default: `False`)
        Whether to show velocity arrows of cells on the phase portraits.
    downsample: `int` (default: 1)
        How much to downsample the cells. The remaining number will be
        `1/downsample` of original.
    figsize: `tuple` (default: `None`)
        Total figure size.
    pointsize: `float` (default: 2)
        Point size for scatter plots.
    markersize: `float` (default: 5)
        Point size for switch time points.
    linewidth: `float` (default: 2)
        Line width for connected anchors.
    cmap: `str` (default: `coolwarm`)
        Color map for log accessibilities or other continuous color keys when
        plotting on U-S plane.
    view_3d_elev: `float` (default: `None`)
        Matplotlib 3D plot `elev` argument. `elev=90` is the same as U-S plane,
        and `elev=0` is the same as C-U plane.
    view_3d_azim: `float` (default: `None`)
        Matplotlib 3D plot `azim` argument. `azim=270` is the same as U-S
        plane, and `azim=0` is the same as C-U plane.
    full_name: `bool` (default: `False`)
        Show full names for chromatin, unspliced, and spliced rather than
        using abbreviated terms c, u, and s.
    """
    from pandas.api.types import is_numeric_dtype, is_categorical_dtype
    if by not in ['us', 'cu', 'cus']:
        raise ValueError("'by' argument must be one of ['us', 'cu', 'cus']")
    if color_by == 'state':
        types = [0, 1, 2, 3]
        colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']
    elif by == 'us' and color_by == 'c':
        types = None
    elif color_by in adata.obs and is_numeric_dtype(adata.obs[color_by]):
        types = None
        colors = adata.obs[color_by].values
    elif color_by in adata.obs and is_categorical_dtype(adata.obs[color_by]) \
            and color_by+'_colors' in adata.uns.keys():
        types = adata.obs[color_by].cat.categories
        colors = adata.uns[f'{color_by}_colors']
    else:
        raise ValueError('Currently, color key must be a single string of '
                         'either numerical or categorical available in adata'
                         ' obs, and the colors of categories can be found in'
                         ' adata uns.')

    if 'velo_s_params' not in adata.uns.keys() \
            or 'fit_anchor_s' not in adata.varm.keys():
        show_anchors = False
    if color_by == 'state' and 'fit_state' not in adata.layers.keys():
        raise ValueError('fit_state is not found. Please run '
                         'recover_dynamics_chrom function first or provide a '
                         'valid color key.')

    downsample = np.clip(int(downsample), 1, 10)
    genes = np.array(genes)
    missing_genes = genes[~np.isin(genes, adata.var_names)]
    if len(missing_genes) > 0:
        main_info(f'{missing_genes} not found')
    genes = genes[np.isin(genes, adata.var_names)]
    gn = len(genes)
    if gn == 0:
        return
    if gn < n_cols:
        n_cols = gn
    if by == 'cus':
        fig, axs = plt.subplots(-(-gn // n_cols), n_cols, squeeze=False,
                                figsize=(3.2*n_cols, 2.7*(-(-gn // n_cols)))
                                if figsize is None else figsize,
                                subplot_kw={'projection': '3d'})
    else:
        fig, axs = plt.subplots(-(-gn // n_cols), n_cols, squeeze=False,
                                figsize=(2.7*n_cols, 2.4*(-(-gn // n_cols)))
                                if figsize is None else figsize)
    fig.patch.set_facecolor('white')
    count = 0
    for gene in genes:
        u = adata[:, gene].layers['Mu'].copy() if 'Mu' in adata.layers \
            else adata[:, gene].layers['unspliced'].copy()
        s = adata[:, gene].layers['Ms'].copy() if 'Ms' in adata.layers \
            else adata[:, gene].layers['spliced'].copy()
        if sp.__version__ < '1.14.0':
            u = u.A if sparse.issparse(u) else u
            s = s.A if sparse.issparse(s) else s

            u, s = np.ravel(u), np.ravel(s)
            if 'ATAC' not in adata.layers.keys() and \
                    'Mc' not in adata.layers.keys():
                show_anchors = False
            elif 'ATAC' in adata.layers.keys():
                c = adata[:, gene].layers['ATAC'].copy()
                c = c.A if sparse.issparse(c) else c
                c = np.ravel(c)
            elif 'Mc' in adata.layers.keys():
                c = adata[:, gene].layers['Mc'].copy()
                c = c.A if sparse.issparse(c) else c
                c = np.ravel(c)
        else:
            u = u.toarray() if sparse.issparse(u) else u
            s = s.toarray() if sparse.issparse(s) else s
            u, s = np.ravel(u), np.ravel(s)
            if 'ATAC' not in adata.layers.keys() and \
                    'Mc' not in adata.layers.keys():
                show_anchors = False
            elif 'ATAC' in adata.layers.keys():
                c = adata[:, gene].layers['ATAC'].copy()
                c = c.toarray() if sparse.issparse(c) else c
                c = np.ravel(c)
            elif 'Mc' in adata.layers.keys():
                c = adata[:, gene].layers['Mc'].copy()
                c = c.toarray() if sparse.issparse(c) else c
                c = np.ravel(c)

        if velocity_arrows:
            if 'velo_u' in adata.layers.keys():
                vu = adata[:, gene].layers['velo_u'].copy()
            elif 'velocity_u' in adata.layers.keys():
                vu = adata[:, gene].layers['velocity_u'].copy()
            else:
                vu = np.zeros(adata.n_obs)
            max_u = np.max([np.max(u), 1e-6])
            u /= max_u
            vu = np.ravel(vu)
            vu /= np.max([np.max(np.abs(vu)), 1e-6])
            if 'velo_s' in adata.layers.keys():
                vs = adata[:, gene].layers['velo_s'].copy()
            elif 'velocity' in adata.layers.keys():
                vs = adata[:, gene].layers['velocity'].copy()
            max_s = np.max([np.max(s), 1e-6])
            s /= max_s
            vs = np.ravel(vs)
            vs /= np.max([np.max(np.abs(vs)), 1e-6])
            if 'velo_chrom' in adata.layers.keys():
                vc = adata[:, gene].layers['velo_chrom'].copy()
                max_c = np.max([np.max(c), 1e-6])
                c /= max_c
                vc = np.ravel(vc)
                vc /= np.max([np.max(np.abs(vc)), 1e-6])

        row = count // n_cols
        col = count % n_cols
        ax = axs[row, col]
        if types is not None:
            for i in range(len(types)):
                if color_by == 'state':
                    filt = adata[:, gene].layers['fit_state'] == types[i]
                else:
                    filt = adata.obs[color_by] == types[i]
                filt = np.ravel(filt)
                if by == 'us':
                    if velocity_arrows:
                        ax.quiver(s[filt][::downsample], u[filt][::downsample],
                                  vs[filt][::downsample],
                                  vu[filt][::downsample], color=colors[i],
                                  alpha=0.5, scale_units='xy', scale=10,
                                  width=0.005, headwidth=4, headaxislength=5.5)
                    else:
                        ax.scatter(s[filt][::downsample],
                                   u[filt][::downsample], s=pointsize,
                                   c=colors[i], alpha=0.7)
                elif by == 'cu':
                    if velocity_arrows:
                        ax.quiver(u[filt][::downsample],
                                  c[filt][::downsample],
                                  vu[filt][::downsample],
                                  vc[filt][::downsample], color=colors[i],
                                  alpha=0.5, scale_units='xy', scale=10,
                                  width=0.005, headwidth=4, headaxislength=5.5)
                    else:
                        ax.scatter(u[filt][::downsample],
                                   c[filt][::downsample], s=pointsize,
                                   c=colors[i], alpha=0.7)
                else:
                    if velocity_arrows:
                        ax.quiver(s[filt][::downsample],
                                  u[filt][::downsample], c[filt][::downsample],
                                  vs[filt][::downsample],
                                  vu[filt][::downsample],
                                  vc[filt][::downsample],
                                  color=colors[i], alpha=0.4, length=0.1,
                                  arrow_length_ratio=0.5, normalize=True)
                    else:
                        ax.scatter(s[filt][::downsample],
                                   u[filt][::downsample],
                                   c[filt][::downsample], s=pointsize,
                                   c=colors[i], alpha=0.7)
        elif color_by == 'c':
            if 'velo_s_params' in adata.uns.keys() and \
                    'outlier' in adata.uns['velo_s_params']:
                outlier = adata.uns['velo_s_params']['outlier']
            else:
                outlier = 99.8
            non_zero = (u > 0) & (s > 0) & (c > 0)
            non_outlier = u < np.percentile(u, outlier)
            non_outlier &= s < np.percentile(s, outlier)
            non_outlier &= c < np.percentile(c, outlier)
            c -= np.min(c)
            c /= np.max(c)
            if velocity_arrows:
                ax.quiver(s[non_zero & non_outlier][::downsample],
                          u[non_zero & non_outlier][::downsample],
                          vs[non_zero & non_outlier][::downsample],
                          vu[non_zero & non_outlier][::downsample],
                          np.log1p(c[non_zero & non_outlier][::downsample]),
                          alpha=0.5,
                          scale_units='xy', scale=10, width=0.005,
                          headwidth=4, headaxislength=5.5, cmap=cmap)
            else:
                ax.scatter(s[non_zero & non_outlier][::downsample],
                           u[non_zero & non_outlier][::downsample],
                           s=pointsize,
                           c=np.log1p(c[non_zero & non_outlier][::downsample]),
                           alpha=0.8, cmap=cmap)
        else:
            if by == 'us':
                if velocity_arrows:
                    ax.quiver(s[::downsample], u[::downsample],
                              vs[::downsample], vu[::downsample],
                              colors[::downsample], alpha=0.5,
                              scale_units='xy', scale=10, width=0.005,
                              headwidth=4, headaxislength=5.5, cmap=cmap)
                else:
                    ax.scatter(s[::downsample], u[::downsample], s=pointsize,
                               c=colors[::downsample], alpha=0.7, cmap=cmap)
            elif by == 'cu':
                if velocity_arrows:
                    ax.quiver(u[::downsample], c[::downsample],
                              vu[::downsample], vc[::downsample],
                              colors[::downsample], alpha=0.5,
                              scale_units='xy', scale=10, width=0.005,
                              headwidth=4, headaxislength=5.5, cmap=cmap)
                else:
                    ax.scatter(u[::downsample], c[::downsample], s=pointsize,
                               c=colors[::downsample], alpha=0.7, cmap=cmap)
            else:
                if velocity_arrows:
                    ax.quiver(s[::downsample], u[::downsample],
                              c[::downsample], vs[::downsample],
                              vu[::downsample], vc[::downsample],
                              colors[::downsample], alpha=0.4, length=0.1,
                              arrow_length_ratio=0.5, normalize=True,
                              cmap=cmap)
                else:
                    ax.scatter(s[::downsample], u[::downsample],
                               c[::downsample], s=pointsize,
                               c=colors[::downsample], alpha=0.7, cmap=cmap)

        if show_anchors:
            min_idx = int(adata[:, gene].var['fit_anchor_min_idx'])
            max_idx = int(adata[:, gene].var['fit_anchor_max_idx'])
            a_c = adata[:, gene].varm['fit_anchor_c']\
                .ravel()[min_idx:max_idx+1].copy()
            a_u = adata[:, gene].varm['fit_anchor_u']\
                .ravel()[min_idx:max_idx+1].copy()
            a_s = adata[:, gene].varm['fit_anchor_s']\
                .ravel()[min_idx:max_idx+1].copy()
            if velocity_arrows:
                a_c /= max_c
                a_u /= max_u
                a_s /= max_s
            if by == 'us':
                ax.plot(a_s, a_u, linewidth=linewidth, color='black',
                        alpha=0.7, zorder=1000)
            elif by == 'cu':
                ax.plot(a_u, a_c, linewidth=linewidth, color='black',
                        alpha=0.7, zorder=1000)
            else:
                ax.plot(a_s, a_u, a_c, linewidth=linewidth, color='black',
                        alpha=0.7, zorder=1000)
            if show_all_anchors:
                a_c_pre = adata[:, gene].varm['fit_anchor_c']\
                    .ravel()[:min_idx].copy()
                a_u_pre = adata[:, gene].varm['fit_anchor_u']\
                    .ravel()[:min_idx].copy()
                a_s_pre = adata[:, gene].varm['fit_anchor_s']\
                    .ravel()[:min_idx].copy()
                if velocity_arrows:
                    a_c_pre /= max_c
                    a_u_pre /= max_u
                    a_s_pre /= max_s
                if len(a_c_pre) > 0:
                    if by == 'us':
                        ax.plot(a_s_pre, a_u_pre, linewidth=linewidth/1.3,
                                color='black', alpha=0.6, zorder=1000)
                    elif by == 'cu':
                        ax.plot(a_u_pre, a_c_pre, linewidth=linewidth/1.3,
                                color='black', alpha=0.6, zorder=1000)
                    else:
                        ax.plot(a_s_pre, a_u_pre, a_c_pre,
                                linewidth=linewidth/1.3, color='black',
                                alpha=0.6, zorder=1000)
            if show_switches:
                t_sw_array = np.array([adata[:, gene].var['fit_t_sw1']
                                      .values[0],
                                      adata[:, gene].var['fit_t_sw2']
                                      .values[0],
                                      adata[:, gene].var['fit_t_sw3']
                                      .values[0]])
                in_range = (t_sw_array > 0) & (t_sw_array < 20)
                a_c_sw = adata[:, gene].varm['fit_anchor_c_sw'].ravel().copy()
                a_u_sw = adata[:, gene].varm['fit_anchor_u_sw'].ravel().copy()
                a_s_sw = adata[:, gene].varm['fit_anchor_s_sw'].ravel().copy()
                if velocity_arrows:
                    a_c_sw /= max_c
                    a_u_sw /= max_u
                    a_s_sw /= max_s
                if in_range[0]:
                    c_sw1, u_sw1, s_sw1 = a_c_sw[0], a_u_sw[0], a_s_sw[0]
                    if by == 'us':
                        ax.plot([s_sw1], [u_sw1], "om", markersize=markersize,
                                zorder=2000)
                    elif by == 'cu':
                        ax.plot([u_sw1], [c_sw1], "om", markersize=markersize,
                                zorder=2000)
                    else:
                        ax.plot([s_sw1], [u_sw1], [c_sw1], "om",
                                markersize=markersize, zorder=2000)
                if in_range[1]:
                    c_sw2, u_sw2, s_sw2 = a_c_sw[1], a_u_sw[1], a_s_sw[1]
                    if by == 'us':
                        ax.plot([s_sw2], [u_sw2], "Xm", markersize=markersize,
                                zorder=2000)
                    elif by == 'cu':
                        ax.plot([u_sw2], [c_sw2], "Xm", markersize=markersize,
                                zorder=2000)
                    else:
                        ax.plot([s_sw2], [u_sw2], [c_sw2], "Xm",
                                markersize=markersize, zorder=2000)
                if in_range[2]:
                    c_sw3, u_sw3, s_sw3 = a_c_sw[2], a_u_sw[2], a_s_sw[2]
                    if by == 'us':
                        ax.plot([s_sw3], [u_sw3], "Dm", markersize=markersize,
                                zorder=2000)
                    elif by == 'cu':
                        ax.plot([u_sw3], [c_sw3], "Dm", markersize=markersize,
                                zorder=2000)
                    else:
                        ax.plot([s_sw3], [u_sw3], [c_sw3], "Dm",
                                markersize=markersize, zorder=2000)
                if max_idx > adata.uns['velo_s_params']['t'] - 4:
                    if by == 'us':
                        ax.plot([a_s[-1]], [a_u[-1]], "*m",
                                markersize=markersize, zorder=2000)
                    elif by == 'cu':
                        ax.plot([a_u[-1]], [a_c[-1]], "*m",
                                markersize=markersize, zorder=2000)
                    else:
                        ax.plot([a_s[-1]], [a_u[-1]], [a_c[-1]], "*m",
                                markersize=markersize, zorder=2000)

        if by == 'cus' and \
                (view_3d_elev is not None or view_3d_azim is not None):
            # US: elev=90, azim=270. CU: elev=0, azim=0.
            ax.view_init(elev=view_3d_elev, azim=view_3d_azim)
        title = gene
        if title_more_info:
            if 'fit_model' in adata.var:
                title += f" M{int(adata[:,gene].var['fit_model'].values[0])}"
            if 'fit_direction' in adata.var:
                title += f" {adata[:,gene].var['fit_direction'].values[0]}"
            if 'fit_likelihood' in adata.var \
                    and not np.all(adata.var['fit_likelihood'].values == -1):
                title += " "
                f"{adata[:,gene].var['fit_likelihood'].values[0]:.3g}"
        ax.set_title(f'{title}', fontsize=11)
        if by == 'us':
            ax.set_xlabel('spliced' if full_name else 's')
            ax.set_ylabel('unspliced' if full_name else 'u')
        elif by == 'cu':
            ax.set_xlabel('unspliced' if full_name else 'u')
            ax.set_ylabel('chromatin' if full_name else 'c')
        elif by == 'cus':
            ax.set_xlabel('spliced' if full_name else 's')
            ax.set_ylabel('unspliced' if full_name else 'u')
            ax.set_zlabel('chromatin' if full_name else 'c')
        if by in ['us', 'cu']:
            if not axis_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            if not frame_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.set_frame_on(False)
        elif by == 'cus':
            if not axis_on:
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_zlabel('')
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])
                ax.zaxis.set_ticklabels([])
            if not frame_on:
                ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.zaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.xaxis._axinfo['tick']['inward_factor'] = 0
                ax.xaxis._axinfo['tick']['outward_factor'] = 0
                ax.yaxis._axinfo['tick']['inward_factor'] = 0
                ax.yaxis._axinfo['tick']['outward_factor'] = 0
                ax.zaxis._axinfo['tick']['inward_factor'] = 0
                ax.zaxis._axinfo['tick']['outward_factor'] = 0
        count += 1
    for i in range(col+1, n_cols):
        fig.delaxes(axs[row, i])
    fig.tight_layout()