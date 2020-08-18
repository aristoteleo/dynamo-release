#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 4 18:29:24 2019

@author: yaz
"""

from numpy import *
from scipy.integrate import odeint
from scipy.optimize import least_squares
# from numba import jitclass  # import the decorator
from numba import float32  # import the types
from ...tools.sampling import lhsclassic

spec = [
    ("a", float32),
    ("b", float32),
    ("alpha_a", float32),
    ("alpha_i", float32),
    ("beta", float32),
    ("gamma", float32),
]

# @jitclass(spec)
class moments:
    def __init__(
        self, a=None, b=None, alpha_a=None, alpha_i=None, beta=None, gamma=None
    ):
        # species
        self.ua = 0
        self.ui = 1
        self.xa = 2
        self.xi = 3
        self.uu = 4
        self.xx = 5
        self.ux = 6

        self.n_species = 7

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
            or alpha_a is None
            or alpha_i is None
            or beta is None
            or gamma is None
        ):
            self.set_params(a, b, alpha_a, alpha_i, beta, gamma)

    def ode_moments(self, x, t):
        dx = zeros(len(x))
        # parameters
        a = self.a
        b = self.b
        aa = self.aa
        ai = self.ai
        be = self.be
        ga = self.ga

        # first moments
        dx[self.ua] = aa - be * x[self.ua] + a * (x[self.ui] - x[self.ua])
        dx[self.ui] = ai - be * x[self.ui] - b * (x[self.ui] - x[self.ua])
        dx[self.xa] = be * x[self.ua] - ga * x[self.xa] + a * (x[self.xi] - x[self.xa])
        dx[self.xi] = be * x[self.ui] - ga * x[self.xi] - b * (x[self.xi] - x[self.xa])

        # second moments
        dx[self.uu] = (
            2 * self.fbar(aa * x[self.ua], ai * x[self.ui]) - 2 * be * x[self.uu]
        )
        dx[self.xx] = 2 * be * x[self.ux] - 2 * ga * x[self.xx]
        dx[self.ux] = (
            self.fbar(aa * x[self.xa], ai * x[self.xi])
            + be * x[self.uu]
            - (be + ga) * x[self.ux]
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

    def set_params(self, a, b, alpha_a, alpha_i, beta, gamma):
        self.a = a
        self.b = b
        self.aa = alpha_a
        self.ai = alpha_i
        self.be = beta
        self.ga = gamma

        # reset solutions
        self.t = None
        self.x = None
        self.K = None
        self.p = None

    def get_all_central_moments(self):
        ret = zeros((4, len(self.t)))
        ret[0] = self.get_nu()
        ret[1] = self.get_nx()
        ret[2] = self.get_var_nu()
        ret[3] = self.get_var_nx()
        return ret

    def get_nosplice_central_moments(self):
        ret = zeros((2, len(self.t)))
        ret[0] = self.get_n_labeled()
        ret[1] = self.get_var_labeled()
        return ret

    def get_nu(self):
        return self.fbar(self.x[:, self.ua], self.x[:, self.ui])

    def get_nx(self):
        return self.fbar(self.x[:, self.xa], self.x[:, self.xi])

    def get_n_labeled(self):
        return self.get_nu() + self.get_nx()

    def get_var_nu(self):
        c = self.get_nu()
        return self.x[:, self.uu] + c - c ** 2

    def get_var_nx(self):
        c = self.get_nx()
        return self.x[:, self.xx] + c - c ** 2

    def get_cov_ux(self):
        cu = self.get_nu()
        cx = self.get_nx()
        return self.x[:, self.ux] - cu * cx

    def get_var_labeled(self):
        return self.get_var_nu() + self.get_var_nx() + 2 * self.get_cov_ux()

    def computeKnp(self):
        # parameters
        a = self.a
        b = self.b
        aa = self.aa
        ai = self.ai
        be = self.be
        ga = self.ga

        K = zeros((self.n_species, self.n_species))
        # E1
        K[self.ua, self.ua] = -be - a
        K[self.ua, self.ui] = a
        K[self.ui, self.ua] = b
        K[self.ui, self.ui] = -be - b

        # E2
        K[self.xa, self.xa] = -ga - a
        K[self.xa, self.xi] = a
        K[self.xi, self.xa] = b
        K[self.xi, self.xi] = -ga - b

        # E3
        K[self.uu, self.uu] = -2 * be
        K[self.xx, self.xx] = -2 * ga

        # E4
        K[self.ux, self.ux] = -be - ga

        # F21
        K[self.xa, self.ua] = be
        K[self.xi, self.ui] = be

        # F31
        K[self.uu, self.ua] = 2 * aa * b / (a + b)
        K[self.uu, self.ui] = 2 * ai * a / (a + b)

        # F34
        K[self.xx, self.ux] = 2 * be

        # F42
        K[self.ux, self.xa] = aa * b / (a + b)
        K[self.ux, self.xi] = ai * a / (a + b)

        # F43
        K[self.ux, self.uu] = be

        p = zeros(self.n_species)
        p[self.ua] = aa
        p[self.ui] = ai

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
                ret[:, i] = (
                    ret[:, i] * (self.ranges[i][1] - self.ranges[i][0])
                    + self.ranges[i][0]
                )
        else:
            for n in range(samples):
                for i in range(self.n_params):
                    r = random.rand()
                    ret[n, i] = (
                        r * (self.ranges[i][1] - self.ranges[i][0]) + self.ranges[i][0]
                    )
        return ret

    def _lhsclassic(self, samples):
        # From PyDOE
        # Generate the intervals
        H = lhsclassic(samples, self.n_params)

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
        res = log(X + 1)
        return res

    def f_lsq(
        self,
        params,
        t,
        x_data_norm,
        method="analytical",
        normalize=True,
        experiment_type=None,
    ):
        self.simulator.set_params(*params)
        if method == "numerical":
            self.simulator.integrate(t, self.simulator.x0)
        elif method == "analytical":
            self.simulator.solve(t, self.simulator.x0)
        if experiment_type is None:
            ret = self.simulator.get_all_central_moments()
        elif experiment_type == "nosplice":
            ret = self.simulator.get_nosplice_central_moments()
        ret = self.normalize_data(ret).flatten() if normalize else ret.flatten()
        ret[isnan(ret)] = 0
        return ret - x_data_norm

    def fit_lsq(
        self,
        t,
        x_data,
        p0=None,
        n_p0=1,
        bounds=None,
        sample_method="lhs",
        method="analytical",
        normalize=True,
        experiment_type=None,
    ):
        if p0 is None:
            p0 = self.sample_p0(n_p0, sample_method)
        else:
            if p0.ndim == 1:
                p0 = [p0]
            n_p0 = len(p0)

        x_data_norm = self.normalize_data(x_data) if normalize else x_data

        if bounds is None:
            bounds = (self.get_bound(0), self.get_bound(1))

        costs = zeros(n_p0)
        X = []
        for i in range(n_p0):
            ret = least_squares(
                lambda p: self.f_lsq(
                    p,
                    t,
                    x_data_norm.flatten(),
                    method,
                    normalize=normalize,
                    experiment_type=experiment_type,
                ),
                p0[i],
                bounds=bounds,
            )
            costs[i] = ret.cost
            X.append(ret.x)
        i_min = argmin(costs)
        return X[i_min], costs[i_min]
