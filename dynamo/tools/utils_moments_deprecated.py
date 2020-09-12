#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 08:27:36 2019

@author: yaz
"""

from numpy import *
from scipy.integrate import odeint
from scipy.optimize import curve_fit, least_squares


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
        dx[self.xa] = (
            be * (1 - si) * x[self.ua] - ga * x[self.xa] + a * (x[self.xi] - x[self.xa])
        )
        dx[self.xi] = (
            be * (1 - si) * x[self.ui] - ga * x[self.xi] - b * (x[self.xi] - x[self.xa])
        )
        dx[self.ya] = (
            be * si * x[self.ua]
            + be * x[self.wa]
            - ga * x[self.ya]
            + a * (x[self.yi] - x[self.ya])
        )
        dx[self.yi] = (
            be * si * x[self.ui]
            + be * x[self.wi]
            - ga * x[self.yi]
            - b * (x[self.yi] - x[self.ya])
        )

        # second moments
        dx[self.uu] = (
            2 * la * self.fbar(aa * x[self.ua], ai * x[self.ui]) - 2 * be * x[self.uu]
        )
        dx[self.ww] = (
            2 * (1 - la) * self.fbar(self.aa * x[self.wa], ai * x[self.wi])
            - 2 * be * x[self.ww]
        )
        dx[self.xx] = 2 * be * (1 - si) * x[self.ux] - 2 * ga * x[self.xx]
        dx[self.yy] = (
            2 * si * be * x[self.uy] + 2 * be * x[self.wy] - 2 * ga * x[self.yy]
        )
        dx[self.uw] = (
            la * self.fbar(aa * x[self.wa], ai * x[self.wi])
            + (1 - la) * self.fbar(aa * x[self.ua], ai * x[self.ui])
            - 2 * be * x[self.uw]
        )
        dx[self.ux] = (
            la * self.fbar(aa * x[self.xa], ai * x[self.xi])
            + be * (1 - si) * x[self.uu]
            - (be + ga) * x[self.ux]
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
        return self.x[:, self.uu] + c - c ** 2

    def get_var_nw(self):
        c = self.get_nw()
        return self.x[:, self.ww] + c - c ** 2

    def get_var_nx(self):
        c = self.get_nx()
        return self.x[:, self.xx] + c - c ** 2

    def get_var_ny(self):
        c = self.get_ny()
        return self.x[:, self.yy] + c - c ** 2

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
        popt, pcov = curve_fit(
            self.f_curve_fit, t, x_data_norm.flatten(), p0=p0, bounds=bounds
        )
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
                lambda p: self.f_lsq(
                    p, t, x_data_norm.flatten(), method, experiment_type
                ),
                p0[i],
                bounds=bounds,
            )
            costs[i] = ret.cost
            X.append(ret.x)
        i_min = argmin(costs)
        return X[i_min], costs[i_min]
