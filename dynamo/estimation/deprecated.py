import functools
import warnings

from numba import float32  # import the types
from numpy import *
from scipy.optimize import least_squares

from ..tools.sampling import lhsclassic
from .tsc.utils_moments import moments


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
