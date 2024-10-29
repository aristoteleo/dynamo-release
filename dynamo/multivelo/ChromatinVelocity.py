import numpy as np
from scipy.sparse import issparse
from typing import Literal

# Import from dynamo
from ..dynamo_logger import (
    main_exception,
)


# ChromatinVelocity class - patterned after MultiVelo, but retains accessibility at individual CRE
class ChromatinVelocity:
    def __init__(self,
                 c,
                 u,
                 s,
                 ss,
                 us,
                 uu,
                 fit_args=None,
                 gene=None,
                 r2_adjusted=False):
        self.gene = gene
        self.outlier = np.clip(fit_args['outlier'], a_min=80, a_max=100)
        self.r2_adjusted = r2_adjusted
        self.total_n = len(u)

        # Convert all sparse vectors to dense ones
        c = c.A if issparse(c) else c
        s = s.A if issparse(s) else s
        u = u.A if issparse(u) else u
        ss = ss.A if ((ss is not None) and issparse(ss)) else ss
        us = us.A if ((us is not None) and issparse(us)) else us
        uu = uu.A if ((uu is not None) and issparse(uu)) else uu

        # In distinction to MultiVelo c will be (total_n, n_peak) array
        # Sweep the minimum value in each column from the array
        self.offset_c = np.min(c, axis=0)
        self.c_all = c - self.offset_c

        # The other moments are (total_n, ) arrays
        self.s_all, self.u_all = np.ravel(np.array(s, dtype=np.float64)), np.ravel(np.array(u, dtype=np.float64))
        self.offset_s, self.offset_u = np.min(self.s_all), np.min(self.u_all)
        self.s_all -= self.offset_s
        self.u_all -= self.offset_u

        # For 'stochastic' method also need second moments
        if ss is not None:
            self.ss_all = np.ravel(np.array(ss, dtype=np.float64))
        if us is not None:
            self.us_all = np.ravel(np.array(us, dtype=np.float64))
        if uu is not None:
            self.uu_all = np.ravel(np.array(uu, dtype=np.float64))

        # Ensure at least one element in each cell is positive
        any_c_positive = np.any(self.c_all > 0, axis=1)
        self.non_zero = np.ravel(any_c_positive) | np.ravel(self.u_all > 0) | np.ravel(self.s_all > 0)

        # remove outliers
        # ... for chromatin, we'll be more stringent - if *any* peak count for a cell
        # is an outlier, we'll remove that cell
        self.non_outlier = np.all(self.c_all <= np.percentile(self.c_all, self.outlier, axis=0), axis=1)
        self.non_outlier &= np.ravel(self.u_all <= np.percentile(self.u_all, self.outlier))
        self.non_outlier &= np.ravel(self.s_all <= np.percentile(self.s_all, self.outlier))
        self.c = self.c_all[self.non_zero & self.non_outlier]
        self.u = self.u_all[self.non_zero & self.non_outlier]
        self.s = self.s_all[self.non_zero & self.non_outlier]
        self.ss = (None if ss is None
                   else self.ss_all[self.non_zero & self.non_outlier])
        self.us = (None if us is None
                   else self.us_all[self.non_zero & self.non_outlier])
        self.uu = (None if uu is None
                   else self.uu_all[self.non_zero & self.non_outlier])
        self.low_quality = len(self.u) < 10

        # main_info(f'{len(self.u)} cells passed filter and will be used to fit regressions.')

        # 4 rate parameters
        self.alpha_c = 0.1
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma_det = 0.0
        self.gamma_stoch = 0.0

        # other parameters or results
        self.loss_det = np.inf
        self.loss_stoch = np.inf
        self.r2_det = 0
        self.r2_stoch = 0
        self.residual_det = None
        self.residual_stoch = None
        self.residual2_stoch = None

        self.steady_state_func = None

        # Select the cells for regression
        w_sub_for_c = np.any(self.c >= 0.1 * np.max(self.c, axis=0), axis=1)
        w_sub = w_sub_for_c & (self.u >= 0.1 * np.max(self.u)) & (self.s >= 0.1 * np.max(self.s))
        c_sub = self.c[w_sub]
        w_sub_for_c = np.any(self.c >= np.mean(c_sub, axis=0) + np.std(c_sub, axis=0))
        w_sub = w_sub_for_c & (self.u >= 0.1 * np.max(self.u)) & (self.s >= 0.1 * np.max(self.s))
        self.w_sub = w_sub
        if np.sum(self.w_sub) < 10:
            self.low_quality = True

    # This method originated from MultiVelo - Corrected R^2
    def compute_deterministic(self):
        # Steady state slope - no different than usual transcriptomic version
        u_high = self.u[self.w_sub]
        s_high = self.s[self.w_sub]
        wu_high = u_high >= np.percentile(u_high, 95)
        ws_high = s_high >= np.percentile(s_high, 95)
        ss_u = u_high[wu_high | ws_high]
        ss_s = s_high[wu_high | ws_high]

        gamma_det = np.dot(ss_u, ss_s) / np.dot(ss_s, ss_s)
        self.steady_state_func = lambda x: gamma_det * x
        residual_det = self.u_all - self.steady_state_func(self.s_all)

        loss_det = np.dot(residual_det, residual_det) / len(self.u_all)

        if self.r2_adjusted:
            gamma_det = np.dot(self.u, self.s) / np.dot(self.s, self.s)
            residual_det = self.u_all - gamma_det * self.s_all

        total_det = self.u_all - np.mean(self.u_all)
        # total_det = self.u_all # Since fitting only slope with zero intercept, should not include mean

        self.gamma_det = gamma_det
        self.loss_det = loss_det
        self.residual_det = residual_det

        self.r2_det = 1 - np.dot(residual_det, residual_det) / np.dot(total_det, total_det)


    # This method originated from MultiVelo
    def compute_stochastic(self):
        self.compute_deterministic()

        var_ss = 2 * self.ss - self.s
        cov_us = 2 * self.us + self.u
        s_all_ = 2 * self.s_all ** 2 - (2 * self.ss_all - self.s_all)
        u_all_ = (2 * self.us_all + self.u_all) - 2 * self.u_all * self.s_all
        gamma2 = np.dot(cov_us, var_ss) / np.dot(var_ss, var_ss)
        residual2 = cov_us - gamma2 * var_ss
        std_first = np.std(self.residual_det)
        std_second = np.std(residual2)

        # chromatin adjusted steady-state slope
        u_high = self.u[self.w_sub]
        s_high = self.s[self.w_sub]
        wu_high = u_high >= np.percentile(u_high, 95)
        ws_high = s_high >= np.percentile(s_high, 95)
        ss_u = u_high * (wu_high | ws_high)
        ss_s = s_high * (wu_high | ws_high)
        a = np.hstack((ss_s / std_first, var_ss[self.w_sub] / std_second))
        b = np.hstack((ss_u / std_first, cov_us[self.w_sub] / std_second))

        gamma_stoch = np.dot(b, a) / np.dot(a, a)
        self.steady_state_func = lambda x: gamma_stoch * x
        self.residual_stoch = self.u_all - self.steady_state_func(self.s_all)
        self.residual2_stoch = u_all_ - self.steady_state_func(s_all_)
        loss_stoch = np.dot(self.residual_stoch, self.residual_stoch) / len(self.u_all)

        self.gamma_stoch = gamma_stoch
        self.loss_stoch = loss_stoch
        self.r2_stoch = 1 - np.dot(self.residual_stoch, self.residual_stoch) / np.dot(self.u_all, self.u_all)

    def get_gamma(self,
                  mode: Literal['deterministic', 'stochastic'] = 'stochastic'):
        if mode == 'deterministic':
            return self.gamma_det
        elif mode == 'stochastic':
            return self.gamma_stoch
        else:
            main_exception(f"Unknown mode {mode} - must be one of 'deterministic' or 'stochastic'")

    def get_loss(self,
                 mode: Literal['deterministic', 'stochastic'] = 'stochastic'):
        if mode == 'deterministic':
            return self.loss_det
        elif mode == 'stochastic':
            return self.loss_stoch
        else:
            main_exception(f"Unknown mode {mode} - must be one of 'deterministic' or 'stochastic'")

    def get_r2(self,
               mode: Literal['deterministic', 'stochastic'] = 'stochastic'):
        if mode == 'deterministic':
            return self.r2_det
        elif mode == 'stochastic':
            return self.r2_stoch
        else:
            main_exception(f"Unknown mode {mode} - must be one of 'deterministic' or 'stochastic'")

    def get_variance_velocity(self,
                              mode: Literal['deterministic', 'stochastic'] = 'stochastic'):
        if mode == 'stochastic':
            return self.residual2_stoch
        else:
            main_exception("Should not call get_variance_velocity for mode other than 'stochastic'")

    def get_velocity(self,
                     mode: Literal['deterministic', 'stochastic'] = 'stochastic'):
        vel = None # Make the lint checker happy
        if mode == 'deterministic':
            vel = self.residual_det
        elif mode == 'stochastic':
            vel = self.residual_stoch
        else:
            main_exception(f"Unknown mode {mode} - must be one of 'deterministic' or 'stochastic'")

        return vel
