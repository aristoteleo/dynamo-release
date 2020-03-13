from .utils import lhsclassic
import numpy as np
from scipy.optimize import least_squares
from .utils_kinetics import *
import warnings

def estimate_p0_deg_nosp(x_data, time):
    '''Roughly estimate p0 with the assumption that time starts at 0 for degradation data without splicing.'''
    u0 = x_data[0][0]
    uu0 = x_data[1][0]
    ga0 = np.clip(np.log(x_data[0][0]/(x_data[0][-1]+1e-6)) / time[-1], 0, 1000)
    return np.array([ga0, u0, uu0])

class Estimation:
    def __init__(self, ranges, simulator, x0=None):
        '''A general parameter estimation framework for all types of time-seris data
        Arguments
        ---------
            ranges: `numpy.ndarray`
                a n-by-2 numpy array containing the lower and upper ranges of n parameters 
                (and initial conditions if not fixed).
            simulator: class
                an instance of python class which solves ODEs. It should have properties 't' (k time points, 1d numpy array),
                'x0' (initial conditions for m species, 1d numpy array), and 'x' (solution, k-by-m array), 
                as well as two functions: integrate (numerical integration), solve (analytical method).
            x0: `numpy.ndarray`
                Initial conditions for the integrators if they are fixed.
        '''
        self.ranges = ranges
        self.n_params = len(ranges)
        self.simulator = simulator
        if x0 is not None:
            self.simulator.x0 = x0
            self.fix_x0 = True
        else:
            self.fix_x0 = False
        self.popt = None
        self.cost = None

    def sample_p0(self, samples=1, method='lhs'):
        ret = np.zeros((samples, self.n_params))
        if method == 'lhs':
            ret = self._lhsclassic(samples)
            for i in range(self.n_params):
                ret[:, i] = ret[:, i] * (self.ranges[i][1] - self.ranges[i][0]) + self.ranges[i][0]
        else:
            for n in range(samples):
                for i in range(self.n_params):
                    r = np.random.rand()
                    ret[n, i] = r * (self.ranges[i][1] - self.ranges[i][0]) + self.ranges[i][0]
        return ret

    def _lhsclassic(self, samples):
        # From PyDOE
        # Generate the intervals
        #from .utils import lhsclassic
        H = lhsclassic(samples, self.n_params)

        return H

    def get_bound(self, axis):
        ret = np.zeros(self.n_params)
        for i in range(self.n_params):
            ret[i] = self.ranges[i][axis]
        return ret

    def normalize_data(self, X):
        return np.log(X + 1)
    
    def extract_data_from_simulator(self):
        return self.simulator.x.T
    
    def f_lsq(self, params, t, x_data, method='matrix', normalize=True):
        if method not in self.simulator.methods: 
            warnings.warn('The simulator does not support method \'{}\'. Using method \'{}\' instead.'.format(method, self.simulator.methods[0]))
            method = self.simulator.methods[0]
        if self.fix_x0:
            self.simulator.set_params(0, *params)
        else:
            self.simulator.set_params(0, *params[:self.n_params - self.simulator.n_species])
        x0 = self.simulator.x0 if self.fix_x0 else params[-self.simulator.n_species:]
        self.simulator.integrate(t, x0, method)
        ret = self.extract_data_from_simulator()
        ret = self.normalize_data(ret) if normalize else ret
        ret[np.isnan(ret)] = 0
        return (ret - x_data).flatten()

    def fit_lsq(self, t, x_data, p0=None, n_p0=1, bounds=None, sample_method='lhs', method='matrix', normalize=True):
        if p0 is None:
            p0 = self.sample_p0(n_p0, sample_method)
        else:
            if p0.ndim == 1:
                p0 = [p0]
            n_p0 = len(p0)

        x_data_norm = self.normalize_data(x_data) if normalize else x_data

        if bounds is None:
            bounds = (self.get_bound(0), self.get_bound(1))
        
        costs = np.zeros(n_p0)
        X = []
        for i in range(n_p0):
            ret = least_squares(lambda p: self.f_lsq(p, t, x_data_norm, method, normalize), p0[i], bounds=bounds)
            costs[i] = ret.cost
            X.append(ret.x)
        i_min = np.argmin(costs)
        self.popt = X[i_min]
        self.cost = costs[i_min]
        return self.popt, self.cost

class Estimation_MomentKin(Estimation):
    '''An estimation class for kinetics experiments.
        Order of species: <unspliced>, <spliced>, <uu>, <ss>, <us>
    '''
    def __init__(self, ranges, x0=None):
        super().__init__(ranges, Moments(), x0)

    def extract_data_from_simulator(self):
        ret = np.zeros((5, len(self.simulator.t)))
        ret[0] = self.simulator.get_nu()
        ret[1] = self.simulator.get_nx()
        ret[2] = self.simulator.x[:, self.simulator.uu]
        ret[3] = self.simulator.x[:, self.simulator.xx]
        ret[4] = self.simulator.x[:, self.simulator.ux]
        return ret

    def get_alpha(self):
        return self.popt[0]

    def get_beta(self):
        return self.popt[1]

    def get_gamma(self):
        return self.popt[2]

    def calc_spl_half_life(self):
        return np.log(2)/self.get_beta()

    def calc_deg_half_life(self):
        return np.log(2)/self.get_gamma()

class Estimation_MomentKinNosp(Estimation):
    '''An estimation class for kinetics experiments (without splicing).
        Order of species: <r>, <rr>
    '''
    def __init__(self, ranges, x0=None):
        super().__init__(ranges, Moments(), x0)

    def extract_data_from_simulator(self):
        ret = np.zeros((2, len(self.simulator.t)))
        ret[0] = self.simulator.get_n_labeled()
        ret[1] = self.simulator.x[:, self.simulator.uu] \
            + self.simulator.x[:, self.simulator.xx]    \
            + 2*self.simulator.x[:, self.simulator.ux]
        return ret

    def get_alpha(self):
        return self.popt[0]

    def get_beta(self):
        return self.popt[1]

    def get_gamma(self):
        return self.popt[2]

    def calc_spl_half_life(self):
        return np.log(2)/self.get_beta()

    def calc_deg_half_life(self):
        return np.log(2)/self.get_gamma()

class Estimation_MomentDeg(Estimation):
    '''An estimation class for degradation (with splicing) experiments.
        Order of species: <unspliced>, <spliced>, <uu>, <ss>, <us>
    '''
    def __init__(self, ranges, x0=None):
        super().__init__(ranges, Moments_NoSwitching(), x0)

    def get_beta(self):
        return self.popt[0]

    def get_gamma(self):
        return self.popt[1]

    def calc_spl_half_life(self):
        return np.log(2)/self.get_beta()

    def calc_deg_half_life(self):
        return np.log(2)/self.get_gamma()

class Estimation_MomentDegNosp(Estimation):
    '''An estimation class for degradation (no splicing) experiments.
        Order of species: <labeled>, <ll>
    '''
    def __init__(self, ranges, x0=None):
        super().__init__(ranges, Moments_NoSwitchingNoSplicing(), x0)

    def get_gamma(self):
        return self.popt[0]

    def calc_half_life(self):
        return np.log(2)/self.get_gamma()

class Estimation_DeterministicDeg(Estimation):
    '''An estimation class for degradation (with splicing) experiments.
        Order of species: <unspliced>, <spliced>
    '''
    def __init__(self, ranges, x0=None):
        super().__init__(ranges, Deterministic(), x0)

    def get_beta(self):
        return self.popt[0]

    def get_gamma(self):
        return self.popt[1]

    def calc_spl_half_life(self):
        return np.log(2)/self.get_beta()

    def calc_deg_half_life(self):
        return np.log(2)/self.get_gamma()
        
class Estimation_DeterministicDegNosp(Estimation):
    def __init__(self, ranges, x0=None):
        super().__init__(ranges, Deterministic_NoSplicing(), x0)

    def get_gamma(self):
        return self.popt[0]

    def calc_half_life(self):
        return np.log(2)/self.get_gamma()