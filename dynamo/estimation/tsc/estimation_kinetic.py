import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.stats import chi2

from ...dynamo_logger import main_warning
from ...tools.moments import strat_mom
from ...tools.sampling import lhsclassic
from .ODEs import *


class kinetic_estimation:
    """A general parameter estimation framework for all types of time-seris data."""

    def __init__(self, param_ranges: np.ndarray, x0_ranges: np.ndarray, simulator: LinearODE):
        """Initialize the kinetic_estimation class.

        Args:
            param_ranges: A n-by-2 numpy array containing the lower and upper ranges of n parameters (and initial
                conditions if not fixed).
            x0_ranges: Lower and upper bounds for initial conditions for the integrators. To fix a parameter, set its
                lower and upper bounds to the same value.
            simulator: An instance of python class which solves ODEs. It should have properties 't' (k time points,
                1d numpy array), 'x0' (initial conditions for m species, 1d numpy array), and 'x' (solution, k-by-m
                array), as well as two functions: integrate (numerical integration), solve (analytical method).

        Returns:
            An instance of the kinetic_estimation class.
        """
        self.simulator = simulator

        self.ranges = []
        self.fixed_parameters = np.ones(len(param_ranges) + len(x0_ranges)) * np.nan
        for i in range(len(param_ranges)):
            if param_ranges[i][0] == param_ranges[i][1]:
                self.fixed_parameters[i] = param_ranges[i][0]
            else:
                self.ranges.append(param_ranges[i])
        self.n_tot_kin_params = len(param_ranges)  # the total number of kinetic parameters
        self.n_kin_params = len(self.ranges)  # the number of unfixed kinetic parameters

        for i in range(len(x0_ranges)):
            if x0_ranges[i][0] == x0_ranges[i][1]:
                self.fixed_parameters[i + self.n_tot_kin_params] = x0_ranges[i][0]
            else:
                self.ranges.append(x0_ranges[i])
        self.n_params = len(self.ranges)  # the number of unfixed parameters (including initial conditions)

        self.popt = None
        self.cost = None

    def sample_p0(self, samples: int = 1, method: str = "lhs") -> np.ndarray:
        """Sample the initial parameters with either Latin Hypercube Sampling or random method.

        Args:
            samples: The number of samples.
            method: The sampling method. Only support "lhs" and random sampling.

        Returns:
            The sampled array.
        """
        ret = np.zeros((samples, self.n_params))
        if method == "lhs":
            ret = self._lhsclassic(samples)
            for i in range(self.n_params):
                ret[:, i] = ret[:, i] * (self.ranges[i][1] - self.ranges[i][0]) + self.ranges[i][0]
        else:
            for n in range(samples):
                for i in range(self.n_params):
                    r = np.random.rand()
                    ret[n, i] = r * (self.ranges[i][1] - self.ranges[i][0]) + self.ranges[i][0]
        return ret

    def _lhsclassic(self, samples: int) -> np.ndarray:
        """Run the Latin Hypercube Sampling function.

        Args:
            samples: The number of samples.

        Returns:
            The sampled data array.
        """
        # From PyDOE
        # Generate the intervals
        # from .utils import lhsclassic
        H = lhsclassic(samples, self.n_params)

        return H

    def get_bound(self, axis: int) -> np.ndarray:
        """Get the bounds of the specified axis for all parameters.

        Args:
            axis: The index of axis.

        Returns:
            An array containing the bounds of the specified axis for all parameters.
        """
        ret = np.zeros(self.n_params)
        for i in range(self.n_params):
            ret[i] = self.ranges[i][axis]
        return ret

    def normalize_data(self, X: np.ndarray) -> np.ndarray:
        """Perform log1p normalization on the data.

        Args:
            X: Target data to normalize.

        Returns:
            The normalized data.
        """
        return np.log1p(X)

    def extract_data_from_simulator(self, t: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Extract data from the ODE simulator.

        Args:
            t: The time information. If provided, the data will be integrated with time information.
            kwargs: Additional keyword arguments.

        Returns:
            The variable from ODE simulator.
        """
        if t is None:
            return self.simulator.x.T
        else:
            x = self.simulator.integrate(t=t, **kwargs)
            return x.T

    def assemble_kin_params(self, unfixed_params: np.ndarray) -> np.ndarray:
        """Assemble the kinetic parameters array.

        Args:
            unfixed_params: Array of unfixed parameters.

        Returns:
            The assembled kinetic parameters.
        """
        p = np.array(self.fixed_parameters[: self.n_tot_kin_params], copy=True)
        p[np.isnan(p)] = unfixed_params[: self.n_kin_params]
        return p

    def assemble_x0(self, unfixed_params: np.ndarray) -> np.ndarray:
        """Assemble the initial conditions array.

        Args:
            unfixed_params: Array of unfixed parameters.

        Returns:
            The assembled initial conditions.
        """
        p = np.array(self.fixed_parameters[self.n_tot_kin_params :], copy=True)
        p[np.isnan(p)] = unfixed_params[self.n_kin_params :]
        return p

    def set_params(self, params: np.ndarray) -> None:
        """Set the parameters of the simulator using assembled kinetic parameters.

        Args:
            params: Array of assembled kinetic parameters.
        """
        self.simulator.set_params(*self.assemble_kin_params(params))

    def get_opt_kin_params(self) -> Optional[np.ndarray]:
        """Get the optimized kinetic parameters.

        Returns:
            Array containing the optimized kinetic parameters, or None if not available.
        """
        if self.popt is not None:
            return self.assemble_kin_params(self.popt)
        else:
            return None

    def get_opt_x0_params(self) -> Optional[np.ndarray]:
        """Get the optimized initial conditions.

        Returns:
            Array containing the optimized initial conditions, or None if not available.
        """
        if self.popt is not None:
            return self.assemble_x0(self.popt)
        else:
            return None

    def f_lsq(
        self,
        params: np.ndarray,
        t: np.ndarray,
        x_data: np.ndarray,
        method: Optional[str] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Calculate the difference between simulated and observed data for least squares fitting.

        Args:
            params: Array of parameters for the simulation.
            t: Array of time values.
            x_data: The input array.
            method: Method for integration.
            normalize: Whether to normalize data.

        Returns:
            Residuals representing the differences between simulated and observed data (flattened).
        """
        self.set_params(params)
        x0 = self.assemble_x0(params)
        self.simulator.integrate(t, x0, method)
        ret = self.extract_data_from_simulator()
        ret = self.normalize_data(ret) if normalize else ret
        ret[np.isnan(ret)] = 0
        return (ret - x_data).flatten()

    def fit_lsq(
        self,
        t: np.ndarray,
        x_data: np.ndarray,
        p0: Optional[np.ndarray] = None,
        n_p0: int = 1,
        bounds: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
        sample_method: str = "lhs",
        method: Optional[str] = None,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """Fit time-seris data using the least squares method.

        This method iteratively optimizes the parameters for different initial conditions (p0) and returns
        the optimized parameters and associated cost.

        Args:
            t: A numpy array of n time points.
            x_data: An m-by-n numpy array of m species, each having n values for the n time points.
            p0: Initial guesses of parameters. If None, a random number is generated within the bounds.
            n_p0: Number of initial guesses.
            bounds: Lower and upper bounds for parameters.
            sample_method: Method used for sampling initial guesses of parameters:
                `lhs`: Latin hypercube sampling;
                `uniform`: Uniform random sampling.
            method: Method used for solving ODEs. See options in simulator classes.
            normalize: Whether to normalize values in x_data across species, so that large values do
                not dominate the optimizer.

        Returns:
            Optimal parameters and the cost function evaluated at the optimum.
        """
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
            ret = least_squares(
                lambda p: self.f_lsq(p, t, x_data_norm, method, normalize),
                p0[i],
                bounds=bounds,
            )
            costs[i] = ret.cost
            X.append(ret.x)
        i_min = np.argmin(costs)
        self.popt = X[i_min]
        self.cost = costs[i_min]
        return self.popt, self.cost

    def export_parameters(self) -> Optional[np.ndarray]:
        """Export the optimized kinetic parameters.

        Returns:
            Array containing the optimized kinetic parameters, or None if not available.
        """
        return self.get_opt_kin_params()

    def export_model(self, reinstantiate: bool = True) -> LinearODE:
        """Export the simulator model.

        Args:
            reinstantiate: Whether to reinstantiate the model class (default: True).

        Returns:
            Exported simulator model.
        """
        if reinstantiate:
            return self.simulator.__class__()
        else:
            return self.simulator

    def get_SSE(self) -> float:
        """Get the sum of squared errors (SSE) from the least squares fitting.

        Returns:
            Sum of squared errors (SSE).
        """
        return self.cost

    def test_chi2(
        self,
        t: np.ndarray,
        x_data: np.ndarray,
        species: Optional[Union[List, np.ndarray]] = None,
        method: str = "matrix",
        normalize: bool = True,
    ) -> Tuple[float, float, int]:
        """Perform a Pearson's chi-square test. The statistics is computed as: sum_i (O_i - E_i)^2 / E_i, where O_i is
        the data and E_i is the model predication.

        The data can be either:
            1. stratified moments: 't' is an array of k distinct time points, 'x_data' is an m-by-k matrix of data,
                where m is the number of species.
        Or
            2. raw data: 't' is an array of k time points for k cells, 'x_data' is an m-by-k matrix of data, where m is
                the number of species. Note that if the method is 'numerical', t has to monotonically increasing.
        If not all species are included in the data, use 'species' to specify the species of interest.

        Returns:
            The p-value of a one-tailed chi-square test, the chi-square statistics and degree of freedom.
        """
        if x_data.ndim == 1:
            x_data = x_data[None]

        self.simulator.integrate(t, method=method)
        x_model = self.simulator.x.T
        if species is not None:
            x_model = x_model[species]

        if normalize:
            scale = np.max(x_data, 1)
            x_data_norm = (x_data.T / scale).T
            x_model_norm = (x_model.T / scale).T
        else:
            x_data_norm = x_data
            x_model_norm = x_model
        c2 = np.sum((x_data_norm - x_model_norm) ** 2 / x_model_norm)
        # df = len(x_data.flatten()) - self.n_params - 1
        df = len(np.unique(t)) - self.n_params - 1
        p = 1 - chi2.cdf(c2, df)
        return p, c2, df


class Estimation_Degradation(kinetic_estimation):
    """The base parameters, estimation class for degradation experiments."""

    def __init__(self, ranges: np.ndarray, x0: np.ndarray, simulator: LinearODE):
        """Initialize the Estimation_Degradation object.

        Args:
            ranges: The lower and upper ranges of parameters.
            x0: Initial conditions.
            simulator: Instance of the Python class to solve ODEs.

        Returns:
            An instance of the Estimation_Degradation class.
        """
        self.kin_param_keys = np.array(["alpha", "gamma"])
        super().__init__(np.vstack((np.zeros(2), ranges)), x0, simulator)

    def guestimate_init_cond(self, x_data: np.ndarray) -> np.ndarray:
        """Roughly estimate initial conditions for parameter estimation.

        Args:
            x_data: A matrix representing RNA data.

        Returns:
            Estimated initial conditions.
        """
        return guestimate_init_cond(x_data)

    def guestimate_gamma(self, x_data: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Roughly estimate initial conditions for parameter estimation.

        Args:
            x_data: A matrix representing RNA data.
            time: A matrix of time information.

        Returns:
            Estimated gamma.
        """
        return guestimate_gamma(x_data, time)

    def get_param(self, key: str) -> np.ndarray:
        """Get the estimated parameter value by key.

        Args:
            key: The key of parameter.

        Returns:
            The estimated parameter value.
        """
        return self.popt[np.where(self.kin_param_keys == key)[0][0]]

    def calc_half_life(self, key: str) -> np.ndarray:
        """Calculate half-life of a parameter.

        Args:
            key: The key of parameter.

        Returns:
            The half-life value.
        """
        return np.log(2) / self.get_param(key)

    def export_dictionary(self) -> Dict:
        """Export parameter estimation results as a dictionary.

        Returns:
            Dictionary containing model name, kinetic parameters, and initial conditions.
        """
        mdl_name = type(self.simulator).__name__
        params = self.export_parameters()
        param_dict = {self.kin_param_keys[i]: params[i] for i in range(len(params))}
        x0 = self.get_opt_x0_params()
        dictionary = {
            "model": mdl_name,
            "kinetic_parameters": param_dict,
            "x0": x0,
        }
        return dictionary


class Estimation_DeterministicDeg(Estimation_Degradation):
    """An estimation class for degradation (with splicing) experiments.
    Order of species: <unspliced>, <spliced>
    """

    def __init__(
        self,
        beta: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ):
        """Initialize the Estimation_DeterministicDeg object.

        Args:
            beta: The splicing rate.
            gamma: The degradation rate.
            x0: The initial conditions.

        Returns:
            An instance of the Estimation_DeterministicDeg class.
        """
        self.kin_param_keys = np.array(["alpha", "beta", "gamma"])
        if beta is not None and gamma is not None and x0 is not None:
            self._initialize(beta, gamma, x0)

    def _initialize(self, beta: np.ndarray, gamma: np.ndarray, x0: np.ndarray) -> None:
        """Initialize the parameters to the default value.

        Args:
            beta: The splicing rate.
            gamma: The degradation rate.
            x0: The initial conditions.
        """
        ranges = np.zeros((2, 2))
        ranges[0] = beta * np.ones(2) if np.isscalar(beta) else beta
        ranges[1] = gamma * np.ones(2) if np.isscalar(gamma) else gamma
        super().__init__(ranges, x0, Deterministic())

    def auto_fit(self, time: np.ndarray, x_data: np.ndarray, **kwargs) -> Tuple[np.ndarray, float]:
        """Estimate the parameters.

        Args:
            time: The time information.
            x_data: A matrix representing RNA data.
            kwargs: The additional keyword arguments.

        Returns:
            The optimized parameters and the cost.
        """
        be0 = self.guestimate_gamma(x_data[0, :], time)
        ga0 = self.guestimate_gamma(x_data[0, :] + x_data[1, :], time)
        x0 = self.guestimate_init_cond(x_data)
        beta_bound = np.array([0, 1e2 * be0])
        gamma_bound = np.array([0, 1e2 * ga0])
        x0_bound = np.hstack((np.zeros((len(x0), 1)), 1e2 * x0[None].T))
        self._initialize(beta_bound, gamma_bound, x0_bound)

        popt, cost = self.fit_lsq(time, x_data, p0=np.hstack((be0, ga0, x0)), **kwargs)
        return popt, cost


class Estimation_DeterministicDegNosp(Estimation_Degradation):
    """An estimation class for degradation (without splicing) experiments."""

    def __init__(self, gamma: Optional[np.ndarray] = None, x0: Optional[np.ndarray] = None):
        """Initialize the Estimation_DeterministicDegNosp object.

        Args:
            gamma: The degradation rate.
            x0: The initial conditions.

        Returns:
            An instance of the Estimation_DeterministicDegNosp class.
        """
        if gamma is not None and x0 is not None:
            self._initialize(gamma, x0)

    def _initialize(self, gamma: np.ndarray, x0: np.ndarray) -> None:
        """Initialize the parameters to the default value.

        Args:
            gamma: The degradation rate.
            x0: The initial conditions.
        """
        ranges = gamma * np.ones(2) if np.isscalar(gamma) else gamma
        if np.isscalar(x0) or x0.ndim > 1:
            x0_ = x0
        else:
            x0_ = np.array([x0])
        super().__init__(ranges, x0_, Deterministic_NoSplicing())

    def auto_fit(
        self,
        time: np.ndarray,
        x_data: np.ndarray,
        sample_method: str = "lhs",
        method: Optional[str] = None,
        normalize: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Estimate the parameters.

        Args:
            time: The time information.
            x_data: A matrix representing RNA data.
            sample_method: Method used for sampling initial guesses of parameters:
                `lhs`: Latin hypercube sampling;
                `uniform`: Uniform random sampling.
            method: Method used for solving ODEs. See options in simulator classes.
            normalize: Whether to normalize the data.

        Returns:
            The optimized parameters and the cost.
        """
        ga0 = self.guestimate_gamma(x_data, time)
        x0 = self.guestimate_init_cond(x_data[None])
        gamma_bound = np.array([0, 1e2 * ga0])
        x0_bound = np.array([0, 1e2 * x0])
        self._initialize(gamma_bound, x0_bound)

        popt, cost = self.fit_lsq(
            time,
            x_data,
            p0=np.hstack((ga0, x0)),
            sample_method=sample_method,
            method=method,
            normalize=normalize,
        )
        return popt, cost


class Estimation_MomentDeg(Estimation_DeterministicDeg):
    """An estimation class for degradation (with splicing) experiments.
    Order of species: <unspliced>, <spliced>, <uu>, <ss>, <us>
    Order of parameters: beta, gamma
    """

    def __init__(
        self,
        beta: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
        include_cov: bool = True,
    ):
        """Initialize the Estimation_MomentDeg object.

        Args:
            beta: The splicing rate.
            gamma: The degradation rate.
            x0: The initial conditions.
            include_cov: Whether to consider covariance when estimating.

        Returns:
            An instance of the Estimation_MomentDeg class.
        """
        self.kin_param_keys = np.array(["alpha", "beta", "gamma"])
        self.include_cov = include_cov
        if beta is not None and gamma is not None and x0 is not None:
            self._initialize(beta, gamma, x0)

    def _initialize(self, beta: np.ndarray, gamma: np.ndarray, x0: np.ndarray) -> None:
        """Initialize the parameters to the default value.

        Args:
            beta: The splicing rate.
            gamma: The degradation rate.
            x0: The initial conditions.
        """
        ranges = np.zeros((2, 2))
        ranges[0] = beta * np.ones(2) if np.isscalar(beta) else beta
        ranges[1] = gamma * np.ones(2) if np.isscalar(gamma) else gamma
        super(Estimation_DeterministicDeg, self).__init__(ranges, x0, Moments_NoSwitching())

    def extract_data_from_simulator(self) -> np.ndarray:
        """Get corresponding data from the LinearODE class."""
        if self.include_cov:
            ret = np.zeros((5, len(self.simulator.t)))
            ret[0] = self.simulator.x[:, self.simulator.u]
            ret[1] = self.simulator.x[:, self.simulator.s]
            ret[2] = self.simulator.x[:, self.simulator.uu]
            ret[3] = self.simulator.x[:, self.simulator.ss]
            ret[4] = self.simulator.x[:, self.simulator.us]
        else:
            ret = np.zeros((4, len(self.simulator.t)))
            ret[0] = self.simulator.x[:, self.simulator.u]
            ret[1] = self.simulator.x[:, self.simulator.s]
            ret[2] = self.simulator.x[:, self.simulator.uu]
            ret[3] = self.simulator.x[:, self.simulator.ss]
        return ret


class Estimation_MomentDegNosp(Estimation_Degradation):
    """An estimation class for degradation (without splicing) experiments. Order of species: <r>, <rr>."""

    def __init__(self, gamma: Optional[np.ndarray] = None, x0: Optional[np.ndarray] = None):
        """Initialize the Estimation_MomentDeg object.

        Args:
            gamma: The degradation rate.
            x0: The initial conditions.

        Returns:
            An instance of the Estimation_MomentDeg class.
        """
        if gamma is not None and x0 is not None:
            self._initialize(gamma, x0)

    def _initialize(self, gamma: np.ndarray, x0: np.ndarray) -> None:
        """Initialize the parameters to the default value.

        Args:
            gamma: The degradation rate.
            x0: The initial conditions.
        """
        ranges = gamma * np.ones(2) if np.isscalar(gamma) else gamma
        super().__init__(ranges, x0, Moments_NoSwitchingNoSplicing())

    def auto_fit(
        self,
        time: np.ndarray,
        x_data: np.ndarray,
        sample_method: str = "lhs",
        method: Optional[str] = None,
        normalize: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Estimate the parameters.

        Args:
            time: The time information.
            x_data: A matrix representing RNA data.
            sample_method: Method used for sampling initial guesses of parameters:
                `lhs`: Latin hypercube sampling;
                `uniform`: Uniform random sampling.
            method: Method used for solving ODEs. See options in simulator classes.
            normalize: Whether to normalize the data.

        Returns:
            The optimized parameters and the cost.
        """
        ga0 = self.guestimate_gamma(x_data[0, :], time)
        x0 = self.guestimate_init_cond(x_data)
        gamma_bound = np.array([0, 1e2 * ga0])
        x0_bound = np.hstack((np.zeros((len(x0), 1)), 1e2 * x0[None].T))
        self._initialize(gamma_bound, x0_bound)

        popt, cost = self.fit_lsq(
            time,
            x_data,
            p0=np.hstack((ga0, x0)),
            sample_method=sample_method,
            method=method,
            normalize=normalize,
        )
        return popt, cost


class Estimation_MomentKin(kinetic_estimation):
    """An estimation class for kinetics experiments.
    Order of species: <unspliced>, <spliced>, <uu>, <ss>, <us>
    """

    def __init__(
        self,
        a: np.ndarray,
        b: np.ndarray,
        alpha_a: np.ndarray,
        alpha_i: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        include_cov: bool = True,
    ):
        """Initialize the Estimation_MomentKin object.

        Args:
            a: Switching rate from active promoter state to inactive promoter state.
            b: Switching rate from inactive promoter state to active promoter state.
            alpha_a: Transcription rate for active promoter.
            alpha_i: Transcription rate for inactive promoter.
            beta: Splicing rate.
            gamma: Degradation rate.
            include_cov: Whether to include the covariance when estimating.

        Returns:
            An instance of the Estimation_MomentKin class.
        """
        self.param_keys = np.array(["a", "b", "alpha_a", "alpha_i", "beta", "gamma"])
        ranges = np.zeros((6, 2))
        ranges[0] = a * np.ones(2) if np.isscalar(a) else a
        ranges[1] = b * np.ones(2) if np.isscalar(b) else b
        ranges[2] = alpha_a * np.ones(2) if np.isscalar(alpha_a) else alpha_a
        ranges[3] = alpha_i * np.ones(2) if np.isscalar(alpha_i) else alpha_i
        ranges[4] = beta * np.ones(2) if np.isscalar(beta) else beta
        ranges[5] = gamma * np.ones(2) if np.isscalar(gamma) else gamma
        super().__init__(ranges, np.zeros((7, 2)), Moments())
        self.include_cov = include_cov

    def extract_data_from_simulator(self) -> np.ndarray:
        """Get corresponding data from the LinearODE class.

        Returns:
            The variable from ODE simulator as an array.
        """
        if self.include_cov:
            ret = np.zeros((5, len(self.simulator.t)))
            ret[0] = self.simulator.get_nu()
            ret[1] = self.simulator.get_nx()
            ret[2] = self.simulator.x[:, self.simulator.uu]
            ret[3] = self.simulator.x[:, self.simulator.xx]
            ret[4] = self.simulator.x[:, self.simulator.ux]
        else:
            ret = np.zeros((4, len(self.simulator.t)))
            ret[0] = self.simulator.get_nu()
            ret[1] = self.simulator.get_nx()
            ret[2] = self.simulator.x[:, self.simulator.uu]
            ret[3] = self.simulator.x[:, self.simulator.xx]
        return ret

    def get_alpha_a(self) -> np.ndarray:
        """Get the transcription rate for active promoter.

        Returns:
            The transcription rate for active promoter.
        """
        return self.popt[2]

    def get_alpha_i(self) -> np.ndarray:
        """Get the transcription rate for inactive promoter.

        Returns:
            The transcription rate for inactive promoter.
        """
        return self.popt[3]

    def get_alpha(self) -> np.ndarray:
        """Get all transcription rates.

        Returns:
            All transcription rates.
        """
        alpha = self.simulator.fbar(self.get_alpha_a(), self.get_alpha_i())
        return alpha

    def get_beta(self) -> np.ndarray:
        """Get the splicing rate.

        Returns:
            The splicing rate.
        """
        return self.popt[4]

    def get_gamma(self) -> np.ndarray:
        """Get the degradation rate.

        Returns:
            The degradation rate.
        """
        return self.popt[5]

    def calc_spl_half_life(self) -> np.ndarray:
        """Calculate the half life of splicing.

        Returns:
            The half life of splicing.
        """
        return np.log(2) / self.get_beta()

    def calc_deg_half_life(self) -> np.ndarray:
        """Calculate the half life of degradation.

        Returns:
            The half life of degradation.
        """
        return np.log(2) / self.get_gamma()

    def export_dictionary(self) -> Dict:
        """Export parameter estimation results as a dictionary.

        Returns:
            Dictionary containing model name, kinetic parameters, and initial conditions.
        """
        mdl_name = type(self.simulator).__name__
        params = self.export_parameters()
        param_dict = {self.param_keys[i]: params[i] for i in range(len(params))}
        x0 = np.zeros(self.simulator.n_species)
        dictionary = {
            "model": mdl_name,
            "kinetic_parameters": param_dict,
            "x0": x0,
        }
        return dictionary


class Estimation_MomentKinNosp(kinetic_estimation):
    """An estimation class for kinetics experiments.
    Order of species: <r>, <rr>
    """

    def __init__(
        self,
        a: np.ndarray,
        b: np.ndarray,
        alpha_a: np.ndarray,
        alpha_i: np.ndarray,
        gamma: np.ndarray,
    ):
        """Initialize the Estimation_MomentKinNosp object.

        Args:
            a: Switching rate from active promoter state to inactive promoter state.
            b: Switching rate from inactive promoter state to active promoter state.
            alpha_a: Transcription rate for active promoter.
            alpha_i: Transcription rate for inactive promoter.
            gamma: Degradation rate.

        Returns:
            An instance of the Estimation_MomentKinNosp class.
        """
        self.param_keys = np.array(["a", "b", "alpha_a", "alpha_i", "gamma"])
        ranges = np.zeros((5, 2))
        ranges[0] = a * np.ones(2) if np.isscalar(a) else a
        ranges[1] = b * np.ones(2) if np.isscalar(b) else b
        ranges[2] = alpha_a * np.ones(2) if np.isscalar(alpha_a) else alpha_a
        ranges[3] = alpha_i * np.ones(2) if np.isscalar(alpha_i) else alpha_i
        ranges[4] = gamma * np.ones(2) if np.isscalar(gamma) else gamma
        super().__init__(ranges, np.zeros((3, 2)), Moments_Nosplicing())

    def extract_data_from_simulator(self) -> np.ndarray:
        """Get corresponding data from the LinearODE class.

        Returns:
            The variable from ODE simulator as an array.
        """
        ret = np.zeros((2, len(self.simulator.t)))
        ret[0] = self.simulator.get_nu()
        ret[1] = self.simulator.x[:, self.simulator.uu]
        return ret

    def get_alpha_a(self) -> np.ndarray:
        """Get the transcription rate for active promoter.

        Returns:
            The transcription rate for active promoter.
        """
        return self.popt[2]

    def get_alpha_i(self) -> np.ndarray:
        """Get the transcription rate for inactive promoter.

        Returns:
            The transcription rate for inactive promoter.
        """
        return self.popt[3]

    def get_alpha(self) -> np.ndarray:
        """Get all transcription rates.

        Returns:
            All transcription rates.
        """
        alpha = self.simulator.fbar(self.get_alpha_a().self.get_alpha_i())
        return alpha

    def get_gamma(self) -> np.ndarray:
        """Get the degradation rate.

        Returns:
            The degradation rate.
        """
        return self.popt[4]

    def calc_deg_half_life(self) -> np.ndarray:
        """Calculate the half life of degradation.

        Returns:
            The half life of degradation.
        """
        return np.log(2) / self.get_gamma()

    def export_dictionary(self) -> Dict:
        """Export parameter estimation results as a dictionary.

        Returns:
            Dictionary containing model name, kinetic parameters, and initial conditions.
        """
        mdl_name = type(self.simulator).__name__
        params = self.export_parameters()
        param_dict = {self.param_keys[i]: params[i] for i in range(len(params))}
        x0 = np.zeros(self.simulator.n_species)
        dictionary = {
            "model": mdl_name,
            "kinetic_parameters": param_dict,
            "x0": x0,
        }
        return dictionary


class Estimation_DeterministicKinNosp(kinetic_estimation):
    """An estimation class for kinetics (without splicing) experiments with the deterministic model.
    Order of species: <unspliced>, <spliced>
    """

    def __init__(
        self,
        alpha: np.ndarray,
        gamma: np.ndarray,
        x0: Union[int, np.ndarray] = 0,
    ):
        """Initialize the Estimation_DeterministicKinNosp object.

        Args:
            alpha: Transcription rate.
            gamma: Degradation rate.
            x0: The initial condition.

        Returns:
            An instance of the Estimation_DeterministicKinNosp class.
        """
        self.param_keys = np.array(["alpha", "gamma"])
        ranges = np.zeros((2, 2))
        ranges[0] = alpha * np.ones(2) if np.isscalar(alpha) else alpha
        ranges[1] = gamma * np.ones(2) if np.isscalar(gamma) else gamma
        if np.isscalar(x0):
            x0 = np.ones((1, 2)) * x0
        super().__init__(ranges, x0, Deterministic_NoSplicing())

    def get_alpha(self) -> np.ndarray:
        """Get the transcription rate.

        Returns:
            The transcription rate.
        """
        return self.popt[0]

    def get_gamma(self) -> np.ndarray:
        """Get the degradation rate.

        Returns:
            The degradation rate.
        """
        return self.popt[1]

    def calc_half_life(self, key: str) -> np.ndarray:
        """Calculate the half life."""
        return np.log(2) / self.get_param(key)

    def export_dictionary(self) -> Dict:
        """Export parameter estimation results as a dictionary.

        Returns:
            Dictionary containing model name, kinetic parameters, and initial conditions.
        """
        mdl_name = type(self.simulator).__name__
        params = self.export_parameters()
        param_dict = {self.param_keys[i]: params[i] for i in range(len(params))}
        x0 = np.zeros(self.simulator.n_species)
        dictionary = {
            "model": mdl_name,
            "kinetic_parameters": param_dict,
            "x0": x0,
        }
        return dictionary


class Estimation_DeterministicKin(kinetic_estimation):
    """An estimation class for kinetics experiments with the deterministic model.
    Order of species: <unspliced>, <spliced>
    """

    def __init__(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        x0: Union[int, np.ndarray] = np.zeros(2),
    ):
        """Initialize the Estimation_DeterministicKin object.

        Args:
            alpha: Transcription rate.
            beta: Splicing rate.
            gamma: Degradation rate.
            x0: The initial condition.

        Returns:
            An instance of the Estimation_DeterministicKin class.
        """
        self.param_keys = np.array(["alpha", "beta", "gamma"])
        ranges = np.zeros((3, 2))
        ranges[0] = alpha * np.ones(2) if np.isscalar(alpha) else alpha
        ranges[1] = beta * np.ones(2) if np.isscalar(beta) else beta
        ranges[2] = gamma * np.ones(2) if np.isscalar(gamma) else gamma

        if x0.ndim == 1:
            x0 = np.vstack((x0, x0)).T

        super().__init__(ranges, x0, Deterministic())

    def get_alpha(self) -> np.ndarray:
        """Get the transcription rate.

        Returns:
            The transcription rate.
        """
        return self.popt[0]

    def get_beta(self) -> np.ndarray:
        """Get the splicing rate.

        Returns:
            The splicing rate.
        """
        return self.popt[1]

    def get_gamma(self) -> np.ndarray:
        """Get the degradation rate.

        Returns:
            The degradation rate.
        """
        return self.popt[2]

    def calc_spl_half_life(self) -> np.ndarray:
        """Calculate the half life of splicing.

        Returns:
            The half life of splicing.
        """
        return np.log(2) / self.get_beta()

    def calc_deg_half_life(self) -> np.ndarray:
        """Calculate the half life of degradation.

        Returns:
            The half life of degradation.
        """
        return np.log(2) / self.get_gamma()

    def export_dictionary(self) -> Dict:
        """Export parameter estimation results as a dictionary.

        Returns:
            Dictionary containing model name, kinetic parameters, and initial conditions.
        """
        mdl_name = type(self.simulator).__name__
        params = self.export_parameters()
        param_dict = {self.param_keys[i]: params[i] for i in range(len(params))}
        x0 = np.zeros(self.simulator.n_species)
        dictionary = {
            "model": mdl_name,
            "kinetic_parameters": param_dict,
            "x0": x0,
        }
        return dictionary


class Mixture_KinDeg_NoSwitching(kinetic_estimation):
    """An estimation class with the mixture model.
    If beta is None, it is assumed that the data does not have the splicing process.
    """

    def __init__(
        self,
        model1: LinearODE,
        model2: LinearODE,
        alpha: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
        x0: Optional[Union[int, np.ndarray]] = None,
        beta: Optional[np.ndarray] = None,
    ):
        """Initialize the Mixture_KinDeg_NoSwitching object.

        Args:
            model1: The first model to mix.
            model2: The second model to mix.
            alpha: Transcription rate.
            gamma: Degradation rate.
            x0: The initial condition.
            beta: Splicing rate.

        Returns:
            An instance of the Mixture_KinDeg_NoSwitching class.
        """
        self.model1 = model1
        self.model2 = model2
        self.scale = 1
        if alpha is not None and gamma is not None:
            self._initialize(alpha, gamma, x0, beta)

    def _initialize(
        self,
        alpha: np.ndarray,
        gamma: np.ndarray,
        x0: Union[int, np.ndarray],
        beta: Optional[np.ndarray] = None,
    ):
        """Initialize the parameters to the default value.

        Args:
            alpha: Transcription rate.
            gamma: Degradation rate.
            x0: The initial condition.
            beta: Splicing rate.

        Returns:
            An instance of the Mixture_KinDeg_NoSwitching class.
        """
        if type(self.model1) in nosplicing_models:
            self.param_distributor = [[0, 2], [1, 2]]
            self.param_keys = ["alpha", "alpha_2", "gamma"]
        else:
            self.param_distributor = [[0, 2, 3], [1, 2, 3]]
            self.param_keys = ["alpha", "alpha_2", "beta", "gamma"]
        self.param_distributor = [[0, 2], [1, 2]] if type(self.model1) in nosplicing_models else [[0, 2, 3], [1, 2, 3]]
        model = MixtureModels([self.model1, self.model2], self.param_distributor)

        ranges = np.zeros((3, 2)) if beta is None else np.zeros((4, 2))
        ranges[0] = alpha
        if beta is None:
            ranges[2] = gamma
        else:
            ranges[2] = beta
            ranges[3] = gamma
        x0_ = np.vstack((np.zeros((self.model1.n_species, 2)), x0))
        super().__init__(ranges, x0_, model)

    def normalize_deg_data(
        self, x_data: Union[csr_matrix, np.ndarray], weight: Union[float, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize the degradation data while preserving the relative proportions between species. It calculates
        scaling factors to ensure the data's range remains within a certain limit.

        Args:
            x_data: A matrix representing RNA data.
            weight: Weight for scaling.

        Returns:
            A tuple containing the normalized degradation data and the scaling factor.
        """
        x_data_norm = np.array(x_data, copy=True)

        x_data_kin = x_data_norm[: self.model1.n_species, :]
        data_max = np.max(np.sum(x_data_kin, 0))

        x_deg_data = x_data_norm[self.model1.n_species :, :]
        scale = np.clip(weight * np.max(x_deg_data) / data_max, 1e-6, None)
        x_data_norm[self.model1.n_species :, :] /= scale

        return x_data_norm, scale

    def auto_fit(
        self,
        time: np.ndarray,
        x_data: Union[csr_matrix, np.ndarray],
        alpha_min: Union[float, int] = 0.1,
        beta_min: Union[float, int] = 50,
        gamma_min: Union[float, int] = 10,
        kin_weight: Union[float, int] = 2,
        use_p0: bool = True,
        **kwargs,
    ) -> Tuple[np.ndarray, float]:
        """Estimate the parameters.

        Args:
            time: The time information.
            x_data: A matrix representing RNA data.
            alpha_min: The minimum limitation on transcription rate.
            beta_min: The minimum limitation on splicing rate.
            gamma_min: The minimum limitation on degradation rate.
            kin_weight: Weight for scaling during normalization.
            use_p0: Whether to use initial parameters when estimating.
            kwargs: The additional keyword arguments.

        Returns:
            The optimized parameters and the cost.
        """
        if kin_weight is not None:
            x_data_norm, self.scale = self.normalize_deg_data(x_data, kin_weight)
        else:
            x_data_norm = x_data

        x0 = guestimate_init_cond(x_data_norm[-self.model2.n_species :, :])
        x0_bound = np.hstack((np.zeros((len(x0), 1)), 1e2 * x0[None].T))

        if type(self.model1) in nosplicing_models:
            al0 = guestimate_alpha(x_data_norm[0, :], time)
        else:
            al0 = guestimate_alpha(x_data_norm[0, :] + x_data_norm[1, :], time)
        alpha_bound = np.array([0, max(1e2 * al0, alpha_min)])

        if type(self.model2) in nosplicing_models:
            ga0 = guestimate_gamma(x_data_norm[self.model1.n_species, :], time)
            p0 = np.hstack((al0, ga0, x0))
            beta_bound = None
        else:
            be0 = guestimate_gamma(x_data_norm[self.model1.n_species, :], time)
            ga0 = guestimate_gamma(
                x_data_norm[self.model1.n_species, :] + x_data_norm[self.model1.n_species + 1, :],
                time,
            )
            p0 = np.hstack((al0, be0, ga0, x0))
            beta_bound = np.array([0, max(1e2 * be0, beta_min)])
        gamma_bound = np.array([0, max(1e2 * ga0, gamma_min)])

        self._initialize(alpha_bound, gamma_bound, x0_bound, beta_bound)

        if use_p0:
            popt, cost = self.fit_lsq(time, x_data_norm, p0=p0, **kwargs)
        else:
            popt, cost = self.fit_lsq(time, x_data_norm, **kwargs)
        return popt, cost

    def export_model(self, reinstantiate: bool = True) -> Union[MixtureModels, LinearODE]:
        """Export the mixture model.

        Args:
            reinstantiate: Whether to reinstantiate the model.

        Returns:
            MixtureModels or LinearODE.
        """
        if reinstantiate:
            return MixtureModels([self.model1, self.model2], self.param_distributor)
        else:
            return self.simulator

    def export_x0(self) -> Optional[np.ndarray]:
        """Export optimized initial conditions for the mixture of models analysis.

        Returns:
            Exported initial conditions.
        """
        x = self.get_opt_x0_params()
        x[self.model1.n_species :] *= self.scale
        return x

    def export_dictionary(self) -> Dict:
        """Export parameter estimation results as a dictionary.

        Returns:
            Dictionary containing model nameS, kinetic parameters, and initial conditions.
        """
        mdl1_name = type(self.model1).__name__
        mdl2_name = type(self.model2).__name__
        params = self.export_parameters()
        param_dict = {self.param_keys[i]: params[i] for i in range(len(params))}
        x0 = self.export_x0()
        dictionary = {
            "model_1": mdl1_name,
            "model_2": mdl2_name,
            "kinetic_parameters": param_dict,
            "x0": x0,
        }
        return dictionary


class Lambda_NoSwitching(Mixture_KinDeg_NoSwitching):
    """An estimation class with the mixture model. If beta is None, it is assumed that the data does not have the
    splicing process.
    """

    def __init__(
        self,
        model1: LinearODE,
        model2: LinearODE,
        alpha: Optional[np.ndarray] = None,
        lambd: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
        x0: Optional[Union[int, np.ndarray]] = None,
        beta: Optional[np.ndarray] = None,
    ):
        """Initialize the Lambda_NoSwitching object.

        Args:
            model1: The first model to mix.
            model2: The second model to mix.
            alpha: Transcription rate.
            lambd: The lambd value.
            gamma: Degradation rate.
            x0: The initial condition.
            beta: Splicing rate.

        Returns:
            An instance of the Lambda_NoSwitching class.
        """
        self.model1 = model1
        self.model2 = model2
        self.scale = 1
        if alpha is not None and gamma is not None:
            self._initialize(alpha, gamma, x0, beta)

    def _initialize(
        self,
        alpha: np.ndarray,
        gamma: np.ndarray,
        x0: Union[int, np.ndarray],
        beta: Optional[np.ndarray] = None,
    ):
        """Initialize the parameters to the default value.

        Args:
            alpha: Transcription rate.
            gamma: Degradation rate.
            x0: The initial condition.
            beta: Splicing rate.

         Returns:
            An instance of the Lambda_NoSwitching class.
        """
        if type(self.model1) in nosplicing_models and type(self.model2) in nosplicing_models:
            self.param_keys = ["alpha", "lambda", "gamma"]
        else:
            self.param_keys = ["alpha", "lambda", "beta", "gamma"]
        model = LambdaModels_NoSwitching(self.model1, self.model2)

        ranges = np.zeros((3, 2)) if beta is None else np.zeros((4, 2))
        ranges[0] = alpha
        ranges[1] = np.array([0, 1])
        if beta is None:
            ranges[2] = gamma
        else:
            ranges[2] = beta
            ranges[3] = gamma
        x0_ = np.vstack((np.zeros((self.model1.n_species, 2)), x0))
        super(Mixture_KinDeg_NoSwitching, self).__init__(ranges, x0_, model)

    def auto_fit(self, time: np.ndarray, x_data: Union[csr_matrix, np.ndarray], **kwargs) -> Tuple[np.ndarray, float]:
        """Estimate the parameters.

        Args:
            time: The time information.
            x_data: A matrix representing RNA data.
            kwargs: The additional keyword arguments.

        Returns:
            The optimized parameters and the cost.
        """
        return super().auto_fit(time, x_data, kin_weight=None, use_p0=False, **kwargs)

    def export_model(self, reinstantiate: bool = True) -> Union[LambdaModels_NoSwitching, LinearODE]:
        """Export the mixture model.

        Args:
            reinstantiate: Whether to reinstantiate the model.

        Returns:
            MixtureModels or LinearODE.
        """
        if reinstantiate:
            return LambdaModels_NoSwitching(self.model1, self.model2)
        else:
            return self.simulator


class Estimation_KineticChase(kinetic_estimation):
    """An estimation class for kinetic chase experiment."""

    def __init__(
        self,
        alpha: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
        x0: Optional[Union[int, np.ndarray]] = None,
    ):
        """Initialize the Estimation_KineticChase object.

        Args:
            alpha: Transcription rate.
            gamma: Degradation rate.
            x0: The initial condition.

        Returns:
            An instance of the Estimation_KineticChase class.
        """
        self.kin_param_keys = np.array(["alpha", "gamma"])
        if alpha is not None and gamma is not None and x0 is not None:
            self._initialize(alpha, gamma, x0)

    def _initialize(
        self,
        alpha: np.ndarray,
        gamma: np.ndarray,
        x0: Union[int, np.ndarray],
    ):
        """Initialize the parameters to the default value.

        Args:
            alpha: Transcription rate.
            gamma: Degradation rate.
            x0: The initial condition.

        Returns:
            An instance of the Estimation_KineticChase class.
        """
        ranges = np.zeros((2, 2))
        ranges[0] = alpha * np.ones(2) if np.isscalar(alpha) else alpha
        ranges[1] = gamma * np.ones(2) if np.isscalar(gamma) else gamma
        super().__init__(ranges, np.atleast_2d(x0), KineticChase())

    def auto_fit(self, time: np.ndarray, x_data: Union[csr_matrix, np.ndarray], **kwargs) -> Tuple[np.ndarray, float]:
        """Estimate the parameters.

        Args:
            time: The time information.
            x_data: A matrix representing RNA data.
            kwargs: The additional keyword arguments.

        Returns:
            The optimized parameters and the cost.
        """
        if len(time) != len(np.unique(time)):
            t = np.unique(time)
            x = strat_mom(x_data, time, np.mean)
        else:
            t, x = time, x_data
        al0, ga0, x0 = guestimate_p0_kinetic_chase(x, t)
        alpha_bound = np.array([0, 1e2 * al0 + 100])
        gamma_bound = np.array([0, 1e2 * ga0 + 100])
        x0_bound = np.array([0, 1e2 * x0 + 100])
        self._initialize(alpha_bound, gamma_bound, x0_bound)
        popt, cost = self.fit_lsq(time, x_data, p0=np.hstack((al0, ga0, x0)), normalize=False, **kwargs)
        return popt, cost

    def get_param(self, key: str) -> np.ndarray:
        """Get corresponding parameter according to the key.

        Returns:
            The corresponding parameter.
        """
        return self.popt[np.where(self.kin_param_keys == key)[0][0]]

    def get_alpha(self) -> np.ndarray:
        """Get the transcription rate.

        Returns:
            The transcription rate.
        """
        return self.popt[0]

    def get_gamma(self) -> np.ndarray:
        """Get the degradation rate.

        Returns:
            The degradation rate.
        """
        return self.popt[1]

    def calc_half_life(self, key: str) -> np.ndarray:
        """Calculate the half life.

        Returns:
            The half life.
        """
        return np.log(2) / self.get_param(key)

    def export_dictionary(self) -> Dict:
        """Export parameter estimation results as a dictionary.

        Returns:
            Dictionary containing model nameS, kinetic parameters, and initial conditions.
        """
        mdl_name = type(self.simulator).__name__
        params = self.export_parameters()
        param_dict = {self.kin_param_keys[i]: params[i] for i in range(len(params))}
        x0 = self.get_opt_x0_params()
        dictionary = {
            "model": mdl_name,
            "kinetic_parameters": param_dict,
            "x0": x0,
        }
        return dictionary


class GoodnessOfFit:
    """Evaluate goodness of fitting.

    This class provides methods for assessing the quality of predictions, using various metrics including Gaussian
    likelihood, Gaussian log-likelihood, and mean squared deviation.
    """

    def __init__(
        self,
        simulator: LinearODE,
        params: Optional[Tuple] = None,
        x0: Optional[Union[int, np.ndarray]] = None,
    ):
        """Initialize the GoodnessOfFit object.

        Args:
            simulator: The linearODE class.
            params: The parameters.
            x0: The initial conditions.

        Returns:
            An instance of the GoodnessOfFit class.
        """
        self.simulator = simulator
        if params is not None:
            self.simulator.set_params(*params)
        if x0 is not None:
            self.simulator.x0 = x0
        self.time = None
        self.mean = None
        self.sigm = None
        self.pred = None

    def extract_data_from_simulator(self, species: Optional[int] = None) -> np.ndarray:
        """Extract data from the simulator's results.

        Args:
            species: Index of the species to extract.

        Returns:
            Extracted data from the simulator's results.
        """
        ret = self.simulator.x.T
        if species is not None:
            ret = ret[species]
        return ret

    def prepare_data(
        self,
        t: np.ndarray,
        x_data: Union[csr_matrix, np.ndarray],
        species: Optional[int] = None,
        method: Optional[str] = None,
        normalize: bool = True,
        reintegrate: bool = True,
    ) -> None:
        """Prepare data for evaluation.

        Args:
            t: The time information.
            x_data: The RNA data.
            species: Index of the species to consider.
            method: Integration method.
            normalize: Whether to normalize data.
            reintegrate: Whether to reintegrate the model.
        """
        if reintegrate:
            self.simulator.integrate(t, method=method)
        x_model = self.extract_data_from_simulator(species=species)
        if x_model.ndim == 1:
            x_model = x_model[None]

        if normalize:
            mean = strat_mom(x_data.T, t, np.mean)
            scale = np.max(mean, 0)
            x_data_norm, x_model_norm = self.normalize(x_data, x_model, scale)
        else:
            x_data_norm, x_model_norm = x_data, x_model
        self.time = np.unique(t)
        self.mean = strat_mom(x_data_norm.T, t, np.mean)
        self.sigm = strat_mom(x_data_norm.T, t, np.std)
        self.pred = strat_mom(x_model_norm.T, t, np.mean)

    def normalize(
        self,
        x_data: Union[csr_matrix, np.ndarray],
        x_model: np.ndarray,
        scale: Optional[Union[float, int, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize data and model predictions.

        Args:
            x_data: The RNA data.
            x_model: Predictions from model.
            scale: Scaling factors for normalization.

        Returns:
            Normalized observations and model predictions.
        """
        scale = np.max(x_data, 1) if scale is None else scale
        x_data_norm = (x_data.T / scale).T
        x_model_norm = (x_model.T / scale).T
        return x_data_norm, x_model_norm

    def calc_gaussian_likelihood(self) -> float:
        """Calculate the Gaussian likelihood between model predictions and observations.

        Returns:
            Gaussian likelihood value.
        """
        sig = np.array(self.sigm, copy=True)
        if np.any(sig == 0):
            main_warning("Some standard deviations are 0; Set to 1 instead.")
            sig[sig == 0] = 1
        err = ((self.pred - self.mean) / sig).flatten()
        ret = 1 / (np.sqrt((2 * np.pi) ** len(err)) * np.prod(sig)) * np.exp(-0.5 * (err).dot(err))
        return ret

    def calc_gaussian_loglikelihood(self) -> float:
        """Calculate the Gaussian log-likelihood between model predictions and observations.

        Returns:
            Gaussian log-likelihood value.
        """
        sig = np.array(self.sigm, copy=True)
        if np.any(sig == 0):
            main_warning("Some standard deviations are 0; Set to 1 instead.")
            sig[sig == 0] = 1
        err = ((self.pred - self.mean) / sig).flatten()
        ret = -len(err) / 2 * np.log(2 * np.pi) - np.sum(np.log(sig)) - 0.5 * err.dot(err)
        return ret

    def calc_mean_squared_deviation(self, weighted: bool = True) -> float:
        """Calculate the mean squared deviation between model predictions and observations.

        Args:
            weighted: Whether to weight the output.

        Returns:
            Mean squared deviation.
        """
        sig = np.array(self.sigm, copy=True)
        if np.any(sig == 0):
            main_warning("Some standard deviations are 0; Set to 1 instead.")
            sig[sig == 0] = 1
        err = self.pred - self.mean
        if weighted:
            err /= sig
        return np.sqrt(err.dot(err))


def guestimate_alpha(x_data: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Roughly estimate alpha for kinetics data, which is simply the averaged ratio of new RNA and labeling time.

    Args:
        x_data: A matrix representing RNA data.
        time: A matrix of labeling time.

    Returns:
        The estimated alpha.
    """
    imax = np.argmax(x_data)
    alpha = x_data[imax] / time[imax]
    return alpha


def guestimate_gamma(x_data: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Roughly estimate gamma0 with the assumption that time starts at 0.

    Args:
        x_data: A matrix representing RNA data.
        time: A matrix of labeling time.

    Returns:
        The estimated gamma.
    """
    ga0 = np.clip(np.log(max(x_data[0], 0) / (x_data[-1] + 1e-6)) / time[-1], 1e-3, 1e3)
    return ga0


def guestimate_init_cond(x_data: np.ndarray) -> np.ndarray:
    """Roughly estimate x0 for degradation data.

    Args:
        x_data: A matrix representing RNA data.

    Returns:
        The estimated x0.
    """
    x0 = np.clip(np.max(x_data, 1), 1e-4, np.inf)
    return x0


def guestimate_p0_kinetic_chase(x_data: np.ndarray, time: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roughly estimated alpha, gamma and initial conditions for the kinetic chase experiment. Initail conditions are
    the average abundance of labeled RNAs across all cells belonging to the initial labeling time point.

    Args:
        x_data: A matrix representing RNA data.
        time: A matrix of labeling time.

    Returns:
        The estimated alpha0, gamma0 and x0.
    """
    t0 = np.min(time)
    x0 = np.mean(x_data[time == t0])
    idx = time != t0
    al0 = np.mean((x0 - x_data[idx]) / (time[idx] - t0))
    ga0 = -np.mean((np.log(x_data[idx]) - np.log(x0)) / (time[idx] - t0))
    ga0 = 1e-3 if not np.isfinite(ga0) else ga0
    x0, al0, ga0 = max(1e-3, x0), max(1e-3, al0), max(1e-3, ga0)
    return al0, ga0, x0
