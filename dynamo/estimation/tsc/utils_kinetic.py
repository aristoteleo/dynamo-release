from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.integrate import odeint

from ...dynamo_logger import main_debug, main_info
from ..csc.utils_velocity import sol_s, sol_u


class LinearODE:
    """A general class for linear odes."""
    def __init__(self, n_species: int, x0: Optional[np.ndarray] = None):
        """Initialize the LinearODE object.

        Args:
            n_species: The number of species.
            x0: The initial condition of variable x.

        Returns:
            An instance of LinearODE.
        """
        self.n_species = n_species
        # solution
        self.t = None
        self.x = None
        self.x0 = np.zeros(self.n_species) if x0 is None else x0
        self.K = None
        self.p = None
        # methods
        self.methods = ["numerical", "matrix"]
        self.default_method = "matrix"

    def ode_func(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """ODE functions to be implemented in the derived class such that dx=f(x, t).

        Args:
            x: The variable.
            t: The array of time.

        Returns:
            The derivatives dx.
        """
        dx = np.zeros(len(x))
        return dx

    def integrate(self, t: np.ndarray, x0: Optional[np.ndarray] = None, method: Optional[str] = None) -> np.ndarray:
        """Integrate the ODE using the given time values.

        Args:
            t: Array of time values.
            x0: Array of initial conditions.
            method: The method to integrate, including "matrix" and "numerical".

        Returns:
            Array containing the integrated solution over the specified time values.
        """
        method = self.default_method if method is None else method
        if method == "matrix":
            sol = self.integrate_matrix(t, x0)
        elif method == "numerical":
            sol = self.integrate_numerical(t, x0)
        else:
            raise NotImplementedError("The LinearODE integrate operation currently not supports method: %s." % (method))
        self.x = sol
        self.t = t
        return sol

    def integrate_numerical(self, t: np.ndarray, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """Numerically integrate the ODE using the given time values.

        Args:
            t: Array of time values.
            x0: Array of initial conditions.

        Returns:
            Array containing the integrated solution over the specified time values.
        """
        if x0 is None:
            x0 = self.x0
        else:
            self.x0 = x0
        sol = odeint(self.ode_func, x0, t)
        return sol

    def reset(self) -> None:
        """Reset the ODE to initial state."""
        # reset solutions
        self.t = None
        self.x = None
        self.K = None
        self.p = None

    def computeKnp(self) -> Tuple[np.ndarray, np.ndarray]:
        """The vectorized ODE functions to be implemented in the derived class such that dx = Kx + p."""
        K = np.zeros((self.n_species, self.n_species))
        p = np.zeros(self.n_species)
        return K, p

    def integrate_matrix(self, t: np.ndarray, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """Integrate the system of ordinary differential equations (ODEs) using matrix exponential.

        Args:
            t: Array of time values.
            x0: Array of initial conditions.

        Returns:
            Array containing the integrated solution over the specified time values.
        """
        # t0 = t[0]
        t0 = 0
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
        x_ss = np.linalg.solve(K, p)
        # x_ss = linalg.inv(K).dot(p)
        y0 = x0 + x_ss

        D, U = np.linalg.eig(K)
        V = np.linalg.inv(U)
        D, U, V = map(np.real, (D, U, V))
        expD = np.exp(D)
        x = np.zeros((len(t), self.n_species))
        # x[0] = x0
        for i in range(len(t)):
            x[i] = U.dot(np.diag(expD ** (t[i] - t0))).dot(V).dot(y0) - x_ss
        return x


class MixtureModels:
    """The base class for mixture models."""
    def __init__(self, models: LinearODE, param_distributor: List):
        """Initialize the MixtureModels class.

        Args:
            models: The models to mix.
            param_distributor: The index to assign parameters.

        Returns:
            An instance of MixtureModels.
        """
        self.n_models = len(models)
        self.models = models
        self.n_species = np.array([mdl.n_species for mdl in self.models])
        self.distributor = param_distributor
        # solution
        self.t = None
        self.x = None
        # methods
        self.methods = ["numerical", "matrix"]
        self.default_method = "matrix"

    def integrate(self, t: np.ndarray, x0: Optional[np.ndarray] = None, method: Optional[Union[str, List]] = None) -> None:
        """Integrate with time values for all models.

        Args:
            t: Array of time values.
            x0: Array of initial conditions.
            method: The method or methods to integrate, including "matrix" and "numerical".
        """
        self.x = np.zeros((len(t), np.sum(self.n_species)))
        for i, mdl in enumerate(self.models):
            x0_ = None if x0 is None else x0[self.get_model_species(i)]
            method_ = method if method is None or type(method) is str else method[i]
            mdl.integrate(t, x0_, method_)
            self.x[:, self.get_model_species(i)] = mdl.x
        self.t = np.array(self.models[0].t, copy=True)

    def get_model_species(self, model_index: int) -> int:
        """Get the indices of species associated with the specified model.

        Args:
            model_index: Index of the model.

        Returns:
            Array containing the indices of species associated with the specified model.
        """
        id = np.hstack((0, np.cumsum(self.n_species)))
        idx = np.arange(id[-1] + 1)
        return idx[id[model_index] : id[model_index + 1]]

    def reset(self) -> None:
        """Reset all models."""
        # reset solutions
        self.t = None
        self.x = None
        for mdl in self.models:
            mdl.reset()

    def param_mixer(self, *params: Tuple) -> Tuple:
        """Unpack the given parameters.

        Args:
            params: Tuple of parameters.

        Returns:
            The unpacked tuple.
        """
        return params

    def set_params(self, *params: Tuple) -> None:
        """Set parameters for all models.

        Args:
            params: Tuple of parameters.
        """
        params = self.param_mixer(*params)
        for i, mdl in enumerate(self.models):
            idx = self.distributor[i]
            p = np.zeros(len(idx))
            for j in range(len(idx)):
                p[j] = params[idx[j]]
            mdl.set_params(*p)
        self.reset()


class LambdaModels_NoSwitching(MixtureModels):
    """Linear ODEs for the lambda mixture model. The order of params is:
            parameter order: alpha, lambda, (beta), gamma
            distributor order: alpha_1, alpha_2, (beta), gamma"""
    def __init__(self, model1: LinearODE, model2: LinearODE):
        """Initialize the LambdaModels_NoSwitching class.

        Args:
            model1: The first model to mix.
            model2: The second model to mix.

        Returns:
            An instance of LambdaModels_NoSwitching.
        """
        models = [model1, model2]
        if type(model1) in nosplicing_models and type(model2) in nosplicing_models:
            param_distributor = [[0, 2], [1, 2]]
        else:
            dist1 = [0, 3] if model1 in nosplicing_models else [0, 2, 3]
            dist2 = [1, 3] if model2 in nosplicing_models else [1, 2, 3]
            param_distributor = [dist1, dist2]
        super().__init__(models, param_distributor)

    def param_mixer(self, *params) -> np.ndarray:
        """Set parameters for all models.

        Args:
            params: Tuple of parameters.
        """
        lam = params[1]
        alp_1 = params[0] * lam
        alp_2 = params[0] * (1 - lam)
        p = np.hstack((alp_1, alp_2, params[2:]))
        return p


class Moments(LinearODE):
    """The class simulates the dynamics of first and second moments of a transcription-splicing system with promoter
    switching."""
    def __init__(
        self,
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        alpha_a: Optional[np.ndarray] = None,
        alpha_i: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ):
        """Initialize the Moments object.

        Args:
            a: Switching rate from active promoter state to inactive promoter state.
            b: Switching rate from inactive promoter state to active promoter state.
            alpha_a: Transcription rate for active promoter.
            alpha_i: Transcription rate for inactive promoter.
            beta: Splicing rate.
            gamma: Degradation rate.
            x0: The initial conditions.

        Returns:
            An instance of Moments.
        """
        # species
        self.ua = 0
        self.ui = 1
        self.xa = 2
        self.xi = 3
        self.uu = 4
        self.xx = 5
        self.ux = 6

        n_species = 7

        # solution
        super().__init__(n_species, x0=x0)

        # parameters
        if not (a is None or b is None or alpha_a is None or alpha_i is None or beta is None or gamma is None):
            self.set_params(a, b, alpha_a, alpha_i, beta, gamma)

    def ode_func(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """ODE functions to solve:
            dx[u_a] = alpha_a - beta * u_a + a * (u_i - u_a)
            dx[u_i] = ai - beta * u_i - b * (u_i - u_a)
            dx[x_a] = beta * u_a - ga * x_a + a * (x_i - x_a)
            dx[x_i] = beta * u_i - ga * x_i - b * (x_i - x_a)
        The second moments is calculated from the variance and covariance of variable u and x.

        Args:
            x: The variable.
            t: The array of time.

        Returns:
            The derivatives dx.
        """
        dx = np.zeros(len(x))
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
        dx[self.uu] = 2 * self.fbar(aa * x[self.ua], ai * x[self.ui]) - 2 * be * x[self.uu]
        dx[self.xx] = 2 * be * x[self.ux] - 2 * ga * x[self.xx]
        dx[self.ux] = self.fbar(aa * x[self.xa], ai * x[self.xi]) + be * x[self.uu] - (be + ga) * x[self.ux]

        return dx

    def fbar(self, x_a: np.ndarray, x_i: np.ndarray) -> np.ndarray:
        """Calculate the count of a variable by averaging active and inactive states.

        Args:
            x_a: The variable x under the active state.
            x_i: The variable x under the inactive state.

        Returns:
            The count of variable x.
        """
        return self.b / (self.a + self.b) * x_a + self.a / (self.a + self.b) * x_i

    def set_params(
        self,
        a: np.ndarray,
        b: np.ndarray,
        alpha_a: np.ndarray,
        alpha_i: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
    ) -> None:
        """Set the parameters.

        Args:
            a: Switching rate from active promoter state to inactive promoter state.
            b: Switching rate from inactive promoter state to active promoter state.
            alpha_a: Transcription rate for active promoter.
            alpha_i: Transcription rate for inactive promoter.
            beta: Splicing rate.
            gamma: Degradation rate.
        """
        self.a = a
        self.b = b
        self.aa = alpha_a
        self.ai = alpha_i
        self.be = beta
        self.ga = gamma

        # reset solutions
        super().reset()

    def get_all_central_moments(self) -> np.ndarray:
        """Get the first and second central moments for all variables.

        Returns:
            An array containing all central moments.
        """
        ret = np.zeros((4, len(self.t)))
        ret[0] = self.get_nu()
        ret[1] = self.get_nx()
        ret[2] = self.get_var_nu()
        ret[3] = self.get_var_nx()
        return ret

    def get_nosplice_central_moments(self) -> np.ndarray:
        """Get the central moments for labeled data.

        Returns:
            The central moments.
        """
        ret = np.zeros((2, len(self.t)))
        ret[0] = self.get_n_labeled()
        ret[1] = self.get_var_labeled()
        return ret

    def get_nu(self) -> np.ndarray:
        """Get the number of the variable u from the mean averaging active and inactive state.

        Returns:
            The number of the variable u.
        """
        return self.fbar(self.x[:, self.ua], self.x[:, self.ui])

    def get_nx(self) -> np.ndarray:
        """Get the number of the variable x from the mean averaging active and inactive state.

        Returns:
            The number of the variable x.
        """
        return self.fbar(self.x[:, self.xa], self.x[:, self.xi])

    def get_n_labeled(self) -> np.ndarray:
        """Get the number of the labeled data by combining the count of two variables.

        Returns:
            The number of the labeled data.
        """
        return self.get_nu() + self.get_nx()

    def get_var_nu(self) -> np.ndarray:
        """Get the variance of the variable u.

        Returns:
            The variance of the variable u.
        """
        c = self.get_nu()
        return self.x[:, self.uu] + c - c**2

    def get_var_nx(self) -> np.ndarray:
        """Get the variance of the variable x.

        Returns:
            The variance of the variable x.
        """
        c = self.get_nx()
        return self.x[:, self.xx] + c - c**2

    def get_cov_ux(self) -> np.ndarray:
        """Get the covariance of the variable x and u.

        Returns:
            The covariance.
        """
        cu = self.get_nu()
        cx = self.get_nx()
        return self.x[:, self.ux] - cu * cx

    def get_var_labeled(self) -> np.ndarray:
        """Get the variance of the labeled data by combining the variance of two variables and covariance.

        Returns:
            The variance of the labeled data.
        """
        return self.get_var_nu() + self.get_var_nx() + 2 * self.get_cov_ux()

    def computeKnp(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the K and p from ODE function such that dx = Kx + p.

        Returns:
            A tuple containing K and p.
        """
        # parameters
        a = self.a
        b = self.b
        aa = self.aa
        ai = self.ai
        be = self.be
        ga = self.ga

        K = np.zeros((self.n_species, self.n_species))
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

        p = np.zeros(self.n_species)
        p[self.ua] = aa
        p[self.ui] = ai

        return K, p


class Moments_Nosplicing(LinearODE):
    """The class simulates the dynamics of first and second moments of a transcription-splicing system with promoter
    switching."""
    def __init__(
        self,
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        alpha_a: Optional[np.ndarray] = None,
        alpha_i: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None
    ):
        """Initialize the Moments_Nosplicing object.

        Args:
            a: Switching rate from active promoter state to inactive promoter state.
            b: Switching rate from inactive promoter state to active promoter state.
            alpha_a: Transcription rate for active promoter.
            alpha_i: Transcription rate for inactive promoter.
            gamma: Degradation rate.
            x0: The initial conditions.

        Returns:
            An instance of Moments_Nosplicing.
        """
        # species
        self.ua = 0
        self.ui = 1
        self.uu = 2

        n_species = 3

        # solution
        super().__init__(n_species, x0=x0)

        # parameters
        if not (a is None or b is None or alpha_a is None or alpha_i is None or gamma is None):
            self.set_params(a, b, alpha_a, alpha_i, gamma)

    def ode_func(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """ODE functions to solve. Ignore the splicing part in the base class.

        Args:
            x: The variable.
            t: The array of time.

        Returns:
            The derivatives dx.
        """
        dx = np.zeros(len(x))
        # parameters
        a = self.a
        b = self.b
        aa = self.aa
        ai = self.ai
        ga = self.ga

        # first moments
        dx[self.ua] = aa - ga * x[self.ua] + a * (x[self.ui] - x[self.ua])
        dx[self.ui] = ai - ga * x[self.ui] - b * (x[self.ui] - x[self.ua])

        # second moments
        dx[self.uu] = 2 * self.fbar(aa * x[self.ua], ai * x[self.ui]) - 2 * ga * x[self.uu]

        return dx

    def fbar(self, x_a: np.ndarray, x_i: np.ndarray) -> np.ndarray:
        """Calculate the count of a variable by averaging active and inactive states.

        Args:
            x_a: The variable x under the active state.
            x_i: The variable x under the inactive state.

        Returns:
            The count of variable x.
        """
        return self.b / (self.a + self.b) * x_a + self.a / (self.a + self.b) * x_i

    def set_params(self, a: np.ndarray, b: np.ndarray, alpha_a: np.ndarray, alpha_i: np.ndarray, gamma: np.ndarray) -> None:
        """Set the parameters.

        Args:
            a: Switching rate from active promoter state to inactive promoter state.
            b: Switching rate from inactive promoter state to active promoter state.
            alpha_a: Transcription rate for active promoter.
            alpha_i: Transcription rate for inactive promoter.
            gamma: Degradation rate.
        """
        self.a = a
        self.b = b
        self.aa = alpha_a
        self.ai = alpha_i
        self.ga = gamma

        # reset solutions
        super().reset()

    def get_all_central_moments(self) -> np.ndarray:
        """Get the first and second central moments for all variables.

        Returns:
            An array containing all central moments.
        """
        ret = np.zeros((2, len(self.t)))
        ret[0] = self.get_nu()
        ret[1] = self.get_var_nu()
        return ret

    def get_nu(self) -> np.ndarray:
        """Get the number of the variable u from the mean averaging active and inactive state.

        Returns:
            The number of the variable u.
        """
        return self.fbar(self.x[:, self.ua], self.x[:, self.ui])

    def get_var_nu(self) -> np.ndarray:
        """Get the variance of the variable u.

        Returns:
            The variance of the variable u.
        """
        c = self.get_nu()
        return self.x[:, self.uu] + c - c**2

    def computeKnp(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the K and p from ODE function such that dx = Kx + p.

        Returns:
            A tuple containing K and p.
        """
        # parameters
        a = self.a
        b = self.b
        aa = self.aa
        ai = self.ai
        ga = self.ga

        K = np.zeros((self.n_species, self.n_species))
        # E1
        K[self.ua, self.ua] = -ga - a
        K[self.ua, self.ui] = a
        K[self.ui, self.ua] = b
        K[self.ui, self.ui] = -ga - b

        # E3
        K[self.uu, self.uu] = -2 * ga

        # F31
        K[self.uu, self.ua] = 2 * aa * b / (a + b)
        K[self.uu, self.ui] = 2 * ai * a / (a + b)

        p = np.zeros(self.n_species)
        p[self.ua] = aa
        p[self.ui] = ai

        return K, p


class Moments_NoSwitching(LinearODE):
    """The class simulates the dynamics of first and second moments of a transcription-splicing system without promoter
    switching."""
    def __init__(
        self,
        alpha: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ):
        """Initialize the Moments_NoSwitching object.

        Args:
            alpha: Transcription rate.
            beta: Splicing rate.
            gamma: Degradation rate.
            x0: The initial conditions.

        Returns:
            An instance of Moments_NoSwitching.
        """
        # species
        self.u = 0
        self.s = 1
        self.uu = 2
        self.ss = 3
        self.us = 4

        n_species = 5

        # solution
        super().__init__(n_species, x0)

        # parameters
        if not (alpha is None or beta is None or gamma is None):
            self.set_params(alpha, beta, gamma)

    def ode_func(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """ODE functions to solve. Ignore the switching part in the base class.

        Args:
            x: The variable.
            t: The array of time.

        Returns:
            The derivatives dx.
        """
        dx = np.zeros(len(x))
        # parameters
        al = self.al
        be = self.be
        ga = self.ga

        # first moments
        dx[self.u] = al - be * x[self.u]
        dx[self.s] = be * x[self.u] - ga * x[self.s]

        # second moments
        dx[self.uu] = al + 2 * al * x[self.u] + be * x[self.u] - 2 * be * x[self.uu]
        dx[self.us] = al * x[self.s] - be * x[self.u] + be * x[self.uu] - (be + ga) * x[self.us]
        dx[self.ss] = be * x[self.u] + 2 * be * x[self.us] + ga * x[self.s] - 2 * ga * x[self.ss]

        return dx

    def set_params(self, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> None:
        """Set the parameters.

        Args:
            alpha: Transcription rate.
            beta: Splicing rate.
            gamma: Degradation rate.
        """
        self.al = alpha
        self.be = beta
        self.ga = gamma

        # reset solutions
        super().reset()

    def get_all_central_moments(self) -> np.ndarray:
        """Get the first and second central moments for all variables.

        Returns:
            An array containing all central moments.
        """
        ret = np.zeros((5, len(self.t)))
        ret[0] = self.get_mean_u()
        ret[1] = self.get_mean_s()
        ret[2] = self.get_var_u()
        ret[3] = self.get_cov_us()
        ret[4] = self.get_var_s()
        return ret

    def get_nosplice_central_moments(self) -> np.ndarray:
        """Get the central moments for labeled data.

        Returns:
            The central moments.
        """
        ret = np.zeros((2, len(self.t)))
        ret[0] = self.get_mean_u() + self.get_mean_s()
        ret[1] = self.x[:, self.uu] + self.x[:, self.ss] + 2 * self.x[:, self.us]
        return ret

    def get_mean_u(self) -> np.ndarray:
        """Get the mean of the variable u.

        Returns:
            The mean of the variable u.
        """
        return self.x[:, self.u]

    def get_mean_s(self) -> np.ndarray:
        """Get the mean of the variable s.

        Returns:
            The mean of the variable s.
        """
        return self.x[:, self.s]

    def get_var_u(self) -> np.ndarray:
        """Get the variance of the variable u.

        Returns:
            The variance of the variable u.
        """
        c = self.get_mean_u()
        return self.x[:, self.uu] - c**2

    def get_var_s(self) -> np.ndarray:
        """Get the variance of the variable s.

        Returns:
            The variance of the variable s.
        """
        c = self.get_mean_s()
        return self.x[:, self.ss] - c**2

    def get_cov_us(self) -> np.ndarray:
        """Get the covariance of the variable s and u.

        Returns:
            The covariance.
        """
        cu = self.get_mean_u()
        cs = self.get_mean_s()
        return self.x[:, self.us] - cu * cs

    def computeKnp(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the K and p from ODE function such that dx = Kx + p.

        Returns:
            A tuple containing K and p.
        """
        # parameters
        al = self.al
        be = self.be
        ga = self.ga

        K = np.zeros((self.n_species, self.n_species))
        # E1
        K[self.u, self.u] = -be
        K[self.s, self.s] = -ga

        # E3
        K[self.uu, self.uu] = -2 * be
        K[self.us, self.us] = -be - ga
        K[self.ss, self.ss] = -2 * ga

        # F21
        K[self.s, self.u] = be

        # F31
        K[self.uu, self.u] = 2 * al + be

        K[self.us, self.s] = al
        K[self.us, self.uu] = be

        K[self.ss, self.u] = be
        K[self.ss, self.us] = 2 * be
        K[self.ss, self.s] = ga

        p = np.zeros(self.n_species)
        p[self.u] = al
        p[self.uu] = al

        return K, p


class Moments_NoSwitchingNoSplicing(LinearODE):
    """The class simulates the dynamics of first and second moments of a transcription system without promoter
    switching."""
    def __init__(
        self,
        alpha: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ):
        """Initialize the Moments_NoSwitchingNoSplicing object.

        Args:
            alpha: Transcription rate.
            gamma: Degradation rate.
            x0: The initial conditions.

        Returns:
            An instance of Moments_NoSwitchingNoSplicing.
        """
        # species
        self.u = 0
        self.uu = 1

        n_species = 2

        # solution
        super().__init__(n_species, x0)

        # parameters
        if not (alpha is None or gamma is None):
            self.set_params(alpha, gamma)

    def ode_func(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """ODE functions to solve. Both splicing and switching part in the base class are ignored.

        Args:
            x: The variable.
            t: The array of time.

        Returns:
            The derivatives dx.
        """
        dx = np.zeros(len(x))
        # parameters
        al = self.al
        ga = self.ga

        # first moments
        dx[self.u] = al - ga * x[self.u]

        # second moments
        dx[self.uu] = al + (2 * al + ga) * x[self.u] - 2 * ga * x[self.uu]

        return dx

    def set_params(self, alpha: np.ndarray, gamma: np.ndarray) -> None:
        """Set the parameters.

        Args:
            alpha: Transcription rate.
            gamma: Degradation rate.
        """
        self.al = alpha
        self.ga = gamma

        # reset solutions
        super().reset()

    def get_all_central_moments(self) -> np.ndarray:
        """Get the first and second central moments for all variables.

        Returns:
            An array containing all central moments.
        """
        ret = np.zeros((2, len(self.t)))
        ret[0] = self.get_mean_u()
        ret[1] = self.get_var_u()
        return ret

    def get_mean_u(self) -> np.ndarray:
        """Get the mean of the variable u.

        Returns:
            The mean of the variable u.
        """
        return self.x[:, self.u]

    def get_var_u(self) -> np.ndarray:
        """Get the variance of the variable u.

        Returns:
            The variance of the variable u.
        """
        c = self.get_mean_u()
        return self.x[:, self.uu] - c**2

    def computeKnp(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the K and p from ODE function such that dx = Kx + p.

        Returns:
            A tuple containing K and p.
        """
        # parameters
        al = self.al
        ga = self.ga

        K = np.zeros((self.n_species, self.n_species))
        # E1
        K[self.u, self.u] = -ga

        # E3
        K[self.uu, self.uu] = -2 * ga

        # F31
        K[self.uu, self.u] = 2 * al + ga

        p = np.zeros(self.n_species)
        p[self.u] = al
        p[self.uu] = al

        return K, p


class Deterministic(LinearODE):
    """This class simulates the deterministic dynamics of a transcription-splicing system."""
    def __init__(
        self,
        alpha: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ):
        """Initialize the Deterministic object.

        Args:
            alpha: The transcription rate.
            beta: The splicing rate.
            gamma: The degradation rate.
            x0: The initial conditions.

        Returns:
            An instance of Deterministic.
        """
        # species
        self.u = 0
        self.s = 1

        n_species = 2

        # solution
        super().__init__(n_species, x0)

        self.methods = ["numerical", "matrix", "analytical"]
        self.default_method = "analytical"

        # parameters
        if not (alpha is None or beta is None or gamma is None):
            self.set_params(alpha, beta, gamma)

    def ode_func(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """The ODE functions to solve:
            dx[u] = alpha - beta * u
            dx[v] = beta * u - gamma * s

        Args:
            x: The x variable.
            t: The time information.

        Returns:
            An array containing the ODEs' output.
        """
        dx = np.zeros(len(x))
        # parameters
        al = self.al
        be = self.be
        ga = self.ga

        # kinetics
        dx[self.u] = al - be * x[self.u]
        dx[self.s] = be * x[self.u] - ga * x[self.s]

        return dx

    def set_params(self, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> None:
        """Set the parameters.

        Args:
            alpha: The transcription rate.
            beta: The splicing rate.
            gamma: The degradation rate.
        """
        self.al = alpha
        self.be = beta
        self.ga = gamma

        # reset solutions
        super().reset()

    def integrate(self, t: np.ndarray, x0: Optional[np.ndarray] = None, method: str = "analytical") -> None:
        """Integrate the ODE using the given time values.

        Args:
            t: Array of time values.
            x0: Array of initial conditions.
            method: The method to integrate, including "matrix" and "numerical".
        """
        method = self.default_method if method is None else method
        if method == "analytical":
            sol = self.integrate_analytical(t, x0)
        else:
            sol = super().integrate(t, x0, method)
        self.x = sol
        self.t = t

    def computeKnp(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the K and p from ODE function such that dx = Kx + p.

        Returns:
            A tuple containing K and p.
        """
        # parameters
        al = self.al
        be = self.be
        ga = self.ga

        K = np.zeros((self.n_species, self.n_species))
        # E1
        K[self.u, self.u] = -be
        K[self.s, self.s] = -ga

        # F21
        K[self.s, self.u] = be

        p = np.zeros(self.n_species)
        p[self.u] = al

        return K, p

    def integrate_analytical(self, t: np.ndarray, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """Integrate the odes with the analytical solution.

        Args:
            t: The time information.
            x0: The initial conditions.

        Returns:
            The solution of unspliced and spliced mRNA wrapped in an array.
        """
        if x0 is None:
            x0 = self.x0
        else:
            self.x0 = x0
        u = sol_u(t, x0[self.u], self.al, self.be)
        s = sol_s(t, x0[self.s], x0[self.u], self.al, self.be, self.ga)
        return np.array([u, s]).T


class Deterministic_NoSplicing(LinearODE):
    """The class simulates the deterministic dynamics of a transcription-splicing system."""
    def __init__(
        self,
        alpha: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ):
        """Initialize the Deterministic_NoSplicing object.

        Args:
            alpha: The transcription rate.
            gamma: The degradation rate.
            x0: The initial conditions.

        Returns:
            An instance of Deterministic_NoSplicing.
        """
        # species
        self.u = 0

        n_species = 1

        # solution
        super().__init__(n_species, x0)

        self.methods = ["numerical", "matrix", "analytical"]
        self.default_method = "analytical"

        # parameters
        if not (alpha is None or gamma is None):
            self.set_params(alpha, gamma)

    def ode_func(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """The ODE functions to solve:
            dx[u] = alpha - gamma * u

        Args:
            x: The x variable.
            t: The time information.

        Returns:
            An array containing the ODEs' output.
        """
        dx = np.zeros(len(x))
        # parameters
        al = self.al
        ga = self.ga

        # kinetics
        dx[self.u] = al - ga * x[self.u]

        return dx

    def set_params(self, alpha: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """Set the parameters.

        Args:
            alpha: The transcription rate.
            gamma: The degradation rate.
        """
        self.al = alpha
        self.ga = gamma

        # reset solutions
        super().reset()

    def integrate(self, t: np.ndarray, x0: Optional[np.ndarray] = None, method: str = "analytical") -> None:
        """Integrate the ODE using the given time values.

        Args:
            t: Array of time values.
            x0: Array of initial conditions.
            method: The method to integrate, including "matrix" and "numerical".
        """
        method = self.default_method if method is None else method
        if method == "analytical":
            sol = self.integrate_analytical(t, x0)
        else:
            sol = super().integrate(t, x0, method)
        self.x = sol
        self.t = t

    def computeKnp(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the K and p from ODE function such that dx = Kx + p.

        Returns:
            A tuple containing K and p.
        """
        # parameters
        al = self.al
        ga = self.ga

        K = np.zeros((self.n_species, self.n_species))
        # E1
        K[self.u, self.u] = -ga

        p = np.zeros(self.n_species)
        p[self.u] = al

        return K, p

    def integrate_analytical(self, t: np.ndarray, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """Integrate the odes with the analytical solution.

        Args:
            t: The time information.
            x0: The initial conditions.

        Returns:
            The solution of unspliced mRNA as an array.
        """
        x0 = self.x0 if x0 is None else x0
        x0 = x0 if np.isscalar(x0) else x0[self.u]
        if self.x0 is None:
            self.x0 = x0
        u = sol_u(t, x0, self.al, self.ga)
        return np.array([u]).T


class KineticChase:
    def __init__(self, alpha=None, gamma=None, x0=None):
        """This class simulates the deterministic dynamics of
        a transcription-splicing system."""
        # species
        self.u = 0
        self.n_species = 1

        self.x0 = x0 if x0 is not None else 0

        self.methods = ["analytical"]
        self.default_method = "analytical"

        # parameters
        self.params = {}
        if not (alpha is None or gamma is None):
            self.set_params(alpha, gamma)

    def set_params(self, alpha, gamma):
        self.params["alpha"] = alpha
        self.params["gamma"] = gamma

    def integrate(self, t, x0=None, method="analytical"):
        if x0 is None:
            x0 = self.x0
        else:
            self.x0 = x0
        al = self.params["alpha"]
        ga = self.params["gamma"]
        self.t = t
        self.x = al / ga * (np.exp(-ga * t) - 1) + x0

    def calc_init_conc(self, t=None):
        if t is not None:
            self.integrate(t)
        h = self.x / np.exp(-self.params["gamma"] * self.t)
        tau = np.max(self.t) - self.t
        return tau, h


nosplicing_models = [
    Deterministic_NoSplicing,
    Moments_Nosplicing,
    Moments_NoSwitchingNoSplicing,
]
