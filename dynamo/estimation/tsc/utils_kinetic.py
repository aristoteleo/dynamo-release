import numpy as np
from scipy.integrate import odeint
from ..csc.utils_velocity import sol_u, sol_s

class LinearODE:
    def __init__(self, n_species, x0=None):
        '''A general class for linear odes'''
        self.n_species = n_species
        # solution
        self.t = None
        self.x = None
        self.x0 = np.zeros(self.n_species) if x0 is None else x0
        self.K = None
        self.p = None
        # methods
        self.methods = ['numerical', 'matrix']
        self.default_method = 'matrix'
        
    def ode_func(self, x, t):
        '''Implement your own ODE functions here such that dx=f(x, t)'''
        dx = np.zeros(len(x))
        return dx
    
    def integrate(self, t, x0=None, method=None):
        method = self.default_method if method is None else method
        if method == 'matrix':
            sol = self.integrate_matrix(t, x0)
        elif method == 'numerical':
            sol = self.integrate_numerical(t, x0)
        self.x = sol
        self.t = t
        return sol

    def integrate_numerical(self, t, x0=None):
        if x0 is None:
            x0 = self.x0
        else:
            self.x0 = x0
        sol = odeint(self.ode_func, x0, t)
        return sol
    
    def reset(self):
        # reset solutions
        self.t = None
        self.x = None
        self.K = None
        self.p = None
        
    def computeKnp(self):
        '''Implement your own vectorized ODE functions here such that dx = Kx + p'''
        K = np.zeros((self.n_species, self.n_species))
        p = np.zeros(self.n_species)
        return K, p
    
    def integrate_matrix(self, t, x0=None):
        #t0 = t[0]
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
        #x_ss = linalg.inv(K).dot(p)
        y0 = x0 + x_ss

        D, U = np.linalg.eig(K)
        V = np.linalg.inv(U)
        D, U, V = map(np.real, (D, U, V))
        expD = np.exp(D)
        x = np.zeros((len(t), self.n_species))
        #x[0] = x0
        for i in range(len(t)):
            x[i] = U.dot(np.diag(expD**(t[i]-t0))).dot(V).dot(y0) - x_ss
        return x

class MixtureModels:
    def __init__(self, models, param_distributor):
        '''A general class for linear odes'''
        self.n_models = len(models)
        self.models = models
        self.n_species = np.array([mdl.n_species for mdl in self.models])
        self.distributor = param_distributor
        # solution
        self.t = None
        self.x = None
        # methods
        self.methods = ['numerical', 'matrix']
        self.default_method = 'matrix'

    def integrate(self, t, x0=None, method=None):
        self.x = np.zeros((len(t), np.sum(self.n_species)))
        for i, mdl in enumerate(self.models):
            x0_ = None if x0 is None else x0[self.get_model_species(i)]
            method_ = method if method is None or type(method) is str else method[i]
            mdl.integrate(t, x0_, method_)
            self.x[:, self.get_model_species(i)] = mdl.x
        self.t = np.array(self.models[0].t, copy=True)

    def get_model_species(self, model_index):
        id = np.hstack((0, np.cumsum(self.n_species)))
        idx = np.arange(id[-1]+1)
        return idx[id[model_index] : id[model_index+1]]

    def reset(self):
        # reset solutions
        self.t = None
        self.x = None
        for mdl in self.models:
            mdl.reset()

    def param_mixer(self, *params):
        return params

    def set_params(self, *params):
        params = self.param_mixer(*params)
        for i, mdl in enumerate(self.models):
            idx = self.distributor[i]
            p = np.zeros(len(idx))
            for j in range(len(idx)):
                p[j] = params[idx[j]]
            mdl.set_params(*p)
        self.reset()

class LambdaModels_NoSwitching(MixtureModels):
    def __init__(self, model1, model2):
        '''
            parameter order: alpha, lambda, (beta), gamma
            distributor order: alpha_1, alpha_2, (beta), gamma
        '''
        models = [model1, model2]
        if type(model1) in nosplicing_models and type(model2) in nosplicing_models:
            param_distributor = [[0, 2], [1, 2]]
        else:
            dist1 = [0, 3] if model1 in nosplicing_models else [0, 2, 3]
            dist2 = [1, 3] if model2 in nosplicing_models else [1, 2, 3]
            param_distributor = [dist1, dist2]
        super().__init__(models, param_distributor)

    def param_mixer(self, *params):
        lam = params[1]
        alp_1 = params[0] * lam
        alp_2 = params[0] * (1 - lam)
        p = np.hstack((alp_1, alp_2, params[2:]))
        return p

class Moments(LinearODE):
    def __init__(self, a=None, b=None, alpha_a=None, alpha_i=None, beta=None, gamma=None, x0=None):
        """This class simulates the dynamics of first and second moments of 
        a transcription-splicing system with promoter switching."""
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
        
    def ode_func(self, x, t):
        dx = np.zeros(len(x))
        # parameters
        a = self.a
        b = self.b
        aa = self.aa
        ai = self.ai
        be = self.be
        ga = self.ga

        # first moments
        dx[self.ua] = aa - be*x[self.ua] + a*(x[self.ui]-x[self.ua])
        dx[self.ui] = ai - be*x[self.ui] - b*(x[self.ui]-x[self.ua])
        dx[self.xa] = be*x[self.ua] - ga*x[self.xa] + a*(x[self.xi]-x[self.xa])
        dx[self.xi] = be*x[self.ui] - ga*x[self.xi] - b*(x[self.xi]-x[self.xa])

        # second moments
        dx[self.uu] = 2*self.fbar(aa*x[self.ua], ai*x[self.ui]) - 2*be*x[self.uu]
        dx[self.xx] = 2*be*x[self.ux] - 2*ga*x[self.xx]
        dx[self.ux] = self.fbar(aa*x[self.xa], ai*x[self.xi]) + be*x[self.uu] - (be+ga)*x[self.ux]

        return dx

    def fbar(self, x_a, x_i):
        return self.b/(self.a + self.b) * x_a + self.a/(self.a + self.b) * x_i

    def set_params(self, a, b, alpha_a, alpha_i, beta, gamma):
        self.a = a
        self.b = b
        self.aa = alpha_a
        self.ai = alpha_i
        self.be = beta
        self.ga = gamma

        # reset solutions
        super().reset()

    def get_all_central_moments(self):
        ret = np.zeros((4, len(self.t)))
        ret[0] = self.get_nu()
        ret[1] = self.get_nx()
        ret[2] = self.get_var_nu()
        ret[3] = self.get_var_nx()
        return ret

    def get_nosplice_central_moments(self):
        ret = np.zeros((2, len(self.t)))
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
        return self.x[:, self.uu] + c - c**2

    def get_var_nx(self):
        c = self.get_nx()
        return self.x[:, self.xx] + c - c**2

    def get_cov_ux(self):
        cu = self.get_nu()
        cx = self.get_nx()
        return self.x[:, self.ux] - cu * cx

    def get_var_labeled(self):
        return self.get_var_nu() + self.get_var_nx() + 2*self.get_cov_ux()

    def computeKnp(self):
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
        K[self.uu, self.uu] = -2*be
        K[self.xx, self.xx] = -2*ga

        # E4
        K[self.ux, self.ux] = -be - ga

        # F21
        K[self.xa, self.ua] = be
        K[self.xi, self.ui] = be

        # F31
        K[self.uu, self.ua] = 2 * aa * b / (a + b)
        K[self.uu, self.ui] = 2 * ai * a / (a + b)

        # F34
        K[self.xx, self.ux] = 2*be

        # F42
        K[self.ux, self.xa] = aa*b/(a+b)
        K[self.ux, self.xi] = ai*a/(a+b)

        # F43
        K[self.ux, self.uu] = be

        p = np.zeros(self.n_species)
        p[self.ua] = aa
        p[self.ui] = ai 

        return K, p

class Moments_Nosplicing(LinearODE):
    def __init__(self, a=None, b=None, alpha_a=None, alpha_i=None, gamma=None, x0=None):
        """This class simulates the dynamics of first and second moments of 
        a transcription-splicing system with promoter switching."""
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
        
    def ode_func(self, x, t):
        dx = np.zeros(len(x))
        # parameters
        a = self.a
        b = self.b
        aa = self.aa
        ai = self.ai
        ga = self.ga

        # first moments
        dx[self.ua] = aa - ga*x[self.ua] + a*(x[self.ui]-x[self.ua])
        dx[self.ui] = ai - ga*x[self.ui] - b*(x[self.ui]-x[self.ua])

        # second moments
        dx[self.uu] = 2*self.fbar(aa*x[self.ua], ai*x[self.ui]) - 2*ga*x[self.uu]

        return dx

    def fbar(self, x_a, x_i):
        return self.b/(self.a + self.b) * x_a + self.a/(self.a + self.b) * x_i

    def set_params(self, a, b, alpha_a, alpha_i, gamma):
        self.a = a
        self.b = b
        self.aa = alpha_a
        self.ai = alpha_i
        self.ga = gamma

        # reset solutions
        super().reset()

    def get_all_central_moments(self):
        ret = np.zeros((2, len(self.t)))
        ret[0] = self.get_nu()
        ret[1] = self.get_var_nu()
        return ret

    def get_nu(self):
        return self.fbar(self.x[:, self.ua], self.x[:, self.ui])

    def get_var_nu(self):
        c = self.get_nu()
        return self.x[:, self.uu] + c - c**2

    def computeKnp(self):
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
        K[self.uu, self.uu] = -2*ga

        # F31
        K[self.uu, self.ua] = 2 * aa * b / (a + b)
        K[self.uu, self.ui] = 2 * ai * a / (a + b)

        p = np.zeros(self.n_species)
        p[self.ua] = aa
        p[self.ui] = ai 

        return K, p

class Moments_NoSwitching(LinearODE):
    def __init__(self, alpha=None, beta=None, gamma=None, x0=None):
        """This class simulates the dynamics of first and second moments of 
        a transcription-splicing system without promoter switching."""
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
        
    def ode_func(self, x, t):
        dx = np.zeros(len(x))
        # parameters
        al = self.al
        be = self.be
        ga = self.ga

        # first moments
        dx[self.u] = al - be*x[self.u]
        dx[self.s] = be*x[self.u] - ga*x[self.s]

        # second moments
        dx[self.uu] = al + 2*al*x[self.u] + be*x[self.u] - 2*be*x[self.uu]
        dx[self.us] = al*x[self.s] + be*x[self.uu] - be*x[self.us] - ga*x[self.us]
        dx[self.ss] = be*x[self.u] + 2*be*x[self.us] + ga*x[self.s] - 2*ga*x[self.ss]

        return dx

    def set_params(self, alpha, beta, gamma):
        self.al = alpha
        self.be = beta
        self.ga = gamma

        # reset solutions
        super().reset()

    def get_all_central_moments(self):
        ret = np.zeros((5, len(self.t)))
        ret[0] = self.get_mean_u()
        ret[1] = self.get_mean_s()
        ret[2] = self.get_var_u()
        ret[3] = self.get_cov_us()
        ret[4] = self.get_var_s()
        return ret

    def get_nosplice_central_moments(self):
        ret = np.zeros((2, len(self.t)))
        ret[0] = self.get_mean_u() + self.get_mean_s()
        ret[1] = self.x[:, self.uu] + self.x[:, self.ss] + 2*self.x[:, self.us]
        return ret

    def get_mean_u(self):
        return self.x[:, self.u]

    def get_mean_s(self):
        return self.x[:, self.s]

    def get_var_u(self):
        c = self.get_mean_u()
        return self.x[:, self.uu] - c**2

    def get_var_s(self):
        c = self.get_mean_s()
        return self.x[:, self.ss] - c**2

    def get_cov_us(self):
        cu = self.get_mean_u()
        cs = self.get_mean_s()
        return self.x[:, self.us] - cu * cs

    def computeKnp(self):
        # parameters
        al = self.al
        be = self.be
        ga = self.ga

        K = np.zeros((self.n_species, self.n_species))
        # E1
        K[self.u, self.u] = -be
        K[self.s, self.s] = -ga

        # E3
        K[self.uu, self.uu] = -2*be
        K[self.us, self.us] = -be - ga
        K[self.ss, self.ss] = -2*ga

        # F21
        K[self.s, self.u] = be

        # F31
        K[self.uu, self.u] = 2*al + be
        
        K[self.us, self.s] = al
        K[self.us, self.uu] = be
        
        K[self.ss, self.u] = be
        K[self.ss, self.us] = 2*be
        K[self.ss, self.s] = ga

        p = np.zeros(self.n_species)
        p[self.u] = al
        p[self.uu] = al

        return K, p

class Moments_NoSwitchingNoSplicing(LinearODE):
    def __init__(self, alpha=None, gamma=None, x0=None):
        """This class simulates the dynamics of first and second moments of 
        a transcription system without promoter switching."""
        # species
        self.u = 0
        self.uu = 1

        n_species = 2

        # solution
        super().__init__(n_species, x0)

        # parameters
        if not (alpha is None or gamma is None):
            self.set_params(alpha, gamma)
        
    def ode_func(self, x, t):
        dx = np.zeros(len(x))
        # parameters
        al = self.al
        ga = self.ga

        # first moments
        dx[self.u] = al - ga*x[self.u]

        # second moments
        dx[self.uu] = al + (2*al + ga)*x[self.u] - 2*ga*x[self.uu]

        return dx

    def set_params(self, alpha, gamma):
        self.al = alpha
        self.ga = gamma

        # reset solutions
        super().reset()

    def get_all_central_moments(self):
        ret = np.zeros((2, len(self.t)))
        ret[0] = self.get_mean_u()
        ret[1] = self.get_var_u()
        return ret

    def get_mean_u(self):
        return self.x[:, self.u]

    def get_var_u(self):
        c = self.get_mean_u()
        return self.x[:, self.uu] - c**2
    
    def computeKnp(self):
        # parameters
        al = self.al
        ga = self.ga

        K = np.zeros((self.n_species, self.n_species))
        # E1
        K[self.u, self.u] = -ga

        # E3
        K[self.uu, self.uu] = -2*ga

        # F31
        K[self.uu, self.u] = 2*al + ga

        p = np.zeros(self.n_species)
        p[self.u] = al
        p[self.uu] = al

        return K, p

class Deterministic(LinearODE):
    def __init__(self, alpha=None, beta=None, gamma=None, x0=None):
        """This class simulates the deterministic dynamics of
        a transcription-splicing system."""
        # species
        self.u = 0
        self.s = 1

        n_species = 2

        # solution
        super().__init__(n_species, x0)

        self.methods = ['numerical', 'matrix', 'analytical']
        self.default_method = 'analytical'

        # parameters
        if not (alpha is None or beta is None or gamma is None):
            self.set_params(alpha, beta, gamma)

    def ode_func(self, x, t):
        dx = np.zeros(len(x))
        # parameters
        al = self.al
        be = self.be
        ga = self.ga

        # kinetics
        dx[self.u] = al - be*x[self.u]
        dx[self.s] = be*x[self.u] - ga*x[self.s]

        return dx

    def set_params(self, alpha, beta, gamma):
        self.al = alpha
        self.be = beta
        self.ga = gamma

        # reset solutions
        super().reset()

    def integrate(self, t, x0=None, method='analytical'):
        method = self.default_method if method is None else method
        if method == 'analytical':
            sol = self.integrate_analytical(t, x0)
        else:
            sol = super().integrate(t, x0, method)
        self.x = sol
        self.t = t

    def computeKnp(self):
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

    def integrate_analytical(self, t, x0=None):
        x0 = self.x0 if x0 is None else x0
        u = sol_u(t, x0[self.u], self.al, self.be)
        s = sol_s(t, x0[self.s], x0[self.u], self.al, self.be, self.ga)
        return np.array([u, s]).T

class Deterministic_NoSplicing(LinearODE):
    def __init__(self, alpha=None, gamma=None, x0=None):
        """This class simulates the deterministic dynamics of
        a transcription-splicing system."""
        # species
        self.u = 0

        n_species = 1

        # solution
        super().__init__(n_species, x0)

        self.methods = ['numerical', 'matrix', 'analytical']
        self.default_method = 'analytical'

        # parameters
        if not (alpha is None or gamma is None):
            self.set_params(alpha, gamma)

    def ode_func(self, x, t):
        dx = np.zeros(len(x))
        # parameters
        al = self.al
        ga = self.ga

        # kinetics
        dx[self.u] = al - ga*x[self.u]

        return dx

    def set_params(self, alpha, gamma):
        self.al = alpha
        self.ga = gamma

        # reset solutions
        super().reset()

    def integrate(self, t, x0=None, method='analytical'):
        method = self.default_method if method is None else method
        if method == 'analytical':
            sol = self.integrate_analytical(t, x0)
        else:
            sol = super().integrate(t, x0, method)
        self.x = sol
        self.t = t

    def computeKnp(self):
        # parameters
        al = self.al
        ga = self.ga

        K = np.zeros((self.n_species, self.n_species))
        # E1
        K[self.u, self.u] = -ga

        p = np.zeros(self.n_species)
        p[self.u] = al

        return K, p

    def integrate_analytical(self, t, x0=None):
        x0 = self.x0 if x0 is None else x0
        u = sol_u(t, x0[self.u], self.al, self.ga)
        return np.array([u]).T

nosplicing_models = [
    Deterministic_NoSplicing, 
    Moments_Nosplicing, 
    Moments_NoSwitchingNoSplicing]
