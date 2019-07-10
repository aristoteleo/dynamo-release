import numpy as np
from scipy.optimize import least_squares
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def sol_u(t, u0, alpha, beta):
    return u0*np.exp(-beta*t) + alpha/beta*(1-np.exp(-beta*t))

def sol_s(t, s0, u0, alpha, beta, gamma):
    exp_gt = np.exp(-gamma*t)
    if beta == gamma:
        s = s0*exp_gt + (beta*u0-alpha)*t*exp_gt + alpha/gamma * (1-exp_gt)
    else:
        s = s0*exp_gt + alpha/gamma * (1-exp_gt) + (alpha - u0*beta)/(gamma-beta) * (exp_gt - np.exp(-beta*t))
    return s

def sol_p(t, p0, s0, u0, alpha, beta, gamma, eta, gamma_p):
    u = sol_u(t, u0, alpha, beta)
    s = sol_s(t, s0, u0, alpha, beta, gamma)
    exp_gt = np.exp(-gamma_p*t)
    p = p0*exp_gt + eta/(gamma_p-gamma)*(s-s0*exp_gt - beta/(gamma_p-beta)*(u-u0*exp_gt-alpha/gamma_p*(1-exp_gt)))
    return p, s, u

def fit_linreg(x, y, intercept=True):
    mask = np.logical_and(~np.isnan(x), ~np.isnan(y))
    xx = x[mask]
    yy = y[mask]
    ym = np.mean(yy)
    xm = np.mean(xx)

    if intercept:
        cov = np.mean(xx * yy) - xm * ym
        var_x = np.mean(xx * xx) - xm * xm
        k = cov / var_x
        b = ym - k * xm
    else:
        k = np.mean(yy) / np.mean(xx)
        b = 0
    return k, b

def fit_beta_lsq(t, l, bounds=(0, np.inf), fix_l0=False, beta_0=None):
    tau = t - np.min(t)
    l0 = np.mean(l[:, tau == 0])
    if beta_0 is None: beta_0 = 1

    if fix_l0:
        f_lsq = lambda b: (sol_u(tau, l0, 0, b) - l).flatten()
        ret = least_squares(f_lsq, beta_0, bounds=bounds)
        beta = ret.x
    else:
        f_lsq = lambda p: (sol_u(tau, p[1], 0, p[0]) - l).flatten()
        ret = least_squares(f_lsq, np.array([beta_0, l0]), bounds=bounds)
        beta = ret.x[0]
        l0 = ret.x[1]
    return beta, l0

def fit_gamma_lsq(t, s, beta, u0, bounds=(0, np.inf), fix_s0=False):
    tau = t - np.min(t)
    s0 = np.mean(s[:, tau == 0])
    g0 = beta * u0/s0

    if fix_s0:
        f_lsq = lambda g: (sol_s(tau, s0, u0, 0, beta, g) - s).flatten()
        ret = least_squares(f_lsq, g0, bounds=bounds)
        gamma = ret.x
    else:
        f_lsq = lambda p: (sol_s(tau, p[1], u0, 0, beta, p[0]) - s).flatten()
        ret = least_squares(f_lsq, np.array([g0, s0]), bounds=bounds)
        gamma = ret.x[0]
        s0 = ret.x[1]
    return gamma, s0

def fit_alpha_synthesis(t, u, beta):
    # fit alpha assuming u=0 at t=0
    expt = np.exp(-beta*t)

    # prepare x
    x = 1 - expt

    return beta * np.mean(u) / np.mean(x)

def fit_alpha_degradation(t, u, beta, mode=None):
    n = u.size
    tau = t - np.min(t)
    expt = np.exp(beta*tau)

    # prepare x
    x = expt - 1
    xm = np.mean(x)

    # prepare y
    y = u * expt
    ym = np.mean(y)

    # calculate slope
    var_x = np.mean(x**2) - xm**2
    cov = np.sum(y.dot(x)) / n - ym * xm
    k = cov / var_x

    # calculate intercept
    b = ym - k * xm if mode != 'fast' else None

    return k * beta, b

class velocity:
    def __init__(self, alpha=None, beta=None, gamma=None, eta=None, delta=None, estimation=None):
        if estimation is not None:
            self.parameters = {}
            self.parameters['alpha'] = estimation.parameters['alpha']
            self.parameters['beta'] = estimation.parameters['beta']
            self.parameters['gamma'] = estimation.parameters['gamma']
            self.parameters['eta'] = estimation.parameters['eta']
            self.parameters['delta'] = estimation.parameters['delta']
        else:
            self.parameters = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'eta': eta, 'delta': delta}

    def vel_u(self, U):
        if self.parameters['alpha'] is not None and self.parameters['beta'] is not None:
            V = self.parameters['alpha'] - (self.parameters['beta'] * U.T).T
        else:
            V = np.nan
        return V

    def vel_s(self, U, S):
        if self.parameters['beta'] is not None and self.parameters['gamma'] is not None:
            V = self.parameters['beta'] * U.T - self.parameters['gamma'] * S.T
            V = V.T
        else:
            V = np.nan
        return V

    def vel_p(self, S, P):
        if self.parameters['eta'] is not None and self.parameters['delta'] is not None:
            V = self.parameters['eta'] * S.T - self.parameters['delta'] * P.T
            V = V.T
        else:
            V = np.nan
        return V

    def get_n_cells(self):
        if self.parameters['alpha'] is not None:
            n_cells = self.parameters['alpha'].shape[1]
        else:
            n_cells = np.nan
        return n_cells

    def get_n_genes(self):
        if self.parameters['alpha'] is not None:
            n_genes = self.parameters['alpha'].shape[0]
        else:
            n_genes = np.nan
        return n_genes
    
class estimation:
    def __init__(self, U=None, Ul=None, S=None, Sl=None, P=None, t=None, experiment_type='deg', assumption_mRNA=None, assumption_protein='ss'):
        self.t = t
        self.data = {'uu': U, 'ul': Ul, 'su': S, 'sl': Sl, 'p': P}

        self.extyp = experiment_type
        self.asspt_mRNA = assumption_mRNA
        self.asspt_prot = assumption_protein
        self.parameters = {'alpha': None, 'beta': None, 'gamma': None, 'eta': None, 'delta': None}

    def fit(self, intercept=True, perc_left=5, perc_right=5, clusters=None):
        n = self.get_n_genes()
        # fit mRNA
        if self.asspt_mRNA == 'ss':
            if np.all(self._exist_data('uu', 'su')):
                self.parameters['beta'] = np.ones(n)
                gamma = np.zeros(n)
                for i in range(n):
                    U = self.data['uu'] if self.data['ul'] is None else self.data['uu'] + self.data['ul']
                    S = self.data['su'] if self.data['sl'] is None else self.data['su'] + self.data['sl']
                    gamma[i], _ = self.fit_gamma_steady_state(U, S,
                            intercept, perc_left, perc_right)
                self.parameters['gamma'] = gamma
        else:
            if self.extyp == 'deg':
                if np.all(self._exist_data('ul', 'sl')):
                    # beta & gamma estimation
                    self.parameters['beta'], self.parameters['gamma'] = self.fit_beta_gamma_lsq(self.t, self.data['ul'], self.data['sl'])
                    if self._exist_data('uu'):
                        # alpha estimation
                        alpha = np.zeros(n)
                        for i in range(n):
                            alpha[i], _ = fit_alpha_degradation(self.t, self.data['uu'][i], self.parameters['beta'][i], mode='fast')
                        self.parameters['alpha'] = alpha
            elif self.extyp == 'kin':
                if self._exist_data('ul'):
                    if not self._exist_parameter('beta'):
                        # beta & gamma estimation: only works when there're at least 2 time points
                        self.parameters['beta'], self.parameters['gamma'] = self.fit_beta_gamma_lsq(self.t, self.data['uu'], self.data['su'])
                    # alpha estimation
                    alpha = np.zeros_like(self.data['ul'])
                    for i in range(n):
                        for j in range(len(self.data['ul'][i])):
                            alpha[i, j] = fit_alpha_synthesis(self.t, self.data['ul'][i], self.parameters['beta'][i])
                    self.parameters['alpha'] = alpha
            # 'one_shot'
            elif self.extyp == 'one_shot':
                if self._exist_data('ul') and self._exist_parameter('beta'):
                    self.parameters['alpha'] = self.fit_alpha_oneshot(self.t, self.data['ul'], self.parameters['beta'], clusters)

        # fit protein
        if np.all(self._exist_data('p', 'su')):
            if self.asspt_prot == 'ss':
                self.parameters['eta'] = np.ones(n)
                delta = np.zeros(n)
                for i in range(n):
                    s = self.data['su'][i] + self.data['sl'][i] if self._exist_data('sl') else self.data['su'][i]
                    delta[i], _ = self.fit_gamma_steady_state(s, self.data['p'][i],
                            intercept, perc_left, perc_right)
                self.parameters['delta'] = delta

    def fit_gamma_steady_state(self, u, s, intercept=True, perc_left=5, perc_right=5):
        n = len(u)
        i_left = np.int(perc_left/100.0*n) if perc_left is not None else n
        i_right = np.int((100-perc_right)/100.0*n) if perc_right is not None else 0
        mask = np.zeros(n, dtype=bool)
        mask[:i_left] = mask[i_right:] = True
        return fit_linreg(s[mask], u[mask], intercept)
    
    def fit_beta_gamma_lsq(self, t, U, S):
        n = len(U)
        beta = np.zeros(n)
        gamma = np.zeros(n)
        for i in range(n):
            beta[i], u0 = fit_beta_lsq(t, U[i])
            gamma[i], _ = fit_gamma_lsq(t, S[i], beta[i], u0)
        return beta, gamma

    def fit_alpha_oneshot(self, t, U, beta, clusters=None):
        n_genes, n_cells = U.shape
        if clusters is None:
            clusters = [[i] for i in range(n_cells)]
        alpha = np.zeros((n_genes, len(clusters)))
        for i, c in enumerate(clusters):
            for j in range(n_genes):
                if len(c) > 0:
                    alpha[j, i] = fit_alpha_synthesis(t, U[j][c], beta[j])
                else:
                    alpha[j, i] = np.nan
        return alpha

    def get_n_genes(self):
        return len(self.data[self.get_exist_data_names()[0]])

    def set_parameter(self, name, value):
        if len(np.shape(value)) == 0:
            value = value * np.ones(self.get_n_genes())
        self.parameters[name] = value

    def _exist_data(self, *data_names):
        if len(data_names) == 1:
            ret = self.data[data_names[0]] is not None
        else:
            ret = np.array([self.data[k] is not None for k in data_names], dtype=bool)
        return ret

    def _exist_parameter(self, *param_names):
        if len(param_names) == 1:
            ret = self.parameters[param_names[0]] is not None
        else:
            ret = np.array([self.parameters[k] is not None for k in param_names], dtype=bool)
        return ret

    def get_exist_data_names(self):
        ret = []
        for k, v in self.data.items():
            if v is not None:
                ret.append(k)
        return ret

    