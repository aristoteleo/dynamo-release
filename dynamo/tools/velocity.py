import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import issparse, csr_matrix
from warnings import warn
from .utils import cal_12_mom
from .moments import strat_mom
# from sklearn.cluster import KMeans
# from sklearn.neighbors import NearestNeighbors


def sol_u(t, u0, alpha, beta):
    """The analytical solution of unspliced mRNA kinetics.

    Arguments
    ---------
    t: :class:`~numpy.ndarray`
        A vector of time points.
    u0: float
        Initial value of u.
    alpha: float
        Transcription rate.
    beta: float
        Splicing rate constant.

    Returns
    -------
    u: :class:`~numpy.ndarray`
        Unspliced mRNA counts at given time points.
    """
    return u0*np.exp(-beta*t) + alpha/beta*(1-np.exp(-beta*t))

def sol_u_2p(t, u0, t1, alpha0, alpha1, beta):
    """The combined 2-piece analytical solution of unspliced mRNA kinetics.

    Arguments
    ---------
    t: :class:`~numpy.ndarray`
        A vector of time points for both steady state and stimulation labeling.
    u0: float
        Initial value of u.
    t1: :class:`~numpy.ndarray`
        The time point when the cells switch from steady state to stimulation.
    alpha0: float
        Transcription rate for steady state labeling.
    alpha1: float
        Transcription rate for stimulation based labeling.
    beta: float
        Splicing rate constant.

    Returns
    -------
    u: :class:`~numpy.ndarray`
        Unspliced mRNA counts at given time points.
    """
    u1 = sol_u(t1, u0, alpha0, beta)
    u_pre = sol_u(t[t <= t1], u0, alpha0, beta)
    u_aft = sol_u(t[t > t1] - t1, u1, alpha1, beta)

    return np.concatenate((u_pre, u_aft))

def sol_s(t, s0, u0, alpha, beta, gamma):
    """The analytical solution of spliced mRNA kinetics.

    Arguments
    ---------
    t: :class:`~numpy.ndarray`
        A vector of time points.
    s0: float
        Initial value of s.
    u0: float
        Initial value of u.
    alpha: float
        Transcription rate.
    beta: float
        Splicing rate constant.
    gamma: float
        Degradation rate constant for spliced mRNA.

    Returns
    -------
    s: :class:`~numpy.ndarray`
        Spliced mRNA counts at given time points.
    """
    exp_gt = np.exp(-gamma*t)
    if beta == gamma:
        s = s0*exp_gt + (beta*u0-alpha)*t*exp_gt + alpha/gamma * (1-exp_gt)
    else:
        s = s0*exp_gt + alpha/gamma * (1-exp_gt) + (alpha - u0*beta)/(gamma-beta) * (exp_gt - np.exp(-beta*t))
    return s

def sol_p(t, p0, s0, u0, alpha, beta, gamma, eta, delta):
    """The analytical solution of protein kinetics.

    Arguments
    ---------
    t: :class:`~numpy.ndarray`
        A vector of time points.
    p0: float
        Initial value of p.
    s0: float
        Initial value of s.
    u0: float
        Initial value of u.
    alpha: float
        Transcription rate.
    beta: float
        Splicing rate constant.
    gamma: float
        Degradation rate constant for spliced mRNA.
    eta: float
        Synthesis rate constant for protein.
    delta: float
        Degradation rate constant for protein.

    Returns
    -------
    p: :class:`~numpy.ndarray`
        Protein counts at given time points.
    s: :class:`~numpy.ndarray`
        Spliced mRNA counts at given time points.
    u: :class:`~numpy.ndarray`
        Unspliced mRNA counts at given time points.
    """
    u = sol_u(t, u0, alpha, beta)
    s = sol_s(t, s0, u0, alpha, beta, gamma)
    exp_gt = np.exp(-delta*t)
    p = p0*exp_gt + eta/(delta-gamma)*(s-s0*exp_gt - beta/(delta-beta)*(u-u0*exp_gt-alpha/delta*(1-exp_gt)))
    return p, s, u

def solve_gamma(t, old, total):
    """ Analytical solution to calculate gamma (degradation rate) using first-order degradation kinetics.

    Parameters
    ----------
    t: `float`
        Metabolic labeling time period.
    old: :class:`~numpy.ndarray` or sparse `csr_matrix`
        A vector of old RNA amount in each cell
    total: :class:`~numpy.ndarray` or sparse `csr_matrix`
        A vector of total RNA amount in each cell

    Returns
    -------
    Returns the degradation rate (gamma)
    """

    old, total = np.mean(old), np.mean(total)
    gamma = - 1/t * np.log(old / total)

    return gamma

def solve_alpha_2p(t0, t1, alpha0, beta, u1):
    """Given known steady state alpha and beta, solve stimulation alpha for a mixed steady state and stimulation labeling experiment.

    Parameters
    ----------
    t0: `float`
        Time period for steady state labeling.
    t1: `float`
        Time period for stimulation labeling.
    alpha0: `float`
        steady state transcription rate calculated from one-shot experiment mode.
    beta: `float`
        steady state (and simulation) splicing rate calculated from one-shot experiment mode.
    u1: :class:`~numpy.ndarray` or sparse `csr_matrix`
        A vector of labeled RNA amount in each cell observed at time t0 + t1.

    Returns
    -------
    Returns the transcription rate (alpha1) for the stimulation period in the data.
    """

    u1 = np.mean(u1)

    u0 = alpha0 / beta * (1 - np.exp(- beta * t0))
    alpha1 = beta * (u1 - u0 * np.exp(-beta * t1)) / (1 - np.exp(-beta * t1))

    return alpha1

def fit_linreg(x, y, intercept=False):
    """Simple linear regression: y = kx + b.

    Arguments
    ---------
    x: :class:`~numpy.ndarray` or sparse `csr_matrix`
        A vector of independent variables.
    y: :class:`~numpy.ndarray` or sparse `csr_matrix`
        A vector of dependent variables.
    intercept: bool
        If using steady state assumption for fitting, then:
        True -- the linear regression is performed with an unfixed intercept;
        False -- the linear regresssion is performed with a fixed zero intercept.

    Returns
    -------
    k: float
        The estimated slope.
    b: float
        The estimated intercept.
    r2: float
        Coefficient of determination or r square.
    """
    x = x.A if issparse(x) else x
    y = y.A if issparse(y) else y

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
        # assume b is always positive
        if b < 0: k, b = np.mean(xx * yy) / np.mean(xx * xx), 0
    else:
        # use uncentered cov and var_x
        cov = np.mean(xx * yy)
        var_x = np.mean(xx * xx)
        k = cov / var_x
        b = 0

    SS_tot_n = np.var(yy)
    SS_res_n = np.mean((yy - k * xx - b) ** 2)
    r2 = 1 - SS_res_n / SS_tot_n
    return k, b, r2

def fit_first_order_deg_lsq(t, l, bounds=(0, np.inf), fix_l0=False, beta_0=1):
    """Estimate beta with degradation data using least squares method.

    Arguments
    ---------
    t: :class:`~numpy.ndarray`
        A vector of time points.
    l: :class:`~numpy.ndarray` or sparse `csr_matrix`
        A vector of unspliced, labeled mRNA counts for each time point.
    bounds: tuple
        The bound for beta. The default is beta > 0.
    fixed_l0: bool
        True: l0 will be calculated by averaging the first column of l;
        False: l0 is a parameter that will be estimated all together with beta using lsq.
    beta_0: float
        Initial guess for beta.

    Returns
    -------
    beta: float
        The estimated value for beta.
    l0: float
        The estimated value for the initial spliced, labeled mRNA count.
    """
    l = l.A.flatten() if issparse(l) else l

    tau = t - np.min(t)
    l0 = np.nanmean(l[tau == 0])

    if fix_l0:
        f_lsq = lambda b: sol_u(tau, l0, 0, b) - l
        ret = least_squares(f_lsq, beta_0, bounds=bounds)
        beta = ret.x
    else:
        f_lsq = lambda p: sol_u(tau, p[1], 0, p[0]) - l
        ret = least_squares(f_lsq, np.array([beta_0, l0]), bounds=bounds)
        beta = ret.x[0]
        l0 = ret.x[1]
    return beta, l0

def solve_first_order_deg(t, l):
    """Solve for the initial amount and the rate constant of a species (for example, labeled mRNA) with time-series data
    under first-order degration kinetics model.

    Parameters
    ----------
    t: :class:`~numpy.ndarray`
        A vector of time points.
    l: :class:`~numpy.ndarray` or sparse `csr_matrix`
        A vector of labeled mRNA counts for each time point.

    Returns
    -------
    l0:: `float`
        The intial counts of the species  (for example, labeled mRNA).
    k: `float`
        Degradation rate constant.
    half_life: `float`
        Half-life the species.
    """

    x = l.A.flatten() if issparse(l) else l

    t_uniq = np.unique(t)
    x_stra = strat_mom(x, t, np.nanmean)
    x_stra_ = x_stra[1:] / x_stra[0]
    k = np.mean(- np.log(x_stra_) / t_uniq[1:])

    l0, half_life = np.mean(x_stra / np.exp(- k * t_uniq)), np.log(2) / k

    return l0, k, half_life

def fit_gamma_lsq(t, s, beta, u0, bounds=(0, np.inf), fix_s0=False):
    """Estimate gamma with degradation data using least squares method.

    Arguments
    ---------
    t: :class:`~numpy.ndarray`
        A vector of time points.
    s: :class:`~numpy.ndarray` or sparse `csr_matrix`
        A vector of spliced, labeled mRNA counts for each time point.
    beta: float
        The value of beta.
    u0: float
        Initial number of unspliced mRNA.
    bounds: tuple
        The bound for gamma. The default is gamma > 0.
    fixed_s0: bool
        True: s0 will be calculated by averaging the first column of s;
        False: s0 is a parameter that will be estimated all together with gamma using lsq.

    Returns
    -------
    gamma: float
        The estimated value for gamma.
    s0: float
        The estimated value for the initial spliced mRNA count.
    """
    s = s.A.flatten() if issparse(s) else s

    tau = t - np.min(t)
    s0 = np.mean(s[tau == 0])
    g0 = beta * u0/s0

    if fix_s0:
        f_lsq = lambda g: sol_s(tau, s0, u0, 0, beta, g) - s
        ret = least_squares(f_lsq, g0, bounds=bounds)
        gamma = ret.x
    else:
        if np.isfinite(g0):
            f_lsq = lambda p: sol_s(tau, p[1], u0, 0, beta, p[0]) - s
            ret = least_squares(f_lsq, np.array([g0, s0]), bounds=bounds)
            gamma = ret.x[0]
            s0 = ret.x[1]
        else:
            gamma, s0 = np.nan, 0
    return gamma, s0

def fit_alpha_synthesis(t, u, beta):
    """Estimate alpha with synthesis data using linear regression with fixed zero intercept.
    It is assumed that u(0) = 0.

    Arguments
    ---------
    t: :class:`~numpy.ndarray`
        A vector of time points.
    u: :class:`~numpy.ndarray` or sparse `csr_matrix`
        A matrix of unspliced mRNA counts. Dimension: cells x time points.
    beta: float
        The value of beta.

    Returns
    -------
    alpha: float
        The estimated value for alpha.
    """
    u = u.A if issparse(u) else u

    # fit alpha assuming u=0 at t=0
    expt = np.exp(-beta*t)

    # prepare x
    x = 1 - expt

    return beta * np.mean(u) / np.mean(x)

def fit_alpha_degradation(t, u, beta, intercept=False):
    """Estimate alpha with degradation data using linear regression. This is a lsq version of the following function that
    constrains u0 to be larger than 0

    Arguments
    ---------
    t: :class:`~numpy.ndarray`
        A vector of time points.
    u: :class:`~numpy.ndarray` or sparse `csr_matrix`
        A matrix of unspliced mRNA counts. Dimension: cells x time points.
    beta: float
        The value of beta.
    intercept: bool
        If using steady state assumption for fitting, then:
        True -- the linear regression is performed with an unfixed intercept;
        False -- the linear regresssion is performed with a fixed zero intercept.

    Returns
    -------
    alpha: float
        The estimated value for alpha.
    u0: float
        The initial unspliced mRNA count.
    r2: float
        Coefficient of determination or r square.
    """
    x = u.A if issparse(u) else u

    tau = t - np.min(t)

    f_lsq = lambda p: sol_u(tau, p[0], p[1], beta) - x
    ret = least_squares(f_lsq, np.array([1, 1]), bounds=(0, np.inf))
    alpha, u0 = ret.x[0], ret.x[1]

    # calculate r-squared
    SS_tot_n = np.var(x)
    SS_res_n = np.mean((f_lsq([alpha, u0])) ** 2)
    r2 = 1 - SS_res_n / SS_tot_n

    return alpha, u0, r2

def solve_alpha_degradation(t, u, beta, intercept=False):
    """Estimate alpha with degradation data using linear regression.

    Arguments
    ---------
    t: :class:`~numpy.ndarray`
        A vector of time points.
    u: :class:`~numpy.ndarray` or sparse `csr_matrix`
        A matrix of unspliced mRNA counts. Dimension: cells x time points.
    beta: float
        The value of beta.
    intercept: bool
        If using steady state assumption for fitting, then:
        True -- the linear regression is performed with an unfixed intercept;
        False -- the linear regresssion is performed with a fixed zero intercept.

    Returns
    -------
    alpha: float
        The estimated value for alpha.
    b: float
        The initial unspliced mRNA count.
    r2: float
        Coefficient of determination or r square.
    """
    u = u.A if issparse(u) else u

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
    b = ym - k * xm if intercept else 0
    SS_tot_n = np.var(y)
    SS_res_n = np.mean((y - k * x - b) ** 2) if b is not None else np.mean((y - k * x) ** 2)
    r2 = 1 - SS_res_n / SS_tot_n

    return k * beta, b, r2

def fit_alpha_beta_synthesis(t, l, bounds=(0, np.inf), alpha_0=1, beta_0=1):
    """Estimate alpha and beta with synthesis data using least square method.
    It is assumed that u(0) = 0.

    Arguments
    ---------
    t: :class:`~numpy.ndarray`
        A vector of time points.
    l: :class:`~numpy.ndarray` or sparse `csr_matrix`
        A matrix of labeled mRNA counts. Dimension: cells x time points.
    bounds: tuple
        The bound for alpha and beta. The default is alpha / beta > 0.
    alpha_0: float
        Initial guess for alpha.
    beta_0: float
        Initial guess for beta.

    Returns
    -------
    alpha: float
        The estimated value for alpha.
    beta: float
        The estimated value for beta.
    """
    l = l.A if issparse(l) else l

    tau = np.hstack((0, t))
    x = np.hstack((0, l))

    f_lsq = lambda p: sol_u(tau, 0, p[0], p[1]) - x
    ret = least_squares(f_lsq, np.array([alpha_0, beta_0]), bounds=bounds)
    return ret.x[0], ret.x[1]

def fit_all_synthesis(t, l, bounds=(0, np.inf), alpha_0=1, beta_0=1, gamma_0=1):
    """Estimate alpha, beta and gamma with synthesis data using least square method.
    It is assumed that u(0) = 0 and s(0) = 0.

    Arguments
    ---------
    t: :class:`~numpy.ndarray`
        A vector of time points.
    l: :class:`~numpy.ndarray` or sparse `csr_matrix`
        A matrix of labeled mRNA counts. Dimension: cells x time points.
    bounds: tuple
        The bound for alpha and beta. The default is alpha / beta > 0.
    alpha_0: float
        Initial guess for alpha.
    beta_0: float
        Initial guess for beta.
    gamma_0: float
        Initial guess for gamma.

    Returns
    -------
    alpha: float
        The estimated value for alpha.
    beta: float
        The estimated value for beta.
    gamma: float
        The estimated value for gamma.
    """
    l = l.A if issparse(l) else l

    tau = np.hstack((0, t))
    x = np.hstack((0, l))

    f_lsq = lambda p: sol_u(tau, 0, p[0], p[1]) + sol_s(tau, 0, 0, p[0], p[1], p[2]) - x
    ret = least_squares(f_lsq, np.array([alpha_0, beta_0, gamma_0]), bounds=bounds)
    return ret.x[0], ret.x[1], ret.x[2]

def concat_time_series_matrices(mats, t=None):
    """Concatenate a list of gene x cell matrices into a single matrix.

    Arguments
    ---------
    mats: :class:`~numpy.ndarray`
        A list of gene x cell matrices. The length of the list equals the number of time points
    t: :class:`~numpy.ndarray` or list
        A vector or list of time points

    Returns
    -------
    ret_mat: :class:`~numpy.ndarray`
        Concatenated gene x cell matrix.
    ret_t: :class:`~numpy.ndarray`
        A vector of time point for each cell.
    """
    ret_mat = np.concatenate(mats, axis=1)
    if t is not None:
        ret_t = np.concatenate([[t[i]]*mats[i].shape[1] for i in range(len(t))])
        return ret_mat, ret_t
    else:
        return ret_mat

class velocity:
    def __init__(self, alpha=None, beta=None, gamma=None, eta=None, delta=None, t=None, estimation=None):
        """The class that computes RNA/protein velocity given unknown parameters.

        Arguments
        ---------
        alpha: :class:`~numpy.ndarray`
            A matrix of transcription rate.
        beta: :class:`~numpy.ndarray`
            A vector of splicing rate constant for each gene.
        gamma: :class:`~numpy.ndarray`
            A vector of spliced mRNA degradation rate constant for each gene.
        eta: :class:`~numpy.ndarray`
            A vector of protein synthesis rate constant for each gene.
        delta: :class:`~numpy.ndarray`
            A vector of protein degradation rate constant for each gene.
        t: :class:`~numpy.ndarray` or None (default: None)
            A vector of the measured time points for cells
        estimation: :class:`~estimation`
            An instance of the estimation class. If this not None, the parameters will be taken from this class instead of the input arguments.
        """
        if estimation is not None:
            self.parameters = {}
            self.parameters['alpha'] = estimation.parameters['alpha']
            self.parameters['beta'] = estimation.parameters['beta']
            self.parameters['gamma'] = estimation.parameters['gamma']
            self.parameters['eta'] = estimation.parameters['eta']
            self.parameters['delta'] = estimation.parameters['delta']
            self.parameters['t'] = estimation.t
        else:
            self.parameters = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'eta': eta, 'delta': delta, 't': t}

    def vel_u(self, U):
        """Calculate the unspliced mRNA velocity.

        Arguments
        ---------
        U: :class:`~numpy.ndarray` or sparse `csr_matrix`
            A matrix of unspliced mRNA count. Dimension: genes x cells.

        Returns
        -------
        V: :class:`~numpy.ndarray` or sparse `csr_matrix`
            Each column of V is a velocity vector for the corresponding cell. Dimension: genes x cells.
        """

        t = self.parameters['t']
        t_uniq, t_uniq_cnt = np.unique(self.parameters['t'], return_counts=True)
        if self.parameters['alpha'] is not None and self.parameters['beta'] is not None:
            if type(self.parameters['alpha']) is not tuple:
                if self.parameters['alpha'].shape[1] == U.shape[1]:
                    alpha = self.parameters['alpha']
                elif self.parameters['alpha'].shape[1] == len(t_uniq) and len(t_uniq) > 1:
                    alpha = np.zeros(U.shape)
                    for i in range(len(t_uniq)):
                        cell_inds = t == t_uniq[i]
                        alpha[:, cell_inds] = np.repeat(self.parameters['alpha'][:, i], t_uniq_cnt[i], axis=1)
                else:
                    alpha = np.repeat(self.parameters['alpha'], U.shape[1], axis=1)
            else: # need to correct the velocity vector prediction when you use mix_std_stm experiments
                if self.parameters['alpha'][1].shape[1] == U.shape[1]:
                    alpha = self.parameters['alpha'][1]
                elif self.parameters['alpha'][1].shape[1] == len(t_uniq) and len(t_uniq) > 1:
                    alpha = np.zeros(U.shape)
                    for i in range(len(t_uniq)):
                        cell_inds = t == t_uniq[i]
                        alpha[:, cell_inds] = np.repeat(self.parameters['alpha'][1][:, i].reshape(-1, 1), t_uniq_cnt[i], axis=1)
                else:
                    alpha = np.repeat(self.parameters['alpha'][1], U.shape[1], axis=1)

            if len(self.parameters['beta'].shape) == 1:
                beta = np.repeat(self.parameters['beta'].reshape((-1, 1)), U.shape[1], axis=1)
            elif self.parameters['beta'].shape[1] == len(t_uniq) and len(t_uniq) > 1:
                beta = np.zeros_like(U.shape)
                for i in range(len(t_uniq)):
                    cell_inds = t == t_uniq[i]
                    beta[:, cell_inds] = np.repeat(self.parameters['beta'][:, i].reshape(-1, 1), t_uniq_cnt[i], axis=1)
            else:
                beta = np.repeat(self.parameters['beta'], U.shape[1], axis=1)

            V = csr_matrix(alpha) - (csr_matrix(beta).multiply(U)) if issparse(U) else \
                    alpha - beta * U
        else:
            V = np.nan
        return V

    def vel_s(self, U, S):
        """Calculate the unspliced mRNA velocity.

        Arguments
        ---------
        U: :class:`~numpy.ndarray` or sparse `csr_matrix`
            A matrix of unspliced mRNA counts. Dimension: genes x cells.
        S: :class:`~numpy.ndarray` or sparse `csr_matrix`
            A matrix of spliced mRNA counts. Dimension: genes x cells.

        Returns
        -------
        V: :class:`~numpy.ndarray` or sparse `csr_matrix`
            Each column of V is a velocity vector for the corresponding cell. Dimension: genes x cells.
        """

        t = self.parameters['t']
        t_uniq, t_uniq_cnt = np.unique(self.parameters['t'], return_counts=True)
        if self.parameters['beta'] is not None and self.parameters['gamma'] is not None:
            if len(self.parameters['beta'].shape) == 1:
                beta = np.repeat(self.parameters['beta'].reshape((-1, 1)), U.shape[1], axis=1)
            elif self.parameters['beta'].shape[1] == len(t_uniq) and len(t_uniq) > 1:
                beta = np.zeros_like(U.shape)
                for i in range(len(t_uniq)):
                    cell_inds = t == t_uniq[i]
                    beta[:, cell_inds] = np.repeat(self.parameters['beta'][:, i], t_uniq_cnt[i], axis=1)
            else:
                beta = np.repeat(self.parameters['beta'], U.shape[1], axis=1)

            if len(self.parameters['gamma'].shape) == 1:
                gamma = np.repeat(self.parameters['gamma'].reshape((-1, 1)), U.shape[1], axis=1)
            elif self.parameters['gamma'].shape[1] == len(t_uniq) and len(t_uniq) > 1:
                gamma = np.zeros_like(U.shape)
                for i in range(len(t_uniq)):
                    cell_inds = t == t_uniq[i]
                    gamma[:, cell_inds] = np.repeat(self.parameters['gamma'][:, i], t_uniq_cnt[i], axis=1)
            else:
                gamma = np.repeat(self.parameters['gamma'], U.shape[1], axis=1)

            V = csr_matrix(beta).multiply(U) - csr_matrix(gamma).multiply(S) if issparse(U) \
                    else beta * U - gamma * S
        else:
            V = np.nan
        return V

    def vel_p(self, S, P):
        """Calculate the protein velocity.

        Arguments
        ---------
        S: :class:`~numpy.ndarray` or sparse `csr_matrix`
            A matrix of spliced mRNA counts. Dimension: genes x cells.
        P: :class:`~numpy.ndarray` or sparse `csr_matrix`
            A matrix of protein counts. Dimension: genes x cells.

        Returns
        -------
        V: :class:`~numpy.ndarray` or sparse `csr_matrix`
            Each column of V is a velocity vector for the corresponding cell. Dimension: genes x cells.
        """

        t = self.parameters['t']
        t_uniq, t_uniq_cnt = np.unique(self.parameters['t'], return_counts=True)
        if self.parameters['eta'] is not None and self.parameters['delta'] is not None:
            if len(self.parameters['eta'].shape) == 1:
                eta = np.repeat(self.parameters['eta'].reshape((-1, 1)), S.shape[1], axis=1)
            elif self.parameters['eta'].shape[1] == len(t_uniq) and len(t_uniq) > 1:
                eta = np.zeros_like(S.shape)
                for i in range(len(t_uniq)):
                    cell_inds = t == t_uniq[i]
                    eta[:, cell_inds] = np.repeat(self.parameters['eta'][:, i], t_uniq_cnt[i], axis=1)
            else:
                eta = np.repeat(self.parameters['eta'], S.shape[1], axis=1)

            if len(self.parameters['delta'].shape) == 1:
                delta = np.repeat(self.parameters['delta'].reshape((-1, 1)), S.shape[1], axis=1)
            elif self.parameters['delta'].shape[1] == len(t_uniq) and len(t_uniq) > 1:
                delta = np.zeros_like(S.shape)
                for i in range(len(t_uniq)):
                    cell_inds = t == t_uniq[i]
                    delta[:, cell_inds] = np.repeat(self.parameters['delta'][:, i], t_uniq_cnt[i], axis=1)
            else:
                delta = np.repeat(self.parameters['delta'], S.shape[1], axis=1)

            V = csr_matrix(eta).multiply(S) - csr_matrix(delta).multiply(P) if issparse(P) else \
                    eta * S - delta * P
        else:
            V = np.nan
        return V

    def get_n_cells(self):
        """Get the number of cells if the parameter alpha is given.

        Returns
        -------
        n_cells: int
            The second dimension of the alpha matrix, if alpha is given.
        """
        if self.parameters['alpha'] is not None:
            n_cells = self.parameters['alpha'].shape[1]
        else:
            n_cells = np.nan
        return n_cells

    def get_n_genes(self):
        """Get the number of genes.

        Returns
        -------
        n_genes: int
            The first dimension of the alpha matrix, if alpha is given. Or, the length of beta, gamma, eta, or delta, if they are given.
        """
        if self.parameters['alpha'] is not None:
            n_genes = self.parameters['alpha'].shape[0]
        elif self.parameters['beta'] is not None:
            n_genes = len(self.parameters['beta'])
        elif self.parameters['gamma'] is not None:
            n_genes = len(self.parameters['gamma'])
        elif self.parameters['eta'] is not None:
            n_genes = len(self.parameters['eta'])
        elif self.parameters['delta'] is not None:
            n_genes = len(self.parameters['delta'])
        else:
            n_genes = np.nan
        return n_genes

class estimation:
    def __init__(self, U=None, Ul=None, S=None, Sl=None, P=None, t=None, ind_for_proteins=None, experiment_type='deg',
                 assumption_mRNA=None, assumption_protein='ss', concat_data=True):
        """The class that estimates parameters with input data.

        Arguments
        ---------
        U: :class:`~numpy.ndarray` or sparse `csr_matrix`
            A matrix of unspliced mRNA count.
        Ul: :class:`~numpy.ndarray` or sparse `csr_matrix`
            A matrix of unspliced, labeled mRNA count.
        S: :class:`~numpy.ndarray` or sparse `csr_matrix`
            A matrix of spliced mRNA count.
        Sl: :class:`~numpy.ndarray` or sparse `csr_matrix`
            A matrix of spliced, labeled mRNA count.
        P: :class:`~numpy.ndarray` or sparse `csr_matrix`
            A matrix of protein count.
        t: :class:`~estimation`
            A vector of time points.
        ind_for_proteins: :class:`~numpy.ndarray`
            A 1-D vector of the indices in the U, Ul, S, Sl layers that corresponds to the row name in the `protein` or
            `X_protein` key of `.obsm` attribute.
        experiment_type: str
            labelling experiment type. Available options are: 
            (1) 'deg': degradation experiment; 
            (2) 'kin': synthesis experiment; 
            (3) 'one-shot': one-shot kinetic experiment;
            (4) 'mix_std_stm': a mixed steady state and stimulation labeling experiment.
        assumption_mRNA: str
            Parameter estimation assumption for mRNA. Available options are: 
            (1) 'ss': pseudo steady state; 
            (2) None: kinetic data with no assumption. 
        assumption_protein: str
            Parameter estimation assumption for protein. Available options are: 
            (1) 'ss': pseudo steady state;
        concat_data: bool (default: True)
            Whether to concatenate data

        Attributes
        ----------
        t: :class:`~estimation`
            A vector of time points.
        data: `dict`
            A dictionary with uu, ul, su, sl, p as its keys.
        extyp: `str`
            labelling experiment type.
        asspt_mRNA: `str`
            Parameter estimation assumption for mRNA.
        asspt_prot: `str`
            Parameter estimation assumption for protein.
        parameters: `dict`
            A dictionary with alpha, beta, gamma, eta, delta as its keys.
                alpha: transcription rate
                beta: RNA splicing rate
                gamma: spliced mRNA degradation rate
                eta: translation rate
                delta: protein degradation rate
        """
        self.t = t
        self.data = {'uu': U, 'ul': Ul, 'su': S, 'sl': Sl, 'p': P}
        if concat_data:
            self.concatenate_data()

        self.extyp = experiment_type
        self.asspt_mRNA = assumption_mRNA
        self.asspt_prot = assumption_protein
        self.parameters = {'alpha': None, 'beta': None, 'gamma': None, 'eta': None, 'delta': None}
        self.aux_param = {'alpha_intercept': None, 'alpha_r2': None, 'gamma_intercept': None, 'gamma_r2': None, 'delta_intercept': None, \
                          'delta_r2': None, "uu0": None, "ul0": None, "su0": None, "sl0": None, 'U0': None, 'S0': None, 'total0': None} # note that alpha_intercept also corresponds to u0 in fit_alpha_degradation, similar to fit_first_order_deg_lsq
        self.ind_for_proteins = ind_for_proteins

    def fit(self, intercept=False, perc_left=5, perc_right=5, clusters=None):
        """Fit the input data to estimate all or a subset of the parameters

        Arguments
        ---------
        intercept: `bool`
            If using steady state assumption for fitting, then:
            True -- the linear regression is performed with an unfixed intercept;
            False -- the linear regression is performed with a fixed zero intercept.
        perc_left: `float` (default: 5)
            The percentage of samples included in the linear regression in the left tail. If set to None, then all the samples are included.
        perc_right: `float` (default: 5)
            The percentage of samples included in the linear regression in the right tail. If set to None, then all the samples are included.
        clusters: `list`
            A list of n clusters, each element is a list of indices of the samples which belong to this cluster.
        """
        n = self.get_n_genes()
        # fit mRNA
        if self.asspt_mRNA == 'ss':
            if np.all(self._exist_data('uu', 'su')):
                self.parameters['beta'] = np.ones(n)
                gamma, gamma_intercept, gamma_r2 = np.zeros(n), np.zeros(n), np.zeros(n)
                U = self.data['uu'] if self.data['ul'] is None else self.data['uu'] + self.data['ul']
                S = self.data['su'] if self.data['sl'] is None else self.data['su'] + self.data['sl']
                for i in range(n):
                    gamma[i], gamma_intercept[i], gamma_r2[i] = self.fit_gamma_steady_state(U[i], S[i],
                            intercept, perc_left, perc_right)
                self.parameters['gamma'], self.aux_param['gamma_intercept'], self.aux_param['gamma_r2'] = gamma, gamma_intercept, gamma_r2
            elif np.all(self._exist_data('uu', 'ul')):
                self.parameters['beta'] = np.ones(n)
                gamma, gamma_intercept, gamma_r2 = np.zeros(n), np.zeros(n), np.zeros(n)
                U = self.data['ul']
                S = self.data['uu'] + self.data['ul']
                for i in range(n):
                    gamma[i], gamma_intercept[i], gamma_r2[i] = self.fit_gamma_steady_state(U[i], S[i],
                            intercept, perc_left, perc_right)
                self.parameters['gamma'], self.aux_param['gamma_intercept'], self.aux_param['gamma_r2'] = gamma, gamma_intercept, gamma_r2
        else:
            if self.extyp == 'deg':
                if np.all(self._exist_data('ul', 'sl')):
                    # beta & gamma estimation
                    ul_m, ul_v, t_uniq = cal_12_mom(self.data['ul'], self.t)
                    sl_m, sl_v, _ = cal_12_mom(self.data['sl'], self.t)
                    self.parameters['beta'], self.parameters['gamma'], self.aux_param['ul0'], self.aux_param['sl0'] = \
                        self.fit_beta_gamma_lsq(t_uniq, ul_m, sl_m)
                    if self._exist_data('uu'):
                        # alpha estimation
                        uu_m, uu_v, _ = cal_12_mom(self.data['uu'], self.t)
                        alpha, uu0, r2 = np.zeros((n, 1)), np.zeros(n), np.zeros(n)
                        for i in range(n):
                            alpha[i], uu0[i], r2[i] = fit_alpha_degradation(t_uniq, uu_m[i], self.parameters['beta'][i], intercept=True)
                        self.parameters['alpha'], self.aux_param['alpha_intercept'], self.aux_param['uu0'], self.aux_param['alpha_r2'] = alpha, uu0, uu0, r2
                elif self._exist_data('ul'):
                    # gamma estimation
                    # use mean + var for fitting degradation parameter k
                    ul_m, ul_v, t_uniq = cal_12_mom(self.data['ul'], self.t)
                    self.parameters['gamma'], self.aux_param['ul0'] = self.fit_gamma_nosplicing_lsq(t_uniq, ul_m)
                    if self._exist_data('uu'):
                        # alpha estimation
                        alpha, alpha_b, alpha_r2 = np.zeros(n), np.zeros(n), np.zeros(n)
                        uu_m, uu_v, _ = cal_12_mom(self.data['uu'], self.t)
                        for i in range(n):
                            alpha[i], alpha_b[i], alpha_r2[i] = fit_alpha_degradation(t_uniq, uu_m[i], self.parameters['gamma'][i])
                        self.parameters['alpha'], self.aux_param['alpha_intercept'], self.aux_param['uu0'], self.aux_param['alpha_r2'] = alpha, alpha_b, alpha_b, alpha_r2
            elif (self.extyp == 'kin' or self.extyp == 'one_shot') and len(np.unique(self.t)) > 1:
                if np.all(self._exist_data('ul', 'uu', 'su')):
                    if not self._exist_parameter('beta'):
                        warn("beta & gamma estimation: only works when there're at least 2 time points.")
                        uu_m, uu_v, t_uniq = cal_12_mom(self.data['uu'], self.t)
                        su_m, su_v, _ = cal_12_mom(self.data['su'], self.t)

                        self.parameters['beta'], self.parameters['gamma'], self.aux_param['uu0'], self.aux_param['su0'] = self.fit_beta_gamma_lsq(t_uniq, uu_m, su_m)
                    # alpha estimation
                    ul_m, ul_v, t_uniq = cal_12_mom(self.data['ul'], self.t)
                    alpha = np.zeros_like(self.data['ul'].A) if issparse(self.data['ul']) else np.zeros_like(self.data['ul'])
                    # assume constant alpha across all cells
                    for i in range(n):
                        # for j in range(len(self.data['ul'][i])):
                        alpha[i, :] = fit_alpha_synthesis(t_uniq, ul_m[i], self.parameters['beta'][i])
                    self.parameters['alpha'] = alpha
                elif np.all(self._exist_data('ul', 'uu')):
                    n = self.data['uu'].shape[0]  # self.get_n_genes(data=U)
                    u0, gamma = np.zeros(n), np.zeros(n)
                    uu_m, uu_v, t_uniq = cal_12_mom(self.data['uu'], self.t)
                    for i in range(n):
                        gamma[i], u0[i] = fit_first_order_deg_lsq(t_uniq, uu_m[i])
                    self.parameters['gamma'], self.aux_param['uu0'] = gamma, u0
                    alpha = np.zeros_like(self.data['ul'].A) if issparse(self.data['ul']) else np.zeros_like(self.data['ul'])
                    # assume constant alpha across all cells
                    ul_m, ul_v, _ = cal_12_mom(self.data['ul'], self.t)
                    for i in range(n):
                        # for j in range(len(self.data['ul'][i])):
                        alpha[i, :] = fit_alpha_synthesis(t_uniq, ul_m[i], self.parameters['gamma'][i])
                    self.parameters['alpha'] = alpha
                    # alpha: one-shot
            # 'one_shot'
            elif self.extyp == 'one_shot':
                t_uniq = np.unique(self.t)
                if len(t_uniq) > 1:
                    raise Exception('By definition, one-shot experiment should involve only one time point measurement!')
                # calculate when having splicing or no splicing
                if np.all(self._exist_data('ul', 'uu', 'sl', 'su')):
                    if self._exist_data('ul') and self._exist_parameter('beta', 'gamma').all():
                        self.parameters['alpha'] = self.fit_alpha_oneshot(self.t, self.data['ul'], self.parameters['beta'], clusters)
                    else:
                        beta, gamma, U0, S0 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
                        for i in range(n): # can also use the two extreme time points and apply sci-fate like approach.
                            S, U = self.data['su'][i] + self.data['sl'][i], self.data['uu'][i] + self.data['ul'][i]

                            S0[i], gamma[i] = np.mean(S), solve_gamma(np.max(self.t), self.data['su'][i], S)
                            U0[i], beta[i] = np.mean(U), solve_gamma(np.max(self.t), self.data['uu'][i], U)
                        self.aux_param['U0'], self.aux_param['S0'], self.parameters['beta'], self.parameters['gamma'] = U0, S0, beta, gamma

                        self.parameters['alpha'] = self.fit_alpha_oneshot(self.t, self.data['ul'], self.parameters['beta'], clusters)
                else:
                    if self._exist_data('ul') and self._exist_parameter('gamma'):
                        self.parameters['alpha'] = self.fit_alpha_oneshot(self.t, self.data['ul'], self.parameters['gamma'], clusters)
                    elif self._exist_data('ul') and self._exist_data('uu'):
                        gamma, total0 = np.zeros(n), np.zeros(n)
                        for i in range(n):
                            total = self.data['uu'][i] + self.data['ul'][i]
                            total0[i], gamma[i] = np.mean(total), solve_gamma(np.max(self.t), self.data['uu'][i], total)
                        self.aux_param['total0'], self.parameters['gamma'] = total0, gamma

                        self.parameters['alpha'] = self.fit_alpha_oneshot(self.t, self.data['ul'], self.parameters['gamma'], clusters)

            elif self.extyp == 'mix_std_stm':
                t_min, t_max = np.min(self.t), np.max(self.t)
                if np.all(self._exist_data('ul', 'uu', 'su')):
                    gamma, beta, total, U = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
                    for i in range(n): # can also use the two extreme time points and apply sci-fate like approach.
                        tmp = self.data['uu'][i, self.t == t_max] + self.data['ul'][i, self.t == t_max] + self.data['su'][i, self.t == t_max] + self.data['sl'][i, self.t == t_max]
                        total[i] = np.mean(tmp)
                        gamma[i] = solve_gamma(t_max, self.data['uu'][i, self.t == t_max] + self.data['su'][i, self.t ==t_max], tmp)
                        # same for beta
                        tmp = self.data['uu'][i, self.t == t_max] + self.data['ul'][i, self.t == t_max]
                        U[i] = np.mean(tmp)
                        beta[i] = solve_gamma(np.max(self.t), self.data['uu'][i, self.t == t_max], tmp)

                    self.parameters['beta'], self.parameters['gamma'], self.aux_param['total0'], self.aux_param['U0'] = beta, gamma, total, U
                    # alpha estimation
                    self.parameters['alpha'] = self.solve_alpha_mix_std_stm(self.t, self.data['ul'], self.parameters['beta'])
                elif np.all(self._exist_data('ul', 'uu')):
                    n = self.data['uu'].shape[0]  # self.get_n_genes(data=U)
                    gamma, U = np.zeros(n), np.zeros(n)
                    for i in range(n): # apply sci-fate like approach (can also use one-single time point to estimate gamma)
                        # tmp = self.data['uu'][i, self.t == 0] + self.data['ul'][i, self.t == 0]
                        tmp_ = self.data['uu'][i, self.t == t_max] + self.data['ul'][i, self.t == t_max]

                        U[i] = np.mean(tmp_)
                        # gamma_1 = solve_gamma(np.max(self.t), self.data['uu'][i, self.t == 0], tmp) # steady state
                        gamma_2 = solve_gamma(t_max, self.data['uu'][i, self.t == t_max], tmp_) # stimulation
                        # gamma_3 = solve_gamma(np.max(self.t), self.data['uu'][i, self.t == np.max(self.t)], tmp) # sci-fate
                        gamma[i] = gamma_2
                        # print('Steady state, stimulation, sci-fate like gamma values are ', gamma_1, '; ', gamma_2, '; ', gamma_3)
                    self.parameters['gamma'], self.aux_param['U0'], self.parameters['beta'] = gamma, U, np.ones(gamma.shape)
                    # alpha estimation
                    self.parameters['alpha'] = self.solve_alpha_mix_std_stm(self.t, self.data['ul'], self.parameters['gamma'])

        # fit protein
        if np.all(self._exist_data('p', 'su')):
            ind_for_proteins = self.ind_for_proteins
            n = len(ind_for_proteins) if ind_for_proteins is not None else 0

            if self.asspt_prot == 'ss' and n > 0:
                self.parameters['eta'] = np.ones(n)
                delta, delta_intercept, delta_r2 = np.zeros(n), np.zeros(n), np.zeros(n)

                s = self.data['su'][ind_for_proteins] + self.data['sl'][ind_for_proteins] \
                    if self._exist_data('sl') else self.data['su'][ind_for_proteins]

                for i in range(n):
                    delta[i], delta_intercept[i], delta_r2[i] = self.fit_gamma_steady_state(s[i], self.data['p'][i],
                            intercept, perc_left, perc_right)
                self.parameters['delta'], self.aux_param['delta_intercept'], self.aux_param['delta_r2'] = delta, delta_intercept, delta_r2

    def fit_gamma_steady_state(self, u, s, intercept=True, perc_left=5, perc_right=5, normalize=True):
        """Estimate gamma using linear regression based on the steady state assumption.

        Arguments
        ---------
        u: :class:`~numpy.ndarray` or sparse `csr_matrix`
            A matrix of unspliced mRNA counts. Dimension: genes x cells.
        s: :class:`~numpy.ndarray` or sparse `csr_matrix`
            A matrix of spliced mRNA counts. Dimension: genes x cells.
        intercept: bool
            If using steady state assumption for fitting, then:
            True -- the linear regression is performed with an unfixed intercept;
            False -- the linear regresssion is performed with a fixed zero intercept.
        perc_left: float
            The percentage of samples included in the linear regression in the left tail. If set to None, then all the samples are included.
        perc_right: float
            The percentage of samples included in the linear regression in the right tail. If set to None, then all the samples are included.
        normalize: bool
            Whether to first normalize the

        Returns
        -------
        k: float
            The slope of the linear regression model, which is gamma under the steady state assumption.
        b: float
            The intercept of the linear regression model.
        r2: float
            Coefficient of determination or r square.
        """
        u = u.A.flatten() if issparse(u) else u.flatten()
        s = s.A.flatten() if issparse(s) else s.flatten()

        n = len(u)

        i_left = np.int(perc_left/100.0*n) if perc_left is not None else n
        i_right = np.int((100-perc_right)/100.0*n) if perc_right is not None else 0

        mask = np.zeros(n, dtype=bool)
        mask[:i_left] = mask[i_right:] = True

        if normalize:
            su = s / np.clip(np.max(s), 1e-3, None)
            su += u / np.clip(np.max(u), 1e-3, None)
        else:
            su = s + u

        extreme_ind = np.argsort(su)[mask]

        return fit_linreg(s[extreme_ind], u[extreme_ind], intercept)

    def fit_beta_gamma_lsq(self, t, U, S):
        """Estimate beta and gamma with the degradation data using the least squares method.

        Arguments
        ---------
        t: :class:`~numpy.ndarray`
            A vector of time points.
        U: :class:`~numpy.ndarray`
            A matrix of unspliced mRNA counts. Dimension: genes x cells.
        S: :class:`~numpy.ndarray`
            A matrix of spliced mRNA counts. Dimension: genes x cells.

        Returns
        -------
        beta: :class:`~numpy.ndarray`
            A vector of betas for all the genes.
        gamma: :class:`~numpy.ndarray`
            A vector of gammas for all the genes.
        u0: float
            Initial value of u.
        s0: float
            Initial value of s.
        """
        n = U.shape[0] # self.get_n_genes(data=U)
        beta = np.zeros(n)
        gamma = np.zeros(n)
        u0, s0 = np.zeros(n), np.zeros(n)

        for i in range(n):
            beta[i], u0[i] = fit_first_order_deg_lsq(t, U[i])
            if np.isfinite(u0[i]):
                gamma[i], s0[i] = fit_gamma_lsq(t, S[i], beta[i], u0[i])
            else:
                gamma[i], s0[i] = np.nan, np.nan
        return beta, gamma, u0, s0

    def fit_gamma_nosplicing_lsq(self, t, L):
        """Estimate gamma with the degradation data using the least squares method when there is no splicing data.

        Arguments
        ---------
        t: :class:`~numpy.ndarray`
            A vector of time points.
        L: :class:`~numpy.ndarray`
            A matrix of labeled mRNA counts. Dimension: genes x cells.

        Returns
        -------
        gamma: :class:`~numpy.ndarray`
            A vector of gammas for all the genes.
        l0: float
            The estimated value for the initial spliced, labeled mRNA count.
        """
        n = L.shape[0] # self.get_n_genes(data=L)
        gamma = np.zeros(n)
        l0 = np.zeros(n)

        for i in range(n):
            gamma[i], l0[i] = fit_first_order_deg_lsq(t, L[i].A[0]) if issparse(L) else fit_first_order_deg_lsq(t, L[i])
        return gamma, l0

    def solve_alpha_mix_std_stm(self, t, ul, beta, clusters=None, alpha_time_dependent=True):
        """Estimate the steady state transcription rate and analytically calculate the stimulation transcription rate
        given beta and steady state alpha for a mixed steady state and stimulation labeling experiment. 
        
        This approach assumes the same constant beta or gamma for both steady state or stimulation period.

        Parameters
        ----------
        t: `list` or `numpy.ndarray`
            Time period for stimulation state labeling for each cell.
        ul:
            A vector of labeled RNA amount in each cell.
        beta: `numpy.ndarray`
            A list of splicing rate for genes.
        clusters: `list`
            A list of n clusters, each element is a list of indices of the samples which belong to this cluster.
        alpha_time_dependent: `bool`
            Whether or not to model the simulation alpha rate as a time dependent variable.

        Returns
        -------
        alpha_std, alpha_stm: `numpy.ndarray`, `numpy.ndarray`
            The constant steady state transcription rate (alpha_std) or time-dependent or time-independent (determined by
            alpha_time_dependent) transcription rate (alpha_stm)
        """

        # calculate alpha initial guess:
        t = np.array(t) if type(t) is list else t
        t_std, t_stm, t_uniq, t_max, t_min = np.max(t) - t, t, np.unique(t), np.max(t), np.min(t)

        alpha_std_ini = self.fit_alpha_oneshot(np.array([t_max]), np.mean(ul[:, t == t_min], 1), beta, clusters).flatten()
        alpha_std, alpha_stm = alpha_std_ini, np.zeros((ul.shape[0], len(t_uniq)))
        alpha_stm[:, 0] = alpha_std_ini # 0 stimulation point is the steady state transcription
        for i in range(ul.shape[0]):
            l = ul[i].A.flatten() if issparse(ul) else ul[i]
            for t_ind in np.arange(1, len(t_uniq)):
                alpha_stm[i, t_ind] = solve_alpha_2p(t_max - t_uniq[t_ind], t_uniq[t_ind], alpha_std[i], beta[i], l[t==t_uniq[t_ind]])
        if not alpha_time_dependent:
            alpha_stm = alpha_stm.mean(1)

        return (alpha_std, alpha_stm)

    def fit_alpha_oneshot(self, t, U, beta, clusters=None):
        """Estimate alpha with the one-shot data.

        Arguments
        ---------
        t: float
            labelling duration.
        U: :class:`~numpy.ndarray`
            A matrix of unspliced mRNA counts. Dimension: genes x cells.
        beta: :class:`~numpy.ndarray`
            A vector of betas for all the genes.
        clusters: list
            A list of n clusters, each element is a list of indices of the samples which belong to this cluster.

        Returns
        -------
        alpha: :class:`~numpy.ndarray`
            A numpy array with the dimension of n_genes x clusters.
        """
        n_genes, n_cells = U.shape
        if clusters is None:
            clusters = [[i] for i in range(n_cells)]
        alpha = np.zeros((n_genes, len(clusters)))
        for i, c in enumerate(clusters):
            for j in range(n_genes):
                if len(c) > 0:
                    alpha[j, i] = fit_alpha_synthesis(t, U[j].A[0][c], beta[j]) if issparse(U) else fit_alpha_synthesis(t, U[j][c], beta[j])
                else:
                    alpha[j, i] = np.nan
        return alpha

    def concatenate_data(self):
        """Concatenate available data into a single matrix. 

        See "concat_time_series_matrices" for details.
        """
        keys = self.get_exist_data_names()
        time_unrolled = False
        for k in keys:
            data = self.data[k]
            if type(data) is list:
                if not time_unrolled and self.t is not None:
                    self.data[k], self.t = concat_time_series_matrices(self.data[k], self.t)
                    time_unrolled = True
                else:
                    self.data[k] = concat_time_series_matrices(self.data[k])

    def get_n_genes(self, key=None, data=None):
        """Get the number of genes."""
        if data is None:
            if key is None:
                data = self.data[self.get_exist_data_names()[0]]
            else:
                data = self.data[key]
        if type(data) is list:
            ret = len(data[0].A) if issparse(data[0]) else len(data[0])
        else:
            ret = data.shape[0]
        return ret

    def set_parameter(self, name, value):
        """Set the value for the specified parameter.

        Arguments
        ---------
        name: string
            The name of the parameter. E.g. 'beta'.
        value: :class:`~numpy.ndarray`
            A vector of values for the parameter to be set to.
        """
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
        """Get the names of all the data that are not 'None'."""
        ret = []
        for k, v in self.data.items():
            if v is not None:
                ret.append(k)
        return ret
