import numpy as np
from scipy.optimize import least_squares
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse

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

def fit_linreg(x, y, intercept=True):
    """Simple linear regression: y = kx + b.

    Arguments
    ---------
    x: :class:`~numpy.ndarray`
        A vector of independent variables.
    y: :class:`~numpy.ndarray`
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
    else:
        k = np.mean(yy) / np.mean(xx)
        b = 0
    return k, b

def fit_first_order_deg_lsq(t, l, bounds=(0, np.inf), fix_l0=False, beta_0=1):
    """Estimate beta with degradation data using least squares method.

    Arguments
    ---------
    t: :class:`~numpy.ndarray`
        A vector of time points.
    l: :class:`~numpy.ndarray`
        A vector of unspliced, labeled mRNA counts for each time point.
    u0: float
        Initial number of unsplcied mRNA.
    bounds: tuple
        The bound for gamma. The default is gamma > 0.
    fixed_l0: bool
        True: l0 will be calculated by averging the first column of l;
        False: l0 is a parameter that will be estimated all together with gamma using lsq.
    beta_0: float
        Initial guess for beta.

    Returns
    -------
    beta: float
        The estimated value for beta.
    l0: float
        The estimated value for the initial spliced, labeled mRNA count.
    """
    l = l.A if issparse(l) else l

    tau = t - np.min(t)
    l0 = np.mean(l[tau == 0])

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

def fit_gamma_lsq(t, s, beta, u0, bounds=(0, np.inf), fix_s0=False):
    """Estimate gamma with degradation data using least squares method.

    Arguments
    ---------
    t: :class:`~numpy.ndarray`
        A vector of time points.
    s: :class:`~numpy.ndarray`
        A vector of spliced, labeled mRNA counts for each time point.
    beta: float
        The value of beta.
    u0: float
        Initial number of unsplcied mRNA.
    bounds: tuple
        The bound for gamma. The default is gamma > 0.
    fixed_s0: bool
        True: s0 will be calculated by averging the first column of s;
        False: s0 is a parameter that will be estimated all together with gamma using lsq.

    Returns
    -------
    gamma: float
        The estimated value for gamma.
    s0: float
        The estimated value for the initial spliced mRNA count.
    """
    s = s.A if issparse(s) else s

    tau = t - np.min(t)
    s0 = np.mean(s[tau == 0])
    g0 = beta * u0/s0

    if fix_s0:
        f_lsq = lambda g: sol_s(tau, s0, u0, 0, beta, g) - s
        ret = least_squares(f_lsq, g0, bounds=bounds)
        gamma = ret.x
    else:
        f_lsq = lambda p: sol_s(tau, p[1], u0, 0, beta, p[0]) - s
        ret = least_squares(f_lsq, np.array([g0, s0]), bounds=bounds)
        gamma = ret.x[0]
        s0 = ret.x[1]
    return gamma, s0

def fit_alpha_synthesis(t, u, beta):
    """Estimate alpha with synthesis data using linear regression with fixed zero intercept.
    It is assumed that u(0) = 0.

    Arguments
    ---------
    u: :class:`~numpy.ndarray`
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

def fit_alpha_degradation(t, u, beta, mode=None):
    """Estimate alpha with degradation data using linear regression.

    Arguments
    ---------
    u: :class:`~numpy.ndarray`
        A matrix of unspliced mRNA counts. Dimension: cells x time points.
    beta: float
        The value of beta.

    Returns
    -------
    alpha: float
        The estimated value for alpha.
    b: float
        The initial unspliced mRNA count.
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
    b = ym - k * xm if mode != 'fast' else None

    return k * beta, b

def fit_alpha_beta_synthesis(t, l, bounds=(0, np.inf), alpha_0=1, beta_0=1):
    """Estimate alpha and beta with synthesis data using least square method.
    It is assumed that u(0) = 0.

    Arguments
    ---------
    l: :class:`~numpy.ndarray`
        A matrix of labeled mRNA counts. Dimension: cells x time points.
    beta: float
        The value of beta.

    Returns
    -------
    alpha: float
        The estimated value for alpha.
    """
    l = l.A if issparse(l) else l

    tau = np.hstack((0, t))
    x = np.hstack((0, l))

    f_lsq = lambda p: sol_u(tau, 0, p[0], p[1]) - x
    ret = least_squares(f_lsq, np.array([alpha_0, beta_0]), bounds=bounds)
    return ret.x[0], ret.x[1]

def fit_all_synthesis(t, l, bounds=(0, np.inf), alpha_0=1, beta_0=1, gamma_0=1):
    """Estimate alpha and beta with synthesis data using least square method.
    It is assumed that u(0) = 0.

    Arguments
    ---------
    l: :class:`~numpy.ndarray`
        A matrix of labeled mRNA counts. Dimension: cells x time points.
    beta: float
        The value of beta.

    Returns
    -------
    alpha: float
        The estimated value for alpha.
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
    def __init__(self, alpha=None, beta=None, gamma=None, eta=None, delta=None, estimation=None):
        """The class that computes RNA velocity given unknown parameters.

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
        estimation: :class:`~estimation`
            An instance of the estimation class. If this not none, the parameters will be taken from this class instead of the input arguments.
        """
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
        """Calculate the unspliced mRNA velocity.

        Arguments
        ---------
        U: :class:`~numpy.ndarray`
            A matrix of unspliced mRNA count. Dimension: genes x cells.

        Returns
        -------
        V: :class:`~numpy.ndarray`
            Each column of V is a velocity vector for the corresponding cell. Dimension: genes x cells.
        """
        if self.parameters['alpha'] is not None and self.parameters['beta'] is not None:
            V = self.parameters['alpha'] - (self.parameters['beta'] * U.T).T
        else:
            V = np.nan
        return V

    def vel_s(self, U, S):
        """Calculate the unspliced mRNA velocity.

        Arguments
        ---------
        U: :class:`~numpy.ndarray`
            A matrix of unspliced mRNA counts. Dimension: genes x cells.
        S: :class:`~numpy.ndarray`
            A matrix of spliced mRNA counts. Dimension: genes x cells.

        Returns
        -------
        V: :class:`~numpy.ndarray`
            Each column of V is a velocity vector for the corresponding cell. Dimension: genes x cells.
        """
        if self.parameters['beta'] is not None and self.parameters['gamma'] is not None:
            V = self.parameters['beta'] * U.T - self.parameters['gamma'] * S.T
            V = V.T
        else:
            V = np.nan
        return V

    def vel_p(self, S, P):
        """Calculate the protein velocity.

        Arguments
        ---------
        S: :class:`~numpy.ndarray`
            A matrix of spliced mRNA counts. Dimension: genes x cells.
        P: :class:`~numpy.ndarray`
            A matrix of protein counts. Dimension: genes x cells.

        Returns
        -------
        V: :class:`~numpy.ndarray`
            Each column of V is a velocity vector for the corresponding cell. Dimension: genes x cells.
        """
        if self.parameters['eta'] is not None and self.parameters['delta'] is not None:
            V = self.parameters['eta'] * S.T - self.parameters['delta'] * P.T
            V = V.T
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
    def __init__(self, U=None, Ul=None, S=None, Sl=None, P=None, t=None, ind_for_proteins=None, experiment_type='deg', assumption_mRNA=None, assumption_protein='ss', concat_data=True):
        """The class that estimates parameters with input data.

        Arguments
        ---------
        U: :class:`~numpy.ndarray`
            A matrix of unspliced mRNA count.
        Ul: :class:`~numpy.ndarray`
            A matrix of unspliced, labeled mRNA count.
        S: :class:`~numpy.ndarray`
            A matrix of spliced mRNA count.
        Sl: :class:`~numpy.ndarray`
            A matrix of spliced, labeled mRNA count.
        P: :class:`~numpy.ndarray`
            A matrix of protein count.
        t: :class:`~estimation`
            A vector of time points.
        ind_for_proteins: :class:`~numpy.ndarray`
            A 1-D vector of the indices in the U, Ul, S, Sl layers that corresponds to the row name in the P layer.
        experiment_type: str
            Labeling experiment type. Available options are: 
            (1) 'deg': degradation experiment; 
            (2) 'kin': synthesis experiment; 
            (3) 'one-shot': one-shot kinetic experiment.
        assumption_mRNA: str
            Parameter estimation assumption for mRNA. Available options are: 
            (1) 'ss': pseudo steady state; 
            (2) None: kinetic data with no assumption. 
        assumption_protein: str
            Parameter estimation assumption for protein. Available options are: 
            (1) 'ss': pseudo steady state;
        concat_data: bool (default: True)
            Whether to concate data

        Attributes
        ----------
        t: :class:`~estimation`
            A vector of time points.
        data: `dict`
            A dictionary with uu, ul, su, sl, p as its keys.
        extyp: `str`
            Labeling experiment type.
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
        self.ind_for_proteins = ind_for_proteins

    def fit(self, intercept=True, perc_left=5, perc_right=5, clusters=None):
        """Fit the input data to estimate all or a subset of the parameters

        Arguments
        ---------
        intercept: bool
            If using steady state assumption for fitting, then:
            True -- the linear regression is performed with an unfixed intercept;
            False -- the linear regression is performed with a fixed zero intercept.
        perc_left: float (default: 5)
            The percentage of samples included in the linear regression in the left tail. If set to None, then all the samples are included.
        perc_right: float (default: 5)
            The percentage of samples included in the linear regression in the right tail. If set to None, then all the samples are included.
        clusters: list
            A list of n clusters, each element is a list of indices of the samples which belong to this cluster.
        """
        n = self.get_n_genes()
        # fit mRNA
        if self.asspt_mRNA == 'ss':
            if np.all(self._exist_data('uu', 'su')):
                self.parameters['beta'] = np.ones(n)
                gamma = np.zeros(n)
                U = self.data['uu'] if self.data['ul'] is None else self.data['uu'] + self.data['ul']
                S = self.data['su'] if self.data['sl'] is None else self.data['su'] + self.data['sl']
                for i in range(n):
                    gamma[i], _ = self.fit_gamma_steady_state(U[i], S[i],
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
                elif self._exist_data('ul'):
                    # gamma estimation
                    self.parameters['beta'] = np.ones(n)
                    self.parameters['gamma'] = self.fit_gamma_nosplicing_lsq(self.t, self.data['ul'])
                    #if self._exist_data('uu'):
                        # alpha estimation
                        #alpha = np.zeros(n)
                        #for i in range(n):
                        #    alpha[i], _ = fit_alpha_synthesis(self.t, self.data['uu'][i], self.parameters['gamma'][i])
                        #self.parameters['alpha'] = alpha
            elif self.extyp == 'kin':
                if np.all(self._exist_data('ul', 'uu', 'su')):
                    if not self._exist_parameter('beta'):
                        # beta & gamma estimation: only works when there're at least 2 time points
                        self.parameters['beta'], self.parameters['gamma'] = self.fit_beta_gamma_lsq(self.t, self.data['uu'], self.data['su'])
                    # alpha estimation
                    alpha = np.zeros_like(self.data['ul'])
                    # assume constant alpha across all cells
                    for i in range(n):
                        # for j in range(len(self.data['ul'][i])):
                        alpha[i, :] = fit_alpha_synthesis(self.t, self.data['ul'][i], self.parameters['beta'][i])
                    self.parameters['alpha'] = alpha
                elif np.all(self._exist_data('ul', 'uu')):
                    pass
            # 'one_shot'
            elif self.extyp == 'one_shot':
                if self._exist_data('ul') and self._exist_parameter('beta'):
                    self.parameters['alpha'] = self.fit_alpha_oneshot(self.t, self.data['ul'], self.parameters['beta'], clusters)

        # fit protein
        if np.all(self._exist_data('p', 'su')):
            ind_for_proteins = self.ind_for_proteins
            n = len(ind_for_proteins) if ind_for_proteins is not None else 0

            if self.asspt_prot == 'ss' and n > 0:
                self.parameters['eta'] = np.ones(n)
                delta = np.zeros(n)
                for i in range(n):
                    s = self.data['su'][i][ind_for_proteins] + self.data['sl'][i][ind_for_proteins] \
                        if self._exist_data('sl') else self.data['su'][i][ind_for_proteins]
                    delta[i], _ = self.fit_gamma_steady_state(s, self.data['p'][i],
                            intercept, perc_left, perc_right)
                self.parameters['delta'] = delta

    def fit_gamma_steady_state(self, u, s, intercept=True, perc_left=5, perc_right=5):
        """Estimate gamma using linear regression based on the steady state assumption.

        Arguments
        ---------
        u: :class:`~numpy.ndarray`
            A matrix of spliced mRNA counts. Dimension: genes x cells.
        s: :class:`~numpy.ndarray`
            A matrix of protein counts. Dimension: genes x cells.
        intercept: bool
            If using steady state assumption for fitting, then:
            True -- the linear regression is performed with an unfixed intercept;
            False -- the linear regresssion is performed with a fixed zero intercept.
        perc_left: float
            The percentage of samples included in the linear regression in the left tail. If set to None, then all the samples are included.
        perc_right: float
            The percentage of samples included in the linear regression in the right tail. If set to None, then all the samples are included.

        Returns
        -------
        k: float
            The slope of the linear regression model, which is gamma under the steady state assumption.
        b: float
            The intercept of the linear regression model.
        """
        u = u.A if issparse(u) else u
        s = s.A if issparse(s) else s

        n = len(u)
        i_left = np.int(perc_left/100.0*n) if perc_left is not None else n
        i_right = np.int((100-perc_right)/100.0*n) if perc_right is not None else 0
        mask = np.zeros(n, dtype=bool)
        mask[:i_left] = mask[i_right:] = True
        return fit_linreg(np.sort(s)[mask], np.sort(u)[mask], intercept)

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
        """
        n = self.get_n_genes(data=U)
        beta = np.zeros(n)
        gamma = np.zeros(n)
        for i in range(n):
            beta[i], u0 = fit_first_order_deg_lsq(t, U[i])
            gamma[i], _ = fit_gamma_lsq(t, S[i], beta[i], u0)
        return beta, gamma

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
        """
        n = self.get_n_genes(data=L)
        gamma = np.zeros(n)
        for i in range(n):
            gamma[i], _ = fit_first_order_deg_lsq(t, L[i])
        return gamma

    def fit_alpha_oneshot(self, t, U, beta, clusters=None):
        """Estimate alpha with the one-shot data.

        Arguments
        ---------
        t: float
            Labeling duration.
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
                    alpha[j, i] = fit_alpha_synthesis(t, U[j][c], beta[j])
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

