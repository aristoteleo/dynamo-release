from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import itertools
from scipy.sparse import csr_matrix
from warnings import warn
from ...tools.utils import one_shot_gamma_alpha, calc_R2, calc_norm_loglikelihood, one_shot_gamma_alpha_matrix
from ...tools.moments import calc_12_mom_labeling
from .utils_velocity import *
from ...tools.utils import update_dict
from ...tools.moments import calc_2nd_moment
# from sklearn.cluster import KMeans
# from sklearn.neighbors import NearestNeighbors


class velocity:
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
         estimation: :class:`~ss_estimation`
             An instance of the estimation class. If this not None, the parameters will be taken from this class instead of the input arguments.
     """

    def __init__(
        self,
        alpha=None,
        beta=None,
        gamma=None,
        eta=None,
        delta=None,
        t=None,
        estimation=None,
    ):
        if estimation is not None:
            self.parameters = {}
            self.parameters["alpha"] = estimation.parameters["alpha"]
            self.parameters["beta"] = estimation.parameters["beta"]
            self.parameters["gamma"] = estimation.parameters["gamma"]
            self.parameters["eta"] = estimation.parameters["eta"]
            self.parameters["delta"] = estimation.parameters["delta"]
            self.parameters["t"] = estimation.t
        else:
            self.parameters = {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "eta": eta,
                "delta": delta,
                "t": t,
            }

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

        t = self.parameters["t"]
        t_uniq, t_uniq_cnt = np.unique(self.parameters["t"], return_counts=True)
        if self.parameters["alpha"] is not None:
            if self.parameters["beta"] is None and self.parameters["gamma"] is not None:
                no_beta = True
                self.parameters["beta"] = self.parameters["gamma"]
            else:
                no_beta = False

            if type(self.parameters["alpha"]) is not tuple:
                if self.parameters["alpha"].ndim == 1:
                    alpha = np.repeat(
                        self.parameters["alpha"].reshape((-1, 1)), U.shape[1], axis=1
                    )
                elif self.parameters["alpha"].shape[1] == U.shape[1]:
                    alpha = self.parameters["alpha"]
                elif (
                    self.parameters["alpha"].shape[1] == len(t_uniq) and len(t_uniq) > 1
                ):
                    alpha = np.zeros(U.shape)
                    for i in range(len(t_uniq)):
                        cell_inds = t == t_uniq[i]
                        alpha[:, cell_inds] = np.repeat(
                            self.parameters["alpha"][:, i], t_uniq_cnt[i], axis=1
                        )
                else:
                    alpha = np.repeat(self.parameters["alpha"], U.shape[1], axis=1)
            else:  # need to correct the velocity vector prediction when you use mix_std_stm experiments
                if self.parameters["alpha"][1].shape[1] == U.shape[1]:
                    alpha = self.parameters["alpha"][1]
                elif (
                    self.parameters["alpha"][1].shape[1] == len(t_uniq)
                    and len(t_uniq) > 1
                ):
                    alpha = np.zeros(U.shape)
                    for i in range(len(t_uniq)):
                        cell_inds = t == t_uniq[i]
                        alpha[:, cell_inds] = np.repeat(
                            self.parameters["alpha"][1][:, i].reshape(-1, 1),
                            t_uniq_cnt[i],
                            axis=1,
                        )
                else:
                    alpha = np.repeat(self.parameters["alpha"][1], U.shape[1], axis=1)

            if self.parameters["beta"].ndim == 1:
                beta = np.repeat(
                    self.parameters["beta"].reshape((-1, 1)), U.shape[1], axis=1
                )
            elif self.parameters["beta"].shape[1] == len(t_uniq) and len(t_uniq) > 1:
                beta = np.zeros_like(U.shape)
                for i in range(len(t_uniq)):
                    cell_inds = t == t_uniq[i]
                    beta[:, cell_inds] = np.repeat(
                        self.parameters["beta"][:, i].reshape(-1, 1),
                        t_uniq_cnt[i],
                        axis=1,
                    )
            else:
                beta = np.repeat(self.parameters["beta"], U.shape[1], axis=1)

            if no_beta:
                self.parameters["beta"] = None
            V = (
                csr_matrix(alpha, dtype=np.float64) - (csr_matrix(beta, dtype=np.float64).multiply(U))
                if issparse(U)
                else alpha - beta * U
            )
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

        t = self.parameters["t"]
        t_uniq, t_uniq_cnt = np.unique(self.parameters["t"], return_counts=True)
        if self.parameters["gamma"] is not None:
            if self.parameters["beta"] is None and self.parameters["alpha"] is not None:
                no_beta = True
                self.parameters["beta"] = self.parameters["alpha"]
            else:
                no_beta = False

            if self.parameters["beta"].ndim == 1:
                beta = np.repeat(
                    self.parameters["beta"].reshape((-1, 1)), U.shape[1], axis=1
                )
            elif self.parameters["beta"].shape[1] == U.shape[1]:
                beta = self.parameters["beta"]
            elif self.parameters["beta"].shape[1] == len(t_uniq) and len(t_uniq) > 1:
                beta = np.zeros_like(U.shape)
                for i in range(len(t_uniq)):
                    cell_inds = t == t_uniq[i]
                    beta[:, cell_inds] = np.repeat(
                        self.parameters["beta"][:, i], t_uniq_cnt[i], axis=1
                    )
            else:
                beta = np.repeat(self.parameters["beta"], U.shape[1], axis=1)

            if len(self.parameters["gamma"].shape) == 1:
                gamma = np.repeat(
                    self.parameters["gamma"].reshape((-1, 1)), U.shape[1], axis=1
                )
            elif self.parameters["gamma"].shape[1] == U.shape[1]:
                gamma = self.parameters["gamma"]
            elif self.parameters["gamma"].shape[1] == len(t_uniq) and len(t_uniq) > 1:
                gamma = np.zeros_like(U.shape)
                for i in range(len(t_uniq)):
                    cell_inds = t == t_uniq[i]
                    gamma[:, cell_inds] = np.repeat(
                        self.parameters["gamma"][:, i], t_uniq_cnt[i], axis=1
                    )
            else:
                gamma = np.repeat(self.parameters["gamma"], U.shape[1], axis=1)

            if no_beta:
                V = (
                    csr_matrix(beta, dtype=np.float64) - csr_matrix(gamma, dtype=np.float64).multiply(S)
                    if issparse(U)
                    else beta - gamma * S
                )
            else:
                V = (
                    csr_matrix(beta, dtype=np.float64).multiply(U) - csr_matrix(gamma, dtype=np.float64).multiply(S)
                    if issparse(U)
                    else beta * U - gamma * S
                )
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

        t = self.parameters["t"]
        t_uniq, t_uniq_cnt = np.unique(self.parameters["t"], return_counts=True)
        if self.parameters["eta"] is not None and self.parameters["delta"] is not None:
            if self.parameters["eta"].ndim == 1:
                eta = np.repeat(
                    self.parameters["eta"].reshape((-1, 1)), S.shape[1], axis=1
                )
            elif self.parameters["eta"].shape[1] == S.shape[1]:
                eta = self.parameters["eta"]
            elif self.parameters["eta"].shape[1] == len(t_uniq) and len(t_uniq) > 1:
                eta = np.zeros_like(S.shape)
                for i in range(len(t_uniq)):
                    cell_inds = t == t_uniq[i]
                    eta[:, cell_inds] = np.repeat(
                        self.parameters["eta"][:, i], t_uniq_cnt[i], axis=1
                    )
            else:
                eta = np.repeat(self.parameters["eta"], S.shape[1], axis=1)

            if len(self.parameters["delta"].shape) == 1:
                delta = np.repeat(
                    self.parameters["delta"].reshape((-1, 1)), S.shape[1], axis=1
                )
            elif self.parameters["delta"].shape[1] == S.shape[1]:
                delta = self.parameters["delta"]
            elif self.parameters["delta"].shape[1] == len(t_uniq) and len(t_uniq) > 1:
                delta = np.zeros_like(S.shape)
                for i in range(len(t_uniq)):
                    cell_inds = t == t_uniq[i]
                    delta[:, cell_inds] = np.repeat(
                        self.parameters["delta"][:, i], t_uniq_cnt[i], axis=1
                    )
            else:
                delta = np.repeat(self.parameters["delta"], S.shape[1], axis=1)

            V = (
                csr_matrix(eta, dtype=np.float64).multiply(S) - csr_matrix(delta, dtype=np.float64).multiply(P)
                if issparse(P)
                else eta * S - delta * P
            )
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
        if self.parameters["alpha"] is not None:
            n_cells = self.parameters["alpha"].shape[1]
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
        if self.parameters["alpha"] is not None:
            n_genes = self.parameters["alpha"].shape[0]
        elif self.parameters["beta"] is not None:
            n_genes = len(self.parameters["beta"])
        elif self.parameters["gamma"] is not None:
            n_genes = len(self.parameters["gamma"])
        elif self.parameters["eta"] is not None:
            n_genes = len(self.parameters["eta"])
        elif self.parameters["delta"] is not None:
            n_genes = len(self.parameters["delta"])
        else:
            n_genes = np.nan
        return n_genes


class ss_estimation:
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
        US: :class:`~numpy.ndarray` or sparse `csr_matrix`
            A matrix of second moment of unspliced/spliced gene expression count for conventional or NTR velocity.
        S2: :class:`~numpy.ndarray` or sparse `csr_matrix`
            A matrix of second moment of spliced gene expression count for conventional or NTR velocity.
        conn: :class:`~numpy.ndarray` or sparse `csr_matrix`
            The connectivity matrix that can be used to calculate first /second moment of the data.
        t: :class:`~ss_estimation`
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
        cores: `int` (default: 1)
            Number of cores to run the estimation. If cores is set to be > 1, multiprocessing will be used to parallel
            the parameter estimation.

    Returns
    ----------
        t: :class:`~ss_estimation`
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

    def __init__(
        self,
        U=None,
        Ul=None,
        S=None,
        Sl=None,
        P=None,
        US=None,
        S2=None,
        conn=None,
        t=None,
        ind_for_proteins=None,
        model='stochastic',
        est_method='gmm',
        experiment_type="deg",
        assumption_mRNA=None,
        assumption_protein="ss",
        concat_data=True,
        cores=1,
        **kwargs
    ):

        self.t = t
        self.data = {"uu": U, "ul": Ul, "su": S, "sl": Sl, "p": P, "us": US, "s2": S2}
        if concat_data:
            self.concatenate_data()

        self.conn = conn
        self.extyp = experiment_type
        self.model = model
        self.est_method = est_method
        self.asspt_mRNA = assumption_mRNA
        self.asspt_prot = assumption_protein
        self.cores = cores
        self.parameters = {
            "alpha": None,
            "beta": None,
            "gamma": None,
            "eta": None,
            "delta": None,
        }
        self.parameters = update_dict(self.parameters, kwargs)
        self.aux_param = {
            "alpha_intercept": None,
            "alpha_r2": None,
            "beta_k": None,
            "gamma_k": None,
            "gamma_intercept": None,
            "gamma_r2": None,
            "gamma_logLL": None,
            "delta_intercept": None,
            "delta_r2": None,
            "uu0": None,
            "ul0": None,
            "su0": None,
            "sl0": None,
            "U0": None,
            "S0": None,
            "total0": None,
        }  # note that alpha_intercept also corresponds to u0 in fit_alpha_degradation, similar to fit_first_order_deg_lsq
        self.ind_for_proteins = ind_for_proteins

    def fit(
        self,
        intercept=False,
        perc_left=None,
        perc_right=5,
        clusters=None,
        one_shot_method="combined",
    ):
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
        cores = max(1, int(self.cores))
        # fit mRNA
        if self.extyp.lower() == "conventional":
            if self.model.lower() == "deterministic":
                if np.all(self._exist_data("uu", "su")):
                    self.parameters["beta"] = np.ones(n)
                    gamma, gamma_intercept, gamma_r2, gamma_logLL = (
                        np.zeros(n),
                        np.zeros(n),
                        np.zeros(n),
                        np.zeros(n),
                    )
                    U = (
                        self.data["uu"]
                        if self.data["ul"] is None
                        else self.data["uu"] + self.data["ul"]
                    )
                    S = (
                        self.data["su"]
                        if self.data["sl"] is None
                        else self.data["su"] + self.data["sl"]
                    )
                    if cores == 1:
                        for i in tqdm(range(n), desc="estimating gamma"):
                            (
                                gamma[i],
                                gamma_intercept[i],
                                _,
                                gamma_r2[i],
                                _,
                                gamma_logLL[i],
                            ) = self.fit_gamma_steady_state(
                                U[i], S[i], intercept, perc_left, perc_right
                            )
                    else:
                        pool = ThreadPool(cores)
                        res = pool.starmap(self.fit_gamma_steady_state,
                                     zip(U, S, itertools.repeat(intercept), itertools.repeat(perc_left),
                                         itertools.repeat(perc_right)))
                        pool.close()
                        pool.join()
                        (gamma, gamma_intercept, _, gamma_r2, _, gamma_logLL) = zip(*res)
                        (gamma, gamma_intercept, gamma_r2, gamma_logLL) = np.array(gamma), np.array(gamma_intercept), \
                                                                          np.array(gamma_r2), np.array(gamma_logLL)
                    (
                        self.parameters["gamma"],
                        self.aux_param["gamma_intercept"],
                        self.aux_param["gamma_r2"],
                        self.aux_param["gamma_logLL"],
                    ) = (gamma, gamma_intercept, gamma_r2, gamma_logLL)
                elif np.all(self._exist_data("uu", "ul")):
                    self.parameters["beta"] = np.ones(n)
                    gamma, gamma_intercept, gamma_r2, gamma_logLL = (
                        np.zeros(n),
                        np.zeros(n),
                        np.zeros(n),
                        np.zeros(n),
                    )
                    U = self.data["ul"]
                    S = self.data["uu"] + self.data["ul"]
                    if cores == 1:
                        for i in tqdm(range(n), desc="estimating gamma"):
                            (
                                gamma[i],
                                gamma_intercept[i],
                                _,
                                gamma_r2[i],
                                _,
                                gamma_logLL[i],
                            ) = self.fit_gamma_steady_state(
                                U[i], S[i], intercept, perc_left, perc_right
                            )
                    else:
                        pool = ThreadPool(cores)
                        res = pool.starmap(self.fit_gamma_steady_state,
                                     zip(U, S, itertools.repeat(intercept), itertools.repeat(perc_left),
                                         itertools.repeat(perc_right)))
                        pool.close()
                        pool.join()
                        (gamma, gamma_intercept, _, gamma_r2, _, gamma_logLL) = zip(*res)
                        (gamma, gamma_intercept, gamma_r2, gamma_logLL) = np.array(gamma), np.array(gamma_intercept), \
                                                                          np.array(gamma_r2), np.array(gamma_logLL)
                    (
                        self.parameters["gamma"],
                        self.aux_param["gamma_intercept"],
                        self.aux_param["gamma_r2"],
                        self.aux_param["gamma_logLL"],
                    ) = (gamma, gamma_intercept, gamma_r2, gamma_logLL)
            elif self.model.lower() == 'stochastic':
                if np.all(self._exist_data("uu", "su")):
                    self.parameters["beta"] = np.ones(n)
                    gamma, gamma_intercept, gamma_r2, gamma_logLL = (
                        np.zeros(n),
                        np.zeros(n),
                        np.zeros(n),
                        np.zeros(n),
                    )
                    U = (
                        self.data["uu"]
                        if self.data["ul"] is None
                        else self.data["uu"] + self.data["ul"]
                    )
                    S = (
                        self.data["su"]
                        if self.data["sl"] is None
                        else self.data["su"] + self.data["sl"]
                    )
                    US = self.data['us'] if 'us' in self.data.keys() else calc_2nd_moment(
                        U.T, S.T, self.conn, mX=U.T, mY=S.T
                    ).T
                    S2 = self.data['s2'] if 's2' in self.data.keys() else calc_2nd_moment(
                        S.T, S.T, self.conn, mX=S.T, mY=S.T
                    ).T
                    if cores == 1:
                        for i in tqdm(range(n), desc="estimating gamma"):
                            (
                                gamma[i],
                                gamma_intercept[i],
                                _,
                                gamma_r2[i],
                                _,
                                gamma_logLL[i],
                            ) = self.fit_gamma_stochastic(
                                self.est_method,
                                U[i],
                                S[i],
                                US[i],
                                S2[i],
                                perc_left=perc_left,
                                perc_right=perc_right,
                                normalize=True,
                            )
                    else:
                        pool = ThreadPool(cores)
                        res = pool.starmap(self.fit_gamma_stochastic,
                                           zip(itertools.repeat(self.est_method), U, S, US, S2, itertools.repeat(perc_left),
                                               itertools.repeat(perc_right), itertools.repeat(True)))
                        pool.close()
                        pool.join()
                        (gamma, gamma_intercept, _, gamma_r2, _, gamma_logLL) = zip(*res)
                        (gamma, gamma_intercept, gamma_r2, gamma_logLL) = np.array(gamma), np.array(gamma_intercept), \
                                                                          np.array(gamma_r2), np.array(gamma_logLL)
                    (
                        self.parameters["gamma"],
                        self.aux_param["gamma_intercept"],
                        self.aux_param["gamma_r2"],
                        self.aux_param["gamma_logLL"],
                    ) = (gamma, gamma_intercept, gamma_r2, gamma_logLL)
                elif np.all(self._exist_data("uu", "ul")):
                    self.parameters["beta"] = np.ones(n)
                    gamma, gamma_intercept, gamma_r2, gamma_logLL = (
                        np.zeros(n),
                        np.zeros(n),
                        np.zeros(n),
                        np.zeros(n),
                    )
                    U = self.data["ul"]
                    S = self.data["uu"] + self.data["ul"]
                    US = self.data['us'] if 'us' in self.data.keys() else calc_2nd_moment(
                        U.T, S.T, self.conn, mX=U.T, mY=S.T
                    ).T
                    S2 = self.data['s2'] if 's2' in self.data.keys() else calc_2nd_moment(
                        S.T, S.T, self.conn, mX=S.T, mY=S.T
                    ).T
                    if cores == 1:
                        for i in tqdm(range(n), desc="estimating gamma"):
                            (
                                gamma[i],
                                gamma_intercept[i],
                                _,
                                gamma_r2[i],
                                _,
                                gamma_logLL[i],
                            ) = self.fit_gamma_stochastic(
                                self.est_method,
                                U[i],
                                S[i],
                                US[i],
                                S2[i],
                                perc_left=perc_left,
                                perc_right=perc_right,
                                normalize=True,
                            )
                    else:
                        pool = ThreadPool(cores)
                        res = pool.starmap(self.fit_gamma_stochastic,
                                     zip(itertools.repeat(self.est_method), U, S, US, S2, itertools.repeat(perc_left),
                                         itertools.repeat(perc_right), itertools.repeat(True)))
                        pool.close()
                        pool.join()
                        (gamma, gamma_intercept, _, gamma_r2, _, gamma_logLL) = zip(*res)
                        (gamma, gamma_intercept, gamma_r2, gamma_logLL) = np.array(gamma), np.array(gamma_intercept), \
                                                                          np.array(gamma_r2), np.array(gamma_logLL)
                    (
                        self.parameters["gamma"],
                        self.aux_param["gamma_intercept"],
                        self.aux_param["gamma_r2"],
                        self.aux_param["gamma_logLL"],
                    ) = (gamma, gamma_intercept, gamma_r2, gamma_logLL)
        else:
            if self.extyp.lower() == "deg":
                if np.all(self._exist_data("ul", "sl")):
                    # beta & gamma estimation
                    ul_m, ul_v, t_uniq = calc_12_mom_labeling(self.data["ul"], self.t)
                    sl_m, sl_v, _ = calc_12_mom_labeling(self.data["sl"], self.t)
                    (
                        self.parameters["beta"],
                        self.parameters["gamma"],
                        self.aux_param["ul0"],
                        self.aux_param["sl0"],
                    ) = self.fit_beta_gamma_lsq(t_uniq, ul_m, sl_m)
                    if self._exist_data("uu"):
                        # alpha estimation
                        uu_m, uu_v, _ = calc_12_mom_labeling(self.data["uu"], self.t)
                        alpha, uu0, r2 = np.zeros((n, 1)), np.zeros(n), np.zeros(n)
                        if cores == 1:
                            for i in range(n):
                                alpha[i], uu0[i], r2[i] = fit_alpha_degradation(
                                    t_uniq,
                                    uu_m[i],
                                    self.parameters["beta"][i],
                                    intercept=True,
                                )
                        else:
                            pool = ThreadPool(cores)
                            res = pool.starmap(fit_alpha_degradation, zip(itertools.repeat(t_uniq), uu_m,
                                                 self.parameters["beta"], itertools.repeat(True)))
                            pool.close()
                            pool.join()
                            (alpha, uu0, r2) = zip(*res)
                            (alpha, uu0, r2) = np.array(alpha), np.array(uu0), np.array(r2)
                        (
                            self.parameters["alpha"],
                            self.aux_param["alpha_intercept"],
                            self.aux_param["uu0"],
                            self.aux_param["alpha_r2"],
                        ) = (alpha, uu0, uu0, r2)
                elif self._exist_data("ul"):
                    # gamma estimation
                    # use mean + var for fitting degradation parameter k
                    ul_m, ul_v, t_uniq = calc_12_mom_labeling(self.data["ul"], self.t)
                    (
                        self.parameters["gamma"],
                        self.aux_param["ul0"],
                    ) = self.fit_gamma_nosplicing_lsq(t_uniq, ul_m)
                    if self._exist_data("uu"):
                        # alpha estimation
                        alpha, alpha_b, alpha_r2 = np.zeros(n), np.zeros(n), np.zeros(n)
                        uu_m, uu_v, _ = calc_12_mom_labeling(self.data["uu"], self.t)
                        if cores == 1:
                            for i in tqdm(range(n), desc="estimating alpha"):
                                alpha[i], alpha_b[i], alpha_r2[i] = fit_alpha_degradation(
                                    t_uniq, uu_m[i], self.parameters["gamma"][i], intercept=True
                                )
                        else:
                            pool = ThreadPool(cores)
                            res = pool.starmap(fit_alpha_degradation, zip(itertools.repeat(t_uniq), uu_m,
                                                 self.parameters["gamma"], itertools.repeat(True)))
                            pool.close()
                            pool.join()
                            (alpha, alpha_b, alpha_r2) = zip(*res)
                            (alpha, alpha_b, alpha_r2) = np.array(alpha), np.array(alpha_b), np.array(alpha_r2)
                        (
                            self.parameters["alpha"],
                            self.aux_param["alpha_intercept"],
                            self.aux_param["uu0"],
                            self.aux_param["alpha_r2"],
                        ) = (alpha, alpha_b, alpha_b, alpha_r2)
            elif (self.extyp.lower() == "kin" or self.extyp.lower() == "one-shot") and len(
                np.unique(self.t)
            ) > 1:
                if np.all(self._exist_data("ul", "uu", "su")):
                    if not self._exist_parameter("beta"):
                        warn(
                            "beta & gamma estimation: only works when there're at least 2 time points."
                        )
                        uu_m, uu_v, t_uniq = calc_12_mom_labeling(
                            self.data["uu"], self.t
                        )
                        su_m, su_v, _ = calc_12_mom_labeling(self.data["su"], self.t)

                        (
                            self.parameters["beta"],
                            self.parameters["gamma"],
                            self.aux_param["uu0"],
                            self.aux_param["su0"],
                        ) = self.fit_beta_gamma_lsq(t_uniq, uu_m, su_m)
                    # alpha estimation
                    ul_m, ul_v, t_uniq = calc_12_mom_labeling(self.data["ul"], self.t)
                    alpha = np.zeros(n)
                    # let us only assume one alpha for each gene in all cells
                    if cores == 1:
                        for i in tqdm(range(n), desc="estimating alpha"):
                            # for j in range(len(self.data['ul'][i])):
                            alpha[i] = fit_alpha_synthesis(
                                t_uniq, ul_m[i], self.parameters["beta"][i]
                            )
                    else:
                        pool = ThreadPool(cores)
                        alpha = pool.starmap(fit_alpha_synthesis, zip(itertools.repeat(t_uniq), ul_m,
                                                                      self.parameters["beta"]))
                        pool.close()
                        pool.join()
                        alpha = np.array(alpha)
                    self.parameters["alpha"] = alpha
                elif np.all(self._exist_data("ul", "uu")):
                    n = self.data["uu"].shape[0]  # self.get_n_genes(data=U)
                    u0, gamma = np.zeros(n), np.zeros(n)
                    uu_m, uu_v, t_uniq = calc_12_mom_labeling(self.data["uu"], self.t)
                    for i in tqdm(range(n), desc="estimating gamma"):
                        try:
                            gamma[i], u0[i] = fit_first_order_deg_lsq(t_uniq, uu_m[i])
                        except:
                            gamma[i], u0[i] = 0, 0
                    self.parameters["gamma"], self.aux_param["uu0"] = gamma, u0
                    alpha = np.zeros(n)
                    # let us only assume one alpha for each gene in all cells
                    ul_m, ul_v, _ = calc_12_mom_labeling(self.data["ul"], self.t)
                    if cores == 1:
                        for i in tqdm(range(n), desc="estimating gamma"):
                            # for j in range(len(self.data['ul'][i])):
                            alpha[i] = fit_alpha_synthesis(
                                t_uniq, ul_m[i], self.parameters["gamma"][i]
                            )
                    else:
                        pool = ThreadPool(cores)
                        alpha = pool.starmap(fit_alpha_synthesis, zip(itertools.repeat(t_uniq), ul_m,
                                                                    self.parameters["gamma"]))
                        pool.close()
                        pool.join()
                        alpha = np.array(alpha)
                    self.parameters["alpha"] = alpha
                    # alpha: one-shot
            # 'one_shot'
            elif self.extyp.lower() == "one-shot":
                t_uniq = np.unique(self.t)
                if len(t_uniq) > 1:
                    raise Exception(
                        "By definition, one-shot experiment should involve only one time point measurement!"
                    )
                # calculate when having splicing or no splicing
                if self.model.lower() == "deterministic":
                    if np.all(self._exist_data("ul", "uu", "su")):
                        if self._exist_parameter("beta", "gamma").all():
                            self.parameters["alpha"] = self.fit_alpha_oneshot(
                                self.t, self.data["ul"], self.parameters["beta"], clusters
                            )
                        else:
                            beta, gamma, U0, S0 = (
                                np.zeros(n),
                                np.zeros(n),
                                np.zeros(n),
                                np.zeros(n),
                            )
                            for i in range(
                                n
                            ):  # can also use the two extreme time points and apply sci-fate like approach.
                                S, U = (
                                    self.data["su"][i] + self.data["sl"][i],
                                    self.data["uu"][i] + self.data["ul"][i],
                                )

                                S0[i], gamma[i] = (
                                    np.mean(S),
                                    solve_gamma(np.max(self.t), self.data["su"][i], S),
                                )
                                U0[i], beta[i] = (
                                    np.mean(U),
                                    solve_gamma(np.max(self.t), self.data["uu"][i], U),
                                )
                            (
                                self.aux_param["U0"],
                                self.aux_param["S0"],
                                self.parameters["beta"],
                                self.parameters["gamma"],
                            ) = (U0, S0, beta, gamma)

                            ul_m, ul_v, t_uniq = calc_12_mom_labeling(
                                self.data["ul"], self.t
                            )
                            alpha = np.zeros(n)
                            # let us only assume one alpha for each gene in all cells
                            if cores == 1:
                                for i in tqdm(range(n), desc="estimating alpha"):
                                    # for j in range(len(self.data['ul'][i])):
                                    alpha[i] = fit_alpha_synthesis(
                                        t_uniq, ul_m[i], self.parameters["beta"][i]
                                    )
                            else:
                                pool = ThreadPool(cores)
                                alpha = pool.starmap(fit_alpha_synthesis,
                                                   zip(itertools.repeat(t_uniq), ul_m, self.parameters["beta"]))
                                pool.close()
                                pool.join()
                                alpha = np.array(alpha)
                            self.parameters["alpha"] = alpha
                            # self.parameters['alpha'] = self.fit_alpha_oneshot(self.t, self.data['ul'], self.parameters['beta'], clusters)
                    else:
                        if self._exist_data("ul") and self._exist_parameter("gamma"):
                            self.parameters["alpha"] = self.fit_alpha_oneshot(
                                self.t, self.data["ul"], self.parameters["gamma"], clusters
                            )
                        elif self._exist_data("ul") and self._exist_data("uu"):
                            if one_shot_method in ["sci-fate", "sci_fate"]:
                                gamma, total0 = np.zeros(n), np.zeros(n)
                                for i in tqdm(range(n), desc="estimating gamma"):
                                    total = self.data["uu"][i] + self.data["ul"][i]
                                    total0[i], gamma[i] = (
                                        np.mean(total),
                                        solve_gamma(
                                            np.max(self.t), self.data["uu"][i], total
                                        ),
                                    )
                                self.aux_param["total0"], self.parameters["gamma"] = (
                                    total0,
                                    gamma,
                                )

                                ul_m, ul_v, t_uniq = calc_12_mom_labeling(
                                    self.data["ul"], self.t
                                )
                                # let us only assume one alpha for each gene in all cells
                                alpha = np.zeros(n)
                                if cores == 1:
                                    for i in tqdm(range(n), desc="estimating alpha"):
                                        # for j in range(len(self.data['ul'][i])):
                                        alpha[i] = fit_alpha_synthesis(
                                            t_uniq, ul_m[i], self.parameters["gamma"][i]
                                        )  # ul_m[i] / t_uniq
                                else:
                                    pool = ThreadPool(cores)
                                    alpha = pool.starmap(fit_alpha_synthesis, zip(itertools.repeat(t_uniq), ul_m,
                                                                                  self.parameters["gamma"]))
                                    pool.close()
                                    pool.join()
                                    alpha = np.array(alpha)
                                self.parameters["alpha"] = alpha
                                # self.parameters['alpha'] = self.fit_alpha_oneshot(self.t, self.data['ul'], self.parameters['gamma'], clusters)
                            elif one_shot_method == "combined":
                                self.parameters["alpha"] = (
                                    csr_matrix(self.data["ul"].shape)
                                    if issparse(self.data["ul"])
                                    else np.zeros_like(self.data["ul"].shape)
                                )
                                t_uniq, gamma, gamma_intercept, gamma_r2, gamma_logLL = (
                                    np.unique(self.t),
                                    np.zeros(n),
                                    np.zeros(n),
                                    np.zeros(n),
                                    np.zeros(n),
                                )
                                U, S = self.data["ul"], self.data["uu"] + self.data["ul"]

                                if cores == 1:
                                    for i in tqdm(range(n), desc="estimating gamma"):
                                        (
                                            k,
                                            gamma_intercept[i],
                                            _,
                                            gamma_r2[i],
                                            _,
                                            gamma_logLL[i],
                                        ) = self.fit_gamma_steady_state(
                                            U[i], S[i], False, None, perc_right
                                        )
                                        (
                                            gamma[i],
                                            self.parameters["alpha"][i],
                                        ) = one_shot_gamma_alpha(k, t_uniq, U[i])
                                else:
                                    pool = ThreadPool(cores)
                                    res1 = pool.starmap(self.fit_gamma_steady_state, zip(U, S, itertools.repeat(False),
                                                    itertools.repeat(None), itertools.repeat(perc_right)))

                                    (k, gamma_intercept, _, gamma_r2, _, gamma_logLL) = zip(*res1)
                                    (k, gamma_intercept, gamma_r2, gamma_logLL) = np.array(k), np.array(gamma_intercept), \
                                                                                  np.array(gamma_r2), np.array(gamma_logLL)

                                    res2 = pool.starmap(one_shot_gamma_alpha, zip(k, itertools.repeat(t_uniq), U))

                                    (gamma, alpha) = zip(*res2)
                                    (gamma, self.parameters["alpha"]) = np.array(gamma), np.array(alpha)

                                    pool.close()
                                    pool.join()
                                (
                                    self.parameters["gamma"],
                                    self.aux_param["gamma_r2"],
                                    self.aux_param["gamma_logLL"],
                                    self.aux_param["alpha_r2"],
                                ) = (gamma, gamma_r2, gamma_logLL, gamma_r2)
                elif self.model.lower() == "stochastic":
                    if np.all(self._exist_data("uu", "ul", "su", "sl")):
                        self.parameters["beta"] = np.ones(n)
                        k, k_intercept, k_r2, k_logLL = (
                            np.zeros(n),
                            np.zeros(n),
                            np.zeros(n),
                            np.zeros(n),
                        )
                        U = self.data["uu"]
                        S = self.data["uu"] + self.data["ul"]
                        US = self.data['us'] if 'us' in self.data.keys() else calc_2nd_moment(
                            U.T, S.T, self.conn, mX=U.T, mY=S.T
                        ).T
                        S2 = self.data['s2'] if 's2' in self.data.keys() else calc_2nd_moment(
                            S.T, S.T, self.conn, mX=S.T, mY=S.T
                        ).T
                        if cores == 1:
                            for i in tqdm(range(n), desc="estimating beta and alpha for one-shot experiment"):
                                (
                                    k[i],
                                    k_intercept[i],
                                    _,
                                    k_r2[i],
                                    _,
                                    k_logLL[i],
                                ) = self.fit_gamma_stochastic(
                                    self.est_method,
                                    U[i],
                                    S[i],
                                    US[i],
                                    S2[i],
                                    perc_left=perc_left,
                                    perc_right=perc_right,
                                    normalize=True,
                                )
                            else:
                                pool = ThreadPool(cores)
                                res = pool.starmap(self.fit_gamma_stochastic, zip(itertools.repeat(self.est_method),
                                                                              U, S, US, S2,
                                                                              itertools.repeat(perc_left),
                                                                              itertools.repeat(perc_right),
                                                                              itertools.repeat(True)))
                                pool.close()
                                pool.join()
                                (k, k_intercept, _, k_r2, _, k_logLL) = zip(*res)
                                (k, k_intercept, k_r2, k_logLL) = np.array(k), np.array(k_intercept), \
                                                                              np.array(k_r2), np.array(k_logLL)
                        beta, alpha0 = one_shot_gamma_alpha_matrix(k, t_uniq, U)

                        self.parameters["beta"], self.aux_param["beta_k"] = beta, k

                        U = self.data["uu"] + self.data["ul"]
                        S = U + self.data["su"] + self.data["sl"]
                        US = self.data['us'] if 'us' in self.data.keys() else calc_2nd_moment(
                            U.T, S.T, self.conn, mX=U.T, mY=S.T
                        ).T
                        S2 = self.data['s2'] if 's2' in self.data.keys() else calc_2nd_moment(
                            S.T, S.T, self.conn, mX=S.T, mY=S.T
                        ).T
                        if cores == 1:
                            for i in tqdm(range(n), desc="estimating gamma and alpha for one-shot experiment"):
                                (
                                    k[i],
                                    k_intercept[i],
                                    _,
                                    k_r2[i],
                                    _,
                                    k_logLL[i],
                                ) = self.fit_gamma_stochastic(
                                    self.est_method,
                                    U[i],
                                    S[i],
                                    US[i],
                                    S2[i],
                                    perc_left=perc_left,
                                    perc_right=perc_right,
                                    normalize=True,
                                )
                        else:
                            pool = ThreadPool(cores)
                            res = pool.starmap(self.fit_gamma_stochastic, zip(itertools.repeat(self.est_method),
                                                                              U, S, US, S2,
                                                                              itertools.repeat(perc_left),
                                                                              itertools.repeat(perc_right),
                                                                              itertools.repeat(True)))
                            pool.close()
                            pool.join()
                            (k, k_intercept, _, k_r2, _, k_logLL) = zip(*res)
                            (k, k_intercept, k_r2, k_logLL) = np.array(k), np.array(k_intercept), \
                                                              np.array(k_r2), np.array(k_logLL)
                        gamma, alpha = one_shot_gamma_alpha_matrix(k, t_uniq, U)
                        (
                            self.parameters["alpha"],
                            self.parameters["gamma"],
                            self.aux_param["gamma_k"],
                            self.aux_param["gamma_intercept"],
                            self.aux_param["gamma_r2"],
                            self.aux_param["gamma_logLL"],
                        ) = ((alpha + alpha0) / 2, gamma, k, k_intercept, k_r2, k_logLL)
                    elif np.all(self._exist_data("uu", "ul")):
                        k, k_intercept, k_r2, k_logLL = (
                            np.zeros(n),
                            np.zeros(n),
                            np.zeros(n),
                            np.zeros(n),
                        )
                        U = self.data["ul"]
                        S = self.data["ul"] + self.data["uu"]
                        US = self.data['us'] if 'us' in self.data.keys() else calc_2nd_moment(
                            U.T, S.T, self.conn, mX=U.T, mY=S.T
                        ).T
                        S2 = self.data['s2'] if 's2' in self.data.keys() else calc_2nd_moment(
                            S.T, S.T, self.conn, mX=S.T, mY=S.T
                        ).T
                        if cores == 1:
                            for i in tqdm(range(n), desc="estimating gamma"):
                                (
                                    k[i],
                                    k_intercept[i],
                                    _,
                                    k_r2[i],
                                    _,
                                    k_logLL[i],
                                ) = self.fit_gamma_stochastic(
                                    self.est_method,
                                    U[i],
                                    S[i],
                                    US[i],
                                    S2[i],
                                    perc_left=perc_left,
                                    perc_right=perc_right,
                                    normalize=True,
                                )
                        else:
                            pool = ThreadPool(cores)
                            res = pool.starmap(self.fit_gamma_stochastic, zip(itertools.repeat(self.est_method),
                                                                              U, S, US, S2,
                                                                              itertools.repeat(perc_left),
                                                                              itertools.repeat(perc_right),
                                                                              itertools.repeat(True)))
                            pool.close()
                            pool.join()
                            (k, k_intercept, _, k_r2, _, k_logLL) = zip(*res)
                            (k, k_intercept, k_r2, k_logLL) = np.array(k), np.array(k_intercept), \
                                                              np.array(k_r2), np.array(k_logLL)
                        gamma, alpha = one_shot_gamma_alpha_matrix(k, t_uniq, U)
                        (
                            self.parameters["alpha"],
                            self.parameters["gamma"],
                            self.aux_param["gamma_k"],
                            self.aux_param["gamma_intercept"],
                            self.aux_param["gamma_r2"],
                            self.aux_param["gamma_logLL"],
                        ) = (alpha, gamma, k, k_intercept, k_r2, k_logLL)
            elif self.extyp.lower() == "mix_std_stm":
                t_min, t_max = np.min(self.t), np.max(self.t)
                if np.all(self._exist_data("ul", "uu", "su")):
                    gamma, beta, total, U = (
                        np.zeros(n),
                        np.zeros(n),
                        np.zeros(n),
                        np.zeros(n),
                    )
                    for i in tqdm(
                        range(n), desc="solving gamma/beta"
                    ):  # can also use the two extreme time points and apply sci-fate like approach.
                        tmp = (
                            self.data["uu"][i, self.t == t_max]
                            + self.data["ul"][i, self.t == t_max]
                            + self.data["su"][i, self.t == t_max]
                            + self.data["sl"][i, self.t == t_max]
                        )
                        total[i] = np.mean(tmp)
                        gamma[i] = solve_gamma(
                            t_max,
                            self.data["uu"][i, self.t == t_max]
                            + self.data["su"][i, self.t == t_max],
                            tmp,
                        )
                        # same for beta
                        tmp = (
                            self.data["uu"][i, self.t == t_max]
                            + self.data["ul"][i, self.t == t_max]
                        )
                        U[i] = np.mean(tmp)
                        beta[i] = solve_gamma(
                            np.max(self.t), self.data["uu"][i, self.t == t_max], tmp
                        )

                    (
                        self.parameters["beta"],
                        self.parameters["gamma"],
                        self.aux_param["total0"],
                        self.aux_param["U0"],
                    ) = (beta, gamma, total, U)
                    # alpha estimation
                    self.parameters["alpha"] = self.solve_alpha_mix_std_stm(
                        self.t, self.data["ul"], self.parameters["beta"]
                    )
                elif np.all(self._exist_data("ul", "uu")):
                    n = self.data["uu"].shape[0]  # self.get_n_genes(data=U)
                    gamma, U = np.zeros(n), np.zeros(n)
                    for i in tqdm(
                        range(n), desc="solving gamma, alpha"
                    ):  # apply sci-fate like approach (can also use one-single time point to estimate gamma)
                        # tmp = self.data['uu'][i, self.t == 0] + self.data['ul'][i, self.t == 0]
                        tmp_ = (
                            self.data["uu"][i, self.t == t_max]
                            + self.data["ul"][i, self.t == t_max]
                        )

                        U[i] = np.mean(tmp_)
                        # gamma_1 = solve_gamma(np.max(self.t), self.data['uu'][i, self.t == 0], tmp) # steady state
                        gamma_2 = solve_gamma(
                            t_max, self.data["uu"][i, self.t == t_max], tmp_
                        )  # stimulation
                        # gamma_3 = solve_gamma(np.max(self.t), self.data['uu'][i, self.t == np.max(self.t)], tmp) # sci-fate
                        gamma[i] = gamma_2
                        # print('Steady state, stimulation, sci-fate like gamma values are ', gamma_1, '; ', gamma_2, '; ', gamma_3)
                    (
                        self.parameters["gamma"],
                        self.aux_param["U0"],
                        self.parameters["beta"],
                    ) = (gamma, U, np.ones(gamma.shape))
                    # alpha estimation
                    self.parameters["alpha"] = self.solve_alpha_mix_std_stm(
                        self.t, self.data["ul"], self.parameters["gamma"]
                    )

        # fit protein
        if np.all(self._exist_data("p", "su")):
            ind_for_proteins = self.ind_for_proteins
            n = len(ind_for_proteins) if ind_for_proteins is not None else 0

            if self.asspt_prot.lower() == "ss" and n > 0:
                self.parameters["eta"] = np.ones(n)
                (
                    delta,
                    delta_intercept,
                    delta_r2,
                    delta_logLL,
                ) = (
                    np.zeros(n),
                    np.zeros(n),
                    np.zeros(n),
                    np.zeros(n),
                )

                s = (
                    self.data["su"][ind_for_proteins]
                    + self.data["sl"][ind_for_proteins]
                    if self._exist_data("sl")
                    else self.data["su"][ind_for_proteins]
                )
                if cores == 1:
                    for i in tqdm(range(n), desc="estimating delta"):
                        (
                            delta[i],
                            delta_intercept[i],
                            _,
                            delta_r2[i],
                            _,
                            delta_logLL[i],
                        ) = self.fit_gamma_steady_state(
                            s[i], self.data["p"][i], intercept, perc_left, perc_right
                        )
                else:
                    pool = ThreadPool(cores)
                    res = pool.starmap(self.fit_gamma_steady_state,
                                       zip(s, self.data["p"], itertools.repeat(intercept), itertools.repeat(perc_left),
                                           itertools.repeat(perc_right)))
                    pool.close()
                    pool.join()
                    (delta, delta_intercept, _, delta_r2, _, delta_logLL) = zip(*res)
                    (delta, delta_intercept, delta_r2, delta_logLL) = np.array(delta), np.array(delta_intercept), \
                                                      np.array(delta_r2), np.array(delta_logLL)
                (
                    self.parameters["delta"],
                    self.aux_param["delta_intercept"],
                    self.aux_param["delta_r2"],
                    _, # self.aux_param["delta_logLL"],
                ) = (delta, delta_intercept, delta_r2, delta_logLL)

    def fit_gamma_steady_state(
        self, u, s, intercept=True, perc_left=None, perc_right=5, normalize=True
    ):
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
                The percentage of samples included in the linear regression in the left tail. If set to None, then all the
                left samples are excluded.
            perc_right: float
                The percentage of samples included in the linear regression in the right tail. If set to None, then all the
                samples are included.
            normalize: bool
                Whether to first normalize the

        Returns
        -------
            k: float
                The slope of the linear regression model, which is gamma under the steady state assumption.
            b: float
                The intercept of the linear regression model.
            r2: float
                Coefficient of determination or r square for the extreme data points.
            r2: float
                Coefficient of determination or r square for the extreme data points.
            all_r2: float
                Coefficient of determination or r square for all data points.
        """
        if intercept and perc_left is None:
            perc_left = perc_right
        u = u.A.flatten() if issparse(u) else u.flatten()
        s = s.A.flatten() if issparse(s) else s.flatten()

        mask = find_extreme(
            s, u, normalize=normalize, perc_left=perc_left, perc_right=perc_right
        )

        if self.est_method.lower() == 'ols':
            k, b, r2, all_r2 = fit_linreg(s, u, mask, intercept)
        else:
            k, b, r2, all_r2 = fit_linreg_robust(s, u, mask, intercept, self.est_method)

        logLL, all_logLL = calc_norm_loglikelihood(s[mask], u[mask], k), calc_norm_loglikelihood(s, u, k)

        return k, b, r2, all_r2, logLL, all_logLL

    def fit_gamma_stochastic(
        self, est_method, u, s, us, ss, perc_left=None, perc_right=5, normalize=True
    ):
        """Estimate gamma using GMM (generalized method of moments) or negbin distrubtion based on the steady state assumption.

        Arguments
        ---------
            est_method: `str` {`gmm`, `negbin`} The estimation method to be used when using the `stochastic` model.
                * Available options when the `model` is 'ss' include:
                (2) 'gmm': The new generalized methods of moments from us that is based on master equations, similar to the
                "moment" model in the excellent scVelo package;
                (3) 'negbin': The new method from us that models steady state RNA expression as a negative binomial distribution,
                also built upon on master equations.
                Note that all those methods require using extreme data points (except negbin, which use all data points) for
                estimation. Extreme data points are defined as the data from cells whose expression of unspliced / spliced
                or new / total RNA, etc. are in the top or bottom, 5%, for example. `linear_regression` only considers the mean of
                RNA species (based on the `deterministic` ordinary different equations) while moment based methods (`gmm`, `negbin`)
                considers both first moment (mean) and second moment (uncentered variance) of RNA species (based on the `stochastic`
                master equations).
                The above method are all (generalized) linear regression based method. In order to return estimated parameters
                (including RNA half-life), it additionally returns R-squared (either just for extreme data points or all data points)
                as well as the log-likelihood of the fitting, which will be used for transition matrix and velocity embedding.
                All `est_method` uses least square to estimate optimal parameters with latin cubic sampler for initial sampling.
            u: :class:`~numpy.ndarray` or sparse `csr_matrix`
                A matrix of unspliced mRNA counts. Dimension: genes x cells.
            s: :class:`~numpy.ndarray` or sparse `csr_matrix`
                A matrix of spliced mRNA counts. Dimension: genes x cells.
            us: :class:`~numpy.ndarray` or sparse `csr_matrix`
                A matrix of unspliced mRNA counts. Dimension: genes x cells.
            ss: :class:`~numpy.ndarray` or sparse `csr_matrix`
                A matrix of spliced mRNA counts. Dimension: genes x cells.
            perc_left: float
                The percentage of samples included in the linear regression in the left tail. If set to None, then all the left samples are excluded.
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
                Coefficient of determination or r square for the extreme data points.
            r2: float
                Coefficient of determination or r square for the extreme data points.
            all_r2: float
                Coefficient of determination or r square for all data points.
        """
        u = u.A.flatten() if issparse(u) else u.flatten()
        s = s.A.flatten() if issparse(s) else s.flatten()
        us = us.A.flatten() if issparse(us) else us.flatten()
        ss = ss.A.flatten() if issparse(ss) else ss.flatten()

        mask = find_extreme(
            s, u, normalize=normalize, perc_left=perc_left, perc_right=perc_right
        )

        if est_method.lower() == 'gmm':
            k = fit_stochastic_linreg(u[mask], s[mask], us[mask], ss[mask])
        elif est_method.lower() == 'negbin':
            phi = compute_dispersion(s, ss)
            k = fit_k_negative_binomial(u[mask], s[mask],  ss[mask], phi)

        r2, all_r2 = calc_R2(s[mask], u[mask], k), calc_R2(s, u, k)
        logLL, all_logLL = calc_norm_loglikelihood(s[mask], u[mask], k), calc_norm_loglikelihood(s, u, k)

        return k, 0, r2, all_r2, logLL, all_logLL

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
        n = U.shape[0]  # self.get_n_genes(data=U)
        beta = np.zeros(n)
        gamma = np.zeros(n)
        u0, s0 = np.zeros(n), np.zeros(n)

        for i in tqdm(range(n), desc="estimating beta, gamma"):
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
        n = L.shape[0]  # self.get_n_genes(data=L)
        gamma = np.zeros(n)
        l0 = np.zeros(n)

        for i in tqdm(range(n), desc="estimating gamma"):
            gamma[i], l0[i] = (
                fit_first_order_deg_lsq(t, L[i].A[0])
                if issparse(L)
                else fit_first_order_deg_lsq(t, L[i])
            )
        return gamma, l0

    def solve_alpha_mix_std_stm(
        self, t, ul, beta, clusters=None, alpha_time_dependent=True
    ):
        """Estimate the steady state transcription rate and analytically calculate the stimulation transcription rate
        given beta and steady state alpha for a mixed steady state and stimulation labeling experiment. 
        
        This approach assumes the same constant beta or gamma for both steady state or stimulation period.

        Arguments
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
        t_std, t_stm, t_uniq, t_max, t_min = (
            np.max(t) - t,
            t,
            np.unique(t),
            np.max(t),
            np.min(t),
        )

        alpha_std_ini = self.fit_alpha_oneshot(
            np.array([t_max]), np.mean(ul[:, t == t_min], 1), beta, clusters
        ).flatten()
        alpha_std, alpha_stm = alpha_std_ini, np.zeros((ul.shape[0], len(t_uniq)))
        alpha_stm[
            :, 0
        ] = alpha_std_ini  # 0 stimulation point is the steady state transcription
        for i in tqdm(
            range(ul.shape[0]), desc="solving steady state alpha and induction alpha"
        ):
            l = ul[i].A.flatten() if issparse(ul) else ul[i]
            for t_ind in np.arange(1, len(t_uniq)):
                alpha_stm[i, t_ind] = solve_alpha_2p(
                    t_max - t_uniq[t_ind],
                    t_uniq[t_ind],
                    alpha_std[i],
                    beta[i],
                    l[t == t_uniq[t_ind]],
                )
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
            for j in tqdm(range(n_genes), desc="estimating alpha"):
                if len(c) > 0:
                    alpha[j, i] = (
                        fit_alpha_synthesis(t, U[j].A[0][c], beta[j])
                        if issparse(U)
                        else fit_alpha_synthesis(t, U[j][c], beta[j])
                    )
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
                    self.data[k], self.t = concat_time_series_matrices(
                        self.data[k], self.t
                    )
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
            ret = np.array(
                [self.parameters[k] is not None for k in param_names], dtype=bool
            )
        return ret

    def get_exist_data_names(self):
        """Get the names of all the data that are not 'None'."""
        ret = []
        for k, v in self.data.items():
            if v is not None:
                ret.append(k)
        return ret
