from scvelo import settings
from scvelo import logging as logg
from scvelo.preprocessing.moments import get_connectivities
from scvelo.tools.utils import make_unique_list, test_bimodality
from scvelo.tools.dynamical_model_utils import (
    BaseDynamics,
    linreg,
    convolve,
    tau_inv,
    unspliced,
)
from scvelo.tools.terminal_states import eigs

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
from scipy.optimize import minimize


class DynamicsRecovery(BaseDynamics):
    def __init__(self, adata=None, gene=None, load_pars=None, **kwargs):
        super(DynamicsRecovery, self).__init__(adata, gene, **kwargs)
        if load_pars and "fit_alpha" in adata.var.keys():
            self.load_pars(adata, gene)
        elif self.recoverable:
            self.initialize()

    def initialize(self):
        # set weights
        u, s, w, perc = self.u, self.s, self.weights, 98
        u_w, s_w, = (
            u[w],
            s[w],
        )

        # initialize scaling
        self.std_u, self.std_s = np.std(u_w), np.std(s_w)
        if self.std_u == 0 or self.std_s == 0:
            self.std_u = self.std_s = 1
        _scaling = self.fit_scaling
        scaling = self.std_u / self.std_s if isinstance(_scaling, bool) else _scaling
        u, u_w = u / scaling, u_w / scaling

        # initialize beta and gamma from extreme quantiles of s
        weights_s = s_w >= np.percentile(s_w, perc, axis=0)
        weights_u = u_w >= np.percentile(u_w, perc, axis=0)

        _prior = self.steady_state_prior
        weights_g = weights_s if _prior is None else weights_s | _prior[w]
        beta = 1
        gamma = linreg(convolve(u_w, weights_g), convolve(s_w, weights_g)) + 1e-6
        # 1e-6 to avoid beta = gamma
        # initialize gamma / beta * scaling clipped to adapt faster to extreme ratios
        if gamma < 0.05 / scaling:
            gamma *= 1.2
        elif gamma > 1.5 / scaling:
            gamma /= 1.2

        u_inf, s_inf = u_w[weights_u | weights_s].mean(), s_w[weights_s].mean()
        u0_, s0_ = u_inf, s_inf
        alpha = u_inf * beta
        # np.mean([s_inf * gamma, u_inf * beta])  # np.mean([s0_ * gamma, u0_ * beta])

        if self.init_vals is not None:  # just to test different EM start
            self.init_vals = np.array(self.init_vals)
            alpha, beta, gamma = np.array([alpha, beta, gamma]) * self.init_vals

        # initialize switching from u quantiles and alpha from s quantiles
        try:
            tstat_u, pval_u, means_u = test_bimodality(u_w, kde=True)
            tstat_s, pval_s, means_s = test_bimodality(s_w, kde=True)
        except:
            logg.warn("skipping bimodality check for", self.gene)
            tstat_u, tstat_s, pval_u, pval_s = 0, 0, 1, 1
            means_u, means_s = [0, 0], [0, 0]

        self.pval_steady = max(pval_u, pval_s)
        self.steady_u = means_u[1]
        self.steady_s = means_s[1]

        if self.pval_steady < 1e-3:
            u_inf = np.mean([u_inf, self.steady_u])
            alpha = gamma * s_inf
            beta = alpha / u_inf
            u0_, s0_ = u_inf, s_inf

        # alpha, beta, gamma = np.array([alpha, beta, gamma]) * scaling
        t_ = tau_inv(u0_, s0_, 0, 0, alpha, beta, gamma)

        # update object with initialized vars
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.scaling, self.alpha_ = scaling, 0
        self.u0_, self.s0_, self.t_ = u0_, s0_, t_
        self.pars = np.array([alpha, beta, gamma, self.t_, self.scaling])[:, None]

        # initialize time point assignment
        self.t, self.tau, self.o = self.get_time_assignment()
        self.loss = [self.get_loss()]

        self.initialize_scaling(sight=0.5)
        self.initialize_scaling(sight=0.1)

        self.steady_state_ratio = self.gamma / self.beta

        self.set_callbacks()

    def initialize_scaling(self, sight=0.5):  # fit scaling and update if improved
        z_vals = self.scaling + np.linspace(-1, 1, num=4) * self.scaling * sight
        for z in z_vals:
            self.update(scaling=z, beta=self.beta / self.scaling * z)

    def fit(self, assignment_mode=None):
        if self.max_iter > 0:
            # for comparison with exact time assignment
            if assignment_mode == "full_projection":
                self.assignment_mode = assignment_mode

            # pre-train with explicit time assignment
            self.fit_t_and_alpha()
            self.fit_scaling_()
            self.fit_rates()
            self.fit_t_()

            # actual EM (each iteration of simplex downhill is
            self.fit_t_and_rates()

            # train with optimal time assignment (oth. projection)
            self.assignment_mode = assignment_mode
            self.update(adjust_t_=False)
            self.fit_t_and_rates(refit_time=False)

        # self.update(adjust_t_=False)
        # self.t, self.tau, self.o = self.get_time_assignment()
        self.update()
        self.tau, self.tau_ = self.get_divergence(mode="tau")
        self.likelihood = self.get_likelihood(refit_time=False)
        self.varx = self.get_variance()

    def fit_alpha(self, sight=0.5, **kwargs):
        def mse(x):
            return self.get_mse(alpha=x[0], **kwargs)

        val = self.alpha
        vals = val + np.linspace(-1, 1, num=4) * val * sight
        for v in vals:
            self.update(alpha=val * v)
        res = minimize(mse, np.array([val]), **self.simplex_kwargs)
        self.update(alpha=res.x[0])

    def fit_beta(self, sight=0.5, **kwargs):
        def mse(x):
            return self.get_mse(beta=x[0], **kwargs)  # scaling=x[1])

        val = self.beta
        vals = val + np.linspace(-1, 1, num=4) * val * sight
        for v in vals:
            self.update(beta=val * v)
        res = minimize(mse, np.array([val]), **self.simplex_kwargs)
        self.update(beta=res.x[0])

    def fit_gamma(self, sight=0.5, **kwargs):
        def mse(x):
            return self.get_mse(gamma=x[0], **kwargs)

        val = self.gamma
        vals = val + np.linspace(-1, 1, num=4) * val * sight
        for v in vals:
            self.update(gamma=val * v)
        res = minimize(mse, np.array([val]), **self.simplex_kwargs)
        self.update(gamma=res.x[0])

    def fit_t_and_alpha(self, **kwargs):
        def mse(x):
            return self.get_mse(t_=x[0], alpha=x[1], **kwargs)

        alpha_vals = self.alpha + np.linspace(-1, 1, num=5) * self.alpha / 10
        for alpha in alpha_vals:
            self.update(alpha=alpha)
        x0, cb = np.array([self.t_, self.alpha]), self.cb_fit_t_and_alpha
        res = minimize(mse, x0, callback=cb, **self.simplex_kwargs)  # using Nelder-Mead
        self.update(t_=res.x[0], alpha=res.x[1])

    def fit_rates(self, **kwargs):
        def mse(x):
            return self.get_mse(alpha=x[0], gamma=x[1], **kwargs)

        x0, cb = np.array([self.alpha, self.gamma]), self.cb_fit_rates
        res = minimize(mse, x0, tol=1e-2, callback=cb, **self.simplex_kwargs)
        self.update(alpha=res.x[0], gamma=res.x[1])

    def fit_t_(self, **kwargs):
        def mse(x):
            return self.get_mse(t_=x[0], **kwargs)

        res = minimize(mse, self.t_, callback=self.cb_fit_t_, **self.simplex_kwargs)
        self.update(t_=res.x[0])

    def fit_rates_all(self, **kwargs):
        def mse(x):
            return self.get_mse(alpha=x[0], beta=x[1], gamma=x[2], **kwargs)

        x0, cb = np.array([self.alpha, self.beta, self.gamma]), self.cb_fit_rates_all
        res = minimize(mse, x0, tol=1e-2, callback=cb, **self.simplex_kwargs)
        self.update(alpha=res.x[0], beta=res.x[1], gamma=res.x[2])

    def fit_t_and_rates(self, **kwargs):
        def mse(x):
            return self.get_mse(t_=x[0], alpha=x[1], beta=x[2], gamma=x[3], **kwargs)

        x0 = np.array([self.t_, self.alpha, self.beta, self.gamma])
        cb = self.cb_fit_t_and_rates
        res = minimize(mse, x0, tol=1e-2, callback=cb, **self.simplex_kwargs)
        self.update(t_=res.x[0], alpha=res.x[1], beta=res.x[2], gamma=res.x[3])

    def fit_scaling_(self, **kwargs):
        def mse(x):
            return self.get_mse(t_=x[0], beta=x[1], scaling=x[2], **kwargs)

        x0, cb = np.array([self.t_, self.beta, self.scaling]), self.cb_fit_scaling_
        res = minimize(mse, x0, callback=cb, **self.simplex_kwargs)
        self.update(t_=res.x[0], beta=res.x[1], scaling=res.x[2])

    # Callback functions for the Optimizer
    def cb_fit_t_and_alpha(self, x):
        self.update(t_=x[0], alpha=x[1])

    def cb_fit_scaling_(self, x):
        self.update(t_=x[0], beta=x[1], scaling=x[2])

    def cb_fit_rates(self, x):
        self.update(alpha=x[0], gamma=x[1])

    def cb_fit_t_(self, x):
        self.update(t_=x[0])

    def cb_fit_t_and_rates(self, x):
        self.update(t_=x[0], alpha=x[1], beta=x[2], gamma=x[3])

    def cb_fit_rates_all(self, x):
        self.update(alpha=x[0], beta=x[1], gamma=x[2])

    def set_callbacks(self):
        # Overwrite callbacks
        if not self.high_pars_resolution:
            self.cb_fit_t_and_alpha = None
            self.cb_fit_scaling_ = None
            self.cb_fit_rates = None
            self.cb_fit_t_ = None
            self.cb_fit_t_and_rates = None
            self.cb_fit_rates_all = None

    def update(
        self,
        t=None,
        t_=None,
        alpha=None,
        beta=None,
        gamma=None,
        scaling=None,
        u0_=None,
        s0_=None,
        adjust_t_=True,
    ):
        loss_prev = self.loss[-1] if len(self.loss) > 0 else 1e6

        _vars = self.get_vars(alpha, beta, gamma, scaling, t_, u0_, s0_)
        _time = self.get_time_assignment(alpha, beta, gamma, scaling, t_, u0_, s0_, t)
        alpha, beta, gamma, scaling, t_ = _vars
        t, tau, o = _time
        loss = self.get_loss(t, t_, alpha, beta, gamma, scaling)
        perform_update = loss < loss_prev

        on = self.o == 1
        if adjust_t_ and np.any(on):
            if not perform_update:
                alpha, beta, gamma, scaling, t_ = self.get_vars()
                t, tau, o = self.get_time_assignment()
                loss = self.get_loss()

            alt_t_ = t[on].max()
            if 0 < alt_t_ < t_:
                # alt_u0_, alt_s0_ = mRNA(alt_t_, 0, 0, alpha, beta, gamma)
                alt_t_ += np.max(t) / len(t) * np.sum(t == t_)
                # np.sum((self.u / self.scaling >= alt_u0_) | (self.s >= alt_s0_))
                _time = self.get_time_assignment(alpha, beta, gamma, scaling, alt_t_)
                alt_t, alt_tau, alt_o = _time
                alt_loss = self.get_loss(alt_t, alt_t_, alpha, beta, gamma, scaling)
                ut_cur = unspliced(t_, 0, alpha, beta)
                ut_alt = unspliced(alt_t_, 0, alpha, beta)

                min_loss = np.min([loss, loss_prev])
                if alt_loss * 0.99 <= min_loss or ut_cur * 0.99 < ut_alt:
                    t, tau, o, t_, loss = alt_t, alt_tau, alt_o, alt_t_, alt_loss
                    perform_update = True

            if False:
                steady_states = t == t_
                if perform_update and np.any(steady_states):
                    t_ += t.max() / len(t) * np.sum(steady_states)
                    _time = self.get_time_assignment(alpha, beta, gamma, scaling, t_)
                    t, tau, o = _time
                    loss = self.get_loss(t, t_, alpha, beta, gamma, scaling)

        if perform_update:
            if scaling is not None:
                self.steady_u *= self.scaling / scaling
                self.u0_ *= self.scaling / scaling
            if u0_ is not None:
                self.u0_ = u0_
            if s0_ is not None:
                self.s0_ = s0_

            self.t, self.tau, self.o = t, tau, o
            self.alpha, self.beta, self.gamma = alpha, beta, gamma
            self.scaling, self.t_ = scaling, t_
            new_pars = np.array([alpha, beta, gamma, t_, scaling])[:, None]
            self.pars = np.c_[self.pars, new_pars]
            self.loss.append(loss)

        return perform_update


default_pars_names = ["alpha", "beta", "gamma", "t_", "scaling", "std_u", "std_s"]
default_pars_names += ["likelihood", "u0", "s0", "pval_steady"]
default_pars_names += ["steady_u", "steady_s", "variance"]


def read_pars(adata, pars_names=None, key="fit"):
    pars = []
    for name in default_pars_names if pars_names is None else pars_names:
        pkey = f"{key}_{name}"
        par = np.zeros(adata.n_vars) * np.nan
        if pkey in adata.var.keys():
            par = adata.var[pkey].values
        pars.append(par)
    return pars


def write_pars(adata, pars, pars_names=None, add_key="fit"):
    for i, name in enumerate(default_pars_names if pars_names is None else pars_names):
        adata.var[f"{add_key}_{name}"] = pars[i]


def recover_dynamics(
    data,
    var_names="velocity_genes",
    n_top_genes=None,
    max_iter=10,
    assignment_mode="projection",
    t_max=None,
    fit_time=True,
    fit_scaling=True,
    fit_steady_states=True,
    fit_connected_states=None,
    fit_basal_transcription=None,
    use_raw=False,
    load_pars=None,
    return_model=None,
    plot_results=False,
    steady_state_prior=None,
    add_key="fit",
    copy=False,
    **kwargs,
):
    """Recovers the full splicing kinetics of specified genes.

    The model infers transcription rates, splicing rates, degradation rates,
    as well as cell-specific latent time and transcriptional states,
    estimated iteratively by expectation-maximization.

    .. image:: https://user-images.githubusercontent.com/31883718/69636459-ef862800-1056-11ea-8803-0a787ede5ce9.png

    Arguments
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
    var_names: `str`,  list of `str` (default: `'velocity_genes`)
        Names of variables/genes to use for the fitting.
    n_top_genes: `int` or `None` (default: `None`)
        Number of top velocity genes to use for the dynamical model.
    max_iter:`int` (default: `10`)
        Maximal iterations in the EM-Algorithm.
    assignment_mode: `str` (default: `projection`)
        Determined how times are assigned to observations.
        If `projection`, observations are projected onto the model trajectory.
        Else uses an inverse approximating formula.
    t_max: `float`, `False` or `None` (default: `None`)
        Total range for time assignments.
    fit_scaling: `bool` or `float` or `None` (default: `True`)
        Whether to fit scaling between unspliced and spliced.
    fit_time: `bool` or `float` or `None` (default: `True`)
        Whether to fit time or keep initially given time fixed.
    fit_steady_states: `bool` or `None` (default: `True`)
        Whether to explicitly model and fit steady states (next to induction/repression)
    fit_connected_states: `bool` or `None` (default: `None`)
        Restricts fitting to neighbors given by connectivities.
    fit_basal_transcription: `bool` or `None` (default: `None`)
        Enables model to incorporate basal transcriptions.
    use_raw: `bool` or `None` (default: `None`)
        if True, use .layers['sliced'], else use moments from .layers['Ms']
    load_pars: `bool` or `None` (default: `None`)
        Load parameters from past fits.
    return_model: `bool` or `None` (default: `True`)
        Whether to return the model as :DynamicsRecovery: object.
    plot_results: `bool` or `None` (default: `False`)
        Plot results after parameter inference.
    steady_state_prior: list of `bool` or `None` (default: `None`)
        Mask for indices used for steady state regression.
    add_key: `str` (default: `'fit'`)
        Key to add to parameter names, e.g. 'fit_t' for fitted time.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to `adata`.

    Returns
    -------
    Returns or updates `adata`
    """
    adata = data.copy() if copy else data
    logg.info("recovering dynamics", r=True)

    if len(set(adata.var_names)) != len(adata.var_names):
        logg.warn("Duplicate var_names found. Making them unique.")
        adata.var_names_make_unique()

    if "Ms" not in adata.layers.keys() or "Mu" not in adata.layers.keys():
        use_raw = True
    if fit_connected_states is None:
        fit_connected_states = not use_raw

    adata.uns["recover_dynamics"] = {
        "fit_connected_states": fit_connected_states,
        "fit_basal_transcription": fit_basal_transcription,
        "use_raw": use_raw,
    }

    if isinstance(var_names, str) and var_names not in adata.var_names:
        if var_names in adata.var.keys():
            var_names = adata.var_names[adata.var[var_names].values]
        elif use_raw or var_names == "all":
            var_names = adata.var_names
        elif "_genes" in var_names:
            from scvelo.tools.velocity import Velocity

            velo = Velocity(adata, use_raw=use_raw)
            velo.compute_deterministic(perc=[5, 95])
            var_names = adata.var_names[velo._velocity_genes]
            adata.var["fit_r2"] = velo._r2
        else:
            raise ValueError("Variable name not found in var keys.")
    if not isinstance(var_names, str):
        var_names = list(np.ravel(var_names))

    var_names = make_unique_list(var_names, allow_array=True)
    var_names = np.array([name for name in var_names if name in adata.var_names])
    if len(var_names) == 0:
        raise ValueError("Variable name not found in var keys.")
    if n_top_genes is not None and len(var_names) > n_top_genes:
        X = adata[:, var_names].layers[("spliced" if use_raw else "Ms")]
        var_names = var_names[np.argsort(np.sum(X, 0))[::-1][:n_top_genes]]
    if return_model is None:
        return_model = len(var_names) < 5

    pars = read_pars(adata)
    alpha, beta, gamma, t_, scaling, std_u, std_s, likelihood = pars[:8]
    u0, s0, pval, steady_u, steady_s, varx = pars[8:]
    # likelihood[np.isnan(likelihood)] = 0
    idx, L, P = [], [], []
    T = np.zeros(adata.shape) * np.nan
    Tau = np.zeros(adata.shape) * np.nan
    Tau_ = np.zeros(adata.shape) * np.nan
    if "fit_t" in adata.layers.keys():
        T = adata.layers["fit_t"]
    if "fit_tau" in adata.layers.keys():
        Tau = adata.layers["fit_tau"]
    if "fit_tau_" in adata.layers.keys():
        Tau_ = adata.layers["fit_tau_"]

    conn = get_connectivities(adata) if fit_connected_states else None
    progress = logg.ProgressReporter(len(var_names))
    for i, gene in enumerate(var_names):
        dm = DynamicsRecovery(
            adata,
            gene,
            use_raw=use_raw,
            load_pars=load_pars,
            max_iter=max_iter,
            fit_time=fit_time,
            fit_steady_states=fit_steady_states,
            fit_connected_states=conn,
            fit_scaling=fit_scaling,
            fit_basal_transcription=fit_basal_transcription,
            steady_state_prior=steady_state_prior,
            **kwargs,
        )
        if dm.recoverable:
            dm.fit(assignment_mode=assignment_mode)

            ix = np.where(adata.var_names == gene)[0][0]
            idx.append(ix)

            T[:, ix], Tau[:, ix], Tau_[:, ix] = dm.t, dm.tau, dm.tau_
            alpha[ix], beta[ix], gamma[ix], t_[ix], scaling[ix] = dm.pars[:, -1]
            u0[ix], s0[ix], pval[ix] = dm.u0, dm.s0, dm.pval_steady
            steady_u[ix], steady_s[ix] = dm.steady_u, dm.steady_s
            beta[ix] /= scaling[ix]
            steady_u[ix] *= scaling[ix]

            std_u[ix], std_s[ix] = dm.std_u, dm.std_s
            likelihood[ix], varx[ix] = dm.likelihood, dm.varx
            L.append(dm.loss)
            if plot_results and i < 4:
                P.append(np.array(dm.pars))

            progress.update()
        else:
            logg.warn(dm.gene, "not recoverable due to insufficient samples.")
    progress.finish()

    _pars = [
        alpha,
        beta,
        gamma,
        t_,
        scaling,
        std_u,
        std_s,
        likelihood,
        u0,
        s0,
        pval,
        steady_u,
        steady_s,
        varx,
    ]
    write_pars(adata, _pars)
    if "fit_t" in adata.layers.keys():
        adata.layers["fit_t"][:, idx] = (
            T[:, idx] if conn is None else conn.dot(T[:, idx])
        )
    else:
        adata.layers["fit_t"] = T if conn is None else conn.dot(T)
    adata.layers["fit_tau"] = Tau
    adata.layers["fit_tau_"] = Tau_

    if L:  # is False if only one invalid / irrecoverable gene was given in var_names
        cur_len = adata.varm["loss"].shape[1] if "loss" in adata.varm.keys() else 2
        max_len = max(np.max([len(l) for l in L]), cur_len) if L else cur_len
        loss = np.ones((adata.n_vars, max_len)) * np.nan

        if "loss" in adata.varm.keys():
            loss[:, :cur_len] = adata.varm["loss"]

        loss[idx] = np.vstack(
            [np.concatenate([l, np.ones(max_len - len(l)) * np.nan]) for l in L]
        )
        adata.varm["loss"] = loss

    if t_max is not False:
        dm = align_dynamics(adata, t_max=t_max, dm=dm, idx=idx)

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        f"    '{add_key}_pars', "
        f"fitted parameters for splicing dynamics (adata.var)"
    )

    if plot_results:  # Plot Parameter Stats
        n_rows, n_cols = len(var_names[:4]), 6
        figsize = [2 * n_cols, 1.5 * n_rows]  # rcParams['figure.figsize']
        fontsize = rcParams["font.size"]
        fig, axes = pl.subplots(nrows=n_rows, ncols=6, figsize=figsize)
        pl.subplots_adjust(wspace=0.7, hspace=0.5)
        for i, gene in enumerate(var_names[:4]):
            if t_max is not False:
                mi = dm.m[i]
                P[i] *= np.array([1 / mi, 1 / mi, 1 / mi, mi, 1])[:, None]
            ax = axes[i] if n_rows > 1 else axes
            for j, pij in enumerate(P[i]):
                ax[j].plot(pij)
            ax[len(P[i])].plot(L[i])
            if i == 0:
                pars_names = ["alpha", "beta", "gamma", "t_", "scaling", "loss"]
                for j, name in enumerate(pars_names):
                    ax[j].set_title(name, fontsize=fontsize)

    if return_model:
        logg.info("\noutputs model fit of gene:", dm.gene)

    return dm if return_model else adata if copy else None


def align_dynamics(
    data, t_max=None, dm=None, idx=None, mode=None, remove_outliers=None, copy=False
):
    """Align dynamics to a common set of parameters

    Arguments
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
    t_max: `float`, `False` or `None` (default: `None`)
        Total range for time assignments.
    dm: :class:`~DynamicsRecovery`
        DynamicsRecovery object to perform alignment on.
    idx: list of `bool` or `None` (default: `None`)
        Mask for indices used for alignment.
    mode: `str` or None (default: `'align_total_time`)
        What to align. Takes the following arguments:
        common_splicing_rate, common_scaling, align_increments, align_total_time
    remove_outliers: `bool` or `None` (default: `None`)
        Whether to remove outliers.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to `adata`.
     Returns
    -------
    Returns or updates `adata` with the attributes
    alpha, beta, gamma, t_, alignment_scaling: `.var`
        aligned parameters
    fit_t, fit_tau, fit_tau_: `.layer`
        aligned time
    """

    adata = data.copy() if copy else data
    pars_names = ["alpha", "beta", "gamma", "t_", "scaling", "alignment_scaling"]
    alpha, beta, gamma, t_, scaling, mz = read_pars(adata, pars_names=pars_names)
    T = np.zeros(adata.shape) * np.nan
    Tau = np.zeros(adata.shape) * np.nan
    Tau_ = np.zeros(adata.shape) * np.nan
    if "fit_t" in adata.layers.keys():
        T = adata.layers["fit_t"]
    if "fit_tau" in adata.layers.keys():
        Tau = adata.layers["fit_tau"]
    if "fit_tau_" in adata.layers.keys():
        Tau_ = adata.layers["fit_tau_"]
    idx = ~np.isnan(np.sum(T, axis=0)) if idx is None else idx
    if "fit_alignment_scaling" not in adata.var.keys():
        mz = np.ones(adata.n_vars)
    if mode is None:
        mode = "align_total_time"

    m = np.ones(adata.n_vars)
    mz_prev = np.array(mz)

    if dm is not None:  # newly fitted
        mz[idx] = 1

    if mode == "align_total_time" and t_max is not False:
        T_max = np.max(T[:, idx] * (T[:, idx] < t_[idx]), axis=0)  # transient 'on'
        T_max += np.max((T[:, idx] - t_[idx]) * (T[:, idx] > t_[idx]), axis=0)  # 'off'

        denom = 1 - np.sum((T[:, idx] == t_[idx]) | (T[:, idx] == 0), axis=0) / len(T)
        denom += denom == 0

        T_max = T_max / denom
        T_max += T_max == 0

        t_max = 20 if t_max is None else t_max
        m[idx] = t_max / T_max
        mz *= m

    else:
        m = 1 / mz
        mz = np.ones(adata.n_vars)

    if remove_outliers:
        mu, std = np.nanmean(mz), np.nanstd(mz)
        mz = np.clip(mz, mu - 3 * std, mu + 3 * std)
        m = mz / mz_prev

    if idx is None:
        alpha, beta, gamma = alpha / m, beta / m, gamma / m
        T, t_, Tau, Tau_ = T * m, t_ * m, Tau * m, Tau_ * m
    else:
        m_ = m[idx]
        alpha[idx] = alpha[idx] / m_
        beta[idx] = beta[idx] / m_
        gamma[idx] = gamma[idx] / m_
        T[:, idx], t_[idx] = T[:, idx] * m_, t_[idx] * m_
        Tau[:, idx], Tau_[:, idx] = Tau[:, idx] * m_, Tau_[:, idx] * m_

    mz[mz == 1] = np.nan
    pars_names = ["alpha", "beta", "gamma", "t_", "alignment_scaling"]
    write_pars(adata, [alpha, beta, gamma, t_, mz], pars_names=pars_names)
    adata.layers["fit_t"] = T
    adata.layers["fit_tau"] = Tau
    adata.layers["fit_tau_"] = Tau_

    if dm is not None and dm.recoverable:
        dm.m = m[idx]
        dm.alpha, dm.beta, dm.gamma, dm.pars[:3] = (
            np.array([dm.alpha, dm.beta, dm.gamma, dm.pars[:3]]) / dm.m[-1]
        )
        dm.t, dm.tau, dm.t_, dm.pars[4] = (
            np.array([dm.t, dm.tau, dm.t_, dm.pars[4]]) * dm.m[-1]
        )

    return adata if copy else dm


def latent_time(
    data,
    vkey="velocity",
    min_likelihood=0.1,
    min_confidence=0.75,
    min_corr_diffusion=None,
    weight_diffusion=None,
    root_key=None,
    end_key=None,
    t_max=None,
    use_fit_t=False,
    method="dpt",
    copy=False,
):
    """Computes a gene-shared latent time.

    Gene-specific latent timepoints obtained from the dynamical model are coupled to a
    universal gene-shared latent time, which represents the cell’s internal clock and
    is based only on its transcriptional dynamics.

    .. image:: https://user-images.githubusercontent.com/31883718/69636500-03318e80-1057-11ea-9e14-ae9f907711cc.png

    Arguments
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix
    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.
    min_likelihood: `float` between `0` and `1` or `None` (default: `.1`)
        Minimal likelihood fitness for genes to be included to the weighting.
    min_confidence: `float` between `0` and `1` (default: `.75`)
        Parameter for local coherence selection.
    min_corr_diffusion: `float` between `0` and `1` or `None` (default: `None`)
        Only select genes that correlate with velocity pseudotime obtained
        from diffusion random walk on velocity graph.
    weight_diffusion: `float` or `None` (default: `None`)
        Weight applied to couple latent time with diffusion-based velocity pseudotime.
    root_key: `str` or `None` (default: `'root_cells'`)
        Key (.uns, .obs) of root cell to be used.
        If not set, it obtains root cells from velocity-inferred transition matrix.
    end_key: `str` or `None` (default: `None`)
        Key (.obs) of end points to be used.
    use_fit_t: `bool` (default: `false`)
        Use fit_t in the dynamical model to calculate the latent time.
    method: `str` or `None` (default: `'dpt1'`)
        The method to compute the relative time.
    t_max: `float` or `None` (default: `None`)
        Overall duration of differentiation process.
        If not set, a overall transcriptional timescale of 20 hours is used as prior.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to `adata`.
     Returns
    -------
    Returns or updates `adata` with the attributes
    latent_time: `.obs`
        latent time from learned dynamics for each cell
    """
    adata = data.copy() if copy else data

    from scvelo.tools.utils import vcorrcoef, scale
    from scvelo.tools.dynamical_model_utils import root_time, compute_shared_time
    from scvelo.tools.terminal_states import terminal_states
    from scvelo.tools.velocity_graph import velocity_graph
    from scvelo.tools.velocity_pseudotime import velocity_pseudotime

    if use_fit_t and "fit_t" not in adata.layers.keys():
        raise ValueError("you need to run `tl.recover_dynamics` first.")

    # NOTE(Haotian): first need the velocity graph
    if f"{vkey}_graph" not in adata.uns.keys():
        velocity_graph(adata, approx=True)

    if root_key is None:
        terminal_keys = ["root_cells", "starting_cells", "root_states_probs"]
        keys = [key for key in terminal_keys if key in adata.obs.keys()]
        if len(keys) > 0:
            root_key = keys[0]
    if root_key not in adata.uns.keys() and root_key not in adata.obs.keys():
        root_key = "root_cells"
    # NOTE(Haotian): computing the root cells
    if root_key not in adata.obs.keys():
        terminal_states(adata, vkey=vkey)

    if use_fit_t:
        t = np.array(adata.layers["fit_t"])
        idx_valid = ~np.isnan(
            t.sum(0)
        )  # this is the valid idx on the gene axis, each has to have valid time t across all cells
        if min_likelihood is not None:  # filter out more unreliable ts
            likelihood = adata.var["fit_likelihood"].values
            idx_valid &= np.array(likelihood >= min_likelihood, dtype=bool)
        t = t[:, idx_valid]
        t_sum = np.sum(t, 1)
    conn = get_connectivities(adata)

    if root_key not in adata.uns.keys():
        roots = np.arange(adata.shape[0])  # instead of np.argsort(t_sum)
        idx_roots = np.array(adata.obs[root_key][roots])
        idx_roots[pd.isnull(idx_roots)] = 0
        if np.any([isinstance(ix, str) for ix in idx_roots]):
            idx_roots = np.array([isinstance(ix, str) for ix in idx_roots], dtype=int)
        idx_roots = idx_roots.astype(np.float) > 1 - 1e-3
        if np.sum(idx_roots) > 0:
            roots = roots[
                idx_roots
            ]  # NOTE(Haotian): 1. Here is the actual root_cell index. 2. I don't think the argsort(8 lines before) is needed, you just need the index.
        else:
            logg.warn(
                "No root cells detected. Consider specifying "
                "root cells to improve latent time prediction."
            )
    else:
        roots = [adata.uns[root_key]]
        root_key = f"root cell {adata.uns[root_key]}"

    if end_key in adata.obs.keys():
        fates = np.argsort(t_sum)[::-1]
        idx_fates = np.array(adata.obs[end_key][fates])
        idx_fates[pd.isnull(idx_fates)] = 0
        if np.any([isinstance(ix, str) for ix in idx_fates]):
            idx_fates = np.array([isinstance(ix, str) for ix in idx_fates], dtype=int)
        idx_fates = idx_fates.astype(np.float) > 1 - 1e-3
        if np.sum(idx_fates) > 0:
            fates = fates[idx_fates]
    else:
        fates = [None]

    logg.info(
        f"computing latent time using {root_key}"
        f"{', ' + end_key if end_key in adata.obs.keys() else ''} as prior",
        r=True,
    )

    # NOTE(Haotian): actual pseudotime call
    VPT = velocity_pseudotime(
        adata, vkey, root_key=roots[0], end_key=fates[0], return_model=True
    )
    vpt = VPT.pseudotime

    if use_fit_t:
        if min_corr_diffusion is not None:
            corr = vcorrcoef(t.T, vpt)
            t = t[:, np.array(corr >= min_corr_diffusion, dtype=bool)]

        if root_key in adata.uns.keys():
            root = adata.uns[root_key]
            t, t_ = root_time(t, root=root)
            latent_time = compute_shared_time(t)
        else:
            roots = roots[:4]  # NOTE(Haotian): you only use the top 4 cells?
            latent_time = np.ones(shape=(len(roots), adata.n_obs))
            for i, root in enumerate(roots):
                t, t_ = root_time(t, root=root)
                latent_time[i] = compute_shared_time(
                    t
                )  # NOTE(Haotian): for each one cell, this is super ad-hoc, just select those ts look better for 2 select rows.
            latent_time = scale(np.mean(latent_time, axis=0))

        if fates[0] is not None:
            fates = fates[:4]
            latent_time_ = np.ones(shape=(len(fates), adata.n_obs))
            for i, fate in enumerate(fates):
                t, t_ = root_time(t, root=fate)
                latent_time_[i] = 1 - compute_shared_time(t)
            latent_time = scale(
                latent_time + 0.2 * scale(np.mean(latent_time_, axis=0))
            )
    elif method == "eig":
        # using the eigenvector method
        T = data.uns[f"{vkey}_graph"] - data.uns[f"{vkey}_graph_neg"]
        # T += T.T
        eigvecs = eigs(T, eps=1e-3, random_state=0)[1]
        pt = T.dot(eigvecs).sum(1)
        # pt = scale(np.clip(pt, 0, np.percentile(pt, 98)))
        latent_time = scale(pt)
    else:
        latent_time = scale(vpt)

    tl = latent_time
    tc = conn.dot(latent_time)

    z = tl.dot(tc) / tc.dot(
        tc
    )  # NOTE(Haotian): what is the purpose of these steps? what is going on here? if z is closed to 1. basically, it is the eigenvector.
    tl_conf = (1 - np.abs(tl / np.max(tl) - tc * z / np.max(tl))) ** 2
    idx_low_confidence = tl_conf < min_confidence

    if weight_diffusion is not None:
        w = weight_diffusion
        latent_time = (1 - w) * latent_time + w * vpt
        latent_time[idx_low_confidence] = vpt[idx_low_confidence]
    else:
        conn_new = conn.copy()
        conn_new[:, idx_low_confidence] = 0
        conn_new.eliminate_zeros()
        latent_time = conn_new.dot(latent_time)

    latent_time = scale(latent_time)
    if t_max is not None:
        latent_time *= t_max

    adata.obs["latent_time"] = latent_time

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint("added \n" "    'latent_time', shared time (adata.obs)")
    return adata if copy else None


recover_latent_time = latent_time
