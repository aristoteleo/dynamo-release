from tqdm import tqdm
import inspect
import numpy as np
import pandas as pd
from scipy.sparse import issparse, SparseEfficiencyWarning

from .moments import moments, strat_mom
from ..estimation.csc.velocity import fit_linreg, velocity, ss_estimation
from ..estimation.tsc.estimation_kinetic import *
from ..estimation.tsc.utils_kinetic import *
from .utils import (
    update_dict,
    get_valid_bools,
    get_data_for_kin_params_estimation,
    get_U_S_for_velocity_estimation,
)
from .utils import set_velocity, set_param_ss, set_param_kinetic
from .moments import (
    prepare_data_no_splicing,
    prepare_data_has_splicing,
    prepare_data_deterministic,
    prepare_data_mix_has_splicing,
    prepare_data_mix_no_splicing,
)

import warnings
warnings.simplefilter('ignore', SparseEfficiencyWarning)

# incorporate the model selection code soon
def dynamics(
    adata,
    tkey=None,
    t_label_keys=None,
    filter_gene_mode="final",
    use_smoothed=True,
    experiment_type='auto',
    assumption_mRNA='auto',
    assumption_protein="ss",
    model="auto",
    est_method="auto",
    NTR_vel=False,
    group=None,
    protein_names=None,
    concat_data=False,
    log_unnormalized=True,
    one_shot_method="combined",
    re_smooth=False,
    sanity_check=False,
    cores=1,
    **est_kwargs
):
    """Inclusive model of expression dynamics considers splicing, metabolic labeling and protein translation. It supports
    learning high-dimensional velocity vector samples for droplet based (10x, inDrop, drop-seq, etc), scSLAM-seq, NASC-seq
    sci-fate, scNT-seq or cite-seq datasets.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object.
        tkey: `str` or None (default: None)
            The column key for the time label of cells in .obs. Used for either "ss" or "kinetic" model.
            mode  with labeled data. When `group` is None, `tkey` will also be used for calculating  1st/2st moment or covariance.
        t_label_keys: `str`, `list` or None (default: None)
            The column key(s) for the labeling time label of cells in .obs. Used for either "ss" or "kinetic" model.
            Not used for now and `tkey` is implicitly assumed as `t_label_key` (however, `tkey` should just be the time
            of the experiment).
        filter_gene_mode: `str` (default: `final`)
            The string for indicating which mode (one of, {'final', 'basic', 'no'}) of gene filter will be used.
        use_smoothed: `bool` (default: 'True')
            Whether to use the smoothed data when estimating kinetic parameters and calculating velocity for each gene.
            When you have time-series data (`tkey` is not None), we recommend to smooth data among cells from each time point.
        experiment_type: `str` {`conventional`, `deg`, `kin`, `one-shot`, `auto`}, (default: `auto`)
            single cell RNA-seq experiment type. Available options are:
            (1) 'conventional': conventional single-cell RNA-seq experiment;
            (2) 'deg': chase/degradation experiment;
            (3) 'kin': pulse/synthesis/kinetics experiment;
            (4) 'one-shot': one-shot kinetic experiment;
            (5) 'auto': dynamo will detect the experimental type automatically.
        assumption_mRNA: `str` `str` {`ss`, `kinetic`, `auto`}, (default: `auto`)
            Parameter estimation assumption for mRNA. Available options are:
            (1) 'ss': pseudo steady state;
            (2) 'kinetic' or None: degradation and kinetic data without steady state assumption.
            If no labelling data exists, assumption_mRNA will automatically set to be 'ss'. For one-shot experiment, assumption_mRNA
            is set to be None. However we will use steady state assumption to estimate parameters alpha and gamma either by a deterministic
            linear regression or the first order decay approach in line of the sci-fate paper;
            (3) 'auto': dynamo will choose a reasonable assumption of the system under study automatically.
        assumption_protein: `str`, (default: `ss`)
            Parameter estimation assumption for protein. Available options are:
            (1) 'ss': pseudo steady state;
        model: `str` {`auto`, `deterministic`, `stochastic`} (default: `auto`)
            String indicates which estimation model will be used.
            (1) 'deterministic': The method based on `deterministic` ordinary differential equations;
            (2) 'stochastic' or `moment`: The new method from us that is based on `stochastic` master equations;
            Note that `kinetic` model doesn't need to assumes the `experiment_type` is not `conventional`. As other labeling
            experiments, if you specify the `tkey`, dynamo can also apply `kinetic` model on `conventional` scRNA-seq datasets.
            A "model_selection" model will be supported soon in which alpha, beta and gamma will be modeled as a function of time.
        est_method: `str` {`ols`, `rlm`, `ransac`, `gmm`, `negbin`, `auto`} This parameter should be used in conjunction with `model` parameter.
            * Available options when the `model` is 'ss' include:
            (1) 'ols': The canonical method or Ordinary Least Squares regression from the seminar RNA velocity paper
            based on deterministic ordinary differential equations;
            (2) 'rlm': The robust linear models from statsmodels. Robust Regression provides an alternative to OLS
            regression by lowering the restrictions on assumptions and dampens the effect of outliers in order to fit
            majority of the data.
            (3) 'ransac': RANSAC (RANdom SAmple Consensus) algorithm for robust linear regression. RANSAC is an iterative
            algorithm for the robust estimation of parameters from a subset of inliers from the complete data set. RANSAC
            implementation is based on RANSACRegressor function from sklearn package. Note that if `rlm` or `ransac`
            failed, it will roll back to the `ols` method. In addition, `ols`, `rlm` and `ransac` can be only used in
            conjunction with the `deterministic` model.
            (4) 'gmm': The new generalized methods of moments from us that is based on master equations, similar to the
            "moment" model in the excellent scVelo package;
            (5) 'negbin': The new method from us that models steady state RNA expression as a negative binomial distribution,
            also built upon on master equations.
            Note that all those methods require using extreme data points (except negbin, which use all data points) for
            estimation. Extreme data points are defined as the data from cells whose expression of unspliced / spliced
            or new / total RNA, etc. are in the top or bottom, 5%, for example. `linear_regression` only considers the mean of
            RNA species (based on the `deterministic` ordinary different equations) while moment based methods (`gmm`, `negbin`)
            considers both first moment (mean) and second moment (uncentered variance) of RNA species (based on the `stochastic`
            master equations).
            (4) 'auto': dynamo will choose the suitable estimation method based on the `assumption_mRNA`, `experiment_type`
            and `model` parameter.
            The above method are all (generalized) linear regression based method. In order to return estimated parameters
            (including RNA half-life), it additionally returns R-squared (either just for extreme data points or all data points)
            as well as the log-likelihood of the fitting, which will be used for transition matrix and velocity embedding.
            * Available options when the `assumption_mRNA` is 'kinetic' include:
            (1) 'auto': dynamo will choose the suitable estimation method based on the `assumption_mRNA`, `experiment_type`
            and `model` parameter.
            Under `kinetic` model, choosing estimation is `experiment_type` dependent. For `kinetics` experiments, dynamo
            supposes methods including RNA bursting or without RNA bursting. Dynamo also adaptively estimates parameters, based
            on whether the data has splicing or without splicing.
            Under `kinetic` assumption, the above method uses non-linear least square fitting. In order to return estimated parameters
            (including RNA half-life), it additionally returns the log-likelihood of the fittingwhich, which will be used for transition
            matrix and velocity embedding.
            All `est_method` uses least square to estimate optimal parameters with latin cubic sampler for initial sampling.
        NTR_vel: `bool` (default: `True`)
            Whether to use NTR (new/total ratio) velocity for labeling datasets.
        group: `str` or None (default: `None`)
            The column key/name that identifies the grouping information (for example, clusters that correspond to different cell types)
            of cells. This will be used to calculate 1/2 st moments and covariance for each cells in each group. It will also enable
            estimating group-specific (i.e cell-type specific) kinetic parameters.
        protein_names: `List`
            A list of gene names corresponds to the rows of the measured proteins in the `X_protein` of the `obsm` attribute.
            The names have to be included in the adata.var.index.
        concat_data: `bool` (default: `False`)
            Whether to concatenate data before estimation. If your data is a list of matrices for each time point, this need to be set as True.
        log_unnormalized: `bool` (default: `True`)
            Whether to log transform the unnormalized data.
        re_smooth: `bool` (default: `False`)
            Whether to re-smooth the adata and also recalculate 1/2 moments or covariance.
        sanity_check: `bool` (default: `False`)
            Whether to perform sanity-check before estimating kinetic parameters and velocity vectors, currently only
            applicable to kinetic or degradation metabolic labeling based scRNA-seq data. The basic idea is that for
            kinetic (degradation) experiment, the total labelled RNA for each gene should increase (decrease) over time.
            If they don't satisfy this criteria, those genes will be ignored during the estimation.
        cores: `int` (default: 1):
            Number of cores to run the estimation. If cores is set to be > 1, multiprocessing will be used to parallel
            the parameter estimation. Currently only applicable cases when assumption_mRNA is `ss` or cases when 
            experiment_type is either "one-shot" or "mix_std_stm".
        **est_kwargs
            Other arguments passed to the fit method (steady state models) or estimation methods (kinetic models).

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            A updated AnnData object with estimated kinetic parameters, inferred velocity and estimation related information
            included. The estimated kinetic parameters are currently appended to .obs (should move to .obsm with the key
            `dynamics` later). Depends on the estimation method, experiment type and whether you applied estimation for
            each groups via `group`, the number of returned parameters can be variable. For conventional scRNA-seq (including
            cite-seq or other types of protein/RNA coassays) and somethings metabolic labeling data, the parameters will
            at mostly include:
                alpha: Transcription rate
                beta: Splicing rate
                gamma: Spliced RNA degradation rate
                eta: Translation rate (only applicable to RNA/protein coassay)
                delta: Protein degradation rate (only applicable to RNA/protein coassay)
                alpha_b: intercept of alpha fit
                beta_b: intercept of beta fit
                gamma_b: intercept of gamma fit
                eta_b: intercept of eta fit (only applicable to RNA/protein coassay)
                delta_b: intercept of delta fit (only applicable to RNA/protein coassay)
                alpha_r2: r-squared for goodness of fit of alpha estimation
                beta_r2: r-squared for goodness of fit of beta estimation
                gamma_r2: r-squared for goodness of fit of gamma estimation
                eta_r2: r-squared for goodness of fit of eta estimation (only applicable to RNA/protein coassay)
                delta_r2: r-squared for goodness of fit of delta estimation (only applicable to RNA/protein coassay)
                alpha_logLL: loglikelihood of alpha estimation (only applicable to stochastic model)
                beta_loggLL: loglikelihood of beta estimation (only applicable to stochastic model)
                gamma_logLL: loglikelihood of gamma estimation (only applicable to stochastic model)
                eta_logLL: loglikelihood of eta estimation (only applicable to stochastic model and RNA/protein coassay)
                delta_loggLL: loglikelihood of delta estimation (only applicable to stochastic model and RNA/protein coassay)
                uu0: estimated amount of unspliced unlabeled RNA at time 0 (only applicable to data with both splicing and labeling)
                ul0: estimated amount of unspliced labeled RNA at time 0 (only applicable to data with both splicing and labeling)
                su0: estimated amount of spliced unlabeled RNA at time 0 (only applicable to data with both splicing and labeling)
                sl0: estimated amount of spliced labeled RNA at time 0 (only applicable to data with both splicing and labeling)
                U0: estimated amount of unspliced RNA (uu + ul) at time 0
                S0: estimated amount of spliced (su + sl) RNA at time 0
                total0: estimated amount of spliced (U + S) RNA at time 0
                half_life: Spliced mRNA's half-life (log(2) / gamma)

            Note that all data points are used when estimating r2 although only extreme data points are used for
            estimating r2. This is applicable to all estimation methods, either `linear_regression`, `gmm` or `negbin`.
            By default we set the intercept to be 0.

            For metabolic labeling data, the kinetic parameters will at most include:
                alpha: Transcription rate (effective - when RNA promoter switching considered)
                beta: Splicing rate
                gamma: Spliced RNA degradation rate
                a: Switching rate from active promoter state to inactive promoter state
                b: Switching rate from inactive promoter state to active promoter state
                alpha_a: Transcription rate for active promoter
                alpha_i: Transcription rate for inactive promoter
                cost: cost of the kinetic parameters estimation
                logLL: loglikelihood of kinetic parameters estimation
                alpha_r2: r-squared for goodness of fit of alpha estimation
                beta_r2: r-squared for goodness of fit of beta estimation
                gamma_r2: r-squared for goodness of fit of gamma estimation
                uu0: estimated amount of unspliced unlabeled RNA at time 0 (only applicable to data with both splicing and labeling)
                ul0: estimated amount of unspliced labeled RNA at time 0 (only applicable to data with both splicing and labeling)
                su0: estimated amount of spliced unlabeled RNA at time 0 (only applicable to data with both splicing and labeling)
                sl0: estimated amount of spliced labeled RNA at time 0 (only applicable to data with both splicing and labeling)
                u0: estimated amount of unspliced RNA (including uu, ul) at time 0
                s0: estimated amount of spliced (including su, sl) RNA at time 0
                total0: estimated amount of spliced (including U, S) RNA at time 0
                p_half_life: half-life for unspliced mRNA
                half_life: half-life for spliced mRNA

            If sanity_check has performed, a column with key `sanity_check` will also included which indicates which gene
            passes filter (`filter_gene_mode`) and sanity check. This is only applicable to kinetic and degradation metabolic
            labeling experiments.

            In addition, the `dynamics` key of the .uns attribute corresponds to a dictionary that includes the following
            keys:
                t: An array like object that indicates the time point of each cell used during parameters estimation
                    (applicable only to kinetic models)
                group: The group that you used to estimate parameters group-wise
                X_data: The input that was used for estimating parameters (applicable only to kinetic models)
                X_fit_data: The data that was fitted during parameters estimation (applicable only to kinetic models)
                asspt_mRNA: Assumption of mRNA dynamics (steady state or kinetic)
                experiment_type: Experiment type (either conventional or metabolic labeling based)
                normalized: Whether to normalize data
                model: Model used for the parameter estimation (either auto, deterministic or stochastic)
                has_splicing: Does the adata has splicing? detected automatically
                has_labeling: Does the adata has labelling? detected automatically
                has_protein: Does the adata has protein information? detected automatically
                use_smoothed: Whether to use smoothed data (or first moment, done via local average of neighbor cells)
                NTR_vel: Whether to estimate NTR velocity
                log_unnormalized: Whether to log transform unnormalized data.
    """

    X_data, X_fit_data = None, None
    filter_list, filter_gene_mode_list = ['use_for_pca', 'pass_basic_filter', 'no'], ['final', 'basic', 'no']
    filter_checker = [i in adata.var.columns for i in filter_list[:2]]
    filter_checker.append(True)
    filter_id = filter_gene_mode_list.index(filter_gene_mode)
    which_filter = np.where(filter_checker[filter_id:])[0][0] + filter_id

    filter_gene_mode = filter_gene_mode_list[which_filter]

    valid_bools = get_valid_bools(adata, filter_gene_mode)
    gene_num = sum(valid_bools)
    if gene_num == 0:
        raise Exception(f"no genes pass filter. Try resetting `filter_gene_mode = 'no'` to use all genes.")

    if model.lower() == "auto":
        model = "stochastic"
        model_was_auto = True
    else:
        model_was_auto = False

    if model.lower() == "stochastic" or use_smoothed or re_smooth:
        M_layers = [i for i in adata.layers.keys() if i.startswith("M_")]
        if re_smooth or len(M_layers) < 2:
            for i in M_layers:
                del adata.layers[i]

        if len(M_layers) < 2 or re_smooth:
            if filter_gene_mode == 'final' and 'X_pca' in adata.obsm.keys():
                adata.obsm['X'] = adata.obsm['X_pca']

            if group is not None and group in adata.obs.columns:
                moments(adata, genes=valid_bools, group=group)
            else:
                moments(adata, genes=valid_bools, group=tkey)
        elif tkey is not None:
            warnings.warn(f"You used tkey {tkey} (or group {group}), but you have calculated local smoothing (1st moment) "
                          f"for your data before. Please ensure you used the desired tkey or group when the smoothing was "
                          f"performed. Try setting re_smooth = True if not sure.")

    valid_adata = adata[:, valid_bools].copy()
    if group is not None and group in adata.obs.columns:
        _group = adata.obs[group].unique()
    else:
        _group = ["_all_cells"]

    for cur_grp_i, cur_grp in enumerate(_group):
        if cur_grp == "_all_cells":
            kin_param_pre = ""
            cur_cells_bools = np.ones(valid_adata.shape[0], dtype=bool)
            subset_adata = valid_adata[cur_cells_bools]
        else:
            kin_param_pre = group + "_" + cur_grp + "_"
            cur_cells_bools = (valid_adata.obs[group] == cur_grp).values
            subset_adata = valid_adata[cur_cells_bools]

            if model.lower() == "stochastic" or use_smoothed:
                moments(subset_adata)
        (
            U,
            Ul,
            S,
            Sl,
            P,
            US,
            U2,
            S2,
            t,
            normalized,
            has_splicing,
            has_labeling,
            has_protein,
            ind_for_proteins,
            assump_mRNA,
            exp_type,
        ) = get_data_for_kin_params_estimation(
            subset_adata,
            model,
            use_smoothed,
            tkey,
            protein_names,
            log_unnormalized,
            NTR_vel,
        )

        if experiment_type.lower() == 'auto':
            experiment_type = exp_type
        else:
            if experiment_type != exp_type:
                warnings.warn(
                "dynamo detects the experiment type of your data as {}, but your input experiment_type "
                "is {}".format(exp_type, experiment_type)
                )

        valid_bools_ = valid_bools.copy()
        if sanity_check and experiment_type in ['kin', 'deg']:
            indices_valid_bools = np.where(valid_bools)[0]
            t, L = t.flatten(), (0 if Ul is None else Ul) + (0 if Sl is None else Sl)
            t_uniq = np.unique(t)

            valid_gene_checker = np.zeros(gene_num, dtype=bool)
            for L_iter, cur_L in tqdm(enumerate(L), desc=f'sanity check of {experiment_type} experiment data:'):
                cur_L = cur_L.A.flatten() if issparse(cur_L) else cur_L.flatten()
                y = strat_mom(cur_L, t, np.nanmean)
                slope, _ = fit_linreg(t_uniq, y, intercept=True, r2=False)
                valid_gene_checker[L_iter] = True if (slope > 0 and experiment_type == 'kin') or \
                                                     (slope < 0 and experiment_type == 'deg') else False
            valid_bools_[indices_valid_bools[~valid_gene_checker]] = False
            warnings.warn(f'filtering {gene_num - valid_gene_checker.sum()} genes after sanity check.')

            if len(valid_bools_) < 5:
                raise Exception(f'After sanity check, you have less than 5 valid genes. Something is wrong about your '
                                f'metabolic labeling experiment!')

            U, Ul, S, Sl = (None if U is None else U[valid_gene_checker, :]), \
                           (None if Ul is None else Ul[valid_gene_checker, :]), \
                           (None if S is None else S[valid_gene_checker, :]), \
                           (None if Sl is None else Sl[valid_gene_checker, :])
            subset_adata = subset_adata[:, valid_gene_checker]
            adata.var[kin_param_pre + 'sanity_check'] = valid_bools_

        if assumption_mRNA.lower() == 'auto': assumption_mRNA = assump_mRNA
        if experiment_type == 'conventional': assumption_mRNA = 'ss'

        if model.lower() == "stochastic" and experiment_type.lower() not in ["conventional", "kinetics", "degradation", "kin", "deg", "one-shot"]:
            """
            # temporially convert to deterministic model as moment model for mix_std_stm
             and other types of labeling experiment is ongoing."""

            model = "deterministic"

        if assumption_mRNA.lower() == "ss" or (experiment_type.lower() in ["one-shot", "mix_std_stm"]):
            if est_method.lower() == "auto": est_method = "gmm" if model == 'stochastic' else 'ols'
            if experiment_type.lower() == "one_shot":
                beta = subset_adata.var.beta if "beta" in subset_adata.var.keys() else None
                gamma = subset_adata.var.gamma if "gamma" in subset_adata.var.keys() else None
                ss_estimation_kwargs = {"beta": beta, "gamma": gamma}

            else:
                ss_estimation_kwargs = {}

            est = ss_estimation(
                U=U.copy() if U is not None else None,
                Ul=Ul.copy() if Ul is not None else None,
                S=S.copy() if S is not None else None,
                Sl=Sl.copy() if Sl is not None else None,
                P=P.copy() if P is not None else None,
                US=US.copy() if US is not None else None,
                S2=S2.copy() if S2 is not None else None,
                conn=subset_adata.obsp['moments_con'],
                t=t,
                ind_for_proteins=ind_for_proteins,
                model=model,
                est_method=est_method,
                experiment_type=experiment_type,
                assumption_mRNA=assumption_mRNA,
                assumption_protein=assumption_protein,
                concat_data=concat_data,
                cores=cores,
                **ss_estimation_kwargs
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if experiment_type in ["one-shot", "one_shot"]:
                    est.fit(one_shot_method=one_shot_method, **est_kwargs)
                else:
                    est.fit(**est_kwargs)

            alpha, beta, gamma, eta, delta = est.parameters.values()

            U, S = get_U_S_for_velocity_estimation(
                subset_adata,
                use_smoothed,
                has_splicing,
                has_labeling,
                log_unnormalized,
                NTR_vel,
            )
            vel = velocity(estimation=est)
            vel_U = vel.vel_u(U)
            if exp_type == 'one-shot':
                vel_S = vel.vel_s(U, U + S)
            else:
                vel_S = vel.vel_s(U, S)
            vel_P = vel.vel_p(S, P)

            adata = set_velocity(
                adata,
                vel_U,
                vel_S,
                vel_P,
                _group,
                cur_grp,
                cur_cells_bools,
                valid_bools_,
                ind_for_proteins,
            )

            adata = set_param_ss(
                adata,
                est,
                alpha,
                beta,
                gamma,
                eta,
                delta,
                experiment_type,
                _group,
                cur_grp,
                kin_param_pre,
                valid_bools_,
                ind_for_proteins,
            )

        elif assumption_mRNA.lower() == "kinetic":
            if model_was_auto and experiment_type.lower() == "kin": model = "mixture"
            data_type = 'smoothed' if use_smoothed else 'sfs'

            params, half_life, cost, logLL, param_ranges, cur_X_data, cur_X_fit_data = kinetic_model(subset_adata, tkey, model, est_method, experiment_type, has_splicing,
                          has_switch=True, param_rngs={}, data_type=data_type, **est_kwargs)

            len_t, len_g = len(np.unique(t)), len(_group)
            if cur_grp == _group[0]:
                if len_g == 1:
                    X_data, X_fit_data = np.array((adata.n_vars, len_t)), np.array((adata.n_vars, len_t))
                else:
                    X_data, X_fit_data = np.array((len_g, adata.n_vars, len_t)), np.array((len_g, adata.n_vars, len_t))

            if len(_group) == 1:
                X_data, X_fit_data = cur_X_data, cur_X_fit_data
            else:
                X_data[cur_grp_i, :, :], X_fit_data[cur_grp_i, :, :] = cur_X_data, cur_X_fit_data

            a, b, alpha_a, alpha_i, alpha, beta, gamma = (
                params.loc[:, 'a'].values if 'a' in params.columns else None,
                params.loc[:, 'b'].values if 'b' in params.columns else None,
                params.loc[:, 'alpha_a'].values if 'alpha_a' in params.columns else None,
                params.loc[:, 'alpha_i'].values if 'alpha_i' in params.columns else None,
                params.loc[:, 'alpha'].values if 'alpha' in params.columns else None,
                params.loc[:, 'beta'].values if 'beta' in params.columns else None,
                params.loc[:, 'gamma'].values if 'gamma' in params.columns else None,
            )
            if alpha is None:
                alpha = fbar(a, b, alpha_a, 0) if alpha_i is None else fbar(a, b, alpha_a, alpha_i)
            all_kinetic_params = ['a', 'b', 'alpha_a', 'alpha_i', 'alpha', 'beta', 'gamma']

            extra_params = params.loc[:, params.columns.difference(all_kinetic_params)]
            # if alpha = None, set alpha to be U; N - gamma R
            params = {"alpha": alpha, "beta": beta, "gamma": gamma, "t": t}
            vel = velocity(**params)

            U, S = get_U_S_for_velocity_estimation(
                subset_adata,
                use_smoothed,
                has_splicing,
                has_labeling,
                log_unnormalized,
                NTR_vel,
            )
            vel_U = vel.vel_u(U)
            vel_S = vel.vel_u(S)
            vel_P = vel.vel_p(S, P)

            adata = set_velocity(
                adata,
                vel_U,
                vel_S,
                vel_P,
                _group,
                cur_grp,
                cur_cells_bools,
                valid_bools_,
                ind_for_proteins,
            )

            adata = set_param_kinetic(
                adata,
                alpha,
                a,
                b,
                alpha_a,
                alpha_i,
                beta,
                gamma,
                cost,
                logLL,
                kin_param_pre,
                extra_params,
                _group,
                cur_grp,
                valid_bools_,
            )
            # add protein related parameters in the moment model below:
        elif model.lower() == "model_selection":
            warnings.warn("Not implemented yet.")

    if group is not None and group in adata.obs[group]:
        uns_key = group + "_dynamics"
    else:
        uns_key = "dynamics"

    if sanity_check and experiment_type in ['kin', 'deg']:
        sanity_check_cols = adata.var.columns.str.endswith('sanity_check')
        adata.var['use_for_dynamics'] = adata.var.loc[:, sanity_check_cols].sum(1).astype(bool)
    else: 
        adata.var['use_for_dynamics'] = adata.var['use_for_pca'].copy()

    adata.uns[uns_key] = {
        "filter_gene_mode": filter_gene_mode,
        "t": t,
        "group": group,
        "X_data": X_data,
        "X_fit_data": X_fit_data,
        "asspt_mRNA": assumption_mRNA,
        "experiment_type": experiment_type,
        "normalized": normalized,
        "model": model,
        "has_splicing": has_splicing,
        "has_labeling": has_labeling,
        "has_protein": has_protein,
        "use_smoothed": use_smoothed,
        "NTR_vel": NTR_vel,
        "log_unnormalized": log_unnormalized,
    }

    return adata


def kinetic_model(subset_adata, tkey, model, est_method, experiment_type, has_splicing, has_switch, param_rngs,
                  data_type='sfs', **est_kwargs):
    """est_method is not used. data_type can either 'sfs' or 'smoothed'."""
    time = subset_adata.obs[tkey].astype('float')

    if experiment_type.lower() == 'kin':
        if has_splicing:
            layers = ['M_ul', 'M_sl', 'M_uu', 'M_su'] if (
                        'M_ul' in subset_adata.layers.keys() and data_type == 'smoothed') \
                else ['X_ul', 'X_sl', 'X_uu', 'X_su']

            if model in ['deterministic', 'stochastic']:
                layer_u = 'M_ul' if ('M_ul' in subset_adata.layers.keys() and data_type == 'smoothed') else 'X_ul'
                layer_s = 'M_sl' if ('M_ul' in subset_adata.layers.keys() and data_type == 'smoothed') else 'X_sl'

                X, X_raw = prepare_data_has_splicing(subset_adata, subset_adata.var.index, time,
                                                     layer_u=layer_u, layer_s=layer_s, total_layers=layers)
            elif model.startswith('mixture'):
                X, _, X_raw = prepare_data_deterministic(subset_adata, subset_adata.var.index, time,
                                                         layers=layers, total_layers=layers)

            if model == 'deterministic':
                X = [X[i][[0, 1], :] for i in range(len(X))]
                _param_ranges = {'alpha': [0, 1000], 'beta': [0, 1000], 'gamma': [0, 1000]}
                x0 = {'u0': [0, 1000], 's0': [0, 1000]}
                Est, _ = Estimation_DeterministicKin, Deterministic
            elif model == 'stochastic':
                x0 = {'u0': [0, 1000], 's0': [0, 1000],
                      'uu0': [0, 1000], 'ss0': [0, 1000],
                      'us0': [0, 1000]}

                if has_switch:
                    _param_ranges = {'a': [0, 1000], 'b': [0, 1000],
                                    'alpha_a': [0, 1000], 'alpha_i': 0,
                                    'beta': [0, 1000], 'gamma': [0, 1000], }
                    Est, _ = Estimation_MomentKin, Moments
                else:
                    _param_ranges = {'alpha': [0, 1000], 'beta': [0, 1000], 'gamma': [0, 1000], }

                    Est, _ = Estimation_MomentKinNoSwitch, Moments_NoSwitching
            elif model == 'mixture':
                _param_ranges = {'alpha': [0, 1000], 'alpha_2': [0, 0], 'beta': [0, 1000], 'gamma': [0, 1000], }
                x0 = {'ul0': [0, 0], 'sl0': [0, 0], 'uu0': [0, 1000], 'su0': [0, 1000]}

                Est = Mixture_KinDeg_NoSwitching(Deterministic(), Deterministic())
            elif model == 'mixture_deterministic_stochastic':
                X, X_raw = prepare_data_mix_has_splicing(subset_adata, subset_adata.var.index, time, layer_u=layers[2],
                                                         layer_s=layers[3], layer_ul=layers[0], layer_sl=layers[1],
                                                         total_layers=layers, mix_model_indices=[0, 1, 5, 6, 7, 8, 9])

                _param_ranges = {'alpha': [0, 1000], 'alpha_2': [0, 0], 'beta': [0, 1000], 'gamma': [0, 1000], }
                x0 = {'ul0': [0, 0], 'sl0': [0, 0],
                      'u0': [0, 1000], 's0': [0, 1000],
                      'uu0': [0, 1000], 'ss0': [0, 1000],
                      'us0': [0, 1000], }
                Est = Mixture_KinDeg_NoSwitching(Deterministic(), Moments_NoSwitching())
            elif model == 'mixture_stochastic_stochastic':
                _param_ranges = {'alpha': [0, 1000], 'alpha_2': [0, 0], 'beta': [0, 1000], 'gamma': [0, 1000], }
                X, X_raw  = prepare_data_mix_has_splicing(subset_adata, subset_adata.var.index, time, layer_u=layers[2],
                                                          layer_s=layers[3], layer_ul=layers[0], layer_sl=layers[1],
                                                          total_layers=layers, mix_model_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                x0 = {'ul0': [0, 1000], 'sl0': [0, 1000],
                      'ul_ul0': [0, 1000], 'sl_sl0': [0, 1000],
                      'ul_sl0': [0, 1000],
                      'u0': [0, 1000], 's0': [0, 1000],
                      'uu0': [0, 1000], 'ss0': [0, 1000],
                      'us0': [0, 1000], }
                Est = Mixture_KinDeg_NoSwitching(Moments_NoSwitching(), Moments_NoSwitching())
            else:
                raise NotImplementedError(f'model {model} with kinetic assumption is not implemented. '
                                f'current supported models for kinetics experiments include: stochastic, deterministic, mixture,'
                                f'mixture_deterministic_stochastic or mixture_stochastic_stochastic')
        else:
            total_layer = 'M_t' if ('M_t' in subset_adata.layers.keys() and data_type == 'smoothed') else 'X_total'

            if model in ['deterministic', 'stochastic']:
                layer = 'M_n' if ('M_n' in subset_adata.layers.keys() and data_type == 'smoothed') else 'X_new'
                X, X_raw = prepare_data_no_splicing(subset_adata, subset_adata.var.index, time, layer=layer,
                                                    total_layer=total_layer)
            elif model.startswith('mixture'):
                layers = ['M_n', 'M_t'] if ('M_n' in subset_adata.layers.keys() and data_type == 'smoothed') \
                    else ['X_new', 'X_total']

                X, _, X_raw = prepare_data_deterministic(subset_adata, subset_adata.var.index, time, layers=layers,
                                                         total_layers=total_layer)

            if model == 'deterministic':
                X = [X[i][0, :] for i in range(len(X))]
                _param_ranges = {'alpha': [0, 1000], 'gamma': [0, 1000], }
                x0 = {'u0': [0, 1000]}
                Est, _ = Estimation_DeterministicKinNosp, Deterministic_NoSplicing
            elif model == 'stochastic':
                x0 = {'u0': [0, 1000], 'uu0': [0, 1000], }
                if has_switch:
                    _param_ranges = {'a': [0, 1000], 'b': [0, 1000],
                                    'alpha_a': [0, 1000], 'alpha_i': 0,
                                    'gamma': [0, 1000], }
                    Est, _ = Estimation_MomentKinNosp, Moments_Nosplicing
                else:
                    _param_ranges = {'alpha': [0, 1000], 'gamma': [0, 1000], }
                    Est, _ = Estimation_MomentKinNoSwitchNoSplicing, Moments_NoSwitchingNoSplicing
            elif model == 'mixture':
                _param_ranges = {'alpha': [0, 1000], 'alpha_2': [0, 0], 'gamma': [0, 1000], }
                x0 = {'u0': [0, 0], 'o0': [0, 1000]}
                Est = Mixture_KinDeg_NoSwitching(Deterministic_NoSplicing(), Deterministic_NoSplicing())
            elif model == 'mixture_deterministic_stochastic':
                X, X_raw = prepare_data_mix_no_splicing(subset_adata, subset_adata.var.index, time,
                                                        layer_n=layers[0], layer_t=layers[1], total_layer=total_layer,
                                                        mix_model_indices=[0, 2, 3])

                _param_ranges = {'alpha': [0, 1000], 'alpha_2': [0, 0], 'gamma': [0, 1000], }
                x0 = {'u0': [0, 1000], 'o0': [0, 1000], 'oo0': [0, 1000]}
                Est = Mixture_KinDeg_NoSwitching(Deterministic_NoSplicing(), Moments_NoSwitchingNoSplicing())
            elif model == 'mixture_stochastic_stochastic':
                X, X_raw = prepare_data_mix_no_splicing(subset_adata, subset_adata.var.index, time,
                                                        layer_n=layers[0], layer_t=layers[1], total_layer=total_layer,
                                                        mix_model_indices=[0, 1, 2, 3])

                _param_ranges = {'alpha': [0, 1000], 'alpha_2': [0, 0], 'gamma': [0, 1000], }
                x0 = {'u0': [0, 1000], 'uu0': [0, 1000], 'o0': [0, 1000], 'oo0': [0, 1000]}
                Est = Mixture_KinDeg_NoSwitching(Moments_NoSwitchingNoSplicing(), Moments_NoSwitchingNoSplicing())
            else:
                raise Exception(f'model {model} with kinetic assumption is not implemented. '
                                f'current supported models for kinetics experiments include: stochastic, deterministic, mixture,'
                                f'mixture_deterministic_stochastic or mixture_stochastic_stochastic')
    elif experiment_type.lower() == 'deg':
        if has_splicing:
            layers = ['M_ul', 'M_sl', 'M_uu', 'M_su'] if (
                        'M_ul' in subset_adata.layers.keys() and data_type == 'smoothed') \
                else ['X_ul', 'X_sl', 'X_uu', 'X_su']

            if model in ['deterministic', 'stochastic']:
                layer_u = 'M_ul' if ('M_ul' in subset_adata.layers.keys() and data_type == 'smoothed') else 'X_ul'
                layer_s = 'M_sl' if ('M_sl' in subset_adata.layers.keys() and data_type == 'smoothed') else 'X_sl'

                X, X_raw = prepare_data_has_splicing(subset_adata, subset_adata.var.index, time,
                                                     layer_u=layer_u, layer_s=layer_s, total_layers=layers)
            elif model.startswith('mixture'):
                X, _, X_raw = prepare_data_deterministic(subset_adata, subset_adata.var.index, time,
                                                         layers=layers, total_layers=layers)

            if model == 'deterministic':
                X = [X[i][[0, 1], :] for i in range(len(X))]
                _param_ranges = {'beta': [0, 1000], 'gamma': [0, 1000], }
                x0 = {'u0': [0, 1000], 's0': [0, 1000], }
                Est, _ = Estimation_DeterministicDeg, Deterministic
            elif model == 'stochastic':
                _param_ranges = {'beta': [0, 1000], 'gamma': [0, 1000], }
                x0 = {'u0': [0, 1000], 's0': [0, 1000],
                      'uu0': [0, 1000], 'ss0': [0, 1000],
                      'us0': [0, 1000], }
                Est, _ = Estimation_MomentDeg, Moments_NoSwitching
            else:
                raise NotImplementedError(f'model {model} with kinetic assumption is not implemented. '
                            f'current supported models for degradation experiment include: '
                            f'stochastic, deterministic.')
        else:
            total_layer = 'M_t' if ('M_t' in subset_adata.layers.keys() and data_type == 'smoothed') else 'X_total'

            layer = 'M_n' if ('M_n' in subset_adata.layers.keys() and data_type == 'smoothed') else 'X_new'
            X, X_raw = prepare_data_no_splicing(subset_adata, subset_adata.var.index, time,
                                                layer=layer, total_layer=total_layer)

            if model == 'deterministic':
                X = [X[i][0, :] for i in range(len(X))]
                _param_ranges = {'gamma': [0, 10], }
                x0 = {'u0': [0, 1000]}
                Est, _ = Estimation_DeterministicDegNosp, Deterministic_NoSplicing
            elif model == 'stochastic':
                _param_ranges = {'gamma': [0, 10], }
                x0 = {'u0': [0, 1000], 'uu0': [0, 1000]}
                Est, _ = Estimation_MomentDegNosp, Moments_NoSwitchingNoSplicing
            else:
                raise NotImplementedError(f'model {model} with kinetic assumption is not implemented. '
                                f'current supported models for degradation experiment include: '
                                f'stochastic, deterministic.')
    elif experiment_type.lower() == 'mix_std_stm':
        raise Exception(f'experiment {experiment_type} with kinetic assumption is not implemented')
    elif experiment_type.lower() == 'mix_pulse_chase':
        raise Exception(f'experiment {experiment_type} with kinetic assumption is not implemented')
    elif experiment_type.lower() == 'pulse_time_series':
        raise Exception(f'experiment {experiment_type} with kinetic assumption is not implemented')
    elif experiment_type.lower() == 'dual_labeling':
        raise Exception(f'experiment {experiment_type} with kinetic assumption is not implemented')
    else:
        raise Exception(f'experiment {experiment_type} is not recognized')

    _param_ranges = update_dict(_param_ranges, param_rngs)
    x0_ = np.vstack([ran for ran in x0.values()]).T

    n_genes = subset_adata.n_vars
    cost, logLL = np.zeros(n_genes), np.zeros(n_genes)
    all_keys = list(_param_ranges.keys()) + list(x0.keys())
    all_keys = [cur_key for cur_key in all_keys if cur_key != 'alpha_i']
    half_life, Estm = np.zeros(n_genes), [None] * n_genes
    X_data, X_fit_data = [None] * n_genes, [None] * n_genes

    for i_gene in tqdm(range(n_genes), desc="estimating kinetic-parameters using kinetic model"):
        if model.startswith('mixture'):
            estm = Est
            if model == 'mixture':
                cur_X_data = np.vstack([X[i_layer][i_gene] for i_layer in range(len(X))])
                if issparse(X_raw[0]):
                    cur_X_raw = np.hstack([X_raw[i_layer][:, i_gene].A for i_layer in range(len(X))])
                else:
                    cur_X_raw = np.hstack([X_raw[i_layer][:, i_gene] for i_layer in range(len(X))])
            else:
                cur_X_data = X[i_gene]
                cur_X_raw = X_raw[i_gene]

                if issparse(cur_X_raw[0, 0]):
                    cur_X_raw = np.hstack((cur_X_raw[0, 0].A, cur_X_raw[1, 0].A))

            _, cost[i_gene] = estm.auto_fit(np.unique(time), cur_X_data)
            model_1, model_2, kinetic_parameters, mix_x0 = estm.export_dictionary().values()
            tmp = list(kinetic_parameters.values())
            tmp.extend(mix_x0)
            Estm[i_gene] = tmp
        else:
            if experiment_type.lower() == 'kin':
                cur_X_data, cur_X_raw = X[i_gene], X_raw[i_gene]

                if has_splicing:
                    alpha0 = guestimate_alpha(np.sum(cur_X_data, 0), np.unique(time))
                else:
                    alpha0 = guestimate_alpha(cur_X_data, np.unique(time)) if cur_X_data.ndim == 1 \
                        else guestimate_alpha(cur_X_data[0], np.unique(time))

                if model =='stochastic':
                    _param_ranges.update({'alpha_a': [0, alpha0*10]})
                elif model == 'deterministic':
                    _param_ranges.update({'alpha': [0, alpha0 * 10]})
                param_ranges = [ran for ran in _param_ranges.values()]

                estm = Est(*param_ranges, x0=x0_) if 'x0' in inspect.getfullargspec(Est) \
                    else Est(*param_ranges)
                _, cost[i_gene] = estm.fit_lsq(np.unique(time), cur_X_data, **est_kwargs)
                if model == 'deterministic':
                    Estm[i_gene] = estm.export_parameters()
                else:
                    tmp = np.ma.array(estm.export_parameters(), mask=False)
                    tmp.mask[3] = True
                    Estm[i_gene] = tmp.compressed()

            elif experiment_type.lower() == 'deg':
                estm = Est()
                cur_X_data, cur_X_raw = X[i_gene], X_raw[i_gene]
                
                _, cost[i_gene] = estm.auto_fit(np.unique(time), cur_X_data)
                Estm[i_gene] = estm.export_parameters()[1:]

            if issparse(cur_X_raw[0, 0]):
                cur_X_raw = np.hstack((cur_X_raw[0, 0].A, cur_X_raw[1, 0].A))
            # model_1, kinetic_parameters, mix_x0 = estm.export_dictionary().values()
            # tmp = list(kinetic_parameters.values())
            # tmp.extend(mix_x0)
            # Estm[i_gene] = tmp

        X_data[i_gene] = cur_X_data
        if model.startswith('mixture'):
            X_fit_data[i_gene] = estm.simulator.x.T
            X_fit_data[i_gene][estm.model1.n_species:] *= estm.scale
        else:
            if hasattr(estm, "extract_data_from_simulator"):
                X_fit_data[i_gene] = estm.extract_data_from_simulator()
            else:
                X_fit_data[i_gene] = estm.simulator.x.T

        half_life[i_gene] = np.log(2)/Estm[i_gene][-1] if experiment_type.lower() == 'kin' else estm.calc_half_life('gamma')

        if model.startswith('mixture'):
            species = [0, 1, 2, 3] if has_splicing else [0, 1]
            gof = GoodnessOfFit(estm.export_model(), params=estm.export_parameters())
            gof.prepare_data(time, cur_X_raw.T, species=species, normalize=True)
        else:
            gof = GoodnessOfFit(estm.export_model(), params=estm.export_parameters(), x0=estm.simulator.x0)
            gof.prepare_data(time, cur_X_raw.T, normalize=True)

        logLL[i_gene] = gof.calc_gaussian_loglikelihood()

    Estm_df = pd.DataFrame(np.vstack(Estm), columns=[*all_keys[:len(Estm[0])]])

    return Estm_df, half_life, cost, logLL, _param_ranges, X_data, X_fit_data


def fbar(a, b, alpha_a, alpha_i):
    if any([i is None for i in [a, b, alpha_a, alpha_i]]):
        return None
    else:
        return b / (a + b) * alpha_a + a / (a + b) * alpha_i


def _get_dispatcher():
    dispatcher = {'Deterministic': Deterministic,
                  'Deterministic_NoSplicing': Deterministic_NoSplicing,
                  'Moments_NoSwitching': Moments_NoSwitching,
                  'Moments_NoSwitchingNoSplicing': Moments_NoSwitchingNoSplicing,
                  'Mixture_KinDeg_NoSwitching': Mixture_KinDeg_NoSwitching,
                  }

    return dispatcher
