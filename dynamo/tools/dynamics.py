import pandas as pd
from .moments import moments
from .velocity import velocity
from .velocity import ss_estimation
from .estimation_kinetic import *
from .utils_kinetic import *
from .utils import (
    update_dict,
    get_valid_inds,
    get_data_for_velocity_estimation,
    get_U_S_for_velocity_estimation,
)
from .utils import set_velocity, set_param_ss, set_param_kinetic
from .moments import prepare_data_no_splicing, prepare_data_has_splicing

# incorporate the model selection code soon
def dynamics(
    adata,
    tkey=None,
    filter_gene_mode="final",
    use_moments=True,
    experiment_type='auto',
    assumption_mRNA='auto',
    assumption_protein="ss",
    model="stochastic",
    est_method="auto",
    NTR_vel=False,
    group=None,
    protein_names=None,
    concat_data=False,
    log_unnormalized=True,
    one_shot_method="combined",
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
            The column key for the time label of cells in .obs. Used for either "steady_state" or non-"steady_state" mode or `moment`
            mode  with labeled data.
        filter_gene_mode: `str` (default: `final`)
            The string for indicating which mode (one of, {'final', 'basic', 'no'}) of gene filter will be used.
        use_moments: `bool` (default: `True`)
            Whether to use the smoothed data when calculating velocity for each gene. `use_smoothed` is only relevant when
            model is `linear_regression` (and experiment_type and assumption_mRNA correspond to `conventional` and `ss` implicitly).
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
        model: `str` {`auto`, `deterministic`, `stochastic`} (default: `stochastic`)
            String indicates which estimation model will be used.
            (1) 'deterministic': The method based on `deterministic` ordinary differential equations;
            (2) 'stochastic' or `moment`: The new method from us that is based on `stochastic` master equations;
            Note that `kinetic` model doesn't need to assumes the `experiment_type` is not `conventional`. As other labeling
            experiments, if you specify the `tkey`, dynamo can also apply `kinetic` model on `conventional` scRNA-seq datasets.
            A "model_selection" model will be supported soon in which alpha, beta and gamma will be modeled as a function of time.
        est_method: `str` {`linear_regression`, `gmm`, `negbin`, `auto`} This parameter should be used in conjunction with `model` parameter.
            * Available options when the `model` is 'ss' include:
            (1) 'linear_regression': The canonical method from the seminar RNA velocity paper based on deterministic ordinary
            differential equations;
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
            of cells. This will be used to estimate group-specific (i.e cell-type specific) kinetic parameters.
        protein_names: `List`
            A list of gene names corresponds to the rows of the measured proteins in the `X_protein` of the `obsm` attribute.
            The names have to be included in the adata.var.index.
        concat_data: `bool` (default: `False`)
            Whether to concatenate data before estimation. If your data is a list of matrices for each time point, this need to be set as True.
        log_unnormalized: `bool` (default: `True`)
            Whether to log transform the unnormalized data.
        **est_kwargs
            Other arguments passed to the estimation methods. Not used for now.
    Returns
    -------
        adata: :class:`~anndata.AnnData`
            A updated AnnData object with estimated kinetic parameters and inferred velocity included.
    """

    filter_list, filter_gene_mode_lit = ['use_for_dynamo', 'pass_basic_filter', 'no'], ['final', 'basic', 'no']
    filter_checker = [i in adata.var.columns for i in filter_list[:2]]
    filter_checker.append(True)
    which_filter = np.where(filter_checker[filter_gene_mode_lit.index(filter_gene_mode):])[0][0]

    filter_gene_mode = filter_gene_mode_lit[which_filter]

    valid_ind = get_valid_inds(adata, filter_gene_mode)

    if model == "stochastic" or use_moments:
        if len([i for i in adata.layers.keys() if i.startswith("M_")]) < 2:
            moments(adata)

    valid_adata = adata[:, valid_ind].copy()
    if group is not None and group in adata.obs.columns:
        _group = adata.obs[group].unique()
    else:
        _group = ["_all_cells"]

    for cur_grp in _group:
        if cur_grp == "_all_cells":
            kin_param_pre = ""
            cur_cells_bools = np.ones(valid_adata.shape[0], dtype=bool)
            subset_adata = valid_adata[cur_cells_bools]
        else:
            kin_param_pre = group + "_" + cur_grp + "_"
            cur_cells_bools = (valid_adata.obs[group] == cur_grp).values
            subset_adata = valid_adata[cur_cells_bools]

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
        ) = get_data_for_velocity_estimation(
            subset_adata,
            model,
            use_moments,
            tkey,
            protein_names,
            log_unnormalized,
            NTR_vel,
        )

        if experiment_type == 'auto':
            experiment_type = exp_type
        else:
            if experiment_type != exp_type:
                warnings.warn(
                "dynamo detects the experiment type of your data as {}, but your input experiment_type "
                "is {}".format(exp_type, experiment_type)
                )

        if assumption_mRNA is 'auto': assumption_mRNA = assump_mRNA

        if model == "stochastic" and experiment_type not in ["conventional", "kinetics", "degradation", "kin", "deg"]:
            """
            # temporially convert to deterministic model as moment model for one-shot, mix_std_stm
             and other types of labeling experiment is ongoing."""

            model = "deterministic"

        if assumption_mRNA == 'ss' or (experiment_type in ['one-shot', 'mix_std_stm']):
            est = ss_estimation(
                U=U,
                Ul=Ul,
                S=S,
                Sl=Sl,
                P=P,
                US=US,
                S2=S2,
                t=t,
                ind_for_proteins=ind_for_proteins,
                model=model,
                est_method=est_method,
                experiment_type=experiment_type,
                assumption_mRNA=assumption_mRNA,
                assumption_protein=assumption_protein,
                concat_data=concat_data,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if experiment_type in ["one-shot", "one_shot"]:
                    est.fit(one_shot_method=one_shot_method)
                else:
                    est.fit()

            alpha, beta, gamma, eta, delta = est.parameters.values()

            U, S = get_U_S_for_velocity_estimation(
                subset_adata,
                use_moments,
                has_splicing,
                has_labeling,
                log_unnormalized,
                NTR_vel,
            )
            vel = velocity(estimation=est)
            vel_U = vel.vel_u(U)
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
                valid_ind,
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
                valid_ind,
                ind_for_proteins,
            )

        elif assumption_mRNA == 'kinetic':
            params, half_life, cost, logLL, param_ranges = kinetic_model(subset_adata, tkey, est_method, experiment_type, has_splicing,
                          has_switch=True, param_rngs={}, **est_kwargs)
            a, b, alpha_a, alpha_i, alpha, beta, gamma = (
                params.loc[:, 'a'] if 'a' in params.columns else None,
                params.loc[:, 'b'] if 'b' in params.columns else None,
                params.loc[:, 'alpha_a'] if 'alpha_a' in params.columns else None,
                params.loc[:, 'alpha_i'] if 'alpha_i' in params.columns else None,
                params.loc[:, 'alpha'] if 'alpha' in params.columns else None,
                params.loc[:, 'beta'] if 'beta' in params.columns else None,
                params.loc[:, 'gamma'] if 'gamma' in params.columns else None,
            )
            all_kinetic_params = ['a', 'b', 'alpha_a', 'alpha_i', 'alpha', 'beta', 'gamma']

            extra_params = params.loc[:, params.columns.difference(all_kinetic_params)]

            params = {"alpha": alpha, "beta": beta, "gamma": gamma, "t": t}
            vel = velocity(**params)

            U, S = get_U_S_for_velocity_estimation(
                subset_adata,
                use_moments,
                has_splicing,
                has_labeling,
                log_unnormalized,
                NTR_vel,
            )
            vel_U = vel.vel_u(U)
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
                valid_ind,
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
                valid_ind,
            )
            # add protein related parameters in the moment model below:
        elif model is "model_selection":
            warnings.warn("Not implemented yet.")

    if group is not None and group in adata.obs[group]:
        uns_key = group + "_dynamics"
    else:
        uns_key = "dynamics"

    adata.uns[uns_key] = {
        "t": t,
        "group": group,
        "asspt_mRNA": assumption_mRNA,
        "experiment_type": experiment_type,
        "normalized": normalized,
        "model": model,
        "has_splicing": has_splicing,
        "has_labeling": has_labeling,
        "has_protein": has_protein,
        "use_smoothed": use_moments,
        "NTR_vel": NTR_vel,
        "log_unnormalized": log_unnormalized,
    }

    return adata


def kinetic_model(subset_adata, tkey, est_method, experiment_type, has_splicing, has_switch, param_rngs, **est_kwargs):
    time = subset_adata.obs[tkey].astype('float')

    if experiment_type == 'kin':
        if has_splicing:
            X = prepare_data_has_splicing(subset_adata, subset_adata.var.index, time, layer_u='X_ul', layer_s='X_sl')

            if est_method == 'deterministic': # 0 - to 10 initial value
                param_ranges = {'alpha': [0, 1000], 'beta': [0, 1000], 'gamma': [0, 1000],
                                'u0': [0, 1000], 's0': [0, 1000],
                                }
                X = X[:, [0, 1]]
                Est = Estimation_DeterministicKin
            else:
                if has_switch:
                    param_ranges = {'a': [0, 1000], 'b': [0, 1000],
                                    'alpha_a': [0, 1000], 'alpha_i': 0,
                                    'beta': [0, 1000], 'gamma': [0, 1000],
                                    'u0': [0, 1000], 's0': [0, 1000],
                                    'uu0': [0, 1000], 'ss0': [0, 1000],
                                    'us0': [0, 1000], }
                    Est = Estimation_MomentKin
                else:
                    param_ranges = {'alpha': [0, 1000], 'beta': [0, 1000], 'gamma': [0, 1000],
                                    'u0': [0, 1000], 's0': [0, 1000],
                                    'uu0': [0, 1000], 'ss0': [0, 1000],
                                    'us0': [0, 1000], }
                    Est = Estimation_MomentKinNoSwitch
        else:
            X = prepare_data_no_splicing(subset_adata, subset_adata.var.index, time, layer='X_new')

            if est_method == 'deterministic':
                param_ranges = {'alpha': [0, 1000], 'gamma': [0, 1000],
                                'u0': [0, 1000]}
                X = X[:, 0]
                Est = Estimation_DeterministicKinNosp
            else:
                if has_switch:
                    param_ranges = {'a': [0, 1000], 'b': [0, 1000],
                                    'alpha_a': [0, 1000], 'alpha_i': 0,
                                    'gamma': [0, 1000],
                                    'u0': [0, 1000], 'uu0': [0, 1000], }
                    Est = Estimation_MomentKinNosp

                else:
                    param_ranges = {'alpha': [0, 1000], 'gamma': [0, 1000],
                                    'u0': [0, 1000], 'uu0': [0, 1000]}
                    Est = Estimation_MomentKinNoSwitchNoSplicing

    elif experiment_type == 'deg':
        if has_splicing:
            X = prepare_data_has_splicing(subset_adata, subset_adata.var.index, time, layer_u='X_ul', layer_s='X_sl')

            if est_method == 'deterministic':
                param_ranges = {'beta': [0, 1000], 'gamma': [0, 1000],
                                'u0': [0, 1000], 's0': [0, 1000],
                                }
                X = X[:, [0, 1]]
                Est = Estimation_DeterministicDeg
            else:
                param_ranges = {'beta': [0, 1000], 'gamma': [0, 1000],
                                'u0': [0, 1000], 's0': [0, 1000],
                                'uu0': [0, 1000], 'ss0': [0, 1000],
                                'us0': [0, 1000], }
                Est = Estimation_MomentDeg
        else:
            X = prepare_data_no_splicing(subset_adata, subset_adata.var.index, time, layer='X_new')

            if est_method == 'deterministic':
                param_ranges = {'gamma': [0, 10], 'u0': [0, 1000]}
                X = X[:, 0]
                Est = Estimation_DeterministicDegNosp
            else:
                param_ranges = {'gamma': [0, 10], 'u0': [0, 1000], 'uu0': [0, 1000]}
                Est = Estimation_MomentDegNosp

    elif experiment_type == 'mix_std_stm':
        raise Exception(f'experiment {experiment_type} with kinetic assumption is not implemented')
    elif experiment_type == 'mix_pulse_chase':
        raise Exception(f'experiment {experiment_type} with kinetic assumption is not implemented')
    elif experiment_type == 'pulse_time_series':
        raise Exception(f'experiment {experiment_type} with kinetic assumption is not implemented')
    elif experiment_type == 'dual_labeling':
        raise Exception(f'experiment {experiment_type} with kinetic assumption is not implemented')
    else:
        raise Exception(f'experiment {experiment_type} is not recognized')

    param_ranges = update_dict(param_ranges, param_rngs)
    param_ranges = [ran for ran in param_ranges.values()]

    cost, logLL = np.zeros(X.shape[0]), np.zeros(X.shape[0])
    half_life, Estm = np.zeros(X.shape[0]), np.zeros((X.shape[0], len(param_ranges)))

    for i in range(len(X)):
        estm = Est(param_ranges)
        if len(param_rngs) == 0: estm.set_param_range_partial()
        Estm[i], cost[i] = estm.fit_lsq(np.unique(time), X[i], **est_kwargs)
        half_life[i] = estm.calc_half_life()
        gof = GoodnessOfFit(Moments_NoSwitching(), params=estm.export_parameters())
        gof.prepare_data(time, X[i], normalize=True)
        logLL[i] = gof.calc_gaussian_loglikelihood()

    Estm_df = pd.DataFrame(Estm, columns=[*param_ranges])

    return Estm_df, half_life, cost, logLL, param_ranges


