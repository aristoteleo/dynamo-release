import warnings
import numpy as np
from .velocity import velocity, estimation
from .moments import MomData, Estimation
from scipy.sparse import issparse, csr_matrix
from .utils import get_U_S_for_velocity_estimation

# incorporate the model selection code soon
def dynamics(adata, filter_gene_mode='final', mode='deterministic', tkey='Time', protein_names=None,
             experiment_type='deg', assumption_mRNA=None, assumption_protein='ss', concat_data=False,
             log_unnormalized=True):
    """Inclusive model of expression dynamics with scSLAM-seq and multiomics.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        filter_gene_mode: `str` (default: `final`)
            The string for indicating which mode (one of, ['final', 'basic', 'no']) of gene filter will be used.
        mode: `str` (default: deterministic)
            string indicates which estimation mode will be used. Currently "deterministic" and "moment" methods are supported.
            A "model_selection" mode will be supported soon in which alpha, beta and gamma will be modeled as a function of time.
        tkey: `str` (default: Time)
            The column key for the time label of cells in .obs. Used for either "steady_state" or non-"steady_state" mode or `moment` mode  with labeled data.
        protein_names: `List`
            A list of gene names corresponds to the rows of the measured proteins in the X_protein of the obsm attribute. The names have to be included
            in the adata.var.index.
        experiment_type: str
            labelling experiment type. Available options are:
            (1) 'deg': degradation experiment;
            (2) 'kin': synthesis experiment;
            (3) 'one-shot': one-shot kinetic experiment.
        assumption_mRNA: str
            Parameter estimation assumption for mRNA. Available options are:
            (1) 'ss': pseudo steady state;
            (2) None: kinetic data with no assumption.
            If no labelling data exists, assumption_mRNA will automatically set to be 'ss'.
        assumption_protein: str
            Parameter estimation assumption for protein. Available options are:
            (1) 'ss': pseudo steady state;
        concat_data: bool (default: False)
            Whether to concatenate data before estimation. If your data is a list of matrices for each time point, this need to be set as True.
        get_fraction_for_deg: bool (default: True)
            Whether to calculate the fraction of relevant (labeled or unlabeled) species when using first order degradation model to estimate beta or gamma.
        log_unnormalized: bool (default: True)
            Whether to concatenate data before estimation. If your data is a list of matrices for each time point, this need to be set as True.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            A updated AnnData object with estimated kinetic parameters and inferred velocity included.
    """
    if 'use_for_dynamo' not in adata.var.columns and 'pass_basic_filter' not in adata.var.columns:
        filter_gene_mode = 'no'

    if filter_gene_mode is 'final':
        valid_ind = adata.var.use_for_dynamo
        # import warnings
        # from scipy.sparse import SparseEfficiencyWarning
        # warnings.simplefilter('ignore', SparseEfficiencyWarning)
    elif filter_gene_mode is 'basic':
        valid_ind = adata.var.pass_basic_filter
    elif filter_gene_mode is 'no':
        valid_ind = np.repeat([True], adata.shape[1])

    subset_adata = adata[:, valid_ind].copy()

    if tkey in adata.obs.columns:
        t = np.array(adata.obs[tkey], dtype='float')
    else:
        raise Exception('the tkey ', tkey, ' provided is not a valid column name in .obs.')

    U, Ul, S, Sl, P = None, None, None, None, None  # U: unlabeled unspliced; S: unlabel spliced: S
    normalized, has_splicing, has_labeling, has_protein = False, False, False, False

    if 'X_unspliced' in subset_adata.layers.keys():
        has_splicing, normalized = True, True
        U = subset_adata.layers['X_unspliced'].T
    elif 'unspliced' in subset_adata.layers.keys():
        has_splicing = True
        raw, row_unspliced = subset_adata.layers['unspliced'].T, subset_adata.layers['unspliced'].T
        if issparse(raw):
            raw.data = np.log(raw.data + 1) if log_unnormalized else raw.data
        else:
            raw = np.log(raw + 1) if log_unnormalized else raw
        U = raw

    elif 'X_new' in subset_adata.layers.keys():  # run new / total ratio (NTR)
        has_labeling, normalized = True, True
        U = subset_adata.layers['X_total'].T - subset_adata.layers['X_new'].T
        Ul = subset_adata.layers['X_new'].T
    elif 'new' in subset_adata.layers.keys():
        has_labeling = True
        raw, raw_new, old = subset_adata.layers['new'].T, subset_adata.layers['new'].T, subset_adata.layers['total'].T - subset_adata.layers['new'].T
        if issparse(raw):
            raw.data = np.log(raw.data + 1) if log_unnormalized else raw.data
            old.data = np.log(old.data + 1) if log_unnormalized else old.data
        else:
            raw = np.log(raw + 1) if log_unnormalized else raw
            old = np.log(old + 1) if log_unnormalized else old
        U = old
        Ul = raw
    elif 'X_uu' in subset_adata.layers.keys():  # only uu, ul, su, sl provided
        has_splicing, has_labeling, normalized = True, True, True

        U = subset_adata.layers['X_uu'].T  # unlabel unspliced: U
    elif 'uu' in subset_adata.layers.keys():
        raw, raw_uu = subset_adata.layers['uu'].T, subset_adata.layers['uu'].T
        if issparse(raw):
            raw.data = np.log(raw.data + 1) if log_unnormalized else raw.data
        else:
            raw = np.log(raw + 1) if log_unnormalized else raw
        U = raw

    if 'X_spliced' in subset_adata.layers.keys():
        S = subset_adata.layers['X_spliced'].T
    elif 'spliced' in subset_adata.layers.keys():
        raw, raw_spliced = subset_adata.layers['spliced'].T, subset_adata.layers['spliced'].T
        if issparse(raw):
            raw.data = np.log(raw.data + 1) if log_unnormalized else raw.data
        else:
            raw = np.log(raw + 1) if log_unnormalized else raw
        S = raw

    elif 'X_su' in subset_adata.layers.keys():  # unlabel spliced: S
        S = subset_adata.layers['X_su'].T
    elif 'su' in subset_adata.layers.keys():
        raw, raw_su = subset_adata.layers['su'].T, subset_adata.layers['su'].T
        if issparse(raw):
            raw.data = np.log(raw.data + 1) if log_unnormalized else raw.data
        else:
            raw = np.log(raw + 1) if log_unnormalized else raw
        S = raw

    if 'X_ul' in subset_adata.layers.keys():
        Ul = subset_adata.layers['X_ul'].T
    elif 'ul' in subset_adata.layers.keys():
        raw, raw_ul = subset_adata.layers['ul'].T, subset_adata.layers['ul'].T
        if issparse(raw):
            raw.data = np.log(raw.data + 1) if log_unnormalized else raw.data
        else:
            raw = np.log(raw + 1) if log_unnormalized else raw
        Ul = raw

    if 'X_sl' in subset_adata.layers.keys():
        Sl = subset_adata.layers['X_sl'].T
    elif 'sl' in subset_adata.layers.keys():
        raw, raw_sl = subset_adata.layers['sl'].T, subset_adata.layers['sl'].T
        if issparse(raw):
            raw.data = np.log(raw.data + 1) if log_unnormalized else raw.data
        else:
            raw = np.log(raw + 1) if log_unnormalized else raw
        Sl = raw

    ind_for_proteins = None
    if 'X_protein' in subset_adata.obsm.keys():
        P = subset_adata.obsm['X_protein'].T
    elif 'protein' in subset_adata.obsm.keys():
        P = adata.obsm['protein'].T
    if P is not None:
        has_protein = True
        if protein_names is None:
            warnings.warn(
                'protein layer exists but protein_names is not provided. No estimation will be performed for protein data.')
        else:
            protein_names = list(set(subset_adata.var.index).intersection(protein_names))
            ind_for_proteins = [np.where(subset_adata.var.index == i)[0][0] for i in protein_names]
            subset_adata.var['is_protein_velocity_genes'] = False
            subset_adata.var.loc[ind_for_proteins, 'is_protein_velocity_genes'] = True

    if not has_labeling:
        assumption_mRNA = 'ss'

    if mode is 'deterministic':
        est = estimation(U=U, Ul=Ul, S=S, Sl=Sl, P=P,
                         t=t, ind_for_proteins=ind_for_proteins,
                         experiment_type=experiment_type,
                         assumption_mRNA=assumption_mRNA, assumption_protein=assumption_protein,
                         concat_data=concat_data)
        est.fit()

        alpha, beta, gamma, eta, delta = est.parameters.values()

        U, S = get_U_S_for_velocity_estimation(subset_adata, has_splicing, log_unnormalized)
        vel = velocity(estimation=est)
        vel_U = vel.vel_u(U)
        vel_S = vel.vel_s(U, S)
        vel_P = vel.vel_p(S, P)

        if type(vel_U) is not float:
            adata.layers['velocity_U'] = csr_matrix((adata.shape))
            adata.layers['velocity_U'][:, np.where(valid_ind)[0]] = vel_U.T.tocsr() if issparse(vel_U) else csr_matrix(
                vel_U.T)  # np.where(valid_ind)[0] required for sparse matrix
        if type(vel_S) is not float:
            adata.layers['velocity_S'] = csr_matrix((adata.shape))
            adata.layers['velocity_S'][:, np.where(valid_ind)[0]] = vel_S.T.tocsr() if issparse(vel_S) else csr_matrix(
                vel_S.T)
        if type(vel_P) is not float:
            adata.obsm['velocity_P'] = csr_matrix((adata.obsm['P'].shape[0], len(ind_for_proteins)))
            adata.obsm['velocity_P'] = vel_P.T.tocsr() if issparse(vel_P) else csr_matrix(vel_P.T)

        if experiment_type is 'mix_std_stm':
            if alpha is not None:
                adata.varm['kinetic_parameter_alpha'] = np.zeros((adata.shape[1], alpha[1].shape[1]))
                adata.varm['kinetic_parameter_alpha'] [valid_ind, :] = alpha[1]
                adata.var['kinetic_parameter_alpha'], adata.var['kinetic_parameter_alpha_std'] = None, None
                adata.var.loc[valid_ind, 'kinetic_parameter_alpha'], adata.var.loc[
                    valid_ind, 'kinetic_parameter_alpha_std'] = alpha[1][:, -1], alpha[0]

            adata.var['kinetic_parameter_beta'], adata.var['kinetic_parameter_gamma'], adata.var['RNA_half_life'] = None, None, None

            adata.var.loc[valid_ind, 'kinetic_parameter_beta'] = beta
            adata.var.loc[valid_ind, 'kinetic_parameter_gamma'] = gamma
            adata.var.loc[valid_ind, 'RNA_half_life'] = np.log(2) / gamma
        else:
            if alpha is not None:
                if len(alpha.shape) > 1:  # for each cell
                    adata.varm['kinetic_parameter_alpha'] = np.zeros((alpha.shape)) # adata.shape
                    adata.varm['kinetic_parameter_alpha'] = alpha # [:, valid_ind]
                    adata.var.loc[valid_ind, 'kinetic_parameter_alpha'] = alpha.mean(1)
                elif len(alpha.shape) is 1:
                    adata.var['kinetic_parameter_alpha'] = None
                    adata.var.loc[valid_ind, 'kinetic_parameter_alpha'] = alpha

            adata.var['kinetic_parameter_beta'], adata.var['kinetic_parameter_gamma'], adata.var[
                'RNA_half_life'] = None, None, None
            adata.var.loc[valid_ind, 'kinetic_parameter_beta'] = beta
            adata.var.loc[valid_ind, 'kinetic_parameter_gamma'] = gamma
            adata.var.loc[valid_ind, 'RNA_half_life'] = np.log(2) / gamma

            alpha_intercept, alpha_r2, gamma_intercept, gamma_r2, delta_intercept, delta_r2, uu0, ul0, su0, sl0, U0, S0, total0 = \
                est.aux_param.values()
            if alpha_r2 is not None:
                alpha_r2[~np.isfinite(alpha_r2)] = 0
            adata.var.loc[valid_ind, 'kinetic_parameter_alpha_intercept'] = alpha_intercept
            adata.var.loc[valid_ind, 'kinetic_parameter_alpha_r2'] = alpha_r2

            if gamma_r2 is not None:
                gamma_r2[~np.isfinite(gamma_r2)] = 0
            adata.var.loc[valid_ind, 'kinetic_parameter_gamma_intercept'] = gamma_intercept
            adata.var.loc[valid_ind, 'kinetic_parameter_gamma_r2'] = gamma_r2

            adata.var.loc[valid_ind, 'kinetic_parameter_uu0'] = uu0
            adata.var.loc[valid_ind, 'kinetic_parameter_ul0'] = ul0
            adata.var.loc[valid_ind, 'kinetic_parameter_su0'] = su0
            adata.var.loc[valid_ind, 'kinetic_parameter_sl0'] = sl0

            if ind_for_proteins is not None:
                delta_r2[~np.isfinite(delta_r2)] = 0
                adata.var['kinetic_parameter_eta'], adata.var['kinetic_parameter_delta'], adata.var[
                    'protein_half_life'] = None, None, None
                adata.var.loc[valid_ind, 'kinetic_parameter_eta'][ind_for_proteins] = eta
                adata.var.loc[valid_ind, 'kinetic_parameter_delta'][ind_for_proteins] = delta
                adata.var.loc[valid_ind, 'kinetic_parameter_delta_intercept'][ind_for_proteins] = delta_intercept
                adata.var.loc[valid_ind, 'kinetic_parameter_delta_r2'][ind_for_proteins] = delta_r2
                adata.var.loc[valid_ind, 'protein_half_life'][ind_for_proteins] = np.log(2) / delta
        # add velocity_offset here
    elif mode is 'moment':
        # a few hard code to set up data for moment mode:
        if log_unnormalized and 'X_total' not in subset_adata.layers.keys():
            if issparse(subset_adata.layers['total']):
                subset_adata.layers['new'].data, subset_adata.layers['total'].data = np.log(subset_adata.layers['new'].data + 1), np.log(subset_adata.layers['total'].data + 1)
            else:
                subset_adata.layers['total'], subset_adata.layers['total'] = np.log(subset_adata.layers['new'] + 1), np.log(subset_adata.layers['total'] + 1)

        Moment = MomData(subset_adata, tkey)
        adata.uns['M'], adata.uns['V'] = Moment.M, Moment.V
        Est = Estimation(Moment, time_key=tkey, normalize=True) #  # data is already normalized
        params, costs = Est.fit()
        a, b, alpha_a, alpha_i, beta, gamma = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:,
                                                                                                      4], params[:, 5]
        def fbar(x_a, x_i, a, b):
            return b / (a + b) * x_a + a / (a + b) * x_i

        alpha = fbar(alpha_a, alpha_i, a, b)[:, None]  ### dimension need to be matched up

        params = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 't': t}
        vel = velocity(**params)

        U, S = get_U_S_for_velocity_estimation(subset_adata, has_splicing, log_unnormalized)
        vel_U = vel.vel_u(U)
        vel_S = vel.vel_s(U, S)
        vel_P = vel.vel_p(S, P)

        if type(vel_U) is not float:
            adata.layers['velocity_U'] = csr_matrix((adata.shape))
            adata.layers['velocity_U'][:, np.where(valid_ind)[0]] = vel_U.T.tocsr() if issparse(vel_U) else csr_matrix(
                vel_U.T)
        if type(vel_S) is not float:
            adata.layers['velocity_S'] = csr_matrix((adata.shape))
            adata.layers['velocity_S'][:, np.where(valid_ind)[0]] = vel_S.T.tocsr() if issparse(vel_S) else csr_matrix(
                vel_S.T)
        if type(vel_P) is not float and ind_for_proteins is not None:
            adata.obsm['velocity_P'] = csr_matrix((adata.obsm['P'].shape[0], len(ind_for_proteins)))
            adata.obsm['velocity_P'] = vel_P.T.tocsr() if issparse(vel_P) else csr_matrix(vel_P.T)

        adata.var['kinetic_parameter_a'], adata.var['kinetic_parameter_b'], adata.var['kinetic_parameter_alpha_a'], \
        adata.var['kinetic_parameter_alpha_i'], adata.var['kinetic_parameter_beta'], adata.var['protein_half_life'], \
        adata.var['kinetic_parameter_gamma'], adata.var[
            'RNA_half_life'] = None, None, None, None, None, None, None, None

        adata.var.loc[valid_ind, 'kinetic_parameter_a'] = a
        adata.var.loc[valid_ind, 'kinetic_parameter_b'] = b
        adata.var.loc[valid_ind, 'kinetic_parameter_alpha_a'] = alpha_a
        adata.var.loc[valid_ind, 'kinetic_parameter_alpha_i'] = alpha_i
        adata.var.loc[valid_ind, 'kinetic_parameter_beta'] = beta
        adata.var.loc[valid_ind, 'kinetic_parameter_gamma'] = gamma
        adata.var.loc[valid_ind, 'RNA_half_life'] = np.log(2) / gamma
        # add protein related parameters in the moment model below:
    elif mode is 'model_selection':
        warnings.warn('Not implemented yet.')

    adata.uns['dynamics'] = {'t': t, 'asspt_mRNA': assumption_mRNA, 'experiment_type': experiment_type, "normalized": normalized, "mode": mode, "has_splicing": has_splicing,
                             "has_labeling": has_labeling, "has_protein": has_protein}
    return adata
