import warnings
import numpy as np
from scipy.sparse import issparse, csr_matrix
from .velocity import velocity, estimation
from .moments import MomData, Estimation
from .utils import get_U_S_for_velocity_estimation

# incorporate the model selection code soon
def dynamics(adata, filter_gene_mode='final', mode='deterministic', tkey='Time', group=None, protein_names=None,
             experiment_type=None, assumption_mRNA=None, assumption_protein='ss', NTR_vel=True, concat_data=False,
             log_unnormalized=True):
    """Inclusive model of expression dynamics considers splicing, metabolic labeling and protein translation. It supports
    learning high-dimensional velocity vector samples for droplet based (10x, inDrop, drop-seq, etc), scSLAM-seq, NASC-seq
    sci-fate, scNT-seq or cite-seq datasets.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object.
        filter_gene_mode: `str` (default: `final`)
            The string for indicating which mode (one of, ['final', 'basic', 'no']) of gene filter will be used.
        mode: `str` (default: `deterministic`)
            string indicates which estimation mode will be used. Currently "deterministic" and "moment" methods are supported.
            A "model_selection" mode will be supported soon in which alpha, beta and gamma will be modeled as a function of time.
        tkey: `str` (default: `Time`)
            The column key for the time label of cells in .obs. Used for either "steady_state" or non-"steady_state" mode or `moment`
            mode  with labeled data.
        group: `str` or None (default: `None`)
            The column key/name that identifies the grouping information (for example, clusters that correspond to different cell types)
            of cells. This will be used to estimate group-specific (i.e cell-type specific) kinetic parameters.
        protein_names: `List`
            A list of gene names corresponds to the rows of the measured proteins in the `X_protein` of the `obsm` attribute.
            The names have to be included in the adata.var.index.
        experiment_type: `str`
            labelling experiment type. Available options are:
            (1) 'deg': degradation experiment;
            (2) 'kin': synthesis/kinetics experiment;
            (3) 'one-shot': one-shot kinetic experiment.
        assumption_mRNA: `str`
            Parameter estimation assumption for mRNA. Available options are:
            (1) 'ss': pseudo steady state;
            (2) None: kinetic data with no assumption.
            If no labelling data exists, assumption_mRNA will automatically set to be 'ss'.
        assumption_protein: `str`
            Parameter estimation assumption for protein. Available options are:
            (1) 'ss': pseudo steady state;
        NTR_vel: `bool` (default: `True`)
            Whether to use NTR (new/total ratio) velocity.
        concat_data: `bool` (default: `False`)
            Whether to concatenate data before estimation. If your data is a list of matrices for each time point, this need to be set as True.
        log_unnormalized: `bool` (default: `True`)
            Whether to log transform the unnormalized data.

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

    valid_adata = adata[:, valid_ind].copy()
    if group is not None and group in adata.obs[group]:
        _group = adata.obs[group].unique()
    else:
        _group = ["_all_cells"]

    for cur_grp in _group:
        if cur_grp == '_all_cells':
            kin_param_pre = ''
            cur_cells_bools = np.ones(valid_adata.shape[0], dtype=bool)
            subset_adata = valid_adata[cur_cells_bools]
        else:
            kin_param_pre = group + '_' + cur_grp + '_'
            cur_cells_bools = (valid_adata.obs[group] == cur_grp).values
            subset_adata = valid_adata[cur_cells_bools]

        U, Ul, S, Sl, P = None, None, None, None, None  # U: unlabeled unspliced; S: unlabel spliced: S
        normalized, has_splicing, has_labeling, has_protein = False, False, False, False

        if 'X_unspliced' in subset_adata.layers.keys():
            has_splicing, normalized, assumption_mRNA = True, True, 'ss'
            U = subset_adata.layers['X_unspliced'].T
        elif 'unspliced' in subset_adata.layers.keys():
            has_splicing, assumption_mRNA = True, 'ss'
            raw, row_unspliced = subset_adata.layers['unspliced'].T, subset_adata.layers['unspliced'].T
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

        elif 'X_new' in subset_adata.layers.keys():  # run new / total ratio (NTR)
            has_labeling, normalized, assumption_mRNA = True, True, 'ss'
            U = subset_adata.layers['X_total'].T - subset_adata.layers['X_new'].T
            Ul = subset_adata.layers['X_new'].T
        elif 'new' in subset_adata.layers.keys():
            has_labeling, assumption_mRNA = True, 'ss'
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
            has_splicing, has_labeling, normalized = True, True, False
            raw, raw_uu = subset_adata.layers['uu'].T, subset_adata.layers['uu'].T
            if issparse(raw):
                raw.data = np.log(raw.data + 1) if log_unnormalized else raw.data
            else:
                raw = np.log(raw + 1) if log_unnormalized else raw
            U = raw
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
        if 'X_su' in subset_adata.layers.keys():  # unlabel spliced: S
            S = subset_adata.layers['X_su'].T
        elif 'su' in subset_adata.layers.keys():
            raw, raw_su = subset_adata.layers['su'].T, subset_adata.layers['su'].T
            if issparse(raw):
                raw.data = np.log(raw.data + 1) if log_unnormalized else raw.data
            else:
                raw = np.log(raw + 1) if log_unnormalized else raw
            S = raw

        ind_for_proteins = None
        if 'X_protein' in subset_adata.obsm.keys():
            P = subset_adata.obsm['X_protein'].T
        elif 'protein' in subset_adata.obsm.keys():
            P = subset_adata.obsm['protein'].T
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

        if experiment_type is not None or mode is 'moment':
            assumption_mRNA = None

        if has_labeling:
            if tkey in adata.obs.columns:
                t = np.array(adata.obs[tkey], dtype='float')
            else:
                raise Exception('the tkey ', tkey, ' provided is not a valid column name in .obs.')
        else:
            t = None

        if mode is 'deterministic':
            est = estimation(U=U, Ul=Ul, S=S, Sl=Sl, P=P,
                             t=t, ind_for_proteins=ind_for_proteins,
                             experiment_type=experiment_type,
                             assumption_mRNA=assumption_mRNA, assumption_protein=assumption_protein,
                             concat_data=concat_data)
            est.fit()

            alpha, beta, gamma, eta, delta = est.parameters.values()

            U, S = get_U_S_for_velocity_estimation(subset_adata, has_splicing, has_labeling, log_unnormalized, NTR_vel)
            vel = velocity(estimation=est)
            vel_U = vel.vel_u(U)
            vel_S = vel.vel_s(U, S)
            vel_P = vel.vel_p(S, P)

            if type(vel_U) is not float:
                if cur_grp == _group[0]: adata.layers['velocity_U'] = csr_matrix((adata.shape))
                adata.layers['velocity_U'][cur_cells_bools,:][:, valid_ind] = vel_U.T.tocsr() if issparse(vel_U) else csr_matrix(
                    vel_U.T)  # np.where(valid_ind)[0] required for sparse matrix
            if type(vel_S) is not float:
                if cur_grp == _group[0]: adata.layers['velocity_S'] = csr_matrix((adata.shape))
                adata.layers['velocity_S'][cur_cells_bools,:][:, valid_ind] = vel_S.T.tocsr() if issparse(vel_S) else csr_matrix(
                    vel_S.T)
            if type(vel_P) is not float:
                if cur_grp == _group[0]: adata.obsm['velocity_P'] = csr_matrix((adata.obsm['P'].shape[0], len(ind_for_proteins)))
                adata.obsm['velocity_P'][cur_cells_bools, :] = vel_P.T.tocsr() if issparse(vel_P) else csr_matrix(vel_P.T)

            if experiment_type is 'mix_std_stm':
                if alpha is not None:
                    if cur_grp == _group[0]: adata.varm[kin_param_pre + 'alpha'] = np.zeros((adata.shape[1], alpha[1].shape[1]))
                    adata.varm[kin_param_pre + 'alpha'][valid_ind, :] = alpha[1]
                    adata.var[kin_param_pre + 'alpha'], adata.var[kin_param_pre + 'alpha_std'] = None, None
                    adata.var.loc[valid_ind, kin_param_pre + 'alpha'], adata.var.loc[
                        valid_ind, kin_param_pre + 'alpha_std'] = alpha[1][:, -1], alpha[0]

                if cur_grp == _group[0]: adata.var[kin_param_pre + 'beta'], adata.var[kin_param_pre + 'gamma'], adata.var[kin_param_pre + 'half_life'] = None, None, None

                adata.var.loc[valid_ind, kin_param_pre + 'beta'] = beta
                adata.var.loc[valid_ind, kin_param_pre + 'gamma'] = gamma
                adata.var.loc[valid_ind, kin_param_pre + 'half_life'] = np.log(2) / gamma
            else:
                if alpha is not None:
                    if len(alpha.shape) > 1:  # for each cell
                        if cur_grp == _group[0]: adata.varm[kin_param_pre + 'alpha'] = np.zeros((alpha.shape)) # adata.shape
                        adata.varm[kin_param_pre + 'alpha'] = alpha # [:, valid_ind]
                        adata.var.loc[valid_ind, kin_param_pre + 'alpha'] = alpha.mean(1)
                    elif len(alpha.shape) is 1:
                        if cur_grp == _group[0]: adata.var[kin_param_pre + 'alpha'] = None
                        adata.var.loc[valid_ind, kin_param_pre + 'alpha'] = alpha

                if cur_grp == _group[0]: adata.var[kin_param_pre + 'beta'], adata.var[kin_param_pre + 'gamma'], adata.var[
                    kin_param_pre + 'half_life'] = None, None, None
                adata.var.loc[valid_ind, kin_param_pre + 'beta'] = beta
                adata.var.loc[valid_ind, kin_param_pre + 'gamma'] = gamma
                adata.var.loc[valid_ind, kin_param_pre + 'half_life'] = np.log(2) / gamma

                alpha_intercept, alpha_r2, gamma_intercept, gamma_r2, delta_intercept, delta_r2, uu0, ul0, su0, sl0, U0, S0, total0 = \
                    est.aux_param.values()
                if alpha_r2 is not None:
                    alpha_r2[~np.isfinite(alpha_r2)] = 0
                if cur_grp == _group[0]: adata.var[kin_param_pre + 'alpha_b'], adata.var[kin_param_pre + 'alpha_r2'], \
                                         adata.var[kin_param_pre + 'gamma_b'], adata.var[kin_param_pre + 'gamma_r2'], \
                                         adata.var[kin_param_pre + 'delta_b'], adata.var[kin_param_pre + 'delta_r2'], \
                                         adata.var[kin_param_pre + 'uu0'], adata.var[kin_param_pre + 'ul0'], \
                                         adata.var[kin_param_pre + 'su0'], adata.var[kin_param_pre + 'sl0'], \
                                         adata.var[kin_param_pre + 'U0'], adata.var[kin_param_pre + 'S0'], \
                                         adata.var[kin_param_pre + 'total0'] = None, None, None, None, None, None, None, None, None, None, None, None, None

                adata.var.loc[valid_ind, kin_param_pre + 'alpha_b'] = alpha_intercept
                adata.var.loc[valid_ind, kin_param_pre + 'alpha_r2'] = alpha_r2

                if gamma_r2 is not None:
                    gamma_r2[~np.isfinite(gamma_r2)] = 0
                adata.var.loc[valid_ind, kin_param_pre + 'gamma_b'] = gamma_intercept
                adata.var.loc[valid_ind, kin_param_pre + 'gamma_r2'] = gamma_r2

                adata.var.loc[valid_ind, kin_param_pre + 'uu0'] = uu0
                adata.var.loc[valid_ind, kin_param_pre + 'ul0'] = ul0
                adata.var.loc[valid_ind, kin_param_pre + 'su0'] = su0
                adata.var.loc[valid_ind, kin_param_pre + 'sl0'] = sl0
                adata.var.loc[valid_ind, kin_param_pre + 'U0'] = U0
                adata.var.loc[valid_ind, kin_param_pre + 'S0'] = S0
                adata.var.loc[valid_ind, kin_param_pre + 'total0'] = total0

                if ind_for_proteins is not None:
                    delta_r2[~np.isfinite(delta_r2)] = 0
                    if cur_grp == _group[0]: adata.var[kin_param_pre + 'eta'], adata.var[kin_param_pre + 'delta'], \
                                             adata.var[kin_param_pre + 'delta_b'], adata.var[kin_param_pre + 'delta_r2'], \
                                             adata.var[kin_param_pre + 'p_half_life'] = None, None, None, None, None
                    adata.var.loc[valid_ind, kin_param_pre + 'eta'][ind_for_proteins] = eta
                    adata.var.loc[valid_ind, kin_param_pre + 'delta'][ind_for_proteins] = delta
                    adata.var.loc[valid_ind, kin_param_pre + 'delta_b'][ind_for_proteins] = delta_intercept
                    adata.var.loc[valid_ind, kin_param_pre + 'delta_r2'][ind_for_proteins] = delta_r2
                    adata.var.loc[valid_ind, kin_param_pre + 'p_half_life'][ind_for_proteins] = np.log(2) / delta

        elif mode is 'moment':
            # a few hard code to set up data for moment mode:
            if 'uu' in subset_adata.layers.keys() or 'X_uu' in subset_adata.layers.keys():
                if log_unnormalized and 'X_uu' not in subset_adata.layers.keys():
                    if issparse(subset_adata.layers['uu']):
                        subset_adata.layers['uu'].data, subset_adata.layers['ul'].data, subset_adata.layers['su'].data, subset_adata.layers['sl'].data = \
                            np.log(subset_adata.layers['uu'].data + 1), np.log(subset_adata.layers['ul'].data + 1), np.log(subset_adata.layers['su'].data + 1), np.log(subset_adata.layers['sl'].data + 1)
                    else:
                        subset_adata.layers['uu'], subset_adata.layers['ul'], subset_adata.layers['su'], subset_adata.layers['sl'] = \
                            np.log(subset_adata.layers['uu'] + 1), np.log(subset_adata.layers['ul'] + 1), np.log(subset_adata.layers['su'] + 1), np.log(subset_adata.layers['sl'] + 1)

                subset_adata_u, subset_adata_s = subset_adata.copy(), subset_adata.copy()
                del subset_adata_u.layers['su'], subset_adata_u.layers['sl'], subset_adata_s.layers['uu'], subset_adata_s.layers['ul']
                subset_adata_u.layers['new'], subset_adata_u.layers['old'], subset_adata_s.layers['new'], subset_adata_s.layers['old'] = \
                    subset_adata_u.layers.pop('ul'), subset_adata_u.layers.pop('uu'), subset_adata_s.layers.pop('sl'), subset_adata_s.layers.pop('su')
                Moment, Moment_ = MomData(subset_adata_s, tkey), MomData(subset_adata_u, tkey)
                if cur_grp == _group[0]:
                    t_ind = 0
                    g_len, t_len = len(_group), len(np.unique(adata.obs[tkey]))
                    adata.uns['M_sl'], adata.uns['V_sl'], adata.uns['M_ul'], adata.uns['V_ul'] = \
                    np.zeros((adata.shape[1], g_len * t_len)), np.zeros((adata.shape[1], g_len * t_len)), np.zeros((adata.shape[1], g_len * t_len)), np.zeros((adata.shape[1], g_len * t_len))

                adata.uns['M_sl'][:, (t_len * t_ind):(t_len * (t_ind + 1))], adata.uns['V_sl'][:, (t_len * t_ind):(t_len * (t_ind + 1))], adata.uns['M_ul'][:, (t_len * t_ind):(t_len * (t_ind + 1))], \
                adata.uns['V_ul'][:, (t_len * t_ind):(t_len * (t_ind + 1))] = Moment.M, Moment.V, Moment_.M, Moment_.V
                del Moment_
                Est = Estimation(Moment, adata_u=subset_adata_u, time_key=tkey, normalize=True) #  # data is already normalized
            else:
                if log_unnormalized and 'X_total' not in subset_adata.layers.keys():
                    if issparse(subset_adata.layers['total']):
                        subset_adata.layers['new'].data, subset_adata.layers['total'].data = np.log(subset_adata.layers['new'].data + 1), np.log(subset_adata.layers['total'].data + 1)
                    else:
                        subset_adata.layers['total'], subset_adata.layers['total'] = np.log(subset_adata.layers['new'] + 1), np.log(subset_adata.layers['total'] + 1)

                Moment = MomData(subset_adata, tkey)
                if cur_grp == _group[0]:
                    t_ind = 0
                    g_len, t_len = len(_group), len(np.unique(adata.obs[tkey]))
                    adata.uns['M'], adata.uns['V'] = np.zeros((adata.shape[1], g_len * t_len)), np.zeros((adata.shape[1],  g_len * t_len))

                adata.uns['M'][:, (t_len * t_ind):(t_len * (t_ind + 1))], adata.uns['V'][:, (t_len * t_ind):(t_len * (t_ind + 1))] = Moment.M, Moment.V
                Est = Estimation(Moment, time_key=tkey, normalize=True) #  # data is already normalized

            t_ind += 1

            params, costs = Est.fit()
            a, b, alpha_a, alpha_i, beta, gamma = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:,
                                                                                                          4], params[:, 5]
            def fbar(x_a, x_i, a, b):
                return b / (a + b) * x_a + a / (a + b) * x_i

            alpha = fbar(alpha_a, alpha_i, a, b)[:, None]

            params = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 't': t}
            vel = velocity(**params)

            U, S = get_U_S_for_velocity_estimation(subset_adata, has_splicing, has_labeling, log_unnormalized, NTR_vel)
            vel_U = vel.vel_u(U)
            vel_S = vel.vel_s(U, S)
            vel_P = vel.vel_p(S, P)

            if type(vel_U) is not float:
                if cur_grp == _group[0]: adata.layers['velocity_U'] = csr_matrix((adata.shape))
                adata.layers['velocity_U'][cur_cells_bools,:][:, valid_ind] = vel_U.T.tocsr() if issparse(vel_U) else csr_matrix(
                    vel_U.T)
            if type(vel_S) is not float:
                if cur_grp == _group[0]: adata.layers['velocity_S'] = csr_matrix((adata.shape))
                adata.layers['velocity_S'][cur_cells_bools,:][:, valid_ind] = vel_S.T.tocsr() if issparse(vel_S) else csr_matrix(
                    vel_S.T)
            if type(vel_P) is not float and ind_for_proteins is not None:
                if cur_grp == _group[0]: adata.obsm['velocity_P'] = csr_matrix((adata.obsm['P'].shape[0], len(ind_for_proteins)))
                adata.obsm['velocity_P'][cur_cells_bools, :] = vel_P.T.tocsr() if issparse(vel_P) else csr_matrix(vel_P.T)

            if cur_grp == _group[0]: adata.var[kin_param_pre + 'a'], adata.var[kin_param_pre + 'b'], adata.var[kin_param_pre + 'alpha_a'], \
            adata.var[kin_param_pre + 'alpha_i'], adata.var[kin_param_pre + 'beta'], adata.var[kin_param_pre + 'p_half_life'], \
            adata.var[kin_param_pre + 'gamma'], adata.var[
                kin_param_pre + 'half_life'] = None, None, None, None, None, None, None, None

            adata.var.loc[valid_ind, kin_param_pre + 'a'] = a
            adata.var.loc[valid_ind, kin_param_pre + 'b'] = b
            adata.var.loc[valid_ind, kin_param_pre + 'alpha_a'] = alpha_a
            adata.var.loc[valid_ind, kin_param_pre + 'alpha_i'] = alpha_i
            adata.var.loc[valid_ind, kin_param_pre + 'beta'] = beta
            adata.var.loc[valid_ind, kin_param_pre + 'gamma'] = gamma
            adata.var.loc[valid_ind, kin_param_pre + 'half_life'] = np.log(2) / gamma
            # add protein related parameters in the moment model below:
        elif mode is 'model_selection':
            warnings.warn('Not implemented yet.')

    if group is not None and group in adata.obs[group]:
        uns_key = group + "_dynamics"
    else:
        uns_key = "dynamics"

    adata.uns[uns_key] = {'t': t, "group": group, 'asspt_mRNA': assumption_mRNA, 'experiment_type': experiment_type, "normalized": normalized, "mode": mode, "has_splicing": has_splicing,
                             "has_labeling": has_labeling, "has_protein": has_protein}
    return adata
