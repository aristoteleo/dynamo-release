from .velocity import velocity, estimation
from .moments import MomData, Estimation
import warnings
import numpy as np


# add the moment code in; and incorporate the model selection code later
def dynamics(adata, mode='steady_state', protein_names=None, experiment_type='deg', assumption_mRNA=None, assumption_protein='ss', concat_data=False):
    """Inclusive model of expression dynamics with scSLAM-seq and multiomics.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        mode: `str` (default: steady_state)
            mode indicates which estimation framework will be used. Currently "steady_state" and "moment" methods are supported.
            A "model_selection" mode will be supported soon in which beta and gamma can be variable over time also.
        protein_names: `List`
            A list of gene names corresponds to the rows of the measured proteins in the P layer. The names have to be included
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

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object
    """

    U, Ul, S, Sl, P = None, None, None, None, None

    if 'X_unspliced' in adata.layers.keys():
        U = adata.layers['X_unspliced'].T
    elif 'unspliced' in adata.layers.keys():
        U = adata.layers['unspliced'].T
    elif 'X_new' in adata.layers.keys(): # run new / total ratio (NTR)
        U = adata.layers['X_new'].T
    elif 'new' in adata.layers.keys():
        U = adata.layers['new'].T
    elif 'X_uu' in adata.layers.keys():  # only uu, ul, su, sl provided
        U = adata.layers['X_uu'].T
    elif 'uu' in adata.layers.keys():
        U = adata.layers['uu'].T

    if 'X_spliced' in adata.layers.keys():
        S = adata.layers['X_spliced'].T
    elif 'spliced' in adata.layers.keys():
        S = adata.layers['spliced'].T
    elif 'X_total' in adata.layers.keys(): # run new / total ratio (NTR)
        U = adata.layers['X_total'].T
    elif 'total' in adata.layers.keys():
        U = adata.layers['total'].T
    elif 'X_su' in adata.layers.keys():
        U = adata.layers['X_su'].T
    elif 'su' in adata.layers.keys():
        U = adata.layers['su'].T

    if 'X_ul' in adata.layers.keys():
        Ul = adata.layers['X_ul'].T
    elif 'ul' in adata.layers.keys():
        Ul = adata.layers['ul'].T

    if 'X_sl' in adata.layers.keys():
        Sl = adata.layers['X_sl'].T
    elif 'sl' in adata.layers.keys():
        Sl = adata.layers['sl'].T

    ind_for_proteins = None
    if 'X_protein' in adata.obsm.keys():
        P = adata.obsm['X_protein'].T
    elif 'protein' in adata.obsm.keys():
        P = adata.obsm['protein'].T
    if P is not None:
        if protein_names is None:
            warnings.warn('protein layer exists but protein_names is not provided. No estimation will be performed for protein data.')
        else:
            ind_for_proteins = [np.where(adata.var.index == i)[0][0] for i in protein_names]

    t = adata.obs.Time if 'Time' in adata.obs.columns else None

    if Ul is None or Sl is None:
        assumption_mRNA = 'ss'

    if mode is 'steady_state':
        est = estimation(U=U, Ul=Ul, S=S, Sl=Sl, P=P, t=t, ind_for_proteins=ind_for_proteins, experiment_type=experiment_type, assumption_mRNA=assumption_mRNA, \
                         assumption_protein=assumption_protein, concat_data=concat_data)
        est.fit()

        alpha, beta, gamma, eta, delta = est.parameters.values()
        # do this for a vector?
        vel = velocity(** est.parameters)
        vel_U = vel.vel_u(U)
        vel_S = vel.vel_s(U, S)
        vel_P = vel.vel_p(S, P)

        if type(vel_U) is not float:
            adata.layers['velocity_U'] = vel_U.T
        if type(vel_S) is not float:
            adata.layers['velocity_S'] = vel_S.T
        if type(vel_P) is not float:
            adata.obsm['velocity_P'] = vel_P.T

        if alpha is not None: # for each cell
            adata.varm['velocity_parameter_alpha'] = alpha

        adata.var['velocity_parameter_avg_alpha'] = alpha.mean(1) if alpha is not None else None
        adata.var['velocity_parameter_beta'] = beta
        adata.var['velocity_parameter_gamma'] = gamma
        if ind_for_proteins is not None:
            adata.var['velocity_parameter_eta'][ind_for_proteins] = eta
            adata.var['velocity_parameter_delta'][ind_for_proteins] = delta
        # add velocity_offset here
    elif mode is 'moment':
        Moment = MomData(adata)
        Est = Estimation(Moment)
        params, costs = Est.fit()
        a, b, alpha_a, alpha_i, beta, gamma = params

        def fbar(x_a, x_i, a, b):
            return b / (a + b) * x_a + a / (a + b) * x_i
        alpha = fbar(alpha_a, alpha_i, a, b)[:, None] ### dimension need to be matched up

        params = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
        vel = velocity(**params)
        vel_U = vel.vel_u(U)
        vel_S = vel.vel_s(U, S)
        vel_P = vel.vel_p(S, P)

        if type(vel_U) is not float:
            adata.layers['velocity_U'] = vel_U.T
        if type(vel_S) is not float:
            adata.layers['velocity_S'] = vel_S.T
        if type(vel_P) is not float:
            adata.obsm['velocity_P'] = vel_P.T

        adata.var['velocity_parameter_a'] = a
        adata.var['velocity_parameter_b'] = b
        adata.var['velocity_parameter_alpha_a'] = alpha_a
        adata.var['velocity_parameter_alpha_i'] = alpha_i
        adata.var['velocity_parameter_beta'] = beta
        adata.var['velocity_parameter_gamma'] = gamma
        # add velocity_offset here
    elif mode is 'model_selection':
        warnings.warn('Not implemented yet.')

    return adata

