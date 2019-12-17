import numpy as np
from scipy.sparse import issparse
from .scatters import scatters
from ..tools.velocity import sol_u, sol_s
from ..simulation.Gillespie import simulate_Gillespie, temporal_average

def metabolic_labeling_fit(adata, vkey, tkey, experiment_type='deg', unit='hours', log=True, group=None, ncols=None, figsize=None, dpi=None, show=True):
    """ Plot the data and fitting of different metabolic labeling experiments.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        vkey: list of `str`
            key for variable or gene names.
        tkey: `str`
            The key for the time annotation in .obs attribute.
        experiment_type: `str` (default: `deg`)
            Experimental type. The type of experiments under study, can be either `deg`, `kin`, `one_shot`, `coassay`.
        unit: `str` (default: `hour`)
            The unit of the labeling time, for example, `hours` or `minutes`.
        log: `bool` (default: `True`)
            Whether or not to log transform the expression value before plotting .
        group: `str` or None (default: `None`)
            The key for the group annotation in .obs attribute.
        ncols: `int` or None (default: `None`)
            The number of columns in the plot.
        figsize: `[float, float]` or `(float, float)` or None
            The size of figure.
        dpi: `float` or None
            Figure resolution.
        show: `bool` (default: `True`)
            Whether to plot the figure.

    Returns
    -------
        Nothing but plot the figure of the metabolic fitting.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    groups = [''] if group is None else np.unique(adata.obs[group])

    T = adata.obs[tkey]
    gene_idx = [np.where(adata.var.index.values == gene)[0][0] for gene in vkey]
    prefix = 'kinetic_parameter_'

    sub_plot_n = 2 if experiment_type is 'deg' else 4 if experiment_type is 'kin' else 8 if experiment_type is 'moment' else 1
    ncols = len(gene_idx) if ncols is None else min(len(gene_idx), ncols)
    nrows = int(np.ceil(len(gene_idx) * sub_plot_n / ncols))
    figsize = [7, 5] if figsize is None else figsize
    gs = plt.GridSpec(nrows, ncols, plt.figure(None, (figsize[0] * ncols, figsize[1] * nrows), dpi=dpi))

    for i, idx in enumerate(gene_idx):
        gene_name = adata.var_names[idx]

        Tsort = np.argsort(T.unique())
        t = np.linspace(Tsort[0], Tsort[-1], 50)

        if experiment_type is 'deg':
            new, old = adata[:, gene_name].layers['new'], adata[:, gene_name].layers['total'] - \
                       adata[:, gene_name].layers['new']
            u, l = (old.toarray().squeeze(), new.toarray().squeeze()) if issparse(old) else (old.squeeze(), new.squeeze())

            if log:
                u, l = np.log(u + 1), np.log(l + 1)
            alpha, gamma, u0, l0 = adata.var.loc[gene_name, [prefix + 'alpha', prefix + 'gamma', prefix + 'uu0', prefix + 'ul0']]
            if u0 is None or np.isnan(u0):
                u0 = np.mean(u[T == np.min(T)])
            # [adata.obs[group] == cur_grp, gene_idx]

            # beta, gamma, alpha
            row_ind = int(np.floor(idx/ncols)) # make sure unlabled and labeled are in the same column.
            ax1 = plt.subplot(gs[row_ind * ncols * sub_plot_n + idx % ncols])
            ax1 = sns.violinplot(x=T.values, y=u.T, ax=ax1) #ax1.plot(T, u.T, linestyle='None', marker='o', markersize=10)
            ax1.plot(t, alpha/gamma + (u0 - alpha/gamma) * np.exp(-gamma*t), 'k--')
            ax1.set_xlabel('time (' + unit + ')')
            ax1.set_ylabel('Expression')
            ax1.set_title(gene_name + '(unlabeled)')

            ax2 = plt.subplot(gs[(1 + sub_plot_n * row_ind) * ncols + idx % ncols])
            ax2 = sns.violinplot(x=T.values, y=l.T, ax=ax2) # ax2.plot(T, l.T, linestyle='None', marker='o', markersize=10)
            ax2.plot(t, l0 * np.exp(-gamma*t), 'k--')
            ax2.set_xlabel('time (' + unit + ')')
            ax2.set_ylabel('Expression')
            ax2.set_title(gene_name + '(labeled)')
        elif experiment_type is 'kin':
            uu, ul, su, sl = adata[:, gene_name].layers['uu'], adata[:, gene_name].layers['ul'], \
                       adata[:, gene_name].layers['su'], adata[:, gene_name].layers['sl']
            uu, ul, su, sl = (uu.toarray().squeeze(), ul.toarray().squeeze(), su.toarray().squeeze(), sl.toarray().squeeze()) \
                if issparse(old) else (uu.squeeze(), ul.squeeze(), su.squeeze(), sl.squeeze())

            alpha, gamma, uu0, su0 = adata.var.loc[gene_name, [prefix + 'alpha', prefix + 'gamma', prefix + 'uu0', prefix + 'su0']]
            beta = 1
            u = sol_u(t, uu0, 0, beta)
            s = sol_s(t, su0, uu0, 0, beta, gamma)
            w = sol_u(t, 0, alpha, beta)
            l = sol_s(t, 0, 0, alpha, beta, gamma)

            Obs, Pred = np.vstack((uu, ul, su, sl)), np.vstack((u, w, s, l))
            title_ = ['(unspliced unlabeled)', '(unspliced labeled)', '(spliced unlabeled)', '(spliced labeled)']

            for j in range(4):
                row_ind = int(np.floor(idx/ncols)) # make sure unlabled and labeled are in the same column.
                ax = plt.subplot(gs[(row_ind * sub_plot_n + i) * ncols + idx % ncols])
                ax = sns.violinplot(x=T.values, y=Obs[i], ax=ax) #ax1.plot(T, u.T, linestyle='None', marker='o', markersize=10)
                ax.plot(t, u, 'k--')
                ax.set_xlabel('time (' + unit + ')')
                ax.set_ylabel('Expression')
                ax.set_title(gene_name + title_[j])

        elif experiment_type is 'one_shot':
            pass
        elif experiment_type is 'multi_time_series':
            pass
        elif experiment_type is 'moment':
            a, b, la, alpha_a, alpha_i, sigma, beta, gamma = adata.var.loc[gene_name, \
             [prefix + 'a', prefix + 'b', prefix + 'la', prefix + 'alpha_a', prefix + 'alpha_i', prefix + 'sigma', prefix + 'beta', prefix + 'gamma']]
            params = {'a': a, 'b': b, 'la': la, 'alpha_a': alpha_a, 'alpha_i': alpha_i, 'sigma': sigma, 'beta': beta, 'gamma': gamma}
            trajs_T, trajs_C = simulate_Gillespie(*list(params.values()), C0=np.zeros(5), t_span=[Tsort[0], Tsort[-1]], n_traj=50, report=True)
            n_species = 5
            n_mean = np.zeros((n_species, len(t)))
            n_2mom = np.zeros((n_species, len(t)))
            for j in range(n_species):
                n_mean[j] = temporal_average(T, trajs_T, trajs_C, j)
                n_2mom[j] = temporal_average(T, trajs_T, trajs_C, j, lambda x: x * (x - 1))
            n_var = n_2mom + n_mean - n_mean ** 2
            title_ = ['(unspliced unlabeled)', '(unspliced labeled)', '(spliced unlabeled)', '(spliced labeled)',
                      '(unspliced unlabeled)', '(unspliced labeled)', '(spliced unlabeled)', '(spliced labeled)']

            Obs_m = adata.uns['M_uu'], adata.uns['M_uu'], adata.uns['M_uu'], adata.uns['M_uu']
            Obs_v = adata.uns['V_uu'], adata.uns['V_uu'], adata.uns['V_uu'], adata.uns['V_uu']
            for j in range(8):
                row_ind = int(np.floor(idx/ncols)) # make sure unlabled and labeled are in the same column.
                ax = plt.subplot(gs[(row_ind * sub_plot_n + i) * ncols + idx % ncols])
                if j < 4:
                    ax = sns.violinplot(x=T.values, y=Obs_m[i], ax=ax)  # ax1.plot(T, u.T, linestyle='None', marker='o', markersize=10)
                    ax.set_ylabel('Expression')
                    ax.plot(t, n_mean[j], 'k--')
                else:
                    ax = sns.violinplot(x=T.values, y=Obs_v[i - j], ax=ax)  # ax1.plot(T, u.T, linestyle='None', marker='o', markersize=10)
                    ax.set_ylabel('Variance')
                    ax.plot(t, n_var[j - 4], 'k--')
                ax.set_xlabel('time (' + unit + ')')
                ax.set_title(gene_name + title_[j])

        elif experiment_type is 'coassay':
            pass

    if show: plt.show()


def dynamics(adata, gene_names, color, dims=[0, 1], current_layer='spliced', use_raw=False, Vkey='S', Ekey='spliced',
             basis='umap', mode='all', cmap=None, gs=None, **kwargs):
    """

    Parameters
    ----------
    adata
    basis
    mode: `str` (default: all)
        Support mode includes: phase, expression, velocity, all

    Returns
    -------

    """

    import matplotlib.pyplot as plt
    genes = list(set(gene_names).intersection(adata.var.index))
    for i, gn in enumerate(genes):
        ax = plt.subplot(gs[i * 3])
        try:
            ix = np.where(adata.var["Gene"] == gn)[0][0]
        except:
            continue

        scatters(adata, gene_names, color, dims=[0, 1], current_layer='spliced', use_raw=False, Vkey='S', Ekey='spliced',
             basis='umap', mode='all', cmap=None, gs=None, **kwargs)

        scatters(adata, gene_names, color, dims=[0, 1], current_layer='spliced', use_raw=False, Vkey='S', Ekey='spliced',
             basis='umap', mode='all', cmap=None, gs=None, **kwargs)

        scatters(adata, gene_names, color, dims=[0, 1], current_layer='spliced', use_raw=False, Vkey='S', Ekey='spliced',
             basis='umap', mode='all', cmap=None, gs=None, **kwargs)

    plt.tight_layout()



