import numpy as np
from scipy.sparse import issparse
from .scatters import scatters
from ..tools.velocity import sol_u, sol_s
from ..tools.utils_moments import moments


def metabolic_labeling_fit(adata, vkey, tkey, unit='hours', log=True, y_log_scale = False, group=None, ncols=None,
                           figsize=None, dpi=None, boxwidth=0.5, barwidth=0.35, show=True):
    """ Plot the data and fitting of different metabolic labeling experiments.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        vkey: list of `str`
            key for variable or gene names.
        tkey: `str`
            The key for the time annotation in .obs attribute.
        unit: `str` (default: `hour`)
            The unit of the labeling time, for example, `hours` or `minutes`.
        y_log_scale: `bool` (default: `True`)
            Whether or not to use log scale for y-axis.
        group: `str` or None (default: `None`)
            The key for the group annotation in .obs attribute. Currently not used.
        ncols: `int` or None (default: `None`)
            The number of columns in the plot.
        figsize: `[float, float]` or `(float, float)` or None
            The size of figure.
        dpi: `float` or None
            Figure resolution.
        boxwidth: `float`
            The width of the box of the boxplot.
        show: `bool` (default: `True`)
            Whether to plot the figure.

    Returns
    -------
        Nothing but plot the figure of the metabolic fitting.
    """

    import matplotlib.pyplot as plt
    # import seaborn as sns

    asspt_mRNA, experiment_type, _, mode, has_splicing, has_labeling, has_protein = adata.uns['dynamics'].values()

    groups = [''] if group is None else np.unique(adata.obs[group])

    T = adata.obs[tkey]
    Tsort = T.unique()
    gene_idx = [np.where(adata.var.index.values == gene)[0][0] for gene in vkey]
    prefix = 'kinetic_parameter_'

    if has_splicing:
        if mode is 'moment':
            sub_plot_n = 4 # number of subplots for each gene
            tmp = [adata[:, gene_idx].layers['X_ul'].A.T, adata.layers['X_sl'].A.T] if 'X_ul' in adata.layers.keys() else \
                    [adata[:, gene_idx].layers['ul'].A.T, adata.layers['sl'].A.T]
            x_data = [tmp[0].A, tmp[1].A] if issparse(tmp[0]) else tmp
        elif experiment_type is 'kin'or experiment_type is 'deg':
            sub_plot_n = 4
        elif experiment_type is 'mix_std_stm':
            sub_plot_n = 8 # ? two alpha
    else:
        if mode is 'moment':
            sub_plot_n = 2
            tmp = adata[:, gene_idx].layers['X_new'].T if 'X_new' in adata.layers.keys() else adata[:, gene_idx].layers['new'].T
            x_data = [tmp.A] if issparse(tmp) else [tmp]
        elif experiment_type is 'kin'or experiment_type is 'deg':
            sub_plot_n = 2
        elif experiment_type is 'mix_std_stm':
            sub_plot_n = 4 #  ? two alpha

    ncols = len(gene_idx) if ncols is None else min(len(gene_idx), ncols)
    nrows = int(np.ceil(len(gene_idx) * sub_plot_n / ncols))
    figsize = [7, 5] if figsize is None else figsize
    gs = plt.GridSpec(nrows, ncols, plt.figure(None, (figsize[0] * ncols, figsize[1] * nrows), dpi=dpi))

    for i, idx in enumerate(gene_idx):
        gene_name = adata.var_names[idx]
        #
        t = np.linspace(Tsort[0], Tsort[-1], Tsort[-1] - Tsort[0])

        if asspt_mRNA is 'ss':
            # run the phase plot
            pass
        elif mode is 'moment':
            a, b, alpha_a, alpha_i, beta, gamma = adata.var.loc[gene_name, \
             [prefix + 'a', prefix + 'b', prefix + 'alpha_a', prefix + 'alpha_i', prefix + 'beta', prefix + 'gamma']]
            params = {'a': a, 'b': b, 'alpha_a': alpha_a, 'alpha_i': alpha_i, 'beta': beta, 'gamma': gamma} # "la": 1, "si": 0,
            mom = moments(*list(params.values()))
            mom.integrate(t)
            mom_data = mom.get_all_central_moments() if has_splicing else mom.get_nosplice_central_moments()

            # n_mean, n_var = x_data[:2, :], x_data[2:, :]
            if has_splicing:
                title_ = ['(unspliced labeled)', '(spliced labeled)',
                          '(unspliced labeled)', '(spliced labeled)']
                Obs_m = [adata.uns['M_ul'], adata.uns['M_sl']]
                Obs_v = [adata.uns['V_ul'], adata.uns['V_sl']]
                j_species = 2 # number of species
            else:
                # only use new key for calculation, so we only have M, V
                title_ = [' (labeled)', ' (labeled)']
                Obs_m, Obs_v = [adata.uns['M']], [adata.uns['V']]
                j_species = 1

            for j in range(sub_plot_n):
                row_ind = int(np.floor(i/ncols)) # make sure all related plots for the same gene in the same column.
                ax = plt.subplot(gs[(row_ind * sub_plot_n + j) * ncols + i % ncols])
                if j < j_species:
                    ax.boxplot(x=[x_data[j][i][T == std] for std in Tsort],  positions=Tsort, widths=boxwidth, showfliers=False)  # x=T.values, y= # ax1.plot(T, u.T, linestyle='None', marker='o', markersize=10)
                    ax.scatter(Tsort, Obs_m[j][i], c='r')  # ax1.plot(T, u.T, linestyle='None', marker='o', markersize=10)
                    if y_log_scale:
                        ax.set_ylabel('Expression (log)')
                    else:
                        ax.set_ylabel('Expression')
                    ax.plot(t, mom_data[j], 'k--')
                else:
                    ax.scatter(Tsort, Obs_v[j - j_species][i], c='r')
                    if y_log_scale:
                        ax.set_ylabel('Variance (log)')
                    else:
                        ax.set_ylabel('Variance')
                    ax.plot(t, mom_data[j], 'k--')
                ax.set_xlabel('time (' + unit + ')')
                ax.set_title(gene_name + title_[j])
                if y_log_scale: ax.set_yscale('log')
        elif experiment_type is 'deg':
            if has_splicing:
                layers = ['X_uu', 'X_ul', 'X_su', 'X_sl'] if 'X_ul' in adata.layers.keys() else ['uu', 'ul', 'su', 'sl']
                uu, ul, su, sl = adata[:, gene_name].layers[layers[0]], adata[:, gene_name].layers[layers[1]], \
                           adata[:, gene_name].layers[layers[2]], adata[:, gene_name].layers[layers[3]]
                uu, ul, su, sl = (uu.toarray().squeeze(), ul.toarray().squeeze(), su.toarray().squeeze(), sl.toarray().squeeze()) \
                    if issparse(uu) else (uu.squeeze(), ul.squeeze(), su.squeeze(), sl.squeeze())
                alpha, beta, gamma, ul0, sl0 = adata.var.loc[
                    gene_name, [prefix + 'alpha', prefix + 'beta', prefix + 'gamma', prefix + 'ul0', prefix + 'sl0']]
                # $u$ - unlabeled, unspliced
                # $s$ - unlabeled, spliced
                # $w$ - labeled, unspliced
                # $l$ - labeled, spliced
                #
                u = sol_u(t, ul0, 0, beta)
                s = sol_s(t, sl0, ul0, 0, beta, gamma)
                w = sol_u(t, 0, alpha, beta)
                l = sol_s(t, 0, 0, alpha, beta, gamma)
            else:
                layers = ['X_new', 'X_total'] if 'X_new' in adata.layers.keys() else ['new', 'total']
                uu, ul, su, sl = adata[:, gene_name].layers[layers[0]], adata[:, gene_name].layers[layers[0]], None, None
                uu, ul = (uu.toarray().squeeze(), ul.toarray().squeeze()) if issparse(uu) else (uu.squeeze(), ul.squeeze())
                alpha, gamma, ul0 = adata.var.loc[gene_name, [prefix + 'alpha', prefix + 'gamma', prefix + 'ul0']]

                # require no beta functions
                u = sol_u(t, ul0, 0, gamma)
                s = None # sol_s(t, su0, uu0, 0, 1, gamma)
                w = sol_u(t, 0, alpha, gamma)
                l = None # sol_s(t, 0, 0, alpha, 1, gamma)

            Obs, Pred = np.vstack((uu, ul, su, sl)), np.vstack((u, w, s, l))
            title_ = ['(unspliced unlabeled)', '(unspliced labeled)', '(spliced unlabeled)', '(spliced labeled)']

            for j in range(sub_plot_n):
                row_ind = int(np.floor(idx/ncols)) # make sure unlabled and labeled are in the same column.
                ax = plt.subplot(gs[(row_ind * sub_plot_n + j) * ncols + i % ncols])
                ax.boxplot(x=[Obs[i][T == std] for std in Tsort], positions=Tsort, widths=boxwidth,
                           showfliers=False)
                ax.plot(t, u, 'k--')
                ax.set_xlabel('time (' + unit + ')')
                ax.set_ylabel('Expression')
                ax.set_title(gene_name + title_[j])
        elif experiment_type is 'kin':
            if has_splicing:
                layers = ['X_uu', 'X_ul', 'X_su', 'X_sl'] if 'X_ul' in adata.layers.keys() else ['uu', 'ul', 'su', 'sl']
                uu, ul, su, sl = adata[:, gene_name].layers[layers[0]], adata[:, gene_name].layers[layers[1]], \
                           adata[:, gene_name].layers[layers[2]], adata[:, gene_name].layers[layers[3]]
                uu, ul, su, sl = (uu.toarray().squeeze(), ul.toarray().squeeze(), su.toarray().squeeze(), sl.toarray().squeeze()) \
                    if issparse(uu) else (uu.squeeze(), ul.squeeze(), su.squeeze(), sl.squeeze())
                alpha, beta, gamma, uu0, su0 = adata.var.loc[
                    gene_name, [prefix + 'alpha', prefix + 'beta', prefix + 'gamma', prefix + 'uu0', prefix + 'su0']]
                # $u$ - unlabeled, unspliced
                # $s$ - unlabeled, spliced
                # $w$ - labeled, unspliced
                # $l$ - labeled, spliced
                #
                u = sol_u(t, uu0, 0, beta)
                s = sol_s(t, su0, uu0, 0, beta, gamma)
                w = sol_u(t, 0, alpha, beta)
                l = sol_s(t, 0, 0, alpha, beta, gamma)
            else:
                layers = ['X_new', 'X_total'] if 'X_new' in adata.layers.keys() else ['new', 'total']
                uu, ul, su, sl = adata[:, gene_name].layers[layers[0]], adata[:, gene_name].layers[layers[0]], None, None
                uu, ul = (uu.toarray().squeeze(), ul.toarray().squeeze()) if issparse(uu) else (uu.squeeze(), ul.squeeze())
                alpha, gamma, uu0, su0 = adata.var.loc[gene_name, [prefix + 'alpha', prefix + 'gamma', prefix + 'uu0', prefix + 'su0']]

                # require no beta functions
                u = sol_u(t, uu0, 0, gamma)
                s = None # sol_s(t, su0, uu0, 0, 1, gamma)
                w = sol_u(t, 0, alpha, gamma)
                l = None # sol_s(t, 0, 0, alpha, 1, gamma)

            Obs, Pred = np.vstack((uu, ul, su, sl)), np.vstack((u, w, s, l))
            title_ = ['(unspliced unlabeled)', '(unspliced labeled)', '(spliced unlabeled)', '(spliced labeled)']

            for j in range(sub_plot_n):
                row_ind = int(np.floor(idx/ncols)) # make sure unlabled and labeled are in the same column.
                ax = plt.subplot(gs[(row_ind * sub_plot_n + j) * ncols + i % ncols])
                ax.boxplot(x=[Obs[i][T == std] for std in Tsort], positions=Tsort, widths=boxwidth,
                           showfliers=False)
                ax.plot(t, u, 'k--')
                ax.set_xlabel('time (' + unit + ')')
                ax.set_ylabel('Expression')
                ax.set_title(gene_name + title_[j])
        elif experiment_type is 'mix_std_stmu':
            # we need to visualize two
            if has_splicing:
                layers = ['X_uu', 'X_ul', 'X_su', 'X_sl'] if 'X_ul' in adata.layers.keys() else ['uu', 'ul', 'su', 'sl']
                uu, ul, su, sl = adata[:, gene_name].layers[layers[0]], adata[:, gene_name].layers[layers[1]], \
                           adata[:, gene_name].layers[layers[2]], adata[:, gene_name].layers[layers[3]]
                uu, ul, su, sl = (uu.toarray().squeeze(), ul.toarray().squeeze(), su.toarray().squeeze(), sl.toarray().squeeze()) \
                    if issparse(uu) else (uu.squeeze(), ul.squeeze(), su.squeeze(), sl.squeeze())
                beta, gamma, uu0, su0 = adata.var.loc[gene_name, [prefix + 'beta', prefix + 'gamma', prefix + 'uu0', prefix + 'su0']]
                group_list = [adata.var.loc[gene_name, prefix + 'alpha_std'].to_list(),
                              adata.var.loc[gene_name, prefix + 'alpha'].to_list()]
                # $u$ - unlabeled, unspliced
                # $s$ - unlabeled, spliced
                # $w$ - labeled, unspliced
                # $l$ - labeled, spliced
                #
                u = sol_u(t, uu0, 0, beta)
                s = sol_s(t, su0, uu0, 0, beta, gamma)
                w = sol_u(t, 0, alpha, beta)
                l = sol_s(t, 0, 0, alpha, beta, gamma)
                j_species = 1
            else:
                layers = ['X_new', 'X_total'] if 'X_new' in adata.layers.keys() else ['new', 'total']
                uu, ul, su, sl = adata[:, gene_name].layers[layers[0]], adata[:, gene_name].layers[layers[0]], None, None
                uu, ul = (uu.toarray().squeeze(), ul.toarray().squeeze()) if issparse(uu) else (uu.squeeze(), ul.squeeze())
                gamma, uu0, su0 = adata.var.loc[gene_name, [prefix + 'gamma', prefix + 'uu0', prefix + 'su0']]
                group_list = [adata.var.loc[gene_name, prefix + 'alpha_std'].to_list(),
                              adata.var.loc[gene_name, prefix + 'alpha'].to_list()]

                # require no beta functions
                u = sol_u(t, uu0, 0, gamma)
                s = None # sol_s(t, su0, uu0, 0, 1, gamma)
                w = sol_u(t, 0, alpha, gamma)
                l = None # sol_s(t, 0, 0, alpha, 1, gamma)
                j_species = 1

            Obs, Pred = np.vstack((uu, ul, su, sl)), np.vstack((u, w, s, l))
            title_ = ['(unspliced unlabeled)', '(unspliced labeled)', '(spliced unlabeled)', '(spliced labeled)']

            x = np.arange(len(gene_name))  # the label locations
            group_width = barwidth / len(group_list)
            bar_coord, group_name, group_ind = [-1, 1], ['steady state', 'stimulation'], 0

            ax.set_title('alpha between steady state or stimulation')
            for cur_group in group_list:
                ax.bar(x + bar_coord[group_ind] * group_width, cur_group, 0.35, label=group_name[group_ind])

                # Add gene name, experimental type, etc.
                ax.set_ylabel('alpha (translation rate)')
                ax.set_xticks(x)
                ax.set_xticklabels(gene_name)
                group_ind += 1
            ax.legend()

            for j in range(sub_plot_n):
                row_ind = int(np.floor(idx/ncols)) # make sure unlabled and labeled are in the same column.
                ax = plt.subplot(gs[(row_ind * sub_plot_n + j) * ncols + i % ncols])
                ax.boxplot(x=[Obs[i][T == std] for std in Tsort], positions=Tsort, widths=boxwidth,
                           showfliers=False)
                ax.plot(t, u, 'k--')
                ax.set_xlabel('time (' + unit + ')')
                ax.set_ylabel('Expression')
                ax.set_title(gene_name + title_[j])

            alpha, gamma, uu0, su0 = adata.var.loc[gene_name, [prefix + 'alpha', prefix + 'gamma', prefix + 'uu0', prefix + 'su0']]
        elif experiment_type is 'one_shot':
            pass
        elif experiment_type is 'multi_time_series':
            pass # group by different groups
        elif experiment_type is 'coassay':
            pass # show protein velocity (steady state and the Gamma distribution model)

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



