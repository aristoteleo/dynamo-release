import numpy as np
import sys
from scipy.sparse import issparse
from .scatters import scatters
from ..tools.velocity import sol_u, sol_s, solve_first_order_deg
from ..tools.utils_moments import moments


def dynamics(adata, vkey, tkey, unit='hours', log=True, log_unnormalized=True, y_log_scale=False, group=None, ncols=None,
                           figsize=None, dpi=None, boxwidth=None, barwidth=None, show=True):
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

    t, asspt_mRNA, experiment_type, _, mode, has_splicing, has_labeling, has_protein = adata.uns['dynamics'].values()

    groups = [''] if group is None else np.unique(adata.obs[group])

    T = adata.obs[tkey].values if t is None else t
    T_uniq = np.unique(T)
    t = np.linspace(0, T_uniq[-1], 1000)
    gene_idx = [np.where(adata.var.index.values == gene)[0][0] for gene in vkey]
    prefix = 'kinetic_parameter_'

    if boxwidth is None and len(T_uniq) > 1:
        boxwidth = 0.8 * (np.diff(T_uniq).min() / 2)

    if barwidth is None and len(T_uniq) > 1:
        barwidth = 0.8 * (np.diff(T_uniq).min() / 2)

    if has_splicing:
        if mode is 'moment':
            sub_plot_n = 4 # number of subplots for each gene
            tmp = [adata[:, gene_idx].layers['X_ul'].A.T, adata.layers['X_sl'].A.T] if 'X_ul' in adata.layers.keys() else \
                    [adata[:, gene_idx].layers['ul'].A.T, adata.layers['sl'].A.T]
            x_data = [tmp[0].A, tmp[1].A] if issparse(tmp[0]) else tmp
            if log_unnormalized and 'X_ul' not in adata.layers.keys():
                x_data = [np.log(tmp[0] + 1), np.log(tmp[1] + 1)]

        elif experiment_type is 'kin' or experiment_type is 'deg':
            sub_plot_n = 4
        elif experiment_type is 'one_shot': # just the labeled RNA
            sub_plot_n = 1
        elif experiment_type is 'mix_std_stm':
            sub_plot_n = 5
    else:
        if mode is 'moment':
            sub_plot_n = 2
            tmp = adata[:, gene_idx].layers['X_new'].T if 'X_new' in adata.layers.keys() else adata[:, gene_idx].layers['new'].T
            x_data = [tmp.A] if issparse(tmp) else [tmp]

            if log_unnormalized and 'X_new' not in adata.layers.keys():
                x_data = [np.log(x_data[0] + 1)]
        elif experiment_type is 'kin'or experiment_type is 'deg':
            sub_plot_n = 2
        elif experiment_type is 'one_shot': # just the labeled RNA
            sub_plot_n = 1
        elif experiment_type is 'mix_std_stm':
            sub_plot_n = 3

    ncols = len(gene_idx) if ncols is None else min(len(gene_idx), ncols)
    nrows = int(np.ceil(len(gene_idx) * sub_plot_n / ncols))
    figsize = [7, 5] if figsize is None else figsize
    gs = plt.GridSpec(nrows, ncols, plt.figure(None, (figsize[0] * ncols, figsize[1] * nrows), dpi=dpi))

    # we need to visualize gene in row-wise mode
    for i, idx in enumerate(gene_idx):
        gene_name = adata.var_names[idx]

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
                    ax.boxplot(x=[x_data[j][i][T == std] for std in T_uniq],  positions=T_uniq, widths=boxwidth, showfliers=False, showmeans=False)  # x=T.values, y= # ax1.plot(T, u.T, linestyle='None', marker='o', markersize=10)
                    ax.scatter(T_uniq, Obs_m[j][i], c='r')  # ax1.plot(T, u.T, linestyle='None', marker='o', markersize=10)
                    if y_log_scale:
                        ax.set_yscale('log')
                    if log_unnormalized:
                        ax.set_ylabel('Expression (log)') #
                    else:
                        ax.set_ylabel('Expression')
                    ax.plot(t, mom_data[j], 'k--')
                else:
                    ax.scatter(T_uniq, Obs_v[j - j_species][i], c='r')
                    if y_log_scale:
                        ax.set_yscale('log')
                    if log_unnormalized:
                        ax.set_ylabel('Variance (log expression)') #
                    else:
                        ax.set_ylabel('Variance')
                    ax.plot(t, mom_data[j], 'k--')
                ax.set_xlabel('time (' + unit + ')')
                ax.set_title(gene_name +  " " + title_[j])

        elif experiment_type is 'deg':
            if has_splicing:
                layers = ['X_uu', 'X_ul', 'X_su', 'X_sl'] if 'X_ul' in adata.layers.keys() else ['uu', 'ul', 'su', 'sl']
                uu, ul, su, sl = adata[:, gene_name].layers[layers[0]], adata[:, gene_name].layers[layers[1]], \
                           adata[:, gene_name].layers[layers[2]], adata[:, gene_name].layers[layers[3]]
                uu, ul, su, sl = (uu.toarray().squeeze(), ul.toarray().squeeze(), su.toarray().squeeze(), sl.toarray().squeeze()) \
                    if issparse(uu) else (uu.squeeze(), ul.squeeze(), su.squeeze(), sl.squeeze())

                if log_unnormalized and layers == ['uu', 'ul', 'su', 'sl']:
                    uu, ul, su, sl = np.log(uu + 1), np.log(ul + 1), np.log(su + 1), np.log(sl + 1)

                alpha, beta, gamma, ul0, sl0, uu0, half_life = adata.var.loc[
                    gene_name, [prefix + 'alpha', prefix + 'beta', prefix + 'gamma', prefix + 'ul0', prefix + 'sl0', prefix + 'uu0', 'RNA_half_life']]
                # $u$ - unlabeled, unspliced
                # $s$ - unlabeled, spliced
                # $w$ - labeled, unspliced
                # $l$ - labeled, spliced
                #
                beta = sys.float_info.epsilon if beta == 0 else beta
                gamma = sys.float_info.epsilon if gamma == 0 else gamma
                u = sol_u(t, uu0, alpha, beta)
                su0 = np.mean(su[T == np.min(T)])
                s = sol_s(t, su0, uu0, 0, beta, gamma)
                w = sol_u(t, ul0, 0, beta)
                l = sol_s(t, sl0, ul0, 0, beta, gamma)
                title_ = ['(unspliced unlabeled)', '(unspliced labeled)', '(spliced unlabeled)', '(spliced labeled)']

                Obs, Pred = np.vstack((uu, ul, su, sl)), np.vstack((u, w, s, l))
            else:
                layers = ['X_new', 'X_total'] if 'X_new' in adata.layers.keys() else ['new', 'total']
                uu, ul = adata[:, gene_name].layers[layers[1]] - adata[:, gene_name].layers[layers[0]], \
                                 adata[:, gene_name].layers[layers[0]]
                uu, ul = (uu.toarray().squeeze(), ul.toarray().squeeze()) if issparse(uu) else (uu.squeeze(), ul.squeeze())

                if log_unnormalized and layers == ['new', 'total']:
                    uu, ul = np.log(uu + 1), np.log(ul + 1)

                alpha, gamma, uu0, ul0, half_life = adata.var.loc[gene_name, [prefix + 'alpha', prefix + 'gamma', prefix + 'uu0', prefix + 'ul0', 'RNA_half_life']]

                # require no beta functions
                u = sol_u(t, uu0, alpha, gamma)
                s = None # sol_s(t, su0, uu0, 0, 1, gamma)
                w = sol_u(t, ul0, 0, gamma)
                l = None # sol_s(t, 0, 0, alpha, 1, gamma)
                title_ = ['(unlabeled)', '(labeled)']

                Obs, Pred = np.vstack((uu, ul)), np.vstack((u, w))

            for j in range(sub_plot_n):
                row_ind = int(np.floor(i/ncols)) # make sure unlabled and labeled are in the same column.
                ax = plt.subplot(gs[(row_ind * sub_plot_n + j) * ncols + i % ncols])
                ax.boxplot(x=[Obs[j][T == std] for std in T_uniq], positions=T_uniq, widths=boxwidth,
                           showfliers=False, showmeans=True)
                ax.plot(t, Pred[j], 'k--')
                if j == sub_plot_n - 1:
                    ax.text(0.8, 0.8, r'$t_{1/2} = $' + "{0:.2f}".format(half_life) + unit[0], ha='right', va='top', transform=ax.transAxes)
                ax.set_xlabel('time (' + unit + ')')
                if y_log_scale:
                    ax.set_yscale('log')
                if log_unnormalized:
                    ax.set_ylabel('Expression (log)')
                else:
                    ax.set_ylabel('Expression')
                ax.set_title(gene_name + " " + title_[j])
        elif experiment_type is 'kin':
            if has_splicing:
                layers = ['X_uu', 'X_ul', 'X_su', 'X_sl'] if 'X_ul' in adata.layers.keys() else ['uu', 'ul', 'su', 'sl']
                uu, ul, su, sl = adata[:, gene_name].layers[layers[0]], adata[:, gene_name].layers[layers[1]], \
                           adata[:, gene_name].layers[layers[2]], adata[:, gene_name].layers[layers[3]]
                uu, ul, su, sl = (uu.toarray().squeeze(), ul.toarray().squeeze(), su.toarray().squeeze(), sl.toarray().squeeze()) \
                    if issparse(uu) else (uu.squeeze(), ul.squeeze(), su.squeeze(), sl.squeeze())

                if log_unnormalized and layers == ['uu', 'ul', 'su', 'sl']:
                    uu, ul, su, sl = np.log(uu + 1), np.log(ul + 1), np.log(su + 1), np.log(sl + 1)

                alpha, beta, gamma, uu0, su0 = adata.var.loc[
                    gene_name, [prefix + 'alpha', prefix + 'beta', prefix + 'gamma', prefix + 'uu0', prefix + 'su0']]
                # $u$ - unlabeled, unspliced
                # $s$ - unlabeled, spliced
                # $w$ - labeled, unspliced
                # $l$ - labeled, spliced
                #
                beta = sys.float_info.epsilon if beta == 0 else beta
                gamma = sys.float_info.epsilon if gamma == 0 else gamma
                u = sol_u(t, uu0, 0, beta)
                s = sol_s(t, su0, uu0, 0, beta, gamma)
                w = sol_u(t, 0, alpha, beta)
                l = sol_s(t, 0, 0, alpha, beta, gamma)

                title_ = ['(unspliced unlabeled)', '(unspliced labeled)', '(spliced unlabeled)', '(spliced labeled)']

                Obs, Pred = np.vstack((uu, ul, su, sl)), np.vstack((u, w, s, l))
            else:
                layers = ['X_new', 'X_total'] if 'X_new' in adata.layers.keys() else ['new', 'total']
                uu, ul = adata[:, gene_name].layers[layers[1]] - adata[:, gene_name].layers[layers[0]], \
                                 adata[:, gene_name].layers[layers[0]]
                uu, ul = (uu.toarray().squeeze(), ul.toarray().squeeze()) if issparse(uu) else (uu.squeeze(), ul.squeeze())

                if log_unnormalized and layers == ['new', 'total']:
                    uu, ul = np.log(uu + 1), np.log(ul + 1)

                alpha, gamma, uu0 = adata.var.loc[gene_name, [prefix + 'alpha', prefix + 'gamma', prefix + 'uu0']]

                # require no beta functions
                u = sol_u(t, uu0, 0, gamma)
                s = None # sol_s(t, su0, uu0, 0, 1, gamma)
                w = sol_u(t, 0, alpha, gamma)
                l = None # sol_s(t, 0, 0, alpha, 1, gamma)

                title_ = ['(unlabeled)', '(labeled)']

                Obs, Pred = np.vstack((uu, ul)), np.vstack((u, w))

            for j in range(sub_plot_n):
                row_ind = int(np.floor(i/ncols)) # make sure unlabled and labeled are in the same column.
                ax = plt.subplot(gs[(row_ind * sub_plot_n + j) * ncols + i % ncols])
                ax.boxplot(x=[Obs[j][T == std] for std in T_uniq], positions=T_uniq, widths=boxwidth,
                           showfliers=False, showmeans=True)
                ax.plot(t, Pred[j], 'k--')
                ax.set_xlabel('time (' + unit + ')')
                if y_log_scale:
                    ax.set_yscale('log')
                if log_unnormalized:
                    ax.set_ylabel('Expression (log)')
                else:
                    ax.set_ylabel('Expression')
                ax.set_title(gene_name +  " " + title_[j])
        elif experiment_type is 'one_shot':
            if has_splicing:
                layers = ['X_uu', 'X_ul', 'X_su', 'X_sl'] if 'X_ul' in adata.layers.keys() else ['uu', 'ul', 'su', 'sl']
                uu, ul, su, sl = adata[:, gene_name].layers[layers[0]], adata[:, gene_name].layers[layers[1]], \
                                 adata[:, gene_name].layers[layers[2]], adata[:, gene_name].layers[layers[3]]
                uu, ul, su, sl = (uu.toarray().squeeze(), ul.toarray().squeeze(), su.toarray().squeeze(), sl.toarray().squeeze()) \
                    if issparse(uu) else (uu.squeeze(), ul.squeeze(), su.squeeze(), sl.squeeze())

                if log_unnormalized and layers == ['uu', 'ul', 'su', 'sl']:
                    uu, ul, su, sl = np.log(uu + 1), np.log(ul + 1), np.log(su + 1), np.log(sl + 1)

                alpha, beta, gamma, U0, S0 = adata.var.loc[
                    gene_name, [prefix + 'alpha', prefix + 'beta', prefix + 'gamma', prefix + 'U0', prefix + 'S0']]
                # $u$ - unlabeled, unspliced
                # $s$ - unlabeled, spliced
                # $w$ - labeled, unspliced
                # $l$ - labeled, spliced
                #
                beta = sys.float_info.epsilon if beta == 0 else beta
                gamma = sys.float_info.epsilon if gamma == 0 else gamma
                U_sol = sol_u(t, U0, 0, beta)
                S_sol = sol_u(t, S0, 0, gamma)
                l = sol_s(t, 0, 0, alpha, beta, gamma)

                L = su + sl
                title_ = ['labeled']
            else:
                layers = ['X_new', 'X_total'] if 'X_new' in adata.layers.keys() else ['new', 'total']
                uu, ul = adata[:, gene_name].layers[layers[1]] - adata[:, gene_name].layers[layers[0]], \
                         adata[:, gene_name].layers[layers[0]]
                uu, ul = (uu.toarray().squeeze(), ul.toarray().squeeze()) if issparse(uu) else (
                uu.squeeze(), ul.squeeze())

                if log_unnormalized and layers == ['new', 'total']:
                    uu, ul = np.log(uu + 1), np.log(ul + 1)

                alpha, gamma, total0 = adata.var.loc[gene_name, [prefix + 'alpha', prefix + 'gamma', prefix + 'total0']]

                # require no beta functions
                old = sol_u(t, total0, 0, gamma)
                s = None  # sol_s(t, su0, uu0, 0, 1, gamma)
                w = None
                l = sol_u(t, 0, alpha, gamma)  # sol_s(t, 0, 0, alpha, 1, gamma)

                L = ul
                title_ = ['labeled']

            Obs, Pred = np.hstack((np.zeros(L.shape), L)), np.hstack(l)

            row_ind = int(np.floor(i / ncols))  # make sure unlabled and labeled are in the same column.
            ax = plt.subplot(gs[(row_ind * sub_plot_n) * ncols + i % ncols])
            ax.boxplot(x=[Obs[np.hstack((np.zeros_like(T), T)) == std] for std in [0, T_uniq[0]]], positions=[0, T_uniq[0]], widths=boxwidth,
                       showfliers=False, showmeans=True)
            ax.plot(t, Pred, 'k--')
            ax.set_xlabel('time (' + unit + ')')
            if y_log_scale:
                ax.set_yscale('log')
            if log_unnormalized:
                ax.set_ylabel('Expression (log)')
            else:
                ax.set_ylabel('Expression')
            ax.set_title(gene_name +  " " + title_[0])
        elif experiment_type is 'mix_std_stm':
            if has_splicing:
                layers = ['X_uu', 'X_ul', 'X_su', 'X_sl'] if 'X_ul' in adata.layers.keys() else ['uu', 'ul', 'su', 'sl']
                uu, ul, su, sl = adata[:, gene_name].layers[layers[0]], adata[:, gene_name].layers[layers[1]], \
                           adata[:, gene_name].layers[layers[2]], adata[:, gene_name].layers[layers[3]]
                uu, ul, su, sl = (uu.toarray().squeeze(), ul.toarray().squeeze(), su.toarray().squeeze(), sl.toarray().squeeze()) \
                    if issparse(uu) else (uu.squeeze(), ul.squeeze(), su.squeeze(), sl.squeeze())

                if log_unnormalized and layers == ['uu', 'ul', 'su', 'sl']:
                    uu, ul, su, sl = np.log(uu + 1), np.log(ul + 1), np.log(su + 1), np.log(sl + 1)

                beta, gamma, alpha_std = adata.var.loc[gene_name, [prefix + 'beta', prefix + 'gamma', prefix + 'alpha_std']]
                alpha_stm = adata[:, gene_name].varm[prefix + 'alpha'].flatten()
                alpha_stm0, k, _ = solve_first_order_deg(T_uniq[1:], alpha_stm)

                # $u$ - unlabeled, unspliced
                # $s$ - unlabeled, spliced
                # $w$ - labeled, unspliced
                # $l$ - labeled, spliced
                #
                # calculate labeled unspliced and spliced mRNA amount
                u1, s1 = np.zeros(len(t) - 1), np.zeros(len(t) - 1)
                for ind in np.arange(1, len(t)):
                    t_i = t[ind]
                    u0 = sol_u(np.max(t) - t_i, 0, alpha_std, beta)
                    alpha_stm_t_i = alpha_stm0 * np.exp(-k * t_i)
                    u1[ind - 1], s1[ind - 1] = sol_u(t_i, u0, alpha_stm_t_i, beta), sol_u(np.max(t), 0, beta, gamma)

                Obs, Pred = np.vstack((ul, sl, uu, su)), np.vstack((u1.reshape(1, -1), s1.reshape(1, -1)))
                j_species, title_ = 4, ['unspliced labeled (new)', 'spliced labeled (new)', 'unspliced unlabeled (old)', 'spliced unlabeled (old)', 'alpha (steady state vs. stimulation)']
            else:
                layers = ['X_new', 'X_total'] if 'X_new' in adata.layers.keys() else ['new', 'total']
                uu, ul = adata[:, gene_name].layers[layers[1]] - adata[:, gene_name].layers[layers[0]], adata[:, gene_name].layers[layers[0]]
                uu, ul = (uu.toarray().squeeze(), ul.toarray().squeeze()) if issparse(uu) else (uu.squeeze(), ul.squeeze())

                if log_unnormalized and layers == ['new', 'total']:
                    uu, ul = np.log(uu + 1), np.log(ul + 1)

                gamma, alpha_std = adata.var.loc[gene_name, [prefix + 'gamma', prefix + 'alpha_std']]
                alpha_stm = adata[:, gene_name].varm[prefix + 'alpha'].flatten()

                alpha_stm0, k, _ = solve_first_order_deg(T_uniq[1:], alpha_stm)
                # require no beta functions
                u1 = np.zeros(len(t) - 1)
                for ind in np.arange(1, len(t)):
                    t_i = t[ind]
                    u0 = sol_u(np.max(t) - t_i, 0, alpha_std, gamma)
                    alpha_stm_t_i = alpha_stm0 * np.exp(-k * t_i)
                    u1[ind - 1] = sol_u(t_i, u0, alpha_stm_t_i, gamma)

                Obs, Pred = np.vstack((ul, uu)), np.vstack((u1.reshape(1, -1)))
                j_species, title_ = 2, ['labeled (new)', 'unlabeled (old)', 'alpha (steady state vs. stimulation)']

            if log: Obs = np.log(Obs + 1)
            group_list = [np.repeat(alpha_std, len(alpha_stm)), alpha_stm]

            for j in range(sub_plot_n):
                row_ind = int(np.floor(i / ncols))  # make sure all related plots for the same gene in the same column.
                ax = plt.subplot(gs[(row_ind * sub_plot_n + j) * ncols + i % ncols])
                if j < j_species / 2:
                    ax.boxplot(x=[Obs[j][T == std] for std in T_uniq[1:]], positions=T_uniq[1:], widths=boxwidth,
                               showfliers=False, showmeans=True)
                    ax.plot(t[1:], Pred[j], 'k--')
                    ax.set_xlabel('time (' + unit + ')')
                    ax.set_title(gene_name + ': ' + title_[j])

                    if y_log_scale:
                        ax.set_yscale('log')
                    if log_unnormalized:
                        ax.set_ylabel('Expression (log)')
                    else:
                        ax.set_ylabel('Expression')
                elif j < j_species:
                    ax.boxplot(x=[Obs[j][T == std] for std in T_uniq], positions=T_uniq, widths=boxwidth,
                               showfliers=False, showmeans=True)
                    ax.set_xlabel('time (' + unit + ')')
                    ax.set_title(gene_name + ': ' + title_[j])

                    if y_log_scale:
                        ax.set_yscale('log')
                    if log_unnormalized:
                        ax.set_ylabel('Expression (log)')
                    else:
                        ax.set_ylabel('Expression')

                else:
                    x = T_uniq[1:]  # the label locations
                    group_width = barwidth / 2
                    bar_coord, group_name, group_ind = [-1, 1], ['steady state', 'stimulation'], 0

                    for group_ind in range(len(group_list)):
                        cur_group = group_list[group_ind]
                        ax.bar(x + bar_coord[group_ind] * group_width, cur_group, barwidth, label=group_name[group_ind])
                        # Add gene name, experimental type, etc.
                        ax.set_xlabel('time (' + unit + ')')
                        ax.set_ylabel('alpha (translation rate)')
                        ax.set_xticks(x)
                        ax.set_xticklabels(x)
                        group_ind += 1
                    ax.legend()

                ax.set_xlabel('time (' + unit + ')')
                ax.set_title(gene_name + ': ' + title_[j])
        elif experiment_type is 'multi_time_series':
            pass # group by different groups
        elif experiment_type is 'coassay':
            pass # show protein velocity (steady state and the Gamma distribution model)

    if show: plt.show()


def dynamics_(adata, gene_names, color, dims=[0, 1], current_layer='spliced', use_raw=False, Vkey='S', Ekey='spliced',
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



