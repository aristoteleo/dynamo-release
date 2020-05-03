import numpy as np
from scipy.sparse import issparse
from ..tools.moments import prepare_data_has_splicing

def plot_kin_det(adata, genes, has_splicing, use_smoothed, log_unnormalized,
                 t, T, T_uniq, unit, X_data, X_fit_data, logLL, true_p,
                 grp_len, sub_plot_n, ncols, boxwidth, gs, fig_mat, gene_order, y_log_scale,
                 true_param_prefix, true_params, est_params,
                 show_variance, show_kin_parameters, ):
    import matplotlib.pyplot as plt
    true_alpha, true_beta, true_gamma = true_params
    alpha, beta, gamma = est_params

    layers = ['M_ul', 'M_sl', 'M_uu', 'M_su'] if (
            'M_ul' in adata.layers.keys() and use_smoothed) \
        else ['ul', 'sl', 'uu', 'su']

    layer_u = 'M_ul' if ('M_ul' in adata.layers.keys() and use_smoothed) else 'ul'
    layer_s = 'M_sl' if ('M_ul' in adata.layers.keys() and use_smoothed) else 'sl'

    _, X_raw = prepare_data_has_splicing(adata, genes, T,
                                         layer_u=layer_u, layer_s=layer_s, total_layers=layers)

    for i, gene_name in enumerate(genes):
        cur_X_data, cur_X_fit_data, cur_logLL = X_data[i], X_fit_data[i], logLL[i]

        title_ = ["ul", "sl"]

        for j in range(sub_plot_n):
            Obs = X_raw[i][j][0].A.flatten() if issparse(X_raw[i][j][0]) else X_raw[i][j][0].flatten()

            row_ind = int(
                np.floor(i / ncols)
            )  # make sure unlabled and labeled are in the same column.

            col_loc = (row_ind * sub_plot_n + j) * ncols * grp_len + \
                      (i % ncols - 1) * grp_len + 1
            row_i, col_i = np.where(fig_mat == col_loc)
            ax = plt.subplot(
                gs[col_loc]
            ) if gene_order == 'column' else \
                plt.subplot(
                    gs[fig_mat[col_i, row_i][0]]
                )
            if j == 0:
                ax.text(0.95, 0.05, r'$logLL=%.2f$' % (cur_logLL), ha='right',
                        va='center', transform=ax.transAxes)
                if true_param_prefix is not None:
                    if show_kin_parameters:
                        if has_splicing:
                            ax.text(
                                0.75,
                                0.90,
                                r"$\alpha$"
                                + ": {0:.2f}; ".format(true_alpha)
                                + r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha)
                                + r"$\beta$"
                                + ": {0:.2f}; ".format(true_beta)
                                + r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta)
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma)
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma),
                                ha="right",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.75,
                                0.90,
                                r"$\alpha$"
                                + ": {0:.2f}; ".format(true_alpha)
                                + r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha)
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma)
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma),
                                ha="right",
                                va="top",
                                transform=ax.transAxes,
                            )
            if show_variance:
                ax.boxplot(
                    x=[Obs[T == std] for std in T_uniq],
                    positions=T_uniq,
                    widths=boxwidth,
                    showfliers=False,
                    showmeans=True,
                )
                ax.plot(T_uniq, cur_X_fit_data.T[j], "b")
                ax.plot(T_uniq, cur_X_data[j], "k--")
                ax.set_title(gene_name + " (" + title_[j] + ")")
            else:
                ax.plot(T_uniq, cur_X_fit_data)
                ax.plot(T_uniq, cur_X_data.T, "k--")
                ax.legend(['ul', 'sl'])
                ax.set_title(gene_name)

            if true_param_prefix is not None:
                ax.plot(t, true_p[j], "r")
            ax.set_xlabel("time (" + unit + ")")
            if y_log_scale:
                ax.set_yscale("log")
            if log_unnormalized:
                ax.set_ylabel("Expression (log)")
            else:
                ax.set_ylabel("Expression")

    # somehow plot the figure here?
    return gs

def plot_kin_sto(ax, t, T, cur_X_data, cur_X_fit_data, gene_name, logLL,
                 has_splicing, show_moms_fit, show_variance, show_kin_parameters, ):
    ax.plot(t, cur_X_fit_data)
    ax.set_prop_cycle(None)
    ax.plot(np.unique(T), cur_X_data, '--')
    ax.xlabel('Time (hrs.)')
    ax.legend(['unspliced, labeled', 'spliced, labeled'])
    ax.title(gene_name)

    ax.text(0.95, 0.3, r'logLL=%.4f$' % (logLL), horizontalalignment='right',
            verticalalignment='center', transform=ax.transAxes)

    return ax

def plot_kin_mix(ax, t, T, cur_X_data, cur_X_fit_data, gene_name, logLL):
    ax.plot(t, cur_X_fit_data)
    ax.set_prop_cycle(None)
    ax.plot(np.unique(T), cur_X_data, '--')
    ax.xlabel('Time (hrs.)')
    ax.legend(['unspliced, labeled', 'spliced, labeled'])
    ax.title(gene_name)

    ax.text(0.95, 0.3, r'logLL=%.4f$' % (logLL), horizontalalignment='right',
            verticalalignment='center', transform=ax.transAxes)

    return ax

def plot_kin_mix_det_sto(ax, t, T, cur_X_data, cur_X_fit_data, gene_name, logLL):
    ax.plot(t, cur_X_fit_data)
    ax.set_prop_cycle(None)
    ax.plot(np.unique(T), cur_X_data, '--')
    ax.xlabel('Time (hrs.)')
    ax.legend(['unspliced, labeled', 'spliced, labeled'])
    ax.title(gene_name)

    ax.text(0.95, 0.3, r'logLL=%.4f$' % (logLL), horizontalalignment='right',
            verticalalignment='center', transform=ax.transAxes)

    return ax


def plot_kin_mix_sto_sto(ax, t, T, cur_X_data, cur_X_fit_data, gene_name, logLL):
    ax.plot(t, cur_X_fit_data)
    ax.set_prop_cycle(None)
    ax.plot(np.unique(T), cur_X_data, '--')
    ax.xlabel('Time (hrs.)')
    ax.legend(['unspliced, labeled', 'spliced, labeled'])
    ax.title(gene_name)

    ax.text(0.95, 0.3, r'logLL=%.4f$' % (logLL), horizontalalignment='right',
            verticalalignment='center', transform=ax.transAxes)

    return ax

def plot_deg_det(ax, t, T, cur_X_data, cur_X_fit_data, gene_name, logLL):
    ax.plot(t, cur_X_fit_data)
    ax.set_prop_cycle(None)
    ax.plot(np.unique(T), cur_X_data, '--')
    ax.xlabel('Time (hrs.)')
    ax.legend(['unspliced, labeled', 'spliced, labeled'])
    ax.title(gene_name)

    ax.text(0.95, 0.3, r'logLL=%.4f$' % (logLL), horizontalalignment='right',
            verticalalignment='center', transform=ax.transAxes)

    return ax

def plot_deg_sto(ax, t, T, cur_X_data, cur_X_fit_data, gene_name, logLL):
    ax.plot(t, cur_X_fit_data)
    ax.set_prop_cycle(None)
    ax.plot(np.unique(T), cur_X_data, '--')
    ax.xlabel('Time (hrs.)')
    ax.legend(['unspliced, labeled', 'spliced, labeled'])
    ax.title(gene_name)

    ax.text(0.95, 0.3, r'logLL=%.4f$' % (logLL), horizontalalignment='right',
            verticalalignment='center', transform=ax.transAxes)

    return ax
