import numpy as np
from scipy.sparse import issparse
from ..tools.moments import (
    prepare_data_no_splicing,
    prepare_data_has_splicing,
    prepare_data_mix_has_splicing,
    prepare_data_mix_no_splicing,
)

def plot_kin_det(adata, genes, has_splicing, use_smoothed, log_unnormalized,
                 t, T, T_uniq, unit, X_data, X_fit_data, logLL, true_p,
                 grp_len, sub_plot_n, ncols, boxwidth, gs, fig_mat, gene_order, y_log_scale,
                 true_param_prefix, true_params, est_params,
                 show_variance, show_kin_parameters, ):
    import matplotlib.pyplot as plt
    true_alpha, true_beta, true_gamma = true_params
    alpha, beta, gamma = est_params

    if has_splicing:
        title_ = ["ul", "sl"]

        layers = ['M_ul', 'M_sl', 'M_uu', 'M_su'] if (
                'M_ul' in adata.layers.keys() and use_smoothed) \
            else ['ul', 'sl', 'uu', 'su']

        layer_u = 'M_ul' if ('M_ul' in adata.layers.keys() and use_smoothed) else 'ul'
        layer_s = 'M_sl' if ('M_ul' in adata.layers.keys() and use_smoothed) else 'sl'

        _, X_raw = prepare_data_has_splicing(adata, genes, T,
                                             layer_u=layer_u, layer_s=layer_s, total_layers=layers)
    else:
        title_ = ["new"]

        total_layer = 'M_t' if ('M_t' in adata.layers.keys() and use_smoothed) else 'total'

        layer = 'M_n' if ('M_n' in adata.layers.keys() and use_smoothed) else 'new'
        _, X_raw = prepare_data_no_splicing(adata, adata.var.index, T, layer=layer,
                                            total_layer=total_layer)

    padding = 0.185 if not show_variance else 0
    for i, gene_name in enumerate(genes):
        cur_X_data, cur_X_fit_data, cur_logLL = X_data[i], X_fit_data[i], logLL[i]

        for j in range(sub_plot_n):

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
                ax.text(0.01 + padding, 0.80, r'$logLL={0:.2f}$'.format(cur_logLL) + ' \n'
                        + r"$t_{1/2} = $" + "{0:.2f}".format(np.log(2) / gamma[i]) + unit[0] \
                        , ha='left', va='top', transform=ax.transAxes)

                if show_kin_parameters:
                    if true_param_prefix is not None:
                        if has_splicing:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\alpha$"
                                + ": {0:.2f}; ".format(true_alpha[i])
                                + r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\beta$"
                                + ": {0:.2f}; ".format(true_beta[i])
                                + r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta[i])
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\alpha$"
                                + ": {0:.2f}; ".format(true_alpha[i])
                                + r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                    else:
                        if has_splicing:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
            if show_variance:
                if has_splicing:
                    Obs = X_raw[i][j][0].A.flatten() if issparse(X_raw[i][j][0]) else X_raw[i][j][0].flatten()
                else:
                    Obs = X_raw[i].A.flatten() if issparse(X_raw[i][0]) else X_raw[i].flatten()
                ax.boxplot(
                    x=[Obs[T == std] for std in T_uniq],
                    positions=T_uniq,
                    widths=boxwidth,
                    showfliers=False,
                    showmeans=True,
                )
                if has_splicing:
                    ax.plot(T_uniq, cur_X_fit_data[j], "b")
                    ax.plot(T_uniq, cur_X_data[j], "k--")
                else:
                    ax.plot(T_uniq, cur_X_fit_data.flatten(), "b")
                    ax.plot(T_uniq, cur_X_data.flatten(), "k--")

                ax.set_title(gene_name + " (" + title_[j] + ")")
            else:
                ax.plot(T_uniq, cur_X_fit_data.T)
                ax.plot(T_uniq, cur_X_data.T, "k--")
                ax.legend(['ul', 'sl'])
                ax.set_title(gene_name)

            if true_param_prefix is not None:
                if has_splicing:
                    ax.plot(t, true_p[j], "r")
                else:
                    ax.plot(t, true_p[j], "r")
            ax.set_xlabel("time (" + unit + ")")
            if y_log_scale:
                ax.set_yscale("log")
            if log_unnormalized:
                ax.set_ylabel("Expression (log)")
            else:
                ax.set_ylabel("Expression")

    return gs


def plot_kin_sto(adata, genes, has_splicing, use_smoothed, log_unnormalized,
                 t, T, T_uniq, unit, X_data, X_fit_data, logLL, true_p,
                 grp_len, sub_plot_n, ncols, boxwidth, gs, fig_mat, gene_order, y_log_scale,
                 true_param_prefix, true_params, est_params,
                 show_moms_fit, show_variance, show_kin_parameters, ):
    import matplotlib.pyplot as plt
    true_alpha, true_beta, true_gamma = true_params
    alpha, beta, gamma = est_params

    if has_splicing:
        title_ = ["ul", "sl", "ul2", "sl2", "ul_sl"] if show_moms_fit else ["ul", "sl"]

        layers = ['M_ul', 'M_sl', 'M_uu', 'M_su'] if (
                'M_ul' in adata.layers.keys() and use_smoothed) \
            else ['ul', 'sl', 'uu', 'su']

        layer_u = 'M_ul' if ('M_ul' in adata.layers.keys() and use_smoothed) else 'ul'
        layer_s = 'M_sl' if ('M_ul' in adata.layers.keys() and use_smoothed) else 'sl'

        _, X_raw = prepare_data_has_splicing(adata, genes, T,
                                             layer_u=layer_u, layer_s=layer_s, total_layers=layers)
    else:
        title_ = ["new", "M_nn"] if show_moms_fit else ["new"]

        total_layer = 'M_t' if ('M_t' in adata.layers.keys() and use_smoothed) else 'total'

        layer = 'M_n' if ('M_n' in adata.layers.keys() and use_smoothed) else 'new'
        _, X_raw = prepare_data_no_splicing(adata, adata.var.index, T, layer=layer,
                                            total_layer=total_layer)

    padding = 0.185 if not show_variance else 0
    for i, gene_name in enumerate(genes):
        cur_X_data, cur_X_fit_data, cur_logLL = X_data[i], X_fit_data[i], logLL[i]

        for j in range(sub_plot_n):
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
                ax.text(0.01 + padding, 0.80, r'$logLL={0:.2f}$'.format(cur_logLL) + ' \n'
                        + r"$t_{1/2} = $" + "{0:.2f}".format(np.log(2) / gamma[i]) + unit[0] \
                        , ha='left', va='top', transform=ax.transAxes)
                if show_kin_parameters:
                    if true_param_prefix is not None:
                        if has_splicing:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\alpha$"
                                + ": {0:.2f}; ".format(true_alpha[i])
                                + r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\beta$"
                                + ": {0:.2f}; ".format(true_beta[i])
                                + r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta[i])
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\alpha$"
                                + ": {0:.2f}; ".format(true_alpha[i])
                                + r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                    else:
                        if has_splicing:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
            max_box_plots = 2 if has_splicing else 1
            # if show_variance first plot box plot
            if show_variance:
                if j < max_box_plots:
                    if has_splicing:
                        Obs = X_raw[i][j][0].A.flatten() if issparse(X_raw[i][j][0]) else X_raw[i][j][0].flatten()
                    else:
                        Obs = X_raw[i].A.flatten() if issparse(X_raw[i][0]) else X_raw[i].flatten()

                    ax.boxplot(
                        x=[Obs[T == std] for std in T_uniq],
                        positions=T_uniq,
                        widths=boxwidth,
                        showfliers=False,
                        showmeans=True,
                    )
                    ax.plot(T_uniq, cur_X_fit_data[j].T, "b")
                    ax.plot(T_uniq, cur_X_data[j], "k--")
                    ax.set_title(gene_name + " (" + title_[j] + ")")
            # if not show_variance then first plot line plot
            else:
                if j == 0:
                    if has_splicing:
                        ax.plot(T_uniq, cur_X_fit_data[[0, 1]].T)
                        ax.plot(T_uniq, cur_X_data[[0, 1]].T, "k--")
                        ax.legend(['ul', 'sl'])
                    else:
                        ax.plot(T_uniq, cur_X_fit_data[j].T)
                        ax.plot(T_uniq, cur_X_data[j].T, "k--")
                        ax.legend(['new'])
                    ax.set_title(gene_name)

            # other subplots
            if not ((show_variance and j < max_box_plots) or
                    (not show_variance and j == 0)):
                ax.plot(T_uniq, cur_X_fit_data[j].T)
                ax.plot(T_uniq, cur_X_data[j], "k--")
                if show_variance:
                    ax.legend([title_[j]])
                else:
                    if has_splicing:
                        ax.legend([title_[j + 1]])
                    else:
                        ax.legend([title_[j]])

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

    return gs


def plot_kin_mix(adata, genes, has_splicing, use_smoothed, log_unnormalized,
                 t, T, T_uniq, unit, X_data, X_fit_data, logLL, true_p,
                 grp_len, sub_plot_n, ncols, boxwidth, gs, fig_mat, gene_order, y_log_scale,
                 true_param_prefix, true_params, est_params,
                 show_variance, show_kin_parameters, ):
    import matplotlib.pyplot as plt
    true_alpha, true_beta, true_gamma = true_params
    alpha, beta, gamma = est_params

    if has_splicing:
        title_ = ["ul", "sl", "uu", "su"]

        layers = ['M_ul', 'M_sl', 'M_uu', 'M_su'] if (
                'M_ul' in adata.layers.keys() and use_smoothed) \
            else ['ul', 'sl', 'uu', 'su']

        layer_u = 'M_ul' if ('M_ul' in adata.layers.keys() and use_smoothed) else 'ul'
        layer_s = 'M_sl' if ('M_ul' in adata.layers.keys() and use_smoothed) else 'sl'

        _, X_raw = prepare_data_has_splicing(adata, genes, T,
                                             layer_u=layer_u, layer_s=layer_s, total_layers=layers)
    else:
        title_ = ["new", "old"]

        total_layer = 'M_t' if ('M_t' in adata.layers.keys() and use_smoothed) else 'total'

        layer = 'M_n' if ('M_n' in adata.layers.keys() and use_smoothed) else 'new'
        _, X_raw = prepare_data_no_splicing(adata, adata.var.index, T, layer=layer,
                                            total_layer=total_layer, return_old=True)

    padding = 0.185 if not show_variance else 0
    for i, gene_name in enumerate(genes):
        cur_X_data, cur_X_fit_data, cur_logLL = X_data[i], X_fit_data[i], logLL[i]

        for j in range(sub_plot_n):
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
                ax.text(0.01 + padding, 0.80, r'$logLL={0:.2f}$'.format(cur_logLL) + ' \n'
                        + r"$t_{1/2} = $" + "{0:.2f}".format(np.log(2) / gamma[i]) + unit[0] \
                        , ha='left', va='top', transform=ax.transAxes)
                if show_kin_parameters:
                    if true_param_prefix is not None:
                        if has_splicing:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\alpha$"
                                + ": {0:.2f}; ".format(true_alpha[i])
                                + r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\beta$"
                                + ": {0:.2f}; ".format(true_beta[i])
                                + r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta[i])
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\alpha$"
                                + ": {0:.2f}; ".format(true_alpha[i])
                                + r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                    else:
                        if has_splicing:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
            max_box_plots = 2 if has_splicing else 1
            max_line_plots = 2 if has_splicing else 1
            # if show_variance first plot box plot
            if show_variance:
                if j < max_box_plots:
                    if has_splicing:
                        Obs = X_raw[i][j][0].A.flatten() if issparse(X_raw[i][j][0]) else X_raw[i][j][0].flatten()
                    else:
                        Obs = X_raw[i][j][0].A.flatten() if issparse(X_raw[i][j][0]) else X_raw[i][j][0].flatten()

                    ax.boxplot(
                        x=[Obs[T == std] for std in T_uniq],
                        positions=T_uniq,
                        widths=boxwidth,
                        showfliers=False,
                        showmeans=True,
                    )
                    ax.plot(T_uniq, cur_X_fit_data[j].T, "b")
                    ax.plot(T_uniq, cur_X_data[j], "k--")
                    ax.set_title(gene_name + " (" + title_[j] + ")")
            # if not show_variance then first plot line plot
            else:
                if has_splicing:
                    if j == 0:
                        ax.plot(T_uniq, cur_X_fit_data[[0, 1]].T)
                        ax.plot(T_uniq, cur_X_data[[0, 1]].T, "k--")
                        ax.legend(['ul', 'sl'])
                    elif j == 1:
                        ax.plot(T_uniq, cur_X_fit_data[[2, 3]].T)
                        ax.plot(T_uniq, cur_X_data[[2, 3]].T, "k--")
                        ax.legend(['uu', 'su'])
                    ax.set_title(gene_name)
                else:
                    if j == 0:
                        ax.plot(T_uniq, cur_X_fit_data[[0, 1]].T)
                        ax.plot(T_uniq, cur_X_data[[0, 1]].T, "k--")
                        ax.legend(['new', "old"])
                    ax.set_title(gene_name)
            # other subplots
            if not ((show_variance and j < max_box_plots) or
                        (not show_variance and j < max_line_plots)):
                ax.plot(T_uniq, cur_X_fit_data[j].T)
                ax.plot(T_uniq, cur_X_data[j], "k--")
                if show_variance:
                    ax.legend([title_[j]])
                else:
                    if has_splicing:
                        ax.legend([title_[j + 2]])
                    else:
                        ax.legend([title_[j + 1]])

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

    return gs

def plot_kin_mix_det_sto(adata, genes, has_splicing, use_smoothed, log_unnormalized,
                 t, T, T_uniq, unit, X_data, X_fit_data, logLL, true_p,
                 grp_len, sub_plot_n, ncols, boxwidth, gs, fig_mat, gene_order, y_log_scale,
                 true_param_prefix, true_params, est_params,
                 show_moms_fit, show_variance, show_kin_parameters, ):
    import matplotlib.pyplot as plt
    true_alpha, true_beta, true_gamma = true_params
    alpha, beta, gamma = est_params

    if has_splicing:
        title_ = ["ul", "sl", "uu", "su", "uu2", "su2", "uu_su"] if show_moms_fit else ["ul", "sl", "uu", "su"]

        layers = ['M_ul', 'M_sl', 'M_uu', 'M_su'] if (
                'M_ul' in adata.layers.keys() and use_smoothed) \
            else ['ul', 'sl', 'uu', 'su']

        _, X_raw = prepare_data_mix_has_splicing(adata, adata.var.index, T, layer_u=layers[2],
                                                 layer_s=layers[3], layer_ul=layers[0], layer_sl=layers[1],
                                                 total_layers=layers, mix_model_indices=[0, 1, 5, 6, 7, 8, 9])
    else:
        title_ = ["new", "old", "o2"] if show_moms_fit else ["new", "old"]
        layers = ['M_n', 'M_t'] if ('M_n' in adata.layers.keys() and use_smoothed) \
            else ['new', 'total']
        total_layer = 'M_t' if ('M_t' in adata.layers.keys() and use_smoothed) else 'total'

        _, X_raw = prepare_data_mix_no_splicing(adata, adata.var.index, T,
                                                        layer_n=layers[0], layer_t=layers[1], total_layer=total_layer,
                                                        mix_model_indices=[0, 2, 3])

    padding = 0.185 if not show_variance else 0
    for i, gene_name in enumerate(genes):
        cur_X_data, cur_X_fit_data, cur_logLL = X_data[i], X_fit_data[i], logLL[i]

        for j in range(sub_plot_n):
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
                ax.text(0.01 + padding, 0.80, r'$logLL={0:.2f}$'.format(cur_logLL) + ' \n'
                        + r"$t_{1/2} = $" + "{0:.2f}".format(np.log(2) / gamma[i]) + unit[0] \
                        , ha='left', va='top', transform=ax.transAxes)
                if show_kin_parameters:
                    if true_param_prefix is not None:
                        if has_splicing:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\alpha$"
                                + ": {0:.2f}; ".format(true_alpha[i])
                                + r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\beta$"
                                + ": {0:.2f}; ".format(true_beta[i])
                                + r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta[i])
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\alpha$"
                                + ": {0:.2f}; ".format(true_alpha[i])
                                + r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                    else:
                        if has_splicing:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )

            max_box_plots = 4 if has_splicing else 2
            max_line_plots = 2 if has_splicing else 1
            # if show_variance first plot box plot
            if show_variance:
                if j < max_box_plots:
                    if has_splicing:
                        Obs = X_raw[i][j][0].A.flatten() if issparse(X_raw[i][j][0]) else X_raw[i][j][0].flatten()
                    else:
                        Obs = X_raw[i][j][0].A.flatten() if issparse(X_raw[i][j][0]) else X_raw[i][j][0].flatten()

                    ax.boxplot(
                        x=[Obs[T == std] for std in T_uniq],
                        positions=T_uniq,
                        widths=boxwidth,
                        showfliers=False,
                        showmeans=True,
                    )
                    ax.plot(T_uniq, cur_X_fit_data[j].T, "b")
                    ax.plot(T_uniq, cur_X_data[j], "k--")
                    ax.set_title(gene_name + " (" + title_[j] + ")")
            # if not show_variance then first plot line plot
            else:
                if has_splicing:
                    if j == 0:
                        ax.plot(T_uniq, cur_X_fit_data[[0, 1]].T)
                        ax.plot(T_uniq, cur_X_data[[0, 1]].T, "k--")
                        ax.legend(['ul', 'sl'])
                    elif j == 1:
                        ax.plot(T_uniq, cur_X_fit_data[[2, 3]].T)
                        ax.plot(T_uniq, cur_X_data[[2, 3]].T, "k--")
                        ax.legend(['uu', 'su'])
                    ax.set_title(gene_name)
                else:
                    if j == 0:
                        ax.plot(T_uniq, cur_X_fit_data[[0, 1]].T)
                        ax.plot(T_uniq, cur_X_data[[0, 1]].T, "k--")
                        ax.legend(['new', "old"])
                    ax.set_title(gene_name)
            # other subplots
            if not ((show_variance and j < max_box_plots) or
                            (not show_variance and j < max_line_plots)):
                ax.plot(T_uniq, cur_X_fit_data[j].T)
                ax.plot(T_uniq, cur_X_data[j], "k--")
                if show_variance:
                    ax.legend([title_[j]])
                else:
                    if has_splicing:
                        ax.legend([title_[j + 2]])
                    else:
                        ax.legend([title_[j + 1]])

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
    return gs


def plot_kin_mix_sto_sto(adata, genes, has_splicing, use_smoothed, log_unnormalized,
                 t, T, T_uniq, unit, X_data, X_fit_data, logLL, true_p,
                 grp_len, sub_plot_n, ncols, boxwidth, gs, fig_mat, gene_order, y_log_scale,
                 true_param_prefix, true_params, est_params,
                 show_moms_fit, show_variance, show_kin_parameters, ):
    import matplotlib.pyplot as plt
    true_alpha, true_beta, true_gamma = true_params
    alpha, beta, gamma = est_params

    if has_splicing:
        title_ = ["ul", "sl", "uu", "su",  "ul2", "sl2", "ul_sl", "uu2", "su2", "uu_su"] \
            if show_moms_fit else ["ul", "sl", "uu", "su"]

        layers = ['M_ul', 'M_sl', 'M_uu', 'M_su'] if (
                'M_ul' in adata.layers.keys() and use_smoothed) \
            else ['ul', 'sl', 'uu', 'su']

        reorder_inds = [0, 1, 5, 6, 2, 3, 4, 7, 8, 9]
        _, X_raw = prepare_data_mix_has_splicing(adata, adata.var.index, T, layer_u=layers[2],
                                                 layer_s=layers[3], layer_ul=layers[0], layer_sl=layers[1],
                                                 total_layers=layers, mix_model_indices=reorder_inds)
    else:
        title_ = ["new", "old", "n2", "o2"] if show_moms_fit else ["new", "old"]

        total_layer = 'M_t' if ('M_t' in adata.layers.keys() and use_smoothed) else 'total'

        layers = ['M_n', 'M_t'] if ('M_n' in adata.layers.keys() and use_smoothed) \
            else ['new', 'total']

        reorder_inds = [0, 2, 1, 3]
        _, X_raw = prepare_data_mix_no_splicing(adata, adata.var.index, T,
                                                layer_n=layers[0], layer_t=layers[1], total_layer=total_layer,
                                                mix_model_indices=reorder_inds)

    padding = 0.185 if not show_variance else 0
    for i, gene_name in enumerate(genes):
        cur_X_data, cur_X_fit_data, cur_logLL = X_data[i], X_fit_data[i], logLL[i]

        cur_X_fit_data = cur_X_fit_data[reorder_inds]
        cur_X_data = cur_X_data[reorder_inds]

        for j in range(sub_plot_n):
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
                ax.text(0.01 + padding, 0.80, r'$logLL={0:.2f}$'.format(cur_logLL) + ' \n'
                        + r"$t_{1/2} = $" + "{0:.2f}".format(np.log(2) / gamma[i]) + unit[0] \
                        , ha='left', va='top', transform=ax.transAxes)
                if show_kin_parameters:
                    if true_param_prefix is not None:
                        if has_splicing:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\alpha$"
                                + ": {0:.2f}; ".format(true_alpha[i])
                                + r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\beta$"
                                + ": {0:.2f}; ".format(true_beta[i])
                                + r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta[i])
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\alpha$"
                                + ": {0:.2f}; ".format(true_alpha[i])
                                + r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                    else:
                        if has_splicing:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.01 + padding,
                                0.99,
                                r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
            max_box_plots = 4 if has_splicing else 2
            max_line_plots = 2 if has_splicing else 1
            # if show_variance first plot box plot
            if show_variance:
                if j < max_box_plots:
                    if has_splicing:
                        Obs = X_raw[i][j][0].A.flatten() if issparse(X_raw[i][j][0]) else X_raw[i][j][0].flatten()
                    else:
                        Obs = X_raw[i][j][0].A.flatten() if issparse(X_raw[i][j][0]) else X_raw[i][j][0].flatten()

                    ax.boxplot(
                        x=[Obs[T == std] for std in T_uniq],
                        positions=T_uniq,
                        widths=boxwidth,
                        showfliers=False,
                        showmeans=True,
                    )
                    ax.plot(T_uniq, cur_X_fit_data[j].T, "b")
                    ax.plot(T_uniq, cur_X_data[j], "k--")
                    ax.set_title(gene_name + " (" + title_[j] + ")")
            # if not show_variance then first plot line plot
            else:
                if has_splicing:
                    if j == 0:
                        ax.plot(T_uniq, cur_X_fit_data[[0, 1]].T)
                        ax.plot(T_uniq, cur_X_data[[0, 1]].T, "k--")
                        ax.legend(['ul', 'sl'])
                    elif j == 1:
                        ax.plot(T_uniq, cur_X_fit_data[[2, 3]].T)
                        ax.plot(T_uniq, cur_X_data[[2, 3]].T, "k--")
                        ax.legend(['uu', 'su'])
                    ax.set_title(gene_name)
                else:
                    if j == 0:
                        ax.plot(T_uniq, cur_X_fit_data[[0, 1]].T)
                        ax.plot(T_uniq, cur_X_data[[0, 1]].T, "k--")
                        ax.legend(['new', "old"])
                    ax.set_title(gene_name)
            # other subplots
            if not ((show_variance and j < max_box_plots) or
                        (not show_variance and j < max_line_plots)):
                ax.plot(T_uniq, cur_X_fit_data[j].T)
                ax.plot(T_uniq, cur_X_data[j], "k--")
                if show_variance:
                    ax.legend([title_[j]])
                else:
                    if has_splicing:
                        ax.legend([title_[j + 2]])
                    else:
                        ax.legend([title_[j + 1]])

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

    return gs

def plot_deg_det(adata, genes, has_splicing, use_smoothed, log_unnormalized,
                 t, T, T_uniq, unit, X_data, X_fit_data, logLL, true_p,
                 grp_len, sub_plot_n, ncols, boxwidth, gs, fig_mat, gene_order, y_log_scale,
                 true_param_prefix, true_params, est_params,
                 show_variance, show_kin_parameters, ):
    import matplotlib.pyplot as plt
    true_alpha, true_beta, true_gamma = true_params
    alpha, beta, gamma = est_params

    if has_splicing:
        title_ = ["ul", "sl"]

        layers = ['M_ul', 'M_sl', 'M_uu', 'M_su'] if (
                'M_ul' in adata.layers.keys() and use_smoothed) \
            else ['ul', 'sl', 'uu', 'su']

        layer_u = 'M_ul' if ('M_ul' in adata.layers.keys() and use_smoothed) else 'ul'
        layer_s = 'M_sl' if ('M_ul' in adata.layers.keys() and use_smoothed) else 'sl'

        _, X_raw = prepare_data_has_splicing(adata, genes, T,
                                             layer_u=layer_u, layer_s=layer_s, total_layers=layers)
    else:
        title_ = ["new"]

        total_layer = 'M_t' if ('M_t' in adata.layers.keys() and use_smoothed) else 'total'

        layer = 'M_n' if ('M_n' in adata.layers.keys() and use_smoothed) else 'new'
        _, X_raw = prepare_data_no_splicing(adata, adata.var.index, T, layer=layer,
                                            total_layer=total_layer)

    for i, gene_name in enumerate(genes):
        cur_X_data, cur_X_fit_data, cur_logLL = X_data[i], X_fit_data[i], logLL[i]

        for j in range(sub_plot_n):

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
                ax.text(0.65, 0.80, r'$logLL={0:.2f}$'.format(cur_logLL) + ' \n'
                        + r"$t_{1/2} = $" + "{0:.2f}".format(np.log(2) / gamma[i]) + unit[0] \
                        , ha='left', va='top', transform=ax.transAxes)
                if show_kin_parameters:
                    if true_param_prefix is not None:
                        if has_splicing:
                            ax.text(
                                0.65,
                                0.99,
                                # r"$\alpha$"
                                # + ": {0:.2f}; ".format(true_alpha[i])
                                # + r"$\hat \alpha$"
                                # + ": {0:.2f} \n".format(alpha[i])
                                r"$\beta$"
                                + ": {0:.2f}; ".format(true_beta[i])
                                + r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta[i])
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.65,
                                0.99,
                                # r"$\alpha$"
                                # + ": {0:.2f}; ".format(true_alpha[i])
                                # + r"$\hat \alpha$"
                                ": {0:.2f} \n".format(alpha[i])
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                    else:
                        if has_splicing:
                            ax.text(
                                0.65,
                                0.99,
                                # r"$\hat \alpha$"
                                # + ": {0:.2f} \n".format(alpha[i])
                                r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.65,
                                0.99,
                                # r"$\hat \alpha$"
                                # + ": {0:.2f} \n".format(alpha[i])
                                r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
            if show_variance:
                if has_splicing:
                    Obs = X_raw[i][j][0].A.flatten() if issparse(X_raw[i][j][0]) else X_raw[i][j][0].flatten()
                else:
                    Obs = X_raw[i].A.flatten() if issparse(X_raw[i][0]) else X_raw[i].flatten()
                ax.boxplot(
                    x=[Obs[T == std] for std in T_uniq],
                    positions=T_uniq,
                    widths=boxwidth,
                    showfliers=False,
                    showmeans=True,
                )
                if has_splicing:
                    ax.plot(T_uniq, cur_X_fit_data[j], "b")
                    ax.plot(T_uniq, cur_X_data[j], "k--")
                else:
                    ax.plot(T_uniq, cur_X_fit_data.flatten(), "b")
                    ax.plot(T_uniq, cur_X_data.flatten(), "k--")

                ax.set_title(gene_name + " (" + title_[j] + ")")
            else:
                ax.plot(T_uniq, cur_X_fit_data.T)
                ax.plot(T_uniq, cur_X_data.T, "k--")
                ax.legend(['ul', 'sl'])
                ax.set_title(gene_name)

            if true_param_prefix is not None:
                if has_splicing:
                    ax.plot(t, true_p[j], "r")
                else:
                    ax.plot(t, true_p[j], "r")
            ax.set_xlabel("time (" + unit + ")")
            if y_log_scale:
                ax.set_yscale("log")
            if log_unnormalized:
                ax.set_ylabel("Expression (log)")
            else:
                ax.set_ylabel("Expression")
    return gs

def plot_deg_sto(adata, genes, has_splicing, use_smoothed, log_unnormalized,
                 t, T, T_uniq, unit, X_data, X_fit_data, logLL, true_p,
                 grp_len, sub_plot_n, ncols, boxwidth, gs, fig_mat, gene_order, y_log_scale,
                 true_param_prefix, true_params, est_params,
                 show_moms_fit, show_variance, show_kin_parameters, ):
    import matplotlib.pyplot as plt
    true_alpha, true_beta, true_gamma = true_params
    alpha, beta, gamma = est_params

    if has_splicing:
        title_ = ["ul", "sl", "ul2", "sl2", "ul_sl"] if show_moms_fit else ["ul", "sl"]

        layers = ['M_ul', 'M_sl', 'M_uu', 'M_su'] if (
                'M_ul' in adata.layers.keys() and use_smoothed) \
            else ['ul', 'sl', 'uu', 'su']

        layer_u = 'M_ul' if ('M_ul' in adata.layers.keys() and use_smoothed) else 'ul'
        layer_s = 'M_sl' if ('M_ul' in adata.layers.keys() and use_smoothed) else 'sl'

        _, X_raw = prepare_data_has_splicing(adata, genes, T,
                                             layer_u=layer_u, layer_s=layer_s, total_layers=layers)
    else:
        title_ = ["new", "M_nn"] if show_moms_fit else ["new"]

        total_layer = 'M_t' if ('M_t' in adata.layers.keys() and use_smoothed) else 'total'

        layer = 'M_n' if ('M_n' in adata.layers.keys() and use_smoothed) else 'new'
        _, X_raw = prepare_data_no_splicing(adata, adata.var.index, T, layer=layer,
                                            total_layer=total_layer)

    for i, gene_name in enumerate(genes):
        cur_X_data, cur_X_fit_data, cur_logLL = X_data[i], X_fit_data[i], logLL[i]

        for j in range(sub_plot_n):
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
                ax.text(0.65, 0.80,  r'$logLL={0:.2f}$'.format(cur_logLL) + ' \n'
                        + r"$t_{1/2} = $" + "{0:.2f}".format(np.log(2) / gamma[i]) + unit[0] \
                        , ha='left', va='top', transform=ax.transAxes)

                if show_kin_parameters:
                    if true_param_prefix is not None:
                        if has_splicing:
                            ax.text(
                                0.65,
                                0.99,
                                # r"$\alpha$"
                                # + ": {0:.2f}; ".format(true_alpha[i])
                                # + r"$\hat \alpha$"
                                # + ": {0:.2f} \n".format(alpha[i])
                                r"$\beta$"
                                + ": {0:.2f}; ".format(true_beta[i])
                                + r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta[i])
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.65,
                                0.99,
                                # r"$\alpha$"
                                # + ": {0:.2f}; ".format(true_alpha[i])
                                # + r"$\hat \alpha$"
                                ": {0:.2f} \n".format(alpha[i])
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                    else:
                        if has_splicing:
                            ax.text(
                                0.65,
                                0.99,
                                # r"$\hat \alpha$"
                                # + ": {0:.2f} \n".format(alpha[i])
                                r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta[i])
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.65,
                                0.99,
                                # r"$\hat \alpha$"
                                # + ": {0:.2f} \n".format(alpha[i])
                                r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma[i]),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
            max_box_plots = 2 if has_splicing else 1
            # if show_variance first plot box plot
            if show_variance:
                if j < max_box_plots:
                    if has_splicing:
                        Obs = X_raw[i][j][0].A.flatten() if issparse(X_raw[i][j][0]) else X_raw[i][j][0].flatten()
                    else:
                        Obs = X_raw[i].A.flatten() if issparse(X_raw[i][0]) else X_raw[i].flatten()

                    ax.boxplot(
                        x=[Obs[T == std] for std in T_uniq],
                        positions=T_uniq,
                        widths=boxwidth,
                        showfliers=False,
                        showmeans=True,
                    )
                    ax.plot(T_uniq, cur_X_fit_data[j].T, "b")
                    ax.plot(T_uniq, cur_X_data[j], "k--")
                    ax.set_title(gene_name + " (" + title_[j] + ")")
            # if not show_variance then first plot line plot
            else:
                if j == 0:
                    if has_splicing:
                        ax.plot(T_uniq, cur_X_fit_data[[0, 1]].T)
                        ax.plot(T_uniq, cur_X_data[[0, 1]].T, "k--")
                        ax.legend(['ul', 'sl'])
                    else:
                        ax.plot(T_uniq, cur_X_fit_data[j].T)
                        ax.plot(T_uniq, cur_X_data[j].T, "k--")
                        ax.legend(['new'])
                    ax.set_title(gene_name)
            # other subplots
            if not ((show_variance and j < max_box_plots) or
                    (not show_variance and j == 0)):
                ax.plot(T_uniq, cur_X_fit_data[j].T)
                ax.plot(T_uniq, cur_X_data[j], "k--")
                if show_variance:
                    ax.legend([title_[j]])
                else:
                    if has_splicing:
                        ax.legend([title_[j + 1]])
                    else:
                        ax.legend([title_[j]])

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

    return gs
