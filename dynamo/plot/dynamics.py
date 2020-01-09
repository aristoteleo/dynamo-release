import numpy as np
import pandas as pd
import sys
import warnings
from scipy.sparse import issparse
from .utilities import despline
from .scatters import scatters
from ..tools.velocity import sol_u, sol_s, solve_first_order_deg
from ..tools.utils_moments import moments


def phase_portraits(adata, genes, x=0, y=1, mode='splicing', vkey='S', ekey='X', basis='umap', color=None, figsize=None, \
                    **kwargs):
    """Draw the phase portrait, velocity, expression values on the low dimensional embedding.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    genes: `list`
        A list of gene names that are going to be visualized.
    x: `int` (default: `0`)
            The column index of the low dimensional embedding for the x-axis
    y: `int` (default: `1`)
            The column index of the low dimensional embedding for the y-axis
    mode: `string` (default: labelling)
        Which mode of data do you want to show, can be one of `labelling`, `splicing` and `full`.
    vkey: `string` (default: velocity)
        Which velocity key used for visualizing the magnitude of velocity. Can be either velocity in the layers slot or the
        keys in the obsm slot.
    ekey: `str`
        The layer of data to represent the gene expression level.
    basis: `string` (default: umap)
        Which low dimensional embedding will be used to visualize the cell.
    color: `string` (default: None)
        Which group will be used to color cells, only used for the phase portrait because the other two plots are colored
        by the velocity magnitude or the gene expression value, respectively.
    figsize: `None` or `[float, float]` (default: None)
            The width and height of a figure.
    **kwargs:
            Additional parameters that will be passed to plt.scatter function

    Returns
    -------
        A matplotlib plot that shows 1) the phase portrait of each category used in velocity embedding, cells' low dimensional
        embedding, colored either by 2) the gene expression level or 3) the velocity magnitude values.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    # sns.set_style('ticks')

    scatter_kwargs = dict(alpha=0.4, s=8, edgecolor=(0, 0, 0, 1), lw=0.15)
    if kwargs is not None:
        scatter_kwargs.update(kwargs)

    # there is no solution for combining multiple plot in the same figure in plotnine, so a pure matplotlib is used
    # see more at https://github.com/has2k1/plotnine/issues/46
    genes, idx = adata.var.index[adata.var.index.isin(genes)].tolist(), np.where(adata.var.index.isin(genes))[0]
    if len(genes) == 0:
        raise Exception('adata has no genes listed in your input gene vector: {}'.format(genes))
    if not 'X_' + basis in adata.obsm.keys():
        raise Exception('{} is not applied to adata.'.format(basis))
    else:
        embedding = pd.DataFrame({basis + '_0': adata.obsm['X_' + basis][:, x], \
                                  basis + '_1': adata.obsm['X_' + basis][:, y]})
        embedding.columns = ['dim_1', 'dim_2']

    if not (mode in ['labelling', 'splicing', 'full']):
        raise Exception('mode can be only one of the labelling, splicing or full')

    layers = list(adata.layers.keys())
    layers.extend(['X', 'protein', 'X_protein'])
    if ekey in layers:
        if ekey is 'X':
            E_vec = adata[:, genes].X
        elif ekey in ['protein', 'X_protein']:
            E_vec = adata[:, genes].obsm[ekey]
        else:
            E_vec = adata[:, genes].layers[ekey]

    n_cells, n_genes = adata.shape[0], len(genes)

    color_vec=np.repeat(np.nan, n_cells)
    if color is not None:
        color_vec = adata.obs[color].values

    if vkey is 'U':
        V_vec = adata[:, genes].layers['velocity_U']
        if 'velocity_P' in adata.obsm.keys():
            P_vec = adata[:, genes].layer['velocity_P']
    elif vkey is 'S':
        V_vec = adata[:, genes].layers['velocity_S']
        if 'velocity_P' in adata.obsm.keys():
            P_vec = adata[:, genes].layers['velocity_P']
    else:
        raise Exception('adata has no vkey {} in either the layers or the obsm slot'.format(vkey))

    if issparse(E_vec):
        E_vec, V_vec = E_vec.A, V_vec.A

    if 'kinetic_parameter_gamma' in adata.var.columns:
        gamma = adata.var.kinetic_parameter_gamma[genes].values
        velocity_offset = [0] * n_genes if not ("velocity_offset" in adata.var.columns) else \
            adata.var.velocity_offset[genes].values
    else:
        raise Exception('adata does not seem to have velocity_gamma column. Velocity estimation is required before '
                        'running this function.')

    if mode is 'labelling' and all([i in adata.layers.keys() for i in ['new', 'total']]):
        new_mat, tot_mat = adata[:, genes].layers['new'], adata[:, genes].layers['total']
        new_mat, tot_mat = (new_mat.A, tot_mat.A) if issparse(new_mat) else (new_mat, tot_mat)

        df = pd.DataFrame({"new": new_mat.flatten(), "total": tot_mat.flatten(), 'gene': genes * n_cells, 'gamma':
                           np.tile(gamma, n_cells), 'velocity_offset': np.tile(velocity_offset, n_cells),
                           "expression": E_vec.flatten(), "velocity": V_vec.flatten(), 'color': np.repeat(color_vec, n_genes)}, index=range(n_cells * n_genes))

    elif mode is 'splicing' and all([i in adata.layers.keys() for i in ['spliced', 'unspliced']]):
        unspliced_mat, spliced_mat = adata[:, genes].layers['X_unspliced'], adata[:, genes].layers['X_spliced']
        unspliced_mat, spliced_mat = (unspliced_mat.A, spliced_mat.A) if issparse(unspliced_mat) else (unspliced_mat, spliced_mat)

        df = pd.DataFrame({"unspliced": unspliced_mat.flatten(), "spliced": spliced_mat.flatten(), 'gene': genes * n_cells,
                           'gamma': np.tile(gamma, n_cells), 'velocity_offset': np.tile(velocity_offset, n_cells),
                           "expression": E_vec.flatten(), "velocity": V_vec.flatten(), 'color': np.repeat(color_vec, n_genes)}, index=range(n_cells * n_genes))

    elif mode is 'full' and all([i in adata.layers.keys() for i in ['uu', 'ul', 'su', 'sl']]):
        uu, ul, su, sl = adata[:, genes].layers['X_uu'], adata[:, genes].layers['X_ul'], adata[:, genes].layers['X_su'], \
                         adata[:, genes].layers['X_sl']
        if 'protein' in adata.obsm.keys():
            if 'kinetic_parameter_eta' in adata.var.columns:
                gamma_P = adata.var.kinetic_parameter_eta[genes].values
                velocity_offset_P = [0] * n_cells if not ("velocity_offset_P" in adata.var.columns) else \
                    adata.var.velocity_offset_P[genes].values
            else:
                raise Exception(
                    'adata does not seem to have velocity_gamma column. Velocity estimation is required before '
                    'running this function.')

            P = adata[:, genes].obsm['X_protein'] if ['X_protein'] in adata.obsm.keys() else adata[:, genes].obsm['protein']
            uu, ul, su, sl, P = (uu.A, ul.A, su.A, sl.A, P.A) if issparse(uu) else (uu, ul, su, sl, P)
            if issparse(P_vec):
                P_vec = P_vec.A

            # df = pd.DataFrame({"uu": uu.flatten(), "ul": ul.flatten(), "su": su.flatten(), "sl": sl.flatten(), "P": P.flatten(),
            #                    'gene': genes * n_cells, 'prediction': np.tile(gamma, n_cells) * uu.flatten() +
            #                     np.tile(velocity_offset, n_cells), "velocity": genes * n_cells}, index=range(n_cells * n_genes))
            df = pd.DataFrame({"new": (ul + sl).flatten(), "total": (uu + ul + sl + su).flatten(), "S": (sl + su).flatten(), "P": P.flatten(),
                               'gene': genes * n_cells, 'gamma': np.tile(gamma, n_cells), 'velocity_offset': np.tile(velocity_offset, n_cells),
                               'gamma_P': np.tile(gamma_P, n_cells), 'velocity_offset_P': np.tile(velocity_offset_P, n_cells),
                               "expression": E_vec.flatten(), "velocity": V_vec.flatten(), "velocity_protein": P_vec.flatten(), 'color': np.repeat(color_vec, n_genes)}, index=range(n_cells * n_genes))
        else:
            df = pd.DataFrame({"new": (ul + sl).flatten(), "total": (uu + ul + sl + su).flatten(),
                               'gene': genes * n_cells, 'gamma': np.tile(gamma, n_cells), 'velocity_offset': np.tile(velocity_offset, n_cells),
                               "expression": E_vec.flatten(), "velocity": V_vec.flatten(), 'color': np.repeat(color_vec, n_genes)}, index=range(n_cells * n_genes))
    else:
        raise Exception('Your adata is corrupted. Make sure that your layer has keys new, old for the labelling mode, '
                        'spliced, ambiguous, unspliced for the splicing model and uu, ul, su, sl for the full mode')

    n_columns = 6 if ('protein' in adata.obsm.keys() and mode is 'full') else 3
    nrow, ncol = int(np.ceil(n_columns * n_genes / 6)), 6
    if figsize is None:
        plt.figure(None, (3 * ncol, 3 * nrow))  # , dpi=160
    else:
        plt.figure(None, (figsize[0] * ncol, figsize[1] * nrow))  # , dpi=160

    # the following code is inspired by https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb
    gs = plt.GridSpec(nrow, ncol)
    for i, gn in enumerate(genes):
        if n_columns is 3:
            ax1, ax2, ax3 = plt.subplot(gs[i*3]), plt.subplot(gs[i*3+1]), plt.subplot(gs[i*3+2])
        elif n_columns is 6:
            ax1, ax2, ax3, ax4, ax5, ax6 = plt.subplot(gs[i*3]), plt.subplot(gs[i*3+1]), plt.subplot(gs[i*3+2]), \
                    plt.subplot(gs[i * 3 + 3]), plt.subplot(gs[i * 3 + 4]), plt.subplot(gs[i * 3 + 5])
        try:
            ix=np.where(adata.var.index == gn)[0][0]
        except:
            continue
        cur_pd = df.loc[df.gene == gn, :]
        if cur_pd.color.unique() != np.nan:
            sns.scatterplot(cur_pd.iloc[:, 1], cur_pd.iloc[:, 0], hue="expression", ax=ax1, palette="viridis", **scatter_kwargs) # x-axis: S vs y-axis: U
        else:
            sns.scatterplot(cur_pd.iloc[:, 1], cur_pd.iloc[:, 0], hue=color, ax=ax1, palette="Set2", **scatter_kwargs) # x-axis: S vs y-axis: U

        ax1.set_title(gn)
        xnew = np.linspace(0, cur_pd.iloc[:, 1].max())
        ax1.plot(xnew, xnew * cur_pd.loc[:, 'gamma'].unique() + cur_pd.loc[:, 'velocity_offset'].unique(), c="k")
        ax1.set_xlim(0, np.max(cur_pd.iloc[:, 1])*1.02)
        ax1.set_ylim(0, np.max(cur_pd.iloc[:, 0])*1.02)

        despline(ax1) # sns.despline()

        df_embedding = pd.concat([cur_pd, embedding], axis=1)
        V_vec = df_embedding.loc[:, 'velocity']

        limit = np.nanmax(np.abs(np.nanpercentile(V_vec, [1, 99])))  # upper and lowe limit / saturation

        V_vec = V_vec + limit  # that is: tmp_colorandum - (-limit)
        V_vec = V_vec / (2 * limit)  # that is: tmp_colorandum / (limit - (-limit))
        V_vec = np.clip(V_vec, 0, 1)

        cmap = plt.cm.RdBu_r # sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        sns.scatterplot(embedding.iloc[:, 0], embedding.iloc[:, 1], hue=df_embedding.loc[:, 'expression'], ax=ax2, palette=cmap, legend=False, **scatter_kwargs)
        ax2.set_title(gn + '(' + ekey + ')')
        ax2.set_xlabel(basis + '_1')
        ax2.set_ylabel(basis + '_2')
        cmap = plt.cm.Greens # sns.diverging_palette(10, 220, sep=80, as_cmap=True)
        sns.scatterplot(embedding.iloc[:, 0], embedding.iloc[:, 1], hue=V_vec, ax=ax3, palette=cmap, legend=False, **scatter_kwargs)
        ax3.set_title(gn + '(' + vkey + ')')
        ax3.set_xlabel(basis + '_1')
        ax3.set_ylabel(basis + '_2')

        if 'protein' in adata.obsm.keys() and mode is 'full' and all([i in adata.layers.keys() for i in ['uu', 'ul', 'su', 'sl']]):
            sns.scatterplot(cur_pd.iloc[:, 3], cur_pd.iloc[:, 2], hue=color, ax=ax4, **scatter_kwargs) # x-axis: Protein vs. y-axis: Spliced
            ax4.set_title(gn)
            xnew = np.linspace(0, cur_pd.iloc[:, 3].max())
            ax4.plot(xnew, xnew * cur_pd.loc[:, 'gamma_P'].unique() + cur_pd.loc[:, 'velocity_offset_P'].unique(), c="k")
            ax4.set_ylim(0, np.max(cur_pd.iloc[:, 3]) * 1.02)
            ax4.set_xlim(0, np.max(cur_pd.iloc[:, 2]) * 1.02)

            despline(ax4)   # sns.despline()

            V_vec = df_embedding.loc[:, 'velocity_p']

            limit = np.nanmax(np.abs(np.nanpercentile(V_vec, [1, 99])))  # upper and lowe limit / saturation

            V_vec = V_vec + limit  # that is: tmp_colorandum - (-limit)
            V_vec = V_vec / (2 * limit)  # that is: tmp_colorandum / (limit - (-limit))
            V_vec = np.clip(V_vec, 0, 1)

            df_embedding = pd.concat([embedding, cur_pd.loc[:, 'gene']], ignore_index=False)

            cmap = sns.light_palette("navy", as_cmap=True)
            sns.scatterplot(embedding.iloc[:, 0], embedding.iloc[:, 1], hue=df_embedding.loc[:, 'P'], \
                            ax=ax5, legend=False, palette=cmap, **scatter_kwargs)
            ax5.set_title(gn + '(protein expression)')
            ax5.set_xlabel(basis + '_1')
            ax5.set_ylabel(basis + '_2')
            cmap = sns.diverging_palette(145, 280, s=85, l=25, n=7)
            sns.scatterplot(embedding.iloc[:, 0], embedding.iloc[:, 1], hue=V_vec, ax=ax6, legend=False, palette=cmap, **scatter_kwargs)
            ax6.set_title(gn + '(protein velocity)')
            ax6.set_xlabel(basis + '_1')
            ax6.set_ylabel(basis + '_2')

    plt.tight_layout()
    plt.show()


def dynamics(adata, vkey, unit='hours', log_unnormalized=True, y_log_scale=False, group=None, ncols=None,
                           figsize=None, dpi=None, boxwidth=None, barwidth=None, true_param_prefix=None, show=True):
    """Plot the data and fitting of different metabolic labeling experiments.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        vkey: list of `str`
            key for variable or gene names.
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
        barwidth: `float`
            The width of the bar of the barplot.
        true_param_prefix: `str`
            The prefix for the column names of true parameters in the .var attributes. Useful for the simulation data.
        show: `bool` (default: `True`)
            Whether to plot the figure.

    Returns
    -------
        Nothing but plot the figure of the metabolic fitting.
    """

    import matplotlib.pyplot as plt
    if group is not None and group + "_dynamics" in adata.uns_keys():
        uns_key = group + "_dynamics"
        _group, grp_len = np.unique(adata.obs[group]), len(np.unique(adata.obs[group]))
    else:
        uns_key = "dynamics"
        _group, grp_len = ['_all_cells'], 1

    T, group, asspt_mRNA, experiment_type, _, mode, has_splicing, has_labeling, has_protein = adata.uns[uns_key].values()
    if asspt_mRNA is 'ss':
        # run the phase plot
        warnings.warn("dynamics plot doesn't support steady state mode, use phase_portraits function instead.")
        phase_portraits(adata)

    T_uniq = np.unique(T)
    t = np.linspace(0, T_uniq[-1], 1000)
    gene_idx = [np.where(adata.var.index.values == gene)[0][0] for gene in vkey]

    if boxwidth is None and len(T_uniq) > 1:
        boxwidth = 0.8 * (np.diff(T_uniq).min() / 2)

    if barwidth is None and len(T_uniq) > 1:
        barwidth = 0.8 * (np.diff(T_uniq).min() / 2)

    if has_splicing:
        if mode is 'moment':
            sub_plot_n = 4 # number of subplots for each gene
        elif experiment_type is 'kin' or experiment_type is 'deg':
            sub_plot_n = 4
        elif experiment_type is 'one_shot': # just the labeled RNA
            sub_plot_n = 1
        elif experiment_type is 'mix_std_stm':
            sub_plot_n = 5
    else:
        if mode is 'moment':
            sub_plot_n = 2
        elif experiment_type is 'kin'or experiment_type is 'deg':
            sub_plot_n = 2
        elif experiment_type is 'one_shot': # just the labeled RNA
            sub_plot_n = 1
        elif experiment_type is 'mix_std_stm':
            sub_plot_n = 3

    ncols = len(gene_idx) * grp_len if ncols is None else min(len(gene_idx) * grp_len, ncols)
    nrows = int(np.ceil(len(gene_idx) * sub_plot_n * grp_len / ncols))
    figsize = [7, 5] if figsize is None else figsize
    gs = plt.GridSpec(nrows, ncols, plt.figure(None, (figsize[0] * ncols, figsize[1] * nrows), dpi=dpi))

    # we need to visualize gene in row-wise mode
    for grp_idx, cur_grp in enumerate(_group):
        if cur_grp == '_all_cells':
            prefix = ''  # kinetic_parameter_
        else:
            prefix = group + '_' + cur_grp + '_'

        for i, idx in enumerate(gene_idx):
            gene_name = adata.var_names[idx]

            if mode is 'moment':
                a, b, alpha_a, alpha_i, beta, gamma = adata.var.loc[gene_name, \
                 [prefix + 'a', prefix + 'b', prefix + 'alpha_a', prefix + 'alpha_i', prefix + 'beta', prefix + 'gamma']]
                params = {'a': a, 'b': b, 'alpha_a': alpha_a, 'alpha_i': alpha_i, 'beta': beta, 'gamma': gamma} # "la": 1, "si": 0,
                mom = moments(*list(params.values()))
                mom.integrate(t)
                mom_data = mom.get_all_central_moments() if has_splicing else mom.get_nosplice_central_moments()
                if true_param_prefix is not None:
                    true_a, true_b, true_alpha_a, true_alpha_i, true_beta, true_gamma = adata.var.loc[gene_name, \
                                                                        [true_param_prefix + 'a', true_param_prefix + 'b', true_param_prefix + 'alpha_a',
                                                                         true_param_prefix + 'alpha_i', true_param_prefix + 'beta',
                                                                         true_param_prefix + 'gamma']]
                    true_params = {'a': true_a, 'b': true_b, 'alpha_a': true_alpha_a, 'alpha_i': true_alpha_i, 'beta': true_beta,
                              'gamma': true_gamma}  # "la": 1, "si": 0,
                    true_mom = moments(*list(true_params.values()))
                    true_mom.integrate(t)
                    true_mom_data = true_mom.get_all_central_moments() if has_splicing else true_mom.get_nosplice_central_moments()

                # n_mean, n_var = x_data[:2, :], x_data[2:, :]
                if has_splicing:
                    tmp = [adata[:, gene_name].layers['X_ul'].A.T,
                           adata.layers['X_sl'].A.T] if 'X_ul' in adata.layers.keys() else \
                        [adata[:, gene_name].layers['ul'].A.T, adata.layers['sl'].A.T]
                    x_data = [tmp[0].A, tmp[1].A] if issparse(tmp[0]) else tmp
                    if log_unnormalized and 'X_ul' not in adata.layers.keys():
                        x_data = [np.log(tmp[0] + 1), np.log(tmp[1] + 1)]

                    title_ = ['(unspliced labeled)', '(spliced labeled)',
                              '(unspliced labeled)', '(spliced labeled)']
                    Obs_m = [adata.uns['M_ul'], adata.uns['M_sl']]
                    Obs_v = [adata.uns['V_ul'], adata.uns['V_sl']]
                    j_species = 2 # number of species
                else:
                    tmp = adata[:, gene_name].layers['X_new'].T if 'X_new' in adata.layers.keys() else \
                    adata[:, gene_name].layers['new'].T
                    x_data = [tmp.A] if issparse(tmp) else [tmp]

                    if log_unnormalized and 'X_new' not in adata.layers.keys():
                        x_data = [np.log(x_data[0] + 1)]
                    # only use new key for calculation, so we only have M, V
                    title_ = [' (labeled)', ' (labeled)']
                    Obs_m, Obs_v = [adata.uns['M']], [adata.uns['V']]
                    j_species = 1

                for j in range(sub_plot_n):
                    row_ind = int(np.floor(i/ncols)) # make sure all related plots for the same gene in the same column.
                    ax = plt.subplot(gs[(row_ind * sub_plot_n + j) * ncols * grp_len + (i % ncols - 1) * grp_len + 1])
                    if true_param_prefix is not None and j == 0:
                        if has_splicing:
                            ax.text(0.15, 0.92, r'$a^{true}$: {0:.2f} $a^{est}$: {0:.2f} \n'
                                                r'$b^{true}$: {0:.2f} $b^{est}$: {0:.2f} \n'
                                                r'$\alpha_a^{true}$: {0:.2f} $\alpha_a^{est}$: {0:.2f} \n'
                                                r'$\alpha_i^{true}$: {0:.2f} $\alpha_i^{est}$: {0:.2f} \n'
                                                r'$\beta^{true}$: {0:.2f} $\beta^{est}$: {0:.2f} \n'
                                                r'$\gamma^{true}$: {0:.2f} $\gamma^{est}$: {0:.2f} \n'.format(true_a, a, true_b, b, true_alpha_a, alpha_a, true_alpha_i, alpha_i, true_beta, beta, true_gamma, gamma),
                                    ha='right', va='top', transform=ax.transAxes)
                        else:
                            ax.text(0.15, 0.92, r'$a^{true}$: {0:.2f} $a^{est}$: {0:.2f} \n'
                                            r'$b^{true}$: {0:.2f} $b^{est}$: {0:.2f} \n'
                                            r'$\alpha_a^{true}$: {0:.2f} $\alpha_a^{est}$: {0:.2f} \n'
                                            r'$\alpha_i^{true}$: {0:.2f} $\alpha_i^{est}$: {0:.2f} \n'
                                            r'$\gamma^{true}$: {0:.2f} $\gamma^{est}$: {0:.2f} \n'.format(true_a, a, true_b, b, true_alpha_a, alpha_a, true_alpha_i, alpha_i, true_gamma, gamma),
                                ha='right', va='top', transform=ax.transAxes)
                    if j < j_species:
                        ax.boxplot(x=[x_data[j][i][T == std] for std in T_uniq],  positions=T_uniq, widths=boxwidth, showfliers=False, showmeans=True)  # x=T.values, y= # ax1.plot(T, u.T, linestyle='None', marker='o', markersize=10)
                        # ax.scatter(T_uniq, Obs_m[j][i], c='r')  # ax1.plot(T, u.T, linestyle='None', marker='o', markersize=10)
                        if y_log_scale:
                            ax.set_yscale('log')
                        if log_unnormalized:
                            ax.set_ylabel('Expression (log)') #
                        else:
                            ax.set_ylabel('Expression')
                        ax.plot(t, mom_data[j], 'k--')
                        if true_param_prefix is not None: ax.plot(t, true_mom_data[j], 'r--')
                    else:
                        ax.scatter(T_uniq, Obs_v[j - j_species][i]) # , c='r'
                        if y_log_scale:
                            ax.set_yscale('log')
                        if log_unnormalized:
                            ax.set_ylabel('Variance (log expression)') #
                        else:
                            ax.set_ylabel('Variance')
                        ax.plot(t, mom_data[j], 'k--')
                        if true_param_prefix is not None: ax.plot(t, true_mom_data[j], 'r--')

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
                        gene_name, [prefix + 'alpha', prefix + 'beta', prefix + 'gamma', prefix + 'ul0', prefix + 'sl0', prefix + 'uu0', 'half_life']]
                    # $u$ - unlabeled, unspliced
                    # $s$ - unlabeled, spliced
                    # $w$ - labeled, unspliced
                    # $l$ - labeled, spliced
                    #
                    beta = sys.float_info.epsilon if beta == 0 else beta
                    gamma = sys.float_info.epsilon if gamma == 0 else gamma
                    u = sol_u(t, uu0, alpha, beta)
                    su0 = np.mean(su[T == np.min(T)]) # this should also be estimated
                    s = sol_s(t, su0, uu0, alpha, beta, gamma)
                    w = sol_u(t, ul0, 0, beta)
                    l = sol_s(t, sl0, ul0, 0, beta, gamma)
                    if true_param_prefix is not None:
                        true_alpha, true_beta, true_gamma = adata.var.loc[gene_name, [true_param_prefix + 'alpha', true_param_prefix + 'beta', true_param_prefix + 'gamma']]
                        true_u = sol_u(t, uu0, true_alpha, true_beta)
                        true_s = sol_s(t, su0, uu0, true_alpha, true_beta, true_gamma)
                        true_w = sol_u(t, ul0, 0, true_beta)
                        true_l = sol_s(t, sl0, ul0, 0, true_beta, true_gamma)

                        true_p = np.vstack((true_u, true_w, true_s, true_l))

                    title_ = ['(unspliced unlabeled)', '(unspliced labeled)', '(spliced unlabeled)', '(spliced labeled)']

                    Obs, Pred = np.vstack((uu, ul, su, sl)), np.vstack((u, w, s, l))
                else:
                    layers = ['X_new', 'X_total'] if 'X_new' in adata.layers.keys() else ['new', 'total']
                    uu, ul = adata[:, gene_name].layers[layers[1]] - adata[:, gene_name].layers[layers[0]], \
                                     adata[:, gene_name].layers[layers[0]]
                    uu, ul = (uu.toarray().squeeze(), ul.toarray().squeeze()) if issparse(uu) else (uu.squeeze(), ul.squeeze())

                    if log_unnormalized and layers == ['new', 'total']:
                        uu, ul = np.log(uu + 1), np.log(ul + 1)

                    alpha, gamma, uu0, ul0, half_life = adata.var.loc[gene_name, [prefix + 'alpha', prefix + 'gamma', prefix + 'uu0', prefix + 'ul0', 'half_life']]

                    # require no beta functions
                    u = sol_u(t, uu0, alpha, gamma)
                    s = None # sol_s(t, su0, uu0, 0, 1, gamma)
                    w = sol_u(t, ul0, 0, gamma)
                    l = None # sol_s(t, 0, 0, alpha, 1, gamma)
                    title_ = ['(unlabeled)', '(labeled)']
                    if true_param_prefix is not None:
                        true_alpha, true_gamma = adata.var.loc[gene_name, [true_param_prefix + 'alpha', true_param_prefix + 'gamma']]
                        true_u = sol_u(t, uu0, true_alpha, true_gamma)
                        true_w = sol_u(t, ul0, 0, true_gamma)

                        true_p = np.vstack((true_u, true_w))

                    Obs, Pred = np.vstack((uu, ul)), np.vstack((u, w))

                for j in range(sub_plot_n):
                    row_ind = int(np.floor(i/ncols)) # make sure unlabled and labeled are in the same column.
                    ax = plt.subplot(gs[(row_ind * sub_plot_n + j) * ncols * grp_len + (i % ncols - 1) * grp_len + 1])
                    if true_param_prefix is not None and j == 0:
                        if has_splicing:
                            ax.text(0.75, 0.70, r'$\alpha^{true}$: {0:.2f} $\alpha^{est}$: {0:.2f} \n'
                                                r'$\beta^{true}$: {0:.2f} $\beta^{est}$: {0:.2f} \n'
                                                r'$\gamma^{true}$: {0:.2f} $\gamma^{est}$: {0:.2f} \n'.format(true_alpha, alpha, true_beta, beta, true_gamma, gamma),
                                    ha='right', va='top', transform=ax.transAxes)
                        else:
                            ax.text(0.75, 0.80, r'$\alpha^{true}$: {0:.2f} $\alpha^{est}$: {0:.2f} \n'
                                                r'$\gamma^{true}$: {0:.2f} $\gamma^{est}$: {0:.2f} \n'.format(true_alpha, alpha, true_gamma, gamma),
                                    ha='right', va='top', transform=ax.transAxes)

                    ax.boxplot(x=[Obs[j][T == std] for std in T_uniq], positions=T_uniq, widths=boxwidth,
                               showfliers=False, showmeans=True)
                    ax.plot(t, Pred[j], 'k--')
                    if true_param_prefix is not None: ax.plot(t, true_p[j], 'r--')
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
                    if true_param_prefix is not None:
                        true_alpha, true_beta, true_gamma = adata.var.loc[gene_name, [true_param_prefix + 'alpha', true_param_prefix + 'beta', true_param_prefix + 'gamma']]
                        true_u = sol_u(t, uu0, 0, true_beta)
                        true_s = sol_s(t, su0, uu0, 0, true_beta, true_gamma)
                        true_w = sol_u(t, 0, true_alpha, true_beta)
                        true_l = sol_s(t, 0, 0, true_alpha, true_beta, true_gamma)

                        true_p = np.vstack((true_u, true_w, true_s, true_l))

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
                    if true_param_prefix is not None:
                        true_alpha, true_gamma = adata.var.loc[gene_name, [true_param_prefix + 'alpha', true_param_prefix + 'gamma']]
                        true_u = sol_u(t, uu0, 0, true_gamma)
                        true_w = sol_u(t, 0, true_alpha, true_gamma)

                        true_p = np.vstack((true_u, true_w))

                    title_ = ['(unlabeled)', '(labeled)']

                    Obs, Pred = np.vstack((uu, ul)), np.vstack((u, w))

                for j in range(sub_plot_n):
                    row_ind = int(np.floor(i/ncols)) # make sure unlabled and labeled are in the same column.
                    ax = plt.subplot(gs[(row_ind * sub_plot_n + j) * ncols * grp_len + (i % ncols - 1) * grp_len + 1])
                    if true_param_prefix is not None and j == 0:
                        if has_splicing:
                            ax.text(0.75, 0.95, r'$\alpha^{true}$: {0:.2f} $\alpha^{est}$: {0:.2f} \n'
                                                r'$\beta^{true}$: {0:.2f} $\beta^{est}$: {0:.2f} \n'
                                                r'$\gamma^{true}$: {0:.2f} $\gamma^{est}$: {0:.2f} \n'.format(true_alpha, alpha, true_beta, beta, true_gamma, gamma),
                                    ha='right', va='top', transform=ax.transAxes)
                        else:
                            ax.text(0.75, 0.95, r'$\alpha^{true}$: {0:.2f} $\alpha^{est}$: {0:.2f} \n'
                                                r'$\gamma^{true}$: {0:.2f} $\gamma^{est}$: {0:.2f} \n'.format(true_alpha, alpha, true_gamma, gamma),
                                    ha='right', va='top', transform=ax.transAxes)

                    ax.boxplot(x=[Obs[j][T == std] for std in T_uniq], positions=T_uniq, widths=boxwidth,
                               showfliers=False, showmeans=True)
                    ax.plot(t, Pred[j], 'k--')
                    if true_param_prefix is not None: ax.plot(t, true_p[j], 'k--')
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
                    l = sol_u(t, 0, alpha, beta) + sol_s(t, 0, 0, alpha, beta, gamma)
                    L = sl + ul
                    if true_param_prefix is not None:
                        true_alpha, true_beta, true_gamma = adata.var.loc[gene_name, [true_param_prefix + 'alpha', true_param_prefix + 'beta', true_param_prefix + 'gamma']]
                        true_l = sol_u(t, 0, true_alpha, true_beta) + sol_s(t, 0, 0, true_alpha, true_beta, true_gamma)

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
                    if true_param_prefix is not None:
                        true_alpha, true_gamma = adata.var.loc[gene_name, [true_param_prefix + 'alpha', true_param_prefix + 'gamma']]
                        true_l = sol_u(t, 0, true_alpha, true_gamma)  # sol_s(t, 0, 0, alpha, 1, gamma)

                    title_ = ['labeled']

                Obs, Pred = np.hstack((np.zeros(L.shape), L)), np.hstack(l)
                if true_param_prefix is not None: true_p = np.hstack(true_l)

                row_ind = int(np.floor(i / ncols))  # make sure unlabled and labeled are in the same column.
                ax = plt.subplot(gs[(row_ind * sub_plot_n) * ncols * grp_len + (i % ncols - 1) * grp_len + 1])
                if true_param_prefix is not None:
                    if has_splicing:
                        ax.text(0.05, 0.95, r'$\alpha^{true}$: {0:.2f} $\alpha^{est}$: {0:.2f} \n'
                                            r'$\beta^{true}$: {0:.2f} $\beta^{est}$: {0:.2f} \n'
                                            r'$\gamma^{true}$: {0:.2f} $\gamma^{est}$: {0:.2f} \n'.format(true_alpha, alpha,
                                                                                                          true_beta, beta,
                                                                                                          true_gamma, gamma),
                                ha='right', va='top', transform=ax.transAxes)
                    else:
                        ax.text(0.05, 0.95, r'$\alpha^{true}$: {0:.2f} $\alpha^{est}$: {0:.2f} \n'
                                            r'$\gamma^{true}$: {0:.2f} $\gamma^{est}$: {0:.2f} \n'.format(true_alpha, alpha,
                                                                                                          true_gamma, gamma),
                                ha='right', va='top', transform=ax.transAxes)
                ax.boxplot(x=[Obs[np.hstack((np.zeros_like(T), T)) == std] for std in [0, T_uniq[0]]], positions=[0, T_uniq[0]], widths=boxwidth,
                           showfliers=False, showmeans=True)
                ax.plot(t, Pred, 'k--')
                if true_param_prefix is not None: ax.plot(t, true_p, 'r--')
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
                    alpha_stm = adata[:, gene_name].varm[prefix + 'alpha'].flatten()[1:]
                    alpha_stm0, k, _ = solve_first_order_deg(T_uniq[1:], alpha_stm)

                    # $u$ - unlabeled, unspliced
                    # $s$ - unlabeled, spliced
                    # $w$ - labeled, unspliced
                    # $l$ - labeled, spliced
                    #
                    # calculate labeled unspliced and spliced mRNA amount
                    u1, s1, u1_, s1_ = np.zeros(len(t) - 1), np.zeros(len(t) - 1), np.zeros(len(T_uniq) - 1), np.zeros(len(T_uniq) - 1)
                    for ind in np.arange(1, len(t)):
                        t_i = t[ind]
                        u0 = sol_u(np.max(t) - t_i, 0, alpha_std, beta)
                        alpha_stm_t_i = alpha_stm0 * np.exp(-k * t_i)
                        u1[ind - 1], s1[ind - 1] = sol_u(t_i, u0, alpha_stm_t_i, beta), sol_u(np.max(t), 0, beta, gamma)
                    for ind in np.arange(1, len(T_uniq)):
                        t_i = T_uniq[ind]
                        u0 = sol_u(np.max(T_uniq) - t_i, 0, alpha_std, beta)
                        alpha_stm_t_i = alpha_stm[ind - 1]
                        u1_[ind - 1], s1_[ind - 1] = sol_u(t_i, u0, alpha_stm_t_i, beta), sol_u(np.max(T_uniq), 0, beta, gamma)

                    Obs, Pred, Pred_ = np.vstack((ul, sl, uu, su)), np.vstack((u1.reshape(1, -1), s1.reshape(1, -1))), np.vstack((u1_.reshape(1, -1), s1_.reshape(1, -1)))
                    j_species, title_ = 4, ['unspliced labeled (new)', 'spliced labeled (new)', 'unspliced unlabeled (old)', 'spliced unlabeled (old)', 'alpha (steady state vs. stimulation)']
                else:
                    layers = ['X_new', 'X_total'] if 'X_new' in adata.layers.keys() else ['new', 'total']
                    uu, ul = adata[:, gene_name].layers[layers[1]] - adata[:, gene_name].layers[layers[0]], adata[:, gene_name].layers[layers[0]]
                    uu, ul = (uu.toarray().squeeze(), ul.toarray().squeeze()) if issparse(uu) else (uu.squeeze(), ul.squeeze())

                    if log_unnormalized and layers == ['new', 'total']:
                        uu, ul = np.log(uu + 1), np.log(ul + 1)

                    gamma, alpha_std = adata.var.loc[gene_name, [prefix + 'gamma', prefix + 'alpha_std']]
                    alpha_stm = adata[:, gene_name].varm[prefix + 'alpha'].flatten()[1:]

                    alpha_stm0, k, _ = solve_first_order_deg(T_uniq[1:], alpha_stm)
                    # require no beta functions
                    u1, u1_ = np.zeros(len(t) - 1), np.zeros(len(T_uniq) - 1) # interpolation or original time point
                    for ind in np.arange(1, len(t)):
                        t_i = t[ind]
                        u0 = sol_u(np.max(t) - t_i, 0, alpha_std, gamma)
                        alpha_stm_t_i = alpha_stm0 * np.exp(-k * t_i)
                        u1[ind - 1] = sol_u(t_i, u0, alpha_stm_t_i, gamma)

                    for ind in np.arange(1, len(T_uniq)):
                        t_i = T_uniq[ind]
                        u0 = sol_u(np.max(T_uniq) - t_i, 0, alpha_std, gamma)
                        alpha_stm_t_i = alpha_stm[ind - 1]
                        u1_[ind - 1] = sol_u(t_i, u0, alpha_stm_t_i, gamma)

                    Obs, Pred, Pred_ = np.vstack((ul, uu)), np.vstack((u1.reshape(1, -1))), np.vstack((u1_.reshape(1, -1)))
                    j_species, title_ = 2, ['labeled (new)', 'unlabeled (old)', 'alpha (steady state vs. stimulation)']

                group_list = [np.repeat(alpha_std, len(alpha_stm)), alpha_stm]

                for j in range(sub_plot_n):
                    row_ind = int(np.floor(i / ncols))  # make sure all related plots for the same gene in the same column.
                    ax = plt.subplot(gs[(row_ind * sub_plot_n + j) * ncols * grp_len + (i % ncols - 1) * grp_len + 1])
                    if j < j_species / 2:
                        ax.boxplot(x=[Obs[j][T == std] for std in T_uniq[1:]], positions=T_uniq[1:], widths=boxwidth,
                                   showfliers=False, showmeans=True)
                        ax.plot(t[1:], Pred[j], 'k--')
                        ax.scatter(T_uniq[1:], Pred_[j], s=20, c='red')
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



