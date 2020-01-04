import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix

from ..preprocessing.preprocess import topTable
from ..preprocessing.utilities import get_layer_keys
from .utilities import despline, minimal_xticks, minimal_yticks


def show_fraction(adata, group=None):
    """Plot the fraction of each category of data used in the velocity estimation.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    group: `string` (default: None)
        Which group to facets the data into subplots. Default is None, or no faceting will be used.

    Returns
    -------
        A violin plot that shows the fraction of each category, produced by seaborn.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('ticks')

    mode = None
    if pd.Series(['spliced', 'unspliced']).isin(adata.layers.keys()).all():
        mode = 'splicing'
    elif pd.Series(['new', 'total']).isin(adata.layers.keys()).all():
        mode = 'labelling'
    elif pd.Series(['uu', 'ul', 'su','sl']).isin(adata.layers.keys()).all():
        mode = 'full'

    if not (mode in ['labelling', 'splicing', 'full']):
        raise Exception("your data doesn't seem to have either splicing or labeling or both information")

    if mode is 'labelling':
        new_mat, total_mat = adata.layers['new'], adata.layers['total']

        new_cell_sum, tot_cell_sum = (np.sum(new_mat, 1), np.sum(total_mat, 1)) if not issparse(new_mat) else (new_mat.sum(1).A1, \
                                     total_mat.sum(1).A1)

        new_frac_cell = new_cell_sum / tot_cell_sum
        old_frac_cell = 1 - new_frac_cell
        df = pd.DataFrame({'new_frac_cell': new_frac_cell, 'old_frac_cell': old_frac_cell}, index=adata.obs.index)

        if group is not None and group in adata.obs.key():
            df['group'] = adata.obs[group]
            res = df.melt(value_vars=['new_frac_cell', 'old_frac_cell'], id_vars=['group'])
        else:
            res = df.melt(value_vars=['new_frac_cell', 'old_frac_cell'])

    elif mode is 'splicing':
        if 'ambiguous' in adata.layers.keys():
            ambiguous = adata.layers['ambiguous']
        else:
            ambiguous = csr_matrix(np.array([[0]])) if issparse(adata.layers['unspliced']) else np.array([[0]])

        unspliced_mat, spliced_mat, ambiguous_mat = adata.layers['unspliced'], adata.layers['spliced'], ambiguous
        un_cell_sum, sp_cell_sum = (np.sum(unspliced_mat, 1), np.sum(spliced_mat, 1)) if not \
            issparse(unspliced_mat) else (unspliced_mat.sum(1).A1, spliced_mat.sum(1).A1)

        if 'ambiguous' in adata.layers.keys():
            am_cell_sum = ambiguous_mat.sum(1).A1 if issparse(unspliced_mat) else np.sum(ambiguous_mat, 1)
            tot_cell_sum = un_cell_sum + sp_cell_sum + am_cell_sum
            un_frac_cell, sp_frac_cell, am_frac_cell = un_cell_sum / tot_cell_sum, sp_cell_sum / tot_cell_sum, am_cell_sum / tot_cell_sum
            df = pd.DataFrame({'unspliced': un_frac_cell, 'spliced': sp_frac_cell, 'ambiguous': am_frac_cell}, index=adata.obs.index)
        else:
            tot_cell_sum = un_cell_sum + sp_cell_sum
            un_frac_cell, sp_frac_cell = un_cell_sum / tot_cell_sum, sp_cell_sum / tot_cell_sum
            df = pd.DataFrame({'unspliced': un_frac_cell, 'spliced': sp_frac_cell}, index=adata.obs.index)

        if group is not None and group in adata.obs.columns:
            df['group'] = adata.obs.loc[:, group]
            res = df.melt(value_vars=['unspliced', 'spliced', 'ambiguous'], id_vars=['group']) if 'ambiguous' in adata.layers.keys() else \
                df.melt(value_vars=['unspliced', 'spliced'], id_vars=['group'])
        else:
            res = df.melt(value_vars=['unspliced', 'spliced', 'ambiguous']) if 'ambiguous' in adata.layers.keys() else \
                 df.melt(value_vars=['unspliced', 'spliced'])

    elif mode is 'full':
        uu, ul, su, sl = adata.layers['uu'], adata.layers['ul'], adata.layers['su'], adata.layers['sl']
        uu_sum, ul_sum, su_sum, sl_sum = np.sum(uu, 1), np.sum(ul, 1), np.sum(su, 1), np.sum(sl, 1) if not issparse(uu) \
            else uu.sum(1).A1, ul.sum(1).A1, su.sum(1).A1, sl.sum(1).A1

        tot_cell_sum = uu + ul + su + sl
        uu_frac, ul_frac, su_frac, sl_frac = uu_sum / tot_cell_sum, ul_sum / tot_cell_sum, su / tot_cell_sum, sl / tot_cell_sum
        df = pd.DataFrame({'uu_frac': uu_frac, 'ul_frac': ul_frac, 'su_frac': su_frac, 'sl_frac': sl_frac}, index=adata.obs.index)

        if group is not None and group in adata.obs.key():
            df['group'] = adata.obs[group]
            res = df.melt(value_vars=['uu_frac', 'ul_frac', 'su_frac', 'sl_frac'], id_vars=['group'])
        else:
            res = df.melt(value_vars=['uu_frac', 'ul_frac', 'su_frac', 'sl_frac'])

    if group is None:
        g = sns.violinplot(x="variable", y="value", data=res)
        g.set_xlabel('Category')
        g.set_ylabel('Fraction')
    else:
        g = sns.catplot(x="variable", y="value", data=res, kind='violin', col="group", col_wrap=4)
        g.set_xlabels('Category')
        g.set_ylabels('Fraction')

    plt.show()


def variance_explained(adata, threshold=0.002, n_pcs=None):
    """Plot the accumulative variance explained by the principal components.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
        threshold: `float` (default: `0.002`)
            The threshold for the second derivative of the cumulative sum of the variance for each principal component.
            This threshold is used to determine the number of principal component used for downstream non-linear dimension
            reduction.
        n_pcs: `int` (default: `None`)
            Number of principal components.

    Returns
    -------
        Nothing but make a matplotlib based plot for showing the cumulative variance explained by each PC.
    """

    import matplotlib.pyplot as plt

    var_ = adata.uns["explained_variance_ratio_"]
    _, ax = plt.subplots()
    ax.plot(var_, c='k')
    tmp = np.diff(np.diff(np.cumsum(var_)) > threshold)
    n_comps = n_pcs if n_pcs is not None else np.where(tmp)[0][0] if np.any(tmp) else 20
    ax.axvline(n_comps, c="r")
    ax.set_xlabel('PCs')
    ax.set_ylabel('Variance explained')
    ax.set_xticks(list(ax.get_xticks()) + [40, 51])
    ax.set_xlim(0, len(var_))

    plt.show()


def feature_genes(adata, layer='X', mode=None):
    """Plot selected feature genes on top of the mean vs. dispersion scatterplot.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        layer: `str` (default: `X`)
            The data from a particular layer (include X) used for making the feature gene plot.
        mode: None or `str` (default: `None`)
            The method to select the feature genes (can be either `dispersion`, `gini` or `SVR`).

    Returns
    -------
        Nothing but plots the selected feature genes via the mean, CV plot.
    """

    import matplotlib.pyplot as plt
    mode = adata.uns['feature_selection'] if mode is None else mode

    layer = get_layer_keys(adata, layer, include_protein=False)[0]
    if layer in ['raw', 'X']:
        key = 'dispFitInfo' if mode is 'dispersion' else 'velocyto_SVR'
    else:
        key = layer + '_dispFitInfo' if mode is 'dispersion' else layer + '_velocyto_SVR'

    if mode is 'dispersion':
        table = topTable(adata, layer)
        x_min, x_max = np.nanmin(table['mean_expression']), np.nanmax(table['mean_expression'])
    elif mode is 'SVR':
        if not np.all(pd.Series(['log_m', 'score']).isin(adata.var.columns)):
            raise Exception('Looks like you have not run support vector machine regression yet, try run SVRs first.')
        else:
            detected_bool = adata.uns[key]['detected_bool']
            table = adata.var.loc[detected_bool, ['log_m', 'log_cv', 'score']]
            table = table.loc[np.isfinite(table['log_m']) & np.isfinite(table['log_cv']), :]
            x_min, x_max = np.nanmin(table['log_m']), np.nanmax(table['log_m'])

    ordering_genes = adata.var['use_for_dynamo'] if 'use_for_dynamo' in adata.var.columns else None

    mu_linspace = np.linspace(x_min, x_max, num=1000)
    fit = adata.uns[key]['disp_func'](mu_linspace) if mode is 'dispersion' else adata.uns[key]['SVR'](mu_linspace.reshape(-1, 1))

    plt.plot(mu_linspace, fit, alpha=0.4, color='r')
    valid_ind = table.index.isin(ordering_genes.index[ordering_genes]) if ordering_genes is not None else np.ones(table.shape[0], dtype=bool)

    valid_disp_table = table.iloc[valid_ind, :]
    if mode is 'dispersion':
        plt.scatter(valid_disp_table['mean_expression'], valid_disp_table['dispersion_empirical'], s=3, alpha=1, color='xkcd:black')
    elif mode is 'SVR':
        plt.scatter(valid_disp_table['log_m'], valid_disp_table['log_cv'], s=3, alpha=1, color='xkcd:black')

    neg_disp_table = table.iloc[~valid_ind, :]

    if mode is 'dispersion':
        plt.scatter(neg_disp_table['mean_expression'], neg_disp_table['dispersion_empirical'], s=3, alpha=0.5, color='xkcd:grey')
    elif mode is 'SVR':
        plt.scatter(neg_disp_table['log_m'], neg_disp_table['log_cv'], s=3, alpha=0.5, color='xkcd:grey')

    # plt.xlim((0, 100))
    if mode is 'dispersion':
        plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Mean (log)')
    plt.ylabel('Dispersion (log)') if mode is 'dispersion' else plt.ylabel('CV (log)')
    plt.show()


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


