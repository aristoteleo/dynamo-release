import numpy as np
import pandas as pd
from scipy.sparse import issparse
from .utilities import despline, minimal_xticks, minimal_yticks


def scatters(adata, genes, x=0, y=1, mode='splicing', type='expression', vkey='S', ekey='X', basis='umap', color=None):
    """Scatter plot of cells for phase portrait or for low embedding embedding, colored by gene expression, velocity or cell groups.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        genes: `list`
            The gene names whose gene expression will be faceted.
        x: `int` (default: `0`)
            The column index of the low dimensional embedding for the x-axis
        y: `int` (default: `1`)
            The column index of the low dimensional embedding for the y-axis
        current_layer: `str` (default: X)
            Which layer of expression value will be used.
        color: `list` or None (default: None)
            A list of attributes of cells (column names in the adata.obs) will be used to color cells.
        use_raw: `bool` (defaul: False)
            `str` (default: X)
                Which layer of expression value will be used.
        Vkey: `str` ('S`)
            The key for the velocity
        Ekey: `str` (`spliced`)
            The key for the gene expression.
        basis`str` (default: `trimap`)
            The reduced dimension embedding of cells to visualize.
        n_columns: `int  (default: 1)
            The number of columns of the resulting plot.
        type: `str` (default: `expression`)
            Which plotting type to use, either embedding, expression, velocity or phase.
        cmap: `plt.cm` or None (default: None)
            The color map function to use.
        figsize: `None` or `[float, float]` (default: None)
            The width and height of a figure.
        gs: `plt.XX`
        **kwargs:
            Additional parameters that will be passed to plt.scatter function

    Returns
    -------
        Nothing but a scatter plot of cells.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    # sns.set_style('ticks')

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
        color_vec = list(set(color).intersection(adata.obs.keys()))

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

    if 'velocity_parameter_gamma' in adata.var.columns:
        gamma = adata.var.velocity_parameter_gamma[genes].values
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
            if 'velocity_parameter_eta' in adata.var.columns:
                gamma_P = adata.var.velocity_parameter_eta[genes].values
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
    plt.figure(None, (ncol * 6, nrow * 6), dpi=160)

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
        if type is 'phase':
            if cur_pd.color.unique() != np.nan:
                sns.scatterplot(cur_pd.iloc[:, 1], cur_pd.iloc[:, 0], hue="expression", ax=ax1, palette="viridis") # x-axis: S vs y-axis: U
            else:
                sns.scatterplot(cur_pd.iloc[:, 1], cur_pd.iloc[:, 0], hue=color, ax=ax1, palette="Set2") # x-axis: S vs y-axis: U

            ax1.set_title(gn)
            xnew = np.linspace(0, cur_pd.iloc[:, 1].max())
            ax1.plot(xnew, xnew * cur_pd.loc[:, 'gamma'].unique() + cur_pd.loc[:, 'velocity_offset'].unique(), c="k")
            ax1.set_title(gn)
            ax1.set_xlim(0, np.max(cur_pd.iloc[:, 1])*1.02)
            ax1.set_ylim(0, np.max(cur_pd.iloc[:, 0])*1.02)

            despline(ax1) # sns.despline()
        elif type is 'velocity':
            df_embedding = pd.concat([cur_pd, embedding], axis=1)
            V_vec = df_embedding.loc[:, 'velocity']

            limit = np.nanmax(np.abs(np.nanpercentile(V_vec, [1, 99])))  # upper and lowe limit / saturation

            V_vec = V_vec + limit  # that is: tmp_colorandum - (-limit)
            V_vec = V_vec / (2 * limit)  # that is: tmp_colorandum / (limit - (-limit))
            V_vec = np.clip(V_vec, 0, 1)

            cmap = plt.cm.RdBu_r # sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
            sns.scatterplot(embedding.iloc[:, 0], embedding.iloc[:, 1], hue=df_embedding.loc[:, 'expression'], ax=ax2, palette=cmap, legend=False)
            ax2.set_title(gn + '(' + ekey + ')')
            ax2.set_xlabel(basis + '_1')
            ax2.set_ylabel(basis + '_2')
        elif type is 'expression':
            cmap = plt.cm.Greens # sns.diverging_palette(10, 220, sep=80, as_cmap=True)
            sns.scatterplot(embedding.iloc[:, 0], embedding.iloc[:, 1], hue=V_vec, ax=ax3, palette=cmap, legend=False)
            ax3.set_title(gn + '(' + vkey + ')')
            ax3.set_xlabel(basis + '_1')
            ax3.set_ylabel(basis + '_2')

        if 'protein' in adata.obsm.keys() and mode is 'full' and all([i in adata.layers.keys() for i in ['uu', 'ul', 'su', 'sl']]):
            sns.scatterplot(cur_pd.iloc[:, 3], cur_pd.iloc[:, 2], hue=group, ax=ax4) # x-axis: Protein vs. y-axis: Spliced
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
            sns.scatterplot(embedding.iloc[:, 0], embedding.iloc[:, 1], hue=df_embedding.loc[:, 'expression'], \
                            ax=ax5, legend=False, palette=cmap)
            ax5.set_title(gn + '(protein expression)')
            ax5.set_xlabel(basis + '_1')
            ax5.set_ylabel(basis + '_2')
            cmap = sns.diverging_palette(145, 280, s=85, l=25, n=7)
            sns.scatterplot(embedding.iloc[:, 0], embedding.iloc[:, 1], hue=V_vec, ax=ax6, legend=False, palette=cmap)
            ax6.set_title(gn + '(protein velocity)')
            ax6.set_xlabel(basis + '_1')
            ax6.set_ylabel(basis + '_2')

    plt.tight_layout()
    plt.show()


