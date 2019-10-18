import numpy as np
import pandas as pd
from scipy.sparse import issparse

from ..preprocessing.preprocess import topTable
from .utilities import despline, minimal_xticks, minimal_yticks


def show_fraction(adata, mode='labelling', group=None):
    """Plot the fraction of each category of data used in the velocity estimation.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    mode: `string` (default: labeling)
        Which mode of data do you want to show, can be one of `labeling`, `splicing` and `full`.
    group: `string` (default: None)
        Which group to facets the data into subplots. Default is None, or no faceting will be used.

    Returns
    -------
        A ggplot-like plot that shows the fraction of category, produced from plotnine (A equivalent of R's ggplot2 in Python).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('ticks')

    if not (mode in ['labelling', 'splicing', 'full']):
        raise Exception('mode can be only one of the labelling, splicing or full')

    if mode is 'labelling' and (all([i in adata.layers.keys() for i in ['new', 'total']])):
        new_mat, total_mat = adata.layers['new'], adata.layers['total']

        new_cell_sum, tot_cell_sum = np.sum(new_mat, 1), np.sum(total_mat, 1) if not issparse(new_mat) else new_mat.sum(1).A1, \
                                     total_mat.sum(1).A1

        new_frac_cell = new_cell_sum / tot_cell_sum
        old_frac_cell = 1 - new_frac_cell
        df = pd.DataFrame({'new_frac_cell': new_frac_cell, 'old_frac_cell': old_frac_cell})

        if group is not None and group in adata.obs.key():
            df['group'] = adata.obs[group]
            res = df.melt(value_vars=['new_frac_cell', 'old_frac_cell'], id_vars=['group'])
        else:
            res = df.melt(value_vars=['new_frac_cell', 'old_frac_cell'])

    elif mode is 'splicing' and all([i in adata.layers.keys() for i in ['spliced', 'unspliced']]):
        ambiguous = adata.layers['ambiguous'] if 'ambiguous' in adata.layers.keys() else np.array(0)

        unspliced_mat, spliced_mat, ambiguous_mat = adata.layers['unspliced'], adata.layers['spliced'], ambiguous
        un_cell_sum, sp_cell_sum, am_cell_sum = (np.sum(unspliced_mat, 1), np.sum(spliced_mat, 1), np.sum(ambiguous_mat, 1))  if not \
            issparse(unspliced_mat) else (unspliced_mat.sum(1).A1, spliced_mat.sum(1).A1, ambiguous_mat.sum(1).A1)

        if ambiguous is not np.array(0):
            tot_cell_sum = un_cell_sum + sp_cell_sum + am_cell_sum
            un_frac_cell, sp_frac_cell, am_frac_cell = un_cell_sum / tot_cell_sum, sp_cell_sum / tot_cell_sum, am_cell_sum / tot_cell_sum
            df = pd.DataFrame({'unspliced': un_frac_cell, 'spliced': sp_frac_cell, 'ambiguous': am_frac_cell}, index=adata.obs.index)
        else:
            tot_cell_sum = un_cell_sum + sp_cell_sum
            un_frac_cell, sp_frac_cell = un_cell_sum / tot_cell_sum, sp_cell_sum / tot_cell_sum
            df = pd.DataFrame({'unspliced': un_frac_cell, 'spliced': sp_frac_cell}, index=adata.obs.index)

        if group is not None and group in adata.obs.columns:
            df['group'] = adata.obs.loc[:, group]
            res = df.melt(value_vars=['unspliced', 'spliced', 'ambiguous'], id_vars=['group']) if ambiguous is not np.array(0) else \
                df.melt(value_vars=['unspliced', 'spliced'], id_vars=['group'])
        else:
            res = df.melt(value_vars=['unspliced', 'spliced', 'ambiguous']) if ambiguous is not np.array(0) else \
                 df.melt(value_vars=['unspliced', 'spliced'])

    elif mode is 'full' and all([i in adata.layers.keys() for i in ['uu', 'ul', 'su', 'sl']]):
        uu, ul, su, sl = adata.layers['uu'], adata.layers['ul'], adata.layers['su'], adata.layers['sl']
        uu_sum, ul_sum, su_sum, sl_sum = np.sum(uu, 1), np.sum(ul, 1), np.sum(su, 1), np.sum(sl, 1) if not issparse(uu) \
            else uu.sum(1).A1, ul.sum(1).A1, su.sum(1).A1, sl.sum(1).A1

        tot_cell_sum = uu + ul + su + sl
        uu_frac, ul_frac, su_frac, sl_frac = uu_sum / tot_cell_sum, ul_sum / tot_cell_sum, su / tot_cell_sum, sl / tot_cell_sum
        df = pd.DataFrame({'uu_frac': uu_frac, 'ul_frac': ul_frac, 'su_frac': su_frac, 'sl_frac': sl_frac})

        if group is not None and group in adata.obs.key():
            df['group'] = adata.obs[group]
            res = df.melt(value_vars=['uu_frac', 'ul_frac', 'su_frac', 'sl_frac'], id_vars=['group'])
        else:
            res = df.melt(value_vars=['uu_frac', 'ul_frac', 'su_frac', 'sl_frac'])

    else:
        raise Exception('Your adata is corrupted. Make sure that your layer has keys new, total for the labelling mode, '
                        'spliced, (ambiguous, optional), unspliced for the splicing model and uu, ul, su, sl for the full mode')

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
        threshold: `float` (default: 0.002)
            The threshold for the second derivative of the cumulative sum of the variance for each principal component.
        n_pcs: `int` (default: None)
            Number of principal components.

    Returns
    -------
        Nothing but make a matplotlib based plot for showing the cumulative variance explained by each PC.
    """

    import matplotlib.pyplot as plt

    var_ = adata.uns["explained_variance_ratio_"]
    plt.plot(np.cumsum(var_))
    tmp = np.diff(np.diff(np.cumsum(var_))>threshold)
    n_comps = n_pcs if n_pcs is not None else np.where(tmp)[0][0] if np.any(tmp) else 20
    plt.axvline(n_comps, c="k")
    plt.xlabel('PCs')
    plt.ylabel('Cumulative variance')
    plt.show()


def featureGenes(adata, layer='X'):
    """Plot selected feature genes on top of the mean vs. dispersion scatterplot.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object
        layer: `str` (default: `X`)
            The data from a particular layer (include X) used for making the feature gene plot.

    Returns
    -------

    """
    import matplotlib.pyplot as plt

    disp_table = topTable(adata, layer)

    ordering_genes = adata.var['use_for_dynamo'] if 'use_for_dynamo' in adata.var.columns else None

    layer_keys = list(adata.layers.keys())
    layer_keys.extend('X')
    layer = list(set(layer_keys).intersection(layer))[0]

    if layer in ['raw', 'X']:
        key = 'dispFitInfo'
    else:
        key = layer + '_dispFitInfo'
    mu_linspace = np.linspace(np.min(disp_table['mean_expression']), np.max(disp_table['mean_expression']), num = 1000)
    disp_fit = adata.uns['dispFitInfo']['disp_func'](mu_linspace)

    plt.plot(mu_linspace, disp_fit, alpha=0.4, color='k')
    valid_ind = disp_table.gene_id.isin(ordering_genes.index[ordering_genes]).values if ordering_genes is not None else np.ones(disp_table.shape[0], dtype=bool)

    valid_disp_table = disp_table.iloc[valid_ind, :]
    plt.scatter(valid_disp_table['mean_expression'], valid_disp_table['dispersion_empirical'], s=3, alpha=0.3, color='tab:red')
    neg_disp_table = disp_table.iloc[~valid_ind, :]

    plt.scatter(neg_disp_table['mean_expression'], neg_disp_table['dispersion_empirical'], s=3, alpha=1, color='tab:blue')

    # plt.xlim((0, 100))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Mean')
    plt.ylabel('Dispersion')
    plt.show()


def show_phase(adata, genes, mode='labeling', vkey='velocity', basis='umap', group=None):
    """Draw the phase portrait, velocity, expression values on the low dimensional embedding.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    genes: `list`
        A list of gene names that are going to be visualized.
    reference_layer: `str`
        reference_layer
    mode: `string` (default: labeling)
        Which mode of data do you want to show, can be one of `labeling`, `splicing` and `full`.
    vkey: `string` (default: velocity)
        Which velocity key used for visualizing the magnitude of velocity. Can be either velocity in the layers slot or the
        keys in the obsm slot.
    basis: `string` (default: umap)
        Which low dimensional embedding will be used to visualize the cell.
    group: `string` (default: None)
        Which group will be used to color cells, only used for the phase portrait because the other two plots are colored
        by the velocity magnitude or the gene expression value, respectively.

    Returns
    -------
        A matplotlib plot that shows 1) the phase portrait of each category used in velocity embedding, cells' low dimensional
        embedding, colored either by 2) the gene expression level or 3) the velocity magnitude values.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('ticks')

    # there is no solution for combining multiple plot in the same figure in plotnine, so a pure matplotlib is used
    # see more at https://github.com/has2k1/plotnine/issues/46
    genes, idx = adata.var.index[adata.var.index.isin(genes)], np.where(adata.var.index.isin(genes))[0]
    if len(genes) == 0:
        raise Exception('adata has no genes listed in your input gene vector: {}'.format(genes))
    if not 'X_' + basis in adata.obsm.keys():
        raise Exception('{} is not applied to adata.}'.format(basis))
    else:
        embedding = pd.DataFrame({basis + '_0': adata.obsm['X_' + basis].iloc[:, 0], \
                                  basis + '_1': adata.obsm['X_' + basis].iloc[:, 1]})

    n_cells, n_genes = adata.shape[0], len(genes)

    if vkey in adata.layers.keys():
        velocity = adata[genes].layers[vkey]
    elif vkey in adata.obsm.keys():
        velocity = adata[genes].obsm[vkey]
    else:
        raise Exception('adata has no vkey {} in either the layers or the obsm slot'.format(vkey))
    velocity = np.sum(velocity**2, 1) #### all genes
    # different layer for different velocities
    if 'velocity_parameter_gamma' in adata.var.columns():
        gamma = adata.var.velocity_parameter_gamma[genes].values
        velocity_offset = [0] * n_cells if not ("velocity_offset" in adata.var.columns()) else \
            adata.var.velocity_offset[genes].values
    else:
        raise Exception('adata does not seem to have velocity_gamma column. Velocity estimation is required before '
                        'running this function.')

    if not (mode in ['labelling', 'splicing', 'full']):
        raise Exception('mode can be only one of the labelling, splicing or full')

    if mode is 'labelling' and all([i in adata.layers.keys() for i in ['new', 'total']]):
        new_mat, tot_mat = adata[:, genes].layers['new'], adata[:, genes].layers['total']
        df = pd.DataFrame({"new": new_mat.flatten(), "total": tot_mat.flatten(), 'gene': genes * n_cells, 'prediction':
                           np.repeat(gamma, n_cells) * new_mat.flatten() + np.repeat(velocity_offset, n_cells),
                           "velocity": genes * n_cells}, index=range(n_cells * n_genes))

    elif mode is 'splicing' and all([i in adata.layers.keys() for i in ['spliced', 'unspliced']]):
        unspliced_mat, spliced_mat = adata[:, genes].layers['unspliced'], adata[:, genes].layers['spliced']
        df = pd.DataFrame({"unspliced": unspliced_mat.flatten(), "spliced": spliced_mat.flatten(), 'gene': genes * n_cells,
                           'prediction': np.repeat(gamma, n_cells) * unspliced_mat.flatten() + np.repeat(velocity_offset, \
                            n_cells), "velocity": genes * n_cells}, index=range(n_cells * n_genes))

    elif mode is 'full' and all([i in adata.layers.keys() for i in ['uu', 'ul', 'su', 'sl']]):
        uu, ul, su, sl = adata[:, genes].layers['uu'], adata[:, genes].layers['ul'], adata[:, genes].layers['su'], \
                         adata[:, genes].layers['sl']
        df = pd.DataFrame({"uu": uu.flatten(), "ul": ul.flatten(), "su": su.flatten(), "sl": sl.flatten(),
                           'gene': genes * n_cells, 'prediction': np.repeat(gamma, n_cells) * uu.flatten() +
                            np.repeat(velocity_offset, n_cells), "velocity": genes * n_cells}, index=range(n_cells * n_genes))

        # add the protein data here

    else:
        raise Exception('Your adata is corrupted. Make sure that your layer has keys new, old for the labelling mode, '
                        'spliced, ambiguous, unspliced for the splicing model and uu, ul, su, sl for the full mode')

    # for cur_axes in g.axes.flatten():
    #     x0, x1 = cur_axes.get_xlim()
    #     y0, y1 = cur_axes.get_ylim()
    #
    #     points = np.linspace(min(x0, y0), max(x1, y1), 100)
    #
    #     cur_axes.plot(points, points, color='red', marker=None, linestyle='--', linewidth=1.0)

    nrow, ncol = np.sqrt(3 * n_genes), np.sqrt(3 * n_genes)
    ncol = ncol - 1 if nrow * (ncol - 1) == 3 * n_genes else ncol
    plt.figure(None, (3*nrow, 3*ncol), dpi=160)

    # the following code is inspired by https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb
    gs = plt.GridSpec(nrow,ncol)
    for i, gn in enumerate(genes):
        ax = plt.subplot(gs[i*3])
        try:
            ix=np.where(adata.var.index == gn)[0][0]
        except:
            continue
        cur_pd = df.iloc[df.gene == gn, :]
        sns.scatterplot(cur_pd.iloc[:, 0], cur_pd.iloc[:, 1], hue=group)
        plt.title(gn)
        plt.plot(cur_pd.iloc[:, 0], cur_pd.loc[:, 'prediction'], c="k")
        plt.ylim(0, np.max(cur_pd.iloc[:, 0])*1.02)
        plt.xlim(0, np.max(cur_pd.iloc[:, 1])*1.02)

        despline() # sns.despline()

        df_embedding = pd.concat([embedding, cur_pd.loc[:, 'gene']], ignore_index=False)
        sns.scatterplot(df_embedding.iloc[:, 0], df_embedding.iloc[:, 1], hue=df_embedding.loc[:, 'gene'], ax=ax)
        sns.scatterplot(df_embedding.iloc[:, 0], df_embedding.iloc[:, 1], hue=df_embedding.loc[:, 'velocity'], ax=ax)

    plt.tight_layout()
