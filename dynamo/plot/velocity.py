import numpy as np
import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns

import yt

import scipy as sc
from scipy.sparse import issparse
#from licpy.lic import runlic


# plotting utility functions from https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb
def despline():
    ax1 = plt.gca()
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

def minimal_xticks(start, end):
    end_ = np.around(end, -int(np.log10(end))+1)
    xlims = np.linspace(start, end_, 5)
    xlims_tx = [""]*len(xlims)
    xlims_tx[0], xlims_tx[-1] = f"{xlims[0]:.0f}", f"{xlims[-1]:.02f}"
    plt.xticks(xlims, xlims_tx)


def minimal_yticks(start, end):
    end_ = np.around(end, -int(np.log10(end))+1)
    ylims = np.linspace(start, end_, 5)
    ylims_tx = [""]*len(ylims)
    ylims_tx[0], ylims_tx[-1] = f"{ylims[0]:.0f}", f"{ylims[-1]:.02f}"
    plt.yticks(ylims, ylims_tx)


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

    if not (mode in ['labelling', 'splicing', 'full']):
        raise Exception('mode can be only one of the labelling, splicing or full')

    if mode is 'labelling' and all([i in adata.layers.keys() for i in ['new', 'old']]):
        new_mat, old_mat = adata.layers['new'], adata.layers['old']
        new_cell_sum, old_cell_sum = np.sum(new_mat, 1), np.sum(old_mat, 1) if not issparse(new_mat) else new_mat.sum(1).A1, \
                                     old_mat.sum(1).A1

        tot_cell_sum = new_cell_sum + old_cell_sum
        new_frac_cell, old_frac_cell = new_cell_sum / tot_cell_sum, old_cell_sum / tot_cell_sum
        df = pd.DataFrame({'new_frac_cell': new_frac_cell, 'old_frac_cell': old_frac_cell})

        if group is not None and group in adata.obs.key():
            df['group'] = adata.obs[group]

        df = df.melt(value_vars=['new_frac_cell', 'old_frac_cell'])

    elif mode is 'splicing' and all([i in adata.layers.keys() for i in ['spliced', 'ambiguous', 'unspliced']]):
        unspliced_mat, spliced_mat, ambiguous_mat = adata.layers['unspliced'], adata.layers['spliced'], adata.layers['ambiguous']
        un_cell_sum, sp_cell_sum, am_cell_sum = np.sum(unspliced_mat, 1), np.sum(spliced_mat, 1), np.sum(ambiguous_mat, 1)  if not \
            issparse(unspliced_mat) else unspliced_mat.sum(1).A1, spliced_mat.sum(1).A1, ambiguous_mat.sum(1).A1

        tot_cell_sum = un_cell_sum + sp_cell_sum + am_cell_sum
        un_frac_cell, sp_frac_cell, am_frac_cell = un_cell_sum / tot_cell_sum, sp_cell_sum / tot_cell_sum, am_cell_sum / tot_cell_sum
        df = pd.DataFrame({'unspliced': un_frac_cell, 'spliced': sp_frac_cell, 'ambiguous': am_frac_cell})

        if group is not None and group in adata.obs.key():
            df['group'] = adata.obs[group]

        df = df.melt(value_vars=['unspliced', 'spliced', 'ambiguous'])

    elif mode is 'full' and all([i in adata.layers.keys() for i in ['uu', 'ul', 'su', 'sl']]):
        uu, ul, su, sl = adata.layers['uu'], adata.layers['ul'], adata.layers['su'], adata.layers['sl']
        uu_sum, ul_sum, su_sum, sl_sum = np.sum(uu, 1), np.sum(ul, 1), np.sum(su, 1), np.sum(sl, 1) if not issparse(uu) \
            else uu.sum(1).A1, ul.sum(1).A1, su.sum(1).A1, sl.sum(1).A1

        tot_cell_sum = uu + ul + su + sl
        uu_frac, ul_frac, su_frac, sl_frac = uu_sum / tot_cell_sum, ul_sum / tot_cell_sum, su / tot_cell_sum, sl / tot_cell_sum
        df = pd.DataFrame({'uu_frac': uu_frac, 'ul_frac': ul_frac, 'su_frac': su_frac, 'sl_frac': sl_frac})

        if group is not None and group in adata.obs.key():
            df['group'] = adata.obs[group]

        df = df.melt(value_vars=['uu_frac', 'ul_frac', 'su_frac', 'sl_frac'])

    else:
        raise Exception('Your adata is corrupted. Make sure that your layer has keys new, old for the labelling mode, '
                        'spliced, ambiguous, unspliced for the splicing model and uu, ul, su, sl for the full mode')

    if group is None:
        (ggplot(df, aes(variable, value)) + geom_violin() + facet_wrap('~group') + xlab('Category') + ylab('Fraction'))
    else:
        (ggplot(df, aes(variable, value)) + geom_violin() + facet_wrap('~group') + xlab('Category') + ylab('Fraction'))


def show_phase(adata, genes, mode='labeling', vkey='velocity', basis='umap', group=None):
    """Draw the phase portrait, velocity, expression values on the low dimensional embedding.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    genes: `list`
        A list of gene names that are going to be visualized.
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

    # there is no solution for combining multiple plot in the same figure in plotnine, so a pure matplotlib is used
    # see more at https://github.com/has2k1/plotnine/issues/46
    genes = genes[genes in adata.var_name]
    if len(genes) == 0:
        raise Exception('adata has no genes listed in your input gene vector: {}'.format(genes))
    if not basis in adata.obsm.keys():
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
    velocity = np.sum(velocity**2, 1)

    if 'velocity_gamma' in adata.var.columns():
        gamma = adata.var.gamma[genes].values
        velocity_offset = [0] * n_cells if not ("velocity_offset" in adata.var.columns()) else \
            adata.var.velocity_offset[genes].values
    else:
        raise Exception('adata does not seem to have velocity_gamma column. Velocity estimation is required before '
                        'running this function.')

    if not (mode in ['labelling', 'splicing', 'full']):
        raise Exception('mode can be only one of the labelling, splicing or full')

    if mode is 'labelling' and all([i in adata.layers.keys() for i in ['new', 'old']]):
        new_mat, old_mat = adata[genes].layers['new'], adata[genes].layers['old']
        df = pd.DataFrame({"new": new_mat.flatten(), "old": old_mat.flatten(), 'gene': genes * n_cells, 'prediction':
                           np.repeat(gamma, n_cells) * new_mat.flatten() + np.repeat(velocity_offset, n_cells),
                           "velocity": genes * n_cells}, index = range(n_cells * n_genes))

    elif mode is 'splicing' and all([i in adata.layers.keys() for i in ['spliced', 'ambiguous', 'unspliced']]):
        unspliced_mat, spliced_mat = adata.layers['unspliced'], adata.layers['spliced']
        df = pd.DataFrame({"unspliced": unspliced_mat.flatten(), "spliced": spliced_mat.flatten(), 'gene': genes * n_cells,
                           'prediction': np.repeat(gamma, n_cells) * unspliced_mat.flatten() + np.repeat(velocity_offset, \
                            n_cells), "velocity": genes * n_cells}, index = range(n_cells * n_genes))

    elif mode is 'full' and all([i in adata.layers.keys() for i in ['uu', 'ul', 'su', 'sl']]):
        uu, ul, su, sl = adata.layers['uu'], adata.layers['ul'], adata.layers['su'], adata.layers['sl']
        df = pd.DataFrame({"uu": uu.flatten(), "ul": ul.flatten(), "su": su.flatten(), "sl": sl.flatten(),
                           'gene': genes * n_cells, 'prediction': np.repeat(gamma, n_cells) * uu.flatten() +
                            np.repeat(velocity_offset, n_cells), "velocity": genes * n_cells}, index = range(n_cells * n_genes))

    else:
        raise Exception('Your adata is corrupted. Make sure that your layer has keys new, old for the labelling mode, '
                        'spliced, ambiguous, unspliced for the splicing model and uu, ul, su, sl for the full mode')

    # use seaborn to draw the plot:

    for cur_axes in g.axes.flatten():
        x0, x1 = cur_axes.get_xlim()
        y0, y1 = cur_axes.get_ylim()

        points = np.linspace(min(x0, y0), max(x1, y1), 100)

        cur_axes.plot(points, points, color='red', marker=None, linestyle='--', linewidth=1.0)

    plt.figure(None, (15,15), dpi=80)
    nrow, ncol = np.sqrt(3 * n_cells), np.sqrt(3 * n_cells)
    ncol = ncol - 1 if nrow * (ncol - 1) == 3 * n_cells else ncol

    # the following code is inspired by https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb
    gs = plt.GridSpec(nrow,ncol)
    for i, gn in enumerate(genes):
        ax = plt.subplot(gs[i*3])
        try:
            ix=np.where(vlm.ra["Gene"] == gn)[0][0]
        except:
            continue
        vcy.scatter_viz(vlm.Sx_sz[ix,:], vlm.Ux_sz[ix,:], c=vlm.colorandum, s=5, alpha=0.4, rasterized=True)
        cur_pd = df.iloc[df.gene == gn, :]
        sns.scatterplot(cur_pd.iloc[:, 0], cur_pd.iloc[:, 1], hue=group)
        plt.title(gn)
        plt.plot(cur_pd.iloc[:, 0], cur_pd.loc[:, 'prediction'], c="k")
        plt.ylim(0, np.max(cur_pd.iloc[:, 0])*1.02)
        plt.xlim(0, np.max(cur_pd.iloc[:, 1])*1.02)
        minimal_yticks(0, np.max(vlm.Ux_sz[ix,:])*1.02)
        minimal_xticks(0, np.max(vlm.Sx_sz[ix,:])*1.02)
        despline()

        df_embedding = pd.concat([embedding, cur_pd.loc[:, 'gene']], ignore_index=False)
        sns.scatterplot(df_embedding.iloc[:, 0], df_embedding.iloc[:, 1], hue=df_embedding.loc[:, 'gene'])
        sns.scatterplot(df_embedding.iloc[:, 0], df_embedding.iloc[:, 1], hue=df_embedding.loc[:, 'velocity'])

    plt.tight_layout()

def plot_fitting(adata, gene, log = True, group = False):
    """GET_P estimates the posterior probability and part of the energy.

    Arguments
    ---------
        Y: 'np.ndarray'
            Original data.
        V: 'np.ndarray'
            Original data.
        sigma2: 'float'
            sigma2 is defined as sum(sum((Y - V)**2)) / (N * D)
        gamma: 'float'
            Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
        a: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.

    Returns
    -------
    P: 'np.ndarray'
        Posterior probability, related to equation 27.
    E: `np.ndarray'
        Energy, related to equation 26.

    """

    groups = [''] if group == False else np.unique(adata.obs[group])

    T = adata.obs['Time']
    gene_idx = np.where(adata.var.index.values == gene)[0][0]

    for cur_grp in groups:
        alpha, gamma, u0, l0 = adata.uns['dynamo_labeling'].loc[gene, :]
        u, l = adata.layers['U'][adata.obs[group] == cur_grp, gene_idx].toarray().squeeze(), \
                    adata.layers['L'][adata.obs[group] == cur_grp, gene_idx].toarray().squeeze()
        if log:
            u, l = np.log(u + 1), np.log(l + 1)

        t = np.linspace(T[0], T[-1], 50)

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(T, u.T, linestyle='None', marker='o', markersize=10)
        plt.plot(t, alpha/gamma + (u0 - alpha/gamma) * np.exp(-gamma*t), '--')
        plt.xlabel('time (hrs)')
        plt.title('unlabeled (' + cur_grp + ')')

        plt.subplot(122)
        plt.plot(T, l.T, linestyle='None', marker='o', markersize=10)
        plt.plot(t, l0 * np.exp(-gamma*t), '--')
        plt.xlabel('time (hrs)')
        plt.title('labeled (' + cur_grp + ')')

# moran'I on the velocity genes, etc.

# cellranger data, velocyto, comparison and phase diagram

def plot_LIC_gray(tex):
    """GET_P estimates the posterior probability and part of the energy.

    Arguments
    ---------
        Y: 'np.ndarray'
            Original data.
        V: 'np.ndarray'
            Original data.
        sigma2: 'float'
            sigma2 is defined as sum(sum((Y - V)**2)) / (N * D)
        gamma: 'float'
            Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
        a: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.

    Returns
    -------
    P: 'np.ndarray'
        Posterior probability, related to equation 27.
    E: `np.ndarray'
        Energy, related to equation 26.

    """

    tex = tex[:, ::-1]
    tex = tex.T

    M, N = tex.shape
    texture = np.empty((M, N, 4), np.float32)
    texture[:, :, 0] = tex
    texture[:, :, 1] = tex
    texture[:, :, 2] = tex
    texture[:, :, 3] = 1

#     texture = scipy.ndimage.rotate(texture,-90)
    plt.figure()
    plt.imshow(texture)


def plot_LIC(U_grid, V_grid, method = 'yt', cmap = "viridis", normalize = False, density = 1, lim=(0,1), const_alpha=False, kernellen=100, xlab='Dim 1', ylab='Dim 2', file = None):
    """Visualize vector field with quiver, streamline and line integral convolution (LIC), using velocity estimates on a grid from the associated data.
    A white noise background will be used for texture as default. Adjust the bounds of lim in the range of [0, 1] which applies
    upper and lower bounds to the values of line integral convolution and enhance the visibility of plots. When const_alpha=False,
    alpha will be weighted spatially by the values of line integral convolution; otherwise a constant value of the given alpha is used.

    Arguments
    ---------
        U_grid: 'np.ndarray'
            Original data.
        V_grid: 'np.ndarray'
            Original data.
        method: 'float'
            sigma2 is defined as sum(sum((Y - V)**2)) / (N * D)
        cmap: 'float'
            Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
        normalize: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.
        density: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.
        lim: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.
        const_alpha: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.
        kernellen: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.

    Returns
    -------
    P: 'np.ndarray'
        Posterior probability, related to equation 27.
    E: `np.ndarray'
        Energy, related to equation 26.

    """

    if method == 'yt':
        velocity_x_ori, velocity_y_ori, velocity_z_ori = U_grid, V_grid, np.zeros(U_grid.shape)
        velocity_x = np.repeat(velocity_x_ori[:, :, np.newaxis], V_grid.shape[1], axis=2)
        velocity_y = np.repeat(velocity_y_ori[:, :, np.newaxis], V_grid.shape[1], axis=2)
        velocity_z = np.repeat(velocity_z_ori[np.newaxis, :, :], V_grid.shape[1], axis=0)

        data = {}

        data["velocity_x"] = (velocity_x, "km/s")
        data["velocity_y"] = (velocity_y, "km/s")
        data["velocity_z"] = (velocity_z, "km/s")
        data["velocity_sum"] = (np.sqrt(velocity_x**2 + velocity_y**2), "km/s")

        ds = yt.load_uniform_grid(data, data["velocity_x"][0].shape, length_unit=(1.0,"Mpc"))
        slc = yt.SlicePlot(ds, "z", ["velocity_sum"])
        slc.set_cmap("velocity_sum", cmap)
        slc.set_log("velocity_sum", False)

        slc.annotate_velocity(normalize = normalize)
        slc.annotate_streamlines('velocity_x', 'velocity_y', density=density)
        slc.annotate_line_integral_convolution('velocity_x', 'velocity_y', lim=lim, const_alpha=const_alpha, kernellen=kernellen)

        slc.set_xlabel(xlab)
        slc.set_ylabel(ylab)

        slc.show()

        if file is not None:
            # plt.rc('font', family='serif', serif='Times')
            # plt.rc('text', usetex=True)
            # plt.rc('xtick', labelsize=8)
            # plt.rc('ytick', labelsize=8)
            # plt.rc('axes', labelsize=8)
            slc.save(file, mpl_kwargs = {figsize: [2, 2]})
    elif method == 'lic':
        velocyto_tex = runlic(V_grid, V_grid, 100)
        plot_LIC_gray(velocyto_tex)

