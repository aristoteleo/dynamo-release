# code adapted from https://github.com/lmcinnes/umap/blob/7e051d8f3c4adca90ca81eb45f6a9d1372c076cf/umap/plot.py
from ..configuration import _themes
from .utilities import despline, minimal_xticks, minimal_yticks

import numpy as np
import pandas as pd
from scipy.sparse import issparse

import numba
from warnings import warn

import matplotlib.colors
import matplotlib.cm
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

import datashader as ds
import datashader.transfer_functions as tf
import datashader.bundling as bd

import bokeh.plotting as bpl
import bokeh.transform as btr
from bokeh.plotting import output_notebook, output_file, show

import holoviews as hv
import holoviews.operation.datashader as hd


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


# https://stackoverflow.com/questions/8468855/convert-a-rgb-colour-value-to-decimal
"""Convert RGB color to decimal RGB integers are typically treated as three distinct bytes where the left-most (highest-order) 
byte is red, the middle byte is green and the right-most (lowest-order) byte is blue. """

@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _red(x):
    return (x & 0xFF0000) >> 16


@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _green(x):
    return (x & 0x00FF00) >> 8


@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _blue(x):
    return x & 0x0000FF


def _embed_datashader_in_an_axis(datashader_image, ax):
    img_rev = datashader_image.data[::-1]
    mpl_img = np.dstack([_blue(img_rev), _green(img_rev), _red(img_rev)])
    ax.imshow(mpl_img)
    return ax


def _get_extent(points):
    """Compute bounds on a space with appropriate padding"""
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    extent = (
        np.round(min_x - 0.05 * (max_x - min_x)),
        np.round(max_x + 0.05 * (max_x - min_x)),
        np.round(min_y - 0.05 * (max_y - min_y)),
        np.round(max_y + 0.05 * (max_y - min_y)),
    )

    return extent


def _select_font_color(background):
    if background == "black":
        font_color = "white"
    elif background.startswith("#"):
        mean_val = np.mean(
            [int("0x" + c) for c in (background[1:3], background[3:5], background[5:7])]
        )
        if mean_val > 126:
            font_color = "black"
        else:
            font_color = "white"

    else:
        font_color = "black"

    return font_color


def scatters(adata, genes, x=0, y=1, theme='fire', mode='splicing', type='expression', vkey='S', ekey='X', basis='umap', n_columns=1, \
             color=None, figsize=None, legend=False, ax=None, **kwargs):
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
        theme: `str` (optional, default `None`)
            A color theme to use for plotting. A small set of
            predefined themes are provided which have relatively
            good aesthetics. Available themes are:
               * 'blue'
               * 'red'
               * 'green'
               * 'inferno'
               * 'fire'
               * 'viridis'
               * 'darkblue'
               * 'darkred'
               * 'darkgreen'
        mode: `string` (default: labelling)
            Which mode of data do you want to show, can be one of `labelling`, `splicing` and `full`.
        type: `str` (default: `expression`)
            Which plotting type to use, either embedding, expression, velocity or phase.
        vkey: `string` (default: velocity)
            Which velocity key used for visualizing the magnitude of velocity. Can be either velocity in the layers slot or the
            keys in the obsm slot.
        ekey: `str`
            The layer of data to represent the gene expression level.
        basis: `string` (default: umap)
            Which low dimensional embedding will be used to visualize the cell.
        figsize: `None` or `[float, float]` (default: None)
            The width and height of a figure.
        figsize: `None` or `[float, float]` (default: None)
            The width and height of a figure.
        legend: `False`, `brief` or `full`
            the legend parameter in seaborn. How to draw the legend. If “brief”, numeric hue and size variables will be
            represented with a sample of evenly spaced values. If “full”, every group will get an entry in the legend.
            If False, no legend data is added and no legend is drawn.
        **kwargs:
            Additional parameters that will be passed to plt.scatter function

    Returns
    -------
        Nothing but a scatter plot of cells.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    font_color = _select_font_color(background)
    point_size = 100.0 / np.sqrt(adata.shape[0])
    scatter_kwargs = dict(alpha=0.4, s=point_size, edgecolor=None, linewidth=0) # (0, 0, 0, 1)
    if kwargs is not None:
        scatter_kwargs.update(kwargs)

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

    color_vec = np.repeat(np.nan, n_cells)
    if color is not None:
        color = list(set(color).intersection(adata.obs.keys()))
        if len(color) > 0 and type is not 'embedding':
            color_vec = adata.obs[color[0]].values
        else:
            n_genes = len(color)
            color_vec = adata.obs[color[0]].values
            full_color_vec = adata.obs[color].values.flatten()

    if type is 'embedding':
        df = pd.DataFrame({basis + '_0': np.repeat(embedding.iloc[:, 0], n_genes), basis + '_1': np.repeat(embedding.iloc[:, 1], n_genes),
                           "color": full_color_vec, "group": np.tile(color, n_cells)})
    else:
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

        if 'gamma' in adata.var.columns:
            gamma = adata.var.gamma[genes].values
            velocity_offset = [0] * n_genes if not ("gamma_b" in adata.var.columns) else \
                adata.var.gamma_b[genes].values
        else:
            raise Exception('adata does not seem to have gamma column. Velocity estimation is required before '
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

    if type is not 'embedding':
        if type is "phase":
            if df.color.unique() != np.nan:
                if theme is None: theme = 'viridis'
                cmap = _themes[theme]["cmap"]
                color_key_cmap = _themes[theme]["color_key_cmap"]
                background = _themes[theme]["background"]

                # num_labels = unique_labels.shape[0]
                # color_key = plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
                # legend_elements = [
                #     Patch(facecolor=color_key[i], label=unique_labels[i])
                #     for i, k in enumerate(unique_labels)
                # ]

            else:
                if theme is None: theme = 'fire'
                cmap = _themes[theme]["cmap"]
                color_key_cmap = _themes[theme]["color_key_cmap"]
                background = _themes[theme]["background"]
        elif type is "velocity":
            if theme is None: theme = 'fire'
            cmap = _themes[theme]["cmap"]
            color_key_cmap = _themes[theme]["color_key_cmap"]
            background = _themes[theme]["background"]
        elif type is 'expression':
            if theme is None: theme = 'green'
            cmap = _themes[theme]["cmap"]
            color_key_cmap = _themes[theme]["color_key_cmap"]
            background = _themes[theme]["background"]
    else:
        if theme is None: theme = 'blue'
        cmap = _themes[theme]["cmap"]
        color_key_cmap = _themes[theme]["color_key_cmap"]
        background = _themes[theme]["background"]

    if type is not 'embedding':
        n_columns = 2 * n_columns if ('protein' in adata.obsm.keys() and mode is 'full') else n_columns
        plot_per_gene = 2 if ('protein' in adata.obsm.keys() and mode is 'full') else 1
        nrow, ncol = int(np.ceil(plot_per_gene * n_genes / n_columns)), n_columns
        if figsize is None:
            plt.figure(None, (3 * ncol, 3 * nrow), facecolor=background)  # , dpi=160
        else:
            plt.figure(None, (figsize[0] * ncol, figsize[1] * nrow), facecolor=background)  # , dpi=160

        # the following code is inspired by https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb
        gs = plt.GridSpec(nrow, ncol)

        for i, gn in enumerate(genes):
            if plot_per_gene is 2:
                ax1, ax2 = plt.subplot(gs[i*2]), plt.subplot(gs[i*2+1])
            elif plot_per_gene is 1:
                ax1 = plt.subplot(gs[i])
            try:
                ix=np.where(adata.var.index == gn)[0][0]
            except:
                continue
            cur_pd = df.loc[df.gene == gn, :]
            if type is 'phase': # viridis, set2
                if cur_pd.color.unique() != np.nan:
                    g = sns.scatterplot(cur_pd.iloc[:, 1], cur_pd.iloc[:, 0], hue="expression", ax=ax1, palette=cmap, legend=legend, **scatter_kwargs) # x-axis: S vs y-axis: U
                else:
                    g = sns.scatterplot(cur_pd.iloc[:, 1], cur_pd.iloc[:, 0], hue=color, ax=ax1, palette=cmap, legend=legend, **scatter_kwargs) # x-axis: S vs y-axis: U

                if legend:
                    g.legend(loc='center left', bbox_to_anchor=(0.125, 0.125), ncol=1)

                ax1.set_title(gn)
                xnew = np.linspace(0, cur_pd.iloc[:, 1].max())
                ax1.plot(xnew, xnew * cur_pd.loc[:, 'gamma'].unique() + cur_pd.loc[:, 'velocity_offset'].unique(), c="k")
                ax1.set_xlim(0, np.max(cur_pd.iloc[:, 1])*1.02)
                ax1.set_ylim(0, np.max(cur_pd.iloc[:, 0])*1.02)

                despline(ax1) # sns.despline()

                if plot_per_gene == 2 and ('protein' in adata.obsm.keys() and mode is 'full' and all([i in adata.layers.keys() for i in ['uu', 'ul', 'su', 'sl']])):
                    g = sns.scatterplot(cur_pd.iloc[:, 3], cur_pd.iloc[:, 2], hue=color, ax=ax2, legend=legend, **scatter_kwargs)  # x-axis: Protein vs. y-axis: Spliced
                    if legend:
                        g.legend(loc='center left', bbox_to_anchor=(0.125, 0.125), ncol=1)

                    ax2.set_title(gn)
                    xnew = np.linspace(0, cur_pd.iloc[:, 3].max())
                    ax2.plot(xnew, xnew * cur_pd.loc[:, 'gamma_P'].unique() + cur_pd.loc[:, 'velocity_offset_P'].unique(),
                             c="k")

                    ax2.set_ylim(0, np.max(cur_pd.iloc[:, 3]) * 1.02)
                    ax2.set_xlim(0, np.max(cur_pd.iloc[:, 2]) * 1.02)

                    despline(ax2)  # sns.despline()

            elif type is 'velocity':
                df_embedding = pd.concat([cur_pd, embedding], axis=1)
                V_vec = df_embedding.loc[:, 'velocity']

                limit = np.nanmax(np.abs(np.nanpercentile(V_vec, [1, 99])))  # upper and lowe limit / saturation

                V_vec = V_vec + limit  # that is: tmp_colorandum - (-limit)
                V_vec = V_vec / (2 * limit)  # that is: tmp_colorandum / (limit - (-limit))
                V_vec = np.clip(V_vec, 0, 1)

                cmap = plt.cm.RdBu_r # sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
                g=sns.scatterplot(embedding.iloc[:, 0], embedding.iloc[:, 1], hue=df_embedding.loc[:, 'expression'], ax=ax1, \
                                palette=cmap, legend=legend, **scatter_kwargs)
                if legend:
                    g.legend(loc='center left', bbox_to_anchor=(0.125, 0.125), ncol=1)

                ax1.set_title(gn + '(' + ekey + ')')
                ax1.set_xlabel(basis + '_1')
                ax1.set_ylabel(basis + '_2')

                if plot_per_gene == 2:
                    V_vec = df_embedding.loc[:, 'velocity_offset_P']

                    limit = np.nanmax(np.abs(np.nanpercentile(V_vec, [1, 99])))  # upper and lowe limit / saturation

                    V_vec = V_vec + limit  # that is: tmp_colorandum - (-limit)
                    V_vec = V_vec / (2 * limit)  # that is: tmp_colorandum / (limit - (-limit))
                    V_vec = np.clip(V_vec, 0, 1)

                    cmap = plt.cm.RdBu_r  # sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
                    g = sns.scatterplot(embedding.iloc[:, 0], embedding.iloc[:, 1], hue=V_vec, ax=ax2, palette=cmap, legend=legend, **scatter_kwargs)

                    if legend:
                        g.legend(loc='center left', bbox_to_anchor=(0.125, 0.125), ncol=1)

                    ax2.set_title(gn + '(' + ekey + ')')
                    ax2.set_xlabel(basis + '_1')
                    ax2.set_ylabel(basis + '_2')
            elif type is 'expression':
                cmap = plt.cm.Greens # sns.diverging_palette(10, 220, sep=80, as_cmap=True)
                g = sns.scatterplot(embedding.iloc[:, 0], embedding.iloc[:, 1], hue=df_embedding.loc[:, 'expression'], ax=ax1, \
                                palette=cmap, legend=legend, **scatter_kwargs)

                if legend:
                    g.legend(loc='center left', bbox_to_anchor=(0.125, 0.125), ncol=1)

                ax1.set_title(gn + '(' + vkey + ')')
                ax1.set_xlabel(basis + '_1')
                ax1.set_ylabel(basis + '_2')

                if 'protein' in adata.obsm.keys() and mode is 'full' and all([i in adata.layers.keys() for i in ['uu', 'ul', 'su', 'sl']]):
                    df_embedding = pd.concat([embedding, cur_pd.loc[:, 'P']], ignore_index=False)

                    cmap = sns.light_palette("navy", as_cmap=True)
                    g = sns.scatterplot(embedding.iloc[:, 0], embedding.iloc[:, 1], hue=df_embedding.loc[:, 'expression'], \
                                    ax=ax2, legend=legend, palette=cmap, **scatter_kwargs)
                    if legend:
                        g.legend(loc='center left', bbox_to_anchor=(0.125, 0.125), ncol=1)

                    ax2.set_title(gn + '(protein expression)')
                    ax2.set_xlabel(basis + '_1')
                    ax2.set_ylabel(basis + '_2')

    else:
        nrow, ncol = int(np.ceil(len(color) / n_columns)), n_columns
        if figsize is None:
            plt.figure(None, (3 * ncol, 3 * nrow), facecolor=background)  # , dpi=160
        else:
            plt.figure(None, (figsize[0] * ncol, figsize[1] * nrow), facecolor=background)  # , dpi=160

        # the following code is inspired by https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb
        gs = plt.GridSpec(nrow, ncol)

        for i, clr in enumerate(color):
            ax1 = plt.subplot(gs[i])

            cur_pd = df.loc[df.group == clr, :]

            cmap = plt.cm.Set2 #if type(cur_pd.loc[:, 'color'][0]) is not float else plt.cm.Greens # sns.diverging_palette(10, 220, sep=80, as_cmap=True)
            g=sns.scatterplot(embedding.iloc[:, 0], embedding.iloc[:, 1], hue=cur_pd.loc[:, 'color'], ax=ax1, palette=cmap, \
                            legend=legend, **scatter_kwargs)
            if legend:
                g.legend(loc='center left', bbox_to_anchor=(0.125, 0.125), ncol=1)

            ax1.set_title(color)
            ax1.set_xlabel(basis + '_1')
            ax1.set_ylabel(basis + '_2')

    plt.tight_layout()
    plt.show()


