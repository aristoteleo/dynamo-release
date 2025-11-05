import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from .colormap import *
from ..sampling import sampling_neighbors
from ..utilities import extract_from_df

def scatter_gene(
    ax=None,
    x=None,
    y=None,
    cellDancer_df=None,
    colors=None,
    custom_xlim=None,
    custom_ylim=None,
    vmin=None,
    vmax=None,
    alpha=0.5, 
    s = 5,
    velocity=False,
    gene=None,
    legend='off',
    arrow_grid = (15,15)):

    """Plot the velocity (splice-unsplice) of a gene, or plot the parameter ('alpha', 'beta', 'gamma', 'splice', 'unsplice') in pseudotime, or customize the parameters in x-axis and y-axis of a gene.
        
    Arguments
    ---------
    ax: `ax of plt.subplots()`
        ax to add subplot.
    x: `str`
        Set x axis as one of {'splice', 'unsplice', 'alpha', 'beta', 'gamma', 'pseudotime'}.
    y: `str`
        Set y axis as one of {'splice', 'unsplice', 'alpha', 'beta', 'gamma', 'pseudotime'}.
    cellDancer_df: `pandas.DataFrame`
        Dataframe of velocity estimation, cell velocity, and pseudotime results. Columns=['cellIndex', 'gene_name', 'unsplice', 'splice', 'unsplice_predict', 'splice_predict', 'alpha', 'beta', 'gamma', 'loss', 'cellID', 'clusters', 'embedding1', 'embedding2', 'velocity1', 'velocity2', 'pseudotime']
    colors: `list`, `dict`, or `str`
        When the input is a list: build a colormap dictionary for a list of cell type; 
        When the input is a dictionary: the customized color map dictionary of each cell type; 
        When the input is a str: one of {'alpha', 'beta', 'gamma', 'splice', 'unsplice', 'pseudotime'} is used as value of color.
    custom_xlim: optional, `float` (default: None)
        Set the x limit of the current axes.
    custom_ylim: optional, `float` (default: None)
        Set the y limit of the current axes.
    vmin: optional, `float` (default: None)
        Set the minimum color limit of the current image.
    vmax: optional, `float` (default: None)
        Set the maximum color limit of the current image.
    alpha: optional, `float` (default: 0.5)
        The alpha blending value, between 0 (transparent) and 1 (opaque).
    s: optional, `float` (default: 5)
        The marker size.
    velocity: optional, `bool` (default: False)
        `True` if velocity in gene level is to be plotted.
    gene: optional, `str` (default: None)
        Gene selected to be plotted.
    legend: optional, `str` (default: 'off')
        `‘off’` if the color map of cell type legend is not to be plotted;
        `‘only’` if only plot the cell type legend.
    arrow_grid: optional, `tuple` (default: (15,15))
        The sparsity of the grids of velocity arrows. The larger, the more compact and more arrows will be shown.

    Returns
    -------
    ax: matplotlib.axes.Axes
    """ 

    def gen_Line2D(label, markerfacecolor):
        return Line2D([0], [0], color='w', marker='o', label=label,
            markerfacecolor=markerfacecolor,
            markeredgewidth=0,
            markersize=s)
    
    if isinstance(colors, list):
        colors = build_colormap(colors)

    if isinstance(colors, dict):
        attr = 'clusters'
        legend_elements= [gen_Line2D(i, colors[i]) for i in colors]
        if legend != 'off':
            lgd=ax.legend(handles=legend_elements,
                bbox_to_anchor=(1.01, 1),
                loc='upper left')
            bbox_extra_artists=(lgd,)
            if legend == 'only':
                return lgd
        else:
            bbox_extra_artists=None

        c=np.vectorize(colors.get)(extract_from_df(cellDancer_df, 'clusters'))
        cmap=ListedColormap(list(colors.values()))

    elif isinstance(colors, str):
        attr = colors
        if colors in ['alpha', 'beta', 'gamma']:
            assert gene, '\nError! gene is required!\n'
            cmap = ListedColormap(colors_alpha_beta_gamma)
        if colors in ['splice', 'unsplice']:
            assert gene, '\nError! gene is required!\n'
            cmap = ListedColormap(colors_splice_unsplice)
        if colors in ['pseudotime']:
            cmap = 'viridis'
        else:
            cmap = 'viridis'

        c = extract_from_df(cellDancer_df, [colors], gene)
    elif colors is None:
        attr = 'basic'
        cmap = None
        c = '#95D9EF'
    
    assert gene, '\nError! gene is required!\n'
    xy = extract_from_df(cellDancer_df, [x, y], gene)
    ax.scatter(xy[:, 0],
               xy[:, 1],
               c=c,
               cmap=cmap,
               s=s,
               alpha=alpha,
               vmin=vmin,
               vmax=vmax,
               edgecolor="none")

    if custom_xlim is not None:
        ax.set_xlim(custom_xlim[0], custom_xlim[1])
    if custom_ylim is not None:
        ax.set_ylim(custom_ylim[0], custom_ylim[1])

                                 
    if velocity:
        assert (x,y) in [('unsplice', 'splice'), ('splice', 'unsplice')]
        u_s = extract_from_df(cellDancer_df, ['unsplice','splice','unsplice_predict','splice_predict'], gene)
        sampling_idx=sampling_neighbors(u_s[:,0:2], step=arrow_grid, percentile=15) # Sampling
        u_s_downsample = u_s[sampling_idx,0:4]

        plt.scatter(u_s_downsample[:, 1], u_s_downsample[:,0], color="none", s=s, edgecolor="k")
        plt.quiver(u_s_downsample[:, 1], u_s_downsample[:, 0], 
                   u_s_downsample[:, 3]-u_s_downsample[:, 1], 
                   u_s_downsample[:, 2]-u_s_downsample[:, 0],
                   angles='xy', clim=(0., 1.))

    return ax

