import os
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.stats import norm as normal
import bezier
import numpy as np
import pandas as pd
from .colormap import *

if __name__ == "__main__":
    sys.path.append('..')
    from utilities import find_nn_neighbors, extract_from_df
else:
    from celldancer.utilities import find_nn_neighbors, extract_from_df

def scatter_cell(
    ax,
    cellDancer_df, 
    colors=None, 
    custom_xlim=None,
    custom_ylim=None,
    vmin=None,
    vmax=None,
    alpha=0.5, 
    s = 5,
    legend_marker_size=5,
    gene=None,
    velocity=False,
    legend='off',
    colorbar='on',
    min_mass=2,
    arrow_grid=(30,30)
): 

    """Plot the RNA velocity on the embedding space; or plot the kinetic parameters ('alpha', 'beta', 'gamma', 'splice', 'unsplice', or 'pseudotime') of one gene on the embedding space.
        
    Arguments
    ---------
    ax: `ax`
        ax of plt.subplots()
    cellDancer_df: `pandas.DataFrame`
        Dataframe of velocity estimation, cell velocity, and pseudotime results. Columns=['cellIndex', 'gene_name', 'unsplice', 'splice', 'unsplice_predict', 'splice_predict', 'alpha', 'beta', 'gamma', 'loss', 'cellID', 'clusters', 'embedding1', 'embedding2', 'velocity1', 'velocity2', 'pseudotime']
    colors: `list`, `dict`, or `str`
        When the input is a list: build a colormap dictionary for a list of cell type;  
        When the input is a dictionary: it is the customized color map dictionary of each cell type; 
        When the input is a str: one of {'alpha', 'beta', 'gamma', 'splice', 'unsplice', 'pseudotime'} is used as input.
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
    legend_marker_size: optional, `float` (default: 5)
        The lengend marker size.
    gene: optional, `str` (default: None)
        Gene name for plotting.
    velocity: optional, `bool` (default: False)
        `True` if plot velocity.
    legend: optional, `str` (default: 'off')
        `'off'` if the color map of cell legend is not plotted. 
        `'only'` if only plot the cell type legend.
    colorbar: optional, `str` (default: 'on')
        `‘on’` if the colorbar of the plot of `alpha`, `beta`, `gamma`, `splice`, or `unsplice` is to be shown. `'off'` if the colorbar is to be not shown.
    min_mass: optional, `float` (default: 2)
        Filter by using the isotropic gaussian kernel to display the arrow on grids. The lower the min_mass, the more arrows.
    arrow_grid: optional, `tuple` (default: (30,30))
        The sparsity of the grids of velocity arrows. The larger, the more compact, and more arrows will be shown.
    Returns
    -------
    ax: matplotlib.axes.Axes
    """  

    def gen_Line2D(label, markerfacecolor):
        return Line2D([0], [0], color='w', marker='o', label=label,
            markerfacecolor=markerfacecolor, 
            markeredgewidth=0,
            markersize=legend_marker_size)

    if isinstance(colors, (list, tuple)):
        #print("\nbuild a colormap for a list of clusters as input\n")
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

        c=np.vectorize(colors.get)(extract_from_df(cellDancer_df, 'clusters', gene))
        cmap=ListedColormap(list(colors.values()))
    elif isinstance(colors, str):
        attr = colors
        if colors in ['alpha', 'beta', 'gamma']:
            assert gene, '\nError! gene is required!\n'
            cmap = LinearSegmentedColormap.from_list("mycmap", colors_alpha_beta_gamma)
        if colors in ['splice', 'unsplice']:
            assert gene, '\nError! gene is required!\n'
            colors = {'splice':'splice', 'unsplice':'unsplice'}[colors]
            cmap = LinearSegmentedColormap.from_list("mycmap",
                    colors_splice_unsplice)
        if colors in ['pseudotime']:
            cmap = 'viridis'
        c = extract_from_df(cellDancer_df, [colors], gene)
        
    elif colors is None:
        attr = 'basic'
        cmap = None
        c = 'Grey'
    
    embedding = extract_from_df(cellDancer_df, ['embedding1', 'embedding2'], gene)
    n_cells = embedding.shape[0]
    
    im=ax.scatter(embedding[:, 0],
                embedding[:, 1],
                c=c,
                cmap=cmap,
                s=s,
                vmin=vmin,
                vmax=vmax,
                alpha=alpha,
                edgecolor="none")
    if colorbar == 'on' and isinstance(colors, str):
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("top", size="5%", pad="-5%")

        # print("   \n ")
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal", shrink=0.1)
        cbar.set_ticks([])

    if velocity:
        sample_cells = cellDancer_df['velocity1'][:n_cells].dropna().index
        embedding_ds = embedding[sample_cells]
        velocity_embedding= extract_from_df(cellDancer_df, ['velocity1', 'velocity2'], gene)
        grid_curve(ax, embedding_ds, velocity_embedding, arrow_grid, min_mass)

    if custom_xlim is not None:
        ax.set_xlim(custom_xlim[0], custom_xlim[1])
    if custom_ylim is not None:
        ax.set_ylim(custom_ylim[0], custom_ylim[1])
    
    return ax

def grid_curve(
    ax, 
    embedding_ds, 
    velocity_embedding, 
    arrow_grid, 
    min_mass
):
    # calculate_grid_arrows
    # kernel grid plot

    def calculate_two_end_grid(embedding_ds, velocity_embedding, smooth=None, steps=None, min_mass=None):
        # Prepare the grid
        grs = []
        for dim_i in range(embedding_ds.shape[1]):
            m, M = np.min(embedding_ds[:, dim_i])-0.2, np.max(embedding_ds[:, dim_i])-0.2
            m = m - 0.025 * np.abs(M - m)
            M = M + 0.025 * np.abs(M - m)
            gr = np.linspace(m, M, steps[dim_i])
            grs.append(gr)

        meshes_tuple = np.meshgrid(*grs)
        gridpoints_coordinates = np.vstack(
            [i.flat for i in meshes_tuple]).T

        n_neighbors = int(velocity_embedding.shape[0]/3)
        dists_head, neighs_head = find_nn_neighbors(
            embedding_ds, gridpoints_coordinates, n_neighbors)
        dists_tail, neighs_tail = find_nn_neighbors(
            embedding_ds+velocity_embedding, gridpoints_coordinates,
            n_neighbors)
        std = np.mean([(g[1] - g[0]) for g in grs])

        # isotropic gaussian kernel
        gaussian_w_head = normal.pdf(
            loc=0, scale=smooth * std, x=dists_head)
        total_p_mass_head = gaussian_w_head.sum(1)
        gaussian_w_tail = normal.pdf(
            loc=0, scale=smooth * std, x=dists_tail)
        total_p_mass_tail = gaussian_w_tail.sum(1)

        
        UZ_head = (velocity_embedding[neighs_head] * gaussian_w_head[:, :, None]).sum(
            1) / np.maximum(1, total_p_mass_head)[:, None]  # weighed average
        UZ_tail = (velocity_embedding[neighs_tail] * gaussian_w_tail[:, :, None]).sum(
            1) / np.maximum(1, total_p_mass_tail)[:, None]  # weighed average

        XY = gridpoints_coordinates

        dists_head2, neighs_head2 = find_nn_neighbors(
            embedding_ds, XY+UZ_head, n_neighbors)
        dists_tail2, neighs_tail2 = find_nn_neighbors(
            embedding_ds, XY-UZ_tail, n_neighbors)

        gaussian_w_head2 = normal.pdf(
            loc=0, scale=smooth * std, x=dists_head2)
        total_p_mass_head2 = gaussian_w_head2.sum(1)
        gaussian_w_tail2 = normal.pdf(
            loc=0, scale=smooth * std, x=dists_tail2)
        total_p_mass_tail2 = gaussian_w_tail2.sum(1)

        UZ_head2 = (velocity_embedding[neighs_head2] * gaussian_w_head2[:, :, None]).sum(
            1) / np.maximum(1, total_p_mass_head2)[:, None]  # weighed average
        UZ_tail2 = (velocity_embedding[neighs_tail2] * gaussian_w_tail2[:, :, None]).sum(
            1) / np.maximum(1, total_p_mass_tail2)[:, None]  # weighed average

        mass_filter = total_p_mass_head < min_mass

        # filter dots
        UZ_head_filtered = UZ_head[~mass_filter, :]
        UZ_tail_filtered = UZ_tail[~mass_filter, :]
        UZ_head2_filtered = UZ_head2[~mass_filter, :]
        UZ_tail2_filtered = UZ_tail2[~mass_filter, :]
        XY_filtered = XY[~mass_filter, :]
        return(XY_filtered, UZ_head_filtered, UZ_tail_filtered, UZ_head2_filtered, UZ_tail2_filtered, mass_filter, grs)

    XY_filtered, UZ_head_filtered, UZ_tail_filtered, UZ_head2_filtered, UZ_tail2_filtered, mass_filter, grs = calculate_two_end_grid(
        embedding_ds, velocity_embedding, smooth=0.8, steps=arrow_grid, min_mass=min_mass)

    # connect two end grid to curve
    n_curves = XY_filtered.shape[0]
    s_vals = np.linspace(0.0, 1.5, 15) # TODO check last
    # get longest distance len and norm ratio
    XYM = XY_filtered
    UVT = UZ_tail_filtered
    UVH = UZ_head_filtered
    UVT2 = UZ_tail2_filtered
    UVH2 = UZ_head2_filtered

    def norm_arrow_display_ratio(XYM, UVT, UVH, UVT2, UVH2, grs, s_vals):
        '''get the longest distance in prediction between the five points,
        and normalize by using the distance between two grids'''

        def distance(x, y):
            # calc disctnce list between a set of coordinate
            calculate_square = np.subtract(
                x[0:-1], x[1:])**2 + np.subtract(y[0:-1], y[1:])**2
            distance_result = (calculate_square)**0.5
            return distance_result

        max_discance = 0
        for i in range(n_curves):
            nodes = np.asfortranarray([[XYM[i, 0]-UVT[i, 0]-UVT2[i, 0], XYM[i, 0]-UVT[i, 0], XYM[i, 0], XYM[i, 0]+UVH[i, 0], XYM[i, 0]+UVH[i, 0]+UVH2[i, 0]],
                                        [XYM[i, 1]-UVT[i, 1]-UVT2[i, 1], XYM[i, 1]-UVT[i, 1], XYM[i, 1], XYM[i, 1]+UVH[i, 1], XYM[i, 1]+UVH[i, 1]+UVH2[i, 1]]])
            curve = bezier.Curve(nodes, degree=4)
            curve_dots = curve.evaluate_multi(s_vals)
            distance_sum = np.sum(
                distance(curve_dots[0], curve_dots[1]))
            max_discance = max(max_discance, distance_sum)
        distance_grid = (
            abs(grs[0][0]-grs[0][1]) + abs(grs[1][0]-grs[1][1]))/2
        norm_ratio = distance_grid/max_discance
        return(norm_ratio)

    norm_ratio = norm_arrow_display_ratio(XYM, UVT, UVH, UVT2, UVH2, grs, s_vals)

    # plot the curve arrow for cell velocity
    XYM = XY_filtered
    UVT = UZ_tail_filtered * norm_ratio
    UVH = UZ_head_filtered * norm_ratio
    UVT2 = UZ_tail2_filtered * norm_ratio
    UVH2 = UZ_head2_filtered * norm_ratio

    def plot_cell_velocity_curve(XYM, UVT, UVH, UVT2, UVH2, s_vals):
        # TO DO: add 'colorful cell velocity' to here, now there is only curve arrows
        for i in range(n_curves):
            nodes = np.asfortranarray([[XYM[i, 0]-UVT[i, 0]-UVT2[i, 0], XYM[i, 0]-UVT[i, 0], XYM[i, 0], XYM[i, 0]+UVH[i, 0], XYM[i, 0]+UVH[i, 0]+UVH2[i, 0]],
                                        [XYM[i, 1]-UVT[i, 1]-UVT2[i, 1], XYM[i, 1]-UVT[i, 1], XYM[i, 1], XYM[i, 1]+UVH[i, 1], XYM[i, 1]+UVH[i, 1]+UVH2[i, 1]]])
            curve = bezier.Curve(nodes, degree=4)
            curve_dots = curve.evaluate_multi(s_vals)
            ax.plot(curve_dots[0], curve_dots[1],
                        linewidth=0.5, color='black', alpha=1)

            # normalize the arrow of the last two points at the tail, to let all arrows has the same size in quiver
            U = curve_dots[0][-1]-curve_dots[0][-2]
            V = curve_dots[1][-1]-curve_dots[1][-2]
            N = np.sqrt(U**2 + V**2)
            U1, V1 = U/N*0.5, V/N*0.5  # 0.5 is to let the arrow have a suitable size
            ax.quiver(curve_dots[0][-2], curve_dots[1][-2], U1, V1, units='xy', angles='xy',
                        scale=1, linewidth=0, color='black', alpha=1, minlength=0, width=0.1)

    plot_cell_velocity_curve(XYM, UVT, UVH, UVT2, UVH2, s_vals)


def plot_kinetic_para(
    ax,
    kinetic_para,
    cellDancer_df,
    color_map=None,
    title=None,
    legend=False
):

    """Plot the UMAP calculated by the kinetic parameter(s).
        
    Arguments
    ---------
    ax: `ax`
        ax of plt.subplots()
    kinetic_para: `str`
        The parameter used to generate the embedding space based on UMAP, could be selected from {'alpha', 'beta', 'gamma', 'alpha_beta_gamma'}.
    cellDancer_df: `pandas.DataFrame`
        Dataframe of velocity estimation results. Columns=['cellIndex', 'gene_name', 'splice', 'unsplice', 'splice_predict', 'unsplice_predict', 'alpha', 'beta', 'gamma', 'loss', 'cellID', 'clusters', 'embedding1', 'embedding2']
    color_map: `dict` (optional, default: None)
        The color map dictionary of each cell type.
    legend: `bool` (optional, default: False)
        `True` if the color map of cell legend is to be plotted. 
    """    
    onegene=cellDancer_df[cellDancer_df.gene_name==cellDancer_df.gene_name[0]]
    umap_para=onegene[[(kinetic_para+'_umap1'),(kinetic_para+'_umap2')]].to_numpy()
    onegene_cluster_info=onegene.clusters
    
    gene=None
    if gene is None:
        if color_map is None:
            from .colormap import build_colormap
            color_map=build_colormap(onegene_cluster_info)

        colors = list(map(lambda x: color_map.get(x, 'black'), onegene_cluster_info))

        if legend:
            markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_map.values()]
            lgd=plt.legend(markers, color_map.keys(), numpoints=1,loc='upper left',bbox_to_anchor=(1.01, 1))
                
        im=ax.scatter(umap_para[:,0], umap_para[:,1],c=colors,s=15,alpha=0.5,edgecolor="none")
        ax.axis('square')
        ax.axis('off')
        ax.set_title('UMAP of '+ kinetic_para)

    else:
        onegene=cellDancer_df[cellDancer_df.gene_name==gene]
        im=ax.scatter(umap_para[:,0], umap_para[:,1],c=np.log(onegene.splice+0.0001),s=15,alpha=1,edgecolor="none")
        ax.axis('square')
        ax.axis('off')
        ax.set_title('spliced reads of '+gene+'\n on UMAP of \n'+ kinetic_para)
        
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("top", size="5%", pad="-5%")
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal", shrink=0.1)
        cbar.set_ticks([])
        
    umap_df=pd.concat([pd.DataFrame({'umap1':umap_para[:,0],'umap2':umap_para[:,1]})],axis=1)
    
    return ax