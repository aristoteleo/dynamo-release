import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from ..tools.utils import flatten, isarray, velocity_on_grid
#from ..tools.Markov import smoothen_drift_on_grid

SchemeDiverge = {
    'cmap': 'Spectral_r', 
    'sym_c': True, 
    'sort_by_c': 'abs',
    }

SchemeDivergeBWR = {
    'cmap': 'bwr', 
    'sym_c': True, 
    'sort_by_c': 'abs',
    }

def plot_X(X, dim1=0, dim2=1, dim3=None, create_figure=False, figsize=(6, 6), 
    sort_by_c='raw', **kwargs):
    if create_figure:
        plt.figure(figsize=figsize)
    
    x, y = X[:, dim1], X[:, dim2]
    c = kwargs.pop('c', None)
    if c is not None and isarray(c):
        if sort_by_c is not None:
            if sort_by_c == 'neg':
                i_sort = np.argsort(-c)
            elif sort_by_c == 'abs':
                i_sort = np.argsort(np.abs(c))
            elif sort_by_c == 'raw':
                i_sort = np.argsort(c)
            x = x[i_sort]
            y = y[i_sort]
            c = c[i_sort]
            if dim3 is not None: 
                z = X[:, dim3][i_sort]
    if dim3 is None:        
        plt.scatter(x, y, c=c, **kwargs)
    else:
        from mpl_toolkits.mplot3d import Axes3D
        plt.gcf().add_subplot(111, projection='3d')
        plt.gca().scatter(x, y, z, c=c, **kwargs)


def plot_V(X, V, dim1=0, dim2=1, create_figure=False, figsize=(6, 6), **kwargs):
    if create_figure:
        plt.figure(figsize=figsize)
    plt.quiver(X[:, dim1], X[:, dim2], V[:, dim1], V[:, dim2])


def zscatter(adata, basis='umap', layer='X', dim1=0, dim2=1, dim3=None,
    color=None, c_layer=None, cbar=True, cbar_shrink=0.4, sym_c=False,
    axis_off=True, **kwargs):

    if layer is None or len(layer) == 0:
        emb = basis
    else:
        emb = '%s_%s'%(layer, basis)
    X = adata.obsm[emb]
    title = None
    if not isarray(color):
        if color in adata.var.index:
            title = color
            if c_layer is None:
                color = flatten(adata[:, color].X)
            else:
                color = flatten(adata[:, color].layers[c_layer])
        elif color in adata.obs.keys():
            title = color
            #color = flatten(np.array(adata.obs[color])) 
            color = adata.obs[color]

    # categorical data
    if color is not None and type(color) is not str and np.any([type(a) is str for a in color]):
        cat_color = True
        try:
            cat = color.cat.categories
        except:
            cat = np.unique(color)
        value_dict = {c: i for i, c in enumerate(cat)}
        if title + '_colors' in adata.uns.keys():
            color_dict = adata.uns[title + '_colors']
            if type(color_dict) is dict:
                color_map = ListedColormap([color_dict[c] for c in cat])
            else:
                color_map = ListedColormap(color_dict)
        else:
            color_map = cm.get_cmap('tab20')
        color = np.array([value_dict[i] for i in color])
        
    else:
        cat_color = False
        color_map = None

    if color_map is None:
        plot_X(X, dim1=dim1, dim2=dim2, dim3=dim3, c=color, **kwargs)
    else:
        plot_X(X, dim1=dim1, dim2=dim2, dim3=dim3, c=color, cmap=color_map, **kwargs)
    if isarray(color):
        if cbar: 
            if cat_color:
                cb = plt.colorbar(ticks=[i for i in value_dict.values()], 
                    values=[i for i in value_dict.values()], 
                    shrink=cbar_shrink)
                cb.ax.set_yticklabels(value_dict.keys())
            else:
                plt.colorbar(shrink=cbar_shrink)
        if sym_c:
            bounds = max(np.abs(np.nanmax(color)), np.abs(np.nanmin(color)))
            bounds = bounds * np.array([-1, 1])
            plt.clim(bounds[0], bounds[1])
    if title is not None:
        plt.title(title)

    if axis_off:
        plt.axis('off')


def zstreamline(adata, basis="umap", v_basis=None, x_layer='X', v_layer='velocity',
    dim1=0, dim2=1, 
    color='k', create_figure=False, figsize=(6, 4),
    grid_num=50, smoothness=1, min_vel_mag=None, cutoff=1.5, return_grid=False,
    linewidth=1, constant_lw=False, density=1, **streamline_kwargs):
    
    if x_layer is None or len(x_layer) == 0:
        emb = basis
    else:
        emb = '%s_%s'%(x_layer, basis)
    v_basis = basis if v_basis is None else v_basis
    if v_layer is None or len(v_layer) == 0:
        v_emb = v_basis
    else:
        v_emb = '%s_%s'%(v_layer, v_basis)
    X = adata.obsm[emb][:, [dim1, dim2]]
    V = adata.obsm[v_emb][:, [dim1, dim2]]

    # set up grids
    #if np.isscalar(grid_num):
    #    grid_num = grid_num * np.ones(2)
    #V_grid, X_grid = smoothen_drift_on_grid(X, V, n_grid=grid_num, smoothness=smoothness)
    #V_grid, X_grid = V_grid.T, X_grid.T
    X_grid, V_grid = velocity_on_grid(X, V, 
        n_grids=grid_num, smoothness=smoothness, cutoff_coeff=cutoff)
    V_grid, X_grid = V_grid.T, X_grid.T

    streamplot_kwargs = {
        "density": density*2,
        "arrowsize": 1,
        "arrowstyle": "fancy",
        "minlength": 0.1,
        "maxlength": 4.0,
        "integration_direction": "both",
        "zorder": 3,
    }

    mass = np.sqrt((V_grid**2).sum(0))
    # velocity filtering
    if min_vel_mag is not None:
        min_vel_mag = np.clip(min_vel_mag, None, np.quantile(mass, 0.4))
        mass[mass<min_vel_mag] = np.nan

    if not constant_lw:
        linewidth *= 2 * mass / mass[~np.isnan(mass)].max()
        linewidth = linewidth.reshape(grid_num, grid_num)
    streamplot_kwargs.update({"linewidth": linewidth})
    streamplot_kwargs.update(streamline_kwargs)

    if np.isscalar(grid_num):
        gnum = grid_num * np.ones(2, dtype=int)
    else:
        gnum = grid_num
    x = X_grid[0].reshape(gnum[0], gnum[1])
    y = X_grid[1].reshape(gnum[0], gnum[1])
    u = V_grid[0].reshape(gnum[0], gnum[1])
    v = V_grid[1].reshape(gnum[0], gnum[1])
    if create_figure: plt.figure(figsize=figsize)
    plt.streamplot(x, y, u, v, color=color, **streamplot_kwargs)
    #plt.set_arrow_alpha(axes_list[i], streamline_alpha)
    #set_stream_line_alpha(s, streamline_alpha)
    if return_grid:
        return X_grid.T, V_grid.T


def multiplot(plot_func, arr, n_row=None, n_col=3, fig=None, subplot_size=(6, 4)):
    if n_col is None and n_row is None: n_col = 3
    n = len(arr[list(arr.keys())[0]]) if type(arr) is dict else len(arr)
    if n_row is None:
        n_row = int(np.ceil(n / n_col))
    elif n_col is None:
        n_col = int(np.ceil(n / n_row))
    else:
        # only the first n plots will be plotted
        n = min(n_row * n_col, n)

    if fig is None: 
        figsize = (subplot_size[0]*n_col, subplot_size[1]*n_row)
        fig=plt.figure(figsize=figsize)
    ax_list = []
    for i in range(n):
        ax_list.append(fig.add_subplot(n_row, n_col, i+1))
        if type(arr) is dict:
            pdict = {key: value[i] for key, value in arr.items()}
            plot_func(**pdict)
        elif isarray(arr[i]):
            plot_func(*arr[i])
        else:
            plot_func(arr[i])
    return ax_list


def plot_jacobian_gene(adata, jkey='jacobian', basis='pca', regulators=None, effectors=None, **kwargs):
    jkey = f'{jkey}_{basis}' if basis is not None else jkey
    J_dict = adata.uns[jkey]
    c_arr = []
    ti_arr = []
    for i, reg in enumerate(J_dict['regulators']):
        if regulators is None or reg in regulators:
            for j, eff in enumerate(J_dict['effectors']):
                if effectors is None or eff in effectors:
                    c_arr.append(J_dict['jacobian_gene'][j, i, :])
                    ti_arr.append(f"{eff} wrt. {reg}")
    multiplot(lambda c, ti: [zscatter(adata, color=c, **kwargs),
                               plt.title(ti)],
                {'c': c_arr, 'ti': ti_arr}, n_col=2, subplot_size=(8, 4))