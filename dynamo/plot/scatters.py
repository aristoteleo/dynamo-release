import numpy as np
import pandas as pd
from ..preprocessing.preprocess import topTable


def scatter(adata, basis, color, size=6):  # we can also use the color to visualize the velocity
    from plotnine import ggplot, geom_point
    X = adata.obsm['X_' + basis]
    df = pd.DataFrame(X)
    df['color_group'] = adata.obs[:, color]
    ggplot(X[0], X[1], color=df['color_group']) + geom_point(size=size)


def mean_cv(adata, group):
    from plotnine import ggplot, geom_point
    mn = adata.obs['mean']
    cv = adata.obs['cv']
    df = pd.DataFrame({'mean': mn, 'cv': CV})

    ggplot(mean, cv, data=df) + geom_point()


def variance_explained(adata):
    from plotnine import ggplot, geom_point
    var_info = adata.uns['pca']

    ggplot(variance, compoent, data=var_info) + geom_point()


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

# def velocity(adata, type) # type can be either one of the three, cellwise, velocity on grid, streamline plot.
#	"""
#
#	"""
#
