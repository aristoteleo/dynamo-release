import numpy as np
from .scatters import scatters


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
    import matplotlib.pyplot as plt

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


def dynamics(adata, gene_names, color, dims=[0, 1], current_layer='spliced', use_raw=False, Vkey='S', Ekey='spliced',
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



