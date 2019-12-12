import numpy as np
from scipy.sparse import issparse
from .scatters import scatters
from ..tools.velocity import sol_u, sol_s

def plot_fitting(adata, vkey, tkey, experiment_type='deg', unit='hours', log=True, group=None, ncols=None, figsize=None, dpi=None, show=True):
    """

    Parameters
    ----------
    adata
    vkey
    tkey
    experiment_type
    unit
    log
    group
    ncols
    figsize
    dpi
    show

    Returns
    -------

    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    groups = [''] if group is None else np.unique(adata.obs[group])

    T = adata.obs[tkey]
    gene_idx = [np.where(adata.var.index.values == gene)[0][0] for gene in vkey]
    prefix = 'kinetic_parameter_'

    ncols = len(gene_idx) if ncols is None else min(len(gene_idx), ncols)
    nrows = int(np.ceil(len(gene_idx) * 2 / ncols))
    figsize = [7, 5] if figsize is None else figsize
    gs = plt.GridSpec(nrows, ncols, plt.figure(None, (figsize[0] * ncols, figsize[1] * nrows), dpi=dpi))

    for idx in gene_idx:
        gene_name = adata.var_names[idx]
        if experiment_type is 'deg':
            alpha, gamma, u0, l0 = adata.var.loc[gene_name, [prefix + 'alpha', prefix + 'gamma', prefix + 'uu0', prefix + 'ul0']]

        new, old = adata[:, gene_name].layers['new'], adata[:, gene_name].layers['total'] - adata[:, gene_name].layers['new']
        u, l = (old.toarray().squeeze(), new.toarray().squeeze()) if issparse(old) else (old.squeeze(), new.squeeze())
        if u0 is None:
            u0 = np.mean(u[T == np.min(T)])
        # [adata.obs[group] == cur_grp, gene_idx]

        if log:
            u, l = np.log(u + 1), np.log(l + 1)

        Tsort = np.argsort(T.unique())
        t = np.linspace(Tsort[0], Tsort[-1], 50)

        # beta, gamma, alpha
        ax1 = plt.subplot(gs[idx * 2])
        ax1=sns.violinplot(x=T.values, y=u.T, ax=ax1) #ax1.plot(T, u.T, linestyle='None', marker='o', markersize=10)
        ax1.plot(t, alpha/gamma + (u0 - alpha/gamma) * np.exp(-gamma*t), '--')
        ax1.set_xlabel('time (' + unit + ')')
        ax1.set_title('unlabeled (' + gene_name + ')')

        ax2 = plt.subplot(gs[idx * 2 + 1])
        ax2=sns.violinplot(x=T.values, y=l.T, ax=ax2) # ax2.plot(T, l.T, linestyle='None', marker='o', markersize=10)
        ax2.plot(t, l0 * np.exp(-gamma*t), '--')
        ax2.set_xlabel('time (' + unit + ')')
        ax2.set_title('labeled (' + gene_name + ')')

    if show: plt.show()


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



