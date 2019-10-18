import numpy as np
from .utilities import despline, minimal_xticks, minimal_yticks


def scatters(adata, gene_names, color, dims=[0, 1], current_layer='spliced', use_raw=False, Vkey='S', Ekey='spliced',
             basis='umap', mode='all', cmap=None, gs=None, **kwargs):
    """

    Parameters
    ----------
    adata
    gene_names `list`
    basis
    mode: `str` (default: all)
        Support mode includes: phase, expression, velocity, all

    Returns
    -------

    """
    import matplotlib.pyplot as plt
    color = adata.obs.loc[color]
    if mode is 'expression' and cmap is None:
        cmap = plt.cm.Greens # qualitative
    elif mode is 'velocity' and cmap is None:
        cmap = plt.cm.RdBu_r # diverging
    elif mode is 'phase':
        from matplotlib import cm
        viridis = cm.get_cmap('viridis', len(np.unique(color)))
        cmap = viridis
    # plt.figure(None, (17, 2.8), dpi=80)
    # gs = plt.GridSpec(1, 6)

    gene_names = list(set(gene_names).intersection(adata.var.index))
    ix = np.where(adata.var.index.isin(gene_names))[0][0]

    # do the RNA/protein phase diagram on the same plot
    if mode is 'phase':
        gamma, q = adata.var.velocity_parameter_gamma[ix], adata.var.velocity_parameter_q[ix]
        if use_raw:
            x, y = adata.layers['spliced'][:, ix], adata.layers['unspliced'][: ix] if current_layer is 'spliced' else \
                adata.layers['protein'][:, ix], adata.layers['spliced'][: ix]
        else:
            x, y = adata.layers['X_spliced'][:, ix], adata.layers['X_unspliced'][
                                                     : ix] if current_layer is 'spliced' else \
                adata.layers['X_protein'][:, ix], adata.layers['X_spliced'][: ix]

        plt.scatter(x, y, s=5, alpha=0.4, rasterized=True) # , c=vlm.colorandum
        plt.title(gene_names)
        xnew = np.linspace(0, x.max())
        plt.plot(xnew, gamma * xnew + q, c="k")
        plt.ylim(0, np.max(y) * 1.02)
        plt.xlim(0, np.max(x) * 1.02)
        minimal_yticks(0, y * 1.02)
        minimal_xticks(0, x * 1.02)
        despline()
    elif mode is 'velocity':
        kwarg_plot = {"alpha": 0.5, "s": 8, "edgecolor": "0.8", "lw": 0.15}
        kwarg_plot.update(kwargs)

        x, y = adata.obsm['X_'+basis][dims[0]], adata.obsm['X_'+basis][dims[1]]

        if gs is None:
            fig = plt.figure(figsize=(10, 10))
            plt.subplot(111)
        else:
            plt.subplot(gs)

        if Vkey is 'U':
            V_vec = adata.layer['velocity_U'][:, ix]
        elif Vkey is 'S':
            V_vec = adata.layer['velocity_S'][:, ix]
        elif Vkey is 'P':
            V_vec = adata.layer['velocity_P'][:, ix]

        if (np.abs(V_vec) > 0.00005).sum() < 10:  # If S vs U scatterplot it is flat
            print("S vs U scatterplot it is flat")
            return
        limit = np.max(np.abs(np.percentile(V_vec, [1, 99])))  # upper and lowe limit / saturation
        V_vec = V_vec + limit  # that is: tmp_colorandum - (-limit)
        V_vec = V_vec / (2 * limit)  # that is: tmp_colorandum / (limit - (-limit))
        V_vec = np.clip(V_vec, 0, 1)

        plt.scatter(x, y, rasterized=True, c=cmap(V_vec), **kwarg_plot)
        plt.axis("off")
        plt.title(f"{gene_names}")
    elif mode is 'expression':
        kwarg_plot = {"alpha": 0.5, "s": 8, "edgecolor": "0.8", "lw": 0.15}
        kwarg_plot.update(kwargs)
        if gs is None:
            fig = plt.figure(figsize=(10, 10))
            plt.subplot(111)
        else:
            plt.subplot(gs)

        x, y = adata.obsm['X_'+basis][dims[0]], adata.obsm['X_'+basis][dims[1]]
        if use_raw:
            E_vec = adata.layers[Ekey][:, ix]
        else:
            E_vec = adata.layers['X_' + Ekey][:, ix]

        E_vec = E_vec / np.percentile(E_vec, 99)
        # tmp_colorandum = np.log2(tmp_colorandum+1)
        E_vec = np.clip(E_vec, 0, 1)

        plt.scatter(x, y, rasterized=True, c=cmap(E_vec), **kwarg_plot)
        plt.axis("off")
        plt.title(f"{gene_names}")


