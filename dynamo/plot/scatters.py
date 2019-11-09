import numpy as np
import pandas as pd
from .utilities import despline, minimal_xticks, minimal_yticks


def scatters(adata, gene_names, color, dims=[0, 1], current_layer='spliced', use_raw=False, Vkey='S', Ekey='spliced',
             basis='umap', mode='expression', label_on_embedding=True, cmap=None, gs=None, **kwargs):
    """Scatter plot of cells in phase portrait or in low embedding, colored by gene expression, velocity or cell groups.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        gene_names: `list`
            The gene names whose gene expression will be faceted.
        color: `list` or None (default: None)
            A list of attributes of cells (column names in the adata.obs) will be used to color cells.
        dims: `list` or None (default: [0, 1])
            The column index of the low dimensional embedding for the x/y-axis
        current_layer: `str` (default: X)
            Which layer of expression value will be used.
        use_raw: `bool` (defaul: False)
            `str` (default: X)
                Which layer of expression value will be used.
        Vkey: `str` ('S`)
            The key for the velocity
        Ekey: `str` (`spliced`)
            The key for the gene expression.
        basis`str` (default: `trimap`)
            The reduced dimension embedding of cells to visualize.
        mode: `str` (default: `expression`)
            Which plotting mode to use, either expression, velocity or phase.
        cmap: `plt.cm` or None (default: None)
            The color map function to use.
        gs: `plt.XX`
        **kwargs:
            Additional parameters that will be passed to plt.scatter function

    Returns
    -------
        Nothing but a scatter plot of cells.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patheffects as PathEffects

    color = adata.obs.loc[:, color]
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

        x, y = adata.obsm['X_'+basis][:, dims[0]], adata.obsm['X_'+basis][:, dims[1]]

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

        plt.scatter(x, y, rasterized=True, c=np.reshape(cmap(V_vec), x.shape), **kwarg_plot)
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

        x, y = adata.obsm['X_'+basis][:, dims[0]], adata.obsm['X_'+basis][:, dims[1]]
        if use_raw:
            if Ekey == 'X':
                E_vec = adata.X[:, ix]
            else:
                E_vec = adata.layers[Ekey][:, ix]
        else:
            if Ekey == 'X':
                E_vec = adata.X[:, ix]
            else:
                E_vec = adata.layers['X_' + Ekey][:, ix]

        E_vec = E_vec / np.percentile(E_vec, 99)
        # tmp_colorandum = np.log2(tmp_colorandum+1)
        E_vec = np.clip(E_vec, 0, 1)

        color_labels = color.unique()
        rgb_values = sns.color_palette("Set2", len(color_labels))

        # Map label to RGB
        color_map = pd.DataFrame(zip(color_labels, rgb_values), index=color_labels)

        # ax.scatter(cur_pd.iloc[:, 0], cur_pd.iloc[:, 1], c=color_map.loc[E_vec, 1].values, **scatter_kwargs)
        df = pd.DataFrame(
            {"x": x, "y": y}, #'gene': np.repeat(np.array(genes), n_cells), "expression": E_vec},
            index=range(x.shape[0]))

        ax = plt.scatter(x, y, rasterized=True, c=color_map.loc[:, 1].values, **kwarg_plot)

        if label_on_embedding:
            for i in color_labels:
                color_cnt = np.median(df.iloc[np.where(E_vec == i)[0], :2], 0)
                txt = ax.text(color_cnt[0], color_cnt[1], str(i),
                              fontsize=13)  # , bbox={"facecolor": "w", "alpha": 0.6}
                txt.set_path_effects([
                    PathEffects.Stroke(linewidth=5, foreground="w", alpha=0.1),
                    PathEffects.Normal()])

        plt.axis("off")
        plt.title(f"{gene_names}")

    plt.show()

