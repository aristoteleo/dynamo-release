import pysal

def spatial_test(adata, type = 'expression'):
    """Identify genes with strong spatial autocorrelation with Moran's I test. This can be used to identify genes that are
    potentially related to critical dynamic process. Moran's I test is first introduced in single cell genomics analysis
    in (Cao, et al, 2019).

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    type:  `string` (default: expression)
        Which type of data would you like to perform Moran's I test.

    Returns
    -------
    Returns an updated `~anndata.AnnData` with a new property `'Moran_' + type` in the uns slot.
    """

    if type == 'expression':
        X = adata.X
    elif type == 'velocity_u':
        X = adata.layers['velocity_u']
    elif type == 'velocity_s':
        X = adata.layers['velocity_s']

    cell_num, gene_num = X.shape

    adj_mat = adata.obsm['adj_mat']

    # convert a sparse adjacency matrix to a dictionary
    adj_dict = {i: np.nonzero(row.toarray().squeeze())[0].tolist() for i,row in enumerate(adj_mat)}

    W = pysal.lib.weights.W(adj_dict)

    Moran_I, p_value = [], []
    for cur_g in range(gene_num): # make this parallel?
        mbi = pysal.explore.esda.moran.Moran(X[:, cur_g], W, two_tailed=False)
        Moran_I.append(mbi.I)
        p_value.append(mbi.p_norm)

    Moran_res = pd.DataFrame({"Moran_I": Moran_I, "p_value": p_value})

    adata.uns['Moran_' + type] = Moran_res

    return adata
