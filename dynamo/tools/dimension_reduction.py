from sklearn.decomposition import PCA, TruncatedSVD
from scipy.sparse import issparse
import warnings
from .psl import *

from .connectivity import umap_conn_indices_dist_embedding


def reduceDimension(adata, layer='X', n_pca_components=30, n_components=2, n_neighbors=30, reduction_method='umap', cores=1):
    """Compute a low dimension reduction projection of an annodata object first with PCA, followed by non-linear dimension reduction methods

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    layer: str (default: X)
            The layer where the dimension reduction will be performed.
    n_pca_components: 'int' (optional, default 30)
        Number of PCA components.  
    n_components: 'int' (optional, default 2)
        The dimension of the space to embed into.
    n_neighbors: 'int' (optional, default 30)
        Number of nearest neighbors when constructing adjacency matrix. 
    reduction_method: 'str' (optional, default umap)
        Non-linear dimension reduction method to further reduce dimension based on the top n_pca_components PCA components. Currently, PSL 
        (probablistic structure learning, a new dimension reduction by us), tSNE (fitsne instead of traditional tSNE used) or umap are supported.
    cores: `int` (optional, default `1`)
        Number of cores. Used only when the tSNE reduction_method is used.

    Returns
    -------
    Returns an updated `adata` with reduced dimension data for spliced counts, projected future transcript counts 'Y_dim'.
    """

    layer = layer if layer.startswith('X_') else 'X_' + layer
    if layer not in adata.layers.keys():
        raise Exception('The layer X_{} you provided is not existed in adata.'.format(layer))
    pca_key = 'X_pca' if layer == 'X' else layer + '_pca'
    embedding_key = 'X_' + reduction_method if layer == 'X' else layer + '_' + reduction_method
    neighbor_key = 'neighbors' if layer == 'X' else layer + '_neighbors'

    if 'use_for_dynamo' in adata.var.keys():
        X = adata.X[:, adata.var.use_for_dynamo.values] if layer == 'X' else \
            np.log(adata[:, adata.var.use_for_dynamo.values].layers[layer] + 1)
    else:
        X = adata.X if layer == 'X' else np.log(adata.layers[layer] + 1)

    if layer == 'X':
        if((pca_key not in adata.obsm.keys()) or 'pca_fit' not in adata.uns.keys()) or reduction_method is "pca":
            if adata.n_obs < 100000:
                fit = PCA(n_components=n_pca_components, svd_solver='arpack', random_state=0)
                X_pca = fit.fit_transform(X.toarray()) if issparse(X) else fit.fit_transform(X)
                adata.obsm[pca_key] = X_pca
            else:
                fit = TruncatedSVD(n_components=n_pca_components + 1, random_state=0)  # unscaled PCA
                X_pca = fit.fit_transform(X)[:, 1:]  # first columns is related to the total UMI (or library size)
                adata.obsm[pca_key] = X_pca
        else:
            X_pca = adata.obsm[pca_key][:, :n_pca_components]
            adata.obsm[pca_key] = X_pca
    else:
        if(pca_key not in adata.obsm.keys()) or reduction_method is "pca":
            if adata.n_obs < 100000:
                fit = PCA(n_components=n_pca_components, svd_solver='arpack', random_state=0)
                X_pca = fit.fit_transform(X.toarray()) if issparse(X) else fit.fit_transform(X)
                adata.obsm[pca_key] = X_pca
            else:
                fit = TruncatedSVD(n_components=n_pca_components + 1, random_state=0)  # unscaled PCA
                X_pca = fit.fit_transform(X)[:, 1:]  # first columns is related to the total UMI (or library size)
                adata.obsm[pca_key] = X_pca
        else:
            X_pca = adata.obsm[pca_key][:, :n_pca_components]
            adata.obsm[pca_key] = X_pca

    if reduction_method == "trimap":
        import trimap
        triplemap = trimap.TRIMAP(n_inliers=20,
                                  n_outliers=10,
                                  n_random=10,
                                  distance='euclidean', # cosine
                                  weight_adj=1000.0,
                                  apply_pca=False)
        X_dim = triplemap.fit_transform(X_pca)

        adata.obsm[embedding_key] = X_dim
        adata.uns[neighbor_key] = {'params': {'n_neighbors': n_neighbors, 'method': reduction_method}, 'connectivities': None, \
                                  'distances': None, 'indices': None}
    elif reduction_method == 'diffusion_map':
        pass
    elif reduction_method == 'tSNE':
        try:
            from fitsne import FItSNE
        except ImportError:
            print('Please first install fitsne to perform accelerated tSNE method. Install instruction is provided here: https://pypi.org/project/fitsne/')

        X_dim=FItSNE(X_pca, nthreads=cores) # use FitSNE

        # bh_tsne = TSNE(n_components = n_components)
        # X_dim = bh_tsne.fit_transform(X_pca)
        adata.obsm[embedding_key] = X_dim
        adata.uns[neighbor_key] = {'params': {'n_neighbors': n_neighbors, 'method': reduction_method}, 'connectivities': None, \
                                  'distances': None, 'indices': None}
    elif reduction_method == 'umap':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph, knn_indices, knn_dists, X_dim = umap_conn_indices_dist_embedding(X_pca, n_neighbors) # X_pca
        adata.obsm[embedding_key] = X_dim
        adata.uns[neighbor_key] = {'params': {'n_neighbors': n_neighbors, 'method': reduction_method}, 'connectivities': graph, \
                                  'distances': knn_dists, 'indices': knn_indices}
    elif reduction_method is 'psl':
        adj_mat, X_dim = psl_py(X_pca, d=n_components, K=n_neighbors) # this need to be updated
        adata.obsm[embedding_key] = X_dim
        adata.uns[neighbor_key] = adj_mat

    else:
        raise Exception('reduction_method {} is not supported.'.format(reduction_method))

    return adata
