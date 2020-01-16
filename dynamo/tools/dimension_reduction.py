from sklearn.decomposition import TruncatedSVD
import warnings
from .psl import *

from .connectivity import umap_conn_indices_dist_embedding


def reduceDimension(adata, n_pca_components=25, n_components=2, n_neighbors=10, reduction_method='trimap', velocity_key='velocity_S', cores=1):
    """Compute a low dimension reduction projection of an annodata object first with PCA, followed by non-linear dimension reduction methods

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        an Annodata object 
    n_pca_components: 'int' (optional, default 50)
        Number of PCA components.  
    n_components: 'int' (optional, default 50)
        The dimension of the space to embed into.
    n_neighbors: 'int' (optional, default 10)
        Number of nearest neighbors when constructing adjacency matrix. 
    reduction_method: 'str' (optional, default trimap)
        Non-linear dimension reduction method to further reduce dimension based on the top n_pca_components PCA components. Currently, PSL 
        (probablistic structure learning, a new dimension reduction by us), tSNE (fitsne instead of traditional tSNE used) or umap are supported.
    velocity_key: `string` (default: `S`)
        Which (suffix of) the velocity key used for visualizing the magnitude of velocity. Can be either in the layers attribute or the
        keys in the obsm attribute. The full key name can be retrieved by `vkey + '_velocity'`.
    cores: `int` (optional, default `1`)
        Number of cores. Used only when the tSNE reduction_method is used.

    Returns
    -------
    Returns an updated `adata` with reduced dimension data for spliced counts, projected future transcript counts 'Y_dim' and adjacency matrix when possible.
    """

    n_obs = adata.shape[0]
    if 'velocity_' not in velocity_key: velocity_key = 'velocity_' + velocity_key

    if 'use_for_dynamo' in adata.var.keys():
        X = adata.X[:, adata.var.use_for_dynamo.values]
        if velocity_key is not None:
            X_t = adata.X[:, adata.var.use_for_dynamo.values] + adata.layers[velocity_key][:, adata.var.use_for_dynamo.values]
    else:
        X = adata.X
        if velocity_key is not None:
            X_t = adata.X + adata.layers[velocity_key]

    if((not 'X_pca' in adata.obsm.keys()) or 'pca_fit' not in adata.uns.keys()) or reduction_method is "pca":
        transformer = TruncatedSVD(n_components=n_pca_components + 1, random_state=0)
        X_fit = transformer.fit(X)
        X_pca = X_fit.transform(X)[:, 1:]
        adata.obsm['X_pca'] = X_pca
        if velocity_key is not None and "_velocity_pca" not in adata.obsm.keys():
            X_t_pca = X_fit.transform(X_t)[:, 1:]
            adata.obsm['_velocity_pca'] = X_t_pca - X_pca
    else:
        X_pca = adata.obsm['X_pca'][:, :n_pca_components]
        if velocity_key is not None and "_velocity_pca" not in adata.obsm.keys():
            X_t_pca = adata.uns['pca_fit'].fit_transform(X_t)
            adata.obsm['_velocity_pca'] = X_t_pca[:, 1:(n_pca_components + 1)] - X_pca
        adata.obsm['X_pca'] = X_pca

    if reduction_method == "trimap":
        import trimap
        triplemap = trimap.TRIMAP(n_inliers=20,
                                  n_outliers=10,
                                  n_random=10,
                                  distance='angular', # cosine
                                  weight_adj=1000.0,
                                  apply_pca=False)
        X_dim = triplemap.fit_transform(X_pca)

        adata.obsm['X_trimap'] = X_dim
        adata.uns['neighbors'] = {'params': {'n_neighbors': n_neighbors, 'method': reduction_method}, 'connectivities': None, \
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
        adata.obsm['X_tSNE'] = X_dim
        adata.uns['neighbors'] = {'params': {'n_neighbors': n_neighbors, 'method': reduction_method}, 'connectivities': None, \
                                  'distances': None, 'indices': None}
    elif reduction_method == 'umap':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph, knn_indices, knn_dists, X_dim = umap_conn_indices_dist_embedding(X_pca) # X_pca
        adata.obsm['X_umap'] = X_dim
        adata.uns['neighbors'] = {'params': {'n_neighbors': n_neighbors, 'method': reduction_method}, 'connectivities': graph, \
                                  'distances': knn_dists, 'indices': knn_indices}
    elif reduction_method is 'psl':
        adj_mat, X_dim = psl_py(X_pca, d=n_components, K=n_neighbors) # this need to be updated
        adata.obsm['X_psl'] = X_dim
        adata.uns['PSL_adj_mat'] = adj_mat

    else:
        raise Exception('reduction_method {} is not supported.'.format(reduction_method))

    return adata
