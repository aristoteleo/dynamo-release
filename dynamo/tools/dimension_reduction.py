from sklearn.decomposition import TruncatedSVD
import warnings
from .psl import *

from .connectivity import umap_conn_indices_dist_embedding


def reduceDimension(adata, n_pca_components=30, n_components=2, n_neighbors=30, reduction_method='umap', cores=1):
    """Compute a low dimension reduction projection of an annodata object first with PCA, followed by non-linear dimension reduction methods

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        an Annodata object 
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

    n_obs = adata.shape[0]

    if 'use_for_dynamo' in adata.var.keys():
        X = adata.X[:, adata.var.use_for_dynamo.values]
    else:
        X = adata.X

    if((not 'X_pca' in adata.obsm.keys()) or 'pca_fit' not in adata.uns.keys()) or reduction_method is "pca":
        transformer = TruncatedSVD(n_components=n_pca_components + 1, random_state=0)
        X_fit = transformer.fit(X)
        X_pca = X_fit.transform(X)[:, 1:]
        adata.obsm['X_pca'] = X_pca
    else:
        X_pca = adata.obsm['X_pca'][:, :n_pca_components]
        adata.obsm['X_pca'] = X_pca

    if reduction_method == "trimap":
        import trimap
        triplemap = trimap.TRIMAP(n_inliers=20,
                                  n_outliers=10,
                                  n_random=10,
                                  distance='euclidean', # cosine
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
            graph, knn_indices, knn_dists, X_dim = umap_conn_indices_dist_embedding(X_pca, n_neighbors) # X_pca
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
