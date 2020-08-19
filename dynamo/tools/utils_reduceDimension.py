from scipy.sparse import issparse
import warnings
from ..preprocessing.utils import pca
from .utils import update_dict, log1p_
from .connectivity import (
    umap_conn_indices_dist_embedding, knn_to_adj
)
from .psl_py import *

# ---------------------------------------------------------------------------------------------------
def prepare_dim_reduction(adata,
                          genes=None,
                          layer=None,
                          basis='pca',
                          dims=None,
                          n_pca_components=30,
                          n_components=2,):
    if genes is not None:
        genes = adata.var_name.intersection(genes).to_list()
        if len(genes) == 0:
            raise ValueError(f'no genes from your genes list appear in your adata object.')
    if layer is not None:
        if layer not in adata.layers.keys():
            raise ValueError(f'the layer {layer} you provided is not included in the adata object!')

    prefix = 'X_' if layer is None else layer + '_'
    has_basis = False

    if basis is not None:
        if basis.split(prefix)[-1] not in ['pca', 'umap', 'trimap', 'tsne', 'diffmap']:
            raise ValueError(f"basis (or the suffix of basis) can only be one of "
                             f"['pca', 'umap', 'trimap', 'tsne', 'diffmap'].")
        if basis.startswith(prefix):
            basis = basis
        else:
            basis = prefix + basis

    if basis is None:
        if layer is None:
            if genes is not None:
                X_data = adata[:, genes].X
            else:
                X_data = adata.X if 'use_for_dynamics' not in adata.var.keys() \
                    else adata[:, adata.var.use_for_dynamics].X
        else:
            if genes is not None:
                X_data = adata[:, genes].layers[layer]
            else:
                X_data = adata.layers[layer] if 'use_for_dynamics' not in adata.var.keys() \
                        else adata[:, adata.var.use_for_dynamics].layers[layer]

            X_data = log1p_(adata, X_data)
    else:
        pca_key = "X_pca" if layer is None else layer + "_pca"
        n_pca_components = max(max(dims), n_pca_components) if dims is not None else n_pca_components

        if basis not in adata.obsm.keys():
            if genes is not None or pca_key not in adata.obsm.keys() or \
                    adata.obsm[pca_key].shape[1] < n_pca_components:
                if layer is None:
                    if genes is not None:
                        CM = adata[:, genes].X
                    else:
                        CM = adata.X if 'use_for_dynamics' not in adata.var.keys() \
                            else adata[:, adata.var.use_for_dynamics].X
                else:
                    if genes is not None:
                        CM = adata[:, genes].layers[layer]
                    else:
                        CM = adata.layers[layer] if 'use_for_dynamics' not in adata.var.keys() \
                            else adata[:, adata.var.use_for_dynamics].layers[layer]

                    CM = log1p_(adata, CM)

                cm_genesums = CM.sum(axis=0)
                valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
                valid_ind = np.array(valid_ind).flatten()
                CM = CM[:, valid_ind]
                adata, fit, _ = pca(adata, CM, n_pca_components=n_pca_components, pca_key=pca_key)
                adata.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_[1:]
        else:
            has_basis = True

        if pca_key in adata.obsm.keys():
            X_data = adata.obsm[pca_key]
        else:
            if genes is not None:
                CM = adata[:, genes].layers[layer]
            else:
                CM = adata.layers[layer] if 'use_for_dynamics' not in adata.var.keys() \
                    else adata[:, adata.var.use_for_dynamics].layers[layer]

            CM = log1p_(adata, CM)

            cm_genesums = CM.sum(axis=0)
            valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
            valid_ind = np.array(valid_ind).flatten()
            CM = CM[:, valid_ind]
            adata, fit, _ = pca(adata, CM, n_pca_components=n_pca_components, pca_key=pca_key)
            adata.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_[1:]

            X_data = adata.obsm[pca_key]

    if dims is not None: X_data = X_data[:, dims]

    return X_data, n_components, has_basis, basis

def run_reduce_dim(adata, X_data, n_components, n_pca_components, reduction_method, embedding_key, n_neighbors,
                   neighbor_key, cores, kwargs):
    if reduction_method == "trimap":
        import trimap

        triplemap = trimap.TRIMAP(
            n_inliers=20,
            n_outliers=10,
            n_random=10,
            distance="euclidean",  # cosine
            weight_adj=1000.0,
            apply_pca=False,
        )
        X_dim = triplemap.fit_transform(X_data)

        adata.obsm[embedding_key] = X_dim
        adata.uns[neighbor_key] = {
            "params": {"n_neighbors": n_neighbors, "method": reduction_method},
            # "connectivities": "connectivities",
            # "distances": "distances",
            # "indices": "indices",
        }
    elif reduction_method == "diffusion_map":
        pass
    elif reduction_method.lower() == "tsne":
        try:
            from fitsne import FItSNE
        except ImportError:
            print(
                "Please first install fitsne to perform accelerated tSNE method. Install instruction is "
                "provided here: https://pypi.org/project/fitsne/"
            )

        X_dim = FItSNE(X_data, nthreads=cores)  # use FitSNE

        # bh_tsne = TSNE(n_components = n_components)
        # X_dim = bh_tsne.fit_transform(X)
        adata.obsm[embedding_key] = X_dim
        adata.uns[neighbor_key] = {
            "params": {"n_neighbors": n_neighbors, "method": reduction_method},
            # "connectivities": "connectivities",
            # "distances": "distances",
            # "indices": "indices",
        }
    elif reduction_method == "umap":
        _umap_kwargs = {
            "n_components": 2,
            "metric": "euclidean",
            "min_dist": 0.5,
            "spread": 1.0,
            "n_epochs": 0,
            "alpha": 1.0,
            "gamma": 1.0,
            "negative_sample_rate": 5,
            "init_pos": "spectral",
            "random_state": 0,
            "verbose": False,
        }
        umap_kwargs = update_dict(_umap_kwargs, kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (
                mapper,
                graph,
                knn_indices,
                knn_dists,
                X_dim,
            ) = umap_conn_indices_dist_embedding(
                X_data, n_neighbors, **umap_kwargs
            )  # X

        adata.obsm[embedding_key] = X_dim
        knn_dists = knn_to_adj(knn_indices, knn_dists)
        adata.uns[neighbor_key] = {
            "params": {"n_neighbors": n_neighbors, "method": reduction_method},
            # "connectivities": "connectivities",
            # "distances": "distances",
            "indices": knn_indices,
        }
        
        layer = neighbor_key.split('_')[0] if neighbor_key.__contains__('_') else None 
        conn_key = "connectivities" if layer is None else layer + "_connectivities"
        dist_key = "distances" if layer is None else layer + "_distances"

        adata.obsp[conn_key], adata.obsp[dist_key] = graph, knn_dists

        adata.uns["umap_fit"] = {"fit": mapper, "n_pca_components": n_pca_components}
    elif reduction_method == "psl":
        adj_mat, X_dim = psl(
            X_data, d=n_components, K=n_neighbors
        )  # this need to be updated
        adata.obsm[embedding_key] = X_dim
        adata.uns[neighbor_key] = adj_mat

    else:
        raise Exception(
            "reduction_method {} is not supported.".format(reduction_method)
        )

    return adata
