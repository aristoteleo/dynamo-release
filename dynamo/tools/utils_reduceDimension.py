import warnings

import numpy as np

from ..configuration import DKM
from ..preprocessing.utils import pca_monocle
from .connectivity import (
    _gen_neighbor_keys,
    knn_to_adj,
    umap_conn_indices_dist_embedding,
)
from .psl_py import psl
from .utils import log1p_, update_dict


# ---------------------------------------------------------------------------------------------------
def prepare_dim_reduction(
    adata,
    genes=None,
    layer=None,
    basis="pca",
    dims=None,
    n_pca_components=30,
    n_components=2,
):
    if genes is not None:
        genes = adata.var_name.intersection(genes).to_list()
        if len(genes) == 0:
            raise ValueError("no genes from your genes list appear in your adata object.")
    if layer is not None:
        if not DKM.check_if_layer_exist(adata, layer):
            raise ValueError(f"the layer {layer} you provided is not included in the adata object!")

    prefix = "X_" if layer is None else layer + "_"

    if basis is not None:
        if basis.split(prefix)[-1] not in [
            "pca",
            "umap",
            "trimap",
            "tsne",
            "diffmap",
        ]:
            raise ValueError(
                "basis (or the suffix of basis) can only be one of ['pca', 'umap', 'trimap', 'tsne', 'diffmap']."
            )
        if basis.startswith(prefix):
            basis = basis
        else:
            basis = prefix + basis

    if basis is None:
        if layer is None:
            if genes is not None:
                X_data = adata[:, genes].X
            else:
                X_data = adata.X if "use_for_pca" not in adata.var.keys() else adata[:, adata.var.use_for_pca].X
        else:
            if genes is not None:
                X_data = adata[:, genes].layers[layer]
            else:
                X_data = (
                    adata.layers[layer]
                    if "use_for_pca" not in adata.var.keys()
                    else adata[:, adata.var.use_for_pca].layers[layer]
                )

            X_data = log1p_(adata, X_data)
    else:
        pca_key = "X_pca" if layer is None else layer + "_pca"
        n_pca_components = max(max(dims), n_pca_components) if dims is not None else n_pca_components

        if basis not in adata.obsm.keys():
            if genes is not None or pca_key not in adata.obsm.keys() or adata.obsm[pca_key].shape[1] < n_pca_components:
                if layer is None:
                    if genes is not None:
                        CM = adata[:, genes].X
                    else:
                        CM = adata.X if "use_for_pca" not in adata.var.keys() else adata[:, adata.var.use_for_pca].X
                else:
                    if genes is not None:
                        CM = adata[:, genes].layers[layer]
                    else:
                        CM = (
                            adata.layers[layer]
                            if "use_for_pca" not in adata.var.keys()
                            else adata[:, adata.var.use_for_pca].layers[layer]
                        )

                    CM = log1p_(adata, CM)

                cm_genesums = CM.sum(axis=0)
                valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
                valid_ind = np.array(valid_ind).flatten()
                CM = CM[:, valid_ind]
                adata, fit, _ = pca_monocle(
                    adata,
                    CM,
                    n_pca_components=n_pca_components,
                    pca_key=pca_key,
                    return_all=True,
                )
                adata.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_[1:]

                # valid genes used for dimension reduction calculation
                adata.uns["pca_valid_ind"] = valid_ind

        if pca_key in adata.obsm.keys():
            X_data = adata.obsm[pca_key]
        else:
            if genes is not None:
                CM = adata[:, genes].layers[layer]
            else:
                CM = (
                    adata.layers[layer]
                    if "use_for_pca" not in adata.var.keys()
                    else adata[:, adata.var.use_for_pca].layers[layer]
                )

            CM = log1p_(adata, CM)

            cm_genesums = CM.sum(axis=0)
            valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
            valid_ind = np.array(valid_ind).flatten()
            CM = CM[:, valid_ind]
            adata, fit, _ = pca_monocle(adata, CM, n_pca_components=n_pca_components, pca_key=pca_key, return_all=True)
            adata.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_[1:]

            # valid genes used for dimension reduction calculation
            adata.uns["pca_valid_ind"] = valid_ind
            X_data = adata.obsm[pca_key]

    if dims is not None:
        X_data = X_data[:, dims]

    return X_data, n_components, basis


def run_reduce_dim(
    adata,
    X_data,
    n_components,
    n_pca_components,
    reduction_method,
    embedding_key,
    n_neighbors,
    neighbor_key,
    cores,
    **kwargs,
):
    if reduction_method == "trimap":
        try:
            import trimap
        except ImportError as exception:
            raise ImportError(
                "Please 1) check if trimap is installed in your environment. 2) if you can import trimap successfully in your python console."
            )

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
    elif reduction_method.lower() == "diffusion_map":
        # support Yan's diffusion map here
        pass
    elif reduction_method.lower() == "tsne":
        try:
            from fitsne import FItSNE
        except ImportError:
            raise ImportError(
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
            "n_components": n_components,
            "metric": "euclidean",
            "min_dist": 0.5,
            "spread": 1.0,
            "max_iter": None,
            "alpha": 1.0,
            "gamma": 1.0,
            "negative_sample_rate": 5,
            "init_pos": "spectral",
            "random_state": 0,
            "densmap": False,
            "dens_lambda": 2.0,
            "dens_frac": 0.3,
            "dens_var_shift": 0.1,
            "output_dens": False,
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
            ) = umap_conn_indices_dist_embedding(X_data, n_neighbors, **umap_kwargs)

        adata.obsm[embedding_key] = X_dim
        knn_dists = knn_to_adj(knn_indices, knn_dists)
        adata.uns[neighbor_key] = {
            "params": {"n_neighbors": n_neighbors, "method": reduction_method},
            # "connectivities": "connectivities",
            # "distances": "distances",
            "indices": knn_indices,
        }

        layer = neighbor_key.split("_")[0] if neighbor_key.__contains__("_") else None
        neighbor_result_prefix = "" if layer is None else layer
        conn_key, dist_key, neighbor_key = _gen_neighbor_keys(neighbor_result_prefix)

        adata.uns["umap_fit"] = {
            "fit": mapper,
            "n_pca_components": n_pca_components,
        }
    elif reduction_method == "psl":
        adj_mat, X_dim = psl(X_data, d=n_components, K=n_neighbors)  # this need to be updated
        adata.obsm[embedding_key] = X_dim
        adata.uns[neighbor_key] = adj_mat

    else:
        raise Exception("reduction_method {} is not supported.".format(reduction_method))

    return adata
