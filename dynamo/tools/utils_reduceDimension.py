import warnings
from typing import List, Optional, Tuple

import numpy as np

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from anndata import AnnData

from ..configuration import DKM
from ..dynamo_logger import main_info_insert_adata_obsm
from ..preprocessing.pca import pca
from .connectivity import (
    generate_neighbor_keys,
    knn_to_adj,
    umap_conn_indices_dist_embedding,
)
from .psl import psl
from .utils import log1p_, update_dict


# ---------------------------------------------------------------------------------------------------
def prepare_dim_reduction(
    adata: AnnData,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    basis: str = "pca",
    dims: Optional[List[int]] = None,
    n_pca_components: int = 30,
    n_components: int = 2,
) -> Tuple[np.ndarray, int, str]:
    """Prepare the data for dimension reduction.

    Args:
        adata: An AnnData object.
        genes: The list of genes that will be used to subset the data for dimension reduction and clustering. If `None`,
            all genes will be used. Defaults to None.
        layer: The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is
            used. Defaults to None.
        basis: The space that will be used for clustering. Defaults to "pca".
        dims: The list of dimensions that will be selected for clustering. If `None`, all dimensions will be used.
            Defaults to None.
        n_pca_components: Number of input PCs (principle components) that will be used for further non-linear dimension
            reduction. If n_pca_components is larger than the existing #PC in adata.obsm['X_pca'] or input layer's
            corresponding pca space (layer_pca), pca will be rerun with n_pca_components PCs requested. Defaults to 30.
        n_components: the dimension of the space to embed into. Defaults to 2.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        A tuple (X_data, n_components, basis) where `X_data` is the extracted data from adata to perform dimension
        reduction, `n_components` is the target dimension of the space to embed into, `basis` is the space that would be
        used for clustering.
    """

    if genes is not None:
        genes = adata.var_names.intersection(genes).to_list()
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
                adata, fit, _ = pca(
                    adata,
                    CM,
                    n_pca_components=n_pca_components,
                    pca_key=pca_key,
                    return_all=True,
                )

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
            adata, fit, _ = pca(adata, CM, n_pca_components=n_pca_components, pca_key=pca_key, return_all=True)

            # valid genes used for dimension reduction calculation
            adata.uns["pca_valid_ind"] = valid_ind
            X_data = adata.obsm[pca_key]

    if dims is not None:
        X_data = X_data[:, dims]

    return X_data, n_components, basis


def run_reduce_dim(
    adata: AnnData,
    X_data: np.ndarray,
    n_components: int,
    n_pca_components: int,
    reduction_method: Literal["trimap", "diffusion_map", "tsne", "umap", "psl"],
    embedding_key: str,
    n_neighbors: int,
    neighbor_key: str,
    cores: int,
    **kwargs,
) -> AnnData:
    """Perform dimension reduction.

    Args:
        adata: An AnnData object.
        X_data: The user supplied data that will be used for dimension reduction directly.
        n_components: The dimension of the space to embed into.
        n_pca_components: Number of input PCs (principal components) that will be used for further non-linear dimension
            reduction.
        reduction_method: Non-linear dimension reduction method to further reduce dimension based on the top
            n_pca_components PCA components.
        embedding_key: The str in .obsm that will be used as the key to save the reduced embedding space.
        n_neighbors: The number of nearest neighbors when constructing adjacency matrix.
        neighbor_key: The str in .uns that will be used as the key to save the nearest neighbor graph.
        cores: The number of threads used for calculation.
        kwargs: Other kwargs passed to umap calculation (see `umap_conn_indices_dist_embedding`).

    Raises:
        ImportError: `trimap` cannot be imported.
        ImportError: `FItSNE` cannot be imported.
        Exception: Invalid `reduction_method`.

    Returns:
        The updated AnnData object with reduced dimension space and related parameters.
    """

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

        main_info_insert_adata_obsm(embedding_key, log_level=20)
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
        main_info_insert_adata_obsm(embedding_key, log_level=20)
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

        main_info_insert_adata_obsm(embedding_key, log_level=20)
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
        conn_key, dist_key, neighbor_key = generate_neighbor_keys(neighbor_result_prefix)

        adata.uns["umap_fit"] = {
            "X_data": X_data,
            "umap_kwargs": umap_kwargs,
            "n_pca_components": n_pca_components,
        }
    elif reduction_method == "psl":
        adj_mat, X_dim = psl(X_data, d=n_components, K=n_neighbors)  # this need to be updated
        main_info_insert_adata_obsm(embedding_key, log_level=20)
        adata.obsm[embedding_key] = X_dim
        adata.uns[neighbor_key] = adj_mat

    else:
        raise Exception("reduction_method {} is not supported.".format(reduction_method))

    return adata
