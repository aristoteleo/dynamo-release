from typing import List, Optional

import numpy as np
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from ..tools.connectivity import generate_neighbor_keys, check_and_recompute_neighbors, k_nearest_neighbors
from ..tools.utils import log1p_
from .utils import VecFldDict, vecfld_from_adata, vector_field_function


def diffusionMatrix(
    adata: AnnData,
    X_data: Optional[np.ndarray] = None,
    V_data: Optional[np.ndarray] = None,
    genes: Optional[List] = None,
    layer: Optional[str] = None,
    basis: str = "umap",
    dims: Optional[List] = None,
    n: int = 30,
    VecFld: Optional[VecFldDict] = None,
    residual: str = "vector_field",
) -> AnnData:
    """Calculate the diffusion matrix from the estimated velocity vector and the reconstructed vector field.

    Args:
        adata: an Annodata object.
        X_data: The user supplied expression (embedding) data that will be used for calculating diffusion matrix directly.
        V_data: The user supplied velocity data that will be used for calculating diffusion matrix directly.
        genes: The list of genes that will be used to subset the data. If `None`, all genes will be used.
        layer: Which layer of the data will be used for diffusion matrix calculation.
        basis: Which basis of the data will be used for diffusion matrix calculation.
        dims: The list of dimensions that will be selected for diffusion matrix calculation. If `None`, all dimensions will be used.
        n: Number of nearest neighbors when the nearest neighbor graph is not included.
        VecFld: The reconstructed vector field function.
        residual: Method to calculate residual velocity vectors for diffusion matrix calculation. If `average`, all velocity
            of the nearest neighbor cells will be minused by its average velocity; if `vector_field`, all velocity will be
            minused by the predicted velocity from the reconstructed deterministic velocity vector field.

    Returns:
        adata: `AnnData` object that is updated with the `diffusion_matrix` key in the `uns` attribute which is a list of
            the diffusion matrix for each cell. A column `diffusion` corresponds to the square root of the sum of all
            elements for each cell's diffusion matrix will also be added.
    """

    if X_data is None or V_data is not None:
        if genes is not None:
            genes = adata.var_name.intersection(genes).to_list()
            if len(genes) == 0:
                raise ValueError(f"no genes from your genes list appear in your adata object.")
        if layer is not None:
            if layer not in adata.layers.keys():
                raise ValueError(f"the layer {layer} you provided is not included in the adata object!")

            if basis is None:
                vkey = "velocity_" + layer[0].upper()
                if vkey not in adata.obsm.keys():
                    raise ValueError(
                        f"the data corresponds to the velocity key {vkey} is not included in the adata object!"
                    )

        if VecFld is None:
            VecFld, func = vecfld_from_adata(adata, basis)
        else:
            func = lambda x: vector_field_function(x, VecFld)

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
                    f"basis (or the suffix of basis) can only be one of "
                    f"['pca', 'umap', 'trimap', 'tsne', 'diffmap']."
                )
            if basis.startswith(prefix):
                basis = basis
                vkey = "velocity_" + basis.split(prefix)[-1]
            else:
                vkey = "velocity_" + basis
                basis = prefix + basis

            if vkey not in adata.obsm_keys():
                raise ValueError(
                    f"the data corresponds to the velocity key {vkey} is not included in the adata object!"
                )

        if basis is None:
            if layer is None:
                vkey = "velocity_S"
                if vkey not in adata.uns_keys():
                    raise ValueError(
                        f"the data corresponds to the velocity key {vkey} is not included in the adata object!"
                    )

                if genes is not None:
                    X_data, V_data = (
                        adata[:, genes].X,
                        adata[:, genes].uns[vkey],
                    )
                else:
                    if "use_for_dynamics" not in adata.var.keys():
                        X_data, V_data = adata.X, adata.uns[vkey]
                    else:
                        X_data, V_data = (
                            adata[:, adata.var.use_for_dynamics].X,
                            adata[:, adata.var.use_for_dynamics].uns[vkey],
                        )
            else:
                vkey = "velocity_" + layer[0].upper()
                if vkey not in adata.uns_keys():
                    raise ValueError(
                        f"the data corresponds to the velocity key {vkey} is not included in the adata object!"
                    )

                if genes is not None:
                    X_data, V_data = (
                        adata[:, genes].layers[layer],
                        adata[:, genes].uns[vkey],
                    )
                else:
                    if "use_for_dynamics" not in adata.var.keys():
                        X_data, V_data = adata.layers[layer], adata.uns[vkey]
                    else:
                        X_data, V_data = (
                            adata[:, adata.var.use_for_dynamics].layers[layer],
                            adata[:, adata.var.use_for_dynamics].uns[vkey],
                        )
                X_data = log1p_(adata, X_data)
        else:
            X_data, V_data = adata.obsm[basis], adata.obsm[vkey]

    if dims is not None:
        X_data, V_data = X_data[:, dims], V_data[:, dims]

    neighbor_result_prefix = "" if layer is None else layer
    conn_key, dist_key, neighbor_key = generate_neighbor_keys(neighbor_result_prefix)
    if neighbor_key not in adata.uns_keys() or (X_data is not None and V_data is not None):
        Idx, _ = k_nearest_neighbors(
            X_data,
            k=n - 1,
            exclude_self=False,
            pynn_rand_state=19491001,
        )
    else:
        check_and_recompute_neighbors(adata, result_prefix=layer)
        conn_key = "connectivities" if layer is None else layer + "_connectivities"
        neighbors = adata.obsp[conn_key]
        Idx = neighbors.tolil().rows

    if residual == "average":
        V_ave = np.zeros_like(V_data)
        for i in range(X_data.shape[0]):
            vv = V_data[Idx[i]]
            V_ave[i] = vv.mean(0)
    elif residual == "vector_field":
        V_ave = func(X_data)
    else:
        raise ValueError(
            f"The method for calculate residual {residual} is not supported. "
            f'Currently only {"average", "vector_field"} supported.'
        )

    V_diff = V_data - V_ave
    val = np.zeros((V_data.shape[0], 1))
    dmatrix = [None] * V_data.shape[0]

    for i in tqdm(range(X_data.shape[0]), "calculating diffusion matrix for each cell."):
        vv = V_diff[Idx[i]]
        d = np.cov(vv.T)
        val[i] = np.sqrt(sum(sum(d)))
        dmatrix[i] = d

    adata.obs["diffusion"] = val
    adata.uns["diffusion_matrix"] = dmatrix


def diffusionMatrix2D(V_mat: np.ndarray) -> np.ndarray:
    """Function to calculate cell-specific diffusion matrix for based on velocity vectors of neighbors.

    This function works for two dimension. See :func:`diffusionMatrix` for generalization to arbitrary dimensions.

    Args:
        V_mat: velocity vectors of neighbors

    Returns:
        Return the cell-specific diffusion matrix

    See also:: :func:`diffusionMatrix`
    """

    D = np.zeros((V_mat.shape[0], 2, 2))

    D[:, 0, 0] = np.mean((V_mat[:, :, 0] - np.mean(V_mat[:, :, 0], axis=1)[:, None]) ** 2, axis=1)
    D[:, 1, 1] = np.mean((V_mat[:, :, 1] - np.mean(V_mat[:, :, 1], axis=1)[:, None]) ** 2, axis=1)
    D[:, 0, 1] = np.mean(
        (V_mat[:, :, 0] - np.mean(V_mat[:, :, 0], axis=1)[:, None])
        * (V_mat[:, :, 1] - np.mean(V_mat[:, :, 1], axis=1)[:, None]),
        axis=1,
    )
    D[:, 1, 0] = D[:, 0, 1]

    return D / 2
