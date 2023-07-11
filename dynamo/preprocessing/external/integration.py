from typing import List, Optional, Union

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix, isspmatrix

# Convert sparse matrix to dense matrix.
to_dense_matrix = lambda X: np.array(X.todense()) if isspmatrix(X) else np.asarray(X)

def integrate(
    adatas: List[AnnData],
    batch_key: str = "slices",
    fill_value: Union[int, float] = 0,
) -> AnnData:
    """Concatenating all anndata objects.

    Args:
        adatas: AnnData matrices to concatenate with.
        batch_key: the key to add the batch annotation to :attr:`obs`.
        fill_value: Scalar value to fill newly missing values in arrays with.

    Returns:
        The concatenated AnnData, where adata.obs[batch_key] stores a categorical variable labeling the batch.
    """

    batch_ca = [adata.obs[batch_key][0] for adata in adatas]

    # Merge the obsm, varm and uns data of all anndata objcets separately.
    obsm_dict, varm_dict, uns_dict = {}, {}, {}
    obsm_keys, varm_keys, uns_keys = [], [], []
    for adata in adatas:
        obsm_keys.extend(list(adata.obsm.keys()))
        varm_keys.extend(list(adata.varm.keys()))
        uns_keys.extend(list(adata.uns_keys()))

    obsm_keys, varm_keys, uns_keys = list(set(obsm_keys)), list(set(varm_keys)), list(set(uns_keys))
    n_obsm_keys, n_varm_keys, n_uns_keys = len(obsm_keys), len(varm_keys), len(uns_keys)

    if n_obsm_keys > 0:
        for key in obsm_keys:
            obsm_dict[key] = np.concatenate([to_dense_matrix(adata.obsm[key]) for adata in adatas], axis=0)
    if n_varm_keys > 0:
        for key in varm_keys:
            varm_dict[key] = np.concatenate([to_dense_matrix(adata.varm[key]) for adata in adatas], axis=0)
    if n_uns_keys > 0:
        for key in uns_keys:
            if "__type" in uns_keys and key == "__type":
                uns_dict["__type"] = adatas[0].uns["__type"]
            else:
                uns_dict[key] = {
                    ca: adata.uns[key] if key in adata.uns_keys() else None for ca, adata in zip(batch_ca, adatas)
                }

    # Delete obsm, varm and uns data.
    for adata in adatas:
        del adata.obsm, adata.varm, adata.uns

    # Concatenating obs and var data which will ignore the uns, obsm, varm attributes.
    integrated_adata = adatas[0].concatenate(
        *adatas[1:],
        batch_key=batch_key,
        batch_categories=batch_ca,
        join="outer",
        fill_value=fill_value,
        uns_merge=None,
    )

    # Add Concatenated obsm data and varm data to integrated anndata object.
    if n_obsm_keys > 0:
        for key, value in obsm_dict.items():
            integrated_adata.obsm[key] = value
    if n_varm_keys > 0:
        for key, value in varm_dict.items():
            integrated_adata.varm[key] = value
    if n_uns_keys > 0:
        for key, value in uns_dict.items():
            integrated_adata.uns[key] = value

    return integrated_adata

def harmony_debatch(
    adata: AnnData,
    key: str,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
    max_iter_harmony: int = 10,
    copy: bool = False,
) -> Optional[AnnData]:
    """Use harmonypy [Korunsky19]_ to remove batch effects.

    This function should be run after performing PCA but before computing the neighbor graph. Original Code Repository
    is https://github.com/slowkow/harmonypy. Interesting example: https://slowkow.com/notes/harmony-animation/

    Args:
        adata: An Anndata object.
        key: The name of the column in ``adata.obs`` that differentiates among experiments/batches.
        basis: The name of the field in ``adata.obsm`` where the PCA table is stored.
        adjusted_basis: The name of the field in ``adata.obsm`` where the adjusted PCA table will be stored after
            running this function.
        max_iter_harmony: Maximum number of rounds to run Harmony. One round of Harmony involves one clustering and one
            correction step.
        copy: Whether to copy `adata` or modify it inplace.

    Returns:
        Updates adata with the field ``adata.obsm[adjusted_basis]``, containing principal components adjusted by
        Harmony.
    """
    try:
        import harmonypy
    except ImportError:
        raise ImportError("\nplease install harmonypy:\n\n\tpip install harmonypy")

    adata = adata.copy() if copy else adata

    # Convert sparse matrix to dense matrix.
    matrix = to_dense_matrix(adata.obsm[basis])

    # Use Harmony to adjust the PCs.
    harmony_out = harmonypy.run_harmony(matrix, adata.obs, key, max_iter_harmony=max_iter_harmony)
    adjusted_matrix = harmony_out.Z_corr.T

    # Convert dense matrix to sparse matrix.
    if isspmatrix(adata.obsm[basis]):
        adjusted_matrix = csr_matrix(adjusted_matrix)

    adata.obsm[adjusted_basis] = adjusted_matrix

    return adata if copy else None