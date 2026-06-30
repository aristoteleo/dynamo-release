import warnings
from typing import Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata
import numpy as np
from anndata import AnnData
from scipy.sparse import issparse

from ..configuration import DKM
from ..dynamo_logger import main_debug, main_info_insert_adata_uns
from ..utils import copy_adata
from .utils import is_integer_arr


def _Freeman_Tukey(X: np.ndarray, inverse=False) -> np.ndarray:
    """perform Freeman-Tukey transform or inverse transform on the given array.

    Args:
        X: a matrix.
        inverse: whether to perform inverse Freeman-Tukey transform. Defaults to False.

    Returns:
        The transformed array.
    """

    if inverse:
        res = (X**2 - 1) ** 2 / (4 * X**2)
    else:
        res = np.sqrt(X) + np.sqrt((X + 1))

    return res


def _log_inplace(data: np.ndarray) -> np.ndarray:
    """Calculate the natural logarithm `log(exp(x)) = x` of an array and update the array inplace.

    Args:
        data: the array for calculation.

    Returns:
        The updated array.
    """

    return np.log(data + 1, out=data)


def _log1p_inplace(data: np.ndarray) -> np.ndarray:
    """Calculate log1p (log(1+x)) of an array and update the array inplace.

    Args:
        data: the array for calculation.

    Returns:
        The updated array.
    """

    return np.log1p(data, out=data)


def _log2_inplace(data: np.ndarray) -> np.ndarray:
    """Calculate Base-2 logarithm of `x` of an array and update the array inplace.

    Args:
        data: the array for calculation.

    Returns:
        The updated array.
    """

    return np.log2(data + 1, out=data)


def Freeman_Tukey_inplace(adata: AnnData, layer: str = DKM.X_LAYER) -> None:
    """Calculate Freeman-Tukey transform for a layer of an AnnData object inplace.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on. Defaults to DKM.X_LAYER.
    """
    mat = DKM.select_layer_data(adata, layer, copy=False)
    if issparse(mat):
        if is_integer_arr(mat.data):
            mat = mat.asfptype()
            DKM.set_layer_data(adata, layer, mat)
        mat.data = _Freeman_Tukey(mat.data)
    else:
        mat = mat.astype(np.float64)
        mat = _Freeman_Tukey(mat)

    DKM.set_layer_data(adata, layer, mat)


def log_inplace(adata: AnnData, layer: str = DKM.X_LAYER) -> None:
    """Calculate the natural logarithm `log(exp(x)) = x` for a layer of an AnnData object inplace.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on. Defaults to DKM.X_LAYER.
    """

    mat = DKM.select_layer_data(adata, layer, copy=False)
    if issparse(mat):
        if is_integer_arr(mat.data):
            mat = mat.asfptype()
            DKM.set_layer_data(adata, layer, mat)
        _log_inplace(mat.data)
    else:
        mat = mat.astype(np.float64)
        _log_inplace(mat)
        DKM.set_layer_data(adata, layer, mat)


def log1p_inplace(adata: AnnData, layer: str = DKM.X_LAYER) -> None:
    """Calculate log1p (log(1+x)) for a layer of an AnnData object inplace.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on. Defaults to DKM.X_LAYER.
    """

    mat = DKM.select_layer_data(adata, layer, copy=False)
    if issparse(mat):
        if is_integer_arr(mat.data):
            mat = mat.asfptype()
            DKM.set_layer_data(adata, layer, mat)
        _log1p_inplace(mat.data)
    else:
        mat = mat.astype(np.float64)
        _log1p_inplace(mat)
        DKM.set_layer_data(adata, layer, mat)


def log2_inplace(adata: AnnData, layer: str = DKM.X_LAYER) -> None:
    """Calculate Base-2 logarithm of `x` for a layer of an AnnData object inplace.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on. Defaults to DKM.X_LAYER.
    """

    mat = DKM.select_layer_data(adata, layer, copy=False)
    if issparse(mat):
        if is_integer_arr(mat.data):
            mat = mat.asfptype()
            DKM.set_layer_data(adata, layer, mat)
        _log2_inplace(mat.data)
    else:
        mat = mat.astype(np.float64)
        _log2_inplace(mat)
        DKM.set_layer_data(adata, layer, mat)


def is_log1p_transformed_adata(adata: anndata.AnnData) -> bool:
    """check if adata data is log transformed by checking a small subset of adata observations.

    Args:
        adata: an AnnData object

    Returns:
        A flag shows whether the adata object is log transformed.
    """

    rng = np.random.default_rng()
    chosen_gene_indices = rng.choice(adata.n_vars, 10)
    _has_log1p_transformed = not np.allclose(
        np.array(adata.X[:, chosen_gene_indices].sum(1)),
        np.array(adata.layers["spliced"][:, chosen_gene_indices].sum(1)),
        atol=1e-4,
    )
    return _has_log1p_transformed


def Freeman_Tukey_adata_layer(adata: AnnData, layer: str = DKM.X_LAYER, copy: bool = False) -> AnnData:
    """Calculate Freeman_Tukey of adata's specific layer.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on. Defaults to DKM.X_LAYER.
        copy: whether operate on the original object or on a copied one and return it. Defaults to False.

    Returns:
        The updated AnnData object.
    """

    _adata = adata
    if copy:
        _adata = copy_adata(adata)
    Freeman_Tukey_inplace(_adata, layer=layer)
    return _adata


def log_adata_layer(adata: AnnData, layer: str = DKM.X_LAYER, copy: bool = False) -> AnnData:
    """Calculate log of adata's specific layer.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on. Defaults to DKM.X_LAYER.
        copy: whether operate on the original object or on a copied one and return it. Defaults to False.

    Returns:
        The updated AnnData object.
    """

    _adata = adata
    if copy:
        _adata = copy_adata(adata)
    log_inplace(_adata, layer=layer)
    return _adata


def log1p_adata_layer(adata: AnnData, layer: str = DKM.X_LAYER, copy: bool = False) -> AnnData:
    """Calculate log1p of adata's specific layer.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on. Defaults to DKM.X_LAYER.
        copy: whether operate on the original object or on a copied one and return it. Defaults to False.

    Returns:
        The updated AnnData object.
    """

    _adata = adata
    if copy:
        _adata = copy_adata(adata)
    log1p_inplace(_adata, layer=layer)
    return _adata


def log2_adata_layer(adata: AnnData, layer: str = DKM.X_LAYER, copy: bool = False) -> AnnData:
    """Calculate log2 of adata's specific layer.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on. Defaults to DKM.X_LAYER.
        copy: whether operate on the original object or on a copied one and return it. Defaults to False.

    Returns:
        The updated AnnData object.
    """

    _adata = adata
    if copy:
        _adata = copy_adata(adata)
    log2_inplace(_adata, layer=layer)
    return _adata


def Freeman_Tukey(adata: AnnData, layers: list = [DKM.X_LAYER], copy: bool = False) -> AnnData:
    """Perform Freeman_Tukey transform on selected adata layers

    Args:
        adata: an AnnData object.
        layers: the layers to operate on. Defaults to [DKM.X_LAYER].
        copy: whether operate on the original object or on a copied one and return it. Defaults to False.

    Returns:
        The updated AnnData object.
    """

    _adata = adata
    if copy:
        _adata = copy_adata(adata)

    main_debug("[Freeman_Tukey] transform applied to layers: %s" % (str(layers)))
    for layer in layers:
        Freeman_Tukey_adata_layer(_adata, layer=layer)

        if layer == DKM.X_LAYER:
            main_info_insert_adata_uns("pp.X_norm_method")
            adata.uns["pp"]["X_norm_method"] = Freeman_Tukey.__name__
        else:
            main_info_insert_adata_uns("pp.layers_norm_method")
            adata.uns["pp"]["layers_norm_method"] = Freeman_Tukey.__name__

    return _adata


def log(adata: AnnData, layers: list = [DKM.X_LAYER], copy: bool = False) -> AnnData:
    """Perform log transform on selected adata layers

    Args:
        adata: an AnnData object.
        layers: the layers to operate on. Defaults to [DKM.X_LAYER].
        copy: whether operate on the original object or on a copied one and return it. Defaults to False.

    Returns:
        The updated AnnData object.
    """

    _adata = adata
    if copy:
        _adata = copy_adata(adata)

    main_debug("[log] transform applied to layers: %s" % (str(layers)))
    for layer in layers:
        log_adata_layer(_adata, layer=layer)

        if layer == DKM.X_LAYER:
            main_info_insert_adata_uns("pp.X_norm_method")
            adata.uns["pp"]["X_norm_method"] = log.__name__
        else:
            main_info_insert_adata_uns("pp.layers_norm_method")
            adata.uns["pp"]["layers_norm_method"] = log.__name__

    return _adata


def log1p(adata: AnnData, layers: list = [DKM.X_LAYER], copy: bool = False) -> AnnData:
    """Perform log1p transform on selected adata layers

    Args:
        adata: an AnnData object.
        layers: the layers to operate on. Defaults to [DKM.X_LAYER].
        copy: whether operate on the original object or on a copied one and return it. Defaults to False.

    Returns:
        The updated AnnData object.
    """

    _adata = adata
    if copy:
        _adata = copy_adata(adata)

    main_debug("[log1p] transform applied to layers: %s" % (str(layers)))
    for layer in layers:
        log1p_adata_layer(_adata, layer=layer)

        if layer == DKM.X_LAYER:
            main_info_insert_adata_uns("pp.X_norm_method")
            adata.uns["pp"]["X_norm_method"] = log1p.__name__
        else:
            main_info_insert_adata_uns("pp.layers_norm_method")
            adata.uns["pp"]["layers_norm_method"] = log1p.__name__

    return _adata


def log2(adata: AnnData, layers: list = [DKM.X_LAYER], copy: bool = False) -> AnnData:
    """Perform log2 transform on selected adata layers

    Args:
        adata: an AnnData object.
        layers: the layers to operate on. Defaults to [DKM.X_LAYER].
        copy: whether operate on the original object or on a copied one and return it. Defaults to False.

    Returns:
        The updated AnnData object.
    """

    _adata = adata
    if copy:
        _adata = copy_adata(adata)

    main_debug("[log2] transform applied to layers: %s" % (str(layers)))
    for layer in layers:
        log2_adata_layer(_adata, layer=layer)

        if layer == DKM.X_LAYER:
            main_info_insert_adata_uns("pp.X_norm_method")
            adata.uns["pp"]["X_norm_method"] = log2.__name__
        else:
            main_info_insert_adata_uns("pp.layers_norm_method")
            adata.uns["pp"]["layers_norm_method"] = log2.__name__

    return _adata


def pflog1ppf_inplace(adata: AnnData, layer: str = DKM.X_LAYER, target_sum: Optional[float] = None) -> None:
    """Apply the PFlog1pPF transform to a layer of an AnnData object inplace.

    PFlog1pPF (Booeshaghi, Hjörleifsson, Gehring & Pachter, 2022; https://github.com/pachterlab/BHGP_2022) is the
    composition of proportional fitting (PF), ``log1p`` and a second proportional fitting:

        ``PF -> log1p -> PF``

    The first PF places every cell at a common sequencing depth, ``log1p`` stabilizes the variance, and the second
    PF re-equalizes the depth that ``log1p`` distorts. All three steps are per-cell rescalings or elementwise
    transforms of nonzero entries, so the result stays sparse.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on. Defaults to DKM.X_LAYER.
        target_sum: the per-cell target depth used by both PF steps. If None, each PF step uses the mean depth of
            the data it sees, which is the canonical BHGP behavior. Defaults to None.
    """
    from .normalization import proportional_fitting

    mat = DKM.select_layer_data(adata, layer, copy=False)
    if issparse(mat):
        if is_integer_arr(mat.data):
            mat = mat.asfptype()
        mat, _ = proportional_fitting(mat, target_sum=target_sum)
        mat.data = np.log1p(mat.data)
        mat, _ = proportional_fitting(mat, target_sum=target_sum)
    else:
        mat = mat.astype(np.float64)
        mat, _ = proportional_fitting(mat, target_sum=target_sum)
        mat = np.log1p(mat)
        mat, _ = proportional_fitting(mat, target_sum=target_sum)

    DKM.set_layer_data(adata, layer, mat)


def pflog1ppf_adata_layer(
    adata: AnnData, layer: str = DKM.X_LAYER, target_sum: Optional[float] = None, copy: bool = False
) -> AnnData:
    """Apply the PFlog1pPF transform to a single layer of an AnnData object.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on. Defaults to DKM.X_LAYER.
        target_sum: the per-cell target depth used by both PF steps. If None, the mean depth is used at each PF
            step. Defaults to None.
        copy: whether operate on the original object or on a copied one and return it. Defaults to False.

    Returns:
        The updated AnnData object.
    """

    _adata = adata
    if copy:
        _adata = copy_adata(adata)
    pflog1ppf_inplace(_adata, layer=layer, target_sum=target_sum)
    return _adata


def pflog1ppf(
    adata: AnnData, layers: list = [DKM.X_LAYER], target_sum: Optional[float] = None, copy: bool = False
) -> AnnData:
    """Perform the PFlog1pPF depth-normalization transform on selected adata layers.

    PFlog1pPF (proportional fitting -> log1p -> proportional fitting) is the depth-normalization recipe of
    Booeshaghi, Hjörleifsson, Gehring & Pachter (2022). It is a drop-in replacement for the size-factor + ``log1p``
    normalization that, unlike a single library-size normalization, also corrects the depth distortion that
    ``log1p`` introduces, while keeping the matrix sparse.

    A. Sina Booeshaghi, Ingileif B. Hjörleifsson, Lambda Moses, Lior Pachter. Depth normalization for single-cell
    genomics count data. bioRxiv (2022). https://doi.org/10.1101/2022.05.06.490859

    Args:
        adata: an AnnData object.
        layers: the layers to operate on. Defaults to [DKM.X_LAYER].
        target_sum: the per-cell target depth used by both PF steps. If None, the mean depth of the data is used at
            each PF step (the canonical behavior). Defaults to None.
        copy: whether operate on the original object or on a copied one and return it. Defaults to False.

    Returns:
        The updated AnnData object.
    """

    _adata = adata
    if copy:
        _adata = copy_adata(adata)

    main_debug("[pflog1ppf] transform applied to layers: %s" % (str(layers)))
    for layer in layers:
        pflog1ppf_adata_layer(_adata, layer=layer, target_sum=target_sum)

        if layer == DKM.X_LAYER:
            main_info_insert_adata_uns("pp.X_norm_method")
            adata.uns["pp"]["X_norm_method"] = pflog1ppf.__name__
        else:
            main_info_insert_adata_uns("pp.layers_norm_method")
            adata.uns["pp"]["layers_norm_method"] = pflog1ppf.__name__

    return _adata


def vstExprs(
    adata: anndata.AnnData,
    expr_matrix: Union[np.ndarray, None] = None,
    round_vals: bool = True,
) -> np.ndarray:
    """Variance stabilization transformation of the gene expression.

    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Args:
        adata: an AnnData object.
        expr_matrix: an matrix of values to transform. Must be normalized (e.g. by size factors) already. Defaults to
            None.
        round_vals: whether to round expression values to the nearest integer before applying the transformation.
            Defaults to True.

    Returns:
        A numpy array of the gene expression after VST.
    """

    fitInfo = adata.uns["dispFitInfo"]

    coefs = fitInfo["coefs"]
    if expr_matrix is None:
        ncounts = adata.X
        if round_vals:
            if issparse(ncounts):
                ncounts.data = np.round(ncounts.data, 0)
            else:
                ncounts = ncounts.round().astype("int")
    else:
        ncounts = expr_matrix

    def vst(q):  # c( "asymptDisp", "extraPois" )
        return np.log(
            (1 + coefs[1] + 2 * coefs[0] * q + 2 * np.sqrt(coefs[0] * q * (1 + coefs[1] + coefs[0] * q)))
            / (4 * coefs[0])
        ) / np.log(2)

    res = vst(ncounts.toarray()) if issparse(ncounts) else vst(ncounts)

    return res
