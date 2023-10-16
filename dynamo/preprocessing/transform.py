import warnings
from typing import Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata
import numpy as np
from anndata import AnnData
from scipy.sparse.base import issparse

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

    chosen_gene_indices = np.random.choice(adata.n_vars, 10)
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
