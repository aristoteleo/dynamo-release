import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.base import issparse

from ..configuration import DKM
from ..dynamo_logger import (
    main_debug,
    main_info_insert_adata_layer,
    main_info_insert_adata_obsm,
    main_warning,
)
from .utils import (
    merge_adata_attrs,
)


def calc_sz_factor(
    adata_ori: anndata.AnnData,
    layers: Union[str, List[str]] = "all",
    total_layers: Union[List[str], None] = None,
    splicing_total_layers: bool = False,
    X_total_layers: bool = False,
    locfunc: Callable = np.nanmean,
    round_exprs: bool = False,
    method: Literal["mean-geometric-mean-total", "geometric", "median"] = "median",
    scale_to: Union[float, None] = None,
    use_all_genes_cells: bool = True,
    genes_use_for_norm: Union[List[str], None] = None,
) -> anndata.AnnData:
    """Calculate the size factor of each cell using geometric mean or median of total UMI across cells for a AnnData
    object.

    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Args:
        adata_ori: an AnnData object.
        layers: the layer(s) to be normalized. Defaults to "all", including RNA (X, raw) or spliced, unspliced, protein,
            etc.
        total_layers: the layer(s) that can be summed up to get the total mRNA. For example, ["spliced", "unspliced"],
            ["uu", "ul", "su", "sl"] or ["new", "old"], etc. Defaults to None.
        splicing_total_layers: whether to also normalize spliced / unspliced layers by size factor from total RNA.
            Defaults to False.
        X_total_layers: whether to also normalize adata.X by size factor from total RNA. Defaults to False.
        locfunc: the function to normalize the data. Defaults to np.nanmean.
        round_exprs: whether the gene expression should be rounded into integers. Defaults to False.
        method: the method used to calculate the expected total reads / UMI used in size factor calculation. Only
            `mean-geometric-mean-total` / `geometric` and `median` are supported. When `mean-geometric-mean-total` is
            used, size factors will be calculated using the geometric mean with given mean function. When `median` is
            used, `locfunc` will be replaced with `np.nanmedian`. When `mean` is used, `locfunc` will be replaced with
            `np.nanmean`. Defaults to "median".
        scale_to: the final total expression for each cell that will be scaled to. Defaults to None.
        use_all_genes_cells: whether all cells and genes should be used for the size factor calculation. Defaults to
            True.
        genes_use_for_norm: A list of gene names that will be used to calculate total RNA for each cell and then the
            size factor for normalization. This is often very useful when you want to use only the host genes to
            normalize the dataset in a virus infection experiment (i.e. CMV or SARS-CoV-2 infection). Defaults to None.

    Returns:
        An updated anndata object that are updated with the `Size_Factor` (`layer_` + `Size_Factor`) column(s) in the
        obs attribute.
    """

    if use_all_genes_cells:
        # let us ignore the `inplace` parameter in pandas.Categorical.remove_unused_categories  warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _adata = adata_ori if genes_use_for_norm is None else adata_ori[:, genes_use_for_norm]
    else:
        cell_inds = adata_ori.obs.use_for_pca if "use_for_pca" in adata_ori.obs.columns else adata_ori.obs.index
        filter_list = ["use_for_pca", "pass_basic_filter"]
        filter_checker = [i in adata_ori.var.columns for i in filter_list]
        which_filter = np.where(filter_checker)[0]

        gene_inds = adata_ori.var[filter_list[which_filter[0]]] if len(which_filter) > 0 else adata_ori.var.index

        _adata = adata_ori[cell_inds, :][:, gene_inds]

        if genes_use_for_norm is not None:
            # let us ignore the `inplace` parameter in pandas.Categorical.remove_unused_categories  warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                _adata = _adata[:, _adata.var_names.intersection(genes_use_for_norm)]

    if total_layers is not None:
        total_layers, layers = DKM.aggregate_layers_into_total(
            _adata,
            layers=layers,
            total_layers=total_layers,
        )

    layers = DKM.get_available_layer_keys(_adata, layers)
    if "raw" in layers and _adata.raw is None:
        _adata.raw = _adata.copy()

    excluded_layers = DKM.get_excluded_layers(
        X_total_layers=X_total_layers,
        splicing_total_layers=splicing_total_layers,
    )

    for layer in layers:
        if layer in excluded_layers:
            sfs, cell_total = sz_util(
                _adata,
                layer,
                round_exprs,
                method,
                locfunc,
                total_layers=None,
                scale_to=scale_to,
            )
        else:
            sfs, cell_total = sz_util(
                _adata,
                layer,
                round_exprs,
                method,
                locfunc,
                total_layers=total_layers,
                scale_to=scale_to,
            )

        sfs[~np.isfinite(sfs)] = 1
        if layer == "raw":
            _adata.obs[layer + "_Size_Factor"] = sfs
            _adata.obs["Size_Factor"] = sfs
            _adata.obs["initial_cell_size"] = cell_total
        elif layer == "X":
            _adata.obs["Size_Factor"] = sfs
            _adata.obs["initial_cell_size"] = cell_total
        elif layer == "_total_":
            _adata.obs["total_Size_Factor"] = sfs
            _adata.obs["initial" + layer + "cell_size"] = cell_total
            del _adata.layers["_total_"]
        else:
            _adata.obs[layer + "_Size_Factor"] = sfs
            _adata.obs["initial_" + layer + "_cell_size"] = cell_total

    adata_ori = merge_adata_attrs(adata_ori, _adata, attr="obs")

    return adata_ori


def get_sz_exprs(
    adata: anndata.AnnData, layer: str, total_szfactor: Union[str, None] = None
) -> Tuple[np.ndarray, npt.ArrayLike]:
    """Get the size factor from an AnnData object.

    Args:
        adata: an AnnData object.
        layer: the layer for which to get the size factor.
        total_szfactor: the key-name for total size factor entry in `adata.obs`. If not None, would override the layer
            selected. Defaults to None.

    Returns:
        A tuple (szfactors, CM), where szfactors is the queried size factor and CM is the data of the layer
        corresponding to the size factor.
    """

    try:
        if layer == "raw":
            CM = adata.raw.X
            szfactors = adata.obs[layer + "Size_Factor"].values[:, None]
        elif layer == "X":
            CM = adata.X
            szfactors = adata.obs["Size_Factor"].values[:, None]
        elif layer == "protein":
            if "protein" in adata.obsm_keys():
                CM = adata.obsm[layer]
                szfactors = adata.obs["protein_Size_Factor"].values[:, None]
            else:
                CM, szfactors = None, None
        else:
            CM = adata.layers[layer]
            szfactors = adata.obs[layer + "_Size_Factor"].values[:, None]

        if total_szfactor is not None and total_szfactor in adata.obs.keys():
            szfactors = adata.obs[total_szfactor].values[:, None]
        elif total_szfactor is not None:
            main_warning("`total_szfactor` is not `None` and it is not in adata object.")
    except KeyError:
        raise KeyError(f"Size factor for layer {layer} is not in adata object. Please run `dynamo.tl.calc_sz_factor`.")

    return szfactors, CM


def normalize(
    adata: anndata.AnnData,
    layers: str = "all",
    total_szfactor: str = None,
    splicing_total_layers: bool = False,
    X_total_layers: bool = False,
    keep_filtered: bool = True,
    recalc_sz: bool = False,
    sz_method: Literal["mean-geometric-mean-total", "geometric", "median"] = "median",
    scale_to: Union[float, None] = None,
) -> anndata.AnnData:
    """Normalize the gene expression value for the AnnData object.

    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Args:
        adata: an AnnData object.
        layers: the layer(s) to be normalized. Default is all, including RNA (X, raw) or spliced, unspliced, protein,
            etc.
        total_szfactor: the column name in the .obs attribute that corresponds to the size factor for the total mRNA.
            Defaults to "total_Size_Factor".
        splicing_total_layers: whether to also normalize spliced / unspliced layers by size factor from total RNA.
            Defaults to False.
        X_total_layers: whether to also normalize adata.X by size factor from total RNA. Defaults to False.
        keep_filtered: whether we will only store feature genes in the adata object. If it is False, size factor will be
            recalculated only for the selected feature genes. Defaults to True.
        recalc_sz: whether we need to recalculate size factor based on selected genes before normalization. Defaults to
            False.
        sz_method: the method used to calculate the expected total reads / UMI used in size factor calculation. Only
            `mean-geometric-mean-total` / `geometric` and `median` are supported. When `mean-geometric-mean-total` is
            used, size factors will be calculated using the geometric mean with given mean function. When `median` is
            used, `locfunc` will be replaced with `np.nanmedian`. When `mean` is used, `locfunc` will be replaced with
            `np.nanmean`. Defaults to "median".
        scale_to: the final total expression for each cell that will be scaled to. Defaults to None.

    Returns:
        An updated anndata object that are updated with normalized expression values for different layers.
    """

    layers = DKM.get_available_layer_keys(adata, layers)

    if recalc_sz:
        if "use_for_pca" in adata.var.columns and keep_filtered is False:
            adata = adata[:, adata.var.loc[:, "use_for_pca"]]

        adata.obs = adata.obs.loc[:, ~adata.obs.columns.str.contains("Size_Factor")]

    if np.count_nonzero(adata.obs.columns.str.contains("Size_Factor")) < len(layers):
        calc_sz_factor(
            adata,
            layers=layers,
            locfunc=np.nanmean,
            round_exprs=False,
            method=sz_method,
            scale_to=scale_to,
        )

    excluded_layers = DKM.get_excluded_layers(
        X_total_layers=X_total_layers,
        splicing_total_layers=splicing_total_layers,
    )

    main_debug("size factor normalize following layers: " + str(layers))
    for layer in layers:
        if layer in excluded_layers:
            szfactors, CM = get_sz_exprs(adata, layer, total_szfactor=None)
        else:
            szfactors, CM = get_sz_exprs(adata, layer, total_szfactor=total_szfactor)

        if layer == "protein":
            """This normalization implements the centered log-ratio (CLR) normalization from Seurat which is computed
            for each gene (M Stoeckius, 2017).
            """
            CM = CM.T
            n_feature = CM.shape[1]

            for i in range(CM.shape[0]):
                x = CM[i].A if issparse(CM) else CM[i]
                res = np.log1p(x / (np.exp(np.nansum(np.log1p(x[x > 0])) / n_feature)))
                res[np.isnan(res)] = 0
                # res[res > 100] = 100
                # no .A is required # https://stackoverflow.com/questions/28427236/set-row-of-csr-matrix
                CM[i] = res

            CM = CM.T
        else:
            CM = size_factor_normalize(CM, szfactors)

        if layer in ["raw", "X"]:
            main_debug("set adata <X> to normalized data.")
            adata.X = CM
        elif layer == "protein" and "protein" in adata.obsm_keys():
            main_info_insert_adata_obsm("X_protein")
            adata.obsm["X_protein"] = CM
        else:
            main_info_insert_adata_layer("X_" + layer)
            adata.layers["X_" + layer] = CM

    return adata


def normalize_mat_monocle(
    mat: np.ndarray,
    szfactors: np.ndarray,
    relative_expr: bool,
    pseudo_expr: int,
    norm_method: Callable = np.log1p,
) -> np.ndarray:
    """Normalize the given array for monocle recipe.

    Args:
        mat: the array to operate on.
        szfactors: the size factors corresponding to the array.
        relative_expr: whether we need to divide gene expression values first by
            size factor before normalization.
        pseudo_expr: a pseudocount added to the gene expression value before
            log/log2 normalization.
        norm_method: the method used to normalize data. Defaults to np.log1p.

    Returns:
        The normalized array.
    """

    if norm_method == np.log1p:
        pseudo_expr = 0
    if relative_expr:
        mat = mat.multiply(csr_matrix(1 / szfactors)) if issparse(mat) else mat / szfactors

    if pseudo_expr is None:
        pseudo_expr = 1

    if issparse(mat):
        mat.data = norm_method(mat.data + pseudo_expr) if norm_method is not None else mat.data
        if norm_method is not None and norm_method.__name__ == "Freeman_Tukey":
            mat.data -= 1
    else:
        mat = norm_method(mat + pseudo_expr) if norm_method is not None else mat

    return mat


def size_factor_normalize(mat: np.ndarray, szfactors: np.ndarray) -> np.ndarray:
    """perform size factor normalization on the given array.

    Args:
        mat: the array to operate on.
        szfactors: the size factors corresponding to the array.

    Returns:
        The normalized array divided by size factor
    """
    return mat.multiply(csr_matrix(1 / szfactors)) if issparse(mat) else mat / szfactors


def sz_util(
    adata: anndata.AnnData,
    layer: str,
    round_exprs: bool,
    method: Literal["mean-geometric-mean-total", "geometric", "median"],
    locfunc: Callable,
    total_layers: List[str] = None,
    CM: pd.DataFrame = None,
    scale_to: Union[float, None] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Calculate the size factor for a given layer.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on.
        round_exprs: whether the gene expression should be rounded into integers.
        method: the method used to calculate the expected total reads / UMI used in size factor calculation. Only
            `mean-geometric-mean-total` / `geometric` and `median` are supported. When `mean-geometric-mean-total` is
            used, size factors will be calculated using the geometric mean with given mean function. When `median` is
            used, `locfunc` will be replaced with `np.nanmedian`. When `mean` is used, `locfunc` will be replaced with
            `np.nanmean`. Defaults to "median".
        locfunc: the function to normalize the data.
        total_layers: the layer(s) that can be summed up to get the total mRNA. For example, ["spliced", "unspliced"],
            ["uu", "ul", "su", "sl"] or ["new", "old"], etc. Defaults to None.
        CM: the data to operate on, overriding the layer. Defaults to None.
        scale_to: the final total expression for each cell that will be scaled to. Defaults to None.

    Raises:
        NotImplementedError: method is invalid.

    Returns:
        A tuple (sfs, cell_total) where sfs is the size factors and cell_total is the initial cell size.
    """

    adata = adata.copy()

    if layer == "_total_" and "_total_" not in adata.layers.keys():
        if total_layers is not None:
            total_layers, _ = DKM.aggregate_layers_into_total(
                adata,
                total_layers=total_layers,
                extend_layers=False,
            )

    CM = DKM.select_layer_data(adata, layer) if CM is None else CM
    if CM is None:
        return None, None

    if round_exprs:
        main_debug("rounding expression data of layer: %s during size factor calculation" % (layer))
        if issparse(CM):
            CM.data = np.round(CM.data, 0)
        else:
            CM = CM.round().astype("int")

    cell_total = CM.sum(axis=1).A1 if issparse(CM) else CM.sum(axis=1)
    cell_total += cell_total == 0  # avoid infinity value after log (0)

    if method in ["mean-geometric-mean-total", "geometric"]:
        sfs = cell_total / (np.exp(locfunc(np.log(cell_total))) if scale_to is None else scale_to)
    elif method == "median":
        sfs = cell_total / (np.nanmedian(cell_total) if scale_to is None else scale_to)
    elif method == "mean":
        sfs = cell_total / (np.nanmean(cell_total) if scale_to is None else scale_to)
    else:
        raise NotImplementedError(f"This method {method} is not supported!")

    return sfs, cell_total
