# from tqdm import tqdm

# from anndata._core.views import ArrayView
# import scipy.sparse as sp
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from anndata._core.anndata import AnnData
from numpy.typing import DTypeLike

from ..dynamo_logger import main_info_insert_adata_uns
from ..tools.utils import (
    create_layer,
    get_rank_array,
    index_gene,
    list_top_genes,
    list_top_interactions,
    table_top_genes,
)
from ..utils import isarray, ismatrix
from .utils import average_jacobian_by_group, intersect_sources_targets

try:
    import dynode

    use_dynode = "vectorfield" in dir(dynode)
except ImportError:
    use_dynode = False

if use_dynode:
    from .scVectorField import dynode_vectorfield


def rank_genes(
    adata: AnnData,
    arr_key: Union[str, np.ndarray],
    groups: Optional[str] = None,
    genes: Optional[List] = None,
    abs: bool = False,
    normalize: bool = False,
    fcn_pool: Callable = lambda x: np.mean(x, axis=0),
    dtype: Optional[DTypeLike] = None,
    output_values: bool = False,
) -> pd.DataFrame:
    """Rank gene's absolute, positive, negative vector field metrics by different cell groups.

    Args:
        adata: AnnData object that contains the array to be sorted in `.var` or `.layer`.
        arr_key: The key of the to-be-ranked array stored in `.var` or or `.layer`.
            If the array is found in `.var`, the `groups` argument will be ignored.
            If a numpy array is passed, it is used as the array to be ranked and must
            be either an 1d array of length `.n_var`, or a `.n_obs`-by-`.n_var` 2d array.
        groups: Cell groups used to group the array.
        genes: The gene list that speed will be ranked. If provided, they must overlap the dynamics genes.
        abs: When pooling the values in the array (see below), whether to take the absolute values.
        normalize: bool (default: False)
            Whether normalize the array across all cells first, if the array is 2d.
        fcn_pool: callable (default: numpy.mean(x, axis=0))
            The function used to pool values in the to-be-ranked array if the array is 2d.
        output_values: bool (default: False)
            Whether output the values along with the rankings.

    Returns:
        A dataframe of gene names and values based on which the genes are sorted for each cell group.
    """

    genes, arr = get_rank_array(
        adata,
        arr_key,
        genes=genes,
        abs=abs,
        dtype=dtype,
    )

    if arr.ndim > 1:
        if normalize:
            arr_max = np.max(np.abs(arr), axis=0)
            arr = arr / arr_max
            arr[np.isnan(arr)] = 0
        if groups is not None:
            if type(groups) is str and groups in adata.obs.keys():
                grps = np.array(adata.obs[groups])
            elif isarray(groups):
                grps = np.array(groups)
            else:
                raise Exception(f"The group information {groups} you provided is not in your adata object.")
            arr_dict = {}
            for g in np.unique(grps):
                arr_dict[g] = fcn_pool(arr[grps == g])
        else:
            arr_dict = {"all": fcn_pool(arr)}
    else:
        arr_dict = {"all": arr}

    ret_dict = {}
    var_names = np.array(index_gene(adata, adata.var_names, genes))
    for g, arr in arr_dict.items():
        if ismatrix(arr):
            arr = arr.A.flatten()
        glst, sarr = list_top_genes(arr, var_names, None, return_sorted_array=True)
        # ret_dict[g] = {glst[i]: sarr[i] for i in range(len(glst))}
        ret_dict[g] = glst
        if output_values:
            ret_dict[g + "_values"] = sarr
    return pd.DataFrame(data=ret_dict)


def rank_cell_groups(
    adata: AnnData,
    arr_key: Union[str, np.ndarray],
    groups: Optional[str] = None,
    genes: Optional[List] = None,
    abs: bool = False,
    fcn_pool: Callable = lambda x: np.mean(x, axis=0),
    dtype: Optional[DTypeLike] = None,
    output_values: bool = False,
) -> pd.DataFrame:
    """Rank cell's absolute, positive, negative vector field metrics by different gene groups.

    Args:
        adata: AnnData object that contains the array to be sorted in `.var` or `.layer`.
        arr_key: The key of the to-be-ranked array stored in `.var` or `.layer`.
            If the array is found in `.var`, the `groups` argument will be ignored.
            If a numpy array is passed, it is used as the array to be ranked and must
            be either a 1d array of length `.n_var`, or a `.n_obs`-by-`.n_var` 2d array.
        groups: Gene groups used to group the array.
        genes: The gene list that speed will be ranked. If provided, they must overlap the dynamics genes.
        abs: When pooling the values in the array (see below), whether to take the absolute values.
        fcn_pool: The function used to pool values in the to-be-ranked array if the array is 2d.
        dtype: The data type of the array to be ranked.
        output_values: Whether output the values along with the rankings.

    Returns:
        A dataframe of cells names and values based on which the genes are sorted for each gene group.
    """

    genes, arr = get_rank_array(
        adata,
        arr_key,
        genes=genes,
        abs=abs,
        dtype=dtype,
    )
    arr = arr.T

    if arr.ndim > 1:
        if groups is not None:
            if type(groups) is str and groups in adata.var.keys():
                grps = np.array(adata.var[groups])  # check this
            elif isarray(groups):
                grps = np.array(groups)
            else:
                raise Exception(f"The group information {groups} you provided is not in your adata object.")
            arr_dict = {}
            for g in np.unique(grps):
                arr_dict[g] = fcn_pool(arr[grps == g])
        else:
            arr_dict = {"all": fcn_pool(arr)}
    else:
        arr_dict = {"all": arr}

    ret_dict = {}
    cell_names = np.array(adata.obs_names)
    for g, arr in arr_dict.items():
        if ismatrix(arr):
            arr = arr.A.flatten()
        glst, sarr = list_top_genes(arr, cell_names, None, return_sorted_array=True)
        # ret_dict[g] = {glst[i]: sarr[i] for i in range(len(glst))}
        ret_dict[g] = glst
        if output_values:
            ret_dict[g + "_values"] = sarr
    return pd.DataFrame(data=ret_dict)


def rank_expression_genes(adata: AnnData, ekey: str = "M_s", prefix_store: str = "rank", **kwargs) -> AnnData:
    """Rank genes based on their expression values for each cell group.

    Args:
        adata: AnnData object that contains the normalized or locally smoothed expression.
        ekey: The expression key, can be any properly normalized layers, e.g. M_s, M_u, M_t, M_n.
        prefix_store: The prefix added to the key for storing the returned in adata.
        kwargs:
            additional keys that will be passed to the `rank_genes` function. It will accept the following arguments:
            group: str or None (default: None)
                The cell group that speed ranking will be grouped-by.
            genes: list or None (default: None)
                The gene list that speed will be ranked. If provided, they must overlap the dynamics genes.
            abs: bool (default: False)
                When pooling the values in the array (see below), whether to take the absolute values.
            normalize: bool (default: False)
                Whether normalize the array across all cells first, if the array is 2d.
            fcn_pool: callable (default: numpy.mean(x, axis=0))
                The function used to pool values in the to-be-ranked array if the array is 2d.
            output_values: bool (default: False)
                Whether output the values along with the rankings.

    Returns:
        AnnData object which has the rank dictionary for expression in `.uns`.
    """

    rdict = rank_genes(adata, ekey, **kwargs)
    adata.uns[prefix_store + "_" + ekey] = rdict
    return adata


def rank_velocity_genes(adata, vkey="velocity_S", prefix_store="rank", **kwargs) -> AnnData:
    """Rank genes based on their raw and absolute velocities for each cell group.

    Args:
        adata: AnnData object that contains the gene-wise velocities.
        vkey: The velocity key.
        prefix_store: The prefix added to the key for storing the returned in adata.
        kwargs:
            additional keys that will be passed to the `rank_genes` function. It will accept the following arguments:
            group: str or None (default: None)
                The cell group that speed ranking will be grouped-by.
            genes: list or None (default: None)
                The gene list that speed will be ranked. If provided, they must overlap the dynamics genes.
            abs: bool (default: False)
                When pooling the values in the array (see below), whether to take the absolute values.
            normalize: bool (default: False)
                Whether normalize the array across all cells first, if the array is 2d.
            fcn_pool: callable (default: numpy.mean(x, axis=0))
                The function used to pool values in the to-be-ranked array if the array is 2d.
            output_values: bool (default: False)
                Whether output the values along with the rankings.

    Returns:
       AnnData object which has the rank dictionary for velocities in `.uns`.
    """
    rdict = rank_genes(adata, vkey, **kwargs)
    rdict_abs = rank_genes(adata, vkey, abs=True, **kwargs)
    adata.uns[prefix_store + "_" + vkey] = rdict
    adata.uns[prefix_store + "_abs_" + vkey] = rdict_abs
    return adata


def rank_divergence_genes(
    adata: AnnData,
    jkey: str = "jacobian_pca",
    genes: Optional[List] = None,
    prefix_store: str = "rank_div_gene",
    **kwargs,
) -> pd.DataFrame:
    """Rank genes based on their diagonal Jacobian for each cell group.
        Be aware that this 'divergence' refers to the diagonal elements of a gene-wise
        Jacobian, rather than its trace, which is the common definition of the divergence.

        Run .vf.jacobian and set store_in_adata=True before using this function.

    Args:
        adata: AnnData object that contains the reconstructed vector field in the `.uns` attribute.
        jkey: The key in .uns of the cell-wise Jacobian matrix.
        genes: A list of names for genes of interest.
        prefix_store: The prefix added to the key for storing the returned ranking info in adata.
        kwargs:
            additional keys that will be passed to the `rank_genes` function. It will accept the following arguments:
            group: str or None (default: None)
                The cell group that speed ranking will be grouped-by.
            genes: list or None (default: None)
                The gene list that speed will be ranked. If provided, they must overlap the dynamics genes.
            abs: bool (default: False)
                When pooling the values in the array (see below), whether to take the absolute values.
            normalize: bool (default: False)
                Whether normalize the array across all cells first, if the array is 2d.
            fcn_pool: callable (default: numpy.mean(x, axis=0))
                The function used to pool values in the to-be-ranked array if the array is 2d.
            output_values: bool (default: False)
                Whether output the values along with the rankings.

    Returns:
        AnnData object which has the rank dictionary for diagonal jacobians in `.uns`.
    """

    if jkey not in adata.uns_keys():
        raise Exception(f"The provided dictionary key {jkey} is not in .uns.")

    reg = [x for x in adata.uns[jkey]["regulators"]]
    eff = [x for x in adata.uns[jkey]["effectors"]]
    if reg != eff:
        raise Exception("The Jacobian should have the same regulators and effectors.")
    else:
        Genes = adata.uns[jkey]["regulators"]
    cell_idx = adata.uns[jkey]["cell_idx"]
    div = np.einsum("iij->ji", adata.uns[jkey]["jacobian_gene"])
    Div = create_layer(adata, div, genes=Genes, cells=cell_idx, dtype=np.float32)

    if genes is not None:
        Genes = list(set(Genes).intersection(genes))

    rdict = rank_genes(
        adata,
        Div,
        fcn_pool=lambda x: np.nanmean(x, axis=0),
        genes=Genes,
        **kwargs,
    )
    adata.uns[prefix_store + "_" + jkey] = rdict
    return rdict


def rank_s_divergence_genes(
    adata: AnnData,
    skey: str = "sensitivity_pca",
    genes: Optional[List] = None,
    prefix_store: str = "rank_s_div_gene",
    **kwargs,
) -> pd.DataFrame:
    """Rank genes based on their diagonal Sensitivity for each cell group.
        Be aware that this 'divergence' refers to the diagonal elements of a gene-wise
        Sensitivity, rather than its trace, which is the common definition of the divergence.

        Run .vf.sensitivity and set store_in_adata=True before using this function.

    Args:
        adata: AnnData object that contains the reconstructed vector field in the `.uns` attribute.
        skey: The key in .uns of the cell-wise sensitivity matrix.
        genes: A list of names for genes of interest.
        prefix_store: The prefix added to the key for storing the returned ranking info in adata.
        kwargs:
            additional keys that will be passed to the `rank_genes` function. It will accept the following arguments:
            group: str or None (default: None)
                The cell group that speed ranking will be grouped-by.
            genes: list or None (default: None)
                The gene list that speed will be ranked. If provided, they must overlap the dynamics genes.
            abs: bool (default: False)
                When pooling the values in the array (see below), whether to take the absolute values.
            normalize: bool (default: False)
                Whether normalize the array across all cells first, if the array is 2d.
            fcn_pool: callable (default: numpy.mean(x, axis=0))
                The function used to pool values in the to-be-ranked array if the array is 2d.
            output_values: bool (default: False)
                Whether output the values along with the rankings.

    Returns:
        adata: AnnData object which has the rank dictionary for diagonal sensitivity in `.uns`.
    """

    if skey not in adata.uns_keys():
        raise Exception(f"The provided dictionary key {skey} is not in .uns.")

    reg = [x for x in adata.uns[skey]["regulators"]]
    eff = [x for x in adata.uns[skey]["effectors"]]
    if reg != eff:
        raise Exception("The Jacobian should have the same regulators and effectors.")
    else:
        Genes = adata.uns[skey]["regulators"]
    cell_idx = adata.uns[skey]["cell_idx"]
    div = np.einsum("iij->ji", adata.uns[skey]["sensitivity_gene"])
    Div = create_layer(adata, div, genes=Genes, cells=cell_idx, dtype=np.float32)

    if genes is not None:
        Genes = list(set(Genes).intersection(genes))

    rdict = rank_genes(
        adata,
        Div,
        fcn_pool=lambda x: np.nanmean(x, axis=0),
        genes=Genes,
        **kwargs,
    )
    adata.uns[prefix_store + "_" + skey] = rdict
    return rdict


def rank_acceleration_genes(adata, akey="acceleration", prefix_store="rank", **kwargs) -> AnnData:
    """Rank genes based on their absolute, positive, negative accelerations for each cell group.

    Args:
        adata: AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        akey: The acceleration key.
        prefix_store: The prefix of the key that will be used to store the acceleration rank result.
        kwargs:
            additional keys that will be passed to the `rank_genes` function. It will accept the following arguments:
            group: str or None (default: None)
                The cell group that speed ranking will be grouped-by.
            genes: list or None (default: None)
                The gene list that speed will be ranked. If provided, they must overlap the dynamics genes.
            abs: bool (default: False)
                When pooling the values in the array (see below), whether to take the absolute values.
            normalize: bool (default: False)
                Whether normalize the array across all cells first, if the array is 2d.
            fcn_pool: callable (default: numpy.mean(x, axis=0))
                The function used to pool values in the to-be-ranked array if the array is 2d.
            output_values: bool (default: False)
                Whether output the values along with the rankings.

    Returns:
        adata: AnnData object that is updated with the `'rank_acceleration'` information in the `.uns`.
    """

    rdict = rank_genes(adata, akey, **kwargs)
    rdict_abs = rank_genes(adata, akey, abs=True, **kwargs)
    adata.uns[prefix_store + "_" + akey] = rdict
    adata.uns[prefix_store + "_abs_" + akey] = rdict_abs
    return adata


def rank_curvature_genes(adata: AnnData, ckey: str = "curvature", prefix_store: str = "rank", **kwargs):
    """Rank gene's absolute, positive, negative curvature by different cell groups.

    Args:
        adata: AnnData object that contains the reconstructed vector field function in the `.uns` attribute.
        ckey: The curvature key.
        prefix_store: The prefix of the key that will be used to store the acceleration rank result.
        kwargs:
            additional keys that will be passed to the `rank_genes` function. It will accept the following arguments:
            group: str or None (default: None)
                The cell group that speed ranking will be grouped-by.
            genes: list or None (default: None)
                The gene list that speed will be ranked. If provided, they must overlap the dynamics genes.
            abs: bool (default: False)
                When pooling the values in the array (see below), whether to take the absolute values.
            normalize: bool (default: False)
                Whether normalize the array across all cells first, if the array is 2d.
            fcn_pool: callable (default: numpy.mean(x, axis=0))
                The function used to pool values in the to-be-ranked array if the array is 2d.
            output_values: bool (default: False)
                Whether output the values along with the rankings.

    Returns:
        AnnData object that is updated with the `'rank_curvature'` related information in the .uns.
    """
    rdict = rank_genes(adata, ckey, **kwargs)
    rdict_abs = rank_genes(adata, ckey, abs=True, **kwargs)
    adata.uns[prefix_store + "_" + ckey] = rdict
    adata.uns[prefix_store + "_abs_" + ckey] = rdict_abs
    return adata


def rank_jacobian_genes(
    adata: AnnData,
    groups: Optional[str] = None,
    jkey: str = "jacobian_pca",
    abs: bool = False,
    mode: str = "full reg",
    exclude_diagonal: bool = False,
    normalize: bool = False,
    return_df: bool = False,
    **kwargs,
) -> Optional[pd.DataFrame]:
    """Rank genes or gene-gene interactions based on their Jacobian elements for each cell group.

        Run .vf.jacobian and set store_in_adata=True before using this function.

    Args:
        adata: AnnData object that contains the reconstructed vector field in the `.uns` attribute.
        groups: Cell groups used to group the Jacobians.
        jkey: The key of the stored Jacobians in `.uns`.
        abs: Whether take the absolute value of the Jacobian.
        mode: {'full reg', 'full eff', 'reg', 'eff', 'int', 'switch'} (default: 'full_reg')
            The mode of ranking:
            (1) `'full reg'`: top regulators are ranked for each effector for each cell group;
            (2) `'full eff'`: top effectors are ranked for each regulator for each cell group;
            (3) '`reg`': top regulators in each cell group;
            (4) '`eff`': top effectors in each cell group;
            (5) '`int`': top effector-regulator pairs in each cell group.
            (6) '`switch`': top effector-regulator pairs that show mutual inhibition pattern in each cell group.
        exclude_diagonal: Whether to consider the self-regulation interactions (diagnoal of the jacobian matrix)
        normalize: Whether to normalize the Jacobian across all cells before performing the ranking.
        return_df: Whether to return the data or to save results in adata object via the key `mode` of adata.uns.
        kwargs: Keyword arguments passed to ranking functions.

    Returns:
        rank_info:
            different modes return different types of return values
            1. full reg and full eff:
                A pandas dataframe containing ranking info based on Jacobian elements
            2. reg eff int:
                A dictionary object whose keys correspond to groups, and whose values are
                specific rank's pd dataframe
    """
    J_dict = adata.uns[jkey]
    J = J_dict["jacobian_gene"]
    if abs:
        J = np.abs(J)

    if normalize:
        Jmax = np.max(np.abs(J), axis=2)
        for i in range(J.shape[2]):
            J[:, :, i] /= Jmax

    if mode == "switch":
        J_transpose = J.transpose(1, 0, 2)
        J_mul = J * J_transpose
        # switch genes will have negative Jacobian between any two gene pairs
        # only True * True = 1, so only the gene pair with both negative Jacobian, this will be non-zero:
        J = J_mul * (np.sign(J) == -1) * (np.sign(J_transpose) == -1)

    if groups is None:
        J_mean = {"all": np.mean(J, axis=2)}
    else:
        if type(groups) is str and groups in adata.obs.keys():
            grps = np.array(adata.obs[groups])
        elif isarray(groups):
            grps = np.array(groups)
        else:
            raise Exception(f"The group information {groups} you provided is not in your adata object.")
        J_mean = average_jacobian_by_group(J, grps[J_dict["cell_idx"]])

    eff = np.array([x for x in J_dict["effectors"]])
    reg = np.array([x for x in J_dict["regulators"]])
    rank_dict = {}
    ov = kwargs.pop("output_values", True)
    if mode in ["full reg", "full_reg"]:
        for k, J in J_mean.items():
            rank_dict[k] = table_top_genes(J, eff, reg, n_top_genes=None, output_values=ov, **kwargs)
    elif mode in ["full eff", "full_eff"]:
        for k, J in J_mean.items():
            rank_dict[k] = table_top_genes(J.T, reg, eff, n_top_genes=None, output_values=ov, **kwargs)
    elif mode == "reg":
        for k, J in J_mean.items():
            if exclude_diagonal:
                for i, ef in enumerate(eff):
                    ii = np.where(reg == ef)[0]
                    if len(ii) > 0:
                        J[i, ii] = np.nan
            j = np.nanmean(J, axis=0)
            if ov:
                rank_dict[k], rank_dict[k + "_values"] = list_top_genes(
                    j, reg, None, return_sorted_array=True, **kwargs
                )
            else:
                rank_dict[k] = list_top_genes(j, reg, None, **kwargs)
        rank_dict = pd.DataFrame(data=rank_dict)
    elif mode == "eff":
        for k, J in J_mean.items():
            if exclude_diagonal:
                for i, re in enumerate(reg):
                    ii = np.where(eff == re)[0]
                    if len(ii) > 0:
                        J[ii, i] = np.nan
            j = np.nanmean(J, axis=1)
            if ov:
                rank_dict[k], rank_dict[k + "_values"] = list_top_genes(
                    j, eff, None, return_sorted_array=True, **kwargs
                )
            else:
                rank_dict[k] = list_top_genes(j, eff, None, **kwargs)
        rank_dict = pd.DataFrame(data=rank_dict)
    elif mode in ["int", "switch"]:
        for k, J in J_mean.items():
            ints, vals = list_top_interactions(J, eff, reg, **kwargs)
            rank_dict[k] = []
            if ov:
                rank_dict[k + "_values"] = []
            for ind, int_val in enumerate(ints):
                if not (exclude_diagonal and int_val[0] == int_val[1]):
                    rank_dict[k].append(int_val[0] + " - " + int_val[1])
                    if ov:
                        rank_dict[k + "_values"].append(vals[ind])
        rank_dict = pd.DataFrame(data=rank_dict)
    else:
        raise ValueError(f"No such mode as {mode}.")

    if return_df:
        return rank_dict
    else:
        main_info_insert_adata_uns(mode)
        adata.uns[mode] = rank_dict


def rank_sensitivity_genes(
    adata: AnnData,
    groups: Optional[str] = None,
    skey: str = "sensitivity_pca",
    abs: bool = False,
    mode: str = "full reg",
    exclude_diagonal: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Rank genes or gene-gene interactions based on their sensitivity elements for each cell group.

        Run .vf.sensitivity and set store_in_adata=True before using this function.

    Args:
        adata: AnnData object that contains the reconstructed vector field in the `.uns` attribute.
        groups: Cell groups used to group the sensitivity.
        skey: The key of the stored sensitivity in `.uns`.
        abs: Whether or not to take the absolute value of the Jacobian.
        mode: {'full reg', 'full eff', 'reg', 'eff', 'int'} (default: 'full_reg')
            The mode of ranking:
            (1) `'full reg'`: top regulators are ranked for each effector for each cell group;
            (2) `'full eff'`: top effectors are ranked for each regulator for each cell group;
            (3) '`reg`': top regulators in each cell group;
            (4) '`eff`': top effectors in each cell group;
            (5) '`int`': top effector-regulator pairs in each cell group.
        exclude_diagonal: Whether to consider the self-regulation interactions (diagnoal of the jacobian matrix)
        kwargs: Keyword arguments passed to ranking functions.

    Returns:
        AnnData object which has the rank dictionary in `.uns`.
    """
    S_dict = adata.uns[skey]
    S = S_dict["sensitivity_gene"]
    if abs:
        S = np.abs(S)
    if groups is None:
        S_mean = {"all": np.mean(S, axis=2)}
    else:
        if type(groups) is str and groups in adata.obs.keys():
            grps = np.array(adata.obs[groups])
        elif isarray(groups):
            grps = np.array(groups)
        else:
            raise Exception(f"The group information {groups} you provided is not in your adata object.")
        S_mean = average_jacobian_by_group(S, grps[S_dict["cell_idx"]])

    eff = np.array([x for x in S_dict["effectors"]])
    reg = np.array([x for x in S_dict["regulators"]])
    rank_dict = {}
    if mode in ["full reg", "full_reg"]:
        for k, S in S_mean.items():
            rank_dict[k] = table_top_genes(S, eff, reg, n_top_genes=None, **kwargs)
    elif mode in ["full eff", "full_eff"]:
        for k, S in S_mean.items():
            rank_dict[k] = table_top_genes(S.T, reg, eff, n_top_genes=None, **kwargs)
    elif mode == "reg":
        ov = kwargs.pop("output_values", False)
        for k, S in S_mean.items():
            if exclude_diagonal:
                for i, ef in enumerate(eff):
                    ii = np.where(reg == ef)[0]
                    if len(ii) > 0:
                        S[i, ii] = np.nan
            j = np.nanmean(S, axis=0)
            if ov:
                rank_dict[k], rank_dict[k + "_values"] = list_top_genes(
                    j, reg, None, return_sorted_array=True, **kwargs
                )
            else:
                rank_dict[k] = list_top_genes(j, reg, None, **kwargs)
        rank_dict = pd.DataFrame(data=rank_dict)
    elif mode == "eff":
        ov = kwargs.pop("output_values", False)
        for k, S in S_mean.items():
            if exclude_diagonal:
                for i, re in enumerate(reg):
                    ii = np.where(eff == re)[0]
                    if len(ii) > 0:
                        S[ii, i] = np.nan
            j = np.nanmean(S, axis=1)
            if ov:
                rank_dict[k], rank_dict[k + "_values"] = list_top_genes(
                    j, eff, None, return_sorted_array=True, **kwargs
                )
            else:
                rank_dict[k] = list_top_genes(j, eff, None, **kwargs)
        rank_dict = pd.DataFrame(data=rank_dict)
    elif mode == "int":
        ov = kwargs.pop("output_values", False)
        for k, S in S_mean.items():
            ints, vals = list_top_interactions(S, eff, reg, **kwargs)
            rank_dict[k] = []
            if ov:
                rank_dict[k + "_values"] = []
            for ind, int_val in enumerate(ints):
                if not (exclude_diagonal and int_val[0] == int_val[1]):
                    rank_dict[k].append(int_val[0] + " - " + int_val[1])
                    if ov:
                        rank_dict[k + "_values"].append(vals[ind])
        rank_dict = pd.DataFrame(data=rank_dict)
    else:
        raise ValueError(f"No such mode as {mode}.")
    return rank_dict


# ---------------------------------------------------------------------------------------------------
# aggregate regulators or targets
def aggregateRegEffs(
    adata: AnnData,
    data_dict: Optional[Dict] = None,
    reg_dict: Optional[Dict] = None,
    eff_dict: Optional[Dict] = None,
    key: str = "jacobian",
    basis: str = "pca",
    store_in_adata: bool = True,
) -> Union[AnnData, Dict]:
    """Aggregate multiple genes' Jacobian or sensitivity.

    Args:
        adata: AnnData object that contains the reconstructed vector field in `.uns`.
        data_dict: A dictionary corresponds to the Jacobian or sensitivity information, must be calculated with either:
            `dyn.vf.jacobian(adata, basis='pca', regulators=genes, effectors=genes)` or
            `dyn.vf.sensitivity(adata, basis='pca', regulators=genes, effectors=genes)`
        reg_dict: A dictionary in which keys correspond to regulator-groups (i.e. TFs for specific cell type) while values
            a list of genes that must have at least one overlapped genes with that from the Jacobian or sensitivity
            dict.
        eff_dict: A dictionary in which keys correspond to effector-groups (i.e. markers for specific cell type) while values
            a list of genes that must have at least one overlapped genes with that from the Jacobian or sensitivity
            dict.
        key: The key in .uns that corresponds to the Jacobian or sensitivity matrix information.
        basis: The embedding data in which the vector field was reconstructed. If `None`, use the vector field function
            that was reconstructed directly from the original unreduced gene expression space.
        store_in_adata: hether to store the divergence result in adata.


    Returns:
        Depending on `store_in_adata`, it will either return a dictionary that include the aggregated Jacobian or
            sensitivity information or the updated AnnData object that is updated with the `'aggregation'` key in the
            `.uns`. This dictionary contains a 3-dimensional tensor with dimensions n_obs x n_regulators x n_effectors
            as well as other information.
    """

    key_ = key if basis is None else key + "_" + basis
    data_dict = adata.uns[key_] if data_dict is None else data_dict

    tensor, cell_idx, tensor_gene, regulators_, effectors_ = (
        data_dict.get(key),
        data_dict.get("cell_idx"),
        data_dict.get(key + "_gene"),
        data_dict.get("regulators"),
        data_dict.get("effectors"),
    )

    Aggregation = np.zeros((len(eff_dict), len(reg_dict), len(cell_idx)))
    reg_ind = 0
    for reg_key, reg_val in reg_dict.items():
        eff_ind = 0
        for eff_key, eff_val in eff_dict.items():
            reg_val, eff_val = (
                list(np.unique(reg_val)) if reg_val is not None else None,
                list(np.unique(eff_val)) if eff_val is not None else None,
            )

            Der, source_genes, target_genes = intersect_sources_targets(
                reg_val,
                regulators_,
                eff_val,
                effectors_,
                tensor if tensor_gene is None else tensor_gene,
            )
            if len(source_genes) + len(target_genes) > 0:
                Aggregation[eff_ind, reg_ind, :] = Der.sum(axis=(0, 1))  # dim 0: target; dim 1: source
            else:
                Aggregation[eff_ind, reg_ind, :] = np.nan
            eff_ind += 1
        reg_ind += 0

    ret_dict = {"aggregation": None, "cell_idx": cell_idx}
    # use 'str_key' in dict.keys() to check if these items are computed, or use dict.get('str_key')
    if Aggregation is not None:
        ret_dict["aggregation_gene"] = Aggregation
    if reg_dict.keys() is not None:
        ret_dict["regulators"] = list(reg_dict.keys())
    if eff_dict.keys() is not None:
        ret_dict["effectors"] = list(eff_dict.keys())

    det = [np.linalg.det(Aggregation[:, :, i]) for i in np.arange(Aggregation.shape[2])]
    key = key + "_aggregation" if basis is None else key + "_aggregation_" + basis
    adata.obs[key + "_det"] = np.nan
    adata.obs[key + "_det"][cell_idx] = det
    if store_in_adata:
        adata.uns[key] = ret_dict
        return adata
    else:
        return ret_dict
