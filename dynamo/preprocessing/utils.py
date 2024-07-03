import warnings
from typing import Iterable, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import scipy.sparse
import statsmodels.api as sm
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse, spmatrix

from ..configuration import DKM, DynamoAdataKeyManager
from ..dynamo_logger import LoggerManager, main_debug, main_info, main_warning
from ..utils import areinstance


# ---------------------------------------------------------------------------------------------------
# symbol conversion related
def convert2gene_symbol(input_names: List[str], scopes: Union[List[str], None] = "ensembl.gene") -> pd.DataFrame:
    """Convert ensemble gene id to official gene names using mygene package.

    Args:
        input_names: the ensemble gene id names that you want to convert to official gene names. All names should come
            from the same species.
        scopes: scopes are needed when you use non-official gene name as your gene indices (or adata.var_name).
            This arugument corresponds to type of types of identifiers, either a list or a comma-separated fields to
            specify type of input qterms, e.g. “entrezgene”, “entrezgene,symbol”, [“ensemblgene”, “symbol”]. Refer to
            official MyGene.info docs (https://docs.mygene.info/en/latest/doc/query_service.html#available_fields) for
            full list of fields. Defaults to "ensembl.gene".

    Raises:
        ImportError: fail to import `mygene`.

    Returns:
        A pandas dataframe that includes the following columns:
            query: the input ensmble ids
            _id: identified id from mygene
            _score: confidence of the retrieved official gene name.
            symbol: retrieved official gene name
    """

    try:
        import mygene
    except ImportError:
        raise ImportError(
            "You need to install the package `mygene` (pip install mygene --user) "
            "See https://pypi.org/project/mygene/ for more details."
        )

    mg = mygene.MyGeneInfo()
    main_info("Storing myGene name info into local cache db: mygene_cache.sqlite.")
    mg.set_caching()

    ensemble_names = [i.split(".")[0] for i in input_names]
    var_pd = mg.querymany(
        ensemble_names,
        scopes=scopes,
        fields="symbol",
        as_dataframe=True,
        df_index=True,
    )
    # var_pd.drop_duplicates(subset='query', inplace=True) # use when df_index is not True
    var_pd = var_pd.loc[~var_pd.index.duplicated(keep="first")]

    return var_pd


def convert2symbol(adata: AnnData, scopes: Union[str, Iterable, None] = None, subset=True) -> AnnData:
    """This helper function converts unofficial gene names to official gene names.

    Args:
        adata: an AnnData object.
        scopes: scopes are needed when you use non-official gene name as your gene indices (or adata.var_name).
            This arugument corresponds to type of types of identifiers, either a list or a comma-separated fields to
            specify type of input qterms, e.g. “entrezgene”, “entrezgene,symbol”, [“ensemblgene”, “symbol”]. Refer to
            official MyGene.info docs (https://docs.mygene.info/en/latest/doc/query_service.html#available_fields) for
            full list of fields. Defaults to None.
        subset: whether to inplace subset the results. Defaults to True.

    Raises:
        Exception: gene names in adata.var_names are invalid.

    Returns:
        The updated AnnData object.
    """

    if np.all(adata.var_names.str.startswith("ENS")) or scopes is not None:
        logger = LoggerManager.gen_logger("dynamo-utils")
        logger.info("convert ensemble name to official gene name", indent_level=1)

        prefix = adata.var_names[0]
        if scopes is None:
            if prefix[:4] == "ENSG" or prefix[:7] == "ENSMUSG":
                scopes = "ensembl.gene"
            elif prefix[:4] == "ENST" or prefix[:7] == "ENSMUST":
                scopes = "ensembl.transcript"
            else:
                raise Exception(
                    "Your adata object uses non-official gene names as gene index. \n"
                    "Dynamo finds those IDs are neither from ensembl.gene or ensembl.transcript and thus cannot "
                    "convert them automatically. \n"
                    "Please pass the correct scopes or first convert the ensemble ID to gene short name "
                    "(for example, using mygene package). \n"
                    "See also dyn.pp.convert2gene_symbol"
                )

        adata.var["query"] = [i.split(".")[0] for i in adata.var.index]
        if scopes is str:
            adata.var[scopes] = adata.var.index
        else:
            adata.var["scopes"] = adata.var.index

        logger.warning(
            "Your adata object uses non-official gene names as gene index. \n"
            "Dynamo is converting those names to official gene names."
        )
        official_gene_df = convert2gene_symbol(adata.var_names, scopes)
        merge_df = adata.var.merge(official_gene_df, left_on="query", right_on="query", how="left").set_index(
            adata.var.index
        )
        if "notfound" in merge_df.columns:
            valid_ind = np.where(merge_df["notfound"] != True)[0]  # noqa: E712
            merge_df.pop("notfound")
        else:
            valid_ind = np.arange(len(merge_df))

        adata.var = merge_df

        if subset is True:
            adata._inplace_subset_var(valid_ind)
            adata.var.index = adata.var["symbol"].values.copy()
        else:
            indices = np.array(adata.var.index)
            indices[valid_ind] = adata.var.loc[valid_ind, "symbol"].values.copy()
            adata.var.index = indices

        if np.sum(adata.var_names.isnull()) > 0:
            main_info(
                "Subsetting adata object and removing Nan columns from adata when converting gene names.",
                indent_level=1,
            )
            adata._inplace_subset_var(adata.var_names.notnull())
    return adata


def compute_gene_exp_fraction(X: scipy.sparse.spmatrix, threshold: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate fraction of each gene's count to total counts across cells and identify high fraction genes.

    Args:
        X: a sparse matrix containing gene expression data.
        threshold: lower bound for valid data. Defaults to 0.001.

    Returns:
        A tuple (frac, valid_ids) where frac is the fraction of each gene's count to total count and valid_ids is the
        indices of valid genes.
    """

    frac = X.sum(0) / X.sum()
    if issparse(X):
        frac = frac.A.reshape(-1, 1)

    valid_ids = np.where(frac > threshold)[0]

    return frac, valid_ids


# ---------------------------------------------------------------------------------------------------
# preprocess utilities
def _infer_labeling_experiment_type(adata: anndata.AnnData, tkey: str) -> Literal["one-shot", "kin", "deg"]:
    """Returns the experiment type of `adata` according to `tkey`s

    Args:
        adata: an AnnData Object.
        tkey: the key for time in `adata.obs`.

    Returns:
        The experiment type, must be one of "one-shot", "kin" or "deg".
    """

    experiment_type = None
    tkey_val = np.array(adata.obs[tkey], dtype="float")
    if len(np.unique(tkey_val)) == 1:
        experiment_type = "one-shot"
    else:
        labeled_frac = adata.layers["new"].T.sum(0) / adata.layers["total"].T.sum(0)
        xx = labeled_frac.A1 if issparse(adata.layers["new"]) else labeled_frac

        yy = tkey_val
        xm, ym = np.mean(xx), np.mean(yy)
        cov = np.mean(xx * yy) - xm * ym
        var_x = np.mean(xx * xx) - xm * xm

        k = cov / var_x

        # total labeled RNA amount will increase (decrease) in kinetic (degradation) experiments over time.
        experiment_type = "kin" if k > 0 else "deg"
    main_debug(
        f"\nDynamo has detected that your labeling data is from a kin experiment. \nIf the experiment type is incorrect, "
        f"please provide the correct experiment_type (one-shot, kin, or deg)."
    )
    return experiment_type


def unique_var_obs_adata(adata: anndata.AnnData) -> anndata.AnnData:
    """Function to make the obs and var attribute's index unique

    Args:
        adata: an AnnData object.

    Returns:
        The updated annData object.
    """

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    return adata


def convert_layers2csr(adata: anndata.AnnData) -> anndata.AnnData:
    """Function to convert a layer of sparse matrix to compressed csr_matrix.

    Args:
        adata: an AnnData object.

    Returns:
        The updated annData object.
    """

    for key in adata.layers.keys():
        adata.layers[key] = csr_matrix(adata.layers[key]) if not issparse(adata.layers[key]) else adata.layers[key]

    return adata


def merge_adata_attrs(adata_ori: AnnData, adata: AnnData, attr: Literal["var", "obs"]) -> AnnData:
    """Merge two adata objects.

    Args:
        adata_ori: an AnnData object to be merged into.
        adata: the AnnData object to be merged.
        attr: the attribution of adata to be merged, either "var" or "obs".

    Returns:
        The merged AnnData object.
    """

    def _merge_by_diff(origin_df: pd.DataFrame, diff_df: pd.DataFrame) -> pd.DataFrame:
        """Merge two DatafFames.

        Args:
            origin_df: the DataFrame to be merged into.
            diff_df: the DataFrame to be merged.

        Returns:
            The merged DataFrame.
        """

        _columns = list(set(diff_df.columns).difference(origin_df.columns))
        new_df = origin_df.merge(diff_df[_columns], how="left", left_index=True, right_index=True)
        return new_df.loc[origin_df.index, :]

    if attr == "var":
        adata_ori.var = _merge_by_diff(adata_ori.var, adata.var)
    elif attr == "obs":
        obs_df = _merge_by_diff(adata_ori.obs, adata.obs)
        if obs_df.shape[0] > adata_ori.n_obs:
            raise ValueError(
                "Left join generates more rows. Please check if you obs names are unique before calling this fucntion."
            )
        adata_ori.obs = obs_df
    return adata_ori


def clip_by_perc(layer_mat):
    """Returns a new matrix by clipping the layer_mat according to percentage."""
    # TODO implement this function (currently not used)
    return


def get_inrange_shared_counts_mask(
    adata: anndata.AnnData, layers: List[str], min_shared_count: int, count_by: Literal["gene", "cells"] = "gene"
) -> np.ndarray:
    """Generate the mask showing the genes having counts more than the provided minimal count.

    Args:
        adata: an AnnData object.
        layers: the layers to be operated on.
        min_shared_count: the minimal shared number of counts for each genes across cell between layers.
        count_by: the count type of the data, either "gene: or "cells". Defaults to "gene".

    Raises:
        ValueError: invalid count type.

    Returns:
        The result mask showing the genes having counts more than the provided minimal count.
    """

    layers = list(set(layers).difference(["X", "matrix", "ambiguous", "spanning"]))
    # choose shared counts sum by row or columns based on type: `gene` or `cells`
    sum_dim_index = None
    ret_dim_index = None
    if count_by == "gene":
        sum_dim_index = 0
        ret_dim_index = 1
    elif count_by == "cells":
        sum_dim_index = 1
        ret_dim_index = 0
    else:
        raise ValueError("Not supported shared account type")

    if len(np.array(layers)) == 0:
        main_warning("No layers exist in adata, skipp filtering by shared counts")
        return np.repeat(True, adata.shape[ret_dim_index])

    layers = np.array(layers)[~pd.DataFrame(layers)[0].str.startswith("X_").values]

    _nonzeros, _sum = None, None

    # TODO fix bug: when some layers are sparse and some others are not (mixed sparse and ndarray), if the first one happens to be sparse,
    # dimension mismatch error will be raised; if the first layer (layers[0]) is not sparse, then the following loop works fine.
    # also check if layers2csr() function works
    for layer in layers:
        main_debug(adata.layers[layer].shape)
        main_debug("layer: %s" % layer)
        if issparse(adata.layers[layers[0]]):
            main_debug("when sparse, layer type:" + str(type(adata.layers[layer])))
            _nonzeros = adata.layers[layer] > 0 if _nonzeros is None else _nonzeros.multiply(adata.layers[layer] > 0)
        else:
            main_debug("when not sparse, layer type:" + str(type(adata.layers[layer])))

            _nonzeros = adata.layers[layer] > 0 if _nonzeros is None else _nonzeros * (adata.layers[layer] > 0)

    for layer in layers:
        if issparse(adata.layers[layers[0]]):
            _sum = (
                _nonzeros.multiply(adata.layers[layer])
                if _sum is None
                else _sum + _nonzeros.multiply(adata.layers[layer])
            )
        else:
            _sum = (
                np.multiply(_nonzeros, adata.layers[layer])
                if _sum is None
                else _sum + np.multiply(_nonzeros, adata.layers[layer])
            )

    return (
        np.array(_sum.sum(sum_dim_index).A1 >= min_shared_count)
        if issparse(adata.layers[layers[0]])
        else np.array(_sum.sum(sum_dim_index) >= min_shared_count)
    )


def get_nan_or_inf_data_bool_mask(arr: np.ndarray) -> np.ndarray:
    """Returns the mask of arr with the same shape, indicating whether each index is nan/inf or not.

    Args:
        arr: an array

    Returns:
        A bool array indicating each element is nan/inf or not
    """

    mask = np.isnan(arr) | np.isinf(arr) | np.isneginf(arr)
    return mask


def get_gene_selection_filter(
    valid_table: pd.Series,
    n_top_genes: int = 2000,
    basic_filter: Optional[pd.Series] = None,
) -> np.ndarray:
    """Generate the mask by sorting given table of scores.

    Args:
        valid_table: the scores used to sort the highly variable genes.
        n_top_genes: number of top genes to be filtered. Defaults to 2000.
        basic_filter: the filter to remove outliers. For example, the `adata.var["pass_basic_filter"]`.

    Returns:
        The filter mask as a bool array.
    """
    if basic_filter is None:
        basic_filter = pd.Series(True, index=valid_table.index)
    feature_gene_idx = np.argsort(-valid_table)[:n_top_genes]
    feature_gene_idx = valid_table.index[feature_gene_idx]
    return basic_filter.index.isin(feature_gene_idx)


def get_svr_filter(
    adata: anndata.AnnData, layer: str = "spliced", n_top_genes: int = 3000, return_adata: bool = False
) -> Union[anndata.AnnData, np.ndarray]:
    """Generate the mask showing the genes with valid svr scores.

    Args:
        adata: an AnnData object.
        layer: the layer to operate on. Defaults to "spliced".
        n_top_genes: number of top genes to be filtered. Defaults to 3000.
        return_adata: whether return an updated AnnData or the mask as an array. Defaults to False.

    Returns:
        adata: updated adata object with the mask.
        filter_bool: the filter mask as a bool array.
    """

    score_name = "score" if layer in ["X", "all"] else layer + "_score"
    valid_idx = np.where(np.isfinite(adata.var.loc[:, score_name]))[0]

    valid_table = adata.var.iloc[valid_idx, :]
    nth_score = np.sort(valid_table.loc[:, score_name])[::-1][np.min((n_top_genes - 1, valid_table.shape[0] - 1))]

    feature_gene_idx = np.where(valid_table.loc[:, score_name] >= nth_score)[0][:n_top_genes]
    feature_gene_idx = valid_idx[feature_gene_idx]

    if return_adata:
        adata.var.loc[:, "use_for_pca"] = False
        adata.var.loc[adata.var.index[feature_gene_idx], "use_for_pca"] = True
        res = adata
    else:
        filter_bool = np.zeros(adata.n_vars, dtype=bool)
        filter_bool[feature_gene_idx] = True
        res = filter_bool

    return res


def seurat_get_mean_var(
    X: Union[csr_matrix, np.ndarray],
    ignore_zeros: bool = False,
    perc: Union[float, List[float], None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Only used in seurat impl to match seurat and scvelo implementation result.

    Args:
        X: a matrix as np.ndarray or a sparse matrix as scipy sparse matrix. Rows are cells while columns are genes.
        ignore_zeros: whether ignore columns with 0 only. Defaults to False.
        perc: clip the gene expression values based on the perc or the min/max boundary of the values. Defaults to None.

    Returns:
        A tuple (mean, var) where mean is the mean of the columns after processing of the matrix and var is the variance
        of the columns after processing of the matrix.
    """

    data = X.data if issparse(X) else X
    mask_nans = np.isnan(data) | np.isinf(data) | np.isneginf(data)

    n_nonzeros = (X != 0).sum(0)
    n_counts = n_nonzeros if ignore_zeros else X.shape[0]

    if mask_nans.sum() > 0:
        if issparse(X):
            data[np.isnan(data) | np.isinf(data) | np.isneginf(data)] = 0
            n_nans = n_nonzeros - (X != 0).sum(0)
        else:
            X[mask_nans] = 0
            n_nans = mask_nans.sum(0)
        n_counts -= n_nans

    if perc is not None:
        if np.size(perc) < 2:
            perc = [perc, 100] if perc < 50 else [0, perc]
        lb, ub = np.percentile(data, perc)
        data = np.clip(data, lb, ub)

    if issparse(X):
        mean = (X.sum(0) / n_counts).A1
        mean_sq = (X.multiply(X).sum(0) / n_counts).A1
    else:
        mean = X.sum(0) / n_counts
        mean_sq = np.multiply(X, X).sum(0) / n_counts
    n_cells = np.clip(X.shape[0], 2, None)  # to avoid division by zero
    var = (mean_sq - mean**2) * (n_cells / (n_cells - 1))

    mean = np.nan_to_num(mean)
    var = np.nan_to_num(var)
    return mean, var


def anndata_bytestring_decode(adata_item: pd.DataFrame) -> None:
    """Decode contents of an annotation of an AnnData object inplace.

    Args:
        adata_item: an annotation of an AnnData object.
    """

    for key in adata_item.keys():
        df = adata_item[key]
        if df.dtype.name == "category" and areinstance(df.cat.categories, bytes):
            cat = [c.decode() for c in df.cat.categories]
            df.cat.rename_categories(cat, inplace=True)


def decode_index(adata_item: pd.DataFrame) -> None:
    """Decode indices of an annotation of an AnnData object inplace.

    Args:
        adata_item: an annotation of an AnnData object.
    """

    if areinstance(adata_item.index, bytes):
        index = {i: i.decode() for i in adata_item.index}
        adata_item.rename(index, inplace=True)


def decode(adata: anndata.AnnData) -> None:
    """Decode an AnnData object.

    Args:
        adata: an AnnData object.
    """

    decode_index(adata.obs)
    decode_index(adata.var)
    anndata_bytestring_decode(adata.obs)
    anndata_bytestring_decode(adata.var)


def add_noise_to_duplicates(adata: anndata.AnnData, basis: str = "pca") -> None:
    """Add noise to duplicated elements of the reduced array inplace.

    Args:
        adata: an AnnData object.
        basis: the type of dimension redduction. Defaults to "pca".
    """

    X_data = adata.obsm["X_" + basis]
    min_val = abs(X_data).min()

    n_obs, n_var = X_data.shape
    while True:
        _, index = np.unique(X_data, axis=0, return_index=True)
        duplicated_idx = np.setdiff1d(np.arange(n_obs), index)

        if len(duplicated_idx) == 0:
            adata.obsm["X_" + basis] = X_data
            break
        else:
            X_data[duplicated_idx, :] += np.random.normal(0, min_val / 1000, (len(duplicated_idx), n_var))


def is_float_integer_arr(arr: Union[np.ndarray, spmatrix, list]) -> bool:
    """Test if an array's elements are integers

    Args:
        arr: an input array.

    Returns:
        A flag whether all elements of the array are integers.
    """

    if issparse(arr):
        arr = arr.data
    return np.all(np.equal(np.mod(arr, 1), 0))


def is_integer_arr(arr: Union[np.ndarray, spmatrix, list]) -> bool:
    """Test if an array like obj's dtype is integer

    Args:
        arr: an array like object.

    Returns:
        A flag whether the array's dtype is integer.
    """

    return np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, int)


def is_nonnegative(mat: Union[np.ndarray, spmatrix, list]) -> bool:
    """Test whether all elements of an array or sparse array are non-negative.

    Args:
        mat: an array in ndarray or sparse array in scipy spmatrix.

    Returns:
        A flag whether all elements are non-negative.
    """

    if scipy.sparse.issparse(mat):
        return np.all(mat.sign().data >= 0)
    return np.all(np.sign(mat) >= 0)


def is_nonnegative_integer_arr(mat: Union[np.ndarray, spmatrix, list]) -> bool:
    """Test if an array's elements are non-negative integers

    Args:
        mat: an input array.

    Returns:
        A flag whether all elements of the array are non-negative integers.
    """

    if (not is_integer_arr(mat)) and (not is_float_integer_arr(mat)):
        return False
    return is_nonnegative(mat)


# ---------------------------------------------------------------------------------------------------
# labeling related


def collapse_species_adata(adata: anndata.AnnData) -> None:
    """Function to collapse the four species data, will be generalized to handle dual-datasets.

    Args:
        adata: an AnnData object.
    """

    (
        only_splicing,
        only_labeling,
        splicing_and_labeling,
    ) = DKM.allowed_layer_raw_names()

    if np.all([name in adata.layers.keys() for name in splicing_and_labeling]):
        if only_splicing[0] not in adata.layers.keys():
            adata.layers[only_splicing[0]] = adata.layers["su"] + adata.layers["sl"]
        if only_splicing[1] not in adata.layers.keys():
            adata.layers[only_splicing[1]] = adata.layers["uu"] + adata.layers["ul"]
        if only_labeling[0] not in adata.layers.keys():
            adata.layers[only_labeling[0]] = adata.layers["ul"] + adata.layers["sl"]
        if only_labeling[1] not in adata.layers.keys():
            adata.layers[only_labeling[1]] = adata.layers[only_labeling[0]] + adata.layers["uu"] + adata.layers["su"]

    return adata


def detect_experiment_datatype(adata: anndata.AnnData) -> Tuple[bool, bool, bool, bool]:
    """Tells what kinds of experiment data are stored in an AnnData object.

    Args:
        adata: an AnnData object.

    Returns:
        A tuple (has_splicing, has_labeling, splicing_labeling, has_protein), where has_splicing represents whether the
        object containing unspliced and spliced data, has_labeling represents whether the object containing new
        expression and total expression (i.e. labelling) data, splicing_labeling represents whether the object
        containing both splicing and labelling data, and has_protein represents whether the object containing protein
        data.
    """

    has_splicing, has_labeling, splicing_labeling, has_protein = (
        False,
        False,
        False,
        False,
    )

    layers = adata.layers.keys()
    if (
        len({"ul", "sl", "uu", "su"}.difference(layers)) == 0
        or len({"X_ul", "X_sl", "X_uu", "X_su"}.difference(layers)) == 0
    ):
        has_splicing, has_labeling, splicing_labeling = True, True, True
    elif (
        len({"unspliced", "spliced", "new", "total"}.difference(layers)) == 0
        or len({"X_unspliced", "X_spliced", "X_new", "X_total"}.difference(layers)) == 0
    ):
        has_splicing, has_labeling = True, True
    elif (
        len({"unspliced", "spliced"}.difference(layers)) == 0
        or len({"X_unspliced", "X_spliced"}.difference(layers)) == 0
    ):
        has_splicing = True
    elif len({"new", "total"}.difference(layers)) == 0 or len({"X_new", "X_total"}.difference(layers)) == 0:
        has_labeling = True

    if "protein" in adata.obsm.keys():
        has_protein = True

    return has_splicing, has_labeling, splicing_labeling, has_protein


def default_layer(adata: anndata.AnnData) -> str:
    """Returns the defualt layer preferred in a given AnnData object.

    Args:
        adata: an AnnData object.

    Returns:
        The key of the default layer.
    """

    has_splicing, has_labeling, splicing_labeling, _ = detect_experiment_datatype(adata)

    if has_splicing:
        if has_labeling:
            if len(set(adata.layers.keys()).intersection(["new", "total", "spliced", "unspliced"])) == 4:
                adata = collapse_species_adata(adata)
            default_layer = (
                "M_t" if "M_t" in adata.layers.keys() else "X_total" if "X_total" in adata.layers.keys() else "total"
            )
        else:
            default_layer = (
                "M_s"
                if "M_s" in adata.layers.keys()
                else "X_spliced"
                if "X_spliced" in adata.layers.keys()
                else "spliced"
            )
    else:
        default_layer = (
            "M_t" if "M_t" in adata.layers.keys() else "X_total" if "X_total" in adata.layers.keys() else "total"
        )

    return default_layer


def calc_new_to_total_ratio(adata: anndata.AnnData) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
    """Calculate the new to total ratio across cells. Note that NTR for the first time point in degradation approximates gamma/beta.

    Args:
        adata: an AnnData object.

    Returns:
        ntr: the new to total ratio of all genes for each cell. Returned if the object has labelling or splicing layers.
        var_ntr: the new to total ratio of all cells for each gene. Returned if the object has labelling or splicing
            layers.
    """

    if len({"new", "total"}.intersection(adata.layers.keys())) == 2:
        ntr = adata.layers["new"].sum(1) / adata.layers["total"].sum(1)
        ntr = ntr.A1 if issparse(adata.layers["new"]) else ntr

        var_ntr = adata.layers["new"].sum(0) / adata.layers["total"].sum(0)
        var_ntr = var_ntr.A1 if issparse(adata.layers["new"]) else var_ntr
    elif len({"uu", "ul", "su", "sl"}.intersection(adata.layers.keys())) == 4:
        new = adata.layers["ul"].sum(1) + adata.layers["sl"].sum(1)
        total = new + adata.layers["uu"].sum(1) + adata.layers["su"].sum(1)
        ntr = new / total

        ntr = ntr.A1 if issparse(adata.layers["uu"]) else ntr

        new = adata.layers["ul"].sum(0) + adata.layers["sl"].sum(0)
        total = new + adata.layers["uu"].sum(0) + adata.layers["su"].sum(0)
        var_ntr = new / total

        var_ntr = var_ntr.A1 if issparse(adata.layers["uu"]) else var_ntr
    elif len({"unspliced", "spliced"}.intersection(adata.layers.keys())) == 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            ntr = adata.layers["unspliced"].sum(1) / (adata.layers["unspliced"] + adata.layers["spliced"]).sum(1)
            var_ntr = adata.layers["unspliced"].sum(0) / (adata.layers["unspliced"] + adata.layers["spliced"]).sum(0)

        ntr = ntr.A1 if issparse(adata.layers["unspliced"]) else ntr
        var_ntr = var_ntr.A1 if issparse(adata.layers["unspliced"]) else var_ntr
    else:
        ntr, var_ntr = None, None

    return ntr, var_ntr


def scale(
    adata: AnnData,
    layers: Union[List[str], str, None] = None,
    scale_to_layer: Optional[str] = None,
    scale_to: float = 1e4,
) -> anndata.AnnData:
    """Scale layers to a particular total expression value, similar to `normalize_expr_data` function.

    Args:
        adata: an AnnData object.
        layers: the layers to scale. Defaults to None.
        scale_to_layer: use which layer to calculate a global scale factor. If None, calculate each layer's own scale
            factor and scale all layers to same total value. Defaults to None.
        scale_to: the total expression value that layers are scaled to. Defaults to 1e6.

    Returns:
        The scaled AnnData object.
    """

    if layers is None:
        layers = DynamoAdataKeyManager.get_available_layer_keys(adata, layers="all")
    has_splicing, has_labeling = detect_experiment_datatype(adata)[:2]

    if scale_to_layer is None:
        scale_to_layer = "total" if has_labeling else None
        scale = scale_to / adata.layers[scale_to_layer].sum(1)
    else:
        scale = None

    for layer in layers:
        # if scale is None:
        scale = scale_to / adata.layers[layer].sum(1)
        adata.layers[layer] = csr_matrix(adata.layers[layer].multiply(scale))

    return adata


# ---------------------------------------------------------------------------------------------------
# ERCC related


def relative2abs(
    adata: anndata.AnnData,
    dilution: float,
    volume: float,
    from_layer: Union[str, None] = None,
    to_layers: Union[str, List[str], None] = None,
    mixture_type: Literal[1, 2] = 1,
    ERCC_controls: Union[np.ndarray, None] = None,
    ERCC_annotation: Union[pd.DataFrame, None] = None,
) -> anndata.AnnData:
    """Converts FPKM/TPM data to transcript counts using ERCC spike-in.

    This is based on the relative2abs function from monocle 2 (Qiu, et. al, Nature Methods, 2017).

    Args:
        adata: an Annodata object.
        dilution: the dilution of the spikein transcript in the lysis reaction mix. Default is 40, 000. The number of
            spike-in transcripts per single-cell lysis reaction was calculated from.
        volume: the approximate volume of the lysis chamber (nanoliters).
        from_layer: the layer in which the ERCC TPM values will be used as the covariate for the ERCC based linear
            regression. Defaults to None.
        to_layers: the layers that our ERCC based transformation will be applied to. Defaults to None.
        mixture_type: the type of spikein transcripts from the spikein mixture added in the experiments. Note that m/c
            we inferred are also based on mixture 1. Defaults to 1.
        ERCC_controls: the FPKM/TPM matrix for each ERCC spike-in transcript in the cells if user wants to perform the
            transformation based on their spike-in data. Note that the row and column names should match up with the
            ERCC_annotation and relative_exprs_matrix respectively. Defaults to None.
        ERCC_annotation: the ERCC_annotation matrix from illumina USE GUIDE which will be ued for calculating the ERCC
            transcript copy number for performing the transformation. Defaults to None.

    Raises:
        Exception: the number of ERCC gene in `ERCC_annotation["ERCC ID"]` is not enough.
        Exception: the layers specified in to_layers are invalid.

    Returns:
        An adata object with the data specified in the to_layers transformed into absolute counts.
    """

    if ERCC_annotation is None:
        ERCC_annotation = pd.read_csv(
            "https://www.dropbox.com/s/cmiuthdw5tt76o5/ERCC_specification.txt?dl=1",
            sep="\t",
        )

    ERCC_id = ERCC_annotation["ERCC ID"]

    ERCC_id = adata.var_names.intersection(ERCC_id)
    if len(ERCC_id) < 10 and ERCC_controls is None:
        raise Exception("The adata object you provided has less than 10 ERCC genes.")

    if to_layers is not None:
        to_layers = [to_layers] if to_layers is str else to_layers
        to_layers = list(set(adata.layers.keys()).intersection(to_layers))
        if len(to_layers) == 0:
            raise Exception(
                f"The layers {to_layers} that will be converted to absolute counts doesn't match any layers"
                f"from the adata object."
            )

    mixture_name = (
        "concentration in Mix 1 (attomoles/ul)" if mixture_type == 1 else "concentration in Mix 2 (attomoles/ul)"
    )
    ERCC_annotation["numMolecules"] = ERCC_annotation.loc[:, mixture_name] * (
        volume * 10 ** (-3) * 1 / dilution * 10 ** (-18) * 6.02214129 * 10 ** (23)
    )

    ERCC_annotation["rounded_numMolecules"] = ERCC_annotation["numMolecules"].astype(int)

    if from_layer in [None, "X"]:
        X, X_ercc = (
            adata.X,
            adata[:, ERCC_id].X if ERCC_controls is None else ERCC_controls,
        )
    else:
        X, X_ercc = (
            adata.layers[from_layer],
            adata[:, ERCC_id] if ERCC_controls is None else ERCC_controls,
        )

    logged = False if X.max() > 100 else True

    if not logged:
        X, X_ercc = (
            np.log1p(X.A) if issparse(X_ercc) else np.log1p(X),
            np.log1p(X_ercc.A) if issparse(X_ercc) else np.log1p(X_ercc),
        )
    else:
        X, X_ercc = (
            X.A if issparse(X_ercc) else X,
            X_ercc.A if issparse(X_ercc) else X_ercc,
        )

    y = np.log1p(ERCC_annotation["numMolecules"])

    for i in range(adata.n_obs):
        X_i, X_ercc_i = X[i, :], X_ercc[i, :]

        X_i, X_ercc_i = sm.add_constant(X_i), sm.add_constant(X_ercc_i)
        res = sm.RLM(y, X_ercc_i).fit()
        k, b = res.params[::-1]

        if to_layers is None:
            X = adata.X
            logged = False if X.max() > 100 else True

            if not logged:
                X_i = np.log1p(X[i, :].A) if issparse(X) else np.log1p(X[i, :])
            else:
                X_i = X[i, :].A if issparse(X) else X[i, :]

            res = k * X_i + b
            res = res if logged else np.expm1(res)
            adata.X[i, :] = csr_matrix(res) if issparse(X) else res
        else:
            for cur_layer in to_layers:
                X = adata.layers[cur_layer]

                logged = False if X.max() > 100 else True
                if not logged:
                    X_i = np.log1p(X[i, :].A) if issparse(X) else np.log1p(X[i, :])
                else:
                    X_i = X[i, :].A if issparse(X) else X[i, :]

                res = k * X_i + b if logged else np.expm1(k * X_i + b)
                adata.layers[cur_layer][i, :] = csr_matrix(res) if issparse(X) else res


# ---------------------------------------------------------------------------------------------------
# coordinate/vector space operations


def affine_transform(X: npt.ArrayLike, A: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    """Perform affine trasform on an array.

    Args:
        X: the array to tranform.
        A: the scaling/rotation/shear matrix.
        b: the transformation matrix.

    Returns:
        The result array.
    """

    X = np.array(X)
    A = np.array(A)
    b = np.array(b)
    return (A @ X.T).T + b


def gen_rotation_2d(degree: float) -> np.ndarray:
    """Calculate the 2D rotation transform matrix for given rotation in degrees.

    Args:
        degree: the degrees to rotate.

    Returns:
        The rotation matrix.
    """

    from math import cos, radians, sin

    rad = radians(degree)
    R = [
        [cos(rad), -sin(rad)],
        [sin(rad), cos(rad)],
    ]
    return np.array(R)


def reset_adata_X(adata: AnnData, experiment_type: str, has_labeling: bool, has_splicing: bool):
    if has_labeling:
        if experiment_type.lower() in [
            "one-shot",
            "kin",
            "mixture",
            "mix_std_stm",
            "kinetics",
            "mix_pulse_chase",
            "mix_kin_deg",
        ]:
            adata.X = adata.layers["total"].copy()
        if experiment_type.lower() in ["deg", "degradation"] and has_splicing:
            adata.X = adata.layers["spliced"].copy()
        if experiment_type.lower() in ["deg", "degradation"] and not has_splicing:
            main_warning(
                "It is not possible to calculate RNA velocity from a degradation experiment which has no "
                "splicing information."
            )
            adata.X = adata.layers["total"].copy()
        else:
            adata.X = adata.layers["total"].copy()
    else:
        adata.X = adata.layers["spliced"].copy()


def del_raw_layers(adata: AnnData):
    layers = list(adata.layers.keys())
    for layer in layers:
        if not layer.startswith("X_"):
            del adata.layers[layer]
