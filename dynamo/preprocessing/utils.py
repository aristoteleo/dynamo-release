import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

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
from scipy.sparse import csc_matrix, csr_matrix, issparse
from scipy.sparse.linalg import LinearOperator, svds
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip
from sklearn.utils.sparsefuncs import mean_variance_axis

from ..configuration import DKM, DynamoAdataKeyManager
from ..dynamo_logger import (
    LoggerManager,
    main_debug,
    main_exception,
    main_info,
    main_info_insert_adata_var,
    main_warning,
)
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
        adata.var = merge_df
        valid_ind = np.where(merge_df["notfound"] != True)[0]  # noqa: E712

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
# implmentation of Cooks' distance (but this is for Poisson distribution fitting)

# https://stackoverflow.com/questions/47686227/poisson-regression-in-statsmodels-and-r

# from __future__ import division, print_function

# https://stats.stackexchange.com/questions/356053/the-identity-link-function-does-not-respect-the-domain-of-the-gamma-
# family
def _weight_matrix(fitted_model: sm.Poisson) -> np.ndarray:
    """Calculates weight matrix in Poisson regression.

    Args:
        fitted_model: a fitted Poisson model

    Returns:
        A diagonal weight matrix in Poisson regression.
    """

    return np.diag(fitted_model.fittedvalues)


def _hessian(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Hessian matrix calculated as -X'*W*X.

    Args:
        X: the matrix of covariates.
        W: the weight matrix.

    Returns:
        The result Hessian matrix.
    """

    return -np.dot(X.T, np.dot(W, X))


def _hat_matrix(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Calculate hat matrix = W^(1/2) * X * (X'*W*X)^(-1) * X'*W^(1/2)

    Args:
        X: the matrix of covariates.
        W: the diagonal weight matrix

    Returns:
        The result hat matrix
    """

    # W^(1/2)
    Wsqrt = W ** (0.5)

    # (X'*W*X)^(-1)
    XtWX = -_hessian(X=X, W=W)
    XtWX_inv = np.linalg.inv(XtWX)

    # W^(1/2)*X
    WsqrtX = np.dot(Wsqrt, X)

    # X'*W^(1/2)
    XtWsqrt = np.dot(X.T, Wsqrt)

    return np.dot(WsqrtX, np.dot(XtWX_inv, XtWsqrt))


def cook_dist(model: sm.Poisson, X: np.ndarray, good: npt.ArrayLike) -> np.ndarray:
    """calculate Cook's distance

    Args:
        model: a fitted Poisson model.
        X: the matrix of covariates.
        good: the dispersion table for MSE calculation.

    Returns:
        The result Cook's distance.
    """

    # Weight matrix
    W = _weight_matrix(model)

    # Hat matrix
    H = _hat_matrix(X, W)
    hii = np.diag(H)  # Diagonal values of hat matrix # fit.get_influence().hat_matrix_diag

    # Pearson residuals
    r = model.resid_pearson

    # Cook's distance (formula used by R = (res/(1 - hat))^2 * hat/(dispersion * p))
    # Note: dispersion is 1 since we aren't modeling overdispersion

    resid = good.disp - model.predict(good)
    rss = np.sum(resid**2)
    MSE = rss / (good.shape[0] - 2)
    # use the formula from: https://www.mathworks.com/help/stats/cooks-distance.html
    cooks_d = r**2 / (2 * MSE) * hii / (1 - hii) ** 2  # (r / (1 - hii)) ** 2 *  / (1 * 2)

    return cooks_d


# ---------------------------------------------------------------------------------------------------
# preprocess utilities
def filter_genes_by_pattern(
    adata: anndata.AnnData,
    patterns: Tuple[str] = ("MT-", "RPS", "RPL", "MRPS", "MRPL", "ERCC-"),
    drop_genes: bool = False,
) -> Union[List[bool], None]:
    """Utility function to filter mitochondria, ribsome protein and ERCC spike-in genes, etc.

    Args:
        adata: an AnnData object.
        patterns: the patterns used to filter genes. Defaults to ("MT-", "RPS", "RPL", "MRPS", "MRPL", "ERCC-").
        drop_genes: whether inplace drop the genes from the AnnData object. Defaults to False.

    Returns:
        A list of indices of matched genes if `drop_genes` is False. Otherwise, returns none.
    """

    logger = LoggerManager.gen_logger("dynamo-utils")

    matched_genes = pd.Series(adata.var_names).str.startswith(patterns).to_list()
    logger.info(
        "total matched genes is " + str(sum(matched_genes)),
        indent_level=1,
    )
    if sum(matched_genes) > 0:
        if drop_genes:
            gene_bools = np.ones(adata.n_vars, dtype=bool)
            gene_bools[matched_genes] = False
            logger.info(
                "inplace subset matched genes ... ",
                indent_level=1,
            )
            # let us ignore the `inplace` parameter in pandas.Categorical.remove_unused_categories  warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                adata._inplace_subset_var(gene_bools)

            logger.finish_progress(progress_name="filter_genes_by_pattern")
            return None
        else:
            logger.finish_progress(progress_name="filter_genes_by_pattern")
            return matched_genes


def basic_stats(adata: anndata.AnnData) -> None:
    """Generate basic stats of the adata, including number of genes, number of cells, and number of mitochondria genes.

    Args:
        adata: an AnnData object.
    """

    adata.obs["nGenes"], adata.obs["nCounts"] = np.array((adata.X > 0).sum(1)), np.array((adata.X).sum(1))
    adata.var["nCells"], adata.var["nCounts"] = np.array((adata.X > 0).sum(0).T), np.array((adata.X).sum(0).T)
    if adata.var_names.inferred_type == "bytes":
        adata.var_names = adata.var_names.astype("str")
    mito_genes = adata.var_names.str.upper().str.startswith("MT-")

    if sum(mito_genes) > 0:
        try:
            adata.obs["pMito"] = np.array(adata.X[:, mito_genes].sum(1) / adata.obs["nCounts"].values.reshape((-1, 1)))
        except:  # noqa E722
            main_exception(
                "no mitochondria genes detected; looks like your var_names may be corrupted (i.e. "
                "include nan values). If you don't believe so, please report to us on github or "
                "via xqiu@wi.mit.edu"
            )
    else:
        adata.obs["pMito"] = 0


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


def clusters_stats(
    U: pd.DataFrame, S: pd.DataFrame, clusters_uid: np.ndarray, cluster_ix: np.ndarray, size_limit: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the averages per cluster for unspliced and spliced data.

    Args:
        U: the unspliced DataFrame.
        S: the spliced DataFrame.
        clusters_uid: the uid of the clusters.
        cluster_ix: the indices of the clusters in adata.obs.
        size_limit: the max number of members to be considered in a cluster during calculation. Defaults to 40.

    Returns:
        U_avgs: the average of clusters for unspliced data.
        S_avgs: the average of clusters for spliced data.
    """

    U_avgs = np.zeros((S.shape[1], len(clusters_uid)))
    S_avgs = np.zeros((S.shape[1], len(clusters_uid)))
    # avgU_div_avgS = np.zeros((S.shape[1], len(clusters_uid)))
    # slopes_by_clust = np.zeros((S.shape[1], len(clusters_uid)))

    for i, uid in enumerate(clusters_uid):
        cluster_filter = cluster_ix == i
        n_cells = np.sum(cluster_filter)
        if n_cells > size_limit:
            U_avgs[:, i], S_avgs[:, i] = (
                U[cluster_filter, :].mean(0),
                S[cluster_filter, :].mean(0),
            )
        else:
            U_avgs[:, i], S_avgs[:, i] = U.mean(0), S.mean(0)

    return U_avgs, S_avgs


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
        szfactors = adata.obs[total_szfactor][:, None]
    elif total_szfactor is not None:
        main_warning("`total_szfactor` is not `None` and it is not in adata object.")

    return szfactors, CM


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


def _Freeman_Tukey(X: np.ndarray, inverse=False) -> np.ndarray:
    """perform Freeman-Tukey transform or inverse transform on the given array.

    Args:
        X: a matrix.
        inverse: whether to perform inverse Freeman-Tukey transform. Defaults to False.

    Returns:
        The transformed array.
    """

    if inverse:
        res = np.sqrt(X) + np.sqrt((X + 1))
    else:
        res = (X**2 - 1) ** 2 / (4 * X**2)

    return res


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


# ---------------------------------------------------------------------------------------------------
# pca


def _truncatedSVD_with_center(
    X: Union[csc_matrix, csr_matrix],
    n_components: int = 30,
    random_state: int = 0,
) -> Dict:
    """Center a sparse matrix and perform truncated SVD on it.

    It uses `scipy.sparse.linalg.LinearOperator` to express the centered sparse
    input by given matrix-vector and matrix-matrix products. Then truncated
    singular value decomposition (SVD) can be solved without calculating the
    individual entries of the centered matrix. The right singular vectors after
    decomposition represent the principal components. This function is inspired
    by the implementation of scanpy (https://github.com/scverse/scanpy).

    Args:
        X: The input sparse matrix to perform truncated SVD on.
        n_components: The number of components to keep. Default is 30.
        random_state: Seed for the random number generator. Default is 0.

    Returns:
        The transformed input matrix and a sklearn PCA object containing the
        right singular vectors and amount of variance explained by each
        principal component.
    """
    random_state = check_random_state(random_state)
    np.random.set_state(random_state.get_state())
    v0 = random_state.uniform(-1, 1, np.min(X.shape))
    n_components = min(n_components, X.shape[1] - 1)

    mean = X.mean(0)
    X_H = X.T.conj()
    mean_H = mean.T.conj()
    ones = np.ones(X.shape[0])[None, :].dot

    # Following callables implements different type of matrix calculation.
    def matvec(x):
        """Matrix-vector multiplication. Performs the operation X_centered*x
        where x is a column vector or an 1-D array."""
        return X.dot(x) - mean.dot(x)

    def matmat(x):
        """Matrix-matrix multiplication. Performs the operation X_centered*x
        where x is a matrix or ndarray."""
        return X.dot(x) - mean.dot(x)

    def rmatvec(x):
        """Adjoint matrix-vector multiplication. Performs the operation
        X_centered^H * x where x is a column vector or an 1-d array."""
        return X_H.dot(x) - mean_H.dot(ones(x))

    def rmatmat(x):
        """Adjoint matrix-matrix multiplication. Performs the operation
        X_centered^H * x where x is a matrix or ndarray."""
        return X_H.dot(x) - mean_H.dot(ones(x))

    # Construct the LinearOperator with callables above.
    X_centered = LinearOperator(
        shape=X.shape,
        matvec=matvec,
        matmat=matmat,
        rmatvec=rmatvec,
        rmatmat=rmatmat,
        dtype=X.dtype,
    )

    # Solve SVD without calculating individuals entries in LinearOperator.
    U, Sigma, VT = svds(X_centered, solver="arpack", k=n_components, v0=v0)
    Sigma = Sigma[::-1]
    U, VT = svd_flip(U[:, ::-1], VT[::-1])
    X_transformed = U * Sigma
    components_ = VT
    exp_var = np.var(X_transformed, axis=0)
    _, full_var = mean_variance_axis(X, axis=0)
    full_var = full_var.sum()

    result_dict = {
        "X_pca": X_transformed,
        "components_": components_,
        "explained_variance_ratio_": exp_var / full_var,
    }

    fit = PCA(
        n_components=n_components,
        random_state=random_state,
    )
    X_pca = result_dict["X_pca"]
    fit.mean_ = mean.A1.flatten()
    fit.components_ = result_dict["components_"]
    fit.explained_variance_ratio_ = result_dict["explained_variance_ratio_"]

    return fit, X_pca


def _pca_fit(
    X: np.ndarray,
    pca_func: Callable,
    n_components: int = 30,
    **kwargs,
) -> Tuple:
    """Apply PCA to the input data array X using the specified PCA function.

    Args:
        X: the input data array of shape (n_samples, n_features).
        pca_func: the PCA function to use, which should have a 'fit' and
            'transform' method, such as the PCA class or the IncrementalPCA
            class from sklearn.decomposition.
        n_components: the number of principal components to compute. If
            n_components is greater than or equal to the number of features in
            X, it will be set to n_features - 1 to avoid overfitting.
        **kwargs: any additional keyword arguments that will be passed to the
            PCA function.

    Returns:
        A tuple containing two elements:
            - The fitted PCA object, which has a 'fit' and 'transform' method.
            - The transformed array X_pca of shape (n_samples, n_components).
    """
    fit = pca_func(
        n_components=min(n_components, X.shape[1] - 1),
        **kwargs,
    ).fit(X)
    X_pca = fit.transform(X)
    return fit, X_pca


def pca(
    adata: AnnData,
    X_data: np.ndarray = None,
    n_pca_components: int = 30,
    pca_key: str = "X_pca",
    pcs_key: str = "PCs",
    genes_to_append: Union[List[str], None] = None,
    layer: Union[List[str], str, None] = None,
    svd_solver: Literal["randomized", "arpack"] = "randomized",
    random_state: int = 0,
    use_truncated_SVD_threshold: int = 500000,
    use_incremental_PCA: bool = False,
    incremental_batch_size: Optional[int] = None,
    return_all: bool = False,
) -> Union[AnnData, Tuple[AnnData, Union[PCA, TruncatedSVD], np.ndarray]]:
    """Perform PCA reduction for monocle recipe.

    When large dataset is used (e.g. 1 million cells are used), Incremental PCA
    is recommended to avoid the memory issue. When cell number is less than half
    a million, by default PCA or _truncatedSVD_with_center (use sparse matrix
    that doesn't explicitly perform centering) will be used. TruncatedSVD is the
    fastest method. Unlike other methods which will center the data first,  it
    performs SVD decomposition on raw input. Only use this when dataset is too
    large for other methods.

    Args:
        adata: an AnnData object.
        X_data: the data to perform dimension reduction on. Defaults to None.
        n_pca_components: number of PCA components reduced to. Defaults to 30.
        pca_key: the key to store the reduced data. Defaults to "X".
        pcs_key: the key to store the principle axes in feature space. Defaults
            to "PCs".
        genes_to_append: a list of genes should be inspected. Defaults to None.
        layer: the layer(s) to perform dimension reduction on. Would be
            overrided by X_data. Defaults to None.
        svd_solver: the svd_solver to solve svd decomposition in PCA.
        random_state: the seed used to initialize the random state for PCA.
        use_truncated_SVD_threshold: the threshold of observations to use
            truncated SVD instead of standard PCA for efficiency.
        use_incremental_PCA: whether to use Incremental PCA. Recommend enabling
            incremental PCA when dataset is too large to fit in memory.
        incremental_batch_size: The number of samples to use for each batch when
            performing incremental PCA. If batch_size is None, then batch_size
            is inferred from the data and set to 5 * n_features.
        return_all: whether to return the PCA fit model and the reduced array
            together with the updated AnnData object. Defaults to False.

    Raises:
        ValueError: layer provided is not invalid.
        ValueError: list of genes to append is invalid.

    Returns:
        The updated AnnData object with reduced data if `return_all` is False.
        Otherwise, a tuple (adata, fit, X_pca), where adata is the updated
        AnnData object, fit is the fit model for dimension reduction, and X_pca
        is the reduced array, will be returned.
    """

    # only use genes pass filter (based on use_for_pca) to perform dimension reduction.
    if X_data is None:
        if "use_for_pca" not in adata.var.keys():
            adata.var["use_for_pca"] = True

        if layer is None:
            X_data = adata.X[:, adata.var.use_for_pca.values]
        else:
            if "X" in layer:
                X_data = adata.X[:, adata.var.use_for_pca.values]
            elif "total" in layer:
                X_data = adata.layers["X_total"][:, adata.var.use_for_pca.values]
            elif "spliced" in layer:
                X_data = adata.layers["X_spliced"][:, adata.var.use_for_pca.values]
            elif "protein" in layer:
                X_data = adata.obsm["X_protein"]
            elif type(layer) is str:
                X_data = adata.layers["X_" + layer][:, adata.var.use_for_pca.values]
            else:
                raise ValueError(
                    f"your input layer argument should be either a `str` or a list that includes one of `X`, "
                    f"`total`, `protein` element. `Layer` currently is {layer}."
                )

        cm_genesums = X_data.sum(axis=0)
        valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
        valid_ind = np.array(valid_ind).flatten()

        bad_genes = np.where(adata.var.use_for_pca)[0][~valid_ind]
        if genes_to_append is not None and len(adata.var.index[bad_genes].intersection(genes_to_append)) > 0:
            raise ValueError(
                f"The gene list passed to argument genes_to_append contains genes with no expression "
                f"across cells or non finite values. Please check those genes:"
                f"{set(bad_genes).intersection(genes_to_append)}!"
            )

        adata.var.iloc[bad_genes, adata.var.columns.tolist().index("use_for_pca")] = False
        X_data = X_data[:, valid_ind]

    if use_incremental_PCA:
        from sklearn.decomposition import IncrementalPCA

        fit, X_pca = _pca_fit(
            X_data,
            pca_func=IncrementalPCA,
            n_components=n_pca_components,
            batch_size=incremental_batch_size,
        )
    else:
        if adata.n_obs < use_truncated_SVD_threshold:
            if not issparse(X_data):
                fit, X_pca = _pca_fit(
                    X_data,
                    pca_func=PCA,
                    n_components=n_pca_components,
                    svd_solver=svd_solver,
                    random_state=random_state,
                )
            else:
                fit, X_pca = _truncatedSVD_with_center(
                    X_data,
                    n_components=n_pca_components,
                    random_state=random_state,
                )
        else:
            # TruncatedSVD is the fastest method we have. It doesn't center the
            # data. It only performs SVD decomposition, which is the second part
            # in our _truncatedSVD_with_center function.
            fit, X_pca = _pca_fit(
                X_data, pca_func=TruncatedSVD, n_components=n_pca_components + 1, random_state=random_state
            )
            # first columns is related to the total UMI (or library size)
            X_pca = X_pca[:, 1:]

    adata.obsm[pca_key] = X_pca
    if use_incremental_PCA or adata.n_obs < use_truncated_SVD_threshold:
        adata.uns[pcs_key] = fit.components_.T
        adata.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_
    else:
        # first columns is related to the total UMI (or library size)
        adata.uns[pcs_key] = fit.components_.T[:, 1:]
        adata.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_[1:]

    adata.uns["pca_mean"] = fit.mean_ if hasattr(fit, "mean_") else np.zeros(X_data.shape[1])

    if return_all:
        return adata, fit, X_pca
    else:
        return adata


def pca_genes(PCs: list, n_top_genes: int = 100) -> np.ndarray:
    """For each gene, if the gene is n_top in some principle component then it is valid. Return all such valid genes.

    Args:
        PCs: principle components(PC) of PCA
        n_top_genes: number of gene candidates in EACH PC. Defaults to 100.

    Returns:
        A bool array indicating whether the gene is valid.
    """

    valid_genes = np.zeros(PCs.shape[0], dtype=bool)
    for q in PCs.T:
        sorted_q = np.sort(np.abs(q))[::-1]
        is_pc_top_n = np.abs(q) > sorted_q[n_top_genes]
        valid_genes = np.logical_or(is_pc_top_n, valid_genes)
    return valid_genes


def top_pca_genes(
    adata: AnnData,
    pc_key: str = "PCs",
    n_top_genes: int = 100,
    pc_components: Union[int, None] = None,
    adata_store_key: str = "top_pca_genes",
) -> AnnData:
    """Define top genes as any gene that is ``n_top_genes`` in some principle component.

    Args:
        adata: an AnnData object.
        pc_key: component key stored in adata.uns. Defaults to "PCs".
        n_top_genes: number of top genes as valid top genes in each component. Defaults to 100.
        pc_components: number of top principle components to use. Defaults to None.
        adata_store_key: the key for storing pca genes. Defaults to "top_pca_genes".

    Raises:
        Exception: invalid pc_key.

    Returns:
        The AnnData object with top genes stored as values of adata.var[adata_store_key].
    """

    if pc_key in adata.uns.keys():
        Q = adata.uns[pc_key]
    elif pc_key in adata.varm.keys():
        Q = adata.varm[pc_key]
    else:
        raise Exception(f"No PC matrix {pc_key} found in neither .uns nor .varm.")
    if pc_components is not None:
        if type(pc_components) == int:
            Q = Q[:, :pc_components]
        elif type(pc_components) == list:
            Q = Q[:, pc_components]

    pcg = pca_genes(Q, n_top_genes=n_top_genes)
    genes = np.zeros(adata.n_vars, dtype=bool)
    if DKM.VAR_USE_FOR_PCA in adata.var.keys():
        genes[adata.var[DKM.VAR_USE_FOR_PCA]] = pcg
    else:
        genes = pcg
    main_info_insert_adata_var(adata_store_key, indent_level=2)
    adata.var[adata_store_key] = genes
    return adata


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
