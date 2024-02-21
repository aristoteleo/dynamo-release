import itertools
import time
import warnings
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from anndata._core.anndata import AnnData
from anndata._core.views import ArrayView
from scipy import interpolate
from scipy import sparse as sp
from scipy import stats
from scipy.integrate import odeint
from scipy.linalg.blas import dgemm
from scipy.spatial import cKDTree
from scipy.spatial.distance import squareform as spsquare
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from ..dynamo_logger import (
    Logger,
    LoggerManager,
    main_debug,
    main_info,
    main_info_insert_adata,
    main_info_verbose_timeit,
    main_warning,
)
from ..preprocessing.transform import _Freeman_Tukey
from ..utils import areinstance, isarray


# ---------------------------------------------------------------------------------------------------
# others
def get_mapper(smoothed: bool = True) -> Dict[str, str]:
    """Return the mapper for layers depending on whether the data is smoothed.

    Args:
        smoothed: Whether the data is smoothed. Defaults to True.

    Returns:
        The mapper dictionary for layers.
    """

    mapper = {
        "X_spliced": "M_s" if smoothed else "X_spliced",
        "X_unspliced": "M_u" if smoothed else "X_unspliced",
        "X_new": "M_n" if smoothed else "X_new",
        "X_old": "M_o" if smoothed else "X_old",
        "X_total": "M_t" if smoothed else "X_total",
        "X_uu": "M_uu" if smoothed else "X_uu",
        "X_ul": "M_ul" if smoothed else "X_ul",
        "X_su": "M_su" if smoothed else "X_su",
        "X_sl": "M_sl" if smoothed else "X_sl",
        "X_protein": "M_p" if smoothed else "X_protein",
        "X": "X" if smoothed else "X",
    }
    return mapper


def get_mapper_inverse(smoothed: bool = True) -> Dict[str, str]:
    """Return the inverse mapper for layers depending on whether the data is smoothed.

    Args:
        smoothed: Whether the data is smoothed. Defaults to True.

    Returns:
        The inverse mapper dictionary for layers.
    """

    mapper = get_mapper(smoothed)

    return dict([(v, k) for k, v in mapper.items()])


def get_finite_inds(X: Union[np.ndarray, sp.csr_matrix], ax: int = 0) -> np.ndarray:
    """Find the indices of finite elements in an array.

    Args:
        X: The matrix to be inspected.
        ax: The axis for indexing. Defaults to 0.

    Returns:
        The indices of finite elements.
    """

    finite_inds = np.isfinite(X.sum(ax).A1) if sp.issparse(X) else np.isfinite(X.sum(ax))

    return finite_inds


def get_pd_row_column_idx(
    df: pd.DataFrame, queries: List[str], type: Literal["column", "row"] = "column"
) -> np.ndarray:
    """Find the numeric indices of multiple index/column matches with a vectorized solution using np.searchsorted.

    The function is adapted from:
    https://stackoverflow.com/questions/13021654/get-column-index-from-column-name-in-python-pandas

    Args:
        df: The dataframe to be inspected.
        queries: A list of either column names or index of the dataframe that will be used for finding indices.
        type: The type of the queries/search, either `column` (list of queries are from column names) or "row" (list of
            queries are from index names). Defaults to "column".

    Returns:
        An one dimensional array for the numeric indices that corresponds to the matches of the queries.
    """

    names = df.columns.values if type == "column" else df.index.values if type == "row" else None
    sidx = np.argsort(names)
    Indices = sidx[np.searchsorted(names, queries, sorter=sidx)]

    return Indices


def update_dict(dict1: dict, dict2: dict) -> dict:
    """Update the values of dict 1 with the values of dict 2. The keys of dict 1 would not be modified.

    Args:
        dict1: The dict to be updated.
        dict2: The dict to provide new values.

    Returns:
        The updated dict.
    """
    dict1.update((k, dict2[k]) for k in dict1.keys() & dict2.keys())

    return dict1


def update_n_merge_dict(dict1: dict, dict2: dict) -> dict:
    """Merge two dictionaries.

    For overlapping keys, the values in dict 2 would replace values in dict 1.

    Args:
        dict1: The dict to be merged into and overwritten.
        dict2: The dict to be merged.

    Returns:
        The updated dict.
    """

    dict = {
        **dict1,
        **dict2,
    }  # dict1.update((k, dict2[k]) for k in dict1.keys() | dict2.keys())

    return dict


def subset_dict_with_key_list(dict: dict, list: list) -> dict:
    """Subset the dict with keys provided.

    Args:
        dict: The dict to be subset.
        list: The keys that should be left in dict.

    Returns:
        The subset dict.
    """

    return {key: value for key, value in dict.items() if key in list}


def nearest_neighbors(coord: np.ndarray, coords: Union[np.ndarray, sp.csr_matrix], k: int = 5) -> np.ndarray:
    """Find the nearest neighbors in a given space for a given point.

    Args:
        coord: The point for which nearest neighbors are searched.
        coords: The space to search neighbors.
        k: The number of neighbors to be searched. Defaults to 5.

    Returns:
        The indices of the nearest neighbors.
    """

    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(coords)
    _, neighs = nbrs.kneighbors(np.atleast_2d(coord))
    return neighs


def nbrs_to_dists(X: np.ndarray, nbrs_idx: np.ndarray) -> List[np.ndarray]:
    """Calculate the distances between neighbors of a given space.

    Args:
        X: The space to find nearest neighbors on.
        nbrs_idx: The indices of nearest neighbors found for each point.

    Returns:
        The distances between neighbors and the point.
    """

    dists = []
    n = X.shape[0]
    for i in range(n):
        d = X[nbrs_idx[i]] - X[i]
        d = np.linalg.norm(d, axis=1)
        dists.append(d)
    return dists


def symmetrize_symmetric_matrix(W: Union[np.ndarray, sp.csr_matrix]) -> sp.csr_matrix:
    """
    Symmetrize a supposedly symmetric matrix W, so that W_ij == Wji strictly.

    Args:
        W: The matrix supposed to be symmetric.

    Returns:
        The matrix that is now strictly symmetric.
    """

    if not sp.issparse(W):
        W = sp.csr_matrix(W)

    _row_inds, _col_inds = W.nonzero()
    _data = W.data.copy()

    row_inds = np.hstack((_row_inds, _col_inds))
    col_inds = np.hstack((_col_inds, _row_inds))
    data = np.hstack((_data, _data))

    I = np.unique(np.vstack((row_inds, col_inds)).T, axis=0, return_index=True)[1]
    W = sp.csr_matrix((data[I], (row_inds[I], col_inds[I])))
    return W


def create_layer(
    adata: AnnData,
    data: np.ndarray,
    layer_key: Optional[str] = None,
    genes: Optional[np.ndarray] = None,
    cells: Optional[np.ndarray] = None,
    **kwargs,
) -> Optional[np.ndarray]:
    """Create a new layer with data supplied.

    Args:
        adata: An AnnData object to insert the layer into.
        data: The main data of the new layer.
        layer_key: The key of the layer when gets inserted into the adata object. If None, the layer would be returned.
            Defaults to None.
        genes: The genes for the provided data. If None, genes from the adata object would be used. Defaults to None.
        cells: The cells for the provided data. If None, cells from the adata object would be used. Defaults to None.

    Returns:
        If the layer key is provided, nothing would be returned. Otherwise, the layer itself would be returned.
    """
    all_genes = adata.var.index
    if genes is None:
        genes = all_genes
    elif areinstance(genes, np.bool_) or areinstance(genes, bool):
        genes = all_genes[genes]

    if cells is None:
        cells = np.arange(adata.n_obs)

    new = np.empty(adata.X.shape, **kwargs)
    new[:] = np.nan
    for i, g in enumerate(genes):
        ig = np.where(all_genes == g)[0]
        if len(ig) > 0:
            for igi in ig:
                new[cells, igi] = data[:, i]

    if layer_key is not None:
        main_info_insert_adata(layer_key, adata_attr="layers")
        adata.layers[layer_key] = new
    else:
        return new


def AddAssay(adata: AnnData, data: pd.DataFrame, key: str, slot: str = "obsm") -> AnnData:
    """Add a new data as a key to the specified slot.

    Args:
        adata: An AnnData object.
        data: The data (in pandas DataFrame format) that will be added to adata.
        key: The key name to be used for the new data.
        slot: The slot of adata to store the new data. Defaults to "obsm".

    Returns:
        An updated anndata object that are updated with a new data as a key to the specified slot.
    """

    if slot == "uns":
        adata.uns[key] = data.loc[adata.obs.index, set(adata.var.index).intersection(data.columns)]
    elif slot == "obsm":
        adata.obsm[key] = data.loc[adata.obs.index, set(adata.var.index).intersection(data.columns)]

    return adata


def getAssay(adata: AnnData, key: str, slot: str = "obsm") -> pd.DataFrame:
    """Retrieve a key named data from the specified slot.

    Args:
        adata: An AnnData object.
        key: The key name of the data to be retrieved. .
        slot: The slot of adata to be retrieved from. Defaults to "obsm".

    Returns:
        The data (in pd.DataFrame) that will be retrieved from adata.
    """

    if slot == "uns":
        data = adata.uns[key]
    elif slot == "obsm":
        data = adata.obsm[key]

    return data


def index_gene(adata: AnnData, arr: np.ndarray, genes: List[str]) -> np.ndarray:
    """A lightweight method for indexing adata arrays by genes.

    The function is designed to have good memory efficiency especially when `.uns` contains large data.

    Args:
        adata: An AnnData object.
        arr: The array to be indexed.
            If 1d, the length of the array has to be equal to `adata.n_vars`.
            If 2d, the second dimension of the array has to be equal to `adata.n_vars`.
        genes: A list of gene names or boolean flags for indexing.

    Raises:
        ValueError: Gene in `genes` not found in adata.
        Exception: The lengths of arr does not match the number of genes.
        Exception: The dimension of arr does not match the number of genes.

    Returns:
        The indexed array.
    """

    if areinstance(genes, [bool, np.bool_]):
        mask = np.array(genes)
    else:
        all_genes = adata.var_names
        # Note: this mask here is in fact an indices vector!
        mask = np.zeros(len(genes), dtype=int)
        for i, g in enumerate(genes):
            if g in all_genes:
                mask[i] = all_genes.get_loc(g)
            else:
                raise ValueError(f"the gene {g} you provided is not included in the data.")

    if arr.ndim == 1:
        if len(arr) != adata.n_vars:
            raise Exception("The length of the input array does not match the number of genes.")
        else:
            return arr[mask]
    else:
        if arr.shape[1] != adata.n_vars:
            raise Exception("The dimension of the input array does not match the number of genes.")
        else:
            return arr[:, mask]


def reserve_minimal_genes_by_gamma_r2(adata: AnnData, var_store_key: str, minimal_gene_num: int = 50) -> pd.DataFrame:
    """Select given number of minimal genes.

    Args:
        adata: An AnnData object.
        var_store_key: The key in adata.var for the gene count data.
        minimal_gene_num: The number of minimal genes to select. Defaults to 50.

    Raises:
        ValueError: `adata.var[var_store_key]` invalid.
        ValueError: `adata.var[var_store_key]` does not have enough genes with non-nan values.

    Returns:
        The minimal gene data.
    """

    vel_params_df = get_vel_params(adata)

    # already satisfy the requirement
    if var_store_key in adata.var.columns and adata.var[var_store_key].sum() >= minimal_gene_num:
        return adata.var[var_store_key]

    if var_store_key not in adata.var.columns:
        raise ValueError("adata.var.%s does not exists." % (var_store_key))

    gamma_r2_not_na = np.array(vel_params_df.gamma_r2[vel_params_df.gamma_r2.notna()])
    if len(gamma_r2_not_na) < minimal_gene_num:
        raise ValueError("adata.var.%s does not have enough values that are not NA." % (var_store_key))

    argsort_result = np.argsort(-np.abs(gamma_r2_not_na))
    adata.var[var_store_key] = False
    adata.var[var_store_key][argsort_result[:minimal_gene_num]] = True
    return adata.var[var_store_key]


def select_cell(
    adata: AnnData,
    grp_keys: Union[str, List[str]],
    grps: Union[str, List[str]],
    presel: Optional[np.ndarray] = None,
    mode: Literal["union", "intersection"] = "union",
    output_format: Literal["mask", "index"] = "index",
) -> np.ndarray:
    """Select cells based on `grep_keys` in .obs.

    Args:
        adata: An AnnData object.
        grp_keys: The key(s) in `.obs` to be used for selecting cells. If a list, each element is a key in .obs that
            corresponds to an element in `grps`.
        grps: The value(s) in `.obs[grp_keys]` to be used for selecting cells. If a list, each element is a value that
            corresponds to an element in `grp_keys`.
        presel: An array of indices or mask of pre-selected cells. It will be combined with selected cells specified by
            `grp_keys` and `grps` according to `mode`. Defaults to None.
        mode: The mode to select cells.
            "union" - the selected cells are the union of the groups specified in `grp_keys` and `grps`;
            "intersection" - the selected cells are the intersection of the groups specified in `grp_keys` and `grps`.
            Defaults to "union".
        output_format: Whether to output a mask of selection or selected items' indices.
            "index" - returns a list of indices of selected cells;
            "mask" - returns an array of booleans. Defaults to "index".

    Raises:
        NotImplementedError: `mode` is invalid.
        Exception: `grp_keys` has key that is not in .obs.
        NotImplementedError: `output_format` is invalid.

    Returns:
        A mask of selection or selected items' indices.
    """

    if type(grp_keys) is str:
        grp_keys = [grp_keys]
    if not isarray(grps):
        grps = [grps]

    if len(grp_keys) == 1 and len(grps) > 1:
        grp_keys = np.repeat(grp_keys, len(grps))

    if mode == "intersection":
        pred = AlwaysTrue()
    elif mode == "union":
        pred = AlwaysFalse()
    else:
        raise NotImplementedError(f"The mode {mode} is not implemented.")

    for i, k in enumerate(grp_keys):
        # check if all keys in grp_keys are in adata.obs
        if k not in adata.obs.keys():
            raise Exception(f"The group key `{k}` is not in .obs.")
        else:
            in_grp = AnnDataPredicate(k, grps[i])
            if mode == "intersection":
                pred = pred & in_grp
            else:
                pred = pred | in_grp

    cell_idx = pred.check(adata.obs)

    if presel is not None:
        if np.issubsctype(presel, int):
            temp = np.zeros(adata.n_obs, dtype=bool)
            temp[presel] = True
            presel = temp
        if mode == "intersection":
            cell_idx = np.logical_and(presel, cell_idx)
        else:
            cell_idx = np.logical_or(presel, cell_idx)

    if output_format == "index":
        cell_idx = np.where(cell_idx)[0]
    elif output_format == "mask":
        pass
    else:
        raise NotImplementedError(f"The output format `{output_format}` is not supported.")

    return cell_idx


def flatten(arr: Union[pd.Series, sp.csr_matrix, np.ndarray]) -> np.ndarray:
    """Flatten the given array-like object.

    Args:
        arr: The array-like object to be flattened.

    Returns:
        The flatten result.
    """
    if type(arr) == pd.core.series.Series:
        ret = arr.values.flatten()
    elif sp.issparse(arr):
        ret = arr.A.flatten()
    else:
        ret = arr.flatten()
    return ret


def closest_cell(coord: np.ndarray, cells: np.ndarray) -> int:
    """Find the closest cell to the specified coord.

    Args:
        coord: The target coordination.
        cells: An array containing cells.

    Returns:
        The column index of the cell that closest to the coordination specified.
    """
    cells = np.asarray(cells)
    dist_2 = np.sum((cells - coord) ** 2, axis=1)

    return np.argmin(dist_2)


def elem_prod(
    X: Union[np.ndarray, sp.csr_matrix], Y: Union[np.ndarray, sp.csr_matrix]
) -> Union[np.ndarray, sp.csr_matrix]:
    """Calculate element-wise production between 2 arrays.

    Args:
        X: The first array.
        Y: The second array.

    Returns:
        The resulted array.
    """
    if sp.issparse(X):
        return X.multiply(Y)
    elif sp.issparse(Y):
        return Y.multiply(X)
    else:
        return np.multiply(X, Y)


def logdet(A: np.ndarray) -> float:
    """Calculate log(det(A)).

    Compared with calculating log(det(A)) directly, this function avoid the overflow/underflow problems that are likely
    to happen when applying det to large matrices.

    Args:
        A: An square matrix.

    Returns:
        log(det(A)).
    """

    v = 2 * sum(np.log(np.diag(np.linalg.cholesky(A))))
    return v


def norm(x: Union[sp.csr_matrix, np.ndarray], **kwargs) -> np.ndarray:
    """Calculate the norm of an array or matrix

    Args:
        x: The array.
        kwargs: Other kwargs passed to `sp.linalg.norm` or `np.linalg.norm`.
    """
    if sp.issparse(x):
        return sp.linalg.norm(x, **kwargs)
    else:
        return np.linalg.norm(x, **kwargs)


def cell_norm(adata: AnnData, key: str, prefix_store: Optional[str] = None, **norm_kwargs) -> np.ndarray:
    """Calculate the norm of vectors of each cell.

    Args:
        adata: An AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        key: The key of the vectors stored in either .obsm or .layers.
        prefix_store: The prefix used in the key for storing the returned in adata.obs. Defaults to None.

    Raises:
        ValueError: `key` not found in .obsm or .layers.

    Returns:
        The norms of the vectors.
    """

    if key in adata.obsm.keys():
        X = adata.obsm[key]
    elif key in adata.layers.keys():
        X = adata.layers[key]
    else:
        raise ValueError("The key is not found in adata.obsm and adata.layers!")

    ret = norm(X, axis=1, **norm_kwargs)

    if prefix_store is not None:
        adata.obs[prefix_store + "_" + key] = ret
    return ret


def einsum_correlation(X: np.ndarray, Y_i: np.ndarray, type: str = "pearson") -> np.ndarray:
    """Calculate pearson or cosine correlation between gene expression data and the velocity vectors.

    Args:
        X: The gene expression data (genes x cells).
        Y_i: The velocity vector.
        type: The type of correlation to be calculated. Defaults to "pearson".

    Returns:
        The correlation matrix.
    """

    if type == "pearson":
        X -= X.mean(axis=1)[:, None]
        Y_i -= np.nanmean(Y_i)
    elif type == "cosine":
        X, Y_i = X, Y_i
    elif type == "spearman":
        # check this
        X = stats.rankdata(X, axis=1)
        Y_i = stats.rankdata(Y_i)
    elif type == "kendalltau":
        corr = np.array([stats.kendalltau(x, Y_i)[0] for x in X])
        return corr[None, :]

    X_norm, Y_norm = norm(X, axis=1), norm(Y_i)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if Y_norm == 0:
            corr = np.zeros(X_norm.shape[0])
        else:
            corr = np.einsum("ij, j", X, Y_i) / (X_norm * Y_norm)[None, :]

    return corr


def form_triu_matrix(arr: np.ndarray) -> np.ndarray:
    """
    Construct upper triangle matrix from a 1d array.

    Args:
        arr: The array used to generate the upper triangle matrix.

    Returns:
        The generated upper triangle matrix.
    """
    n = int(np.ceil((np.sqrt(1 + 8 * len(arr)) - 1) * 0.5))
    M = np.zeros((n, n))
    c = 0
    for i in range(n):
        for j in range(n):
            if j >= i:
                if c < len(arr):
                    M[i, j] = arr[c]
                    c += 1
                else:
                    break
    return M


def index_condensed_matrix(n: int, i: int, j: int) -> int:
    """Return the index of an element in a condensed n-by-n square matrix by the row index i and column index j of the
    square form.

    Args:
        n: Size of the square form.
        i: Row index of the element in the square form.
        j: Column index of the element in the square form.

    Returns:
        The index of the element in the condensed matrix.
    """

    if i == j:
        main_warning("Diagonal elements (i=j) are not stored in condensed matrices.")
        return None
    elif i > j:
        i, j = j, i
    return int(i * (n - (i + 3) * 0.5) + j - 1)


def condensed_idx_to_squareform_idx(arr_len: int, i: int) -> Tuple[int, int]:
    """Return the row index i and column index j of the square matrix by giving the index of an element in the matrix's
    condensed form.

    Args:
        arr_len: The size of the array in condensed form.
        i: The index of the element in the condensed array.

    Returns:
        A tuple (x, y) of the row and column index of the element in sqaure form of the matrix.
    """
    n = int((1 + np.sqrt(1 + 8 * arr_len)) / 2)

    def fr(x):
        return int(x * (n - (x + 3) * 0.5) + n - 1)

    for x in range(n):
        d = fr(x) - (i + 1)
        if d >= 0:
            break
    y = n - d - 1
    return x, y


def squareform(arr: npt.ArrayLike, antisym: bool = False, **kwargs) -> npt.ArrayLike:
    """Convert the input array to a square form matrix.

    Args:
        arr: The input array.
        antisym: Whether to treat the input array as containing the lower-triangular part of a symmetric matrix, and
            set the upper-triangular elements to their negative values.
        **kwargs: Additional keyword arguments to be passed to the `spsquare` function.

    Returns:
        A square form matrix.
    """
    M = spsquare(arr, **kwargs)
    if antisym:
        tril_idx = np.tril_indices_from(M, k=-1)
        M[tril_idx] = -M[tril_idx]
    return M


def moms2var(
    m1: Union[np.ndarray, sp.csr_matrix],
    m2: Union[np.ndarray, sp.csr_matrix],
) -> Union[np.ndarray, sp.csr_matrix]:
    """Calculate the variance from the first and second moments of a distribution.

    Args:
        m1: The first moments of the distribution.
        m2: The second moments of the distribution.

    Returns:
        The variance of the distribution.
    """
    var = m2 - elem_prod(m1, m1)
    return var


def var2m2(
    var: Union[np.ndarray, sp.csr_matrix],
    m1: Union[np.ndarray, sp.csr_matrix],
) -> Union[np.ndarray, sp.csr_matrix]:
    """Calculate the second moments from the variance and first moments of a distribution.

    Args:
        var: The variance of the distribution.
        m1: The first moments of the distribution.

    Returns:
        The second moments of the distribution.
    """
    m2 = var + elem_prod(m1, m1)
    return m2


def gaussian_1d(x: npt.ArrayLike, mu: float = 0, sigma: float = 1) -> npt.ArrayLike:
    """Calculate the probability density at x with given mean and standard deviation.

    Args:
        x: The x to calculate probability density. If x is an array, the probability density would be calculated
        element-wisely.
        mu: The mean of the distribution. Defaults to 0.
        sigma: The standard deviation of the distribution. Defaults to 1.

    Returns:
        The probability density of the distribution at x.
    """

    y = (x - mu) / sigma
    return np.exp(-y * y / 2) / np.sqrt(2 * np.pi) / sigma


def timeit(method: Callable) -> Callable:
    """Wrap a Callable that if its kwargs contains "timeit" = True, measures how much time the function takes to finish.

    Args:
        method: The Callable to be measured.

    Returns:
        The wrapped Callable.
    """

    def timed(*args, **kw):
        ti = kw.pop("timeit", False)
        if ti:
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            main_info_verbose_timeit("Time elapsed for %r: %.4f s" % (method.__name__, (te - ts)))
        else:
            result = method(*args, **kw)
        return result

    return timed


def velocity_on_grid(
    X: npt.ArrayLike,
    V: npt.ArrayLike,
    n_grids: Union[int, np.ndarray],
    nbrs: Optional[NearestNeighbors] = None,
    k: Optional[int] = None,
    smoothness: int = 1,
    cutoff_coeff: int = 2,
    margin_coeff: float = 0.025,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate velocity on a grid from a given set of data points and velocities.

    Args:
        X: An array of shape representing the data points.
        V: An array of shape representing the velocities associated with the data points.
        n_grids: Number of grid points along each feature dimension.
        nbrs: A nearest neighbor model or a class that implements a nearest neighbor search.
        k: The number of nearest neighbors to consider.
        smoothness: A parameter controlling the smoothness of the velocity estimation.
        cutoff_coeff: A coefficient to control the cutoff distance for weighting the neighbors.
        margin_coeff: A coefficient to expand the grid range beyond the data points to avoid edge effects.

    Returns:
        A tuple containing the grid points X_grid and the estimated velocity on the grid V_grid.
    """
    # codes adapted from velocyto
    _, D = X.shape
    if np.isscalar(n_grids):
        n_grids *= np.ones(D, dtype=int)
    # Prepare the grid
    grs = []
    for dim_i in range(D):
        m, M = np.min(X[:, dim_i]), np.max(X[:, dim_i])
        m -= margin_coeff * np.abs(M - m)
        M += margin_coeff * np.abs(M - m)
        gr = np.linspace(m, M, n_grids[dim_i])
        grs.append(gr)
    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    if nbrs is None:
        k = 100 if k is None else k
        if X.shape[0] > 200000 and X.shape[1] > 2:
            from pynndescent import NNDescent

            nbrs = NNDescent(
                X,
                metric="euclidean",
                n_neighbors=k + 1,
                n_jobs=-1,
                random_state=19491001,
            )
        else:
            alg = "ball_tree" if X.shape[1] > 10 else "kd_tree"
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm=alg, n_jobs=-1).fit(X)

    if hasattr(nbrs, "kneighbors"):
        dists, neighs = nbrs.kneighbors(X_grid)
    elif hasattr(nbrs, "query"):
        neighs, dists = nbrs.query(X_grid, k=k + 1)

    std = np.mean([(g[1] - g[0]) for g in grs])
    # isotropic gaussian kernel
    sigma = smoothness * std
    w = gaussian_1d(dists[:, :k], sigma=sigma)
    if cutoff_coeff is not None:
        w_cut = gaussian_1d(cutoff_coeff * sigma, sigma=sigma)
        w[w < w_cut] = 0
    w_mass = w.sum(1)
    w_mass[w_mass == 0] = 1
    w = (w.T / w_mass).T

    V[np.isnan(V)] = 0
    V_grid = np.einsum("ijk, ij -> ik", V[neighs[:, :k]], w)
    return X_grid, V_grid


def argsort_mat(mat: np.ndarray, order: Literal[-1, 1] = 1) -> List[Tuple[int, int]]:
    """Sort a 2D array and return the index of sorted elements.

    Args:
        mat: The 2D array to be sort.
        order: Sort the array ascending if set to 1 and descending if set to -1. Defaults to 1.

    Returns:
        A list containing 2D index of sorted elements.
    """

    isort = np.argsort(mat, axis=None)[::order]
    index = np.zeros((len(isort), 2), dtype=int)
    index[:, 0] = isort // mat.shape[1]
    index[:, 1] = isort % mat.shape[1]
    return [(index[i, 0], index[i, 1]) for i in range(len(isort))]


def list_top_genes(
    arr: np.ndarray,
    gene_names: np.ndarray,
    n_top_genes: int = 30,
    order: Literal[1, -1] = -1,
    return_sorted_array: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """List top genes in a set gene data.

    Args:
        arr: An 1D array containing expression value of a set of genes.
        gene_names: The gene name corresponding to each value in `arr`.
        n_top_genes: The number of top genes to be selected. Defaults to 30.
        order: Could be 1 or -1. If set to 1, most expressed genes are selected; otherwise, least expressed genes are
            selected. Defaults to -1.
        return_sorted_array: Whether to return the sorted expression array together with sorted gene names. Defaults to
            False.

    Returns:
        The names of the sorted genes in the specified order. If `return_sorted_array` is set to True, the sorted
        expression array would also be returned.
    """
    imax = np.argsort(arr)[::order]
    if return_sorted_array:
        return gene_names[imax][:n_top_genes], arr[imax][:n_top_genes]
    else:
        return gene_names[imax][:n_top_genes]


def list_top_interactions(
    mat: np.ndarray, row_names: np.ndarray, column_names: np.ndarray, order: Literal[1, -1] = -1
) -> Tuple[List[List[str]], np.ndarray]:
    """Sort a 2D array with raw and column names in specified order.

    Args:
        mat: The array to be sorted.
        row_names: The name for each row of `mat`.
        column_names: The name for each column of `mat`.
        order: Could be 1 or -1. If set to 1, sort ascending. Otherwise, sort descending. Defaults to -1.

    Returns:
        A tuple (ints, sorted_mat) where `ints` is a sorted list whose elements are pairs of row name and column name
        corresponding to the element in the mat. `sorted_mat` is a list containing the sorted values of the mat.
    """
    imax = argsort_mat(mat, order=order)
    ints = []
    sorted_mat = []
    for im in imax:
        ints.append([row_names[im[0]], column_names[im[1]]])
        sorted_mat.append(mat[im])
    return ints, np.array(sorted_mat)


def table_top_genes(
    arrs: np.ndarray,
    item_names: str,
    gene_names: np.ndarray,
    return_df: bool = True,
    output_values: bool = False,
    **kwargs,
) -> Union[pd.DataFrame, dict]:
    """Sort gene expressions for multiple items (cells) and save the result in a dict or a DataFrame.

    Args:
        arrs: A 2D array with each row corresponding to an item and each column corresponding to a gene.
        item_names: The names of items corresponding to the rows of `arrs`.
        gene_names: The names of genes corresponding to the columns of `arrs`.
        return_df: Whether to return the result in DataFrame or dict. Defaults to True.
        output_values: Whether to return the genes expression value together with sorted gene names. Defaults to False.

    Returns:
        A DataFrame or a dict containing sorted genes for each item.
    """
    table = {}
    for i, item in enumerate(item_names):
        if output_values:
            table[item], table[item + "_values"] = list_top_genes(
                arrs[i], gene_names, return_sorted_array=True, **kwargs
            )
        else:
            table[item] = list_top_genes(arrs[i], gene_names, **kwargs)
    if return_df:
        return pd.DataFrame(data=table)
    else:
        return table


def table_rank_dict(
    rank_dict: dict, n_top_genes: int = 30, order: int = 1, output_values: bool = False
) -> pd.DataFrame:
    """Generate a pandas.Dataframe from a rank dictionary. A rank dictionary is a dictionary of gene names and values,
    based on which the genes are sorted, for each group of cells.

    Args:
        rank_dict: The rank dictionary.
        n_top_genes: The number of top genes put into the Dataframe. Defaults to 30.
        order: The order of picking the top genes from the rank dictionary.
            1: ascending, -1: descending. Defaults to 1.
        output_values: Whether output the values along with gene names. Defaults to False.

    Returns:
        The table of top genes of each group.
    """

    data = {}
    for g, r in rank_dict.items():
        d = [k for k in r.keys()][::order]
        data[g] = d[:n_top_genes]
        if output_values:
            dd = [v for v in r.values()][::order]
            data[g + "_values"] = dd[:n_top_genes]
    return pd.DataFrame(data=data)


# ---------------------------------------------------------------------------------------------------
# data transformation related:
def log1p_(adata: AnnData, X_data: np.ndarray) -> np.ndarray:
    """Perform log(1+x) X_data if adata.uns["pp"]["layers_norm_method"] is None.

    Args:
        adata: The AnnData that has been preprocessed.
        X_data: The data to perform log1p on.

    Returns:
        The log1p result data if "layers_norm_method" in adata is None; otherwise, X_data would be returned unchanged.
    """
    if "layers_norm_method" not in adata.uns["pp"].keys():
        return X_data
    else:
        if adata.uns["pp"]["layers_norm_method"] is None:
            if sp.issparse(X_data):
                X_data.data = np.log1p(X_data.data)
            else:
                X_data = np.log1p(X_data)

        return X_data


def inverse_norm(adata: AnnData, layer_x: Union[np.ndarray, sp.csr_matrix]) -> np.ndarray:
    """Perform inverse normalization on the given data. The normalization method is stored in adata after preprocessing.

    Args:
        adata: An AnnData object that has been preprocessed.
        layer_x: The data to perform inverse normalization on.

    Returns:
        The inverse normalized data.
    """

    if sp.issparse(layer_x):
        layer_x.data = (
            np.expm1(layer_x.data)
            if adata.uns["pp"]["layers_norm_method"] == "log1p"
            else 2**layer_x.data - 1
            if adata.uns["pp"]["layers_norm_method"] == "log2"
            else np.exp(layer_x.data) - 1
            if adata.uns["pp"]["layers_norm_method"] == "log"
            else _Freeman_Tukey(layer_x.data + 1, inverse=True) - 1
            if adata.uns["pp"]["layers_norm_method"] == "Freeman_Tukey"
            else layer_x.data
        )
    else:
        layer_x = (
            np.expm1(layer_x)
            if adata.uns["pp"]["layers_norm_method"] == "log1p"
            else 2**layer_x - 1
            if adata.uns["pp"]["layers_norm_method"] == "log2"
            else np.exp(layer_x) - 1
            if adata.uns["pp"]["layers_norm_method"] == "log"
            else _Freeman_Tukey(layer_x, inverse=True)
            if adata.uns["pp"]["layers_norm_method"] == "Freeman_Tukey"
            else layer_x
        )

    return layer_x


# ---------------------------------------------------------------------------------------------------
# kinetic parameters related:
def one_shot_alpha(labeled: npt.ArrayLike, gamma: npt.ArrayLike, t: npt.ArrayLike) -> npt.ArrayLike:
    """Calculate the alpha parameter in one-shot experiment.

    Args:
        labeled: The array of labeled data.
        gamma: The degradation rate.
        t: The time of labeling.

    Returns:
        The alpha parameter.
    """
    alpha = labeled * gamma / (1 - np.exp(-gamma * t))
    return alpha


def one_shot_alpha_matrix(
    labeled: Union[np.ndarray, sp.csr_matrix],
    gamma: Union[np.ndarray, sp.csr_matrix],
    t: Union[np.ndarray, sp.csr_matrix],
) -> sp.csr_matrix:
    """Calculate the alpha parameter in one-shot experiment with sparse matrix support.

    Args:
        labeled: The array of labeled data.
        gamma: The degradation rate.
        t: The time of labeling.

    Returns:
        The alpha parameter.
    """
    alpha = elem_prod(gamma[:, None], labeled) / (1 - np.exp(-elem_prod(gamma[:, None], t[None, :])))
    return sp.csr_matrix(alpha)


def one_shot_gamma_alpha(k: np.ndarray, t: np.ndarray, labeled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the gamma and alpha parameters in one-shot experiment.

    Args:
        k: The slope of labeled and total data under the steady state.
        t: The time of labeling.
        labeled: The array of labeled data.

    Returns:
        A tuple containing the gamma and alpha parameters.
    """
    gamma = -np.log(1 - k) / t
    alpha = labeled * (gamma / k)[0]
    return gamma, alpha


def one_shot_k(gamma: npt.ArrayLike, t: npt.ArrayLike) -> npt.ArrayLike:
    """Calculate the slope of labeled and total data from the gamma and time information.

    Args:
        gamma: The degradation rate.
        t: The time of labeling.

    Returns:
        The slope calculated from gamma and t.
    """
    k = 1 - np.exp(-gamma * t)
    return k


def one_shot_gamma_alpha_matrix(k: np.ndarray, t: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the gamma and alpha parameters in one-shot experiment. Assume U is a sparse matrix and only tested on
    one-shot experiment.

    Args:
        k: The slope of labeled and total data under the steady state.
        t: The time of labeling.
        U: The sparse matrix data.

    Returns:
        A tuple containing the gamma and alpha parameters.
    """
    Kc = np.clip(k, 0, 1 - 1e-3)
    gamma = -(np.log(1 - Kc) / t)
    alpha = U.multiply((gamma / k)[:, None])

    return gamma, alpha


def _one_shot_gamma_alpha_matrix(
    K: npt.ArrayLike,
    tau: Union[float, int, np.ndarray],
    N: Union[np.ndarray, sp.csr_matrix],
    R: Union[np.ndarray, sp.csr_matrix],
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the gamma and alpha parameters in one-shot experiment.

    Supports sparse matrix input. Original code from Yan.

    Args:
        K: The slope of labeled and total data under the steady state.
        tau: The time of labeling.
        N: The sparse matrix of labeled data.
        R: The sparse matrix of total data.

    Returns:
        A tuple containing the gamma and alpha parameters.
    """
    N, R = N.A.T, R.A.T
    K = np.array(K)
    tau = tau[0]
    Kc = np.clip(K, 0, 1 - 1e-3)
    if np.isscalar(tau):
        B = -np.log(1 - Kc) / tau
    else:
        B = -(np.log(1 - Kc)[None, :].T / tau).T
    return B, (elem_prod(B, N) / K).T - elem_prod(B, R).T


def compute_velocity_labeling_B(
    B: Union[np.ndarray, sp.csr_matrix],
    alpha: Union[np.ndarray, sp.csr_matrix],
    R: Union[np.ndarray, sp.csr_matrix],
) -> Union[np.ndarray, sp.csr_matrix]:
    """Calculate the velocity from the alpha and parameter B representing the degradation rate.

    Args:
        B: The parameter representing the degradation rate.
        alpha: The alpha parameter.
        R: The total data.

    Returns:
        The velocity calculated from the alpha and B parameters.
    """
    return alpha - elem_prod(B, R.T).T


# ---------------------------------------------------------------------------------------------------
# dynamics related:
def remove_2nd_moments(adata: AnnData) -> None:
    """Delete layers of 2nd moments.

    Args:
        adata: The AnnData object from which 2nd moment layers are deleted.
    """
    layers = list(adata.layers.keys())
    for layer in layers:
        if layer.startswith("M_") and len(layer) == 4:
            del adata.layers[layer]


def get_valid_bools(adata: AnnData, filter_gene_mode: Literal["final", "basic", "no"]) -> np.ndarray:
    """Get a boolean array showing the gene passing the filter specified.

    Args:
        adata: An AnnData object.
        filter_gene_mode: The gene filter. Could be one of "final", "basic", and "no".

    Raises:
        NotImplementedError: Invalid `filter_gene_mode`.

    Returns:
        A boolean array showing the gene passing the filter specified.
    """
    if filter_gene_mode == "final":
        valid_ind = adata.var.use_for_pca.values
    elif filter_gene_mode == "basic":
        valid_ind = adata.var.pass_basic_filter.values
    elif filter_gene_mode == "no":
        valid_ind = np.repeat([True], adata.shape[1])
    else:
        raise NotImplementedError("Invalid filter_gene_mode. ")
    return valid_ind


def log_unnormalized_data(
    raw: Union[np.ndarray, sp.csr_matrix], log_unnormalized: bool
) -> Union[np.ndarray, sp.csr_matrix]:
    """Perform log1p on unnormalized data.

    Args:
        raw: The matrix to be operated on.
        log_unnormalized: Whether the matrix is unnormalized and log1p should be performed.

    Returns:
        Updated matrix with log1p if it is unormalized; otherwise, return original `raw`.
    """
    if sp.issparse(raw):
        raw.data = np.log(raw.data + 1) if log_unnormalized else raw.data
    else:
        raw = np.log1p(raw) if log_unnormalized else raw

    return raw


def get_data_for_kin_params_estimation(
    subset_adata: AnnData,
    has_splicing: bool,
    has_labeling: bool,
    model: str,
    use_moments: bool,
    tkey: str,
    protein_names: List[str],
    log_unnormalized: bool,
    NTR_vel: bool,
) -> Tuple:
    """Get the data for kinetic experiments parameters estimation.

    Args:
        subset_adata: the AnnData object containing the data.
        has_splicing: whether the data has splicing information.
        has_labeling: whether the data has labeling information.
        model: the model used to estimate kinetic parameters.
        use_moments: whether to use the moments data instead of the original layer.
        tkey: the key in `adata.obs` that stores the time information.
        protein_names: the names of proteins to perform estimation.
        log_unnormalized: whether the data needs log normalization.
        NTR_vel: whether to use new / total ratio (NTR) to estimate velocity.

    Returns:
        A tuple containing the data for kinetic experiments parameters estimation.
    """
    if not NTR_vel:
        if has_labeling and not has_splicing:
            main_warning(
                "Your adata only has labeling data, but `NTR_vel` is set to be "
                "`False`. Dynamo will reset it to `True` to enable this analysis."
            )
        NTR_vel = True

    U, Ul, S, Sl, P, US, U2, S2, = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )  # U: (unlabeled) unspliced; S: (unlabeled) spliced; U / Ul: old and labeled; U, Ul, S, Sl: uu/ul/su/sl
    normalized, assumption_mRNA = (
        False,
        None,
    )

    mapper = get_mapper()

    # labeling plus splicing
    if np.all(([i in subset_adata.layers.keys() for i in ["X_ul", "X_sl", "X_su"]])) or np.all(
        ([mapper[i] in subset_adata.layers.keys() for i in ["X_ul", "X_sl", "X_su"]])
    ):  # only uu, ul, su, sl provided
        normalized, assumption_mRNA = (
            True,
            "ss" if NTR_vel else "kinetic",
        )
        U = (
            subset_adata.layers[mapper["X_uu"]].T if use_moments else subset_adata.layers["X_uu"].T
        )  # unlabel unspliced: U

        Ul = subset_adata.layers[mapper["X_ul"]].T if use_moments else subset_adata.layers["X_ul"].T

        Sl = subset_adata.layers[mapper["X_sl"]].T if use_moments else subset_adata.layers["X_sl"].T

        S = (
            subset_adata.layers[mapper["X_su"]].T if use_moments else subset_adata.layers["X_su"].T
        )  # unlabel spliced: S

    elif np.all(([i in subset_adata.layers.keys() for i in ["uu", "ul", "sl", "su"]])):
        normalized, assumption_mRNA = (
            False,
            "ss" if NTR_vel else "kinetic",
        )
        raw, _ = subset_adata.layers["uu"].T, subset_adata.layers["uu"].T
        U = log_unnormalized_data(raw, log_unnormalized)

        raw, _ = subset_adata.layers["ul"].T, subset_adata.layers["ul"].T
        Ul = log_unnormalized_data(raw, log_unnormalized)

        raw, _ = subset_adata.layers["sl"].T, subset_adata.layers["sl"].T
        Sl = log_unnormalized_data(raw, log_unnormalized)

        raw, _ = subset_adata.layers["su"].T, subset_adata.layers["su"].T
        S = log_unnormalized_data(raw, log_unnormalized)

    # labeling without splicing
    if not has_splicing and (
        ("X_new" in subset_adata.layers.keys() and not use_moments)
        or (mapper["X_new"] in subset_adata.layers.keys() and use_moments)
    ):  # run new / total ratio (NTR)
        normalized, assumption_mRNA = (
            True,
            "ss" if NTR_vel else "kinetic",
        )
        U = (
            subset_adata.layers[mapper["X_total"]].T - subset_adata.layers[mapper["X_new"]].T
            if use_moments
            else subset_adata.layers["X_total"].T - subset_adata.layers["X_new"].T
        )
        Ul = subset_adata.layers[mapper["X_new"]].T if use_moments else subset_adata.layers["X_new"].T

    elif not has_splicing and "new" in subset_adata.layers.keys():
        assumption_mRNA = ("ss" if NTR_vel else "kinetic",)
        raw, _, old = (
            subset_adata.layers["new"].T,
            subset_adata.layers["new"].T,
            subset_adata.layers["total"].T - subset_adata.layers["new"].T,
        )
        if sp.issparse(raw):
            raw.data = np.log1p(raw.data) if log_unnormalized else raw.data
            old.data = np.log1p(old.data) if log_unnormalized else old.data
        else:
            raw = np.log1p(raw) if log_unnormalized else raw
            old = np.log1p(old) if log_unnormalized else old
        U = old
        Ul = raw

    # splicing data
    if not has_labeling and (
        ("X_unspliced" in subset_adata.layers.keys() and not use_moments)
        or (mapper["X_unspliced"] in subset_adata.layers.keys() and use_moments)
    ):
        normalized, assumption_mRNA = (
            True,
            "kinetic" if tkey in subset_adata.obs.columns else "ss",
        )
        U = subset_adata.layers[mapper["X_unspliced"]].T if use_moments else subset_adata.layers["X_unspliced"].T
    elif not has_labeling and "unspliced" in subset_adata.layers.keys():
        assumption_mRNA = "kinetic" if tkey in subset_adata.obs.columns else "ss"
        raw, _ = (
            subset_adata.layers["unspliced"].T,
            subset_adata.layers["unspliced"].T,
        )
        if sp.issparse(raw):
            raw.data = np.log1p(raw.data) if log_unnormalized else raw.data
        else:
            raw = np.log1p(raw) if log_unnormalized else raw
        U = raw
    if not has_labeling and (
        ("X_spliced" in subset_adata.layers.keys() and not use_moments)
        or (mapper["X_spliced"] in subset_adata.layers.keys() and use_moments)
    ):
        S = subset_adata.layers[mapper["X_spliced"]].T if use_moments else subset_adata.layers["X_spliced"].T
    elif not has_labeling and "spliced" in subset_adata.layers.keys():
        raw, _ = (
            subset_adata.layers["spliced"].T,
            subset_adata.layers["spliced"].T,
        )
        if sp.issparse(raw):
            raw.data = np.log1p(raw.data) if log_unnormalized else raw.data
        else:
            raw = np.log1p(raw) if log_unnormalized else raw
        S = raw

    # protein
    ind_for_proteins = None
    if ("X_protein" in subset_adata.obsm.keys() and not use_moments) or (
        mapper["X_protein"] in subset_adata.obsm.keys() and use_moments
    ):
        P = subset_adata.obsm[mapper["X_protein"]].T if use_moments else subset_adata.obsm["X_protein"].T
    elif "protein" in subset_adata.obsm.keys():
        P = subset_adata.obsm["protein"].T
    if P is not None:
        if protein_names is None:
            main_warning(
                "protein layer exists but protein_names is not provided. No estimation will be performed for protein "
                "data."
            )
        else:
            protein_names = list(set(subset_adata.var.index).intersection(protein_names))
            ind_for_proteins = [np.where(subset_adata.var.index == i)[0][0] for i in protein_names]
            subset_adata.var["is_protein_dynamics_genes"] = False
            subset_adata.var.loc[ind_for_proteins, "is_protein_dynamics_genes"] = True

    if has_labeling:
        if assumption_mRNA is None:
            assumption_mRNA = "ss" if NTR_vel else "kinetic"
        if tkey in subset_adata.obs.columns:
            t = np.array(subset_adata.obs[tkey], dtype="float")
        else:
            raise Exception(
                "the tkey ",
                tkey,
                " provided is not a valid column name in .obs.",
            )
        if model == "stochastic" and all([x in subset_adata.layers.keys() for x in ["M_tn", "M_nn", "M_tt"]]):
            US, U2, S2 = (
                subset_adata.layers["M_tn"].T if NTR_vel else subset_adata.layers["M_us"].T,
                subset_adata.layers["M_nn"].T if NTR_vel else subset_adata.layers["M_uu"].T,
                subset_adata.layers["M_tt"].T if NTR_vel else subset_adata.layers["M_ss"].T,
            )
    else:
        t = None
        if model == "stochastic":
            US, U2, S2 = (
                subset_adata.layers["M_us"].T,
                subset_adata.layers["M_uu"].T,
                subset_adata.layers["M_ss"].T,
            )

    return (
        U,
        Ul,
        S,
        Sl,
        P,
        US,
        U2,
        S2,
        t,
        normalized,
        ind_for_proteins,
        assumption_mRNA,
    )


def set_velocity(
    adata,
    vel_U,
    vel_S,
    vel_N,
    vel_T,
    vel_P,
    _group,
    cur_grp,
    cur_cells_bools,
    valid_ind,
    ind_for_proteins,
):
    cur_cells_ind, valid_ind_ = (
        np.where(cur_cells_bools)[0][:, np.newaxis],
        np.where(valid_ind)[0],
    )
    if type(vel_U) is not float:
        if cur_grp == _group[0]:
            adata.layers["velocity_U"] = sp.csr_matrix((adata.shape), dtype=np.float64)
        vel_U = vel_U.T.tocsr() if sp.issparse(vel_U) else sp.csr_matrix(vel_U, dtype=np.float64).T
        adata.layers["velocity_U"][cur_cells_ind, valid_ind_] = vel_U
    if type(vel_S) is not float:
        if cur_grp == _group[0]:
            adata.layers["velocity_S"] = sp.csr_matrix((adata.shape), dtype=np.float64)
        vel_S = vel_S.T.tocsr() if sp.issparse(vel_S) else sp.csr_matrix(vel_S, dtype=np.float64).T
        adata.layers["velocity_S"][cur_cells_ind, valid_ind_] = vel_S
    if type(vel_N) is not float:
        if cur_grp == _group[0]:
            adata.layers["velocity_N"] = sp.csr_matrix((adata.shape), dtype=np.float64)
        vel_N = vel_N.T.tocsr() if sp.issparse(vel_N) else sp.csr_matrix(vel_N, dtype=np.float64).T
        adata.layers["velocity_N"][cur_cells_ind, valid_ind_] = vel_N
    if type(vel_T) is not float:
        if cur_grp == _group[0]:
            adata.layers["velocity_T"] = sp.csr_matrix((adata.shape), dtype=np.float64)
        vel_T = vel_T.T.tocsr() if sp.issparse(vel_T) else sp.csr_matrix(vel_T, dtype=np.float64).T
        adata.layers["velocity_T"][cur_cells_ind, valid_ind_] = vel_T
    if type(vel_P) is not float:
        if cur_grp == _group[0]:
            adata.obsm["velocity_P"] = sp.csr_matrix((adata.obsm["P"].shape[0], len(ind_for_proteins)), dtype=float)
        adata.obsm["velocity_P"][cur_cells_bools, :] = (
            vel_P.T.tocsr() if sp.issparse(vel_P) else sp.csr_matrix(vel_P, dtype=float).T
        )

    return adata


def set_param_ss(
    adata,
    est,
    alpha,
    beta,
    gamma,
    eta,
    delta,
    experiment_type,
    _group,
    cur_grp,
    kin_param_pre,
    valid_ind,
    ind_for_proteins,
):
    params_df = pd.DataFrame(index=adata.var.index)
    if experiment_type == "mix_std_stm":
        if alpha is not None:
            if cur_grp == _group[0]:
                adata.varm[kin_param_pre + "alpha"] = np.zeros((adata.shape[1], alpha[1].shape[1]))
            adata.varm[kin_param_pre + "alpha"][valid_ind, :] = alpha[1]
            (
                params_df[kin_param_pre + "alpha"],
                params_df[kin_param_pre + "alpha_std"],
            ) = (None, None)
            (
                params_df.loc[valid_ind, kin_param_pre + "alpha"],
                params_df.loc[valid_ind, kin_param_pre + "alpha_std"],
            ) = (alpha[1][:, -1], alpha[0])

        if cur_grp == _group[0]:
            (
                params_df[kin_param_pre + "beta"],
                params_df[kin_param_pre + "gamma"],
                params_df[kin_param_pre + "half_life"],
            ) = (None, None, None)

        params_df.loc[valid_ind, kin_param_pre + "beta"] = beta
        params_df.loc[valid_ind, kin_param_pre + "gamma"] = gamma
        params_df.loc[valid_ind, kin_param_pre + "half_life"] = np.log(2) / gamma
    else:
        if alpha is not None:
            if len(alpha.shape) > 1:  # for each cell
                if cur_grp == _group[0]:
                    adata.varm[kin_param_pre + "alpha"] = (
                        sp.csr_matrix(np.zeros(adata.shape[::-1]))
                        if sp.issparse(alpha)
                        else np.zeros(adata.shape[::-1])
                    )  #
                adata.varm[kin_param_pre + "alpha"][valid_ind, :] = alpha  #
                params_df.loc[valid_ind, kin_param_pre + "alpha"] = alpha.mean(1)
            elif len(alpha.shape) == 1:
                if cur_grp == _group[0]:
                    params_df[kin_param_pre + "alpha"] = None
                params_df.loc[valid_ind, kin_param_pre + "alpha"] = alpha

        if cur_grp == _group[0]:
            (
                params_df[kin_param_pre + "beta"],
                params_df[kin_param_pre + "gamma"],
                params_df[kin_param_pre + "half_life"],
            ) = (None, None, None)
        params_df.loc[valid_ind, kin_param_pre + "beta"] = beta
        params_df.loc[valid_ind, kin_param_pre + "gamma"] = gamma
        params_df.loc[valid_ind, kin_param_pre + "half_life"] = None if gamma is None else np.log(2) / gamma

        (
            alpha_intercept,
            alpha_r2,
            beta_k,
            gamma_k,
            gamma_intercept,
            gamma_r2,
            gamma_logLL,
            delta_intercept,
            delta_r2,
            bs,
            bf,
            uu0,
            ul0,
            su0,
            sl0,
            U0,
            S0,
            total0,
        ) = est.aux_param.values()
        if alpha_r2 is not None:
            alpha_r2[~np.isfinite(alpha_r2)] = 0
        if cur_grp == _group[0]:
            (
                params_df[kin_param_pre + "alpha_b"],
                params_df[kin_param_pre + "alpha_r2"],
                params_df[kin_param_pre + "gamma_b"],
                params_df[kin_param_pre + "gamma_r2"],
                params_df[kin_param_pre + "gamma_logLL"],
                params_df[kin_param_pre + "delta_b"],
                params_df[kin_param_pre + "delta_r2"],
                params_df[kin_param_pre + "bs"],
                params_df[kin_param_pre + "bf"],
                params_df[kin_param_pre + "uu0"],
                params_df[kin_param_pre + "ul0"],
                params_df[kin_param_pre + "su0"],
                params_df[kin_param_pre + "sl0"],
                params_df[kin_param_pre + "U0"],
                params_df[kin_param_pre + "S0"],
                params_df[kin_param_pre + "total0"],
            ) = (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        params_df.loc[valid_ind, kin_param_pre + "alpha_b"] = alpha_intercept
        params_df.loc[valid_ind, kin_param_pre + "alpha_r2"] = alpha_r2

        if gamma_r2 is not None:
            gamma_r2[~np.isfinite(gamma_r2)] = 0
        params_df.loc[valid_ind, kin_param_pre + "gamma_b"] = gamma_intercept
        params_df.loc[valid_ind, kin_param_pre + "gamma_r2"] = gamma_r2
        params_df.loc[valid_ind, kin_param_pre + "gamma_logLL"] = gamma_logLL

        params_df.loc[valid_ind, kin_param_pre + "bs"] = bs
        params_df.loc[valid_ind, kin_param_pre + "bf"] = bf

        params_df.loc[valid_ind, kin_param_pre + "uu0"] = uu0
        params_df.loc[valid_ind, kin_param_pre + "ul0"] = ul0
        params_df.loc[valid_ind, kin_param_pre + "su0"] = su0
        params_df.loc[valid_ind, kin_param_pre + "sl0"] = sl0
        params_df.loc[valid_ind, kin_param_pre + "U0"] = U0
        params_df.loc[valid_ind, kin_param_pre + "S0"] = S0
        params_df.loc[valid_ind, kin_param_pre + "total0"] = total0

        if experiment_type == "one-shot":
            params_df[kin_param_pre + "beta_k"] = None
            params_df[kin_param_pre + "gamma_k"] = None
            params_df.loc[valid_ind, kin_param_pre + "beta_k"] = beta_k
            params_df.loc[valid_ind, kin_param_pre + "gamma_k"] = gamma_k

        if ind_for_proteins is not None:
            delta_r2[~np.isfinite(delta_r2)] = 0
            if cur_grp == _group[0]:
                (
                    params_df[kin_param_pre + "eta"],
                    params_df[kin_param_pre + "delta"],
                    params_df[kin_param_pre + "delta_b"],
                    params_df[kin_param_pre + "delta_r2"],
                    params_df[kin_param_pre + "p_half_life"],
                ) = (None, None, None, None, None)
            params_df.loc[valid_ind, kin_param_pre + "eta"][ind_for_proteins] = eta
            params_df.loc[valid_ind, kin_param_pre + "delta"][ind_for_proteins] = delta
            params_df.loc[valid_ind, kin_param_pre + "delta_b"][ind_for_proteins] = delta_intercept
            params_df.loc[valid_ind, kin_param_pre + "delta_r2"][ind_for_proteins] = delta_r2
            params_df.loc[valid_ind, kin_param_pre + "p_half_life"][ind_for_proteins] = np.log(2) / delta

    adata.varm[kin_param_pre + "vel_params"] = params_df.to_numpy().astype(float)
    adata.uns[kin_param_pre + "vel_params_names"] = list(params_df.columns)

    return adata


def set_param_kinetic(
    adata,
    alpha,
    a,
    b,
    alpha_a,
    alpha_i,
    beta,
    gamma,
    cost,
    logLL,
    kin_param_pre,
    extra_params,
    _group,
    cur_grp,
    cur_cells_bools,
    valid_ind,
):
    params_df = pd.DataFrame(index=adata.var.index)
    if cur_grp == _group[0]:
        (
            params_df[kin_param_pre + "alpha"],
            params_df[kin_param_pre + "a"],
            params_df[kin_param_pre + "b"],
            params_df[kin_param_pre + "alpha_a"],
            params_df[kin_param_pre + "alpha_i"],
            params_df[kin_param_pre + "beta"],
            params_df[kin_param_pre + "p_half_life"],
            params_df[kin_param_pre + "gamma"],
            params_df[kin_param_pre + "half_life"],
            params_df[kin_param_pre + "cost"],
            params_df[kin_param_pre + "logLL"],
        ) = (None, None, None, None, None, None, None, None, None, None, None)

    if isarray(alpha) and alpha.ndim > 1:
        params_df.loc[valid_ind, kin_param_pre + "alpha"] = (
            np.asarray(alpha.mean(1))
            if sp.issparse(alpha)
            else alpha.mean(1)
        )
        cur_cells_ind, valid_ind_ = (
            np.where(cur_cells_bools)[0][:, np.newaxis],
            np.where(valid_ind)[0],
        )
        if cur_grp == _group[0]:
            adata.layers["cell_wise_alpha"] = sp.csr_matrix((adata.shape), dtype=np.float64)
        alpha = alpha.T.tocsr() if sp.issparse(alpha) else sp.csr_matrix(alpha, dtype=np.float64).T
        adata.layers["cell_wise_alpha"][cur_cells_ind, valid_ind_] = alpha
    else:
        params_df.loc[valid_ind, kin_param_pre + "alpha"] = alpha
    params_df.loc[valid_ind, kin_param_pre + "a"] = a
    params_df.loc[valid_ind, kin_param_pre + "b"] = b
    params_df.loc[valid_ind, kin_param_pre + "alpha_a"] = alpha_a
    params_df.loc[valid_ind, kin_param_pre + "alpha_i"] = alpha_i
    params_df.loc[valid_ind, kin_param_pre + "beta"] = beta
    params_df.loc[valid_ind, kin_param_pre + "gamma"] = gamma
    params_df.loc[valid_ind, kin_param_pre + "half_life"] = np.log(2) / gamma
    params_df.loc[valid_ind, kin_param_pre + "cost"] = cost
    params_df.loc[valid_ind, kin_param_pre + "logLL"] = logLL
    # add extra parameters (u0, uu0, etc.)
    extra_params.columns = [kin_param_pre + i for i in extra_params.columns]
    extra_params = extra_params.set_index(adata.var.index[valid_ind])
    var = pd.concat((params_df, extra_params), axis=1, sort=False)
    adata.varm[kin_param_pre + "vel_params"] = var.to_numpy().astype(float)
    adata.uns[kin_param_pre + "vel_params_names"] = list(var.columns)

    return adata


def get_vel_params(
    adata: AnnData,
    params: Optional[Union[List, str]] = None,
    genes: Optional[List] = None,
    kin_param_pre: str = "",
    skip_cell_wise: bool = False,
) -> Union[Tuple, pd.DataFrame, List]:
    """Get the velocity parameters based on input names.

    Args:
        adata: The anndata object which contains the parameters.
        params: The names of parameters to query. If set to None, the entire velocity parameters DataFrame from `.varm`
            will be returned.
        kin_param_pre: The prefix used to kinetic parameters related to RNA dynamics.
        skip_cell_wise: Whether to skip the detected cell wise parameters. If set to True, the mean will be returned
            instead of cell wise parameters.

    Returns:
        All velocity parameters with the same order of query `params`.
    """
    if type(params) is str:
        params = [params]

    if kin_param_pre + "vel_params" not in adata.varm.keys():
        raise KeyError("The key of velocity related parameters are not found in varm.")

    array_data = adata.varm[kin_param_pre + "vel_params"]
    df_columns = adata.uns[kin_param_pre + "vel_params_names"]
    df = pd.DataFrame(array_data, index=adata.var_names, columns=df_columns)
    target_params = []

    if genes is None:
        genes = df.index

    if params is None:
        return df.loc[genes]

    for param in params:
        if param == "alpha":
            if not skip_cell_wise:
                if "cell_wise_alpha" in adata.layers.keys():
                    target_params.append(adata[:, genes].layers["cell_wise_alpha"])
                elif "alpha" in adata.varm.keys():
                    target_params.append(adata[:, genes].varm[kin_param_pre + "alpha"])
                else:
                    target_params.append(df.loc[genes, kin_param_pre + "alpha"].values)
                continue
        target_params.append(df.loc[genes, kin_param_pre + param].values)

    if len(target_params) > 1:
        return tuple(target_params)
    else:
        return target_params[0]


def update_vel_params(adata: AnnData, params_df: pd.DataFrame, kin_param_pre: str = "") -> None:
    """Update the kinetic parameters related to RNA velocity calculation.

    Args:
        adata: The AnnData object whose kinetic parameters related to RNA velocity calculation will be updated.
        params_df: The dataframe of kinetic parameters related to RNA velocity calculation.
        kin_param_pre: The prefix used to kinetic parameters related to RNA dynamics.

    Returns:
        The anndata object will be updated with parameters and columns names from given dataframe.
    """
    adata.varm[kin_param_pre + "vel_params"] = params_df.to_numpy()
    adata.uns[kin_param_pre + "vel_params_names"] = list(params_df.columns)


def get_U_S_for_velocity_estimation(
    subset_adata: AnnData,
    use_moments: bool,
    has_splicing: bool,
    has_labeling: bool,
    log_unnormalized: bool,
    NTR: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the unspliced and spliced matrices for velocity estimation.

    Args:
        subset_adata: The anndata object which contains the data.
        use_moments: Whether to use the moments data instead of the original data.
        has_splicing: Whether the data has splicing information.
        has_labeling: Whether the data has labeling information.
        log_unnormalized: Whether the data needs log normalization.
        NTR: Whether to use new, total instead of unspliced, spliced.

    Returns:
        The unspliced and spliced matrices for velocity estimation.
    """
    mapper = get_mapper()

    if has_splicing:
        if has_labeling:
            if "X_new" in subset_adata.layers.keys():  # unlabel spliced: S
                if use_moments:
                    U, S = (
                        subset_adata.layers[mapper["X_unspliced"]].T,
                        subset_adata.layers[mapper["X_spliced"]].T,
                    )
                    N, T = (
                        subset_adata.layers[mapper["X_new"]].T,
                        subset_adata.layers[mapper["X_total"]].T,
                    )
                else:
                    U, S = (
                        subset_adata.layers["X_unspliced"].T,
                        subset_adata.layers["X_spliced"].T,
                    )
                    N, T = (
                        subset_adata.layers["X_new"].T,
                        subset_adata.layers["X_total"].T,
                    )
            else:
                U, S = (
                    subset_adata.layers["unspliced"].T,
                    subset_adata.layers["spliced"].T,
                )
                N, T = (
                    subset_adata.layers["new"].T,
                    subset_adata.layers["total"].T,
                )
                if sp.issparse(U):
                    U.data, S.data = (
                        np.log1p(U.data) if log_unnormalized else U.data,
                        np.log1p(S.data) if log_unnormalized else S.data,
                    )
                    N.data, T.data = (
                        np.log1p(N.data) if log_unnormalized else N.data,
                        np.log1p(T.data) if log_unnormalized else T.data,
                    )
                else:
                    U, S = (
                        np.log1p(U) if log_unnormalized else U,
                        np.log1p(S) if log_unnormalized else S,
                    )
                    N, T = (
                        np.log1p(N) if log_unnormalized else N,
                        np.log1p(T) if log_unnormalized else T,
                    )
            U, S = (N, T) if NTR else (U, S)
        else:
            if ("X_unspliced" in subset_adata.layers.keys()) or (
                mapper["X_unspliced"] in subset_adata.layers.keys()
            ):  # unlabel spliced: S
                if use_moments:
                    U, S = (
                        subset_adata.layers[mapper["X_unspliced"]].T,
                        subset_adata.layers[mapper["X_spliced"]].T,
                    )
                else:
                    U, S = (
                        subset_adata.layers["X_unspliced"].T,
                        subset_adata.layers["X_spliced"].T,
                    )
            else:
                U, S = (
                    subset_adata.layers["unspliced"].T,
                    subset_adata.layers["spliced"].T,
                )
                if sp.issparse(U):
                    U.data = np.log1p(U.data) if log_unnormalized else U.data
                    S.data = np.log1p(S.data) if log_unnormalized else S.data
                else:
                    U = np.log1p(U) if log_unnormalized else U
                    S = np.log1p(S) if log_unnormalized else S
    else:
        if ("X_new" in subset_adata.layers.keys()) or (
            mapper["X_new"] in subset_adata.layers.keys()
        ):  # run new / total ratio (NTR)
            if use_moments:
                U = subset_adata.layers[mapper["X_new"]].T
                S = (
                    subset_adata.layers[mapper["X_total"]].T
                    # if NTR
                    # else subset_adata.layers[mapper["X_total"]].T
                    # - subset_adata.layers[mapper["X_new"]].T
                )
            else:
                U = subset_adata.layers["X_new"].T
                S = (
                    subset_adata.layers["X_total"].T
                    # if NTR
                    # else subset_adata.layers["X_total"].T
                    # - subset_adata.layers["X_new"].T
                )
        elif "new" in subset_adata.layers.keys():
            U = subset_adata.layers["new"].T
            S = (
                subset_adata.layers["total"].T
                # if NTR
                # else subset_adata.layers["total"].T - subset_adata.layers["new"].T
            )
            if sp.issparse(U):
                U.data = np.log1p(U.data) if log_unnormalized else U.data
                S.data = np.log1p(S.data) if log_unnormalized else S.data
            else:
                U = np.log1p(U) if log_unnormalized else U
                S = np.log1p(S) if log_unnormalized else S

    return U, S


# ---------------------------------------------------------------------------------------------------
# retrieving data related


def fetch_X_data(adata: AnnData, genes: List, layer: str, basis: Optional[str] = None) -> Tuple:
    """Get the X data according to given parameters.

    Args:
        adata: Anndata object containing gene expression data.
        genes: List of gene names to be fetched. If None, all genes are considered.
        layer: Layer of the data to fetch.
        basis: Dimensionality reduction basis. If provided, the data is fetched from a specific embedding.

    Returns:
        A tuple containing a list of fetched gene names and the corresponding gene expression data (X data).
    """
    if basis is not None:
        return None, adata.obsm["X_" + basis]

    if genes is not None:
        genes = adata.var_names.intersection(genes).to_list()
        if len(genes) == 0:
            raise ValueError("No genes from your genes list appear in your adata object.")

    if layer is None:
        if genes is not None:
            X_data = adata[:, genes].X
        else:
            if "use_for_dynamics" not in adata.var.keys():
                X_data = adata.X
                genes = adata.var_names
            else:
                X_data = adata[:, adata.var.use_for_dynamics].X
                genes = adata.var_names[adata.var.use_for_dynamics]
    else:
        if genes is not None:
            X_data = adata[:, genes].layers[layer]
        else:
            if "use_for_dynamics" not in adata.var.keys():
                X_data = adata.layers[layer]
                genes = adata.var_names
            else:
                X_data = adata[:, adata.var.use_for_dynamics].layers[layer]
                genes = adata.var_names[adata.var.use_for_dynamics]

            X_data = log1p_(adata, X_data)

    return genes, X_data


class AnnDataPredicate(object):
    """The predicate class for item selection for anndata.

    Attributes:
        key: The key in the AnnData object (specified as `data` in `.check()`) that will be used for selection.
        value: The value that will be used based on `op` to select items.
        op: Operators for selection.
    """

    def __init__(
        self,
        key: str,
        value: Any,
        op: Literal[
            "==",
            "!=",
            ">",
            "<",
            ">=",
            "<=",
        ] = "==",
    ) -> None:
        """Initialize an AnnDataPredicate object.

        Args:
            key: The key in the AnnData object (specified as `data` in `.check()`) that will be used for selection.
            value: The value that will be used based on `op` to select items.
            op: Operators for selection:
                '==' - equal to `value`; '!=' - unequal to; '>' - greater than; '<' - smaller than;
                '>=' - greater than or equal to; '<=' - less than or equal to. Defaults to "==".
        """

        self.key = key
        self.value = value
        self.op = op

    def check(self, data: AnnData) -> np.ndarray:
        """Filter out the elements in data[AnnDataPredicate.key] that fullfill the requirement specified as
        AnnDataPredicate's `value` and `op` attr.

        Args:
            data: The AnnData object to be tested.

        Raises:
            NotImplementedError: Invalid `op`.

        Returns:
            A boolean array with `True` at positions where the element pass the check.
        """
        if self.op == "==":
            return data[self.key] == self.value
        elif self.op == "!=":
            return data[self.key] != self.value
        elif self.op == ">":
            return data[self.key] > self.value
        elif self.op == "<":
            return data[self.key] < self.value
        elif self.op == ">=":
            return data[self.key] >= self.value
        elif self.op == "<=":
            return data[self.key] <= self.value
        else:
            raise NotImplementedError(f"Unidentified operator {self.op}!")

    def __or__(self, other):
        """Combine requirement from another AnnDataPredicate object and return an AnnDataPredicates that set True on
        elements that pass at least one requirement from the AnnDataPredicate objects.

        Args:
            other (AnnDataPredicate): Another AnnDataPredicate object containing requirement for "or" test.

        Returns:
            PredicateUnion: the updated Predicates object.
        """
        return PredicateUnion(self, other)

    def __and__(self, other):
        """Combine requirement from another AnnDataPredicate object and return an AnnDataPredicates that set True on
        elements that pass all requirements from the AnnDataPredicate objects.

        Args:
            other (AnnDataPredicate): Another AnnDataPredicate object containing requirement for "and" test.

        Returns:
            PredicateIntersection: the updated Predicates object.
        """
        return PredicateIntersection(self, other)

    def __invert__(self):
        """Inverse the current requirement.

        Raises:
            NotImplementedError: Invalid `op`.

        Returns:
            AnnDataPredicate: the updated Predicate object.
        """
        if self.op == "==":
            op = "!="
        elif self.op == "!=":
            op = "=="
        elif self.op == ">":
            op = "<="
        elif self.op == "<":
            op = ">="
        elif self.op == ">=":
            op = "<"
        elif self.op == "<=":
            op = ">"
        else:
            raise NotImplementedError(f"Unidentified operator {self.op}!")

        return AnnDataPredicate(self.key, self.value, op)


class AlwaysTrue(AnnDataPredicate):
    """A class inherited from AnnDataPredicate. Will return true for all elements under any requirement.

    Attributes:
        key: The key in the AnnData object (specified as `data` in `.check()`) that will be used for selection.
    """

    def __init__(self, key: Optional[str] = None) -> None:
        """Initialize an AlwaysTrue object.

        Args:
            key: The key in the AnnData object (specified as `data` in `.check()`) that will be used for selection. If
                None, the first layer of the AnnData for comparison would be used. Defaults to None.
        """
        self.key = key

    def check(self, data: AnnData) -> np.ndarray:
        """Return an all-true boolean array with the shape of the layer.

        Args:
            data: The AnnData object to be tested.

        Returns:
            An all-true boolean array with the shape of the layer.
        """
        key = self.key if self.key is not None else data.keys()[0]
        return np.ones(len(data[key]), dtype=bool)

    def __invert__(self):
        """Inverse the AlwaysTrue object to AlwaysFalse object.

        Returns:
            An AlwaysFalse object.
        """
        return AlwaysFalse(key=self.key)


class AlwaysFalse(AnnDataPredicate):
    """A class inherited from AnnDataPredicate. Will return false for all elements under any requirement.

    Attributes:
        key: The key in the AnnData object (specified as `data` in `.check()`) that will be used for selection.
    """

    def __init__(self, key=None):
        """Initialize an AlwaysTrue object.

        Args:
            key: The key in the AnnData object (specified as `data` in `.check()`) that will be used for selection. If
                None, the first layer of the AnnData for comparison would be used. Defaults to None.
        """
        self.key = key

    def check(self, data):
        """Return an all-false boolean array with the shape of the layer.

        Args:
            data: The AnnData object to be tested.

        Returns:
            An all-false boolean array with the shape of the layer.
        """
        key = self.key if self.key is not None else data.keys()[0]
        return np.zeros(len(data[key]), dtype=bool)

    def __invert__(self):
        """Inverse the AlwaysTrue object to AlwaysTrue object.

        Returns:
            An AlwaysTrue object.
        """
        return AlwaysTrue(key=self.key)


class AnnDataPredicates(object):
    """A set of multiple AnnDataPredicate objects.

    Attributions:
        predicates (List[AnnDataPredicate]): a tuple containing multiple AnnDataPredicate objects.
    """

    def __init__(self, *predicates) -> None:
        """Initialize an AnnDataPredicates object.

        Args:
            *predicates: One or more AnnDataPredicate objects.
        """
        self.predicates = predicates

    def binop(self, other, op):
        """Bind two Predicate(s) objects together with given operation type.

        Args:
            other: Another Predicate(s) object to bind with.
            op: How to bind the requirement of Predicates (Union or Intersection).

        Raises:
            NotImplementedError: `other` is not an Predcate(s) instance.

        Returns:
            AnnDataPredicates: a new Predicates object with requirement binded.
        """
        if isinstance(other, AnnDataPredicate):
            return op(*self.predicates, other)
        elif isinstance(other, AnnDataPredicates):
            return op(*self.predicates, *other)
        else:
            raise NotImplementedError(f"Unidentified predicate type {type(other)}!")


class PredicateUnion(AnnDataPredicates):
    """Inherited from AnnDataPredicates. If at least 1 requirement from all predicates is fulfilled, the data would pass
    the check.
    """

    def check(self, data: AnnData) -> np.ndarray:
        """Check whether the data could fulfill at least 1 requirement by all Predicates.

        Args:
            data (AnnData): An AnnData object.

        Returns:
            np.ndarray: A boolean array with `True` at positions where the element can pass at least one check.
        """
        ret = None
        for pred in self.predicates:
            ret = np.logical_or(ret, pred.check(data)) if ret is not None else pred.check(data)
        return ret

    def __or__(self, other):
        """Bind with other Predicate(s) with union.

        Args:
            other: Another Predicate(s) object to bind with.

        Returns:
            PredicateUnion: the binded predicates.
        """
        return self.binop(other, PredicateUnion)

    def __and__(self, other):
        """Bind with other Predicate(s) with intersection.

        Args:
            other: Another Predicate(s) object to bind with.

        Returns:
            PredicateIntersection: the binded predicates.
        """
        return self.binop(other, PredicateIntersection)


class PredicateIntersection(AnnDataPredicates):
    """Inherited from AnnDataPredicates. If all requirements from all predicates are fulfilled, the data would pass
    the check.
    """

    def check(self, data):
        """Check whether the data could fulfill all requirements by all Predicates.

        Args:
            data (AnnData): An AnnData object.

        Returns:
            np.ndarray: A boolean array with `True` at positions where the element can pass all checks.
        """
        ret = None
        for pred in self.predicates:
            ret = np.logical_and(ret, pred.check(data)) if ret is not None else pred.check(data)
        return ret

    def __or__(self, other):
        """Bind with other Predicate(s) with union.

        Args:
            other: Another Predicate(s) object to bind with.

        Returns:
            PredicateUnion: the binded predicates.
        """
        return self.binop(other, PredicateUnion)

    def __and__(self, other):
        """Bind with other Predicate(s) with intersection.

        Args:
            other: Another Predicate(s) object to bind with.

        Returns:
            PredicateIntersection: the binded predicates.
        """
        return self.binop(other, PredicateIntersection)


def select(
    array: np.ndarray, pred: AnnDataPredicate = AlwaysTrue(), output_format: Literal["mask", "index"] = "mask"
) -> np.ndarray:
    """Select part of the array based on the condition provided by the used.

    Args:
        array: The original data to be selected from.
        pred: The condition provided by the user. Defaults to AlwaysTrue().
        output_format: Whether to output a mask of selection or selected items' indices. Defaults to "mask".

    Returns:
        A mask of selection or selected items' indices.
    """

    ret = pred.check(array)
    if output_format == "mask":
        pass
    elif output_format == "index":
        ret = np.where(ret)[0]
    return ret


# ---------------------------------------------------------------------------------------------------
# estimation related


def calc_R2(
    X: np.ndarray,
    Y: np.ndarray,
    k: Union[float, np.ndarray],
    f: Callable = lambda X, k: np.einsum("ij,i -> ij", X, k),
) -> float:
    """Calculate R-square. X, Y: n_species (mu, sigma) x n_obs

    Args:
        X: The input data.
        Y: The output data observed as the ground truth.
        k: The parameter(s) used to calculate the output data.
        f: Prediction function that takes X and k as input and returns predicted values.

    Returns:
        The R-squared value indicating the goodness of fit between observed and predicted values.
    """
    if X.ndim == 1:
        X = X[None]
    if Y.ndim == 1:
        Y = Y[None]
    if np.isscalar(k):
        k = np.array([k])

    Y_bar = np.mean(Y, 1)
    d = Y.T - Y_bar
    SS_tot = np.sum(np.einsum("ij,ij -> i", d, d))

    F = f(X, k) if len(signature(f).parameters) == 2 else f(X, *k)
    d = F - Y
    SS_res = np.sum(np.einsum("ij,ij -> j", d, d))

    return 1 - SS_res / SS_tot


def norm_loglikelihood(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Calculate log-likelihood for the data.

    Args:
        x: The input data.
        mu: The mean of the data.
        sig: The standard deviation of the data.

    Returns:
        The log-likelihood of the data.
    """
    err = (x - mu) / sig
    ll = -len(err) / 2 * np.log(2 * np.pi) - 0.5 * len(err) * np.log(sig**2) - 0.5 * err.dot(err.T)
    return np.sum(ll, 0)


def calc_norm_loglikelihood(
    X: np.ndarray,
    Y: np.ndarray,
    k: Union[float, np.ndarray],
    f: Callable = lambda X, k: np.einsum("ij,i -> ij", X, k),
) -> float:
    """Calculate log likelihood based on normal distribution. X, Y: n_species (mu, sigma) x n_obs

    Args:
        X: The input data.
        Y: The output data observed as the ground truth.
        k: The parameter(s) used to calculate the output data.
        f: Prediction function that takes X and k as input and returns predicted values.

    Returns:
        The log-likelihood value.
    """
    if X.ndim == 1:
        X = X[None]
    if Y.ndim == 1:
        Y = Y[None]
    if np.isscalar(k):
        k = np.array([k])

    n = X.shape[0]
    F = f(X, k)

    d = F - Y
    sig = np.einsum("ij,ij -> i", d, d)

    LogLL = 0
    for i in range(Y.shape[0]):
        LogLL += norm_loglikelihood(Y[i], F[i], np.sqrt(sig[i] / n))

    return LogLL


# ---------------------------------------------------------------------------------------------------
# velocity related


def find_extreme(
    s: Union[np.ndarray, sp.spmatrix],
    u: Union[np.ndarray, sp.spmatrix],
    normalize: bool = True,
    perc_left: Optional[float] = None,
    perc_right: Optional[float] = None,
) -> np.ndarray:
    """Find extreme regions in a combination of two arrays.

    This function combines two arrays `s` and `u` and identifies regions that are considered extreme. The combination is
    performed by either normalizing each array separately and adding them, or directly adding them without
    normalization.

    Args:
        s: The spliced data.
        u: The unspliced data.
        normalize: Whether to normalize each array before adding them. Default is True.
        perc_left: The percentile value used to identify the left tail extreme regions.
        perc_right: The percentile value used to identify the right tail extreme regions.

    Returns:
        A boolean mask identifying the extreme regions based on the provided percentiles.
    """
    s, u = (s.A if sp.issparse(s) else s, u.A if sp.issparse(u) else u)

    if normalize:
        su = s / np.clip(np.max(s), 1e-3, None)
        su += u / np.clip(np.max(u), 1e-3, None)
    else:
        su = s + u

    if perc_left is None:
        mask = su >= np.percentile(su, 100 - perc_right, axis=0)
    elif perc_right is None:
        mask = np.ones_like(su, dtype=bool)
    else:
        left, right = np.percentile(su, [perc_left, 100 - perc_right], axis=0)
        mask = (su <= left) | (su >= right)

    return mask


def get_group_params_indices(adata: AnnData, param_name: str) -> np.ndarray:
    """Get the indices of columns in an AnnData object associated with a specific group parameter.

    Args:
        adata: An AnnData object containing group parameter information.
        param_name: The name of the group parameter to search for.

    Returns:
        An array of boolean values representing column indices matching the specified parameter.
    """
    return adata.var.columns.str.endswith(param_name)


def set_transition_genes(
    adata: AnnData,
    vkey: str = "velocity_S",
    min_r2: float = None,
    min_alpha: float = None,
    min_gamma: float = None,
    min_delta: float = None,
    use_for_dynamics: bool = True,
    store_key: str = "use_for_transition",
    minimal_gene_num: int = 50,
) -> AnnData:
    """Set the transition genes in the AnnData object.

    Args:
        adata: The AnnData object.
        vkey: The key of estimated velocity.
        min_r2: The minimum value to filter r2.
        min_alpha: The minimum value to filter alpha.
        min_gamma: The minimum value to filter gamma.
        min_delta: The minimum value to filter delta.
        use_for_dynamics: Whether to save use_for_dynamics.
        store_key: The key to store transition data.
        minimal_gene_num: The minimum gene to deal with the situation of too few genes.

    Returns:
        The updated AnnData object.
    """
    layer = vkey.split("_")[1]
    vel_params_df = get_vel_params(adata)

    if adata.uns["dynamics"]["est_method"] == "twostep" and adata.uns["dynamics"]["experiment_type"] == "kin":
        # if adata.uns['dynamics']['has_splicing']:
        #     min_r2 = 0.5 if min_r2 is None else min_r2
        # else:
        min_r2 = 0.9 if min_r2 is None else min_r2
    elif adata.uns["dynamics"]["experiment_type"] in [
        "mix_kin_deg",
        "mix_pulse_chase",
        "kin",
    ]:
        logLL_col = vel_params_df.columns[vel_params_df.columns.str.endswith("logLL")]
        if len(logLL_col) > 1:
            main_warning(f"there are two columns ends with logLL: {logLL_col}")

        adata.var[store_key] = vel_params_df[logLL_col[-1]].astype(float) < np.nanpercentile(
            vel_params_df[logLL_col[-1]].astype(float), 10
        )
        if layer in ["N", "T"]:
            return adata
        else:
            min_r2 = 0.01
    else:
        min_r2 = 0.01 if min_r2 is None else min_r2

    if min_alpha is None:
        min_alpha = 0.01
    if min_gamma is None:
        min_gamma = 0.01
    if min_delta is None:
        min_delta = 0.01

    # the following parameters aggreation for different groups can be improved later
    if layer == "U":
        if "alpha" not in vel_params_df.columns:
            is_group_alpha, is_group_alpha_r2 = (
                get_group_params_indices(adata, "alpha"),
                get_group_params_indices(adata, "alpha_r2"),
            )
            if is_group_alpha.sum() > 0:
                vel_params_df["alpha"] = vel_params_df.loc[:, is_group_alpha].mean(1, skipna=True)
                vel_params_df["alpha_r2"] = vel_params_df.loc[:, np.hstack((is_group_alpha_r2, False))].mean(1, skipna=True)
            else:
                raise Exception("there is no alpha/alpha_r2 parameter estimated for your adata object")

        if "alpha_r2" not in vel_params_df.columns:
            vel_params_df["alpha_r2"] = None
        if np.all(vel_params_df.alpha_r2.values is None):
            vel_params_df.alpha_r2 = 1
        adata.var[store_key] = (
            (vel_params_df.alpha > min_alpha) & (vel_params_df.alpha_r2 > min_r2) & adata.var.use_for_dynamics
            if use_for_dynamics
            else (vel_params_df.alpha > min_alpha) & (vel_params_df.alpha_r2 > min_r2)
        )
    elif layer == "S":
        if "gamma" not in vel_params_df.columns:
            is_group_gamma, is_group_gamma_r2 = (
                get_group_params_indices(adata, "gamma"),
                get_group_params_indices(adata, "gamma_r2"),
            )
            if is_group_gamma.sum() > 0:
                vel_params_df["gamma"] = vel_params_df.loc[:, is_group_gamma].mean(1, skipna=True)
                vel_params_df["gamma_r2"] = vel_params_df.loc[:, np.hstack((is_group_gamma_r2, False))].mean(1, skipna=True)
            else:
                raise Exception("there is no gamma/gamma_r2 parameter estimated for your adata object")

        if "gamma_r2" not in vel_params_df.columns:
            main_debug("setting all gamma_r2 to 1")
            vel_params_df["gamma_r2"] = 1
        if np.all(vel_params_df.gamma_r2.values is None) or np.all(vel_params_df.gamma_r2.values == ""):
            main_debug("Since all gamma_r2 values are None or '', setting all gamma_r2 values to 1.")
            vel_params_df.gamma_r2 = 1

        adata.var[store_key] = (vel_params_df.gamma > min_gamma) & (vel_params_df.gamma_r2 > min_r2)
        if use_for_dynamics:
            adata.var[store_key] = adata.var[store_key] & adata.var.use_for_dynamics

    elif layer == "P":
        if "delta" not in vel_params_df.columns:
            is_group_delta, is_group_delta_r2 = (
                get_group_params_indices(adata, "delta"),
                get_group_params_indices(adata, "delta_r2"),
            )
            if is_group_delta.sum() > 0:
                vel_params_df["delta"] = vel_params_df.loc[:, is_group_delta].mean(1, skipna=True)
                vel_params_df["delta_r2"] = vel_params_df.loc[:, np.hstack((is_group_delta_r2, False))].mean(1, skipna=True)
            else:
                raise Exception("there is no delta/delta_r2 parameter estimated for your adata object")

        if "delta_r2" not in vel_params_df.columns:
            vel_params_df["delta_r2"] = None
        if np.all(vel_params_df.delta_r2.values is None):
            vel_params_df.delta_r2 = 1
        adata.var[store_key] = (
            (vel_params_df.delta > min_delta) & (vel_params_df.delta_r2 > min_r2) & adata.var.use_for_dynamics
            if use_for_dynamics
            else (vel_params_df.delta > min_delta) & (vel_params_df.delta_r2 > min_r2)
        )
    if layer == "T":
        if "gamma" not in vel_params_df.columns:
            is_group_gamma, is_group_gamma_r2 = (
                get_group_params_indices(adata, "gamma"),
                get_group_params_indices(adata, "gamma_r2"),
            )
            if is_group_gamma.sum() > 0:
                vel_params_df["gamma"] = vel_params_df.loc[:, is_group_gamma].mean(1, skipna=True)
                vel_params_df["gamma_r2"] = vel_params_df.loc[:, np.hstack((is_group_gamma_r2, False))].mean(1, skipna=True)
            else:
                raise Exception("there is no gamma/gamma_r2 parameter estimated for your adata object")

        if "gamma_r2" not in vel_params_df.columns:
            vel_params_df["gamma_r2"] = None
        if np.all(vel_params_df.gamma_r2.values is None):
            vel_params_df.gamma_r2 = 1
        if sum(vel_params_df.gamma_r2.isna()) == adata.n_vars:
            gamm_r2_checker = vel_params_df.gamma_r2.isna()
        else:
            gamm_r2_checker = vel_params_df.gamma_r2 > min_r2
        adata.var[store_key] = (
            (vel_params_df.gamma > min_gamma) & gamm_r2_checker & adata.var.use_for_dynamics
            if use_for_dynamics
            else (vel_params_df.gamma > min_gamma) & gamm_r2_checker
        )

    if adata.var[store_key].sum() < 5 and adata.n_vars > 5:
        main_warning(
            "Only less than 5 genes satisfies transition gene selection criteria, which may be resulted "
            "from: \n"
            "  1. Very low intron/new RNA ratio, try filtering low ratio and poor quality cells \n"
            "  2. Your selection criteria may be set to be too stringent, try loosing those thresholds \n"
            "  3. Your data has strange expression kinetics. Welcome to report to dynamo team for more insights.\n"
            "We auto correct this behavior by selecting the %d top genes according to gamma_r2 values."
        )
        reserve_minimal_genes_by_gamma_r2(adata, store_key, minimal_gene_num=minimal_gene_num)

    update_vel_params(adata, vel_params_df)

    return adata


def get_ekey_vkey_from_adata(adata: AnnData) -> Tuple[str, str, str]:
    """Get the corresponding ekey and vkey from anndata.

    Args:
        adata: The AnnData object.

    Returns:
        A tuple containing:
            ekey: expression from which to extrapolate velocity
            vkey: velocity key
            layer: the states cells will be used in velocity embedding.
    """
    dynamics_key = [i for i in adata.uns.keys() if i.endswith("dynamics")][0]
    experiment_type, use_smoothed = (
        adata.uns[dynamics_key]["experiment_type"],
        adata.uns[dynamics_key]["use_smoothed"],
    )
    has_splicing, has_labeling = (
        adata.uns[dynamics_key]["has_splicing"],
        adata.uns[dynamics_key]["has_labeling"],
    )
    NTR = adata.uns[dynamics_key]["NTR_vel"]

    mapper = get_mapper()
    layer = []

    if has_splicing:
        if has_labeling:
            if "X_new" not in adata.layers.keys():  # unlabel spliced: S
                raise Exception("The input data you have is not normalized or normalized + smoothed!")

            if experiment_type.lower() in [
                "kin",
                "mix_pulse_chase",
                "mix_kin_deg",
            ]:
                ekey, vkey, layer = (
                    (
                        mapper["X_total"] if NTR else mapper["X_spliced"],
                        "velocity_T" if NTR else "velocity_S",
                        ("X_total" if NTR else "X_spliced"),
                    )
                    if use_smoothed
                    else (
                        "X_total" if NTR else "X_spliced",
                        "velocity_T" if NTR else "velocity_S",
                        "X_total" if NTR else "X_spliced",
                    )
                )
            elif experiment_type.lower() == "deg":
                ekey, vkey, layer = (
                    (
                        mapper["X_total"] if NTR else mapper["X_spliced"],
                        "velocity_T" if NTR else "velocity_S",
                        ("X_total" if NTR else "X_spliced"),
                    )
                    if use_smoothed
                    else (
                        "X_total" if NTR else "X_spliced",
                        "velocity_T" if NTR else "velocity_S",
                        "X_total" if NTR else "X_spliced",
                    )
                )
            elif experiment_type.lower() in ["one_shot", "one-shot"]:
                ekey, vkey, layer = (
                    (
                        mapper["X_total"] if NTR else mapper["X_spliced"],
                        "velocity_T" if NTR else "velocity_S",
                        ("X_total" if NTR else "X_spliced"),
                    )
                    if use_smoothed
                    else (
                        "X_total" if NTR else "X_spliced",
                        "velocity_T" if NTR else "velocity_S",
                        "X_total" if NTR else "X_spliced",
                    )
                )
            elif experiment_type.lower() == "mix_std_stm":
                ekey, vkey, layer = (
                    (
                        mapper["X_total"] if NTR else mapper["X_spliced"],
                        "velocity_T" if NTR else "velocity_S",
                        ("X_total" if NTR else "X_spliced"),
                    )
                    if use_smoothed
                    else (
                        "X_total" if NTR else "X_spliced",
                        "velocity_T" if NTR else "velocity_S",
                        "X_total" if NTR else "X_spliced",
                    )
                )
        else:
            if not (("X_unspliced" in adata.layers.keys()) or (mapper["X_unspliced"] in adata.layers.keys())):
                raise Exception(
                    "The input data you have is not normalized/log transformed or smoothed and normalized/log "
                    "transformed!"
                )
            ekey, vkey, layer = (
                (mapper["X_spliced"], "velocity_S", "X_spliced")
                if use_smoothed
                else ("X_spliced", "velocity_S", "X_spliced")
            )
    else:
        # use_smoothed: False
        if ("X_new" in adata.layers.keys()) or (mapper["X_new"] in adata.layers.keys):  # run new / total ratio (NTR)
            # we may also create M_U, M_S layers?
            if experiment_type == "kin":
                ekey, vkey, layer = (
                    (mapper["X_total"], "velocity_T", "X_total")
                    if use_smoothed
                    else ("X_total", "velocity_T", "X_total")
                )
            elif experiment_type == "deg":
                ekey, vkey, layer = (
                    (mapper["X_total"], "velocity_T", "X_total")
                    if use_smoothed
                    else ("X_total", "velocity_T", "X_total")
                )
            elif experiment_type in ["one-shot", "one_shot"]:
                ekey, vkey, layer = (
                    (mapper["X_total"], "velocity_T", "X_total")
                    if use_smoothed
                    else ("X_total", "velocity_T", "X_total")
                )
            elif experiment_type == "mix_std_stm":
                ekey, vkey, layer = (
                    (mapper["X_total"], "velocity_T", "X_total")
                    if use_smoothed
                    else ("X_total", "velocity_T", "X_total")
                )

        else:
            raise Exception(
                "The input data you have is not normalized/log trnasformed or smoothed and normalized/log trnasformed!"
            )

    return ekey, vkey, layer


# ---------------------------------------------------------------------------------------------------
# cell velocities related
def get_neighbor_indices(
    adjacency_list: np.ndarray,
    source_idx: int,
    n_order_neighbors: int = 2,
    max_neighbors_num: Optional[int] = None,
) -> np.ndarray:
    """Get a list (np.array) of `n_order_neighbors` neighbor indices of source_idx.

    If `max_neighbors_num` is set and the n order neighbors of `source_idx` is larger than `max_neighbors_num`, a list
    of neighbors will be randomly chosen and returned.

    Args:
        adjacency_list: The adjacency list of the graph.
        source_idx: The index of the source node.
        n_order_neighbors: The number of neighbor orders to consider, which controls the depth of iterative neighbor
            search.
        max_neighbors_num: The maximum number of neighbors to return.

    Returns:
        A list of neighbor indices.
    """
    _indices = [source_idx]
    for _ in range(n_order_neighbors):
        _indices = np.append(_indices, adjacency_list[_indices])
        if np.isnan(_indices).any():
            _indices = _indices[~np.isnan(_indices)]
    _indices = np.unique(_indices)
    if max_neighbors_num is not None and len(_indices) > max_neighbors_num:
        _indices = np.random.choice(_indices, max_neighbors_num, replace=False)
    return _indices


def append_iterative_neighbor_indices(
    indices: np.ndarray,
    n_recurse_neighbors: int = 2,
    max_neighbors_num: Optional[int] = None,
) -> List[np.ndarray]:
    """Append iterative neighbor indices for each index in the input array.

    Args:
        indices: The input array of indices as a 1D numpy array.
        n_recurse_neighbors: The number of neighbor orders to consider, which controls the depth of iterative neighbor
            search.
        max_neighbors_num: The maximum number of neighbor indices to be stored for each index.

    Returns:
        A list of array containing iterative neighbor indices for each index in the input array.
    """
    indices_rec = []
    for i in range(indices.shape[0]):
        neig = get_neighbor_indices(indices, i, n_recurse_neighbors, max_neighbors_num)
        indices_rec.append(neig)
    return indices_rec


def split_velocity_graph(
    G: Union[sp.csr_matrix, npt.ArrayLike],
    neg_cells_trick: bool = True,
) -> Union[sp.csr_matrix, Tuple[sp.csr_matrix, sp.csr_matrix]]:
    """Split velocity graph (built either with correlation or with cosine kernel
    into one positive graph and one negative graph"""

    if not sp.issparse(G):
        G = sp.csr_matrix(G)
    if neg_cells_trick:
        G_ = G.copy()
    G.data[G.data < 0] = 0
    G.eliminate_zeros()

    if neg_cells_trick:
        G_.data[G_.data > 0] = 0
        G_.eliminate_zeros()

        return (G, G_)
    else:
        return G


# ---------------------------------------------------------------------------------------------------
# vector field related

#  Copyright (c) 2013 Alexandre Drouin. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of
#  this software and associated documentation files (the "Software"), to deal in
#  the Software without restriction, including without limitation the rights to
#  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
#  of the Software, and to permit persons to whom the Software is furnished to do
#  so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  If you happen to meet one of the copyright holders in a bar you are obligated
#  to buy them one pint of beer.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  https://gist.github.com/aldro61/5889795


def linear_least_squares(
    a: npt.ArrayLike, b: npt.ArrayLike, residuals: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Return the least-squares solution to a linear matrix equation.

    Solves the equation `a x = b` by computing a vector `x` that minimizes the Euclidean 2-norm `|| b - a x ||^2`.
    The equation may be under-, well-, or over- determined (i.e., the number of linearly independent rows of `a` can be
    less than, equal to, or greater than its number of linearly independent columns).  If `a` is square and of full
    rank, then `x` (but for round-off error) is the "exact" solution of the equation.

    Args:
        a: The coefficient matrix.
        b: The ordinate or "dependent variable" values.
        residuals: Whether to compute the residuals associated with the least-squares solution. Defaults to False.

    Returns:
        The least-squares solution. If `residuals` is True, the sum of residuals (squared Euclidean 2-norm for each
        column in ``b - a*x``) would also be returned.
    """

    if type(a) != np.ndarray or not a.flags["C_CONTIGUOUS"]:
        main_warning(
            "Matrix a is not a C-contiguous numpy array. The solver will create a copy, which will result"
            + " in increased memory usage."
        )

    a = np.asarray(a, order="c")
    i = dgemm(alpha=1.0, a=a.T, b=a.T, trans_b=True)
    x = np.linalg.solve(i, dgemm(alpha=1.0, a=a.T, b=b))

    if residuals:
        return x, np.linalg.norm(np.dot(a, x) - b)
    else:
        return x


def integrate_vf(
    init_states: np.ndarray,
    t: np.ndarray,
    args: Tuple,
    integration_direction: Literal["forward", "backward", "both"],
    f: Callable,
    interpolation_num: Optional[int]=None,
    average: bool = True,
):
    """Integrating along vector field function.

    Args:
        init_states: The initial states for numerical integration as a 2D numpy array.
        t: The time points for numerical integration as a 1D numpy array.
        args: Additional arguments to be passed to the vector field function `f`.
        integration_direction: The direction of integration.
            - "forward": Integrate the vector field function in the forward direction.
            - "backward": Integrate the vector field function in the backward direction.
            - "both": Integrate the vector field function both forward and backward in time.
        f: The vector field function.
        interpolation_num: The number of points for interpolation.
        average: Whether to average the integrated states over cells when multiple initial states are provided.

    Returns:
        A tuple containing the integrated time array `t` and the integrated states `Y`.
    """

    n_cell, n_feature, n_steps = (
        init_states.shape[0],
        init_states.shape[1],
        len(t) if interpolation_num is None else interpolation_num,
    )

    if n_cell > 1:
        if integration_direction == "both":
            if average:
                avg = np.zeros((n_steps * 2, n_feature))
        else:
            if average:
                avg = np.zeros((n_steps, n_feature))

    Y = None
    if interpolation_num is not None:
        valid_ids = None
    for i in tqdm(range(n_cell), desc="integrating vector field"):
        y0 = init_states[i, :]
        if integration_direction == "forward":
            y = odeint(lambda x, t: f(x), y0, t, args=args)
            t_trans = t
        elif integration_direction == "backward":
            y = odeint(lambda x, t: f(x), y0, -t, args=args)
            t_trans = -t
        elif integration_direction == "both":
            y_f = odeint(lambda x, t: f(x), y0, t, args=args)
            y_b = odeint(lambda x, t: f(x), y0, -t, args=args)
            y = np.hstack((y_b[::-1, :], y_f))
            t_trans = np.hstack((-t[::-1], t))

            if interpolation_num is not None:
                interpolation_num = interpolation_num * 2
        else:
            raise Exception("both, forward, backward are the only valid direction argument strings")

        if interpolation_num is not None:
            vids = np.where((np.diff(y.T) < 1e-3).sum(0) < y.shape[1])[0]
            valid_ids = vids if valid_ids is None else list(set(valid_ids).union(vids))

        Y = y if Y is None else np.vstack((Y, y))

    if interpolation_num is not None:
        valid_t_trans = t_trans[valid_ids]

        _t, _Y = None, None
        for i in range(n_cell):
            ind_vec = np.arange(i, (i + 1) * len(t_trans))
            cur_Y = Y[ind_vec, :][valid_ids, :]
            t_linspace = np.linspace(valid_t_trans[0], valid_t_trans[-1], interpolation_num)
            f = interpolate.interp1d(valid_t_trans, cur_Y.T)
            _Y = f(t_linspace) if _Y is None else np.hstack((_Y, f(t_linspace)))
            _t = t_linspace if _t is None else np.hstack((_t, t_linspace))

        t, Y = _t, _Y.T

    if n_cell > 1 and average:
        t_len = int(len(t) / n_cell)
        for i in range(t_len):
            avg[i, :] = np.mean(Y[np.arange(n_cell) * t_len + i, :], 0)
        Y = avg

    return t, Y


# ---------------------------------------------------------------------------------------------------
# fetch states
def fetch_states(
    adata: AnnData,
    init_states: np.ndarray,
    init_cells: Union[str, List],
    basis: str,
    layer: str,
    average: Union[str, bool],
    t_end: float,
) -> Tuple[np.ndarray, Dict[str, Any], float, Optional[List[str]]]:
    """Fetch initial states for the vector field modeling of single-cell data.

    This function retrieves the initial states for the vector field modeling of single-cell data from the provided
    `adata` object. It allows providing either the `init_states` directly or the `init_cells` names and the `basis`
    (e.g., pca) from which the initial states should be derived.

    Args:
        adata: An AnnData object containing the single-cell data.
        init_states: The initial states to use for the vector field modeling. If not provided, `init_cells` and `basis`
            should be used to derive the initial states.
        init_cells: The cell names to use for deriving the initial states.
        basis: The basis to use for deriving the initial states.
        layer: The layer of the data to use for deriving the initial states.
        average: Determines how to handle multiple initial states when provided. If "origin" or True, the initial states
            will be averaged to a single state. If "trajectory", the initial states will be kept as separate states.
        t_end: The end time point for the vector field modeling.

    Returns:
        A tuple containing the following:
            - init_states: the derived initial states for the vector field modeling.
            - VecFld: a dictionary containing information about the vector field.
            - t_end: the end time point for the vector field modeling.
            - valid_genes: a list of valid gene names used for the vector field modeling,
    """
    if basis is not None:
        vf_key = "VecFld_" + basis
    else:
        vf_key = "VecFld"
    VecFld = adata.uns[vf_key]
    X = VecFld["X"]
    valid_genes = None

    if init_states is None and init_cells is None:
        raise Exception("Either init_state or init_cells should be provided.")
    elif init_states is None and init_cells is not None:
        if type(init_cells) == str:
            init_cells = [init_cells]
        intersect_cell_names = sorted(
            set(init_cells).intersection(adata.obs_names),
            key=lambda x: list(init_cells).index(x),
        )
        _cell_names = init_cells if len(intersect_cell_names) == 0 else intersect_cell_names

        if basis is not None:
            init_states = adata[_cell_names].obsm["X_" + basis].copy()
            if len(_cell_names) == 1:
                init_states = init_states.reshape((1, -1))
            VecFld = adata.uns["VecFld_" + basis]
            X = adata.obsm["X_" + basis]

            valid_genes = [basis + "_" + str(i) for i in np.arange(init_states.shape[1])]
        else:
            # valid_genes = list(set(genes).intersection(adata.var_names[adata.var.use_for_transition]) if genes is not
            # None \
            #     else adata.var_names[adata.var.use_for_transition]
            # ----------- enable the function to only only a subset genes -----------

            vf_key = "VecFld" if layer == "X" else "VecFld_" + layer
            valid_genes = adata.uns[vf_key]["genes"]
            init_states = (
                adata[_cell_names, :][:, valid_genes].X
                if layer == "X"
                else log1p_(adata, adata[_cell_names, :][:, valid_genes].layers[layer])
            )
            if sp.issparse(init_states):
                init_states = init_states.A
            if len(_cell_names) == 1:
                init_states = init_states.reshape((1, -1))

            if layer == "X":
                VecFld = adata.uns["VecFld"]
                X = adata[:, valid_genes].X
            else:
                VecFld = adata.uns["VecFld_" + layer]
                X = log1p_(adata, adata[:, valid_genes].layers[layer])

    if init_states.shape[0] > 1 and average in ["origin", True]:
        init_states = init_states.mean(0).reshape((1, -1))

    if t_end is None:
        t_end = getTend(X, VecFld["V"])

    if sp.issparse(init_states):
        init_states = init_states.A

    return init_states, VecFld, t_end, valid_genes


def getTend(X: np.ndarray, V: np.ndarray) -> float:
    """Compute the end time for the vector field modeling.

    Args:
        X: The cell embeddings.
        V: The estimated velocities.

    Returns:
        The end time (t_end) for the vector field modeling.
    """
    xmin, xmax = X.min(0), X.max(0)
    V_abs = np.abs(V)
    t_end = np.max(xmax - xmin) / np.percentile(V_abs[V_abs > 0], 1)

    return t_end


def getTseq(init_states: np.ndarray, t_end: float, step_size: Optional[Union[int, float]] = None) -> np.ndarray:
    """Generate a time sequence for the vector field modeling.

    Args:
        init_states: The initial states for the vector field modeling as a 2D numpy array.
        t_end: The end time for the vector field modeling.
        step_size: The time step size between each time point in the sequence.

    Returns:
        An array containing the time sequence for the vector field modeling.
    """
    if step_size is None:
        max_steps = int(max(7 / (init_states.shape[1] / 300), 4)) if init_states.shape[1] > 300 else 7
        t_linspace = np.linspace(0, t_end, 10 ** (np.min([int(np.log10(t_end)), max_steps])))
    else:
        t_linspace = np.arange(0, t_end + step_size, step_size)

    return t_linspace


# ---------------------------------------------------------------------------------------------------
# spatial related
def compute_smallest_distance(
    coords: np.ndarray, leaf_size: int = 40, sample_num: Optional[int] = None, use_unique_coords: bool = True
) -> float:
    """Compute and return smallest distance.

    This function is a wrapper for sklearn API.

    Args:
        coords: NxM matrix. N is the number of data points and M is the dimension of each point's feature.
        leaf_size: The leaf size parameter for building Kd-tree. Defaults to 40.
        sample_num: The number of cells to be sampled. Defaults to None.
        use_unique_coords: Whether to remove duplicate coordinates. Defaults to True.

    Raises:
        ValueError: The dimension of coords is not 2x2

    Returns:
        The minimum distance between points.
    """

    if len(coords.shape) != 2:
        raise ValueError("Coordinates should be a NxM array.")
    if use_unique_coords:
        main_info("using unique coordinates for computing smallest distance")
        coords = [tuple(coord) for coord in coords]
        coords = np.array(list(set(coords)))
    # use cKDTree which is implmented in C++ and is much faster than KDTree
    kd_tree = cKDTree(coords, leafsize=leaf_size)
    if sample_num is None:
        sample_num = len(coords)
    N, _ = min(len(coords), sample_num), coords.shape[1]
    selected_estimation_indices = np.random.choice(len(coords), size=N, replace=False)

    # Note k=2 here because the nearest query is always a point itself.
    distances, _ = kd_tree.query(coords[selected_estimation_indices, :], k=2)
    min_dist = min(distances[:, 1])

    return min_dist


# ---------------------------------------------------------------------------------------------------
# multiple core related

# Pass kwargs to starmap while using Pool
# https://stackoverflow.com/questions/45718523/pass-kwargs-to-starmap-while-using-pool-in-python
def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    """Apply a function with arguments and keyword arguments to an iterable using multiprocessing.

    Args:
        pool: The multiprocessing pool.
        fn: The function to apply.
        args_iter: The iterable of arguments.
        kwargs_iter: The iterable of keyword arguments.

    Returns:
        A list of the results of the function applied to the iterable.
    """
    args_for_starmap = zip(itertools.repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def apply_args_and_kwargs(fn, args, kwargs):
    """Apply a function with arguments and keyword arguments.

    Args:
        fn: The function to apply.
        args: The arguments.

    Returns:
        The result of the function applied to the arguments.
    """
    return fn(*args, **kwargs)


# ---------------------------------------------------------------------------------------------------
# ranking related
def get_rank_array(
    adata: AnnData,
    arr_key: Union[str, np.ndarray],
    genes: Optional[List[str]] = None,
    abs: bool = False,
    dtype: Optional[np.dtype] = None,
) -> Tuple[List[str], np.ndarray]:
    """Get the data array that will be used for gene-wise or cell-wise ranking.

    Args:
        adata: AnnData object that contains the array to be sorted in `.var` or `.layer`.
        arr_key: The key of the to-be-ranked array stored in `.var` or or `.layer`. If the array is found in `.var`, the
            `groups` argument will be ignored. If a numpy array is passed, it is used as the array to be ranked and must
            be either a 1d array of length `.n_var`, or a `.n_obs`-by-`.n_var` 2d array.
        genes: The gene list that speed will be ranked. If provided, they must overlap the dynamics genes. Defaults to
            None.
        abs: When pooling the values in the array (see below), whether to take the absolute values. Defaults to False.
        dtype: The dtype of the result array would be formated into. Defaults to None.

    Raises:
        TypeError: invalid `arr_key`.
        ValueError: invalid `genes`.
        Exception: invalid `arr_key`.

    Returns:
        A tuple (genes, arr) where `genes` is a list containing genes be ranked and `arr` is an array that stores
        information required for gene/cell-wise ranking.
    """

    dynamics_genes = (
        adata.var.use_for_dynamics if "use_for_dynamics" in adata.var.keys() else np.ones(adata.n_vars, dtype=bool)
    )
    if genes is not None:
        if type(genes) is str:
            genes = adata.var[genes].to_list()
            genes = np.logical_and(genes, dynamics_genes.to_list())
        elif areinstance(genes, str):
            genes_ = adata.var_names[dynamics_genes].intersection(genes).to_list()
            genes = adata.var_names.isin(genes_)
        elif areinstance(genes, bool) or areinstance(genes, np.bool_):
            genes = np.array(genes)
            genes = np.logical_and(genes, dynamics_genes.to_list())
        else:
            raise TypeError(
                "The provided genes should either be a key of adata.var, an array of gene names, or of booleans."
            )
    else:
        genes = dynamics_genes

    if not np.any(genes):
        raise ValueError("The list of genes provided does not contain any dynamics genes.")

    if type(arr_key) is str:
        if arr_key in adata.layers.keys():
            arr = index_gene(adata, adata.layers[arr_key], genes)
        elif arr_key in adata.var.keys():
            arr = index_gene(adata, adata.var[arr_key], genes)
        else:
            raise Exception(f"Key {arr_key} not found in neither .layers nor .var.")
    else:
        arr = index_gene(adata, arr_key, genes)

    if type(arr) == ArrayView:
        arr = np.array(arr)
    if sp.issparse(arr):
        arr = arr.A
    arr[np.isnan(arr)] = 0

    if dtype is not None:
        arr = np.array(arr, dtype=dtype)
    if abs:
        arr = np.abs(arr)

    return genes, arr


# ---------------------------------------------------------------------------------------------------
# projection related
def projection_with_transition_matrix(
    T: Union[np.ndarray, sp.csr_matrix], X_embedding: np.ndarray, correct_density: bool = True, norm_dist: bool = True
) -> np.ndarray:
    """Project velocity vectors to a low-dimensional embedding using a transition matrix.

    Args:
        T: The transition matrix representing velocity vectors. It can be a dense numpy array or a sparse csr_matrix.
        X_embedding: The low-dimensional embedding coordinates as a 2D numpy array. Each row represents the
            embedding coordinates of a cell in `T`.
        correct_density: Whether to correct the density of the projected velocity vectors.
        norm_dist: Whether to normalize the difference in embedding coordinates between connected cells before
            projecting the velocity vectors.

    Returns:
        A numpy array containing the projected velocity vectors for each cell in the embedding.
    """
    n = T.shape[0]
    delta_X = np.zeros((n, X_embedding.shape[1]))

    if not sp.issparse(T):
        T = sp.csr_matrix(T)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in LoggerManager.progress_logger(
            range(n),
            progress_name="projecting velocity vector to low dimensional embedding",
        ):
            idx = T[i].indices
            diff_emb = X_embedding[idx] - X_embedding[i, None]
            if norm_dist:
                diff_emb /= norm(diff_emb, axis=1)[:, None]
            if np.isnan(diff_emb).sum() != 0:
                diff_emb[np.isnan(diff_emb)] = 0
            T_i = T[i].data
            delta_X[i] = T_i.dot(diff_emb)
            if correct_density:
                delta_X[i] -= T_i.mean() * diff_emb.sum(0)

    return delta_X

def density_corrected_transition_matrix(T: Union[npt.ArrayLike, sp.csr_matrix]) -> sp.csr_matrix:
    """Compute the density corrected transition matrix.

    Args:
        T: The transition matrix to be corrected.

    Returns:
        The transition matrix with density correction from T.
    """
    T = sp.csr_matrix(T, copy=True)

    for i in range(T.shape[0]):
        idx = T[i].indices
        T_i = T[i].data
        T_i -= T_i.mean()
        T[i, idx] = T_i

    return T


# ---------------------------------------------------------------------------------------------------
# differential gene expression test related
def fdr(p_vals: np.ndarray) -> np.ndarray:
    """Calculate False Discovery Rate using Benjamini–Hochberg (non-negative) method.

    Args:
        p_vals: The p-values describes the likelihood of an observation based on a probability distribution.

    Returns:
        The corrected False Discovery Rate.
    """
    from scipy.stats import rankdata

    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr
