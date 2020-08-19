from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import interpolate, stats
from scipy.sparse import issparse, csr_matrix
import scipy.sparse.linalg as splinalg
from scipy.integrate import odeint
from scipy.linalg.blas import dgemm
from sklearn.neighbors import NearestNeighbors
import warnings
import time

from ..preprocessing.utils import Freeman_Tukey

# ---------------------------------------------------------------------------------------------------
# others
def get_mapper(smoothed=True):
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


def get_mapper_inverse(smoothed=True):
    mapper = get_mapper(smoothed)

    return dict([(v, k) for k, v in mapper.items()])


def get_finite_inds(X, ax=0):
    finite_inds = np.isfinite(X.sum(ax).A1) if issparse(X) else np.isfinite(X.sum(ax))

    return finite_inds


def get_pd_row_column_idx(df, queries, type='column'):
    """Find the numeric indices of multiple index/column matches with a vectorized solution using np.searchsorted method.
    adapted from: https://stackoverflow.com/questions/13021654/get-column-index-from-column-name-in-python-pandas

    Parameters
    ----------
        df: `pd.DataFrame`
            Pandas dataframe that will be used for finding indices.
        queries: `list`
            List of strings, corresponding to either column names or index of the `df` that will be used for finding
            indices.
        type: `{"column", "row:}` (default: "row")
            The type of the queries / search, either `column` (list of queries are from column names) or "row" (list of
            queries are from index names).

    Returns
    -------
        Indices: `np.ndarray`
            One dimensional array for the numeric indices that corresponds to the matches of the queries.
    """

    names = df.columns.values if type == 'column' else df.index.values if type == 'row' else None
    sidx = np.argsort(names)
    Indices = sidx[np.searchsorted(names, queries, sorter=sidx)]

    return Indices


def update_dict(dict1, dict2):
    dict1.update((k, dict2[k]) for k in dict1.keys() & dict2.keys())

    return dict1


def update_n_merge_dict(dict1, dict2):
    dict1.update((k, dict2[k]) for k in dict1.keys() | dict2.keys())
    
    return dict1


def flatten(arr):
    if issparse(arr):
        ret = arr.A.flatten()
    else:
        ret = arr.flatten()
    return ret


def isarray(arr):
    """
        Check if a variable is an array. Essentially the variable has the attribute 'len'
        and it is not a string.
    """
    return hasattr(arr, '__len__') and (not isinstance(arr, str))


def closest_cell(coord, cells):
    cells = np.asarray(cells)
    dist_2 = np.sum((cells - coord) ** 2, axis=1)

    return np.argmin(dist_2)


def elem_prod(X, Y):
    if issparse(X):
        return X.multiply(Y)
    elif issparse(Y):
        return Y.multiply(X)
    else:
        return np.multiply(X, Y)


def norm(x, **kwargs):
    """calculate the norm of an array or matrix"""
    if issparse(x):
        return splinalg.norm(x, **kwargs)
    else:
        return np.linalg.norm(x, **kwargs)


def einsum_correlation(X, Y_i, type="pearson"):
    """calculate pearson or cosine correlation between X (genes/pcs/embeddings x cells) and the velocity vectors Y_i for gene i"""

    if type == "pearson":
        X -= X.mean(axis=1)[:, None]
        Y_i -= np.nanmean(Y_i)
    elif type == "cosine":
        X, Y_i = X, Y_i
    elif type == 'spearman':
        X = stats.rankdata(X, axis=1)
        Y_i = stats.rankdata(Y_i)
    elif type == 'kendalltau':
        corr = np.array([stats.kendalltau(x, Y_i)[0] for x in X])
        return corr[None, :]

    X_norm, Y_norm = norm(X, axis=1), norm(Y_i)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if Y_norm == 0:
            corr = np.zeros(X_norm.shape[0])
        else:
            corr = np.einsum('ij, j', X, Y_i) / (X_norm * Y_norm)[None, :]

    return corr

  
def form_triu_matrix(arr):
    '''
        Construct upper triangle matrix from an 1d array.
    '''
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


def index_condensed_matrix(n, i, j):
    """
    Return the index of an element in a condensed n-by-n square matrix 
    by the row index i and column index j of the square form.

    Arguments
    ---------
        n: int
            Size of the squareform.
        i: int
            Row index of the element in the squareform.
        j: int
            Column index of the element in the the squareform.

    Returns
    -------
        k: int
            The index of the element in the condensed matrix.
    """
    if i == j:
        warnings.warn('Diagonal elements (i=j) are not stored in condensed matrices.')
        return None
    elif i > j:
        i, j = j, i
    return int(i * (n - (i + 3) * 0.5) + j - 1)


def moms2var(m1, m2):
    var = m2 - elem_prod(m1, m1)
    return var


def var2m2(var, m1):
    m2 = var + elem_prod(m1, m1)
    return m2


def gaussian_1d(x, mu=0, sigma=1):
    y = (x-mu)/sigma
    return np.exp(-y*y/2) / np.sqrt(2*np.pi)/sigma


def timeit(method):
    def timed(*args, **kw):
        ti = kw.pop('timeit', False)
        if ti:
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            print ('Time elapsed for %r: %.4f s' %(method.__name__, (te - ts)))
        else:
            result = method(*args, **kw)
        return result
    return timed


def velocity_on_grid(X, V, n_grids, nbrs=None, k=None, 
    smoothness=1, cutoff_coeff=2, margin_coeff=0.025):
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
            nbrs = NNDescent(X, metric='euclidean', n_neighbors=k+1, n_jobs=-1, random_state=19491001)
        else:
            alg = 'ball_tree' if X.shape[1] > 10 else 'kd_tree'
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm=alg, n_jobs=-1).fit(X)

    if hasattr(nbrs, 'kneighbors'): 
        dists, neighs = nbrs.kneighbors(X_grid)
    elif hasattr(nbrs, 'query'): 
        neighs, dists = nbrs.query(X_grid, k=k+1)

    std = np.mean([(g[1] - g[0]) for g in grs])
    # isotropic gaussian kernel
    sigma = smoothness * std
    w = gaussian_1d(dists[:, :k], sigma=sigma)
    if cutoff_coeff is not None:
        w_cut = gaussian_1d(cutoff_coeff*sigma, sigma=sigma)
        w[w<w_cut] = 0
    w_mass = w.sum(1)
    w_mass[w_mass == 0] = 1
    w = (w.T / w_mass).T

    V[np.isnan(V)] = 0
    V_grid = np.einsum('ijk, ij -> ik', V[neighs[:, :k]], w)
    return X_grid, V_grid


def list_top_genes(arr, gene_names, n_top_genes=30, order=-1):
    imax = np.argsort(arr)[::order]
    return gene_names[imax][:n_top_genes]


def table_top_genes(arrs, item_names, gene_names, return_df=True, **kwargs):
    table = {}
    for i, item in enumerate(item_names):
        table[item] = list_top_genes(arrs[i], gene_names, **kwargs)
    if return_df:
        return pd.DataFrame(data=table)
    else:
        return table


# ---------------------------------------------------------------------------------------------------
# data transformation related:
def log1p_(adata, X_data):
    if 'pp_norm_method' not in adata.uns.keys():
        return X_data
    else:
        if adata.uns['pp_norm_method'] is None:
            if issparse(X_data):
                X_data.data = np.log1p(X_data.data)
            else:
                X_data = np.log1p(X_data)

        return X_data


def inverse_norm(adata, layer_x):
    if issparse(layer_x):
        layer_x.data = (
            np.expm1(layer_x.data)
            if adata.uns["pp_norm_method"] == "log1p"
            else 2 ** layer_x.data - 1
            if adata.uns["pp_norm_method"] == "log2"
            else np.exp(layer_x.data) - 1
            if adata.uns["pp_norm_method"] == "log"
            else Freeman_Tukey(layer_x.data + 1, inverse=True)
            if adata.uns["pp_norm_method"] == "Freeman_Tukey"
            else layer_x.data
        )
    else:
        layer_x = (
            np.expm1(layer_x)
            if adata.uns["pp_norm_method"] == "log1p"
            else 2 ** layer_x - 1
            if adata.uns["pp_norm_method"] == "log2"
            else np.exp(layer_x) - 1
            if adata.uns["pp_norm_method"] == "log"
            else Freeman_Tukey(layer_x, inverse=True)
            if adata.uns["pp_norm_method"] == "Freeman_Tukey"
            else layer_x
        )

    return layer_x

# ---------------------------------------------------------------------------------------------------
# dynamics related:
def one_shot_gamma_alpha(k, t, l):
    gamma = -np.log(1 - k) / t
    alpha = l * (gamma / k)[0]
    return gamma, alpha


def one_shot_k(gamma, t):
    k = 1 - np.exp(-gamma * t)
    return k

def one_shot_gamma_alpha_matrix(k, t, U):
    """Assume U is a sparse matrix and only tested on one-shot experiment"""
    Kc = np.clip(k, 0, 1 - 1e-3)
    gamma = -(np.log(1 - Kc) / t)
    alpha = U.multiply((gamma / k)[:, None])

    return gamma, alpha

def _one_shot_gamma_alpha_matrix(K, tau, N, R):
    """original code from Yan"""
    N, R = N.A.T, R.A.T
    K = np.array(K)
    tau = tau[0]
    Kc = np.clip(K, 0, 1-1e-3)
    if np.isscalar(tau):
        B = -np.log(1-Kc)/tau
    else:
        B = -(np.log(1-Kc)[None, :].T/tau).T
    return B, (elem_prod(B, N)/K).T - elem_prod(B, R).T


def compute_velocity_labeling_B(B, alpha, R):
    return (alpha - elem_prod(B, R.T).T)

# ---------------------------------------------------------------------------------------------------
# dynamics related:
def get_valid_bools(adata, filter_gene_mode):
    if filter_gene_mode == "final":
        valid_ind = adata.var.use_for_pca.values
    elif filter_gene_mode == "basic":
        valid_ind = adata.var.pass_basic_filter.values
    elif filter_gene_mode == "no":
        valid_ind = np.repeat([True], adata.shape[1])

    return valid_ind


def log_unnormalized_data(raw, log_unnormalized):
    if issparse(raw):
        raw.data = np.log(raw.data + 1) if log_unnormalized else raw.data
    else:
        raw = np.log(raw + 1) if log_unnormalized else raw

    return raw


def get_data_for_kin_params_estimation(
    subset_adata,
    model,
    use_moments,
    tkey,
    protein_names,
    log_unnormalized,
    NTR_vel,
):
    U, Ul, S, Sl, P, US, U2, S2, = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )  # U: unlabeled unspliced; S: unlabel spliced
    normalized, has_splicing, has_labeling, has_protein, assumption_mRNA = (
        False,
        False,
        False,
        False,
        None,
    )

    mapper = get_mapper()

    # labeling plus splicing
    if (
        np.all(
            ([i in subset_adata.layers.keys() for i in ["X_ul", "X_sl", "X_su"]])
        ) or np.all(
            ([mapper[i] in subset_adata.layers.keys() for i in ["X_ul", "X_sl", "X_su"]])
        )
    ):  # only uu, ul, su, sl provided
        has_splicing, has_labeling, normalized, assumption_mRNA = (
            True,
            True,
            True,
            "ss" if NTR_vel else 'kinetic',
        )
        U = subset_adata.layers[mapper["X_uu"]].T if use_moments \
            else subset_adata.layers["X_uu"].T # unlabel unspliced: U

        Ul = subset_adata.layers[mapper["X_ul"]].T if use_moments \
            else subset_adata.layers["X_ul"].T

        Sl = subset_adata.layers[mapper["X_sl"]].T if use_moments \
            else subset_adata.layers["X_sl"].T

        S = subset_adata.layers[mapper["X_su"]].T if use_moments \
            else subset_adata.layers["X_su"].T # unlabel spliced: S

    elif np.all(
            ([i in subset_adata.layers.keys() for i in ["uu", "ul", "sl", "su"]])
    ):
        has_splicing, has_labeling, normalized, assumption_mRNA = (
            True,
            True,
            False,
            "ss" if NTR_vel else 'kinetic',
        )
        raw, raw_uu = subset_adata.layers["uu"].T, subset_adata.layers["uu"].T
        U = log_unnormalized_data(raw, log_unnormalized)

        raw, raw_ul = subset_adata.layers["ul"].T, subset_adata.layers["ul"].T
        Ul = log_unnormalized_data(raw, log_unnormalized)

        raw, raw_sl = subset_adata.layers["sl"].T, subset_adata.layers["sl"].T
        Sl = log_unnormalized_data(raw, log_unnormalized)

        raw, raw_su = subset_adata.layers["su"].T, subset_adata.layers["su"].T
        S = log_unnormalized_data(raw, log_unnormalized)

    # labeling without splicing
    if (
        ("X_new" in subset_adata.layers.keys() and not use_moments)
        or (mapper["X_new"] in subset_adata.layers.keys() and use_moments)
    ):  # run new / total ratio (NTR)
        has_labeling, normalized, assumption_mRNA = (
            True,
            True,
            "ss" if NTR_vel else 'kinetic',
        )
        U = (
            subset_adata.layers[mapper["X_total"]].T
            - subset_adata.layers[mapper["X_new"]].T
            if use_moments
            else subset_adata.layers["X_total"].T - subset_adata.layers["X_new"].T
        )
        Ul = (
            subset_adata.layers[mapper["X_new"]].T
            if use_moments
            else subset_adata.layers["X_new"].T
        )
    elif "new" in subset_adata.layers.keys():
        has_labeling, assumption_mRNA = (
            True,
            "ss" if NTR_vel else 'kinetic',
        )
        raw, raw_new, old = (
            subset_adata.layers["new"].T,
            subset_adata.layers["new"].T,
            subset_adata.layers["total"].T - subset_adata.layers["new"].T,
        )
        if issparse(raw):
            raw.data = np.log(raw.data + 1) if log_unnormalized else raw.data
            old.data = np.log(old.data + 1) if log_unnormalized else old.data
        else:
            raw = np.log(raw + 1) if log_unnormalized else raw
            old = np.log(old + 1) if log_unnormalized else old
        U = old
        Ul = raw

    # splicing data
    if (
        ("X_unspliced" in subset_adata.layers.keys() and not use_moments)
        or (mapper["X_unspliced"] in subset_adata.layers.keys() and use_moments)
    ):
        has_splicing, normalized, assumption_mRNA = True, True, "kinetic" \
            if tkey in subset_adata.obs.columns else 'ss'
        U = (
            subset_adata.layers[mapper["X_unspliced"]].T
            if use_moments
            else subset_adata.layers["X_unspliced"].T
        )
    elif "unspliced" in subset_adata.layers.keys():
        has_splicing, assumption_mRNA = True, "kinetic" \
            if tkey in subset_adata.obs.columns else 'ss'
        raw, raw_unspliced = (
            subset_adata.layers["unspliced"].T,
            subset_adata.layers["unspliced"].T,
        )
        if issparse(raw):
            raw.data = np.log(raw.data + 1) if log_unnormalized else raw.data
        else:
            raw = np.log(raw + 1) if log_unnormalized else raw
        U = raw
    if (
        ("X_spliced" in subset_adata.layers.keys() and not use_moments)
        or (mapper["X_spliced"] in subset_adata.layers.keys() and use_moments)
    ):
        S = (
            subset_adata.layers[mapper["X_spliced"]].T
            if use_moments
            else subset_adata.layers["X_spliced"].T
        )
    elif "spliced" in subset_adata.layers.keys():
        raw, raw_spliced = (
            subset_adata.layers["spliced"].T,
            subset_adata.layers["spliced"].T,
        )
        if issparse(raw):
            raw.data = np.log(raw.data + 1) if log_unnormalized else raw.data
        else:
            raw = np.log(raw + 1) if log_unnormalized else raw
        S = raw

    ind_for_proteins = None
    if (
        ("X_protein" in subset_adata.obsm.keys() and not use_moments)
        or (mapper["X_protein"] in subset_adata.obsm.keys() and use_moments)
    ):
        P = (
            subset_adata.obsm[mapper["X_protein"]].T
            if use_moments
            else subset_adata.obsm["X_protein"].T
        )
    elif "protein" in subset_adata.obsm.keys():
        P = subset_adata.obsm["protein"].T
    if P is not None:
        has_protein = True
        if protein_names is None:
            warnings.warn(
                "protein layer exists but protein_names is not provided. No estimation will be performed for protein data."
            )
        else:
            protein_names = list(
                set(subset_adata.var.index).intersection(protein_names)
            )
            ind_for_proteins = [
                np.where(subset_adata.var.index == i)[0][0] for i in protein_names
            ]
            subset_adata.var["is_protein_velocity_genes"] = False
            subset_adata.var.loc[ind_for_proteins, "is_protein_velocity_genes"] = True

    experiment_type = "conventional"

    if has_labeling:
        if tkey is None:
            warnings.warn(
                "dynamo finds that your data has labeling, but you didn't provide a `tkey` for"
                "metabolic labeling experiments, so experiment_type is set to be `one-shot`."
            )
            experiment_type = "one-shot"
            t = np.ones_like(subset_adata.n_obs)
        elif tkey in subset_adata.obs.columns:
            t = np.array(subset_adata.obs[tkey], dtype="float")
            if len(np.unique(t)) == 1:
                experiment_type = "one-shot"
            else:
                labeled_sum = U.sum(0) if Ul is None else Ul.sum(0)
                xx, yy = labeled_sum.A1 if issparse(U) else labeled_sum, t
                xm, ym = np.mean(xx), np.mean(yy)
                cov = np.mean(xx * yy) - xm * ym
                var_x = np.mean(xx * xx) - xm * xm

                k = cov / var_x

                # total labeled RNA amount will increase (decrease) in kinetic (degradation) experiments over time.
                experiment_type = "kin" if k > 0 else "deg"
        else:
            raise Exception(
                "the tkey ", tkey, " provided is not a valid column name in .obs."
            )
        if model == "stochastic" and all(
            [x in subset_adata.layers.keys() for x in ["M_tn", "M_nn", "M_tt"]]
        ):
            US, U2, S2 = (
                subset_adata.layers["M_tn"].T,
                subset_adata.layers["M_nn"].T if not has_splicing else None,
                subset_adata.layers["M_tt"].T if not has_splicing else None,
            )
    else:
        t = None
        if model == "stochastic":
            US, U2, S2 = subset_adata.layers["M_us"].T, subset_adata.layers["M_uu"].T, subset_adata.layers["M_ss"].T

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
        has_splicing,
        has_labeling,
        has_protein,
        ind_for_proteins,
        assumption_mRNA,
        experiment_type,
    )

def set_velocity(
    adata,
    vel_U,
    vel_S,
    vel_P,
    _group,
    cur_grp,
    cur_cells_bools,
    valid_ind,
    ind_for_proteins,
):
    cur_cells_ind, valid_ind_ = np.where(cur_cells_bools)[0][:, np.newaxis], np.where(valid_ind)[0]
    if type(vel_U) is not float:
        if cur_grp == _group[0]:
            adata.layers["velocity_U"] = csr_matrix((adata.shape), dtype=np.float64)
        vel_U = vel_U.T.tocsr() if issparse(vel_U) else csr_matrix(vel_U, dtype=np.float64).T
        adata.layers["velocity_U"][cur_cells_ind, valid_ind_] = vel_U
    if type(vel_S) is not float:
        if cur_grp == _group[0]:
            adata.layers["velocity_S"] = csr_matrix((adata.shape), dtype=np.float64)
        vel_S = vel_S.T.tocsr() if issparse(vel_S) else csr_matrix(vel_S, dtype=np.float64).T
        adata.layers["velocity_S"][cur_cells_ind, valid_ind_] = vel_S
    if type(vel_P) is not float:
        if cur_grp == _group[0]:
            adata.obsm["velocity_P"] = csr_matrix(
                (adata.obsm["P"].shape[0], len(ind_for_proteins)), dtype=float
            )
        adata.obsm["velocity_P"][cur_cells_bools, :] = (
            vel_P.T.tocsr() if issparse(vel_P) else csr_matrix(vel_P, dtype=float).T
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
    if experiment_type == "mix_std_stm":
        if alpha is not None:
            if cur_grp == _group[0]:
                adata.varm[kin_param_pre + "alpha"] = np.zeros(
                    (adata.shape[1], alpha[1].shape[1])
                )
            adata.varm[kin_param_pre + "alpha"][valid_ind, :] = alpha[1]
            (
                adata.var[kin_param_pre + "alpha"],
                adata.var[kin_param_pre + "alpha_std"],
            ) = (None, None)
            (
                adata.var.loc[valid_ind, kin_param_pre + "alpha"],
                adata.var.loc[valid_ind, kin_param_pre + "alpha_std"],
            ) = (alpha[1][:, -1], alpha[0])

        if cur_grp == _group[0]:
            (
                adata.var[kin_param_pre + "beta"],
                adata.var[kin_param_pre + "gamma"],
                adata.var[kin_param_pre + "half_life"],
            ) = (None, None, None)

        adata.var.loc[valid_ind, kin_param_pre + "beta"] = beta
        adata.var.loc[valid_ind, kin_param_pre + "gamma"] = gamma
        adata.var.loc[valid_ind, kin_param_pre + "half_life"] = np.log(2) / gamma
    else:
        if alpha is not None:
            if len(alpha.shape) > 1:  # for each cell
                if cur_grp == _group[0]:
                    adata.varm[kin_param_pre + "alpha"] = (
                        csr_matrix(np.zeros(adata.shape[::-1]))
                        if issparse(alpha)
                        else np.zeros(adata.shape[::-1])
                    )  #
                adata.varm[kin_param_pre + "alpha"][valid_ind, :] = alpha  #
                adata.var.loc[valid_ind, kin_param_pre + "alpha"] = alpha.mean(1)
            elif len(alpha.shape) == 1:
                if cur_grp == _group[0]:
                    adata.var[kin_param_pre + "alpha"] = None
                adata.var.loc[valid_ind, kin_param_pre + "alpha"] = alpha

        if cur_grp == _group[0]:
            (
                adata.var[kin_param_pre + "beta"],
                adata.var[kin_param_pre + "gamma"],
                adata.var[kin_param_pre + "half_life"],
            ) = (None, None, None)
        adata.var.loc[valid_ind, kin_param_pre + "beta"] = beta
        adata.var.loc[valid_ind, kin_param_pre + "gamma"] = gamma
        adata.var.loc[valid_ind, kin_param_pre + "half_life"] = None if gamma is None else np.log(2) / gamma

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
                adata.var[kin_param_pre + "alpha_b"],
                adata.var[kin_param_pre + "alpha_r2"],
                adata.var[kin_param_pre + "gamma_b"],
                adata.var[kin_param_pre + "gamma_r2"],
                adata.var[kin_param_pre + "gamma_logLL"],
                adata.var[kin_param_pre + "delta_b"],
                adata.var[kin_param_pre + "delta_r2"],
                adata.var[kin_param_pre + "uu0"],
                adata.var[kin_param_pre + "ul0"],
                adata.var[kin_param_pre + "su0"],
                adata.var[kin_param_pre + "sl0"],
                adata.var[kin_param_pre + "U0"],
                adata.var[kin_param_pre + "S0"],
                adata.var[kin_param_pre + "total0"],
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
            )

        adata.var.loc[valid_ind, kin_param_pre + "alpha_b"] = alpha_intercept
        adata.var.loc[valid_ind, kin_param_pre + "alpha_r2"] = alpha_r2

        if gamma_r2 is not None:
            gamma_r2[~np.isfinite(gamma_r2)] = 0
        adata.var.loc[valid_ind, kin_param_pre + "gamma_b"] = gamma_intercept
        adata.var.loc[valid_ind, kin_param_pre + "gamma_r2"] = gamma_r2
        adata.var.loc[valid_ind, kin_param_pre + "gamma_logLL"] = gamma_logLL

        adata.var.loc[valid_ind, kin_param_pre + "uu0"] = uu0
        adata.var.loc[valid_ind, kin_param_pre + "ul0"] = ul0
        adata.var.loc[valid_ind, kin_param_pre + "su0"] = su0
        adata.var.loc[valid_ind, kin_param_pre + "sl0"] = sl0
        adata.var.loc[valid_ind, kin_param_pre + "U0"] = U0
        adata.var.loc[valid_ind, kin_param_pre + "S0"] = S0
        adata.var.loc[valid_ind, kin_param_pre + "total0"] = total0

        if experiment_type == 'one-shot':
            adata.var[kin_param_pre + "beta_k"] = None
            adata.var[kin_param_pre + "gamma_k"] = None
            adata.var.loc[valid_ind, kin_param_pre + "beta_k"] = beta_k
            adata.var.loc[valid_ind, kin_param_pre + "gamma_k"] = gamma_k

        if ind_for_proteins is not None:
            delta_r2[~np.isfinite(delta_r2)] = 0
            if cur_grp == _group[0]:
                (
                    adata.var[kin_param_pre + "eta"],
                    adata.var[kin_param_pre + "delta"],
                    adata.var[kin_param_pre + "delta_b"],
                    adata.var[kin_param_pre + "delta_r2"],
                    adata.var[kin_param_pre + "p_half_life"],
                ) = (None, None, None, None, None)
            adata.var.loc[valid_ind, kin_param_pre + "eta"][ind_for_proteins] = eta
            adata.var.loc[valid_ind, kin_param_pre + "delta"][ind_for_proteins] = delta
            adata.var.loc[valid_ind, kin_param_pre + "delta_b"][
                ind_for_proteins
            ] = delta_intercept
            adata.var.loc[valid_ind, kin_param_pre + "delta_r2"][
                ind_for_proteins
            ] = delta_r2
            adata.var.loc[valid_ind, kin_param_pre + "p_half_life"][
                ind_for_proteins
            ] = (np.log(2) / delta)

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
    valid_ind,
):
    if cur_grp == _group[0]:
        (
            adata.var[kin_param_pre + "alpha"],
            adata.var[kin_param_pre + "a"],
            adata.var[kin_param_pre + "b"],
            adata.var[kin_param_pre + "alpha_a"],
            adata.var[kin_param_pre + "alpha_i"],
            adata.var[kin_param_pre + "beta"],
            adata.var[kin_param_pre + "p_half_life"],
            adata.var[kin_param_pre + "gamma"],
            adata.var[kin_param_pre + "half_life"],
            adata.var[kin_param_pre + "cost"],
            adata.var[kin_param_pre + "logLL"],
        ) = (None, None, None, None, None, None, None, None, None, None, None)

    adata.var.loc[valid_ind, kin_param_pre + "alpha"] = alpha
    adata.var.loc[valid_ind, kin_param_pre + "a"] = a
    adata.var.loc[valid_ind, kin_param_pre + "b"] = b
    adata.var.loc[valid_ind, kin_param_pre + "alpha_a"] = alpha_a
    adata.var.loc[valid_ind, kin_param_pre + "alpha_i"] = alpha_i
    adata.var.loc[valid_ind, kin_param_pre + "beta"] = beta
    adata.var.loc[valid_ind, kin_param_pre + "gamma"] = gamma
    adata.var.loc[valid_ind, kin_param_pre + "half_life"] = np.log(2) / gamma
    adata.var.loc[valid_ind, kin_param_pre + "cost"] = cost
    adata.var.loc[valid_ind, kin_param_pre + "logLL"] = logLL
    # add extra parameters (u0, uu0, etc.)
    extra_params.columns = [kin_param_pre + i for i in extra_params.columns]
    extra_params = extra_params.set_index(adata.var.index[valid_ind])
    var = pd.concat((adata.var, extra_params), axis=1, sort=False)
    adata.var = var

    return adata


def get_U_S_for_velocity_estimation(
    subset_adata, use_moments, has_splicing, has_labeling, log_unnormalized, NTR
):
    mapper = get_mapper()

    if has_splicing:
        if has_labeling:
            if "X_uu" in subset_adata.layers.keys():  # unlabel spliced: S
                if use_moments:
                    uu, ul, su, sl = (
                        subset_adata.layers[mapper["X_uu"]].T,
                        subset_adata.layers[mapper["X_ul"]].T,
                        subset_adata.layers[mapper["X_su"]].T,
                        subset_adata.layers[mapper["X_sl"]].T,
                    )
                else:
                    uu, ul, su, sl = (
                        subset_adata.layers["X_uu"].T,
                        subset_adata.layers["X_ul"].T,
                        subset_adata.layers["X_su"].T,
                        subset_adata.layers["X_sl"].T,
                    )
            else:
                uu, ul, su, sl = (
                    subset_adata.layers["uu"].T,
                    subset_adata.layers["ul"].T,
                    subset_adata.layers["su"].T,
                    subset_adata.layers["sl"].T,
                )
                if issparse(uu):
                    uu.data = np.log(uu.data + 1) if log_unnormalized else uu.data
                    ul.data = np.log(ul.data + 1) if log_unnormalized else ul.data
                    su.data = np.log(su.data + 1) if log_unnormalized else su.data
                    sl.data = np.log(sl.data + 1) if log_unnormalized else sl.data
                else:
                    uu = np.log(uu + 1) if log_unnormalized else uu
                    ul = np.log(ul + 1) if log_unnormalized else ul
                    su = np.log(su + 1) if log_unnormalized else su
                    sl = np.log(sl + 1) if log_unnormalized else sl
            U, S = (ul + sl, uu + ul + su + sl) if NTR else (uu + ul, su + sl)
            # U, S = (ul + sl, uu + ul + su + sl) if NTR else (ul, sl)
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
                if issparse(U):
                    U.data = np.log(U.data + 1) if log_unnormalized else U.data
                    S.data = np.log(S.data + 1) if log_unnormalized else S.data
                else:
                    U = np.log(U + 1) if log_unnormalized else U
                    S = np.log(S + 1) if log_unnormalized else S
    else:
        if ("X_new" in subset_adata.layers.keys()) or (
            mapper["X_new"] in subset_adata.layers.keys()
        ):  # run new / total ratio (NTR)
            if use_moments:
                U = subset_adata.layers[mapper["X_new"]].T
                S = (
                    subset_adata.layers[mapper["X_total"]].T
                    if NTR
                    else subset_adata.layers[mapper["X_total"]].T
                    - subset_adata.layers[mapper["X_new"]].T
                )
            else:
                U = subset_adata.layers["X_new"].T
                S = (
                    subset_adata.layers["X_total"].T
                    if NTR
                    else subset_adata.layers["X_total"].T
                    - subset_adata.layers["X_new"].T
                )
        elif "new" in subset_adata.layers.keys():
            U = subset_adata.layers["new"].T
            S = (
                subset_adata.layers["total"].T
                if NTR
                else subset_adata.layers["total"].T - subset_adata.layers["new"].T
            )
            if issparse(U):
                U.data = np.log(U.data + 1) if log_unnormalized else U.data
                S.data = np.log(S.data + 1) if log_unnormalized else S.data
            else:
                U = np.log(U + 1) if log_unnormalized else U
                S = np.log(S + 1) if log_unnormalized else S

    return U, S


# ---------------------------------------------------------------------------------------------------
# retrieving data related

def fetch_X_data(adata, genes, layer, basis=None):
    if basis is not None:
        return None, adata.obsm['X_' + basis]

    if genes is not None:
        genes = adata.var_names.intersection(genes).to_list()
        if len(genes) == 0:
            raise ValueError(f'No genes from your genes list appear in your adata object.')

    if layer == None:
        if genes is not None:
            X_data = adata[:, genes].X
        else:
            if 'use_for_dynamics' not in adata.var.keys():
                X_data = adata.X
                genes = adata.var_names
            else:
                X_data = adata[:, adata.var.use_for_dynamics].X
                genes = adata.var_names[adata.var.use_for_dynamics]
    else:
        if genes is not None:
            X_data = adata[:, genes].layers[layer]
        else:
            if 'use_for_dynamics' not in adata.var.keys():
                X_data = adata.layers[layer]
                genes = adata.var_names
            else:
                X_data = adata[:, adata.var.use_for_dynamics].layers[layer]
                genes = adata.var_names[adata.var.use_for_dynamics]

            X_data = log1p_(adata, X_data)

    return genes, X_data

# ---------------------------------------------------------------------------------------------------
# estimation related

def calc_R2(X, Y, k, f=lambda X, k: np.einsum('ij,i -> ij', X, k)):
    """calculate R-square. X, Y: n_species (mu, sigma) x n_obs"""
    if X.ndim == 1:
        X = X[None]
    if Y.ndim == 1:
        Y = Y[None]
    if np.isscalar(k):
        k = np.array([k])
    
    Y_bar = np.mean(Y, 1)
    d = Y.T - Y_bar
    SS_tot = np.sum(np.einsum('ij,ij -> i', d, d))

    F = f(X, k)
    d = F - Y
    SS_res = np.sum(np.einsum('ij,ij -> j', d, d))

    return 1 - SS_res/SS_tot


def norm_loglikelihood(x, mu, sig):
    """Calculate log-likelihood for the data.
    """
    err = (x - mu) / sig
    ll = -len(err)/2*np.log(2*np.pi) - np.sum(np.log(sig)) - 0.5*err.dot(err.T)
    return np.sum(ll, 0)


def calc_norm_loglikelihood(X, Y, k, f=lambda X, k: np.einsum('ij,i -> ij', X, k)):
    """calculate log likelihood based on normal distribution. X, Y: n_species (mu, sigma) x n_obs"""
    if X.ndim == 1:
        X = X[None]
    if Y.ndim == 1:
        Y = Y[None]
    if np.isscalar(k):
        k = np.array([k])

    n = X.shape[0]
    F = f(X, k)

    d = F - Y
    sig = np.einsum('ij,ij -> i', d, d)

    LogLL = 0
    for i in range(Y.shape[0]):
        LogLL += norm_loglikelihood(Y[i], F[i], np.sqrt(sig[i] / n))

    return LogLL

# ---------------------------------------------------------------------------------------------------
# velocity related

def find_extreme(s, u, normalize=True, perc_left=None, perc_right=None):
    s, u = (s.A if issparse(s) else s, u.A if issparse(u) else u)

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

def get_group_params_indices(adata, param_name):
    return adata.var.columns.str.endswith(param_name)


def set_velocity_genes(
    adata,
    vkey="velocity_S",
    min_r2=0.01,
    min_alpha=0.01,
    min_gamma=0.01,
    min_delta=0.01,
    use_for_dynamics=True,
):
    layer = vkey.split("_")[1]

    # the following parameters aggreation for different groups can be improved later
    if layer == "U":
        if 'alpha' not in adata.var.columns:
            is_group_alpha, is_group_alpha_r2 = get_group_params_indices(adata, 'alpha'), \
                                                get_group_params_indices(adata, 'alpha_r2')
            if is_group_alpha.sum() > 0:
                adata.var['alpha'] = adata.var.loc[:, is_group_alpha].mean(1, skipna=True)
                adata.var['alpha_r2'] = adata.var.loc[:, is_group_alpha_r2].mean(1, skipna=True)
            else:
                raise Exception("there is no alpha/alpha_r2 parameter estimated for your adata object")

        if 'alpha_r2' not in adata.var.columns: adata.var['alpha_r2'] = None
        if np.all(adata.var.alpha_r2.values == None):
            adata.var.alpha_r2 = 1
        adata.var["use_for_velocity"] = (
            (adata.var.alpha > min_alpha)
            & (adata.var.alpha_r2 > min_r2)
            & adata.var.use_for_dynamics
            if use_for_dynamics
            else (adata.var.alpha > min_alpha) & (adata.var.alpha_r2 > min_r2)
        )
    elif layer == "S":
        if 'gamma' not in adata.var.columns:
            is_group_gamma, is_group_gamma_r2 = get_group_params_indices(adata, 'gamma'), \
                                                get_group_params_indices(adata, 'gamma_r2')
            if is_group_gamma.sum() > 0:
                adata.var['gamma'] = adata.var.loc[:, is_group_gamma].mean(1, skipna=True)
                adata.var['gamma_r2'] = adata.var.loc[:, is_group_gamma_r2].mean(1, skipna=True)
            else:
                raise Exception("there is no gamma/gamma_r2 parameter estimated for your adata object")

        if 'gamma_r2' not in adata.var.columns: adata.var['gamma_r2'] = None
        if np.all(adata.var.gamma_r2.values == None): adata.var.gamma_r2 = 1
        adata.var["use_for_velocity"] = (
            (adata.var.gamma > min_gamma)
            & (adata.var.gamma_r2 > min_r2)
            & adata.var.use_for_dynamics
            if use_for_dynamics
            else (adata.var.gamma > min_gamma) & (adata.var.gamma_r2 > min_r2)
        )
    elif layer == "P":
        if 'delta' not in adata.var.columns:
            is_group_delta, is_group_delta_r2 = get_group_params_indices(adata, 'delta'), \
                                                get_group_params_indices(adata, 'delta_r2')
            if is_group_delta.sum() > 0:
                adata.var['delta'] = adata.var.loc[:, is_group_delta].mean(1, skipna=True)
                adata.var['delta_r2'] = adata.var.loc[:, is_group_delta_r2].mean(1, skipna=True)
            else:
                raise Exception("there is no delta/delta_r2 parameter estimated for your adata object")

        if 'delta_r2' not in adata.var.columns: adata.var['delta_r2'] = None
        if np.all(adata.var.delta_r2.values == None):
            adata.var.delta_r2 = 1
        adata.var["use_for_velocity"] = (
            (adata.var.delta > min_delta)
            & (adata.var.delta_r2 > min_r2)
            & adata.var.use_for_dynamics
            if use_for_dynamics
            else (adata.var.delta > min_delta) & (adata.var.delta_r2 > min_r2)
        )

    return adata


def get_ekey_vkey_from_adata(adata):
    """ekey: expression from which to extrapolate velocity; vkey: velocity key; layer: the states cells will be used in
    velocity embedding. """
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
            if "X_uu" in adata.layers.keys():  # unlabel spliced: S
                if use_smoothed:
                    uu, ul, su, sl = (
                        adata.layers[mapper["X_uu"]],
                        adata.layers[mapper["X_ul"]],
                        adata.layers[mapper["X_su"]],
                        adata.layers[mapper["X_sl"]],
                    )
                    if 'M_n' not in adata.layers.keys():
                        adata.layers["M_n"] = ul + sl
                    elif NTR and "M_t" not in adata.layers.keys():
                        adata.layers["M_t"] = uu + ul + su + sl
                    elif not NTR and "M_s" not in adata.layers.keys():
                        adata.layers["M_s"] = sl + su

                uu, ul, su, sl = (
                    adata.layers["X_uu"],
                    adata.layers["X_ul"],
                    adata.layers["X_su"],
                    adata.layers["X_sl"],
                )

                if 'X_new' not in adata.layers.keys():
                    adata.layers["X_new"] = ul + sl
                elif NTR and "X_total" not in adata.layers.keys():
                    adata.layers["X_total"] = uu + ul + su + sl
                elif not NTR and "X_spliced" not in adata.layers.keys():
                    adata.layers["X_spliced"] = sl + su
            else:
                raise Exception(
                    "The input data you have is not normalized or normalized + smoothed!"
                )

            if experiment_type == "kin":
                ekey, vkey, layer = (
                    (mapper["X_total"] if NTR else mapper["X_spliced"], "velocity_S", ("X_total" if NTR else "X_spliced"))
                    if use_smoothed
                    else ("X_total" if NTR else "X_spliced", "velocity_S", "X_total" if NTR else "X_spliced")
                )
            elif experiment_type == "deg":
                ekey, vkey, layer = (
                    (mapper["X_total"] if NTR else mapper["X_spliced"], "velocity_S", ("X_total" if NTR else "X_spliced"))
                    if use_smoothed
                    else ("X_total" if NTR else "X_spliced", "velocity_S", "X_total" if NTR else "X_spliced")
                )
            elif experiment_type == "one_shot":
                ekey, vkey, layer = (
                    (mapper["X_total"] if NTR else mapper["X_spliced"], "velocity_S", ("X_total" if NTR else "X_spliced"))
                    if use_smoothed
                    else ("X_total" if NTR else "X_spliced", "velocity_S", "X_total" if NTR else "X_spliced")
                )
            elif experiment_type == "mix_std_stm":
                ekey, vkey, layer = (
                    (mapper["X_total"] if NTR else mapper["X_spliced"], "velocity_S", ("X_total" if NTR else "X_spliced"))
                    if use_smoothed
                    else ("X_total" if NTR else "X_spliced", "velocity_S", "X_total" if NTR else "X_spliced")
                )
        else:
            if not (("X_unspliced" in adata.layers.keys()) or (
                mapper["X_unspliced"] in adata.layers.keys()
            )):
                raise Exception(
                    "The input data you have is not normalized/log transformed or smoothed and normalized/log transformed!"
                )
            ekey, vkey, layer = (
                (mapper["X_spliced"], "velocity_S", "X_spliced")
                if use_smoothed
                else ("X_spliced", "velocity_S", "X_spliced")
            )
    else:
        # use_smoothed: False
        if ("X_new" in adata.layers.keys()) or (
            mapper["X_new"] in adata.layers.keys
        ):  # run new / total ratio (NTR)
            # we may also create M_U, M_S layers?
            if experiment_type == "kin":
                ekey, vkey, layer = (
                    (mapper["X_total"], "velocity_S", "X_total")
                    if use_smoothed
                    else ("X_total", "velocity_S", "X_total")
                )
            elif experiment_type == "deg":
                ekey, vkey, layer = (
                    (mapper["X_total"], "velocity_S", "X_total")
                    if use_smoothed
                    else ("X_total", "velocity_S", "X_total")
                )
            elif experiment_type == "one-shot" or experiment_type == "one_shot":
                ekey, vkey, layer = (
                    (mapper["X_total"], "velocity_S", "X_total")
                    if use_smoothed
                    else ("X_total", "velocity_S", "X_total")
                )
            elif experiment_type == "mix_std_stm":
                ekey, vkey, layer = (
                    (mapper["X_total"], "velocity_S", "X_total")
                    if use_smoothed
                    else ("X_total", "velocity_S", "X_total")
                )

        else:
            raise Exception(
                "The input data you have is not normalized/log trnasformed or smoothed and normalized/log trnasformed!"
            )

    return ekey, vkey, layer


# ---------------------------------------------------------------------------------------------------
# cell velocities related
def get_iterative_indices(indices, index, n_recurse_neighbors=2, max_neighs=None):
    # These codes are borrowed from scvelo. Need to be rewritten later.
    def iterate_indices(indices, index, n_recurse_neighbors):
        if n_recurse_neighbors > 1:
            index = iterate_indices(indices, index, n_recurse_neighbors - 1)
        ix = np.append(index, indices[index])  # append to also include direct neighbors, otherwise ix = indices[index]
        if np.isnan(ix).any():
            ix = ix[~np.isnan(ix)]
        return ix.astype(int)

    indices = np.unique(iterate_indices(indices, index, n_recurse_neighbors))
    if max_neighs is not None and len(indices) > max_neighs:
        indices = np.random.choice(indices, max_neighs, replace=False)
    return indices


def append_iterative_neighbor_indices(indices, n_recurse_neighbors=2, max_neighs=None):
    indices_rec = []
    for i in range(indices.shape[0]):
        neig = get_iterative_indices(indices, i, n_recurse_neighbors, max_neighs)
        indices_rec.append(neig)
    return indices_rec

def split_velocity_graph(G, neg_cells_trick=True):
    """split velocity graph (built either with correlation or with cosine kernel
     into one positive graph and one negative graph"""

    if not issparse(G): G = csr_matrix(G)
    if neg_cells_trick: G_ = G.copy()
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

def linear_least_squares(a, b, residuals=False):
    """
    Return the least-squares solution to a linear matrix equation.
    Solves the equation `a x = b` by computing a vector `x` that
    minimizes the Euclidean 2-norm `|| b - a x ||^2`.  The equation may
    be under-, well-, or over- determined (i.e., the number of
    linearly independent rows of `a` can be less than, equal to, or
    greater than its number of linearly independent columns).  If `a`
    is square and of full rank, then `x` (but for round-off error) is
    the "exact" solution of the equation.
    Parameters
    ----------
    a : (M, N) array_like
        "Coefficient" matrix.
    b : (M,) array_like
        Ordinate or "dependent variable" values.
    residuals : bool
        Compute the residuals associated with the least-squares solution
    Returns
    -------
    x : (M,) ndarray
        Least-squares solution. The shape of `x` depends on the shape of
        `b`.
    residuals : int (Optional)
        Sums of residuals; squared Euclidean 2-norm for each column in
        ``b - a*x``.
    """
    if type(a) != np.ndarray or not a.flags['C_CONTIGUOUS']:
        warnings.warn('Matrix a is not a C-contiguous numpy array. The solver will create a copy, which will result' + \
             ' in increased memory usage.')

    a = np.asarray(a, order='c')
    i = dgemm(alpha=1.0, a=a.T, b=a.T, trans_b=True)
    x = np.linalg.solve(i, dgemm(alpha=1.0, a=a.T, b=b))

    if residuals:
        return x, np.linalg.norm(np.dot(a, x) - b)
    else:
        return x

def integrate_vf(
    init_states, t, args, integration_direction, f, interpolation_num=None, average=True
):
    """integrating along vector field function"""

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
            raise Exception(
                "both, forward, backward are the only valid direction argument strings"
            )

        if interpolation_num is not None:
            vids = np.where((np.diff(y.T) < 1e-3).sum(0) < y.shape[1])[0]
            valid_ids = vids if valid_ids is None else list(set(valid_ids).union(vids))

        Y = y if Y is None else np.vstack((Y, y))

    if interpolation_num is not None:
        valid_t_trans = t_trans[valid_ids]

        _t, _Y = None, None
        for i in range(n_cell):
            cur_Y = Y[i : (i + 1) * len(t_trans), :][valid_ids, :]
            t_linspace = np.linspace(
                valid_t_trans[0], valid_t_trans[-1], interpolation_num
            )
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

