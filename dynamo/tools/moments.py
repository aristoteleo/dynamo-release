import numpy as np
import warnings
from anndata import AnnData
from scipy.sparse import issparse, csr_matrix, lil_matrix, diags
from tqdm import tqdm
from .utils_moments import estimation
from .utils import get_mapper, elem_prod
from .connectivity import mnn, normalize_knn_graph, umap_conn_indices_dist_embedding
from ..preprocessing.utils import get_layer_keys, allowed_X_layer_names, pca


# ---------------------------------------------------------------------------------------------------
# use for calculating moments for stochastic model:
def moments(adata, use_gaussian_kernel=True, use_mnn=False, layers="all"):
    mapper = get_mapper()
    only_splicing, only_labeling, splicing_and_labeling = allowed_X_layer_names()

    if use_mnn:
        if "mnn" not in adata.uns.keys():
            adata = mnn(
                adata,
                n_pca_components=30,
                layers="all",
                use_pca_fit=True,
                save_all_to_adata=False,
            )
        kNN = adata.uns["mnn"]
    else:
        if 'X_pca' not in adata.obsm.keys():
            CM = adata.X
            cm_genesums = CM.sum(axis=0)
            valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
            valid_ind = np.array(valid_ind).flatten()
            CM = CM[:, valid_ind]
            adata, fit, _ = pca(adata, CM)

            adata.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_[1:]
        X = adata.obsm["X_pca"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kNN, knn_indices, knn_dists, _ = umap_conn_indices_dist_embedding(
                X, n_neighbors=30, return_mapper=False
            )

    if use_gaussian_kernel and not use_mnn:
        conn = gaussian_kernel(X, knn_indices, sigma=10, k=None, dists=knn_dists)
    else:
        conn = normalize_knn_graph(kNN > 0)

    layers = get_layer_keys(adata, layers, False, False)
    layers = [
        layer
        for layer in layers
        if layer.startswith("X_")
           and (not layer.endswith("matrix") and not layer.endswith("ambiguous"))
    ]
    layers.sort(
        reverse=True
    )  # ensure we get M_us, M_tn, etc (instead of M_su or M_nt).
    for i, layer in enumerate(layers):
        layer_x = adata.layers[layer].copy()
        layer_x_group = np.where([layer in x for x in
                                  [only_splicing, only_labeling, splicing_and_labeling]])[0][0]

        if issparse(layer_x):
            layer_x.data = (
                2 ** layer_x.data - 1
                if adata.uns["pp_log"] == "log2"
                else np.exp(layer_x.data) - 1
            )
        else:
            layer_x = (
                2 ** layer_x - 1
                if adata.uns["pp_log"] == "log2"
                else np.exp(layer_x) - 1
            )

        if mapper[layer] not in adata.layers.keys():
            if use_gaussian_kernel:
                adata.layers[mapper[layer]], conn = calc_1nd_moment(layer_x, conn, True)
            else:
                conn.dot(layer_x)

        for layer2 in layers[i:]:
            layer_y = adata.layers[layer2].copy()

            layer_y_group = np.where([layer2 in x for x in
                                      [only_splicing, only_labeling, splicing_and_labeling]])[0][0]
            if layer_x_group != layer_y_group:
                continue

            if issparse(layer_y):
                layer_y.data = (
                    2 ** layer_y.data - 1
                    if adata.uns["pp_log"] == "log2"
                    else np.exp(layer_y.data) - 1
                )
            else:
                layer_y = (
                    2 ** layer_y - 1
                    if adata.uns["pp_log"] == "log2"
                    else np.exp(layer_y) - 1
                )

            if mapper[layer2] not in adata.layers.keys():
                adata.layers[mapper[layer2]] = (
                    calc_1nd_moment(layer_y, conn, False)
                    if use_gaussian_kernel
                    else conn.dot(layer_y)
                )

            adata.layers["M_" + layer[2] + layer2[2]] = calc_2nd_moment(
                layer_x, layer_y, conn, mX=layer_x, mY=layer_y
            )

    if (
            "X_protein" in adata.obsm.keys()
    ):  # may need to update with mnn or just use knn from protein layer itself.
        adata.obsm[mapper["X_protein"]] = conn.dot(adata.obsm["X_protein"])
    adata.uns['moments_con'] = conn

    return adata

# ---------------------------------------------------------------------------------------------------
# use for kinetic assumption
def get_layer_pair(layer):
    pair = {'new': "old", 'old': "new",
            'X_new': "X_total", "X_total": 'X_new'}
    return pair[layer]

def prepare_data_deterministic(adata, genes, time, layers,
                               use_total_layers=True, log=False):
    from ..preprocessing.utils import sz_util, normalize_util
    if use_total_layers:
        if 'total_Size_Factor' not in adata.obs.keys():
            total_layers = ["uu", "ul", "su", "sl"] if 'uu' in adata.layers.keys() else ['total']
            sfs, _ = sz_util(adata, '_total_', False, "median", np.nanmean, total_layers=total_layers)
        else:
            sfs = adata.obs.total_Size_Factor
        sfs_x, sfs_y = sfs[:, None], sfs[:, None]

    m = [None] * len(layers)
    v = [None] * len(layers)
    for i, layer in enumerate(layers):
        if layer in ['X_new', 'new']:
            if layer == 'X_new':
                x_layer = adata[:, genes].layers[layer]
                x_layer = adata[:, genes].layers[get_layer_pair(layer)] - x_layer
            else:
                if not use_total_layers:
                    sfs_x, _ = sz_util(adata, layer, False, "median", np.nanmean, total_layers=None)
                    sfs_y, _ = sz_util(adata, get_layer_pair(layer), False, "median", np.nanmean, total_layers=None)
                    sfs_x, sfs_y = sfs_x[:, None], sfs_y[:, None]

                x_layer = normalize_util(adata[:, genes].layers[layer], sfs_x, relative_expr=True, pseudo_expr=0,
                                   norm_method=None)
                y_layer = normalize_util(adata[:, genes].layers[layer], sfs_y, relative_expr=True, pseudo_expr=0,
                                   norm_method=None)

                x_layer = y_layer - x_layer
        else:
            if layer == 'X_new':
                x_layer = adata[:, genes].layers[layer].A
            else:
                x_layer = normalize_util(adata[:, genes].layers[layer], sfs[:, None], relative_expr=True, pseudo_expr=0,
                                   norm_method=None)

        x_layer = np.log(x_layer + 1) if log else x_layer
        m[i], v[i], _ = calc_12_mom_labeling(x_layer.T, time)

    return m, v # each list element corresponds to a layer

def prepare_data_has_splicing(adata, genes, time, layer_u, layer_s,
                              use_total_layers=True,
                              return_cov=False):
    """Prepare data when assumption is kinetic and data has splicing"""
    from ..preprocessing.utils import sz_util, normalize_util
    res = [0] * len(genes)

    if use_total_layers:
        if 'total_Size_Factor' not in adata.obs.keys():
            total_layers = ["uu", "ul", "su", "sl"]
            sfs, _ = sz_util(adata, '_total_', False, "median", np.nanmean, total_layers=total_layers)
            sfs_u, sfs_s = sfs[:, None], sfs[:, None]
        else:
            sfs = adata.obs.total_Size_Factor
            sfs_u, sfs_s = sfs[:, None], sfs[:, None]
    else:
        sfs_u, _ = sz_util(adata, layer_u, False, "median", np.nanmean, total_layers=None)
        sfs_s, _ = sz_util(adata, layer_s, False, "median", np.nanmean, total_layers=None)
        sfs_u, sfs_s = sfs_u[:, None], sfs_s[:, None]

    U = normalize_util(adata[:, genes].layers[layer_u], sfs_u, relative_expr=True, pseudo_expr=0, norm_method=None)
    S = normalize_util(adata[:, genes].layers[layer_s], sfs_s, relative_expr=True, pseudo_expr=0, norm_method=None)

    for i, g in enumerate(genes):
        u = U[:, i].A.flatten() if issparse(U) else U[:, i]
        s = S[:, i].A.flatten() if issparse(S) else S[:, i]
        ut = strat_mom(u, time, np.mean)
        st = strat_mom(s, time, np.mean)
        uut = strat_mom(elem_prod(u, u), time, np.mean)
        ust = strat_mom(elem_prod(u, s), time, np.mean)
        sst = strat_mom(elem_prod(s, s), time, np.mean)
        x = np.array([ut, st, uut, sst, ust]) if return_cov else np.array([ut, st, uut, sst])

        res[i] = x

    return res


def prepare_data_no_splicing(adata, genes, time, layer, use_total_layers=True):
    """Prepare data when assumption is kinetic and data has no splicing"""
    from ..preprocessing.utils import get_sz_exprs, sz_util, normalize_util
    res = [0] * len(genes)

    if use_total_layers:
        if 'total_Size_Factor' not in adata.obs.keys():
            sfs, _ = sz_util(adata, 'total', False, "median", np.nanmean, total_layers=None)
        else:
            sfs = adata.obs.total_Size_Factor
    else:
        sfs, _ = sz_util(adata, layer, False, "median", np.nanmean, total_layers=None)

    U = normalize_util(adata[:, genes].layers[layer], sfs, relative_expr=True, pseudo_expr=0, norm_method=None)

    for i, g in enumerate(genes):
        u = U[:, i].A.flatten() if issparse(U) else U[:, i]
        ut = strat_mom(u, time, np.mean)
        uut = strat_mom(elem_prod(u, u), time, np.mean)
        res[i] = np.array([ut, uut])

    return res

# ---------------------------------------------------------------------------------------------------
# moment related:

def stratify(arr, strata):
    s = np.unique(strata)
    return [arr[strata == s[i]] for i in range(len(s))]


def strat_mom(arr, strata, fcn_mom):
    x = stratify(arr, strata)
    return np.array([fcn_mom(y) for y in x])


def calc_mom_all_genes(T, adata, fcn_mom):
    ng = adata.var.shape[0]
    nT = len(np.unique(T))
    Mn = np.zeros((ng, nT))
    Mo = np.zeros((ng, nT))
    Mt = np.zeros((ng, nT))
    Mr = np.zeros((ng, nT))
    for g in tqdm(range(ng), desc="calculating 1/2 moments"):
        L = np.array(adata[:, g].layers["new"], dtype=float)
        U = np.array(adata[:, g].layers["old"], dtype=float)
        rho = L / (L + U + 0.01)
        Mn[g] = strat_mom(L, T, fcn_mom)
        Mo[g] = strat_mom(U, T, fcn_mom)
        Mt[g] = strat_mom(L + U, T, fcn_mom)
        Mr[g] = strat_mom(rho, T, fcn_mom)
    return Mn, Mo, Mt, Mr


def _calc_1nd_moment(X, W, normalize_W=True):
    """deprecated"""
    if normalize_W:
        d = np.sum(W, 1)
        W = np.diag(1 / d) @ W
    return W @ X


def _calc_2nd_moment(X, Y, W, normalize_W=True, center=False, mX=None, mY=None):
    """deprecated"""
    if normalize_W:
        d = np.sum(W, 1)
        W = np.diag(1 / d) @ W
    XY = np.multiply(W @ Y, X)
    if center:
        mX = calc_1nd_moment(X, W, False) if mX is None else mX
        mY = calc_1nd_moment(Y, W, False) if mY is None else mY
        XY = XY - np.multiply(mX, mY)
    return XY


def gaussian_kernel(X, nbr_idx, sigma, k=None, dists=None):
    n = X.shape[0]
    if dists is None:
        dists = []
        for i in range(n):
            d = X[nbr_idx[i][:k]] - X[i]
            dists.append(np.sum(elem_prod(d, d), 1).flatten())
    W = lil_matrix((n, n))
    s2_inv = 1 / (2 * sigma ** 2)
    for i in range(n):
        W[i, nbr_idx[i][:k]] = np.exp(-s2_inv * dists[i][:k] ** 2)

    return csr_matrix(W)


def calc_12_mom_labeling(data, t, calculate_2_mom=True):
    t_uniq = np.unique(t)

    m = np.zeros((data.shape[0], len(t_uniq)))
    if calculate_2_mom: v =np.zeros((data.shape[0], len(t_uniq)))

    for i in range(data.shape[0]):
        data_ = (
            np.array(data[i].A.flatten(), dtype=float)
            if issparse(data)
            else np.array(data[i], dtype=float)
        )  # consider using the `adata.obs_vector`, `adata.var_vector` methods or accessing the array directly.
        m[i] = strat_mom(data_, t, np.nanmean)
        if calculate_2_mom: v[i] = strat_mom(data_, t, np.nanvar)

    return (m, v, t_uniq) if calculate_2_mom else (m, t_uniq)


def calc_1nd_moment(X, W, normalize_W=True):
    if normalize_W:
        if type(W) == np.ndarray:
            d = np.sum(W, 1).flatten()
        else:
            d = np.sum(W, 1).A.flatten()
        W = diags(1 / d) @ W if issparse(W) else np.diag(1 / d) @ W
        return W @ X, W
    else:
        return W @ X


def calc_2nd_moment(X, Y, W, normalize_W=True, center=False, mX=None, mY=None):
    if normalize_W:
        if type(W) == np.ndarray:
            d = np.sum(W, 1).flatten()
        else:
            d = W.sum(1).A.flatten()
        W = diags(1 / d) @ W if issparse(W) else np.diag(1 / d) @ W

    XY = W @ elem_prod(Y, X)

    if center:
        mX = calc_1nd_moment(X, W, False) if mX is None else mX
        mY = calc_1nd_moment(Y, W, False) if mY is None else mY
        XY = XY - elem_prod(mX, mY)

    return XY

# ---------------------------------------------------------------------------------------------------
# old moment estimation code
class MomData(AnnData):
    """deprecated"""
    def __init__(self, adata, time_key="Time", has_nan=False):
        # self.data = adata
        self.__dict__ = adata.__dict__
        # calculate first and second moments from data
        self.times = np.array(self.obs[time_key].values, dtype=float)
        self.uniq_times = np.unique(self.times)
        nT = self.get_n_times()
        ng = self.get_n_genes()
        self.M = np.zeros((ng, nT))  # first moments (data)
        self.V = np.zeros((ng, nT))  # second moments (data)
        for g in tqdm(range(ng), desc="calculating 1/2 moments"):
            tmp = self[:, g].layers["new"]
            L = (
                np.array(tmp.A, dtype=float)
                if issparse(tmp)
                else np.array(tmp, dtype=float)
            )  # consider using the `adata.obs_vector`, `adata.var_vector` methods or accessing the array directly.
            if has_nan:
                self.M[g] = strat_mom(L, self.times, np.nanmean)
                self.V[g] = strat_mom(L, self.times, np.nanvar)
            else:
                self.M[g] = strat_mom(L, self.times, np.mean)
                self.V[g] = strat_mom(L, self.times, np.var)

    def get_n_genes(self):
        return self.var.shape[0]

    def get_n_cell(self):
        return self.obs.shape[0]

    def get_n_times(self):
        return len(self.uniq_times)


class Estimation:
    """deprecated"""
    def __init__(
        self,
        adata,
        adata_u=None,
        time_key="Time",
        normalize=True,
        param_ranges=None,
        has_nan=False,
    ):
        # initialize Estimation
        self.data = MomData(adata, time_key, has_nan)
        self.data_u = (
            MomData(adata_u, time_key, has_nan) if adata_u is not None else None
        )
        if param_ranges is None:
            param_ranges = {
                "a": [0, 10],
                "b": [0, 10],
                "alpha_a": [10, 1000],
                "alpha_i": [0, 10],
                "beta": [0, 10],
                "gamma": [0, 10],
            }
        self.normalize = normalize
        self.param_ranges = param_ranges
        self.n_params = len(param_ranges)

    def param_array2dict(self, parr):
        if parr.ndim == 1:
            return {
                "a": parr[0],
                "b": parr[1],
                "alpha_a": parr[2],
                "alpha_i": parr[3],
                "beta": parr[4],
                "gamma": parr[5],
            }
        else:
            return {
                "a": parr[:, 0],
                "b": parr[:, 1],
                "alpha_a": parr[:, 2],
                "alpha_i": parr[:, 3],
                "beta": parr[:, 4],
                "gamma": parr[:, 5],
            }

    def fit_gene(self, gene_no, n_p0=10):
        estm = estimation(list(self.param_ranges.values()))
        if self.data_u is None:
            m = self.data.M[gene_no, :].T
            v = self.data.V[gene_no, :].T
            x_data = np.vstack((m, v))
            popt, cost = estm.fit_lsq(
                self.data.uniq_times,
                x_data,
                p0=None,
                n_p0=n_p0,
                normalize=self.normalize,
                experiment_type="nosplice",
            )
        else:
            mu = self.data_u.M[gene_no, :].T
            ms = self.data.M[gene_no, :].T
            vu = self.data_u.V[gene_no, :].T
            vs = self.data.V[gene_no, :].T
            x_data = np.vstack((mu, ms, vu, vs))
            popt, cost = estm.fit_lsq(
                self.data.uniq_times,
                x_data,
                p0=None,
                n_p0=n_p0,
                normalize=self.normalize,
                experiment_type=None,
            )
        return popt, cost

    def fit(self, n_p0=10):
        ng = self.data.get_n_genes()
        params = np.zeros((ng, self.n_params))
        costs = np.zeros(ng)
        for i in tqdm(range(ng), desc="fitting genes"):
            params[i], costs[i] = self.fit_gene(i, n_p0)
        return params, costs


# ---------------------------------------------------------------------------------------------------
# use for kinetic assumption with full data, deprecated
def moment_model(adata, subset_adata, _group, cur_grp, log_unnormalized, tkey):
    """deprecated"""
    # a few hard code to set up data for moment mode:
    if "uu" in subset_adata.layers.keys() or "X_uu" in subset_adata.layers.keys():
        if log_unnormalized and "X_uu" not in subset_adata.layers.keys():
            if issparse(subset_adata.layers["uu"]):
                (
                    subset_adata.layers["uu"].data,
                    subset_adata.layers["ul"].data,
                    subset_adata.layers["su"].data,
                    subset_adata.layers["sl"].data,
                ) = (
                    np.log(subset_adata.layers["uu"].data + 1),
                    np.log(subset_adata.layers["ul"].data + 1),
                    np.log(subset_adata.layers["su"].data + 1),
                    np.log(subset_adata.layers["sl"].data + 1),
                )
            else:
                (
                    subset_adata.layers["uu"],
                    subset_adata.layers["ul"],
                    subset_adata.layers["su"],
                    subset_adata.layers["sl"],
                ) = (
                    np.log(subset_adata.layers["uu"] + 1),
                    np.log(subset_adata.layers["ul"] + 1),
                    np.log(subset_adata.layers["su"] + 1),
                    np.log(subset_adata.layers["sl"] + 1),
                )

        subset_adata_u, subset_adata_s = subset_adata.copy(), subset_adata.copy()
        del (
            subset_adata_u.layers["su"],
            subset_adata_u.layers["sl"],
            subset_adata_s.layers["uu"],
            subset_adata_s.layers["ul"],
        )
        (
            subset_adata_u.layers["new"],
            subset_adata_u.layers["old"],
            subset_adata_s.layers["new"],
            subset_adata_s.layers["old"],
        ) = (
            subset_adata_u.layers.pop("ul"),
            subset_adata_u.layers.pop("uu"),
            subset_adata_s.layers.pop("sl"),
            subset_adata_s.layers.pop("su"),
        )
        Moment, Moment_ = MomData(subset_adata_s, tkey), MomData(subset_adata_u, tkey)
        if cur_grp == _group[0]:
            t_ind = 0
            g_len, t_len = len(_group), len(np.unique(adata.obs[tkey]))
            (
                adata.uns["M_sl"],
                adata.uns["V_sl"],
                adata.uns["M_ul"],
                adata.uns["V_ul"],
            ) = (
                np.zeros((Moment.M.shape[0], g_len * t_len)),
                np.zeros((Moment.M.shape[0], g_len * t_len)),
                np.zeros((Moment.M.shape[0], g_len * t_len)),
                np.zeros((Moment.M.shape[0], g_len * t_len)),
            )

        (
            adata.uns["M_sl"][:, (t_len * t_ind) : (t_len * (t_ind + 1))],
            adata.uns["V_sl"][:, (t_len * t_ind) : (t_len * (t_ind + 1))],
            adata.uns["M_ul"][:, (t_len * t_ind) : (t_len * (t_ind + 1))],
            adata.uns["V_ul"][:, (t_len * t_ind) : (t_len * (t_ind + 1))],
        ) = (Moment.M, Moment.V, Moment_.M, Moment_.V)

        del Moment_
        Est = Estimation(
            Moment, adata_u=subset_adata_u, time_key=tkey, normalize=True
        )  # # data is already normalized
    else:
        if log_unnormalized and "X_total" not in subset_adata.layers.keys():
            if issparse(subset_adata.layers["total"]):
                subset_adata.layers["new"].data, subset_adata.layers["total"].data = (
                    np.log(subset_adata.layers["new"].data + 1),
                    np.log(subset_adata.layers["total"].data + 1),
                )
            else:
                subset_adata.layers["total"], subset_adata.layers["total"] = (
                    np.log(subset_adata.layers["new"] + 1),
                    np.log(subset_adata.layers["total"] + 1),
                )

        Moment = MomData(subset_adata, tkey)
        if cur_grp == _group[0]:
            t_ind = 0
            g_len, t_len = len(_group), len(np.unique(adata.obs[tkey]))
            adata.uns["M"], adata.uns["V"] = (
                np.zeros((adata.shape[1], g_len * t_len)),
                np.zeros((adata.shape[1], g_len * t_len)),
            )

        (
            adata.uns["M"][:, (t_len * t_ind) : (t_len * (t_ind + 1))],
            adata.uns["V"][:, (t_len * t_ind) : (t_len * (t_ind + 1))],
        ) = (Moment.M, Moment.V)
        Est = Estimation(
            Moment, time_key=tkey, normalize=True
        )  # # data is already normalized

    return adata, Est, t_ind

