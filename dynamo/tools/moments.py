import numpy as np
import warnings
from anndata import AnnData
from scipy.sparse import issparse
from tqdm import tqdm
from .utils_moments import estimation
from .utils import get_mapper, gaussian_kernel, calc_1nd_moment, calc_2nd_moment
from .connectivity import mnn, normalize_knn_graph, umap_conn_indices_dist_embedding
from ..preprocessing.utils import get_layer_keys, allowed_X_layer_names


def moments(adata, use_gaussian_kernel=True, use_mnn=False, layers="all"):
    # if we have uu, ul, su, sl, let us set total and new

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
        layer_x_group = np.where([layer_x in x for x in
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

            layer_y_group = np.where([layer_y in x for x in
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

    return adata


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


def calc_1nd_moment(X, W, normalize_W=True):
    if normalize_W:
        d = np.sum(W, 1)
        W = np.diag(1 / d) @ W
    return W @ X


def calc_2nd_moment(X, Y, W, normalize_W=True, center=False, mX=None, mY=None):
    if normalize_W:
        d = np.sum(W, 1)
        W = np.diag(1 / d) @ W
    XY = np.multiply(W @ Y, X)
    if center:
        mX = calc_1nd_moment(X, W, False) if mX is None else mX
        mY = calc_1nd_moment(Y, W, False) if mY is None else mY
        XY = XY - np.multiply(mX, mY)
    return XY


class MomData(AnnData):
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
