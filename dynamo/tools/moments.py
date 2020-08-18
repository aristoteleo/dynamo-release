import numpy as np
import warnings
from anndata import AnnData
from scipy.sparse import issparse, csr_matrix, lil_matrix, diags
from tqdm import tqdm
from .utils import get_mapper, elem_prod, inverse_norm
from .connectivity import mnn, normalize_knn_graph, umap_conn_indices_dist_embedding
from ..preprocessing.utils import get_layer_keys, allowed_X_layer_names, pca

# ---------------------------------------------------------------------------------------------------
# use for calculating moments for stochastic model:
def moments(adata,
            genes=None,
            group=None,
            use_gaussian_kernel=False,
            normalize=True,
            use_mnn=False,
            layers="all",
            n_pca_components=30,
            n_neighbors=30,
            ):
    """Calculate kNN based first and second moments (including uncentered covariance) for
     different layers of data.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object.
        genes: `np.array` (default: `None`)
            The one-dimensional numpy array of the genes that you want to perform pca analysis (if adata.obsm['X'] is not
            available). `X` keyname (instead of `X_pca`) was used to enable you use a different set of genes for flexible
            connectivity graph construction. If `None`, by default it will select genes based `use_for_pca` key in .var
            attributes if it exists otherwise it will also all genes stored in adata.X
        group: `str` or None (default: `None`)
            The column key/name that identifies the grouping information (for example, clusters that correspond to
            different cell types or different time points) of cells. This will be used to compute kNN graph for each
            group (i.e cell-type/time-point). This is important, for example, we don't want cells from different labeling
            time points to be mixed when performing the kNN graph for calculating the moments.
        use_gaussian_kernel: `bool` (default: `True`)
            Whether to normalize the kNN graph via a Guasian kernel.
        normalize: `bool` (default: `True`)
            Whether to normalize the connectivity matrix so that each row sums up to 1. When `use_gaussian_kernel` is False,
            this will be reset to be False because we will already normalize the connectivity matrix matrix by dividing
            each row the total number of connections.
        use_mnn: `bool` (default: `False`)
            Whether to use mutual kNN across different layers as for the moment calculation.
        layers: `str` or a list of str (default: `str`)
            The layers that will be used for calculating the moments.
        n_pca_components: `int` (default: `30`)
            The number of pca components to use for constructing nearest neighbor graph and calculating 1/2-st moments.
        n_neighbors: `int` (default: `30`)
            The number of neighbors for constructing nearest neighbor graph used to calculate 1/2-st moments.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            A updated AnnData object with calculated first/second moments (including uncentered covariance) included.
    """
    mapper = get_mapper()
    only_splicing, only_labeling, splicing_and_labeling = allowed_X_layer_names()

    if genes is None and 'use_for_pca' in adata.var.keys(): genes = adata.var_names[adata.var.use_for_pca]
    if use_mnn:
        if "mnn" not in adata.uns.keys():
            adata = mnn(
                adata,
                n_pca_components=n_pca_components,
                layers="all",
                use_pca_fit=True,
                save_all_to_adata=False,
            )
        conn = adata.uns["mnn"]
    else:
        if 'X' not in adata.obsm.keys():
            if not any([i.startswith('X_') for i in adata.layers.keys()]):
                from ..preprocessing.preprocess import recipe_monocle
                genes_to_use = adata.var_names[genes] if genes.dtype == 'bool' else genes
                adata = recipe_monocle(adata, genes_to_use=genes_to_use, n_pca_components=n_pca_components)
                adata.obsm["X"] = adata.obsm["X_pca"]
            else:
                CM = adata.X if genes is None else adata[:, genes].X
                cm_genesums = CM.sum(axis=0)
                valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
                valid_ind = np.array(valid_ind).flatten()
                CM = CM[:, valid_ind]
                adata, fit, _ = pca(adata, CM, n_pca_components=n_pca_components)

                adata.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_[1:]

        X = adata.obsm["X"][:, :n_pca_components]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if group is None:
                kNN, knn_indices, knn_dists, _ = umap_conn_indices_dist_embedding(
                    X, n_neighbors=np.min((n_neighbors, adata.n_obs - 1)), return_mapper=False
                )

                if use_gaussian_kernel and not use_mnn:
                    conn = gaussian_kernel(X, knn_indices, sigma=10, k=None, dists=knn_dists)
                else:
                    conn = normalize_knn_graph(kNN > 0)
                    normalize = False
            else:
                if group not in adata.obs.keys():
                    raise Exception(f'the group {group} provided is not a column name in .obs attribute.')
                conn = csr_matrix((adata.n_obs, adata.n_obs))
                cells_group = adata.obs[group]
                uniq_grp = np.unique(cells_group)
                for cur_grp in uniq_grp:
                    cur_cells = cells_group == cur_grp
                    cur_X = X[cur_cells, :]
                    cur_kNN, cur_knn_indices, cur_knn_dists, _ = umap_conn_indices_dist_embedding(
                        cur_X, n_neighbors=np.min((n_neighbors, sum(cur_cells) - 1)), return_mapper=False
                    )

                    if use_gaussian_kernel and not use_mnn:
                        cur_conn = gaussian_kernel(cur_X, cur_knn_indices, sigma=10, k=None, dists=cur_knn_dists)
                    else:
                        cur_conn = normalize_knn_graph(cur_kNN > 0)

                    cur_cells_ = np.where(cur_cells)[0]
                    conn[cur_cells_[:, None], cur_cells_] = cur_conn

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
        layer_x = inverse_norm(adata, layer_x)

        if mapper[layer] not in adata.layers.keys():
            adata.layers[mapper[layer]], conn = (
                calc_1nd_moment(layer_x, conn, normalize_W=normalize)
                if use_gaussian_kernel
                else (conn.dot(layer_x), conn)
            )
        for layer2 in layers[i:]:
            layer_y = adata.layers[layer2].copy()

            layer_y_group = np.where([layer2 in x for x in
                                      [only_splicing, only_labeling, splicing_and_labeling]])[0][0]
            # don't calculate 2 moments among uu, ul, su, sl -
            # they should be time-dependent moments and
            # those calculations are model specific
            if (layer_x_group != layer_y_group) or layer_x_group == 2:
                continue
            layer_y = inverse_norm(adata, layer_y)

            if mapper[layer2] not in adata.layers.keys():
                adata.layers[mapper[layer2]], conn = (
                    calc_1nd_moment(layer_y, conn, normalize_W=normalize)
                    if use_gaussian_kernel
                    else (conn.dot(layer_y), conn)
                )

            adata.layers["M_" + layer[2] + layer2[2]] = calc_2nd_moment(
                layer_x, layer_y, conn, normalize_W=normalize, mX=None, mY=None
            )

    if (
            "X_protein" in adata.obsm.keys()
    ):  # may need to update with mnn or just use knn from protein layer itself.
        adata.obsm[mapper["X_protein"]] = conn.dot(adata.obsm["X_protein"])
    adata.obsp['moments_con'] = conn

    return adata

def time_moment(adata,
    tkey,
    has_splicing,
    has_labeling=True,
    t_label_keys=None,
):
    """Calculate time based first and second moments (including uncentered covariance) for
     different layers of data.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object.
        tkey: `str` or None (default: None)
            The column key for the time label of cells in .obs. Used for either "ss" or "kinetic" model.
            mode  with labeled data.
        has_splicing: `bool`
            Whether the data has splicing information.
        has_labeling: `bool` (default: True)
            Whether the data has labeling information.
        t_label_keys: `str`, `list` or None (default: None)
            The column key(s) for the labeling time label of cells in .obs. Used for either "ss" or "kinetic" model.
            Not used for now and `tkey` is implicitly assumed as `t_label_key` (however, `tkey` should just be the time
            of the experiment).

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            A updated AnnData object with calculated first/second moments (including uncentered covariance) for
             each time point for each layer included.
    """

    if has_labeling:
        if has_splicing:
            layers = ['uu', 'ul', 'su', 'sl']
        else:
            layers = ['new', 'total']
    else:
        layers = ['unspliced', 'spliced']

    time = adata.obs[tkey]
    m, v = prepare_data_deterministic(adata, adata.var.index, time, layers,
                                          use_total_layers=True, log=False)
    adata.uns['time_moments'] = {'time': time}
    adata.varm['m_t'] = m
    adata.varm['v_t'] = v

    return adata

# ---------------------------------------------------------------------------------------------------
# use for kinetic assumption
def get_layer_pair(layer):
    pair = {'new': "total", 'total': "new",
            'X_new': "X_total", "X_total": 'X_new',
            'M_t': 'M_n', "M_n": 'M_t'}
    return pair[layer] if layer in pair.keys() else None


def get_layer_group(layer):
    group = {'uu': "ul", 'ul': "uu", 'su': "sl", "sl": 'su',
            'X_uu': "X_ul", 'X_ul': "X_uu", 'X_su': "X_sl", "X_sl": 'X_su',
            'M_uu': "M_ul", 'M_ul': "M_uu", 'M_su': "M_sl", "M_sl": 'M_su',
            }
    return group[layer] if layer in group.keys() else None



def prepare_data_deterministic(adata, genes, time, layers,
                               use_total_layers=True,
                               total_layers=['X_ul', 'X_sl', 'X_uu', 'X_su'],
                               log=False):
    from ..preprocessing.utils import sz_util, normalize_util
    if use_total_layers:
        if 'total_Size_Factor' not in adata.obs.keys():
            # total_layers = ["uu", "ul", "su", "sl"] if 'uu' in adata.layers.keys() else ['total']
            sfs, _ = sz_util(adata, '_total_', round_exprs=False, method="median",
                                       locfunc=np.nanmean, total_layers=total_layers)
        else:
            sfs = adata.obs.total_Size_Factor
        sfs_x, sfs_y = sfs[:, None], sfs[:, None]

    m = [None] * len(layers)
    v = [None] * len(layers)
    raw = [None] * len(layers)
    for i, layer in enumerate(layers):
        if layer in ['X_total', 'total', 'M_t']:
            if (layer == 'X_total' and adata.uns['pp_norm_method'] is None) or layer == 'M_t':
                x_layer = adata[:, genes].layers[layer]
                x_layer = x_layer - adata[:, genes].layers[get_layer_pair(layer)]
            else:
                x_layer = adata.layers[layer]
                group_pair_x_layer_ = get_layer_group(get_layer_pair(layer))
                pair_x_layer, group_x_layer, group_pair_x_layer = adata.layers[get_layer_pair(layer)], \
                                                                  adata.layers[get_layer_group(layer)], \
                                                                  None if group_pair_x_layer_ is None else \
                                                                      adata.layers[group_pair_x_layer_]
                if layer.startswith('X_'):
                    x_layer, pair_x_layer, group_x_layer, group_pair_x_layer = inverse_norm(adata, x_layer), \
                                                                               inverse_norm(adata, pair_x_layer), \
                                                                               inverse_norm(adata, group_x_layer), \
                                                                               0 if group_pair_x_layer_ is None else \
                                                                                   inverse_norm(adata, group_pair_x_layer)

                if not use_total_layers:
                    sfs_x, _ = sz_util(adata, layer, round_exprs=False, method="median",
                                       locfunc=np.nanmean, total_layers=None, CM=x_layer+group_x_layer)
                    sfs_y, _ = sz_util(adata, get_layer_pair(layer), round_exprs=False,
                                       method="median", locfunc=np.nanmean, total_layers=None,
                                       CM=pair_x_layer + group_pair_x_layer)
                    sfs_x, sfs_y = sfs_x[:, None], sfs_y[:, None]

                x_layer = normalize_util(x_layer[:, adata.var_names.isin(genes)], sfs_x, relative_expr=True,
                                         pseudo_expr=0, norm_method=None)
                y_layer = normalize_util(pair_x_layer[:, adata.var_names.isin(genes)], sfs_y, relative_expr=True,
                                         pseudo_expr=0, norm_method=None)

                x_layer = x_layer - y_layer
        else:
            if (layer == ['X_new'] and adata.uns['pp_norm_method'] is None) or layer == 'M_n':
                x_layer = adata[:, genes].layers[layer]
            else:
                x_layer = adata.layers[layer]
                if layer.startswith('X_'):
                    x_layer = inverse_norm(adata, x_layer)

                if not use_total_layers:
                    sfs, _ = sz_util(adata, layer, round_exprs=False, method="median",
                                       locfunc=np.nanmean, total_layers=None, CM=x_layer)
                x_layer = normalize_util(x_layer[:, adata.var_names.isin(genes)], szfactors=sfs[:, None],
                                         relative_expr=True, pseudo_expr=0, norm_method=None)

        if log:
            if issparse(x_layer):
                x_layer.data = np.log1p(x_layer.data)
            else:
                x_layer = np.log1p(x_layer)

        m[i], v[i], _ = calc_12_mom_labeling(x_layer.T, time)
        raw[i] = x_layer

    return m, v, raw # each list element corresponds to a layer


def prepare_data_has_splicing(adata, genes, time, layer_u, layer_s,
                              use_total_layers=True,
                              total_layers=['X_ul', 'X_sl', 'X_uu', 'X_su'],
                              return_cov=True):
    """Prepare data when assumption is kinetic and data has splicing"""
    from ..preprocessing.utils import sz_util, normalize_util
    res = [0] * len(genes)
    raw = [0] * len(genes)

    U, S = adata[:, genes].layers[layer_u] if layer_u == 'M_ul' else None, \
           adata[:, genes].layers[layer_s] if layer_s == 'M_sl' else None

    layer_ul_data, layer_sl_data = adata.layers[layer_u], adata.layers[layer_s]
    layer_uu_data, layer_su_data = adata.layers[total_layers[2]], adata.layers[total_layers[3]]
    layer_ul_data, layer_sl_data = layer_ul_data if layer_u == 'M_ul' else inverse_norm(adata, layer_ul_data), \
                                   layer_sl_data if layer_s == 'M_sl' else inverse_norm(adata, layer_sl_data)
    layer_uu_data, layer_su_data = layer_uu_data if total_layers[2] == 'M_uu' else inverse_norm(adata, layer_uu_data), \
                                   layer_su_data if total_layers[3] == 'M_su' else inverse_norm(adata, layer_su_data)

    if use_total_layers:
        if 'total_Size_Factor' not in adata.obs.keys():
            sfs, _ = sz_util(adata, '_total_', round_exprs=False, method="median", locfunc=np.nanmean,
                             total_layers=total_layers, CM=layer_ul_data + layer_sl_data + layer_uu_data + layer_su_data)
            sfs_u, sfs_s = sfs[:, None], sfs[:, None]
        else:
            sfs = adata.obs.total_Size_Factor
            sfs_u, sfs_s = sfs[:, None], sfs[:, None]
    else:
        sfs_u, _ = sz_util(adata, layer_u, round_exprs=False, method="median",
                                       locfunc=np.nanmean, total_layers=None, CM=layer_ul_data + layer_uu_data)
        sfs_s, _ = sz_util(adata, layer_s, round_exprs=False, method="median",
                                       locfunc=np.nanmean, total_layers=None, CM=layer_sl_data + layer_su_data)
        sfs_u, sfs_s = sfs_u[:, None], sfs_s[:, None]

    if U is None: U = normalize_util(layer_ul_data[:, adata.var_names.isin(genes)], sfs_u, relative_expr=True,
                                     pseudo_expr=0, norm_method=None)
    if S is None: S = normalize_util(layer_sl_data[:, adata.var_names.isin(genes)], sfs_s, relative_expr=True,
                                     pseudo_expr=0, norm_method=None)

    for i, g in enumerate(genes):
        u = U[:, i]
        s = S[:, i]
        ut = strat_mom(u, time, np.mean)
        st = strat_mom(s, time, np.mean)
        uut = strat_mom(elem_prod(u, u), time, np.mean)
        ust = strat_mom(elem_prod(u, s), time, np.mean)
        sst = strat_mom(elem_prod(s, s), time, np.mean)
        x = np.vstack([ut, st, uut, sst, ust]) if return_cov else np.vstack([ut, st, uut, sst])

        res[i] = x
        raw[i] = np.vstack((u, s))

    return res, raw


def prepare_data_no_splicing(adata, genes, time, layer,
                             use_total_layers=True,
                             total_layer='X_total',
                             return_old=False):
    """Prepare data when assumption is kinetic and data has no splicing"""
    from ..preprocessing.utils import sz_util, normalize_util
    res = [0] * len(genes)
    raw = [0] * len(genes)

    U, T = adata[:, genes].layers[layer] if layer == 'M_n' else None, \
           adata[:, genes].layers[total_layer] if total_layer == 'M_t' else None

    layer_data = adata.layers[layer]
    total_layer_data = adata.layers[total_layer]

    layer_data, total_layer_data = layer_data if layer == 'M_n' else inverse_norm(adata, layer_data), \
                                   total_layer_data if total_layer == 'M_t' else inverse_norm(adata, total_layer_data)

    if use_total_layers:
        if 'total_Size_Factor' not in adata.obs.keys():
            sfs, _ = sz_util(adata, '_total_', round_exprs=False, method="median",
                                       locfunc=np.nanmean, total_layers=total_layer, CM=total_layer_data)
        else:
            sfs = adata.obs.total_Size_Factor
        sfs, tot_sfs = sfs[:, None], sfs[:, None]
    else:
        sfs, _ = sz_util(adata, layer, round_exprs=False, method="median",
                                       locfunc=np.nanmean, total_layers=None, CM=layer_data)
        tot_sfs, _ = sz_util(adata, layer, round_exprs=False, method="median",
                                       locfunc=np.nanmean, total_layers=None, CM=total_layer_data)
        sfs, tot_sfs = sfs[:, None], tot_sfs[:, None]

    if U is None: U = normalize_util(layer_data[:, adata.var_names.isin(genes)], sfs, relative_expr=True, pseudo_expr=0,
                       norm_method=None)
    if T is None: T = normalize_util(total_layer_data[:, adata.var_names.isin(genes)], tot_sfs, relative_expr=True,
                                     pseudo_expr=0, norm_method=None)

    for i, g in enumerate(genes):
        u, t = U[:, i], T[:, i]
        ut = strat_mom(u, time, np.mean)
        uut = strat_mom(elem_prod(u, u), time, np.mean)
        res[i] = np.vstack([ut, uut])
        raw[i] = np.vstack([u, t - u]) if return_old else u

    return res, raw


def prepare_data_mix_has_splicing(adata, genes, time, layer_u='X_uu', layer_s='X_su',
                                  layer_ul='X_ul', layer_sl='X_sl', use_total_layers=True,
                                  total_layers=['X_ul', 'X_sl', 'X_uu', 'X_su'], mix_model_indices=None):
    """Prepare data for mixture modeling when assumption is kinetic and data has splicing.
    Note that the mix_model_indices is indexed on 10 total species, which can be used to specify
    the data required for different mixture models.
    """
    from ..preprocessing.utils import sz_util, normalize_util
    res = [0] * len(genes)
    raw = [0] * len(genes)

    U, S = adata[:, genes].layers[layer_u] if layer_u == 'M_uu' else None, \
           adata[:, genes].layers[layer_s] if layer_u == 'M_su' else None
    Ul, Sl = adata[:, genes].layers[layer_ul] if layer_u == 'M_ul' else None, \
           adata[:, genes].layers[layer_sl] if layer_u == 'M_sl' else None

    layer_u_data, layer_s_data = adata.layers[layer_u], adata.layers[layer_s]
    layer_ul_data, layer_sl_data = adata.layers[layer_ul], adata.layers[layer_sl]
    layer_u_data, layer_s_data = layer_u_data if layer_u == 'M_uu' else inverse_norm(adata, layer_u_data), \
                                 layer_s_data if layer_s == 'M_su' else inverse_norm(adata, layer_s_data)
    layer_ul_data, layer_sl_data = layer_ul_data if layer_ul == 'M_ul' else inverse_norm(adata, layer_ul_data), \
                                   layer_sl_data if layer_sl == 'M_sl' else inverse_norm(adata, layer_sl_data)

    if use_total_layers:
        if 'total_Size_Factor' not in adata.obs.keys():
            sfs, _ = sz_util(adata, '_total_', False, "median", np.nanmean,
                             total_layers=total_layers, CM=layer_u_data + layer_s_data + layer_ul_data + layer_sl_data)
            sfs_u, sfs_s = sfs[:, None], sfs[:, None]
        else:
            sfs = adata.obs.total_Size_Factor
            sfs_u, sfs_s = sfs[:, None], sfs[:, None]
    else:
        sfs_u, _ = sz_util(adata, layer_u, False, "median", np.nanmean, total_layers=None,
                           CM=layer_u_data + layer_ul_data)
        sfs_s, _ = sz_util(adata, layer_s, False, "median", np.nanmean, total_layers=None,
                           CM=layer_s_data + layer_sl_data)
        sfs_u, sfs_s = sfs_u[:, None], sfs_s[:, None]

    if U is None: U = normalize_util(layer_u_data[:, adata.var_names.isin(genes)], sfs_u, relative_expr=True,
                                     pseudo_expr=0, norm_method=None)
    if S is None: S = normalize_util(layer_s_data[:, adata.var_names.isin(genes)], sfs_s, relative_expr=True,
                                     pseudo_expr=0, norm_method=None)
    if Ul is None: Ul = normalize_util(layer_ul_data[:, adata.var_names.isin(genes)], sfs_u, relative_expr=True,
                                       pseudo_expr=0, norm_method=None)
    if Sl is None: Sl = normalize_util(layer_sl_data[:, adata.var_names.isin(genes)], sfs_s, relative_expr=True,
                                       pseudo_expr=0, norm_method=None)

    for i, g in enumerate(genes):
        ul = Ul[:, i]
        sl = Sl[:, i]
        ult = strat_mom(ul, time, np.mean)
        slt = strat_mom(sl, time, np.mean)
        ul_ult = strat_mom(elem_prod(ul, ul), time, np.mean)
        ul_slt = strat_mom(elem_prod(ul, sl), time, np.mean)
        sl_slt = strat_mom(elem_prod(sl, sl), time, np.mean)

        u = U[:, i]
        s = S[:, i]
        ut = strat_mom(u, time, np.mean)
        st = strat_mom(s, time, np.mean)
        uut = strat_mom(elem_prod(u, u), time, np.mean)
        ust = strat_mom(elem_prod(u, s), time, np.mean)
        sst = strat_mom(elem_prod(s, s), time, np.mean)

        x = np.vstack([ult, slt, ul_ult, sl_slt, ul_slt, ut, st, uut, sst, ust])
        if mix_model_indices is not None:
            x = x[mix_model_indices]

        res[i] = x
        raw[i] = np.vstack((ul, sl, u, s))

    return res, raw


def prepare_data_mix_no_splicing(adata, genes, time, layer_n, layer_t, use_total_layers=True,
                                 total_layer='X_total', mix_model_indices=None):
    """Prepare data for mixture modeling when assumption is kinetic and data has NO splicing.
    Note that the mix_model_indices is indexed on 4 total species, which can be used to specify
    the data required for different mixture models.
    """
    from ..preprocessing.utils import sz_util, normalize_util
    res = [0] * len(genes)
    raw = [0] * len(genes)

    N, T = adata[:, genes].layers[layer_n] if layer_n == 'M_n' else None, \
           adata[:, genes].layers[layer_t] if layer_t == 'M_t' else None

    layer_n_data = adata.layers[layer_n]
    layer_t_data = adata.layers[layer_t]
    layer_n_data, layer_t_data = layer_n_data if layer_n == 'M_n' else inverse_norm(adata, layer_n_data), \
                                 layer_t_data if layer_t == 'M_t' else inverse_norm(adata, layer_t_data)

    if use_total_layers:
        if 'total_Size_Factor' not in adata.obs.keys():
            sfs, _ = sz_util(adata, total_layer, False, "median", np.nanmean, total_layers='total', CM=layer_t_data)
            sfs_n, sfs_t = sfs[:, None], sfs[:, None]
        else:
            sfs = adata.obs.total_Size_Factor
            sfs_n, sfs_t = sfs[:, None], sfs[:, None]
    else:
        sfs_n, _ = sz_util(adata, layer_n, False, "median", np.nanmean, total_layers=None, CM=layer_n_data)
        sfs_t, _ = sz_util(adata, layer_t, False, "median", np.nanmean, total_layers=None, CM=layer_t_data)
        sfs_n, sfs_t = sfs_n[:, None], sfs_t[:, None]

    if N is None: N = normalize_util(layer_n_data[:, adata.var_names.isin(genes)], sfs_n, relative_expr=True,
                                     pseudo_expr=0, norm_method=None)
    if T is None: T = normalize_util(layer_t_data[:, adata.var_names.isin(genes)], sfs_t, relative_expr=True,
                                     pseudo_expr=0, norm_method=None)

    for i, g in enumerate(genes):
        n = N[:, i]
        nt = strat_mom(n, time, np.mean)
        nnt = strat_mom(elem_prod(n, n), time, np.mean)
        o = T[:, i] - n
        ot = strat_mom(o, time, np.mean)
        oot = strat_mom(elem_prod(o, o), time, np.mean)

        x = np.vstack([nt, nnt, ot, oot])
        if mix_model_indices is not None:
            x = x[mix_model_indices]

        res[i] = x
        raw[i] = np.vstack((n, o))

    return res, raw

# ---------------------------------------------------------------------------------------------------
# moment related:

def stratify(arr, strata):
    s = np.unique(strata)
    return [arr[strata == s[i]] for i in range(len(s))]


def strat_mom(arr, strata, fcn_mom):
    arr = arr.A if issparse(arr) else arr
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
        L = np.array(adata[:, g].layers["X_new"], dtype=float)
        U = np.array(adata[:, g].layers["X_total"], dtype=float) - L
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
        from ..estimation.tsc.utils_moments import estimation
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

