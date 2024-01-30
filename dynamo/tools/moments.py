import warnings
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix, diags, issparse, lil_matrix
from tqdm import tqdm

from ..configuration import DKM, DynamoAdataKeyManager
from ..dynamo_logger import LoggerManager
from ..preprocessing.normalization import normalize_mat_monocle, sz_util
from ..preprocessing.pca import pca
from ..utils import copy_adata
from .connectivity import mnn, normalize_knn_graph, umap_conn_indices_dist_embedding
from .utils import elem_prod, get_mapper, inverse_norm


# ---------------------------------------------------------------------------------------------------
# use for calculating moments for stochastic model:
def moments(
    adata: AnnData,
    X_data: Optional[np.ndarray] = None,
    genes: Optional[list] = None,
    group: Optional[str] = None,
    conn: Optional[csr_matrix] = None,
    use_gaussian_kernel: bool = False,
    normalize: bool = True,
    use_mnn: bool = False,
    layers: Union[List[str], str] = "all",
    n_pca_components: int = 30,
    n_neighbors: int = 30,
    copy: bool = False,
) -> Optional[AnnData]:
    """Calculate kNN based first and second moments (including uncentered covariance) for different layers of data.

    Args:
        adata: An AnnData object.
        X_data: The user supplied data that will be used for constructing the nearest neighbor graph directly. Defaults
            to None.
        genes: The one-dimensional numpy array of the genes that you want to perform pca analysis (if adata.obsm['X'] is
             not available). `X` keyname (instead of `X_pca`) was used to enable you use a different set of genes for
             flexible connectivity graph construction. If `None`, by default it will select genes based `use_for_pca`
             key in .var attributes if it exists otherwise it will also all genes stored in adata.X. Defaults to None.
        group: The column key/name that identifies the grouping information (for example, clusters that correspond to
            different cell types or different time points) of cells. This will be used to compute kNN graph for each
            group (i.e. cell-type/time-point). This is important, for example, we don't want cells from different
            labeling time points to be mixed when performing the kNN graph for calculating the moments. Defaults to
            None.
        conn: The connectivity graph that will be used for moment calculations. Defaults to None.
        use_gaussian_kernel: Whether to normalize the kNN graph via a Gaussian kernel. Defaults to False.
        normalize: Whether to normalize the connectivity matrix so that each row sums up to 1. When
            `use_gaussian_kernel` is False, this will be reset to be False because we will already normalize the
            connectivity matrix by dividing each row the total number of connections. Defaults to True.
        use_mnn: Whether to use mutual kNN across different layers as for the moment calculation. Defaults to False.
        layers: The layers that will be used for calculating the moments. Defaults to "all".
        n_pca_components: The number of pca components to use for constructing nearest neighbor graph and calculating
            1/2-st moments. Defaults to 30.
        n_neighbors: The number of pca components to use for constructing nearest neighbor graph and calculating 1/2-st
            moments. Defaults to 30.
        copy: Whether to return a new updated AnnData object or update inplace. Defaults to False.

    Raises:
        Exception: `group` is invalid.
        ValueError: `conn` is invalid. It should be a square array with dimension equal to the cell number.

    Returns:
        The updated AnnData object if `copy` is true. Otherwise, the AnnData object passed in would be updated inplace
        and None would be returned.
    """

    logger = LoggerManager.gen_logger("dynamo-moments")
    logger.info("calculating first/second moments...", indent_level=1)
    logger.log_time()

    adata = copy_adata(adata) if copy else adata

    mapper = get_mapper()

    (
        only_splicing,
        only_labeling,
        splicing_and_labeling,
    ) = DKM.allowed_X_layer_names()

    if conn is None:
        if genes is None and "use_for_pca" in adata.var.keys():
            genes = adata.var_names[adata.var.use_for_pca]
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
            if X_data is not None:
                X = X_data
            else:
                if DKM.X_PCA not in adata.obsm.keys():
                    if not any([i.startswith("X_") for i in adata.layers.keys()]):
                        from ..preprocessing import Preprocessor

                        genes_to_use = adata.var_names[genes] if genes.dtype == "bool" else genes
                        preprocessor = Preprocessor(force_gene_list=genes_to_use)
                        preprocessor.preprocess_adata(adata, recipe="monocle")
                    else:
                        CM = adata.X if genes is None else adata[:, genes].X
                        cm_genesums = CM.sum(axis=0)
                        valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
                        valid_ind = np.array(valid_ind).flatten()
                        CM = CM[:, valid_ind]
                        adata, fit, _ = pca(
                            adata,
                            CM,
                            n_pca_components=n_pca_components,
                            return_all=True,
                        )

                        adata.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_[1:]

                X = adata.obsm[DKM.X_PCA][:, :n_pca_components]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if group is None:
                    (kNN, knn_indices, knn_dists, _,) = umap_conn_indices_dist_embedding(
                        X,
                        n_neighbors=np.min((n_neighbors, adata.n_obs - 1)),
                        return_mapper=False,
                    )

                    if use_gaussian_kernel and not use_mnn:
                        conn = gaussian_kernel(X, knn_indices, sigma=10, k=None, dists=knn_dists)
                    else:
                        conn = normalize_knn_graph(kNN > 0)
                        normalize = False
                else:
                    if group not in adata.obs.keys():
                        raise Exception(f"the group {group} provided is not a column name in .obs attribute.")
                    conn = csr_matrix((adata.n_obs, adata.n_obs))
                    cells_group = adata.obs[group]
                    uniq_grp = np.unique(cells_group)
                    for cur_grp in uniq_grp:
                        cur_cells = cells_group == cur_grp
                        cur_X = X[cur_cells, :]
                        (cur_kNN, cur_knn_indices, cur_knn_dists, _,) = umap_conn_indices_dist_embedding(
                            cur_X,
                            n_neighbors=np.min((n_neighbors, sum(cur_cells) - 1)),
                            return_mapper=False,
                        )

                        if use_gaussian_kernel and not use_mnn:
                            cur_conn = gaussian_kernel(
                                cur_X,
                                cur_knn_indices,
                                sigma=10,
                                k=None,
                                dists=cur_knn_dists,
                            )
                        else:
                            cur_conn = normalize_knn_graph(cur_kNN > 0)

                        cur_cells_ = np.where(cur_cells)[0]
                        conn[cur_cells_[:, None], cur_cells_] = cur_conn
    else:
        if conn.shape[0] != conn.shape[1] or conn.shape[0] != adata.n_obs:
            raise ValueError(
                "The connectivity data `conn` you provided should a square array with dimension equal to "
                "the cell number!"
            )

    layers = DynamoAdataKeyManager.get_available_layer_keys(adata, layers, False, False)
    layers = [
        layer
        for layer in layers
        if layer.startswith("X_") and (not layer.endswith("matrix") and not layer.endswith("ambiguous"))
    ]
    layers.sort(reverse=True)  # ensure we get M_us, M_tn, etc (instead of M_su or M_nt).
    for i, layer in enumerate(layers):
        layer_x = adata.layers[layer].copy()
        matched_x_group_indices = np.where([layer in x for x in [only_splicing, only_labeling, splicing_and_labeling]])
        if len(matched_x_group_indices[0]) == 0:
            logger.warning(
                f"layer {layer} is not in any of the {only_splicing, only_labeling, splicing_and_labeling} groups, skipping..."
            )
            continue
        layer_x_group = matched_x_group_indices[0][0]

        if mapper[layer] not in adata.layers.keys():
            adata.layers[mapper[layer]], conn = (
                calc_1nd_moment(layer_x, conn, normalize_W=normalize)
                if use_gaussian_kernel
                else (conn.dot(layer_x), conn)
            )
        for layer2 in layers[i:]:
            matched_y_group_indices = np.where(
                [layer2 in x for x in [only_splicing, only_labeling, splicing_and_labeling]]
            )
            if len(matched_y_group_indices[0]) == 0:
                logger.warning(
                    f"layer {layer2} is not in any of the {only_splicing, only_labeling, splicing_and_labeling} groups, skipping..."
                )
                continue
            layer_y = adata.layers[layer2].copy()

            layer_y_group = matched_y_group_indices[0][0]
            # don't calculate 2 moments among uu, ul, su, sl -
            # they should be time-dependent moments and
            # those calculations are model specific
            if (layer_x_group != layer_y_group) or layer_x_group == 2:
                continue

            if mapper[layer2] not in adata.layers.keys():
                adata.layers[mapper[layer2]], conn = (
                    calc_1nd_moment(layer_y, conn, normalize_W=normalize)
                    if use_gaussian_kernel
                    else (conn.dot(layer_y), conn)
                )

            adata.layers["M_" + layer[2] + layer2[2]] = calc_2nd_moment(
                layer_x, layer_y, conn, normalize_W=normalize, mX=None, mY=None
            )

    if "X_protein" in adata.obsm.keys():  # may need to update with mnn or just use knn from protein layer itself.
        adata.obsm[mapper["X_protein"]] = conn.dot(adata.obsm["X_protein"])
    adata.obsp["moments_con"] = conn

    logger.finish_progress("moments calculation")

    if copy:
        return adata
    return None


def time_moment(
    adata: AnnData,
    tkey: Optional[str],
    has_splicing: bool,
    has_labeling: bool = True,
    t_label_keys: Union[List[str], str, None] = None,
) -> AnnData:
    """Calculate time based first and second moments (including uncentered covariance) for different layers of data.

    Args:
        adata: An AnnData object.
        tkey: The column key for the time label of cells in .obs. Used for either "ss" or "kinetic" model.
        has_splicing: Whether the data has splicing information.
        has_labeling: Whether the data has labeling information. Defaults to True.
        t_label_keys: (not used for now) The column key(s) for the labeling time label of cells in .obs. Used for either
            "ss" or "kinetic" model. `tkey` is implicitly assumed as `t_label_key` (however, `tkey` should just be the
            time of the experiment). Defaults to None.

    Returns:
        An updated AnnData object with calculated first/second moments (including uncentered covariance) for each time
        point for each layer included.
    """

    if has_labeling:
        if has_splicing:
            layers = ["uu", "ul", "su", "sl"]
        else:
            layers = ["new", "total"]
    else:
        layers = ["unspliced", "spliced"]

    time = adata.obs[tkey]
    m, v = prepare_data_deterministic(adata, adata.var.index, time, layers, use_total_layers=True, log=False)
    adata.uns["time_moments"] = {"time": time}
    adata.varm["m_t"] = m
    adata.varm["v_t"] = v

    return adata


# ---------------------------------------------------------------------------------------------------
# use for kinetic assumption
def get_layer_pair(layer: str) -> Optional[str]:
    """Get the layer in pair for the input layer.

    Args:
        layer: The key for the input layer.

    Returns:
        The key for corresponding layer in pair.
    """
    pair = {
        "new": "total",
        "total": "new",
        "X_new": "X_total",
        "X_total": "X_new",
        "M_t": "M_n",
        "M_n": "M_t",
    }
    return pair[layer] if layer in pair.keys() else None


def get_layer_group(layer: str) -> Optional[str]:
    """Get the layer group in pair for the input layer group.

    Args:
        layer: The key for the input layer group.

    Returns:
        The key for corresponding layer group in pair.
    """
    group = {
        "uu": "ul",
        "ul": "uu",
        "su": "sl",
        "sl": "su",
        "X_uu": "X_ul",
        "X_ul": "X_uu",
        "X_su": "X_sl",
        "X_sl": "X_su",
        "M_uu": "M_ul",
        "M_ul": "M_uu",
        "M_su": "M_sl",
        "M_sl": "M_su",
    }
    return group[layer] if layer in group.keys() else None


def prepare_data_deterministic(
    adata: AnnData,
    genes: List[str],
    time: np.ndarray,
    layers: List[str],
    use_total_layers: bool = True,
    total_layers: List[str] = ["X_ul", "X_sl", "X_uu", "X_su"],
    log: bool = False,
    return_ntr: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Union[np.ndarray, csr_matrix]]]:
    """Prepare the data for kinetic calculation based on deterministic model.

    Args:
        adata: An AnnData object.
        genes: The genes to be estimated.
        time: The array containing time stamp.
        layers: The layer keys in adata object to be processed.
        use_total_layers: Whether to use total layers embedded in the AnnData object. Defaults to True.
        total_layers: The layer(s) that can be summed up to get the total mRNA. Defaults to ["X_ul", "X_sl", "X_uu",
            "X_su"].
        log: Whether to perform log1p (i.e. log(1+x)) on result data. Defaults to False.
        return_ntr: Whether to deal with new/total ratio. Defaults to False.

    Returns:
        A tuple [m, v, raw], where `m` is the first momentum, `v` is the second momentum, and `raw` is the normalized
        expression data.
    """

    if return_ntr:
        use_total_layers = True
    if use_total_layers:
        if "total_Size_Factor" not in adata.obs.keys():
            # total_layers = ["uu", "ul", "su", "sl"] if 'uu' in adata.layers.keys() else ['total']
            tot_sfs, _ = sz_util(
                adata,
                "_total_",
                round_exprs=False,
                method="median",
                locfunc=np.nanmean,
                total_layers=total_layers,
            )
        else:
            tot_sfs = adata.obs.total_Size_Factor
        sfs_x, sfs_y = tot_sfs[:, None], tot_sfs[:, None]

    m = [None] * len(layers)
    v = [None] * len(layers)
    raw = [None] * len(layers)
    for i, layer in enumerate(layers):
        if layer in ["X_total", "total", "M_t"]:
            if (layer == "X_total" and adata.uns["pp"]["layers_norm_method"] is None) or layer == "M_t":
                x_layer = adata[:, genes].layers[layer]
                if return_ntr:
                    T_genes = adata[:, genes].layers[get_layer_pair(layer)]
                    T_genes = T_genes.A if issparse(T_genes) else T_genes
                    x_layer = x_layer / (T_genes + 1e-5)
                else:
                    x_layer = x_layer - adata[:, genes].layers[get_layer_pair(layer)]
            else:
                x_layer = adata.layers[layer]
                group_pair_x_layer_ = get_layer_group(get_layer_pair(layer))
                pair_x_layer, group_x_layer, group_pair_x_layer = (
                    adata.layers[get_layer_pair(layer)],
                    adata.layers[get_layer_group(layer)],
                    None if group_pair_x_layer_ is None else adata.layers[group_pair_x_layer_],
                )

                if layer.startswith("X_"):
                    x_layer, pair_x_layer, group_x_layer, group_pair_x_layer = (
                        inverse_norm(adata, x_layer),
                        inverse_norm(adata, pair_x_layer),
                        inverse_norm(adata, group_x_layer),
                        0 if group_pair_x_layer_ is None else inverse_norm(adata, group_pair_x_layer),
                    )

                if layer.startswith("M_"):
                    t_layer_key = "M_t"
                elif layer.startswith("X_"):
                    t_layer_key = "X_total"
                else:
                    t_layer_key = "total"

                if not use_total_layers:
                    sfs_x, _ = sz_util(
                        adata,
                        layer,
                        round_exprs=False,
                        method="median",
                        locfunc=np.nanmean,
                        total_layers=None,
                        CM=x_layer + group_x_layer,
                    )
                    sfs_y, _ = sz_util(
                        adata,
                        get_layer_pair(layer),
                        round_exprs=False,
                        method="median",
                        locfunc=np.nanmean,
                        total_layers=None,
                        CM=pair_x_layer + group_pair_x_layer,
                    )
                    sfs_x, sfs_y = sfs_x[:, None], sfs_y[:, None]

                x_layer = normalize_mat_monocle(
                    x_layer[:, adata.var_names.isin(genes)],
                    sfs_x,
                    relative_expr=True,
                    pseudo_expr=0,
                    norm_method=None,
                )
                y_layer = normalize_mat_monocle(
                    pair_x_layer[:, adata.var_names.isin(genes)],
                    sfs_y,
                    relative_expr=True,
                    pseudo_expr=0,
                    norm_method=None,
                )

                if return_ntr:
                    T_genes = adata[:, genes].layers[t_layer_key]
                    T_genes = T_genes.A if issparse(T_genes) else T_genes
                    x_layer = (x_layer - y_layer) / (T_genes + 1e-5)
                else:
                    x_layer = x_layer - y_layer

        else:
            if (layer == ["X_new"] and adata.uns["pp"]["layers_norm_method"] is None) or layer == "M_n":
                total_layer = "X_total" if layer == ["X_new"] else "M_t"

                if return_ntr:
                    T_genes = adata[:, genes].layers[total_layer]
                    T_genes = T_genes.A if issparse(T_genes) else T_genes
                    x_layer = adata[:, genes].layers[layer] / (T_genes + 1e-5)
                else:
                    x_layer = adata[:, genes].layers[layer]
            else:
                x_layer = adata.layers[layer]
                total_layer = adata.layers["X_total"]
                if layer.startswith("X_"):
                    x_layer = inverse_norm(adata, x_layer)
                    total_layer = inverse_norm(adata, total_layer)

                if not use_total_layers:
                    tot_sfs, _ = sz_util(
                        adata,
                        layer,
                        round_exprs=False,
                        method="median",
                        locfunc=np.nanmean,
                        total_layers=None,
                        CM=x_layer,
                    )
                x_layer = normalize_mat_monocle(
                    x_layer[:, adata.var_names.isin(genes)],
                    szfactors=tot_sfs[:, None],
                    relative_expr=True,
                    pseudo_expr=0,
                    norm_method=None,
                )

                if return_ntr:
                    total_layer = normalize_mat_monocle(
                        total_layer[:, adata.var_names.isin(genes)],
                        szfactors=tot_sfs[:, None],
                        relative_expr=True,
                        pseudo_expr=0,
                        norm_method=None,
                    )
                    total_layer = total_layer.A if issparse(total_layer) else total_layer
                    x_layer /= total_layer + 1e-5
        if log:
            if issparse(x_layer):
                x_layer.data = np.log1p(x_layer.data)
            else:
                x_layer = np.log1p(x_layer)

        m[i], v[i], _ = calc_12_mom_labeling(x_layer.T, time)
        raw[i] = x_layer

    return m, v, raw  # each list element corresponds to a layer


def prepare_data_has_splicing(
    adata: AnnData,
    genes: List[str],
    time: np.ndarray,
    layer_u: str,
    layer_s: str,
    use_total_layers: bool = True,
    total_layers: List[str] = ["X_ul", "X_sl", "X_uu", "X_su"],
    total_layer: str = "X_total",
    return_cov: bool = True,
    return_ntr: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Prepare data when assumption is kinetic and data has splicing.

    Args:
        adata: An AnnData object.
        genes: The genes to be estimated.
        time: The array containing time stamps.
        layer_u: The layer key for unspliced data.
        layer_s: The layer key for spliced data.
        use_total_layers: Whether to use total layers embedded in the AnnData object. Defaults to True.
        total_layers: The layer(s) that can be summed up to get the total mRNA. Defaults to ["X_ul", "X_sl", "X_uu",
            "X_su"].
        total_layer: The layer key for the precalculated total mRNA data. Defaults to "X_total".
        return_cov: Whether to calculate the covariance between spliced and unspliced data. Defaults to True.
        return_ntr: Whether to return the new to total ratio or original expression data. Defaults to False.

    Returns:
        A tuple [res, raw] where `res` is the calculated momentum data and `raw` is the normalized expression data.
    """

    res = [0] * len(genes)
    raw = [0] * len(genes)

    U, S = (
        adata[:, genes].layers[layer_u] if layer_u == "M_ul" else None,
        adata[:, genes].layers[layer_s] if layer_s == "M_sl" else None,
    )
    T = adata[:, genes].layers[total_layer] if total_layer == "M_t" else None

    layer_ul_data, layer_sl_data = adata.layers[layer_u], adata.layers[layer_s]
    layer_uu_data, layer_su_data = (
        adata.layers[total_layers[2]],
        adata.layers[total_layers[3]],
    )
    layer_ul_data, layer_sl_data = (
        layer_ul_data if layer_u == "M_ul" else inverse_norm(adata, layer_ul_data),
        layer_sl_data if layer_s == "M_sl" else inverse_norm(adata, layer_sl_data),
    )
    layer_uu_data, layer_su_data = (
        layer_uu_data if total_layers[2] == "M_uu" else inverse_norm(adata, layer_uu_data),
        layer_su_data if total_layers[3] == "M_su" else inverse_norm(adata, layer_su_data),
    )

    total_layer_data = adata.layers[total_layer]
    total_layer_data = total_layer_data if total_layer == "M_t" else inverse_norm(adata, total_layer_data)

    if use_total_layers:
        if "total_Size_Factor" not in adata.obs.keys():
            tot_sfs, _ = sz_util(
                adata,
                "_total_",
                round_exprs=False,
                method="median",
                locfunc=np.nanmean,
                total_layers=total_layers,
                CM=layer_ul_data + layer_sl_data + layer_uu_data + layer_su_data,
            )
            sfs_u, sfs_s = tot_sfs[:, None], tot_sfs[:, None]
        else:
            tot_sfs = adata.obs.total_Size_Factor
            sfs_u, sfs_s = tot_sfs[:, None], tot_sfs[:, None]
    else:
        sfs_u, _ = sz_util(
            adata,
            layer_u,
            round_exprs=False,
            method="median",
            locfunc=np.nanmean,
            total_layers=None,
            CM=layer_ul_data + layer_uu_data,
        )
        sfs_s, _ = sz_util(
            adata,
            layer_s,
            round_exprs=False,
            method="median",
            locfunc=np.nanmean,
            total_layers=None,
            CM=layer_sl_data + layer_su_data,
        )
        sfs_u, sfs_s = sfs_u[:, None], sfs_s[:, None]

    if U is None:
        U = normalize_mat_monocle(
            layer_ul_data[:, adata.var_names.isin(genes)],
            sfs_u,
            relative_expr=True,
            pseudo_expr=0,
            norm_method=None,
        )
    if S is None:
        S = normalize_mat_monocle(
            layer_sl_data[:, adata.var_names.isin(genes)],
            sfs_s,
            relative_expr=True,
            pseudo_expr=0,
            norm_method=None,
        )

    if "total_Size_Factor" not in adata.obs.keys():
        tot_sfs, _ = sz_util(
            adata,
            "_total_",
            round_exprs=False,
            method="median",
            locfunc=np.nanmean,
            total_layers=total_layer,
            CM=total_layer_data,
        )
    else:
        tot_sfs = adata.obs.total_Size_Factor
        tot_sfs = tot_sfs[:, None]

    if T is None:
        T = normalize_mat_monocle(
            total_layer_data[:, adata.var_names.isin(genes)],
            tot_sfs,
            relative_expr=True,
            pseudo_expr=0,
            norm_method=None,
        )

    for i, g in enumerate(genes):
        if return_ntr:
            T_i = T[:, i].A if issparse(T[:, i]) else T[:, i]
            u = U[:, i] / (T_i + 1e-5)
            s = S[:, i] / (T_i + 1e-5)
        else:
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


def prepare_data_no_splicing(
    adata: AnnData,
    genes: List[str],
    time: np.ndarray,
    layer: str,
    use_total_layers: bool = True,
    total_layer: str = "X_total",
    return_old: bool = False,
    return_ntr: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Prepare the data when assumption is kinetic and data has no splicing.

    Args:
        adata: An AnnData object.
        genes: The genes to be estimated.
        time: The array containing time stamps.
        layer: The layer containing the expression data.
        use_total_layers: Whether to use the total data embedded in the AnnData object. Defaults to True.
        total_layer: The layer key for the precalculated total mRNA data. Defaults to "X_total".
        return_old: Whether to return the old expression data together or the newly expressed gene data only. Defaults
            to False.
        return_ntr: Whether to return the new to total ratio or the original expression data. Defaults to False.

    Returns:
        A tuple [res, raw] where `res` is the calculated momentum data and `raw` is the normalized expression data.
    """

    from ..preprocessing.normalization import normalize_mat_monocle, sz_util

    res = [0] * len(genes)
    raw = [0] * len(genes)

    U, T = (
        adata[:, genes].layers[layer] if layer == "M_n" else None,
        adata[:, genes].layers[total_layer] if total_layer == "M_t" else None,
    )

    layer_data = adata.layers[layer]
    total_layer_data = adata.layers[total_layer]

    layer_data, total_layer_data = (
        layer_data if layer == "M_n" else inverse_norm(adata, layer_data),
        total_layer_data if total_layer == "M_t" else inverse_norm(adata, total_layer_data),
    )

    if use_total_layers:
        if "total_Size_Factor" not in adata.obs.keys():
            sfs, _ = sz_util(
                adata,
                "_total_",
                round_exprs=False,
                method="median",
                locfunc=np.nanmean,
                total_layers=total_layer,
                CM=total_layer_data,
            )
        else:
            sfs = adata.obs.total_Size_Factor
        sfs, tot_sfs = sfs[:, None], sfs[:, None]
    else:
        sfs, _ = sz_util(
            adata,
            layer,
            round_exprs=False,
            method="median",
            locfunc=np.nanmean,
            total_layers=None,
            CM=layer_data,
        )
        tot_sfs, _ = sz_util(
            adata,
            layer,
            round_exprs=False,
            method="median",
            locfunc=np.nanmean,
            total_layers=None,
            CM=total_layer_data,
        )
        sfs, tot_sfs = sfs[:, None], tot_sfs[:, None]

    if U is None:
        U = normalize_mat_monocle(
            layer_data[:, adata.var_names.isin(genes)],
            sfs,
            relative_expr=True,
            pseudo_expr=0,
            norm_method=None,
        )
    if T is None:
        T = normalize_mat_monocle(
            total_layer_data[:, adata.var_names.isin(genes)],
            tot_sfs,
            relative_expr=True,
            pseudo_expr=0,
            norm_method=None,
        )

    for i, g in enumerate(genes):
        if return_ntr:
            T_i = T[:, i].A if issparse(T[:, i]) else T[:, i]
            u, t = U[:, i] / (T_i + 1e-5), T[:, i]
        else:
            u, t = U[:, i], T[:, i]
        ut = strat_mom(u, time, np.mean)
        uut = strat_mom(elem_prod(u, u), time, np.mean)
        res[i] = np.vstack([ut, uut])
        raw[i] = np.vstack([u, t - u]) if return_old else u

    return res, raw


def prepare_data_mix_has_splicing(
    adata: AnnData,
    genes: List[str],
    time: np.ndarray,
    layer_u: str = "X_uu",
    layer_s: str = "X_su",
    layer_ul: str = "X_ul",
    layer_sl: str = "X_sl",
    use_total_layers: bool = True,
    total_layers: List[str] = ["X_ul", "X_sl", "X_uu", "X_su"],
    mix_model_indices: Optional[List[int]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Prepare data for mixture modeling when assumption is kinetic and data has splicing.

    Note that the mix_model_indices is indexed on 10 total species, which can be used to specify the data required for
    different mixture models.

    Args:
        adata: An AnnData object.
        genes: The genes to be estimated.
        time: The array containing time stamps.
        layer_u: The layer key for unspliced mRNA count data. Defaults to "X_uu".
        layer_s: The layer key for spliced mRNA count data. Defaults to "X_su".
        layer_ul: The layer key for unspliced, labeled mRNA count data. Defaults to "X_ul".
        layer_sl: The layer key for spliced, labeled mRNA count data. Defaults to "X_sl".
        use_total_layers: Whether to use total layers embedded in the AnnData object. Defaults to True.
        total_layers: The layer(s) that can be summed up to get the total mRNA. Defaults to ["X_ul", "X_sl", "X_uu",
            "X_su"].
        mix_model_indices: The indices for data required by the mixture model. If None, all data would be returned.
            Defaults to None.

    Returns:
        A tuple [res, raw] where `res` is the calculated momentum data and `raw` is the normalized expression data.
    """

    from ..preprocessing.normalization import normalize_mat_monocle, sz_util

    res = [0] * len(genes)
    raw = [0] * len(genes)

    U, S = (
        adata[:, genes].layers[layer_u] if layer_u == "M_uu" else None,
        adata[:, genes].layers[layer_s] if layer_u == "M_su" else None,
    )
    Ul, Sl = (
        adata[:, genes].layers[layer_ul] if layer_u == "M_ul" else None,
        adata[:, genes].layers[layer_sl] if layer_u == "M_sl" else None,
    )

    layer_u_data, layer_s_data = adata.layers[layer_u], adata.layers[layer_s]
    layer_ul_data, layer_sl_data = (
        adata.layers[layer_ul],
        adata.layers[layer_sl],
    )
    layer_u_data, layer_s_data = (
        layer_u_data if layer_u == "M_uu" else inverse_norm(adata, layer_u_data),
        layer_s_data if layer_s == "M_su" else inverse_norm(adata, layer_s_data),
    )
    layer_ul_data, layer_sl_data = (
        layer_ul_data if layer_ul == "M_ul" else inverse_norm(adata, layer_ul_data),
        layer_sl_data if layer_sl == "M_sl" else inverse_norm(adata, layer_sl_data),
    )

    if use_total_layers:
        if "total_Size_Factor" not in adata.obs.keys():
            sfs, _ = sz_util(
                adata,
                "_total_",
                False,
                "median",
                np.nanmean,
                total_layers=total_layers,
                CM=layer_u_data + layer_s_data + layer_ul_data + layer_sl_data,
            )
            sfs_u, sfs_s = sfs[:, None], sfs[:, None]
        else:
            sfs = adata.obs.total_Size_Factor
            sfs_u, sfs_s = sfs[:, None], sfs[:, None]
    else:
        sfs_u, _ = sz_util(
            adata,
            layer_u,
            False,
            "median",
            np.nanmean,
            total_layers=None,
            CM=layer_u_data + layer_ul_data,
        )
        sfs_s, _ = sz_util(
            adata,
            layer_s,
            False,
            "median",
            np.nanmean,
            total_layers=None,
            CM=layer_s_data + layer_sl_data,
        )
        sfs_u, sfs_s = sfs_u[:, None], sfs_s[:, None]

    if U is None:
        U = normalize_mat_monocle(
            layer_u_data[:, adata.var_names.isin(genes)],
            sfs_u,
            relative_expr=True,
            pseudo_expr=0,
            norm_method=None,
        )
    if S is None:
        S = normalize_mat_monocle(
            layer_s_data[:, adata.var_names.isin(genes)],
            sfs_s,
            relative_expr=True,
            pseudo_expr=0,
            norm_method=None,
        )
    if Ul is None:
        Ul = normalize_mat_monocle(
            layer_ul_data[:, adata.var_names.isin(genes)],
            sfs_u,
            relative_expr=True,
            pseudo_expr=0,
            norm_method=None,
        )
    if Sl is None:
        Sl = normalize_mat_monocle(
            layer_sl_data[:, adata.var_names.isin(genes)],
            sfs_s,
            relative_expr=True,
            pseudo_expr=0,
            norm_method=None,
        )

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


def prepare_data_mix_no_splicing(
    adata: AnnData,
    genes: List[str],
    time: np.ndarray,
    layer_n: str,
    layer_t: str,
    use_total_layers: bool = True,
    total_layer: bool = "X_total",
    mix_model_indices: Optional[List[int]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Prepare data for mixture modeling when assumption is kinetic and data has NO splicing.

    Note that the mix_model_indices is indexed on 4 total species, which can be used to specify
    the data required for different mixture models.

    Args:
        adata: An AnnData object.
        genes: The genes to be estimated.
        time: The array containing time stamps.
        layer_n: The layer key for new mRNA count.
        layer_t: The layer key for total mRNA count.
        use_total_layers: Whether to use total layers embedded in the AnnData object. Defaults to True.
        total_layer: The layer key for the precalculated total mRNA data. Defaults to "X_total".
        mix_model_indices: The indices for data required by the mixture model. If None, all data would be returned.
            Defaults to None.

    Returns:
        A tuple [res, raw] where `res` is the calculated momentum data and `raw` is the normalized expression data.
    """

    from ..preprocessing.normalization import normalize_mat_monocle, sz_util

    res = [0] * len(genes)
    raw = [0] * len(genes)

    N, T = (
        adata[:, genes].layers[layer_n] if layer_n == "M_n" else None,
        adata[:, genes].layers[layer_t] if layer_t == "M_t" else None,
    )

    layer_n_data = adata.layers[layer_n]
    layer_t_data = adata.layers[layer_t]
    layer_n_data, layer_t_data = (
        layer_n_data if layer_n == "M_n" else inverse_norm(adata, layer_n_data),
        layer_t_data if layer_t == "M_t" else inverse_norm(adata, layer_t_data),
    )

    if use_total_layers:
        if "total_Size_Factor" not in adata.obs.keys():
            sfs, _ = sz_util(
                adata,
                total_layer,
                False,
                "median",
                np.nanmean,
                total_layers="total",
                CM=layer_t_data,
            )
            sfs_n, sfs_t = sfs[:, None], sfs[:, None]
        else:
            sfs = adata.obs.total_Size_Factor
            sfs_n, sfs_t = sfs[:, None], sfs[:, None]
    else:
        sfs_n, _ = sz_util(
            adata,
            layer_n,
            False,
            "median",
            np.nanmean,
            total_layers=None,
            CM=layer_n_data,
        )
        sfs_t, _ = sz_util(
            adata,
            layer_t,
            False,
            "median",
            np.nanmean,
            total_layers=None,
            CM=layer_t_data,
        )
        sfs_n, sfs_t = sfs_n[:, None], sfs_t[:, None]

    if N is None:
        N = normalize_mat_monocle(
            layer_n_data[:, adata.var_names.isin(genes)],
            sfs_n,
            relative_expr=True,
            pseudo_expr=0,
            norm_method=None,
        )
    if T is None:
        T = normalize_mat_monocle(
            layer_t_data[:, adata.var_names.isin(genes)],
            sfs_t,
            relative_expr=True,
            pseudo_expr=0,
            norm_method=None,
        )

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
def calc_1nd_moment(
    X: np.ndarray, W: np.ndarray, normalize_W: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Calculate first moment for the layers.

    Args:
        X: The layer to calculate the moment.
        W: The connectivity graph that will be used for moment calculations.
        normalize_W: Whether to normalize W before calculation. Defaults to True.

    Returns:
        The first moment of the layer.
    """
    if normalize_W:
        if type(W) == np.ndarray:
            d = np.sum(W, 1).flatten()
        else:
            d = np.sum(W, 1).A.flatten()
        W = diags(1 / d) @ W if issparse(W) else np.diag(1 / d) @ W
        return W @ X, W
    else:
        return W @ X


def calc_2nd_moment(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    normalize_W: bool = True,
    center: bool = False,
    mX: np.ndarray = None,
    mY: np.ndarray = None,
) -> np.ndarray:
    """Calculate the 2nd moment for the layers.

    Args:
        X: The first layer to be used.
        Y: The second layer to be used.
        W: The connectivity graph that will be used for moment calculations.
        normalize_W: Whether to normalize W before calculation. Defaults to True.
        center: Whether to correct the center. Defaults to False.
        mX: The moment matrix to correct the center. Defaults to None.
        mY: The moment matrix to correct the center. Defaults to None.

    Returns:
        The second moment of the layers.
    """
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


def calc_12_mom_labeling(
    data: np.ndarray, t: np.ndarray, calculate_2_mom: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Calculate 1st and 2nd momentum for given data.

    Args:
        data: The normalized mRNA expression data.
        t: The time stamp array.
        calculate_2_mom: Whether to calculate 2nd momentum. Defaults to True.

    Returns:
        A tuple (m, [v], t_uniq) where `m` is the first momentum, `v` is the second momentum which would be returned
        only if `calculate_2_mom` is true, and `t_uniq` is the unique time stamps.
    """

    t_uniq = np.unique(t)

    m = np.zeros((data.shape[0], len(t_uniq)))
    if calculate_2_mom:
        v = np.zeros((data.shape[0], len(t_uniq)))

    for i in range(data.shape[0]):
        data_ = (
            np.array(data[i].A.flatten(), dtype=float) if issparse(data) else np.array(data[i], dtype=float)
        )  # consider using the `adata.obs_vector`, `adata.var_vector` methods or accessing the array directly.
        m[i] = strat_mom(data_, t, np.nanmean)
        if calculate_2_mom:
            v[i] = strat_mom(data_, t, np.nanvar)

    return (m, v, t_uniq) if calculate_2_mom else (m, t_uniq)


def calc_mom_all_genes(
    T: np.ndarray, adata: AnnData, fcn_mom: Callable
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate momentum for all genes in an AnnData object.

    Args:
        T: The time stamp array.
        adata: An AnnData object.
        fcn_mom: The function used to calculate momentum.

    Returns:
        A tuple (Mn, Mo, Mt, Mr), where `Mn` is momentum calculated from labeled (new) mRNA count, `Mo` is from
        unlabeled (old) mRNA count, `Mt` is from total mRNA count, and `Mr` is from new to total ratio.
    """
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


def strat_mom(arr: Union[np.ndarray, csr_matrix], strata: np.ndarray, fcn_mom: Callable) -> np.ndarray:
    """Stratify the mRNA expression data and calculate its momentum.

    Args:
        arr: The mRNA expression data.
        strata: The time stamp array used to stratify `arr`.
        fcn_mom: The function used to calculate the momentum.

    Returns:
        The momentum for each stratum.
    """

    arr = arr.A if issparse(arr) else arr
    x = stratify(arr, strata)
    return np.array([fcn_mom(y) for y in x])


def stratify(arr: np.ndarray, strata: np.ndarray) -> List[np.ndarray]:
    """Stratify the given array with the given reference strata.

    Args:
        arr: The array to be stratified.
        strata: The reference strata vector.

    Returns:
        A list containing the strata from the array, with each element of the list to be the components with line index
        corresponding to the reference strata vector's unique elements' index.
    """

    s = np.unique(strata)
    return [arr[strata == s[i]] for i in range(len(s))]


def gaussian_kernel(
    X: np.ndarray, nbr_idx: np.ndarray, sigma: int, k: Optional[int] = None, dists: Optional[np.ndarray] = None
) -> csr_matrix:
    """Normalize connectivity map with Gaussian kernel.

    Args:
        X: The mRNA expression data.
        nbr_idx: The indices of nearest neighbors of each cell.
        sigma: The standard deviation for gaussian model.
        k: The number of nearest neighbors to be considered. Defaults to None.
        dists: The distances to the n_neighbors' closest points in knn graph. Defaults to None.

    Returns:
        The normalized connectivity map.
    """
    n = X.shape[0]
    if dists is None:
        dists = []
        for i in range(n):
            d = X[nbr_idx[i][:k]] - X[i]
            dists.append(np.sum(elem_prod(d, d), 1).flatten())
    W = lil_matrix((n, n))
    s2_inv = 1 / (2 * sigma**2)
    for i in range(n):
        W[i, nbr_idx[i][:k]] = np.exp(-s2_inv * dists[i][:k] ** 2)

    return csr_matrix(W)
