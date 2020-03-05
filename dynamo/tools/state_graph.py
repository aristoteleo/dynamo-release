import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from .fate import _fate
from .utils import fetch_states

def remove_redundant_points_trajectory(X, tol=1e-4, output_discard=False):
    """"""
    X = np.atleast_2d(X)
    discard = np.zeros(len(X), dtype=bool)
    if X.shape[0] > 1:
        for i in range(len(X)-1):
            dist = np.linalg.norm(X[i+1] - X[i])
            if dist < tol:
                discard[i+1] = True
        X = X[~discard]
    if output_discard:
        return X, discard
    else:
        return X

def arclength_sampling(X, step_length, t=None):
    """uniformly sample data points on an arc curve that generated from vector field predictions."""
    Y = []
    x0 = X[0]
    T = [] if t is not None else None
    t0 = t[0] if t is not None else None
    i = 1
    terminate = False
    arclength = 0
    while(i < len(X) - 1 and not terminate):
        l = 0
        for j in range(i, len(X)-1):
            tangent = X[j] - x0 if j==i else X[j] - X[j-1]
            d = np.linalg.norm(tangent)
            if l + d >= step_length:
                x = x0 if j==i else X[j-1]
                y = x + (step_length-l) * tangent/d
                if t is not None:
                    tau = t0 if j==i else t[j-1]
                    tau += (step_length-l)/d * (t[j] - tau)
                    T.append(tau)
                    t0 = tau
                Y.append(y)
                x0 = y
                i = j
                break
            else:
                l += d
        arclength += step_length
        if l + d < step_length:
            terminate = True
    if T is not None:
        return np.array(Y), arclength, T
    else:
        return np.array(Y), arclength


def classify_clone_cell_type(adata, clone, clone_column, cell_type_column, cell_type_to_excluded):
    """find the dominant cell type of all the cells that are from the same clone"""
    cell_ids = np.where(adata.obs[clone_column] == clone)[0]

    to_check = adata[cell_ids].obs[cell_type_column].value_counts().index.isin(list(cell_type_to_excluded))

    cell_type = np.where(to_check)[0]

    return cell_type

def state_graph(adata, group, basis='umap', layer=None, sample_num=100):
    """Estimate the transition probability between cell types using method of vector field integrations.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that will be used to calculate a cell type (group) transition graph.
        group: `str`
            The attribute to group cells (column names in the adata.obs).
        basis: `str` or None (default: `umap`)
            The embedding data to use for predicting cell fate. If `basis` is either `umap` or `pca`, the reconstructed trajectory
            will be projected back to high dimensional space via the `inverse_transform` function.
        layer: `str` or None (default: `None`)
            Which layer of the data will be used for predicting cell fate with the reconstructed vector field function.
            The layer once provided, will override the `basis` argument and then predicting cell fate in high dimensional space.
        sample_num: `int` (default: 100)
            The number of cells to sample in each group that will be used for calculating the transitoin graph between cell
            groups. This is required for facilitating the calculation.

    Returns
    -------
        An updated adata object that is added with the `group + '_graph'` key, including the transition graph
        and the average transition time.
    """

    groups, uniq_grp = adata.obs[group], adata.obs[group].unique().to_list()
    grp_graph = np.zeros((len(uniq_grp), len(uniq_grp)))
    grp_avg_time = np.zeros((len(uniq_grp), len(uniq_grp)))

    all_X, VecFld, t_end, valid_genes = fetch_states(adata, init_states=None, init_cells=adata.obs_names, basis=basis,
                                                           layer=layer, average=False, t_end=None)
    kdt = cKDTree(all_X, leaf_size=30, metric='euclidean')

    for i, cur_grp in enumerate(tqdm(uniq_grp, desc='iterate groups:')):
        init_cells = adata.obs_names[groups == cur_grp]
        if sample_num is not None:
            cell_num = np.min((sample_num, len(init_cells)))
            ind = np.random.choice(len(init_cells), cell_num, replace=False)
            init_cells = init_cells[ind]

        init_states, _, _, _ = fetch_states(adata, init_states=None, init_cells=init_cells, basis=basis,
                                                               layer=layer, average=False, t_end=None)
        t, X = _fate(VecFld, init_states, VecFld_true=None, t_end=t_end, step_size=None, direction='forward',
                              interpolation_num=250, average=False)

        len_per_cell = len(t)
        cell_num = int(X.shape[0] / len(t))

        for j in np.arange(cell_num):
            cur_ind = np.arange(j * len_per_cell, (j + 1) * len_per_cell)
            # Y, arclength, T = arclength_sampling(X[cur_ind], 0.1, t=t[cur_ind])
            Y = X[cur_ind]

            knn_ind, knn_dist = kdt.query(Y, k=1, return_distance=True)

            for k, cur_knn_dist in enumerate(knn_dist):
                if cur_knn_dist < 1e-3:
                    cell_id = knn_ind[k]
                    ind_other_cell_type = uniq_grp.index(groups[cell_id])
                    grp_graph[i, ind_other_cell_type] += 1
                    grp_avg_time[i, ind_other_cell_type] += t[k]

        grp_avg_time[i, :] /= grp_graph[i, :]
        grp_graph[i, :] /= cell_num

    adata.uns[group + '_graph'] = {"group_graph": grp_graph, "group_avg_time": grp_avg_time}

    return adata
