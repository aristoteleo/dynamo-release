import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from .fate import _fate
from .utils import fetch_states, remove_redundant_points_trajectory, arclength_sampling, integrate_streamline

def classify_clone_cell_type(adata, clone, clone_column, cell_type_column, cell_type_to_excluded):
    """find the dominant cell type of all the cells that are from the same clone"""
    cell_ids = np.where(adata.obs[clone_column] == clone)[0]

    to_check = adata[cell_ids].obs[cell_type_column].value_counts().index.isin(list(cell_type_to_excluded))

    cell_type = np.where(to_check)[0]

    return cell_type

def state_graph(adata, group, approx=True, basis='umap', layer=None, arc_sample=False, sample_num=100):
    """Estimate the transition probability between cell types using method of vector field integrations.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that will be used to calculate a cell type (group) transition graph.
        group: `str`
            The attribute to group cells (column names in the adata.obs).
        approx: `bool` (default: False)
            Whether to use streamplot to get the integration lines from each cell.
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
    kdt = cKDTree(all_X, leafsize=30)

    for i, cur_grp in enumerate(tqdm(uniq_grp, desc='iterate groups:')):
        init_cells = adata.obs_names[groups == cur_grp]
        if sample_num is not None:
            cell_num = np.min((sample_num, len(init_cells)))
            ind = np.random.choice(len(init_cells), cell_num, replace=False)
            init_cells = init_cells[ind]

        init_states, _, _, _ = fetch_states(adata, init_states=None, init_cells=init_cells, basis=basis,
                                                               layer=layer, average=False, t_end=None)
        if approx:
            X_grid, V_grid =  adata.uns['VecFld_' + basis]["VecFld"]['grid'], adata.uns['VecFld_' + basis]["VecFld"]['grid_V']
            N = int(np.sqrt(V_grid.shape[0]))
            X_grid, V_grid = np.array([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])]), \
                             np.array([V_grid[:, 0].reshape((N, N)), V_grid[:, 1].reshape((N, N))])

            t, X = integrate_streamline(X_grid[0], X_grid[1], V_grid[0], V_grid[1], integration_direction='forward',
                                        init_states=init_states, interpolation_num=250, average=False)

        else:
            t, X = _fate(VecFld, init_states, VecFld_true=None, t_end=t_end, step_size=None, direction='forward',
                         interpolation_num=250, average=False)

        len_per_cell = len(t)
        cell_num = int(X.shape[0] / len(t))

        knn_dist_, knn_ind_ = kdt.query(init_states, k=2)
        dist_min, dist_threshold = np.max([knn_dist_[:, 1].min(), 1e-3]), np.mean(knn_dist_[:, 1])

        for j in np.arange(cell_num):
            cur_ind = np.arange(j * len_per_cell, (j + 1) * len_per_cell)
            Y, arclength, T_bool = remove_redundant_points_trajectory(X[cur_ind], tol=dist_min, output_discard=True)

            if arc_sample:
                Y, arclength, T = arclength_sampling(Y, arclength / 1000, t=t[~T_bool])
            else:
                T = t[~T_bool]

            knn_dist, knn_ind = kdt.query(Y, k=1)

            avg_time = [0] * len(uniq_grp)
            _graph = [0] * len(uniq_grp)

            for k, cur_knn_dist in enumerate(knn_dist):
                if cur_knn_dist < dist_threshold:
                    cell_id = knn_ind[k]
                    ind_other_cell_type = uniq_grp.index(groups[cell_id])
                    _graph[ind_other_cell_type] += 1
                    avg_time[ind_other_cell_type] += T[k]

            grp_avg_time[i, :] += (np.array(avg_time) / np.array(_graph)).flatten()
            grp_graph[i, :] += (np.array(_graph) > 0).astype('float').flatten()

        grp_avg_time[i, :] /= grp_graph[i, :]
        grp_graph[i, :] /= cell_num

    adata.uns[group + '_graph'] = {"group_graph": grp_graph, "group_avg_time": grp_avg_time}

    return adata
