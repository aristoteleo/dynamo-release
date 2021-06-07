import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp

# from sklearn.preprocessing import OrdinalEncoder

from ..tools.Markov import DiscreteTimeMarkovChain
from ..prediction.fate import _fate
from ..vectorfield import vector_field_function
from ..tools.utils import fetch_states
from .utils import (
    remove_redundant_points_trajectory,
    arclength_sampling,
    integrate_streamline,
)
from ..dynamo_logger import LoggerManager


def classify_clone_cell_type(adata, clone, clone_column, cell_type_column, cell_type_to_excluded):
    """find the dominant cell type of all the cells that are from the same clone"""
    cell_ids = np.where(adata.obs[clone_column] == clone)[0]

    to_check = adata[cell_ids].obs[cell_type_column].value_counts().index.isin(list(cell_type_to_excluded))

    cell_type = np.where(to_check)[0]

    return cell_type


def state_graph(
    adata,
    group,
    method="vf",
    transition_mat_key="pearson_transition_matrix",
    approx=False,
    eignum=5,
    basis="umap",
    layer=None,
    arc_sample=False,
    sample_num=100,
):
    """Estimate the transition probability between cell types using method of vector field integrations or Markov chain
    lumping.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that will be used to calculate a cell type (group) transition graph.
        group: `str`
            The attribute to group cells (column names in the adata.obs).
        method: `str` (default: 'vf')
            The method that will be used to construct lumped cell state graph. Must be one of {`vf` or `markov`}
        transition_mat_key: `str` (default: 'pearson_transition_matrix')
            The key that corresponds to the transition graph used in the KernelMarkovChain class for lumping.
        approx: `bool` (default: False)
            Whether to use streamplot to get the integration lines from each cell.
        eignum: `int` (default: 5)
            The number of eigen-vectors when performing the eigen-decomposition to obtain the stationary
            distribution. 5 should be sufficient as the stationary distribution will be the first eigenvector. This also
            accelerates the calculation.
        basis: `str` or None (default: `umap`)
            The embedding data to use for predicting cell fate. If `basis` is either `umap` or `pca`, the reconstructed
            trajectory will be projected back to high dimensional space via the `inverse_transform` function.
        layer: `str` or None (default: `None`)
            Which layer of the data will be used for predicting cell fate with the reconstructed vector field function.
            The layer once provided, will override the `basis` argument and then predicting cell fate in high
            dimensional space.
        sample_num: `int` (default: 100)
            The number of cells to sample in each group that will be used for calculating the transitoin graph between
            cell groups. This is required for facilitating the calculation.

    Returns
    -------
        An updated adata object that is added with the `group + '_graph'` key, including the transition graph
        and the average transition time.
    """
    logger = LoggerManager.get_main_logger()
    timer_logger = LoggerManager.get_temp_timer_logger()
    timer_logger.log_time()

    logger.info("Estimating the transition probability between cell types...")
    groups, uniq_grp = adata.obs[group], list(adata.obs[group].unique())

    if method.lower() in ["naive", "markov"]:
        logger.info("Applying kernel Markov chain")
        T = adata.obsp[transition_mat_key]
        if np.isclose(T.sum(1), 1).sum() > np.isclose(T.sum(0), 1).sum():
            logger.info("KernelMarkovChain assuming column sum to be 1. Transposing transition matrix")
            T = T.T
        if sp.issparse(T):
            T = T.A
        dtmc = DiscreteTimeMarkovChain(P=T, eignum=eignum, check_norm=False)

        # ord_enc = OrdinalEncoder()
        # labels = ord_enc.fit_transform(adata.obs[[group]])
        # labels = labels.flatten().astype(int)
        labels = np.zeros(len(groups), dtype=int)
        for i, grp in enumerate(uniq_grp):
            labels[groups == grp] = i

        grp_graph = dtmc.lump(labels).T if method == "markov" else dtmc.naive_lump(T.A, labels).T
        label_len, grp_avg_time = len(np.unique(labels)), None
        grp_graph = grp_graph[:label_len, :label_len]

    elif method == "vf":
        logger.info("Applying vector field")
        grp_graph = np.zeros((len(uniq_grp), len(uniq_grp)))
        grp_avg_time = np.zeros((len(uniq_grp), len(uniq_grp)))

        all_X, VecFld, t_end, _ = fetch_states(
            adata,
            init_states=None,
            init_cells=adata.obs_names,
            basis=basis,
            layer=layer,
            average=False,
            t_end=None,
        )
        logger.report_progress(percent=0, progress_name="KDTree parameter preparation computation")
        logger.log_time()
        kdt = cKDTree(all_X, leafsize=30)
        logger.finish_progress(progress_name="KDTree computation")
        vf_dict = adata.uns["VecFld_" + basis]

        for i, cur_grp in enumerate(LoggerManager.progress_logger(uniq_grp, progress_name="iterate groups")):
            init_cells = adata.obs_names[groups == cur_grp]
            if sample_num is not None:
                cell_num = np.min((sample_num, len(init_cells)))
                ind = np.random.choice(len(init_cells), cell_num, replace=False)
                init_cells = init_cells[ind]

            init_states, _, _, _ = fetch_states(
                adata,
                init_states=None,
                init_cells=init_cells,
                basis=basis,
                layer=layer,
                average=False,
                t_end=None,
            )
            if approx and basis != "pca" and layer is None:
                X_grid, V_grid = (
                    vf_dict["grid"],
                    vf_dict["grid_V"],
                )
                N = int(np.sqrt(V_grid.shape[0]))
                X_grid, V_grid = (
                    np.array([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])]),
                    np.array(
                        [
                            V_grid[:, 0].reshape((N, N)),
                            V_grid[:, 1].reshape((N, N)),
                        ]
                    ),
                )

                t, X = integrate_streamline(
                    X_grid[0],
                    X_grid[1],
                    V_grid[0],
                    V_grid[1],
                    integration_direction="forward",
                    init_states=init_states,
                    interpolation_num=250,
                    average=False,
                )
            else:
                t, X = _fate(
                    lambda x: vector_field_function(x=x, vf_dict=vf_dict),
                    init_states,
                    t_end=t_end,
                    step_size=None,
                    direction="forward",
                    interpolation_num=250,
                    average=False,
                )
                # t, X = np.hstack(t), np.hstack(X).T

            len_per_cell = None if type(t) == list else len(t)
            cell_num = len(t) if type(X) == list else int(X.shape[0] / len(t))

            knn_dist_, knn_ind_ = kdt.query(init_states, k=2)
            dist_min, dist_threshold = (
                np.max([knn_dist_[:, 1].min(), 1e-3]),
                np.mean(knn_dist_[:, 1]),
            )

            for j in np.arange(cell_num):
                if len_per_cell is not None:
                    cur_ind = np.arange(j * len_per_cell, (j + 1) * len_per_cell)
                    Y, arclength, T_bool = remove_redundant_points_trajectory(
                        X[cur_ind], tol=dist_min, output_discard=True
                    )

                    if arc_sample:
                        Y, arclength, T = arclength_sampling(Y, arclength / 1000, t=t[~T_bool])
                    else:
                        T = t[~T_bool]
                else:
                    Y, T = X[j].T, t[j] if type(t[j]) == np.ndarray else np.array(t[j])

                knn_dist, knn_ind = kdt.query(Y, k=1)

                # set up a dataframe with group and time
                pass_t = np.where(knn_dist < dist_threshold)[0]
                pass_df = pd.DataFrame({"group": adata[knn_ind[pass_t]].obs[group], "t": T[pass_t]})
                # only consider trajectory that pass at least 10 cells in group as confident pass
                pass_group_counter = pass_df.group.value_counts()
                pass_groups, confident_pass_check = (
                    pass_group_counter.index.tolist(),
                    np.where(pass_group_counter > 10)[0],
                )
                # assign the transition matrix and average transition time
                if len(confident_pass_check) > 0:
                    ind_other_cell_type = [uniq_grp.index(k) for k in np.array(pass_groups)[confident_pass_check]]
                    grp_graph[i, ind_other_cell_type] += 1
                    grp_avg_time[i, ind_other_cell_type] += (
                        pass_df.groupby("group")["t"].mean()[confident_pass_check].values
                    )

            # average across cells
            grp_avg_time[i, :] /= grp_graph[i, :]
            grp_graph[i, :] /= cell_num

    else:
        raise NotImplementedError("Only vector field (vf) or Markov chain (markov) based lumping are supported.")

    adata.uns[group + "_graph"] = {"group_graph": grp_graph, "group_avg_time": grp_avg_time, "group_names": uniq_grp}
    timer_logger.finish_progress(progress_name="State graph estimation")
    return adata
